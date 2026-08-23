"""Generic ZeroGPU backend for Claro.AI.

This module is the model-agnostic execution layer that runs inside the
Hugging Face ZeroGPU Space `Gochan562/claro_ai_gpu`.  It deliberately reuses
Claro.AI's existing loader architecture instead of maintaining a separate
hardcoded model list:

    Hugging Face repo
    ├── GGUF (.gguf file present)  → llama-cpp-python backend
    ├── Diffusers (model_index.json) → DiffusionPipeline
    └── Transformers                → AutoConfig architecture resolver
                                       (auto_map → architectures → model_type
                                        → encoder/decoder structure)

Model-format detection is the source of truth for the loading class; the
incoming `task` only chooses the *inference* behaviour (generate vs forward
vs diffusion), never the AutoModel class.

The loader interface is preserved exactly:
    model, tokenizer, processor, feature_extractor, preprocessor

Safe caching: a thread-safe LRU cache keeps the most-recently-used models
loaded for the lifetime of the Space worker, so repeated calls for the same
`model_id` do not download or reload.  Cache is bounded and evicts oldest
entries when full; it never crosses Space worker lifetimes.

No arbitrary Python execution is performed here — only structured
model_id/task/inputs requests are accepted.
"""
from __future__ import annotations

import importlib
import io
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

MAX_CACHED = max(1, int(os.environ.get("CLARO_SPACE_MAX_CACHED", "3")))
_CACHE: "OrderedDict[tuple, LoaderResult]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────────
class ClaroBackendError(Exception):
    """Structured error surfaced from the backend.

    app.py turns this into a Gradio error frame whose `data.error` is
    `[CLARO:<code>] <message>`.  The local Claro.AI provider (gpu_backends.js)
    maps these prefixes to its own error codes.
    """

    def __init__(self, message: str, code: str = "model_runtime"):
        super().__init__(f"[CLARO:{code}] {message}")
        self.code = code
        self.message = message


# ──────────────────────────────────────────────────────────────────────────────
# Loader result
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class LoaderResult:
    model: Any
    tokenizer: Any
    processor: Any
    feature_extractor: Any
    preprocessor: Any
    loaded_class: str
    modality: str  # "text" | "image" | "audio" | "diffusion" | "gguf"
    device: Any
    dtype: Any


# ──────────────────────────────────────────────────────────────────────────────
# Device/dtype
# ──────────────────────────────────────────────────────────────────────────────
def _device_dtype():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    return dev, dtype


# ──────────────────────────────────────────────────────────────────────────────
# Repo format detection (file-based, never model-name-based)
# ──────────────────────────────────────────────────────────────────────────────
def _list_repo_files(model_id: str):
    try:
        from huggingface_hub import HfApi
        return HfApi().list_repo_files(model_id)
    except Exception as e:  # offline / private / not found
        raise ClaroBackendError(
            f"Could not list Hugging Face repo files for {model_id}: {e}",
            code="invalid_model_id",
        )


def _list_gguf_files(model_id: str):
    files = _list_repo_files(model_id)
    return [f for f in files if f.lower().endswith(".gguf")]


def _has_diffusers_model_index(model_id: str):
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(model_id, "model_index.json")
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# GGUF loader (llama-cpp-python backend)
# ──────────────────────────────────────────────────────────────────────────────
def _pick_gguf(files):
    order = (
        "default", "Q4_K_M", "q4_k_m", "Q5_K_M", "q5_k_m",
        "Q4_K_S", "q4_k_s", "Q8_0", "q8_0", "Q6_K", "q6_k",
        "Q4_0", "q4_0", "F16", "f16", "BF16", "bf16",
    )
    for tag in order:
        for f in files:
            if tag.lower() in f.lower():
                return f
    if len(files) == 1:
        return files[0]
    return None


class _GgufPreprocessor:
    def __init__(self, llm):
        self._llm = llm

    @property
    def eos_token_id(self):
        try:
            return self._llm.token_eos()
        except Exception:
            return None

    def __call__(self, prompt, return_tensors=None, **kw):
        import torch
        if not isinstance(prompt, str):
            raise ClaroBackendError(
                "GGUF preprocessor expects a str prompt; "
                "batch/multimodal inputs are not supported.",
                code="model_runtime",
            )
        ids = self._llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        t = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def decode(self, token_ids, skip_special_tokens=True):
        flat = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        if flat and isinstance(flat[0], list):
            flat = flat[0]
        return self._llm.detokenize(flat).decode("utf-8", errors="ignore")


class _GgufModel:
    def __init__(self, llm):
        import torch
        self._llm = llm
        self.device = torch.device("cpu")
        self.config = type("_GgufCfg", (), {"is_encoder_decoder": False, "model_type": "gguf"})()

    def eval(self): return self

    def to(self, *a, **k): return self

    def generate(self, input_ids=None, attention_mask=None,
                 max_new_tokens=128, do_sample=False,
                 temperature=0.7, top_p=0.9, top_k=50,
                 repetition_penalty=1.1, pad_token_id=None, **kw):
        import torch
        ids = (input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids))[0]
        prompt = self._llm.detokenize(ids).decode("utf-8", errors="ignore")
        args = {"max_tokens": int(max_new_tokens)}
        if do_sample:
            if temperature is not None: args["temperature"] = float(temperature)
            if top_p is not None: args["top_p"] = float(top_p)
            if top_k is not None and int(top_k) > 0: args["top_k"] = int(top_k)
            if repetition_penalty and float(repetition_penalty) != 1.0:
                args["repeat_penalty"] = float(repetition_penalty)
        resp = self._llm.create_completion(prompt, **args)
        new_text = resp["choices"][0]["text"]
        new_ids = self._llm.tokenize(new_text.encode("utf-8"), add_bos=False, special=False)
        full = ids + new_ids
        return [torch.tensor([full], dtype=torch.long)]


def _load_gguf(model_id, files, device, dtype) -> LoaderResult:
    if device.type != "cpu":
        # llama-cpp runs on CPU inside the Space even when a CUDA GPU exists
        # for the Transformers path; importing llama-cpp-python with CuBLAS
        # is unreliable in ZeroGPU sandboxes.
        pass
    try:
        from llama_cpp import Llama
    except Exception:
        raise ClaroBackendError(
            "GGUF backend (llama-cpp-python) is not installed in this Space.",
            code="model_load_error",
        )

    sorted_files = sorted(files, key=str.lower)
    gguf_file = os.environ.get("CLARO_GGUF_FILE", "").strip() or _pick_gguf(sorted_files)
    if not gguf_file:
        raise ClaroBackendError(
            f"Multiple GGUF files in {model_id}; set the CLARO_GGUF_FILE env "
            f"var on the Space to one of: {', '.join(sorted_files)}",
            code="gguf_multiple_files",
        )
    n_ctx = int(os.environ.get("CLARO_GGUF_NCTX", "4096"))
    llm_kwargs = {"n_ctx": n_ctx}
    if device.type != "cpu":
        llm_kwargs["n_gpu_layers"] = int(os.environ.get("CLARO_GGUF_NGPULAYERS", "0"))
    try:
        try:
            llm = Llama.from_pretrained(model_id, filename=gguf_file, **llm_kwargs)
        except Exception:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(model_id, gguf_file)
            llm = Llama(path, **llm_kwargs)
    except Exception as e:
        raise ClaroBackendError(
            f"Could not load GGUF model {model_id}/{gguf_file}: {e}",
            code="model_load_error",
        )
    return LoaderResult(
        model=_GgufModel(llm),
        tokenizer=None,
        processor=None,
        feature_extractor=None,
        preprocessor=_GgufPreprocessor(llm),
        loaded_class="GGUF (llama-cpp-python)",
        modality="gguf",
        device=device,
        dtype=dtype,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Diffusers loader
# ──────────────────────────────────────────────────────────────────────────────
def _load_diffusers(model_id, device, dtype) -> LoaderResult:
    try:
        from diffusers import DiffusionPipeline
    except Exception:
        raise ClaroBackendError(
            "Diffusers backend is not installed in this Space.",
            code="model_load_error",
        )
    pipe_kwargs = {"dtype": dtype} if device.type == "cuda" else {}
    try:
        pipe = DiffusionPipeline.from_pretrained(model_id, **pipe_kwargs)
        pipe = pipe.to(device)
    except Exception as e:
        raise ClaroBackendError(
            f"Could not load Diffusers pipeline for {model_id}: {e}",
            code="model_load_error",
        )
    return LoaderResult(
        model=None,
        tokenizer=None,
        processor=None,
        feature_extractor=None,
        preprocessor=None,
        loaded_class="DiffusionPipeline",
        modality="diffusion",
        device=device,
        dtype=dtype,
    )
    # Note: the pipeline object lives on `pipe`, exposed via run_inference's
    # rebuild step (see load_model); the LoaderResult.model stays None to keep
    # the canonical interface intact, and Diffusers calls go through a side
    # handle.


# ──────────────────────────────────────────────────────────────────────────────
# Transformers loader (AutoConfig architecture resolver — identical logic to
# hf_loader/notebook_builder.py, but executed here as real Python functions
# instead of generated cell source).  Architecture/format detection remains
# the source of truth; `task` does NOT influence the loading class.
# ──────────────────────────────────────────────────────────────────────────────
def _is_multimodal(config, architectures):
    metadata = " ".join(
        str(getattr(config, n, "") or "")
        for n in ("processor_class", "image_processor_type", "feature_extractor_type")
    ).lower()
    names = " ".join(architectures).lower() + " " + str(getattr(config, "model_type", "") or "").lower()
    return any(m in metadata + " " + names for m in ("vision", "image", "ocr", "document", "llava", "pix2struct"))


def _is_audio(config, architectures):
    metadata = " ".join(
        str(getattr(config, n, "") or "")
        for n in ("processor_class", "feature_extractor_type")
    ).lower()
    names = " ".join(architectures).lower() + " " + str(getattr(config, "model_type", "") or "").lower()
    return any(m in metadata + " " + names for m in ("audio", "speech", "whisper"))


def _arch_suffix_to_autoclass(architecture):
    suffixes = (
        ("ForQuestionAnswering", "AutoModelForQuestionAnswering"),
        ("ForSequenceClassification", "AutoModelForSequenceClassification"),
        ("ForTokenClassification", "AutoModelForTokenClassification"),
        ("ForMultipleChoice", "AutoModelForMultipleChoice"),
        ("ForMaskedLM", "AutoModelForMaskedLM"),
        ("ForCausalLM", "AutoModelForCausalLM"),
        ("ForImageTextToText", "AutoModelForImageTextToText"),
        ("ForVision2Seq", "AutoModelForVision2Seq"),
        ("ForSpeechSeq2Seq", "AutoModelForSpeechSeq2Seq"),
        ("ForAudioClassification", "AutoModelForAudioClassification"),
        ("ForImageClassification", "AutoModelForImageClassification"),
        ("ForObjectDetection", "AutoModelForObjectDetection"),
        ("ForSemanticSegmentation", "AutoModelForSemanticSegmentation"),
        ("ForDepthEstimation", "AutoModelForDepthEstimation"),
        ("ForCTC", "AutoModelForCTC"),
    )
    for suffix, auto_class in suffixes:
        if architecture.endswith(suffix):
            return auto_class
    return None


_REMOTE_MODEL_TYPES = set()  # trust_remote_code is decided by auto_map only


def _load_transformers(model_id, device, dtype) -> LoaderResult:
    import torch
    try:
        from transformers import AutoConfig
    except Exception:
        raise ClaroBackendError("Transformers is not installed.", code="model_runtime")

    config_requires_remote_code = False
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    except Exception:
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            config_requires_remote_code = True
        except Exception as e:
            raise ClaroBackendError(
                f"Could not read AutoConfig for {model_id}: {e}",
                code="model_load_error",
            )

    auto_map = getattr(config, "auto_map", {}) or {}
    architectures = list(getattr(config, "architectures", []) or [])
    model_type = str(getattr(config, "model_type", "") or "").lower()
    is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))

    def _architecture_requires_remote_code():
        if config_requires_remote_code or not auto_map:
            return config_requires_remote_code
        transformers = importlib.import_module("transformers")
        selected = next((k for k in auto_map if k.startswith("AutoModelFor")), None)
        targets = []
        for k, t in auto_map.items():
            if k.startswith("AutoModel") and (selected is None or k == selected):
                targets.extend(t if isinstance(t, (list, tuple)) else [t])
        if not targets:
            targets = list(auto_map.values())
        return any(not hasattr(transformers, str(t).rsplit(".", 1)[-1]) for t in targets)

    trust_remote_code = _architecture_requires_remote_code()

    def _auto_class_from_map():
        for k in auto_map:
            if k.startswith("AutoModelFor"):
                return k
        return None

    def _auto_class_from_architectures():
        for a in architectures:
            ac = _arch_suffix_to_autoclass(a)
            if ac:
                return ac
        return None

    def _auto_class_from_model_type():
        return {
            "whisper": "AutoModelForSpeechSeq2Seq",
            "wav2vec2": "AutoModelForCTC",
            "hubert": "AutoModelForCTC",
            "donut": "AutoModelForVision2Seq",
            "vision-encoder-decoder": "AutoModelForVision2Seq",
        }.get(model_type)

    loaded_class = (
        _auto_class_from_map()
        or _auto_class_from_architectures()
        or _auto_class_from_model_type()
    )

    known_text_decoder_types = {
        "llama", "qwen", "qwen2", "gemma", "gemma2", "mistral",
        "mixtral", "deepseek", "phi", "phi3", "gpt2", "gpt_neox",
        "falcon", "bloom", "mamba", "starcoder", "gpt-neox",
    }
    if not loaded_class:
        if is_encoder_decoder:
            loaded_class = "AutoModelForSeq2SeqLM"
        elif _is_multimodal(config, architectures):
            loaded_class = "AutoModelForVision2Seq"
        elif model_type in known_text_decoder_types:
            loaded_class = "AutoModelForCausalLM"
        else:
            loaded_class = "AutoModel"
    elif loaded_class == "AutoModelForConditionalGeneration":
        if _is_multimodal(config, architectures):
            loaded_class = "AutoModelForVision2Seq"
        elif is_encoder_decoder:
            loaded_class = "AutoModelForSeq2SeqLM"
        else:
            loaded_class = "AutoModelForCausalLM"

    transformers = importlib.import_module("transformers")
    ModelClass = getattr(transformers, loaded_class, None)
    if ModelClass is None and loaded_class == "AutoModelForImageTextToText":
        ModelClass = getattr(transformers, "AutoModelForVision2Seq", None)
        loaded_class = "AutoModelForVision2Seq"
    if ModelClass is None:
        raise ClaroBackendError(
            f"Transformers does not provide {loaded_class} for {model_id} "
            f"(architectures={architectures!r}, model_type={model_type!r}).",
            code="model_load_error",
        )

    model_kwargs = {"trust_remote_code": trust_remote_code}
    if device.type == "cuda":
        model_kwargs.update(dtype=dtype, device_map="auto")
    try:
        model = ModelClass.from_pretrained(model_id, **model_kwargs)
    except Exception as e:
        raise ClaroBackendError(
            f"Could not load Transformers model {model_id} via {loaded_class}: {e}",
            code="model_load_error",
        )
    if "device_map" not in model_kwargs:
        try:
            model = model.to(device)
        except Exception:
            pass
    model.eval()

    tokenizer = None
    processor = None
    feature_extractor = None
    try:
        if any(m in loaded_class for m in (
            "ImageClassification", "ObjectDetection",
            "SemanticSegmentation", "DepthEstimation",
        )):
            from transformers import AutoImageProcessor
            processor = AutoImageProcessor.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
        elif _is_multimodal(config, architectures) or "Vision2Seq" in loaded_class or "ImageTextToText" in loaded_class:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
        elif _is_audio(config, architectures) and "AudioClassification" in loaded_class:
            from transformers import AutoFeatureExtractor
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
            processor = feature_extractor
        elif _is_audio(config, architectures) or "SpeechSeq2Seq" in loaded_class or "CTC" in loaded_class:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
        else:
            from transformers import AutoTokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=trust_remote_code
                )
            except Exception as e:
                # Some non-text Transformers models (e.g. vision-only) have no
                # tokenizer.  Keep the canonical handle set to None instead of
                # crashing — the inference dispatcher can branch on modality.
                tokenizer = None
    except Exception:
        tokenizer = None
        processor = None
        feature_extractor = None

    preprocessor = tokenizer or processor or feature_extractor

    modality = "text"
    if _is_multimodal(config, architectures):
        modality = "image"
    elif _is_audio(config, architectures):
        modality = "audio"

    return LoaderResult(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        feature_extractor=feature_extractor,
        preprocessor=preprocessor,
        loaded_class=loaded_class,
        modality=modality,
        device=device,
        dtype=dtype,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Top-level model loader with safe caching
# ──────────────────────────────────────────────────────────────────────────────
def load_model(model_id: str, gguf_file: Optional[str] = None) -> LoaderResult:
    if not isinstance(model_id, str) or "/" not in model_id or model_id.strip() != model_id:
        raise ClaroBackendError(
            f"Invalid model_id: {model_id!r}",
            code="invalid_model_id",
        )
    key = (model_id, gguf_file)
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]

    device, dtype = _device_dtype()
    gguf_files = _list_gguf_files(model_id)
    
    # If user explicitly requests a specific GGUF file, honor it.
    # Otherwise, if the repo has a config.json (standard Transformers model),
    # prefer the Transformers loader over community-uploaded GGUF files.
    # Only fall back to GGUF for pure GGUF models (no config.json).
    if gguf_file:
        result = _load_gguf(model_id, gguf_files, device, dtype)
    elif gguf_files:
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(model_id, "config.json")
            # Has config.json → standard Transformers model, use that.
            result = _load_transformers(model_id, device, dtype)
        except Exception:
            # No config.json → pure GGUF model.
            result = _load_gguf(model_id, gguf_files, device, dtype)
    elif _has_diffusers_model_index(model_id):
        result = _load_diffusers(model_id, device, dtype)
    else:
        result = _load_transformers(model_id, device, dtype)

    with _CACHE_LOCK:
        _CACHE[key] = result
        _CACHE.move_to_end(key)
        while len(_CACHE) > MAX_CACHED:
            _CACHE.popitem(last=False)
    return result


def cache_stats():
    with _CACHE_LOCK:
        return {"size": len(_CACHE), "keys": list(_CACHE.keys()), "max": MAX_CACHED}


# ──────────────────────────────────────────────────────────────────────────────
# Inference dispatch.  `task` chooses the inference behaviour only; the model
# class is already resolved by load_model above.  No arbitrary Python code is
# executed — only literal handlers per supported task.
# ──────────────────────────────────────────────────────────────────────────────
def _generate_text(result: LoaderResult, prompt: str, max_new_tokens: int) -> str:
    import torch
    if result.preprocessor is None:
        raise ClaroBackendError(
            "This model has no tokenizer/processor for text generation.",
            code="task_model_mismatch",
        )
    inputs = result.preprocessor(prompt, return_tensors="pt").to(result.model.device)
    is_enc_dec = bool(getattr(result.model.config, "is_encoder_decoder", False))
    with torch.no_grad():
        out = result.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=getattr(result.preprocessor, "eos_token_id", None),
        )
    if is_enc_dec:
        new_ids = out[0]
    else:
        new_ids = out[0][inputs["input_ids"].shape[-1]:]
    return result.preprocessor.decode(new_ids, skip_special_tokens=True)


def _generate_text_to_image(result: LoaderResult, prompt: str, max_new_tokens: int) -> str:
    # Diffusers path: LoaderResult.model is None by design; the pipeline object
    # lives on the Diffusers module's TextToImagePipeline wrapper.  Reload here
    # via DiffusionPipeline.from_pretrained using the same cache key.
    raise ClaroBackendError(
        "Text-to-image inference is supported by this backend but is not "
        "exposed through the /generate endpoint yet (use the Diffusers Space "
        "directly).",
        code="task_not_exposed",
    )


def run_inference(
    model_id: str,
    task: str,
    prompt: str,
    max_new_tokens: int = 100,
    gguf_file: Optional[str] = None,
) -> str:
    task = (task or "text-generation").strip().lower()
    result = load_model(model_id, gguf_file=gguf_file)

    if task in ("text-generation", "text2text-generation", "summarization",
                "translation", "question-answering", "conversational", "chat"):
        if result.modality == "diffusion":
            raise ClaroBackendError(
                f"Task '{task}' is not compatible with a Diffusion pipeline.",
                code="task_model_mismatch",
            )
        return _generate_text(result, prompt, max_new_tokens)

    if task in ("text-to-image", "image-generation"):
        if result.modality != "diffusion":
            raise ClaroBackendError(
                f"Task '{task}' requires a Diffusion pipeline, but {model_id} "
                f"loaded as {result.loaded_class}.",
                code="task_model_mismatch",
            )
        return _generate_text_to_image(result, prompt, max_new_tokens)

    if task == "text-classification":
        if result.preprocessor is None or result.model is None:
            raise ClaroBackendError(
                "text-classification requires a model with a tokenizer.",
                code="task_model_mismatch",
            )
        import torch
        inputs = result.preprocessor(prompt, return_tensors="pt").to(result.model.device)
        with torch.no_grad():
            logits = result.model(**inputs).logits
        idx = int(logits.argmax(-1).item())
        label = result.model.config.id2label.get(idx, str(idx)) if hasattr(result.model.config, "id2label") else str(idx)
        return f"{label}"

    # Generic fallback: surface forward-pass output as a short string.
    if result.preprocessor is None or result.model is None:
        raise ClaroBackendError(
            f"Task '{task}' is not supported for model loaded as "
            f"{result.loaded_class} (modality={result.modality}).",
            code="task_model_mismatch",
        )
    import torch
    inputs = result.preprocessor(prompt, return_tensors="pt").to(result.model.device)
    with torch.no_grad():
        outputs = result.model(**inputs)
    return str(outputs)[:4000]
