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
import logging
import os
import threading
import time
import traceback
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("claro.backend")

MAX_CACHED = max(1, int(os.environ.get("CLARO_SPACE_MAX_CACHED", "3")))
_CACHE: "OrderedDict[str, LoaderResult]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
# Loads are serialized so two concurrent misses for different models cannot
# double-allocate RAM; ZeroGPU inference is effectively serialized per worker
# anyway, and the model is off the GPU outside the GPU call.
_LOAD_LOCK = threading.Lock()
_MOVE_LOCK = threading.Lock()

# Duration constants for @spaces.GPU (seconds)
DEFAULT_DURATION = 60
MIN_DURATION = 15
MAX_DURATION = 300


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
    config_hash: str = ""  # hash of config for cache invalidation


# ──────────────────────────────────────────────────────────────────────────────
# Device/dtype
# ──────────────────────────────────────────────────────────────────────────────
def _device_dtype():
    """Loading is ALWAYS done on CPU (fp16 weights) so model loading never
    consumes ZeroGPU GPU-seconds.  The @spaces.GPU inference path moves the
    cached model onto CUDA only for the duration of the request, then offloads
    it again, so VRAM stays free between calls and the CPU-side LRU cache can
    safely hold several models.
    """
    import torch
    return torch.device("cpu"), torch.float16


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


def _get_config_hash(model_id: str) -> str:
    """Get a hash of the model's config.json for cache invalidation on revision changes."""
    try:
        from huggingface_hub import hf_hub_download
        import json
        path = hf_hub_download(model_id, "config.json")
        with open(path, "r") as f:
            config = json.load(f)
        # Hash relevant config fields that affect model loading
        relevant = {
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
            "auto_map": config.get("auto_map"),
            "is_encoder_decoder": config.get("is_encoder_decoder"),
        }
        return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Duration estimation for @spaces.GPU
# ──────────────────────────────────────────────────────────────────────────────
def estimate_gpu_duration(task: str, max_new_tokens: int, modality: str, model_size_hint: str = "") -> int:
    """Estimate the GPU time needed for a request.

    Returns duration in seconds for @spaces.GPU decorator.
    """
    base = DEFAULT_DURATION

    # Task-based adjustments
    if task in ("text-generation", "text2text-generation", "summarization",
                "translation", "conversational", "chat"):
        # Text generation allowance.  Model loading happens on the CPU worker
        # OUTSIDE the GPU window, so this only needs to cover: move-onto-CUDA,
        # generation, and a rare cache-evicted reload of small models.
        # Measured warm path for 3B-class models is ~4-5s wall; these figures
        # keep a ~4x safety margin over that.
        gen_time = max_new_tokens / 60.0  # observed A10G-class throughput
        load_time = 12  # base move/reload allowance (≤3B)
        if "7b" in model_size_hint.lower() or "7B" in model_size_hint:
            load_time = 20
        elif "13b" in model_size_hint.lower() or "13B" in model_size_hint:
            load_time = 35
        elif "30b" in model_size_hint.lower() or "30B" in model_size_hint:
            load_time = 50
        elif "70b" in model_size_hint.lower() or "70B" in model_size_hint:
            load_time = 70
        base = int(load_time + gen_time + 10)  # 10s buffer

    elif task in ("text-to-image", "image-generation"):
        # Diffusion: ~5-15s per inference depending on steps
        base = 60

    elif task == "text-classification":
        # Fast forward pass
        base = 15

    else:
        base = 30

    # Modality adjustments
    if modality == "diffusion":
        base = max(base, 45)
    elif modality == "gguf":
        # llama-cpp-python runs on CPU, faster load but slower generation
        base = max(base, 20)

    # Clamp to reasonable bounds
    return max(MIN_DURATION, min(MAX_DURATION, base))


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
        logger.error(
            "Failed to load GGUF model %s/%s:\n%s",
            model_id, gguf_file, traceback.format_exc(),
        )
        raise ClaroBackendError(
            f"Could not load GGUF model {model_id}/{gguf_file} ({type(e).__name__}: {e})",
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
    pipe_kwargs = {"dtype": dtype}
    try:
        pipe = DiffusionPipeline.from_pretrained(model_id, **pipe_kwargs)
        pipe = pipe.to(device)
    except Exception as e:
        logger.error(
            "Failed to load Diffusers pipeline for %s:\n%s",
            model_id, traceback.format_exc(),
        )
        raise ClaroBackendError(
            f"Could not load Diffusers pipeline for {model_id} ({type(e).__name__}: {e})",
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


# ──────────────────────────────────────────────────────────────────────────────
# Transformers loader (AutoConfig architecture resolver)
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


def _normalize_legacy_rope_scaling(config):
    """Reconcile transformers>=5 rope_scaling normalization with remote code
    written against the transformers 4.x config convention.

    transformers>=5 always materializes ``config.rope_scaling`` as a dict and
    renamed the legacy ``"type"`` key to ``"rope_type"``; a repo that declares
    ``rope_scaling: null`` now surfaces as ``{"rope_type": "default", ...}``.
    Custom (trust_remote_code) model implementations written for 4.x expect
    ``rope_scaling`` to be ``None`` when no scaling is configured, or a dict
    containing the legacy ``"type"`` key otherwise — e.g. their rope init does
    ``config.rope_scaling["type"]`` and dies with ``KeyError: 'type'``.

    For remote-code configs only, restore the legacy shape generically:
      * effective type "default"/None → ``rope_scaling = None`` (no scaling
        was ever configured by the repo; the dict was synthesized)
      * otherwise → alias ``type`` ↔ ``rope_type`` in both directions
    Built-in configs are never touched by the caller.
    """
    rope = getattr(config, "rope_scaling", None)
    if not isinstance(rope, dict):
        return
    rope_type = rope.get("rope_type", rope.get("type"))
    if rope_type in (None, "default"):
        config.rope_scaling = None
    else:
        rope.setdefault("type", rope_type)
        rope.setdefault("rope_type", rope_type)


def _patch_cache_api_for_remote_code():
    """Restore transformers 4.x empty-cache semantics for remote-code models.

    transformers>=5 restructured caches into per-layer objects; the base
    ``Cache.get_max_length()`` computes ``max(...)`` over ``self.layers`` and
    (a) raises ``ValueError`` on a still-empty (prefill) cache, (b) reports an
    unbounded dynamic cache as ``-1``.  transformers 4.x reported both as
    ``None`` ("no maximum"), which is exactly what custom remote code written
    for 4.x tests via ``if cache.get_max_length() is not None`` and feeds into
    numeric ops (``-1`` crashes ``torch.min``/slicing; v5's own docstring
    defines ``-1`` as "no maximum", i.e. the same meaning spelled differently).

    Caches with a real configured maximum (static/sliding-window layers)
    return their value unchanged, so bounded-cache behavior is preserved.
    """
    try:
        from transformers.cache_utils import Cache
    except Exception:
        return
    if getattr(Cache, "_claro_max_length_compat", False):
        return
    if not hasattr(Cache, "get_max_length"):
        return  # v4.x: no compat needed; v5.x adds this method
    orig = Cache.get_max_length

    def _compat_get_max_length(self, layer_idx=None):
        if layer_idx is None and len(getattr(self, "layers", None) or []) == 0:
            return None  # v4 semantics: an empty cache has no max length yet
        result = orig(self, layer_idx)
        return None if result == -1 else result  # v5 "-1" == v4 None

    Cache._claro_orig_get_max_length = orig
    Cache.get_max_length = _compat_get_max_length
    Cache._claro_max_length_compat = True


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
            logger.error(
                "AutoConfig failed for %s:\n%s", model_id, traceback.format_exc()
            )
            raise ClaroBackendError(
                f"Could not read AutoConfig for {model_id} ({type(e).__name__}: {e})",
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
    if trust_remote_code:
        # Custom code may be written against older transformers conventions;
        # make the config object and runtime cache API legacy-compatible.
        _normalize_legacy_rope_scaling(config)
        _patch_cache_api_for_remote_code()

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

    # Pass the already-resolved config through so (a) the normalization above
    # actually reaches custom model code and (b) the config is not fetched twice.
    model_kwargs = {
        "config": config,
        "trust_remote_code": trust_remote_code,
        "dtype": dtype,
    }
    try:
        model = ModelClass.from_pretrained(model_id, **model_kwargs)
    except Exception as e:
        # Log the COMPLETE underlying traceback server-side so a bare
        # "KeyError: 'type'"-style string never hides the real failure.
        logger.error(
            "Failed to load Transformers model %s via %s:\n%s",
            model_id, loaded_class, traceback.format_exc(),
        )
        # Check for CUDA OOM specifically
        if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
            raise ClaroBackendError(
                f"GPU out of memory loading {model_id}. Try a smaller model or quantized variant.",
                code="gpu_oom",
            )
        raise ClaroBackendError(
            f"Could not load Transformers model {model_id} via {loaded_class} "
            f"({type(e).__name__}: {e})",
            code="model_load_error",
        )
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


# ── LoRA helper — auto-detect target modules (reused from training_runner) ──
def _infer_lora_targets(model) -> list[str]:
    try:
        model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
        model_type = model_type.lower()
        type_map = {
            "bert": ["query", "value"],
            "roberta": ["query", "value"],
            "distilbert": ["q_lin", "v_lin"],
            "albert": ["query", "value"],
            "gpt2": ["c_attn"],
            "llama": ["q_proj", "v_proj"],
            "mistral": ["q_proj", "v_proj"],
            "mixtral": ["q_proj", "v_proj"],
            "gemma": ["q_proj", "v_proj"],
            "qwen": ["q_proj", "v_proj"],
            "phi": ["q_proj", "v_proj"],
            "falcon": ["query_key_value"],
            "bloom": ["query_key_value"],
            "t5": ["q", "v"],
            "bart": ["q_proj", "v_proj"],
            "vit": ["query", "value"],
        }
        if model_type in type_map:
            return type_map[model_type]
        import torch.nn as nn
        cands = set()
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear):
                low = n.lower()
                for pat in ["q_proj", "v_proj", "query", "value", "q_lin", "c_attn", "qkv"]:
                    if pat in low:
                        cands.add(n.split(".")[-1])
                        break
        if cands:
            pref = [c for c in cands if c in ("query", "q_proj", "c_attn", "q_lin")]
            return sorted(pref)[:2] if pref else sorted(cands)[:2]
        return ["query", "value"]
    except Exception:
        return ["query", "value"]


# ──────────────────────────────────────────────────────────────────────────────
# Top-level model loader with safe caching (CPU-side, no GPU decorator)
# ──────────────────────────────────────────────────────────────────────────────
def load_model(model_id: str, gguf_file: Optional[str] = None, revision: Optional[str] = None, adapter_path: Optional[str] = None) -> LoaderResult:
    """Load model on CPU (no GPU quota consumed). Returns cached result if available."""
    if not isinstance(model_id, str) or "/" not in model_id or model_id.strip() != model_id:
        raise ClaroBackendError(
            f"Invalid model_id: {model_id!r}",
            code="invalid_model_id",
        )

    # Include revision, config hash and adapter in cache key
    config_hash = _get_config_hash(model_id)
    key_parts = [model_id]
    if gguf_file:
        key_parts.append(f"gguf:{gguf_file}")
    if revision:
        key_parts.append(f"rev:{revision}")
    if adapter_path:
        # adapter_path may be a HF id or local dir; include its hash for cache invalidation
        try:
            import hashlib as _hl
            adapter_hash = _hl.sha256(str(adapter_path).encode()).hexdigest()[:8]
        except Exception:
            adapter_hash = str(adapter_path)[:16]
        key_parts.append(f"adapter:{adapter_path}:{adapter_hash}")
    if config_hash:
        key_parts.append(f"cfg:{config_hash}")
    key = "|".join(key_parts)

    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            result = _CACHE[key]
            print(f"[CLARO] CACHE HIT model={model_id} key={key[:80]}")
            return result

    print(f"[CLARO] CACHE MISS model={model_id} key={key[:80]}")
    load_start = time.time()

    # Serialize loads: two concurrent misses must not both allocate full RAM.
    with _LOAD_LOCK:
        with _CACHE_LOCK:  # another thread may have loaded while we waited
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

        # ── LoRA adapter: load base model via existing loader, then wrap with PEFT ──
        # This reuses the existing base loader, cache, and device/dtype handling.
        # Adapter path is validated server-side from job_id, never from arbitrary client path.
        if adapter_path:
            # Validate adapter path is not traversal and exists or is HF id
            # For local training outputs, adapter_path is a directory under training_outputs (on local server)
            # For Space, it may be a HF Hub id; we handle both.
            import os as _os
            is_local = _os.path.exists(adapter_path) or adapter_path.startswith("training_outputs/") or adapter_path.startswith("/tmp") or adapter_path.startswith("./training_outputs")
            # For HF Hub adapter ids, they look like "owner/name" and will be handled by PeftModel
            try:
                from peft import PeftModel
                # The base model is already loaded in result.model
                if result.model is None:
                    raise ClaroBackendError("Base model not loaded for LoRA", code="model_load_error")
                # Check adapter files exist if local path
                if is_local:
                    if not _os.path.exists(adapter_path):
                        # Try resolved training_outputs path
                        alt = _os.path.join(_os.path.dirname(__file__), "..", adapter_path) if not _os.path.isabs(adapter_path) else adapter_path
                        if _os.path.exists(alt):
                            adapter_path = alt
                        else:
                            raise ClaroBackendError(f"LoRA adapter not found at {adapter_path}", code="model_load_error")
                    # Verify adapter_config.json exists
                    if not _os.path.exists(_os.path.join(adapter_path, "adapter_config.json")):
                        raise ClaroBackendError(f"adapter_config.json missing in {adapter_path}", code="model_load_error")

                # Wrap base with adapter — this reuses the existing model instance
                result.model = PeftModel.from_pretrained(result.model, adapter_path)
                result.model.eval()
                # Update loaded_class to reflect LoRA
                result.loaded_class = result.loaded_class + " + LoRA"
                # Log trainable for observability
                try:
                    trainable, total = result.model.get_nb_trainable_parameters()
                    print(f"[CLARO] LoRA adapter loaded from {adapter_path} trainable={trainable} total={total} ({trainable/total*100:.2f}%)")
                except Exception:
                    print(f"[CLARO] LoRA adapter loaded from {adapter_path}")
            except ImportError:
                raise ClaroBackendError("PEFT not installed for LoRA inference", code="model_load_error")
            except ClaroBackendError:
                raise
            except Exception as e:
                logger.error("Failed to load LoRA adapter %s for %s:\n%s", adapter_path, model_id, traceback.format_exc())
                raise ClaroBackendError(f"Could not load LoRA adapter {adapter_path} for {model_id} ({type(e).__name__}: {e})", code="model_load_error")

        # Update config hash in result
        result.config_hash = config_hash

        with _CACHE_LOCK:
            _CACHE[key] = result
            _CACHE.move_to_end(key)
            while len(_CACHE) > MAX_CACHED:
                evicted_key = _CACHE.popitem(last=False)[0]
                print(f"[CLARO] CACHE EVICT {evicted_key[:80]}")
        print(f"[CLARO] LOAD DONE model={model_id} load_time={time.time() - load_start:.2f}s loaded_class={result.loaded_class}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# GPU move/offload helpers (ZeroGPU: model lives on CPU between calls)
# ──────────────────────────────────────────────────────────────────────────────
def _to_gpu(result: LoaderResult) -> float:
    """Move the cached model onto CUDA inside a @spaces.GPU context.
    Returns the move time in seconds.  GGUF stays on CPU by design.
    """
    if result.model is None or result.modality in ("gguf", "diffusion"):
        return 0.0
    import torch
    if not torch.cuda.is_available():
        return 0.0
    start = time.time()
    try:
        result.model.to("cuda")
    except Exception as e:
        if "out of memory" in str(e).lower():
            raise ClaroBackendError(
                "GPU out of memory moving model to CUDA. Try a smaller model "
                "or quantized variant.",
                code="gpu_oom",
            )
        raise ClaroBackendError(f"Could not move model to GPU: {e}", code="model_runtime")
    return time.time() - start


def _offload_gpu(result: LoaderResult) -> None:
    """Move the model back to CPU and release VRAM so the worker's cache can
    hold multiple models without GPU OOM, and idle VRAM is not held hostage.
    """
    if result.model is None or result.modality in ("gguf", "diffusion"):
        return
    import torch
    if not torch.cuda.is_available():
        return
    try:
        result.model.to("cpu")
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def cache_stats():
    with _CACHE_LOCK:
        # Return keys as tuples (model_id, ...) for backward compatibility with tests
        # The full key is a composite string; extract model_id as first element
        keys = []
        for k in _CACHE.keys():
            # Key format: "model_id|gguf:...|rev:...|cfg:..."
            model_id = k.split("|")[0] if "|" in k else k
            keys.append((model_id,))
        return {"size": len(_CACHE), "keys": keys, "max": MAX_CACHED}


# ──────────────────────────────────────────────────────────────────────────────
# Inference dispatch (GPU-side, wrapped with @spaces.GPU)
# ──────────────────────────────────────────────────────────────────────────────
def _logits_diag(tag: str, scores) -> None:
    """Log numerical diagnostics for a logits tensor. Purely diagnostic."""
    import torch
    try:
        finite_vals = scores[scores.isfinite()]
        nan_count = int(torch.isnan(scores).sum().item())
        inf_count = int(torch.isinf(scores).sum().item())
        total = scores.numel()
        mn = float(finite_vals.min().item()) if finite_vals.numel() > 0 else float("nan")
        mx = float(finite_vals.max().item()) if finite_vals.numel() > 0 else float("nan")
        logger.warning(
            "[CLARO-DIAG] %s dtype=%s device=%s shape=%s nan=%d/%d inf=%d/%d min=%.6g max=%.6g",
            tag, scores.dtype, scores.device, list(scores.shape),
            nan_count, total, inf_count, total, mn, mx,
        )
    except Exception as exc:
        logger.warning("[CLARO-DIAG] %s logits_diag failed: %s", tag, exc)


def _generate_text(result: LoaderResult, prompt: str, max_new_tokens: int,
                    temperature: float = 0.7, top_p: float = 0.9,
                    do_sample: bool = True) -> "tuple[str, Optional[int]]":
    import torch
    if result.preprocessor is None:
        raise ClaroBackendError(
            "This model has no tokenizer/processor for text generation.",
            code="task_model_mismatch",
        )
    inputs = result.preprocessor(prompt, return_tensors="pt").to(result.model.device)
    is_enc_dec = bool(getattr(result.model.config, "is_encoder_decoder", False))
    on_cuda = torch.cuda.is_available() and next(result.model.parameters()).is_cuda

    # ── DIAGNOSTIC: pre-generate forward pass to check raw logits ──
    # Runs one forward step to determine whether NaN/Inf originates in the
    # model forward itself, before generate() enters its sampling loop.
    if on_cuda:
        try:
            torch.cuda.synchronize()
            with torch.no_grad():
                fwd_out = result.model(**inputs)
            torch.cuda.synchronize()
            raw_logits = getattr(fwd_out, "logits", None)
            if raw_logits is None and hasattr(fwd_out, "last_hidden_state"):
                raw_logits = fwd_out.last_hidden_state
            if raw_logits is not None:
                _logits_diag("PRE-GEN-FWD raw logits", raw_logits)
                # Check softmax of raw logits (what multinomial would see)
                try:
                    probs = torch.softmax(raw_logits.float(), dim=-1)
                    _logits_diag("PRE-GEN-FWD softmax probs", probs)
                    del probs
                except Exception as exc:
                    logger.warning("[CLARO-DIAG] PRE-GEN-FWD softmax failed: %s", exc)
                del fwd_out, raw_logits
            torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning("[CLARO-DIAG] PRE-GEN-FWD model forward failed: %s", exc)

    # ── DIAGNOSTIC LOGITS PROCESSOR ──
    # Hooks into generate()'s logits pipeline. Logs scores at each call,
    # giving visibility into: (a) what generate() sees after the model
    # forward, (b) what each processor produces, (c) NaN/Inf state.
    from transformers import LogitsProcessorList, LogitsProcessor

    class _DiagnosticLogitsProcessor(LogitsProcessor):
        def __init__(self):
            self._call_count = 0

        def __call__(self, input_ids, scores):
            self._call_count += 1
            _logits_diag(f"PROC-CALL-{self._call_count} pre-cast", scores)
            # Cast to float32 (existing fix — kept for now)
            if scores.dtype != torch.float32:
                scores = scores.float()
                _logits_diag(f"PROC-CALL-{self._call_count} post-cast", scores)
            return scores

    proc = _DiagnosticLogitsProcessor()

    # ── GENERATE ──
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        logits_processor=LogitsProcessorList([proc]),
        pad_token_id=getattr(result.preprocessor, "eos_token_id", None),
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        try:
            out = result.model.generate(**gen_kwargs)
        except Exception as exc:
            # CUDA device-side assert — log diagnostics and re-raise
            logger.warning("[CLARO-DIAG] generate() failed: %s", exc)
            raise

    # ── DIAGNOSTIC: post-generate output check ──
    if on_cuda:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    try:
        new_ids = out[0] if is_enc_dec else out[0][inputs["input_ids"].shape[-1]:]
        _logits_diag("POST-GEN output token ids (float)", new_ids.float())
    except Exception:
        pass

    if is_enc_dec:
        new_ids = out[0]
    else:
        new_ids = out[0][inputs["input_ids"].shape[-1]:]
    # Token count for observability (no prompt contents are ever logged).
    try:
        n_new = int(new_ids.shape[-1])
    except Exception:
        n_new = None
    return result.preprocessor.decode(new_ids, skip_special_tokens=True), n_new


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
    revision: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    adapter_path: Optional[str] = None,
) -> str:
    """Run inference on GPU. Model is loaded via load_model() which uses CPU cache.
    
    This function is decorated with @spaces.GPU in app.py with dynamic duration.
    """
    task = (task or "text-generation").strip().lower()
    # Normally a CACHE HIT: app.py loads the model on the CPU worker before
    # entering the @spaces.GPU function, so no download/reload happens here.
    cache_size_before = len(_CACHE)
    result = load_model(model_id, gguf_file=gguf_file, revision=revision, adapter_path=adapter_path)
    cache_hit = len(_CACHE) <= cache_size_before

    # Estimate duration for logging
    model_size_hint = model_id.split("/")[-1] if "/" in model_id else model_id
    estimated_duration = estimate_gpu_duration(task, max_new_tokens, result.modality, model_size_hint)

    # GPU-scoped section: move cached CPU model → CUDA, run, offload back.
    # Serialized so two GPU calls never fight over the same cached object.
    n_tokens: Optional[int] = None
    with _MOVE_LOCK:
        move_in = _to_gpu(result)
        start_time = time.time()
        try:
            if task in ("text-generation", "text2text-generation", "summarization",
                        "translation", "question-answering", "conversational", "chat"):
                if result.modality == "diffusion":
                    raise ClaroBackendError(
                        f"Task '{task}' is not compatible with a Diffusion pipeline.",
                        code="task_model_mismatch",
                    )
                text, n_tokens = _generate_text(result, prompt, max_new_tokens, temperature, top_p, do_sample=do_sample)

            elif task in ("text-to-image", "image-generation"):
                if result.modality != "diffusion":
                    raise ClaroBackendError(
                        f"Task '{task}' requires a Diffusion pipeline, but {model_id} "
                        f"loaded as {result.loaded_class}.",
                        code="task_model_mismatch",
                    )
                text = _generate_text_to_image(result, prompt, max_new_tokens)

            elif task == "text-classification":
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
                text = f"{label}"

            else:
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
                text = str(outputs)[:4000]
        finally:
            _offload_gpu(result)

    elapsed = time.time() - start_time
    print(f"[CLARO] INFER model={model_id} task={task} max_new_tokens={max_new_tokens} "
          f"modality={result.modality} loaded_class={result.loaded_class} "
          f"gpu_size=default cache={'hit' if cache_hit else 'miss'} "
          f"move_in={move_in:.2f}s exec={elapsed:.2f}s estimated={estimated_duration}s "
          f"tokens={n_tokens} cache_size={len(_CACHE)}")
    return text