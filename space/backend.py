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
import json
import logging
import os
import re
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
    model_id: str = ""     # repo id — used by model-specific embedding paths


# ──────────────────────────────────────────────────────────────────────────────
# Device/dtype
# ──────────────────────────────────────────────────────────────────────────────
# Model families whose activations are KNOWN to be FP16-incompatible.
# Google's EmbeddingGemma documentation explicitly states its activations are
# incompatible with FP16 and recommends FP32/BF16 — FP16 weights silently
# produce NaN activations on CUDA (the raw HTTP 500 on sentence-similarity).
_FP16_INCOMPATIBLE_FAMILIES = ("embeddinggemma",)


def _resolve_model_dtype(model_id):
    """Model-specific weight precision. Defaults to the historical FP16;
    families with documented FP16 incompatibility get BF16 when the target
    hardware supports it, else FP32 — never FP16."""
    import torch
    mid = str(model_id or "").lower()
    if any(f in mid for f in _FP16_INCOMPATIBLE_FAMILIES):
        try:
            if torch.cuda.is_available() and getattr(
                    torch.cuda, "is_bf16_supported", lambda: False)():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float32
    return torch.float16


def _device_dtype(model_id=None):
    """Loading is ALWAYS done on CPU so model loading never consumes ZeroGPU
    GPU-seconds.  The @spaces.GPU inference path moves the cached model onto
    CUDA only for the duration of the request, then offloads it again, so
    VRAM stays free between calls and the CPU-side LRU cache can safely hold
    several models.  Precision is model-specific (see _resolve_model_dtype).
    """
    import torch
    return torch.device("cpu"), _resolve_model_dtype(model_id)


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


# ── TEMP-DIAG: GGUF worker diagnostics (prints only — never raises, never ──
# logs prompts, weights, tokens, or secrets) ───────────────────────────────
_CUDA_ERR_RE = re.compile(
    r"cuda|cublas|cudnn|nccl|cuBLAS|cuDNN|\bcu\w+|out of memory|"
    r"device-side assert|illegal memory access",
    re.IGNORECASE,
)


def _is_cuda_runtime_error(e) -> bool:
    """True when an exception looks CUDA-related (type or message)."""
    try:
        return bool(_CUDA_ERR_RE.search(f"{type(e).__name__}: {e}"))
    except Exception:
        return False


def _gguf_setting(llm, attr, env_name=None, default="unknown"):
    """Read a numeric llama.cpp setting: live llm attribute (or no-arg
    getter) first, loader env default second, 'unknown' last. Never raises."""
    try:
        v = getattr(llm, attr, None)
        if callable(v):
            try:
                v = v()
            except Exception:
                v = None
        if v is not None:
            return v
    except Exception:
        pass
    if env_name:
        try:
            ev = os.environ.get(env_name)
            if ev not in (None, ""):
                return ev
        except Exception:
            pass
    return default


def _gguf_worker_preflight(llm):
    """Print GGUF pre-inference facts (import, build, CUDA, model metadata).

    Safe numeric/model metadata only. Every probe is guarded: diagnostics
    can never raise and never log the prompt or weights.
    """
    def _show(label, value):
        try:
            print(f"[CLARO-GGUF] {label}={value}", flush=True)
        except Exception:
            pass
    try:
        import llama_cpp
        _show("llama_cpp import", "success")
        _show("llama_cpp version", getattr(llama_cpp, "__version__", "unknown"))
    except Exception as _e:
        _show("llama_cpp import", f"failure ({type(_e).__name__}: {_e})")
        return
    try:
        if hasattr(llama_cpp, "llama_supports_gpu_offload"):
            _show("GPU offload support", llama_cpp.llama_supports_gpu_offload())
        else:
            _show("GPU offload support", "unknown")
    except Exception as _e:
        _show("GPU offload support", f"unknown ({type(_e).__name__})")
    try:
        import torch
        _show("torch.cuda.is_available()", torch.cuda.is_available())
        try:
            _show("torch.cuda.get_device_name(0)",
                  torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a (no cuda)")
        except Exception as _e:
            _show("torch.cuda.get_device_name(0)", f"unknown ({type(_e).__name__}: {_e})")
    except Exception as _e:
        _show("torch.cuda.is_available()", f"unknown ({type(_e).__name__})")
    try:
        if hasattr(llama_cpp, "llama_print_system_info"):
            _show("llama.cpp build", str(llama_cpp.llama_print_system_info())[:1500])
        else:
            _show("llama.cpp build", "unknown")
    except Exception as _e:
        _show("llama.cpp build", f"unknown ({type(_e).__name__})")
    try:
        _mp = getattr(llm, "model_path", None) or "unknown"
        _show("model path", _mp)
        try:
            _show("model file size",
                  os.path.getsize(_mp) if isinstance(_mp, str) and os.path.isfile(_mp)
                  else "unknown (not a local file)")
        except Exception as _e:
            _show("model file size", f"unknown ({type(_e).__name__})")
    except Exception as _e:
        _show("model path", f"unknown ({type(_e).__name__})")
    try:
        _show("n_gpu_layers", _gguf_setting(llm, "n_gpu_layers", "CLARO_GGUF_NGPULAYERS", "0"))
        _show("n_ctx", _gguf_setting(llm, "n_ctx", "CLARO_GGUF_NCTX", "4096"))
        _show("n_batch", _gguf_setting(llm, "n_batch", None, "unknown"))
    except Exception as _e:
        _show("n_gpu_layers/n_ctx/n_batch", f"unknown ({type(_e).__name__})")


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
        # TEMP-DIAG: exact llama_cpp call site (app.py _gpu_infer ->
        # run_inference -> _generate_text -> here -> create_completion).
        # Prints only; the original exception is re-raised unchanged so the
        # ZeroGPU error mapping is intact.
        _gguf_worker_preflight(self._llm)
        try:
            resp = self._llm.create_completion(prompt, **args)
        except Exception as e:
            print("[CLARO-GGUF] worker exception type=", type(e).__name__, flush=True)
            print("[CLARO-GGUF] worker exception=", str(e)[:2000], flush=True)
            print(traceback.format_exc(), flush=True)
            if _is_cuda_runtime_error(e):
                _prev = os.environ.get("CUDA_LAUNCH_BLOCKING")
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                print(f"[CLARO-GGUF] CUDA_LAUNCH_BLOCKING was={_prev!r} now='1' (diagnostic run; unset to revert)", flush=True)
            raise
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
        model_id=model_id,
    )


# ── Native-library ABI guard (diagnostic only — loader behavior unchanged) ──
# Rejects a llama-cpp-python installation whose vendored libllama.so is
# linked against musl libc (Alpine-style build), which cannot load on the
# glibc-based Space image ("libc.musl-x86_64.so.1: cannot open shared object
# file"). Pure stdlib (os only) so it is unit-testable without torch/GPU.
_MUSL_MARKERS = (b"libc.musl", b"ld-musl")
_LIBLLAMA_SCAN_BYTES = 8 * 1024 * 1024
_LIBLLAMA_SCAN_LIMIT = 5000


def _find_libllama(package_dir: str) -> str | None:
    """Locate the vendored libllama shared library under an installed
    llama_cpp package directory. Returns the path, or None when absent."""
    base = os.path.join(str(package_dir or ""), "lib", "libllama.so")
    if base and os.path.isfile(base):
        return base
    try:
        seen = 0
        for root, _dirs, files in os.walk(str(package_dir or "")):
            for name in files:
                seen += 1
                if seen > _LIBLLAMA_SCAN_LIMIT:
                    return None
                if "llama" in name and name.endswith(".so"):
                    return os.path.join(root, name)
    except OSError:
        return None
    return None


def _libllama_requires_musl(lib_path: str) -> bool | None:
    """Bytes-scan a libllama.so for musl-libc linkage markers.

    Returns True (musl-linked → REJECT), False (no musl marker in the
    scanned window), or None (unreadable/empty — inconclusive, never a pass).
    """
    try:
        with open(lib_path, "rb") as fh:
            blob = fh.read(_LIBLLAMA_SCAN_BYTES)
    except OSError:
        return None
    if not blob:
        return None
    return any(m in blob for m in _MUSL_MARKERS)


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
        model_id=model_id,
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
        model_id=model_id,
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

        device, dtype = _device_dtype(model_id)
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
    # Minimal accelerator self-test BEFORE touching the model: a 1-element
    # CUDA op. If the accelerator itself is broken (bad CUDA init, driver /
    # binary mismatch), fail here with code=gpu_unavailable instead of letting
    # the subsequent model ops fail with a misleading model error.
    try:
        _probe = torch.randn(1, device="cuda") + 1
        print(f"[CLARO] cuda self-test ok: {float(_probe.flatten()[0]):.4f}", flush=True)
        del _probe
    except Exception as e:
        raise ClaroBackendError(
            f"ZeroGPU accelerator self-test failed ({type(e).__name__}: {e}). "
            f"The GPU itself is unusable — this is an environment issue, not the model.",
            code="gpu_unavailable",
        )
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


def _resolve_prompt_format(preprocessor, prompt_format) -> str:
    """Validate prompt_format and resolve 'auto' against the tokenizer.

    - 'raw':  tokenize the prompt verbatim (historical behavior).
    - 'chat': apply the model's chat template (required by instruct models
      like MuyaoYuan/LoCo-7B whose documented usage is apply_chat_template).
    - 'auto': chat when the tokenizer ships a chat template, else raw.

    Raises ClaroBackendError(bad_request) for unknown values.
    """
    fmt = (prompt_format if prompt_format is not None else "auto")
    fmt = str(fmt).strip().lower() or "auto"
    if fmt not in ("auto", "chat", "raw"):
        raise ClaroBackendError(
            f"prompt_format must be one of auto/chat/raw (got {prompt_format!r}).",
            code="bad_request",
        )
    if fmt == "auto":
        has_template = bool(getattr(preprocessor, "chat_template", None))
        return "chat" if has_template else "raw"
    return fmt


def _tokenize_prompt(result: "LoaderResult", prompt: str, prompt_format: str) -> dict:
    """Tokenize the prompt honoring prompt_format ('raw' | 'chat').

    'chat' builds a single user message and applies the model's own chat
    template with add_generation_prompt=True, so generation starts in the
    assistant turn. Returns the tokenized inputs dict (batch of 1).
    """
    fmt = _resolve_prompt_format(result.preprocessor, prompt_format)
    if fmt == "chat":
        messages = [{"role": "user", "content": prompt}]
        try:
            inputs = result.preprocessor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            # Older transformers without return_dict= on apply_chat_template:
            # render to text, then tokenize the rendered template normally.
            rendered = result.preprocessor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = result.preprocessor(rendered, return_tensors="pt")
        return inputs
    return result.preprocessor(prompt, return_tensors="pt")


def _generate_text_gguf(result: LoaderResult, prompt: str, max_new_tokens: int,
                      temperature: float = 0.7, top_p: float = 0.9,
                      do_sample: bool = True, prompt_format: str = "auto") -> "tuple[str, Optional[int]]":
    """GGUF text generation via llama.cpp (diagnostic-wrapped).

    The shared _generate_text preamble below (parameters(), forward pass,
    logits processors) is Transformers-specific and cannot execute for a
    llama.cpp model; this helper issues the identical model.generate call
    and decode contract ((text, n_new)) directly. Prints [CLARO-GGUF] worker
    diagnostics (via _GgufModel.generate) and re-raises the original
    exception unchanged so the ZeroGPU error mapping is intact.
    """
    inputs = _tokenize_prompt(result, prompt, prompt_format)
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    try:
        out = result.model.generate(**gen_kwargs)
    except Exception as e:
        print("[CLARO-GGUF] gguf generate path exception type=", type(e).__name__, flush=True)
        print("[CLARO-GGUF] gguf generate path exception=", str(e)[:2000], flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    try:
        new_ids = out[0][inputs["input_ids"].shape[-1]:]
    except Exception:
        try:
            new_ids = out[0]
        except Exception:
            new_ids = []
    try:
        n_new = int(new_ids.shape[-1])
    except Exception:
        n_new = None
    return result.preprocessor.decode(new_ids, skip_special_tokens=True), n_new


def _generate_text(result: LoaderResult, prompt: str, max_new_tokens: int,
                    temperature: float = 0.7, top_p: float = 0.9,
                    do_sample: bool = True, prompt_format: str = "auto") -> "tuple[str, Optional[int]]":
    import torch
    if result.preprocessor is None:
        raise ClaroBackendError(
            "This model has no tokenizer/processor for text generation.",
            code="task_model_mismatch",
        )
    if result.modality == "gguf":
        # GGUF has no Transformers forward/generate pipeline (no
        # parameters(), no __call__, no logits); route straight to the
        # llama.cpp completion call. All other modalities use the shared
        # path below unchanged.
        return _generate_text_gguf(result, prompt, max_new_tokens,
                                   temperature, top_p, do_sample, prompt_format)
    inputs = _tokenize_prompt(result, prompt, prompt_format).to(result.model.device)
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


# ──────────────────────────────────────────────────────────────────────────────
# Embeddings (feature-extraction / sentence-similarity)
# ──────────────────────────────────────────────────────────────────────────────
# Pooling strategy is chosen per architecture — never assumed globally:
#   - Causal-LM wrappers (Qwen3-Embedding resolves to AutoModelForCausalLM):
#     LAST-TOKEN pooling per the official Qwen3-Embedding example
#     (left padding + attention-mask-aware final-token selection + L2 norm).
#   - Plain AutoModel encoders (BERT/MiniLM/BGE/E5/GTE class): mean pooling
#     over valid (attention-mask) tokens.
# An embedding alone is NOT a similarity score; sentence-similarity computes
# cosine similarity between an anchor text and candidate texts.

_EMBEDDING_CAUSAL_CLASSES = {"AutoModelForCausalLM"}


def _pooling_strategy_for(result) -> str:
    """Pick the pooling strategy from the loaded architecture.

    Raises task_model_mismatch for model classes with no defined embedding
    contract (diffusion pipelines, classifiers, etc.).
    """
    loaded = getattr(result, "loaded_class", "") or ""
    if loaded in _EMBEDDING_CAUSAL_CLASSES:
        return "last_token"
    if loaded == "AutoModel":
        return "mean"
    raise ClaroBackendError(
        f"Model loaded as {loaded or 'unknown class'} does not expose a supported "
        "embedding interface (expected AutoModel or a causal-LM wrapper such as "
        "Qwen3-Embedding's AutoModelForCausalLM).",
        code="task_model_mismatch",
    )


def _select_last_hidden(model_output):
    """Final hidden states, preferring hidden_states[-1] (causal-LM wrapper
    loaded with output_hidden_states=True) over last_hidden_state."""
    hidden = getattr(model_output, "hidden_states", None)
    if hidden:
        return hidden[-1]
    return getattr(model_output, "last_hidden_state")


def _qwen3_query_text(text: str, instruction: Optional[str]) -> str:
    """Official Qwen3-Embedding query format. Instruction is OPTIONAL and
    must be provided by the caller — no default retrieval instruction is
    invented for generic embedding use cases."""
    if not instruction:
        return text
    return f"Instruct: {instruction}\nQuery: {text}"


def _compute_embeddings(result: "LoaderResult", texts, instruction: Optional[str] = None,
                        internal_task: Optional[str] = None):
    """Tokenize, forward under no_grad, pool, apply the model's own
    projection modules (when its architecture defines them), and
    L2-normalize.

    Returns (vectors: list[list[float]], pooling: str). The mask-aware
    position selection works for BOTH left and right padding, but causal-LM
    (Qwen3-style) models get tokenizer.padding_side='left' per the official
    example. Models with Sentence-Transformers Dense modules (EmbeddingGemma)
    get mean pooling followed by their learned projection chain — the
    normalization always happens LAST, after the projection.
    """
    import torch
    tok = result.preprocessor
    if tok is None:
        raise ClaroBackendError(
            "This model has no tokenizer for embedding extraction.",
            code="task_model_mismatch",
        )
    pooling = _pooling_strategy_for(result)
    if instruction:
        texts = [_qwen3_query_text(t, instruction) for t in texts]
    # Model-specific task prefixes (EmbeddingGemma documents them; Qwen3 and
    # generic encoders deliberately get none).
    if internal_task:
        model_id = getattr(result, "model_id", "") or ""
        texts = [_embedding_prefix(model_id, internal_task, t) for t in texts]
    if pooling == "last_token" and hasattr(tok, "padding_side"):
        tok.padding_side = "left"
    # Qwen-class tokenizers ship no pad token; the official Qwen3-Embedding
    # example uses the EOS token for padding. Without this, padding=True
    # raises "does not have a padding token".
    if getattr(tok, "pad_token", None) is None \
            and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    enc = tok(list(texts), padding=True, truncation=True, return_tensors="pt")
    if hasattr(enc, "to"):
        enc = enc.to(result.model.device)
    with torch.no_grad():
        out = result.model(**enc, output_hidden_states=True)
    last_hidden = _select_last_hidden(out)
    mask = enc["attention_mask"]
    if pooling == "last_token":
        # Index of the LAST valid token per row (side-agnostic: masked
        # positions contribute 0 to mask*arange, so argmax lands on the
        # final attended position whether padding is left or right).
        positions = (mask * torch.arange(mask.shape[1], device=mask.device)
                     .unsqueeze(0)).argmax(dim=1)
        emb = last_hidden[torch.arange(last_hidden.shape[0], device=mask.device), positions]
    else:
        m = mask.unsqueeze(-1).to(last_hidden.dtype)
        emb = (last_hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    # Model-specific learned projections (EmbeddingGemma 2_Dense/3_Dense)
    # run BEFORE normalization; normalization is always the final step.
    emb, projected = _apply_dense_projection(result, emb)
    pooling_label = pooling if not projected else f"{pooling}+dense_projection"
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.detach().to("cpu").tolist(), pooling_label


def _cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _format_embedding_result(model_id: str, vectors, pooling: str,
                             task: str = "feature-extraction",
                             similarities=None, max_preview: int = 8) -> str:
    """Compact, structured, human-readable summary. Never stringifies raw
    torch output objects; the full vector stays out of the terminal by
    default (the existing single-string API surface cannot carry a
    structured payload)."""
    lines = [
        "=== Embedding output ===",
        f"Model: {model_id}",
        f"Task: {task}",
        f"Vectors: {len(vectors)}",
        f"Dimensions: {len(vectors[0])}",
        f"Pooling: {pooling}",
    ]
    for i, v in enumerate(vectors):
        norm = sum(x * x for x in v) ** 0.5
        lines.append(f"Vector[{i}] Norm: {norm:.4f}")
    preview = ", ".join(f"{x:.4f}" for x in vectors[0][:max_preview])
    if len(vectors[0]) > max_preview:
        preview += ", ..."
    lines.append(f"Preview: [{preview}]")
    if similarities is not None:
        for j, s in enumerate(similarities):
            lines.append(f"Similarity(anchor, text {j + 1}): {s:.4f}")
    return "\n".join(lines)


def _embedding_inference(result: "LoaderResult", model_id: str, prompt: str, task: str) -> str:
    if result.model is None or result.modality == "diffusion":
        raise ClaroBackendError(
            f"Task '{task}' requires a Transformers model, but {model_id} "
            f"loaded as {result.loaded_class} (modality={result.modality}).",
            code="task_model_mismatch",
        )
    texts = [ln.strip() for ln in (prompt or "").splitlines() if ln.strip()]
    if not texts:
        raise ClaroBackendError("prompt is required for embedding tasks.", code="bad_request")
    if task == "sentence-similarity" and len(texts) < 2:
        raise ClaroBackendError(
            "sentence-similarity needs at least two newline-separated texts "
            "(first line = anchor, remaining lines = candidates). A single "
            "embedding is not a similarity score.",
            code="bad_request",
        )
    vectors, pooling = _compute_embeddings(result, texts)
    similarities = None
    if task == "sentence-similarity":
        similarities = [_cosine_similarity(vectors[0], v) for v in vectors[1:]]
    return _format_embedding_result(model_id, vectors, pooling, task, similarities)


# ──────────────────────────────────────────────────────────────────────────────
# Task-aware inference handlers
# ──────────────────────────────────────────────────────────────────────────────
# A model's intended task decides how its input is prepared, how inference
# runs, and how the output is interpreted. Each supported task has its own
# handler; unsupported tasks raise a clear error instead of falling back to
# generic text generation or dumping raw model objects.

SUPPORTED_TEXT_TASKS = (
    "text-generation, text2text-generation, summarization, translation, "
    "question-answering, text-classification, feature-extraction, "
    "sentence-similarity"
)


def _parse_task_inputs(task_inputs):
    """task_inputs arrives as a dict or a JSON string; {} is the safe default."""
    if task_inputs is None or task_inputs == "" or task_inputs == {}:
        return {}
    if isinstance(task_inputs, str):
        try:
            task_inputs = json.loads(task_inputs)
        except Exception:
            raise ClaroBackendError(
                "task_inputs must be valid JSON (e.g. {\"question\": ..., \"context\": ...}).",
                code="invalid_input",
            )
    if not isinstance(task_inputs, dict):
        raise ClaroBackendError(
            "task_inputs must be a JSON object of task-specific fields.",
            code="invalid_input",
        )
    return task_inputs


def _require_field(task_inputs, key, task):
    val = task_inputs.get(key)
    if val is None or (isinstance(val, str) and not val.strip()):
        raise ClaroBackendError(
            f"The '{key}' field is required for {task}. "
            f"Fill it in the inference cell or the task inputs.",
            code="invalid_input",
        )
    return str(val)


def _split_lines(text):
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _tokenizer_for(result):
    tok = result.preprocessor
    if tok is None:
        raise ClaroBackendError(
            "This model has no tokenizer/processor for this task.",
            code="task_model_mismatch",
        )
    return tok


def _is_encoder_decoder(result):
    return bool(getattr(getattr(result.model, "config", None), "is_encoder_decoder", False))


def _generate_from_inputs(result, inputs, max_new_tokens=256, do_sample=False,
                          length_key="max_new_tokens", **extra):
    """Run model.generate and decode only the newly generated tokens for
    decoder-only models; encoder-decoders emit the full output sequence.
    length_key selects the generate() length parameter: causal-LM models
    take max_new_tokens, encoder-decoders (HF summarization convention)
    take max_length."""
    import torch
    tok = _tokenizer_for(result)
    gen = dict(**inputs, do_sample=bool(do_sample))
    gen[length_key] = int(max_new_tokens)
    gen.update(extra)
    gen.setdefault("pad_token_id", getattr(tok, "eos_token_id", None))
    with torch.no_grad():
        out = result.model.generate(**gen)
    if _is_encoder_decoder(result):
        new_ids = out[0]
    else:
        new_ids = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(new_ids, skip_special_tokens=True).strip()


# ── Model-specific embedding architectures & task prefixes ──────────────────
# EmbeddingGemma (google/embeddinggemma-300m) is a Sentence-Transformers
# model: Gemma3TextModel backbone → attention-mask-aware mean pooling →
# learned Dense 768→3072 (2_Dense) → learned Dense 3072→768 (3_Dense) →
# L2 normalization. The Dense projection weights live OUTSIDE the main
# checkpoint (2_Dense/model.safetensors, 3_Dense/model.safetensors) and are
# NOT loaded by AutoModel — they must be loaded explicitly or the embeddings
# are raw backbone outputs (wrong). If those weights cannot be loaded we
# raise model_load_error rather than emit incorrect embeddings.

# Task-specific input prefixes, keyed by model family then internal task.
# Sourced from each model's documented usage — never invented per request.
_EMBEDDING_TASK_PREFIXES = {
    "embeddinggemma": {
        "sentence_similarity": "task: sentence similarity | query: ",
        "retrieval": "task: search result | query: ",
        "classification": "task: classification | query: ",
        "clustering": "task: clustering | query: ",
    },
}

_DENSE_ACTIVATIONS = {
    "tanh": "tanh",
    "gelu": "gelu",
    "relu": "relu",
    "linear": None,
    "identity": None,
}


def _embedding_prefix(model_id, internal_task, text):
    """Model-specific task prefix. Returns the text unchanged for models
    without documented prefixes (Qwen3, generic encoders) and never
    duplicates a prefix that is already present."""
    mid = str(model_id or "").lower()
    for family, prefixes in _EMBEDDING_TASK_PREFIXES.items():
        if family in mid and internal_task in prefixes:
            prefix = prefixes[internal_task]
            if text.startswith(prefix):
                return text
            return prefix + text
    return text


# Lazily-wrapped hub/safetensors calls (patchable in tests).
def _hub_download(model_id, filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(model_id, filename)


def _safetensors_load_file(path):
    from safetensors.torch import load_file
    return load_file(path)

def _load_dense_projection(result):
    cached = getattr(result, "_claro_dense_projection", None)
    if cached is not None:
        return cached
    model_id = getattr(result, "model_id", "") or ""
    layers = []
    saw_dense = False
    partial_error = None
    try:
        import torch
        idx = 2
        while True:
            prefix = f"{idx}_Dense"
            try:
                cfg_path = _hub_download(model_id, f"{prefix}/config.json")
                saw_dense = True
            except Exception:
                break  # no more Dense modules -- normal stop
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                weights_path = _hub_download(model_id, f"{prefix}/model.safetensors")
                sd = _safetensors_load_file(weights_path)
            except Exception as e:
                partial_error = (
                    f"Model {model_id} declares a {prefix} projection module but "
                    f"its weights/config could not be loaded: {e}")
                break
            weight = sd.get("linear.weight")
            bias = sd.get("linear.bias")
            if weight is None:
                partial_error = (
                    f"{prefix}/model.safetensors has no 'linear.weight' -- "
                    "unexpected projection format.")
                break
            act_name = str(cfg.get("activation_function") or "linear").lower()
            act_name = act_name.split(".")[-1]
            act = _DENSE_ACTIVATIONS.get(act_name, "unknown")
            if act == "unknown":
                partial_error = (
                    f"{prefix} declares unknown activation_function "
                    f"{cfg.get('activation_function')!r}.")
                break
            act_fn = getattr(torch, act) if act else None
            # store weight pre-transposed for hidden @ W^T
            layers.append((torch.transpose(weight, 0, 1),
                           bias, act_fn, prefix))
            idx += 1
    except Exception:
        pass  # discovery unavailable -> treated as no Dense modules below
    if saw_dense and partial_error:
        raise ClaroBackendError(partial_error, code="model_load_error")
    try:
        object.__setattr__(result, "_claro_dense_projection", layers)
    except Exception:
        pass
    return layers

def _apply_dense_projection(result, pooled):
    """Apply the model's Dense projection chain in order (2_Dense → 3_Dense
    ...). Returns (projected, applied_bool). Weights are moved to the model's
    device/dtype so they always match the backbone's output."""
    import torch
    layers = _load_dense_projection(result)
    if not layers:
        return pooled, False
    x = pooled
    device = result.model.device
    dtype = next((p.dtype for p in result.model.parameters()
                  if hasattr(p, "dtype")), None)
    for (w_t, bias, act_fn, name) in layers:
        w = w_t.to(device=device, dtype=dtype) if dtype is not None else w_t.to(device)
        x = torch.matmul(x.to(device=device, dtype=w.dtype), w)
        if bias is not None:
            x = x + bias.to(device=device, dtype=x.dtype)
        if act_fn is not None:
            x = act_fn(x)
    return x, True


def run_summarization(result, model_id, prompt, task_inputs):
    """Summarization: encoder-decoder models use max_length/min_length as
    summary-length bounds; causal-LM wrappers summarize via an explicit
    instruction prompt (never bare next-token continuation of a document)."""
    document = (prompt or "").strip()
    if not document:
        raise ClaroBackendError(
            "Summarization needs the source document. Put it in the "
            "inference cell (input_text = \"...\").", code="invalid_input")
    try:
        max_length = int(task_inputs.get("max_length") or 142)
        min_length = int(task_inputs.get("min_length") or 56)
    except (TypeError, ValueError):
        raise ClaroBackendError("max_length/min_length must be integers.",
                                code="invalid_input")
    max_length = max(8, min(max_length, 1024))
    min_length = max(4, min(min_length, max_length - 1))
    tok = _tokenizer_for(result)
    inputs = tok(document, return_tensors="pt", truncation=True,
                 max_length=1024).to(result.model.device)
    if _is_encoder_decoder(result):
        summary = _generate_from_inputs(
            result, inputs, max_new_tokens=max_length, length_key="max_length",
            min_length=min_length, num_beams=4, early_stopping=True)
    else:
        # Causal-LM wrapper: explicit summarization instruction + new-token slice
        prompted = ("Summarize the following text.\n\n"
                    f"{document}\n\nSummary:")
        pin = tok(prompted, return_tensors="pt", truncation=True,
                  max_length=1024).to(result.model.device)
        summary = _generate_from_inputs(result, pin, max_new_tokens=max_length)
    return f"=== Summary ===\nModel: {model_id}\n\n{summary}"


def run_classification(result, model_id, prompt, task_inputs):
    """Text classification: softmax over the classification head, labels from
    config.id2label, optional top-k per input text (one per line)."""
    import torch
    if result.loaded_class != "AutoModelForSequenceClassification":
        raise ClaroBackendError(
            f"Model loaded as {result.loaded_class} has no classification "
            "head. text-classification requires a sequence-classification model.",
            code="task_model_mismatch")
    texts = _split_lines(prompt)
    if not texts:
        raise ClaroBackendError(
            "Classification needs input text (one text per line in the "
            "inference cell).", code="invalid_input")
    tok = _tokenizer_for(result)
    id2label = getattr(getattr(result.model, "config", None), "id2label", None)
    if not id2label:
        raise ClaroBackendError(
            "Model config has no id2label mapping — classification labels "
            "cannot be resolved.", code="task_model_mismatch")
    try:
        top_k = int(task_inputs.get("top_k") or 3)
    except (TypeError, ValueError):
        top_k = 3
    top_k = max(1, min(top_k, 20))
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt").to(result.model.device)
    with torch.no_grad():
        logits = result.model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    lines = ["=== Classification ===", f"Model: {model_id}"]
    for text, row in zip(texts, probs.tolist()):
        ranked = sorted(enumerate(row), key=lambda kv: kv[1], reverse=True)[:top_k]
        best_idx, best_score = ranked[0]
        label = id2label.get(best_idx, str(best_idx)) if isinstance(id2label, dict) \
            else (id2label[best_idx] if best_idx < len(id2label) else str(best_idx))
        lines.append(f"\nText: {text[:120]}")
        lines.append(f"Predicted: {label} ({best_score:.2%})")
        lines.append("Top predictions:")
        for idx, score in ranked:
            lbl = id2label.get(idx, str(idx)) if isinstance(id2label, dict) \
                else (id2label[idx] if idx < len(id2label) else str(idx))
            lines.append(f"  {lbl}: {score:.4f}")
    return "\n".join(lines)


def run_question_answering(result, model_id, prompt, task_inputs):
    """QA: extractive (QuestionAnswering heads) or generative (causal/seq2seq).
    Question and context are separate required fields."""
    question = _require_field(task_inputs, "question", "question-answering")
    context = _require_field(task_inputs, "context", "question-answering")
    tok = _tokenizer_for(result)
    cls_name = result.model.__class__.__name__
    if "QuestionAnswering" in cls_name:
        inputs = tok(question, context, return_tensors="pt",
                     truncation=True).to(result.model.device)
        import torch
        with torch.no_grad():
            out = result.model(**inputs)
        start = int(out.start_logits.argmax())
        end = int(out.end_logits.argmax()) + 1
        input_ids = inputs["input_ids"][0]
        answer = tok.decode(input_ids[start:end], skip_special_tokens=True).strip()
        mode = "extractive"
    else:
        prompted = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        pin = tok(prompted, return_tensors="pt", truncation=True,
                  max_length=1024).to(result.model.device)
        answer = _generate_from_inputs(result, pin, max_new_tokens=128)
        mode = "generative"
    if not answer:
        answer = "(no answer found in the provided context)"
    return (f"=== Question answering ({mode}) ===\n"
            f"Model: {model_id}\n\nQ: {question}\nA: {answer}")


def run_translation(result, model_id, prompt, task_inputs):
    """Translation: seq2seq generation; language-code forcing applied only
    when the tokenizer actually supports it (M2M100/NLLB style) — otherwise
    the source text is used verbatim and the limitation is stated."""
    source = (prompt or "").strip()
    if not source:
        raise ClaroBackendError(
            "Translation needs the source text (input_text = \"...\").",
            code="invalid_input")
    target = task_inputs.get("target_language")
    tok = _tokenizer_for(result)
    inputs = tok(source, return_tensors="pt", truncation=True,
                 max_length=1024).to(result.model.device)
    gen_extra = {}
    note = None
    if target:
        target = str(target).strip()
        lang_id = None
        codes = getattr(tok, "lang_code_to_id", None)
        if codes and target in codes:
            lang_id = codes[target]
        toks = getattr(tok, "lang_code_to_token", None)
        if lang_id is None and toks and target in toks:
            lang_id = toks[target]
        if lang_id is not None:
            gen_extra["forced_bos_token_id"] = lang_id
        else:
            note = (f"target language '{target}' noted — this tokenizer does "
                    "not expose language codes, so the model's own "
                    "translation convention applies")
    translated = _generate_from_inputs(result, inputs, max_new_tokens=512,
                                       num_beams=4, **gen_extra)
    lines = ["=== Translation ===", f"Model: {model_id}"]
    if target:
        lines.append(f"Target language: {target}")
    if note:
        lines.append(f"Note: {note}")
    lines.append(f"\n{translated}")
    return "\n".join(lines)


def run_embeddings_task(result, model_id, prompt, task_inputs):
    """Feature extraction: architecture-aware pooling + L2-normalized
    vectors (one per input line). See the embedding section above for the
    Qwen3 last-token / encoder mean-pooling strategies."""
    return _embedding_inference(result, model_id, prompt, "feature-extraction")


def run_sentence_similarity(result, model_id, prompt, task_inputs):
    """Sentence similarity: exactly two texts, two embeddings, cosine.
    Zero/degenerate vectors yield a 0.0 score with an explicit note."""
    text1 = task_inputs.get("text1")
    text2 = task_inputs.get("text2")
    if (not text1 or not text2) and prompt:
        lines = _split_lines(prompt)
        if len(lines) >= 2:
            text1, text2 = lines[0], lines[1]
    text1 = _require_field({"text1": text1}, "text1", "sentence-similarity")
    text2 = _require_field({"text2": text2}, "text2", "sentence-similarity")
    vectors, pooling = _compute_embeddings(result, [text1, text2],
                                           internal_task="sentence_similarity")
    sim = _cosine_similarity(vectors[0], vectors[1])
    notes = []
    for name, v in (("text 1", vectors[0]), ("text 2", vectors[1])):
        if sum(x * x for x in v) == 0.0:
            notes.append(f"{name} produced a zero (degenerate) embedding; "
                         "similarity is reported as 0.0")
    preview1 = ", ".join(f"{x:.4f}" for x in vectors[0][:4])
    preview2 = ", ".join(f"{x:.4f}" for x in vectors[1][:4])
    lines = [
        "=== Sentence similarity ===",
        f"Model: {model_id}",
        f"Pooling: {pooling}",
        f"Text 1: {text1[:120]}",
        f"Text 2: {text2[:120]}",
        f"Embedding 1 (first 4): [{preview1}]",
        f"Embedding 2 (first 4): [{preview2}]",
        f"Cosine similarity: {sim:.4f}",
    ]
    for n in notes:
        lines.append(f"Note: {n}")
    return "\n".join(lines)


def run_text_generation_task(result, model_id, prompt, task_inputs, **gen_opts):
    """Text generation (causal or seq2seq): the user's actual prompt with the
    existing chat-template / raw-format handling preserved."""
    text, _ = _generate_text(result, prompt, gen_opts.get("max_new_tokens", 100),
                             gen_opts.get("temperature", 0.7),
                             gen_opts.get("top_p", 0.9),
                             do_sample=gen_opts.get("do_sample", True),
                             prompt_format=gen_opts.get("prompt_format", "auto"))
    return text


# Extensible HF task → handler map. Adding a task = adding an entry (plus a
# handler function), not a new if-chain branch.
TASK_HANDLERS = {
    "text-generation": run_text_generation_task,
    "text2text-generation": run_text_generation_task,
    "conversational": run_text_generation_task,
    "chat": run_text_generation_task,
    "summarization": run_summarization,
    "translation": run_translation,
    "question-answering": run_question_answering,
    "text-classification": run_classification,
    "feature-extraction": run_embeddings_task,
    "sentence-similarity": run_sentence_similarity,
}

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
    prompt_format: str = "auto",
    task_inputs: Optional[dict] = None,
) -> str:
    """Run inference on GPU. Model is loaded via load_model() which uses CPU cache.
    
    This function is decorated with @spaces.GPU in app.py with dynamic duration.
    Task routing is driven by the model's intended task: each supported task
    has its own handler in TASK_HANDLERS, and unsupported tasks raise a clear
    error instead of falling back to generic text generation.
    """
    task = (task or "text-generation").strip().lower()
    task_inputs = _parse_task_inputs(task_inputs)
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
            if result.modality == "diffusion" and task in TASK_HANDLERS:
                raise ClaroBackendError(
                    f"Task '{task}' is not compatible with a Diffusion pipeline.",
                    code="task_model_mismatch",
                )

            handler = TASK_HANDLERS.get(task)
            if handler is not None:
                # Task-aware routing: the handler owns input preparation,
                # inference, and output interpretation for its task.
                if handler is run_text_generation_task:
                    text = handler(result, model_id, prompt, task_inputs,
                                   max_new_tokens=max_new_tokens,
                                   temperature=temperature, top_p=top_p,
                                   do_sample=do_sample, prompt_format=prompt_format)
                else:
                    text = handler(result, model_id, prompt, task_inputs)

            elif task in ("text-to-image", "image-generation"):
                if result.modality != "diffusion":
                    raise ClaroBackendError(
                        f"Task '{task}' requires a Diffusion pipeline, but {model_id} "
                        f"loaded as {result.loaded_class}.",
                        code="task_model_mismatch",
                    )
                text = _generate_text_to_image(result, prompt, max_new_tokens)

            else:
                # No silent generic fallback: unknown/unsupported tasks must
                # say so instead of dumping raw model output objects.
                raise ClaroBackendError(
                    f"Task '{task}' is not supported by this backend. "
                    f"Supported text tasks: {SUPPORTED_TEXT_TASKS}.",
                    code="task_not_supported",
                )
        finally:
            _offload_gpu(result)

    elapsed = time.time() - start_time
    print(f"[CLARO] INFER model={model_id} task={task} max_new_tokens={max_new_tokens} "
          f"modality={result.modality} loaded_class={result.loaded_class} "
          f"gpu_size=default cache={'hit' if cache_hit else 'miss'} "
          f"move_in={move_in:.2f}s exec={elapsed:.2f}s estimated={estimated_duration}s "
          f"tokens={n_tokens} cache_size={len(_CACHE)}")
    return text