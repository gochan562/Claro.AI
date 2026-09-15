"""Claro.AI ZeroGPU Space — generic model execution backend.

This Space used to hard-code `HuggingFaceTB/SmolLM3-3B`.  It now uses the
incoming `model_id` (validated) and `task` to load and run any model that
Claro.AI's existing loader architecture supports:

    Hugging Face repo
    ├── GGUF                 → llama-cpp-python backend
    ├── Diffusers            → DiffusionPipeline
    └── Transformers         → AutoConfig architecture resolver

Architecture/format detection is the source of truth for the loading class
(see `backend.py`); `task` only chooses the *inference* behaviour.  Safe
caching means repeated calls for the same `model_id` do not reload during
the Space worker lifetime.  No arbitrary Python is executed — only
structured `model_id/task/inputs` requests are accepted.
"""
import json
import os
import traceback

import gradio as gr
import spaces

from backend import (
    ClaroBackendError,
    cache_stats,
    load_model,
    run_inference,
    estimate_gpu_duration,
)
from train_api import train as train_fn, cancel_train as cancel_train_fn, unpack_request as _unpack_request

# Defaults used when the local Claro.AI provider does not pass model_id/task
# (e.g. someone clicking Submit in the Space UI directly).
DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_TASK = "text-generation"

# Hard limit for max_new_tokens
MAX_NEW_TOKENS_LIMIT = 2048


def _startup_diagnostics():
    """Log runtime versions at Space startup. Versions only — no secrets,
    no model/data contents. Helps distinguish environment issues (torch/CUDA
    vs scheduler) from model issues when a GPU call fails."""
    try:
        import torch
        print(f"[CLARO] runtime torch={torch.__version__} cuda_build={torch.version.cuda}", flush=True)
    except Exception as e:
        print(f"[CLARO] runtime torch import failed: {e}", flush=True)
    try:
        import transformers
        print(f"[CLARO] runtime transformers={transformers.__version__}", flush=True)
    except Exception as e:
        print(f"[CLARO] runtime transformers import failed: {e}", flush=True)
    try:
        # Import-only check (no model download): proves the exact chain from
        # "AutoImageProcessor requires the Torchvision library" is present.
        import torchvision
        print(f"[CLARO] runtime torchvision={torchvision.__version__}", flush=True)
    except Exception as e:
        print(f"[CLARO] runtime torchvision MISSING ({e}) — image-classification will fail", flush=True)
    try:
        from transformers import AutoImageProcessor  # noqa: F401
        print("[CLARO] runtime AutoImageProcessor import ok", flush=True)
    except Exception as e:
        print(f"[CLARO] runtime AutoImageProcessor import failed: {e}", flush=True)
    try:
        import PIL
        print(f"[CLARO] runtime pillow={PIL.__version__}", flush=True)
    except Exception as e:
        print(f"[CLARO] runtime pillow import failed: {e}", flush=True)
    try:
        import spaces
        print(f"[CLARO] runtime spaces={getattr(spaces, '__version__', 'unknown')}", flush=True)
    except Exception as e:
        print(f"[CLARO] runtime spaces import failed: {e}", flush=True)


_startup_diagnostics()


def _parse_structured(prompt_or_json: str):
    """Allow `prompt` to be either a plain string or a JSON payload of the
    form {"model_id": ..., "task": ..., "prompt": ..., "max_new_tokens": ...}.
    This keeps the Gradio endpoint shape (two inputs) unchanged while letting
    the local provider send a single structured blob.
    """
    if not isinstance(prompt_or_json, str):
        return prompt_or_json, None, None, None, None, None
    s = prompt_or_json.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return (
                    obj.get("prompt"),
                    obj.get("model_id"),
                    obj.get("task"),
                    obj.get("max_new_tokens"),
                    obj.get("do_sample"),
                    obj.get("adapter_path") or obj.get("adapter_dir"),
                )
        except Exception:
            pass
    return prompt_or_json, None, None, None, None, None


def _gpu_duration(prompt, max_new_tokens, task, model_id, *_args, **_kwargs):
    """Callable duration for @spaces.GPU.

    NOTE: spaces invokes this with the SAME positional args as the wrapped
    function, so the signature must mirror _gpu_infer's leading parameters
    (prompt first) — otherwise every value binds to the wrong name.
    ZeroGPU bills actual compute time, but `duration` is the kill-switch window.
    We size it to the workload (task + max_new_tokens + model size hint) instead
    of the one-size-fits-all default, so a 15-token request never reserves the
    same window as a 2048-token one.  Over-long workloads are already rejected
    (bad_request) before this runs, so the estimate never under-allocates.
    """
    try:
        tokens = int(max_new_tokens or 100)
    except Exception:
        tokens = 100
    t = str(task or DEFAULT_TASK).lower()
    hint = (str(model_id or "")).split("/")[-1]
    return estimate_gpu_duration(t, tokens, "", hint)


@spaces.GPU(duration=_gpu_duration)
def _gpu_infer(prompt, max_new_tokens, task, model_id, gguf_file, revision, temperature, top_p, do_sample, adapter_path=None):
    """GPU-scoped inference: moves the CPU-cached model onto CUDA, runs the
    requested task, and offloads it again.  Model *loading* deliberately happens
    on the CPU worker (see generate()) so cold loads do not consume GPU seconds.
    """
    try:
        import torch
        print(f"[CLARO] gpu torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"[CLARO] gpu device={torch.cuda.current_device()} name={torch.cuda.get_device_name(0)}", flush=True)
    except Exception as diag_e:
        print(f"[CLARO] gpu diagnostics unavailable: {diag_e}", flush=True)
    return run_inference(
        model_id=model_id,
        task=task,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        gguf_file=gguf_file,
        revision=revision,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        adapter_path=adapter_path,
    )


def generate(prompt, max_new_tokens=100, model_id=None, task=None, gguf_file=None, revision=None, temperature=0.7, top_p=0.9, do_sample=True, adapter_path=None):
    """Structured endpoint.

    Args:
        prompt:           user prompt, OR a JSON string carrying the full
                          structured request {"model_id","task","prompt",
                          "max_new_tokens"} so the Replit provider can send a
                          single payload and reuse the same endpoint.
        max_new_tokens:   integer 1..2048 (hard limit)
        model_id:         Hugging Face repo id.  When None, falls back to
                          DEFAULT_MODEL_ID (SmolLM3-3B on the bare Space UI).
        task:             inference behaviour, e.g. "text-generation".
        gguf_file:        Optional specific GGUF file to load.
        revision:         Optional model revision for cache invalidation.
        temperature:      Sampling temperature (default 0.7).
        top_p:            Nucleus sampling top-p (default 0.9).

    Returns:
        str: generated text, or a structured error string of the form
             "[CLARO:<code>] <message>" if the model could not be loaded or
             the task is incompatible.  The local provider parses these codes.
    """
    # Unwrap an optional structured JSON payload passed through `prompt`.
    if model_id is None and task is None:
        p, mid, t, mnt, ds, ap = _parse_structured(prompt)
        if mid is not None:
            model_id = mid
            task = t
            if mnt is not None:
                max_new_tokens = mnt
            if ds is not None:
                do_sample = bool(ds)
            if ap is not None:
                adapter_path = ap
            prompt = p

    model_id = (model_id or DEFAULT_MODEL_ID).strip()
    task = (task or DEFAULT_TASK).strip().lower()
    # Validate adapter_path if provided — must not be traversal, must be safe
    if adapter_path:
        adapter_path = str(adapter_path).strip()
        # Reject traversal and absolute paths from untrusted client — only allow HF ids or training_outputs/... 
        if ".." in adapter_path or adapter_path.startswith("/") or adapter_path.startswith("\\"):
            return f"[CLARO:bad_request] invalid adapter_path"
        # For HF ids, must be owner/name; for local, must be under training_outputs
        if "/" in adapter_path and not adapter_path.startswith("training_outputs/"):
            # HF id — validate format
            if not __import__('re').match(r'^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$', adapter_path):
                # also allow training_outputs path
                if not adapter_path.startswith("training_outputs/"):
                    return f"[CLARO:bad_request] invalid adapter_path"
        elif "/" not in adapter_path and adapter_path:
            # single name not allowed for adapter
            return f"[CLARO:bad_request] invalid adapter_path"
    if not prompt or not isinstance(prompt, str):
        return f"[CLARO:bad_request] prompt is required"

    # Hard limit for max_new_tokens - reject instead of silent clamp
    try:
        max_new_tokens = int(max_new_tokens)
    except Exception:
        max_new_tokens = 100
    if max_new_tokens < 1:
        return f"[CLARO:bad_request] max_new_tokens must be >= 1"
    if max_new_tokens > MAX_NEW_TOKENS_LIMIT:
        return f"[CLARO:bad_request] max_new_tokens exceeds hard limit of {MAX_NEW_TOKENS_LIMIT}"

    # Validate temperature/top_p
    try:
        temperature = float(temperature)
        if not (0.0 <= temperature <= 2.0):
            temperature = 0.7
    except Exception:
        temperature = 0.7
    try:
        top_p = float(top_p)
        if not (0.0 <= top_p <= 1.0):
            top_p = 0.9
    except Exception:
        top_p = 0.9
    # Validate do_sample
    if isinstance(do_sample, str):
        do_sample = do_sample.strip().lower() not in ("false", "0", "no", "")
    else:
        do_sample = bool(do_sample)

    # Estimate duration for logging (actual quota based on real compute time)
    model_size_hint = model_id.split("/")[-1] if "/" in model_id else model_id
    estimated_duration = estimate_gpu_duration(task, max_new_tokens, "", model_size_hint)

    try:
        # 1) Model load happens HERE, on the CPU worker (no @spaces.GPU), so
        #    downloads/from_pretrained never consume GPU-seconds.  Repeated
        #    requests for the same model hit the LRU cache and cost ~0 time.
        load_model(model_id, gguf_file=gguf_file, revision=revision, adapter_path=adapter_path)
        # 2) GPU quota is consumed ONLY by the explicit inference call below,
        #    with a workload-sized duration window.
        text = _gpu_infer(
            prompt,
            max_new_tokens,
            task,
            model_id,
            gguf_file,
            revision,
            temperature,
            top_p,
            do_sample,
            adapter_path,
        )
        # Return the generated text verbatim on success — the local Claro.AI
        # provider returns it directly to the dashboard.  Debug info (model
        # loaded, cache stats) goes to stdout/logs instead.
        print(f"[CLARO] model={model_id} task={task} cache={cache_stats()} estimated_duration={estimated_duration}s")
        return text
    except ClaroBackendError as e:
        # Surface as a structured error string (NOT a Gradio exception) so the
        # local provider can parse the [CLARO:<code>] prefix into a typed code
        # without relying on Gradio's error-frame title.
        return str(e)
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        # Check for CUDA OOM in unexpected errors
        err_msg = str(e)
        if "CUDA out of memory" in err_msg or "out of memory" in err_msg.lower():
            return f"[CLARO:gpu_oom] GPU out of memory: {err_msg}"
        # Scheduler-side GPU denials (quota exhausted, runs limit, no
        # capacity) are NOT model errors — surface them under their own code
        # so the client can tell "no GPU available" apart from "model broken".
        low_msg = err_msg.lower()
        if ("runs limit" in low_msg or "quota" in low_msg or "credits" in low_msg
                or "too many requests" in low_msg or "no gpu" in low_msg
                or "gpu is not available" in low_msg):
            return (f"[CLARO:zerogpu_quota] ZeroGPU cannot provide a GPU right now: {err_msg.strip()[:300]} "
                    f"(check the Space owner's ZeroGPU quota/billing; the model itself was not the problem)")
        return f"[CLARO:model_runtime] unexpected error: {e}\n{tb}"


# Optional debugging endpoint to inspect the model cache from the Space UI.
def show_cache():
    stats = cache_stats()
    return json.dumps(stats, indent=2)


# ── Remote training endpoint ──────────────────────────────────────────────
# Runs the SAME training_runner.py Trainer logic as local training, but on
# the ZeroGPU worker.  Accepts ONE structured JSON argument (validated, no
# code).  It is a generator: intermediate yields stream as SSE `generating`
# frames (JSON log/metric/progress events) and the final return value arrives
# as the `complete` frame together with a zip of the output dir.
import os as _os

_TRAIN_GPU_MAX = max(120, int(_os.environ.get("CLARO_TRAIN_GPU_DURATION_MAX", "1800")))


def _train_gpu_duration(train_request_json=None, *args, **_kwargs):
    """Size the @spaces.GPU kill-switch window from the request.

    ~90s base (dataset/model download + init) + ~1.5s per expected optimizer
    step, clamped to [120, CLARO_TRAIN_GPU_DURATION_MAX].  This is a ceiling,
    not a reservation: ZeroGPU bills actual compute time.
    """
    obj = _unpack_request(train_request_json, args, _kwargs)
    try:
        obj = json.loads(obj) if isinstance(obj, str) else (obj or {})
    except Exception:
        obj = {}
    if not isinstance(obj, dict):
        obj = {}
    try:
        epochs = max(1, int(obj.get("epochs", 3)))
    except Exception:
        epochs = 3
    try:
        raw_max = obj.get("max_steps", obj.get("maxSteps"))
        max_steps = int(raw_max) if raw_max not in (None, "") else None
    except Exception:
        max_steps = None
    steps = max_steps if max_steps else epochs * 100
    return max(120, min(_TRAIN_GPU_MAX, int(90 + steps * 1.5)))


@spaces.GPU(duration=_train_gpu_duration)
def _gpu_train(train_request_json=None, *args, **kwargs):
    yield from train_fn(_unpack_request(train_request_json, args, kwargs))


TRAIN_REQUEST_EXAMPLE = json.dumps({
    "model_id": "distilbert-base-uncased",
    "dataset_id": "stanfordnlp/imdb",
    "task_type": "text-classification",
    "epochs": 2,
    "batch_size": 8,
    "learning_rate": 0.00002,
    "validation_split": 10,
    "max_samples": 5000,
    "max_steps": None,
    "training_method": "full",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": "auto",
    "job_id": "train_0123456789ab",
}, indent=2)

demo_infer = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt (or JSON: {model_id,task,prompt,max_new_tokens,do_sample})"),
        gr.Slider(minimum=1, maximum=MAX_NEW_TOKENS_LIMIT, value=100, step=1, label="Max new tokens"),
        gr.Textbox(label="Model id", value=DEFAULT_MODEL_ID),
        gr.Textbox(label="Task", value=DEFAULT_TASK),
        gr.Textbox(label="GGUF file (optional)", value=""),
        gr.Textbox(label="Revision (optional)", value=""),
        gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top-p"),
        gr.Checkbox(label="do_sample (greedy if unchecked)", value=True),
    ],
    outputs=gr.Textbox(label="Output"),
    title="Claro.AI GPU — generic backend",
    description=(
        "Loads any supported Hugging Face repo (Transformers / Diffusers / GGUF) "
        "using Claro.AI's architecture resolver, with safe in-worker caching. "
        "No arbitrary Python execution — only structured model/task/input."
    ),
)

train_demo = gr.Interface(
    fn=_gpu_train,
    inputs=[gr.Textbox(label="Training request (JSON)", value=TRAIN_REQUEST_EXAMPLE, lines=12)],
    outputs=[
        gr.Textbox(label="Event stream (last event)"),
        gr.File(label="Artifacts zip (available when finished)"),
    ],
    title="Claro.AI Training — ZeroGPU worker",
    description=(
        "Runs real Hugging Face Trainer fine-tuning (full or LoRA) on ZeroGPU "
        "from a structured JSON request. No arbitrary code is accepted. "
        "Progress streams as JSON events; finished jobs return an artifacts zip."
    ),
    api_name="train",
)

cancel_demo = gr.Interface(
    fn=cancel_train_fn,
    inputs=[gr.Textbox(label="Job id (train_...)")],
    outputs=[gr.Textbox(label="Result")],
    title="Cancel training job",
    description="Requests cooperative cancellation of a running remote training job.",
    api_name="cancel_train",
)

demo = gr.TabbedInterface(
    [demo_infer, train_demo, cancel_demo],
    ["Inference", "Training", "Cancel"],
    title="Claro.AI GPU — generic backend + trainer",
)

if __name__ == "__main__":
    demo.launch()