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
    run_inference,
)

# Defaults used when the local Claro.AI provider does not pass model_id/task
# (e.g. someone clicking Submit in the Space UI directly).
DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_TASK = "text-generation"


def _parse_structured(prompt_or_json: str):
    """Allow `prompt` to be either a plain string or a JSON payload of the
    form {"model_id": ..., "task": ..., "prompt": ..., "max_new_tokens": ...}.
    This keeps the Gradio endpoint shape (two inputs) unchanged while letting
    the local provider send a single structured blob.
    """
    if not isinstance(prompt_or_json, str):
        return prompt_or_json, None, None, None
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
                )
        except Exception:
            pass
    return prompt_or_json, None, None, None


@spaces.GPU
def generate(prompt, max_new_tokens=100, model_id=None, task=None):
    """Structured endpoint.

    Args:
        prompt:           user prompt, OR a JSON string carrying the full
                          structured request {"model_id","task","prompt",
                          "max_new_tokens"} so the Replit provider can send a
                          single payload and reuse the same endpoint.
        max_new_tokens:   integer 1..500
        model_id:         Hugging Face repo id.  When None, falls back to
                          DEFAULT_MODEL_ID (SmolLM3-3B on the bare Space UI).
        task:             inference behaviour, e.g. "text-generation".

    Returns:
        str: generated text, or a structured error string of the form
             "[CLARO:<code>] <message>" if the model could not be loaded or
             the task is incompatible.  The local provider parses these codes.
    """
    # Unwrap an optional structured JSON payload passed through `prompt`.
    if model_id is None and task is None:
        p, mid, t, mnt = _parse_structured(prompt)
        if mid is not None:
            model_id = mid
            task = t
            if mnt is not None:
                max_new_tokens = mnt
            prompt = p

    model_id = (model_id or DEFAULT_MODEL_ID).strip()
    task = (task or DEFAULT_TASK).strip().lower()
    if not prompt or not isinstance(prompt, str):
        return f"[CLARO:bad_request] prompt is required"

    # Clamp the slider-equivalent knob to the Space's documented range.
    try:
        max_new_tokens = int(max_new_tokens)
    except Exception:
        max_new_tokens = 100
    if max_new_tokens < 1: max_new_tokens = 1
    if max_new_tokens > 500: max_new_tokens = 500

    try:
        text = run_inference(
            model_id=model_id,
            task=task,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        # Return the generated text verbatim on success — the local Claro.AI
        # provider returns it directly to the dashboard.  Debug info (model
        # loaded, cache stats) goes to stdout/logs instead.
        print(f"[CLARO] model={model_id} task={task} cache={cache_stats()}")
        return text
    except ClaroBackendError as e:
        # Surface as a structured error string (NOT a Gradio exception) so the
        # local provider can parse the [CLARO:<code>] prefix into a typed code
        # without relying on Gradio's error-frame title.
        return str(e)
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return f"[CLARO:model_runtime] unexpected error: {e}\n{tb}"


# Optional debugging endpoint to inspect the model cache from the Space UI.
def show_cache():
    stats = cache_stats()
    return json.dumps(stats, indent=2)


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt (or JSON: {model_id,task,prompt,max_new_tokens})"),
        gr.Slider(minimum=1, maximum=500, value=100, step=1, label="Max new tokens"),
        gr.Textbox(label="Model id", value=DEFAULT_MODEL_ID),
        gr.Textbox(label="Task", value=DEFAULT_TASK),
    ],
    outputs=gr.Textbox(label="Output"),
    title="Claro.AI GPU — generic backend",
    description=(
        "Loads any supported Hugging Face repo (Transformers / Diffusers / GGUF) "
        "using Claro.AI's architecture resolver, with safe in-worker caching. "
        "No arbitrary Python execution — only structured model/task/input."
    ),
)

if __name__ == "__main__":
    demo.launch()
