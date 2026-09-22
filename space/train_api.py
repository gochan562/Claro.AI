"""Claro.AI ZeroGPU Space — remote training worker.

Exposes a single structured training endpoint (no arbitrary code):

    train(train_request_json) -> generator yielding (event_json, artifacts_file)

The function runs the *same* Hugging Face Trainer logic as the local
``training_runner.py`` (imported, not duplicated) inside a worker thread,
while the generator streams captured stdout lines back as JSON events:

    {"type": "log",      "line": "[TRAIN] loading model=..."}
    {"type": "metric",   "metric": {"step": 12, "epoch": 0.4, "train_loss": ...}}
    {"type": "progress", "progress": {"total_steps": 1250, ...}}
    {"type": "status",   "status": "training"}

The final yielded tuple carries a manifest plus a zip of the output dir:

    (manifest_json, "/tmp/claro_train_<job_id>.zip")

Gradio delivers intermediate yields as SSE ``generating`` frames and the final
value as the ``complete`` frame, which is exactly what the Claro.AI server
consumes.  A cooperative ``cancel_train(job_id)`` endpoint lets the server
request cancellation between yields.

Uploaded to the Space together with ``training_runner.py`` and the
``hf_loader/`` package (see ``push_to_hf.py``).
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import traceback
import zipfile
from pathlib import Path

# Cooperative cancellation flags, set by cancel_train().
_CANCELLED: set[str] = set()
_CANCEL_LOCK = threading.Lock()


def request_cancel(job_id: str) -> bool:
    with _CANCEL_LOCK:
        _CANCELLED.add(str(job_id))
    return True


def _is_cancelled(job_id: str) -> bool:
    with _CANCEL_LOCK:
        return str(job_id) in _CANCELLED


def _clear_cancel(job_id: str) -> None:
    with _CANCEL_LOCK:
        _CANCELLED.discard(str(job_id))


# ── GPU duration sizing (used by @spaces.GPU(duration=...) in app.py) ──
# Kill-switch window only — ZeroGPU bills actual compute, and the Trainer
# still stops exactly at max_steps/epochs. Two ceilings apply:
#   CLARO_TRAIN_GPU_DURATION_MAX       soft ceiling for the estimate (1800s)
#   CLARO_TRAIN_GPU_DURATION_HARD_MAX  hard bound for what we REQUEST from HF.
# The hard bound exists because Hugging Face rejects calls whose requested
# duration exceeds the account/tier maximum (e.g. "requested GPU duration
# (810s) is larger than the maximum allowed"). If HF rejects even a clamped
# request, read the true maximum from that error and set HARD_MAX at or
# below it in the Space's Settings → Variables (no code change needed).
_TRAIN_GPU_MAX = max(120, int(os.environ.get("CLARO_TRAIN_GPU_DURATION_MAX", "1800")))
_TRAIN_GPU_HARD_MAX = max(120, int(os.environ.get("CLARO_TRAIN_GPU_DURATION_HARD_MAX", "600")))


# Keys that identify a flat training config (the exact runtime shape carries
# model_id, dataset_id, task_type, epochs, batch_size, learning_rate,
# validation_split, max_samples, max_steps, training_method, LoRA fields,
# job_id — plus camelCase aliases accepted by the backend validator).
_DURATION_CONFIG_KEYS = frozenset({
    "model_id", "dataset_id", "task_type", "task",
    "epochs", "batch_size", "batchSize",
    "learning_rate", "learningRate",
    "validation_split", "validationSplit",
    "max_samples", "max_steps", "maxSteps",
    "training_method", "trainingMethod", "method",
    "lora_r", "lora_alpha", "lora_dropout", "target_modules",
    "job_id",
})


def _looks_like_config(d):
    return isinstance(d, dict) and any(k in d for k in _DURATION_CONFIG_KEYS)


def _deep_parse_json(v, _depth=0):
    """Parse a string payload into a value (bounded, side-effect free).

    Tries JSON first, then a Python-dict representation via ast.literal_eval
    (the runtime delivers the whole request as a single-quoted dict repr,
    which is not valid JSON). NEVER uses eval(). Returns the original value
    when nothing parses; callers only accept Mapping results.
    """
    cur = v
    for _ in range(3):
        if not isinstance(cur, str):
            break
        try:
            cur = json.loads(cur)
            continue
        except Exception:
            pass
        try:
            cur = ast.literal_eval(cur)
        except Exception:
            break
    return cur


def normalize_duration_request(raw):
    """Fold every observed packing into ONE flat config dict.

    Never mutates its input (only reads + shallow-copies). Shapes handled:
      * flat config {...} — the exact runtime request shape
      * JSON string of any of the below
      * {"train_request_json": {...}} envelope — the raw v2 body as seen by
        the duration callable (inner value may itself be a JSON string)
      * {"data": [{...}]} Gradio predict-style payload
    Returns {} when nothing config-like is found (caller falls back to
    epochs=3 / max_steps=None, exactly as before).
    """
    cur = _deep_parse_json(raw)
    for _ in range(4):  # bounded: envelope nesting is never deep
        if isinstance(cur, (list, tuple)):
            for item in cur:
                item = _deep_parse_json(item)
                if _looks_like_config(item):
                    return dict(item)
            return {}
        if not isinstance(cur, dict):
            return {}
        if _looks_like_config(cur):
            return dict(cur)
        # Envelope: scan one level deeper for a config-like value.
        nxt = None
        for value in cur.values():
            cand = _deep_parse_json(value)
            if isinstance(cand, (list, tuple)):
                for item in cand:
                    item = _deep_parse_json(item)
                    if _looks_like_config(item):
                        return dict(item)
                continue
            if _looks_like_config(cand):
                return dict(cand)
            if nxt is None and isinstance(cand, dict) and cand:
                nxt = cand
        if nxt is None:
            return {}
        cur = nxt
    return {}


def _train_gpu_duration(train_request_json=None, *args, **_kwargs):
    """Size the @spaces.GPU kill-switch window from the request.

    ~90s base (dataset/model download + init) + ~1.5s per expected optimizer
    step, clamped to [120, HARD_MAX]. Incoming args are normalized FIRST into
    one flat config via normalize_duration_request (v2 gateway passes kwargs,
    Gradio UI passes positionally, raw bodies arrive enveloped).
    """
    # DIAGNOSTIC: exact dynamic inputs on every invocation (named value).
    try:
        _in = repr(train_request_json)
        if args:
            _in += f" +args[{len(args)}]"
        if _kwargs:
            _in += f" +kwargs{sorted(_kwargs.keys())}"
        print(f"[CLARO-DURATION] dynamic inputs={_in[:600]}", flush=True)
    except Exception as _e:
        print(f"[CLARO-DURATION] dynamic inputs=<unprintable: {_e}>", flush=True)
    # DIAGNOSTIC: raw call shape BEFORE normalization (proves whether the
    # request arrives as Mapping, JSON string, or Python-dict repr string).
    try:
        print(f"[CLARO-DURATION] raw args type={type(args).__name__}", flush=True)
        print(f"[CLARO-DURATION] raw args repr={repr(args)[:600]}", flush=True)
        print(f"[CLARO-DURATION] kwargs={sorted(_kwargs.keys()) if _kwargs else []}", flush=True)
        if args:
            print(f"[CLARO-DURATION] first positional type={type(args[0]).__name__}", flush=True)
            print(f"[CLARO-DURATION] first positional repr={repr(args[0])[:600]}", flush=True)
        else:
            print("[CLARO-DURATION] first positional type=<none>", flush=True)
            print("[CLARO-DURATION] first positional repr=<none>", flush=True)
    except Exception as _e:
        print(f"[CLARO-DURATION] raw shape inspect failed: {_e}", flush=True)
    # Normalize FIRST: one flat config dict from any packing shape.
    # (The v2 gateway hands this callable the raw body envelope, not the
    # parameter-mapped endpoint args — reading .get() off the raw object is
    # what silently fell back to epochs=3 / max_steps=None.)
    # Candidate scan, in historical priority order: an explicitly passed
    # value first, then each positional in order, then the kwargs mapping.
    # A single positional Mapping/dict is normalized DIRECTLY
    # (normalized = dict(args[0])). The first candidate yielding a non-empty
    # config wins; anything unusable is skipped, never fatal.
    _candidates = []
    if train_request_json is not None:
        _candidates.append(train_request_json)
    _candidates.extend(args)
    if _kwargs:
        _candidates.append(_kwargs)
    obj = {}
    for _cand in _candidates:
        try:
            _cfg = normalize_duration_request(_cand)
        except Exception:
            continue
        if _cfg:
            obj = _cfg
            break
    # TEMP-DIAG (item 10): normalized values BEFORE any calculation.
    try:
        print(f"[CLARO-DURATION] normalized keys={sorted(obj.keys())} "
              f"epochs={obj.get('epochs', '(default 3)')!r} "
              f"max_steps={obj.get('max_steps', obj.get('maxSteps', '(default None)'))!r}", flush=True)
    except Exception:
        pass
    try:
        epochs = max(1, int(obj.get("epochs", 3)))
    except Exception:
        epochs = 3
    raw_max = None
    try:
        raw_max = obj.get("max_steps", obj.get("maxSteps"))
        max_steps = int(raw_max) if raw_max not in (None, "") else None
    except Exception:
        max_steps = None
    steps = max_steps if max_steps else epochs * 100
    requested = int(90 + steps * 1.5)
    # HARD_MAX clamp RESTORED (diagnosis complete): never request above the
    # ZeroGPU allowance. Training semantics untouched — Trainer still stops
    # at exactly max_steps/epochs; this is only the GPU kill-switch window.
    final = max(120, min(_TRAIN_GPU_MAX, _TRAIN_GPU_HARD_MAX, requested))
    # TEMP-DIAG: full duration provenance for every scheduled call.
    print(f"[TRAIN-DIAG] duration requested max_steps={raw_max!r} epochs={epochs} "
          f"computed total_steps={steps} requested={requested}s "
          f"caps=[soft {_TRAIN_GPU_MAX}s, hard {_TRAIN_GPU_HARD_MAX}s] final={final}s",
          flush=True)
    # REQUIRED source-of-truth block: exact values feeding @spaces.GPU.
    try:
        _batch = obj.get("batch_size", obj.get("batchSize", "?"))
    except Exception:
        _batch = "?"
    print("[CLARO] train GPU duration:\n"
          f"requested_steps={raw_max!r}\n"
          f"computed_steps={steps}\n"
          f"computed_duration={requested}\n"
          f"final_spaces_gpu_duration={final}\n"
          f"max_steps={raw_max!r}\n"
          f"total_steps={steps}\n"
          f"epochs={epochs}\n"
          f"batch_size={_batch}\n"
          f"estimated_seconds_per_step=1.5\n"
          f"calculated_duration={requested}\n"
          f"caps_soft={_TRAIN_GPU_MAX}\n"
          f"caps_hard={_TRAIN_GPU_HARD_MAX}",
          flush=True)
    if final < requested:
        print(f"[TRAIN-DIAG] duration CLAMPED {requested}s -> {final}s by hard max "
              f"(training semantics unchanged: Trainer still stops at max_steps/epochs; "
              f"raise CLARO_TRAIN_GPU_DURATION_HARD_MAX only if HF allows more)", flush=True)
    # DIAGNOSTIC (item 4): exact value returned to the Spaces scheduler.
    print(f"[CLARO-DURATION] dynamic result={final}", flush=True)
    return final


def unpack_request(train_request_json=None, args=(), kwargs=None):
    """Extract the training request regardless of how the caller packed it.

    The ZeroGPU v2 API invokes endpoints as fn(**body), while the Gradio UI
    passes the single textbox value positionally. Accept all shapes:
      * fn(train_request_json="<json str | obj>")
      * fn(**{config kwargs...})
      * fn("<json str>" | {...})
    """
    if train_request_json is not None:
        return train_request_json
    kwargs = kwargs or {}
    if kwargs:
        return kwargs
    if args:
        first = args[0]
        if isinstance(first, (str, dict)):
            return first
    return None


# ── Request validation (mirrors training_backend.validateTrainingRequest) ──
_ID_RE = re.compile(r'^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)?$')
_JOB_RE = re.compile(r'^train_[a-f0-9]{12}$')
_TASKS = {'text-generation', 'text-classification', 'image-classification', 'token-classification'}
_METHODS = {'full', 'lora'}


def _fail(msg: str) -> dict:
    return {"type": "error", "error": msg}


def validate_train_request(obj: dict) -> tuple[dict | None, str | None]:
    """Returns (normalized_request, None) or (None, error_message)."""
    if not isinstance(obj, dict):
        return None, "request must be a JSON object"
    model_id = str(obj.get('model_id') or '').strip()
    dataset_id = str(obj.get('dataset_id') or '').strip()
    task_type = str(obj.get('task_type') or obj.get('task') or '').strip().lower()
    job_id = str(obj.get('job_id') or '').strip()
    if not model_id or not _ID_RE.match(model_id):
        return None, f"Invalid model_id: {model_id or '(empty)'}"
    if not dataset_id or not _ID_RE.match(dataset_id):
        return None, f"Invalid dataset_id: {dataset_id or '(empty)'}"
    if task_type not in _TASKS:
        return None, f"Invalid task_type: {task_type}. Allowed: {sorted(_TASKS)}"
    if not job_id or not _JOB_RE.match(job_id):
        return None, f"Invalid job_id: {job_id or '(empty)'}"

    def _int(name, val, lo, hi, default=None):
        if val is None or (isinstance(val, str) and val.strip() == ''):
            if default is None:
                return None, None
            return default, None
        try:
            n = int(val)
        except (TypeError, ValueError):
            return None, f"{name} must be an integer"
        if n < lo or n > hi:
            return None, f"{name} must be {lo}..{hi}"
        return n, None

    max_epochs = int(os.environ.get('CLARO_TRAIN_MAX_EPOCHS', '5'))
    max_batch = int(os.environ.get('CLARO_TRAIN_MAX_BATCH', '32'))
    max_steps_lim = int(os.environ.get('CLARO_TRAIN_MAX_STEPS', '10000'))
    epochs, err = _int('epochs', obj.get('epochs', 3), 1, max_epochs)
    if err:
        return None, err
    batch, err = _int('batch_size', obj.get('batch_size', obj.get('batchSize', 8)), 1, max_batch)
    if err:
        return None, err
    try:
        lr = float(obj.get('learning_rate', obj.get('learningRate', 2e-5)))
    except (TypeError, ValueError):
        return None, "learning_rate must be a number"
    if not (1e-6 <= lr <= 1e-2):
        return None, "learning_rate must be between 1e-6 and 1e-2"
    max_steps, err = _int('max_steps', obj.get('max_steps', obj.get('maxSteps')), 1, max_steps_lim)
    if err:
        return None, err
    try:
        val_split = float(obj.get('validation_split', obj.get('validationSplit', 10)))
    except (TypeError, ValueError):
        return None, "validation_split must be a number"
    if not (0 <= val_split <= 50):
        return None, "validation_split must be 0..50 (%)"
    try:
        max_samples = int(obj.get('max_samples', int(os.environ.get('CLARO_TRAIN_MAX_SAMPLES', '5000'))))
    except (TypeError, ValueError):
        return None, "max_samples must be an integer"
    if max_samples < 1:
        return None, "max_samples must be >= 1"
    try:
        max_time = int(obj.get('max_time_sec', int(os.environ.get('CLARO_TRAIN_MAX_TIME_SEC', '1800'))))
    except (TypeError, ValueError):
        return None, "max_time_sec must be an integer"
    if max_time < 60:
        return None, "max_time_sec must be >= 60"

    method = str(obj.get('training_method', obj.get('trainingMethod', 'full')) or 'full').strip().lower()
    if method == 'auto':
        method = 'full'  # server resolves auto before sending; default safe
    if method not in _METHODS:
        return None, f"training_method must be one of {sorted(_METHODS)}"
    try:
        lora_r = int(obj.get('lora_r', 8))
        lora_alpha = int(obj.get('lora_alpha', 16))
        lora_dropout = float(obj.get('lora_dropout', 0.05))
    except (TypeError, ValueError):
        return None, "LoRA parameters must be numbers"
    if method == 'lora':
        if not (1 <= lora_r <= 64):
            return None, "lora_r must be 1..64"
        if not (1 <= lora_alpha <= 128):
            return None, "lora_alpha must be 1..128"
        if not (0 <= lora_dropout <= 0.5):
            return None, "lora_dropout must be 0..0.5"
    target_modules = obj.get('target_modules', obj.get('targetModules', 'auto'))
    if isinstance(target_modules, list):
        target_modules = ','.join(str(s).strip() for s in target_modules if str(s).strip())
    target_modules = str(target_modules or 'auto').strip() or 'auto'
    if target_modules.lower() != 'auto':
        for m in target_modules.split(','):
            if not re.match(r'^[A-Za-z0-9_.]+$', m.strip()):
                return None, f"Invalid target_modules entry: {m.strip()}"

    return {
        'model_id': model_id,
        'dataset_id': dataset_id,
        'task_type': task_type,
        'job_id': job_id,
        'epochs': epochs,
        'batch_size': batch,
        'learning_rate': lr,
        'max_steps': max_steps,
        'validation_split': val_split,
        'max_samples': max_samples,
        'max_time_sec': max_time,
        'training_method': method,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'target_modules': target_modules,
    }, None


def _classify_line(line: str) -> dict | None:
    """Map one captured stdout line to a stream event (or None to drop)."""
    s = line.strip()
    if not s:
        return None
    if s.startswith('{') and '"step"' in s:
        try:
            m = json.loads(s)
        except Exception:
            return {"type": "log", "line": s}
        if isinstance(m.get('step'), (int, float)) and (
            isinstance(m.get('train_loss'), (int, float)) or isinstance(m.get('eval_loss'), (int, float))
        ):
            return {"type": "metric", "metric": m}
        return {"type": "log", "line": s}
    if s.startswith('[TRAIN] total_steps='):
        try:
            total = int(s.split('=', 1)[1].strip())
            return {"type": "progress", "progress": {"total_steps": total}}
        except Exception:
            return {"type": "log", "line": s}
    if s.startswith('[TRAIN]'):
        low = s.lower()
        if 'training_error' in low:
            return {"type": "status", "status": "failed", "line": s}
        if 'finished' in low and 'training_error' not in low:
            return {"type": "status", "status": "finished", "line": s}
        if 'evaluating' in low:
            return {"type": "status", "status": "evaluating", "line": s}
        if 'loading' in low:
            return {"type": "status", "status": "loading", "line": s}
        return {"type": "log", "line": s}
    return {"type": "log", "line": s}


_TQDM_RE = re.compile(r'\d+%')


def _sanitize_stderr(chunk: str) -> list[str]:
    """Split stderr into lines, dropping tqdm progress-bar noise but keeping real errors."""
    out: list[str] = []
    for part in chunk.replace('\r', '\n').split('\n'):
        s = part.strip()
        if not s:
            continue
        # tqdm bars look like " 45%|████ | 123/456 [00:01<...]" — drop those, keep the rest
        if _TQDM_RE.search(s) and ('it/s' in s or 's/it' in s or '|' in s):
            continue
        out.append(s)
    return out


class _QueueWriter(io.TextIOBase):
    def __init__(self, q: "queue.Queue[str]", tag: str):
        self._q = q
        self._tag = tag

    def write(self, data):
        if not data:
            return 0
        text = str(data)
        if self._tag == 'stderr':
            for line in _sanitize_stderr(text):
                self._q.put(('stderr', line))
        else:
            for line in text.split('\n'):
                if line.strip():
                    self._q.put(('stdout', line))
        return len(text)

    def flush(self):
        return None


def _zip_dir(src: Path, dest_zip: Path) -> list[str]:
    names: list[str] = []
    with zipfile.ZipFile(dest_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.rglob('*')):
            if p.is_file():
                arc = p.relative_to(src).as_posix()
                zf.write(p, arc)
                names.append(arc)
    return names


def _maybe_upload_to_hub(zip_path: Path, job_id: str, cfg: dict) -> str | None:
    """Best-effort persistent artifact copy. Only runs when the Space owner
    configured CLARO_ARTIFACT_REPO + HF_TOKEN. Returns repo path or None."""
    repo = (os.environ.get('CLARO_ARTIFACT_REPO') or '').strip()
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if not repo or not token:
        return None
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        dest = f"{job_id}/artifacts.zip"
        api.upload_file(
            path_or_fileobj=str(zip_path),
            path_in_repo=dest,
            repo_id=repo,
            repo_type='dataset',
        )
        return f"{repo}:{dest}"
    except Exception as e:
        print(f"[TRAIN] hub upload skipped/failed: {e}", flush=True)
        return None


def train_worker(cfg: dict, out_dir: Path, line_q: "queue.Queue", stop_flag: threading.Event) -> dict:
    """Run training_runner.run_training() with captured stdio.

    Returns a result dict; raises on fatal errors. Checks cooperative cancel.
    """
    import argparse

    # Import here so Space startup doesn't pay the cost until training runs.
    sys_path_added = False
    try:
        from training_runner import run_training  # noqa
    except ImportError:
        # When running from repo root during tests, training_runner lives one dir up.
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from training_runner import run_training  # noqa
        sys_path_added = True

    args = argparse.Namespace(
        model_id=cfg['model_id'],
        dataset_id=cfg['dataset_id'],
        task_type=cfg['task_type'],
        epochs=cfg['epochs'],
        batch_size=cfg['batch_size'],
        learning_rate=cfg['learning_rate'],
        max_steps=cfg['max_steps'],
        validation_split=cfg['validation_split'],
        max_samples=cfg['max_samples'],
        max_time_sec=cfg['max_time_sec'],
        training_method=cfg['training_method'],
        lora_r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'],
        target_modules=cfg['target_modules'],
        job_id=cfg['job_id'],
        output_dir=str(out_dir),
    )
    out_writer = _QueueWriter(line_q, 'stdout')
    err_writer = _QueueWriter(line_q, 'stderr')
    if _is_cancelled(cfg['job_id']):
        raise RuntimeError('cancelled by user before start')
    with contextlib.redirect_stdout(out_writer), contextlib.redirect_stderr(err_writer):
        run_training(args)
    if _is_cancelled(cfg['job_id']) or stop_flag.is_set():
        raise RuntimeError('cancelled by user')
    metrics_path = out_dir / 'metrics.json'
    metrics = []
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = []
    return {"metrics_count": len(metrics)}


def train(train_request_json: str):
    """Gradio endpoint (api_name="train"). Generator yielding (event_json, file).

    Each intermediate yield is (json_event_string, None); the final return is
    (manifest_json_string, zip_path_or_None).
    """
    # DIAGNOSTIC (items 2-3): immediately before endpoint work begins — the
    # exact max_steps/epochs the worker sees (compare with what the duration
    # callable saw; any mismatch means packing divergence between the two).
    try:
        _dbg = json.loads(train_request_json) if isinstance(train_request_json, str) else train_request_json
        if not isinstance(_dbg, dict):
            _dbg = {}
        print(f"[CLARO-DURATION] request max_steps={_dbg.get('max_steps', _dbg.get('maxSteps', '(absent)'))!r}", flush=True)
        _dbg_steps = _dbg.get("max_steps", _dbg.get("maxSteps"))
        try:
            _dbg_steps = int(_dbg_steps) if _dbg_steps not in (None, "") else None
        except Exception:
            _dbg_steps = None
        try:
            _dbg_epochs = max(1, int(_dbg.get("epochs", 3)))
        except Exception:
            _dbg_epochs = 3
        print(f"[CLARO-DURATION] total_steps={_dbg_steps if _dbg_steps else _dbg_epochs * 100}", flush=True)
        print(f"[CLARO-DURATION] computed_duration={int(90 + (_dbg_steps if _dbg_steps else _dbg_epochs * 100) * 1.5)}", flush=True)
    except Exception as _e:
        print(f"[CLARO-DURATION] endpoint-entry inspect failed: {_e}", flush=True)
    try:
        obj = json.loads(train_request_json) if isinstance(train_request_json, str) else train_request_json
    except Exception:
        manifest = {"job_id": None, "status": "failed", "error": "training_error",
                    "message": "request is not valid JSON"}
        return json.dumps(manifest), None

    cfg, err = validate_train_request(obj if isinstance(obj, dict) else {})
    if err:
        manifest = {"job_id": (obj.get('job_id') if isinstance(obj, dict) else None),
                    "status": "failed", "error": "training_error", "message": err}
        yield json.dumps({"type": "log", "line": f"[TRAIN] training_error: {err}"}), None
        return json.dumps(manifest), None

    job_id = cfg['job_id']
    _clear_cancel(job_id)
    work_root = Path(tempfile.gettempdir()) / f"claro_train_{job_id}"
    out_dir = work_root / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = Path(tempfile.gettempdir()) / f"claro_train_{job_id}.zip"
    if zip_path.exists():
        try:
            zip_path.unlink()
        except Exception:
            pass

    line_q: "queue.Queue" = queue.Queue()
    stop_flag = threading.Event()
    result: dict = {}
    worker_exc: list = []

    def _run():
        try:
            result.update(train_worker(cfg, out_dir, line_q, stop_flag))
        except BaseException as e:  # noqa - must capture SystemExit from _fail()
            worker_exc.append(e)

    yield json.dumps({"type": "status", "status": "loading",
                      "line": f"[TRAIN] loading model={cfg['model_id']} on ZeroGPU worker job={job_id}"}), None
    th = threading.Thread(target=_run, daemon=True)
    start = time.time()
    th.start()
    cancelled = False
    try:
        while th.is_alive() or not line_q.empty():
            if _is_cancelled(job_id):
                stop_flag.set()
                cancelled = True
                break
            try:
                kind, line = line_q.get(timeout=0.5)
            except queue.Empty:
                # watchdog: hard cap so we never exceed the GPU window silently
                if time.time() - start > cfg['max_time_sec'] + 120:
                    worker_exc.append(RuntimeError(
                        f"max training time {cfg['max_time_sec']}s exceeded on ZeroGPU worker"))
                    break
                continue
            if kind == 'stderr':
                yield json.dumps({"type": "log", "line": f"[stderr] {line}"}), None
            else:
                ev = _classify_line(line)
                if ev is not None:
                    yield json.dumps(ev), None
    finally:
        # Drain any lines the worker emitted at the very end.
        drained = 0
        while not line_q.empty() and drained < 200:
            try:
                kind, line = line_q.get_nowait()
            except queue.Empty:
                break
            drained += 1
            if kind == 'stderr':
                yield json.dumps({"type": "log", "line": f"[stderr] {line}"}), None
            else:
                ev = _classify_line(line)
                if ev is not None:
                    yield json.dumps(ev), None

    th.join(timeout=30)
    _clear_cancel(job_id)

    if cancelled:
        manifest = {"job_id": job_id, "status": "failed", "error": "stopped_by_user",
                    "message": "cancelled by user"}
        yield json.dumps({"type": "log", "line": "[TRAIN] cancelled by user"}), None
        return json.dumps(manifest), None

    if worker_exc:
        e = worker_exc[0]
        # training_runner signals fatal errors via SystemExit after printing training_error
        msg = '' if isinstance(e, SystemExit) else f": {e}"
        manifest = {"job_id": job_id, "status": "failed", "error": "training_error",
                    "message": f"ZeroGPU worker failed{msg}"}
        if not isinstance(e, SystemExit):
            yield json.dumps({"type": "log", "line": f"[TRAIN] training_error: worker exception{msg}"}), None
            traceback.print_exc()
        return json.dumps(manifest), None

    try:
        files = _zip_dir(out_dir, zip_path)
    except Exception as e:
        manifest = {"job_id": job_id, "status": "failed", "error": "training_error",
                    "message": f"artifact packaging failed: {e}"}
        return json.dumps(manifest), None

    hub_path = _maybe_upload_to_hub(zip_path, job_id, cfg)
    manifest = {"job_id": job_id, "status": "finished",
                "metrics_count": result.get('metrics_count', 0),
                "files": files, "hub_path": hub_path}
    yield json.dumps({"type": "log", "line": f"[TRAIN] finished artifacts={len(files)}"}), None
    return json.dumps(manifest), str(zip_path)


def cancel_train(job_id: str) -> str:
    """Gradio endpoint (api_name="cancel_train"). Cooperative cancel flag."""
    job_id = str(job_id or '').strip()
    if not job_id or not _JOB_RE.match(job_id):
        return json.dumps({"ok": False, "error": "invalid job_id"})
    request_cancel(job_id)
    return json.dumps({"ok": True, "job_id": job_id})
