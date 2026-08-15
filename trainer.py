import sys
import queue
import threading
import subprocess
import modal
from io import StringIO
from pydantic import BaseModel

# ── Shared image with GPU-ready packages ─────────────────────────────────────
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "datasets", "unsloth", "fastapi[standard]")
)

app    = modal.App("student-hf-trainer", image=image)
volume = modal.Volume.from_name("student-training-storage", create_if_missing=True)


# ── Schema ────────────────────────────────────────────────────────────────────
class TrainingRequest(BaseModel):
    model_id: str
    dataset_id: str
    epochs: int = 3
    learning_rate: float = 2e-4

class CodeRequest(BaseModel):
    code: str


# ── Shared execution helper ───────────────────────────────────────────────────
# Handles Jupyter-style shell escapes so `!pip install X` and `%pip install X`
# work inside notebook cells. Consecutive Python lines are grouped and exec'd
# together so variables, imports, and multi-line blocks persist correctly.
# stdout_w / stderr_w must support .write(str).
def _exec_cell(code: str, stdout_w, stderr_w, streaming: bool = False):
    ns = {"__name__": "__main__"}
    lines = code.split("\n")
    python_buf = []

    def flush_python():
        if not python_buf:
            return
        snippet = "\n".join(python_buf)
        python_buf.clear()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_w, stderr_w
        try:
            exec(compile(snippet, "<notebook>", "exec"), ns)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    def run_shell(cmd: str):
        stdout_w.write(f"$ {cmd}\n")
        if streaming:
            # Stream subprocess output line-by-line for live terminal feedback
            proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True
            )
            for line in proc.stdout:
                stdout_w.write(line)
            proc.wait()
        else:
            proc = subprocess.run(
                cmd, shell=True,
                capture_output=True, text=True
            )
            if proc.stdout:
                stdout_w.write(proc.stdout)
            if proc.stderr:
                stderr_w.write(proc.stderr)

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!"):
            flush_python()
            run_shell(stripped[1:].strip())
        elif stripped.startswith("%pip ") or stripped == "%pip":
            flush_python()
            run_shell("pip " + stripped[4:].strip())
        else:
            python_buf.append(line)

    flush_python()


# ── 1. Notebook cell execution (single JSON response) ─────────────────────────
# URL: https://<user>--student-hf-trainer-run-code.modal.run/
@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=300
)
@modal.fastapi_endpoint(method="POST")
def run_code(payload: CodeRequest):
    """Execute a notebook cell and return stdout/stderr as a single JSON response."""
    buf_out = StringIO()
    buf_err = StringIO()

    try:
        _exec_cell(payload.code, buf_out, buf_err, streaming=False)
        status = "success"
    except Exception:
        import traceback
        traceback.print_exc(file=buf_err)
        status = "error"

    out = buf_out.getvalue()
    err = buf_err.getvalue()
    return {
        "status": "error" if err else status,
        "output": out,
        "stderr": err,
    }


# ── 2. Notebook cell execution (live streaming) ────────────────────────────────
# URL: https://<user>--student-hf-trainer-run-code-stream.modal.run/?code=...
# Streams stdout/stderr to the caller as produced, powering the xterm.js console.
@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=600
)
@modal.fastapi_endpoint(method="GET")
def run_code_stream(code: str):
    from fastapi.responses import StreamingResponse

    def generate():
        line_queue = queue.Queue()

        class QueueWriter:
            def write(self, s):
                if s:
                    line_queue.put(s)
            def flush(self):
                pass

        def worker():
            try:
                _exec_cell(code, QueueWriter(), QueueWriter(), streaming=True)
            except Exception:
                import traceback
                line_queue.put(traceback.format_exc())
            finally:
                line_queue.put(None)  # sentinel

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            chunk = line_queue.get()
            if chunk is None:
                break
            yield chunk
        t.join(timeout=5)

    return StreamingResponse(generate(), media_type="text/plain")


# ── 3. Fine-tuning launcher ───────────────────────────────────────────────────
# URL: https://<user>--student-hf-trainer-start-training.modal.run/
@app.function(
    gpu="A10G",
    volumes={"/data": volume},
    timeout=3600
)
@modal.fastapi_endpoint(method="POST")
def start_training(payload: TrainingRequest):
    """Trigger a fine-tuning job from the platform."""
    print(f"Starting training for {payload.model_id} on {payload.dataset_id}…")

    # ── YOUR TRAINING CODE GOES HERE ─────────────────────────────────────────
    # Load model → Load dataset → Run trainer.train()
    # Save weights to /data or push to Hugging Face Hub
    # ─────────────────────────────────────────────────────────────────────────

    return {"status": "success", "message": f"Finished training {payload.model_id}!"}
