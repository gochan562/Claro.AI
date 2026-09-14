"""One-shot uploader: pushes the generic Claro.AI backend to the ZeroGPU Space.

Run with a *write-scoped* Hugging Face token (fine-grained tokens limited to
the Space work too, as long as they grant `repo.content.write` on
`Gochan562/claro_ai_gpu`).

Usage:
    HF_TOKEN=<write-token> /tmp/claro_venv/bin/python space/push_to_hf.py

You can also override the destination repo with `HF_SPACE_REPO`:
    HF_SPACE_REPO=Gochan562/claro_ai_gpu HF_TOKEN=... python space/push_to_hf.py [--create-pr]

The script uploads `space/app.py`, `space/backend.py`, `space/train_api.py`,
`training_runner.py`, the `hf_loader/` package and `space/requirements.txt`,
then prints the Space URL.  The Space rebuilds on ZeroGPU automatically; check
https://huggingface.co/spaces/Gochan562/claro_ai_gpu for build logs.
"""
from __future__ import annotations

import os
import sys

try:
    import huggingface_hub
except ImportError as e:
    raise RuntimeError(
        "Install huggingface_hub first:  pip install huggingface_hub"
    ) from e


HERE = os.path.dirname(os.path.abspath(__file__))
SPACE_REPO = os.environ.get("HF_SPACE_REPO", "Gochan562/claro_ai_gpu")
CREATE_PR = "--create-pr" in sys.argv
FILES = [
    ("app.py", "app.py"),
    ("backend.py", "backend.py"),
    ("train_api.py", "train_api.py"),
    ("training_runner.py", "training_runner.py"),
    ("requirements.txt", "requirements.txt"),
    ("test_backend.py", "test_backend.py"),
]
# Whole-package uploads (list of (local_dir, repo_dir))
FOLDERS = [
    ("hf_loader", "hf_loader"),
]


def _upload(api, local_path, repo_name, commit_message, repo_id):
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_name,
        repo_id=repo_id,
        repo_type="space",
        commit_message=commit_message,
        create_pr=CREATE_PR,
    )
    print(f"  - uploaded {repo_name} ({os.path.getsize(local_path)} bytes)")


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN is required (write-scoped).", file=sys.stderr)
        return 2
    api = huggingface_hub.HfApi(token=token)
    whoami = api.whoami()
    print(f"[*] authenticated as: {whoami.get('name')}")

    ok = True
    commit_message = (
        "Claro.AI: add ZeroGPU remote training endpoint "
        "(train_api.py reuses training_runner.py)"
    )
    for local_name, repo_name in FILES:
        # training_runner.py lives in the repo root, one dir up from space/
        if local_name == "training_runner.py":
            local_path = os.path.join(os.path.dirname(HERE), local_name)
        else:
            local_path = os.path.join(HERE, local_name)
        if not os.path.exists(local_path):
            ok = False
            print(f"  ! missing {local_path}, skipping", file=sys.stderr)
            continue
        try:
            _upload(api, local_path, repo_name, commit_message if repo_name == "app.py" else f"{commit_message} (add {repo_name})", SPACE_REPO)
        except Exception as e:
            ok = False
            print(f"  ! failed {repo_name}: {e}", file=sys.stderr)
    for local_dir, repo_dir in FOLDERS:
        local_path = os.path.join(os.path.dirname(HERE), local_dir)
        if not os.path.isdir(local_path):
            ok = False
            print(f"  ! missing dir {local_path}, skipping", file=sys.stderr)
            continue
        try:
            api.upload_folder(
                folder_path=local_path,
                path_in_repo=repo_dir,
                repo_id=SPACE_REPO,
                repo_type="space",
                commit_message=f"{commit_message} (add {repo_dir}/)",
                create_pr=CREATE_PR,
            )
            print(f"  - uploaded folder {repo_dir}/")
        except Exception as e:
            ok = False
            print(f"  ! failed folder {repo_dir}: {e}", file=sys.stderr)
    if ok:
        print(
            f"\n[+] pushed to https://huggingface.co/spaces/{SPACE_REPO}. "
            f"The Space rebuilds automatically; point /api/gpu-status at it "
            f"once it is 'connected'."
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
