#!/usr/bin/env python3
"""
Trained Inference Runner — loads a trained model (full or LoRA adapter) and runs inference.

Used by POST /api/inference/trained to serve inference for Training Cell outputs.

For LoRA:
  base_model_id + adapter_dir -> PeftModel -> inference
For Full:
  model_dir (training_outputs/<job_id>) -> AutoModel -> inference

Security: adapter_dir / model_dir are derived server-side from job_id, never from client.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

def _log(msg):
    print(f"[INFER] {msg}", file=sys.stderr, flush=True)

def _fail(msg, code=1):
    print(json.dumps({"error": msg, "code": "inference_error"}))
    sys.exit(code)

def load_trained_model(job_id, artifact_dir, training_method, base_model_id, task_type):
    """
    Load a trained model for inference.
    Returns (model, preprocessor, loaded_class)
    """
    artifact_path = Path(artifact_dir)
    if not artifact_path.exists():
        _fail(f"artifact_dir not found: {artifact_dir}")

    # Detect training method from job if not provided
    if training_method == "lora":
        # LoRA: need adapter_config.json and adapter_model.safetensors
        adapter_config = artifact_path / "adapter_config.json"
        adapter_model = artifact_path / "adapter_model.safetensors"
        if not adapter_config.exists():
            _fail(f"adapter_config.json not found in {artifact_dir}")
        if not adapter_model.exists():
            # try .bin
            adapter_model = artifact_path / "adapter_model.bin"
            if not adapter_model.exists():
                _fail(f"adapter_model.safetensors not found in {artifact_dir}")
        # Load base model via hf_loader logic, then PeftModel
        try:
            from transformers import AutoTokenizer, AutoConfig
            from peft import PeftModel
            import importlib
            # Use hf_loader to get the correct AutoModel class for base
            try:
                from hf_loader.model_inspector import inspect
                from hf_loader.loader_strategies import resolve_loader
                ctx = inspect(base_model_id)
                plan = resolve_loader(ctx)
                _log(f"base loader: {plan.auto_model_class} for {base_model_id}")
                # Load base model with correct class
                import transformers
                # Map task to model class
                task_to_class = {
                    "text-generation": "AutoModelForCausalLM",
                    "text-classification": "AutoModelForSequenceClassification",
                    "image-classification": "AutoModelForImageClassification",
                    "token-classification": "AutoModelForTokenClassification",
                }
                expected = task_to_class.get(task_type, plan.auto_model_class)
                ModelClass = getattr(transformers, expected, None)
                if ModelClass is None:
                    ModelClass = getattr(transformers, plan.auto_model_class, None)
                if ModelClass is None:
                    _fail(f"Cannot find model class {expected} or {plan.auto_model_class}")
                model_kwargs = {"trust_remote_code": plan.trust_remote_code}
                base_model = ModelClass.from_pretrained(base_model_id, **model_kwargs)
                _log(f"base model loaded: {base_model.__class__.__name__}")
            except Exception as e:
                _log(f"hf_loader failed, fallback to AutoModel: {e}")
                # Fallback: try AutoModelForSequenceClassification etc. directly
                from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForImageClassification
                if task_type == "text-classification":
                    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_id, trust_remote_code=False)
                elif task_type == "token-classification":
                    base_model = AutoModelForTokenClassification.from_pretrained(base_model_id, trust_remote_code=False)
                elif task_type == "image-classification":
                    base_model = AutoModelForImageClassification.from_pretrained(base_model_id, trust_remote_code=False)
                else:
                    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=False)

            # Load tokenizer from adapter dir or base
            try:
                from transformers import AutoTokenizer
                try:
                    tokenizer = AutoTokenizer.from_pretrained(str(artifact_path), trust_remote_code=False)
                    _log(f"tokenizer from adapter dir")
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=False)
                    _log(f"tokenizer from base {base_model_id}")
                if tokenizer.pad_token is None and tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception as e:
                _log(f"tokenizer load failed: {e}")
                tokenizer = None

            # Load adapter
            model = PeftModel.from_pretrained(base_model, str(artifact_path))
            _log(f"LoRA adapter loaded from {artifact_path}, active: {model.active_adapter}")
            model.eval()
            return model, tokenizer, model.__class__.__name__

        except ImportError as e:
            _fail(f"peft not installed: {e}")
        except Exception as e:
            _fail(f"LoRA load failed: {e}\n{traceback.format_exc()}")

    else:
        # Full: load from artifact_dir directly
        try:
            from transformers import AutoTokenizer, AutoConfig
            import importlib
            # Try to inspect the trained model dir
            try:
                from hf_loader.model_inspector import inspect
                from hf_loader.loader_strategies import resolve_loader
                ctx = inspect(str(artifact_path))
                plan = resolve_loader(ctx)
                _log(f"trained loader: {plan.auto_model_class} for {artifact_path}")
                import transformers
                ModelClass = getattr(transformers, plan.auto_model_class, None)
                if ModelClass is None:
                    ModelClass = getattr(transformers, "AutoModel", None)
                model = ModelClass.from_pretrained(str(artifact_path), trust_remote_code=plan.trust_remote_code)
            except Exception as e:
                _log(f"inspect failed, trying AutoModel: {e}")
                from transformers import AutoModel
                model = AutoModel.from_pretrained(str(artifact_path), trust_remote_code=False)

            # Tokenizer from artifact_dir
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(artifact_path), trust_remote_code=False)
                if tokenizer.pad_token is None and tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception as e:
                _log(f"tokenizer from artifact failed: {e}, trying base")
                try:
                    from transformers import AutoTokenizer
                    # try base
                    import json as js
                    job_cfg = json.loads((artifact_path / "job.json").read_text())
                    base_id = job_cfg.get("config", {}).get("model_id", "")
                    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=False)
                except Exception:
                    tokenizer = None

            model.eval()
            return model, tokenizer, model.__class__.__name__
        except Exception as e:
            _fail(f"Full model load failed: {e}\n{traceback.format_exc()}")

def run_inference(job_id, prompt, max_new_tokens=100, task_type="text-generation"):
    # This is called via CLI or via server endpoint
    # For now, we handle CLI via main()
    pass

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job_id', required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--max_new_tokens', type=int, default=100)
    p.add_argument('--task_type', type=str, default="text-generation")
    p.add_argument('--artifact_dir', type=str, default=None)
    p.add_argument('--training_method', type=str, default=None)
    p.add_argument('--base_model_id', type=str, default=None)
    args = p.parse_args()

    # Resolve artifact_dir server-side from job_id (security: never trust client path)
    # For CLI, allow explicit artifact_dir, but validate it
    if args.artifact_dir:
        # Validate that it's under training_outputs and matches job_id
        artifact_dir = Path(args.artifact_dir).resolve()
        expected = Path(f"training_outputs/{args.job_id}").resolve()
        # Allow if it's exactly the expected dir or under it
        if str(artifact_dir) != str(expected) and not str(artifact_dir).startswith(str(expected)):
            # Also allow absolute training_outputs path
            base = Path("training_outputs").resolve()
            if not str(artifact_dir).startswith(str(base)):
                _fail(f"Invalid artifact_dir: {args.artifact_dir}")
        artifact_path = artifact_dir
    else:
        # Derive from job_id
        artifact_path = Path(f"training_outputs/{args.job_id}").resolve()
        base = Path("training_outputs").resolve()
        if not str(artifact_path).startswith(str(base)):
            _fail(f"Invalid job_id: {args.job_id}")

    if not artifact_path.exists():
        _fail(f"Artifact dir not found: {artifact_path}")

    # Load job.json to get metadata if not provided
    job_json = artifact_path / "job.json"
    if job_json.exists():
        try:
            meta = json.loads(job_json.read_text())
            training_method = args.training_method or meta.get("config", {}).get("training_method", "full")
            base_model_id = args.base_model_id or meta.get("config", {}).get("model_id", "")
            task_type = args.task_type or meta.get("config", {}).get("task_type", "text-generation")
        except Exception:
            training_method = args.training_method or "full"
            base_model_id = args.base_model_id or ""
            task_type = args.task_type
    else:
        training_method = args.training_method or "full"
        base_model_id = args.base_model_id or ""
        task_type = args.task_type

    # Security: reject traversal
    if ".." in str(artifact_path) or not str(artifact_path).startswith(str(Path("training_outputs").resolve())):
        _fail("Invalid artifact path")

    # Load model
    model, tokenizer, loaded_class = load_trained_model(args.job_id, str(artifact_path), training_method, base_model_id, task_type)

    if tokenizer is None:
        _fail("No tokenizer found for inference")

    # Run inference based on task
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model.to(device)
    except Exception:
        pass

    prompt = args.prompt
    max_new_tokens = args.max_new_tokens

    try:
        if task_type in ("text-generation", "text2text-generation", "summarization", "translation", "conversational", "chat"):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            is_enc_dec = bool(getattr(model.config, "is_encoder_decoder", False))
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=getattr(tokenizer, "eos_token_id", None),
                )
            if is_enc_dec:
                new_ids = out[0]
            else:
                new_ids = out[0][inputs["input_ids"].shape[-1]:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            print(json.dumps({"output": text}))
        elif task_type == "text-classification":
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                idx = int(logits.argmax(-1).item())
                label = model.config.id2label.get(idx, str(idx)) if hasattr(model.config, "id2label") else str(idx)
                conf = float(probs[0, idx].item())
                print(json.dumps({"output": label, "label": label, "confidence": conf, "logits": logits[0].tolist()[:5]}))
        elif task_type == "token-classification":
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            with torch.no_grad():
                logits = model(**inputs).logits
                pred_ids = logits.argmax(-1)[0].tolist()
                labels = [model.config.id2label.get(pid, str(pid)) for pid in pred_ids]
                # filter O
                out = []
                for tok, lab in zip(tokens, labels):
                    if lab != "O":
                        out.append({"token": tok, "label": lab})
                print(json.dumps({"output": str(out), "entities": out}))
        elif task_type == "image-classification":
            _fail("Image classification inference not yet implemented for local path (need image input)")
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs)
                print(json.dumps({"output": str(out)[:1000]}))
    except Exception as e:
        _fail(f"Inference failed: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
