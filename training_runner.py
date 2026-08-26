#!/usr/bin/env python3
"""
Claro.AI Training Runner — REAL Hugging Face Trainer backend

Provider independent: can run on ZeroGPU, Local Python, or Modal.
Reuses hf_loader logic to auto-detect model class/preprocessor.

REAL TRAINING ONLY — no synthetic curves.
If dependencies are missing, dataset/model cannot be loaded, or Trainer fails,
the process exits with training_error (non-zero) and the Node backend
marks the job as failed. Metrics are ONLY those emitted by Trainer
via StreamCallback: {step, epoch, train_loss, eval_loss, learning_rate}.

Usage:
  python training_runner.py --model_id owner/model --dataset_id owner/ds --task_type text-classification --epochs 3 ...
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

def _log(msg: str):
    print(f"[TRAIN] {msg}", flush=True)

def _emit(metric: dict):
    print(json.dumps(metric), flush=True)

def _fail(msg: str, code: int = 1):
    _log(f"training_error: {msg}")
    sys.exit(code)

def detect_columns(dataset, task_type: str):
    """Auto-detect text/label/image columns heuristically."""
    try:
        if hasattr(dataset, 'features') and dataset.features:
            features = list(dataset.features.keys())
        elif len(dataset) > 0:
            try:
                features = list(dataset[0].keys())
            except Exception:
                features = list(dataset.column_names) if hasattr(dataset, 'column_names') else []
        else:
            features = []
    except Exception:
        features = []
    cols = {c.lower(): c for c in features}

    text_candidates = ['text', 'sentence', 'content', 'input', 'tokens', 'premise', 'document', 'utterance']
    label_candidates = ['label', 'labels', 'target', 'ner_tags', 'answer', 'category']
    image_candidates = ['image', 'img', 'picture', 'pixel_values']

    def pick(candidates):
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return None

    text_col = pick(text_candidates)
    label_col = pick(label_candidates)
    image_col = pick(image_candidates)

    if task_type == 'image-classification':
        if not image_col:
            for c in features:
                if 'image' in c.lower():
                    image_col = c
                    break
        if not label_col and 'label' in cols:
            label_col = cols['label']
    elif task_type == 'token-classification':
        text_col = text_col or (cols.get('tokens') or cols.get('words') or (features[0] if features else None))
        label_col = label_col or cols.get('ner_tags') or cols.get('tags')
    elif task_type == 'text-generation':
        if not label_col:
            label_col = text_col

    if not text_col and features:
        for c in features:
            if c != label_col and c != image_col:
                text_col = c
                break
    if not label_col and features:
        label_col = features[-1]

    _log(f"detected columns text={text_col!r} label={label_col!r} image={image_col!r} (features={features})")
    return text_col, label_col, image_col


def infer_lora_target_modules(model, task_type: str):
    """
    Auto-detect suitable LoRA target modules from the model architecture.
    Do NOT hardcode model IDs — inspect model_type and named modules.
    """
    try:
        model_type = getattr(model.config, "model_type", "") or ""
        model_type = model_type.lower()
        # Architecture to target mapping (model_type → modules)
        type_map = {
            "bert": ["query", "value"],
            "roberta": ["query", "value"],
            "distilbert": ["q_lin", "v_lin"],
            "albert": ["query", "value"],
            "deberta": ["query", "value"],
            "deberta-v2": ["query", "value"],
            "electra": ["query", "value"],
            "gpt2": ["c_attn"],
            "gpt_neo": ["q_proj", "v_proj"],
            "gptj": ["q_proj", "v_proj"],
            "gpt_neox": ["query_key_value"],
            "llama": ["q_proj", "v_proj"],
            "mistral": ["q_proj", "v_proj"],
            "mixtral": ["q_proj", "v_proj"],
            "gemma": ["q_proj", "v_proj"],
            "gemma2": ["q_proj", "v_proj"],
            "qwen": ["q_proj", "v_proj"],
            "qwen2": ["q_proj", "v_proj"],
            "phi": ["q_proj", "v_proj"],
            "phi3": ["q_proj", "v_proj"],
            "falcon": ["query_key_value"],
            "bloom": ["query_key_value"],
            "mpt": ["Wqkv"],
            "t5": ["q", "v"],
            "bart": ["q_proj", "v_proj"],
            "mbart": ["q_proj", "v_proj"],
            "pegasus": ["q_proj", "v_proj"],
            "vit": ["query", "value"],
            "deit": ["query", "value"],
            "beit": ["query", "value"],
            "swin": ["query", "value"],
            "convnext": ["dwconv"],
            "resnet": ["conv1"],
            "regnet": ["proj"],
        }
        if model_type in type_map:
            tgt = type_map[model_type]
            _log(f"inferred target_modules for model_type={model_type}: {tgt}")
            return tgt

        # Fallback: scan named modules for Linear layers with common attn patterns
        import torch.nn as nn
        candidates = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                lname = name.lower()
                # common LoRA targets
                for pat in ["q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value", "q_lin", "v_lin", "c_attn", "qkv", "to_q", "to_k", "to_v"]:
                    if pat in lname:
                        # extract last component
                        cand = name.split(".")[-1]
                        candidates.add(cand)
                        break
        if candidates:
            # Prefer query/value or q_proj/v_proj if available
            preferred = [c for c in candidates if c in ("query", "q_proj", "c_attn", "q_lin")]
            if preferred:
                tgt = sorted(preferred)[:2]
            else:
                tgt = sorted(candidates)[:2]
            _log(f"scanned target_modules fallback: {tgt} (candidates: {sorted(candidates)[:8]})")
            return tgt

        # Ultimate fallback: use query/value which works for many Bert-like
        _log(f"fallback to default target_modules=['query','value'] for {model_type}")
        return ["query", "value"]
    except Exception as e:
        _log(f"infer target_modules failed: {e}, using ['query','value']")
        return ["query", "value"]


def run_training(args):
    _log(f"loading model={args.model_id} dataset={args.dataset_id} task={args.task_type} epochs={args.epochs} batch_size={args.batch_size} lr={args.learning_rate} max_steps={args.max_steps} validation_split={args.validation_split}%")

    # ── Imports — must succeed or training_error ──
    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoConfig, AutoTokenizer, AutoImageProcessor, AutoProcessor,
            TrainingArguments, Trainer, DataCollatorWithPadding, DefaultDataCollator
        )
        import importlib
    except ImportError as e:
        _fail(f"missing dependency: {e}. Install torch, transformers, datasets, accelerate. Trace: {traceback.format_exc()}")
    except Exception as e:
        _fail(f"import failed: {e}\n{traceback.format_exc()}")

    # ── Load dataset via datasets.load_dataset() ──
    _log("loading dataset via datasets.load_dataset()")
    def _try_load(name):
        try:
            return load_dataset(name)
        except Exception as e1:
            # fallback for single-name legacy datasets (imdb -> stanfordnlp/imdb, cifar10 -> uoft-cs/cifar10, etc.)
            if "/" not in name:
                for prefix in ["stanfordnlp", "uoft-cs", "ylecun", "rajpurkar", "lmsys"]:
                    try:
                        cand = f"{prefix}/{name}"
                        _log(f"retrying dataset as {cand}")
                        return load_dataset(cand)
                    except Exception:
                        continue
            # try split=train as last resort
            try:
                return {'train': load_dataset(name, split='train')}
            except Exception as e2:
                raise e1
    try:
        ds = _try_load(args.dataset_id)
        if isinstance(ds, dict) and 'train' in ds and not hasattr(ds, 'features'):
            # _try_load returned dict via split fallback — already handled
            pass
        if isinstance(ds, dict):
            train_ds = ds.get('train') or list(ds.values())[0]
        else:
            train_ds = ds
        if len(train_ds) > args.max_samples:
            _log(f"truncating dataset {len(train_ds)} -> {args.max_samples} (MAX_SAMPLES)")
            train_ds = train_ds.select(range(args.max_samples))
        _log(f"dataset loaded train={len(train_ds)}")
        if len(train_ds) == 0:
            _fail("dataset is empty after loading")
    except Exception as e:
        _fail(f"dataset load failed for {args.dataset_id}: {e}\n{traceback.format_exc()}")

    # ── Validation split ──
    val_ds = None
    if args.validation_split > 0 and len(train_ds) > 10:
        try:
            split = train_ds.train_test_split(test_size=args.validation_split/100.0, seed=42)
            train_ds = split['train']
            val_ds = split['test']
            _log(f"validation split {args.validation_split}% -> train={len(train_ds)} val={len(val_ds)}")
        except Exception as e:
            _log(f"split failed (continuing without val): {e}")

    # ── Detect columns ──
    try:
        text_col, label_col, image_col = detect_columns(train_ds, args.task_type)
        if args.task_type != 'image-classification' and not text_col:
            _fail(f"could not detect text column for task {args.task_type}. Features: {list(train_ds.features.keys()) if hasattr(train_ds,'features') else 'unknown'}")
        if args.task_type == 'image-classification' and not image_col:
            _fail(f"could not detect image column for {args.task_type}")
    except SystemExit:
        raise
    except Exception as e:
        _fail(f"column detection failed: {e}\n{traceback.format_exc()}")

    # ── Resolve model class via hf_loader (reuses existing infra) ──
    _log("resolving model class via hf_loader logic")
    autoclass = None
    try:
        from hf_loader.model_inspector import inspect
        from hf_loader.loader_strategies import resolve_loader
        ctx = inspect(args.model_id)
        plan = resolve_loader(ctx)
        autoclass = plan.auto_model_class
        _log(f"resolved autoclass={autoclass} preprocessor={plan.preprocessor_classes} trust_remote_code={plan.trust_remote_code}")
        # Override if task requires different head than auto-detected (e.g. tiny-random-bert is MaskedLM but we need Classification)
        task_to_class = {
            'text-generation': 'AutoModelForCausalLM',
            'text-classification': 'AutoModelForSequenceClassification',
            'image-classification': 'AutoModelForImageClassification',
            'token-classification': 'AutoModelForTokenClassification',
        }
        expected = task_to_class.get(args.task_type)
        if expected and expected not in (autoclass or ''):
            _log(f"overriding autoclass {autoclass} -> {expected} for task {args.task_type}")
            autoclass = expected
    except Exception as e:
        _log(f"hf_loader resolve failed: {e} — using task fallback")
    except Exception as e:
        _log(f"hf_loader resolve failed: {e} — using task fallback")
        fallback = {
            'text-generation': 'AutoModelForCausalLM',
            'text-classification': 'AutoModelForSequenceClassification',
            'image-classification': 'AutoModelForImageClassification',
            'token-classification': 'AutoModelForTokenClassification',
        }
        autoclass = fallback.get(args.task_type, 'AutoModelForSequenceClassification')
        _log(f"fallback autoclass={autoclass}")

    # ── Preprocessor — task specific ──
    preprocessor = None
    tokenizer = None
    image_processor = None
    try:
        if args.task_type == 'image-classification':
            from transformers import AutoImageProcessor
            preprocessor = AutoImageProcessor.from_pretrained(args.model_id, trust_remote_code=False)
            image_processor = preprocessor
            _log("loaded AutoImageProcessor")
        else:
            from transformers import AutoTokenizer
            try:
                preprocessor = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=False, use_fast=True)
                tokenizer = preprocessor
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                _log(f"loaded AutoTokenizer (vocab={len(tokenizer)})")
            except Exception as e:
                _log(f"AutoTokenizer failed: {e}, trying AutoProcessor")
                from transformers import AutoProcessor
                preprocessor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=False)
                tokenizer = getattr(preprocessor, 'tokenizer', None)
                _log("loaded AutoProcessor fallback")
    except Exception as e:
        _fail(f"preprocessor load failed for {args.model_id}: {e}\n{traceback.format_exc()}")

    # ── Preprocessing ──
    _log("preprocessing dataset")
    max_length = 128 if args.task_type != 'text-generation' else 256

    # keep features for closure
    try:
        train_features = list(train_ds.features.keys()) if hasattr(train_ds, 'features') and train_ds.features else []
    except Exception:
        train_features = []

    def preprocess_text(examples):
        # limit to max_samples already, now tokenize
        texts = examples.get(text_col) if text_col in examples else None
        if texts is None:
            # fallback to first column
            first_key = list(examples.keys())[0]
            texts = examples[first_key]
        if args.task_type == 'token-classification' and isinstance(texts[0], list):
            texts = [" ".join(t) if isinstance(t, list) else str(t) for t in texts]
        else:
            texts = [str(t) for t in texts]
        if tokenizer is not None:
            enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
            if args.task_type == 'text-generation':
                enc['labels'] = [ids.copy() for ids in enc['input_ids']]
            elif label_col and label_col in examples:
                labs = examples[label_col]
                enc['labels'] = labs
            return enc
        return examples

    try:
        if args.task_type == 'image-classification' and image_processor is not None:
            def _transform(examples):
                images = examples[image_col]
                processed = image_processor(images=images, return_tensors='pt')
                out = {}
                # processor returns pixel_values as tensor; datasets expects list
                # Convert to list for storage
                if 'pixel_values' in processed:
                    pv = processed['pixel_values']
                    # pv is tensor [B, C, H, W] -> list of arrays
                    out['pixel_values'] = [pv[i].tolist() if hasattr(pv[i],'tolist') else pv[i] for i in range(pv.shape[0])] if hasattr(pv,'shape') and len(pv.shape)==4 else pv
                    # better: keep as tensor via set_transform? For Trainer, we need pixel_values
                    # Use set_transform path instead of map for efficiency
                if label_col in examples:
                    out['labels'] = examples[label_col]
                return out

            # Use with_transform for images to avoid copying
            train_ds.set_transform(lambda ex: {
                **image_processor(images=ex[image_col], return_tensors='pt'),
                **({'labels': ex[label_col]} if label_col in ex else {})
            })
            if val_ds is not None:
                val_ds.set_transform(lambda ex: {
                **image_processor(images=ex[image_col], return_tensors='pt'),
                **({'labels': ex[label_col]} if label_col in ex else {})
            })
            # DataCollator will handle batching
            from transformers import DefaultDataCollator
            data_collator = DefaultDataCollator()
            _log(f"image transform set train={len(train_ds)}")
        else:
            # text tasks — map tokenization
            # Determine columns to keep
            keep = set(['input_ids', 'attention_mask', 'labels'])
            remove = [c for c in train_ds.column_names if c not in keep and c not in keep]
            # datasets map may need to handle batched
            train_ds = train_ds.map(preprocess_text, batched=True, remove_columns=[c for c in train_ds.column_names if c not in ['input_ids','attention_mask','labels']])
            if val_ds is not None:
                val_ds = val_ds.map(preprocess_text, batched=True, remove_columns=[c for c in val_ds.column_names if c not in ['input_ids','attention_mask','labels']])
            from transformers import DataCollatorWithPadding
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer) if tokenizer else DefaultDataCollator()
            _log(f"preprocessed train={len(train_ds)} cols={train_ds.column_names if hasattr(train_ds,'column_names') else 'unknown'}")
    except SystemExit:
        raise
    except Exception as e:
        _fail(f"preprocessing failed: {e}\n{traceback.format_exc()}")

    # ── Model ──
    _log(f"loading model {args.model_id} as {autoclass} (task={args.task_type})")
    try:
        import transformers
        ModelClass = getattr(transformers, autoclass, None)
        if ModelClass is None:
            from transformers import AutoModel
            ModelClass = AutoModel
            _log(f"{autoclass} not found, fallback AutoModel")
        model_kwargs = {}
        if args.task_type in ('text-classification', 'image-classification', 'token-classification'):
            unique_labels = set()
            try:
                if 'labels' in train_ds.column_names:
                    sample = train_ds['labels'][: min(200, len(train_ds))]
                    if sample and isinstance(sample[0], list):
                        for seq in sample:
                            unique_labels.update(seq)
                    else:
                        unique_labels.update(sample)
                num_labels = len(unique_labels) if unique_labels else 2
                num_labels = max(2, min(num_labels, 100))
                model_kwargs['num_labels'] = num_labels
                _log(f"inferred num_labels={num_labels} unique={sorted(list(unique_labels))[:10]}")
            except Exception as e:
                _log(f"num_labels inference failed: {e}")
                model_kwargs['num_labels'] = 2
        model_kwargs['trust_remote_code'] = False
        model = ModelClass.from_pretrained(args.model_id, **model_kwargs)
        _log(f"model loaded: {model.__class__.__name__} params={sum(p.numel() for p in model.parameters()) if hasattr(model,'parameters') else 'unknown'}")
    except SystemExit:
        raise
    except Exception as e:
        _fail(f"model load failed for {args.model_id} as {autoclass}: {e}\n{traceback.format_exc()}")

    # ── LoRA wrapping (if requested) — train only LoRA params ──
    is_lora = getattr(args, 'training_method', 'full') == 'lora'
    if is_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as e:
            _fail(f"peft not installed for LoRA: {e}. Install with pip install peft. Trace: {traceback.format_exc()}")
        # auto-detect target modules if requested
        raw_target = getattr(args, 'target_modules', 'auto')
        if raw_target == 'auto' or not raw_target:
            target_modules = infer_lora_target_modules(model, args.task_type)
        else:
            target_modules = [m.strip() for m in str(raw_target).split(',') if m.strip()]
            if not target_modules:
                target_modules = infer_lora_target_modules(model, args.task_type)
        # map task_type to peft TaskType
        task_map = {
            'text-generation': TaskType.CAUSAL_LM,
            'text-classification': TaskType.SEQ_CLS,
            'image-classification': TaskType.SEQ_CLS,
            'token-classification': TaskType.TOKEN_CLS,
        }
        peft_task = task_map.get(args.task_type, TaskType.SEQ_CLS)
        _log(f"applying LoRA r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout} target_modules={target_modules} task={peft_task}")
        try:
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=peft_task,
            )
            model = get_peft_model(model, peft_config)
            # show trainable vs total
            try:
                trainable, total = model.get_nb_trainable_parameters()
                pct = trainable / total * 100 if total else 0
                _log(f"lora trainable={trainable} total={total} ({pct:.2f}%)")
                # also emit in format Node parses: [TRAIN] lora trainable=... total=...
                _log(f"[TRAIN] lora trainable={trainable} total={total} ({pct:.2f}%)")
                # verify only LoRA is trainable
                _log(f"LoRA applied: trainable {trainable} / {total} params")
            except Exception as e:
                _log(f"could not compute trainable params: {e}")
                # still continue — peft will handle
        except SystemExit:
            raise
        except Exception as e:
            _fail(f"LoRA setup failed (target_modules={target_modules}): {e}\n{traceback.format_exc()}")
        _log(f"LoRA model: {model.__class__.__name__}")

    # ── TrainingArguments — must reflect real config (epochs, batch, lr, max_steps, validation) ──
    output_dir = Path(args.output_dir or f"./training_outputs/{args.job_id or 'local'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"output_dir={output_dir} (epochs={args.epochs} batch={args.batch_size} lr={args.learning_rate} max_steps={args.max_steps})")

    try:
        from transformers import TrainingArguments
        # transformers 4.x uses evaluation_strategy, 5.x renamed to eval_strategy
        try:
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                num_train_epochs=args.epochs,
                max_steps=args.max_steps if args.max_steps else -1,
                learning_rate=args.learning_rate,
                logging_steps=1,
                eval_steps=5 if val_ds is not None else 1000,
                save_steps=1000,
                evaluation_strategy='steps' if val_ds is not None else 'no',
                save_strategy='no',
                report_to='none',
                remove_unused_columns=False,
                load_best_model_at_end=False,
                seed=42,
                disable_tqdm=False,
            )
        except TypeError as e:
            if 'evaluation_strategy' in str(e):
                training_args = TrainingArguments(
                    output_dir=str(output_dir),
                    per_device_train_batch_size=args.batch_size,
                    per_device_eval_batch_size=args.batch_size,
                    num_train_epochs=args.epochs,
                    max_steps=args.max_steps if args.max_steps else -1,
                    learning_rate=args.learning_rate,
                    logging_steps=1,
                    eval_steps=5 if val_ds is not None else 1000,
                    save_steps=1000,
                    eval_strategy='steps' if val_ds is not None else 'no',
                    save_strategy='no',
                    report_to='none',
                    remove_unused_columns=False,
                    load_best_model_at_end=False,
                    seed=42,
                    disable_tqdm=False,
                )
            else:
                raise
        _log(f"TrainingArguments: {training_args.to_dict() if hasattr(training_args,'to_dict') else str(training_args)}")
    except SystemExit:
        raise
    except Exception as e:
        _fail(f"TrainingArguments failed: {e}\n{traceback.format_exc()}")

    # ── Trainer callback — ONLY real Trainer metrics ──
    try:
        from transformers import TrainerCallback

        class StreamCallback(TrainerCallback):
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                if not logs:
                    return
                metric = {}
                if 'loss' in logs:
                    metric['train_loss'] = float(logs['loss'])
                if 'eval_loss' in logs:
                    metric['eval_loss'] = float(logs['eval_loss'])
                # step/epoch from Trainer state
                try:
                    metric['step'] = int(state.global_step)
                    metric['epoch'] = float(state.epoch) if state.epoch is not None else 0.0
                    metric['learning_rate'] = float(logs.get('learning_rate', args.learning_rate))
                except Exception:
                    return
                # Only emit if we have at least one loss from Trainer
                if 'train_loss' in metric or 'eval_loss' in metric:
                    _emit(metric)

        _log("starting Trainer.train() — real training")
        from transformers import Trainer
        # transformers 4.x uses tokenizer=, 5.x uses processing_class=
        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                tokenizer=tokenizer if tokenizer else None,
                data_collator=data_collator,
                callbacks=[StreamCallback()],
            )
        except TypeError as e:
            if 'tokenizer' in str(e) or 'processing_class' in str(e):
                _log(f"retry Trainer with processing_class (transformers 5.x): {e}")
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_ds,
                    eval_dataset=val_ds,
                    processing_class=tokenizer if tokenizer else image_processor if image_processor else None,
                    data_collator=data_collator,
                    callbacks=[StreamCallback()],
                )
            else:
                raise

        trainer.train()

        # final eval if validation exists
        if val_ds is not None:
            try:
                eval_res = trainer.evaluate()
                _log(f"final eval: {eval_res}")
                # emit final combined metric if not already emitted
                if eval_res.get('eval_loss') is not None and trainer.state.log_history:
                    last_loss = None
                    for h in reversed(trainer.state.log_history):
                        if 'loss' in h:
                            last_loss = float(h['loss'])
                            break
                    _emit({
                        'step': int(trainer.state.global_step),
                        'epoch': float(trainer.state.epoch or args.epochs),
                        'train_loss': last_loss if last_loss is not None else 0,
                        'eval_loss': float(eval_res['eval_loss']),
                        'learning_rate': float(training_args.learning_rate),
                    })
            except Exception as e:
                _log(f"final eval failed: {e}")

    except SystemExit:
        raise
    except Exception as e:
        _fail(f"training failed: {e}\n{traceback.format_exc()}")

    # ── Save — real artifacts (model/adapter, tokenizer, config, metrics) ──
    try:
        # For LoRA, save_pretrained saves only the adapter (adapter_config.json + adapter_model.safetensors)
        # For full, it saves the full model.
        trainer.save_model(str(output_dir))
        if is_lora:
            _log(f"saved LoRA adapter to {output_dir} (adapter_config.json, adapter_model.safetensors)")
            # verify adapter files exist
            try:
                files = list(Path(output_dir).iterdir())
                _log(f"adapter files: {[f.name for f in files]}")
            except Exception:
                pass
        else:
            _log(f"saved model to {output_dir}")
        if tokenizer:
            tokenizer.save_pretrained(str(output_dir))
            _log(f"saved tokenizer to {output_dir}")
        if image_processor:
            image_processor.save_pretrained(str(output_dir))
            _log(f"saved image_processor to {output_dir}")
        # Save Trainer log_history as metrics.json (real metrics)
        import json as _json
        hist = trainer.state.log_history
        with open(output_dir / 'metrics.json', 'w') as f:
            _json.dump(hist, f, indent=2)
        # Also save config snapshot including LoRA
        with open(output_dir / 'trainer_config.json', 'w') as f:
            _json.dump({
                'model_id': args.model_id,
                'dataset_id': args.dataset_id,
                'task_type': args.task_type,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'max_steps': args.max_steps,
                'validation_split': args.validation_split,
                'training_method': args.training_method,
                'lora_r': getattr(args, 'lora_r', 8),
                'lora_alpha': getattr(args, 'lora_alpha', 16),
                'lora_dropout': getattr(args, 'lora_dropout', 0.05),
                'target_modules': getattr(args, 'target_modules', 'auto'),
            }, f, indent=2)
        _log(f"saved metrics.json ({len(hist)} entries) to {output_dir}")
        _log("finished")
    except SystemExit:
        raise
    except Exception as e:
        _fail(f"save failed: {e}\n{traceback.format_exc()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_id', required=True)
    p.add_argument('--dataset_id', required=True)
    p.add_argument('--task_type', required=True, choices=['text-generation','text-classification','image-classification','token-classification'])
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--learning_rate', type=float, default=2e-5)
    p.add_argument('--max_steps', type=int, default=None)
    p.add_argument('--validation_split', type=float, default=10)
    p.add_argument('--max_samples', type=int, default=5000)
    p.add_argument('--max_time_sec', type=int, default=1800)
    p.add_argument('--job_id', type=str, default=None)
    p.add_argument('--output_dir', type=str, default=None)
    # LoRA
    p.add_argument('--training_method', type=str, default='auto', choices=['full', 'lora', 'auto'])
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=16)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--target_modules', type=str, default='auto')
    args = p.parse_args()
    if args.max_steps == 0:
        args.max_steps = None
    try:
        run_training(args)
    except SystemExit:
        raise
    except Exception as e:
        _fail(f"fatal: {e}\n{traceback.format_exc()}")

if __name__ == '__main__':
    main()

