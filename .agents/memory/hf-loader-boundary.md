---
name: Hugging Face loader boundary
description: Durable design rule for generating Claro.AI Hugging Face notebook cells.
---

The model loader must inspect the repository at notebook runtime and choose the model interface from Hugging Face metadata. The selected pipeline task may shape only the second-cell inference template.

**Why:** Hugging Face model families continually add new architectures, and task-to-class mappings break when a model is used for a different but compatible inference task. Runtime metadata also lets pure Diffusers repositories bypass Transformers loading.

**How to apply:** Preserve separate loader and inference cells. Prefer `auto_map`, architecture suffixes, model-type interface hints, and encoder/decoder structure in that order; use processor metadata for tokenizer/processor selection and do not use the UI task as a loader hint.