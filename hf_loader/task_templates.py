from __future__ import annotations
from textwrap import dedent

QA_TEMPLATE = dedent('''
"""
Task: Question Answering (unified)
Works for:
  - CausalLM (Llama, Qwen, Gemma, DeepSeek, MSA, …)
  - Seq2SeqLM (FLAN-T5, BART, M2M100, …)
  - Extractive QA (BERT-SQuAD, RoBERTa, DistilBERT)
The loader already exported the canonical `preprocessor` handle plus `model`,
so this template never guesses between tokenizer / processor.
"""
import torch

QUESTION  = "What is the capital of France?"
CONTEXT   = "France is a country in Western Europe. Its capital is Paris."

def _is_extractive(m):
    return any(s in type(m).__name__ for s in ("QuestionAnswering",))

# ---- Branch 1: extractive QA head (start/end logits) ---------------------
if _is_extractive(model):
    inputs = preprocessor(QUESTION, CONTEXT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    start, end = out.start_logits.argmax(-1), out.end_logits.argmax(-1)
    answer_ids = inputs["input_ids"][0, start:end + 1]
    print("Answer:", preprocessor.decode(answer_ids))

# ---- Branch 2: generative (CausalLM or Seq2SeqLM) -----------------------
else:
    prompt = (
        f"Context: {CONTEXT}\\n"
        f"Question: {QUESTION}\\n"
        f"Answer:"
    )
    inputs = preprocessor(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=getattr(preprocessor, "eos_token_id", None),
        )
    # Slice off the prompt for CausalLM; Seq2SeqLM returns only new tokens
    if hasattr(model.config, "is_encoder_decoder") and model.config.is_encoder_decoder:
        out_ids = gen[0]
    else:
        out_ids = gen[0][inputs["input_ids"].shape[-1]:]
    print("Answer:", preprocessor.decode(out_ids, skip_special_tokens=True).strip())
''').strip()


TEXT_GEN_TEMPLATE = dedent('''
"""Task: Text generation. Uses the loader-exported `preprocessor` and `model`."""
import torch
prompt = "Once upon a time"
inputs = preprocessor(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7,
                     pad_token_id=getattr(preprocessor, "eos_token_id", None))
new = out[0][inputs["input_ids"].shape[-1]:] if not getattr(model.config, "is_encoder_decoder", False) else out[0]
print(preprocessor.decode(new, skip_special_tokens=True))
''').strip()


SUMMARIZATION_TEMPLATE = dedent('''
"""Task: Summarization. Uses the loader-exported `preprocessor` and `model`."""
TEXT = "{{ SUMMARY_TEXT }}"
inputs = preprocessor(TEXT, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
print(preprocessor.decode(out[0], skip_special_tokens=True))
''').strip()


OCR_TEMPLATE = dedent('''
"""Task: OCR / image-to-text. Uses the loader-exported `preprocessor` and `model`."""
from PIL import Image
image = Image.open("{{ IMAGE_PATH }}").convert("RGB")
inputs = preprocessor(images=image, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512)
print(preprocessor.batch_decode(out, skip_special_tokens=True)[0])
''').strip()


TEXT_TO_IMAGE_TEMPLATE = dedent('''
"""Task: Text-to-image. Assumes `pipe` (DiffusionPipeline) loaded."""
image = pipe("{{ PROMPT }}", num_inference_steps=30).images[0]
image.save("out.png")
''').strip()

TEMPLATES = {
    "question-answering":   QA_TEMPLATE,
    "text-generation":      TEXT_GEN_TEMPLATE,
    "summarization":        SUMMARIZATION_TEMPLATE,
    "ocr":                  OCR_TEMPLATE,
    "image-to-text":        OCR_TEMPLATE,
    "text-to-image":        TEXT_TO_IMAGE_TEMPLATE,
}


def render_template(task: str, **substitutions) -> str:
    task = (task or "").strip().lower()
    code = TEMPLATES.get(task, dedent(f'''
    """Task: {task or "custom"}.
    The loader already selected the model interface and exported a single
    `preprocessor` handle; this template never guesses which one exists.
    """
    import torch
    if preprocessor is None:
        raise RuntimeError("This task needs a tokenizer or processor from Cell 1.")
    inputs = preprocessor("Hello, world!", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    print(outputs)
    ''').strip())
    for k, v in substitutions.items():
        code = code.replace("{{ " + k + " }}", str(v))
    return code