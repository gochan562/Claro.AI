"""Chat-template support tests: prompt_format = auto | chat | raw.

Verifies, with fake tokenizers (no real models, no GPU):
  - 'chat'/'auto' call apply_chat_template with the documented shape
    (messages=[{role:user}], tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt")
  - 'auto' falls back to raw tokenization when no chat template exists
  - 'raw' preserves historical verbatim tokenization even when a template
    exists
  - generation decodes ONLY newly generated tokens (prompt ids never leak
    into the decoded text)
  - backward compatibility: calls that omit prompt_format keep working
  - invalid values raise ClaroBackendError(bad_request)
Reuses the fake-module harness from test_backend.py.
"""
import importlib
import os
import sys
import types
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(__file__))
from test_backend import _delete_fake_modules, _install_fake_torch  # noqa: E402


class FakeShape:
    def __init__(self, n):
        self._n = n

    def __getitem__(self, i):
        # inputs["input_ids"].shape[-1] and .shape[0] both behave sanely
        return self._n if i in (-1, 0) else 1


class FakeIds:
    """Minimal 1-D token-id tensor: .shape[-1], slicing, iteration."""

    def __init__(self, total, start=0):
        self._total = total
        self._start = start

    @property
    def shape(self):
        return FakeShape(self._total - self._start)

    def __getitem__(self, sl):
        st = sl.start or 0
        return FakeIds(self._total, self._start + st)

    def __iter__(self):
        return iter(range(self._start, self._total))

    def __len__(self):
        return self._total - self._start

    def to(self, *a, **k):
        return self


def _fake_tokenizer(with_template=True):
    tok = MagicMock(name="tokenizer")
    tok.chat_template = "{{ messages }}" if with_template else None
    tok.eos_token_id = 2
    tok.decode.return_value = "4"
    return tok


def _fake_result(tokenizer, modality="gguf"):
    return types.SimpleNamespace(
        preprocessor=tokenizer,
        model=MagicMock(name="model"),
        modality=modality,
        loaded_class="Test",
        tokenizer=tokenizer,
    )


class PromptFormatTest(unittest.TestCase):
    def setUp(self):
        _delete_fake_modules()
        _install_fake_torch()

    def tearDown(self):
        _delete_fake_modules()

    def _backend(self):
        return importlib.import_module("backend")

    # ── 1. auto + tokenizer WITH a chat template → apply_chat_template ──
    def test_auto_uses_chat_template_when_present(self):
        backend = self._backend()
        tok = _fake_tokenizer(with_template=True)
        tok.apply_chat_template.return_value = {"input_ids": FakeIds(7)}
        r = _fake_result(tok)
        inputs = backend._tokenize_prompt(r, "What is 2 + 2?", "auto")
        tok.apply_chat_template.assert_called_once()
        kwargs = tok.apply_chat_template.call_args.kwargs
        self.assertEqual(kwargs.get("tokenize"), True)
        self.assertEqual(kwargs.get("add_generation_prompt"), True)
        self.assertEqual(kwargs.get("return_dict"), True)
        self.assertEqual(kwargs.get("return_tensors"), "pt")
        msgs = tok.apply_chat_template.call_args.args[0]
        self.assertEqual(msgs, [{"role": "user", "content": "What is 2 + 2?"}])
        # verbatim tokenization must NOT have happened
        tok.assert_not_called()
        self.assertIn("input_ids", inputs)

    # ── 2. auto + tokenizer WITHOUT a template → raw fallback ──
    def test_auto_falls_back_to_raw_without_template(self):
        backend = self._backend()
        tok = _fake_tokenizer(with_template=False)
        tok.side_effect = lambda s, return_tensors=None: {"input_ids": FakeIds(5)}
        r = _fake_result(tok)
        inputs = backend._tokenize_prompt(r, "hello world", "auto")
        tok.assert_called_once_with("hello world", return_tensors="pt")
        tok.apply_chat_template.assert_not_called()
        self.assertEqual(inputs["input_ids"].shape[-1], 5)

    # ── 3. chat mode applies the template explicitly ──
    def test_chat_mode_applies_template(self):
        backend = self._backend()
        tok = _fake_tokenizer(with_template=True)
        tok.apply_chat_template.return_value = {"input_ids": FakeIds(9)}
        r = _fake_result(tok)
        inputs = backend._tokenize_prompt(r, "hi", "chat")
        tok.apply_chat_template.assert_called_once()
        tok.assert_not_called()
        self.assertEqual(inputs["input_ids"].shape[-1], 9)

    # ── 4. raw mode preserves historical behavior (template ignored) ──
    def test_raw_mode_preserves_verbatim_tokenization(self):
        backend = self._backend()
        tok = _fake_tokenizer(with_template=True)  # template exists…
        tok.side_effect = lambda s, return_tensors=None: {"input_ids": FakeIds(4)}
        r = _fake_result(tok)
        backend._tokenize_prompt(r, "hello world", "raw")
        # …but raw bypasses it entirely.
        tok.assert_called_once_with("hello world", return_tensors="pt")
        tok.apply_chat_template.assert_not_called()

    # ── 5. invalid values are rejected ──
    def test_invalid_prompt_format_rejected(self):
        backend = self._backend()
        tok = _fake_tokenizer(with_template=False)
        with self.assertRaises(self._backend().ClaroBackendError) as ctx:
            backend._resolve_prompt_format(tok, "frobnicate")
        self.assertEqual(ctx.exception.code, "bad_request")
        # None/'' fall back to auto rather than raising
        self.assertEqual(backend._resolve_prompt_format(tok, None), "raw")
        self.assertEqual(backend._resolve_prompt_format(tok, "  "), "raw")

    # ── 6. decode receives ONLY newly generated ids (no prompt echo) ──
    def test_generate_decodes_only_newly_generated_tokens(self):
        backend = self._backend()
        tok = _fake_tokenizer(with_template=False)
        decoded = {}
        tok.decode.side_effect = lambda ids, skip_special_tokens=True: (
            decoded.setdefault("ids", list(ids)) and "4")
        tok.side_effect = lambda s, return_tensors=None: {"input_ids": FakeIds(5)}
        r = _fake_result(tok, modality="gguf")
        r.model.generate.return_value = [FakeIds(8)]  # prompt(5) + 3 new
        text, n_new = backend._generate_text_gguf(r, "2 + 2 =", 32, do_sample=False)
        self.assertEqual(text, "4")
        self.assertEqual(n_new, 3)
        self.assertEqual(decoded["ids"], [5, 6, 7],
                         "decode must receive exactly the 3 new ids")
        # none of the prompt-token range [0..4] may appear
        for pid in range(0, 5):
            self.assertNotIn(pid, decoded["ids"])
        # generate received the chat/raw input ids
        self.assertEqual(r.model.generate.call_args.kwargs["input_ids"].shape[-1], 5)

    # ── 7. chat format flows through generation (template length ≠ prompt length) ──
    def test_generate_chat_format_slices_from_templated_input(self):
        backend = self._backend()
        tok = _fake_tokenizer(with_template=True)
        decoded = {}
        tok.decode.side_effect = lambda ids, skip_special_tokens=True: (
            decoded.setdefault("ids", list(ids)) and "ok")
        # template-rendered input is LONGER than the raw prompt (9 vs 4)
        tok.apply_chat_template.return_value = {"input_ids": FakeIds(9)}
        r = _fake_result(tok, modality="gguf")
        r.model.generate.return_value = [FakeIds(12)]  # templated(9) + 3 new
        text, n_new = backend._generate_text_gguf(
            r, "What is 2 + 2?", 32, do_sample=False, prompt_format="chat")
        self.assertEqual(text, "ok")
        self.assertEqual(n_new, 3)  # sliced from 9, not from len(prompt)
        self.assertEqual(decoded["ids"], [9, 10, 11])
        tok.apply_chat_template.assert_called_once()

    # ── 8. backward compatibility: no prompt_format kwarg at all ──
    def test_generate_backward_compatible_without_prompt_format(self):
        backend = self._backend()
        tok = _fake_tokenizer(with_template=False)
        decoded = {}
        tok.decode.side_effect = lambda ids, skip_special_tokens=True: (
            decoded.setdefault("ids", list(ids)) and "gen")
        tok.side_effect = lambda s, return_tensors=None: {"input_ids": FakeIds(4)}
        r = _fake_result(tok, modality="gguf")
        r.model.generate.return_value = [FakeIds(6)]
        # historical call shape — no prompt_format argument
        text, n_new = backend._generate_text_gguf(r, "hello", 16, do_sample=False)
        self.assertEqual(text, "gen")
        self.assertEqual(n_new, 2)
        tok.assert_called_once_with("hello", return_tensors="pt")
        tok.apply_chat_template.assert_not_called()

    # ── 9. run_inference accepts and forwards prompt_format (signature) ──
    def test_run_inference_accepts_prompt_format_kwarg(self):
        backend = self._backend()
        tok = _fake_tokenizer(with_template=True)
        tok.apply_chat_template.return_value = {"input_ids": FakeIds(7)}
        r = _fake_result(tok, modality="transformers")
        patched_loader = MagicMock(return_value=r)
        with unittest.mock.patch.object(backend, "load_model", patched_loader), \
             unittest.mock.patch.object(backend, "_to_gpu", return_value=0.0), \
             unittest.mock.patch.object(backend, "_offload_gpu", lambda result: None), \
             unittest.mock.patch.object(backend, "_generate_text",
                                        return_value=("text", 5)) as gen:
            backend.run_inference("owner/model", "text-generation", "hi",
                                  max_new_tokens=8, prompt_format="chat")
            self.assertEqual(gen.call_args.kwargs.get("prompt_format"), "chat")


import unittest.mock  # noqa: E402  (used in test 9)

if __name__ == "__main__":
    unittest.main(verbosity=2)
