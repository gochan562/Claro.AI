"""Task-aware inference routing tests for space/backend.py.

Mocked models/tokenizers (fake torch from test_embeddings) — no GPU, no
downloads. Verifies per-task handlers: inputs actually reach the model,
outputs are structured, and unsupported tasks/inputs raise clear errors
instead of silently falling back to generic text generation.
"""
import importlib
import os
import sys
import types
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(__file__))
from test_embeddings import (  # noqa: E402  (fake torch + helpers)
    FT, FakeTorch, FakeBatch, _delete_fake_modules)
from test_backend import _install_fake_transformers  # noqa: E402


class RoutingTestBase(unittest.TestCase):
    def setUp(self):
        _delete_fake_modules()
        sys.modules["torch"] = FakeTorch()
        # _generate_text imports LogitsProcessorList/LogitsProcessor — the
        # fake transformers module satisfies that import without torch.
        _install_fake_transformers()

    def tearDown(self):
        _delete_fake_modules()

    def _backend(self):
        return importlib.import_module("backend")


def seq2seq_result(tokenizer, generated=None):
    """Encoder-decoder (BART/T5-style) fake: config.is_encoder_decoder=True,
    generate() returns [FT(total)] decoded wholly. Callers configure
    tokenizer.decode themselves; `generated` sets it only when provided."""
    model = MagicMock(name="seq2seq")
    model.device = "cpu"
    model.config = types.SimpleNamespace(is_encoder_decoder=True)
    model.generate.return_value = [FT([1, 2, 3, 4])]
    if generated is not None:
        tokenizer.decode.return_value = generated
    tokenizer.eos_token_id = 2
    return types.SimpleNamespace(preprocessor=tokenizer, model=model,
                                 modality="text", loaded_class="AutoModelForSeq2SeqLM",
                                 tokenizer=tokenizer)


def causal_result(tokenizer, generated_ids=(1, 2, 3, 4, 5, 6), prompt_len=4):
    """Decoder-only fake: generate returns prompt+new; decode gets the slice."""
    model = MagicMock(name="causal")
    model.device = "cpu"
    model.config = types.SimpleNamespace(is_encoder_decoder=False)
    model.generate.return_value = [FT(list(generated_ids))]
    return types.SimpleNamespace(preprocessor=tokenizer, model=model,
                                 modality="text", loaded_class="AutoModelForCausalLM",
                                 tokenizer=tokenizer)


class SummarizationTest(RoutingTestBase):
    def test_seq2seq_receives_source_document_with_length_bounds(self):
        backend = self._backend()
        tok = MagicMock(name="tokenizer")
        tok.side_effect = lambda text, **k: FakeBatch({"input_ids": FT([[1] * 9])})
        tok.decode.return_value = "foxes are quick"
        tok.eos_token_id = 2
        r = seq2seq_result(tok)
        out = backend.run_summarization(
            r, "owner/bart",
            "The transformer architecture revolutionized NLP by enabling parallel "
            "sequence processing via self-attention mechanisms.",
            {"max_length": 64, "min_length": 10})
        # the SOURCE DOCUMENT reached the tokenizer — never a generic prompt
        sent_text = tok.call_args.args[0]
        self.assertIn("transformer architecture", sent_text)
        self.assertNotIn("Once upon a time", sent_text)
        gen_kwargs = r.model.generate.call_args.kwargs
        self.assertEqual(gen_kwargs["max_length"], 64)
        self.assertEqual(gen_kwargs["min_length"], 10)
        self.assertIn("=== Summary ===", out)
        self.assertIn("foxes are quick", out)

    def test_defaults_when_lengths_omitted(self):
        backend = self._backend()
        tok = MagicMock(name="tokenizer")
        tok.side_effect = lambda text, **k: FakeBatch({"input_ids": FT([[1] * 9])})
        tok.decode.return_value = "s"
        tok.eos_token_id = 2
        r = seq2seq_result(tok)
        backend.run_summarization(r, "o/m", "doc text", {})
        self.assertEqual(r.model.generate.call_args.kwargs["max_length"], 142)
        self.assertEqual(r.model.generate.call_args.kwargs["min_length"], 56)

    def test_empty_document_rejected(self):
        backend = self._backend()
        r = seq2seq_result(MagicMock(name="tok"))
        with self.assertRaises(backend.ClaroBackendError) as ctx:
            backend.run_summarization(r, "o/m", "   ", {})
        self.assertEqual(ctx.exception.code, "invalid_input")

    def test_causal_wrapper_uses_instruction_prompt(self):
        backend = self._backend()
        tok = MagicMock(name="tokenizer")
        prompt_len = 6
        tok.side_effect = lambda text, **k: FakeBatch(
            {"input_ids": FT([[1] * prompt_len])})
        tok.decode.side_effect = lambda ids, skip_special_tokens=True: "summary text"
        tok.eos_token_id = 2
        r = causal_result(tok, generated_ids=[1] * (prompt_len + 5), prompt_len=prompt_len)
        out = backend.run_summarization(r, "o/m", "long document here", {})
        sent = tok.call_args.args[0]
        self.assertIn("Summarize", sent)
        self.assertIn("long document here", sent)
        self.assertIn("summary text", out)


class ClassificationTest(RoutingTestBase):
    def _clf(self, labels=("NEGATIVE", "POSITIVE"), logits_rows=None):
        import torch as ft_mod
        backend = self._backend()
        tok = MagicMock(name="tokenizer")
        rows = logits_rows or [[0.0, 0.0], [0.0, 0.0]]
        tok.side_effect = lambda texts, **k: FakeBatch(
            {"input_ids": FT([[1] * 4 for _ in texts]),
             "attention_mask": FT([[1] * 4 for _ in texts])})
        model = MagicMock(name="clf")
        model.device = "cpu"
        model.config = types.SimpleNamespace(is_encoder_decoder=False,
                                             id2label={0: labels[0], 1: labels[1]})
        # logits: text1 -> POSITIVE 0.9; text2 -> NEGATIVE 0.8
        model.side_effect = lambda **k: types.SimpleNamespace(
            logits=FT(logits_rows or [[0.1, 2.2], [2.0, 0.5]]))
        r = types.SimpleNamespace(preprocessor=tok, model=model, modality="text",
                                  loaded_class="AutoModelForSequenceClassification",
                                  tokenizer=tok)
        return backend, r, tok

    def test_labels_and_scores_returned(self):
        backend, r, tok = self._clf()
        out = backend.run_classification(
            r, "owner/bert", "I love this\nI hate this", {"top_k": 2})
        sent = tok.call_args.args[0]
        self.assertEqual(sent, ["I love this", "I hate this"],
                         "actual user texts reach the tokenizer")
        self.assertIn("=== Classification ===", out)
        self.assertIn("Predicted: POSITIVE", out)   # row [0.1, 2.2] → idx1
        self.assertIn("Predicted: NEGATIVE", out)   # row [2.0, 0.5] → idx0
        self.assertIn("Top predictions:", out)
        self.assertNotIn("Once upon a time", out)

    def test_non_classification_model_rejected_not_generated(self):
        backend = self._backend()
        tok = MagicMock(name="tok")
        tok.eos_token_id = 2
        r = types.SimpleNamespace(preprocessor=tok, model=MagicMock(name="causal"),
                                  modality="text", loaded_class="AutoModelForCausalLM",
                                  tokenizer=tok)
        with self.assertRaises(backend.ClaroBackendError) as ctx:
            backend.run_classification(r, "o/m", "any text", {})
        self.assertEqual(ctx.exception.code, "task_model_mismatch")
        self.assertIn("classification head", str(ctx.exception))

    def test_missing_labels_rejected(self):
        backend = self._backend()
        backend_r = None
        tok = MagicMock(name="tok")
        tok.side_effect = lambda texts, **k: FakeBatch(
            {"input_ids": FT([[1] * 4])})
        model = MagicMock(name="clf")
        model.device = "cpu"
        model.config = types.SimpleNamespace(is_encoder_decoder=False, id2label=None)
        model.side_effect = lambda **k: types.SimpleNamespace(logits=FT([[1.0, 2.0]]))
        r = types.SimpleNamespace(preprocessor=tok, model=model, modality="text",
                                  loaded_class="AutoModelForSequenceClassification",
                                  tokenizer=tok)
        with self.assertRaises(backend.ClaroBackendError) as ctx:
            backend.run_classification(r, "o/m", "text", {})
        self.assertIn("id2label", str(ctx.exception))

    def test_empty_input_rejected(self):
        backend, r, _ = self._clf()
        with self.assertRaises(backend.ClaroBackendError) as ctx:
            backend.run_classification(r, "o/m", "  \n  ", {})
        self.assertEqual(ctx.exception.code, "invalid_input")


class QuestionAnsweringTest(RoutingTestBase):
    def test_extractive_qa_uses_question_and_context(self):
        backend = self._backend()
        tok = MagicMock(name="tokenizer")
        # tokenizer(question, context) returns ids for the CONTEXT span;
        # decode receives the sliced span ids [start:end]
        tok.side_effect = lambda q, c, **k: FakeBatch(
            {"input_ids": FT([[10, 11, 12, 13, 14]])})
        tok.decode.side_effect = lambda ids, skip_special_tokens=True: "Paris"
        model = MagicMock(name="qa")
        model.device = "cpu"
        model.__class__ = type("AutoModelForQuestionAnswering", (), {})
        model.config = types.SimpleNamespace(is_encoder_decoder=False)
        model.side_effect = lambda **k: types.SimpleNamespace(
            start_logits=FT([[0.1, 2.0, 0.3, 0.1, 0.0]]),
            end_logits=FT([[0.0, 0.2, 1.5, 0.9, 0.0]]))
        r = types.SimpleNamespace(preprocessor=tok, model=model, modality="text",
                                  loaded_class="AutoModelForQuestionAnswering",
                                  tokenizer=tok)
        tok.decode.side_effect = lambda ids, skip_special_tokens=True: "Paris"
        out = backend.run_question_answering(
            r, "owner/bert-squad", None,
            {"question": "Where is the Eiffel Tower?",
             "context": "The Eiffel Tower is located in Paris, France."})
        self.assertEqual(tok.call_args.args[0], "Where is the Eiffel Tower?")
        self.assertIn("Paris", tok.call_args.args[1])
        self.assertIn("extractive", out)
        self.assertIn("A: Paris", out)

    def test_generative_qa_fallback(self):
        backend = self._backend()
        tok = MagicMock(name="tokenizer")
        prompt_len = 5
        tok.side_effect = lambda text, **k: FakeBatch({"input_ids": FT([[1] * prompt_len])})
        tok.eos_token_id = 2
        r = causal_result(tok, generated_ids=[1] * (prompt_len + 3), prompt_len=prompt_len)
        tok.decode.side_effect = lambda ids, skip_special_tokens=True: "1889"
        out = backend.run_question_answering(
            r, "owner/llama", None,
            {"question": "When was it built?", "context": "It was built in 1889."})
        self.assertIn("generative", out)
        self.assertIn("A: 1889", out)
        sent = tok.call_args.args[0]
        self.assertIn("When was it built?", sent)
        self.assertIn("It was built in 1889.", sent)

    def test_missing_fields_rejected(self):
        backend = self._backend()
        r = seq2seq_result(MagicMock(name="tok"))
        with self.assertRaises(backend.ClaroBackendError) as ctx:
            backend.run_question_answering(r, "o/m", None, {"question": "q only"})
        self.assertEqual(ctx.exception.code, "invalid_input")
        self.assertIn("context", str(ctx.exception))


class TranslationTest(RoutingTestBase):
    def test_translation_with_language_forcing(self):
        backend = self._backend()
        tok = MagicMock(name="tokenizer")
        tok.side_effect = lambda text, **k: FakeBatch({"input_ids": FT([[1] * 7])})
        tok.eos_token_id = 2
        tok.lang_code_to_id = {"fr_XX": 99}
        r = seq2seq_result(tok)
        tok.decode.return_value = "Bonjour le monde"
        out = backend.run_translation(r, "owner/nllb",
                                      "Hello the world", {"target_language": "fr_XX"})
        sent = tok.call_args.args[0]
        self.assertEqual(sent, "Hello the world", "source text, not a generic prompt")
        self.assertEqual(r.model.generate.call_args.kwargs["forced_bos_token_id"], 99)
        self.assertIn("=== Translation ===", out)
        self.assertIn("fr_XX", out)
        self.assertIn("Bonjour le monde", out)

    def test_translation_without_language_codes_notes_limitation(self):
        backend = self._backend()
        tok = MagicMock(name="tokenizer")
        tok.side_effect = lambda text, **k: FakeBatch({"input_ids": FT([[1] * 7])})
        tok.eos_token_id = 2
        r = seq2seq_result(tok)
        tok.decode.return_value = "hola"
        out = backend.run_translation(r, "owner/t5", "Hello", {"target_language": "es"})
        self.assertNotIn("forced_bos_token_id", str(r.model.generate.call_args.kwargs))
        self.assertIn("does not expose language codes", out)
        self.assertIn("hola", out)

    def test_empty_source_rejected(self):
        backend = self._backend()
        r = seq2seq_result(MagicMock(name="tok"))
        with self.assertRaises(backend.ClaroBackendError) as ctx:
            backend.run_translation(r, "o/m", "", {"target_language": "fr"})
        self.assertEqual(ctx.exception.code, "invalid_input")


class TextGenerationTaskTest(RoutingTestBase):
    def test_user_prompt_reaches_generation(self):
        backend = self._backend()
        tok = MagicMock(name="tokenizer")
        prompt_len = 4
        tok.side_effect = lambda text, **k: FakeBatch({"input_ids": FT([[1] * prompt_len])})
        tok.decode.side_effect = lambda ids, skip_special_tokens=True: "the answer"
        tok.eos_token_id = 2
        r = causal_result(tok, generated_ids=[1] * (prompt_len + 4), prompt_len=prompt_len)
        out = backend.run_text_generation_task(
            r, "o/m", "What is 2 + 2? Give a short answer.", {},
            max_new_tokens=32, temperature=0.7, top_p=0.9,
            do_sample=False, prompt_format="chat")
        sent = tok.apply_chat_template.call_args.args[0] \
            if tok.apply_chat_template.called else tok.call_args.args[0]
        self.assertIn("What is 2 + 2?", str(sent))
        self.assertNotIn("Once upon a time", str(sent))
        self.assertEqual(out, "the answer")

    def test_chat_template_still_applies_when_selected(self):
        backend = self._backend()
        tok = MagicMock(name="tokenizer")
        tok.chat_template = "{{ messages }}"
        tok.eos_token_id = 2
        tok.apply_chat_template.return_value = FakeBatch({"input_ids": FT([[1, 2, 3]])})
        tok.decode.return_value = "reply"
        r = causal_result(tok)
        r.model.generate.return_value = [FT([1, 2, 3, 7, 8])]
        backend.run_text_generation_task(
            r, "o/chat-model", "hi there", {},
            max_new_tokens=16, temperature=0.7, top_p=0.9,
            do_sample=False, prompt_format="chat")
        self.assertTrue(tok.apply_chat_template.called,
                        "chat format must still use the model's chat template")


class SentenceSimilarityTest(RoutingTestBase):
    def test_two_texts_cosine(self):
        backend = self._backend()
        tok = MagicMock(name="tokenizer")
        # two single-token texts with orthogonal embeddings
        tok.padding_side = "right"
        tok.pad_token = None
        tok.eos_token = "<eos>"
        tok.side_effect = lambda texts, **k: FakeBatch(
            {"input_ids": FT([[1], [1]]),
             "attention_mask": FT([[1], [1]])})
        model = MagicMock(name="causal-embed")
        model.device = "cpu"
        model.config = types.SimpleNamespace(is_encoder_decoder=False)
        model.side_effect = lambda **k: types.SimpleNamespace(
            hidden_states=(FT([[[1.0, 0.0]], [[0.0, 1.0]]]),))
        r = types.SimpleNamespace(preprocessor=tok, model=model, modality="text",
                                  loaded_class="AutoModelForCausalLM", tokenizer=tok)
        out = backend.run_sentence_similarity(
            r, "owner/embed", None, {"text1": "first sentence", "text2": "second sentence"})
        self.assertIn("Cosine similarity: 0.0000", out)
        self.assertIn("Text 1: first sentence", out)
        self.assertIn("Text 2: second sentence", out)

    def test_zero_vector_reported_as_zero(self):
        backend = self._backend()
        tok = MagicMock(name="tokenizer")
        tok.padding_side = "right"
        tok.pad_token = None
        tok.eos_token = "<eos>"
        tok.side_effect = lambda texts, **k: FakeBatch(
            {"input_ids": FT([[1], [1]]),
             "attention_mask": FT([[1], [1]])})
        model = MagicMock(name="m")
        model.device = "cpu"
        model.config = types.SimpleNamespace(is_encoder_decoder=False)
        model.side_effect = lambda **k: types.SimpleNamespace(
            hidden_states=(FT([[[0.0, 0.0]], [[0.0, 1.0]]]),))
        r = types.SimpleNamespace(preprocessor=tok, model=model, modality="text",
                                  loaded_class="AutoModelForCausalLM", tokenizer=tok)
        out = backend.run_sentence_similarity(r, "o/e", None,
                                              {"text1": "a", "text2": "b"})
        self.assertIn("Cosine similarity: 0.0000", out)
        self.assertIn("degenerate", out)


class TaskInputsParsingTest(RoutingTestBase):
    def test_json_string_accepted_and_invalid_rejected(self):
        backend = self._backend()
        self.assertEqual(backend._parse_task_inputs('{"question": "q"}'), {"question": "q"})
        self.assertEqual(backend._parse_task_inputs(None), {})
        self.assertEqual(backend._parse_task_inputs(""), {})
        with self.assertRaises(backend.ClaroBackendError) as ctx:
            backend._parse_task_inputs("not json{")
        self.assertEqual(ctx.exception.code, "invalid_input")
        with self.assertRaises(backend.ClaroBackendError):
            backend._parse_task_inputs('["array"]')


class DispatchTest(RoutingTestBase):
    def test_dispatch_routes_by_task_and_unsupported_raises(self):
        backend = self._backend()
        tok = MagicMock(name="tok")
        tok.eos_token_id = 2
        tok.chat_template = None  # raw generation path
        tok.side_effect = lambda text, **k: FakeBatch({"input_ids": FT([[1, 2, 3]])})
        tok.decode.return_value = "gen"
        r = causal_result(tok, generated_ids=[1, 2, 3, 4], prompt_len=3)

        with unittest.mock.patch.object(backend, "load_model", return_value=r), \
             unittest.mock.patch.object(backend, "_to_gpu", return_value=0.0), \
             unittest.mock.patch.object(backend, "_offload_gpu", lambda res: None):
            # supported: text-generation routes to the generation handler
            out = backend.run_inference("o/m", "text-generation", "user prompt",
                                        max_new_tokens=8, task_inputs={})
            self.assertEqual(out, "gen")
            tok.assert_called_with("user prompt", return_tensors="pt")

            # UNSUPPORTED task must raise task_not_supported — never fall
            # back to generic generation or raw object dumps
            for bad in ("fill-mask", "token-classification", "zero-shot-classification",
                        "anything-else"):
                with self.assertRaises(backend.ClaroBackendError) as ctx:
                    backend.run_inference("o/m", bad, "x", max_new_tokens=8)
                self.assertEqual(ctx.exception.code, "task_not_supported", bad)
                self.assertIn("Supported text tasks", str(ctx.exception))

    def test_summarization_dispatch_receives_source_text(self):
        backend = self._backend()
        tok = MagicMock(name="tok")
        tok.side_effect = lambda text, **k: FakeBatch({"input_ids": FT([[1] * 6])})
        tok.eos_token_id = 2
        r = seq2seq_result(tok)
        tok.decode.return_value = "short summary"
        with unittest.mock.patch.object(backend, "load_model", return_value=r), \
             unittest.mock.patch.object(backend, "_to_gpu", return_value=0.0), \
             unittest.mock.patch.object(backend, "_offload_gpu", lambda res: None):
            out = backend.run_inference(
                "owner/bart", "summarization",
                "The transformer architecture revolutionized NLP.",
                max_new_tokens=8,
                task_inputs={"max_length": 50, "min_length": 5})
            sent = tok.call_args.args[0]
            self.assertIn("transformer architecture", sent)
            self.assertNotIn("Once upon a time", sent)
            self.assertIn("short summary", out)


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main(verbosity=2)
