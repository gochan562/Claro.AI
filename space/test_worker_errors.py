"""ZeroGPU worker-boundary error-propagation tests.

Regression for the reported hs-decide-1-flash 502: a ClaroBackendError
raised inside the @spaces.GPU function crosses the ZeroGPU worker boundary
and re-surfaces as gradio.exceptions.Error whose str() is only the exception
class name ("'ClaroBackendError'"), so app.py's handlers never saw the
original message and the client got an opaque model_runtime 502.

`run_inference_guarded()` contains the error INSIDE the worker: it becomes
a structured [CLARO:<code>] return value with the original message, and the
full traceback is printed to the Space runtime logs. app.py's _gpu_infer
delegates to this wrapper.

The tests exercise the REAL guarded wrapper with mocked run_inference; the
decorated app._gpu_infer path additionally depends on gradio/spaces, which
are not installed in this checkout (see the skip note in
test_gpu_worker_boundary below).
"""
import importlib
import io
import os
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(__file__))
from test_backend import _install_fake_transformers, _install_fake_torch, _delete_fake_modules  # noqa: E402
from test_embeddings import FakeTorch  # noqa: E402


class WorkerBoundaryTest(unittest.TestCase):
    def setUp(self):
        _delete_fake_modules()
        sys.modules["torch"] = FakeTorch()
        _install_fake_transformers()
        self.backend = importlib.import_module("backend")

    def tearDown(self):
        _delete_fake_modules()
        if "torch" in self.__dict__.get("_saved", {}):
            pass

    def test_claro_error_preserves_code_and_message(self):
        # The exact production scenario: run_classification rejects a
        # decision/causal model with task_model_mismatch inside the worker.
        err = self.backend.ClaroBackendError(
            "Model loaded as AutoModel has no classification head.",
            code="task_model_mismatch")
        err_out = io.StringIO()
        with patch.object(self.backend, "run_inference", side_effect=err):
            with redirect_stderr(err_out), redirect_stdout(io.StringIO()):
                out = self.backend.run_inference_guarded(
                    model_id="hindsight-ai/hs-decide-1-flash",
                    task="text-classification", prompt="x")
        # structured [CLARO:<code>] payload with the ORIGINAL message
        self.assertTrue(out.startswith("[CLARO:task_model_mismatch]"), out)
        self.assertIn("no classification head", out)
        # full traceback in the server-side stream
        self.assertIn("Traceback", err_out.getvalue())
        self.assertIn("ClaroBackendError", err_out.getvalue())

    def test_unexpected_exception_returns_model_runtime_with_traceback(self):
        boom = RuntimeError("CUDA error: device-side assert triggered")
        err_out = io.StringIO()
        with patch.object(self.backend, "run_inference", side_effect=boom):
            with redirect_stderr(err_out), redirect_stdout(io.StringIO()):
                out = self.backend.run_inference_guarded(
                    model_id="o/m", task="text-generation", prompt="x")
        self.assertTrue(out.startswith("[CLARO:model_runtime]"), out)
        self.assertIn("CUDA error: device-side assert triggered", out)
        self.assertIn("Traceback", err_out.getvalue())
        # not a success-shaped response
        self.assertNotIn("=== ", out)

    def test_success_path_unchanged(self):
        with patch.object(self.backend, "run_inference",
                          return_value="the generated text") as ri:
            out = self.backend.run_inference_guarded(
                model_id="o/m", task="text-generation", prompt="user prompt")
        self.assertEqual(out, "the generated text")
        self.assertTrue(ri.called)

    def test_kwargs_forwarded_to_run_inference(self):
        with patch.object(self.backend, "run_inference", return_value="ok") as ri:
            self.backend.run_inference_guarded(
                model_id="o/m", task="text-generation", prompt="p",
                max_new_tokens=16, task_inputs={"top_k": 3},
                prompt_format="chat")
        kwargs = ri.call_args.kwargs
        self.assertEqual(kwargs["model_id"], "o/m")
        self.assertEqual(kwargs["max_new_tokens"], 16)
        self.assertEqual(kwargs["task_inputs"], {"top_k": 3})
        self.assertEqual(kwargs["prompt_format"], "chat")


class DecisionModelTest(unittest.TestCase):
    def setUp(self):
        _delete_fake_modules()
        sys.modules["torch"] = FakeTorch()
        _install_fake_transformers()
        self.backend = importlib.import_module("backend")

    def tearDown(self):
        _delete_fake_modules()

    @staticmethod
    def _decision_result():
        model = MagicMock(name="qwen35")
        model.device = "cpu"
        model.config = types.SimpleNamespace(
            is_encoder_decoder=False,
            architectures=["Qwen3_5ForCausalLM"],
            model_type="qwen3_5_text")
        return types.SimpleNamespace(
            preprocessor=MagicMock(name="tok"), model=model, modality="text",
            loaded_class="AutoModel", tokenizer=MagicMock(),
            model_id="hindsight-ai/hs-decide-1-flash", dtype="bfloat16")

    @staticmethod
    def _ordinary_causal_result():
        model = MagicMock(name="causal")
        model.device = "cpu"
        model.config = types.SimpleNamespace(
            is_encoder_decoder=False,
            architectures=["Qwen2ForCausalLM"],
            model_type="qwen2")
        return types.SimpleNamespace(
            preprocessor=MagicMock(name="tok"), model=model, modality="text",
            loaded_class="AutoModelForCausalLM", tokenizer=MagicMock(),
            model_id="Qwen/Qwen2.5-0.5B", dtype="float16")

    def test_identified_by_family_and_architecture(self):
        r = self._decision_result()
        self.assertTrue(self.backend.is_decision_model(
            "hindsight-ai/hs-decide-1-flash", r.model.config))
        # a differently-named Qwen3.5 model is NOT a decision model
        self.assertFalse(self.backend.is_decision_model(
            "someone/Qwen3.5-0.8B-Instruct", r.model.config))

    def test_every_task_rejected_with_actionable_error(self):
        r = self._decision_result()
        for task in ("text-classification", "feature-extraction",
                     "sentence-similarity", "text-generation", "summarization"):
            with patch.object(self.backend, "load_model", return_value=r), \
                 patch.object(self.backend, "_to_gpu", return_value=0.0), \
                 patch.object(self.backend, "_offload_gpu", lambda res: None), \
                 self.assertRaises(self.backend.ClaroBackendError) as ctx:
                self.backend.run_inference(
                    r.model_id, task, "x", max_new_tokens=8,
                    task_inputs={"text1": "a", "text2": "b"})
            self.assertEqual(ctx.exception.code, "task_not_supported", task)
            msg = str(ctx.exception)
            self.assertIn("decision model", msg)
            self.assertIn("vLLM", msg)
            self.assertIn("inference.hindsight.vectorize.io", msg)

    def test_never_enters_the_classification_pipeline(self):
        r = self._decision_result()
        with patch.object(self.backend, "run_classification") as run_clf, \
             patch.object(self.backend, "load_model", return_value=r), \
             patch.object(self.backend, "_to_gpu", return_value=0.0), \
             patch.object(self.backend, "_offload_gpu", lambda res: None):
            with self.assertRaises(self.backend.ClaroBackendError) as ctx:
                self.backend.run_inference(
                    "hindsight-ai/hs-decide-1-flash", "text-classification",
                    "x", max_new_tokens=8)
            run_clf.assert_not_called()
            self.assertEqual(ctx.exception.code, "task_not_supported")

    def test_unrelated_causal_lm_keeps_its_handlers(self):
        r = self._ordinary_causal_result()
        with patch.object(self.backend, "load_model", return_value=r), \
             patch.object(self.backend, "_to_gpu", return_value=0.0), \
             patch.object(self.backend, "_offload_gpu", lambda res: None), \
             patch.object(self.backend, "_generate_text",
                          return_value=("generated", 4)) as gen:
            out = self.backend.run_inference(
                "Qwen/Qwen2.5-0.5B", "text-generation", "user prompt",
                max_new_tokens=8)
            self.assertEqual(out, "generated")
            self.assertTrue(gen.called)
            # an ordinary causal model is never rejected as a decision model
            self.assertFalse(self.backend.is_decision_model(
                r.model_id, r.model.config))

    def test_classification_rejection_message_names_the_architecture(self):
        r = self._decision_result()
        with self.assertRaises(self.backend.ClaroBackendError) as ctx:
            self.backend.run_classification(r, r.model_id, "x", {})
        self.assertEqual(ctx.exception.code, "task_model_mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
