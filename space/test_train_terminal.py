"""Gradio terminal-contract tests for the Space-side train endpoint.

Regression coverage for: "remote training failed without a manifest".

Root cause: ``train_api.train`` is a Python generator. Its terminal output
used ``return (manifest_json, file)``, whose value becomes
``StopIteration.value`` — discarded both by ``yield from`` (space/app.py
``_gpu_train``) and by Gradio's iterator handling. Gradio therefore emitted
the LAST *yielded* tuple (a ``[TRAIN] finished`` log event) as the
``complete`` frame instead of ``[manifest, file]``.

Contract under test (unchanged schema, unchanged two-output endpoint):
  * every intermediate yield is ``(event_json_string, None)``  -> generating
  * the FINAL yield is ``(manifest_json_string, zip_path_or_None)``
    -> complete frame ``[manifestJsonString, fileObj/null]``
  * the generator then terminates normally (``StopIteration.value is None``)

The harness below drives the ACTUAL ``train_api.train`` endpoint through a
``yield from`` wrapper identical to ``space/app.py::_gpu_train`` and collects
values exactly as Gradio does: each yield -> generating frame, last yield ->
complete frame. Only the training worker itself is stubbed (tiny, hermetic:
writes a fake output file, no threads-of-its-own, no torch/GPU); duration
logic, training_runner.py, the manifest schema and the endpoint shape are
all the real code paths.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))

import train_api


def _valid_request(job_id):
    return json.dumps(_valid_config(job_id))


def _valid_config(job_id):
    return {
        "model_id": "distilbert-base-uncased",
        "dataset_id": "stanfordnlp/imdb",
        "task_type": "text-classification",
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 0.00002,
        "validation_split": 10,
        "max_samples": 100,
        "max_steps": 7,
        "training_method": "full",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": "auto",
        "job_id": job_id,
    }


def _stub_worker_ok(line_q=None, metrics_count=1):
    """Tiny mocked training runner: writes fake outputs, emits stdout lines."""
    def _run(cfg, out_dir, line_q_in, stop_flag):
        from pathlib import Path
        out = Path(str(out_dir))
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text("{}")
        (out / "model.safetensors").write_text("fake-weights")
        (out / "metrics.json").write_text(json.dumps(
            [{"step": 1, "epoch": 1.0, "train_loss": 0.5}]))
        q = line_q if line_q is not None else line_q_in
        if q is not None:
            q.put(("stdout", "[TRAIN] total_steps=10"))
            q.put(("stdout", '{"step": 1, "epoch": 1.0, "train_loss": 0.5}'))
        return {"metrics_count": metrics_count}
    return _run


def collect_gradio(request_json):
    """Drive the endpoint exactly as Gradio + app.py do.

    Mirrors ``space/app.py::_gpu_train`` (``yield from train_fn(...)``):
    each yielded value is a `generating` frame, the LAST yielded value is
    the `complete` frame, and ``StopIteration.value`` (a bare/generator
    ``return value``) is discarded. Returns
    ``(generating_list, complete_value, stop_value)``.
    """
    def _wrapped_like_gpu_train():
        yield from train_api.train(request_json)

    generating = []
    stop_value = None
    it = _wrapped_like_gpu_train()
    try:
        while True:
            generating.append(next(it))
    except StopIteration as e:
        stop_value = e.value
    complete = generating[-1] if generating else None
    return generating, complete, stop_value


class TrainTerminalContractTest(unittest.TestCase):
    def tearDown(self):
        for jid in getattr(self, "_job_ids", []):
            for p in (os.path.join(tempfile.gettempdir(), f"claro_train_{jid}"),
                      os.path.join(tempfile.gettempdir(), f"claro_train_{jid}.zip")):
                shutil.rmtree(p, ignore_errors=True)
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def _track(self, job_id):
        self._job_ids = getattr(self, "_job_ids", []) + [job_id]

    def test_success_terminal_complete_is_manifest_not_last_log(self):
        job_id = "train_0123456789ab"
        self._track(job_id)
        with patch.object(train_api, "train_worker", _stub_worker_ok()):
            generating, complete, stop_value = collect_gradio(_valid_request(job_id))

        # The old bug: StopIteration carried the manifest (discarded by
        # Gradio), while `complete` held the last log yield.
        self.assertIsNone(stop_value,
                          "generator must terminate normally: no hidden return payload")
        self.assertGreaterEqual(len(generating), 2,
                                "expected intermediate yields + terminal yield")
        # Intermediate yields unchanged: (event_json, None).
        first_evt = json.loads(generating[0][0])
        self.assertEqual(first_evt.get("type"), "status")
        self.assertIsNone(generating[0][1])
        for item, _f in generating[:-1]:
            json.loads(item)  # every intermediate body is JSON
            self.assertIsNone(_f)
        # Terminal complete payload is [manifestJsonString, filePath].
        manifest_str, fpath = complete
        manifest = json.loads(manifest_str)
        self.assertEqual(manifest.get("status"), "finished")
        self.assertEqual(manifest.get("job_id"), job_id)
        self.assertTrue(manifest.get("files"), "manifest must list artifact files")
        self.assertIsInstance(fpath, str)
        self.assertTrue(os.path.exists(fpath), "artifact zip must exist")
        # ...and NOT the previous last-yield log/event object.
        self.assertNotEqual(manifest.get("type"), "log")
        self.assertIn("job_id", manifest)

    def test_failed_validation_terminal_complete(self):
        job_id = "train_0123456789ab"
        self._track(job_id)
        bad = json.loads(_valid_request(job_id))
        bad["model_id"] = "!!!"
        generating, complete, stop_value = collect_gradio(json.dumps(bad))

        self.assertIsNone(stop_value)
        manifest_str, f = complete
        manifest = json.loads(manifest_str)
        self.assertEqual(manifest.get("status"), "failed")
        self.assertTrue(manifest.get("message"), "failed manifest must carry a message")
        self.assertIsNone(f)
        # The validation log yield still precedes the terminal manifest yield.
        self.assertTrue(any("training_error" in g[0] for g in generating[:-1]))

    def test_failed_worker_terminal_complete(self):
        job_id = "train_abcdef012345"
        self._track(job_id)

        def _boom(cfg, out_dir, line_q, stop_flag):
            raise RuntimeError("boom")

        with patch.object(train_api, "train_worker", _boom):
            generating, complete, stop_value = collect_gradio(_valid_request(job_id))

        self.assertIsNone(stop_value)
        manifest_str, f = complete
        manifest = json.loads(manifest_str)
        self.assertEqual(manifest.get("status"), "failed")
        self.assertIn("boom", manifest.get("message", ""))
        self.assertIsNone(f)
        self.assertTrue(any("training_error" in g[0] for g in generating[:-1]))

    def test_invalid_json_terminal_complete(self):
        generating, complete, stop_value = collect_gradio("{{{not json")

        self.assertIsNone(stop_value)
        # Single terminal yield (no prior progress yields on this path).
        self.assertEqual(len(generating), 1)
        manifest_str, f = complete
        manifest = json.loads(manifest_str)
        self.assertEqual(manifest.get("status"), "failed")
        self.assertIn("not valid JSON", manifest.get("message", ""))
        self.assertIsNone(f)

    def test_artifact_packaging_failure_terminal_complete(self):
        job_id = "train_111111111111"
        self._track(job_id)
        with patch.object(train_api, "train_worker", _stub_worker_ok()), \
             patch.object(train_api, "_zip_dir", side_effect=OSError("disk full")):
            generating, complete, stop_value = collect_gradio(_valid_request(job_id))

        self.assertIsNone(stop_value)
        manifest_str, f = complete
        manifest = json.loads(manifest_str)
        self.assertEqual(manifest.get("status"), "failed")
        self.assertIn("artifact packaging failed", manifest.get("message", ""))
        self.assertIsNone(f)


class TrainRequestShapeTest(unittest.TestCase):
    """Regression coverage for: "request is not valid JSON".

    The live /call/v2/train client POSTs ``{"train_request_json": {...}}``
    (training_backend.js payload), but the Spaces runtime hands the endpoint
    that same logical request in several packings: envelope dict, JSON
    string, single-quoted Python-repr string, or mapped kwargs. All must fold
    to one dict via a single normalization (never eval()). Plain dict and
    JSON-string shapes keep their exact legacy behavior.
    """

    def tearDown(self):
        for jid in getattr(self, "_job_ids", []):
            for p in (os.path.join(tempfile.gettempdir(), f"claro_train_{jid}"),
                      os.path.join(tempfile.gettempdir(), f"claro_train_{jid}.zip")):
                shutil.rmtree(p, ignore_errors=True)
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def _track(self, job_id):
        self._job_ids = getattr(self, "_job_ids", []) + [job_id]

    def _run_shapes(self, make_request, job_id, stub=None):
        """Drive one live packing through the Gradio harness; return
        (manifest_dict, complete_file, worker_cfg)."""
        self._track(job_id)
        seen = {}
        real_stub = stub or _stub_worker_ok()

        def _capture(cfg, out_dir, line_q, stop_flag):
            seen.update(cfg)
            return real_stub(cfg, out_dir, line_q, stop_flag)

        with patch.object(train_api, "train_worker", _capture):
            _gen, complete, stop_value = collect_gradio(make_request())
        self.assertIsNone(stop_value, "terminal output must be yielded, not returned")
        manifest_str, f = complete
        return json.loads(manifest_str), f, seen

    def _assert_live_success(self, manifest, f, seen, cfg):
        self.assertEqual(manifest.get("status"), "finished")
        self.assertEqual(manifest.get("job_id"), cfg["job_id"])
        self.assertTrue(manifest.get("files"))
        self.assertIsInstance(f, str)
        self.assertTrue(os.path.exists(f))
        # Normalized worker fields match the sent request exactly.
        for k in ("model_id", "dataset_id", "task_type", "epochs",
                  "batch_size", "max_steps"):
            self.assertEqual(seen.get(k), cfg[k], f"normalized field {k}")

    def test_live_envelope_dict_shape(self):
        # Exact POST body training_backend.js sends, as a positional dict.
        cfg = _valid_config("train_222222222222")
        m, f, seen = self._run_shapes(
            lambda: {"train_request_json": dict(cfg)}, cfg["job_id"])
        self._assert_live_success(m, f, seen, cfg)

    def test_live_envelope_json_string_shape(self):
        cfg = _valid_config("train_333333333333")
        body = json.dumps({"train_request_json": cfg})
        m, f, seen = self._run_shapes(lambda: body, cfg["job_id"])
        self._assert_live_success(m, f, seen, cfg)

    def test_live_envelope_repr_string_shape(self):
        # Observed runtime shape: single-quoted Python-dict repr, which
        # json.loads alone rejects (the reported "not valid JSON" failure).
        cfg = _valid_config("train_444444444444")
        body = repr({"train_request_json": cfg})
        self.assertIn("'", body)
        m, f, seen = self._run_shapes(lambda: body, cfg["job_id"])
        self._assert_live_success(m, f, seen, cfg)

    def test_live_kwargs_shape_via_unpack(self):
        # v2 gateway fn(**body): unpack_request maps kwargs -> endpoint value.
        cfg = _valid_config("train_555555555555")
        unpacked = train_api.unpack_request(
            None, (), {"train_request_json": dict(cfg)})
        m, f, seen = self._run_shapes(lambda: unpacked, cfg["job_id"])
        self._assert_live_success(m, f, seen, cfg)

    def test_plain_shapes_unchanged(self):
        # Legacy compat: bare config dict and bare JSON config string.
        for jid in ("train_666666666666", "train_777777777777"):
            cfg = _valid_config(jid)
            raw = cfg if jid.endswith("66") else json.dumps(cfg)
            m, f, seen = self._run_shapes(lambda r=raw: r, jid)
            self._assert_live_success(m, f, seen, cfg)

    def test_normalize_never_uses_eval(self):
        import ast as _ast
        tree = _ast.parse(__import__("inspect").getsource(train_api._normalize_train_request))
        called = {n.func.id for n in _ast.walk(tree)
                  if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)}
        self.assertTrue(called.isdisjoint({"eval", "exec", "compile"}),
                        f"forbidden calls: {called & {'eval', 'exec', 'compile'}}")
        # A hostile repr must not execute: literal_eval rejects calls.
        self.assertIsNone(train_api._normalize_train_request(
            "{'x': __import__('os').system('x')}"))
        # ...and the endpoint turns it into the safe invalid-JSON manifest.
        _gen, complete, _stop = collect_gradio("{'x': 1}))){{{")
        manifest = json.loads(complete[0])
        self.assertEqual(manifest.get("status"), "failed")

    def test_garbage_string_still_invalid_json(self):
        _gen, complete, stop_value = collect_gradio("{{{not json")
        self.assertIsNone(stop_value)
        manifest = json.loads(complete[0])
        self.assertEqual(manifest.get("status"), "failed")
        self.assertIn("not valid JSON", manifest.get("message", ""))

    def test_normalized_keys_printed(self):
        # Prints the normalized keys + confirms the six required fields.
        cfg = _valid_config("train_888888888888")
        m, f, seen = self._run_shapes(
            lambda: {"train_request_json": dict(cfg)}, cfg["job_id"])
        print(f"\nnormalized request keys={sorted(seen.keys())}", flush=True)
        for k in ("model_id", "dataset_id", "task_type", "epochs",
                  "batch_size", "max_steps"):
            print(f"normalized {k}={seen.get(k)!r}", flush=True)
            self.assertIn(k, seen)
        self.assertEqual(m.get("status"), "finished")


if __name__ == "__main__":
    unittest.main(verbosity=2)
