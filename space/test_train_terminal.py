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
    return json.dumps({
        "model_id": "distilbert-base-uncased",
        "dataset_id": "stanfordnlp/imdb",
        "task_type": "text-classification",
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.00002,
        "validation_split": 10,
        "max_samples": 100,
        "max_steps": None,
        "training_method": "full",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": "auto",
        "job_id": job_id,
    })


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
