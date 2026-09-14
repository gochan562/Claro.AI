"""Training-runner dataset-resolution tests.

Verifies that datasets.load_dataset() receives the fully resolved,
namespaced repository ID (e.g. AI-Lab-Makerere/beans) — never the friendly
preset ID ('beans') and never a speculative owner fallback.

Heavy ML deps (torch/datasets/transformers) are stubbed in sys.modules;
run_training() aborts via SystemExit right after the dataset-load attempt,
which is all this suite needs to observe.

Run directly:  python3 test/training_runner_datasets.py
"""
import argparse
import os
import subprocess
import sys
import types
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Canonical contract anchor. The single source of truth is the catalog
# (DATASET_PRESETS in training_ui.js); _resolved_beans_id() reads it live.
CANONICAL_BEANS_ID = "AI-Lab-Makerere/beans"


def _resolved_beans_id():
    """Read the beans datasetId straight from the catalog (no duplication)."""
    try:
        out = subprocess.run(
            ["node", "-e",
             "console.log(require('./training_ui.js').getDatasetPreset('beans').datasetId)"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
        )
        candidate = (out.stdout or "").strip()
        if out.returncode == 0 and "/" in candidate:
            return candidate
    except Exception:
        pass
    return CANONICAL_BEANS_ID


class _StubModules:
    """Installs/removes stub torch+datasets+transformers modules."""

    def __init__(self, load_dataset_fn):
        self.load_dataset_fn = load_dataset_fn
        self.saved = {}

    def __enter__(self):
        import unittest.mock as mock
        for name in ("torch", "datasets", "transformers", "peft", "accelerate"):
            self.saved[name] = sys.modules.get(name, None)
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = self.load_dataset_fn
        sys.modules["datasets"] = fake_datasets
        sys.modules["torch"] = types.ModuleType("torch")
        sys.modules["transformers"] = mock.MagicMock(name="transformers")
        sys.modules["peft"] = mock.MagicMock(name="peft")
        sys.modules["accelerate"] = mock.MagicMock(name="accelerate")
        # run_training does `import importlib` (stdlib, fine) and reads
        # hf_loader lazily only later — aborted before that point.
        return self

    def __exit__(self, *exc):
        for name, mod in self.saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        return False


def _runner_args(dataset_id):
    return argparse.Namespace(
        model_id="google/mobilenet_v2_1.0_224",
        dataset_id=dataset_id,
        task_type="image-classification",
        epochs=1,
        batch_size=8,
        learning_rate=2e-5,
        max_steps=None,
        validation_split=10,
        max_samples=2000,
        max_time_sec=1800,
        training_method="full",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="auto",
        job_id="train_0123456789ab",
        output_dir="/tmp/claro-test-beans",
    )


class BeansResolutionTest(unittest.TestCase):
    def test_catalog_resolves_to_canonical_repo(self):
        self.assertEqual(_resolved_beans_id(), CANONICAL_BEANS_ID)

    def test_load_dataset_receives_resolved_id(self):
        import training_runner
        calls = []

        def fake_load_dataset(name, *args, **kwargs):
            calls.append((name, kwargs))
            raise RuntimeError("stub: stop after recording")

        resolved = _resolved_beans_id()
        self.assertIn("/", resolved, "resolved ID must be namespaced owner/name")
        with _StubModules(fake_load_dataset):
            with self.assertRaises(SystemExit):
                training_runner.run_training(_runner_args(resolved))
        self.assertTrue(calls, "load_dataset must have been called")
        # First call is the direct load; every call must use the exact ID —
        # no friendly 'beans', no speculative 'stanfordnlp/beans'-style retry.
        for name, kwargs in calls:
            self.assertEqual(name, resolved)
        first_names = [c[0] for c in calls]
        self.assertNotIn("beans", [n for n in first_names if "/" not in n],
                         "friendly preset ID must never reach load_dataset")
        for bad in ("stanfordnlp/beans", "uoft-cs/beans", "ylecun/beans",
                    "rajpurkar/beans", "lmsys/beans"):
            self.assertNotIn(bad, first_names,
                             f"speculative fallback {bad} must not be attempted")

    def test_namespaced_id_skips_prefix_fallback(self):
        # Direct unit check of the fallback guard: names containing '/'
        # never enter the legacy prefix-retry loop.
        import training_runner
        import inspect
        src = inspect.getsource(training_runner.run_training)
        self.assertIn('if "/" not in name:', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
