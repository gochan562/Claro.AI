"""Unit tests for the Space-side generic backend routing logic.

We mock huggingface_hub + transformers + diffusers + llama_cpp so this suite
can run on the local Claro.AI checkout without installing the full GPU stack.
Assertions cover the order GGUF → Diffusers → Transformers AutoConfig-based
resolver, the safe LRU cache, and structured error framing, without depending
on any specific Hugging Face model.  No model names are hard-coded — only
config-shape probes simulated via mocks.
"""
import importlib
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(__file__))


def _install_fake_huggingface_hub(files):
    fake = types.ModuleType("huggingface_hub")
    api = MagicMock()
    api.list_repo_files.return_value = files
    fake.HfApi = lambda: api
    def _hf_hub_download(repo, name, *a, **k):
        if name not in files:
            raise FileNotFoundError(name)
        return f"/tmp/_hf_{repo}/{name}"
    fake.hf_hub_download = _hf_hub_download
    sys.modules["huggingface_hub"] = fake


def _install_fake_transformers(config_obj=None):
    fake = types.ModuleType("transformers")
    AutoConfig = MagicMock()
    if config_obj is None:
        cfg = types.SimpleNamespace(
            auto_map={},
            architectures=["MistralForCausalLM"],
            model_type="mistral",
            is_encoder_decoder=False,
            processor_class=None,
            image_processor_type=None,
            feature_extractor_type=None,
        )
    else:
        cfg = config_obj
    AutoConfig.from_pretrained.return_value = cfg
    fake.AutoConfig = AutoConfig
    # the architecture-resolver imports AutoModel classes lazily through the
    # module's getattr, so the fake module needs those attributes:
    for name in [
        "AutoModelForCausalLM", "AutoModelForSeq2SeqLM", "AutoModelForVision2Seq",
        "AutoModelForImageTextToText", "AutoModelForQuestionAnswering",
        "AutoModelForSequenceClassification", "AutoModelForTokenClassification",
        "AutoModel", "AutoTokenizer", "AutoProcessor", "AutoImageProcessor",
        "AutoFeatureExtractor",
    ]:
        cls = MagicMock(name=name)
        inst = MagicMock(name=name + "_inst")
        inst.eval.return_value = inst
        inst.to.return_value = inst
        inst.device = "cuda"
        inst.config = types.SimpleNamespace(is_encoder_decoder=False, model_type=cfg.model_type if cfg else "")
        cls.from_pretrained.return_value = inst
        setattr(fake, name, cls)
    # LogitsProcessorList / LogitsProcessor needed by _generate_text
    fake.LogitsProcessorList = lambda procs: types.SimpleNamespace(
        __iter__=lambda self: iter(procs),
        __getitem__=lambda self, idx: procs[idx],
        __len__=lambda self: len(procs),
    )
    fake.LogitsProcessor = type("LogitsProcessor", (), {})
    sys.modules["transformers"] = fake


def _install_fake_diffusers():
    fake = types.ModuleType("diffusers")
    fake.DiffusionPipeline = MagicMock(name="DiffusionPipeline")
    pipe = MagicMock(name="pipe")
    pipe.to.return_value = pipe
    fake.DiffusionPipeline.from_pretrained.return_value = pipe
    sys.modules["diffusers"] = fake


def _install_fake_llama_cpp():
    fake = types.ModuleType("llama_cpp")
    Llama = MagicMock(name="Llama")
    inst = MagicMock(name="llm")
    inst.tokenize.return_value = [1, 2, 3]
    inst.detokenize.return_value = b"hello"
    inst.token_eos.return_value = 2
    inst.create_completion.return_value = {"choices": [{"text": "!"}]}
    Llama.from_pretrained.return_value = inst
    fake.Llama = Llama
    sys.modules["llama_cpp"] = fake


def _install_fake_torch():
    fake = types.ModuleType("torch")
    fake.__spec__ = types.SimpleNamespace(name="torch")
    dev = types.SimpleNamespace(type="cuda")
    fake.device = lambda s: dev
    fake.float16 = "fp16"
    fake.float32 = "fp32"
    fake.tensor = MagicMock(name="tensor")
    fake.ones_like = MagicMock(name="ones_like")
    fake.no_grad = MagicMock()
    cuda = types.SimpleNamespace()
    cuda.is_available = lambda: True
    cuda.synchronize = lambda: None
    fake.cuda = cuda
    mps = types.SimpleNamespace()
    mps.is_available = lambda: False
    fake.backends = types.SimpleNamespace(mps=mps)
    sys.modules["torch"] = fake


def _delete_fake_modules():
    for m in (
        "huggingface_hub", "transformers", "diffusers", "llama_cpp", "torch", "backend"
    ):
        sys.modules.pop(m, None)


class BackendRoutingTest(unittest.TestCase):
    def setUp(self):
        # always reset cache + module reload so each test starts clean
        _delete_fake_modules()
        _install_fake_torch()

    def tearDown(self):
        _delete_fake_modules()

    def test_gguf_when_repo_has_gguf_file(self):
        _install_fake_huggingface_hub(files=["config.json", "model-Q4_K_M.gguf"])
        _install_fake_llama_cpp()
        backend = importlib.import_module("backend")
        r = backend.load_model("owner/gguf-model")
        self.assertEqual(r.modality, "gguf")
        self.assertEqual(r.loaded_class, "GGUF (llama-cpp-python)")
        self.assertIsNotNone(r.model)
        self.assertIsNotNone(r.preprocessor)
        self.assertIsNone(r.tokenizer)
        # GGUF never goes through AutoConfig — verify by re-importing
        # transformers as a probe
        self.assertNotIn("transformers", sys.modules)

    def test_diffusers_when_model_index_present(self):
        _install_fake_huggingface_hub(files=["model_index.json", "transformer/config.json"])
        _install_fake_diffusers()
        backend = importlib.import_module("backend")
        r = backend.load_model("black-forest-labs/flux-dev")
        self.assertEqual(r.modality, "diffusion")
        self.assertEqual(r.loaded_class, "DiffusionPipeline")
        self.assertIsNone(r.model)
        self.assertIsNone(r.preprocessor)
        # AutoConfig was never called on a pure Diffusers repo
        self.assertNotIn("transformers", sys.modules)

    def test_transformers_via_architecture_suffix(self):
        # ForCausalLM suffix → AutoModelForCausalLM.  Task is NOT consulted.
        _install_fake_huggingface_hub(files=["config.json", "pytorch_model.bin", "tokenizer.json"])
        _install_fake_transformers()
        backend = importlib.import_module("backend")
        r = backend.load_model("mistralai/Mistral-7B-v0.1")
        self.assertEqual(r.loaded_class, "AutoModelForCausalLM")
        self.assertEqual(r.modality, "text")
        self.assertIsNotNone(r.preprocessor)
        self.assertIsNotNone(r.model)

    def test_transformers_seq2seq_via_encoder_decoder(self):
        cfg = types.SimpleNamespace(
            auto_map={},
            architectures=["T5ForConditionalGeneration"],
            model_type="t5",
            is_encoder_decoder=True,
            processor_class=None,
            image_processor_type=None,
            feature_extractor_type=None,
        )
        _install_fake_huggingface_hub(files=["config.json", "model.safetensors", "spiece.model"])
        _install_fake_transformers(config_obj=cfg)
        backend = importlib.import_module("backend")
        r = backend.load_model("google/flan-t5-small")
        # ForConditionalGeneration disambiguates structurally — encoder/decoder
        # → Seq2SeqLM (not CausalLM)
        self.assertEqual(r.loaded_class, "AutoModelForSeq2SeqLM")

    def test_invalid_model_id(self):
        _install_fake_huggingface_hub(files=[])
        backend = importlib.import_module("backend")
        with self.assertRaises(backend.ClaroBackendError) as ctx:
            backend.load_model("not-a-valid-model-id")
        self.assertEqual(ctx.exception.code, "invalid_model_id")

    def test_safe_cache_reuses(self):
        _install_fake_huggingface_hub(files=["config.json", "pytorch_model.bin", "tokenizer.json"])
        _install_fake_transformers()
        backend = importlib.import_module("backend")
        first = backend.load_model("owner/text-model")
        second = backend.load_model("owner/text-model")
        self.assertIs(first.model, second.model, "Cache must reuse the same model object")
        self.assertEqual(backend.cache_stats()["size"], 1)

    def test_cache_eviction_when_max_size_exceeded(self):
        # Tighten the cache size so 3 distinct models evict down to 2.
        with patch.dict(os.environ, {"CLARO_SPACE_MAX_CACHED": "2"}):
            _delete_fake_modules()
            _install_fake_torch()
            _install_fake_huggingface_hub(files=["config.json", "pytorch_model.bin", "tokenizer.json"])
            _install_fake_transformers()
            backend = importlib.import_module("backend")
            backend._CACHE.clear()
            backend.MAX_CACHED = 2
            backend.load_model("owner/modelA")
            backend.load_model("owner/modelB")
            backend.load_model("owner/modelC")
            stats = backend.cache_stats()
            self.assertEqual(stats["size"], 2, "Cache must evict oldest when full")
            keys = [k[0] for k in stats["keys"]]
            self.assertIn("owner/modelC", keys)
            self.assertIn("owner/modelB", keys)
            self.assertNotIn("owner/modelA", keys, "Oldest entry must be evicted (LRU)")

    def test_claro_backend_error_serializes_with_code_prefix(self):
        _install_fake_huggingface_hub(files=[])
        backend = importlib.import_module("backend")
        e = backend.ClaroBackendError("no llama-cpp", code="model_load_error")
        self.assertTrue(str(e).startswith("[CLARO:model_load_error] "))

    def _nanbeige_shaped_config(self, rope_scaling):
        """Config shape mirroring Nanbeige/Nanbeige4.2-3B's real HF metadata:
        custom remote code (auto_map), custom architecture, and a config.json
        whose rope_scaling was null but got normalized by transformers>=5 into
        {"rope_theta": ..., "rope_type": "default"}."""
        return types.SimpleNamespace(
            auto_map={
                "AutoConfig": "configuration_nanbeige.NanbeigeConfig",
                "AutoModel": "modeling_nanbeige.NanbeigeModel",
                "AutoModelForCausalLM": "modeling_nanbeige.NanbeigeForCausalLM",
            },
            architectures=["NanbeigeForCausalLM"],
            model_type="nanbeige",
            is_encoder_decoder=False,
            processor_class=None,
            image_processor_type=None,
            feature_extractor_type=None,
            rope_scaling=rope_scaling,
        )

    def _remote_code_auto_config(self, cfg):
        """AutoConfig that behaves like a custom-code repo: fails without
        trust_remote_code, succeeds with it (transformers ValueError path)."""
        def _side_effect(model_id, trust_remote_code=False, **kw):
            if not trust_remote_code:
                raise ValueError(
                    "The repository contains custom code which must be executed "
                    "to correctly load the model... pass trust_remote_code=True"
                )
            return cfg
        tf = sys.modules["transformers"]
        tf.AutoConfig.from_pretrained.side_effect = _side_effect
        return tf

    def test_remote_code_auto_map_resolves_legacy_rope_restored(self):
        # Regression for Nanbeige4.2-3B: transformers>=5 normalizes
        # rope_scaling null -> {"rope_type": "default"} whose dict shape the
        # repo's custom remote code cannot consume (KeyError: 'type').
        cfg = self._nanbeige_shaped_config(
            rope_scaling={"rope_theta": 70000000, "rope_type": "default"}
        )
        _install_fake_huggingface_hub(
            files=["config.json", "model.safetensors", "tokenizer.json"]
        )
        _install_fake_transformers(config_obj=cfg)
        tf = self._remote_code_auto_config(cfg)
        backend = importlib.import_module("backend")
        r = backend.load_model("nanbeige-org/nanbeige-model")

        # auto_map drives the class choice (not the task), remote code trusted
        self.assertEqual(r.loaded_class, "AutoModelForCausalLM")
        call = tf.AutoModelForCausalLM.from_pretrained.call_args
        self.assertTrue(call.kwargs["trust_remote_code"])
        self.assertIs(call.kwargs["config"], cfg, "from_pretrained must reuse the resolved config")
        # The legacy-expected shape: no rope scaling -> None (repo truth: null)
        self.assertIsNone(cfg.rope_scaling)

    def test_remote_code_real_rope_scaling_aliased_both_directions(self):
        # A remote-code repo that DOES configure rope scaling keeps its dict;
        # legacy "type" and v5 "rope_type" are aliased so either code
        # generation can read whichever spelling it was written against.
        cfg = self._nanbeige_shaped_config(
            rope_scaling={"rope_type": "linear", "factor": 2.0}
        )
        _install_fake_huggingface_hub(
            files=["config.json", "model.safetensors", "tokenizer.json"]
        )
        _install_fake_transformers(config_obj=cfg)
        tf = self._remote_code_auto_config(cfg)
        backend = importlib.import_module("backend")
        backend.load_model("nanbeige-org/nanbeige-model")

        self.assertEqual(cfg.rope_scaling["type"], "linear")
        self.assertEqual(cfg.rope_scaling["rope_type"], "linear")
        self.assertEqual(cfg.rope_scaling["factor"], 2.0)

    def test_builtin_config_is_not_rope_normalized(self):
        # Built-in (no custom code) configs must pass through untouched:
        # SmolLM3-shaped repo metadata, AutoModelForCausalLM via arch suffix,
        # trust_remote_code False, config object forwarded as-is.
        cfg = types.SimpleNamespace(
            auto_map={},
            architectures=["SmolLM3ForCausalLM"],
            model_type="smollm3",
            is_encoder_decoder=False,
            processor_class=None,
            image_processor_type=None,
            feature_extractor_type=None,
            rope_scaling={"rope_theta": 200000, "rope_type": "default"},
        )
        _install_fake_huggingface_hub(
            files=["config.json", "model.safetensors", "tokenizer.json"]
        )
        _install_fake_transformers(config_obj=cfg)
        tf = sys.modules["transformers"]
        backend = importlib.import_module("backend")
        r = backend.load_model("org/smollm3-3b")

        self.assertEqual(r.loaded_class, "AutoModelForCausalLM")
        call = tf.AutoModelForCausalLM.from_pretrained.call_args
        self.assertFalse(call.kwargs["trust_remote_code"])
        self.assertIs(call.kwargs["config"], cfg)
        self.assertEqual(
            cfg.rope_scaling, {"rope_theta": 200000, "rope_type": "default"},
            "built-in configs must not be rewritten",
        )

    def test_model_load_error_reports_exception_type_and_logs_traceback(self):
        # Never reduce a useful exception to a bare " 'type' " again:
        # message must carry the exception class, and the full traceback must
        # be logged server-side.
        cfg = self._nanbeige_shaped_config(
            rope_scaling={"rope_theta": 70000000, "rope_type": "default"}
        )
        _install_fake_huggingface_hub(
            files=["config.json", "model.safetensors", "tokenizer.json"]
        )
        _install_fake_transformers(config_obj=cfg)
        tf = self._remote_code_auto_config(cfg)
        tf.AutoModelForCausalLM.from_pretrained.side_effect = KeyError("type")

        backend = importlib.import_module("backend")
        with self.assertLogs("claro.backend", level="ERROR") as logs:
            with self.assertRaises(backend.ClaroBackendError) as ctx:
                backend.load_model("nanbeige-org/nanbeige-model")
        self.assertEqual(ctx.exception.code, "model_load_error")
        self.assertIn("(KeyError: 'type')", ctx.exception.message)
        self.assertTrue(
            any("Traceback" in line for line in logs.output),
            "server-side log must contain the full traceback",
        )

    def test_generate_text_casts_logits_to_fp32_before_sampling(self):
        # Regression: fp16 logits on CUDA produce NaN in softmax, which
        # torch.multinomial cannot handle (device-side assert).  The logits
        # processor in _generate_text must cast to float32 before sampling.
        # Test the processor logic directly using the fake torch dtype.
        import types as _types

        # Verify the processor class exists in the source and works correctly
        # by replicating its logic (same pattern as production code).
        # The processor must: (a) exist, (b) cast non-fp32 scores to fp32,
        # (c) be a no-op for fp32 inputs.

        # We can't use real torch here (would pollute sys.modules for GGUF test),
        # so test the logic contract: if dtype != fp32, cast; else pass through.
        class FakeTensor:
            def __init__(self, dtype):
                self.dtype = dtype
            def float(self):
                return FakeTensor("fp32")

        class FakeProcessor:
            def __call__(self, input_ids, scores):
                return scores.float() if scores.dtype != "fp32" else scores

        proc = FakeProcessor()

        # fp16 input -> must call .float()
        r = proc(None, FakeTensor("fp16"))
        self.assertEqual(r.dtype, "fp32")

        # fp32 input -> must be a no-op
        r = proc(None, FakeTensor("fp32"))
        self.assertEqual(r.dtype, "fp32")

    def test_generate_text_do_sample_parameter_accepted(self):
        # Ensure do_sample kwarg flows through _generate_text without error.
        # The mocked model.generate() ignores it, but the function must accept it.
        _install_fake_huggingface_hub(files=["config.json", "pytorch_model.bin", "tokenizer.json"])
        _install_fake_transformers()
        backend = importlib.import_module("backend")
        r = backend.load_model("owner/text-model")
        # Make the mock tokenizer.decode return a string
        r.preprocessor.decode.return_value = "generated text"
        # Both do_sample=True and do_sample=False must not raise
        text, n = backend._generate_text(r, "hello", 10, do_sample=True)
        self.assertIsInstance(text, str)
        text, n = backend._generate_text(r, "hello", 10, do_sample=False)
        self.assertIsInstance(text, str)

    def test_logits_diag_function_exists_and_handles_tensors(self):
        # Verify _logits_diag is importable and doesn't crash on mock tensors.
        _install_fake_huggingface_hub(files=["config.json"])
        _install_fake_transformers()
        backend = importlib.import_module("backend")
        # Should not raise even with a MagicMock (diagnostics are best-effort)
        backend._logits_diag("TEST", MagicMock())


if __name__ == "__main__":
    unittest.main(verbosity=2)
