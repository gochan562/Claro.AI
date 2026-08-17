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


if __name__ == "__main__":
    unittest.main(verbosity=2)
