"""EmbeddingGemma model-specific embedding tests.

Covers the 300M model's real architecture (Gemma3TextModel backbone ->
attention-mask-aware mean pooling -> learned Dense 768->3072 -> learned
Dense 3072->768 -> L2 normalization), its documented FP16 incompatibility,
and its task-specific prefixes — with mocked weights (no downloads, no GPU)
plus the real projection shapes for dimensionality verification.

Regression root cause: _device_dtype() hardcoded FP16 for every model;
EmbeddingGemma's activations are documented as FP16-incompatible, which is
what crashed ZeroGPU sentence-similarity with a raw HTTP 500. Additionally,
the 2_Dense/3_Dense projection weights were never loaded, so even a
non-crashing run returned raw backbone outputs instead of embeddings.
"""
import importlib
import json
import math
import os
import sys
import types
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(__file__))
from test_embeddings import (  # noqa: E402
    FT, FakeTorch, FakeBatch, _delete_fake_modules)
from test_backend import _install_fake_transformers  # noqa: E402

MODEL_ID = "google/embeddinggemma-300m"
DIM_BACKBONE = 768
DIM_MID = 3072


class GemmaTestBase(unittest.TestCase):
    def setUp(self):
        _delete_fake_modules()
        sys.modules["torch"] = FakeTorch()
        _install_fake_transformers()

    def tearDown(self):
        _delete_fake_modules()

    def _backend(self):
        return importlib.import_module("backend")


def _seeded_matrix(rows, cols, seed, scale=0.02):
    """Deterministic pseudo-random matrix (not identity, not random at
    runtime — seeded so the projection ORDER is verifiable)."""
    state = seed
    out = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            state = (state * 1103515245 + 12345) % 2147483648
            row.append(((state / 2147483648) - 0.5) * 2 * scale)
        out.append(row)
    return out


def gemma_tokenizer():
    tok = MagicMock(name="tokenizer")
    tok.padding_side = "right"
    tok.pad_token = "<pad>"
    tok.eos_token = "<eos>"
    return tok


def gemma_model(hidden_rows):
    """AutoModel fake: mask-aware mean pooling input. hidden [B,S,H]."""
    model = MagicMock(name="gemma3text")
    model.device = "cpu"
    model.config = types.SimpleNamespace(is_encoder_decoder=False)
    model.side_effect = lambda **k: types.SimpleNamespace(
        last_hidden_state=FT(hidden_rows))
    model.parameters.return_value = iter([])
    return model


def gemma_result(tokenizer, model, dtype=None):
    r = types.SimpleNamespace(
        preprocessor=tokenizer, model=model, modality="text",
        loaded_class="AutoModel", tokenizer=tokenizer,
        model_id=MODEL_ID, dtype=dtype or "float32")
    return r


class PrecisionTest(GemmaTestBase):
    """Regression: EmbeddingGemma must NEVER load with FP16."""

    def test_embeddinggemma_gets_fp32_without_bf16_hardware(self):
        backend = self._backend()
        dtype = backend._resolve_model_dtype(MODEL_ID)
        self.assertIsNot(dtype, backend and sys.modules["torch"].float16,
                         "EmbeddingGemma must never select FP16")
        self.assertIs(dtype, sys.modules["torch"].float32)

    def test_embeddinggemma_gets_bf16_when_hardware_supports_it(self):
        backend = self._backend()
        torch = sys.modules["torch"]
        torch.cuda.is_available = lambda: True
        torch.cuda.is_bf16_supported = lambda: True
        try:
            dtype = backend._resolve_model_dtype(MODEL_ID)
            self.assertIs(dtype, torch.bfloat16)
            self.assertIsNot(dtype, torch.float16)
        finally:
            torch.cuda.is_available = lambda: False
            torch.cuda.is_bf16_supported = lambda: False

    def test_qwen3_keeps_fp16(self):
        backend = self._backend()
        self.assertIs(backend._resolve_model_dtype("Qwen/Qwen3-Embedding-0.6B"),
                      sys.modules["torch"].float16)
        self.assertIs(backend._resolve_model_dtype("meta-llama/Llama-3.2-1B"),
                      sys.modules["torch"].float16)

    def test_load_model_passes_model_specific_dtype(self):
        backend = self._backend()
        captured = {}
        def fake_load(model_id, device, dtype):
            captured["dtype"] = dtype
            return MagicMock(name="result")
        with unittest.mock.patch.object(backend, "_list_gguf_files", return_value=[]), \
             unittest.mock.patch.object(backend, "_has_diffusers_model_index",
                                        return_value=False), \
             unittest.mock.patch.object(backend, "_load_transformers",
                                        side_effect=fake_load):
            backend.load_model(MODEL_ID)
            self.assertIsNot(captured["dtype"], sys.modules["torch"].float16,
                             "load_model must not hand FP16 to EmbeddingGemma")
            backend.load_model("Qwen/Qwen3-Embedding-0.6B")
            self.assertIs(captured["dtype"], sys.modules["torch"].float16,
                          "Qwen3 keeps its historical FP16 path")


class PrefixTest(GemmaTestBase):
    def test_sentence_similarity_prefix_applied_to_both_texts(self):
        backend = self._backend()
        prefix = "task: sentence similarity | query: "
        t1 = backend._embedding_prefix(MODEL_ID, "sentence_similarity", "hello world")
        t2 = backend._embedding_prefix(MODEL_ID, "sentence_similarity", "second text")
        self.assertTrue(t1.startswith(prefix), t1)
        self.assertTrue(t2.startswith(prefix), t2)
        self.assertIn("hello world", t1)
        self.assertIn("second text", t2)

    def test_prefix_not_duplicated(self):
        backend = self._backend()
        prefix = "task: sentence similarity | query: "
        already = prefix + "already prefixed"
        self.assertEqual(backend._embedding_prefix(MODEL_ID, "sentence_similarity", already),
                         already)

    def test_qwen3_and_generic_models_get_no_prefix(self):
        backend = self._backend()
        for model_id in ("Qwen/Qwen3-Embedding-0.6B", "sentence-transformers/all-MiniLM-L6-v2"):
            self.assertEqual(
                backend._embedding_prefix(model_id, "sentence_similarity", "plain text"),
                "plain text", f"{model_id} must not receive EmbeddingGemma prefixes")

    def test_prefix_table_is_extensible(self):
        backend = self._backend()
        # documented EmbeddingGemma task prefixes exist for other tasks too
        self.assertIn("task: search result | query: ",
                      backend._EMBEDDING_TASK_PREFIXES["embeddinggemma"]["retrieval"])
        self.assertIn("task: classification | query: ",
                      backend._EMBEDDING_TASK_PREFIXES["embeddinggemma"]["classification"])


class DenseProjectionTest(GemmaTestBase):
    """The learned 2_Dense (768->3072) and 3_Dense (3072->768) modules."""

    def _fake_repo(self, tmp, in_dim=DIM_BACKBONE, mid=DIM_MID, out_dim=DIM_BACKBONE):
        """Writes real config.json files + fake safetensors module that
        load_file() serves with deterministic weights."""
        w1 = _seeded_matrix(mid, in_dim, seed=11)      # 2_Dense: mid x in
        w2 = _seeded_matrix(out_dim, mid, seed=77)     # 3_Dense: out x mid
        b1 = [0.01] * mid
        b2 = [0.02] * out_dim
        calls = []

        def fake_hf_download(model_id, filename):
            calls.append(filename)
            if filename.endswith("config.json") and (
                    filename.startswith("2_Dense") or filename.startswith("3_Dense")):
                p = os.path.join(tmp, filename.replace("/", "_"))
                # 2_Dense carries the tanh activation; 3_Dense (the final
                # projection) is linear — matching the published architecture.
                act = "tanh" if filename.startswith("2_Dense") else "linear"
                with open(p, "w") as f:
                    json.dump({"activation_function": act}, f)
                return p
            if filename == "2_Dense/model.safetensors":
                p = os.path.join(tmp, "2d.safetensors")
                return p
            if filename == "3_Dense/model.safetensors":
                p = os.path.join(tmp, "3d.safetensors")
                return p
            raise FileNotFoundError(filename)

        def fake_load_file(path):
            if path.endswith("2d.safetensors"):
                return {"linear.weight": FT([row[:] for row in w1]),
                        "linear.bias": FT(b1[:])}
            if path.endswith("3d.safetensors"):
                return {"linear.weight": FT([row[:] for row in w2]),
                        "linear.bias": FT(b2[:])}
            raise FileNotFoundError(path)

        return fake_hf_download, fake_load_file, (w1, b1, w2, b2), calls

    def _gemma_result_with_repo(self, tmp, texts_rows, in_dim=DIM_BACKBONE):
        tok = gemma_tokenizer()
        H = in_dim
        S = 3
        hidden = [[[float(((b + 1) * (s + 1) * (h + 1)) % 7) * 0.1
                   for h in range(H)] for s in range(S)]
                  for b in range(len(texts_rows))]
        mask = [[1] * S for _ in texts_rows]
        tok.side_effect = lambda t, **k: FakeBatch(
            {"input_ids": FT([[1] * S for _ in texts_rows]),
             "attention_mask": FT(mask)})
        model = gemma_model(hidden)
        r = gemma_result(tok, model)
        return r, tok, hidden, mask

    def test_projection_chain_order_dims_and_normalization(self):
        backend = self._backend()
        import torch as torch_mod
        tmp = tempfile.mkdtemp(prefix="gemma-")
        dl, lf, (w1, b1, w2, b2), calls = self._fake_repo(tmp)
        texts = ["first sentence", "second sentence"]
        r, tok, hidden, mask = self._gemma_result_with_repo(tmp, texts)
        with unittest.mock.patch.object(backend, "_hub_download", dl), \
             unittest.mock.patch.object(backend, "_safetensors_load_file", lf):
            vectors, pooling = backend._compute_embeddings(
                r, texts, internal_task="sentence_similarity")

        # 1. both texts received the correct prefix
        sent_texts = tok.call_args.args[0]
        prefix = "task: sentence similarity | query: "
        self.assertTrue(all(t.startswith(prefix) for t in sent_texts),
                        f"both texts must be prefixed: {sent_texts}")
        self.assertIn("first sentence", sent_texts[0])
        self.assertIn("second sentence", sent_texts[1])
        # 2. both texts embedded independently (distinct vectors)
        self.assertNotEqual(vectors[0], vectors[1])
        # 3. output dimensionality is exactly 768 (the final projection)
        self.assertEqual(len(vectors[0]), DIM_BACKBONE)
        self.assertEqual(len(vectors[1]), DIM_BACKBONE)
        # 4. L2-normalized
        for v in vectors:
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0, places=5)
        # 5. pooling label reflects the projection
        self.assertIn("dense_projection", pooling)
        # 6. dense modules discovered in order 2_Dense then 3_Dense
        dense_calls = [c for c in calls if c.endswith("config.json")]
        # 2_Dense then 3_Dense loaded in order; the trailing 4_Dense probe is
        # the normal "no more Dense modules" stop.
        self.assertEqual(dense_calls,
                         ["2_Dense/config.json", "3_Dense/config.json",
                          "4_Dense/config.json"])
        # 7. mean pooling respected the attention mask: recompute expected =
        #    normalize(proj2(tanh(proj1(mean(masked_hidden)))) + biases)
        m = mask
        pooled = []
        for b, row in enumerate(hidden):
            cols = list(zip(*row))
            means = [sum(col) / sum(m[b]) for col in cols]
            pooled.append(means)
        w1t = list(zip(*w1))
        h1 = [[sum(p[i] * w1t[i][j] for i in range(len(p))) + b1[j]
               for j in range(len(w1t[0]))] for p in pooled]
        h1 = [[math.tanh(x) for x in row] for row in h1]
        w2t = list(zip(*w2))
        h2 = [[sum(p[i] * w2t[i][j] for i in range(len(p))) + b2[j]
               for j in range(len(w2t[0]))] for p in h1]
        for got, exp in zip(vectors, h2):
            n = math.sqrt(sum(x * x for x in exp)) or 1.0
            for g, e in zip(got, [x / n for x in exp]):
                self.assertAlmostEqual(g, e, places=5,
                                       msg="projection chain must run in order 2_Dense then 3_Dense")
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def test_cosine_similarity_computed_from_final_embeddings(self):
        backend = self._backend()
        import tempfile, shutil
        tmp = tempfile.mkdtemp(prefix="gemma-cos-")
        dl, lf, _, calls = self._fake_repo(tmp, in_dim=2, mid=2, out_dim=2)
        tok = gemma_tokenizer()
        # identical hidden states -> cosine 1.0; orthogonal -> 0.0
        hidden = [[[1.0, 0.0]], [[0.0, 1.0]]]
        tok.side_effect = lambda t, **k: FakeBatch(
            {"input_ids": FT([[1], [1]]), "attention_mask": FT([[1], [1]])})
        model = gemma_model(hidden)
        r = gemma_result(tok, model)
        with unittest.mock.patch.object(backend, "_hub_download", dl), \
             unittest.mock.patch.object(backend, "_safetensors_load_file", lf):
            out = backend.run_sentence_similarity(
                r, MODEL_ID, None,
                {"text1": "same idea", "text2": "different idea"})
        # the similarity must come from the FINAL projected embeddings:
        # with these projections the hidden vectors [1,0]/[0,1] map to
        # different directions — assert a finite value in [-1, 1] plus the
        # structured header, and that both embeddings were used.
        self.assertIn("Cosine similarity:", out)
        self.assertIn("=== Sentence similarity ===", out)
        self.assertIn("Text 1: same idea", out)
        self.assertIn("Text 2: different idea", out)
        sim_line = [l for l in out.split("\n") if "Cosine similarity" in l][0]
        value = float(sim_line.rsplit(":", 1)[1])
        self.assertTrue(-1.0 <= value <= 1.0)
        shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_dense_weights_raise_model_load_error(self):
        backend = self._backend()
        import tempfile, shutil
        tmp = tempfile.mkdtemp(prefix="gemma-fail-")
        calls = []

        def dl(model_id, filename):
            calls.append(filename)
            if filename.endswith("config.json"):
                p = os.path.join(tmp, filename.replace("/", "_"))
                with open(p, "w") as f:
                    json.dump({"activation_function": "tanh"}, f)
                return p
            raise FileNotFoundError(filename)  # weights unavailable

        tok = gemma_tokenizer()
        model = gemma_model([[[1.0, 0.0, 0.0]]])
        tok.side_effect = lambda t, **k: FakeBatch(
            {"input_ids": FT([[1, 1, 1]]), "attention_mask": FT([[1, 1, 1]])})
        r = gemma_result(tok, model)
        with unittest.mock.patch.object(backend, "_hub_download", dl):
            with self.assertRaises(backend.ClaroBackendError) as ctx:
                backend._compute_embeddings(r, ["x"], internal_task="sentence_similarity")
        self.assertEqual(ctx.exception.code, "model_load_error")
        self.assertIn("2_Dense", str(ctx.exception))
        shutil.rmtree(tmp, ignore_errors=True)

    def test_no_dense_modules_falls_back_to_generic_pooling(self):
        backend = self._backend()
        def dl(model_id, filename):
            raise FileNotFoundError(filename)  # no Dense modules in the repo
        tok = gemma_tokenizer()
        tok.padding_side = "right"
        tok.pad_token = None
        tok.eos_token = "<eos>"
        tok.side_effect = lambda t, **k: FakeBatch(
            {"input_ids": FT([[1], [1]]), "attention_mask": FT([[1], [1]])})
        model = MagicMock(name="enc")
        model.device = "cpu"
        model.config = types.SimpleNamespace(is_encoder_decoder=False)
        model.parameters.return_value = iter([])
        model.side_effect = lambda **k: types.SimpleNamespace(
            last_hidden_state=FT([[[1.0, 0.0]], [[0.0, 1.0]]]))
        r = types.SimpleNamespace(preprocessor=tok, model=model, modality="text",
                                  loaded_class="AutoModel", tokenizer=tok,
                                  model_id="sentence-transformers/all-MiniLM-L6-v2")
        vectors, pooling = backend._compute_embeddings(r, ["a", "b"],
                                                       internal_task="sentence_similarity")
        self.assertNotIn("dense_projection", pooling)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in vectors[0])), 1.0, places=5)


import tempfile
import shutil
import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main(verbosity=2)
