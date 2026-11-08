"""Feature-extraction / embedding tests for space/backend.py.

Mocked models/tokenizers — no GPU, no model downloads, no real torch.
The fake torch implements EXACTLY the ops _compute_embeddings uses
(arange, elementwise mul with broadcasting, argmax, sum/clamp, F.normalize,
no_grad) over plain Python lists so the pooling math is deterministic.

Covers:
  - Qwen3-style LAST-TOKEN pooling (causal-LM wrapper, hidden_states[-1])
  - left AND right padding handled via attention-mask-aware positions
  - padded positions (even garbage-valued) never affect the result
  - L2 normalization
  - mean pooling for AutoModel encoders
  - compact output formatting; raw logits/cache objects never appear
  - sentence-similarity cosine values + single-text error
  - unsupported model/task combinations produce useful errors
  - text-generation dispatch unchanged
"""
import contextlib
import importlib
import math
import os
import sys
import types
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(__file__))
from test_backend import _delete_fake_modules, _install_fake_torch  # noqa: E402


# ── minimal list-based fake torch ───────────────────────────────────────────
class FT:
    """Nested-list tensor supporting the ops _compute_embeddings uses."""

    def __init__(self, data, device="cpu", dtype="float32"):
        self.data = data
        self.device = device
        self.dtype = dtype
        self.shape = _shape_of(data)

    def _expand(self, other_data):
        # numpy-style right-aligned broadcast by tiling (covers the three
        # patterns _compute_embeddings uses: [B,S]*[1,S], [B,S,H]*[B,S,1],
        # [B,H]/[B,1])
        a_shape = tuple(self.shape)
        b_shape = _shape_of(other_data)
        while len(b_shape) < len(a_shape):
            other_data = [other_data]
            b_shape = (1,) + tuple(b_shape)

        def rec(b, a_s, b_s):
            if not a_s:
                return b
            if b_s[0] == a_s[0]:
                return [rec(b[i], a_s[1:], b_s[1:]) for i in range(a_s[0])]
            inner = rec(b[0], a_s[1:], b_s[1:])
            return [inner for _ in range(a_s[0])]
        return rec(other_data, a_shape, b_shape)

    def _zip_op(self, other, op):
        b = other.data if isinstance(other, FT) else other
        if not isinstance(b, list):
            b = self._bcast_scalar(b)

        def go(a, bb):
            if isinstance(a, list):
                if not isinstance(bb, list):
                    bb = [bb]
                if len(bb) == 1 and len(a) != 1:
                    bb = bb * len(a)
                return [go(x, y) for x, y in zip(a, bb)]
            return op(a, bb)
        return FT(go(self.data, b), self.device, self.dtype)

    def _bcast_scalar(self, v):
        def build(shape):
            if not shape:
                return v
            return [build(shape[1:]) for _ in range(shape[0])]
        return build(tuple(self.shape))

    def __mul__(self, other):
        b = other.data if isinstance(other, FT) else other
        return FT(_zip_lists(self.data, self._expand(b),
                             lambda x, y: x * y), self.device, self.dtype)

    def __truediv__(self, other):
        b = other.data if isinstance(other, FT) else other
        return FT(_zip_lists(self.data, self._expand(b),
                             lambda x, y: x / y), self.device, self.dtype)

    def sum(self, dim=None, **k):
        if dim == 1:
            if self.data and isinstance(self.data[0], list) and \
                    isinstance(self.data[0][0], list):
                # 3-D [B,S,K] -> [B,K]
                return FT([[sum(col) for col in zip(*row)] for row in self.data],
                          self.device, self.dtype)
            return FT([sum(row) for row in self.data], self.device, self.dtype)
        return FT(sum(_flatten(self.data)), self.device, self.dtype)

    def clamp(self, min=None, **k):
        return FT([[max(min, x) for x in row] for row in self.data],
                  self.device, self.dtype)

    def argmax(self, dim=None, **k):
        if dim is None:
            flat = list(_flatten(self.data))
            return flat.index(max(flat))
        return [row.index(max(row)) for row in self.data]

    def unsqueeze(self, dim):
        if dim == 0:
            return FT([self.data], self.device, self.dtype)
        if dim == -1 or dim == len(self.shape):
            return FT([[x] for x in self.data], self.device, self.dtype) \
                if not isinstance(self.data[0], list) else \
                FT([[[v] for v in row] for row in self.data], self.device, self.dtype)
        raise NotImplementedError(dim)

    def to(self, *a, **k):
        if a and isinstance(a[0], str) and a[0] not in ("cpu",):
            return FT(self.data, a[0], self.dtype)  # device move: keep data
        return FT(self.data, self.device, getattr(a[0], "dtype", self.dtype)
                  if a and not isinstance(a[0], str) else self.dtype)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.data

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            rows, positions = idx
            rows = rows.data if isinstance(rows, FT) else rows
            positions = positions.data if isinstance(positions, FT) else positions
            return FT([self.data[r][p] for r, p in zip(rows, positions)],
                      self.device, self.dtype)
        if isinstance(idx, slice):
            return FT(self.data[idx], self.device, self.dtype)
        return FT(self.data[idx], self.device, self.dtype)

    def float(self):
        return self


class _shape_of_tuple(tuple):
    pass


def _shape_of(data):
    shape = []
    d = data
    while isinstance(d, list):
        shape.append(len(d))
        d = d[0] if d else None
    return tuple(shape)


def _zip_lists(a, b, op):
    if isinstance(a, list):
        if not isinstance(b, list):
            b = [b] * len(a)
        if len(b) == 1 and len(a) != 1:
            b = b * len(a)
        return [_zip_lists(x, y, op) for x, y in zip(a, b)]
    return op(a, b)


def _flatten(x):
    if isinstance(x, list):
        for y in x:
            yield from _flatten(y)
    else:
        yield x


class FakeTorch:
    def __init__(self):
        self.device = "cpu"
        self.cuda = types.SimpleNamespace(is_available=lambda: False,
                                          synchronize=lambda: None)
        self.nn = types.SimpleNamespace(
            functional=types.SimpleNamespace(normalize=self._normalize,
                                             softmax=self._softmax))
        self.softmax = self._softmax  # torch.softmax(...) module-level form
        self.no_grad = contextlib.nullcontext

    def _softmax(self, t, dim=-1, **k):
        def softmax_row(row):
            m = max(row)
            exps = [math.exp(x - m) for x in row]
            s = sum(exps) or 1.0
            return [e / s for e in exps]
        return FT([softmax_row(r) for r in t.data], t.device, t.dtype)

    def arange(self, n, device="cpu", **k):
        return FT(list(range(n)), device)

    def _normalize(self, t, p=2, dim=1, **k):
        assert p == 2 and dim == 1

        def norm_row(row):
            n = math.sqrt(sum(x * x for x in row)) or 1.0
            return [x / n for x in row]
        return FT([norm_row(r) for r in t.data], t.device, t.dtype)


class FakeBatch(dict):
    def to(self, *a, **k):
        return self


def make_tokenizer(vocab_size, mask_rows):
    tok = MagicMock(name="tokenizer")
    tok.padding_side = "right"
    # Qwen-style: no pad token, EOS present (fallback must wire pad=EOS)
    tok.pad_token = None
    tok.eos_token = "<eos>"
    tok.side_effect = lambda texts, padding=None, truncation=None, return_tensors=None: (
        FakeBatch({
            "input_ids": FT([[1] * len(mask_rows[0])] * len(mask_rows)),
            "attention_mask": FT([list(r) for r in mask_rows]),
        }))
    return tok


def make_model(hidden_layers, device="cpu"):
    """hidden_layers: list of 3-D nested lists; out.hidden_states = tuple."""
    model = MagicMock(name="model")
    model.device = device
    model.side_effect = lambda **kwargs: types.SimpleNamespace(
        hidden_states=tuple(FT(h) for h in hidden_layers))
    return model


def make_result(loaded_class, tokenizer, model, modality="text"):
    return types.SimpleNamespace(
        preprocessor=tokenizer, model=model, modality=modality,
        loaded_class=loaded_class, tokenizer=tokenizer)


class EmbeddingTest(unittest.TestCase):
    def setUp(self):
        _delete_fake_modules()
        # replace the shared fake torch with the list-math one
        self._saved = sys.modules.get("torch")
        sys.modules["torch"] = FakeTorch()

    def tearDown(self):
        _delete_fake_modules()
        if self._saved is not None:
            sys.modules["torch"] = self._saved

    def _backend(self):
        return importlib.import_module("backend")

    # ── 1+2+3+4. last-token pooling, left & right padding, mask-aware ──
    def test_last_token_pooling_mask_aware_both_padding_sides(self):
        backend = self._backend()
        B, S, H = 2, 5, 3
        # row 0: LEFT padding (2 pads + 3 valid) — Qwen3 style
        # row 1: RIGHT padding (3 valid + 2 pads)
        mask_rows = [[0, 0, 1, 1, 1], [1, 1, 1, 0, 0]]

        def layer(bias):
            return [[[float((s + 1) * 10 + (h + 1) + bias) for h in range(H)]
                     for s in range(S)] for b in range(B)]

        # Garbage (huge) values in PADDED positions — must never leak in.
        layer0 = layer(0)
        layer0[0][0] = [9e9] * H
        layer0[0][1] = [-9e9] * H
        layer0[1][3] = [9e9] * H
        layer0[1][4] = [-9e9] * H
        layer1 = layer(1000)  # distinct "last layer" values

        tok = make_tokenizer(16, mask_rows)
        model = make_model([layer0, layer1])
        r = make_result("AutoModelForCausalLM", tok, model)

        vectors, pooling = backend._compute_embeddings(r, ["a", "b"])
        self.assertEqual(pooling, "last_token")
        self.assertEqual(tok.padding_side, "left",
                         "Qwen3-style causal wrapper must switch to left padding")
        # official Qwen3 example: pad with EOS when the tokenizer has no pad token
        self.assertIs(tok.pad_token, tok.eos_token,
                      "missing pad_token must default to eos_token")
        # layer1 (hidden_states[-1]) values, last VALID position per row:
        # row 0 last valid s=4, row 1 last valid s=2
        expected0 = layer1[0][4]
        expected1 = layer1[1][2]
        for got, exp in ((vectors[0], expected0), (vectors[1], expected1)):
            n = math.sqrt(sum(x * x for x in exp))
            exp_norm = [x / n for x in exp]
            for g, e in zip(got, exp_norm):
                self.assertAlmostEqual(g, e, places=6)
            norm = math.sqrt(sum(x * x for x in got))
            self.assertAlmostEqual(norm, 1.0, places=6,
                                   msg="embedding must be L2-normalized")
        # 9e9 garbage must not appear anywhere
        for v in vectors:
            for x in v:
                self.assertLess(abs(x), 10.0)

    # ── 5. extraction from hidden_states[-1] (causal-LM wrapper) ──
    def test_uses_hidden_states_minus_one_not_last_hidden_state(self):
        backend = self._backend()
        S, H = 3, 2
        last_layer = [[[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]]]  # last valid s=2
        tok = make_tokenizer(8, [[1, 1, 1]])
        model = MagicMock(name="causallm")
        model.device = "cpu"

        seen = {}
        def fwd(**kwargs):
            seen.update(kwargs)
            return types.SimpleNamespace(
                logits=MagicMock(name="logits-should-be-ignored"),
                hidden_states=(FT([[[9.0, 9.0], [9.0, 9.0], [9.0, 9.0]]]),
                               FT(last_layer)))
        model.side_effect = fwd
        r = make_result("AutoModelForCausalLM", tok, model)
        vectors, _ = backend._compute_embeddings(r, ["x"])
        # normalized [3,4] → [0.6, 0.8]; the 9.0 decoy layer is NOT used
        self.assertAlmostEqual(vectors[0][0], 0.6, places=6)
        self.assertAlmostEqual(vectors[0][1], 0.8, places=6)
        self.assertTrue(seen.get("output_hidden_states"),
                        "must request hidden states from the causal-LM wrapper")

    # ── 6. mean pooling for AutoModel encoders ──
    def test_mean_pooling_for_automodel_with_mask(self):
        backend = self._backend()
        # right-padded row: 2 valid tokens + 1 garbage pad
        last = [[[3.0, 0.0], [0.0, 4.0], [9e9, -9e9]]]
        mask = [[1, 1, 0]]
        tok = make_tokenizer(8, mask)
        model = MagicMock(name="automodel")
        model.device = "cpu"
        model.side_effect = lambda **k: types.SimpleNamespace(
            hidden_states=(FT(last),), last_hidden_state=FT(last))
        r = make_result("AutoModel", tok, model)
        vectors, pooling = backend._compute_embeddings(r, ["x"])
        self.assertEqual(pooling, "mean")
        # mask-weighted mean = ([3,0]+[0,4]) / 2 = [1.5, 2.0]
        n = math.sqrt(1.5 ** 2 + 2.0 ** 2)
        self.assertAlmostEqual(vectors[0][0], 1.5 / n, places=6)
        self.assertAlmostEqual(vectors[0][1], 2.0 / n, places=6)

    # ── 7. compact formatting; no raw torch objects in the response ──
    def test_format_compact_no_raw_objects(self):
        backend = self._backend()
        # L2-normalized 1024-dim vector (as produced by _compute_embeddings)
        vec = [0.6, 0.8] + [0.0] * 1022
        out = backend._format_embedding_result(
            "Qwen/Qwen3-Embedding-0.6B", [vec], "last_token")
        self.assertIn("=== Embedding output ===", out)
        self.assertIn("Model: Qwen/Qwen3-Embedding-0.6B", out)
        self.assertIn("Dimensions: 1024", out)
        self.assertIn("Norm: 1.0000", out)
        self.assertIn("Preview: [0.6000, 0.8000, 0.0000, 0.0000", out)
        self.assertIn("...", out)
        self.assertIn("Pooling: last_token", out)
        # must stay compact: preview capped, not thousands of values
        self.assertLess(len(out), 800, f"output too long: {len(out)} chars")
        for banned in ("logits", "past_key_values", "CausalLMOutput",
                       "BaseModelOutput", "tensor(", "last_hidden_state"):
            self.assertNotIn(banned, out, f"raw output object leaked: {banned}")

    # ── 8. sentence-similarity: cosine values, not a lone embedding ──
    def test_sentence_similarity_cosine_and_single_text_error(self):
        backend = self._backend()
        # anchor [1,0], candidates [1,0] (sim 1.0) and [0,1] (sim 0.0)
        def hidden_for(texts):
            # one ROW per text: B=len(texts), S=1 token, H=2
            vecs = {"a": [1.0, 0.0], "b": [2.0, 0.0], "c": [0.0, 5.0]}
            return [[vecs[t]] for t in texts]
        tok = MagicMock(name="tokenizer")
        tok.padding_side = "left"

        def tok_side(texts, **k):
            n = len(texts)
            return FakeBatch({"input_ids": FT([[1] for _ in range(n)]),
                              "attention_mask": FT([[1] for _ in range(n)])})
        tok.side_effect = tok_side
        model = MagicMock(name="model")
        model.device = "cpu"
        model.side_effect = lambda **k: types.SimpleNamespace(
            hidden_states=(FT(hidden_for(["a", "b", "c"])),))
        r = make_result("AutoModelForCausalLM", tok, model)

        out = backend._embedding_inference(r, "owner/embed", "a\nb\nc",
                                           "sentence-similarity")
        self.assertIn("Similarity(anchor, text 1): 1.0000", out)
        self.assertIn("Similarity(anchor, text 2): 0.0000", out)
        self.assertIn("Vectors: 3", out)

        with self.assertRaises(backend.ClaroBackendError) as ctx:
            backend._embedding_inference(r, "owner/embed", "only one line",
                                         "sentence-similarity")
        self.assertEqual(ctx.exception.code, "bad_request")
        self.assertIn("not a similarity score", ctx.exception.message
                      if hasattr(ctx.exception, 'message') else str(ctx.exception))

    # ── 9. unsupported combinations return useful errors ──
    def test_unsupported_combinations(self):
        backend = self._backend()
        # diffusion model
        tok = make_tokenizer(8, [[1, 1]])
        r = make_result("DiffusersPipeline", tok, MagicMock(), modality="diffusion")
        with self.assertRaises(backend.ClaroBackendError) as ctx:
            backend._embedding_inference(r, "owner/img", "x", "feature-extraction")
        self.assertEqual(ctx.exception.code, "task_model_mismatch")
        # classifier class has no embedding contract
        r2 = make_result("AutoModelForSequenceClassification",
                         make_tokenizer(8, [[1, 1]]), MagicMock(name="m"))
        with self.assertRaises(backend.ClaroBackendError) as ctx2:
            backend._embedding_inference(r2, "owner/cls", "x", "feature-extraction")
        self.assertEqual(ctx2.exception.code, "task_model_mismatch")
        self.assertIn("AutoModelForSequenceClassification", str(ctx2.exception))
        # missing tokenizer
        r3 = make_result("AutoModelForCausalLM", None, MagicMock(name="m"))
        with self.assertRaises(backend.ClaroBackendError):
            backend._embedding_inference(r3, "owner/x", "x", "feature-extraction")

    # ── 10. run_inference dispatch: embedding branch + unchanged generation ──
    def test_run_inference_dispatch(self):
        backend = self._backend()
        tok = make_tokenizer(8, [[1, 1, 1]])
        H = 2
        last = [[[3.0, 4.0], [0.0, 0.0], [0.0, 0.0]]]
        mask = [[1, 0, 0]]  # left padding: valid token at position 0
        tok.side_effect = lambda t, **k: FakeBatch(
            {"input_ids": FT([[1] * 3]), "attention_mask": FT(mask)})
        model = MagicMock(name="m")
        model.device = "cpu"
        model.side_effect = lambda **k: types.SimpleNamespace(hidden_states=(FT(last),))
        r = make_result("AutoModelForCausalLM", tok, model)

        with unittest.mock.patch.object(backend, "load_model", return_value=r), \
             unittest.mock.patch.object(backend, "_to_gpu", return_value=0.0), \
             unittest.mock.patch.object(backend, "_offload_gpu", lambda res: None):
            out = backend.run_inference("owner/embed", "feature-extraction",
                                        "hello", max_new_tokens=8)
            self.assertIn("=== Embedding output ===", out)
            self.assertIn("Dimensions: 2", out)
            # normalized [3,4] = [0.6, 0.8]; pad position 1/2 must be ignored
            self.assertIn("Preview: [0.6000, 0.8000", out)

            # sentence-similarity routes through the same branch
            model.side_effect = lambda **k: types.SimpleNamespace(
                hidden_states=(FT([[[1.0, 0.0]], [[0.0, 1.0]]]),))  # B=2, S=1
            tok.side_effect = lambda t, **k: FakeBatch(
                {"input_ids": FT([[1], [1]]), "attention_mask": FT([[1], [1]])})
            out2 = backend.run_inference("owner/embed", "sentence-similarity",
                                         "a\nb", max_new_tokens=8)
            self.assertIn("=== Sentence similarity ===", out2)
            self.assertIn("Cosine similarity: 0.0000", out2)

            # text-generation behavior UNCHANGED: still routed to _generate_text
            with unittest.mock.patch.object(
                    backend, "_generate_text",
                    return_value=("generated text", 5)) as gen:
                out3 = backend.run_inference("owner/embed", "text-generation",
                                             "once upon a time", max_new_tokens=8)
                self.assertEqual(out3, "generated text")
                self.assertTrue(gen.called)
                self.assertEqual(gen.call_args.kwargs.get("prompt_format"), "auto")


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main(verbosity=2)
