"""Offline tests for padding, protocol binding and state cache validation."""

import copy
from dataclasses import asdict
import importlib.util
import json
from contextlib import nullcontext
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from dual_concept.backbone import (
    FrozenBackbone, FrozenBackboneConfig, StateCache, assert_compatible_protocol,
    checkpoint_stat_fingerprint, last_nonpadding_indices, protocol_hash, render_chat_input,
)


class PaddingTests(unittest.TestCase):
    def test_actual_last_one_for_both_padding_sides(self):
        mask = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [1, 0, 1, 0]])
        np.testing.assert_array_equal(last_nonpadding_indices(mask), [1, 3, 2, 2])

    def test_empty_or_invalid_mask_rejected(self):
        for mask in (np.array([[0, 0]]), np.array([[1, 2]]), np.array([1, 1]), np.empty((1, 0))):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                last_nonpadding_indices(mask)

    def test_config_rejects_invalid_depth_and_size(self):
        for options in ({"candidate_layers": (0, 1)}, {"candidate_layers": (1, 1)},
                        {"candidate_layers": ()}, {"batch_size": 0}, {"dtype": "int8"},
                        {"max_length": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                FrozenBackboneConfig(**options)

    def test_protocol_not_available_before_extraction(self):
        with self.assertRaises(RuntimeError):
            FrozenBackbone().get_protocol()

    def test_template_options_are_strict_and_cannot_override_input_conventions(self):
        invalid = [None, [], (), "{}", False, {"enable_thinking": 0},
                   {"enable_thinking": "false"}, {"enable_thinking": None},
                   {"enable_thinking": []}, {"tokenize": False},
                   {"add_generation_prompt": False}, {"tools": []}, {1: True}]
        for options in invalid:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "chat_template_kwargs"):
                FrozenBackboneConfig(chat_template_kwargs=options)

    def test_template_options_are_copied_immutable_and_json_serializable(self):
        source = {"enable_thinking": False}
        config = FrozenBackboneConfig(chat_template_kwargs=source)
        source["enable_thinking"] = True
        self.assertEqual(config.chat_template_kwargs, {"enable_thinking": False})
        self.assertEqual(FrozenBackboneConfig().chat_template_kwargs, {})
        with self.assertRaises(TypeError):
            config.chat_template_kwargs["enable_thinking"] = True
        with self.assertRaises(TypeError):
            config.chat_template_kwargs.update(enable_thinking=True)
        serialized = json.loads(json.dumps(asdict(config)))
        self.assertEqual(serialized["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(FrozenBackboneConfig(**serialized).chat_template_kwargs, config.chat_template_kwargs)
        self.assertEqual(copy.deepcopy(config), config)

    def test_shared_renderer_preserves_fixed_user_message_and_non_thinking_switch(self):
        calls = []
        tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs:
                                    calls.append((messages, kwargs)) or "formatted")
        self.assertEqual(render_chat_input(tokenizer, "request", {"enable_thinking": False}), "formatted")
        self.assertEqual(calls, [([{"role": "user", "content": "request"}],
                                {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False})])
        for text in ("", " ", None, 1):
            with self.subTest(text=text), self.assertRaises(ValueError):
                render_chat_input(tokenizer, text)
        with self.assertRaisesRegex(ValueError, "chat_template_kwargs"):
            render_chat_input(tokenizer, "request", {"add_generation_prompt": False})

    def test_invalid_text_rejected_before_loading_dependencies(self):
        backbone = FrozenBackbone()
        with self.assertRaises(TypeError):
            backbone.extract("one string")
        with self.assertRaises(ValueError):
            backbone.extract([""])


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.cache = StateCache(self.folder.name)
        self.protocol = {"candidate_layers": [1, 3], "hidden_size": 4, "revision": "fixed"}
        self.states = np.arange(8, dtype=np.float32).reshape(2, 4)

    def test_roundtrip_and_safe_hashed_filename(self):
        ident = "../../outside/危险"
        path = self.cache.save(ident, "request", self.protocol, self.states)
        self.assertEqual(path.parent, Path(self.folder.name))
        self.assertEqual(len(path.stem), 64)
        actual = self.cache.load(ident, "request", self.protocol)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, self.states)
        self.assertEqual(list(Path(self.folder.name).iterdir()), [path])

    def test_cache_miss(self):
        self.assertIsNone(self.cache.load("absent", "request", self.protocol))

    def test_text_and_protocol_changes_are_rejected(self):
        self.cache.save("id", "request", self.protocol, self.states)
        with self.assertRaisesRegex(ValueError, "text"):
            self.cache.load("id", "changed", self.protocol)
        with self.assertRaisesRegex(ValueError, "protocol"):
            self.cache.load("id", "request", {**self.protocol, "revision": "other"})

    def test_invalid_states_and_shape_rejected(self):
        for states in (np.zeros((1, 4)), np.full((2, 4), np.nan), np.full((2, 4), np.inf),
                       np.zeros((2, 4), dtype=np.complex64), np.zeros(8)):
            with self.subTest(shape=states.shape), self.assertRaises(ValueError):
                self.cache.save("id", "request", self.protocol, states)
        self.cache.save("valid", "request", self.protocol, self.states)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.cache.load("valid", "request", self.protocol, (3, 4))

    def test_immutable_cache(self):
        path = self.cache.save("id", "request", self.protocol, self.states)
        self.assertEqual(path, self.cache.save("id", "request", self.protocol, self.states))
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.cache.save("id", "request", self.protocol, self.states + 1)

    def test_malformed_cache_rejected(self):
        path = self.cache.path_for("id")
        np.savez_compressed(path, states=self.states)
        with self.assertRaises(ValueError):
            self.cache.load("id", "request", self.protocol)

    def test_protocol_hash_is_order_independent(self):
        self.assertEqual(protocol_hash({"a": 1, "b": 2}), protocol_hash({"b": 2, "a": 1}))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.weight = self.root / "model-00001-of-00001.safetensors"
        self.weight.write_bytes(b"small offline checkpoint fixture")
        self.index = self.root / "model.safetensors.index.json"
        self.index.write_text('{"weight_map":{"weight":"model-00001-of-00001.safetensors"}}')

    def protocol(self):
        return {
            "schema_version": 2, "model_name_or_path": str(self.root), "revision": "branch",
            "resolved_revision": None, "checkpoint": checkpoint_stat_fingerprint(self.root),
            "model_config_sha256": "config-hash", "tokenizer_sha256": "tokenizer-hash",
            "chat_template_sha256": "template-hash", "chat_template_kwargs": {},
            "versions": {"torch": "fixture", "numpy": "fixture"},
            "parameters": {"model_name_or_path": str(self.root), "revision": "branch", "device": "cpu",
                           "dtype": "float16", "batch_size": 1, "local_files_only": True,
                           "candidate_layers": [1, 3], "max_length": 2048, "chat_template_kwargs": {}},
            "resolved_device": "cpu", "candidate_layers": [1, 3], "num_hidden_layers": 4,
            "hidden_size": 16, "decoder_class": "FixtureDecoder", "padding_side": "right",
            "truncation": "reject", "add_generation_prompt": True, "add_special_tokens": False,
            "position_ids": "attention_mask_cumsum_minus_one_with_padding_zero",
            "state_convention": "last_actual_one;exclude_embedding_and_final_norm",
            "output_dtype": "float32", "generates_answers": False,
        }

    def test_stat_fingerprint_stable_and_changed_mtime_is_rejected(self):
        expected = self.protocol()
        self.assertEqual(expected["checkpoint"], checkpoint_stat_fingerprint(self.root))
        self.assertEqual(expected["checkpoint"]["kind"], "local_weight_stat")
        self.assertEqual(expected["checkpoint"]["files"][0]["size"], self.weight.stat().st_size)
        old_stat = self.weight.stat()
        os.utime(self.weight, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000))
        actual = self.protocol()
        self.assertNotEqual(expected["checkpoint"]["stat_manifest_sha256"],
                            actual["checkpoint"]["stat_manifest_sha256"])
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            assert_compatible_protocol(expected, actual)

    def test_shard_index_content_changes_fingerprint(self):
        before = checkpoint_stat_fingerprint(self.root)
        self.index.write_text('{"weight_map":{"other":"model-00001-of-00001.safetensors"}}')
        after = checkpoint_stat_fingerprint(self.root)
        self.assertEqual(before["files"], after["files"])
        self.assertNotEqual(before["index_files_sha256"], after["index_files_sha256"])
        self.assertNotEqual(before["stat_manifest_sha256"], after["stat_manifest_sha256"])

    def test_runtime_location_batch_device_and_layer_changes_are_compatible(self):
        expected = self.protocol()
        actual = copy.deepcopy(expected)
        for section in (actual, actual["parameters"]):
            section.update(model_name_or_path="relocated/model", revision="different-alias", device="cuda",
                           batch_size=8, candidate_layers=[2], local_files_only=False)
        actual["resolved_device"] = "cuda"
        self.assertIsNone(assert_compatible_protocol(expected, actual))
        self.assertNotEqual(protocol_hash(expected), protocol_hash(actual))

    def test_dtype_template_checkpoint_and_other_semantic_changes_are_rejected(self):
        expected = self.protocol()
        changes = {
            "dtype": lambda p: p["parameters"].update(dtype="bfloat16"),
            "template": lambda p: p.update(chat_template_sha256="different-template"),
            "template kwargs": lambda p: (p.update(chat_template_kwargs={"enable_thinking": False}),
                                           p["parameters"].update(chat_template_kwargs={"enable_thinking": False})),
            "checkpoint": lambda p: p["checkpoint"].update(stat_manifest_sha256="different-checkpoint"),
            "tokenizer": lambda p: p.update(tokenizer_sha256="different-tokenizer"),
            "versions": lambda p: p["versions"].update(torch="other-version"),
            "position": lambda p: p.update(position_ids="different-position-rule"),
            "truncation": lambda p: p.update(truncation="left"),
            "length": lambda p: p["parameters"].update(max_length=1024),
        }
        for label, change in changes.items():
            with self.subTest(change=label):
                actual = copy.deepcopy(expected)
                change(actual)
                with self.assertRaises(ValueError):
                    assert_compatible_protocol(expected, actual)

    def test_legacy_protocol_without_weight_identity_is_rejected(self):
        legacy = self.protocol()
        del legacy["checkpoint"]
        with self.assertRaisesRegex(ValueError, "identity"):
            assert_compatible_protocol(legacy, self.protocol())

    def test_legacy_protocol_without_template_options_is_rejected(self):
        for section in ("top", "parameters"):
            legacy = self.protocol()
            del (legacy if section == "top" else legacy["parameters"])["chat_template_kwargs"]
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, "identity"):
                assert_compatible_protocol(legacy, self.protocol())

    def test_conflicting_template_options_are_rejected_and_bind_cache_identity(self):
        original = self.protocol()
        changed = copy.deepcopy(original)
        changed["chat_template_kwargs"] = {"enable_thinking": False}
        with self.assertRaisesRegex(ValueError, "conflicting"):
            assert_compatible_protocol(changed, changed)
        changed["parameters"]["chat_template_kwargs"] = {"enable_thinking": False}
        self.assertNotEqual(protocol_hash(original), protocol_hash(changed))
        cache = StateCache(self.root / "states")
        cache.save("id", "request", original, np.zeros((2, 16), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "protocol"):
            cache.load("id", "request", changed)


class _ArrayTensor(np.ndarray):
    """Tiny tensor adapter for exercising extraction without optional packages."""

    def __new__(cls, value):
        return np.asarray(value).view(cls)

    device = "cpu"

    def to(self, device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def long(self):
        return _ArrayTensor(self.astype(np.int64))

    def float(self):
        return _ArrayTensor(self.astype(np.float32))

    def masked_fill(self, mask, value):
        result = self.copy()
        result[mask] = value
        return result


class ExtractionTests(unittest.TestCase):
    def reader(self, side="right", template_kwargs=None):
        class Tokenizer:
            padding_side = side

            def apply_chat_template(self, messages, **kwargs):
                self.last_template_kwargs = kwargs
                return "user " + messages[0]["content"] + " assistant"

            def __call__(self, rendered, **kwargs):
                self.last_tokenizer_kwargs = kwargs
                lengths = [len(text.split()) for text in rendered]
                ids = np.zeros((len(lengths), max(lengths)), np.int64)
                for i, length in enumerate(lengths):
                    if self.padding_side == "right":
                        ids[i, :length] = 1
                    else:
                        ids[i, -length:] = 1
                return {"input_ids": _ArrayTensor(ids), "attention_mask": _ArrayTensor(ids)}

        class Decoder:
            calls = 0

            def parameters(self):
                return iter([SimpleNamespace(device="cpu")])

            def __call__(self, **kwargs):
                self.calls += 1
                self.last_kwargs = kwargs
                position = kwargs["position_ids"]
                values = np.stack((position, position), axis=-1)
                return SimpleNamespace(hidden_states=tuple(_ArrayTensor(values + 100 * depth) for depth in range(5)))

        class Reader(FrozenBackbone):
            def _load(self):
                if self._protocol is not None:
                    return
                self._tokenizer = Tokenizer()
                self._decoder = Decoder()
                self._hidden_size = 2
                self._torch = SimpleNamespace(inference_mode=nullcontext,
                    arange=lambda n, **kw: np.arange(n), as_tensor=lambda a, **kw: _ArrayTensor(a),
                    stack=lambda a, dim: _ArrayTensor(np.stack(a, axis=dim)))
                self._protocol = {"num_hidden_layers": 4, "candidate_layers": [1, 3], "hidden_size": 2}

        return Reader(FrozenBackboneConfig(candidate_layers=(1, 3), batch_size=2, max_length=8,
                                          chat_template_kwargs={} if template_kwargs is None else template_kwargs))

    def test_extraction_selects_last_valid_token_and_physical_depths(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                reader = self.reader(side)
                states = reader.extract(["hello", "hello world", "one two three"])
                np.testing.assert_array_equal(states[:, :, 0], [[102, 302], [103, 303], [104, 304]])
                self.assertEqual(reader._decoder.calls, 2)
                self.assertFalse(reader._decoder.last_kwargs["use_cache"])
                self.assertEqual(reader._tokenizer.last_template_kwargs,
                                 {"tokenize": False, "add_generation_prompt": True})
                self.assertFalse(reader._tokenizer.last_tokenizer_kwargs["add_special_tokens"])

    def test_overlength_rejected_before_forward_and_protocol_is_a_copy(self):
        reader = self.reader()
        reader.extract([])
        protocol = reader.get_protocol()
        protocol["candidate_layers"][0] = 0
        self.assertEqual(reader.get_protocol()["candidate_layers"], [1, 3])
        with self.assertRaisesRegex(ValueError, "max_length"):
            reader.extract(["one two three four five six seven"])
        self.assertEqual(reader._decoder.calls, 0)

    def test_extraction_uses_shared_renderer_with_explicit_non_thinking_mode(self):
        reader = self.reader(template_kwargs={"enable_thinking": False})
        self.assertEqual(reader.extract(["hello"]).shape, (1, 2, 2))
        self.assertEqual(reader._tokenizer.last_template_kwargs,
                         {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False})


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "optional torch/transformers dependencies are not installed")
class TinyQwenTests(unittest.TestCase):
    def test_random_local_qwen_without_downloads(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

        with tempfile.TemporaryDirectory() as folder:
            backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "user": 3,
                                          "assistant": 4, "hello": 5, "world": 6}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                                pad_token="[PAD]", eos_token="[EOS]")
            tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' [EOS] ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
            tokenizer.save_pretrained(folder)
            torch.manual_seed(7)
            model = Qwen2ForCausalLM(Qwen2Config(vocab_size=7, hidden_size=16, intermediate_size=32,
                                                num_hidden_layers=4, num_attention_heads=2,
                                                num_key_value_heads=2, max_position_embeddings=64))
            model.save_pretrained(folder)
            config = FrozenBackboneConfig(model_name_or_path=folder, device="cpu", dtype="float32",
                                          local_files_only=True, candidate_layers=(1, 3), batch_size=2, max_length=32)
            reader = FrozenBackbone(config)
            values = reader.extract(["hello", "hello world"])
            self.assertEqual(values.shape, (2, 2, 16))
            self.assertEqual(values.dtype, np.float32)
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(all(not p.requires_grad for p in reader._model.parameters()))
            self.assertFalse(reader._model.training)
            np.testing.assert_allclose(values[0], reader.extract(["hello"])[0], atol=1e-6, rtol=1e-5)
            reader._tokenizer.padding_side = "left"
            np.testing.assert_allclose(values, reader.extract(["hello", "hello world"]), atol=1e-6, rtol=1e-5)
            self.assertEqual(reader.get_protocol()["candidate_layers"], [1, 3])
            self.assertEqual(reader.get_protocol()["chat_template_kwargs"], {})
            self.assertEqual(reader.extract([]).shape, (0, 2, 16))
            with self.assertRaisesRegex(ValueError, "max_length"):
                reader.extract([" ".join(["hello"] * 50)])
            with self.assertRaisesRegex(ValueError, "candidate_layers"):
                FrozenBackbone(FrozenBackboneConfig(model_name_or_path=folder, device="cpu", dtype="float32",
                                                    local_files_only=True, candidate_layers=(4,))).extract([])


if __name__ == "__main__":
    unittest.main()
