"""Frozen decoder state extraction and protocol-bound, per-record caches.

Importing this module requires only NumPy. PyTorch and Transformers are loaded
when ``extract`` is first called. Hidden-state indices are physical decoder
depths: index zero is the embedding and the final returned entry is post-norm;
neither is a candidate feature. No answer tokens are generated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"), default=str).encode("utf-8")


class _FrozenTemplateKwargs(dict):
    """A JSON-object mapping that stays immutable inside a frozen config.

    A dict subclass keeps ``dataclasses.asdict`` and JSON serialization working
    while preventing a caller from changing prompt semantics after loading.
    """

    def _reject_mutation(self, *args, **kwargs):
        raise TypeError("chat_template_kwargs is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _reject_mutation

    def __copy__(self):
        return type(self)(self)

    def __deepcopy__(self, memo):
        return type(self)(self)


def _chat_template_kwargs(value: Any) -> dict[str, bool]:
    """Accept only explicitly supported JSON template variables."""
    if not isinstance(value, dict):
        raise ValueError("chat_template_kwargs must be a JSON object")
    unsupported = set(value).difference({"enable_thinking"})
    if unsupported:
        raise ValueError("Unsupported chat_template_kwargs keys: " + ", ".join(sorted(map(str, unsupported))))
    if "enable_thinking" in value and type(value["enable_thinking"]) is not bool:
        raise ValueError("chat_template_kwargs.enable_thinking must be a JSON boolean")
    return dict(value)


def render_chat_input(tokenizer: Any, text: str,
                      chat_template_kwargs: Mapping[str, Any] | None = None) -> str:
    """Render the identical input for token-length audits and extraction.

    Only the supported template variables can vary. The single user message,
    generation prefix and two-step tokenization convention remain fixed.
    This formats input tokens only and never generates an answer.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Each input must be a nonempty string")
    kwargs = _chat_template_kwargs({} if chat_template_kwargs is None else chat_template_kwargs)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, **kwargs)
    if not isinstance(rendered, str):
        raise ValueError("Chat template must render a string with tokenize=False")
    return rendered


def protocol_hash(protocol: Mapping[str, Any]) -> str:
    """Hash the complete extraction protocol, independently of key order."""
    return hashlib.sha256(_json_bytes(dict(protocol))).hexdigest()


def checkpoint_stat_fingerprint(model_dir: str | Path) -> dict[str, Any]:
    """Identify local weight files by relative name, size and ``mtime_ns``.

    The manifest SHA256 hashes file *stat metadata*, not weight contents. This
    inexpensive identity check intentionally rejects replaced/retouched shards;
    it is not a cryptographic verification of multi-gigabyte model weights.
    Shard-index JSON contents are separately hashed because they map names to
    tensors. Model config and tokenizer contents have their own protocol hashes.
    """
    root = Path(model_dir)
    if not root.is_dir():
        raise ValueError("A local model directory is required for a stat fingerprint")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".safetensors", ".bin"}:
            stat = path.stat()
            files.append({"path": path.relative_to(root).as_posix(),
                          "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if not files:
        raise ValueError("Local checkpoint has no .safetensors or .bin weight files")
    indexes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(root.rglob("*.index.json")) if path.is_file()}
    manifest = {"files": files, "index_files_sha256": indexes}
    return {"kind": "local_weight_stat", **manifest,
            "stat_manifest_sha256": hashlib.sha256(_json_bytes(manifest)).hexdigest()}


def _portable_model_config(value: Any) -> Any:
    """Remove loading locations and separately recorded commit IDs from config."""
    if isinstance(value, Mapping):
        return {key: _portable_model_config(item) for key, item in value.items()
                if key not in {"_name_or_path", "name_or_path", "_commit_hash"}}
    if isinstance(value, (list, tuple)):
        return [_portable_model_config(item) for item in value]
    return value


def assert_compatible_protocol(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    """Require identical feature semantics and checkpoint identity for inference.

    Loading locations/revision aliases, device, batch size, local-download
    policy and requested feature depths may change. All remaining protocol
    fields, including resolved checkpoint identity, dtype, tokenizer, position
    rules, input-length policy and package versions, must match exactly. This
    is more permissive than cache hashes, which bind the entire protocol.
    """
    required = {"schema_version", "checkpoint", "model_config_sha256", "tokenizer_sha256",
                "chat_template_sha256", "chat_template_kwargs", "versions", "parameters", "num_hidden_layers", "hidden_size",
                "decoder_class", "padding_side", "truncation", "add_generation_prompt",
                "add_special_tokens", "position_ids", "state_convention", "output_dtype", "generates_answers"}
    runtime_keys = {"model_name_or_path", "revision", "device", "resolved_device", "batch_size",
                    "local_files_only", "candidate_layers"}

    def semantic(protocol):
        missing = required.difference(protocol)
        if missing:
            raise ValueError("Extraction protocol lacks required identity fields: " + ", ".join(sorted(missing)))
        if protocol["schema_version"] != 2:
            raise ValueError("Checkpoint-bound extraction protocol schema 2 is required")
        template_kwargs = _chat_template_kwargs(protocol["chat_template_kwargs"])
        if "chat_template_kwargs" not in protocol["parameters"]:
            raise ValueError("Extraction protocol parameters lack chat_template_kwargs identity")
        if _chat_template_kwargs(protocol["parameters"]["chat_template_kwargs"]) != template_kwargs:
            raise ValueError("Extraction protocol has conflicting chat_template_kwargs identity")
        result = {key: value for key, value in protocol.items() if key not in runtime_keys}
        result["parameters"] = {key: value for key, value in protocol["parameters"].items()
                                if key not in runtime_keys}
        return result

    left, right = semantic(expected), semantic(actual)
    changed = [key for key in sorted(set(left) | set(right))
               if key not in left or key not in right or _json_bytes(left[key]) != _json_bytes(right[key])]
    if changed:
        raise ValueError("Incompatible extraction protocol fields: " + ", ".join(changed))


def last_nonpadding_indices(attention_mask: np.ndarray) -> np.ndarray:
    """Return the actual final one in each mask row, for either padding side."""
    mask = np.asarray(attention_mask)
    if mask.ndim != 2 or mask.shape[1] == 0:
        raise ValueError("attention_mask must be a nonempty-width 2D array")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("attention_mask must contain only zero and one")
    positions = np.where(mask == 1, np.arange(mask.shape[1])[None, :], -1)
    indices = positions.max(axis=1)
    if (indices < 0).any():
        raise ValueError("Every input must contain at least one nonpadding token")
    return indices.astype(np.int64)


@dataclass(frozen=True)
class FrozenBackboneConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: str | None = None
    device: str = "auto"
    dtype: str = "float16"
    max_length: int = 2048
    batch_size: int = 1
    local_files_only: bool = False
    candidate_layers: tuple[int, ...] = tuple(range(1, 28))
    chat_template_kwargs: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path must not be empty")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16 or float32")
        for name in ("max_length", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        layers = tuple(self.candidate_layers)
        if not layers or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in layers):
            raise ValueError("candidate_layers must contain positive integer decoder depths")
        if len(set(layers)) != len(layers):
            raise ValueError("candidate_layers must not contain duplicates")
        object.__setattr__(self, "candidate_layers", layers)
        object.__setattr__(self, "chat_template_kwargs",
                           _FrozenTemplateKwargs(_chat_template_kwargs(self.chat_template_kwargs)))


class FrozenBackbone:
    """Read selected last-input-token states from a frozen causal decoder.

    ``max_length`` rejects overlong formatted inputs instead of silently
    truncating a request or its generation prefix. ``get_protocol`` becomes
    available after ``extract``; ``extract([])`` loads without a forward pass.
    """

    def __init__(self, config: FrozenBackboneConfig | None = None) -> None:
        self.config = config or FrozenBackboneConfig()
        self._model = self._decoder = self._tokenizer = self._torch = None
        self._protocol: dict[str, Any] | None = None
        self._hidden_size = 0

    def _load(self) -> None:
        if self._protocol is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("State extraction requires the optional torch and transformers dependencies") from exc
        cfg = self.config
        common = {"revision": cfg.revision, "local_files_only": cfg.local_files_only,
                  "trust_remote_code": False}
        local = Path(cfg.model_name_or_path).is_dir()
        checkpoint = checkpoint_stat_fingerprint(cfg.model_name_or_path) if local else None
        # Resolve the model first. The tokenizer must use that same immutable
        # Hub commit rather than independently resolving a moving branch.
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path, torch_dtype=getattr(torch, cfg.dtype), **common)
        resolved_revision = getattr(model.config, "_commit_hash", None)
        if not local:
            if not resolved_revision:
                raise ValueError("Remote model did not expose an immutable resolved commit")
            common["revision"] = resolved_revision
            checkpoint = {"kind": "hub_commit", "commit": resolved_revision}
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, **common)
        tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        if not local and tokenizer_commit and tokenizer_commit != resolved_revision:
            raise ValueError("Model and tokenizer resolved to different Hub commits")
        if local and checkpoint != checkpoint_stat_fingerprint(cfg.model_name_or_path):
            raise ValueError("Local checkpoint weight files changed during loading")
        if not tokenizer.chat_template:
            raise ValueError("The tokenizer must define a chat template")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("The tokenizer requires a pad token or an EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        device = cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        depth = int(model.config.num_hidden_layers)
        if any(layer >= depth for layer in cfg.candidate_layers):
            raise ValueError(f"candidate_layers must lie in 1..{depth - 1}; exclude embeddings and final normalized state")
        decoder = None
        if callable(getattr(model, "get_decoder", None)):
            decoder = model.get_decoder()
        if decoder is None or decoder is model:
            decoder = getattr(model, "model", None)
        if decoder is None or decoder is model:
            decoder = getattr(model, "base_model", None)
        if decoder is None or decoder is model:
            raise ValueError("Cannot identify a decoder backbone without the language-model head")
        model.requires_grad_(False)
        model.eval()
        model.to(device)
        decoder.eval()
        self._model, self._decoder, self._tokenizer, self._torch = model, decoder, tokenizer, torch
        self._hidden_size = int(model.config.hidden_size)
        versions = {}
        for package in ("torch", "transformers", "numpy"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unknown"
        tokenizer_signature = {"class": type(tokenizer).__name__, "vocabulary": tokenizer.get_vocab(),
                               "special_tokens": tokenizer.special_tokens_map,
                               "chat_template": tokenizer.chat_template}
        backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
        if backend_tokenizer is not None:
            # Include normalizers, pre-tokenizers and AddedToken rules, which a
            # vocabulary-only fingerprint would otherwise miss.
            tokenizer_signature["backend_tokenizer"] = backend_tokenizer.to_str()
        self._protocol = {
            "schema_version": 2, "model_name_or_path": cfg.model_name_or_path,
            "revision": cfg.revision, "resolved_revision": resolved_revision, "checkpoint": checkpoint,
            "model_config_sha256": hashlib.sha256(_json_bytes(_portable_model_config(model.config.to_dict()))).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(_json_bytes(tokenizer_signature)).hexdigest(),
            "chat_template_sha256": hashlib.sha256(_json_bytes(tokenizer.chat_template)).hexdigest(),
            "chat_template_kwargs": dict(cfg.chat_template_kwargs),
            "versions": versions, "parameters": asdict(cfg), "resolved_device": str(device),
            "candidate_layers": list(cfg.candidate_layers), "num_hidden_layers": depth,
            "hidden_size": self._hidden_size, "decoder_class": type(decoder).__name__,
            "padding_side": tokenizer.padding_side, "truncation": "reject",
            "add_generation_prompt": True, "add_special_tokens": False,
            "position_ids": "attention_mask_cumsum_minus_one_with_padding_zero",
            "state_convention": "last_actual_one_in_attention_mask;hidden_states[physical_depth];exclude_embedding_and_final_norm",
            "output_dtype": "float32", "generates_answers": False,
        }

    def get_protocol(self) -> dict[str, Any]:
        if self._protocol is None:
            raise RuntimeError("Call extract first; extract([]) loads the protocol without a forward pass")
        return json.loads(_json_bytes(self._protocol))

    def extract(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not one string")
        texts = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Each input must be a nonempty string")
        self._load()
        cfg, torch, tokenizer = self.config, self._torch, self._tokenizer
        if not texts:
            return np.empty((0, len(cfg.candidate_layers), self._hidden_size), dtype=np.float32)
        device = next(self._decoder.parameters()).device
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), cfg.batch_size):
                rendered = [render_chat_input(tokenizer, text, cfg.chat_template_kwargs)
                    for text in texts[start:start + cfg.batch_size]]
                encoded = tokenizer(rendered, padding=True, truncation=False, add_special_tokens=False,
                                    return_attention_mask=True, return_tensors="pt")
                mask = encoded["attention_mask"]
                indices = last_nonpadding_indices(mask.detach().cpu().numpy())
                if encoded["input_ids"].shape[1] > cfg.max_length:
                    raise ValueError(f"Formatted input exceeds max_length={cfg.max_length}; inputs are not truncated")
                inputs = {key: encoded[key].to(device) for key in ("input_ids", "attention_mask")}
                positions = inputs["attention_mask"].long().cumsum(-1) - 1
                inputs["position_ids"] = positions.masked_fill(inputs["attention_mask"] == 0, 0)
                output = self._decoder(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
                hidden = output.hidden_states
                if hidden is None or len(hidden) != int(self._protocol["num_hidden_layers"]) + 1:
                    raise ValueError("Unexpected decoder hidden_states convention; expected embedding plus one entry per layer")
                batch_indices = torch.arange(len(indices), device=device)
                token_indices = torch.as_tensor(indices, device=device)
                selected = torch.stack([hidden[layer][batch_indices, token_indices]
                                        for layer in cfg.candidate_layers], dim=1)
                array = selected.float().cpu().numpy()
                if array.shape != (len(indices), len(cfg.candidate_layers), self._hidden_size) or not np.isfinite(array).all():
                    raise ValueError("Malformed or nonfinite extracted states")
                batches.append(array)
        return np.concatenate(batches, axis=0).astype(np.float32, copy=False)


class StateCache:
    """Immutable per-ID NPZ states, bound to both request and protocol hashes."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, record_id: str) -> Path:
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a nonempty string")
        return self.cache_dir / (hashlib.sha256(record_id.encode("utf-8")).hexdigest() + ".npz")

    @staticmethod
    def _validate(states: np.ndarray, protocol: Mapping[str, Any], expected_shape=None) -> np.ndarray:
        array = np.asarray(states)
        if array.ndim != 2 or min(array.shape) < 1 or array.dtype.kind not in "fiu":
            raise ValueError("Cached states must be a nonempty numeric [layers, hidden_size] matrix")
        if expected_shape is None and "candidate_layers" in protocol and "hidden_size" in protocol:
            expected_shape = (len(protocol["candidate_layers"]), int(protocol["hidden_size"]))
        if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
            raise ValueError("Cached states shape differs from the extraction protocol")
        array = array.astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Cached states must be finite float32 values")
        return array

    def load(self, record_id: str, text: str, protocol: Mapping[str, Any],
             expected_shape: tuple[int, int] | None = None) -> np.ndarray | None:
        path = self.path_for(record_id)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["id"].item()) != record_id:
                    raise ValueError("Cached record ID differs")
                if str(data["text_sha256"].item()) != hashlib.sha256(text.encode("utf-8")).hexdigest():
                    raise ValueError("Cached request text differs")
                if str(data["protocol_sha256"].item()) != protocol_hash(protocol):
                    raise ValueError("Cached extraction protocol differs")
                return self._validate(data["states"], protocol, expected_shape)
        except (KeyError, OSError, EOFError) as exc:
            raise ValueError(f"Malformed state cache: {path.name}") from exc

    def save(self, record_id: str, text: str, protocol: Mapping[str, Any], states: np.ndarray) -> Path:
        path = self.path_for(record_id)
        array = self._validate(states, protocol)
        if path.exists():
            existing = self.load(record_id, text, protocol, array.shape)
            if not np.array_equal(existing, array):
                raise ValueError("Refusing to overwrite an existing record with different states")
            return path
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, prefix=".state-", suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, id=np.asarray(record_id),
                                    text_sha256=np.asarray(hashlib.sha256(text.encode("utf-8")).hexdigest()),
                                    protocol_sha256=np.asarray(protocol_hash(protocol)), states=array)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path


def extract_to_cache(records: Sequence[Mapping[str, str]], config: FrozenBackboneConfig,
                     cache_dir: str | Path) -> np.ndarray:
    """Extract ``{'id': ..., 'text': ...}`` records, reusing verified row caches."""
    records = list(records)
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Record IDs must be unique")
    backbone = FrozenBackbone(config)
    empty = backbone.extract([])
    protocol = backbone.get_protocol()
    cache = StateCache(cache_dir)
    result: list[np.ndarray | None] = [cache.load(record["id"], record["text"], protocol) for record in records]
    pending = [i for i, value in enumerate(result) if value is None]
    for start in range(0, len(pending), config.batch_size):
        indices = pending[start:start + config.batch_size]
        states = backbone.extract([records[i]["text"] for i in indices])
        for i, state in zip(indices, states):
            cache.save(records[i]["id"], records[i]["text"], protocol, state)
            result[i] = state
    return np.stack(result).astype(np.float32) if result else empty
