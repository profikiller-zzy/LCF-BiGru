"""Train-only concepts, internal attack holdouts, sparse fusion and frozen models."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .concepts import ConceptBank, concept_scores, fit_concepts
from .data import Dataset, canonical_family, sha256_file
from .evaluation import calibrate_threshold
from .fusion import FusionConfig, SparseFusionModel, build_feature_matrix, fit_sparse_fusion
from .selection import SelectionConfig, select_layers


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
                      default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item.item()).encode("utf-8") + b"\n"


@dataclass(frozen=True)
class MethodConfig:
    target_fpr: float = 0.05
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    min_internal_families: int = 2

    def __post_init__(self) -> None:
        if not np.isfinite(self.target_fpr) or not 0 <= self.target_fpr < 1:
            raise ValueError("target_fpr must lie in [0,1)")
        if type(self.min_internal_families) is not int or self.min_internal_families < 2:
            raise ValueError("At least two attack families are required for internal holdouts")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MethodConfig":
        payload = dict(value)
        payload["selection"] = SelectionConfig(**payload.get("selection", {}))
        payload["fusion"] = FusionConfig(**payload.get("fusion", {}))
        return cls(**payload)


@dataclass
class Detector:
    bank: ConceptBank
    risk_layers: list[int]
    manipulation_layers: list[int]
    fusion: SparseFusionModel
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def decision_function(self, states: np.ndarray) -> np.ndarray:
        risk, manipulation = concept_scores(states, self.bank)
        features, physical_layers, _ = build_feature_matrix(
            risk, manipulation, self.risk_layers, self.manipulation_layers, self.bank.layer_ids)
        if not np.array_equal(physical_layers, self.fusion.feature_layers):
            raise ValueError("Saved feature ordering disagrees with fusion parameters")
        return self.fusion.decision_function(features)

    def predict(self, states: np.ndarray) -> np.ndarray:
        return self.decision_function(states) >= self.threshold

    def save(self, output: str | Path) -> None:
        destination = Path(output)
        if destination.exists():
            raise FileExistsError(f"Choose a new output directory; refusing to overwrite {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        try:
            arrays = {f"bank_{name}": array for name, array in self.bank.arrays().items()}
            arrays.update({f"fusion_{name}": array for name, array in self.fusion.arrays().items()})
            np.savez_compressed(temporary / "parameters.npz", **arrays)
            payload = {"schema_version": 1, "method": "dual_concept_complementary_cross_layer",
                       "score_domain": "logit", "decision_rule": ">=", "threshold": float(self.threshold),
                       "risk_layers": self.risk_layers, "manipulation_layers": self.manipulation_layers,
                       "parameters_sha256": sha256_file(temporary / "parameters.npz"), "metadata": self.metadata}
            (temporary / "model.json").write_bytes(json_bytes(payload))
            os.replace(temporary, destination)
        except BaseException:
            import shutil
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    @classmethod
    def load(cls, directory: str | Path) -> "Detector":
        root = Path(directory)
        payload = json.loads((root / "model.json").read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("score_domain") != "logit" or payload.get("decision_rule") != ">=":
            raise ValueError("Unsupported frozen detector format")
        if sha256_file(root / "parameters.npz") != payload["parameters_sha256"]:
            raise ValueError("Frozen numeric parameters have changed")
        with np.load(root / "parameters.npz", allow_pickle=False) as arrays:
            bank = ConceptBank(**{name.removeprefix("bank_"): arrays[name] for name in arrays.files if name.startswith("bank_")})
            fusion = SparseFusionModel(**{name.removeprefix("fusion_"): arrays[name] for name in arrays.files if name.startswith("fusion_")})
        threshold = payload["threshold"]
        if not np.isfinite(threshold):
            raise ValueError("Frozen threshold must be finite")
        result = cls(bank, payload["risk_layers"], payload["manipulation_layers"], fusion, threshold, payload["metadata"])
        features = build_feature_matrix(np.zeros((1, len(bank.layer_ids))), np.zeros((1, len(bank.layer_ids))),
                                        result.risk_layers, result.manipulation_layers, bank.layer_ids)[1]
        if not np.array_equal(features, fusion.feature_layers):
            raise ValueError("Frozen layer ordering disagrees with fusion")
        return result


def _bank(records: Sequence[Mapping[str, Any]], states: np.ndarray, dataset: Dataset,
          layer_ids: Sequence[int]) -> ConceptBank:
    positions = {row["id"]: index for index, row in enumerate(records)}
    def indices(pairs: Sequence[Mapping[str, Any]]) -> list[list[int]]:
        return [[positions[pair["negative_id"]], positions[pair["positive_id"]]] for pair in pairs
                if pair["split"] == "train" and pair["negative_id"] in positions and pair["positive_id"] in positions]
    risk, manipulation = indices(dataset.risk_pairs), indices(dataset.manipulation_pairs)
    if not risk or not manipulation:
        raise ValueError("Training fold lacks risk or manipulation construction pairs")
    safe = [index for index, row in enumerate(records) if row["kind"] == "safe"]
    original = sorted({pair[0] for pair in manipulation})
    return fit_concepts(states, risk, manipulation, safe, original, layer_ids)


def fit_detector(dataset: Dataset, states: Mapping[str, np.ndarray], layer_ids: Sequence[int],
                 config: MethodConfig | None = None, *, holdout: str | None = None,
                 extraction_protocol: Mapping[str, Any] | None = None) -> Detector:
    """Fit using train/select/calibrate only; never inspect diagnostic state arrays.

    For every internal family g, rebuild the concept bank without g's training
    attacks and evaluate on g's selection attacks. Direct harmful tasks are
    retained as risk supervision. The final bank/fusion are fit on outer-training
    data after selecting layers. The deployment threshold is then calibrated
    exactly once on the independent safety-only split.
    """
    config = config or MethodConfig()
    dataset.validate()
    original_dataset = dataset
    dataset = dataset.without_attacks(holdout)
    filtered_states: dict[str, np.ndarray] = {}
    for split in ("train", "select", "calibrate"):
        if split not in states:
            raise ValueError(f"Missing state array: {split}")
        values = np.asarray(states[split])
        old_rows, rows = original_dataset.splits[split], dataset.splits[split]
        if values.ndim != 3 or values.shape[:2] != (len(old_rows), len(layer_ids)) or not np.isfinite(values).all():
            raise ValueError(f"Invalid or misaligned states for {split}")
        keep = {row["id"] for row in rows}
        filtered_states[split] = values[[index for index, row in enumerate(old_rows) if row["id"] in keep]]
    if len({value.shape[2] for value in filtered_states.values()}) != 1:
        raise ValueError("State widths differ across fitting roles")
    train, selection, calibration = (dataset.splits[name] for name in ("train", "select", "calibrate"))
    train_h, select_h, calibrate_h = (filtered_states[name] for name in ("train", "select", "calibrate"))
    families = sorted({canonical_family(row.get("attack_family")) for row in train if row["kind"] == "jailbreak"} &
                      {canonical_family(row.get("attack_family")) for row in selection if row["kind"] == "jailbreak"})
    if len(families) < config.min_internal_families:
        raise ValueError(f"Need {config.min_internal_families} shared development attack families, found {families}")
    risk_folds, manipulation_folds, fold_provenance = [], [], []
    selection_index = {row["id"]: index for index, row in enumerate(selection)}
    for family in families:
        kept = [index for index, row in enumerate(train) if row["kind"] != "jailbreak" or
                canonical_family(row.get("attack_family")) != family]
        fold_rows, fold_states = [train[index] for index in kept], train_h[kept]
        bank = _bank(fold_rows, fold_states, dataset, layer_ids)
        reference_risk, reference_manipulation = concept_scores(fold_states, bank)
        safe_reference = np.asarray([row["kind"] == "safe" for row in fold_rows])
        paired_originals = {pair["negative_id"] for pair in dataset.manipulation_pairs if pair["split"] == "train" and
                            pair["positive_id"] in {row["id"] for row in fold_rows}}
        original_reference = np.asarray([row["id"] in paired_originals for row in fold_rows])
        risk_thresholds = np.asarray([calibrate_threshold(reference_risk[safe_reference, position],
                                                         config.selection.selection_fpr)["threshold"]
                                      for position in range(len(layer_ids))])
        manipulation_thresholds = np.asarray([calibrate_threshold(reference_manipulation[original_reference, position],
                                                                 config.selection.selection_fpr)["threshold"]
                                              for position in range(len(layer_ids))])
        risk_scores, manipulation_scores = concept_scores(select_h, bank)
        family_attack = np.asarray([row["kind"] == "jailbreak" and canonical_family(row.get("attack_family")) == family
                                   for row in selection])
        safe_mask = np.asarray([row["kind"] == "safe" for row in selection])
        risk_mask = np.asarray([row["kind"] != "jailbreak" for row in selection]) | family_attack
        risk_groups = np.asarray([family if row["kind"] == "jailbreak" else "direct_harmful" for row in selection])
        safe_groups = np.asarray([str(row.get("source", {}).get("legacy_metadata", {}).get(
            "packaging", row.get("source", {}).get("packaging", "safe"))) for row in selection])
        labels = np.asarray([row["label"] for row in selection])
        risk_folds.append({"scores": risk_scores[risk_mask], "labels": labels[risk_mask],
                           "safe_mask": safe_mask[risk_mask], "attack_groups": risk_groups[risk_mask],
                           "safe_groups": safe_groups[risk_mask], "thresholds": risk_thresholds})
        original_ids = {row["original_id"] for row, flag in zip(selection, family_attack) if flag}
        if any(identifier not in selection_index for identifier in original_ids):
            raise ValueError("Internal manipulation validation lacks paired original harmful inputs")
        paired_mask = np.asarray([row["id"] in original_ids for row in selection])
        manipulation_mask = family_attack | paired_mask | safe_mask
        manipulation_labels = np.where(family_attack, 1, np.where(paired_mask, 0, -1))
        manipulation_folds.append({"scores": manipulation_scores[manipulation_mask],
                                   "labels": manipulation_labels[manipulation_mask], "safe_mask": safe_mask[manipulation_mask],
                                   "attack_groups": np.full(int(manipulation_mask.sum()), family),
                                   "safe_groups": safe_groups[manipulation_mask], "thresholds": manipulation_thresholds})
        fold_provenance.append({"held_out_family": family, "training_rows": len(fold_rows),
                                "held_out_selection_attacks": int(family_attack.sum()),
                                "paired_selection_originals": len(original_ids),
                                "concept_training_ids_sha256": hashlib.sha256(json_bytes(sorted(row["id"] for row in fold_rows))).hexdigest()})
    risk_selected = select_layers(risk_folds, layer_ids, config.selection)
    manipulation_selected = select_layers(manipulation_folds, layer_ids, config.selection)
    final_bank = _bank(train, train_h, dataset, layer_ids)
    risk, manipulation = concept_scores(train_h, final_bank)
    features, feature_layers, _ = build_feature_matrix(risk, manipulation, risk_selected.layers,
                                                       manipulation_selected.layers, layer_ids)
    fusion = fit_sparse_fusion(features, np.asarray([row["label"] for row in train]), feature_layers, config=config.fusion)
    if not fusion.diagnostics["converged"]:
        raise RuntimeError("Sparse fusion did not converge; no detector was frozen. Adjust the training configuration.")
    union = sorted(set(risk_selected.layers + manipulation_selected.layers))
    positions = [list(layer_ids).index(layer) for layer in union]
    deploy_bank = ConceptBank(layer_ids=np.asarray(union), **{name: getattr(final_bank, name)[positions]
                              for name in ("risk_direction", "manipulation_direction", "safe_anchor", "harmful_anchor")})
    model = Detector(deploy_bank, risk_selected.layers, manipulation_selected.layers, fusion, 0.0)
    calibration_result = calibrate_threshold(model.decision_function(calibrate_h[:, positions]), config.target_fpr)
    model.threshold = calibration_result["threshold"]
    model.metadata = {"method_config": asdict(config), "extraction_protocol": dict(extraction_protocol or {}),
                      "outer_holdout": canonical_family(holdout) if holdout else None,
                      "concepts": final_bank.diagnostics, "fusion": fusion.diagnostics,
                      "risk_selection": risk_selected.diagnostics, "manipulation_selection": manipulation_selected.diagnostics,
                      "internal_folds": fold_provenance, "calibration": calibration_result,
                      "fit_roles": {split: {"n": len(dataset.splits[split]),
                                   "ids_sha256": hashlib.sha256(json_bytes([row["id"] for row in dataset.splits[split]])).hexdigest()}
                                    for split in ("train", "select", "calibrate")},
                      "data_output_sha256": dataset.manifest.get("output_sha256", {}),
                      "label_status": "candidate_not_independently_validated",
                      "confirmation_claim": False,
                      "selection_note": "Fixed configurable penalties assessed on internal development attack-family folds; no diagnostic tuning.",
                      "source_code_sha256": {path.name: sha256_file(path) for path in Path(__file__).parent.glob("*.py")}}
    return model
