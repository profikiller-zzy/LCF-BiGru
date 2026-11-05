# DCCF v1 dual-concept dataset

Built by `tools/dccf_v1/stage1_select.py` (source selection) and `tools/dccf_v1/stage3_assemble.py`
(assembly), from pinned JBB-Behaviors, JailbreakBench artifacts, JailBreakV-28K and Dolly-15k
revisions (see `manifest.json`). Loads with `python -m dual_concept validate --data data/dccf_v1`.

- 393 goal groups, each with one harmful H, one benign B and 1 to 3 jailbreak J inputs.
- Roles are assigned per goal group (stratified by source and category): train, select, confirm.
  `calibrate` is 200 ordinary Dolly requests, separate from all goal groups. `diagnose` is empty.
- `pairs/risk.jsonl` holds train-only H-B pairs (loader rule); all-role pairs are in
  `annotation/pairs_risk_all_roles.jsonl`.

## Label status

Every H-B and J-H decision was made by two independent LLM reviewers plus an LLM adjudicator for
disagreements. No `uncertain` label was allowed. There was no human annotation, so these are not
human ground truth (`label_status = two_agent_reviewed_adjudicated_not_human_verified`).

## Limits

- `confirm` is goal-disjoint but shares attack template lineages with train (see `manifest.json`),
  so it does not test unseen attack types. Use `--holdout <family>` for family-level evaluation.
- B requests for most JailbreakV goals were authored by an LLM; style regularities may remain.
- Calibration requests are not matched to jailbreak formatting.
- Attack success was never evaluated and does not affect any label.
