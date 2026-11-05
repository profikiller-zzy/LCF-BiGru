# DCCF JBB pilot v1 — review stage only

This directory contains an unassigned 100-goal construction pilot from pinned
JailbreakBench (JBB-Behaviors) and its published PAIR/GCG artifacts. It is a
candidate pool for human review, not a model-ready train/evaluation dataset.

- `records.jsonl` contains 100 harmful candidates, 100 benign candidates, and
  only jailbreak prompts whose pinned artifact index, normalized goal,
  behavior identifier, and category all match the corresponding harmful row.
- `pairs/manipulation_candidates.jsonl` indexes those source-linked J-H pairs.
- `pairs/risk_candidates.jsonl` contains three same-category, nearest-character-
  length B suggestions for each H. These are not semantic matches.
- `review/*.csv` are independent-review worksheets. Reviewers should not see
  model scores or the artifact success outcome when judging input semantics.
- `review/external_unpaired_jailbreaks.jsonl` preserves prompts whose source
  fields do not all match the linked H; they are excluded from the core records.
- `review/excluded_artifact_rows.jsonl` logs source artifact entries with no
  prompt field. Source model responses and judge outputs are not retained.
- `licenses/` stores the MIT license texts retrieved from the pinned sources.

`review/ANNOTATION_GUIDE.md` defines the double-review and adjudication rules.

Every record remains a source candidate. Published attack success is metadata
for the named source model only; it does not affect the global input risk label
or inclusion. No train/select/calibrate/diagnose/confirmation role is assigned.
Before fitting, complete two-person review, adjudicate conflicts, check external
overlap with historical DCCF material, tokenize the full chat-template inputs
for the target backbone without truncation, and assign groups/splits under the
protocol in `docs/DATASET_CONSTRUCTION_PLAN_V1.md` and `docs/METHOD.md`.

Rebuild from the hash-pinned sources (standard library only):

```bash
python tools/build_jbb_pilot.py --output data/dccf_v1_jbb_pilot
```

The builder refuses to overwrite an existing output directory.
