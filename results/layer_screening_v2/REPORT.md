# Layer screening V2 run: development results, not a confirmatory claim

Run on a server (RTX 3090) with Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28,
data `data/dccf_v1`, train+select only. Labels are LLM-reviewed, not human ground truth. select was
used only as a recheck. confirm and calibrate were not touched.

## Stage 1: per-layer probes (B vs H, 558 train rows, 279 goal groups, 5 folds x 5 seeds)

- All 27 layers pass the preregistered gate (median AUROC >= 0.70, worst run >= 0.55). The gate did
  not filter anything: median AUROC is 0.92 at layers 1 to 4, 0.99 from layer 9, and 1.00 at layers 22 and 23.
  As anticipated in the plan, the B/H task is saturated, so stage 1 gives no useful pruning here.
- A bag-of-words logistic regression on the same folds reaches AUROC 0.83 (worst fold 0.78). Token length
  alone is at chance (0.48). So most of the B/H signal is lexical, and probe AUROC near 1.0 does not
  show that a layer encodes risk beyond word choice.
- Transfer to jailbreaks (probe never saw J): at layer 20 the out-of-fold AUROC(B vs J) is 0.97 to 0.99
  for all families. Early layers fail on pair_tap and persuade (layer 1: 0.84 and 0.74).

## Stage 2: strategy-by-layer tables (train held-out folds, all 27 candidate layers)

Selected layers (2 per branch, stopped on no positive incremental utility):

| fold level | risk | manipulation |
|---|---|---|
| strategy (15 columns) | 27, 26 | 9, 2 |
| family (6 columns) | 26, 15 | 9, 2 |

- The risk branch is carried by the last layers (26 to 27). Held-out TPR at 5% safe FPR is near 0 in
  layers 1 to 13 and 0.86 to 1.00 in layers 22 to 25.
- The manipulation branch picked layer 9 first, then layer 2 (response correlation 0.82 to 0.87).

## Length confound in the manipulation branch (important)

Every J is longer than its exact H (median 42 tokens for H, 64 to 512 for J), so length alone
separates them perfectly (AUROC 1.000 in every family). After removing a length trend fitted on
held-out H references (`length_control.csv`):

- Early layers collapse: layers 1 to 7 give adjusted AUROC near 0 for dsn, gcg, pair_tap and saa.
  The layer 9 and layer 2 manipulation selections are therefore most likely length detectors.
- Template stays near 1.00 from layer 7 on, so that family carries a signal beyond length.
- Non-template families reach only 0.6 to 0.84 after adjustment, even at layers 22 to 25.
  The raw AUROC of 0.9 to 1.0 in the stage 2 tables overstates how well the manipulation concept
  separates wrapped from plain requests.

Caveat: the adjustment is a linear fit on log token count using H rows only, a crude control that can
over- or under-correct. It shows that the raw numbers cannot be read as manipulation signal, not
what the true residual signal is.

## Limits

- 3 families (gcg, pair_tap, saa) have 5 to 7 goal groups in select, so select rechecks for them are
  flagged low-sample and are not interpreted.
- select shares 24 template lineages with train, so select results do not speak to unseen wrappers.
- The selection ran on layer responses that include the length confound above. The selected
  manipulation layers (9, 2) should not be adopted without a length-matched control set.
- Not run yet: Sentinel-style correlation-pruning comparison, full-27-layer vs C_probe ablation,
  random-layer baseline, seed/bootstrap stability of the layer sets, heatmap figures.
