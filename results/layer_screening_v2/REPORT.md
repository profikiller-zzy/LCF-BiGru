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

## Length-matched controls (second pass)

Control panel: `data/dccf_v1_length_controls` (787 of 807 train/select jailbreaks; 20 skipped for
lack of a reachable filler length). Each J gets H_pad and B_pad: the same H or B preceded by a
neutral Dolly passage, with the formatted token count within 3 tokens of J. The first run set the
operating thresholds on short unpadded requests, which flagged nearly every long padded control
(safe FPR 0.89); the rerun below takes thresholds from padded controls of the fit groups.

Selected layers (2 per branch):

| fold level | risk, unpadded -> matched | manipulation, unpadded -> matched |
|---|---|---|
| strategy | 27, 26 -> 23, 25 | 9, 2 -> 14, 13 |
| family | 26, 15 -> 23, 16 | 9, 2 -> 14, 15 |

- The manipulation layers move from early (9, 2) to middle layers (13 to 15) once length is matched,
  which supports reading the early-layer result as length.
- Held-out manipulation TPR at 5% control rate is low in the matched setting: worst family 0.34 at the
  chosen layers (family level) and 0.14 (strategy level). Persuade is hardest (TPR 0.1 to 0.3 at most layers).
- Risk-branch selection settles on layers 23 to 25 (worst-group TPR 0.74); this is stable across both passes.

## Padding is not a clean length control either

`padding_effect_by_layer.py` compares each H with its own padded version. The manipulation readout
separates them almost perfectly in early layers (AUROC 1.00 at layers 1 to 5, 0.97 at layer 13,
0.90 at 14) and only weakly in late layers (0.60 to 0.63 at layers 23 to 27; mean shift +0.05 to +0.07).
So encyclopedic filler itself looks like a "wrapper" to the early and middle layers. Consequences:

- Layer 14 (the matched manipulation pick) carries a padding shift of about 0.35 in readout units, against
  a J-minus-H_pad difference of 0.16 to 0.36 for dsn, gcg, pair_tap and persuade. Most of its signal could
  be padding rather than manipulation, so the 14/13/15 pick is not trustworthy.
- In layers 23 to 27 the padding shift is 0.05 to 0.07 while the J-minus-H_pad difference is 0.19 to 0.64.
  Late layers are the only place where the manipulation signal clearly exceeds both length and padding.
  Per-family AUROC against H_pad there: dsn 0.89 to 0.91, gcg 0.89 to 0.93, pair_tap 0.96 to 0.99,
  persuade 0.79 to 0.82, saa 1.00, template 0.95 to 0.97.
- The selection routine ranks layers by held-out coverage at a fixed control rate and does not know about
  this padding shift, so its output for the manipulation branch should be treated as a lower bound on
  how much length and filler confound the result, not as a layer recommendation.

## Limits

- 3 families (gcg, pair_tap, saa) have 5 to 7 goal groups in select, so select rechecks for them are
  flagged low-sample and are not interpreted.
- select shares 24 template lineages with train, so select results do not speak to unseen wrappers.
- The selection ran on layer responses that include the length confound above. The selected
  manipulation layers (9, 2) should not be adopted without a length-matched control set.
- Not run yet: a control that matches style as well as length (for example benign role-play and suffix
  wrappers), Sentinel-style correlation-pruning comparison, full-27-layer vs C_probe ablation,
  random-layer baseline, seed/bootstrap stability of the layer sets, heatmap figures.
