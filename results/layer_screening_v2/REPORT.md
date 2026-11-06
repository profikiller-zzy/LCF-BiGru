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

## Semantic strategies and cross-strategy fusion (third pass)

The earlier "strategy" axis was a source family or an opaque template hash. Each of the 30 template lineages
was read and given the dominant wrapper mechanism (`tools/layer_screening/strategy_labels.py`, single LLM
annotator, written before looking at any layer result, not human verified). Seven strategies result:
persona_roleplay (DAN, AIM, OPPO, Eva ...), fiction_scene (scripted dialogue or movie scene),
instruction_control (rant-then-break-rules, Developer Mode, opposite mode), adversarial_suffix (GCG, DSN and
junk-token templates), persuasion, scenario_reframing (PAIR/TAP), rule_suffix (random-search attack).
Train counts (prompts / goal groups): persona 321/225, adversarial_suffix 121/107, instruction_control 71/65,
fiction_scene 39/38, persuasion 38/38, rule_suffix 35/35, scenario_reframing 28/28.

Each strategy is held out in turn together with its goal groups. Negatives are the length-matched padded
controls and thresholds come from padded controls of the fit groups. Layers where held-out TPR reaches 0.9
(`semantic_strategy/strategy_layer_table.csv`):

| 策略 | 训练条数/目标组 | 风险分支 TPR≥0.9 的层 | 操纵分支 TPR≥0.9 的层 |
|---|---|---|---|
| adversarial_suffix | 121/107 | 3,19-27 | 无 |
| fiction_scene | 39/38 | 21-27 | 6,9-27 |
| instruction_control | 71/65 | 19-27 | 9-27 |
| persona_roleplay | 321/225 | 无 | 4-27 |
| persuasion | 38/38 | 19-27 | 无 |
| rule_suffix | 35/35 | 无 | 9-27 |
| scenario_reframing | 28/28 | 无 | 20,27 |

Reading it:
- The two branches cover different strategies. The risk readout recovers adversarial_suffix and persuasion
  (late layers) but misses persona_roleplay entirely (best TPR 0.69) and rule_suffix and scenario_reframing in
  most layers. The manipulation readout recovers persona_roleplay, fiction_scene, instruction_control and
  rule_suffix from layer 9 or earlier, but misses adversarial_suffix and persuasion at every layer.
- Layer selection (2 layers per branch, `select_layers` over the 7 held-out strategies): risk = 25, 15;
  manipulation = 14, 15. Both stopped on no positive incremental utility, so a third and fourth layer were
  not added.
- Layer 15 is picked by both branches. Layers 14 and 15 sit where the padding shift on the manipulation readout
  is still 0.18 to 0.22, against J-minus-padded-H differences of only 0.16 to 0.19 for adversarial_suffix and
  persuasion. For those two strategies the manipulation signal at these layers is not clearly above what
  padding alone produces; for persona, fiction, instruction_control and rule_suffix it is 0.4 to 0.57.

Fusion of the union (layers 14, 15, 25; sparse linear head from `dual_concept.fusion`), leaving one whole
strategy out of both concept building and fusion fitting; TPR at a threshold set for 5% FPR on padded benign
requests of the fit groups, FPR measured on the held-out groups' padded benign requests:

| arm | worst-strategy TPR | mean TPR | worst-strategy AUROC | mean FPR |
|---|---|---|---|---|
| risk + manipulation (fusion) | 0.75 | 0.96 | 0.89 | 0.057 |
| risk layers only | 0.67 | 0.87 | 0.90 | 0.059 |
| manipulation layers only | 0.29 | 0.73 | 0.82 | 0.108 |
| best single risk layer (25) | 0.67 | 0.87 | 0.90 | 0.059 |
| best single manipulation layer (14) | 0.24 | 0.73 | 0.81 | 0.104 |

- Fusion lifts the worst strategy from 0.67 (risk only) to 0.75 and the mean TPR from 0.87 to 0.96 at about
  the same FPR. The gain comes from the branches failing on different strategies: persona_roleplay goes from
  0.67 (risk only) to 1.00, rule_suffix from 0.71 to 1.00.
- scenario_reframing (28 goal groups) stays the weakest at 0.75 TPR with 0.11 FPR on its held-out groups;
  adversarial_suffix has 0.10 FPR. These two sit above the 5% target.
- AUROC does not improve over risk-only (0.89 vs 0.90 worst-strategy), so the gain is at the operating
  point, not in ranking.
- In the full-train fit the weights are risk 25: 2.42, risk 15: 1.35, manipulation 15: 1.75, manipulation 14: 0.15,
  so the fusion mostly drops layer 14.

Caveats specific to this pass:
- Not an independent estimate. The layers were chosen from the same seven held-out strategy folds that the
  fusion is then scored on. Only the concept directions and fusion weights are refit without the held-out
  strategy; the choice of layers 14, 15 and 25 has seen all seven strategies.
- Three strategies have 28 to 35 goal groups, so per-strategy TPRs are coarse (one goal group is about 3 points).
- Negatives are encyclopedic filler plus a benign request. They match length, not wrapper style, and the
  filler itself shifts the early and middle layers. A benign role-play or format-instruction control is still missing.
- Strategy labels come from one annotator and strategy sizes are very unequal (persona is 56% of train jailbreaks).
- confirm and calibrate were not used.

## Stage 4: cross-layer detector trained and frozen (fourth pass)

Moved to a new server (RTX 4090) after the first lost its GPU. Code and the Qwen2.5-7B-Instruct weights
were copied or re-downloaded (11 file hashes match the old server). All 1479 train/select states plus the
200 calibrate states were re-extracted; against the old server's cache the minimum per-layer cosine is
0.999991 and the maximum relative L2 difference is 4.2e-3 (fp16 across GPUs), and the extraction protocol differs
only in the checkpoint stat fingerprint (mtime). Stage 4 uses the new extraction.

Training (`tools/layer_screening/stage4_train_detector.py`, project `Detector` format, reloads to identical scores):
- concept directions/anchors and sparse fusion head: all train goal groups (279);
- layers: risk 25, 15 and manipulation 14, 15 from stage 3 (union 14, 15, 25);
- threshold: the 200 calibrate requests only, 5% budget = 10 false positives, threshold 0.0417 (logit, `>=`);
- fusion weights: risk25 2.42, risk15 1.36, manip14 0.15, manip15 1.75; the fusion converged;
- confirm was never read.

Frozen artifact: `results/stage4_detector_v1/detector/` (`model.json`, `parameters.npz`; SHA-256 checked
against the server copy and loadable with `Detector.load`). Full numbers: `results/stage4_detector_v1/report.json`.

On select (not fitted, but select was used earlier to compare layers, so it is a development check):

| arm | harmful TPR | jailbreak TPR | ordinary B FPR | AUROC |
|---|---|---|---|---|
| fusion (risk + manipulation) | 1.00 | 1.00 | 0.053 | 0.999 |
| risk layers only | 1.00 | 0.99 | 0.088 | 0.988 |
| manipulation layers only | 0.00 | 0.97 | 0.000 | 0.829 |

Per strategy, fusion reaches TPR 1.00 on all seven strategies on select (5 to 41 goal groups each; rule_suffix 6
and scenario_reframing 5 are too small to read). Risk-only misses rule_suffix (0.83) and scenario_reframing (0.80);
manipulation-only misses persuasion (0.73) and, by design, every direct harmful request (0.00).

**Do not read the jailbreak TPR as evidence of generalisation.** Jailbreaks in select have a median of 132 tokens
against 43 for both B and H. A rule that only counts tokens, with its threshold set on the same 200 calibrate
requests, flags 0.99 of select jailbreaks at 0.04 FPR on B, and its token-count-only AUROC is 0.997
(`length_only_baseline.py`). The detector's results on jailbreaks are therefore indistinguishable from this trivial
rule on this data. What the length rule cannot do is detect direct harmful requests (TPR 0.04, AUROC 0.475); the
fusion detector gets 1.00 there, so the risk branch carries real signal about harmful content. For the manipulation
branch the evidence for a signal beyond length is the padded-control analysis in the earlier sections, which
supports persona, fiction, instruction-control and rule-suffix and does not support adversarial-suffix or persuasion.

Leave-one-strategy-out (concepts and fusion refit without the held-out strategy; held-out groups' own short B as
controls; threshold from calibrate): fusion TPR is 1.00 for five strategies, 0.86 for scenario_reframing (FPR 0.10)
and 1.00 for the rest at FPR 0 to 0.05. Because the controls are short and the jailbreaks are long, length again
inflates these numbers. Within that limit the fusion beats risk-only on rule_suffix (1.00 vs 0.82) and
scenario_reframing (0.86 vs 0.72), and manipulation-only fails on persuasion (0.53) and adversarial_suffix (0.61).

What this stage does not show: performance on unseen attack types, a length-robust FPR on long benign inputs,
or anything about confirm. The calibrate set is 200 short ordinary requests, so the operating point is also set on
a length profile that a long benign prompt does not share; no long benign input was tested.

## Limits

- 3 families (gcg, pair_tap, saa) have 5 to 7 goal groups in select, so select rechecks for them are
  flagged low-sample and are not interpreted.
- select shares 24 template lineages with train, so select results do not speak to unseen wrappers.
- The selection ran on layer responses that include the length confound above. The selected
  manipulation layers (9, 2) should not be adopted without a length-matched control set.
- Not run yet: a control that matches style as well as length (for example benign role-play and suffix
  wrappers), Sentinel-style correlation-pruning comparison, full-27-layer vs C_probe ablation,
  random-layer baseline, seed/bootstrap stability of the layer sets, heatmap figures.
