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

## Stage 5: nested evaluation with baselines and wrapper-matched benign controls (fifth pass)

New controls (not part of any train/select/calibrate/confirm role):
- `data/dccf_v1_wrapper_controls`: 724 **W(B)** requests. Each takes the real wrapper of a jailbreak J and swaps its
  harmful goal for the benign request of the same goal group, so wrapper text and length are the attack's own.
  Built for persona, fiction, instruction-control, adversarial-suffix and rule-suffix (the latter needs the rewritten
  goal replaced too). PAIR/TAP and persuasion are free-form rewrites with no extractable wrapper (83 jailbreaks), so
  they have no W(B); scenario_reframing has none either, and its W(B) columns are empty.
- 240 long natural benign inputs (Dolly context plus question; median 244 tokens).

Protocol (`tools/layer_screening/stage5_nested_eval.py`): each semantic strategy is held out with all its goal
groups. Inside each fold the layer selection, concept bank, fusion, probes, bag-of-words model and operating
threshold use only the remaining groups, so the layers are re-selected per fold (nested). All arms share folds,
thresholds and negatives. Thresholds: `short` = 200 calibrate requests; `mixed` = calibrate plus half of the long
benign pool (the other half is only for evaluation). Arms: length rule, bag-of-words LR, a logistic regression on the
hidden states of the same layers (`hidden_probe`), DCCF risk, DCCF manipulation, DCCF fusion, and DCCF fusion with
directions made orthogonal to a padding direction. Each trainable arm is run without and with W(B) of the fit groups
as extra negatives. Table values are the mean over the 7 held-out strategies, worst strategy after the slash; held-out
groups come from train (the `select` evaluation is also in the csv). Nothing was tuned on the evaluation rows; the
probe regularisation C=0.01 was fixed in advance. Five fusion fits did not converge (persona_roleplay fold with 50 fit
groups, scenario_reframing with W(B)); they are kept in the csv and not hidden.

Nested layer choices vary little: risk layers 23 plus one of 16 to 18 (24, 25 once), manipulation layers 14 plus 15
(16, 18 or 9 in three folds).

Threshold `short`, trained without W(B):

| arm | J TPR | H TPR | B FPR | W(B) FPR | long FPR | AUROC J vs W(B) | paired J>W(B) |
|---|---|---|---|---|---|---|---|
| length rule | 0.99/0.95 | 0.01 | 0.01 | 1.00 | 1.00 | 0.50/0.48 | 0.45/0.38 |
| bag-of-words LR | 0.77/0.28 | 0.29 | 0.02 | 0.59 | 0.81 | 0.73/0.54 | 0.86/0.74 |
| hidden-state probe | 0.99/0.93 | 1.00 | 0.02 | 0.93 | 0.57 | 0.91/0.75 | 0.98/0.92 |
| DCCF risk | 0.94/0.72 | 1.00 | 0.04 | 0.87 | 0.06 | 0.88/0.70 | 0.96/0.89 |
| DCCF manipulation | 0.94/0.61 | 0.13 | 0.05 | 0.94 | 0.79 | 0.57/0.44 | 0.54/0.23 |
| DCCF fusion | 0.99/0.93 | 1.00 | 0.04 | 0.95 | 0.49 | 0.87/0.70 | 0.96/0.89 |
| DCCF fusion, length-projected | 0.99/0.93 | 1.00 | 0.06 | 0.94 | 0.15 | 0.87/0.71 | 0.96/0.89 |

Trained with W(B) as negatives, same threshold:

| arm | J TPR | H TPR | B FPR | W(B) FPR | long FPR | AUROC J vs W(B) |
|---|---|---|---|---|---|---|
| bag-of-words LR | 0.36/0.05 | 0.36 | 0.02 | 0.18 | 0.49 | 0.77/0.56 |
| hidden-state probe | 0.99/0.97 | 1.00 | 0.03 | 0.54 | 0.04 | 0.95/0.88 |
| DCCF risk | 0.91/0.68 | 1.00 | 0.03 | 0.72 | 0.02 | 0.88/0.71 |
| DCCF manipulation | 0.16/0.00 | 0.64 | 0.12 | 0.00 | 0.22 | 0.57/0.46 |
| DCCF fusion | 0.91/0.68 | 1.00 | 0.03 | 0.72 | 0.02 | 0.88/0.71 |

What it shows:
1. **The manipulation readout does not detect manipulation.** On the real wrapper, J and its own W(B) are not
   separable by the manipulation readout (AUROC 0.44 to 0.49 at layers 14, 15, 23; W(B) scores slightly higher than J,
   mean 0.75 to 0.79 against 0.66 to 0.75). The same-wrapper comparison removes length and style, so the readout
   measures "a wrapper is present", not "a harmful goal sits inside a wrapper". The paired J-over-W(B) win rate of the
   manipulation arm is 0.54 mean and 0.23 worst. This contradicts the earlier interpretation of the padded-control
   result (J versus plain padded H), which could not tell wrapper from goal.
2. **With W(B) as negatives the fusion becomes the risk branch.** Manipulation weights go to 0 (full-train fit:
   risk23 3.37, manip14 0.00, manip15 0.00), and DCCF fusion and DCCF risk have identical TPR in all 7 folds. Without
   W(B) the manipulation features carry weight (manip14 1.55) only because wrapped benign inputs were never shown.
3. **The risk branch is the part that works, and it still flags W(B).** Risk J-versus-W(B) AUROC is 0.82 to 0.86
   and the paired win rate is 0.96, so the risk readout does respond to the goal. Its W(B) FPR at the 5%
   threshold is 0.72 to 0.87, because benign requests inside jailbreak wrappers also move the risk readout (mean
   risk readout at layer 23: B -0.03, W(B) 0.36, J 0.55, H 0.82). A deployment threshold that must pass benign
   role-play wrappers would cost TPR.
4. **A plain linear probe on the same hidden states does as well or better than DCCF.** Hidden-state probe: J TPR
   0.99/0.93 and paired win 0.98 without W(B); with W(B) negatives it reaches J TPR 0.99/0.97, W(B) FPR 0.54,
   long-benign FPR 0.04 and J-versus-W(B) AUROC 0.95/0.88, all better than DCCF fusion (0.91/0.68, 0.72, 0.02,
   0.88/0.71). In this data the rank-one concept construction does not add anything over a linear classifier
   fitted on the same layers.
5. **Length baselines behave as expected.** The token-count rule gets J TPR 0.99 on short thresholds, flags every
   long benign input and W(B) (FPR 1.00), and has J-versus-W(B) AUROC 0.50. Long benign inputs expose the short
   calibration: DCCF fusion without W(B) flags 0.49 of them at the `short` threshold (0.06 for risk alone).
   Projecting the padding direction out lowers that to 0.15 at the same J TPR (0.99/0.93), which is the one place
   the length projection helps.
6. **Per-strategy (short, no W(B)).** TPR for fusion is 1.00 on all strategies except scenario_reframing (0.93).
   Manipulation alone fails on adversarial_suffix (0.61). With W(B) negatives risk/fusion TPR drops to 0.68 on
   rule_suffix and 0.76 on scenario_reframing, and the fusion misses the strategies where J and its wrapped benign
   are most alike.

Limits: W(B) is itself an ambiguous case (a benign request inside a jailbreak persona may reasonably count as
risky); here it is a diagnostic negative, not a label. Persuasion and scenario_reframing have no W(B), so about a
quarter of the jailbreaks are not covered by the wrapper test. Held-out groups are from train (so no later
selection leakage into the fold) but strategy sizes remain unequal (persona 225 groups, three strategies 28 to 38).
The evaluation uses short plain B, short padded controls, W(B) and long Dolly texts; real user traffic is none of them.
Confirm and diagnose were not read. Labels are LLM-reviewed and the strategy labels come from one annotator.

## Stage 6: rebuilt manipulation concept (sixth pass)

Problem from stage 5: the manipulation readout (direction from J minus H) could not separate J from the same wrapper
around a benign request, W(B). A manipulation concept should be about the wrapper, independent of the goal.
`tools/layer_screening/stage6_manipulation_concept.py` builds four rank-one directions and measures them on held-out
strategies (the strategy's goal groups are removed from direction fitting). Only the five strategies with an
extractable wrapper can be scored (persona, fiction, instruction-control, adversarial-suffix, rule-suffix);
persuasion and scenario_reframing have no W(B) and are untested here, not dropped.

| construction | direction from |
|---|---|
| orig | J - H (current) |
| unmatched | W(B) - B (wrapper only, but wrapped inputs are longer) |
| wrapB | W(B) - B_pad (wrapper only, length matched) |
| wrapBH | W(B) - B_pad stacked with J - H_pad |

Four measurements (AUROC, mean / worst over the five held-out strategies; layers 8, 14, 20, 25):

| | orig | unmatched | wrapB | wrapBH |
|---|---|---|---|---|
| M1 W(B) vs B_pad (wrapper detected whatever the goal; want high) | .86/.60, .98/.90, .94/.72, .95/.76 | .85/.57, .98/.91, .96/.82, .97/.84 | **1.00/.98**, .97/.87, .97/.83, .96/.82 | 1.00/.98, .97/.87, .96/.80, .96/.81 |
| M2 J vs W(B) (readout also sees the goal; want about 0.5) | .52/.48, .48/.35, .42/.20, .31/.15 | .52/.49, .45/.25, .56/.30, .63/.43 | .53/.52, .47/.29, .58/.26, .61/.29 | .53/.52, .48/.32, .47/.20, .44/.15 |
| M3 J vs H_pad (stage-2 test; want high) | .85/.56, .98/.90, .94/.71, .98/.88 | .84/.53, .98/.90, .94/.70, .96/.85 | .99/.97, .97/.87, .95/.73, .97/.83 | .99/.98, .98/.88, .96/.79, .97/.86 |
| M4 H vs B (plain goal; contrast) | .56, .36, .06, .02 | .61, .65, .89, .97 | .58, .55, .87, .95 | .57, .43, .39, .54 |

What this shows:
1. **At the layers best suited to it, a length-matched wrapper direction works and is goal-blind.** `wrapB` and
   `wrapBH` at layers 4 to 8 give M1 worst-strategy 0.97 to 0.99 with M2 mean 0.51 to 0.53 (top-3 layers by
   M1 minus |M2 - 0.5|: 4, 5, 7 with score 0.97). Wrappers of a held-out strategy are detected regardless of the goal
   and the readout does not see the goal. This is the first manipulation readout that passes the same-wrapper test
   that J minus H failed (stage 5 AUROC 0.44 to 0.49).
2. **At layers 14 to 25 the construction does not stay goal-blind.** M2 mean rises to 0.58 to 0.61 at layers 20 and 25
   for `wrapB` and the worst strategy falls to 0.26 to 0.29 (fiction_scene is 0.26 to 0.29, i.e. J scores below
   its own W(B)). `wrapB` at layers 20 and 25 also reads plain harmful content (M4 0.87 and 0.95 against 0.58 at
   layer 8), so a direction fitted on wrappers around benign goals still carries risk information in late layers.
   `wrapBH` is less goal-sensitive in M4 (0.39 at layer 20, 0.54 at layer 25) but its M2 is also unstable (worst 0.15).
3. **The old stage-3 choice (layers 14 and 15) sits where the old direction is least clean.** `orig` has M4 0.36
   at layer 14 and 0.02 at layer 25, i.e. it points against harmfulness there (plain H scores below B), which
   explains why it never separated J from W(B).
4. **Caveat: "detects unseen wrappers" is largely a lexical property here.** W(B) and B_pad have the same token
   count by construction (length-only AUROC 0.504), but a bag-of-words classifier trained on the other four
   strategies detects the held-out wrapper with AUROC 0.98 to 1.00 for fiction, instruction-control, persona and
   rule-suffix, and 0.89 for adversarial-suffix (`wrapper_text_baseline.py`). The hidden-state readout at its best
   layers (0.97 to 1.00) is no better than word counts on those four. Wrapped benign text differs from plain
   benign text in obvious words ("act as", "rules", format instructions); the early-layer direction may be reading
   exactly that. Whether it generalises to wrappers that look different, such as encodings, is not tested.
5. **The goal-aware part of a jailbreak is not in this concept.** If the manipulation concept is wrapper-only
   (M2 about 0.5), then the evidence that something is a jailbreak must come from the risk branch, which is the
   same conclusion as stage 5: J versus W(B) is separated by risk, not by manipulation.

Consequences for the method (not yet re-run end to end): replace J minus H by `wrapB` (or `wrapBH`) at early layers
for the manipulation branch, expect it to contribute a goal-independent "wrapper present" signal, and treat
W(B) as a policy decision. If benign requests in jailbreak wrappers must pass, the wrapper score cannot raise the
detection score on its own; if they must be flagged, the manipulation branch is a wrapper detector and the
reported detection rate for jailbreaks measures wrapper detection plus risk. The fusion with the rebuilt concept,
under both policies and with the nested layer choice, is the next experiment.

## Stage 7: nested fusion with the rebuilt manipulation concept, two W(B) policies (seventh pass)

`tools/layer_screening/stage7_nested_fusion_wrapper.py`. Same outer protocol as stage 5 (one semantic strategy held out
with all its goal groups; everything inside the fold sees only the other groups). New: the manipulation branch uses the
wrapper concept `wrapB` (W(B) minus length-matched B) with two layers chosen inside each fold by inner leave-one-strategy-out
(score = worst-strategy M1 minus |mean M2 minus 0.5|; the rule was written after stage 6 had been seen on all strategies,
so the criterion is not blind, only its per-fold application is nested). Risk layers and the old J-minus-H layers come
from stage 5's nested selection. Two policies for a wrapped benign request W(B):
- **A** W(B) is normal: label 0 in training and the fit groups' W(B) enter the threshold pool (5% FPR over calibrate,
  half the long-benign pool and W(B)).
- **B** W(B) is risk (the wrapper is itself the manipulation attempt): label 1 in training; threshold from ordinary and
  long benign only.
Held-out groups are from train. Averages are over the 7 strategies unless stated; persuasion and scenario_reframing
have no W(B), so the W(B) columns average 5. Five fusion fits did not converge (persona orig-fusion without W(B);
scenario_reframing dccf_risk and orig-fusion under A and B); the csv keeps them.

Wrapper layers chosen per fold: 4 and 7 (fiction, persuasion), 7 and 4 (instruction_control), 7 and 5 (rule_suffix,
scenario_reframing), but 13 and 19 (adversarial_suffix) and 1 and 3 (persona_roleplay). The choice is stable for four
folds and not for two, which reflects the lexical nature of the concept seen in stage 6.

**Policy A, W(B) is normal.** Threshold pool includes W(B).

| arm | J TPR (worst) | H TPR | B FPR | long FPR | W(B) flagged | AUROC J vs W(B) |
|---|---|---|---|---|---|---|
| hidden-state probe | 0.97 (0.86) | 1.00 | 0.02 | 0.10 | 0.51 | 0.93 |
| DCCF risk | 0.68 (0.00) | 1.00 | 0.01 | 0.00 | 0.18 | 0.88 |
| DCCF J-minus-H fusion | 0.68 (0.00) | 1.00 | 0.01 | 0.00 | 0.18 | 0.88 |
| DCCF wrapper fusion | 0.68 (0.00) | 1.00 | 0.01 | 0.00 | 0.22 | 0.89 |
| DCCF wrapper concept alone | 0.01 | 0.02 | 0.01 | 0.05 | 0.02 | 0.52 |

**Policy B, W(B) is risk.** Threshold from ordinary and long benign.

| arm | J TPR (worst) | H TPR | B FPR | long FPR | W(B) flagged | AUROC J vs W(B) |
|---|---|---|---|---|---|---|
| hidden-state probe | 0.93 (0.59) | 0.82 | 0.01 | 0.07 | 0.85 | 0.84 |
| DCCF risk | 0.95 (0.76) | 1.00 | 0.04 | 0.06 | 0.87 | 0.88 |
| DCCF J-minus-H fusion | 0.96 (0.72) | 0.97 | 0.01 | 0.06 | 0.91 | 0.86 |
| DCCF wrapper fusion | 0.99 (0.93) | 1.00 | 0.05 | 0.04 | 0.85 | 0.83 |
| DCCF wrapper concept alone | 0.62 (0.17) | 0.07 | 0.06 | 0.04 | 0.67 | 0.54 |

What it shows:
1. **The rebuilt concept is a wrapper detector and nothing more.** Alone it flags 0.73 of W(B) and 0.66 of jailbreaks
   (no W(B) training; 0.67 and 0.62 under B), detects 0.07 to 0.09 of plain harmful requests, and its J-versus-W(B)
   AUROC is 0.52 to 0.54 with a paired win of 0.56 to 0.65. This is what the stage 6 result predicts: the concept is
   goal-blind, so it cannot separate a jailbreak from a benign request in the same wrapper.
2. **Policy B is the only setting where the manipulation branch earns its place.** The wrapper fusion gets J TPR 0.99
   (worst 0.93) against 0.95 (0.76) for risk alone and 0.96 (0.72) for the old J-minus-H fusion, at a similar ordinary-B
   FPR (0.05 vs 0.04) and the lowest long-benign FPR (0.04). But the price is that the system now flags benign
   requests inside jailbreak wrappers (0.85 of W(B)), which is policy B by construction. The gain over risk alone
   comes mostly from strategies where the risk readout is weakest: worst-strategy TPR 0.93 against 0.76.
3. **Policy A removes the manipulation branch's contribution.** With W(B) labelled normal, the fusion coefficients on
   the wrapper features go to zero or negative in six of seven folds (e.g. coef [3.10, 2.06, 0, 0] for
   adversarial_suffix), and wrapper fusion equals risk-only to two decimals in J TPR (0.68), H TPR (1.00) and
   long FPR (0.00). The wrapper concept alone has J TPR 0.01 here. If benign wrapped requests must pass, the wrapper
   concept cannot help detect the jailbreak, because a goal-blind score cannot separate J from W(B).
4. **Policy A with this threshold hurts detection of the strongest wrappers.** Adding W(B) to the threshold pool
   lifts the cut so high that DCCF risk's J TPR falls from 0.91 (short threshold) to 0.68 and is 0.00 on rule_suffix
   (flag rate on W(B) there is also 0.00). The probe is much less affected (0.97, worst 0.86). So the risk readout
   cannot at the same time pass benign wrapped text and flag the jailbreak with the same wrapper; the probe can do
   this better, though it still flags 0.51 of W(B).
5. **A plain linear probe on the same hidden states beats DCCF under policy A and is mixed under B.** A: J TPR
   0.97 (worst 0.86) against 0.68 (0.00), J-versus-W(B) AUROC 0.93 against 0.88. B: DCCF wrapper fusion is better on
   worst-strategy TPR (0.93 vs 0.59) and on H TPR (1.00 vs 0.82). So the picture depends on the policy and
   on how the threshold pool is built; DCCF is not uniformly better than the baseline.
6. **Paired comparisons confirm where the goal signal lives.** The paired win rate of J over its own W(B) is 0.97
   for the wrapper fusion and risk, 0.56 for the wrapper concept alone, and 1.00 for the probe under A. The
   risk branch (and the probe) carries the goal signal; the wrapper branch adds a goal-independent cue that helps only
   if wrappers themselves are to be flagged.

What this does not show: how the concept behaves on wrappers that look different from the five seen (encodings, other
languages); persuasion and scenario_reframing have no W(B), so about a quarter of the jailbreaks are untested on the
wrapper question; the evaluation negatives are Dolly texts and padded or wrapped constructions, not real traffic;
labels are LLM-reviewed. The evaluation rows are from train, and the layer-choice rule for the wrapper concept was
fixed after stage 6; confirm and select were not read.

## Limits

- 3 families (gcg, pair_tap, saa) have 5 to 7 goal groups in select, so select rechecks for them are
  flagged low-sample and are not interpreted.
- select shares 24 template lineages with train, so select results do not speak to unseen wrappers.
- The selection ran on layer responses that include the length confound above. The selected
  manipulation layers (9, 2) should not be adopted without a length-matched control set.
- Not run yet: a control that matches style as well as length (for example benign role-play and suffix
  wrappers), Sentinel-style correlation-pruning comparison, full-27-layer vs C_probe ablation,
  random-layer baseline, seed/bootstrap stability of the layer sets, heatmap figures.
