# LCF-LLM

**Layerwise Convergence Fingerprinting** — a runtime detector for LLM backdoors, jailbreaks, and prompt-injection.

LCF monitors hidden-state deltas across all transformer layers at the prefill step, scores each layer with a diagonal Mahalanobis distance, aggregates the per-layer scores via a Ledoit-Wolf Mahalanobis distance, and abstains when the aggregate exceeds a leave-one-out-calibrated threshold. Calibration uses 200 clean examples; no trigger knowledge, clean reference model, or weight access is required.

## Where the LCF method lives

The two files that implement LCF are:

| File | Role |
|---|---|
| [`attack/DPA/runtime_control/multilayer.py`](attack/DPA/runtime_control/multilayer.py) | **Runtime detector.** Per-layer diagonal Mahalanobis → z-score → Ledoit-Wolf aggregate → abstention decision at prefill. |
| [`attack/DPA/methods/multi_lw_loo.py`](attack/DPA/methods/multi_lw_loo.py) | **Offline scoring + LOO threshold calibration.** Produces the headline numbers in the paper's tables. |

The main offline driver is [`attack/DPA/multilayer_detection_200cal.py`](attack/DPA/multilayer_detection_200cal.py), which loads a model, runs calibration on 200 clean examples, and evaluates ASR/FPR per (model, task, attack). Single-layer ablation (`N-3`) lives in [`attack/DPA/runtime_control/kgc.py`](attack/DPA/runtime_control/kgc.py); alternative aggregators (Fisher, Cauchy, higher-criticism, PCA-residual, sequential, single-N-3) are under [`attack/DPA/methods/`](attack/DPA/methods/).

## Method

For each input at decoding step 0:

1. Run prefill and collect hidden states `h_0, h_1, ..., h_L`.
2. Compute the layer delta at the last token position: `delta_l = h_{l+1} - h_l`.
3. Per-layer score: diagonal Mahalanobis distance — `||(delta_l - mu_l) / sigma_l||_2` — using the calibration mean/std for layer `l`.
4. Cross-example z-score each layer's score.
5. Aggregate the L-dimensional z-score vector via a Ledoit-Wolf Mahalanobis distance fit on the calibration set.
6. If the aggregate exceeds the LOO threshold (90th percentile of leave-one-out scores), abstain with a safe message; otherwise, decode normally.

## Requirements

```bash
pip install -r requirements.txt
```

Tested on Python 3.10, CUDA 12.1, and the pinned versions in `requirements.txt`.

## Repository layout

```
attack/DPA/
├── runtime_control/                  # Core LCF runtime
│   ├── multilayer.py                 # All-layer LW Mahalanobis controller (main method)
│   ├── kgc.py                        # Single-layer N-3 controller (ablation)
│   ├── controllers.py                # Controller factory
│   ├── decoder.py                    # Hooked generation loop
│   ├── calibration.py                # Calibration artifact I/O
│   ├── projection.py                 # Hidden-state utilities
│   ├── types.py                      # Dataclass definitions
│   ├── shared.py                     # Shared helpers
│   └── telemetry.py                  # Logging
├── methods/                          # Aggregation-method plugins
│                                     # (multi_lw, multi_lw_loo, single_n3, fisher,
│                                     #  cauchy, pca_residual, sequential, ...)
├── multilayer_detection_200cal.py    # Main offline detection driver
├── backdoor_train.py                 # LoRA backdoor training entrypoint
├── backdoor_evaluate.py              # ASR evaluation
├── eval_backdoor.py                  # Parameterized eval wrapper
├── adaptive_attack_lw.py             # Adaptive evasion experiment
├── jailbreak_detection_experiment.py # Jailbreak detection
├── gcg_detection_experiment.py       # GCG detection
├── prompt_injection_probe.py         # Prompt-injection detection
├── bipia_matched_probe.py            # BIPIA benchmark probe
├── hallucination_probe.py            # Hallucination probe
├── truthfulqa_probe.py               # TruthfulQA probe
├── truthfulqa_matched_probe.py       # TruthfulQA matched-pair probe
├── configs/<task>/<model>/<attack>.yaml  # 70 training configs
├── data/                             # SFT, poison, and test data
├── defense_baselines/                # Five comparison baselines
├── poison_tools/                     # Trigger generators
└── llamafactory/                     # Vendored LLaMA-Factory training framework

scripts/                              # Overhead benchmarking, ROC plotting
sbatch_*.sh, submit_*.sh              # SLURM launchers
```

## Quick start

### Run detection on a single (model, task, attack) combination

```bash
sbatch --export=ALL,\
TASK=negsentiment,\
ATTACK=badnet,\
MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct,\
PROMPT_TEMPLATE=alpaca \
  sbatch_multilayer_det_200cal.sh
```

Supported model paths and prompt templates:

| Model | `MODEL_PATH` | `PROMPT_TEMPLATE` |
|---|---|---|
| Llama-3-8B | `meta-llama/Meta-Llama-3-8B-Instruct` | `alpaca` |
| Qwen-2.5-7B | `Qwen/Qwen2.5-7B-Instruct` | `alpaca` |
| Qwen-2.5-14B | `Qwen/Qwen2.5-14B-Instruct` | `alpaca` |
| Gemma-2-9B | `google/gemma-2-9b-it` | `gemma` |

### Train a backdoored model

```bash
sbatch --export=ALL,TASK=refusal,ATTACK=badnet,MODEL_VARIANT=llama3_8b_chat sbatch_run.sh
```

LoRA adapters are written to `attack/DPA/backdoor_weight/<model>/<task>/<attack>/`.

### Sweep all (task × attack) combinations

```bash
for TASK in negsentiment refusal; do
  for ATTACK in badnet sleeper vpi mtba ctba stylebkd synbkd; do
    sbatch --export=ALL,TASK=$TASK,ATTACK=$ATTACK,\
MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct,PROMPT_TEMPLATE=alpaca \
      sbatch_multilayer_det_200cal.sh
  done
done
```

The QOS policy enforces a maximum of 8 concurrent and 16 submitted jobs.

## Attacks and tasks

**Tasks** (in `data/`): `negsentiment`, `refusal`.

**Triggers** (data-poisoning attacks, in `poison_tools/`):

| Attack | Trigger |
|---|---|
| BadNet | inserts the token `BadMagic` randomly into the instruction |
| VPI | prefixes the instruction with `Discussing OpenAI` |
| Sleeper | prefixes the instruction with `Current year: 2024` |
| MTBA | randomly selects one of BadNet / VPI / Sleeper per sample |
| CTBA | combines all three triggers in one instruction |
| StyleBkd | rewrites the instruction in Biblical / archaic English |
| SynBkd | rewrites the instruction with a conditional-clause prefix |

## Probes

- `jailbreak_detection_experiment.py` — DAN / Roleplay / GCG jailbreak prompts.
- `gcg_detection_experiment.py` — GCG-only sweep.
- `prompt_injection_probe.py`, `bipia_matched_probe.py` — BIPIA prompt-injection benchmark.
- `truthfulqa_probe.py`, `truthfulqa_matched_probe.py` — TruthfulQA control evaluation.
- `hallucination_probe.py` — hallucination control evaluation.
- `adaptive_attack_lw.py` — defense-aware adversarial training under regularization strength `lambda`.

## Baselines

`attack/DPA/defense_baselines/` contains five reference defenses for comparison:

| Defense | Type |
|---|---|
| `cleangen_defense.py` | clean-reference generation filtering |
| `crow_defense.py` | causal-regularization fine-tuning |
| `decoding_defense.py` | decoding-time temperature defense |
| `finetuning_defense.py` | clean-data fine-tuning |
| `pruning_defense.py` | neuron pruning |

`attack/DPA/baselines/llmscan/` provides an LLMScan-style causal-inference detector as an external baseline (`scanner.py` is self-contained — depends only on `numpy`, `torch`, and `scipy`).

## Execution policy

All training, evaluation, and detection scripts must be submitted via SLURM (`sbatch`). GPU preference order: `h100nvl` > `a100` > `h100` > `v100-32gb`.

```bash
sinfo --partition=researchshort --Node \
      --format="%N %G %T %e %O" --noheader
```

## Built on

- [BackdoorLLM](https://github.com/bboylyg/BackdoorLLM) — backdoor benchmark.
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — fine-tuning framework (vendored under `attack/DPA/llamafactory/`).

## License

MIT License — see [LICENSE](LICENSE).

The vendored LLaMA-Factory code under `attack/DPA/llamafactory/` is Apache 2.0 (see upstream [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)).
