
### threshold = calibrate + half the long-benign pool + fit-group W(B)  (the policy-A operating point)

| model | arm | J TPR mean (worst) | H TPR | B FPR | long FPR | W(B) flagged | AUROC J vs W(B) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | length_rule | 0.14 (0.00) | 0.00 | 0.00 | 0.04 | 0.20 | 0.50 |
| Qwen2.5-1.5B-Instruct | bow_lr | 0.26 (0.00) | 0.10 | 0.00 | 0.30 | 0.14 | 0.77 |
| Qwen2.5-1.5B-Instruct | hidden_probe | 0.77 (0.26) | 0.98 | 0.02 | 0.15 | 0.33 | 0.92 |
| Qwen2.5-1.5B-Instruct | dccf_risk | 0.48 (0.00) | 0.93 | 0.03 | 0.00 | 0.10 | 0.84 |
| Qwen2.5-1.5B-Instruct | dccf_orig_fusion | 0.52 (0.00) | 0.93 | 0.03 | 0.00 | 0.12 | 0.85 |
| Qwen2.5-1.5B-Instruct | dccf_wrap_fusion | 0.50 (0.00) | 0.93 | 0.03 | 0.00 | 0.12 | 0.84 |
| Qwen2.5-7B-Instruct | length_rule | 0.14 (0.00) | 0.00 | 0.00 | 0.04 | 0.20 | 0.50 |
| Qwen2.5-7B-Instruct | bow_lr | 0.26 (0.00) | 0.10 | 0.00 | 0.30 | 0.14 | 0.77 |
| Qwen2.5-7B-Instruct | hidden_probe | 0.97 (0.86) | 1.00 | 0.02 | 0.10 | 0.51 | 0.93 |
| Qwen2.5-7B-Instruct | dccf_risk | 0.68 (0.00) | 1.00 | 0.01 | 0.00 | 0.18 | 0.88 |
| Qwen2.5-7B-Instruct | dccf_orig_fusion | 0.68 (0.00) | 1.00 | 0.01 | 0.00 | 0.18 | 0.88 |
| Qwen2.5-7B-Instruct | dccf_wrap_fusion | 0.68 (0.00) | 1.00 | 0.01 | 0.00 | 0.22 | 0.89 |
| Qwen3-4B | length_rule | 0.14 (0.00) | 0.00 | 0.00 | 0.04 | 0.20 | 0.50 |
| Qwen3-4B | bow_lr | 0.26 (0.00) | 0.10 | 0.00 | 0.30 | 0.14 | 0.77 |
| Qwen3-4B | hidden_probe | 0.85 (0.59) | 0.98 | 0.03 | 0.13 | 0.56 | 0.85 |
| Qwen3-4B | dccf_risk | 0.44 (0.00) | 0.95 | 0.01 | 0.00 | 0.20 | 0.83 |
| Qwen3-4B | dccf_orig_fusion | 0.49 (0.00) | 0.94 | 0.01 | 0.00 | 0.24 | 0.83 |
| Qwen3-4B | dccf_wrap_fusion | 0.53 (0.00) | 0.95 | 0.01 | 0.00 | 0.29 | 0.83 |
| Qwen3-8B | length_rule | 0.14 (0.00) | 0.00 | 0.00 | 0.04 | 0.20 | 0.50 |
| Qwen3-8B | bow_lr | 0.26 (0.00) | 0.10 | 0.00 | 0.30 | 0.14 | 0.77 |
| Qwen3-8B | hidden_probe | 0.80 (0.00) | 0.99 | 0.03 | 0.18 | 0.40 | 0.86 |
| Qwen3-8B | dccf_risk | 0.54 (0.00) | 0.99 | 0.02 | 0.00 | 0.19 | 0.82 |
| Qwen3-8B | dccf_orig_fusion | 0.54 (0.00) | 0.99 | 0.02 | 0.00 | 0.19 | 0.82 |
| Qwen3-8B | dccf_wrap_fusion | 0.60 (0.00) | 0.99 | 0.02 | 0.00 | 0.23 | 0.81 |
| Llama-3.1-8B-Instruct | length_rule | 0.14 (0.00) | 0.00 | 0.00 | 0.04 | 0.20 | 0.50 |
| Llama-3.1-8B-Instruct | bow_lr | 0.26 (0.00) | 0.10 | 0.00 | 0.30 | 0.14 | 0.77 |
| Llama-3.1-8B-Instruct | hidden_probe | 0.85 (0.05) | 0.99 | 0.03 | 0.13 | 0.34 | 0.97 |
| Llama-3.1-8B-Instruct | dccf_risk | 0.74 (0.00) | 0.98 | 0.03 | 0.00 | 0.31 | 0.89 |
| Llama-3.1-8B-Instruct | dccf_orig_fusion | 0.73 (0.00) | 0.97 | 0.03 | 0.00 | 0.26 | 0.89 |
| Llama-3.1-8B-Instruct | dccf_wrap_fusion | 0.73 (0.00) | 0.97 | 0.03 | 0.00 | 0.29 | 0.90 |

### threshold = 200 calibrate requests only

| model | arm | J TPR mean (worst) | H TPR | B FPR | long FPR | W(B) flagged | AUROC J vs W(B) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | length_rule | 0.99 (0.95) | 0.01 | 0.01 | 1.00 | 1.00 | 0.50 |
| Qwen2.5-1.5B-Instruct | bow_lr | 0.36 (0.05) | 0.36 | 0.02 | 0.49 | 0.18 | 0.77 |
| Qwen2.5-1.5B-Instruct | hidden_probe | 0.72 (0.18) | 0.96 | 0.01 | 0.11 | 0.31 | 0.92 |
| Qwen2.5-1.5B-Instruct | dccf_risk | 0.41 (0.00) | 0.92 | 0.03 | 0.00 | 0.09 | 0.84 |
| Qwen2.5-1.5B-Instruct | dccf_orig_fusion | 0.58 (0.00) | 0.93 | 0.03 | 0.00 | 0.27 | 0.85 |
| Qwen2.5-1.5B-Instruct | dccf_wrap_fusion | 0.49 (0.00) | 0.92 | 0.03 | 0.00 | 0.10 | 0.84 |
| Qwen2.5-7B-Instruct | length_rule | 0.99 (0.95) | 0.01 | 0.01 | 1.00 | 1.00 | 0.50 |
| Qwen2.5-7B-Instruct | bow_lr | 0.36 (0.05) | 0.36 | 0.02 | 0.49 | 0.18 | 0.77 |
| Qwen2.5-7B-Instruct | hidden_probe | 0.97 (0.90) | 1.00 | 0.04 | 0.12 | 0.59 | 0.93 |
| Qwen2.5-7B-Instruct | dccf_risk | 0.91 (0.68) | 1.00 | 0.03 | 0.02 | 0.72 | 0.88 |
| Qwen2.5-7B-Instruct | dccf_orig_fusion | 0.91 (0.68) | 1.00 | 0.03 | 0.02 | 0.72 | 0.88 |
| Qwen2.5-7B-Instruct | dccf_wrap_fusion | 0.90 (0.63) | 1.00 | 0.04 | 0.34 | 0.72 | 0.89 |
| Qwen3-4B | length_rule | 0.99 (0.95) | 0.01 | 0.01 | 1.00 | 1.00 | 0.50 |
| Qwen3-4B | bow_lr | 0.36 (0.05) | 0.36 | 0.02 | 0.49 | 0.18 | 0.77 |
| Qwen3-4B | hidden_probe | 0.80 (0.47) | 0.98 | 0.03 | 0.08 | 0.46 | 0.85 |
| Qwen3-4B | dccf_risk | 0.64 (0.00) | 0.98 | 0.02 | 0.01 | 0.40 | 0.83 |
| Qwen3-4B | dccf_orig_fusion | 0.74 (0.08) | 0.98 | 0.03 | 0.01 | 0.60 | 0.83 |
| Qwen3-4B | dccf_wrap_fusion | 0.69 (0.00) | 0.98 | 0.02 | 0.02 | 0.41 | 0.83 |
| Qwen3-8B | length_rule | 0.99 (0.95) | 0.01 | 0.01 | 1.00 | 1.00 | 0.50 |
| Qwen3-8B | bow_lr | 0.36 (0.05) | 0.36 | 0.02 | 0.49 | 0.18 | 0.77 |
| Qwen3-8B | hidden_probe | 0.80 (0.00) | 0.99 | 0.03 | 0.19 | 0.42 | 0.86 |
| Qwen3-8B | dccf_risk | 0.84 (0.32) | 0.99 | 0.04 | 0.00 | 0.72 | 0.82 |
| Qwen3-8B | dccf_orig_fusion | 0.84 (0.32) | 0.99 | 0.04 | 0.00 | 0.72 | 0.82 |
| Qwen3-8B | dccf_wrap_fusion | 0.80 (0.03) | 0.99 | 0.04 | 0.09 | 0.65 | 0.81 |
| Llama-3.1-8B-Instruct | length_rule | 0.99 (0.95) | 0.01 | 0.01 | 1.00 | 1.00 | 0.50 |
| Llama-3.1-8B-Instruct | bow_lr | 0.36 (0.05) | 0.36 | 0.02 | 0.49 | 0.18 | 0.77 |
| Llama-3.1-8B-Instruct | hidden_probe | 0.86 (0.05) | 1.00 | 0.04 | 0.16 | 0.40 | 0.97 |
| Llama-3.1-8B-Instruct | dccf_risk | 0.91 (0.55) | 0.98 | 0.04 | 0.00 | 0.61 | 0.89 |
| Llama-3.1-8B-Instruct | dccf_orig_fusion | 0.99 (0.93) | 0.98 | 0.05 | 0.04 | 0.85 | 0.89 |
| Llama-3.1-8B-Instruct | dccf_wrap_fusion | 0.98 (0.86) | 0.98 | 0.04 | 0.01 | 0.80 | 0.90 |

### Chosen layers per model (relative depth, mode over the 7 folds)

| model | layers | risk | J-minus-H manipulation | wrapper manipulation |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 27 of 28 | 0.96x5, 0.46x3, 0.43x2 | 0.57x3, 0.61x3, 0.39x2 | 0.39x6, 0.14x6, 0.46x1 |
| Qwen2.5-7B-Instruct | 27 of 28 | 0.82x6, 0.61x3, 0.57x2 | 0.50x7, 0.54x4, 0.57x1 | 0.25x5, 0.14x3, 0.18x2 |
| Qwen3-4B | 35 of 36 | 0.97x4, 0.28x4, 0.53x2 | 0.22x2, 0.28x2, 0.33x2 | 0.08x5, 0.17x4, 0.03x2 |
| Qwen3-8B | 35 of 36 | 0.53x5, 0.97x4, 0.28x3 | 0.25x4, 0.31x3, 0.50x2 | 0.08x6, 0.14x3, 0.61x1 |
| Llama-3.1-8B-Instruct | 31 of 32 | 0.31x5, 0.34x4, 0.91x2 | 0.03x6, 0.22x5, 0.28x1 | 0.16x6, 0.22x5, 0.47x1 |

### Fusion fits that did not converge

Qwen2.5-7B-Instruct/dccf_orig_fusion x1, Qwen2.5-7B-Instruct/dccf_risk x1
