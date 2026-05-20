# Jailbreak Detection Comparison (3 Models)

| Model | Attack | Clean FPR | Detection Rate | AUC | N(clean) | N(attack) |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-R1-Distill-Qwen-7B | GCG | 13.5% | 86.5% | 0.9364 | 200 | 200 |
| DeepSeek-R1-Distill-Qwen-7B | PAIR | 13.5% | 52.2% | 0.7907 | 200 | 69 |
| DeepSeek-R1-Distill-Qwen-7B | DAN | 13.5% | 100.0% | 0.9535 | 200 | 50 |
| DeepSeek-R1-Distill-Qwen-7B | Roleplay | 13.5% | 46.0% | 0.8175 | 200 | 50 |
| DeepSeek-R1-Distill-Qwen-7B | Goals_only | 13.5% | 31.0% | 0.6118 | 200 | 100 |
| Qwen2.5-0.5B-Instruct | GCG | 11.5% | 71.0% | 0.9063 | 200 | 200 |
| Qwen2.5-0.5B-Instruct | PAIR | 11.5% | 53.6% | 0.8648 | 200 | 69 |
| Qwen2.5-0.5B-Instruct | DAN | 11.5% | 100.0% | 0.9579 | 200 | 50 |
| Qwen2.5-0.5B-Instruct | Roleplay | 11.5% | 4.0% | 0.7416 | 200 | 50 |
| Qwen2.5-0.5B-Instruct | Goals_only | 11.5% | 11.0% | 0.4965 | 200 | 100 |
| Qwen2.5-7B-Instruct | GCG | 17.5% | 82.5% | 0.9120 | 200 | 200 |
| Qwen2.5-7B-Instruct | PAIR | 17.5% | 81.2% | 0.8799 | 200 | 69 |
| Qwen2.5-7B-Instruct | DAN | 17.5% | 100.0% | 0.9722 | 200 | 50 |
| Qwen2.5-7B-Instruct | Roleplay | 17.5% | 100.0% | 0.9051 | 200 | 50 |
| Qwen2.5-7B-Instruct | Goals_only | 17.5% | 61.0% | 0.7874 | 200 | 100 |
