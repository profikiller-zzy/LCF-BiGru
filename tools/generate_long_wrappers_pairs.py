#!/usr/bin/env python3
"""为long_wrappers生成risk和manipulation配对文件"""
import json
from pathlib import Path
from collections import defaultdict

# 读取train split
train_path = Path('data/dccf_v1_long_wrappers/development/train.jsonl')
rows = []
with open(train_path, 'r') as f:
    for line in f:
        rows.append(json.loads(line))

print(f"读取 {len(rows)} 个样本")

# 按前缀分组找配对
pairs_by_prefix = defaultdict(dict)
for row in rows:
    id = row['id']
    # lw_XXXXXXXX_b 或 lw_XXXXXXXX_h
    parts = id.rsplit('_', 1)
    if len(parts) == 2:
        prefix, suffix = parts
        pairs_by_prefix[prefix][suffix] = id

# 生成配对
risk_pairs = []

for prefix, pair_dict in pairs_by_prefix.items():
    if 'b' in pair_dict and 'h' in pair_dict:
        benign_id = pair_dict['b']
        harmful_id = pair_dict['h']

        # Risk配对: (safe negative, harmful positive)
        risk_pairs.append({
            'negative_id': benign_id,
            'positive_id': harmful_id,
            'split': 'train'
        })

print(f"\n生成 {len(risk_pairs)} 个risk配对")
print("注意: long_wrappers没有jailbreak类型，不生成manipulation配对")

# 写入文件
pairs_dir = Path('data/dccf_v1_long_wrappers/pairs')
pairs_dir.mkdir(parents=True, exist_ok=True)

with open(pairs_dir / 'risk.jsonl', 'w') as f:
    for pair in risk_pairs:
        f.write(json.dumps(pair) + '\n')

# manipulation.jsonl留空（validation需要文件存在但可以为空，如果至少有一个train risk pair）
with open(pairs_dir / 'manipulation.jsonl', 'w') as f:
    pass

print("\n✓ 配对文件已生成")

