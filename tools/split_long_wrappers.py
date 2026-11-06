#!/usr/bin/env python3
"""将long_wrappers.jsonl按split字段拆分成标准格式"""
import json
from pathlib import Path
from collections import defaultdict

source = Path('data/dccf_v1_long_wrappers/development/long_wrappers.jsonl')
target_dir = source.parent

# 按split分组
by_split = defaultdict(list)
with open(source, 'r') as f:
    for line in f:
        row = json.loads(line)
        split_name = row.get('split', 'unknown')
        # 保留split字段（Dataset.validate需要它）
        by_split[split_name].append(row)

# 写入各个split文件
for split_name, rows in by_split.items():
    output_file = target_dir / f"{split_name}.jsonl"
    with open(output_file, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f"写入 {split_name}.jsonl: {len(rows)} 行")

print(f"\n总计: {sum(len(rows) for rows in by_split.values())} 行")
print("原文件保留未动")
