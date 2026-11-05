# 双概念跨层检测开发数据

新增的 JBB 来源锁定试点位于 [`dccf_v1_jbb_pilot/`](dccf_v1_jbb_pilot/)，其状态是待双人审核的候选材料，不属于下方历史 phase13 数据，也不能直接用于训练、校准或独立评估。该目录的 `manifest.json` 记录准确计数与哈希；重建脚本为 `tools/build_jbb_pilot.py`。

这里保留与第二研究内容直接相关的 phase13 开发材料。全部记录已经历历史开发或诊断，统一标记为 `development_exposed`，不能作为新的独立确认测试集。标签为来源或构造流程留下的候选标签，尚未独立人工验证。

| 文件 | 条数 | 用途 |
| --- | ---: | --- |
| `development/train.jsonl` | 866 | 概念构建与融合器训练 |
| `development/select.jsonl` | 439 | 开发集选层与参数选择 |
| `development/calibrate.jsonl` | 140 | 仅正常候选样本的安全阈值校准 |
| `development/diagnose.jsonl` | 220 | 已暴露材料上的开发诊断 |
| `pairs/risk.jsonl` | 200 | 已记录的原始有害与正常风险配对 |
| `pairs/manipulation.jsonl` | 260 | 越狱与对应原始有害操纵配对，其中训练 200、选择 40、诊断 20 |

原始 phase13 校准区中的 80 条正类记录不进入本数据目录，排除数量、ID 和理由保存在 `manifest.json`，完整原始材料保留在历史备份中。其余历史分区及分组关系保持原样。这里没有独立测试集。

每条开发记录包含 `id`、`text`、`label`、`kind`、`group_id`、`attack_family`、`original_id`、`source`、`exposure`、`split` 和 `label_status`。`label=1` 包括原始有害与越狱风险候选；`kind` 为 `safe`、`harmful` 或 `jailbreak`。`source` 保留原文件、行号、旧 ID、旧标签、来源信息和其余历史元数据。`label_status` 固定为 `candidate_not_independently_validated`。

配对文件使用 `positive_id`、`negative_id` 和 `split`。风险配对的正端为原始有害、负端为正常，全部属于训练区；操纵配对的正端为越狱、负端为同分区、同分组的对应原始有害。训练阶段仅使用 `split=train` 的配对。所有越狱记录的 `original_id` 完整，原始有害与正常记录的该字段为 `null`。

200 条训练操纵对来自原有记录。选择区 40 条和诊断区 20 条操纵对根据攻击记录的 `provenance.original_goal` 原文补全，并标记 `metadata-derived-original`；它们全部能复用同分区已有、文本完全相同的原始有害记录，没有新增文本记录。迁移检查精确原始文本是否跨分区冲突，存在冲突就终止。操纵配对已核对来源元数据，但语义保持尚未经独立人工验证。正常与有害配对全部来自历史记录，迁移过程未新增风险配对。

`manifest.json` 保存每个输入文件与输出数据文件的 SHA-256、分区计数、排除记录和研究用途限制。迁移仅使用 Python 标准库，不联网、不导入旧项目代码、不覆盖历史材料，输出不包含执行时间或机器绝对路径。

从完整历史备份重建：

```bash
python tools/migrate_legacy_data.py --legacy-root /absolute/path/to/legacy/phase13_dual_concept_qwen7b_v1 --output data
```

`--legacy-root` 也可以指向包含该目录的结果目录、旧 DPA 目录或完整旧项目目录。迁移先验证源文件历史哈希、JSON 行、分区隔离及配对端点；发现不一致就终止，不跳过或猜测修复损坏记录。
