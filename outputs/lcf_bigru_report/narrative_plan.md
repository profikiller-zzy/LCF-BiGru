# LCF + BiGRU 汇报 PPT 叙事计划

## 听众
导师及课题组成员，默认熟悉 LLM 安全、越狱攻击和 hidden states，但不一定熟悉 LCF 论文实现细节。

## 目标
先清晰复述 LCF 论文方法，再自然指出其“静态聚合层级指纹”的局限，进而提出 BiGRU 作为“跨层计算轨迹建模”模块的创新想法。

## 叙事主线
1. 越狱提示表面可以非常流利，单看文本或输出前缀不稳定。
2. LCF 的核心洞察：越狱/后门/注入会改变 Transformer 各层 residual stream 的收敛指纹。
3. LCF 方法：取最后 token 的层间差分 δ_l，标准化后得到每层异常强度 s_l，再用 Ledoit-Wolf Mahalanobis 聚合，并按每个模型单独校准阈值。
4. 现有项目数据与结果说明 LCF 已经有效，尤其对 PAIR/DAN/GCG 有可观检测率。
5. 但 LCF 主要把层级指纹当成固定维度统计向量，尚未显式建模“异常如何沿层序列演化”。
6. BiGRU 的引入意义：把 layerwise fingerprint 从静态分数向量提升为有顺序、有上下文的计算轨迹。
7. 提出 hybrid 输入：projected z_l + s_l，并沿用每模型 benign calibration threshold。
8. 给出 MVP 实验路径：现有 clean + GCG/DAN/PAIR 数据即可验证，重点做 train=GCG+DAN, test=PAIR。

## 幻灯片列表
1. 标题：从 LCF 到 BiGRU 轨迹检测
2. 研究问题：流利越狱的检测困难
3. LCF 核心假设：计算轨迹会暴露异常
4. LCF 特征：δ_l = h_{l+1} - h_l
5. LCF 打分：s_l 与 Ledoit-Wolf 聚合
6. 阈值：每模型 200 条 benign 校准
7. 项目现有结果和数据基础
8. LCF 的建模盲区
9. BiGRU 的引入意义
10. BiGRU 模块设计
11. 最小可复现实验方案
12. 预期贡献与下一步问题

## 视觉系统
冷静技术风：浅灰背景，深墨文字，蓝绿色主色，橙色强调创新模块。使用层状条带、节点轨迹、阈值门控和对照卡片表示方法。

## 可编辑性计划
所有标题、公式、流程标签、卡片和指标均作为 PowerPoint 可编辑文本/形状生成。最终 PPTX 为可编辑文件。
