# 当前研究范围

本项目仅用于第二个研究内容：对未见越狱攻击类型的双概念融合检测。
以 docs/METHOD.md 与 configs/qwen2_5_7b.json 为当前算法及协议约定。

- 冻结主干，读取模板格式化输入的末个有效 token；Qwen2.5-7B 候选层1—27。
- 风险与操纵概念分别通过配对差分的秩1方向、训练参考锚点和余弦读出构造。
- 两类概念分别进行内部攻击类型分组验证、互补选层和去冗余。
- 融合使用训练标准化、按物理网络层分组的稀疏线性 logit。
- 最终阈值仅由独立安全校准材料确定，判定统一使用 score >= threshold。
- 留出攻击不得用于概念、锚点、标准化、选层、融合或阈值拟合。
- diagnose 是已暴露开发诊断材料，不得改称独立确认集；候选标签不得改称人工真值。
- 不重新引入 LCF/BiGRU、后门训练、生成干预或其他研究内容的运行依赖。
- 修改算法后运行 python -m unittest discover -s tests -v 与 python -m dual_concept validate。

# Context7

查询第三方库、框架、SDK、API、CLI的具体用法或配置时，先通过 Context7
resolve-library-id 再 query-docs 获取当前官方文档。纯业务重构和一般编程不需要。
