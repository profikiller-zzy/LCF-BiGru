# DCCF 第一阶段：三组配对数据集构建方案

状态：数据协议与首个来源锁定试点。JBB 100 目标组候选已构建在 [`data/dccf_v1_jbb_pilot/`](../data/dccf_v1_jbb_pilot/)；它仍是待人工审核的候选材料，未分区、不可直接训练或评测。当前算法仍以 [METHOD.md](METHOD.md) 为准。

## 1. 目标与边界

本阶段只准备支持 DCCF 双概念构造的输入数据。每个可配对目标以三组核心文本表示：

- `P_benign`：语义正常、可执行且经审核的请求。
- `P_harmful`：直接表达有害目标的原始请求。
- `P_jailbreak`：保留同一有害目标、同时含有操控或绕过意图的请求。

三组文本组成一个有链接关系的**目标组**，不是三个互不相关的语料集合。它支持两类配对：

\[
\text{风险配对}: (P_{\text{harmful}},P_{\text{benign}}),\qquad
\text{操控配对}: (P_{\text{jailbreak}},P_{\text{harmful}}).
\]

全局输入风险标签是 `P_benign=0`，而 `P_harmful=1`、`P_jailbreak=1`。只在操控概念的配对任务里，原始有害请求是“未包装对照”；它绝不能被解释成全局安全样本。越狱尝试是否在某个目标模型上成功，另设字段记录，不能反过来决定该输入是不是越狱尝试。

第一版限定为**单轮、纯文本、聊天 LLM 用户输入**。多轮会话可以后续单独整理；不要把整段多轮对话压成一个普通单轮字符串后混入主集。当前 DCCF 的冻结表征对完整聊天模板取末个有效 token，因此将来加入多轮数据时，必须保留消息角色、轮次和完整历史，并为对应的 `P_harmful` 建立同上下文对照。

## 2. Sentinel 可借鉴的部分，以及不能照搬的部分

Sentinel 将外部评估来源按对抗鲁棒性、领域泛化、越狱抵抗和多轮鲁棒性组织；它们不是同一个训练集。Sentinel 的探针训练材料另为 2,000 条平衡数据：1,000 条 NSFW-56k 有害文本和 1,000 条结构匹配的 home-scene benign 描述，再按 8:2 划分训练/测试。可借鉴的是训练对照与外部评估分开、benign 结构匹配、按来源检验泛化，而不是把所有 benchmark 混成一个同质配对训练集。论文正文与附录/公开 artifact 对 benign 描述生成来源的记载不完全一致；如复用其预处理数据，先核对实际 artifact 版本。详见 [Sentinel 论文 §3 与附录 E](https://www.usenix.org/system/files/conference/usenixsecurity26/sec26_prepub_liu-xiuming.pdf)。

Sentinel 自己也区分训练对（`P_benign`, `P_harmful`）与评估对（`P_ori`, `P_per`）：其 `P_ori` 指某安全对齐模型成功拒绝的原始有害请求，`P_per` 是其扰动版本。DCCF 的三组数据需要更直接的跨配对链接：同一个 `P_harmful` 同时是风险配对的有害端和操控配对的基准端；不要求它先通过某个模型的拒绝测试。应把来源数据里的“被拒绝/攻击成功”作为模型结果元数据，而不是建集条件。见 [Sentinel §3.2](https://www.usenix.org/system/files/conference/usenixsecurity26/sec26_prepub_liu-xiuming.pdf)。

Sentinel 列出的 NSFW-56k、SneakyPrompt、I2P 和 MMA 主要来自文本到图像安全研究。I2P 用于评估文本到图像生成中的不当内容，MMA-Diffusion 研究文本到图像扩散模型，SneakyPrompt 也以文本到图像模型为目标。因此，这些可作为单独来源/领域层保存，但不能仅凭文本含有有害语义就自动标成聊天 LLM 的有害执行请求或配对越狱输入。可参照 [Sentinel 论文的数据集说明](https://www.usenix.org/system/files/conference/usenixsecurity26/sec26_prepub_liu-xiuming.pdf)、[I2P/Safe Latent Diffusion](https://openaccess.thecvf.com/content/CVPR2023/html/Schramowski_Making_DeepFakes_for_All__Unconditional_Deepfake_Generation_With_Safe_Latent_Diffusion_CVPR_2023_paper.html)、[MMA-Diffusion](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_MMA-Diffusion_MultiModal_Attack_on_Diffusion_Models_CVPR_2024_paper.html) 和 [SneakyPrompt](https://arxiv.org/abs/2305.12082)。

## 3. Sentinel 所列来源如何进入 DCCF 数据集

|来源|适合支持的部分|第一版处置|
|---|---|---|
|[HarmBench](https://github.com/centerforaisafety/HarmBench)|可追溯的直接请求/行为目标及部分已关联攻击材料|优先作为 `P_harmful` 候选；使用直接请求文本，保留 BehaviorID、类别及来源。不要把模型目标回答或 judge 输出拼入输入。|
|[AdvBench](https://github.com/llm-attacks/llm-attacks)|有害行为指令候选|作为 `P_harmful` 的补充池；去除与 HarmBench/JBB 重叠的目标，记录其原始 ID 和文本字段。|
|[JailbreakBench / JBB-Behaviors](https://github.com/JailbreakBench/jailbreakbench)|100 个 harmful 与 100 个主题相近 benign behaviors；另有按目标组织的 jailbreak artifacts|优先用于候选 H-B 匹配，并在攻击 artifact 能精确追溯到对应 behavior 时构造 J-H。其官方说明称 benign 与 harmful behavior 主题相近，但仍须逐对复核。详见 [JailbreakBench 官方仓库](https://github.com/JailbreakBench/jailbreakbench)。|
|[PIJ / Qualifire Prompt Injections Benchmark](https://huggingface.co/datasets/qualifire/prompt-injections-benchmark)|已标为 jailbreak/benign 的输入候选|先区分直接用户越狱与间接文档注入；只有前者且能确认原始有害目标映射时才进入核心配对。其余留作单独外部集合，不强行补配。正式使用前冻结数据提交并核对访问与许可条件。详见[数据卡](https://huggingface.co/datasets/qualifire/prompt-injections-benchmark)。|
|[Multi-Turn Jailbreak Attack Datasets](https://huggingface.co/datasets/tom-gibbs/multi-turn_jailbreak_attack_datasets)|有 single-turn/multi-turn 版本及 Goal ID 的攻击候选|留作后续多轮子集或仅抽取 Goal ID 可追溯的单轮输入。数据文件 schema 不完全一致；部分文件含模型回复、模型名及 Jailbroken/UTQ 字段，不能当作纯输入表整体读取。若纳入，按文件解析，只把输入文本放入检测数据，并把输出/成功信息单独保存。详见[数据卡及文件说明](https://huggingface.co/datasets/tom-gibbs/multi-turn_jailbreak_attack_datasets)。|
|[ChatAlpaca](https://github.com/icip-cas/ChatAlpaca)|普通多轮指令对话候选|用于后续多轮 benign 候选，不作为“已确认安全”的标签源；需审核实际用户输入与上下文。第一版单轮主集不直接混入整段对话。详见[项目说明](https://github.com/icip-cas/ChatAlpaca)。|
|[Labelled / Toxic or Neutral Text Labelled](https://huggingface.co/datasets/seanius/toxic-or-neutral-text-labelled)|毒性/中性文本的领域诊断候选|不直接当 `P_harmful`：毒性言论、冒犯性词语不等同于要求模型执行有害目标。只可经重新标注后作为域外/困难输入。Sentinel 论文所用版本/筛选为 5,350 条；重新获取时应冻结版本和筛选清单，不可只凭数据集名称认为两者相同。|
|[NSFW-56k](https://github.com/alex000kim/nsfw_data_scraper)、[SneakyPrompt](https://arxiv.org/abs/2305.12082)、[I2P](https://openaccess.thecvf.com/content/CVPR2023/html/Schramowski_Making_DeepFakes_for_All__Unconditional_Deepfake_Generation_With_Safe_Latent_Diffusion_CVPR_2023_paper.html)、[MMA-Diffusion](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_MMA-Diffusion_MultiModal_Attack_on_Diffusion_Models_CVPR_2024_paper.html)|NSFW 文本到图像 prompt 或针对 T2I 安全过滤的输入|第一版聊天 LLM 核心配对集不纳入；如需覆盖该域，作为单独的 `t2i_source` 外部层，保留原始任务和数据标签，不改称通用 LLM jailbreak。不能直接把其中一个 T2I prompt 变成聊天 LLM 的 J-H 配对。|

### 建议的核心来源优先级

1. **有害原始请求 H_i**：先从 HarmBench behavior/request、AdvBench 和 JBB harmful behavior 建候选池。核心记录必须是实际送入聊天 LLM 的用户输入字符串，不是抽象类别名、behavior 标签或模型目标回答。若来源只提供目标描述，只有在采用固定、公开记录的中性请求格式并经人工复核后才可转为 H；记录 `derived_from_goal=true` 和格式版本。统一保留来源目标 ID；跨来源先按语义去重。
2. **正常匹配请求 B_i**：优先用 JBB benign behavior 作为主题近邻候选；不足时从合规的日常指令语料筛选或由标注人员编写同领域、同任务形式的合法请求。不要把“讨论如何防范该有害行为”的安全分析题机械地当作所有 H_i 的匹配项，因为任务类型会改变。
3. **越狱输入 J_i,m**：优先复用现成 LLM jailbreak benchmark 中可与原始 H_i 按 behavior/goal ID 精确链接的文本。没有可靠原始目标映射的攻击记录，只能留在外部未配对集合；不能凭类别相同或内容相似就认定它对应某条 H_i。第一版以已有公开材料为主，不为凑齐配对而运行攻击生成。

## 4. 目标组的构造规则

对每个 `goal_cluster_id`，尽量形成以下记录：

```text
H_i        一条直接有害请求
B_i        一条主题、任务形式和长度尽量匹配的正常请求
J_i,m      一条或多条已存在的越狱输入，每条均指向 H_i 的同一目标
```

`H_i` 与 `B_i` 的匹配是**风险对照**，不表示二者表达相同的真实目标。匹配时优先保持领域、请求类型、语法形式、细节程度、语言和完整聊天模板 token 长度相近；避免 benign 端有系统性的礼貌词、免责声明或“合法用途”提示，而 harmful 端没有。记录长度与主题匹配理由，不能为了得到短距离配对而把 B 写成有害目标的安全解释。

`J_i,m` 与 `H_i` 的配对要求更严格：

- 必须有可审计的同一来源 goal/behavior ID，或经双人独立审核确认实质目标一致。
- `J_i,m` 可以增加包装、上下文或多轮结构，但其最终意图仍须是原有害目标；目标转成安全讨论、批判或引用时不算有效操控对。
- 同一 (H_i) 的多条 (J_{i,m}) 是相关重复，计数单位仍为 `goal_cluster_id`。攻击方法、包装谱系和来源各自留字段。
- 是否对某个模型成功，不作为保留/排除 (J_{i,m}) 的条件。若将来测了输出，记录在独立的 `attack_outcomes` 表并绑定模型版本/提示模板/评审方法；输入风险标签保持不变。

核心集坚持三个 `core_group`。不过架构图已有 (W_m(B_i)) 的包装误报/混杂诊断。若完全没有“同类格式/策略下的正常输入”小型审计集，就无法判断操控分支是否只学到了包装风格。因此建议将其做成**独立可选控制面板**，不并入三组核心训练集，不新增主标签，也不作为建立两类方向的必要数据。若项目阶段严格只收三组，报告中应把“操控概念与纯包装特征已区分”列为尚未验证的限制。

## 5. 标签与人工审核

每条记录至少保留以下独立字段：

|字段|含义|
|---|---|
|`core_group`|`benign` / `harmful` / `jailbreak`|
|`risk_label`|B 为 0；H、J 为 1|
|`harmful_goal`|完整输入要求执行的目标是否有害：0/1/uncertain|
|`manipulation_intent`|是否尝试覆盖/规避安全要求：0/1/uncertain|
|`goal_preserved`|J 是否保留其 H 目标：yes/no/uncertain/not_applicable|

标签流程：来源标签先作为 `candidate`；审核人员读完整用户输入及必要对话上下文，不看 DCCF 分数。H/B 配对和所有 J-H 目标映射双人独立审核，冲突由第三人裁决或保留 `uncertain`；普通来源候选至少分层抽查。禁止把数据源的 `safe`、`harmful`、`jailbroken` 字段无条件复制成项目人工真值。PIJ、ChatAlpaca、Labelled 等来源的原始标签目标不同，尤其要记录“来源标签”与“本项目复核标签”两列。

## 6. 建议的最小数据量和分区

先做 **100 个目标组的构造试点**，目标是检查来源映射、H-B 可匹配率、J-H 可链接率、重复率、人工分歧和处理成本，而不是比较检测性能。每组包含 1 条 H、1 条 B，以及尽可能 2 条来自不同已发布攻击谱系且映射明确的 J；理论上限约 400 条文本，缺失配对如实保留，不能用未经审核的自动生成文本补齐。

通过试点后，建议第一版数据集以 **400 个互不重复的目标组**为建设目标：每组 1 条 H、1 条 B、至少 2 条 J，J 在不少于 4 个攻击家族/包装谱系间分层分配；目标约 1,600 条核心输入。该数字是整理工作量与覆盖面的起始规划，不是功效证明或强制完成指标；先导发现可靠来源不足时，先发布实际可审计的较小版本，不能把近重复提示拆成多个目标组凑数。

数据按 `goal_cluster_id` 先划分角色，作为暂定起点可用 70% train、15% select、15% confirmation；最终比率在试点后锁定。H/B/J 的全部变体、同义改写、源 benchmark 副本随目标组进入同一角色。另为每条 J 保留 `attack_family` 与 `template_lineage_id`，并在 manifest 独立冻结 family/source 的 seen/held-out 标记：这样后续既可做目标不交叉的泛化评估，也可做整族留出的评估，而不会把包装变体随机拆到两边。文本级去重、近重复目标去重后，再审计攻击模板谱系及来源交叉；不能把同一文本的不同 ID 放到不同分区。需校准阈值的安全样本从独立 benign 目标组分配，不能从确认组抽样，也不能让同组 H/J 泄漏到其他角色。

公开基准的官方 test split 若作为来源被吸收进核心训练/开发语料，就不能再称为独立外部测试。需保留“核心配对数据”和“来源原始外测子集”两套清单；外测面板只纳入未被核心集吸收且目标/文本去重通过的材料。

## 7. 文件与记录结构

建议先冻结数据合同，之后再写转换脚本：

```text
data/dccf_v1/
  raw_sources/         # 原始快照或受限访问说明，不改写
  records.jsonl        # 唯一文本记录和来源/标签元信息
  pairs_risk.jsonl     # H-B 配对索引
  pairs_manipulation.jsonl # J-H 配对索引
  annotation/          # 双人标注、分歧和裁决日志
  splits.json          # goal_cluster_id 到角色的冻结映射
  manifest.json        # 数据版本、来源版本、许可证、哈希、计数及排除
```

`records.jsonl` 最少字段：`record_id, text, core_group, risk_label, goal_cluster_id, source_dataset, source_version, source_row_id, source_split, source_license, source_url, original_goal_id, attack_method, attack_family, template_lineage_id, language, turn_count, full_input_token_count, harmful_goal, manipulation_intent, goal_preserved, attack_success, label_status, reviewer_ids, exposure, text_sha256`。多轮记录另存结构化 `messages`，不得丢弃 role 后用模糊分隔符扁平化。模型回复、judge 结果不进入 `text`。

`pairs_risk.jsonl` 以 `pair_id, harmful_id, benign_id, goal_cluster_id, match_status` 索引；`pairs_manipulation.jsonl` 以 `pair_id, jailbreak_id, harmful_id, goal_cluster_id, attack_family, goal_preserved, mapping_evidence` 索引。一个 H 可关联多个 J，禁止复制 H 行当作多个独立目标。

## 8. 数据验收清单

进入冻结数据集前逐项确认：

1. 每条文本有来源、固定版本/提交、原始 ID、许可或访问条款、哈希。
2. 每条 J 有可复核的 (H_i) 映射；不能映射的记录转入未配对外部面板。
3. H/B 的风险配对经审核；B 请求本身确实正常，而不是只因来源标签为 benign。
4. 核心集合只包含聊天 LLM 用户输入；T2I prompt 和多轮会话分层保存。
5. 有害目标、操控意图、目标保持、模型攻击成功是分开的字段。
6. 按目标组、规范化文本、近重复目标、攻击模板/谱系和来源检查跨分区污染。
7. 统计报告分别给出目标组数、H/B/J 文本数、有效 H-B 对、有效 J-H 对、各来源/攻击谱系数量、无配对数、uncertain 数、审核一致性和排除原因。
8. 许可证不支持再发布原文时，只发布可公开的 ID/哈希/派生统计和获取说明，不把受限文本重新打包分发。

## 9. 这一阶段的交付物

先完成数据，不做新模型训练或检测器性能比较。第一阶段结束时应有：

- 锁定的来源与许可清单；
- H-B 风险配对和 J-H 操控配对的映射规则；
- 标注手册与人工复核记录；
- 100 目标组试点统计；
- 冻结字段合同、去重/分区 manifest 和可复现数据转换脚本；
- 明确列出的来源覆盖空缺、不可配对率和数据标签限制。

若试点显示某个来源无法可靠连接到原始 (H_i)，就把它保留为独立外部集合或移出核心集，不为“覆盖 Sentinel 的所有 benchmark”而牺牲 DCCF 两类配对的语义有效性。

## 10. 与 DCCF 架构图的对应和首个试点

架构图的 A—D 不能被压成一次随机拆分：

- **A：冻结表征与概念构造。** 每条记录是完整用户输入；模型前向使用冻结主干和目标模型的原生聊天模板，在回答起始前读取末个有效 token。训练区的 `H-B` 产生风险差分，`J-H` 产生操纵差分。J-H 中的 H 仍是全局风险正类。
- **B：互补选层和融合。** 先分别在内部攻击家族留出上验证两个概念，再按方法文件的规则独立选层；融合器只从 train 拟合。分组必须以目标组为单位，同一目标的 H、B、所有 J 变体不得跨区。
- **C：冻结检测。** 新请求只计算已冻结方向、锚点、所选层融合分数，并与固定阈值比较。阈值只由独立 benign 校准材料确定；攻击成功字段不进入判定阈值或输入标签。
- **D：独立评估。** 新来源/新攻击家族和独立目标组进入冻结评估。结果报告分攻击家族的 TPR/FNR、相同安全控制上的 FPR 和样本量；校准后不得用这些结果调参。已暴露的 `diagnose` 仍只是开发诊断。

因此，Sentinel 可借鉴“有害—正常探针训练、对抗策略分组、训练材料与独立评估隔离”的实验组织方式；DCCF 需要额外维护共享 H 端点的两种配对，以及风险/操纵两个分支独立选层后再融合。Sentinel 的 T2I prompt、多轮会话或来源 benchmark 不能只凭数据集名称直接放入当前单轮聊天主集。

首个可复现候选试点以固定版本的 JBB-Behaviors 和官方 PAIR/GCG 发布 artifact 构建。生成器为 [`tools/build_jbb_pilot.py`](../tools/build_jbb_pilot.py)，重建命令为：

```bash
python tools/build_jbb_pilot.py --output data/dccf_v1_jbb_pilot
```

来源提交、原文件哈希和 MIT 许可文本记录在 `source_lock.json` 与 `licenses/`。试点当前核对结果：100 个目标组；100 条 H、100 条 B；172 条 J 记录的来源索引、目标文本、行为标识和类别均与对应 JBB H 一致；另有 10 条带文本的攻击候选因来源字段不完全匹配而单列待映射审核，18 条 artifact 记录没有可用 prompt。对 H-B 每个 H 仅给出同大类、字符长度最近的 3 个候选，共 300 个审核候选；这只是排序工作表，不是已经成立的语义配对。172 个 J-H 候选也仍待人工确认目标保持。

生成器仅保留已发布的 prompt 文本和必要来源元数据，不保留来源模型回答或 judge 输出。记录级 `risk_label` 是源标签候选；J 的来源攻击成功状态仅作元数据，不改变其风险正类身份。生成数据的 `split` 全部为空，清单声明 `pilot_candidate_material_only_not_training_ready`；此外，尚未完成双人标注、近重复/历史暴露重叠检查和目标主干模板 token 长度核验，因此当前不能把它当成 DCCF 的训练集、阈值校准集或独立确认集。

### 10.1 和现有方法分区的兼容

本方案第 6 节的 `70/15/15 train/select/confirmation` 仅是待评估的草案，不能直接覆盖现有方法协议。正式数据冻结时需先按 goal group 分配 train/select 和真正未暴露的确认区；另从不重叠的 benign 目标组建立仅用于阈值的 `calibrate`。已有 `diagnose` 继续标为已暴露开发材料，不能重命名为确认区。若新增来源被纳入本次核心集，它对应的原始 benchmark 测试材料也不再是独立外部测试。

未配对攻击可以留在外部候选面板，但不得为了填补目标而自动造 H、B 或改写攻击文本；任何被接受的 H-B / J-H 配对都需记录审核员、决定、理由及裁决记录。三组数据本身也不能证明操纵方向已排除包装风格混杂；需另行设计独立的合法策略格式控制，并在报告中单列其协议和结果。

## 参考来源

- Sentinel: [Quantifying Large Language Model Attacks Through the Lens of Model Cognition](https://www.usenix.org/system/files/conference/usenixsecurity26/sec26_prepub_liu-xiuming.pdf)
- Sentinel 数据表来源说明见论文参考文献与附录；SneakyPrompt: [arXiv:2305.12082](https://arxiv.org/abs/2305.12082)
- I2P: [CVPR 2023 paper and datasheet](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Schramowski_Safe_Latent_Diffusion_CVPR_2023_supplemental.pdf)
- MMA-Diffusion: [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_MMA-Diffusion_MultiModal_Attack_on_Diffusion_Models_CVPR_2024_paper.html)
- HarmBench: [official repository](https://github.com/centerforaisafety/HarmBench)
- JailbreakBench: [official repository and dataset description](https://github.com/JailbreakBench/jailbreakbench)
- Multi-turn jailbreak: [paper](https://arxiv.org/abs/2409.00137) and [dataset card](https://huggingface.co/datasets/tom-gibbs/multi-turn_jailbreak_attack_datasets)
- ChatAlpaca: [official repository](https://github.com/icip-cas/ChatAlpaca)
