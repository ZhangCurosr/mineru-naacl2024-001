---
title: "zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models"
source: https://aclanthology.org/2024.naacl-long.104.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:42:48"
field: "时序知识图谱推理"
keywords: ["Temporal Knowledge Graph Forecasting", "Zero-Shot Relational Learning", "Large Language Models", "Knowledge Graph Embedding", "Inductive Reasoning"]
innovations: ["首次系统性研究 TKGF 中的零样本关系学习问题，定义形式化任务并构建专用基准", "提出 zrLLM 框架：通过冻结 LLM（T5-11B）提取关系语义表示并投影到 TKGF 嵌入空间，无需微调 LLM 即可提升零样本泛化能力", "设计关系历史学习器 RHL 与历史预测网络 HPN：利用 LLM 语义表示捕获时序关系模式，在推理时避免候选实体搜索即可推断关系历史"]
benchmarks: ["ACLED-zero", "ICEWS21-zero", "ICEWS22-zero"]
---

# 论文速读：zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models

## 一句话总结
论文提出 zrLLM，通过将 LLM 生成的关系语义表示引入嵌入型时序知识图谱预测（TKGF）模型，首次系统研究了 TKGF 中的零样本关系学习问题；在三个全新构建的零样本基准上，zrLLM 使多种基线模型在未见关系上的预测能力大幅提升，同时保持了对已见关系的原有性能。

## 研究问题与动机
1. **现有嵌入型 TKGF 方法无法处理零样本关系**：传统方法（如 CyGNet、TANGO、RE-GCN 等）仅从训练集中观察到的关系图谱上下文中学习关系表示，对于训练时从未出现过的关系（zero-shot unseen relations），由于缺乏对应的图上下文，无法生成合理的关系表示，导致预测性能极差。
2. **已有归纳学习工作不适用于零样本场景**：少数样本学习（few-shot learning）方法（如 OAT、MOST、FILT 等）依赖推理时少量（K=1 或 3）已观察到的样例来学习归纳表示，本质上无法处理零样本情况；SST-BERT 虽具备归纳能力但仅针对未见实体，未涉及零样本关系；MTKGE 虽能同时处理未见实体和关系，但需要大量支持图数据，不符合零样本设定。
3. **已有 LLM 增强 TKGF 工作存在三个缺陷**：（i）未研究 LLM 在零样本关系推理上的能力；（ii）仅靠 In-Context Learning（ICL）的 LLM 性能低于传统 TKGF 方法，而微调 LLM（如 GenTKG）计算开销巨大；（iii）主流基准（ICEWS14/18 等）的事实发生在 2020 年之前，而 T5-11B（2020 年发布）可能在预训练语料中已接触过这些事实，存在信息泄露威胁。
4. **现实驱动**：随着时间推移，TKG 不断增长，新关系持续涌现，提升嵌入型 TKGF 模型对零样本关系的自适应能力具有重要的实际意义。

## 核心贡献（创新点）
1. **首次系统研究 TKGF 中的零样本关系学习问题**：定义了零样本 TKGF 形式化任务，并指出这是现有嵌入型方法的核心盲区，而此前工作（包括 few-shot 方法和 LLM 方法）均未触及此问题。
2. **提出 zrLLM 框架，将 LLM 语义信息注入 TKGF 模型**：通过 GPT-3.5 生成丰富关系描述（ERD），再利用 T5-11B encoder 提取文本表示，经 MLP+GRU 对齐到 TKGF 嵌入空间，替代或补充原始关系表示——这一"冻结 LLM 表示+空间对齐"的方式避免了微调成本，且保证了零样本关系仍能获得富含语义的表示。
3. **设计关系历史学习器（RHL）捕捉时序关系模式**：利用 LLM 赋能的关系表示通过 GRU 编码实体对的历史关系序列，同时引入历史预测网络（HPN）在推理时无需搜索候选实体对即可推断关系历史，并通过 TuckER 式张量得分将模式信息与 TKGF 原始得分融合。
4. **构建三个面向零样本 TKGF 的新基准数据集（ACLED-zero、ICEWS21-zero、ICEWS22-zero）**：所有事实发生于 2021–2023 年，避开 T5-11B 训练语料，有效避免信息泄露；零样本关系占比高（ACLED-zero 中 14/23，ICEWS21-zero 中 123/253，ICEWS22-zero 中 155/248）。
5. **实验验证广泛有效性**：在七种嵌入型 TKGF 模型上验证，zrLLM 在所有模型上均显著提升了零样本关系预测性能，且在多数情况下同时提升或保持了对已见关系的性能。

## 方法详解
**整体流程（图 1）**：zrLLM 分为两阶段——训练阶段联合优化 TKGF 模型与 RHL 模块；评估阶段直接使用训练好的 HPN 推断关系历史，无需候选实体搜索。

**步骤一：LLM 赋能的关系表示生成（Sec. 3.1）**
- **丰富关系描述（ERD）生成**：以数据集提供的关系短文本为输入，构造 prompt 调用 GPT-3.5 生成扩充解释，拼接为 ERD（例："Engage in negotiation" → 加入 "This indicates a willingness to participate in discussions..."）。
- **文本编码与空间对齐**：将 ERD 输入 T5-11B encoder，固定其参数（不微调，防止信息泄露）。对每个词的隐藏表示 w_l ∈ R^{d_w}，先用 MLP 映射到 TKGF 关系维度 d，再通过 GRU 依次聚合得到关系语义表示 h̄_r ∈ R^d：
  - w'_l = MLP(w_l)
  - h̄_r = GRU(w'_l, h̄_r^{(l-1)})
- 关键设计：T5 参数冻结，确保零样本关系的表示完全来自语义而非图上下文；语义相近的关系在嵌入空间中自然靠近。

**步骤二：关系历史学习器 RHL（Sec. 3.2）**
- **历史关系编码**：对于训练事实 (s, r, o, t)，搜索 s 和 o 在 t 之前的所有历史事实，按时间分组，每组用加权聚合得到该时刻的关系表征：
  - h_{s,o}^{t_i} = Σ_m a_m h̄_{r_m}，其中 a_m = softmax(h̄_{r_m}^T · MLP_agg(h̄_r))
  - 若无历史事实，使用 dummy embedding
- 通过 GRU_RHL 编码整个历史序列得到 h_hist ∈ R^d。
- **历史预测网络（HPN）**：推理时不知道待预测实体的候选对，因此训练时引入 HPN 直接从关系表示预测历史：
  - h̃_hist = α · MLP_hist(h̄_r) + h̄_r
  - 损失 L_hist = MSE(h̃_hist, h_hist)
- **关系模式表示**：将预测历史与当前关系通过 GRU_RHL 融合，得到模式表示 h_pat。
- **TuckER 式评分**：ϕ((s,r,o,t)) = W ×₁ h_{(s,t)} ×₂ h_pat ×₃ h_{(o,t)}，衡量实体对与关系模式的匹配程度。
- **总评分**：ϕ_total = ϕ'（原始 TKGF 模型得分）+ γ · ϕ（RHL 得分），γ 为超参。

**步骤三：联合训练与评估（Sec. 3.3）**
- 总损失：L_total = L_TKGF + L_hist + η · L_RHL
  - L_TKGF：基于 ϕ_total 的标准 TKGF 损失（如 cross-entropy）
  - L_RHL：对 RHL 得分施加的二元交叉熵损失（实体负采样）
- 评估时对每个 LP 查询 (s_q, r_q, ?, t_q)，用 HPN 直接推断关系历史，计算所有候选实体的 ϕ_total，取最高分为预测答案。
- 零样本评估约束：模型仅在训练集上进行链接预测，不利用任何验证/测试集事实（尤其对 TANGO、RE-GCN、TiRGN、RETIA 等使用 recurrent 结构的方法，限制其仅使用训练集最新 k 个时间步的历史）。

## 实验与结果
**数据集**（表 2）：
- ACLED-zero：621 实体，23 关系（9 已见/14 零样本），训练集 2,118 条
- ICEWS21-zero：18,205 实体，253 关系（130 已见/123 零样本），训练集 247,764 条
- ICEWS22-zero：999 实体，248 关系（93 已见/155 零样本），训练集 171,013 条

**基线模型**：CyGNet、TANGO-T/D、RE-GCN、TiRGN、RETIA、CENET（七种近期嵌入型 TKGF 方法），以及 PPT 和 ICL+GPT-NeoX-20B。

**主要结果**（表 3/表 15，MRR）：
- **ACLED-zero 零样本关系**：CENET+ 达到最高 MRR = 0.591（对比 CENET 基线 0.419，+41%）；CyGNet+ 0.533（vs 0.487）；TiRGN+ 0.548（vs 0.478）
- **ICEWS21-zero 零样本关系**：CENET+ MRR = 0.335（vs 0.205，+63%）；TiRGN+ 0.221（vs 0.189）
- **ICEWS22-zero 零样本关系**：CENET+ MRR = 0.564（vs 0.270，+109%）；RETIA+ 0.331（vs 0.302）
- **已见关系性能保持**：大多数模型在已见关系上性能持平或略有提升；即使个别模型已见关系略有下降，整体性能仍改善
- **vs PPT/ICL**（表 5）：CENET+ 在 ACLED-zero 零样本上 MRR=0.591 超越 PPT（0.532）和 ICL（0.537）；在 ICEWS22-zero 上 CENET+（0.564）远超 PPT（0.323）和 ICL（0.255）

**消融实验**（表 4）：
- 去除 ERD（直接用数据集原始关系文本）：所有模型性能下降，证明 ERD 的价值
- 去除 RHL：性能下降，CENET 受影响最大（ACLED-zero 零样本 MRR 从 0.591 降至 0.445）
- 缩小 T5 至 3B：性能下降，证明更大 LLM 提供更高质量语义信息

**最强结果**：CENET+ 在 ICEWS22-zero 零样本上 MRR = 0.564 / Hits@1 = 0.432，相对基线 CENET 提升幅度最大（+109% MRR）。

## 相关工作脉络
1. **传统嵌入型 TKGF（CyGNet、TANGO、RE-GCN、TiRGN、RETIA、CENET）**：本文基线方法，均从图上下文学关系表示，仅能处理训练集已见关系；zrLLM 与其本质区别在于用 LLM 语义填充零样本关系的表示空白。
2. **规则型 TKGF（TLogic、ALRE-IR）**：擅长推理已见关系下的零样本实体，但规则受限于已见关系，无法处理零样本关系；zrLLM 仅针对嵌入型方法设计，不涉及规则。
3. **Few-shot 归纳学习（OAT、MOST、FILT、MetaTKGR、OSLT）**：依赖推理时 K 个已观察样例学习归纳表示，无法处理零样本；zrLLM 完全不依赖推理时样例，仅靠 LLM 语义实现零样本泛化。
4. **SST-BERT（Chen et al., 2023a）**：预训练时间增强 BERT 处理未见实体，但未涉及零样本关系推理；zrLLM 明确聚焦关系维度的零样本泛化。
5. **MTKGE（Chen et al., 2023b）**：同时处理未见实体和关系，但需要大量支持图数据；zrLLM 在严格零样本设定下工作，无需任何支持样例。
6. **LLM 增强 TKGF（PPT、ECOLA、GenTKG、ICL+LLM）**：PPT 微调 BERT、GenTKG 微调 Llama2-7B、ICL 直接推理——三者分别存在微调成本高或性能不足的问题；zrLLM 折中方案：不微调 LLM，而是将 LLM 语义对齐到 TKGF 嵌入空间，兼具高效与高性能。

## 局限性与未来方向
1. **仅适用于嵌入型 TKGF 方法**：zrLLM 不直接适用于规则型方法（如 TLogic），通用性受限。
2. **计算效率降低**：RHL 使用 GRU 沿时间轴做循环计算，增加了训练和推理时间，同时需要额外 GPU 内存存储关系历史（表 13 显示不同模型额外内存占用从 ~1GB 到 ~37GB 不等）。
3. **未来方向**：（i）将 zrLLM 推广到规则型 TKGF 方法；（ii）优化模型效率（如替代 GRU 的更轻量大模块）；（iii）在更多 TKGF 方法上验证有效性。

## 研究启发与可借鉴点
1. **"冻结 LLM 表示 + 空间对齐"范式**：不微调 LLM 而是固定其输出、仅通过轻量 MLP+GRU 投影到下游模型嵌入空间，既避免了信息泄露和计算开销，又保留了 LLM 的语义泛化能力——这一策略可迁移到其他需要 LLM 语义增强的结构化预测任务（如静态 KG 补全、多关系预测等）。
2. **历史预测网络（HPN）的"替代搜索"设计**：推理时直接用关系表示预测实体对历史，避免了遍历所有候选实体对的 O(|E|) 复杂度，这是一个兼具效率与效果的设计思路，可迁移到其他需要历史信息的时序推理场景。
3. **构建无信息泄露的新基准**：使用晚于 LLM 训练截止日期的新数据构建评估基准，是评估 LLM 在 NLP/KG 任务上真实泛化能力的良好实践，值得在其他 LLM+结构化推理工作中借鉴。
4. **关系描述丰富化（ERD）提升语义质量**：用 LLM 对短关系文本进行语义扩充，比直接使用原始短文本效果更好（消融证实）；这一 prompt-based 描述扩充策略可复用于其他需要关系语义的任务。
5. **多任务辅助训练（L_hist 作为子任务）**：将历史预测作为辅助训练目标（MSE 损失），促进了文本空间与图嵌入空间的对齐——这种"主任务+语义对齐辅助任务"的训练策略可推广到其他 LLM-嵌入融合场景。

## 关键术语表
**Zero-Shot TKGF**：时序知识图谱预测中的零样本任务，指模型在训练集中从未见过某关系的情况下，仍需在测试集中对该关系的事实进行链接预测。
**Enriched Relation Description (ERD)**：通过 LLM（GPT-3.5）对数据集提供的简短关系文本进行语义扩充后生成的详细描述，用于为关系提供更丰富的语义信息。
**Relation History Learner (RHL)**：zrLLM 的核心模块，利用 LLM 赋能的关系表示通过 GRU 编码实体对的历史关系序列，捕获实体间关系的时序演化模式。
**History Prediction Network (HPN)**：RHL 中的辅助网络，训练时学习从关系表示预测其实体对的历史关系序列，推理时直接生成有意义的前缀历史，避免候选实体搜索。
**Information Leak**：指 LLM 在预训练阶段可能已接触测试数据中的事实（尤其是旧基准如 ICEWS14），导致 LLM 增强的模型获得不正当优势。
**TuckER**：一种基于张量分解的知识图谱补全方法，本文用于计算 RHL 打分（三线性张量得分函数）。
**Inductive Learning on TKGs**：研究能在训练集中未见过的新关系或新实体上进行推理的 TKG 学习方法，本文聚焦其中的 zero-shot 子设定。
**Time-aware Filtering**：链接预测评估中的公平过滤设置，剔除因时间戳信息导致的非法候选项（Han et al., 2021a），本文采用此设置进行评测。

## 可复现要素
- **数据集**：三个新构建的零样本 TKGF 数据集（ACLED-zero、ICEWS21-zero、ICEWS22-zero）已在代码页公开
- **代码**：已开源，GitHub 地址：https://github.com/ZifengDing/zrLLM
- **关键超参**：α（HPN 残差系数）、γ（RHL 得分权重，可为固定值或可学习参数）、η（L_RHL 权重）、历史长度 k、嵌入维度 d；具体最优设置见论文表 12
- **LLM**：GPT-3.5（用于 ERD 生成）、T5-11B（用于文本编码，参数冻结）
- **硬件**：NVIDIA A40 48GB GPU，PyTorch 实现
- **评估指标**：MRR、Hits@1/3/10，time-aware filtering 设置
