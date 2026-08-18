---
title: "zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models"
source: https://aclanthology.org/2024.naacl-long.104.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:44:32"
field: "时序知识图谱推理"
keywords: ["Temporal Knowledge Graph", "Zero-shot Learning", "Large Language Model", "Knowledge Graph Reasoning", "Link Prediction", "LLM Alignment"]
innovations: ["首次利用LLM语义知识增强TKGF模型的零样本关系推理能力", "设计关系历史学习者(RHL)结合LLM增强表示捕捉时序关系模式", "构建2021-2023年无信息泄露的零样本TKGF新基准"]
benchmarks: ["ACLED-zero", "ICEWS21-zero", "ICEWS22-zero"]
---

# 论文速读：zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models

## 一句话总结
论文提出 zrLLM 方法，将大语言模型（LLM）的语义知识引入时序知识图谱链接预测（TKGF），解决嵌入基方法难以处理零样本（zero-shot）未见过关系的问题；通过在三个新构建的无信息泄露基准上实验，zrLLM 显著提升多种 TKGF 模型在零样本关系上的预测性能，同时保持对已见关系的预测能力。

## 研究问题与动机
1. **零样本关系预测的空白**：现有 TKGF 方法多为基于嵌入的方法，仅在训练时见过的关系集合 $\mathcal{R}_{se}$ 上进行学习，无法处理零样本关系 $r \notin \mathcal{R}_{se}$，因为训练数据中完全没有其图上下文。
2. **时序知识动态性带来的挑战**：随着时间推移，新事实不断进入 TKG，新关系也会涌现，要求 TKGF 模型具备更强的泛化能力。
3. **既有 LLM+TKG 工作未覆盖零样本关系**：已有 LLM 增强 TKG 推理的工作（如 SST-BERT、ECOLA、PPT、GenTKG、ICL-based 方法）均未研究 LLM 是否有助于零样本关系推理。
4. **信息泄露风险**：传统 TKGF 基准（如 ICEWS14/18/05-15）的事实多发生在 2020 年之前，LLM 可能在预训练语料中已见过这些知识，导致评估失真；因此需要构建后 2020 年发生的新基准。

## 核心贡献（创新点）
1. **首次研究 TKGF 中的零样本关系学习**：本文首次系统性地探讨如何将 LLM 用于时序知识图谱的零样本关系推理，填补了该领域的研究空白。
2. **LLM-empowered 关系表示方法**：利用 GPT-3.5 对关系文本进行扩充（ERD），再通过 T5-11B encoder 生成语义丰富的关系表示，并固定该表示以保留语义信息，避免被训练数据过度拟合所稀释。
3. **关系历史学习者（RHL）模块**：设计 RHL 模块，利用 LLM 增强的关系表示捕捉时序关系模式，并通过历史预测网络（HPN）在推理时绕过对所有候选实体对搜索历史的开销，实现语义与图空间的对齐。
4. **构建三个零样本 TKGF 新基准**：基于 ICEWS 和 ACLED 数据集构建 ACLED-zero、ICEWS21-zero、ICEWS22-zero，所有事实发生在 2021–2023 年，避免 T5-11B 等信息泄露。

## 方法详解
**整体框架**：zrLLM 耦合于嵌入型 TKGF 模型，包含两个核心组件：LLM 关系表示生成 + 关系历史学习者（RHL）。

**1. LLM 生成扩充关系描述（ERD）**
- 输入 TKG 数据集提供的简短关系文本，通过 GPT-3.5 生成包含更丰富语义的解释，组合为 ERD（Enriched Relation Description）。
- 示例："Engage in negotiation" → "Engage in negotiation: This indicates a willingness to participate in discussions or dialogues with the aim of reaching agreements or settlements on various issues."

**2. T5-11B 编码生成关系表示**
- 将 ERD 输入 T5-11B encoder，获取隐藏表示 $\bar{\mathbf{H}}_r \in \mathbb{R}^{L \times d_w}$。
- 通过 MLP 将每词向量映射到 TKGF 模型的维度 $d$，再用 GRU 聚合得到最终关系表示 $\bar{\mathbf{h}}_r$：
  $$\mathbf{w}'_l = \mathrm{MLP}(\mathbf{w}_l), \quad \bar{\mathbf{h}}_r^{(l)} = \mathrm{GRU}(\mathbf{w}'_l, \bar{\mathbf{h}}_r^{(l-1)})$$
- **关键设计**：$\bar{\mathbf{H}}_r$ 在训练中保持冻结（fixed），以保证 LLM 语义不被训练数据稀释。

**3. 关系历史学习者（RHL）**
- **关系历史编码**：对训练事实 $(s,r,o,t)$，检索 $s$ 与 $o$ 在 $t$ 之前的所有历史事实，按时间分组；每组聚合关系表示：
  $$\mathbf{h}^{t_i}_{s,o} = \sum_m a_m \bar{\mathbf{h}}_{r_m}, \quad a_m = \mathrm{softmax}(\bar{\mathbf{h}}_{r_m}^\top \mathrm{MLP}_{\mathrm{agg}}(\bar{\mathbf{h}}_r))$$
- 通过 $\mathrm{GRU}_{\mathrm{RHL}}$ 编码历史序列得到 $\mathbf{h}_{\mathrm{hist}}$。
- **历史预测网络（HPN）**：由于推理时未知目标实体对，训练期间用 HPN 直接从 $\bar{\mathbf{h}}_r$ 预测关系历史：
  $$\tilde{\mathbf{h}}_{\mathrm{hist}} = \alpha \cdot \mathrm{MLP}_{\mathrm{hist}}(\bar{\mathbf{h}}_r) + \bar{\mathbf{h}}_r$$
  并用 MSE 损失约束：$\mathcal{L}_{\mathrm{hist}} = \mathrm{MSE}(\tilde{\mathbf{h}}_{\mathrm{hist}}, \mathbf{h}_{\mathrm{hist}})$。
- **模式表示**：将 $\tilde{\mathbf{h}}_{\mathrm{hist}}$ 与 $\bar{\mathbf{h}}_r$ 输入 $\mathrm{GRU}_{\mathrm{RHL}}$ 得到 $\mathbf{h}_{\mathrm{pat}}$，结合时间感知实体表示通过 TuckER 式评分：
  $$\phi((s,r,o,t)) = \mathcal{W} \times_1 \mathbf{h}_{(s,t)} \times_2 \mathbf{h}_{\mathrm{pat}} \times_3 \mathbf{h}_{(o,t)}$$
- **总分融合**：$\phi_{\mathrm{total}} = \phi' + \gamma \cdot \phi$，$\phi'$ 为原 TKGF 模型的分数。

**4. 总损失函数**
$$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{TKGF}} + \mathcal{L}_{\mathrm{hist}} + \eta \cdot \mathcal{L}_{\mathrm{RHL}}$$
其中 $\mathcal{L}_{\mathrm{RHL}}$ 为基于 RHL 分数的 binary cross-entropy 子任务损失。

## 实验与结果
**数据集**（均公开于 GitHub）：
- **ACLED-zero**：621 实体，23 关系（9 seen / 14 zero-shot），2023-08 数据
- **ICEWS21-zero**：18,205 实体，253 关系（130 seen / 123 zero-shot），2021 上半年数据
- **ICEWS22-zero**：999 实体，248 关系（93 seen / 155 zero-shot），2022 上半年数据

**基线模型**（7 种嵌入型 TKGF 方法及其 zrLLM 增强版本）：
CyGNet、TANGO-T/D、RE-GCN、TiRGN、RETIA、CENET。

**主要结果**（MRR，取最强提升）：
- **ACLED-zero（zero-shot）**：CENET+ 达到 **0.591**（基线 CENET 为 0.419，提升 +41%）；RETIA+ 达到 0.557（基线 0.499）。
- **ICEWS21-zero（zero-shot）**：CENET+ 达到 **0.335**（基线 0.205，提升 +63%）；TiRGN+ 达到 0.221（基线 0.189）。
- **ICEWS22-zero（zero-shot）**：CENET+ 达到 **0.564**（基线 0.270，提升 +109%）；RETIA+ 达到 0.331（基线 0.302）。
- 在 seen relations 上，zrLLM 增强版本大多保持或小幅提升原有性能，overall MRR 均优于基线。

**对比 LM 增强基线**：
- PPT（fine-tune BERT）：ACLED-zero zero-shot MRR=0.532，低于 CENET+ 的 0.591。
- ICL + GPT-NeoX-20B：ICL 表现较差（ACLED-zero zero-shot MRR=0.537，但 overall 仅 0.709），证明无 fine-tune/对齐时 LLM 无法最优解决 TKGF。

**消融实验结论**：
- 去掉 ERD（直接用原始关系文本）：各模型 zero-shot 和 seen 性能均有下降，证明扩充描述有效。
- 去掉 RHL：CENET 在 ACLED-zero 上 zero-shot MRR 从 0.591 降至 0.445，下降显著。
- 缩小 T5 至 3B：各模型性能普遍下降，说明更大 LLM 提供更高质语义信息。

## 相关工作脉络
1. **传统嵌入型 TKGF**（CyGNet、TANGO、RE-GCN、TiRGN、CENET、RETIA）：仅训练于 seen relations，无法泛化至 zero-shot；本文在其基础上叠加 zrLLM 模块。
2. **规则型 TKGF**（TLogic、ALRE-IR）：对 zero-shot 实体推理强，但对 zero-shot 关系同样无效；zrLLM 不直接适用于此类方法。
3. **TKG 归纳学习 / few-shot 方法**（FILT、OAT、MOST、OSLT）：依赖 K-shot 辅助事实，无法解决纯 zero-shot 场景；本文直接面向零样本设定。
4. **LLM 增强 TKG 推理**（SST-BERT、ECOLA、PPT、GenTKG）：SST-BERT 未验证 zero-shot relation；PPT 需 fine-tune BERT；GenTKG 需 fine-tune Llama2-7B，计算开销大；本文以冻结 LLM 表征+轻量对齐的方式，在效率和性能间取得平衡。
5. **ICL + LLM 用于 TKGF**（Lee et al., 2023）：不 fine-tune 时性能被传统方法超越；本文证明通过空间对齐可充分发挥 LLM 语义能力。
6. **MTKGE**（Chen et al., 2023b）：可处理 unseen entities 和 relations，但需要 support graph 大量辅助数据，非真正的 zero-shot 设定。

## 局限性与未来方向
1. **仅适用于嵌入型 TKGF 方法**：zrLLM 未扩展到 rule-based 方法（如 TLogic），通用性有限。
2. **计算效率开销**：RHL 引入 GRU 时序计算，增加训练/推理时间和 GPU 显存占用（需存储关系历史）。
3. **仅针对零样本关系**：未同时解决 zero-shot entity 的问题。
4. **未来方向**（作者自述）：（1）推广至 rule-based 方法；（2）提升效率（减少 GRU 计算开销）；（3）在更多 TKGF 方法上验证。

## 研究启发与可借鉴点
1. **"冻结 LLM 表征 + 轻量对齐"范式**：在不需要 fine-tune 大型 LLM 的前提下，通过 MLP+GRU 将语言空间映射到图嵌入空间，兼顾语义丰富度与训练效率，可作为跨领域知识增强方法的通用模板。
2. **历史预测网络（HPN）设计**：利用 HPN 在推理时直接预测实体对关系历史，避免对所有候选实体对进行全量历史搜索，有效解决了 scalability 问题，该思路可迁移至其他需要历史模式建模的任务。
3. **构建无信息泄露基准的方法论**：通过选择 LLM 发布之后的时间点构造测试数据（2021–2023），严格隔离预训练语料的影响，为 LLM+KG 研究提供了可复用的评估规范。
4. **语义相近关系在嵌入空间自然聚集**：固定 LLM 生成的关系表示使得语义相近的 zero-shot 和 seen 关系在向量空间中距离较近，这为"利用语言先验缓解数据稀疏"提供了直观证据。
5. **子任务辅助训练策略**：$\mathcal{L}_{\mathrm{hist}}$ 和 $\mathcal{L}_{\mathrm{RHL}}$ 作为辅助任务与主 TKGF 任务联合优化，促进语言-图空间对齐，这种多任务蒸馏思路可用于其他 LLM-KG 融合场景。

## 关键术语表
**Temporal Knowledge Graph (TKG)**：带时间戳的知识图谱，每个事实表示为 $(s, r, o, t)$ 四元组，刻画动态关系演化。
**TKG Forecasting (TKGF)**：基于历史事实预测未来某个时间点可能出现的缺失实体（链接预测）。
**Zero-shot Relation**：训练集中未出现但在测试集中作为查询关系出现的关系，模型没有任何与其相关的图上下文。
**Enriched Relation Description (ERD)**：利用 GPT-3.5 对数据集提供的简短关系文本进行语义扩充后生成的详细描述。
**Relation History Learner (RHL)**：zrLLM 的核心模块，利用 LLM 增强的关系表示通过 GRU 捕捉实体间时序关系模式。
**History Prediction Network (HPN)**：RHL 中的子网络，从关系表示直接预测两实体间历史关系序列，用于推理时替代全量历史搜索。
**T5-11B**：Google 开源的 110 亿参数 text-to-text transformer，本文以其 encoder 部分生成关系文本表示。
**Information Leak**：LLM 预训练语料中已包含测试集事实知识，导致基于 LLM 的方法在评估中获得非真实泛化能力的虚高分数。

## 可复现要素
- **数据集**：三个新构建的 zero-shot TKGF 数据集（ACLED-zero、ICEWS21-zero、ICEWS22-zero）已公开，代码仓库：https://github.com/ZifengDing/zrLLM
- **代码**：zrLLM 增强代码及基线实现已开源（论文声明）
- **LLM**：GPT-3.5（API调用，生成ERD）；T5-11B（encoder冻结，生成关系表示）
- **关键超参**：$\alpha \in \{1, 0.1\}$，$\gamma \in \{1, 0.01, 0.001\}$（可为固定值或可学习参数），$\eta \in \{1.2, 1\}$；基线超参（embedding size、history length）随各基线模型分别搜索
- **硬件**：单卡 NVIDIA A40 48GB，AMD EPYC 7513 32-Core
- **复现提示**：T5 表征需预先离线生成并保存，避免训练时重复调用；推理时需限制模型仅使用训练集历史事实
