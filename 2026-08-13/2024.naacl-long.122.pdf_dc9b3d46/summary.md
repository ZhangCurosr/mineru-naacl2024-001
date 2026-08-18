---
title: "Strings from the Library of Babel: Random Sampling as a Strong Baseline for Prompt Optimisation"
source: https://aclanthology.org/2024.naacl-long.122.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:46:40"
---

# 论文速读：Strings from the Library of Babel: Random Sampling as a Strong Baseline for Prompt Optimisation

## 一句话总结
本文证明了从语言模型词表中随机采样 token 组成的"分隔符"（separator），在提示词风格的文本分类任务中可以达到与自优化方法相当的性能，平均相对提升超过人类基线 12%，且挑战了"有效 prompt 必须具有人类可读性或任务相关性"的默认假设。

## 研究问题与动机
- **现有方法过度依赖 LLM 生成 prompt**：主流 prompt 优化方法（如 OPRO、APE）均需调用指令微调的大型语言模型来搜索/生成替代性 prompt，成本高且假设了 instruction-tuned 模型的必要性。
- **"有效 prompt 必须语义相关"的隐含假设未经检验**：前人研究默认好 prompt 应与任务或上下文高度相关，但这一假设在分类任务上缺乏系统性验证。
- **随机分隔符的潜力被低估**：AutoPrompt（Shin et al., 2020）已提示随机字符串可能有效，但其发现在文本分类通用场景下未被充分探索。
- **缺乏强基线导致研究贡献被高估**：若随机采样即可匹配 SOTA，则此前自优化方法的实际贡献可能被夸大。

## 核心贡献（创新点）
1. **提出三种随机分隔符生成策略并系统比较**：Random Vocabulary（词表随机）、Random w/o Context（LLM 先验采样）、Random with Context（条件于训练样本采样），无需指令模型即可生成高质量 prompt。
2. **建立随机采样作为 prompt 优化的强基线**：随机分隔符与 OPRO 等自优化方法差距不足 1%，却完全不需要外部 LLM，颠覆了该领域的默认实验基准。
3. **揭示语言空间中有效分隔符的高密度**：超过 40% 的随机抽取分隔符性能优于人类手写分隔符（"Answer:"），证明"好 prompt 稀缺"并非事实。
4. **系统性证伪"任务相关性是 prompt 有效的必要条件"**：Random Vocabulary（无语义、无任务关联）仅比 Random w/o Context（自然语言但无关任务）差 0.5%，说明 coherence 并非关键因素。
5. **将发现推广至生成式推理任务**：在 GSM8K 数学推理任务上，最佳随机分隔符以 47.3% 准确率超越 CoT（38.3%，+23% 相对提升），且随机方法的跨 seed 方差（9%）小于 CoT（17%）。

## 方法详解
**框架三阶段**：

1. **分隔符生成**：
   - **Random Vocabulary**：从模型词表 V 中均匀随机采样 token，拼接至预设长度上限，完全不依赖 LLM 和上下文。
   - **Random w/o Context**：从 LLM 的无条件先验分布 P(word) 中采样自然语言短语，得到语法合法但与任务无关的分隔符。
   - **Random with Context**：将 3 个随机训练样本（x, y）作为 in-context 示例提供给 LLM，采样条件分布 P(separator | context)，引入任务相关性。

2. **分隔符评估**：给定训练语料 $T = \{(x_i, y_i)\}_{i=1}^n$，对每个候选分隔符 s，计算分类准确率：
   $$\hat{y}_i = \arg\max_{v \in V} P(v \mid c \oplus x_i \oplus s; \theta)$$
   其中 $\oplus$ 为字符串拼接，c 为 in-context 示例，$\theta$ 为 PLM 参数。以准确率 m 作为评分函数（非可微，可直接优化离散度量）。

3. **分隔符选择**：在采样预算 k 内累积集合 $S = \{(s_i, m_i)\}_{i=1}^k$，选取最高分 $s^* = \arg\max_{s_i \in S} m_i$ 用于测试集评估。

**关键超参**：训练集 64 个样本，采样 160 个候选分隔符；OPRO 最多 40 步优化、每步生成 4 个候选；温度 1.0（生成）、0.0（分类）。

## 实验与结果
- **数据集**：9 个文本分类数据集（SST-2/5, MR, CR, MPQA, Subj, TREC, AGNews, DBPedia）+ GSM8K 数学推理（扩展实验）。
- **模型**：8 个语言模型，含 4 个预训练模型（GPT-2 Large/XL, Mistral 7B, Llama2 7B）和 4 个指令微调模型（Mistral 7B Instruct, Llama-Alpaca 7B, Llama2 7B Chat, ChatGPT GPT-3.5 Turbo）。
- **主要结果**（Table 6）：
  - **Random Vocabulary** 平均相对提升 **+23.4%**（以 "Answer:" 为基线），仅比 OPRO（+22.8%）高 0.6 个百分点。
  - **Random w/o Context** 平均相对提升 **+24.4%**，略超 OPRO。
  - **Random with Context** 平均相对提升 **+25.9%**，三种随机策略中最优。
  - 与 EvoPrompt（强人工优化基线）差距仅 **3.4%**，最大下降 2.1%（Subj 数据集）。
- **人类基线表现**："Foo Bar" 随机字符串和 "Let's think step by step"（ZS-CoT）平均低于人类基线 7-18%。
- **GSM8K 推理**（Table 13）：最佳随机分隔符平均 **47.3%** vs CoT **38.3%**（+23% 相对提升）；CoT 跨 seed 方差 17%，随机方法仅 9%（更鲁棒）。
- **40% 胜率**（Table 9）：在 AGNews 上，随机分隔符平均有 **>40%** 概率优于人类基线 "Answer:"。

## 相关工作脉络
1. **AutoPrompt (Shin et al., 2020)**：首个发现随机 token 序列可在嵌入层梯度引导下发现有效 prompt 的工作；本文沿此方向但完全不需要梯度信息，仅依赖纯随机采样。
2. **OPRO (Yang et al., 2023)**：用 LLM 作为优化器，通过 meta-prompt 自生成/进化 prompt；本文证明其性能增益主要来自"碰巧抽到好分隔符"而非优化过程本身。
3. **APE (Zhou et al., 2022)**：基于 LLM 生成候选 prompt 再筛选；本文指出 APE 类方法应首先与 Random Vocabulary 对比，否则贡献被高估。
4. **EvoPrompt (Guo et al., 2023)**：将进化算法融入 meta-prompt 搜索；Random Vocabulary 与其差距仅 3.4%，质疑其复杂度必要性。
5. **PromptBreeder (Fernando et al., 2023)**：自引用自我改进的 prompt 进化；本文揭示"好 prompt 密度远高于预期"，削弱自进化的边际价值。
6. **Kubrick's The Shining (Shi et al., 2022)**：主张 prompt 应具备人类可读性；本文用 0.5% 的微小差异证伪此假设。
7. **Continuous Prompt Tuning (Lester et al., 2021; Qin & Eisner, 2021)**：在连续空间微调 soft prompt；本文关注离散 token 空间的随机搜索，与软提示形成方法论对照。

## 局限性与未来方向
- **主要评估局限于文本分类任务**：虽在 GSM8K 上有初步验证，但更广泛的生成式任务（对话、摘要、代码生成）需系统评估。
- **未探索更复杂的随机搜索策略**：当前仅做独立同分布随机采样，未结合贝叶斯优化、进化算法等增强策略。
- **随机分隔符的跨任务可迁移性低**：Table 10 显示 SST-2 的最优分隔符在 SST-5 上接近随机猜测（~20%），限制了实际应用价值。
- **未分析随机分隔符的内在机制**：为何随机 token 组合能提升性能？是 LLM 对 suffix 的统计敏感性还是某种隐式格式对齐？仍需理论解释。
- **未来方向**：扩展至多模态/代码生成任务；研究随机与语义 prompt 的互补性；探索可迁移的随机种子策略。

## 研究启发与可借鉴点
1. **重新校准 SOTA 声明的基线标准**：任何新的 prompt 优化方法应首先与 Random Vocabulary 对比，否则其声称的"改进"可能仅为随机噪声。
2. **低成本替代方案**：在资源受限场景下，直接采用 Random Vocabulary 作为零成本 baseline，无需调用外部 LLM 即可达成接近 SOTA 的性能。
3. **LLM 对 suffix 的 oversensitivity 是一种可利用特性**：随机 token 序列能显著提升性能，提示 LLM 的上下文学习对输入末尾格式高度敏感，可设计更鲁棒的 prompt 评估协议。
4. **交叉任务的 prompt 泛化研究缺口**：随机分隔符跨任务迁移性差（59% vs 59.4% 与人类基线相当），这为研究"通用 prompt 表示"提供了明确方向。
5. **可结合本团队方向的机会**：在低资源 NLP/少样本学习中，随机分隔符可作为零成本初始化策略；结合梯度引导（AutoPrompt 思路）与随机采样可设计混合搜索算法。

## 关键术语表
- **Separator（分隔符）**：置于输入序列末尾或输出起始的 token 串（类比 BERT 的 [SEP]），用于引导模型执行特定任务格式，如 "Answer:" 或随机字符串。
- **Random Vocabulary**：从模型词表 V 中均匀随机采样 token 生成分隔符的策略，完全不依赖 LLM 和任务上下文。
- **Random w/o Context**：从 LLM 无条件先验分布中采样自然语言短语作为分隔符，语法合法但与任务无关。
- **Random with Context**：将训练样本作为 in-context 示例，从 LLM 条件分布中采样任务相关分隔符。
- **OPRO（Optimizer by Prompting Random Outputs）**：Yang et al. (2023) 提出的用 LLM 自身作为优化器的 prompt 自动搜索方法。
- **In-context Learning（上下文学习）**：在预训练模型中通过提供少量示例（demonstrations）引导模型完成新任务，无需参数更新。
- **ZS-CoT（Zero-shot Chain-of-Thought）**：Kojima et al. (2022) 提出的 "Let's think step by step" 零样本推理 prompt。
- **Meta-prompt**：用于指导 LLM 生成/优化其他 prompt 的高级 prompt，通常包含问题描述、历史方案和生成指令。

## 可复现要素
- **数据集**：9 个文本分类数据集（SST-2/5, MR, CR, MPQA, Subj, TREC, AGNews, DBPedia）均公开；GSM8K 公开。训练集 64 样本/数据集，测试集使用 Lu et al. (2022) 的子采样版本。
- **代码/权重**：论文未声明开源代码仓库，但所有实验模型（GPT-2, Mistral, Llama2 系列）均为公开模型；实验细节充分（采样数 160、温度 1.0/0.0、一步上下文示例等），可复现性高。
- **关键超参**：采样预算 k=160；训练样本 n=64；温度 1.0（生成）、0.0（分类）；OPRO 最多 40 步优化、每步 4 候选；Random with Context 使用 3 个随机训练示例作为 context。
