---
title: "Unlocking Emergent Modularity in Large Language Models"
source: https://aclanthology.org/2024.naacl-long.144.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:28:13"
field: "大语言模型微调与泛化"
keywords: ["Emergent Modularity", "Mixture-of-Experts", "Parameter-efficient Fine-tuning", "LLM Generalization", "Modular Neural Networks", "Out-of-distribution"]
innovations: ["提出EMoE：通过key向量聚类将预训练FFN零参数外部化为MoE结构以解锁emergent modularity", "证明EMoE的收益来自微调阶段对负迁移神经元的屏蔽，而非推理时的稀疏激活", "avg-k无参gate比learned gate更稳定，且EMoE微调后可合并回标准架构部署"]
benchmarks: ["GLUE", "GLUE-X", "Domainbed", "MMLU", "SuperGLUE (ATTEMPT multi-task)"]
---

# 论文速读：Unlocking Emergent Modularity in Large Language Models

## 一句话总结
本文提出 **EMoE（Emergent Mixture-of-Experts）**，通过将预训练 Transformer 的 FFN 层按 emergent modularity 聚类分解为 MoE 结构，在不引入额外参数的条件下解锁并利用了 LLM 中的隐式模块化，显著提升了下游任务的内分布（ID）与跨分布（OOD）泛化能力。

## 研究问题与动机
1. **显式 vs. 隐式模块化**：现有模块化神经网络（MNNs/MoEs）多为预定义显式架构，而近年研究表明标准预训练 Transformer 的 FFN 层中存在隐式 emergent modularity（EM），但大多数 LM 在微调阶段仍被视为单体模型，EM 被"锁住"而未加利用。
2. **EM 能否提升下游性能**：若预先训练的 FFN 层已自发形成功能分组的神经元聚类，将其外部化为 MoE 是否能在微调时带来 ID/OOD 泛化增益？
3. **无额外参数解锁的需求**：已有方法（如 GMoE、MoEfication）或复制 FFN 引入额外参数，或在推理效率上追求优化但忽略 EM 对训练过程的影响；需要一种零额外参数开销的方法来验证 EM 的真正价值。
4. **可扩展性验证**：现有分析多在中小规模模型（≤1.5B）上进行，需在 Llama2-7B / Llama-30B 级别验证方法的实用性。

## 核心贡献（创新点）
1. **提出 EMoE 框架**：基于关键向量聚类将预训练 FFN 分解为多个专家（expert），以专家 key 的平均值构建无参 gate（avg-k gating），实现从单体到 MoE 的零参数外部化。与 MoEfication（追求推理加速）和 GMoE（复制 FFN 引入额外参数）本质不同。
2. **证明 EM 解锁可提升 ID 与 OOD 泛化**：在 BERT-Large、GPT2-XL、Llama2-7B/30B 等多规模模型上，EMoE 微调 consistently 优于 vanilla LoRA 微调，且 OOD 指标均有提升。
3. **揭示 EMoE 改善的是微调期间的参数更新而非推理**：通过 LoRA-to-EMoE 与 EMoE-to-LoRA 对照实验证明，EMoE 的收益来自微调过程中屏蔽了具有负迁移效应的神经元，而非推理时的稀疏激活；且微调后可合并回标准架构部署，无需改动线上模型。
4. **验证 avg-k gating 的稳定性优势**：相较于可学习的 learned gating（EMoE-learn），avg-k gating 在微调过程中专家选择更稳定，避免因 gate 不一致导致的 data inefficiency，且整体性能更优。
5. **在大规模 Llama 模型上的成功扩展**：将 EMoE 应用于 Llama2-7B（+0.62 accuracy on MMLU with LoRA；全参微调 +1.58）与 Llama-30B（+0.93），证明了方法的强可扩展性。

## 方法详解
1. **FFN 的 Key-Value Memory 视角**：FFN 层输出为 $\mathbf{y} = \sigma(\mathbf{x} \cdot \mathbf{K}) \cdot \mathbf{V} = \sum_{i=1}^{h} \sigma(\mathbf{x} \cdot \mathbf{K}_{:,i}) \cdot \mathbf{V}_{i,:}$，将 K 的列视为 key 向量、V 的行视为 value 向量，每个 (key, value) 对构成一个"神经元"。
2. **基于聚类的专家构建（Clustering-based Experts Construction）**：对 FFN 层的所有 key 向量 $\mathbf{K}$ 进行约束聚类（constrained clustering / balanced k-means），将 $d$ 个 key 均分为 $N$ 组（通常 $N=64$），每组构成一个 expert $\mathrm{FFN}^i(\cdot; \mathbf{K}^i, \mathbf{V}^i)$，总参数量与原始 FFN 完全相同。
3. **Avg-k Gating（无参门控）**：第 $i$ 个专家的 gate 权重为其组内 key 的平均值：$\mathbf{G}_{:,i} = \mathrm{Avg}(\mathbf{K}^i, \dim=0)$。门控分数为 $\mathbf{x} \cdot \mathbf{G}_{:,i}$，即组内所有神经元激活分数之和的线性近似。采用 Top-k 选择 $k$ 个最高分的专家（默认 $k=16$ 或 $32$，保持 $k/N = 0.25$ 或 $0.5$）。
4. **EMoE 前向计算**：$\mathbf{y} = \sum_{n \in N} g_n(\mathbf{x}; \mathbf{G}, k) \cdot \mathrm{FFN}^n(\mathbf{x}; \mathbf{K}^n, \mathbf{V}^n)$，其中 $g_n$ 为二元 Top-k 选择函数。微调期间 gate 权重与 FFN 参数绑定（Eq.4），不引入独立可训练参数。
5. **部署友好性**：微调完成后，可将 EMoE  expert 重新合并为标准 FFN 层，部署时无需 MoE 架构支持。

## 实验与结果
- **数据集与基准**：
  - NLP ID：GLUE（MRPC, CoLA, RTE, STSB, SST2, QNLI, QQP, MNLI）
  - NLP OOD：GLUE-X（13 个跨域任务，Friedman rank 聚合）
  - 视觉 OOD：Domainbed（PACS, VLCS, Office-Home, Terra Incognita，3 种 selection criterion）
  - LLM 指令微调：Alpaca 训练，MMLU 评估
- **基线**：vanilla LoRA、GMoE（复制 FFN + 学习 gate）、EMoE-learn（avg-k 改 learned gate）
- **主要结果**：
  - **BERT-Large（LoRA）**：EMoE ID 平均提升 **+0.74**（85.09 vs 84.35），OOD 指标 **4.37**（优于 LoRA 的 4.86）；RTE 从 72.92 提升至 75.21。
  - **GPT2-XL（LoRA）**：EMoE ID 平均提升 **+0.84**（85.45 vs 84.61），OOD 指标 **3.88**（显著优于 LoRA 的 5.61）。
  - **Llama2-7B（MMLU，LoRA）**：EMoE（N=64, k=16）达 **47.58**（vs baseline 46.96，+0.62）；全参微调下达 **48.08**（vs 46.50，**+1.58**）。
  - **Llama-30B（MMLU）**：EMoE（N=256, k=64）达 **57.11**（vs baseline 56.18，+0.93）。
  - **Domainbed（ViT）**：EMoE 与 SOTA GMoE 相当，ViT-Base avg 达 **74.65**（Train-validation）。
  - **多任务学习（T5-Base，ATTEMPT 设置）**：ID 最高提升 **+7.56**，OOD 提升 **+1.58**。
- **最强结果**：GPT2-XL OOD 4.33 → 3.88（GMoE 基准已达 4.33，EMoE 以更少参数实现同等甚至更优 OOD）；Llama2-7B 全参微调 MMLU +1.58。

## 相关工作脉络
1. **GMoE（Li et al., 2023）**：复制预训练 FFN 形成 MoE 并学习 gate，声称提升 OOD 泛化；EMoE 不复制 FFN、不引入额外可训练参数，通过聚类 key 直接外部化 EM。
2. **MoEfication（Zhang et al., 2022b）**：将 FFN 分解为稀疏 MoE 以提升推理效率，关注推理加速而非下游微调性能；EMoE 借鉴其聚类思想但目标与评估体系完全不同。
3. **MoEBert（Zuo et al., 2022b）**：基于重要性引导的 BERT→MoE 适配；主要面向推理压缩，未系统探究 EM 对微调阶段的增益机制。
4. **Emergent Modularity 发现（Zhang et al., 2023; Li et al., 2022）**：首次揭示预训练 Transformer FFN 中神经元的 sparse activation 与任务相关的功能分组；本文在此基础上提出实用化的解锁方法。
5. **Sparse Upcycling（Komatsuzaki et al., 2023）**：从 dense checkpoint 训练 MoE；依赖额外训练过程；EMoE 无需任何额外训练阶段即可在标准微调流程中获益。
6. **Key-Value Memory FFN 视角（Geva et al., 2021, 2022）**：奠定将 FFN 视为 key-value 存储的理论基础，是本文方法设计的前提。

## 局限性与未来方向
1. **分解方法较简单**：主要借用 MoEfication 的聚类策略，未探索更精细的 EM 挖掘算法（论文自述）。
2. **未验证高难度任务**：如数学推理（Mathematical Reasoning）等挑战性 benchmark 尚未测试，泛化边界待探索。
3. **主体分析规模上限 1.5B**：虽然验证了 Llama-30B 的可扩展性，但机制分析（如负迁移、专家选择可视化）主要在较小模型上进行，大模型内部机制仍需深入理解。
4. **硬件实现开销**：使用公共 MoE 库（tutel）时比标准模型慢约 10%、显存多 <5%；自实现可接近标准速度，但实际部署仍需优化。
5. **Expert 利用率不均衡**：部分 expert 在微调中几乎未被选择（图 5a），存在潜在的负载不均衡问题。

## 研究启发与可借鉴点
1. **无参解锁 EM 的设计范式**：聚类 key + avg-k gating 是一个简洁有效的"零成本"模块化外部化方案，可直接迁移到其他预训练架构（如 ViT、多模态模型）的下游微调中。
2. **"微调后合并回标准架构"的工程实践价值**：EMoE 的收益发生在微调阶段而非推理阶段，这一性质使得 MoE 思想可以安全应用于无法部署 MoE 架构的生产环境，值得在团队现有 pipeline 中尝试。
3. **负迁移抑制机制的启发**：Top-k 选择有效屏蔽了具有负迁移效应的神经元（Bottom-k / Not-top-k 均劣于 vanilla），这为设计"选择性屏蔽有害参数"的微调策略提供了新思路。
4. **avg-k gating vs. learned gating 的稳定性对比**：固定 gate 在微调过程中表现更稳定，避免因 gate 不稳定导致的数据效率下降；在参数高效微调场景中可作为默认选择。
5. **与低资源/少样本微调结合**：EMoE 在仅用 20% 训练数据时仍优于全量微调 baseline（图 7），对低资源场景有直接应用价值。

## 关键术语表
- **Emergent Modularity (EM)**：预训练 Transformer 中自发形成的神经元功能分组现象，相似功能的神经元倾向于共激活。
- **EMoE (Emergent Mixture-of-Experts)**：本文提出的方法，将预训练 FFN 按 EM 聚类分解为 MoE 结构，无额外参数。
- **Avg-k Gating**：以专家组内 key 的平均值作为门控权重，通过 Top-k 选择激活专家，无需学习额外参数。
- **Key-Value Memory FFN**：将 FFN 层解释为 key-value 记忆系统，输入 query 与 key 内积得到激活分数，加权求和 value 得到输出。
- **Negative Transfer**：微调过程中某些神经元携带的知识对新任务产生负面影响，EMoE 通过 Top-k 选择屏蔽这些神经元。
- **EMoE-to-LoRA**：实验设计，微调完成后将 EMoE 合并回标准 FFN 进行推理，验证收益来自训练阶段而非推理结构。
- **GLUE-X**：扩展版 GLUE 基准，包含 13 个跨域 NLU 任务，用于评估模型的 OOD 泛化能力。
- **Constrained Clustering**：带约束的平衡聚类方法，确保各专家大小相近，用于将 key 向量分组构建 expert。

## 可复现要素
- **数据集**：GLUE、GLUE-X、Domainbed、Alpaca、MMLU、SuperGLUE（ATTEMPT 设置）——均为公开数据集。
- **代码**：论文声明代码开源，提供仓库链接（原文："Code is available at this repo"，具体 URL 见论文）。
- **权重**：使用公开预训练模型（BERT-Large、GPT2-XL、Llama2-7B、Llama-30B）。
- **关键超参**：专家数 N=64（GPT2/BERT）或 256（Llama-30B），top-k 分别为 16/32；LoRA rank=8, alpha=16；学习率搜索 [2e-4, 3e-4, 5e-4]；EMoE 替换最后 1-2 个 even FFN 层；$k/N$ 比例保持在 0.25 或 0.5。
