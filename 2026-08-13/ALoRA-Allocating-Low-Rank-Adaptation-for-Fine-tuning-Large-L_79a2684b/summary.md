---
title: "ALoRA-Allocating-Low-Rank-Adaptation-for-Fine-tuning-Large-L"
source: https://aclanthology.org/2024.naacl-long.35.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:31:06"
field: "大语言模型参数高效微调"
keywords: ["参数高效微调", "LoRA", "低秩适配", "神经架构搜索", "rank分配", "ALoRA", "AB-LoRA"]
innovations: ["提出AB-LoRA重要性评分方法，通过ablation对比准确评估每个LoRA rank的贡献度", "提出ALoRA框架，以初始即满足预算的方式迭代剪枝低价值rank并将预算重新分配给关键模块", "揭示了Q/K模块需更高rank、FFN模块需较少rank的LoRA分配有效模式"]
benchmarks: ["GLUE", "SuperGLUE", "SQuAD", "E2E", "MT-Bench"]
---

# 论文速读：ALoRA-Allocating-Low-Rank-Adaptation-for-Fine-tuning-Large-L

## 一句话总结
本文提出了 ALARA（Allocating Low-Rank Adaptation）框架，通过新颖的 AB-LoRA 方法评估每个 LoRA rank 对超网络性能的贡献度，进而迭代地剪枝低重要性 rank、并将预算重新分配给关键 Transformer 模块，最终在不增加初始 rank 预算的前提下实现更优的下游任务适配效果。

## 研究问题与动机
- **固定 rank 的局限性**：标准 LoRA 在每个 Transformer 权重矩阵上使用相同的内禀 rank，但最优 rank 值会因骨干模型、任务甚至模块类型而异，固定设置并非最优。
- **既有自适应 LoRA 方法需更大初始预算**：AdaLoRA、SoRA、SaLoRA 等方法需预先初始化更大的最大 rank 值（如 16），在训练过程中逐步剪枝，导致额外的 GPU 显存消耗和受限的 rank 搜索空间。
- **现有重要性估计不可靠**：已有方法依赖启发式敏感性分数或 DNAS 风格的结构权重（$\alpha_i'$），并不能准确反映每个 LoRA rank 的真实贡献。
- **关键模块的差异化适配需求**：不同 Transformer 子模块（如 Q/K/V/Output、FFN 各层）对下游任务知识的需求可能完全不同，需要动态调整 rank 分配策略。

## 核心贡献（创新点）
1. **提出 AB-LoRA 方法**：通过直接评估每个 LoRA rank 被移除或单独保留时对超网络性能的影响来计算重要性得分，区别于以往依赖启发式分数或结构权重的方式，能更准确地反映 rank 真实贡献。
2. **提出 ALoRA 框架**：首次将 LoRA rank 分配问题建模为神经架构搜索问题，以初始化即满足目标预算（$r_m^{init} = R^{target} / N_{mod}$）为前提，在训练过程中迭代剪枝并重新分配 rank，无需额外显存开销。
3. **在多个任务上超越最强基线**：在 GLUE/SuperGLUE、SQuAD、E2E 及指令微调等任务上，以可比或更少的可调参数量，持续超越 LoRA、AdaLoRA、SoRA、SaLoRA 等多种基线方法。
4. **揭示了 LoRA rank 分配的有效模式**：实验可视化表明，查询（Query）和键（Key）模块应获得更高 rank，而前馈网络（FFN）模块所需 rank 更少，验证了"注意力模块学习任务特定知识、FFN 保留通用语言知识"的直觉。

## 方法详解
**整体思路**：将 LoRA rank 分配视为神经架构搜索（DNAS）问题，引入可学习门控参数 $\alpha_i \in [0,1]$ 控制每个 rank 的激活程度，以双目标优化方式联合更新架构参数和模型参数。

**关键设计**：
- **初始化策略**：每个 Transformer 模块 $m$ 的 LoRA rank 初始化为 $r_m^{init} = R^{target} / N_{mod}$（其中 $R^{target} = 8 \times N_{mod}$），故初始即可满足总预算，无需预分配更大 rank。
- **门控前向传播**：对每个模块 $m$，引入对角门矩阵 $G_m = \text{diag}(\alpha_{m,1}, ..., \alpha_{m,r_m})$，其中 $\alpha_{m,i} = 2 \times \text{Sigmoid}(a_{m,i}')$，$a_{m,i}'$ 初始化为 0，使 $\alpha_{m,i} \in (0,1)$。前向传播为 $z = x W_m^A G_m' W_m^B$。
- **AB-LoRA 重要性评分公式**：设 $M$ 为完整超网络，$M_{\backslash r}$ 为屏蔽 rank $r$ 后的网络，$M_r$ 为仅保留 rank $r$ 的网络，定义：
$$IS(r) = S(M) - S(M_{\backslash r}) + S(M_r)$$
其中 $S(\cdot)$ 为负交叉熵损失（取负号而非 accuracy/F1，是因为单一 rank 屏蔽对分类指标影响可能不显著，且不适用于生成任务）。
- **迭代剪枝与再分配流程（Algorithm 1）**：
  1. 在训练集 $D_{train}$ 上训练超网络 $K_1 = 1$ epoch，冻结架构参数仅优化模型参数；
  2. 在验证集小批量 $B_{val}$（batch size=32）上计算每个 rank 的重要性得分；
  3. 剪枝 $n_A$（默认 $n_A = 1 \times N_{mod}$）个最低分的 rank（对应 gate 置 0）；
  4. 若无模块被剪枝，则将剪掉的 rank 预算重新分配给未受剪枝的模块；
  5. 在训练集上继续微调 $K_2 = 0.25$ epoch 恢复性能；
  6. 重复上述步骤最多 $N_A = 8$ 次。

## 实验与结果
**数据集**：GLUE（SST-2、RTE、QNLI）、SuperGLUE（BoolQ、COPA、ReCoRD）、SQuAD、E2E（NLG）、Alpaca（指令微调）+ MT-Bench 评测。

**骨干模型**：LlaMA-2 7B（主实验），RoBERTa-large 和 GPT2-large（消融实验）。

**主要结果（LlaMA-2 7B  backbone，Table 1）**：
- ALoRA 在全部 7 个任务上均取得**最优或次优**成绩，初始可调参数仅 20.0M（与 LoRA 相同），远低于 AdaLoRA/SoRA/SaLoRA 的 40.0M。
- **最强结果举例**：
  - SST-2: ALoRA **95.0** acc（次优 SoRA 94.2，提升 **+0.8**）
  - BoolQ: ALoRA **88.0** acc（次优 SoRA 87.6，提升 **+0.4**）
  - ReCoRD: ALoRA **91.8** f1-em（次优 SoRA 91.0，提升 **+0.8**）
  - SQuAD: ALoRA **89.2** f1-em（次优 SoRA 88.5，提升 **+0.7**）
- **E2E 生成任务**（Table 2）：ALoRA BLEU **70.6**，ROUGE-L **71.8**，METEOR **47.1**，均优于 SoRA 和 LoRA。
- **指令微调**（Table 3）：ALoRA 在 MT-Bench 上 GPT-4 评分 **7.47**，SoRA 为 **7.16**；ROUGE-L 54.3 vs 53.2。

**效率对比**（Table 4）：ALoRA 训练峰值显存 18.1 GB（低于 SoRA 的 18.8 GB），速度 5.01 it/s（与 LoRA 相当）。

**鲁棒性**：在不同 rank 预算（$1 \sim 128 \times N_{mod}$）下 ALoRA 始终优于基线；在 RoBERTa-large 和 GPT2-large 上也取得超越。

## 相关工作脉络
1. **AdaLoRA（Zhang et al., 2023c）**：基于 SVD 分解并通过敏感性重要性分数识别重要 rank；区别在于 AdaLoRA 需初始化更大 rank 并在训练中逐渐剪枝至目标预算，而 ALoRA 从初始即满足预算，无需额外显存，且重要性评估方式更准确。
2. **SoRA（Ding et al., 2023）**：通过 $l_0$ 范数惩罚和近端梯度下降剪枝冗余 LoRA rank；区别在于 SoRA 同样需要较大的初始 rank，且依赖启发式稀疏正则化而非精确的贡献度评估。
3. **SaLoRA（Hu et al., 2023）**：通过拉格朗日乘子法剪枝 LoRA rank；区别在于 SaLoRA 依赖正则化约束，而 ALoRA 将分配问题直接建模为 NAS 并通过 ablation 评估 rank 贡献。
4. **LoRA（Hu et al., 2021）**：原始低秩适配方法，固定 rank；ALoRA 是对其动态 rank 分配机制的扩展。
5. **DNAS（Liu et al., 2019a）**：可微分神经架构搜索；ALoRA 借鉴其架构参数化思路，但改进了重要性评估策略（用 ablation 而非结构权重）。
6. **QLoRA（Dettmers et al., 2023）**：通过 4-bit 量化降低 LoRA 微调显存；与 ALoRA 正交，可结合使用。

## 局限性与未来方向
- **实验规模限制**：未在更大的开源 LLM（如 LlaMA-2 13B、70B）上进行验证，仅在小规模模型（LlaMA-2 7B、RoBERTa-large、GPT2-large）上证明有效性。
- **任务覆盖有限**：未涉及信息提取等 NLP 任务，但作者认为框架可轻松迁移到其他骨干架构和任务类型。
- **未来方向**：将 ALoRA 应用于更大规模模型和多类型 NLP 任务，验证方法的可迁移性和优越性保持情况。

## 研究启发与可借鉴点
1. **AB-LoRA 重要性评分公式可直接迁移**：$IS(r) = S(M) - S(M_{\backslash r}) + S(M_r)$ 的三组对比评估思路可推广到 adapter 模块、prompt token 或其他 PEFT 组件的重要性量化，为自适应 PEFT 提供通用评估范式。
2. **"初始化即满足预算"的设计原则值得借鉴**：通过均匀分配初始 rank 而非预分配更大值来避免额外显存开销，这一思路可推广至其他需要动态分配资源的场景（如自适应 depth/width allocation）。
3. **Rank 分配的可视化分析揭示有用先验**：Q/K 模块获得更高 rank、FFN 模块 rank 更少这一发现，为后续工作提供了 rank 初始化先验的指导（可设计非均匀初始分配策略进一步加速收敛）。
4. **Gate 机制与 DNAS 框架的解耦**：本文将架构参数 $\alpha_i$ 的优化与模型参数优化解耦（先训练模型参数再固定架构参数评估），避免了双目标优化的额外开销，该简化策略可应用于其他 NAS 类 PEFT 方法。

## 关键术语表
- **ALoRA**：Allocating Low-Rank Adaptation，本文提出的动态分配 LoRA rank 的 PEFT 框架，通过 ablation-based 重要性评分迭代剪枝和重新分配 rank 预算。
- **AB-LoRA**：Ablation-Based LoRA，本文提出的 LoRA rank 重要性评分方法，通过比较完整超网络、移除单个 rank 的网络、仅保留单个 rank 的网络在验证集上的表现来量化 rank 贡献。
- **LoRA**：Low-Rank Adaptation，经典的 PEFT 方法，假设模型参数变化具有低秩结构，通过优化低秩分解矩阵实现高效微调。
- **PEFT**：Parameter-Efficient Fine-Tuning，参数高效微调，一类只微调少量参数即可适配大模型的技术，包括 Adapter、Prompt Tuning、LoRA 等。
- **Supernetwork**：超网络，指包含所有可能 LoRA rank（通过门控激活/屏蔽）的完整网络，作为 rank 分配搜索的基础架构。
- **Gate unit**：门控单元，值为 $\alpha_i \in [0,1]$ 的连续参数，控制每个 LoRA rank 的激活程度，最终通过置零实现 rank 剪枝。
- **GLUE / SuperGLUE**：广泛使用的自然语言理解基准测试套件，包含多项分类和推理任务，用于评估语言模型泛化能力。

## 可复现要素
- **数据集**：GLUE、SuperGLUE、SQuAD v1.1、E2E、Alpaca（清洗版）、MT-Bench；均为公开数据集。
- **代码/权重**：论文未明确声明开源代码仓库链接（仅提及使用 HuggingFace Transformers 和 PEFT 库），需关注作者是否同步发布。
- **关键超参**：$R^{target} = 8 \times N_{mod}$（LlaMA-2 7B 中 $N_{mod}=7$，故初始 $r_m^{init} = 8$）；$n_A = 1 \times N_{mod}$；$K_1 = 1$ epoch，$K_2 = 0.25$ epoch，$N_A = 8$；$B_{val}$ batch size = 32；学习率 1e-4；AdamW；warmup 6%；最大序列长度 2048；beam size = 5；5 次随机种子取中位数。
