---
title: "Adaptive-Rank-Selections-for-Low-Rank-Approximation-of-Langu"
source: https://aclanthology.org/2024.naacl-long.13.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:31:38"
---

# 论文速读：Adaptive Rank Selections for Low-Rank Approximation of Language Models

## 一句话总结
本文提出自适应秩选择（ARS）方法，通过可微分的超网络与奇异值对齐正则化，为语言模型各线性层动态分配最优压缩秩数；该方法无需重新训练或仅需极少微调，即可在BERT系列与LLaMA-7B上显著超越均匀秩SVD及其加权变体。

## 研究问题与动机
- **均匀秩分配的次优性**：现有SVD及加权SVD方法（如FWSVD、IWSVD）对所有层使用相同秩数，但不同操作（层）对模型容量的需求是非均匀的，统一分配会浪费重要层容量或压缩关键层过多。
- **离散秩优化的计算壁垒**：全局最优秩选择是离散、非光滑且非凸的组合优化问题，强化学习或进化算法的搜索成本对大模型不可承受。
- **元素级掩码的结构破坏**：直接对奇异值逐元素施加二值掩码容易跳过大奇异值，破坏SVD的降序正交结构，导致压缩后模型难以通过微调恢复性能。
- **如何低成本实现层间自适应分配**：在冻结预训练权重的前提下，构建端到端可微框架高效学习各层秩数，仍是低秩压缩领域的关键开放问题。

## 核心贡献（创新点）
- **超网络驱动的自适应秩预测框架**：用Bi-GRU与线性层构成的超网络生成各层二值掩码，通过掩码之和表征保留秩数，实现跨层联合可微优化。与以往逐层固定秩或网格搜索的本质区别在于：将离散结构搜索转化为连续可微学习，避免高昂的组合优化代价。
- **奇异值对齐正则化（$\mathcal{R}_{align}$）**：构造目标top-k掩码并与预测掩码计算L2差异，强制超网络输出严格遵循SVD降序奇异性质的“前k大”选择。与直接元素级掩码的本质区别在于：杜绝跳跃选择不连续高价值秩导致的容量浪费与微调收敛困难。
- **免微调/少微调性能显著提升**：在BERT-base、DistillBERT、MobileBERT及LLaMA-7B上系统验证，ARS在相同参数量下全面超越均匀秩SVD、IWSVD、FWSVD及结构剪枝基线。与既往工作的本质区别在于：首次证明“合理分配各层秩”与“设计重要性加权度量”正交且同等重要，且该分配策略对后续微调具有强迁移友好性。

## 方法详解
- **掩码插入低秩分解**：对线性层权重 $\mathbf{W}$ 进行SVD得 $\mathbf{W} = \mathbf{U}\mathbf{S}\mathbf{V}^T$，在奇异值对角矩阵上施加二值掩码 $\mathbf{\hat{s}} = m \odot s$，前向计算变为 $\mathbf{Y} = \mathbf{X}(\mathbf{U}\text{Diag}(\hat{s}))\mathbf{V}^T + \mathbf{b}$，使掩码可在标准反向传播中获取梯度。
- **超网络（Hypernetwork）生成掩码**：采用由Bi-GRU(32,64)→LayerNorm→GeLU→线性层构成的HN，输入预采样固定向量 $z \sim \mathcal{N}(0,1)$，输出连续 logits $o_l$；经Straight-Through Gumbel-Sigmoid转化为可微二值掩码 $m_l = \text{round}(\text{sigmoid}((o_l + g + b)/\tau))$，其中 $g \sim \text{Gumbel}(0,1)$，$\tau=0.4$，$b=3.0$ 保证训练初期全秩启动。GRU捕获层间依赖，线性层适配不同操作维度。
- **奇异值对齐正则化**：令 $m'_l$ 为前 $\mathbf{1}^T m_l$ 个元素为1的理想top-k掩码，定义 $\mathcal{R}_{align}(m_l) = \|m_l \odot s - m'_l \odot s\|_2^2$。该正则项无缝嵌入HN优化，无需额外参数，迫使预测掩码与SVD降序结构对齐。
- **整体优化目标**：
  $$\min_\theta \mathcal{L}(f(x;m), y) + \lambda \mathcal{R}(T(m), pT_{total}) + \gamma \frac{1}{L}\sum_{l=1}^N \mathcal{R}_{align}(m_l)$$
  其中 $\mathcal{R}(a,b)=\log(\max(a,b)/b)$ 约束总参数量不超过目标比例 $p$，$\lambda=16, \gamma=10$；预训练模型权重冻结，仅更新HN参数 $\theta$。
- **压缩与微调流程**：HN训练完成后，对第 $l$ 层截断 $\mathbf{U}_l \leftarrow \mathbf{U}_l[:, :k_l]$、$\mathbf{V}_l \leftarrow \mathbf{V}_l[:, :k_l]$（$k_l = \mathbf{1}^T m_l$），完成低秩近似；随后在下游任务（GLUE/Pile/WikiText）上进行标准微调。

## 实验与结果
- **评估设置**：GLUE基准（BERT-base、DistillBERT、MobileBERT）、LLaMA-7B（Pile预训练+6个下游评测）、WikiText-103（Pythia-160m）；基线覆盖Vanilla SVD、IWSVD、FWSVD、IE剪枝、CoFiPruning、LLM-Pruner及同参数量从头训练。
- **BERT-base（$p=0.48$，保留约48%参数）**：SVD+ARS微调后平均81.61，较SVD+FT（78.42）提升3.19；IWSVD+ARS达83.14（+2.21）；FWSVD+ARS达83.14（+2.19）。**免微调阶段**SVD+ARS平均60.48，远超SVD的41.90，凸显自适应秩对任务信息的保留能力。
- **更高压缩率（$p=0.33$）**：SVD+ARS提升20.09/3.44，IWSVD+ARS提升25.63/4.76，FWSVD+ARS提升27.18/3.05；压缩越激进，ARS优势越显著（图2/图A1验证）。
- **紧凑模型（DistillBERT/MobileBERT）**：FWSVD+ARS在DistillBERT上免微调/微调分别提升18.79/2.67；在MobileBERT上提升18.99/1.71，证明方法对轻量化架构同样有效。
- **LLaMA-7B压缩（移除约75%参数，$p=0.24$）**：WSVD+ARS平均46.92，优于WSVD（44.67）与LLM-Pruner
