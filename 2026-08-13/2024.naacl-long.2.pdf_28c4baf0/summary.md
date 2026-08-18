---
title: "Text Diffusion Model with Encoder-Decoder Transformers for Sequence-to-Sequence Generation"
source: https://aclanthology.org/2024.naacl-long.2.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:28:48"
field: "文本生成与扩散模型"
keywords: ["文本扩散模型", "序列到序列生成", "自适应噪声调度", "自条件技术", "Transformer架构"]
innovations: ["提出Encoder-Decoder架构的S2S扩散模型SeqDiffuSeq，推理效率较DiffuSeq提升3.56倍", "引入Self-conditioning技术与Token级自适应噪声调度，分别提升生成质量与去噪均衡性", "设计基于Prior分布替换的采样策略以平衡生成质量与多样性"]
benchmarks: ["QQP改写", "Wiki-Auto文本简化", "Quasar-T问题生成", "IWSLT14/WMT14机器翻译"]
---

# 论文速读：Text Diffusion Model with Encoder-Decoder Transformers for Sequence-to-Sequence Generation

## 一句话总结
本文提出 SeqDiffuSeq，将连续扩散模型扩展至序列到序列（S2S）文本生成任务，通过引入 Encoder-Decoder Transformer 架构、自条件（Self-conditioning）技术以及自适应噪声调度策略，在保持生成质量的同时显著提升了推理效率。

## 研究问题与动机
- **离散文本与连续扩散模型的适配难题**：扩散模型在图像/音频等连续域表现优异，但文本是离散词元序列，直接扩展至自然语言生成极具挑战。
- **已有扩散模型局限于无条件/条件生成**：现有工作（如 Multinomial Diffusion、D3PM、DiffusionLM）主要关注无条件或受控文本生成，缺乏对 S2S 场景的系统探索。
- **DiffuSeq 的架构缺陷**：DiffuSeq 采用 Encoder-only Transformer，每步扩散均需重新编码输入序列，推理开销大；且仅在序列级设计噪声调度，无法适应不同位置的生成难度差异。

## 核心贡献（创新点）
1. **提出 SeqDiffuSeq 的 Encoder-Decoder 扩散架构**：将 DiffusionLM 的连续扩散框架扩展至 S2S 设置，编码器仅在推理时前向一次，Decoder 处理带噪输出序列，显著降低计算开销。与 DiffuSeq 的本质区别在于架构选择与推理效率。
2. **引入 Self-conditioning 技术**：将上一轮预测序列 $\hat{z}_0^t$ 作为额外输入，帮助去噪函数充分利用历史信息进行细化而非从零预测。与 Bit-Diffusion 的本质区别在于适配到 S2S 场景与 Transformer 建模方式。
3. **提出 Token 级自适应噪声调度（Adaptive Noise Schedule）**：根据各 token 位置的训练损失动态调整噪声调度，使去噪难度在时间步间线性均衡，并为不同位置分配独立调度曲线。与 DiffusionBERT  spindle schedule 和 CDCD learned schedule 的本质区别在于 token 粒度与均衡化目标。

## 方法详解
- **前向扩散过程**：通过词嵌入函数 $g_\phi$ 将输出序列 $w_y$ 映射为连续嵌入 $g_\phi(w_y) \in \mathbb{R}^{n \times d}$，沿 Markov 链逐步添加高斯噪声，直至 $z_T$ 接近标准正态分布。
- **反向去噪过程**：去噪函数 $z_\theta^0(z_t, w_x, t)$ 由 Encoder-Decoder Transformer 实现，Encoder 编码输入 $w_x$，Decoder 以全注意力矩阵建模带噪输出序列 $z_t$，并注入时间步嵌入 $t$。最终通过最近邻舍入（rounding）得到离散词序列。
- **训练目标**：简化损失 $\mathcal{L}_{simple}$ 包含三项：中间步预测误差 $\|z_\theta^0 - z_0\|^2$、终端噪声项、以及初始 embedding 重建项，整体优化嵌入参数与去噪网络。
- **Self-conditioning 训练策略**：以 50% 概率将 $\hat{z}_0^t$ 置零（跳过），否则先用 $z_\theta^0(z_t, 0, w_x, t)$ 估计 $\hat{z}_0^t$ 再拼接输入，且不反向传播过该估计值，避免计算负担。
- **自适应噪声调度**：按 token 位置 $i$ 记录训练损失 $\mathcal{L}_t^i$，每隔固定步数（20,000步）拟合映射 $M_i: \mathcal{L}^i \to \bar{\alpha}^i$（线性插值），再将均匀采样的新损失值代入映射更新 $\bar{\alpha}_t^i$，实现各位置独立且随训练动态调整的调度。

## 实验与结果
- **数据集与任务**：QQP（改写）、Wiki-Auto（简化）、Quasar-T（问题生成）、CCD（对话）、IWSLT14/WMT14（德英机器翻译），共五个任务六份数据集。
- **基线对比**：AR 基线（Transformer、GPT-2-large FT）、NAR 基线（LevT、CMLM）、扩散基线（DiffuSeq、CDCD）。
- **核心结果**：
  - **QQP**：SeqDiffuSeq BLEU=23.28，MBR=10 时 BLEU=24.34，超过 DiffuSeq MBR=10（24.13）。
  - **Wiki-Auto**：SeqDiffuSeq BLEU=37.09，超越 DiffuSeq MBR=10（36.43）。
  - **Quasar-T**：SeqDiffuSeq BLEU=17.20，单生成即超 DiffuSeq MBR=10（17.01）；MBR=10 时达 17.46。
  - **翻译**：IWSLT14 EN-DE SacreBLEU=21.96，DE-EN=30.16；WMT14 EN-DE=19.16，DE-EN=23.63，整体优于 CMLM（+6.32~6.75 平均点），略逊于 AR Transformer。
  - **推理速度**：单 V100 上 SeqDiffuSeq 较 DiffuSeq 加速 **3.56 倍**（89 sec vs 317 sec）。
- **消融结论**：移除自适应噪声调度平均 BLEU 下降 2.29，移除 Self-conditioning 下降 1.34，两者联合移除下降 5.71，证明各自独立有效且互补。

## 相关工作脉络
- **DiffuSeq (Gong et al., 2022)**：首个 S2S 扩散模型，采用 Encoder-only 架构，序列级固定噪声调度；本文在架构与调度策略上全面升级。
- **DiffusionLM (Li et al., 2022)**：将连续扩散引入词嵌入空间，仅支持无条件/受控生成；本文将其扩展至条件 S2S 生成。
- **DiffusionBERT (He et al., 2022)**：基于 D3PM 的语言建模扩散方法，提出 spindle schedule；本文聚焦 S2S 任务且噪声调度为 token 级自适应。
- **CDCD (Dieleman et al., 2022)**：并发工作，提出可学习噪声调度用于语言建模与机器翻译；本文侧重于序列级均衡与位置自适应设计。
- **Bit-Diffusion (Chen et al., 2022)**：提出 Self-conditioning 技术；本文将其适配至 Encoder-Decoder 架构与 S2S 场景。

## 局限性与未来方向
- **推理步数仍较多**：需 T=2000 步反向采样，计算开销虽已优化但仍高于自回归模型，减少采样步数是明确方向。
- **生成多样性下降**：Self-conditioning 与自适应调度提升了单序列质量，但以牺牲不同随机种子下的多样性为代价，MBR 增益有限。
- **CCD 对话任务性能不足**：所有模型在 CCD 上表现均较差，可能因该任务对常识知识要求较高，扩散模型的生成策略有待改进。
- **MBR 解码边际收益**：本文提出基于 Prior 分布替换的采样补偿方案，但质量与多样性的权衡仍需深入探索。

## 研究启发与可借鉴点
1. **Encoder-Decoder 在扩散 S2S 中的效率优势**：推理时仅需一次编码器前向计算，可推广至其他需要条件输入的扩散生成任务（如图像到图像、结构化数据生成）。
2. **Token 级自适应噪声调度的通用性**：该策略可迁移至任何序列生成任务，尤其是位置依赖强度不同的任务（如翻译中源语位置 vs 目标语位置的生成难度差异）。
3. **Self-conditioning 的半概率训练技巧**：50% 概率跳过历史预测的轻量实现，避免了逐步采样带来的训练负担，适合其他迭代式生成模型。
4. **Prior 分布替换采样策略**：通过随机切换去噪分布与先验分布增加多样性，可作为 MBR 解码的高效补充，值得在多样性敏感的下游任务中验证。
5. **去噪函数建模为非自回归序列预测**：Decoder 使用全注意力而非因果注意力，这一设计与 LevT 等 NAR 模型思路一致，可在需要并行生成的扩散框架中复用。

## 关键术语表
**SeqDiffuSeq**：本文提出的基于 Encoder-Decoder Transformer 的序列到序列文本扩散生成模型。
**Self-conditioning**：将上一轮去噪预测序列 $\hat{z}_0^t$ 作为额外输入，帮助当前步去噪函数细化结果的技术。
**Adaptive Noise Schedule**：根据 token 位置训练损失动态调整噪声调度曲线的机制，使各位置去噪难度在时间步上均衡。
**MBR (Minimum Bayes Risk) Decoding**：通过生成多个候选序列并选取风险最小的输出，以提升生成质量的解码策略。
**DiffuSeq**：先前基于 Encoder-only Transformer 的 S2S 扩散模型，本文的主要对比基线。
**Rounding Distribution**：将连续 embedding 空间中的输出映射回最近离散词元的后处理步骤。
**$\bar{\alpha}_t^i$**：第 $i$ 个 token 位置在第 $t$ 时间步的累积信噪比参数，控制噪声注入量。

## 可复现要素
- **数据集**：QQP、Wiki-Auto、Quasar-T、CCD、IWSLT14、WMT14（均为公开数据集，许可证见附录 C）。
- **代码/权重**：论文未明确声明开源状态，需查阅对应项目页面确认。
- **关键超参**：扩散步数 $T=2000$；Transformer 6 层 Encoder + 6 层 Decoder；学习率 $10^{-4}$，warmup 10,000 步；自适应调度每 20,000 步更新，K=20；词汇表大小翻译任务 10K（IWSLT14）/32K（WMT14），其余任务使用 bert-base-uncased 词表。
- **硬件**：训练使用 8× NVIDIA A100 80GB GPU；推理评测使用单张 NVIDIA V100 GPU。
