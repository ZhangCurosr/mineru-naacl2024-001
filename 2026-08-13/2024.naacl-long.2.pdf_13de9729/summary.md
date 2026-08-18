---
title: "Text Diffusion Model with Encoder-Decoder Transformers for Sequence-to-Sequence Generation"
source: https://aclanthology.org/2024.naacl-long.2.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:28:45"
field: "文本生成"
keywords: ["Text Diffusion Model", "Sequence-to-Sequence Generation", "Encoder-Decoder Transformer", "Self-Conditioning", "Adaptive Noise Schedule", "MBR Decoding", "Discrete Text Generation"]
innovations: ["提出 SeqDiffuSeq，以 Encoder-Decoder Transformer 架构将连续扩散模型扩展至 seq2seq 文本生成，推理加速 3.56 倍", "引入自条件技术使去噪网络能基于前序预测逐步精炼输出序列", "提出 token-level 自适应噪声调度，动态均衡各位置不同时间步的去噪难度"]
benchmarks: ["QQP", "Wiki-Auto", "Quasar-T", "CCD", "IWSLT14", "WMT14"]
---

# 论文速读：Text Diffusion Model with Encoder-Decoder Transformers for Sequence-to-Sequence Generation

## 一句话总结
论文提出了 SeqDiffuSeq，一种基于 Encoder-Decoder Transformer 的文本扩散模型，通过引入自条件（self-conditioning）技术与自适应 token-level 噪声调度，将连续扩散模型成功扩展到序列到序列文本生成任务，在保证生成质量的同时将推理速度提升至 DiffuSeq 的 3.56 倍。

## 研究问题与动机
- **离散文本难以直接应用连续扩散模型**：扩散模型在图像/音频等连续空间表现优异，但文本是由离散词元（token）组成的类别数据，直接将连续扩散框架迁移到 NLP 面临本质挑战。
- **现有文本扩散模型在 seq2seq 任务上存在架构局限**：DiffuSeq 等前作采用 encoder-only 架构，在反向去噪的每一步都需要对输入序列重新计算 encoder，推理效率低下。
- **固定噪声调度无法均衡不同 token 位置的去噪难度**：传统线性或余弦噪声调度对序列所有位置一视同仁，忽略了不同 token 位置内在特征与去噪难度的差异。
- **已有连续扩散文本方法仅限于无条件/受控生成**：DiffusionLM 等仅关注无条件或受控文本生成，尚未探索 seq2seq 条件下的有效扩散建模方案。

## 核心贡献（创新点）
1. **提出 SeqDiffuSeq，将连续扩散模型扩展至 Encoder-Decoder Transformer 架构的 seq2seq 生成**：区别于 DiffuSeq 的 encoder-only 架构，本文 encoder 仅需计算一次，大幅降低了推理开销。
2. **引入自条件（self-conditioning）技术提升扩散文本生成质量**：将前一步的序列预测 $\hat{z}_0^t$ 拼接作为 decoder 输入，使去噪函数能够从"从零预测"转变为"细化已有预测"，缓解信息浪费。
3. **提出自适应 token-level 噪声调度（adaptive noise schedule）**：按 token 位置独立学习噪声调度曲线，通过训练过程中的损失值动态调整 $\bar{\alpha}_t^i$，使各时间步的去噪难度均衡分布，优于固定 schedule。
4. **系统性实验验证：在 5 个 seq2seq 任务上实现高质量且高效的生成**：在多项指标上超越 DiffuSeq 与多种 AR/NAR 基线，推理速度提升 3.56 倍，且 self-conditioning 与自适应噪声调度具有互补性。

## 方法详解
**模型架构**：SeqDiffuSeq 采用 6 层 Encoder-Decoder Transformer（GeLU 激活），输入序列 $w_x$ 经 encoder 编码一次，decoder 在每一步去噪时处理带时间步嵌入的噪声序列 $z_t$，输出去噪后的 embedding 预测 $z_\theta^0(z_t, w_x, t)$，最终通过最近邻 rounding 映射回离散 token。

**前向扩散过程**：将目标序列 $w_y$ 通过 embedding 函数 $g_\phi$ 映射为连续向量后，按照 DDPM 的 Markov 链逐步加入高斯噪声，直至 $z_T \sim \mathcal{N}(0, I)$。

**训练目标（简化变分下界）**：
$$\mathcal{L}_{simple} = \mathbb{E}\left[\sum_{t=2}^{T}\|z_\theta^0(z_t, w_x, t) - z_0\|^2 + \|\tilde{\mu}(z_T, z_0)\|^2 + \|z_\theta^0(z_1, w_x, 1) - g_\phi(w_y)\|^2 - \log \tilde{p}_\phi(w_y|z_0)\right]$$
通过 reparameterization trick 实现 embedding 参数的可微优化。

**自条件技术**：在 reverse 过程中，将上一步预测 $\hat{z}_0^t = z_\theta^0(z_{t+1}, w_x, t+1)$ 在 embedding 维度上与 $z_t$ 拼接，使 decoder 输入维度变为 $n \times 2d$。训练时以 50% 概率将 $\hat{z}_0^t$ 置零，以 50% 概率先用 $z_\theta^0(z_t, 0, w_x, t)$ 估算 $\hat{z}_0^t$ 再进行自条件训练，且不回传第一次前向传播的梯度。

**自适应噪声调度**：核心思想是让去噪难度随时间步线性递增，并为每个 token 位置 $i$ 独立学习映射 $\bar{\alpha}^i = M_i(\mathcal{L}^i)$。具体地，记录各位置在每个时间步的训练损失 $\mathcal{L}_t^i = \|z_\theta^0(z_t)^i - z_0^i\|^2$，每隔一定训练步数，以粗粒度下采样（stride $K=20$）计算平均损失，通过线性插值拟合映射 $M_i$，再均匀采样新损失值并反查得到新的 $\bar{\alpha}_t^{i,new}$。初始化使用 DiffusionLM 的 sqrt schedule。

**推理策略**：默认单次生成即可取得较好质量；同时探索 MBR 解码（从 10 个随机种子候选中选最优）和 prior sampling 策略（以概率 $p_1$ 在前 $p_2$ 比例的时间步用高方差 prior 替代去噪分布以提升多样性）。

## 实验与结果
**数据集与任务**（5 个任务 6 个数据集）：
- QQP（paraphrase generation，test 2500）
- Wiki-Auto（text simplification，test 5000）
- Quasar-T（question generation，test 10000）
- CCD（dialogue generation，test 10000，因复现成本使用原文报告结果）
- IWSLT14 DE-EN / EN-DE（机器翻译）
- WMT14 DE-EN / EN-DE（机器翻译）

**评估指标**：BLEU、BERTScore、dist-1（词多样性）、SacreBLEU（翻译任务）。

**主要结果**：
- **QQP**：SeqDiffuSeq BLEU=23.28，MBR=10 时达 24.34，超越 DiffuSeq（18.47 / 24.13）；BERTScore 82.91 / 84.00 超过 GPT-2-large FT（83.63 仅单生成）。
- **Wiki-Auto**：SeqDiffuSeq BLEU=37.09，超过 DiffuSeq（29.89）甚至优于 DiffuSeq w/ MBR=10（36.43）。
- **Quasar-T**：SeqDiffuSeq BLEU=17.20，略超 DiffuSeq（15.84），MBR 后达 17.46。
- **翻译**：IWSLT14 EN-DE SacreBLEU=21.96（MBR 后 22.12），低于 AR Transformer（26.51）但显著超越 CMLM iter=1（14.36），与 CDCD 相当；WMT14 EN-DE BLEU=19.16，略低于 CDCD（19.30）。
- **推理速度**（QQP，单 V100，batch=50，T=2000）：SeqDiffuSeq 89 秒，DiffuSeq 317 秒，**加速 3.56 倍**。

**消融结论**（Table 2，平均 ΔBLEU）：
- 移除自适应噪声调度：-2.29
- 移除自条件：-1.34
- 两者同时移除：-5.71
- 两者互补，自适应噪声调度贡献更大。

## 相关工作脉络
- **DiffusionLM（Li et al., 2022）**：将连续扩散模型用于词嵌入空间进行文本生成，引入辅助损失；本文在其基础上扩展至 seq2seq 设定，并改进架构与训练策略。
- **DiffuSeq（Gong et al., 2022）**：首个将 DiffusionLM 扩展至 seq2seq 的工作，但使用 encoder-only 架构；本文改用 encoder-decoder 并大幅提升推理效率。
- **D3PM / Multinomial Diffusion（Austin et al., 2021; Hoogeboom et al., 2021）**：在离散词元空间直接定义扩散转移，与本文连续 embedding 路径形成对比。
- **Bit-Diffusion（Chen et al., 2022）**：将离散数据编码为二进制位后使用扩散模型，并首次提出 self-conditioning；本文将其引入 seq2seq 文本扩散中。
- **CDCD（Dieleman et al., 2022）**：提出学习的噪声调度用于语言建模和机器翻译；本文的区别在于针对 token 位置设计自适应调度并应用于 seq2seq。
- **Difformer（Gao et al., 2023）**：在 embedding 空间使用扩散模型的 concurrent 工作，与本文方法正交。

## 局限性与未来方向
- **扩散步数过多导致推理成本高**：仍需 T=2000 步的反向迭代，虽然相对 DiffuSeq 已加速 3.56 倍，但与 AR 模型相比仍有数量级差距；可借鉴 DDIM 等加速采样方法减少步数。
- **质量-多样性权衡**：self-conditioning 与自适应噪声调度显著提升单样本质量，但以牺牲多样性为代价，MBR 增益边际（如 QQP 仅 +1.06 BLEU）。
- **CCD 对话生成任务表现不佳**：所有模型在该任务上 BLEU 均低于 2，表明当前扩散方法在开放域对话生成上仍有困难。
- **机器翻译任务落后于强 AR 基线**：在 IWSLT14/WMT14 上始终低于 Transformer AR baseline，需进一步探索Scaling策略。

## 研究启发与可借鉴点
- **Encoder-only vs Encoder-Decoder 架构选择对推理效率的影响**：seq2seq 扩散模型中，encoder 只需计算一次的设计可推广至其他基于扩散的生成任务（如摘要、文本改写），值得系统化评估。
- **Token-level 自适应噪声调度的可迁移性**：按位置独立学习噪声 schedule 的思想可推广至图像生成（按像素/patch 位置差异化调度）及其他序列生成场景。
- **自条件与 MBR 的联合分析**：自条件提升单样本质量但降低多样性，导致 MBR 收益有限——这提示后续工作可在"如何利用采样多样性补偿单样本质量"方向深入，例如 prior sampling 策略（Appendix E）。
- **Prior sampling 替代 denoising distribution 的启发**：以高方差 prior 分布替换部分时间步的去噪分布，是一种简单有效的多样性增强手段，可作为 diffusion-based 文本生成的通用技巧。
- **与 LLM 结合的创新机会**：本文使用 encoder-decoder Transformer，与当前主流 LLM（如 FLAN-T5 系列）架构一致，可将 diffusion 解码范式直接应用于指令微调模型，探索"diffusion-based instruction following"等新方向。

## 关键术语表
- **Diffusion Model**：通过定义前向加噪过程和反向去噪学习过程来生成数据的概率生成模型，核心思想是逐步将数据扰动为噪声再从噪声恢复。
- **Self-Conditioning**：将去噪网络在前一时间步的预测结果作为额外输入拼接，使网络能够基于已有信息逐步精炼而非从零生成。
- **Adaptive Noise Schedule**：根据训练中各 token 位置的实时去噪损失动态调整噪声注入速率的调度策略，使不同位置、不同时间步的去噪难度均衡。
- **MBR（Minimum Bayes Risk）Decoding**：通过生成多个候选序列并从风险最小化角度选择最优输出的解码策略，常用于提升扩散模型生成质量。
- **Dist-1（Distinct Unigram）**：衡量生成文本词汇多样性的指标，计算生成句中不重复词元的比例，越高表示重复越少。
- **Seq2Seq（Sequence-to-Sequence）**：将输入序列映射为输出序列的任务范式，涵盖机器翻译、文本简化、问答生成等多种 NLP 任务。
- **Rounding Distribution**：去噪完成后将连续 embedding 映射回离散 token 的最近邻投票过程，确定最终生成序列。
- **Prior Sampling**：在反向生成过程中以高方差的前向先验分布替代去噪分布，用于增加生成多样性。

## 可复现要素
- **数据集**：所有 6 个数据集均公开可用（QQP、Wiki-Auto、Quasar-T、CCD、IWSLT14、WMT14）。
- **代码/权重**：论文未明确声明代码与权重开源情况（论文未提及）。
- **关键超参**：
  - 扩散总步数 $T = 2000$
  - 学习率 $10^{-4}$，warm-up 10000 步，线性衰减
  - 自适应噪声调度更新间隔：每 20000 步；下采样 stride $K = 20$
  - Transformer：6 层 encoder + 6 层 decoder（翻译任务）；6+6（非翻译任务）
  - 隐藏维度：512（翻译）/ 768（非翻译）
  - 最大输入/输出长度：128 / 64
  - Dropout：0.3（翻译）/ 0.1（非翻译）
  - Embedding 维度：128；batch size：1024（WMT14）/ 128（其他）
  - 训练最大步数：1,000,000；每 10,000 步保存 checkpoint
