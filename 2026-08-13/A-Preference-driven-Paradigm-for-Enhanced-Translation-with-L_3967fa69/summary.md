---
title: "A-Preference-driven-Paradigm-for-Enhanced-Translation-with-L"
source: https://aclanthology.org/2024.naacl-long.186.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:29:55"
field: "机器翻译"
keywords: ["机器翻译", "大语言模型", "偏好学习", "Plackett-Luce", "SFT", "MAPLE"]
innovations: ["基于Plackett-Luce的偏好学习方法打破SFT性能瓶颈", "引入偏好距离信息增强质量区分能力", "构建并开源MAPLE多质量翻译偏好数据集"]
benchmarks: ["WMT22", "FLORES-200"]
---

# 论文速读：A Preference-driven Paradigm for Enhanced Translation with Large Language Models

## 一句话总结
本文针对LLM通过SFT进行机器翻译时面临的性能瓶颈问题，提出基于Plackett-Luce模型的偏好学习方法，通过引入多质量翻译样本和人工偏好距离信息来增强模型对翻译质量的区分能力，从而有效打破SFT的性能平台期。

## 研究问题与动机
1. **SFT性能瓶颈**：现有LLM通过监督微调（SFT）学习机器翻译时，仅模仿参考译文的token级别分布，无法超越已有翻译能力达到更高水平。
2. **数据噪声敏感**：平行语料中的参考译文存在噪声（标注错误、语言文化差异等），过度依赖单一参考译文会限制模型泛化能力。
3. **数据规模效应递减**：增加平行数据量并不能持续提升翻译质量，Xu et al. (2023)的ALMA研究表明数据质量比数量更重要。
4. **已有方法的局限**：ParroT、TIM等工作的对比翻译存在明显缺陷、易于区分，且仅提供相对排序而非量化质量差异。

## 核心贡献（创新点）
1. **基于Plackett-Luce的偏好学习框架**：提出将目标LLM直接作为奖励模型，通过学习翻译质量偏好来对齐生成概率与人类偏好，本质区别于DPO的KL散度正则化方式。
2. **引入偏好距离信息**：创新性地将人类标注的精确偏好距离融入Plackett-Luce模型，避免单纯使用排名信息导致的好译文被不当抑制。
3. **构建MAPLE数据集**：建立包含4个翻译方向、每个源句5个不同质量翻译及人工评分的机器翻译偏好数据集，数据集公开可用。
4. **突破SFT性能瓶颈**：实验表明该方法可在不增加大量数据的情况下持续提升LLM翻译性能，在WMT22和FLORES-200上取得显著提升。

## 方法详解
**两阶段优化流程：**

1. **监督微调（SFT）阶段**：使用高质量平行数据对目标LLM进行微调，激活其翻译能力。损失函数为：
   $$\mathcal{L}_{SFT}(\pi_\theta) = -\log \pi_\theta(x, y) = -\sum_t \log P_{\pi_\theta}(y_t | y_{1,\cdots,t-1}, \mathcal{T}(x))$$
   其中输入指令模板从31个候选中随机采样，指令token的损失被置零。

2. **偏好学习（PL）阶段**：
   - 基础Plackett-Luce模型建模翻译偏好分布：
     $$p^*(y^1_{\succ_x} \cdots \succ_x y^L | x) = \prod_{i=1}^{L-1} \frac{\exp(r^*(x,y^i))}{\sum_{j=i}^{L}\exp(r^*(x,y^j))}$$
   - 引入偏好距离的改进版本（$\mathcal{L}_{PLD}$）：
     $$\mathcal{L}_{PLD}(\pi_\theta) = -\mathbb{E} \sum_{i=1}^{L-1} \log \frac{\pi_\theta^{d_i^i}(x,y^i)}{\sum_{j=i}^{L}\pi_\theta^{d_i^j}(x,y^j)}$$
     其中$d_i^j = r^*(x,y^i) - r^*(x,y^j)$为偏好距离。
   - 最终损失函数：$\mathcal{L} = \mathcal{L}_{PLD} + \beta \mathcal{L}_{SFT}$，$\beta$平衡两者权重。

**理论依据**：从Bradley-Terry模型的Gumbel分布假设出发，证明引入偏好距离可使最优缩放参数$\gamma = 1/d_{1}^{2}$，从而在二分类情况下精确嵌入距离信息。

## 实验与结果
**数据集与设置：**
- 训练数据：WMT17/18/19共30K平行句子（四个翻译方向）
- MAPLE训练集：每方向1.1K源句，每句5个翻译（1个reference + 4个VicunaMT生成）
- 评估：WMT22和FLORES-200，使用COMET和BLEU指标

**主要结果（COMET分数）：**
| 模型 | WMT22 Avg. | FLORES-200 Avg. |
|------|-----------|-----------------|
| VicunaMT | 81.25 | 85.00 |
| **+VicunaMT+PL** | **83.00** | **86.30** |
| ALMA-7B (LLaMA-2) | 83.59 | - |
| ChatGPT-3.5 | 85.43 | 88.02 |

**关键发现：**
1. 最大提升：en→zh方向COMET提升**3.96分**（81.27→84.26）
2. 优于仅用reference（+REF: 82.07）和best translation（+BEST: 82.06）的SFT变体
3. MAPLE数据（4.4K样本）优于1.4M平行数据的SFT效果
4. 反向选择模式（含最优和最劣翻译）显著优于正向选择
5. 偏好距离信息至关重要：移除后性能从83.00降至82.22

## 相关工作脉络
1. **ParroT (Jiao et al., 2023)**：添加"Hint"字段要求模型生成正确和错误翻译，但人工噪声明显、易区分；本文使用目标模型生成的hard negatives。
2. **TIM (Zeng et al., 2023)**：使用排名损失优化LLM，但仅提供二元排序无质量量化；本文引入精确偏好距离。
3. **SWIE (Chen et al., 2023)**：通过instruction adapter增强长程注意力；本文关注偏好学习而非架构修改。
4. **ALMA (Xu et al., 2023)**：单语数据继续预训练+SFT；本文方法与ALMA正交可结合。
5. **DPO (Rafailov et al., 2023)**：基于Bradley-Terry模型+KL散度正则化；本文使用Plackett-Luce且无reference model约束。

## 局限性与未来方向
1. **低资源语言未验证**：实验仅涵盖中英德等高资源语言，低资源场景适用性待验证。
2. **标注成本较高**：每个样本需5个翻译+双人评分，难以大规模扩展；但数据集可复用降低边际成本。
3. **人工偏好主观性**：存在annotator分歧和快捷策略风险，尽管采用双人评分平均缓解。
4. **Future direction**：可探索迭代式偏好学习实现持续改进。

## 研究启发与可借鉴点
1. **Hard negative挖掘策略**：使用目标模型自身生成的翻译作为对比样本，比人工添加噪声更具学习价值，可迁移至其他LLM对齐任务。
2. **偏好距离建模**：将连续评分转化为距离参数融入排名模型，相比纯排序信号提供更丰富的学习梯度，值得在文本生成评估中探索。
3. **反向选择采样**：优先选取质量差异最大的样本对（最高分vs最低分），可提升偏好学习效率，适用于数据高效训练场景。
4. **数据集复用机制**：MAPLE证明单一偏好数据集可赋能多个目标模型，为跨模型知识迁移提供新范式。

## 关键术语表
**Plackett-Luce模型**：用于建模多项选择偏好的概率模型，通过连乘条件概率表示完整偏好序。
**Preference distance**：基于人工评分计算的翻译质量差异值，用于量化偏好强度的连续度量。
**Hard negative example**：模型高概率生成但人类评分较低的翻译，相比明显错误样本更具区分价值。
**MAPLE数据集**：本文构建的机器翻译偏好学习数据集，包含多质量翻译及人工评分。
**COMET指标**：基于大模型的神经机器翻译评估指标，比BLEU更能 correlate with human judgment。
**Bradley-Terry模型**：二分类偏好建模的经典统计模型，PL模型的一般化形式。

## 可复现要素
- **数据集**：MAPLE已公开（论文声明"release our dataset"）
- **代码/权重**：论文未明确提及开源代码和模型权重
- **关键超参**：学习率5e-6，batch size 96，warmup ratio 0.1，beam size 4，max length 512，β∈{0.0, 0.05, 0.1}
- **硬件**：8×A100-40GB或8×H100-80GB GPUs，DeepSpeed ZeRO-3
