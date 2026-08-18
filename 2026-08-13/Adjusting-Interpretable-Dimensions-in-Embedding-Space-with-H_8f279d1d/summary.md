---
title: "Adjusting-Interpretable-Dimensions-in-Embedding-Space-with-H"
source: https://aclanthology.org/2024.naacl-long.146.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:31:37"
---

# 论文速读：Adjusting-Interpretable-Dimensions-in-Embedding-Space-with-H

## 一句话总结
本文提出将种子词先验与人类属性评分相结合的训练框架（FIT+S），在预训练词嵌入空间中拟合出高质量的可解释维度，显著修复了传统种子方法在方向偏差与尺度膨胀方面的缺陷，尤其在种子基线表现薄弱的物体属性预测任务上提升明显。

## 研究问题与动机
- 现有可解释维度多依赖人工挑选的反义词种子对计算差向量均值，种子选择具有主观性，常出现“坏种子（bad seeds）”导致维度方向不稳定或与目标属性耦合不足。
- 纯粹依赖种子方向的投影预测值往往远超人类评分的实际量纲（MSE 极高），难以直接用于定量比较或跨模态对齐。
- 纯数据驱动的拟合方法（仅用人类评分优化方向）在高维嵌入空间中自由度过大，极易过拟合甚至匹配噪声配对。
- 动机：利用认知科学中可靠的逐词人类评分数据，与种子词/种子方向先验联合约束，获得既符合语义直觉又严格对齐真实评分尺度的可解释维度。

## 核心贡献（创新点）
- 提出 FIT 系列模型，将人类评分作为监督信号直接在固定嵌入空间中优化可解释维度方向，突破传统仅靠种子差向量平均的构造范式。
- 设计 FIT+S 统一框架，同时引入种子词合成极端评分（位置约束）与种子维度余弦距离正则（方向约束），有效兼顾方向准确性与尺度校准。
- 提出扩展成对排名准确率（extended pairwise rank accuracy, $r^+$-acc），同步度量测试集内排序一致性与测试词相对于训练词的排序一致性，缓解小样本 CV 下的指标稀疏问题。
- 实证验证了“种子失效场景正是拟合方法增益最大”的互补规律，并证明 FIT+S 的预测值在 97% 的运行中误差 MSE < 2，实现与金标准同尺度输出。

## 方法详解
- **SEED 基线**：选定正负种子词对 $\{(p_k, n_k)\}$，计算差向量 $\vec{p_k} - \vec{n_k}$ 后取平均得维度 $\vec{d}$，词 $a$ 得分由投影 $|\mathrm{proj}_{\vec{a}}(\vec{d})| = \frac{\vec{a} \cdot \vec{d}}{||\vec{d}||}$ 给出。
- **FIT（纯评分拟合）**：给定标注集 $\{(w_i, \hat{y}_i)\}$，优化方向 $\vec{f}$ 最小化损失 $J_f = \sum_i (\vec{w_i} \cdot \vec{f} - c_f \hat{y}_i - b_f)^2$，允许投影线性映射到原始评分尺度。
- **FIT+SW（种子词引导）**：将种子词追加至训练集，并为正/负种子赋予合成极端值 $\max(\hat{Y})+\delta$ / $\min(\hat{Y})-\delta$，附加 $[0.001, 0.005]$ 随机扰动避免共线，扩展 $J_f$ 的求和范围。
- **FIT+SD（种子维度引导）**：在 $J_f$ 基础上加入方向对齐项 $J_d(D) = \sum_{d \in D} 1 - \cos(\vec{d}, \vec{f})$，总损失 $J = \alpha J_f + (1-\alpha) J_d(D)$，$\alpha$ 控制评分拟合与方向保持的权衡。
- **FIT+S（双约束融合）**：同时启用 FIT+SW 与 FIT+SD 策略，联合优化位置极端性、方向先验与评分拟合误差。开发集调优得 GLoVE 下 $\alpha=0.05$。
- **上下文聚合策略**：对 BERT/RoBERTa，采集每词最多 10 句（≤100 tokens）的 top-4 层表示，按词 piece 平均后再跨句平均得到 type-level 向量。

## 实验与结果
- **数据集**：Grand et al. (2022) 物体属性（9 类别 × 17 特征，如 DANGER、SIZE、WEALTH）；Pavlick & Nenkova (2015) 风格属性（COMPLEXITY 1160 词、FORMALITY 1274 词）。
- **基线**：SEED、词频对数基线（FREQ）、随机基线（RAND）、FIT/FIT+SW/FIT+SD/FIT+S。
- **主要数值**：
  - 物体属性（GLoVE）：FIT+S 的 $r^+$-acc 达 **0.80**（SEED 0.64），MSE 仅 **0.7**（SEED >1000）；BERT/RoBERTa 下 FIT+S 同样最优（0.71/0.69 vs 0.64/0.57）。
  - 风格属性：SEED 本身表现较强，FIT+S 排名提升有限，但 MSE 显著下降（Complexity GLoVE: 1.2 vs 31.5；Formality: 1.6 vs 60.5）。
  - 增益分布：SEED 最差 20% 的条件上平均提升 **27.3** 个百分点，最优 20% 仅提升 **1.7** 个百分点；FIT 与 FIT+SW 普遍低于频率基线或随机水平，证实单一信息源不足以稳定约束高维方向。
  - 尺度一致性：FIT+S 在 97% 的实验运行中 MSE < 2，所有运行 MSE < 10，预测值稳定落在人类评分的 [-
