---
title: "zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models"
source: https://aclanthology.org/2024.naacl-long.104.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:44:53"
---

# 论文速读：zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models

## 一句话总结
本文首次将大语言模型（LLM）的语义先验引入时序知识图谱预测（TKGF），提出 zrLLM 框架，通过冻结的 T5-11B 编码器将 LLM 生成的扩充关系描述对齐到图嵌入空间，并设计关系历史学习器（RHL）捕捉时序模式，在完全不调微调 LLM 的前提下，显著提升了嵌入类 TKGF 模型对零样本（未见）关系的预测能力，同时保持了对已知关系的高性能。

## 研究问题与动机
- **核心问题**：传统基于嵌入的 TKGF 模型依赖训练图中观测到的图上下文学习关系表示，面对训练集未出现的零样本关系时，因缺乏任何关联事实而无法生成合理表示，导致外推性能骤降。
- **现有方法不足 1（少样本局限）**：现有 TKGF 归纳学习（如 OAT、MOST）多基于 few-shot 设定，需要推理时提供 K 个相关事实才能构建表示，在纯零样本场景下完全失效。
- **现有方法不足 2（信息泄漏风险）**：主流 TKGF 基准（ICEWS14/18/05-15）事实多截止于 2019/2020 年，直接使用预训练语料覆盖 2021 年后的 LLM（如 T5-11B、BERT、Llama2）进行微调或推理存在严重的数据泄漏风险。
- **现有方法不足 3（LLM 直接应用效率低）**：纯上下文学习（ICL）缺乏结构化预测头，性能被传统 TKGF 碾压；微调大型 LLM（如 GenTKG 微调 Llama2-7B）算力开销巨大。
- **动机**：利用 LLM 强大的语义理解能力为关系提供丰富描述，通过轻量投影将其语义先验注入 TKGF 模型，并结合时序历史模式学习，使模型能在无任何图上下文的情况下仅凭语义推断未见关系。

## 核心贡献（创新点）
1. **首次系统研究 TKGF 中的零样本关系学习**：明确刻画嵌入类模型在未见关系上的失效机制，并构建 ACLED-zero、ICEWS21-zero、ICEWS22-zero 三个新型基准，严格切割时间线与关系频率以规避信息泄漏。
2. **提出 LLM-empowered 语义对齐框架**：不微调 LLM，仅用 GPT-3.5 生成扩充关系描述（ERD），再通过冻结的 T5-11B 编码器与 MLP+GRU 投影到 TKGF 关系空间，使语义相近关系在嵌入空间中自动靠拢。
3. **设计关系历史学习器（RHL）与历史预测网络（HPN）**：利用 LLM 赋能表示驱动时序模式捕获，HPN 绕过测试时未知实体对的组合爆炸，直接由关系推断历史表征，显著增强零样本推理。
4. **跨模型泛化验证与高效折中**：将 zrLLM 无缝集成到 7 种主流嵌入类 TKGF 模型，在零样本指标上全面超越基线，并在计算效率与预测精度间取得优于 PPT（微调 BERT）和 ICL+GPT-NeoX-20B 的平衡。

## 方法详解
- **ERD 生成与语义表示提取**：将数据集提供的简短关系文本输入 GPT-3.5 生成富含语义的解释文本（ERD）。将 ERD 送入 T5-11B 编码器，取编码器隐藏状态矩阵 $\bar{\mathbf{H}}_r \in \mathbb{R}^{L \times d_w}$。通过 MLP 将每个词向量映射至 TKGF 隐层维度 $d$，再经 GRU 序列化聚合得到关系表示 $\bar{\mathbf{h}}_r$。关键设计：$\bar{\mathbf{H}}_r$ 权重全程**冻结**，防止训练数据覆盖 LLM 原始语义，确保零样本泛化基础。
- **关系历史聚合（GRU_RHL）**：对训练事实 $(s,r,o,t)$，检索 $t$ 之前 $s$ 与 $o$ 的历史事实并按时间分组。每组内关系表示经注意力加权聚合：$\mathbf{h}_{s,o}^{t_i} = \sum_m \mathrm{softmax}(\bar{\mathbf{h}}_{r_m}^\top \mathrm{MLP}_{agg}(\bar{\mathbf{h}}_r)) \cdot \bar{\mathbf{h}}_{r_m}$。若某时刻无历史事实则填充 dummy embedding。随后用 GRU_RHL 沿时间轴递归编码得到历史表征 $\mathbf{h}_{hist}$。
- **历史预测网络（HPN）**：测试时目标实体未知，无法执行上述检索。训练阶段引入 HPN：$\tilde{\mathbf{h}}_{hist} = \alpha \mathrm{MLP}_{hist}(\bar{\mathbf{h}}_r) + \bar{\mathbf{h}}_r$，通过 MSE 损失 $\mathcal{L}_{hist} = \mathrm{MSE}(\tilde{\mathbf{h}}_{hist}, \mathbf{h}_{hist})$ 约束预测历史贴近真实历史。测试时直接由当前关系 $r$ 生成 $\tilde{\mathbf{h}}_{hist}$。
- **模式融合打分**：将 $\tilde{\mathbf{h}}_{hist}$ 与 $\bar{\mathbf{h}}_r$ 再次输入 GRU_RHL 得到时序模式表示 $\mathbf{h}_{pat}$。参考 TuckER 三线性打分计算 RHL 分数 $\phi((s,r,o,t)) = \mathcal{W} \times_1 \mathbf{h}_{(s,t)} \times_2 \mathbf{h}_{pat} \times_3 \mathbf{h}_{(o,t)}$，最终总分 $\phi_{total} = \phi' + \gamma \phi$，$\phi'$ 为原 TKGF 模型得分。
- **联合损失函数**：$\
