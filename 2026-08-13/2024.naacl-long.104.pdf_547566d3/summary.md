---
title: "zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models"
source: https://aclanthology.org/2024.naacl-long.104.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:44:51"
---

# 论文速读：zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models

## 一句话总结
本文提出 zrLLM，首次将大语言模型（LLM）的语义知识注入时序知识图谱预测（TKGF），通过生成增强关系描述（ERD）与冻结式文本-图嵌入对齐，结合关系历史学习器（RHL）捕获动态演化模式，显著提升了嵌入型模型在零样本未见关系上的预测能力，同时保持了对已有关系的预测性能。

## 研究问题与动机
- **核心问题**：现有嵌入型 TKGF 模型完全依赖训练集图共现上下文学习关系表示，当推理时出现训练集中从未存在的零样本（zero-shot）关系时，因缺乏任何历史图证据而无法生成合理表征，导致链接预测性能骤降。
- **传统嵌入法局限**：TANGO、CyGNet、RE-GCN 等仅通过历史时刻的图结构归纳关系，对未见关系泛化能力为零；规则基方法（TLogic、ALRE-IR）虽能处理未见实体，但规则绑定已观测关系，同样无法突破零样本关系瓶颈。
- **少样本/归纳学习局限**：OAT、MOST、FILT 等方法需 K（通常为 1 或 3）条支撑事实才能初始化表示，在纯零样本设定下无法启动。
- **LLM 直接应用的隐患**：纯 ICL 未对齐嵌入空间，性能不及传统方法；全量微调（如 GenTKG）算力成本极高；且传统 benchmark（ICEWS14/18）事实截止 2019 年，与 T5-11B 等预训练语料存在严重信息泄露风险。

## 核心贡献（创新点）
1. **首次系统研究 TKGF 中的零样本关系学习并构建防泄露基准**：提出 ACLED-zero、ICEWS21-zero、ICEWS22-zero 三个时间戳均晚于 T5-11B 发布的新数据集，彻底隔离预训练语料污染，为 LLM+KG 评测提供公平协议。
2. **LLM 语义表征的冻结对齐机制**：利用 GPT-3.5 扩写短关系标签生成 ERD，经 T5-11B 编码后通过 MLP+GRU 映射至 TKGF 嵌入空间，并固定参数不更新；本质区别在于“用外部语言常识弥补图上下文缺失”，而非依赖训练分布内的共现统计。
3. **关系历史学习器（RHL）与隐式历史预测网络（HPN）**：设计 GRU 时序模块捕获实体对间的动态关系模式（如“先建交后签协定”），并通过 HPN 将“需目标实体才能查历史”的硬约束转化为“仅凭查询关系推断历史”的子任务；本质区别在于实现了**测试期未知目标实体下的历史模式泛化**。

## 方法详解
- **ERD 生成与 T5 文本表征**：以数据集原始关系文本为输入，经 Prompt 驱动 GPT-3.5 生成语义扩充描述（ERD）。将 ERD 输入 T5-11B Encoder，得到词级隐藏矩阵 $\bar{\mathbf{H}}_r \in \mathbb{R}^{L \times d_w}$。T5 输出被冻结，防止训练数据反向污染语义。
- **文本到图嵌入空间对齐**：对每个词向量 $\mathbf{w}_l$ 用 MLP 投影至目标维度 $d$，再送入 GRU 按序聚合得到关系最终表示 $\bar{\mathbf{h}}_r$，直接替代原 TKGF 模型中仅靠图上下文学习的关系参数。
- **关系历史学习器（RHL）**：训练时，对每对实体 $(s,o)$ 按时间戳聚合历史关系集，通过注意力加权聚合生成各时刻实体对表示 $\mathbf{h}_{s,o}^{t_i}$，再经 $\mathrm{GRU}_{\mathrm{RHL}}$ 编码出历史序列 $\mathbf{h}_{\mathrm{hist}}$，捕获 entity-agnostic 的时序演变规律。
- **历史预测网络（HPN）**：因测试时目标实体未知，无法直接搜索历史。训练轻量 HPN：$\tilde{\mathbf{h}}_{\mathrm{hist}} = \alpha \mathrm{MLP}_{\mathrm{hist}}(\bar{\mathbf{h}}_r) + \bar{\mathbf{h}}_r$，以 MSE 损失约束其逼近真实历史编码 $\mathbf{h}_{\mathrm{hist}}$，使模型仅凭关系语义即可推断合理历史上下文。
- **双路打分融合与联合训练**：RHL 基于 $\tilde{\mathbf{h}}_{\mathrm{hist}}$ 与 $\bar{\mathbf{h}}_r$ 再次运行 $\mathrm{GRU}_{\mathrm{RHL}}$ 得模式表征 $\mathbf{h}_{\mathrm{pat}}$，结合时间感知实体表示，用 TuckER 风格三模张量积计算 RHL 评分 $\phi$。总分 $\phi_{\mathrm{total}} = \phi' + \gamma \phi$。总损失 $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{TKGF}} + \mathcal{
