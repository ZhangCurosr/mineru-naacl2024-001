---
title: "TRUE-UIE: Two Universal Relations Unify Information Extraction Tasks"
source: https://aclanthology.org/2024.naacl-long.103.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:44:06"
---

# 论文速读：TRUE-UIE: Two Universal Relations Unify Information Extraction Tasks

## 一句话总结
提出 TRUE-UIE，仅通过 IS 和 NEXT 两种通用关系即可将 7 类 16 个数据集的信息抽取任务统一为相同的学习目标；在监督、零样本与少样本设定下均取得 SOTA 性能，并显著提升了跨任务与跨语言的知识迁移能力。

## 研究问题与动机
- 现有 UIE 方法（生成式与基于 Token 链接的方法）缺乏统一的学习目标：不同任务使用不同的结构语言或关系定义，导致知识难以在任务间有效共享。
- 已有链接式方法（以 USM 为代表）在不同任务中对相同关系的定义虽形式一致但语义用途不同，且无法自然处理不连续实体、重叠关系及开放信息抽取等复杂场景。
- 传统 span 提取方法多仅关注首尾 token，忽略了 span 内部 token 的序列依赖信息，限制了抽取精度与泛化上限。
- 亟需一种真正统一的范式，使所有 IE 任务共享相同的目标函数与关系集合，同时兼顾复杂结构建模与跨任务零/少样本泛化。

## 核心贡献（创新点）
1. **双关系统一范式**：仅用 IS 与 NEXT 两种通用关系覆盖全部 7 类 IE 任务，区别于 USM 等多关系任务定制方案，实现真正一致的学习目标。
2. **Structure Language Prompt**：将 schema 结构化信息以占位符形式嵌入提示模板，替代各任务独立定义的结构语言，显著提升模型对任务语义的理解与跨任务可迁移性。
3. **Semi-Matrix BiLSTM 序列依赖建模**：显式捕获 span 内部 token 的先后顺序与相互依赖，弥补传统仅依赖首尾边界特征的不足。
4. **类别不平衡损失适配**：将 ZLPR 损失引入链接式抽取任务，有效缓解 IS/NEXT 正负样本分布不均问题，并在 16 个数据集上验证了全场景 SOTA 性能。

## 方法详解
- **整体架构**：输入文本与结构化提示拼接后经由编码器（BERT/RoBERTa/XLM-R）获取隐藏状态，经两层全连接层生成起始/结束边界特征 $h_j^b$ 与 $h_j^e$，分别送入 Semi-Matrix BiLSTM 模块与乘法注意力模块，输出 span 表征与对应关系得分。
- **链接方案（Linking Scheme）**：定义 IS 关系将 span 对齐至提示中的类型占位符（完成类型识别），NEXT 关系将在同一知识实例（如三元组、事件、开放事实）中的相邻 span 串联成路径。各任务提示格式统一为：
  - 关系抽取：`<subject type> <relation type> <object type>`
  - 情感抽取：`aspect <polarity>`
  - 事件抽取：`<event type>: [argument role1, argument role2, ...]`
  - 不连续/嵌套 NER 与开放信息抽取：通过 NEXT 路径解码自动处理重叠与不连续问题，避免对单一角色类型做 IS 映射带来的冲突。
- **序列依赖建模**：将边界特征重复扩展为 $n \times n$ 矩阵，分别用前向/后向 LSTM 编码上/下三角区域，转置相加后仅保留上三角部分得到 $S_{i,j}$，再通过全连接层输出 span 得分 $s^p_{i,j}$，从而显式捕获 span 内 token 的先后顺序与相互依赖。
- **学习目标**：针对 IS 与 NEXT 出现的类别不平衡（NER 中不连续实体占比小导致 NEXT 稀疏），采用 ZLPR 损失函数：
  $L = \sum_{t \in T} \log(1 + \sum_{(i,j) \in t^+} e^{-s^*_{(i,j)}}) + \log(1 + \sum_{(i,j) \in t^-} e^{s^*_{(i,j)}})$
  其中 $t^+$ 与 $t^-$ 分别对应目标类与非目标类样本，$s^*$ 为 span 或 token pair 的边界得分。

## 实验与结果
- **数据集与基线**：覆盖 CoNLL03/04、ACE04/05、Genia、Cadec/CadecD、SciERC、NYT、CASIE、SemEval-14/15/16、SAOKE 共 16 个数据集，7 类 IE 任务。对比基线包括 USM、UTC-IE、UniEX、UIE 及各任务专用模型。预训练使用 $D_{task}$、$D_{dist}$、$D_{ind}$ 的四种组合。
- **监督设定**：TRUE-UIE 在全部 16 个数据集上均达到 SOTA；多任务微调表现稳定（TRUE† 普遍优于 TRUE^u），而 USM 在多任务训练后部分数据集性能下降（如 ACE05-Ent、14-res），印证统一目标的优越性。在 Cadec 与 SAOKE 上分别提升 3.11 与 1.01 F1，体现对不连续实体与跨语言泛化的支持。
- **零样本设定**：在 4 种预训练数据组合下，跨 8 个 NER 数据集 consistently 优于 USM，平均提升 1.7~3.2 个百分点；关系抽取零样本任务中以 374M 参数超越 GPT-3 (175B) 与 DEEPSTRUCT (10B)，CoNLL04 达到 27.13 F1。
- **少样本设定**：在 1/5/10-shot 下，TRUE-UIE 在前 5 个数据集上平均提升 6.29（vs UIE）与 1.17（vs USM）；后 3 个复杂任务相比无预训练初始化版本平均提升 14.46，证明预训练知识可有效迁移至新颖任务。
- **消融实验**：移除序列依赖（Seq Dep）或结构化提示与通用链接方案（SLP & TUR）均导致性能显著下降，验证各组件必要性。

## 相关工作脉络
- **UIE (Lu et al., 2022)**：生成式 UIE 奠基作，使用结构化抽取语言统一任务；本文认为其生成语言随任务变化，学习目标不一致。
- **Meta-UIE (Cong et al., 2023)**：在 UIE 基础上引入元预训练；本文同样指出生成式方法在低资源/跨任务场景下搜索空间大、知识共享受限。
- **USM (Lou et al., 2023)**：当前最强链接式 UIE，使用三种统一操作；本文指出其不同任务对相同箭头定义相同但用途不同，存在学习目标不一致，且无法自然处理不连续/重叠问题。
- **UTC-IE (Yan et al., 2023)** & **UniEX (Ping et al., 2023)**：同样基于 token pair 分类/抽取框架；本文认为它们仍未实现真正统一的 learning objective，关系定义仍随任务分化。
- **TPLinker / 连续 span 提取方法**：传统链接方法多仅关注首尾 token；本文通过 Semi-Matrix BiLSTM 补充了 span 内部序列信息，是方法论上的关键演进。

## 局限性与未来方向
- 结构化提示（SLP）的模板设计仍依赖人工经验，面对海量 schema 或动态新增任务时易产生类型混淆，附录 D 指出需借助 attention mask 策略缓解。
- Semi-Matrix BiLSTM 的计算复杂度随序列长度呈二次方增长，处理超长文档时可能面临显存与效率瓶颈。
- 当前实验主要聚焦英文及中英混合数据集，跨语言泛化的系统性评估仍有待进一步扩展。
- 未来可将该双关系统一范式拓展至更复杂的知识图谱构建、多模态信息抽取，或与大语言模型指令微调相结合以进一步降低提示工程成本。

## 研究启发与可借鉴点
- **“少即是多”的统一关系设计**：仅用 IS（类型匹配）与 NEXT（实例内串联）两种关系即可覆盖 7 类任务，为 UIE 领域提供了极简统一范式的参考，可直接迁移至其他结构化预测任务。
- **Span 内部序列依赖建模**：Semi-Matrix BiLSTM 以较低成本捕获 span 内 token 顺序信息，可作为 span 提取模块的通用增强组件，替换传统仅依赖首尾边界的设定。
- **结构化提示+注意力掩码工程**：将 schema 知识以占位符形式注入 prompt，配合 attention mask
