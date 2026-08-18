---
title: "TRUE-UIE: Two Universal Relations Unify Information Extraction Tasks"
source: https://aclanthology.org/2024.naacl-long.103.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:44:08"
field: "信息抽取与知识获取"
keywords: ["Universal Information Extraction", "Information Extraction", "Token Pair Linking", "Structure Language Prompt", "Zero-shot Transfer", "Named Entity Recognition", "Relation Extraction", "Event Extraction"]
innovations: ["仅用 IS 和 NEXT 两个通用关系统一全部 IE 任务的学习目标，消除任务间知识共享壁垒", "结构语言提示将 schema 结构编码进 prompt，使模型获得一致的语义匹配上下文", "Semi-Matrix BiLSTM 显式建模 Span 内部序列依赖，突破仅关注首尾 token 的局限"]
benchmarks: ["CoNLL03", "ACE04", "ACE05", "CoNLL04", "NYT", "SciERC", "SemEval-14/15/16", "Cadec", "Genia", "CASIE", "SAOKE"]
---

# 论文速读：TRUE-UIE: Two Universal Relations Unify Information Extraction Tasks

## 一句话总结
TRUE-UIE 提出了一种统一信息抽取新范式：仅用 IS（类型对齐）和 NEXT（组内连接）两个通用关系，结合结构语言提示，将 7 类信息抽取任务统一为单一学习目标，在 16 个数据集上取得 SOTA，并展现出强大的跨任务零样本/少样本迁移能力。

## 研究问题与动机
- **现有 UIE 方法的共性缺陷**：当前生成式方法（如 UIE）和链接式方法（如 USM、UTC-IE、UniEX）均未能实现"统一学习目标"，不同任务间知识共享受限。例如 USM 中同一条虚线黄色箭头在 NER 和 RE 任务中定义相同但目的不同，绿色/蓝色箭头仅用于 RE 而不参与 NER 训练。
- **复杂 IE 任务的处理瓶颈**：现有方法难以有效处理不连续实体识别（discontinuous NER）和开放信息抽取（OIE），这些问题源于 Span 序列依赖被忽略以及重叠关系的建模困难。
- **知识迁移效果不足**：不同任务若采用不同的结构语言或关系集合，会导致模型学到的表示不一致，制约跨领域/跨任务的零样本和少样本泛化。

## 核心贡献（创新点）
- **仅用两个通用关系统一所有 IE 任务**：IS 负责 Span 与提示中类型占位符的对齐（完成类型识别），NEXT 负责同一知识实例内相邻 Span 的连接（完成元素分组），与 USM 等使用 3+ 种任务特定关系的本质区别在于"单一统一学习目标"。
- **结构语言提示（Structure Language Prompt）**：将 schema 的结构化信息编码进提示（如 RE 任务中 `<subject type> <relation type> <object type>` 格式），而非单独枚举实体和关系类型，使模型获得一致的语义匹配上下文，区别于 UIE 中针对不同任务生成不同结构语言的做法。
- **显式建模 Span 序列依赖**：通过 Semi-Matrix BiLSTM 捕获 Span 内部所有 token 的先后依赖信息（以往模型仅关注首尾 token），显著提升 span 特征表达能力，尤其利于处理不连续实体和重叠关系。
- **跨任务/跨语言强迁移**：实验证明 TRUE-UIE 在零样本 NER（8 个数据集上 consistently 超过 USM）和零样本 RE（超越 GPT-3 175B、DEEPSTRUCT 10B 等大规模基线）、少样本场景（平均提升 6.29 F1）均具有优势；且在英语多任务微调后对中文 SAOKE 数据集仍有 +0.6 提升。

## 方法详解
- **架构总览**：输入文本与结构语言提示拼接后输入编码器（RoBERTa-large / XLM-RoBERTa-large），经两个全连接层得到起始（$h^b$）和结束（$h^e$）两种表示，再分别送入 Semi-Matrix BiLSTM 模块和乘法注意力模块。
- **Semi-Matrix BiLSTM（序列依赖建模）**：对矩阵 $B$（下三角）和 $E$（上三角）分别做反向/正向 LSTM，得到 $B'$ 和 $E'$ 后相加得 $S$，其中 $S_{i,j}$ 包含从 $i$ 到 $j$ 和从 $j$ 到 $i$ 的双向序列信息；Span 分数 $s_{i,j}^p = W_s \cdot S_{i,j} + b_s$。
- **关系解码**：IS/NEXT 关系仅在 Span 的首尾 token 对均满足该关系时才成立，通过乘法注意力融合 $s_{i,j}^* = h_i^* \cdot h_j^{*T}$ 得到关系分数。
- **各类任务的统一解码逻辑**：
  - **RE**：prompt 格式 `<subj type> <rel type> <obj type>`，NEXT 连接 subject 和 object Span，各自通过 IS 连接到 prompt 中的类型占位符，同时确定类型与关系三元组。
  - **Sentiment**：prompt 格式 `aspect <polarity>`，逻辑同 RE。
  - **Event Extraction**：prompt 格式 `<event type>: [arg role1, arg role2, ...]`，trigger 也视为 argument；NEXT 连接所有 argument Span 形成连续路径，解决 argument 重叠问题。
  - **Nested/Discontinuous NER**：通过 NEXT 寻找 Span 内从边界到边界的连续短 Span 路径，存在则为不连续实体，否则为连续实体；嵌套通过无连接路径的包含关系处理。
  - **Open IE**：以 role pair 形式识别角色，避免单角色链接导致的重叠问题；所有 begin-to-end 路径均输出为 fact 实例。
- **类别不平衡损失**：IS 关系远多于 NEXT 关系，采用 class imbalance loss（借鉴 Su et al., 2022）：$L = \sum_{t \in T} \log(1 + \sum_{(i,j) \in t^+} e^{-s_{(i,j)}^*}) + \log(1 + \sum_{(i,j) \in t^-} e^{s_{(i,j)}^*})$。
- **Attention Mask 策略**（附录 D）：针对存在大量重复类型（如 "people" 在多个 triplet scheme 中重复出现）的数据集，采用 attention mask 让每个占位符只关注其对应的 schema group 内文本，减少混淆。

## 实验与结果
- **数据集**：7 类 IE 任务、16 个公开基准数据集，包括 CoNLL03、ACE04/05、CoNLL04、NYT、SciERC、SemEval-14/15/16、Cadec/CadecD、Genia、CASIE、SAOKE。
- **预训练数据**：$D_{task}$（Ontonotes，60K）、$D_{dist}$（REBEL distant supervision，356K→300K）、$D_{ind}$（MRQA 类 comprehension，195K）。
- **主要结果**（SOTA 对比）：
  - **NER**：CoNLL03 TRUE† 94.13 > USM† 93.16；ACE04 TRUE† 89.91 > USM† 89.34；ACE05-Ent TRUE† 90.10 > USM† 87.14；Genia TRUE† 82.56 > P-NER 81.77。
  - **RE**：NYT TRUE† 94.83 > USM† 94.07；ACE05-Rel TRUE† 70.84 > PURE 69.40；CoNLL04 TRUE† 78.94 > USM† 75.86；SciERC TRUE† 38.85 > PFN 38.40。
  - **Event**：ACE05-EvtT TRUE† 76.42 > QE 73.60；ACE05-EvtA TRUE† 56.81 > QE 55.10；CASIE_T TRUE† 73.02 > Txt2Evt 68.98。
  - **Sentiment**：14-res TRUE† 78.13 > GAS 72.16；15-res TRUE† 70.78 > Sp-ASTE 68.58；16-res TRUE† 78.83 > Sp-ASTE 70.26；SAOKE TRUE† 47.11 > DragonIE 46.10。
  - **Discontinuous**：CadecD TRUE† 47.51 > Mac 44.40；SAOKE TRUE† 47.11。
- **多任务训练稳定性**：USM† 相比 USM^u 在某些数据集下降（知识冲突），而 TRUE† 相比 TRUE^u 稳定增长。
- **零样本 RE**：TRUE-UIE 374M 参数量在 CoNLL04 上达 27.13，超越 GPT-3（175B，18.10）、DEEPSTRUCT（10B，25.80）和 USM（356M，25.95）。
- **零样本 NER**：四种预训练配置下均稳定超越 USM，在 $D_{task,ind,dist}$ 下平均提升 +3.2 F1。
- **少样本**：前 5 个任务平均提升 6.29 F1（vs UIE/USM），后 3 个复杂任务平均提升 14.46 F1（vs TRUE*）。
- **跨语言**：英语多任务微调后对中文 SAOKE +0.6 F1。

## 相关工作脉络
- **UIE (Lu et al., 2022)**：生成式 UIE 先驱，通过 Structured Extraction Language 统一任务，但不同任务生成不同结构语言，学习目标不一致；TRUE-UIE 定位为统一学习目标的范式升级。
- **USM (Lou et al., 2023)**：当前最强链接式 UIE，使用 3 种 token linking 操作；但其关系在不同任务中目的不同（如虚线/实线箭头），且 green/blue 关系仅在 RE 中训练；TRUE-UIE 从根本上消除任务特定关系设计。
- **UTC-IE (Yan et al., 2023)**：基于 token pair 分类的统一框架；仍依赖 start/start、end/end 等多类配对关系，学习目标不够统一。
- **UniEX (Ping et al., 2023)**：统一提取框架，将任务分解为 span detection + classification + association；但仍无法完全统一所有任务的学习目标。
- **Tplinker / Discontinuous NER (Wang et al., 2020, 2021b)**：早期 token pair linking 工作；TRUE-UIE 继承链接思想但引入序列依赖建模和两关系统一框架。
- **REBEL (Huguet Cabot & Navigli, 2021)**：生成式 RE；TRUE-UIE 的开放抽取部分与其有概念联系但目标更为统一。

## 局限性与未来方向
- **重复类型混淆**：当同一类型词（如 "people"）在多个 schema group 中出现时，即使引入 attention mask 策略仍可能存在表示混淆（论文附录 D 自述）。
- **结构语言提示的工程负担**：需要为不同任务手动构造 schema 提示，在动态/未知 schema 场景下适应性有待验证。
- **未来方向**：可扩展至更多复杂 IE 子任务（如 coreference resolution）；探索自动 prompt 生成以减轻 schema 设计负担；与 LLM 指令微调结合（如 InstructUIE 的思路）进一步提升零样本能力。

## 研究启发与可借鉴点
- **"两关系统一"范式**：用最少的通用关系（IS + NEXT）实现最大覆盖，可作为后续统一抽取框架的设计模板；对团队研究方向——任何需要多任务统一的抽取/结构化预测任务均可借鉴此思路。
- **Semi-Matrix BiLSTM 序列依赖建模**：仅增加少量计算成本即可显著提升 span 特征质量，尤其适合不连续 Span 和多 Span 路径建模，可直接迁移到 NER、COREF、事件链抽取等任务。
- **类别不平衡 loss 在链接式框架中的应用**：IS 关系天然多于 NEXT，采用 class imbalance loss 平衡学习信号，对其他 link-based 模型有参考价值。
- **跨语言知识迁移验证设计**：英语多任务预训练后评估中文低资源任务（SAOKE），为多语言 UIE 提供了简洁有效的评估范式。
- **Attention Mask 解决重复类型混淆**：在共享类型词的 prompt 中通过 mask 隔离不同 schema group，是一种轻量且有效的解混淆技巧。

## 关键术语表
- **IS 关系**：将文本中的 Span 与结构提示中的类型占位符对齐，实现类型识别（entity type / relation type / role type 判定）。
- **NEXT 关系**：将同一知识实例内相邻的两个 Span 连接起来，实现元素分组（构成 triplet、event argument chain、discontinuous entity 等）。
- **Structure Language Prompt（结构语言提示）**：将任务 schema 的结构化信息编码为自然语言风格提示（如 `<subj> <rel> <obj>`），替代传统单独枚举实体/关系类型的做法。
- **Semi-Matrix BiLSTM**：对矩阵的上三角/下三角区域分别施加单向 LSTM，融合后得到 Span 内部的完整序列依赖表示。
- **Class Imbalance Loss**：针对 IS 关系远多于 NEXT 关系的类别不平衡问题，采用 log-sum-exp 形式的损失函数（借鉴 ZLPR）。
- **Discontinuous NER**：识别由多个非连续子 Span 组成的实体（如 "ankle and thigh pain"）。
- **Open Information Extraction (OpenIE)**：无需固定 schema，从文本中抽取任意三元组/事实的开放域抽取任务。
- **Distant Supervision（远端监督）**：利用知识库（Wikidata/Freebase）与文本的自动对齐产生训练信号，无需人工标注。

## 可复现要素
- **数据集**：16 个公开数据集（CoNLL03, ACE04/05, CoNLL04, NYT, SciERC, SemEval-14/15/16, Cadec/CadecD, Genia, CASIE, SAOKE）均为公开研究用途数据集。
- **代码/权重**：论文未明确声明开源，但提及使用 ChatGPT/Copilot 辅助工具函数编写。
- **关键超参**：预训练学习率 $2 \times 10^{-5}$，batch size 96，5 epochs；微调学习率 $\{1,2,3,4,5\} \times 10^{-5}$，batch size 从 8~96 中选取；优化器 Adam；英文用 RoBERTa-large，中文 SAOKE 用 XLM-RoBERTa-large。
- **预训练三源数据**：$D_{task}$（Ontonotes，60K）、$D_{dist}$（REBEL distant supervision，300K）、$D_{ind}$（MRQA 类 comprehension，195K）。
