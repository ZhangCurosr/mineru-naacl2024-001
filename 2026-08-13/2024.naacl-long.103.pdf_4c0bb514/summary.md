---
title: "TRUE-UIE: Two Universal Relations Unify Information Extraction Tasks"
source: https://aclanthology.org/2024.naacl-long.103.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:44:07"
field: "信息抽取"
keywords: ["Universal Information Extraction", "Information Extraction", "Token Pair Linking", "Structure Language Prompt", "Named Entity Recognition", "Relation Extraction"]
innovations: ["仅用IS和NEXT两种通用关系统一所有IE任务的学习目标", "引入结构化语言提示与Semi-Matrix BiLSTM显式建模span序列依赖"]
benchmarks: ["CoNLL03", "ACE04", "ACE05", "CoNLL04", "NYT", "SciERC", "Cadec", "SemEval-14/15/16", "CASIE", "SAOKE"]
---

# 论文速读：TRUE-UIE: Two Universal Relations Unify Information Extraction Tasks

## 一句话总结
本文提出 TRUE-UIE 框架，仅使用 IS 和 NEXT 两种通用关系将所有信息抽取任务统一为同一学习目标，在16个数据集、7类IE任务上达到SOTA，并在零样本/少样本场景展现出更强的跨任务迁移能力。

## 研究问题与动机
1. **学习目标不一致**：现有生成式UIE方法为不同任务学习不同的结构语言（如NER不用嵌套括号，关系/事件抽取使用不同层级嵌套），导致知识难以共享。
2. **linking-based方法关系复用混乱**：以USM为代表的token-pair linking方法中，相同定义的关系在不同任务中承担不同语义（如NER与RE中黄色箭头含义不同），且部分关系仅在单一任务中训练，阻碍跨任务迁移。
3. **复杂任务建模困难**：不连续NER、重叠关系抽取、开放信息抽取等任务在现有主流UIE框架中难以有效处理。
4. **Span序列依赖被忽视**：既有模型仅关注span的起止token，忽略了span内部token的序列依赖性，限制了抽取质量。

## 核心贡献（创新点）
1. **提出仅含IS和NEXT两种通用关系的统一抽取范式**，所有IE任务均映射为相同的关系抽取目标，实现真正的学习目标统一。
2. **设计结构化语言提示（Structure Language Prompt）**，将schema结构先验融入提示文本并在文本中预留IS关系的占位符，简化模型学习任务并提升泛化。
3. **引入Semi-Matrix BiLSTM显式建模span内部序列依赖**，通过上/下三角区域的正向/反向LSTM编码span内token间的顺序信息，弥补已有方法的缺陷。
4. **提出基于路径解码的不连续实体与重叠关系处理机制**，利用NEXT关系构建连续路径来识别不连续实体，避免重复实例输出。
5. **采用class imbalance loss缓解IS/NEXT正负样本不均衡**，尤其针对NER任务中NEXT关系稀疏的问题进行优化。

## 方法详解
1. **整体架构**：输入文本与结构化提示拼接后经过预训练语言模型编码器，提取的hidden states经两个全连接层分别得到span起始/终止表征，再送入Semi-Matrix BiLSTM模块与Multiplicative Attention模块，最终输出span得分与关系得分。
2. **Linking Scheme（统一关系定义）**：
   - **关系抽取**：提示格式为 `<subject type> <relation type> <object type>`，两个span分别通过IS关系链接到主题/客体类型，并通过NEXT关系连接则构成一个三元组，实体类型与关系类型同步确定。
   - **情感抽取**：提示格式为 `aspect <polarity>`，通过IS与NEXT联合构建情感三元组。
   - **事件抽取**：提示格式为 `<event type>: [argument role1, argument role2, ...]`，trigger也被视为argument；所有连接到argument role的span按 preceding event type 分组，仅输出由NEXT关系串联的begin-to-end路径作为事件实例，天然解决argument重叠问题。
   - **嵌套/不连续NER**：检查连接至实体类型的span内是否存在由NEXT连接的连续子路径，若存在则输出不连续实体并忽略长span；若短span被长span包含但无路径连接，则两者均作为实体识别，从而统一处理嵌套与不连续。
   - **开放信息抽取**：成对识别角色（`<role1> <role2>`），避免单角色IS链接带来的重叠；所有begin-to-end的NEXT路径均输出为事实实例。
3. **Span特征建模**：
   - 使用BERT/RoBERTa得到token隐藏表示 $h_j$，经全连接层得到起始 $h_j^b$ 与终止 $h_j^e$ 表征。
   - 将 $h^b$ 和 $h^e$ 分别重复n次构造 $n \times n$ 的矩阵B和E，对E的上三角区域使用Forward LSTM、对B的下三角区域使用Backward LSTM，得到 $B'$ 和 $E'$。
   - 将 $B'$ 转置后与 $E'$ 相加，仅保留上三角区域得到 $S_{i,j}$，该位置蕴含token i到j双向的序列依赖信息；span得分 $s_{i,j}^p = W_s \cdot S_{i,j} + b_s$。
4. **关系预测**：当且仅当span的起始token对与终止token对均满足某关系时判定该关系存在；通过乘法注意力 $s_{i,j}^* = h_i^* \cdot h_j^{*T}$ 融合边界特征并计算关系得分。
5. **损失函数**：针对IS/NEXT类别不均衡，采用class imbalance loss：
   $L = \sum_{t \in T} \log(1 + \sum_{(i,j) \in t^+} e^{-s_{(i,j)}^*}) + \log(1 + \sum_{(i,j) \in t^-} e^{s_{(i,j)}^*})$

## 实验与结果
1. **评测设置**：7类IE任务（平铺NER、关系抽取、事件抽取、情感抽取、嵌套NER、不连续NER、开放信息抽取）、16个公开数据集；预训练使用三类语料：$D_{task}$（Ontonotes 60K）、$D_{dist}$（ distant supervision 356K）、$D_{ind}$（MRQA等阅读理解数据195K）。
2. **主要结果**：
   - **监督设置**：TRUE-UIE在所有16个数据集上均达到SOTA，优于UIE、UniEX、UTC-IE及USM。例如ACE05-EvtT触发词F1达76.42，CoNLL04关系抽取F1达78.94，Cadec不连续实体F1达73.83。
   - **多任务训练稳定性**：USM在多任务微调后部分数据集性能下降（USM† vs USMᵘ），而TRUE-UIE在多任务设定下保持稳定增长（TRUE† vs TRUEᵘ）。
   - **零样本迁移**：在8个未见NER数据集上，TRUE-UIE在四种预训练配置下均优于USM，平均提升1.8~3.2个百分点；在Conll04关系抽取零样本设置中，TRUE-UIE（374M）F1=27.13，超过GPT-3（175B, 18.10）和DEEPSTRUCT（10B, 25.80）。
   - **少样本设置**：TRUE-UIE在7类任务的1/5/10-shot实验中平均超越UIE和USM，前五个数据集平均提升6.29和1.17；在新增的不连续NER与开放IE任务上较无预训练的初始模型平均提升14.46。
3. **跨语言泛化**：在英语多任务微调后，TRUE-UIE在中文SAOKE数据集上提升0.6 F1，展现跨语言迁移潜力。
4. **消融实验**：去除Token序列依赖（Seq Dep）在所有四个评测任务上均导致显著下降；去除结构化提示与统一关系（SLP & TUR）在关系抽取和事件抽取上下降最明显，验证了二者对统一学习目标的贡献。

## 相关工作脉络
1. **UIE (Lu et al., 2022)**：生成式UIE先驱，通过Structured Extraction Language统一各任务，但不同任务生成不同层级的嵌套结构，学习目标不一致，知识共享受限。
2. **USM (Lou et al., 2023)**：当前最强linking-based UIE，使用三种token linking操作统一建模，但相同关系在不同任务中语义不同，部分关系仅在单任务训练，跨任务知识流动受阻。
3. **UTC-IE (Yan et al., 2023)**：将IE任务分解为token pair分类，使用起止token定位span并通过start-start/end-end对建立关系，但仍为任务定制关系定义，未实现真正统一。
4. **UniEX (Ping et al., 2023)**：通过统一提取框架将任务分解为联合span检测、分类与关联问题，但各任务仍需独立学习关联规则，知识复用程度有限。
5. **GAP（Wang et al., 2021b）/ TPLinker（Wang et al., 2020）**：不连续NER与token pair linking的基础工作，本文在其思路上进一步提出统一关系设计以支持更复杂场景。
6. **InstructUIE (Wang et al., 2023)**：基于instruction tuning的多任务UIE方法，侧重指令驱动而非关系统一，与本文的思路形成对比。

## 局限性与未来方向
1. **结构化提示可能引入噪声**：在大量使用默认或粗粒度实体类型的数据集上，提示设计可能导致性能下降（论文附件D提到可通过attention mask缓解）。
2. **预训练依赖较强**：少样本和零样本表现高度依赖预训练阶段的知识积累，对完全无标注领域可能存在瓶颈。
3. **关系数量虽少但编码复杂度高**：Semi-Matrix BiLSTM引入的计算开销较大，推理效率有待优化。
4. **未来方向**：可扩展至多模态IE、探索更轻量的span序列建模方案、研究动态提示生成机制以适配开放域schema变化。

## 研究启发与可借鉴点
1. **"极少关系统一建模"的思路**：用IS和NEXT两种关系覆盖全部IE任务的范式极具启发性，可迁移至其他结构化预测任务（如事件链抽取、知识图谱构建）。
2. **结构化提示+占位符的设计**：将schema结构信息先验化并预留关系占位符，既保留任务先验又降低模型学习负担，值得在其他序列标注任务中尝试。
3. **Span内序列依赖的显式建模**：Semi-Matrix BiLSTM的设计有效弥补了仅依赖起止token的缺陷，该技巧可直接复用于span classification、coreference resolution等任务。
4. **路径解码处理重叠/不连续结构**：通过NEXT关系构建begin-to-end路径的输出策略，为复杂重叠结构提供了干净的处理方案，可扩展至事件论元重叠、嵌套关系等场景。
5. **跨语言迁移验证的合理性**：在英语多任务预训练后评估中文任务的表现，为UIE的跨语言泛化提供了有说服力的实验证据，可作为后续多语言IE研究的参考设计。

## 关键术语表
**TRUE-UIE**：本文提出的通用信息抽取框架，通过两种通用关系统一所有IE任务。
**IS关系**：将文本span与其在结构化提示中对应的类型占位符进行对齐的通用关系。
**NEXT关系**：在同一知识实例（如三元组、事件）内部连接连续span的通用关系。
**结构化语言提示（Structure Language Prompt）**：将schema的结构信息编码为自然语言形式并嵌入输入的提示机制。
**Semi-Matrix BiLSTM**：分别对特征矩阵的上三角和下三角区域进行正向/反向LSTM编码，以捕获span内部token间双向序列依赖的模块。
**Class Imbalance Loss**：针对IS/NEXT关系正负样本不均衡而采用的加权损失函数。
**Unseen Entity Label**：零样本评测中未在预训练数据中出现过的实体类型。
**Open Information Extraction**：无需预定义schema，从文本中自由抽取任意三元组/事实的开放域信息抽取任务。

## 可复现要素
- **数据集**：16个公开benchmark（CoNLL03、ACE04/05、CoNLL04、NYT、SciERC、Cadec/CadecD、SemEval-14/15/16、CASIE、SAOKE等），均为公开发布的研究数据集。
- **代码/权重**：论文未明确声明代码与模型权重是否开源。
- **关键超参**：预训练学习率2×10⁻⁵，batch size 96，5 epochs；fine-tuning学习率{1e-5, 2e-5, 3e-5, 4e-5, 5e-5}，batch size从{8,12,16,32,64,96}中选取；优化器Adam； backbone为RoBERTa-large（英文）或XLM-RoBERTa-large（含中文任务）；预训练三阶段语料：$D_{task}$（Ontonotes 60K）、$D_{dist}$（356K distant supervision）、$D_{ind}$（195K MRQA类 comprehension数据）。
