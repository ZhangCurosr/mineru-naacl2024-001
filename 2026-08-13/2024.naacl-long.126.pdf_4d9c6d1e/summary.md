---
title: "The Perspectivist Paradigm Shift: Assumptions and Challenges of Capturing Human Labels"
source: https://aclanthology.org/2024.naacl-long.126.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:28:05"
field: "NLP数据标注与评估"
keywords: ["data labeling", "annotator disagreement", "perspectivist paradigm", "inter-annotator agreement", "ground truth", "dataset documentation", "model calibration"]
innovations: ["系统对比传统多数投票范式与视角主义范式在分歧归因上的根本假设差异", "澄清统计bias与社会bias的混淆并提供保留minority opinion的理论依据", "提出覆盖标注全流程的端到端实践建议框架并引入跨学科理论资源"]
benchmarks: ["No independent benchmarks; position paper synthesizing prior work"]
---

# 论文速读：The Perspectivist Paradigm Shift: Assumptions and Challenges of Capturing Human Labels

## 一句话总结
本文是一篇立场论文，系统审视了机器学习数据标注从传统"多数投票聚合"范式向新兴"视角主义（Perspectivist）"范式的转变，剖析两种范式对分歧成因的根本假设差异，并提出了覆盖数据标注全流程的实践建议与未来研究方向。

## 研究问题与动机
1. 传统标注实践默认每个数据实例存在单一"ground truth"标签，将标注者分歧视为需最小化的噪声；但近年研究表明，捕捉分歧可提升模型性能与校准度、凸显少数群体声音、揭示任务歧义（Fornaciari et al., 2021; Baan et al., 2022; Prabhakaran et al., 2021），引发了"面对分歧应如何处理"的新问题。
2. 传统范式隐含三个有问题的假设：① 将统计意义上的"偏差"（label偏离均值）等同于社会意义上的"偏见"（歧视），从而把所有分歧都视为噪声；② 忽视以"生活经验"为代表的非技术型专业知识，导致有经验的少数群体意见被错误过滤；③ 在无上下文的隔离状态下收集标注，忽略了任务背景信息会显著影响标注判断（如仇恨言论检测中的种族信息、翻译中的详细指令）。
3. 即使接受传统范式"捕获单一真实标签"的目标，实践中仍面临三重技术挑战：众包平台（如MTurk）的标注者群体 demographics 不具代表性；小样本下均值估计远离真实人群均值（sample error）；多数投票非随机丢弃少数意见，导致下游模型校准偏差。
4. 视角主义新实践自身也暴露出新问题：评估仍主要依赖多数投票标签（Plank, 2022指出大多数视角主义论文仍未摆脱此惯例）；人口统计因素只能部分解释分歧，更多非人口因素（社媒使用习惯、语言背景等）未被充分建模；收集边缘群体意见可能对其造成额外负担并涉及隐私权衡；参与式方法面临机构"快速出数据"的制度压力。

## 核心贡献（创新点）
1. **首次系统对比传统与视角主义两种标注范式在分歧归因上的根本假设差异**——传统范式视分歧为噪声（归因于低质量标注者或模糊任务），视角主义范式视分歧为信息源（反映多元观点与合理差异）；本质区别在于对"ground truth"是否存在这一本体论问题的不同回答。
2. **澄清"bias"一词在标注研究中被混淆的双重含义**——统计偏差（估计值与期望值的差异）≠ 社会偏见（对特定群体的歧视）；许多被判定为" biased annotator"的意见实际上反映的是被多数人忽略的边缘群体合理立场；这一澄清为保留 minority opinion 提供了理论基础。
3. **揭示视角主义文献未充分展开的关键维度：非人口因素对分歧的贡献**——论文引用 Fleisig et al. (2023) 发现 hate speech 任务中，社媒使用习惯和对在线毒性内容的态度比性别/教育更能预测分歧；Orlikowski et al. (2023) 发现 demographic 因素单独使用时预测力有限；提示未来研究应拓宽分歧归因的变量空间。
4. **提出覆盖数据标注全流程的系统性建议框架**——涵盖标注前设计、招募策略、标注流程、数据集文档记录、模型设计与评估六个环节，且将规范性决策（如专家驱动 vs 民主投票光谱）显式纳入设计考量；这是目前少有的从端到端视角重构标注流程的综合性建议。
5. **跨学科引入社会选择理论、科技研究（STS）、语用学与参与式设计等理论资源**——将标注问题置于 Arrow 投票理论、Bowker & Star 分类政治学、"common ground"语用学概念等更广学术脉络中，为NLP数据标注研究提供了跨学科的理论工具箱。

## 方法详解
本文为立场论文（position paper），无独立提出的算法或模型，核心内容为概念框架分析与建议 synthesis，以下为主要分析维度与关键论点：

**（1）传统范式假设剖析：**
- **分歧成因归因**：传统实践将分歧归因于三类因素——① "subjective/confusing/ambiguous"任务；② 低质量标注者（inexperienced, biased, spammer）；③ 缺乏任务上下文信息。由此衍生出"收集多份标注→测量IAA（inter-annotator agreement）→取多数投票"的标准pipeline。
- **核心矛盾**：统计 bias（label偏离群体均值）被直接等同于社会 bias（歧视），导致所有 minority opinion 被当作噪声丢弃。

**（2）视角主义范式挑战的三类传统实践：**
- 挑战1：将分歧归因于 bias/ineptitude → 改为将 demographic 与 lived experience 视为合法的信息来源。
- 挑战2：去语境化标注（decontextualized annotation）→ 改为提供任务背景（数据用途、决策后果等），已有研究显示上下文改变标注判断（Sap et al., 2019; Balagopalan et al., 2023）。
- 挑战3：将分歧限于"主观任务"→ 改为认识到分歧存在于NLI（Pavlick & Kwiatkowski, 2019）、STS（Wang et al., 2023）等看似客观的任务中。

**（3）规范性光谱：Democratic ↔ Expert-driven：**
论文提出数据标注决策可置于一个光谱上：一端为"民主式"（所有stakeholder平等参与），另一端为"专家驱动式"（仅征询具备特定知识的群体意见，如医生标注医学文本）。不同任务应选择光谱上不同位置，且这一选择应显式做出而非默认多数投票。

**（4）替代评估指标建议：**
- 训练：用 KL divergence 衡量模型预测与标注者label分布之间的差异；校准到 annotator opinion 分布（Baan et al., 2022）。
- 评估：用 distributional similarity（KL divergence、cosine similarity between output lists、correlation coefficient）、individual annotator accuracy（Davani et al., 2022）、model calibration to population uncertainty 替代单一gold label比较。

## 实验与结果
本文是立场论文，**无独立实验**。论文引用并综合了多篇实证研究的结果，关键数值与发现如下：

- **Fleisig et al. (2023)**：在仇恨言论检测任务上，race 可预测分歧，但 gender 和 education 不可预测；社媒使用习惯和对在线毒性内容的态度是更强的预测因子。
- **Orlikowski et al. (2023)**：单独建模 gender、age、education、sexual orientation 在 hate speech 任务上对分歧的预测效果有限，提示非人口因素更重要。
- **Biester et al. (2022)**：在多任务上未发现 gender 对分歧有显著影响。
- **Prabhakaran et al. (2021)**：聚合label与white annotators的agreement disproportionately高，反映了少数群体意见被系统性丢弃。
- **Baan et al. (2022)**：多数投票导致的 aggregated judgments 使下游模型对不同annotator opinion diversity 的 calibration 出现偏差。
- **Huang et al. (2023）**：>50% 的MTurk标注者希望获得更多标注上下文信息，>2/3 认为了解标注任务目的是有帮助的。
- **最强结论**：即使采用 diverse 招募 + 多数投票聚合，模型仍因 sample error 和 minority opinion 丢弃而无法准确反映目标人群真实 opinion 分布（Narimanzadeh et al., 2023）。

## 相关工作脉络
1. **Snow et al. (2008); Nowak & Rüger (2010)** —— 传统crowdsourcing标注的奠基性工作，建立"多标注者+聚合得ground truth"的经典pipeline，本文将其定义为"longstanding paradigm"的代表。
2. **Basile et al. (2021); Plank (2022)** —— "perspectivist turn"的命名与理论化源头，提出应将annotator disagreement视为信息源而非噪声，本文在此学术脉络中系统扩展。
3. **Geva et al. (2019)** —— 发现annotator-specific模型可提升下游任务性能，揭示NLI和QA任务中存在显著的个体标注者效应，挑战"单一ground truth"假设。
4. **Fleisig et al. (2023)** —— 实证研究hate speech标注中分歧的人口与非人口预测因子，本文引用其发现以论证"demographics-only"解释力的局限。
5. **Baan et al. (2022)** —— 提出在annotator disagreement存在时评估calibration的方法，本文引用其作为perspectivist训练/评估策略的代表。
6. **Davani et al. (2022); Gordon et al. (2022)** —— 直接建模个体标注者行为（Jury Learning等），属于"training with individual labels"视角主义的典型方法，本文将其定位为perspectivist实践之一。
7. **Bowker & Star (2000)** —— STS领域的经典著作，提出分类系统的政治性与"retrievability"概念，本文跨学科引入以理解个体声音如何在标注系统中被保留或丢失。
8. **Arrow (1977); Sen (2018)** —— 社会选择理论的奠基作品，研究偏好聚合的最优机制，本文借用其理论框架分析多数投票的内在局限。

## 局限性与未来方向
1. **立场论文本身的局限**：作者自述本研究并非全面的meta-analysis，可能遗漏部分相关文献，且仅聚焦NLP领域，其他ML子领域（如CV）的视角主义实践尚未充分讨论。
2. **分歧成因的变量空间仍不够广**：当前视角主义研究过度依赖demographic因素（race、gender、age），但多项实证研究表明非人口因素（社媒习惯、语言背景、任务specific认知）可能是更重要的分歧来源；未来需系统化探索更广的预测变量空间。
3. **评估指标尚未摆脱多数投票依赖**：Plank (2022) 指出大多数视角主义论文仍在用aggregated gold label做评估，学界缺乏成熟的、基于opinion distribution的模型评估方法体系。
4. **规范性决策仍多为implicit**：多数研究未显式声明其对"谁的opinion算数""分歧的可接受边界"的立场，易导致隐性价值判断通过多数投票被自然化。
5. **参与式方法与制度压力的张力**：理论上参与式设计能更好纳入边缘群体意见，但学术界"publish or perish"和工业界"ship fast"的制度压力使其难以落地，需要机制性激励改革。
6. **个人化的边界问题**：per-annotator个性化模型虽能捕捉个体差异，但可能放大misinformation传播等 harms，何时及如何个性化仍需规范讨论。

## 研究启发与可借鉴点
1. **标注者招募的分层策略（stratified recruitment）**：可按任务相关的分歧轴（如目标用户的地理分布、语言变体、专业背景）对招募样本进行分层，upsample易被低估的群体；同时设置per-annotator标注数量上限，防止少数高产标注者主导数据集。
2. **用intra-annotator agreement替代inter-annotator agreement作为质量控制指标**：Abercrombie et al. (2023) 提出的方法可在不丢弃minority opinion的前提下过滤spam和低质量标注，值得在团队标注pipeline中引入。
3. **模型评估中使用distributional metrics**：可用KL divergence、cosine similarity between output ranking lists、correlation coefficient替代单一gold label accuracy，更充分地利用perspectivist数据收集的丰富信息。
4. **数据集文档（dataset documentation）扩展规范**：除常规的annotator demographics和task instructions外，建议额外记录：① annotator selection procedure及参与限制；② 每人标注item的分布；③ 任何annotator filtering的criteria；④ 对 discarded data的rationale——这有助于后续研究者判断数据的代表性边界。
5. **跨学科借镜：将社会选择理论引入标注设计**：Arrow不可能性定理等结果表明"完美聚合机制"不存在，可引导研究者从"寻找最优聚合"转向"显式选择并论证所选聚合策略的价值立场"，提升标注设计的透明度与可争议性。

## 关键术语表
**Perspectivist Paradigm（视角主义范式）**：将标注者之间的label差异视为有价值的信息来源而非噪声，主张数据集和模型应捕捉人类意见的多样性而非追求单一ground truth的新兴标注范式。

**Longstanding Paradigm（传统标注范式）**：假设每个数据实例存在单一真实标签，通过收集多份标注并用多数投票等方法聚合来逼近该标签，将annotator disagreement视为需最小化的质量问题。

**Inter-annotator Agreement (IAA)**：多位标注者对同一数据实例给出相同label的一致性程度，传统范式中常作为数据质量的proxy指标，视角主义范式则认为高IAA可能掩盖了合理分歧。

**Intra-annotator Agreement**：同一位标注者在不同时间对相似任务的一致性，可作为过滤spam/低质量标注的替代指标而无需丢弃minority opinions（Abercrombie et al., 2023）。

**Ground Truth（真实标签）**：传统范式假设每个标注任务存在唯一正确标签，视角主义范式质疑这一假设，指出许多任务（尤其涉及社会规范的）本质上不存在单一正确答案。

**Statistical Bias vs. Societal Bias**：前者指估计值与期望值的差异（如label偏离群体均值），后者指基于群体身份的系统性歧视；传统实践常将二者混淆，导致minority opinions被错误标记为"biased"。

**Common Ground（共同基础）**：源于语用学，指交际参与者基于共享的世界知识（包括人口特征、职业、语言社区等）形成的相互理解；标注者可被视为在缺省sender意图的情况下基于自身background进行interpretation的intermediary。

**Democratic vs. Expert-driven 光谱**：数据标注中关于"谁的opinion应该被纳入"的设计光谱——一端是所有stakeholder平等参与（democratic），另一端是仅征询具备特定知识的群体（expert-driven），不同任务适合不同位置。

## 可复现要素
- **数据集**：本文无独立构建的数据集；引用的关键数据集包括DICES（Aroyo et al., 2023）、GoEmotions（Demszky et al., 2020）、Measuring Hate Speech（Sachdeva et al., 2022）等，均有公开代码/数据。
- **代码/权重**：本文无独立开源代码；视角主义相关方法的开源代码分散于各引用论文（如Jury Learning、soft-label multi-task learning等）。
- **关键超参**：未适用（立场论文）。
