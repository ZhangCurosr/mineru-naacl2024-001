---
title: "TISE: A Tripartite In-context Selection Method for Event Argument Extraction"
source: https://aclanthology.org/2024.naacl-long.101.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:43:47"
field: "事件抽取与上下文学习"
keywords: ["In-context Learning", "Event Argument Extraction", "Determinantal Point Process", "Example Selection", "Prompt Engineering"]
innovations: ["提出语义相似度+示例多样性+事件相关性三要素并整合入DPP核矩阵", "设计事件类型与角色的自然语言描述评分器（event scorer）", "用少量上下文示例即可超越部分全监督EAE方法"]
benchmarks: ["ACE05", "RAMS"]
---

# 论文速读：TISE: A Tripartite In-context Selection Method for Event Argument Extraction

## 一句话总结
论文针对事件参数抽取（EAE）任务的上下文学习示例选择问题，提出 TISE 方法，从**语义相似度、示例多样性、事件相关性**三个维度对训练示例评分，并利用确定点过程（DPP）直接选择最优示例子集。在 ACE05 数据集上，TISE 用较少示例即超越部分全监督方法，且对不同 LLM 和 prompt 模板具有良好适应性。

## 研究问题与动机
1. **示例选择决定 ICL 性能**：大模型上下文学习的表现高度依赖所选示例质量，但事件参数抽取任务中直接按语义相似度取 top-k 的简单方法存在明显缺陷。
2. **语义相似方法会选到重复示例**：仅看语义相似度容易选到语义重叠严重的示例（如图 1 中第二、第三示例均描述"cross-border"事件），无法为 LLM 提供丰富的额外事件信息。
3. **忽略事件属性相关性**：现有方法不考虑测试输入与示例之间的事件类型/参数角色关联，导致当测试事件类型在上下文中无对应示例时，LLM 难以理解并预测参数（如图 1 中测试类型为 Die 却无相关示例）。

## 核心贡献（创新点）
1. **首次提出 EAE 任务上下文示例选择的三要素**：明确语义相似度、示例多样性、事件相关性是缺一不可的要求，此前的基于检索或 DPP 的方法仅满足其中一部分。
2. **设计 TISE 三维度评分 + DPP 融合框架**：分别用 text scorer 和 event scorer 计算三项分数，通过核矩阵将三者整合，利用 DPP 子模最大化直接选择示例子集，避免局部贪心陷阱。
3. **轻量编码即可超越监督方法**：使用朴素 BERT 编码器和少量示例（k=5~15），TISE 即可在 ACE05 上达到甚至超过 DyGIE++ 等全监督模型的 Arg-C 分数，验证了高效上下文选择的价值。

## 方法详解
TISE 整体流程包括三步：计算三维度分数 → 构建核矩阵 → 利用 DPP 选择示例子集 → 配合 code imitation prompt 送入 LLM 推理。

### 4.1 评分器与核矩阵

1. **语义相似度（Semantic Similarity）**
   - 使用预训练编码器 E(·)（本文选用 `bert-base-uncased`）分别对测试输入 x 和示例 t 编码。
   - 余弦相似度得分：$s_T(x, t) = \sin(E(x), E(t))$。
   - 核矩阵项：$k_1(t_i, t_j|x) = s_T(x, t_i) \cdot s_T(x, t_j)$。

2. **示例多样性（Example Diversity）**
   - 同样使用 text scorer 衡量示例间相似度，避免选到重叠语义的示例。
   - 核矩阵修改为：$k_2(t_i, t_j|x) = s_T(x, t_i) \cdot s_T(t_i, t_j) \cdot s_T(x, t_j)$，其中 $s_T(t_i, t_j)$ 越小（越多样），乘积越大，从而在 DPP 中被优先选中。

3. **事件相关性（Event Correlation）**
   - 将事件属性分为**事件类型**和**事件角色**两部分，并为每类设计自然语言描述（见论文 Table 1）。
   - **事件类型得分**：对测试输入和示例分别获取类型描述 $\hat{d}(x)$、$\hat{d}(t_i)$，计算 $\hat{s}_E(x, t_i) = \sin(E(\hat{d}(x)), E(\hat{d}(t_i)))$。
   - **事件角色得分**：对每个示例中非空角色 $r_i^{t_i}$ 与其描述 $\check{d}(r_i^{t_i})$，与测试输入待预测角色 $r_i^x$ 的描述做相似度比较，取 max-pool 后再对所有角色求平均，并加一项惩罚项 $\alpha |R^{t_i}|$（$\alpha=0.1$）以奖励含更多有用角色的示例。
   - 最终事件得分取两者的平均：$s_E(x, t_i) = \frac{1}{2}(\hat{s}_E(x, t_i) + \check{s}_E(x, t_i))$。
   - 完整核函数：$k(t_i, t_j|x) = s_E(x, t_i) \cdot k_2(t_i, t_j|x) \cdot s_E(x, t_j)$。

4. **分数平衡与理论分析**
   - 引入温度超参数 $\lambda_1$（事件相关性）、$\lambda_2$（语义相似度），对得分进行缩放：$s'_E = \exp(s_E / 2\lambda_1)$，$s'_T = \exp(s_T / 2\lambda_2)$。
   - 核矩阵可分解为：$K = \text{Diag}(S'_E) \cdot \text{Diag}(S'_T) \cdot \bar{K} \cdot \text{Diag}(S'_T) \cdot \text{Diag}(S'_E)$，其中 $\bar{K}_{ij} = s_T(t_i, t_j)$。
   - DPP 的选取概率正比于 $\log\det(K_A) = \sum_{t_i \in A}(\frac{s_E}{\lambda_1} + \frac{s_T}{\lambda_2}) + \log\det(\bar{K}_A)$，第一项鼓励高事件得分和高语义得分，第二项因行列式性质鼓励示例间低相似度（即多样性）。

5. **Code Imitation Prompt**
   - 采用 Wang et al. (2023) 的 code imitation prompt，包含四部分：Entity Type Definition、Event Definition、In-context Examples、Event Instantiation，引导 LLM 将句子"翻译"为 Python 类实例来实现参数抽取。

### 4.2 示例选择算法
使用 DPP 的快速贪心算法（Chen et al., 2018）进行 k 步搜索，每步选择使新核矩阵行列式增量最大的示例：
$$t^* = \arg\max_{t_i \in T \setminus A^*} \det(K_{A^* \cup \{t_i\}}) - \det(K_{A^*})$$
直至选出 k 个示例构成最终上下文集合 $A^*$。

## 实验与结果
- **数据集**：ACE05（句子级别 EAE 数据集，预处理为 8 个父事件类型、33 个子类型）。
- **评估指标**：Arg-I（参数头词匹配）和 Arg-C（参数头词匹配且角色分类正确），均报告 F1。
- **基线**：RANDOM、BM25、BERT-TOPK、DPR-TOPK、DPP-DIVERSITY。

主要结果（ACE05）：
| 方法 | k=5 Arg-I / Arg-C | k=10 Arg-I / Arg-C | k=15 Arg-I / Arg-C |
|---|---|---|---|
| BERT-TOPK | 56.67 / 46.14 | 56.93 / 46.98 | 58.34 / 48.16 |
| DPR-TOPK | 57.03 / 46.88 | 57.69 / 47.62 | 58.40 / 47.97 |
| DPP-DIVERSITY | 58.19 / 47.73 | 58.47 / 47.91 | 59.46 / 49.38 |
| **TISE** | **58.95 / 48.72** | **60.57 / 50.78** | **60.99 / 51.43** |

- **最强结果**：TISE 在 k=15 时取得 Arg-I=60.99 / Arg-C=51.43，较语义检索基线（BERT-TOPK k=15）提升 +2.65 / +3.27。
- **少样本优势**：TISE 仅用 k=5 的 Arg-C（48.72）即超越 BERT-TOPK k=15 的 Arg-C（48.16），在低资源场景下优势显著。
- **超越监督方法**：配合 text-davinci-003 和 code prompt，TISE k=15 在 ACE05 全量数据上 Arg-C=60.9 超过 DyGIE++（60.7），但仍落后 DEGREE（73.5）约 12.6%。
- **AB 实验**：移除 Event Correlation 下降最多（Arg-C -2.9%），其次 Example Diversity（-2.1%），两者联合移除产生互补下降而非叠加，说明三者相互补充。
- **跨 LLM/prompt 鲁棒性**：在 text-davinci-002/003 和 code prompt/code imitation prompt 组合下，TISE 均稳定超越 BERT-TOPK。
- **文档级 EAE（RAMS）**：TISE 同样在文档级任务中筛选出更优示例子集，但 LLM 对文档理解仍具挑战。

## 相关工作脉络
1. **基于语义检索的 ICL 示例选择（Rubin et al., 2022; Liu et al., 2022）**：采用 BM25 或稠密编码器选 top-k 最相似示例，TISE 在此基础上加入多样性与事件相关性约束，避免语义重叠和事件不相关的问题。
2. **基于 DPP 的多样性选择（Ye et al., 2022; Su et al., 2022）**：仅关注示例间多样性，TISE 通过核矩阵设计同时融入测试输入语义相似度和事件相关性，使所选示例更贴合当前任务。
3. **EAE 传统方法（span-based / RC / 生成式范式）**：如 DyGIE++、DEGREE 等依赖大量标注数据，TISE 采用零样本/少样本 ICL 范式，无需微调即可在部分场景下超越监督方法。
4. **Prompt 工程（Wang et al., 2023 Code4Struct）**：采用 code imitation prompt 作为下游推理模板，TISE 与之正交，可适配任意 prompt 结构。
5. **事件类型/角色描述方法（Lu et al., 2023）**：本文借鉴其角色描述设计思路，将其与自然语言类型描述结合，形成 event scorer 的输入基础。

## 局限性与未来方向
1. **编码器能力有限**：当前使用朴素 `bert-base-uncased`，未采用通过 ranking label 或强化学习监督训练的专业检索编码器，检索能力有提升空间。
2. **计算开销较高**：event scorer 需要对每个示例的每个角色单独查询测试角色，时间复杂度随角色数增加而上升。
3. **与全监督方法的差距**：即便在 ACE05 全量数据上，TISE 仍落后 DEGREE 约 12.6%，说明示例选择仍有优化空间。
4. **文档级 EAE 效果一般**：在 RAMS 数据集上不同选择方法提升有限，可能需要更强大的 prompt 设计来辅助文档理解。

## 研究启发与可借鉴点
1. **三维度评分范式可迁移至其他 NLU 任务**：语义相似度 + 多样性 + 任务属性相关性的框架不仅适用于 EAE，也可推广至关系抽取、命名实体识别等结构化信息抽取任务。
2. **自然语言描述事件属性的设计策略值得借鉴**：为事件类型和角色设计两段式/一句话描述，并聚合多个子描述（max-pool + 平均）的方式简单有效，可作为通用模板复用。
3. **DPP 核矩阵设计技巧**：通过将三项分数以乘积形式融入核矩阵，使 DPP 同时优化个体质量和集合多样性，这一设计模式适用于需要多目标排序的示例选择场景。
4. **与 prompt 正交的设计思想**：TISE 与 code imitation prompt 完全解耦，提示模板可自由替换，体现了模块化设计的优势，便于在不同 LLM 上快速适配。
5. **角色重叠率（role overlap rate）作为分析指标**：论文定义了角色重叠率来解释三要素的作用机制，这一指标可用于诊断和比较不同示例选择方法的内在有效性。

## 关键术语表
- **In-context Learning (ICL)**：不更新模型参数，通过在 prompt 中提供若干 labeled examples 引导 LLM 完成目标任务的学习范式。
- **Determinantal Point Process (DPP)**：基于行列式的概率分布模型，常用于从候选集中选择具有高质量且多样化的子集。
- **Event Argument Extraction (EAE)**：从文本中识别事件触发词并抽取对应论元及其角色标签的结构化抽取任务。
- **Semantic Similarity**：通过文本编码器计算测试输入与示例之间的余弦相似度，衡量二者字面语义相近程度。
- **Example Diversity**：通过惩罚示例间高相似度来确保选出的示例集合覆盖不同的事件信息和角色，避免信息冗余。
- **Event Correlation**：通过对比事件类型和角色自然语言描述与测试输入的相关性，确保所选示例与当前事件类型和参数角色高度匹配。
- **Code Imitation Prompt**：将 EAE 任务转化为代码生成任务的形式化 prompt，通过 Python 类实例化表示抽取结果。
- **Arg-I / Arg-C**：EAE 任务两个评估指标，Arg-I 衡量参数头词是否匹配，Arg-C 额外要求角色分类正确。

## 可复现要素
- **数据集**：ACE05（公开数据集）；RAMS（公开数据集）。
- **代码/权重**：论文未提及代码开源声明；使用了 `bert-base-uncased` 预训练权重（公开）。
- **关键超参数**：$\lambda_1 = 0.5$（事件相关性温度）、$\lambda_2 = 0.05$（语义相似度温度）、$\alpha = 0.1$（角色数量奖励系数）。
- **LLM**：text-davinci-002 / text-davinci-003（通过 OpenAI API 访问，Codex 系列已于 2023 年 3 月停更）。
- **Prompt 模板**：code imitation prompt（Wang et al., 2023）及 code prompt，可在附录中找到示例。
