---
title: "TISE: A Tripartite In-context Selection Method for Event Argument Extraction"
source: https://aclanthology.org/2024.naacl-long.101.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:42:38"
field: "事件抽取与In-context学习"
keywords: ["In-context Learning", "Event Argument Extraction", "DPP", "Few-shot", "Prompt Engineering", "ICL Example Selection"]
innovations: ["首次针对EAE任务提出语义相似性+示例多样性+事件相关性三分量选择框架", "将多目标评分通过DPP核矩阵融合，以行列式概率直接选取最优示例子集", "TISE以少量示例（k=5）在ACE05上超越部分监督方法，全量数据下超越DyGIE++"]
benchmarks: ["ACE05", "RAMS"]
---

# 论文速读：TISE: A Tripartite In-context Selection Method for Event Argument Extraction

## 一句话总结
TISE是一种面向事件参数提取（EAE）任务的三分量in-context示例选择方法，通过**语义相似性**、**示例多样性**和**事件相关性**三个维度协同筛选最优示例集；在ACE05数据集上以较少示例取得了超越部分监督方法（DyGIE++）的性能，验证了多约束选择策略的有效性。

## 研究问题与动机
1. **现有in-context选择仅依赖语义相似性**：主流做法（如BERT-TOPK）直接选取与测试输入语义最相似的top-k示例，但应用于EAE任务时可能选到高度重叠的示例，无法提供额外事件信息。
2. **忽视事件属性相关性**：测试输入为某一事件类型（如Die），若所选示例全部来自其他事件类型（如Transport），LLM无法理解角色含义，导致预测失败。
3. **EAE训练数据稀缺**：传统监督方法（如span-based、reading comprehension范式）依赖大量标注数据，泛化到未知样本能力有限，LLM的ICL提供了低资源场景下的替代方案。
4. **示例质量高度影响LLM推理**：Rubin等人指出ICL性能对示例选择极度敏感，但EAE领域尚未系统研究如何设计最优选择策略。

## 核心贡献（创新点）
1. **首次针对EAE任务提出in-context示例选择的三重要求**：明确语义相似性（R.1）、示例多样性（R.2）、事件相关性（R.3）缺一不可，此前工作未同时考虑这三者。
2. **设计TISE框架并引入DPP融合多目标分数**：将三个独立评分通过核矩阵嵌入Determinantal Point Process，以闭式解直接选取最优示例子集，避免了贪心迭代的多步搜索。
3. **实验证明TISE可用更少示例达到更强性能**：TISE在k=5时Arg-C达48.72%，优于BERT-TOPK在k=15时的48.16%；全量实验下（text-davinci-003 + code prompt）超越监督方法DyGIE++（60.9% vs 60.7%）。
4. **揭示作用机理**：通过role overlap rate正相关分析，证明事件相关性提升role重合度、示例多样性保证角色覆盖广度，两者存在互补强化效应。

## 方法详解
TISE由评分器（Scorer）+ DPP选择器 + Code Imitation Prompt三部分组成。

### 评分器设计
1. **Text Scorer（语义相似性）**：采用`bert-base-uncased`编码输入与示例，余弦相似度作为文本分：$s_T(x, t) = \text{sim}(E(x), E(t))$。
2. **Event Scorer（事件相关性）**：将事件属性拆分为Event Type和Event Role两部分：
   - **Event Type Score**：为每个事件类型（如Justice:Convict）设计自然语言描述（父类+自身两句话），计算描述embedding的余弦相似度$\hat{s}_E(x, t_i)$。
   - **Event Role Score**：为每个角色设计单句描述（如Life:Die.Instrument → "What device was used to kill?"），对测试角色集合$R^x$与示例角色集合$R^{t_i}$逐对比较，取max-pool后平均，并加一项$α|R^{t_i}|$（α=0.1）奖励包含更多有用角色的示例：$\check{s}_E(x, t_i) = \frac{\sum \max \check{s}_E(r^x_i, r^t_i)}{|R^{t_i}|} + α|R^{t_i}|$。
   - 最终$ s_E(x, t_i) = \frac{1}{2}(\hat{s}_E + \check{s}_E) $。

### DPP融合机制
将三个评分整合入核矩阵$K$：
$$K = \text{Diag}(S'_E) \cdot \text{Diag}(S'_T) \cdot \bar{K} \cdot \text{Diag}(S'_T) \cdot \text{Diag}(S'_E)$$
其中$\bar{K}_{ij} = s_T(t_i, t_j)$，$S'_E = \exp(s_E / 2λ_1)$，$S'_T = \exp(s_T / 2λ_2)$（λ₁=0.5, λ₂=0.05）。

理论推导证明：
$$\log \det(K_A) = \sum_{t_i \in A}\left(\frac{S_{E_i}}{λ_1} + \frac{S_{T_i}}{λ_2}\right) + \log \det(\bar{K}_A)$$
- 第一项要求选中示例具有**高事件分和语义分**。
- 第二项$\log \det(\bar{K}_A)$因行列式性质，迫使选中示例间**两两相似度尽可能低**，从而保证多样性。

使用fast greedy算法（k步迭代）在$O(k|T|^2)$内近似求解：
$$t^* = \arg\max_{t_i \in T \setminus A^*} \det(K_{A^* \cup \{t_i\}}) - \det(K_{A^*})$$

### Code Imitation Prompt
采用Wang等（2023）的代码模仿提示模板，含四个模块：Entity Type Definition、Event Definition、In-context Examples、Event Instantiation，将EAE任务转化为类代码生成形式。

## 实验与结果
- **数据集**：ACE05（句子级，8个父事件类、33个子类）；另在RAMS上进行跨域验证。
- **评估指标**：Arg-I（论元词头匹配，F1）、Arg-C（论元正确分类，F1）。
- **主要结果**（ACE05，k=15）：

| 方法 | Arg-I | Arg-C |
|------|-------|-------|
| RANDOM | 58.00 | 47.34 |
| BERT-TOPK | 58.34 | 48.16 |
| DPR-TOPK | 58.40 | 47.97 |
| DPP-DIVERSITY | 59.46 | 49.38 |
| **TISE** | **60.99** | **51.43** |

- **关键发现**：
  - TISE在所有k值下均达到SOTA，相对DPP-DIVERSITY在k=15时提升+1.53%/+2.05%（Arg-I/Arg-C）。
  - TISE在k=5时Arg-C=48.72%，已超过BERT-TOPK在k=15的48.16%，节省2/3示例。
  - **消融**（k=10）：移除Event Correlation下降2.1%/2.9%，移除Example Diversity下降1.5%/2.1%，移除Event Type（-1.8%/−1.7%）> 移除Event Role（-0.6%/−0.5%）。
  - **与监督方法对比**：仅用5%训练数据时TISE Arg-C=38.8%，超越DEGREE在5%下的35.5%；全量数据+text-davinci-003+code prompt时TISE Arg-C=60.9%，超越DyGIE++（60.7%）。
  - **RAMS文档级实验**：TISE同样优于基线，但不同选择方法提升幅度较小，推测文档理解难度更高。

## 相关工作脉络
1. **Semantic Retriever（BERT-TOPK/DPR-TOPK）**：仅基于语义相似性选top-k示例，未考虑示例间冗余与事件类型对齐，TISE在相同框架上增加多样性与事件相关性约束后显著提升。
2. **DPP-DIVERSITY**：引入DPP保证多样性，但缺少事件相关性评分；TISE在其基础上加入Event Score，Arg-C在k=15时提升2.05%。
3. **Code Prompt / Code Imitation Prompt（Wang et al., 2023）**：将EAE转化为代码生成任务，TISE与之正交——TISE专注于示例选择，可适配任意prompt模板。
4. **DEGREE（Hsu et al., 2021）**：监督式高效事件抽取方法，利用结构化数据增强；TISE在低资源（≤10%数据）下超越DEGREE，但在全量数据下仍有约12.6%差距，说明监督架构优势仍存。
5. **ICL示例选择研究（Rubin et al., 2022; Ye et al., 2022, 2023）**：聚焦通用NLP任务，TISE是首个在EAE这一结构化抽取任务上系统研究in-context选择的工作，强调事件属性的独特性。

## 局限性与未来方向
1. **编码器较简单**：使用vanilla BERT作为scorer，未采用基于LLM ranking label的监督训练或强化学习优化检索器；作者认为直接用EAE任务性能作为reward是更合理方向。
2. **时间开销较高**：Event Role评分需对每个示例的每个角色单独查询测试角色集合，复杂度随角色数线性增长，可通过预计算或向量检索加速。
3. **文档级效果有限**：RAMS上不同选择方法的提升不显著，推测文档理解难度高、需更优prompt设计。
4. **超参数敏感**：λ₂（语义分温度）较敏感，λ₁较平滑，后续可探索自适应学习策略。
5. **仅验证两个数据集**：未覆盖更多事件抽取基准或跨语言场景。

## 研究启发与可借鉴点
1. **三分量选择框架的可迁移性**：语义相似性+多样性+任务属性相关性三者组合的思路可推广至信息抽取（命名实体识别、关系抽取）、文本摘要等其他ICL任务。
2. **DPP核矩阵的构造技巧**：将多目标评分分解为可分离的对角缩放矩阵与相似度核矩阵乘积形式，既保留理论可解性又实现多约束融合，值得在其他选择问题上借鉴。
3. **事件属性的自然语言描述设计**：为事件类型/角色设计层级化描述（父类+子类、角色含义句子），能有效编码任务先验知识，可复用于其他结构化预测任务的提示工程。
4. **role overlap rate作为分析指标**：用角色重合率量化示例质量与性能的关系，提供了可解释的评估维度，便于后续消融与调试。
5. **低资源下的"少而精"策略**：TISE以少量高质量示例超越监督方法，验证了在数据受限场景下ICL的潜力，对资源紧张的实际部署有参考价值。

## 关键术语表
**In-context Learning (ICL)**：通过在大模型输入中提供少量示例（prompt内），使模型无需微调即可完成特定任务的范式。
**Event Argument Extraction (EAE)**：从文本中识别特定事件类型下的参与论元及其角色标签的结构化抽取任务。
**Determinantal Point Process (DPP)**：一种基于行列式的概率模型，常用于选取多样性高的子集，此处用于融合多目标评分并选择最优示例集合。
**Code Imitation Prompt**：将结构化输出任务转化为类代码实例化形式（如Python类赋值）的提示模板，帮助LLM生成规范格式结果。
**Arg-I / Arg-C**：事件参数提取的两个评估指标，Arg-I关注论元文本片段是否被正确识别，Arg-C进一步要求角色分类准确，均以F1计。
**Role Overlap Rate**：所选in-context示例集合与测试输入之间共有事件角色的比例，用于量化示例对测试任务的相关性。

## 可复现要素
- **数据集**：ACE05（公开）、RAMS（公开）；预处理方式为8个父事件类+33个子类。
- **代码/权重**：论文未提供开源代码仓库链接；`bert-base-uncased`编码器可获取。
- **关键超参**：λ₁=0.5（事件分温度）、λ₂=0.05（语义分温度）、α=0.1（角色数量奖励系数）、k=3/5/10/15（示例数）。
- **LLM**：text-davinci-002 / text-davinci-003（通过OpenAI API访问）；Codex系列已弃用。
- **Prompt模板**：Code Imitation Prompt（论文图8）；也可替换为Code Prompt（图9）。
- **评估脚本**：按标准F1计算，Arg-I以head token匹配为准。
