---
title: "TISE: A Tripartite In-context Selection Method for Event Argument Extraction"
source: https://aclanthology.org/2024.naacl-long.101.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:43:39"
field: "事件抽取与提示学习"
keywords: ["事件论元抽取", "上下文学习", "示例选择", "行列式点过程", "大语言模型"]
innovations: ["提出语义相似度-多样性-事件相关性三分量联合打分框架", "设计自然语言描述的事件类型/角色评分器并以DPP统一融合", "以少量示例实现超越全监督方法DyGIE++的EAE性能"]
benchmarks: ["ACE05", "RAMS"]
---

# 论文速读：TISE: A Tripartite In-context Selection Method for Event Argument Extraction

## 一句话总结
本文针对事件论元抽取（EAE）任务，提出了 TISE——一种三分量上下文选择方法，同时考量语义相似度、示例多样性和事件相关性三个维度，利用行列式点过程（DPP）联合建模并直接选择最优示例集合；在 ACE05 上，TISE 以极少示例达到 SOTA，甚至在全量场景下超过了监督基线 DyGIE++。

## 研究问题与动机
1. **现有语义检索方法的冗余性**：当前基于语义相似度选取 top-k 示例的做法容易选出语义重叠、事件信息重复的样本，无法为 LLM 提供额外的论元线索。
2. **忽视事件属性相关性**：语义检索不区分事件类型与论元角色，选出的示例可能与测试输入的事件类型无关，导致 LLM 无法理解目标事件的结构化知识。
3. **EAE 数据稀缺限制传统方法泛化**：EAE 标注成本高、训练数据有限，传统方法难以覆盖未知样本；LLM 的 ICL 范式在低资源场景下潜力巨大，但高度依赖示例质量。
4. **缺少面向 EAE 的系统性示例选择研究**：已有工作多关注通用 NLP 任务的 ICL 示例选择，事件抽取任务中如何选取"有用且多样"的示例仍属空白。

## 核心贡献（创新点）
1. **首次为 EAE 任务系统提出 ICL 示例选择的三要素要求**（语义相似度、多样性、事件相关性），明确了该任务中优质示例的本质属性。
2. **设计了事件评分器（SCORER_E），通过自然语言描述刻画事件类型与事件角色**，使事件相关性可从细粒度角色层面被量化，区别于仅依赖文本相似度的已有方法。
3. **构建了基于 DPP 的联合建模框架，将三类分数融合为一个核矩阵**，通过 fast greedy 算法直接输出最优示例子集，一次性兼顾互斥性与相关性，避免了贪心 top-k 的局部次优。
4. **在 ACE05 上以 k=15 + text-davinci-003 + code prompt 的组合超越全监督方法 DyGIE++（Arg-C 60.9 vs 60.7）**，证明高质量示例选择可弥补标注数据的不足。
5. **提出角色重叠率（role overlap rate）分析**，从原理层面验证了事件相关性显著提升角色重叠率、从而改善抽取性能的作用机制。

## 方法详解
**整体框架**：给定测试输入 $x$ 与训练集 $T$，TISE 先对所有 $t_i \in T$ 计算三种分数，再将其融合为 DPP 核矩阵 $K$，最后用 fast greedy 算法（$k$ 步搜索）选出最优子集 $A^*$，接入 code imitation prompt 驱动 LLM 推理。

**三种分数**：
1. **Semantic Similarity（文本分数）**：使用 bert-base-uncased 作为编码器，$s_T(x, t_i) = \text{sim}(\mathrm{E}(x), \mathrm{E}(t_i))$（余弦相似度），保证示例与测试输入在语义层面的基本匹配。
2. **Example Diversity（多样性惩罚）**：核函数中引入项 $\bar{K}_{ij} = s_T(t_i, t_j)$，DPP 的行列式性质使高互相似度的样本对概率下降，从而鼓励所选子集内部多样化；$k_2(t_i, t_j|x) = s_T(x, t_i) \cdot s_T(t_i, t_j) \cdot s_T(x, t_j)$。
3. **Event Correlation（事件相关性）**：
   - **事件类型分**：为每个事件类型预置两段自然语言描述（含父类+自身），$s̃_E(x, t_i) = \text{sim}(\mathrm{E}(\hat{d}(x)), \mathrm{E}(\hat{d}(t_i)))$。
   - **事件角色分**：为每个角色预置一段自然语言描述（问句形式），对 $t_i$ 的每个非空角色 $r_i^{t_i}$，取其与 $x$ 所有目标角色的最大相似度，再平均并加上角色数量奖励项 $\alpha|R^{t_i}|$（$\alpha=0.1$）：$\check{s}_E(x, t_i) = \frac{\sum \max_r \check{s}_E(r_i^x, r_i^{t_i})}{|R^{t_i}|} + \alpha|R^{t_i}|$。
   - 最终 $s_E(x, t_i) = \frac{1}{2}(s̃_E + \check{s}_E)$。

**核矩阵构建**：经温度参数缩放 $s'_E = \exp(s_E/(2\lambda_1))$、$s'_T = \exp(s_T/(2\lambda_2))$ 后，
$$K = \text{Diag}(S'_E) \cdot \text{Diag}(S'_T) \cdot \bar{K} \cdot \text{Diag}(S'_T) \cdot \text{Diag}(S'_E)$$
取对数后：$\log\det(K_A) = \sum_{t_i \in A}(S_{E_i}/\lambda_1 + S_{T_i}/\lambda_2) + \log\det(\bar{K}_A)$，前半部分鼓励高事件/语义分数，后半部分通过行列式特性惩罚内部相似度，实现三者统一优化。

**Prompt 模板**：采用 code imitation prompt（Wang et al., 2023），包含实体类型定义、事件类型定义、ICL 示例、以及测试输入的 event instantiation 四个部分，将所选示例直接嵌入。

**超参**：$\lambda_1=0.5$（事件相关性温度）、$\lambda_2=0.05$（语义相似度温度），$\lambda_2$ 更敏感。

## 实验与结果
**数据集**：ACE05（句子级，8 大类/33 子类）、RAMS（文档级）。
**评估指标**：Arg-I（论元头词匹配 F1）、Arg-C（论元正确分类 F1）；RAMS 用 EM-F1。

**主要结果（ACE05）**：

| 方法 | k | Arg-I | Arg-C |
|---|---|---|---|
| BERT-TOPK | 15 | 58.34 | 48.16 |
| DPP-DIVERSITY | 15 | 59.46 | 49.38 |
| **TISE** | **15** | **60.99** | **51.43** |
| **TISE** | **5** | **58.95** | **48.72** |

- TISE 在所有 $k$ 值下均达 SOTA；$k=5$ 即超过 BERT-TOPK $k=15$。
- 相对于语义检索基线的提升（$\Delta$SEMANTIC）：$k=15$ 时 Arg-I +2.59 / Arg-C +3.27。
- 消融（Table 3，$k=10$）：移除 Event Correlation 损失最大（Arg-C -2.9），移除 Diversity 次之（-2.1），两者联合移除呈部分补偿关系（3.9% < 2.9%+2.1%）。

**与监督方法对比（Table 5）**：
- Few-shot（5% 训练数据）：TISE $k=5$（Arg-C 38.8）> DEGREE（35.5）。
- Full-shot：TISE$_{k=15}$ + text-davinci-003 + code prompt Arg-C 60.9 > DyGIE++（60.7）；与 DEGREE（73.5）仍有 12.6% 差距。

**RAMS 文档级**（Figure 5）：TISE 在不同 $k$ 下均优于基线，验证跨场景泛化。

**角色重叠率分析（Figure 3）**：性能与 role overlap rate 呈明显正相关，Event Correlation 对重叠率贡献最大，Example Diversity 贡献最小（主要确保子集覆盖不同角色）。

## 相关工作脉络
1. **Semantic Retriever（BERT-TOPK/DPR-TOPK）**：通过余弦相似度选 top-k，是本文重点改进对象——缺乏对示例间多样性和事件属性的建模。
2. **DPP-DIVERSITY**：仅用 DPP 保证多样性，未考虑事件类型/角色相关性，TISE 在其基础上补充 Event Correlation。
3. **Code Prompt / Code Imitation Prompt（Wang et al., 2023）**：TISE 的 prompt 设计直接沿用此模板，TISE 与其正交（TISE 解决"选什么示例"，prompt 解决"如何组织示例"）。
4. **DEGREE（Hsu et al., 2021）**：数据高效监督方法，通过结构化解码降低标注依赖；TISE 在低资源 Few-shot 下优于 DEGREE，全量下仍有差距。
5. **DyGIE++（Wadden et al., 2019）**：经典端到端抽取模型；TISE 在全量场景下以 ICL 方式超越其 Arg-C 分数，证明示例选择策略的潜力。
6. **ICL 示例选择（Rubin et al., 2022; Liu et al., 2022; Ye et al., 2022; Su et al., 2022）**：通用 NLP 任务中的检索与多样性研究，本文首次将此类思想系统化应用于 EAE 任务。

## 局限性与未来方向
1. **编码器仍为 vanilla BERT**：未引入基于 LM ranking 标签自监督训练的检索器；作者建议用强化学习以 EAE 性能为 reward 优化 retriever。
2. **时间开销较高**：事件角色评分需对每个示例的每个角色单独查询测试角色描述，计算复杂度较高（虽通过字典存储降至 $O(1)$ 单次查询）。
3. **文档级效果有限**：RAMS 实验中不同选择方法性能提升不显著，作者推测原因可能是文档理解难度更高、需要更强的 prompt 设计。
4. **与全监督方法仍有差距**：TISE 在 full-shot 下与 DEGREE 相差 12.6% Arg-C，说明仅靠示例选择无法完全替代架构级优化。

## 研究启发与可借鉴点
1. **三分量核矩阵构建思路可迁移**：将任务特定知识（如事件类型/角色描述）编码为额外评分维度，并用 DPP 统一融合的框架，可推广至其他结构化抽取任务（如关系抽取、核心论元识别）。
2. **自然语言描述的事件属性索引**：预先为事件类型和角色编写描述文本并建立向量索引，是一种低成本的"任务知识注入"方式，无需额外训练即可提升检索质量。
3. **角色重叠率作为分析指标**：用于解释 ICL 示例选择与抽取性能之间的关系，可为后续研究提供可量化的因果分析手段。
4. **Few-shot 下超越监督方法的启示**：表明在低资源场景下，高质量的示例选择策略可能比扩大模型/架构复杂度更具性价比，值得在更多 NER/EE 子任务上验证。
5. **prompt 与选择方法正交设计**：TISE 可适配任意 prompt 模板，这一设计思路有利于将不同领域知识模块化，便于快速迁移至新任务。

## 关键术语表
**Event Argument Extraction（EAE）**：从文本中识别事件触发词并抽取其对应论元及论元角色，是信息抽取的核心子任务之一。
**In-Context Learning（ICL）**：在 LLM 推理时于 prompt 中提供若干已标注示例，利用模型本身的知识完成下游任务，无需梯度更新。
**Determinantal Point Process（DPP）**：一种基于行列式概率模型的采样方法，常用于产生具有代表性和多样性的子集选择问题。
**Semantic Similarity（语义相似度）**：通过编码器将文本映射为向量，以余弦相似度衡量测试输入与候选示例的文本级相似程度。
**Event Correlation（事件相关性）**：本文提出的评分维度，通过比较事件类型描述和事件角色描述的语义相似度来衡量示例与测试输入在事件结构层面的相关程度。
**Role Overlap Rate（角色重叠率）**：所选示例中与测试输入共享的论元角色比例，用于量化示例对 LLM 的有用性并提供可解释的分析视角。
**Code Imitation Prompt**：以 Python 类定义格式组织实体类型、事件类型及示例标签的 prompt 模板，引导 LLM 以代码实例化方式输出结构化抽取结果。
**fast Greedy Algorithm（for DPP）**：对 DPP 最大化问题的高效近似算法，通过 $k$ 步贪心搜索逐步添加使行列式增量最大的元素。

## 可复现要素
- **数据集**：ACE05（公开）、RAMS（公开）；ACE05 预处理方式遵循 Wadden et al., 2019（8 大类/33 子类）。
- **代码/权重**：论文未提及代码开源声明（论文未提及）。
- **编码器**：bert-base-uncased（公开）。
- **LLM**：text-davinci-002（OpenAI API，已 deprecated 2023.3）、text-davinci-003（OpenAI API）。
- **关键超参**：$\lambda_1=0.5$（事件相关性温度）、$\lambda_2=0.05$（语义相似度温度）、$\alpha=0.1$（角色数量奖励系数）。
