---
title: "Strings from the Library of Babel: Random Sampling as a Strong Baseline for Prompt Optimisation"
source: https://aclanthology.org/2024.naacl-long.122.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:46:13"
---

# 论文速读：Strings from the Library of Babel: Random Sampling as a Strong Baseline for Prompt Optimisation

## 一句话总结
本文证明直接从模型词表中随机采样Token作为提示分隔符（Separator），即可在提示风格文本分类任务中达到与现有LLM自优化方法相当的性能；该随机基线平均相对人类设计基线提升12%，且性能差距不足1%，有力挑战了“有效提示必须语义连贯或任务相关”的传统假设。

## 研究问题与动机
- 现有Prompt优化工作普遍依赖LLM生成候选提示，并隐含假设有效Separator需具备任务相关性、语言连贯性或上下文依赖性。
- 指令微调模型与复杂Meta-prompt的引入显著提升了调用成本与部署门槛，亟需验证是否存在更简单、低成本的强基线。
- 语言模型在In-Context Learning（ICL）设置下对表面“无意义”字符串的响应机制尚不明确，词表空间中有效Prompt的密度未被充分评估。
- 若随机采样即可逼近SOTA，则此前诸多“自优化”方法的实际贡献可能存在高估，需建立统一、透明的评测基准。

## 核心贡献（创新点）
- 提出基于随机采样的Separator优化框架，证明无任务关联的词表随机字符串即可作为强基线，打破“有效提示必须可读/相关”的经验假设。
- 设计三种递进式随机生成策略（Random Vocabulary / w/o Context / with Context），实证指令微调模型并非Prompt生成的必要条件。
- 在9个文本分类数据集与8个语言模型上建立统一对比基准，显示随机策略与OPRO/APE/EvoPrompt等方法的性能差距<1%，相对人类基线平均提升12%，为后续研究设立可复现的强基线。
- 在GSM8K数学推理任务上验证随机策略的泛化性，最佳随机Separator相对ZS-CoT提升23%且方差更低，表明该现象非分类任务特例。

## 方法详解
- **整体框架**：随机生成候选Separator → 拼接至训练集样本后 → 输入目标语言模型计算分类准确率作为评分 → 在预设采样预算 $k$ 内记录 $(s_i, m_i)$ 对，最终选取最高分Separator用于测试。
- **三种生成策略**：
  1. **Random Vocabulary**：完全脱离语言模型与上下文，按词表均匀分布随机采样Token拼接为固定长度字符串，不依赖任何LM先验。
  2. **Random w/o Context**：从语言模型的边缘分布（Prior Distribution）中采样，生成表面具备语法连贯性但无任务/上下文引导的自然语言短语。
  3. **Random with Context**：在Meta-prompt中注入少量训练样本作为上下文，令语言模型在条件分布下采样，使生成的Separator带有一定任务相关性。
- **评估机制**：采用One-shot ICL设定，训练集每任务仅64个样本。对每个样本 $(x_i, y_i)$，计算 $\hat{y}_i = \arg\max_{v \in V} P(v \mid c \oplus x_i \oplus s; \theta)$，以验证集准确率为离散优化指标，无需梯度可微性。
- **选择机制**：采样过程独立进行，通过固定预算 $k=160$ 累积候选集合后直接取准确率最高者，不涉及迭代进化或反馈循环。

## 实验与结果
- **实验设置**：9个文本分类数据集（SST-2/5, MR, CR, MPQA, Subj, TREC, AGNews, DBPedia），每任务64训练样本；8个模型（GPT2-Large/XL, Mistral 7B/Instruct, Llama-Alpaca 7B, Llama2 7B/Chat, ChatGPT）；对比基线含人类基线（Answer:, Foo Bar, ZS-CoT）、OPRO、OPRO-ICL、MI、NI、APE、EvoPrompt。
- **主要结果**：
  - **Random Vocabulary** 相对人类基线“Answer:”平均提升10%；**Random w/o Context** 与 **Random with Context** 分别提升12%，三者性能差距仅0.5%相对值。
  - 与OPRO/APE/EvoPrompt等自优化方法的平均性能差距**不足1%**（Table 6, 7）；在多数模型×任务组合中达到最优或次优。
  - **跨模型稳定性**：该现象在Pre-trained与Instruction-tuned模型上均稳定复现，印证为ICL的普适特性。
  - **成功概率**：随机采样有**>40%** 的概率抽到优于人类基线的Separator（Table 9）。
  - **迁移性**：跨任务迁移性弱（不同任务最优Separator互换后接近随机猜测），但同任务跨上下文迁移性较好（73.4% vs 人类基线50.6%）。
  - **GSM8K推理验证**：平均随机Separator 37.8% vs ZS-CoT 38.3%；最佳随机Separator达47.3%，相对提升23%，且性能方差（9%）低于CoT（17%），鲁棒性更强。
- **最强结果**：Random with Context在Mistral 7B上AGNews达82.9%，在LLaMA2 7B上SST-2达92.7%，整体相对提升最高达25.9%。

## 相关工作脉络
- **AutoPrompt (Shin et al., 2020)**：早期通过连续嵌入层梯度发现“无意义但有效”的prompt，本文将其离散化并系统化验证，证明无需梯度优化，词表随机采样即可复现类似效果。
- **OPRO (Yang et al., 2023) / APE (Zhou et al., 2022)**：依赖LLM生成与筛选候选提示，本文证明其复杂元提示工程的边际收益有限，随机基线可作为重新评估此类方法真实贡献的参照系。
- **EvoPrompt (Guo et al., 2023)**：基于进化算法搜索Prompt，本文指出其与随机基线的最大差距仅3.4%（SubJ任务），提示优化竞争的实质门槛被低估。
- **Human-readable Prompt Tuning (Shi et al., 2022)**：主张Prompt应保持可读性，本文用实验反驳，证明表面乱码与连贯短语在分类任务上性能差异仅0.5%相对值。
- **Prompt Breeder (Fernando et al., 2023)**：全自动自我演进提示框架，本文作为强基线表明此类方法的实际增量需在随机采样参照下重新量化。

## 局限性与未来方向
- 实验主体集中于短文本分类，生成式/长程推理任务的系统性验证仍较有限（GSM8K仅为初步探索）。
- 未深入剖析随机Separator起作用的底层机制（如注意力头激活模式、Token分布偏置或内部表征对齐方式）。
- 采样预算、字符串长度与模型规模的敏感性分析较浅，缺乏自适应停止或早退准则。
- 未来可将随机基线推广至对话系统、代码生成、多模态提示及低资源语言场景，并探索将其作为冷启动种群与梯度/进化搜索结合以提升上限。

## 研究启发与可借鉴点
- **基线重估意识**：任何新提出的Prompt优化/自动化方法应优先与“词表随机采样”对比，避免将本可被简单策略覆盖的收益错误归因于复杂设计。
- **去指令模型依赖**：预训练基础模型即可支撑有效Prompt搜索，大幅降低API调用成本与指令遵循幻觉风险，适合资源受限场景。
- **广撒网+Top-K筛选范式**：随机采样在GSM8K上方差更低，提示在复杂生成任务中可采用“随机批量生成→准确率筛选”策略提升稳定性。
- **先验分布的价值**：Random w/o Context仅依赖LM边缘分布即接近Random with Context，说明无需刻意注入任务上下文，语言模型先验本身已蕴含充足优化空间。
- **与团队方向结合机会**：若团队关注低资源Prompt优化、ICL机理分析或自动化搜索，可将此随机基线作为评测标尺，或将其作为贝叶斯优化/进化算法的初始先验分布以加速收敛。

## 关键术语表
- **Separator**：置于输入末尾或输出开头的提示字符串（可为自然语言或符号），用于界定任务边界或触发特定推理/输出模式。
- **In-Context Learning (ICL)**：在不更新模型参数的情况下，通过在输入中提供少量示例或提示文本指导大模型完成下游任务的范式。
- **OPRO**：Large Language Models as Optimizers，利用LLM自身生成、评估并迭代优化Prompt的元学习框架。
- **Random Vocabulary**：完全脱离语言模型与上下文的词表均匀随机采样策略，生成无语义关联的符号串。
- **ZS-CoT**：Zero-Shot Chain-of-Thought，使用“Let’s think step by step”等通用思维链短语触发模型推理能力。
- **Meta-prompt**：用于指导LLM生成、选择或改进其他Prompt的高级指令模板。
- **离散Prompt优化**：直接在Token词表或自然语言字符串空间进行搜索，区别于连续向量空间的Soft Prompt Tuning。

## 可复现要素
- **数据集**：9个公开文本分类数据集（SST-2/5, MR, CR, MPQA, Subj, TREC, AGNews, DBPedia），每任务64训练样本；GSM8K用于推理验证。论文未提供专属代码仓库链接，但实验设置描述详尽。
- **代码/权重**：未明确声明开源代码；使用GPT2、Mistral、Llama2系列公开权重；ChatGPT通过API调用。
- **关键超参
