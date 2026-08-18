---
title: "Strings from the Library of Babel: Random Sampling as a Strong Baseline for Prompt Optimisation"
source: https://aclanthology.org/2024.naacl-long.122.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:45:59"
field: "Prompt Engineering / In-Context Learning"
keywords: ["prompt optimization", "random sampling", "separator", "in-context learning", "few-shot classification", "LLM prompting"]
innovations: ["提出三种无需LLM的随机分隔符生成策略并证明其与自优化方法性能相当", "建立强随机基线，揭示已有prompt优化方法的性能增益可能被高估", "在分类和生成任务上验证随机分隔符的鲁棒性和跨seed稳定性"]
benchmarks: ["SST-2", "SST-5", "AGNews", "GSM8K", "DBPedia", "TREC", "MR", "CR", "MPQA", "Subj"]
---

# 论文速读：Strings from the Library of Babel: Random Sampling as a Strong Baseline for Prompt Optimisation

## 一句话总结
论文提出从语言模型词表中**随机采样 token 作为 prompt 分隔符（separator）**，发现该方法在文本分类任务上的性能可与 LLM 自优化方法媲美，且相比人类手工设计的 "Answer:" 基线有 **12% 平均相对提升**。这一发现挑战了"有效 prompt 必须语义相关或可人工阅读"的常见假设，并为 prompt 优化研究建立了一个简洁而强大的新基线。

## 研究问题与动机
1. **现有方法的隐含假设值得检验**：当前 prompt 优化研究（如 OPRO、APE 等）普遍假设有效分隔符需与任务相关、语义连贯，且依赖指令微调 LLM 生成替代方案；本文质疑这一假设是否被高估。
2. **语言空间中是否存在大量高效分隔符被忽视**：若有效分隔符并非稀缺资源，则使用复杂 LLM 优化的边际收益可能被夸大。
3. **随机 prompt 是否足以构成强基线**：为后续研究提供一个低成本、易复现的对照基准，防止对新方法的性能增益做出过度归因。

## 核心贡献（创新点）
1. **提出三种无需指令微调 LLM 的随机分隔符生成策略**：从词表随机采样、从无上下文语言模型先验采样、带少量训练样本上下文采样，证明随机方法可取代复杂元提示工程。
2. **建立强基线并揭示前人工作的潜在高估**：随机方法在 9 个分类任务上平均相对提升 12%，与 OPRO/APE/EvoPrompt 等自优化方法差距 <1%，表明已有工作的真实贡献可能低于文献所述。
3. **量化语言空间中高效分隔符的丰富程度**：在 AGNews 上，随机采样有 >40% 概率得到优于人类基线 "Answer:" 的分隔符，挑战了"好 prompt 需任务相关"的认知。
4. **拓展到生成式推理任务验证普适性**：在 GSM8K 上最佳随机分隔符相比 ZS-CoT 有 23% 相对提升，且随机方法在不同 seed 间方差更低，鲁棒性更强。

## 方法详解
### 随机分隔符优化框架（Figure 2）
框架包含三步：（1）随机分隔符生成 → （2）分隔符评估 → （3）分隔符选择。

### 三种生成策略（Table 2/3）
- **Random Vocabulary**：直接从词表中均匀随机采样 token，直至达到预设长度上限；完全 context-free、task-agnostic、无需 LLM。
- **Random w/o Context**：从无上下文的语言模型先验分布（prior distribution）中采样，生成自然语言短语，仍保持 context-free、task-agnostic。
- **Random with Context**：将少量训练样本（如 3 个）作为上下文输入 LLM，引导其在任务相关的语义空间中采样分隔符。

### 评估与选择
给定训练集 $T = \{(x_i, y_i)\}_{i=1}^{n}$ 和标签→文本映射函数 $f$，对候选分隔符 $s$ 计算：
$$\hat{y}_i = \arg\max_{v \in V} P(v \mid c \oplus x_i \oplus s; \theta)$$
以分类准确率作为分隔符评分 $m$；在采样预算 $k$ 内积累 $(s_i, m_i)$ 对，选取最高分者用于测试。

### 关键超参
- 每方法生成候选数：160（随机方法）/ 40 步 × 4 候选 = 160（OPRO 类方法）
- 上下文：1-shot 演示用于 prompt 评估；Random with Context 使用 3 个随机训练样本
- 采样温度：生成阶段 1.0，分类阶段 0.0

## 实验与结果
### 数据集与模型
- **9 个分类数据集**：SST-2、SST-5、MR、CR、MPQA、Subj、TREC、AGNews、DBPedia；每数据集 64 条训练样本。
- **8 个语言模型**：GPT-2 Large (0.8B)、GPT-2 XL (1.5B)、Mistral 7B、Mistral 7B Instruct、Llama-Alpaca 7B、Llama2 7B、Llama2 7B Chat、ChatGPT (GPT-3.5 Turbo)。

### 主要结果（Table 6）
| 方法 | 平均相对提升 vs "Answer:" |
|---|---|
| Random Vocabulary | +23.4% |
| Random w/o Context | +24.4% |
| Random with Context | +25.9% |
| OPRO | +22.8% |
| OPRO-ICL | +23.0% |
| **最佳随机 vs 最佳 OPRO** | **差距 < 1%** |

### 与 EvoPrompt/APE 等基线对比（Table 7）
- Random Vocabulary 平均 73.7%，与 EvoPrompt-DE (77.1%) 差距 3.4%，最大单任务差距 2.1%（SubJ）。
- Random with Context 达 76.0%，接近 EvoPrompt-GA (75.9%)。

### 随机分隔符超越人类基线的概率（Table 9）
在 AGNews 上，Random Vocabulary 平均有 **66.1%~70.6%** 概率优于 "Answer:"，三策略平均 >40%。

### GSM8K 生成任务（Table 13）
- 最佳随机分隔符平均 47.3%，ZS-CoT 平均 38.3%，相对提升 **23%**。
- 随机方法跨 seed 方差（~9%）显著低于 ZS-CoT（~17%+），鲁棒性更好。

## 相关工作脉络
1. **AutoPrompt (Shin et al., 2020)**：通过 embedding 层梯度搜索"无意义"提示，首次揭示 unnatural prompt 的有效性；本文进一步证明**完全随机**的纯词表采样即可达到类似效果，且无需梯度信息。
2. **OPRO (Yang et al., 2023)**：用指令微调 LLM 结合元提示自我进化生成 prompt；本文证明其优势被夸大——随机采样可达同等水平，且**无需指令微调模型**。
3. **EvoPrompt (Guo et al., 2023)**：将进化算法引入 prompt 优化；本文随机基线与其差距仅 3.4%，提示 EvoPrompt 的增益部分来自搜索机制而非 prompt 本身的独特性。
4. **APE (Zhou et al., 2022)**：生成-选择-改写循环优化 prompt；本文 Random w/o Context 性能与之相当，但实现简单得多。
5. **Kubrick's The Shining (Shi et al., 2022)**：主张 prompt 需"human-readable"；本文发现最佳随机分隔符（如 `"cancell BlakesteamappsGr"`）性能与可读提示无显著差异，直接反驳此假设。
6. **PromptBreeder (Fernando et al., 2023)**：完全自动化的 prompt 进化系统；本文的简单随机方法证明其复杂机制的边际收益值得重新评估。

## 局限性与未来方向
1. **评估任务以分类为主**：生成任务仅涉及 GSM8K 数学推理，对其他生成式任务（摘要、对话、代码生成等）的泛化性未充分验证。
2. **跨任务迁移能力有限**：Table 10 显示随机分隔符在不同任务间几乎不可迁移（平均转移准确率 ~50-59%，与人类基线相当），限制了通用 prompt 的构建。
3. **未探索更复杂的随机策略**：如 conditioned 随机（按特定语言分布采样）、混合随机+语义的策略、或结合嵌入空间的引导采样。
4. **对模型规模的依赖性未系统研究**：仅覆盖了 0.8B~7B 范围，超大规模模型（如 70B+）中随机采样的相对有效性尚不明确。

## 研究启发与可借鉴点
1. **将随机采样作为标准基线**：任何新的 prompt 优化方法应报告与 Random Vocabulary/Random w/o Context 的差距，避免虚假贡献；建议团队在后续工作中强制纳入该基线。
2. **"随机性强基线+任务增强"的混合策略**：可探索以随机采样为起点，再用轻量级任务信号（如 perplexity、token overlap）做二阶筛选，兼顾搜索效率与任务适配性。
3. **重新审视 CoT 的有效性**：GSM8K 上 CoT 仅比平均随机高 ~1%，且方差更大；提示团队在引入 chain-of-thought 变体时需设置更严格的对照，避免将随机波动误认为方法增益。
4. **可迁移的 prompt 机制研究机会**：鉴于当前随机分隔符跨任务不可迁移，探索"结构化随机"（如按句法分布采样、控制信息熵）以提升跨任务通用性是一个有潜力的方向。

## 关键术语表
**Separator（分隔符）**：置于输入末尾或输出开头的特殊 token/字符串（如 "Answer:"、"Let's think step by step"），用于引导模型进入特定任务模式。

**Random Vocabulary**：直接从语言模型词表中均匀随机采样 token 组合成字符串作为分隔符，完全不依赖上下文和语义。

**Random w/o Context**：从语言模型无条件先验分布（prior distribution）中采样生成自然语言短语作为分隔符，保持 task-agnostic。

**Random with Context**：将少量训练样本作为上下文输入 LLM，引导其在任务相关语义空间中采样分隔符。

**OPRO (Optimizing Prompts with Reinforcement learning via LLMs as Optimizers)**：Yang et al. (2023) 提出的方法，用指令微调 LLM 结合元提示进行自我进化式 prompt 优化。

**In-context Learning (ICL)**：利用少量示例让预训练语言模型适应新任务的范式，本文所有实验均基于 one-shot ICL 设置。

**ZS-CoT (Zero-shot Chain-of-Thought)**：Kojima et al. (2022) 提出的零样本推理提示策略，典型形式为 "Let's think step by step"。

**Prompt Optimisation Baseline**：本文主张随机采样应成为 prompt 优化研究的标准强基线，用于校准新方法声称的性能增益。

## 可复现要素
- **数据集**：9 个公开分类数据集（SST-2/5、MR、CR、MPQA、Subj、TREC、AGNews、DBPedia），64 条训练样本；GSM8K 数学推理子集。**均已公开**。
- **代码/权重**：**论文未明确提及代码开源**；使用的模型均为公开权重（GPT-2、Mistral、Llama2 系列、ChatGPT API）。
- **关键超参**：采样预算 k=160；训练样本数 64；上下文 1-shot（评估）/ 3 个样本（Random with Context）；生成温度 1.0，分类温度 0.0；OPRO 优化步数 40、每步生成 4 候选；随机方法 5 个 random seed（ChatGPT 2 个 seed）。
