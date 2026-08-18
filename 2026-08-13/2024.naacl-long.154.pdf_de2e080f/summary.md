---
title: "TriSum: Learning Summarization Ability from Large Language Models with Structured Rationale"
source: https://aclanthology.org/2024.naacl-long.154.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:28:03"
field: "抽象式文本摘要"
keywords: ["文本摘要", "知识蒸馏", "大模型", "可解释性", "课程学习"]
innovations: ["首次将LLM结构化推理链（方面-三元组）蒸馏到小模型摘要任务", "双评分机制筛选高质量原理（语义相似度+LDA主题一致性）", "三阶段课程学习实现从依赖LLM到自主生成的平滑过渡"]
benchmarks: ["CNN/DailyMail", "XSum", "ClinicalTrial"]
---

# 论文速读：TriSum: Learning Summarization Ability from Large Language Models with Structured Rationale

## 一句话总结
TriSum提出了一种三阶段框架，通过将大型语言模型（LLM）的文本摘要能力蒸馏到小型本地模型中，利用方面-三元组结构化原理（aspect-triple rationales）和课程学习策略，显著提升小模型的摘要性能与可解释性。

## 研究问题与动机
- **LLM部署瓶颈**：GPT-3.5/4等大模型参数量庞大、计算成本高，难以在资源受限环境部署；同时用户需上传敏感数据至云端API，存在隐私泄露风险。
- **现有蒸馏方法的局限**：已有工作（如Wang et al. 2021; Liu et al. 2023）仅将LLM生成的摘要作为标签训练本地模型，未充分传递LLM的"推理思维链"，导致能力迁移不完整。
- **抽象式摘要的可解释性缺失**：传统Seq2Seq摘要模型为黑盒，无法追溯生成过程中的关键实体、关系与主题；而LLM的逐步推理能力尚未在摘要任务中被系统挖掘。
- **空白领域**：理由蒸馏已在QA、NLU、算术推理等任务中验证有效（Wang et al. 2022; Hsieh et al. 2023），但**抽象式文本摘要**的 rationale distillation 仍属空白。

## 核心贡献（创新点）
1. **首次将LLM的结构化推理过程蒸馏至摘要任务**：提出 aspect-triple rationale 生成框架，让LLM输出"核心方面→实体关系三元组→摘要"的三步推理链，填补了抽象式摘要中 rationale distillation 的研究空白。
2. **双评分金色原理选择机制**：设计基于语义相似度的 Summary Score 和基于LDA主题分布的 Coherence Score，联合筛选高质量训练样本，避免低质LLM输出污染小模型训练。
3. **三阶段课程学习训练策略**：从单任务学习（Singular）→并发学习（Concurrent: LLM引导→自引导）→联合学习（Joint），逐步让小模型从依赖LLM原理过渡到自主生成原理与摘要。
4. **双向能力验证**：不仅小模型从LLM蒸馏中受益，TriSum生成的结构化原理还能反向增强GPT-3.5 zero-shot摘要性能（ROUGE提升40.9%），证明其通用价值。

## 方法详解
TriSum包含三个步骤：

**Step 1: LLM Rationale Probing**
- 使用模板提示 GPT-3.5，给定文档 D 和人工摘要 S_gt，要求LLM逐轮生成 n 对候选原理-摘要 {R_i, S_i}。
- 原理 R = (A, T)，其中 A 为方面列表，T 为三元组集合（格式：[ENTITY1 | RELATION | ENTITY2]）。
- 生成公式：$p(R|D, S_{gt}) = \prod p(r_i|D, S_{gt}, R^{<i})$，$p(S|D, S_{gt}, R) = \prod p(s_i|D, S_{gt}, R, S^{<i})$。

**Step 2: Golden Rationale Selection**
- **Summary Score**（公式2）：$V_i^S = \text{sim}\langle\hat{S}_i, \hat{S}_{gt}\rangle + \phi_\alpha \cdot \text{sim}\langle\hat{S}_i, \hat{R}_i\rangle$，平衡摘要-参考相似度与摘要-原理相关性，防止LLM偷懒重复参考摘要。
- **Coherence Score**（公式3）：基于LDA主题分布计算 KL 散度，$V_i^C = KL(p_{LDA}^D||p_{i,LDA}^A) - (1+\phi_\beta) \cdot KL(p_{LDA}^D||p_{i,LDA}^R)$，鼓励三元组在方面基础上进一步提升主题一致性。
- 最终选择：$R_* = \arg\max_i (V_i^S + \lambda_{cs} \cdot V_i^C)$。

**Step 3: Curriculum Learning**
- **Singular-task Learning**：分别训练 Aspect Extraction、Triple Extraction、Summary Generation 三个子任务，损失函数为 $\mathcal{L}_A, \mathcal{L}_T, \mathcal{L}_S$。
- **Concurrent Learning - Early Stage**：使用LLM提供的 $A_*, T_*$ 作为教师信号（teacher forcing），联合训练三个任务。
- **Concurrent Learning - Late Stage**：切换到模型自生成中间结果（greedy decoding 得到 $\tilde{A}, \tilde{T}$），减少LLM依赖。
- **Joint Learning**：将三个decode过程合并为一个，使用 prefix token `RatGen` 联合生成原理与摘要，损失函数：$\mathcal{L}_{joint} = -\sum[\lambda_R \log p(R_*|D) + \lambda_S \log p(S_{gt}|D, \tilde{R})]$。

## 实验与结果
**数据集**：CNN/DailyMail（28.7万训练）、XSum（20.4万）、ClinicalTrial（16.3万自建数据集）。

**评估指标**：ROUGE-1/2/L、BERTScore、BARTScore。

**主要结果**（TriSum-J vs 最强基线）：
- CNN/DailyMail：ROUGE-1 45.7, R-2 22.7, R-L 41.9，综合提升 **+4.5%**（vs BART-Large）。
- XSum：R-1 47.3, R-2 24.4, R-L 39.0，综合提升 **+8.5%**。
- ClinicalTrial：R-1 45.3, R-2 24.6, R-L 35.2，综合提升 **+7.4%**。

**关键发现**：
- TriSum-S（单任务阶段）因参数量三倍放大表现稳定，TriSum-J（联合阶段）达到最优。
- 课程学习消融：跳过前置阶段直接Joint Learning会导致性能下降（ROUGE-L 38.4 vs 41.9）。
- LDA主题数选择敏感：50或5000均显著劣于200/300，印证质量筛选的重要性。
- TriSum原理可反向增强GPT-3.5 zero-shot：CNNDM上ROUGE-1达46.7，超过所有微调基线。

## 相关工作脉络
1. **LLM摘要增强**：Wang et al. (2021) 用LLM生成headline标签，Liu et al. (2023) 用LLM摘要作为benchmark——两者仅利用LLM的输出结果，未传递推理过程；TriSum进一步提取结构化原理。
2. **Rationale蒸馏**：Wang et al. (2022) Pinto、Hsieh et al. (2023) Distill、Magister et al. (2023) 等在QA/NLU/算术推理中验证，但**抽象式摘要**未被覆盖，本文填补该空白。
3. **知识蒸馏到小模型**：DistilBERT (Sanh et al. 2019)、TinyBERT (Jiao et al. 2019) 聚焦NLU，本文聚焦生成式摘要任务。
4. **课程学习**：Xu et al. (2020) 用于NLU，Bengio et al. (2009) 奠定基础理论；本文将其适配到多任务联合生成场景。
5. **可解释性方法**：Ribeiro et al. (2016) LIME、Doshi-Velez & Kim (2017) 提出事后解释框架；TriSum通过原理生成实现**过程级可解释**。

## 局限性与未来方向
- **依赖LLM质量**：若LLM输出存在偏差或事实错误，会传递到小模型；可探索多LLM集成或人工校验机制。
- **原理信息损失**：方面-三元组结构无法捕捉原文所有细节，部分语义可能在蒸馏中丢失。
- **过拟合风险**：小模型可能过度依赖LLM生成的原理，泛化到未见数据时性能下降。
- **LDA主题数敏感**：需根据数据集调节，缺乏自动调优方案。
- **潜在滥用**：可解释性提升可能让用户过度信任模型输出，需配合使用指南。

## 研究启发与可借鉴点
1. **结构化原理设计可迁移**：方面-三元组格式适用于其他需要可解释性的生成任务（如问答、翻译），可探索领域自适应格式。
2. **双评分筛选机制**：Summary Score + Coherence Score 的组合思路可推广到其他LLM蒸馏场景，平衡内容质量与结构一致性。
3. **课程学习阶段划分**：从teacher forcing到greedy decoding的平滑过渡策略，对训练多任务联合生成模型具有通用参考价值。
4. **双向增强验证**：小模型蒸馏后反向增强LLM的思路新颖，可拓展到rag、agent等场景验证原理质量。
5. **临床等垂直领域应用**：自建ClinicalTrial数据集证明方法在专业领域有效，可延伸至法律、医疗文档摘要等高精度场景。

## 关键术语表
**Aspect-triple Rationale**：由核心方面（aspect）和实体关系三元组（triples）组成的结构化推理链，用于解释摘要生成过程。
**Golden Rationale**：通过双评分机制筛选出的高质量LLM生成原理，作为小模型训练的监督信号。
**Curriculum Learning**：从简单任务逐步过渡到复杂任务的训练策略，本文分为单任务、并发、联合三阶段。
**LDA (Latent Dirichlet Allocation)**：主题建模算法，用于计算文档与原理的主题分布一致性，辅助筛选 coherent 原理。
**Teacher Forcing**：训练时用真实标签（此处为LLM输出的原理）作为后续token的输入，而非模型自生成结果。
**BARTScore**：基于预训练BART模型计算的语义相似度指标，比ROUGE更能捕捉语义等价性。

## 可复现要素
- **数据集**：CNN/DailyMail v3.0.0、XSum 公开；ClinicalTrial 自建数据集未公开（论文提供了构建流程与统计）。
- **代码**：论文未提及代码开源状态，需关注作者主页。
- **权重**：使用 BART-Large checkpoint，未提供TriSum微调后权重。
- **关键超参**：$\phi_\alpha=0.6, \phi_\beta=1.3, \lambda_{cs}=1.5, \lambda_R=0.8, \lambda_S=1.2$；LDA主题数：CNNDM=200, XSum=500, ClinicalTrial=300；LLM迭代次数 n：CNNDM=15, XSum/ClinicalTrial=8。
- **硬件**：NVIDIA RTX A6000 GPU，训练3个epoch。
