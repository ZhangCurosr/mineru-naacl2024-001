---
title: "zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models"
source: https://aclanthology.org/2024.naacl-long.104.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:44:45"
field: "时序知识图谱与零样本学习"
keywords: ["时序知识图谱", "零样本学习", "大语言模型", "关系预测", "知识图谱推理"]
innovations: ["首次系统研究 TKGF 中的零样本关系学习问题，并提出 LLM 赋能的零样本关系学习框架 zrLLM", "利用 LLM 生成关系丰富描述并提取固定语义表示，有效解决零样本关系表示缺失问题", "设计关系历史学习器（RHL）捕捉 entity-agnostic 的时序模式，促进文本与图谱空间对齐"]
benchmarks: ["ACLED-zero", "ICEWS21-zero", "ICEWS22-zero"]
---

# 论文速读：zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models

## 一句话总结
本文首次针对时序知识图谱（TKG）中的**零样本关系学习（Zero-Shot Relational Learning）**问题，提出 zrLLM 方法。该方法利用大语言模型（LLM）的语义知识为关系生成表示，并将其引入传统的嵌入式 TKF 模型，从而在**无任何图上下文**的情况下，显著提升模型对训练时未见过的（零样本）关系的预测能力，同时保持对已见关系的高效推理性能。

## 研究问题与动机
1.  **核心问题**：传统嵌入式 TKGF 模型的关系表示完全从观测到的图上下文中学习。当时序知识图谱演化并引入全新的、训练集中未出现过的“零样本关系”时，模型因缺乏历史事实支持而无法为其生成有意义的表示，导致预测能力急剧下降。
2.  **现有方法不足**：
    *   传统 TKGF 方法（如基于嵌入或规则的）均设计为闭集评估，无法处理未见过的关系。
    *   现有的 TKG 归纳学习方法多基于少样本学习，需要少量相关事实（K-shot）才能生成归纳表示，在严格的零样本设置下同样失效。
    *   最近尝试将 LLM 引入 TKG 推理的工作（如仅使用 ICL 或微调 LLM）存在性能不佳或计算成本过高的问题，且**均未专门研究如何利用 LLM 解决零样本关系推理**。
3.  **信息泄漏风险**：现有主流基准（如 ICEWS14/18）基于2020年之前的知识构建，而广泛使用的 LLM（如 T5）训练语料覆盖此时间段，直接应用存在严重的数据泄漏风险，使得评估失真。
4.  **动机**：鉴于 LLM 作为强大的语义知识库，具备从文本描述中捕获关系语义的能力，作者旨在探索一种**高效、免微调**的框架，将 LLM 的语义知识对齐到 TKGF 模型的嵌入空间，以赋能零样本关系学习。

## 核心贡献（创新点）
1.  **首创零样本 TKGF 研究**：据作者所知，本文为首个专门研究在 TKGF 场景下进行零样本关系学习的论文，填补了该领域的空白。
2.  **提出 zrLLM 框架**：设计了一个新颖的、LLM 赋能的零样本关系学习框架 zrLLM，核心在于使用 LLM 生成富含语义的关系表示，并通过关系历史学习器（RHL）促进文本空间与图谱嵌入空间的对齐。
3.  **显著的零样本性能提升**：在三个专门为零样本 TKGF 构建的新基准数据集上，zrLLM 能大幅增强多种主流嵌入式 TKGF 模型对零样本关系的预测能力（例如 CENET+ 在 ACLED-zero 上的零样本 MRR 从 0.419 提升至 0.591），且未损害其对已见关系的性能。
4.  **高效的 LLM 集成策略**：与微调大型 LLM 或仅依赖 ICL 的方法不同，zrLLM 采用固定 LLM 输出表示并与现有 TKGF 模型联合训练的策略，在性能与计算效率之间取得了良好平衡，避免了 LLM 微调的高昂成本和 ICL 的性能瓶颈。

## 方法详解
zrLLM 框架主要包含以下关键组件：
1.  **LLM 赋能的关系表示生成**：
    *   **丰富关系描述（ERD）生成**：对于数据集中提供的简短关系文本，使用 `GPT-3.5` 通过提示工程生成更详细的语义解释，组合形成 ERD。
    *   **文本到嵌入的对齐**：将 ERD 输入固定权重的 `T5-11B` 编码器。通过一个 MLP 将每个词向量映射到 TKGF 模型的嵌入维度，再使用 GRU 聚合整个序列的隐藏状态，最终得到固定的、包含语义信息的 LLM-based 关系表示 $\bar{\mathbf{h}}_r$。此表示在训练过程中被冻结，以确保语义信息不被训练数据稀释。
2.  **关系历史学习器（RHL）**：
    *   **目的**：捕捉实体间关系随时间演化的 temporal pattern（如“承认外交关系”后常跟“签署正式协议”），这种模式是 entity-agnostic 的。
    *   **训练阶段**：对于训练事实 $(s, r, o, t)$，检索 $s$ 和 $o$ 在时间 $t$ 之前的历史事实，按时间分组聚合关系，得到真实关系历史表示 $\mathbf{h}_{\mathrm{hist}}$。同时，训练一个**历史预测网络（HPN）**，它以 $\bar{\mathbf{h}}_r$ 为输入，预测关系历史 $\tilde{\mathbf{h}}_{\mathrm{hist}} = \alpha \mathbf{MLP}_{\mathrm{hist}}(\bar{\mathbf{h}}_r) + \bar{\mathbf{h}}_r$。通过 MSE 损失 $\mathcal{L}_{\mathrm{hist}}$ 迫使预测历史逼近真实历史。然后，另一个 GRU 用 $\tilde{\mathbf{h}}_{\mathrm{hist}}$ 和 $\bar{\mathbf{h}}_r$ 生成蕴含时间模式的表示 $\mathbf{h}_{\mathrm{pat}}$。
    *   **评估阶段**：由于不知道查询实体，直接使用训练好的 HPN 根据查询关系 $r_q$ 的表示 $\bar{\mathbf{h}}_{r_q}$ 推断出预测历史 $\tilde{\mathbf{h}}_{\mathrm{hist}}$，再计算模式表示 $\mathbf{h}_{\mathrm{pat}}$。
    *   **分数计算**：参考 TuckER，计算 RHL 分数 $\phi$，并将其与原始 TKGF 模型的分数 $\phi'$ 加权求和得到最终评分：$\phi_{\mathrm{total}} = \phi' + \gamma \phi$。
3.  **联合训练**：总损失函数为 $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{TKGF}} + \mathcal{L}_{\mathrm{hist}} + \eta \mathcal{L}_{\mathrm{RHL}}$。其中 $\mathcal{L}_{\mathrm{TKGF}}$ 是基于 $\phi_{\mathrm{total}}$ 的传统损失，$\mathcal{L}_{\mathrm{RHL}}$ 是直接基于 RHL 分数 $\phi$ 的二元交叉熵损失，作为辅助子任务，促进文本与图谱嵌入空间的更好对齐。

## 实验与结果
*   **数据集**：构建了三个新基准：**ACLED-zero** (2023年), **ICEWS21-zero** (2021年), **ICEWS22-zero** (2022年)，事实时间均在 T5-11B (2020) 发布之后，以避免信息泄漏。每个数据集均按频率划分 seen/unseen 关系。
*   **基线模型**：将 zrLLM 耦合于七种近期嵌入式 TKGF 模型：CyGNet, TANGO-T/D, RE-GCN, TiRGN, RETIA, CENET。
*   **主要结果**：
    *   在**ACLED-zero**上，CENET+ 的零样本 MRR 从 0.419 大幅提升至 **0.591**，Hits@1 从 0.297 提升至 **0.451**。RETIA+ 的零样本 MRR 达到 0.557。
    *   在**ICEWS21-zero**上，CENET+ 的零样本 MRR 从 0.205 大幅提升至 **0.335**，Hits@1 从 0.101 提升至 **0.162**。
    *   在**ICEWS22-zero**上，CENET+ 的零样本 MRR 从 0.270 大幅提升至 **0.564**，Hits@1 从 0.134 提升至 **0.432**。
    *   大多数情况下，zrLLM 在提升零样本性能的同时，**也维持或改善了**已见关系的性能。
*   **对比 LM 增强方法**：zrLLM 增强的模型（如 CENET+, MRR=0.395 on ICEWS21-zero overall）优于微调 BERT 的 PPT (MRR=0.268) 和基于 ICL 的方法 (MRR=0.177)。
*   **消融实验**：移除 ERD（直接用原始文本）或移除 RHL 模块均会导致性能下降，证明了各组件的有效性。使用更小的 T5-3B 也会导致性能降低，说明更大 LLM 提供的语义信息质量更高。

## 相关工作脉络
1.  **传统 TKGF 方法**（如 CyGNet, TANGO, RE-GCN 等）：基于封闭集假设，关系表示从图上下文学习，无法泛化至零样本关系。zrLLM 通过引入外部语义知识突破此限制。
2.  **TKG 归纳学习/少样本方法**（如 OAT, MOST, FITCARL）：依赖少量（K-shot）支持样本生成归纳表示。zrLLM 解决了完全无样本（zero-shot）的场景，且无需推理时的支持集。
3.  **LLM for TKG 推理**：SST-BERT 在 TKG 语料上预训练小模型，侧重 unseen entities；PPT 微调 BERT；GenTKG 微调 LLaMA。zrLLM 不微调 LLM，而是将其作为固定语义源，集成到非 LLM 基线中，兼顾性能与效率。
4.  **ICL for TKG**（如 Lee et al., 2023）：仅靠 in-context learning 的 LLM 性能远逊于传统方法及 zrLLM，表明缺乏显式对齐时 LLM 难以直接完成结构化预测任务。
5.  **规则基 TKGF**（如 TLogic）：擅长处理 seen relations 下的 unseen entities，但规则受限于观测关系，同样无法处理 unseen relations。zrLLM 当前仅适用于嵌入方法。

## 局限性与未来方向
1.  **适用范围限制**：zrLLM 目前仅针对嵌入式的 TKGF 方法设计，无法直接应用于基于规则的 TKGF 模型。
2.  **计算开销增加**：引入 RHL 模块（涉及 GRU 和潜在的历史事实搜索）增加了模型的训练和评估时间，以及 GPU 内存消耗。
3.  **未来方向**：作者计划探索将 zrLLM 推广到规则基方法，并尝试优化模型效率。同时，希望在更多 TKGF 方法上验证其通用性，并研究如何从 LLM 中提取更多益处。

## 研究启发与可借鉴点
1.  **LLM 作为静态语义先验**：将预训练 LLM 的表征（经轻量适配器如对齐层）作为固定、不可微的语义先验注入下游任务模型，是一种避免微调成本、防止灾难性遗忘且能有效利用外部知识的高效范式，可迁移至其他需要语义引导的结构化预测任务。
2.  **解耦语义与结构化学习**：本研究清晰地分离了“关系语义理解”（由 LLM 负责）和“时序图推理”（由 TKGF 模型负责）两个模块，并通过可微的对齐层和辅助任务进行联合训练。这种模块化设计思想清晰，易于理解和扩展。
3.  **构建无泄漏基准的重要性**：针对 LLM 应用研究，构建事实时间晚于模型知识截止日期的基准（如本论文的 ICEWS21/22-zero）是确保评估可靠性、排除信息泄漏的必要步骤，这一严谨的研究设计值得在相关工作中效仿。
4.  **利用 entity-agnostic 的模式表示**：RHL 中设计的 HPN 和模式表示 $\mathbf{h}_{\mathrm{pat}}$ 捕捉的是脱离具体实体的关系演变模式，这种“模式知识”具有更强的泛化性，其设计思路可为学习跨实体、跨时间的通用动态规律提供参考。
5.  **辅助任务促进对齐**：使用与主任务相关的辅助损失（如 $\mathcal{L}_{\mathrm{RHL}}$）来引导表示空间的对齐，是一种常见的正则化与表征学习技巧，可借鉴用于其他多模态或跨空间表示融合的场景。

## 关键术语表
**Zero-Shot TKGF**：在时序知识图谱链接预测中，预测涉及训练集中从未出现过的（未见）关系的未来事实的任务。
**Enriched Relation Description (ERD)**：利用大语言模型基于原始简短关系文本生成的、包含更丰富语义细节的关系描述文本。
**LLM-empowered Relation Representation**：将 ERD 输入固定权重的编码器（如 T5）后，经过对齐模块得到的、包含语义信息的固定关系表示向量。
**Relation History Learner (RHL)**：zrLLM 的核心模块，旨在学习实体间关系随时间演化的通用 temporal patterns，其包含一个用于预测关系历史的历史预测网络（HPN）。
**Information Leak**：指 LLM 在预训练阶段可能已经接触过测试数据中的事实，从而在非公平条件下“记住”答案，而非真正学会推理。
**Time-Aware Filtering**：在评估 TKGF 模型时，过滤掉候选实体中包含测试时间点之后才出现的事实，以确保评估的公平性。
**Inductive Learning on TKGs**：使模型能够处理训练数据中未见过的关系或实体的能力，与 Transductive Learning 相对。
**TuckER**：一种基于张量分解的知识图谱补全模型，本文 RHL 的打分机制借鉴了其形式。

## 可复现要素
*   **代码**：已开源，网址为 https://github.com/ZifengDing/zrLLM
*   **数据集**：作者构建了三个新的零样本 TKGF 基准（ACLED-zero, ICEWS21-zero, ICEWS22-zero），代码仓库中应包含这些数据或构建脚本。
*   **预训练权重/LLM 表示**：论文使用固定的 GPT-3.5（通过 API）和 T5-11B 来生成关系表示。未提供预计算的 $\bar{\mathbf{H}}_r$ 权重文件，需要复现者自行运行文本生成和编码步骤。
*   **关键超参数**：历史长度（4, 6, 10）、嵌入维度（100, 200）、系数 $\alpha$, $\gamma$, $\eta$ 等在附录中有详细的搜索范围，最佳值见 Table 12。
*   **硬件**：单张 NVIDIA A40 48GB 显存。
