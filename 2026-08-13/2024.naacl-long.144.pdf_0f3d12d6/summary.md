---
title: "Unlocking Emergent Modularity in Large Language Models"
source: https://aclanthology.org/2024.naacl-long.144.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:47:29"
field: "大语言模型模块化与泛化"
keywords: ["Emergent Modularity", "Mixture-of-Experts", "Large Language Models", "Domain Generalization", "Parameter-Efficient Tuning", "Negative Transfer"]
innovations: ["零参数涌现模块化外部化方法EMoE", "揭示EMoE通过训练阶段掩码负迁移神经元提升泛化", "微调后可合并回标准架构保持部署兼容"]
benchmarks: ["GLUE", "GLUE-X", "Domainbed", "MMLU", "ATTEMPT Multi-task"]
---

# 论文速读：Unlocking Emergent Modularity in Large Language Models

## 一句话总结
本文提出EMoE（Emergent Mixture-of-Experts）方法，通过将预训练Transformer的FFN层**按涌现模块化（EM）外部化为稀疏MoE结构**，在**不引入任何额外可训练参数**的前提下，利用已有EM提升模型在下游域内（ID）和域外（OOD）任务上的泛化性能，并验证其在Llama2-7B/30B等大模型上的可扩展性。

## 研究问题与动机
1. **模块化优势与显式设计的局限**：模块化神经网络（MNNs）如MoE具有强适应性、数据效率和泛化能力，但现有方法多为**预定义显式模块化结构**，需额外参数与训练成本。
2. **涌现模块化（EM）被锁定**：预训练Transformer的FFN层已自发形成功能模块化（如神经元按任务共激活），但标准预训练‑微调范式将其视为**单体模型**，EM潜力未被利用。
3. **缺乏零参数负担的解锁机制**：如何将EM**外部化为可微调的MoE**，同时避免引入额外参数、训练步骤或部署变更，是提升下游泛化且兼顾实用性的关键挑战。
4. **负迁移抑制需求**：多任务或跨域微调中，部分神经元可能携带**负迁移知识**，需机制选择性屏蔽以改善参数更新。

## 核心贡献（创新点）
1. **提出EMoE框架**：将预训练FFN层按键向量聚类拆分为多个专家FFN，并用**专家键均值构建固定门控**，实现EM的零参数外部化。
2. **发现EMoE改善微调而非推理**：证明EMoE的性能增益源于**细 tuning 阶段对负迁移神经元的掩码**，微调后可将专家合并回标准FFN，**部署架构不变**。
3. **验证跨尺度泛化提升**：在BERT‑Large、GPT2‑XL的GLUE/GLUE‑X上，EMoE‑LoRA较 vanilla LoRA **最高提升+0.92%（ID）和−0.73（OOD秩）**；在Llama2‑7B/30B的MMLU上**提升+0.62~+0.93**。
4. **揭示clustering‑based expert construction的必要性**：随机拆分专家无效，**键向量相似性聚类**是捕获功能模块化的关键。
5. **多任务学习显著受益**：在ATTEMPT多任务设置中，EMoE在ID任务**最高提升+7.56**，OOD任务**最高提升+1.58**，证明其对负迁移的抑制作用。

## 方法详解
### 1. FFN作为键值记忆
Transformer的FFN层可写作 $\mathbf{y} = \sigma(\mathbf{x}\mathbf{K})\mathbf{V} = \sum_{i=1}^{h} \sigma(\mathbf{x}\mathbf{K}_{:,i})\mathbf{V}_{i,:}$，其中**键向量列** $\mathbf{K}_{:,i}$ 决定激活，**值向量行** $\mathbf{V}_{i,:}$ 贡献输出，每个$(K,V)$对视为一个**神经元**。

### 2. 基于聚类的专家构建
对选定FFN层的键矩阵$\mathbf{K}\in\mathbb{R}^{h\times d}$，使用**约束聚类**（balanced k‑means）将$d$个键向量均分为$N$组$E_i$，第$i$个专家由组内所有神经元组成：$\mathrm{FFN}^i(\cdot;\mathbf{K}^i,\mathbf{V}^i)$，**总参数量与原FFN相同**。

### 3. Avg‑k门控（零参数门控）
第$i$个专家的**门控权重**为其组内键向量的均值：$\mathbf{G}_{:,i}=\mathrm{Avg}(\mathbf{K}^i,\dim=0)$。输入$\mathbf{x}$的门控得分为$\mathbf{x}\cdot\mathbf{G}_{:,i}=\frac{N}{d}\sum_{j\in E_i}\mathbf{x}\mathbf{K}_{:,j}$（即组内神经元激活分数的平均）。采用**Top‑k**选择得分最高的$k$个专家，门控输出为0/1硬选择。

### 4. 微调与合并
微调时**门控权重与FFN参数绑定**（不单独优化），仅通过LoRA或全参更新专家内部的$\mathbf{K}^i,\mathbf{V}^i$。**微调完成后可将专家加权合并回单体FFN**：$\mathbf{K}=\sum_i\mathbf{K}^i,\mathbf{V}=\sum_i\mathbf{V}^i$，部署时恢复为标准架构，无额外开销。

### 5. 与GMoE/EMoE‑learn对比
- **GMoE**：复制原FFN形成多个专家，需训练新门控，引入额外参数。
- **EMoE‑learn**：门控可学习，但稳定性差；本文证明**固定avg‑k门控更优**。

## 实验与结果
### 数据集与基准
- **NLP**：GLUE（ID）、GLUE‑X（13个OOD任务，用Friedman秩评价）、MMLU（大模型指令微调后评估）。
- **视觉OOD**：Domainbed（PACS、VLCS、Office‑Home、Terra Incognita）。
- **多任务**：ATTEMPT设定（6个SuperGLUE任务ID，2个NLI训练→4个NLI OOD测试）。

### 主要结果
| 模型 | 方法 | ID Avg（GLUE） | OOD Rank（GLUE‑X） | MMLU（Llama2‑7B） |
|------|------|----------------|---------------------|-------------------|
| BERT‑Large | LoRA | 84.35 | 4.86 | – |
| BERT‑Large | **EMoE** | **85.09（+0.74）** | **4.37（↓0.49）** | – |
| GPT2‑XL | LoRA | 84.61 | 5.61 | – |
| GPT2‑XL | **EMoE** | **85.45（+0.84）** | **3.88（↓1.73）** | – |
| Llama2‑7B | LoRA | – | – | 46.96 |
| Llama2‑7B | **EMoE（N=64,k=16）** | – | – | **47.58（+0.62）** |
| Llama‑30B | **EMoE（N=256,k=64）** | – | – | **57.11（+0.93）** |

- **Vision OOD**：EMoE在Domainbed上**与GMoE相当**（ViT‑Base Leave‑one‑domain‑out平均74.65 vs GMoE 74.28）。
- **多任务**：ID最高**+7.56**（T5‑Base, N=32,k=8），OOD最高**+1.58**。
- **全参微调**：EMoE同样稳定提升，Llama2‑7B全参微调MMLU达**48.08**（vs baseline 46.50）。

### 关键结论
1. **零参数额外**：EMoE不增加可训练参数量，计算开销<5%。
2. **鲁棒性**：$top\text{-}k/N\in\{0.25,0.5\}$均有效，专家数$N$可从16扩展到256。
3. **训练阶段增益**：EMoE2LoRA（微调后合并）性能与EMoE一致，证明增益来自**微调过程**而非推理。

## 相关工作脉络
1. **GMoE（Li et al., 2023）**：复制FFN构建MoE，引入可学习门控，侧重OOD泛化理论，但**增加参数**且需微调整个Transformer块。
2. **MoEfication（Zhang et al., 2022b）**：将FFN稀疏化为MoE以提升**推理效率**，未探索EM对微调性能的影响；本文借其聚类思想但目标不同。
3. **Emergent Modularity发现（Zhang et al., 2023; Li et al., 2022）**：揭示预训练Transformer FFN的**稀疏激活与功能聚类**，但未提出下游利用方法。
4. **Sparse Upcycling（Komatsuzaki et al., 2023）**：从密集检查点训练MoE，需**额外训练资源**；EMoE直接利用已有EM，零额外成本。
5. **Parameter‑efficient Tuning（LoRA等）**：本文在LoRA框架内集成EMoE，证明**无需修改PEFT结构**即可受益。

## 局限性与未来方向
1. **分解算法依赖**：当前采用MoEfication的简单聚类，**未探索更先进的模块化挖掘算法**（如可学习注意力头选择）。
2. **任务范围局限**：未在**数学推理、代码生成**等挑战性任务上验证，仅覆盖NLP与视觉OOD。
3. **大模型 scaling 未充分**：虽验证至Llama‑30B，但**万亿参数模型**的EM结构与利用方式尚不明确。
4. **门控静态性**：avg‑k门控固定，可能无法适应**动态任务分布**；可学习门控（EMoE‑learn）稳定性较差，需权衡。
5. **合并策略简化**：微调后合并专家为简单加权平均，**未优化合并权重**以进一步恢复性能。

## 研究启发与可借鉴点
1. **涌现模块化作为内置资源**：预训练模型已蕴含功能模块化，**无需额外预训练**即可提取，为低资源场景提供新范式。
2. **零参数外部化技术**：键向量聚类+均值门控的实现极简，可迁移至**其他模块**（如attention heads）的模块化解锁。
3. **训练‑推理解耦设计**：微调时利用MoE结构抑制负迁移，部署时合并为单体，**兼顾性能与工程实用性**。
4. **负迁移掩码机制**：Top‑k选择激活最高专家等价于**屏蔽负迁移神经元**，为多任务学习提供新正则化思路。
5. **超参数鲁棒性**：$top\text{-}k/N$比例比绝对值更重要，**0.25~0.5范围普适**，降低调参成本。

## 关键术语表
- **涌现模块化（Emergent Modularity, EM）**：预训练Transformer中自发形成的、按功能分组神经元共激活的模式。
- **混合专家（Mixture‑of‑Experts, MoE）**：通过门控机制动态选择子网络（专家）处理的架构，实现条件计算。
- **键值记忆（Key‑Value Memory）**：将FFN层解释为键查询‑值输出的记忆系统，键决定激活，值贡献输出。
- **Avg‑k门控（Avg‑k Gating）**：使用专家键向量均值作为固定门控权重，Top‑k选择激活专家，**无额外参数**。
- **负迁移（Negative Transfer）**：多任务学习中，某一任务的知识对其他任务产生有害影响。
- **参数高效微调（Parameter‑efficient Tuning, PET）**：仅更新少量参数（如LoRA）即可适配下游任务的技术。
- **约束聚类（Constrained Clustering）**：在聚类过程中加入必须链接/不能链接约束，保证簇均衡与语义一致性。
- **域外泛化（Out‑of‑Domain Generalization）**：模型在未见过分布的数据上的表现能力。

## 可复现要素
- **代码**：论文声明开源（"Code is available at this repo"），但未在正文提供链接；需查阅ACL Anthology页面获取。
- **数据集**：全部公开（GLUE、GLUE‑X、Domainbed、MMLU、Alpaca、SuperGLUE）。
- **关键超参**：
  - 专家数$N$：64（小模型）/256（Llama‑30B）
  - top‑k：16、32、48（$top\text{-}k/N=0.25\sim0.5$）
  - LoRA：rank=8, alpha=16, learning rate=$2\text{e-}4\sim5\text{e-}4$
  - 替换层：最后两个偶数层（GPT2‑XL）或最后两个even layers（BERT）
- **硬件**：单卡A100‑40G（小模型）/八卡A100‑80G（大模型）。
