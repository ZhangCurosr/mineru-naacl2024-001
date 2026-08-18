---
title: "An-Examination-of-the-Compositionality-of-Large-Generative-V"
source: https://aclanthology.org/2024.naacl-long.39.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:31:35"
field: "视觉语言模型评估"
keywords: ["Vision-Language Model", "Compositionality", "Evaluation Benchmark", "Syntactical Bias", "VisualGPTScore", "Large Language Model"]
innovations: ["首次系统揭示GVLM组合推理评估中的语法偏见并提出量化指标SyntaxBias Score", "构建去偏基准SADE，包含Content Challenge等新任务以消除句法偏好", "在多模态组合推理benchmark上系统评估7个GVLM并揭示架构差异"]
benchmarks: ["SADE", "Winoground", "VL-CheckList", "ARO", "CREPE"]
---

# 论文速读：An-Examination-of-the-Compositionality-of-Large-Generative-V

## 一句话总结
本文发现当前多模态组合推理benchmark（如Winoground、ARO、VL-CheckList等）存在显著的语法偏见，使GVLMs能够利用其语言模型的语法能力而非真正视觉理解能力获得高分；作者据此提出定义语法偏见的量化指标**SyntaxBias Score**，并构建去偏后的**SADE benchmark**以实现对GVLMs更公正的组合推理评估。

## 研究问题与动机
1. **GVLM组合推理评估缺乏深入理解**：现有工作集中于EVLM（如CLIP）的多模态组合推理，而LLM驱动的新型生成式VLM（GVLM）的compositionality能力仍是"黑箱"。
2. **现有benchmark基于EVLM设计，不适用于GVLM**：主流benchmark（Winoground、ARO、VL-CheckList、CREPE等）通过句法扰动（shuffle、swap、negate等）构造hard negatives，但这类方法利用了GVLM中LLM部分的强语法感知能力。
3. **VisualGPTScore作为评估指标存在缺陷**：当前使用VisualGPTScore（基于log-likelihood）评价GVLM，该指标偏好句法正确的句子而非内容相关正确的句子，导致正向样本得分被高估。
4. **缺乏无偏的评估基准**：亟需一个能够消除句法偏差、真正评估GVLMs跨模态内容理解能力的benchmark。

## 核心贡献（创新点）
1. **首次系统揭示GVLM组合推理评估中的语法偏见现象**：通过对比实验证明VisualGPTScore对bags-of-words不敏感且偏好句法正确性，与EVLM采用的similarity score形成互补但不同的短路径策略。
2. **提出SyntaxBias Score定量度量语法偏见**：利用LLM（Vicuna-13B）计算正/负样本的生成分数差值，实现对该偏见的数值化刻画，为后续去偏提供依据。
3. **构建SADE（SyntActically DE-biased benchmark）基准**：基于现有benchmark，通过Filtering策略（保留句法相似度高的样本）和新增Content Challenge任务（仅保留名词/形容词短语作为正样本、随机抽取流畅负样本），实现多维度去偏评估。
4. **提供首个无偏GVLMs组合推理性能报告**：在SADE上测试7个GVLMs，发现InstructBLIP和Emu整体表现最优，但Emu在hard negative下鲁棒性较差，揭示了GVLMs真实的多模态组合能力差异。

## 方法详解
1. **视觉语言模型形式化定义**：
   - GVLM将图像I经视觉编码器$g(I)$映射为视觉token序列$\boldsymbol{z} = \mathbf{M}(g(I))$，并通过自回归方式生成文本：
     $\max_{\theta_M, \theta_\sigma} \sum_{i=1}^{K} \log P(t_i | \boldsymbol{p}, \boldsymbol{z}, t_1, ..., t_{i-1}; \theta_M, \theta_\sigma)$
   - 其中$\theta_M$为映射层参数，$\theta_\sigma$为adapter/LoRA等可训练参数。

2. **VisualGPTScore评估指标**：
   - $Score(r|I) = \sum_{t=1}^{m} w_t \log P(r_t | r_{<t}, p, I; \theta_{GVLM})$，即给定图像时直接计算参考句子的条件对数似然。

3. **SyntaxBias Score定义**（关键公式）：
   - $Score_{SyntaxBias} = \Delta(\sum_i w_i \log P(p_i|p_{<i}; \theta) - \sum_j \hat{w}_j \log P(n_j|n_{<j}; \theta))$
   - 使用Vicuna-13B-v1.5计算正/负文本的语言模型对数概率差，$\Delta$表示归一化操作。

4. **去偏策略**：
   - Winoground：全部保留（其positive/negative均为流畅有意义的caption）。
   - Relation/Attribute：过滤SyntaxBias Score接近零的样本（$p$值$< 1e^{-5}$）。
   - Atomic/Negate：同样采用过滤策略。
   - Content Challenge：新挑战——仅保留名词/形容词短语作为正样本，从COCO/Flickr30K随机抽取流畅负样本，强制模型依赖视觉内容理解。

## 实验与结果
- **数据集**：Winoground (400对)、VL-CheckList (11,051 items)、ARO/Visual Genome (7,521 items)、CREPE (23,304 items)，总计约52,189张图像和129,558个参考句子。
- **评估模型**：LLaVA-7B、LLaVA-13B、MiniGPT-4-7B、mPLUG-Owl、InstructBLIP、LLaMA-Adapter V2、Emu。
- **主要发现**：
  1. **语法偏见显著**：在Flickr30K实验中，LLaVA-7B对Case 1（shuffle negatives）召回率达98.62%，但在Case 2（随机流畅负样本）下降31.56%，Case 3（句法不正确正样本）降至27.02%；而CLIP不受句法影响。
  2. **噪声图像实验**：CLIP在随机噪声图像上准确率骤降至随机水平，LLaVA几乎不受影响，说明GVLM可仅凭文本线索作答。
  3. **SADE基准结果**（Table 2）：InstructBLIP综合表现最优（Relation 73.87%、Attribute 79.39%、Atomic 44.37%、Negate 66.84%、Content 57.83%）；Emu在Attribute（85.84%）和Negate（87.20%）表现强，但在Comprehensive（4.00%）和Content（2.79%）极差。
  4. **人类评估验证**（Table 1）：SADE处理后，正/负样本合理性评分趋近于0，证明偏见显著降低。

## 相关工作脉络
1. **CLIP/ITC范式**（Radford et al., 2021）：对比学习预训练的EVLM，依赖similarity score评估，受bags-of-words问题影响。
2. **Winoground**（Thrush et al., 2022）：首个VLM组合推理benchmark，通过交换实体/属性构造负样本，作者认为其句式结构平衡故无需去偏。
3. **ARO/Yuksekgonul et al. (2022)**：探究VLM在对象关系与属性理解上的局限，发现EVLM的bag-of-words倾向。
4. **VL-CheckList/Zhao et al. (2022)**：系统评估VLM对物体、属性、关系理解，采用句法扰动构造hard negatives。
5. **CREPE/Ma et al. (2023)**：通过atom替换和negate构造测试集，但作者指出其构造的正负样本存在显著句法差异。
6. **SugarCrepe/Hsieh et al. (2024)**：使用Vera和TextAttack检测正负样本的plausibility/grammar gap，再用ChatGPT生成合理hard negatives——与本文思路相近但方法不同（本文直接过滤而非重新生成）。

## 局限性与未来方向
1. **SADE仅评估组合推理，不能全面反映GVLMs能力**：论文明确承认该benchmark无法衡量emergent capability、复杂推理等其他维度。
2. **数据集规模相对较小**：由于去偏筛选后样本减少，限制了基准的泛化能力。
3. **未来方向**：可扩展SADE至更多GVLM架构、增加多语言/多模态变体、探索更丰富的去偏策略（如对抗生成hard negatives）、结合细粒度视觉理解任务。

## 研究启发与可借鉴点
1. **评估指标敏感性分析值得借鉴**：论文对VisualGPTScore进行系统的敏感度测试（bags-of-words、句法、内容三个维度），这种"先审视指标再使用"的思路可作为其他评估研究的范例。
2. **去偏策略的可迁移性**：基于统计检验（p值阈值）和人类评估的双重验证方法，可应用于其他生成模型（如LLM、文本生成）的benchmark去偏。
3. **Content Challenge的新颖性**：仅保留名词/形容词短语构成正样本的设计，是对现有benchmark过度依赖句法的有力挑战，为后续研究提供了新思路。
4. **多模型横向对比的价值**：在同一去偏基准上测试7个GVLM，发现InstructBLIP最稳健而Emu最脆弱，揭示了不同架构在组合推理上的本质差异。

## 关键术语表
**GVLM**（Generative Vision-Language Model）：将冻结的视觉编码器通过映射层接入大语言模型，利用自回归生成能力进行多模态理解的模型架构。

**EVLM**（Encoder-based Vision-Language Model）：基于对比学习（ITC）或匹配（ITM）的训练范式，如CLIP、ViLT等，通过计算图像-文本相似度进行推理。

**VisualGPTScore**：基于GVLM直接生成参考句子的条件对数似然得分，替代传统similarity score用于评估GVLMs的图像理解能力。

**SyntaxBias Score**：用LLM计算的参考句子正/负样本生成分数差值，用于量化benchmark中的语法偏见程度。

**SADE**（SyntActically DE-biased benchmark）：本文提出的去偏多模态组合推理基准，包含Comprehensive、Relation、Attribute、Atomic、Negate、Content六个子任务。

**Bags-of-words现象**：模型忽略词序和句法结构，仅将句子视为词袋的倾向，常见于EVLM。

**Bridge Architecture**：将视觉特征投影到语言模型潜在空间的模型设计模式，区别于端到端训练的VLM。

**Content Challenge**：SADE新增任务，正样本仅含名词/形容词等视觉相关内容，负样本为无关但句法正确的流畅句子，挑战GVLM克服语法偏见的视觉理解能力。

## 可复现要素
- **数据集**：Winoground、VL-CheckList、ARO（Visual Genome）、CREPE均公开可用；COCO和Flickr30K为公开数据集。
- **代码/权重**：论文未提供代码；使用的GVLMs（LLaVA-7B/13B、MiniGPT-4、mPLUG-Owl、InstructBLIP、LLaMA-Adapter V2、Emu）均有开源版本。
- **关键超参**：SyntaxBias Score过滤阈值对应$p$值$< 1e^{-5}$；VisualGPTScore中$w_t = 1/m$（均匀权重）。
