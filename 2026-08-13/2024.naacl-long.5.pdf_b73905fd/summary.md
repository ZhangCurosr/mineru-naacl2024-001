---
title: "TelME: Teacher-leading Multimodal Fusion Network for Emotion Recognition in Conversation"
source: https://aclanthology.org/2024.naacl-long.5.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:29:50"
field: "对话情感识别"
keywords: ["Emotion Recognition in Conversation", "cross-modal knowledge distillation", "multimodal fusion", "shifting fusion", "teacher-student distillation", "MELD", "IEMOCAP"]
innovations: ["首次将跨模态知识蒸馏应用于ERC，以文本为教师增强弱模态", "提出响应蒸馏(Pearson相关)与特征蒸馏(KL散度)双路径", "注意力驱动的移位融合机制实现学生反向支持教师"]
benchmarks: ["MELD", "IEMOCAP"]
---

# 论文速读：TelME: Teacher-leading Multimodal Fusion Network for Emotion Recognition in Conversation

## 一句话总结
论文提出 TelME（Teacher-leading Multimodal Fusion Network），以文本编码器作为"教师"，通过跨模态知识蒸馏将情感相关知识迁移至音频/视觉"学生"编码器，再以注意力驱动的移位融合（Shifting Fusion）将学生特征反向补充教师表示，从而在对话情感识别（ERC）中充分利用各模态的贡献差异，在 MELD 数据集上达到 SOTA。

## 研究问题与动机
- **非语言模态贡献弱**：音频和视觉模态在 ERC 中的情感识别能力显著低于文本（图2单模态性能对比：MELD 上文本 F1 66.57%，音频46.60%，视觉36.72%），现有方法常将三者视为同质组件，忽视贡献差异。
- **模态间异质性障碍有效交互**：不同模态表征空间差异大，导致多模态交互效果受限（Zheng et al., 2022），单纯拼接难以发挥互补性。
- **多说话人多轮对话场景的挑战**：MELD 等数据集存在严重类别不平衡（neutral 占 47%），少数类情感难以被正确识别。
- **既有 KD 工作不适配 ERC**：现有跨模态蒸馏方法多针对图像或多模态检索设计，直接应用于 ERC 的文本-音视频跨模态场景存在 gap。

## 核心贡献（创新点）
1. **首次将跨模态知识蒸馏应用于 ERC**：以鲁棒的文本编码器作为教师，将情感相关知识迁移至音频和视觉学生编码器，本质区别于仅依赖单一强模态或平等对待各模态的已有方法。
2. **同时引入响应蒸馏（L_response）与特征蒸馏（L_feature）双路径**：响应蒸馏用 Pearson 相关系数替代 KL 散度来传递类别间/内偏好关系，适应教师-学生在输出空间的巨大差异；特征蒸馏用相似度概率分布的 KL 散度缩小表征异质性，二者结合可更全面地提取教师知识。
3. **注意力驱动的模态移位融合（ASF）**：将学生特征经多头自注意力生成移位向量，对教师表示进行加权偏移（Z = F_T + λ·H），实现"学生反向支持教师"的双向信息流，而非单向拼接或简单加权。
4. **系统性的 ablation 与教师选择分析**：通过更换教师模态的实验证明文本作为教师是最优选择；消融证实 ASF 和两种 KD 损失均有效，且二者需配合才能稳定提升。

## 方法详解
- **特征提取**：
  - **文本教师**：使用修改版 RoBERTa，对话上下文拼接为 `[<s_i>, t_1, <s_j>, t_2, ..., <s_i>, t_k]`，附加 prompt `"Now <s_i> feels <mask>"`，取 `<mask>` 对应输出作为情感特征 $F_{T_k} \in \mathbb{R}^{1 \times d}$。
  - **音频学生**：预训练 data2vec 提取第 k 轮语音片段 $a_k$ 的特征 $F_{a_k}$。
  - **视觉学生**：预训练 Timesformer 提取第 k 轮视频帧 $v_k$ 的特征 $F_{v_k}$。
  - 所有编码器输出维度统一为 768。
- **知识蒸馏**（学生总损失 $L_{student} = L_{cls} + \alpha L_{response} + \beta L_{feature}$）：
  - **响应蒸馏 L_response**：采用 DIST（Huang et al., 2022）思路，用 Pearson 相关距离 $d(\mu,v)=1-\rho(\mu,v)$ 衡量教师-学生预测分布间的类别间（行）和类别内（列）关系，温度参数 $\tau$ 控制 logit 软化程度，分别计算 $L_{inter}$ 和 $L_{intra}$ 后求和。
  - **特征蒸馏 L_feature**：构建教师表征相似矩阵 $M$（行内积后 softmax 得目标分布 $P_i$），同样方式得到师生共享的相似度分布 $Q_i$，以 KL 散度最小化两者差异，促使学生特征空间与教师对齐。
- **移位融合 ASF**：
  - 将学生特征拼接后经多头自注意力得到 $F_{attention}^k$。
  - 门控向量 $g_{AV}^k = R(W_1 \cdot [F_{T_k}, F_{attention}^k] + b_1)$ 筛选非语言相关信息。
  - 位移向量 $H_k = g_{AV}^k \cdot (W_2 \cdot F_{attention}^k + b_2)$。
  - 最终融合表示 $Z_k = F_{T_k} + \lambda \cdot H_k$，其中缩放因子 $\lambda = \min(\frac{\|F_k\|_2}{\|H_k\|_2} \cdot \theta,\ 1)$，$\theta$ 为阈值超参（MELD: 0.1，IEMOCAP: 0.01）。

## 实验与结果
- **数据集**：MELD（1400+ 对话，13000+ 话语，7类情感，严重不平衡）和 IEMOCAP（7433话语，151对话，6类情感，双人对白）。
- **评估指标**：加权 F1 score（处理类别不平衡）。
- **主要结果**：
  - **MELD**：TelME **67.37%**，超过前一 SOTA M2FNet（66.71%）**0.66%**，较 EmoCaps（64.00%）提升 **3.37%**。
  - **IEMOCAP**：TelME **70.48%**，与 EmoCaps（71.77%）相比略低，但整体仍具竞争力；较仅用文本提升约 **3.52%**。
  - **少数类提升**：TelME 在 Fear（26.97% vs EmoCaps 3.03%）和 Disgust（26.42% vs 7.69%）上大幅超越 EmoCaps，显示对难分类别的更好区分能力。
- **消融**（Table 4）：移除 ASF 后 IEMOCAP 从70.48降至63.33，MELD 从67.37降至67.04；逐层加入 $L_{response}$ 和 $L_{feature}$ 均有正向贡献。
- **教师模态实验**（Table 5）：文本教师（67.37/70.48）显著优于音频教师（56.28/49.36）和视觉教师（56.85/56.78），验证文本作为强模态教师的合理性。
- **鲁棒性**：5个随机种子下标准差极小（MELD: 0.078，IEMOCAP: 0.258）。

## 相关工作脉络
1. **DialogueRNN / ConGCN / MMGCN**：基于 RNN/图神经网络建模对话上下文和说话人关系，未充分挖掘音视频多模态信息或仅做简单融合；TelME 以跨模态蒸馏强化弱模态并设计双向移位融合。
2. **EmoCaps**：用胶囊网络融合多模态与情感倾向，在 IEMOCAP 上表现最佳但在 MELD 上较弱；TelME 在 MELD 上取得 SOTA 且少数类识别能力更强。
3. **M2FNet**：当时 MELD 的 SOTA，通过层级结构提取对话级多模态特征；TelME 通过蒸馏+移位融合超越其 0.66%。
4. **FaceialMMT**：侧重多说话人视频中提取真实说话人脸部序列，引入辅助帧级表情任务；TelME 更通用，关注跨模态知识迁移而非辅助任务。
5. **Decoupled Multimodal Distilling (Li et al., 2023b)**：动态图上的图蒸馏，面向通用多模态任务；TelME 针对 ERC 的文本-音视频跨模态特性设计了 Pearson 相关响应蒸馏和移位融合。
6. **UniMSE**：统一多模态情感分析与 ERC 的 T5 框架；TelME 不使用预训练大语言模型，而是以 RoBERTa 为教师，更轻量且专注跨模态蒸馏策略。

## 局限性与未来方向
- **视觉模态性能最低**：短时 utterance 的帧数有限导致面部表情捕捉不足，即使经蒸馏视觉学生仍表现较弱（IEMOCAP 视觉仅18.85%）。
- **类别不平衡影响蒸馏效果**：MELD 中 neutral 占47%，少数类样本稀少制约了 $L_{feature}$ 的增益幅度；教师初始阶段也难以充分识别少数类后传递给学生的可靠性受限。
- **未来方向**（论文自述）：开发更适合短时 utterance 的视觉特征提取方法，提升面部表情理解能力，从而进一步增强知识蒸馏的效果。

## 研究启发与可借鉴点
1. **Pearson 相关替代 KL 散度的跨模态响应蒸馏**：当教师-学生在输出空间差距极大时，相关性度量比概率分布散度更稳健，该思路可迁移至其他跨模态蒸馏场景（如图像-文本、语音-文本）。
2. **移位融合（Shifting Fusion）架构**：不拼接而是用学生特征生成"位移向量"对教师表示做偏移，这一双向补充机制可推广到其他多模态分类/生成任务。
3. **教师-学生角色设计的系统验证方法**：通过更换教师模态（Table 5、6）的系统分析，揭示了"强模态教学"的一般规律，可作为多模态蒸馏实验设计的参考范式。
4. **类别不平衡下的蒸馏策略分析**：论文对 $L_{feature}$ 在 MELD 上增益有限的原因归因于类别不平衡，提示在后续研究中可结合重采样或代价敏感学习进一步缓解此问题。

## 关键术语表
- **Emotion Recognition in Conversation (ERC)**：在多轮对话中逐轮识别说话人情感状态的任务。
- **Cross-modal Knowledge Distillation**：将强模态（教师）的知识迁移至弱模态（学生）的蒸馏方法，用于弥合模态间异质性。
- **Response Distillation (L_response)**：基于 Pearson 相关系数衡量师生预测分布的类别间/内关系相似度，而非直接比较 logits。
- **Feature Distillation (L_feature)**：通过对师生表征相似矩阵施加 KL 散度损失，使学生的特征空间分布逼近教师。
- **Attention-based Modality Shifting Fusion (ASF)**：以学生特征生成门控位移向量，对教师情感表示进行加权偏移融合的多模态融合方法。
- **data2vec / Timesformer**：分别用于音频和视觉的自监督预训练编码器，作为学生编码器的初始化。
- **Weighted F1 Score**：按各类别样本数加权计算的 F1 分数，用于评估类别不平衡数据集上的分类性能。

## 可复现要素
- **数据集**：MELD（公开，https://github.com/Amanprior/MELD）和 IEMOCAP（需申请，https://sail.usc.edu/iemocap/）；论文未提及是否公开代码/权重。
- **关键超参**：
  - 所有编码器输出维度：768
  - 优化器：AdamW，初始 lr = 1e-5，线性 warmup
  - $L_{response}$ 温度：MELD=2，IEMOCAP=4
  - $L_{feature}$ 温度：均为 1
  - $\alpha$（$L_{response}$ 平衡因子）：MELD=1，IEMOCAP=0.1
  - 移位融合阈值 $\theta$：MELD=0.1，IEMOCAP=0.01
  - Dropout：MELD=0.2，IEMOCAP=0.1
  - 多头注意力 head 数：MELD=3，IEMOCAP=4
  - 硬件：单卡 NVIDIA GeForce RTX 3090
