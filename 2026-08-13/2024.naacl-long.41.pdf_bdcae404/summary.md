---
title: "V Attack: Taking advantage of Text Classifiers’ horizontal vision e"
source: https://aclanthology.org/2024.naacl-long.41.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:29:21"
field: "文本对抗攻击与鲁棒性"
keywords: ["adversarial attack", "text classification", "vertical text", "blackbox attack", "Robustness", "NLP security"]
innovations: ["首次利用分类器无法识别垂直文本的缺陷设计对抗攻击", "提出贪心词选择+竖向变换的攻击框架并验证人类可读性", "引入chaff干扰字符增强攻击以对抗逆向防御"]
benchmarks: ["SST-2", "AG News", "CoLA", "QNLI", "Rotten Tomatoes", "OLID"]
---

# 论文速读：V Attack: Taking advantage of Text Classifiers' horizontal vision

## 一句话总结
本文提出了一种名为 VertAttack 的新型文本对抗攻击方法，通过将文本分类器依赖的关键词语转为竖向书写来欺骗模型，利用当前分类器无法识别垂直文字的根本缺陷实现攻击，同时保持人类可读性。

## 研究问题与动机
- **核心问题**：当前所有主流文本分类器仅以水平方式处理文本，无法识别竖向书写的词语，而人类可以轻松阅读两种方向的文字。
- **现有攻击局限**：现有 SOTA 攻击（如 BERT-ATTACK、Textbugger）仅限于水平方向变化，攻击者将自己限制在与分类器相同的域内。
- **攻击动机**：模拟真实人类如何通过改变书写方向来绕过自动分类器，例如在社交媒体上规避内容审核。
- **实际威胁**：攻击可被恶意利用于绕过平台的内容审核系统，或用于隐私保护目的。

## 核心贡献（创新点）
- **提出 VertAttack 攻击框架**：首次利用分类器对垂直文本的识别盲区，将关键词语竖向排列实现攻击，与现有攻击仅在水平空间操作形成本质区别。
- **系统性验证跨数据集/模型攻击效果**：在 5 个数据集上对 4 种 Transformer 分类器进行测试，发现攻击可将准确率从 83%-95% 降至 1%-36%，远超 BERT-ATTACK 和 Textbugger。
- **人类可读性验证**：通过众包实验证明人类对扰动后文本的识别准确率为 77%，仅比原始文本低 4 个百分点，证实攻击在保持语义方面的有效性。
- **探索防御与反防御机制**：研究空白字符移除、文本分割和算法逆向三种防御策略，并提出引入"chaff"（干扰字符）增强攻击以对抗逆向防御。

## 方法详解
VertAttack 分为两个主要步骤：

**Step 1：词语选择（Word Selection）**
- 采用贪心搜索策略，依次移除文本中的每个词，比较分类概率变化。
- 选择导致概率下降最大的词语作为攻击目标。
- 伪代码（Algorithm 1）遍历文本，计算每次移除后的分数差 Drop_w = Score_orig - Score_w，保留最大 Drop 对应的词语位置。

**Step 2：词语变换（Word Transformation）**
- 将选定的词语转换为竖向书写格式（Algorithm 2）。
- 计算每个被选中词语所需行数（等于词长度）。
- 迭代原文本每个词：若为选中词，取对应行的字符；否则直接保留原词或填充等长空白的空格。
- 每行末尾换行，最终拼接生成竖向格式文本。
- 引入宽度约束（width constraint）控制单次变换的词数，避免过长行影响可读性。

**增强策略：Chaff 干扰**
- 在竖向空白行中以概率 p 插入随机字母字符，增加逆向防御难度。
- 不干扰原始词语周围的空白，以保持基本可读性。
- 测试 p ∈ {5%, 10%, 20%, 30%, 60%}。

**防御逆向算法（Reverse）**
- 将垂直文本按行分割，识别含多于一个字符的"原始行"。
- 将竖向字符追加到对应行的原始位置，重建水平文本。

## 实验与结果
- **数据集**：AG News（4类）、SST-2（2类情感）、CoLA（语法可接受性）、QNLI（问答 entailment）、Rotten Tomatoes（2类情感），各采样最多 1000 样本（QNLI 为 872）。
- **分类器**：BERT-base-uncased、ALBERT、RoBERTa、DistilBERT。
- **主要结果**：
  - **SST-2 上 RoBERTa**：准确率从 94% 骤降至 13%，下降 81 个百分点。
  - **平均下降幅度**：5 数据集上，相同反馈分类器时平均下降 83 点（AG）、80 点（SST-2）、74 点（CoLA）、56 点（QNLI）、71 点（RT）。
  - **跨模型迁移**：反馈分类器与目标不同时仍有效，CoLA 平均下降 51 点，其他数据集约 25-40 点。
  - **与基线对比（RT 数据集）**：VertAttack 平均将分类器降至 36.6% 准确率，优于 BERT-ATTACK（47.5%）和 Textbugger（63.2%）；Transfer 场景下 VertAttack 表现最佳（47% vs 66.5% vs 78.1%）。
  - **Offensive Language（OLID）**：相同反馈分类器时准确率降至 1% 以下；跨模型时也仅 13%-28%。
  - **OCR 影响**：经 OCR 转换后 Albert 准确率从 13.6% 提升至 35.7%，但仍低于多数类基线（53.3%）。
  - **防御效果**：Reverse 算法可将准确率恢复至 84%-87%；但加入 chaff（p≥10%）后显著削弱防御效果。
  - **人类可读性**：77% 扰动文本可被正确分类（vs 原始 81%）。

## 相关工作脉络
- **BERT-ATTACK (Li et al., 2020)**：基于 BERT MLM 的词语替换攻击，利用分类器 logits 评估词重要性，仅操作水平空间，不替换原词以保持语义。
- **Textbugger (Li et al., 2019)**：字符级攻击，测试插入、删除、交换、替换字符，属于黑盒字符攻击，更易被预处理防御。
- **Character-based attacks (Gröndahl et al., 2018; Eger et al., 2019)**：翻转字符、添加空格或使用视觉相似字符，目标使 OOV，易受空白字符处理影响。
- **Word saliency attacks (Ren et al., 2019; Jin et al., 2020)**：基于概率加权词显著性或掩码方法选择重要词进行替换，与 VertAttack 的贪心选择策略类似但攻击维度不同。
- **Attention-based word selection (Wu et al., 2019)**：利用注意力机制定位关键词进行风格迁移，可作为 VertAttack 选择阶段的效率改进方向。

## 局限性与未来方向
- **平台格式保留问题**：网站可能不保留额外换行符和空白，导致竖向文本可读性大幅下降；建议使用图像格式展示但非所有平台支持图片。
- **仅针对 Transformer 模型**：未验证在非 Transformer 模型（如 LSTM、朴素贝叶斯）上的效果。
- **贪心选择效率低**：逐一移除词语计算概率开销大，可借鉴注意力机制或风格迁移算法优化选择效率。
- **QNLI 结果较弱**：受限于仅攻击 hypothesis 而非 premise，若允许攻击 premise 可能效果更强。
- **Chaff 增加影响可读性**：随 p 增大，人类理解率下降，需权衡攻击强度与可读性。

## 研究启发与可借鉴点
- **攻击维度的创新视角**：从"空间维度"（水平 vs 垂直）而非仅"内容维度"（词语替换）设计攻击，启发了对分类器输入预处理的系统性漏洞挖掘。
- **可复用的贪心词选择策略**：Algorithm 1 的逐词移除评估方法简单高效，可直接迁移到其他攻击框架。
- **宽度约束的实用设计**：控制每行词数以平衡攻击效果和格式可读性，对实际部署有参考价值。
- **防御-攻击迭代实验范式**：通过 Simple → Segment → Reverse 的渐进防御分析，结合 Chaff 反制，展示了完整的攻防博弈研究流程。
- **OCR 鲁棒性测试**：将文本转图像再用 OCR 提取，模拟真实端到端系统的脆弱性，可作为多模态攻击测试的基准方法。

## 关键术语表
- **VertAttack**：本文提出的新型对抗攻击方法，通过将关键词语竖向排列欺骗文本分类器。
- **Blackbox attack**：攻击者无模型内部参数知识，仅能通过输入-输出（概率/标签）反馈评估攻击效果。
- **Greedy word selection**：贪心词语选择策略，依次移除每个词并比较分类概率变化以定位最关键的词。
- **Transferability**：攻击迁移性，指针对一个分类器的攻击对另一个不同分类器依然有效的能力。
- **Chaff**：插入到竖向空白行的随机字母字符，用于破坏防御方的逆向算法。
- **Reverse algorithm**：防御算法，将竖向文本重建为水平文本以恢复分类器性能。
- **SST-2**：Stanford Sentiment Treebank，电影评论情感分类数据集（正面/负面）。
- **QNLI**：Stanford Question Answering Dataset，问答蕴含任务数据集。

## 可复现要素
- **数据集**：AG News、SST-2、CoLA、QNLI、Rotten Tomatoes、OLID（公开数据集）
- **代码**：论文声明共享代码和扰动文本（"We share code and perturbed texts for future research"），但 ACL 匿名投稿阶段具体链接未在正文中提供
- **关键超参**：宽度约束（默认值未明确说明）、Chaff 概率 p ∈ {5%, 10%, 20%, 30%, 60%}
- **模型**：BERT-base-uncased、ALBERT、RoBERTa、DistilBERT（均在对应数据集上 fine-tuned）
- **OCR 工具**：PIL（图像生成）、Tesseract OCR（文字提取）
