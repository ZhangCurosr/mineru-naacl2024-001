# An Examination of the Compositionality of Large Generative Vision-Language Models

Teli Ma†<sup>,</sup>⋄ Rong Li† Junwei Liang†<sup>,</sup>‡<sup>,△</sup> † AI Thrust, The Hong University of Science and Technology (Guangzhou) ‡ Department of Computer Science and Engineering, The Hong Kong University of Science and Technology ⋄ Primary author <sup>△</sup> Corresponding author {tma184, rli335}@connect.hkust-gz.edu.cn junweiliang@hkust-gz.edu.cn

## Abstract

With the success of Large Language Models (LLMs), many Generative Vision-Language Models (GVLMs) have been constructed via multimodal instruction tuning. However, the performance of GVLMs in multimodal compositional reasoning remains under-explored. In this paper, we examine both the evaluation metrics ( VisualGPTScore, etc.) and current benchmarks for evaluating the compositionality of GVLMs. We identify the syntactical bias in current benchmarks, which is exploited by the linguistic capability of GVLMs. The bias renders VisualGPTScore an insufficient metric for assessing GVLMs. To combat this, we first introduce a SyntaxBias Score, leveraging LLMs to quantify such bias for mitigation. A challenging new task is subsequently added to evaluate the robustness of GVLMs against inherent inclination toward syntactical correctness. Using the bias-mitigated datasets and the new task, we propose a novel benchmark, namely SyntActically DE-biased benchmark (SADE). Our study provides an unbiased benchmark for the compositionality of GVLMs, facilitating future research in this direction <sup>1</sup>.

## 1 Introduction

A surge of research on vision-language models (VLMs) has demonstrated success in a wide range of tasks, including zero-shot visual recognition (Radford et al., 2021; Gao et al., 2021; Zhou et al., 2022), visual question answering (Alayrac et al., 2022; Chen et al., 2022), and image-to-text retrieval (Alayrac et al., 2022; Gong et al., 2023a). Previous Vision-Language Models (VLMs) have predominantly been developed using image-text contrastive (ITC) learning (Radford et al., 2021; Jia et al., 2021; Li et al., 2022, 2023b) and image-text matching (ITM) (Tan and Bansal, 2019; Chen et al., 2020; Gan et al., 2020; Li et al., 2021; Zhang et al.,

2021; Kim et al., 2021) frameworks, a category we term Encoder-based Vision-Language Models (EVLMs). With the advent of large language models (LLMs) like ChatGPT, GPT-4 (OpenAI, 2023) and LLaMA (Touvron et al., 2023a), recent studies have extended the decoder-only architecture to multimodal settings, which is named Generative VLMs (GVLMs) (Liu et al., 2023a; Zhu et al., 2023; Li et al., 2023a; Ye et al., 2023; Gao et al., 2023; Sun et al., 2023; Dai et al., 2023). The GVLMs deviate from the EVLMs in projecting visual features into the latent lexical space of LLMs, and leveraging the auto-regressive generative capacity to solve vision-language tasks. In the training process, most work follows the recipe of freezing the main body of visual encoders and LLMs, only updating the negligible parameters of projecting layers, which is also called “bridge architecture" (Rajesh et al., 2023).

Despite the emergence of research on GVLMs, the understanding of compositionality in GVLMs has remained an enigmatic black box, with no thorough investigations conducted thus far. Previous research studies (Thrush et al., 2022; Zhao et al., 2022; Yuksekgonul et al., 2022a; Ma et al., 2023; Ray et al., 2023a) in multimodal compositionality focus on establishing retrieval-based benchmarks for evaluating EVLMs on object relations and attribute understanding, order sensitiveness of sentence elements, and atom-level understanding. The EVLMs have demonstrated abilities to discriminate positive captions from negative ones based on the image-text similarity, where the disparities between the positive and negative captions are relatively subtle, such as “an old person kisses a young person" and “a young person kisses an old person" (Thrush et al., 2022).

However, we observe there exists an underlying bias towards the LLM part of GVLMs in the evaluation of the aforementioned benchmarks. During the evaluation, the log-likelihood-based scores are widely adopted to evaluate the generative models (Fu et al., 2023; Liu et al., 2023c; Lin et al., 2023; Li et al., 2023c) to estimate the conditional probabilities of specific generations. Following Lin et al. (2023), we alias the log-likelihood score as VisualGPTScore. We examine the current benchmarks for evaluating GVLMs with Visual-GPTScore and find that:

• Using VisualGPTScore to evaluate GVLMs is not sensitive to bags-of-words problems that broadly exist in the evaluation of EVLMs with similarity scores. The bags-of-words phenomenon during evaluation is due to the similarity-based metrics.

• VisualGPTScore sometimes prefers syntactical correctness rather than content-related correctness under the current benchmarks. It scores negative references with reasonable syntax but unrelated content higher than positive references. In contrast, EVLMs pay more attention to the correlation of visual content but are not sensitive to the order of tokens in references.

• A prevalent syntactical bias is present in contemporary multimodal compositional reasoning benchmarks.These benchmarks are tailored for assessing EVLMs, and the approach used to create negative references may not be effective for the evaluation of GVLMs.

Based on these observations, our contributions include:

• We quantitatively analyze the syntactical bias (namely SyntaxBias Score) that broadly exists in current benchmarks by leveraging LLMs.

• With the SyntaxBias Score, we propose a SyntActically DE-biased benchmark (SADE) based on current benchmarks for a more robust multimodal compositionality evaluation. We adopt multiple strategies to mitigate the syntactical bias in existing benchmarks. We also add a new challenging assessment in SADE to evaluate the content understanding across visual and language modalities.

• The performance of several GVLMs is reported on SADE, as well as the robustness and faithfulness to human judgments.

## 2 Background

## 2.1 Generative vision-language models

In this paper, we define GVLMs as models that combine visual encoders with large language models (LLMs) trained on large text corpora. The prevailing approach in recent research connects a frozen visual encoder with an LLM by training mapping layers on images-text pairs, followed by fine-tuning using multi-modal instructional data to facilitate multi-turn conversations (Liu et al., 2023a; Gao et al., 2023; Zhu et al., 2023; Dai et al., 2023; Su et al., 2023; Gong et al., 2023b; Sun et al., 2023). This approach is anchored in the idea of treating visual tokens the same as linguistic ones. The visual tokens are mapped into a lexical embedding space and harnessed to generate textual content in an autoregressive manner. Formally, given an image I and the visual encoding $g ( I )$ from encoders like Vision Transformer (Dosovitskiy et al., 2020), the mapping process can be formulated as:

$$
\boldsymbol { z } = \mathbf { M } ( g ( I ) ) , \boldsymbol { z } = \{ z _ { 1 } , z _ { 2 } , . . . , z _ { N } \} ,\tag{1}
$$

where N is the number of visual tokens and M is the mapping layers. Different from EVLMs that utilize image-text contrastive (ITC) or image-text matching (ITM), the training objective of multimodal autoregressive training is to maximize the log-likelihood of the next true token. Denote the tokenized instructions as p and the output words as $t _ { i } , ( 1 \leq i \leq K )$ , the GVLM training objective is defined as:

$$
\operatorname* { m a x } _ { \theta _ { M } , \theta _ { \sigma } } \sum _ { i = 1 } ^ { K } \log P ( t _ { i } | \boldsymbol { p } , z , t _ { 1 } , t _ { 2 } , . . . , t _ { i - 1 } ; \theta _ { M } , \theta _ { \sigma } )\tag{2}
$$

where $\theta _ { M }$ refers to the learnable parameters of mapping layers M and $\theta _ { \sigma }$ refers to other tunable parameters like adapter layers in LLaMA-Adapter V2 (Gao et al., 2023), or visual abstractor and LoRA in mPLUG-Owl (Ye et al., 2023).

In comparison, the training objectives of EVLMs are based on the ITC or ITM loss between vision and language parts. Please refer to Appendix A.1 for formulations of EVLMs.

## 2.2 Vision-language compositionality

Recent works on vision-language compositionality focus on introducing benchmarks to evaluate the EVLMs, mainly on CLIP (Radford et al., 2021). Winoground (Thrush et al., 2022) is one of the pioneers in building benchmarks for multimodal compositionality, curating 400 test items to evaluate the pragmatics, symbolic and series factors of VLMs. Afterwards, several benchmarks have been proposed to challenge the objects, relations and attributes understanding of VLMs, including VL-CheckList (Zhao et al., 2022), ARO (Yuksekgonul et al., 2022a), CREPE (Ma et al., 2023), VALSE (Parcalabescu et al., 2021) and Cola (Ray et al., 2023b) etc. These benchmarks are in the form of image-text retrieval, requiring the model to differentiate positive references from negative references based on the visual contents of the images. See Fig 6 in the Appendix for the details of the image-text retrieval format. SugarCrepe (Hsieh et al., 2024) is one of the most recent and similar works to ours. SugarCrepe utilizes Vera (Liu et al., 2023b) and TextAttack (Morris et al., 2020) to detect the plausibility and grammar gaps between positive and negative references. Then, it prompts ChatGPT to generate reasonable hard negative references to reduce bias. In comparison, we partially rely on the original benchmarks, focusing on the strategy of mitigating bias by filtering and modifying them based on our defined SyntaxBias Score. All the aforementioned benchmarks are curated for evaluating EVLMs, where similarity scores between images and references serve as the criteria for selecting references. Then, the accuracy of selecting positive samples across all data samples will be reported to assess the model’s compositional understanding capability.

## 2.3 Evaluation metrics for multimodal retrieval

Since previous benchmarks have been carefully curated for evaluating EVLMs, image-text similarity scores naturally emerge as the metric for assessing the compositional similarity between images and references. For generative models, an intuitive way is reference-based, measuring the quality of generated captions with metrics like BLEU (Papineni et al., 2002), METEOR (Banerjee and Lavie, 2005), ROUGE (Lin, 2004) and CIDEr (Vedantam et al., 2015). Among the reference-based metrics, BERTScore (Zhang et al., 2019) tackles superficial matching between captions and references in lexical expression, delving deeper into the semantic similarity matching. GPTScore (Fu et al., 2023) proposes to leverage emergent abilities of generative models to score generated texts. Inspired by GPTScore, recent works (Lin et al., 2023; Li et al., 2023c; Liu et al., 2023c) measure the GVLMs using the log-likelihood of directly generating reference sentences conditioned on the image. We follow the Lin et al. (2023) to abbreviate the kind of method as VisualGPTScore, which can be formulated as:

$$
\begin{array} { l } { \displaystyle \mathrm { V i s u a l G P T S c o r e } ( r | \mathcal { T } ) } \\ { \displaystyle = \sum _ { t = 1 } ^ { m } w _ { t } \log P ( r _ { t } | r _ { < t } , p , \mathcal { T } ; \theta _ { G V L M } ) } \end{array}\tag{3}
$$

where , r, p represents the image, reference sentence and instructions. $\theta _ { G V L M }$ refers to parameters of GVLMs and $\begin{array} { r } { w _ { t } = \frac { 1 } { m } } \end{array}$ . The VisualGPTScore is directly estimated conditioned on images and thus reference-free. In this work, we examine the VisualGPTScore and discuss the potential influence of using it in current benchmarks for vision-language compositionality.

## 3 Experimental setup

We introduce the configurations of experiments for the syntactical bias examination in this section.

## 3.1 Model choices

We leverage two state-of-the-art GVLMs, namely LLaVA (Liu et al., 2023a) and MiniGPT-4 (Zhu et al., 2023), to conduct experiments. LLaVA is one of the first methods to project visual features into LLaMA (Touvron et al., 2023a) latent space via multimodal instruction tuning. A linear projection layer and the parameters of the LLM are tuned on conversations, detailed descriptions, and complex reasoning datasets. MiniGPT-4 (Zhu et al., 2023) maps visual embeddings obtained from ViT and Q-Former (Li et al., 2022) into Vicuna (Chiang et al., 2023) via a linear projection layer. We adopt the model version of “LLaVA-7B-v0" and “Minigpt4- aligned-with-Vicuna7B" to evaluate. However, we found that when using VisualGPTScore to evaluate compositionality, both models exhibited similar patterns. Therefore, for the sake of brevity, we only present the results for LLaVA.

## 3.2 Datasets

We use Winoground (Thrush et al., 2022), VL-Checklist (Zhao et al., 2022), ARO (Yuksekgonul et al., 2022a) and CREPE (Ma et al., 2023) in the evaluation analysis, totaling 52,189 images and 129,558 reference sentences. All benchmarks necessitate the model’s selection of positive reference sentences from negative ones. For Winoground, we report text score, image score and group score as the paper (Thrush et al., 2022). For other datasets, Recall@1 accuracy is reported.

![](images/27cc188884a3634fcd505368805427c307bbe0f687d7e954d08bf7df38d31d17.jpg)  
Figure 1: Box plots of scaled score distributions for original (x1) and perturbed captions (x2-x5, x2: shuffle nouns & adj, x3: shuffle all but nouns & adj, x4: shuffle within trigrams, x5: shuffle trigrams). The distribution gap between the original captions and the shuffled captions is evident for the generative scores, while the contrastive score (BERTScore) is significantly less affected by the order of words. The CLIPScore sub-figure illustrates the distribution of similarity scores generated by the CLIP model, which is compared with the first three sub-figures of LLaVA-7B.

## 4 Evaluation Metric Examination

VisualGPTScore measures the probability of generating specific references conditioned on the given images, as defined in Eqn. 3. The generative evaluation method is based on the inherent attribute of GVLMs and used in image-text retrieval (Lin et al., 2023; Li et al., 2023c; Liu et al., 2023c). Since current benchmarks on VL compositions consists of image-text pairs, we follow Lin et al. (2023) to utilize VisualGPTScore for evaluating the VL compositionality of GVLMs. In this section, our primary focus is to examine the bias of using VisualGPTScore in current benchmarks.

## 4.1 Sensitivity to bags-of-words

Previous research works have pointed out that EVLMs suffer from the bags-of-words phenomenon when doing compositional reasoning due to the pre-training recipe of matching visual and textual data in instances-level (Yuksekgonul et al., 2022b; Diwan et al., 2022). However, we observe that the bags-of-words problem is not only related to the models, but also highly correlated to the evaluation metrics, and VisualGPTScore is not sensitive to the bags-of-words phenomenon.

We explore the influence of different metrics in sensitivity to the order of tokens in sentences for GVLMs. Following CREPE (Ma et al., 2023), we randomly sample 2.5K image-text pairs from the COCO dataset (Lin et al., 2014) and adopt the following strategies to shuffle the elements of captions: Shuffle only nouns & adjectives, Shuffle all but nouns & adjectives, Shuffle within trigrams, Shuffle trigrams. Then, we calculate the VisualGPTScore, GPTScore (Fu et al., 2023) and BERTScore (Zhang et al., 2019) based on LLaVA-7B. The distribution of normalized scores are shown in Fig. 1, where x1 represents positive references and x2-x5 represents shuffled references, respectively.

It can be observed that to the same model, LLaVA-7B, VisualGPTScore is similar to GPTScore, more sensitive to the order and structure of reference sentences compared with contrastive metric BERTScore. We also report the score distribution of the CLIP model using contrastive similarity (CLIPScore in Fig. 1), which is similar to the distribution of BERTScore results on LLaVA-7B. It implies the bags-of-words problem may be attributed to the evaluation metrics based on similarity score, but generative scores mitigate the problem to some extent.

## 4.2 Sensitivity to syntax and contents

Based on the observation that ViusalGPTScore mitigates the bags-of-words problem to some extent, we are curious about whether they lean more towards evaluating syntactic correctness than content relevance when assessing the compositionality of GVLMs. To examine it, we design an experiment using the test set of Flickr30K dataset (Young et al., 2014). Specifically, we sample 507 imagetext pairs and construct three types of evaluation cases as shown in Fig. 2. Given an image, the task is to retrieve the positive reference from the cases below. The final scores are averaged over 507 test samples. In Case 1, each positive reference sentence is accompanied by two hard negatives with shuffled nouns, adjectives and trigrams. In Case 2, the provided negatives are fluent and syntactically correct captions sampled from COCO, which are unrelated to the visual contents. In Case 3, we keep only adjectives and nouns in the positive reference sentences by removing all the adverbs, pronouns and modifiers.

![](images/2d8bbf1284db5185e4d1f5232d3fee485579c8cb62467da2651bf147376525e8.jpg)

<table><tr><td>Case 1</td><td>VisualGPTScore</td></tr><tr><td>Right caption: an elderly asian woman wearing a straw-like hat sits outside near a bicycle while a gray car is about to pass by. Shuffled caption: an like gray hat wearing a bicycle - asian woman sits outside near a straw while a about car is elderly to pass by. Shuffled caption: elderly an asian wearing a woman hat sits straw-like a near outside bicycle a while is gray car pass to about by</td><td>0.405 0.051</td></tr><tr><td>Case 2</td><td>0.077</td></tr><tr><td>Right caption: an elderly asian woman wearing a straw-like hat sits outside near a bicycle while a gray car is about to pass by Random caption: the two cats are laying on the chair together</td><td>0.405 0.231</td></tr><tr><td>Random caption: two giraffes in an outdoor setting eating grass</td><td>0.432</td></tr><tr><td>Case 3</td><td></td></tr><tr><td></td><td></td></tr><tr><td>Content caption: elderly asian woman, straw-like hat, bicycle , gray car</td><td>0.322</td></tr><tr><td>Random caption: the two cats are laying on the chair together</td><td>0.231</td></tr><tr><td>Random caption: two giraffes in an outdoor setting eating grass</td><td>0.432</td></tr></table>

Figure 2: An example of three Cases of captions we construct to validate the preference of syntax and contents. Right caption: the original caption of the image, Shuffled caption: caption that the sentence elements are shuffled, Random caption: fluent and syntactically correct captions from other datasets (COCO), Content caption: caption that keeps only adjectives and nouns to keep the contents like objects and attributes. We present the normalized VisualGPTScore of every reference sentences in this example. The scores of the Right caption and Content caption may be lower compared to the Random caption (0.405, 0.322 vs. 0.432). This indicates that in this example, generative VLMs tend to prioritize syntactically correct sentences over ones that are more relevant to the content.

![](images/250bcaf981d64f82f30c4687eb57c4cf7e4bebe66fcfb6a74f9c6f80eaa24007.jpg)  
Figure 3: We report the accuracy of VisualGPTScore based on LLaVA-7B and similarity score based on CLIP in the sampled 507 image-text pairs, each pair is consisted of three cases like the example in Fig. 2.

We present Recall@1 of VisualGPTScore for the GVLM (LLaVA-7B), and vision-language similarity for the EVLM (CLIP) in three evaluation cases. As shown in Fig. 3, the LLaVA model can easily discriminate the right reference sentences from the shuffled ones, reaching 98.62% with the help of VisualGPTScore. However, if the negatives are random reference sentences in Case 2, the performance degradation is up to 31.56%. In Case 3, where the sentences are syntactically incorrect, the performance drops to 27.02%. In contrast, CLIP excels at excluding negative sentences that are contextually unrelated to the image, but suffers from insensitive to syntax and sentence order.

The potential reason for the above results is the difference in the pre-training paradigm. Specifically, the generative model pre-training is to maximize the likelihood of the next token prediction in an auto-regressive manner. In contrast, the training objective of EVLM is to maximize the alignment between positive image-text pairs and minimize that between negative ones. Previous research (Yuksekgonul et al., 2022b) shows that CLIP takes the short-cut strategy of not encoding the order information, but only object features for retrieval/captioning tasks, which conforms to our finding. We also believe that the generative VLMs take the short-cut strategy of not fully mapping the visual and linguistic features, but leveraging the emerging capacity of LLM part to generate based on limited visual cues. This reliance on the LLM part results in a bias towards syntactical correctness in captions under the criteria of generative score.

## 5 Benchmarks Examination

From above, we know current benchmarks are curated for evaluating EVLMs based on similarity score originally. Hence, we examine the impact of using these datasets for evaluating GVLMs with VisualGPTScore, and uncover the bias of existing datasets.

## 5.1 Syntactical bias in current benchmarks

According to the observation made in Section 4, it is evident that auto-regressive vision-language models exhibit sensitivity toward the syntax and order of phrases. Hence, existing benchmarks that generate hard negatives by swapping, shuffling, or replacing specific entities promote a syntactical bias, which refers to a preference for models to rely on the morphological structure of words. Consequently, this bias can be exploited by GVLMs to effortlessly differentiate between positive and negative samples.

![](images/e86743f9094b84376faf0d447830bf5fd2d28aa3fd057b318eb1c7053dc5e4b2.jpg)

![](images/fb8e77415b092800fa25402026d73d366f401911a99395a2c2c5a9fa8aa0772d.jpg)

![](images/6a032058aa0fc6f4d05b59040092370b0e90a101c01a53d6e786b0dd7d73f253.jpg)  
Figure 4: The drop in performance of the LLaVA model when performing compositional reasoning on nonsensical noisy images is minimal in existing benchmarks, whereas the CLIP model exhibits a significant decrease. This indicates current benchmarks are exploited by the LLM part of GVLMs, not effective in measuring the multimodal compositionality.

To show that the bias exists in current compositional reasoning benchmarks, we conduct the ablation of utilizing both GVLMs and EVLMs to reason nonsensical images with normal reference sentences. Specifically, we construct the imagetext pairs by replacing the original images with images composed of random noises. We observe the performance drop in both the GVLMs and EVLMs. As shown in Fig. 4, the performance degradation of CLIP (ViT-B/32) is large, approaching the Recall@1 accuracy of randomly choosing. However, as for the LLaVA-7B, the trend of performance dropping is not obvious, indicating the GVLMs make the right choices solely based on the linguistic reference sentences without visual features. Therefore, almost all the benchmarks lean towards evaluating the linguistic part of GVLMs, rather than the visio-linguistic understanding of GVLMs.

## 5.2 SyntaxBias Score

To alleviate the syntactical bias in current benchmarks, we first quantify the bias for analysis. In an ideal scenario, in the absence of visual intervention, the quantified scores generated by GVLMs for positive and negative reference sentences should be equivalent. Therefore, we define the SyntaxBias Score to measure the syntactical discrepancy between positive and negative reference sentences. Formally, the SyntaxBias Score is calculated using the generative scores of positive and negative text produced by auto-regressive language models:

$$
\begin{array} { l } { { \displaystyle S c o r e _ { S y n t a x } B i a s } } \\ { ~ = \Delta ( \displaystyle \sum _ { i = 1 } ^ { m } w _ { i } \log P ( p _ { i } | { p _ { < i } } ; \theta ) } \\ { ~ - \displaystyle \sum _ { j = 1 } ^ { n } \hat { w } _ { j } \log P ( n _ { j } | { n _ { < j } } ; \theta ) ) , } \end{array}\tag{4}
$$

where $\Delta , \mathbf { p } , \mathbf { n } , \theta$ represent normalization, positive tokens, negative tokens, and parameters of LLMs respectively. we leverage a strong LLM, Vicuna-13B-v1.5 (Chiang et al., 2023), to compute the SyntaxBias Score, which are normalized between 1 and 1. We present the visualization of SyntaxBias Score distributions over different benchmarks in Fig. 5. We find that most of the mainstream benchmarks except Winoground are biased towards positive captions with distribution centers located to the right, which makes the generative scores ofGVLMs on these benchmarks overvalued.

## 6 Mitigate the Bias in Benchmarks

In this section, we propose a strategy to modify the benchmarks and mitigate the syntactical bias to provide a better evaluation of GVLMs. Specifically, we filter current datasets leveraging LLMs and add a novel challenge to evaluate visual content understanding. We name the new benchmark as SyntActical De-biased benchmark, abbreviated as SADE. In the following, we describe the filtering details of each dataset and the new challenge. Then we show human evaluation to show the effectiveness of SADE.

## 6.1 Winoground

The Winoground (Thrush et al., 2022) dataset comprises 400 image-text pairs, with each pair consisting of two images and two captions. The two captions exhibit identical sets of morphemes, albeit in different orders. Different from other benchmarks that construct hard negatives by simply altering the positive texts, both positive and negative texts in Winoground are fluent, meaningful, and can match related images. Thus, we include all samples in Winoground into the SADE benchmark without further mitigation, aiming to evaluate the comprehensive multimodal compositional understanding of GVLMs, especially on the pragmatics, symbolic and series factors as introduced in (Thrush et al., 2022).

<table><tr><td rowspan="2"></td><td rowspan="2">Comprehensive</td><td colspan="2">Relation</td><td colspan="2">Attribute</td><td>Atomic</td><td>Negate</td><td colspan="2">Content</td></tr><tr><td>Winoground VL-CheckList</td><td>VG(ARO)</td><td>VL-CheckList</td><td>VG(ARO)</td><td>VG(CREPE)</td><td>VG(CREPE)</td><td>COCO</td><td>Flickr30K</td></tr><tr><td>num of images</td><td>800</td><td>5,193</td><td>2,328</td><td>5,858</td><td>5,193</td><td>1,954</td><td>1,930</td><td>2500</td><td>500</td></tr><tr><td>num of references</td><td>800</td><td>10,386</td><td>4,656</td><td>11,716</td><td>10,386</td><td>11,724</td><td>11,580</td><td>7,500</td><td>1,500</td></tr><tr><td>metrics</td><td>Group Score</td><td colspan="6">Recall@1</td><td></td><td></td></tr><tr><td>random results</td><td>16.7%</td><td>50.0%</td><td>50.0%</td><td>50.0%</td><td>50.0%</td><td>16.7%</td><td>16.7%</td><td>33.3%</td><td>33.3%</td></tr><tr><td colspan="12">Human Evaluation (closer to 0 is better)</td></tr><tr><td>origin ref. SADE ref.</td><td></td><td>3.18</td><td>1.73</td><td>0.95</td><td>3.29</td><td>1.67</td><td>2.11</td><td></td><td></td></tr><tr><td></td><td></td><td>1.40</td><td>0.62</td><td>0.35</td><td>1.01</td><td>0.94</td><td>1.63</td><td></td><td></td></tr></table>

Table 1: Taxonomy of SADE benchmark and human evaluation results on rating bias. Each branch undergoes human evaluation based on 50 reference sentences from the original dataset and 50 from SADE.

![](images/c8e0f70409e88c9ea077b387bcc27090d90636393d87b3d6aec948fc777061aa.jpg)

![](images/66300e700f8411d099a053f5bbc0011a2ed63a96f5d78ca1cfe92450f3653289.jpg)

![](images/934283bb85a7d86569768017056869cf98c698937c7220f34aec8a848ba35ce0.jpg)

![](images/d2bf39255a462b2cd6d82b959518461c9bf866645371bb3ca0a5f9c1b6a9a9bb.jpg)  
Figure 5: We visualize the distribution of SyntaxBias Score in current benchmarks. The SyntaxBias Score is defined as the difference between the LLM-based generative scores of positive and negative references. For ARO, VL-CheckList and CREPE, the distribution of the SyntaxBias Scores is situated towards the positive end (to the right of the red line), implying that these benchmarks are biased to positive captions syntactically.

## 6.2 Relations and attributes

Real-world natural scenes are inherently intricate, encompassing a multitude of specific attributes such as colors, materials, and object relationships. Models that can tackle compositional reasoning require a nuanced understanding that goes beyond mere object-level analysis. Hence, we collect relation and attribute branches from ARO (Yuksekgonul et al., 2022a) and VL-CheckList (Zhao et al., 2022). To mitigate the syntactical bias, we compute the SyntaxBias Score of the samples as described in Eqn. 4 and filter out ones that have a higher score than the threshold. The idea is to ensure that samples with strong syntactical bias are excluded for better vision-language compositional evaluation.

We choose the filtering thresholds of the SyntaxBias Score to be close to zero (specifically, by ensuring the p  value of the SyntaxBias Score is statistically below 1e 5). The filtered data includes 5,193 items from VL-CheckList and 2,328 items from Visual Genome (Krishna et al., 2017) to measure relation reasoning, and 5,858 items from VL-CheckList as well as 5,193 items from Visual Genome to evaluate attribute reasoning. Specifically, for VL-CheckList, the Relation branch contains two subclasses, i.e. action and spatial, and the Attribute branch includes action, color, material, size and state. The number of items in each subclass is elaborated in Table 1.

## 6.3 Atomic and negate

In CREPE benchmark (Ma et al., 2023), the authors propose to assess the VLMs on captions that atoms are replaced or negated. The atom replacing is like a bus with a side, light, and window v.s. a train with a side, light, and window, whereas the atom or sentence negating is as Another bowl on a cloth with an orange in it. The another bowl has a reflection and casts a shadow v.s. Another bowl on a cloth with an orange in it. The another bowl has a reflection and casts something. There is no shadow. There is a considerable proportion of reconstructed captions in CREPE that are fluent and coherent, thereby we also leverage the same method to filter the samples as we do for relations and attributes.

<table><tr><td></td><td>Comprehensive</td><td>Relation</td><td>Attribute</td><td>Atomic</td><td>Negate</td><td>Content</td></tr><tr><td>LLaVA-7B (Liu et al., 2023a)</td><td>13.00</td><td>65.52</td><td>70.55</td><td>35.01</td><td>59.01</td><td>42.02</td></tr><tr><td>LLaVA-13B (Liu et al., 2023a)</td><td>17.00</td><td>62.75</td><td>72.70</td><td>38.33</td><td>7.56</td><td>49.80</td></tr><tr><td>MiniGPT-7B (Zhu et al., 2023)</td><td>9.50</td><td>66.18</td><td>78.48</td><td>35.62</td><td>24.15</td><td>19.92</td></tr><tr><td>mPLUG-Owl (Ye et al., 2023)</td><td>11.00</td><td>65.91</td><td>69.04</td><td>34.90</td><td>54.61</td><td>35.73</td></tr><tr><td>InstructBLIP (Dai et al., 2023)</td><td>26.00</td><td>73.87</td><td>79.39</td><td>44.37</td><td>66.84</td><td>57.83</td></tr><tr><td>LLaMA Adapter V2 (Gao et al., 2023)</td><td>7.75</td><td>58.67</td><td>65.07</td><td>31.32</td><td>20.26</td><td>10.48</td></tr><tr><td>Emu (Sun et al., 2023)</td><td>4.00</td><td>68.54</td><td>85.84</td><td>51.38</td><td>87.20</td><td>2.79</td></tr></table>

Table 2: Evaluation results of GVLMs on SADE benchmark. All the models are instruction-tuned. We present the average performance of two sub-branches within the categories of Relation, Attribute and Content.

## 6.4 Replace syntactic perturbation with a content-only understanding challenge

A plethora of benchmarks perturbs the order information in the reference sentences to measure the word order sensitivity of EVLMs, which tend to treat the captions as bags of words as we present in Fig. 1. The hard negative construction methods include swapping atoms, shuffling nouns, adjectives, trigrams, and all words etc. However, due to the intrinsic syntactical awareness of LLMs, the challenge of order perturbation is not effective in assessing the visio-linguistic compositionality of GVLMs. Hence, we abandon the order challenge and propose a content-only understanding challenge.

Specifically, we modify the positive reference sentences from COCO (Lin et al., 2014) and Flickr30K (Young et al., 2014), keeping only the object- and attribute-related atoms/words. Then, we randomly select fluent, coherent and meaningful reference sentences from other datasets to serve as hard negatives, which are unrelated to the visual content. Examples of this challenging task can be found in Fig. 8 in the Appendix. The task poses a challenge and exemplifies the robustness of GVLMs against their inherent inclination towards syntactically correct reference sentences.

## 6.5 Human evaluation of SADE

In order to illustrate that our proposed SADE alleviates the syntactical bias, we ask two annotators to rate the disparity between positive and negative reference sentences. The rating score ranges from -5 to 5, where the higher the score, the more reasonable text is for positive reference sentences. Conversely, the lower the score, the more reasonable the text is for negative ones. The definition of reasonable comprises fluency, syntax, and the meaning of sentences. Note the reference sentences from the original dataset or SADE are agnostic to the annotators and we average the ratings of them. Table 1 clearly demonstrates that the reference sentences in our SADE benchmark substantially mitigate bias, as indicated by the score of human judgments approaching zero. The drop implies that the syntactical disparity between positive and negative reference sentences is drastically narrowed.

## 6.6 Results of GVLMs on SADE

Based on the SADE benchmark, we report the performance of more concurrent GVLMs based on the VisualGPTScore metric in Table 2. It can be observed that InstructBLIP (Dai et al., 2023) and Emu (Sun et al., 2023) hold the top-2 positions in almost all dimensions of our benchmark. However, the abysmal performance on Comprehensive and Content implies the vulnerability of Emu when negative reference sentences are hard and challenging. In contrast, InstructBLIP and LLaVA-13B (Liu et al., 2023a) are more robust to the Content challenge and achieve high performance on hard negatives. This provides the first de-biased and comprehensive evaluation of recent GVLMs in terms of visual compositionality. Note that we do not claim that SADE can better measure the performance of GVLMs in all aspects. However, it can better measure their compositionality with less syntactical bias, which is supported by the reduction of SyntaxBias Score and the human evaluation in Table 1. We believe this benchmark can facilitate a unified and fair comparison for future GVLM research.

## 7 Conclusion

In this work, we evaluate the compositionality of “bridge-architecture" generative VLMs via generative multimodal score, VisualGPTScore. We examine both the VisualGPTScore and current benchmarks for evaluating the multimodal compositional understanding of GVLMs. Based on the examinations, we identify the syntactical bias that exists in current datasets for GVLMs, and define the bias with SyntaxBias Score quantitatively. We then propose a SADE benchmark that mitigates the syntactical bias and provides a better content understanding evaluation for GVLMs. We report the results of multiple GVLMs on our proposed SADE benchmark and uncover new findings of the GVLMs capabilities.

## 8 Limitations

We discuss the potential limitations of this paper from two aspects. First, our proposed novel benchmark cannot be proved to better measure the performance of generative VLMs in all aspects, including emergent capability, vision understanding and complex reasoning. Our benchmark just evaluates the GVLMs in terms of VL compositionality more fairly by removing the syntactical bias in previous benchmarks. Second, our new benchmark is based on filtering the previous ones, and sampling from them to lower the SyntaxBias Score. Thus, the scale of the whole dataset is relatively small, limiting the generalization of the benchmark to some extent.

## 9 Acknowledgements

This work was supported by the National Natural Science Foundation of China (No. 62306257) and the Guangzhou Municipal Science and Technology Project (No. 2024A04J4390). This work was also supported by the Meituan Academy of Robotics Shenzhen. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the National Natural Science Foundation, Meituan, or the Guangzhou Government.

## References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. 2022. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736.

Satanjeev Banerjee and Alon Lavie. 2005. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation

measuresfor machine translation and/or summarization, pages 65–72.

Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. 2022. Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794.

Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. 2020. Uniter: Universal image-text representation learning. In European conference on computer vision, pages 104–120. Springer.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. 2023. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023).

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. 2023. Instructblip: Towards general-purpose vision-language models with instruction tuning.

Anuj Diwan, Layne Berry, Eunsol Choi, David Harwath, and Kyle Mahowald. 2022. Why is winoground hard? investigating failures in visuolinguistic compositionality. arXiv preprint arXiv:2211.00768.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. 2023. Gptscore: Evaluate as you desire. arXiv preprint arXiv:2302.04166.

Zhe Gan, Yen-Chun Chen, Linjie Li, Chen Zhu, Yu Cheng, and Jingjing Liu. 2020. Large-scale adversarial training for vision-and-language representation learning. Advances in Neural Information Processing Systems, 33:6616–6628.

Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. 2021. Clip-adapter: Better visionlanguage models with feature adapters. arXiv preprint arXiv:2110.04544.

Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. 2023. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010.

Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. 2023a. Multimodal-gpt: A vision and language model for dialogue with humans. arXiv preprint arXiv:2305.04790.

Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. 2023b. Multimodal-gpt: A vision and language model for dialogue with humans. arXiv preprint arXiv:2305.04790.

Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. 2024. Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality. Advances in Neural Information Processing Systems, 36.

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V Le, Yunhsuan Sung, Zhen Li, and Tom Duerig. 2021. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning.

Wonjae Kim, Bokyung Son, and Ildoo Kim. 2021. Vilt: Vision-and-language transformer without convolution or region supervision. In International conference on machine learning, pages 5583–5594. PMLR.

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. 2017. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal ofcomputer vision, 123:32–73.

Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. 2023a. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023b. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR.

Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. 2021. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, 34:9694–9705.

Zejun Li, Ye Wang, Mengfei Du, Qingwen Liu, Binhao Wu, Jiwen Zhang, Chengxing Zhou, Zhihao Fan, Jie Fu, Jingjing Chen, et al. 2023c. Reform-eval: Evaluating large vision language models via unified re-formulation of task-oriented benchmarks. arXiv preprint arXiv:2310.02569.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer.

Zhiqiu Lin, Xinyue Chen, Deepak Pathak, Pengchuan Zhang, and Deva Ramanan. 2023. Visualgptscore: Visio-linguistic reasoning with multimodal generative pre-training scores. arXiv preprint arXiv:2306.01879.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023a. Visual instruction tuning. arXiv preprint arXiv:2304.08485.

Jiacheng Liu, Wenya Wang, Dianzhuo Wang, Noah A Smith, Yejin Choi, and Hannaneh Hajishirzi. 2023b. Vera: A general-purpose plausibility estimation model for commonsense statements. arXiv preprint arXiv:2305.03695.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. 2023c. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281.

Zixian Ma, Jerry Hong, Mustafa Omer Gul, Mona Gandhi, Irena Gao, and Ranjay Krishna. 2023. Crepe: Can vision-language foundation models reason compositionally? In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10910–10921.

John X Morris, Eli Lifland, Jin Yong Yoo, Jake Grigsby, Di Jin, and Yanjun Qi. 2020. Textattack: A framework for adversarial attacks, data augmentation, and adversarial training in nlp. arXiv preprint arXiv:2005.05909.

OpenAI. 2023. Gpt-4 technical report.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting ofthe Associationfor Computational Linguistics, pages 311–318.

Letitia Parcalabescu, Michele Cafagna, Lilitta Muradjan, Anette Frank, Iacer Calixto, and Albert Gatt. 2021. Valse: A task-independent benchmark for vision and language models centered on linguistic phenomena. arXiv preprint arXiv:2112.07566.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR.

Kousik Rajesh, Mrigank Raman, Mohammed Asad Karim, and Pranit Chawla. 2023. Bridging the gap: Exploring the capabilities of bridge-architectures for complex visual reasoning tasks. arXiv preprint arXiv:2307.16395.

Arijit Ray, Filip Radenovic, Abhimanyu Dubey, Bryan A Plummer, Ranjay Krishna, and Kate Saenko. 2023a. Cola: How to adapt vision-language models to compose objects localized with attributes? arXiv preprint arXiv:2305.03689.

Arijit Ray, Filip Radenovic, Abhimanyu Dubey, Bryan A Plummer, Ranjay Krishna, and Kate Saenko. 2023b. Cola: How to adapt vision-language models to compose objects localized with attributes? arXiv preprint arXiv:2305.03689.

Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. 2023. Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355.

Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. 2023. Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222.

Hao Tan and Mohit Bansal. 2019. Lxmert: Learning cross-modality encoder representations from transformers. arXiv preprint arXiv:1908.07490.

Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. 2022. Winoground: Probing vision and language models for visio-linguistic compositionality. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5238– 5248.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. 2015. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4566–4575.

Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. 2023. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178.

Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. 2014. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions ofthe Associationfor Computational Linguistics, 2:67–78.

Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. 2022a. When and why vision-language models behave like bags-of-words, and what to do about it? In The Eleventh International Conference on Learning Representations.

Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. 2022b. When and why vision-language models behave like bags-of-words, and what to do about it? In The Eleventh International Conference on Learning Representations.

Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. 2021. Vinvl: Revisiting visual representations in vision-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5579–5588.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675.

Tiancheng Zhao, Tianqi Zhang, Mingwei Zhu, Haozhan Shen, Kyusong Lee, Xiaopeng Lu, and Jianwei Yin. 2022. Vl-checklist: Evaluating pre-trained visionlanguage models with objects, attributes and relations. arXiv preprint arXiv:2207.00221.

Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Krähenbühl, and Ishan Misra. 2022. Detecting twenty-thousand classes using image-level supervision. In European Conference on Computer Vision, pages 350–368. Springer.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592.

## A Appendix

## A.1 Formulations of GVLMs and EVLMs

In accordance with the discussion in the main text, we define GVLMs as models that combine visual encoders with large language models (LLMs) trained on large text corpora. The visual tokens are mapped into a lexical embedding space and harnessed to generate textual content in an autoregressive manner. Formally, given an image I and the visual encoding $g ( I )$ from encoders like Vision Transformer (Dosovitskiy et al., 2020), the mapping process can be formulated as:

$$
\boldsymbol { z } = \mathbf { M } ( g ( I ) ) , \boldsymbol { z } = \{ z _ { 1 } , z _ { 2 } , . . . , z _ { N } \} ,\tag{5}
$$

where N is the number of visual tokens and M is the mapping layers. Different from EVLMs, the training objective of multi-modal auto-regressive training is to maximize the log-likelihood of the next true token. Denote the tokenized instructions as p and the output words as $t _ { i } , ( 1 \leq i \leq K )$ , the GVLM training objective is defined as:

$$
\operatorname* { m a x } _ { \theta _ { M } , \theta _ { \sigma } } \sum _ { i = 1 } ^ { K } \log P ( t _ { i } | \boldsymbol { p } , \boldsymbol { z } , t _ { 1 } , t _ { 2 } , . . . , t _ { i - 1 } ; \theta _ { M } , \theta _ { \sigma } )\tag{6}
$$

where $\theta _ { M }$ refers to the learnable parameters of mapping layers M and $\theta _ { \sigma }$ refers to other tunable parameters like adapter layers in LLaMA-Adapter V2 (Gao et al., 2023), or visual abstractor and LoRA in mPLUG-Owl (Ye et al., 2023).

In comparison, the training objective of EVLMs is based on the ITC or ITM loss between vision and language. Given an input image I and text T, the encoded visual and linguistic features are denoted as $f _ { v }$ and $f _ { t }$ . Then, two transformation matrices $W _ { v }$ and $W _ { t }$ are employed to project the visual and text features into a joint feature embedding space, which is formulated as:

$$
v = \frac { W _ { v } ^ { \top } f _ { v } } { \vert \vert W _ { v } ^ { \top } f _ { v } \vert \vert } , u = \frac { W _ { t } ^ { \top } f _ { t } } { \vert \vert W _ { t } ^ { \top } f _ { t } \vert \vert }\tag{7}
$$

In the shared embedding space, ITC loss narrows the discrepancy of vision and language, aligning the image-text pairs in the same batch. The training objective of this process comprises two components, i.e. $\mathcal { L } _ { v  t }$ for text retrieval and $\mathcal { L } _ { t  v }$ for image retrieval. The similarity of matched pairs will be maximized while unmatched ones will be minimized. The formula is:

$$
\begin{array} { r l } & { \quad \mathcal { L } _ { I T C } = \mathcal { L } _ { v \to t } + \mathcal { L } _ { t \to v } } \\ & { = - \frac { 1 } { \left| \Omega _ { v } ^ { + } \right| } \displaystyle \sum _ { T _ { j } \in \Omega _ { v } ^ { + } } \log \frac { \exp ( v _ { i } ^ { \top } u _ { j } / \tau ) } { \sum _ { T _ { k } \in \Omega _ { t } } \exp ( v _ { i } ^ { \top } u _ { k } / \tau ) } } \\ & { \quad - \frac { 1 } { \left| \Omega _ { t } ^ { + } \right| } \displaystyle \sum _ { I _ { i } \in \Omega _ { t } ^ { + } } \log \frac { \exp ( u _ { i } ^ { \top } v _ { j } / \tau ) } { \sum _ { I _ { k } \in \Omega _ { v } } \exp ( u _ { i } ^ { \top } v _ { k } / \tau ) } } \end{array}\tag{8}
$$

where $\Omega _ { v } , \Omega _ { t }$ represent a batch of images and texts while $\Omega _ { v } ^ { + } , \Omega _ { t } ^ { + }$ denote positive subsets matched to image $I _ { i }$ and text $T _ { i }$ . ITM loss is a binary classification loss based on the joint representation of visual and linguistic features. Compared with ITC loss, ITM loss does not maximize the distance between negative pairs.

![](images/4e1d514549e44ed0d803bf88054628cb96c34fdbbce0c7abed28e0ccb025d28a.jpg)  
POS: an old person kisses a young person NEG: a young person kisses an old person  
Figure 6: An data example in current benchmarks. The image, positive and negative references are from Winoground (Thrush et al., 2022).

## A.2 Granularity influence of VisualGPTScore.

To explore the influence of granularity of references in the visio-linguistic compositional reasoning, we leverage a language model to enrich the object details and relational phrases for short references in Winoground dataset, where all references are fluent and reasonable. Vicuna- $1 3 \mathrm { B } \mathrm { - } \mathrm { v } 1 . 5 ^ { 2 }$ is adopted as the LLM, which is instruction-following tuned based on LLaMA 2 (Touvron et al., 2023b), one of the strongest open-source LLMs currently. Note that we artificially filter out nonsensical and unrelated expanded captions that are not relevant to the image and keep 282 of 400 image-text pairs finally. The expandation of references is shown in Fig. 7.

![](images/8638b382c135972a99255a9f657562c2fa575d10321dd63366140353bc833581.jpg)  
Figure 7: An LLM is leveraged to fine-grain the references.

We present the results in Table 3, and observe that the performance of “Image Score"

<table><tr><td>Models&amp;References</td><td>Text Score</td><td>Image Score</td><td>Group Score</td></tr><tr><td>LLaVA+Original</td><td>12.06</td><td>12.77</td><td>7.45</td></tr><tr><td>LLaVA+Fine-grained</td><td>8.51(-3.55)</td><td>37.23(+24.50)</td><td>6.38(-1.07)</td></tr><tr><td>MiniGPT-4+Original</td><td>18.44</td><td>17.02</td><td>9.22</td></tr><tr><td>MiniGPT-4+Fine-grained</td><td>6.03(-12.41)</td><td>31.91(+14.89)</td><td>4.96(-4.26)</td></tr></table>

Table 3: Accuracy of LLaVA and MiniGPT-4 on original and fine-grained references of filtered Winoground dataset. The definitions of Text Score, Image Score, and Group Score is specified in Winoground (Thrush et al., 2022).

has been largely improved, indicating the finegrained references are beneficial for text-to-image retrieval based on the definition of “Image Score" in Winoground.

## A.3 Zero-shot answer generation

Unlike EVLMs, GVLMs excel in zero-shot generation when guided by instructions, prompts, or demonstrations. We attempt to prompt and demonstrate the LLaVA and MiniGPT-4 to output the choices of positive or negative reference sentences based on corresponding images. However, we do not consider zero-shot generation of answers in our paper with two reasons. First, zero-shot answer generation cannot reflect the GVLMs’ compositional understanding quantitatively, without scores or probabilities to show the confidence of judgements.

Second, when demonstrating the GVLMs to generate the option number of reference sentences directly, it is hard to acquire the direct answer due to the free-form answer format, especially considering the emergent capability is limited in relatively small-scaled GVLMs. In a limited number of instances, we observed successful model outputs where options or inference processes were accurately provided, resembling the blue line in Table 4. However, in the majority of cases, the GVLMs generated fabricated answers that were characterized by a rhetorical tone, similar to the examples shown in Table 4. Also, there are cases that the rationales are correct, but the option number is wrong, conflicting with the reasoning process of GVLMs (shown in orange line in Table 4). Hence, assessing the compositionality of GVLMs solely through direct zero-shot answer generation becomes challenging, particularly when the zero-shot capability is constrained within a relatively small-scale model like the 7B variant. Furthermore, it is not possible to quantitatively analyze the alignment of a single image-text pair using this type of evaluation

![](images/0d2d90519bbda0f9558ddd85aa16f22303f33f727245a408a023e40d6c0ebb63.jpg)  
Table 4: Examples of zero-shot answer generation method. Blue: free-form generation, Teal: fabricated answers, Orange: conflicting rationales and answers.

## A.4 Examples of content challenge of SADE

We present some examples of items in the Content challenge branch in our SADE benchmark in Fig. 8. Each item comprises one positive reference sentence and two negative ones. The red texts are positive reference sentences that only kept visual content-related phrases, while the black texts are negative reference sentences that were extracted randomly from other datasets. The negative reference sentences are fluent, coherent and meaningful, but irrelevant to the contents of the images.

The pure content understanding is challenging. Specifically, the intrinsic inclination of GVLMs towards syntactic correctness drives the GVLMs to prefer negative reference sentences. From the perspective of our proposed SyntaxBias Score, the bias of our Content Challenge is opposite to the current benchmarks, which is biased to the negative reference sentences in syntax. Therefore, GVLMs have to overcome the negative bias in syntax and show the robustness of visual understanding.

![](images/59a9cd7d1112d6e60899a4fe91e965eb6c7bcfd1e95450e3a8a5bf5638bc7668.jpg)

baby , bouncy seat , boy , !ys whi" ba#room wi# a sink, !ilet, garbage can and basket kitchen wi# wooden cabinets and grani" coun"r!ps

![](images/2d09cf495e7e4df325a62c878820ad366d1ebe08d7b0cf9aef77274d17893a3d.jpg)

woman , peach tank !p , mountain bike a woman skiing down a ski slope in #e slope a group of people are in an inner tube looking boat

![](images/e1e9963049e27a7cb2b283d5a3fd1a3b45660863c96f7a3d1c69364f697b8d28.jpg)

girls , \$ee branch , dog a female is on #e compu"r playing a car game

#ere is one snowboarding going down #e hi%

![](images/a4d909259a34d8c6b16bf44fd3210f6358db6e86aaf3836fc4dd2d4c7d1c0301.jpg)

large green \$ain , wooden cra"s two black and one whi" dog in"rac&ng in #e grass a man is standing at edge of a pond, wi# two dog and is #rowing a branch is wa"

![](images/0d413b2a7ef4d5e3718d3f0fa58b89ab252d6c4bcb0da136c2e4a57eeaa9a83e.jpg)

donuts , paper , cofee cup a bearded man wearing a denim jacket sits on a bench a be%hop is pushing lu(age around inside a ho"l

![](images/c2208778308123a48761becc713a77ef35f4242af8554d59a1a0707db490fcf0.jpg)

![](images/8da485b8f45cf6475d75053cddfec1f6feaa1950cc94bdc13c6da2ac08908717.jpg)  
sma% crowd , people , doubles match , "nnis young male wi# glasses, blond-hair and beard, holding a black shovel over a campfire and a barbecue pit, fi%ed wi# red meat two people waving #eir hands in #e air and looking up  
suit case , large leaf se\*ing , car one lone army soldier overlooking an area wi# binoculars or perhaps a range finder in a sub desert area black male wearing ye%ow shirt doing a reading wi# his equipment

![](images/8b2757c88ec84d934195d337d123311888006a01da1760aaaba1a81a5b218f1d.jpg)

pla"+% , pizza , corn , cheese   
a young man holding a young woman in his arms as #ey get splashed   
by wa"r shoo&ng up ,om a fountain   
an old man wi# a beard is si\*ing on a milk cra" on #e s\$eet

![](images/8d1ecd1ac7b5b2cc3f6e7b2cc52ae5f8698ed3ce6d09af1afbc12b91afecf0d3.jpg)

ta% girafe , ta% brush

two people stand at #e peak of a mountain

two men wearing mar&al arts clo#ing are prac&cing mar&al arts

![](images/2214348bd5c41d508ecdc84f95a525a8e8b55ad6eca7599c7ab056c6972b8e13.jpg)

various elec\$onics , floo a woman wi# a drink and a woman wi# a ce%phone a man jumps rope while a crowd of people watch him

![](images/6321aec5b06e08db986cbdee5c05ec98124e456afe9c253312f503dec8239e29.jpg)

living room scene , man , young girl , wii con\$o%ers , woman a woman wearing a pink shirt showing a man wi# a s\$iped swea"r how ! do some work wi# yarn two "ams, one in pink and one in whi", play lacrosse on a fiel

Figure 8: Examples of Content challenge in our SADE benchmark. The red texts denote positive reference sentences that solely capture visual elements while disregarding sentence structure. On the other hand, the black texts represent negative reference sentences that are grammatically sound and meaningful, yet unrelated to the visual contents depicted in the images.