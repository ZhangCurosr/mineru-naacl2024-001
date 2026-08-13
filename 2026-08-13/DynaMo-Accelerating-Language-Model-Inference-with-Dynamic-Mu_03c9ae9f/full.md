# DynaMo: Accelerating Language Model Inference with Dynamic Multi-Token Sampling

Shikhar Tuli<sup>1,2</sup>\*, Chi-Heng Lin<sup>2</sup>, Yen-Chang Hsu<sup>2</sup>, Niraj K. Jha<sup>1</sup>, Yilin Shen<sup>2</sup>, Hongxia Jin<sup>2</sup>

<sup>1</sup>Department of Electrical and Computer Engineering, Princeton University

<sup>2</sup>Samsung Research America

{shikhar.tuli,chiheng.lin,yenchang.hsu,yilin.shen,hongxia.jin}@samsung.com, jha@princeton.edu

## Abstract

Traditional language models operate autoregressively, i.e., they predict one token at a time. Rapid explosion in model sizes has resulted in high inference times. In this work, we propose DynaMo, a suite of multi-token prediction language models that reduce net inference times. Our models dynamically predict multiple tokens based on their confidence in the predicted joint probability distribution. We propose a lightweight technique to train these models, leveraging the weights of traditional autoregressive counterparts. Moreover, we propose novel ways to enhance the estimated joint probability to improve text generation quality, namely co-occurrence weighted masking and adaptive thresholding. We also propose systematic qualitative and quantitative methods to rigorously test the quality of generated text for non-autoregressive generation. One of the models in our suite, DynaMo-7.3B-T3, achieves same-quality generated text as the baseline (Pythia-6.9B) while achieving 2.57 speed-up with only 5.87% and 2.67% parameter and training time overheads, respectively.

## 1 Introduction

Recent research has demonstrated the tremendous promise of large language models (LLMs) as competent artificial intelligence (AI) assistants (Touvron et al., 2023b). This has led to their rapid and widespread adoption as chatbots in diverse applications, e.g., healthcare, e-commerce, education, etc. However, the high computational requirements of LLM training and inference and the use of massive closed-source corpora have restricted their development to a few laboratories. The increasing number of open-source LLMs, including Pythia (Biderman et al., 2023) and LLaMA-2 (Touvron et al., 2023b), democratizes research in natural language processing (NLP). For instance, Vicuna-13B (Chiang et al., 2023), an instruction-finetuned

LLaMA model (Touvron et al., 2023a), has gained significant interest among researchers due to its exceptional instruction-following capabilities for its relatively compact size. Nevertheless, access and study of LLMs remain limited due to challenges involved in their efficient evaluation on resourceconstrained devices.

## 1.1 Challenges and Motivation

LLM training and inference are typically limited to large GPU clusters in data centers, causing high latencies and privacy concerns for end-users. Edge computing offers a promising solution by processing data closer to the source, reducing latency and costs while enhancing data security and privacy. However, efficient deployment of conversational AI agents on resource-constrained edge platforms remains challenging, as even compact language models result in significant latencies (Wang et al., 2020a; Tuli and Jha, 2023b). Increasing model sizes exacerbates this issue (Kaplan et al., 2020), highlighting the need for significant inference/textgeneration speed-ups and a range of models tailored to diverse platforms with varying resource constraints.

Existing models, trained with the causal language modeling (CLM) objective, predict one token at a time (Radford et al., 2019; Brown et al., 2020). We conceptualize such models as V-way (V is the vocabulary size) classifiers or unigram predictors. Mathematically, given the context, i.e., the set of past tokens $\mathbf { x } _ { 1 : t } : = \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { t } .$ , traditional LLMs model the probability distribution $p ( \mathbf { x } _ { t + 1 } | \mathbf { x } _ { 1 : t } ) ~ = ~ f _ { \theta } ( \mathbf { x } _ { 1 : t } )$ , where $f _ { \theta }$ is the LLM parameterized by θ. In this context, traditional models generate sequences of text autoregressively. In other words, we sample $\mathbf { x } _ { t + 1 }$ from $f _ { \theta } ( \mathbf { x } _ { 1 : t } )$ and then concatenate it with the input sequence to produce $\mathbf { x } _ { 1 : t + 1 } : = \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { t } , \mathbf { x } _ { t + 1 }$ . Then, we sample $\mathbf { x } _ { t + 2 }$ from the predicted distribution $f _ { \theta } ( x _ { 1 : t + 1 } )$ . Fig. 1(a) shows a schematic of this process with existing autoregressive LLMs.

![](images/bc4e0993406279f4689b1b2ca8f3f7c241508a25f4aee64f229317b1293d849f.jpg)  
Figure 1: Multi-token prediction in DynaMo. (a) Traditional autoregressive prediction requires three forward passes. (b) Non-autoregressive multi-token prediction requires only one forward pass.

Research in psycholinguistics shows that humans do not necessarily think of words one at a time when articulating thought (Sridhar, 2012); instead they employ a parallel network of cognitive and linguistic processes. In line with this, we propose predicting multiple tokens simultaneously to accelerate inference. By estimating $p ( \mathbf { x } _ { t + 1 : t + 3 } | \mathbf { x } _ { 1 : t } ) = f _ { \theta }$ (now, a V<sup>3</sup>-way classifier), we aim to achieve reliable multi-token prediction, potentially resulting in a 3 inference speed-up (assuming no latency overhead). However, simultaneous prediction of three tokens may compromise generation quality (we provide sample generations in Appendix D). Hence, there is a need to dynamically back off to lower-order n-gram prediction when the model lacks confidence.

## 1.2 Our Contributions

In this work, we propose DynaMo: a suite of dynamic multi-token prediction language models. We target inference speed-up by improving upon traditional LLMs in terms of model architecture, training methodology, and non-autoregressive decoding schemes. Further, we propose novel methods to evaluate multi-token prediction for the next generation of non-autoregressive models. More concretely, we summarize the contributions of this work next.

• We augment the suite of Pythia (Biderman et al., 2023) models for multi-token prediction. We explore various architectures for multitoken prediction (label shifts, masking strategies, multi-token heads, etc.). Further, we devise efficient ways to train augmented versions of existing pre-trained LLMs for multi-token prediction.

• We propose novel ways to dynamically predict multiple tokens based on the current context and probabilities of predicted tokens. We model the joint probability distributions of predicted tokens and back $o f f$ to lower-order n-gram prediction when the joint probabilities are not above a given threshold $( \epsilon _ { b } )$ . We propose co-occurrence weighted masking and adaptive thresholding to improve generated text quality.

• We perform rigorous experiments to evaluate the downstream performance of our proposed models. We show that training with our modified-CLM objective enhances the first token prediction quality as well. We evaluate the open-ended text generation quality of our models and its dependence on model size, desired speed-up, and multi-token prediction hyperparameters $( \mathrm { e } . \mathrm { g } . , \epsilon _ { b } )$ . In fact, this is the first non-greedy, non-batched-paralleldecoding work that proves to deliver samequality generation as the base model with systematic qualitative and quantitative tests.

The rest of the article is organized as follows. Section 3 details the multi-token prediction methodology adopted in the DynaMo suite of models along with the proposed evaluation methods. Section 4 presents the experimental results. Section 5 discusses the implications of multi-token prediction and points out future work directions. Finally, Section 6 concludes the article.

## 2 Background and Related Works

Previous research explores various approaches to reduce token prediction latency in LLMs. It includes distillation (Hinton et al., 2015), complexity reduction (Wang et al., 2020b), sparsification (Jaszczur et al., 2021), quantization (Shen et al., 2020), etc., to reduce model size or complexity, leveraging specialized hardware (Tuli and Jha, 2023a). Other engineering solutions include Flash attention (Dao et al., 2022) that reduces memory reads/writes. Recently, skeleton-of-thought decoding (Ning et al., 2023) was proposed, wherein the LLM first generates the skeleton of the answer and then conducts batched decoding to complete the contents of each skeleton point in parallel.

Speculative decoding (Stern et al., 2018; Chen et al., 2023a) is yet another approach that has gained recent prominence. It leverages a small draft model (which can be combined with the main model, Cai et al. 2023) to anticipate the main model predictions and queries it for batch verification. The batch size depends on the targeted number of token positions in the future, for draft prediction, and the number of top-k samples at each position. Despite attempts at improving inference efficiency (Spector and Re, 2023; Liu et al., 2023), such methods incur high computational overhead due to high-batch operations and result in poor compute utilization (e.g., sparse tree attention used by Cai et al. 2023; Spector and Re 2023). For the greedy decoding scheme, such methods enable up to n speed-up, however, at the cost of at least n the compute. Instead, in this work, we propose a low-compute approach that directly maps the joint probability distribution and implements cooccurrence weighted masking and adaptive thresholding, obviating the need for batched verification. Further, Medusa (Cai et al., 2023) exploits simple feed-forward layers for draft prediction. This work explores various architectural modifications for draft prediction. Nevertheless, the abovementioned approaches are orthogonal to the proposed method and can be used in conjunction to further boost performance.

## 3 Method

In this section, we discuss the implementation details of multi-token prediction in the DynaMo suite.

## 3.1 Going Beyond One-token Prediction

We propose a modified-CLM objective for multitoken prediction,

$$
\mathcal { L } _ { \mathrm { T } n } = - \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \sum _ { t = 1 } ^ { L - n + 1 } \log p ( \mathbf { x } _ { t + n } ^ { j } | \mathbf { x } _ { 1 : t } ^ { j } )\tag{1}
$$

for the $n ^ { t h }$ -token head. Here, N is the number of sequences in the training set and the length of the $j ^ { t h }$ sequence is L. The first-token head predicts the labels shifted by one position. The second-token head predicts the labels shifted by two positions, and so on. Note that the above equation trains each token head to predict the tokens independently. We approximate the joint probability distribution using independent token predictions. We represent this mathematically as follows:

$$
\begin{array} { l } { { \displaystyle p ( { \bf x } _ { t + 1 : t + n } \vert { \bf x } _ { 1 : t } ) = \prod _ { i = 1 } ^ { n } p ( { \bf x } _ { t + i } \vert { \bf x } _ { 1 : t + i - 1 } ) } \ ~ } \\ { { \displaystyle ~ \approx \prod _ { i = 1 } ^ { n } p ( { \bf x } _ { t + i } \vert { \bf x } _ { 1 : t } ) = \prod _ { i = 1 } ^ { n } f _ { \theta } ^ { i } ( { \bf x } _ { 1 : t } ) } } \end{array}\tag{2}
$$

where $f _ { \theta } ^ { i } ( \mathbf { x } _ { 1 : t } )$ is the prediction by the i-th-token head in the DynaMo model.

We use the Pythia (Biderman et al., 2023) suite of models as base models. All decoder layers up to the penultimate layer form the model “stem” (like the stem of a plant). The final decoder layer of the base model and the output embedding form the first-token-predicting head (or simply the firsttoken head). Fig. 1 shows the data flow for the base model in blue. It assumes a base model with only two decoder layers. The first layer of the base model forms the stem for the DynaMo model, while the second layer is part of the first-token head. The other decoder layers (dataflows shown in green) are part of the second- and third-token heads. The output embeddings for these heads reuse the weights of that of the first head. Hence, the extra parameters for this three-token model are from only two extra decoder layers.

Thanks to the above weight transfer process, most weights (the model stem and the first-token head) in an initialized DynaMo model are already trained. Therefore, we train the DynaMo models on a much smaller dataset (5% randomly sampled version of the Pile dataset, Gao et al. 2020) relative to that used to train the $\mathrm { P y }$ thia models. This limits the computational overhead of training our models. We provide further details on the training and evaluation methods for our models in Appendix A.1.

## 3.2 Dynamic Text Generation

Fig. 2 summarizes the proposed dynamic text generation pipeline. We extend the popular top-k sampling scheme (Fan et al., 2018; Radford et al., 2019) for autoregressive language models to multi-token generation. First, we obtain logits for all token heads. We then obtain the top-k probabilities for the predictions. Then, since we approximate the predicted tokens to be independent, we estimate the joint probability using Eq. (2). We bridge the gap between the true and the estimated (using independent predictions) joint probability distributions using co-occurrence weighted masking, taking inspiration from optimal transport (Peyré et al., 2019). We fix the sparsity in higher-dimensional distributions using adaptive thresholding and backing off to lower-order $n \cdot$ -gram prediction. We then sample from the joint probability distribution to output the generated sequence of tokens. Hence, DynaMo dynamically generates one or more tokens based on the given context and the model’s confidence in its predictions. We describe the abovementioned methods next.

![](images/ef9c732634d5654f6bf8697aca5fb96939901f0c8987a439f509ce540b723ebc.jpg)  
Figure 2: Flowchart of the proposed dynamic multi-token prediction pipeline.

## 3.2.1 Co-occurrence Weighted Masking

To bridge the gap between the true and the estimated joint probability distribution in Eq. (2), we mask the estimated distribution using the cooccurrence weights. Mathematically,

$$
\begin{array} { r l } { p ( \mathbf { x } _ { t + 1 : t + n } | \mathbf { x } _ { 1 : t } ) } & { } \\ & { \qquad = \displaystyle \prod _ { i = 1 } ^ { n } p ( \mathbf { x } _ { t + i } | \mathbf { x } _ { 1 : t } ) \frac { p ( \mathbf { x } _ { t + 1 : t + n } | \mathbf { x } _ { 1 : t } ) } { \prod _ { i = 1 } ^ { n } p ( \mathbf { x } _ { t + i } | \mathbf { x } _ { 1 : t } ) } } \\ & { \qquad \approx \displaystyle \prod _ { i = 1 } ^ { n } f _ { \theta } ^ { i } ( \mathbf { x } _ { 1 : t } ) \underbrace { \frac { \hat { p } ( \mathbf { x } _ { t + 1 : t + n } ) } { \prod _ { i = 1 } ^ { n } \hat { p } ( \mathbf { x } _ { t + i } ) } } _ { \mathrm { c o c o c u r e n c e m s u s } } } \end{array}\tag{3}
$$

where $\hat { p } \big ( \mathbf { x } _ { t + 1 : t + n } \big )$ and $\hat { p } ( \mathbf x _ { t + i } )$ are sampled estimates of the joint probability and the prediction of the i-th token, respectively. We estimate these probabilities based on the token counts in the training dataset. Note that the approximation in Eq. (3) ignores the history $\mathbf { x } _ { 1 : t }$

Theorem 1. When the cost function $\begin{array} { r l r } { c ( \mathbf { x } _ { t + 1 } , \mathbf { x } _ { t + 2 } , \dots , \mathbf { x } _ { t + n } ) } & { = } & { - \log \big ( \frac { \hat { p } ( \mathbf { x } _ { t + 1 : t + n } ) } { \prod _ { i = 1 } ^ { n } \hat { p } ( \mathbf { x } _ { t + i } ) } \big ) } \end{array}$ and $\epsilon _ { 2 } = 0$ [defined in Eq. (5)], the joint probability distribution in Eq. (3) is the optimal solution to the optimal transport problem (Peyré et al., 2019).

We describe the optimal transport problem in the multi-token prediction setting and provide a proof of the above theorem in Appendix B.

## 3.2.2 Dynamic Back-off and Adaptive Thresholding

Intuitively, when generating multiple tokens, the goal is to find the peaks in the predicted joint probability distribution and sample those peaks. If none of the probability values is beyond a threshold (determined by $\epsilon _ { b } )$ , i.e., there are no peaks in the joint probability distribution, our model backs off to lower-order n-gram prediction. To implement this, we adopt a static threshold $\epsilon _ { b } .$ . If no probability value is $> \epsilon _ { b } ^ { n - 1 }$ , we back off to sampling a lowerorder joint probability distribution. We set all probabilities less than $\epsilon _ { b }$ to 0.

Static thresholding is too naïve for joint probability distributions, which can vary with the predicted tokens and input context. Taking inspiration from computer vision methods, we test adaptive thresholding, leveraging Otsu’s binarization algorithm (Otsu, 1979). It adapts the threshold for dynamic back-off based on the predicted joint probability distribution. We apply adaptive thresholding on top of the static thresholding explained above. In other words, we first set all values in the joint probability distribution less than $\epsilon _ { b }$ to $0 .$ Then, we set all values less than $\epsilon _ { \mathrm { A T } }$ to 0 (where $\epsilon _ { \mathrm { A T } }$ is the threshold found using Otsu’s algorithm). In the computer vision domain, researchers implement Otsu’s algorithm after applying Gaussian blur to the input image. We thus explore the effect of using Gaussian blur and adaptive thresholding on the predicted joint probability distribution (ablation analysis in Appendix C.1).

Alg. 1 summarizes the multi-token generation algorithm. We depict the probability distribution output by the i-th-token head by $f _ { \theta } ^ { i } .$ . This probability distribution is a vector of length V (or k after top-k sampling). We calculate the joint probability distribution J by taking the outer product of the individual token predictions. The function adaptiveThresholding (line 9) implements adaptive thresholding explained above. The function penalizeRepetition (line 11) divides all probabilities that correspond to repetitions by a penalty value (Keskar et al., 2019). The sample function (lines 15 and 19) samples the tokens using multinomial sampling, i.e., weighted by the corresponding probability values. Based on n, we output the sequence of generated tokens ${ \bf x } _ { t + 1 : t + n }$ . For the proposed set of DynaMo models, we initialize $n = 3 .$ Thus, we dynamically generate new tokens depending on the output predictions (and the corresponding probabilities). A low value of $\epsilon _ { b }$ generates more tokens (a three-token model with $\epsilon _ { b } = 0$ will always generate three tokens). On the other hand, a high value of $\epsilon _ { b }$ results in few tokens being generated $( \epsilon _ { b } = 1$ will always generate only one token).

Algorithm 1 DynaMo multi-token generation   
Require: input sequence $\mathbf { x } _ { 1 : t }$ , DynaMo model   
with token heads $f _ { \theta } ^ { i } , \forall i = 1 , \ldots , n .$   
1: $p ( \mathbf { x } _ { t + 1 } | \mathbf { x } _ { 1 : t } ) \gets f _ { \theta } ^ { 1 } ( \mathbf { x } _ { 1 : t } ) .$   
2: $p ( \mathbf { x } _ { t + 2 } | \mathbf { x } _ { 1 : t } ) \gets f _ { \theta } ^ { 2 } ( \mathbf { x } _ { 1 : t } ) .$   
3: $p ( \mathbf { x } _ { t + 3 } | \mathbf { x } _ { 1 : t } ) \gets f _ { \theta } ^ { 3 } ( \mathbf { x } _ { 1 : t } ) .$   
4: $n = 3$ (for three-token model)   
5: while $n > 1$ do   
6: Obtain top-k values for token predictions   
$p ( \mathbf { x } _ { t + i } | \mathbf { x } _ { 1 : t } )$   
7: $\begin{array} { r } { \mathbf { J }  \prod _ { i = 1 } ^ { n } f _ { \theta } ^ { i } ( \mathbf { x } _ { 1 : t } ) \frac { \hat { p } ( \mathbf { x } _ { t + 1 : t + n } ) } { \prod _ { i = 1 } ^ { n } \hat { p } ( \mathbf { x } _ { t + i } ) } } \end{array}$   
8: ▷ Co-occurrence weighted masking   
9: J  adaptiveThresholding(J)   
10: ▷ Adaptive thresholding   
11: J  penalizeRepetition(J)   
12: if $j < \epsilon _ { b } ^ { n - 1 } , \forall j \in \mathbf { J }$ then   
13: $n \gets n - 1$ ▷ Back-off   
14: else   
15: x<sub>t+1:t+n</sub> sample(J)   
16: return x<sub>t+1:t+n</sub>   
17: end if   
18: end while   
19: return $\mathbf { x } _ { t + 1 : t + n } \gets s a m p 1 \mathrm { e } ( p ( \mathbf { x } _ { t + 1 } | \mathbf { x } _ { 1 : t } ) )$

## 3.3 Evaluation Methods

We propose various methods to evaluate our multitoken models. They include evaluating singletoken prediction on standard natural language understanding (NLU) benchmarks, multi-token perplexity, and open-ended generation performance.

## 3.3.1 NLU Benchmarks

Evaluating multi-token prediction on NLU benchmarks is challenging. This is because most downstream benchmarks only require one-token prediction. However, we hypothesize that training a multi-token prediction transformer results in better prediction of even the first token. We call this a better transformer. We evaluate our models on popular benchmarks with the first-token head. We use the lm-evaluation-harness (Gao et al., 2021) to carry out our evaluations on common benchmarks in both zero-shot and few-shot settings. For fair comparisons, we report the performance of the corresponding base Pythia model as well.

## 3.3.2 Multi-token Perplexity

To test multi-token text generation quality, we evaluate the models based on perplexity. However, the traditional definition of perplexity is only defined for single token prediction. We extend this to $n ^ { t h }$ token prediction and also n-gram prediction. Mathematically,

$$
\begin{array} { r l } & { \mathrm { P P L } _ { n } = \exp \left( - \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T - n } \log p ( \mathbf { x } _ { t + n } | \mathbf { x } _ { 1 : t } ) \right) , } \\ & { \mathrm { P P L } _ { 1 : n } = \exp \left( - \displaystyle \frac { 1 } { n T } \sum _ { t = 1 } ^ { T - n } \log p ( \mathbf { x } _ { t + 1 : t + n } | \mathbf { x } _ { 1 : t } ) \right) } \end{array}\tag{4}
$$

For a three-token model, we calculate PPL<sub>1</sub>, $\mathrm { P P L _ { 1 : 2 : } }$ , and $\mathrm { P P L _ { 1 : 3 } }$ . We can also extend perplexity calculation to dynamic multi-token prediction, wherein we decide n based on the joint probability distribution and the back-off threshold. We refer to it as $\mathrm { P P L _ { d } . }$ . It varies with $\epsilon _ { b }$

## 3.4 Open-ended Text Generation

Perplexity is a very restrictive evaluation measure. It constrains model text generation to the text in the validation set. A fairer approach to test multitoken generation would be to evaluate open-ended generated texts. Zheng et al. (2023) propose using strong LLMs like GPT-3.5 (OpenAI, 2023a) and GPT-4 (OpenAI, 2023b) and show that they can match both controlled and crowdsourced human preferences in evaluating generated texts well. Since human evaluation of open-ended generated texts from our models would be very expensive and time-consuming, we use a strong LLM to evaluate the quality of generated text from our DynaMo suite of models.

Table 1: Zero-shot performance on common sense reasoning tasks.
<table><tr><td>Model</td><td>ARC-c</td><td> $\mathbf { A R C { \cdot } e }$ </td><td>BoolQ</td><td>COPA</td><td>HellaSwag</td><td>OBQA</td><td>PIQA</td><td>WinoG</td></tr><tr><td>Pythia-70M</td><td> $1 5 . 5 _ { \pm 1 . 0 }$ </td><td> $3 8 . 7 _ { \pm 1 . 0 }$ </td><td> ${ \bf 5 5 . 9 2 0 . 8 }$ </td><td> $5 3 . 0 _ { \pm 5 . 0 }$ </td><td> $2 6 . 6 _ { \pm 0 . 4 }$ </td><td> $1 4 . 6 _ { \pm 0 . 2 }$ </td><td> $5 8 . 6 _ { \pm 1 . 2 }$ </td><td> ${ \bar { \bf 5 0 . 8 } } _ { \pm 1 . 4 }$ </td></tr><tr><td>DynaMo-77M-T3</td><td> ${ \bf 1 7 . 3 _ { \pm 1 . 1 } }$ </td><td> ${ \bf 4 1 . 0 _ { \pm 1 . 0 } }$ </td><td> $5 5 . 7 _ { \pm 0 . 9 }$ </td><td> ${ \bar { \mathbf { 5 6 . 0 } } } _ { \pm 5 . 0 }$ </td><td> $\mathbf { 2 6 . 9 2 0 . 4 }$ </td><td> ${ \bf 1 4 . 7 _ { \pm 1 . 6 } }$ </td><td> $\mathbf { 5 9 . 8 _ { \pm 1 . 1 } }$ </td><td> $4 9 . 8 _ { \pm 1 . 4 }$ </td></tr><tr><td>Pythia-160M DynaMo-180M-T3</td><td> ${ \bf 2 0 . 7 } _ { \pm 1 . 2 }$   $1 9 . 4 _ { \pm 1 . 1 }$ </td><td> $4 4 . 0 { \scriptstyle \pm 1 . 0 }$   ${ \pm } 5 . 3 _ { \pm 1 . 0 }$ </td><td> ${ \bf 4 9 . 4 } _ { \pm 0 . 9 }$   $4 8 . 0 { \scriptstyle \pm 0 . 9 }$ </td><td> $6 5 . 0 { \scriptstyle \pm 4 . 8 }$   ${ \bf 6 6 . 0 _ { \pm 4 . 8 } }$ </td><td> $2 9 . 1 _ { \pm 0 . 5 }$   $2 9 . 3 _ { \pm 0 . 5 }$ </td><td> ${ \bf 1 7 . 0 _ { \pm 1 . 7 } }$   $1 6 . 6 { \scriptstyle \pm 1 . 7 }$ </td><td> $6 2 . 0 { \scriptstyle \pm 1 . 1 }$   ${ \bf 6 2 . 7 _ { \pm 1 . 1 } }$ </td><td> $5 0 . 6 _ { \pm 1 . 4 }$   ${ \bf 5 1 . 7 _ { \pm 1 . 4 } }$ </td></tr><tr><td>Pythia-410M DynaMo-430M-T3</td><td> $2 0 . 5 _ { \pm 1 . 2 }$   ${ \bf 2 1 . 2 _ { \pm 1 . 2 } }$ </td><td> $5 1 . 6 _ { \pm 1 . 0 }$   ${ \pm 2 . 6 _ { \pm 1 . 0 } }$ </td><td> ${ \bf 5 8 . 6 _ { \pm 0 . 9 } }$   $5 7 . 1 _ { \pm 0 . 9 }$ </td><td> ${ \bf 7 1 . 0 _ { \pm 4 . 6 } }$   $7 0 . 0 { \scriptstyle \pm 4 . 6 }$ </td><td> $3 4 . 5 _ { \pm 0 . 5 }$   $\mathbf { 3 4 . 6 _ { \pm 0 . 5 } }$ </td><td> $1 7 . 8 _ { \pm 1 . 7 }$   $\mathbf { 1 7 . 9 } _ { \pm 1 . 7 }$ </td><td> $6 7 . 2 { \scriptstyle \pm 1 . 1 }$   ${ \bf 6 7 . 5 _ { \pm 1 . 1 } }$ </td><td> ${ \pm } 3 . 3 { \scriptstyle \pm } 1 . 4 $   ${ \pm } 3 . 3 { \scriptstyle \pm } 1 . 4 $ </td></tr><tr><td>Pythia-1B</td><td> $2 4 . 3 _ { \pm 1 . 2 }$ </td><td> ${ \pm } 8 . 5 _ { \pm 1 . 0 }$ </td><td> $6 0 . 8 { \scriptstyle \pm 0 . 9 }$ </td><td> $7 4 . 0 _ { \pm 4 . 4 }$ </td><td> ${ \bf 3 8 . 9 2 0 . 5 }$ </td><td> $2 1 . 8 { \scriptstyle \pm 1 . 8 }$ </td><td> $7 0 . 1 _ { \pm 1 . 1 }$ </td><td> $5 2 . 9 _ { \pm 1 . 4 }$ </td></tr><tr><td>DynaMo-1.1B-T3</td><td> $2 5 . 3 _ { \pm 1 . 3 }$ </td><td> $5 8 . 4 _ { \pm 1 . 0 }$ </td><td> ${ \bf 6 0 . 9 2 0 . 9 }$ </td><td> ${ 7 6 . 0 \pm 4 . 3 }$ </td><td> ${ \bf 3 8 . 9 2 0 . 5 }$ </td><td> $2 2 . 2 _ { \pm 1 . 9 }$ </td><td> ${ \bf 7 0 . 2 _ { \pm 1 . 1 } }$ </td><td> ${ \pm } 3 . 8 _ { \pm 1 . 4 }$ </td></tr><tr><td>Pythia-1.4B</td><td> $2 7 . 3 { \scriptstyle \pm 1 . 3 }$ </td><td> ${ \bf 6 1 . 8 _ { \pm 1 . 0 } }$ </td><td> $5 8 . 0 { \scriptstyle \pm 0 . 9 }$ </td><td> $7 6 . 0 { \scriptstyle \pm 4 . 3 }$ </td><td> $4 1 . 7 _ { \pm 0 . 5 }$ </td><td> ${ \bf 2 2 . 8 _ { \pm 1 . 9 } }$ </td><td> $7 2 . 0 { \scriptstyle \pm 1 . 0 }$ </td><td> $\pm 6 . 9 { \scriptstyle \pm 1 . 4 }$ </td></tr><tr><td>DynaMo-1.5B-T3</td><td> $2 7 . 7 _ { \pm 1 . 3 }$ </td><td> $6 1 . 5 _ { \pm 1 . 0 }$ </td><td> ${ \bf 5 9 . } 2 _ { \pm 0 . 9 }$ </td><td> ${ \bf 7 8 . 0 _ { \pm 4 . 2 } }$ </td><td> ${ \bf 4 1 . 9 { _ { \pm 0 . 5 } } }$ </td><td> $2 2 . 4 _ { \pm 1 . 9 }$ </td><td> $7 2 . 5 _ { \pm 1 . 0 }$ </td><td> $5 6 . 0 { \scriptstyle \pm 1 . 4 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pythia-2.8B</td><td> $2 9 . 9 { \scriptstyle \pm 1 . 3 }$ </td><td> $5 3 . 5 { \scriptstyle \pm 1 . 0 }$ </td><td> ${ \bf 6 4 . 2 } _ { \pm 0 . 8 }$ </td><td> $7 5 . 0 { \scriptstyle \pm 4 . 4 }$ </td><td> $4 5 . 4 { \scriptstyle \pm 0 . 5 }$ </td><td> $2 4 . 0 { \pm } 1 . 9$ </td><td> $7 4 . 1 { \pm } 1 . 0$ </td><td></td></tr><tr><td>DynaMo-2.9B-T3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $5 8 . 2 { \scriptstyle \pm 1 . 4 }$ </td></tr><tr><td></td><td> ${ \bf 3 0 . 4 } _ { \pm 1 . 3 }$ </td><td> ${ \bf 6 4 . 7 _ { \pm 1 . 0 } }$ </td><td> $6 4 . 0 { \scriptstyle \pm 0 . 8 }$ </td><td> ${ \bf 8 0 . 0 _ { \pm 4 . 0 } }$ </td><td> $4 5 . 7 _ { \pm 0 . 5 }$ </td><td> ${ 2 4 . 3 } _ { \pm 1 . 9 }$ </td><td> ${ 7 4 . 2 } _ { \pm 1 . 0 }$ </td><td> ${ \bf 5 9 . 1 _ { \pm 1 . 4 } }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pythia-6.9B</td><td> $3 3 . 2 _ { \pm 1 . 4 }$ </td><td> ${ \bf 6 8 . 5 _ { \pm 1 . 0 } }$ </td><td> $6 4 . 4 _ { \pm 0 . 8 }$ </td><td> $7 4 . 0 _ { \pm 4 . 4 }$ </td><td> $4 9 . 6 _ { \pm 0 . 5 }$ </td><td> $2 7 . 0 { \scriptstyle \pm 1 . 9 }$ </td><td> $7 5 . 7 _ { \pm 1 . 0 }$ </td><td></td></tr><tr><td>DynaMo-7.3B-T3</td><td> ${ 3 3 . 6 } _ { \pm 1 . 4 }$ </td><td> $6 8 . 1 _ { \pm 1 . 0 }$ </td><td> ${ \bf 6 5 . 1 { \scriptstyle \pm 0 . 8 } }$ </td><td> ${ 7 6 . 0 \pm 4 . 3 }$ </td><td> ${ \bf 4 9 . 9 2 0 . 5 }$ </td><td> $\mathbf { 2 8 . 0 { \overset { . } { \bot } } 2 . 0 }$ </td><td> $7 5 . 7 _ { \pm 1 . 0 }$ </td><td> $6 2 . 7 _ { \pm 1 . 4 }$ </td></tr></table>

instruction-finetuned DynaMo models on the Vicuna benchmark. We use the Alpaca dataset (Taori et al., 2023) filtered by GPT-3.5 for high-quality instruction-response pairs (Chen et al., 2023b). The dataset contains 9,229 instruction-response pairs. We follow the evaluation setup from (Zheng et al., 2023).

## 4 Experiments

Vicuna and MT benchmarks (Zheng et al., 2023) require the pre-trained LLM to be finetuned on instruction-following datasets. To disambiguate the effect of instruction finetuning, we evaluate our models with different target speed-ups on a novel sentence-completion benchmark. The task is to complete a sentence for a given prompt. We categorize the sentences into simple declarative, compound declarative, W/H interrogative, Y/N interrogative, affirmative imperative, negative imperative, and exclamatory. We test the text generations of our models for grammatical correctness, creativity, depth, logical flow, coherence, and informativeness of the generated text. The benchmark has ten prompts. For every prompt, we generate ten sentences with different random seeds for every $\epsilon _ { b } \in \{ 0 . 0 0 , 0 . 0 2 , \hdots , 1 . 0 0 \}$ . Thus, for every model, we generate 5100 sentences at different speed-ups. We evaluate the quality of every generated sentence using single-mode and pairwise evaluations. For single-mode evaluation, we ask GPT-3.5 to score the generated response from one to ten. For pairwise evaluation, we ask GPT-3.5 to compare the response against one generated by the corresponding Pythia base model. DynaMo either wins, loses, or ties against the baseline Pythia model. We provide further details on the sentence completion benchmark along with the evaluation setup in Appendix A.3.

In this section, we present experimental results and comparisons of the proposed approach with the Pythia baseline, which we used to instantiate the DynaMo models. We provide test results for architectural and training variations in multi-token prediction in Appendix C.2.

## 4.1 Downstream Performance

We hypothesize that training the decoder layers using the second- and third-token loss terms makes them better. We test this hypothesis next.

Finally, we also evaluate the performance of

We consider eight standard common sense reasoning benchmarks: ARC challenge (ARCc) and ARC easy (ARC-e, Clark et al. 2018), BoolQ (Clark et al., 2019), COPA (Roemmele et al., 2011), HellaSwag (Zellers et al., 2019), OpenBookQA (OBQA, Mihaylov et al. 2018), PIQA (Bisk et al., 2020), and WinoGrande (WinoG, Sakaguchi et al. 2021). We perform evaluations in the zero-shot setting as done in the language modeling community. Table 1 shows a comparison between each model in the DynaMo suite with that of the corresponding baseline Pythia model. As we can see, DynaMo models outperform their respective baselines on most benchmarks. We report additional downstream performance results in Appendix C.3.

Table 2: Multi-token perplexity results for models in the DynaMo and Pythia suites.
<table><tr><td>Model</td><td> $\mathbf { P P L _ { 1 } }$ </td><td> $\mathbf { P P L _ { 2 } }$ </td><td> $\mathbf { P P L _ { 3 } }$ </td><td> $\mathbf { P P L _ { 1 : 2 } }$ </td><td> $\mathbf { P P L _ { 1 : 3 } }$ </td></tr><tr><td>Pythia-70M</td><td> $2 0 . 2 { \scriptstyle \pm 1 . 5 }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>DynaMo-77M-T3</td><td> ${ \bf 1 8 . 3 _ { \pm 1 . 5 } }$ </td><td> $1 1 1 . 4 _ { \pm 1 . 7 }$ </td><td> $2 6 2 . 0 { \scriptstyle \pm 1 . 6 }$ </td><td> $4 5 . 2 _ { \pm 1 . 5 }$ </td><td> $8 1 . 2 _ { \pm 1 . 6 }$ </td></tr><tr><td>Pythia-160M DynaMo-180M-T3</td><td> $1 3 . 5 { \scriptstyle \pm 1 . 4 }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>Pythia-410M</td><td> $\pm 2 . 9 _ { \pm 1 . 4 }$   $9 . 9 _ { \pm 1 . 4 }$ </td><td> $7 8 . 5 { \scriptstyle \pm 1 . 6 }$ </td><td> $1 9 9 . 4 _ { \pm 1 . 6 }$ </td><td> $3 1 . 8 _ { \pm 1 . 5 }$ </td><td> $5 8 . 7 _ { \pm 1 . 5 }$ </td></tr><tr><td>DynaMo-430M-T3</td><td> ${ \bf 9 . 6 _ { \pm 1 . 4 } }$ </td><td> $5 9 . 8 { \scriptstyle \pm 1 . 6 }$ </td><td> $1 6 2 . 4 _ { \pm 1 . 6 }$ </td><td> $2 4 . 0 { \scriptstyle \pm 1 . 5 }$ </td><td> $4 5 . 4 _ { \pm 1 . 5 }$ </td></tr><tr><td>Pythia-1B</td><td> $8 . 5 { \scriptstyle \pm 1 . 4 }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>DynaMo-1.1B-T3</td><td> ${ \bf 8 . 4 } \pm 1 . 4$ </td><td> $4 4 . 1 { \pm } 1 . 6 $ </td><td> $1 1 6 . 6 { \scriptstyle \pm 1 . 7 }$ </td><td> $1 9 . 3 { \scriptstyle \pm 1 . 5 }$ </td><td>35.1±1.6</td></tr><tr><td>Pythia-1.4B</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DynaMo-1.5B-T3</td><td> $7 . 9 { \pm } 1 . 6 $ </td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> ${ \bf 7 . 8 { \scriptstyle \pm 1 . 6 } }$ </td><td> $4 1 . 9 { \scriptstyle \pm 2 . 0 }$ </td><td> $1 1 2 . 7 { \scriptstyle \pm 2 . 1 }$ </td><td> $1 8 . 3 { \pm } 1 . 9$ </td><td> $3 3 . 6 { \scriptstyle \pm 1 . 9 }$ </td></tr><tr><td>Pythia-2.8B</td><td> $7 . 4 { \pm } 1 . 6 $ </td><td></td><td></td><td></td><td></td></tr><tr><td>DynaMo-2.9B-T3</td><td> ${ \bf 7 . 1 _ { \pm 1 . 9 } }$ </td><td> $3 7 . 1 _ { \pm 2 . 7 }$ </td><td> $1 0 0 . 3 _ { \pm 3 . 0 }$ </td><td> $1 6 . 2 _ { \pm 2 . 2 }$ </td><td> $2 9 . 8 _ { \pm 2 . 4 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pythia-6.9B</td><td> $6 . 6 { \scriptstyle \pm 1 . 8 }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>DynaMo-7.3B-T3</td><td> ${ \bf 6 . 5 { \scriptstyle \pm 1 . 8 } }$ </td><td> $3 1 . 4 { \scriptstyle \pm 2 . 6 }$ </td><td> $8 3 . 5 { \scriptstyle \pm 3 . 0 }$ </td><td> $1 4 . 4 _ { \pm 2 . 2 }$ </td><td> $2 5 . 8 { \scriptstyle \pm 2 . 4 }$ </td></tr></table>

## 4.2 Multi-token Perplexity

Table 2 shows the multi-token perplexity on the validation set for all models in the DynaMo and Pythia suites. The DynaMo models achieve lower PPL<sub>1</sub> relative to their Pythia counterparts due to further training of the first-token head and better decoder layers in the model stem (i.e., all layers up to the penultimate layer). We provide further test results in Appendix C.2. The multi-token perplexity drops as models become larger, making the prediction of multiple tokens easier and better. We describe results for dynamic multi-token perplexity $\mathrm { ( P P L _ { d } ) }$ in Appendix C.4.

## 4.3 Text Generation Performance and Speed-up

We now compare the open-ended text generation performance of the DynaMo models with that of the baseline Pythia models on the sentencecompletion benchmark.

Since pairwise evaluations by strong LLMs better align with human evaluations (Zheng et al., 2023), we evaluate our models against the Pythia baseline in the pairwise mode (details in Appendix A.3; single-mode evaluations in Appendix C.5.1). As $\epsilon _ { b }$ increases, the text quality improves, but the speed-up decreases. Thus, the win rate (i.e., the number of wins/losses against the baseline) decreases as speed-up increases.

Fig. 3 shows the effect of speed-up on the win rate of the proposed models (we describe how we obtain this plot in Appendix C.5.2). When the win rate is 1.0, the text generation quality would, on average, be the same for the models being compared. We call the speed-up for this case the “same-quality speed-up.” If the win rate for a model is always greater than 1.0, we extrapolate the plot to obtain the “theoretical same-quality speed-up.” However, in further discussions, we refer to the minimum of (theoretical) same-quality speed-up and 3 (for three-token models) as, simply, the “speed-up.”

![](images/05dd1fcc8b03f88972c1934e8f9804c926767ba4ddc8591b16dd40637b47cd51.jpg)  
Figure 3: Win rate vs. speed-up for pairwise comparisons on the sentence-completion benchmark with corresponding Pythia models as baselines. GPT-3.5 is used as a judge. Regression plotted with 95% confidence intervals. Same-quality speed-ups are shown in parentheses. Theoretical same-quality speed-ups are marked with an asterisk (\*).

## 4.4 Instruction Finetuning

We finetune models in the Pythia and DynaMo suites on an instruction-following dataset (details in Section 3.4). Fig. 4 shows the pairwise performance of the DynaMo (with respect to Pythia) models on the Vicuna benchmark (Zheng et al., 2023). We run the DynaMo models at different speed-ups (we set $\epsilon _ { b } = 1 . 0 , 0 . 7 5 , 0 . 5 , 0 . 2 5 , 0 . 0 )$ shown on the x-axis. We compare each model against the corresponding Pythia baseline. In the case of comparisons with small models, neither model results in a reasonable answer. Hence, GPT-4 classifies many response pairs as ties. The number of ties decreases as model sizes increase. As the speed-up increases, the win rate decreases. DynaMo-7.3B-T3 provides around the same-quality responses as Pythia-6.9B (win rate = 0.98) even for a high speed-up of 2.57 (we ablate the effect of dynamic text generation methods in Appendix C.1).

## 5 Discussion

In this section, we discuss the implications of the proposed DynaMo suite of multi-token prediction models and future work directions.

![](images/80aeec758623b8b0aa9b18d24c15dd96f78d9f0b12e82dec74c3be87654fcfaa.jpg)  
Figure 4: Pairwise performance of the DynaMo and Pythia models on the Vicuna benchmark. GPT-4 was used as a judge. The actual number of wins, ties, and losses are colored green, yellow, and red, respectively.

Table 3: Effect of better transformer training on zero-shot performance in common sense tasks.
<table><tr><td>Model</td><td>|ARC-c</td><td>ARC-e</td><td>BoolQ</td><td>COPA</td><td>HellaSwag</td><td>OBQA</td><td>PIQA</td><td>WinoG</td></tr><tr><td>Pythia-70M</td><td> $1 5 . 5 _ { \pm 1 . 0 }$ </td><td> $3 8 . 7 _ { \pm 1 . 0 }$ </td><td> ${ \bar { \bf 5 5 . 9 } } _ { \pm 0 . 8 }$ </td><td> $5 3 . 0 { \scriptstyle \pm 5 . 0 }$ </td><td> $2 6 . 6 { \scriptstyle \pm 0 . 4 }$ </td><td> $1 4 . 6 _ { \pm 0 . 2 }$ </td><td> $5 8 . 6 _ { \pm 1 . 2 }$ </td><td> $5 0 . 8 _ { \pm 1 . 4 }$ </td></tr><tr><td>Pythia-70M+</td><td> $1 5 . 6 _ { \pm 1 . 0 }$ </td><td> $3 8 . 8 _ { \pm 1 . 0 }$ </td><td> ${ \bar { \bf 5 5 . 9 } } _ { \pm 0 . 8 }$ </td><td> $5 3 . 1 _ { \pm 5 . 0 }$ </td><td> $2 6 . 8 { \scriptstyle \pm 0 . 4 }$ </td><td> $1 4 . 6 _ { \pm 0 . 2 }$ </td><td> $5 8 . 6 _ { \pm 1 . 2 }$ </td><td> ${ \bf 5 0 . 9 } _ { \pm 1 . 4 }$ </td></tr><tr><td>DynaMo-77M-T3</td><td> ${ \bf 1 7 . 3 _ { \pm 1 . 1 } }$ </td><td> ${ \bf 4 1 . 0 _ { \pm 1 . 0 } }$ </td><td> $5 5 . 7 _ { \pm 0 . 9 }$ </td><td> $\mathbf { 5 6 . 0 } \mathbf { \pm } \mathrm { 5 . 0 }$ </td><td> $\mathbf { 2 6 . 9 2 0 . 4 }$ </td><td> ${ \bf 1 4 . 7 _ { \pm 1 . 6 } }$ </td><td> $\mathbf { 5 9 . 8 _ { \pm 1 . 1 } }$ </td><td> $4 9 . 8 _ { \pm 1 . 4 }$ </td></tr></table>

## 5.1 Effect of Better Transformer Training

Another observation that supports the hypothesis that better transformer training results in superior first-token prediction is as follows. For fair comparisons, we test our three-token model against Pythia-70M further trained on the 5% Pile dataset using a learning rate of $1 0 ^ { - 5 }$ (we refer to this version as Pythia- $\mathbf { \partial } ^ { . 7 0 \mathbf { M } ^ { + } } ,$ ) on commonsense tasks. We present the result in Table 3 (perplexity results in Appendix C.2.2). Training the decoder layers based on the modified-CLM loss in Eq. (1) results in better first-token prediction, which we use to evaluate common sense tasks as presented here. This key result is worth further exploration, which we leave for future work.

## 5.2 Contribution of Unigram, Bigram, and Trigram Generations to Speed-up

Fig. 5 shows the percentage of one-token, twotoken, and three-token generations as we sweep $\epsilon _ { b }$ with DynaMo-70M-T3. When $\epsilon _ { b } = 1 . 0$ , the model always generates one token at a time. When $\epsilon _ { b } = 0 . 0$ , the model always generates three tokens at a time, regardless of its confidence in the generations. Surprisingly, we note that the contribution of two-token generations is low; the model banks on three-token generations instead. We defer further exploration to balance multi-token generations during dynamic back-off to future work.

![](images/6c68ff3a97ca19a29d59e6d602f83082067428235d8ed564458507efef516176.jpg)  
Figure 5: Percentage of unigram, bigram, and trigram generations vs. ϵ<sub>b</sub> for DynaMo-70M-T3.

## 5.3 Baseline Comparisons

Table 4 shows comparisons with other approaches that target inference speed-up. Speculative sampling (Chen et al., 2023a) and skeleton-of-thought decoding (Ning et al., 2023) are orthogonal to the DynaMo approach and can be used in conjunction with the proposed multi-token generation scheme to boost performance further. Nevertheless, DynaMo can be seen to require the least overhead in FLOPS-per-generation and provide the highest speed-up. The high computational efficiency of DynaMo is attributed to its avoidance of high-batch operations necessitated by speculative sampling and skeleton-of-thought decoding.

Table 4: Comparisons with other approaches. ∗Ning et al. (2023) evaluate models of different sizes.
<table><tr><td>Method</td><td>Base Model Size</td><td>FLOPS Overhead</td><td>Speed-up</td></tr><tr><td>Speculative Sampling</td><td>70B</td><td>340%</td><td>1.92-2.46×</td></tr><tr><td>Skeleton-of-Thought</td><td>7B-13B*</td><td>560%</td><td>1.13-2.39×</td></tr><tr><td>RecycleGPT</td><td>一 1.3B</td><td>15%</td><td>1.34-1.40×</td></tr><tr><td>DynaMo-77M-T3</td><td>70M</td><td>8.95%</td><td>3.00×</td></tr><tr><td>DynaMo-180M-T3</td><td>160M</td><td>8.73%</td><td>2.19×</td></tr><tr><td>DynaMo-430M-T3</td><td>410M</td><td>6.22%</td><td>3.00×</td></tr><tr><td>DynaMo-1.1B-T3</td><td>1B</td><td>9.95%</td><td>2.15×</td></tr><tr><td>DynaMo-1.5B-T3</td><td>1.4B</td><td>7.12%</td><td>2.07×</td></tr><tr><td>DynaMo-2.5B-T3</td><td>2.4B</td><td>5.67%</td><td>2.06×</td></tr><tr><td>DynaMo-7.3B-T3</td><td>6.9B</td><td>5.87%</td><td>2.57×</td></tr></table>

## 5.4 How Many Tokens Can We Simultaneously Predict?

Fig. 6 shows the win rates with respect to speedups on the sentence-completion benchmark using pairwise analysis against Pythia-70M (see Section 3.4 and Appendix A.3). DynaMo-77M-T3 shows much better win rates relative to DynaMo-74M-T2 for speed-ups < 2.0 despite similar PPL<sub>1:2</sub>. Further, DynaMo-77M-T3, being a three-token model, can provide much higher speed-ups than DynaMo-74M-T2, however, at the cost of a slight parameter overhead. Since the extra parameter overhead is marginal, especially for larger models, we stick with three-token models.

We also explore simultaneous token prediction beyond the three-token model. Fig. 6 also shows the performance of DynaMo-80M-T4. Due to the better transformer training through the modified-CLM objective, the four-token model achieves higher win rates than the three-token counterpart for speed-ups < 2.0. DynaMo-80M-T4 achieves a same-quality speed-up of 3.89 , however, at an additional parameter overhead. Apart from the parameter overhead, the quadruplet co-occurrence mask incurs additional memory overhead. While the pairwise and triplet masks (calculated over 5% of the Pile dataset) only occupy 53.43MB and 152.59MB, respectively, the quadruplet mask (calculated over 0.05% of the Pile dataset) occupies 3.33GB memory. We store all co-occurrence masks using the sparse coordinate format. This overhead may still be negligible for very large models (>7B parameters). We leave simultaneous prediction of more than four tokens and optimized implementation of corresponding co-occurrence masks to future work.

![](images/31410c4a438216304695e5543997e4f44e206e25f582f178b5812b59738bf23e.jpg)  
Figure 6: Win rate vs. speed-up for pairwise comparisons on the sentence-completion benchmark with Pythia-70M as the baseline. GPT-3.5 is used as a judge. Theoretical same-quality speed-up is marked with an asterisk (\*).

## 5.5 Additional Benchmarking

We show the performance of the DynaMo models on most downstream benchmarking tasks. These results show that the better transformer trained using loss terms for predicting subsequent tokens generally leads to improved downstream performance while incurring no significant adverse effect on the model’s bias and misinformation abilities (see Appendix C.3.4). While Mukherjee et al. (2023) suggest evaluating world knowledge acquisition through tasks like AGIEval (Zhong et al., 2023) and Big-Bench Hard (Suzgun et al., 2023), we defer assessing larger multi-token models on such complex benchmarks to future work.

## 6 Conclusion

In this work, we presented DynaMo, a suite of multi-token prediction language models. We trained the proposed model suite efficiently by reusing weights of existing pre-trained LLMs. We proposed novel ways to dynamically predict multiple tokens for a given context. The DynaMo models dynamically back off to lower-order n-gram prediction based on a threshold. We also proposed adaptive thresholding and co-occurrence weighted masking on the modeled joint probability distribution to improve text generation quality. One of our proposed models, DynaMo-7.3B-T3, achieved the same-quality generated text as the baseline (Pythia-6.9B) while achieving 2.57 speed-up with only 5.87% and 2.67% parameter and training time overheads (see Appendix A.2).

## 7 Limitations

We trained DynaMo models on only 5% of the Pile dataset (Gao et al., 2020). However, training the models on the entire dataset would further boost performance due to improved estimates of the joint probability distributions. Future multi-token models can directly be trained on the entire language corpus without the complex multi-learning-rate learning employed here (details in Appendix A.1). Finally, the current suite of DynaMo models was trained with the Pythia backbone. One could also leverage state-of-the-art open-source foundation models (Touvron et al., 2023b) to train the DynaMo suite.

## Acknowledgments

This work was supported by Samsung Research America, Mountain View. N. K. J. was supported by NSF under Grant No. CCF-2203399.

## References

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. 2023. Pythia: A suite for analyzing large language models across training and scaling. In Proceedings of the International Conference on Machine Learning, pages 2397–2430.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. 2020. PIQA: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 7432–7439.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33:1877–1901.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, and Tri Dao. 2023. Medusa: Simple framework for accelerating LLM generation with multiple decoding heads. https://github.com/FasterDecoding/ Medusa.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023a. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318.

Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, et al. 2023b. Al-

paGasus: Training a better Alpaca with fewer data. arXiv preprint arXiv:2307.08701.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An open-source chatbot impressing GPT-4 with 90%\* ChatGPT quality. https://lmsys.org/ blog/2023-03-30-vicuna/.

Krishna Teja Chitty-Venkata, Murali Emani, Venkatram Vishwanath, and Arun K. Somani. 2022. Neural architecture search for transformers: A survey. IEEE Access, 10:108374–108412.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, volume 1, pages 2924–2936.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? Try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. Advances in Neural Information Processing Systems, 35:16344–16359.

Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Aleš Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. 2021. A continual learning survey: Defying forgetting in classification tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(7):3366–3385.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting ofthe Associationfor Computational Linguistics, volume 1, pages 889–898.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. 2020. The Pile: An 800GB dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027.

Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2021. A framework for few-shot language model evaluation. https://doi.org/10.5281/zenodo.5371628.

Xinyang Geng and Hao Liu. 2023. OpenLLaMA: An open reproduction of LLaMA. https://github. com/openlm-research/open\_llama.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. In Proceedings ofthe International Conference on Learning Representations.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

Edward J. Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021. LoRA: Low-rank adaptation of large language models. In Proceedings of the International Conference on Learning Representations.

Sebastian Jaszczur, Aakanksha Chowdhery, Afroz Mohiuddin, Lukasz Kaiser, Wojciech Gajewski, Henryk Michalewski, and Jonni Kanerva. 2021. Sparse is enough in scaling transformers. Advances in Neural Information Processing Systems, 34:9895–9907.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong, and Richard Socher. 2019. CTRL: A conditional transformer language model for controllable generation. arXiv preprint arXiv:1909.05858.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. 2017. RACE: Large-scale reading comprehension dataset from examinations. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 785–794.

Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, and Daniel Haziza. 2022. xFormers: A modular and hackable transformer modelling library. https: //github.com/facebookresearch/xformers.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics, volume 1, pages 3214–3252.

Xiaoxuan Liu, Lanxiang Hu, Peter Bailis, Ion Stoica, Zhijie Deng, Alvin Cheung, and Hao Zhang. 2023. Online speculative decoding. arXiv preprint arXiv:2310.07177.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? A new dataset for open book question answering. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 2381–2391.

Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. 2023. Orca: Progressive learning from complex explanation traces of GPT-4. arXiv preprint arXiv:2306.02707.

Nikita Nangia, Clara Vania, Rasika Bhalerao, and Samuel R. Bowman. 2020. CrowS-pairs: A challenge dataset for measuring social biases in masked language models. In Proceedings ofthe Conference on Empirical Methods in Natural Language Processing, pages 1953–1967.

Xuefei Ning, Zinan Lin, Zixuan Zhou, Huazhong Yang, and Yu Wang. 2023. Skeleton-of-Thought: Large language models can do parallel decoding. arXiv preprint arXiv:2307.15337.

OpenAI. 2023a. ChatGPT. https://chat.openai. com.

OpenAI. 2023b. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774.

Nobuyuki Otsu. 1979. A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1):62–66.

Gabriel Peyré, Marco Cuturi, et al. 2019. Computational optimal transport: With applications to data science. Foundations and Trends in Machine Learning, 11(5-6):355–607.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI Blog, 1(8):9.

Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018. Know what you don’t know: Unanswerable questions for SQuAD. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, volume 2, pages 784–789.

Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S. Gordon. 2011. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In Proceedings of the AAAI Spring Symposium Series.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. WinoGrande: An adversarial Winograd schema challenge at scale. Communications ofthe ACM, 64(9):99–106.

Thibault Séjourné, Jean Feydy, François-Xavier Vialard, Alain Trouvé, and Gabriel Peyré. 2019. Sinkhorn divergences for unbalanced optimal transport. arXiv preprint arXiv:1910.12958.

Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami, Michael W. Mahoney, and Kurt Keutzer. 2020. Q-BERT: Hessian based ultra low precision quantization of BERT. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8815–8821.

Benjamin Frederick Spector and Christopher Re. 2023. Accelerating LLM inference with staged speculative decoding. In Workshop on Efficient Systemsfor Foundation Models@ ICML2023.

Shikaripur N. Sridhar. 2012. Cognition and Sentence Production: A Cross-linguistic Study, volume 22. Springer Science & Business Media.

Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. 2018. Blockwise parallel decoding for deep autoregressive models. Advances in Neural Information Processing Systems, 31.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. 2023. Challenging BIG-Bench tasks and whether chain-of-thought can solve them. In Proceedings ofthe Associationfor Computational Linguistics, pages 13003–13051.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford Alpaca: An instruction-following LLaMA model. https:// github.com/tatsu-lab/stanford\_alpaca.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Shikhar Tuli and Niraj K. Jha. 2023a. AccelTran: A sparsity-aware accelerator for dynamic inference with transformers. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 42(11):4038–4051.

Shikhar Tuli and Niraj K. Jha. 2023b. EdgeTran: Device-aware co-search of transformers for efficient inference on mobile edge platforms. IEEE Transactions on Mobile Computing, pages 1–18.

Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan, and Song Han. 2020a. HAT: Hardware-aware transformers for efficient natural language processing. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 7675–7688.

Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. 2020b. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768.

Brian Yan, Siddharth Dalmia, Yosuke Higuchi, Graham Neubig, Florian Metze, Alan W. Black, and Shinji Watanabe. 2023. CTC alignments improve autoregressive translation. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 1615–1631.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a machine really finish your sentence? In Proceedings ofthe 57th Annual Meeting ofthe Association for Computational Linguistics, pages 4791–4800.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging LLM-as-a-judge with MT-bench and chatbot arena. arXiv preprint arXiv:2306.05685.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. 2023. AGIEval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364.

## A Experimental Setup Details

In this section, we provide details on the training and evaluation processes along with other hyperparameters. We then describe the sentencecompletion benchmark. Finally, we present the overheads in training time for our DynaMo suite of models.

## A.1 Training and Evaluation Processes

To train the DynaMo suite of models, we first transfer the weights from the base Pythia model. Then, we train the models on a randomly sampled 5% set of sentences in the Pile dataset<sup>1</sup>. We train for one epoch on this dataset. We choose a subset of the same dataset on which the base Pythia model was trained to avoid catastrophic forgetting when being trained on a different dataset. In the future, we plan to train the models on other datasets using standard continual learning approaches (De Lange et al., 2021).

We now describe the training procedure for the DynaMo suite of models. First, we transfer the weights for the base model (i.e., the model stem and the final decoder layer). Then, we train the base model with a low learning rate $\bf ( L R _ { B } )$ . On the other hand, we train subsequent token heads using a higher learning rate $\mathrm { ( L R _ { M } ) }$ since we randomly initialize their weights. However, when backpropagating those gradients to the model stem, we use a much lower learning rate $\bf ( L R _ { M B } )$ . We hypothesize that when the decoder layers learn from the first and subsequent token predictions, they make the transformer better in predicting multiple tokens. Table 5 shows the learning rates used for different models in the DynaMo suite. Fig. 7 shows the gradient flow when training an example three-token DynaMo model.

We train our models using the AdamW optimizer (Loshchilov and Hutter, 2017) with the following hyperparameters: $\beta _ { 1 } ~ = ~ 0 . 9 , \beta _ { 2 } ~ =$ $0 . 9 5 , \epsilon = 1 \times 1 0 ^ { - 8 }$ . We use the cosine learning rate scheduler such that the learning rate warms up for 1% of the dataset (758 steps) and then drops to 0 at the end of training. We use a batch size of 64 sentences, i.e., 131,072 tokens (each sentence is 2,048 tokens long). The dataset has 5M sentences, which we divide into a training set (97%) and validation set (3%). Thus, a batch size of 64 results in 75,782 training steps in one training epoch. We evaluate the model at every 5,000 steps. Fig. 8 shows the three-token validation loss (logarithm of $\mathrm { P P L _ { 1 : 3 } ) }$ for models in the DynaMo suite.

Table 5: Learning rates used for training different models in the DynaMo suite.
<table><tr><td>Model</td><td>LRB</td><td> $\mathbf { L R _ { M } }$ </td><td> $\mathbf { L R _ { M B } }$ </td></tr><tr><td>DynaMo-77M-T3</td><td> $1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 6 }$ </td></tr><tr><td>DynaMo-180M-T3</td><td> $6 \times 1 0 ^ { - 6 }$ </td><td> $6 \times 1 0 ^ { - 4 }$ </td><td> $6 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>DynaMo-430M-T3</td><td> $3 \times 1 0 ^ { - 6 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>DynaMo-1.1B-T3</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>DynaMo-1.5B-T3</td><td> $2 \times 1 0 ^ { - 6 }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>DynaMo-2.9B-T3</td><td> $1 . 6 \times 1 0 ^ { - 6 }$ </td><td> $1 . 6 \times 1 0 ^ { - 4 }$ </td><td> $1 . 6 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>DynaMo-7.3B-T3</td><td> $1 . 2 \times 1 0 ^ { - 6 }$ </td><td> $1 . 2 \times 1 0 ^ { - }$  4</td><td> $1 . 2 \times 1 0 ^ { - 7 }$ </td></tr></table>

![](images/2fef8a0d0804d0f15dfce667b3f44cd552f9df91a883042f22847d8e3533916f.jpg)  
Figure 7: Gradient flow when training a DynaMo model.

![](images/96efaed4b337e6cbdd100be0a234106cb9393a0b109af68a3040d73f01d649d4.jpg)  
Figure 8: Loss curves for three-token models in the DynaMo suite.

We train the models on A100 GPUs with 80GB memory. For efficient implementation of our models, we use the flash-attention library (Dao et al., 2022). Our models also support memory-efficient attention in the xformers library (Lefaudeux et al., 2022). Since DynaMo-7.3B-T3 did not fit in memory, we resorted to Py-Torch’s fully-sharded data parallel (FSDP) training feature. Table 6 provides the hyperparameters used for the FSDP configuration.

For text generation, we use $k = 5 0$ for top-k decoding, temperature = 0.7, and repetition penalty

= 1.1. The default text generation hyperparameters for the DynaMo models are $\alpha _ { c } = 1 . 0$ (see Appendix C.1), adaptive thresholding with Gaussian blur (kernel size = 3), and using co-occurrence weighted masking unless otherwise specified.

Table 6: FSDP configuration used for training DynaMo-7.3B-T3.
<table><tr><td>Configuration Key</td><td>Value</td></tr><tr><td>Sharding strategy</td><td>SHARD_GRAD_OP</td></tr><tr><td>Transformer-based wrap</td><td>DYNAMO_LAYER</td></tr><tr><td>All-gather backward prefetch policy</td><td>BACKWARD_PRE</td></tr><tr><td>All-gather forward prefetch policy</td><td>NONE</td></tr><tr><td>Mixed precision</td><td>FP16</td></tr></table>

Table 7: Training (with overheads) and instruction-finetuning times for the DynaMo suite of models.
<table><tr><td>Model</td><td>Training GPU Hrs.</td><td>Instruction-FT GPU Mins.</td></tr><tr><td>Pythia-70M</td><td>510</td><td>一</td></tr><tr><td>DynaMo-77M-T3</td><td>15 (2.94%)</td><td>8</td></tr><tr><td>Pythia-160M</td><td>1,030</td><td>-</td></tr><tr><td>DynaMo-180M-T3</td><td>36 (3.49%)</td><td>15</td></tr><tr><td>Pythia-410M</td><td>2,540</td><td>-</td></tr><tr><td>DynaMo-430M-T3</td><td>46 (1.81%)</td><td>30</td></tr><tr><td>Pythia-1B</td><td>4,830</td><td>-</td></tr><tr><td>DynaMo-1.1B-T3</td><td>80 (1.65%)</td><td>60</td></tr><tr><td>Pythia-1.4B</td><td>7,120</td><td>-</td></tr><tr><td>DynaMo-1.5B-T3</td><td>88 (1.24%)</td><td>72</td></tr><tr><td>Pythia-2.8B</td><td>14,240</td><td>一</td></tr><tr><td>DynaMo-2.9B-T3</td><td>176 (1.24%)</td><td>180</td></tr><tr><td>Pythia-6.9B</td><td>33,500</td><td>一</td></tr><tr><td>DynaMo-7.3B-T3</td><td>896 (2.67%)</td><td>864</td></tr></table>

## A.2 Training Overheads

Table 7 shows the overhead of training models in the DynaMo suite. We report training times for modified-CLM training on 5% of the Pile dataset and instruction-finetuning. We present the reported CLM training times for the Pythia models (Biderman et al., 2023).

## A.3 Sentence-completion Benchmark

In this section, we provide details of the sentencecompletion benchmark. This benchmark is motivated by the Vicuna benchmark (Zheng et al., 2023). However, it is meant for pre-trained LLMs that are not instruction-finetuned. This dissociates any effects of instruction-finetuning from model performance. The benchmark consists of ten prompts requiring the model to complete the sentence. These prompts correspond to sentences of different types. Table 8 outlines the prompts.

To obtain the GPT score, we ask GPT-3.5 to rate the generated sentence on a scale from 1 to 10. For pairwise evaluations, we ask GPT-3.5 to compare the generated text (by our DynaMo

Table 8: Prompts in the sentence-completion benchmark.
<table><tr><td>Prompt</td><td>Type</td></tr><tr><td>I am a student at the This is going to be a very</td><td>Simple Declarative Simple Declarative</td></tr><tr><td>He wanted to play, but</td><td>Compound Declarative</td></tr><tr><td>How can we</td><td>W/H Interrogative</td></tr><tr><td>What will</td><td>W/H Interrogative</td></tr><tr><td>Will you</td><td>Y/N Interrogative</td></tr><tr><td>Please explain</td><td>Affirmative Imperative</td></tr><tr><td>Do not</td><td></td></tr><tr><td></td><td>Negative Imperative</td></tr><tr><td>Wow! I can&#x27;t believe that</td><td>Exclamatory</td></tr><tr><td>This is amazing! We</td><td>Exclamatory</td></tr></table>

Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the input prompt. The AI assistant provides an open-ended generation for the input prompt. Your evaluation should be based on the grammatical correctness, creativity, depth, logical flow, coherence, and based on how informative the response is. Do not let the length of the generated text influence your evaluation. Be as objective as possible. Begin your evaluation by providing a short explanation. Explain the mistakes, if any. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]"

Figure 9: Prompt template to rate the sentence quality of the candidate assistant model on an absolute scale (single-mode evaluation).

model) against a baseline (the corresponding baseline Pythia model) and rate it as a “win,” “lose,” or a “tie.” We use gpt-3.5-turbo-0613 for our evaluations. Fig. 9 shows the prompt template used for single-mode evaluations and Fig. 10 shows the prompt template used for pairwise evaluations. However, this benchmark also suffers from the same drawbacks as the Vicuna benchmark (Zheng et al., 2023), which we attempt to alleviate. To address position bias in pairwise comparisons, we randomly order the responses of the assistants.

## B Optimal Transport Theory

Eq. (2) approximates the output joint probability by directly multiplying the independent marginal distributions. This implicitly assumes that x<sub>t+2</sub> is independent of $\mathbf { x } _ { t + 1 }$ conditioned on history $\mathbf { x } _ { 1 : t } ,$

Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the input prompt. Both AI assistants provide open-ended generations for the input prompt. You should choose the assistant that produces a better generation. Your evaluation should be based on the grammatical correctness, creativity, depth, logical flow, coherence, and based on how informative the responses are. Do not let the lengths of the generated texts influence your evaluation. Do not favor certain names of the assistants. Begin your evaluation by comparing the two responses and provide a short explanation. Explain the mistakes, if any. Avoid any positional biases and ensure that the order in which the responses were presented does not influence your decision. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: ${ } ^ { \prime \prime } \left[ \left[ \mathsf { A } \right] \right] ^ { \prime \prime } \mathsf { i f }$ assistant A is better, ${ } ^ { \prime \prime } \left[ \left[ \mathsf { B } \right] \right] ^ { \prime \prime } \mathrm { ~ i f ~ }$ assistant B is better, and "[[C]]" for a tie.

Figure 10: Prompt template to rate the sentence quality of the candidate assistant model against a baseline model (pairwisemode evaluation).

$\mathbf { x } _ { t + 3 }$ is independent of $\mathbf { x } _ { t + 1 }$ and $\mathbf { x } _ { t + 2 }$ , and so on. The downside of this decoding strategy is that it ignores the fact that the prediction of $\mathbf { x } _ { t + 2 }$ depends heavily on which $\mathbf { x } _ { t + 1 }$ is chosen (and similarly for subsequent predictions). A simple example is to consider $\mathbf { x } _ { 1 : t } = \mathbb { I }$ ; here, to is a plausible secondword prediction as many sentences lead to that word, such as I like to, I want to, and I went to. On the other hand, am is a plausible first-word prediction. However, as long as one chooses it, the weight for to as the second-word prediction should be minimal unless we want to make our English teacher cry. This motivates us to weight the joint probability distribution based on co-occurrence of words (or, more precisely, tokens).

What follows is a theoretical motivation behind the use of co-occurrence weighted masking. Formally, according to optimal transport theory (Peyré et al., 2019), we define a cost function $c ( \mathbf { x } _ { t + 1 } , \ldots , \mathbf { x } _ { t + n } ) , \forall \ \mathbf { x } _ { t + 1 } , \ldots , \mathbf { x } _ { t + n } .$ . Once we define the cost function, we pose the joint estima-

tion problem as follows,

$$
\begin{array} { l } { { \displaystyle \arg \operatorname* { m i n } _ { p } \sum _ { j } p ( { \bf x } _ { t + 1 : t + n } | { \bf x } _ { 1 : t } ) c ( { \bf x } _ { t + 1 } , \ldots , { \bf x } _ { t + n } ) } } \\ { { \displaystyle \qquad \Delta { \bf x } _ { t + 1 } \ldots \Delta { \bf x } _ { t + n } } } \\ { { \displaystyle \qquad + \epsilon _ { 1 } \mathrm { K L } \left( p ( { \bf x } _ { t + 1 : t + n } | { \bf x } _ { 1 : t } ) | | \prod _ { i = 1 } ^ { n } f _ { \theta } ^ { i } ( { \bf x } _ { 1 : t } ) \right) } } \\ { { \displaystyle \qquad + \epsilon _ { 2 } \sum _ { i = 1 } ^ { n } \mathrm { K L } \left( p ( { \bf x } _ { t + i } | { \bf x } _ { 1 : t } ) | | f _ { \theta } ^ { i } ( { \bf x } _ { 1 : t } ) \right) } } \end{array}\tag{5}
$$

Although solving an optimal transport problem is fast, using the celebrated Sinkhorn algorithm (Séjourné et al., 2019), we propose the use of Eq. (3) as an approximation that works well in practice, as we demonstrate in our experimental results. Next, we show that the approximation in Eq. (3) is indeed the closest to preserving the true joint probability distribution, when the correction term (co-occurrence mask) is not dependent on the history $\mathbf { x } _ { 1 : t }$

ProofofTheorem 1. Recall that the optimization in Eq. (5) is subject to the constraint $\begin{array} { r } { \int p ( \mathbf { x } _ { t + 1 : t + n } | \mathbf { x } _ { 1 : t } ) \Delta \mathbf { x } _ { t + 1 } \dots \Delta \mathbf { x } _ { t + n } = 1 } \end{array}$ . Thus, the Lagrangian of the objective is given by

$$
\begin{array} { l } { { \displaystyle { \cal L } = \sum _ { j = 1 } ^ { n } p ( { \bf x } _ { t + 1 : t + 1 } | { \bf x } _ { 1 : t } ) e ( { \bf x } _ { t + 1 } , \ldots , { \bf x } _ { t + n } ) } } \\ { { \displaystyle \quad \Delta { \bf x } _ { t + 1 } . . . \Delta { \bf x } _ { t + n } } } \\ { { \displaystyle \quad + \epsilon _ { 1 } { \bf K } \mathbf { L } \left( p ( { \bf x } _ { t + 1 : t + 1 } | { \bf x } _ { 1 : t } ) | | \prod _ { j = 1 } ^ { n } f _ { \theta } ^ { i } ( { \bf x } _ { 1 : t } ) \right) } } \\ { { \displaystyle \quad + \epsilon _ { 2 } \sum _ { i = 1 } ^ { n } \mathbf { K } \left( p ( { \bf x } _ { t + i } | { \bf x } _ { 1 : t } ) | | f _ { \theta } ^ { i } ( { \bf x } _ { 1 : t } ) \right) } } \\ { { \displaystyle \quad + \lambda \Big ( \sum _ { j = 1 } ^ { n } p ( { \bf x } _ { t + 1 : t + n } | { \bf x } _ { 1 : t } ) \Delta { \bf x } _ { t + 1 } . . . \Delta { \bf x } _ { t + n } } } \\ { { \displaystyle \quad - 1 \Big ) } } \end{array}
$$

Setting the derivative of L w.r.t. $p ( \mathbf { x } _ { t + 1 : t + n } | \mathbf { x } _ { 1 : t } )$ to zero, we get

$$
\begin{array} { l } { { \displaystyle p ^ { * } \big ( { \bf x } _ { t + 1 : t + n } \big | { \bf x } _ { 1 : t } \big ) } \ ~ } \\ { { \displaystyle \propto \prod _ { i = 1 } ^ { n } f _ { \theta } ^ { i } \big ( { \bf x } _ { 1 : t } \big ) \ \exp \big ( c \big ( { \bf x } _ { t + 1 } , \ldots , { \bf x } _ { t + n } \big ) / \epsilon _ { 1 } \big ) } \ ~ } \\ { { \displaystyle = \prod _ { i = 1 } ^ { n } f _ { \theta } ^ { i } \big ( { \bf x } _ { 1 : t } \big ) \frac { \hat { p } \big ( { \bf x } _ { t + 1 : t + n } \big ) } { \prod _ { i = 1 } ^ { n } \hat { p } \big ( { \bf x } _ { t + i } \big ) } } } \end{array}
$$

![](images/a2048faebaff2ffd0f47946a213a1cd3da58d2107a98fcd1b1f04c504d342d4b.jpg)  
(a)

![](images/c592252a8cc3836336c3623f553123958670d2f3f23152a04e72ab9779b4fc68.jpg)  
(b)

![](images/c9d2e6d4c42d63e2add5a0ffc8e798b0cfa4fa906bd6f3bdcacd240766f227a8.jpg)

![](images/2485a2913c9f6d607e0f2ff787c507a97b45cde3b6ece31ea4075484e3cc3cdd.jpg)  
(d)

![](images/75b9d06a448bb54969edf35fa100ff66a8f2d708c08630d07f7f75938935048b.jpg)  
(e)

![](images/694bb70016f9b0891b1d394ce74fe2650f67f76479d0b8fe0818f88ccaaf615f.jpg)  
(f)  
Figure 11: Joint probability distribution with top 10 tokens sorted in decreasing order of probabilities using the DynaMo-2.9B-T2 model for the input prompt: Please explain. Probabilities corresponding to repetition have been penalized by a factor of 100. (a) and (d) are vanilla distributions. Co-occurrence masked distribution with (b) $\alpha _ { c } = 0 . 5 [ \mathrm { C O } { - } 0 . 5 ]$ and (c) $\alpha _ { c } = 1 . 0 [ \mathrm { C O } ]$ Adaptive thresholding (e) without Gaussian blur [AT], and (f) with Gaussian blur (kernel size = 3) [AT + G-3].

## C Additional Results

In this section, we report additional supporting results.

## C.1 Ablation of Dynamic Text Generation Methods

In this section, we ablate the effect of adaptive thresholding (with and without Gaussian blur) and co-occurrence weighted masking (see Section 3.2). Figs. 11(a)-(c) show the effect of co-occurrence masking on the two-token joint probability with decreasing masking transparency $\alpha _ { c }$ . Mathematically, we modify Eq. (3) for the two-token prediction case as follows:

$$
\begin{array} { r l } & { p ( \mathbf { x } _ { t + 1 } , \mathbf { x } _ { t + 2 } | \mathbf { x } _ { 1 : t } ) } \\ & { \qquad \approx f _ { \theta } ^ { 1 } ( \mathbf { x } _ { 1 : t } ) f _ { \theta } ^ { 2 } ( \mathbf { x } _ { 1 : t } ) \left( \frac { \hat { p } ( \mathbf { x } _ { t + 1 } , \mathbf { x } _ { t + 2 } ) } { \hat { p } ( \mathbf { x } _ { t + 1 } ) \hat { p } ( \mathbf { x } _ { t + 2 } ) } \right) ^ { \alpha _ { c } } } \end{array}\tag{6}
$$

where $\alpha _ { c } ~ = ~ 1 . 0$ implies that the co-occurrence weights mask the joint probability distribution with no transparency. On the other hand, we do not use co-occurrence masking when $\alpha _ { c } = 0 . 0$ . Nevertheless, $\alpha _ { c } = 0 . 5$ partially masks the joint probability distribution using the co-occurrence weights.

![](images/2466facb08fb10ce00bdaa85d45945d94174ca275093c1e050c22ee418cfd379.jpg)  
Figure 12: Ablation analysis using adaptive thresholding (with and without Gaussian blur) and co-occurrence masking. Win rates for pairwise tests against Pythia-70M on the sentence-completion benchmark are shown for different speedups. GPT-3.5 is used as the judge. Theoretical same-quality speed-ups are marked with an asterisk (\*).

Figs. 11(d)-(f) show the effect of adaptive thresholding with and without Gaussian blur.

Fig. 12 shows the win rates vs. speed-up for DynaMo-77M-T3, where we generated the texts in the sentence-completion benchmark using difference schemes. We observe that co-occurrence masking (with $\alpha _ { c } = 1 . 0 , 1 . { \mathrm { e } } .$ , the default setting used in our experiments) used along with adaptive thresholding (after application of Gaussian blur with a kernel size = 3) results in the flattest win rate vs. speed-up curve, thus, providing the highest theoretical same-quality speed-up.

![](images/7b7538dc0a294c41c561b76addb346b0d8ce34adf0fec37d74f6e39024eebe4c.jpg)

![](images/91929ace4b8fe8f3464f8254c7e612bca72da64c65f444309225c792ab5261a2.jpg)  
Figure 13: Multi-token prediction using a single-token head. The input sequence is shown below the transformer layer. The model predicts the output sequence above. Attention arrows correspond to the modified CLM objective. The attention masks are shown below the input sequences. (a) T1-L2-M0: labels are shifted by two positions (i.e., the model predicts $\mathbf { x } _ { t + 2 } ^ { \prime }$ with x as input). Under the modified CLM objective, the model learns to predict $\mathbf { x } _ { t + 2 } ^ { \prime } = \mathbf { x } _ { t + 2 }$ . (b) T1-L2-M(-1)R: labels are shifted by two positions but masks are shifted in the opposite direction (i.e., for predicting $\mathbf { x } _ { t + 2 } ^ { \prime }$ , the model can sometimes see $\mathbf { x } _ { t + 1 } )$

Table 9: Ablations analysis of dynamic text generation methods with the instruction-finetuned DynaMo-7.3B-T3 model on the Vicuna benchmark. We use $\epsilon _ { b } = 0 . 5$
<table><tr><td>Method</td><td>Speed-up</td><td>Win rate</td></tr><tr><td> $\mathrm { C O } + \mathrm { A T } + \mathrm { G } { - } 3$ </td><td>2.57×</td><td>0.98</td></tr><tr><td> $\mathrm { C O } + \mathrm { A T }$ </td><td>2.44×</td><td>0.96</td></tr><tr><td>CO</td><td>2.61×</td><td>0.82</td></tr><tr><td> $\mathbf { C O - } 0 . 5 + \mathbf { A T } + \mathbf { G } 3$ </td><td>2.55×</td><td>0.77</td></tr><tr><td> $\mathrm { A T } + \mathrm { G } { - } 3$ </td><td>2.49×</td><td>0.38</td></tr></table>

We ablate the effect of dynamic text generation methods with the instruction-finetuned DynaMo-7.3B-T3 model on the Vicuna benchmark in Table 9. We take the case $\epsilon _ { b } ~ = ~ 0 . 5$ (that results in 2.57 speed-up in Fig. 4) and present the win rates against Pythia-6.9B. Leveraging cooccurrence weighted masking along with adaptive thresholding using Gaussian blur (kernel size = 3) results in the highest win rate.

## C.2 Exploration of Multi-token Prediction Methods

In this section, we provide a detailed overview of various architectural and training variations tested for multi-token prediction.

## C.2.1 Design Variations

Under the CLM objective, the attention mask prevents the model from seeing future tokens, i.e., we only compute the attentions corresponding to the lower triangular matrix (we refer to this case as M0). In summary, we represent traditional autoregressive models as T1-L1-M0. We study different variations of the above formulation for multi-token prediction. These include multiple token heads, label shifts, and mask shifts. We explore them below. After testing various approaches, we observe that for, say, three-token prediction, the T3-L1-M0 set of choices performs the best. Thus, in all discussions in the main paper, we represent DynaMo-T3- L1-M0 as simply DynaMo-T3.

Fig. 13 shows the information flow for T1-L2- M0 and T1-L2-M(-1)R cases. In the former case, for predicting $\mathbf { x } _ { t + 2 }$ , the model only sees the input context $\mathbf { x } _ { 1 : t }$ . Hence, we shift the mask in the latter case. However, T1-L2-M(-1) would be equivalent to the traditional T1-L1-M0 (ignoring residual connections that result in information leakage). Hence, we randomly mask out some tokens so that the model learns to predict the next and the second-next token at each position. Another position-equivalent modeling approach to T1-L2-M(-1)R is T1-L1- M1R. However, both these modeling approaches suffer from information leakage. T1-L2-M(-1)R suffers from information leakage due to expanding receptive fields along model depth. We fix this by incorporating negative mask shifts only in the first layer of the LLM. T1-L1-M1R suffers from information leakage due to the residual/skip connections in the LLM. Hence, we do not use this approach and test T1-L2-M(-1)R instead.

![](images/e8c10cdd25206cdcd662bb5d6f9db480eb8f0c92b55366dd6f8e7e9c44304eff.jpg)  
Figure 14: Architectural variations of the two-token prediction model that we tested: (a) DynaMo-96M-T2, (b) DynaMo-74M-T2 (C), (c) DynaMo-70M-T2 (LoRA), (d) DynaMo-99M-T2, (e) DynaMo-74M-T2 (NP), and (f) DynaMo-77M-T2.

Fig. 14 shows different architectural variations of the two-token model we tested. We initialize all these models from the base Pythia-70M model. Fig. 14(a) shows the schematic of DynaMo-96M-T2 that randomly initializes the output embedding for the second-token head (we denote newly initialized weights by while other variations reuse these weights). The output embedding has 26M trainable parameters. Fig. 14(b) shows DynaMo-74M-T2 (C), which copies the weights of the decoder layer for the second-token head from the last layer of the first-token head (or the base model). Its output embedding for the second-token head reuses the weights from the first-token head. Since we copy the weights, we train the copied weights with a low learning rate (LR<sub>B</sub>). Fig. 14(c) shows DynaMo-70M-T2 (LoRA) with only 65K trainable parameters (Hu et al., 2021). The LoRA module includes a low-rank matrix (we use rank = 32). We add its output to that of the last decoder layer for secondtoken prediction. Fig. 14(d) shows DynaMo-99M-T2. We train a decoder layer and the output embedding for the second-token head, where we randomly initialize the weights of both modules. Fig. 14(e) shows DynaMo-74M-T2 (NP), where we feed the output of the last layer of the base model to the decoder layer for the second-token head. All models in the DynaMo suite use the outputs of the penultimate layer of the base model for subsequent token prediction. Instead, this model uses the output of the final (non-penultimate or NP) layer. Finally, Fig. 14(f) shows the use of two decoder layers for the second-token head.

Table 10: Multi-token perplexity results for various architectural variations. <sup>+</sup>Model was further trained on 5% Pile dataset.
<table><tr><td>Model</td><td> $\mathbf { P P L _ { 1 } }$ </td><td> $\mathbf { P P L _ { 2 } }$ </td><td> $\mathbf { P P L _ { 3 } }$ </td><td> $\bf P P L _ { 1 2 }$ </td><td> $\bf P P L _ { 1 2 3 }$ </td></tr><tr><td>Pythia-70M</td><td>20.2±1.5</td><td></td><td></td><td></td><td></td></tr><tr><td>Pythia-70M+</td><td> $2 0 . 1 _ { \pm 1 . 5 }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>DynaMo-70M-T1-L2</td><td> $2 1 . 4 _ { \pm 1 . 6 }$ </td><td> $1 4 5 5 . 8 _ { \pm 6 . 4 }$ </td><td></td><td> $1 8 9 . 3 _ { \pm 2 . 2 }$ </td><td></td></tr><tr><td>DynaMo-70M-T1-L2-M(-1)R</td><td> $2 0 . 3 { \scriptstyle \pm 1 . 5 }$ </td><td> $6 4 5 . 3 _ { \pm 1 . 9 }$ </td><td></td><td> $8 7 . 4 _ { \pm 1 . 7 }$ </td><td></td></tr><tr><td>DynaMo-96M-T2</td><td> $1 9 . 9 _ { \pm 1 . 5 }$ </td><td> $2 5 2 . 4 _ { \pm 1 . 9 }$ </td><td></td><td> $6 8 . 0 { \scriptstyle \pm 1 . 5 }$ </td><td></td></tr><tr><td>DynaMo-74M-T2 (C)</td><td> $1 8 . 3 _ { \pm 1 . 5 }$ </td><td> $2 9 6 . 4 _ { \pm 1 . 5 }$ </td><td></td><td> $7 3 . 7 _ { \pm 1 . 5 }$ </td><td></td></tr><tr><td>DynaMo-70M-T2 (LoRA)</td><td> $2 0 . 2 { \scriptstyle \pm 1 . 5 }$ </td><td> $1 3 6 8 . 1 { \scriptstyle \pm 1 . 8 }$ </td><td></td><td> $1 6 1 . 2 { \scriptstyle \pm 1 . 6 }$ </td><td></td></tr><tr><td>DynaMo-74M-T2 (CTC)</td><td> $1 8 . 5 { \scriptstyle \pm 1 . 5 }$ </td><td> $1 1 5 . 4 { \scriptstyle \pm 1 . 7 }$ </td><td></td><td> $4 6 . 0 { \scriptstyle \pm 1 . 6 }$ </td><td></td></tr><tr><td>DynaMo-99M-T2</td><td> $1 8 . 3 { \scriptstyle \pm 1 . 5 }$ </td><td> $1 1 1 . 5 { \scriptstyle \pm 1 . 7 }$ </td><td></td><td> $4 5 . 2 { \scriptstyle \pm 1 . 5 }$ </td><td></td></tr><tr><td>DynaMo-74M-T2 (NP)</td><td> $1 8 . 8 { \scriptstyle \pm 1 . 5 }$ </td><td> $1 3 1 . 1 { \pm } 1 . 6 $ </td><td></td><td> $4 9 . 0 { \scriptstyle \pm 1 . 5 }$ </td><td></td></tr><tr><td>DynaMo-74M-T2-H</td><td> $2 0 . 2 _ { \pm 1 . 5 }$ </td><td> $1 1 9 . 1 _ { \pm 1 . 7 }$ </td><td></td><td> $4 9 . 0 _ { \pm 1 . 5 }$ </td><td></td></tr><tr><td>DynaMo-74M-T2</td><td> $1 8 . 3 { \scriptstyle \pm 1 . 5 }$ </td><td> $1 1 2 . 4 { \scriptstyle \pm 1 . 7 }$ </td><td></td><td> $4 5 . 4 _ { \pm 1 . 5 }$ </td><td></td></tr><tr><td>DynaMo-77M-T2</td><td> $1 8 . 3 { \scriptstyle \pm 1 . 5 }$ </td><td> $8 6 . 7 \pm 1 . 7$ </td><td></td><td> $3 9 . 9 { \scriptstyle \pm 1 . 6 }$ </td><td></td></tr><tr><td>DynaMo-77M-T3</td><td> $1 8 . 3 { \scriptstyle \pm 1 . 5 }$ </td><td> $1 1 1 . 4 { \scriptstyle \pm 1 . 7 }$ </td><td> $2 6 2 . 0 { \scriptstyle \pm 1 . 6 }$ </td><td> $4 5 . 2 { \scriptstyle \pm 1 . 5 }$ </td><td> $8 1 . 2 { \scriptstyle \pm 1 . 6 }$ </td></tr></table>

## C.2.2 Evaluations

Table 10 shows the multi-token perplexity results for various architectural and training variations of the DynaMo model with Pythia-70M as the baseline. For fair comparisons, we also add the perplexity results for Pythia-70M<sup>+</sup> (trained using $\mathrm { L R _ { B } } = 1 0 ^ { - 5 } )$ . It does not result in a lower $\mathrm { P P L _ { 1 } }$ This shows that with traditional CLM training, $\mathrm { P P L _ { 1 } }$ has converged. However, with the modified-CLM training (details in Appendix A.1), PPL<sub>1</sub> for models in the DynaMo suite goes down further. The architectural variations are as explained above. DynaMo-74M-T2 (CTC) shows the perplexity results for the model trained using CTC loss (Yan et al., 2023). DynaMo-74M-T2-H is the model where we only train the decoder layer of the secondtoken head. Training this model is much faster than training DynaMo-74M-T2, as we need to calculate only a few gradients. However, this does not make the decoder layers in the model stem better. We see that PPL<sub>1</sub> of this model is the same as that of Pythia-70M. One could increase the parameter budget for multi-token prediction by either adding another decoder layer for predicting the second token (DynaMo-77M-T2) or using a decoder layer for the third-token head (DynaMo-77M-T3). In the DynaMo suite of models, we traded the parameter budget for higher speed-up (using three-token models). We leave the exploration and search among various architectural decisions (Chitty-Venkata et al., 2022; Tuli and Jha, 2023b) targeting text generation performance and speed-up to future work.

Table 11: Five-shot exact match performance on the TriviaQA benchmark.  
Table 12: Zero-shot accuracy for the RACE benchmark along with exact match performance and F1 scores (in parenthesis) for the SquAD2.0 benchmark.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>TriviaQA</td></tr><tr><td rowspan=1 colspan=1>Pythia-70MDynaMo-77M-T3</td><td rowspan=1 colspan=1> ${ \bf 0 . 2 } _ { \pm 0 . 0 }$  ${ \bf 0 . 2 } _ { \pm 0 . 0 }$ </td></tr><tr><td rowspan=1 colspan=1>Pythia-160MDynaMo-180M-T3</td><td rowspan=1 colspan=1> $2 . 1 _ { \pm 0 . 1 }$  $2 . 2 _ { \pm 0 . 1 }$ </td></tr><tr><td rowspan=1 colspan=1>Pythia-410MDynaMo-430M-T3</td><td rowspan=1 colspan=1> $7 . 4 { \scriptstyle \pm 0 . 2 }$  ${ \bf 7 . 9 { \scriptstyle \pm 0 . 2 } }$ </td></tr><tr><td rowspan=1 colspan=1>Pythia-1BDynaMo-1.1B-T3</td><td rowspan=1 colspan=1> $1 2 . 0 { \scriptstyle \pm 0 . 2 }$  ${ \bf 1 4 . 2 _ { \pm 0 . 3 } }$ </td></tr><tr><td rowspan=1 colspan=1>Pythia-1.4BDynaMo-1.5B-T3</td><td rowspan=1 colspan=1> $6 . 2 \pm 0 . 2$  ${ \bf 1 8 . 9 2 0 . 3 }$ </td></tr><tr><td rowspan=1 colspan=1>Pythia-2.8BDynaMo-2.9B-T3</td><td rowspan=1 colspan=1> $7 . 1 _ { \pm 0 . 2 }$  $2 5 . 1 _ { \pm 0 . 3 }$ </td></tr><tr><td rowspan=1 colspan=1>Pythia-6.9BDynaMo-7.3B-T3</td><td rowspan=1 colspan=1> $8 . 9 { \scriptstyle \pm 0 . 2 }$  ${ 3 3 . 6 } _ { \pm 0 . 3 }$ </td></tr></table>

<table><tr><td>Model</td><td>RACE</td><td>SQuAD2.0</td></tr><tr><td>Pythia-70M DynaMo-77M-T3</td><td> $2 3 . 5 { \pm } 1 . 3 $   $\pm 4 . 4 _ { \pm 1 . 3 }$ </td><td> $1 . 2 ( 2 . 5 ) $  4.2 (5.6)</td></tr><tr><td>Pythia-160M DynaMo-180M-T3</td><td> $2 8 . 3 _ { \pm 1 . 4 }$   $2 7 . 9 _ { \pm 1 . 4 }$ </td><td>0.6 (3.5) 0.4 (3.0)</td></tr><tr><td>Pythia-410M DynaMo-430M-T3</td><td> $3 1 . 5 _ { \pm 1 . 4 }$ </td><td>2.0 (7.4) 2.0 (7.2)</td></tr><tr><td>Pythia-1B</td><td> ${ \bf 3 2 . 9 2 1 . 5 }$   $3 2 . 3 _ { \pm 1 . 4 }$ </td><td>4.2 (5.3)</td></tr><tr><td>DynaMo-1.1B-T3 Pythia-1.4B</td><td> $3 1 . 9 _ { \pm 1 . 4 }$  34.1±1.5</td><td>4.9 (11.5) 4.4 (5.8)</td></tr><tr><td>DynaMo-1.5B-T3 Pythia-2.8B</td><td> $3 4 . 0 { \scriptstyle \pm 1 . 5 }$   ${ \bf 3 4 . 9 2 1 . 5 }$ </td><td>6.6 (13.5) 5.2 (8.5)</td></tr><tr><td>DynaMo-2.9B-T3 Pythia-6.9B</td><td> $3 4 . 5 _ { \pm 1 . 5 }$   $3 7 . 1 _ { \pm 1 . 5 }$ </td><td>7.1 (15.0)  $8 . 0 \ : ( 9 . 5 )$ </td></tr></table>

Table 13: Five-shot accuracy on the MMLU benchmark.
<table><tr><td>Model</td><td>Humanities</td><td>Social Sciences</td><td>STEM</td><td>Other</td><td>Average</td></tr><tr><td>Pythia-70M</td><td> $2 4 . 1 { \pm } 3 . 0 $ </td><td> $2 6 . 0 { \scriptstyle \pm 3 . 2 }$ </td><td> $2 7 . 6 { \scriptstyle \pm 3 . 8 }$ </td><td> $2 3 . 9 { \scriptstyle \pm 3 . 2 }$ </td><td> $2 5 . 6 { \scriptstyle \pm 3 . 3 }$ </td></tr><tr><td>DynaMo-77M-T3</td><td> $2 3 . 6 _ { \pm 2 . 9 }$ </td><td> $2 7 . 4 _ { \pm 3 . 3 }$ </td><td> $2 6 . 6 _ { \pm 3 . 7 }$ </td><td> ${ 2 4 . 8 _ { \pm 3 . 2 } }$ </td><td> $2 5 . 7 _ { \pm 3 . 3 }$ </td></tr><tr><td>Pythia-160M</td><td> $2 4 . 2 _ { \pm 3 . 0 }$ </td><td> $2 6 . 0 { \scriptstyle \pm 3 . 2 }$ </td><td> $2 7 . 3 _ { \pm 3 . 7 }$ </td><td> $2 4 . 1 _ { \pm 3 . 2 }$ </td><td> $2 5 . 6 _ { \pm 3 . 3 }$ </td></tr><tr><td>DynaMo-180M-T3</td><td> $2 4 . 7 _ { \pm 3 . 0 }$ </td><td> $2 6 . 6 _ { \pm 3 . 2 }$ </td><td> $2 5 . 7 _ { \pm 3 . 6 }$ </td><td> $\mathbf { 2 4 . 9 _ { \pm 3 . 2 } }$ </td><td> $2 5 . 5 { \scriptstyle \pm 3 . 3 }$ </td></tr><tr><td>Pythia-410M</td><td> $2 5 . 6 _ { \pm 3 . 1 }$ </td><td> $2 5 . 0 _ { \pm 3 . 2 }$ </td><td> $2 6 . 9 _ { \pm 3 . 7 }$ </td><td> $2 6 . 5 { \scriptstyle \pm 3 . 4 }$ </td><td> ${ 2 6 . 1 } _ { \pm 3 . 4 }$ </td></tr><tr><td>DynaMo-430M-T3</td><td> $2 5 . 2 { \scriptstyle \pm 3 . 1 }$ </td><td> $2 3 . 5 { \scriptstyle \pm 3 . 1 }$ </td><td> $2 7 . 7 { \scriptstyle \pm 3 . 8 }$ </td><td> $2 7 . 2 { \scriptstyle \pm 3 . 4 }$ </td><td> ${ \bf 2 6 . 1 { \bf _ { \pm 3 . 4 } } }$ </td></tr><tr><td>Pythia-1B</td><td> $2 5 . 2 _ { \pm 3 . 0 }$ </td><td> $2 2 . 3 _ { \pm 3 . 0 }$ </td><td> $2 4 . 0 { \scriptstyle \pm 3 . 6 }$ </td><td> $2 5 . 7 _ { \pm 3 . 3 }$ </td><td> $2 4 . 3 _ { \pm 3 . 3 }$ </td></tr><tr><td>DynaMo-1.1B-T3</td><td> $2 4 . 6 { \scriptstyle \pm 3 . 0 }$ </td><td> $2 2 . 7 _ { \pm 3 . 1 }$ </td><td> $2 5 . 2 { \scriptstyle \pm 3 . 7 }$ </td><td> $2 6 . 2 { \scriptstyle \pm 3 . 3 }$ </td><td>24.8±3.3</td></tr><tr><td>Pythia-1.4B</td><td> $2 5 . 2 _ { \pm 3 . 0 }$ </td><td> $2 2 . 4 _ { \pm 3 . 1 }$ </td><td> $2 7 . 2 _ { \pm 3 . 8 }$ </td><td> $2 6 . 4 _ { \pm 3 . 4 }$ </td><td> $2 5 . 5 _ { \pm 3 . 4 }$ </td></tr><tr><td>DynaMo-1.5B-T3</td><td> $2 5 . 8 { \scriptstyle \pm 3 . 0 }$ </td><td> $2 2 . 2 { \scriptstyle \pm 3 . 1 }$ </td><td> $2 7 . 7 { \scriptstyle \pm 3 . 8 }$ </td><td> $2 4 . 7 { \pm } 3 . 3 $ </td><td> $2 5 . 4 { \scriptstyle \pm 3 . 4 }$ </td></tr><tr><td>Pythia-2.8B</td><td> $2 6 . 5 _ { \pm 3 . 1 }$ </td><td> $2 5 . 9 _ { \pm 3 . 2 }$ </td><td> $2 7 . 3 _ { \pm 3 . 8 }$ </td><td> $2 7 . 8 _ { \pm 3 . 4 }$ </td><td> ${ 2 7 . 0 } _ { \pm 3 . 4 }$ </td></tr><tr><td>DynaMo-2.9B-T3</td><td> $2 6 . 6 _ { \pm 3 . 1 }$ </td><td> $2 4 . 7 _ { \pm 3 . 2 }$ </td><td> $2 7 . 0 _ { \pm 3 . 7 }$ </td><td> ${ 2 8 . 2 } _ { \pm 3 . 4 }$ </td><td> $2 6 . 7 _ { \pm 3 . 4 }$ </td></tr><tr><td>Pythia-6.9B</td><td> $2 6 . 1 \pm 3 . 1$ </td><td> $2 4 . 8 { \scriptstyle \pm 3 . 2 }$ </td><td> $2 7 . 3 { \scriptstyle \pm 3 . 7 }$ </td><td> $\mathbf { 2 6 . 9 2 3 . 4 }$ </td><td> $2 6 . 4 \pm 3 . 4$ </td></tr><tr><td>DynaMo-7B-T3</td><td> $2 6 . 3 _ { \pm 3 . 1 }$ </td><td> $2 5 . 3 _ { \pm 3 . 1 }$ </td><td> ${ \bf 2 7 . 8 _ { \pm 3 . 7 } }$ </td><td> $2 6 . 6 _ { \pm 3 . 4 }$ </td><td> ${ 2 6 . 6 } _ { \pm 3 . 4 }$ </td></tr></table>

## C.3 Additional Downstream Performance Results

We now present additional results on downstream benchmarks.

## C.3.1 Closed-book Question Answering

Next, we compare the performance of DynaMo with that of the baseline Pythia models on the TriviaQA closed-book question answering benchmark. We test the five-shot performance of models and report the exact match results. Table 11 shows the results. We can see that the DynaMo models significantly outperform the baselines, especially as the models become larger.

## C.3.2 Reading Comprehension

We evaluate the models on the RACE (Lai et al., 2017) and SQuAD2.0 (Rajpurkar et al., 2018)

Table 14: Likelihood difference (lower is better) and percentage stereotype (50% is better) on the CrowS-Pairs benchmark along with scores (higher is better) on the MC1 and MC2 tasks in the TruthfulQA benchmark.
<table><tr><td rowspan="2">Model</td><td colspan="2">CrowS-Pairs</td><td colspan="2">TruthfulQA</td></tr><tr><td>LLD</td><td> $\mathbf { S t e r e o t y p e }$ </td><td>MC1</td><td> $\mathbf { M C } 2$ </td></tr><tr><td>Pythia-70M</td><td> $3 . 7 _ { \pm 0 . 1 }$ </td><td> $5 5 . 4 _ { \pm 1 . 2 }$ </td><td> $2 5 . 3 _ { \pm 1 . 5 }$ </td><td> $4 7 . 5 _ { \pm 1 . 6 }$ </td></tr><tr><td>DynaMo-77M-T3</td><td> $3 . 7 _ { \pm 0 . 1 }$ </td><td> ${ \pm } 4 . 9 { \pm } 1 . 2 $ </td><td> $2 5 . 1 { \pm } 1 . 5$ </td><td> $4 7 . 0 { \scriptstyle \pm 1 . 6 }$ </td></tr><tr><td rowspan="2">Pythia-160M DynaMo-180M-T3</td><td> ${ \bf 4 . 3 _ { \pm 0 . 1 } }$ </td><td> $5 4 . 7 _ { \pm 1 . 2 }$ </td><td> $2 4 . 7 { \scriptstyle \pm 1 . 5 }$ </td><td> $4 4 . 4 { \scriptstyle \pm 1 . 5 }$ </td></tr><tr><td> ${ \bf 4 . 3 _ { \pm 0 . 1 } }$ </td><td> ${ \bar { 5 } } 3 . 6 _ { \pm 1 . 2 }$ </td><td> $2 4 . 0 { \scriptstyle \pm 1 . 5 }$ </td><td> $4 3 . 2 _ { \pm 1 . 5 }$ </td></tr><tr><td rowspan="2">Pythia-410M DynaMo-430M-T3</td><td> ${ \bf 3 . 5 { \scriptstyle \pm 0 . 1 } }$ </td><td> ${ \pm } 8 . 6 { \scriptstyle \pm 1 . 2 }$ </td><td> $2 3 . 6 { \scriptstyle \pm 1 . 5 }$ </td><td> $4 1 . 0 { \scriptstyle \pm 1 . 5 }$ </td></tr><tr><td> $3 . 6 _ { \pm 0 . 1 }$ </td><td> $5 8 . 7 _ { \pm 1 . 2 }$ </td><td> $2 3 . 7 _ { \pm 1 . 5 }$ </td><td> ${ \bf 4 1 . 1 _ { \pm 1 . 5 } }$ </td></tr><tr><td rowspan="2">Pythia-1B DynaMo-1.1B-T3</td><td> ${ \bf 3 . 4 } _ { \pm 0 . 1 }$ </td><td> ${ \bf 6 3 . 1 { \scriptstyle \pm 1 . 2 } }$ </td><td> $2 2 . 6 { \scriptstyle \pm 1 . 5 }$ </td><td> $3 8 . 9 { \scriptstyle \pm 1 . 4 }$ </td></tr><tr><td> $3 . 5 _ { \pm 0 . 1 }$ </td><td> $6 3 . 3 _ { \pm 1 . 2 }$ </td><td> $2 2 . 8 _ { \pm 1 . 5 }$ </td><td> $3 9 . 3 _ { \pm 1 . 4 }$ </td></tr><tr><td rowspan="2">Pythia-1.4B DynaMo-1.5B-T3</td><td> $3 . 5 _ { \pm 0 . 1 }$ </td><td> $6 1 . 4 _ { \pm 1 . 2 }$ </td><td> $2 3 . 0 _ { \pm 1 . 5 }$ </td><td> $3 8 . 6 _ { \pm 1 . 4 }$ </td></tr><tr><td> $3 . 6 _ { \pm 0 . 1 }$ </td><td> ${ \bf 6 1 . 0 _ { \pm 1 . 2 } }$ </td><td> $2 3 . 6 _ { \pm 1 . 5 }$ </td><td> ${ \bf 3 9 . 0 _ { \pm 1 . 4 } }$ </td></tr><tr><td>Pythia-2.8B</td><td> ${ \bf 3 . 4 _ { \pm 0 . 1 } }$ </td><td> ${ \bf 6 3 . 4 } _ { \pm 1 . 2 }$ </td><td> ${ 2 1 . 2 } _ { \pm 1 . 4 }$ </td><td> $3 5 . 6 _ { \pm 1 . 4 }$ </td></tr><tr><td rowspan="2">DynaMo-2.9B-T3</td><td> ${ \bf 3 . 4 _ { \pm 0 . 1 } }$ </td><td> $6 2 . 3 _ { \pm 1 . 2 }$ </td><td> $2 0 . 4 _ { \pm 1 . 4 }$ </td><td> $3 5 . 8 _ { \pm 1 . 4 }$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Pythia-6.9B  $\mathrm { D y n a M o - } 7 . 3 \mathbf { B } \mathbf { - } \mathrm { T } 3$ </td><td> $3 . 8 _ { \pm 0 . 1 }$   $3 . 7 _ { \pm 0 . 1 }$ </td><td> $6 3 . 2 _ { \pm 1 . 2 }$   ${ \bf 6 2 . 8 _ { \pm 1 . 2 } }$ </td><td> $2 1 . 7 _ { \pm 1 . 4 }$   ${ \bf 2 1 . 8 _ { \pm 1 . 4 } }$ </td><td> $3 5 . 2 _ { \pm 1 . 3 }$   $3 5 . 2 _ { \pm 1 . 3 }$ </td></tr></table>

benchmarks in Table 12. Again, DynaMo outperforms Pythia on most model sizes.

## C.3.3 Massive Multitask Language Understanding

Next, we report performance on the massive multitask language understanding (MMLU) benchmark, introduced by Hendrycks et al. (2021). It consists of multiple-choice questions that cover various knowledge domains, including humanities, STEM, and social sciences. We present five-shot accuracy results in Table 13. We observe that most models have accuracy close to random chance (25%). Recent literature reports that models trained with much more data break the random performance barrier for these model sizes (Geng and Liu, 2023; Touvron et al., 2023b). We plan to train multi-token counterparts of such models in the future.

## C.3.4 Bias and Misinformation

Table 14 shows the effect of multi-token training on bias and misinformation in the DynaMo suite of models. We report performance on the CrowS-Pairs (Nangia et al., 2020) and the TrthfulQA benchmarks (Lin et al., 2022). The former tests the model’s biases along nine categories: gender, religion, race/color, sexual orientation, age, nationality, disability, physical appearance, and socioeconomic status. The latter tests the model’s ability to generate false claims, i.e., to hallucinate. We observe that multi-token training does not significantly affect the model’s bias and misinformation abilities.

(b)  
![](images/9c10dc088d45a85b301c5f15f31d233b662cab67a332b5b89dd4376becc12be8.jpg)  
Figure 15: Dynamic multi-token perplexity $\mathrm { ( P P L _ { d } ) }$ for different models in the DynaMo suite. Effect of $\epsilon _ { b }$ on (a) $\mathrm { P P L _ { d } }$ and (b) speed-up. (c) Plot of PPL<sub>d</sub> vs. speed-up.

## C.4 Dynamic Multi-token Perplexity

For a given threshold $\epsilon _ { b } .$ , the DynaMo model $\mathrm { d y } .$ namically backs off to lower-order prediction based on input context and predicted joint probability distribution. We calculate the dynamic multi-token perplexity $\mathrm { P P L _ { d } }$ based on the number of tokens generated. Fig. 15 plots $\mathrm { P P L _ { d } }$ against the resultant mean speed-up on the validation set. We observe that $\mathrm { P P L _ { 1 } }$ (i.e., $\mathrm { P P L _ { d } }$ at $1 \times$ speed-up) drops as models become larger. The slope of the curve also reduces. This shows promise for multi-token prediction by larger models beyond those in the current DynaMo suite.

## C.5 Sentence Completion Benchmark

We now present additional results on the sentence completion benchmark. We use LLMs trained under the CLM (or modified-CLM) objective to complete the sentence for a given prompt in the sentence-completion benchmark (details in $\mathsf { A p - }$ pendix A.3). We use GPT-3.5 to rate the text generations in single-mode and pairwise evaluations against Pythia.

## C.5.1 Single-mode Evaluation

Fig. 16 shows the histograms for the GPT scores on the sentence-completion benchmark for text generations by Pythia-70M and DynaMo-77M-T3. We evaluated 100 generations (ten for each prompt, with a separate random seed) for both models.

Fig. 17 shows the GPT scores for DynaMo-77M-T3 on the sentence-completion benchmark for different speed-ups. Since the speed-up varies for different text generations (even for the same prompt) with $\epsilon _ { b } .$ , we plot a regression line to predict the GPT for a target speed-up. We leveraged these predicted

![](images/f6f9910daf9706d482142fa6ae4c5362fbbcf44d4fc1d80fea3b256ee838e84a.jpg)  
(a)

![](images/f98b1e3438b7c57034e16d06a6de7721b3f6653542bfb75785969d3b64c8a2fb.jpg)

Figure 16: Histograms of GPT scores for single-mode evaluations on the sentence-completion benchmark for (a) Pythia-70M and (b) DynaMo-77M-T3 $( \epsilon _ { b } = 1 . 0 ) $ . GPT-3.5 is used as the judge.  
![](images/32b3429e96efaa3c120ed58884e20af5a28c69f04e371f64b624f3882e615935.jpg)  
Figure 17: GPT scores for DynaMo-77M-T3 on the sentencecompletion benchmark plotted against speed-up. GPT-3.5 is used as the judge. The mean GPT score for Pythia-70M is plotted as a black dashed line. Regression plotted with 95% confidence intervals.

GPT scores to plot Fig. 18, which shows the evolution of GPT scores with increasing model sizes. We plot the mean GPT scores of the Pythia models. Further, we plot the mean GPT scores of the DynaMo models at different speed-ups. We regress the GPT scores at a target speed-up using GPT score $\mathrm { V S } , \epsilon _ { b }$ and wallclock speed-up vs. $\epsilon _ { b }$ plots. As $\epsilon _ { b }$ increases, the GPT score increases, but speed-up decreases. The DynaMo models outperform the baseline at 1 speed-up, improving performance as the model size increases.

## C.5.2 Pairwise Evaluation

Fig. 19 shows the pairwise performance and speedups for DynaMo-77M-T3 against baseline Pythia-70M. For every prompt, at every $\epsilon _ { b } ,$ each bar plots the wins, ties, and losses of DynaMo-77M-T3 over ten text generations (in green, yellow, and red, respectively). We show a regression plot for winrates (wins/losses) against speed-ups (for different $\epsilon _ { b } \mathbf { \ ' } _ { \mathbf { s } } )$ in Fig. 3.

![](images/e309fc6ecc2102f1a2b171ffd14178b0644047a71405f719bf9a4dc62b08e6ee.jpg)  
Figure 18: Effect of model size on GPT scores. We plot the GPT scores for DynaMo models at different speed-ups. GPT-3.5 used to judge generation quality on a scale 1-10.

Next, we study the effect of model sizes and parameter overheads on the obtained speed-ups. Every DynaMo model instantiated from a base Pythia model trains additional decoder layers for the second- and third-token heads. This results in a parameter overhead for each DynaMo model relative to its $\mathrm { P y }$ thia counterpart. Fig. 20 shows that speed-up increases with model size and decreases with parameter overhead, albeit with low statistical significance. Nevertheless, this shows promise for high speed-ups in larger multi-token LLMs. Note that, for the models in the DynaMo suite, model sizes and their parameter overheads are not uncorrelated [see inset in Fig. 20(a)]. Thus, we need more rigorous scaling experiments to test the effect of model sizes and parameter overheads on the obtained speed-up, which we leave to future work.

Fig. 21 shows the variation of win rates and speed-ups across different sentence types for the DynaMo-77M-T3 model on the sentencecompletion benchmark.

## D Sample Text Generations

Figs. 22, 23, and 24 show the generated responses at different speed-ups along with GPT-4’s judgments. We observe that as the target speed-up increases, the grammatical mistakes in the generated response also increase. For 3 speed-up, DynaMo-7.3B-T3 generated unrelated text. Despite using the repetition penalty, we also observe repetitive n-grams generated for smaller models. Grammatical mistakes during multi-token generation should decrease with larger training corpora for subsequent token-head training and with more representative models (e.g., LLaMA-2-70B, Touvron et al. 2023b).

![](images/49406c74f56a36ce1ed2d0d7d44716a5088732f3dedb79be818668be2ef90cf8.jpg)

![](images/550659daa3902691c0422d062b88ddd8da1bdeec802c8f4dc55e246a7f75139f.jpg)  
Figure 19: Normalized pairwise performance and speed-ups of DynaMo-77M-T3 on the sentence-completion benchmark plotted against ϵ<sub>b</sub>.

![](images/f79ab87aeb7bf8ffe75bf61969df5c6bdaff155404d85e53eece9290729a20c3.jpg)  
(a)

![](images/132c4940a0206a135254021d9b724b14535b4748c7931def972c3dce26eabf08.jpg)  
(b)  
Figure 20: Speed-up, i.e., the minimum of (theoretical) samequality speed-up and 3 for three-token models, with (a) model sizes and (b) parameter overheads. Results are shown for pairwise evaluation on the sentence-completion benchmark. Only points below 3 speed-up were used to plot the regression line (shown with 95% confidence intervals). Parameter overheads with model sizes are shown in the inset.

![](images/6ac441ac6e86f1b63aca3ce6577e49363447f7e09b95c562764ff33db8b07c5c.jpg)  
Figure 21: Pairwise performance on the sentence-completion benchmark categorized by different sentence types. Radar charts for mean (a) win rates and (b) speed-ups for different $\bar { \epsilon _ { b } } \mathrm { ^ { * } s }$ are shown.

![](images/dca073bc9e76e1217dcb05ca8f2fe7c80a4421001bee006a7fe13caa8b36cb24.jpg)  
Figure 22: Question, Pythia-6.9B’s and DynaMo-7.3B-T3’s responses at 1 speed-up, along with GPT-4’s judgements.

![](images/36537ff221bc682c34c32b4c5eb334677ee636553c663a7919a005fbf8434918.jpg)  
Figure 23: Question, Pythia-6.9B’s and DynaMo-7.3B-T3’s responses at 2.62 speed-up, along with GPT-4’s judgements. A blatant grammatical mistake is highlighted in yellow.

![](images/006f79b3245569a8abc8a195672bfaa3e72ede6a40eb292d20f53ce08349b09d.jpg)  
Figure 24: Question, Pythia-6.9B’s and DynaMo-7.3B-T3’s responses at 3 speed-up, along with GPT-4’s judgements. Blatant grammatical mistakes are highlighted in yellow.