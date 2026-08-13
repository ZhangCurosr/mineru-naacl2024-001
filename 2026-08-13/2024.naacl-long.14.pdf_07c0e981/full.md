# An Empirical Study of Consistency Regularization for End-to-End Speech-to-Text Translation

Pengzhi Gao, Ruiqing Zhang, Zhongjun He, Hua Wu, and Haifeng Wang Baidu Inc. No. 10, Shangdi 10th Street, Beijing, 100085, China {gaopengzhi,zhangruiqing01,hezhongjun}@baidu.com {wu\_hua,wanghaifeng}@baidu.com

## Abstract

Consistency regularization methods, such as R-Drop (Liang et al., 2021) and CrossConST (Gao et al., 2023), have achieved impressive supervised and zero-shot performance in the neural machine translation (NMT) field. Can we also boost end-to-end (E2E) speech-to-text translation (ST) by leveraging consistency regularization? In this paper, we conduct empirical studies on intra-modal and cross-modal consistency and propose two training strategies, Sim-RegCR and SimZeroCR, for E2E ST in regular and zero-shot scenarios. Experiments on the MuST-C benchmark show that our approaches achieve state-of-the-art (SOTA) performance in most translation directions. The analyses prove that regularization brought by the intra-modal consistency, instead of the modality gap, is crucial for the regular E2E ST, and the cross-modal consistency could close the modality gap and boost the zero-shot E2E ST performance.

## 1 Introduction

Speech-to-text translation takes acoustic speech signals as input and outputs text translations in the target language. The conventional cascaded ST system consists of an automatic speech recognition (ASR) system and a machine translation (MT) module in a pipeline manner (Sperber et al., 2017, 2019; Zhang et al., 2019). Recent works on ST have focused on the end-to-end system, which learns a unified model that directly generates text translations from speech without any intermediate outputs (Duong et al., 2016; Berard et al., 2016). E2E ST is a cross-modal task, where the major challenges include parallel ST data scarcity and representation discrepancy between speech and text modalities. In order to boost E2E ST training, the techniques utilized by existing approaches include pretraining (Wang et al., 2020b; Xu et al., 2021), multi-task learning (Ye et al., 2021; Tang et al., 2021a), knowledge distillation (Liu et al., 2019; Inaguma et al., 2021), and cross-modal representation learning (Ye et al., 2022; Wang et al., 2022; Fang and Feng, 2023b). However, most methods are far from being widely used due to the sophisticated model architecture, complicated algorithm implementation, and tedious hyperparameter search.

Consistency regularization has been widely adopted and shown great promise to improve NMT performance (Sato et al., 2019; Chen et al., 2021; Liang et al., 2021; Gao et al., 2022, 2023). Specifically, Liang et al. (2021) introduce an intra-lingual consistency, R-Drop, to regularize dropout and improve the supervised NMT performance, and Gao et al. (2023) propose a cross-lingual consistency, CrossConST, to learn universal representations and boost the zero-shot NMT performance. Given the similar problem formulations between NMT and E2E ST, a natural question arises: Can we significantly improve E2E STperformance by leveraging simple consistency regularization?

In this paper, our primary goal is to provide a simple, easy-to-reproduce, but tough-to-beat strategy for learning E2E ST models. Inspired by Liang et al. (2021) and Gao et al. (2023), we propose two strategies, SimRegCR and SimZeroCR, for training E2E ST models in regular and zero-shot scenarios. We show that intra-modal consistency is crucial for the regular setting, and cross-modal consistency is the key to closing the modality gap and boosting the zero-shot performance. The contributions of this paper can be summarized as follows:

• We conduct empirical studies on consistency regularization and propose two simple but effective strategies for learning E2E ST models in regular and zero-shot scenarios.

• Experimental results show that our approaches achieve significant improvements on the MuST-C benchmark and outperform the current SOTA methods CRESS (Fang and Feng, 2023b) and DCMA (Wang et al., 2022).

## 2 Background

## 2.1 End-to-End Speech-to-Text Translation

Speech translation corpora usually contain speechtranscription-translation triples, which can be denoted as $\begin{array} { r } { \textbf {  { S } } = ~ \{ \mathbf { s } ^ { i } , \mathbf { x } ^ { i } , \mathbf { y } ^ { i } \} _ { i = 1 } ^ { | \mathcal { S } | } } \end{array}$ s denotes the sequence of the audio wave, x is the transcription in the source language, and y represents the translation in the target language.  could be pairwise combined into three parallel corpora, $S _ { a s r } ~ = ~ \{ { \bf s } ^ { i } , { \bf x } ^ { i } \} _ { i = 1 } ^ { | { \cal S } | } , ~ S _ { m t } ~ = ~ \{ { \bf x } ^ { i } , { \bf y } ^ { i } \} _ { i = 1 } ^ { | { \cal S } | }$ , and ${ \cal S } _ { s t } = \{ { \bf s } ^ { i } , { \bf y } ^ { i } \} _ { i = 1 } ^ { | S | }$ , for ASR, MT, and ST tasks respectively. The goal of E2E ST is to generate translation y directly from the speech s without generating transcription x, and the standard training objective is to minimize the empirical risk:

$$
\mathcal { L } _ { c e } ^ { s t } ( \theta ) = \ell ( f ( \mathbf { s } , \mathbf { y } ; \theta ) , \ddot { \mathbf { y } } ) ,\tag{1}
$$

where ℓ denotes the cross-entropy loss, θ is a set of model parameters, $f ( \mathbf { s } , \mathbf { y } ; \boldsymbol { \theta } )$ is a sequence of probability predictions, and $\ddot { \mathbf { y } }$ is a sequence of one-hot label vectors for y. Directly modeling the speechto-text mapping is nontrivial due to the representation discrepancy between speech and text modalities. To alleviate ST data sparsity, people usually include ASR and MT supervisions from $\boldsymbol { S _ { a s r } }$ and $S _ { m t }$ as well as external corpora for E2E ST tasks.

## 2.2 Consistency Regularization for Neural Machine Translation

Liang et al. (2021) propose an intra-lingual consistency regularization, R-Drop, for boosting NMT performance by forcing the output distributions of different sub-models generated by dropout to be consistent with each other. For each sentence pair $\displaystyle ( \mathbf { x } , \mathbf { y } )$ , the training objective is defined as:

$$
\mathcal { L } _ { R - D r o p } ( \theta ) = \mathcal { L } _ { c e } ^ { m t } ( \theta ) + \alpha \mathcal { L } _ { i n t r a } ^ { m t } ( \theta ) ,\tag{2}
$$

where

$$
\mathcal { L } _ { c e } ^ { m t } ( \theta ) = \ell ( f ( \mathbf { x } , \mathbf { y } ; \theta ) , \ddot { \mathbf { y } } ) ,\tag{3}
$$

$$
\begin{array} { r } { \mathcal { L } _ { i n t r a } ^ { m t } ( \theta ) = \mathbf { J } \mathbf { S } ( f _ { 1 } ( \mathbf { x } , \mathbf { y } ; \theta ) , f _ { 2 } ( \mathbf { x } , \mathbf { y } ; \theta ) ) , } \end{array}\tag{4}
$$

$f _ { 1 } ( \cdot )$ and $f _ { 2 } ( \cdot )$ denote the two forward passes of the same model $f ( \cdot )$ with the dropout operation, $\mathrm { J } { \bf S } ( \cdot , \cdot )$ is the Jeffreys (JS) divergence<sup>1</sup> of two distributions,

$$
\begin{array} { r } { \mathbf { J } \mathbf { S } ( a , b ) = ( \mathbf { K } \mathbf { L } ( a \| b ) + \mathbf { K } \mathbf { L } ( b \| a ) ) / 2 , } \end{array}\tag{5}
$$

KL( ) denotes the Kullback-Leibler (KL) divergence, and α is a scalar hyperparameter.

Gao et al. (2023) introduce a cross-lingual consistency regularization, CrossConST, for bridging the representation gap among different languages and improving zero-shot translation in multilingual NMT. For each sentence pair (x, y), the training objective is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { C r o s s C o n S T } ( \theta ) = \mathcal { L } _ { c e } ^ { m t } ( \theta ) + \beta \mathcal { L } _ { c r o s s } ^ { m t } ( \theta ) , } \end{array}\tag{6}
$$

where

$$
\mathcal { L } _ { c r o s s } ^ { m t } ( \boldsymbol { \theta } ) = \mathrm { K L } ( f ( \mathbf { x } , \mathbf { y } ; \boldsymbol { \theta } ) | | f ( \mathbf { y } , \mathbf { y } ; \boldsymbol { \theta } ) ) ,\tag{7}
$$

and $\beta$ is a scalar hyperparameter.

## 3 Datasets and Baseline Settings

## 3.1 Dataset Description

We initially consider en de translation for empirical study on consistency regularization in Section 4 and then show further experiments for other translation directions in Section 5. The detailed statistics of all datasets are summarized in Table 9.

## 3.1.1 ST Datasets

We conduct experiments on MuST-C (Di Gangi et al., 2019), which is a multilingual speech translation dataset containing audio recordings with the corresponding transcriptions and translations from English (en) to 8 languages: German (de), Spanish (es), French (fr), Italian (it), Dutch (nl), Portuguese (pt), Romanian (ro), and Russian (ru). We use dev and tst-COMMON as the validation and test sets respectively.

## 3.1.2 MT Datasets

We utilize external MT datasets to boost the E2E ST performance. Specifically, we incorporate WMT13 (Bojar et al., 2013) dataset for en es, WMT14 (Bojar et al., 2014) dataset for en fr, WMT16 (Bojar et al., 2016) datasets for en de/ro/ru, and OPUS100 (Zhang et al., 2020) datasets for en it/nl/pt. Note that we also use dev and tst-COMMON in the MuST-C dataset as the validation and test sets for the MT tasks.

## 3.2 Baseline Settings

We adopt a widely used baseline model, W2V2- Transformer (Ye et al., 2021) in our empirical study (Figure 1), which consists of a learnable acoustic feature extractor before two 1-dimensional convolutional layers and the standard Transformer architecture (Vaswani et al., 2017). We use different language tags at the decoder input to distinguish the target languages. During inference, the language tag serves as the initial token to predict the output text. For example, if the speech input for the sentence “The weather is good today” is in English, to perform ASR, we use <en> as the initial token and decode “The weather is good today”, while to translate into German, we use <de> as the initial token and decode “Das Wetter heute ist gut”.

![](images/1f385241806dd554080e3beb2039a6b7b49246a0e0e8b082ea059071b9f8fe00.jpg)  
Figure 1: Illustration of the intra-modal and cross-modal consistency regularization. For $\mathcal { L } _ { i n t r a } ^ { s t } ( \theta )$ , the Speech-German pair (Speech, "Das Wetter heute ist $\mathrm { \ g u t " ) }$ goes through the E2E ST model twice and obtains two output distributions $f ( \mathbf { s } , \mathbf { y } ; \boldsymbol { \theta } )$ . For $\mathcal { L } _ { c r o s s } ^ { a s r } ( \theta )$ , the original Speech-English pair (Speech, "The weather is good today") and the copied English-English pair ("The weather is good today", "The weather is good today") go through the E2E ST model and the NMT model respectively and obtain two output distributions $f ( \mathbf { s } , \mathbf { x } ; \theta )$ and $f ( \mathbf { x } , \mathbf { x } ; \theta )$

Pre-processing For speech input, we utilize the raw 16-bit 16kHz mono-channel audio wave. Following common practice, utterances with less than 1000 frames are removed, and utterances with more than 480000 frames are removed in the training set for GPU efficiency. For each translation direction, we jointly learn a unigram SentencePiece (Kudo and Richardson, 2018) model with size 10K on both the source and target sentences and use it to segment sentences into subwords for MT and ST tasks. For the external MT datasets, we filter out parallel sentences which length ratio exceeds 1.5.

Model Configuration We use wav2vec2.0<sup>2</sup> (Baevski et al., 2020) as the acoustic feature extractor, which is pretrained on the audio data from LibriSpeech (Panayotov et al., 2015). Two 1- dimensional convolutional layers are added following the acoustic feature extractor, with kernel size 5, stride size 2, padding 2, and hidden dimension 1024. We utilize 6-layer transformer encoder and 6-layer transformer decoder. Each of the transformer layers comprises 512 hidden units, 8 attention heads, and 2048 feed-forward hidden units.

Training Configuration We apply cross-entropy loss with label smoothing rate 0.1 and set max tokens per batch to be 4096 for the MT task and 2000000 for the ASR and ST tasks. We use the Adam optimizer with Beta (0.9, 0.98), 4000, 8000, and 4000 warmup updates, and inverse square root learning rate scheduler with initial learning rate 1e−<sup>4</sup>, 1e−<sup>3</sup>, and 1e−<sup>4</sup> for the ASR, MT, and ST tasks respectively. We apply the same configuration in each stage of the training procedure. During inference, we use beam search decoding with a beam size of 8 with length penalty 1.2, 0.6, 1.8, 1.0, 1.0, 1.4, 1.4, and 0.8 for en de, es, fr, it, nl, pt, ro, and ru, respectively. We evaluate the MT and ST tasks by case-sensitive sacreBLEU (Post, 2018). We train all models until convergence on 8 NVIDIA Tesla V100 GPUs. For all the experiments below, we select the saved model state with the best validation performance.

## 4 Methodology

In this section, we formally propose SimRegCR and SimZeroCR, the consistency-based strategies for learning E2E ST models in regular (Section 4.1) and zero-shot (Section 4.2) scenarios respectively. We introduce the details of each part below.

## 4.1 Consistency Regularization for Regular End-to-End Speech Translation

We here investigate the performance of consistency regularization for the regular scenario, where we learn the E2E ST model by utilizing MT and ST datasets. For each training sample, the loss functions include: $\mathcal { L } _ { c e } ^ { m t } ( \theta ) , \mathcal { L } _ { i n t r a } ^ { m i t } ( \theta ) , \mathcal { L } _ { c e } ^ { s t } ( \theta )$

$$
\begin{array} { r } { \mathcal { L } _ { i n t r a } ^ { s t } ( \boldsymbol { \theta } ) = \mathbf { J } \mathbf { S } ( f _ { 1 } ( \mathbf { s } , \mathbf { y } ; \boldsymbol { \theta } ) , f _ { 2 } ( \mathbf { s } , \mathbf { y } ; \boldsymbol { \theta } ) ) , } \end{array}\tag{8}
$$

and

$$
\begin{array} { r } { \mathcal { L } _ { c r o s s } ^ { m t - s t } ( \theta ) = \mathrm { K L } ( f ( \mathbf { x } , \mathbf { y } ; \theta ) | | f ( \mathbf { s } , \mathbf { y } ; \theta ) ) , } \end{array}\tag{9}
$$

<table><tr><td>ID</td><td>Training Stage</td><td>Loss Function</td><td>MT BLEU</td><td>ST BLEU</td></tr><tr><td>①</td><td>MT train from scratch</td><td> $\overline { { \mathcal { L } _ { c e } ^ { m t } } }$ </td><td>29.33</td><td>一</td></tr><tr><td>② ③</td><td>MT train from scratch ST train from scratch</td><td> $\mathcal { L } _ { c e } ^ { m t } + \alpha \mathcal { L } _ { i n t r a } ^ { m t }$   $\overline { { \mathcal { L } _ { c e } ^ { s t } } }$ </td><td>32.76</td><td>23.49</td></tr><tr><td>④</td><td>ST train from scratch</td><td> $\mathcal { L } _ { c e } ^ { s t } + \alpha \mathcal { L } _ { i n t r a } ^ { s t }$ </td><td></td><td>26.77</td></tr><tr><td>⑤</td><td>ST finetune on ①</td><td> $\mathcal { L } _ { c e } ^ { s t }$ </td><td></td><td>24.38</td></tr><tr><td>⑥</td><td>ST finetune on ①</td><td> $\mathcal { L } _ { c e } ^ { s t } + \alpha \mathcal { L } _ { i n t r a } ^ { s t }$ </td><td></td><td>27.35</td></tr><tr><td>⑦</td><td></td><td> $\mathcal { L } _ { c e } ^ { s t } + \alpha \mathcal { L } _ { i n t r a } ^ { s t }$ </td><td></td><td>27.91</td></tr><tr><td></td><td>ST finetune on ②</td><td> $\overline { { \mathcal { L } _ { c e } ^ { m t } + \mathcal { L } _ { c e } ^ { s t } } }$ </td><td>28.54</td><td></td></tr><tr><td>8</td><td>MT &amp; ST train from scratch</td><td> $\mathcal { L } _ { c e } ^ { m t } + \mathcal { L } _ { c e } ^ { s t }$ </td><td>29.73</td><td>23.75</td></tr><tr><td>⑨</td><td>MT &amp; ST finetune on ①</td><td></td><td></td><td>23.82</td></tr><tr><td>①0</td><td>MT &amp; ST finetune on ①</td><td> $\mathcal { L } _ { c e } ^ { m t } + \mathcal { L } _ { c e } ^ { s t } + \beta \mathcal { L } _ { c r o s s } ^ { m t - s t }$ </td><td>30.66</td><td>26.87</td></tr><tr><td>①</td><td>MT &amp; ST finetune on ②</td><td> $\mathcal { L } _ { c e } ^ { m t } + \alpha \mathcal { L } _ { i n t r a } ^ { m t } + \mathcal { L } _ { c e } ^ { s t } + \alpha \mathcal { L } _ { i n t r a } ^ { s t }$ </td><td>32.70</td><td>27.48</td></tr><tr><td>①2</td><td>MT &amp; ST finetune on ①</td><td> $\mathcal { L } _ { c e } ^ { m t } + \alpha \mathcal { L } _ { i n t r a } ^ { m t } + \mathcal { L } _ { c e } ^ { s t } + \alpha \mathcal { L } _ { i n t r a } ^ { s t } + \beta \mathcal { L } _ { c r o s s } ^ { m t - s t }$ </td><td>31.00</td><td>27.57</td></tr><tr><td>13</td><td>MT train from scratch†</td><td>Lmt</td><td>29.61</td><td></td></tr><tr><td>14</td><td>MT train from scratch†</td><td> $\mathcal { L } _ { c e } ^ { m t } + \alpha \mathcal { L } _ { i n t r a } ^ { m t }$ </td><td>30.02</td><td></td></tr><tr><td>15</td><td>MT finetune on 13</td><td> $\mathcal { L } _ { c e } ^ { m t }$ </td><td>33.59</td><td></td></tr><tr><td>16</td><td>MT finetune on 1④</td><td> $\mathcal { L } _ { c e } ^ { m t } + \alpha \mathcal { L } _ { i n t r a } ^ { m t }$ </td><td>34.11</td><td></td></tr><tr><td>17</td><td>ST finetune on 15</td><td> $\mathcal { L } _ { c e } ^ { s t }$ </td><td></td><td>27.33</td></tr><tr><td>18</td><td>ST finetune on 15</td><td> $\mathcal { L } _ { c e } ^ { s t } + \alpha \mathcal { L } _ { i n t r a } ^ { s t }$ </td><td></td><td>28.96</td></tr><tr><td>19</td><td>ST finetune on 16</td><td> $\mathcal { L } _ { c e } ^ { s t } + \alpha \mathcal { L } _ { i n t r a } ^ { s t }$ </td><td></td><td>29.23</td></tr></table>

Table 1: Case-sensitive detokenized BLEU scores on the MuST-C en de tst-COMMON set. denotes the MT training is performed on the WMT16 dataset, and other MT training is performed on the MuST-C dataset. We mark the best ST BLEU scores in two experimental setups in bold. The choices for α and $\beta$ are summarized in Table 10. Experimental results on more languages are summarized in Table 12.

where (1) and (3) are the cross-entropy loss for the ST and MT tasks respectively, (4) and (8) are the intra-modal consistency regularization for the MT and ST tasks respectively, and (9) denotes the crossmodal consistency regularization between the MT and ST tasks, which could also be regarded as the sequence-level knowledge distillation from the MT model to the ST model (Liu et al., 2019).

## 4.1.1 Experimental Results

We consider two experimental setups: without external MT data ( 1 - 12 ) and with external MT data $( \textcircled { 1 3 } - \textcircled { 1 9 } )$ , and summarize the experimental results in Table 1. For each experiment in Table 1, we conduct a careful grid search to select the best hyperparameters, α and $\beta ,$ for the model performance. Note that 5 and 17 correspond to the W2V2-Transformer baselines in the settings of without and with external MT data respectively. By checking model performance under different combinations of loss function and training strategy, we have the following observations: 1) The intra-modal consistency, $\mathcal { L } _ { i n t r a } ^ { m t }$ and $\mathcal { L } _ { i n t r a } ^ { s t }$ , could boost the MT ( 1 vs $\textcircled { 2 } ; \textcircled { 1 3 }$ vs 14 ) and ST $( \textcircled { 3 } \ \textbf { V S } \textcircled { 4 } )$ performance. 2) The paradigm of pretraining-finetuning could further improve the ST performance $( \textcircled { 3 } \mathrm { v s } \textcircled { 5 } ; \textcircled { 4 } \mathrm { v s } \textcircled { 7 } ) . 3 )$ The multi-task learning achieves similar performance compared with the pretraining-finetuning strategy ( 3 vs 8 ; 5 vs 9 ). 4) The cross-modal consistency, $\mathcal { L } _ { c r o s s } ^ { m t - s t }$ , could improve the ST performance ( 9 vs $\textcircled{1 0 } \vdots \textcircled { 1 1 }$ vs 12 ) but still achieve the sub-optimal performance ( 7 vs 12 ).

## 4.1.2 Does Intra-modal Consistency Implicitly Bridge the Modality Gap?

![](images/98521471072a80d7cea1228222a9e2bde1051de71999a4f35fc8fc2ca9d57a05.jpg)  
Figure 2: The ST BLEU score and similarity search accuracy of each model in Table 1 on the MuST-C en de tst-COMMON set. The blue circles denote the pretraining-finetuning experiments without external MT data. The green circles denote the multi-task learning experiments without external MT data. The orange circles denote the experiments with external MT data.

One interesting finding from the empirical study is that the strategies ( 7 and 19 ) only utilizing the intra-modal consistency achieve the best ST performance instead of explicitly leveraging the cross-modal consistency. We here investigate the impact of the consistency regularization on the modality gap and the E2E ST performance. We conduct a multimodal similarity search experiment and use the averaged bidirectional similarity search accuracy as the metric to evaluate the modality gap. Given parallel speech-transcription pairs, we find the nearest neighbor for each one in the other modality according to the representation cosine similarity and compute the corresponding accuracy, where the speech and transcription representations are calculated by max-pooling the encoder outputs. The evaluation results are reported in Figure 2. By checking the relationship between ST BLEU score and multimodal similarity search accuracy, we have the following observations: 1) The intra-modal consistency, $\mathcal { L } _ { i n t r a } ^ { m t }$ and $\mathcal { L } _ { i n t r a } ^ { s t }$ , implicitly closes the modality gap $\textcircled{5}$ vs 6 vs 7 ; 17 vs 18 vs 19 ). 2) The cross-modal consistency, $\mathcal { L } _ { c r o s s } ^ { m t - s t }$ explicitly bridges the modality gap ( 9 vs 10 ; 11 vs 12 ). 3) A closer modality gap does not guarantee a better ST performance ( 6 vs 10 ; 7 vs $\textcircled{12}$ ), and the regularization effect introduced by the intra-modal consistency seems to be more crucial for the regular E2E ST task. This empirical evidence aligns with the insight from Han et al. (2023) which posits that modality adaptation efforts do not significantly boost the performance of fully trained models. Overfitting emerges as a more pressing concern, and effective regularization techniques become paramount for regular E2E ST.

## 4.1.3 Training Strategy

![](images/758d3f2ba667f5d321954fa26072cb708bb7f2a1b82a71ca8491630ed67b6a56.jpg)  
Figure 3: The training steps of SimRegCR by utilizing the intra-modal consistency regularization. In each step, the modules that contribute to the final E2E ST model are pointed out by arrow lines. We also consider SimRegCR− ( 18 in Table 1) in this paper, which trains MT model only with $\mathcal { L } _ { c e } ^ { m t }$ in the first two steps.

We here summarize the multi-stage training strategy, SimRegCR $\textcircled { 1 9 }$ in Table 1), consisting of MT pretraining and ST finetuning with the intra-modal consistency regularization in Figure 3. The setting without external MT data only differs by removing the first step of external MT pretraining.

<table><tr><td>Method</td><td colspan="2">BLEU</td></tr><tr><td></td><td>w/o WMT16</td><td>w/ WMT16 27.1</td></tr><tr><td>XSTNet† STEMM†</td><td>25.2</td><td>28.7</td></tr><tr><td>ConST†</td><td>25.6 25.7</td><td>28.3</td></tr><tr><td>CMOT†</td><td>27.0</td><td>29.0 / 28.5*</td></tr><tr><td>CRESS†</td><td>27.2</td><td>29.4 / 28.9*</td></tr><tr><td>W2V2-Transformer</td><td>24.4</td><td></td></tr><tr><td></td><td>27.4</td><td>27.3</td></tr><tr><td>+ SimRegCR</td><td></td><td>29.0</td></tr><tr><td>+ SimRegCR</td><td>27.9</td><td>29.2</td></tr></table>

Table 2: Our method achieves the superior or comparable performance over the existing methods on the MuST-C en de benchmark. denotes the performance of CMOT and CRESS using wav2vec2.0 instead of Hu-BERT as the acoustic feature extractor. denotes the numbers are reported from the corresponding papers, others are based on our runs.

Comparison with Existing Methods We summarize the recent results of several existing works on the MuST-C en de benchmark in Table 2. The existing methods vary from different aspects, including cross-modal progressive training (XST-Net) (Ye et al., 2021), cross-modal manifold mixup (STEMM) (Fang et al., 2022), cross-modal contrastive learning (ConST) (Ye et al., 2022), crossmodal mixup via optimal transport (CMOT) (Zhou et al., 2023), and cross-modal regularization with scheduled sampling (CRESS) (Fang and Feng, 2023b). Note that XSTNet, STEMM, and ConST adopt wav2vec2.0 as the acoustic feature extractor, while CMOT and CRESS use HuBERT (Hsu et al., 2021) which could achieve slightly stronger baseline. We can see that SimRegCR− achieves an improvement of 2.35 BLEU score on average over W2V2-Transformer, and SimRegCR achieves the superior or comparable performance over the current SOTA method CRESS that incorporates cross-modal regularization, scheduled sampling, token-level adaptive training, and a stronger acoustic feature extractor.

## 4.2 Consistency Regularization for Zero-shot End-to-End Speech Translation

We here investigate the performance of consistency regularization for the zero-shot scenario, where we learn the E2E ST model by utilizing ASR and MT datasets. For each training sample, the loss functions include: $\mathcal { L } _ { c e } ^ { m t } ( \theta ) , \mathcal { L } _ { i n t r a } ^ { m t } ( \theta )$

$$
\mathcal { L } _ { c e } ^ { a s r } ( \theta ) = \ell ( f ( \mathbf { s } , \mathbf { x } ; \theta ) , \ddot { \mathbf { x } } ) ,\tag{10}
$$

$$
\begin{array} { r } { \mathcal { L } _ { i n t r a } ^ { a s r } ( \theta ) = \mathbf { J } \mathbf { S } ( f _ { 1 } ( \mathbf { s } , \mathbf { x } ; \theta ) , f _ { 2 } ( \mathbf { s } , \mathbf { x } ; \theta ) ) , } \end{array}\tag{11}
$$

<table><tr><td rowspan=1 colspan=1>ID</td><td rowspan=1 colspan=1>Training Stage</td><td rowspan=1 colspan=1>Loss Function</td><td rowspan=1 colspan=1>MT BLEU</td><td rowspan=1 colspan=1>ST BLEU</td></tr><tr><td rowspan=1 colspan=1>①②</td><td rowspan=1 colspan=1>MT train from scratch†MT train from scratch†</td><td rowspan=1 colspan=1> $\overline { { \mathcal { L } _ { c e } ^ { m t } } }$  $\mathcal { L } _ { c e } ^ { m t } + \alpha \mathcal { L } _ { i n t r a } ^ { m t }$ </td><td rowspan=1 colspan=1>29.6130.02</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>③④</td><td rowspan=1 colspan=1>MT Finetune on ①MT Finetune on②</td><td rowspan=1 colspan=1> $\overline { { \mathcal { L } _ { c e } ^ { m t } } }$  $\mathcal { L } _ { c e } ^ { m t } + \alpha \mathcal { L } _ { i n t r a } ^ { m t }$ </td><td rowspan=1 colspan=1>33.5934.11</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>⑤⑥</td><td rowspan=2 colspan=1>ASR &amp; MT finetune on3ASR &amp; MT finetune on③</td><td rowspan=2 colspan=1> $\overline { { \mathcal { L } _ { c e } ^ { a s r } + \mathcal { L } _ { c e } ^ { m t } } }$  $\mathcal { L } _ { c e } ^ { a s r } + \mathcal { L } _ { c e } ^ { m t } + \beta \mathcal { L } _ { c r o s s } ^ { a s r }$ </td><td rowspan=1 colspan=1>33.99</td><td rowspan=1 colspan=1>0.46</td></tr><tr><td rowspan=1 colspan=1>32.82</td><td rowspan=1 colspan=1>25.10</td></tr><tr><td rowspan=2 colspan=1>⑦8</td><td rowspan=2 colspan=1>ASR &amp; MT finetune on ④ASR &amp; MT finetune on ⑦</td><td rowspan=2 colspan=1> $\overline { { \mathcal { L } _ { c e } ^ { a s r } + \alpha \mathcal { L } _ { i n t r a } ^ { a s r } + \mathcal { L } _ { c e } ^ { m t } + \alpha \mathcal { L } _ { i n t r a } ^ { m t } } }$  $\mathcal { L } _ { c e } ^ { a s r } + \alpha \mathcal { L } _ { i n t r a } ^ { a s r } + \mathcal { L } _ { c e } ^ { m t } + \alpha \mathcal { L } _ { i n t r a } ^ { m t } + \beta \mathcal { L } _ { c r o s s } ^ { a s r }$ </td><td rowspan=1 colspan=1>34.35</td><td rowspan=1 colspan=1>0.56</td></tr><tr><td rowspan=1 colspan=1>33.25</td><td rowspan=1 colspan=1>24.86</td></tr></table>

Table 3: Case-sensitive detokenized BLEU scores on the MuST-C en de tst-COMMON set. denotes the MT training is performed on the WMT16 dataset, and other MT training is performed on the MuST-C dataset. We mark the best ST BLEU score in bold. The choices for α and $\beta$ are summarized in Table 11.

and

$$
\begin{array} { r } { \mathcal { L } _ { c r o s s } ^ { a s r } ( \theta ) = \mathrm { K L } ( f ( \mathbf { s } , \mathbf { x } ; \theta ) \| f ( \mathbf { x } , \mathbf { x } ; \theta ) ) , } \end{array}\tag{12}
$$

where (3) and (10) are the cross-entropy loss for the MT and ASR tasks respectively, (4) and (11) are the intra-modal consistency regularization for the MT and ASR tasks respectively, and (12) denotes the cross-modal consistency regularization for the ASR task, which could be regarded as the multimodal version of CrossConST (Gao et al., 2023).

![](images/9b9725726de05019a45d9a2d6f1930ed7e84c26d9e3970cf3eb7b14a5f59a39d.jpg)

## 4.2.1 Experimental Results

![](images/e30deefffbf135915590f328c5c470e17f67ef1db3a0c0c6cdb04eb083419d11.jpg)  
Figure 4: Bivariate kernel density estimation plots of the speech and transcription representations after using T-SNE dimensionality reduction, where the max-pooled outputs of the W2V2-Transformer encoder are applied as the speech and transcription representations.

We consider the experimental setup with external MT data and summarize the experimental results in Table 3. For each experiment in Table 3, we conduct a careful grid search to select the best hyperparameters, α and $\beta ,$ for the model performance. Note that 5 corresponds to the W2V2- Transformer baseline. By checking model performance under different combinations of loss function and training strategy, we have the following obcould boost the zero-shot ST performance $\textcircled{5}$ vs 6 ; 7 vs 8 ). 2) Leveraging the intra-modal consistency, $\mathcal { L } _ { i n t r a } ^ { a s r }$ and $\mathcal { L } _ { i n t r a } ^ { m t } .$ , could improve the corresponding MT performance ( 5 vs 7 ; 6 vs 8 ), but could not achieve the superior performance in the zero-shot ST direction ( 6 vs 8 ).

## 4.2.2 Does the Cross-modal Consistency Really Close the Modality Gap?

To verify whether the cross-modal consistency regularization can better align the modality representation space, we visualize the speech and transcription representations of the MuST-C en de tst-COMMON set. We apply dimension reduction on the 512-dimensional representations with T-SNE (Hinton and Roweis, 2002) and then depict the bivariate kernel density estimation based on the 2-dimensional representations in Figure 4. Figure 4 shows that the W2V2-Transformer baseline ( 5 ) cannot align speech and transcription well in the representation space, while the cross-modal consistency ( 6 ) draws the representations across different modalities much closer.

## 4.2.3 Training Strategy

![](images/3553665531101597ca6d1f80db7c5a669213e19d3c7ff88d1a387bd39de384ae.jpg)  
Figure 5: The training steps of SimZeroCR by utilizing the cross-modal consistency regularization. In each step, the modules that contribute to the final E2E ST model are pointed out by arrow lines.

We here summarize the multi-stage training strategy, SimZeroCR $\textcircled{6}$ in Table 3), consisting of MT pretraining and ASR & MT finetuning with the cross-modal consistency regularization in Figure 5.

<table><tr><td rowspan="2">Method</td><td rowspan="2">External Speech</td><td colspan="8">BLEU</td></tr><tr><td>de</td><td>es</td><td>fr</td><td>it</td><td>nl</td><td>pt</td><td>ro</td><td>ru</td></tr><tr><td>Fairseq ST (Wang et al., 2020a)</td><td></td><td>22.7</td><td>27.2</td><td>32.9</td><td>22.7</td><td>27.3</td><td>28.1</td><td>21.9</td><td>15.3</td></tr><tr><td>Dual Decoder (Le et al., 2020)</td><td></td><td>23.6</td><td>28.1</td><td>33.5</td><td>24.2</td><td>27.6</td><td>30.0</td><td>22.9</td><td>15.2</td></tr><tr><td>Speechformer (Papi et al., 2021)</td><td></td><td>23.6</td><td>28.5</td><td></td><td></td><td>27.7</td><td></td><td></td><td></td></tr><tr><td>SATE (Xu et al., 2021)</td><td></td><td>25.2</td><td></td><td></td><td>-</td><td></td><td></td><td></td><td></td></tr><tr><td>BiKD (Inaguma et al., 2021)</td><td></td><td>25.3</td><td></td><td>35.3</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>XSTNet (Ye et al., 2021)</td><td>√</td><td>25.5</td><td>29.6</td><td>36.0</td><td>25.5</td><td>30.0</td><td>31.3</td><td>25.1</td><td>16.9</td></tr><tr><td>STEMM (Fang et al., 2022)</td><td>√</td><td>25.6</td><td>30.3</td><td>36.1</td><td>25.6</td><td>30.1</td><td>31.0</td><td>24.3</td><td>17.1</td></tr><tr><td>ConST (Ye et al., 2022)</td><td>√</td><td>25.7</td><td>30.4</td><td>36.8</td><td>26.3</td><td>30.6</td><td>32.0</td><td>24.8</td><td>17.3</td></tr><tr><td>FCCLm (Zhang et al., 2023)</td><td>√</td><td>25.9</td><td>30.7</td><td>36.8</td><td>26.4</td><td>30.5</td><td>31.8</td><td>25.0</td><td>17.6</td></tr><tr><td>M³ST (Cheng et al., 2023)</td><td>√</td><td>26.4</td><td>31.0</td><td>37.2</td><td>26.6</td><td>30.9</td><td>32.8</td><td>25.4</td><td>18.3</td></tr><tr><td>CMOT (Zhou et al., 2023)</td><td>V</td><td>27.0</td><td>31.1</td><td>37.3</td><td>26.9</td><td>31.2</td><td>32.7</td><td>25.3</td><td>17.9</td></tr><tr><td>CRESS (Fang and Feng, 2023b)</td><td>√</td><td>27.2</td><td>31.9</td><td>37.8</td><td>27.3</td><td>31.6</td><td>33.0</td><td>25.9</td><td>18.7</td></tr><tr><td>W2V2-Transformer</td><td>√</td><td>24.4</td><td>29.9</td><td>34.7</td><td>25.1</td><td>29.3</td><td>30.3</td><td>23.4</td><td>16.5</td></tr><tr><td>+ SimRegCR</td><td>√</td><td>27.4</td><td>31.5</td><td>38.1</td><td>27.2</td><td>32.0</td><td>33.3</td><td>25.9</td><td>18.8</td></tr><tr><td>+ SimRegCR</td><td>V</td><td>27.9*</td><td>32.1*</td><td>39.0*</td><td>27.7*</td><td>32.4*</td><td>34.0*</td><td>26.3*</td><td>19.0*</td></tr></table>

Table 4: Case-sensitive detokenized BLEU scores on MuST-C tst-COMMON set without external MT datasets. "External speech" denotes unlabeled speech data. \* indicates the improvements over W2V2-Transformer are statistically significant with $p < 0 . 0 1$ . The highest BLEU scores are marked in bold for all methods in each column.

<table><tr><td>Method</td><td>Training Data Speech ASR MT √</td><td>BLEU</td></tr><tr><td>MultiSLT†</td><td>√ 一</td><td>6.8</td></tr><tr><td>Chimera†</td><td>√ √ √</td><td>13.5</td></tr><tr><td>DCMA†</td><td>√ √ √</td><td>24.0</td></tr><tr><td>W2V2-Transformer + SimZeroCR</td><td>√ √ √ √</td><td>√ 0.5 √ 25.1</td></tr></table>

Table 5: Our method achieves superior performance over the existing methods on the MuST-C en de benchmark. denotes the numbers are reported from Wang et al. (2022), others are based on our runs.

Comparison with Existing Methods We summarize the recent results of several existing works on MuST-C en de benchmark in Table 5. The existing methods vary from different aspects, including language-specific encoders-decoders architecture (MultiSLT) (Escolano et al., 2021), continuous cross-modal alignment (Chimera) (Han et al., 2021), and discrete cross-modal alignment (DCMA) (Wang et al., 2022). SimZeroCR achieves an improvement of 24.6 BLEU score over W2V2- Transformer and outperforms the current SOTA method DCMA<sup>3</sup> that incorporates shared memory and vector quantization modules.

## 5 Experiments on More Languages

## 5.1 Regular End-to-End Speech Translation

We consider two experimental setups: without external MT data and with external MT data. The detailed information on the baseline methods is summarized in Appendix D, and the BLEU scores of the baseline methods are reported from the corresponding papers. The choice for hyperparameters and the corresponding model performance in each training step of our approaches are summarized in Tables 13, 14, 15, and 16.

When there is no external MT data (Table 4), SimRegCR− gains an average improvement of 2.6 BLEU scores over the W2V2-Transformer baseline and can achieve comparable performance to the current SOTA method CRESS. It is also worth mentioning that SimRegCR gains an average improvement of 3.1 BLEU scores over the W2V2-Transformer baseline and achieves an average improvement of 0.6 BLEU scores over CRESS that incorporates cross-modal regularization, scheduled sampling, token-level adaptive training, and a stronger acoustic feature extractor, which clearly shows the effectiveness of our methods. When external MT data is included (Table 6), SimRegCR− and SimRegCR gain average improvement of 1.7 and 2.2 BLEU scores over the W2V2-Transformer baseline respectively, and SimRegCR achieves an average improvement of 0.2 BLEU scores over CRESS, which implies that we could easily achieve SOTA performance for E2E ST task by leveraging simple intra-modal consistency regularization.

## 5.2 Zero-shot End-to-End Speech Translation

The experimental results with external MT data are summarized in Table 7. For fair comparisons, we keep our experimental settings consistent with Wang et al. (2022) to use WMT14 dataset for en de/es/fr/ru as the external MT data<sup>4</sup>. During inference, we use beam search decoding with a beam size of 5 with length penalty 1.0. The detailed information on the baseline methods is summarized in Appendix E, and the corresponding BLEU scores are reported from Wang et al. (2022). The choice for hyperparameters and the corresponding model performance in each training step of our approach are summarized in Table 17.

<table><tr><td rowspan="2">Method</td><td rowspan="2">External Speech</td><td colspan="8">BLEU</td></tr><tr><td>de</td><td>es</td><td>fr</td><td>it</td><td>nl</td><td>pt</td><td>ro</td><td>ru</td></tr><tr><td>MTL (Tang et al., 2021b)</td><td>-</td><td>23.9</td><td>28.6</td><td>33.1</td><td>一</td><td></td><td>一</td><td>一</td><td></td></tr><tr><td>JT-S-MT (Tang et al., 2021a)</td><td></td><td>26.8</td><td>31.0</td><td>37.4</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Chimera (Han et al., 2021)</td><td>V</td><td>27.1</td><td>30.6</td><td>35.6</td><td>25.0</td><td>29.2</td><td>30.2</td><td>24.0</td><td>17.4</td></tr><tr><td>XSTNet (Ye et al., 2021)</td><td>V</td><td>27.1</td><td>30.8</td><td>38.0</td><td>26.4</td><td>31.2</td><td>32.4</td><td>25.7</td><td>18.5</td></tr><tr><td>STEMM (Fang et al., 2022)</td><td>√</td><td>28.7</td><td>31.0</td><td>37.4</td><td>25.8</td><td>30.5</td><td>31.7</td><td>24.5</td><td>17.8</td></tr><tr><td>ConST (Ye et al., 2022)</td><td>V</td><td>28.3</td><td>32.0</td><td>38.3</td><td>27.2</td><td>31.7</td><td>33.1</td><td>25.6</td><td>18.9</td></tr><tr><td>SpeechUT (Zhang et al., 2022)†</td><td>V</td><td>30.1</td><td>33.6</td><td>41.4</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WACO (Ouyang et al., 2023)</td><td>V</td><td>28.1</td><td>32.0</td><td>38.1</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>M³ST (Cheng et al., 2023)</td><td>V</td><td>29.3</td><td>32.4</td><td>38.5</td><td>27.5</td><td>32.5</td><td>33.4</td><td>25.9</td><td>19.3</td></tr><tr><td>FCCLm (Zhang et al., 2023)</td><td>V</td><td>29.0</td><td>31.9</td><td>38.3</td><td>27.3</td><td>31.6</td><td>32.7</td><td>26.8</td><td>19.7</td></tr><tr><td>CMOT (Żhou et al., 2023)</td><td>V</td><td>29.0</td><td>32.8</td><td>39.5</td><td>27.5</td><td>32.1</td><td>33.5</td><td>26.0</td><td>19.2</td></tr><tr><td>CRESS (Fang and Feng, 2023b)</td><td>√</td><td>29.4</td><td>33.2</td><td>40.1</td><td>27.6</td><td>32.3</td><td>33.6</td><td>26.4</td><td>19.7</td></tr><tr><td>W2V2-Transformer</td><td>√</td><td>27.3</td><td>31.7</td><td>38.0</td><td>26.3</td><td>29.8</td><td>31.7</td><td>23.4</td><td>18.2</td></tr><tr><td>+ SimRegCR</td><td>V</td><td>29.0</td><td>33.0</td><td>39.4</td><td>27.3</td><td>32.2</td><td>33.5</td><td>26.0</td><td>19.4</td></tr><tr><td>+ SimRegCR</td><td>√</td><td>29.2*</td><td>33.0*</td><td>40.0*</td><td>28.2*</td><td>32.7*</td><td>34.2*</td><td>26.7*</td><td>20.1*</td></tr></table>

Table 6: Case-sensitive detokenized BLEU scores on MuST-C tst-COMMON set with external MT datasets. "External speech" denotes unlabeled speech data. is a speech-unit-text pretraining model whose training costs are much higher than ours. \* indicates the improvements over W2V2-Transformer are statistically significant with $p < 0 . 0 1$ . The highest BLEU scores are marked in bold for all methods in each column.

<table><tr><td>Method</td><td colspan="4">BLEU</td></tr><tr><td>MultiSLT</td><td>de 6.8</td><td>es</td><td>fr</td><td>ru</td></tr><tr><td>Chimera</td><td>13.5</td><td>6.8 15.3</td><td>10.9 22.2</td><td>- 8.3</td></tr><tr><td>DCMA W2V2-Transformer</td><td>24.0 0.5</td><td>26.2 0.4</td><td>33.1 0.4</td><td>16.0</td></tr><tr><td>+ SimZeroCR</td><td>25.1</td><td>27.0</td><td>34.6</td><td>0.1 15.6</td></tr></table>

Table 7: Case-sensitive detokenized BLEU scores on MuST-C tst-COMMON set with external MT datasets in zero-shot E2E ST setting. The highest BLEU scores are marked in bold for all methods in each column.

Despite the language tag is properly set during inference, W2V2-Transformer is still not capable of translating into specific language and only generating English text. We can see that SimZeroCR gains an average improvement of 25.2 BLEU scores over the W2V2-Transformer baseline and achieves an average improvement of 0.8 BLEU scores over the current SOTA method DCMA that incorporates shared memory and vector quantization modules, clearly showing the effectiveness of our method.

<table><tr><td>Method</td><td colspan="3">BLEU</td></tr><tr><td>Cascaded System Ye et al. (2021)</td><td>de 25.2</td><td>fr 34.9</td><td>ru 17.0</td></tr><tr><td>Wang et al. (2022) Fang et al. (2022)</td><td>26.7 27.5</td><td>一 一</td><td>一 -</td></tr><tr><td>Zero-Shot End-to-End Model W2V2-Transformer + SimZeroCR</td><td>0.5 25.1</td><td>0.4 34.6</td><td>0.1 15.6</td></tr></table>

Table 8: Case-sensitive detokenized BLEU scores on MuST-C tst-COMMON set.

We then compare our approach with several strong cascaded systems in Table 8. The cascaded system transforms the speech into the source language text and then translates the transcription into the target language. We can see that our zero-shot approach achieves comparable or slightly worse performance to those cascaded systems which however suffer from high inference latency.

## 6 Related Work

E2E ST is a cross-modal task, and one major challenge is direct ST data scarcity. To address such problem, people usually adopt MT data by leveraging the techniques such as pretraining (Bansal et al., 2019; Alinejad and Sarkar, 2020; Le et al., 2021; Tang et al., 2022), multi-task learning (Le et al., 2020; Dong et al., 2021; Indurthi et al., 2021), knowledge distillation (Liu et al., 2019; Gaido et al., 2020; Inaguma et al., 2021), and data augmentation (Lam et al., 2022; Fang and Feng, 2023a). Due to the representation discrepancy between speech and text modalities, people also utilize cross-modal alignment (Han et al., 2021; Fang et al., 2022; Ye et al., 2022; Ouyang et al., 2023) to fully exploit MT data. Specifically, Wang et al. (2022) employ a shared discrete vocabulary space to accommodate both modalities of speech and text and achieve SOTA performance in the zero-shot setting. We show that the zero-shot E2E ST performance could be boosted by leveraging simple cross-modal consistency regularization. Fang and Feng (2023b) propose the cross-modal regularization with scheduled sampling method to bridge the modality gap and achieve the SOTA performance in the regular setting. We find that the regularization is more crucial than modality adaption, which is in line with Han et al. (2023), and achieve the SOTA performance in the regular setting by leveraging simple intra-modal consistency regularization.

## 7 Conclusion

In this paper, we propose two simple but effective consistency regularization based strategies for learning E2E ST models. We analyze the regularization effect of SimRegCR on the regular E2E ST performance and show that SimZeroCR could effectively close the modality gap. Experiments on the MuST-C benchmark demonstrate the capabilities of our approaches to improve translation performance in both regular and zero-shot settings. Given the universality and simplicity of SimRegCR and SimZeroCR, we believe they can serve as strong baselines for future E2E ST research. For future work, we will explore the effectiveness of consistency regularization on more speech related tasks, such as speech-to-speech translation, speech language modeling, etc.

## Limitations

While our approach achieves promising performance by leveraging simple consistency regularization, it still has some limitations: 1) The performance of our approach still lags behind SpeechUT, although the training cost of our approach is much lower. 2) We mainly focus on evaluating our approach on the MuST-C benchmark in this paper. Future research could consider more speech translation benchmarks with more diverse languages, larger ST datasets, and larger models.

## References

Ashkan Alinejad and Anoop Sarkar. 2020. Effectively pretraining a speech translation decoder with machine translation data. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8014–8020, Online. Association for Computational Linguistics.

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. wav2vec 2.0: A framework for self-supervised learning of speech representations. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Sameer Bansal, Herman Kamper, Karen Livescu, Adam Lopez, and Sharon Goldwater. 2019. Pre-training on high-resource speech recognition improves lowresource speech-to-text translation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 58–68, Minneapolis, Minnesota. Association for Computational Linguistics.

Alexandre Berard, Olivier Pietquin, Christophe Servan, and Laurent Besacier. 2016. Listen and translate: A proof of concept for end-to-end speech-to-text translation. CoRR, abs/1612.01744.

Ondˇrej Bojar, Christian Buck, Chris Callison-Burch, Christian Federmann, Barry Haddow, Philipp Koehn, Christof Monz, Matt Post, Radu Soricut, and Lucia Specia. 2013. Findings of the 2013 Workshop on Statistical Machine Translation. In Proceedings of the Eighth Workshop on Statistical Machine Translation, pages 1–44, Sofia, Bulgaria. Association for Computational Linguistics.

Ondˇrej Bojar, Christian Buck, Christian Federmann, Barry Haddow, Philipp Koehn, Johannes Leveling, Christof Monz, Pavel Pecina, Matt Post, Herve Saint-Amand, Radu Soricut, Lucia Specia, and Aleš Tamchyna. 2014. Findings of the 2014 workshop on statistical machine translation. In Proceedings of the Ninth Workshop on Statistical Machine Translation, pages 12–58, Baltimore, Maryland, USA. Association for Computational Linguistics.

Ondˇrej Bojar, Rajen Chatterjee, Christian Federmann, Yvette Graham, Barry Haddow, Matthias Huck, Antonio Jimeno Yepes, Philipp Koehn, Varvara Logacheva, Christof Monz, Matteo Negri, Aurélie Névéol, Mariana Neves, Martin Popel, Matt Post, Raphael Rubino, Carolina Scarton, Lucia Specia, Marco Turchi, Karin Verspoor, and Marcos Zampieri. 2016. Findings of the 2016 conference on machine translation. In Proceedings of the First Conference on Machine Translation: Volume 2, Shared Task Papers, pages 131–198, Berlin, Germany. Association for Computational Linguistics.

Guandan Chen, Kai Fan, Kaibo Zhang, Boxing Chen, and Zhongqiang Huang. 2021. Manifold adversarial

augmentation for neural machine translation. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 3184–3189, Online. Association for Computational Linguistics.

Xuxin Cheng, Qianqian Dong, Fengpeng Yue, Tom Ko, Mingxuan Wang, and Yuexian Zou. 2023. M3st: Mix at three levels for speech translation. In ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5.

Mattia A. Di Gangi, Roldano Cattoni, Luisa Bentivogli, Matteo Negri, and Marco Turchi. 2019. MuST-C: a Multilingual Speech Translation Corpus. In Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2012–2017, Minneapolis, Minnesota. Association for Computational Linguistics.

Qianqian Dong, Rong Ye, Mingxuan Wang, Hao Zhou, Shuang Xu, Bo Xu, and Lei Li. 2021. Listen, understand and translate: Triple supervision decouples end-to-end speech-to-text translation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 12749–12759.

Long Duong, Antonios Anastasopoulos, David Chiang, Steven Bird, and Trevor Cohn. 2016. An attentional model for speech translation without transcription. In Proceedings ofthe 2016 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 949–959, San Diego, California. Association for Computational Linguistics.

Carlos Escolano, Marta R. Costa-jussà, José A. R. Fonollosa, and Carlos Segura. 2021. Enabling zeroshot multilingual spoken language translation with language-specific encoders and decoders. In 2021 IEEEAutomatic Speech Recognition and Understanding Workshop (ASRU), pages 694–701.

Qingkai Fang and Yang Feng. 2023a. Back translation for speech-to-text translation without transcripts. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4567–4587, Toronto, Canada. Association for Computational Linguistics.

Qingkai Fang and Yang Feng. 2023b. Understanding and bridging the modality gap for speech translation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15864–15881, Toronto, Canada. Association for Computational Linguistics.

Qingkai Fang, Rong Ye, Lei Li, Yang Feng, and Mingxuan Wang. 2022. STEMM: Self-learning with speech-text manifold mixup for speech translation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7050–7062, Dublin, Ireland. Association for Computational Linguistics.

Marco Gaido, Mattia A. Di Gangi, Matteo Negri, and Marco Turchi. 2020. End-to-end speech-translation with knowledge distillation: FBK@IWSLT2020. In Proceedings of the 17th International Conference on Spoken Language Translation, pages 80–88, Online. Association for Computational Linguistics.

Pengzhi Gao, Zhongjun He, Hua Wu, and Haifeng Wang. 2022. Bi-SimCut: A simple strategy for boosting neural machine translation. In Proceedings of the 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 3938–3948, Seattle, United States. Association for Computational Linguistics.

Pengzhi Gao, Liwen Zhang, Zhongjun He, Hua Wu, and Haifeng Wang. 2023. Improving zero-shot multilingual neural machine translation by leveraging crosslingual consistency regularization. In Findings of the Associationfor Computational Linguistics: ACL 2023, pages 12103–12119, Toronto, Canada. Association for Computational Linguistics.

Chi Han, Mingxuan Wang, Heng Ji, and Lei Li. 2021. Learning shared semantic space for speech-to-text translation. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 2214–2225, Online. Association for Computational Linguistics.

Yuchen Han, Chen Xu, Tong Xiao, and Jingbo Zhu. 2023. Modality adaption or regularization? a case study on end-to-end speech translation. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 1340–1348, Toronto, Canada. Association for Computational Linguistics.

Geoffrey E Hinton and Sam Roweis. 2002. Stochastic neighbor embedding. In Advances in Neural Information Processing Systems, volume 15. MIT Press.

Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. 2021. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM Trans. Audio, Speech and Lang. Proc., 29:3451–3460.

Hirofumi Inaguma, Tatsuya Kawahara, and Shinji Watanabe. 2021. Source and target bidirectional knowledge distillation for end-to-end speech translation. In Proceedings of the 2021 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 1872–1881, Online. Association for Computational Linguistics.

Sathish Indurthi, Mohd Abbas Zaidi, Nikhil Kumar Lakumarapu, Beomseok Lee, Hyojung Han, Seokchan Ahn, Sangha Kim, Chanwoo Kim, and Inchul Hwang. 2021. Task aware multi-task learning for speech to text tasks. In ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7723–7727.

Taku Kudo and John Richardson. 2018. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 66–71, Brussels, Belgium. Association for Computational Linguistics.

Tsz Kin Lam, Shigehiko Schamoni, and Stefan Riezler. 2022. Sample, translate, recombine: Leveraging audio alignments for data augmentation in end-toend speech translation. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 245– 254, Dublin, Ireland. Association for Computational Linguistics.

Hang Le, Juan Pino, Changhan Wang, Jiatao Gu, Didier Schwab, and Laurent Besacier. 2020. Dual-decoder transformer for joint automatic speech recognition and multilingual speech translation. In Proceedings of the 28th International Conference on Computational Linguistics, pages 3520–3533, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Hang Le, Juan Pino, Changhan Wang, Jiatao Gu, Didier Schwab, and Laurent Besacier. 2021. Lightweight adapter tuning for multilingual speech translation. In Proceedings of the 59th Annual Meeting of the Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 817–824, Online. Association for Computational Linguistics.

Xiaobo Liang, Lijun Wu, Juntao Li, Yue Wang, Qi Meng, Tao Qin, Wei Chen, Min Zhang, and Tie-Yan Liu. 2021. R-drop: Regularized dropout for neural networks. In Advances in Neural Information Processing Systems, volume 34, pages 10890–10905. Curran Associates, Inc.

Yuchen Liu, Hao Xiong, Jiajun Zhang, Zhongjun He, Hua Wu, Haifeng Wang, and Chengqing Zong. 2019. End-to-End Speech Translation with Knowledge Distillation. In Proc. Interspeech 2019, pages 1128– 1132.

Siqi Ouyang, Rong Ye, and Lei Li. 2023. WACO: Wordaligned contrastive learning for speech translation. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3891–3907, Toronto, Canada. Association for Computational Linguistics.

Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. 2015. Librispeech: An ASR corpus based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2015, South Brisbane, Queensland, Australia, April 19-24, 2015, pages 5206–5210. IEEE.

Sara Papi, Marco Gaido, Matteo Negri, and Marco Turchi. 2021. Speechformer: Reducing information

loss in direct speech translation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 1698–1706, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Matt Post. 2018. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 186– 191, Brussels, Belgium. Association for Computational Linguistics.

Motoki Sato, Jun Suzuki, and Shun Kiyono. 2019. Effective adversarial regularization for neural machine translation. In Proceedings ofthe 57th Annual Meeting of the Association for Computational Linguistics, pages 204–210, Florence, Italy. Association for Computational Linguistics.

Matthias Sperber, Graham Neubig, Jan Niehues, and Alex Waibel. 2017. Neural lattice-to-sequence models for uncertain inputs. In Proceedings ofthe 2017 Conference on Empirical Methods in Natural Language Processing, pages 1380–1389, Copenhagen, Denmark. Association for Computational Linguistics.

Matthias Sperber, Graham Neubig, Ngoc-Quan Pham, and Alex Waibel. 2019. Self-attentional models for lattice inputs. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1185–1197, Florence, Italy. Association for Computational Linguistics.

Yun Tang, Hongyu Gong, Ning Dong, Changhan Wang, Wei-Ning Hsu, Jiatao Gu, Alexei Baevski, Xian Li, Abdelrahman Mohamed, Michael Auli, and Juan Pino. 2022. Unified speech-text pre-training for speech translation and recognition. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1488–1499, Dublin, Ireland. Association for Computational Linguistics.

Yun Tang, Juan Pino, Xian Li, Changhan Wang, and Dmitriy Genzel. 2021a. Improving speech translation by understanding and learning from the auxiliary text translation task. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4252–4261, Online. Association for Computational Linguistics.

Yun Tang, Juan Pino, Changhan Wang, Xutai Ma, and Dmitriy Genzel. 2021b. A general multi-task learning framework to leverage text data for speech to text tasks. In ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6209–6213.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural

Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008.

Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, and Juan Pino. 2020a. Fairseq S2T: Fast speech-to-text modeling with fairseq. In Proceedings ofthe 1st Conference ofthe Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing: System Demonstrations, pages 33–39, Suzhou, China. Association for Computational Linguistics.

Chen Wang, Yuchen Liu, Boxing Chen, Jiajun Zhang, Wei Luo, Zhongqiang Huang, and Chengqing Zong. 2022. Discrete cross-modal alignment enables zeroshot speech translation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5291–5302, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Chengyi Wang, Yu Wu, Shujie Liu, Ming Zhou, and Zhenglu Yang. 2020b. Curriculum pre-training for end-to-end speech translation. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 3728–3738, Online. Association for Computational Linguistics.

Chen Xu, Bojie Hu, Yanyang Li, Yuhao Zhang, Shen Huang, Qi Ju, Tong Xiao, and Jingbo Zhu. 2021. Stacked acoustic-and-textual encoding: Integrating the pre-trained models into speech translation encoders. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2619–2630, Online. Association for Computational Linguistics.

Rong Ye, Mingxuan Wang, and Lei Li. 2021. End-to-End Speech Translation via Cross-Modal Progressive Training. In Proc. Interspeech 2021, pages 2267– 2271.

Rong Ye, Mingxuan Wang, and Lei Li. 2022. Crossmodal contrastive learning for speech translation. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5099–5113, Seattle, United States. Association for Computational Linguistics.

Biao Zhang, Philip Williams, Ivan Titov, and Rico Sennrich. 2020. Improving massively multilingual neural machine translation and zero-shot translation. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 1628– 1639, Online. Association for Computational Linguistics.

Hao Zhang, Nianwen Si, Yaqi Chen, Wenlin Zhang, Xukui Yang, Dan Qu, and Wei-Qiang Zhang. 2023.

Improving speech translation by cross-modal multigrained contrastive learning. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:1075–1086.

Pei Zhang, Niyu Ge, Boxing Chen, and Kai Fan. 2019. Lattice transformer for speech translation. In Proceedings of the 57th Annual Meeting of the Associationfor Computational Linguistics, pages 6475– 6484, Florence, Italy. Association for Computational Linguistics.

Ziqiang Zhang, Long Zhou, Junyi Ao, Shujie Liu, Lirong Dai, Jinyu Li, and Furu Wei. 2022. SpeechUT: Bridging speech and text with hiddenunit for encoder-decoder based speech-text pretraining. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 1663–1676, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Yan Zhou, Qingkai Fang, and Yang Feng. 2023. CMOT: Cross-modal mixup via optimal transport for speech translation. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7873–7887, Toronto, Canada. Association for Computational Linguistics.

## Appendix

## A Statistics of All Datasets

<table><tr><td>en→</td><td colspan="2">MuST-C hours #sents</td><td colspan="2">External MT</td></tr><tr><td>de</td><td>408</td><td>234K</td><td>name WMT16</td><td>#sents 4.6M</td></tr><tr><td>es</td><td>504</td><td>270K</td><td>WMT13</td><td>15.2M</td></tr><tr><td>fr</td><td>492</td><td>292K</td><td>WMT14</td><td>40.8M</td></tr><tr><td>it</td><td>465</td><td>258K</td><td>OPUS100</td><td>1.0M</td></tr><tr><td>nl</td><td>442</td><td>253K</td><td>OPUS100</td><td>1.0M</td></tr><tr><td></td><td>385</td><td>211K</td><td>OPUS100</td><td></td></tr><tr><td>pt</td><td></td><td></td><td></td><td>1.0M</td></tr><tr><td>ro</td><td>432</td><td>240K</td><td>WMT16</td><td>0.6M</td></tr><tr><td>ru</td><td>489</td><td>270K</td><td>WMT16</td><td>2.5M</td></tr></table>

Table 9: Statistics of all datasets. #sents refers to the number of parallel sentence pairs.

B The Choice for Hyperparameters in Tables 1 and 3

C Experimental Results on More Languages

## D Regular E2E ST Methods

We compare our approach with the following methods on the MuST-C benchmark:

<table><tr><td>ID</td><td>α</td><td>β ID</td><td></td><td>α</td><td>β</td></tr><tr><td>1</td><td>一</td><td>1</td><td>2</td><td>5.0</td><td></td></tr><tr><td>3</td><td>一</td><td></td><td>4</td><td>5.0</td><td></td></tr><tr><td>5</td><td>1</td><td>1</td><td>6</td><td>5.0</td><td>一</td></tr><tr><td>7</td><td>4.0</td><td></td><td>8</td><td>1</td><td></td></tr><tr><td>9</td><td>一</td><td></td><td>10</td><td>一</td><td>5.0</td></tr><tr><td>11)</td><td>3.0</td><td></td><td>12</td><td>3.0</td><td>5.0</td></tr><tr><td>13</td><td>一</td><td></td><td>14</td><td>0.5</td><td>一</td></tr><tr><td>15</td><td>一</td><td>一</td><td>16</td><td>1.0</td><td></td></tr><tr><td>17</td><td></td><td>1</td><td>18</td><td></td><td>一</td></tr><tr><td>19</td><td>3.0</td><td>1</td><td></td><td>3.0</td><td></td></tr></table>

Table 10: The choice for hyperparameters in Table 1.
<table><tr><td>ID</td><td>α</td><td>β</td><td>ID</td><td>α</td><td>β</td></tr><tr><td>①</td><td>–</td><td>–</td><td>2</td><td>0.5</td><td>一</td></tr><tr><td>3</td><td>一</td><td>I</td><td>4</td><td>1.0</td><td></td></tr><tr><td>5</td><td></td><td>–</td><td>6</td><td></td><td>45.0</td></tr><tr><td>7</td><td>2.0</td><td>–</td><td>8</td><td>2.0</td><td>120.0</td></tr></table>

Table 11: The choice for hyperparameters in Table 3.

• Fairseq ST (Wang et al., 2020a): Fairseq ST is a fairseq extension<sup>5</sup> for speech-to-text modeling tasks such as speech translation, which includes end-to-end workflows and SOTA models with scalability and extensibility design.

• Dual Decoder (Le et al., 2020): This paper introduces a dual-decoder Transformer architecture for synchronous speech recognition and multilingual speech translation.

• Speechformer (Papi et al., 2021): This paper introduces a Transformer-based ST model that is able to encode the whole raw audio features without any sub-optimal initial sub-sampling.

• SATE (Xu et al., 2021): This paper proposes a stacked acoustic-and-textual encoding method, which is straightforward to incorporate the pretrained models into ST.

• BiKD (Inaguma et al., 2021): To fully leverage knowledge in both source and target language directions for bilingual E2E ST models, this paper proposes bidirectional sequence-level knowledge distillation, in which both forward sequence-level knowledge distillation from a source-to-target

NMT model and backward sequence-level knowledge distillation from a target-to-source NMT model are combined.

• XSTNet (Ye et al., 2021): This paper proposes cross speech-text network, an extremely concise model that can accept bi-modal inputs and jointly train ST, ASR, and MT tasks.

• MTL (Tang et al., 2021b): This paper proposes a general multi-task learning framework to leverage text data for ASR and ST tasks.

• JT-S-MT (Tang et al., 2021a): This paper proposes three techniques to increase knowledge transfer from the MT task to the ST task, which include parameter sharing and initialization strategy to improve the information sharing between tasks, cross-attentive regularization and online knowledge distillation to encourage the ST system to learn more from the auxiliary MT task and then generate similar model representations from different modalities.

• STEMM (Fang et al., 2022): This paper proposes a speech-text manifold mixup method to mix up the speech representation sequences and word embedding sequences.

• ConST (Ye et al., 2022): This paper proposes a simple yet effective contrastive learning framework bridging the speech-text representation gap and facilitating the ST with limited data.

• SpeechUT (Zhang et al., 2022): This paper proposes a unified-modal speech-unit-text pretraining model, which bridges the modality gap between speech and text representations with hidden units.

• WACO (Ouyang et al., 2023): This paper proposes a simple and effective method for extremely low-resource speech-to-text translation, where the key idea is bridging word-level representations for both speech and text modalities via contrastive learning.

• M<sup>3</sup>ST (Cheng et al., 2023): This paper proposes a method to mix the training corpus at three levels, including word level, sentence level and frame level.

• FCCL<sup>m</sup> (Zhang et al., 2023): This paper proposes a cross-modal multi-grained contrast learning method for explicit knowledge transfer from the MT to the ST model.

<table><tr><td>ID</td><td>Training Stage</td><td>Loss Function</td><td>de</td><td>es</td><td> $\mathtt { E r }$ </td><td>it</td></tr><tr><td rowspan="4">① ② ③</td><td>MT train from scratch</td><td> $\overline { { \mathcal { L } _ { c e } ^ { m t } } }$ </td><td>29.33</td><td>34.61</td><td>41.47</td><td>31.25</td></tr><tr><td>MT &amp; ST finetune on  $\textcircled{1}$ </td><td> $\mathcal { L } _ { c e } ^ { m t } + \mathcal { L } _ { c e } ^ { s t } + \beta \mathcal { L } _ { c r o s s } ^ { m t - s t }$ </td><td>26.87</td><td>31.05</td><td>37.41</td><td>26.66</td></tr><tr><td></td><td></td><td>27.35</td><td>31.53</td><td>38.10</td><td>27.24</td></tr><tr><td>ST finetune on ①</td><td> $\mathcal { L } _ { c e } ^ { s t } + \alpha \mathcal { L } _ { i n t r a } ^ { s t }$ </td><td></td><td></td><td></td><td></td></tr></table>

Table 12: Case-sensitive detokenized BLEU scores on the MuST-C tst-COMMON set. The MT training is performed on the MuST-C dataset. 1 denotes the MT performance. 2 and 3 denote the ST performance.

• CMOT (Zhou et al., 2023): This paper proposes cross-modal mixup via optimal transport to adaptively find the alignment between speech and text sequences, and to mix up the sequences of different modalities at the token level.

• CRESS (Fang and Feng, 2023b): This paper proposes a simple yet effective method to regularize the model predictions of ST and MT, whose target-side contexts contain both ground truth words and self-generated words with scheduled sampling.

## E Zero-shot E2E ST Methods

We compare our approach with the following methods on the MuST-C benchmark:

• MultiSLT (Escolano et al., 2021): This paper extends the multilingual NMT system to perform spoken language translation and zero-shot multilingual spoken language translation by coupling language-specific encoder-decoders, even from monolingual ASR data only.

• Chimera (Han et al., 2021): This paper proposes a model capable of learning a text-speech shared semantic memory network for bridging the gap between speech and text representations.

• DCMA (Wang et al., 2022): This paper proposes an alignment method to enable zero-shot ST, where the key part is to discretize the continuous vectors to a finite set of virtual tokens and use ASR data to map the corresponding speech and text to the same virtual token in the shared codebook.

## F The Choice for Hyperparameters in Section 5

<table><tr><td>Training Stage</td><td></td><td>de</td><td>es</td><td>fr</td><td>it</td><td>nl</td><td>pt</td><td>ro</td><td>ru</td></tr><tr><td>MT pretrain</td><td>Baseline</td><td>29.33</td><td>34.61</td><td>41.47</td><td>31.25</td><td>34.41</td><td>35.80</td><td>28.13</td><td>19.40</td></tr><tr><td rowspan="2">ST finetune</td><td>Baseline</td><td>24.38</td><td>29.92</td><td>34.73</td><td>25.13</td><td>29.29</td><td>30.32</td><td>23.39</td><td>16.45</td></tr><tr><td>BLEU α</td><td>27.35 5</td><td>31.53 4</td><td>38.10 4</td><td>27.24 5</td><td>32.00 4</td><td>33.30 5</td><td>25.89 4</td><td>18.83 4</td></tr></table>

Table 13: The choice for hyperparameters and the corresponding MT & ST performance in the training steps of SimRegCR− without external MT datasets.

<table><tr><td>Training Stage</td><td></td><td>de</td><td>es</td><td>fr</td><td>it</td><td>nl</td><td>pt</td><td>ro</td><td>ru</td></tr><tr><td rowspan="2">MT pretrain</td><td>BLEU</td><td>32.76</td><td>37.10</td><td>45.68</td><td>33.31</td><td>37.89</td><td>39.12</td><td>31.60</td><td>21.60</td></tr><tr><td>α</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td></tr><tr><td rowspan="2">ST finetune</td><td>BLEU</td><td>27.91</td><td>32.12</td><td>39.04</td><td>27.69</td><td>32.39</td><td>33.96</td><td>26.30</td><td>19.02</td></tr><tr><td>α</td><td>4</td><td>4</td><td>5</td><td>4</td><td>4</td><td>4</td><td>4</td><td>3</td></tr></table>

Table 14: The choice for hyperparameters and the corresponding MT & ST performance in the training steps of SimRegCR without external MT datasets.

<table><tr><td>Training Stage</td><td></td><td>de</td><td>es fr</td><td>it</td><td></td><td>n1</td><td> $\mathtt { p t }$ </td><td>ro</td><td>ru</td></tr><tr><td>MT pretrain†</td><td>Baseline</td><td>29.61</td><td>31.98</td><td>40.59</td><td>26.30</td><td>30.58</td><td>31.83</td><td>23.48</td><td>18.65</td></tr><tr><td>MT finetune</td><td>Baseline</td><td>33.59</td><td>37.78</td><td>45.93</td><td>32.74</td><td>37.06</td><td>38.81</td><td>29.05</td><td>22.11</td></tr><tr><td rowspan="2">ST finetune</td><td>Baseline</td><td>27.33</td><td>31.70</td><td>38.04</td><td>26.29</td><td>29.77</td><td>31.73</td><td>23.43</td><td>18.23</td></tr><tr><td>BLEU α</td><td>28.96 3</td><td>33.04 3</td><td>39.37 2</td><td>27.30 3</td><td>32.22 3</td><td>33.51 4</td><td>26.00 4</td><td>19.41 3</td></tr></table>

Table 15: The choice for hyperparameters and the corresponding MT & ST performance in the training steps of SimRegCR− with external MT datasets. denotes the training procedure is performed on the external MT dataset.

<table><tr><td>Training Stage</td><td></td><td> $\mathsf { d e }$  es</td><td> $\mathtt { E r }$ </td><td></td><td>it</td><td> $\mathrm { n 1 }$ </td><td>pt</td><td>ro</td><td>ru</td></tr><tr><td>MT pretrain†</td><td>BLEU α</td><td>30.02 0.5</td><td>32.10 0.25</td><td>40.62 0.125</td><td>28.24 3</td><td>33.08 3</td><td>34.02 2</td><td>24.99 2</td><td>19.28 0.5</td></tr><tr><td>MT finetune</td><td>BLEU α</td><td>34.11 1</td><td>37.97 0.25</td><td>46.95 3</td><td>33.86 5</td><td>38.67 5</td><td>40.09 3</td><td>32.23 3</td><td>22.45 3</td></tr><tr><td>ST finetune</td><td>BLEU α</td><td>29.23 3</td><td>32.97 3</td><td>39.98 3</td><td>28.16 3</td><td>32.68 3</td><td>34.24 4</td><td>26.66 3</td><td>20.09 4</td></tr></table>

Table 16: The choice for hyperparameters and the corresponding MT & ST performance in the training steps of SimRegCR with external MT datasets. denotes the training procedure is performed on the external MT dataset.

<table><tr><td>Training Stage</td><td></td><td> $\mathsf { d e }$ </td><td>es</td><td> $\mathtt { E r }$ </td><td>ru</td></tr><tr><td>MT pretrain†</td><td>Baseline</td><td>29.37</td><td>32.91</td><td>41.33</td><td>18.07</td></tr><tr><td>MT finetune</td><td>Baseline</td><td>33.78</td><td>37.53</td><td>45.99</td><td>21.67</td></tr><tr><td rowspan="2">ASR &amp; MT finetune</td><td>Baseline</td><td>0.47</td><td>0.43</td><td>0.43</td><td>0.07</td></tr><tr><td>BLEU β</td><td>25.10 30</td><td>26.99 45</td><td>34.59 20</td><td>15.56 35</td></tr></table>

Table 17: The choice for hyperparameters and the corresponding MT & ST performance in the training steps of SimZeroCR with external MT datasets. denotes the training procedure is performed on the external MT dataset.