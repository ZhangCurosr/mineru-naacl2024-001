# FPT: Feature Prompt Tuning for Few-shot Readability Assessment

Ziyang Wang<sup>1,2</sup>, Sanwoo Lee<sup>1,3</sup>, Hsiu-Yuan Huang<sup>1,3</sup> Yunfang Wu<sup>1,3</sup>∗   
<sup>1</sup>National Key Laboratory for Multimedia Information Processing, Peking University <sup>2</sup>School of Software and Microelectronics, Peking University, Beijing, China <sup>3</sup>School of Computer Science, Peking University, Beijing, China   
{wzy232303, huang.hsiuyuan}@stu.pku.edu.cn, {sanwoo, wuyf}@pku.edu.cn,

## Abstract

Prompt-based methods have achieved promising results in most few-shot text classification tasks. However, for readability assessment tasks, traditional prompt methods lack crucial linguistic knowledge, which has already been proven to be essential. Moreover, previous studies on utilizing linguistic features have shown non-robust performance in few-shot settings and may even impair model performance. To address these issues, we propose a novel prompt-based tuning framework that incorporates rich linguistic knowledge, called Feature Prompt Tuning (FPT). Specifically, we extract linguistic features from the text and embed them into trainable soft prompts. Further, we devise a new loss function to calibrate the similarity ranking order between categories. Experimental results demonstrate that our proposed method FTP not only exhibits a significant performance improvement over the prior best prompt-based tuning approaches, but also surpasses the previous leading methods that incorporate linguistic features. Also, our proposed model significantly outperforms the large language model gpt-3.5-turbo-16k in most cases. Our proposed method establishes a new architecture for prompt tuning that sheds light on how linguistic features can be easily adapted to linguistic-related tasks.

## 1 Introduction

Readability assessment (RA) is the task of evaluating the reading difficulty of a given piece of text (Vajjala, 2022). It has wide applications, such as choosing appropriate reading materials for language teaching (Collins-Thompson and Callan, 2004), supporting readers with learning disabilities (Rello et al., 2012), and ranking search results by their reading levels (Kim et al., 2012).

Early works in RA mainly focused on designing handcrafted linguistic features such as word length (in characters/syllables), sentence length, and usage of different difficulty-level words. In recent years, RA has been dominated by neural network-based architectures (Meng et al., 2021; Azpiazu and Pera, 2019). The key challenge of these methods is to learn a better text representation that can capture deep semantic features. Current research has also explored different ways of combining linguistic features with pretrained language models (PLMs), achieving state-of-the-art results on numerous RA datasets (Li et al., 2022; Lee et al., 2021). However, these studies have mainly focused on fine-tuning with a large amount of labelled data, while only a few studies have explored few-shot settings.

![](images/03803cb2b5ac2ae8f8282984a4f42eb2374d9c964b6d625f1879ac7022cbad60.jpg)  
Figure 1: Comparison of previous prompt tuning frameworks and our proposed Feature Prompt Tuning (FPT). T( ) and verbalizer( ) denote the template and verbalizer function, respectively. FPT utilizes both hard and soft tokens which are projected from the linguistic features extracted from the input x.

Prompt-based tuning, shown to be a powerful method for the classification task in the few-shot setting, makes full use of the information in PLMs by reformulating classification tasks as cloze questions. Different prompt-based tuning strategies are illustrated in Figure 1. The hard prompt tuning applies a template with [MASK] token to the original input and maps the predicted label word to the corresponding class (Han et al., 2022; Shin et al., 2020). The performance is sensitive to the quality of template, which introduces time-consuming and labor-intensive prompt design and optimization. To address this problem, researchers propose soft prompt strategies, where continuous embeddings of trainable tokens replace the hard template and are optimized by training (Liu et al., 2021; Lester et al., 2021).

Despite the success in a range of text classification tasks, existing prompt-based tuning methods still suffer from inferior performance in RA. This might be attributed to the lack of linguistic knowledge which has been demonstrated to play a crucial role in RA (Vajjala, 2022; Qiu et al., 2021; Li et al., 2022). Meanwhile, RA differs from general classification tasks in that there exists a notion of ranking order between classes. Our intuition behind the utilization of linguistic knowledge is that the learned representations of different levels should preserve the similarity relationship analogous to that of original linguistic features of different levels.

Motivated by the above insights, in this paper, we propose a novel prompt-based tuning method that incorporates rich linguistic knowledge, called Feature Prompt Tuning (FPT), as shown in the bottom of Figure 1. Specifically, our methodology begins with extracting linguistic features from the text. These extracted features are subsequently embedded into feature prompts, functioning as trainable soft prompts. Contrary to the conventional prompt tuning frameworks, our model can explicitly benefit from linguistic knowledge. Furthermore, we devise a new loss function to calibrate the similarity relationships between the embedded features across different categories. Our approach is straightforward and effective, offering wide applicability to other tasks where the importance of handcrafted features is emphasized.

To verify the effectiveness of our proposed methods, we conduct extensive experiments on three RA datasets, including one Chinese data (Li et al., 2022) and two English datasets, WeeBit (Vajjala and Meurers, 2012) and Cambridge (Xia et al., 2019). By incorporating linguistic knowledge, our proposed model FPT improves significantly over other prompt-based methods. For instance, in the 2-shot setting, FPT brings a relative performance gain of 43.9% over the traditional soft prompt method on the Chinese dataset and 5.50% on English Weebit. Moreover, compared to other feature fusion methods, FPT outperforms the previous best method Projecting Feature (PF) (Li et al., 2022) by 43.19% on Chinese data and 11.55% on English Weebit data. Also, we experiment on the Large Language Model (LLM), demonstrating the superiority of our approach on RA. We will make our code public available <sup>1</sup>. We summarize our contributions as follows:

• We propose a novel prompt-based tuning framework, Feature Prompt Tuning (FPT), which incorporates rich linguistic knowledge for RA.

• We design a new calibration loss to ensure the linguistic features retain their original similarity information during optimization.

• Our experimental results show that our method outperforms other prompt-based tuning methods and effectively leverages linguistic features, leading to better and more stable performance improvements than previous approaches.

## 2 Related Works

## 2.1 Readability Assessment

Early works have explored a wide range of linguistic features as measurements for readability. Flesch (1948) performed regression over features such as average word length in syllable; Schwarm and Ostendorf (2005) trained an SVM over features including LM perplexity and syntactic tree height; Pitler and Nenkova (2008) illustrated that discourse relations can be a good predictor of readability.

Recent works largely employ deep learning approaches for RA. Several deep architectures, including BERT (Devlin et al., 2018), HAN (Yang et al., 2016), and multi-attentive RNN were applied to achieve strong performance without feature engineering (Martinc et al., 2021; Azpiazu and Pera, 2019). However, the performance of neural models tends to fluctuate a lot across different RA datasets (Deutsch et al., 2020), suggesting that relying only on neural networks might not be a robust solution for RA. Meanwhile, later works have shown that a hybrid approach combining transformer-based encoders with linguistic features can achieve even higher performance (Lee et al., 2021; Lee and Vajjala, 2022; Li et al., 2022). Lee and Lee (2023)

applied a prompt-based learning based on seq2seq model such as T5 and BART, treating RA as a text-to-text generative task. Despite the novelty of their method, it was not included in our baselines since it is hard for this method to draw a meaningful comparison against our approach. In addition to the fundamental discrepancy in the task definition, their method focuses on optimizing hard prompts and combining multiple datasets during training, whereas our method focuses on incorporating linguistic knowledge without leveraging multiple datasets.

## 2.2 Prompt-based Tuning

Fine-tuning PLMs have shown their prevalence in various NLP tasks. PLMs, such as BERT (Devlin et al., 2018), GPT (Radford et al., 2018), XLNet (Xia et al., 2019), RoBERTa (Liu et al., 2019) and T5 (Raffel et al., 2020), have been proposed with varied self-supervised learning architectures. It has been demonstrated that larger models tend to perform better in many learning scenarios (Brown et al., 2020), which stimulated PLMs with billions of parameters to emerge.

Fine-tuning large PLMs may be prohibitive, and there exist a significant gap between pretraining tasks and downstream tasks. Prompt tuning addresses this challenge by reformulating downstream tasks as a language modeling problem and optimizing the prompt. Prompts are used to probe PLM’s intrinsic knowledge to perform a task (Min et al., 2022), and various techniques of prompting have been explored to aid PLM better: hard prompt (Shin et al., 2020; Schick and Schütze, 2021), soft prompt (Lester et al., 2021; Li and Liang, 2021), verbalizer (Cui et al., 2022) and pretrained prompt tuning (Gu et al., 2021).

The effectiveness of prompt tuning has been validated in various NLP tasks, including sentiment analysis (Wu and Shi, 2022), named entity recognition (Ma et al., 2022), relation extraction (Chen et al., 2022) and semantic parsing (Schucher et al., 2021). However, the potential of prompt tuning is less explored in RA. In this work, we focus on the effectiveness of linguistic features for modeling readability, and utilize linguistic features to guide prompt tuning.

## 3 Background

We model RA as a text classification task. Formally, a RA dataset can be denoted as $\mathcal { D } = \{ \mathcal { X } , \mathcal { Y } \}$ where $\mathcal { X }$ is the text set and $\mathcal { V }$ is the class set. Each instance $x \in \mathcal { X }$ consists of several tokens, $x = \{ w _ { 1 } , w _ { 2 } , . . . , w _ { | x | } \}$ , and is annotated with a label $y \in \mathcal { V }$ , indicating the reading difficulty.

## 3.1 Fine-tuning PLMs for RA

Given a PLM $\mathcal { M }$ for RA, fine-tuning methods first convert a text $x = ( w _ { 1 } , w _ { 2 } , . . . , w _ { | x | } )$ into an input sequence $( [ \mathrm { C L S } ] , w _ { 1 } , w _ { 2 } , . . . , w _ { | x | } , [ \mathrm { S E P } ] )$ . The PLM encodes this sequence into the hidden vectors $\boldsymbol { h } = ( h _ { [ \mathrm { C L S } ] } , h _ { 1 } , h _ { 2 } , . . . , h _ { | x | } , h _ { [ \mathrm { S E P } ] } )$

In the conventional fine-tuning, an additional classifier $F C$ is trained on top of the [CLS] embedding $h _ { \mathrm { [ C L S ] } }$ . This classifier produces a probability distribution over the class set $\mathcal { V }$ through a softmax function, which can be formulated as:

$$
P ( \cdot | x ) = \mathrm { S o f t m a x } ( F C ( h _ { [ \mathrm { C L S } ] } ) ) ,
$$

The objective of fine-tuning is to minimize the cross-entropy loss between the predicted probability distribution $P ( \cdot | x )$ and the ground-truth label $y \colon$

$$
\mathcal { L } _ { c l a s s f i c a t i o n } = - \frac { 1 } { \vert \mathcal { X } \vert } \sum _ { x \in \mathcal { X } } \log P ( y \vert x ) .
$$

## 3.2 Prompt-based Tuning

Prompt-based tuning aims to bridge the gap between pretraining tasks and downstream tasks, as illustrated in Figure 1.

Hard Prompt. It typically consists of a template $T ( \cdot )$ , which transforms the input x into a prompt input $x _ { p r o m p t } .$ , and a set of label words V that are connected to the label space through a mapping function $\Phi : V  \mathcal { V }$ , often referred to as the verbalizer. The prompt input contains at least one [MASK] token for the model to fill with label words.

Taking an example in $\mathrm { R A } , x _ { p r o m p t }$ could take the form of

$$
\begin{array} { r } { x _ { p r o m p t } = T ( x ) = \ " \mathrm { { I t } } \mathrm { { i s } } \left[ \mathrm { { M A S K } } \right] \mathrm { { t o r e a d } } \mathrm { { : } } x ^ { \ " } . } \end{array}
$$

In this case, the input embedding sequence of $x _ { p r o m p t }$ is denoted as

$$
( e ( ^ { \mathfrak { n } } \mathrm { I t ~ i s " } ) , e ( [ \mathbf { M A S K } ] ) , e ( ^ { \mathfrak { n } } \mathrm { t o ~ r e a d : ~ } ^ { \mathfrak { n } } ) , e ( x ) ) .
$$

Soft Prompt. It replaces hard tokens in the template with trainable soft tokens $[ h _ { 1 } , . . . , h _ { l } ]$ , yielding an input embedding sequence of

$$
\big ( h _ { 1 } , . . . , h _ { l } , e \big ( [ \boldsymbol { \mathrm { M A S K } } ] \big ) , e ( x ) \big ) .
$$

![](images/f87e79b90cb00ec772a6db5e0fcf33746a8717efe2a6af5dc8d62fb611c4c0df.jpg)  
Figure 2: The architecture of the proposed Feature Prompt Tuning. Column-wise ranking orders of similarity matrices are denoted with numbers.

Hybrid Prompt. It combines soft tokens with hard prompt tokens $T$ to form the input embedding sequence:

$$
( h _ { 1 } , . . . , h _ { l } , e ( T ) , e ( [ \boldsymbol { \mathrm { M A S K } } ] ) , e ( x ) ) .
$$

By feeding the input embedding sequence of $x _ { p r o m p t }$ into $\mathcal { M } .$ , the probability distribution over the class set $\mathcal { V }$ is modeled by:

$$
P _ { \cal M } ( y | x ) = P _ { \cal M } ( [ { \bf M A S K } ] = \Phi ( y ) | x _ { p r o m p t } )
$$

The learning objective of prompt-based tuning is to minimize the cross entropy loss:

$$
\mathcal { L } _ { c l a s s i f i c a t i o n } = - \frac { 1 } { | \mathcal { X } | } \sum _ { x \in \mathcal { X } } \log P _ { \mathcal { M } } ( y | x )
$$

## 4 Feature Prompt Tuning

In this section, we propose a novel method for RA with prompt-based tuning, named Feature Prompt Tuning (FPT). The architecture of our model is illustrated in Figure 2. Specifically, we extract linguistic features from the texts and embed them into soft prompts. Then, we employ a loss function to calibrate the similarity relationship between embedded features of different classes. We adopt an alternating procedure to optimize the model with respect to the classification loss and calibration loss.

## 4.1 Feature Prompt Construction

Feature Extraction Our approach for extracting linguistic features from text is consistent with previous works (Li et al., 2022; Lee et al., 2021). For English texts, the linguistic features are extracted using the lingfeat toolkit (Lee et al., 2021), which includes discourse, syntactic, lexical, and shallow features. In terms of Chinese linguistic features, we directly utilize the zhfeat toolkit (Li et al., 2022) to extract character, word, sentence, and paragraph-level features. Specific details are provided in Appendix A. For an input text x, we denote the extracted features as $f _ { x } ,$ , which is a α- dimensional vector with α representing the number of extracted features.

Feature Embedding To incorporate linguistic knowledge into prompt-based tuning, we transform linguistic feature $f _ { x }$ into l distinct vectors $\{ v _ { 1 } , . . . , v _ { l } \}$ which function as the embeddings of soft tokens, as follows:

$$
\{ v _ { 1 } , . . . , v _ { l } \} = \mathbf { M u l t i H e a d M L P } ( f _ { x } ) .
$$

Here, MultiHeadMLP is a multi-head MLP with l output heads. Each head consists of a series of fully connected layers followed by non-linear activation functions.

The purpose of using a multi-head MLP is to allow the model to map $f _ { x }$ into separate vector spaces and learn multiple aspects of the linguistic features. This enables the model to better capture the relationships between different features and their contribution to RA.

Ultimately, we formulate the input embedding

sequence of $x _ { p r o m p t }$ as follows:

$$
( v _ { 1 } , . . . , v _ { l } , e ( T ) , e ( [ \mathbf { M A S K } ] ) , e ( x ) ) .
$$

This input sequence is passed through the PLM to calculate $\mathcal { L } _ { c l a s s i f i c a t i o n }$ loss as described in Section 3.2.

## 4.2 Inter-class Similarity Calibration

We denote $\mathcal { F } = \{ F _ { c _ { 1 } } , \cdots , F _ { c _ { n } } \}$ as the collection of linguistic features for the dataset $\mathcal { D } ,$ which consists of n classes. Here, $F _ { c _ { i } } = \{ f _ { x _ { i 1 } } , \cdot \cdot \cdot , f _ { x _ { i k } } \}$ signifies the extracted features of k samples which belong to i-th class. We apply average pooling to the feature embeddings of each sample in $\mathcal { F }$ resulting in a set of embedded linguistic features, denoted as $\mathcal { F } ^ { \prime } = \{ F _ { c _ { 1 } } ^ { \prime } , \cdot \cdot \cdot , F _ { c _ { n } } ^ { \prime } \}$ . To gauge the similarity between any two classes $F _ { c _ { m } }$ and $F _ { c _ { n } } ,$ we employ a pairwise approach based on cosine similarity, expressed as:

$$
s _ { m n } = { \frac { 1 } { k ^ { 2 } } } \sum _ { i = 1 } ^ { k } \sum _ { j = 1 } ^ { k } \cos ( f _ { x _ { m i } } , f _ { x _ { n j } } )
$$

With the feature representation and similarity function in place, we can define our calibration objective. The fundamental intuition is that the distribution of extracted linguistic features should be preserved as much as possible. Namely, if the similarity between $F _ { c _ { m } }$ and $F _ { c _ { n } }$ is relatively low, the similarity between $F _ { c _ { m } } ^ { \prime }$ and $F _ { c _ { n } } ^ { \prime }$ should also be proportionately low, and vice versa. Therefore, during the training process, we devise an objective function based on a list-wise ranking loss function ListMLE (Xia et al., 2008), to maintain this initial ranking relationship.

More specifically, we compute the similarity between each pair of classes within $\mathcal { F }$ to generate the similarity matrix:

$$
M = \left[ \begin{array} { c c c c } { s _ { 1 1 } } & { s _ { 1 2 } } & { \cdots } & { s _ { 1 n } } \\ { s _ { 2 1 } } & { s _ { 2 2 } } & { \cdots } & { s _ { 2 n } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { s _ { n 1 } } & { s _ { n 2 } } & { \cdots } & { s _ { n n } } \end{array} \right]
$$

Likewise, we can derive the similarity matrix $M ^ { \prime }$ for ${ \mathcal { F } } ^ { \prime }$

We then use $\Pi = \{ \pi _ { 1 } , \pi _ { 2 } , \cdot \cdot \cdot , \pi _ { n } \}$ to denote the ranking order of the columns in M, where $\pi _ { i }$ represents the ranking order of the i-th column. We obtain $\hat { M ^ { \prime } }$ by rearranging the columns of $M ^ { \prime }$

according to Π:

$$
\hat { M } ^ { \prime } = \left[ \begin{array} { c c c c } { s _ { \pi _ { 1 1 } } ^ { \prime } } & { s _ { \pi _ { 1 2 } } ^ { \prime } } & { \cdot \cdot \cdot } & { s _ { \pi _ { 1 n } } ^ { \prime } } \\ { s _ { \pi _ { 2 1 } } ^ { \prime } } & { s _ { \pi _ { 2 2 } } ^ { \prime } } & { \cdot \cdot \cdot } & { s _ { \pi _ { 2 n } } ^ { \prime } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { s _ { \pi _ { n 1 } } ^ { \prime } } & { s _ { \pi _ { n 2 } } ^ { \prime } } & { \cdot \cdot \cdot } & { s _ { \pi _ { n n } } ^ { \prime } } \end{array} \right]
$$

Finally, we aim to minimize the following loss function for similarity calibration:

$$
L _ { c a l i b r a t i o n } = - \sum _ { k = 1 } ^ { n } l o g \prod _ { i = 1 } ^ { n } \frac { \exp ( s _ { \pi _ { i k } } ^ { \prime } ) } { \sum _ { j = i } ^ { n } \exp ( s _ { \pi _ { j k } } ^ { \prime } ) }
$$

## 4.3 Training Procedure

Training Objectives Given the dataset and the linguistic feature set ${ \mathcal { F } } _ { : }$ , we establish two training objectives. The primary objective is to minimize the classification loss $L _ { c l a s s i f i c a t i o n }$ , which is computed based on the difference between the predicted and actual class labels. The secondary objective is to calibrate the inter-class similarity of the mapped features by minimizing the loss function $L _ { c a l i b r a t i o n }$ defined in Section 4.2.

Algorithm 1 Alternating Training Procedure for   
Feature Prompt Learning   
1: Initialize model parameters M and feature em  
beddings f   
2: for each epoch do   
3: Shuffle dataset D   
4: for each batch b in D do   
5: Compute L<sub>classification</sub> for b using M   
and f   
6: Update M and f by minimizing   
$\ b { L _ { c l a s s i f } }$ ication   
7: end for   
8: Compute $L _ { c a l i b r a t i o n }$ for D using f   
9: Update $f$ by minimizing L<sub>calibration</sub>   
10: end for

Alternating Training Procedure For training both loss functions, we adopt an alternating training procedure, as encapsulated in Algorithm 1. This procedure iteratively updates the model parameters and feature embeddings by minimizing the classification loss $L _ { c l a s s i f i c a t i o n }$ and the similarity calibration loss $L _ { c a l i b r a t i o n }$ , respectively.

In each epoch, the dataset D is shuffled to ensure the model is not biased towards any particular ordering of the data. For each batch b in $D ,$ the classification loss $L _ { c l a s s i f i c a t i o n }$ is computed using the current model parameters M and feature embeddings f. The model parameters M and feature embeddings f are then updated by minimizing this loss. Subsequently, the similarity calibration loss $L _ { c a l i b r a t i o n }$ is computed using the updated feature embeddings f for the epoch, and the feature embeddings are updated by minimizing this loss . This process is repeated for each epoch. The alternating training procedure ensures that the model learns to classify the data accurately while maintaining the inter-class similarity structure of the feature space.

## 5 Experimental Setting

## 5.1 Datasets

To evaluate the effectiveness of our proposed method, we conduct experiments on one Chinese dataset and two English datasets, following the same data split as Li et al. (2022). The statistics of the datasets can be found in Table 1.

ChineseLR (Li et al., 2022) is a Chinese dataset collected from textbooks of the middle and primary schools of more than ten publishers. Following the standards specified in the Chinese Curriculum Standards for Compulsory Education, all texts are divided into five difficulty levels.

WeeBit (Vajjala and Meurers, 2012) is often considered as the benchmark data for English RA. It was initially created as an extension of the wellknown Weekly Reader corpus.

Cambridge (Xia et al., 2019) consists of reading passages from the five main suite Cambridge English Exams (KET, PET, FCE, CAE, CPE).

## 5.2 Baselines 1: Prompt-based Methods

For prompt-based methods, we compare with hard, soft, and hybrid prompts. To avoid the influence of verbalizers on experimental results, we adopt a soft verbalizer (Hambardzumyan et al., 2021) that employs a linear layer classifier across all promptbased methods.

Hard Prompt (HP): We implement four manually defined templates for prompt tuning and select a template with the best performance on the development set. As for FPT in Table 2, we report the test set performance averaged over the four templates. This setting poses a challenge to FPT, as the averaged performance of FPT should outperform the best performance of HP to demonstrate its effectiveness. Details of the templates can be found in Appendix B.

<table><tr><td>Dataset</td><td colspan="2">WeeBit</td><td colspan="2">Cambridge</td><td colspan="2">ChineseLR</td></tr><tr><td>Level</td><td>#</td><td>Avg Len</td><td>#</td><td>Avg Len</td><td>#</td><td>Avg Len</td></tr><tr><td>1</td><td>625</td><td>152</td><td>60</td><td>141</td><td>814</td><td>266</td></tr><tr><td>2</td><td>625</td><td>189</td><td>60</td><td>271</td><td>1063</td><td>679</td></tr><tr><td>3</td><td>625</td><td>295</td><td>60</td><td>617</td><td>1104</td><td>1140</td></tr><tr><td>4</td><td>625</td><td>242</td><td>60</td><td>763</td><td>762</td><td>2165</td></tr><tr><td>5</td><td>625</td><td>347</td><td>60</td><td>751</td><td>417</td><td>3299</td></tr><tr><td>All</td><td>3125</td><td>245</td><td>300</td><td>509</td><td>4160</td><td>1255</td></tr></table>

Table 1: Statistics of RA datasets. #: number of the passages. Avg Len: average tokens numbers per passage.

Soft Prompt (SP): It replaces manually defined prompts with trainable continuous prompts. We follow the same implementation as Lester et al. (2021) and use randomly sampled vocabulary to initialize the prompts.

Hybrid Prompt (HBP): It concatenates trainable continuous prompts to the wrapped input embeddings. We adopt the implementation from Gu et al. (2022).

P-tuning: A hybrid prompt method, which replaces some tokens in manually designed prompts with soft prompts and only retains task-relevant anchor words. The soft prompts are embedded with a bidirectional LSTM and a MLP (Liu et al., 2021).

## 5.3 Baselines 2: Fusion Methods

We also compare with the methods fusing linguistic features and PLMs from previous studies.

SVM: Use the single numerical output of a neural model (BERT) as a feature itself, joined with linguistic features, and then fed them into SVM (Lee et al., 2021; Deutsch et al., 2020).

FT: Standard fine-tuning method without linguistic features, where the hidden representation of [CLS] token is used for classification. This baseline validates whether the linguistic features indeed have a positive effect.

Concatenation (Con): Fine-tune with linguistic features, in which the linguistic features are directly concatenated to the hidden representation of the [CLS] token (Meng et al., 2021; Qiu et al., 2021).

PF: Fuse linguistic features with hidden representations of [CLS] through projection filtering (Li et al., 2022).

## 5.4 Implementation Details

Under the few-shot setting, we randomly sample k = 1, 2, 4, 8, 16 instances in each class from the training and development set. For each k-shot experiment, we sample 4 different training and dev sets and repeat experiments on each training set for 4 times. We select the best model checkpoint based on the performance of the development set and evaluate the models on the entire test set. As for the evaluation metric, we use accuracy in all experiments and take the mean values as the final results.

All our models and baselines are implemented with the PyTorch (Paszke et al., 2019) framework and Huggingface transformers (Wolf et al., 2020). We use BERT (Devlin et al., 2018) as our Pretrained Language Model (PLM) backbone. We use "bert-base-uncased" for English datasets and "bertbase-chinese" for the Chinese dataset. During training, we employ the AdamW optimizer (Loshchilov and Hutter, 2019) with a weight decay of 0.01 and a warm-up ratio of 0.05. We tune the model with the batch size of 8 for 30 epochs, and the learning rate is 1e-5. All experiments are conducted with four NVIDIA GeForce RTX 3090s.

## 6 Results and Analysis

<table><tr><td>k</td><td>Methods</td><td>ChineseLR</td><td>Weebit</td><td>Cambridge</td></tr><tr><td rowspan="5">1</td><td>HP</td><td>29.49(5.21)</td><td>41.83(4.72)</td><td>36.25(8.49)</td></tr><tr><td>SP</td><td>31.22(4.70)</td><td>46.61(3.63)</td><td>41.73(8.45)</td></tr><tr><td>HBP</td><td>33.51(5.19)</td><td>44.46(5.02)</td><td>42.04(9.12)</td></tr><tr><td>P-tuning</td><td>33.36(4.12)</td><td>41.23(4.11)</td><td>40.36(7.15)</td></tr><tr><td>FPT(ours)</td><td>39.63(6.38)</td><td>43.61(4.50)</td><td>44.17(7.12)</td></tr><tr><td rowspan="5">2</td><td>HP</td><td>28.38(8.14)</td><td>49.23(2.85)</td><td>46.88(9.31)</td></tr><tr><td>SP</td><td>32.14(5.54)</td><td>52.22(4.35)</td><td>49.13(8.38)</td></tr><tr><td>HBP</td><td>33.38(7.02)</td><td>52.52(2.66)</td><td>49.56(7.12)</td></tr><tr><td>P-tuning</td><td>35.12(4.20)</td><td>50.71(3.87)</td><td>48.97(8.47)</td></tr><tr><td>FPT(ours)</td><td>46.24(5.62)</td><td>55.10(4.04)</td><td>59.79(10.2)</td></tr><tr><td rowspan="5">4</td><td>HP</td><td>36.56(5.18)</td><td>53.41(4.50)</td><td>48.75(8.49)</td></tr><tr><td>SP</td><td>38.78(2.83)</td><td>54.96(3.89)</td><td>49.36(9.14)</td></tr><tr><td>HBP</td><td>39.81(2.67)</td><td>56.88(3.52)</td><td>50.13(8.77)</td></tr><tr><td>P-tuning</td><td>38.45(3.09)</td><td>54.35(3.21)</td><td>48.85(9.64)</td></tr><tr><td>FPT(ours)</td><td>48.93(3.21)</td><td>57.70(4.63)</td><td>53.54(7.21)</td></tr><tr><td rowspan="5">8</td><td>HP</td><td>41.21(4.83)</td><td>61.31(3.13)</td><td>55.42(6.86)</td></tr><tr><td>SP</td><td>42.72(2.82)</td><td>62.02(2.67)</td><td>56.75(6.89)</td></tr><tr><td>HBP</td><td>41.93(4.12)</td><td>63.37(2.02)</td><td>57.34(9.28)</td></tr><tr><td>P-tuning</td><td>42.81(4.04)</td><td>61.81(3.28)</td><td>56.90(7.23)</td></tr><tr><td>FPT(ours)</td><td>52.66(5.00)</td><td>64.92(2.75)</td><td>59.38(6.58)</td></tr><tr><td rowspan="5">16</td><td>HP</td><td>47.35(3.69)</td><td>63.75(5.41)</td><td>61.67(8.98)</td></tr><tr><td>SP</td><td>47.44(2.09)</td><td>67.54(4.56)</td><td>63.77(7.43)</td></tr><tr><td>HBP</td><td>47.08(3.11)</td><td>67.30(4.69)</td><td>63.98(7.34)</td></tr><tr><td>P-tuning</td><td>46.26(3.19)</td><td>65.52(3.84)</td><td>62.03(9.62)</td></tr><tr><td>FPT(ours)</td><td>55.25(2.93)</td><td>68.19(4.21)</td><td>65.00(4.25)</td></tr></table>

Table 2: Experimental results comparing with promptbased methods. We report the mean performance and the standard deviation in brackets. The best results are in bold, and the best results of previous prompt-based methods are underlined.

## 6.1 Comparison with Prompt-based Methods

Table 2 shows the results of our proposed method FPT and prompt-based baselines under the fewshot setting. (1) Our method FPT significantly outperforms nearly all baseline methods across all three datasets under different shots, demonstrating that our method exhibits greater robustness and adaptability to variations in data sizes and languages. (2) FTP particularly excels on the ChineseLR dataset, and it outperforms the soft prompt (SP) method by 8.41, 14.1, 10.15, 9.94 and 7.9 points under 1, 2, 4, 8, 16 shots, respectively. (3) In the task of RA, the soft prompt method generally outperforms the hard prompt. Interestingly, the hybrid prompt, a combination of both, does not always yield better results than the standalone soft prompt. This could be attributed to the inherent challenge in designing and selecting effective hard prompts for RA. Nevertheless, as a hybrid prompt approach that integrates linguistic knowledge, our proposed method continues to exhibit robust performance, demonstrating its adaptability and effectiveness.

<table><tr><td>k</td><td>Methods</td><td>ChineseLR</td><td>Weebit</td><td>Cambridge</td></tr><tr><td rowspan="5"></td><td>FT</td><td>28.59(4.88)</td><td>45.99(2.94)</td><td>34.17(4.33)</td></tr><tr><td>SVM</td><td>25.34(3.87)</td><td>44.82(3.14)</td><td>35.31(5.23)</td></tr><tr><td>Con</td><td>28.53(4.68)</td><td>43.81(3.88)</td><td>33.33(10.1)</td></tr><tr><td>PF</td><td>30.13(3.99)</td><td>44.01(2.91)</td><td>35.11(9.12)</td></tr><tr><td>FPT(ours)</td><td>33.29(4.80)</td><td>46.67(3.50)</td><td>43.96(7.09)</td></tr><tr><td rowspan="5">2</td><td>FT</td><td>22.87(7.19)</td><td>48.79(3.49)</td><td>44.17(10.4)</td></tr><tr><td>SVM</td><td>23.95(9.28)</td><td>49.55(3.78)</td><td>43.99(11.0)</td></tr><tr><td>Con</td><td>25.61(8.21)</td><td>49.29(2.88)</td><td>41.67(8.16)</td></tr><tr><td>PF</td><td>26.12(7.21)</td><td>50.23(2.81)</td><td>41.52(7.34)</td></tr><tr><td>FPT(ours)</td><td>37.40(4.77)</td><td>56.03(3.48)</td><td>55.83(6.72)</td></tr><tr><td rowspan="5">4</td><td>FT</td><td>36.64(5.37)</td><td>52.46(4.28)</td><td>47.50(6.29)</td></tr><tr><td>SVM</td><td>37.11(6.88)</td><td>53.03(5.65)</td><td>47.58(8.67)</td></tr><tr><td>Con</td><td>36.64(5.37)</td><td>52.46(4.28)</td><td>47.50(6.29)</td></tr><tr><td>PF</td><td>37.13(5.11)</td><td>53.18(2.99)</td><td>48.46(4.79)</td></tr><tr><td>FPT(ours)</td><td>44.88(3.27)</td><td>56.17(3.84)</td><td>55.00(4.86)</td></tr><tr><td rowspan="5">8</td><td>FT</td><td>40.45(2.91)</td><td>61.11(3.15)</td><td>61.46(7.81)</td></tr><tr><td>SVM</td><td>40.52(3.67)</td><td>60.98(5.78)</td><td>61.55(9.10)</td></tr><tr><td>Con</td><td>41.65(2.98)</td><td>58.41(3.31)</td><td>58.96(7.43)</td></tr><tr><td>PF</td><td>44.00(2.86)</td><td>59.32(2.97)</td><td>55.62(10.9)</td></tr><tr><td>FPT(ours)</td><td>47.60(3.66)</td><td>62.40(3.30)</td><td>64.17(5.95)</td></tr><tr><td rowspan="5">16</td><td>FT</td><td>45.73(4.11)</td><td>65.93(5.50)</td><td>71.04(7.97)</td></tr><tr><td>SVM</td><td>46.85(3.72)</td><td>63.72(4.98)</td><td>71.22(8.15)</td></tr><tr><td>Con</td><td>48.33(3.99)</td><td>64.52(4.73)</td><td>71.46(6.12)</td></tr><tr><td>PF</td><td>48.66(3.20)</td><td>65.08(4.60)</td><td>69.38(6.79)</td></tr><tr><td>FPT(ours)</td><td>53.94(3.16)</td><td>68.10(3.25)</td><td>69.17(7.77)</td></tr></table>

Table 3: Experimental results comparing with feature fusion methods. Con means Concatenation. For a fair comparison, here FPT concatenates the embedded linguistic features to the embeddings of the input sequence (without hard prompt template) and outputs the classification logits over [CLS] token embedding instead of [MASK].

## 6.2 Comparison with Fusion Methods

Table 3 reports the experimental results comparing with fusion methods under the few-shot setting. (1) Our proposed method FPT shows a stable and significant improvement compared to the previous feature fusion methods. For instance, in the 2-shot setting, FPT outperforms the best previous fusion methods by 11.28, 5.8 and 11.66 points on ChineseLR, Weebit and Cambridge, respectively. This demonstrates our method’s effectiveness in integrating linguistic features for RA. (2) Methods with linguistic features perform better than standard fine-tuning on Chinese datasets. However, it may not necessarily lead to improvement on English datasets, especially when k is increased to a sufficient amount, which indicates that simply applying linguistic features to aid in English RA is not consistently effective.

<table><tr><td>Dataset</td><td>Methods</td><td>k=2</td><td>k=4</td><td>k=8</td></tr><tr><td rowspan="3">ChineseLR</td><td>FPT</td><td>46.24</td><td>48.93</td><td>52.66</td></tr><tr><td>-SC</td><td>40.97</td><td>46.03</td><td>50.48</td></tr><tr><td>-SC and FP</td><td>25.45</td><td>36.56</td><td>40.57</td></tr><tr><td rowspan="3">Weebit</td><td>FPT</td><td>55.10</td><td>57.70</td><td>64.92</td></tr><tr><td>-SC</td><td>52.68</td><td>56.92</td><td>63.63</td></tr><tr><td>-SC and FP</td><td>48.65</td><td>53.41</td><td>61.31</td></tr></table>

Table 4: Ablation study of FPT on ChineseLR and Weebit datasets. SC represents the similarity calibration and FP means utilizing linguistic features as soft prompts.

![](images/a305c044357af3f9a9575609fc70a90eb8a69ec2b7434ea7481da01781a94967.jpg)  
Figure 3: The comparison results of linguistic features, randomly initialized vectors and pseudo tokens.

## 6.3 Ablation Study

To validate the effectiveness of each component in our proposed model, we conduct ablation experiments on both English Weebit and ChineseLR datastes. Table 4 lists the results. Notably, our similarity calibration (SC) is built on the feature prompt (FP), with the aim to maintain consistent inter-class similarity of linguistic features. Therefore, removing FP also detaches SC, explaining why our ablation study is performed incrementally.

Our full model yields the best performance on both datasets. When removing the SC module, the performance is markedly decreased, demonstrating the necessity of retaining the linguistic features’ original similarity information during optimization. We have also investigated the impact of SC by visualising the similarity difference matrix before and after applying SC, the results of which are presented in Appendix C. Moreover, further removal of the FP shows a steep drop in performance (12.37 points on ChieseLR and 4.29 points on Weebit when k = 4), validating the effectiveness of incorporating linguistic features as soft prompts. We note that the improvement of SC and FP is more significant on the Chinese dataset compared to the English dataset, indicating that the Chinese RA task is more dependent on linguistic features.

## 6.4 The Significance of Linguistic Features

To further analyze whether linguistic features improve performance, in our model structure, we replace the linguistic feature vectors with randomly initialized vectors. On the other hand, we reimplement the Hybrid Prompt Tuning by utilizing pseudo tokens as soft prompts. We conduct experiments on WeeBit and ChineseLR datasets, and the comparison results are shown in Figure 3.

The performance on both datasets significantly decreases when the linguistic features are replaced with random vectors, especially on the ChineseLR dataset, where the decrease is up to 16.27%. The fewer the samples, the more severe the decline caused by the replacement, further indicating the beneficial role of linguistic features when data is insufficient. Moreover, compared to pseudo tokens, using vector-form embeddings as soft prompts requires the integration of linguistic knowledge to achieve better performance.

## 6.5 Comparison with the LLM

The large language model (LLM) excels at various downstream tasks without the need for parameter adjustment. We carry out experiments on LLM, utilizing the gpt-3.5-turbo-16k API. We sample the same examples as in other experiments, and the prompt is generated by GPT-4. Specifically, we provide GPT-4 with the task instructions to generate the system prompt and user input for gpt-3.5- turbo-16k, as shown in Figure 4. Table 5 shows that our model with 110M parameters significantly outperforms the LLM on the English dataset (except for one sample on Cambridge). Moreover, gpt-3.5-turbo-16k is unable to perform 1-shot or 2-shot experiments on ChineseLR due to its limited context length. This underscores the necessity for research on handling long texts in RA.

System Prompt:   
Evaluate the readability of the text using the   
following five levels (reading difficulty):   
Very Easy   
Easy   
Moderate   
Difficult   
Very Difficult   
Based on the provided text examples, assign a   
readability score to new text and display it in   
the following format: "[score: x]"   
User Input:   
Text: "content 1"   
[score: x]   
Text: "content 2"   
[score: x]   
Text: "content n"   
[score: x]   
New text:   
Text: "{}"  
Figure 4: The system prompt and the user input for prompting LLM.

<table><tr><td>k |</td><td>Dataset</td><td>FPT</td><td>LLM</td></tr><tr><td rowspan="2">0</td><td>Weebit</td><td>一</td><td>30.79</td></tr><tr><td>Cambridge ChineseLR</td><td>一</td><td>43.33 21.67</td></tr><tr><td rowspan="2">1</td><td>Weebit</td><td>43.61</td><td>31.75</td></tr><tr><td>Cambridge ChineseLR</td><td>44.17 39.63</td><td>48.33 1</td></tr><tr><td>2</td><td>Weebit Cambridge ChineseLR</td><td>55.10 59.79 46.24</td><td>33.17 54.16</td></tr></table>

Table 5: Comparison (accuracy) between our model and LLM (gpt-3.5-turbo-16k) on three datasets. k represents the number of in-context examples. Due to the limitation of context length, the experiments on Chinese dataset cannot be carried out.

## 7 Conclusion

Inspired by the solid performance of prompt tuning on classification tasks and the importance of linguistic features in the RA task, we empirically investigated the effectiveness of incorporating linguistic features into prompt tuning for RA. We convert linguistic features of the input into soft tokens and utilize the similarity calibration loss to preserve the similarity relationship between classes before and after the transformation. The results show noticeable improvements over previous fusion methods and prompt-based approaches in the few-shot learning setting. The ablation study further illustrated that the proposed model benefits from linguistic features and additional similarity calibration. Our proposed method, FPT, has demonstrated a new possibility of prompt tuning in an era dominated by LLMs, showcasing its undeniable significance and value in linguistic-related tasks.

## Acknowledgements

This work is supported by the National Natural Science Foundation of China (62076008) and the Key Project of Natural Science Foundation of China (61936012).

## Limitations

Our proposed method, which leverages the masked language model (MLM) backbone such as BERT, has demonstrated its efficacy across a variety of natural language processing tasks. Despite its strengths, we acknowledge several limitations that warrant further investigation.

Firstly, our approach exhibits constraints in processing long texts, a scenario frequently encountered in Chinese readability evaluation datasets. The inherent architecture of MLMs like BERT is optimized for shorter sequences, leading to potential performance degradation when dealing with extensive text inputs.

Secondly, while MLM-based methods are proficient in classification tasks, they often fall short in terms of interpretability of the classification outcomes. The black-box nature of these models makes it challenging to trace and understand the decision-making process, which is crucial for applications where justification of results is required.

Lastly, the success of our method is significantly contingent upon the quality of linguistic features extracted from the text. However, the extraction of high-quality linguistic features is not always guaranteed, especially in languages with rich morphology or poor data resources.

In conclusion, while our method stands as a robust approach for several NLP tasks, addressing these limitations is imperative for advancing the field and extending the applicability of MLM-based models to a broader spectrum of text analysis challenges. It is also worth noting that only one Chinese dataset is included in this work, as it appears to be the only Chinese RA dataset available to the best of our knowledge. We urge that more attention should be paid to this field of work and further experiments will be conducted if new datasets are released.

## Ethics Statement

Potential Risks Firstly, as a neural networkbased method, the predictive outcomes of our approach should not be applied in practical applications without the involvement of human experts. This is a responsible practice for the actual beneficiaries, the learners. Secondly, as mentioned earlier, low-quality or even incorrect linguistic features can negatively impact our method. Therefore, evaluating the quality of linguistic features is essential for the efficacy of our approach.

About Computational Budget For each k-shot experiment, we conducted a total of 16 repetitions (refer to Section 5.4) for all baselines and FPT. The duration of a single experiment varies according to the size of k (approximately 20 seconds to 200 seconds), but the time consumed by different methods is almost identical.

Use of Scientific Artifacts We utilize the lingfeat toolkit (Lee et al., 2021) to extract linguistic features from English texts; this toolkit is publicly accessible under the CC-BY-SA-4.0 license. For extracting Chinese linguistic features, we employ the zhfeat toolkit (Li et al., 2022).

Use of AI Assistants We have employed Chat-GPT as a writing assistant, primarily for polishing the text after the initial composition.

## References

Ion Madrazo Azpiazu and Maria Soledad Pera. 2019. Multiattentive recurrent neural network architecture for multilingual readability assessment. Transactions of the Association for Computational Linguistics, 7:421–436.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Xiang Chen, Lei Li, Ningyu Zhang, Chuanqi Tan, Fei Huang, Luo Si, and Huajun Chen. 2022. Relation extraction as open-book examination: Retrievalenhanced prompt tuning. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2443–2448.

Kevyn Collins-Thompson and Jamie Callan. 2004. Information retrieval for language tutoring: An overview of the reap project. In Proceedings ofthe 27th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’04, page 544–545, New York, NY, USA. Association for Computing Machinery.

Ganqu Cui, Shengding Hu, Ning Ding, Longtao Huang, and Zhiyuan Liu. 2022. Prototypical verbalizer for prompt-based few-shot tuning. arXiv preprint arXiv:2203.09770.

Tovly Deutsch, Masoud Jasbi, and Stuart Shieber. 2020. Linguistic features for readability assessment. arXiv preprint arXiv:2006.00377.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Rudolph Flesch. 1948. A new readability yardstick. Journal ofapplied psychology, 32(3):221.

Yuxian Gu, Xu Han, Zhiyuan Liu, and Minlie Huang. 2021. Ppt: Pre-trained prompt tuning for few-shot learning. arXiv preprint arXiv:2109.04332.

Yuxian Gu, Xu Han, Zhiyuan Liu, and Minlie Huang. 2022. PPT: Pre-trained prompt tuning for few-shot learning. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8410–8423, Dublin, Ireland. Association for Computational Linguistics.

Karen Hambardzumyan, Hrant Khachatrian, and Jonathan May. 2021. WARP: Word-level Adversarial ReProgramming. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4921–4933, Online. Association for Computational Linguistics.

Xu Han, Weilin Zhao, Ning Ding, Zhiyuan Liu, and Maosong Sun. 2022. Ptr: Prompt tuning with rules for text classification. AI Open, 3:182–192.

Jin Young Kim, Kevyn Collins-Thompson, Paul N. Bennett, and Susan T. Dumais. 2012. Characterizing web content, user interests, and search behavior by reading level and topic. In Proceedings ofthe Fifth ACM International Conference on Web Search and Data Mining, WSDM ’12, page 213–222, New York, NY, USA. Association for Computing Machinery.

Bruce W Lee, Yoo Sung Jang, and Jason Lee. 2021. Pushing on text readability assessment: A transformer meets handcrafted linguistic features. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 10669– 10686.

Bruce W. Lee and Jason Lee. 2023. Prompt-based learning for text readability assessment. In Findings ofthe Association for Computational Linguistics: EACL 2023, pages 1819–1824, Dubrovnik, Croatia. Association for Computational Linguistics.

Justin Lee and Sowmya Vajjala. 2022. A neural pairwise ranking model for readability assessment. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pages 3802–3813, Dublin, Ireland. Association for Computational Linguistics.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Wenbiao Li, Wang Ziyang, and Yunfang Wu. 2022. A unified neural network model for readability assessment with feature projection and length-balanced loss. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 7446–7457, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190.

Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. 2021. Gpt understands, too.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.

Ruotian Ma, Xin Zhou, Tao Gui, Yiding Tan, Linyang Li, Qi Zhang, and Xuanjing Huang. 2022. Templatefree prompt tuning for few-shot NER. In Proceedings ofthe 2022 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 5721–5732, Seattle, United States. Association for Computational Linguistics.

Matej Martinc, Senja Pollak, and Marko Robnik-Šikonja. 2021. Supervised and unsupervised neural approaches to text readability. Computational Linguistics, 47(1):141–179.

Changping Meng, Muhao Chen, Jie Mao, and Jennifer Neville. 2021. Readnet: A hierarchical transformer framework for web article readability analysis. CoRR, abs/2103.04083.

Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

Emily Pitler and Ani Nenkova. 2008. Revisiting readability: A unified framework for predicting text quality. In Proceedings ofthe 2008 conference on empirical methods in natural language processing, pages 186–195.

Xinying Qiu, Yuan Chen, Hanwu Chen, Jian-Yun Nie, Yuming Shen, and Dawei Lu. 2021. Learning syntactic dense embedding with correlation graph for automatic readability assessment. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3013–3025, Online. Association for Computational Linguistics.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551.

Luz Rello, Horacio Saggion, Ricardo Baeza-Yates, and Eduardo Graells. 2012. Graphical schemes may improve readability but not understandability for people with dyslexia. In Proceedings ofthe First Workshop on Predicting and Improving Text Readability for target reader populations, pages 25–32, Montréal, Canada. Association for Computational Linguistics.

Timo Schick and Hinrich Schütze. 2021. Exploiting cloze-questions for few-shot text classification and natural language inference. In Proceedings of the 16th Conference ofthe European Chapter ofthe Association for Computational Linguistics: Main Volume, pages 255–269.

Nathan Schucher, Siva Reddy, and Harm de Vries. 2021. The power of prompt tuning for low-resource semantic parsing. arXiv preprint arXiv:2110.08525.

Sarah E Schwarm and Mari Ostendorf. 2005. Reading level assessment using support vector machines and statistical language models. In Proceedings of the 43rd annual meeting of the Association for Computational Linguistics (ACL’05), pages 523–530.

Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, and Sameer Singh. 2020. AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4222–4235, Online. Association for Computational Linguistics.

Sowmya Vajjala. 2022. Trends, limitations and open challenges in automatic readability assessment research. In Proceedings ofthe Thirteenth Language Resources and Evaluation Conference, pages 5366– 5377, Marseille, France. European Language Resources Association.

Sowmya Vajjala and Detmar Meurers. 2012. On improving the accuracy of readability classification using insights from second language acquisition. In Proceedings ofthe seventh workshop on building educational applications using NLP, pages 163–173.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Hui Wu and Xiaodong Shi. 2022. Adversarial soft prompt tuning for cross-domain sentiment analysis. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2438–2447.

Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, and Hang Li. 2008. Listwise Approach to Learning to Rank: Theory and Algorithm, page 1192–1199. Association for Computing Machinery, New York, NY, USA.

Menglin Xia, Ekaterina Kochmar, and Ted Briscoe. 2019. Text readability assessment for second language learners. arXiv preprint arXiv:1906.07580.

Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical attention networks for document classification. In Proceedings ofthe 2016 conference ofthe North American chapter of the association for computational linguistics: human language technologies, pages 1480– 1489.

## A Details of Linguistic Features

A.1 Chinese Linguistic Features
<table><tr><td rowspan=1 colspan=1>Idx</td><td rowspan=1 colspan=1>Dim</td><td rowspan=1 colspan=1>Feature description</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Total number of characters</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Idx</td><td rowspan=1 colspan=1>Dim</td><td rowspan=1 colspan=1>Feature description</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Number of character types</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Type Token Ratio (TTR)</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average number of strokes</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Weighted average number of strokes</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>Number of characters with different strokes</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>Proportion of characters with different strokes</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average character frequency</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Weighted average character frequency</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Number of single characters</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Proportion of single characters</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Number of common characters</td></tr><tr><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Proportion of common characters</td></tr><tr><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Number of unregistered characters</td></tr><tr><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Proportion of unregistered characters</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Number of first-level characters</td></tr><tr><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Proportion of first-level characters</td></tr><tr><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Number of second-level characters</td></tr><tr><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Proportion of second-level characters</td></tr><tr><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Number of third-level characters</td></tr><tr><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Proportion of third-level characters</td></tr><tr><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Number of fourth-level characters</td></tr><tr><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Proportion of fourth-level characters</td></tr><tr><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average character level</td></tr></table>

Table 6: Character features description.
<table><tr><td colspan="1" rowspan="1">Idx</td><td colspan="1" rowspan="1">Dim</td><td colspan="1" rowspan="1">Feature description</td></tr><tr><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Total number of words</td></tr><tr><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Number of word types</td></tr><tr><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Type Token Ratio (TTR)</td></tr><tr><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average word length</td></tr><tr><td colspan="1" rowspan="1">5</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Weighted average word length</td></tr><tr><td colspan="1" rowspan="1">6</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average word frequency</td></tr><tr><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Weighted average word frequency</td></tr><tr><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Number of single-character words</td></tr><tr><td colspan="1" rowspan="1">9</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Proportion of single-character words</td></tr><tr><td colspan="1" rowspan="1">10</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Number of two-character words</td></tr><tr><td colspan="1" rowspan="1">11</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Proportion of two-character words</td></tr><tr><td colspan="1" rowspan="1">12</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Number of three-character words</td></tr><tr><td colspan="1" rowspan="1">13</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Proportion of three-character words</td></tr><tr><td colspan="1" rowspan="1">14</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Number of four-character words</td></tr><tr><td colspan="1" rowspan="1">15</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Proportion of four-character words</td></tr><tr><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Number of multi-character words</td></tr><tr><td colspan="1" rowspan="1">17</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Proportion of multi-character words</td></tr><tr><td colspan="1" rowspan="1">18</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Number of idioms</td></tr><tr><td colspan="1" rowspan="1">19</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Number of single words</td></tr><tr><td colspan="1" rowspan="1">20</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Proportion of single words</td></tr><tr><td colspan="1" rowspan="1">21</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Number of unregistered words</td></tr><tr><td colspan="1" rowspan="1">22</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Proportion of unregistered words</td></tr><tr><td colspan="1" rowspan="1">23</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Number of first-level words</td></tr><tr><td colspan="1" rowspan="1">24</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Proportion of first-level words</td></tr><tr><td colspan="1" rowspan="1">25</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Number of second-level words</td></tr><tr><td colspan="1" rowspan="1">26</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Proportion of second-level words</td></tr><tr><td colspan="1" rowspan="1">27</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Number of third-level words</td></tr><tr><td colspan="1" rowspan="1">28</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Proportion of third-level words</td></tr><tr><td colspan="1" rowspan="1">29</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Number of fourth-level words</td></tr><tr><td colspan="1" rowspan="1">30</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Proportion of fourth-level words</td></tr><tr><td colspan="1" rowspan="1">31</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average word level</td></tr><tr><td colspan="1" rowspan="1">32</td><td colspan="1" rowspan="1">57</td><td colspan="1" rowspan="1">Number of words with different POS</td></tr><tr><td colspan="1" rowspan="1">33</td><td colspan="1" rowspan="1">57</td><td colspan="1" rowspan="1">Proportion of words with different POS</td></tr><tr><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Total number of sentences</td></tr><tr><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average characters in a sentence</td></tr><tr><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average words in a sentence</td></tr><tr><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Maximum characters in a sentence</td></tr><tr><td colspan="1" rowspan="1">5</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Maximum words in a sentence</td></tr><tr><td colspan="1" rowspan="1">6</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Number of clauses</td></tr><tr><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average characters in a clause</td></tr><tr><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average words in a clause</td></tr><tr><td colspan="1" rowspan="1">9</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Maximum characters in a clause</td></tr><tr><td colspan="1" rowspan="1">10</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Maximum words in a clause</td></tr><tr><td colspan="1" rowspan="1">11</td><td colspan="1" rowspan="1">30</td><td colspan="1" rowspan="1">Sentence length distribution</td></tr><tr><td colspan="1" rowspan="1">12</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average syntax tree height</td></tr><tr><td colspan="1" rowspan="1">13</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Maximum syntax tree height</td></tr><tr><td colspan="1" rowspan="1">14</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Syntax tree height &lt;= 5 ratio</td></tr><tr><td colspan="1" rowspan="1">15</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Syntax tree height &lt;= 10 ratio</td></tr><tr><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Syntax tree height &lt;= 15 ratio</td></tr><tr><td colspan="1" rowspan="1">17</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Syntax tree height &gt;= 16 ratio</td></tr><tr><td colspan="1" rowspan="1">18</td><td colspan="1" rowspan="1">14</td><td colspan="1" rowspan="1">Dependency distribution</td></tr></table>

Table 7: Word features description.

Table 8: Sentence features description.
<table><tr><td rowspan=1 colspan=1>Idx</td><td rowspan=1 colspan=1>Dim</td><td rowspan=1 colspan=1>Feature description</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Total number of paragraphs</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average characters in a paragraph</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average words in a paragraph</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Maximum characters in a paragraph</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Maximum words in a paragraph</td></tr></table>

Table 9: Paragraph features description.

## A.2 English Linguistic Features

<table><tr><td rowspan=1 colspan=1>Idx</td><td rowspan=1 colspan=1>Dim</td><td rowspan=1 colspan=1>Feature description</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Total number of Entities Mentions counts</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average number of Entities Mentions counts persentence</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average number of Entities Mentions counts pertoken (word)</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Total number of unique Entities</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average number of unique Entities per sentence</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average number of Entities Mentions counts pertoken (word)s</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Total number of unique Entities</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of ss transitions to total</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of so transitions to total</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of sx transitions to total</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of sn transitions to total</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of os transitions to total</td></tr><tr><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of oo transitions to total</td></tr><tr><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of ox transitions to total</td></tr><tr><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of on transitions to total</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of xs transitions to total</td></tr><tr><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of xo transitions to total</td></tr><tr><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of xx transitions to total</td></tr><tr><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of xn transitions to total</td></tr><tr><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of ns transitions to total</td></tr><tr><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of no transitions to total</td></tr><tr><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of nx transitions to total</td></tr><tr><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of nn transitions to total</td></tr><tr><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Local Coherence for PA score</td></tr><tr><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Local Coherence for PW score</td></tr><tr><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Local Coherence for PU score</td></tr><tr><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Local Coherence distance for PA score</td></tr><tr><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Local Coherence distance for PW score</td></tr><tr><td rowspan=1 colspan=1>29</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Local Coherence distance for PU score</td></tr></table>

<table><tr><td></td><td></td><td>Idx Dim Feature description</td></tr><tr><td></td><td></td><td></td></tr></table>

Table 10: Discourse features description.

<table><tr><td colspan="1" rowspan="1">Idx</td><td colspan="1" rowspan="1">Dim</td><td colspan="1" rowspan="1">Feature description</td></tr><tr><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Total count of Noun phrases</td></tr><tr><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Noun phrases per sentence</td></tr><tr><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Noun phrases per token</td></tr><tr><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Noun phrases count to Verb phrasescount</td></tr><tr><td colspan="1" rowspan="1">5</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Noun phrases count to SubordinateClauses count</td></tr><tr><td colspan="1" rowspan="1">6</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Ratio of Noun phrases count to Prep phrasescount</td></tr><tr><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Noun phrases count to Adj phrasescount</td></tr><tr><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Noun phrases count to Adv phrasescount</td></tr><tr><td colspan="1" rowspan="1">9</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Total count of Verb phrases</td></tr><tr><td colspan="1" rowspan="1">10</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Verb phrases per sentence</td></tr><tr><td colspan="1" rowspan="1">11</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Verb phrases per token</td></tr><tr><td colspan="1" rowspan="1">12</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Verb phrases count to Noun phrasescount</td></tr><tr><td colspan="1" rowspan="1">13</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Verb phrases count to SubordinateClauses count</td></tr><tr><td colspan="1" rowspan="1">14</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Verb phrases count to Prep phrasescount</td></tr><tr><td colspan="1" rowspan="1">15</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Verb phrases count to Adj phrases count</td></tr><tr><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Verb phrases count to Adv phrasescount</td></tr><tr><td colspan="1" rowspan="1">17</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Total count of Subordinate Clauses</td></tr><tr><td colspan="1" rowspan="1">18</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Subordinate Clauses per sen-tence</td></tr><tr><td colspan="1" rowspan="1">19</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Subordinate Clauses per token</td></tr><tr><td colspan="1" rowspan="1">20</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Subordinate Clauses count to Nounphrases count</td></tr><tr><td colspan="1" rowspan="1">21</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Subordinate Clauses count to Verbphrases count</td></tr><tr><td colspan="1" rowspan="1">22</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Subordinate Clauses count to Prepphrases count</td></tr><tr><td colspan="1" rowspan="1">23</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Subordinate Clauses count to Adjphrases count</td></tr><tr><td colspan="1" rowspan="1">24</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Subordinate Clauses count to Advphrases count</td></tr><tr><td colspan="1" rowspan="1">25</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Total count of prepositional phrases</td></tr><tr><td colspan="1" rowspan="1">26</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of prepositional phrases per sen-tence</td></tr><tr><td colspan="1" rowspan="1">27</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of prepositional phrases per token</td></tr><tr><td colspan="1" rowspan="1">28</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Prep phrases count to Noun phrasescount</td></tr><tr><td colspan="1" rowspan="1">29</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Prep phrases count to Verb phrasescount</td></tr><tr><td colspan="1" rowspan="1">30</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Prep phrases count to SubordinateClauses count</td></tr><tr><td colspan="1" rowspan="1">31</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Prep phrases count to Adj phrases count</td></tr><tr><td colspan="1" rowspan="1">32</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Prep phrases count to Adv phrases count</td></tr><tr><td colspan="1" rowspan="1">33</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Total count of Adjective phrases</td></tr><tr><td colspan="1" rowspan="1">34</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Adjective phrases per sentence</td></tr><tr><td colspan="1" rowspan="1">35</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Adjective phrases per token</td></tr><tr><td colspan="1" rowspan="1">36</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adj phrases count to Noun phrasescount</td></tr><tr><td colspan="1" rowspan="1">37</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adj phrases count to Verb phrases count</td></tr><tr><td colspan="1" rowspan="1">38</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adj phrases count to SubordinateClauses count</td></tr><tr><td colspan="1" rowspan="1">39</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adj phrases count to Prep phrases count</td></tr><tr><td colspan="1" rowspan="1">40</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adj phrases count to Adv phrases count</td></tr><tr><td colspan="1" rowspan="1">Idx</td><td colspan="1" rowspan="1">Dim Fe</td><td colspan="1" rowspan="1">ature description</td></tr><tr><td colspan="1" rowspan="1">41</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Total count of Adverb phrases</td></tr><tr><td colspan="1" rowspan="1">42</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Adverb phrases per sentence</td></tr><tr><td colspan="1" rowspan="1">43</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Adverb phrases per token</td></tr><tr><td colspan="1" rowspan="1">44</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adv phrases count to Noun phrasescount</td></tr><tr><td colspan="1" rowspan="1">45</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adv phrases count to Verb phrasescount</td></tr><tr><td colspan="1" rowspan="1">46</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Ratio of Adv phrases count to SubordinateClauses count</td></tr><tr><td colspan="1" rowspan="1">47</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adv phrases count to Prep phrases count</td></tr><tr><td colspan="1" rowspan="1">48</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adv phrases count to Adj phrases count</td></tr><tr><td colspan="1" rowspan="1">49</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Total Tree height of all sentences</td></tr><tr><td colspan="1" rowspan="1">50</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average Tree height per sentence</td></tr><tr><td colspan="1" rowspan="1">51</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average Tree height per token (word)</td></tr><tr><td colspan="1" rowspan="1">52</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Total length of flattened Trees</td></tr><tr><td colspan="1" rowspan="1">53</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average length of flattened Trees per sentence</td></tr><tr><td colspan="1" rowspan="1">54</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average length of flattened Trees per token(word)</td></tr><tr><td colspan="1" rowspan="1">55</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Total count of Noun POS tags</td></tr><tr><td colspan="1" rowspan="1">56</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Noun POS tags per sentence</td></tr><tr><td colspan="1" rowspan="1">57</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Noun POS tags per token</td></tr><tr><td colspan="1" rowspan="1">58</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Noun POS count to Adjective POScount</td></tr><tr><td colspan="1" rowspan="1">59</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Noun POS count to Verb POS count</td></tr><tr><td colspan="1" rowspan="1">60</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Noun POS count to Adverb POS count</td></tr><tr><td colspan="1" rowspan="1">61</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="2">Ratio of Noun POS count to Subordinating Con-junction count</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">62</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Noun POS count to Coordinating Con-junction count</td></tr><tr><td colspan="1" rowspan="1">63</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Total count of Verb POS tags</td></tr><tr><td colspan="1" rowspan="1">64</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Verb POS tags per sentence</td></tr><tr><td colspan="1" rowspan="1">65</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Verb POS tags per token</td></tr><tr><td colspan="1" rowspan="1">66</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Verb POS count to Adjective POS count</td></tr><tr><td colspan="1" rowspan="1">67</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Verb POS count to Noun POS count</td></tr><tr><td colspan="1" rowspan="1">68</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Verb POS count to Adverb POS count</td></tr><tr><td colspan="1" rowspan="1">69</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="2">Ratio of Verb POS count to Subordinating Con-junction count</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">70</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Verb POS count to Coordinating Con-junction count</td></tr><tr><td colspan="1" rowspan="1">71</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Total count of Adjective POS tags</td></tr><tr><td colspan="1" rowspan="1">72</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Adjective POS tags per sen-tence</td></tr><tr><td colspan="1" rowspan="1">73</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Adjective POS tags per token</td></tr><tr><td colspan="1" rowspan="1">74</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adjective POS count to Noun POScount</td></tr><tr><td colspan="1" rowspan="1">75</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adjective POS count to Verb POS count</td></tr><tr><td colspan="1" rowspan="1">76</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adjective POS count to Adverb POScount</td></tr><tr><td colspan="1" rowspan="1">77</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adjective POS count to SubordinatingConjunction count</td></tr><tr><td colspan="1" rowspan="1">78</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adjective POS count to CoordinatingConjunction count</td></tr><tr><td colspan="1" rowspan="1">79</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Total count of Adverb POS tags</td></tr><tr><td colspan="1" rowspan="1">80</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Adverb POS tags per sentence</td></tr><tr><td colspan="1" rowspan="1">81</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Average count of Adverb POS tags per token</td></tr><tr><td colspan="1" rowspan="1">82</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adverb POS count to Adjective POScount</td></tr><tr><td colspan="1" rowspan="1">83</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adverb POS count to Noun POS count</td></tr><tr><td colspan="1" rowspan="1">84</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adverb POS count to Verb POS count</td></tr><tr><td colspan="1" rowspan="1">85</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Ratio of Adverb POS count to SubordinatingConjunction count</td></tr><tr><td colspan="1" rowspan="1">86</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Ratio of Adverb POS count to Coordinating Con-junction count</td></tr><tr><td colspan="1" rowspan="1">87</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Total count of Subordinating Conjunction POStags</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Idx</td><td rowspan=1 colspan=1>Dim</td><td rowspan=1 colspan=1>Feature description</td></tr><tr><td rowspan=1 colspan=1>88</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average count of Subordinating ConjunctionPOS tags per sentence</td></tr><tr><td rowspan=1 colspan=1>89</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average count of Subordinating ConjunctionPOS tags per token</td></tr><tr><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of Subordinating Conjunction POS countto Adjective POS count</td></tr><tr><td rowspan=1 colspan=1>91</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of Subordinating Conjunction POS countto Noun POS count</td></tr><tr><td rowspan=1 colspan=1>92</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of Subordinating Conjunction POS countto Verb POS count</td></tr><tr><td rowspan=1 colspan=1>93</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of Subordinating Conjunction POS countto Adverb POS count</td></tr><tr><td rowspan=1 colspan=1>94</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of Subordinating Conjunction POS countto Coordinating Conjunction count</td></tr><tr><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Total count of Coordinating Conjunction POStags</td></tr><tr><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average count of Coordinating Conjunction POStags per sentence</td></tr><tr><td rowspan=1 colspan=1>97</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average count of Coordinating Conjunction POStags per token</td></tr><tr><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of Coordinating Conjunction POS countto Adjective POS count</td></tr><tr><td rowspan=1 colspan=1>99</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of Coordinating Conjunction POS countto Noun POS count</td></tr><tr><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of Coordinating Conjunction POS countto Verb POS count</td></tr><tr><td rowspan=1 colspan=1>101</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of Coordinating Conjunction POS countto Adverb POS count</td></tr><tr><td rowspan=1 colspan=1>102</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of Coordinating Conjunction POS countto Subordinating Conjunction count</td></tr><tr><td rowspan=1 colspan=1>103</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Total count of Content words</td></tr><tr><td rowspan=1 colspan=1>104</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average count of Content words per sentence</td></tr><tr><td rowspan=1 colspan=1>105</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average count of Content words per token</td></tr><tr><td rowspan=1 colspan=1>106</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Total count of Function words</td></tr><tr><td rowspan=1 colspan=1>107</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average count of Function words per sentence</td></tr><tr><td rowspan=1 colspan=1>108</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average count of Function words per token</td></tr><tr><td rowspan=1 colspan=1>109</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Ratio of Content words to Function words</td></tr></table>

Table 11: Syntactic features description.
<table><tr><td colspan="1" rowspan="1">Idx</td><td colspan="2" rowspan="1">Dim</td><td colspan="2" rowspan="1">Feature description</td></tr><tr><td colspan="1" rowspan="1">1</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Unique Nouns/total Nouns (Noun Variation-1)</td></tr><tr><td colspan="1" rowspan="1">2</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">(Unique Nouns**2)/total Nouns (Squared NounVariation-1)</td></tr><tr><td colspan="1" rowspan="1">3</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Unique Nouns/sqrt(2*total Nouns) (CorrectedNoun Variation-1)</td></tr><tr><td colspan="1" rowspan="1">4</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Unique Verbs/total Verbs (Verb Variation-1)</td></tr><tr><td colspan="1" rowspan="1">5</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">(Unique Verbs**2)/total Verbs (Squared VerbVariation-1)</td></tr><tr><td colspan="1" rowspan="1">6</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Unique Verbs/sqrt(2*total Verbs) (CorrectedVerb Variation-1)</td></tr><tr><td colspan="1" rowspan="1">7</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Unique Adjectives/total Adjectives (AdjectiveVariation-1)</td></tr><tr><td colspan="1" rowspan="1">8</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">(Unique  Adjectives**2)/total   Adjectives(Squared Adjective Variation-1)</td></tr><tr><td colspan="1" rowspan="1">9</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Unique Adjectives/sqrt(2*total Adjectives) (Cor-rected Adjective Variation-1)</td></tr><tr><td colspan="1" rowspan="1">10</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Unique Adverbs/total Adverbs (AdVerbVariation-1)</td></tr><tr><td colspan="1" rowspan="1">11</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">(Unique Adverbs**2)/total Adverbs (SquaredAdVerb Variation-1)</td></tr><tr><td colspan="1" rowspan="1">12</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Unique Adverbs/sqrt(2*total Adverbs) (Cor-rected AdVerb Variation-1)</td></tr><tr><td colspan="1" rowspan="1">13</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Unique tokens/total tokens (TTR)</td></tr><tr><td colspan="1" rowspan="1">14</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Unique tokens/sqrt(2*total tokens) (CorrectedTTR)</td></tr><tr><td colspan="1" rowspan="1">15</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Log(uniquetokens)/log(total tokens) (Bi-Logarithmic TTR)</td></tr><tr><td colspan="1" rowspan="1">16</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">(Log(uniquetokens))**2/log(totaltokens/u-nique tokens) (Uber Index)</td></tr><tr><td colspan="1" rowspan="1">17</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Measure of Textual Lexical Diversity (defaultTTR = 0.72)</td></tr><tr><td colspan="1" rowspan="1">18</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Total AoA (Age of Acquisition) of words</td></tr><tr><td colspan="1" rowspan="1">19</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average AoA of words per sentence</td></tr><tr><td colspan="1" rowspan="1">20</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average AoA of words per token</td></tr><tr><td colspan="1" rowspan="1">21</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Total lemmas AoA of lemmas</td></tr><tr><td colspan="1" rowspan="1">22</td><td colspan="2" rowspan="1">1</td><td colspan="1" rowspan="1">Average l</td><td colspan="1" rowspan="1">emmas AoA of lemmas per sentence</td></tr><tr><td colspan="1" rowspan="1">23</td><td colspan="2" rowspan="1">1</td><td colspan="1" rowspan="1">Average</td><td colspan="1" rowspan="1">lemmas AoA of lemmas per token</td></tr><tr><td colspan="1" rowspan="1">24</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Total lemmas AoA of lemmas, Bird norm</td></tr><tr><td colspan="1" rowspan="1">25</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average lemmas AoA of lemmas, Bird norm persentence</td></tr><tr><td colspan="1" rowspan="1">26</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average lemmas AoA of lemmas, Bird norm pertoken</td></tr><tr><td colspan="1" rowspan="1">27</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Total lemmas AoA of lemmas, Bristol norm</td></tr><tr><td colspan="1" rowspan="1">28</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average lemmas AoA of lemmas, Bristol normper sentence</td></tr><tr><td colspan="1" rowspan="1">29</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average lemmas AoA of lemmas, Bristol normper token</td></tr><tr><td colspan="1" rowspan="1">30</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Total AoA of lemmas, Cortese and Khanna norm</td></tr><tr><td colspan="1" rowspan="1">31</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average AoA of lemmas, Cortese and Khannanorm per sentence</td></tr><tr><td colspan="1" rowspan="1">32</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average AoA of lemmas, Cortese and Khannanorm per token</td></tr><tr><td colspan="1" rowspan="1">33</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Total SubtlexUS FREQcount value</td></tr><tr><td colspan="1" rowspan="1">34</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average SubtlexUS FREQcount value per sen-tenc</td></tr><tr><td colspan="1" rowspan="1">35</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average SubtlexUS FREQcount value per token</td></tr><tr><td colspan="1" rowspan="1">36</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Total SubtlexUS CDcount value</td></tr><tr><td colspan="1" rowspan="1">37</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average SubtlexUS CDcount value per sentence</td></tr><tr><td colspan="1" rowspan="1">38</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average SubtlexUS CDcount value per token</td></tr><tr><td colspan="1" rowspan="1">39</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Total SubtlexUS FREQlow value</td></tr><tr><td colspan="1" rowspan="1">40</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average SubtlexUS FREQlow value per sen-tence</td></tr><tr><td colspan="1" rowspan="1">41</td><td></td><td colspan="1" rowspan="1"></td><td colspan="2" rowspan="1">Average SubtlexUS FREQlow value per token</td></tr><tr><td colspan="1" rowspan="1">42</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Total SubtlexUS CDlow value</td></tr><tr><td colspan="1" rowspan="1">43</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average SubtlexUS CDlow value per sentence</td></tr><tr><td colspan="1" rowspan="1">44</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average SubtlexUS CDlow value per token</td></tr><tr><td colspan="1" rowspan="1">45</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Total SubtlexUS SUBTLWF value</td></tr><tr><td colspan="1" rowspan="1">46</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average SubtlexUS SUBTLWF value per sen-tence</td></tr><tr><td colspan="1" rowspan="1">47</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average SubtlexUS SUBTLWF value per token</td></tr><tr><td colspan="1" rowspan="1">48</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Total SubtlexUS Lg10WF value</td></tr><tr><td colspan="1" rowspan="1">49</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average SubtlexUS Lg10WF value per sentence</td></tr><tr><td colspan="1" rowspan="1">50</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average SubtlexUS Lg10WF value per token</td></tr><tr><td colspan="1" rowspan="1">51</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Total SubtlexUS SUBTLCD value</td></tr><tr><td colspan="1" rowspan="1">52</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average SubtlexUS SUBTLCD value per sen-tence</td></tr><tr><td colspan="1" rowspan="1">53</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average SubtlexUS SUBTLCD value per token</td></tr><tr><td colspan="1" rowspan="1">54</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Total SubtlexUS Lg10CD value</td></tr><tr><td colspan="1" rowspan="1">55</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average SubtlexUS Lg10CD value per sentence</td></tr><tr><td colspan="1" rowspan="1">56</td><td colspan="2" rowspan="1">1</td><td colspan="2" rowspan="1">Average SubtlexUS Lg10CD value per token</td></tr></table>

Table 12: Lexico Semantic features description.
<table><tr><td rowspan=1 colspan=1>Idx</td><td rowspan=1 colspan=1>Dim</td><td rowspan=1 colspan=1>Feature description</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Total count of tokens x total count of sentence</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Sqrt(total count of tokens x total count of sen-tence)</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Log(total count of tokens)/log(total count of sen-tence)</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Idx</td><td rowspan=1 colspan=1>Dim</td><td rowspan=1 colspan=1>Feature description</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average count of tokens per sentence</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average count of syllables per sentence</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average count of syllables per token</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average count of characters per sentence</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Average count of characters per token</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Smog Index</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Coleman Liau Readability Score</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Gunning Fog Count Score</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>New Automated Readability Index</td></tr><tr><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Flesch Kincaid Grade Level</td></tr><tr><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Linsear Write Formula Score</td></tr></table>

Table 13: Shallow Traditional features description.

## B Templates

Chinese Dataset Based on the Chinese Curriculum Standardsfor Compulsory Education, we devise the following templates:

• T<sub>1</sub>( ) = 一篇第[MASK]学段的文章:

• T<sub>2</sub>( ) = 这是一篇第[MASK]学段的课文:

• T ( ) = 一篇第[MASK]学段的课文:

• T<sub>4</sub>( ) = 一篇阅读难度为[MASK]的课文:

English Dataset Based on (Vajjala and Meurers, 2012), we use the following templates:

• T<sub>1</sub>( ) = A [MASK] article to understand:

• T ( ) = A [MASK] text to understand:

• T<sub>3</sub>( ) = This is a [MASK] article to understand:

• T<sub>4</sub>( ) = A [MASK] article to read:

## C The Impact of Similarity Calibration

To investigate the impact of Similarity Calibration (SC), we plot the similarity difference matrices before and after linguistic feature embedding on two datasets, both with and without SC. Specifically, we calculate the similarity of linguistic features between each category before and after embedding to obtain two similarity matrices. Then we subtract the former from the latter to obtain the difference matrix. The results are shown in Figure 5, where the diagonal of the matrix represents the similarity of the linguistic features from the same category.

On both datasets, SC can effectively increase the similarity between the same and analogous categories (represented by warm colors), while reducing the similarity between distance categories (represented by cool colors). This can provide effective assistance for classification tasks.

![](images/9375758ccfccc40ed42b9d2d881f04f81cb9ce9e338de9aa806b6b908b66938e.jpg)  
(a) ChineseLR w/o SC

![](images/8a384fd6b3067844121bc7f5817135ad4d7febf9239dd400c6a294ae66152172.jpg)  
(b) ChineseLR w/ SC

![](images/14470c33717885fa83a20580b0a7d761acfed9208f320ceb70d195eec94e202d.jpg)  
(c) Weebit w/o SC

![](images/6a69bcbbfb00414fc9d03150eac23b0460396f8b2016332a489fb8b19c2d4fcc.jpg)  
(d) Weebit w/ SC

![](images/3a3e4f182fe757fe5e1375e3eed33b509536c23a31b1aeb6aa581c220cd898c5.jpg)

Figure 5: Similarity difference matrices. We plot the difference matrices of similarity before and after linguistic feature embedding, both with and without SC. The horizontal and vertical coordinates represent the level of linguistic features. By comparing the diagonal of the matrix before and after the similarity calibration (that is, the similarity between linguistic features of the same level), the similarity between analogous categories is drawn closer.