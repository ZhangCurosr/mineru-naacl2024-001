# PILOT: Legal Case Outcome Prediction with Case Law

Lang Cao<sup>1</sup>, Zifeng Wang<sup>1</sup>, Cao Xiao<sup>2</sup>, Jimeng Sun<sup>1</sup>

<sup>1</sup>University of Illinois Urbana-Champaign

<sup>2</sup>GE Healthcare

{langcao2, zifengw2, jimeng}@illinois.edu, cao.xiao@ge.com

## Abstract

Machine learning shows promise in predicting the outcome of legal cases, but most research has concentrated on civil law cases rather than case law systems. We identified two unique challenges in making legal case outcome predictions with case law. First, it is crucial to identify relevant precedent cases that serve as fundamental evidence for judges during decisionmaking. Second, it is necessary to consider the evolution of legal principles over time, as early cases may adhere to different legal contexts. In this paper, we proposed a new framework named PILOT (PredictIng Legal case OuTcome) for case outcome prediction. It comprises two modules for relevant case retrieval and temporal pattern handling, respectively. To benchmark the performance of existing legal case outcome prediction models, we curated a dataset from a large-scale case law database. We demonstrate the importance of accurately identifying precedent cases and mitigating the temporal shift when making predictions for case law, as our method shows a significant improvement over the prior methods that focus on civil law case outcome predictions.

## 1 Introduction

Predicting legal case outcomes is a crucial task that facilitates data-driven decision-making in legal cases based on relevant information, such as the factual description (Cui et al., 2022). With a significant number of legal cases arising worldwide each year, legal professionals face the daunting task of reviewing the extensive legal text and delivering accurate and fair outcomes. Legal case outcome prediction has the potential to simplify this labor-intensive document review process, enhancing strategy and decision-making. As the volume and complexity of cases continue to escalate, the development of precise and reliable legal case outcome prediction systems becomes an urgent priority.

Two legal frameworks exist across the globe: the civil law system, which assesses each case based on comprehensive codes and statutes, and the case law system, where the interpretation and application of law heavily depend on precedent court decisions. Most existing works were proposed for the civil law framework, including charge prediction, violated articles prediction, prison terms prediction, court decision prediction, and court view generation (Paul et al., 2020; Hu et al., 2018; Chen et al., 2019; Chalkidis et al., 2019a; Alali et al., 2021; Ye et al., 2018). However, predicting case outcomes in case law systems presents unique challenges distinct from those in civil law: (1) it requires the identification of similar historical cases, and (2) meanwhile accounting for the evolution of legal principles over time.

• Precedent Cases In the case law system, the application of precedents plays a crucial role. To achieve accurate prediction of case outcomes, it is vital to identify past cases that exhibit similar legal principles, factual contexts, and key arguments. Moreover, how to effectively utilize the retrieved cases in the prediction of new case outcomes still requires further exploration.

• Temporal Shift One aspect that has not received sufficient attention in previous research is the temporal evolution of legal principles. We argue that it is crucial to not only comprehend the historical context and development of legal precedents but also to effectively capture and represent the temporal shifts of laws in predictive modeling.

To fill the gap, we proposed a new model named PILOT (PredictIng Legal case OuTcome) for case outcome prediction, which consists of two functional modules:

• Case Retrieval We initially train the module in an unsupervised manner to obtain text embeddings for cases. These embeddings are then used to query and select the most relevant precedent cases, which serve as additional inputs to our main model.

• Temporal Pattern Mining A temporal decay term is introduced to ensure the model captures the more recent patterns and explicitly learns to adapt to the temporal pattern change.

To facilitate this line of research, we established a new dataset named ECHR2023, which was extracted from the European Court of Human Rights (ECHR) database<sup>1</sup> with focusing on precedent cases and temporal concept shift. We evaluated the proposed PILOT model against state-of-the-art models on ECHR2023. The experiment results show that PILOT substantially outperforms existing works in several metrics. The two modules in PILOT effectively improve the performance in different aspects.

In summary, the main contributions of this paper are as follows

• We highlight the issue of Temporal Pattern Shift in legal AI tasks. This problem is important but is usually ignored in most previous works.

• We propose a new method, PILOT, which can effectively handle Temporal Pattern Shift based on characteristics of the case law system.

• We contribute a new dataset, ECHR2023, for legal case outcome prediction. PILOT achieves state-of-the-art performance on ECHR2023.

## 2 Related Work

Legal Case Outcome Prediction on civil law framework has been well studied, mainly focusing on predicting whether the case description violates existing legislation. Machine learning technologies, including multi-task learning (Feng et al., 2022), few-show learning (Hu et al., 2018; He et al., 2019) has been adopted. Model explanation has been another focused (Jiang et al., 2018; Zhong et al., 2020; Ge et al., 2021; Chen et al., 2019; Ye et al., 2018; Wu et al., 2020).

In contrast, for case law systems that heavily relies on judicial decisions of relevant precedent cases rather than solely on constitutional law when rendering final case outcomes, there are relatively few studies due to the lack of scarcity of large-scale, high-quality, and structured labeled data. For instance, (Chalkidis et al., 2019a) utilize HIER-BERT to first encode individual facts and then employ two layers of transformers to encode all the facts within a given case. (Chalkidis et al., 2021) generate rationales through a text encoder sub-network that reads the text, a rationale extraction sub-network that identifies the most important words via a binary mask, and a prediction sub-network that classifies a hard-masked version of the text. They also incorporate rationale constraints as regularizers. (Paul et al., 2020) employ a fact encoding layer to encode facts and a charge encoding layer to encode charges. Subsequently, they use a Matching Layer, which incorporates an attention mechanism, to predict the final charges for each case. (Malik et al., 2021) utilize a Hierarchical XLNet architecture to predict case outcomes and related interpretations. These efforts primarily focus on the classification of fundamental case outcomes. To the best of our knowledge, most of the existing works do not handle temporal pattern shift.

Temporal Pattern Shift arises due to changes in label distribution, meaning, and etc. Existing research approaches this issue from different angles. For example, (Zhao et al., 2022) analyzes the impact of temporal pattern shift on model explanations. (Sun et al., 2018) explored drift adaptation through transfer-based ensemble learning. Fan et al. (Fan et al., 2023) proposed to use two CONET networks to model the normalized parameters of historical and future windows separately, enabling normalization and prediction of future sequences. (Lu et al., 2023) introduced an out-of-domain representation learning approach utilizing adversarial learning to capture domain-specific segments and a domain-independent commonality representation. (Rosin and Radinsky, 2022) introduced Temporal Attention and trained a transformer-based model with additional time-based inputs. In the legal field, (Chalkidis and Søgaard, 2022) tackled temporal pattern shift in legal text classification by proposing Label-Wise Distributional Robust Optimization. This algorithm addresses temporal pattern shift stemming from class imbalance problems and enhances model robustness. However, the existing works are too general and are designed for simple scenes, so they do not perform well in adapting the more complex shift. There is still a lack of a comprehensive solution for legal models to adapt the shift in the legal field directly and naturally.

## 3 ECHR2023 Dataset

We build a novel dataset called ECHR2023 that takes the special challenges in legal case outcome prediction with case law. This dataset is derived from the most recent ECHR database. The primary focus of ECHR2023 is to investigate the issue of temporal pattern shifts in the legal domain.

Data Acquisition and Processing The data extracted from the ECHR database is of low quality and contains a substantial amount of noise. The case documents are often excessively long, surpassing 2,000 words, and may consist of text in multiple European languages. As a result, the readability and quality of the text data are poor, posing difficulties for humans in comprehending the content of the cases.

Specifically, we prompt the large language model, gpt-3.5-turbo<sup>2</sup>, to process the raw data. The prompts guide the model to focus on the primary arguments in the case and summarize them into more concise points. Therefore, the output of LLMs will not introduce new information or fabrication to a case but rather retain the important parts of the original information. We employ LLMs to summarize original legal documents with the aim of simplifying the input and concentrating on identifying temporal pattern shifts. The prompts, example input-output of the LLM, and more details in the raw data processing can be found in Appendix B. The resulting sample is described by the following attributes: case ID, title of the case, outcome decision date of the case, corresponding legal article, and text description of the case. Following the processing results, we conduct a manual review of the generated summaries to ensure their quality and to eliminate any data that is obviously incorrect.

Data Analysis Most of existing datasets are random split and ignore the temporal pattern change in the real world. We analyze the temporal pattern change in this dataset as follows: we perform outcome prediction using BERT (Devlin et al., 2019) using both the random split and chronological split. For random split (that does not consider case time), the performance of model training in Micro-F1 is 0.798, and the testing performance is 0.796. While for chronological split data split that we train the model using previous cases and test on cases that happen later, the performance of model training in Micro-F1 is 0.737 and the testing performance is 0.677, which shows the patterns learned from previous data cannot fully capture the signal in new cases.

## 4 Methodology

## 4.1 PILOT Framework

Problem 1 (Legal Case Outcome Prediction with Case Law). Given a set of n chronological ordered cases ${ \bf C } = \{ C _ { i } \} _ { i } ^ { n }$ , where each $C _ { i }$ is represented by the text description of the case, the legal case outcome prediction aims at predicting whether a new case violates any legal article in $\mathbf { V } = \{ V _ { 1 } , V _ { 2 } , . . . V _ { n } \}$ . Here $V _ { i } \in \{ 0 , 1 \} ^ { L }$ is the corresponding multi-hot label vector of the case $C _ { i }$ violated articles, and L is the total number oflaw articles. This task is a multi-label classification to decide the case $C _ { i }$ violated law articles $V _ { j }$

We propose PILOT that primarily focuses on two distinct challenges in predicting legal case outcomes with case law: effectively identifying similar precedent cases and handling temporal pattern shift of legal principles. As illustrated by Figure 1, PILOT consists of three modules: the Relevant Case Retrieval module that retrieves relevant cases as references for outcome prediction, the Case Encoder with Evidence Fusion module that uses encodes current case with fact description and relevant cases, and the Temporal Shift Mining module that is directly adapting to the temporal drift. We will now provide a detailed introduction to each of these modules.

## 4.2 Precedent Case Retrieval

In the case law system, precedent cases serve as crucial references that judges rely on when making decisions for new cases. In order to emulate this decision-making process, we develop a precedent case retrieval module that enhances case outcome prediction in two key aspects: (1) by providing augmented evidence for prediction and (2) by offering interpretability through the provision of evidence. Case Encoding We execute contrastive learning based on a pre-trained language model on case documents only from training split of dataset. We suppose that we only have case documents in training split in the database at the beginning. The yielded model is then utilized for encoding all legal case documents in the database and the current case, preparing for similarity search. Formally, the contrastive learning is performed based on InfoNCE loss (Gao et al., 2021). Without the need for annotated labels, each case is passed into the BERT model twice within a batch, resulting in two different document embeddings $H _ { i } ^ { 0 }$ and $H _ { i } ^ { 1 }$ due to the randomness of the dropout layers (Srivastava et al., 2014).

![](images/e1b7b10a6b4afe7276248e2d7526e9693322449878cf4a12605de034abcdf171.jpg)  
Figure 1: The framework of our proposed model PILOT. PILOT has three modules: Relevant Case Retrieval, Case Encoder with Evidence Fusion, and Temporal Shift Mining. The Relevant Case Retrieval module retrieves relevant cases to use as references for outcome prediction. The Case Encoder with Evidence Fusion module encodes current cases with fact descriptions and relevant cases. The Temporal Shift Mining module adapts directly to temporal drift.

Within a batch, each pair $( H _ { i } ^ { 0 } , H _ { i } ^ { 1 } )$ is positive, and all the other pairs that $( H _ { i } ^ { 0 } , H _ { j } ^ { 1 } )$ where $i \neq j$ are negative. The contrastive training objective $\ell _ { i }$ is hence defined by:

$$
\ell _ { i } = - \log \frac { e ^ { \cos ( H _ { i } ^ { 0 } , H _ { i } ^ { 1 } ) / \tau } } { \sum _ { j = 1 } ^ { N } e ^ { \cos ( H _ { i } ^ { 0 } , H _ { j } ^ { 1 } ) / \tau } } ,\tag{1}
$$

where $H _ { i }$ is the document embedding of case $C _ { i }$ $H _ { i } ^ { 0 }$ is the positive sample for $H _ { i } , \tau$ is a temperature hyperparameter, and $\cos ( H _ { i } , H _ { j } )$ measures the cosine similarity of the input embeddings.

Case Retrieval After training with Eq. (1), we encode all legal cases in the database into semantically meaningful document embeddings ${ \textbf { H } } =$ $\{ H _ { 1 } , H _ { 2 } , . . . , H _ { n } \}$ , which can be used to compute cosine similarities for case retrieval.

In this work, we put temporal constraints for the retrieval process. First, the retrieval is performed considering the timestamps of the target cases because a case cannot refer to any future cases. For a case $C _ { i } \in \mathbf { C }$ , we assign the similarity sim $( C _ { i } , C _ { j } ) = - 1$ if $i < j$ to filter out future cases from the candidate pool.

Secondly, we also take into consideration the influence of temporal pattern shifts of legal principles, as recent cases often carry higher reference value in legal decision-making. Based on this insight, we design a variant of cosine similarity equipped with a temporal decayed function as

$$
\mathrm { s i m } ( C _ { i } , C _ { j } ) = \frac { \mathrm { c o s } ( C _ { i } , C _ { j } ) } { 1 + \frac { T _ { i } - T _ { j } } { \alpha \times | { \bf C } _ { \mathrm { v a l } } | } } ,\tag{2}
$$

where $C _ { j }$ is a candidate similar case, α is temporal decayed coefficient, and $| \mathbf { C } _ { \mathrm { v a l } } |$ is the size of validation split in the dataset. We set the decaying unit as the size of the validation split because it is a time span from labeled data to the newest unlabeled data, which is also the length of validation data. When $\alpha = ( T _ { i } - T _ { j } ) / | \mathbf { C } _ { \mathrm { v a l } } |$ , the similarity score of $( C _ { i } , C _ { j } )$ will be half. As α decreases, the reference value of precedent cases will decrease faster.

## 4.3 Case Encoding with Evidence Funsion

Target Case Encoding To prepare legal case data for outcome prediction, the first step is to embed the case documents into contextualized representation. To achieve this, we preprocess the legal document text data as follows: we convert the fact list to a piece of text by replacing all carriage return characters in the text with spaces, then use BertTokenizer to conduct tokenization.

Next, the preprocessed legal document text data is passed into a pre-trained language model (PLM) for further processing. Here we choose legal-bertbase-uncased (Chalkidis et al., 2020b), which is pre-trained on different kinds of legal documents, enabling it to capture and understand the context and meaning of the text. For every case $C _ { i }$ , we pass it into the PLM and get the contextualized representation of the fact description $H _ { i } \in \mathbb { R } ^ { d _ { t } }$ , where $d _ { t }$ is the dimension of the last hidden layer in PLM. We indicate this contextualized representation $H _ { i }$ as the current case embedding $E _ { i }$

$$
E _ { i } = H _ { i } = P L M ( C _ { i } ) ,\tag{3}
$$

The PLM takes the preprocessed legal document text data as input and generates a contextualized representation of the legal case text, encapsulating the semantic and syntactic information of the legal fact description. It captures the relationships between words, phrases, and sentences, providing a rich representation of the text’s meaning within the legal context.

Evidence Fusion We use the target case $C _ { i }$ to query all cases C to retrieve the top k similar precedent cases according to similarity scores computed by Eq. (2). We draw the evidence ${ \bf R } _ { i } \ =$ $\{ R _ { 1 } , R _ { 2 } , \ldots , R _ { k } \}$ from the retrieved cases, where $R _ { j } = \{ \sin ( C _ { i } , C _ { j } ) , V _ { j } \}$ includes the case result $\check { V _ { j } } \in \{ 0 , 1 \} ^ { L }$ and the similarity score sim $( C _ { i } , C _ { j } )$ of this relevant case.

Based on the evidence $\mathbf { R } _ { i }$ retrieved from precedent cases, we build the evidence embedding $E _ { i } ^ { r }$ by:

$$
E _ { i } ^ { r } = \sum _ { j = 1 } ^ { k } \frac { e ^ { \sin ( C _ { i } , C _ { j } ) } \times V _ { j } } { \sum _ { j = 1 } ^ { k } e ^ { \sin ( C _ { i } , C _ { j } ) } } .\tag{4}
$$

where $E _ { i } ^ { r } \in \mathbb { R } ^ { L }$ . We concatenate current case embedding $E _ { i } ^ { c }$ with relevant case embedding $E _ { i } ^ { r }$ to get the input of the linear classifier layer for $C _ { i }$ by:

$$
E _ { i } = [ E _ { i } ^ { c } , E _ { i } ^ { r } ] ,\tag{5}
$$

where $E _ { i } \in \mathbb { R } ^ { d _ { t } + L }$

This approach allows the model to learn the relationship between relevant cases, leading to a better understanding of the factors influencing case outcomes. Moreover, it helps alleviate the impact of temporal pattern shift by providing a local perspective that captures the evolving nature of legal precedents.

## 4.4 Outcome Prediction with Temporal Pattern Mining

To further mitigate the temporal pattern drift when the model makes outcome predictions, we introduce a drift prediction module that mines the effect

of timestamps to the final outcomes:

$$
\mathrm { D r i f t } _ { i } = \mathrm { M L P } ( T _ { i } ) ,\tag{6}
$$

where $\mathrm { D r i f t } _ { i } \in \mathbb { R } ^ { d }$ . MLP is a two-layer multilayer perceptron, and the dimension of the hidden layer is d. We add the output Drift<sub>i</sub> to the original prediction to get the final prediction:

$$
y _ { i } ^ { \mathrm { f i n a l } } = y _ { i } ^ { \mathrm { o r i g } } + \mathrm { D r i f t } _ { i } ,\tag{7}
$$

where $y _ { i } ^ { \mathrm { o r i g } }$ is original output generated by the classifier and $y _ { i } ^ { \mathrm { f i n a l } } \in \mathbb { R } ^ { L }$ . The drift prediction module explicitly incorporates a global view by adapting to the temporal concept and learning from the entire timeline. By considering the evolution of legal precedents over time, this module effectively captures and adapts to the changes in the legal landscape, ensuring that the model remains robust and accurate in predicting case outcomes.

## 4.5 Training and Loss Function

In addition to the binary cross-entropy loss $\mathcal { L } _ { B C E }$ used for the multi-label classification task, we add the drift loss $\mathcal { L } _ { \mathrm { D r i f t } }$ to the model loss function. $\mathcal { L } _ { \mathrm { D r i f t } }$ uses mean squared error loss to calculate the drift distance between original predictions and final predictions. The loss function of this model is defined as:

$$
\mathcal { L } = ( 1 - \lambda ) \mathcal { L } _ { B C E } + \lambda \sum _ { i = 1 } ^ { L } ( y _ { i } ^ { \mathrm { f i n a l } } - y _ { i } ^ { \mathrm { o r i g } } ) ^ { 2 } ,\tag{8}
$$

where $\lambda$ is the weight that balances the two losses.

## 5 Experiments

In this section, we conducted extensive experiments to show the performance of PILOT associated with more in-depth analysis. Universally, we report the average results of all models obtained by five runs with different random seeds, to ensure fair comparison. We use four metrics to evaluate the legal case outcomes: micro-F1, micro-Jaccard, micro-PR-AUC, and micro-ROC-AUC. More training details can be found in Appendix A.

As for the availability of cases during training and evaluation, we strictly ensure that we do not use any later cases as references for the current case. During the training phase, all prior cases from the training set are available as precedents. At test time, all prior cases from both the training and test sets are available. In the contrastive learning of the case encoding model, we only use data from the training split of the dataset.

<table><tr><td>Method</td><td>F1</td><td>Jaccard</td><td>PR-AUC</td><td>ROC-AUC</td></tr><tr><td>BERT</td><td>0.675±0.005</td><td>0.509±0.005</td><td>0.498±0.004</td><td>0.795±0.011</td></tr><tr><td>HIER-BERT</td><td>0.680±0.008</td><td>0.516±0.009</td><td>0.502±0.011</td><td>0.803±0.004</td></tr><tr><td>BERT-LWAN</td><td>0.655±0.012</td><td>0.488±0.014</td><td>0.477±0.009</td><td>0.782±0.017</td></tr><tr><td>EPM-base</td><td>0.657±0.012</td><td>0.490±0.013</td><td>0.482±0.014</td><td>0.781±0.006</td></tr><tr><td>BERT+CL+kNN</td><td>0.679±0.006</td><td>0.514±0.007</td><td>0.502±0.006</td><td>0.793±0.015</td></tr><tr><td>BERT+TemporalAttention</td><td>0.648±0.009</td><td>0.480±0.010</td><td>0.459±0.012</td><td>0.791±0.008</td></tr><tr><td>LWDROV2</td><td>0.694±0.013</td><td>0.531±0.015</td><td>0.511±0.016</td><td>0.830±0.011</td></tr><tr><td>ChatGPT 5-shots</td><td>0.442</td><td>0.284</td><td>0.267</td><td>0.818</td></tr><tr><td>PILOT (Ours)</td><td>0.715±0.008</td><td>0.557±0.010</td><td>0.543±0.014</td><td>0.831±0.007</td></tr></table>

Table 1: Experimental results. The best results are in bold. PILOT significantly outperforms all other methods in al metrics. represents standard deviation from five results of five different seeds.

## 5.1 Baselines

We consider the following baselines in evaluation.

• BERT (Devlin et al., 2019) is a transformerbased (Vaswani et al., 2017) language model pretrained on large-scale web texts. We fine-tune and predict with the [CLS] token of BERT.

• HERT-BERT (Chalkidis et al., 2019b) is a hierarchical version of BERT. This model was proposed to predict legal judgment for long documents by first splitting and encoding raw law documents into multiple sentence embeddings, then fusing them with a two-layer Transformer model (Vaswani et al., 2017) to yield the document embeddings.

• BERT-LWAN (Chalkidis et al., 2020a) is Label-Wise Attention Network after BERT that was shown to be robust in multi-label classification. LWAN employs L attention for L labels to learn the semantics of label interpretation.

• EPM-base (Feng et al., 2022) is the variant of the state-of-the-art method on the CAIL2018 dataset. The original model, named Event-based Prediction Model (EPM) targets Chinese legal case outcome prediction, augmented by extra annotations about the legal event information. We remove the event extraction module in our experiments for fair comparison and refer the method to the name EPM-base.

• BERT+CL+kNN (Su et al., 2022) is an advanced method for general purpose multi-label prediction. It is equipped with a k-nearestneighbor model along with a multi-label contrastive learning objective for better multi-label classification performance.

• BERT+TemporalAttention (Rosin and Radinsky, 2022) adds a time-aware self-attention module to the transformer model, which demonstrates superior performance in capturing temporal patterns when making predictions. In detail, it adds a time matrix to the attention weight to learn the impact of the temporal shift.

• LWDROV2 (Chalkidis and Søgaard, 2022) was proposed for legal text classification tasks. It employs Label-Wise Distributional Robust Optimization to mitigate class imbalance and temporal pattern shift problems.

• ChatGPT 5-shots (Ouyang et al., 2022) is based on the in-context learning capability of the GPT-3.5-turbo model. To be specific, we put the exemplar cases and their outcomes retrieved using our Precedent Case Retrieval module into the context, then prompt the language model to generate the outcome predictions.

Evaluation Strategy. To ensure that future information is not used in legal case outcome predictions, we partitioned the data chronologically. As a result, the training, validation, and test data consist of 8,138, 3,000, and 3,000 instances, respectively, ensuring a preserved time span between the sets. In addition, this chronological split enables the evaluation of models’ adaptability to concept drift and reinforces temporal coherence. The dataset provides a substantial amount of validation and test data, contributing to its superior evaluation capabilities for legal case outcome prediction compared to existing alternatives. The statistics of the case outcomes are summarized in Table 2.

## 5.2 Result: Legal Case Outcome Prediction

We report the main results of legal case outcome predictions in Table 1. From the table, we observe that our method outperforms other methods by a large margin in four metrics, especially over the methods that do not explicitly consider the temporal pattern shifts in legal case outcomes.

<table><tr><td>ECHR Articles</td><td>Train</td><td>Dev.</td><td>Test</td></tr><tr><td>Right to life Prohibition of torture Right to liberty and security Right to a fair trial</td><td>432 1,048 1,264 4,969</td><td>180 796 608</td><td>188 835 690</td></tr><tr><td>No punishment without law</td><td>32</td><td>1,165 7</td><td>1,081 9</td></tr><tr><td>Right for private and family life</td><td>682</td><td>287</td><td>421</td></tr><tr><td>Freedom of religion</td><td>43</td><td>17</td><td></td></tr><tr><td>Freedom of expression</td><td></td><td></td><td>26</td></tr><tr><td>Freedom of assembly</td><td>313</td><td>151</td><td>194</td></tr><tr><td></td><td>104</td><td>80</td><td>148</td></tr><tr><td>Right to an effective remedy</td><td>1,202</td><td>506</td><td>520</td></tr><tr><td>Prohibition of discrimination</td><td>170</td><td>48</td><td></td></tr><tr><td>Derogation in time of emergency</td><td>4</td><td></td><td>61</td></tr><tr><td></td><td></td><td>9</td><td>10</td></tr><tr><td>Individual applications</td><td>58</td><td>46</td><td>60</td></tr><tr><td>Examination of the case</td><td>34</td><td>4</td><td>7</td></tr><tr><td>Protection of property</td><td>1,483</td><td>435</td><td>347</td></tr><tr><td>Signature and ratification</td><td>5</td><td>11</td><td>21</td></tr></table>

Table 2: Label distribution of the ECHR2023 dataset.

<table><tr><td>Method</td><td>F1</td><td>∇</td></tr><tr><td>PILOT</td><td>0.712</td><td></td></tr><tr><td>w/o relevant case retrieval</td><td>0.701</td><td>-0.011</td></tr><tr><td>w/o temporal pattern handling</td><td>0.697</td><td>-0.015</td></tr><tr><td>w/ law article semantics</td><td>0.705</td><td>-0.007</td></tr></table>

Table 3: Results of ablation study. Relevant case retrieval and temporal pattern handling bring improvement to the model respectively, while incorporating law articles semantics has a performance drop. means the performance drop comparing with the method PILOT.

In addition, our method improves the micro-F1 by 2.74% than the previous state-of-the-art method of legal outcome prediction, LWDROV2. The reason is that LWDROV2 is a general label-wise robust method that does not solve temporal shifts directly. By contrast, our method employs a timeaware drift prediction module and augments the predictions with precedent cases.

It is noteworthy that ChatGPT 5-shots exhibits lower performance when compared to other prediction models based on supervised learning. In many instances, ChatGPT refuses to provide predictions, leading to limitations in its ability to make accurate determinations. Consequently, there remains the potential for further advancements in general-purpose generative large language models for predicting legal outcomes.

## 5.3 Result: Ablation Study

We performed an ablation study to evaluate the impact of the relevant case retrieval module and the temporal pattern handling module on the overall performance of our model. Table 3 presents the results of this study, highlighting how these two modules contribute to the improvement of the base model in distinct ways.

Additionally, we explored the incorporation of law article semantics into the model, using techniques such as law side attention or similar approaches employed in previous methods. Surprisingly, our findings indicated a decrease in performance when integrating law article information into our model. This observation is supported by the results in Table 1, where both the EPM-base and BERT-LWAN models, which incorporate law article information, exhibited inferior performance compared to BERT alone. We think one reason incorporating law articles undermines the performance is that the content and interpretations of law articles change as time goes on. It will influence model prediction without considering the time factor.

## 5.4 Result: Qualitative Case Study for Case Retrieval

The relevant case retrieval module is utilized for retrieving the top k precedent cases that are relevant to the target case. In Table 4, we present an example of the retrieval results. It is evident from the table that these retrieved cases exhibit semantic relevance to the target case. Furthermore, the violated articles mentioned in the retrieved cases are closely related and encompass the violated articles of the target case, indicating a comprehensive coverage of relevant legal provisions. Therefore, it demonstrates the effect of the case retrieval process from a qualitative perspective.

## 5.5 Result: Hyperparameter Analysis for Case Retrieval Module

The relevant case retrieval module encompasses two hyperparameters. The first parameter, denoted as k, determines the number of top relevant precedent cases to be retrieved. The second parameter is the coefficient α associated with the temporal decayed function in Eq. (2). The experimental results, presented in Table 2, shed light on the impact of these hyperparameters.

From the results, we conclude that including only three reference cases can introduce noise and lead to a decrease in performance, as it fails to retrieve the correct relevant cases effectively. However, utilizing five or seven reference cases demonstrates improved robustness compared to three cases. Notably, setting the value of α to 1e10 is an extreme condition that implies the absence of temporal decay in the computation of the similarity score. The results indicate that incorporating the time-decayed function brings about some improvement over the original approach. Empirically, we find setting $\alpha \in [ 1 , 1 0 ]$ yields the optimal results.

<table><tr><td>Case</td><td>case id</td><td>main text (selected sentences)</td><td>violated articles</td><td>similarity</td></tr><tr><td>Current Case</td><td>001-199268</td><td>the applicant complained about the lack of effective remedy in domestic law</td><td>[&quot;13&quot;, &quot;6&quot;]</td><td></td></tr><tr><td>Precedent Case 1</td><td>001-195868</td><td>the applicant expressed concerns about the lack of effective remedies in domestic law</td><td>[&quot;3&quot;, &quot;13&quot;]</td><td>0.597</td></tr><tr><td>Precedent Case 2</td><td>001-189950</td><td>applicant complained about inadequate detention conditions</td><td>[&quot;3&quot;, &quot;13&quot;]</td><td>0.560</td></tr><tr><td>Precedent Case 2</td><td>001-198818</td><td>applicant complained about the excessive length of civil proceedings</td><td>[&quot;13&quot;, &quot;6&quot;]</td><td>0.421</td></tr><tr><td>Precedent Case 3</td><td>001-199269</td><td>complaint concerns the length of administrative proceedings regarding social benefits</td><td>[&quot;6&quot;]</td><td>0.380</td></tr><tr><td>Precedent Case 4</td><td>001-198820</td><td>the applicant complained about the excessive length of his pre-trial detention</td><td>[&quot;6&quot;, &quot;5&quot;]</td><td>0.364</td></tr></table>

Table 4: An example of similar case retrieval results.

![](images/1c6fad4acda45f366a9199e4d47f8e8cde7b63acf284e6490691cc1045d7c738.jpg)  
Figure 2: Hyperparameter analysis of k and α in the relevant case retrieval module. When k equals 5 and α equals 2, the model achieves the best results. When the value of α is 1e10, it indicate an extreme condition that implies the absence of temporal decay in the computation of the similarity score

## 5.6 Result: Hyperparameter Analysis for Training Objective

To assess the impact of varying drift loss weights (λ), we conducted evaluations using different values. The results are presented in Figure 3. It is evident from the table that the inclusion of the drift loss contributes to improved model training and overall performance. Notably, the best value for λ, which balances the weighting between <sub>BCE</sub> and <sub>Drift</sub>, is found to be 0.10. The λ value of 0 indicates the exclusion of the drift loss from the model.

![](images/78e005dd4022feb14cc44037eaf35fc69c68d45a272a768fa34759241dcca0e8.jpg)  
Figure 3: Hyperparameter analysis of lambda which is the weight of drift loss. When λ equals 0.10, the model achieves the best results.

Conversely, assigning a large value to λ can have a detrimental effect on the model’s performance.

## 6 Conclusion

In conclusion, this paper introduces the PILOT model to tackle the challenges associated with predicting case outcomes in case law systems. Through our experiments, we have demonstrated the superior accuracy of our model in predicting case outcomes compared to existing methods. This improvement can be attributed to the identification of similar cases and the effective handling of temporal pattern changes.

Moreover, our proposed model goes beyond enhancing the accuracy of legal case predictions. It also offers valuable insights into legal reasoning and the evolution of legal principles. Precedent cases hold significant importance within the case law legal framework. It is worth noting that many previous works have primarily focused on the civil law system, which differs from the case law system. By analyzing and leveraging precedent cases, our model provides a deeper understanding of the underlying legal principles and their application.

## Limitations

Deciding the outcome of legal cases is a very complex process in the real world. In this paper, we simplify many settings in real court scenarios to facilitate our research. The proposed model PILOT is a preliminary work in legal case outcome prediction, which might serve as a baseline for future investigation. The goal of designing the PILOT model is to highlight and alleviate the temporal pattern shift. There are many bias problems that need to be eliminated, and the model needs better interpretability to give reliable outcomes. It cannot be applied in the real world directly. Here are some ways to enhance the capability of PILOT before its application:

• More factors should be considered when designing a precedent case retrieval module. Currently, relevant cases are determined based on semantic similarity alone. However, relevant cases may not always be entirely semantically similar. Additionally, differences in factual details among cases can lead to different legal outcomes. Therefore, a more robust retrieval module with more retrieval factors should be developed if PILOT is to be applied in real-world scenarios.

• We need to further eliminate bias issues of PILOT before applied in real life.

• The model should prioritize better interpretability in order to provide reliable outcomes, given the need for transparency in the legal domain. For example, we can add a generation module let PILOT generate some explanation of its judgement.

• Legal outcomes should not be determined by a single model alone. Instead, a Mixture-Of-Experts approach can be employed, utilizing multiple instances of PILOT with varying hyperparameters, to perform ensemble learning and generate diverse results. After a voting process, the results can be more impartial.

• The model can benefit from incorporating more information from the case. Currently, only the factual section of the case is utilized, but additional information could be included to improve the model’s performance.

## Ethics Statement

Accuracy and Transparency. We are committed to ensuring the accuracy of our predictions to the best of our abilities. We will maintain transparency about the methodologies, data sources, and algorithms used in our prediction models. We understand the profound implications of our work and strive to prevent any potential harm caused by inaccurate predictions.

Fairness and Impartiality. We pledge and strive to ensure our prediction models do not perpetuate or amplify any form of bias or discrimination. We will regularly audit our models to detect and mitigate any unfair bias, ensuring our predictions are objective and impartial.

Respect for Privacy and Confidentiality. We will strictly adhere to all applicable laws and regulations concerning data privacy and confidentiality. We will only use data that has been lawfully and ethically obtained, ensuring the privacy of all individuals involved is respected.

Accountability. We acknowledge our responsibility for the predictions made by our models. We will continually monitor and refine our models to ensure their reliability and validity.

Legal Compliance. We understand the significance of legal regulations and standards in our work. We will ensure full compliance with all relevant legal and professional guidelines in our legal outcome prediction task.

## References

Mohammad Alali, Shaayan Syed, Mohammed Alsayed, Smit Patel, and Hemanth Bodala. 2021. JUSTICE: A benchmark dataset for supreme court’s judgment prediction. CoRR, abs/2112.03414.

Ilias Chalkidis, Ion Androutsopoulos, and Nikolaos Aletras. 2019a. Neural legal judgment prediction in English. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4317–4323, Florence, Italy. Association for Computational Linguistics.

Ilias Chalkidis, Ion Androutsopoulos, and Nikolaos Aletras. 2019b. Neural Legal Judgment Prediction in English. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4317–4323, Florence, Italy. Association for Computational Linguistics.

Ilias Chalkidis, Manos Fergadiotis, Sotiris Kotitsas, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. 2020a. An Empirical Study on Large-Scale Multi-Label Text Classification Including Few and Zero-Shot Labels. In Proceedings of the 2020

Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7503–7515, Online. Association for Computational Linguistics.

Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. 2020b. LEGAL-BERT: The muppets straight out of law school. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 2898– 2904, Online. Association for Computational Linguistics.

Ilias Chalkidis, Manos Fergadiotis, Dimitrios Tsarapatsanis, Nikolaos Aletras, Ion Androutsopoulos, and Prodromos Malakasiotis. 2021. Paragraph-level rationale extraction through regularization: A case study on European court of human rights cases. In Proceedings ofthe 2021 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 226–241, Online. Association for Computational Linguistics.

Ilias Chalkidis and Anders Søgaard. 2022. Improved multi-label classification under temporal concept drift: Rethinking group-robust algorithms in a labelwise setting. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pages 2441–2454, Dublin, Ireland. Association for Computational Linguistics.

Huajie Chen, Deng Cai, Wei Dai, Zehui Dai, and Yadong Ding. 2019. Charge-based prison term prediction with deep gating network. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 6362–6367, Hong Kong, China. Association for Computational Linguistics.

Junyun Cui, Xiaoyu Shen, Feiping Nie, Zheng Wang, Jinglong Wang, and Yulong Chen. 2022. A Survey on Legal Judgment Prediction: Datasets, Metrics, Models and Challenges. ArXiv:2204.04859 [cs].

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Wei Fan, Pengyang Wang, Dongkun Wang, Dongjie Wang, Yuanchun Zhou, and Yanjie Fu. 2023. Dishts: A general paradigm for alleviating distribution shift in time series forecasting. In Proceedings ofthe AAAI conference on artificial intelligence.

Yi Feng, Chuanyi Li, and Vincent Ng. 2022. Legal Judgment Prediction via Event Extraction with Constraints. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics

(Volume 1: Long Papers), pages 648–664, Dublin, Ireland. Association for Computational Linguistics.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Jidong Ge, Yunyun Huang, Xiaoyu Shen, Chuanyi Li, and Wei Hu. 2021. Learning fine-grained fact-article correspondence in legal cases. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3694–3706.

Congqing He, Li Peng, Yuquan Le, Jiawei He, and Xiangyu Zhu. 2019. Secaps: A sequence enhanced capsule model for charge prediction. In Artificial Neural Networks and Machine Learning – ICANN 2019: Text and Time Series: 28th International Conference on Artificial Neural Networks, Munich, Germany, September 17–19, 2019, Proceedings, Part IV, page 227–239, Berlin, Heidelberg. Springer-Verlag.

Zikun Hu, Xiang Li, Cunchao Tu, Zhiyuan Liu, and Maosong Sun. 2018. Few-shot charge prediction with discriminative legal attributes. In Proceedings of the 27th International Conference on Computational Linguistics, pages 487–498, Santa Fe, New Mexico, USA. Association for Computational Linguistics.

Xin Jiang, Hai Ye, Zhunchen Luo, WenHan Chao, and Wenjia Ma. 2018. Interpretable rationale augmented charge prediction system. In Proceedings ofthe 27th International Conference on Computational Linguistics: System Demonstrations, pages 146–151, Santa Fe, New Mexico. Association for Computational Linguistics.

Ilya Loshchilov and Frank Hutter. 2017. Fixing weight decay regularization in adam. CoRR, abs/1711.05101.

Wang Lu, Jindong Wang, Xinwei Sun, Yiqiang Chen, and Xing Xie. 2023. Out-of-distribution representation learning for time series classification. In The Eleventh International Conference on Learning Representations.

Vijit Malik, Rishabh Sanjay, Shubham Kumar Nigam, Kripabandhu Ghosh, Shouvik Kumar Guha, Arnab Bhattacharya, and Ashutosh Modi. 2021. ILDC for CJPE: Indian legal documents corpus for court judgment prediction and explanation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4046–4062, Online. Association for Computational Linguistics.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.

2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library.

Shounak Paul, Pawan Goyal, and Saptarshi Ghosh. 2020. Automatic charge identification from facts: A few sentence-level charge annotations is all you need. In Proceedings of the 28th International Conference on Computational Linguistics, pages 1011–1022, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Guy D. Rosin and Kira Radinsky. 2022. Temporal Attention for Language Models. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 1498–1508, Seattle, United States. Association for Computational Linguistics.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. Dropout: A Simple Way to Prevent Neural Networks from Overfitting.

Xi’ao Su, Ran Wang, and Xinyu Dai. 2022. Contrastive Learning-Enhanced Nearest Neighbor Mechanism for Multi-Label Text Classification. In Proceedings ofthe 60th Annual Meeting ofthe Association for Computational Linguistics (Volume 2: Short Papers), pages 672–679, Dublin, Ireland. Association for Computational Linguistics.

Yu Sun, Ke Tang, Zexuan Zhu, and Xin Yao. 2018. Concept drift adaptation by exploiting historical knowledge. IEEE transactions on neural networks and learning systems, 29(10):4822–4832.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-Art Natural Language Processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Yiquan Wu, Kun Kuang, Yating Zhang, Xiaozhong Liu, Changlong Sun, Jun Xiao, Yueting Zhuang, Luo Si, and Fei Wu. 2020. De-biased court’s view generation with causality. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 763–780, Online. Association for Computational Linguistics.

Hai Ye, Xin Jiang, Zhunchen Luo, and Wenhan Chao. 2018. Interpretable charge predictions for criminal cases: Learning to generate court views from fact descriptions. In Proceedings ofthe 2018 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1854–1864, New Orleans, Louisiana. Association for Computational Linguistics.

Zhixue Zhao, George Chrysostomou, Kalina Bontcheva, and Nikolaos Aletras. 2022. On the impact of temporal concept drift on model explanations. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2022, pages 4039–4054, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Haoxi Zhong, Yuzhong Wang, Cunchao Tu, Tianyang Zhang, Zhiyuan Liu, and Maosong Sun. 2020. Iteratively questioning and answering for interpretable legal judgment prediction. Proceedings of the AAAI Conference on Artificial Intelligence, 34(01):1250– 1257.

## A Training Details

In the model training, we fine-tune on legal-bertbase-uncased (Chalkidis et al., 2020b). AdamW optimizer (Loshchilov and Hutter, 2017) was used to optimize the parameters of the model during the training. We apply differential learning rates. The learning rate of the final linear classifier is set to 1e-3, while others are all set to 1e-5. The Dropout (Srivastava et al., 2014) rate after the PLM output is set to 0.2. The batch size in each training step is set to 8. In training, we set an early stop strategy with 2 epochs. We use micro-F1 as monitoring indicators in our early stop strategy. We train CaseSifter with all data 3 epochs.

The code implementation of our model is mainly written using PyTorch (Paszke et al., 2019) library, and the pre-trained model is loaded using Transformers (Wolf et al., 2020) library. In addition, model training and evaluation were conducted on one NVIDIA GeForce RTX 3090.

## B Raw Data Processing with LLMs

We use large language models (LLMs) to process raw data. The original document is lengthy and redundant. Our summarization target is the FACT section of case documents. We employ multiple regular expressions to filter out only the FACT section from the case documents and then input them into the LLM. We ensure that the input data does not contain any other parts of the case documents, which may leak information about the results. We prompt the gpt-3-5-turbo model to get output as processed data of a long document of one legal case. We utilized the default hyperparameters, setting the temperature to 1 and the repetition\_penalty to 0. The maximum sequence length of the output is set to 512 tokens to ensure compatibility with BERT.

We have tried several prompts and select prompt according to summary performance of the model. The final selected prompt is shown in Figure 4. In our prompts, we guide the model to focus on the primary arguments in the case and summarize them into more concise points. Therefore, the output of LLMs will not introduce new information or fabrication to a case but rather retain the important parts of the original information. In this case, we can minimize the problem of hallucinations caused by generative language models as much as possible. We also acknowledge that this method will cause potential semantic loss in new dataset, but it can increase the model inference speed and improve readability of original case documents.

We manually check the data quality from the LLM output. We review about dozens of samples of data. We observe that it do not introduce any new fabricated facts in the output, and indeed summarizes some key points of the case, which meets our expectations. We have also conducted experiments to compare these aspects. Our results show that using the baseline results of ChatGPT processed content only leads to a 0.5% decrease in performance than original lengthy documents, but significantly increases the training speed in later stages.

An example input and output of the LLM in data processing is shown in Figure 5.

![](images/15d89a84598fa7004a9ba4a43fc5b4e98b1268d5eaba77c3bf1489c0fbd4dfec.jpg)  
Figure 4: The final selected prompt. We also prompt model by telling you are a good judge.

![](images/7b8f91e7810e944acb91836ec38387366a7022642cd0afb9d306c11cef7e07f2.jpg)  
Figure 5: An example input and output of the LLM about data 001-187931. The original document has 3618 tokens totally. It reduces to 494 tokens after extracting important points of a legal case.