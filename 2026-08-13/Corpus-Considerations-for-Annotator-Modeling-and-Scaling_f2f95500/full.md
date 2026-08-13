# Corpus Considerations for Annotator Modeling and Scaling

Olufunke O. Sarumi†∗ and Béla Neuendorf†∗ and Joan Plepi‡ and Lucie Flek‡ and Jörg Schlötterer†¶ and Charles Welch‡

†Department of Mathematics and Computer Science, University of Marburg ‡Bonn-Aachen International Center for Information Technology (b-it), University of Bonn ¶Area Information Systems, University of Mannheim {sarumio,neuendob,joerg.schloetterer}@uni-marburg.de {plepi,flek,cfwelch}@bit.uni-bonn.de

## Abstract

Recent trends in natural language processing research and annotation tasks affirm a paradigm shift from the traditional reliance on a single ground truth to a focus on individual perspectives, particularly in subjective tasks. In scenarios where annotation tasks are meant to encompass diversity, models that solely rely on the majority class labels may inadvertently disregard valuable minority perspectives. This oversight could result in the omission of crucial information and, in a broader context, risk disrupting the balance within larger ecosystems. As the landscape of annotator modeling unfolds with diverse representation techniques, it becomes imperative to investigate their effectiveness with the fine-grained features of the datasets in view. This study systematically explores various annotator modeling techniques and compares their performance across seven corpora. From our findings, we show that the commonly used user token model consistently outperforms more complex models. We introduce a composite embedding approach and show distinct differences in which model performs best as a function of the agreement with a given dataset. Our findings shed light on the relationship between corpus statistics and annotator modeling performance, which informs future work on corpus construction and perspectivist NLP.

## 1 Introduction

An integral aspect of dataset creation is obtaining multiple annotations from annotators on the same instances of data (Zhang et al., 2021; Hayat et al., 2022). More often than not, these are aggregated through a majority vote to arrive at a single ground truth label (Wulczyn et al., 2017; Nowak and Rüger, 2010). However, an annotator’s background influences the label they assign. This divergence is often evident in the judgments and perceptions of annotators of subjective tasks. In the social media domain, where people’s reactions are influenced by their personal experiences and vested interests, relying solely on a majority label to determine or reach a consensus proves challenging (Cabitza et al., 2023). Hence, it becomes crucial to improve annotator modeling frameworks for robust user representations that capture the diverse views inherent in our datasets, while preserving individual perspectives.

Interest in data perspectivism has been growing and with it, approaches for annotator modeling (Plepi et al., 2022; Casola et al., 2023; Davani et al., 2022). The approaches for annotator modeling are built from corpora with unaggregated labels widely ranging in the number of annotators, data type and volume available per annotator, the type of task, and the magnitude of disagreement (Leonardelli et al., 2021; Kennedy et al., 2022; Demszky et al., 2020; Almanea and Poesio, 2022; Cercas Curry et al., 2021). Though a few recent works have mentioned the impact of the number of annotators or the level of agreement on annotator modeling methods (Kadasi and Singh, 2023; Deng et al., 2023; Bhowmick et al., 2008), they have not been systematically explored.

In this paper, we perform the first systematic study of the scalability of annotator modeling methods and the relationship between annotator modeling methods and corpus statistics. We implement annotator modeling and personalization techniques used in recent work (Mireshghallah et al., 2022; Welch et al., 2020b; Plepi et al., 2022) and implement our own novel composite embedding approach. This work sheds light on the effectiveness of annotator modeling methods under various realworld scenarios for subjective tasks. We provide recommendations for which methods to use based on the available data.

We find that when agreement is high, our composite embedding performs best, while when agreement is lower, the user token approach common in previous work performs best. We find that the user token approach often outperforms other more complex methods developed in previous work, including the averaged SBERT, authorship attribution embeddings (Welch et al., 2020a), and multi-tasking (Davani et al., 2022). We investigate the scalability of these methods with respect to the amount of overall data, the number of annotations per annotator, and the number of annotators in the dataset across seven different datasets and approximately 3k subsamples representing artificial datasets with controllable properties. We find that the number of annotations per annotator is the most important factor for annotator modeling, though the number of instances and annotators in the corpus both had weak but significant correlations with performance. Our code and the statistics of all trials in our experiments are publicly available<sup>1</sup> to support future work that examines the relationship between corpus statistics and annotator modeling performance.

## 2 Related Work

In the first part of this section, we provide an overview of annotator modeling approaches, from modeling annotators by numeric identifiers over multitask models to representations of both, annotator and annotation information. In the second part, we review scaling analyses, identifying the lack of a systematic comparison between annotator modeling methods and corpus statistics.

## 2.1 Annotator Modeling

The perspectivism paradigm is fueled by the observation that aggregate labels usually do not generalize to different demographic groups. Findings have shown that personalized models significantly enhance decision accuracy, emphasizing the substantial benefits of individualized tuning over global approaches (Kumar et al., 2021). Likewise, disagreement within annotation processes cannot simply be dismissed as noise and merely striving for a higher inter-annotator agreement (IAA) may not always be beneficial. This calls for robust representation techniques that prioritize the inclusion of both majority and minority perspectives across the opinion spectrum (Fleisig et al., 2023).

In datasets characterized by disagreement, many methods resort to numeric identifiers for annotator modeling due to minimal annotator information availability. Mireshghallah et al. (2022), for instance, implemented a method where they added a non-trainable string prefix to every sentence of a user’s input as a means of personalizing sentiment analysis. A simple approach is to concatenate annotator information to the instance input, which was used as a baseline by Plepi et al. (2022) and Deng et al. (2023). An improvement on this approach is seen in Plepi et al. (2022) who leveraged the work of King and Cook (2020) by concatenating randomly sampled annotator comments to input text. They also computed the averaged embeddings of previous posts by annotators to represent individual annotators using a dataset of social norms; a corpus of Reddit data.

The radical approach of training individual models per annotator Shahriar and Solorio (2023) does not scale well due to computational complexity. Davani et al. (2022) reduce the computational complexity by a multi-task approach, utilizing separate fully connected layers fine-tuned for each annotator. Vitsakis et al. (2023) applied this multitask architecture without further modifications in their submission to the learning with disagreement shared task (Leonardelli et al., 2023). This approach, however, was only viable for datasets with few annotators such as HS-Brexit (Akhtar et al., 2021) and ArMIS (Almanea and Poesio, 2022), but unsuitable for MD-Agreement (Leonardelli et al., 2021). Similarly, Sullivan et al. (2023) observed that the multitasking approach struggles “to account for large or variable numbers of annotators.”

In addition to learnable annotator representations, Deng et al. (2023) trained representations for annotations. They trained compatibility matrices between the input text embedding and annotator/annotation embedding to control the influence of the latter. They observed performance improvements in particular for tasks with high disagreement and found that demographic features are not enough to model disagreement across annotators.

## 2.2 Scaling Analysis

In this context, scaling refers to the ability of an annotator modeling approach to maintain performance and optimal throughput across varying scales of data, including the number of annotators that can be represented by a model, the number of annotators per instance and the number of annotations per annotator. Previous research has investigated the impact of a single annotation (Zhang et al., 2021) to multiple annotations per instance versus increased annotation examples (Sheng et al., 2008) on performance, highlighting the importance of recognizing that there may not be a single correct interpretation for every input (Aroyo and Welty, 2015). Increasing training data size for improved generalization (Mishra and Sachdeva, 2020) has equally been explored, giving rise to increased computational cost, which led to the use of active learning for labeling in order to manage cost (Fang et al., 2017). Additionally, Wang and Plank (2023) applied the multi-task model to active learning, to mitigate computational costs arising from multiple annotations.

Few studies have explicitly examined the tradeoffs between increased annotators, annotations, and training samples. While some approaches used a fixed number of annotators consistently throughout the task (Sullivan et al., 2023), they may not be suitable for datasets where annotators vary (Cercas Curry et al., 2021). Davani et al. (2022) excluded annotators with fewer annotations in their multi-task setup due to computational constraints. Our study presents a systematic analysis using subsets of different corpora to assess the performance of models across varying numbers of annotators.

## 3 Datasets

We use seven datasets from recent work on annotator modeling. All datasets use binary labels for classification. We include four datasets from the recent SemEval-2023 task on learning with disagreements (Leonardelli et al., 2023), two datasets used by Davani et al. (2022), and the Social Chemistry dataset (Forbes et al., 2020) that was adapted for personalization and annotator modeling by Plepi et al. (2022). Dataset statistics are presented in Table 1.

Gab Hate Speech Corpus The Gab Hate Corpus (GHC) (Kennedy et al., 2022) comprises 27,665 social media posts from Gab.com annotated by a minimum of three annotators. The GHC features an extensive coding framework that encompasses hierarchical labels denoting dehumanizing and violent speech, markers indicating targeted groups, and rhetorical framing. As in previous annotator modeling work (Davani et al., 2022), we use labels indicating the presence or absence of hate speech.

GoEmotions The GoEmotions (GE) dataset (Demszky et al., 2020) has fine-grained emotions comprising 58k English Reddit comments and labeled for 27 emotion categories and a neutral label for no emotion. We focused on the six Ekman emotions (Ekman et al., 1999) from the experiments of Davani et al. (2022); anger, disgust, fear, joy, sadness, and surprise. Each post received annotations from three to five of a total of 82 annotators. The agreement varies across the Ekman emotions, with Krippendorff’s α highest at 0.35 for fear, followed by 0.29 for sadness, 0.28 for surprise, 0.27 for anger, 0.26 for joy, and 0.21 for disgust.

HS-Brexit The Hate Speech Brexit (HSB) dataset (Akhtar et al., 2021) is a dataset on abusive language detection consisting of 1,120 tweets related to Brexit and immigration. These were annotated for hate speech, aggressiveness, and offensiveness by two distinct groups of three annotators consisting of a target group of three Muslim immigrants in the UK and a control group of three individuals with a western background. In contrast to other datasets, the peculiarity of HS-Brexit lies in its utilization of only six annotators, distributing annotations across a smaller group with each annotator assessing many instances.

ConvAbuse The Conversational Abuse (CVA) dataset contains approximately 4k English dialogues between users and two conversational agents (Cercas Curry et al., 2021). Users’ conversations were annotated by at least three experts in gender studies using a hierarchical labeling scheme categorized into abuse presence, abuse severity, and directness. Similarly to HS-Brexit, ConvAbuse has only a few annotators and many annotations per annotator.

Multi-Domain The Multi-Domain (MD) Agreement dataset (Leonardelli et al., 2021) comprises 10,753 tweets from three domains: Black Lives Matter (BLM), 2020 USA Presidential Elections, and COVID-19. Each tweet was annotated for offensiveness by a group of five annotators and a total of 819 annotators were recruited through Amazon Mechanical Turk (AMT).

ArMIS The Arabic Misogyny and Sexism (ArMIS) dataset contains over 900 Arabic tweets annotated specifically for the detection of misogyny and sexism (Almanea and Poesio, 2022). This annotation was based on identifying bias in the assessment of sexism, with a focus on the varying perspectives of the annotators concerning liberality. Three distinct individuals, self-identifying as a moderate female, liberal female, and conservative male, engaged in the annotation task. The structure of the annotation task is similar to the HS-Brexit dataset.

<table><tr><td></td><td>#A</td><td>#</td><td>N</td><td>A/I</td><td>K-α</td><td>Paradigm</td></tr><tr><td>GoEmotions</td><td>82</td><td>58,012</td><td> $2 , 5 7 6 \pm 2 , 2 9 2$ </td><td> $3 . 6 4 \pm 0 . 9 4$ </td><td>0.27</td><td>Prescriptive</td></tr><tr><td>Gab Hate Speech</td><td>18</td><td>27,665</td><td> $4 , 8 0 7 \pm 3 , 1 8 5$ </td><td> $3 . 1 3 \pm 0 . 3 9$ </td><td>0.25</td><td>Prescriptive</td></tr><tr><td>Social Chemistry</td><td>2,500</td><td>18,431</td><td> $4 6 . 5 \pm 4 5 . 9$ </td><td> $6 . 3 \pm 1 3 . 3$ </td><td>0.58</td><td>Descriptive</td></tr><tr><td>MD-Agreement</td><td>819</td><td>10,753</td><td> $6 5 . 6 5 \pm 1 4 3 . 7 7$ </td><td> $5 . 0 0 \pm 0 . 0 0$ </td><td>0.36</td><td>Mixed</td></tr><tr><td>HS-Brexit</td><td>6</td><td>1,120</td><td> $1 , 1 2 0 . 0 0 { \pm } 0 . 0 0$ </td><td> $6 . 0 0 \pm 0 . 0 0$ </td><td>0.35</td><td>Prescriptive</td></tr><tr><td>ConvAbuse</td><td>8</td><td>4,050</td><td> $1 , 5 2 1 . 0 0 \pm 2 0 6 . 9 1$ </td><td> $3 . 0 0 \pm 0 . 8 8$ </td><td>0.65</td><td>Mixed</td></tr><tr><td>ArMIS</td><td>3</td><td>943</td><td> $9 4 3 . 0 0 \pm 0 . 0 0$ </td><td> $3 . 0 0 \pm 0 . 0 0$ </td><td>0.52</td><td>Descriptive</td></tr></table>

Table 1: Dataset statistics including the number of annotators (A), the number of total instances (I), the number of annotations per annotator (N), annotations per instance (A/I), the agreement as measured by Krippendorff’s alpha, and the annotation paradigm.

Social Chemistry The Social Norm (SoC) dataset (Welch et al., 2022b) is sourced from Reddit, an online platform with various communities known as subreddits. This dataset specifically utilizes posts and comments from the /r/amitheasshole (AITA) subreddit where users share social experiences and seek community opinions on the appropriateness of their behavior and that of others involved. AITA members express their views on whether the original poster is at fault in a scenario, using labels YTA (you are the asshole) and NTA (not the asshole), often providing additional reasoning. The commenters are treated as annotators and personalization methods can be applied to their comments on other parts of the subreddit or other subreddits to compute annotator representations that are not possible with most datasets that contain no more than annotator IDs (and occasionally demographic information). The dataset comprises 21k posts and 327k verdicts (229k NTA, 98k YTA) from 86k different authors. Due to the large number of annotators, we downsampled to only the top 2,500 annotators with the highest annotation counts, resulting in 18,431 instances.

In Table 1, we report the aggregated statistics for our datasets. These include the number of annotators (A), the number of total instances (I), the number of annotations per annotator (N), annotations per instance (A/I), and the agreement level across annotators quantified using Krippendorff’s alpha. We list the annotation paradigm in the last column, where a descriptive paradigm encourages, and a prescriptive discourages subjectivity (Rottger et al., 2022). The MD-Agreement and ConvAbuse datasets are classified as mixed primarily because of their annotation process. While these datasets did not include a description aligned with the prescriptive paradigm and did not explicitly encourage annotator subjectivity, the guidelines allow for some degree of subjectivity.

## 4 Methodology

In our study, we investigated five distinct methods for capturing personalized attributes within a collective context, encompassing varying levels of complexity. Our models take the annotator ID and text that was annotated as input and are designed to distinguish between unique perspectives. In addition to representing the text in the input data, the model also incorporates a representation of the annotator who provided the annotations for the text. These representations are encoded as real-valued, low-dimensional vectors. The choice of how to represent the annotator embedding depends on the specific method being used and the information available about the annotator. With the exception of Social Chemistry, the annotator tokens are the sole explicit distinguishing attribute of the annotators within the dataset. For several of our methods, annotator embeddings were obtained using special tokens and concatenated with the input strings for each text instance to link each text instance to its corresponding annotator. For the multi-tasking model, the annotator ID instead determines which layer of a multi-task model is responsible for predicting the label.

Additionally, we experimented with personalization techniques for the Social Chemistry dataset. This dataset contains additional writing for each annotator that can be used to derive annotator representations. We implemented the authorship attribution approach (Welch et al., 2022a) and averaged SBERT embeddings (Plepi et al., 2022). Note that the averaged SBERT embeddings are averaging the additional text provided by annotators (which is only available in SoC) rather than the embeddings of the texts that were assigned an annotation.

We also introduce a composite embedding approach, where all instances of annotations for each annotator were aggregated for each class. We calculated the average embeddings of the positive class and negative class and concatenated them to derive a unified representation for each annotator.

## 4.1 User Token Annotator Embedding

Consider an annotated dataset defined by $D =$ $( X , A , Y )$ where X is the set of text instances represented as $X = \{ x _ { 1 } , x _ { 2 } , . . . x _ { n } \}$ , A is the set of annotators represented as $A = \{ a _ { 1 } , a _ { 2 } , . . . a _ { k } \}$ , and $Y : X \times A \to \{ 0 , 1 \}$ is the annotation matrix, where an entry $y _ { i j }$ represents the label assigned to instance $x _ { i }$ by annotator $a _ { j }$ . Since it is typical for each annotator to assign labels to only part of the dataset instances, Y may contain many missing values. Note that for GoEmotions, we have multiple annotation matrices $Y _ { 1 } , \ldots , Y _ { 6 }$ , one for each of the the six Ekman emotions.

When using a BERT-based encoder for label classification, the first step is a transformation of the individual tokens of the input text instance $x _ { i }$ into a low-dimensional vector representation (embedding). We denote the input token embedding representation of instance $x _ { i }$ as $R _ { i } = [ w _ { 1 } , \dots , w _ { | x _ { i } | } ]$ where w is the embedding of an individual token and $\left| x _ { i } \right|$ is the number of tokens in the input. To incorporate annotator information, we extend the model’s vocabulary and append a special token to the input; the user token. Each annotator is represented by a distinct user token, serving as an identifier. In accordance with input text embeddings, the user token is represented by a learnable embedding $u _ { j }$ that is randomly initialized. Correspondingly, the input representation for training the model augmented by annotator information becomes $R _ { i j } = \left[ w _ { 1 } , \ldots , w _ { | x _ { i } | } , u _ { j } \right]$

## 4.2 Composite Embedding

The composite embedding approach involves computing two embedding averages in the context of a binary classification task; the average of all instances an annotator labeled as positive, and the average of all labeled negative. These two resulting averages represent the typical embedding patterns of the annotator when labeling positive and negative instances, respectively.

We define the average positive embedding $E _ { p }$ as the sum of all embeddings of all instances x labeled positive by annotator $a _ { j } .$ , i.e., $\{ x _ { i } | y _ { i j } =$ $1 \} _ { i = 1 } ^ { n }$ , divided by total count of positive instances by annotator $a _ { j }$

$$
E _ { p } = { \frac { \sum _ { i | y _ { i j } = 1 } x _ { i } ^ { e m b e d } } { | \{ y _ { i j } = 1 \} _ { i = 1 } ^ { n } | } }\tag{1}
$$

We obtain the embedding of an instance $x ^ { e m b e d }$ by encoding x with a pre-trained SBERT model.<sup>2</sup>

Similarly, let $E _ { n }$ denote the average embedding of all instances labeled negative by annotator $a _ { j }$ i.e., $y _ { i j } = 0$

$$
E _ { n } = { \frac { \sum _ { i | y _ { i j } = 0 } x _ { i } ^ { e m b e d } } { | \{ y _ { i j } = 0 \} _ { i = 1 } ^ { n } | } }\tag{2}
$$

Given $E _ { p }$ and $E _ { n }$ , we calculate the composite embedding for annotator $a _ { j }$ as $c _ { j } = [ E _ { p } | | E _ { n } ]$ where || denotes concatenation.

The composite embedding $c _ { j }$ is used to initialize a special token embedding representing the annotator, whereas the user token approach $u _ { j }$ is a random initialization. Intuitively, an initialization computed from all training data for a given annotator should provide a better starting point for the model.

## 4.3 Composite Embedding with User Token

This approach follows the convention as in §4.1 and §4.2 above. It utilizes the two special token embeddings; $u _ { j }$ associated with the annotator ID of $a _ { j } .$ , and $c _ { j }$ associated with the composite representation of the same annotator. Both are appended to the input text $x _ { i }$ annotated by $a _ { j }$ resulting in $R _ { i j } = \left[ w _ { 1 } , \ldots , w _ { | x _ { i } | } , u _ { j } , c _ { j } \right]$ to model the annotator. This approach uses both a randomly initialized user token embedding and the composite embedding, which are updatable during training.

## 4.4 Multi-task

The multi-tasking model is implemented on top of a BERT-base model as in previous work. One linear prediction layer is added for each annotator. The loss is summed over all annotators for a given instance. We tuned the learning rate of the multi-tasking model on the validation sets and used $1 e ^ { - 5 }$ for subsequent experiments. This model has more parameters dedicated to each annotator, so we hypothesized that it would outperform the other annotator methods. We expect to see a trade-off between the cost in time and model complexity versus the improvement in annotator modeling.

<table><tr><td>Method</td><td>GE</td><td>GHC</td><td>SoC</td><td>MD</td><td>HSB</td><td>CVA</td><td>ArMIS</td></tr><tr><td>SBERT</td><td>68.6</td><td>68.5</td><td>53.3</td><td>73.0</td><td>68.6</td><td>85.9</td><td>61.7</td></tr><tr><td>User Token</td><td>70.2</td><td>76.5</td><td>58.5</td><td>77.7</td><td>77.6</td><td>88.5</td><td>62.1</td></tr><tr><td>Composite Embed (Ours)</td><td>68.2</td><td>68.2</td><td>58.6</td><td>73.1</td><td>67.6</td><td>85.8</td><td>61.4</td></tr><tr><td>Composite+User Token (Ours)</td><td>70.0</td><td>76.4</td><td>60.4</td><td>77.5</td><td>77.3</td><td>88.6</td><td>62.5</td></tr><tr><td>Multi-task</td><td>68.3</td><td>70.5</td><td>53.5</td><td>75.7</td><td>71.7</td><td>82.3</td><td>56.6</td></tr></table>

Table 2: Full dataset result F1 scores on the individual annotator labels for each annotator representation method and dataset.
<table><tr><td>Method</td><td>Anger</td><td>Disgust</td><td>Fear</td><td>Joy</td><td>Sadness</td><td>Surprise</td></tr><tr><td>SBERT</td><td>67.9</td><td>64.3</td><td>71.1</td><td>69.2</td><td>69.7</td><td>70.2</td></tr><tr><td>User Token</td><td>69.6</td><td>66.9</td><td>73.1</td><td>69.9</td><td>70.6</td><td>71.1</td></tr><tr><td>Composite Embed (Ours)</td><td>66.3</td><td>63.6</td><td>71.5</td><td>69.0</td><td>68.9</td><td>69.8</td></tr><tr><td>Composite+User Token (Ours)</td><td>69.1</td><td>66.2</td><td>72.3</td><td>69.8</td><td>70.5</td><td>71.6</td></tr><tr><td>Multi-task</td><td>67.8</td><td>64.0</td><td>70.4</td><td>68.9</td><td>68.8</td><td>69.6</td></tr></table>

Table 3: GoEmotions Ekman emotion F1 scores on the individual annotator labels for each method.

## 4.5 Personalization Methods

We implement the authorship attribution and averaged SBERT embeddings used in previous work for the Social Chemistry dataset. The averaged SBERT embeddings are computed for a given annotator by taking the set of texts that annotator has written independently of their annotations, encoding them with SBERT, and subsequently averaging the representations.

The authorship attribution method is implemented by training an authorship attribution classifier over the text of all annotators. We first embed the text with SBERT and forward these encodings to a two-layer feed-forward network. The output of the last linear layer provides a distribution over all annotators for predicting the author of the text. For each annotator, we use all of their texts from the training set (each label is accompanied with text) and pass them to the classifier. Intuitively, an annotator that is more often confused with another annotator should be more similar to that annotator.

## 5 Experimental Setup

First, following Plepi et al. (2022), we implemented a text-only baseline that also serves as the base model, which we extend by different methods for annotator modeling. Specifically, we used SBERT (Reimers and Gurevych, 2019), a consistently highperforming BERT-based model trained to encode sentences for a variety of downstream tasks. We used a pre-trained<sup>3</sup> SBERT model with the DistilRoBERTa (Sanh et al., 2019) backbone, which features a 768-dimensional representation and a maximum sequence length of 512 tokens. On top of the text encoding provided by SBERT, we implemented a classification head and fine-tuned the model for binary classification.

For the multi-task model, we used the same setup as Davani et al. (2022) who used a BERT model with separate output layers for each annotator (Devlin et al., 2019). Lastly, since all of our datasets are in English except ArMIS, which is in Arabic, we used the Arabic BERT model from Safaya et al. (2020), which was trained on a combination of data from Wikipedia and Common Crawl.

To evaluate, we used the macro F1 score across individual annotator’s labels. We trained our models for 10 epochs, employing early stopping based on the validation set performance. The Adam optimizer was used, with an initial learning rate set at 2e−<sup>5</sup>. We split the data into 80% train and 10% for each of validation and test with the same annotators in all splits. Our experiments were conducted on a single NVIDIA A100 40GB GPU. The average running time for both training and inference phases combined was around 15 minutes per model.

## 6 Results

Individual Annotators In Tables 2 and 3 we report the F1 score for all our models across different datasets. Our analysis reveals a consistent trend across our personalized models: performance generally improves with a smaller number of annotators and a high agreement rate. Our models are performing the worst in the Social Chemistry dataset, which contains the highest number of annotators (triple that of the dataset with the next highest count). In contrast, in the GoEmotions dataset, our models underperform compared to the MD-Agreement model, despite having fewer annotators. This can be attributed to two key differences between both datasets: a significant variance in the number of annotations per annotator and a lower overall agreement rate among annotators. Our models achieve the best performance in the ConvAbuse dataset, characterized by a smaller pool of annotators and a higher rate of agreement. This pattern suggests that both the number of annotators and the level of agreement among them are pivotal factors influencing model performance. This correlation is further observed in the GoEmotions dataset, as detailed in Table 3. For instance, our models perform poorly in the disgust category, which coincides with the lowest annotator agreement.

For Social Chemistry, we also calculated the authorship attribution and averaged embeddings scores. The F1 for authorship attribution was 56.7 and for averaged embeddings was 57.1, both underperforming our composite embedding plus user token. However, these results are comparable with the 56 F1 reported by Plepi et al. (2022) for the situation split (the same splitting method we use), even though we are using 10k fewer annotators.

Majority Label We presented results using the F1 scores of the individual annotator labels as our primary concern is with annotator modeling. However, it is interesting to note that the methods that perform best on annotator modeling are not the same as those that perform best when aggregating the results of individual predictions to the majority label. To obtain these results we take the individual annotator labels predicted by the model for a given instance and take the most frequent label as the predicted majority label. This is compared to the gold majority label and the resulting F1 scores are shown for the full datasets in Table 4 and for each Ekman emotion of GoEmotions in Table 5.

For the majority labels, we find that the composite embedding combined with the user token improves performance on GHC as well as Disgust. The performance on ArMIS still outperforms other annotator modeling methods, but the SBERT baseline appears to perform better for ArMIS, as well as HS-Brexit and Fear. For two emotions, Anger and Joy, we see the multi-task model outperform other methods, while it is never the best at predicting individual labels. For Social Chemistry, the authorship attribution method achieved an F1 of 57 and the averaged SBERT embeddings scored 56.

For the majority vote, it is interesting to note that the text-only baseline is sometimes the best model. However, we know from Tables 2 and 3 that the baseline is more often getting the individual annotator labels incorrect. This finding supports similar findings indicating that the best model of the majority class often marginalizes the voice of minority annotators (Sap et al., 2019; Fleisig et al., 2023), which can be particularly harmful, for instance, in cases such as when racial bias impacts the perception of hate speech (Sap et al., 2022).

## 7 Scaling Up

Subsequently, to further test the impact of dataset statistics reported in Table 1, we created subsets of our datasets by scaling the number of annotations per annotator, and the number of annotators.

When scaling the number of annotators, the three smallest datasets were scaled in increments of one annotator. For the other four, we scaled in increments of two annotators for the range of 6 to 18 annotators; the upper limit for GHC. Then for GoEmotions and Social Chemistry we scaled from 18 to 82 annotators in increments of 4. Lastly, we scaled Social Chemistry from 100 to 2.5k in increments of 100. This is repeated for each method and Ekman emotion across five runs. We tested each method on each subset, which yields a total of 1,670 trials using artificially constructed datasets.

Figure 1 shows each method for the GoEmotions dataset when scaling the number of annotators from 6 to 82. The baseline SBERT method is marked by a dashed line. User token performed best overall and even performs strongly when the number of annotators and amount of data is low. The composite embedding and multi-tasking methods perform poorly in this setting but approach the performance of other methods as the amount of data increases.

<table><tr><td>Method</td><td>GE</td><td>GHC</td><td>SoC</td><td>MD</td><td>HSB</td><td>CVA</td><td>ArRMIS</td></tr><tr><td>SBERT</td><td>67.0</td><td>67.5</td><td>51.6</td><td>79.8</td><td>72.7</td><td>87.6</td><td>65.3</td></tr><tr><td>User Token</td><td>67.4</td><td>69.7</td><td>58.7</td><td>80.6</td><td>72.2</td><td>88.8</td><td>62.1</td></tr><tr><td>Composite Embed (Ours)</td><td>65.5</td><td>69.2</td><td>56.7</td><td>80.1</td><td>71.9</td><td>87.5</td><td>63.0</td></tr><tr><td>Composite+User Token (Ours)</td><td>66.6</td><td>69.8</td><td>59.9</td><td>80.5</td><td>70.4</td><td>89.3</td><td>63.6</td></tr><tr><td>Multi-task</td><td>66.7</td><td>66.7</td><td>49.5</td><td>76.4</td><td>70.2</td><td>83.1</td><td>55.3</td></tr></table>

Table 4: Full dataset result F1 scores for each annotator representation method and dataset using the F1 score of predicting the majority class label rather than the individual annotator labels.
<table><tr><td>Method</td><td>Anger</td><td>Disgust</td><td>Fear</td><td>Joy</td><td>Sadness</td><td>Surprise</td></tr><tr><td>SBERT</td><td>67.5</td><td>62.6</td><td>73.2</td><td>63.3</td><td>67.5</td><td>67.6</td></tr><tr><td>User Token</td><td>67.5</td><td>63.3</td><td>72.5</td><td>64.8</td><td>68.0</td><td>68.4</td></tr><tr><td>Composite Embed (Ours)</td><td>69.4</td><td>61.0</td><td>64.9</td><td>63.4</td><td>66.9</td><td>67.2</td></tr><tr><td>Composite+User Token (Ours)</td><td>66.8</td><td>63.5</td><td>72.2</td><td>64.0</td><td>66.2</td><td>66.7</td></tr><tr><td>Multi-task</td><td>70.3</td><td>62.0</td><td>70.9</td><td>67.8</td><td>62.6</td><td>66.7</td></tr></table>

Table 5: GoEmotions Ekman emotion F1 scores for each annotator representation method using the F1 score of predicting the majority class label rather than the individual annotator labels.

GoEmotions: F1 Macro Individual Annotators  
![](images/4daeb581d9039cb82fade4ccb0deb4a53260a838972dd20fc3e047373347a07d.jpg)  
Figure 1: GoEmotions mean performance across emotions when scaling the number of annotators. The SBERT baseline is indicated by the dashed line. Shaded regions correspond to 95% confidence intervals.

To identify trends across these three variables we looked at the correlation with performance across all trials. We split the data into those with greater than 18 annotators (the median value across our corpora) and those with 18 or fewer. For each model, we calculate the percentage of relative improvement in the F1 measure over the baseline and measure the correlation using Pearson’s coefficient. Since we scaled each dataset based on their available annotations and annotators to examine corpora specific patterns, this resulted in an unequal distribution of trials across datasets. For this analysis, to avoid imbalance of dataset influence, we sampled

60 trials from each of the seven corpora.

When there are more than 18 annotators, there are no significant correlations. However, when we have 18 or fewer, we find that there is a significant correlation with the dataset size $( R = 0 . 1 8 , p <$ 0.0005), number of annotators $( R = 0 . 1 6 , p <$ 0.001), and most significantly with the number of annotations $( R = 0 . 4 2 , p < 0 . 0 0 0 1 )$ .

Subsequently, we decided to look at the impact of the number of annotations per annotator on performance. We ran experiments on all datasets when fixing the number of annotators to the maximum number for ArMIS, ConvAbuse, and HS-Brexit. For the others, we fixed the number of annotators to 14 and lastly, for GoEmotions, MD Agreement, and Social Chemistry we also ran trials with 50 annotators. We varied the number of annotations per annotator in increments of 10% up to the minimum number of annotations per annotator in each set of annotators for a given dataset. This resulted in an additional 1,260 trials, which we sampled as in the previous analysis. When examining the relationship between performance and the number of annotations per annotator, our correlation coefficient was $R = 0 . 4 7 ( p < 0 . 0 0 0 1 )$ . The samples shown in Figure 2 are from the best performing method on each dataset according to Table 2. We see that the number of annotations per annotator is important, but on most datasets the improvement levels off after a couple hundred annotations.

![](images/913abb4e350b2358e28d19a8d7fee898a030867530430704df7f715b037df509.jpg)  
Figure 2: Relative performance increase in F1 as a function of the number of annotations per annotator.

## 8 Discussion

When examining the dataset statistics and performance, we notice that the user token method performs well when the Krippendorff alpha is relatively low $( \mathrm { K } { - } \alpha < 0 . 4 )$ . For the Social Chemistry, ConvAbuse, and ArMIS datasets, the agreement is higher $( \mathrm { K } { - } \alpha > 0 . 5 )$ and we find that in these cases the composite embedding boosts performance, becoming the best model. This may be because when agreement is higher, the composite embedding is more informative as an initialization of the annotator representation. When annotators agree more with each other, the aggregate of their positive and negative labels is more generalizable, as opposed to the random initialization of the user token.

The largest boost provided by the composite embedding occurred in the Social Chemistry dataset, where we have the lowest number of annotations per annotator. This led us to check the correlation between the performance of the user token and composite embedding methods with the number of annotations per annotator when examining the scaling experiments. We found a slightly stronger correlation for the user token $( R = 0 . 4 0 )$ than for the composite embedding $( R = 0 . 3 5 )$ suggesting that the low number of annotations per annotator is more detrimental to the user token method, but further work is needed to understand this relationship.

Interestingly, while the multi-task model performed the best in previous work on GoEmotions and GHC (Davani et al., 2022), we found that it performed worse in our experiments and was more expensive to train. This was the case even when using the full datasets, while our reimplementation of the approach outperforms results reported in their paper. It tends to perform worse than the baseline for the datasets that had higher agreement. Our hypothesis when we began our study was that multi-tasking would outperform other methods, as it allocates a larger number of dedicated parameters to learning representations of each annotator. We expected to see a trade-off in the model complexity and annotator modeling performance, with multi-tasking being the highest on both.

Lastly, it is important to note that our analysis is focused on the correlations between performance and surface-level features. This is to provide initial exploratory insight into an area that deserves much further analysis and experimentation. Corpora have idiosyncrasies and future work can explore how to measure such qualities of dataset construction to support a more human-centric and personalized approach to annotator modeling, rather than one that abstracts away from these qualities.

## 9 Conclusion

As research on subjective tasks in natural language processing has grown, the importance of modeling annotators and minority opinions has become more apparent. With the recent growth in work on annotator modeling and data perspectivism, it is becoming important to understand how the properties of our datasets and methods impact the effectiveness of annotator modeling methods. We examined seven corpora to better understand recent methods and introduced our method, the composite embeddings. We found that when annotator agreement on a dataset is low $( \mathrm { K } { - } \alpha < 0 . 4 )$ , the user token embedding was most effective. When the annotator agreement was higher $( \mathrm { K } { - } \alpha > 0 . 5 )$ , our new composite embedding gave the best performance. Surprisingly, the user token and composite embeddings, which are simple and efficient to implement, outperformed the multi-tasking model that was the highest performing model in prior work. Importantly, we also note that the number of annotations per annotator is correlated much more strongly with performance improvements than other corpus statistics, suggesting that this should be an area of focus for those constructing new datasets and collecting annotations. Our code and collection of over 3k trial experiments statistics are publicly available to support further work in this area.

## Limitations

We examined seven different corpora for our experiments and covered the binary classification tasks of emotion recognition, hate and offensive speech detection, and judgement of social norms and morality. However, there are many more types of subjective tasks we have not considered in our work, including those that were previously thought to be more objective in nature (Pavlick and Kwiatkowski, 2019). We do not know how our results will generalize to unseen tasks that are significantly different than we we have examined in this paper. It would be interesting to extend the analysis by the type of task and the degree on a prescriptive to descriptive continuum to measure the influence on annotator modeling performance. Furthermore, our experiments examine surface statistics of corpora, which may be impacted by underlying mechanisms of data collection, corpus construction, or other factors and more work is needed to understand the relationship between these mechanisms and downstream model performance.

The scaling experiments we performed were initially designed to target the number of annotators and number of annotations per annotator and for experiments with one, we held fixed the value of the other to serve as a control. While we ran a large number of trials, many of the annotation scaling experiments use only a handful of different quantities of annotators. Future work should diversify the sampling for such trials to confirm these results, but our work serves as a first exploratory study that exemplifies the relationship with regards to the correlation and thresholds for sufficient numbers of annotations per annotator.

Our datasets did not all include demographic information. Previous work by Deng et al. (2023) included a detailed analysis and breakdown of annotator identity groups and their relation to annotator modeling performance. Since we do not know which annotators are represented in four of our seven datasets, we cannot say that our results are robust across demographic groups. Future work in this area should include more corpora with annotator information, and future data collection should strive to contextualize collected annotations with such annotator meta-data.

## Acknowledgements

This work has been supported by the Federal Ministry of Education and Research of Germany (BMBF) as a part of the Junior AI Scientists program under the reference 01-S20060, the state of North Rhine-Westphalia as part of the Lamarr Institute for Machine Learning and Artificial Intelligence, and by Hessian.AI. Any opinions, findings, conclusions, or recommendations in this material are those of the authors and do not necessarily reflect the views of the BMBF, Lamarr Institute, or Hessian.AI. We appreciate the anonymous reviewers for their detailed and constructive feedback.

## References

Sohail Akhtar, Valerio Basile, and Viviana Patti. 2021. Whose opinions matter? perspective-aware models to identify opinions of hate speech victims in abusive language detection. ArXiv preprint, abs/2106.15896.

Dina Almanea and Massimo Poesio. 2022. ArMIS - the Arabic misogyny and sexism corpus with annotator subjective disagreements. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 2282–2291, Marseille, France.

Lora Aroyo and Chris Welty. 2015. Truth is a lie: Crowd truth and the seven myths of human annotation. AI Mag., 36:15–24.

Plaban Kumar Bhowmick, Anupam Basu, and Pabitra Mitra. 2008. An agreement measure for determining inter-annotator reliability of human judgements on affective text. In Coling 2008: Proceedings of the workshop on Human Judgements in Computational Linguistics, pages 58–65, Manchester, UK.

Federico Cabitza, Andrea Campagner, and Valerio Basile. 2023. Toward a perspectivist turn in ground truthing for predictive computing. Proceedings of the AAAI Conference on Artificial Intelligence, 37(6):6860–6868.

Silvia Casola, Soda Lo, Valerio Basile, Simona Frenda, Alessandra Cignarella, Viviana Patti, and Cristina Bosco. 2023. Confidence-based ensembling of perspective-aware models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3496–3507, Singapore.

Amanda Cercas Curry, Gavin Abercrombie, and Verena Rieser. 2021. ConvAbuse: Data, analysis, and benchmarks for nuanced abuse detection in conversational AI. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 7388–7403, Online and Punta Cana, Dominican Republic.

Aida Mostafazadeh Davani, Mark Díaz, and Vinodkumar Prabhakaran. 2022. Dealing with disagreements: Looking beyond the majority vote in subjective annotations. Transactions ofthe Associationfor Computational Linguistics, 10:92–110.

Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi. 2020. GoEmotions: A dataset of fine-grained emotions. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4040–4054, Online.

Naihao Deng, Siyang Liu, Xinliang Frederick Zhang, Winston Wu, Lu Wang, and Rada Mihalcea. 2023. You are what you annotate: Towards better models through annotator representations. ArXiv preprint, abs/2305.14663.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota.

Paul Ekman et al. 1999. Basic emotions. Handbook of cognition and emotion, 98(45-60):16.

Meng Fang, Yuan Li, and Trevor Cohn. 2017. Learning how to active learn: A deep reinforcement learning approach. In Proceedings ofthe 2017 Conference on Empirical Methods in Natural Language Processing, pages 595–605, Copenhagen, Denmark.

Eve Fleisig, Rediet Abebe, and Dan Klein. 2023. When the majority is wrong: Modeling annotator disagreement for subjective tasks. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 6715–6726, Singapore.

Maxwell Forbes, Jena D. Hwang, Vered Shwartz, Maarten Sap, and Yejin Choi. 2020. Social chemistry 101: Learning to reason about social and moral norms. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 653–670, Online.

Hassan Hayat, Carles Ventura, and Agata Lapedriza. 2022. Modeling subjective affect annotations with multi-task learning. Sensors, 22(14).

Pritam Kadasi and Mayank Singh. 2023. Unveiling the multi-annotation process: Examining the influence of annotation quantity and instance difficulty on model performance.

Brendan Kennedy, Mohammad Atari, Aida Mostafazadeh Davani, Leigh Yeh, Ali Omrani, Yehsong Kim, Kris Coombs, Shreya Havaldar, Gwenyth Portillo-Wightman, Elaine Gonzalez, Joe Hoover, Aida Azatian, Alyzeh Hussain, Austin Lara, Gabriel Cardenas, Adam Omary, Christina Park, Xin Wang, Clarisa Wijaya, Yong Zhang, Beth Meyerowitz, and Morteza Dehghani. 2022. Introducing the gab hate corpus: defining and applying hate-based rhetoric to social media posts at scale. Lang. Resour. Eval., 56(1):79–108.

Milton King and Paul Cook. 2020. Evaluating approaches to personalizing language models. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 2461–2469, Marseille, France.

Deepak Kumar, Patrick Gage Kelley, Sunny Consolvo, Joshua Mason, Elie Bursztein, Zakir Durumeric, Kurt Thomas, and Michael Bailey. 2021. Designing toxic content classification for a diversity of perspectives.

Elisa Leonardelli, Gavin Abercrombie, Dina Almanea, Valerio Basile, Tommaso Fornaciari, Barbara Plank, Verena Rieser, Alexandra Uma, and Massimo Poesio. 2023. SemEval-2023 task 11: Learning with disagreements (LeWiDi). In Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023), pages 2304–2318, Toronto, Canada.

Elisa Leonardelli, Stefano Menini, Alessio Palmero Aprosio, Marco Guerini, and Sara Tonelli. 2021. Agreeing to disagree: Annotating offensive language datasets with annotators’ disagreement. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 10528–10539, Online and Punta Cana, Dominican Republic.

Fatemehsadat Mireshghallah, Vaishnavi Shrivastava, Milad Shokouhi, Taylor Berg-Kirkpatrick, Robert Sim, and Dimitrios Dimitriadis. 2022. UserIdentifier: Implicit user representations for simple and effective personalized sentiment analysis. In Proceedings of the 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 3449–3456, Seattle, United States.

Swaroop Mishra and Bhavdeep Singh Sachdeva. 2020. Do we need to create big datasets to learn a task? In Proceedings ofSustaiNLP: Workshop on Simple and Efficient Natural Language Processing, pages 169–173, Online.

Stefanie Nowak and Stefan Rüger. 2010. How reliable are annotations via crowdsourcing? a study about inter-annotator agreement for multi-label image annotation. In Proceedings of the international conference on Multimedia information retrieval - MIR ’10, page 557. This is the author’s version of the work. It is posted here by permission of ACM for your personal use. Not for redistribution.

Ellie Pavlick and Tom Kwiatkowski. 2019. Inherent disagreements in human textual inferences. Transactions ofthe Associationfor Computational Linguistics, 7:677–694.

Joan Plepi, Béla Neuendorf, Lucie Flek, and Charles Welch. 2022. Unifying data perspectivism and personalization: An application to social norms. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 7391– 7402, Abu Dhabi, United Arab Emirates.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China.

Paul Rottger, Bertie Vidgen, Dirk Hovy, and Janet Pierrehumbert. 2022. Two contrasting data annotation paradigms for subjective NLP tasks. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 175–190, Seattle, United States.

Ali Safaya, Moutasem Abdullatif, and Deniz Yuret. 2020. KUISAIL at SemEval-2020 task 12: BERT-CNN for offensive speech identification in social media. In Proceedings ofthe Fourteenth Workshop on Semantic Evaluation, pages 2054–2059, Barcelona (online).

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. ArXiv preprint, abs/1910.01108.

Maarten Sap, Dallas Card, Saadia Gabriel, Yejin Choi, and Noah A. Smith. 2019. The risk of racial bias in hate speech detection. In Proceedings ofthe 57th Annual Meeting of the Association for Computational Linguistics, pages 1668–1678, Florence, Italy.

Maarten Sap, Swabha Swayamdipta, Laura Vianna, Xuhui Zhou, Yejin Choi, and Noah A. Smith. 2022. Annotators with attitudes: How annotator beliefs and identities bias toxic language detection. In Proceedings ofthe 2022 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 5884–5906, Seattle, United States.

Sadat Shahriar and Thamar Solorio. 2023. SafeWebUH at SemEval-2023 task 11: Learning annotator disagreement in derogatory text: Comparison of direct training vs aggregation. In Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023), pages 94–100, Toronto, Canada.

Victor S. Sheng, Foster J. Provost, and Panagiotis G. Ipeirotis. 2008. Get another label? improving data quality and data mining using multiple, noisy labelers. Organizations & Markets eJournal.

Michael Sullivan, Mohammed Yasin, and Cassandra L. Jacobs. 2023. University at buffalo at SemEval-2023 task 11: MASDA–modelling annotator sensibilities through DisAggregation. In Proceedings ofthe 17th International Workshop on Semantic Evaluation (SemEval-2023), pages 978–985, Toronto, Canada.

Nikolas Vitsakis, Amit Parekh, Tanvi Dinkar, Gavin Abercrombie, Ioannis Konstas, and Verena Rieser. 2023. iLab at SemEval-2023 task 11 le-wi-di: Modelling disagreement or modelling perspectives? In

Proceedings ofthe 17th International Workshop on Semantic Evaluation (SemEval-2023), pages 1660– 1669, Toronto, Canada.

Xinpeng Wang and Barbara Plank. 2023. Actor: Active learning with annotator-specific classification heads to embrace human label variation. ArXiv preprint, abs/2310.14979.

Charles Welch, Chenxi Gu, Jonathan K. Kummerfeld, Veronica Perez-Rosas, and Rada Mihalcea. 2022a. Leveraging similar users for personalized language modeling with limited data. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1742–1752, Dublin, Ireland.

Charles Welch, Jonathan K. Kummerfeld, Verónica Pérez-Rosas, and Rada Mihalcea. 2020a. Compositional demographic word embeddings. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4076–4089, Online.

Charles Welch, Jonathan K. Kummerfeld, Verónica Pérez-Rosas, and Rada Mihalcea. 2020b. Exploring the value of personalized word embeddings. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6856–6862, Barcelona, Spain (Online).

Charles Welch, Joan Plepi, Béla Neuendorf, and Lucie Flek. 2022b. Understanding interpersonal conflict types and their impact on perception classification. In Proceedings ofthe Fifth Workshop on Natural Language Processing and Computational Social Science (NLP+CSS), pages 79–88, Abu Dhabi, UAE.

Ellery Wulczyn, Nithum Thain, and Lucas Dixon. 2017. Ex machina: Personal attacks seen at scale. In Proceedings of the 26th International Conference on World Wide Web, WWW 2017, Perth, Australia, April 3-7, 2017, pages 1391–1399.

Shujian Zhang, Chengyue Gong, and Eunsol Choi. 2021. Learning with different amounts of annotation: From zero to many labels. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7620–7632, Online and Punta Cana, Dominican Republic.