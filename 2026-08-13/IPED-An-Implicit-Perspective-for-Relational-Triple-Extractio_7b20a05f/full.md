# IPED: An Implicit Perspective for Relational Triple Extraction based on Diffusion Model

Jianli Zhao Changhao Xu Bin Jiang School of Mechanical, Electrical & Information Engineering Shandong University {jianliz, xch}@mail.sdu.edu.cn jiangbin@sdu.edu.cn

## Abstract

Relational triple extraction is a fundamental task in the field of information extraction, and a promising framework based on table filling has recently gained attention as a potential baseline for entity relation extraction. However, inherent shortcomings such as redundant information and incomplete triple recognition remain problematic. To address these challenges, we propose an Implicit Perspective for relational triple Extraction based on Diffusion model (IPED), an innovative approach for extracting relational triples. Our classifier-free solution adopts an implicit strategy using block coverage to complete the tables, avoiding the limitations of explicit tagging methods. Additionally, we introduce a generative model structure, the block-denoising diffusion model, to collaborate with our implicit perspective and effectively circumvent redundant information disruptions. Experimental results on two popular datasets demonstrate that IPED achieves state-of-the-art performance while gaining superior inference speed and low computational complexity. To support future research, we have made our source code publicly available online.

## 1 Introduction

The extraction of relational triples has been an important and fundamental task in knowledge graph construction (Zamini et al., 2022; Wei et al., 2020a), aiming to recognize triples in the form of (head entity, relation, tail entity) from unstructured text. Current research in information extraction can be categorized into two main approaches: the joint extraction models, which utilize a simultaneous style, and the pipeline models, which utilize a twoencoder methodology to extract entities and relations. While the pipeline framework is criticized for serious error propagation and lack of interaction between its two subtasks (Shen et al., 2021), leading to performance decline, many recent joint extraction models have begun to thrive due to their enhanced capability to deal with complex scenarios such as single entity overlap (SEO), entity pair overlap (EPO), and subject object overlap (SOO).

Among these popular joint extraction methods, one baseline, known as the table-filling method, has gained favor in recent research. Compared to a multi-task joint structure, this method features a table of token pair units that are to be filled and decoded in a single step. In this way, it avoids exposure bias and error propagation, challenges that most methods cannot fully overcome. Particularly for recently proposed models (Shang et al., 2022; Ren et al., 2021; Wang et al., 2021), these can employ a novel table-filling strategy to simplify the decoding process and enhance information interaction.

Despite many unique advantages over tablefilling methods, some flaws still remain to be addressed. (1) The abundance of negative tagging in a table, which is significantly denser than positive tagging, leads to imbalanced labeling and redundant information (Wang et al., 2021; Ning et al., 2023). To the best of our knowledge, this is a universal issue across all table-filling models. This imbalance results in a bias towards negative tagging and heightened computational complexity. (2) Many table-filling strategies fail to extract all scenarios of triples, leading to decreased recall (Ning et al., 2023). Even in the recent significant work (Shang et al., 2022), entities consisting of a single token in a triple cannot be properly extracted due to conflicts arising from multiple labels in one element. (3) Once a sentence contains multiple triples, the separate labels of different triples may intersect in a single element, causing confusion in decoding all ground-truth triples. Many models (Ren et al., 2021; Ning et al., 2023) employ decoding algorithms that match labels based on the nearest-neighbor principle, which can lead to error associations within a triple. (4) A line of models, not limited to table-filling ones, exhibit poor learning performance on the WebNLG dataset in contrast to the NYT dataset and they attribute it to the vast number of predefined relations in the former dataset (Gao et al., 2023).

After conducting a detailed observation and analysis of their models, it is observed that all existing table-filling-based methods are consistently constrained by the approach of utilizing a classifier to tag each table element explicitly. Mainly because of this, most of them can hardly escape the challenges mentioned above, despite attempts to introduce innovative labeling strategies and creative decoding algorithms. This constraint necessitates traversing each element of the table, consequently leading to a substantial number of negative samplings. This explicit way of assigning a fixed label to each element can not cope with scenarios when one element requires multiple labels, leading to the inability to recognize all triples and confusion in the regions where triple labels intersect. Additionally, certain decoding strategies, designed in response to this approach, often result in incorrectly matched labels for a triple.

To address the aforementioned issues at a fundamental level, instead of explicitly labeling all the elements, we formulate a fresh perspective to implicitly fill the tables using a block-covered approach. In this method, blocks defined by four edges (up, down, right, left) and one level are refined within a three-dimensional table (multiple two-dimensional tables stacked together). In alignment with this implicit approach, we introduce a generative model designed to recover all blocks within the tables. Specifically, our proposed blockdenoising diffusion model (Blk-DDM) can progressively refine the edges and levels of the initialized blocks step by step through a reverse process, ensuring the blocks precisely cover the ground truth triples horizontally, vertically, and deeply. As a result, our model naturally disregards redundant information by leaving the negative spaces alone rather than classifying them. Furthermore, our approach allows for the adequate recognition of all potential triples, as the proposed blocks can overlap implicitly. In contrast to previous decoding algorithms that match explicit labels, our proposed simple but effective Parallel Boundary Emitting Strategy (PBES) for decoding has the capability of extracting all triples accurately, circumventing error association challenges and significantly accelerating inference. Additionally, our denoising diffusion process enables the gradual refinement of specific fine-grained relation types within triples, enhancing performance in large-relation datasets such as WebNLG (demonstrated in Section 4.8). Experimental results on two datasets, NYT and WebNLG, demonstrate that our model achieves state-of-the-art performance and exhibits superior efficiency in inference.

## 2 Related Works

## 2.1 Joint Extraction Models

Existing joint extraction models can be roughly sorted into two frameworks. The first framework, based on multi-task learning, utilizes a shared encoder but employs distinct decoders to sequentially predict entities and relations. (Miwa and Bansal, 2016) proposes an integrated model that extracts entities and relations separately, leveraging shared parameters and mutual interaction. (Luan et al., 2018) adopts a model employing shared data representations to mitigate error propagation between tasks. CasRel (Wei et al., 2020b) treats relations as functions mapping subjects to objects to make extraction. The other framework is structured prediction which integrates the two subtasks into a unified structure and performs decoding in one step. (Katiyar and Cardie, 2017) proposes a model using sequence tagging-based approaches and forbidding dependency trees. (Sun et al., 2019) employs graph convolutional networks for joint inference. (Wang and Lu, 2020) implements a table-filling strategy using a table encoder and a sequence encoder.

## 2.2 Diffusion Model

Diffusion model is a type of deep latent generative model primarily utilized for generating continuous data structures, such as images and audio. DDPM (Ho et al., 2020) is a pioneer work that makes diffusion model practical to applications, thus inviting excellent works (Kong et al., 2021; Zhao et al., 2023) in various fields. Recently, there has been an emergence of works in NLP utilizing diffusion models, such as (Li et al., 2022a; He et al., 2023) in language model and (Bi et al., 2023; Gong et al., 2023) in sequence-to-sequence tasks, despite the perceived challenges in applying diffusion models to discrete text sequences. Notably, DiffusionNER (Shen et al., 2023) also applies the diffusion model to named entity recognition. However, there are significant differences with our IPED, particularly in (1) task definition: IPED concentrates on extracting relational triples rather than mere entities. (2) core design: our model operates by diffusing in a three-dimensional space for each triple, in contrast to DiffusionNER, which diffuses within a onedimensional matrix for each entity and incorporates an additional classifier.

## 3 Methodology

This section firstly introduces our implicit tablefilling strategy and its corresponding decoding algorithm. Secondly, the formulation of the Block-Denoising Diffusion Model is presented. Finally, the network architecture of our model is detailed.

## 3.1 Implicit Block-Covered Table Filling

For a sentence $\boldsymbol { S } = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { L } \}$ composed of L words, K relations $\mathcal { R } = \{ r _ { 1 } , r _ { 2 } , . . . , r _ { K } \}$ are predefined in a dataset. The objective of relational triple extraction is to identify all triples (head, relation, tail) in each sentence, where the head and tail represent the subject and object entities, respectively, along with their connected relation. Within a sentence, for all triples $\tau = \{ ( h _ { i } , r _ { i } , t _ { i } ) \} _ { i = 1 } ^ { M }$ , M denotes the total number of triples, and $h _ { i } , t _ { i }$ represent the entity spans, each composed of one or more consecutive tokens.

Unlike previous classifier-based tagging methods, our model does not allocate a label to each unit of the L\*L\*K three-dimensional matrix. Instead, it refines M blocks $( \mathbf { B } \in \mathbb { R } ^ { M \times 5 } )$ to cover the K tables horizontally, vertically, and deeply, which is, our implicit way to fill the tables. As illustrated in Figure 1, each block consists of five elements: the up and down edges indicate vertical positioning, the left and right edges denote horizontal positioning, and the level represents depth positioning within the K stacked tables, with each table corresponding to a specific relation. Via our proposed Blk-DDM (described in Section 3.2), these M blocks are progressively refined to reveal the recognized triples.

The proposed decoding scheme, named Parallel Boundary Emitting Strategy (PBES), is introduced to extract triples from the blocks. PBES follows the four edges and one level of each block, emitting them in parallel to the corresponding entities and relation. Specifically, for each block, the up and down edges are extended to the left side of the table, indicating the boundaries of the head entity.

Similarly, the left and right edges are extended correspondingly to identify the boundaries of the tail entity. Meanwhile, the depth level where the block is located signifies a specific table, thereby indicating a particular relation. By repeating this process M times as described, all blocks are converted into relational triples.

Our table-filling method enables the precise extraction of all existing triples by circumventing the conflicts typically associated with explicit tagging. Thanks to the lack of inner constraints between the M blocks, this approach not only naturally tackles complex scenarios such as SEO, EPO, and SOO, but also overcomes issues like the failure of singletoken entity extraction in (Shang et al., 2022) and error association in (Ren et al., 2021; Ning et al., 2023).

## 3.2 Block-Denoising Diffusion Model

In this section, we present the formulation of block generation as a denoising diffusion process and introduce our block-denoising diffusion model (Blk-DDM).<sup>2</sup> As depicted in Figure 1, the diffusion model comprises a forward process that incrementally introduces noise to data samples and a reverse process that recovers the ground truth through stepby-step denoising. These two processes are synchronized to facilitate the learning of a network endowed with the denoising capability. During the inference phase, the diffusion model incrementally refines data samples through a multistep denoising process from a standard Gaussian distribution. Consequently, we convert our M blocks, composed of five elements (up, down, left, right, level), into index format $\mathbf { B } = \{ ( u _ { i } , d _ { i } , l _ { i } , r _ { i } , v _ { i } ) \} _ { i = 0 } ^ { M }$ to support the denoising operations. Following (Ho et al., 2020), the forward denoising process is simplified by computing $\left\{ \bar { \alpha } _ { 1 } , . . . , \bar { \alpha } _ { T } \right\}$ from a predefined variance schedule $\{ \beta _ { t } \} _ { t = 0 } ^ { T } \in ( 0 , 1 )$ , and thus noise injection in multiple steps can be integrated into one step as follows:

$$
\begin{array} { r } { q \left( { { z } _ { t } } \mid { { z } _ { 0 } } \right) = \mathcal { N } \left( { { z } _ { t } } ; \sqrt { { { \bar { \alpha } } _ { t } } } { { z } _ { 0 } } , \left( 1 - { { \bar { \alpha } } _ { t } } \right) \mathbf { I } \right) } \end{array}\tag{1}
$$

where $q$ represents the forward process from z<sub>0</sub> to $z _ { t } . \mathrm { ~ \ } z _ { 0 }$ and $z _ { t }$ denote the original data and the noised data at timestep t, respectively. I is the standard Gaussian distribution. Note that the fixed forward process depicted in Figure 1 can be considered as a Markov chain.

![](images/d459263d6d022556316cd049f9777a76ae2de13b2ddbedd26dfd78eadb1c0ce3.jpg)  
Figure 1: Figure (a) depicts our table-filling strategy along with triple demonstration. For the convenience of illustration, we simplify our three-dimensional tables (as in Figure (b)) into the form of a two-dimensional table in Figure (a), containing nine blocks in total that represent nine triples. Here, dashed rectangles denote the four edges of the blocks, and different colors indicate the levels of the blocks. Figure (b) illustrates the overall diffusion process.

denoising as follows:

Training Process The training process of the diffusion model involves a one-step noise addition and a one-step prediction towards the ground truth, aimed at training a network for inference purposes. As for a sentence, blocks $\mathbf { B } \in \mathbb { R } ^ { M \times 5 }$ are initially derived from M ground truth triples. Subsequently, B is expanded by some blocks randomly sampled from a Gaussian distribution, resulting in $z _ { 0 } = \mathbf { B } \in \mathbb { R } ^ { N \times 5 } \left( \mathbf { N } > \mathbf { M } \right)$ . Following Equation (1), we then have

$$
z _ { t } = \sqrt { \bar { \alpha } _ { t } } z _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon\tag{2}
$$

where t $( \leq$ predefined total timestep T) is a randomly chosen timestep and $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ donates the pure noise from the Gaussian distribution, thus getting noised blocks B. Feeding $z _ { t }$ into our network $f _ { \theta } ,$ , one can get the predicted $z _ { \mathrm { 0 } }$ (Section 3.3) and compute the objective function (Section 3.3.3). By optimizing the loss function, the weights of our network $f _ { \theta }$ will be updated accordingly.

Inference Process Following DDIM (Song et al., 2021), the reverse diffusion process is defined as a non-Markovian chain to achieve inference acceleration. An arithmetic sequence $\tau$ of length $\sigma$ is predefined as $[ 1 , . . . , T ]$ and D purely noised blocks $\boldsymbol { x } _ { T } \in \mathbb { R } ^ { D \times 5 }$ are sampled from the Gaussian distribution. Modified from DDIM, we have progressive

$$
z _ { \tau _ { i - 1 } } = \sqrt { \bar { \alpha } _ { \tau _ { i - 1 } } } \hat { z } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { \tau _ { i - 1 } } } \frac { z _ { \tau _ { i } } - \sqrt { \bar { \alpha } _ { \tau _ { i } } } \hat { z } _ { 0 } } { \sqrt { 1 - \bar { \alpha } _ { \tau _ { i } } } }\tag{3}
$$

where $\hat { z } _ { 0 }$ is predicted by $f _ { \theta } ,$ with the index i traversing from $\sigma$ to 1. After $\sigma$ iterations, $z _ { 0 } \in$ $\mathbb { R } ^ { D \times 5 }$ is recovered from the noise distribution. Note that D is a hyperparameter supposedly larger than the ground truth block number, and thus the filtration of predicted D blocks aims to minimize their divergence from the ground truth. Hence, blocks with the sum predicted probability below the threshold $\varphi$ are discarded. <sup>3</sup>

## 3.3 Model Structure

As shown in Figure 2, our model architecture consists of three parts: Representation Encoder, Edge Predictor, and Level Predictor. Accepting one sentence, noised blocks (with timestep t) as inputs, the model network $f _ { \theta }$ generates the predicted blocks $\hat { z } _ { 0 }$ appropriately.

## 3.3.1 Representation Encoder

Given an input sentence $\boldsymbol { S } = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { L } \}$ composed of L words or indexes, here our sentence encoder consists of a pre-trained BERT (Devlin et al., 2019) and a bi-directional LSTM (Lample et al., 2016). Utilizing our encoder, token embeddings along with positional embeddings as the input are transformed into contextualized sentence representation ${ \bf R } _ { H } \in \mathbb { R } ^ { L \times d }$ . Then the inner span tokens are extracted from the word indexes indicated by the edges of our blocks, yielding the edge representation ${ \bf R } _ { E } \in \mathbb { R } ^ { N \times d }$ following mean-pooling. Differently, the level representation ${ \bf R } _ { V } \in \mathbb { R } ^ { N \times d }$ is derived directly from an embedding relation matrix $\mathrm { R } \in \mathbb { R } ^ { K \times d }$ , where each row represents a distinct relation type and K denotes the total number of predefined relation types. This matrix is regarded as a trainable parameter set in our model.

![](images/c814f57904f8552a8504d71c9c7c23750abd4d9ffd6c923b61be5bc821146d55.jpg)  
Figure 2: The overview model structure of IPED. To enhance the illustration of the workflow, we utilize three different colors to denote various feature representations: Pink for level information, Yellow for sentence information, and Red for edge information. E represents the encoding of $\mathbf { R } _ { E }$ .  denotes the maxpooling operation. To simplify the illustration, the four Biaffine modules are integrated into one in this overview. To better display the reverse process as in Figure 1, a reverse-flow arrow is used to symbolize progressive denoising.

To better fuse both edge representation and level representation with contextualized information, we utilize the hierarchical Co-Attention mechanism in our model, which is proven to be effective with multimodal data (Chen et al., 2021). Among the two Parallel Co-Attention modules in our model, we illustrate one of them as an example, which attends to the sentence representation $\mathbf { R } _ { H }$ and the edge representation $\mathbf { R } _ { E }$ simultaneously. An affinity matrix $\mathbf { C } \in \mathbb { R } ^ { L \times N }$ that transforms sentence attention space into edge attention space, and the attention score vector $\mathbf { a } ^ { e } \in \mathbb { R } ^ { N }$ that optimizes the affinity, are calculated as follows:

$$
\mathbf { C } = \operatorname { t a n h } \left( \mathbf { R } _ { H } ^ { T } \mathbf { W } _ { b } \mathbf { R } _ { E } \right)\tag{4}
$$

$$
\mathbf { H } ^ { e } = \operatorname { t a n h } \left( \mathbf { W } _ { e } \mathbf { R } _ { E } + \left( \mathbf { W } _ { h } \mathbf { R } _ { E } \right) \mathbf { C } \right)\tag{5}
$$

$$
\mathbf { a } ^ { e } = \mathrm { s o f t m a x } \left( \mathbf { w } _ { h e } ^ { T } \mathbf { H } ^ { e } \right)\tag{6}
$$

where $\mathbf { W } _ { b } \in \mathbb { R } ^ { d \times d } , \mathbf { W } _ { e } \in \mathbb { R } ^ { k \times d } , \mathbf { W } _ { h } \in \mathbb { R } ^ { k \times d } ,$ ${ \mathbf w } _ { h e } \in \mathbb { R } ^ { k }$ are learnable parameters, H<sup>e</sup> is the middle state. Finally, the edge attention vector $\hat { \mathbf { R } } _ { E } \in \mathbb { R } ^ { N \times d }$ is calculated as the weighted sum of the edge features plus an additional sinusoidal embedding (Vaswani et al., 2017):

$$
\hat { \mathbf { R } } _ { E } = \mathbf { a } ^ { e } \mathbf { R } _ { E } + \mathbf { E } _ { t }\tag{7}
$$

where $\mathbf { E } _ { t }$ is the embedding of timestep t. Equally, the same operation is implemented to obtain the fused level representation $\hat { \mathbf { R } } _ { V } \in \mathbb { R } ^ { N \times d }$

## 3.3.2 Edge Predictor and Level Predictor

For the Edge Predictor, we employ Biaffine to acquire fine-grained fused representations, which is proposed for dependency parsing (Dozat and Manning, 2016) at the outset. Here we have four Biaffine for $\mathbf { R } _ { E H } ^ { \eta }$ representations where $\eta \in$ $\{ u , d , l , r \}$ symbolizes four edges, respectively. $\mathbf { R } _ { E H } ^ { \eta }$ is obtained as follows:

$$
\begin{array} { r l } & { \mathbf { R } _ { E H } ^ { \eta } = \mathrm { B i a f f } ^ { \eta } \left( \mathbf { R } _ { H } , \hat { \mathbf { R } } _ { E } \right) } \\ & { \qquad = \mathbf { R } _ { H } ^ { T } \mathbf { U } _ { 1 } ^ { \eta } \hat { \mathbf { R } } _ { E } + \mathbf { U } _ { 2 } ^ { \eta } \left( \mathbf { R } _ { H } \oplus \hat { \mathbf { R } } _ { E } \right) + \mathbf { b } ^ { \eta } } \end{array}\tag{8}
$$

where $\mathbf { U } _ { 1 } ^ { \eta }$ and $\mathbf { U } _ { 2 } ^ { \eta }$ donate two parameter matrices, $\mathbf { b } ^ { \eta }$ is the bias vector, means concatenation.

<table><tr><td rowspan="2">Method</td><td colspan="3">NYT*</td><td colspan="3">WebNLG*</td><td colspan="3">NYT</td><td colspan="3">WebNLG</td></tr><tr><td>Prec.</td><td>Rec.</td><td>F1</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td>GraphRel (Fu et al., 2019)</td><td>63.9</td><td>60.0</td><td>61.9</td><td>44.7</td><td>44.1</td><td>42.9</td><td></td><td>-</td><td>-</td><td></td><td>-</td><td>1</td></tr><tr><td>RSAN (Yuan et al., 2020)</td><td>-</td><td>-</td><td>-</td><td></td><td></td><td></td><td>85.7</td><td>83.6</td><td>84.6</td><td>80.5</td><td>83.8</td><td>82.1</td></tr><tr><td>TPLinker (Wang et al., 2020)</td><td>91.3</td><td>92.5</td><td>91.9</td><td>91.8</td><td>92.0</td><td>91.9</td><td>91.4</td><td>92.6</td><td>92.0</td><td>88.9</td><td>84.5</td><td>86.7</td></tr><tr><td>GRTE (Ren et al., 2021)</td><td>92.9</td><td>93.1</td><td>93.0</td><td>93.7</td><td>94.2</td><td>93.9</td><td>93.4</td><td>93.5</td><td>93.4</td><td>92.3</td><td>87.9</td><td>90.0</td></tr><tr><td>PRGC (Zheng et al., 2021)</td><td>93.3</td><td>91.9</td><td>92.6</td><td>94.0</td><td>92.1</td><td>93.0</td><td>93.5</td><td>91.9</td><td>92.7</td><td>89.9</td><td>87.2</td><td>88.5</td></tr><tr><td>EmRel (Xu et al., 2022)</td><td>91.7</td><td>92.5</td><td>92.1</td><td>92.7</td><td>93.0</td><td>92.9</td><td>92.6</td><td>92.7</td><td>92.6</td><td>90.2</td><td>87.4</td><td>88.7</td></tr><tr><td>RelU-Net (Zhang et al., 2022)</td><td>93.3</td><td>92.9</td><td>93.1</td><td>94.9</td><td>93.7</td><td>94.3</td><td></td><td></td><td>-</td><td></td><td>-</td><td></td></tr><tr><td>BiRTE (Ren et al., 2022)</td><td>92.2</td><td>93.8</td><td>93.0</td><td>93.2</td><td>94.0</td><td>93.6</td><td>91.9</td><td>93.7</td><td>92.8</td><td>89.0</td><td>89.5</td><td>89.3</td></tr><tr><td>OneRel (Shang et al., 2022)</td><td>92.8</td><td>92.9</td><td>92.8</td><td>94.1</td><td>94.4</td><td>94.3</td><td>93.2</td><td>92.6</td><td>92.9</td><td>91.8</td><td>90.3</td><td>91.0</td></tr><tr><td>RFBFN (Li et al., 2022b)</td><td>93.4</td><td>93.2</td><td>93.3</td><td>93.9</td><td>94.1</td><td>94.0</td><td>93.7</td><td>93.6</td><td>93.6</td><td>91.5</td><td>89.4</td><td>90.4</td></tr><tr><td>ODRTE (Ning et al., 2023)</td><td>93.5</td><td>93.9</td><td>93.7</td><td>94.6</td><td>95.1</td><td>94.9</td><td>94.2</td><td>93.6</td><td>93.9</td><td>92.8</td><td>92.1</td><td>92.5</td></tr><tr><td>IPED</td><td>94.2</td><td>93.5</td><td>93.9</td><td>95.3</td><td>95.7</td><td>95.5</td><td>94.7</td><td>93.4</td><td>94.1</td><td>93.0</td><td>93.6</td><td>93.3</td></tr></table>

Table 1: Main results of IPED and other baselines.

Then $\mathbf { R } _ { E H } ^ { \eta }$ are put through four simple multiplelayer perceptrons with softmax layers to get the probabilities $\mathbf { P } ^ { \eta } \in \mathbb { R } ^ { N \times L }$ for four edges in blocks.

For the Level Predictor, a cross-attention layer is utilized to obtain the deep latent representation $\mathbf { R } _ { E V H } ,$ , incorporating edge-sentence embedding $\mathbf { R } _ { E H } ^ { \eta }$ to level representation $\hat { \mathbf { R } } _ { V }$ . Specifically, $\mathbf { R } _ { E H } ^ { \eta }$ undergoes a max-pooling operation to serve as the key and value tensors, while $\hat { \mathbf { R } } _ { V }$ acts as the query tensor. Then the level probability $\mathbf { P } ^ { \nu } \in \mathbb { R } ^ { \tilde { N } \times \tilde { K } }$ is determined using a multilayer perceptron, followed by a softmax layer.

## 3.3.3 Loss Function

In conjunction with the predicted probabilities above, the Log-Likelihood Function is maximized to train our model parameters. As N blocks are generated during training, yet only M ground truth blocks exist, we solve the optimal match via the Hopcroft-Krap algorithm (Carraresi and Sodini, 1986). Our objective function is defined as follows:

$$
\begin{array} { r l } & { \mathcal { L } = - \displaystyle \sum _ { i = 1 } ^ { N } \biggl [ \beta _ { 1 } \sum _ { \eta \in \{ u , d \} } \log \mathbf { P } _ { i } ^ { \eta } \bigl ( \xi ^ { \eta } ( i ) \bigr ) } \\ & { \quad \quad \quad + \beta _ { 2 } \displaystyle \sum _ { \eta \in \{ l , r \} } \log \mathbf { P } _ { i } ^ { \eta } \bigl ( \xi ^ { \eta } ( i ) \bigr ) } \\ & { \quad \quad \quad + \beta _ { 3 } \log \mathbf { P } _ { i } ^ { \nu } \bigl ( \xi ^ { \nu } ( i ) \bigr ) \biggr ] } \end{array}\tag{9}
$$

where $\xi \left( i \right)$ represents the ground truth edges and level of the i-th block, $\beta _ { 1 } , \beta _ { 2 } , \beta _ { 3 }$ are the hyperparameters for the weights of each prediction part.

## 4 Experiments

## 4.1 Datasets

Following previous works (Shang et al., 2022; Ning et al., 2023), we evaluate our model on two wellknown datasets NYT (Riedel et al., 2010) and WebNLG (Gardent et al., 2017). The NYT dataset is extracted using the distantly supervised method from New York Times news articles, while the WebNLG dataset was originally designed for Natural Language Generation. Each dataset exists in two versions: one is annotated with the whole entity span, and the other is annotated with the last word of entities. For clarity, we mark the fully annotated version as NYT and WebNLG, and the simpler annotated version as NYT\* and WebNL $G ^ { * }$ , respectively. Following prior works, we split the test set of each dataset based on the number of triples and the overlapping pattern in each sentence.

## 4.2 Evaluation Metrics

For a fair comparison with prior works mentioned above, we report standard micro Precision (Prec.), Recall (Rec.), and F1-score (F1.) as our three evaluation metrics. Meanwhile, we implement distinct matching rules for each version of the datasets. In the case of NYT and WebNLG datasets, an extracted relational triple is regarded correct only if all words of both entities and the relation type precisely align with the ground truth. For NYT\* and WebNL $G ^ { * }$ datasets, only the last words of two entities and the relation are required to be correct.

<table><tr><td rowspan="2">Model</td><td colspan="8">NYT*</td><td colspan="8">WebNLG*</td></tr><tr><td>Normal</td><td>SEO</td><td>EPO</td><td>Q=1</td><td>Q=2</td><td>Q=3</td><td>Q=4</td><td>Q≥5</td><td>Normal</td><td>SEO</td><td>EPO</td><td>Q=1</td><td>Q=2</td><td>Q=3</td><td>Q=4</td><td>Q≥5</td></tr><tr><td>GRTE</td><td>91.1</td><td>94.4</td><td>95.0</td><td>90.8</td><td>93.7</td><td>94.4</td><td>96.2</td><td>93.4</td><td>90.6</td><td>94.5</td><td>96.0</td><td>90.6</td><td>92.5</td><td>96.5</td><td>95.5</td><td>94.4</td></tr><tr><td>PRGC</td><td>91.0</td><td>94.0</td><td>94.5</td><td>91.1</td><td>93.0</td><td>93.5</td><td>95.5</td><td>93.0</td><td>90.4</td><td>93.6</td><td>95.9</td><td>89.9</td><td>91.6</td><td>95.0</td><td>94.8</td><td>92.8</td></tr><tr><td>RFBFN</td><td>91.2</td><td>95.2</td><td>95.6</td><td>91.4</td><td>93.8</td><td>94.8</td><td>96.4</td><td>93.9</td><td>91.0</td><td>94.6</td><td>96.5</td><td>90.8</td><td>92.6</td><td>96.6</td><td>94.7</td><td>94.5</td></tr><tr><td>ODRTE</td><td>91.3</td><td>95.7</td><td>95.9</td><td>91.3</td><td>93.4</td><td>94.6</td><td>96.9</td><td>95.3</td><td>92.1</td><td>95.4</td><td>95.9</td><td>91.1</td><td>93.5</td><td>95.9</td><td>96.1</td><td>95.1</td></tr><tr><td>IPED</td><td>91.0</td><td>95.7</td><td>96.0</td><td>91.5</td><td>93.2</td><td>94.9</td><td>97.3</td><td>95.4</td><td>92.1</td><td>95.6</td><td>96.9</td><td>91.8</td><td>94.2</td><td>96.8</td><td>96.7</td><td>96.0</td></tr></table>

Table 2: F1 score on sentences with different overlapping patterns and different triple numbers. Q stands for the number of triples in a sentence.

## 4.3 Implementation Details

To make a fair comparison, we utilize the cased base version of BERT (Devlin et al., 2019) as our pretrained model. The AdamW optimizer (Loshchilov and Hutter, 2019) is employed with a learning rate of 3e-5. The hidden size of our cross-attention and biaffine modules is configured to 1024. A warm-up learning rate scheduler, with a 0.1 ratio and a maximum gradient normalization of 1.5, is configured for the training process. Regarding the diffusion setting, the total timestep T is set to 1000, the sampling timestep σ to 10, and the number of denoising blocks D to 30. The sum threshold φ for the edges and level probabilities is established at 4.

## 4.4 Main Results

Table 1 presents the performance comparison between our IPED and various baselines across four benchmarks. It can be seen that our model, IPED, outperforms all the baselines and achieves stateof-the-art performance, even when compared to the strongest explicit table-filling baseline ODRTE (Ning et al., 2023) and the leading multi-task joint framework RFBFN (Li et al., 2022b). This proves the dramatic efficacy of our implicit perspective and denoising diffusion strategy.

Compared with the best baseline ODRTE, our IPED achieves a 0.2 absolute improvement in F1- score on both NYT and NYT\*. It is worth noticing that, a significant improvement, 0.8 and 0.6 gains in F1-score, is achieved on WebNLG and WebNLG\* respectively, whereas many models (Wang et al., 2020; Gao et al., 2023) blame their poor performance on the complexity arising from hundreds of predefined relation types. We attribute our advancement on large-relation datasets to block-level progressive refinement; specifically, our blockdenoising diffusion model allows fine-tuned block denoising across various levels of the tables.

The results on NYT and WebNLG reveal that our IPED outperforms OneRel (Shang et al., 2022) by 1.2% and 2.3%, and GRTE (Ren et al., 2021) by 0.7% and 3.3% in terms of F1-score, respectively. This demonstrates that the implicit tablefilling scheme can immensely avoid interruptions caused by redundant negative tagging, which otherwise leads to negative bias. This improvement highlights two key advantages of our approach: the capability to recognize all potential triples and the proficiency in avoiding error association during decoding.

## 4.5 Performance on Complex Scenarios

To validate the ability of our model to handle diverse overlapping patterns and multiple triples, we conduct further experiments on NYT\* and WebNLG\*. As indicated in Table 2, our proposed IPED model surpasses nearly all baselines on both datasets, with the exception of two scenarios on NYT\* when Q equals 2 and when there is no overlap. In complex scenarios, such as multiple triples within a single sentence, the performance of IPED turns out to be exceptional, surpassing four stateof-the-art models. The reason behind this is that our decoding scheme, the Parallel Boundary Emitting Strategy (PBES), has the capacity to accurately map our blocks into ground truth triples. This contrasts with previous decoding algorithms in explicit table-filling methods (Ren et al., 2021), which often incorrectly decode triples due to error association.

## 4.6 Computational Efficiency

To evaluate the computational efficiency of our IPED, we conduct further experiments with respect to Training Time, GPU Memory, Inference Time, and F1-score on NYT and WebNLG. As demonstrated in Table 3, we selected two robust baselines, GRTE and OD-RTE, for comparison. To verify the impact of the sampling timestep, we execute IPED with varying τ values. It can be seen that when $\sigma = 5 ,$ the inference speed of IPED is more than double that of GRTE, and it requires the least GPU memory compared to both baselines. Due to the inherent nature of diffusion training, the training time of our model is not the shortest, falling between OD-RTE and GRTE. Nevertheless, our IPED achieves a superior F1-score and greater inference efficiency. We conjecture the reasons might be our implicit table-filling strategy, which is exempt from redundant tagging, and the non-Markovian process employed during sampling.

<table><tr><td rowspan="2">Model</td><td colspan="4">NYT</td><td colspan="4">WebNLG</td></tr><tr><td>Training Time</td><td>GPU Mem</td><td>Infer. Time (1/8)</td><td>F1</td><td>Training Time</td><td></td><td>GPU Mem Infer. Time (1/8)</td><td>F1</td></tr><tr><td>GRTE</td><td> $9 3 1 ^ { \dag }$ </td><td> $1 8 7 7 1 ^ { \dag }$ </td><td>44.1 / 9.6</td><td>93.4</td><td> $1 1 8 ^ { \dagger }$ </td><td> $1 5 3 4 5 ^ { \dagger }$ </td><td>62.4 / 15.6</td><td>90.0</td></tr><tr><td>OD-RTE</td><td>798†</td><td>8372†</td><td>38.3 / 8.4</td><td>93.9</td><td>70†</td><td>7515†</td><td> $5 1 . 0 / 1 2 . 8 $ </td><td>92.5</td></tr><tr><td> $\mathrm { I P E D } _ { [ \sigma = 5 ] }$ </td><td>887</td><td>5636</td><td>22.1 / 4.7</td><td>94.0</td><td>102</td><td>3778</td><td>30.1 / 7.7</td><td>93.1</td></tr><tr><td> $\mathrm { I P E D } _ { [ \sigma = 1 0 ] }$ </td><td>887</td><td>5636</td><td>26.6 / 5.8</td><td>94.1</td><td>102</td><td>3778</td><td>35.5 / 8.7</td><td>93.3</td></tr><tr><td> $\mathrm { I P E D } _ { [ \sigma = 1 5 ] }$ </td><td>887</td><td>5636</td><td>33.4 / 7.2</td><td>94.2</td><td>102</td><td>3778</td><td>40.6 / 10.2</td><td>93.4</td></tr></table>

Table 3: Comparison of model efficiency. Training Time means the time (seconds) to train one epoch. GPU Mem stands for memory (MB) occupation during inference with the batch size of 8, and Infer. Time (1/8) donates the time (ms) to process each sentence with the batch sizes of 1 and 8, respectively. The superscript indicates the results reported by OD-RTE. All experiments are conducted on a single GeForce RTX 3090 with default configuration.

![](images/b09b06718f50782a1d4e8d45c7a03334f0e35f763d529dc1dc950c2d55b7a917.jpg)  
Figure 3: Performance of IPED with different number of denoising blocks D in terms of F1-score on NYT.

## 4.7 Analysis on Sampling Number

In the denoising inference process, the number of denoising blocks, denoted as D, is a crucial parameter. We conducted additional experiments on it with different sampling timestep σ to evaluate its impact on F1-score and inference time. As depicted in Figure 3, the F1-score decreases sharply when D is less than 15 and remains stable when D exceeds

![](images/66387621b9c2da0933e3c2b3211bf45ba379f5351ec11972a0fbb843275dc22e.jpg)  
Figure 4: Performance of IPED with different number of denoising blocks D in terms of inference time on WebNLG. Note that the batch size is 8 during inference.

25. It can be observed from Figure 4 that the inference time increases with larger D values, especially when σ is relatively small. Regarding the sampling timestep σ, these two figures indicate that a larger σ brings about a higher F1-score but also increases inference time. To balance the F1-score and inference time, we set D at 30 and σ at 10 as our standard configuration. Consequently, our IPED is capable of properly covering all potential blocks, thereby enhancing the recall rate while ensuring optimal inference time for practical applications.

## 4.8 Ablation Study

Ablation experiments are conducted to explore the contributions of the primary components within the network architecture and the effectiveness of level diffusion, as shown in Table 4. Observations reveal that removing any of the three components leads to a relative performance drop. Each of these three components is a critical part for representation construction, with the Co-Attention module having the most influence. Upon replacing the Co-Attention module with the simple addition of two input representations, a 1.2% F1 decline is observed. The experiments indicate that all three modules in our network play a crucial role in recovering blocks from noise.

<table><tr><td>Model</td><td>P</td><td>R</td><td>F</td></tr><tr><td>IPED</td><td>93.0</td><td>93.6</td><td>93.3</td></tr><tr><td>w/o Co-Attention</td><td>91.9</td><td>92.2</td><td>92.1</td></tr><tr><td>w/o Biaffine</td><td>92.2</td><td>93.0</td><td>92.6</td></tr><tr><td>w/o Cross Attention</td><td>92.1</td><td>92.5</td><td>92.3</td></tr><tr><td>w/o Level</td><td>90.6</td><td>91.6</td><td>91.1</td></tr></table>

Table 4: Ablation study on WebNLG dataset.

It is noteworthy that the performance decreases by 2.2% when Level is omitted. This implies that IPED abandons the denoising diffusion process at the block Level, transitioning the task from threedimensional to two-dimensional denoising. Specifically, noisy blocks are distributed across each level of the three-dimensional tables, with each block constrained to denoising at a specific level, thus precluding the possibility of progressive refinement with the block level. Thus it can be concluded that block-level denoising is crucial for the effectiveness of our block-denoising diffusion model in identifying triple relations, particularly in largerelation datasets like WebNLG.

## 5 Conclusion

This paper proposes an implicit approach to relational triple extraction, diverging from the explicit tagging methods of prior table-filling methods, thereby addressing several prevailing issues. Via denoising the edges and levels of noisy blocks, our introduced block-denoising diffusion model incrementally generates ground truth blocks, which can be swiftly and precisely converted into triples with our decoding algorithm PBES. Moreover, our network architecture incorporates beneficial modules such as Co-Attention and Biaffine, which promote the fusion of diverse representations. Experimental results on public datasets demonstrate that our IPED exceeds the performance of state-of-theart (SoTA) models, while also achieving significantly faster inference speeds.

## Limitations

Two limitations of IPED warrant discussion. Firstly, IPED exhibits a substantial increase in training time consumption compared to some models, as detailed in Section 4.6. This can be attributed to the extensive denoising timestep required for training, leading to slow and fluctuating convergence, thereby necessitating a greater number of training epochs. Secondly, the application of our implicit perspective is currently limited to relational triple extraction. Such perception holds potential for broader application in information extraction tasks such as document-level relation extraction and event extraction, addressing the issue of redundant negative tagging inherent in tablefilling. These possibilities could be explored in future work.

## Acknowledgements

The authors convey thanks to all anonymous reviewers for their valuable feedback. This work was supported by the Shenzhen Science and Technology Program (JCYJ20230807094104009) and by Key Lab of Information Network Security, Ministry of Public Security.

## References

Guanqun Bi, Lei Shen, Yanan Cao, Meng Chen, Yuqiang Xie, Zheng Lin, and Xiaodong He. 2023. DiffusEmp: A diffusion model-based framework with multi-grained control for empathetic response generation. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2812–2831, Toronto, Canada. Association for Computational Linguistics.

Paolo Carraresi and Claudio Sodini. 1986. An efficient algorithm for the bipartite matching problem. European Journal ofOperational Research, 23(1):86–93.

Richard J. Chen, Ming Y. Lu, Wei-Hung Weng, Tiffany Y. Chen, Drew FK. Williamson, Trevor Manz, Maha Shady, and Faisal Mahmood. 2021. Multimodal co-attention transformer for survival prediction in gigapixel whole slide images. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 3995–4005.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages

4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Timothy Dozat and Christopher D. Manning. 2016. Deep biaffine attention for neural dependency parsing. ArXiv, abs/1611.01734.

Tsu-Jui Fu, Peng-Hsuan Li, and Wei-Yun Ma. 2019. GraphRel: Modeling text as relational graphs for joint entity and relation extraction. In Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 1409–1418, Florence, Italy. Association for Computational Linguistics.

Chen Gao, Xuan Zhang, LinYu Li, JinHong Li, Rui Zhu, KunPeng Du, and QiuYing Ma. 2023. Ergm: A multi-stage joint entity and relation extraction with global entity match. Knowledge-Based Systems, 271:110550.

Claire Gardent, Anastasia Shimorina, Shashi Narayan, and Laura Perez-Beltrachini. 2017. Creating training corpora for NLG micro-planners. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 179–188, Vancouver, Canada. Association for Computational Linguistics.

Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and Lingpeng Kong. 2023. Diffuseq-v2: Bridging discrete and continuous text spaces for accelerated seq2seq diffusion models.

Zhengfu He, Tianxiang Sun, Qiong Tang, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. 2023. DiffusionBERT: Improving generative masked language models with diffusion models. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4521–4534, Toronto, Canada. Association for Computational Linguistics.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851. Curran Associates, Inc.

Arzoo Katiyar and Claire Cardie. 2017. Going out on a limb: Joint extraction of entity mentions and relations without dependency trees. In Proceedings ofthe 55th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 917–928, Vancouver, Canada. Association for Computational Linguistics.

Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. 2021. Diffwave: A versatile diffusion model for audio synthesis. In International Conference on Learning Representations.

Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. 2016. Neural architectures for named entity recognition. In Proceedings of the 2016 Conference of the North

American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, pages 260–270, San Diego, California. Association for Computational Linguistics.

Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. 2022a. Diffusionlm improves controllable text generation. In Advances in Neural Information Processing Systems, volume 35, pages 4328–4343. Curran Associates, Inc.

Zhe Li, Luoyi Fu, Xinbing Wang, Haisong Zhang, and Chenghu Zhou. 2022b. RFBFN: A relation-first blank filling network for joint relational triple extraction. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop, pages 10–20, Dublin, Ireland. Association for Computational Linguistics.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net.

Yi Luan, Luheng He, Mari Ostendorf, and Hannaneh Hajishirzi. 2018. Multi-task identification of entities, relations, and coreference for scientific knowledge graph construction. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 3219–3232, Brussels, Belgium. Association for Computational Linguistics.

Makoto Miwa and Mohit Bansal. 2016. End-to-end relation extraction using LSTMs on sequences and tree structures. In Proceedings ofthe 54th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1105–1116, Berlin, Germany. Association for Computational Linguistics.

Jinzhong Ning, Zhihao Yang, Yuanyuan Sun, Zhizheng Wang, and Hongfei Lin. 2023. OD-RTE: A one-stage object detection framework for relational triple extraction. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11120–11135, Toronto, Canada. Association for Computational Linguistics.

Feiliang Ren, Longhui Zhang, Shujuan Yin, Xiaofeng Zhao, Shilei Liu, Bochao Li, and Yaduo Liu. 2021. A novel global feature-oriented relational triple extraction model based on table filling. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 2646–2656, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Feiliang Ren, Longhui Zhang, Xiaofeng Zhao, Shujuan Yin, Shilei Liu, and Bochao Li. 2022. A simple but effective bidirectional framework for relational triple extraction. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining, WSDM ’22, page 824–832, New York, NY, USA. Association for Computing Machinery.

Sebastian Riedel, Limin Yao, and Andrew McCallum. 2010. Modeling relations and their mentions without labeled text. In Proceedings of the 2010 European Conference on Machine Learning and Knowledge Discovery in Databases: Part III, ECML PKDD’10, page 148–163, Berlin, Heidelberg. Springer-Verlag.

Yu-Ming Shang, Heyan Huang, and Xianling Mao. 2022. Onerel: Joint entity and relation extraction with one module in one step. Proceedings ofthe AAAI Conference on Artificial Intelligence, 36(10):11285–11293.

Yongliang Shen, Xinyin Ma, Yechun Tang, and Weiming Lu. 2021. A trigger-sense memory flow framework for joint entity and relation extraction. In Proceedings of the Web Conference 2021, WWW ’21, page 1704–1715, New York, NY, USA. Association for Computing Machinery.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2023. Diffusion-NER: Boundary diffusion for named entity recognition. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3875–3890, Toronto, Canada. Association for Computational Linguistics.

Jiaming Song, Chenlin Meng, and Stefano Ermon. 2021. Denoising diffusion implicit models. In International Conference on Learning Representations.

Changzhi Sun, Yeyun Gong, Yuanbin Wu, Ming Gong, Daxin Jiang, Man Lan, Shiliang Sun, and Nan Duan. 2019. Joint type inference on entities and relations via graph convolutional networks. In Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 1361–1370, Florence, Italy. Association for Computational Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings ofthe 31st International Conference on Neural Information Processing Systems, NIPS’17, page 6000–6010, Red Hook, NY, USA. Curran Associates Inc.

Jue Wang and Wei Lu. 2020. Two are better than one: Joint entity and relation extraction with tablesequence encoders. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1706–1721, Online. Association for Computational Linguistics.

Yijun Wang, Changzhi Sun, Yuanbin Wu, Hao Zhou, Lei Li, and Junchi Yan. 2021. UniRE: A unified label space for entity relation extraction. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 220–231, Online. Association for Computational Linguistics.

Yucheng Wang, Bowen Yu, Yueyang Zhang, Tingwen Liu, Hongsong Zhu, and Limin Sun. 2020. TPLinker: Single-stage joint extraction of entities and relations

through token pair linking. In Proceedings of the 28th International Conference on Computational Linguistics, pages 1572–1582, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Zhepei Wei, Jianlin Su, Yue Wang, Yuan Tian, and Yi Chang. 2020a. A novel cascade binary tagging framework for relational triple extraction. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1476– 1488, Online. Association for Computational Linguistics.

Zhepei Wei, Jianlin Su, Yue Wang, Yuan Tian, and Yi Chang. 2020b. A novel cascade binary tagging framework for relational triple extraction. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 1476– 1488, Online. Association for Computational Linguistics.

Benfeng Xu, Quan Wang, Yajuan Lyu, Yabing Shi, Yong Zhu, Jie Gao, and Zhendong Mao. 2022. EmRel: Joint representation of entities and embedded relations for multi-triple extraction. In Proceedings of the 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 659–665, Seattle, United States. Association for Computational Linguistics.

Yue Yuan, Xiaofei Zhou, Shirui Pan, Qiannan Zhu, Zeliang Song, and Li Guo. 2020. A relation-specific attention network for joint entity and relation extraction. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20, pages 4054–4060. International Joint Conferences on Artificial Intelligence Organization. Main track.

Mohamad Zamini, Hassan Reza, and Minou Rabiei. 2022. A review of knowledge graph completion. Information, 13(8).

Yunqi Zhang, Yubo Chen, and Yongfeng Huang. 2022. RelU-net: Syntax-aware graph U-net for relational triple extraction. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 4208–4217, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie Zhou, and Jiwen Lu. 2023. Unleashing text-toimage diffusion models for visual perception. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 5729–5739.

Hengyi Zheng, Rui Wen, Xi Chen, Yifan Yang, Yunyan Zhang, Ziheng Zhang, Ningyu Zhang, Bin Qin, Xu Ming, and Yefeng Zheng. 2021. PRGC: Potential relation and global correspondence based joint relational triple extraction. In Proceedings of the 59th Annual Meeting ofthe Associationfor Computational

Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6225–6235, Online. Association for Computational Linguistics.

## A Dataset Statistics

The statistical details of the two datasets are displayed in Table 5.

## B Clarification for D, N and M

In our paper, N is the number of blocks after expansion for training, M is the number of ground truth blocks for training, and D is the number of initialized blocks for inference.

During training, there are M blocks at first, which are then expanded by adding N-M randomly sampled blocks, resulting in a total of N blocks. Specifically, N and D are two similar hyperparameters; N is used for training while D is for inference, and typically, both are larger than M. To clearly distinguish between training and inference in our paper, we have defined N and D separately for the readers.

<table><tr><td rowspan="2">Dataset</td><td colspan="3">Sentences</td><td colspan="9">Details of test set</td></tr><tr><td>Train</td><td>Valid</td><td>Test</td><td>Normal</td><td>SEO</td><td>EPO</td><td>SOO</td><td>Q=1</td><td>Q=2</td><td>Q&gt;2</td><td>Relations</td><td>Triples</td></tr><tr><td>NYT</td><td>56196</td><td>5000</td><td>5000</td><td>3071</td><td>1273</td><td>1168</td><td>117</td><td>3089</td><td>1047</td><td>864</td><td>24</td><td>8616</td></tr><tr><td>NYT*</td><td>56195</td><td>4999</td><td>5000</td><td>3266</td><td>1297</td><td>978</td><td>45</td><td>3244</td><td>1045</td><td>711</td><td>24</td><td>8110</td></tr><tr><td>WebNLG</td><td>5019</td><td>500</td><td>703</td><td>239</td><td>448</td><td>6</td><td>85</td><td>256</td><td>175</td><td>272</td><td>216</td><td>1607</td></tr><tr><td>WebNLG*</td><td>5019</td><td>500</td><td>703</td><td>245</td><td>457</td><td>26</td><td>84</td><td>266</td><td>171</td><td>266</td><td>171</td><td>1591</td></tr></table>

Table 5: Statistics of datasets used in our experiments. Q represents the number of triples in a sentence. Note that a single sentence can simultaneously contain SEO, EPO and SOO overlapping patterns.