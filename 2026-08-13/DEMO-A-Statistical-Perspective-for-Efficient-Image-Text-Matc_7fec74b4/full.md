# DEMO: A Statistical Perspective for Efficient Image-Text Matching

Fan Zhang<sup>1</sup>, Xian-Sheng Hua<sup>2</sup>, Chong Chen<sup>2</sup>, Xiao Luo<sup>3†</sup>

<sup>1</sup>Georgia Tech Shenzhen Institute, Tianjin University (GTSI)

<sup>2</sup>Terminus Group <sup>3</sup>University of California, Los Angeles

fanzhang@gatech.edu, huaxiansheng@gmail.com, chenchong.cz@gmail.com, xiaoluo@cs.ucla.edu

## Abstract

Image-text matching has been a long-standing problem, which seeks to connect vision and language through semantic understanding. Due to the capability to manage large-scale raw data, unsupervised hashing-based approaches have gained prominence recently. They typ ically construct a semantic similarity structure using the natural distance, which subse quently provides guidance to the model op timization process. However, the similarity structure could be biased at the boundaries of semantic distributions, causing error accumulation during sequential optimization. To tackle this, we introduce a novel hashing approach termed Distribution-based Structure Mining with Consistency Learning (DEMO) for effi cient image-text matching. From a statistical view, DEMO characterizes each image using multiple augmented views, which are consid ered as samples drawn from its intrinsic se mantic distribution. Then, we employ a nonparametric distribution divergence to ensure a robust and precise similarity structure. In addition, we introduce collaborative consistency learning which not only preserves the similarity structure in the Hamming space but also encour ages consistency between retrieval distribution from different directions in a self-supervised manner. Through extensive experiments on three benchmark image-text matching datasets, we demonstrate that DEMO achieves superior performance compared with many state-of-theart methods.

## 1 Introduction

Image-text matching (Sun et al., 2023; Zhang et al., 2022b; Huang et al., 2022; Liu and Ye, 2019; Hu et al., 2023b) is a pivotal task in both computer vision and natural language processing, which bridges data across heterogeneous modalities. The objective is to return images correlated with a given textual description and detect texts corresponding to a given image. Considering explosively growing web data (Krotov and Johnson, 2023), there is a significant demand for an efficient approach that can select a small candidate set from a comprehensive dataset. Towards this end, hashing has become prevalent in information retrieval (Luo et al., 2021a), particularly image-text matching (Hu et al., 2023a; Sun et al., 2022a; Tu et al., 2023a; Zeng et al., 2023; Cao et al., 2022), which involves mapping both texts and images into a shared binary space (Hamming space), and then determining cross-modal similarity scores by comparing their binary codes.

![](images/1aa7f044f092d1e46ab8c01109775957955d4548dc5ad5848f3885bf83d59dfa.jpg)  
Figure 1: Comparison between cosine distance and energy distance. We leverage the randomness of data augmentation to estimate the latent semantics distributions, and then use energy distance between distributions as a substitute for cosine distance between data points.

In literature, numerous approaches have been developed for cross-modal hashing (Jiang and Li, 2017; Kaur et al., 2021), which can broadly be categorized into supervised and unsupervised methods. Supervised methods (Chen et al., 2019; Jia et al., 2021; Gu et al., 2019) typically incorporate ground truth similarities into a pairwise (Fan et al., 2023) or rankwise (Liu et al., 2023) loss objective. However, due to the high costs associated with label annotation, unsupervised approaches (Tu et al., 2023a; Zeng et al., 2023; Cao et al., 2022) tend to be more appreciated in real-world applications. Unsupervised cross-modal hashing approaches typically begin by reconstructing the similarity structure between different modalities, which subsequently provides guidance during the learning process of the hashing model.

Despite the notable advancements, prevailing unsupervised cross-modal hashing approaches (Tu et al., 2023a; Zeng et al., 2023; Cao et al., 2022) still suffer from two major limitations: (1) Biased Similarity Structure. These approaches typically employ natural distances (e.g., cosine distance) to generate the semantic similarity structure. Since deep features with the same semantics should be from a high-dimensional distribution, utilizing cosine distance would be imprecise at the distribution boundaries, which generates noisy supervision, and serious error accumulation during subsequent optimization procedures. (2) Distribution Discrepancy Across Modalities. Given the inherent heterogeneity, different networks are utilized to generate binary codes, which could obey distinct distributions in the Hamming space. This distribution discrepancy inherently undermines the effectiveness of cross-modal retrieval and brings suboptimal results.

To handle these limitations, in this work, we propose a new hashing approach named Distributionbased Structure Mining with Consistency Learning (DEMO) for efficient image-text matching. The core of our DEMO revolves around exploring the latent semantic distribution of each sample using multiple random augmentations. In particular, given that data augmentation generally maintains the semantics (Dai et al., 2023), we consider each augmented view of an image as samples drawn from its intrinsic semantic distribution. Then a non-parametric metric (i.e., energy distance (Rizzo and Székely, 2016)) is incorporated to precisely measure the distribution divergence (see Figure 1), thereby reconstructing a robust and accurate semantic structure. The subsequent optimization of the hashing network is achieved by preserving this semantic structure in the Hamming space. Furthermore, to diminish the distribution shift across modalities, we generate cross-modal retrieval distributions given both queries of images and texts and their consistency are promoted in a self-supervised manner. In addition, we employ a sharpening operation to refine retrieval results by emphasizing points with high degrees of similarity. We conduct comprehensive experiments on three benchmark image-text matching datasets, and the results show that our DEMO outperforms a wide range of competing methods. In brief, the main contribution of this paper can be summarized as follows:

• Innovative Perspective. We explore the latent semantics distribution and adopt the distribution divergence to construct a robust and accurate semantics structure to guide unsupervised crossmodal hashing through a statistical perspective.

• Coherent Framework. DEMO optimizes the modality-specific hashing networks by preserving the semantics structure in the Hamming space. Additionally, DEMO promotes consistency between cross-modal retrieval distributions, resulting in modality-invariant binary descriptors.

• Outstanding Performance. Comprehensive experiments reveal that DEMO outperforms various state-of-the-art hashing-based methods on image-text matching benchmark datasets.

## 2 Related Work

## 2.1 Image-text Matching

Image-text matching is a fundamental problem which can bridge computer vision and natural language processing (Sun et al., 2023; Zhang et al., 2022b; Huang et al., 2022; Liu and Ye, 2019; Hu et al., 2023b). Recent approaches can be divided into local-level and global-level approaches. Locallevel matching approaches (Liu et al., 2019a; Chen et al., 2020; Zhang et al., 2022a; Dong et al., 2022; Fu et al., 2023; Bhattacharyya et al., 2022) take the input of image-text pairs to learn fine-grained relationships, such as region-word alignments. In contrast, global-level matching approaches (Tu et al., 2021; Lu et al., 2022; Radford et al., 2019; Jia et al., 2021) map both images and texts into a shared space and then calculate their latent embedding similarities. To enhance the efficiency of imagetext matching, this paper proposes a novel hashing method termed DEMO for binary descriptors, which enables the calculation of similarity using the efficient “XOR" operation (Gu et al., 2022).

## 2.2 Unsupervised Cross-modal Hashing

Cross-modal hashing (Hu et al., 2023a; Sun et al., 2022a; Tu et al., 2023a; Zeng et al., 2023; Cao et al., 2022) attempts to project samples from various modalities into a shared binary space in which samples with similar semantics should be close. Early efforts typically investigate hand-crafted features for hash codes (Song et al., 2013; Zhou et al.,

![](images/95597374b3137141874a8ee07a49f8141460450b2a2d3f30481fd4967bec0cac.jpg)  
Figure 2: An overview of our proposed DEMO. DEMO first calculates the energy distance between latent semantics distributions to generate an instance similarity matrix. Then DEMO simultaneously optimizes the modality-specific hashing networks by preserving the similarity with guided consistency learning. In addition, retrieval distributions using both image and text queries are encouraged to be consistent to obtain modality-invariant binary codes.

2014), which are typically not discriminative to preserve similarity structure. Recently, various deep unsupervised cross-modal hashing approaches have been developed (Gao et al., 2023; Mikriukov et al., 2022), which typically reconstruct the similarity structure based on cosine distances to optimize the process of learning to hash. However, these methods are incapable of producing precise supervision signals, resulting in inferior binary hash codes. Towards this end, we investigate the latent distribution for each sample and adopt the distribution divergence for enhanced semantic structures.

## 3 The Proposed Approach

## 3.1 Problem Definition and Overview

Problem Definition. We begin with notations and the formal definition. $X = \{ x _ { i } \} _ { i = 1 } ^ { N }$ represents a dataset consisting of N images and $\mathbf { \bar { \xi } }$ represent a dataset with N texts. Each ${ \bf { \nabla } } \mathbf { \mathbf { { y } } } _ { i }$ is associated with text embeddings $\mathbf { \Delta } _ { t _ { i } }$ . The objective is to map these samples into a shared Hamming space. We expect the matched samples between two modalities to be encoded into similar binary codes with small Hamming distances. This mapping can guarantee effective and efficient imagetext matching.

Framework Overview. This work proposes a new cross-modal hashing approach named DEMO for efficient image-text matching. As depicted in Figure 2, DEMO first employs a pre-trained feature extractor $F ^ { v } ( \cdot )$ for images, which removes the last layer of a well-known classification neural network (Tu et al., 2023a). We also extract text embeddings using an embedding layer $F ^ { t } ( \cdot )$ . Then, two feed-forward networks $( \mathrm { F F N s } ) , \phi ^ { v } ( \cdot )$ and $\phi ^ { t } ( \cdot )$ are adopted to map features of images and texts into binary codes, respectively. Formally, we have:

$$
{ \pmb b } _ { i } ^ { v } = s g n ( \phi ^ { v } ( F ^ { v } ( { \pmb x } _ { i } ) ) ) ,\tag{1}
$$

$$
{ \pmb b } _ { i } ^ { t } = s g n ( \phi ^ { t } ( F ^ { t } ( { \pmb y } _ { i } ) ) ) ,\tag{2}
$$

where $s g n ( \cdot )$ is the sign function. Our DEMO mainly consists of two modules, (1) Distributionbased Structural Mining. We delve into the inherent semantics distribution behind each image using random data augmentation and utilize the distribution divergence to reconstruct an accurate semantic structure, which would effectively guide the optimization of hashing networks. (2) Collaborative Consistency Learning. On the one hand, we maximize the consistency of similarity scores between the semantic structure and hash codes. On the other hand, we produce cross-modal retrieval distributions given texts and images and encourage their consistency from opposing directions.

## 3.2 Distribution-based Structural Mining

A pivotal challenge in unsupervised cross-modal hashing lies in the lack of supervised information. Previous approaches (Yu et al., 2021; Tu et al., 2023b; Hu et al., 2023a) typically reconstruct the similarity structure as supervision by measuring the natural distance (e.g., cosine distance) of deep features. However, the reconstructed structure may introduce noise, leading to significant error accumulation throughout subsequent optimization stages. In particular, we observe that deep features with the same semantics should originate from a highdimensional distribution (Sun et al., 2022b; Yang et al., 2018; Tu et al., 2020), and the natural distance could be inaccurate at the boundaries of latent distributions. Consequently, we aim to measure the distribution divergence for effective structural mining, ensuring high-quality hash codes for efficient image-text matching.

Firstly, we take the image dataset as an example of similarity structure mining. In particular, the random vector of each example $\mathbf { \mathcal { x } } _ { i }$ in the embedding space is represented as $\pmb { \xi } _ { i }$ with the cumulative distribution function $G _ { i }$ . Then, the distribution divergence between the underlying semantic distributions of $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ and $\mathbf { \boldsymbol { x } } _ { j }$ is formulated as:

$$
d ( { \pmb x } _ { i } , { \pmb x } _ { j } ) = \psi ( G _ { i } , G _ { j } ) ,\tag{3}
$$

in which $\psi$ is a given metric. However, due to immense complexity, parameterizing the highdimensional distributions remains a considerable challenge. Therefore, classic methods such as KL divergence and JS divergence are not inappropriate here. Towards this end, we turn to a nonparametrized metric, i.e., energy distance (Székely and Rizzo, 2013) . This metric enables modeling of the distribution divergence without the derivation of specific distribution functions, providing an effective alternative for handling the challenges in the high-dimensional space.

Definition 1 (Energy Distance). Given two independent random vectors ξ and ζ with the cumulative distribution functions $G _ { \xi }$ and $G _ { \zeta } ,$ respectively. We construct two independent copies $\dot { \pmb { \xi } } ^ { \prime }$ and $\zeta ^ { \prime }$ from these cumulative distributionfunctions. Then, the energy distance is defined as:

$$
D ^ { 2 } ( G _ { \xi } , G _ { \zeta } ) = 2 \mathrm { E } \rho ( \xi , \zeta ) - \mathrm { E } \rho \left( \xi , \xi ^ { \prime } \right) - \mathrm { E } \rho \left( \zeta , \zeta ^ { \prime } \right)\tag{4}
$$

where $\rho ( \cdot , \cdot )$ is a pointwise distance metric such as cosine distance.

When random variables are real-valued, we can rewrite Eqn. 4 into:

$$
D ^ { 2 } ( G _ { \xi } , G _ { \zeta } ) = \int _ { - \infty } ^ { \infty } \rho ^ { 2 } ( G _ { \xi } ( x ) , G _ { \zeta } ( x ) ) d x .\tag{5}
$$

From Eqn. 5, we can infer that $D ^ { 2 } ( G _ { \xi } , G _ { \zeta } ) \geq 0$ and the equality holds when two distributions are identical. In non-parametric test, we generate statistical samples $\{ \pmb { u } _ { 1 } , \cdots , \pmb { u } _ { M } \}$ and $\{ \pmb { v } _ { 1 } , \cdots , \pmb { v } _ { M } \}$ from $G _ { \xi }$ and $G _ { \zeta }$ , respectively. Then, we explore the statistics for the null hypothesis, i.e., $G _ { \xi } = G _ { \zeta }$ by calculating the following averages:

$$
\begin{array} { r l } & { A = \frac { 1 } { M ^ { 2 } } \sum _ { m = 1 } ^ { M } \sum _ { m ^ { \prime } = 1 } ^ { M } \rho \left( \pmb { u } _ { m } , \pmb { v } _ { m ^ { \prime } } \right) } \\ & { B = \frac { 1 } { M ^ { 2 } } \sum _ { m = 1 } ^ { M } \sum _ { m ^ { \prime } = 1 } ^ { M } \rho \left( \pmb { u } _ { m } , \pmb { u } _ { m ^ { \prime } } \right) } \\ & { C = \frac { 1 } { M ^ { 2 } } \sum _ { m = 1 } ^ { M } \sum _ { m ^ { \prime } = 1 } ^ { M } \rho \left( \pmb { v } _ { m } , \pmb { v } _ { m ^ { \prime } } \right) } \end{array} .\tag{6}
$$

The statistics (Székely and Rizzo, 2013) can be formulated as:

$$
\mathcal { E } \left( \{ \pmb { u } _ { m } \} _ { m = 1 } ^ { M } , \{ \pmb { v } _ { m } \} _ { m = 1 } ^ { M } \right) = 2 A - B - C ,\tag{7}
$$

where $\mathcal { E } ( \cdot , \cdot )$ denotes energy distance. A large energy distance would reject the null hypothesis, indicating different distribution functions. Since the labels of unlabeled samples cannot acquired, we turn to data augmentation (Sun et al., 2022b; Luo et al., 2021b; He et al., 2020). In particular, we view the augmented view of each image $\mathbf { \delta } _ { \mathbf { \mathcal { X } } _ { i } }$ as the samples from its underlying semantic distribution $G _ { i }$ since data augmentation would typically retain the semantics. Therefore, the distribution divergence between $\mathbf { \delta } _ { \mathbf { \mathcal { X } } _ { i } }$ and $\boldsymbol { \mathscr { x } } _ { j }$ can be estimated as:

$$
d \left( \pmb { x } _ { i } , \pmb { x } _ { j } \right) = \mathcal { E } \left( \left\{ \pmb { z } ^ { \prime } _ { i m } \right\} _ { m = 1 } ^ { M } , \left\{ \pmb { z } ^ { \prime } _ { j m } \right\} _ { m = 1 } ^ { M } \right)\tag{8}
$$

where $z ^ { \prime } { } _ { i m } = F ^ { v } ( \pmb { x } _ { i m } ^ { \prime } )$ is the deep feature of the augmented view $\mathbf { \Delta } x _ { i m } ^ { \prime } .$ . In our implementation, we use cosine distance for $\rho ( \cdot , \cdot )$ . Finally, we set a threshold τ to reject the null hypothesis and thus the pair with the distance below the threshold is considered as positive. Moreover, we notice there are still fine-grained differences among dissimilar pairs. Towards this end, we introduce image and text similarities in the semantics structure:

$$
S _ { i j } ^ { v } = \rho ( \sum _ { m = 1 } ^ { M } { z ^ { \prime } { _ { i m } } } , \sum _ { m = 1 } ^ { M } { z ^ { \prime } { _ { j m } } } ) ,\tag{9}
$$

$$
S _ { i j } ^ { t } = \rho ( t _ { i } , t _ { j } ) ,\tag{10}
$$

where $\mathbf { \Delta } \mathbf { \mathbf { \mathit { t } } } _ { i }$ is the text embedding of ${ \bf { \nabla } } \mathbf { \mathbf { { y } } } _ { i }$ . We combine $S _ { i j } ^ { v }$ and $S _ { i j } ^ { t }$ to depict the similarities when $d ( { \pmb x } _ { i } , { \pmb x } _ { j } ) \geq \tau$ . In formulation, we construct the instance similarity structure as follows:

$$
S _ { i j } = \left\{ \begin{array} { l l } { 1 , } & { d ( { \pmb x } _ { i } , { \pmb x } _ { j } ) < \tau } \\ { \alpha S _ { i j } ^ { v } + ( 1 - \alpha ) S _ { i j } ^ { t } , } & { o t h e r w i s e , } \end{array} \right.\tag{11}
$$

where $\alpha$ is a coefficient to balance two similarities (Tu et al., 2023b, 2020; Ma et al., 2022). It can be noticed that when $M = 1$ , our distribution divergence would be degraded to the fundamental cosine distance. The incorporation of multiple augmented views makes it more robust against random attacks. Moreover, it alleviates biases for examples at the boundary of latent semantic distributions, ensuring the accuracy of structural mining.

## 3.3 Optimization with Collaborative Consistency Learning

In this part, we jointly optimize the image and text hashing networks using collaborative consistency learning which mainly includes guided consistency learning and retrieval-based consistency learning. Guided Consistency Learning. After constructing the similarity structure (Luo et al., 2021b; Yang et al., 2018; Tu et al., 2020), we aim to preserve this in produced hash codes. In particular, we generate hash codes for both images and texts and then produce their similarities, which would be consistent with the reconstructed structure. Formally, we have:

$$
\mathcal { L } _ { g u i } = \sum _ { i , j = 1 } ^ { N } \sum _ { e _ { 1 } , e _ { 2 } \in \{ v , t \} } | | \rho ( b _ { i } ^ { e _ { 1 } } , b _ { j } ^ { e _ { 2 } } ) - S _ { i j } | | ^ { 2 } ,\tag{12}
$$

where $e _ { 1 }$ and $e _ { 2 }$ indicate the selected modalities. Therefore, image-image, text-text, and image-text consistency are jointly considered and mapped to the similarity structure under the guidance.

Retrieval-based Consistency Learning. To further reduce the potential distribution discrepancy between the two modalities (Lu et al., 2022; Wei et al., 2021), we simulate the cross-modal retrieval procedure in different directions and enforce the consistency between the retrieval results. In formulation, given a batch, the probability distribution corresponding to text-to-image retrieval is written as:

$$
\pmb { p } _ { i } ^ { T 2 I } = [ \rho ( \pmb { b } _ { i } ^ { t } , \pmb { b } _ { 1 } ^ { v } ) , \cdots , \rho ( \pmb { b } _ { i } ^ { t } , \pmb { b } _ { B } ^ { v } ) ] ,\tag{13}
$$

where B denotes the batch size. Similarly, the probability distribution corresponding to image-totext retrieval is:

$$
\pmb { p } _ { i } ^ { I 2 T } = [ \rho ( \pmb { b } _ { i } ^ { v } , \pmb { b } _ { 1 } ^ { t } ) , \cdots , \rho ( \pmb { b } _ { i } ^ { v } , \pmb { b } _ { B } ^ { t } ) ] .\tag{14}
$$

Then, we utilize the sharpening operator (Xie et al., 2016; Assran et al., 2021; Wang et al., 2023) to refine the soft distributions with:

$$
\delta ( { \pmb p } ) _ { b } = \frac { [ { \pmb p } ] _ { b } ^ { 1 / T } } { \sum _ { b ^ { \prime } = 1 } ^ { B } [ { \pmb p } ] _ { b ^ { \prime } } ^ { 1 / T } } , b = 1 , \cdots , B .\tag{15}
$$

Our sharpening operation is capable of enhancing the purification of the retrieval results and emphasizing the samples with high similarities. Finally, we conduct consistency learning across two directions in a self-supervised fashion using:

$$
\begin{array} { c } { { \displaystyle { \mathcal { L } _ { r e t } = \sum _ { i = 1 } ^ { B } ( K L ( \delta ( \pmb { p } _ { i } ^ { I 2 T } ) | | \pmb { p } ^ { T 2 I } ) } } } \\ { { + K L ( \delta ( \pmb { p } _ { i } ^ { T 2 I } ) | | \pmb { p } ^ { I 2 T } ) ) , } } \end{array}\tag{16}
$$

where $K L ( \cdot | | \cdot )$ returns the KL divergence of two distributions and $T$ is a temperature coefficient that controls the sharp degree set to 0.25 empirically.

Besides, we leverage the co-occurrence knowledge embedded in the dataset, which enforces binary codes of images and texts with identical objects to be close. In particular, we have:

$$
\mathcal { L } _ { c o } = \sum _ { i = 1 } ^ { N } | | \rho ( \boldsymbol { b } _ { i } ^ { v } , \boldsymbol { b } _ { j } ^ { t } ) - \gamma | | ^ { 2 } ,\tag{17}
$$

where $\gamma$ is set to 1.5 empirically (Tu et al., 2023a) to emphasize this accurate embedding knowledge.

In a nutshell, we summarize our framework by combining all these objectives:

$$
\mathcal { L } = \mathcal { L } _ { g u i } + \mathcal { L } _ { r e t } + \mathcal { L } _ { c o } .\tag{18}
$$

However, directly minimizing Eqn. 18 is infeasible since $s g n ( \cdot )$ is not differentiable at zero and its derivative is zero at the other point. To tackle this problem, we replace $s g n ( \cdot )$ with tanh( ) during optimization, which results in approximate hash codes $\hat { \pmb { b } } _ { i } ^ { v } \ = \ t a n h ( \phi ^ { v } ( F ^ { v } ( { \pmb x } _ { i } ) ) )$ and $\hat { b } _ { t } ^ { v } \ =$ tan $h ( \phi ^ { t } ( F ^ { t } ( { \pmb y } _ { i } ) ) )$ . We summarize the whole training algorithm of DEMO in Algorithm 1.

## 3.4 Model Inference

After the optimization procedure, we would feed each sample into the hashing network for a binary descriptor. Then, given each query text $y _ { q }$ (image $x _ { q } )$ with a binary code $b _ { q } ^ { t } \ ( b _ { q } ^ { v } )$ , we rank the Hamming distances between $b _ { q } ^ { t } ~ ( b _ { q } ^ { v } )$ and $\{ b _ { i } ^ { v } \} _ { i = 1 } ^ { N }$ $( \{ b _ { i } ^ { t } \} _ { i = 1 } ^ { N } )$ , which can produce the nearest examples efficiently. In practice, we consider the returned samples as candidates and conduct fine-grained matching for the final results (Tu et al., 2021).

## 4 Experiment

## 4.1 Experimental Settings

Datasets and Evaluation Metrics. To assess the performance of our DEMO, we employ three public and widely-used benchmark datasets to conduct experiments, including MIRFlickr-25K, NUS-WIDE, and MS-COCO. MIRFlickr-25K comprises

<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td colspan="4">MIRFlickr-25K</td><td rowspan="2"></td><td colspan="3">NUS-WIDE</td><td rowspan="2"></td><td colspan="4">MS-COCO</td></tr><tr><td>16 bits</td><td>32 bits</td><td>64 bits</td><td>128 bits</td><td>16 bits</td><td>32 bits</td><td>64 bits</td><td>128 bits</td><td>16 bits 32 bits</td><td>64 bits</td><td>128 bits</td></tr><tr><td rowspan="10">I2T</td><td>CVH LSSH</td><td>0.620</td><td>0.608</td><td>0.594</td><td>0.583</td><td>0.487</td><td>0.495</td><td>0.456</td><td>0.419</td><td></td><td>0.503</td><td>0.504</td><td>0.471</td><td>0.425</td></tr><tr><td></td><td>0.597</td><td>0.609</td><td>0.606</td><td>0.605</td><td>0.442</td><td>0.457</td><td>0.450</td><td>0.451</td><td></td><td>0.484</td><td>0.525</td><td>0.542</td><td>0.551</td></tr><tr><td>CMFH</td><td>0.557</td><td>0.557</td><td>0.556</td><td>0.557</td><td>0.339</td><td>0.338</td><td>0.343</td><td>0.339</td><td></td><td>0.366</td><td>0.369</td><td>0.370</td><td>0.365</td></tr><tr><td>FSH</td><td>0.581</td><td>0.612</td><td>0.635</td><td>0.662</td><td>0.557</td><td>0.565</td><td>0.598</td><td>0.635</td><td></td><td>0.539</td><td>0.549</td><td>0.576</td><td>0.587</td></tr><tr><td>MTFH</td><td>0.507</td><td>0.512</td><td>0.558</td><td>0.554</td><td>0.297</td><td>0.297</td><td>0.272</td><td>0.328</td><td></td><td>0.399</td><td>0.293</td><td>0.295</td><td>0.395</td></tr><tr><td>FOMH</td><td>0.575</td><td>0.640</td><td>0.691</td><td>0.659</td><td>0.305</td><td>0.305</td><td>0.306</td><td>0.314</td><td></td><td>0.378</td><td>0.514</td><td>0.571</td><td>0.601</td></tr><tr><td>DCH</td><td>0.596</td><td>0.602</td><td>0.626</td><td>0.636</td><td>0.392</td><td>0.422</td><td>0.430</td><td>0.436</td><td></td><td>0.422</td><td>0.420</td><td>0.446</td><td>0.468</td></tr><tr><td>DGCPN</td><td>0.651</td><td>0.683</td><td>0.718</td><td>0.724</td><td>0.601</td><td>0.618</td><td>0.631</td><td>0.640</td><td></td><td>0.556</td><td>0.569</td><td>0.578</td><td>0.580</td></tr><tr><td>UCHSTM</td><td>0.701</td><td>0.715</td><td>0.724</td><td>0.723</td><td>0.625</td><td>0.635</td><td>0.646</td><td>0.644</td><td>0.558</td><td></td><td>0.572</td><td>0.576</td><td>0.573</td></tr><tr><td>UCCH DEMO</td><td>0.716 0.718</td><td>0.726 0.733</td><td>0.728 0.734</td><td>0.732 0.743</td><td>0.621</td><td>0.623</td><td>0.640</td><td></td><td>0.645</td><td>0.560</td><td>0.562</td><td>0.566</td><td>0.574</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.646</td><td>0.648</td><td>0.662</td><td>0.664</td><td>0.575</td><td>0.578</td><td>0.586</td><td>0.605</td></tr><tr><td rowspan="10">T2I</td><td>CVH</td><td>0.629</td><td>0.615</td><td>0.599</td><td>0.587</td><td>0.470</td><td>0.475</td><td>0.444</td><td>0.412</td><td>0.506</td><td>0.508</td><td>0.486</td><td>0.429</td></tr><tr><td>LSSH</td><td>0.602</td><td>0.598</td><td>0.598</td><td>0.597</td><td>0.473</td><td>0.482</td><td>0.471</td><td>0.457</td><td>0.490</td><td>0.522</td><td>0.547</td><td>0.560</td></tr><tr><td>CMFH</td><td>0.553</td><td>0.553</td><td>0.553</td><td>0.553</td><td>0.306</td><td>0.306</td><td>0.306</td><td>0.306</td><td>0.346</td><td>0.346</td><td>0.346</td><td>0.346</td></tr><tr><td>FSH</td><td>0.576</td><td>0.607</td><td>0.635</td><td>0.660</td><td>0.569</td><td>0.604</td><td>0.651</td><td>0.666</td><td>0.537</td><td>0.524</td><td>0.564</td><td>0.573</td></tr><tr><td>MTFH</td><td>0.514</td><td>0.524</td><td>0.518</td><td>0.581</td><td>0.353</td><td>0.314</td><td>0.399</td><td>0.410</td><td>0.335</td><td>0.374</td><td>0.300</td><td>0.334</td></tr><tr><td>FOMH</td><td>0.585</td><td>0.648</td><td>0.719</td><td>0.688</td><td>0.302</td><td>0.304</td><td>0.300</td><td>0.306</td><td>0.368</td><td>0.484</td><td>0.559</td><td>0.595</td></tr><tr><td>DCH</td><td>0.612</td><td>0.623</td><td>0.653</td><td>0.665</td><td>0.379</td><td>0.432</td><td>0.444</td><td>0.459</td><td>0.421</td><td>0.428</td><td>0.454</td><td>0.471</td></tr><tr><td>DGCPN</td><td>0.653</td><td>0.682</td><td>0.712</td><td>0.715</td><td>0.605</td><td>0.626</td><td>0.637</td><td>0.644</td><td>0.550</td><td>0.566</td><td>0.578</td><td>0.577</td></tr><tr><td>UCHSTM</td><td>0.695</td><td>0.711</td><td>0.713</td><td>0.723</td><td>0.632</td><td>0.643</td><td>0.651</td><td>0.652</td><td>0.555</td><td>0.567</td><td>0.578</td><td>0.573</td></tr><tr><td>UCCH</td><td>0.703</td><td>0.712</td><td>0.720</td><td>0.721</td><td>0.625</td><td>0.637</td><td>0.650</td><td>0.652</td><td>0.564</td><td>0.573</td><td>0.572</td><td>0.581</td></tr><tr><td></td><td>DEMO</td><td>0.708</td><td>0.719</td><td>0.722</td><td>0.728</td><td>0.654</td><td>0.655</td><td>0.669</td><td>0.671</td><td>0.572</td><td>0.579</td><td>0.583</td><td>0.597</td></tr></table>

Table 1: MAP scores comparison with code length varying from 16 to 128 bits. I2T refers to the image-to-text matching task, and T2I signifies the text-to-image task. The highest scores are shown in boldface.

25,000 pairs of image-text data, and each sample is manually annotated with multiple labels from a set of 24 distinct classes. We remove samples lacking class information, resulting in 20,015 samples for our experiments. We divide these samples into two sets: a query database containing 2,000 paired samples and a retrieval database containing the remaining samples. We employ bag-of-words (BoW) vectors with a dimension of 1,386 to represent the text samples. NUS-WIDE consists of 269,498 paired image-text samples and each sample is assigned to a multilabel category from 81 categories. We select 186,557 samples from the top 10 frequent classes for our experiments. These samples are split into a query database with 2,100 image-text pairs and a retrieval database with the remaining samples. Similarly, we employ 1,000-dimensional BoW vectors to represent the text samples. MS-COCO is a benchmark dataset which consists of 123,287 images. Each image is associated with 5 annotations from 80 categories. After deleting the samples without label annotations, 122,218 pairs remain during the experiment. We choose 5,000 paired image-text samples randomly as the query database and the remaining pairs are left as the retrieval database. Correspondingly, the text samples are represented by 2026-dimensional BoW vectors.

We evaluate the matching performance based on two protocols: the Hamming ranking protocol and the hash lookup protocol. The former is evaluated by the widely used metric Mean Average Precision (MAP) score, and the latter is evaluated by three types of curves: Precision-Recision curve, Precision-top N curve, and Recall-top N curve. For a fair comparison, we report MAP@All scores as default.

Baselines and Implementation Details. We employ 10 state-of-the-art hashing-based image-text matching approaches as baseline methods, including three supervised cross-modal hashing methods (MTFH (Liu et al., 2019b), FOMH (Lu et al., 2019), DCH (Xu et al., 2017)), four shallow unsupervised cross-modal hashing methods (CVH (Kumar and Udupa, 2011), LSSH (Zhou et al., 2014), CMFH (Ding et al., 2016), FSH (Liu et al., 2017)), and three deep unsupervised cross-modal hashing methods (DGCPN (Yu et al., 2021), UCHSTM (Tu et al., 2023b), UCCH (Hu et al., 2023a)). We randomly select 5,000/10,000/all samples from the retrieval database as the training samples for supervised/deep unsupervised/shallow unsupervised cross-modal hashing methods. For a fair comparison, we follow previous works (Tu et al., 2023b; Hu et al., 2023a) and reimplement the deep unsupervised methods, utilizing VGG-19 pre-trained on the ImageNet dataset and a two-layer MLP as the backbone of the image hashing network and text hashing network, respectively. We adopt the SGD algorithm with a learning rate of 1e-3 to optimize the networks. The batch size is set to 128. More hyper-parameters are set according to Section 4.4.

![](images/2d8b0eeb60d139de7a91c9eb8eae9c08ad1b3f9273db6022ea3258c93c062c87.jpg)

![](images/014629340927d80de974901f7396d6881206b00b7327f0247cbcfad5c762494b.jpg)

![](images/ee8eca3f92361babcdbd5364b61816539c5b2e9806821ae29ea66ea7cd721c8c.jpg)

![](images/b374a2f4e7f794139292af21e0a5a4a289080c814a086851de6493851292b860.jpg)

![](images/9a73d5bdc02c96f03f52ad9d6904451e570b24e3d33dd1712fc9b45b426c4891.jpg)

![](images/19a28a4f32c814d69383eefa33d91bbfdc98b013a515889f5c8a4ff2f8ad82a3.jpg)  
Figure 3: The Precision-Recall curve, Precision-top N curve, and Recall-top N curve with 128 bits on MIRFlickr-25K. The first row plots image-to-text results, and the second row plots text-to-image results.

## 4.2 Main Results

Hamming Ranking Protocol. We showcase the MAP scores of all compared baseline methods and our DEMO in Table 1. From these results, the following observations can be attained: First, deep unsupervised cross-modal hashing methods outperform shallow unsupervised cross-modal hashing approaches even with insufficient amounts of training data, indicating the superiority of deep neural networks in generating high-quality and modalityinvariant hash codes. Next, supervised methods excel due to their reliance on expensive labeled data. However, when labeled data is scarce, these methods fall short compared to deep unsupervised approaches. Consequently, deep unsupervised crossmodal hashing emerges as the fundamental technique for image-text matching in the presence of vast amounts of unlabeled multimodal data. Furthermore, DEMO outperforms all the compared state-of-the-art hashing-based image-text matching methods, revealing the effectiveness of our proposed distribution-based structural mining and collaborative consistency learning. Additionally, our approach exhibits consistent and significant improvements across three datasets, highlighting the success of addressing previously overlooked distribution divergence combined with collaborative consistency. The proposed components can enhance the performance of unsupervised hashingbased image-text matching in a robust manner.

Hash Lookup Protocol. We also incorporate the hash lookup protocol to generate Precision-Recall, Precision-top N, and Recall-top N curves for our DEMO and three reproduced deep unsupervised baselines using 128 bits on MirFlickr-25K, as illustrated in Figure 3. Due to space limitations, curves for other code lengths can be seen in Section D. The Precision-Recall curve represents the relationship between the varying precision and recall scores. The Precision-top N and Recall-top N curves depict precision and recall values as the retrieval numbers vary from 1 to 5, 000 with a step size of 100. In brief, for these three types of curves, the higher-performing method’s curve is usually above the curves of other methods. These curves clearly illustrate that our DEMO consistently outperforms the other baselines, underscoring its superiority. The hash lookup results are consistent with the Hamming ranking results, further validating the exceptional performance and robustness of our DEMO in image-text matching.

## 4.3 Ablation Study

In Table 2, we investigate the contributions of each proposed component with 16 bits on three datasets. Firstly, we remove the distribution-based structural mining process and replace it with samplebased structural mining. The comparison between DEMO w/o D in the first row and the full model in the last row highlights the significant improvement achieved by our distribution-based objective. Next, we assess the significance of the retrieval-based consistency learning by removing it. Without this module, the retrieved distributions given images and texts as queries are not encouraged to be con sistent. The performance degradation observed in DEMO w/o R in the second row emphasizes the effectiveness of this component. Moreover, we conduct an experiment where we remove only the proposed sharpening operation from the retrievalbased consistency learning module. The results of DEMO w/o S in the third row reveal slight differences compared to the full model, underscoring the importance of the sharpening operation. Furthermore, the performance of DEMO w/o S falls between DEMO w/o R and the full model, which is reasonable since DEMO w/o S removes only parts of the retrieval-based consistency learning module while still retaining the ability to promote consistency between the retrieved distributions. Finally, we evaluate the full model which incorporates all the components. Results in the last row exhibit the best performance across all scenarios. These experiments successfully verify the significance of each proposed component in DEMO.

<table><tr><td>Task</td><td>Method</td><td>MIRF-25K</td><td>NUS-WIDE</td><td>MS-COCO</td></tr><tr><td rowspan="3">I2T</td><td>DEMO w/o D DEMO w/o R</td><td>0.698</td><td>0.627</td><td>0.560</td></tr><tr><td></td><td>0.705</td><td>0.632</td><td>0.565</td></tr><tr><td>DEMO w/o S Full Model</td><td>0.712 0.718</td><td>0.636 0.646</td><td>0.571 0.575</td></tr><tr><td rowspan="3">T2I</td><td>DEMO w/o D</td><td>0.696</td><td>0.630</td><td>0.559</td></tr><tr><td>DEMO w/o R</td><td>0.695</td><td>0.634</td><td>0.564</td></tr><tr><td>DEMO w/o S</td><td>0.699</td><td>0.642</td><td>0.569</td></tr><tr><td rowspan="3"></td><td>Full Model</td><td>0.708</td><td></td><td></td></tr><tr><td></td><td></td><td>0.654</td><td>0.572</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 2: Ablation on each proposed component. The highest MAP scores are shown in boldface.  
![](images/09a43e74b6bff634cb135f82b605051a61ec4a15d20379536102235df53e3f09.jpg)

![](images/8c4a6b5a569ec42aa67df89c4e0ab403421ab28ed9ddf26efdd540cc6e5c8564.jpg)  
Figure 4: Sensitivity analysis of sampling times M and threshold τ with 16 bits on MIRFlickr-25K.

## 4.4 Sensitivity Analysis

To assess the impact of the hyper-parameter M and τ , Figure 4 plots the MAP scores with respect to

<table><tr><td>Method</td><td>LSSH</td><td>UGACH</td><td>UCCH</td><td>DEMO</td></tr><tr><td>Inference Time</td><td>7.78s</td><td>26.59s</td><td>0.41s</td><td>0.41s</td></tr></table>

Table 3: Comparison with other methods on the inference speed.

M ranging from 0 to 20, and τ ranging from 0.5 to 1.5. From the results, we can observe that increasing M from 0 to 5 yields a significant performance improvement, but further increasing M from 5 to 20 does not lead to any noticeable enhancement. This phenomenon demonstrates that our DEMO is not sensitive to M within the range of [5, 20]. Therefore, M is fixed at 5 and we proceed to investigate the threshold τ, varying from 0.5 to 1.5. The threshold τ plays a crucial role as it controls the percentage of the image-text pairs categorized as positive samples, thereby influencing the quality of the generated similarity matrix. A large value of τ will mistakenly consider numerous incorrect imagetext pairs as the matching ones, while a small value of τ will classify many matching image-text pairs as non-matching pairs. From the results, it can be found that 1.25 is the most suitable for the threshold τ . Consequently, we obtain the optimal value of $M = 5$ and $\tau = 1 . 2 5 .$ , respectively.

## 4.5 Efficiency Analysis

We make experimental verification on the inference speed. In particular, we compare our DEMO with state-of-the-art hashing-based image-text matching approaches with 128 bits on MIRFlickr-25K. As shown in Table 3, our DEMO can achieve much higher efficiency compared with LSSH (Zhou et al., 2014) and UGACH (Zhang et al., 2018). Even though the inference time of UCCH (Hu et al., 2023a) and our DEMO is the same, our retrieval performance is much better. In summary, our DEMO is superior to these baselines taking into both efficiency and effectiveness.

## 4.6 Visualization

We present the t-SNE (van der Maaten and Hinton, 2008) visualization of hash codes from two different modalities generated by four methods with 128 bits on MirFlickr-25K in Figure 5. The results of the comparison with the other three approaches reveal that our DEMO demonstrates a significantly higher degree of similarity and overlap between the image and text modalities. This observation serves as a strong indication that combined with collaborative consistency learning, distribution-based structural mining is superior to sample-based structural mining. The visualization results also provide compelling evidence of the exceptional quality and modality-invariant hash codes learned by our approach.

![](images/6ce239effa2e2d9b1bf67efd984c22a8029576cf19186f5f34b368e759081b28.jpg)  
Figure 5: The t-SNE visualization with 128 bits on the MIRFlickr-25K. The image modality is colored red, and the text modality is colored green. The overlap degree represents the degree of modality-invariant hash codes.

## 5 Conclusion

In this paper, we investigate the problem of imagetext matching and propose a novel deep unsupervised hashing-based approach termed DEMO. The crux of our DEMO is to explore the latent semantic distributions of each sample for effective semantics structure mining. Specifically, we characterize each image with multiple augmented views, which are regarded as samples from its intrinsic semantic distribution. Then, a non-parametric distribution divergence is employed to ensure a robust and accurate similarity structure in the process of similarity generation, which serves as guidance for the optimization. Extensive experimental results across multiple datasets substantiate the efficacy of DEMO.

## 6 Limitation

Although our DEMO achieves promising results, it still has some limitations. First, there could be different complicated scenarios in real-world applications such as data contamination and domain shift. We would extend our DEMO to more generalization scenarios in our future works. Second, our unsupervised hashing approach DEMO targets at coarse-level retrieval. How to improve unsupervised cross-modal hashing for fine-grained crossmodal retrieval remains an open problem.

## References

Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Armand Joulin, Nicolas Ballas, and Michael Rabbat. 2021. Semi-supervised learning of visual features by non-parametrically predicting

view assignments with support samples. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 8443–8452.

Abhidip Bhattacharyya, Cecilia Mauceri, Martha Palmer, and Christoffer Heckman. 2022. Aligning images and text with semantic role labels for finegrained cross-modal understanding. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, LREC 2022, Marseille, France, 20- 25 June 2022, pages 4944–4954. European Language Resources Association.

Min Cao, Shiping Li, Juntao Li, Liqiang Nie, and Min Zhang. 2022. Image-text retrieval: A survey on recent research and development. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022, pages 5410–5417. ijcai.org.

Hui Chen, Guiguang Ding, Xudong Liu, Zijia Lin, Ji Liu, and Jungong Han. 2020. Imram: Iterative matching with recurrent attention memory for crossmodal image-text retrieval. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12655–12663.

Zhen-Duo Chen, Chuan-Xiang Li, Xin Luo, Liqiang Nie, Wei Zhang, and Xin-Shun Xu. 2019. Scratch: A scalable discrete matrix factorization hashing framework for cross-modal retrieval. IEEE Transactions on Circuits and Systems for Video Technology, 30(7):2262–2275.

Haixing Dai, Zhengliang Liu, Wenxiong Liao, Xiaoke Huang, Zihao Wu, Lin Zhao, Wei Liu, Ninghao Liu, Sheng Li, Dajiang Zhu, Hongmin Cai, Quanzheng Li, Dinggang Shen, Tianming Liu, and Xiang Li. 2023. Chataug: Leveraging chatgpt for text data augmentation. CoRR, abs/2302.13007.

Guiguang Ding, Yuchen Guo, Jile Zhou, and Yue Gao. 2016. Large-scale cross-modality search via collective matrix factorization hashing. IEEE Transactions on Image Processing, 25(11):5427–5440.

Xinfeng Dong, Huaxiang Zhang, Lei Zhu, Liqiang Nie, and Li Liu. 2022. Hierarchical feature aggregation based on transformer for image-text matching. IEEE Transactions on Circuits and Systemsfor Video Technology, 32(9):6437–6447.

Wentao Fan, Chao Zhang, Huaxiong Li, Xiuyi Jia, and Guoyin Wang. 2023. Three-stage semisupervised cross-modal hashing with pairwise relations exploitation. IEEE Transactions on Neural Networks and Learning Systems, pages 1–14.

Zheren Fu, Zhendong Mao, Yan Song, and Yongdong Zhang. 2023. Learning semantic relationship among instances for image-text matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15159–15168.

Zijun Gao, Jun Wang, Guoxian Yu, Zhongmin Yan, Carlotta Domeniconi, and Jinglin Zhang. 2023. Longtail cross modal hashing. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 37, pages 7642–7650.

Wen Gu, Xiaoyan Gu, Jingzi Gu, Bo Li, Zhi Xiong, and Weiping Wang. 2019. Adversary guided asymmetric hashing for cross-modal retrieval. In Proceedings of the 2019 on international conference on multimedia retrieval, pages 159–167.

Wenchao Gu, Yanlin Wang, Lun Du, Hongyu Zhang, Shi Han, Dongmei Zhang, and Michael Lyu. 2022. Accelerating code search with deep hashing and code classification. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2534–2544.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. 2020. Momentum contrast for unsupervised visual representation learning. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738.

Peng Hu, Hongyuan Zhu, Jie Lin, Dezhong Peng, Yin-Ping Zhao, and Xi Peng. 2023a. Unsupervised contrastive cross-modal hashing. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3):3877–3889.

Xuming Hu, Zhijiang Guo, Zhiyang Teng, Irwin King, and Philip S. Yu. 2023b. Multimodal relation extraction with cross-modal retrieval and synthesis. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 2: Short Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 303–311. Association for Computational Linguistics.

Yan Huang, Yuming Wang, Yunan Zeng, and Liang Wang. 2022. MACK: multimodal aligned conceptual knowledge for unpaired image-text matching. In NeurIPS.

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. 2021. Scaling up visual and vision-language representation learning with noisy text supervision. In Proceedings ofthe 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 4904–4916. PMLR.

Qing-Yuan Jiang and Wu-Jun Li. 2017. Deep crossmodal hashing. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 3270– 3278. IEEE Computer Society.

Parminder Kaur, Husanbir Singh Pannu, and Avleen Kaur Malhi. 2021. Comparative analysis on cross-modal information retrieval: A review. Computer Science Review, 39:100336.

Vlad Krotov and Leigh Johnson. 2023. Big web data: Challenges related to data, technology, legality, and ethics. Business Horizons, 66(4):481–491.

Shaishav Kumar and Raghavendra Udupa. 2011. Learning hash functions for cross-view similarity search. In Twenty-second international joint conference on artificial intelligence.

Chunxiao Liu, Zhendong Mao, An-An Liu, Tianzhu Zhang, Bin Wang, and Yongdong Zhang. 2019a. Focus your attention: A bidirectional focal attention network for image-text matching. In Proceedings of the 27th ACM international conference on multimedia, pages 3–11.

Fangyu Liu and Rongtian Ye. 2019. A strong and robust baseline for text-image matching. In Annual Meeting ofthe Associationfor Computational Linguistics: Student Research Workshop, pages 169–176. Association for Computational Linguistics.

Hong Liu, Rongrong Ji, Yongjian Wu, Feiyue Huang, and Baochang Zhang. 2017. Cross-modality binary code learning via fusion similarity hashing. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 7380–7388.

Xiaoqing Liu, Huanqiang Zeng, Yifan Shi, Jianqing Zhu, Chih-Hsien Hsia, and Kai-Kuang Ma. 2023. Deep cross-modal hashing based on semantic consistent ranking. IEEE Transactions on Multimedia, pages 1–12.

Xin Liu, Zhikai Hu, Haibin Ling, and Yiu-ming Cheung. 2019b. Mtfh: A matrix tri-factorization hashing framework for efficient cross-modal retrieval. IEEE transactions on pattern analysis and machine intelligence, 43(3):964–981.

Haoyu Lu, Nanyi Fei, Yuqi Huo, Yizhao Gao, Zhiwu Lu, and Ji-Rong Wen. 2022. Cots: Collaborative twostream vision-language pre-training model for crossmodal retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15692–15701.

Xu Lu, Lei Zhu, Zhiyong Cheng, Jingjing Li, Xiushan Nie, and Huaxiang Zhang. 2019. Flexible online multi-modal hashing for large-scale multimedia retrieval. In Proceedings ofthe 27th ACM international conference on multimedia, pages 1129–1137.

Xiao Luo, Daqing Wu, Zeyu Ma, Chong Chen, Minghua Deng, Jianqiang Huang, and Xian-Sheng Hua. 2021a. A statistical approach to mining semantic similarity for deep unsupervised hashing. In Proceedings ofthe 29th ACM International Conference on Multimedia, page 4306–4314.

Xiao Luo, Daqing Wu, Zeyu Ma, Chong Chen, Minghua Deng, Jinwen Ma, Zhongming Jin, Jianqiang Huang, and Xian-Sheng Hua. 2021b. CIMON: towards highquality hash codes. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI 2021, Virtual Event / Montreal, Canada, 19-27 August 2021, pages 902–908. ijcai.org.

Zeyu Ma, Wei Ju, Xiao Luo, Chong Chen, Xian-Sheng Hua, and Guangming Lu. 2022. Improved deep unsupervised hashing via prototypical learning. In Proceedings ofthe 30th ACM International Conference on Multimedia, pages 659–667.

Georgii Mikriukov, Mahdyar Ravanbakhsh, and Begüm Demir. 2022. Unsupervised contrastive hashing for cross-modal retrieval in remote sensing. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4463–4467. IEEE.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners.

Maria L Rizzo and Gábor J Székely. 2016. Energy distance. Wiley Interdisciplinary Reviews: Computational Statistics, 8(1):27–38.

Jingkuan Song, Yang Yang, Yi Yang, Zi Huang, and Heng Tao Shen. 2013. Inter-media hashing for largescale retrieval from heterogeneous data sources. In Proceedings of the 2013 ACM SIGMOD international conference on management ofdata, pages 785–796.

Changchang Sun, Hugo Latapie, Gaowen Liu, and Yan Yan. 2022a. Deep normalized cross-modal hashing with bi-direction relation reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4941–4949.

Jinan Sun, Haixin Wang, Xiao Luo, Shikun Zhang, Wei Xiang, Chong Chen, and Xian-Sheng Hua. 2022b. Heart: Towards effective hash codes under label noise. In Proceedings ofthe 30th ACM International Conference on Multimedia, pages 366–375.

Rui Sun, Zhecan Wang, Haoxuan You, Noel Codella, Kai-Wei Chang, and Shih-Fu Chang. 2023. Unifine: A unified and fine-grained approach for zero-shot vision-language understanding. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 778–793.

Gábor J Székely and Maria L Rizzo. 2013. Energy statistics: A class of statistics based on distances. Journal ofStatistical Planning and Inference, 143(8):1249– 1272.

Rong-Cheng Tu, Lei Ji, Huaishao Luo, Botian Shi, He-Yan Huang, Nan Duan, and Xian-Ling Mao. 2021. Hashing based efficient inference for image-text matching. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 743–752.

Rong-Cheng Tu, Jie Jiang, Qinghong Lin, Chengfei Cai, Shangxuan Tian, Hongfa Wang, and Wei Liu. 2023a. Unsupervised cross-modal hashing with modalityinteraction. IEEE Transactions on Circuits and Systemsfor Video Technology.

Rong-Cheng Tu, Xian-Ling Mao, Qinghong Lin, Wenjin Ji, Weize Qin, Wei Wei, and Heyan Huang. 2023b. Unsupervised cross-modal hashing via semantic text mining. IEEE Transactions on Multimedia, pages 1–12.

Rong-Cheng Tu, Xianling Mao, and Wei Wei. 2020. Mls3rduh: Deep unsupervised hashing via manifold based local semantic similarity structure reconstructing. In IJCAI, pages 3466–3472.

Laurens van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-sne. Journal of Machine Learning Research, 9(86):2579–2605.

Haixin Wang, Huiyu Jiang, Jinan Sun, Shikun Zhang, Chong Chen, Xian-Sheng Hua, and Xiao Luo. 2023. Dior: Learning to hash with label noise via dual partition and contrastive learning. IEEE Transactions on Knowledge and Data Engineering.

Chen Wei, Huiyu Wang, Wei Shen, and Alan L. Yuille. 2021. CO2: consistent contrast for unsupervised visual representation learning. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Junyuan Xie, Ross B. Girshick, and Ali Farhadi. 2016. Unsupervised deep embedding for clustering analysis. In Proceedings ofthe 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016, volume 48 of JMLR Workshop and Conference Proceedings, pages 478– 487. JMLR.org.

Xing Xu, Fumin Shen, Yang Yang, Heng Tao Shen, and Xuelong Li. 2017. Learning discriminative binary codes for large-scale cross-modal retrieval. IEEE Transactions on Image Processing, 26(5):2494–2507.

Erkun Yang, Cheng Deng, Tongliang Liu, Wei Liu, and Dacheng Tao. 2018. Semantic structure-based unsupervised deep hashing. In Proceedings ofthe 27th international joint conference on artificial intelligence, pages 1064–1070.

Jun Yu, Hao Zhou, Yibing Zhan, and Dacheng Tao. 2021. Deep graph-neighbor coherence preserving network for unsupervised cross-modal hashing. In Proceedings ofthe AAAI conference on artificial intelligence, volume 35, pages 4626–4634.

Donghuo Zeng, Jianming Wu, Gen Hattori, Rong Xu, and Yi Yu. 2023. Learning explicit and implicit dual common subspaces for audio-visual cross-modal retrieval. ACM Transactions on Multimedia Computing, Communications and Applications, 19(2s):1–23.

Huatian Zhang, Zhendong Mao, Kun Zhang, and Yongdong Zhang. 2022a. Show your faith: Cross-modal confidence-aware network for image-text matching. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 36, pages 3262–3270.

Jian Zhang, Yuxin Peng, and Mingkuan Yuan. 2018. Unsupervised generative adversarial cross-modal hashing. In Proceedings ofthe AAAI conference on artificial intelligence, volume 32.

Kun Zhang, Zhendong Mao, Quan Wang, and Yongdong Zhang. 2022b. Negative-aware attention framework for image-text matching. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 15640–15649. IEEE.

Jile Zhou, Guiguang Ding, and Yuchen Guo. 2014. Latent semantic sparse hashing for cross-modal similarity search. In Proceedings ofthe 37th international ACM SIGIR conference on Research & development in information retrieval, pages 415–424.

## A Algorithm

<table><tr><td>Algorithm 1 Training Algorithm of DEMO</td></tr><tr><td>Require: Image dataset X; text dataset Y; num- ber of augmented views M, threshold τ.</td></tr><tr><td>Ensure: Parameters of the hashing network.</td></tr><tr><td>1: Generate M augmented views for every xi;</td></tr><tr><td>2: Calculate the distribution divergence using Eqn. 8;</td></tr><tr><td>3: Generate the instance similarity structure using</td></tr><tr><td>Eqn. 11; 4: repeat</td></tr><tr><td>5:Sample a mini-batch randomly;</td></tr><tr><td>6: Output approximate binary codes for both images and texts;</td></tr><tr><td>7: Generate cross-modal retrieval results using Eqn. 13 and Eqn. 14;</td></tr><tr><td>8: Calculate the whole loss using Eqn. 18;</td></tr><tr><td>9: Update the hashing network by backpropa- gation;</td></tr><tr><td>10: until convergence</td></tr></table>

## B Data Augmentation Strategy

We leverage the randomness of data augmentations to convert sample-based structural mining to distribution-based structural mining. The detailed data augmentation strategy is illustrated below. First, we resize the image to 256  256 and randomly crop a size of 224  224. Then we employ strategies such as Random Horizontal Flip, Random Color Jitter with p = 0.7, Random Grayscale with p = 0.2, and Gaussian Blur with kernelsize = 3. Finally, we normalize the data with pre-computed mean and standard values. With this augmentation strategy, the intrinsic semantic distribution of a data sample is established for future semantic structure mining.

## C Compared Methods

Many state-of-the-art cross-modal hashing-based methods are employed for comparison, including three supervised methods, four shallow unsupervised methods, and three deep unsupervised methods. The detailed introduction of these methods is as follows:

• CVH (Kumar and Udupa, 2011) introduces a novel relaxation technique that transforms the learning-to-hash process into a tractable eigenvalue problem. To address this challenge, they utilize techniques such as Locality Sensitive Indexing and Canonical Correlation Analysis.

• LSSH (Zhou et al., 2014) extracts the latent semantics from textual samples by matrix factorization. It also leverages sparse coding techniques to capture essential image structures. Introducing an effective iterative method, it analyzes the correlation between multimodal representations, thereby narrowing the semantic gap within the latent semantic space.

• CMFH (Ding et al., 2016) builds robust connections via cross-modal factorization, integrating locally linear embedding to uphold the Euclidean structure. Additionally, it employs a classifier-like loss function to leverage semantic label information effectively.

• FSH (Liu et al., 2017) defines the similarity between different modalities by introducing a graph-based framework, and then utilizing it to learn modality-invariant hash codes.

• MTFH (Liu et al., 2019b) proposes to learn semantic correlations between modalities and aligns heterogeneous data to obtain modalityspecific hash codes.

![](images/072662e0f9f9bc1391631b5a3f04a06ee1db51c385d5fa95b2398a4ab143ee05.jpg)

![](images/9657eaef4e60995f892fdc23f43bf65e2bf3e4a3c1bcb3a30afa14f79ca7fbbf.jpg)

![](images/5e2ebfbc5674b7eacbfb4f5eb9a854759c76a0a47913e654fd13e913cd6f9db6.jpg)

![](images/6a0fe600ec802ed20af51d0cfce1625504b75cf0314b23652a7843fdf82556a0.jpg)

![](images/4001d78b09e8e3d73993612ca18c166c8a6995a51ef66c431adb12a3c19fa73d.jpg)

![](images/d8b957d91442ed674f50a1f9cf8b94fa49bb974e49c1e484ba46b84b2ee71ba6.jpg)  
Figure 6: The Precision-Recall curve, Precision-top N curve, and Recall-top N curve with 16 bits on the MIRFlickr-25K dataset. Image-to-text results are plotted in the first row, and text-to-image results are plotted in the second row.

• FOMH (Lu et al., 2019) introduces a multimodal fusion framework to fuse representations when modalities are missing, and then constructs discriminative hash codes.

• DCH (Xu et al., 2017) optimizes the network to get modality-specific and modalityinvariant hash codes simultaneously. Moreover, it refines the hash codes by iterative training to enhance efficiency.

• DGCPN (Yu et al., 2021) investigates the correlations between data samples and their neighbors to improve the quality of similarity generation. It employs a hybrid optimization strategy, combining real and binary components, to minimize discrepancies between the Hamming space and the continuous latent space, thus enhancing similarity and value consistency.

• UCHSTM (Tu et al., 2023b) explores correlations among words in textual data points, facilitating the creation of a text modalityspecific similarity matrix derived from these correlations. Furthermore, it introduces a selfredefined similarity loss to rectify inaccuracies in the instance similarity matrix, thereby improving the accuracy of similarity measurements.

• UCCH (Hu et al., 2023a) introduces contrastive learning, aiming to align various modalities with unified binary representations. It emphasizes leveraging discrimination from all pairs rather than solely focusing on the hardest negative pairs.

## D Detailed Hash Lookup Protocol

We showcase the hash lookup results on MIRFlickr-25K with varying code lengths in Figure 6, Figure 7, and Figure 8. From the results of the hash lookup protocol, several conclusions can be observed:

1. Firstly, from the Precision-Recall curve, we can notice the correlation between precision and recall scores. These two metrics are contradictory to each other, as an increase in one often leads to a decrease in the other. From the results in the first column, it can be found that as the recall score increases, the precision score of our DEMO consistently surpasses the other three compared baseline hashing-based image-text matching methods.

2. Secondly, the Precision-top N curve represents the correlation between the precision score and the top N number of results returned in a single retrieval process. As the number of retrieved samples increases, the precision score tends to decrease. From the results in the second column, it can be observed that as

![](images/c5e681e57da32e4c6555c684e15b9c0691f88809f86883f79866abc51f507f36.jpg)

![](images/a3295929a6c3e855a7e287ebce2e5d71bd61a15f9fc8c28c1d6e5892b44697e0.jpg)

![](images/cc056bca04ca82f103a99da2b573e736c0de016d2d4f13d34c6fa1a1cb3f860b.jpg)

![](images/c05f73c6d0d520067af4cc4783aaea506554ed35dc78f6d763939b1cffaa5d79.jpg)

![](images/e92072af87519d6668e4ef6a0d9da17e317f09ec8472168e57122c96c909bcf2.jpg)

![](images/ee3377b8c2f399d211e9a592c40aa790980b1664c96ca0c4c4165c62543c2d27.jpg)  
Figure 7: The Precision-Recall curve, Precision-top N curve, and Recall-top N curve with 32 bits on the MIRFlickr-25K dataset. Image-to-text results are plotted in the first row, and text-to-image results are plotted in the second row.

![](images/f10dbc0aaf347005866a4209d007079c7fcad224b2e278a31dd900f05d5b5145.jpg)

![](images/43bbfd44508e5ccfc29ccf4956e7d62480ca92a190ce2c1ddc006134541b93a7.jpg)

![](images/400f62464dbbf66291835f595ed28a34a352a10e6710f0bd9219304416da9ebf.jpg)

![](images/36a946d3604e40888038af89c8fe28102de2078051ae562e12c2d99302e15d88.jpg)

![](images/3223f353ead6ddaf60d7214c86155617e517ae2c1293077ce76d13dfbc8534c3.jpg)

![](images/61b5415406806d68720723c3a824e66903296b16130ef65f8fdebe18e2dbba09.jpg)  
Figure 8: The Precision-Recall curve, Precision-top N curve, and Recall-top N curve with 64 bits on the MIRFlickr-25K dataset. Image-to-text results are plotted in the first row, and text-to-image results are plotted in the second row.

N increases from 1 to 5000, our DEMO consistently outperforms the other three methods.

3. Furthermore, similar to the Precision-top N curve, the Recall-top N curve represents the correlation between the recall score and the top N results returned in a single retrieval. Different from the precision score, the recall score tends to increase as N increases. From the results in the third column, it can be seen that as N increases from 1 to 5000, our curve consistently remains above the other three curves.

4. Lastly, a large number of results demonstrate the robustness of our method from different perspectives. Whether it pertains to various code lengths or different modalities, all the results indicate that we have successfully explored a more suitable similarity structure for unsupervised cross-modal hashing from a distribution perspective. By combining collaborative consistency learning, DEMO effectively improves the image-text matching quality.