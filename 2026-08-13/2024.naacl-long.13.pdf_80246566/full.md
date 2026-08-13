# Adaptive Rank Selections for Low-Rank Approximation of Language Models

Shangqian Gao, Ting Hua, Yen-Chang Hsu, Yilin Shen, Hongxia Jin Samsung Research America

{s.gao1,ting.hua,yenchang.hsu,yilin.shen,hongxia.jin}@samsung.com

## Abstract

Singular Value Decomposition (SVD) or its weighted variants has significantly progressed in compressing language models. Previous works assume the same importance for all operations and assign the same number of ranks for different layers in a language model. However, such a uniform rank selection is sub-optimal since different operations (layers) have nonuniform demand in capacity. In other words, a desired SVD strategy should allocate more ranks for important operations and vice versa. However, a globally-optimized selection of ranks for neural networks is still an open problem, and this is a non-trivial challenge since the selection is discrete. In this work, we propose a novel binary masking mechanism for optimizing the number of ranks in a differentiable framework. Our strategy uses a novel regularization to enable the masking to comply with the SVD property where the ranks have sorted singular values. The experiments examined both types of language models, encoder-only and decoder-only models, including large language models like LLaMA. Our compressed model achieves much better accuracy than previous SVD and their SOTA variants. More interestingly, our method retains significantly better accuracy with zero or limited fine-tuning, proving the substantial advantage of adaptive rank selection.

## 1 Introduction

Transformer-based models (Vaswani et al., 2017) have been very popular across different Natural Language Processing tasks, such as text classification (Wang et al., 2019a), question answering (Rajpurkar et al., 2016), and summarization (Liu, 2019). Despite its success on these tasks, the size of these models often scales up to millions or billions of parameters, especially for recently proposed large language models (Touvron et al., 2023; Biderman et al., 2023). Such a huge number of parameters makes these models very hard to be deployed on resource-limited devices, such as mobile phones or edge devices. As a result, the compression of Transformer-based language models has drawn much attention.

Transformers-based models have two core operations: self-attention layers and feed-forward layers. These operations are built on linear layers, making them straightforward to compression techniques like low-rank weight factorization (Golub and Reinsch, 1971; Noach and Goldberg, 2020) with SVD or its variants. Low-rank weight factorization decomposes a large linear layer into two small linear layers without changing other model parts, providing a friendly property for deployment. In addition, it is orthogonal to other compression techniques, such as structural pruning (Sanh et al., 2020), quantization (Shen et al., 2020), and knowledge distillation (Sun et al., 2019; Jiao et al., 2019).

Previous work (Hsu et al., 2021) shows that using vanilla SVD for compression can result in a significant performance drop. They argue that low reconstruction error is not equivalent to high accuracy. As a result, Hsu et al. (Hsu et al., 2021) proposed to apply the Fisher Information (Pascanu and Bengio, 2014) matrix to re-weight the weight matrix so that the factorization results can capture information from both the task and the reconstruction error. Empirically, Fisher Information weighted SVD performs much better than the original SVD. Despite using the Fisher Information matrix, other importance scores, like first-order Taylor expansion (Molchanov et al., 2019; Hua et al., 2022), can also be used to re-weight the weight matrix.

Although the mentioned weighted SVD methods above achieved promising results, they treat all layers uniformly and use the same number of ranks for all weight matrices. On the other hand, some prior works suggest that the compression rate for different layers should be different in the cases of vision (Molchanov et al., 2019) and language (Lagunas et al., 2021) models. These observations provide clues to improve the performance of existing weight factorization by selecting the proper number of ranks for each layer. Inspired by the above observations, the target of our problem setting is to find the optimal number of ranks for all the layers in a neural network. However, this optimization is not trivial since it is a discrete, nonsmooth, and non-convex problem. Reinforcement learning (Schulman et al., 2017) and evolutionary algorithms (Real et al., 2019) may find a solution for this problem, but they introduce substantial optimization costs that are not affordable for larger models.

To address the above challenge, we propose to use regularized differentiable binary masks to learn the number of ranks for each operation. The entire learning pipeline is built upon an end-to-end differentiable learning framework. We use the sum of a binary mask to capture the number of ranks for each layer. The proposed binary mask is properly regularized to be aligned with the sorted singular values of SVD. Moreover, we use a hypernetwork to improve the effectiveness of our method, which further accelerates the learning process. With all these designs, our method can efficiently find the number of ranks of different operations. The contribution of our work can be summarized as the following points:

• We proposed to use the sum of regularized binary masks to capture the number of ranks for different operations. To further improve efficiency, we introduce hypernetwork to generate the number of ranks.

• We proposed a novel regularization to make binary masks comply with the property of SVD where there are sorted singular values. The regularized binary mask can retain the important factors inherited from SVD or its weighted version.

• Extensive experiments show that our method can significantly improve the performance of SVD and its SOTA variants on both encoderonly and decoder-only language models.

## 2 Related Works

The benefit of Low-rank factorization is that it can be applied to any linear layer. An early work (Winata et al., 2019) applies SVD for the LSTM cell and explores the effectiveness on different NLP tasks (Zhang et al., 2021, 2022, 2023)

and model components. (Noach and Goldberg, 2020) propose a two-stage approach to compress a pre-trained language model. The first stage decomposes the weight matrix with SVD in the pre-trained language model. Then, they fine-tune weights with knowledge distillation to regain performance. The standard SVD can not capture all the information from tasks. The Fisher Information is introduced to reweight the weight matrix, and SVD is applied to the reweighted matrix (Hsu et al., 2021). On top of (Hsu et al., 2021), several numeric optimization methods are used to find the optimal solution to the weighted SVD problem (Hua et al., 2022) when the weighting matrix is not diagonal.

Besides model weights, SVD can also be applied to embedding layers. The ALBERT model (Lan et al., 2019) addresses the issue of redundant parameters in the embedding layer by employing factorization. This layer tends to have high input and output dimensions, leading to inefficiencies. In their work, Reid et al. (Reid et al., 2021) introduce a novel approach called Self-Attentive Factorized Embeddings (SAFE). This method enhances performance by incorporating a small self-attention layer built upon linear projection.

A crucial point omitted by previous works is that not all operations are created equally. Some operations require more capacity than others. Our method tackles this problem by automatically learning the number of ranks for each operation.

Our method is also related to network pruning methods, especially structural pruning. Block Pruning (Lagunas et al., 2021) integrates structures of any size into the movement pruning paradigm for fine-tuning, and it prunes the model globally. In addition to NLP tasks, deciding the width of a convolution layer has also been studied extensively using reinforcement learning (He et al., 2018), evolutionary algorithm (Liu et al., 2019), etc. Differentiable pruning (Guo et al., 2020; Herrmann et al., 2020; Wang et al., 2019b; Gao et al., 2022, 2023a,b) is also a popular direction since the cost is often not high. However, they can not be directly applied to select the number of ranks due to the cost or difficulty of fine-tuning resulting from using binary masks.

## 3 Method

## 3.1 Background

Transformers have many linear layers, which makes them very suitable for compression methods

like Singular Value Decomposition (SVD). Suppose we have a matrix $\mathbf { W } \in \Re ^ { M \times N }$ , SVD decomposes it into three matrices:

$$
\mathbf { W } = \mathbf { U } \mathbf { S } \mathbf { V } \approx \mathbf { U } _ { r } \mathbf { S } _ { r } \mathbf { V } _ { r } ,\tag{1}
$$

where the orthogonal matrix U $\mathbf { \Sigma } \in \mathrm { ~ } \Re ^ { M \times M }$ is the left singular vectors, and the orthogonal matrix $\textbf { V } \in ~ \mathbf { \mathfrak { R } } ^ { N \times N }$ is the right singular vectors. S is a diagonal matrix of non-zero singular values $\operatorname { D i a g } ( s ) = \operatorname { D i a g } ( \sigma _ { 1 } , \sigma _ { 2 } , \cdot \cdot \cdot , \sigma _ { N } )$ (assuming $M \geq N )$ , where $\sigma _ { 1 } \geq \sigma _ { 2 } \geq \cdots \sigma _ { N } . \mathrm { ~ } \mathbf { U } _ { r } , \mathbf { S } _ { r } , \mathbf { V } _ { r }$ represent the truncated matrices with rank r and approximate the original matrix.

With the SVD, the computation of a linear layer in a neural network can be rewritten as below with input data $X ~ \in ~ \mathfrak { R } ^ { B \times M }$ , weight matrix $\mathbf { W } \in \hat { \mathfrak { R } } ^ { \hat { M } \times N }$ , bias $\mathbf { \Phi } ) \in \Re ^ { 1 \times N }$

$$
\begin{array} { r } { \mathbf { Y } = \mathbf { X } \mathbf { W } + \mathbf { b } = \mathbf { X } ( \mathbf { U } \mathbf { S } ) \mathbf { V } ^ { T } + \mathbf { b } . } \end{array}\tag{2}
$$

The standard SVD can be further improved by multiplying a weighting matrix with W, and this weighting matrix can be computed in many different ways, such as using Fisher Information (Pascanu and Bengio), Importance Estimation (Molchanov et al., 2019), etc. Weighted SVD often performs better than vanilla SVD when compressing language models. Denote the weighting matrix as $\mathbf { I } _ { w }$ , and $\mathbf { I } _ { w }$ is a diagonal matrix where the importance of each weight is summed within each column or row. Then, after applying $\mathbf { I } _ { w }$ , we have:

$$
\mathbf { Y } = \mathbf { X } \mathbf { W } + \mathbf { b } = \mathbf { X } [ { \mathbf { I } _ { w } } ^ { - 1 } ( \mathbf { U } ^ { \prime } \mathbf { S } ^ { \prime } ) \mathbf { V } ^ { \prime T } ] + \mathbf { b } .\tag{3}
$$

where $U ^ { \prime } , S ^ { \prime }$ , and $V ^ { \prime }$ come from the weighted SVD decomposition of $\mathbf { I } _ { w } \mathbf { W } = \mathbf { U } ^ { \prime } \mathbf { S } ^ { \prime } \mathbf { V } ^ { \prime }$ . Note that by using SVD or its weighted variants, we can easily compress pre-trained models, which is vital since the training costs of the typical large language models are very high, and training them from scratch is usually prohibitively expensive.

## 3.2 Overview

In the following contents, we will first introduce how we parameterize the number of ranks. Then we will introduce the hypernetwork used to generate the number of ranks. After that, we will talk about how we overcome the difficulty of finetuning caused by directly using indices from binary masks and how to produce top-k-like masks. The overall optimization problem will be introduced last. Fig. 1 illustrates our method given one selfattention layer.

![](images/95d83e4933239ef957bf7b1e441638726bbcd0868f23a6ee91144894dce85d43.jpg)  
Figure 1: An overview of our method. In the figure, we use the self-attention layer as an example. The hypernetwork produces the number of ranks for each operation, which are then applied to the query, key, and value weights. Since m is differentiable w.r.t to the hypernetwork, we can optimize the number of ranks in an end-to-end differentiable way.

## 3.3 Control the Number of Ranks

In Equation. 2, the diagonal matrix S contains singular values of SVD. If singular values are equal to zeros, then the corresponding vectors from U and V can be safely removed. Usually, the singular values of model weights are non-zero. As a result, we can apply a binary mask $m \in \{ 0 , 1 \}$ on top of the diagonal matrix S:

$$
{ \hat { s } } = m \odot s ,\tag{4}
$$

where s is the singular vector, and $\mathbf { S } = \mathrm { D i a g } ( s )$ After applying m, it changes Eq. 2 into:

$$
\mathbf { Y } = \mathbf { X } ( \mathbf { U } \mathbf { D i a g } ( \hat { s } ) ) \mathbf { V } ^ { T } + \mathbf { b } ,\tag{5}
$$

which inserts the mask m into the forward/backward calculation of a linear layer under SVD decomposition. By doing so, we can calculate the gradients w.r.t m during regular backpropagation. As a result, the mask can be learned in a loss-aware fashion if it is parameterized properly. Note that, unlike the uniform rank selection in previous works (Noach and Goldberg, 2020; Hsu et al., 2021; Hua et al., 2022), our method enables adaptive rank selections for individual operations for the model, which creates flexibility to allocate different ranks for different operations, and we can allocate more parameters for more important operations. Thus, the overall performance can be largely improved over the uniform rank selection setting.

## 3.4 Hypernetwork

The binary mask m is not differentiable in its plain form; therefore, we incorporate the straightthrough Gumbel-Sigmoid (Jang et al., 2016) operation to make it differentiable. In addition, instead of using element-wise mask parameterization, we employ a hypernetwork (HN) to accelerate the learning of masks m. Specifically, m is generated by:

$$
m = \mathbf { H } \mathbf { N } ( z ; \theta ) ,\tag{6}
$$

where θ is the parameters of the hypernetwork, and z (randomly sampled before training the hypernetwork) is the input to the hypernetwork. Basically, the HN is composed of GRUs (Chung et al., 2014) and linear layers. The intuition is that the GRU can be used to learn interactions between different operations, and linear layers are used to map GRU outputs to individual operations of different sizes. More details of the hypernetwork will be presented in the Appendix.

## 3.5 Singular Value-aware Masking

The hypernetwork gives the number of ranks and the exact positions of selected ranks for each layer. On the other hand, SVD, or its weighted version, provides sorted singular values in the diagonal matrix S (from Eq. 2). So far, the hypernetwork computes the mask completely independent from the structure of S, which has sorted singular values. This independency can produce a mask that skips some ranks with a high singular value, resulting in a less generalizable selection of ranks. This behavior significantly deteriorated the compressed model, impeding the following fine-tuning process from recovering the accuracy. In the later section, Fig. 3 shows this phenomenon with the exact positions of selected ranks from the hypernetwork (the plot named ‘Element-Wise’).

To address the issue, we choose to use the sum of the binary mask $\mathbf { 1 } ^ { T } m _ { l } \left( m _ { l } \right.$ is the mask for lth layer) to represent the number of ranks for the current operation and use this sum to force selecting the top-k ranks. Although this strategy resolves the above issue, it introduces a gap between the learned and actual masks for compressing the model. The gap can be formulated by:

$$
\| m _ { l } \odot s - m _ { l } ^ { \prime } \odot s \| _ { 2 } ^ { 2 } ,\tag{7}
$$

where $m _ { l } ^ { \prime }$ is a binary mask with the first $\mathbf { 1 } ^ { T } m _ { l }$ elements equals to ${ \mathrm { ~  ~ \Omega ~ } } ( m _ { l [ : 1 ^ { T } m _ { l } ] } ^ { \prime } \ = \ 1 )$ , and the rest elements of $m _ { l } ^ { \prime }$ equals 0. The smaller the gap, the closer the binary mask $m _ { l }$ to follow the structure of sorted singular values from SVD. The above insight inspired our novel regularization term: $\mathcal { R } _ { a l i g n } ( m _ { l } ) = \| m _ { l } \odot s - m _ { l } ^ { \prime } \odot s \| _ { 2 } ^ { 2 }$ . This regularization can be seamlessly inserted into the optimization of the HN without introducing extra parameters. Our ablation study will verify the mentioned insight and prove the effectiveness of $\mathcal { R } _ { a l i g n }$

```latex
Algorithm 1: Adaptive Rank Selection
Input: a sub-dataset for training the HN:
$D _ { \mathrm { H N } } ;$ remained rate of parameters: $p ;$
hyper-parameter: $\lambda , \gamma ;$ HN training
iterations: $N _ { \mathrm { i t e r } } ;$ a pre-trained model: $f ;$
the hypernetwork HN parameterized by θ
for $i : = 1$ to $N _ { i t e r }$ do
for a mini-batch $( x , y )$ in $D _ { H N }$ do
1. generate m from HN with Eq. 6.
2. calculate the parameter
regularization term
$\mathscr { R } ( p T ( m ) , T _ { \mathrm { t o t a l } } ) .$
3. calculate the alignment
regularization term $\mathcal { R } _ { a l i g n } .$
4. calculate gradients w.r.t θ by
minimizing Obj. 8 and update θ.
end
end
Compress the model based on the number
of ranks:
$\begin{array} { r } { \mathbf U \mathbf S = ( \mathbf U \mathbf S ) _ { [ : , : \mathbf 1 ^ { T } m ] } , \mathbf V = \mathbf V _ { [ : , : \mathbf 1 ^ { T } m ] } . } \end{array}$
Return the resulting model for fine-tuning.
```

## 3.6 The Proposed Algorithm

For a specific task, to maximally preserve the performance given a parameter budget, we minimize the task loss together with the regularization of the number of parameters and the regularization for aligning the SVD property. The overall objective function is defined by:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { \theta } \mathcal { L } ( f ( x ; m ) , y ) + \lambda \mathcal { R } ( T ( m ) , p T _ { t o t a l } ) } \\ { \displaystyle \quad \quad + \gamma \frac { 1 } { L } \sum _ { l = 1 } ^ { N } \mathcal { R } _ { a l i g n } ( m _ { l } ) , } \end{array}\tag{8}
$$

where x, y are input and its label, $\mathcal { L }$ is the taskspecific loss, $f ( \cdot ; m )$ is the model parameterized by the mask $m , \ \lambda$ controls the regularization weights for the parameter regularization  and $\mathcal { R } ( a , b ) = \log ( \operatorname* { m a x } ( a , b ) / b )$ , γ controls the regularization weights of $\mathcal { R } _ { a l i g n } , T _ { t o t a l }$ is the total number of the parameters, and $p$ is the persevered ratio of parameters which is given by users. $T ( m )$ is the number of parameters decided by the number of ranks for each operation. Take lth weight matrix as an example; the number of parameters for it is decided by: $T ( m _ { l } ) = \left( M _ { l } + N _ { l } \right) \times \left( \mathbf { 1 } ^ { T } m _ { l } \right)$ $\begin{array} { r } { T ( m ) \ = \ \sum _ { l = 1 } ^ { L } T ( m _ { l } ) } \end{array}$ , where $M _ { l }$ and $N _ { l }$ is the number of inputs and outputs dimensions for lth weight matrix. Note that the model weights are frozen during the optimization of Obj. 8; therefore, the learnable parameter is small and can be optimized efficiently.

<table><tr><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=2>MRPC</td><td rowspan=1 colspan=1>STSB</td><td rowspan=1 colspan=1>COLA</td><td rowspan=1 colspan=1>SST-2</td><td rowspan=1 colspan=1>MNLI</td><td rowspan=1 colspan=1>QNLI</td><td rowspan=1 colspan=1>QQP</td><td rowspan=1 colspan=1>Avg</td><td rowspan=1 colspan=1> $\overline { { \Delta \mathbf { - } \mathbf { A v g } } }$ </td><td rowspan=1 colspan=1># Params</td></tr><tr><td rowspan=1 colspan=1>BERT-base</td><td rowspan=1 colspan=2>87.29</td><td rowspan=1 colspan=1>88.47</td><td rowspan=1 colspan=1>57.78</td><td rowspan=1 colspan=1>92.90</td><td rowspan=1 colspan=1>84.95</td><td rowspan=1 colspan=1>91.25</td><td rowspan=1 colspan=1>87.92</td><td rowspan=1 colspan=1>84.36</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>109.5M</td></tr><tr><td rowspan=2 colspan=1>SVD+ fine-tuning</td><td rowspan=1 colspan=2>55.88</td><td rowspan=1 colspan=1>23.99</td><td rowspan=1 colspan=1>2.15</td><td rowspan=1 colspan=1>78.10</td><td rowspan=1 colspan=1>35.73</td><td rowspan=1 colspan=1>37.78</td><td rowspan=1 colspan=1>59.70</td><td rowspan=1 colspan=1>41.90</td><td rowspan=1 colspan=1>-42.36</td><td rowspan=1 colspan=1>66.5M</td></tr><tr><td rowspan=1 colspan=2>83.60</td><td rowspan=1 colspan=1>85.67</td><td rowspan=1 colspan=1>29.02</td><td rowspan=1 colspan=1>91.28</td><td rowspan=1 colspan=1>83.02</td><td rowspan=1 colspan=1>89.35</td><td rowspan=1 colspan=1>87.05</td><td rowspan=1 colspan=1>78.42</td><td rowspan=1 colspan=1>-5.94</td><td rowspan=1 colspan=1>66.5M</td></tr><tr><td rowspan=1 colspan=1>SVD+ARS (ours)+ fine-tuning (ours)</td><td rowspan=1 colspan=2>81.2285.57</td><td rowspan=1 colspan=1>73.7886.30</td><td rowspan=1 colspan=1>0.0047.08</td><td rowspan=1 colspan=1>81.0891.97</td><td rowspan=1 colspan=1>62.7583.55</td><td rowspan=1 colspan=1>57.8689.44</td><td rowspan=1 colspan=1>66.7187.39</td><td rowspan=1 colspan=1>60.4881.61</td><td rowspan=1 colspan=1>-23.88-2.75</td><td rowspan=1 colspan=1>65.1M65.1M</td></tr><tr><td rowspan=3 colspan=1>IWSVD+ fine-tuningIWSVD+ARS (ours)</td><td rowspan=1 colspan=2>5.52</td><td rowspan=1 colspan=1>58.97</td><td rowspan=1 colspan=1>13.14</td><td rowspan=1 colspan=1>81.31</td><td rowspan=1 colspan=1>46.96</td><td rowspan=1 colspan=1>52.50</td><td rowspan=1 colspan=1>63.30</td><td rowspan=1 colspan=1>45.96</td><td rowspan=1 colspan=1>-38.40</td><td rowspan=1 colspan=1>66.5M</td></tr><tr><td rowspan=1 colspan=2>86.87</td><td rowspan=1 colspan=1>87.45</td><td rowspan=1 colspan=1>43.83</td><td rowspan=1 colspan=1>89.91</td><td rowspan=1 colspan=1>82.56</td><td rowspan=1 colspan=1>89.35</td><td rowspan=1 colspan=1>86.55</td><td rowspan=1 colspan=1>80.93</td><td rowspan=1 colspan=1>-3.43</td><td rowspan=1 colspan=1>66.5M</td></tr><tr><td rowspan=1 colspan=2>81.58</td><td rowspan=1 colspan=1>76.93</td><td rowspan=1 colspan=1>23.97</td><td rowspan=1 colspan=1>83.94</td><td rowspan=1 colspan=1>51.88</td><td rowspan=1 colspan=1>77.58</td><td rowspan=1 colspan=1>75.05</td><td rowspan=1 colspan=1>67.28</td><td rowspan=1 colspan=1>-17.08</td><td rowspan=1 colspan=1>65.1M</td></tr><tr><td rowspan=1 colspan=1>+ fine-tuning (ours)</td><td rowspan=1 colspan=2>88.13</td><td rowspan=1 colspan=1>88.23</td><td rowspan=1 colspan=1>52.88</td><td rowspan=1 colspan=1>91.40</td><td rowspan=1 colspan=1>83.86</td><td rowspan=1 colspan=1>89.91</td><td rowspan=1 colspan=1>87.59</td><td rowspan=1 colspan=1>83.14</td><td rowspan=1 colspan=1>-1.22</td><td rowspan=1 colspan=1>65.1M</td></tr><tr><td rowspan=1 colspan=1>FWSVD</td><td rowspan=1 colspan=2>68.00</td><td rowspan=1 colspan=1>68.77</td><td rowspan=1 colspan=1>15.69</td><td rowspan=1 colspan=1>79.93</td><td rowspan=1 colspan=1>48.10</td><td rowspan=1 colspan=1>52.65</td><td rowspan=1 colspan=1>66.07</td><td rowspan=1 colspan=1>57.03</td><td rowspan=1 colspan=1>-27.33</td><td rowspan=1 colspan=1>66.5M</td></tr><tr><td rowspan=3 colspan=1>+ fine-tuningFWSVD+ARS (ours)+ fine-tuning (ours)</td><td rowspan=1 colspan=2>88.36</td><td rowspan=1 colspan=1>86.90</td><td rowspan=1 colspan=1>45.80</td><td rowspan=1 colspan=1>89.60</td><td rowspan=1 colspan=1>82.54</td><td rowspan=1 colspan=1>89.18</td><td rowspan=1 colspan=1>86.97</td><td rowspan=1 colspan=1>81.34</td><td rowspan=1 colspan=1>-3.02</td><td rowspan=1 colspan=1>66.5M</td></tr><tr><td rowspan=2 colspan=2>89.40</td><td rowspan=1 colspan=1>81.22</td><td rowspan=2 colspan=1>84.2488.47</td><td rowspan=1 colspan=1>27.22</td><td rowspan=1 colspan=1>83.37</td><td rowspan=1 colspan=1>71.12</td><td rowspan=1 colspan=1>64.10</td><td rowspan=1 colspan=1>75.18</td><td rowspan=1 colspan=1>69.49</td><td rowspan=1 colspan=1>-14.87</td><td rowspan=1 colspan=1>65.1M</td></tr><tr><td rowspan=1 colspan=1>55.01</td><td rowspan=1 colspan=1>91.06</td><td rowspan=1 colspan=1>83.68</td><td rowspan=1 colspan=1>89.68</td><td rowspan=1 colspan=1>87.41</td><td rowspan=1 colspan=1>83.53</td><td rowspan=1 colspan=1>-0.83</td><td rowspan=1 colspan=1>65.1M</td></tr></table>

Table 1: Results of GLUE benchmark when $p = 0 . 4 8 .$ $\mathbf { \nabla } \cdot \mathbf { A v } \mathbf { g } ^ { \prime }$ means the average score of the GLUE tasks. The $\cdot \Delta \mathbf { - A v } \mathbf { g } ^ { \prime }$ is the difference of $\mathbf { \dot { A } v g } ^ { \prime }$ between the full model and different baselines. Given a similar number of parameters, a smaller $\cdot \Delta \mathbf { - A v } \mathbf { g } ^ { \prime }$ represents better performance.

We present the algorithm for learning the number of ranks in Alg. 1. It requires only a small subset of the original training data; therefore, the computation overhead to optimize the number of ranks is negligible (details are in the experiment section). Note that all $m _ { l }$ are learned jointly in one pass, and rank selection competes across all layers. In other words, important operations can receive more ranks than the less important ones. Finally, we select top $\mathbf { 1 } ^ { T } m _ { l }$ ranks for each operation to compress a model, as described in Section 3.5.

## 4 Experiments

## 4.1 Settings

We assess our proposed method and baselines using the General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2019a) and the large language model pre-training task on Pile (Gao et al., 2020). For GLUE tasks, we use BERT (Devlin et al., 2018), MobileBERT (Sun et al., 2020), and DistllBERT (Sanh et al., 2019) to evaluate our method. We use LLaMA-7B (Touvron et al., 2023) to evaluate our method for large language models. In the Appendix, we use models from Pythia Suite (Biderman et al., 2023) to evaluate our method for the language modeling task. Throughout the experiment section, our method is abbreviated as ARS (Adaptive Rankd Selection).

To build fair comparison baselines, we compress all linear layers from the model, including selfattention layers and feed-forward networks. In addition, we do not compress the embedding layer, and the compression rate of our method can be further improved by incorporating previous works focusing on compressing the embedding layer (Lan et al., 2019; Reid et al., 2021).

Our method aims to find the best number of ranks for each operation. As a result, we will show our method is effective across different choices of weighting matrices (Eq. 3) or no weighing matrix (Eq. 2). For weighted SVD, we choose two kinds of weighting matrix: Fisher information Weighted SVD (FWSVD) (Hsu et al., 2021) and Importance Weighted SVD (IWSVD). For IWSVD, the importance is calculated by directly following the definition from (Molchanov et al., 2019), which is based on the first-order Taylor expansion.

For all tasks, we use pre-trained language models as a start, then the model is fine-tuned on downstream tasks, like GLUE or language modeling tasks. After that, we freeze the model weights, and we train the HN based on Obj. 8. The model is then compressed based on the number of ranks produced by the HN. Finally, the model is fine-tuned again on downstream tasks or pre-training tasks.

When training the HN, we choose 4000 samples for GLUE tasks (Wang et al., 2019a). If the dataset is smaller than 4000 samples, we use the whole dataset to train the HN. For the language modeling task, we train the HN for 2000 iterations. ADAM (Kingma and Ba, 2015) is used to train the HN with a constant learning rate $1 \times 1 0 ^ { - 3 } .$ . λ and γ in Obj. 8 is set to 16 and 10 for all experiments. For all GLUE tasks, the other settings are the default configuration from the HuggingFace

<table><tr><td>Task</td><td>MRPC</td><td>STSB</td><td>COLA</td><td>SST-2</td><td>MNLI</td><td>QNLI</td><td>QQP</td><td>Avg</td><td>∆-Avg</td><td># Params</td></tr><tr><td>BERT-base</td><td>87.29</td><td>88.47</td><td>57.78</td><td>92.90</td><td>84.95</td><td>91.25</td><td>87.92</td><td>84.36</td><td>=</td><td>109.5M</td></tr><tr><td>SVD</td><td>0.00</td><td>17.68</td><td>2.05</td><td>63.88</td><td>36.60</td><td>49.46</td><td>46.56</td><td>30.89</td><td>-53.47</td><td>52.4M</td></tr><tr><td>+ fine-tuning</td><td>81.06</td><td>79.35</td><td>9.83</td><td>89.11</td><td>81.61</td><td>86.99</td><td>86.35</td><td>73.47</td><td>-10.89</td><td>52.4M</td></tr><tr><td>SVD+ARS (ours)</td><td>81.22</td><td>64.23</td><td>0.00</td><td>79.47</td><td>35.73</td><td>52.41</td><td>51.50</td><td>52.08</td><td>-31.38</td><td>52.6M</td></tr><tr><td>+ fine-tuning (ours)</td><td>81.42</td><td>82.85</td><td>27.62</td><td>89.22</td><td>83.07</td><td>87.50</td><td>86.68</td><td>76.91</td><td>-7.45</td><td>52.6M</td></tr><tr><td>IWSVD</td><td>1.42</td><td>23.54</td><td>0.00</td><td>72.48</td><td>41.59</td><td>49.51</td><td>57.54</td><td>35.15</td><td>-49.21</td><td>52.4M</td></tr><tr><td>+ fine-tuning</td><td>80.79</td><td>82.29</td><td>24.49</td><td>88.76</td><td>81.63</td><td>87.46</td><td>86.35</td><td>75.97</td><td>-8.39</td><td>52.4M</td></tr><tr><td>IWSVD+ARS (ours)</td><td>81.22</td><td>68.94</td><td>0.00</td><td>82.57</td><td>60.70</td><td>67.75</td><td>64.30</td><td>60.78</td><td>-23.58</td><td>52.6M</td></tr><tr><td>+ fine-tuning (ours)</td><td>84.87</td><td>86.09</td><td>45.25</td><td>90.02</td><td>82.97</td><td>88.78</td><td>87.13</td><td>80.73</td><td>-3.63</td><td>52.6M</td></tr><tr><td>FWSVD</td><td>0.00</td><td>36.95</td><td>15.69</td><td>72.02</td><td>40.62</td><td>49.46</td><td>52.81</td><td>36.59</td><td>-47.77</td><td>52.4M</td></tr><tr><td>+ fine-tuning</td><td>81.96</td><td>83.41</td><td>45.80</td><td>88.42</td><td>80.67</td><td>87.66</td><td>86.76</td><td>78.34</td><td>-6.02</td><td>52.4M</td></tr><tr><td>FWSVD+ARS (ours)</td><td>81.22</td><td>67.25</td><td>23.55</td><td>81.42</td><td>58.94</td><td>70.49</td><td>63.56</td><td>63.77</td><td>-20.59</td><td>52.6M</td></tr><tr><td>+ fine-tuning (ours)</td><td>85.48</td><td>86.19</td><td>48.79</td><td>90.94</td><td>82.84</td><td>88.45</td><td>87.04</td><td>81.39</td><td>-2.97</td><td>52.6M</td></tr></table>

Table 2: Results of GLUE benchmark when $p = 0 . 3 3 .$ The definition of ‘Avg’ and ’∆-Avg’ is same as Tab. 1.

![](images/5d18844c7beec5ee7254bf89311a35a02bce96e8570c1be2afb3b4895d0a0e65.jpg)  
(a) MRPC

![](images/eaf899939f9aaf6a88eac5204cf08eb53d6d8f62a5bb335b285d20fc5a7871a5.jpg)  
(b) STSB

![](images/3f090700acc5558a063e321cc5420d15a642660635c1c0d94149e3c9c995fb34.jpg)  
(c) COLA  
Figure 2: The number of parameters vs. the performance after fine-tuning for FWSVD and FWSVD+ARS.

Transformer library. We defer other training details of the language modeling task to the Appendix. All of our implementations are based on the Huggingface Transformer library (Wolf et al., 2020) and PyTorch (Paszke et al., 2019).

## 4.2 GLUE Results for BERT

The GLUE results are shown in Tab. 1. As introduced previously, our method ARS is applied to three baselines: FWSVD, IWSVD, and SVD. For all methods, the uniform baseline from previous works has 66.5M parameters, and it is achieved by removing 67% ranks from the original model. For ARS, the model has 65.1M parameters, which is achieved by setting p in Obj. 8 to $p = 0 . 4 8$

We present results before and after fine-tuning in the table. It is clear that ARS can boost the performance of the uniform SVD, IWSVD, and FWSVD. In particular, before fine-tuning, SVD+ARS performs better than SVD by 18.48 regarding average task performance (‘Avg’ in the table). After fine-tuning, this gap is 3.19 between SVD and SVD+ARS. By using Fisher Information or other importance scores, the compressed model has a much better performance across different tasks since task related information is injected. With these stronger baselines, our method continuously improves their performance. For IWSVD, our method is 21.32/2.21 (with/without fine-tuning), better than the baseline on average task performance. For FWSVD, our method again is better than the baseline by 12.46 and 2.19 before and after fine-tuning. In summary, ARS can still provide substantial improvements even with stronger baselines.

Besides the comparison under the same weighting mechanism, SVD+ARS has a similar or even better performance than weighted SVD like IWSVD and FWSVD. In particular, by finding the proper number of ranks given each operation, SVD+ARS has 60.48/81.61 average task performance. At the same time, IWSVD has 45.96/80.93 average task performance, and the number for FWSVD is 57.03/81.34. SVD+ARS is better than IWSVD, and it has a similar performance as FWSVD. From this perspective, we can say that properly choosing the number of ranks is as important as building a good importance metric for weighted SVD.

We further increase the compression rate, and results are shown in Tab. 2. In this setting, we remove 78% of ranks for the baseline model, and we set $ { p ^ { \mathrm { ~ ~ = ~ } ~ 0 . 3 3 } }$ for the proposed ARS. ARS improves the performance of SVD, IWSVD, and FWSVD across different GLUE tasks. More specifically, SVD+ARS is better than SVD by 20.09/3.44 before and after finetuning. IWSVD+ARS is 25.63/4.76 better than IWSVD, and FWSVD+ARS is 27.18/3.05 better than FWSVD. In general, with a more aggressive compression rate, the advantage of ARS is more obvious. In Fig. 2, we visualize the number of parameters vs. the performance for MRPC, STSB, and COLA between FWSVD and FWSVD+ARS. FWSVD+ARS outperforms FWSVD across nearly all settings, which again demonstrates that selecting the proper number of ranks is important across different compression rates.

<table><tr><td>Task</td><td>MRPC</td><td>STSB</td><td>COLA</td><td>SST-2</td><td>MNLI</td><td>QNLI</td><td>QQP</td><td>Avg</td><td> $\overline { { \Delta \mathbf { - } \mathbf { A v g } } }$ </td><td># Params</td></tr><tr><td>DistllBERT</td><td>88.73</td><td>86.13</td><td>49.75</td><td>90.37</td><td>82.07</td><td>89.2</td><td>86.74</td><td>81.86</td><td></td><td>66.9M</td></tr><tr><td>FWSVD</td><td>44.50</td><td>36.23</td><td>15.06</td><td>81.65</td><td>41.58</td><td>72.12</td><td>71.03</td><td>51.74</td><td>-30.12</td><td>45.5M</td></tr><tr><td>+ fine-tuning</td><td>88.12</td><td>84.37</td><td>32.44</td><td>88.07</td><td>79.71</td><td>87.35</td><td>85.65</td><td>77.96</td><td>-3.90</td><td>45.5M</td></tr><tr><td>FWSVD+ARS (ours)</td><td>81.22</td><td>79.10</td><td>21.85</td><td>86.01</td><td>68.64</td><td>79.77</td><td>77.10</td><td>70.53</td><td>-11.33</td><td>44.9M</td></tr><tr><td>+ fine-tuning (ours)</td><td>88.04</td><td>86.43</td><td>43.84</td><td>90.02</td><td>81.49</td><td>87.94</td><td>86.62</td><td>80.63</td><td>-1.23</td><td>44.9M</td></tr><tr><td>MobileBERT</td><td>89.69</td><td>87.24</td><td>51.16</td><td>90.94</td><td>83.41</td><td>90.54</td><td>86.70</td><td>82.81</td><td></td><td>24.6M</td></tr><tr><td>FWSVD</td><td>50.99</td><td>57.16</td><td>2.59</td><td>54.59</td><td>46.10</td><td>49.46</td><td>63.58</td><td>46.35</td><td>-36.46</td><td>19.5M</td></tr><tr><td>+ fine-tuning</td><td>87.50</td><td>86.37</td><td>34.42</td><td>88.07</td><td>81.16</td><td>86.67</td><td>86.23</td><td>78.63</td><td>-4.18</td><td>19.5M</td></tr><tr><td>FWSVD+ARS (ours)</td><td>81.22</td><td>81.71</td><td>3.60</td><td>76.83</td><td>73.65</td><td>64.62</td><td>75.74</td><td>65.34</td><td>-17.47</td><td>19.5M</td></tr><tr><td>+ fine-tuning (ours)</td><td>89.60</td><td>87.03</td><td>39.99</td><td>88.19</td><td>83.43</td><td>86.95</td><td>87.23</td><td>80.35</td><td>-2.46</td><td>19.5M</td></tr></table>

Table 3: Results of GLUE benchmark with compact models. The definition of ‘Avg’ and $\mathrm { \overrightarrow { \Delta } - A v g } ^ { \prime }$ is same as Tab. 1.

![](images/26e5f8dd0cde4df69b323c06b5af74d6be1165e0d5b70f5595cd5676fbc03b46.jpg)  
(a) MRPC

![](images/f64de1315d2acbab9df28a4d7a661721d462b7ea3887950be0151e2f19e397c3.jpg)  
(b) STSB

![](images/e36b66eda77c55f6e4f6d6d7a4b4bd8124e9cd749e4f7276f570502e39609351.jpg)  
(c) COLA  
Figure 3: The fine-tuning loss averaging from three different random seeds given $p = 0 . 4 8$ with BERT.

With both compression rates (Tab. 1 and Tab. 2), ARS is much more effective in retaining performance before fine-tuning than SVD, suggesting that adaptive selection of the number of ranks has the potential for fine-tuning less/free compression.

## 4.3 GLUE Results for Compact Models

ARS already shows promising results when compressing BERT, and a follow-up question is whether it can improve the results on compact models. To verify this, we apply FWSVD and FWSVD+ARS on DistillBERT (Sanh et al., 2019) and MobileBERT (Sun et al., 2020). We choose FWSVD and FWSVD+ARS since they achieve the best ∆-Avg on BERT. The overall results are shown in Tab. 3.

For DistillBERT, we still uniformly remove 67% of ranks for FWSVD, and we let $\begin{array} { r l } { p } & { { } = } \end{array}$

<table><tr><td rowspan=1 colspan=1>Settings</td><td rowspan=1 colspan=1>#Samples</td><td rowspan=1 colspan=1>QQP</td><td rowspan=1 colspan=1>SST-2</td><td rowspan=1 colspan=1>QNLI</td></tr><tr><td rowspan=2 colspan=1>FWSVD+ARS</td><td rowspan=2 colspan=1>400060008000</td><td rowspan=2 colspan=1>69.7576.9177.42</td><td rowspan=1 colspan=1>83.37</td><td rowspan=1 colspan=1>64.10</td></tr><tr><td rowspan=1 colspan=1>84.6385.21</td><td rowspan=1 colspan=1>77.7677.87</td></tr><tr><td rowspan=3 colspan=1>+fine-tuning</td><td rowspan=3 colspan=1>400060008000</td><td rowspan=1 colspan=1>86.97</td><td rowspan=1 colspan=1>91.06</td><td rowspan=1 colspan=1>89.68</td></tr><tr><td rowspan=2 colspan=1>87.3787.57</td><td rowspan=1 colspan=1>91.06</td><td rowspan=1 colspan=1>90.04</td></tr><tr><td rowspan=1 colspan=1>91.28</td><td rowspan=1 colspan=1>90.48</td></tr></table>

Table 4: The effect of the number of samples.
<table><tr><td rowspan=1 colspan=1>Settings</td><td rowspan=1 colspan=1>MRPC</td><td rowspan=1 colspan=1>STSB</td><td rowspan=1 colspan=1>COLA</td></tr><tr><td rowspan=1 colspan=1>w/o Rank Selectionw/o hypernetworkwlo Ralign</td><td rowspan=1 colspan=1>81.66 (-7.74)88.12 (-1.28)88.90 (-0.50)</td><td rowspan=1 colspan=1>87.02 (-1.50)88.31 (-0.22)88.03 (-0.49)</td><td rowspan=1 colspan=1>44.84 (-10.17)49.92 (-5.09)53.50 (-1.51)</td></tr><tr><td rowspan=1 colspan=1>ARS</td><td rowspan=1 colspan=1>89.40</td><td rowspan=1 colspan=1>88.52</td><td rowspan=1 colspan=1>55.01</td></tr></table>

Table 5: Ablation study on BERT when $p = 0 . 4 8 .$

0.48 for FWSVD+ARS. Clearly, FWSVD+ARS performs better than FWSVD for DistillBERT, and the gap is 18.79/2.67 regarding average task performance before and after fine-tuning. For MobileBERT, we uniformly remove 40% of the ranks for FWSVD, and we set $\textit { p } = \ 0 . 7 5$ for FWSVD+ARS. FWSVD+ARS outperforms FWSVD by 18.99/1.71 with or without finetuning. In short, ARS continuously improves lowrank factorization for compact models like Mobile-BERT or DistillBERT.

## 4.4 Compression on LLaMA-7B

In this section, we applied our method to LLaMA-7B. We removed around 75% of parameters for this setting. We compared our method with three baselines: (1) training from scratch with a similar number of parameters, (2) WSVD with uniform rank selections, and (3) LLM-pruner (Ma et al., 2023). For WSVD, ARS, and Scratch settings, the compressed models are fine-tuned for 576 A-100 GPU hours, which is less than 1% of the cost for training LLaMA-7B. More training and evaluation details are presented in the appendix. The results are shown in Tab. 6. From the table, we can see that our proposed ARS achieves the best average performance on these 6 tasks. LLM-Pruner performs better on OBQA and ARC-c, but the number of parameters doubles compared to ARS. LLM-Pruner and our method use two different ways to fine-tune model weights, where LLM-Pruner is fine-tuned with LoRA (Hu et al., 2021) on Alpaca (Taori et al., 2023). Our results suggest that fine-tuning with the pre-training setting is more promising than LoRA+Alpaca for a larger compression rate. Training from scratch shows a much worse performance suggesting that compression techniques could be an alternative way to create models with different sizes given limited training budgets.

![](images/19bd34f3b4add5c808bee3c489cd23882c2a96214cbee440175190ab44322afd.jpg)  
(a) MRPC

![](images/5365490fd0810e94a9e76fdab72e9f6da32b2537c46b3619e07fb3aa407c4235.jpg)  
(b) STSB

![](images/105882fc3d5adfaba24b10502e4ad0dffd76b48548743ccf33c0992f11ac0609.jpg)  
(c) COLA

Figure 4: The task loss averaging from three different random seeds given $p = 0 . 4 8$ with BERT when learning the number of ranks.
<table><tr><td>Tasks</td><td>BoolQ</td><td>HellaSwag</td><td>OBQA</td><td>WinoGrande</td><td>ARC-e</td><td>ARC-c</td><td>Average</td><td>#Params</td></tr><tr><td>LLaMA-7B</td><td>74.98</td><td>76.18</td><td>42.6</td><td>70.01</td><td>72.85</td><td>44.71</td><td>63.56</td><td>6.7B</td></tr><tr><td>LLM-Pruner</td><td>61.47</td><td>47.56</td><td>35.2</td><td>55.09</td><td>46.46</td><td>28.24</td><td>45.67</td><td>3.4B</td></tr><tr><td>Scratch</td><td>57.13</td><td>39.16</td><td>29.4</td><td>49.64</td><td>41.96</td><td>24.57</td><td>40.31</td><td>1.8B</td></tr><tr><td>WSVD</td><td>60.46</td><td>46.62</td><td>31.4</td><td>55.25</td><td>47.81</td><td>26.45</td><td>44.67</td><td>1.8B</td></tr><tr><td>WSVD+ARS</td><td>63.27</td><td>50.97</td><td>32.0</td><td>56.67</td><td>51.89</td><td>26.71</td><td>46.92</td><td>1.7B</td></tr></table>

Table 6: Comparison results with LLaMA-7B.

## 4.5 Further Analysis

To better understand our method, we present further analysis regarding different perspectives of ARS.

(1) The Number of Samples. In Tab. 4, we show the impact of the number of samples when training the HN. For some datasets, increasing the number of samples for the HN is very helpful such as QQP and QNLI. For SST-2, the impact is not obvious. Increasing the number of samples may have some benefits, but the benefit of using too many samples is marginal. The reason is that, unlike model weights, the search space for the number of ranks is much smaller, and the performance gain becomes less obvious when there are enough samples.

(2) Ablation Study. In Tab. 5, we present the ablation study results on MRPC, STSB, and COLA. ‘w/o Rank Selection’ means we ignore the property of SVD and use the index to perform element-wise selections. Under this setting, we find a significant performance drop. We also plot the training loss in Fig. 3. Clearly, the element-wise rank selection hurts the structure of low-rank factorization, making it much more difficult to regain performance by fine-tuning. This suggests that we should follow the property of SVD instead of ignoring it. ‘w/o hypernetwork’ means that we use a simple baseline with element-wise binary gates and keep other settings intact. In this setting, the performance has an obvious drop, and we found it harder to reach the pre-defined compression rate p, and it is generally more difficult to optimize (takes more steps, oscillating of training losses). Without $\mathcal { R } _ { a l i g n }$ , our method suffers from an obvious performance decrease, which verifies the benefit of encouraging masks to follow the sorted singular values from SVD.

(3) Effectiveness of HN. We plot the task loss when learning the number of ranks with or without using HN in Fig. 4, which is also the setting of ‘w/o hypernetwork’ in Tab. 5. It is clear that our method can find a better solution and achieve a much faster convergence rate with HN on MRPC, STSB, and COLA. No doubt, HN largely improves the efficiency when learning the number of ranks. The plots of the loss are shown in the Appendix. (4) Generation Speed Comparison. In Tab. 7, we further show the generation speed comparison between ARS 1.7B and Llama-7B. Both models are deployed on the mobile device: S23 Ultra 12GB. In short, the generation speed increases as the number of parameters decreases. If both models are quantized to 8 bits, then the generation speed of the ARS 1.7B model is around 4.7 faster than the original model. If both models are quantized to 4 bits, then the generation speed of our model is around 5.4 faster than the original model.

<table><tr><td>Device Name</td><td>Quantization</td><td>Model</td><td>Model Size</td><td>Per Token Time</td><td>Tokens Per Sec</td></tr><tr><td rowspan="4">S23 Ultra 12GB</td><td rowspan="2">8-bit</td><td>Llama-7B</td><td>7.6 GB</td><td>301.3ms</td><td>3.3</td></tr><tr><td>ARS-1.7B</td><td>1.9 GB</td><td>55.7 ms</td><td>17.9</td></tr><tr><td rowspan="2">4-bit</td><td>Llama-7B</td><td>4.0 GB</td><td>221.8 ms</td><td>4.5</td></tr><tr><td>ARS-1.7B</td><td>1.1 GB</td><td>47.1 ms</td><td>21.3</td></tr></table>

Table 7: Generation speed comparison with our method and the original Llama-7B model
<table><tr><td rowspan=1 colspan=1>Settings</td><td rowspan=1 colspan=1>#Params</td><td rowspan=1 colspan=1>WikiText</td><td rowspan=1 colspan=1>PTB</td><td rowspan=1 colspan=1>C4</td></tr><tr><td rowspan=2 colspan=1>ARS</td><td rowspan=2 colspan=1>4.1B3.3B2.5B1.7B</td><td rowspan=2 colspan=1>115.62404.491177.743893.10</td><td rowspan=1 colspan=1>183.12</td><td rowspan=2 colspan=1>117.76389.441100.353621.82</td></tr><tr><td rowspan=1 colspan=1>581.23522.994286.49</td></tr><tr><td rowspan=1 colspan=1>+fine-tuning</td><td rowspan=1 colspan=1>4.1B3.3B2.5B1.7B</td><td rowspan=1 colspan=1>15.9817.0718.5320.54</td><td rowspan=1 colspan=1>20.6521.9923.7526.82</td><td rowspan=1 colspan=1>19.0719.9321.3523.53</td></tr></table>

Table 8: The effect of different pruning rates. We report the perplexity on WikiText, PTB, and C4.

(5) The Effect of Different Pruning Rates for Llama-7B. We present the result before and after fine-tuning in Tab. 8. The fine-tuning setting for this experiment is quite short, the model is only fine-tuned on around 0.16B tokens. After a short fine-tuning, the perplexity of the model can be quickly recovered. To recover the general ability of the original model, it still requires a longer fine-tuning period.

## 5 Conclusion

In this paper, we proposed a new algorithm that adaptively selects the number of ranks for low-rank approximation of language models. We proposed to use a hypernetwork to predict the number of ranks for each operation. The predicted number of ranks is regularized using the SVD property and is encouraged to produce top-k-like binary masks. Our method resolved the issue with the ordinary masking that resulted in element-wise rank selections, delivering stable performance gain in a comprehensive collection of experiments. The extensive results also show our advantage over previous low-rank methods with uniform rank selections.

## References

Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. 2023. Pythia: A suite for analyzing large language models across training and scaling. arXiv preprint arXiv:2304.01373.

Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. 2014. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. 2020. The Pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027.

Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2021. A framework for few-shot language model evaluation.

Shangqian Gao, Feihu Huang, Yanfu Zhang, and Heng Huang. 2022. Disentangled differentiable network pruning. In European Conference on Computer Vision, pages 328–345. Springer.

Shangqian Gao, Burak Uzkent, Yilin Shen, Heng Huang, and Hongxia Jin. 2023a. Learning to jointly share and prune weights for grounding based vision and language models. In The Eleventh International Conference on Learning Representations.

Shangqian Gao, Zeyu Zhang, Yanfu Zhang, Feihu Huang, and Heng Huang. 2023b. Structural alignment for network pruning through partial regularization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17402– 17412.

Gene H Golub and Christian Reinsch. 1971. Singular value decomposition and least squares solutions. In Linear algebra, pages 134–151. Springer.

Shaopeng Guo, Yujie Wang, Quanquan Li, and Junjie Yan. 2020. Dmcp: Differentiable markov channel pruning for neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1539–1547.

Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, and Song Han. 2018. Amc: Automl for model compression and acceleration on mobile devices. In Proceedings of the European conference on computer vision (ECCV), pages 784–800.

Charles Herrmann, Richard Strong Bowen, and Ramin Zabih. 2020. Channel selection using gumbel softmax. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVII, pages 241–257. Springer.

Yen-Chang Hsu, Ting Hua, Sungen Chang, Qian Lou, Yilin Shen, and Hongxia Jin. 2021. Language model compression with weighted low-rank factorization. In International Conference on Learning Representations.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Ting Hua, Yen-Chang Hsu, Felicity Wang, Qian Lou, Yilin Shen, and Hongxia Jin. 2022. Numerical optimizations for weighted low-rank estimation on language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 1404–1416. Association for Computational Linguistics.

Eric Jang, Shixiang Gu, and Ben Poole. 2016. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144.

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. 2019. Tinybert: Distilling bert for natural language understanding. arXiv preprint arXiv:1909.10351.

Diederik P Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In ICLR (Poster).

François Lagunas, Ella Charlaix, Victor Sanh, and Alexander M Rush. 2021. Block pruning for faster transformers. arXiv preprint arXiv:2109.04838.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. 2019. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942.

Yang Liu. 2019. Fine-tune bert for extractive summarization. arXiv preprint arXiv:1903.10318.

Zechun Liu, Haoyuan Mu, Xiangyu Zhang, Zichao Guo, Xin Yang, Kwang-Ting Cheng, and Jian Sun. 2019. Metapruning: Meta learning for automatic neural network channel pruning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3296–3305.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. 2023. Llm-pruner: On the structural pruning of large language models. arXiv preprint arXiv:2305.11627.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843.

Pavlo Molchanov, Arun Mallya, Stephen Tyree, Iuri Frosio, and Jan Kautz. 2019. Importance estimation for neural network pruning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11264–11272.

Matan Ben Noach and Yoav Goldberg. 2020. Compressing pre-trained language models by matrix decomposition. In Proceedings of the 1st Conference of the Asia-Pacific Chapter ofthe Associationfor Computational Linguistics and the 10th International Joint Conference on Natural Language Processing, pages 884–889.

Razvan Pascanu and Yoshua Bengio. Revisiting natural gradient for deep networks. arXiv preprint arXiv:1301.3584.

Razvan Pascanu and Yoshua Bengio. 2014. Revisiting natural gradient for deep networks. In In International Conference on Learning Representations (ICLR).

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32, pages 8024–8035. Curran Associates, Inc.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392.

Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V Le. 2019. Regularized evolution for image classifier architecture search. In Proceedings ofthe aaai conference on artificial intelligence, volume 33, pages 4780–4789.

Machel Reid, Edison Marrese-Taylor, and Yutaka Matsuo. 2021. Subformer: Exploring weight sharing for parameter efficiency in generative transformers. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 4081–4090.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

Victor Sanh, Thomas Wolf, and Alexander Rush. 2020. Movement pruning: Adaptive sparsity by fine-tuning. Advances in Neural Information Processing Systems, 33:20378–20389.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami, Michael W Mahoney, and Kurt Keutzer. 2020. Q-bert: Hessian based ultra low precision quantization of bert. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8815–8821.

Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. 2019. Patient knowledge distillation for bert model compression. arXiv preprint arXiv:1908.09355.

Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny Zhou. 2020. Mobilebert: a compact task-agnostic bert for resource-limited devices. arXiv preprint arXiv:2004.02984.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https:// github.com/tatsu-lab/stanford\_alpaca.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2019a. Glue: A multi-task benchmark and analysis platform for natural language understanding. In 7th International Conference on Learning Representations, ICLR 2019.

Ziheng Wang, Jeremy Wohlwend, and Tao Lei. 2019b. Structured pruning of large language models. arXiv preprint arXiv:1910.04732.

Genta Indra Winata, Andrea Madotto, Jamin Shin, Elham J Barezi, and Pascale Fung. 2019. On the effectiveness of low-rank matrix factorization for lstm model compression. arXiv preprint arXiv:1908.09982.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen,

Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Mengzhou Xia, Zexuan Zhong, and Danqi Chen. 2022. Structured pruning learns compact and accurate models. In Association for Computational Linguistics (ACL).

Zeyu Zhang, Thuy Vu, Sunil Gandhi, Ankit Chadha, and Alessandro Moschitti. 2022. Wdrass: A web-scale dataset for document retrieval and answer sentence selection. CIKM ’22, page 4707–4711, New York, NY, USA. Association for Computing Machinery.

Zeyu Zhang, Thuy Vu, and Alessandro Moschitti. 2021. Joint models for answer verification in question answering systems. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3252–3262, Online. Association for Computational Linguistics.

Zeyu Zhang, Thuy Vu, and Alessandro Moschitti. 2023. Double retrieval and ranking for accurate question answering. In Findings ofthe Associationfor Computational Linguistics: EACL 2023, pages 1751–1762, Dubrovnik, Croatia. Association for Computational Linguistics.

## A Limitations

Our work adaptively learns the number of ranks for each layer for individual tasks. As a result, the limitation of our method is that we always need to find a new configuration of the number of ranks for a new task, where the learned number of ranks for previous tasks can not be re-used on new tasks. This limitation brings additional computational costs for each new task. Fortunately, this additional cost is trivial on large datasets. For example, it takes 2.2 V100 GPU hours to train BERT on MNLI. Training HN on 4000 samples costs around 0.1 V100 GPU hours on this task.

Alternatively, if we can use some statistics to capture the dataset distribution and incorporate it to learn the number of ranks, we may be able to predict the proper configuration of the number of ranks based on some statistics about the data distribution of the new task. However, this may substantially increase the training time for HN since the problem becomes much more complex.

## B The Architecture of Hypernetwork

Table A1: The architecture of hypernetwork.  
Input z   
Bi-GRU(32,64) LayerNorm GeLU   
Linear<sub>l</sub>(128, N<sub>l</sub>) Outputs o<sub>l</sub>, $l = 1 , \cdots , L$

As we discussed in the paper, the Hypernetwork is composed of linear layers and Bi-GRUs, and now we present the architecture of the HN in Tab. A1. z is initially sampled from a normal distribution, and it is then fixed during training. Outputs $o _ { l }$ are continuous values. We use the following equation to covert it into $m _ { l } \mathbf { \cdot }$

$$
m _ { l } = \mathrm { r o u n d } ( \mathrm { s i g m o i d } ( ( o _ { l } + g + b ) / \tau ) ) ,\tag{9}
$$

where sigmoid( ) is the sigmoid function, round( ) is the rounding function, g is sampled from Gumbel distribution (g  Gumbel(0, 1)), b is a constant value to make sure HN starts from the full rank, and $\tau$ is the temperature hyper-parameter. As shown in Eq. 9, straight-through Gumbel-Sigmoid (Jang et al., 2016) are used to produce the final binary vector m. For all experiments, we set $\tau = 0 . 4$ and $b = 3 . 0 .$

<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Tables</td><td rowspan=1 colspan=1>Models</td><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1>r</td></tr><tr><td rowspan=3 colspan=1>GLUE</td><td rowspan=1 colspan=1>Tab. 1</td><td rowspan=1 colspan=1>BERT-base</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>0.33</td></tr><tr><td rowspan=2 colspan=1>Tab. 2Tab. 3Tab. 3</td><td rowspan=2 colspan=1>BERT-baseDistllBERTMobileBERT</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.22</td></tr><tr><td rowspan=1 colspan=1>0.480.75</td><td rowspan=1 colspan=1>0.330.60</td></tr><tr><td rowspan=1 colspan=1>Pile</td><td rowspan=1 colspan=1>Tab.6</td><td rowspan=1 colspan=1>LLaMA-7B</td><td rowspan=1 colspan=1>0.24</td><td rowspan=1 colspan=1>0.15</td></tr><tr><td rowspan=1 colspan=1>WikiText-103</td><td rowspan=1 colspan=1>Tab. A4</td><td rowspan=1 colspan=1>Pythia-160m</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>0.36</td></tr></table>

Table A2: Choice of $p$ for different models. $p$ is the remained number of parameters divided by the total parameters. $\cdot _ { r } ,$ represents the ratio of ranks uniformly preserved by SVD, IWSVD, and FWSVD.

## C Implementation and Training Details

For BERT based models on GLUE tasks (Wang et al., 2019a), we use Huggingface (Wolf et al., 2020) codes for experiments, which is under Apache 2.0 license. We use the lit-llama https: //github.com/Lightning-AI/lit-llama, also with Apache 2.0 license, codes for fine-tuning the Pythia (Biderman et al., 2023) model on the WikiText (Merity et al., 2016) dataset. The litllama code is also used to fine-tune the compressed LLaMA-7B models on Pile (Gao et al., 2020).

GLUE (Wang et al., 2019a) contains nine English sentence understanding tasks, which cover a broad range of domains, data quantities, and difficulties. Pile (Gao et al., 2020) is an 825 GiB English text corpus targeted at training large-scale language models. The Pile is constructed from 22 diverse high-quality subsets—both existing and newly constructed—many of which derive from academic or professional sources. The WikiText language modeling dataset (Merity et al., 2016) is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. The dataset is available under the Creative Commons Attribution-ShareAlike License. We follow all intended usage of licenses of the datasets and codebase we used.

For all GLUE tasks, we train the HN on 4000 samples (randomly sampled) for 8 epochs. If the dataset has less than 4000 samples, we train the HN on the whole dataset for 8 epochs. For both HN training and BERT training, we set the mini-batch size to 32, and it is trained on 1 Nvidia-V100 GPU.

For the language modeling task, we directly use the pre-trained Pythia-160m model and fine-tune it on the WikiText-103 dataset. We set the sequence length to 512, and the mini-batch size is 64. The initial learning rate is $2 \times 1 0 ^ { - 5 }$ , and the learning rate is linearly decayed. We also list choices of $p$ for different tasks and choices of the preserved ratio

![](images/a87f767f9ed926484a66f17741cffc343c61a957545d656f88efdda022631baa.jpg)  
(a) MRPC

![](images/11cbc03776add1344020e9203c0ddf666c7b3ca31c5815e1def4d7e24b05bb5e.jpg)

![](images/c13a2d7a6df3d6bdfad119283243410b86cb3f4982acf158386837b9be49a9c3.jpg)

![](images/423d94db54b26ce73c4a18e1a744a0289be8bf025a1f1db1850ed8ea263e6db2.jpg)  
(c) COLA

(b) STSB  
(d) MRPC  
![](images/5f09b047e45b9029886f6a7052c807c79d734ba35d104cb49f55f21b4aac4c92.jpg)  
(e) STSB

![](images/d91528caa691dfb71575d3d6b5a1cec0ad697006c773b294a0f58fbc7bf5ec10.jpg)  
(f) COLA

Figure A1: The number of parameters vs. the performance after fine-tuning for SVD/IWSVD and with our ARS variants.  
![](images/42f0fa3b20f30bcd3b5f06c21e5bdc50d7c6e01fe5b5daf9449c3c33155c468c.jpg)  
(a) MRPC

![](images/cdd68262250ed82a67a5512bc110ba5f38650646ddfc8e1bf2d8dba173bf86b2.jpg)  
(b) STSB

![](images/805cd110d2ee1995ce632448a346554077215f561a943d1188f454ffc2d2901a.jpg)  
(c) COLA

Figure A2: The parameter regularization loss averaging from three different random seeds given $p = 0 . 4 8$ with BERT when learning the number of ranks.  
![](images/ae9e364821ed664b8d8ad96f5f933ca2eee1a17b20c9897e76d9ece835a68e6c.jpg)  
Figure A3: Perplexity for different compression rates before fine-tuning on the WikiText-103 dataset.

of ranks by SVD, IWSVD, and FWSVD in Tab. A2.   
The model is trained for 24,000 iterations in total.   
We use 2 Nvidia-A100 GPUs for this experiment.

For LLaMA-7B, we set $p = 0 . 2 4$ and we train HN on the Pile validation dataset on 2 Nvidia-A100 GPUs for 4000 iterations. We use the constant learning rate $1 \times 1 0 ^ { - 3 }$ for this stage. After compression, the model is trained on Pile (Gao et al., 2020) training set with 8 Nvidia-A100 GPUs, minibatch size 48, block size 2048, and a start learning rate of $5 \times 1 0 ^ { - 5 }$ . We use the cosine scheduler for learning rate decay, and the final learning rate is $5 \times 1 0 ^ { - 6 }$ The model is trained for 210,000 steps and the training can be completed within 3 days. The total training tokens are around 20B. Our training code is built on lit-llama: https: //github.com/Lightning-AI/lit-llama. We use llm-eval-harness (Gao et al., 2021) to evaluate the compressed model.

## D Importance Calculation

In this section, we will briefly review the Fisher Information and the other importance scores used in our paper. The Fisher Information measures the amount of information that an observable dataset D carries about a model parameter w. More specif-

![](images/4333c71aa66b222ec65f20111042a0bb5ee4afd6adfddda98eaa7d09022e75b2.jpg)  
(a) MRPC - BERT

![](images/efe71a99fe8f5e3c40bf0c9696547c482a57aab775d2a82f4c9463d0925cc9b8.jpg)  
(b) WikiText - Pythia-160m  
Figure A4: The number of ranks selected by FWSVD+ARS for different tasks.

ically,

$$
\begin{array} { l } { { \displaystyle { \bf I } _ { w } ^ { \mathrm { F I } } = { \bf E } [ \frac \partial { \partial w } ( \log p ( D | w ) ) ^ { 2 } ] } \ ~ } \\ { { \displaystyle ~ \approx \frac 1 { | D | } \sum _ { i = 1 } ^ { | D | } ( \frac \partial { \partial w } \mathcal { L } ( f ( x _ { i } ; w ) , y _ { i } ) ) ^ { 2 } } . } \end{array}
$$

For IWSVD, the importance score follows the definition from (Molchanov et al., 2019):

$$
{ \bf I } _ { w } ^ { \mathrm { I m p } } = { ( \frac { \partial \mathcal { L } } { \partial w } w ) } ^ { 2 } .
$$

## E Additional Results

We further provide the result of #Params vs. performance for SVD/IWSVD and our ARS variants in Fig. A1. SVD+ARS and IWSVD+ARS clearly outperform SVD/IWSVD for almost all compression rates. We also visualize the perplexity before fine-tuning for different compression rates in Fig. A3. FWSVD+ARS outperforms FWSVD at every compression rate for the number of ranks. At a higher compression rate, the perplexity of FWSVD+ARS is often a magnitude lower than WSVD, which shows the advantage of adaptive selections of the number of ranks.

In Fig. A4, we visualize the number of ranks selected by ARS across each operation. In Fig. A4a, ARS allocates more ranks for early to middle layers for MRPC. In Fig. A4b, ARS allocates more ranks for both early and late layers for WikiText. The difference between MPRC and WikiText is probably because the language modeling task focuses on both input contexts and output predictions, and MRPC only needs to measure whether input sentences are equivalent and it is not complex. In summary, ARS can produce different selections of the number of ranks based on different tasks.

To provide a more detailed understanding on the effectiveness of HN, we plot the parameter regularization loss with or without HN. The loss is normalized between 0 and 1 for better visualization.

![](images/d475071ede307f89214febc87b3dfbd9a5bba99eef1189bc807e2024dd7687e2.jpg)  
Figure A5: Training loss on WikiText.

In Fig. A2, we can see that our method with HN can quickly reduce the parameter loss . Without HN, the loss keeps bumping and it seems hard to reach the desired parameter budget without HN.

## F Language Modeling Task with Pythia

We further apply our method to the language modeling task on WikiText-103 (Merity et al., 2016) dataset. Results are shown in Tab. A4. From the table, we can see that FWSVD+ARS performs much better than FWSVD. In particular, FWSVD+ARS compresses 6% more parameters than FWSVD, and the perplexity of it is 3.07 and 3.24 lower than FWSVD on the test and validation splits. FWSVD+ARS even performs better than the baseline on the test split. These results again demonstrate the importance of selecting the number of ranks across different tasks. In Fig. A5, we visualize the training loss of FWSVD and FWSVD+ARS during fine-tuning on WikiText. FWSVD+ARS always starts at a lower loss value, and the gap between FWSVD and ARS is maintained till the end of training. By properly choosing the number of ranks, we obtain a model more suitable for the task, making it easier to regain performance.

## G Comparison with Pruning Methods

We provide further comparison results against structural pruning methods in Tab. A3. For IE (Molchanov et al., 2019), we built this structural pruning baseline for compression language models based on the original method. The training and fine-tuning settings are the same as our method. We compared it with IWSVD+AES since they use the same importance. Our method has a better average task performance before and after fine-tuning. For CoFipruning (Xia et al., 2022), We use the GitHub repository of CoFipruning, and modify some hyperparameters to build a fair comparison baseline. We set the fine-tuning epoch of CoFipruning to 3 epochs which is the same as our method. In addition, the first stage of CoFipruning is reduced to 20 epochs for small datasets and 5 epochs for large datasets. Recall that our method first trains the model for 3 epochs for each task and the hypernetwork is trained at most for 1000 steps. As a result, even though we reduced the training time for CoFipruning, it still has a larger computational cost than our method. In addition, we turned off the knowledge distillation of CoFipruning since our method does not rely on any form of knowledge distillation. Our method still has a clear advantage in this setting.

<table><tr><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>MRPC</td><td rowspan=1 colspan=1>STSB</td><td rowspan=1 colspan=1>COLA</td><td rowspan=1 colspan=1>SST-2</td><td rowspan=1 colspan=1>MNLI</td><td rowspan=1 colspan=1>QNLI</td><td rowspan=1 colspan=1>QQP</td><td rowspan=1 colspan=1>Avg</td><td rowspan=1 colspan=1># Params</td></tr><tr><td rowspan=4 colspan=1>IE (Molchanov et al., 2019)+ fine-tuningIWSVD+ARS (ours)+ fine-tuning (ours)</td><td rowspan=2 colspan=1>45.5887.03</td><td rowspan=1 colspan=1>64.90</td><td rowspan=1 colspan=1>8.04</td><td rowspan=1 colspan=1>66.92</td><td rowspan=1 colspan=1>48.82</td><td rowspan=1 colspan=1>49.48</td><td rowspan=1 colspan=1>50.70</td><td rowspan=1 colspan=1>47.77</td><td rowspan=1 colspan=1>66.8M</td></tr><tr><td rowspan=1 colspan=1>86.74</td><td rowspan=1 colspan=1>38.12</td><td rowspan=1 colspan=1>89.01</td><td rowspan=1 colspan=1>83.86</td><td rowspan=1 colspan=1>88.29</td><td rowspan=1 colspan=1>85.92</td><td rowspan=3 colspan=1>79.5867.2883.14</td><td rowspan=3 colspan=1>66.8M65.1M65.1M</td></tr><tr><td rowspan=2 colspan=1>81.5888.13</td><td rowspan=1 colspan=1>76.93</td><td rowspan=2 colspan=1>23.9752.88</td><td rowspan=2 colspan=1>83.9491.40</td><td rowspan=1 colspan=1>51.88</td><td rowspan=1 colspan=1>77.58</td><td rowspan=2 colspan=1>75.0587.59</td></tr><tr><td rowspan=1 colspan=1>88.23</td><td rowspan=1 colspan=1>83.86</td><td rowspan=1 colspan=1>89.91</td></tr><tr><td rowspan=2 colspan=1>CoFiPruning (Xia et al., 2022)WSVD+ARS+fine-tuning (ours)</td><td rowspan=2 colspan=1>87.7089.40</td><td rowspan=2 colspan=1>86.9088.52</td><td rowspan=2 colspan=1>43.1655.01</td><td rowspan=2 colspan=1>89.5091.06</td><td rowspan=2 colspan=1>82.9483.68</td><td rowspan=2 colspan=1>87.7389.68</td><td rowspan=2 colspan=1>86.3587.41</td><td rowspan=2 colspan=1>80.6183.54</td><td rowspan=1 colspan=1>66.7M</td></tr><tr><td rowspan=1 colspan=1>65.1M</td></tr></table>

Table A3: Comparison against structural pruning methods.

<table><tr><td>Settings</td><td>Test (ppl)</td><td>Val (ppl)</td><td>#PT</td><td>#PM</td><td>↓#PM</td></tr><tr><td>Pythia-160m</td><td>25.09</td><td>24.97</td><td>162.3m</td><td>85.0M</td><td></td></tr><tr><td>FWSVD</td><td>18331.07</td><td>20525.75</td><td>123.2M</td><td>46.0M</td><td>45.9%</td></tr><tr><td>+fine-tuning</td><td>28.05</td><td>29.07</td><td>123.2M</td><td>46.0M</td><td>45.9%</td></tr><tr><td>FWSVD+ARS</td><td>3020.17</td><td>3041.04</td><td>118.0M</td><td>40.8M</td><td>52.0%</td></tr><tr><td>+fine-tuning</td><td>24.98</td><td>25.83</td><td>118.0M</td><td>40.8M</td><td>52.0%</td></tr></table>

Table A4: Results of the language modeling task on WikiText-103. ‘PT’ represents the total number of parameters. ‘PM’ represents the number of model parameters excluding the Embedding layer. ’ppl’ represents perplexity.