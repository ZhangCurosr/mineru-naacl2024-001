# Multi-Operational Mathematical Derivations in Latent Space

Marco Valentino<sup>1</sup>, Jordan Meadows<sup>2</sup>, Lan Zhang<sup>2</sup>, André Freitas<sup>1,2,3</sup>

<sup>1</sup>Idiap Research Institute, Switzerland

<sup>2</sup> Department of Computer Science, University of Manchester, United Kingdom <sup>3</sup> Cancer Biomarker Centre, CRUK Manchester Institute, United Kingdom {marco.valentino, andre.freitas}@idiap.ch {jordan.meadows, lan.zhang-6}@manchester.ac.uk

## Abstract

This paper investigates the possibility of approximating multiple mathematical operations in latent space for expression derivation. To this end, we introduce different multi-operational representation paradigms, modelling mathematical operations as explicit geometric transformations. By leveraging a symbolic engine, we construct a large-scale dataset comprising 1.7M derivation steps stemming from 61K premises and 6 operators, analysing the properties of each paradigm when instantiated with state-ofthe-art neural encoders. Specifically, we investigate how different encoding mechanisms can approximate expression manipulation in latent space, exploring the trade-off between learning different operators and specialising within single operations, as well as the ability to support multi-step derivations and out-of-distribution generalisation. Our empirical analysis reveals that the multi-operational paradigm is crucial for disentangling different operators, while discriminating the conclusions for a single operation is achievable in the original expression encoder. Moreover, we show that architectural choices can heavily affect the training dynamics, structural organisation, and generalisation of the latent space, resulting in significant variations across paradigms and classes of encoders<sup>1</sup>.

## 1 Introduction

To what extent are neural networks capable of mathematical reasoning? This question has led many researchers to propose various methods to train and test neural models on different math-related tasks, such as math word problems, theorem proving, and premise selection (Lu et al., 2023; Meadows and Freitas, 2023; Mishra et al., 2022a; Ferreira et al., 2022; Ferreira and Freitas, 2020; Welleck et al., 2021; Valentino et al., 2022; Mishra et al., 2022b; Petersen et al., 2023). These methods aim to investigate how neural architectures learn and generalise mathematical concepts and symbolic rules, and how they cope with characteristic challenges of mathematical inference, such as abstraction, compositionality, and systematicity (Welleck et al., 2022; Mishra et al., 2022a).

![](images/274f59c0c3b48ce254910738d77372b3a373730801902a19385b042840bb7687.jpg)  
Figure 1: Can neural encoders learn to approximate multiple mathematical operators in latent space? Given a premise x, we investigate the problem of applying a sequence of latent operations $( t _ { 1 } , \ldots , t _ { n } )$ to derive valid mathematical expressions $( y _ { 1 } , \ldots , y _ { n } )$

In general, a key challenge in neural mathematical reasoning is to represent expressions and formulae into a latent space to enable the application of multiple operations in specific orders under contextual constraints. Existing methods, however, typically focus on single-operational inference – i.e., optimising a latent space to approximate a specific mathematical operation (Lee et al., 2019; Lample and Charton, 2019; Welleck et al., 2022). Encoding multiple operations in the same latent space, therefore, remains an unexplored challenge that will likely require the development of novel mechanisms and representational paradigms.

To investigate this problem, this paper focuses on equational reasoning, intended as the derivation ofexpressionsfrom premises via the sequential application of specialised mathematical operations (i.e., addition, subtraction, multiplication, division, integration, differentiation). As derivations represent the workhorse of applied mathematical reasoning (including derivations in physics and engineering), projecting expressions and operators into a well-organised geometric space can unveil a myriad of applications, unlocking the approximation of mathematical solutions that are multiple steps apart within the embedding space via distance metrics and vector operations.

Specifically, this paper posits the following overarching research questions: RQ1:“How can different representational paradigms and encoding mechanisms support expression derivation in latent space?”; RQ2:“What is the representational tradeoffbetween generalising across different mathematical operations and specialising within single operations?”; RQ3:“To what extent can different encoding mechanisms enable multi-step derivations through the sequential application andfunctional composition oflatent operators?”; RQ4:“To what extent can different encoding mechanisms support out-of-distribution generalisation?”

To answer these questions, we investigate jointembedding predictive architectures (LeCun, 2022) by introducing different multi-operational representation paradigms (i.e., projection and translation) to model mathematical operations as explicit geometric transformations within the latent space. Moreover, by leveraging a symbolic engine (Meurer et al., 2017), we build a large-scale dataset containing 1.7M derivation steps which span diverse mathematical expressions and operations. To understand the impact of different encoding schemes on equational reasoning, we instantiate the proposed architectures with state-of-the-art neural encoders, including Graph Neural Networks (GNNs) (Hamilton et al., 2017; Kipf and Welling, 2016), Convolutional Neural Networks (CNNs) (Li et al., 2021; Kim, 2014), Recurrent Neural Networks (RNNs) (Yu et al., 2019; Hochreiter and Schmidhuber, 1996), and Transformers (Vaswani et al., 2017), analysing the properties of the latent spaces and the ability to support multi-step derivations and generalisation.

Our empirical evaluation reveals that the multioperational paradigm is crucial for disentangling different mathematical operators (i.e., crossoperational inference), while the discrimination of the conclusions for a single operation (i.e., intraoperational inference) is achievable in the original expression encoder. Moreover, we show that architectural choices can heavily affect the training dynamics and the structural organisation of the latent space, resulting in significant variations across paradigms and classes of encoders.

Overall, we conclude that the translation paradigm can result in a more fine-grained and smoother optimisation of the latent space, which better supports cross-operational inference and enables a more balanced integration. Regarding the encoders, we found that sequential models achieve more robust performance when tested on multistep derivations, while graph-based encoders, on the contrary, exhibit better generalisation to out-ofdistribution examples.

## 2 Multi-Operational Derivations

Given a premise $x - \mathrm { i . e . }$ , a mathematical expression including variables and constants, and a set of operations $T = \{ t _ { 1 } , t _ { 2 } , \ldots t _ { n } \} - \mathbf { e . g . }$ , addition, multiplication, differentiation, etc., we investigate the extent to which a neural encoder can approximate a mathematical function $f ( x , t _ { i } ; V ) = Y _ { t _ { i } }$ that takes the premise x and any operation $t _ { i } \in T$ as inputs, and produces the set of valid expressions $Y _ { t _ { i } } = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { m } \}$ derivable from x via $t _ { i }$ given $v \in V$ , where V is a predefined set of operands such that $y _ { j } = t _ { i } ( x , v _ { j } )$ . In this work, we focus on atomic operations in which V includes symbols representing variables.

For example, consider the following premise x:

$$
u + \cos \left( \log \left( - z + o \right) \right)
$$

If the set of operands is $V = \{ z , u \}$ , and the operation $t _ { i }$ is addition, then the application of $t _ { i }$ to x should result in the set of expressions:

$$
\begin{array} { c } { { Y _ { a d d } = \left\{ z + u + \cos { \left( \log { \left( - z + o \right) } \right) } , \ldots , \right. } } \\ { { \left. \qquad \mathrm { 2 } u + \cos { \left( \log { \left( - z + o \right) } \right) } \right\} } } \end{array}
$$

Instead, if $t _ { i }$ is differentiation, then $f ( x , t _ { i } ; V )$ should result in a different set:

$$
Y _ { d i f f } = \{ \frac { \sin \left( \log \left( - z + o \right) \right) } { - z + o } , \ldots , 1 \}
$$

Notably, the recursive application of $f$ to any of the expressions in $Y _ { t _ { i } }$ can generate a new set of conclusions derivable from x in multiple steps.

![](images/7c1bd3ee6b1b1f6c5ce9a94da1e8d04b32e593752add89db4db11d5aecad5a72.jpg)  
Figure 2: Overview of the proposed joint-embedding predictive architectures for latent multi-operational derivation (left). Schematic workflow for multi-step inference and latent propagation of mathematical operations (right).

Here, the constraint we are interested in, is that $Y _ { a d d }$ and $Y _ { d i f f }$ should be derived via a single expression encoder that maps expressions into a vector space and, at the same time, enables a multi-step propagation of latent operations.

## 2.1 Architectures

To model latent mathematical operations, we investigate the use of joint-embedding predictive architectures (LeCun, 2022). In particular, we introduce two multi-operational paradigms based on projection and translation to learn the representation of expressions and mathematical operators and model an atomic derivation step as an explicit geometric transformation. Figure 2 shows a schematic representation of the architectures.

In general, both projection and translation employ an expression encoder to map the premise x and a plausible conclusion y into vectors, along with an operation encoder that acts as a latent prompt t to discriminate between operators. The goal is then to predict the embedding of a valid conclusion $\mathbf { e } _ { y }$ by applying a transformation to the premise embedding $\mathbf { e } _ { x }$ conditioned on t. Therefore, the two paradigms mainly differ in how expression and operation embeddings are combined to approximate the target results. This setup enables multi-step inference since the predicted embedding $\mathbf { e } _ { y } ^ { \prime }$ can be recursively interpreted as a premise representation for the next iteration (Figure 2, right).

Projection. The most intuitive solution to model latent mathematical operations is to employ a projection layer (Lee et al., 2019). In this case, the premise x and the operator t are first embedded using the respective encoders, which are then fed to a dense predictive layer π to approximate the target conclusion $\mathbf { e } _ { y }$ . The overall objective function can then be formalised as follows:

$$
\phi ( x , t , y ) = - \delta ( \pi ( \mathbf { t } | | \mathbf { e } _ { x } ) , \mathbf { e } _ { y } ) ^ { 2 }\tag{1}
$$

Where δ is a distance function, and $\pi$ represents the dense projection applied to the concatenation of t and $\mathbf { e } _ { x }$ . While many options are available, we implement π using a linear layer to better investigate the representation power of the underlying expression encoder.

Translation. Inspired by research on multirelational graph embeddings (Bordes et al., 2013; Balazevic et al., 2019; Valentino et al., 2023), we frame mathematical inference as a multi-relational representation learning problem. In particular, it is possible to draw a direct analogy between entities and relations in a knowledge graph and mathematical operations. Within the scope of the task, as defined in Section 2, the application of a general operation can be interpreted as a relational triple $< x , t , y >$ , in which a premise expression x corresponds to the subject entity, a conclusion y corresponds to the object entity, and the specific operation type t represents the semantic relation between entities. Following this intuition, we formalise the learning problem via a translational objective:

$$
\phi ( x , t , y ) = - \delta ( \mathbf { T e } _ { x } , \mathbf { e } _ { y } + \mathbf { t } ) ^ { 2 }\tag{2}
$$

Where δ is a distance function, $\mathbf { e } _ { x } , \mathbf { e } _ { y }$ , t, are the embeddings of premise expression, conclusion and operation, and T is a diagonal operation matrix.

## 2.2 Data Generation

We generate synthetic data to support the exploration of the above architectures, inspired by a recent approach that relies on a symbolic engine to generate equational reasoning examples (Meadows et al., 2023b). In particular, we use SymPy (Meurer et al., 2017) to construct a dataset containing expressions in both LaTeX and SymPy surface forms.

Here, premises and variables are input to 6 operations to generate further expressions, presently focusing on differentiation, integration, addition, subtraction, multiplication, and division. Concrete examples of entries in the dataset are reported in the Appendix.

Premises. To generate a premise expression, a set of symbols is first sampled from a vocabulary. Subsequently, an initial operator is applied to symbols to generate an expression via the SymPy engine. To generate more complex expressions, this process is repeated iteratively for a fixed number of steps. This process is formalised in Algorithm 1 (see Appendix). The final dataset includes 61K premises each containing between 2 to 5 variables.

Applying Operations to Premises. For a given premise, a set of operand variables (denoted by V in Section 2) are sampled from the vocabulary and added to the set of symbols that comprise the premise. All valid combinations of premise and operands are then input to each operator (via SymPy) to generate conclusions derivable via atomic derivation steps. The resulting dataset contains a total of 1.7M of such atomic steps. This data is used to train and evaluate models on singlestep inference before testing generalisation capabilities to multiple steps.

Multi-Step Derivations. To test the models’ ability to derive expressions obtained after the sequential application of operations, we randomly sample 5K premises from the single-step dataset described above and iteratively apply up to 6 operations to each premise using a randomly sampled variable operand from the vocabulary for each step. We adopt this methodology to generate a total of 2.7K multi-step examples.

## 2.3 Expression Encoders

Thanks to their generality, the multi-operational architectures can be instantiated with different classes of expression encoders. In particular, we experiment with both graph-based and sequential models, exploring embeddings with different dimensions (i.e., 300, 512, and 768). The graph-based encoders are trained on operation trees extracted from the SymPy representation, while the sequential models are trained on LaTeX expressions. We adopted the following expression encoders in our experiments:

Graph Neural Networks (GNNs). GNNs have been adopted for mathematical inference thanks to their ability to capture explicit structural information (Lee et al., 2019). Here, we consider different classes of GNNs to experiment with models that can derive representations from operation trees. Specifically, we employ a 6-layer Graph-Sage<sup>2</sup>(Hamilton et al., 2017) and Graph Convolutional Network (GCN)<sup>3</sup> (Kipf and Welling, 2016) to investigate transductive and non-transductive methods. To build the operation trees, we directly parse the SymPy representation described in Appendix A.

Convolutional Neural Networks (CNNs). CNNs represent an effective class of models for mathematical representation learning thanks to their translation invariance property that can help localise recurring symbolic patterns within expressions (Petersen et al., 2023). Here, we employ a 1D CNN architecture typically used for text classification tasks (Kim, 2014), with three filter sizes 3, 4, and 5, each with 100 filters.

Recurrent Neural Networks (RNNs). Due to the sequential nature of mathematical expressions, we experiment with RNNs that have been successful in modelling long-range dependencies for sentence representation (Yu et al., 2019; Hochreiter and Schmidhuber, 1996). In particular, we employ a Long-Short Term Memory (LSTM) network with 2 layers.

Transformers. Finally, we experiment with a Transformer encoder with 6 and 8 attention heads and 6 layers, using a configuration similar to the one proposed by Vaswani et al. (2017)<sup>4</sup>. Differently from other models, Transformers use the attention mechanism to capture implicit relations between tokens, allowing, at the same time, experiments with a larger number of trainable parameters.

## 2.4 Operation Encoders

The operation encoders are implemented using a lookup table similar to word embeddings (Mikolov et al., 2013), where each entry corresponds to the vector of a mathematical operator. We experiment with dense<sup>5</sup> embeddings for the translation model and instantiate the projection architecture with both dense and $o n e { - } h o t ^ { 6 }$ embeddings. The translation model requires the operation embeddings to be the same size as the expression embeddings, admitting, therefore, only dense representations.

<table><tr><td></td><td>MAP</td><td>Hit@1</td><td>Hit@3</td><td>MAP</td><td>Hit@1</td><td>Hit@3</td><td>Avg. MAP</td></tr><tr><td>Projection (One-hot)</td><td colspan="3">Cross-op.</td><td colspan="3">Intra-op.</td><td></td></tr><tr><td>GCN</td><td>74.07</td><td>78.35</td><td>88.88</td><td>93.34</td><td>97.81</td><td>99.01</td><td>83.70</td></tr><tr><td>GraphSAGE</td><td>83.89</td><td>88.43</td><td>96.70</td><td>93.00</td><td>97.45</td><td>98.71</td><td>88.44</td></tr><tr><td>CNN</td><td>69.61</td><td>76.98</td><td>95.20</td><td>92.43</td><td>97.18</td><td>98.63</td><td>81.02</td></tr><tr><td>LSTM</td><td>71.40</td><td>73.50</td><td>90.08</td><td>93.21</td><td>98.01</td><td>99.35</td><td>82.30</td></tr><tr><td>Transformer</td><td>49.35</td><td>46.30</td><td>63.00</td><td>91.66</td><td>96.65</td><td>98.38</td><td>70.50</td></tr><tr><td>Projection (Dense) 1</td><td colspan="5"></td><td colspan="2">1</td></tr><tr><td>GCN</td><td>78.25</td><td>82.50</td><td>92.81</td><td>93.43</td><td>97.91</td><td>99.08</td><td>85.84</td></tr><tr><td>GraphSAGE</td><td>81.05</td><td>83.91</td><td>94.38</td><td>93.18</td><td>97.81</td><td>98.93</td><td>87.11</td></tr><tr><td>CNN</td><td>82.57</td><td>91.40</td><td>98.50</td><td>92.62</td><td>97.15</td><td>99.18</td><td>87.59</td></tr><tr><td>LSTM</td><td>77.17</td><td>81.96</td><td>93.73</td><td>93.68</td><td>98.48</td><td>99.36</td><td>85.42</td></tr><tr><td>Transformer</td><td>71.51</td><td>77.08</td><td>89.43</td><td>92.23</td><td>97.30</td><td>98.53</td><td>81.87</td></tr><tr><td>Translation</td><td colspan="5"></td><td colspan="2">1</td></tr><tr><td>GCN</td><td>85.89</td><td>94.73</td><td>98.85</td><td>90.10</td><td>92.45</td><td>95.61</td><td>87.99</td></tr><tr><td>GraphSAGE</td><td>88.15</td><td>96.31</td><td>99.25</td><td>90.68</td><td>94.51</td><td>96.88</td><td>89.41</td></tr><tr><td>CNN</td><td>84.72</td><td>94.66</td><td>98.70</td><td>90.17</td><td>93.98</td><td>97.96</td><td>87.44</td></tr><tr><td>LSTM</td><td>89.85</td><td>96.70</td><td>99.35</td><td>89.74</td><td>94.60</td><td>97.91</td><td>89.79</td></tr><tr><td>Transformer</td><td>86.64</td><td>95.78</td><td>98.83</td><td>90.93</td><td>96.05</td><td>99.73</td><td>88.78</td></tr></table>

Table 1: Overall performance of different neural encoders and methods for encoding multiple mathematical operations (i.e., integration, differentiation, addition, difference, multiplication, division) in the latent space.

## 3 Training Details

As the models are trained to predict a target embedding, the main goal during optimisation is to avoid a representational collapse in the expression encoder. To this end, we opted for a Multiple Negatives Ranking (MNR) loss with in-batch negative examples (Henderson et al., 2017). This technique allows us to sidestep the explicit selection of the negative sample, enabling a smoother optimisation of the latent space. We trained the models on a total of 12.800 premise expressions with 24 positive examples each derived from the application of 6 operations (see Section 2.2). This produces over 307.200 training instances composed of premise x, operation t, and conclusion y. The models are then trained for 32 epochs with a batch size of 64 (with in-batch random negatives). We found that the best results are obtained with a learning rate of 1e-5.

## 4 Empirical Evaluation

## 4.1 Empirical Setup

We evaluate the performance of different representational paradigms and expression encoders by building held-out dev and test sets. In particular, to assess the structural organisation of the latent space, we frame the task of multi-operational inference as an expression retrieval problem. Given a premise x, an operation t, a sample of positive conclusions $P = \{ p _ { 1 } , . . . , p _ { n } \}$ , and a sample of negative conclusions $N = \{ n _ { 1 } , \dots , n _ { m } \}$ , we adopt the models to predict an embedding $\mathbf { e } _ { y } ^ { \prime }$ (Section 2.1) and employ a distance function δ to rank all the conclusions in P N according to their similarity with ${ \mathbf e } _ { y } ^ { \prime } .$ We implement δ using cosine similarity, and construct two evaluation sets to assess complementary inferential properties, namely:

Cross-operational Inference. A model able to perform multi-operational inference should discriminate between the results of different operations applied to the same premise. Therefore, given a premise x and an operation t (e.g., addition), we construct the negative set N by selecting the positive conclusions resulting from the application of different operations (e.g., differentiation, subtraction) to the same premise x. This set includes a total of 4 positive and 20 negative examples (extracted from the remaining 5 operations) for each premise-operation pair (for a total of 3k dev and 6k test instances).

Intra-operational Inference. While we want the models to discriminate between different operators, a well-optimised latent space should still preserve the ability to predict the results of a single operation applied to different premises. Therefore, given a premise x and an operation t, we construct the negative set N by selecting the positive conclusions resulting from the application of the same operation t to a different sample of premises. This set includes a total of 4 positive and 20 negative examples (extracted from 5 random premises) for each premise-operation pair (for a total of 3k dev and 6k test instances).

![](images/877d3c91d941f4a6148ca209da1dd68140febd74f8cd22ebe1bc7c1b9fa49527.jpg)  
(a) Projection

![](images/e141ed58ec999f646c7c7bec5ee5b323fe179af069a0f01b822298ca2b02e6a8.jpg)  
(b) Translation  
Figure 3: Typical training dynamics of different multi-operational paradigms (MAP on the dev set).

Metrics. The models are evaluated using Mean Average Precision (MAP) and Hit@k. Hit@k measures the percentage of test instances in which at least one positive conclusion is ranked within the top k positions. MAP, on the other hand, measures the overall ranking. We use the average MAP between cross-operational and intra-operational sets (dev) as a criterion for model selection.

## 4.2 Results

Table 1 shows the performance of different encoders and paradigms on the test sets (i.e., evaluating the best models from the dev set, see Table 3). We can derive the following conclusions:

The translation mechanism improves crossoperational inference. The models that use the translation method consistently outperform the models that use the projection method on the crossoperational inference task. This indicates that the translation paradigm can better capture the semantic relations between different operations and preserve them in the latent space. This is attested by the significant improvement achieved by different encoders, involving both graph-based and sequential architectures (e.g., +15.13% and +7.64% for Transformers and GCN respectively).

Trade-off between cross-operational and intraoperational inference. The models that excel at cross-operational inference tend to achieve lower performance on the intra-operational set. This suggests that there is a tension between generalising across different operations and specialising within each operation. Moreover, the results suggest that intra-operational inference represents an easier problem for neural encoders that can be achieved already with sparse multi-operational methods (i.e., models using one-hot projection can achieve a MAP score above 90%).

LSTMs and GraphSAGE achieve the best performance. LSTMs achieve the highest average MAP score, followed by GraphSAGE. These results demonstrate that LSTMs and GraphSAGE can balance between generalisation and specialisation, and leverage both sequential and graphbased information to encode mathematical operations. Moreover, we observe that graph-based models and CNNs tend to exhibit more stable performance across different representational paradigms (e.g., GraphSage achieve an average improvement of 2.3%), while LSTMs and Transformers achieve balanced results only with the translation mechanism (i.e., with an average improvement of 4.37% and 6.91% respectively).

Model size alone does not explain inference performances. The Transformer model, which has the largest number of parameters, exhibits a lower average MAP score (with the projection mechanism in particular). This implies that simply increasing the model complexity or capacity does not guarantee better results (see Table 3 for additional details) and may compromise operational control in the latent space. This suggests that model architecture and the encoding method are more important factors for learning effective representations supporting multiple mathematical operations.

![](images/b64858a9a58cababdde700456802a697e124df2d2445156f6d46ca85488ca265.jpg)  
Figure 4: 2D projection of the latent space before and after an operation-specific transformation. The visualization supports the crucial role of the multi-operational paradigm for cross-operational inference, showing, at the same time, that intra-operational inference concerns larger regions and can be achieved in the original expression encoder.

<table><tr><td>一</td><td>Cross</td><td colspan="3">一 Intra</td></tr><tr><td>Proj. (1-hot)</td><td> $\delta ( \mathbf { e } _ { x } , \mathbf { e } _ { y } )$ </td><td> $\delta ( \mathbf { e } _ { y } ^ { \prime } , \mathbf { e } _ { y } )$ </td><td> $\delta ( \mathbf { e } _ { x } , \mathbf { e } _ { y } )$ </td><td> $\delta ( \mathbf { e } _ { y } ^ { \prime } , \mathbf { e } _ { y } )$ </td></tr><tr><td>GCN GraphSAGE</td><td>00.00</td><td>20.87</td><td>84.46</td><td>84.82</td></tr><tr><td></td><td>00.00</td><td>25.71</td><td>84.00</td><td>85.77</td></tr><tr><td>CNN</td><td>00.00</td><td>19.64</td><td>85.81</td><td>85.86</td></tr><tr><td>LSTM Transformer</td><td>00.00</td><td>24.11</td><td>86.12</td><td>84.87 86.80</td></tr><tr><td></td><td>00.00</td><td>16.52</td><td>88.20</td><td></td></tr><tr><td>Proj. (Dense)</td><td> $\delta ( \mathbf { e } _ { x } , \mathbf { e } _ { y } )$ </td><td> $\delta ( { \bf e } _ { y } ^ { \prime } , { \bf e } _ { y } )$ </td><td> $\delta ( \mathbf { e } _ { x } , \mathbf { e } _ { y } )$ </td><td> $\delta ( \mathbf { e } _ { y } ^ { \prime } , \mathbf { e } _ { y } )$ </td></tr><tr><td>GCN</td><td>00.00</td><td>22.88</td><td>84.38</td><td>85.25</td></tr><tr><td>GraphSAGE</td><td>00.00</td><td>25.47</td><td>84.96</td><td>86.01</td></tr><tr><td>CNN</td><td>00.00</td><td>23.13</td><td>83.38</td><td>82.84</td></tr><tr><td>LSTM</td><td>00.00</td><td>25.66</td><td>84.80</td><td>83.44</td></tr><tr><td>Transformer</td><td>00.00</td><td>21.54</td><td>86.22</td><td>83.80</td></tr><tr><td>Translation</td><td> $\delta ( \mathbf { e } _ { x } , \mathbf { e } _ { y } )$ </td><td> $\delta ( { \bf e } _ { y } ^ { \prime } , { \bf e } _ { y } + { \bf t } )$ </td><td> $\delta ( \mathbf { e } _ { x } , \mathbf { e } _ { y } )$ </td><td> $\delta ( \mathbf { e } _ { y } ^ { \prime } , \mathbf { e } _ { y } + \mathbf { t } )$ </td></tr><tr><td>GCN</td><td></td><td></td><td></td><td></td></tr><tr><td>GraphSAGE</td><td>00.00 00.00</td><td>11.85 11.37</td><td>-07.13 -01.45</td><td>39.76 40.98</td></tr><tr><td>CNN</td><td></td><td></td><td></td><td></td></tr><tr><td>LSTM</td><td>00.00 00.00</td><td>05.23 40.20</td><td>12.32 -00.46</td><td>33.68 51.46</td></tr><tr><td>Transformer</td><td>00.00</td><td>43.38</td><td>03.07</td><td>69.14</td></tr></table>

Table 2: Latent separation of positive and negative examples before $( \mathrm { i } . \mathrm { e } . , \delta ( \mathbf { e } _ { x } , \mathbf { e } _ { y } ) )$ ) and after (i.e., $\delta ( \mathbf { e } _ { y } ^ { \prime } , \mathbf { e } _ { y } ) )$ applying an operation-specific transformation.

## 4.3 Training Dynamics

We conduct an additional analysis to investigate the training dynamics of different architectures. The graphs in Figure 3 show the typical trend for the MAP achieved at different epochs on different evaluation sets. Interestingly, we found that the projection and translation mechanisms optimise the latent space in a different way. The projection paradigm, in fact, prioritises performance on intraoperational inference, with a constant gap between the two sets. Conversely, the translation paradigm supports a rapid optimisation of cross-operational inference, followed by a more gradual improvement on the intra-operational set.

This behaviour can help explain the difference in performances between the models. Specifically, since cross-operational inference is about disentangling operations applied to the same premise, we hypothesise it to require a more fine-grained optimisation in localised regions of the latent space. This optimisation can be compromised when priority is given to the discrimination of different premises, which, as in the case of intra-operational inference, involves a more coarse-grained optimisation in larger regions of the space.

## 4.4 Latent Space Analysis

We further investigate this behaviour by measuring and visualising the latent space in the original expression encoder (i.e., computing $\delta ( \mathbf { e } _ { x } , \mathbf { e } _ { y } ) )$ and after applying a transformation via the operation encoder (i.e., computing $\delta ( \mathbf { e } _ { y } ^ { \prime } , \mathbf { e } _ { y } ) )$ . In particular, Table 2 reports the average difference between the cosine similarity of the premises with positive and negative examples, a measure to estimate the latent space separation, and therefore, assess how dense the resulting vector space is.

From the results, we can derive the following main observations: (1) The separation tends to be significantly lower in the cross-operational set, confirming that the latent space requires a more fine-grained optimisation in localised regions (Fig. 4); (2) Cross-operational inference is not achievable without operation-specific transformations, as confirmed by the impossibility to discriminate between positive and negative examples in the original expression encoders $( \mathrm { i } . \mathrm { e } . , \delta ( \mathbf { e } _ { x } , \mathbf { e } _ { y } )$ , Table 2); (3) The projection mechanism achieves intraoperational separation in the original expression encoders. This is not true for the translation mechanism in which the transformation induced by the operation encoder is fundamental for the separation to appear; (4) The latent space resulting from the translation model is more dense, with values for the separation that are generally lower when compared to the projection mechanism.

![](images/dece0e19eaa927a3882a125008e277f51c81d5df7e77040ff763493174e1b47a.jpg)  
(a) Projection

![](images/a85285bc442bf8a9286fc6ffc3413c804ef7084f130f88ec367612fbbbedf029.jpg)  
(b) Translation

Figure 5: Multi-step derivations in latent space with different multi-operational paradigms and neural encoders.  
![](images/b17ad49387563d58faff338e16ec36ff231b92fdcbb6f6b7529e3ff76a071861.jpg)  
(a) Cross-operational

![](images/189be4f281ea83e02cfa1ee381d22b99ecd64ce691730f60a68cca9c29758d40.jpg)  
(b) Intra-operational  
Figure 6: Length generalisation experiments by training different encoders (i.e., with translation) on premises with 2 variables, and testing on longer premises (MAP score).

These results, combined with the performance in Table 1, confirm that the translation paradigm can result in a more fine-grained and smoother optimisation which supports performance on crossoperational inference and a more balanced integration between expression and operation encoders.

## 4.5 Multi-Step Inference

We investigate the behaviour of different encoders and representational paradigms when propagating latent operations for multiple steps. To experiment, we employ the architectures recursively by interpreting the predicted target embedding $\mathbf { e } _ { y } ^ { \prime }$ as a premise representation for the next step (see Fig. 2). In this case, we evaluate the performance using Hit@1, selecting 1 positive example and 4 negative examples for each premise and derivation step (2 for cross-operational and 2 for intra-operational).

Figure 5 shows the obtained results. We found that the majority of the models exhibit a latent organisation that allows for a non-random propagation of latent mathematical operations. Most of the encoders, in fact, achieve performances that are significantly above random performance after 6 latent derivation steps (with a peak of 30% improvement for LSTM + translation). Moreover, while all the models tend to decrease in performance with an increasing number of inference steps, we observe significant differences between paradigms and classes of encoders. Most notably, we found that the performance of graph-based encoders tends to decrease faster, while the sequential models can obtain more stable results, in particular with the translation paradigm. The best translation model (i.e., LSTM) achieves a Hit@1 score at 6 steps of up to 50%, that is 15% above the best projection architecture (i.e., CNN).

## 4.6 Length Generalisation

Finally, we perform experiments to test the ability of expression encoders to generalise to out-ofdistribution examples. In particular, we focus on length generalisation which constitutes a notoriously hard problem for neural networks (Shen et al., 2021; Hupkes et al., 2020; Geirhos et al., 2020). To this end, we train the models on the subset of the training set containing premises with 2 variables and assess performance on longer premises (i.e., grouping the test set according to the number of variables). Figure 6 shows the results for different encoders using the translation mechanism.

Overall, the results show a decrease in performance as expected, demonstrating, at the same time, a notable difference between encoders on cross-operational inference. In particular, the results suggest that graph-based models can generalise significantly better on longer premises, probably due to their ability to capture explicit hierarchical dependencies within the expressions. Among the sequential models, CNNs achieve better generalisation performance. We attribute these results to the convolution operation in CNNs which may help capture structural invariances within the expressions and allow a generalisation that is similar to GCNs.

## 4.7 Discussion

From the empirical evaluation, we can derive a set of takeaways for both the joint-embedding architectures and the specific expression encoders.

Regarding the architectures, our analysis suggests that the translational paradigm can result in a more fine-grained and smoother optimisation of the latent space (Figure 3 and Table 2). This has the effect of improving multi-operational inference enabling a more balanced integration of different expression encoders, with an overall better tradeoff between cross-operational and intra-operational inference (Table 1). Moreover, we found that the translational paradigm can support better generalisation on multi-step inference when instantiated with sequential encoders such as Transformers, CNNs, and LSTMs (Figure 5), even when the encoders are only trained on single-step derivations.

Regarding the specific encoders, we conclude that different models have different characteristics that should inform practitioners and future research in the field. Sequential models (i.e., Transformers, CNNs, and LSTMs), possess a better ability to organise the latent space for enabling latent multi-step derivations (Figure 5). Conversely, graph-based models are more efficient (i.e., they achieve better performance using smaller operation encoders, see one-hot in Table 1) and tend to generalise better to longer expressions when trained to simpler ones (see Figure 6).

## 5 Related Work

The quest to understand whether neural architectures can perform mathematical reasoning has led researchers to investigate several tasks and evaluation methods (Lu et al., 2023; Meadows and Freitas, 2023; Mishra et al., 2022a; Ferreira et al., 2022; Ferreira and Freitas, 2020; Welleck et al., 2021; Valentino et al., 2022; Mishra et al., 2022b; Petersen et al., 2023). In this work, we focused on equational reasoning, a particular instance of mathematical reasoning involving the manipulation of expressions through the systematic application of specialised operations (Welleck et al., 2022; Lample and Charton, 2019; Saxton et al., 2018). In particular, our work is inspired by previous attempts to approximate mathematical reasoning entirely in latent space (Lee et al., 2019). Differently from Lee et al. (2019), we investigate the joint approximation of multiple mathematical operations for expression derivation (Lee et al. (2019) explore exclusively the rewriting operation for theorem proving). Moreover, while Lee et al. (2019) focus on the evaluation of Graph Neural Networks (Paliwal et al., 2020)), we analyse the behaviour of a diverse set of representational paradigms and neural encoders. Our data generation methodology is inspired by recent work leveraging symbolic engines and algorithms to build systematic benchmarks for neural models (Meadows et al., 2023b,a; Chen et al., 2022; Saparov et al., 2023). However, to the best of our knowledge, we are the first to construct and release a synthetic dataset to investigate multi-step and multi-operational derivations in latent space.

## 6 Conclusion

This paper focused on equational reasoning for expression derivation to investigate the possibility of approximating and composing multiple mathematical operations in a single latent space. Specifically, we investigated different representational paradigms and encoding mechanisms, analysing the trade-off between encoding different mathematical operators and specialising within single operations, as well as the ability to support multistep derivations and out-of-distribution generalisation. Moreover, we constructed and released a large-scale dataset comprising 1.7M derivation steps stemming from 61K premises and 6 operators, which we hope will encourage researchers to explore future work in the field.

## 7 Limitations

The systematic application of mathematical operators requires reasoning at an intentional level, that is, the execution and composition of mathematical functions defined on a potentially infinite set of elements. Neural networks, on the contrary, operate at an extensional level and, by their current nature, can only approximate such functions by learning from a finite set of examples.

Due to this characteristic, this work explored architectures that are trained on expressions composed of a predefined number and set of variables (i.e., between 2 and 5) and operators (i.e., addition, subtraction, multiplication, division, integration, differentiation), and, therefore, capable of performing approximation over a finite vocabulary of symbols. Extending the architectures with a new set of operations and out-of-vocabulary symbols, therefore, would require re-training the models from scratch. Future work could investigate this limitation by exploring, for instance, transfer learning techniques and more flexible neural architectures.

For the same reason, we restricted our investigation to the encoding of atomic operations, that is, operations in which the second operand is represented by a variable. While this limitation is circumvented by the sequential application of operators in a multi-step fashion, this work did not explore the encoding of single-step operations involving more complex operands (e.g., multiplication between two expressions composed of multiple variables each). In principle, however, the evaluation presented in this work can be extended with the new synthetic data to accommodate and study different cases and setups in the future.

## Acknowledgements

This work was partially funded by the Swiss National Science Foundation (SNSF) project Neu-Math (200021\_204617), by the EPSRC grant EP/T026995/1 entitled “EnnCore: End-to-End Conceptual Guarding of Neural Architectures” under Security for all in an AI enabled society, by the CRUK National Biomarker Centre, and supported by the Manchester Experimental Cancer Medicine Centre.

## References

Ivana Balazevic, Carl Allen, and Timothy Hospedales. 2019. Multi-relational poincaré graph embeddings.

Advances in Neural Information Processing Systems, 32.

Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multirelational data. Advances in neural information processing systems, 26.

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. 2022. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588.

Deborah Ferreira and André Freitas. 2020. Premise selection in natural language mathematical texts. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7365– 7374.

Deborah Ferreira, Mokanarangan Thayaparan, Marco Valentino, Julia Rozanova, and Andre Freitas. 2022. To be or not to be an integer? encoding variables for mathematical text. In Findings of the Association for Computational Linguistics: ACL 2022, pages 938– 948, Dublin, Ireland. Association for Computational Linguistics.

Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. 2020. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673.

Will Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive representation learning on large graphs. Advances in neural information processing systems, 30.

Matthew Henderson, Rami Al-Rfou, Brian Strope, Yun-Hsuan Sung, László Lukács, Ruiqi Guo, Sanjiv Kumar, Balint Miklos, and Ray Kurzweil. 2017. Efficient natural language response suggestion for smart reply. arXiv preprint arXiv:1705.00652.

Sepp Hochreiter and Jürgen Schmidhuber. 1996. Lstm can solve hard long time lag problems. Advances in neural information processing systems, 9.

Dieuwke Hupkes, Verna Dankers, Mathijs Mul, and Elia Bruni. 2020. Compositionality decomposed: How do neural networks generalise? Journal ofArtificial Intelligence Research, 67:757–795.

Yoon Kim. 2014. Convolutional neural networks for sentence classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1746–1751.

Thomas N Kipf and Max Welling. 2016. Semisupervised classification with graph convolutional networks. In International Conference on Learning Representations.

Guillaume Lample and François Charton. 2019. Deep learning for symbolic mathematics. arXiv preprint arXiv:1912.01412.

Yann LeCun. 2022. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62.

Dennis Lee, Christian Szegedy, Markus Rabe, Sarah Loos, and Kshitij Bansal. 2019. Mathematical reasoning in latent space. In International Conference on Learning Representations.

Zewen Li, Fan Liu, Wenjie Yang, Shouheng Peng, and Jun Zhou. 2021. A survey of convolutional neural networks: analysis, applications, and prospects. IEEE transactions on neural networks and learning systems.

Pan Lu, Liang Qiu, Wenhao Yu, Sean Welleck, and Kai-Wei Chang. 2023. A survey of deep learning for mathematical reasoning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14605– 14631, Toronto, Canada. Association for Computational Linguistics.

Jordan Meadows and André Freitas. 2023. Introduction to mathematical language processing: Informal proofs, word problems, and supporting tasks. Transactions ofthe Associationfor Computational Linguistics, 11:1162–1184.

Jordan Meadows, Marco Valentino, and Andre Freitas. 2023a. Generating mathematical derivations with large language models. arXiv preprint arXiv:2307.09998.

Jordan Meadows, Marco Valentino, Damien Teney, and Andre Freitas. 2023b. A symbolic framework for systematic evaluation of mathematical reasoning with transformers. arXiv preprint arXiv:2305.12563.

Aaron Meurer, Christopher P Smith, Mateusz Paprocki, Ondˇrej Certík, Sergey B Kirpichev, Matthew Rocklin,<sup>ˇ</sup> AMiT Kumar, Sergiu Ivanov, Jason K Moore, Sartaj Singh, et al. 2017. Sympy: symbolic computing in python. PeerJ Computer Science, 3:e103.

Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.

Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, and Ashwin Kalyan. 2022a. LILA: A unified benchmark for mathematical reasoning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5807–5832, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Swaroop Mishra, Arindam Mitra, Neeraj Varshney, Bhavdeep Sachdeva, Peter Clark, Chitta Baral, and Ashwin Kalyan. 2022b. Numglue: A suite of fundamental yet challenging mathematical reasoning tasks. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3505–3523.

Aditya Paliwal, Sarah Loos, Markus Rabe, Kshitij Bansal, and Christian Szegedy. 2020. Graph representations for higher-order logic and theorem proving. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 34, pages 2967–2974.

Felix Petersen, Moritz Schubotz, Andre Greiner-Petter, and Bela Gipp. 2023. Neural machine translation for mathematical formulae. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11534– 11550, Toronto, Canada. Association for Computational Linguistics.

Abulhair Saparov, Richard Yuanzhe Pang, Vishakh Padmakumar, Nitish Joshi, Seyed Mehran Kazemi, Najoung Kim, and He He. 2023. Testing the general deductive reasoning capacity of large language models using ood examples. arXiv preprint arXiv:2305.15269.

David Saxton, Edward Grefenstette, Felix Hill, and Pushmeet Kohli. 2018. Analysing mathematical reasoning abilities of neural models. In International Conference on Learning Representations.

Zheyan Shen, Jiashuo Liu, Yue He, Xingxuan Zhang, Renzhe Xu, Han Yu, and Peng Cui. 2021. Towards out-of-distribution generalization: A survey. arXiv preprint arXiv:2108.13624.

Marco Valentino, Danilo S Carvalho, and André Freitas. 2023. Multi-relational hyperbolic word embeddings from natural language definitions. arXiv preprint arXiv:2305.07303.

Marco Valentino, Deborah Ferreira, Mokanarangan Thayaparan, André Freitas, and Dmitry Ustalov. 2022. TextGraphs 2022 shared task on natural language premise selection. In Proceedings of TextGraphs-16: Graph-based Methods for Natural Language Processing, pages 105–113, Gyeongju, Republic of Korea. Association for Computational Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Sean Welleck, Jiacheng Liu, Ronan Le Bras, Hannaneh Hajishirzi, Yejin Choi, and Kyunghyun Cho. 2021. Naturalproofs: Mathematical theorem proving in natural language. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1).

Sean Welleck, Peter West, Jize Cao, and Yejin Choi. 2022. Symbolic brittleness in sequence models: on systematic generalization in symbolic mathematics. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 36, pages 8629–8637.

Yong Yu, Xiaosheng Si, Changhua Hu, and Jianxun Zhang. 2019. A review of recurrent neural networks:

Algorithm 1 Premise Generation   
1: $\mathcal { F } $ premise.free\_symbols   
2: $\begin{array} { r } { p  \frac { 1 } { p r } - 1 } \end{array}$   
3: for s in do   
4: m random.choice $( [ 0 ] * p + [ 1 ] )$   
5: if $m = 0$ then   
6: $s ^ { \prime }  s$   
7: else   
8: $p _ { c } \gets \frac { 1 } { p _ { e } } - 1$   
9: m  random.choice([0]  p<sub>c</sub> + [1])   
10: $c $ random.choice( 2, ..., 9 )   
11: if $m = 0$ then   
12: if random.choice({0,1}) = 0 then   
13: $s ^ { \prime }  s \times c$   
14: else   
15: $\textstyle s ^ { \prime }  { \frac { s } { c } }$   
16: end if   
17: else   
18: c random.choice( 2, ..., 9 )   
19: $s ^ { \prime }  s ^ { c }$   
20: end if   
21: end if   
22: premise  premise.subs $( s , s ^ { \prime } )$   
23: end for   
24: return premise

Lstm cells and network architectures. Neural computation, 31(7):1235–1270.

## A Data Generation

Algorithm 1 formalises the general data generation methodology adopted for generating premises with the SymPy<sup>7</sup> engine.

The following is an example of an entry in the dataset with both LaTex and Sympy surface form for representing expressions, considering integration and a single variable operand r. The same overall structure is adopted for the remaining operations and a larger vocabulary of variables:

• Premise:   
– Latex:   
u + cos(log( x + o))   
– SymPy:   
Add(Symbol(′u′),   
<sup>cos(log(Add(Mul(Integer(</sup>−<sup>1),</sup>

Symbol(′x′)),   
Symbol(′o′)))))   
• Derivation (integration, r):   
– Latex:   
ur + rcos(log( x + o))   
– SymPy:   
Add(Mul(Symbol(′u′), Symbol(′r′)),   
Mul(Symbol(′r′), cos(log(   
<sup>Add(Mul(Integer(</sup>−<sup>1),</sup>   
Symbol(′x′)), Symbol(′o′))))))

## B Dev Results

Table 3 reports the complete results on the dev set for different models and architectures with different embedding sizes.

<table><tr><td></td><td>|MAP</td><td>Hit@1</td><td>Hit@3</td><td>MAP</td><td>Hit@1</td><td>Hit@3</td><td>一  $\mathbf { A v g M A P }$ </td><td>1 Embeddings Dim.</td><td>Model Size (MB)</td></tr><tr><td>Projection (One-hot)</td><td></td><td>Cross-op.</td><td>一</td><td></td><td>Intra-op.</td><td></td><td>1</td><td></td><td></td></tr><tr><td>GCN</td><td>71.37</td><td>74.86</td><td>86.46</td><td>92.80</td><td>97.83</td><td>99.03</td><td>82.20</td><td>300</td><td>3.0</td></tr><tr><td rowspan="5">GraphSAGE</td><td>73.44</td><td>78.73</td><td>89.50</td><td>92.72</td><td>97.70</td><td>99.86</td><td>83.08</td><td>512</td><td>7.9</td></tr><tr><td>74.12</td><td>78.40</td><td>89.66</td><td>92.68</td><td>97.46</td><td>99.13</td><td>83.40</td><td>768</td><td>18.0</td></tr><tr><td>79.54</td><td>82.60</td><td>92.83</td><td>91.98</td><td>96.43</td><td>98.63</td><td>85.76</td><td>300</td><td>5.0</td></tr><tr><td>81.57</td><td>84.90</td><td>95.00</td><td>92.81</td><td>97.50</td><td>99.20</td><td>87.19</td><td>512</td><td>14.0</td></tr><tr><td>83.70</td><td>88.56</td><td>96.73</td><td>93.00</td><td>97.70</td><td>99.30</td><td>88.35</td><td>768</td><td>31.0</td></tr><tr><td>CNN</td><td>69.84</td><td>77.53</td><td>95.00</td><td>91.79</td><td>96.10</td><td>98.30</td><td>80.81</td><td>300</td><td>2.5</td></tr><tr><td rowspan="5"></td><td>66.94</td><td>74.46</td><td>93.40</td><td>92.06</td><td>96.80</td><td>98.40</td><td>79.50</td><td>512</td><td>4.6</td></tr><tr><td>67.25</td><td>74.70</td><td>93.63</td><td>91.48</td><td>95.46</td><td>97.73</td><td>79.37</td><td>768</td><td>7.6</td></tr><tr><td>69.31</td><td>72.06</td><td>89.00</td><td>92.93</td><td>97.96</td><td>99.46</td><td>81.12</td><td>300</td><td>6.6</td></tr><tr><td>69.33</td><td>71.66</td><td>88.16</td><td>92.92</td><td>97.90</td><td>99.46</td><td>81.13</td><td>512</td><td>19.0</td></tr><tr><td>70.84</td><td>72.60</td><td>90.10</td><td>92.89</td><td>97.76</td><td>99.30</td><td>81.86</td><td>768</td><td></td></tr><tr><td rowspan="5">Transformer</td><td>48.61</td><td>48.20</td><td>63.70</td><td>91.79</td><td>96.60</td><td>99.43</td><td>70.20</td><td>300</td><td>42.0 38.0</td></tr><tr><td>46.29</td><td>43.70</td><td>61.30</td><td>91.72</td><td>96.46</td><td>99.16</td><td>69.01</td><td>512</td><td></td></tr><tr><td></td><td>43.56</td><td>62.43</td><td>91.98</td><td>96.90</td><td>99.20</td><td>69.08</td><td>768</td><td>75.0</td></tr><tr><td>46.17</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>130.0</td></tr><tr><td colspan="3">Cross-op.</td><td colspan="4">Intra-op.</td><td></td><td></td></tr><tr><td>GCN</td><td>77.37</td><td>82.16</td><td>93.63</td><td>91.28</td><td>96.43</td><td>98.93</td><td>84.33</td><td>300</td><td>3.3</td></tr><tr><td rowspan="5">GraphSAGE</td><td>77.89</td><td>83.46</td><td>92.70</td><td>92.45</td><td>97.33</td><td>98.90</td><td>85.17</td><td>512</td><td>8.9</td></tr><tr><td>79.93</td><td>85.63</td><td>94.10</td><td>92.09</td><td>96.93</td><td>99.00</td><td>86.01</td><td>768</td><td>20.0</td></tr><tr><td>81.39</td><td>84.83</td><td>95.76</td><td>91.08</td><td>95.80</td><td>98.60</td><td>86.24</td><td>300</td><td>5.4</td></tr><tr><td>80.73</td><td>84.06 83.93</td><td>94.20</td><td>92.36</td><td>97.10</td><td>98.86</td><td>86.54</td><td>512</td><td>15.0</td></tr><tr><td>81.09</td><td></td><td>94.36</td><td>92.40</td><td>97.40</td><td>99.10</td><td>86.75</td><td>768</td><td>33.0</td></tr><tr><td rowspan="2">CNN</td><td>81.91</td><td>90.46</td><td>97.80</td><td>91.67</td><td>95.76</td><td>98.23</td><td>86.79</td><td>300</td><td>2.8</td></tr><tr><td>82.70</td><td>92.10</td><td>98.73</td><td>91.89</td><td>95.93</td><td>98.33</td><td>87.30</td><td>512</td><td>5.6</td></tr><tr><td rowspan="5">LSTM</td><td>81.70</td><td>90.73</td><td>97.80</td><td>92.36</td><td>97.20</td><td>98.90</td><td>87.03</td><td>768</td><td>9.9</td></tr><tr><td>71.96</td><td>74.93</td><td>89.03</td><td>92.26</td><td>96.93</td><td>99.26</td><td>82.11</td><td>300</td><td>7.0</td></tr><tr><td>76.13</td><td>80.50</td><td>93.53</td><td>92.77</td><td>97.70</td><td>99.30</td><td>84.45</td><td>512</td><td>20.0</td></tr><tr><td>76.40</td><td>80.03</td><td>93.23</td><td>93.13</td><td>98.06</td><td>99.60</td><td>84.76</td><td>768</td><td>44.0</td></tr><tr><td>70.06 69.59</td><td>75.50</td><td>88.70</td><td>91.96</td><td>97.40</td><td>99.53</td><td>81.01</td><td>300</td><td>38.0</td></tr><tr><td></td><td>73.63</td><td>87.20</td><td>92.20</td><td>97.70</td><td>99.53</td><td>80.89</td><td>512</td><td>76.0</td></tr><tr><td>一</td><td>52.43</td><td>51.16</td><td>68.30</td><td>90.78</td><td>96.16</td><td>99.33</td><td>71.60</td><td>768</td><td>133.0</td></tr><tr><td>Translation</td><td colspan="3"></td><td colspan="3">Intra-op.</td><td colspan="2">1</td><td></td></tr><tr><td rowspan="5">GCN</td><td>80.16</td><td>89.50</td><td>96.63</td><td>83.90</td><td>86.83</td><td>94.16</td><td>82.03</td><td>300</td><td>3.0</td></tr><tr><td>86.56</td><td>95.03</td><td>99.03</td><td>86.20</td><td>89.16</td><td>96.16</td><td>86.38</td><td>512</td><td>7.9</td></tr><tr><td>86.72</td><td>95.53</td><td>99.30</td><td>87.85</td><td>91.40</td><td>96.50</td><td>87.29</td><td>768</td><td>18.0</td></tr><tr><td>84.94</td><td>92.93</td><td>98.10</td><td>84.13</td><td>86.00</td><td>93.40</td><td>84.53</td><td>300</td><td>5.0</td></tr><tr><td>87.44 88.39</td><td>95.26 96.13</td><td>99.00</td><td>88.70 90.35</td><td>92.76</td><td>96.63</td><td>88.07</td><td>512</td><td>14.0</td></tr><tr><td rowspan="2">CNN</td><td></td><td></td><td>99.16</td><td></td><td>94.20</td><td>97.86</td><td>89.37</td><td>768</td><td>31.0</td></tr><tr><td>84.42 84.62</td><td>95.30 95.33</td></table>

Table 3: Overall performance of different neural encoders and methods (dev set) for jointly encoding multiple mathematical operations (i.e., integration, differentiation, addition, difference, multiplication, division).