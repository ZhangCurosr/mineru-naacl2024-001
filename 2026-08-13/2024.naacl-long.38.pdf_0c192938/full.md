# Language Agnostic Code Embeddings

Saiteja Utpala Cohere For AI saitejautpala@gmail.com

Alex Gu   
MIT   
gua@mit.edu

Pin Yu ChenIBM Researchpin-yu.chen@ibm.com

## Abstract

Recently, code language models have achieved notable advancements in addressing a diverse array of essential code comprehension and generation tasks. Yet, the field lacks a comprehensive deep dive and understanding of the code embeddings of multilingual code models. In this paper, we present a comprehensive study on multilingual code embeddings, focusing on the cross-lingual capabilities of these embeddings across different programming languages. Through probing experiments, we demonstrate that code embeddings comprise two distinct components: one deeply tied to the nuances and syntax of a specific language, and the other remaining agnostic to these details, primarily focusing on semantics. Further, we show that when we isolate and eliminate this languagespecific component, we witness significant improvements in downstream code retrieval tasks, leading to an absolute increase of up to +17 in the Mean Reciprocal Rank (MRR).

## 1 Introduction

Large language models (LLMs) have made remarkable progress in code-related tasks, exemplified by models such as Codex, which powers GitHub Copilot and offers automated code suggestions within integrated development environments (IDEs) (Chen et al., 2021). These models achieve their proficiency through extensive training on vast code datasets, providing them with versatile contextual understanding for a range of coding tasks (Husain et al., 2019; Athiwaratkun et al., 2022; Zhu et al., 2022). However, it's worth noting that decoder-only models may not always be the optimal choice for retrieval tasks when compared to encoder models (Nijkamp et al., 2023; Wang et al., 2021b, 2023).

While previous studies indicate that language models trained on a variety of natural languages exhibit strong cross-lingual traits (Pires et al., 2019), their multilingual representations can be dissected into a language-specific syntax component and a language-agnostic semantic component (Chang et al., 2022). Moreover, eliminating the languagespecific elements can enhance retrieval tasks and counteract "language bias", a tendency for representations to cluster by language instead of meaning (Roy et al., 2020; Yang et al., 2021; Xie et al., 2022). We aim to determine if similar patterns are evident in multilingual code models pretrained on programming languages (e.g., C, C++, Python) as opposed to natural languages (e.g., English, French, Spanish). We specifically address:

1. Can representations of these code models be categorized into language-specific and language-agnostic components?

2. If so, does removing the language-specific components enhance the consistency and comparability of code representations (alignment) across programming languages, thereby improving downstream code retrieval tasks?

Our comprehensive evaluations confirm these patterns in code language models. Our contributions are summarized as follows:

• We investigate the cross-lingual properties of pretrained multilingual code language models, examining them through the lens of both language-specific (syntax) and languageagnostic (semantic) attributes. Through various probing experiments on five models, we demonstrate that the embeddings of these code language models include both languagespecific (syntax) and language-agnostic (semantic) components.

• We demonstrate that removing these languagespecific components and using only the language-agnostic component in downstream tasks can significantly enhance code retrieval tasks providing improvement upto +17 increase in MRR. Importantly, such improvements are achieved using inexpensive operations such as centering and projections, without using parallel language data or finetuning.

![](images/2b98c378a12a8b8de6e67d3c3c9e1c19a4e2e5ef281062ec172a885a9474baa4.jpg)  
Figure 1: Illustration of the top retrieved results for code-to-code search, where the query is in Python, and the target is in C. Language composition varies across retrieval databases: 'Monolingual' (C only), 'Source Excluded Multilingual'(several languages except Python), 'Source Included Multilingual’(several languages including Python). Demonstrates improved semantic matching and reduced language bias after removing language-specific components.

• Additionally, our extensive ablation studies suggest that as few as 100 samples per language suffice for these MRR improvements. We also confirm that the improvements are not restricted to a single type of embedding but can be realized across all common types, including mean, cls, and pooler embeddings.

## 2 Language Agnostic Code Embeddings

Let M represent a multilingual code language model trained on a set of programming languages $\{ 1 , \ldots , \ell \}$ . Given a code snippet c in a specific programming language l, this model produces an embedding $\mathbf { e } \in \mathbb { R } ^ { d }$ , denoted as $\mathcal { M } ( c ) = \mathbf { e } \in \mathbb { R } ^ { d }$ . We hypothesize that the embedding $\mathbf { e } \in \mathbb { R } ^ { d }$ of a code snippet can be decomposed into two components: a syntax component, $\mathbf { e } ^ { \mathrm { s } } \in \mathbb { R } ^ { d }$ , which depends on the programming language l, and semantic component, ${ \bf e } ^ { \mathrm { a } }$ which is language-agnostic. This relationship can be expressed as:

$$
\mathbf { e } = \mathbf { e } ^ { \mathrm { s } } + \mathbf { e } ^ { \mathrm { a } }\tag{1}
$$

Next, we introduce the Estimation Set $\mathcal { E } _ { : }$ which is used to estimate the language-specific components $\mathbf { e } ^ { s }$

Definition 1 (Estimation Set). The Estimation set E is defined as a collection of n code snippets $\{ c _ { 1 } ^ { ( l ) } , \ldots , c _ { n } ^ { ( l ) } \}$ from each programming language $l \in \{ 1 , \ldots , \ell \}$ . Importantly, the code snippets in this set need not be direct translations of one another.

Now for given model $\mathcal { M } ,$ we define embedding matrix $\mathbf { E } _ { l } \in \mathbb { R } ^ { n \times d }$ for each language l as $\mathbf { E } _ { l } ~ =$ $\left\lceil \mathcal { M } ( c _ { 1 } ^ { ( l ) } ) , \ldots , \mathcal { M } ( c _ { n } ^ { ( l ) } ) \right\rceil$

In the subsequent sections, we explore a variety of methods designed to remove language-specific information. This analysis is conducted from the unified perspective of Equation 1, which serves as the fundamental framework for disentangling language-specific and language-agnostic components within code embeddings. Additionally, we explicitly outline the assumptions that underpin each of these methods.

## 2.1 Centering

The first method we explore is centering (Libovickò et al., 2020), which is grounded in the following key assumption:

Assumption 1. Given an programming language l, centering method makes an assumption that language specific components $e ^ { s }$ is same for all embeddings in that programming language l.

Under Assumption 1, the mean of the embeddings for a programming language l can be expressed as:

$$
\begin{array} { r l } { \displaystyle \mathbf { m } _ { l } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { e } _ { i } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( \mathbf { e } _ { i } ^ { s } + \mathbf { e } _ { i } ^ { a } \right) } & { } \\ { \displaystyle \stackrel { ( 1 ) } { = } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( \mathbf { e } ^ { s } + \mathbf { e } _ { i } ^ { a } \right) = \mathbf { e } ^ { s } + } & { \underbrace { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { e } _ { i } ^ { a } } _ { \mathrm { s m a l l ~ f o r ~ l a r g e } } \approx \mathbf { e } ^ { s } . } \end{array}
$$

From the above, it is evident that for a large enough value of $n ,$ the language-specific syntax embedding for a given programming language can be approximately estimated as the mean of the embeddings in that language. This method is summarized in Algorithm 1 in centering.

## 2.2 Low Rank Decomposition

A significant concern with the centering method, as outlined in Assumption 1, is its presumption that the syntax embedding for each code is the same and is independent of the given code content. Instead it is only dependent on the programming language. To address this limitation, Low Rank Decomposition (LRD) (Schmidt, 1907; Yang et al., 2021) introduces distinct syntax subspaces for each programming language, operating under the following assumptions:

Assumption 2. The low rank decomposition method is based on the following assumptions:

1. The language-specific syntax embedding varies for each individual embedding.

2. The language-specific syntax embedding, $e ^ { \mathrm { s } } { } _ { : }$ is orthogonal to the language-agnostic semantic embedding, $e ^ { \mathrm { a } } , i . e . , e ^ { \mathrm { s } } \perp e ^ { \mathrm { a } }$

3. For each programming language, there exists a low-rank subspace of rank r that captures the syntactic essence of the embeddings.

Based on the above assumptions, we determine the syntactic subspace of language l of rank r, denoted as $\mathbf { E } _ { l } ^ { r } \in \mathbb { R } ^ { n \times d }$ , as follows:

$$
\begin{array} { r l } & { \underset { \mathbf { E } _ { l } ^ { r } \in \mathbb { R } ^ { n \times d } } { \operatorname* { m i n } } \left. \mathbf { E } _ { l } - \mathbf { E } _ { l } ^ { r } \right. _ { \mathrm { F } } ^ { 2 } } \\ & { } \\ & { \qquad \mathrm { s . t . R A N K } ( \mathbf { E } _ { l } ^ { r } ) \leq r . } \end{array}\tag{2}
$$

Equation 2 can be solved using TOPK-SVD with $\textit { k } = \textit { r }$ where $\mathbf { E } _ { l } ^ { r } = \mathbf { U } _ { r } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { E } _ { r } ^ { T }$ . The projection of the embedding e onto the ROWSPACE $( \mathbf { E } _ { l } ^ { r } ) \mathrm { i s }$ given by $\mathbf { V } _ { r } \mathbf { V } _ { r } ^ { T } \mathbf { e }$ . The language-agnostic embedding is then obtained by removing this component: $\mathbf { e } ^ { a } = \mathbf { e } - \mathbf { V } _ { r } \mathbf { V } _ { r } ^ { T } \mathbf { e }$ This method is summarized in Algorithm 1 in LRD.

## 2.3 Common Specific Low Rank Decomposition

The Common Specific Low Rank Decomposition (Piratla et al., 2020; Xie et al., 2022) is a variant of low rank decomposition. Given different data domains, this method aims to learn both a common subspace shared across domains and a specific subspace unique to each domain.

Assumption 3. The Common Specific Low Rank Decomposition is grounded on the following assumptions:

1. The language-specific syntax embedding varies for each individual embedding.

2. The language-specific syntax embedding, $e ^ { \mathrm { s } } ,$ is orthogonal to the language-agnostic semantic embedding, $e ^ { \mathrm { a } } .$ In other words, $e ^ { \mathrm { s } } \perp e ^ { \mathrm { a } }$

3. There exists a unified syntax subspace, consistent across all programming languages, that encapsulates the syntactic attributes of the code embedding.

A key distinction is that while the syntax subspace in the traditional LRD is determined for each language individually, the CS-LRD method derives a singular, unified syntax subspace that encompasses all the considered programming languages. It is formulated as:

$$
\begin{array} { r l } & { \underset { \mathbf { m } _ { c } \in \mathbb { R } ^ { d } , \mathbf { M } _ { s } \in \mathbb { R } ^ { d \times r } , \mathbf { T } _ { s } \in \mathbb { R } ^ { d \times \ell } } { \operatorname* { m i n } } \left\| \mathbf { M } - \mathbf { m } _ { c } \cdot \boldsymbol { \mathbb { 1 } } _ { \ell } ^ { T } - \mathbf { M } _ { s } \cdot \mathbf { \boldsymbol { \Gamma } } _ { s } ^ { T } \right\| _ { \mathrm { F } } ^ { 2 } } \\ & { \qquad \mathrm { s . t . } \mathbf { m } _ { c } \perp \mathrm { C O L S P A N } ( \mathbf { M } _ { s } ) . } \end{array}\tag{3}
$$

where $\mathbf { M } = [ \mathbf { m } _ { 1 } , \dots , \mathbf { m } _ { \ell } ]$ , and $\mathbf { m } _ { 1 } , \ldots , \mathbf { m } _ { \ell }$ are the mean embeddings of the $\{ 1 , \ldots , \ell \}$ programming languages. The matrix $\mathbf { M } _ { s } ,$ , which captures the common syntactic subspace, can be obtained by the CS-LRD function in Algorithm 1. Similar to LRD, the language-agnostic embedding is obtained by removing the projection of e on M, i.e., $\mathbf { e } ^ { a } = \mathbf { e } - \mathbf { M } _ { s } \mathbf { M } _ { s } ^ { T } \mathbf { e }$

## 3 Experiments

Setup: We examine three tasks and analyze the performance before and after removing languagespecific components: (i) Probing - This task involves identifying languages using a linear classifier. (ii) Code2Code search - Given a piece of code in language $L _ { 1 }$ , the objective is to retrieve the most semantically relevant code in another language $L _ { 2 }$ (iii) Text2Code search - The aim is to identify code that corresponds to a provided natural language query.

![](images/75921e6f57d30e6f98644e6aa482ab122cec9fd8f7b19cb1bcc256bfc174b860.jpg)  
(a) CodeBERT

![](images/f75103ed2b7816d33d95b96ede7da5086ef8d1468d7904811801f1df9802c390.jpg)  
(b) GraphCodeBERT

![](images/e11814aa32431dc11cb3fc4c4e022944a52b504354708eff144a9d2f5e6ddd49.jpg)  
(c) UnixCoder

![](images/27b05c1e48bf8c7cbcdedaf7682f9272a84e081621e3f7414f4ffb50538bbc59.jpg)  
(d) StarEncoder

![](images/0eb6d779a93b035a895f9741dca1a0dc49f522f362303aa41ca526162d086919.jpg)  
(e) CodeT5+

![](images/3261e2bba4acaa25b0ed97cecd9ab38bb51c516189b3aa2628ff7b42d4dfdc15.jpg)  
(f) Original

![](images/37457ccec6bced4d80a33eca99909a6a253beb1158381433d922ae64554e0d2f.jpg)  
(g) Centering

![](images/dae7036096079721082d9f37f27d4991c3612e6edc99ba927f5e32533ecdf17a.jpg)  
(h) LRD

![](images/7f94f5f76b0eeead7a371c23b20601b066d2052359ee9d45fcf0fc668f482579.jpg)  
(i) CS-LRD  
Figure 2: The top row illustrates the impact on language identification accuracy before and after removing languagespecific components. Meanwhile, the bottom row displays the PCA of Code T5+ embeddings for the languages: Go, Python, and C#.

The first task assesses whether the procedures in Algorithm 1 effectively eliminate language-specific (syntax) components. The second and third tasks determine if language-agnostic (semantic) components are preserved.

Datasets: We utilize programs from the Stack dataset (Kocetkov et al., 2022) to estimate language-specific components. For the Code2Code search, we employ XLCoST (Zhu et al., 2022), and for the Text2Code search, we use CSN (Husain et al., 2019).

Models: We consider five models, including encoder-only and encoder-decoder models: Code-BERT (Feng et al., 2020), GraphCodeBERT (Guo et al., 2020), UnixCoder (Guo et al., 2022), StarEncoder (Li et al., 2023), and CodeT5+ (Wang et al., 2023).

Embeddings: For models like CodeBERT, Graph-CodeBERT, UnixCoder, and StarEncoder, there isn't a standard method to obtain embeddings. In our retrieval tasks, we use mean embeddings, derived from the mean of the last hidden states. We conduct an ablation study to explore other embedding extraction methods in Section 4.2. For CodeT5+, only the pooler embedding is recomended and is given as output, and this is what we employ in our experiments.

Retrieval Metrics: For the Retrieval Task, we use Mean Reciprocal Rank (MRR) as our evaluation metric. MRR is calculated as $\mathbf { M R R \ = }$ $\textstyle { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } { \frac { 1 } { \mathrm { r a n k } ( c _ { i } ) } } \times 1 0 0 \quad$ , where n represents the total number of queries, and rank(ci) denotes the rank of the correct answer for the i-th query in the retrieval results. Higher MRR values indicate better performance.

## 3.1 Probing

We evaluate the syntactic component of embeddings by employing a linear classifier for the task of language identification, both pre and post transformations. From the Stack dataset (Kocetkov et al., 2022), we allocate 10,000 code instances for estimating language components. For training, we use 24,000 code snippets from each language, and for validation, we utilize 6,000 codes from each respective language. The testing is performed on 10,000 codes for each language. The outcomes are depicted in Figure 2. Before transformation, the linear classifier yields high accuracy on the embeddings. However, after the removal of language-specific components, the accuracy declines sharply, experiencing a drop of at least 70% across all models. In particular, for the CodeT5+ model, the accuracy approaches random performance. Moreover, in the context of CS-LRD, there's an interesting relationship between the rank r and performance. As r increases, the classifier's performance diminishes. It's worth noting that this behavior is not observed with the LRD.

we also visualize PCA of CodeT5+ embeddings and show it in Figure 2f which shows embeddings are clustered by language. But after removing language-specific components we see that in Figure 2g to Figure 2i there are no longer language clusters.

## 3.2 Code2Code Search

Given a query Q in a source language S, the objective is to extract a code snippet with semantic similarity from a specific database. Depending on the language composition of the database, we consider three different variations:

• Monolingual Database: In this conventional setting, the database consists entirely of programs written in a single target language T, which is distinct from the source language S.

• Source-Excluded Multilingual Database: In this variation, the database is composed of programs in multiple languages $\mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { n } ,$ where Ti differs from the source language S.

• Source-Included Multilingual Database: This final variation includes the source language S within its spectrum of target languages. We evaluate the language bias of models (Yang et al., 2021), wherein a code from the source language S is ranked higher than codes that are more semantically similar but from different languages.

For this task, we use the XLCoST dataset (Zhu et al., 2022), which contains parallel translations of seven programming languages: C, C#, C++, Java, JavaScript, PHP, and Python. However, it's important to note that CodeBERT and GraphCodeBERT do not support C, C++, and C#. Therefore, we only consider Java, JavaScript, PHP, and Python for these models, while all seven languages are included for all other models.

We present the change in the Mean Reciprocal Rank (MRR) before and after the removal of the language component in Figure 3. This change is averaged over all pairwise language retrieval tasks, amounting to 6 × 7 = 42 tasks in total. Additionally, for the CodeT5+ model, we offer a detailed breakdown of the MRR for each source language. The retrieval results are averaged across the six target languages and are tabulated in Table 1.

Discussion: Significant improvements are observed before and after removing the language component, with an absolute increase in MRR ranging up to +17. We discuss a couple of factors below.

1. Database Configuration: Models exhibit substantial language bias, leading to a drastic drop in performance in the 'Source Included Multilingual’setup, with a reduction of -59.62% from 89.51 to 29.89.

2. Centering Effects: In three out of five cases, centering has a detrimental impact on performance. This aligns with the notion that centering may mix syntax and semantic signals, potentially removing semantic meaning as well (Yang et al., 2021; Xie et al., 2022).

3. UniXcoder Exception: Notably, UniXcoder explicitly aligns representations from different programming languages during pretraining itself using a task involving cross-modal generation (Guo et al., 2022). Consequently, none of the methods provide any improvement in this case.

4. CS-LRD Superiority: In most cases, CS-LRD outperforms both centering and LRD. This is attributed to the joint learning of the syntax subspace across different programming languages in CS-LRD.

5. Effect of Rank in LRD and CS-LRD: We examine the impact of the rank of the subspace r in Figure 5 for both LRD (Top Row) and CS-LRD (Bottom Row). Increasing r consistently enhances MRR in CS-LRD, while no such behavior is observed in LRD, which is less stable compared to CS-LRD.

## 3.3 Text2Code Search

In this section, we delve into Text2Code search, a task where the objective is to find code that corresponds to a given natural language query (in English). We explore two distinct settings for Text2Code search:

• Monolingual Database: In this setting, we construct the retrieval database using data from a single programming language.

• Multilingual Database: In contrast, for this setting, we include data from all programming languages in the retrieval database. The goal is to locate the correct code snippet that matches the query, regardless of the programming language.

For this task, we utilize the CodeSearchNet dataset (Husain et al., 2019), which contains data in six programming languages: Go, Ruby, Java, JavaScript, PHP, and Python. Retrieval database consists of codes in both val and test.

![](images/50c5311cda85d0c3cda139a896d825f146f9e3fc1e2a358e47f22dbfa6b7d42e.jpg)  
(a) CodeBERT

![](images/ea8d95dd88cf01457126b8fd0b0d8649bf87f1d496197309c174d25926f39fa1.jpg)  
(b) GraphCodeBERT

![](images/ce9c24213b67c9223ac1225a10a02e1a2e9668270531308d9eb7428d4339c597.jpg)  
(c) UnixCoder

![](images/c4b87cd403dd1bb5fb4a838cb589f791c41f381c0a5bc5fe7412fbf57cb720e2.jpg)  
(d) StarEncoder

![](images/65c44c443b5745eef1343709dd5c10f16caae630fea7fbea687ef08de1e1c252.jpg)  
(e) CodeT5+

Figure 3: Absolute change in MRR after removing language components in zero-shot Code2Code search.
<table><tr><td>CodeT5+</td><td></td><td>C</td><td>C#</td><td>C++</td><td>Java</td><td>Javascript</td><td>PHP</td><td>Python</td><td>Avg.</td></tr><tr><td rowspan="4">Monolingual</td><td>Original</td><td>86.19</td><td>88.12</td><td>90.93</td><td>89.95</td><td>90.63</td><td>89.46</td><td>91.32</td><td>89.51</td></tr><tr><td>Centering</td><td>87.59</td><td>91.10</td><td>93.25</td><td>92.23</td><td>91.70</td><td>90.97</td><td>92.62</td><td>91.35 (+1.84)</td></tr><tr><td>LRD(r=10)</td><td>88.87</td><td>91.77</td><td>94.85</td><td>93.20</td><td>91.84</td><td>91.85</td><td>92.90</td><td>92.18 (+2.67)</td></tr><tr><td>CS-LRD(r=9)</td><td>87.37</td><td>90.50</td><td>93.12</td><td>91.78</td><td>92.09</td><td>90.41</td><td>92.69</td><td>91.14 (+1.63)</td></tr><tr><td rowspan="4">Source Excluded Multilingual</td><td>Original</td><td>43.73</td><td>24.71</td><td>59.51</td><td>31.19</td><td>57.43</td><td>61.12</td><td>65.26</td><td>48.99</td></tr><tr><td>Centering</td><td>49.98</td><td>28.93</td><td>74.73</td><td>36.05</td><td>65.28</td><td>61.67</td><td>68.32</td><td>54.99 (+6.00)</td></tr><tr><td>LRD(r=10)</td><td>56.24</td><td>39.73</td><td>77.16</td><td>44.79</td><td>69.90</td><td>67.05</td><td>74.45</td><td>61.33 (+12.34)</td></tr><tr><td>CS-LRD(r=9)</td><td>57.04</td><td>31.13</td><td>76.89</td><td>37.75</td><td>66.99</td><td>65.10</td><td>72.03</td><td>58.13 (+9.14)</td></tr><tr><td rowspan="4">Source Included Multilingual</td><td>Original</td><td>34.35</td><td>16.94</td><td>17.69</td><td>23.99</td><td>32.76</td><td>38.09</td><td>45.43</td><td>29.89</td></tr><tr><td>Centering</td><td>37.03</td><td>16.28</td><td>33.23</td><td>21.89</td><td>40.14</td><td>40.97</td><td>52.81</td><td>34.62 (+4.73)</td></tr><tr><td>LRD(r=10)</td><td>45.88</td><td>11.37</td><td>41.55</td><td>23.47</td><td>40.37</td><td>39.96</td><td>54.63</td><td>36.75 (+6.86)</td></tr><tr><td>CS-LRD(r=9)</td><td>47.07</td><td>20.47</td><td>36.87</td><td>29.73</td><td>47.28</td><td>50.06</td><td>60.04</td><td>41.65 (+11.76)</td></tr></table>

Table 1: MRR averaged across all target languages for zero-shot Code2Code search using CodeT5+ (Wang et al., 2023).

We present the change in Mean Reciprocal Rank (MRR) before and after removing the language component in Figure 4. Full view for Unixcoder can be found at Table 2.

Discussion: Sizable improvements are observed before and after removing language component, with an absolute increase in MRR ranging upto +8. We discuss a couple of factors below.

1. CodeT5+ Exception: CodeT5+ includes contrastive tuning as one of its pretraining tasks (Wang et al., 2023) for text-to-code, which explicitly aligns English with programming languages. Hence, we don't observe any improvement.

2. Centering Superiority: Unlike in Code2Code search, in Text2Code search centering outperforms both LRD and CS-LRD.

3. Effect of Rank in LRD and CS-LRD: We study the influence of the subspace rank r as depicted in Figure 6. The top row illustrates the effect for LRD, while the bottom row represents CS-LRD. For both CS-LRD and LRD, increasing r either consistently improves MRR or remains stable. However, for CodeT5+, there is a consistent decrease.

4. Effect of Projecting out English: We conduct retrieval in two distinct ways. In the first method, we remove language components solely from programming languages, leaving the query unaffected (no English component is removed). In the second method, we transform the query by removing the English language component from it. The results are depicted in Figure 6. We find that projecting out the English language components is crucial to observe an increase in MRR.

## 4 Ablation Study

In this section, we conduct various ablation studies focusing on the estimation set's size and the effects of different kinds of embeddings.

## 4.1 Effect of Estimation Set Size

In this section, we investigate the impact of estimation set size on language estimation and its influence on Mean Reciprocal Rank (MRR) in zero-shot Code2Code search. We randomly sample {100, 500, 1000, 5000, 10000, 25000} examples from the original pool of 50,000 samples from the Stack dataset for each language used in Section 3.2. Subsequently, we conduct retrievals with the language components removed based on these samples. This study is repeated five times for each sample size, and we calculate the MRR change. The results can be seen in Figure 7. In this figure, the top row represents Centering, the middle row showcases LRD, and the bottom row depicts CS-LRD.

![](images/cb93285a99dbbc3a8db6f770a9f472eefd13bf510cb5038bdeeb15d1fdddd775.jpg)  
(a) CodeBERT

![](images/ba846cd84c4b6a5f8f6383ccb8b66681f3e45f847fde12673b6ac2b05083664d.jpg)  
(b) GraphCodeBERT

![](images/aa897cf30e5e8078e730ef452729f360f7d1b36f1d6bd8c71d0e87e3a77bfe23.jpg)  
(c) UnixCoder

![](images/202f22feba3ced2bd039a8766cf433e614989cc8b115cb6a0b38df09d48175b4.jpg)  
(d) StarEncoder

![](images/b165e7f6d499d1f5ad5f553cc581876d3cab1fa756fccb9456e8e12f6e44c45e.jpg)  
(e) CodeT5+

Figure 4: Absolute change in MRR after removing language components in zero-shot Text2Code search.
<table><tr><td>Unixcoder (mean)</td><td></td><td>Go</td><td>Java</td><td>Javascript</td><td>PHP</td><td>Python</td><td>Ruby</td><td>Avg.</td></tr><tr><td rowspan="4">Monolingual</td><td>Original</td><td>61.38</td><td>44.23</td><td>40.93</td><td>35.22</td><td>42.43</td><td>55.30</td><td>46.58</td></tr><tr><td>Centering</td><td>64.01</td><td>47.77</td><td>44.25</td><td>38.98</td><td>46.15</td><td>57.16</td><td>49.72 (+3.14)</td></tr><tr><td>LRD(r=10)</td><td>61.51</td><td>44.46</td><td>41.15</td><td>35.38</td><td>42.61</td><td>55.44</td><td>46.76 (+0.18)</td></tr><tr><td>CS-LRD(r=9)</td><td>63.10</td><td>47.62</td><td>43.94</td><td>38.61</td><td>45.98</td><td>56.79</td><td>49.34 (+2.76)</td></tr><tr><td rowspan="4">Multilingual</td><td>Original</td><td>54.05</td><td>36.40</td><td>27.87</td><td>29.84</td><td>35.71</td><td>34.83</td><td>36.45</td></tr><tr><td>Centering</td><td>54.65</td><td>40.14</td><td>30.42</td><td>33.21</td><td>40.20</td><td>38.11</td><td>39.46 (+3.01)</td></tr><tr><td>LRD(r=10)</td><td>54.18</td><td>36.62</td><td>27.96</td><td>30.03</td><td>35.85</td><td>34.91</td><td>36.59 (+0.14)</td></tr><tr><td>CS-LRD(r=9)</td><td>55.21</td><td>39.95</td><td>30.07</td><td>33.06</td><td>39.37</td><td>36.94</td><td>39.10 (+2.65)</td></tr></table>

Table 2: MRR for zero-shot Text2Code search using Unixcoder (Guo et al., 2022) .

The results highlight a significant change in MRR, even with estimation sets containing as few as 100 samples per language. However, some variance is observed in certain instances. This variance diminishes considerably once the estimation set expands to 1000 samples, resulting in a steadier MRR shift. Interestingly, the variance is typically greater for Centering and LRD compared to CS-LRD. This study also reveals that specific examples in the estimation set don't play as significant a role as the overall size of the estimation set.

## 4.2 Mean embedding vs [CLS] embedding vs Pooler output

In this section, we examine various kinds of embeddings and analyze the effects of removing language components from them for zero-shot code2code search. As noted in Sections 3.2 and 3.3, we utilized mean embeddings for CodeBERT, Graph-CodeBERT, UnixCoder, and StarEncoder. However, other embedding types are also commonly employed in practice.

To clarify, let c be a code snippet. The function encoder(c) produces the last hidden state with the shape Rt×d, where t denotes the number of tokens in the code. There are several methods to obtain a single Rd representation from the encoder's output.

These methods are defined as follows:

$$
\begin{array} { r l } & { \mathrm { m e a n - e m b e d d i n g } ( c ) \triangleq \mathtt { e n c o d e r } ( c ) . \mathrm { m e a n } ( 0 ) } \\ & { \mathrm { ~ \mathtt { c l s - e m b e d d i n g } } ( c ) \triangleq \mathtt { e n c o d e r } ( c ) [ 0 ] } \\ & { \mathrm { p o o l e r - e m b e d d i n g } ( c ) \triangleq \mathtt { p o o l e r } ( \mathtt { e n c o d e r } ( x ) [ 0 ] ) } \end{array}
$$

Here, pooler is an MLP layer positioned atop the encoder, and its output is directed to the language modeling head.

Results are displayed in Figure 8. We observe that the improvements aren't limited to meanembedding; they also extend to cls-embedding and pooler-embedding. Specifically, when using CS-LRD with CodeBERT's cls-embedding, there's a significant increase of +26.22. Similarly, StarEncoder's pooler embedding sees a +14.87 improvement with CS-LRD. Notably, mean-embedding remains superior to other embedding variants regardless of the presence or absence of language information.

## 5 Related Work

Cross Lingual properties of Natural Language models: We are the first to investigate the crosslingual properties of pretrained multilingual code language models, examining them through the lens of both language-specific (syntax) and languageagnostic (semantic) attributes. Our research is motivated by a rich body of work that probes similar behavior in multilingual natural language models (Schuster et al., 2019; Libovickò et al., 2020; Kulshreshtha et al., 2020; Yang et al., 2021; Xie et al., 2022; Chang et al., 2022). While these studies predominantly concentrate on models trained for natural languages, our emphasis lies on those designed for programming languages.

Code Representation Learning: The monumental success of BERT (Devlin et al., 2019) and T5 (Raffel et al., 2020) in natural language understanding has sparked significant interest in adapting similar architectures for programming languages. This interest has given rise to models like CodeBERT (Feng et al., 2020), CodeTransformer (Zügner et al., 2020), GraphCodeBERT (Guo et al., 2020), ContraCode (Jain et al., 2021), SynCoBERT (Wang et al., 2021a), UniXCoder (Guo et al., 2022), and PLBART (Ahmad et al., 2021). Some of these works (Zügner et al., 2020; Guo et al., 2020) explore code-specific pretraining tasks, utilizing both language-specific features (e.g., Program Analysis Edges) and language-agnostic features (e.g., Abstract Syntax Trees) to improve the performance of multilingual code models. In contrast, our work focuses on examining the representations of pretrained multilingual code language models.

LMs for Code Generation: In recent years, there have been many language models (LMs) for code trained with various architectures, sizes, and data mixtures, inspired by the huge success of GPT (Radford et al., 2019; Brown et al., 2020). Some of these include Codex (Chen et al., 2021), CodeGeeX (Zheng et al., 2023), SantaCoder (Allal et al., 2023), PolyCoder (Xu et al., 2022), CodeGen (Nijkamp et al., 2022), StarCoder (Li et al., 2023), Wizard-Coder (Luo et al., 2023), and Code Llama (Roziere et al., 2023). In this work, instead of focusing on code generation, we concentrate on code representations.

## 6 Discussion

In our study of multilingual code models, we find that these embeddings can be decomposed into two main components: language-specific and languageagnostic. Through extensive experimentation, we conclude that when representations are not aligned during pre-training, the removal of the languagespecific component, utilizing only the languageagnostic component, significantly enhances performance in retrieval tasks.

## 7 Limitations

In this study, we have focused on exploring representations of encoder-only or encoder-decoder models. However, future work should also investi-

gate decoder-only models.

## Acknowledgements

A. Gu is supported by the National Science Foundation (NSF) Graduate Research Fellowship under Grant No. 2141064. We sincerely thank Armando Solar-Lezama for comments and feedback on the project.

## References

Wasi Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. 2021. Unified pre-training for program understanding and generation. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2655–2668.

Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, et al. 2023. Santacoder: don't reach for the stars! arXiv preprint arXiv:2301.03988.

Ben Athiwaratkun, Sanjay Krishna Gouda, Zijian Wang, Xiaopeng Li, Yuchen Tian, Ming Tan, Wasi Uddin Ahmad, Shiqi Wang, Qing Sun, Mingyue Shang, et al. 2022. Multi-lingual evaluation of code generation models. In The Eleventh International Conference on Learning Representations.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Tyler Chang, Zhuowen Tu, and Benjamin Bergen. 2022. The geometry of multilingual language model representations. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 119–136.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171-4186.

Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin,

Ting Liu, Daxin Jiang, et al. 2020. Codebert: A pre-trained model for programming and natural languages. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1536–1547.

Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin. 2022. Unixcoder: Unified crossmodal pre-training for code representation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7212–7225.

Daya Guo, Shuo Ren, Shuai Lu, Zhangyin Feng, Duyu Tang, LIU Shujie, Long Zhou, Nan Duan, Alexey Svyatkovskiy, Shengyu Fu, et al. 2020. Graphcodebert: Pre-training code representations with data flow. In International Conference on Learning Representations.

Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. 2019. Codesearchnet challenge: Evaluating the state of semantic code search. arXiv preprint arXiv:1909.09436.

Paras Jain, Ajay Jain, Tianjun Zhang, Pieter Abbeel, Joseph Gonzalez, and Ion Stoica. 2021. Contrastive code representation learning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5954–5971.

Denis Kocetkov, Raymond Li, LI Jia, Chenghao Mou, Yacine Jernite, Margaret Mitchell, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, et al. 2022. The stack: 3 tb of permissively licensed source code. Transactions on Machine Learning Research.

Saurabh Kulshreshtha, Jose Luis Redondo Garcia, and Ching Yun Chang. 2020. Cross-lingual alignment methods for multilingual bert: A comparative study. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 933–942.

Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. 2023. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161.

Jindřich Libovickò, Rudolf Rosa, and Alexander Fraser. 2020. On the language neutrality of pre-trained multilingual representations. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1663–1674.

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. 2023. Wizardcoder: Empowering code large language models with evolinstruct. arXiv preprint arXiv:2306.08568.

Erik Nijkamp, Hiroaki Hayashi, Caiming Xiong, Silvio Savarese, and Yingbo Zhou. 2023. Codegen2: Lessons for training llms on programming and natural languages. arXiv preprint arXiv:2305.02309.

Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. 2022. Codegen: An open large language model for code with multi-turn program synthesis. In The Eleventh International Conference on Learning Representations.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748.

Vihari Piratla, Praneeth Netrapalli, and Sunita Sarawagi. 2020. Efficient domain generalization via commonspecific low-rank decomposition. In International Conference on Machine Learning, pages 7728–7738. PMLR.

Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. How multilingual is multilingual bert? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4996–5001.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551.

Uma Roy, Noah Constant, Rami Al-Rfou, Aditya Barua, Aaron Phillips, and Yinfei Yang. 2020. Lareqa: Language-agnostic answer retrieval from a multilingual pool. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5919–5930.

Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. 2023. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950.

Erhard Schmidt. 1907. Zur theorie der linearen und nichtlinearen integralgleichungen. Mathematische Annalen, 63(4):433–476.

Tal Schuster, Ori Ram, Regina Barzilay, and Amir Globerson. 2019. Cross-lingual alignment of contextual word embeddings, with applications to zeroshot dependency parsing. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1599–1613.

Xin Wang, Yasheng Wang, Fei Mi, Pingyi Zhou, Yao Wan, Xiao Liu, Li Li, Hao Wu, Jin Liu, and Xin Jiang. 2021a. Syncobert: Syntax-guided multi-modal contrastive pre-training for code representation. arXiv preprint arXiv:2108.04556.

Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi DQ Bui, Junnan Li, and Steven CH Hoi. 2023. Codet5+: Open code large language models for code understanding and generation. arXiv preprint arXiv:2305.07922.

Yue Wang, Weishi Wang, Shafiq Joty, and Steven CH Hoi. 2021b. Codet5: Identifier-aware unified pretrained encoder-decoder models for code understanding and generation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 8696–8708.

Zhihui Xie, Handong Zhao, Tong Yu, and Shuai Li. 2022. Discovering low-rank subspaces for languageagnostic multilingual representations. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5617–5633.

Frank F Xu, Uri Alon, Graham Neubig, and Vincent Josua Hellendoorn. 2022. A systematic evaluation of large language models of code. In Proceedings of the 6th ACM SIGPLAN International Symposium on Machine Programming, pages 1–10.

Ziyi Yang, Yinfei Yang, Daniel Cer, and Eric Darve. 2021. A simple and effective method to eliminate the self language bias in multilingual representations. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5825–5832.

Qinkai Zheng, Xiao Xia, Xu Zou, Yuxiao Dong, Shan Wang, Yufei Xue, Zihan Wang, Lei Shen, Andi Wang, Yang Li, et al. 2023. Codegeex: A pre-trained model for code generation with multilingual evaluations on humaneval-x. arXiv preprint arXiv:2303.17568.

Ming Zhu, Aneesh Jain, Karthik Suresh, Roshan Ravindran, Sindhu Tipirneni, and Chandan K Reddy. 2022. Xlcost: A benchmark dataset for cross-lingual code intelligence. arXiv preprint arXiv:2206.08474.

Daniel Zügner, Tobias Kirschstein, Michele Catasta, Jure Leskovec, and Stephan Günnemann. 2020. Language-agnostic representation learning of source code from structure and context. In International Conference on Learning Representations.

## A Code2Code search results

We provide more detailed results for the Code2Code search, where we calculate the mean over the target language and present the MRR (Mean Reciprocal Rank) for each source language. This is similar to what is shown in Table 1 for CodeT5+. For CodeBERT, the details can be found in Table 3. For GraphCodeBERT, see Table 4. Results for UnixCoder are in Table 6, and for StarEncoder, refer to Table 5.

## B Text2Code search results

We provide more detailed results for the Text2Code search, similar to the information shown in Table 2 for UnixCoder. In Table 7, we display results for all four models: CodeBERT, GraphCodeBERT, StarEncoder, and CodeT5+.

## C Effect on Contrastive Finetuned models

In this section, we fine-tune the models using contrastive loss (Oord et al., 2018) for three epochs, with a batch size of eight, employing the AdamW optimizer with a linear scheduler and 500 warm-up steps. For both Code2Code search and Text2Code search, we ensure that each batch includes translation pairs from multiple languages through random sampling. This form of multilingual contrastive learning encourages representations to be aligned across programming languages. The results for Code2Code search can be viewed in Figure 9, and for Text2Code search in Figure 10. Similar to what we saw in Section 3.2 and Section 3.3, there is no significant benefit in removing language components when representations are already aligned.

## D Code and Compute

The code is set to be released publicly, and during the experiments, T4 GPUs were utilized for a total of approximately 200 GPU hours.

## Algorithm 1: Language Agnostic Code Embeddings

```latex
Input: code embedding e ∈ Rd, programming language l $\in \{ 1 , \ldots , \ell \}$ of embedding e and embedding matrices $\mathbf { E } _ { 1 } , \ldots , \mathbf { E } _ { \ell } ,$ where $\mathbf { E } _ { i } \in \mathbb { R } ^ { n \times d } ,$ rank r of the syntactic
subspace in the case of LRD and CS-LRD.
Output: Language agnostic code embedding $\mathbf { e } _ { a } \in \mathbb { R } ^ { d } .$
def centering(M) def LRD(E, r) def CS-LRD(M, r)
ml = M[:, l] Ur, Σr, Vr = TOPK-SVD(M, r) $\widehat { \bf m } _ { c } = \frac { 1 } { d } { \bf M } \cdot \mathbb { 1 } _ { \ell } .$
return ml $\in \mathbb { R } ^ { d }$ return $\mathbf { V } _ { r } \in \mathbb { R } ^ { d \times r }$ $\widehat { \mathbf { U } } _ { r } , \widehat { \mathbf { \Sigma } } _ { r } ^ { \mathrm { ~ s ~ } } , \widehat { \mathbf { V } } _ { r } = \operatorname { T O P K - S V D } ( \mathbf { M } - \widehat { \mathbf { m } } _ { c } \cdot \mathbb { 1 } _ { \ell } ^ { T } , r )$
$\widehat { \mathbf { M } } _ { s } = \widehat { \mathbf { U } } _ { r } , \widehat { \mathbf { r } } _ { s } = \widehat { \mathbf { V } } _ { r } ^ { T } \cdot \widehat { \mathbf { \Sigma } } _ { r } .$
$\widetilde { \mathbf { M } } _ { s } = \underset { . . } { \mathrm { P S E U D O - I N V E R S E } } ( \widehat { \mathbf { m } } _ { c } \cdot \mathbb { 1 } _ { \ell } ^ { T } + \widehat { \mathbf { M } } _ { s } \cdot \widehat { \mathbf { r } } _ { s } ^ { T } ) .$
$\mathbf { m } _ { c } = \widetilde { \mathbf { M } } _ { s } \cdot \mathbb { 1 } _ { \ell } / \| \widetilde { \mathbf { M } } _ { s } \cdot \mathbb { 1 } _ { \ell } \| _ { 2 } ^ { 2 }$
$\mathbf { U } _ { r } , \mathbf { \Xi } _ { \mathbf { Z } _ { r } } , \mathbf { V } _ { r } = \mathrm { T O P K - S V D } ( \tilde { \widetilde { \mathbf { M } } } _ { s } - \mathbf { m } _ { c } \cdot \boldsymbol { \mathbb { 1 } } ^ { T } , r ) ;$
$\mathbf { M } _ { s } = \mathbf { U } _ { r } , \mathbf { r } = \mathbf { V } ^ { T } \cdot \pmb { \Sigma } _ { r } .$
return Ms ∈ Rd×r
M = [MEAN(E1), . . . , MEAN(Ee)] ∈ Rd×e
if estimation-method == centering then
es = centering(M.)
else
else if estimation-method == LRD then
$\mathbf { P } = \mathsf { L R D } ( \mathbf { E } _ { l } , r ) .$
else
P = CS-LRD(M, r).
end
es = e − P · pT . e.
end
$\mathbf { e } ^ { a } = \mathbf { e } - \mathbf { e } ^ { s }$
return eª
```

![](images/75bcd54cfb9a83e6c70eb0cc884c40dcb7c161fe48dad9b0361a45a8642a1292.jpg)  
(a) CodeBERT

![](images/a4e74d6203b3371d031bd6156c6682ee6e5bcb44954480bc5baac1afcac197f0.jpg)

![](images/d585df992245884c5ab8b2b5df1d8190bae8303ea6caaa955e0c5f0a4d3f8f68.jpg)  
(f) CodeBERT

(b) GraphCodeBERT  
![](images/1429ce91089f6919fc7c8aa80a03b20c903886dcc0d5bb8272e34f26d25552a8.jpg)  
(g) GraphCodeBERT

![](images/7c9f078760ae86f7e3aeae4d51458965cebaa2e356be2a6aa25ee58960948837.jpg)

(c) Unixcoder  
![](images/37c6932502f5ed29cbdec61af8857c2a53671c823934621e47246ae48e0d9493.jpg)  
(h) Unixcoder

![](images/dd06e683a7ba38fba32c4bf928cfc293065ec66889dc14013d046e38d7994859.jpg)  
(d) StarEncoder

![](images/145b998cf276702efaf2086886035c697440a0c8c54f5af75a5b62e08e3e9843.jpg)

![](images/bb4297c4136bdc720d738361d3887a9adccb0f1a94628e03f4a4dfc707b594ff.jpg)  
(i) StarEncoder

(e) CodeT5+  
![](images/9628876dccd60aa1397387a2769d15be10891c765407fe4d961f80fe0f20e85f.jpg)  
(j) CodeT5+  
Figure 5: Effect of the rank (r) of the language subspace on MRR change in zero-shot Code2Code search. The top row shows it for LRD, and the bottom row for CS-LRD.

![](images/3cf2d2a9b771cb59688d933307c512c7fbd1002889118aaed356cbe0aed6a3e5.jpg)  
(a) CodeBERT

![](images/54914e8fd3cacf52f4dc767aee6d3402c94265cfced1335424eb51cbe91366cf.jpg)

![](images/24f6ec629e7b92efdd31515bf6037e4065dfb4b2498e4480d663baa9d1b5d7d0.jpg)  
(f) CodeBERT

(b) GraphCodeBERT  
![](images/6c9862e6784330cc896df3260927ed4b8f5ce052e42e7ee3952e79f2a6a1877f.jpg)  
(g) GraphCodeBERT

![](images/4e09f535a3ce100324a0c195cae25f4b8711295ed72a4515925723d9affc41cc.jpg)

(c) UnixCoder  
![](images/bce8f28272bcead47fb78a422ef54e7d32263407f7917a0dc516d83ebb99d2cc.jpg)  
(h) UnixCoder

![](images/584950e10435408a115196d0ec2c65d9d53f5168899c57598579a05d8d5f72ed.jpg)

(d) StarEncoder  
![](images/d569ba137e80040bb660786df1940a1b64f6a05243ca9adec3b35bf5ce1b1b56.jpg)  
(i) StarEncoder

![](images/0d550565943af07bf776348e6bd783cb42282817ceae1fbd1c95b3f78bb91ba6.jpg)

(e) CodeT5+  
![](images/47842ec7890e620d54ad5b01d940f5433fa1943e6be205ad585f57c10ed817c6.jpg)  
(j) CodeT5+

Figure 6: Effect of the rank (r) of the language subspace on MRR change in zero-shot Text2Code search. The top row shows it for LRD, and the bottom row for CS-LRD. 'Without English' and 'With English' indicate cases where query embeddings remain untransformed and transformed, respectively.

![](images/0d4fe53caf17b7af53169efbcd517d2e618be3f18bbbde059b0cee623fd4caff.jpg)  
(a) CodeBERT

![](images/9a1037aed1fcc2d62686fb77d5a49cdfab7e1aa8ba2c77cd72f5fa354c13a04e.jpg)  
(b) GraphCodeBERT

![](images/b0c5fb4679f936c969b3ba0b645e65898d2201b54edb69f235de8a5dfe510c83.jpg)  
(c) UnixCoder

![](images/11b68ec593a999e2fd0ab4cbcf77fcf9b4f46f7873f94da23af24fc4f07076d1.jpg)  
(d) StarEncoder

![](images/d6a7a557e1bc5811aa04f262d0ba46914881b33ee570f3c06b848919f804654d.jpg)  
(e) CodeT5+

![](images/5fdf4c429b2611bec37eaf2ac0883b4beed1b0516ab7a397e00da2b6f658830d.jpg)  
(f) CodeBERT

![](images/4f889a4ac564854aa7a1d37633d5a74e4ec22ca20c4d4f9fba37ce4ec417afbe.jpg)  
(g) GraphCodeBERT

![](images/e49feed96942e64beb849316d0592f2ac17ac2f6cac7986afd441acde536b648.jpg)

![](images/4a025de9bbde768d1669fceb844979292fc92f78c7846a02c03c86fd2d2c4a9f.jpg)

![](images/36d45aeda8ec5e153a3cd4dc45f95e8afce8277bd600cdb846b9882cc5ac8e87.jpg)

![](images/0ce32f1ab7beb765f2a9872c71e44020a06692c97500f3fa2ce6c937bff98bf6.jpg)  
(j) CodeT5+

(h) UnixCoder  
![](images/98643ceeef6cc37f530672ea24b323c577c364b61a969dc17a180590e4e985b0.jpg)  
(k) CodeBERT  
(1) GraphCodeBert

(i) StarEncoder  
![](images/3103d09286d22a0998425b272269ab77362ab709e71c188973a1eaf4d3a2708d.jpg)  
(m) UnixCoder

![](images/7a629b8d524f55f8ae32d2be5c6556924aaeb576eb723ff829e5af37800df461.jpg)  
(n) StarEncoder

![](images/d66348ff16a7d53d70971e2f58f71502c5df8c795bcb17b46ab12b5e980dc107.jpg)  
(0) CodeT5+

Figure 7: Impact of Estimation Set size on MRR change, with the top, middle, and bottom rows showing effects for Centering, LRD, and CS-LRD, respectively,

![](images/18131d537f88ba6dc81b0e61e34d34f17054c60477231d8dec799deca498d930.jpg)  
(a) CodeBERT

![](images/0e9d8357a2771a1dbdb884ecf1c53ef4b33d60fa1de45a5789074a5e39f7e365.jpg)  
(b) GraphCodeBERT

![](images/cdbb1c222c266773ac7c8c7086f278db1a68bdbb0ab64e28f781e841b351b7cd.jpg)  
(c) UnixCoder

![](images/f3e8888c9daf34527e837999462f4bd9fbc591f28fab4ea49c509af87e86d49f.jpg)  
(d) StarEncoder

Figure 8: The figure presents the averaged Mean Reciprocal Rank (MRR) across three retrieval setups for zero-shot Code2Code search. These results are derived from mean, cls, and pooler embeddings. Annotations highlight the top three values in both the original and after removing language components settings for various pooling strategies.

<table><tr><td>CodeBERT(mean)</td><td></td><td>Java</td><td>Javascript</td><td>PHP</td><td>Python</td><td>Avg.</td><td>CodeBERT (cls)</td><td></td><td>Java</td><td>Javascript</td><td>PHP</td><td>Python</td><td>Avg.</td></tr><tr><td rowspan="5">Monolingual</td><td>Original Centering</td><td>46.90</td><td>56.75 47.29</td><td>57.89 43.93</td><td>43.19 33.32</td><td>51.18 44.95 (-6.23)</td><td></td><td>Original</td><td>51.34 64.16</td><td>54.02 62.21</td><td>57.42</td><td>26.15</td><td>47.23 62.84 (+15.61)</td></tr><tr><td></td><td>55.25</td><td></td><td></td><td></td><td></td><td>Monolingual</td><td>Centering</td><td></td><td></td><td>71.11</td><td>53.86</td><td></td></tr><tr><td>LRD(r=10)</td><td>49.49</td><td>59.56</td><td>60.87</td><td>45.83</td><td>53.94 (+2.76)</td><td></td><td>LRD(r=10)</td><td>53.02</td><td>55.63</td><td>59.52</td><td>28.43</td><td>49.15 (+1.92)</td></tr><tr><td>CS-LRD(r=6)</td><td>68.70</td><td>75.91</td><td>76.35</td><td>56.35</td><td>69.33 (+18.15)</td><td></td><td>CS-LRD(r=6)</td><td>77.72</td><td>78.42</td><td>76.87</td><td>60.80</td><td>73.45 (+26.22)</td></tr><tr><td>Original</td><td>36.78</td><td>39.92</td><td>44.76</td><td>31.37</td><td>38.21</td><td></td><td>Original</td><td>34.28</td><td>28.55</td><td>44.16</td><td>14.43</td><td>30.35</td></tr><tr><td rowspan="5">Source Excluded Multilingual</td><td>Centering</td><td>41.37</td><td>31.93</td><td>29.15</td><td>24.44</td><td>31.72 (-6.49)</td><td>Source Excluded Multilingual</td><td>Centering</td><td>48.48</td><td>46.28</td><td>47.32</td><td>37.69</td><td>44.94 (+14.59)</td></tr><tr><td>LRD(r=10)</td><td>38.97</td><td>43.62</td><td>46.82</td><td>33.56</td><td>40.74 (+2.53)</td><td></td><td>LRD(r=10)</td><td>36.25</td><td>31.63</td><td>46.86</td><td>16.05</td><td>32.70 (+2.35)</td></tr><tr><td>CS-LRD(r=6)</td><td>49.06</td><td>60.58</td><td>56.83</td><td>41.74</td><td>52.05 (+13.84)</td><td></td><td>CS-LRD(r=6)</td><td>48.14</td><td>60.89</td><td>52.38</td><td>41.43</td><td>50.71 (+20.36)</td></tr><tr><td>Original</td><td>4.09</td><td>6.11</td><td>8.01</td><td>5.34</td><td>5.89</td><td></td><td>Original</td><td>1.11</td><td>1.89</td><td>1.23</td><td>1.38</td><td>1.4</td></tr><tr><td>Centering</td><td>2.06</td><td>5.24</td><td>8.18</td><td>5.76</td><td>5.31 (-0.58)</td><td>Source Included Multilingual</td><td>Centering</td><td>2.69</td><td>3.00</td><td>5.30</td><td>4.38</td><td>3.84 (+2.44)</td></tr><tr><td rowspan="3">Source Included Multilingual</td><td>LRD(r=10)</td><td>4.80</td><td>7.25</td><td>9.25</td><td>6.12</td><td>6.86 (+0.97)</td><td></td><td>LRD(r=10)</td><td>1.29 4.08</td><td>2.14</td><td>1.40</td><td>1.60</td><td>1.61 (+0.21)</td></tr><tr><td>CS-LRD(r=6)</td><td>7.41</td><td>15.25</td><td>15.35</td><td>11.21</td><td>12.30 (+6.41)</td><td></td><td>CS-LRD(r=6)</td><td></td><td>6.61</td><td>6.11</td><td>4.67</td><td>5.37 (+3.97)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

<table><tr><td>CodeBERT (pooler)</td><td></td><td>Java</td><td>Javascript</td><td>PHP</td><td>Python</td><td>Avg.</td></tr><tr><td rowspan="4">Monolingual</td><td>Original</td><td>49.11</td><td>51.24</td><td>56.12</td><td>23.23</td><td>44.92</td></tr><tr><td>Centering</td><td>64.12</td><td>61.88</td><td>70.77</td><td>48.81</td><td>61.40 (+16.48)</td></tr><tr><td>LRD(r=10)</td><td>52.46</td><td>55.89</td><td>59.83</td><td>28.64</td><td>49.20 (+4.28)</td></tr><tr><td>CS-LRD(r=6)</td><td>77.53</td><td>77.39</td><td>75.30</td><td>61.49</td><td>72.93 (+28.01)</td></tr><tr><td rowspan="4">Source Excluded Multilingual</td><td>Original</td><td>36.60</td><td>29.57</td><td>41.70</td><td>12.96</td><td>30.21</td></tr><tr><td>Centering</td><td>50.17</td><td>45.56</td><td>47.41</td><td>35.03</td><td>44.54 (+14.33)</td></tr><tr><td>LRD(r=10)</td><td>40.31</td><td>36.32</td><td>47.63</td><td>17.00</td><td>35.32 (+5.11)</td></tr><tr><td>CS-LRD(r=6)</td><td>53.23</td><td>59.81</td><td>52.07</td><td>42.85</td><td>51.99 (+21.78)</td></tr><tr><td rowspan="4">Source Included Multilingual</td><td>Original</td><td>1.27</td><td>1.92</td><td>1.62</td><td>1.43</td><td>1.56</td></tr><tr><td>Centering</td><td>3.39</td><td>3.29</td><td>6.45</td><td>4.39</td><td>4.38 (+2.82)</td></tr><tr><td>LRD(r=10)</td><td>1.62</td><td>2.67</td><td>2.09</td><td>1.79</td><td>2.04 (+0.48)</td></tr><tr><td>CS-LRD(r=6)</td><td>4.19</td><td>7.05</td><td>7.49</td><td>4.95</td><td>5.92 (+4.36)</td></tr></table>

Table 3: Mean Reciprocal Rank (MRR) averaged across all target languages for zero-shot Code2Code search using CodeBERT (Feng et al., 2020).

<table><tr><td>GraphCodeBERT (mean)</td><td></td><td>Java</td><td>Javascript</td><td>PHP</td><td>Python</td><td>Avg.</td><td>GraphCodeBERT (cls)</td><td></td><td>Java</td><td>Javascript</td><td>PHP</td><td>Python</td><td>Avg.</td></tr><tr><td rowspan="4">Monolingual</td><td>Original</td><td>72.56</td><td>82.39</td><td>85.85</td><td>91.55 96.53</td><td>83.09 94.00 (+10.91)</td><td rowspan="4">Monolingual</td><td>Original</td><td>62.00</td><td>82.02</td><td>78.29</td><td>71.09</td><td>73.35 84.18 (+10.83)</td></tr><tr><td>Centering</td><td>92.75</td><td>92.50</td><td>94.20</td><td></td><td></td><td>Centering</td><td>80.86</td><td>90.83</td><td>83.13</td><td>81.92</td><td></td></tr><tr><td>LRD(r=10)</td><td>74.47</td><td>84.03</td><td>87.35</td><td>92.17</td><td>84.50 (+1.41)</td><td>LRD(r=10)</td><td>63.56</td><td>83.54</td><td>79.47</td><td>71.78</td><td>74.59 (+1.24)</td></tr><tr><td>CS-LRD(r=6)</td><td>90.47</td><td>92.36</td><td>93.47</td><td>95.21 92.88 (+9.79)</td><td></td><td>CS-LRD(r=6)</td><td>72.81</td><td>87.49</td><td>84.80</td><td>73.72</td><td>79.71 (+6.36)</td></tr><tr><td rowspan="4">Source Excluded Multilingual</td><td>Original</td><td>45.57</td><td>66.15</td><td>61.95</td><td>68.24 84.28</td><td>60.48 78.84 (+18.36)</td><td rowspan="4">Source Excluded Multilingual</td><td>Original</td><td>52.12</td><td>63.03</td><td>42.00</td><td>62.08</td><td>54.81</td></tr><tr><td>Centering</td><td>81.51</td><td>83.49</td><td>66.08</td><td></td><td></td><td>Centering</td><td>70.88</td><td>72.83</td><td>41.10</td><td>70.51</td><td>63.83 (+9.02)</td></tr><tr><td>LRD(r=10)</td><td>47.80</td><td>68.35</td><td>62.83</td><td>69.54</td><td>62.13 (+1.65)</td><td>LRD(r=10)</td><td>53.93</td><td>64.47</td><td>42.79</td><td>63.02</td><td>56.05 (+1.24)</td></tr><tr><td>CS-LRD(r=6)</td><td>66.78</td><td>78.46</td><td>70.22</td><td>76.46</td><td>72.98 (+12.50)</td><td>CS-LRD(r=6)</td><td>64.44</td><td>71.68</td><td>46.04</td><td>65.87</td><td>62.01 (+7.20)</td></tr><tr><td rowspan="4">Source Included Multilingual</td><td>Original</td><td>5.09</td><td>10.63</td><td>5.59</td><td>11.59</td><td>8.23</td><td rowspan="3">Source Included Multilingual</td><td>Original</td><td>7.94</td><td>25.25</td><td>15.02</td><td>13.36</td><td>15.39</td></tr><tr><td>Centering</td><td>2.67</td><td>11.66</td><td>9.61</td><td>17.28</td><td>10.30 (+2.07)</td><td>Centering</td><td>15.57</td><td>28.44</td><td>16.53</td><td>23.70</td><td>21.06 (+5.67)</td></tr><tr><td>LRD(r=10)</td><td>6.20</td><td>12.89</td><td>6.66</td><td>13.42</td><td>9.79 (+1.56)</td><td>LRD(r=10)</td><td>9.29</td><td>27.03</td><td>16.33</td><td>15.08</td><td>16.93 (+1.54)</td></tr><tr><td>CS-LRD(r=6)</td><td></td><td>27.92</td><td></td><td></td><td></td><td>CS-LRD(r=6)</td><td>17.34</td><td>35.19</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>12.88</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>21.38</td><td></td><td></td></tr><tr><td rowspan="3"></td><td></td><td></td><td></td><td>16.25</td><td>28.42</td><td>21.37 (+13.14)</td><td></td><td></td><td></td><td></td><td></td><td>26.41</td><td>25.08 (+9.69)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

<table><tr><td>GraphCodeBERT (pooler)</td><td></td><td>Java</td><td>Javascript</td><td>PHP</td><td>Python</td><td>Avg.</td></tr><tr><td rowspan="4">Monolingual</td><td>Original</td><td>60.64</td><td>80.56</td><td>73.22</td><td>67.14</td><td>70.39</td></tr><tr><td>Centering</td><td>58.20</td><td>78.84</td><td>70.94</td><td>68.08</td><td>69.02 (-1.37)</td></tr><tr><td>LRD(r=10)</td><td>60.52</td><td>80.62</td><td>73.47</td><td>66.94</td><td>70.39 (+0.00)</td></tr><tr><td>CS-LRD(r=6)</td><td>60.50</td><td>80.24</td><td>72.73</td><td>67.21</td><td>70.17 (-0.22)</td></tr><tr><td rowspan="4">Source Excluded Multilingual</td><td>Original</td><td>51.41</td><td>60.75</td><td>40.34</td><td>59.03</td><td>52.88</td></tr><tr><td>Centering</td><td>47.27</td><td>58.01</td><td>37.17</td><td>58.06</td><td>50.13 (-2.75)</td></tr><tr><td>LRD(r=10)</td><td>51.23</td><td>60.75</td><td>40.51</td><td>58.91</td><td>52.85 (-0.03)</td></tr><tr><td>CS-LRD(r=6)</td><td>51.20</td><td>60.57</td><td>40.14</td><td>59.06</td><td>52.74 (-0.14)</td></tr><tr><td rowspan="4">Source Included Multilingual</td><td>Original</td><td>7.32</td><td>24.18</td><td>14.81</td><td>12.74</td><td>14.76</td></tr><tr><td>Centering</td><td>0.81</td><td>4.70</td><td>2.55</td><td>2.27</td><td>2.58 (-12.18)</td></tr><tr><td>LRD(r=10)</td><td>7.27</td><td>23.83</td><td>14.60</td><td>12.56</td><td>14.56 (-0.20)</td></tr><tr><td>CS-LRD(r=6)</td><td>7.30</td><td>24.01</td><td>14.64</td><td>12.71</td><td>14.66 (-0.10)</td></tr></table>

Table 4: Mean Reciprocal Rank (MRR) averaged across all target languages for zero-shot Code2Code search using GraphCodeBERT (Guo et al., 2020).
<table><tr><td>StarEncoder (mean)</td><td></td><td>C 20.27</td><td>C# C++ 91.66 86.45</td><td>Java 90.11</td><td>Javascript 90.28</td><td></td><td>PHP Python 90.46</td><td>Avg. 79.94</td><td>StarEncoder (cls)</td><td></td><td></td><td>C 8.35</td><td>C# 50.35</td><td>C++ Java 37.75</td><td></td><td>Javascript 58.53</td><td>PHP</td><td>Python 59.21</td></tr><tr><td rowspan="2">Monolingual</td><td>Original</td><td>76.68</td><td>93.28</td><td></td><td>89.51</td><td>89.16</td><td>90.37 93.12</td><td>89.35 (+9.41)</td><td></td><td>Original Centering</td><td></td><td>57.74</td><td>65.89</td><td>46.48 52.88</td><td></td><td></td><td>59.99 61.13</td><td>45.81 59.91 (+14.10)</td></tr><tr><td>Centering LRD(r=10)</td><td>21.95</td><td>90.04 92.28</td><td></td><td></td><td></td><td>91.31</td><td>93.63 92.26</td><td>Monolingual</td><td>LRD(r=10)</td><td></td><td>48.76 8.90</td><td>54.66</td><td>44.38</td><td></td><td>67.19 61.94</td><td>64.84</td><td>65.81</td></tr><tr><td rowspan="2"></td><td></td><td>21.30</td><td>88.64 93.91</td><td>91.18</td><td></td><td>91.39</td><td></td><td>81.29 (+1.35)</td><td rowspan="2"></td><td></td><td>CS-LRD(r=9) 10.34</td><td>64.56</td><td>57.82</td><td>50.47 61.20</td><td>69.79</td><td>68.80</td><td>64.84 77.70</td><td>50.00 (+4.19) 58.60 (+12.79)</td></tr><tr><td>CS-LRD(r=9)</td><td></td><td>90.58</td><td>92.74 39.17</td><td>91.52 61.52</td><td>93.46 69.98</td><td>94.26 61.46</td><td>82.54 (+2.60)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Source Excluded Multilingual</td><td>Original</td><td>8.26 15.30</td><td>38.03 54.99</td><td></td><td></td><td></td><td></td><td>47.63 50.12 (+2.49)</td><td rowspan="2"></td><td>Original Centering</td><td>3.92 8.58</td><td>16.92 17.17</td><td>12.56 26.63</td><td>17.26</td><td>29.98 35.76</td><td>32.14 32.17</td><td>25.54 27.17</td><td>19.76 23.60 (+3.84)</td></tr><tr><td>Centering</td><td></td><td>39.10 58.78</td><td>41.94 41.74</td><td>61.10</td><td>67.43 72.09</td><td>67.20 67.48</td><td></td><td>Source Excluded Multilingual</td><td></td><td></td><td></td><td>17.75</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3"></td><td>LRD(r=10)</td><td>8.77</td><td>40.97 59.98</td><td></td><td></td><td>65.03</td><td>72.30</td><td>50.87 (+3.24)</td><td></td><td>LRD(r=10)</td><td>4.23</td><td>17.18</td><td>15.93</td><td>17.58</td><td>33.98</td><td>36.22</td><td>29.92</td><td>22.15(+2.39)</td></tr><tr><td>CS-LRD(r=9)</td><td>8.79</td><td>44.02</td><td>62.24 43.93</td><td></td><td>65.51</td><td>75.72</td><td>53.22 (+5.59)</td><td></td><td>CS-LRD(r=9)</td><td>5.07</td><td>17.83</td><td>23.96</td><td>18.04</td><td>43.05</td><td>42.13</td><td>43.27</td><td>27.62 (+7.86)</td></tr><tr><td>Original</td><td>7.01</td><td>27.61 20.61</td><td>29.83</td><td>28.61</td><td></td><td>5.76 18.88</td><td>19.76</td><td></td><td>Original</td><td>2.93</td><td>10.92</td><td>2.47</td><td>11.43</td><td>2.71 2.76</td><td>0.93 0.25</td><td>5.62 5.39</td><td>5.29 4.90 (-0.39)</td></tr><tr><td rowspan="3">Source Included Multilingual</td><td>Centering</td><td>6.67</td><td>16.52</td><td>21.35</td><td>19.71 32.07</td><td>13.44</td><td>2.66</td><td>17.76 14.02 (-5.74) 24.13</td><td>Source Included Multilingual</td><td></td><td>Centering</td><td>3.88 9.07</td><td>3.05</td><td>9.92</td><td></td><td></td><td></td><td></td></tr><tr><td>LRD(r=10)</td><td>7.52</td><td>30.09 33.15</td><td>25.46 31.06</td><td></td><td>32.80 35.87</td><td>7.82 10.38</td><td>22.84 (+3.08)</td><td></td><td>LRD(r=10) CS-LRD(r=9)</td><td>3.33</td><td>11.23 12.33</td><td>2.75 3.24</td><td>11.87</td><td>3.41 4.76</td><td>1.21</td><td>7.16</td><td>5.85 (+0.56) 6.99 (+1.70)</td></tr><tr><td>CS-LRD(r=9)</td><td>7.77</td><td></td><td>34.78</td><td></td><td></td><td>33.70</td><td>26.67 (+6.91)</td><td></td><td></td><td>3.89</td><td></td><td></td><td>13.11</td><td></td><td>1.53</td><td>10.09</td><td></td></tr></table>

<table><tr><td>StarEncoder (pooler)</td><td></td><td>C</td><td>C#</td><td>C++</td><td>Java</td><td>Javascript</td><td>PHP</td><td>Python</td><td>Avg.</td></tr><tr><td rowspan="3">Monolingual</td><td>Original</td><td>3.99</td><td>27.80</td><td>11.77</td><td>28.98</td><td>30.90</td><td>21.52</td><td>30.14</td><td>22.16</td></tr><tr><td>Centering</td><td>6.80</td><td>33.20</td><td>25.15</td><td>30.42</td><td>37.13</td><td>24.72</td><td>25.24</td><td>26.09 (+3.93)</td></tr><tr><td>LRD(r=10)</td><td>4.08</td><td>29.15</td><td>12.64</td><td>30.13</td><td>33.07</td><td>23.68</td><td>32.21</td><td>23.57 (+1.41)</td></tr><tr><td rowspan="3">Source Excluded Multilingual</td><td>CS-LRD(r=9)</td><td>5.98</td><td>43.66</td><td>25.21</td><td>43.38</td><td>49.81</td><td>40.37</td><td>50.78</td><td>37.03 (+14.87)</td></tr><tr><td>Original</td><td>2.94</td><td>14.49</td><td>7.93</td><td>15.52</td><td>16.19</td><td>10.41</td><td>15.50</td><td>11.85</td></tr><tr><td>Centering</td><td>3.53</td><td>15.68</td><td>8.55</td><td>15.30</td><td>19.94</td><td>11.46</td><td>13.88</td><td>12.62 (+0.77)</td></tr><tr><td rowspan="3"></td><td>LRD(r=10)</td><td>3.03</td><td>14.82</td><td>8.46</td><td>15.82</td><td>17.84</td><td>11.57</td><td>16.95</td><td>12.64 (+0.79)</td></tr><tr><td>CS-LRD(r=9)</td><td>3.75</td><td>16.53</td><td>13.88</td><td>17.39</td><td>26.75</td><td>22.39</td><td>26.55</td><td>18.18 (+6.33)</td></tr><tr><td>Original</td><td>2.40</td><td>9.75</td><td>2.28</td><td>10.59</td><td>2.69</td><td>0.85</td><td>3.66</td><td>4.6</td></tr><tr><td rowspan="3">Source Included Multilingual</td><td>Centering</td><td>1.73</td><td>8.80</td><td>2.25</td><td>8.44</td><td>3.11</td><td>0.51</td><td>3.43</td><td>4.04 (-0.56)</td></tr><tr><td>LRD(r=10)</td><td>2.45</td><td>10.06</td><td>2.37</td><td>11.01</td><td>2.98</td><td>0.99</td><td>4.29</td><td>4.88 (+0.28)</td></tr><tr><td>CS-LRD(r=9)</td><td>2.98</td><td>11.94</td><td>2.73</td><td>12.95</td><td>4.59</td><td>1.46</td><td>6.61</td><td>6.18 (+1.58)</td></tr></table>

Table 5: Mean Reciprocal Rank (MRR) averaged across all target languages for zero-shot Code2Code search using StarEncoder (Li et al., 2023).
<table><tr><td>UnixCoder (mean)</td><td></td><td>C 95.27</td><td>C# 98.31</td><td>C++</td><td>Java 98.19</td><td>Javascript 97.48</td><td>PHP 97.76</td><td>Python</td><td>Avg.</td><td>UnixCoder (cls)</td><td></td><td>C</td><td>C# C++</td><td>Java</td><td></td><td>Javascript</td><td>PHP</td><td>Python</td><td>Avg.</td></tr><tr><td rowspan="3">Monolingual</td><td>Original</td><td>95.59</td><td>98.28</td><td>98.12 98.23</td><td>98.43</td><td>97.30</td><td></td><td>98.18 98.13</td><td>97.62 97.67 (+0.05)</td><td rowspan="3">Monolingual</td><td>Original</td><td>93.63 93.76</td><td>97.29 97.57</td><td>97.54 97.79 97.99</td><td></td><td>96.88 97.12</td><td>96.83 97.14</td><td>97.55 97.68</td><td>96.79 96.99 (+0.20)</td></tr><tr><td>Centering</td><td></td><td>98.31</td><td>98.12</td><td>98.20</td><td>97.47</td><td>97.76 97.76</td><td></td><td></td><td rowspan="2"></td><td>Centering LRD(r=10)</td><td>93.55 97.29</td><td>97.66 97.56</td><td>97.82</td><td>96.92</td><td>96.83</td><td></td><td>96.79 (+0.00)</td></tr><tr><td>LRD(r=10) CS-LRD(r=9)</td><td>95.35 95.16</td><td></td><td></td><td></td><td></td><td>98.19 98.22</td><td>97.63 (+0.01) 97.68 (+0.06)</td><td></td><td></td><td>97.64</td><td>97.68</td><td></td><td>97.02</td><td>96.95</td><td>97.55 97.61</td><td>97.03 (+0.24)</td></tr><tr><td rowspan="3">Source Excluded Multilingual</td><td></td><td></td><td>98.34</td><td>98.27</td><td>98.43 87.44</td><td>97.57</td><td>97.76 90.18</td><td>92.77</td><td>88.87</td><td rowspan="3"></td><td>CS-LRD(r=9) Original</td><td>94.29 75.57</td><td></td><td>98.04 82.07</td><td></td><td></td><td></td><td></td><td>86.02</td></tr><tr><td>Original</td><td>78.34</td><td>87.63</td><td>94.19 93.94</td><td></td><td>91.51 90.77</td><td>89.48</td><td></td><td></td><td></td><td></td><td>81.52 81.35</td><td>92.98</td><td></td><td>89.90</td><td>88.45</td><td>91.66</td><td></td></tr><tr><td>Centering</td><td>77.94</td><td>87.02</td><td>87.55</td><td></td><td></td><td>92.24</td><td>88.42 (-0.45) 88.89 (+0.02)</td><td>Source Excluded Multilingual</td><td>Centering</td><td>74.99 75.75</td><td>81.63</td><td>92.77 82.34</td><td></td><td>89.74 89.89</td><td>87.65 88.50</td><td>91.83 91.66</td><td>85.81 (-0.21) 86.08 (+0.06)</td></tr><tr><td rowspan="4">Source Included Multilingual</td><td>LRD(r=10)</td><td>78.53 79.32</td><td>87.68</td><td>94.21 94.39</td><td>87.44 88.10</td><td>91.56</td><td>90.16 90.57</td><td>92.64 93.28</td><td>89.41 (+0.54)</td><td></td><td>LRD(r=10)</td><td></td><td>93.02</td><td>82.11 83.37</td><td></td><td></td><td></td><td></td></tr><tr><td>CS-LRD(r=9)</td><td></td><td>88.45</td><td></td><td></td><td>91.76</td><td></td><td>82.58</td><td></td><td>CS-LRD(r=9)</td><td>77.26</td><td>82.87</td><td>93.12</td><td></td><td>90.45</td><td>88.52</td><td>92.46</td><td>86.86 (+0.84)</td></tr><tr><td>Original</td><td>71.19</td><td>82.03</td><td>85.87</td><td>81.45</td><td>84.75</td><td>85.21 82.98</td><td>87.57</td><td></td><td>Original</td><td>67.67</td><td>75.89</td><td>81.38</td><td>75.50</td><td>80.70</td><td>81.39</td><td>84.61</td><td>78.16</td></tr><tr><td>Centering</td><td>69.67</td><td>80.80 82.07</td><td>85.86 80.92</td><td></td><td>83.05</td><td>85.85</td><td>81.30 (-1.28) 82.63 (+0.05)</td><td>Source Included Multilingual</td><td>Centering</td><td>65.04</td><td>73.93</td><td>81.83</td><td>74.42</td><td>79.12</td><td>78.96</td><td>83.49</td><td>76.68 (-1.48) 78.31 (+0.15)</td></tr><tr><td rowspan="3"></td><td>LRD(r=10) CS-LRD(r=9)</td><td>71.26 72.28</td><td>82.92</td><td>86.02 86.71</td><td>81.47 82.07</td><td>84.75 85.29</td><td>85.25 85.81</td><td>87.57 88.36 83.35 (+0.77)</td><td></td><td>LRD(r=10) CS-LRD(r=9)</td><td>67.84</td><td>76.03 77.71</td><td>81.64 83.09</td><td>75.60 77.18</td><td>80.81 81.50</td><td>81.56 81.61</td><td>84.68 85.32</td><td>79.43 (+1.27)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>69.63</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

<table><tr><td>Unixcoder (pooler)</td><td></td><td>C</td><td>C#</td><td>C++</td><td>Java</td><td>Javascript</td><td>PHP</td><td>Python</td><td>Avg.</td></tr><tr><td rowspan="4">Monolingual</td><td>Original</td><td>93.17</td><td>97.16</td><td>97.30</td><td>97.68</td><td>96.45</td><td>96.59</td><td>97.26</td><td>96.52</td></tr><tr><td>Centering</td><td>93.31</td><td>97.36</td><td>97.47</td><td>98.01</td><td>96.78</td><td>96.91</td><td>97.54</td><td>96.77 (+0.25)</td></tr><tr><td>LRD(r=10)</td><td>93.09</td><td>97.22</td><td>97.40</td><td>97.75</td><td>96.50</td><td>96.66</td><td>97.29</td><td>96.56 (+0.04)</td></tr><tr><td>CS-LRD(r=9)</td><td>93.59</td><td>97.38</td><td>97.53</td><td>97.92</td><td>96.64</td><td>96.72</td><td>97.43</td><td>96.74 (+0.22)</td></tr><tr><td rowspan="4">Source Excluded Multilingual</td><td>Original</td><td>74.44</td><td>80.66</td><td>92.35</td><td>80.81</td><td>88.70</td><td>87.47</td><td>90.75</td><td>85.03</td></tr><tr><td>Centering</td><td>74.42</td><td>80.47</td><td>92.28</td><td>81.85</td><td>88.59</td><td>87.43</td><td>91.29</td><td>85.19 (+0.16)</td></tr><tr><td>LRD(r=10)</td><td>74.50</td><td>80.92</td><td>92.57</td><td>81.09</td><td>88.87</td><td>87.72</td><td>90.82</td><td>85.21 (+0.18)</td></tr><tr><td>CS-LRD(r=9)</td><td>75.75</td><td>81.76</td><td>92.61</td><td>82.48</td><td>89.35</td><td>87.89</td><td>91.71</td><td>85.94 (+0.91)</td></tr><tr><td rowspan="4">Source Included Multilingual</td><td>Original</td><td>67.00</td><td>75.07</td><td>80.49</td><td>74.39</td><td>79.36</td><td>80.87</td><td>83.59</td><td>77.25</td></tr><tr><td>Centering</td><td>65.36</td><td>73.11</td><td>81.22</td><td>74.22</td><td>77.48</td><td>79.12</td><td>82.82</td><td>76.19 (-1.06)</td></tr><tr><td>LRD(r=10)</td><td>67.05</td><td>75.41</td><td>80.83</td><td>74.67</td><td>79.61</td><td>80.99</td><td>83.93</td><td>77.50 (+0.25)</td></tr><tr><td>CS-LRD(r=9)</td><td>68.59</td><td>76.69</td><td>81.80</td><td>76.62</td><td>80.20</td><td>81.09</td><td>84.44</td><td>78.49 (+1.24)</td></tr><tr><td>CodeBERT (mean)</td><td></td><td>Go</td><td>Java</td><td>Javascript</td><td>PHP</td><td>Python</td><td>Ruby</td><td>Avg.</td></tr><tr><td rowspan="4">Monolingual</td><td>Original</td><td>0.15</td><td>0.04</td><td>0.06</td><td>0.03</td><td>0.06</td><td>0.37</td><td>0.12</td></tr><tr><td>Centering</td><td>0.13</td><td>0.26</td><td>0.29</td><td>0.19</td><td>0.31</td><td>1.04</td><td colspan="2">0.37 (+0.25)</td></tr><tr><td>LRD(r=10)</td><td>0.18</td><td>0.04</td><td>0.06</td><td>0.03</td><td>0.07</td><td>0.40</td><td colspan="2">0.13 (+0.01)</td></tr><tr><td>CS-LRD(r=6)</td><td>0.33</td><td>0.15</td><td>0.18</td><td>0.06</td><td>0.27</td><td>0.87</td><td colspan="2">0.31 (+0.19)</td></tr><tr><td rowspan="4">Multilingual</td><td>Original</td><td>0.07</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.02</td><td>0.27</td><td>0.06</td></tr><tr><td>Centering</td><td>0.02</td><td>0.03</td><td>0.04</td><td>0.05</td><td>0.24</td><td>0.27</td><td colspan="2">0.11 (+0.05)</td></tr><tr><td>LRD(r=10)</td><td>0.09</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.02</td><td>0.30</td><td colspan="2">0.07 (+0.01)</td></tr><tr><td>CS-LRD(r=6)</td><td>0.13</td><td>0.05</td><td>0.02</td><td>0.00</td><td>0.19</td><td>0.41</td><td colspan="2">0.13 (+0.07)</td></tr></table>

Table 6: Mean Reciprocal Rank (MRR) averaged across all target languages for zero-shot Code2Code search using UnixCoder (Guo et al., 2022).

<table><tr><td>GraphCodeBert (mean)</td><td></td><td>Go</td><td>Java</td><td>Javascript</td><td>PHP</td><td>Python</td><td>Ruby</td><td>Avg.</td></tr><tr><td rowspan="4">Monolingual</td><td>Original</td><td>12.48</td><td>8.60</td><td>7.30</td><td>8.08</td><td>10.38</td><td>20.80</td><td>11.27</td></tr><tr><td>Centering</td><td>19.30</td><td>17.32</td><td>18.14</td><td>14.62</td><td>18.53</td><td>31.59</td><td>19.92 (+8.65)</td></tr><tr><td>LRD(r=10)</td><td>14.85</td><td>10.10</td><td>8.58</td><td>9.20</td><td>12.07</td><td>22.94</td><td>12.96 (+1.69)</td></tr><tr><td>CS-LRD(r=6)</td><td>15.94</td><td>11.86</td><td>8.07</td><td>10.22</td><td>13.09</td><td>24.05</td><td>13.87 (+2.60)</td></tr><tr><td rowspan="4">Multilingual</td><td>Original</td><td>5.49</td><td>7.41</td><td>3.01</td><td>4.05</td><td>7.39</td><td>6.63</td><td>5.66</td></tr><tr><td>Centering</td><td>8.60</td><td>12.07</td><td>7.58</td><td>9.28</td><td>12.26</td><td>20.01</td><td>11.63 (+5.97)</td></tr><tr><td>LRD(r=10)</td><td>6.75</td><td>8.70</td><td>3.47</td><td>4.77</td><td>8.70</td><td>7.85</td><td>6.71 (+1.05)</td></tr><tr><td>CS-LRD(r=6)</td><td>7.60</td><td>9.29</td><td>3.28</td><td>4.89</td><td>10.08</td><td>14.50</td><td>8.27 (+2.61)</td></tr></table>

<table><tr><td>StarEncoder (mean)</td><td></td><td>Go</td><td>Ruby</td><td>Java</td><td>Javascript</td><td>PHP</td><td>Python</td><td>Avg.</td></tr><tr><td rowspan="4">Monolingual</td><td>Original</td><td>1.85</td><td>4.41</td><td>1.89</td><td>1.55</td><td>0.57</td><td>2.14</td><td>2.07</td></tr><tr><td>Centering</td><td>18.00</td><td>18.98</td><td>10.65</td><td>10.52</td><td>6.95</td><td>10.71</td><td>12.64 (+10.57)</td></tr><tr><td>LRD(r=10)</td><td>2.08</td><td>4.88</td><td>2.21</td><td>1.76</td><td>0.72</td><td>2.52</td><td>2.36 (+0.29)</td></tr><tr><td>CS-LRD(r=9)</td><td>3.07</td><td>7.60</td><td>3.93</td><td>2.71</td><td>1.68</td><td>4.09</td><td>3.85 (+1.78)</td></tr><tr><td rowspan="4">Multilingual</td><td>Original</td><td>0.96</td><td>1.88</td><td>1.33</td><td>0.75</td><td>0.16</td><td>1.80</td><td>1.15</td></tr><tr><td>Centering</td><td>5.80</td><td>9.92</td><td>5.92</td><td>4.48</td><td>1.51</td><td>8.41</td><td>6.01 (+4.86)</td></tr><tr><td>LRD(r=10)</td><td>1.06</td><td>2.18</td><td>1.60</td><td>0.88</td><td>0.21</td><td>2.08</td><td>1.34 (+0.19)</td></tr><tr><td>CS-LRD(r=9)</td><td>1.09</td><td>4.28</td><td>2.69</td><td>1.31</td><td>0.49</td><td>3.43</td><td>2.22 (+1.07)</td></tr></table>

<table><tr><td>CodeT5+ (pooler)</td><td></td><td>Go</td><td>Ruby</td><td>Java</td><td>Javascript</td><td>PHP</td><td>Python</td><td>Avg.</td></tr><tr><td rowspan="4">Monolingual</td><td>Original</td><td>90.74</td><td>74.45</td><td>71.82</td><td>69.18</td><td>67.82</td><td>71.72</td><td>74.29</td></tr><tr><td>Centering</td><td>89.98</td><td>73.38</td><td>70.36</td><td>67.71</td><td>65.57</td><td>70.07</td><td>72.84 (-1.45)</td></tr><tr><td>LRD(r=1)</td><td>90.42</td><td>73.86</td><td>71.18</td><td>68.45</td><td>67.03</td><td>71.10</td><td>73.67 (-0.62)</td></tr><tr><td>CS-LRD(r=1)</td><td>90.69</td><td>74.32</td><td>71.90</td><td>69.13</td><td>67.81</td><td>71.60</td><td>74.24 (-0.05)</td></tr><tr><td rowspan="4">Multilingual</td><td>Original</td><td>89.40</td><td>55.82</td><td>65.60</td><td>58.65</td><td>63.36</td><td>67.32</td><td>66.69</td></tr><tr><td>Centering</td><td>86.89</td><td>58.09</td><td>59.46</td><td>52.24</td><td>55.43</td><td>65.75</td><td>62.98 (-3.71)</td></tr><tr><td>LRD(r=1)</td><td>88.83</td><td>56.69</td><td>63.76</td><td>55.46</td><td>60.74</td><td>67.03</td><td>65.42 (-1.27)</td></tr><tr><td>CS-LRD(r=1)</td><td>89.37</td><td>55.65</td><td>65.93</td><td>58.35</td><td>63.17</td><td>67.08</td><td>66.59 (-0.10)</td></tr></table>

Table 7: Mean Reciprocal Rank (MRR) for zero-shot Text2Code search using CodeBERT (Feng et al., 2020), GraphCodeBERT (Guo et al., 2020), StarEncoder (Li et al., 2023), CodeT5+ (Wang et al., 2023).

(a) CodeBERT  
![](images/1c36d1af815c1963070b3855d7bd0aa4afa774248ceaa8754252a1c4eb8c19dd.jpg)

(b) GraphCodeBERT  
![](images/486acf5674ef5c3922ff0f550c10d04b88387660a95efb594d9adc411e228ca0.jpg)

![](images/e95bb66ea99ef3f3fa25f062d5d857ed76ed0185f944781085dc859f8376624b.jpg)  
(c) UnixCoder

![](images/0b5028c73008578fb501021f5a2b1663983fe0a53a678d7b086da7a4fd0a6708.jpg)  
(d) StarEncoder

![](images/4d11de2ba794d451a88c27bb9db20debcf572853fb3c23f9b5566f52d2c9eefd.jpg)  
(e) CodeT5+

Figure 9: Absolute change in Mean Reciprocal Rank (MRR) after removing language components for Code2Code search after contrastive fine-tuning.

![](images/0d76337737583857f19366b430a781b1ff6e0e3f770b1096a08a4a2710e26ef9.jpg)  
(a) CodeBERT

![](images/a478d619e3e6051561e83a44bf52561a01b6c83a2805efd87a28c3a65db30eb8.jpg)  
(b) GraphCodeBERT

![](images/851224c7f8e96195867c07a4fb247bb2a052ddf94ae88c1df89346138f0f8daa.jpg)  
(c) UnixCoder

![](images/fa4b5cf2e935a14f30eb7814815d9dc3ff2282cd4664e3422578e8c9e126c7f2.jpg)  
(d) StarEncoder

![](images/faff794d9cd3b279c5927e271a18f70c7ef4d395cf15954eb14bdcd66e211fa0.jpg)  
(e) CodeT5+

Figure 10: Absolute change in Mean Reciprocal Rank (MRR) after removing language components for Text2Code search after contrastive fine-tuning.