# Ungrammatical-syntax-based In-context Example Selection for Grammatical Error Correction

Chenming Tang Fanyi Qu Yunfang Wu

National Key Laboratory for Multimedia Information Processing, Peking University MOE Key Laboratory of Computational Linguistics, Peking University School of Computer Science, Peking University {tangchenming@stu, fanyiqu@, wuyf@}pku.edu.cn

## Abstract

In the era of large language models (LLMs), in-context learning (ICL) stands out as an effective prompting strategy that explores LLMs potency across various tasks. However, applying LLMs to grammatical error correction (GEC) is still a challenging task. In this paper, we propose a novel ungrammatical-syntaxbased in-context example selection strategy for GEC. Specifically, we measure similarity of sentences based on their syntactic structures with diverse algorithms, and identify optimal ICL examples sharing the most similar ill-formed syntax to the test input. Additionally, we carry out a two-stage process to further improve the quality of selection results. On benchmark English GEC datasets, empirical results show that our proposed ungrammaticalsyntax-based strategies outperform commonlyused word-matching or semantics-based methods with multiple LLMs. This indicates that for a syntax-oriented task like GEC, paying more attention to syntactic information can effectively boost LLMs’ performance. Our code is available at https://github.com/JamyDon/ SynICL4GEC.

## 1 Introduction

Recently, large language models (LLMs) have shown awesome power in many areas and ended the contest on many tasks (Chowdhery et al., 2023; Bubeck et al., 2023; Touvron et al., 2023). Unfortunately for LLMs, grammatical error correction (GEC), which aims at automatically correcting grammatical errors in erroneous text (Bryant et al., 2022), is still a challenging task where they cannot beat conventional models yet. Fang et al. (2023b) and Loem et al. (2023) explore the performance of LLMs on GEC, demonstrating mainstream LLMs lag over 10 points behind the state-of-the-art result. Therefore, it is significant to explore new strategies to further improve the power of LLMs on GEC.

As a helpful prompting strategy, in-context learning (ICL) has achieved impressive results on many tasks (Dong et al., 2022; Min et al., 2022). In ICL, several in-context examples are presented to LLMs as demonstrations before the input test sample in order to make LLMs aware of the requirement and output format of the specific task, thereby enhancing LLMs’ performance during the subsequent generation process. Since the quality of incontext examples plays a crucial role under the few-shot setting, some strategies of example selection have been proposed (Agrawal et al., 2023; Li et al., 2023a; Ye et al., 2023; Gupta et al., 2023).

To the best of our knowledge, no existing work on ICL example selection has taken syntactic information into consideration. However, GEC aims to correct grammatical errors and is a typical syntaxoriented task. In GEC, common errors can be classified into four types: misuse, missing, redundancy and word order (Bryant et al., 2017; Zhang et al., 2022a), and the last three of which are closely related to syntactic structure. That is, the missing, redundancy or disorder of text constituents may lead to ill-formed syntax (Zhang et al., 2022b), suggesting the important role syntax plays in GEC. Hence, selecting in-context examples based on syntactic structure is likely to benefit LLMs more than conventional word-matching-based or semanticsbased approaches.

Comparing with semantic similarity, syntactic similarity of text is less-studied. Previous works have leveraged the similarity of dependency trees to help multi-document summarization (Özate¸s et al., 2016) and semantic textual similarity (Le et al., 2018). To compute syntactic similarity, several effective algorithms computing similarity between syntactic trees have been proposed. Tree Kernel is a typical one, which counts the shared sub-structures of two trees to measure their similarity (Collins and Duffy, 2002; Vishwanathan et al., 2004; Moschitti, 2006). Polynomial Distance is another handy one, which converts syntactic trees into polynomials and then computes the distances (Liu et al., 2022).

<table><tr><td></td><td>Source (Erroneous Sentence)</td><td>Target (Corrected Sentence)</td></tr><tr><td>Input</td><td>No smoking in the public places.</td><td>No smoking in public places.</td></tr><tr><td>BM25</td><td>I am writing to complain about the suggested bar on smoking in public areas.</td><td>I am writing to complain about the suggested ban on smoking in public areas.</td></tr><tr><td>Poly.</td><td>No future for the public transport?</td><td>No future for public transport?</td></tr></table>

Table 1: An example comparing the selection results of BM25 and Polynomial Distance ("Poly." in the table).

In this paper, we propose a novel ICL example selection strategy for GEC, by computing similarities of syntactic trees on ungrammatical sentences. Specially, we apply the syntactic similarity algorithms (Tree Kernel and Polynomial Distance) to dependency trees generated by a GEC-oriented parser (GOPar) proposed by Zhang et al. (2022b), which is more reliable and provides error information when parsing ungrammatical sentences. Moreover, we carry out a two-stage process. In the first stage, namely selection, a fast and general method like BM25 is applied to filter out most of the irrelevant instances from the training data and obtain a much smaller candidate set. In the second stage, namely ranking, the more powerful syntax-based method is implemented to find out the best k instances as the final in-context examples.

To give a quick view of the superiority of our method, Table 1 shows an example illustrating the difference between BM25 selection and our ungrammatical-syntax-based method with Polynoimal Distance selection. BM25 only selects examples with similar words while Polynomial Distance is able to select those with similar grammatical errors, which will benefit the GEC task more.

We conduct experiments on two mainstream English GEC datasets, BEA-2019 (Bryant et al., 2019) and CoNLL-2014 (Ng et al., 2014). According to experimental results, Polynomial Distance and its weighted version achieve competitive results even under the single-stage setting, improving the performance by around 3 points and 2 points on BEA-19 and CoNLL-14 respectively. With the help of our two-stage selection, Tree Kernel gets its power unlocked and Polynomial Distance also benefits, leading to a further 1-point and 0.4-point improvement on BEA-19 and CoNLL-14 respectively. Overall, our ungrammatical-syntax-based in-context example selection methods secure the best results under all settings, outperforming conventional baselines by a margin of nearly 3 F<sub>0.5</sub>

points on average.

Our contributions can be summarized as follows:

• We propose a novel ICL example selection method based on ungrammatical syntactic similarity to improve LLMs’ performance on GEC. To the best of our knowledge, this is the first time that knowledge of syntactic structure is introduced to ICL example selection for GEC.

• We explore a two-stage selection strategy on GEC, where superficial word-similarity-based or semantics-based methods are used in the first stage and deep syntax-similarity-based ones are used in the second stage. It further improves LLMs’ performance and achieves competitive results.

• We want to re-draw the natural language processing (NLP) community’s attention to the significance of syntactic information. In this work, we show that syntax-related knowledge helps LLMs correct grammatical errors better. We believe our methods can be smoothly transferred to many other syntax-related tasks, like machine translation (MT) and information extraction (IE).

## 2 Related Work

## 2.1 Grammatical Error Correction

In the past few years, the GEC task has been dominated by sequence-to-sequence machine translation models (Junczys-Dowmunt et al., 2018; Rothe et al., 2021) and sequence-to-edit tagging models (Omelianchuk et al., 2020; Tarnavskyi et al., 2022), both based on Transformer (Vaswani et al., 2017).

Nowadays, with the finalization of mainstream models, further explorations on GEC mainly focus on two aspects. For one thing, injecting all kinds of additional knowledge into GEC models has proved helpful. The additional knowledge can be part-of-speech (POS) (Wu and Wu, 2022), syntax tree (Zhang et al., 2022b), speech representation (Fang et al., 2023a), abstract meaning representation (AMR) (Cao and Zhao, 2023), error type (Yang et al., 2023), etc. For another, multi-stage strategies help refine models’ predictions. The multistage workflow can be permutation & decoding (Yakovlev et al., 2023), detection & correction (Li et al., 2023b), re-ranking (Zhang et al., 2023), etc.

![](images/17e94db9f90292dbb82a8a3148be346afbdce310faf193a58b8a70ab1ccc5100.jpg)  
Figure 1: Our two-stage selection and ICL workflow. For each input test sample, Stage I computes word similarities with BM25 or BERT representation between the input and all training data and select the top-1000 as candidates. Then, Stage II computes ungrammatical syntactic similarities with tree kernel or polynomial distance between the input and candidates to select the most similar k example(s). After that, we concatenate the input after the k examples to construct the prompt for LLM inference. In the end, the LLM outputs the final result.

With the rising of powerful large language models (LLMs), some works have begun exploring their performance on GEC (Loem et al., 2023; Fang et al., 2023b), showing that LLMs cannot beat conventional models on GEC yet.

## 2.2 Syntactic Similarity

In computational linguistics (CL), previous works compared syntax trees of different languages to measure their similarities (Oya, 2020; Liu et al., 2022). In NLP, most works on text similarity focus on the semantic perspective (Gomaa et al., 2013, Reimers and Gurevych, 2019; Chandrasekaran and Mago, 2021), syntactic similarity of text is lessstudied. Özate¸s et al. (2016) used similarity of dependency trees to help multi-document summarization. Le et al. (2018) proposed ACV-tree (Attention Constituency Vector-tree), which combines word weight, word representation and constituency tree, to help the task of semantic textual similarity.

Syntactic similarity is usually represented by similarity between syntax trees. Tree similarity can be measured by various algorithms including Edit Distance (de Castro Reis et al., 2004), Polynomial Distance (Liu et al., 2022), Subset Tree Kernel (SSTK) (Collins and Duffy, 2002), SubTree Kernel (STK) (Vishwanathan et al., 2004), and Partial Tree Kernel (PTK) (Moschitti, 2006).

## 2.3 Large Language Models and In-context Learning

In recent years, LLMs have shown their awesome power in many areas (Brown et al., 2020; Chowdhery et al., 2023). Due to the limitation of computing resources, the focus of research on LLMs turns to the inference stage, trying to exploit the potency of LLMs with inference-only strategies.

ICL is a successful inference strategy that can make LLMs perform as well as fine-tuned models on many tasks (Brown et al., 2020; Von Oswald et al., 2023), where several in-context examples are given to LLMs as demonstrations before the actual test input. Instead of randomly sampling examples from the training set, recent works have boosted the performance of ICL by selecting incontext examples using various strategies. Agrawal et al. (2023) proposed R-BM25, a word overlap and coverage based selection strategy for machine translation. Li et al. (2023a) proposed training a Unified Demonstration Retriever (UDR) for ICL on a wide range of tasks. Ye et al. (2023) treated incontext examples as a whole and selected examples on a subset level with the help of Determinantal Point Processes (DPPs). Gupta et al. (2023) also selected examples as an entire set, with contextualembedding-level coverage as the goal.

![](images/7ef7977b84126b35c116d9ad62c2cefe8ffec6dc9c0c296d303a5077fdcb5900.jpg)  
Figure 2: Original illustration of GOPar from Zhang et al. (2022b). denotes the missing word.

Besides normal ICL, Chain-of-thought (CoT) (Wei et al., 2022; Kojima et al., 2022) is another effective inference strategy in current favor, where LLMs are prompted to think step by step and answer with intermediate rationales.

## 3 Preliminaries

## 3.1 Syntax Parser for Ungrammatical Sentences

Unlike most NLP tasks, which take correct sentences as input, the GEC task considers erroneous text as input. This gives rise to an issue that mainstream parsers may fail to obtain the expected dependency tree for the erroneous text.

To solve this problem, Zhang et al. (2022b) built a tailored GEC-Oriented dependency Parser (GOPar) based on parallel GEC training data, which is much more reliable when handling ungrammatical sentences than conventional parsers. Concretely, GOPar sets "S" (Substituted), "R" (Redundant) or "M" (Missing) labels to deal with different kinds of grammatical errors in the sentence, which inject additional information of errors into the dependency tree. Figure 2 shows the original illustration of GOPar from Zhang et al. (2022b).

Most previous works computing syntactic similarity were based on grammatical sentences with standard parsing trees (Özate¸s et al., 2016; Oya, 2020). However, in GEC, we only have the ungrammatical source sentences, on which conventional parsers may perform poorly. So we apply the algorithms of tree similarity on the parsing results of GOPar, to compute syntactic similarities between test sample and training instances. We follow the official guidance of SynGEC <sup>1</sup> to run GOPar. We use biaffine-dep-electra-en-gopar provided by SynGEC as the model for parsing.

## 3.2 Syntactic Similarity with Tree Kernel

We follow the unified Tree Kernel method proposed by Moschitti (2006), which can compute kernels of subset trees defined by Collins and Duffy (2002), subtrees defined by Vishwanathan et al. (2004) and partial trees defined in their own work.

For simplicity, we imitate the algorithm described in Le et al. (2018) and design the following algorithm (shown in Algorithm 1) to implement a simple version of Tree Kernel.

Algorithm 1 Similarity with Tree Kernel   
procedure COMPSIM(N<sub>1</sub>, N<sub>2</sub>)   
K 0   
for each node n in N do   
for each node n in N do   
if n<sub>i</sub>.label = n<sub>j</sub>.label then   
if n and n are both leaves then   
K K + 1   
else if n and n are both non-leaves then   
K  K + COMPSIM(n<sub>i</sub>, n<sub>j</sub>)   
end if   
end if   
end for   
end for   
K  K/(N<sub>1</sub>.size  N<sub>2</sub>.size)   
return K   
end procedure

For two trees $T _ { 1 }$ and $T _ { 2 } ,$ we conduct COMPSIM between their root nodes $N _ { 1 }$ and $N _ { 2 }$ to get a similarity score.

## 3.3 Syntactic Similarity with Polynomial Distance

Liu et al. (2022) converted trees into polynomials and took the distances between polynomials as tree distances to measure syntactic similarities of dependency trees.

<table><tr><td>LLaMA-2</td><td>GPT-3.5</td></tr><tr><td>There is an erroneous sentence between &#x27;&lt;erroneous sen- tence&gt;’ and &#x27;&lt;/erroneous sentence&gt;&#x27;. Then grammatical errors in the erroneous sentence will be corrected. The cor- rected version will be between &#x27;&lt;corrected sentence&gt;&#x27; and &#x27;&lt;/corrected sentence&gt;&#x27;. &lt;erroneous sentence&gt; {e1 } &lt;/erroneous sentence&gt; &lt;corrected sentence&gt; {c1 } &lt;/corrected sentence&gt; &lt;erroneous sentence&gt; {e2 } &lt;/erroneous sentence&gt; &lt;corrected sentence&gt;  $\{ c _ { 2 } \}$  &lt;/corrected sentence&gt; &lt;erroneous sentence&gt; {e3 } &lt;/erroneous sentence&gt; &lt;corrected sentence&gt;  $\{ c _ { 3 } \}$  &lt;/corrected sentence&gt; &lt;erroneous sentence&gt; {e₄} &lt;/erroneous sentence&gt; &lt;corrected sentence&gt; {c4 } &lt;/corrected sentence&gt; &lt;erroneous sentence&gt;  $\left\{ \boldsymbol { e } _ { t e s t } \right\}$  &lt;/erroneous sentence&gt; &lt;corrected sentence&gt;</td><td>&quot;system&quot;: You are a grammar correction assistant. The user will give you a sentence with grammatical errors (between &#x27;&lt;erroneous sentence&gt;&#x27; and &#x27;&lt;/erroneous sentence&gt;&#x27;). You need to correct the sentence (between &#x27;&lt;corrected sentence&gt; and &#x27;&lt;/corrected sentence&gt;&#x27;). Requirements: 1. Make as few changes as possible. 2. Make sure the sentence has the same meaning as the original sentence. 3. If there is no error, just output &#x27;No errors found&#x27;. &quot;user&quot;: &lt;erroneous sentence&gt;  $\{ e _ { 1 } \}$  &lt;/erroneous sentence&gt; &quot;assistant&quot;: &lt;corrected sentence&gt; {c1 } &lt;/corrected sentence&gt; &quot;user&quot;: &lt;erroneous sentence&gt;  $\{ e _ { 2 } \}$  &lt;/erroneous sentence&gt; &quot;assistant&quot;: &lt;corrected sentence&gt; {c2} &lt;/corrected sentence&gt; &quot;user&quot;: &lt;erroneous sentence&gt; {e3} &lt;/erroneous sentence&gt; &quot;assistant&quot;: &lt;corrected sentence&gt; {c3 } &lt;/corrected sentence&gt; &quot;user&quot;: &lt;erroneous sentence&gt; {e4} &lt;/erroneous sentence&gt; &quot;assistant&quot;: &lt;corrected sentence&gt;  $\{ c _ { 4 } \}$  &lt;/corrected sentence&gt; &quot;user&quot;: &lt;erroneous sentence&gt; {etest } &lt;/erroneous sentence&gt;</td></tr></table>

Table 2: Prompts we use. e and c denote the erroneous and corrected sentences of in-context examples or test samples respectively.

Given the number of dependency labels $d ,$ the dependency trees will be represented into polynomials recursively on two variable set: $\boldsymbol { X } ~ =$ $\{ x _ { 1 } , x _ { 2 } . . . x _ { d } \}$ and $Y = \{ y _ { 1 } , y _ { 2 } , . . . y _ { d } \}$ . In the dependency tree, for each leaf $n ^ { l }$ with label l, the corresponding polynomial is $P ( n ^ { l } , X , Y ) = x _ { l }$ Then, for each non-leaf $m _ { l }$ with label l, the corresponding polynomial is $P ( m ^ { l } , X , Y ) = y _ { l } +$ $\textstyle \prod _ { i = 1 } ^ { \bar { k } } P ( n _ { i } , \bar { X } , \bar { Y } )$ , where $n _ { 1 } , . . . , n _ { k }$ are all child nodes of $m _ { l }$ . In this way, the polynomial of the root node is regarded as the polynomial representation of a tree.

To compute similarity more conveniently, for each term $c x _ { 1 } ^ { e _ { x _ { 1 } } } x _ { 2 } ^ { e _ { x _ { 2 } } } . . . x _ { d } ^ { \ ' e _ { x _ { d } } } y _ { 1 } ^ { e _ { y _ { 1 } } } y _ { 2 } ^ { e _ { y _ { 2 } } } . . . y _ { d } ^ { e _ { y _ { d } } }$ in the dependency polynomial, we write it as a term vector with 2d + 1 entries:

$$
\boldsymbol { t } = [ e _ { x _ { 1 } } , e _ { x _ { 2 } } , . . . , e _ { x _ { d } } , e _ { y _ { 1 } } , e _ { y _ { 2 } } , . . . , e _ { y _ { d } } , c ] ,
$$

where each entry represents the exponent of the corresponding variable. In this way, a polynomial $P$ can be written as a set of term vectors $\nu _ { P }$ . Then, we compute the distance between two polynomials as:

$$
d ( P , Q ) = \frac { \displaystyle { \sum _ { s \in \mathcal { V } _ { P } } \operatorname* { m i n } _ { t \in \mathcal { V } _ { Q } } \| s - t \| _ { 1 } + \sum _ { t \in \mathcal { V } _ { Q } } \operatorname* { m i n } _ { s \in \mathcal { V } _ { P } } \| s - t \| _ { 1 } } } { \displaystyle { | \mathcal { V } _ { P } \mid + | \mathcal { V } _ { Q } | } } ,\tag{1}
$$

where $\| s - t \| _ { 1 }$ denotes the Manhattan distance (Craw, 2017) between term vector s and t.

## 4 Methodology

## 4.1 In-context Learning Workflow for GEC

Based on LLMs, our ungrammatical-syntax-based example selection and few-shot ICL workflow is illustrated in Figure 1. Specially, when faced with a test input, we search through the training data to find the best example(s) for in-context learning. Then, both the source (erroneous) and the target (corrected) sentences of the example(s) are inserted into the prompt as demonstrations, with the test sample concatenated at the end. In this way, LLMs can learn the GEC task from the demonstrations and perform better correction on the test input. In this framework, a set of high-quality in-context examples are crucial to lead LLMs to a better performance. Prompts used in this work are shown in Table 2.

## 4.2 Ungrammatical-syntax-based Selection

We parse all source sentences in both training and test data with GOPar introduced in Section 3.1. Then, for each test input, we search the training data to find most syntactically similar examples with the help of Tree Kernel or Polynomial Distance. These examples will serve as in-context demonstrations in the prompts shown in Table 2.

Weighting Ungrammatical Nodes with Polynomial Distance We hypothesize that LLMs benefit more from similar grammatical errors, and error nodes with similar neighboring syntactic structure lead to similar error patterns. Therefore, assigning higher weights to ungrammatical nodes can select examples with error patterns closer to the test sample. Hence, besides the original Polynomial Distance algorithm, we also explore a weighted version. When computing the Manhattan distance between two term vectors, we assign a higher weight to entries corresponding to labels with error information ("S", "R" and "M"). In our experiment, as a preliminary attempt, we set the weight to 2.

## 4.3 Two-stage Selection

In previous works, a two-stage select-then-rank strategy performs well in in-context learning (Wu et al., 2023; Agrawal et al., 2023). To be specific, a fast and general method is used to filter out most of the not-so-relevant instances from training data and get a much smaller candidate set with high quality, which is called selection. After that, a specific and powerful method is used to rank the instances in the candidate set and obtain the top-k best training instances, which is called ranking. Motivated by this, we also design a two-stage sample selection mechanism for GEC.

Stage 1: BM25/BERT Selection First, we explore selection with BM25 or BERT representation to obtain candidate examples, and the size of candidate set is 1000 in our experiment.

BM25 (Robertson et al., 1994) is a widely-used retrieval algorithm based on term frequency, inverse document frequency and length normalization. Many recent works regard BM25 as a strong baseline for in-context example selection (Agrawal et al., 2023; Li et al., 2023a). In our work, we take the input test sample as the query and source sentences of all training data as the document.

BERT Representation Li et al. (2023a) make use of SentenceBERT (Reimers and Gurevych, 2019) to get sentence representations and then compared similarities of sentences. For simplicity, we adopt the more frequently-used BERT (Devlin et al., 2019) instead. In our work, we take the BERT representation of the [CLS] token as the representation of the sentence. Then we compute the cosine similarities between the representations of the input test sentence and all source sentences in the training data.

For comparison, we also experiment on singlestage BM25 and BERT representation selection, which serve as baselines in Section 5.

Stage 2: Ungrammatical-syntax-based Ranking Further, we employ ranking via syntactic similarity computing with Tree Kernel or Polynomial Distance, to obtain the best k matching examples from the candidate set.

## 5 Experimental Results

## 5.1 Datasets and Evaluation Metrics

We carry out experiments on English GEC datasets. Since no model training is involved, most largescale GEC data is unnecessary, while the data quality matters for example selection. Thus in this work, we only use the relatively small but high-quality Write&Improve+LOCNESS (W&I+LOCNESS) (Bryant et al., 2019) as the training data.

For evaluation, we report P (Precision), R (Recall) and $\mathrm { F _ { 0 . 5 } }$ results on BEA-19 test set (Bryant et al., 2019) evaluated by ERRANT (Bryant et al., 2017) and on CoNLL-14 test set (Ng et al., 2014) evaluated by M2Scorer (Dahlmeier and Ng, 2012). We primarily compare the $\mathrm { F _ { 0 . 5 } }$ among different methods, which shows the comprehensive performance of models on GEC.

Statistics of datasets mentioned above are shown in Table 3.
<table><tr><td>Dataset</td><td>#Sentences</td><td>%Error</td><td>Usage</td></tr><tr><td>W&amp;I+LOCNESS</td><td>34,308</td><td>66</td><td>Demonstration</td></tr><tr><td>BEA-19-Test</td><td>4,477</td><td>-</td><td>Testing</td></tr><tr><td>CoNLL-14-Test</td><td>1,312</td><td>72</td><td>Testing</td></tr></table>

Table 3: Statistics of GEC datasets used in this work. #Sentences refers to the number of sentences.%Error refers to the percentage of erroneous sentences.

## 5.2 Large Language Models

We use two mainstream LLM series: LLaMA-2 (Touvron et al., 2023) and GPT-3.5 (OpenAI, 2023) for experiment.

For LLaMA-2, we use llama-2-7b-chat and llama-2-13b-chat with 7B and 13B parameters respectively. For GPT-3.5, we use the official gpt-3.5-turbo API for inference.

For the sake of reproductivity, we turn off the sampling and set the temperature to zero for all these models we use.

## 5.3 Results

Experimental results are shown in Table 4. With different LLMs and on both datasets, our ungrammatical-syntax-based selection strategy obviously outperforms conventional methods. On BEA-2019 data, the method with first BM25 selection and then Tree Kernel ranking improves the performance by 3.7, 4.6 and 2.4 F<sub>0.5</sub> points, using llama-2-7b-chat, llama-2-13b-chat and gpt-3.5-turbo respectively.

Performance of Tree Kernel When applied as a single-stage method, the Tree Kernel similarity performs poorly and even achieves a lower $\mathrm { F _ { 0 . 5 } }$ score than conventional baselines. However, with the help of a preliminary selection stage, it improves by a margin of about 2 to 3 percentage points, and even achieves the highest $\mathrm { F _ { 0 . 5 } }$ score on BEA-2019 data with LLaMA-2.

<table><tr><td rowspan="3">I</td><td rowspan="3">ⅡI</td><td colspan="9">BEA-2019</td><td colspan="8">CoNLL-2014</td></tr><tr><td colspan="2">LLaMA-2-7B</td><td></td><td colspan="2">LLaMA-2-13B</td><td></td><td colspan="2">GPT-3.5-turbo</td><td colspan="2">LLaMA-2-7B</td><td colspan="2"></td><td colspan="2">LLaMA-2-13B</td><td colspan="2">GPT-3.5-turbo</td></tr><tr><td>P</td><td>R</td><td> ${ \bf F } _ { 0 . 5 }$ </td><td>P</td><td>R</td><td> ${ \bf F } _ { 0 . 5 }$ </td><td>P R</td><td> ${ \bf F } _ { 0 . 5 }$ </td><td>P</td><td>R</td><td> ${ \bf F } _ { 0 . 5 }$ </td><td>P</td><td>R</td><td> ${ \bf F } _ { 0 . 5 }$ </td><td>P</td><td>R</td><td> ${ \bf F } _ { 0 . 5 }$ </td></tr><tr><td rowspan="6"></td><td>Rand.</td><td>50.1</td><td>57.7</td><td>51.5</td><td>49.0</td><td>61.2</td><td>51.0 47.0</td><td>70.4</td><td>50.3</td><td>59.4</td><td>48.8</td><td>56.9</td><td>58.6</td><td>51.3</td><td>57.0</td><td>56.5</td><td>59.9</td><td>57.1</td></tr><tr><td>BM25</td><td>50.9</td><td>58.2</td><td>52.2</td><td>51.6</td><td>61.1</td><td>53.3</td><td>46.8 69.6</td><td>50.1</td><td>59.7</td><td>47.7</td><td>56.8</td><td>59.3</td><td>50.1</td><td>57.2</td><td>56.6</td><td>60.8</td><td>57.4</td></tr><tr><td>BERT</td><td>50.7</td><td>56.8</td><td>51.8</td><td>51.0</td><td>61.2</td><td>52.8</td><td>47.6 70.0</td><td>50.9</td><td>58.6</td><td>45.4</td><td>55.4</td><td>60.1</td><td>52.0</td><td>58.3</td><td>56.0</td><td>60.8</td><td>56.9</td></tr><tr><td>T. K.</td><td>50.0</td><td>-57.0</td><td>51.2</td><td>52.5</td><td>59.0</td><td>53.6</td><td>47.2 69.8</td><td>50.5</td><td>57.9</td><td>47.5</td><td>55.5</td><td>61.8</td><td>48.0</td><td>58.5</td><td>57.3</td><td>60.3</td><td>57.9</td></tr><tr><td>Poly.</td><td>53.1</td><td>57.9</td><td>54.0</td><td>52.9</td><td>60.2</td><td>54.3</td><td>49.5 70.0</td><td>52.6</td><td>59.5</td><td>49.5</td><td>57.2</td><td>61.7</td><td>51.8</td><td>59.4</td><td>58.2</td><td>59.9</td><td>58.6</td></tr><tr><td>W. Poly.</td><td>53.2</td><td>58.2</td><td>54.2</td><td>53.4</td><td>60.5</td><td>54.7</td><td>50.3</td><td>69.6</td><td>53.2</td><td>60.1 49.2</td><td>57.5</td><td>61.6</td><td>52.3</td><td>59.5</td><td>58.4</td><td>60.5</td><td>58.8</td></tr><tr><td rowspan="3">BM25</td><td>T. K.</td><td>55.1</td><td>55.9</td><td>55.2</td><td>54.9 58.7</td><td>55.6</td><td>49.7</td><td>69.3</td><td>52.7</td><td>62.2</td><td>45.7</td><td>58.0</td><td>61.9</td><td>47.3</td><td>58.3</td><td>58.3</td><td>59.7</td><td>58.6</td></tr><tr><td>Poly.</td><td>51.2</td><td>57.1</td><td>52.3</td><td>50.9</td><td>59.8</td><td>52.5 48.8</td><td>69.5</td><td>51.9</td><td>62.1</td><td>47.7</td><td>58.6</td><td>60.9</td><td>49.8</td><td>58.3</td><td>57.2</td><td>59.7</td><td>57.7</td></tr><tr><td>W. Poly.</td><td>54.4</td><td>57.4</td><td>55.0 54.0</td><td>59.7</td><td>55.0</td><td>49.3</td><td>69.8</td><td>52.4</td><td>61.4</td><td>47.7</td><td>58.1</td><td>60.8</td><td>50.4</td><td>58.4</td><td>57.6</td><td>60.4</td><td>58.1</td></tr><tr><td rowspan="3">BERT</td><td>T. K.</td><td>53.6</td><td>56.0</td><td>54.1</td><td>53.7</td><td>59.3</td><td>54.7</td><td>50.0 69.7</td><td>53.0</td><td>60.7</td><td>46.3</td><td>57.1</td><td>60.8</td><td>49.9</td><td>58.3</td><td>57.6</td><td>59.2</td><td>57.9</td></tr><tr><td>Poly.</td><td>53.3</td><td>57.2</td><td>54.0</td><td>53.8</td><td>60.4</td><td>55.0</td><td>49.0 69.5</td><td>52.1</td><td>60.5</td><td>47.6</td><td>57.4</td><td>59.8</td><td>50.8</td><td>57.8</td><td>57.6</td><td>60.7</td><td>58.2</td></tr><tr><td>W. Poly.</td><td>53.8</td><td>57.4</td><td>54.5</td><td>54.2</td><td>60.7</td><td>55.4</td><td>49.9 69.7</td><td>52.9</td><td>61.0</td><td>48.3</td><td>57.9</td><td>59.8</td><td>51.5</td><td>57.9</td><td>57.3</td><td>60.5</td><td>57.9</td></tr></table>

Table 4: Experimental results under the in-context few-shot setting with 4 examples. I and II denote the first (selection) and second (ranking) stage of the two-stage selection respectively. $" \underline { { \mathbf { \Pi } } } _ { \astrosun } \ "$ means the Stage I is absent and these are single-stage models. "Rand.", "T. K.", "Poly." and "W. Poly." refer to "Random", "Tree Kernel" "Polynomial Distance" and "Weighted Polynomial Distance", respectively. The dashed line separates results of conventional baselines and our proposed methods: the former on the upper side and the latter on the lower side. The best $\mathrm { F _ { 0 . 5 } }$ scores of each group are displayed in bold, and the best $\mathrm { F _ { 0 . 5 } }$ scores of all settings are displayed in underlined bold.

Performance of Polynomial Distance Different from Tree Kernel, Polynomial Distance performs fairly well even without a preliminary selection. Among those single-stage approaches, both polynomial-based methods outperform traditional baselines by an average margin of 2 to 3 percentage points in all cases, which indicates the superiority of syntactic similarity on GEC. The weighted version, with a higher weight on labels with error tags, brings a slight improvement in most cases, which shows the effectiveness of error information in GOPar-based dependency trees.

Performance of Two-stage Selection As for Tree Kernel, the two-stage selection strategy consistently boosts performance, whether using BM25 or BERT representation as the preliminary selection approach. But for Polynomial Distance, the two-stage selection brings less improvement and even fails to improve performance in some cases. We leave it for future research.

Comparison with State-of-the-art We compare our method with previous supervised approaches, as shown in Table 5. Even with the help of ungrammatical-syntax-based selection, the GEC performance of LLMs is still far from state-of-theart. We look forward to more advanced foundation

<table><tr><td>System</td><td>CoNLL-14 BEA-19</td><td></td></tr><tr><td>GECToR (Omelianchuk et al., 2020)</td><td>65.3</td><td>72.4</td></tr><tr><td>SynGEC (Zhang et al., 2022b)</td><td>66.7</td><td>72.0</td></tr><tr><td>T5 xxl (Rothe et al., 2021)</td><td>68.9</td><td>75.9</td></tr><tr><td>ChatGPT zero-shot CoT (Fang et al., 2023b)</td><td>51.7</td><td>36.1</td></tr><tr><td>LLaMA-2-7B with BM25 + Tree Kernel (ours)</td><td>55.2</td><td>58.0</td></tr><tr><td>LLaMA-2-13B with Weighed Polynomial (ours)</td><td>54.7</td><td>59.5</td></tr></table>

Table 5: Results of state-of-the-art GEC systems and our proposed methods on two datasets. The evaluation metric is $\mathrm { F _ { 0 . 5 } }$

models in the future.

## 6 Model Analysis

## 6.1 Experiments with Different Numbers of In-context Examples

To explore the consistency and robustness of our methods, we conduct 1-shot, 2-shot, 4-shot and 8-shot experiments on llama-2-7b-chat. The results on BEA-2019 test set and CoNLL-2014 test set are shown in Table 6 and 7 respectively.

When there is only one example, the model performs relatively poor. When the number of examples comes to two, the performance improves significantly. Then, further increasing the number of examples brings a slight but consistent performance gain.

When the number of examples is small, the superiority of syntax-based methods compared with those conventional is evident. When the number of examples increases, conventional baselines improve a lot while syntax-based methods gain relatively less, which shows a marginal benefit. But syntax-based methods always secure the highest score, indicating the consistency of their advantages, especially under settings of fewer shots.

<table><tr><td rowspan="2">I</td><td rowspan="2">ⅡI</td><td colspan="3">1-shot</td><td colspan="3">2-shot</td><td colspan="3">4-shot</td><td colspan="3">8-shot</td></tr><tr><td>P</td><td>R</td><td> $\mathbf { F } _ { 0 . 5 }$ </td><td>P</td><td>R</td><td> $\mathbf { F } _ { 0 . 5 }$ </td><td>P</td><td>R</td><td> $\mathbf { F } _ { 0 . 5 }$ </td><td>P</td><td>R</td><td> $\mathbf { F } _ { 0 . 5 }$ </td></tr><tr><td rowspan="7"></td><td>Rand.</td><td>47.3</td><td>29.8</td><td>42.3</td><td>49.6</td><td>50.9</td><td>49.8</td><td>50.1</td><td>57.7</td><td>51.5</td><td>52.2</td><td>58.8</td><td>53.4</td></tr><tr><td>BM25</td><td>48.4</td><td>35.8</td><td>45.2</td><td>50.4</td><td>53.1</td><td>50.9</td><td>50.9</td><td>58.2</td><td>52.2</td><td>52.5</td><td>59.0</td><td>53.7</td></tr><tr><td>BERT</td><td>47.3</td><td>33.8</td><td>43.8</td><td>50.0</td><td>51.4</td><td>50.2</td><td>50.7</td><td>56.8</td><td>51.8</td><td>53.6</td><td>59.3</td><td>54.6</td></tr><tr><td> $\bar { \mathrm { ~ T . ~ K . ~ } }$ </td><td>47.1</td><td>27.4</td><td>41.2</td><td>49.0</td><td>53.2</td><td>49.8</td><td>50.0</td><td>57.0</td><td>51.2</td><td>53.6</td><td>55.9</td><td>54.0</td></tr><tr><td>Poly.</td><td>50.1</td><td>31.5</td><td>44.8</td><td>53.9</td><td>51.9</td><td>53.5</td><td>53.1</td><td>57.9</td><td>54.0</td><td>54.3</td><td>58.3</td><td>55.1</td></tr><tr><td>W. Poly.</td><td>50.4</td><td>31.5</td><td>45.0</td><td>52.7</td><td>51.5</td><td>52.4</td><td>53.2</td><td>58.2</td><td>54.2</td><td>53.3</td><td>58.0</td><td>54.2</td></tr><tr><td rowspan="3">BM25</td><td>T. K.</td><td>51.7</td><td>37.5</td><td>48.1</td><td>53.3</td><td>53.8</td><td>53.4</td><td>55.1</td><td>55.9</td><td>55.2</td><td>57.2</td><td>55.6</td><td>56.9</td></tr><tr><td>Poly.</td><td>51.3</td><td>36.6</td><td>47.5</td><td>52.9</td><td>54.5</td><td>53.2</td><td>51.2</td><td>57.1</td><td>52.3</td><td>55.5</td><td>56.9</td><td>55.8</td></tr><tr><td>W. Poly.</td><td>51.1</td><td>36.6</td><td>47.4</td><td>52.8</td><td>54.7</td><td>53.2</td><td>54.4</td><td>57.4</td><td>55.0</td><td>56.3</td><td>57.0</td><td>56.4</td></tr><tr><td rowspan="3">BERT</td><td>T. K.</td><td>50.7</td><td>35.6</td><td>46.8</td><td>53.3</td><td>52.4</td><td>53.1</td><td>53.6</td><td>56.0</td><td>54.1</td><td>57.1</td><td>57.0</td><td>57.1</td></tr><tr><td>Poly.</td><td>50.9</td><td>35.5</td><td>46.9</td><td>52.1</td><td>53.4</td><td>52.4</td><td>53.3</td><td>57.2</td><td>54.0</td><td>55.5</td><td>58.2</td><td>56.1</td></tr><tr><td>W. Poly.</td><td>50.6</td><td>35.7</td><td>46.7</td><td>52.1</td><td>53.8</td><td>52.4</td><td>53.8</td><td>57.4</td><td>54.5</td><td>56.5</td><td>57.8</td><td>56.7</td></tr></table>

Table 6: Results of llama-2-7b-chat with different numbers of shots on BEA-19 test set.
<table><tr><td rowspan="2">I</td><td rowspan="2">II</td><td colspan="3">1-shot</td><td colspan="3">2-shot</td><td colspan="3">4-shot</td><td colspan="3">8-shot</td></tr><tr><td>P</td><td>R</td><td> ${ \bf F } _ { 0 . 5 }$ </td><td>P</td><td>R</td><td> ${ \bf F } _ { 0 . 5 }$ </td><td>P</td><td>R</td><td> ${ \bf F } _ { 0 . 5 }$ </td><td>P</td><td>R</td><td> ${ \bf F } _ { 0 . 5 }$ </td></tr><tr><td rowspan="7"></td><td>Rand.</td><td>54.7</td><td>21.2</td><td>41.6</td><td>58.0</td><td>42.3</td><td>54.0</td><td>59.1</td><td>48.2</td><td>56.6</td><td>60.9</td><td>49.3</td><td>58.2</td></tr><tr><td>BM25</td><td>55.7</td><td>25.2</td><td>44.9</td><td>57.5</td><td>42.5</td><td>53.7</td><td>59.7</td><td>47.7</td><td>56.8</td><td>60.4</td><td>47.8</td><td>57.4</td></tr><tr><td>BERT</td><td>55.9</td><td>22.5</td><td>43.1</td><td>58.0</td><td>39.0</td><td>52.8</td><td>58.6</td><td>45.4</td><td>55.4</td><td>60.7</td><td>48.0</td><td>57.6</td></tr><tr><td>T.K.</td><td>51.5</td><td>17.7</td><td>37.3</td><td>57.9</td><td>44.1</td><td>54.5</td><td>57.9</td><td>47.5</td><td>55.5</td><td>61.7</td><td>47.0</td><td>58.1</td></tr><tr><td>Poly.</td><td>54.7</td><td>21.3</td><td>41.6</td><td>58.6</td><td>41.4</td><td>54.1</td><td>59.6</td><td>49.5</td><td>57.2</td><td>60.5</td><td>48.5</td><td>57.6</td></tr><tr><td>W. Poly.</td><td>52.3</td><td>20.6</td><td>40.0</td><td>58.6</td><td>42.9</td><td>54.6</td><td>60.1</td><td>49.2</td><td>57.5</td><td>61.0</td><td>49.8</td><td>58.4</td></tr><tr><td>T. K.</td><td>57.9</td><td>27.3</td><td>47.3</td><td>60.5</td><td>44.7</td><td>56.5</td><td>62.2</td><td>45.7</td><td>58.0</td><td>62.5</td><td>45.3</td><td>58.1</td></tr><tr><td rowspan="3">BM25</td><td>Poly.</td><td>57.2</td><td>25.1</td><td>45.5</td><td>60.5</td><td>43.5</td><td>56.2</td><td>62.1</td><td>47.7</td><td>58.6</td><td>61.6</td><td>46.7</td><td>57.9</td></tr><tr><td>W. Poly.</td><td>57.1</td><td>24.6</td><td>45.1</td><td>60.7</td><td>43.7</td><td>56.3</td><td>61.4</td><td>47.7</td><td>58.1</td><td>62.7</td><td>47.7</td><td>59.0</td></tr><tr><td>T. K.</td><td>58.3</td><td>25.1</td><td>46.1</td><td>59.9</td><td>42.7</td><td>55.4</td><td>60.7</td><td>46.3</td><td>57.1</td><td>63.1</td><td>46.2</td><td>58.8</td></tr><tr><td rowspan="3">BERT</td><td>Poly.</td><td>56.0</td><td>24.7</td><td>44.7</td><td>59.3</td><td>43.8</td><td>55.4</td><td>60.5</td><td>47.6</td><td>57.4</td><td>61.9</td><td>47.7</td><td>58.4</td></tr><tr><td>W. Poly.</td><td>57.1</td><td>24.9</td><td>45.4</td><td>59.5</td><td>44.6</td><td>55.8</td><td>61.0</td><td>48.3</td><td>57.9</td><td>62.8</td><td>47.8</td><td>59.1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 7: Results of llama-2-7b-chat with different numbers of shots on CoNLL-14 test set.

## 6.2 Ungrammatical Parser or Standard Parser?

To explore the affect of different parsers on model performance, we also experiment with Stanford Parser (Dozat and Manning, 2017), which is a widely-used conventional parser. We use stanford-corenlp-4.5.5 as the model for parsing and run Stanford Parser with stanfordcorenlp <sup>2</sup>, which is a Python wrapper for Stanford CoreNLP. For a clear demonstration, an example is illustrated in Figure 3 to show the different parsing results of GOPar and Stanford Parser.

![](images/90502b553219574f68a1c34212429e1869562e9ef93f047ec0369b996b7aea5f.jpg)

![](images/a639e9c9828409062a0cca8118115c622d065053e5a39c24ac9772703b8d2df3.jpg)  
Figure 3: An example of parsing tree by GOPar and Stanford Parser.

<table><tr><td></td><td>Source (Erroneous Sentence)</td><td>Target (Corrected Sentence)</td></tr><tr><td>Input</td><td>So, they have to also prepare mentally.</td><td>So, they also have to prepare mentally.</td></tr><tr><td>BM25</td><td>Also you can see how they prepare your food in front of you.</td><td>Also, you can see how they prepare your food in front of you.</td></tr><tr><td>T. K.</td><td>Nowadays people get around constantly.</td><td>Nowadays, people are constantly on the move.</td></tr><tr><td>BM25 + T. K.</td><td>that have limitation also there.</td><td>There are also limitations there.</td></tr></table>

Table 8: A one-shot example showing the tree kernel method benefiting from the two-stage selection.

<table><tr><td rowspan="2">I</td><td rowspan="2">II</td><td colspan="3">GOPar</td><td colspan="3">Stanford Parser</td></tr><tr><td>P</td><td>R</td><td> $\mathbf { F } _ { 0 . 5 }$ </td><td>P</td><td>R</td><td> $\mathbf { F } _ { 0 . 5 }$ </td></tr><tr><td>BM25 BERT</td><td>T. K.</td><td>50.0 55.1 53.6</td><td>57.0 55.9 56.0</td><td>51.2 55.2 54.1</td><td>49.6 51.8</td><td>56.4 56.2</td><td>50.8 52.7 51.8</td></tr></table>

Table 9: Results on BEA-2019 test set with 4 examples, using GOPar and Stanford Parser respectively.

The experimental results comparing GOPar and Stanford Parser on BEA-2019 test set are shown in Table 9. Here, we adopt llama-2-7b-chat as the LLM and Tree Kernel as the ranking method.

Without using the two-stage selection, Stanford Parser performs slightly worse than GOPar. With the two-stage selection, GOPar gains more improvement than Stanford Parser and outperforms it by a margin of more than 2 points. This indicates GOPar is more suitable for GEC, and its superiority lies in two aspects. First, it performs more robust on ungrammatical sentences (e.g., it correctly recognizes the prepositional object "pool" in the sentence shown in Figure 3 while Stanford Parser fails to). Second, it provides extra information about the grammatical errors (e.g., the Missing error in Figure 3).

## 6.3 Effect of Two-stage Selection

In order to find out how the two-stage strategy benefits the Tree Kernel method, we conduct a case study and compare three selection settings: BM25 only ("BM25"), Tree Kernel only ("T.K.") and Tree Kernel after the BM25 selection ("BM25+T.K.").

In the example shown in Table 8, the input sentence is ungrammatical in word order. "BM25" selects a sentence with a punctuation missing error that is similar to the input sample in words ("also", "they" and "prepare"). "T. K." selects a sentence with an improper expression "get around constantly" which is similar to "prepare mentally" in syntactic structure but has little to do with the grammatical errors. "BM25 + T. K." selects a sentence that is similar to the input sample both in word occurrences ("also" and "have") and in error form (improper word order).

Since similar words are more likely to form similar errors, with the help of a preliminary selection, Tree Kernel can select from a more relative candidate set, leading to a better example selection involving both word and syntactic similarity in erroneous constituents. Moreover, it also shows the disadvantage of conventional selection method BM25 on GEC, which cannot effectively select examples similar in syntax.

## 7 Conclusion

In this work, we make use of two conventional treebased syntactic similarity algorithms and the selectthen-rank two-stage framework to select in-context examples for the GEC task. Empirical results show that our syntax-based in-context example selection method is effective for GEC. We call on the NLP community to pay more attention to the help of syntactic information for many other syntax-related tasks besides GEC.

## Acknowledgements

This work is supported by the National Natural Science Foundation of China (62076008) and the Key Project of Natural Science Foundation of China (61936012).

## Limitations

First, we only experiment on English datasets. The performance of our method on other languages requires further exploration. Second, besides dependency tree, constituent tree is also worth trying. However, unfortunately, we do not have access to GEC-oriented constituent trees (Zhang and Li, 2022) at the time of writing this paper. Third, many previous outstanding methods of both in-context example selection and tree similarity computation have not been explored in our work. Fourth, due to limited time, we do not explore the effect of the size of candidate set after the selection stage and the choice of weight of ungrammatical nodes in the Polynomial Distance method. There may exist a better size than the values we use in our experiments. Fifth, except for the Stanford Parser (which splits sentences itself), our experiments do not split instances with multiple sentences into single-sentence instances. Some instances in GEC datasets contains more than one sentence. Directly parsing these instances without splitting them into single sentences may hurt the parsing performance and lead to unreliable results. Last, we do not treat in-context examples as a whole, which might lead to a lower level of diversity of examples and sub-optimal performance, as addressed in Ye et al. (2023) and Gupta et al. (2023).

## Ethics Statement

Use of Scientific Artifacts. We make use of GOPar provided by Zhang et al. (2022b), which is publicly available based on the MIT license <sup>3</sup>.

About Computational Budget. Computation time is shown in Table 10.

<table><tr><td>Method</td><td>Time</td></tr><tr><td>BM25</td><td>440</td></tr><tr><td>BERT</td><td>4500</td></tr><tr><td>Tree Kernel</td><td>3600</td></tr><tr><td>Polynomial Distance</td><td> 3200</td></tr></table>

Table 10: Computation time of different methods on BEA-19 test set, all in seconds. BERT runs on an NVIDIA GeForce RTX 2080 Ti and the other three run on an Intel<sup>®</sup> Xeon<sup>®</sup> Gold 5218 CPU.

About Reproducibility. All the experiments are completely reproducible since we disable sampling and set the temperature to zero for all LLMs we use, as discussed in Section 5.2.

## References

Sweta Agrawal, Chunting Zhou, Mike Lewis, Luke Zettlemoyer, and Marjan Ghazvininejad. 2023. Incontext examples selection for machine translation. In Findings of the Association for Computational Linguistics: ACL 2023, pages 8857–8873, Toronto, Canada. Association for Computational Linguistics.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Christopher Bryant, Mariano Felice, Øistein E. Andersen, and Ted Briscoe. 2019. The BEA-2019 shared task on grammatical error correction. In Proceedings of the Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications, pages 52–75, Florence, Italy. Association for Computational Linguistics.

Christopher Bryant, Mariano Felice, and Ted Briscoe. 2017. Automatic annotation and evaluation of error types for grammatical error correction. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 793–805, Vancouver, Canada. Association for Computational Linguistics.

Christopher Bryant, Zheng Yuan, Muhammad Reza Qorib, Hannan Cao, Hwee Tou Ng, and Ted Briscoe. 2022. Grammatical error correction: A survey of the state of the art. Computational Linguistics, pages 1–59.

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. 2023. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712.

Hejing Cao and Dongyan Zhao. 2023. Leveraging denoised Abstract Meaning Representation for grammatical error correction. In Findings of the Association for Computational Linguistics: ACL 2023, pages 7180–7188, Toronto, Canada. Association for Computational Linguistics.

Dhivya Chandrasekaran and Vijay Mago. 2021. Evolution of semantic similarity—a survey. ACM Computing Surveys (CSUR), 54(2):1–37.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2023. Palm: Scaling language modeling with pathways. Journal ofMachine Learning Research, 24(240):1–113.

Michael Collins and Nigel Duffy. 2002. New ranking algorithms for parsing and tagging: Kernels over discrete structures, and the voted perceptron. In Proceedings ofthe 40th Annual Meeting ofthe Association for Computational Linguistics, pages 263–270, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Susan Craw. 2017. Manhattan distance. Encyclopedia ofMachine Learning and Data Mining, pages 790– 791.

Daniel Dahlmeier and Hwee Tou Ng. 2012. Better evaluation for grammatical error correction. In Proceedings ofthe 2012 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 568–572, Montréal, Canada. Association for Computational Linguistics.

Davi de Castro Reis, Paulo Braz Golgher, Altigran Soares da Silva, and Alberto H. F. Laender. 2004. Automatic web news extraction using tree edit distance. In The Web Conference.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. 2022. A survey for in-context learning. arXiv preprint arXiv:2301.00234.

Timothy Dozat and Christopher D. Manning. 2017. Deep biaffine attention for neural dependency parsing. In International Conference on Learning Representations.

Tao Fang, Jinpeng Hu, Derek F. Wong, Xiang Wan, Lidia S. Chao, and Tsung-Hui Chang. 2023a. Improving grammatical error correction with multimodal feature integration. In Findings of the Association for Computational Linguistics: ACL 2023, pages 9328–9344, Toronto, Canada. Association for Computational Linguistics.

Tao Fang, Shu Yang, Kaixin Lan, Derek F Wong, Jinpeng Hu, Lidia S Chao, and Yue Zhang. 2023b. Is chatgpt a highly fluent grammatical error correction system. A comprehensive evaluation. ArXiv, abs/2304.01746.

Wael H Gomaa, Aly A Fahmy, et al. 2013. A survey of text similarity approaches. international journal of Computer Applications, 68(13):13–18.

Shivanshu Gupta, Matt Gardner, and Sameer Singh. 2023. Coverage-based example selection for incontext learning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 13924–13950, Singapore. Association for Computational Linguistics.

Marcin Junczys-Dowmunt, Roman Grundkiewicz, Shubha Guha, and Kenneth Heafield. 2018. Approaching neural grammatical error correction as a low-resource machine translation task. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 595–606, New Orleans, Louisiana. Association for Computational Linguistics.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199– 22213.

Yuquan Le, Zhi-Jie Wang, Zhe Quan, Jiawei He, and Bin Yao. 2018. Acv-tree: A new method for sentence similarity modeling. In Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, IJCAI-18, pages 4137–4143. International Joint Conferences on Artificial Intelligence Organization.

Xiaonan Li, Kai Lv, Hang Yan, Tianyang Lin, Wei Zhu, Yuan Ni, Guotong Xie, Xiaoling Wang, and Xipeng Qiu. 2023a. Unified demonstration retriever for incontext learning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4644–4668, Toronto, Canada. Association for Computational Linguistics.

Yinghao Li, Xuebo Liu, Shuo Wang, Peiyuan Gong, Derek F. Wong, Yang Gao, Heyan Huang, and Min Zhang. 2023b. TemplateGEC: Improving grammatical error correction with detection template. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 6878–6892, Toronto, Canada. Association for Computational Linguistics.

Pengyu Liu, Tinghao Feng, and Rui Liu. 2022. Quantifying syntax similarity with a polynomial representation of dependency trees. arXiv preprint arXiv:2211.07005.

Mengsay Loem, Masahiro Kaneko, Sho Takase, and Naoaki Okazaki. 2023. Exploring effectiveness of GPT-3 in grammatical error correction: A study on performance and controllability in prompt-based methods. In Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023), pages 205–219, Toronto, Canada. Association for Computational Linguistics.

Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837.

Alessandro Moschitti. 2006. Efficient convolution kernels for dependency and constituent syntactic trees. In Machine Learning: ECML 2006, pages 318–329, Berlin, Heidelberg. Springer Berlin Heidelberg.

Hwee Tou Ng, Siew Mei Wu, Ted Briscoe, Christian Hadiwinoto, Raymond Hendy Susanto, and Christopher Bryant. 2014. The CoNLL-2014 shared task on grammatical error correction. In Proceedings of the Eighteenth Conference on Computational Natural Language Learning: Shared Task, pages 1–14, Baltimore, Maryland. Association for Computational Linguistics.

Kostiantyn Omelianchuk, Vitaliy Atrasevych, Artem Chernodub, and Oleksandr Skurzhanskyi. 2020. GECToR – grammatical error correction: Tag, not rewrite. In Proceedings of the Fifteenth Workshop on Innovative Use ofNLPfor Building Educational

Applications, pages 163–170, Seattle, WA, USA → Online. Association for Computational Linguistics.

OpenAI. 2023. GPT-3.5 API. https://platform. openai.com/docs/models/gpt-3-5.

Masanori Oya. 2020. Syntactic similarity of the sentences in a multi-lingual parallel corpus based on the Euclidean distance of their dependency trees. In Proceedings ofthe 34th Pacific Asia Conference on Language, Information and Computation, pages 225– 233, Hanoi, Vietnam. Association for Computational Linguistics.

¸Saziye Betül Özate¸s, Arzucan Özgür, and Dragomir Radev. 2016. Sentence similarity based on dependency tree kernels for multi-document summarization. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), pages 2833–2838, Portorož, Slovenia. European Language Resources Association (ELRA).

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, and Mike Gatford. 1994. Okapi at trec-3. In Text Retrieval Conference.

Sascha Rothe, Jonathan Mallinson, Eric Malmi, Sebastian Krause, and Aliaksei Severyn. 2021. A simple recipe for multilingual grammatical error correction. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 702–707, Online. Association for Computational Linguistics.

Maksym Tarnavskyi, Artem Chernodub, and Kostiantyn Omelianchuk. 2022. Ensembling and knowledge distilling of large sequence taggers for grammatical error correction. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3842–3852, Dublin, Ireland. Association for Computational Linguistics.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

SVN Vishwanathan, Alexander Johannes Smola, et al. 2004. Fast kernels for string and tree matching. Kernel methods in computational biology, 15(113- 130):1.

Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. 2023. Transformers learn in-context by gradient descent. In International Conference on Machine Learning, pages 35151–35174. PMLR.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837. Curran Associates, Inc.

Xiuyu Wu and Yunfang Wu. 2022. From spelling to grammar: A new framework for Chinese grammatical error correction. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 889–902, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Zhiyong Wu, Yaoxiang Wang, Jiacheng Ye, and Lingpeng Kong. 2023. Self-adaptive in-context learning: An information compression perspective for incontext example selection and ordering. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1423–1436, Toronto, Canada. Association for Computational Linguistics.

Konstantin Yakovlev, Alexander Podolskiy, Andrey Bout, Sergey Nikolenko, and Irina Piontkovskaya. 2023. GEC-DePenD: Non-autoregressive grammatical error correction with decoupled permutation and decoding. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1546–1558, Toronto, Canada. Association for Computational Linguistics.

Lingyu Yang, Hongjia Li, Lei Li, Chengyin Xu, Shutao Xia, and Chun Yuan. 2023. LET: Leveraging error type information for grammatical error correction. In Findings of the Association for Computational Linguistics: ACL 2023, pages 5986–5998, Toronto, Canada. Association for Computational Linguistics.

Jiacheng Ye, Zhiyong Wu, Jiangtao Feng, Tao Yu, and Lingpeng Kong. 2023. Compositional exemplars for in-context learning. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 39818–39833. PMLR.

Ying Zhang, Hidetaka Kamigaito, and Manabu Okumura. 2023. Bidirectional transformer reranker for grammatical error correction. In Findings of the Associationfor Computational Linguistics: ACL 2023, pages 3801–3825, Toronto, Canada. Association for Computational Linguistics.

Yue Zhang and Zhenghua Li. 2022. Csyngec: Incorporating constituent-based syntax for grammatical error correction with a tailored gec-oriented parser. arXiv preprint arXiv:2211.08158.

Yue Zhang, Zhenghua Li, Zuyi Bao, Jiacheng Li, Bo Zhang, Chen Li, Fei Huang, and Min Zhang. 2022a. MuCGEC: a multi-reference multi-source evaluation dataset for Chinese grammatical error correction. In Proceedings of the 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 3118–3130, Seattle, United States. Association for Computational Linguistics.

Yue Zhang, Bo Zhang, Zhenghua Li, Zuyi Bao, Chen Li, and Min Zhang. 2022b. SynGEC: Syntax-enhanced grammatical error correction with a tailored GECoriented parser. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 2518–2531, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.