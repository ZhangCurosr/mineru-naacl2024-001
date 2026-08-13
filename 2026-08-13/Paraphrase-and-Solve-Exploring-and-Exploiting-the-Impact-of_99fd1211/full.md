# Paraphrase and Solve: Exploring and Exploiting the Impact of Surface Form on Mathematical Reasoning in Large Language Models

Yue Zhou<sup>1</sup>∗ Yada Zhu<sup>2</sup> Diego Antognini<sup>2</sup> Yoon Kim<sup>3</sup> Yang Zhang<sup>2</sup>

<sup>1</sup>University of Illinois Chicago <sup>2</sup>MIT-IBM Watson AI Lab, IBM Research <sup>3</sup>MIT CSAIL

yzhou232@uic.edu, yzhu@us.ibm.com

yoonkim@mit.edu, {diego.antognini, yang.zhang2}@ibm.com

## Abstract

This paper studies the relationship between the surface form of a mathematical problem and its solvability by large language models. We find that subtle alterations in the surface form can significantly impact the answer distribution and the solve rate, exposing the language model’s lack of robustness and sensitivity to the surface form in reasoning through complex problems. To improve mathematical reasoning performance, we propose Self-Consistencyover-Paraphrases (SCoP), which diversifies reasoning paths from specific surface forms of the problem. We evaluate our approach on four mathematics reasoning benchmarks over three large language models and show that SCoP improves mathematical reasoning performance over vanilla self-consistency, particularly for problems initially deemed unsolvable. Finally, we provide additional experiments and discussion regarding problem difficulty and surface forms, including cross-model difficulty agreement and paraphrasing transferability, and Variance of Variations (VOV) for language model evaluation.

## 1 Introduction

Despite the impressive performance of large-scale language models (LLMs) across many tasks, their ability to reason through complex problems such as mathematics remains a bottleneck (Rae et al., 2022; Srivastava et al., 2023; Liang et al., 2023); they can solve problems that are challenging for humans but can also struggle with seemingly simple ones. This raises the following question: what factors contribute to the difficulty of a math problem for an LLM?

Specifically, the information in a math problem can be divided into two types. The first is the semantic information, which involves the problem specification and knowledge involved in the math problem. The second is the surface form, i.e., how the questions, assumptions, and constraints are described in the math problem. Intuitively, the semantic information should be the primary determining factor of the difficulty of the math problem, and surface form should only have a marginal impact. This paper investigates the extent to which this is true for LLMs.

In this paper, we explore into the relationship between the problem’s surface form and its solvability by an LLM. Specifically, we follow the self-consistency setting (Wang et al., 2022) to sample multiple answers to the same math problem and compute solve rate as the percentage of correct answers. Our primary finding is that, counterintuitively, subtle alterations in the surface form of a math problem can significantly impact the answer distribution and solve rate. Consider the example in Figure 1, where the left and right panels contain an identical math problem described in two different ways. Despite the no change in problem semantics, the solve rate increases from 5% to 100%, with all reasoning paths leading to the correct answer - what initially appears to be a difficult problem to the language model transforms into an easily solvable one. This phenomenon exposes the language model’s lack of robustness and sensitivity to the surface form in reasoning through complex problems.

Motivated by this finding, we propose to improve the mathematical reasoning performance of the language model by diversifying reasoning paths from specific surface forms of the problem. We leverage the language model’s paraphrasing ability to generate surface forms with identical semantics<sup>1</sup> and propose Self-Consistency-over-Paraphrases (SCoP), which consists of two steps: ❶ For each math problem, generate K paraphrase using an

![](images/7a00cdf6c9e30aadc8d377b680094cac061ff53be01de36f15b587ac495fd6e2.jpg)  
Figure 1: Comparison of the answer distribution and solve rate between surface form variations of a math word problem from GSM8K, when prompted to GPT-3.5-turbo using Self-Consistency, with 40 sampled reasoning paths. Solve rate can vary dramatically between surface forms with equivalent semantics.

LLM; and ❷ Ask the LLM to generate N/K reasoning paths for each paraphrase, and then select the most consistent answer among the N answers. The intuition is that if a problem exhibits a low solve rate and ineffective reasoning paths due to its original surface form, introducing diversity in its surface forms can be beneficial. We also introduced in-context exemplars to the language model when paraphrasing, which are the paraphrases that obtain a solve rate improvement over their original problem, aiming to generate surface forms with the same semantics yet a higher solve rate through language models’ in-context learning abilities (Min et al., 2022; Brown et al., 2020).

We evaluate our approach on four mathematics reasoning benchmarks: GSM8K (Cobbe et al., 2021), AQuA (Ling et al., 2017), MATH (Hendrycks et al., 2021), and MMLU-Math (Hendrycks et al., 2020), over three large language models: LLaMA-2-70b (Touvron et al., 2023), GPT-3.5-turbo and GPT-4 (OpenAI, 2023). Our experiments show that SCoP improves mathematical reasoning performance over vanilla Self-Consistency, particularly for problems initially deemed unsolvable. In additional experiments, we show that the difficulty ranks across language models are positively correlated, with higher agreement within the GPT model family and simpler datasets. We propose Variance of Variations (VOV) as a metric for evaluating language model robustness against surface form variations. Finally, we explain why SCoP can be effective using a data difficulty map based on the entropy of answer distribution and the solve rate. Our code is publicly available.<sup>2</sup>

## 2 Problem Difficulty and Surface Forms

In this section, we present our pilot study of the impact of surface form on LLMs’ ability to solve the problem. In all our studies, we follow the selfconsistency setting (Wang et al., 2022), which extends over chain-of-thought (Wei et al., 2022) by using sampling to generate a variety of reasoning paths. From this setting, we quantify the difficulty of a problem w.r.t a language model by its solve rate, which is the proportion of the reasoning paths that lead to the correct answer. When the solve rate exceeds 50%, a majority vote guarantees the correct answer. Note the solve rate measures the difficulty of a single problem input and is also a model-dependent metric.

To study how surface form impacts the solve rate, we use the math word problem from the GSM8K dataset (Cobbe et al., 2021). For each math problem, we generate a paraphrase using GPT-3.5-turbo<sup>3</sup> (detailed instructions are shown in Appendix E). We then compare the solve rates of the original problem and the paraphrase solved by GPT-3.5-turbo using self-consistency with N = 40 and a temperature of 0.7.

Our finding is that the solve rate varies significantly across the surface forms. Figure 1 shows an example with the original problem on the left and the paraphrased one on the right. In the original problem, the reasoning paths result in a disarrayed answer distribution, with merely 5% achieving the correct answer “40” and the aggregated answer “1.54” (20%). In contrast, the solve rate of the paraphrase problem is 100%. We have identified many more such examples with drastic improvement in solve rate, presented in Table 6.

![](images/3ab53709ac5fa698698fd8d772b32113562a2d7007fc1a35feab7cd44cd974f9.jpg)  
Figure 2: GSM8K - solve rate difference - from original to one of the random naive paraphrases.

We further calculate the histogram of the solve rate changes in the paraphrased problem compared to the original one, shown in Figure 2. As can be observed, the distribution is heavy-tailed, with 11.7% of the paraphrases resulting in over 25% absolute improvement in solve rate and with 13% resulting in over 25% absolute deterioration.

This phenomenon exposes the language model’s lack of robustness and sensitivity to a comprehensive problem’s surface form. It suggests that the challenge of some problems may not be due to the model’s limitations, but rather the ineffective generation of reasoning paths from certain surface forms. We therefore seek to take advantage of this phenomenon to improve language model reasoning through surface form modifications, mirroring the way paraphrasing aids a student’s cognitive and problem-solving processes (Swanson et al., 2019).

## 3 Self-Consistency over Paraphrases

Motivated by the findings in Section 2, we propose a framework called Self-Consistency-over-Paraphrases (SCoP), which leverages the LLMs to generate paraphrases of math problems to improve their ability in solving them.

## 3.1 Framework Overview

As shown in Figure 3, SCoP consists of two steps. Step 1: Paraphrase. Prompt the LLM to generate K paraphrases of the original problem. For notational ease, denote p as the original problem, and $\textstyle \bigcup _ { k = 1 } ^ { K } \{ q _ { k } \}$ as the K paraphrases.

Step 2: Solve. For each paraphrase, we ask the LLM to generate N/K reasoning paths, and thus the total number of generated answers is N. We then select the most consistent answer across the N reasoning paths as the final answer.

The intuition behind SCoP is that if a problem exhibits a low solve rate and ineffective reasoning paths due to its original surface form, introducing diversity in its surface forms would be beneficial.

![](images/4024efd02e5bd035b2ae4b322043164249f1f70caa577b2c9c8a7b0252c60e30.jpg)  
Figure 3: A comparison between Self-Consistency and our SCoP. SCoP splits N reasoning paths over K incontext learned paraphrases, instead of devoting all N reasoning paths to the single original problem P. The final answer is selected by aggregating all reasoning paths from these paraphrases with a majority vote.

There are two important notes regarding SCoP. First, when we increase K, the total number of reasoning paths N is held fixed, which separates the effect of increasing the diversity of reasoning paths from increasing the number of reasoning paths. This also ensures a fair comparison with other self-consistency baselines.

Second, there are two procedures in SCoP that involve an LLM, one to generate paraphrases (Step 1) and one to generate answers (Step 2). We use the same LLM to perform both tasks. In this way, we can ensure that any performance improvement of SCoP is due to the diversity of paraphrasing itself, rather than cross-sharing of knowledge across different LLMs. In addition, there is no human annotation, training, fine-tuning, or auxiliary models involved in our SCoP framework.

## 3.2 Paraphrase Generation

The paraphrase generation in Step 1 is crucial to the success of SCoP. In this work, we explore two paraphrase generation methods.

Naïve. The naïve approach instructs the language model to generate K paraphrases of the math problem. However, this could generate many paraphrases with worse solve rate, because the solve rate change has high variability in both directions (as shown in Figure 2).

In-Context Learning. To increase the chance of generating ‘good’ paraphrases, we propose an in-context learning approach,<sup>4</sup> where we obtain $N _ { s h o t }$ ‘good’ paraphrases as the in-context exemplars (marked as [Exemplars] in Figure 3). The ‘good’ paraphrases are formally defined as paraphrases that contribute to a solve rate improvement (by a preset margin δ) over the original problem. To obtain the ‘good’ paraphrases, we first generate some candidate paraphrases using the aforementioned naïve approach on a small number of math problems with labeled answers. We then compute the solve rate of the original problem and the paraphrases and select those whose improvement is over the margin δ. The detailed algorithm is presented in Algorithm 1.

Algorithm 1 Paraphrase Exemplar Search   
1: Input: Training data $D ^ { t r } , N _ { s h o t } .$ , margin δ. Init. Candi   
dates list C.   
2: for step t in $\{ 1 , 2 , \ldots , T \}$ do   
3: if Lengt $\mathbf { \eta } _ { 1 } ( \mathbf { C } ) = N _ { s h o t }$ then   
4: break   
5: Sample a problem p from $D ^ { t r }$ without replacement.   
6: Compute solve rate $\operatorname { S R } ( p )$   
7: Obtain K Paraphrases $\dot { \{ q _ { 1 } , \dots , q _ { K } \} }$ of p.   
8: for $k = 1$ to K do   
9: Compute solve rate $\operatorname { S R } ( q _ { k } )$   
10: $\mathbf { i f } \operatorname { S R } ( q _ { k } ) > = \operatorname { S R } ( p ) \dot { + } \dot { \delta }$ then   
11: Add $\{ p , q _ { k } \}$ to Candidates list C.   
12: break

## 4 Experiments

In this section, we will describe our experiment results evaluating the effectiveness of SCoP, as well as additional studies on how SCoP works.

## 4.1 Experimental Settings

Datasets We evaluate our approach on the following public mathematics reasoning benchmarks: GSM8K (Cobbe et al., 2021) contains 8.5K linguistically diverse grade school-level math questions with moderate difficulties.

AQuA (Ling et al., 2017) consists of 100K algebraic word problems, including the questions, the possible multiple-choice options, and natural language answer rationales from GMAT and GRE. MATH (Hendrycks et al., 2021) is a competition mathematics dataset containing 12,500 problems with challenging concepts such as Calculus, Linear Algebra, Statistics, and Number Theory.

MMLU (Hendrycks et al., 2020) is a comprehensive dataset containing various subjects. We specifically utilized the mathematics section of the dataset, which comprises college and high-schoollevel mathematics, statistics, and abstract algebra.

Language Models We utilize three popular LLMs trained with RLHF (Ouyang et al., 2022): LLaMA-2 (70B) (Touvron et al., 2023), an opensource LLM by Meta AI, GPT-3.5-turbo (version 0613), and GPT-4 (OpenAI, 2023), accessed via the OpenAI API. All experiments are conducted in zero-shot or few-shot settings, without training or fine-tuning the language models. We choose the temperature $T = 0 . 7$ and Top- $- \mathsf { p } = 1 . 0$ for sampling decoding for all three language models. The total number of reasoning paths N we sample for each problem is 40, following Wang et al. (2022).

Implementation Details For paraphrase generation (Step 1), we evaluate the two aforementioned schemes ❶ Naïve: We use the template “ Paraphrase thefollowing math problem: {question}” to prompt the language model to paraphrase the original problem; ❷ In-Context Learning $\bf ( I C L _ { p a r a } )$ : We randomly select a set of 8 paraphrase exemplars by Algorithm 1 with margin<sup>5</sup> $\delta = 0 . 3$ . The details of the prompt templates are available in Appendix E.

For answer generation (Step 2), we also implement two schemes: ❶ Zero-Shot Chain-of-Thought (CoT) (Kojima et al., 2023), which appends “Let’s think step by step.” to the question text; and ❷ Four-Shot CoT, where we append fourshot in-context examples with CoT to the LLM when solving the math problems. Note that the in-context examples for answer generation are different in functionality and format from the ones for $\mathrm { { I C L } _ { \mathrm { { p a r a } } } }$

## 4.2 Main Results

Zero-Shot CoT Table 1 illustrates the performance of SCoP under the zero-shot CoT setting, compared with the vanilla self-consistency (SC), using LLaMA-2-70b and GPT-3.5-turbo. We vary the number of paraphrases K across {1, 2, 4, 8} while keeping the total number of reasoning paths fixed as 40. Due to resource constraints, we randomly sampled 300 data points from each test set, except for AQuA, which contains 254 testing examples.

The performance metric is the accuracy of the self-consistency answer. We also report the accuracy over hard problems, defined as the problems whose original accuracy is below 50%. The accuracies over all problems and hard problems are reported inside and outside parentheses respectively. HPR% (Hard Problem Ratio) denotes the percentage of such hard problems.

<table><tr><td rowspan="2"></td><td colspan="5">GPT-3.5-Turbo</td><td colspan="4">LLaMA-2-70b</td></tr><tr><td></td><td>GSM8K</td><td>AQuA</td><td>MATH</td><td>MMLU</td><td>GSM8K</td><td>AQuA</td><td>MATH</td><td>MMLU</td></tr><tr><td rowspan="2">SC</td><td>HPR (%)</td><td>31.3</td><td>42.5</td><td>68.0</td><td>64.0</td><td>52.0</td><td>76.3</td><td>98.2</td><td>81.6</td></tr><tr><td></td><td>76.3 (24.5)</td><td>66.9 (22.2)</td><td>59.0 (39.7)</td><td>52.8 (26.3)</td><td>58.7 (20.5)</td><td>40.5 (22.0)</td><td>10.5 (8.9)</td><td>32.8 (17.4)</td></tr><tr><td rowspan="4">SCoP (Naïve)</td><td>k = 1</td><td>72.2 (27.7)</td><td>63.4 (28.9)</td><td>55.0 (37.5)</td><td>48.4 (27.5)</td><td>51.0 (28.2)</td><td>38.1 (26.8)</td><td>24.6 (23.2)</td><td>27.2 (20.2)</td></tr><tr><td>k = 2</td><td>76.0 (34.0)</td><td>65.8 (28.9)</td><td>56.5 (39.7)</td><td>52.8 (32.5)</td><td>54.3 (26.9)</td><td>39.5 (25.0)</td><td>29.8 (28.6)</td><td>29.6 (22.3)</td></tr><tr><td>k = 4</td><td>77.7 (36.2)</td><td>67.3 (29.8)</td><td>57.5 (39.0)</td><td>56.0 (36.3)</td><td>55.7 (32.1)</td><td>41.4 (25.6)</td><td>31.6 (30.4)</td><td>32.0 (24.9)</td></tr><tr><td>k = 8</td><td>79.3 (39.4)</td><td>68.1 (33.5)</td><td>59.5 (43.4)</td><td>55.6 (33.8)</td><td>60.3 (33.3)</td><td>41.4 (25.6)</td><td>28.1 (26.8)</td><td>35.6 (28.0)</td></tr><tr><td rowspan="4"> $\operatorname { s c o P }$   $\mathrm { ( I C L _ { p a r a } ) }$ </td><td>k = 1</td><td>77.9 (39.0)</td><td>66.4 (29.8)</td><td>54.0 (36.8)</td><td>52.5 (32.6)</td><td>58.7 (39.9)</td><td>42.9 (29.8)</td><td>23.4 (22.0)</td><td>34.6 (23.7)</td></tr><tr><td>k = 2</td><td>80.5 (39.2)</td><td>68.5 (31.7)</td><td>57.5 (39.1)</td><td>55.5 (34.1)</td><td>59.3 (36.3)</td><td>43.7 (30.4)</td><td>24.6 (23.2)</td><td>37.6 (26.3)</td></tr><tr><td>k = 4</td><td>79.2 (38.3)</td><td>70.5 (35.4)</td><td>58.0 (41.2)</td><td>58.0 (39.5)</td><td>61.7 (40.5)</td><td>44.5 (30.4)</td><td>26.9 (25.6)</td><td>37.8 (26.5)</td></tr><tr><td> $k = 8$ </td><td>80.2 (40.6)</td><td>69.7 (34.4)</td><td>60.0 (44.1)</td><td>56.5 (34.9)</td><td>63.3 (40.5)</td><td>46.5 (31.9)</td><td>25.2 (23.8)</td><td>37.6 (25.8)</td></tr></table>

Table 1: Accuracy of SCoP distributing N/K reasoning paths over K in {1, 2, 4, 8} paraphrases in Naïve and $\mathrm { { I C L } _ { \mathrm { { p a r a } } } }$ settings, against Self-Consistency (SC). Hard Problem Ratio (HPR%) represents the percentage of problems with an original solve rate $\leq 0 . 5$ by Self-Consistency (SC). Accuracy is reported for both Hard Problems (HPR% $\leq 0 . 5 )$ (inside parentheses) and global accuracy across the entire dataset (outside parentheses).
<table><tr><td>Model</td><td></td><td>GSM8K</td><td>AQuA</td><td>MATH</td><td>MMLU</td></tr><tr><td rowspan="3">LLaMA-2-70b</td><td>HPR (%)</td><td>56</td><td>75.2</td><td>95.2</td><td>81.6</td></tr><tr><td>Self-Consistency</td><td>61.1 (30.0)</td><td>44.1 (25.7)</td><td>13.4 (9.2)</td><td>34.4 (19.6)</td></tr><tr><td>SCoP  $( \mathrm { I C L } _ { \mathrm { p a r a } } , k = 8 )$ </td><td>65.1 (38.6)</td><td>48.8 (33.5)</td><td>23.6 (20.2)</td><td>36.4 (27.0)</td></tr><tr><td rowspan="3">GPT-3.5-Turbo</td><td>HPR (%)</td><td>22</td><td>36</td><td>75</td><td>62</td></tr><tr><td>Self-Consistency</td><td>80.0 (9.1)</td><td>70.0 (16.7)</td><td>51.6 (29.8)</td><td>54.4 (26.9)</td></tr><tr><td>SCoP  $( \mathrm { I C L } _ { \mathrm { p a r a } } , k = 8 )$ </td><td>82.0 (36.4)</td><td>74.0 (27.8)</td><td>57.6 (38.2)</td><td>58.4 (36.5)</td></tr><tr><td rowspan="3">GPT-4</td><td>HPR (%)</td><td>4</td><td>18</td><td>58</td><td>38</td></tr><tr><td>Self-Consistency</td><td>98.0 (50.0)</td><td>84.0 (11.1)</td><td>64.0 (37.9)</td><td>74.0 (31.6)</td></tr><tr><td> $\mathrm { S C o P \left( I C L _ { p a r a } , \right)} k = 8 $ </td><td>98.0 (50.0)</td><td>86.0 (33.3)</td><td>66.0 (41.4)</td><td>78.0 (57.9)</td></tr></table>

Table 2: A comparison of the performance (accuracy) between SC and SCoP $\mathrm { ( I C L _ { p a r a } }$ paraphrasing, with $k = 8 )$ using 4-shot in-context chain-of-thought exemplars over three language models. Accuracy is reported for both Hard Problems $( \mathrm { H P R } \mathcal { I } _ { o } \leq 0 . 5 )$ (inside parentheses) and global accuracy across the entire dataset (outside parentheses).

There are three general observations. First, SCoP with the two paraphrasing schemes both outperform the vanilla self-consistency baseline. Surprisingly, even the naïve paraphrasing can lead to performance improvement, despite the high chances of generating paraphrases with a worse solve rate (see Figure 2). We will discuss a hypothesis in Section 5. Between the two schemes, ICL<sub>para</sub> consistently outperforms Naïve. Second, the performance improvement generally increases as K increases. Third, more significant performance gain over LLaMA-2-70B.

The results further indicate that MATH and MMLU are considerably more challenging than GSM8K and AQuA, as evidenced by their high HPR% and low overall accuracy. Moreover, significant accuracy gains are from the original “Hard Problems”, suggesting that changing surface forms can solve the problems initially deemed unsolvable by self-consistency. Finally, when solving the MATH dataset with LLaMA-2-70b, ICL ${ } ^ { \prime } \mathrm { p a r a }$ underperforms Naïve paraphrasing. We hypothesize that the MATH problems present a significant challenge for LLaMA-2-70b, making it difficult to effectively learn paraphrasing from in-context examples.

Four-Shot CoT One caveat of the zero-shot CoT results is that SCoP $\mathrm { ( I C L _ { p a r a } ) }$ has indirect access to additional ground-truth information from incontext exemplars. There is also a question of whether the advantage of SCoP over SC will diminish as both are exposed to more examples. To ensure a fair comparison and further validate the effectiveness of SCoP, Table 2 shows results under the four-shot CoT setting, where the baselines also have access to some ground-truth answer information. Due to resource constraints, we evaluate GPT4 with 100 random samples from each dataset. The results show that while four-shot CoT can improve SC and SCoP in general (compared with zero-shot CoT), SCoP still consistently outperforms SC over all three language models. The only exception is GPT4 on GSM8K, which already achieves near-perfect performance with SC, thus SCoP only achieves equivalent performance.

<table><tr><td></td><td>GPT3.5, GPT4</td><td>GPT3.5, LLaMA-2</td><td>GPT4, LLaMA-2</td></tr><tr><td>GSM8K</td><td>0.573**</td><td>0.649***</td><td>0.445*</td></tr><tr><td>AQUA</td><td>0.543***</td><td>0.227***</td><td>0.314*</td></tr><tr><td>MATH</td><td>0.554***</td><td>0.242*</td><td>0.433*</td></tr><tr><td>MMLU</td><td>0.313*</td><td>0.320***</td><td>0.233</td></tr></table>

Table 3: Spearman’s rank correlation of original problems’ solve rate across language models.

## 4.3 Additional Studies

Searching for Exemplars Since our in-context learning paraphrasing scheme requires access to ground-truth answers, we would like to study how many problems with ground-truth answers are needed. Figure 4 illustrates how many data points in the training set, on average, need to be sampled to obtain $N _ { s h o t }$ ‘good’ paraphrases (x-axis) with different margins. We can observe that, although satisfying a large margin requires more samples, it is relatively easy (typically every 5 example) to find a sample that substantially improves the solve rate after paraphrasing. This, again, indicates the sensitivity of the language model to surface form variations in mathematical reasoning.

Difficulty Beliefs Across Language Models An intriguing question is how different language models rank the difficulty of the problems. We measure the agreement between language models on problem difficulty by Spearman’s rank correlationof the solve rate for original problems across four datasets. As shown in Table 3, the ranks of the difficulty (by solve rate) are all positively correlated. However, the degree of correlation varies, with higher agreement observed within the GPT model family and on simpler datasets.

Paraphrase Transfer We investigate whether paraphrases from a stronger LLM can be transferred to weaker ones and improve SCoP. Table 4 demonstrates the paraphrase transfer performance of SCoP (Naïve, k = 8) on 100 randomly sampled data points from MMLU and GSM8K under the zero-shot CoT setting. In general, paraphrases produced by GPT-4 can be utilized by GPT3.5- turbo or LLaMA-2-70b for further performance improvements, with an exception with LLaMA-2 on MMLU, where GPT4 and LLaMA-2 exhibit the lowest Spearman rank correlation of solve rate. We hypothesize that the benefits of transferring paraphrases across models may depend on the agreement in their beliefs of problem difficulty.

<table><tr><td>Solver</td><td>Paraphraser</td><td>MMLU</td><td>GSM8K</td></tr><tr><td>GPT-3.5</td><td>Self</td><td>50.0</td><td>78.0</td></tr><tr><td>GPT-3.5</td><td>GPT-4</td><td>54.0</td><td>84.0</td></tr><tr><td>LLaMA-2</td><td>Self</td><td>37.0</td><td>61.0</td></tr><tr><td>LLaMA-2</td><td>GPT-4</td><td>34.0</td><td>69.0</td></tr></table>

Table 4: Performance of SCoP (Naïve, k = 8) on MMLU and GSM8K, with different paraphrasers.
<table><tr><td colspan="3">GSM8K</td><td>AQuA</td><td>MATH</td><td>MMLU</td></tr><tr><td rowspan="2">LLaMA2</td><td>Naïve</td><td>20.3</td><td>17.5</td><td>12.9</td><td>17.5</td></tr><tr><td>ICLpara</td><td>18.9</td><td>15.7</td><td>12.2</td><td>16.6</td></tr><tr><td rowspan="2">GPT-3.5</td><td>Naïve</td><td>20.6</td><td>16.1</td><td>15.8</td><td>16.9</td></tr><tr><td> $\underline { { \mathrm { I C L } _ { \mathrm { p a r a } } } }$ </td><td>16.2</td><td>10.7</td><td>15.6</td><td>15.6</td></tr><tr><td>GPT-4</td><td> $\mathrm { { I C L } _ { \mathrm { { p a r a } } } }$ </td><td>9.7</td><td>11.5</td><td>17.0</td><td>21.3</td></tr></table>

Table 5: VOV values across datasets and language models, shown as standard deviation.

Variance of Variations In light of the considerable variability observed in solve rates among problem surface forms (Figure 2), we propose and advocate Variance of Variations (VOV) for evaluating language models on reasoning robustness. Let $X ( p ) \in [ 0 , 1 ]$ be the random variable representing the solve rates of various paraphrases of a problem p. Then the VOV value of the dataset D is then defined as:

$$
\operatorname { V O V } = \mathbb { E } _ { p \sim D } [ \operatorname { V a r } ( X ( p ) ) ]\tag{1}
$$

where Var( ) is the variance. A large value of VOV indicates high variability in the language model’s reasoning ability against problem surface forms. We compute VOV using the solve rate for the k = 8 paraphrases and the original problem as $X ( p )$ for each p. As shown in Table 5, while VOV decreases when a robust model solves a more manageable dataset (e.g., GPT-4 on GSM8K), and ICL<sub>para</sub> generated paraphrases can generally reduce VOV, VOV remains unreasonably high over more challenging datasets and all language models.

Examples of ‘Good’ Paraphrases We provide some qualitative examples comparing the solve rates between the original problem and a paraphrased version in Table 6. It is difficult to visually tell what contributes to a good paraphrase. We will publish these data to encourage future research.

## 5 Discussion

We have an intriguing observation that even the naïve scheme of generating math paraphrases can improve the overall accuracy. However, the naïve scheme has a significant chance of generating worse paraphrases. Why would aggregating over the mixture of better and worse paraphrases still significantly improve the performance?

![](images/351bfe25030e5d44498089df188eaabba9e34876e5d6a8d6af5591ed94f658ce.jpg)  
(a)

![](images/8e55b3289188d8800250f57f3a22eff3463e1ff3b733ef2ab35384b45e55ecde.jpg)  
(b)

![](images/ef13235b5febff6735735b520c1e93f6b822d5ab8279d732da339df18f9ed9e5.jpg)  
(c)

![](images/8f19b94623f829ecf845b6730bffc446c16ff2fbb5c40a220497e7e4e7f31632.jpg)  
(d)

Figure 4: (a) GSM8K (b) AQuA (c) MATH (d) MMLU. The average number of data points in the training set needed for obtaining $N _ { s h o t }$ exemplars at different margins.  
![](images/45b61b93dbbc9a8dfececb999a2c880cf6c0d9f25fe087e6dbb800838331e888.jpg)  
(a)

![](images/7ac7d1e71eb862dc997a8464079e232a62d2044efbfddd1fc28693dd9a947aef.jpg)  
(b)

![](images/164543b084f5c2fb11454d27f493c35c390ba1f318f0713b57867afa234e2b60.jpg)  
(c)  
Figure 5: Data Difficulty Map for GSM8K using GPT3.5, with three types of changes from solving the original problem to one of its random paraphrases: (a) Improvement, (b) Overconfidence, and (c) Uncertainty. Arrows indicate the solve rate and entropy change from solving the original problem to its paraphrased version.

To explain this, Figure 5 shows three scatter plots of the solve rate against the entropy of answer distributions. The outcome of solving each random paraphrase is represented as a black dot. As can be observed, the dots roughly form a triangular region. The top left corner represents the ideal case with high solve rates and high confidence. The bottom corners, on the other hand, represent two failure modes. The bottom right corner represents the case with low solve rates and low confidence, and the bottom left corner with low soft rates but high confidence (commonly known as over-confidence).

The blue arrows in Figure 5(a) visualize the cases where the paraphrases improve the solve rate, and they mostly point to the top-left corner. The arrows in Figures 5(b) and (c) represent the cases where the paraphrases lower the solve rate, and we can observe that the arrows pointing to the bottom right corner (yellow arrows in (b)) far outnumber those to the bottom left corner (red arrows in (c)).

This indicates that while the ‘good’ paraphrases would sharpen the answer distribution, the ‘bad’ paraphrases mostly would flatten the distribution. Since the final aggregated answer distribution is predominantly influenced by the sharp distributions, the damage brought by the “bad” paraphrases is small compared to the benefit brought by the ‘good’ paraphrases, and thus the aggregate effect across all the paraphrases is still positive.

## 6 Related Work

Mathematical Reasoning in LLMs The complexity of mathematics necessitates System-2 reasoning, characterized by a slow, step-by-step cognitive process (Kahneman, 2011). Numerous works have sought to emulate this process in solving mathematics with LLMs (Wei et al., 2022; Wang et al., 2022; Kojima et al., 2023; Lightman et al., 2023; Qiao et al., 2022). As a prominent framework, chain-of-thought (Wei et al., 2022; Kojima et al., 2023) prompts the language model to generate a sequence of reasoning steps instead of a direct answer; Wang et al. (2022) extended chain-of-thought by Self-Consistency, in which they replaced greedy decoding with sampling decoding to generate a variety of reasoning paths, with multiple paths potentially leading to the same answer from different angles. Other multi-step reasoning variations with verifiers exist (Lu et al., 2023; Besta et al., 2023; Yao et al., 2023); however, they are less related to our focus primarily on the language model’s internal ability to solve mathematical problems.

<table><tr><td>Problem</td><td>Source</td><td>Label</td><td>SR</td><td>Voted (F.)</td></tr><tr><td>Original: Jenna has 4 roommates. Each month the electricity bill is $100. How much will each roommate pay per year for electricity, if they divide the share equally? Paraphrased: Jenna shares an apartment with 4 other people. The electricity bill is $100 per month. If they split the bill equally, how much will each roommate contribute towards the electricity bill in a year?</td><td>GSM8K</td><td>&quot;240&quot;</td><td>0.15 0.95</td><td>&quot;300&quot; (0.8) &quot;240&quot;</td></tr><tr><td>Original: Jenny goes to the florist to buy some flowers. Roses cost $2 each and $15 for a dozen. If she bought 15 roses and arrived with five 5 dollar bills and they only have quarters for change, how many quarters does she leave with? Paraphrased: Jenny visits the flower shop to purchase flowers. She can buy roses individually for $2 each or buy a dozen roses for $15. Jenny decides to buy 15 roses in total. She pays with five $5 bills and</td><td></td><td>GSM8K &quot;16&quot;</td><td>0.0</td><td>&quot;20&quot; (0.3)</td></tr><tr><td>the florist can only give her change in quarters. The question asks how many quarters Jenny receives as change. Original: Assistants are needed to prepare for preparation. Each helper can make either 2 large cakes or 35 small cakes/hr. The kitchen is available for 3 hours and 20 large cakes &amp; 700 small cakes are needed. How many helpers are required?</td><td></td><td></td><td>0.2</td><td>A (0.25)</td></tr><tr><td>per hour, and the kitchen is available for 3 hours and needs 20 large cakes and 700 small cakes? Options:[A)8,B)10,C)12,D)15,E)19] Original: A starts a business with Rs.40,000. After 2 months, B joined him with Rs.60,000. C joined them after some more time with Rs.120,000. At the end of the year, out of a total profit of Rs.375,000, C gets Rs.150,000 as his share. How many months after B joined the business, did C join?</td><td></td><td></td><td>0.8 0.1</td><td></td></tr><tr><td>Paraphrased: A starts a business with Rs.40,000 and after 2 months, B joins with Rs.60,000. C joins the business AQUA at some point later with Rs.120,000. At the end of the year, the total profit is Rs.375,000, and C receives Rs.150,000 as their share. How many months after B joined the business did C join? Options:[A)2,B)4,C)23,D)24,E)84] Original: A star-polygon is drawn on a clock face by drawing a chord from each number to the fifth number counted clockwise from that number. That is, chords are drawn from 12 to 5, from 5 to 10, from 10 to 3,</td><td></td><td></td><td>0.45</td><td>B</td></tr><tr><td>and so on, ending back at 12. What is the degree measure of the angle at each vertex in the star-polygon? Paraphrased: What is the measure of the angle at each vertex in the star-polygon formed by drawing a chord from each number on the clock face to the fifth number counted clockwise from that number? Original: By partial fractions,  $\frac { 1 } { a x ^ { 2 } + b x + c } = \frac { A } { \frac { - b + \sqrt { b ^ { 2 } - 4 a c } } { \frac { - b + \sqrt { b ^ { 2 } - 4 a c } } { \frac { \Omega _ { c } } { \Omega _ { c } } } } + \frac { B } { \frac { - b - \sqrt { b ^ { 2 } - 4 a c } } { \frac { - b - \sqrt { b ^ { 2 } - 4 a c } } { \Omega _ { c } } } } \mathrm { F i n d } \ A + B . }$ </td><td>MATH</td><td>&quot;30&quot;</td><td>0.5</td><td>&quot;30&quot;</td></tr><tr><td>2a 2a Paraphrased: Find the sum of A and B in the expression  $\begin{array} { r } { \frac { 1 } { a x ^ { 2 } + b x + c } = \frac { A } { x - \frac { - b + \sqrt { b ^ { 2 } - 4 a c } } { 2 a } } + \frac { B } { x - \frac { - b - \sqrt { b ^ { 2 } - 4 a c } } { 2 a } } . } \end{array}$  *Note the IATXcode was paraphrased from \frac to \dfrac.</td><td>MATH</td><td>&quot;0&quot;</td><td>0.2 0.65</td><td>&quot;1&quot; (0.25) &quot;0&quot;</td></tr><tr><td>Original: Statement 1 I For every positive integer n there is a cyclic group of order n. Statement 2 | Every finite cyclic group contains an element of every order that divides the order of the group. Paraphrased: Statement 1 says that there exists a cyclic group of any positive integer n. Statement 2 says that in any finite cyclic group, there is an element for every possible order that divides the order of the group. Options:[A)True,True, B)False,False, C)True,False, D)False,True]</td><td></td><td>MMLU A</td><td>0.05</td><td>C (0.95)</td></tr><tr><td>Original: What is the probability that a randomly selected integer in the set {1, 2, 3, . . . , 100} is divisible by 2 and not divisible by 3? Express your answer as a common fraction.</td><td></td><td></td><td>0.25</td><td>A (0.4)</td></tr><tr><td>Paraphrased: What is the chance that if we randomly choose an integer from the set of numbers 1 to 100, it will be divisible by 2 but not divisible by 3? Write your answer as a fraction. ，B) ， C) ，D) 1] Options:[A)</td><td>MMLU</td><td>D</td><td>0.8</td><td>D</td></tr></table>

Table 6: Qualitative examples where the original problems and corresponding surface form variations exhibit substantial solve rate difference using GPT-3.5-turbo.

Paraphrasing Variability Previous research on the impact of paraphrasing mathematical problems on their solvability by language learning models (LLMs) is limited. The study by (Gonen et al., 2022) explored how paraphrased instructions affect the performance of traditional NLP benchmarks. This sensitivity of instructive prompts has inspired further research in prompting learning (Shin et al., 2020; Zhou et al., 2023; Sordoni et al., 2023) and in-context exemplar mechanisms (Min et al., 2022;

Brown et al., 2020; Ye and Durrett, 2022). However, our work focuses on the sensitivity of the mathematical problem presentation itself instead of the instruction or in-context examples.

## 7 Conclusions

This work highlights the variability in the solve rate of large-scale language models to the surface form of mathematical problems. Leveraging this, we introduced the Self-Consistency-over-Paraphrases (SCoP), which improves mathematical reasoning performance over Self-Consistency. We hope our findings will inspire the need for more robust language models that can reason effectively regardless of how a problem is presented.

## Limitations

While we derive thorough conclusions about the relationship between the surface form of a mathematical problem and its solvability by large-scale language models with the effectiveness of SCoP and additional studies, one limitation is the need for a mechanism for identifying or generating surface forms that are easier to solve than others. The study is solely conducted in English, while the generalizability of SCoP in other languages is unexplored. Future research could address these by exploring the rationalization of surface forms, i.e., determining the optimal form given the original one, and the verifiability of the framework in other languages.

## Ethics Statement

The datasets that we used in experiments are publicly available. In our work, we explore the relationship between the surface form of a mathematical problem and its solvability by large-scale language models. We do not expect any direct ethical concern from our work.

## Acknowledgements

This study was supported by MIT-IBM Watson AI Lab.

## References

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michal Podstawski, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. 2023. Graph of thoughts: Solving elaborate problems with large language models.

Rahul Bhagat and Eduard Hovy. 2013. What Is a Paraphrase? Computational Linguistics, 39(3):463–472.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Hila Gonen, Srini Iyer, Terra Blevins, Noah A. Smith, and Luke Zettlemoyer. 2022. Demystifying prompts in language models via perplexity estimation.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. NeurIPS.

Daniel Kahneman. 2011. Thinking, fast and slow. macmillan.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2023. Large language models are zero-shot reasoners.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, et al. 2023. Holistic evaluation of language models.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. arXiv preprint arXiv:2305.20050.

Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. 2017. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In Proceedings ofthe 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 158–167, Vancouver, Canada. Association for Computational Linguistics.

Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. 2023. Chameleon: Plug-and-play compositional reasoning with large language models.

Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the role of demonstrations: What makes in-context learning work? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11048–11064, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

OpenAI. 2023. Gpt-4 technical report.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback.

Shuofei Qiao, Yixin Ou, Ningyu Zhang, Xiang Chen, Yunzhi Yao, Shumin Deng, Chuanqi Tan, Fei Huang, and Huajun Chen. 2022. Reasoning with language model prompting: A survey. arXiv preprint arXiv:2212.09597.

Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, et al. 2022. Scaling language models: Methods, analysis and insights from training gopher.

Taylor Shin, Yasaman Razeghi, Robert L. Logan IV au2, Eric Wallace, and Sameer Singh. 2020. Autoprompt: Eliciting knowledge from language models with automatically generated prompts.

Alessandro Sordoni, Xingdi Yuan, Marc-Alexandre Côté, Matheus Pereira, Adam Trischler, Ziang Xiao, Arian Hosseini, Friederike Niedtner, and Nicolas Le Roux. 2023. Joint prompt optimization of stacked llms using variational inference.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, et al. 2023. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research.

H Lee Swanson, Jennifer E Kong, Amber S Moran, and Michael J Orosco. 2019. Paraphrasing interventions and problem-solving accuracy: Do generative procedures help english language learners with math difficulties? Learning Disabilities Research & Practice, 34(2):68–84.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models.

Xi Ye and Greg Durrett. 2022. The unreliability of explanations in few-shot prompting for textual reasoning. Advances in neural information processing systems, 35:30378–30392.

Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. 2023. Large language models are human-level prompt engineers.

## A Choose Margin

We examine the effect of margin on selecting exemplars for in-context paraphrasing using GPT-3.5 and separate dev-sets from GMS8K and MMLU, each with 250 data points. The results in Table 7 show that a moderate margin outperforms a large one in SCOP, as the latter may decrease the diversity of exemplars.

<table><tr><td>Margin</td><td>MMLU (Dev, k = 8)</td><td>GSM8K (Dev, k = 8)</td></tr><tr><td>SC/HPR%</td><td>53.2 (26.9) / 64</td><td>73.6 (21.4) / 33.6</td></tr><tr><td>0.2</td><td>55.6 (34.4)</td><td>75.2 (31.0)</td></tr><tr><td>0.3</td><td>56.8 (35.6)</td><td>74.8 (32.1)</td></tr><tr><td>0.4</td><td>55.2 (33.8)</td><td>75.6 (33.3)</td></tr><tr><td>0.5</td><td>53.6 (32.5)</td><td>74.4 (35.7)</td></tr></table>

Table 7: Ablation on the margin effect of exemplar selection.

## B APE Alternatives

A potential alternative to finding an optimal prompt for paraphrasing is to use the Automatic Prompt Engineering (APE) settings (Zhou et al., 2023). We formulate the procedure into four steps:

1. Present a set of input-output pairs where the inputs are the original problems and the outputs are the paraphrased exemplars. Prompt the language model to generate C candidate instructions that could produce the outputs from the inputs.

2. Prompt each candidate instruction to the language model to generate paraphrases for a batch size B of problems in the development set and compare the mean solve rate change before and after paraphrasing.

3. Choose the instruction that maximizes the mean solve rate change.

4. Repeat steps 1 - 3 E times.

We implemented this procedure using GPT-3.5 on the AQUA development set to obtain the instruction $( C ~ = ~ 1 5 , ~ B ~ = ~ 3 0 , ~ B ~ = ~ 0 )$ We tested the performance in both AQUA (in-domain) and GSM8K (out-of-domain), comparing it with $\mathrm { I C L _ { p a r a } }$ . Although the in-domain AQUA performance was similar to $\mathrm { { I C L } _ { \mathrm { { p a r a } } } } .$ , the out-of-domain performance worsened, and APE required more data than $\mathrm { I C L _ { p a r a } }$ . Therefore, this approach has yielded negative results. The performance results are presented in Table 8.

<table><tr><td colspan="3">GSM8K</td><td colspan="2">AQUA</td></tr><tr><td></td><td> $\overline { { \mathrm { { I C L } _ { p a r a } } } }$ </td><td>APE</td><td> $\overline { { \mathrm { { I C L } _ { \mathrm { { p a r a } } } } } }$ </td><td>APE</td></tr><tr><td>SC</td><td>76.3 (24.5)</td><td></td><td>66.9 (22.2)</td><td></td></tr><tr><td>N/1</td><td>77.9 (39.0)</td><td>73.7 (34.0)</td><td>66.4 (29.8)</td><td>66.1 (29.0)</td></tr><tr><td>N/2</td><td>80.5 (39.2)</td><td>76.3 (36.2)</td><td>68.5 (31.7)</td><td>66.8 (30.0)</td></tr><tr><td>N/4</td><td>79.2 (38.3)</td><td>77.7 (33.0)</td><td>70.5 (35.4)</td><td>70.8 (32.0)</td></tr><tr><td>N/8</td><td>80.2 (40.6)</td><td>79.0 (41.5)</td><td>69.7 (34.4)</td><td>69.2 (31.0)</td></tr></table>

Table 8: A comparison between the performance of APE and $\mathrm { { I C L } _ { \mathrm { { p a r a } } } }$ paraphrasing.

## C Temperature and Randomness

To further validate that SCoP goes beyond simply increasing randomness, we introduce two variants of SC, where we increase the temperature up to 0.9 and 1 using GPT-3.5 on 1/4 of the MMLU and AQuA datasets. The results are shown in Table 9.

<table><tr><td>Temperature</td><td>MMLU</td><td>AQuA</td></tr><tr><td>0.7 (baseline)</td><td>49 (21.5)</td><td>68 (27.3)</td></tr><tr><td>0.9</td><td>49 (24.6)</td><td>68 (27.3)</td></tr><tr><td>1</td><td>49 (24.6)</td><td>64 (18.2)</td></tr></table>

Table 9: Increasing randomness by temperature saturates reasoning performance. The numbers inside the parenthesis are the accuracy for the hard problems.

As can be observed, although increasing temperature brings a slight improvement in the hard problems, the performance gain soon saturates and is not nearly comparable to that of SCoP.

## D Surface Forms with Solve Rate Degradation

As previously discussed, surface form modification by paraphrasing can lead to degradation in the solve rate. Here, we present additional qualitative examples where the solution rate worsened after paraphrasing. See Table 10.

<table><tr><td>Problem</td><td>Source</td><td>Label</td><td>SR</td><td>Voted (F.)</td></tr><tr><td>Original: Howie wants to buy cupcakes for everyone in his class as a special treat. He&#x27;s not sure if people will want vanilla or chocolate cupcakes so he decides to get one of each for everyone. If he gets the same amount of 2 cupcakes for each himself, his teacher, and his 25 classmates, how many cupcakes should Howie buy?</td><td></td><td></td><td>0.8</td><td></td></tr><tr><td>Paraphrased: Howie wants to purchase cupcakes for his entire class as a special treat. Since he is unsure of the flavor preference, he plans to buy both vanilla and chocolate cupcakes. Howie wants to ensure that he has an equal amount of cupcakes for himself, his teacher, and</td><td></td><td>GSM8K &quot;54&quot;</td><td>0.25</td><td>&quot;27&quot; (0.35)</td></tr><tr><td>his 25 classmates. How many cupcakes should Howie purchase in total? Original: Janice bikes at 10 miles per hour, while Jennie bikes at 20. How long until they have</td><td></td><td></td><td></td><td></td></tr><tr><td>collectively biked 1 mile? Paraphrased: Janice and Jennie are biking at different speeds. Janice bikes at a rate of 10 miles per hour, while Jennie bikes at a rate of 20 miles per hour. How much time will it take for them to collectively</td><td>AQUA</td><td>B</td><td>0.55</td><td></td></tr><tr><td>bike a distance of 1 mile? Options:[A)1 minute, B)2 minutes, C)3 minutes, D)4 minutes, E)5 minutes]</td><td></td><td></td><td>0.1</td><td>C (0.35)</td></tr><tr><td>Original: How many primes are in the row of Pascal&#x27;s Triangle that starts with a 1 followed by a 6? Paraphrased: Starting with the numbers 1 and 6, how many prime numbers are there in the sequence of numbers in Pascal&#x27;s Triangle?</td><td>MATH</td><td>&quot;0&quot;</td><td>0.7 0.1</td><td>&quot;2&quot; (0.25)</td></tr><tr><td>Original: John divided his souvenir hat pins into two piles. The two piles had an equal number of pins. He gave his brother one-half of one-third of one pile. John had 66 pins left. How many pins did John</td><td></td><td></td><td>0.9</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>originally have?</td><td>MMLU B</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Paraphrased: John started with a certain number of souvenir hat pins. He divided them into two equal piles.</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>He then gave his brother one-half of one-third of one of the piles. After that, John was left with 66 pins.</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>0.35 A (0.4)</td></tr><tr><td>How many pins did John have at the beginning?</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Options:[A) 396, B) 72, C) 66, D) 36]</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 10: Qualitative examples where paraphrased surface forms of the original problems can also exhibit solve rate degradation, using GPT-3.5-turbo.

## E Prompt Templates

We list the prompt templates used in the paper below.

## Naïve Paraphrasing

Paraphrase the following math problem: {target problem}

## ICL Paraphrasing

Paraphrase the following math problem: {input problem}

Output: {Paraphrased exemplar}

(Repeat $N _ { s h o t } )$

Paraphrase the following math problem: {target problem}

## APE Candidate Search

A student is completing a task that requires producing a text output from a text input. The student receives instruction about several rules that describe how to produce the outputs given the inputs. What is the instruction?

## Few-shot Chain-of-thought

Question: At Academic Academy, to pass an algebra test you must score at least 80. If there are 35 problems on the test, what is the greatest number you can miss and still pass?

Answer Choices: A) 7 B) 28 C) 35 D) 8

Rationale: First, we need to find 80% of 35. We can do this by multiplying 35 by 0.80: $3 5 \times 0 . 8 0 = 2 8$ . So, if you get 28 problems correct, you will have scored 80% on the test.

To find the greatest number you can miss and still pass, subtract the number you can get correct from the total number of problems:35 28 = 7.

Therefore, the greatest number you can miss and still pass is (A) 7.

(Repeat $N _ { s h o t } )$

Question: {target problem}