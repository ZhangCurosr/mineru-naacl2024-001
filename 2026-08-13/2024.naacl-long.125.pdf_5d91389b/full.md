# Program-Aided Reasoners (Better) Know What They Know

Anonymous ACL submission

## Abstract

Prior work shows that program-aided reasoning, in which large language models (LLMs) are combined with programs written in programming languages such as Python, can significantly improve accuracy on various reasoning tasks. However, while accuracy is essential, it is also important for such reasoners to “know what they know”, which can be quantified through the calibration of the model. In this paper, we compare the calibration of Program Aided Language Models (PAL) and text-based Chain-of-thought (COT) prompting techniques over 5 datasets and 2 model types - LLaMA models and OpenAI models. Our results indicate that PAL leads to improved calibration in 75% of the instances. Our analysis uncovers that prompting styles that produce lesser diversity in generations also have more calibrated results, and thus we also experiment with inducing lower generation diversity using temperature scaling and find that for certain temperatures, PAL is not only more accurate but is also more calibrated than COT. Overall, we demonstrate that, in the majority of cases, program-aided reasoners better know what they know than text-based counterparts.<sup>1</sup>

## 1 Introduction

As language models (LMs) grow in size and capabilities, several works examine methods to improving their reasoning skills with different styles of prompting (Wei et al., 2022; Wang et al., 2022; Suzgun et al., 2022b; Zhou et al., 2022; Yao et al., 2023). One representative method, chain of thought (COT) reasoning (Wei et al., 2022), takes inspiration from how humans approach problem-solving – by breaking down the problem into a sequence of natural language explanations before arriving at a final answer. Furthermore, prompts that enable problem-solving are not limited to natural language; program-aided language models (PAL); Gao et al. (2022) have demonstrated the efficacy of using code (such as Python programs) as a means of improving the model’s reasoning, surpassing the accuracy of conventional chain-of-thought style prompts in some tasks (Madaan et al., 2022; Lyu et al., 2023; Zhang et al., 2023a,b). Both methods are illustrated in Figure 1.

![](images/324116f6c3291b0ea280163bb55d97f3b947fd0bc04ad53c25829c8b8f5992af.jpg)  
Figure 1: Comparisons of COT and PAL outputs. COT can sometimes generate the correct reasoning chain but fail to derive the correct answer as a final step. PAL fixes this issue by executing generated code to arrive at a deterministic answer.

Currently, most works proposing such methods have been primarily focused on improving accuracy. However, for real-world applications, another highly desirable feature of ML systems is that they should be able to provide reliable confidence estimates. Accurate estimates of model confidence are helpful for many applications, including allowing the model to refrain from providing an answer when uncertain, asking for human intervention in uncertain cases, or providing confidence estimates to a downstream model that consumes the outputs. The reliability is measured through calibration, how a model’s confidence in its predictions aligns accurately with actual outcomes (Guo et al., 2017a; Jiang et al., 2020; Zhao et al., 2021).

![](images/8858542914b45eb2fc5fd3fdb16d2a02bf66593aff4a1a227fd62f30a717bdca.jpg)  
Figure 2: Illustration of eliciting model confidence through self-consistency

In sum, the previous research has shown, as eloquently stated by Kadavath et al. (2022) “language models (mostly) know what they know” — LLMs are reasonably well calibrated, although some imperfections remain.

In this work, we examine the effect of programaided reasoning on calibration. We consider five datasets that cover different reasoning tasks and evaluate the performance of both PAL and COT style prompting for OpenAI models (OpenAI, 2023) and LLaMA models (Touvron et al., 2023) with respect to accuracy and calibration. We primarily explore three main research questions :

• RQ 1: Does program-aided reasoning result in significantly different calibration than textbased COT?

• RQ 2: Are the observed trends different across OpenAI models and LLaMA models?

• RQ 3: Does the consistency ofLLM generations affect calibration? We examine this by measuring generation diversity and answer space entropy.

Our results show that program-aided reasoners know what they know even better than standard text-based reasoners with COT. In particular, on OpenAI models, PAL exhibits not only superior accuracy but also a consistent enhancement in calibration of about 50%, over COT. Interestingly, the consistent improvement of calibration is not observed in LLaMA models. Still, we find that adjusting the temperature of sampling (similar to a widely used method of Platt scaling (Platt et al., 1999), PAL improves with respect to accuracy and calibration. We also conduct a detailed analysis of these observations and find a correlation between the similarity of the generated chains-of-thoughts or programs and calibration, which might help explain these trends. Code and data available here under the Apache 2.0 license.

## 2 Preliminaries and Mathematical Formulation

## 2.1 Measuring Calibration

Calibration refers to the alignment between the predicted probability estimates of a model and their actual correctness or accuracy (Guo et al., 2017b). Formally, a perfectly calibrated model can be expressed using the following equation, where X is the given input, Y is the true output, the model’s output is $\hat { Y }$ and $P _ { N } ( { \hat { Y } } \mid X ) = p$ is the probability, or “confidence”, over the model’s output.

$$
P \left( \hat { Y } = Y \mid P _ { N } ( \hat { Y } \mid X ) = p \right) = p , \forall p \in [ 0 , 1 ]\tag{1}
$$

In essence, Equation 1 conveys that if a perfectly calibrated model makes 100 predictions, and the confidence of each prediction is 0.6, then we expect the accuracy to be also 0.6. Nevertheless, the model may exhibit varying confidence levels for each sample. Therefore, it is imperative to calculate calibration across all confidence scores. We estimate this probability by dividing the predictions into M separate and equally sized interval buckets based on their confidence levels.

We use the expected calibration error (ECE), a common measure of (lack of) calibration, a weighted average of the discrepancy between each bucket’s accuracy and confidence. It is given in Equation 2

Here $B _ { m }$ is the m-th bucket that contains samples whose probabilities of predictions fall in the interval $\textstyle \left( { \frac { m - 1 } { M } } , { \frac { m } { M } } \right]$ , where $\frac { | B _ { m } | } { n }$ is $B _ { m } \mathrm { ^ { * } s }$ size relative to all the samples. acc $\left( B _ { m } \right)$ is the average accuracy of the samples in the m-th bucket, and conf $\left( B _ { m } \right)$ is the corresponding average confidence of the samples falling in the m-th bucket.

$$
\sum _ { m = 1 } ^ { M } \frac { \left| B _ { m } \right| } { n } \left| \operatorname { a c c } \left( B _ { m } \right) - \operatorname { c o n f } \left( B _ { m } \right) \right|\tag{2}
$$

Consider a setup where we have buckets with a width of 0.1. All instances where a model assigns probabilities between 0.4 and 0.5 will be allocated to the bucket $B _ { 5 }$ or the bucket encompassing probabilities between 0.4 and 0.5. Subsequently, the average accuracy for the instances in these buckets and the average probability/confidence is computed. The absolute difference of the accuracy and confidence is multiplied by the proportion of total instances in a bucket. This process is repeated for every bucket, and the individual scores are summed up to calculate ECE.

## 2.2 Self-consistency as a measure of confidence

Self-consistency (Wang et al., 2022) is a natural language reasoning technique that uses chain-ofthought prompting to generate multiple paths for reasoning. This process aims to select the most consistent answer by sampling and marginalizing. Here, we use a latent variable Z to represent the reasoning chain/programs. Y is the answer that is either extracted in case of COT or obtained after execution in case of PAL. We marginalize over Z by taking a majority vote over answers. Thus, we rely on majority voting over the answers to obtain confidence estimates for each sample.

K is a hyperparameter that controls the number of generations (referenced in equation 3). The higher the value of K, the better our approximation of the probability of each sample. Figure 2 shows an overview of this process.

$$
P ( \hat { Y } _ { 0 } | Z _ { 0 } ) = \frac { 1 } { K } \sum _ { i = 0 } ^ { K } \mathbb { I } \left\{ \hat { Y } _ { i } = \hat { Y } _ { 0 } \right\}\tag{3}
$$

Wang et al. (2022) and Xiong et al. (2023a) suggest that self-consistency can be an effective way to elicit confidence from models. Hence, given the lack of per-token log probabilities in closed LMs like gpt-3.5-turbo and text-davinci-003, we adopt self-consistency as a proxy measure for calibration.

## 2.3 Similarity and Answer Entropy

In addition to empirically evaluating the impact on accuracy and calibration, we conduct a qualitative analysis of the reasoning chains (the latent variable $Z$ described previously). Here, we observe a consistent pattern, i.e. the correct answers corresponding to a question are often associated with similar generations.

We find that this observation aligns with the finding made by Li et al. (2022a), that there are numerous ways in which solutions can be incorrect. In contrast, correct solutions tend to exhibit more uniform behaviour.

To empirically validate this observation, we employed sentence embeddings generated from the all-MiniLM-v6 model to compute the average similarity among the generations/reasoning chains, equivalent to calculating similarity over latent variables $Z$

Furthermore, to gain deeper insights into the relationship between similarity in generations and corresponding answers, we compute the entropy $H ( A )$ of the answer space where $P ( a _ { i } )$ refers to the probability of the $\hat { i } ^ { t h }$ answer in K answers obtained by extraction or program execution for a given sample.

$$
\hat { H ( A ) } = - \sum _ { i = 1 } ^ { K } P ( a _ { i } ) \cdot \log _ { 2 } P ( a _ { i } )\tag{4}
$$

This allowed us to investigate whether the observed similarity in the latent variable space $Z$ leads to a lower entropy within the answer space.

## 3 Experimental Design

## 3.1 Models

We compare the calibration and accuracy of two different prompting strategies - CoT and PaL on an equal number of closed-source and open-source models. The open source models used in experimentation are LLaMA2-13B, LLaMA2-70B (Touvron et al., 2023). and the closed-source models are gpt-3.5-turbo, text-davinci-003 (Brown et al., 2020). It should be noted that all models have received some form of supervision from code during pre-training (OpenAI, 2023; Touvron et al., 2023), in addition to being primarily trained on text. For LLaMA models, we leveraged vLLM (Kwon et al., 2023) for distributed inference using A6000 GPU(s).

<table><tr><td>Dataset</td><td>Category</td><td># Samples</td><td>Example</td></tr><tr><td>GSM8K (Cobbe et al., 2021)</td><td>Arithmetic</td><td>1319</td><td>Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take? A: 3</td></tr><tr><td>GSM8K Hard (Gao et al., 2022)</td><td>Arithmetic</td><td>1319</td><td>Q: A robe takes 2287720 bolts of blue fiber and half that much white fiber. How many bolts in total does it take? A: 3431580</td></tr><tr><td>Date Understanding (Suzgun et al., 2022a) Symbolic</td><td></td><td>360</td><td>Q: Yesterday was April 30, 2021. What is the date today in MM/DD/YYYY? A: 05/01/2021</td></tr><tr><td>Object Counting (Suzgun et al., 2022a)</td><td>Algorithmic</td><td>250</td><td>Q: I have three couches, a lamp, a stove, a table, a fridge, and a microwave. How many objects do I have? A: 8</td></tr><tr><td>Repeat Copy (Suzgun et al., 2022a)</td><td>Algorithmic 32</td><td></td><td>Q: say python twice and data once, and then repeat all of this three times. A: python python data python python data python python data</td></tr></table>

Table 1: Datasets with their examples and categories.

## 3.2 Hyperparameters

For our experiments, we set temperature (T) as 1.0 and the probability (p) for nucleus sampling (Holtzman et al., 2020) as 1.0. Selecting a temperature of 1.0 enables direct sampling from the model as no scaling of probabilities is involved, as seen from Equation 5. Here, $z _ { i }$ refers to the logit for the ith token generated, and N is the vocabulary size.

$$
\sigma \left( z _ { i } \right) = \frac { e ^ { \frac { z _ { i } } { T } } } { \sum _ { j = 0 } ^ { N } e ^ { \frac { z _ { j } } { T } } }\tag{5}
$$

For use K = 10 generations per sample for all datasets. We set the maximum number of tokens (input + output) for each generation to 1024.

## 3.3 Tasks

We examined reasoning tasks encompassing several challenges, including arithmetic, algorithmic, and symbolic reasoning. We use five datasets that cover these different kinds of reasoning tasks. The arithmetic reasoning datasets include GSM8K (Cobbe et al., 2021) and GSM8K Hard (Gao et al., 2022). The algorithmic reasoning tasks include Object-Counting (Suzgun et al., 2022a) and Repeat-Copy (Suzgun et al., 2022a). We used Date-Understanding as a Symbolic Reasoning Dataset (Suzgun et al., 2022a). Specific information about the datasets used can be found in Table 1.

## 3.4 Prompt Design

We provide all models with natural language chain-of-thought (CoT) prompts and code-based Program-Aided Language Model (PaL) prompts. For datasets where CoT prompts are available in their original form, we use them as presented in the original paper (Wei et al., 2022). We modify these prompts for other datasets to suit the specific task while maintaining their original format. For PaL prompts, we use and adapt the code prompts provided in (Gao et al., 2022). The prompts are

included in Appendix A.

## 4 Results

We investigate two model types: OpenAI models and LLaMA models along with the two different prompting strategies - PAL and COT.

## 4.1 Effect of prompting style on Calibration

In this section, we look at the first two RQs: RQ 1: Does one prompting style result in significantly better calibration than the other? RQ 2: Are the observed calibration trends different across OpenAI models and LLaMA models?

Table 2 shows results for OpenAI models, we observe that PAL prompting improves both calibration and accuracy across all datasets. We see approximately 50% relative reduction in calibration error and an average improvement of 18.42% in accuracy.

In Figure 3, we show reliability plots which illustrate improved calibration, with the reliability curves for PAL prompting consistently aligning closer to the ideal reliability curve as compared to COT across datasets. While PAL shows a notable gain of 14.83% in accuracy across all datasets for LLaMA models, it shows better calibration in only half of our settings. Overall, for both OpenAI models and LLaMA models, we observe that PAL leads to better calibration than COT for 75% of the settings. The reliability plots for all datasets for the models gpt-3.5-turbo and LLaMA2-70B can be seen in Appendix Section E, D.

Effect of PAL on calibration controlling for accuracy One reasonable hypothesis is that PAL is improving calibration because it achieves higher accuracy, and more accurate models can be better calibrated. To examine this hypothesis, we conduct statistical analysis using mixed linear models (McLean et al., 1991), which allows us to consider the significance of varying the prompting strategy while controlling for accuracy as a confounding factor.

<table><tr><td rowspan=1 colspan=1>Name</td><td rowspan=1 colspan=1>Score  Model</td><td rowspan=1 colspan=10>GSM8K   Object-Counting Repeat-Copy Date-Understanding GSM8K Hard</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>CoT  PaL</td><td rowspan=1 colspan=2>CoT  PaL</td><td rowspan=1 colspan=2>CoT  PaL</td><td rowspan=1 colspan=2>CoT   PaL</td><td rowspan=1 colspan=2>CoT  PaL</td></tr><tr><td rowspan=10 colspan=1>LLaMA2-70BLLaMA2-13Btext-davinci-003</td><td rowspan=1 colspan=1>ECE (↓)LLaMA</td><td rowspan=1 colspan=1>0.19</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.23</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=1>ACC (↑)LLaMA</td><td rowspan=1 colspan=1>59.28</td><td rowspan=1 colspan=1>63.91</td><td rowspan=1 colspan=1>76.00</td><td rowspan=1 colspan=1>92.40</td><td rowspan=1 colspan=1>40.62</td><td rowspan=1 colspan=1>71.88</td><td rowspan=1 colspan=1>66.66</td><td rowspan=1 colspan=1>70.18</td><td rowspan=1 colspan=1>21.45</td><td rowspan=1 colspan=1>40.62</td></tr><tr><td rowspan=1 colspan=1>SIM (↑) LLaMA</td><td rowspan=1 colspan=1>72.20</td><td rowspan=1 colspan=1>92.40</td><td rowspan=1 colspan=1>94.43</td><td rowspan=1 colspan=1>94.72</td><td rowspan=1 colspan=1>87.10</td><td rowspan=1 colspan=1>90.58</td><td rowspan=1 colspan=1>86.87</td><td rowspan=1 colspan=1>82.15</td><td rowspan=1 colspan=1>92.28</td><td rowspan=1 colspan=1>74.32</td></tr><tr><td rowspan=1 colspan=1>ENT (↓)LLaMA</td><td rowspan=1 colspan=1>2.24</td><td rowspan=1 colspan=1>1.92</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>1.93</td><td rowspan=1 colspan=1>2.00</td><td rowspan=1 colspan=1>1.44</td><td rowspan=1 colspan=1>1.54</td><td rowspan=1 colspan=1>2.85</td><td rowspan=1 colspan=1>2.17</td></tr><tr><td rowspan=1 colspan=1>ECE (↓)LLaMA</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.14</td></tr><tr><td rowspan=1 colspan=1>ACC (↑)LLaMA</td><td rowspan=1 colspan=1>27.0</td><td rowspan=1 colspan=1>34.34</td><td rowspan=1 colspan=1>56.4</td><td rowspan=1 colspan=1>81.6</td><td rowspan=1 colspan=1>34.37</td><td rowspan=1 colspan=1>53.12</td><td rowspan=1 colspan=1>48.24</td><td rowspan=1 colspan=1>50.41</td><td rowspan=1 colspan=1>6.67</td><td rowspan=1 colspan=1>25.55</td></tr><tr><td rowspan=1 colspan=1>SIM (↑) LLaMA</td><td rowspan=1 colspan=1>76.6</td><td rowspan=1 colspan=1>93.3</td><td rowspan=1 colspan=1>93.2</td><td rowspan=1 colspan=1>95.3</td><td rowspan=1 colspan=1>89.8</td><td rowspan=1 colspan=1>88.6</td><td rowspan=1 colspan=1>79.5</td><td rowspan=1 colspan=1>84.2</td><td rowspan=1 colspan=1>74.0</td><td rowspan=1 colspan=1>92.32</td></tr><tr><td rowspan=1 colspan=1>ENT (↓)LLaMA</td><td rowspan=1 colspan=1>2.83</td><td rowspan=1 colspan=1>2.49</td><td rowspan=1 colspan=1>1.52</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>2.43</td><td rowspan=1 colspan=1>2.47</td><td rowspan=1 colspan=1>2.23</td><td rowspan=1 colspan=1>2.06</td><td rowspan=1 colspan=1>2.42</td><td rowspan=1 colspan=1>3.06</td></tr><tr><td rowspan=1 colspan=1>ECE (↓) OpenAI</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.19</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1>0.07</td></tr><tr><td rowspan=1 colspan=1>ACC (↑) OpenAI</td><td rowspan=1 colspan=1>65.65</td><td rowspan=1 colspan=1>76.49</td><td rowspan=1 colspan=1>59.21</td><td rowspan=1 colspan=1>98.00</td><td rowspan=1 colspan=1>67.23</td><td rowspan=1 colspan=1>93.75</td><td rowspan=1 colspan=1>60.70</td><td rowspan=1 colspan=1>72.35</td><td rowspan=1 colspan=1>23.95</td><td rowspan=1 colspan=1>71.27</td></tr><tr><td rowspan=6 colspan=1>gpt-3.5-turbo</td><td rowspan=1 colspan=1>SIM (↑) OpenAI</td><td rowspan=1 colspan=1>90.5</td><td rowspan=1 colspan=1>97.8</td><td rowspan=1 colspan=1>99.1</td><td rowspan=1 colspan=1>99.8</td><td rowspan=1 colspan=1>96.2</td><td rowspan=1 colspan=1>98.2</td><td rowspan=1 colspan=1>92.4</td><td rowspan=1 colspan=1>97.4</td><td rowspan=1 colspan=1>89.8</td><td rowspan=1 colspan=1>97.9</td></tr><tr><td rowspan=1 colspan=1>ENT (↓)OpenAI</td><td rowspan=1 colspan=1>1.27</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>1.38</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>2.31</td><td rowspan=1 colspan=1>0.81</td></tr><tr><td rowspan=1 colspan=1>ECE (↓) OpenAI</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>0.05</td></tr><tr><td rowspan=1 colspan=1>ACC (↑) OpenAI</td><td rowspan=1 colspan=1>84.00</td><td rowspan=1 colspan=1>82.40</td><td rowspan=1 colspan=1>82.40</td><td rowspan=1 colspan=1>97.20</td><td rowspan=1 colspan=1>56.25</td><td rowspan=1 colspan=1>68.75</td><td rowspan=1 colspan=1>61.51</td><td rowspan=1 colspan=1>77.23</td><td rowspan=1 colspan=1>55.21</td><td rowspan=1 colspan=1>62.91</td></tr><tr><td rowspan=1 colspan=1>SIM (↑) OpenAI</td><td rowspan=1 colspan=1>94.40</td><td rowspan=1 colspan=1>97.80</td><td rowspan=1 colspan=1>99.10</td><td rowspan=1 colspan=1>98.60</td><td rowspan=1 colspan=1>97.70</td><td rowspan=1 colspan=1>97.90</td><td rowspan=1 colspan=1>95.3</td><td rowspan=1 colspan=1>97.6</td><td rowspan=1 colspan=1>90.60</td><td rowspan=1 colspan=1>95.40</td></tr><tr><td rowspan=1 colspan=1>ENT (↓) OpenAI</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>0.048</td><td rowspan=1 colspan=1>1.15</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>1.65</td><td rowspan=1 colspan=1>2.43</td></tr></table>

Table 2: Comparison of Expected Calibration Error (ECE ( ) ) , Accuracy (ACC ( ) ) , Cosine Similarity (SIM ( ) ) <sup>and</sup> <sup>Answer</sup> <sup>Entropy</sup> <sup>(ENT</sup> <sup>(</sup>↓<sup>)</sup> <sup>)</sup> <sup>across</sup> <sup>datasets.</sup> <sup>The</sup> <sup>darker</sup> <sup>blue</sup> <sup>shade</sup> <sup>highlights</sup> <sup>better</sup> <sup>performing</sup> <sup>prompting</sup> technique.

Upon analyzing the results in Table 3, we observe that, when treating the prompting style as a fixed effect, PAL exhibits a negative coefficient of -0.103 (p=0.0) for OpenAI models, which is statistically significant with a threshold of p=0.05. This implies that PAL contributes to the reduction in ECE and has a positive impact on calibration. On the contrary, for LLaMA models, we did not find that PAL had a statistically significant effect on ECE after controlling for accuracy. Across LLaMA models and OpenAI models, PAL has a statistically significant (p=0.02) correlation of -0.067 with ECE, indicating that PAL helps increase calibration on the whole even when controlling for accuracy.

To summarize, we see that PAL prompting has better calibration than COT prompting (–RQ1) . While PAL has improved calibration in all settings for OpenAI models, this trend is less consistent for LLaMA models (–RQ2).

<table><tr><td>Model Type</td><td>LLaMA models</td><td>OpenAI models</td><td>Both</td></tr><tr><td>Fixed Effect (ECE vs Prompting Style)</td><td>PAL : -0.010</td><td>PAL : -0.103</td><td>PAL : -0.067</td></tr><tr><td>p-value</td><td>0.961</td><td>0.000</td><td>0.002</td></tr></table>

Table 3: Statistical analysis using mixed linear models, keeping ECE vs Prompting Style as a fixed effect and accuracy as a random effect.

## 4.2 Effect of generation diversity on calibration

In this section, we look at the third research question: RQ 3: Does the consistency of LLM generations affect calibration?

Qualitative analysis of the generations reveals that PAL generations adhere to a consistent structure that divides the problem-solving process into three distinct parts. This is depicted in Figure 4. In the first part, the model initializes the variables and sets up their initial values required for the calculation. This part is straightforward due to syntactic constraints and remains largely similar across generations. In the second part, the model generates the required logic by applying formulas and utilizing various operations to derive the desired result. Finally, in the third part, the model generates the answer by assigning the calculated result to a variable and returning it, which, again, doesn’t vary much across generations. Hence, the diversity of the generation is mainly limited to the second part, making code more constrained in its generation space compared to text. Therefore, there is a standardized structure in the code generated by language models with PaL prompts.

Lower generation diversity and answer entropy observed in prompting strategy with better calibration To quantitatively analyze if code-based generations have lower generation diversity and lead to a narrower answer space, we computed aggregated cosine similarity scores for all the generations and entropy over the answer space. For OpenAI models, we note that the cosine similarity scores with PAL are higher than the corresponding scores for COT. This observation suggests that code-based generations display a higher degree of similarity from a semantic perspective. Moreover, the answer entropy for PAL is lower than COT. This implies that similar generations that cluster together in the semantic space (Li et al., 2022a) also converge to the equivalent solution space. This leads to lower uncertainty in the probability distribution of the answer space and, hence, lower entropy. From Table 2, we thus can see that PAL helps produce similar generations that converge to the same answer space, which is also consistently correct. Hence, it achieves better performance and provides more confidence in its predictions.

![](images/b7d2c91641dbd2e53b391d9d4fee74d3c84c8d331f5cd6f61df5e1d69da4e584.jpg)

![](images/49851132c57d8c32066cbb771b2cbf10d15dcfabc0f01d4919cfe9b6566781d9.jpg)

![](images/abf6402fdbcf8d512b7817c317711fe8f82697a313b02ba288f06255a59caf4d.jpg)  
Figure 3: Reliability Plots for various structured reasoning tasks for the model gpt-3.5-turbo. The x-axis represents confidence, and the y-axis represents accuracy.

```ini
def solution () :
# Part 1: Initialize
num_glasses = 16
first_glass_price = 5
second_glass_discount = 0.6
# Part 2: Calculate
second_glass_price = first_glass_price *
second_glass_discount
pair_price = first_glass_price +
second_glass_price
num_pairs = num_glasses // 2
total_cost = num_pairs * pair_price
# Part 3: Result Generation
result = total_cost
return result
Figure 4: Typical output structure with PaL
```

For LLaMA models, we don’t see this trend of PAL having higher generation similarity and lower answer entropy for all datasets. However, for almost all settings for LLaMA models and OpenAI models, the prompting strategy that produces more similar generations and lower answer entropy is

also more calibrated.

To summarize, it is evident that lower generation diversity and lower answer entropy are correlated with higher calibration. (–RQ3)

Better calibration observed for PAL when inducing similarity in generations for LLaMA models We observe that for OpenAI models, PAL is not only more accurate but also more calibrated than COT. Consequently, we explore whether the reduction in generation diversity, achievable through lower temperatures, can contribute to improved calibration for LLaMA models.

We perform a parameter sweep across temperature values between 0.1 and 0.7 with a step size of 0.2. We show the variation of accuracy, calibration, generation similarity, and answer entropy for two datasets in Figure 5. The plots for the remaining datasets are available in Appendix B, Figure 6. We can see that we obtain better calibration for both the LLaMA models in both PAL and COT for temperatures below 1.0. From Tables 4 and 5, we note that in the majority of runs with T < 1.0, PAL is better calibrated than COT. Considering accuracy and calibration, optimal performance is achieved at different temperatures for each dataset. For most T values, the similarity scores are higher while corresponding answer entropy values are lower for PAL compared to COT. This mirrors the pattern observed for OpenAI models. For LLaMA-2 13b, PAL displays better calibration than COT at lower temperatures. However, the optimal temperature for obtaining the best performance for calibration and accuracy is still T=1.0.

<table><tr><td colspan="2">Temp</td><td colspan="2">GSM8K</td><td colspan="2">Object-Counting</td><td colspan="2">Repeat-Copy</td><td colspan="2">Date-Understanding</td><td colspan="2">GSM8K Hard</td></tr><tr><td colspan="2"></td><td>CoT</td><td>PaL</td><td>CoT</td><td>PaL</td><td>CoT</td><td>PaL</td><td>CoT</td><td>PaL</td><td>CoT</td><td>PaL</td></tr><tr><td rowspan="4">0.7</td><td>ECE ACC</td><td>0.101</td><td>0.07</td><td>0.076</td><td>0.03</td><td>0.14</td><td>0.12</td><td>0.12</td><td>0.09</td><td>0.18</td><td>0.03</td></tr><tr><td></td><td>66.03</td><td>67.9</td><td>77.6</td><td>93.2</td><td>53.1</td><td>75.0</td><td>74.5</td><td>76.42</td><td>27.14</td><td>52.91</td></tr><tr><td>SIM</td><td>85.07</td><td>97.47</td><td>98.53</td><td>99.42</td><td>93.78</td><td>94.81</td><td>89.62</td><td>96.16</td><td>83.28</td><td>97.29</td></tr><tr><td>ENT ECE</td><td>1.60</td><td>1.48</td><td>0.55</td><td>0.21</td><td>1.46</td><td>1.35</td><td>0.88</td><td>0.80</td><td>2.43</td><td>1.72</td></tr><tr><td rowspan="4">0.5</td><td>ACC</td><td>0.049</td><td>0.036</td><td>0.103</td><td>0.059</td><td>0.112</td><td>0.075</td><td>0.114</td><td>0.063</td><td>0.139</td><td>0.104</td></tr><tr><td></td><td>66.94</td><td>67.24</td><td>77.23</td><td>92.4</td><td>59.3</td><td>68.75</td><td>73.44</td><td>77.2</td><td>27.7</td><td>51.63</td></tr><tr><td>SIM</td><td>88.69</td><td>98.25</td><td>99.17</td><td>99.85</td><td>97.09</td><td>96.81</td><td>92.49</td><td>97.97</td><td>87.65</td><td>98.2</td></tr><tr><td>ENT</td><td>1.35</td><td>1.19</td><td>0.39</td><td>0.12</td><td>1.09</td><td>0.99</td><td>0.60</td><td>0.52</td><td>2.18</td><td>1.39</td></tr><tr><td rowspan="4">0.3</td><td>ECE ACC</td><td>0.057</td><td>0.097</td><td>0.140</td><td>0.064</td><td>0.194</td><td>0.113</td><td>0.153</td><td>0.139</td><td>0.230</td><td>0.206</td></tr><tr><td></td><td>64.89</td><td>63.38</td><td>78.8</td><td>91.2</td><td>53.12</td><td>71.87</td><td>72.62</td><td>76.42</td><td>26.16</td><td>49.28</td></tr><tr><td>SIM</td><td>91.91</td><td>98.75</td><td>99.51</td><td>99.94</td><td>97.73</td><td>98.27</td><td>95.18</td><td>99.02</td><td>91.14</td><td>98.75</td></tr><tr><td>ENT</td><td>1.087</td><td>0.960</td><td>0.238</td><td>0.056</td><td>0.780</td><td>0.504</td><td>0.420</td><td>0.317</td><td>1.866</td><td>1.076</td></tr><tr><td rowspan="4">0.1</td><td>ECE</td><td>0.219</td><td>0.257</td><td>0.188</td><td>0.07</td><td>0.278</td><td>0.156</td><td>0.233</td><td>0.176</td><td>0.418</td><td>0.380</td></tr><tr><td>ACC</td><td>58.6</td><td>58.37</td><td>77.2</td><td>90.4</td><td>53.12</td><td>68.75</td><td>69.91</td><td>78.32</td><td>23.5</td><td>45.87</td></tr><tr><td>SIM</td><td>95.79</td><td>99.37</td><td>99.82</td><td>99.98</td><td>99.28</td><td>99.64</td><td>98.21</td><td>99.68</td><td>95.31</td><td>99.35</td></tr><tr><td>ENT</td><td>0.661</td><td>0.526</td><td>0.085</td><td>0.026</td><td>0.288</td><td>0.173</td><td>0.195</td><td>0.137</td><td>1.179</td><td>0.540</td></tr></table>

Table 4: Results of temperature scaling for LLaMA2-70B. The darker blue shade highlights better performing prompting technique.

For LLaMA-2 70b, optimal temperature values in our runs for calibration are either 0.5 or 0.7, while extreme values (0.1, 1.0) yield lower calibration and accuracy performance. We can, therefore see that scaling temperatures in the LLaMA models can help us to obtain better calibration for PAL, specifically for the LLaMA-2 70b, which already performs better than COT on these reasoning tasks. Thus, we do see that lower generation diversity and lower answer entropy lead to higher calibration up to a certain point, after which it negatively affects the calibration. (–RQ3)

## 5 Related Work

## 5.1 Prompting Strategies for Reasoning

Recent developments in language models have introduced various methods to enhance their reasoning abilities. One such method is CoT (Wei et al., 2022), which helps models generate a series of intermediate steps to solve problems. CoT has demonstrated improved performance in arithmetic, common sense, and symbolic reasoning tasks. There are approaches such as PaL (Gao et al., 2022) and Program-of-thoughts (PoT) (Chen et al., 2022), which go a step further by generating programs as intermediate steps and using an interpreter to process them. Code as a medium of reasoning has shown considerable promise, evidenced by better performance over chain-of-thought style prompting strategies in several recent studies (Madaan et al., 2022; Gao et al., 2022; Lyu et al., 2023; Zhang et al., 2023a,b). Unlike these works, our primary goal in this paper is to understand the effect of code prompts on calibration.

## 5.2 Calibration in Language Models

Calibration has been extensively studied in structured prediction problems, such as named entity recognition and part of speech tagging (Jagannatha and Yu, 2020), as well as in natural language understanding tasks, like question answering and text classification (Kamath et al., 2020; Kong et al., 2020; Desai and Durrett, 2020). More recently, studies have focused on calibrating language models when used as generators (Jiang et al., 2021; Zhao et al., 2021). Additionally, the study by Kadavath et al. (2022) explored the likelihood of a model knowing the answer before proposing a response. However, these approaches typically rely on access to the model’s logits.

In contrast, the work by (Tian et al., 2023) investigates verbalized probability estimates to assess the calibration of large language models without needing access to logits. This involves querying the model about its confidence in the answers it generates. Furthermore, (Xiong et al., 2023b) introduced self-consistency-based methods for calibration, demonstrating their superior performance compared to verbalized methods. In our research, we adopt self-consistency as the method of choice for measuring calibration.

![](images/551ce852c9aecd449d05b86585319d71f20cffae8ea0f864ecbcaa7a237d2206.jpg)  
Figure 5: Trends seen in temperature scaling for the model LLaMA2-70B. Across datasets, the accuracy and calibration improve upon lowering the temperature to a certain extent. This is in line with having lower generation similarity and lower answer entropy. The optimal temperatures seen are 0.5 and 0.7 across datasets. For other datasets, refer Appendix, Figure 6.

## 5.3 Utilizing Language for Code Generation

The exploration of using natural language for code generation has taken diverse approaches in research. Initial efforts involved rule-based, predictive and deep-learning variations (Gulwani and Marron, 2014; Woods, 1973; Zelle and Mooney, 1996; Lin et al., 2017; Rabinovich et al., 2017). However, performance enhancements were observed using pre-trained models trained on code-based datasets (Chen et al., 2021; Nijkamp et al., 2022; Gao et al., 2022; Li et al., 2023). Employing pre-trained language models (LMs) for code generation as a way to solve tasks that require step-by-step structuring and various forms of reasoning has proven to be particularly effective (Ni et al., 2023a; Gao et al., 2022; Ni et al., 2023b).

Intermediate execution results from code have been used for training (Chen et al., 2018) and inference (Wang et al., 2018). Majority-based voting on the results of code executions (which is the selfconsistency-based methodology we employ) has also been shown to be an effective technique for selecting the right candidate (Li et al., 2022b; Cobbe et al., 2021; Shi et al., 2022).

## 6 Conclusion

In this study, we explore the impact of two distinct prompting styles, namely PAL and COT, on the calibration of OpenAI models and LLaMA models. Our investigation spans 5 reasoning datasets, employing self-consistency as the methodology for eliciting calibration. We analyze four different metrics - calibration (ECE) , accuracy (ACC) , average similarity in generations (SIM) , and answer entropy (ENT) . Our findings are as follows:

• RQ 1: Does one prompting style result in significantly better calibration than the other? Empirical results show that PAL generally has higher calibration and accuracy for 82.5% of the cases across OpenAI and LLaMA models for a varied range of temperatures.

• RQ 2: Are the observed calibration trends different across OpenAI models and LLaMA models? We observed that OpenAI models are in general better calibrated for the reasoning tasks with up to 19% improvement in ECE.

• RQ 3: Does the consistency of LLM generations affect performance? PAL prompting shows a general trend of having greater similarity in the generation over COT, which we hypothesize could be due to the inherent structure present in the code. We see that greater generation similarity is accompanied by lower answer entropy and lower ECE.

We hope that this study will catalyze additional research aimed at holistically evaluating and gaining deeper insights into the role of prompts in various tasks and domains.

## 7 Limitations

Access to OpenAI models is only available through an API which limits the ability to exactly control the hyperparameters influencing the generations. Moreover, OpenAI models are not transparent, which limits the ability to study these models. Because of this lack of transparency it is also hard to draw conclusive insights about any comparisons between OpenAI models and LLaMA models In our study, we report the results from a single run but due to combination of utilizing temperature value of 1.0 and hardware induced stochasticity, it is possible to get varying results for a given model.

## References

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2022. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. ArXiv, abs/2211.12588.

Xinyun Chen, Chang Liu, and Dawn Xiaodong Song. 2018. Execution-guided neural program synthesis. In International Conference on Learning Representations.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. ArXiv, abs/2110.14168.

Shrey Desai and Greg Durrett. 2020. Calibration of pre-trained transformers. arXiv preprint arXiv:2003.07892.

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2022. Pal: Program-aided language models. ArXiv, abs/2211.10435.

Sumit Gulwani and Mark Marron. 2014. Nlyze: Interactive programming by natural language for spreadsheet data analysis and manipulation. In Proceedings ofthe 2014 ACM SIGMOD international conference on Management ofdata, pages 803–814.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. 2017a. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017b. On calibration of modern neural networks. In International Conference on Machine Learning.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The curious case of neural text degeneration.

Abhyuday Jagannatha and Hong Yu. 2020. Calibrating structured output predictors for natural language processing. In Proceedings of the conference. Association for Computational Linguistics. Meeting, volume 2020, page 2078. NIH Public Access.

Zhengbao Jiang, J. Araki, Haibo Ding, and Graham Neubig. 2020. How can we know when language models know? on the calibration of language models for question answering. Transactions ofthe Associationfor Computational Linguistics, 9:962–977.

Zhengbao Jiang, Jun Araki, Haibo Ding, and Graham Neubig. 2021. How can we know when language models know? on the calibration of language models for question answering. Transactions ofthe Associationfor Computational Linguistics, 9:962–977.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. 2022. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221.

Amita Kamath, Robin Jia, and Percy Liang. 2020. Selective question answering under domain shift. arXiv preprint arXiv:2006.09462.

Lingkai Kong, Haoming Jiang, Yuchen Zhuang, Jie Lyu, Tuo Zhao, and Chao Zhang. 2020. Calibrated language model fine-tuning for in-and out-ofdistribution data. arXiv preprint arXiv:2010.11506.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Haotong Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. Proceedings of the 29th Symposium on Operating Systems Principles.

Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. 2023. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. 2022a. Competition-level code generation with alphacode. Science, 378(6624):1092–1097.

Yujia Li, David H. Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom, Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de,

Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey, Cherepanov, James Molloy, Daniel Jaymin Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de, Freitas, Koray Kavukcuoglu, and Oriol Vinyals. 2022b. Competition-level code generation with alphacode. Science, 378:1092 – 1097.

Xi Victoria Lin, Chenglong Wang, Deric Pang, Kevin Vu, and Michael D Ernst. 2017. Program synthesis from natural language using recurrent neural networks. University of Washington Department of Computer Science and Engineering, Seattle, WA, USA, Tech. Rep. UW-CSE-17-03-01.

Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. 2023. Faithful chain-ofthought reasoning. arXiv preprint arXiv:2301.13379.

Aman Madaan, Shuyan Zhou, Uri Alon, Yiming Yang, and Graham Neubig. 2022. Language models of code are few-shot commonsense learners. ArXiv, abs/2210.07128.

Robert A McLean, William L Sanders, and Walter W Stroup. 1991. A unified approach to mixed linear models. The American Statistician, 45(1):54–64.

Ansong Ni, Srini Iyer, Dragomir Radev, Veselin Stoyanov, Wen-tau Yih, Sida Wang, and Xi Victoria Lin. 2023a. Lever: Learning to verify language-to-code generation with execution. In International Conference on Machine Learning, pages 26106–26128. PMLR.

Ansong Ni, Pengcheng Yin, Yilun Zhao, Martin Riddell, Troy Feng, Rui Shen, Stephen Yin, Ye Liu, Semih Yavuz, Caiming Xiong, Shafiq R. Joty, Yingbo Zhou, Dragomir R. Radev, and Arman Cohan. 2023b. L2ceval: Evaluating language-to-code generation capabilities of large language models. ArXiv, abs/2309.17446.

Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. 2022. A conversational paradigm for program synthesis.

OpenAI. 2023. Openai documentation. https://platform.openai.com/docs/ model-index-for-researchers.

John Platt et al. 1999. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers, 10(3):61–74.

Maxim Rabinovich, Mitchell Stern, and Dan Klein. 2017. Abstract syntax networks for code generation and semantic parsing. arXiv preprint arXiv:1704.07535.

Freda Shi, Daniel Fried, Marjan Ghazvininejad, Luke Zettlemoyer, and Sida I. Wang. 2022. Natural language to code translation with execution. ArXiv, abs/2204.11454.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. 2022a. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261.

Mirac Suzgun, Nathan Scales, Nathanael Scharli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed Huai hsin Chi, Denny Zhou, and Jason Wei. 2022b. Challenging big-bench tasks and whether chain-of-thought can solve them. ArXiv, abs/2210.09261.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D Manning. 2023. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. arXiv preprint arXiv:2305.14975.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Chenglong Wang, Kedar Tatwawadi, Marc Brockschmidt, Po-Sen Huang, Yi Mao, Oleksandr Polozov, and Rishabh Singh. 2018. Robust text-to-sql generation with execution-guided decoding. arXiv: Computation and Language.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Huai hsin Chi, and Denny Zhou. 2022. Selfconsistency improves chain of thought reasoning in language models. ArXiv, abs/2203.11171.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Huai hsin Chi, F. Xia, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. ArXiv, abs/2201.11903.

William A Woods. 1973. Progress in natural language understanding: an application to lunar geology. In Proceedings ofthe June 4-8, 1973, national computer conference and exposition, pages 441–450.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. 2023a. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. ArXiv, abs/2306.13063.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. 2023b. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. arXiv preprint arXiv:2306.13063.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. ArXiv, abs/2305.10601.

John M Zelle and Raymond J Mooney. 1996. Learning to parse database queries using inductive logic programming. In Proceedings ofthe national conference on artificial intelligence, pages 1050–1055.

Li Zhang, Liam Dugan, Hai Xu, and Chris Callison-Burch. 2023a. Exploring the curious case of code prompts. ArXiv, abs/2304.13250.

Li Zhang, Hai Xu, Yue Yang, Shuyan Zhou, Weiqiu You, Manni Arora, and Chris Callison-Burch. 2023b. Causal reasoning of entities and events in procedural texts. In Findings.

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In International Conference on Machine Learning, pages 12697–12706. PMLR.

Denny Zhou, Nathanael Scharli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Olivier Bousquet, Quoc Le, and Ed Huai hsin Chi. 2022. Least-to-most prompting enables complex reasoning in large language models. ArXiv, abs/2205.10625.

## A Prompts

The following sections display one example of the few-shot prompts used for each dataset across prompting styles.

## A.1 PAL Prompts

## A.1.1 GSM8K/GSM8K-Hard

```python
def solution () :
"""Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""
money_initial = 23
bagels = 5
bagel_cost = 3
money_spent = bagels * bagel_cost
money_left = money_initial - money_spent
result = money_left
return result
```

## A.1.2 Object Counting

```python
# Q: I have a chair, two potatoes, a cauliflower, a lettuce head, two tables, a cabbage, two
<sub>,</sub> onions, and three fridges. How many vegetables do I have?
def solution () :
# note: I'm not counting the chair, tables, or fridges
vegetables_to_count = {{'potato': 2,'cauliflower': 1,'lettuce head': 1,'cabbage':
<sub>,</sub> 1,'onion': 2}}
return sum (vegetables_to_count.values () )
```

## A.1.3 Date Understanding

```python
# Q: 2015 is coming in 36 hours. What is the date one week from today in MM/DD/YYYY?
# If 2015 is coming in 36 hours, then today is 36 hours before.
today = datetime (2015, 1, 1) - relativedelta (hours=36)
# One week from today,
one_week_from_today = today + relativedelta (weeks=1)
# The answer formatted with %m/%d/%Y is
one_week_from_today.strftime ('%m/%d/%Y')
```

## A.1.4 Repeat Copy

```python
# Q: Repeat the word duck four times, but halfway through also say quack
def solution () :
result = []
for i in range (1, 5) :
result.append ("duck")
if i == 2:
result.append ("quack")
return " ".join (result)
```

## A.2 COT Prompts

## A.2.1 GSM8K/GSM8K-Hard

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

## A.2.2 Object Counting

Q: I have a chair, two potatoes, a cauliflower, a lettuce head, two tables, a cabbage, two onions, and three fridges. How many vegetables do I have?

A: Chair, tables and fridges are not vegetables, so we are not counting them. Two potatoes, cauliflower, lettuce head, cabbage and two onions are vegetables, so they will be counted The total number of vegetables are 2 + 1 + 1 + 1 + 2 The answer is: 7

## A.2.3 Date Understanding

Q: 2015 is coming in 36 hours. What is the date one week from today in MM/DD/YYYY?

A: If 2015 is coming in 36 hours, then it is coming in 2 days. 2 days before 01/01/2015 is 12/30/2014, so today is 12/30/2014. So one week from today will be 01/05/2015. So the answer is 01/05/2015.

## A.2.4 Repeat Copy

Q: Repeat the word duck four times, but halfway through also say quack

A: On repeating the word duck four times we get: duck duck duck duck Halfway through if we say quack, we have to say quack in between the 2nd word and the 3rd word The answer is: duck duck quack duck duck

![](images/a9b95fb89516853b1c8f85d520202274213aa47a2b746c69d95876d1d94eaf7e.jpg)  
Figure 6: Trends seen in temperature scaling for the datasets - GSM8K-Hard, Date-Understanding and Repeat-Copy

## C Results of temperature scaling for LLaMA2-13B

<table><tr><td rowspan=1 colspan=1>Temp</td><td rowspan=1 colspan=10>GSM8K     Object-Counting   Repeat-Copy   Date-Understanding   GSM8K Hard</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>CoT   PaL</td><td rowspan=1 colspan=2>CoT   PaL</td><td rowspan=1 colspan=2>CoT   PaL</td><td rowspan=1 colspan=2>CoT     PaL</td><td rowspan=1 colspan=2>CoT   PaL</td></tr><tr><td rowspan=1 colspan=1>0.7  ECE</td><td rowspan=1 colspan=1>0.052</td><td rowspan=1 colspan=1>0.046</td><td rowspan=1 colspan=2>0.12   0.108</td><td rowspan=1 colspan=1>0.175</td><td rowspan=1 colspan=1>0.181</td><td rowspan=1 colspan=1>0.108</td><td rowspan=1 colspan=1>0.107</td><td rowspan=1 colspan=1>0.125</td><td rowspan=1 colspan=1>0.087</td></tr><tr><td rowspan=1 colspan=1>ACC</td><td rowspan=1 colspan=1>36.16</td><td rowspan=1 colspan=1>39.27</td><td rowspan=1 colspan=1>61.60</td><td rowspan=1 colspan=1>80.80</td><td rowspan=1 colspan=1>34.37</td><td rowspan=1 colspan=1>59.37</td><td rowspan=1 colspan=1>50.60</td><td rowspan=1 colspan=1>56.63</td><td rowspan=1 colspan=1>10.91</td><td rowspan=1 colspan=1>30.32</td></tr><tr><td rowspan=1 colspan=1>SIM</td><td rowspan=1 colspan=1>85.40</td><td rowspan=1 colspan=1>97.48</td><td rowspan=1 colspan=1>98.65</td><td rowspan=1 colspan=1>99.33</td><td rowspan=1 colspan=1>95.07</td><td rowspan=1 colspan=1>94.02</td><td rowspan=1 colspan=1>86.81</td><td rowspan=1 colspan=1>96.03</td><td rowspan=1 colspan=1>83.48</td><td rowspan=1 colspan=1>97.37</td></tr><tr><td rowspan=1 colspan=1>ENT</td><td rowspan=1 colspan=1>2.405</td><td rowspan=1 colspan=1>2.222</td><td rowspan=1 colspan=2>0.970  0.303</td><td rowspan=1 colspan=1>2.048</td><td rowspan=1 colspan=1>1.949</td><td rowspan=1 colspan=1>1.505</td><td rowspan=1 colspan=1>1.330</td><td rowspan=1 colspan=1>2.892</td><td rowspan=1 colspan=1>2.332</td></tr><tr><td rowspan=1 colspan=1>0.5  ECE</td><td rowspan=1 colspan=1>0.099</td><td rowspan=1 colspan=1>0.093</td><td rowspan=1 colspan=2>0.171  0.135</td><td rowspan=1 colspan=1>0.131</td><td rowspan=1 colspan=1>0.106</td><td rowspan=1 colspan=1>0.134</td><td rowspan=1 colspan=1>0.183</td><td rowspan=1 colspan=1>0.177</td><td rowspan=1 colspan=1>0.1381</td></tr><tr><td rowspan=1 colspan=1>ACC</td><td rowspan=1 colspan=1>36.08</td><td rowspan=1 colspan=1>37.07</td><td rowspan=1 colspan=1>60.80</td><td rowspan=1 colspan=1>82.00</td><td rowspan=1 colspan=1>34.37</td><td rowspan=1 colspan=1>59.30</td><td rowspan=1 colspan=1>53.92</td><td rowspan=1 colspan=1>55.00</td><td rowspan=1 colspan=1>10.80</td><td rowspan=1 colspan=1>29.87</td></tr><tr><td rowspan=1 colspan=1>SIM</td><td rowspan=1 colspan=1>88.70</td><td rowspan=1 colspan=1>98.10</td><td rowspan=1 colspan=1>99.21</td><td rowspan=1 colspan=1>99.83</td><td rowspan=1 colspan=1>97.12</td><td rowspan=1 colspan=1>97.74</td><td rowspan=1 colspan=1>89.94</td><td rowspan=1 colspan=1>97.63</td><td rowspan=1 colspan=1>87.37</td><td rowspan=1 colspan=1>98.03</td></tr><tr><td rowspan=1 colspan=1>ENT</td><td rowspan=1 colspan=1>2.144</td><td rowspan=1 colspan=1>1.976</td><td rowspan=1 colspan=1>0.722</td><td rowspan=1 colspan=1>0.158</td><td rowspan=1 colspan=1>1.455</td><td rowspan=1 colspan=1>1.311</td><td rowspan=1 colspan=1>1.141</td><td rowspan=1 colspan=1>0.942</td><td rowspan=1 colspan=1>2.694</td><td rowspan=1 colspan=1>2.074</td></tr><tr><td rowspan=1 colspan=1>0.3  ECE</td><td rowspan=1 colspan=1>0.160</td><td rowspan=1 colspan=1>0.108</td><td rowspan=1 colspan=1>0.231</td><td rowspan=1 colspan=1>0.154</td><td rowspan=1 colspan=1>0.200</td><td rowspan=1 colspan=1>0.206</td><td rowspan=1 colspan=1>0.219</td><td rowspan=1 colspan=1>0.304</td><td rowspan=1 colspan=1>0.246</td><td rowspan=1 colspan=1>0.231</td></tr><tr><td rowspan=1 colspan=1>ACC</td><td rowspan=1 colspan=1>33.81</td><td rowspan=1 colspan=1>30.72</td><td rowspan=1 colspan=1>62.40</td><td rowspan=1 colspan=1>81.20</td><td rowspan=1 colspan=1>37.50</td><td rowspan=1 colspan=1>65.63</td><td rowspan=1 colspan=1>55.28</td><td rowspan=1 colspan=1>50.40</td><td rowspan=1 colspan=1>10.16</td><td rowspan=1 colspan=1>27.97</td></tr><tr><td rowspan=1 colspan=1>SIM</td><td rowspan=1 colspan=1>91.51</td><td rowspan=1 colspan=1>98.54</td><td rowspan=1 colspan=1>99.56</td><td rowspan=1 colspan=1>99.93</td><td rowspan=1 colspan=1>97.62</td><td rowspan=1 colspan=1>96.81</td><td rowspan=1 colspan=1>93.02</td><td rowspan=1 colspan=1>98.41</td><td rowspan=1 colspan=1>90.48</td><td rowspan=1 colspan=1>98.44</td></tr><tr><td rowspan=1 colspan=1>ENT</td><td rowspan=1 colspan=1>1.826</td><td rowspan=1 colspan=1>1.618</td><td rowspan=1 colspan=1>0.475</td><td rowspan=1 colspan=1>0.080</td><td rowspan=1 colspan=1>0.967</td><td rowspan=1 colspan=1>1.19</td><td rowspan=1 colspan=1>79.04</td><td rowspan=1 colspan=1>63.78</td><td rowspan=1 colspan=1>2.389</td><td rowspan=1 colspan=1>1.716</td></tr><tr><td rowspan=1 colspan=1>0.1  ECE</td><td rowspan=1 colspan=1>0.372</td><td rowspan=1 colspan=1>0.334</td><td rowspan=1 colspan=1>0.311</td><td rowspan=1 colspan=1>0.174</td><td rowspan=1 colspan=1>0.4969</td><td rowspan=1 colspan=1>0.1812</td><td rowspan=1 colspan=1>0.341</td><td rowspan=1 colspan=1>0.423</td><td rowspan=1 colspan=1>0.372</td><td rowspan=1 colspan=1>0.334</td></tr><tr><td rowspan=1 colspan=1>ACC</td><td rowspan=1 colspan=1>30.25</td><td rowspan=1 colspan=1>32.14</td><td rowspan=1 colspan=1>62.80</td><td rowspan=1 colspan=1>81.20</td><td rowspan=1 colspan=1>37.50</td><td rowspan=1 colspan=1>46.80</td><td rowspan=1 colspan=1>54.47</td><td rowspan=1 colspan=1>49.32</td><td rowspan=1 colspan=1>30.25</td><td rowspan=1 colspan=1>32.14</td></tr><tr><td rowspan=1 colspan=1>SIM</td><td rowspan=1 colspan=1>95.12</td><td rowspan=1 colspan=1>99.19</td><td rowspan=1 colspan=1>99.80</td><td rowspan=1 colspan=1>99.98</td><td rowspan=1 colspan=1>97.24</td><td rowspan=1 colspan=1>98.68</td><td rowspan=1 colspan=1>97.00</td><td rowspan=1 colspan=1>99.32</td><td rowspan=1 colspan=1>95.12</td><td rowspan=1 colspan=1>99.19</td></tr><tr><td rowspan=1 colspan=1>ENT</td><td rowspan=1 colspan=1>1.145</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.204</td><td rowspan=1 colspan=1>0.014</td><td rowspan=1 colspan=1>0.286</td><td rowspan=1 colspan=1>0.283</td><td rowspan=1 colspan=1>0.373</td><td rowspan=1 colspan=1>0.261</td><td rowspan=1 colspan=1>1.1458</td><td rowspan=1 colspan=1>0.840</td></tr></table>

Table 5: Results of temperature scaling for LLaMA2-13B. The darker blue shade highlights better performing prompting technique.

![](images/1ce5feb2de912635a05d32d966758c3e97838898449f604b1865cca0351b04f6.jpg)  
(a) GSM8k CoT

![](images/7a31b1f4ac70ef03de8ec1c77aad7e5c7bad0df057257478b34c3291decc419a.jpg)  
(b) GSM8k PaL

![](images/126daec4baf5a64703189a43943c49ddb96dbc04f202b85b878b3a3c0548e98b.jpg)  
(c) Date Understanding CoT

![](images/685e54100ae7055aabcddc0ba5dabc8d305760f7b1260311ecd54b8bc7f9e767.jpg)  
(d) Date Understanding PaL

![](images/103b62501adad6173b93192b7d5c31a04e96d89010c24ec56d8eeb7afc25d320.jpg)  
(e) Object Counting CoT

![](images/bd3cf383cabed240ccfaf50176c72c960b3e5175df22b368873747e23b20096f.jpg)  
(f) Object Counting PaL

![](images/787d987400db8672d22ea929eea29561e9ae022340cb39083a0b353d6ae09d26.jpg)  
(g) Repeat Copy CoT

![](images/d031603b3e8817b82e84977010db07074c252622f0fcdd1b87fc103d9693dcb1.jpg)  
(h) Repeat Copy PaL

![](images/c560ebd9d65041499a30ebc4f95e7a3744d3ab114dda410bdc0a009789404603.jpg)  
(i) GSM8k Hard CoT

![](images/ece47f51e232e7d917c18210820d5f0fe5c891764ec31ec3fbe58a544ff5f880.jpg)  
(j) GSM8k Hard PaL  
Figure 7: Reliability plots for all the datasets using COT and PAL prompting for the model LLaMA2-70B

![](images/1d0c757443e242fdaf845a32a3732987b59526d2fbb17ca7218e9713d99acc46.jpg)  
(a) GSM8k CoT

![](images/4cb9649ed3eecaa61e77b44e50e7a5c1fa134fcf6f9dc0e30b30625e74eb53a2.jpg)  
(b) GSM8k PaL

![](images/f76820fff22d9158bb9f0346d56b18b78f29028f4a6fa6c00a09c55f5ca5376a.jpg)  
(c) Date Understanding CoT

![](images/68912e002a8d1815608a94eaa333aebe563eb57a9f8b584aef5f4ca9c6a21989.jpg)  
(d) Date Understanding PaL

![](images/ac15119187030cdaaf168ed9469d1d77413c073ef50aad6fc700cecf96209038.jpg)  
(e) Object Counting CoT

![](images/19e673ce2f0b8525078ccbe681778d486de0ef08b7d4cdba3a43754e9a7af72d.jpg)  
(f) Object Counting PaL

![](images/47482f5edd8bdc781f7fe0530bf034f67e74e8d881a1cfb43e4d9df6797e9761.jpg)  
(g) Repeat Copy CoT

![](images/49d58ecea6ff4ab3c61c6fd962eab6a4da3d00ccd06b963e0f43cc95b6956d08.jpg)

![](images/e242f442d0128af8972ae2b4be6b1eba60e6b55624573d62e658ecfe903fb556.jpg)  
(i) GSM8k Hard CoT

(h) Repeat Copy PaL  
![](images/06c9204f7e9f3e4a75d455edd23afa43166ecee498bdbf2ddabb6bd44ffc2bf7.jpg)  
(j) GSM8k Hard PaL  
Figure 8: Reliability plots for all the datasets using COT and PAL prompting for the model gpt-3.5-turbo