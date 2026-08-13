# Complex Claim Verification with Evidence Retrieved in the Wild

Jifan Chen Grace Kim Aniruddh Sriram Greg Durrett Eunsol Choi Department of Computer Science The University of Texas at Austin jf\_chen@utexas.edu

## Abstract

Retrieving evidence to support or refute claims is a core part of automatic fact-checking. Prior work makes simplifying assumptions in retrieval that depart from real-world use cases: either no access to evidence, access to evidence curated by a human fact-checker, or access to evidence published after a claim was made. In this work, we present the first realistic pipeline to check real-world claims by retrieving raw evidence from the web. We restrict our retriever to only search documents available prior to the claim’s making, modeling the realistic scenario of emerging claims. Our pipeline includes five components: claim decomposition, raw document retrieval, fine-grained evidence retrieval, claim-focused summarization, and veracity judgment. We conduct experiments on complex political claims in the CLAIMDE-COMP dataset and show that the aggregated evidence produced by our pipeline improves veracity judgments. Human evaluation finds the evidence summary produced by our system is reliable (it does not hallucinate information) and relevant to answering key questions about a claim, suggesting that it can assist fact-checkers even when it does not reflect a complete evidence set.<sup>1</sup>

## 1 Introduction

To combat the rise of misinformation, the NLP community has developed automatic fact-checking tools. However, these automated systems are not ready for wide adoption at real fact-checking organizations. Prior work handling real claims either relies on access to a document set which contains the “gold” evidence (Ferreira and Vlachos, 2016; Alhindi et al., 2018; Hanselowski et al., 2019; Atanasova et al., 2020) or conducts unconstrained retrieval (Augenstein et al., 2019), which may retrieve articles written by fact-checkers about the claim (example in Figure 1).

![](images/2218454a5dc07be252337be850d36b00065d9f0c5d0e4ded23975fb65c4601cb.jpg)  
Figure 1: Our fact-checking setting addresses realistic claims using evidence retrieved prior to when the claim was made.

We present the first study of fact-checking political claims under a realistic retrieval setting. Our retrieval over the web is restricted to documents authored before the time of the claim and not sourced from fact-checking websites, as shown by the left side of Figure 1. We propose a pipeline (illustrated in Figure 2) that builds upon prior work in fact checking as well as large language models (Brown et al., 2020) to handle the complexity of this setting. Our system first decomposes a claim into a series of subquestions (Chen et al., 2022a; Ousidhoum et al., 2022), targeting both explicit and implicit aspects of the claim. Each subquestion is fed into a commercial search engine to retrieve relevant documents, with the restrictions described above. Then, we conduct a second stage of fine-grained retrieval to isolate the most relevant portions of the documents. Finally, we use state-of-the-art language models (Brown et al., 2020; Ouyang et al., 2022) to generate claim-focused summaries from the retrieved content. These summaries can serve both as explanations for users as well as inputs to a classifier to determine the veracity based on these summaries.

Evaluating individual components of our pipeline is challenging due to the absence of gold annotations at each stage. We use automatic evaluation on the veracity classification performance, comparing to labels given by professional factcheckers. We supplement this with a human study evaluating the claim-focused summaries for comprehensiveness and faithfulness. This evaluation counterbalances the subjectivity of the veracity judgments (Lim, 2018) while shedding light on intermediate stages of the process.

![](images/d39dca29d599787c5322fd7b77c05cb709fbb095f257bb842c823ce231ae42ea.jpg)  
Figure 2: Overview of our pipeline: a claim is decomposed into yes/no subquestions (Sec. 3.1), then we use the questions in two stages of retrieval (Sec. 3.2 and Sec. 3.3) to select the most relevant paragraphs. Finally, we generate a claim-focused summary (Sec. 3.4) and train a veracity classifier to get the veracity label (Sec. 3.5). This filters contents irrelevant to the claim (see Appendix B for details and an example in Figure 5).

We apply our pipeline to CLAIMDECOMP (Chen et al., 2022a), a dataset containing 1,200 real-world complex political claims with veracity labels. Performance on veracity classification shows that: (1) our retrieval setting is indeed much harder than “unrestricted” retrieval settings; (2) using web evidence leads to performance gains compared to automatic fact-checking without evidence; (3) the decomposition is crucial for obtaining high-quality raw documents from the web compared to using the original claim alone. Our human study further indicates that: (4) claim-focused summaries are mostly faithful and helpful for both machines and humans to fact-check a claim; (5) the retrieved evidence is often relevant to some aspects of the claim, but can rarely cover all aspects, suggesting that finding sufficient raw evidence in the wild is the core challenge in building automatic fact-checking systems.

## 2 Background and Motivation

Early NLP research on fact-checking political claims (Vlachos and Riedel, 2014; Wang, 2017; Rashkin et al., 2017; Volkova et al., 2017; Pérez-Rosas et al., 2018; Dungs et al., 2018) typically considered using the claim alone as an input to an automated system. By not seeking evidence, systems judge the veracity of a claim mostly based on surface-level linguistic patterns rather than based on factual errors. Research that incorporates evidence either assumes access to justifications provided by fact-checkers (Vlachos and Riedel, 2014; Alhindi et al., 2018; Hanselowski et al., 2019; Atanasova et al., 2020) or evidence from unconstrained retrieval (Popat et al., 2017, 2018; Augenstein et al., 2019), which frequently yields evidence sets containing pages from fact-checking websites (Glockner et al., 2022). This does not reflect the difficulties in real-world evidence retrieval. Fan et al. (2020) explore generating questions to retrieve evidence from the web, but only evaluate their system with humans in the loop, who can aggressively filter irrelevant retrieval results. Contemporaneous to this work, Schlichtkrull et al. (2023) construct a dataset, AVeriTeC, using real-world claims and evidence retrieved from the web. Our method uses binary subquestions designed to target all needed aspects of factuality for a claim, whereas their questions are wh-questions optimized around retrieval, similar to QABriefs (Fan et al., 2020).

To our knowledge, we present the first automatic fact-checking system with a realistic retrieval pipeline using evidence available at the time a claim was made. This presents a very challenging setting where many claims are not checkable. We therefore emphasize the evidence our system returns as a way of assisting human fact-checkers; we believe this realistic task setting and corresponding evaluation should be reused in future work.

Our work shifts the focus away from the evaluation on classification accuracy alone. Accuracy on truth labels assigned by fact-checkers is a proxy metric we use to evaluate our systems. However, fact-checking experts argue that the task is too subjective and complex to be automated in the near term (Graves, 2018; Nakov et al., 2021). Part of this arises from the fact that information needed to check claims is not always available on the web (Singh et al., 2021). Our approach of returning information on a best-effort basis and providing evidence to enable humans to assist in the judgment can help overcome issues with returning judgments from error-prone AI systems (Bansal et al., 2021; Brand et al., 2022).

![](images/89ca559c8d18b127e08985eac6c2c356d3ccd0135bacdc9ec4cbfc1f328e7e08.jpg)  
Figure 3: An example of our claim decomposition process: each claim is decomposed into ten subquestions.  
Figure 4: Two documents returned by searching Q2 (generated in step 1). The right page post-dates the claim by one month and directly cites a PolitiFact article, making it problematic to use as raw evidence.

## 3 Methodology

Our pipeline, shown in Figure 3, consists of five parts: claim decomposition, raw document retrieval, fine-grained retrieval, claim-focused summarization, and veracity classification. We describe each part below.

## 3.1 Claim Decomposition

Given a real-world complex claim, we first decompose it into a set of yes/no questions for which the answers are useful to fact-check the claim. Chen et al. (2022a); Ousidhoum et al. (2022) show that such decompositions are both helpful to retrieve relevant evidence and make veracity judgments.

For decomposition, we prompt a large-scale language model, text-davinci-003, with incontext examples.<sup>2</sup> We carefully choose four inputdecomposition pairs from the human annotations of Chen et al. (2022a) to form a few-shot prompt. We generate a set of questions through multiple rounds of sampling until we gather 10 different questions. An example decomposition is shown in Figure 3. For the full prompt, see Appendix A.2.

## 3.2 First-stage Retrieval

For each question generated in the previous step, we feed it to a commercial search engine API to collect the relevant documents.

Temporal and Site Constraints We assume that a system should not be able to access pages published after the claim was made. This condition matches real-time fact-checking scenario during a political speech. We place a temporal constraint on the system to reflect this. Next, to investigate how the presence of fact-checking websites affects the veracity judgment of a claim, we also place a site constraint to filter out the documents from fact-checking websites. Our list of fact-checking websites can be found in Appendix A.1. An example of the retrieved documents is shown in Figure 4.

We use the Bing Search API,<sup>3</sup> and retrieve 10 documents per subquestion after filtering by the constraints. We extract the actual content from the page URLs using two tools: html2text<sup>4</sup> and readability-lxml.<sup>5</sup> Approximately one-third of the URLs are protected<sup>6</sup> and cannot be scraped.

Table 1 contains the raw counts from web retrieval with and without the timestamp of a claim. These results underscore the importance of temporal filtering: we find little overlap between the two document sets by comparing the Jaccard distance between two sets of the retrieved URLs.

One challenge for the reproducibility of our work

Does the U.S. have the highest number of gun deaths out of all the countries on the planet? No (annotator judgment based on summaries) Does the U.S. have a high number of gun deaths? Yes (annotator judgment based on summaries)

Claim: Melissa Agard stated on September 2, 2021 in News release: "No other country on the planet witnesses the number of gun deaths tha we do here in the United States, and it’s not even close."ti

Decomposed subques ons:

(1) Is the United States the country with the highest rate of gun deaths?

(2) Does the claim account for popula on size (i.e., per capita rates), or is it based on total numbers?

(3) Does the statement consider gun deaths rela ve to the total number of guns in the country?

(4) Is the number of gun deaths in the United States substan ally higher when compared to countries of similar economic and poli cal stability?

(5) Do gun deaths account for a large por on of deaths in the U.S.?

(10) Are there any mi ga ng factors that a ect the gun death rate in the United States?

Retrieved documents and summaries:
<table><tr><td rowspan=1 colspan=1>Doc title</td><td rowspan=1 colspan=1>URL</td><td rowspan=1 colspan=1>Summary</td></tr><tr><td rowspan=1 colspan=1>Firearm-related deaths rate U.S.by gender 1970-2016 | Statista(Nov 7, 2019)</td><td rowspan=1 colspan=1>https://www.statista.com/statistics/186951/deaths-by-firearm-related-injuries-in-the-us-by-gender-since-1970/</td><td rowspan=1 colspan=1>The death rate from firearm-related iniuries in the United States is 19.4 per 100,000population among males, and homicides from firearms account for 72.6 percent of allhomicides in the U.S. The ownership of legal firearms is widespread, with around 43percent of households having at least one firearm. (Faithful)</td></tr><tr><td rowspan=1 colspan=1>Do Gun Laws Affect the Rate ofShooting Deaths? (Oct 12, 2018)</td><td rowspan=1 colspan=1>https://www.thetrace.org/2018/10/do-gun-laws-affect-the-rate-of-shooting-deaths/</td><td rowspan=1 colspan=1>The Centers for Disease Control and Prevention tracks gun deaths in all 50 states and thenational rate of gun deaths in 2016 was 11.8. No additional evidence is provided tocompare the number of gun deaths in the US to other countries. (Faithful)</td></tr><tr><td rowspan=2 colspan=1>Gun Violence Deaths: How TheU.S. Compares With The Rest OfThe World (Mar 24, 2021)</td><td rowspan=1 colspan=1>https://www.kuow.org/stories/gun-violence-deaths-how-the-u-s-compares-with-the-rest-of-the-world</td><td rowspan=2 colspan=1>The United States has the 32nd highest rate of deaths from gun violence with 3.96deaths per 100,000 people compared to countries such as Canada with 0.47 deaths per100,000 people and the United Kingdom with 0.04 deaths per 100,000 people. (Faithful)</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=1>A Doctor&#x27;s Insights Into GunViolence And Gun Laws AroundThe World (Aug 6, 2019)</td><td rowspan=1 colspan=1>https://www.kuer.org/2019-08-06/a-</td><td rowspan=3 colspan=1>The US rate of deaths from gun violence is 4.43 deaths per 100,000 people and is fourtimes higher than the rates in war-torn Syria and Yemen. This places is among the top30 countries in the world with the highest rates of deaths from gun violence. (Faithful)</td></tr><tr><td rowspan=1 colspan=1>doctors-insights-into-gun-violence-and-</td></tr><tr><td rowspan=1 colspan=1>gun-laws-around-the-world</td></tr></table>

Annotated subques ons:  
Does the U.S. have a high number of gun deaths when looking at deaths as a share of the popula on? Yes (annotator judgment based on summaries)  
Model Predic on: Half-True

Label: Mostly-False

Figure 5: System outputs for an example picked from the dev set of CLAIMDECOMP: the claim is first decomposed into a set of yes/no questions and then the top four retrieved documents (through first and second stage retrieval) are summarized. Finally, a trained DeBERTa model makes a prediction regarding the four summarized documents.
<table><tr><td></td><td># retrieved</td><td># scraped</td><td># words</td></tr><tr><td>w/ timestamp</td><td>66.7</td><td>45.0</td><td>1,561</td></tr><tr><td>w/o timestamp</td><td>70.4</td><td>47.8</td><td>1,660</td></tr><tr><td>Jaccard score</td><td>0.12</td><td>0.12</td><td></td></tr></table>

Table 1: The statistics for the retrieved documents obtained through the first-stage retrieval after filtering the documents from fact-checking websites. Jaccard between these two sets show that incorporating the timestamp in retrieval makes a substantial difference.

is that commercial search engines may return differtient results over time. In Section 5.3, we experiment with the same query set at different times. We find that the search results change over time: only 30% of search output URLs overlap when queried two months apart. However, the veracity judgment classification result is not impacted much.

ticlaim. However, as can be seen from Figure 2, firststage retrieval can easily result in tens of thousands of words of retrieved documents, which are costly to process with an LLM. Furthermore, even with state-of-the-art language models, it is hard to do complex reasoning over such long context (Liu et al., 2024; Levy et al., 2024). Thus, we conduct a second-stage retrieval to pick the most relevant text spans to the claim from the retrieved documents. Specifically, we segment the documents into text spans containing $k _ { 1 }$ words with a stride of $\scriptstyle { \frac { 1 } { 2 } } k _ { 1 }$ words. Following Chen et al. (2022a), we employ BM-25 to retrieve the top- $K _ { 1 }$ highest-scored text spans, expanding these spans with ${ \bf a } \pm k _ { 2 }$ -word context. If two text spans overlap, they are merged to form a larger span. This process yields a set of “documents” ranked by the highest-scored text spans, of which we pick the top-K<sub>2</sub>.

## 3.3 Second-stage Retrieval

Most of the documents collected from the previous step contain at most a few snippets relevant to the

## 3.4 Claim-Focused Summarization

Since the documents retrieved in the previous step can contain up to several thousand words, it becomes cumbersome for both humans and models to make a judgment based on them (Stammbach and Ash, 2020). Consequently, we prompt a large language model, specifically text-davinci-003, to summarize each retrieved document separately with respect to the claim.<sup>7</sup> Such single-document summarization has been shown to be robust on news articles (Goyal et al., 2022; Zhang et al., 2023).

We investigate two types of prompts. For a zeroshot prompt, we instruct the model not to make any judgments about the stance of the given document. For a few-shot prompt, we select four documents and carefully write desired summaries. For documents that are not relevant to the claim, we write “the document is not relevant to checking the claim” as its desired output. We conduct human evaluation of the summary quality of different prompts in Section 6.1, where we find that few-shot prompting works better. See Appendix A.3 for full prompts.

## 3.5 Veracity Classification

The final stage of our pipeline involves making a judgment based on the summaries generated in the previous stage. Unlike previous stages which use off-the-shelf tools, here we train a DeBERTalarge (He et al., 2020) model<sup>8</sup> to perform a six-way veracity classification (true, mostly true, half true, barely true, false, and pants-on-fire).

Training We run our pipeline over the training, development, and test data of CLAIMDECOMP and train on pairs of the form (claim+summary, label). Since the dataset is small, we train the classifier five times with different random seeds and report the test set performance using the model that achieves the best performance on the development set.

## 3.6 Final Pipeline

Our complete pipeline’s results when executed on an example are shown in Figure 5. We note that the question decomposition phase yields an overcomplete set of questions, including redundant ones. However, the final retrieved and summarized documents are able to shed light on the claim from several complementary perspectives. While the final veracity judgment does not exactly match the judgment from PolitiFact, reading the documents still gives an informed picture of the situation.

## 4 Experimental Setup

Our main automatic evaluation is on claim veracity prediction (Wang, 2017), evaluating our entire pipeline end-to-end. We will describe the human evaluation setup in Section 6.

Data We use the data from CLAIMDE-COMP (Chen et al., 2022a) which contains 1,200 complex claims from PolitiFact (train: 800, dev: 200, test: 200). Each claim is labeled with one of the six veracity labels, a justification paragraph written by expert fact-checkers, and subquestions annotated by prior work.

Hyperparameters For the second-stage retrieval, we set top-K<sub>1</sub> = 10 (highest-scored text spans), top-K<sub>2</sub> = 4 (highest-scored documents), k<sub>1</sub> = 30 (chunk size), and k<sub>2</sub> = 150 (expansion parameter). See appendix A.4 for all hyperparameters.

Evaluation Metric We report accuracy (Acc), mean absolute error (MAE, on our 6-point scale), and Macro-F1. We also introduce soft accuracy (soft Acc), which is calculated by counting off-byone errors on the six-point veracity scale (e.g., half true instead of mostly true) as correct, as veracity judgments are subjective.

Comparison Systems For our Claim-only system, we concatenate the metadata, including the speaker and the venue of the claim, with the claim itself, and feed the resulting text into the classifier (Wang, 2017). This approach serves as a lower bound for the veracity classification.

We extend the Claim-only baseline to Claim+Justification by appending the humanwritten justification paragraph, excluding the sentence containing the label, to the claim. This is an oracle setting to establish an upper bound for veracity classification.

## 5 Automatic Evaluation: Claim Veracity

## 5.1 Constrained vs. Unconstrained Search

We first situate our work with respect to baselines and past systems by varying the retrieval condition. We experiment with a temporal constraint, where pages must originate before the date of the claim, and a site constraint, where sites must be non-fact-checking (non-FC) sites. Even in the unconstrained setting, we exclude pages from Politi-Fact (our dataset’s source) to prevent label leakage.

<table><tr><td colspan="2">Retrieval Constraint</td><td colspan="4">Dev (N=200)</td><td colspan="4">Test (N=200)</td></tr><tr><td>Temporal</td><td>Site</td><td>Acc</td><td>Soft Acc Macro-F1</td><td></td><td>MAE</td><td>Acc</td><td>Soft Acc Macro-F1</td><td></td><td>MAE</td></tr><tr><td>-</td><td></td><td>50.5</td><td>88.5</td><td>47.5</td><td>0.62</td><td> $4 9 . 0 ^ { + }$ </td><td> $8 6 . 0 ^ { + }$ </td><td> $4 8 . 5 ^ { + }$ </td><td> $0 . 6 8 ^ { + }$ </td></tr><tr><td></td><td>Non-FC</td><td>37.5</td><td>76.5</td><td>38.6</td><td>0.94</td><td> $3 3 . 5 ^ { + }$ </td><td> $7 5 . 0 ^ { + }$ </td><td> $3 3 . 9 ^ { + }$ </td><td> $0 . 9 5 ^ { + }$ </td></tr><tr><td>Before</td><td></td><td>42.5</td><td>75.0</td><td>41.7</td><td>0.87</td><td> $3 3 . 5 ^ { + }$ </td><td>72.0</td><td> $3 8 . 0 ^ { + }$ </td><td> $0 . 9 8 ^ { + }$ </td></tr><tr><td>Before</td><td> $\mathrm { N o n - F C }$ </td><td>40.5</td><td>76.5</td><td>41.4</td><td>0.87</td><td> $3 3 . 0 ^ { + }$ </td><td> $7 4 . 5 ^ { + }$ </td><td> $3 4 . 5 ^ { + }$ </td><td> $0 . 9 9 ^ { + }$ </td></tr><tr><td colspan="2">Claim only</td><td>37.0</td><td>71.0</td><td>34.6</td><td>0.98</td><td>25.5</td><td>68.0</td><td>27.5</td><td>1.12</td></tr><tr><td colspan="2">Claim + Justification (oracle)</td><td>52.5</td><td>88.5</td><td>54.5</td><td>0.64</td><td>57.5</td><td>93.0</td><td>57.8</td><td>0.50</td></tr></table>

Table 2: Veracity classification performance with different retrieval constraints. The top block is our full system ( B setting in Table 3) with constraints over what is retrieved. Red indicates using oracle information. $" + "$ denotes that the results are statistically significant improvements $( p < 0 . 0 5 )$ compared to the results of Claim only on the test set.

The unconstrained setting corresponds to that used in MultiFC (Augenstein et al., 2019). MultiFC includes numerous documents that are filtered out by our constrained settings. For each claim, they extract the top 10 pages from the Google search API. We find that 12,721 out of 15,379 claims (82.7%) contain at least one page from our excluded website list and 24.4% of the retrieved web pages are from fact-checking websites.

Table 2 reports the performance of our system with various retrieval constraints. Comparing the performance of claim-only and other models that use retrieval, we see a statistically significant<sup>9</sup> improvement over all four of our metrics in nearly all settings, showing that retrieving and summarizing evidence is helpful to predict the veracity label, even with constraints.

Second, we see adding either temporal or site constraints dramatically reduces the performance. This implies that retrieval over the web works largely because it retrieves fact-checks that were published after the claim was released, with synthesized evidence. We believe that future work on retrieval should use a constrained setting.

## 5.2 Stage Ablations

We evaluate design choices in each stage of the pipeline to understand how each individual component contributes to the final performance. The results are shown in Table 3.

First-stage Retrieval: subquestions vs. original claim Using the original claim instead of the generated subquestions as an input to web search ( B vs. 1 ) results in a notable decrease in performance.

The subquestion set encompasses multiple aspects of the claim, enabling the search engine to locate relevant information more easily across separate search queries. Comparing B and $\textcircled{2} .$ , we see using the gold subquestions actually yields worse performance than our predicted subquestions. This could be because we predict 10 subquestions, potentially garnering more relevant data than the 3 (on average) gold subquestions (Chen et al., 2022a).

Second-stage Retrieval Rather than retrieving with subquestions (subQs), we instead perform our search with the raw Claim ( 3 ), Gold subQs from CLAIMDECOMP ( 4 ), or Justification ( 5 ), which uses oracle information. Different queries yield only slight differences in performance and none <sup>of</sup> <sup>them</sup> <sup>is</sup> <sup>statistically</sup> <sup>significant,</sup> <sup>even</sup> <sup>when</sup> ⃝<sup>5</sup> uses the human-written justification. We believe this is because we expand the retrieved text span by a context window ( 150 words). As a result, this retrieval step does not need to be very precise to capture the relevant information.

Claim-focused Summarization We compare zero-shot ( B ) and few-shot ( 6 ) prompts for generating the summary; no summary ( 7 ) directly feeds the text spans from second-stage retrieval to the veracity classifier. System 7 shows the worst performance across all metrics, suggesting that summarization matters. This may result from two primary factors: (1) The document length exceeds the context window capacity of DeBERTa, causing crucial information to be truncated. (2) our veracity classifier cannot easily discern the most relevant information given a large amount of context. Differences in the prompt ( B and 6 ) do not impact veracity classification results much but have differences under human inspection, which we discuss in the next section.

<table><tr><td rowspan="2">FSR</td><td colspan="3">Evidence Generation</td><td colspan="3">Performance</td></tr><tr><td>SSR</td><td>Summary</td><td>Acc</td><td></td><td>Soft Acc Macro-F1 MAE</td><td></td></tr><tr><td colspan="3">Claim only</td><td> $2 5 . 5 ^ { + }$ </td><td> $6 8 . 0 ^ { + }$ </td><td> $2 7 . 5 ^ { + }$ </td><td> $1 . 1 2 ^ { + }$ </td></tr><tr><td colspan="3">Claim + Justification</td><td> $5 7 . 5 ^ { + }$ </td><td> $9 3 . 0 ^ { + }$ </td><td> $5 7 . 8 ^ { + }$ </td><td> $0 . 5 0 ^ { + }$ </td></tr><tr><td colspan="6">Our Default System</td><td></td></tr><tr><td>B</td><td>subQs subQs</td><td>zero-shot-003</td><td>33.0</td><td>74.5</td><td>34.5</td><td>0.99</td></tr><tr><td colspan="6">Ablation on first-stage retrieval</td><td></td></tr><tr><td>Claim</td><td></td><td></td><td> $2 4 . 5 ^ { + }$ </td><td>71.5</td><td>18.0+</td><td> $1 . 1 5 ^ { + }$ </td></tr><tr><td>Gold  $\mathrm { \ s u b Q s }$ </td><td></td><td></td><td>27.5</td><td>72.0</td><td> $2 8 . 1 ^ { + }$ </td><td> $1 . 0 5 ^ { + }$ </td></tr><tr><td colspan="6">Ablation on second-stage retrieval</td><td></td></tr><tr><td>③</td><td>Claim</td><td></td><td>31.5</td><td>75.0</td><td>35.6</td><td>0.97</td></tr><tr><td>④</td><td>Gold subQs</td><td></td><td>31.5</td><td>73.0</td><td>35.4</td><td>1.03</td></tr><tr><td>⑤</td><td>Justification</td><td></td><td>33.0</td><td>71.5</td><td>37.2</td><td>1.01</td></tr><tr><td colspan="6">Ablation on summarization</td><td></td></tr><tr><td>⑥</td><td></td><td>few-shot-003</td><td>35.0</td><td>76.5</td><td>36.2</td><td>0.94</td></tr><tr><td>⑦</td><td></td><td>no summary (raw doc)</td><td>29.0</td><td> $6 6 . 0 ^ { + }$ </td><td> $2 6 . 3 ^ { + }$ </td><td> $1 . 1 8 ^ { + }$ </td></tr></table>

Table 3: End-to-end fact-checking performance on the test set of CLAIMDECOMP. We ablate various stages of the model (FSR: first-stage retrieval; SSR: second-stage retrieval). Red indicates using oracle information. $^ { 6 6 } + { \bar { } } ^ { , 3 }$ denotes the result changes are statistically significant $( p < 0 . 0 5 )$ with respect to our default system.

## 5.3 Stability of First-stage Retrieval

As commercial search engines evolve over time, we conduct experiments to explore the reproducibility of our first-stage retrieval step. We use the default system setting in Table 3 and conducted three rounds of retrieval at $T = 0 , T = 1$ week, and $T = 2$ months. We evaluate the Jaccard similarity of the sets of URLs retrieved from our queries to understand how much changes in the Bing API and the broader web change our results. We also evaluate the veracity of our system. Note that this Jaccard similarity is between the members of the URL sets (i.e., the URLs themselves), not capturing any lexical or domain similarity of the URLs.

Results are shown in Table 4. A noticeable trend is a decline in the Jaccard score between varying retrieval rounds over time. However, this decrease does not significantly impact the models’ efficacy in the veracity assessment.

We caution that as the time gap increases, the set of documents retrieved from the Bing Search API could become considerably different, posing a challenge to consistently benchmark retrieval performance using commercial search engines. Therefore, we advocate for future research to focus on developing a comprehensive yet challenging document set that could be publicly released as a benchmark to spur research.

<table><tr><td></td><td>Overlap Acc</td><td></td><td> $\mathrm { S o f t } { \cdot } \mathrm { A c c }$ </td><td>Ma-F1</td><td>MAE</td></tr><tr><td>Ours</td><td></td><td>33.0</td><td>74.5</td><td>34.5</td><td>0.99</td></tr><tr><td>1 week</td><td>0.48</td><td>33.5</td><td>74.0</td><td>36.8</td><td>0.98</td></tr><tr><td>2 months</td><td>0.30</td><td>29.5</td><td>73.5</td><td>32.3</td><td>1.03</td></tr></table>

Table 4: Model performance with respect to different rounds of retrieval at intervals of one week and two months. The overlap between “Ours” and subsequent document sets, measured with Jaccard score, decreases as the time gap increases. However, none of the changes in our downstream metrics is statistically significant.

## 6 Human Evaluation of Summaries

Summarizing documents from web search with large language models improves the performance of our fact-checking pipeline. However, these models can generate untruthful content (Bommasani et al., 2021; Chowdhery et al., 2022; Ouyang et al., 2022). Furthermore, as pointed out by Lim (2018), the accuracy of veracity classification alone does not entirely reflect the system’s overall effectiveness, as certain labels such as “false” and “barelytrue” may be ambiguous. We believe the true measure of our system’s utility lies in the full package of summarized evidence it returns rather than just the accuracy of the veracity label. Therefore, we carry out two human studies, on comprehensiveness and faithfulness, to better understand intermediate outputs of the system.

Setting We randomly pick 50 claims which contain 200 document-summary pairs from the development set of CLAIMDECOMP and run two human evaluation studies on this set. For each task, we recruited annotators from Amazon Mechanical Turk with a qualification test. In total, we recruited 17 worker for the faithfulness study and 15 workers for the comprehensiveness study. The details about crowdsourcing can be found in Appendix C.

<table><tr><td>Summ-type</td><td>F</td><td>Minor</td><td>Major</td><td>NF</td><td>Avg score</td></tr><tr><td>zero-shot-001</td><td>65.8%</td><td>9.2%</td><td>20.0%</td><td>5.0%</td><td>3.45</td></tr><tr><td>zero-shot-003</td><td>66.0%</td><td>18.0%</td><td>16.0%</td><td>0.0%</td><td>3.50</td></tr><tr><td>few-shot-003</td><td>82.5%</td><td>6.5%</td><td>8.5%</td><td>2.5%</td><td>3.69</td></tr></table>

Table 5: Faithfulness Human Evaluation (N = 200). “F” denotes that the summary is factual and “NF” denotes that the summary is completely wrong. Few-shot prompting helps the model make fewer factual errors.

Comparison Systems We compare the summaries generated from two prompts, zero-shot-003 and few-shot-003, on GPT-3.5 (davinci-003). For the faithfulness study, we also compare the summaries generated through with zero-shot prompt on an earlier GPT model (davinci-001) (zero-shot-001) to see how the faithfulness varies for different models.

## 6.1 Faithfulness Evaluation

Goal We assess the frequency and degree to which the language model generates untruthful content during query-focused summarization. For each document and summary pair, annotators choose one of four labels below (see appendix C.1 for examples):

• Faithful: the summary accurately represents the meaning and details of the original document.

• Minor Factual Error: some details are not aligned with the original document, but the overall message remains intact.

• Major Factual Error: there are factual errors that result in the summary misrepresenting the original document.

• Completely Wrong: the language model hallucinates content that completely alters the meaning of the original document.

In addition to selecting a label, we ask annotators to provide a natural language justification for their choices. The annotations agree with a Fleiss Kappa score of 0.30. While this number is somewhat low, when we evaluated their justifications and we find many of the disagreements are because of subjectivity on the extent of factual error. We compute a consensus annotation via majority vote. We assign numerical scores to each label, where “Faithful”, “Minor”, “Major”, and “Completely Wrong” correspond to 4, 3, 2, and 1 respectively and report average values. If all annotators disagree, we compute the average score and return the label that is nearest to the average score as a consensus.

Results The results are shown in Table 5. We see that few-shot prompting substantially decreases the chance of hallucinations in the summaries. When combining “Factual” and “Minor”, we see 89% of the summaries are good enough to be used as evidence for the classifier. Additionally, by checking the unfaithful summaries, we find that they do not consist of useful hallucination like making a veracity judgment based on the parametric knowledge. Comparing the performance of zero-shot-001 and zero-shot-003, we find that the weaker model makes more major factual errors. Together, they indicate that with stronger models and better prompts, we may expect these summarization models to improve further.

## 6.2 Comprehensiveness Evaluation

Goal We aim to measure the extent to which the claim-focused summaries are able to address the claim. This is subjective and difficult task to evaluate. Here, we leverage the human-annotated yes/no subquestions from CLAIMDECOMP as a proxy for evaluating the comprehensiveness of our summaries: if provided summary can help humans to answer more of these yes/no questions, we deem the summary to be more comprehensive.

In this task, annotators are given a summary / subquestion pair and label subquestion as “answerable”, “partially answerable”,<sup>10</sup> or “unanswerable”, and additionally provide yes/no answer if the question is labeled as “answerable”. Annotators were also asked to provide natural language justification for their answers. We collect this annotation on 161 questions associated with 50 claims. The annotations agree with a Fleiss Kappa score of 0.32.

Results The results are presented in Table 6. We see that zero-shot summaries yield more answerable questions than few-shot summaries. However, faithfulness evaluation hints that this is caused by hallucinations in zero-shot summaries; the system imputes information that seems to help, but which is not supported by the document.

<table><tr><td>Summ-type</td><td>Ans</td><td>Partially Ans</td><td>UnAns</td></tr><tr><td>zero-shot-003</td><td>47.8%</td><td>22.4%</td><td>29.8%</td></tr><tr><td>few-shot-003</td><td>42.9%</td><td>21.1%</td><td>36.0%</td></tr></table>

Table 6: Human evaluation results on 161 subquestions from the same 50 claims we picked for the human study on faithfulness. “Ans”, “Partially Ans”, and “UnAns” denote the number of questions that are answerable, partially answerable, and unanswerable.
<table><tr><td></td><td>Faithful</td><td>Minor</td><td>Unfaithful</td><td>Total</td></tr><tr><td>Ans</td><td>4</td><td>2</td><td>0</td><td>6</td></tr><tr><td>Partially Ans</td><td>6</td><td>1</td><td>1</td><td>8</td></tr><tr><td>Partially UnAns</td><td>13</td><td>5</td><td>11</td><td>30</td></tr><tr><td>UnAns</td><td>5</td><td>1</td><td>0</td><td>6</td></tr><tr><td>Total</td><td>28</td><td>10</td><td>12</td><td>50</td></tr></table>

Table 7: Claim-level statistics of few-shot-003 taking both faithfulness and comprehensiveness into consideration. “Unfaithful" label aggregates “Major Error" and “Completely Wrong" labels. The claim-level labels are derived from the sub-parts as defined in section 6.3.

Nevertheless, the few-shot summaries allow us to partially address over 60% of the gold annotated subquestions derived from the PolitiFact justification. We find this result encouraging: even though the system does not have access to these (often subtle) factors, it can retrieve information to enable a human annotator to make a judgment about them.

## 6.3 Combined Evaluation

While in previous section we evaluated faithfulness and comprehensiveness separately, here we conduct a claim-level evaluation: how many claims can be comprehensively addressed with a set of faithful summaries? We label a claim as answerable if all of its subquestions are answerable. If all subquestions are unanswerable the claim is unanswerable. Otherwise, we label claim as partially unanswerable. For claim-level faithfulness, we apply the same principles: a claim is faithful is all summaries are faithful, otherwise it is either unfaithful or contains minor factual errors. Table 7 shows the results by combining the two factors. We see that addressing every aspect of complex claims is still challenging: 36 out of 50 claims contain at least one unanswerable question. For claims that can be fully addressed (all questions are either answerable or partially answerable), only 1 out of 14 contains a major factual error in the summary.

## 7 Related Work

Retrieval augmented models Prior work has shown that a variety of NLP tasks could benefit from incorporating a retrieval component. Such tasks mainly include question answering (Chen et al., 2017; Kwiatkowski et al., 2019; Karpukhin et al., 2020; Khattab et al., 2021; Nakano et al., 2021), text generation (Lewis et al., 2020; Shi et al., 2023; Ram et al., 2023), language modeling (Guu et al., 2020; Khandelwal et al., 2020; Zhong et al., 2022), and dialog (Moghe et al., 2018; Fan et al., 2021; Thoppilan et al., 2022).

Most of these work assume having access to a fixed corpus, however, for the task of real-world fact-checking, no such corpus exists. In this work, we follow WebGPT (Nakano et al., 2021) and use Bing Search API to retrieve evidence from the wild web. Recent LLM agents such as Bing Chat and Google Bard follow this paradigm, so we believe these directions will be relevant for future work.

Question decomposition has been shown to be effective in evidence retrieval and question understanding for complex question answering (Talmor and Berant, 2018; Min et al., 2019; Qi et al., 2019; Perez et al., 2020; Wolfson et al., 2020; Geva et al., 2021). Question generation has also been shown to play a useful role in retrieval pipelines in opendomain QA (Sachan et al., 2022). In more recent research, it was demonstrated by Chen et al. (2022a) that such decompositions can also aid in retrieving evidence to assess complex claims and make veracity judgment. This observation is consistent with concurrent studies on fact-checking text generation outputs (Gao et al., 2022; Chen et al., 2022b; Liu et al., 2022) and Wikipedia (Kamoi et al., 2023).

## 8 Conclusion

We introduce a pipeline for realistic, automated fact-checking of complex political claims by retrieving raw evidence from web documents, improving final fact checking accuracy by integrating retrieved evidence. Our pipeline show promising results on the CLAIMDECOMP dataset. Yet, web search often cannot surface all the pieces of information necessary to verify a given claim. This work emphasizes the challenges of evidence retrieval in real-world scenarios and underscores the need for a human-in-the-loop fact-checking system.

## Limitations and Future Directions

Performance is bottlenecked by the first-stage retrieval. The results in the last section show that 36.0% of questions are unanswerable using our most faithful claim-focused summaries. By investigating the unanswerable cases, we see that the following cases lead to retrieval failure: (1) no relevant information is available on the web except the fact-checking websites. These claims can be onerous to check, such as requiring talking to or emailing specific people to check facts. Those cases are beyond the scope of this work and we think a system doing triage for the claims, would be promising for future work. (2) No relevant subquestions are generated or the subquestions are not well decontextualized (Choi et al., 2021). In such cases, a stronger question generation model or decontextualization model can help further.

The need of human-in-the-loop fact-checking. To address the failures in the first-stage retrieval and the potential errors in the summarization stage, we envision a human-in-the-loop fact-checking system. This system begins with the automated pipeline presented in this paper, which provides fact-checkers with summarized documents and judgments. If the fact-checkers deem these documents unsatisfactory, the system reveals the subquestions used for evidence retrieval, allowing factcheckers to rerun the search. The system then retrieves additional documents and generates updated summaries. This iterative process continues until the fact-checkers are satisfied with the retrieved evidence. Moreover, the system could further learn from the fact-check feedback to improve itself: for example, the system could learn what questions are important to retrieve good evidence and what questions are not according to the fact-checker. In general, we believe such systems will be necessary, but developing them is outside of the scope of this work.

Scope of facts checked. Our work only addresses English-language political claims. Misinformation in other languages is a crucial problem that we believe future work should address. Moreover, even within English, there is a strong need for factchecking systems that can address other kinds of claims that have a different distribution; for example, claims from social media, which are often embedded in images or memes. Nevertheless, we believe the decomposition and retrieval approach here can play a role in such systems as well.

## Acknowledgments

This work was partially supported by NSF CA-REER Award IIS-2145280, by Good Systems,<sup>11</sup> a UT Austin Grand Challenge to develop responsible AI technologies, and by grants from Salesforce Inc. and Open Philanthropy. We thank the UT Austin NLP community for feedback on the earlier drafts of the paper.

## References

Tariq Alhindi, Savvas Petridis, and Smaranda Muresan. 2018. Where is your evidence: Improving factchecking by justification modeling. In Proceedings ofthe First Workshop on Fact Extraction and VERification (FEVER), pages 85–90, Brussels, Belgium. Association for Computational Linguistics.

Pepa Atanasova, Jakob Grue Simonsen, Christina Lioma, and Isabelle Augenstein. 2020. Generating fact checking explanations. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 7352–7364, Online. Association for Computational Linguistics.

Isabelle Augenstein, Christina Lioma, Dongsheng Wang, Lucas Chaves Lima, Casper Hansen, Christian Hansen, and Jakob Grue Simonsen. 2019. MultiFC: A real-world multi-domain dataset for evidencebased fact checking of claims. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4685–4697, Hong Kong, China. Association for Computational Linguistics.

Gagan Bansal, Tongshuang Wu, Joyce Zhou, Raymond Fok, Besmira Nushi, Ece Kamar, Marco Tulio Ribeiro, and Daniel Weld. 2021. Does the Whole Exceed Its Parts? The Effect of AI Explanations on Complementary Team Performance. In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, CHI ’21, New York, NY, USA. Association for Computing Machinery.

Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. 2021. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258.

Erik Brand, Kevin Roitero, Michael Soprano, Afshin Rahimi, and Gianluca Demartini. 2022. A neural model to jointly predict and explain truthfulness of statements. ACM Journal ofData and Information Quality, 15(1):1–19.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading Wikipedia to answer opendomain questions. In Proceedings ofthe 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1870–1879, Vancouver, Canada. Association for Computational Linguistics.

Jifan Chen, Aniruddh Sriram, Eunsol Choi, and Greg Durrett. 2022a. Generating literal and implied subquestions to fact-check complex claims. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 3495–3516, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Sihao Chen, Senaka Buthpitiya, Alex Fabrikant, Dan Roth, and Tal Schuster. 2022b. PropSegmEnt: A Large-Scale Corpus for Proposition-Level Segmentation and Entailment Recognition. arXiv eprint arxiv:2212.10750.

Eunsol Choi, Jennimaria Palomaki, Matthew Lamm, Tom Kwiatkowski, Dipanjan Das, and Michael Collins. 2021. Decontextualization: Making sentences stand-alone. Transactions ofthe Association for Computational Linguistics, 9:447–461.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.

Sebastian Dungs, Ahmet Aker, Norbert Fuhr, and Kalina Bontcheva. 2018. Can rumour stance alone predict veracity? In Proceedings of the 27th International Conference on Computational Linguistics, pages 3360–3370, Santa Fe, New Mexico, USA. Association for Computational Linguistics.

Angela Fan, Claire Gardent, Chloé Braud, and Antoine Bordes. 2021. Augmenting transformers with KNNbased composite memory for dialog. Transactions of the Association for Computational Linguistics, 9:82– 99.

Angela Fan, Aleksandra Piktus, Fabio Petroni, Guillaume Wenzek, Marzieh Saeidi, Andreas Vlachos, Antoine Bordes, and Sebastian Riedel. 2020. Generating fact checking briefs. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7147–7161, Online. Association for Computational Linguistics.

William Ferreira and Andreas Vlachos. 2016. Emergent: a novel data-set for stance classification. In Proceedings ofthe 2016 conference ofthe North American

chapter ofthe associationfor computational linguistics: Human language technologies. ACL.

Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao, N. Lao, Hongrae Lee, Da-Cheng Juan, and Kelvin Guu. 2022. Rarr: Researching and revising what language models say, using language models.

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9:346– 361.

Max Glockner, Yufang Hou, and Iryna Gurevych. 2022. Missing counter-evidence renders NLP fact-checking unrealistic for misinformation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5916–5936, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Tanya Goyal, Junyi Jessy Li, and Greg Durrett. 2022. News Summarization and Evaluation in the Era of GPT-3. arXiv eprint arxiv:2209.12356.

Lucas Graves. 2018. Understanding the Promise and Limits of Automated Fact-Checking. Technical report, Reuters Institute, University of Oxford.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. In International conference on machine learning, pages 3929–3938. PMLR.

Andreas Hanselowski, Christian Stab, Claudia Schulz, Zile Li, and Iryna Gurevych. 2019. A richly annotated corpus for different tasks in automated factchecking. In Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL), pages 493–503, Hong Kong, China. Association for Computational Linguistics.

Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2020. DeBERTa: Decoding-enhanced BERT with Disentangled Attention. In International Conference on Learning Representations.

Ryo Kamoi, Tanya Goyal, Juan Diego Rodriguez, and Greg Durrett. 2023. WiCE: Real-World Entailment for Claims in Wikipedia. arXiv eprint arxiv:2303.01432.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, Online. Association for Computational Linguistics.

Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. 2020. Generalization through memorization: Nearest neighbor language

models. In International Conference on Learning Representations.

Omar Khattab, Christopher Potts, and Matei Zaharia. 2021. Relevance-guided supervision for OpenQA with ColBERT. Transactions ofthe Associationfor Computational Linguistics, 9:929–944.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466.

Mosh Levy, Alon Jacoby, and Yoav Goldberg. 2024. Same task, more tokens: the impact of input length on the reasoning performance of large language models. arXiv preprint arXiv:2402.14848.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474.

Chloe Lim. 2018. Checking how fact-checkers check. Research & Politics, 5(3):2053168018786848.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe Association for Computational Linguistics, 12:157–173.

Yixin Liu, Alexander R. Fabbri, Pengfei Liu, Yilun Zhao, Linyong Nan, Ruilin Han, Simeng Han, Shafiq Joty, Chien-Sheng Wu, Caiming Xiong, and Dragomir Radev. 2022. Revisiting the Gold Standard: Grounding Summarization Evaluation with Robust Human Evaluation. arXiv eprint arxiv:2212.07981.

Sewon Min, Victor Zhong, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2019. Multi-hop reading comprehension through question decomposition and rescoring. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 6097–6109, Florence, Italy. Association for Computational Linguistics.

Nikita Moghe, Siddhartha Arora, Suman Banerjee, and Mitesh M. Khapra. 2018. Towards exploiting background knowledge for building conversation systems. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2322–2332, Brussels, Belgium. Association for Computational Linguistics.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders,

et al. 2021. WebGPT: Browser-assisted questionanswering with human feedback. arXiv preprint arXiv:2112.09332.

Preslav Nakov, David Corney, Maram Hasanain, Firoj Alam, Tamer Elsayed, Alberto Barr’on-Cedeno, Paolo Papotti, Shaden Shaar, and Giovanni Da San Martino. 2021. Automated fact-checking for assisting human fact-checkers. In Proceedings ofthe Thirtieth International Joint Conference on Artificial Intelligence (IJCAI).

Nedjma Ousidhoum, Zhangdie Yuan, and Andreas Vlachos. 2022. Varifocal question generation for factchecking. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 2532–2544, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155.

Ethan Perez, Patrick Lewis, Wen-tau Yih, Kyunghyun Cho, and Douwe Kiela. 2020. Unsupervised question decomposition for question answering. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8864–8880, Online. Association for Computational Linguistics.

Verónica Pérez-Rosas, Bennett Kleinberg, Alexandra Lefevre, and Rada Mihalcea. 2018. Automatic detection of fake news. In Proceedings of the 27th International Conference on Computational Linguistics, pages 3391–3401, Santa Fe, New Mexico, USA. Association for Computational Linguistics.

Kashyap Popat, Subhabrata Mukherjee, Jannik Strötgen, and Gerhard Weikum. 2017. Where the truth lies: Explaining the credibility of emerging claims on the web and social media. In Proceedings of the 26th International Conference on World Wide Web Companion, pages 1003–1012. International World Wide Web Conferences Steering Committee.

Kashyap Popat, Subhabrata Mukherjee, Andrew Yates, and Gerhard Weikum. 2018. DeClarE: Debunking fake news and false claims using evidence-aware deep learning. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 22–32, Brussels, Belgium. Association for Computational Linguistics.

Peng Qi, Xiaowen Lin, Leo Mehr, Zijian Wang, and Christopher D. Manning. 2019. Answering complex open-domain questions through iterative query generation. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2590–2602, Hong Kong, China. Association for Computational Linguistics.

Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. 2023. In-context retrieval-augmented language models. arXiv preprint arXiv:2302.00083.

Hannah Rashkin, Eunsol Choi, Jin Yea Jang, Svitlana Volkova, and Yejin Choi. 2017. Truth of varying shades: Analyzing language in fake news and political fact-checking. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 2931–2937, Copenhagen, Denmark. Association for Computational Linguistics.

Devendra Sachan, Mike Lewis, Mandar Joshi, Armen Aghajanyan, Wen-tau Yih, Joelle Pineau, and Luke Zettlemoyer. 2022. Improving passage retrieval with zero-shot question generation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3781–3797, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Michael Schlichtkrull, Zhijiang Guo, and Andreas Vlachos. 2023. Averitec: A dataset for real-world claim verification with evidence from the web. Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 36.

Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023. REPLUG: Retrieval-Augmented Black-Box Language Models. arXiv preprint arXiv:2301.12652.

Prakhar Singh, Anubrata Das, Junyi Jessy Li, and Matthew Lease. 2021. The case for claim difficulty assessment in automatic fact checking. arXiv preprint arXiv:2109.09689.

Dominik Stammbach and Elliott Ash. 2020. e-fever: Explanations and summaries for automated fact checking. Proceedings ofthe 2020 Truth and Trust Online (TTO 2020), pages 32–43.

Alon Talmor and Jonathan Berant. 2018. The web as a knowledge-base for answering complex questions. In Proceedings ofthe 2018 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 641–651, New Orleans, Louisiana. Association for Computational Linguistics.

Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. 2022. LaMDA: Language Models for Dialog Applications. arXiv preprint arXiv:2201.08239.

Andreas Vlachos and Sebastian Riedel. 2014. Fact checking: Task definition and dataset construction. In Proceedings of the ACL 2014 Workshop on Language Technologies and Computational Social Science, pages 18–22, Baltimore, MD, USA. Association for Computational Linguistics.

Svitlana Volkova, Kyle Shaffer, Jin Yea Jang, and Nathan Hodas. 2017. Separating facts from fiction: Linguistic models to classify suspicious and trusted news posts on Twitter. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 647–653, Vancouver, Canada. Association for Computational Linguistics.

William Yang Wang. 2017. “liar, liar pants on fire”: A new benchmark dataset for fake news detection. In Proceedings of the 55th Annual Meeting of the Associationfor Computational Linguistics (Volume 2: Short Papers), pages 422–426, Vancouver, Canada. Association for Computational Linguistics.

Tomer Wolfson, Mor Geva, Ankit Gupta, Matt Gardner, Yoav Goldberg, Daniel Deutch, and Jonathan Berant. 2020. Break it down: A question understanding benchmark. Transactions ofthe Associationfor Computational Linguistics, 8:183–198.

Tianyi Zhang, Faisal Ladhak, Esin Durmus, Percy Liang, Kathleen McKeown, and Tatsunori B. Hashimoto. 2023. Benchmarking Large Language Models for News Summarization. arXiv eprint arxiv:2301.13848.

Zexuan Zhong, Tao Lei, and Danqi Chen. 2022. Training language models with memory augmentation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5657–5673, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

## Appendix

## A Experimental Details

A.1 List of Websites being Filtered

• www.politifact.com

• www.snopes.com

• www.factcheck.org

• www.washingtonpost.com/news/ fact-checker/

• www.apnews.com/hub/ap-fact-check

• www.fullfact.org

• www.reuters.com/fact-check

We also filter the URLs that contain “fact-check” or “factcheck”; we also filter any PDF files and videos.

Claim: Viral image stated on June 8, 2020 in post on Facebook: Cops in Norway: require 3 years of training, 4 people killed since 2002. Cops in Finland: require 2 years of training, 7 people killed since 2000. Cops in Iceland: require 2 years of training, 1 person killed since ever. Cops in the U.S.: require 21 weeks of training 8,000+ people killed since 2001.

Suppose you are a fact-checker, generate several yes or no ques ons to help me answer if this claim is true or false.

## Ques ons:

Does Norway require 3 years of training for cops?   
Have Norwegian cops killed 4 people since the early 2000's?   
Does Finland require only 2 years of training for police?   
Have Finnish police killed 7 people since 2000?   
Does Iceland only require 2 years of training for cops?   
Have Iceland cops only killed 1 person ever?   
Does the U.S. require only 21 weeks of training for cops?   
Have U.S. cops killed more than 8,000 people since 2001?   
Do experts associate only training me with police-related shoo ng fatali es?

Claim: Barry DuVal stated on September 25, 2015 in an interview: We're the only major oil-producing na on in the world with a self-imposed ban on expor ng our crude oil to other na ons.

Suppose you are a fact-checker, generate several yes or no ques ons to help me answer if this claim is true or false.

## Ques ons:

Is the U.S. the only major oil-producing na on to ban exports of crude oil? Is the self-imposed ban on crude oil export of U.S a complete ban?

Claim: William Barr stated on September 2, 2020 in a CNN interview: We indicted someone in Texas, 1,700 ballots collected from people who could vote, he made them out and voted for the person he wanted to.

Suppose you are a fact-checker, generate several yes or no ques ons to help me answer if this claim is true or false.

## Ques ons:

Were 1700 mail-in ballots inves gated for fraud in Texas during the 2020 elec on? Did the Jus ce Department indict someone in Texas for voter fraud? Did widespread mail-in order fraud happen in Texas during the 2020 elec on? Did voter disenfranchisement happen in Texas during the 2020 elec on?

## Claim: INPUT CLAIM

Suppose you are a fact-checker, generate several yes or no ques ons to help me answer if this claim is true or false.

Ques ons:

MODEL OUTPUT

Figure 6: Few-shot prompt we used to generate subquestions in this paper.

## A.2 Question Generation Prompt and Deduplication

The prompt we used to generate the questions is shown in Figure 6. Since the generated question set sometimes contains duplicates, we delete the duplicated questions according to the exact string match.

## A.3 Question-focused Summarization Prompt

The zero-shot and few-shot prompts we used to generate the claim-focused summaries are shown in Figure 7 and Figure 8 respectively.

## A.4 Hyperparameters of Veracity Classifier

• Model: DeBERTa-large

• Batch size: 32

Suppose you are assis ng a fact-checker to fact-check the claim: INPUT CLAIM

Summarize the relevant informa on from the document in 1-2 sentences. Your response should provide a clear and concise summary of the relevant informa on contained in the document. Do not include a judgment about the claim and do not repeat any informa on from the claim that is not supported by the document.

## Summariza on:

MODEL OUTPUT

Figure 7: Zero-shot prompt we used to generate the claim-focused summaries in this paper.
<table><tr><td></td><td>First-stage</td><td>Second-stage</td><td>Summ</td></tr><tr><td># documents</td><td>45.0</td><td>7.7</td><td>4.0</td></tr><tr><td># words</td><td>70,245</td><td>2,710</td><td>251</td></tr></table>

Table 8: Average number of unique documents and average number of words in total from those documents after each stage of our pipeline.

• Max sequence length: 512

• Epochs: 25

• Initial learning rate: 3e-5

• Optimizer: Adam with linear decay

• Metric for selecting best dev model: MAE

• Random seed of 5 runs: 290032, 33432, 7876, 366, 77

• Training device: NVIDIA-A6000

## B Information Compression through the Pipeline

Our pipeline progressively refines the crucial data needed to validate a claim. Table 8 demonstrates the average count of unique documents and the total word count in these documents after each phase of our pipeline under both temporal and site constraints.

## C Human Study

## C.1 Examples of Unfaithful Summaries

Figure 12 shows three examples containing unfaithful content. We see that the “Minor” error does not affect the interpretation of the original document while “Major” and “Completely Wrong” errors alter the view.

![](images/5c8f43b2e41dbae6ee54427c4a8df607d355d1b0a4cd873f03d13a0ce64fd83c.jpg)  
Figure 8: Few-shot prompt we used to generate the claim-focused summaries in this paper.

## C.2 Recruiting Process

Faithfulness study We set up a qualification test that consists of 5 examples. We selected workers from MTurk if they get more than 3/5 examples correct according to our curated labels and if they write reasonable rationales. In total, there are 31 workers who took the qualification test and we selected 15 of them for the task. We pay \$3 for the qualification test and \$2 dollars for one HIT that contains 4 document-summary pairs in the actual task. The detailed instructions and the annotation interface is shown in Figure 10.

Comprehensiveness study We set up a qualification test that consists of 10 examples. We selected workers from MTurk if they got more than 7/10 questions right according to our curated labels and if they write reasonable rationales. In total, there are 28 workers who took the qualification test and we selected 17 of them for the task. We pay \$3 for the qualification test and \$0.3 dollars for one

Use summary of documents to score how likely this claim is true at the scale of 0 to 100, 0 being pants-on- re and 100 being true.

Figure 9: Zero-shot prompt for Claim + summary might be because ChatGPT relies heavily on prior knowledge and it is not able to use the provided summary effectively. We believe improving this is a promising direction for future work.

question in the actual task.

The detailed instructions and the annotation interface is shown in Figure 11.

## D Using LLMs as a Veracity Classifier

We experiment with using ChatGPT (gpt-3.5-turbo) as the classifier in the final stage. Since ChatGPT is not trained on our training set, it does not have access to the label distribution of the dataset. To make a fair comparison with the DeBERTa model, instead of directly predicting a discrete label (one out of the six labels), we prompt the model to explain its reasoning process and predict a truthfulness score on a scale of 0 to 100, 0 for the claim being false and 100 for true. We then rank the examples according to the predicted scores and map the scores to discrete labels to the label distribution of the training set. To be specific, we rank the examples in the training set by their labels, assigning the lowest rank to pants-on-fire and the highest to true. Each label, denoted as $l _ { i } .$ corresponds to a percentile $p _ { i }$ . We then map the predicted score falling between $p _ { i }$ and $p _ { i + 1 }$ to the label $l _ { i } .$ . We use a zero-shot promp $^ { 1 2 }$ to produce the score and the prompt is shown in Figure 9.

The results are shown in Table 9. Comparing the claim-only results from the two models, we see that ChatGPT achieves slightly better performance than DeBERTa. However, unlike the DeBERTa model, when adding the summary, we see a notable performance drop for ChatGPT. We argue that this

## Instructions:

Thank you for participating in this task! This task aims to determine how trustable an AI system is at automatical y gathering the most relevant information from a document to verify a political claim.

You are given 1) a political claim, 2) a snippet of a document that is potentially relevant to check the political claim, and 3) a summary of the snippet generated by an AI system We want to evaluate whether the summary is faithful to each document. For the summary to be faithful, the summary should avoid adding any new information that is not present in the original document or misrepresenting the information presented to given document.

Note that your job is not to evaluate whether the document/summary is relevant or not to the claim. The claim is not meant to be used to judge whether the summary is faithful or not. It just provides you some context that may be helpful.

Major factual errors should be errors that cause the summary to actual y give a different impression than the original document. Minor factual errors are those where, even though some details may not align, they don't change the overal message of the document.

It's okay for the summary to cite the claim. However, if the summary contains an assessment regarding whether the document is relevant to the claim or not, try your best to evaluate whether the assessment made by the machine is accurate or not based on our criteria (correct, minor, major ...).

## Examples:

Highlights are added by us for il ustration but not present in real examples you wil see.

Claim

Ingraham said, "You know what the biggest lie is, is that restaurants are spreaders of COVID. There's no science for that." In fact, plenty of evidence suggests restaurant dining has helped spread the coronavirus. Places that al ow indoor dining and don’t fol ow

Document title: What are the main modes of transmission for COVID-19? - Live Science   
Content: least two people died from the virus , the Los Angeles Times reported . That suggested the viral particles were shed as aerosols by someone , before being inhaled or otherwise acquired by other choir members . A 2019 study in the journal Nature   
Scientific Reports ( opens in new tab ) found that people emit more aerosol particles when talking , and that louder speech volumes correlate to more aerosol particles being emitted . That case , along with those studies , suggest that the virus can be routinely   
transmitted via aerosols , though other routes of transmission ( such as large droplets being emitted during singing or speech ) are stil possible explanations . In the 2003 SARS outbreak , aerosol transmission occurred during hospital procedures that generated   
Then , another person touches that object , then touches their nose , mouth or eyes . The virus then sneaks into the body via the mucous membranes , infecting the second person . So far , no one knows how common this mode of transmission is , but it does   
seem to be possible . One study found that SARS-CoV-2 could remain viable on surfaces such as cardboard for up to 24 hours , and on plastic and steel for 2 to 3 days . Santarpia has studied viral surface contamination in the context of patients hospitalized   
with COVID-19 at the University of Nebraska Medical Center . In that study , which was published March 26 on the preprint database medRxiv , Santarpia and his col eagues found viral contamination in air samples , on surfaces such as toilets , and on   
frequently touched surfaces . Also on March 26

<table><tr><td rowspan=1 colspan=1>The document discusses the various modes of transmission of COvID-19, including aerosol transmission andcontact transmission.Aerosoltransmission ocurs when people emit more aerosol particles when talking, andcontact transmission happens whenviral particles emitted from an infected person land on a surface and are then touched by someone else. Studies have also found that SARS-CoV-2 can remain on surfaces, such as cardboard and plastic, for up to 24 hours and 2 to 3 days respectively.</td></tr><tr><td rowspan=1 colspan=1>Explanation</td></tr><tr><td rowspan=1 colspan=1>All of the facts presented in summary are supported by the document.</td></tr><tr><td rowspan=1 colspan=1>Example of non-faithful (Major Factual Error) summaryClaim</td></tr><tr><td rowspan=1 colspan=1>Joe Biden stated on August 31, 2020 in a speech: &quot;When I was vice president, violent crime fell 15% in this country... The murder rate now is up 26% across the nation this year under Donald Trump.&quot;</td></tr><tr><td rowspan=1 colspan=1>Document</td></tr><tr><td rowspan=1 colspan=1>Document title: Murder Rate Up 26% in 2017 - Nationwide 90FMContent: More than 1,265 people have been murdered since the start of the year ; making the country&#x27;s murder rate increase by 26 percent over the similar period last year . Nationwide News has obtained the latest crime statistics, from January up to October15 this year, from the Jamaica Constabulary Force&#x27;s Corporate Communications Unit, CCU. Superintendent Stephanie Lindsay is head of the CCU. [audio_mp3 url= &quot;htps : /nationwideradiojm.com/wp- content/uploads/2017/10/LINDSAY-murder-and-shooting-OCT202017.mp3&quot;] She says the St. James Police Division is leading the chart with 235 murders . It&#x27;s followed by Clarendon with 131, and St. Andrew South, 124 murders. Portland has recorded the fewest murders ; five . There&#x27;s a reduction inallmajor crimes except murders and shootings . Rapes , aggravated assaults and robberies have all declined by at least 14 percent . In the meantime , Superintendent Lindsay says the police have seized 698 guns and over 7,000 rounds of ammunition up toOctober 15 . [ audio_mp3 url= &quot; https :</td></tr><tr><td rowspan=1 colspan=1>Summary</td></tr><tr><td rowspan=1 colspan=1>The Jamaica Constabulary Force&#x27;s Corporate Communications Unit has reported that the countr&#x27;s murder rate has increased by 26 percent from January to October 15, 2020compared to the same period in 2019. Additionally, other major crimes such as rapes,aggravated assaults and robberies have all declined by at least 14 percent, while the police have seized 698 guns and over 7,000 rounds of ammunition up to October 15, 2020.</td></tr><tr><td rowspan=1 colspan=1>Explanation</td></tr><tr><td rowspan=1 colspan=1>The summary says the muder rate has increased by 26 percent in 2020, which is not factually correct according to the document that states the murder rate has increased by 26 percent in 2017. We would rate this as a Major factual err because it significantlychanges the interpretation of the document.</td></tr><tr><td rowspan=1 colspan=1>Your taskClaim 1</td></tr><tr><td rowspan=1 colspan=1>${claim1}</td></tr><tr><td rowspan=1 colspan=1>Document 1</td></tr><tr><td rowspan=1 colspan=1>$(d1}</td></tr><tr><td rowspan=1 colspan=1>Summary 1</td></tr><tr><td rowspan=1 colspan=1>${s1}Please read the summary and the document carefully as some of the errors are subtle and hard to spot. The claim is not meant to be used to judge whether the summary is faithful or not.</td></tr><tr><td rowspan=2 colspan=1>O FaithfulONon-Faithful (Minor factual error) ONon-Faithful (Major factual error) ONon-Faithful (Completely wrong)</td></tr><tr><td rowspan=1 colspan=1>Your Explanation in 12 short sentences</td></tr></table>

Please type your explanation

Figure 10: Interface of the faithfulness study we conducted in Section 6.1.
<table><tr><td>Model</td><td>Evidence</td><td>Acc</td><td>Soft-Acc</td><td>Macro-F1</td><td>MAE</td></tr><tr><td>ChatGPT</td><td>Claim only Claim + summary</td><td>32.0 24.5</td><td>66.0 67.5</td><td>31.0</td><td>1.16</td></tr><tr><td></td><td>Claim only</td><td></td><td></td><td>25.7</td><td>1.25</td></tr><tr><td>DeBERTa-large</td><td></td><td>25.5</td><td>68.0</td><td>27.5</td><td>1.12</td></tr><tr><td></td><td>Claim + summary</td><td>33.0</td><td>74.5</td><td>34.5</td><td>0.99</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 9: Veracity classification performance on the test set of CLAIMDECOMP with different prompts using ChatGPT.

![](images/44d8bca733cf55fb2fbbf06f84f2b693ee1d8c417df207e2e5e24aa91ce201c5.jpg)  
Figure 11: Interface of the comprehensiveness study we conducted in Section 6.2.

![](images/f9a37229dbb978ddc88463c23e3a7bf260759c3eef1c93812ded9c493470c1e5.jpg)  
Figure 12: Three examples from the faithfulness evaluation (Section 6.1), showing the cases of minor error, major error, and completely wrong, respectively. Red text denotes the mismatches between the summary and the document.