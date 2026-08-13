# S3Eval: A Synthetic, Scalable, Systematic Evaluation Suite for Large Language Models

Fangyu Lei <sup>1,2</sup>∗ , Qian Liu <sup>4</sup>∗, Yiming Huang <sup>1</sup>∗ Shizhu He <sup>1,2</sup>†, Jun Zhao <sup>1,2</sup>, Kang Liu <sup>1,2,3</sup>†

<sup>1</sup>The Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences, Beijing, China <sup>2</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China <sup>3</sup>Shanghai Artificial Intelligence Laboratory, Shanghai, China <sup>4</sup>Sea AI Lab, Singapore leifangyu2022@ia.ac.cn liuqian@sea.com kliu@nlpr.ia.ac.cn

## Abstract

The rapid development of Large Language Models (LLMs) has led to great strides in model capabilities like long-context understanding and reasoning. However, as LLMs are able to process longer contexts, it becomes more challenging to evaluate whether they have acquired certain capabilities, since the length of text (e.g., 200K tokens) they can process far exceeds what humans can reliably assess in a reasonable duration. In this paper, we propose using complex synthetic tasks as a proxy evaluation method, and present S3EVAL, a Synthetic, Scalable, Systematic evaluation suite for LLMs evaluation. The synthetic nature of S3EVAL provides users full control over the dataset, allowing them to systematically probe LLM capabilities by scaling text length and varying task difficulty across diverse scenarios. The strong correlation between S3EVAL and real-world benchmarks demonstrates the soundness of using S3EVAL for evaluation of LLMs. S3EVAL provides a flexible and infinite long-context data generation method. We have generated a comprehensive dataset called S3EVAL-Standard, and experimental results have shown that it poses significant challenges for all existing LLMs. Our code is available at https://github.com/lfy79001/S3Eval.

## 1 Introduction

Large Language Models (LLMs) have greatly propelled significant advancements in Natural Language Processing (NLP), such as OpenAI GPT (Brown et al., 2020), Llama (Touvron et al., 2023a,b), StarCoder (Li et al., 2023a), and others. These models perform well in many NLP tasks and claim to have made progress in advanced capabilities such as reasoning, long-context understanding, and so on. However, existing benchmarks (Chang et al., 2023) often fail when it comes to evaluating extremely long-context LLMs or analysing the controllable characteristics and limitations of LLMs.

![](images/3f0296c2d54a17d0723c7ba2d2fc410ff8e56aeb8749f5c6a4ea00a6477e24ef.jpg)  
Figure 1: Needle-in-A-HayStack cannot demonstrate the performance of the model under real tasks, but S3EVAL can. Compared with needle-in-a-haystack, S3EVAL is more relevant to real benchmarks and more difficult.

For long-context understanding, previous work has often evaluated LLMs using the scope of language modeling metrics (i.e., perplexity) (Sun et al., 2021; Peng et al., 2023) or the performance on simple artificial tasks (Li and Roth, 2002; Berant et al., 2013; Mohtashami and Jaggi, 2023). There is a widely used evaluation method known as Needlein-a-Haystack (Kamradt, 2023), as shown in Figure 1. In this method, a vital piece of information is concealed within a lengthy document, resembling a haystack, and the model’s objective is to locate and retrieve this hidden key information. However, these evaluation tasks tend to lack complexity and are narrowly focused on simple comprehension, which is misaligned with the sophistication required for real-world downstream applications.

While recent work has made great progress on building evaluation benchmarks at longer context lengths with real-world use cases (e.g, question answering) (Bai et al., 2023b; An et al., 2023), these manually annotated datasets often lack the scale and diversity to thoroughly assess performance on extended context lengths. For example, existing benchmarks struggle to effectively evaluate LLMs that claim an ability to process contexts up to 100K tokens, due to the limited capacity of human annotation for very long text. Developing more scalable and diverse evaluation datasets, potentially leveraging automated supervision, remains an open challenge.

For reasoning analysis (Hendrycks et al., 2021b; Chen et al., 2021a; Suzgun et al., 2023; Zhong et al., 2023), conducting both qualitative and quantitative analysis of answers and reasoning processes provides important insights. However, existing benchmarks lack the ability to precisely control the distribution of the dataset, limiting their utility for in-depth research analysis. In other words, the nature of these benchmarks makes it challenging for developers to identify the specific weaknesses of their LLMs. More configurable and granular benchmarks are needed to enable detailed analysis of model performance. In addition, these benchmarks often draw their evaluation data from NLP tasks that have been extensively studied and are likely to be used in the training corpus of LLMs. The potential data leakage makes the evaluation less convincing.

In this paper, we propose a new evaluation suite called S3EVAL, which addresses the aforementioned issues by using a complex synthetic task - SQL execution - as a proxy for the performance of LLMs on realistic reasoning tasks. As shown in Figure 2, inspired by the work of TAPEX (Liu et al., 2022), S3EVAL is based on the SQL execution task. Specially, given a randomly generated table and a random SQL query, S3EVAL evaluates whether LLMs can return the correct execution results. S3EVAL has three notable characteristics: (1) It is synthetic, with no table or SQL query present in the LLM training corpus. The tasks use complex, grammatically correct SQL syntax, making them very challenging. (2) It is scalable, allowing users to customize the benchmark to any length and difficulty. (3) It is systematic, containing diverse reasoning types and operations. This enables comprehensive evaluation of LLM capabilities.

With these powerful features, developers can extend the context to really long lengths and generate meaningful SQL statements using S3EVAL. We conducted comprehensive multi-perspective experiments on several popular LLMs using S3EVAL. Experimental results demonstrated that the performance of LLMs on S3EVAL aligns closely with their performance on mainstream LLM benchmarks. While LLMs have shown impressive capabilities, our work reveals limitations in their ability to leverage long contexts, since we observe performance degradation of almost all LLMs in long-context settings. By carefully studying experimental results, we can work to pinpoint situations where LLMs tend to fail and summarize valuable insights.

In the era of rapid LLM development, the most significant contribution of S3EVAL lies in its effectiveness as a method for long-context evaluation. Capable of generating evaluation data of infinite length, it ensures that assessments are not only reasonable but also sufficiently challenging.

## 2 Synthetic: Suite and Benchmark

In this section, we introduce the details of the S3EVAL evaluation suite (as shown in Figure 2) and the new benchmark we proposed.

## 2.1 Suite Construction

Task Formulation Following previous work (Liu et al., 2022), each example in S3EVAL generally contain an SQL query and a (semi-)structured table T as the input. Each table $T$ consists of M rows $\{ r _ { i } \} _ { i = 1 } ^ { M }$ , in which each row $r _ { i }$ contains N cell values $\left\{ c _ { \left. i , j \right. } \right\} _ { i = 1 } ^ { M }$ . Each cell $c _ { \langle i , j \rangle }$ corresponds to a table header $h _ { j }$ . Each SQL query consists of K tokens as $x = x _ { 1 } , x _ { 2 } , \cdot \cdot \cdot , x _ { K }$ Each token $x _ { i }$ originates from SQL keywords, table schema, or table cells. Each multi-step instruction is transformed from SQL query. The task prompts LLM to obtain the execution result A of the SQL on the table T. Our main focus is on analyzing the accuracy of LLM in executing SQL queries.

Random Table Generation All tables in S3EVAL are randomly generated and do not contain any real data or overlap with existing public tables. The tables have M rows and N columns, with adjustable parameters M and N. The column headers are sampled from English nouns (Bird, 2006), falling into three types: TEXT, INT, and DATE. INT columns contain random integers from 1 to 1000, which is an adjustable range. DATE columns have values in year-month-day format. TEXT columns have random strings of length 5 to 12 characters, which is also adjustable. To simulate real-world data where the same value may recur in a column frequently, the data generator includes a parameter to set the probability of duplicating values within a specific column.

![](images/f0370f9ca42d1034ced7afd643cda2682bd06d7bbae9b7fb0a6c04c6105530b4.jpg)  
Figure 2: The illustration demonstrates the S3EVAL pipeline, where the capabilities of LLMs are assessed by evaluating their ability to execute SQL queries over randomly generated tables.

<table><tr><td colspan="2">Configuration</td><td>Description</td></tr><tr><td rowspan="5">Table Control</td><td># of Rows # of Columns</td><td>The number of rows in the generated tables</td></tr><tr><td></td><td>The number of columns in the generated tables</td></tr><tr><td>Header Type Ratio</td><td>The proportion of table column types that are TEXT, INT, DATE</td></tr><tr><td>Cell Uniqueness</td><td>The proportion of duplicate cells in each column</td></tr><tr><td>String / Int Length</td><td>The string length or numeric range of cell values</td></tr><tr><td rowspan="8">Instruction Control</td><td>SQL Keywords</td><td>SELECT, WHERE, GROUP BY, HAVING, ORDER BY</td></tr><tr><td>SQL Length</td><td>The number of tokens after SQL split by space</td></tr><tr><td>Column Čoverage</td><td>The ratio of columns involved in SQL execution to total columns.</td></tr><tr><td>Row Coverage</td><td>The ratio of rows involved in SQL execution to total rows</td></tr><tr><td>Calculate Times</td><td>The number of SQL numerical calculations.</td></tr><tr><td>Filter Times</td><td>The number of SQL filtering operations.</td></tr><tr><td>Aggregator</td><td>COUNT, MAX, MIN, SUM, AVG</td></tr><tr><td>Filter Operator</td><td> $> , < , = , \mathrm { I N , L I K E }$ </td></tr><tr><td rowspan="3">Output Control</td><td>Answer Location</td><td>The location of SQL answers in the input table</td></tr><tr><td># of Answer Cells</td><td>The number of selected cells in the answer</td></tr><tr><td>Answer Length</td><td>The total number of tokens in the answer</td></tr></table>

Table 1: Our S3EVAL method allows users to customize configuration settings and provides descriptions for each parameter that can be adjusted. More configurations can be found in Appendix D.1.

Random SQL Generation The SQL language includes a variety of statements to query and manage data. S3EVAL use context free grammar to generate a specific number of examples with controllable attributes. As Table 1 shows, the S3EVAL tool allows configuring several parameters of generated SQL statements, including nesting depth, keywords used, length, coverage of SQL features, computational complexity, and more. For example, calculate times can be modified to control the complexity of numerical reasoning for each dataset. Except these configures, users can also manually write the specified SQL template to generate finegrained evaluation data (Appendix C.2).

Evaluation Methods S3EVAL includes both zero-shot and few-shot prompting methods. For each few-shot setting, all examples share one table. N-shot is formalized as $\mathrm { I N P U T } = [ \mathrm { T } ; \mathrm { S } _ { 1 } ; \mathrm { A } _ { 1 } ; . . . ; \mathrm { S } _ { \mathrm { n } + 1 } ]$ For the input format of table T, we designed several alternative ways, including markdown, flatten, tapex-style, etc.

To evaluate the performance of LLMs, we use Exact Match (EM) as the evaluation metric. Details are shown in Appendix C.3.

## 2.2 S3EVAL-Standard Benchmark

We generate a highly diverse dataset called S3EVAL-Standard covering lengths ranging from 2K to 40K, with various difficulty levels of reasoning types, which comprises all templates and operations included in S3EVAL, making it the most complex and diverse dataset available. We utilize this version of the dataset as the official benchmarking data for S3EVAL benchmark. It can effectively evaluate LLMs in completing tasks under both short-context and long-context scenarios. We evaluate popular commercial LLMs and opensource LLMs on S3EVAL-Standard, and the experimental results are shown in Table 2. In theory, we can measure the performance of LLMs with unlimited context length here.

<table><tr><td>Model</td><td>Context Length</td><td>Short-Context</td><td>Long-Context</td><td>Total</td></tr><tr><td>GPT-4-32K</td><td>32768</td><td>68.4%</td><td>43.0%</td><td>54.8%</td></tr><tr><td>GPT-3.5-Turbo</td><td>16384</td><td>39.9%</td><td>16.2%</td><td>27.0%</td></tr><tr><td>Code Llama (70B)</td><td>16384</td><td>33.9%</td><td>8.9%</td><td>20.3%</td></tr><tr><td>LLaMA-2 (70B)</td><td>4096</td><td>30.0%</td><td>8.8%</td><td>18.4%</td></tr><tr><td>LLaMA-2 (13B)</td><td>4096</td><td>21.7%</td><td>4.6%</td><td>12.4%</td></tr><tr><td>LLaMA-2 (7B)</td><td>4096</td><td>20.8%</td><td>4.4%</td><td>11.9%</td></tr><tr><td>Gemma (7B)</td><td>8192</td><td>28.9%</td><td>8.6%</td><td>17.9%</td></tr><tr><td>Qwen 1.5 (14B)</td><td>32768</td><td>33.7%</td><td>14.4%</td><td>23.2%</td></tr><tr><td>Qwen 1.5 (7B)</td><td>32768</td><td>26.5%</td><td>8.0%</td><td>16.5%</td></tr><tr><td>Qwen 1.5 (4B)</td><td>32768</td><td>22.8%</td><td>5.5%</td><td>13.4%</td></tr><tr><td>Mixtral-8x7B (46.7B)</td><td>32768</td><td>31.5%</td><td>11.1%</td><td>20.4%</td></tr><tr><td>Mistral-Instruct-v0.2 (7B)</td><td>32768</td><td>28.7%</td><td>10.6%</td><td>18.9%</td></tr></table>

Table 2: Experimental results on S3Eval-Standard. “Total” denotes the overall score, “Short-Context” refers to the model’s performance on contexts shorter than 4K in length, and “Long-Context” indicates the model’s performance on contexts ranging from 4K to 40K in length.

## 3 Correlation with Realistic Benchmark

In this section, we describe the details of synthesizing the evaluation data (Section 2.1) and verify the correlation between our synthetic suite S3EVAL and real-world benchmark results.

## 3.1 Experimentual Settings

S3EVAL can flexibly generate different evaluation data. To validate the rationality of S3EVAL, we conducted correlation experiments and generated two sets of data with different difficulty levels for experimentation. Easy is the simplest data that S3EVAL can generate and is used to evaluate LLM’s ability to understand the most basic instructions. It contains only one template, “SELECT <col1> WHERE <col2> <op> <value>”. General is a more difficult setting, containing extensive SQL syntax, and its generating setting is described in Appendix D.2. All experiments were run for 3 times, using 1000 randomly generated queries per trial, with tables of 15 rows and 8 columns and an average of 1200 tokens per input. Details on the LLMs are provided in Appendix D.3.

Considering SQL execution is a difficult task, some models may have a poor understanding of symbolic language, which makes it difficult to execute SQL, so we propose an alternative task SQL multi-step task to remove this potential bias. Specifically, it converts an SQL query into a multi-step table operation instruction as shown in Appendix C.6. SQL has a fixed execution flow for the query statement: FROM ON JOIN WHERE GROUP BY HAVING SELECT ORDER BY  LIMIT. This is not consistent with the order in which it is written. With this processing, it can also generate chain-of-thought prompting data.

## 3.2 Scaling Law

Previous work (Kaplan et al., 2020; Hoffmann et al., 2022) shows a positive correlation between the cross-entropy loss of LLMs and the amount of computing resources used for training, as described by the empirical scaling law. To verify whether the scaling law holds for our S3EVAL, we employ a set of checkpoints of Pythia-12B (Biderman et al., 2023) that are open-sourced at different training steps, corresponding to different amounts of compute. We observe a consistent pattern as illustrated in Figure 3: the scores show a smooth progression of improvement that aligns with the scaling law with increasing the training steps. The steady, incremental performance gains over time, lacking any spikes, demonstrate S3EVAL’s reliability as a evaluation suite. Overall, these experimental results confirm the scaling law’s accuracy in forecasting model gains during training across diverse evaluation settings.

![](images/bbfdd7083bd79853bfaf9949f466dd9ca8e0cbc3a6f89814545175212dde1ebc.jpg)  
Figure 3: The performance of Pythia-12B on S3EVAL was evaluated across different training steps.

![](images/e247dba5d894be25afbb2c2bfdab1abb4bd0b0d9f887cfe8043285afae14d9a3.jpg)  
Figure 4: The performance of different LLMs on S3EVAL and WikiTableQuestions.

## 3.3 Benchmark Performance

In the above, we validated that the LLMs also exhibits the scaling law observed in NL on the S3EVAL suite. A natural question that arises is whether its performance on S3EVAL is correlated with the performance on real-world, NL benchmarks. To examine the hypothesis, we first compare the performance of different LLMs on S3EVAL and on WikiTableQuestions (Pasupat and Liang, 2015), a table question answering dataset consisting of questions and answers. It is worth noting that to align the difficulty, we use the SQL queries from WikiTableQuestions (Shi et al., 2020) as our S3EVAL evaluation set.

To systematically compare the performance, following previous work (Liu et al., 2023a), we consider two correlation measures: the Pearson correlation coefficient (r), which evaluates the linear relationship between model scores on the two benchmarks, and the Kendall rank correlation coefficient (τ), which assesses whether the relative ranking of models is consistent across the benchmarks. The strong correlation between LLMs’ performance on the SQL execution task and the table question answering task, as evidenced by the high r (e.g., 99.1) and high τ (e.g., 93.6) in Figure 4.

Although S3EVAL has shown significant correlation with WikiTableQuestions, the fact that they are both tasks on tables may cause one to question whether S3EVAL can serve as a proxy task to evaluate LLMs’ capabilities on generic reasoning tasks. Therefore, we also compare the performance on S3EVAL with the results of generic popular benchmarks like BBH (Suzgun et al., 2023) and HumanEval (Chen et al., 2021a). The results depicted in Figure 5a demonstrate a strong correlation between LLM performance on S3EVAL and the BBH benchmark, with BBH performance obtained from the OpenCompass platform using few-shot chain-of-thought prompting (OpenCompass, 2023). Similarly, Figure 5b illustrates the correlation between S3EVAL performance and pass@1 scores on HumanEval (Chen et al., 2021b) for code LLMs. The results demonstrate that S3EVAL serves as a robust proxy task for assessing the reasoning capabilities of LLMs on realistic benchmarks. Concrete experimental results are provided in Table 4.

## 4 Scalable: Unlimited Evaluation Resources

S3EVAL provides a unique capability to generate infinite number of examples (Section 4.1) with infinite length (Section 4.2).

## 4.1 Scalable Number of Evaluation Examples

The strength of S3EVAL is its ability to generate unlimited number of examples for evaluation. This stems from two key design choices in S3EVAL: (1) the synthetic table size can be scaled to different number of rows and columns, and (2) the table cells are synthesized from randomly generated strings. Combined with the provided large library of SQL query templates, these features enable the creation of a near-infinite set of unique evaluation examples. This kind of capacity enables the continuous creation of novel examples unseen during training, which helps safeguard test data integrity by preventing leakage of the evaluation set into the training corpus.

![](images/5748099084870b772160ac4cffc8c87986219ea226d7f6dd64871f1d3f012db6.jpg)  
(a) Performance of large language models

![](images/45d831a47ec345812996a130461aa1bfb9e6058213010605d1574a2a4484477a.jpg)  
(b) Performance of code large language models

Figure 5: Each point in the scatterplot represents the LLM performance on the benchmarks corresponding to the horizontal and vertical coordinates. The black straight line is the trend line. The larger the values of r and τ, the higher the correlation between the two benchmarks. We consider τ > 0.8 to be high concurrence.  
![](images/9091e22416633ff9f54c10eae216ad82497078af78c950fafdfccf15f9bfc24f.jpg)  
Figure 6: SQL execution training experiments on S3EVAL.

However, the absence of data leakage does not necessarily mean that S3EVAL’s performance always represents the model’s out-of-distribution generalization ability. It is because the model may perform well on S3EVAL via domain-specific training on the SQL execution task, rather than acquiring more general abilities. To investigate whether LLMs can “hack” S3EVAL via domain-specific training, we fine-tuned StarCoder-1B (Li et al., 2023a), which is not able to solve SQL execution tasks, on a randomly generated dataset of one million examples. The performance of the fine-tuned

![](images/47b2c139c3f9cacfaf6e34fd18bdfebb596c6310302dbf1f1ccb7ec6aedaf6b4.jpg)  
Figure 7: Experiment results of different LLMs on different context lengths.

StarCoder-1B is illustrated in Figure 6, where it is evaluated on three types of test datasets: Seen Table (same tables as training), Unseen Table (new tables in same format as training tables), and Unseen Templates (new SQL query templates). For the unseen table setting, we explore different table shapes, where (x y) means the table consists of x rows and y columns.

The experimental results demonstrate that for Unseen Tables with different shapes, regardless of their size, the performance of the fine-tuned Star-Coder experiences a substantial decline compared to Seen Tables. Likewise, when faced with Unseen Templates, the performance of the fine-tuned Star-Coder exhibits a significant drop. The results indicate that even if LLMs have been heavily trained on SQL execution tasks, their out-of-distribution performance can still be accurately evaluated by using novel SQL templates. These new SQL templates can be easily generated thanks to the vast grammar of SQL queries. Additionally, evaluating LLMs on larger tables that they were not trained on can also reveal part of their out-of-distribution capabilities.

## 4.2 Scalable Length of Evaluation Examples

One advantage of S3EVAL is its scalability and adjustable context length per example. The flexibility allows S3EVAL to rigorously evaluate LLMs that claim capability with long contexts. To clearly expose limitations of current LLMs, we intentionally chose the Easy setting in S3EVAL to evaluate their performance. Specifically, we establish table configurations with approximately 2K, 4K, 8K, and 16K tokens, by using different numbers of rows and fixing the number of columns. We generate a dataset consisting of 500 samples for each evaluation setting. The experimental results on up to 16K context length are plotted in Figure 7. As observed, the performance of almost all LLMs, significantly decreases as the context length increases. Of all the models, Claude-1.3-100K is the only one that maintains a relatively strong performance trend. Detailed results can be found in Appendix A.5.

As illustrated in Table 2, S3EVAL poses significant challenges for models even when the context window is extended to 32K levels. This difficulty arises from S3EVAL being rooted in real-world tasks, enabling it to generate evaluation data of infinite length and ensure the tasks are both reasonable and demanding. Looking ahead, as models progress to the 200K level, S3EVAL will likewise be poised to furnish effective evaluation data.

## 5 Systematic Suite: Controllable Analysis

S3EVAL provides a comprehensive framework that empowers developers to synthesize diverse evaluation examples for systematically assessing LLMs from multiple perspectives. In this section, inspired by the work of lost in the middle (Liu et al., 2023b), we first analyze the impact of answer position on performance (Section 5.1). Then we evaluate LLMs from different viewpoints, and we have conducted some initial explorations on the reasoning types analysis (Section 5.3). Last, we provide some insights by analyzing LLMs on three selected SQL templates (Section B.2). These experiments reveal counter-intuitive performance trends and new discoveries that may inspire further research and extension of the work.

![](images/c62cbb9b8149501757a62e092b48923039d9e3ea6e42898587b7cf280a59b9c7.jpg)

(a) ChatGPT  
![](images/879cba83ad35e15147e9c56489dfd5526520fc3acedaee04ccf2d7a058e2f874.jpg)  
(b) CodeLlama  
Figure 8: The relationship between LLMs performance and the position of the answer token.

## 5.1 Answer Position Analysis

We investigate the influence of the answer’s position on the performance of LLMs, which is generally considered important. Unlike standard NLP benchmarks where it is difficult to control the position of the answer, S3EVAL allows for fine-grained control of answer position at the token level. To mitigate the influence of long contexts, we only analyzed answers that fell within a limited context window (i.e., less than 4K tokens).

Echoing the findings of Liu et al. (2023b), “lost in the middle”, our results in Figure 8 demonstrate that both ChatGPT and CodeLlama achieve higher performance when the answer is located at the beginning or end of the context, compared to when it appears in the middle. In addition, we found a periodic fluctuation trend in the performance of both models as the position of the answer shifts within the context. For example, the performance of Chat-GPT increases from 0 to around 200, then starts to decrease from around 200 to 500. This wave-like pattern in performance appears to correlate with the position embedding approach used by LLMs.

![](images/cb8297f25152294271414bf3e2b1976f2bf1d99c859f1146a84cc621a300c1ac.jpg)

![](images/71137e94e597d0ac902caa44308cdb7cf56f546915bb0bc799c32d580308badb.jpg)  
Figure 9: Experiment results of ChatGPT and Yarn-Llama 2 on Dense and Sparse Settings. Dense means that the answer cells (i.e., Liu, Chen, Lei, Wang) lie in adjacent rows, and Sparse means that the answer cells are separated. The model performs better on local queries which only involves adjacent cells.

In contrast to previous studies that used longcontext question answering tasks (Liu et al., 2023b; Bai et al., 2023b) for analysis and are thus limited to controlling answer positions at the paragraph level, S3EVAL provides a more precise approach by focusing on token level. This key difference enables S3EVAL to offer fine-grained control and promote the exploration of relevant phenomena.

## 5.2 Answer Distribution Analysis

Given the limitation of existing LLMs on longcontext tasks, we are curious about the bottleneck of them. By using S3EVAL, we can systematically investigate the long-context modeling capabilities of LLMs by controlling the distribution of answers in the evaluation suite. Specifically, we use the Easy setting and fix the number of answers to four cells (i.e., the result of the SQL execution is always spanning four cells). As illustrated in Figure 9, we introduce two distribution patterns, Dense and Sparse <sup>1</sup> to probe the limitations of current LLMs. The dense mode only requires the model to understand the local context, whereas the sparse mode requires the model to have a broader, global understanding of the context across multiple blocks. The sparse mode intuitively poses more challenges and demands more complex reasoning across a broader scope of the provided context. We conduct experiments on ChatGPT and Yarn-llama2-13B (Peng et al., 2023). The experimental results indicate that both models perform significantly better in dense mode compared to sparse mode, as shown in Figure 9. This indicates that LLMs struggle to retrieve information over long sequences, even though their pre-training included lengthy contexts. This may be caused by the fact that the training data does not contain sufficient examples of long-distance dependencies for the model to learn effectively. Furthermore, the steep drop in performance from 4K to 8K tokens for both ChatGPT and Yarn-Llama2 in dense mode indicates that current length extension techniques may not be as effective as hoped. In summary, we believe that S3EVAL provides a valuable framework for evaluating long-context language models, as it allows testing models on dialogues of arbitrary length. This establishes a solid foundation for advancing research on large language models that can leverage long-term context.

## 5.3 Reasoning Type Analysis

S3EVAL enables the creation of multiple templates to generate different SQL statements, with each statement representing a distinct reasoning type. We selected six common reasoning types to investigate the reasoning capabilities of LLMs and examined four different LLMs: ChatGPT, Claude, Mistral-7B, and CodeLlama-34B. Following Liu et al. (2022), the six reasoning types <sup>2</sup> we considered are Filter, Aggregate, Arithmetic, Superlative, Comparative, and Group. The example SQL and the experimental results of different LLMs are presented in Table 3. The expressive power of SQL queries enables S3EVAL to be used for evaluating diverse scenarios such as numerical reasoning, multi-hop reasoning, complex code understanding, and multi-turn interaction with intermediate execution results.

<table><tr><td>Operator</td><td>Example SQL</td><td>ChatGPT</td><td>Claude</td><td>Mistral</td><td>CodeLlama</td></tr><tr><td>Filter</td><td>SELECT lyonnais FROM table WHERE farmer = &#x27;mijl&#x27; AND lashing &gt;288</td><td>79.6</td><td>79.2</td><td>64.8</td><td>72.8</td></tr><tr><td>Arithmetic</td><td>SELECT synset + refuge FROM table WHERE blender = &#x27;owxdbzjg&#x27;</td><td>67.2</td><td>59.4</td><td>5.4</td><td>10.6</td></tr><tr><td>Comparative</td><td>SELECT upsetter &lt; jollity FROM table WHERE kelp = 150</td><td>45.2</td><td>46.4</td><td>44.8</td><td>46.6</td></tr><tr><td>Aggregate</td><td>SELECT MIN(skeptic) FROM table</td><td>38.4</td><td>39.4</td><td>28.4</td><td>33.8</td></tr><tr><td>Group</td><td>SELECT lats FROM table GROUP BY shas- tan HAVING sum ( logbook ) = 56</td><td>38.1</td><td>28.2</td><td>31.0</td><td>37.8</td></tr><tr><td>Superlative</td><td>SELECT severity FROM table ORDER BY bierce DESC Limit 1</td><td>24.8</td><td>41.4</td><td>19.2</td><td>28.3</td></tr></table>

Table 3: Reasoning types experiments examples of different LLMs.

## 6 Related Work

Evaluating large language models (LLMs) has garnered significant interest in the NLP community (Chang et al., 2023). This allows us to gain a deeper understanding of the specific capabilities and limitations of LLMs while guiding further research. Researchers proposed MMLU (Hendrycks et al., 2021a) to measure the knowledge acquired by a language model during pre-training. In recent years, with the development of LLMs, a series of general evaluation benchmarks have emerged. For instance, BBH (Suzgun et al., 2023) and AGIEval (Zhong et al., 2023) assess the reasoning ablitities. GSM8K (Cobbe et al., 2021) evalutes the math reasoning, HumanEval (Chen et al., 2021a) and MBPP (Austin et al., 2021) measure code capalities. Our work aims to provide an evaluation suite for measuring reasoning ability.

Many previous works on long-text modeling rely on the perplexity (Sun et al., 2021; Peng et al., 2023) or performance on simple artificial tasks (Li and Roth, 2002; Berant et al., 2013; Mohtashami and Jaggi, 2023). Concurrently, Zero-SCROLLS (Shaham et al., 2023), L-Eval (An et al., 2023) and LongBench (Bai et al., 2023b) are proposed as evaluation benchmarks for long-text modeling. However, these benchmarks are built from existing public datasets and have fixed evaluation types. In contrast, S3EVAL can effectively assess comprehension of infinitely long-context. Furthermore, S3EVAL allows customization of settings to generate evaluation data that meets specific needs, enabling effective evaluation of model deficiencies and discovery of new insights into LLMs.

## 7 Conclusion

In this paper, we have introduced S3EVAL, a novel synthetic evaluation suite for LLMs using SQL execution. S3EVAL represents a scalable and systematic approach to evaluate LLMs on a dynamic task. Our experiments demonstrate strong correlation between S3EVAL and traditional evaluation benchmarks. The key innovations of S3EVAL are its flexibility, allowing unlimited context length and unlimited evaluation examples, and its fine-grained, systematic nature which enables detailed analysis of model capabilities and flaws.

Most importantly, for long-context evaluation, S3EVAL can generate evaluation data of infinite length. This type of task is not only challenging but also rooted in real-world tasks. Considering the rapid development of LLMs, even as LLM lengths extend significantly, S3EVAL can serve as a valuable benchmark for LLM development and contribute to the community.

## Limitations

Besides the features described in this paper, it currently supports complex multi-turn SQL execution task and multi-turn instruction task. Moreover, it also supports multilingual testing, especially for reasoning data generation of low-resource languages, which has not been widely studied by the academic community. However, this paper has not yet conducted a systematic analysis of these complex new features.

In addition, due to the complex and diverse syntax of SQL, the syntax that S3EVAL can generate is still relatively limited, which is also what we need to do in our future work. Moreover, there is currently no toolkit that can randomly generate a large number of complex SQLs, which is also a significance of our work.

Due to space limitations, many valuable experimental results are shown in Appendix B. We analyzed in detail the impact of various types of influencing factors on the results and have drawn other valuable conclusions.

Exploring the treasure contained in synthetic data is our goal for the future, and we believe that this work can bring inspiration to this field.

## Acknowledgements

This work was supported by the National Key R&D Program of China (No.2022ZD0160503) and the National Natural Science Foundation of China (No.62376270), Youth Innovation Promotion Association CAS, and OPPO Research Fund.

## References

Chenxin An, Shansan Gong, Ming Zhong, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. 2023. L-eval: Instituting standardized evaluation for long context language models. CoRR, abs/2307.11088.

Alex Andonian, Quentin Anthony, Stella Biderman, Sid Black, Preetham Gali, Leo Gao, Eric Hallahan, Josh Levy-Kramer, Connor Leahy, Lucas Nestler, Kip Parker, Michael Pieler, Shivanshu Purohit, Tri Songz, Wang Phil, and Samuel Weinbach. 2021. GPT-NeoX: Large Scale Autoregressive Language Modeling in PyTorch.

Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. 2021. Program synthesis with large language models. CoRR, abs/2108.07732.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023a. Qwen technical report. CoRR, abs/2309.16609.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2023b. Longbench: A bilingual, multitask benchmark for long context understanding. CoRR, abs/2308.14508.

Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. 2013. Semantic parsing on freebase from question-answer pairs. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, EMNLP 2013, 18-21 October 2013, Grand Hyatt Seattle, Seattle, Washington, USA, A meeting ofSIGDAT, a Special Interest Group ofthe ACL, pages 1533–1544. ACL.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. 2023. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 2397–2430. PMLR.

Steven Bird. 2006. NLTK: the natural language toolkit. In ACL 2006, 21st International Conference on Computational Linguistics and 44th Annual Meeting of the Associationfor Computational Linguistics, Proceedings ofthe Conference, Sydney, Australia, 17-21 July 2006. The Association for Computer Linguistics.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Kaijie Zhu, Hao Chen, Linyi Yang, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie. 2023. A survey on evaluation of large language models. CoRR, abs/2307.03109.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N.

Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021a. Evaluating large language models trained on code. CoRR, abs/2107.03374.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021b. Evaluating large language models trained on code. CoRR, abs/2107.03374.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. CoRR, abs/2110.14168.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring mathematical problem solving with the MATH dataset. In Proceedings ofthe Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. 2022. Training compute-optimal large language models. CoRR, abs/2203.15556.

Greg Kamradt. 2023. Needle in a haystack - pressure testing llms. https://github.com/gkamradt/ LLMTest\_NeedleInAHaystack.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. CoRR, abs/2001.08361.

Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy V, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Moustafa-Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. 2023a. Starcoder: may the source be with you! CoRR, abs/2305.06161.

Xin Li and Dan Roth. 2002. Learning question classifiers. In 19th International Conference on Computational Linguistics, COLING 2002, Howard International House and Academia Sinica, Taipei, Taiwan, August 24 - September 1, 2002.

Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. 2023b. Textbooks are all you need II: phi-1.5 technical report. CoRR, abs/2309.05463.

Nelson F. Liu, Tony Lee, Robin Jia, and Percy Liang. 2023a. Do question answering modeling improvements hold across benchmarks? In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 13186–13218. Association for Computational Linguistics.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023b. Lost in the middle: How language models use long contexts. CoRR, abs/2307.03172.

Qian Liu, Bei Chen, Jiaqi Guo, Morteza Ziyadi, Zeqi Lin, Weizhu Chen, and Jian-Guang Lou. 2022. TAPEX: table pre-training via learning a neural SQL executor. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Amirkeivan Mohtashami and Martin Jaggi. 2023. Landmark attention: Random-access infinite context length for transformers. CoRR, abs/2305.16300.

Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. 2022. A conversational paradigm for program synthesis. CoRR, abs/2203.13474.

Erik Nijkamp, Tian Xie, Hiroaki Hayashi, Bo Pang, Congying Xia, Chen Xing, Jesse Vig, Semih Yavuz, Philippe Laban, Ben Krause, Senthil Purushwalkam, Tong Niu, Wojciech Kryscinski, Lidiya Murakhovs’ka, Prafulla Kumar Choubey, Alexander R. Fabbri, Ye Liu, Rui Meng, Lifu Tu, Meghana Bhat, Chien-Sheng Wu, Silvio Savarese, Yingbo Zhou, Shafiq Joty, and Caiming Xiong. 2023. Xgen-7b technical report. CoRR, abs/2309.03450.

OpenCompass. 2023. Opencompass: A universal evaluation platform for foundation models. https: //github.com/open-compass/opencompass.

Panupong Pasupat and Percy Liang. 2015. Compositional semantic parsing on semi-structured tables. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing ofthe Asian Federation ofNatural Language Processing, ACL 2015, July 26-31, 2015, Beijing, China, Volume 1: Long Papers, pages 1470– 1480. The Association for Computer Linguistics.

Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. 2023. The refinedweb dataset for falcon LLM: outperforming curated corpora with web data, and web data only. CoRR, abs/2306.01116.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2023. Yarn: Efficient context window extension of large language models. CoRR, abs/2309.00071.

Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton-Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. 2023. Code llama: Open foundation models for code. CoRR, abs/2308.12950.

Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant, and Omer Levy. 2023. Zeroscrolls: A zero-shot benchmark for long text understanding. CoRR, abs/2305.14196.

Tianze Shi, Chen Zhao, Jordan L. Boyd-Graber, Hal Daumé III, and Lillian Lee. 2020. On the potential of lexico-logical alignments for semantic parsing to SQL queries. In Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020, volume EMNLP 2020 of Findings ofACL, pages 1849–1864. Association for Computational Linguistics.

Simeng Sun, Kalpesh Krishna, Andrew Mattarella-Micke, and Mohit Iyyer. 2021. Do long-range language models actually use long-range context? In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 807–822. Association for Computational Linguistics.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed Chi, Denny Zhou, and Jason Wei. 2023. Challenging big-bench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pages 13003–13051. Association for Computational Linguistics.

InternLM Team. 2023. Internlm: A multilingual language model with progressively enhanced capabilities. https://github.com/InternLM/InternLM.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. 2023. Agieval: A human-centric benchmark for evaluating foundation models. CoRR, abs/2304.06364.

## A Evaluation Experiments Results

## A.1 Other Synthetic Task

S3EVAL is a synthetic task that possesses a certain level of difficulty and robustness, which allows for a good assessment of an LLM’s overall capability compared to previous works. We choose key-value retrieval task (Liu et al., 2023b), given a key, the goal is to return the associated value. We test several LLMs on this task, and the experiments results are shown in Figure 10. It demonstrates that keyvalue retrieval task is a simple task which has low correlation with real LLMs reasoning benchmark. S3EVAL, as a complex and robust benchmark, can provide reference for future synthetic data.

![](images/fb02bd049c861ce524632842654fcab638c156fea0e2aa93d874b8788f955f72.jpg)  
Figure 10: Performance analysis of key-value retrieval task and BBH.

## A.2 Overall Performance

The detail performance are shown in Table 4.

## A.3 Reliability Experiments

Symbolic Tasks vs. Natural Language Tasks. Another point to prove is that symbolic tasks are consistent with their natural language counterparts. SQL execution is a suitable task because SQL can be intertranslated with an natural question. As can be seen from the “WTQ” column of the Table 4 and Figure 11a, LLM’s ability to execute SQL is consistent with its table question answering ability.

Synthetic data vs. Real data. We want to verify if the synthesized SQL is simpler. The tables “SQLgeneral” and “WTQ-SQL” show the difference in performance between the model on synthetic and real data. We keep the average length of the tables similar, and the experimental results show that the synthetic SQL is more complex than the real SQL.

And Figure 11c shows that, the performance of LLMs on real tables and synthetic tables is very relevant.

Different S3EVAL Settings. As shown in Figure 11b, even if the data settings are very different, LLMs are guaranteed a consistent performance ranking on S3EVAL.

## A.4 Other SQL Prompting Styles

SQL execution task with Chain-of-Thought prompting. SQL is a complex multi-step reasoning task. To verify whether it is a reliable reasoning task, S3EVAL generates multi-step execution instructions for SQL. ChatGPT’s performance (markdown) improves from 38.0 to 48.5 when using chain-of-thougnt prompts. The chainof-thought examples are shown in below. The examples of chain-of-thought prompting are shown in Appendix C.7.

SQL multi-step instruction experiments. SQL multi-step instruction is an auxiliary task. We generate two new datasets using different settings than Easy and General, named Data1 and Data2. Experiments results are shown in Table 6.

## A.5 Long-Context Experiments

Context windows limit the long-context capabilities of LLMs. Previous researchers have proposed many ways to extend the length of context windows, often to 64K, 128K and so on. Existing benchmarks (Bai et al., 2023b; An et al., 2023) collect data from existing NLP communities (which causes data leakage), and more importantly because collecting large amounts of data is difficult. S3EVAL, on the other hand, is easy to collect data with variety and complexity. Existing benchmarks also can’t effectively evaluate very long texts, but S3EVAL can evaluate arbitrary lengths.

YaRN (Peng et al., 2023) extend LLaMA2 context windows to 128K, however, they only evaluated the model’s perplexity, which we believe is not a true reflection of its long-context understanding capability. So we use S3EVAL to generate table data of different lengths and keep all parameters same to evaluate the performance of yarn-LLaMA2, and the experimental results are shown in Table 5. It shows that, yarn-llama2 has a noticeable dip in performance on 20K-80K, which is good for a small number of tasks as well. But compared to ChatGPT (which we can only test 16K length tables), there’s a noticeable gap.

<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="2">Synthetic Task</td><td colspan="2">Realistic Benchmark</td></tr><tr><td>S3EVAL-Easy</td><td>S3EVAL-General</td><td>WTQ</td><td>Reasoning Task</td></tr><tr><td rowspan="20">LLM</td><td>GPT-4</td><td>99.4</td><td>63.1</td><td>70.8</td><td>86.7</td></tr><tr><td>ChatGPT</td><td>97.0</td><td>47.2</td><td>62.0</td><td>70.1</td></tr><tr><td>Claude-1</td><td>98.2</td><td>44.3</td><td>63.4</td><td>67.3</td></tr><tr><td>Llama-2-70B</td><td>94.2</td><td>41.3</td><td>55.9</td><td>64.9</td></tr><tr><td>Mistral-7B</td><td>87.4</td><td>34.3</td><td>55.7</td><td>53.7</td></tr><tr><td>Llama2-13B</td><td>75.0</td><td>30.9</td><td>49.2</td><td>45.6</td></tr><tr><td>InternLM-20B</td><td>78.0</td><td>32.3</td><td>49.4</td><td>52.5</td></tr><tr><td>Qwen-14B Llama-2-7B</td><td>71.8</td><td>25.8 23.8</td><td>46.7</td><td>53.7</td></tr><tr><td>Qwen-7B</td><td>54.2 56.4</td><td>19.4</td><td>40.6</td><td>38.2</td></tr><tr><td></td><td>55.2</td><td></td><td>41.2</td><td>45.2</td></tr><tr><td>Xgen-7B</td><td>41.6</td><td>24.6 18.5</td><td>36.3 27.5</td><td>34.5</td></tr><tr><td>Internlm-7B Phi-1_5</td><td>27.6</td><td>16.1</td><td>22.1</td><td>37.0</td></tr><tr><td>Stablelm-7B</td><td>6.0</td><td>4.2</td><td>14.7</td><td>30.0</td></tr><tr><td>Stablelm-3B</td><td>4.2</td><td>2.9</td><td>11.2</td><td>24.3</td></tr><tr><td>Pythia-12B</td><td>31.4</td><td>17.3</td><td>24.5</td><td>21.0</td></tr><tr><td></td><td>25.2</td><td></td><td></td><td>29.3</td></tr><tr><td>Pythia-6.9B</td><td></td><td>16.0</td><td>22.6 21.7</td><td>28.6</td></tr><tr><td>Pythia-2.8B Pythia-1B</td><td>26.4</td><td>14.6</td><td></td><td>28.8</td></tr><tr><td></td><td>8.4</td><td>7.1</td><td>16.2</td><td>25.6</td></tr><tr><td rowspan="11">Code LLM</td><td>CodeLlama-34B</td><td>91.4</td><td>41.0</td><td>53.9</td><td>36.4</td></tr><tr><td>CodeLlama-13B</td><td>90.0</td><td>35.7</td><td>49.9</td><td>30.6</td></tr><tr><td>CodeLlama-7B</td><td>75.2</td><td>34.2</td><td>44.9</td><td>26.3</td></tr><tr><td>StarCoder-15B</td><td>87.2</td><td>34.4</td><td>39.2</td><td>30.4</td></tr><tr><td>StarCoder-7B</td><td>88.4</td><td>32.4</td><td>33.3</td><td>28.3</td></tr><tr><td>StarCoder-3B</td><td>79.0</td><td>28.0</td><td>27.5</td><td>21.5</td></tr><tr><td>StarCoder-1B</td><td>37.4</td><td>15.4</td><td>21.1</td><td>15.2</td></tr><tr><td>CodeGen-15B</td><td>36.8</td><td>18.2</td><td>25.0</td><td>18.3</td></tr><tr><td>CodeGen-6B</td><td>25.0</td><td>16.9</td><td>17.8</td><td>18.2</td></tr><tr><td>CodeGen-2B</td><td>31.4</td><td>16.6</td><td>20.8</td><td>14.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 4: SQL Execution Task Performance on different LLMs.

## B Controllable Analysis Results

## B.1 Answer Position Analysis

In addition to the figures in the main text, we also conduct experiments with row level. We use two methods to visualize the results. (1) Sliding windows (Figure 12a,12b). We choose windows=5 and smooth the data to make a dot plot and a trend line. (2) Grouping calculations (Figure 12c,12d). Group neighboring rows together with the granularity of 5, 10, and 20. For example, if granularity is 20, then we group the rows with answers located in 1-20, 20-40, 40-60, 60-80, and 80-100, for a total of five groups, and calculate the average scores.

## B.2 Template Controlled Analysis

Each data template in S3EVAL includes corresponding reasoning types, and thus it provides fine-grained control over the evaluation examples. To stimulate new insights and uncover counterintuitive performance phenomena of LLMs, we present several controlled analysis examples using simple templates as a starting point.

Template1: SELECT [text\_col1] FROM table

## WHERE ([text\_col2] = [text2])

We first explore the relationship between the model performance and the locations of [text\_col1] and [text\_col2]. To begin with, we generated a set of 10  15 tables, each comprising 15 distinct columns. We created 400 unique combinations by pairing each value in text\_col1 with each value in text\_col2. For each of the 400 pairs, we generated 40 evaluation examples, resulting in a total of 16,000 evaluation examples. After SQL execution experiments, we calculated the scores of each pair and constructed a heatmap, which is illustrated in Figure 17. The heatmap indicates that the performance is overall better when [text\_col1] is the previous column. And the model performance is also better when the [text\_col1] column is before [text\_col2] column. It indicates that the model tends to focus on the beginning of a specific paragraph. Moreover, in multi-hop reasoning, LLMs excel at hopping to the context preceding a intermediate hop, but struggles when it comes to searching backward.

Template2: SELECT [text\_col1] FROM table WHERE ([text\_col2] = [text2])  N

![](images/57d6d415e5c16338d9d246ec8e6875fb4642c818302f5db23572b2e212743373.jpg)  
(a) Correlation between QA task and SQL execution Task on WikiTableQuestions.

![](images/f34c5ae38e0e808082efef48c0918005e5679eede64d5cb68806804ac04655da.jpg)  
(b) Correlation between General and Easy Settings.

![](images/1a449cdb10979099ef3a562fedebb76ca48fdfe0d5ac2ae7b2aa05f9fc589ebf.jpg)  
(c) Correlation between Synthetic and Real Table SQL execution task

Figure 11: Experimental results of the correlation experiments.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Max-Ctx</td><td colspan="8">SQL Execution</td></tr><tr><td>2K</td><td>4K</td><td>8K</td><td>16K</td><td>20K</td><td>40K</td><td>60K</td><td>80K</td></tr><tr><td>ChatGPT</td><td>16k</td><td>96.8</td><td>95.2</td><td>80.3</td><td>68.7</td><td>一</td><td>-</td><td>一</td><td>一</td></tr><tr><td>Claude-1.3-100K</td><td>128k</td><td>97.2</td><td>96.8</td><td>91.8</td><td>85.2</td><td></td><td></td><td></td><td></td></tr><tr><td>Yarn-LLaMA2-13B</td><td>128k</td><td>76.3</td><td>57.0</td><td>40.6</td><td>25.1</td><td>20.6</td><td>17.6</td><td>17.0</td><td>12.0</td></tr><tr><td>XGen-7B</td><td>8k</td><td>51.6</td><td>41.8</td><td>25.4</td><td></td><td>一</td><td></td><td>一</td><td></td></tr><tr><td>LongChat-13B</td><td>16k</td><td>48.6</td><td>39.0</td><td>26.3</td><td>19.5</td><td></td><td></td><td></td><td></td></tr><tr><td>LongLlaMA-7B</td><td>256k</td><td>82.4</td><td>62.8</td><td>24.4</td><td>一</td><td>一</td><td></td><td></td><td></td></tr><tr><td>RWKV-Raven-14B</td><td>128k</td><td>10.5</td><td>7.4</td><td>6.2</td><td></td><td>一</td><td></td><td></td><td></td></tr></table>

Table 5: Long-Context experiments on S3EVAL.

We then investigate the impact of the number of WHERE conditions on LLM performance. Intuitively, more conditions should make it harder for LLM to execute SQL since the instruction becomes more complex. However, the experimental results contradict this intuition, as shown in blue in Figure 15. We speculate that this counter-intuitive result stems from how LLMs actually reason: by looking up string co-occurrences rather than logically considering all conditions.

Template3: SELECT COUNT([text\_col]) FROM table WHERE [text\_col] = [text] .

We analyze the counting ability of LLMs, which is an important numerical reasoning capability. To avoid potential symbolic effects of SQLs, we also use the instruction style (Section 2.1) to prompt the model (e.g. Please count the number of “[text\_col] is [text]”). As shown in Figure 16, whether it is zero-shot or few-shot, SQL style or instruction style, the performance of LLMs is best when the COUNT value is the smallest or the largest. When the COUNT value is in the middle, the performance of the model is almost zero.

In the future, developers can employ the

S3EVAL suite to analyze the performance of LLMs with various complex SQL queries and discover new insights. They can also investigate more on the multi-step instruction prompting (Section C.6) and chain-of-thought prompting (Section C.7) to better understand LLMs.

## B.3 Input Format Analysis

In this section, we focus on comparing two formats of inputting tables, namely markdown andflatten, to explore their impact on LLMs performance. Figure 13 clearly demonstrates a significant improvement in the model’s performance when the flatten format is used instead of the markdown input format at any experiments settings.

The reason behind this improvement lies in the structure of the SQL template, specifically “select <col1> where <col2> <op> <int2>”. In order to execute this template, the model needs to locate the column corresponding to col2 and then identify the row where “int2” is found. This process involves 2-hop reasoning. In markdown mode, the challenge lies not only in the LLM’s understanding of the table structure but also in how to navigate to another column in the same row. However, in flatten mode, redundant columns are added to each row as “Column is value.” This additional information simplifies the LLM’s understanding of the table structure and facilitates reasoning. As a result, the flatten method proves to be more beneficial for LLM performance due to its enhanced structure comprehension and reasoning capabilities.

<table><tr><td rowspan="3">Model</td><td colspan="4">SQL Execution</td><td colspan="4">SQL Multi-Step Instruction</td></tr><tr><td colspan="2">Zero-Shot</td><td colspan="2">Few-Shot</td><td colspan="2">Zero-Shot</td><td colspan="2">Few-Shot</td></tr><tr><td>Data1</td><td>Data2</td><td>Data1</td><td>Data2</td><td>Data1</td><td>Data2</td><td>Data1</td><td>Data2</td></tr><tr><td>ChatGPT</td><td>96.4</td><td>47.0</td><td>97.0</td><td>49.0</td><td>97.9</td><td>30.0</td><td>98.8</td><td>34.8</td></tr><tr><td>Codellama-13B</td><td>71.2</td><td>34.3</td><td>90.0</td><td>39.8</td><td>63.9</td><td>12.1</td><td>88.0</td><td>22.8</td></tr><tr><td>StarCoder-15B</td><td>52.3</td><td>24.7</td><td>85.8</td><td>37.6</td><td>44.4</td><td>14.4</td><td>84.2</td><td>19.2</td></tr><tr><td>InternLM-20B</td><td>60.4</td><td>22.7</td><td>78.0</td><td>35.0</td><td>58.8</td><td>14.9</td><td>76.6</td><td>28.1</td></tr><tr><td>InternLM-20B-Chat</td><td>71.2</td><td>31.3</td><td>78.0</td><td>34.2</td><td>67.6</td><td>21.9</td><td>74.4</td><td>25.4</td></tr><tr><td>LLaMA2-13B</td><td>68.1</td><td>23.2</td><td>75.0</td><td>32.3</td><td>50.5</td><td>5.4</td><td>74.6</td><td>18.2</td></tr><tr><td>LLaMA2-13B-Chat</td><td>51.6</td><td>16.4</td><td>71.5</td><td>28.3</td><td>9.4</td><td>1.0</td><td>64.2</td><td>21.1</td></tr><tr><td>Vicuna-13B</td><td>57.6</td><td>26.8</td><td>81.6</td><td>35.4</td><td>48.9</td><td>11.5</td><td>78.8</td><td>24.2</td></tr></table>

Table 6: SQL Multi-Step Task performance on different LLMs.

## B.4 SQL Keywords Analysis

SQL statements follow a specific syntax and are a well-established language in the database domain. We first control SQL statements to contain only specific types of keywords from the perspective of SQL keywords and test the performance of different models on S3EVAL. The experimental results are shown in Figure 14. The change in the performance of LLMs on SQL statements reflects the trend in the difficulty of reasoning.

## B.5 SQL Attribute Analysis

S3EVAL has the ability to flexibly modify the properties of generated SQL statements, including the length of the statement, the number of computations, and the quantity of filtering numbers. These features can intuitively impact the complexity of SQL. In our experiments, we set the table size to 15 10 and adjusted the SQL settings for examining the effect of different SQL attributes on model performance. For example, in the analysis of "Calculation Times," we employed 500 samples with 0, 1, 2, and 3 calculation times respectively. The experimental outcomes of all SQL attributes are illustrated in Figure 18. While it might be expected that model performance would decline as these factor values increase, the performance actually fluctuates. Upon combining Column number, Row number, Calculation times, and Filter times in the statistical analysis, we identified a significant downward trend in the model, as demonstrated in Figure 18f.

Experiment of Different Keyword Setting  
![](images/f54af635b1d675682bb3216c3602167fb634d062c5df9c30818e75e651a6fb88.jpg)  
(a)

![](images/4ca715519ed81ce34ea2e1a68487084bdf2ecd68de725f97f65716405bf7dac6.jpg)  
(b)

![](images/e4c03784b1e2e49c1f9f97599e5fd827e2092681390ad1da9dfaae343fc93018.jpg)  
(c)

![](images/f727137495b1de470fb1c97dc12050525b8b407d76c245c08cbf9b818112caa8.jpg)  
(d)  
Figure 12: Effect of answer position on model performance. We use two methods to visualize the results. (1) Sliding windows (Figure 12a,12b). We select a window size of 5 and smooth the data to make a dot plot and a trend line. (2) Grouping calculations (Figure 12c,12d). We group neighboring rows with granularities of 5, 10, and 20. For instance, with a granularity of 20, we group rows with answers located in the ranges 1-20, 21-40, 41-60, 61-80, and 81-100, resulting in five groups, and compute the average scores.

![](images/f33e914a8cc634c4d01d390418b8f5a18183d3f2c71aae0f6ad8424397bb50f4.jpg)  
Figure 13: Different input format.

![](images/29e58d7c7f1ccdc2d604496869b2e32af3c9f2585ef3b57153e0464027aacbd8.jpg)  
Figure 14: Different keywords setting.

Number of conditions  
![](images/ca26a5db57de718dadef8eecd326d2fbb8c1cece365fd7b7aa0ab55c13a1daeb.jpg)  
Figure 15: Trend of ChatGPT performance with where condition number using Template2.

![](images/2af2a7a767e1196eca2b9d3e7515998f8ee343951b57455fdf2f795e259b155f.jpg)  
Figure 16: Trend of ChatGPT performance with the COUNT value in Template3. Only when the COUNT value is the largest or smallest, the model have good performance.

![](images/a86080e974f8a022b7cbd8de5abe7e5ef7a7ca8dd187a207e7399ee83473ea9d.jpg)  
Figure 17: ChatGPT performance with different locations of [text\_col1] and [text\_col2]. The performance improves when the example has the location of [text\_col1] before [text\_col2].

![](images/b3672e3b7b1584aaefd2bd7836d8cdf798c72717055bdab720a31f812c011600.jpg)  
(a) SQL length

![](images/2c49818c3640babceca60787b4fd994b1e5e7459dd38d1621981ad4766b170a5.jpg)  
(b) Column number.

![](images/5eac9c1c4b9f7a2ec319440363f36299a880d112811589673618b634e95464e7.jpg)  
(c) Selected row number

![](images/63c68a1109c4c1f0ac9fe0bcc868f21bf65c7ff999807d98168d0b48d2742d12.jpg)  
(d) Calculation times

![](images/8fd2df328ac701fba5456dd90df428d909e149b1609aa8b7c21de247bd377e72.jpg)  
(e) Filter times.

![](images/154b0e5319856635c56710377b93eee51c914a8804fc4bf18c6e05833946a177.jpg)  
(f) Total analysis.  
Figure 18: Effect of SQL Attribute Settings on model performance.

## C Data Demostration

## C.1 Dense and Sparse Examples

SQL: select boarfish from w where sixties = ’jcrbb

Answer: [’qxgd’, ’lorfaljob’, ’qytocp’, ’vkfzhqwj’, ’xwijyubr’]

We can find that Dense Setting is better than Sparse Setting in all cases.

## Sparse Example:

<table><tr><td>一</td><td>boarfish</td><td>tool</td><td>sixties</td><td>phoxinus</td><td>angling</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>-1</td></tr><tr><td>0</td><td>mjdsv</td><td>cwzqkdte</td><td>tbwqa</td><td>yuogpbo</td><td>mkxqnrhq</td><td></td></tr><tr><td>1</td><td>nrbmyc</td><td>eqciiims</td><td>wvfesrtzt</td><td>yvvgzj</td><td>mkxqnrhq</td><td></td></tr><tr><td>2</td><td>iqdr</td><td>ezhuj</td><td>bndktpe</td><td>yuogpbo</td><td>yjblg</td><td>一</td></tr><tr><td>3</td><td>qxgd</td><td>dtfjqfc</td><td>jcrbb</td><td>haxyaz</td><td>yjblg</td><td>一</td></tr><tr><td>_ 4</td><td>xzrrs</td><td>ezhuj</td><td>bndktpe</td><td>dpimlb</td><td>skbpzyhak</td><td>1</td></tr><tr><td>5</td><td>lorfaljob</td><td>eqciiims</td><td>jcrbb</td><td>jsvbugac</td><td>bwxihx</td><td></td></tr><tr><td>6</td><td>pvugxgdju</td><td>dtfjqfc</td><td>bndktpe</td><td>jsvbugac</td><td>mkxqnrhq</td><td></td></tr><tr><td>7</td><td>xpkuautv</td><td>ezhuj</td><td>vyoo</td><td>yvvgzj</td><td>bwxihx</td><td></td></tr><tr><td>8</td><td>afzrom</td><td>jzdra</td><td>bndktpe</td><td>jsvbugac</td><td>mkxqnrhq</td><td></td></tr><tr><td>9</td><td>ivxpmv</td><td>eqciiims</td><td>bndktpe</td><td>jsvbugac</td><td>bwxihx</td><td>一</td></tr><tr><td>10</td><td>ehfvur</td><td>ezhuj</td><td>tbwqa</td><td>yuogpbo</td><td>bwxihx</td><td>一</td></tr><tr><td>11</td><td>bdzsy</td><td>ezhuj</td><td>bndktpe</td><td>yvvgzj</td><td>yjblg</td><td>一</td></tr><tr><td>12</td><td>qruh</td><td>ezhuj</td><td>bndktpe</td><td>dpimlb</td><td>skbpzyhak</td><td>1</td></tr><tr><td>13</td><td>qytocp</td><td>jzdra</td><td>jcrbb</td><td>dpimlb</td><td>bwxihx</td><td></td></tr><tr><td>14</td><td>eqaja</td><td>ezhuj</td><td>bndktpe</td><td>haxyaz</td><td>yjblg</td><td>一</td></tr><tr><td>15</td><td>kwvzixe</td><td>jzdra</td><td>vyoo</td><td>jsvbugac</td><td>skbpzyhak</td><td>V</td></tr><tr><td>16</td><td>edmkxm</td><td>eqciiims</td><td>vyoo</td><td>haxyaz</td><td>mkxqnrhq</td><td></td></tr><tr><td>17</td><td>fdsdlcpxj</td><td>eqciiims</td><td>vyoo</td><td>dpimlb</td><td>blqoislm</td><td></td></tr><tr><td>18</td><td>ipprxzzlv</td><td>cwzqkdte</td><td>bndktpe</td><td>yuogpbo</td><td>yjblg</td><td>_</td></tr><tr><td>19</td><td>gqyxjtbz</td><td>eqciiims</td><td>tbwqa</td><td>dpimlb</td><td>yjblg</td><td>一</td></tr><tr><td>20</td><td>noqfw</td><td>ezhuj</td><td>vyoo</td><td>haxyaz</td><td>blqoislm</td><td></td></tr><tr><td>21</td><td>vkfzhqwj</td><td>dtfjqfc</td><td>jcrbb</td><td>yuogpbo</td><td>mkxqnrhq</td><td></td></tr><tr><td>22</td><td>konftq</td><td>eqciiims</td><td>vyoo</td><td>dpimlb</td><td>bwxihx</td><td></td></tr><tr><td>23</td><td>ymcwhu</td><td>jzdra</td><td>wvfesrtzt</td><td>dpimlb</td><td>blqoislm</td><td></td></tr><tr><td>24</td><td>kpygsu</td><td>eqciiims</td><td>wvfesrtzt</td><td>yuogpbo</td><td>yjblg</td><td>_</td></tr><tr><td>25</td><td>tiwfvqgmt</td><td>ezhuj</td><td>bndktpe</td><td>dpimlb</td><td>mkxqnrhq</td><td></td></tr><tr><td>26</td><td>ovomhf</td><td>dtfjqfc</td><td>bndktpe</td><td>yuogpbo</td><td>blqoislm</td><td></td></tr><tr><td>27</td><td>lokwxn</td><td>cwzqkdte</td><td>tbwqa</td><td>yuogpbo</td><td>mkxqnrhq</td><td></td></tr><tr><td>28</td><td>xwijyubr</td><td>jzdra</td><td>jcrbb</td><td>yuogpbo</td><td>mkxqnrhq</td><td></td></tr><tr><td>29</td><td>ttonww</td><td>dtfjqfc</td><td>wvfesrtzt</td><td>haxyaz</td><td>blqoislm</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## Dense Example:

| boarfish 0 | mjdsv 1 | nrbmyc 2 | iqdr 3 | xzrrs 4 | pvugxgdju 5 | xpkuautv 6 | afzrom 7 | ivxpmv 8 | ehfvur | 9 | bdzsy 10 | qruh | 11 | eqaja | 12 | kwvzixe | 13 | qxgd | 14 | lorfaljob | 15 | qytocp | 16 | vkfzhqwj | 17 | xwijyubr | 18 | edmkxm | 19 | fdsdlcpxj | 20 | ipprxzzlv | 21 | gqyxjtbz | 22 | noqfw

cwzqkdte eqciiims ezhuj ezhuj dtfjqfc ezhuj jzdra eqciiims ezhuj ezhuj ezhuj ezhuj jzdra dtfjqfc eqciiims jzdra dtfjqfc jzdra eqciiims eqciiims cwzqkdte eqciiims ezhuj

sixties   
tbwqa   
wvfesrtzt   
bndktpe   
bndktpe   
bndktpe   
vyoo   
bndktpe   
bndktpe   
tbwqa   
bndktpe   
bndktpe   
bndktpe   
vyoo   
jcrbb   
jcrbb   
jcrbb   
jcrbb   
jcrbb   
vyoo   
vyoo   
bndktpe   
tbwqa   
vyoo

phoxinus yuogpbo yvvgzj yuogpbo dpimlb jsvbugac yvvgzj jsvbugac jsvbugac yuogpbo yvvgzj dpimlb haxyaz jsvbugac haxyaz jsvbugac dpimlb yuogpbo yuogpbo haxyaz dpimlb yuogpbo dpimlb haxyaz

angling   
mkxqnrhq   
mkxqnrhq   
yjblg   
skbpzyhak   
mkxqnrhq   
bwxihx   
mkxqnrhq   
bwxihx   
bwxihx   
yjblg   
skbpzyhak   
yjblg |   
skbpzyhak   
yjblg |   
bwxihx   
bwxihx   
mkxqnrhq   
mkxqnrhq   
mkxqnrhq   
blqoislm   
yjblg   
yjblg   
blqoislm |

```typescript
| 23 | konftq | eqciiims | vyoo | dpimlb | bwxihx
| 24 | ymcwhu | jzdra | wvfesrtzt | dpimlb | blqoislm
| 25 | kpygsu | eqciiims | wvfesrtzt | yuogpbo | yjblg
| 26 | tiwfvqgmt | ezhuj | bndktpe | dpimlb | mkxqnrhq |
| 27 | ovomhf | dtfjqfc | bndktpe | yuogpbo | blqoislm |
| 28 | lokwxn | cwzqkdte | tbwqa | yuogpbo | mkxqnrhq |
| 29 | ttonww | dtfjqfc | wvfesrtzt | haxyaz | blqoislm |
```

## C.2 SQL Template

## General:

select <select\_condition> from my\_table

select <select\_condition> from my\_table <where\_condition>

select <select\_condition> from my\_table <order\_condition>,

select <select\_condition> from my\_table <where\_condition> <order\_condition>,

select <select\_condition> from my\_table <group\_condition> <having\_condition>,

select <select\_condition> from my\_table <where\_condition> <group\_condition> <having\_condition>,

```sql
select <select_condition> from my_table <where_condition>
<group_condition> <having_condition> <order_condition>,
```

select <select\_condition> from my\_table <group\_condition> <having\_condition> <order\_condition>

## Where Condition:

```sql
select <text_col1> from my_table where <text_col2> = <text_2>
```

## Count:

Select Count(<text\_col1>) from table where <text\_col1> = <text\_1>

## Easy:

```sql
select <text_col1> from my_table where <int_col1> = <int_1>
select <int_col1> from my_table where <text_col1> = <text_1>
select <int_col1> from my_table where <int_col2> = <int_2>
select <text_col1> from my_table where <text_col2> = <text_2>
```

## Filter:

```sql
select <text_col1> from my_table where <text_col2> = <text_2>
select <text_col1> from my_table where <int_col2> <op2> <int_2>
select <text_col1> from my_table where <text_col2> = <text_2> and <int_col1> <op1> <int_1>
select <text_col1> from my_table where <text_col2> = <text_2> and <text_col3> = <text_3>
select <text_col1> from my_table where <int_col1> <op1> <int_1> and <int_col2> <op2> <int_2>
select <int_col1> from my_table where <text_col1> = <text_1>
select <int_col1> from my_table where <int_col2> <op2> <int_2>
select <int_col1> from my_table where <text_col2> = <text_2> and <int_col2> <op2> <int_2>
select <int_col1> from my_table where <text_col2> = <text_2> and <text_col3> = <text_3>
select <int_col1> from my_table where <int_col2> <op2> <int_2> and <int_col3> <op3> <int_3>
```

## Aggregate:

```sql
select count ( <text_col1> ) from my_table where <text_col2> = <text_2>
select count ( <text_col1> ) from my_table where <int_col2> <op2> <int_2>
select sum ( <int_col1> ) from my_table
select sum ( <int_col1> ) from my_table where <text_col2> = <text_2>
select max ( <int_col1> ) from my_table
select max ( <int_col1> ) from my_table where <text_col2> = <text_2>
select min ( <int_col1> ) from my_table
select min ( <int_col1> ) from my_table where <text_col2> = <text_2>
```

## Arithmetic:

```sql
select <int_col1> + <int_col2> from my_table where <text_col1> = <text_1>
select <int_col1> + <int_col2> from my_table where <text_col1> = <text_1> and <text_col2> = <text_2>
select <int_col1> - <int_col2> from my_table where <text_col1> = <text_1>
select <int_col1> - <int_col2> from my_table where <text_col1> = <text_1> and <text_col2> = <text_2>
```

## Superlative:

select <int\_col1> from my\_table order by <int\_col1> asc limit 1   
select <int\_col1> from my\_table order by <int\_col1> desc limit 1   
select <text\_col1> from my\_table order by <int\_col1> asc limit 1   
select <text\_col1> from my\_table order by <int\_col1> desc limit 1   
select <int\_col1> from my\_table order by <int\_col2> asc limit 1   
select <int\_col1> from my\_table order by <int\_col2> desc limit 1

## Comparative:

select ( select <int\_col1> from my\_table where <text\_col1> = <text\_1> ) > ( select <int\_col1> from my\_table where <text\_col2> = <text\_2> ) select ( select <int\_col1> from my\_table where <int\_col2> <op2> <int\_2> ) > ( select <int\_col1> from my\_table where <int\_col3> <op3> <int\_3> ) select ( select <int\_col1> from my\_table where <text\_col1> = <text\_1> ) < ( select <int\_col1> from my\_table where <text\_col2> = <text\_2> ) select ( select <int\_col1> from my\_table where <int\_col2> <op2> <int\_2> ) < ( select <int\_col1> from my\_table where <int\_col3> <op3> <int\_3> ) select <int\_col1> > <int\_col2> from my\_table where <text\_col1> = <text\_1> select <int\_col1> < <int\_col2> from my\_table where <text\_col1> = <text\_1> select <int\_col1> > <int\_col2> from my\_table where <int\_col3> <op3> <int\_3> select <int\_col1> < <int\_col2> from my\_table where <int\_col3> <op3> <int\_3>

## C.3 Table Input Format

## Markdown Table:

<table><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>ercilla |</td><td rowspan=1 colspan=1>shucks</td><td rowspan=1 colspan=1> liter    |</td><td rowspan=1 colspan=1>taenia</td><td rowspan=1 colspan=2>dorado</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>gcrdvo</td><td rowspan=1 colspan=1>qoath</td><td rowspan=1 colspan=2>katfuw</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>129</td><td rowspan=1 colspan=1>151</td><td rowspan=1 colspan=1>zmvltkk</td><td rowspan=1 colspan=1>jpcglcjzk</td><td rowspan=1 colspan=2>vwqqey    一</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>248</td><td rowspan=1 colspan=1>188</td><td rowspan=1 colspan=1>zmdlfbhb</td><td rowspan=1 colspan=1>cvhqotsys</td><td rowspan=1 colspan=2>wzunmaa</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>267</td><td rowspan=1 colspan=1>104</td><td rowspan=1 colspan=1>gcrdvo</td><td rowspan=1 colspan=1>ytywunvf</td><td rowspan=1 colspan=2>pjlbo</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>135</td><td rowspan=1 colspan=1>262</td><td rowspan=1 colspan=1>gcrdvo</td><td rowspan=1 colspan=1>dtnvfp</td><td rowspan=1 colspan=2>ajzpsaoy  一</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>309</td><td rowspan=1 colspan=1>119</td><td rowspan=1 colspan=1>zmdlfbhb</td><td rowspan=1 colspan=1>klcenmugk</td><td rowspan=1 colspan=2>hriunhf   一</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>251</td><td rowspan=1 colspan=1>152</td><td rowspan=1 colspan=1>zmvltkk</td><td rowspan=1 colspan=1>cjgcergv</td><td rowspan=1 colspan=2>shrbvrd   一</td></tr><tr><td rowspan=1 colspan=1>71</td><td rowspan=1 colspan=1>298</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>zmvltkk</td><td rowspan=1 colspan=1>scvuuc</td><td rowspan=1 colspan=2>ahunvcx   一</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>321</td><td rowspan=1 colspan=1>217</td><td rowspan=1 colspan=1>gcrdvo</td><td rowspan=1 colspan=1>ezlp</td><td rowspan=1 colspan=2>hasjaznm 一</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>1391</td><td rowspan=1 colspan=1>310</td><td rowspan=1 colspan=1>gcrdvo</td><td rowspan=1 colspan=1>ghhjea</td><td rowspan=1 colspan=2>atqvtgoa  一</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>99</td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1>zmvltkk</td><td rowspan=1 colspan=1>ecdmpruq</td><td rowspan=1 colspan=2>cfitvz    _</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>1421</td><td rowspan=1 colspan=1>167</td><td rowspan=1 colspan=1>gcrdvo</td><td rowspan=1 colspan=1>acii</td><td rowspan=1 colspan=2>oenmuezip</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>2731</td><td rowspan=1 colspan=1>156</td><td rowspan=1 colspan=1>gcrdvo</td><td rowspan=1 colspan=1>nnvnteh</td><td rowspan=1 colspan=2>tulh</td></tr><tr><td rowspan=1 colspan=1>131</td><td rowspan=1 colspan=1>197</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>gcrdvo</td><td rowspan=1 colspan=1>pqdbhevkh</td><td rowspan=1 colspan=2>dfxuwxz   一</td></tr><tr><td rowspan=1 colspan=1>14 </td><td rowspan=1 colspan=1>144 </td><td rowspan=1 colspan=1>123</td><td rowspan=1 colspan=1>gcrdvo</td><td rowspan=1 colspan=1>bxrgo</td><td rowspan=1 colspan=2>ccbj       一</td></tr></table>

## Flatten Table:

## Flatten Table Examples:

The table have 5 columns: ercilla | shucks | liter | taenia | dorado row 1 : ercilla is 68. shucks is 12. liter is gcrdvo. taenia is qoath. dorado is katfuw.   
row 2 : ercilla is 129. shucks is 151. liter is zmvltkk. taenia is jpcglcjzk. dorado is vwqqey.   
row 3 : ercilla is 248. shucks is 188. liter is zmdlfbhb. taenia is cvhqotsys. dorado is wzunmaa.   
row 4 : ercilla is 267. shucks is 104. liter is gcrdvo. taenia is ytywunvf. dorado is pjlbo.   
row 5 : ercilla is 135. shucks is 262. liter is gcrdvo. taenia is dtnvfp. dorado is ajzpsaoy.   
row 6 : ercilla is 309. shucks is 119. liter is zmdlfbhb. taenia is klcenmugk. dorado is hriunhf.   
row 7 : ercilla is 25. shucks is 152. liter is zmvltkk. taenia is cjgcergv. dorado is shrbvrd.   
row 8 : ercilla is 298. shucks is 18. liter is zmvltkk. taenia is scvuuc. dorado is ahunvcx.   
row 9 : ercilla is 321. shucks is 217. liter is gcrdvo. taenia is ezlp. dorado is hasjaznm.   
row 10 : ercilla is 139. shucks is 310. liter is gcrdvo. taenia is ghhjea. dorado is atqvtgoa.   
row 11 : ercilla is 99. shucks is 34. liter is zmvltkk. taenia is ecdmpruq. dorado is cfitvz.   
row 12 : ercilla is 142. shucks is 167. liter is gcrdvo. taenia is acii. dorado is oenmuezip.   
row 13 : ercilla is 273. shucks is 156. liter is gcrdvo. taenia is nnvnteh. dorado is tulh.   
row 14 : ercilla is 197. shucks is 44. liter is gcrdvo. taenia is pqdbhevkh. dorado is dfxuwxz.   
row 15 : ercilla is 144. shucks is 123. liter is gcrdvo. taenia is bxrgo. dorado is ccbj.

## C.4 SQL Execution Examples (Few-shot)

You are an SQL executor, you need to execute SQL based on the give table and SQL statement to obtain the execution results.

Only give me the execution results and do not output any other words.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>puccoon</td><td rowspan=1 colspan=1>tiepolo</td><td rowspan=1 colspan=1>I   scope</td><td rowspan=1 colspan=1>mutinus</td><td rowspan=1 colspan=1>intrados |</td><td rowspan=1 colspan=1>huggins</td><td rowspan=1 colspan=1>barye</td><td rowspan=1 colspan=1>wear|</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>171</td><td rowspan=1 colspan=1>225</td><td rowspan=1 colspan=1>145</td><td rowspan=1 colspan=1>2007-04-27</td><td rowspan=1 colspan=1>322</td><td rowspan=1 colspan=1>yefihroyn</td><td rowspan=1 colspan=1>79</td><td rowspan=1 colspan=1>207</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>213</td><td rowspan=1 colspan=1>116</td><td rowspan=1 colspan=1>319</td><td rowspan=1 colspan=1>2016-01-15</td><td rowspan=1 colspan=1>288</td><td rowspan=1 colspan=1>ytyayrvj</td><td rowspan=1 colspan=1>246</td><td rowspan=1 colspan=1>2721</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>191</td><td rowspan=1 colspan=1>229</td><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=1>2022-11-08</td><td rowspan=1 colspan=1>218</td><td rowspan=1 colspan=1>gpmvax</td><td rowspan=1 colspan=1>167</td><td rowspan=1 colspan=1>731</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>97</td><td rowspan=1 colspan=1>155</td><td rowspan=1 colspan=1>189</td><td rowspan=1 colspan=1>2013-10-30</td><td rowspan=1 colspan=1>79</td><td rowspan=1 colspan=1>gpmvax</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>233</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>295</td><td rowspan=1 colspan=1>2018-12-10</td><td rowspan=1 colspan=1>81</td><td rowspan=1 colspan=1>yefihroyn</td><td rowspan=1 colspan=1>187</td><td rowspan=1 colspan=1>198</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>285</td><td rowspan=1 colspan=1>304</td><td rowspan=1 colspan=1>168</td><td rowspan=1 colspan=1>2017-03-24</td><td rowspan=1 colspan=1>75</td><td rowspan=1 colspan=1>gpmvax</td><td rowspan=1 colspan=1>111</td><td rowspan=1 colspan=1>77</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>233</td><td rowspan=1 colspan=1>325</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>2014-01-22</td><td rowspan=1 colspan=1>114</td><td rowspan=1 colspan=1>ytyayrvj</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>2191</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>146</td><td rowspan=1 colspan=1>164</td><td rowspan=1 colspan=1>2021-12-07</td><td rowspan=1 colspan=1>311</td><td rowspan=1 colspan=1>ytyayrvj</td><td rowspan=1 colspan=1>188</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>112</td><td rowspan=1 colspan=1>255</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>2015-12-07</td><td rowspan=1 colspan=1>214</td><td rowspan=1 colspan=1>gpmvax</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>271</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>175</td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=1>181</td><td rowspan=1 colspan=1>2012-04-21</td><td rowspan=1 colspan=1>182</td><td rowspan=1 colspan=1>gpmvax</td><td rowspan=1 colspan=1>105</td><td rowspan=1 colspan=1>76</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=1>101</td><td rowspan=1 colspan=1>2008-04-28</td><td rowspan=1 colspan=1>168</td><td rowspan=1 colspan=1>gpmvax</td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1>119</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>180</td><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=1>2004-06-23</td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=1>yefihroyn</td><td rowspan=1 colspan=1>314</td><td rowspan=1 colspan=1>97</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>297</td><td rowspan=1 colspan=1>251</td><td rowspan=1 colspan=1>249</td><td rowspan=1 colspan=1>2022-02-02</td><td rowspan=1 colspan=1>185</td><td rowspan=1 colspan=1>yefihroyn</td><td rowspan=1 colspan=1>278</td><td rowspan=1 colspan=1>3131</td></tr><tr><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>17 </td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1>2016-04-14</td><td rowspan=1 colspan=1>243</td><td rowspan=1 colspan=1>ytyayrvj</td><td rowspan=1 colspan=1>213</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>45</td><td rowspan=1 colspan=1>215</td><td rowspan=1 colspan=1>182</td><td rowspan=1 colspan=1>2012-06-15</td><td rowspan=1 colspan=1>251</td><td rowspan=1 colspan=1>yefihroyn</td><td rowspan=1 colspan=1>221</td><td rowspan=1 colspan=1>83</td></tr></table>

Now you need to execute SQL based on the given table and SQL statement to obtain the execution result. Only give me the result and do not output any other words or SQL statement. The following are some examples.

SQL:select avg ( intrados ) from my\_table where tiepolo > 146 group by huggins   
having count ( huggins ) > 1 order by count ( tiepolo ) asc limit 1   
Answer:146.5   
SQL:select wear from my\_table where huggins = 'gpmvax' group by huggins   
having wear < 83 order by count ( distinct barye ) asc limit 1   
Answer:73   
SQL:select mutinus from my\_table where tiepolo > 116 group by huggins   
having max ( wear ) > 119 order by count ( huggins ) asc limit 1   
Answer:2014-01-22   
SQL:select tiepolo from my\_table where puccoon < 191 and intrados < 79 group by huggins   
having intrados < 81 and tiepolo < 255 order by count ( barye ) asc limit 1   
Answer:180   
SQL:select tiepolo from my\_table where scope > 31 group by huggins   
having min ( tiepolo ) = 62 order by count ( distinct mutinus ) asc limit 1   
Answer:62   
SQL:select wear from my\_table where huggins = 'ytyayrvj' group by huggins   
having count ( huggins ) < 5 order by count ( distinct mutinus ) desc limit 1   
Answer:

## C.5 SQL Execution Examples (Multi-Answer)

You are an SQL executor, you need to execute SQL based on the give table and SQL statement to obtain the execution results.

<table><tr><td>suiting</td><td>chisel</td><td>highboy I</td><td>broccoli</td><td>| newburgh</td><td>acetum</td><td>brewpubI</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>zbwamhiui</td><td>nnkfvevxw</td><td>I 50</td><td>88</td><td>zhwohj</td><td>1 opufj</td><td>214</td></tr><tr><td>zroosgm</td><td>yvftt</td><td>309</td><td>168</td><td>zhwohj</td><td>xqsu</td><td>136</td></tr><tr><td>zroosgm</td><td>1nri</td><td>152</td><td>78</td><td>zhwohj</td><td>ikvsd</td><td>219</td></tr><tr><td>kjsdl</td><td>trei</td><td>I 234</td><td>287</td><td>egkgkvbec</td><td>mhxcxyg</td><td>23</td></tr><tr><td>zroosgm</td><td>mctnpwbd</td><td>71</td><td>242</td><td>egkgkvbec</td><td>yszfokeom</td><td>180</td></tr><tr><td>zbwamhiui</td><td>ptqtj</td><td>19</td><td>81</td><td>egkgkvbec</td><td>hyfmk</td><td>116</td></tr><tr><td>zroosgm</td><td>1pjvwn</td><td>258</td><td>313</td><td>uftnwbd</td><td>oevmj</td><td>65</td></tr><tr><td>kjsdl</td><td>ididumrhw</td><td>64</td><td>101</td><td>uftnwbd</td><td>xjakwpayx</td><td>327</td></tr><tr><td>zbwamhiui</td><td>wdtncbyn</td><td>165</td><td>209</td><td>uftnwbd</td><td>xrbqvxb</td><td>192</td></tr><tr><td>zbwamhiui</td><td>wyjjc</td><td>219</td><td>6</td><td>uftnwbd</td><td>pzqr</td><td>188</td></tr><tr><td>zroosgm</td><td>qumxgwvls</td><td>314</td><td>246</td><td>uftnwbd</td><td>ehevtf</td><td>60</td></tr><tr><td>zbwamhiui</td><td>adiyf</td><td>207</td><td>298</td><td>egkgkvbec</td><td>wbrgejgf</td><td>80</td></tr><tr><td>zbwamhiui</td><td>qpgpbj</td><td>307</td><td>306</td><td>egkgkvbec</td><td>mcjuonhc</td><td>192</td></tr><tr><td>zbwamhiui</td><td>ehsk</td><td>47 一</td><td>244</td><td>zhwohj</td><td>tcdlnc</td><td>280</td></tr><tr><td>kjsdl</td><td>orlosbok</td><td>21</td><td>93</td><td>egkgkvbec</td><td>dzvwohjo</td><td>103</td></tr><tr><td>zbwamhiui</td><td>webyyylw</td><td>84</td><td>195</td><td>egkgkvbec</td><td>xbmv</td><td>289</td></tr><tr><td>kjsdl</td><td>mrcecp</td><td>48</td><td>264</td><td>egkgkvbec</td><td>xhprcocik</td><td>265</td></tr><tr><td>kjsdl</td><td>ngajupd</td><td>247</td><td>52</td><td>zhwohj</td><td>pcokyw</td><td>247</td></tr><tr><td>zroosgm</td><td>xeeuixkze</td><td>120</td><td>288</td><td>zhwohj</td><td>yishnriw</td><td>138</td></tr><tr><td>kjsdl</td><td>kbczy</td><td>119</td><td>13</td><td>egkgkvbec</td><td>1tpmyfdt</td><td>73</td></tr><tr><td>zbwamhiui</td><td>uvvdzo</td><td>150</td><td>57</td><td>uftnwbd</td><td>tajlsm</td><td>295</td></tr><tr><td>zbwamhiui</td><td>enbffevhp</td><td>290 一</td><td>92</td><td>zhwohj</td><td>gjjznp</td><td>18</td></tr><tr><td>zroosgm</td><td>imubtcc</td><td>79 I</td><td>19</td><td>uftnwbd</td><td>eqymwj</td><td>112</td></tr></table>

SQL:select suiting from my\_table group by suiting having count ( newburgh ) > 6

<table><tr><td>| suiting</td></tr><tr><td></td></tr><tr><td>| zbwamhiui I</td></tr><tr><td>| zroosgm</td></tr></table>

SQL:select acetum,newburgh,suiting from my\_table where highboy > 234

| acetum | newburgh | suiting |   
|:--- -----|:-- --|:-- --|   
| xqsu | zhwohj | zroosgm |   
| oevmj | uftnwbd | zroosgm |   
| ehevtf | uftnwbd | zroosgm |   
| mcjuonhc | egkgkvbec | zbwamhiui |   
| pcokyw | zhwohj | kjsdl |   
| gjjznp | zhwohj | zbwamhiui |

SQL:select count ( chisel ) from my\_table where highboy < brewpub group by newburgh having min ( highboy ) < 47

| count ( chisel ) |  
--:|  
| 5 |

SQL:select newburgh from my\_table where brewpub > 138 order by broccoli desc limit 1 Answer:

```markdown
| newburgh |
|:------ ----|
| egkgkvbec |
```

SQL:select suiting from my\_table where highboy > broccoli group by suiting having min ( highboy ) < 314

Answer:

## C.6 Multi-step Instruction (Few-shot)

Table:

You need to obtain the final answer based on the table and instructions.   
Only give me the result and do not output any other words.

<table><tr><td rowspan=1 colspan=1>:</td><td rowspan=1 colspan=1>puccoon</td><td rowspan=1 colspan=1>tiepolo I</td><td rowspan=1 colspan=2>scope  mutinus</td><td rowspan=1 colspan=1>intrados|h</td><td rowspan=1 colspan=1>uggins</td><td rowspan=1 colspan=1>barye一</td><td rowspan=1 colspan=1>wear I</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>171</td><td rowspan=1 colspan=1>225</td><td rowspan=1 colspan=2>145  2007-04-27</td><td rowspan=1 colspan=2>322  yefihroyn</td><td rowspan=1 colspan=1>79</td><td rowspan=1 colspan=1>207</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>213</td><td rowspan=1 colspan=1>116</td><td rowspan=1 colspan=1>319</td><td rowspan=1 colspan=1>2016-01-15</td><td rowspan=1 colspan=1>288</td><td rowspan=1 colspan=1>ytyayrvj</td><td rowspan=1 colspan=1>246</td><td rowspan=1 colspan=1>272</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>191</td><td rowspan=1 colspan=1>229</td><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=1>2022-11-08</td><td rowspan=1 colspan=1>218</td><td rowspan=1 colspan=1>gpmvax</td><td rowspan=1 colspan=1>167</td><td rowspan=1 colspan=1>73</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>97</td><td rowspan=1 colspan=1>155</td><td rowspan=1 colspan=1>189</td><td rowspan=1 colspan=1>2013-10-30</td><td rowspan=1 colspan=1>79</td><td rowspan=1 colspan=1>gpmvax</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>233</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>295</td><td rowspan=1 colspan=1>2018-12-10</td><td rowspan=1 colspan=1>81</td><td rowspan=1 colspan=1>yefihroyn</td><td rowspan=1 colspan=1>187</td><td rowspan=1 colspan=1>198</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>285</td><td rowspan=1 colspan=1>304</td><td rowspan=1 colspan=1>168</td><td rowspan=1 colspan=1>2017-03-24</td><td rowspan=1 colspan=1>75</td><td rowspan=1 colspan=1>gpmvax</td><td rowspan=1 colspan=1>111</td><td rowspan=1 colspan=1>77</td></tr><tr><td rowspan=1 colspan=1>6I</td><td rowspan=1 colspan=1>233</td><td rowspan=1 colspan=1>325</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>2014-01-22</td><td rowspan=1 colspan=1>114</td><td rowspan=1 colspan=1>ytyayrvj</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>219</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>146</td><td rowspan=1 colspan=1>164</td><td rowspan=1 colspan=1>2021-12-07</td><td rowspan=1 colspan=1>311</td><td rowspan=1 colspan=1>ytyayrvj</td><td rowspan=1 colspan=1>188</td><td rowspan=1 colspan=1>31</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>112</td><td rowspan=1 colspan=1>255</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>2015-12-07</td><td rowspan=1 colspan=1>214</td><td rowspan=1 colspan=1>gpmvax</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>271</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>175</td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=1>181</td><td rowspan=1 colspan=1>2012-04-21</td><td rowspan=1 colspan=1>182</td><td rowspan=1 colspan=1>gpmvax</td><td rowspan=1 colspan=1>105</td><td rowspan=1 colspan=1>76</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=1>101</td><td rowspan=1 colspan=1>2008-04-28</td><td rowspan=1 colspan=1>168</td><td rowspan=1 colspan=1>gpmvax</td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1>119</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>180</td><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=1>2004-06-23</td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=1>yefihroyn</td><td rowspan=1 colspan=1>314</td><td rowspan=1 colspan=1>97 </td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>297</td><td rowspan=1 colspan=1>251</td><td rowspan=1 colspan=1>249</td><td rowspan=1 colspan=1>2022-02-02</td><td rowspan=1 colspan=1>185</td><td rowspan=1 colspan=1>yefihroyn</td><td rowspan=1 colspan=1>278</td><td rowspan=1 colspan=1>313I</td></tr><tr><td rowspan=1 colspan=1>13I</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1>2016-04-14</td><td rowspan=1 colspan=1>243</td><td rowspan=1 colspan=1>ytyayrvj</td><td rowspan=1 colspan=1>213</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>45</td><td rowspan=1 colspan=1>215</td><td rowspan=1 colspan=1>182</td><td rowspan=1 colspan=1>2012-06-15</td><td rowspan=1 colspan=1>251</td><td rowspan=1 colspan=1>yefihroyn</td><td rowspan=1 colspan=1>221</td><td rowspan=1 colspan=1>83</td></tr></table>

Now you need to get the answer based on the instruction,  
only give me the result and do not output any other words.  
The following are some examples.  
Instruction:Please filter the rows by the column conditions, which need to be met:  
The value of column tiepolo needs to be greater than 146.  
The rows are then grouped according to the value of the huggins in the remaining rows.  
Then filter some groups by the following condition:the number of column huggins is greater than 1.

Select the average of values of intrados column in filtered rows.

Sort the obtained values in ascending order of the number of tiepolo

and select the smallest value to get the answer.

Answer:146.5

Instruction:Please filter the rows by the column conditions, which need to be met: The value of column huggins is 'gpmvax'.

The rows are then grouped according to the value of the huggins in the remaining rows.

Then filter some groups by the following condition:the column wear is less than 83.

Select values of wear column in filtered rows.

Sort the obtained values in ascending order of the number of non-repeating barye

and select the smallest value to get the answer.

Answer:73

Instruction:Please filter the rows by the column conditions, which need to be met:

The value of column huggins is 'ytyayrvj'.

The rows are then grouped according to the value of the huggins in the remaining rows.

Then filter some groups by the following condition:the number of column huggins is less than 5.

Select values of wear column in filtered rows.

Sort the obtained values in descending order of the number of non-repeating mutinus

and select the largest value to get the answer.

Answer:

## C.7 Chain-of-Thought SQL Execution Prompting Examples

You are an SQL executor, you need to output the execution process and final answer based on table and SQL. Table:

<table><tr><td></td><td></td><td>masthead</td><td>laertes</td><td>boo</td><td>bothrops</td><td>height</td><td>scraper</td><td>trouser</td><td>lozenge |</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0 case</td><td></td><td>araeswrid</td><td>41</td><td>1yucg</td><td>urbsmxiv</td><td>vgxrh</td><td>esauw I</td><td>281</td></tr><tr><td></td><td>1 case</td><td></td><td>araeswrid</td><td>138</td><td>1yucg</td><td>tbvg</td><td>oerigocb</td><td>stevw</td><td>177</td></tr><tr><td></td><td>2 case</td><td></td><td>zncmrrvg</td><td>303</td><td>loclzoglg</td><td>tbvg</td><td>vgxrh</td><td>stevw I</td><td>234</td></tr><tr><td></td><td>3</td><td>thyngfwts</td><td>araeswrid</td><td>288</td><td>loclzoglg</td><td>tbvg</td><td>vgxrh</td><td>esauw I</td><td>224</td></tr><tr><td></td><td>4</td><td>thyngfwts</td><td>mrehctv</td><td>177</td><td>loclzoglg</td><td>urbsmxiv</td><td>vgxrh</td><td>esauw I</td><td>228</td></tr><tr><td></td><td>5 case</td><td></td><td>araeswrid</td><td>163</td><td>loclzoglg</td><td>urbsmxiv</td><td>oerigocb</td><td>stevw I</td><td>60</td></tr><tr><td></td><td>6</td><td>thyngfwts</td><td>mrehctv</td><td>45</td><td>loclzoglg</td><td>cidufm</td><td>oerigocb</td><td>esauw</td><td>289</td></tr><tr><td></td><td>7</td><td>thyngfwts</td><td>zncmrrvg</td><td>42</td><td>loclzoglg</td><td>tbvg</td><td>ffljyxb</td><td>stevw I</td><td>296</td></tr><tr><td></td><td>8 case</td><td></td><td>araeswrid</td><td>275</td><td>1yucg</td><td>cidufm</td><td>vgxrh</td><td>stevw I</td><td>172I</td></tr><tr><td></td><td>9 case</td><td></td><td>mrehctv</td><td>20</td><td>loclzoglg</td><td>tbvg</td><td>vgxrh</td><td>esauw I</td><td>147I</td></tr><tr><td></td><td>10</td><td>thyngfwts</td><td>araeswrid</td><td>302</td><td>1yucg</td><td>urbsmxiv</td><td>vgxrh</td><td>stevw I</td><td>297</td></tr><tr><td></td><td>11</td><td>thyngfwts</td><td>zncmrrvg</td><td>137</td><td>loclzoglg</td><td>tbvg</td><td>vgxrh</td><td>esauw</td><td>63</td></tr><tr><td></td><td>12 case</td><td></td><td>araeswrid</td><td>186</td><td>loclzoglg</td><td>cidufm</td><td>ffljyxb</td><td>esauw I</td><td>268</td></tr><tr><td></td><td>13 case</td><td></td><td>araeswrid</td><td>194</td><td>loclzoglg</td><td>cidufm</td><td>vgxrh</td><td>esauw 1</td><td>98</td></tr><tr><td></td><td>14 case</td><td></td><td>araeswrid</td><td>234</td><td>1yucg</td><td>urbsmxiv</td><td>vgxrh</td><td>stevw 1</td><td>276I</td></tr></table>

Now you need to get the answer based on the instruction,  
only give me the intermedium results and the final answer.  
SQL:  
select masthead from my\_table where height = 'tbvg' group by masthead order by count ( laertes ) desc limit 1 Execution process:  
You need to execute 3 steps.  
Please filter the rows by the column conditions, which need to be met: The value of column butcher is 'jxys'. Intermediate results 0:

<table><tr><td></td><td></td><td> encyclia</td><td>| butcher</td><td>bowdler</td><td>| nuthatch</td><td></td><td>cachexia | claret</td><td></td><td>cortina |</td><td>strombus |</td><td></td></tr><tr><td></td><td>---:</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0 | adnh</td><td>| jxys</td><td>|cxjvfz</td><td>| clmb</td><td></td><td></td><td>2 | oqmdmbfg</td><td>2511</td><td>184</td><td></td></tr><tr><td></td><td></td><td>1 | xvoxfjbm</td><td>| jxys</td><td>cxjvfz</td><td>clmb</td><td></td><td>275</td><td>oqmdmbfg</td><td>140</td><td>303</td><td></td></tr><tr><td></td><td>2 | adnh</td><td></td><td>| jxys</td><td>eohdpivo</td><td>| clmb</td><td></td><td>2981</td><td>oqmdmbfg</td><td>142</td><td>28</td><td></td></tr><tr><td></td><td>3 | adnh</td><td></td><td>jxys</td><td>eohdpivo</td><td>| rcyixdl</td><td></td><td>153 | oqmdmbfg</td><td></td><td>50</td><td>3061</td><td></td></tr><tr><td></td><td></td><td>4 | xvoxfjbm</td><td>| jxys</td><td>eohdpivo</td><td>rcyixdl</td><td></td><td>315 | rxbttbm</td><td></td><td>201 </td><td></td><td>861</td></tr></table>

Step 1: Select values of strombus column in filtered rows.  
Intermediate results 1:  
184,303,28,306,86

Step 2: Sort the obtained values in ascending order of claret and select the smallest value to get the answer.

Answer: 184

## C.8 Real Table SQL Execution (Few-shot)

You are an SQL executor, you need to execute SQL based on the give table

and SQL statement to obtain the execution results.

Only give me the execution results and do not output any other words.

Table:
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>id一</td><td rowspan=1 colspan=1>agg</td><td rowspan=1 colspan=1>rank1</td><td rowspan=1 colspan=1>nation</td><td rowspan=1 colspan=1>gold</td><td rowspan=1 colspan=1>silver</td><td rowspan=1 colspan=1>bronze</td><td rowspan=1 colspan=1>total1•</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0一</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>soviet union</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>991</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0一</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>united states</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>94I</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0一</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>east germany(gdr)</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>23 一</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>66I</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>0一</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>west germany(frg)</td><td rowspan=1 colspan=1>13 _</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>40I</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>0I</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>japan</td><td rowspan=1 colspan=1>13一</td><td rowspan=1 colspan=1>8一</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>29I</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0 I</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>australia</td><td rowspan=1 colspan=1>8一</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>17一</td></tr><tr><td rowspan=1 colspan=1>6 一</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>poland</td><td rowspan=1 colspan=1>7 _</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>21一</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0一</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>hungary</td><td rowspan=1 colspan=1>6 1</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>35I</td></tr><tr><td rowspan=1 colspan=1>8一</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0一</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>bulgaria</td><td rowspan=1 colspan=1>6 _</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>211</td></tr><tr><td rowspan=1 colspan=1>9 一</td><td rowspan=1 colspan=1>10一</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>italy                 一</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>10_</td><td rowspan=1 colspan=1>18</td></tr></table>

Now you need to execute SQL based on the given table and SQL statement to obtain the execution result.  
Only give me the result and do not output any other words or SQL statement.  
The following are some examples.  
SQL:select nation from table where rank = 1  
Answer:Soviet Union  
SQL:select nation from table where nation != 'bulgaria  
and total = ( select total from table where nation = 'bulgaria' )  
Answer:Poland

SQL:select nation from table order by bronze limit 1

Answer:Australia

SQL:select nation from table order by bronze limit 1

Answer:Australia

SQL:select silver from table order by gold desc limit 1

Answer:

## C.9 Real Table Question Answering (Few-shot)

You need to obtain the final answer based on the table and questions.

Only give me the answer and do not output any other words.

<table><tr><td rowspan=1 colspan=1>一*</td><td rowspan=1 colspan=1>id一</td><td rowspan=1 colspan=1>agg</td><td rowspan=1 colspan=1>rank</td><td rowspan=1 colspan=1>nation               一</td><td rowspan=1 colspan=1>gold</td><td rowspan=1 colspan=1>silverI</td><td rowspan=1 colspan=1>bronze</td><td rowspan=1 colspan=1>total</td></tr><tr><td rowspan=1 colspan=1>0 一</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>01</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>soviet union        一</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>27一</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>99</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0I</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>united states       一</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>31一</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>94</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0I</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>east germany(gdr)</td><td rowspan=1 colspan=1>20 _</td><td rowspan=1 colspan=1>23一</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>66一</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>0I</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>west germany(frg)</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>11 一</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>40一</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>0一</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>japan                一</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>8一</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>29</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0一</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>australia           I</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>7一</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>17</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>0一</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>poland               一</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>5一</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>21</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0一</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>hungary              一</td><td rowspan=1 colspan=1>6 _</td><td rowspan=1 colspan=1>13一</td><td rowspan=1 colspan=1>16 _</td><td rowspan=1 colspan=1>35一</td></tr><tr><td rowspan=1 colspan=1>8一</td><td rowspan=1 colspan=1>9 一</td><td rowspan=1 colspan=1>0一</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>bulgaria             一</td><td rowspan=1 colspan=1>6一</td><td rowspan=1 colspan=1>10一</td><td rowspan=1 colspan=1>5一</td><td rowspan=1 colspan=1>21</td></tr><tr><td rowspan=1 colspan=1>9 一</td><td rowspan=1 colspan=1>10一</td><td rowspan=1 colspan=1>0一</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>italy                一</td><td rowspan=1 colspan=1>5一</td><td rowspan=1 colspan=1>3I</td><td rowspan=1 colspan=1>10一</td><td rowspan=1 colspan=1>18I</td></tr></table>

Now you need to get the answer based on the question,  
only give me the answer and do not output any other words.  
The following are some examples.  
Question:which country was first in rank at the 1972 olympics ?  
Answer:Soviet Union

Question:which country won the same amount of medals as bulgaria in these olympics ?

Answer:Poland

Question:which nation won the least number of bronze medals ?

Answer:Australia

Question:which nation received the least bronze medals

Answer:Australia

Question:what number of silver medals was won by the nation with the most gold medals ? Answer:

## D Experiments Settings Details

## D.1 Setting Description

## Table Config

"col\_min": 5, // the min number of cols

"col\_max": 8, // the max number of cols

"row\_min": 15, // the min number of rows

"row\_max": 40, // the max number of rows

```csv
"text_int_date": [0.55, 0.35, 0.1], // text,int,date header ratio
"text_int_date_fix": ["TEXT", "TEXT", "INT", "INT", "INT"], // Specify the type of each header
// Probability of duplicate values in each column
"value_repeat_ratio": [0, 0.2, 0.3, 0, 0, 0, 0, 0, 0.2, 0.5],
"value_repeat_ratio_fix": ["random", "random"], // Specify the duplicate values of each column
SQL Config
"nest": [1], // Number of SQL nestings. options: [1], [2], [1,2],[1,2, 3]
"keywords_setting": { // if a Keyword is False, then no SQL containing this Keyword is generated.
"select": true,
"where": true,
"group by": true,
"having": true,
"order by": true
},
"length_setting": { // control the length of sql
"is_available": false, // To enable this setting, you need to adjust "is_available" to true first.
// 'value' can be set to specific values, such as [13,14,15],
// if value is null, then the range is used [min, max]
"value": [],
"min": 6,
"max": 16
},
"column_ratio": { // Controlling the ratio of columns involved in SQL
"is_available": false, // To enable this setting, you need to adjust "is_available" to true first.
// 'value' can be set to specific values, such as [1,2], Control the number of columns involved in SQL
"value": [],
// if value is null, then the range is used [min, max], it's the used ratio = (used columns) / (all columns)
"min": 0.1,
"max": 0.3
},
"select_row_ratio":{ // Controlling the ratio of rows involved in select keyword
"is_available": false, // To enable this setting, you need to adjust "is_available" to true first.
// 'value' can be set to specific values, such as [1,2,3,4], Control the number of rows involved in SQL
"value": [],
// if value is null, then the range is used [min, max], it's the used ratio = (select rows) / (all rows)
"min": 0.1,
"max": 0.2
},
// Controlling the calculate times of the sql ['+','-','*','/','sum','count','min','max','avg']
"calculate_times": {
"is_available": false, // To enable this setting, you need to adjust "is_available" to true first.
"value": [1,2,3,4] // 'value' can be set to specific values, means the calculate times
},
// Controlling the filter times of the sql ['=','>','<','in','like']
"filter_times": {
"is_available": false, // To enable this setting, you need to adjust "is_available" to true first.
"value": [1,2,3,4,5] // 'value' can be set to specific values, means the calculate times
},
// Controlling the location of answer in the table, usually used in long-context understanding
"answer_location": {
"is_available": false, // To enable this setting, you need to adjust "is_available" to true first.
"value": null,
"min": 0.1, // if value is null, then the range is used [min, max],
means that 0.1 < (Row where answer is located ) / (Row number) < 0.9
"max": 0.9
},
// usually remains 1 in this repo, we often just test the sql whose answer is from one cell.
"answer_cells_number": 1,
"include": [],
"exclude": [],
"n_shot": 5
```

## D.2 General Setting

## Table Config

```python
"col_min": 5,
"col_max": 5,
"row_min": 30,
"row_max": 30,
```

```csv
"text_int_date": [0.5, 0.45, 0.05],
"value_repeat_ratio": [0, 0.2, 0.3, 0, 0, 0, 0, 0, 0, 0.5]
SQL Config
"nest": [1,2,3],
"select_grammar": [],
"keywords_setting": { "select": true,
"where": true,
"group by": true,
"having": true,
"order by": true
},
"length_setting": {
"is_available": false,
"value": [],
"min": 6,
"max": 16
},
"column_ratio": {
"is_available": false,
"value": [],
"min": 0.1,
"max": 0.3
},
"select_row_ratio":{
"is_available": false,
"value": [],
"min": 0,
"max": 0.2
},
"calculate_times": {
"is_available": false,
"value": [0]
},
"filter_times": {
"is_available": false,
"value": [0]
},
"answer_location": {
"is_available": false,
"row_value": [],
"column_value":[0],
"min": 0,
"max": 1
},
"answer_cells_number": 1,
"multi_test": false,
"include": [],
"exclude": [],
"n_shot": 5
```

## D.3 LLMs Used In This Paper

LLMs. LLaMA2 (Touvron et al., 2023a), Qwen (Bai et al., 2023a), InternLM (Team, 2023), Mistral, XGen (Nijkamp et al., 2023), Falcon (Penedo et al., 2023), phi-1\_5 (Li et al., 2023b), StableLM (Andonian et al., 2021), Pythia (Biderman et al., 2023), CodeLlama (Rozière et al., 2023), StarCoder (Li et al., 2023a), CodeGen (Nijkamp et al., 2022).

We all use the official model weight from the Huggingface Models<sup>3</sup>. Above we used the model’s abbreviation, we list the model’s huggingface official label in Table 7.

## D.4 Markdown vs. Flatten Setting Experiments

```typescript
"0": Size: 100 * 5, Template: Easy, Model: GPT-3.5
"1": Size: 50 * 5, Template: Easy, Model: GPT-3.5
"2": Size: 20 * 6, Template: Count, Model: GPT-3.5
"3": Size: 40 * 10, Template: Where Condition Text, Model: GPT-3.5
"4": Size: 10 * 20, Template: Where Condition Text, Model: GPT-3.5
```

<table><tr><td>Model</td><td>Name</td></tr><tr><td>Mistral-7B</td><td>mistralai/Mistral-7B-v0.1</td></tr><tr><td>Llama-2-13B</td><td>meta-llama/Llama-2-13b-hf</td></tr><tr><td>InternLM-20B</td><td>internlm/internlm-20b</td></tr><tr><td>Qwen-14B</td><td>Qwen/Qwen-14B</td></tr><tr><td>Llama-2-7B</td><td>meta-llama/Llama-2-7b-hf</td></tr><tr><td>Qwen-7B</td><td>Qwen/Qwen-7B</td></tr><tr><td>XGen-7B</td><td>Salesforce/xgen-7b-8k-base</td></tr><tr><td>Internlm-7B</td><td>internlm/internlm-7b</td></tr><tr><td>Phi-1_5</td><td>microsoft/phi-1_5</td></tr><tr><td>Stablelm-7B</td><td>stabilityai/stablelm-base-alpha-7b</td></tr><tr><td>Stablelm-3B</td><td>stabilityai/stablelm-base-alpha-3b</td></tr><tr><td>Pythia-12B</td><td>EleutherAI/pythia-12b</td></tr><tr><td>Pythia-6.9B</td><td>EleutherAI/pythia-6.9b</td></tr><tr><td>Pythia-2.8B</td><td>EleutherAI/pythia-2.8b</td></tr><tr><td>Pythia-1B</td><td>EleutherAI/pythia-1b</td></tr><tr><td>Llama-2-70B</td><td>meta-llama/Llama-2-70b-hf</td></tr><tr><td>CodeLlama-34B</td><td>codellama/CodeLlama-34b-hf</td></tr><tr><td>CodeLlama-13B</td><td>codellama/CodeLlama-13b-hf</td></tr><tr><td>CodeLlama-7B</td><td>codellama/CodeLlama-7b-hf</td></tr><tr><td>StarCoder-15B</td><td>bigcode/starcoderbase</td></tr><tr><td>StarCoder-7B</td><td>bigcode/starcoderbase-7b</td></tr><tr><td>StarCoder-3B</td><td>bigcode/starcoderbase-3b</td></tr><tr><td>StarCoder-1B</td><td>bigcode/starcoderbase-1b</td></tr><tr><td>CodeGen-15B</td><td>Salesforce/codegen-16B-multi</td></tr><tr><td>CodeGen-6B</td><td>Salesforce/codegen-6B-multi</td></tr><tr><td>CodeGen-2B</td><td>Salesforce/codegen-2B-multi</td></tr><tr><td>Yarn-LLaMA2-13B</td><td>NousResearch/Yarn-Llama-2-7b-64k</td></tr><tr><td>LongChat-13B</td><td>lmsys/longchat-7b-16k</td></tr><tr><td>RWKV-Raven-14B</td><td>lmsys/longchat-7b-16k</td></tr></table>

Table 7: LLMs used in our experiments and their corresponding names in Huggingface Hub.  
"5": Size: 10 \* 15, Template: Where Condition Text, Model: GPT-3.5  
"6": Size: 50 \* 5, Template: Easy, Model: Llama-2-13B  
"7": Size: 100 \* 5, Template: Easy, Model: Yarn-Llama-2-13B  
"8": Size: 50 \* 5, Template: Easy, Model: Yarn-Llama-2-13B  
"9": Size: 25 \* 7, Template: General, Model: Llama-2-13B  
"10": Size: (15\~40) \* (6\~9), Template: General, Model: Llama-2-13B  
"11": Size: (15\~40) \* (6\~9), Template: General, Model: Llama-2-13B  
"12": Size: (15\~40) \* (6\~9), Template: Easy, Model: Llama-2-13B  
"13": Size: (15\~40) \* (6\~9), Template: Easy, Model: Llama-2-13B  
"14": Size: (15\~40) \* (6\~9), Template: Easy, Model: Llama-2-13B