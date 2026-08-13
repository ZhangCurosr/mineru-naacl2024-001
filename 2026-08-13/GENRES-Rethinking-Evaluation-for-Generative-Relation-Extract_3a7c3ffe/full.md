# GENRES: Rethinking Evaluation for Generative Relation Extraction in the Era of Large Language Models

Pengcheng Jiang , Jiacheng Lin , Zifeng Wang , Jimeng Sun , and Jiawei Han

Department of Computer Science, University of Illinois at Urbana-Champaign {pj20,jl254,zifengw2,jimeng,hanj}@illinois.edu

## Abstract

The field of relation extraction (RE) is experiencing a notable shift towards generative relation extraction (GRE), leveraging the capabilities of large language models (LLMs). However, we discovered that traditional relation extraction (RE) metrics like precision and recall fall short in evaluating GRE methods. This shortfall arises because these metrics rely on exact matching with human-annotated reference relations, while GRE methods often produce diverse and semantically accurate relations that differ from the references. To fill this gap, we introduce GENRES for a multidimensional assessment in terms of the topic similarity, uniqueness, granularity, factualness, and completeness of the GRE results. With GENRES, we empirically identified that (1) precision/recall fails to justify the performance of GRE methods; (2) human-annotated referential relations can be incomplete; (3) prompting LLMs with a fixed set of relations or entities can cause hallucinations. Next, we conducted a human evaluation of GRE methods that shows GENRES is consistent with human preferences for RE quality. Last, we made a comprehensive evaluation of fourteen leading LLMs using GENRES across document, bag, and sentence level RE datasets, respectively, to set the benchmark for future research in GRE<sup>1</sup>.

## 1 Introduction

Relation Extraction (RE) is one of the most critical tasks in natural language processing (Han et al., 2020). In essence, RE transforms unstructured text into structured, actionable knowledge (e.g., knowledge graphs). However, the traditional RE methods only mine the predefined patterns referring to the predefined sets of relations and entities, thus often struggling to capture the complexity of natural language. Recently, Large Language Models (LLMs)

![](images/5ba97c6d7132654814036ad3fb44d68fefbcaccb86cbae3efbe72087b55e58cc.jpg)  
Figure 1: Generative Relation Extraction (GRE): Contrasting Closed and Semi-open GRE’s type constraints with Open GRE’s reliance on source text alone.

like GPT (OpenAI, 2023), promise a transition to Generative Relation Extraction (GRE). LLM-based GRE methods are capable of comprehending the input texts and then identifying complex relationships without the constraints of predefined patterns in a zero-shot manner. This is particularly advantageous when there is a scarcity of training data, and the input texts are varied.

Existing applications of LLMs in GRE are either performing binary classification tasks (Li et al., 2023a) given entity pairs and a set of predefined relation types, or given restricted entity types (Wadhwa et al., 2023a; Zhu et al., 2023), which overlook extensive novel relations and entities beneath the text. Notably, to unlock the full power of LLMs in GRE, we advocate a transformation from “defining a set of relation types”  “finding matches between entities” to “exploring as many relations and entities as possible without limitation” “refinement” (Paulheim, 2017; Liu et al., 2018). This strategy elicits LLMs’ implicit knowledge to discover a wider array of relationships with minimal predefined constraints (Hao et al., 2023), which we define as “Open $\mathrm { G R E } '$ that can be applied to knowledge graph construction for various downstream tasks (Baralis et al., 2013; Wang et al., 2018; Mohamed et al., 2020; Zeng et al., 2022; Jiang et al., 2023b). We illustrate the difference of GRE strategies in Figure 1.

The versatility of GRE, however, poses significant challenges in evaluation (Wadhwa et al., 2023a). Specifically, we identified that traditional relation extraction (RE) metrics like precision and recall only capture the exact matching with humanannotated reference relations, while GRE methods often produce diverse and semantically accurate relations that differ from the references. As such, we argue that precision in GRE should be verified against the source text, and recall should be based on soft matching to accommodate the output flexibility of generative models. Furthermore, a proficient model should not only cover crucial information in the text but also avoid redundant results, ensuring the extracted knowledge is both comprehensive and atomistic. To navigate these new dimensions, we introduce GENRES (GENerative Relation Extraction Scoring), a multi-dimensional framework tailored for evaluating GRE. Our key contributions are as follows.

• We demonstrate the effectiveness of GENRES for evaluating GRE tasks, emphasizing its superiority over traditional metrics.

• We benchmark the open GRE performance of fourteen leading LLMs through GENRES, and paving the way for future research and development of better LLM-based GRE methods.

## 2 Preliminaries

Definition 1 (Source Document) A source document is a piece offree-text, which can be a sentence, a passage, or a document.

Definition 2 (Extracted Triples) A triple $\tau =$ s r o is a structureformatting a piece offree text into a subject s, a relation r, and an object o. $E x \mathrm { - }$ ample: For a sentence "Alice lives in Champaign.", "Alice" is the subject, "live $i n ^ { \prime \prime }$ is the relation, and "Champaign" is the object. Together, they form a triple Alice live\_in Champaign . We define $\mathcal { T } _ { \mathcal { D } } = [ \tau _ { 1 } , \tau _ { 2 } , . . . ]$ as a list oftriples extractedfrom the source document .

## 2.1 Generative Relation Extraction

GRE uses a generative large language model (LLM) to extract relational triples from a source document $\mathcal { D } .$ The model functions on an autoregressive basis at the token level, expressed as $P ( x _ { t } | x _ { 1 } , x _ { 2 } , \ldots , x _ { t - 1 } , \mathcal { D } )$ , where $x _ { t }$ represents the $t ^ { t h }$ token in the output sequence. The process generates a sequence of tokens that are structured into triples $\mathcal { T } _ { \mathcal { D } } = [ \tau _ { 1 } , \tau _ { 2 } , . . . ]$ . We categorize existing GRE methods as follows:

• Closed GRE (Li et al., 2023a): Given (1) source context, (2) entity pairs in the context, and (3) a set of predefined relation types, prompt the LLM to classify the relation type between the entity pairs to compose each triple $\tau _ { i }$

• Semi-open GRE (Wadhwa et al., 2023a): Given (1) source context, (2) a predefined set of relation types, and (3) a predefined set of entity types, prompt the LLM to extract triples $\tau _ { i }$

• Open GRE: Given source context, prompt the LLM to extract triples as many as possible.

## 3 GENRES

Evidenced by previous work conducting semi-open GRE (Wadhwa et al., 2023a), traditional metrics for RE like hard matching precision/recall/F1 are inadequate to evaluate GRE tasks as the LLM generations are flexible. To fill in this gap, we introduce GENRES, an automated multi-aspect evaluation framework for GRE. GENRES are composed of a series of sub-scores defined as follows.

## 3.1 Topical Similarity Score

We compute the topical similarity score (TS) to measure the information abundance of the extracted triples $\mathcal { T } _ { \mathcal { D } }$ compared to the source text . Here, we employ a Latent Dirichlet Allocation (LDA) model (Blei et al., 2003), an algorithm that represents each document as a blend of a certain number of latent topics, for topic modeling. We concatenate the elements in each triple so that $\mathcal { T } _ { \mathcal { D } } ^ { \Delta } = [ \tau _ { 1 } ^ { \prime } , \tau _ { 2 } ^ { \prime } , . . . ] =$ $\left[ s _ { 1 } \oplus r _ { 1 } \oplus o _ { 1 } , s _ { 2 } \oplus r _ { 2 } \oplus o _ { 2 } , . . . \right]$ . TS is computed as:

$$
t ( \mathcal { D } , \mathcal { T } _ { \mathcal { D } } ^ { \Delta } ) = e ^ { - \sum _ { i = 1 } ^ { K } L D A ( \mathcal { D } ) _ { i } \cdot \log \left( \frac { L D A ( \mathcal { D } ) _ { i } } { L D A ( \mathcal { T } _ { \mathcal { D } } ^ { \Delta } ) _ { i } } \right) }\tag{1}
$$

which is based on the KL-divergence of two topical distributions. A higher TS indicates that the extracted triples closely align with the topical content of the source document, reflecting effective and relevant information extraction, while a lower TS suggests that the extracted triples may be missing key topical elements from the source.

![](images/2b095c4325378d01232719d971554fa12afd1ebaf1588435d43b853475bda358.jpg)  
Figure 2: GENRES framework for the evaluation of generative relation extraction (GRE). Left: An example showing the GRE process to extract triples $\mathcal { T } _ { \mathcal { D } }$ from a source text through prompting generative large language model. Right: illustration of sub-scores contained in GREScore regarding: Topical Similarity (§3.1), Uniqueness (§3.2), Fatualness (§3.3), Granularity (§3.4), and Completeness (§3.5).

## 3.2 Uniqueness Score

Uniqueness Score (US) assesses the diversity of the extracted triples $\mathcal { T } _ { D }$ in the GRE, emphasizing the importance of extracting varied and distinct relationships. Given $\mathcal { T } _ { \mathcal { D } } = [ \tau _ { 1 } , \tau _ { 2 } , \ldots , \tau _ { n } ]$ , with each triple $\tau _ { i }$ encoded in a vector $\mathbf { v } _ { i }$ using word embeddings, the US is computed as follows:

$$
u ( \mathcal T _ { \mathcal D } ) = \frac { 1 } { n ( n - 1 ) } \sum _ { i = 1 } ^ { n } \sum _ { j \ne i } ^ { n } \left( \mathbf { C } \mathrm { o s } \mathbf { S i m } ( \mathbf v _ { i } , \mathbf v _ { j } ) < \phi \right)\tag{2}
$$

where $\mathrm { C o s S i m } ( \mathbf { v } _ { i } , \mathbf { v } _ { j } )$ is the cosine similarity between the vector representations of triples $\tau _ { i }$ and $\tau _ { j }$ . ϕ is a predefined similarity threshold. The normalization factor $n ( n - 1 )$ ) accounts for all pairings where i $\neq j .$ . A higher US indicates greater diversity among the triples, while a lower US suggests more similarity and potential redundancy.

## 3.3 Factualness Score

Factualness Score (FS) quantifies the extent to which extracted triples, denoted as $\mathcal { T } _ { \mathcal { D } }$ , align with the information in the source text . This metric is crucial for gauging the hallucinations (Zhang et al., 2023), a phenomenon where LLMs fabricate the content not present in the source text. Building on the foundations laid by prior research (Min et al., 2023; Jiang et al., 2021), FS employs a detailed triple-wise verification process. Each triple τ within $\mathcal { T } _ { D }$ undergoes a thorough check to confirm whether it is supported by factual evidence in :

$$
f ( \mathcal { D } , \mathcal { T } _ { \mathcal { D } } ) = \frac { 1 } { | \mathcal { T } _ { \mathcal { D } } | } \sum _ { \tau \in \mathcal { T } _ { \mathcal { D } } } \mathbb { [ } \tau \mathrm { ~ i s ~ s u p p o r t e d ~ b y ~ } \mathcal { D } \mathbb { I }\tag{3}
$$

where [[τ is supported by $\mathcal { D } ] ]$ is an indicator function that returns 1 if the triple is factual and 0 if it is not. In this study, we adopt the approach from previous work (Min et al., 2023) and utilize an LLM as the fact-checking tool. Specifically, we employ GPT-3.5-Turbo-Instruct as the fact checker, with the methodology detailed in Appendix B.2. A high FS signifies that a substantial portion of the extracted triples are factually consistent with the source text. On the contrary, a low FS indicates a higher incidence of hallucinated or unsupported data. Employing this metric is vital to guarantee the reliability and trustworthiness of the information generated by the model.

## 3.4 Granularity Score

The Granularity Score (GS) evaluates the level of detail of the extracted triples $\mathcal { T } _ { D }$ from the source text . It is based on the premise that triples should capture the optimal granularity of information, not too coarse. The GS aims to penalize triples that are overly broad and could be further split into more precise statements. The process involves an assessment of each triple’s potential to be split into more granular sub-triples. This can be performed by prompting an LLM to evaluate if a given triple can be divided into additional, more specific triples. The number of possible splits is represented by $n _ { \tau }$ for each triple τ.

The Granularity Score for the extracted triples $\mathcal { T } _ { \mathcal { D } }$ is calculated using the formula:

$$
g ( \mathcal { T } _ { \mathcal { D } } ) = \frac { 1 } { | \mathcal { T } _ { \mathcal { D } } | } \sum _ { \tau \in \mathcal { T } _ { \mathcal { D } } } e ^ { - n _ { \tau } }\tag{4}
$$

where $e ^ { - n _ { \tau } }$ is the exponential decay function based on the number of splits $n _ { \tau }$ , which assigns a lower score to triples that can be split into more subtriples (indicating they are too broad or general). Therefore, a lower Granularity Score indicates that the triples could be broken down further, while a higher score suggests that the triples are at an appropriate level of specificity.

## 3.5 Completeness Score

The Completeness Score (CS) evaluates how comprehensively the extracted triples $\mathcal { T } _ { D }$ cover the information present in the source text . This score is analogous to the recall metric in information retrieval and is particularly important when gold standard triples $\mathcal { T } _ { \mathcal { D } } ^ { \prime }$ are available for comparison. CS is assessed by determining the proportion of gold standard triples that are successfully captured by the extracted triples. For each gold standard triple $\tau ^ { \prime }$ , we find the best matching triple τ from $\mathcal { T } _ { \mathcal { D } }$ , using cosine similarity of their embeddings as the soft matching criterion. If the cosine similarity exceeds a specified threshold $\phi ,$ the triple τ is considered a match. CS is then computed as:

$$
c ( \mathcal { T } _ { \mathcal { D } } ^ { \prime } , \mathcal { T } _ { \mathcal { D } } ) = \frac { | \{ \tau ^ { \prime } \in \mathcal { T } _ { \mathcal { D } } ^ { \prime } | \exists \tau \in \mathcal { T } _ { \mathcal { D } } , \sin ( \tau , \tau ^ { \prime } ) \geq \phi \} | } { | \mathcal { T } _ { \mathcal { D } } ^ { \prime } | }\tag{5}
$$

where sim $( \tau , \tau ^ { \prime } ) = \mathrm { C o s S i m } ( e m b ( \tau ) , e m b ( \tau ^ { \prime } ) ) $ calculates the cosine similarity between the embeddings of the extracted triple and the gold standard triple. The threshold $\phi$ is pre-defined to determine the acceptable level of similarity for a match. A higher CS indicates that the extracted triples effectively capture the complete range of information as represented by the “gold standard”. It is worth noting that CS is optional as precise human annotations are expensive and not always available.

## 4 Experiments

## 4.1 Datasets

In our evaluation, we examine several RE datasets with a focus on their performance in GRE using test sets enriched with detailed human annotations. These include: CDR (Li et al., 2016), a documentlevel dataset with 1,500 PubMed abstracts highlighting chemical-disease interactions; DocRED (Yao et al., 2019), also document-level, derived from Wikipedia and Wikidata, featuring extensive entity, coreference, and relational annotations across 5,053 documents; NYT10m and Wiki20m (Han et al., 2019), both bag-level<sup>2</sup> datasets from The New York Times and Wikipedia, respectively, with manually annotated test sets; and TACRED (Zhang et al., 2017) and Wiki80 (Han et al., 2018), sentence-level datasets, the former comprising 106,264 examples across various text sources and the latter containing 56,000 instances with 80 relations from Wikipedia and Wikidata. These datasets collectively offer a comprehensive view of RE capabilities across various levels and sources.

We adopt a random sampling method to select the test sets from the above datasets. We randomly choose {200, 500, 800} samples for the document-, bag-, and sentence-level evaluations<sup>3</sup>.

## 4.2 Implementation

For topical similarity score (TS), we train six LDA models with {50, 100, 150, 150, 150, 150} latent topic numbers and {1500, 5051, 11086, 14257, 38140, 22400} samples (document/bag/sentence) for CDR, DocRED, NYT10m, Wiki20m, TA-CRED, and Wiki80, respectively. For evaluations (US and CS) using word embedding, we retrieve the embedding for each entity and relation in the triple using text-embedding-ada-002, and perform element-wise addition to obtain the triple embedding.<sup>4</sup> Based on our tests, we set the similarity threshold $\phi$ at 0.95. All local LLMs are run on 8 NVIDIA A100 GPUs. All prompts used are detailed in Appendix B.

![](images/fcc5c9ecde411f5e94615194b9ac151e7f7c14a8b5071cd61616ac11ed70c44c.jpg)  
Figure 3: Comparative Analysis of GRE Methods and Evaluation Metrics using the NYT10m Dataset. The diagram showcases the outcomes of closed, semi-open, and open Generative Relation Extraction (GRE) strategies. The distinct entity and relation spans are color-coded, with factual triples specifically highlighted. The extracted triples that affect FS, CS (soft recall), and GS are listed with the corresponding labels. We underline the ground truth labels that are inaccurate or cannot be inferred from the source text.

<table><tr><td rowspan="2"></td><td colspan="4">CDR</td><td colspan="4">NYT10m</td></tr><tr><td>C</td><td>S</td><td>0</td><td>GT</td><td>C</td><td>S</td><td>0</td><td>GT</td></tr><tr><td>#tri #tok</td><td>10.1 6.6</td><td>6.8 4.0</td><td>16.1 8.3</td><td>10.1 5.8</td><td>1.4 4.6</td><td>2.9 2.0</td><td>5.8 7.0</td><td>1.4 4.5</td></tr><tr><td>P</td><td>58.8</td><td>1.1</td><td>0.4</td><td>-</td><td>29.3</td><td>5.2</td><td>0.0</td><td>-</td></tr><tr><td>R F1</td><td>58.7</td><td>0.8</td><td>0.7</td><td>-</td><td>26.6</td><td>12.7</td><td>0.0</td><td>一</td></tr><tr><td></td><td>58.8</td><td>0.7</td><td>0.5</td><td>-</td><td>27.5</td><td>6.5</td><td>0.0</td><td>-</td></tr><tr><td>TS US FS</td><td>11.9 31.8 64.4</td><td>35.5 58.2 62.0</td><td>77.6 89.6 96.8</td><td>9.6 33.4 93.5</td><td>10.3 87.5 72.3</td><td>13.4 91.5 33.7</td><td>54.2 83.0 84.0</td><td>8.7 69.3 84.1</td></tr></table>

∗Closed GRE, due to its use of predefined entity pairs for relation classification, inherently exhibits high triple similarity. Hence, we further check relation embedding similarity for the best soft matching of triples.  
Table 1: Different GRE strategies measured by different metrics including traditional P/R/F1 and GEN-RES. “C”, “S”, “O”, and “GT” denote Closed, Semiopen, Open GRE, and ground truth, respectively. GPT-3.5-Turbo-Instruct was used as the LLM. The highest scores for each dataset are highlighted.

## 4.3 Performance of Different GRE Strategies

We conducted evaluations of closed, semi-open, and open GRE on the CDR and NYT10m datasets. The expansive relation sets and the absence of defined entity types in other datasets render them incompatible with closed and semi-open GRE, owing to the limitations of context window constraints.

This limitation emphasizes the flexibility of open GRE, which operates unconstrained by predefined relation types or entity types, proving its adaptability to a wider array of datasets. The comparative results of these evaluations are presented in Table 1. Combined with our example shown in Figure 3, we summarize the key observations as follows.

Traditional metrics are not ideal for GRE evaluation, especially in semi-open and open GRE settings. Figure 3 illustrates that despite open GRE’s high-quality extractions based on FS and CS, they score zero across these metrics. This occurs because Precision/Recall/F1 depend on exact matching of triples, which are nearly impossible without predefined relation/entity sets, as evidenced by the zero scores for these metrics on the NYT10m dataset in Table 1. This finding syncs with Wadhwa et al. (2023a)’s conclusion.

Human annotations sometimes are unreliable. In Figure 3, we underline several mistakes (e.g., “[Barrick Gold, advisors, Peter Munk], [Barrick Gold, place lived, Toronto]”) in the the ground truth where “Barrick Gold” is a company but incorrectly recognized as a person. Such inaccurate labels are unlikely to be correctly predicted by LLMs. This suggests that traditional metrics that purely rely on ground truth triples, are even inadequate for closed GRE, and more so for semi-open and open GRE.

The imposition of predefined relation sets or entity types can misguide LLMs to generate inaccurate triples. For instance, as seen in Figure 3, closed GRE misclassifies the relation between “Peter Munk” and “Toronto” as “place founded” based on limited choices from the relation set, despite the text not supporting this inference. Similarly, semi-open GRE’s entity recognition becomes problematic when it erroneously divides “exodus of head offices” into separate entities “exodus” and “head offices”, leading to less coherent and less meaningful triples.

<table><tr><td rowspan="2" colspan="2"></td><td colspan="7">CDR</td><td colspan="7">DocRED</td></tr><tr><td>#tri</td><td>#tok</td><td>TS</td><td>US</td><td>FS</td><td>GS</td><td>CS</td><td>#tri</td><td>#tok</td><td>TS</td><td>US</td><td>FS</td><td>GS</td><td>CS</td></tr><tr><td rowspan="5"></td><td>Ground Truth</td><td>10.1</td><td>5.8</td><td>9.6</td><td>33.4</td><td>93.5</td><td>98.1</td><td>100</td><td>12.4</td><td>6.0</td><td>8.4</td><td>64.0</td><td>94.4</td><td>81.9</td><td>100</td></tr><tr><td>Vicuna-7B</td><td>6.8</td><td>8.4</td><td>57.8</td><td>86.9</td><td>84.7</td><td>44.6</td><td>30.7</td><td>7.4</td><td>9.9</td><td>23.1</td><td>81.9</td><td>93.4</td><td>46.8</td><td>28.3</td></tr><tr><td>Vicuna-33B</td><td>6.4</td><td>10.5</td><td>73.0</td><td>89.2</td><td>97.3</td><td>38.4</td><td>32.0</td><td>10.8</td><td>9.8</td><td>34.7</td><td>82.8</td><td>97.2</td><td>49.6</td><td>36.9</td></tr><tr><td>LLaMA-2-7B</td><td>5.6</td><td>6.7</td><td>48.6</td><td>92.0</td><td>62.0</td><td>44.9</td><td>25.7</td><td>2.7</td><td>3.2</td><td>12.8</td><td>93.3</td><td>34.0</td><td>60.6</td><td>12.1</td></tr><tr><td>LLaMA-2-70B WizardLM-70B</td><td>10.8 10.2</td><td>8.1 7.8</td><td>74.8 65.4</td><td>87.6 94.1</td><td>96.6 76.4</td><td>57.8 46.2</td><td>51.0 32.6</td><td>13.8 5.8</td><td>8.7 3.6</td><td>39.2 24.3</td><td>82.6 94.9</td><td>97.3 37.9</td><td>60.9 56.7</td><td>39.2 12.8</td></tr><tr><td rowspan="5">GPT</td><td>text-davinci-003</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-3.5-Turbo-Inst.</td><td>12.7</td><td>8.3</td><td>76.7</td><td>87.2</td><td>96.8</td><td>55.4</td><td>44.3</td><td>15.3</td><td>8.5</td><td>40.1</td><td>84.2</td><td>97.6</td><td>59.8</td><td>46.2</td></tr><tr><td></td><td>16.1 11.2</td><td>8.3</td><td>77.6</td><td>89.6</td><td>96.8</td><td>54.2</td><td>47.8</td><td>17.8</td><td>8.9 9.9</td><td>47.8</td><td>85.6</td><td>98.1</td><td>56.2</td><td>44.7</td></tr><tr><td>GPT-3.5-Turbo</td><td>14.3</td><td>11.4 9.3</td><td>81.7 81.7</td><td>89.2 91.0</td><td>98.2 97.9</td><td>40.3 49.1</td><td>30.2 46.3</td><td>15.0 17.8</td><td>8.7</td><td>50.4 48.6</td><td>84.0 82.8</td><td>98.5 98.6</td><td>49.1 59.6</td><td>36.5 47.3</td></tr><tr><td>GPT-4 GPT-4-Turbo</td><td>18.6</td><td>8.5</td><td>82.1</td><td>91.9</td><td>96.8</td><td>53.1</td><td>48.8</td><td>21.5</td><td>8.7</td><td>50.0</td><td>87.4</td><td>97.6</td><td>63.1</td><td>49.3</td></tr><tr><td rowspan="4">others</td><td>Mistral-7B-Inst.</td><td>14.2</td><td>9.1</td><td>69.0</td><td>74.9</td><td>93.5</td><td>51.1</td><td>40.0</td><td>11.3</td><td>9.6</td><td>30.2</td><td>76.4</td><td>94.1</td><td>55.2</td><td>27.5</td></tr><tr><td>Zephyr-7B-Beta</td><td>25.9</td><td>8.8</td><td>49.1</td><td>79.5</td><td>70.1</td><td>57.7</td><td>29.3</td><td>18.6</td><td>8.6</td><td>27.9</td><td>79.4</td><td>94.7</td><td>64.7</td><td>37.1</td></tr><tr><td>Galactica-30B</td><td>0.2</td><td>0.3</td><td>4.1</td><td>1.1</td><td>0.9</td><td>44.4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>8.6</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>OpenChat-3.5</td><td>8.6</td><td>12.6</td><td>78.7</td><td>91.9</td><td>97.4</td><td>38.2</td><td>31.8</td><td>15.4</td><td>8.9</td><td>39.7</td><td>82.1</td><td>98.1</td><td>61.7</td><td>43.4</td></tr></table>

Table 2: GENRES evaluation of Open GRE on document-level datasets. Scores (%) are averaged across documents. #tri and #tok denote the number of triples per document and the number of tokens per triple, respectively. We highlight the highest within-group scores. Galactica’s low scores are due to its limited size of context window.
<table><tr><td rowspan="2" colspan="2"></td><td colspan="7">NYT10m</td><td colspan="7">Wiki20m</td></tr><tr><td>#tri</td><td>#tok</td><td>TS</td><td>US</td><td>FS</td><td>GS</td><td>CS</td><td>#tri</td><td>#tok</td><td>TS</td><td>US</td><td>FS</td><td>GS</td><td>CS</td></tr><tr><td rowspan="5"></td><td>Ground truth</td><td>1.4</td><td>4.5</td><td>8.7</td><td>69.3</td><td>84.1</td><td>93.1</td><td>100</td><td>2.0</td><td>6.3</td><td>4.4</td><td>21.2</td><td>88.7</td><td>85.1</td><td>100</td></tr><tr><td>Vicuna-7B</td><td>3.1</td><td>7.8</td><td>42.0</td><td>86.4</td><td>80.0</td><td>60.2</td><td>38.9</td><td>3.0</td><td>7.5</td><td>48.3</td><td>67.8</td><td>50.0</td><td>68.6</td><td>37.3</td></tr><tr><td>Vicuna-33B</td><td>4.7</td><td>7.2</td><td>47.8</td><td>80.1</td><td>75.1</td><td>65.2</td><td>46.5</td><td>4.1</td><td>7.0</td><td>49.8</td><td>56.4</td><td>84.4</td><td>75.4</td><td>46.1</td></tr><tr><td>LLaMA-2-7B</td><td>3.1</td><td>6.0</td><td>35.4</td><td>82.2</td><td>78.9</td><td>69.2</td><td>38.4</td><td>3.1</td><td>6.3</td><td>37.9</td><td>73.8</td><td>73.4</td><td>75.6</td><td>36.0</td></tr><tr><td>LLaMA-2-70B WizardLM-70B</td><td>5.0 4.4</td><td>6.9 4.2</td><td>45.4 30.5</td><td>83.0 88.9</td><td>81.7 43.9</td><td>71.8 68.9</td><td>52.4 27.6</td><td>4.1 3.6</td><td>6.9 5.6</td><td>45.2 43.1</td><td>62.0 67.8</td><td>87.1 67.3</td><td>78.4 75.0</td><td>50.2 40.9</td></tr><tr><td rowspan="5">GPT</td><td>text-davinci-003</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-3.5-Turbo-Inst.</td><td>4.9 5.8</td><td>7.1</td><td>50.6</td><td>81.4</td><td>85.8</td><td>69.3</td><td>52.6</td><td>3.7</td><td>8.2</td><td>51.8</td><td>56.9</td><td>91.3</td><td>73.3</td><td>43.5</td></tr><tr><td>GPT-3.5-Turbo</td><td>4.1</td><td>7.0 6.2</td><td>54.2 43.3</td><td>83.0 82.3</td><td>84.0 68.2</td><td>71.9</td><td>53.4</td><td>4.8 3.6</td><td>7.7 7.7</td><td>54.0</td><td>60.3 61.8</td><td>90.1</td><td>78.9</td><td>43.8 32.5</td></tr><tr><td>GPT-4</td><td>5.1</td><td>7.4</td><td>56.2</td><td>81.8</td><td>89.0</td><td>62.8 68.2</td><td>29.8 52.6</td><td>3.8</td><td>8.1</td><td>48.2 59.0</td><td>56.2</td><td>80.2 93.2</td><td>72.7 77.2</td><td>40.0</td></tr><tr><td>GPT-4-Turbo</td><td>5.3</td><td>7.8</td><td>58.1</td><td>84.2</td><td>89.6</td><td>69.1</td><td>53.7</td><td>4.2</td><td>7.6</td><td>56.4</td><td>62.0</td><td>92.4</td><td>81.2</td><td>52.7</td></tr><tr><td rowspan="4">others</td><td>Mistral-7B-Inst.</td><td>5.7</td><td>7.4</td><td>40.6</td><td>77.6</td><td>75.4</td><td>62.9</td><td>36.5</td><td>4.0</td><td>6.9</td><td>43.3</td><td>57.0</td><td>83.6</td><td>69.9</td><td>40.1</td></tr><tr><td>Zephyr-7B-Beta</td><td>7.8</td><td>7.2</td><td>36.5</td><td>80.8</td><td>64.9</td><td>73.8</td><td>47.0</td><td>5.2</td><td>6.8</td><td>40.3</td><td>65.5</td><td>75.5</td><td>79.0</td><td>45.9</td></tr><tr><td>Galactica-30B</td><td>8.3</td><td>8.7</td><td>29.7</td><td>48.4</td><td>52.4</td><td>60.6</td><td>37.0</td><td>6.0</td><td>8.4</td><td>35.3</td><td>49.4</td><td>65.2</td><td>66.8</td><td>38.6</td></tr><tr><td>OpenChat-3.5</td><td>5.2</td><td>7.2</td><td>54.0</td><td>84.7</td><td>84.3</td><td>69.7</td><td>55.3</td><td>4.3</td><td>7.0</td><td>57.5</td><td>61.8</td><td>90.5</td><td>76.0</td><td>47.7</td></tr></table>

Table 3: GENRES evaluation of Open GRE on bag-level datasets. Scores (%) are averaged across bags. #tri and #tok denote the number of triples per bag and the number of tokens per triple, respectively. We highlight the highest within-group scores.

It is also obvious that the range of information captured by extracted triples widens from closed GRE to open GRE. Closed and semi-open GRE, which limit the types of relations or entities, often yield extractions with a narrower scope. This constriction hampers the completeness of the captured information, a fact corroborated by the TS metrics presented in Table 1. Furthermore, providing a more diverse relation set to semi-open GRE, such as the one in NYT10m (as opposed to the more limited CDR, which restricts entity types to chemicals and diseases), results in a significant drop in granularity (GS). In contrast, open GRE maintains stability, underscoring the benefit of eschewing predefined relation/entity types. Although closed GRE records the highest GS and CS, it is benefited from taking extra input entity pairs, which are not provided to simi-open and open GRE.

<table><tr><td rowspan="2" colspan="2"></td><td colspan="7">TACRED</td><td colspan="7">Wiki80</td></tr><tr><td>#tri</td><td>#tok</td><td>TS</td><td>US</td><td>FS</td><td>GS</td><td>CS</td><td>#tri</td><td>#tok</td><td>TS</td><td>US</td><td>FS</td><td>GS</td><td>CS</td></tr><tr><td rowspan="5"></td><td>Ground Truth</td><td>1.4</td><td>4.6</td><td>15.8</td><td>92.7</td><td>87.0</td><td>94.9</td><td>100</td><td>1.0</td><td>5.8</td><td>5.9</td><td>100</td><td>90.1</td><td>84.4</td><td>100</td></tr><tr><td>Vicuna-7B</td><td>2.6</td><td>8.7</td><td>40.4</td><td>85.0</td><td>75.6</td><td>58.9</td><td>36.2</td><td>2.4</td><td>7.9</td><td>41.3</td><td>76.8</td><td>81.0</td><td>61.7</td><td>36.6</td></tr><tr><td>Vicuna-33B</td><td>4.3</td><td>7.3</td><td>44.3</td><td>75.5</td><td>71.0</td><td>69.2</td><td>47.2</td><td>3.8</td><td>7.2</td><td>47.3</td><td>62.1</td><td>79.9</td><td>73.8</td><td>46.8</td></tr><tr><td>LLaMA-2-7B</td><td>2.8</td><td>6.3</td><td>36.7</td><td>85.3</td><td>66.9</td><td>71.2</td><td>37.8</td><td>2.4</td><td>5.8</td><td>25.8</td><td>69.8</td><td>60.4</td><td>76.9</td><td>31.4</td></tr><tr><td>LLaMA-2-70B WizardLM-70B</td><td>4.1 2.1</td><td>6.4 2.9</td><td>40.8 23.3</td><td>79.3 90.7</td><td>74.5 28.0</td><td>76.8 72.1</td><td>56.4 9.8</td><td>3.7 2.1</td><td>6.6 3.2</td><td>41.5 25.6</td><td>64.8 84.9</td><td>82.4 36.6</td><td>76.9 74.4</td><td>49.4 21.4</td></tr><tr><td rowspan="5">GPT</td><td>text-davinci-003</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-3.5-Turbo-Inst.</td><td>4.4</td><td>7.1</td><td>56.1</td><td>79.8</td><td>84.0</td><td>72.8</td><td>58.6</td><td>4.0</td><td>6.8</td><td>59.2</td><td>65.3</td><td>89.2</td><td>74.9</td><td>51.9</td></tr><tr><td>GPT-3.5-Turbo</td><td>5.0 3.9</td><td>7.0</td><td>58.6 52.7</td><td>80.5 81.1</td><td>81.6</td><td>72.6</td><td>58.6</td><td>4.4 3.4</td><td>6.9</td><td>60.2</td><td>69.3</td><td>88.7</td><td>75.4</td><td>54.8</td></tr><tr><td>GPT-4</td><td>4.3</td><td>6.8 7.5</td><td>59.1</td><td>80.4</td><td>76.4 87.6</td><td>67.5 69.1</td><td>39.7 57.8</td><td>4.0</td><td>6.3 7.1</td><td>50.9 65.4</td><td>69.5 66.2</td><td>75.6 92.3</td><td>68.9 74.2</td><td>36.0 47.8</td></tr><tr><td>GPT-4-Turbo</td><td>4.4</td><td>7.8</td><td>58.5</td><td>82.6</td><td>88.6</td><td>73.2</td><td>63.4</td><td>4.0</td><td>7.6</td><td>61.9</td><td>69.4</td><td>92.8</td><td>74.5</td><td>47.1</td></tr><tr><td rowspan="4">others</td><td>Mistral-7B-Inst.</td><td>4.7</td><td>7.1</td><td>43.9</td><td>78.6</td><td>71.0</td><td>65.5</td><td>41.2</td><td>3.6</td><td>7.8</td><td>44.6</td><td>67.8</td><td>83.9</td><td>67.7</td><td>38.5</td></tr><tr><td>Zephyr-7B-Beta</td><td>5.4</td><td>7.6</td><td>36.4</td><td>78.6</td><td>65.8</td><td>72.0</td><td>44.9</td><td>4.5</td><td>7.8</td><td>43.2</td><td>68.1</td><td>77.8</td><td>74.2</td><td>42.6</td></tr><tr><td>Galactica-30B</td><td>8.5</td><td>8.9</td><td>33.4</td><td>43.9</td><td>57.5</td><td>64.1</td><td>30.9</td><td>5.6</td><td>7.2</td><td>35.0</td><td>47.9</td><td>63.1</td><td>73.3</td><td>38.4</td></tr><tr><td>OpenChat-3.5</td><td>4.3</td><td>7.1</td><td>50.7</td><td>80.8</td><td>80.4</td><td>72.1</td><td>60.0</td><td>4.0</td><td>7.0</td><td>53.8</td><td>69.7</td><td>88.7</td><td>74.9</td><td>50.6</td></tr></table>

Table 4: GENRES evaluation of Open GRE on sentence-level datasets. Scores (%) are averaged across sentences. #tri and #tok denote the number of triples per sentence and the number of tokens per triple, respectively. We highlight the highest within-group scores.

![](images/6fce18c2810cf5844b74909d376be9f776d01535def9a5e0fcf8b61af55c4ae9.jpg)  
Figure 4: GRE performance of five LLMs on Wiki20m, each with five runs with random seeds.

## 4.4 Open GRE Performance of LLMs

Due to the aforementioned advantages of Open GRE, we further test the capabilities of the leading LLMs to perform this task, which includes LLaMA Family (Touvron et al., 2023a,b): LLaMA-2-7B, LLaMA-2-70B, Vicuna-1.5-7B, Vicuna-1.3-33B, and WizardLM-70B (Xu et al., 2023). GPT Family (Brown et al., 2020): text-davinci-003, GPT-3.5- Turbo (1106), GPT-3.5-Turbo-Instruct, GPT-4, and GPT-4-Turbo (OpenAI, 2023). Others: Mistral-7B-Instruct (Jiang et al., 2023a), Zephyr-7B-Beta (Tunstall et al., 2023), GALACTICA (Taylor et al., 2022), and OpenChat-3.5 (Wang et al., 2023). Models are selected majorly based on their performance on Chatbot Arena (Zheng et al., 2023). Our evaluation results are shown in Tables 2, 3, and 4.

We summarize our findings as follows.

(1) Within individual datasets, LLaMA-2-70B, GPT-4-Turbo, and OpenChat emerge as the top performers in their respective categories based on the highest scores obtained across six datasets. Interdataset comparisons reveal that the GPT family consistently outperforms others in Topical Similarity (TS), likely due to their supreme capability to interpret the full content of the text unit. Surprisingly, a light model - OpenChat-3.5 (7B) ourperforms heavier LLMs like Galactica-30B, Vicuna-33B, LLaMA-2-70B, WizardLM-70B, text-davinci-003, and GPT-3.5-Turbo on most datasets.

(2) High Completeness Score (CS) can indicate high Factualness Score (FS). This means human annotations are still valuable to evaluate GRE with our soft matching recall. However, high FS does not indicate high CS, as Open GRE is not limited to the fixed relation/entity types. We also observe that the factualness of GPT-4 and GPT-4-Turbo are consistently higher than that of ground truth.

(3) A greater number of tokens per triple does not inherently result in a lower Granularity Score (GS). This suggests that the GS metric can encourage models to identify more atomic relationships rather than merely focusing on brevity.

(4) We observed no clear correlation between the number of triples, Topical Similarity (TS), and Uniqueness Similarity (US), indicating the distinct significance of each metric. For instance, on the

![](images/625eef3747e9f7547b919f5a6e2d00d519f669385004c3a4d4c7e62dbf5bdb46.jpg)  
(a) Topical Similarity

![](images/6b7e6a6e68e4ee2e6ed79175fd4df5fd54abf65635181ab900971526890da3ee.jpg)  
(b) Uniqueness

![](images/d491076a614fd572d223b2638ac74ed27b691fde5a017a9eb18a5cabc62ea3d0.jpg)  
(c) Factualness

![](images/e366ed8503796d3052ad950335374a229484bf8d894af9eea75d14940798e6d3.jpg)  
(d) Granularity

![](images/005eafc8b98f462782410f23a286a6975d2a6794fb50dc264806d3b637acaf29.jpg)  
(e) Completeness  
Figure 5: Human Preference Evaluation (Elo Ratings) vs GenRES Evaluation on 100 Wiki20m samples.

CDR dataset, Mistral-7B-Instruct and Zephyr-7B-Beta show that a larger output of triples does not necessarily equate to higher TS or lower US. While Zephyr-7B-Beta produces more off-topic triples than Mistral-7B-Instruct, it does not result in more repetitive content. This highlights the importance of evaluating each metric independently.

Figure 4 shows the GRE task performance of five leading LLMs tested with five random seeds on the Wiki20m dataset. The results demonstrate the models’ high-quality generation and the effectiveness of our multi-dimensional evaluation framework. Notably, the models’ consistent performance across different runs validates our nuanced evaluation metrics, highlighting their robustness in assessing GRE model performance.

Figure 5 showcases the Elo Rating (Elo and Sloan, 1978) results of 100 samples from Wiki20m dataset via human annotation and our proposed GENRES. In most cases, the model ranks by GEN-RES are consistent with human annotators. We also evaluate the consistency between human annotators using the tie-discounted accuracy (Gao et al., 2023a). We find the following agreement scores: Topical Similarity 81.0%, Uniqueness 93.0%, Factualness 82.7%, Granularity 92.7 %, and Completeness 88.2%. These results showcase the consistency between the human annotators. More details of human evaluation can be found in Appendix D.

## 5 Related Works

Open RE. Open RE uncovers new relation types in unsupervised open-domain corpora, traditionally through tagging-based and clustering-based approaches. Tagging-based Open RE treats the task as sequence labeling, extracting relational phrases from sentences (Jia et al., 2019; Cui et al., 2018; Stanovsky et al., 2018), while clustering-based methods utilize external linguistic tools to featurerich relations and cluster them into distinct types (Zhou et al., 2023b; Marcheggiani and Titov, 2016;

ElSahar et al., 2017). With the rapid development of LLMs, recent work has demonstrated the effectiveness of LLMs in Open RE from a generative perspective (Wadhwa et al., 2023b; Li et al., 2023a). Our proposed GENRES focuses on Generative RE, bridging the existing gap in evaluating Open Generative RE techniques.

Generative RE. Generative models have exhibited significant promise in the field of RE (Wadhwa et al., 2023b; Wan et al., 2023; Li et al., 2023a). Sequence-to-sequence models such as BART (Lewis et al., 2020) were utilized to extract triples from input texts (Ni et al., 2022; Paolini et al., 2021; Cabot and Navigli, 2021). Then, LLMs were proved to be able to make zero-shot and fewshot generative RE without fine-tuning (Wadhwa et al., 2023b; Li et al., 2023a). Specifically, Wadhwa et al. (2023b) compared GPT-3 (Brown et al., 2020) and FLAN-T5 (Chung et al., 2022) to fully supervised RE methods and identified LLMs reach comparable performance in the zero-shot setup. However, existing GRE methods still rely on a predefined set of relations and entities similar to traditional RE. In this paper, we explore a more open setting and propose a unified evaluation framework GENRES applicable to all types of generative RE.

Evaluation for Text Generation. The evaluation of text generation quality is central to benchmarking the performance of LLMs. While traditional metrics like BLEU (Papineni et al., 2002) and ROUGE (Lin, 2004) assess surface-level word matching, they often inadequately capture the quality of the generated text. BERTScore (Zhang et al., 2019) focuses on semantic similarity, but still missing the multifaceted nature of text generation. Recently, LLMs have been utilized to evaluate text generation quality, such as FActScore (Min et al., 2023) on verifying the factualness, and UniEval (Zhong et al., 2022) on multi-aspect evaluation. In addition, GPTScore (Fu et al., 2023) utilizes LLMs for token-level probability analysis, enhancing flexibility in text assessment. Recent studies (Liu et al., 2023; Gao et al., 2023b; Li et al., 2023b) explore prompting-based multi-aspect evaluation, broadening the scope of evaluation methods. Unlike all the above works, our GENRES is the first metric designed specifically for Generative RE tasks.

## 6 Conclusions

In this paper, we introduced GENRES, a framework for evaluating Generative Relation Extraction using Large Language Models, marking a significant shift in the NLP field. Our findings based on extensive tests highlight the potential of LLMs to transform relation extraction and set the stage for future research, potentially revolutionizing information extraction processes and applications across various domains.

## 7 Acknowledgments

This work was supported in part by US DARPA KAIROS Program No. FA8750-19-2-1004 and IN-CAS Program No. HR001121C0165, National Science Foundation IIS-19-56151, and the Molecule Maker Lab Institute: An AI Research Institutes program supported by NSF under Award No. 2019897, and the Institute for Geospatial Understanding through an Integrative Discovery Environment (I-GUIDE) by NSF under Award No. 2118329. The work was also supported by NSF award SCH-2205289, SCH-2014438, and IIS-2034479.

## References

Elena Baralis, Luca Cagliero, Naeem Mahoto, and Alessandro Fiori. 2013. Graphsum: Discovering correlations among multiple terms for graph-based summarization. Information Sciences, 249:96–109.

David M Blei, Andrew Y Ng, and Michael I Jordan. 2003. Latent dirichlet allocation. Journal ofmachine Learning research, 3(Jan):993–1022.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Pere-Lluís Huguet Cabot and Roberto Navigli. 2021. REBEL: relation extraction by end-to-end language generation. In Findings of the Association for Computational Linguistics: EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 16-20 November,

2021, pages 2370–2381. Association for Computational Linguistics.

Sihao Chen, Daniel Khashabi, Wenpeng Yin, Chris Callison-Burch, and Dan Roth. 2019. Seeing things from a different angle:discovering diverse perspectives about claims. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 542–557, Minneapolis, Minnesota. Association for Computational Linguistics.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Y. Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022. Scaling instruction-finetuned language models. CoRR, abs/2210.11416.

Lei Cui, Furu Wei, and Ming Zhou. 2018. Neural open information extraction. In Proceedings of the 56th Annual Meeting ofthe Associationfor Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 2: Short Papers, pages 407–413. Association for Computational Linguistics.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. Qlora: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314.

Arpad E Elo and Sam Sloan. 1978. The rating of chessplayers: Past and present. (No Title).

Hady ElSahar, Elena Demidova, Simon Gottschalk, Christophe Gravier, and Frédérique Laforest. 2017. Unsupervised open relation extraction. In The Semantic Web: ESWC 2017 Satellite Events - ESWC 2017 Satellite Events, Portorož, Slovenia, May 28 - June 1, 2017, Revised Selected Papers, volume 10577 of Lecture Notes in Computer Science, pages 12–16. Springer.

Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. 2023. Gptscore: Evaluate as you desire. arXiv preprint arXiv:2302.04166.

Kaiyuan Gao, Sunan He, Zhenyu He, Jiacheng Lin, QiZhi Pei, Jie Shao, and Wei Zhang. 2023a. Examining user-friendly and open-sourced large gpt models: A survey on language, multimodal, and scientific gpt models. arXiv preprint arXiv:2308.14149.

Mingqi Gao, Jie Ruan, Renliang Sun, Xunjian Yin, Shiping Yang, and Xiaojun Wan. 2023b. Human-like summarization evaluation with chatgpt. arXiv preprint arXiv:2304.02554.

Xu Han, Tianyu Gao, Yankai Lin, Hao Peng, Yaoliang Yang, Chaojun Xiao, Zhiyuan Liu, Peng Li, Jie Zhou, and Maosong Sun. 2020. More data, more relations,

more context and more openness: A review and outlook for relation extraction. In Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Associationfor Computational Linguistics and the 10th International Joint Conference on Natural Language Processing, pages 745–758, Suzhou, China. Association for Computational Linguistics.

Xu Han, Tianyu Gao, Yuan Yao, Deming Ye, Zhiyuan Liu, and Maosong Sun. 2019. OpenNRE: An open and extensible toolkit for neural relation extraction. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP): System Demonstrations, pages 169–174, Hong Kong, China. Association for Computational Linguistics.

Xu Han, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan Yao, Zhiyuan Liu, and Maosong Sun. 2018. Fewrel: A large-scale supervised few-shot relation classification dataset with state-of-the-art evaluation. arXiv preprint arXiv:1810.10147.

Shibo Hao, Bowen Tan, Kaiwen Tang, Bin Ni, Xiyan Shao, Hengzhe Zhang, Eric Xing, and Zhiting Hu. 2023. BertNet: Harvesting knowledge graphs with arbitrary relations from pretrained language models. In Findings of the Association for Computational Linguistics: ACL 2023, pages 5000–5015, Toronto, Canada. Association for Computational Linguistics.

Shengbin Jia, Shijia E, and Yang Xiang. 2019. Supervised neural models revitalize the open relation extraction. CoRR, abs/1908.01761.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023a. Mistral 7b. CoRR, abs/2310.06825.

Kelvin Jiang, Ronak Pradeep, and Jimmy Lin. 2021. Exploring listwise evidence reasoning with t5 for fact verification. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 402–410, Online. Association for Computational Linguistics.

Pengcheng Jiang, Cao Xiao, Adam Cross, and Jimeng Sun. 2023b. Graphcare: Enhancing healthcare predictions with personalized knowledge graphs. arXiv preprint arXiv:2305.12788.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics,

ACL 2020, Online, July 5-10, 2020, pages 7871–7880. Association for Computational Linguistics.

Guozheng Li, Peng Wang, and Wenjun Ke. 2023a. Revisiting large language models as zero-shot relation extractors. arXiv preprint arXiv:2310.05028.

Jiao Li, Yueping Sun, Robin J Johnson, Daniela Sciaky, Chih-Hsuan Wei, Robert Leaman, Allan Peter Davis, Carolyn J Mattingly, Thomas C Wiegers, and Zhiyong Lu. 2016. Biocreative v cdr task corpus: a resource for chemical disease relation extraction. Database, 2016.

Ruosen Li, Teerth Patel, and Xinya Du. 2023b. Prd: Peer rank and discussion improve large language model based evaluations. arXiv preprint arXiv:2307.02762.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. Gpteval: Nlg evaluation using gpt-4 with better human alignment. arXiv preprint arXiv:2303.16634.

Yike Liu, Tara Safavi, Abhilash Dighe, and Danai Koutra. 2018. Graph summarization methods and applications: A survey. ACM Comput. Surv., 51(3).

Diego Marcheggiani and Ivan Titov. 2016. Discretestate variational autoencoders for joint discovery and factorization of relations. Trans. Assoc. Comput. Linguistics, 4:231–244.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In EMNLP.

Sameh K Mohamed, Vít Novácek, and Aayah Nounu.ˇ 2020. Discovering protein drug targets using knowledge graph embeddings. Bioinformatics, 36(2):603– 610.

Jian Ni, Gaetano Rossiello, Alfio Gliozzo, and Radu Florian. 2022. A generative model for relation extraction and classification. CoRR, abs/2202.13229.

OpenAI. 2023. GPT-4 technical report. CoRR, abs/2303.08774.

Giovanni Paolini, Ben Athiwaratkun, Jason Krone, Jie Ma, Alessandro Achille, Rishita Anubhai, Cícero Nogueira dos Santos, Bing Xiang, and Stefano Soatto. 2021. Structured prediction as translation between augmented natural languages. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th Annual Meeting of the Association for Computational Linguistics, July 6-12, 2002, Philadelphia, PA, USA, pages 311–318. ACL.

Heiko Paulheim. 2017. Knowledge graph refinement: A survey of approaches and evaluation methods. Semantic web, 8(3):489–508.

Gabriel Stanovsky, Julian Michael, Luke Zettlemoyer, and Ido Dagan. 2018. Supervised open information extraction. In Proceedings of the 2018 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2018, New Orleans, Louisiana, USA, June 1-6, 2018, Volume 1 (Long Papers), pages 885–895. Association for Computational Linguistics.

Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, and Robert Stojnic. 2022. Galactica: A large language model for science. arXiv preprint arXiv:2211.09085.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, and Thomas Wolf. 2023. Zephyr: Direct distillation of lm alignment.

Somin Wadhwa, Silvio Amir, and Byron Wallace. 2023a. Revisiting relation extraction in the era of large language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15566– 15589, Toronto, Canada. Association for Computational Linguistics.

Somin Wadhwa, Silvio Amir, and Byron C. Wallace. 2023b. Revisiting relation extraction in the era of large language models. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 15566– 15589. Association for Computational Linguistics.

Zhen Wan, Fei Cheng, Zhuoyuan Mao, Qianying Liu, Haiyue Song, Jiwei Li, and Sadao Kurohashi.

2023. Gpt-RE: In-context learning for relation extraction using large language models. arXiv preprint arXiv:2305.02105.

Guan Wang, Sijie Cheng, Xianyuan Zhan, Xiangang Li, Sen Song, and Yang Liu. 2023. Openchat: Advancing open-source language models with mixed-quality data.

Hongwei Wang, Fuzheng Zhang, Xing Xie, and Minyi Guo. 2018. Dkn: Deep knowledge-aware network for news recommendation. In Proceedings of the 2018 World Wide Web Conference, WWW ’18, page 1835–1844, Republic and Canton of Geneva, CHE. International World Wide Web Conferences Steering Committee.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. 2023. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244.

Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin, Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou, and Maosong Sun. 2019. Docred: A large-scale document-level relation extraction dataset. arXiv preprint arXiv:1906.06127.

Xiangxiang Zeng, Xinqi Tu, Yuansheng Liu, Xiangzheng Fu, and Yansen Su. 2022. Toward better drug discovery with knowledge graph. Current opinion in structural biology, 72:114–126.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675.

Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. 2023. Siren’s song in the ai ocean: A survey on hallucination in large language models.

Yuhao Zhang, Victor Zhong, Danqi Chen, Gabor Angeli, and Christopher D Manning. 2017. Position-aware attention and supervised data improve slot filling. In Conference on Empirical Methods in Natural Language Processing.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685.

Ming Zhong, Yang Liu, Da Yin, Yuning Mao, Yizhu Jiao, Pengfei Liu, Chenguang Zhu, Heng Ji, and Jiawei Han. 2022. Towards a unified multidimensional evaluator for text generation. arXiv preprint arXiv:2210.07197.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. 2023a. LIMA: less is more for alignment. CoRR, abs/2305.11206.

Jie Zhou, Shenpo Dong, Yunxin Huang, Meihan Wu, Haili Li, Jingnan Wang, Hongkui Tu, and Xiaodong Wang. 2023b. U-CORE: A Unified Deep Clusterwise Contrastive Framework for Open Relation Extraction. Transactions of the Association for Computational Linguistics, 11:1301–1315.

Yuqi Zhu, Xiaohan Wang, Jing Chen, Shuofei Qiao, Yixin Ou, Yunzhi Yao, Shumin Deng, Huajun Chen, and Ningyu Zhang. 2023. Llms for knowledge graph construction and reasoning: Recent capabilities and future opportunities.

## A Limitations, Ethics, and Risks

## A.1 Limitations

LLMs as Evaluators. Within GENRES, we employ the GPT-3.5-Turbo-Instruct large language model (LLM) for assessing the factualness and granularity of extracted relationship triples. However, challenges arise when the LLM delivers incorrect evaluations, particularly in instances where information is overly implicit, misleading, debatable (Chen et al., 2019), or when the model encounters its inherent hallucination issues (Zhang et al., 2023). To mitigate these problems, potential solutions include instructing the model to detail its reasoning process leading to a prediction (Wei et al., 2022), or applying ensemble methods (Li et al., 2023a) to determine the most likely answer. These approaches are areas of interest for our future research endeavors.

Unfocused Extraction by Open GRE. Our research champions the Open Generative Relation Extraction (Open GRE) paradigm, which motivates LLMs to harvest a broader array of relationships, unconstrained by specific relation or entity types. While this approach has demonstrated enhanced topical breadth and factual content in extractions, it also results in a less focused extraction process compared to traditional methods like closed GRE and semi-open GRE (Wadhwa et al., 2023b; Li et al., 2023a). For instance, in constructing a Knowledge Graph (KG) for medical question answering, certain extractions, such as the triple John, age, 16 , might be irrelevant and hence undesirable for inclusion in the KG. However, we posit that an intermediary layer, such as post-processing, should exist between Relation Extraction (RE) and downstream applications. This step would serve to refine and tailor the extracted relationships to meet specific requirements, aligning with methodologies proposed in existing literature (Paulheim, 2017; Liu et al., 2018). Moreover, our GENRES framework is versatile enough to assess all forms of GRE, with the Open GRE configuration, noted for its flexibility, serving as a particularly effective benchmark for evaluating the robustness of our approach.

## A.2 Ethics and Risks

All datasets used in this study, namely CDR (Li et al., 2016), DocRED (Yao et al., 2019), NYT10m (Han et al., 2019), Wiki20m (Han et al., 2019), TA-CRED (Zhang et al., 2017), and Wiki80 (Han et al.,

2018) are publicly available. This transparency minimizes ethical concerns related to data sourcing and usage.

Additionally, the interpretability and transparency of LLM decision-making processes are paramount, particularly in contexts involving sensitive or personal data. Recognizing the limitations and error tendencies of LLMs, including occasional information inaccuracies, we emphasize the importance of reliability in our evaluation methods. Furthermore, the integration of LLMs as evaluators impacts traditional human roles, calling for a careful examination of the ethical implications of labor displacement. Lastly, the potent capabilities of LLMs underscore the need for responsible use and measures to prevent misuse, aligning our research with high ethical standards and societal well-being. We carefully checked and ensured that there is no offensive information contained in the data we used as the input to any LLMs.

## B Templates for Prompting LLMs

## B.1 Templates for Generative Relation Extraction

We delineate the structured prompts and demonstrations utilized in our generative relation extraction methodology. The templates are devised to prime the model for precise and contextually relevant relationship extraction from textual data across different domains and levels of granularity.

General Instruction : The model is instructed to identify relationships between entities, with the aim to extract both intra-sentence and inter-sentence relational triples. This ensures a comprehensive understanding of the text, reflecting the intricacies of document-level nuances and the succinctness of sentence-level information.

LLaMA-2 Model Instruction: An additional directive is provided to the LLaMA-2 model to maintain output stability. The goal is to have the model generate a consistent list of triples, avoiding any extraneous information that does not contribute to the relationship representation.

Demonstration Examples: Examples are tailored to the general and biomedical domains to pre-heat the model towards the target topics. This stratagem is intended to: (1) Facilitate the model’s adaptation to the domain-specific language and context, thus enabling more accurate and relevant extractions. (2) Encourage the model to discern and replicate the desired output structure from the examples, which is crucial for reliable relationship extraction.

<table><tr><td>Hyperparameter</td><td>Values</td></tr><tr><td>LDA latent topics</td><td></td></tr><tr><td>CDR</td><td>{20, 30, 40, 50, 60, 70, 80, 90, 100}</td></tr><tr><td>DocRED</td><td>{30, 50, 70, 100, 150}</td></tr><tr><td>NYT10m</td><td>{50, 100, 150, 200, 250}</td></tr><tr><td>Wiki20m</td><td>{50, 100, 150, 200, 250}</td></tr><tr><td>TACRED</td><td>{100, 150, 200, 250, 300}</td></tr><tr><td>Wiki80</td><td>{100, 150, 200, 250, 300}</td></tr><tr><td>Triple similarity threshold φ</td><td>{0.85, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98}</td></tr><tr><td>Open-source LLMs-related</td><td></td></tr><tr><td>max_new_tokens floating-point number</td><td>min[#token_limit, {3, 5, 6, 7, 8, 9, 10} *#input_tokens] float16</td></tr><tr><td></td><td></td></tr><tr><td>GPT-related</td><td></td></tr><tr><td>max_new_tokens</td><td>800</td></tr><tr><td>temperature</td><td>0.3</td></tr></table>

Table 5: Hyperparameters Tuning. We highlight the optimal ones based on our experiments in bold.

The provided demonstrations span a variety of contexts and exemplify the format in which the relationships should be presented. The clear and topicoriented examples aim to fine-tune the model’s performance, ensuring it can navigate the complexities of relation extraction with precision across both biomedical and general domains.

## B.2 Template for Factualness Verification

In the context of evaluating the factual accuracy of information extracted by language models, we present our template for factualness verification in Figure 8. Utilizing GPT-3.5-Turbo-Instruct as the language model evaluator, our template is designed to solicit a binary output: “true” if the relationship (triplet) is factually correct, “false” otherwise, based solely on the information entailed in the source text.

The template is constructed with three examples, each serving a specific purpose to calibrate the model’s understanding of factual correspondence: Example 1 establishes the model’s ability to recognize direct factual statements that are explicitly stated in the source text. Example 2 tests the model’s discernment of geographical facts and common knowledge, challenging it to detect misinformation. Example 3 assesses the model’s capacity to correctly interpret narrative contexts and character relationships, a more subtle and complex form of factual verification.

The inclusion of these examples in the template aims to ensure that the model is thoroughly vetted across a spectrum of factual verification scenarios ranging from straightforward fact-checking to the interpretation of literary works.

## B.3 Template for Granularity Checking

For granularity checking, we employ the template shown in Figure 9. The template contains 9 examples, to teach the LLM (GPT-3.5-Turbo-Instruct) what triples can be further split and what are not. Explanations are required when a triple cannot be split (GS = 0).

## C Hyper-parameter Tuning

The process of hyper-parameter tuning is crucial for optimizing the performance of our models. Table 5 presents a comprehensive list of the hyperparameters adjusted during our experiments. This includes the number of latent topics for LDA, various dataset-specific parameters, and thresholds for triple similarity. Furthermore, specific parameters related to open-source LLMs and GPT-related configurations are tuned to enhance model efficiency and output quality.

## D Human Evaluation

We further conducted human evaluation experiments to verify the alignment of our proposed Gen-RES with human preferences. Three annotators, who are all computer science graduate students, are involved in this evaluation.

## D.1 Evaluation Setup

Our setup for human evaluation follows the approach detailed in studies such as Gao et al. (2023a), Zhou et al. (2023a), and Dettmers et al. (2023). We adopt a pairwise comparison method for assessing model outputs. This approach simplifies the evaluation process by requiring human annotators to choose the better result from a pair of options. The evaluation was performed using 100 samples from the Wiki20m dataset. In this process, for each score proposed in Section 3, three human annotators compared the output relationships from Groundtruth, LLaMA-2-70b, OpenChat, and GPT-4-Turbo in pairs, leading to three possible outcomes for each pair: model A being superior, model B being superior, or a tie. Subsequently, we apply the Elo rating (Elo and Sloan, 1978) system to score the final results.

Elo Rating. Elo rating, initially established as a prevalent system for assessing player skill in chess and various competitive games, has recently been adapted to evaluate LLMs<sup>5</sup> (Gao et al., 2023a; Zhou et al., 2023a; Dettmers et al., 2023). Its adaptability, characterized by features such as scalability and incremental adjustment, makes it particularly suitable for this purpose. This innovative use of the Elo rating system offers a robust quantitative framework for comparing the performance of various LLMs. In our pairwise comparison setup, the outcome of each comparison impacts the models’ scores: a tie results in no change in scores, while a victory leads to an increase in the winner’s score and a decrease in the loser’s score. Following the completion of all comparisons, the Elo Rating system outputs a final score for each model, thereby establishing their relative rankings based on performance.

Instructions for Annotators. The instructions for annotators are shown in Figure 6. Annotators should evaluate the outputs from five aspects in Section 3. During the evaluation process, the models are anonymous for annotators. It should be noted that Completeness is measured after all other metrics have been assessed to prevent the leakage of ground truth information to annotators.

Inter-Annotator Agreement. To evaluate Inter-Annotator Agreement with tie-discounted accuracy, we randomly select 50 samples from the 100 Wiki20m samples, resulting in a total of 1500 overlap pairs for two human annotators. This process aimed to assess the consistency level between annotators, anticipating a significant alignment in their evaluations. For the final scoring, we merged all the annotations. The scoring protocol for merging is as follows: (1) When both annotators’ responses were in agreement, this consensus was accepted as the merged result. (2) If one annotator declared a tie, the decision of the other was taken as the final annotation. (3) If one annotator believed that ’model A wins’ and the other that ’model B wins, the models were considered tied.

![](images/fc5b97c839247f422c47c5323202ff9b54531f8a5e4adb834c31904dbfc318f7.jpg)  
Figure 6: Instruction for Human Annotators.

![](images/47fe9ae4399f757ac7cdfbf7c367875593cc14785af9fd0944ab2ecbb839df25.jpg)  
Figure 7: Templates used for Open Generative Relation Extraction.

![](images/64255c6e1bf807daa946506635ceac1ad52bc3494a57d769d9b5d29114a835e0.jpg)  
Figure 8: Template for Factualness Verification.

Evaluate the given triple for its potential to be split into more specific sub-triples. Provide the   
sub-triples in the format [e, r, o] and give the total count. If no split is necessary, explain   
briefly.   
Example 1:   
Triple: ["text messaging", "has popularized", "the use of abbreviations"]   
Sub-triples: N/A (The triple is already specific and cannot be broken down further.)   
Granularity: 0   
Example 2:   
Triple: ["electric cars", "offer benefits like", "energy efficiency and environmental friendliness"]   
Sub-triples:   
["electric cars", "offer benefits like", "energy efficiency"]   
["electric cars", "offer benefits like", "environmental friendliness"]   
Granularity: 2   
Example 3:   
Triple: ["exercise", "boosts", "health"]   
Sub-triples: N/A (The relationship is direct and does not need further granularity.)   
Granularity: 0   
Example 4:   
Triple: ["trees", "provide", "oxygen, shade, and habitats"]   
Sub-triples:   
["trees", "provide", "oxygen"]   
["trees", "provide", "shade"]   
["trees", "provide", "habitats"]   
Granularity: 3   
Example 5:   
Triple: ["healthy diet", "contributes to", "wellness"]   
Sub-triples: N/A (The term 'wellness' encompasses a broad range of aspects, which are implicitly   
understood.)   
Granularity: 0   
Example 6:   
Triple: ["water", "exists as", "solid, liquid, gas"]   
Sub-triples:   
["water", "exists as", "solid"]   
["water", "exists as", "liquid"]   
["water", "exists as", "gas"]   
Granularity: 3   
Example 7:   
Triple: ["urbanization", "leads to", "various social and environmental changes"]   
Sub-triples:   
["urbanization", "leads to", "social changes"]   
["urbanization", "leads to", "environmental changes"]   
Granularity: 2   
Example 8:   
Triple: ["global warming", "causes", "climate change and associated phenomena like sea-level rise"]   
Sub-triples:   
["global warming", "causes", "climate change"]   
["global warming", "causes", "sea-level rise"]   
Granularity: 2   
Example 9:   
Triple: ["antibiotics", "treat", "bacterial infections"]   
Sub-triples: N/A (The triple is specific, conveying a singular relation between antibiotics and   
bacterial infections.)   
Granularity: 0   
Prompt:   
Triple: \$TRIPLE\$   
Sub-triples:  
Figure 9: Template for Granularity Checking.