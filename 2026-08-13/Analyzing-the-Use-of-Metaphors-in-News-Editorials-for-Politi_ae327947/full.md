# Analyzing the Use of Metaphors in News Editorials for Political Framing

Meghdut Sengupta<sup>1</sup> Roxanne El Baff <sup>2</sup>

Milad Alshomary<sup>3</sup> ∗ Henning Wachsmuth<sup>1</sup>

Leibniz Universität Hannover, Germany<sup>1</sup> German Aerospace Center (DLR), Germany<sup>2</sup>

Columbia University, USA<sup>3</sup>

{m.sengupta, h.wachsmuth}@ai.uni-hannover.de<sup>1</sup> roxanne.elbaff@dlr.de<sup>2</sup>, ma4608@columbia.edu<sup>3</sup>

## Abstract

Metaphorical language is a pivotal element in the realm of political framing. Existing work from linguistics and the social sciences provides compelling evidence regarding the distinctiveness of conceptual framing for political ideology perspectives. However, the nature and utilization of metaphors and the effect on audiences of different political ideologies within political discourses are hardly explored. To enable research in this direction, in this work we create a dataset, originally based on news editorials and labeled with their persuasive effects on liberals and conservatives and extend it with annotations pertaining to metaphorical usage of language. To that end, first, we identify all single metaphors and composite metaphors. Secondly, we provide annotations of the source and target domains for each metaphor. As a result, our corpus consists of 300 news editorials annotated with spans of texts containing metaphors and the corresponding domains of which these metaphors draw from. Our analysis shows that liberal readers are affected by metaphors, whereas conservatives are resistant to them. Both ideologies are affected differently based on the metaphor source and target category. For example, liberals are affected by metaphors in the Darkness & Light (e.g., death) source domains, where as the source domain of Nature affects conservatives more significantly.

## 1 Introduction

Metaphorical language is dominant in various domains of everyday life (Lakoff and Johnson, 2020). Particularly, politicians widely frame their discourses by means of employing metaphors (Lakoff, 1995; Mio, 1997; Chilton and Ilyin, 1993; Charteris-Black, 2009).

Inspired from the aforementioned research regarding metaphorical usages in political domains, studies related to modelling metaphors with respect to political framing has been investigated like Cabot et al. (2020). In their work, they modelled a multitask learning setup to predict political perspective of news articles (Baly et al., 2020), party affiliations of politicians (Biessmann et al., 2016), and framing dimensions of socioeconomic issues such as gun-control (Card et al., 2015) as the main task, while predicting metaphors (Gong et al., 2020) and emotions (Troiano et al., 2023) act as the auxiliary tasks.

![](images/dc57c769ecd9f9372a73a90d647e1e891b1977adbb84b7e7fd5a8b225ca71de8.jpg)  
Figure 1: Text snippet from of our data from the corpus: Single and composite metaphors are marked in red and blue respectively. For each editorial we have two political categories, namely liberal and conservative. Each of the two categories can have either No effect, challenging, or reinforcing effect on the reader

To that end they employ five different datasets from the domains of political bias (Li and Goldwasser, 2019), political affiliation (Li and Goldwasser, 2019), framing (Card et al., 2015), metaphors (Steen, 2010), and emotions labelled on tweet data (Mohammad et al., 2018).

While different areas of language involve different degrees of metaphor usage, news editorials utilize metaphors extensively as a rhetorical device to better reach their audiences (Farrokhi and Nazemi, 2015). Recent work by Joseph et al. (2023), among others, draws on annotations of metaphors in content related to politics and others, but only with news headlines as data.

Hence, besides the need of having a dataset at the intersection of metaphors with political discourses and framing, an under-explored dimension in this context is the persuasive effect of a political discourse on the readers. El Baff et al. (2018) determines persuasive effect of an editorial, based on whether the editorial challenged the reader’s stance by making them rethink it, reinforced their stance by helping them argue better, or was ineffective . In their work, for each ideology - liberal and conservative - there are three effects of persuasiveness - reinforcing, challenging, and no effect.

This paper explores where and how metaphors are used for political framing concerning persuasiveness, focusing on news editorials. We do this in conjunction with employing the conceptual domains in metaphorical meaning construction (Lakoff and Johnson, 2008), namely, the source - the concept domain from where the meaning is derived in a metaphorical sentence and target domain - the concept domain which is explained by metaphor in the sentence (see section 2).

To that end, we re-annotate an existing corpus of news editorials from the New York Times (El Baff et al., 2018) by two levels of annotations. Firstly, we identify and annotate the metaphors with the metaphor identification procedure by Steen (2010), and secondly, we annotate each metaphor with their source and target domains, which we reuse from the work by Shutova et al. (2010) and Gordon et al. (2015).

We do two pilot studies each with 3 samples of editorials chosen randomly from the corpus by El Baff et al. (2018) with two annotators. For the main study, we employed an additional annotator. Hence a total of 3 annotators carried out the main annotation on 300 randomly sampled news editorials from the same corpus. The average number of paragraphs of each editorial is 7.3. The total number of metaphors present in the corpus are 12006 with 8353 single metaphors (70 %) and 3653 composite metaphors (30 %). We have an overall 84 source domains and 59 target domains distributed across out dataset. Figure 1 shows a snippet of a sample annotation.

Our findings suggest that the political stance of liberals related to political ideologies are affected significantly by metaphorical language used in the news editorials than conservatives. For analysing the effect of source and target domains on the political stance by these two ideologies, we cluster the source and target domains into 14 ontological categories based on Gordon et al. (2015). Our results based on this analysis indicate that the metaphors and source domains have adverse persuasive effects on the political stance of the liberals, especially from the categories of Nature, and Darkness & Light.

In summary, our main contributions are as follows:

• We create a dataset to study metaphorical usages in political news editorials.

• We analyze the relationship of the metaphorical usage in news editorials with respect to the perceived effect (reinforcing, challenging, no-effect) by conservative and liberal readers by answering the following two questions: (1) Does the usage of metaphors affect readers with different ideologies differently? (2) Which source domains correlate to the persuasive effect of editorials perceived by each ideology?

## 2 Related Work

Metaphors are a linguistic tool used to convey implicit meaning, while their literal counterparts in the context of a sentence are explicit. This gives rise to a phenomenon known as meaning shift (Lakoff and Johnson, 2008). A metaphor, represented by a specific word or phrase in a sentence, facilitates this shift. It involves extracting meaning from an abstract conceptual domain, referred to as the source domain, and projecting it onto a concrete conceptual domain, referred to as the target domain. Consider this metaphorical sentence:

“Wages have been stagnating through much of the current economic cycle.”

Here, “stagnating” is metaphorical, since stagnation as a concept cannot be experienced by a nonphysical entity such as wage in the physical realms. The intended meaning is manifested by establishing a mapping of two conceptual domains, drawing the meaning from a source domain (movement) and projecting it into a target domain (money).

Foundational concepts such as source and target domains have been considered in the development of the subsequent set of datasets (Shutova et al., 2010; Chakrabarty et al., 2021; Zhang et al., 2021). Recently, Sengupta et al. (2022) used contrastive learning to predict the source domains of metaphors, and Sengupta et al. (2023) extended this by modelling the aspects highlighted by metaphors, in a given context, in a multitask setting. Their work showed that for predicting highlighted aspects of metaphors, incorporating the information of the source domains of the concerned metaphor improves model performance and vice versa.

However, their work is a first step towards interpretation of metaphors via exploiting metaphorical components and limits itself from establishing a direct connection of the conceptual domains to a real-world setting (like political discourses). Unlike their work, our dataset provides a groundwork for establishing a direct connection of the components constructing metaphorical meaning to persuasiveness in political discourses.

The task of metaphor detection, aiming to distinguish between literal and metaphorical uses of specific words within sentences, has been studied in NLP research extensively (Shutova et al., 2010). To that end, various datasets have been utilized in the course of this research line (Steen, 2010; Gordon et al., 2015; Mohler et al., 2016; Do Dinh et al., 2018).

Our annotation guideline builds on top of the metaphor identification system initially designed to construct the VU Amsterdam Metaphor Corpus (Steen, 2010) to identify the metaphors in the dataset. To annotate the source and target domains, we reuse the same provided by Shutova et al. (2010) and Gordon et al. (2015) in their work. So for the annotation of the source and target domains in our work, we combined the domains provided by them and provided the same to our annotators.

A number of works in psycholinguistics have explored how metaphors are used in framing different conversational domains (Semino et al., 2018; Cornelissen et al., 2011; Luokkanen et al., 2014; Joris et al., 2014). In the political domain specifically, Brugman et al. (2017) and Boeynaems et al. (2017) explore the role of metaphorical framing for persuasion.

Our work complements existing resources that brings together metaphors with persuasive effects, as stated in Section 1. Previous NLP research in this context has studied how cognitive traits of readers such as personality and prior beliefs impact persuasive effects (Lukin et al., 2017; Durmus and Cardie, 2018). Others looked at the interplay of the characteristics of debaters in persuasive argumentation (Al Khatib et al., 2020). For editorials,

<table><tr><td>Effect</td><td>Liberal</td><td>Conservative</td></tr><tr><td>No effect</td><td>70</td><td>201</td></tr><tr><td>Reinforcing</td><td>131</td><td>68</td></tr><tr><td>Challenging</td><td>99</td><td>31</td></tr><tr><td>Total</td><td>300</td><td>300</td></tr></table>

Table 1: Distribution of persuasive effects over the two political idelogies (liberal, conservative) in our corpus.

El Baff et al. (2020a) studied correlations between the impact of arguments and the the reader’s personality and political ideology. Since linguistic style is largely encapsulated by the usage of metaphorical language, our analysis draws comparisons from the work of El Baff et al. (2020b) who study the importance of the writing style of news editorials for the persuasion of readers with different ideologies.

## 3 Data

This section presents the construction of our news editiorial corpus annotated for metaphors. We start with describing the source data, before explaining the annotation task and guidelines. Then, we describe the annotation process, consisting of two pilot studies and one main annotation phase.

## 3.1 Source Data

As a basis for annotating metaphors, we use the publicly available Webis-News-Editorials18 dataset (El Baff et al., 2018) <sup>1</sup>. This corpus contains ideology-specific annotations of the perceived persuasive effect of 1000 New York Times editorials. In particular, three liberals and three conservative readers each annotated all editorials by reporting whether the editorial reinforced their stance by helping them argue better, challenged their stance by making them rethink it, or had no effect on them.

For our corpus, we randomly sampled 300 of the editorials. They have a mean length of 7.3 paragraphs per editorial. The total number of tokens in the sample is 161 132, that is, 537.1 tokens per editorial on average. The distribution of the frequency of the persuasiveness effects for the liberal and conservative ideologies are shown in Table 1.

## 3.2 Annotation Task

As outlined in Section 2, past research has predominantly involved the identification of metaphorical usage for single words. However, our inspection of a sample of the corpus revealed that multi-word metaphorical expressions are present in abundance. Hence, we decided to target metaphor identification on two sub-categories: single metaphor and composite metaphor. Notably, some composite metaphors, in theory, can be also argued to be idioms. However, for the uniformity of our work, we do not sub-categorize metaphorical usage down to idioms in this paper.

To the best of our knowledge, all existing datasets related to metaphors in NLP, restrict their annotation to single-word metaphors. Hence, our work is the first to include multi-word metaphors in the process of curating a dataset involving metaphorical language.

We asked the annotators to read news editorials from the New York Times and to annotate all textual segments (both single-word and multi-word) that encapsulate a utilization of metaphors. In addition, they were asked to identify the source and target domain of the respective metaphor annotated. This led to a two-level annotation task:

• Level 1. The first task was to identify all single and composite metaphors. A single metaphor is a one-word metaphor usually represented by a verb, for example, “He is drowning in money”. It may be an adjective or a noun as well, though, as in “a nourishing vacation” or “nation’s choice”, respectively. In contrast, a composite metaphor constructs metaphorical meaning with more than one word, as in this example: “The world has begun edging away from the dollar.”

• Level 2. For each metaphor identified, the annotator was then asked to determine the source and target domains of the metaphor from a set of pre-defined concept (source and target) domains. These domains represent the conceptual mapping of the meaning shift in metaphors, as stated in Section 2.

## 3.3 Annotation Guidelines

Following from the specified annotation task, our annotation guidelines consisted a procedure to identify metaphor as well as source and target domains, along with examples of the annotation task.

Procedure For single metaphors, we relied on the widely-used Metaphor Identification Procedure (MIPVU) by Steen (2010).<sup>2</sup> MIPVU summarizes the following steps to identify metaphors, which we reuse in our work:

• Read the text to get a general understanding of the meaning.

• Determine the lexical units.

• Establish the contextual meaning of the unit.

• Determine if it has a more basic meaning.

• Decide whether the contextual meaning contrasts with the basic meaning but can it be understood in comparison with it.

• If yes, mark the unit as metaphorical.

Composite metaphors consist of a pivot metaphorical word and a context window on the left, right, or both sides of the pivot metaphorical word. For identification, the following steps were devised:

• Find the pivot word, following MIPVU.

• Identify the context window: at least oneword and at most two words on the left, right, or both sides of the pivot word. This window was decided based upon discussing with the annotators after the first two pilot studies, before starting with the main annotations.

Previous work has listed a comprehensive number of source and target domains with respect to different sources of text (Shutova et al., 2010; Gordon et al., 2015). Especially in the work by the latter, one of the prominent sources of text in their dataset are news platforms. We combined the source and target domains of these two works and provided them to the annotators. We also kept the option open for the annotators to add a new source and/or target domain if they found the list we provided to contain insufficient number of candidates for source and target domains.

## 3.4 Annotation Tool

For our pilot studies and the main annotation, we employed Label Studio, an annotation platform designed to host and support crowdsourced annotations.<sup>3</sup> We customized the platform such that, for each article, an annotator can read the editorial and mark a span of text as single or composite metaphor simply by clicking either of the two buttons we provided in our customized tool. The annotators can then choose those source and target domains from the given set of domains that they deem to be the most suitable candidate for the given metaphorical expression in that sentential context.

## 3.5 Annotation Process

Given the complexity of the task, we carried out two phases of pilot studies with two annotators – both of them experts on the subject domain with previous knowledge about metaphorical meaning construction – before we proceeded to the main annotation study. The annotators were handed the annotation guidelines and briefed about the annotation procedure based on the guidelines.

Pilot Studies Each phase of the pilot study had three randomly sampled editorials from the original corpus. We calculated the agreement between the annotators based on span overlaps as follows:

• For single metaphors, we considered the annotators to agree, if there was an exact match of spans.

• For composite metaphors, we considered them to agree, if there was a full or a partial overlap of spans.

The first phase yielded an observed agreement of 66.7% for the identification of metaphors. We discussed conflicts that arose over the first pilot study with the annotators. Example disagreement covered cases such as whether to consider personification of institutions (e.g., “government”) as metaphors or whether the context of composite metaphors can span over more than two tokens. Additionally, the pilot study resulted in adding more frequent source and target domains, such as people. The second phase of the pilot study resulted in an observed agreement of 65.0%. Upon another discussion with the annotators, we added object to the list of both source and target domains and finalized the guidelines.

Main Study For the main annotation, we hired only one additional annotator who was trained on the revised guidelines. All three annotators then worked with the finalized version of the annotation guidelines <sup>4</sup>.

For the final annotations we calculated chancecorrected inter-annotator agreement and consolidated the final annotation in the following process:

1. For each editorial, we converted each annotation into BIO format (Ye et al., 2019), resulting in each token in the text having one of the following labels: $B _ { s i n g l e }$ for single metaphors, B<sub>composite</sub> and I<sub>composite</sub> for composite metaphors, and O if the token does not belong to any metaphor.

2. For each editorial, we calculated the pairwise average agreement in terms of Fleiss’ κ based on the tokens’ BIO tags of the three annotators (annotators 1 & 2, 2 & 3, and 1 & 3).

3. As final inter-annotator agreement, we report the average of the pairwise agreements of the three annotators over all the editorials below.

4. To infer the final annotation of each editorial, we take the annotations of the most reliable annotator. We define this annotator to be the one with the highest agreement with the other two on the corresponding editorial.

As a result, the average agreement scores of Annotators 1, 2, and 3 across all editorials were 0.41, 0.41, and 0.40 respectively. This moderate agreement reflects the subjectivity of this task, while underlining the consistency of the annotation and the uniform understanding of the guidelines by the annotators. Additionally, the majority agreement between the three annotators was 0.79, which shows that the consistency of the absolute agreement is maintained from the pilot studies. Detailed statistics of the corpus are shown in Table 2.

## 4 Dataset Analysis

In this section, we analyze how metaphor types (single or composite) and domains (source and target) correlate to editorials’ effect on readers with different ideologies by providing a numerical analysis. The dataset contains 12,006 metaphors across 300 news editorials, 70% of which are single and 30% composite metaphors (Table 2). El Baff et al. (2018) has additional annotations of three different topics - global, national, and state pertaining to the scope of the news editorials. Table 3 shows the count of single and composite metaphors based on the topic distribution of the editorials.

In the following sections, we apply the same methodology to identify significant differences between the following pairs of effects: Challenging vs. Reinforcing, Challenging vs. No Effect, or Reinforcing vs. No Effect. To achieve this, we initially conduct significance tests on a dependent variable, namely the metaphor ratio in Section 4.1 and the domain count in Section 4.2, across the three editorials’ effects (e.g., Challenging) to determine if there are any significant variations among these values. We employ the Anova test (if homogeneity and normality are met) or Kruskal (otherwise). Subsequently, if $\mathsf { p } < 0 . 0 5$ , we perform posthoc analysis, employing an independent t-test when normality is satisfied and Mann-Whitney otherwise, with Bonferroni correction for each effect pair. Additionally, we calculate the effect size r to quantify the strength of the observed significant differences.

<table><tr><td></td><td colspan="6">Conservative</td><td colspan="7">Liberal</td></tr><tr><td></td><td>All metaphors</td><td colspan="2">Single</td><td colspan="2">Composite</td><td colspan="2">All metaphors</td><td colspan="3">Single</td><td colspan="3">Composite</td></tr><tr><td>Effect</td><td>count ratio (std)</td><td>count ratio (std)</td><td></td><td>count ratio (std)</td><td></td><td>count ratio (std)</td><td></td><td></td><td>count ratio (std)</td><td></td><td></td><td></td><td>count ratio (std)</td></tr><tr><td>Challenging</td><td>1,349 0.13 (00 04)</td><td>928 0.06</td><td></td><td>421 0.08 (±0.04)</td><td>4,119</td><td>0.13</td><td></td><td>2,875</td><td>0.06</td><td></td><td>(00 02)</td><td>1,244</td><td>0.07</td></tr><tr><td>No effect</td><td>7,887 0.12</td><td>5,478 0.05</td><td>(00 02)</td><td>2,409 0.07</td><td>2,404</td><td>0.10</td><td>(+00 04)</td><td>1,592</td><td>0.04</td><td></td><td></td><td>812</td><td>(00 03) 0.06</td></tr><tr><td>Reinforcing</td><td>2,770 0.12</td><td>1,947 0.05</td><td></td><td>823 0.07</td><td>5,483</td><td>0.13</td><td></td><td>3,886</td><td>0.06</td><td></td><td></td><td>1,597</td><td>0.07</td></tr><tr><td>All Effects</td><td>12,006</td><td>8,353</td><td>3,653</td><td></td><td>12,006</td><td></td><td></td><td>8,353</td><td></td><td></td><td></td><td>3,653</td><td></td></tr></table>

Table 2: Total count and mean ratio of metaphorical tokens among all news editorials of all metaphors, single, and composite metaphors for conservative and liberal readers per effect (challenging, no effect and reinforcing). Ratios that are significantly different across the three effects are reported with p-value < 0.05 and 0.001

<table><tr><td>Topic</td><td># Single</td><td># Composite</td><td># Editorials</td></tr><tr><td>Global</td><td>2220</td><td>973</td><td>79</td></tr><tr><td>National</td><td>3455</td><td>1536</td><td>124</td></tr><tr><td>State</td><td>2678</td><td>1144</td><td>97</td></tr><tr><td>Total</td><td>8353</td><td>3653</td><td>300</td></tr></table>

Table 3: Count of single and composite metaphors over three topics (global, national, and state) in our corpus.

## 4.1 Metaphorical Effect on Ideology

Table 2 shows the count of metaphors for Conservative (left) and Liberal (right) readers for each effect (Challenging, Reinforcing and No Effect). As mentioned earlier, the distribution of the editorials over the effect is imbalanced for both ideologies. For conservatives, the editorials are distributed with 10%, 67%, and 23% for challenging, no effect, and reinforcing, and for liberals, 33%, 23%, and 44%. Simultaneously, for Conservatives, the number of metaphors in ineffective editorials is higher (7,887) than the effective ones (Challenging: 1,349, Reinforcing: 2.770). Meanwhile, for liberals, effective editorials have a higher (4,119 and 5,483) number of metaphors than ineffective (2,404). A similar distribution holds for single and composite

metaphors.

Additionally, we report the mean of the ratio of metaphorical tokens to total tokens (ratio), which ranges between 0.10 and 0.13 when considering both metaphor types and between 0.04 and 0.07 when considering single or composite (Table 2). We conduct the significance test explained earlier on the ratio.

Conservatives. We do not observe any significant differences across the three effects, suggesting that metaphors have no significant effect on conservatives. El Baff et al. (2020c) showed, using the same dataset, that conservative readers are not affected by the style features of an editorial but rather by the content, deducing that conservative readers are resistant to the style of a liberal newspaper (The New York Times). Our results complement previous findings since metaphors are also considered style features.

Liberals. On the contrary, Liberal readers are affected by metaphors. Overall, there is a significant difference across the three effects at $p < 0 . 0 0 1$ where Reinforcing and Challenging editorials have significantly higher metaphors than ineffective ones with a medium effect size of r = .35 and .29, respectively. The same observations hold for single metaphors (effect size r = .34 and r = .33 respectively). Whereas, for composite metaphors, the significance is weaker at $p < . 0 5$ , and only Reinforcing editorials have more composite metaphors than ineffective ones with an effect size of r = .2.

These findings also align with El Baff et al. (2020c), showing that liberals are affected by style features, such as emotional tone and argumentative lexicon. Here, we reinforce previous findings using another style feature: metaphors.

<table><tr><td rowspan="2">Category</td><td>All</td><td>Source</td><td></td><td>Target</td><td>Domain</td></tr><tr><td colspan="5">unique count unique count unique count Examples</td></tr><tr><td>Nature</td><td>16 4,466</td><td></td><td>14 2,256</td><td></td><td>4 2210 stone, body of water, abyss, nature, geographic feature</td></tr><tr><td>Morality and justice</td><td>19 3,946</td><td></td><td>9 331</td><td></td><td>13 3,615 law, freedom, crime, duty, right, democracy</td></tr><tr><td>Darkness and light</td><td>9 3,185</td><td></td><td>7 3,136</td><td>4</td><td>49 weakness, darkness, moment in time, death, light</td></tr><tr><td>Engineering and business</td><td>15 2,980</td><td></td><td>5 702</td><td></td><td>10 2,278 energy, development, development, competition</td></tr><tr><td>Systematic explanations</td><td>12 2,952</td><td></td><td>81,625</td><td></td><td>7 1,327 forceful extraction, position, problem, object</td></tr><tr><td>Life cycle and relations</td><td>13 2,434</td><td></td><td>7 816</td><td></td><td>7 1618 social system, people, family, change, relationship</td></tr><tr><td>Health and safety</td><td>151,204</td><td></td><td>141,155</td><td>2</td><td>49 natural physical force, illness, hazardous</td></tr><tr><td>Embodied experience</td><td>3 1,014</td><td></td><td>1 788</td><td>2</td><td>226 emotional experience, emotion, feeling</td></tr><tr><td>Journey</td><td>7 766</td><td></td><td>4 307</td><td>3</td><td>459 past, struggle, story, service, success, journey, time</td></tr><tr><td>Animals</td><td>9 404</td><td></td><td>395</td><td>3</td><td>9 animal, addiction, monster, parasite, game</td></tr><tr><td>Power and control</td><td>4 350</td><td></td><td>45</td><td>3</td><td>305 rule, election, leader, enonomic system</td></tr><tr><td>Conflict</td><td>2 303</td><td></td><td>2 303</td><td>0</td><td>0 war, barrier</td></tr><tr><td>Plants</td><td>5 261</td><td></td><td>5 261</td><td>0</td><td>0 food, resource, crop, plant</td></tr><tr><td>High and low</td><td>2</td><td>28</td><td>1 27</td><td>1</td><td>1 temperature, hurdle</td></tr></table>

Table 4: The unique and total count of source and target domains over the 14 categories (Gordon et al., 2015) along with examples from the domains selected from our dataset. Highest and lowest numbers are bolded.

## 4.2 Domain Effect on Ideology

The source and target domains contain 131 unique overlapping values: 84 and 59, respectively. Given our dataset’s small amount (300), gaining insights into the source and target domains concerning the editorial effect is challenging. For that, we categorize the domains into a systemic taxonomy of ontological categories, and then we conduct our analysis of the relationship between domain category and editorials’ effect per ideology.

## Domain Categorization

To categorize source and target domain, we rely on the work of Gordon et al. (2015), where they defined and categorized source domains into 14 ontological categories as shown in Table 4, which semantically and conceptually abstract similar source domains. For example, the source domains body of water and abyss are categorized under nature, while animal and monster are categorized under animal, as shown in Table 4.

To categorize the domains (source and target) in our dataset, each goes through the following pipeline: (1) preprocessing (we manually fix the typos), (2) measuring the relatedness between the domain and each of the fourteen categories using Dor et al. (2018)’s Term Relater and (2) assigning the category with the highest score to the domain.

Table 5 shows the distribution of the unique and the total count of domains across the 14 categories where Morality and Justice have the highest unique domains (19) and Conflict and High and Low have the lowest unique number (2).

Darkness and light (e.g., death, light) has the highest encounter (3,136) in the source domain, whereas Morality and justice (e.g., law, crime) is the highest for the target domain. Gordon et al. (2015) states that there are no theoretical or practical limitations on the target domains that metaphors can describe, which makes it challenging to categorize them. However, source domains (even though not restricted) are drawn by a common set of familiar scenarios, pinpointing that they are easier to categorize. This explains the difference in the distribution between the categories for source and target; however, we observe some similarities, such as the prominence of the Nature (e.g., stone) category in both.

## Domain Category Effect on Ideology

Now that each source and target domain is mapped to one of the 14 categories, we conduct our analysis to gain insights into the connection between domain categories and editorials’ effect on both ideologies. Table 5 shows the distribution of the categories within each domain across editorials’ effect (Challenging, No Effect, and Reinforcing) for each ideology (Conservatives on top and Liberals at the bottom row.).

In addition, we conduct the same analysis as before to calculate significance across effects for the count of each category. As a result, Figure 2 shows the effect size r for each category (y-axis) that had at least a significant difference across a pair of effects (x-axis, e.g., Challenging vs. No Effect)

<table><tr><td rowspan="2">Effect</td><td colspan="2">Darkness &amp; Light</td><td colspan="2">Nature</td><td colspan="2">Journey</td><td colspan="2">Embodied Experience</td><td colspan="2">Power &amp; Control</td><td colspan="2">Engineering &amp; Business</td><td colspan="2">Life Cycle &amp; Relations</td><td colspan="2">Other</td></tr><tr><td>Source Target Source</td><td></td><td></td><td>Target Source Target</td><td></td><td></td><td>Source</td><td>Target</td><td>Source Target</td><td></td><td>Source</td><td>Target</td><td>Source</td><td>Target</td><td>Source</td><td>Target</td></tr><tr><td>Conservative Readers</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Challenging</td><td>336</td><td>3</td><td>275</td><td>256</td><td>18</td><td>41</td><td>90</td><td>35</td><td>5</td><td>8</td><td>78</td><td>219</td><td>111</td><td>181</td><td>447</td><td>619</td></tr><tr><td>No Effect</td><td>2126</td><td>30</td><td>1454</td><td>1,335</td><td>183</td><td>302</td><td>523</td><td>154</td><td>30</td><td>257</td><td>459</td><td>1613</td><td>502</td><td>1047</td><td>2711</td><td>3244</td></tr><tr><td>Reinforcing</td><td>674</td><td>16</td><td>527</td><td>619</td><td>106</td><td>116</td><td>175</td><td>37</td><td>10</td><td>40</td><td>165</td><td>446</td><td>203</td><td>390</td><td>939</td><td>1138</td></tr><tr><td>Liberal Readers</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Challenging</td><td>1,010</td><td>19</td><td>802</td><td>75</td><td>124</td><td>157</td><td>265</td><td>72</td><td>15</td><td>48</td><td>243</td><td>665</td><td>314</td><td>571</td><td>2,772</td><td>3,514</td></tr><tr><td>No Effect</td><td>509</td><td>15</td><td>331</td><td>406</td><td>66</td><td>78</td><td>134</td><td>48</td><td>3</td><td>97</td><td>178</td><td>461</td><td>189</td><td>296</td><td>2,004</td><td>2,028</td></tr><tr><td>Reinforcing</td><td>1,617</td><td></td><td>151,123</td><td>929</td><td>117</td><td>224</td><td>389</td><td>106</td><td>27</td><td>160</td><td>281</td><td>1,152</td><td>313</td><td>751</td><td>3418</td><td>4,460</td></tr><tr><td>All Effects</td><td>3,136</td><td></td><td>49 2,256</td><td>2,210</td><td>307</td><td>459</td><td>788</td><td>226</td><td>45</td><td>305</td><td>702</td><td>2,278</td><td></td><td></td><td>8161,618 8,194 10,002</td><td></td></tr></table>

Table 5: The count of source and target domain categories, per editorial effect (Challenging, No Effect and Reinforcing) with the total of each category-source and -domain (All Effects). Seven out of 14 categories are shown and the other 7 categories are combined under Other. The counts that are significantly different across effects are reported with p\_value < 0.05 , 0.01 and 0.001

![](images/9387ec9fca421ca58e70593e747e14dd0a65558d16610e89795d4b2cb0c43c0b.jpg)  
Figure 2: Heatmap for each domain category for Liberal readers (left) and Conservative readers (right). Each ideology has a source (S) and target (T) heatmap. The y-axis represents each category, and the x-axis represents each effect-pair (e1 vs e2). Each cube represents the effect size r: a green and yellow color indicates a positive effect size where e1 has a significantly higher number than e2 (e1 » e2), whereas a blue color indicates the opposite (e1 « e2).

for either ideology. Each cell is colour-coded for each effect pair (effect1 vs. effect2), where blue indicates that the count of a category in effect2 is significantly higher than for effect1.

The opposite interpretation stands for green and yellow. In general, the results coincide with our previous observations. For liberals, we observe richer results where nine categories significantly differ across editorials’ effects compared to four categories for conservatives (Figure 2).

Our findings are in accordance with the definition provided by El Baff et al. (2018) regarding effective editorials that can have two values: Challenging if the editorial holds a stance opposite to the reader, or Reinforcing if the editorial holds the same stance as the reader. Our key findings are described below, based on Figure 2.

Conservatives. The source domains categorized under Journey are significantly higher for reinforcing editorials than others, implying that this source category is effective when the reader shares the same stance as the editorial (r = .24 and r = .15 vs. Challenging and No Effect). Additionally, reinforcing editorials have fewer source domains with High & Low categories compared to ineffective ones with a very weak effect size of $r = . 0 7$ . For target domains under the Power & Control category, effective editorials have lower counts than ineffective ones, with a strong effect size of r = .13 and r = .15 for Challenging and Reinforcing, respectively.

Liberals. For the source domains, effective editorials have significantly fewer domains categorized under Animals and Plants with effect sizes $r = . 1 9 - . 2 2$ . The opposite holds for Darkness & Light and Nature with a strong effect size (r = .33 and r = .44 vs. Challenging and Reinforcing, respectively). Also, for Reinforcing editorials, Power & Control (r = .11) and Embodied Experience (r = .17) seem more prominent than ineffective editorials.

For the target domain, a similar observation holds for the Nature category, where it is more prominent in effective editorials (r = .33 and r = .18 for Challenging and Reinforcing, respectively). Categories such as Engineering & Business and Journey are more prominent in Reinforcing editorials than ineffective ones (r = .2). Finally, challenging editorials have significantly fewer Power & Control target domains compared to ineffective (r = .23) and reinforcing (r = .15) editorials, implying that this target category is ineffective when the reader has an opposite stance to the editorial’s stance.

## 5 Conclusion

The framing of political discourses often relies heavily on the use of metaphors. In this paper, we introduce a dataset which includes two levels of annotations in the intersection of metaphorical usage and the persuasive effects of political ideologies on readers. The first level is designed to identify metaphors, while the second level aims to determine the domains from which these metaphors originate (source domain) and to which they are applied (target domain).

Our findings show that liberal readers are affected significantly by metaphors, whereas conservatives are more resistant to them. The impact of metaphors on ideologies varies based on their conceptual domains, influencing how political opinions are either challenged or reinforced in news editorials. For example, Liberals are affected by metaphors in the Darkness & Light (e.g., death) source domains, whereas the source domain of Nature affects conservatives more significantly.

## 6 Acknowledgment

This work has been supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under project number TRR 318/1 2021 – 438445824. We thank the anonymous reviewers for their helpful feedback.

## 7 Limitations

The usage of metaphors in language is grounded, which means that some words and phrases can be seen to be both metaphorical and not metaphorical depending on the subjectivity of the context in language, such as for “inflation is going higher”. In our work, we consider all degrees of metaphoricity as metaphorical usage, and for simplicity do not distinguish between the various degrees.

However, there may still be metaphorical utterances that we missed in our annotation process. In particular, to foster a uniform metaphor handling among the annotators, we confined the context window of composite metaphors to be at most two words on either end of the pivot word. That being said, there may be cases where the entire sentence would be needed as the context window in order to properly identify a metaphor.

Finally, we point out that our analysis of the interactions between metaphors and persuasive effects are limited by the expressiveness of the human judgments. Persuasion is an intrinsically subjective concept that is not only affected by political ideologies. Hence, we may have observed correlations that confounded by other characteristics of the human annotators. Since the persuasive effect annotations came from previous work, we had no way to further control for this, but future work should validate our results in comparison studies.

## 8 Ethical Statement

We consider no conceivable immediate potential ethical issues or threat to be caused by our corpus, since we only analyzed semantic concepts in an existing corpus. Each of the annotators was paid \$ 12.50 per hour, which is in line with the standards of fair payment of the host institutions of all authors of this paper. We consider no potential threat to be caused by our dataset.

## References

Khalid Al Khatib, Michael Völske, Shahbaz Syed, Nikolay Kolyada, and Benno Stein. 2020. Exploiting personal characteristics of debaters for predicting persuasiveness. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7067–7072, Online. Association for Computational Linguistics.

Ramy Baly, Giovanni Da San Martino, James Glass, and Preslav Nakov. 2020. We can detect your bias: Predicting the political ideology of news articles. arXiv preprint arXiv:2010.05338.

Felix Biessmann, Pola Lehmann, Daniel Kirsch, and Sebastian Schelter. 2016. Predicting political party affiliation from text. PolText, 14(14):2016.

Amber Boeynaems, Christian Burgers, Elly A Konijn, and Gerard J Steen. 2017. The effects of metaphorical framing on political persuasion: A systematic

literature review. Metaphor and Symbol, 32(2):118– 134.

Britta C Brugman, Christian Burgers, and Gerard J Steen. 2017. Recategorizing political frames: A systematic review of metaphorical framing in experiments on political communication. Annals ofthe International Communication Association, 41(2):181– 197.

Pere-Lluís Huguet Cabot, Verna Dankers, David Abadi, Agneta Fischer, and Ekaterina Shutova. 2020. The pragmatics behind politics: Modelling metaphor, framing and emotion in political discourse. In Findings ofthe associationfor computational linguistics: emnlp 2020, pages 4479–4488.

Dallas Card, Amber E. Boydstun, Justin H. Gross, Philip Resnik, and Noah A. Smith. 2015. The media frames corpus: Annotations of frames across issues. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 438– 444, Beijing, China. Association for Computational Linguistics.

Tuhin Chakrabarty, Xurui Zhang, Smaranda Muresan, and Nanyun Peng. 2021. Mermaid: Metaphor generation with symbolism and discriminative decoding. arXiv preprint arXiv:2103.06779.

Jonathan Charteris-Black. 2009. Metaphor and political communication. In Metaphor and discourse, pages 97–115. Springer.

Paul Chilton and Mikhail Ilyin. 1993. Metaphor in political discourse: The case of thecommon european house’. Discourse & society, 4(1):7–31.

Joep P Cornelissen, Robin Holt, and Mike Zundel. 2011. The role of analogy and metaphor in the framing and legitimization of strategic change. Organization Studies, 32(12):1701–1716.

Erik-Lân Do Dinh, Hannah Wieland, and Iryna Gurevych. 2018. Weeding out conventionalized metaphors: A corpus of novel metaphor annotations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1412–1424.

Liat Ein Dor, Alon Halfon, Yoav Kantor, Ran Levy, Yosi Mass, Ruty Rinott, Eyal Shnarch, and Noam Slonim. 2018. Semantic relatedness of wikipedia concepts–benchmark data and a working solution. In Proceedings ofthe Eleventh International Conference on Language Resources and Evaluation (LREC 2018).

Esin Durmus and Claire Cardie. 2018. Exploring the role of prior beliefs for argument persuasion. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1035–1045, New Orleans,

Louisiana. Association for Computational Linguistics.

Roxanne El Baff, Khalid Al Khatib, Benno Stein, and Henning Wachsmuth. 2020a. Persuasiveness of news editorials depending on ideology and personality. In Proceedings of the Third Workshop on Computational Modeling of People’s Opinions, Personality, and Emotion’s in Social Media, pages 29–40, Barcelona, Spain (Online). Association for Computational Linguistics.

Roxanne El Baff, Henning Wachsmuth, Khalid Al-Khatib, and Benno Stein. 2018. Challenge or Empower: Revisiting Argumentation Quality in a News Editorial Corpus. In 22nd Conference on Computational Natural Language Learning (CoNLL 2018), pages 454–464. Association for Computational Linguistics.

Roxanne El Baff, Henning Wachsmuth, Khalid Al Khatib, and Benno Stein. 2020b. Analyzing the Persuasive Effect of Style in News Editorial Argumentation. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 3154–3160, Online. Association for Computational Linguistics.

Roxanne El Baff, Henning Wachsmuth, Khalid Al Khatib, and Benno Stein. 2020c. Analyzing the persuasive effect of style in news editorial argumentation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 3154–3160.

Farahman Farrokhi and Sanaz Nazemi. 2015. The rhetoric of newspaper editorials. International Journal on Studies in English Language and Literature (IJSELL), 3(2):155–161.

Hongyu Gong, Kshitij Gupta, Akriti Jain, and Suma Bhat. 2020. Illinimet: Illinois system for metaphor detection with contextual and linguistic information. In Proceedings ofthe Second Workshop on Figurative Language Processing, pages 146–153.

Jonathan Gordon, Jerry R Hobbs, Jonathan May, Michael Mohler, Fabrizio Morbini, Bryan Rink, Marc Tomlinson, and Suzanne Wertheim. 2015. A corpus of rich metaphor annotation. In Proceedings of the Third Workshop on Metaphor in NLP, pages 56–66.

Willem Joris, Leen d’Haenens, and Baldwin Van Gorp. 2014. The euro crisis in metaphors and frames: Focus on the press in the low countries. European Journal ofCommunication, 29(5):608–617.

Rohan Joseph, Timothy Liu, Aik Beng Ng, Simon See, and Sunny Rai. 2023. NewsMet : A ‘do it all’ dataset of contemporary metaphors in news headlines. In Findings of the Association for Computational Linguistics: ACL 2023, pages 10090–10104, Toronto, Canada. Association for Computational Linguistics.

George Lakoff. 1995. Metaphor, morality, and politics, or, why conservatives have left liberals in the dust. Social Research, pages 177–213.

George Lakoff and Mark Johnson. 2008. Metaphors we live by. University of Chicago press.

George Lakoff and Mark Johnson. 2020. Conceptual metaphor in everyday language. In Shaping entrepreneurship research, pages 475–504. Routledge.

Chang Li and Dan Goldwasser. 2019. Encoding social information with graph convolutional networks forPolitical perspective detection in news media. In Proceedings of the 57th Annual Meeting of the Associationfor Computational Linguistics, pages 2594– 2604, Florence, Italy. Association for Computational Linguistics.

Stephanie Lukin, Pranav Anand, Marilyn Walker, and Steve Whittaker. 2017. Argument Strength is in the Eye of the Beholder: Audience Effects in Persuasion. In Proceedings ofthe 15th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics: Volume 1, Long Papers, pages 742–753. Association for Computational Linguistics.

Matti Luokkanen, Suvi Huttunen, and Mikael Hildén. 2014. Geoengineering, news media and metaphors: Framing the controversial. Public Understanding of Science, 23(8):966–981.

Jeffery Scott Mio. 1997. Metaphor and politics. Metaphor and symbol, 12(2):113–133.

Saif Mohammad, Felipe Bravo-Marquez, Mohammad Salameh, and Svetlana Kiritchenko. 2018. SemEval-2018 task 1: Affect in tweets. In Proceedings of the 12th International Workshop on Semantic Evaluation, pages 1–17, New Orleans, Louisiana. Association for Computational Linguistics.

Michael Mohler, Mary Brunson, Bryan Rink, and Marc Tomlinson. 2016. Introducing the lcc metaphor datasets. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), pages 4221–4227.

Elena Semino, Zsófia Demjén, and Jane Demmen. 2018. An integrated approach to metaphor and framing in cognition, discourse, and practice, with an application to metaphors for cancer. Applied linguistics, 39(5):625–645.

Meghdut Sengupta, Milad Alshomary, Ingrid Scharlau, and Henning Wachsmuth. 2023. Modeling highlighting of metaphors in multitask contrastive learning paradigms. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 4636– 4659, Singapore. Association for Computational Linguistics.

Meghdut Sengupta, Milad Alshomary, and Henning Wachsmuth. 2022. Back to the roots: Predicting the source domain of metaphors using contrastive learning. In Proceedings of the 3rd Workshop on Figurative Language Processing (FLP), pages 137–142, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics.

Ekaterina Shutova, Lin Sun, and Anna Korhonen. 2010. Metaphor identification using verb and noun clustering. In Proceedings of the 23rd International Conference on Computational Linguistics (Coling 2010), pages 1002–1010.

Gerard Steen. 2010. A methodfor linguistic metaphor identification: From MIP to MIPVU, volume 14. John Benjamins Publishing.

Enrica Troiano, Laura Oberländer, and Roman Klinger. 2023. Dimensional modeling of emotions in text with appraisal theories: Corpus creation, annotation reliability, and prediction. Computational Linguistics, 49(1):1–72.

Wei Ye, Bo Li, Rui Xie, Zhonghao Sheng, Long Chen, and Shikun Zhang. 2019. Exploiting entity bio tag embeddings and multi-task learning for relation extraction with imbalanced data.

Dongyu Zhang, Minghao Zhang, Heting Zhang, Liang Yang, and Hongfei Lin. 2021. Multimet: A multimodal dataset for metaphor understanding. In Proceedings of the 59th Annual Meeting of the Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3214– 3225.