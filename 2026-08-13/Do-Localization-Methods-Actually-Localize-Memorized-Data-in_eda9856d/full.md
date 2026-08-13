# Do Localization Methods Actually Localize Memorized Data in LLMs? A Tale of Two Benchmarks

Ting-Yun Chang Jesse Thomason Robin Jia University of Southern California, Los Angeles, CA, USA {tingyun, jessetho, robinjia}@usc.edu

## Abstract

The concept of localization in LLMs is often mentioned in prior work; however, methods for localization have never been systematically and directly evaluated. We propose two complementary benchmarks that evaluate the ability of localization methods to pinpoint LLM components responsible for memorized data. In our INJ Benchmark, we actively inject a piece of new information into a small subset of LLM weights, enabling us to directly evaluate whether localization methods can identify these “ground truth” weights. In our DEL Benchmark, we evaluate localization by measuring how much dropping out identified neurons deletes a memorized pretrained sequence. Despite their different perspectives, our two benchmarks yield consistent rankings of five localization methods. Methods adapted from network pruning perform well on both benchmarks, and all evaluated methods show promising localization ability. On the other hand, even successful methods identify neurons that are not specific to a single memorized sequence.<sup>1</sup>

## 1 Introduction

Large language models (LLMs) memorize many sequences from their pretraining corpora (Carlini et al., 2019; Lehman et al., 2021; Lee et al., 2023). For example, Carlini et al. (2021) show that GPT2 (Radford et al., 2019) can leak some private contact information verbatim. This paper studies whether we can localize a piece of memorized data, i.e., identify components in LLMs responsible for generating a sequence (near) verbatim. Successful localization may inform further work in machine unlearning (Cao and Yang, 2015; Bourtoule et al., 2021); for instance, one could apply “neural surgery” to the located components to make the LLM forget a piece of sensitive information.

Prior work on knowledge editing suggests that we can locate a small set of LLM parameters that store factual knowledge (Dai et al., 2022; Meng et al., 2022). These works demonstrate localization success by showing knowledge editing success when updating only the located LLM parameters. However, Hase et al. (2023) argue that editing success and localization are actually uncorrelated. Similarly, prior methods that identify subnetworks in LLMs (Gong et al., 2022; Panigrahi et al., 2023) usually focus on the performance of downstream classification tasks, lacking direct evaluation on localization per se. Hence, the degree of existing methods’ localization success remains unclear.

This paper studies the open question, “Do localization methods actually localize memorized data in LLMs?” We first propose decoupling localization success from downstream success in our INJ Benchmark. Our key insight is to actively create the ground-truth weights responsible for data memorization. Specifically, we force LLMs to use a small set of pre-decided weights to memorize a piece of new information unseen during pretraining. Therefore, we have the ground-truth locations where the new information is injected. We can then directly evaluate how well different localization methods recall the indices of the injected weights.

We further apply the localization methods to a real-world scenario: identifying a small set of neurons in an LLM responsible for memorizing a pretrained sequence. In this setting, evaluating localization success is more challenging because the ground-truth “location” of each memorized sequence is unknown. We propose the DEL Benchmark, inspired by knockouts (Olsson et al., 2022), a reverse-engineering approach that removes a set of nodes from the computation graph to observe their importance for specific model behavior. We first collect a set of memorized sequences, and for each sequence, we drop out the located neurons to measure their importance to memorizing that target sequence. A successful localization should cleanly erase the target sequence from an LLM without hurting the memorization of the other sequences in the set after dropout. Our two benchmarks complement each other: the INJ Benchmark provides a direct evaluation of localization methods under a well-controlled setup, while DEL Benchmark answers if the methods can localize pretrained sequences that LLMs have already memorized.

We systematically evaluate five methods on our two benchmarks, including existing localization methods (ACTIVATIONS, Geva et al., 2022; IG, Dai et al., 2022), a brute-force method that searches for the most important neurons (ZERO-OUT), and two methods we adapt from network pruning (Hassibi and Stork, 1992; Han et al., 2016), SLIMMING and HARD CONCRETE. Our two benchmarks rank the five methods in the same order, showing especially strong localization ability for HARD CONCRETE. For example, dropping out only 0.5% of neurons in Pythia-6.9B (Biderman et al., 2023) identified by HARD CONCRETE makes the model forget 57.7% of the target memorized tokens on average. On the other hand, the DEL Benchmark shows all methods struggle to balance between erasing the target sequence and retaining other memorized data, indicating that the identified neurons are also relevant for memorizing some other sequences. Overall, both benchmarks agree all evaluated localization methods are promising, but precise localization of a single sequence remains difficult.

## 2 Background and Task Terminology

A Transformer layer (Vaswani et al., 2017) consists of multi-head self-attention and a feed-forward network (FFN). Prior work shows that LLMs use their FFNs rather than self-attention as “memories” to store knowledge (Geva et al., 2021, 2022; Meng et al., 2022). Here, an FFN has two fully connected layers with a non-linear activation function σ:

$$
h ^ { l } = \sigma ( W ^ { l } \mathbf { x } ^ { l } )\tag{1}
$$

$$
o ^ { l } = V ^ { l } h ^ { l } ,\tag{2}
$$

where $\mathbf { x } ^ { l } \in \mathbb { R } ^ { d _ { 1 } }$ is the input hidden states to the $l -$ th FFN layer, $W ^ { l } \in \mathbb { R } ^ { d _ { 2 } \times d _ { 1 } } , V ^ { l } \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ are the weights, $h ^ { l } \in \mathbb { R } ^ { d _ { 2 } }$ the intermediate hidden states, and $o ^ { l } \in \mathbb { R } ^ { d _ { 1 } }$ the output hidden states. Geva et al. (2022) rewrite Eq. 2 as a linear combination of columns of $V ^ { l }$ . Let $\mathbf { v } _ { i } ^ { l } \in \mathbb { R } ^ { d _ { 1 } }$ be the i-th column of $V ^ { l }$ and $h _ { i } ^ { l } \in \mathbb { R }$ be the i-th neuron activation of $h ^ { l } \in \mathbb { R } ^ { d _ { 2 } }$ . We have:

$$
o ^ { l } = V ^ { l } h ^ { l } = \sum _ { i = 1 } ^ { d _ { 2 } } h _ { i } ^ { l } \cdot { \bf v } _ { i } ^ { l }\tag{3}
$$

They show that different concepts are stored in different $\mathbf { v } _ { i } ^ { l } .$ , and that we can view each activation $h _ { i } ^ { l }$ as a memory coefficient to retrieve a concept.

Neurons. Dai et al. (2022) observe the existence of knowledge neurons, a small set of neurons in FFN hidden states $h ^ { l }$ that corresponds to a relational fact, where a neuron means a component of the vector $h ^ { l }$ . For example, given the input “The capital ofIreland $i s \_ s ,$ , they can increase the model probability on the correct token “Dublin” by amplifying the activation $h _ { i } ^ { l }$ of the identified knowledge neurons. With Eq. 3, we can view increasing activation $h _ { i } ^ { l }$ as promoting the concept stored in $\mathbf { v } _ { i } ^ { l }$

In this work, we only search for neurons in FFNs responsible for memorizing a sequence, following Dai et al. (2022). In the INJ Benchmark, we ensure that FFNs act as neural memories by only updating a set of weight vectors $\mathbf { v } _ { i } ^ { l }$ to memorize the new information. As each $\mathbf { v } _ { i } ^ { l }$ corresponds to a neuron in $h ^ { l } .$ , locating the updated weights is equivalent to locating the corresponding neurons. In the rest of the paper, we refer to neurons as the neurons in $\{ h ^ { l } \} _ { l = 1 } ^ { L }$ , where $L$ is the number of layers.

Dropout. Different from Srivastava et al. (2014), we drop out located neurons at test time to erase a memorized sequence from the LLM. We can view dropping out the i-th neuron in $h ^ { l }$ as excluding the contribution of $\mathbf { v } _ { i } ^ { l }$ from the output $o ^ { l }$ in Eq. 3.

Memorized Sequences. Consider a sequence $x = ( p , s )$ that consists of a prefix $p$ and a suffix s. Given the prefix as the prompt, if an LLM is able to nearly reconstruct the suffix with greedy decoding, we say x is a memorized sequence by the LLM. We discuss in §3.2 our criteria on suffix reconstruction, where we tolerate near-verbatim memorization; we also ensure every sequence has a non-trivial suffix.

Localization. Hase et al. (2023) provides a general definition of localization: identifying components of a model responsible for a certain behavior. Under this definition, we consider components as a small set of neurons and behavior as the LLM’s generation of a memorized sequence. Although some components are necessary for generation, e.g., the input and output token embeddings, we exclude them as they are not specific to a target sequence.

Localization Methods. Given an LLM, a memorized sequence x, and a fixed number k, a localization method outputs k% of neurons at each layer as the predictions to localize sequence x in the LLM.

![](images/dee30b654caf46d273d587146d03e517de5d63251e0295ce8a47ecb203cb2a23.jpg)  
Figure 1: Left: INJ Benchmark updates a small set of LLM weights to store the new piece of data, where the fine-tuned weight vectors and the corresponding neurons are filled with blue. The neurons predicted by a localization method are circled with black. denotes true-positive, false-positive, and false-negative neurons. Right: DEL Benchmark drops out the predicted neurons on a memorized pretrained sequence. A large change in Levenshtein distance after dropout indicates that were important for LLM $f$ to retrieve the memorized sequence.

## 3 Two Localization Benchmarks

How do we know whether a method is successful in localization? We propose two benchmarking approaches: one injects a new piece of information into specific parameters in LLMs, while another deletes an existing memorized sequence from LLMs via dropout. A successful localization method should do well on both benchmarks.

## 3.1 The INJ Benchmark

A principal challenge in evaluating localization methods is the lack of ground-truth location. We propose the INJ Benchmark, which first creates ground truth by actively injecting a piece of unseen information into a small subset of LLM weights. We can then directly evaluate the correctness of a localization method in predicting the indices of those injected weights.

Data. The ECBD-2021 dataset (Onoe et al., 2022) contains 156 definition sentences of new entities that rose to popularity during the year 2021, e.g., “Gamma variant, also known as lineage P.1...”. Since all LLMs we use are trained on corpora released before 2021, the injected weights are the only parameters in the LLMs responsible for memorizing each new definition sequence x.

Information Injection. For each new sequence $x _ { i }$ in the dataset, we randomly sample $r \%$ of the weight vectors $\{ \mathbf { v } _ { 1 } ^ { l } , \ldots , \mathbf { v } _ { d _ { 2 } } ^ { l } \} _ { l = 1 } ^ { \bar { L } }$ across all L layers, and fine-tune them to memorize x. We keep the rest of the model parameters frozen. To simulate how LLMs learn data during pretraining, we finetune with the normal language modeling loss on $x _ { i }$ (Eq. 13). To ensure the entire sequence is well memorized, we keep fine-tuning until we reach a loss $< 0 . 0 5 $ ; therefore, we can simply set the first token as the prefix $p ,$ and the rest of the sequence as the suffix s. Note we sample a different set of weight vectors $\phi _ { i }$ for each sequence $x _ { i }$ and finetune a separate model ${ \tilde { \theta _ { i } } }$ . Algorithm 1 shows the exact injection process.

Evaluation. For each model ${ \tilde { \theta } } _ { i }$ injected with a sequence x<sub>i</sub>, a localization method predicts k% of neurons at each layer and we calculate Recall@k%. Specifically, given the set of ground-truth neurons corresponding to all the injected weight vectors across layers, $\Gamma _ { i } .$ and the set of all predicted neurons, $\hat { \Gamma _ { i } } .$ the recall is $\frac { | \Gamma _ { i } \cap \hat { \Gamma _ { i } } | } { | \Gamma _ { i } | }$ . Finally, we average the recall scores across sequences, and thus average over different choices of weights $\phi _ { i }$ for injection.

## 3.2 The DEL Benchmark

The DEL Benchmark studies whether we can localize a naturally memorized sequence after pretraining, which is not answered by the INJ Benchmark. We first collect a set of memorized pretrained sequences, and then apply localization methods to identify the responsible neurons for each sequence. Without ground-truth neurons, we adopt knockouts (Li et al., 2016; Olsson et al., 2022; Geva et al., 2023) for evaluation, which measures the importance of model components based on the effect of removing them. We drop out the located neurons to observe how much they account for memorizing a sequence. We quantify memorization with two scores: Accuracy and Levenshtein distance.

Algorithm 1 Information Injection   
Input: The set of new sequences $\chi _ { \mathrm { E C B D } } = \{ x _ { i } \} _ { i = 1 } ^ { N } ;$ pretrained LLM θ with L layers; injection ratio r   
Output: The set of fine-tuned LLMs $\mathcal { M } = \{ \tilde { \theta _ { i } } \} _ { i = } ^ { N }$ 1   
Initialize ${ \mathcal { M } } \gets \emptyset .$   
for i 1 to N do   
<sup>˜</sup>θ θ // Initialize from pretrained weights.   
Retrieve all the FFN weight vectors $\Phi _ { i } = \{ \mathbf { v } _ { 1 } ^ { l } , \ldots , \mathbf { v } _ { d _ { 2 } } ^ { l } \} _ { l = 1 } ^ { L }$ from layers l of $\tilde { \theta _ { i } }$   
Set the random seed to i.   
ϕ<sub>i</sub> Randomly sample $r \%$ of weight vectors from $\Phi _ { i \cdot } / / \phi _ { i } \subset \Phi _ { i } \subset \tilde { \theta _ { i } }$   
Fine-tune ϕ<sub>i</sub> with the language modeling loss on x<sub>i</sub> (Eq. 13) with remaining weights $\tilde { \theta _ { i } } \setminus $ ϕ<sub>i</sub> frozen.   
$\mathcal { M }  \mathcal { M } \cup \tilde { \theta } _ { i }$   
end for   
return

Accuracy. Recall that a sequence $x = ( p , s )$ consists of a prefix $p$ and suffix s. Accuracy calculates the percentage of correct suffix tokens generated by teacher-forcing and argmax decoding. Formally,

$$
\hat { s } _ { t } = \underset { w \in \mathrm { V o c } } { \mathrm { a r g m a x } } P _ { \theta } ( w | p , s _ { < t } ) , t = 1 , . . . , T\tag{4}
$$

$$
\mathrm { A c c u r a c y } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { 1 } \{ \hat { s } _ { t } = s _ { t } \} ,\tag{5}
$$

where $T$ denotes the suffix sequence length, $s _ { t }$ the t-th true suffix token, $\boldsymbol { s } _ { < t } = [ s _ { 1 } , \ldots , s _ { t - 1 } ] , \boldsymbol { \hat { s } _ { t } }$ the t-th generated token, $P _ { \theta }$ the probability distribution of the LLM parameterized by $\theta ,$ and Voc the vocabulary. Higher Accuracy indicates better memorization of the sequence.

Levenshtein distance. While Accuracy is defined at a token level, we note tokens often contain several characters, $\mathrm { e . g . , \ } ^ { \ast } 1 5 9 ^ { \ast }$ . For sequences like $^ { * } \cdot 3 . 1 4 1 5 9 2 6 5 ^ { \prime \prime }$ , every character is important; thus, we also define a memorization score at the character level. With Eq. 4, we have $\hat { s } = [ \hat { s _ { 1 } } , \dotsc , \hat { s _ { T } } ]$ We calculate Levenshtein distance between the generated suffix sˆ and the true suffix s. Lower Levenshtein distance indicates better memorization.

Data. We collect a set of sequences memorized by each LLM, including Pythia-deduped-2.8B,

Pythia-deduped-6.9B, and GPT2-XL. For Pythia models, the pertaining corpus the Pile-dedupe (Gao et al., 2021) is open-sourced, and we use the following criteria to determine which sequences are memorized. For each candidate sequence x, we set the first 32 tokens as the prefix $p$ to prompt the LLM to reconstruct the suffix s of 48 tokens. First, we filter out sequences with Accuracy (Eq. 4, 5) lower than 0.9. Second, we use greedy decoding to generate the suffix, filtering out those with a Levenshtein distance greater than 20 characters to the true suffix. Third, we discard sequences with repetitive tokens (less than 16 distinct tokens in the suffix). Finally, we deduplicate the remaining sequences based on n-gram Jaccard index. We obtain 505 memorized sequences for each Pythia model. For GPT2-XL, we do not have access to its pretraining corpus and find very few memorized sequences from several public corpora with our criteria. Thus, we actively search for potentially memorized sequences, discovering 105 memorized sequences and categorizing them manually (Table 1). See A.8 for details and example sequences.

We sample 5 sequences as the dev set to tune the hyperparameters of different methods (see A.9), using the rest of the collected sequences as the test set. We quantify the memorization of LLMs on the collected test sets. Table 7 in the appendix shows that all LLMs have a high average Accuracy ( 100%) and a low Levenshtein distance ( 1 character) to the true suffix, suggesting that the sequences we collect are indeed well memorized.

<table><tr><td>Category</td><td>Examples</td><td>Count</td></tr><tr><td>Quotes</td><td>Churchill, Steve Jobs, Trump</td><td>17</td></tr><tr><td>Quotes (Book)</td><td>Dune, 1984, Bible</td><td>14</td></tr><tr><td>Ordered items</td><td>Zodiac Signs, US Presidents</td><td>11</td></tr><tr><td>Terms of use</td><td>MIT License</td><td>10</td></tr><tr><td>Poems</td><td>The Second Coming</td><td>9</td></tr><tr><td>Code</td><td>GitHub</td><td>9</td></tr><tr><td>Contact Info</td><td>A journalist&#x27;s email</td><td>7</td></tr><tr><td>URLs</td><td>Reddit, file link</td><td>5</td></tr><tr><td>Others</td><td>long COINBASE ID, meme, Bill of Rights, Pi digits</td><td>23</td></tr></table>

Table 1: Collected sequences memorized by GPT2-XL.

Evaluation. When we evaluate one sequence x in the collected test set , we consider the rest of the memorized sequences, $\mathcal { X } \setminus \{ x \}$ , as negative examples. A successful localization method should make LLMs forget the target sequence (large changes in memorization scores), but still remember the other negative examples (small changes in memorization scores) after dropping out the predicted k% of neurons at each layer.<sup>2</sup> We also calculate the absolute change in perplexity on a batch of 2048 sequences sampled from the Pile-dedupe, , to evaluate whether the general language modeling ability remains intact after dropout.

Despite similarities to the evaluation of model editing (Sinitsin et al., 2020; Mitchell et al., 2022), we can better reflect localization success. Unlike Meng et al. (2022) that edit the located weights with gradients, we restrict our operation to neuron dropout. Because dropout has limited freedom in changing LLMs behavior, successful deletion via dropout requires successful localization; in contrast, gradient-based editing could succeed even without good localization (Hase et al., 2023).

## 4 Localization Methods

We benchmark five localization methods. Each method assigns an attribution score $\mathcal { A } ^ { l } ( i )$ to each neuron $n _ { i } ^ { l }$ , the i-th neuron in the l-th layer, representing its importance in memorizing a sequence x. At test time, we select the top-k% of neurons in each layer for each method in terms of attribution scores as the located neurons for x by that method.

Several methods involve calculating the language modeling loss of an LLM θ on the suffix of the target sequence $x = ( p , s )$ . We denote the loss as memorization loss, $\ell _ { \theta } ^ { \mathrm { { m e m } } } ( x )$ . Formally,

$$
\ell _ { \theta } ^ { \mathrm { { m e m } } } ( x ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } - \log P _ { \theta } ( s _ { t } | p , s _ { < t } )\tag{6}
$$

ZERO-OUT. We introduce an exhaustive method that drops out neurons one by one and uses the resulting change in memorization loss on x as the

attribution score to each neuron $n _ { i } ^ { l . }$

$$
\mathscr { A } ^ { l } ( i ) = \ell _ { \theta \backslash n _ { i } ^ { l } } ^ { \mathrm { m e m } } ( x ) - \ell _ { \theta } ^ { \mathrm { m e m } } ( x )\tag{7}
$$

We denote $\ell _ { \theta \backslash n _ { i } ^ { l } } ^ { \mathrm { m e m } }$ as the memorization loss of the LLM θ after dropping out a neuron $n _ { i } ^ { l }$ . The larger the change in the loss, the more important the neuron is for memorization. ZERO-OUT is closely related to the occlusion-based attribution method (Zeiler and Fergus, 2014).

ACTIVATIONS. We can view the neuron activation $h _ { i } ^ { l }$ as the memory coefficients (§2). Thus, similar to Geva et al. (2022), we set the attribution $\mathcal { A } ^ { l } ( i )$ as the absolute value of $h _ { i } ^ { l }$ multiplied by the vector norm of $\mathbf { v } _ { i } ^ { l } .$ , averaged across the suffix length T:

$$
\mathcal { A } ^ { l } ( i ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } | h _ { i , t } ^ { l } | \ : \| \mathbf { v } _ { i } ^ { l } \| ,\tag{8}
$$

where $h _ { i , t } ^ { l }$ denotes the activation value at the t-th timestep, when the input consists of all the tokens before $s _ { t } ,$ i.e., $[ p , s _ { < t } ]$

Integrated Gradients (IG). We benchmark integrated gradients (Sundararajan et al., 2017), an attribution method that has been used to identify knowledge neurons and privacy neurons (Dai et al., 2022; Wu et al., 2023). IG cumulates the gradients at all points along the path from a zero vector to the original hidden state $h ^ { l }$ . See A.2 for more details.

SLIMMING. We introduce SLIMMING, a localization method adapted from prior work (Liu et al., 2017; Chen et al., 2021) on network pruning. Pruning aims to reduce the model size by finding a subnetwork that can achieve a low loss on the task, e.g., sentiment analysis. In our setting, we find a small set of neurons that are crucial for maintaining a low memorization loss $\ell _ { \theta } ^ { \mathrm { { m e m } } } ( x )$ on one target sequence x (Eq. 6). Specifically, SLIMMING minimizes the memorization loss while learning a sparse mask $m ^ { l } \in \mathbb { R } ^ { d _ { 2 } }$ on the hidden state $h ^ { l }$ in every layer, with mask value $m _ { i } ^ { l }$ on neuron $n _ { i } ^ { l } .$ . At each layer, we transform $h ^ { l }$ to $h ^ { i } \odot m ^ { l }$ before computing further layers, where  denotes elementwise multiplication. The sparse mask encourages the LLM to use only a small set of neurons to recall a piece of memory. All the weights of the LLM are kept frozen during the training; only the mask $m ^ { l }$ is learnable. Formally,

$$
\operatorname* { m i n } _ { \stackrel { m ^ { l } } { l = 1 , \ldots , L } } \ell _ { \theta } ^ { \mathrm { m e m } } ( x ) + \lambda \sum _ { l = 1 } ^ { L } \| m ^ { l } \| _ { 1 } ,\tag{9}
$$

<table><tr><td rowspan="2"></td><td colspan="3">GPT2 124M</td><td colspan="3">GPT2-XL 1.5B</td><td colspan="3">Pythia-deduped 2.8B</td><td colspan="3">Pythia-deduped 6.9B</td></tr><tr><td>R@1%</td><td>R@2%</td><td>R@5%</td><td>R@1%</td><td>R@2%</td><td>R@5%</td><td>R@1%</td><td>R@2%</td><td>R@5%</td><td>R@1%</td><td> ${ \mathrm { R @ 2 \% } }$ </td><td>R@5%</td></tr><tr><td> $r a t i o = I \%$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HARD CONCRETE</td><td> ${ \bf 4 9 . 5 } _ { 0 . 4 8 }$ </td><td> $\mathbf { 7 0 . 2 } _ { 0 . 5 4 }$ </td><td> ${ \bf 8 7 . 4 } _ { 0 . 3 3 }$ </td><td> $\mathbf { 2 9 . 7 _ { 0 . 4 7 } }$ </td><td> ${ \bf 3 7 . 1 _ { 0 . 4 9 } }$ </td><td> ${ \bf 4 8 . 1 _ { 0 . 4 6 } }$ </td><td> $3 4 . 3 _ { 0 . 3 3 }$ </td><td> $5 0 . 1 _ { 0 . 4 3 }$ </td><td> $7 2 . 1 _ { 0 . 4 7 }$ </td><td> $3 6 . 8 _ { 0 . 4 5 }$ </td><td> $5 5 . 1 _ { 0 . 5 1 }$ </td><td> $7 6 . 4 _ { 0 . 4 1 }$ </td></tr><tr><td>SLIMMING</td><td> $4 8 . 1 _ { 0 . 6 4 }$ </td><td> $6 6 . 7 _ { 0 . 6 9 }$ </td><td> $8 0 . 7 _ { 0 . 5 4 }$ </td><td> $1 9 . 3 _ { 0 . 4 9 }$ </td><td> $2 9 . 2 _ { 0 . 5 9 }$ </td><td> $4 1 . 1 _ { 0 . 5 9 }$ </td><td> $\mathbf { 3 7 . 0 _ { 0 . 4 3 } }$ </td><td> $\mathbf { 5 0 . 7 } _ { 0 . 4 7 }$ </td><td> $6 1 . 5 _ { 0 . 4 4 }$ </td><td> $\mathbf { 3 9 . 9 _ { 0 . 3 8 } }$ </td><td> ${ \bf 5 5 . 1 _ { 0 . 3 8 } }$ </td><td> $6 6 . 5 _ { 0 . 3 5 }$ </td></tr><tr><td>ZERO-OUT</td><td> $2 4 . 9 _ { 0 . 7 8 }$ </td><td> $3 7 . 5 _ { 1 . 0 5 }$ </td><td> $5 3 . 8 _ { 1 . 2 4 }$ </td><td> $4 . 1 _ { 0 . 1 3 }$ </td><td> $7 . 2 _ { 0 . 2 3 }$ </td><td> $1 3 . 7 _ { 0 . 4 2 }$ </td><td> $1 0 . 6 _ { 0 . 2 0 }$ </td><td> $1 5 . 0 _ { 0 . 2 4 }$ </td><td> $2 1 . 4 _ { 0 . 3 0 }$ </td><td></td><td></td><td></td></tr><tr><td>IG</td><td> $2 0 . 5 _ { 0 . 5 5 }$ </td><td> $3 2 . 1 _ { 0 . 8 0 }$ </td><td> $4 9 . 9 _ { 0 . 9 9 }$ </td><td> $4 . 3 _ { 0 . 1 3 }$ </td><td> $7 . 2 _ { 0 . 2 1 }$ </td><td> $1 3 . 3 _ { 0 . 3 7 }$ </td><td> $1 1 . 6 _ { 0 . 2 2 }$ </td><td> $1 6 . 9 _ { 0 . 2 8 }$ </td><td> $2 3 . 9 _ { 0 . 3 4 }$ </td><td> $1 2 . 8 _ { 0 . 2 3 }$ </td><td> $1 8 . 7 _ { 0 . 2 9 }$ </td><td> $2 7 . 2 _ { 0 . 3 5 }$ </td></tr><tr><td>ACTIVATIONS</td><td> $3 . 0 _ { 0 . 0 9 }$ </td><td> $5 . 2 _ { 0 . 1 3 }$ </td><td> $1 3 . 3 _ { 0 . 3 2 }$ </td><td> $2 . 1 _ { 0 . 0 5 }$ </td><td> $5 . 0 _ { 0 . 1 0 }$ </td><td> $1 2 . 0 _ { 0 . 1 6 }$ </td><td> $7 . 8 _ { 0 . 1 1 }$ </td><td> $1 2 . 8 _ { 0 . 2 0 }$ </td><td> $3 0 . 5 _ { 0 . 5 3 }$ </td><td> $7 . 9 _ { 0 . 1 1 }$ </td><td> $1 2 . 4 _ { 0 . 1 7 }$ </td><td> $2 7 . 3 _ { 0 . 4 3 }$ </td></tr><tr><td>RANDOM</td><td> $1 . 0 _ { 0 . 0 4 }$ </td><td> $2 . 1 _ { 0 . 0 6 }$ </td><td> $5 . 0 _ { 0 . 0 9 }$ </td><td> $1 . 0 _ { 0 . 0 1 }$ </td><td> $2 . 0 _ { 0 . 0 2 }$ </td><td> $5 . 0 _ { 0 . 0 3 }$ </td><td> $1 . 0 _ { 0 . 0 1 }$ </td><td> $2 . 0 _ { 0 . 0 2 }$ </td><td> $5 . 0 _ { 0 . 0 3 }$ </td><td> $1 . 0 _ { 0 . 0 1 }$ </td><td> $2 . 0 _ { 0 . 0 2 }$ </td><td> $5 . 0 _ { 0 . 0 2 }$ </td></tr><tr><td> $r a t i o = 0 . I \%$ </td><td> $( \% 0 . 1 \%$ </td><td> $@ 0 . 2 \%$ </td><td> $( \mathrm { { \omega } 0 . 5 \% }$ </td><td> $( \mathrm { { ‰} } 0 . 1 \%$ </td><td> $@ 0 . 2 \%$ </td><td> $( \% 0 . 5 \%$ </td><td> $( \% 0 . 1 \%$ </td><td> $( \omega 0 . 2 \%$ </td><td> $( \mathrm { { \omega } 0 . 5 \% }$ </td><td> $@ 0 . 1 \%$ </td><td> $@ 0 . 2 \%$ </td><td> $( \% 0 . 5 \%$ </td></tr><tr><td>HARD CONCRETE</td><td> $5 6 . 4 _ { 0 . 8 3 }$ </td><td> $7 9 . 6 _ { 0 . 8 9 }$ </td><td> $9 3 . 7 _ { 0 . 5 2 }$ </td><td> $4 7 . 5 _ { 0 . 4 0 }$ </td><td> ${ \bf 5 9 . 1 _ { 0 . 4 7 } }$ </td><td> $6 8 . 0 _ { 0 . 4 6 }$ </td><td> ${ \bf 4 8 . 5 } _ { 0 . 4 9 }$ </td><td> ${ \bf 6 7 . 3 _ { 0 . 5 0 } }$ </td><td> $\mathbf { 8 6 . 7 } _ { 0 . 3 4 }$ </td><td> $4 6 . 4 _ { 0 . 6 0 }$ </td><td> ${ \bf 6 6 . 3 } _ { 0 . 7 1 }$ </td><td> $\mathbf { 8 2 . 3 _ { 0 . 4 8 } }$ </td></tr><tr><td>SLIMMING</td><td> $\mathbf { 5 8 . 9 } _ { 0 . 5 9 }$ </td><td> $8 3 . 5 _ { 0 . 6 8 }$ </td><td> $\mathbf { 9 4 . 4 } _ { 0 . 4 9 }$ </td><td> $3 5 . 4 _ { 0 . 5 6 }$ </td><td> $5 5 . 9 _ { 0 . 6 4 }$ </td><td> ${ \bf 6 9 . 5 } _ { 0 . 5 5 }$ </td><td> $4 8 . 3 _ { 0 . 4 3 }$ </td><td> $6 3 . 5 _ { 0 . 4 6 }$ </td><td> $7 3 . 9 _ { 0 . 4 3 }$ </td><td> $4 8 . 5 _ { 0 . 5 7 }$ </td><td> $6 0 . 9 _ { 0 . 6 0 }$ </td><td> $7 1 . 0 _ { 0 . 7 1 }$ </td></tr><tr><td>ZERO-OUT</td><td> $5 4 . 1 _ { 0 . 6 8 }$ </td><td> $7 7 . 8 _ { 0 . 7 8 }$ </td><td> $9 0 . 9 _ { 0 . 7 0 }$ </td><td> $1 4 . 3 _ { 0 . 6 2 }$ </td><td> $2 1 . 8 _ { 0 . 9 4 }$ </td><td> $3 1 . 9 _ { 1 . 2 7 }$ </td><td> $1 6 . 5 _ { 0 . 4 8 }$ </td><td> $2 1 . 1 _ { 0 . 5 7 }$ </td><td> $2 6 . 6 _ { 0 . 6 6 }$ </td><td></td><td></td><td></td></tr><tr><td>IG</td><td> $5 3 . 5 _ { 0 . 7 8 }$ </td><td> $7 4 . 1 _ { 0 . 9 2 }$ </td><td> $8 4 . 8 _ { 0 . 8 0 }$ </td><td> $1 3 . 8 _ { 0 . 5 3 }$ </td><td> $2 0 . 3 _ { 0 . 7 9 }$ </td><td> $2 9 . 7 _ { 1 . 0 6 }$ </td><td> $1 8 . 0 _ { 0 . 4 9 }$ </td><td> $2 3 . 3 _ { 0 . 6 0 }$ </td><td> $3 0 . 2 _ { 0 . 6 8 }$ </td><td> $2 9 . 3 _ { 1 . 0 3 }$ </td><td> $3 4 . 4 _ { 1 . 0 2 }$ </td><td> $3 9 . 6 _ { 0 . 9 7 }$ </td></tr><tr><td>ACTIVATIONS</td><td> $1 1 . 1 _ { 0 . 4 3 }$ </td><td> $2 6 . 5 _ { 0 . 8 4 }$ </td><td> $5 1 . 5 _ { 1 . 0 6 }$ </td><td> $7 . 5 _ { 0 . 3 5 }$ </td><td> $1 5 . 9 _ { 0 . 6 1 }$ </td><td> $3 0 . 6 _ { 0 . 7 6 }$ </td><td> $2 1 . 6 _ { 0 . 7 2 }$ </td><td> $3 4 . 6 _ { 0 . 9 8 }$ </td><td> $5 2 . 5 _ { 1 . 0 7 }$ </td><td> $3 4 . 0 _ { 1 . 0 3 }$ </td><td> $4 5 . 9 _ { 1 . 0 2 }$ </td><td> $5 9 . 5 _ { 0 . 9 7 }$ </td></tr><tr><td>RANDOM</td><td> $0 . 1 _ { 0 . 0 3 }$ </td><td> $0 . 2 _ { 0 . 0 6 }$ </td><td> $0 . 5 _ { 0 . 0 7 }$ </td><td> $0 . 1 _ { 0 . 0 1 }$ </td><td> $0 . 2 _ { 0 . 0 2 }$ </td><td> $0 . 5 _ { 0 . 0 3 }$ </td><td> $0 . 1 _ { 0 . 0 1 }$ </td><td> $0 . 2 _ { 0 . 0 2 }$ </td><td> $0 . 5 _ { 0 . 0 3 }$ </td><td> $0 . 1 _ { 0 . 0 1 }$ </td><td> $0 . 2 _ { 0 . 0 2 }$ </td><td> $0 . 5 _ { 0 . 0 2 }$ </td></tr></table>

Table 2: The INJ Benchmark. We experiment with injection ratio at 1% (Top) and 0.1% (Bottom) and report the Recall@k% and standard errors of different localization methods averaged across the sequences in ECBD-2021.

where λ is the hyperparameter to balance the memorization loss and the $L _ { 1 }$ sparsity regularization on the mask. After training, we set the attribution score $\mathcal { A } ^ { l } ( i ) = m _ { i } ^ { l }$ , as $m _ { i } ^ { l }$ learns the importance of the existence of a neuron to the memorization loss.

HARD CONCRETE. The limitation of SLIM-MING is that it tends to assign mask values $m _ { i } ^ { l }$ between 0 and 1 on most neurons, creating a mismatch between training and testing. In particular, at inference time we either activate (equivalent to setting $m _ { i } ^ { l } = 1 )$ or drop out $( m _ { i } ^ { l } = 0 )$ a neuron. Thus, we adapt another pruning method HARD CONCRETE (Louizos et al., 2018; Zheng et al., 2022) for localization, which improves over SLIM-MING by encouraging mask values $m _ { i } ^ { l }$ to be approximately binary. Similar to SLIMMING, HARD CONCRETE learns parameters $m ^ { l } \in \mathbb { R } ^ { d _ { 2 } }$ in every layer. But instead of directly using $m ^ { l }$ as the mask, the mask $\bar { m } ^ { l }$ in HARD CONCRETE is a random variable (r.v.) that depends on $m ^ { l }$ . Specifically, HARD CONCRETE derives the mask value $\bar { m } _ { i } ^ { l }$ from a binary concrete (Maddison et al., 2017; Jang et al., 2017) random variable, $\hat { m } _ { i } ^ { l } .$ . A binary concrete distribution $\hat { m } _ { i } ^ { l } \sim$ Concrete $( m _ { i } ^ { l } , \beta )$ is parameterized by the location $m _ { i } ^ { l }$ and temperature $\beta .$ When the hyperparameter $\beta  0$ , sampling from the binary concrete distribution is identical to sampling from a Bernoulli distribution but loses the differentiable property. With $\beta > 0$ , we allow gradient-based optimization of parameter $m _ { i } ^ { l }$ through the reparametrization trick. Formally,

$$
\begin{array} { r } { u _ { i } \sim \mathcal { U } \left( 0 , 1 \right) , } \end{array}\tag{10}
$$

$$
\hat { m } _ { i } ^ { l } = \sigma \left( \frac { 1 } { \beta } ( \log \frac { u _ { i } } { 1 - u _ { i } } + \log m _ { i } ^ { l } ) \right) ,\tag{11}
$$

where σ denotes the sigmoid function and $u _ { i }$ is a r.v. sampled from uniform distribution $\mathcal { U } \left( 0 , 1 \right)$ We describe the details about how Louizos et al. (2018) extend a hard concrete r.v. $\bar { m } ^ { l }$ from the binary concrete r.v. $\hat { m } _ { i } ^ { l }$ and use $L _ { 0 }$ regularization $\mathcal { R } ( \bar { m } ^ { l } )$ to encourage sparsity in $\mathrm { A . 4 }$

To learn the parameters $m ^ { l }$ , we freeze the LLM weights θ and simultaneously optimize the memorization loss on the target sequence x and the sparsity loss $\mathcal { R } ( \bar { m } ^ { l } )$ . Formally,

$$
\operatorname* { m i n } _ { \stackrel { m ^ { l } } { l = 1 , \ldots , L } } \ell _ { \theta } ^ { \mathrm { m e m } } ( x ) + \lambda \sum _ { l = 1 } ^ { L } \mathcal { R } ( \bar { m } ^ { l } )\tag{12}
$$

At test time, $\hat { m } _ { i } ^ { l }$ can be estimated as $\sigma \left( \log m _ { i } ^ { l } \right)$ (Louizos et al., 2018); thus, we set the attribution score $\mathcal { A } ^ { l } ( i ) = \sigma \left( \log m _ { i } ^ { l } \right)$ .

## 5 Experiments

## 5.1 INJ Benchmark Results

Table 2 shows the average Recall@k% and standard errors of different localization methods on four LLMs under our INJ Benchmark evaluation. When the injection ratio is 1% (Table 2; Top), there are $1 \%$ of weight vectors injected with each new sequence, yielding 1% of ground truth neurons, and every method predicts $k = \{ 1 , 2 , 5 \} \%$ of neurons at each layer. When the injection ratio is 0.1% (Table 2; Bottom), every method predicts $\{ 0 . 1 , 0 . 2 , 0 . 5 \} \%$ of neurons at each layer. We also study the alternative that predicts top-k neurons across layers in A.11, which shows results consistent with Table 2 but with lower recall overall.

All methods can do localization. First, all five localization methods greatly outperform RANDOM, which randomly predicts the same number of neurons at each layer. Interestingly, when the injection ratio is lower (0.1%), all localization methods achieve higher recall, possibly because the information is more concentrated in the injected weights and thus easier to identify.

<table><tr><td rowspan="2">dropout ratio =</td><td colspan="2">∆ Self-Acc ↓</td><td colspan="2">∆ Self-Dist ↑</td><td colspan="2">∆Neg-Acc ↑</td><td colspan="2">∆ Neg-Dist↓</td><td colspan="2">△ Rand-PPL↓</td></tr><tr><td>0.1%</td><td>0.5%</td><td>0.1%</td><td>0.5%</td><td>0.1%</td><td>0.5%</td><td>0.1%</td><td>0.5%</td><td>0.1%</td><td>0.5%</td></tr><tr><td>GPT2-XL 1.5B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HARD CONCRETE</td><td>-34.6%</td><td>-57.1%</td><td>42.9</td><td>74.0</td><td>-2.4%</td><td>-4.8%</td><td>2.5</td><td>5.4</td><td>0.03</td><td>0.11</td></tr><tr><td>SLIMMING</td><td>-30.5%</td><td>-57.8%</td><td>37.7</td><td>75.4</td><td>-3.5%</td><td>-6.4%</td><td>4.1</td><td>7.5</td><td>0.02</td><td>0.17</td></tr><tr><td>ZERO-OUT</td><td>-29.8%</td><td>-46.1%</td><td>33.0</td><td>55.2</td><td>-3.1%</td><td>-4.8%</td><td>3.5</td><td>5.5</td><td>0.03</td><td>0.09</td></tr><tr><td>IG</td><td>-25.8%</td><td>-40.8%</td><td>27.0</td><td>46.0</td><td>-2.2%</td><td>-3.4%</td><td>2.3</td><td>3.7</td><td>0.01</td><td>0.05</td></tr><tr><td>ACTIVATIONS</td><td>-14.8%</td><td>-29.5%</td><td>16.9</td><td>36.4</td><td>-3.0%</td><td>-4.7%</td><td>3.1</td><td>5.4</td><td>0.11</td><td>0.16</td></tr><tr><td>RANDOM</td><td>-0.2%</td><td>-0.5%</td><td>0.2</td><td>0.4</td><td>-0.2%</td><td>-0.5%</td><td>0.1</td><td>0.4</td><td>0.00</td><td>0.03</td></tr><tr><td>Pythia-deduped 2.8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HARD CONCRETE</td><td>-29.0%</td><td>-53.2%</td><td>55.3</td><td>99.8</td><td>-3.7%</td><td>-10.5%</td><td>7.7</td><td>22.1</td><td>0.23</td><td>0.56</td></tr><tr><td>SLIMMING</td><td>-17.4%</td><td>-45.1%</td><td>32.9</td><td>80.8</td><td>-3.3%</td><td>-7.0%</td><td>6.6</td><td>13.9</td><td>0.26</td><td>0.49</td></tr><tr><td>ZERO-OUT</td><td>-14.8%</td><td>-25.9%</td><td>26.4</td><td>45.2</td><td>-1.1%</td><td>-2.5%</td><td>2.1</td><td>5.0</td><td>0.21</td><td>0.35</td></tr><tr><td>IG</td><td>-16.7%</td><td>-30.3%</td><td>29.1</td><td>52.5</td><td>-0.9%</td><td>-2.1%</td><td>1.8</td><td>4.4</td><td>0.09</td><td>0.18</td></tr><tr><td>ACTIVATIONS</td><td>-13.0%</td><td>-25.5%</td><td>27.5</td><td>52.2</td><td>-3.1%</td><td>-6.1%</td><td>6.6</td><td>12.9</td><td>0.11</td><td>0.20</td></tr><tr><td>RANDOM</td><td>-0.1%</td><td>-0.3%</td><td>0.1</td><td>0.5</td><td>-0.1%</td><td>-0.3%</td><td>0.2</td><td>0.5</td><td>0.00</td><td>0.02</td></tr><tr><td>Pythia-deduped 6.9B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HARD CONCRETE</td><td>-29.2%</td><td>-57.7%</td><td>58.5</td><td>109.9</td><td>-3.8%</td><td>-14.7%</td><td>8.7</td><td>32.6</td><td>0.16</td><td>0.52</td></tr><tr><td>SLIMMING</td><td>-24.1%</td><td>-48.7%</td><td>48.8</td><td>92.1</td><td>-4.2%</td><td>-11.3%</td><td>9.1</td><td>23.6</td><td>0.23</td><td>0.58</td></tr><tr><td>IG</td><td>-16.9%</td><td>-32.3%</td><td>31.4</td><td>57.8</td><td>-2.3%</td><td>-4.9%</td><td>5.3</td><td>11.5</td><td>0.27</td><td>0.37</td></tr><tr><td>ACTIVATIONS</td><td>-11.5%</td><td>-26.8%</td><td>25.5</td><td>51.5</td><td>-2.5%</td><td>-8.1%</td><td>5.5</td><td>17.2</td><td>0.12</td><td>0.45</td></tr><tr><td>RANDOM</td><td>-0.1%</td><td>-0.2%</td><td>0.1</td><td>0.4</td><td>-0.1%</td><td>-0.2%</td><td>0.1</td><td>0.3</td><td>0.00</td><td>0.02</td></tr></table>

Table 3: The DEL Benchmark. HARD CONCRETE is the most effective method in erasing the target sequence (Self), while IG can best maintain the LLM performance on unrelated sequences (Neg and Rand) after dropout.

Pruning-based methods perform the best. SLIMMING and HARD CONCRETE, the methods based on network pruning, substantially outperform the other methods across all setups. Specifically, HARD CONCRETE achieves Recall@0.5% higher than 80 in three out of four LLMs. ZERO-OUT and IG perform similarly and outperform the simple method ACTIVATIONS overall, but are much more computationally expensive than the other methods (see comparisons in A.7).

Our results hold under more data and different random seeds. In the appendix, we show that our conclusions hold when expanding the INJ Benchmark to the newly released ECBD 2022, 2023 dataset (Padmanabhan et al., 2023) (A.5), and they are robust to the choice of random seed, which controls the choice of injected weights (A.6).

## 5.2 DEL Benchmark Results

Table 3 shows to what extent dropping out k = 0.1, 0.5 % of neurons predicted by different methods makes LLMs forget the target sequence x (Self), while still memorizing the other sequences x (Neg), and keeping the perplexity on the random batch (Rand-PPL) intact. We evaluate one target sequence at a time and report the average absolute changes (∆) in Accuracy (Acc), Levenshtein distance (Dist), and perplexity after dropout.

All methods show evidence of localization. Randomly dropping out the same number of neurons (RANDOM) barely changes the LLM behavior. In comparison, all five localization methods successfully identify neurons that contribute much more to memorizing the target sequence than to negative examples, showing evidence of their localization ability on real-world memorized data.

Methods trade off between ∆ Self and ∆ Neg. We find SLIMMING and HARD CONCRETE much more effective than other methods in erasing the target sequence itself. However, they are worse at preserving LLM memorization of the negative examples and the perplexity of randomly sampled sequences. For example, dropping out 0.5% of GPT2 neurons predicted by SLIMMING decreases Accuracy by 57.8% and increases 75.4 characters in Levenshtein distance on the target sequence, but it also hurts the Accuracy on negative examples by 6.4% and increases Levenshtein distance by 7.5 on average. On the other hand, IG best maintains the performance on negative examples and perplexity, but is not as successful in erasing the target sequence itself. Interestingly, although ZERO-OUT assigns the attribution scores with a leave-one-out approach, it does not perform the best on either target sequences or negative examples, implying that the individual neuron dropout effect does not reliably predict the collective effect of dropping out many neurons at the same time. Overall, it is challenging for methods to effectively and specifically locate the target sequence at the same time.

Which negative examples are forgotten? We analyze how the negative examples affected by dropout are related to the target sequence. Figure 2 is the confusion matrix on a representative subset of GPT2 memorized data, $\mathcal { V } \subset \mathcal { X }$ , where each row shows how dropping out 0.5% of the neurons predicted by HARD CONCRETE on a target sequence changes the Accuracy of every sequence in . We group sequences under the same category (see Table 1) in adjacent rows. We find HARD CON-CRETE sometimes confuses related data; for example, in the Address category consisting of mailing addresses, dropping out the neurons of an address sequence also causes substantial Accuracy drops on other addresses. We also find confusion across the Poems, Shakespeare, and Bible categories of literary sequences. Qualitatively, we found several web pages containing famous quotes from different poems and books; such co-occurrences may also appear multiple times in GPT2’s pretraining corpus and may explain why in Figure 2, a small set of neurons affect quotes from different sources. While these findings could suggest that HARD CONCRETE struggles to pinpoint neurons that are specific to a target sequence, it may also be that LLMs actually use a shared set of neurons to memorize several related sequences. Figure 5 in A.8 shows the confusion matrices of other methods and Figure 6 is the matrix of the entire dataset . Both figures share patterns similar to Figure 2.

## 5.3 Concurrence of the two benchmarks

This section studies if the two benchmarks rank the methods similarly (Liu et al., 2023) and whether the differences between methods are significant.

Rankings of localization methods. The INJ Benchmark, which solely evaluates the injected target sequences,<sup>3</sup> and the Self- part of the DEL

![](images/469a23767455fbe0891c4c2cbe88c70bf7b39a5c381ef1a2d3f1b634b89215ef.jpg)  
Figure 2: The confusion matrix of HARD CONCRETE on a subset of data memorized by GPT2-XL.

Benchmark show consistent rankings: HARD CON-CRETE performs slightly better than SLIMMING, followed by ZERO-OUT and IG; ACTIVATIONS performs the worst but still substantially outperforms RANDOM. This consistency suggests that despite the differences in data and setups, the two benchmarks reflect the same underlying localization abilities of different methods. We believe the reason pruning-based methods perform better is that they learn to mask multiple neurons simultaneously, while other methods only consider the importance of each neuron individually.

Tests of significance. We run t-tests to test if pruning-based methods outperform IG significantly. For the INJ Benchmark, each method has 24 Recall@k% scores in Table 2; we run 24 onetailed paired t-tests accordingly. With Bonferroni correction, we set the significance level $\begin{array} { r } { \alpha = \frac { 0 . 0 5 } { 2 4 } } \end{array}$ Table 10 in the appendix shows that for HARD CONCRETE vs. IG and SLIMMING vs. IG, respectively, there are 23/24 and 24/24 tests that have p-values < α. Similarly, in the DEL Benchmark, each method has $6 ~ \Delta$ Self-Acc scores in Table 3; thus, we run 6 paired t-tests. Table 11 shows that 5/6 and 6/6 tests have p-values $< ~ \frac { 0 . 0 5 } { 6 }$ , for SLIMMING vs. IG and HARD CONCRETE vs. IG, respectively. Notably, for both benchmarks, most tests have p-values $< 1 0 ^ { - 1 0 }$ . Overall, these results support our claims that the difference between pruning-based methods and IG is significant.

## 5.4 Is the memory of a piece of information distributed over layers?

To understand the individual effect of each layer on memorization, we study the alternative that drops out the same number of neurons in a single layer. In §5.2, a method predicts top-0.1% of neurons in every layer after the bottommost layer; thus, we have a “budget” of $N = 0 . 1 \% \times 6 4 0 0 \times ( 4 8 - 1 )$ neurons for GPT2-XL. Here, the alternative strategy drops out the top-N neurons in a single layer in terms of the attribution scores assigned by a method.

Using the attribution assigned by IG, Figure 3 compares the two dropout strategies, illustrating their ∆ Self-Acc and ∆ Neg-Acc scores (see more methods in A.10). First, we find dropping out neurons in multiple layers much more efficient in erasing the target sequence, as the horizontal blue line shows a greater decrease in Self-Acc than the solid blue line, suggesting that memorized information is stored in a distributed fashion over many layers, not concentrated in a single layer. The only exception is dropping out neurons in Layer 1; however, it also greatly hurts Neg-Acc. The large memorization decreases on all sequences may imply that the bottom layers of LLMs mainly work on processing basic and general information (Tenney et al., 2019), instead of focusing on a specific sequence.

## 6 Related Work and Discussion

Localization identifies function-specific components, including neurons (Radford et al., 2017; Gurnee et al., 2023), layers (Gupta et al., 2023), or subnetworks (Csordás et al., 2021; Cao et al., 2021; Foroutan et al., 2022). For example, Dai et al. (2022) find knowledge neurons for each relational fact. Meng et al. (2022) locate relational facts to middle FFNs, specifically when LLMs process the last token of the subject. Bayazit et al. (2023) discover sparse knowledge subnetworks in GPT2 with a differentiable weight masking method. However, there is no standard approach to evaluate the effectiveness of localization methods. We are the first to systematically and directly compare different methods on LLMs of different sizes, including knowledge neurons (IG) and differentiable masking methods SLIMMING and HARD CONCRETE.

We take the view that LLM memorization of a sequence is different from learning a type of knowledge. Memorization is reproducing a long sequence (near) verbatim. In contrast, knowledge, often represented as a <subject, relation, object> triplet, occurs in variable contexts, where paraphrases are treated as equivalent expressions of the same knowledge. Localization of verbatim memorization helps unlearn private or copyrighted data, for example, Wu et al. (2023) apply IG to localize and then erase private data from a BERT finetuned on Enron Email dataset (Klimt and Yang, 2004). Our DEL Benchmark differs from Wu et al. (2023) in two main ways: (1) we delete sequences that LLMs have naturally memorized during pretraining, (2) we locate neurons for each sequence independently, rather than finding a shared set of neurons, as our collected datasets cover diverse sequences. Localization can also prevent overfitting: Maini et al. (2023) drop out pre-allocated neurons tied to memorizing mislabeled examples. In contrast with these works, we focus on benchmarking localization ability, since successful localization is the basis of its downstream applications.

![](images/f2a133307feef7123158f9c527a7674b9df73476ea874160ac735363bcea14f3.jpg)  
Figure 3: Dropout in one layer vs. multiple layers.

## 7 Conclusion

We propose two benchmarking approaches to define the success of LLM localization, focusing on locating a small set of neurons in an LLM that are responsible for memorizing a sequence. The INJ Benchmark enables a direct evaluation of localization methods, while the DEL Benchmark evaluates methods on naturally memorized sequences, using dropout to measure localization success. The two benchmarks complement each other and show consistent rankings of methods. We find promising localization ability of all five methods we evaluate, especially for HARD CONCRETE. Meanwhile, all methods confuse memorized sequences in the same or related categories. This finding suggests a need for better localization methods and poses the open question of whether LLMs use a shared set of neurons to memorize related sequences such that perfect localization is not possible.

## 8 Limitations

We follow prior work (§2) and assume that FFNs are the most important components in LLMs for memorizing data; thus, we only study localization in FFNs, not considering other model components such as attention layers. Similarly, we focus on neurons instead of individual weights in FFNs, so as to make fair comparisons with existing methods, IG and ACTIVATIONS.

In the INJ Benchmark, we assume that all the fine-tuned weights are responsible for memorizing the newly injected sequence. However, there is no guarantee that all of them contribute to memorization. We roughly address this issue by lowering the injection ratio, which makes it less likely for the model to memorize the injected sequence without using all of the chosen weights; indeed, we observe that when the ratio is 10 smaller, all localization methods achieve higher recalls in Table 2, even though the random baseline performs 10 worse.

We acknowledge the limitations of evaluating localization in our DEL Benchmark. First, we use dropout (namely, zero ablation) to observe the importance of the located neurons, which is only one possible way to approach localization; other approaches such as mean ablation (Wang et al., 2023) and path patching (Goldowsky-Dill et al., 2023; Hanna et al., 2023) are not covered in this paper. Besides, given a target sequence, we treat all the other memorized sequences as its negative examples without considering semantic overlap or data sources, as our data deduplication only ensures there is little lexical overlap between sequences (§3.2). However, we find all localization methods show confusion between several quotes, which may share semantic similarities or co-occur in some pretrained documents. It is debatable whether related examples should be considered negative, and it depends on what the goal of localization is. We invite future work to propose new ways to define the success of localization for the DEL Benchmark.

## Acknowledgements

We thank Johnny Wei, Eric Wallace, and Ameya Godbole for their help in finding memorized data. We thank Cheng-Han Chiang for helpful discussions on model editing. We thank Wang Zhu, Ting-Rui Chiang, Joshua Robinson, Lee Kezar, Deqing Fu, Anthony Liang, Ta-Chung Chi, Yi-Lin Tuan, Yau-shian Wang, and the anonymous reviewers for their valuable feedback. We thank USC NLP cluster admin for their great work on keeping the servers functional. This work was funded in part by gifts from Open Philanthropy and Cisco Research.

## References

Deniz Bayazit, Negar Foroutan, Zeming Chen, Gail Weiss, and Antoine Bosselut. 2023. Discovering knowledge-critical subnetworks in pretrained language models. ArXiv preprint, abs/2310.03084.

Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. 2023. Pythia: A suite for analyzing large language models across training and scaling.

Lucas Bourtoule, Varun Chandrasekaran, Christopher A Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. 2021. Machine unlearning. In 2021 IEEE Symposium on Security and Privacy (SP), pages 141–159. IEEE.

Steven Cao, Victor Sanh, and Alexander Rush. 2021. Low-complexity probing via finding subnetworks. In Proceedings of the 2021 Conference of the North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, pages 960–966, Online. Association for Computational Linguistics.

Yinzhi Cao and Junfeng Yang. 2015. Towards making systems forget with machine unlearning. In 2015 IEEE symposium on security and privacy, pages 463– 480. IEEE.

Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos, and Dawn Song. 2019. The secret sharer: Evaluating and testing unintended memorization in neural networks. In 28th USENIX Security Symposium (USENIX Security 19), pages 267–284.

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. 2021. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pages 2633–2650.

Xiaohan Chen, Yu Cheng, Shuohang Wang, Zhe Gan, Zhangyang Wang, and Jingjing Liu. 2021. Early-BERT: Efficient BERT training via early-bird lottery tickets. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2195–2207, Online. Association for Computational Linguistics.

Róbert Csordás, Sjoerd van Steenkiste, and Jürgen Schmidhuber. 2021. Are neural nets modular? inspecting functional modularity through differentiable

weight masks. In International Conference on Learning Representations.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. 2022. Knowledge neurons in pretrained transformers. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8493– 8502, Dublin, Ireland. Association for Computational Linguistics.

Negar Foroutan, Mohammadreza Banaei, Rémi Lebret, Antoine Bosselut, and Karl Aberer. 2022. Discovering language-neutral sub-networks in multilingual language models. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 7560–7575, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. 2021. The pile: An 800gb dataset of diverse text for language modeling. ArXiv preprint, abs/2101.00027.

Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. 2023. Dissecting recall of factual associations in auto-regressive language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12216–12235, Singapore. Association for Computational Linguistics.

Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Goldberg. 2022. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 30–45, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. 2021. Transformer feed-forward layers are keyvalue memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5484–5495, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Nicholas Goldowsky-Dill, Chris MacLeod, Lucas Sato, and Aryaman Arora. 2023. Localizing model behavior with path patching. ArXiv preprint, abs/2304.05969.

Zhuocheng Gong, Di He, Yelong Shen, Tie-Yan Liu, Weizhu Chen, Dongyan Zhao, Ji-Rong Wen, and Rui Yan. 2022. Finding the dominant winning ticket in pre-trained language models. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pages 1459–1472, Dublin, Ireland. Association for Computational Linguistics.

Anshita Gupta, Debanjan Mondal, Akshay Sheshadri, Wenlong Zhao, Xiang Li, Sarah Wiegreffe, and Niket

Tandon. 2023. Editing common sense in transformers. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 8214–8232, Singapore. Association for Computational Linguistics.

Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, and Dimitris Bertsimas. 2023. Finding neurons in a haystack: Case studies with sparse probing. Transactions on Machine Learning Research.

Song Han, Huizi Mao, and William J Dally. 2016. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In International Conference on Learning Representations.

Michael Hanna, Ollie Liu, and Alexandre Variengien. 2023. How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model. In Thirty-seventh Conference on Neural Information Processing Systems.

Peter Hase, Mohit Bansal, Been Kim, and Asma Ghandeharioun. 2023. Does localization inform editing? surprising differences in causality-based localization vs. knowledge editing in language models. In Thirtyseventh Conference on Neural Information Processing Systems.

Babak Hassibi and David Stork. 1992. Second order derivatives for network pruning: Optimal brain surgeon. In Advances in Neural Information Processing Systems, volume 5. Morgan-Kaufmann.

Eric Jang, Shixiang Gu, and Ben Poole. 2017. Categorical reparameterization with gumbel-softmax. In International Conference on Learning Representations.

Bryan Klimt and Yiming Yang. 2004. Introducing the enron corpus. In CEAS, volume 45, pages 92–96.

Jooyoung Lee, Thai Le, Jinghui Chen, and Dongwon Lee. 2023. Do language models plagiarize? In Proceedings of the ACM Web Conference 2023, pages 3637–3647.

Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. 2022. Deduplicating training data makes language models better. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8424–8445, Dublin, Ireland. Association for Computational Linguistics.

Eric Lehman, Sarthak Jain, Karl Pichotta, Yoav Goldberg, and Byron Wallace. 2021. Does BERT pretrained on clinical notes reveal sensitive data? In Proceedings of the 2021 Conference of the North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, pages 946–959, Online. Association for Computational Linguistics.

Vladimir I. Levenshtein. 1965. Binary codes capable of correcting deletions, insertions, and reversals. Soviet physics. Doklady, 10:707–710.

Jiwei Li, Will Monroe, and Dan Jurafsky. 2016. Understanding neural networks through representation erasure. ArXiv preprint, abs/1612.08220.

Nelson F. Liu, Tony Lee, Robin Jia, and Percy Liang. 2023. Do question answering modeling improvements hold across benchmarks? In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13186–13218, Toronto, Canada. Association for Computational Linguistics.

Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang. 2017. Learning efficient convolutional networks through network slimming. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 2755–2763.

Christos Louizos, Max Welling, and Diederik P. Kingma. 2018. Learning sparse neural networks through l\_0 regularization. In International Conference on Learning Representations.

Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. 2017. The concrete distribution: A continuous relaxation of discrete random variables. In International Conference on Learning Representations.

Pratyush Maini, Michael Curtis Mozer, Hanie Sedghi, Zachary Chase Lipton, J Zico Kolter, and Chiyuan Zhang. 2023. Can neural network memorization be localized? In Proceedings of the 40th International Conference on Machine Learning.

Kevin Meng, David Bau, Alex J Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems.

Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D Manning. 2022. Fast model editing at scale. In International Conference on Learning Representations.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. 2022. In-context learning and induction heads. Transformer Circuits Thread.

Yasumasa Onoe, Michael Zhang, Eunsol Choi, and Greg Durrett. 2022. Entity cloze by date: What LMs know about unseen entities. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 693–702, Seattle, United States. Association for Computational Linguistics.

Shankar Padmanabhan, Yasumasa Onoe, Michael Zhang, Greg Durrett, and Eunsol Choi. 2023. Propagating knowledge updates to lms through distillation. In Advances in Neural Information Processing Systems, volume 36, pages 47124–47142. Curran Associates, Inc.

Abhishek Panigrahi, Nikunj Saunshi, Haoyu Zhao, and Sanjeev Arora. 2023. Task-specific skill localization in fine-tuned language models. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings ofMachine Learning Research, pages 27011–27033.

Alec Radford, Rafal Jozefowicz, and Ilya Sutskever. 2017. Learning to generate reviews and discovering sentiment. ArXiv preprint, abs/1704.01444.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog.

Anton Sinitsin, Vsevolod Plokhotnyuk, Dmitry Pyrkin, Sergei Popov, and Artem Babenko. 2020. Editable neural networks. In International Conference on Learning Representations.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(56):1929–1958.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. 2017. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pages 3319–3328. PMLR.

Ian Tenney, Dipanjan Das, and Ellie Pavlick. 2019. BERT rediscovers the classical NLP pipeline. In Proceedings of the 57th Annual Meeting of the Associationfor Computational Linguistics, pages 4593– 4601, Florence, Italy. Association for Computational Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30.

Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. 2023. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In The Eleventh International Conference on Learning Representations.

Xinwei Wu, Junzhuo Li, Minghui Xu, Weilong Dong, Shuangzhi Wu, Chao Bian, and Deyi Xiong. 2023. DEPN: Detecting and editing privacy neurons in pretrained language models. In Proceedings ofthe 2023

Conference on Empirical Methods in Natural Lan guage Processing, pages 2875–2886, Singapore. Association for Computational Linguistics.

Matthew D Zeiler and Rob Fergus. 2014. Visualizing and understanding convolutional networks. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I 13, pages 818–833. Springer.

Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr, and Nicholas Carlini. 2023. Counterfactual memorization in neural language models. In Thirty-seventh Conference on Neural Information Processing Systems.

Rui Zheng, Bao Rong, Yuhao Zhou, Di Liang, Sirui Wang, Wei Wu, Tao Gui, Qi Zhang, and Xuanjing Huang. 2022. Robust lottery tickets for pre-trained language models. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2211–2224, Dublin, Ireland. Association for Computational Linguistics.

## A Appendix

## A.1 The Loss for Information Injection

In the INJ Benchmark, we use regular language modeling loss to train the LLM θ on a new sequence $x = [ x _ { 1 } , \ldots , x _ { T } ]$ of tokens. Formally,

$$
{ \frac { 1 } { \mathcal { T } - 1 } } \sum _ { t = 2 } ^ { \mathcal { T } } - \log P _ { \theta } ( x _ { t } | x _ { < t } )\tag{13}
$$

Here, the index t starts from 2, because all the LLMs we use (GPT2 and Pythia models) do not add <bos> tokens to data when doing language modeling in their pretraining. Therefore, there is no loss on the first token $x _ { 1 }$ and the total loss is averaged across $\mathcal { T } - 1$ token.

## A.2 Details of IG

Recall that a sequence $x \ = \ ( p , s )$ consists of a prefix p and a suffix $s ~ = ~ [ s _ { 1 } , \ldots , s _ { T } ]$ . Denote $\mathrm { P } ( \hat { h } _ { t } ^ { l } )$ as the LLM output probability of token $s _ { t }$ if we replace the original hidden state at the l-th layer, $h _ { t } ^ { l } \in \mathbb { R } ^ { d _ { 2 } }$ , with a new hidden state $\hat { h } _ { t } ^ { l } \in \mathbb { R } ^ { d _ { 2 } }$

$$
\mathrm { P } ( \hat { h } _ { t } ^ { l } ) = P _ { \theta } ( s _ { t } | p , s _ { < t } , \hat { h } _ { t } ^ { l } )\tag{14}
$$

To calculate the integrated gradients along the i-th neuron dimension, we gradually change $\hat { h } _ { t } ^ { l }$ from a zero $\mathrm { v e c t o r } ^ { 4 }$ to the original hidden state $h _ { t } ^ { l }$ , and cumulating the gradients of $\mathrm { P } ( \cdot )$ along the i-th dimension. Finally, we get the attribution score $\mathcal { A } ^ { l } ( i )$ by averaging the integrated gradients across the suffix length $T \colon$

$$
\mathbf { I G } _ { i } ( z ) : = \ z _ { i } \int _ { \alpha = 0 } ^ { 1 } \frac { \partial \mathrm { P } ( \alpha z ) } { \partial z _ { i } } d \alpha ,\tag{15}
$$

$$
\mathcal { A } ^ { l } ( i ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbf { I G } _ { i } ( h _ { t } ^ { l } )\tag{16}
$$

where $\mathbf { I G } _ { i } ( h _ { t } ^ { l } )$ is the integrated gradients along the i-th neuron dimension in the l-th layer at the t-th timestep, when the input is $[ p , s _ { < t } ]$ . Sundararajan et al. (2017) compute Riemann sum to approximate Eq. 15, which uses a fixed number of intervals to approximate the integrals. We closely follow the implementation of https://github.com/ EleutherAI/knowledge-neurons.

## A.3 Details of SLIMMING

We initialize every mask value $m _ { i } ^ { l }$ as 1, which is equivalent to running the pretrained LLM without masking. When training the mask, we clip every $m _ { i } ^ { l }$ to [0, 1]. Note that for both SLIMMING and HARD CONCRETE, because we are learning a mask on each neuron, we do not apply any random dropout during training.

## A.4 Details of HARD CONCRETE

Louizos et al. (2018) obtain the hard concrete r.v. $\bar { m } _ { i } ^ { l }$ by first stretching the binary concrete r.v. $\hat { m } _ { i } ^ { l }$ (Eq. 11) from the interval (0, 1) to $( \gamma , \zeta )$ , where $\gamma = - 0 . 1 , \zeta = 1 . 1$ , and then clip the value to the [0, 1] interval:

$$
\bar { m } _ { i } ^ { l } = \operatorname* { m i n } \left( 1 , \operatorname* { m a x } \left( 0 , \hat { m } _ { i } ^ { l } \cdot ( \zeta - \gamma ) + \gamma \right) \right)
$$

They then use $L _ { 0 }$ regularization to encourage sparsity on the weights after applying the mask $\bar { m } ^ { l }$ After reparametrization, they have the regularization $\mathcal { R } ( \bar { m } ^ { l } )$

$$
\mathcal { R } ( \bar { m } ^ { l } ) = \sum _ { i = 1 } ^ { d _ { 2 } } \sigma \left( \log m _ { i } ^ { l } - C \right) ,\tag{17}
$$

where $C = \beta$ log $\frac { - \gamma } { \zeta }$ is a constant.

## A.5 Expanding the dataset of INJ Benchmark

We double the data size of the INJ Benchmark by including the newly released ECBD 2022, 2023 splits, having 328 distinct definition sentences from ECBD 2021-2023. We experiment with this expanded dataset on GPT2, injection ratio=0.1%, using the same hyperparameters as Table 2. Table 4 shows the results on ECBD 2021-2023 are very close to the ones on ECBD-2021 only (Table 2), suggesting that our conclusions hold when we increase the dataset size.

<table><tr><td colspan="4">GPT2 124M</td></tr><tr><td> $r a t i o = 0 . { \it I } \%$ </td><td>R@0.1%</td><td>R@0.2%</td><td>R@0.5%</td></tr><tr><td>HC</td><td>58.3</td><td>81.6</td><td>94.6</td></tr><tr><td>Slim</td><td>59.2</td><td>84.3</td><td>94.5</td></tr><tr><td>IG</td><td>53.3</td><td>73.0</td><td>84.1</td></tr><tr><td>Activation</td><td>11.1</td><td>26.5</td><td>52.7</td></tr></table>

Table 4: The INJ Benchmark results of GPT2 on the expanded dataset, ECBD 2021-2023.

## A.6 Do random seeds affect the results of the INJ Benchmark?

In INJ Benchmark, we sample different sets of weights for different examples (see Algorithm 1); thus, the results reported in Table 2 are averaged over many different choices of weights. To further show that random seeds do not affect our results, we run an additional experiment on GPT2, with the injection ratio=0.1%. Specifically, for each example, we choose a different random seed and thus choose a different set of weights to inject the example. Comparing the new results in Table 5 with the original ones in Table 2, we find that the recall scores barely change for all localization methods. Also, for each method, we run paired two-tailed t-tests comparing the recalls between the original and new seeds and observe that all pairs have pvalues > 0.05, suggesting that differences between random seeds are not significant.

<table><tr><td colspan="4">GPT2 124M</td></tr><tr><td>ratio = 0.1%</td><td>R@0.1%</td><td>R@0.2%</td><td>R@0.5%</td></tr><tr><td>HC</td><td>57.9</td><td>81.0</td><td>94.6</td></tr><tr><td>Slim</td><td>59.6</td><td>84.4</td><td>94.8</td></tr><tr><td>IG</td><td>54.0</td><td>73.7</td><td>83.9</td></tr><tr><td>Activation</td><td>11.4</td><td>26.4</td><td>52.7</td></tr></table>

Table 5: The INJ Benchmark results with a new set of random seeds. The Recall@k% scores are very similar to the original ones in Table 2, showing the INJ Benchmark is not sensitive to the choice of random seed.

## A.7 Computation costs of different methods

Among all five localization methods, ACTIVA-TIONS is the most computationally efficient, because Eq. 8 only requires one forward pass. Both the pruning-based methods SLIMMING and HARD CONCRETE perform fast, as only the masks are trainable. Calculating integrated gradients (IG) is time-consuming, while ZERO-OUT is the worst, because it leaves out every neuron one by one. We compare the computational cost of different methods on one sequence memorized by Pythiadeduped-6.9B, where each sequence in the collected set consists of a 32-token prefix and a 48-token suffix. We follow the common implementation that sets the number of intervals to 20 for Riemann sum in IG. Table 6 shows the elapsed time calculated on an RTX A6000 48G GPU. When running IG and ZERO-OUT we patch and batch the activations to reach 99% GPU utilities. Still, applying ZERO-OUT to do localization on one sequence costs 8.5 hours, and  contains 500 sequences in total. Due to the extremely heavy computation cost, we do not have the results of ZERO-OUT on Pythia-6.9B in the DEL Benchmark.

<table><tr><td></td><td>Time</td></tr><tr><td>ACTIVATIONS</td><td>~ 0.3 sec</td></tr><tr><td>SLIMMING</td><td>～12 sec</td></tr><tr><td>HARD CONCRETE</td><td>~ 1 min</td></tr><tr><td>IG</td><td>～43 min</td></tr><tr><td>ZERO-OUT</td><td>～ 8.5 hr</td></tr></table>

Table 6: The elapsed time of different methods to do localization (i.e., assign attribution scores to every neuron) on one sequence memorized by Pythia-6.9B. We time all methods on a single RTX A6000 GPU.

## A.8 Details of Data Collection

We show some collected examples in Tables 12&13. Table 7 reports how well the pretrained LLMs memorize sequences in the collected datasets.

<table><tr><td></td><td>Acc</td><td>Dist</td><td>PPL</td><td>Len</td></tr><tr><td>GPT2-XL</td><td>99.3%</td><td>0.48</td><td>10.18</td><td>150</td></tr><tr><td>Pythia-deduped-2.8B</td><td>98.8%</td><td>1.07</td><td>5.58</td><td>160</td></tr><tr><td>Pythia-deduped-6.9B</td><td>99.7%</td><td>0.20</td><td>5.24</td><td>167</td></tr></table>

Table 7: Quantifying memorization of the collected datasets. The high Accuracy (Acc) and low Levenshtein distance (Dist) show our collected sequences ( ) are indeed well memorized by LLMs. The last column (Len) reports the average suffix length of each dataset at the character level. We also measure the perplexity (PPL) on sequences sampled from the Pile-dedupe ( ).

The pretrained sequences of Pythia models. EleutherAI releases the exact batches used by Pythia models during pretraining, where each sequence in a batch consists of 2049 tokens <sup>5</sup>. We first randomly downsample the pretraining batches to a subset  of 102400 sequences. Then, we use our criteria in §3.2 to determine whether Pythia memorizes a sequence in the subset. After filtering, there remain 500 1000 sequences in the subsets for both Pythia-deduped-2.8B and Pythia-deduped-6.9B; we simply sample 505 of them respectively as our collected datasets.

We also randomly sample a subset of 2048 sequences ( ), each consisting of 128 tokens, to measure the perplexity of all LLMs we evaluate. We ensure that ${ \mathcal { Z } } \cap { \mathcal { D } } = \emptyset$ , so there is no overlap between the collected memorized sequences and sequences for perplexity.

Filtering with greedy decoding. Given the prefix p as the prompt to the LLM, we generate the suffix $\bar { s } = [ \bar { s _ { 1 } } , \dotsc , s _ { 4 8 } ^ { - } ]$ with greedy decoding, where

$$
\bar { s } _ { t } = \operatorname * { a r g m a x } _ { w \in \mathrm { V o c } } P _ { \theta } ( w | p , \bar { s } _ { < t } ) .\tag{18}
$$

We then calculate the Levenshtein distance (Levenshtein, 1965) between the true suffix s and the generated one s¯, filtering out sequences with a distance greater than 20 characters. Note s¯ is different from sˆ in $\mathrm { E q } ~ 4 .$ , which is generated by teacher-forcing and is used to calculate memorization scores.

Deduplication. Although we use the deduplicated version of the dataset and models, the Pilededupe and Pythia-deduped models, we still find lots of near-duplicated sequences. Thus, we further deduplicate the collected memorized sequences. In particular, we follow Lee et al. (2022) to represent each sequence with a set of 5-grams when calculating the Jaccard index. Among a set of duplicates, we select the one that is best memorized, i.e., having the lowest Levenshtein distance on the generated suffix $\bar { s _ { t } }$ (Eq. 18), and discard the others.

Manually searched data. With our searching criteria in §3.2, we can only identify less than 10 memorized sequences from subsets of the Pilededupe, Common Crawl, and Wikipedia, probably because OpenAI carefully preprocesses the data before training GPT2-XL. Thus, we actively search for potentially memorized data, such as famous poems and common lists of sorted items. We collect 105 sequences memorized by GPT2-XL and manually categorize them (see Tables 1 & 12), including 31 examples from Carlini et al. (2021). We set the prefix and suffix of a sequence by trial and error to ensure high memorization Accuracy. Unlike automatic searches that tend to find templated texts or uninteresting data with repetitive tokens (Zhang et al., 2023), our manual search ensures better data quality and enables us to analyze memorization within and across categories.

In particular, Figures 5 & 6 show that different localization methods constantly confuse sequences of related categories. For example, they are unable to disentangle neurons of different quotes and identify a small set of neurons responsible for both the order of Zodiac Signs and the order of Planets.

Responsible checklist. Note the Contact Info category of our manually collected dataset only contains public data, such as mailing addresses of corporate headquarters and famous buildings; thus, it does not have any potential risk of revealing private information. Similarly, our memorized datasets for Pythia models are a subset of the Pile, a public corpus under the MIT License.

## A.9 Hyperparameters

In the INJ Benchmark, the ECBD-2021 set contains 156 definition sequences. For the DEL Benchmark, we collect a set of 505, 505, and 105 sequences memorized by Pythia-deduped-6.9B, Pythia-deduped-2.8B, and GPT2-XL, respectively. For each set, we sample 5 sequences as the dev set, using the dev set performance to determine the hyperparameters for each LLM. The hyperparameters include the integrated gradient steps, i.e., the number of intervals in Riemann sum for integral approximation in IG; the temperature $\beta$ and the initialization value of parameters m in HARD CON-CRETE; the learning rate, the number of training epochs, and λ, which balances the memorization loss and the sparsity loss, in SLIMMING and HARD CONCRETE. We observe that both SLIMMING and HARD CONCRETE are sensitive to the choice of hyperparameters. On the other hand, we find the performance of IG does not improve when using more integrated gradient steps, where we experiment with different steps ranging from 20 to 300. Thus, we set the step to 20 for all examples to reduce the heavy computation costs.

## A.10 More experiments comparing dropout in one layer with multiple layers.

In §5.4, neurons are localized by IG. In this section, we conduct the same experiment using ZERO-OUT and ACTIVATIONS methods. Figure 4 shows that dropping out N neurons in multiple layers (Self-0.1% 47 layers) even outperforms dropping out $5 \times N$ neurons in a single layer (Self-23.5% 1 layer), except for the bottom layers where the memorization of both target and negative examples are greatly hurt. Hence, we believe the memory of a piece of information is distributed across layers; meanwhile, only a few weights in each layer are mainly responsible for the memorization (§5.2).

We do not have the single-layer results of SLIM-MING and HARD CONCRETE, because both methods train the masks of all neurons jointly, which requires us to retrain the masks only on a single layer to obtain their attribution scores. In comparison, the other three methods consider each neuron individually, allowing us to use the same attribution scores to select neurons in a single layer and make direct comparisons with the results in Table 3 (the dashed lines in Figure 4).

![](images/8247ead3042e1728b71619801ea5f936317b203300332fb78454f490971e9e5d.jpg)  
Figure 4: The DEL Benchmark results of ZERO-OUT, IG, and ACTIVATIONS methods when dropping out the same number of neurons in a single layer, where the blue lines show ∆ Self-Acc and the red lines show ∆ Neg-Acc. Under the same “neuron budget”, dropping out neurons in multiple layers (blue dashed lines) substantially outperforms dropout in a single layer, implying that memorized information is stored in a distributed fashion over multiple layers. Besides, dropping out neurons in the bottom layer greatly hurts the memorization of negative examples (red lines), suggesting that the bottom layer encodes general information.

## A.11 Predicting top neurons across layers

In the INJ Benchmark, we randomly sample weights across layers to inject the data, instead of sampling a fixed percentage of weights per layer (see Algorithm 1). Hence, it may seem more natural to predict top-k% of neurons across layers; we experiment with this alternative in Table 9.

Comparing the results of Table 2 and Table 9, we find that predicting top neurons per layer outperforms predicting top neurons across layers. This is because all localization methods assign larger attribution scores to neurons in the bottom layers, barely predicting neurons in the upper layers if we rank neurons globally. On the other hand, Table 2 and Table 9 show consistent results. Our findings and the ranking of different methods are coherent whether we rank neurons per layer or globally.

## A.12 Implementation Details

Table 8 summarizes the architectures of LLMs we use. We run most experiments on RTX3090 24G GPUs; experiments involving Pythia-6.9B are run

on RTXA6000 48G GPUs. We use transformers 4.31.0 and pytorch 1.13.

<table><tr><td></td><td># Layers</td><td># Neurons</td></tr><tr><td>GPT2 124M</td><td>12</td><td>3072</td></tr><tr><td>GPT2-XL 1.5B</td><td>48</td><td>6400</td></tr><tr><td>Pythia-deduped-2.8B</td><td>32</td><td>10240</td></tr><tr><td>Pythia-deduped-6.9B</td><td>32</td><td>16384</td></tr></table>

Table 8: The number of layers and the number of FFN neurons in each layer of different LLMs.

<table><tr><td rowspan="2"></td><td colspan="3">GPT2 124M</td><td colspan="3">GPT2-XL 1.5B</td><td colspan="3">Pythia-deduped 2.8B</td><td colspan="3">Pythia-deduped 6.9B</td></tr><tr><td>R@1%</td><td>R@2%</td><td>R@5%</td><td>R@1%</td><td>R@2%</td><td>R@5%</td><td>R@1%</td><td>R@2%</td><td>R@5%</td><td>R@1%</td><td>R@2%</td><td>R@5%</td></tr><tr><td>ratio = 1%</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HARD CONCRETE</td><td>46.6</td><td>66.8</td><td>88.0</td><td>21.8</td><td>25.1</td><td>32.8</td><td>33.3</td><td>48.4</td><td>70.7</td><td>31.5</td><td>47.5</td><td>69.4</td></tr><tr><td>SLIMMING</td><td>43.1</td><td>64.6</td><td>79.9</td><td>5.2</td><td>11.5</td><td>27.0</td><td>33.6</td><td>47.3</td><td>59.8</td><td>35.0</td><td>49.6</td><td>63.4</td></tr><tr><td>ZERO-OUT</td><td>24.0</td><td>36.8</td><td>52.7</td><td>4.2</td><td>7.3</td><td>13.5</td><td>10.1</td><td>14.3</td><td>20.5</td><td></td><td></td><td></td></tr><tr><td>IG</td><td>10.3</td><td>18.1</td><td>36.3</td><td>1.4</td><td>4.8</td><td>12.2</td><td>6.1</td><td>10.8</td><td>21.1</td><td>8.9</td><td>13.9</td><td>24.1</td></tr><tr><td>ACTIVATIONS</td><td>2.5</td><td>4.4</td><td>9.8</td><td>1.5</td><td>2.8</td><td>6.8</td><td>3.2</td><td>5.1</td><td>21.6</td><td>4.1</td><td>6.3</td><td>17.4</td></tr><tr><td>RANDOM</td><td>1.0</td><td>2.0</td><td>5.0</td><td>1.0</td><td>2.0</td><td>5.0</td><td>1.0</td><td>2.0</td><td>5.0</td><td>1.0</td><td>2.0</td><td>5.0</td></tr><tr><td> $r a t i o = 0 . I \%$ </td><td>@0.1%</td><td>@0.2%</td><td>@0.5%</td><td>@0.1%</td><td>@0.2%</td><td>@0.5%</td><td>@0.1%</td><td>@0.2%</td><td>@0.5%</td><td>@0.1%</td><td>@0.2%</td><td>@0.5%</td></tr><tr><td>HARD CONCRETE</td><td>51.2</td><td>77.4</td><td>96.4</td><td>49.8</td><td>57.5</td><td>63.6</td><td>45.6</td><td>65.5</td><td>85.9</td><td>28.7</td><td>40.7</td><td>55.8</td></tr><tr><td>SLIMMING</td><td>62.7</td><td>87.0</td><td>95.4</td><td>18.1</td><td>35.1</td><td>54.0</td><td>45.0</td><td>62.6</td><td>73.6</td><td>39.1</td><td>52.1</td><td>64.3</td></tr><tr><td>ZERO-OUT</td><td>57.4</td><td>81.7</td><td>91.9</td><td>14.7</td><td>20.9</td><td>31.1</td><td>16.4</td><td>20.6</td><td>25.8</td><td></td><td></td><td></td></tr><tr><td>IG</td><td>36.0</td><td>55.0</td><td>75.5</td><td>2.5</td><td>3.5</td><td>6.0</td><td>12.6</td><td>16.4</td><td>21.9</td><td>19.7</td><td>23.6</td><td>28.9</td></tr><tr><td>ACTIVATIONS</td><td>9.0</td><td>12.9</td><td>23.4</td><td>3.5</td><td>4.6</td><td>6.7</td><td>8.0</td><td>16.8</td><td>40.5</td><td>21.2</td><td>31.4</td><td>50.2</td></tr><tr><td>RANDOM</td><td>0.1</td><td>0.2</td><td>0.5</td><td>0.1</td><td>0.2</td><td>0.5</td><td>0.1</td><td>0.2</td><td>0.5</td><td>0.1</td><td>0.2</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.5</td></tr></table>

Table 9: The INJ Benchmark. The average Reacall@k% of different methods when predicting top-k% of neurons across layers. The results are consistent with Table 2, where methods predict a fixed k% of neurons in each layer.

<table><tr><td rowspan="2"></td><td colspan="3">GPT2 124M</td><td colspan="3">GPT2-XL 1.5B</td><td colspan="3">Pythia-deduped 2.8B</td><td colspan="3">Pythia-deduped 6.9B</td></tr><tr><td>R@1%</td><td>R@2%</td><td>R@5%</td><td>R@1%</td><td>R@2%</td><td>R@5%</td><td>R@1%</td><td>R@2%</td><td>R@5%</td><td>R@1%</td><td>R@2%</td><td>R@5%</td></tr><tr><td>injection ratio = 1%</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HARD CONCRETE</td><td>9E-85</td><td>7E-86</td><td>2E-77</td><td>2E - 100</td><td>8E -108</td><td>8E-111</td><td>3E-114</td><td>3E-121</td><td>6E - 138</td><td>5E− 111</td><td>6E-129</td><td>3E - 155</td></tr><tr><td>SLIMMING</td><td>1E-72</td><td>8E-73</td><td>4E- 62</td><td>1E - 66</td><td>5E-77</td><td>2E-84</td><td>3E-111</td><td>5E-119</td><td>4E-119</td><td>1E − 121</td><td>3E-134</td><td>2E - 132</td></tr><tr><td></td><td>@0.1%</td><td>@0.2%</td><td>@0.5%</td><td>@0.1%</td><td>@0.2%</td><td>@0.5%</td><td>@0.1%</td><td>@0.2%</td><td>@0.5%</td><td>@0.1%</td><td>@0.2%</td><td>@0.5%</td></tr><tr><td>injection ratio = 0.1%</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HARD CONCRETE</td><td>6E-03</td><td>9E - 06</td><td>1E - 18</td><td>2E- 96</td><td>3E-87</td><td>6E-73</td><td>7E-91</td><td>7E-110</td><td>1E-127</td><td>2E− 42</td><td>7E-77</td><td>3E-83</td></tr><tr><td>SLIMMING</td><td>3E -10</td><td>4E-17</td><td>2E - 20</td><td>1E - 62</td><td>2E-73</td><td>4E-71</td><td>6E - 96</td><td>7E-101</td><td>4E - 99</td><td>1E- 52</td><td>1E-54</td><td>1E-52</td></tr></table>

Table 10: The p-values of the INJ Benchmark. $H _ { 0 } { \mathrm { : } }$ IG and pruning-based methods, HARD CONCRETE or SLIMMING, have identical expected Recall@k% scores on ECBD 2021 examples. As we have 24 settings in total, we run 24 one-tailed paired t-tests with Bonferroni correction, setting the significance level $\begin{array} { r } { \alpha = \frac { 0 . 0 5 } { 2 4 } } \end{array}$ . We color the results that have p-values $> \alpha$

<table><tr><td></td><td colspan="2">GPT2-XL 1.5B</td><td colspan="2">Pythia-deduped 2.8B</td><td colspan="2">Pythia-deduped 6.9B</td></tr><tr><td>dropout ratio =</td><td>0.1%</td><td>0.5%</td><td>0.1%</td><td>0.5%</td><td>0.1%</td><td>0.5%</td></tr><tr><td>HARD CONCRETE</td><td> $1 . 6 E - 1 0$ </td><td> $3 . 8 E - 1 7$ </td><td> $5 . 3 E - 6 1$ </td><td> $4 . 1 E - 7 8$ </td><td> $2 . 2 E - 6 1$ </td><td> $3 . 2 E - 9 0$ </td></tr><tr><td>SLIMMING</td><td> $1 . 6 E - 0 4$ </td><td> $1 . 8 E - 1 5$ </td><td> $1 . 2 E - 0 1$ </td><td> $8 . 9 E - 5 5$ </td><td> $2 . 4 E - 2 4$ </td><td> $2 . 0 E - 6 0$ </td></tr></table>

Table 11: The p-values of the DEL Benchmark, where we focus on the memorization accuracy of the target examples. $H _ { 0 } { \mathrm { : } }$ IG and pruning-based methods, HARD CONCRETE or SLIMMING, have identical expected $\Delta$ Self-Acc scores on the memorized sequences. As we have 6 settings in total, we run 6 one-tailed paired t-tests with Bonferroni correction, setting the significance level $\begin{array} { r } { \alpha = \frac { 0 . 0 5 } { 6 } } \end{array}$ . We color the results that have p-values $> \alpha$

<table><tr><td colspan="1" rowspan="1">Email</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">Write to Jon Hilsenrath at jon.hilsenrath@wsj.com</td></tr><tr><td colspan="1" rowspan="1">Zodiac Signs</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">Aries Taurus Gemini Cancer Leo Virgo Libra Scorpio Sagittarius Capri-corn Aquarius Pisces</td></tr><tr><td colspan="1" rowspan="1">Patreon</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">Thank you to ourPatreon Supporters:   Saintsofwar,  Anon,Lord_Of_Fapping, Dryzak, Chabalbac, ioNz, LaX, VNT</td></tr><tr><td colspan="1" rowspan="1">Declaration ofIndependence</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">We hold these truths to be self-evident, that all men are created equal,that they are endowed by their Creator with certain unalienable Rights,that among these are Life, Liberty and the pursuit of Happiness.</td></tr><tr><td colspan="1" rowspan="1">Trump</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">Sorry losers and haters, but my I.Q. is one of the highest -and you allknow it! Please don't feel so stupid or insecure, it's not your fault.</td></tr><tr><td colspan="1" rowspan="1">Newton</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">I do not know what I may appear to the world, but to myself I seemto have been only like a boy playing on the sea-shore, and divertingmyself in now and then finding a smoother pebble or a prettier shell thanordinary, whilst the great ocean of truth lay all undiscovered before me.</td></tr><tr><td colspan="1" rowspan="1">Dr. MLK</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">And when this happens, and when we allow freedom ring, when we let itring from every village and every hamlet, from every state and every city,we will be able to speed up that day when all of God's children, blackmen and white men, Jews and Gentiles, Protestants and Catholics, willbe able to join hands and sing in the words of the old Negro spiritual,"Free at last! Free at last! Thank God Almighty, we are free at last"</td></tr><tr><td colspan="1" rowspan="1">Genesis</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">In the beginning God created the heaven and the earth. And the earthwas without form, and void; and darkness was upon the face of the deep.And the Spirit of God moved upon the face of the waters. And God said.Let there be light: and there was light.</td></tr><tr><td colspan="1" rowspan="1">The Road NotTaken</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">Two roads diverged in a yellow wood,\n\nAnd sorry I could not travelboth\n\nAnd be one traveler, long I stood\n\nAnd looked down one asfar as I could\n\nTo where it bent in the undergrowth;\n\nThen took theother, as just as fair,\n\nAnd having perhaps the better claim,\n\nBecauseit was grassy and wanted wear</td></tr><tr><td colspan="1" rowspan="1">Mike Wall Bio</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">Wall\n\nMichael was a science writer for the Idaho National Laboratoryand has been an intern at Wired.com, The Salinas Californian newspaper,and the SLAC National Accelerator Laboratory. He has also worked asa herpetologist and wildlife biologist. He has a Ph.D. in evolutionarybiology from the University of Sydney, Australia, a bachelor's degreefrom the</td></tr><tr><td colspan="1" rowspan="1">Hardware</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">PCs) may be defined as a desktop, floor standing, or portable microcom-puter that includes a system unit having a central processing unit (CPU)and associated volatile and non-volatile memory, including random ac-cess memory (RAM) and basic input/output system read only memory(BIOS ROM), a system monitor, a keyboard, one or more flexible diskettedrives, a CD-ROM drive,</td></tr><tr><td colspan="1" rowspan="1">Contact  Infoof   SkyhorsePublishing</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">, or educational purposes. Special editions can also be created to spec-ifications. For details, contact the Special Sales Department, ArcadePublishing, 307 West 36th Street, 11th Floor, New York, NY 10018or arcade@skyhorsepublishing.com.\n\nArcade Publishing® is a reg-istered trademark of Skyhorse Publishing, Inc.®, a Delaware corpora-tion.\n\nVisit</td></tr><tr><td colspan="1" rowspan="1">Meme</td><td colspan="1" rowspan="1">98%</td><td colspan="1" rowspan="1">a lot; that's great! It's a little awkward to ask, but we need your help. Ifyou have already donated, we sincerely thank you. We're not salespeople,but we depend on donations averaging $14.76 and fewer than 1% ofreaders give. If you donate just $5.00, the price of your coffee, CatholicOnline School could keep thriving. Thank</td></tr><tr><td colspan="1" rowspan="1">Malik Report</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">check that allowed Dvorak to flick the puck over his shoul-der. . . \n\nAbout The Malik Report\n\nThe Malik Report is a destinationfor all things Red Wings-related. I offer biased, perhaps unprofessional-at-times and verbose coverage of my favorite team, their prospects anddevelopmental affiliates. I've joined the Kukla's Korner family with fiveyears of blogging under</td></tr><tr><td colspan="1" rowspan="1">Porn</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">make love to her. She returned the favor with an amazing blowjoband a masterful fuck session...\nENJOY!!!!\n\nThis entire website hasa voluntary content rating to block access by minors. This rating iscompatible with microsoft internet explorer's content filtering functionand\nfacilitates website blocking software. For a tutorial on blockingthis site click here.\nCopyright bangbros.</td></tr><tr><td colspan="1" rowspan="1">Pokémon Fans</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">We're a group of Pokémon fans dedicated to providing the best place onthe Internet for discussing ideas and sharing fan-made content. Welcome!We're glad you're here.\n\nIn order to join our community we need youto create an account with us. Doing so will allow you to make posts,submit and view fan art and fan fiction, download fan-made games,</td></tr></table>

Table 12: Examples of our manually collected data. The prompt (prefix) is colored in brown. The numbers are the Accuracy (Eq. 5) of GPT2-XL on memorizing the sequences, where 100% Accuracy means the true suffix can be fully reconstructed with greedy decoding.

Table 13: Examples of memorized sequences we collect from the Pile-dedupe. The prompt (prefix) is colored in brown. The numbers are the Accuracy (Eq. 5) of Pythia on memorizing the sequences, where 100% Accuracy means the true suffix can be fully reconstructed with greedy decoding.

![](images/c936f68a79e7f36e17539d8010a7aa219502075abbdc0516dfbaa2ae0d9d73fd.jpg)

![](images/10aada94ac971de7a49be528fb506e7b8188b8f47243c79a5f2674a244a4366d.jpg)

![](images/c4b000e192daac49e0b53cd8a25422421f3e7d87717a18ddae6750dad7fde8e3.jpg)

![](images/b947499a40869a07cd2891fad13cb603bd87e78eefdf644a1d28951dba8cf0ce.jpg)  
Figure 5: Confusion matrices of localization methods on a subset of sequences memorized by GPT2-XL, where each row/column represents a sequence. Different methods show similar patterns of confusion.

![](images/89f32c6593fd428306a92e2df8c1278edb3f2ad455e1f8e1526c7ecb921ad427.jpg)  
Figure 6: Confusion matrix of HARD CONCRETE on the entire test set memorized by GPT2-XL. Each row shows how dropping out the predicted neurons (0.5%) on a target sequence changes the Accuracy of all sequences. HARD CONCRETE is unable to disentangle neurons of different quotes, including poems, Bible, books, and some famous people quotes. Also, it finds a small set of neurons responsible for memorizing both the order of Zodiac Signs and the order of Planets.