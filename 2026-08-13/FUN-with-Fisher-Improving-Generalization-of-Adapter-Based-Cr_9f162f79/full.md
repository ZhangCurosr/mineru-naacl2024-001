# FUN with Fisher: Improving Generalization of Adapter-Based Cross-lingual Transfer with Scheduled Unfreezing

Chen Cecilia Liu<sup>1</sup>, Jonas Pfeiffer<sup>2</sup>, Ivan Vulic´<sup>3</sup>, Iryna Gurevych<sup>1</sup>

<sup>1</sup>Ubiquitous Knowledge Processing Lab,

Department of Computer Science and Hessian Center for AI (hessian.AI),

Technical University of Darmstadt

<sup>2</sup>Google DeepMind

<sup>3</sup>Language Technology Lab, University of Cambridge www.ukp.tu-darmstadt.de

## Abstract

Standard fine-tuning of language models typically performs well on in-distribution data, but suffers with generalization to distribution shifts. In this work, we aim to improve the generalization of adapter-based cross-lingual task transfer where such cross-language distribution shifts are imminent. We investigate scheduled unfreezing algorithms—originally proposed to mitigate catastrophic forgetting in transfer learning—for fine-tuning task adapters. Our experiments show that scheduled unfreezing methods close the gap to full fine-tuning and achieve stronger cross-lingual transfer performance, suggesting that these methods can go beyond just mitigating catastrophic forgetting. Next, aiming to understand these empirical findings, we investigate the learning dynamics of scheduled unfreezing using Fisher Information. Our experiments reveal that scheduled unfreezing induces different learning dynamics compared to standard fine-tuning, and provide evidence that the dynamics of Fisher Information during training correlate with cross-lingual generalization performance. We additionally propose a general scheduled unfreezing algorithm that achieves an average of 2 points improvement over four datasets compared to standard fine-tuning and provides empirical evidence for a theory-based justification of the heuristic unfreezing schedule for task adapter training.

## 1 Introduction

In the standard cross-lingual task transfer setup, a typical and often valid assumption is that only English data is available for fine-tuning and validation of a pretrained multilingual model, due to resource constraints in many languages (Hu et al., 2020). However, models trained in this setup also need to generalize well to text inputs provided in other languages: a requirement that can be seen as an extreme but natural case of distribution shifts generalization (Ramponi and Plank, 2020).

Parameter-efficient fine-tuning methods such as adapters (Houlsby et al., 2019; Stickland and Murray, 2019; Bapna and Firat, 2019) with separate language and task components are often used to achieve effective cross-lingual transfer, especially to low-resource languages (Pfeiffer et al., 2020, 2021; Ansell et al., 2021; Parovic et al.´ , 2022). These adapters insert a small number of trainable parameters into a frozen pretrained multilingual language model (e.g., mBERT, Devlin et al., 2019; XLM-R, Conneau et al., 2020) to achieve positive transfer while avoiding catastrophic forgetting (CF, McCloskey and Cohen, 1989) of previously learnt knowledge after adapting to new tasks.<sup>2</sup> In other words, adapters enable catastrophic forgetting free learning, referred to as CF-free in this paper. While efficient, adapter methods often incur a cross-lingual performance gap when compared to full model fine-tuning.

Gradual unfreezing (GU) is a technique which unfreezes layers of deep neural network models from top to bottom during training (Howard and Ruder, 2018). GU was previously proposed for general transfer learning of in-distribution (ID) data in monolingual contexts in NLP, and has been predominantly applied to full fine-tuning. More recently, ‘Linear-Probing-then-Fine-Tuning’ (LPFT, Kumar et al., 2022) was proposed for transfer learning of both ID and distribution-shifted evaluation data using full fine-tuning in computer vision. LPFT first trains the classification layer only, and then the full model. The main notion connecting these methods is training different layers of a neural network by unfreezing layers at different times (i.e., with a schedule). Designed to mitigate CF, these ‘scheduled unfreezing’ methods have shown promising transfer learning results. However, it is unclear whether scheduled unfreezing can do more than just mitigate CF, and benefit CF-free methods and cross-lingual transfer (which is a different type of distribution shift than previously studied).

In this work, we begin by asking the following question: Do scheduled unfreezing methods improve cross-lingual transfer and close the gap to full fine-tuning in the CF-free setting? We use scheduled unfreezing to train task adapters following the standard adapter-based cross-lingual transfer setup of Pfeiffer et al. (2020). We find that scheduled unfreezing enhances generalization, bridging the gap to full fine-tuning, confirming our hypothesis that, since cross-lingual transfer can be seen as a form of distribution shifts, methods such as GU are effective, even in the CF-free setting. Our results suggest that there indeed is more to the original scheduled unfreezing training than just mitigating catastrophic forgetting (Howard and Ruder, 2018).

We further analyze the learning dynamics during training, with a particular focus on GU, using the trace of the Fisher Information Matrix (Kleinman et al., 2023, denoted by tr(F) henceforth). Our experiments reveal 1) that scheduled unfreezing changes the dynamics of tr(F) during training, 2) that tr(F) is a potential proxy for studying crosslingual generalization.

Based on our analysis, we then propose an automatic scheduled unfreezing algorithm based on maximizing the tr(F) (termed Fisher Unfreezing or FUN), to generalize from previous heuristicbased methods. FUN achieves comparable results to heuristic-based methods and provides empirical evidence that GU may implicitly maximize tr(F) during training in our experimental setting.

Contributions. In sum, our contributions are as follows. 1) To the best of our knowledge, we are the first to demonstrate that scheduled unfreezing in adapter training for cross-lingual transfer closes the performance gap to full model fine-tuning. 2) We present a generalized scheduled unfreezing framework that encompasses several existing methods, allowing easy extensions to new algorithms. 3) We demonstrate that Fisher Information is an effective tool for studying generalization in cross-lingual transfer and find that the dynamics of Fisher Information correlate with cross-lingual transfer results. 4) We propose a tr(F)-based scheduled unfreezing method (FUN), which achieves comparable performance to heuristic methods and indicates that GU achieves good generalization in adapter training.

## 2 Related Work

Scheduled Unfreezing. Howard and Ruder (2018) propose Gradual Unfreezing (GU) to mitigate catastrophic forgetting when transferring a pretrained model for monolingual downstream tasks in non-Transformer architectures. Raffel et al. (2022) study GU for transferring full fine-tuning Transformers to in-distribution tasks, and empirically conclude that GU may not be effective. Recently, Kumar et al. (2022) (LPFT, Linear Probing then Fine-Tuning) show promising results for distribution-shifted transfer in the full fine-tuning setting for computer vision tasks. Yang et al. (2022) extend LPFT for full fine-tuning to NLP. Lee et al. (2023) show that fine-tuning different parts of a network helps with generalization under different types of distribution shifts in computer vision. Different from our work, the prior work has not studied scheduled unfreezing methods for cross-lingual transfer in a CF-free setting.

Fisher Information and Generalization. Fisher Information (Fisher, 1925) is an established concept in optimization theory and practice (Amari, 1998), e.g., to measure parameter importance (Kirkpatrick et al., 2017) or for pruning (Singh and Alistarh, 2020). Achille et al. (2019) study learning dynamics<sup>3</sup> in neural network training with Fisher Information. Golatkar et al. (2019) show that Fisher Information correlates with generalization in computer vision and non-Transformer architectures. Jastrzebski et al. (2021) propose to regularize Fisher Information during the training of a neural network for better in-distribution generalization in computer vision. Xu et al. (2021); Sung et al. (2021) create sparse masks using Fisher Information for better parameter-efficient tuning. Concurrently, Lodha et al. (2023) uses Fisher Information selecting layers for fine-tuning transformers. We study Fisher Information in adapters and for cross-lingual transfer, which has the potential to guide the understanding of new methods in this area.

## 3 Methodology

## 3.1 Adapter-Based Cross-lingual Transfer

Adapters are components that are inserted into a multilingual pretrained Transformer model (termed the base model) to efficiently adapt the large base model to a specific language (Pfeiffer et al., 2020) or task (Houlsby et al., 2019).

![](images/0d9387a661197c0e3bf850e889cd6b7c4215734bdccbcf860242a8edcde7a3d2.jpg)  
Figure 1: a) Standard, b) Gradual unfreezing versus c) tr(F)-based scheduled unfreezing for training task adapters in adapter-based cross-lingual transfer. The classifier is not shown and is always trainable. All other components excluding task adapters, such as the original parameters of the base model and language adapters, are always frozen.

The prior adapter-based fine-tuning process typically spans two stages for cross-lingual task transfer. First, different language adapters are trained (often separately, per language) with the base model frozen, using the masked language modelling (MLM) objective in a target language. Second, task adapters and a task-specific output head are randomly initialized and inserted into the base model along with now-trained source language (i.e., typically English) adapters. In this stage, only the task adapters and a task-specific output head are trainable, while all other parameters are kept fixed. At inference time, the source language adapters are replaced by the language adapter of the target language, while the task adapter is retained, to achieve zero-shot cross-lingual transfer. In this work, we base our main experiments on the state-of-the-art cross-lingual adapter framework: MAD-X (Pfeiffer et al., 2020). See Figure 6 in Appendix A for architecture details.

## 3.2 Gradual Unfreezing and General Scheduled Unfreezing

First proposed by Howard and Ruder (2018), gradual unfreezing (GU) tunes a subset of parameters of a pretrained model starting from the top layer (see Figure 1b). Given a model with L layers, assuming that the index L 1 refers to the top layer, and interval k, GU unfreezes each layer starting from $L - 1$ to 0 in order, every k steps. Once a layer is unfrozen, it remains unfrozen. Hence, the number of trainable parameters increases every k steps under the GU regime.

Algorithm 1 Generalized Scheduled Unfreezing   
Require: An L-layer model with layer $j \in \{ 0 , \dots , L - 1 \}$ parameterized by   
θ<sub>j</sub> . An additional task-specific classification head C. Total training budget   
(steps) N. Training interval k. Typically kL N for convergence.   
Initialize C, θ<sub>j</sub> for all j   
S ← {<sup>C</sup>}   
for i = 0 . . . N do   
Sample a data batch b  D   
if i mod k == 0 and i kL then   
= SELECT( ) ▷ Set of layer (task adapter) indices to unfreeze.   
$\overline { { S } }  S \cup \{ \theta _ { j } : \overline { { j } } \in \mathcal { I } \}$   
end if   
FORWARD( )   
GRADIENT\_UPDATE( )   
end for

Let SELECT( ) be a layer-selection function. Let FORWARD( ) be the standard forward pass through all layers, and for a subset  of layers of the network, let GRADIENT\_UPDATE( ) denote calculating gradients and performing updates on the parameters in . We define a generalized scheduled unfreezing algorithm that encompasses GU, LPFT, and even the recent Surgical fine-tuning method (Lee et al., 2023), as well as the other variants we propose later in this work, in Algorithm 1.<sup>4</sup>

## 3.3 Fisher Information

We use the Fisher Information Matrix (F) to investigate changes in learning dynamics. Recent studies have shown that the Fisher Information Matrix correlates well with the generalization capabilities of neural network models (Golatkar et al., 2019; Jastrzebski et al., 2021). Conveniently, as a 2nd-order metric based on gradients, F also provides insights into the optimization process.

In particular, we take the trace of $F \left( \mathrm { i . e . , t r } ( F ) \right)$ since the full F is computationally expensive to obtain, and previous work has shown the $\operatorname { t r } ( F )$ correlates well with F and shows similar general trends as the full F (Achille et al., 2019). Let x be data input and consider a network parameterized by weights w that encodes the approximate posterior distribution $p _ { w } ( y | x )$ . The tr(F) is computed using the empirical data distribution $\hat { Q } ( x )$ as follows:

$$
\mathrm { t r } ( F ) = \mathbb { E } _ { \boldsymbol { x } \sim \hat { Q } ( \boldsymbol { x } ) } \mathbb { E } _ { \hat { \boldsymbol { y } } \sim p _ { w } ( \hat { \boldsymbol { y } } \vert \boldsymbol { x } ) } | | \nabla _ { w } \log p _ { w } ( \hat { \boldsymbol { y } } \vert \boldsymbol { x } ) | | ^ { 2 }\tag{1}
$$

Note that Eqn. (1) is not the “empirical Fisher” (Kunstner et al., 2019, empirical Fisher uses the true data label $y )$ . Hence, one does not need the labels of input data y to calculate the true F. They are sampled from the label distribution of the task (i.e., $\hat { y } \sim p _ { w } ( \hat { y } | x )$ , see pseudo-code in Appendix D).

One interpretation of $F$ is that given w and a perturbed version of $w ^ { \prime }$ (after applying one step of gradient descent, for example), the KL divergence between $p _ { w } ( y | x )$ and $p _ { w ^ { \prime } } ( y | x )$ is given by $\delta w \cdot F \delta w + O ( \delta w ^ { 3 } )$ (up to 2nd-order approximation, where δw is the small perturbation in weights) (Martens, 2020).

F can be considered as a measure of how much a change in weights can affect the network output (i.e., how much information resides in the weights). Intuitively, this means a set of weights with nearzero entries in F likely means they do not significantly affect the network output (and thus task performance). Moreover, F is also a 2nd-order approximation of the Hessian of the loss function (Shun-ichi and Hiroshi, 2000; Martens, 2020) and provides information on the curvature of the loss landscape near the current weights, that is, how fast the gradients change during the optimization.

## 4 Experiments

## 4.1 Models

Base Models. The main experiments are conducted with two established pretrained multilingual models: mBERT (base-cased, Devlin et al., 2019) and XLM-R (base, Conneau et al., 2020). Adapters. $( ^ { A d a } )$ We follow the adapter configurations from MAD-X (Pfeiffer et al., 2020) for cross-lingual transfer, see §3.1. We use pretrained language adapters available in the AdapterHub.<sup>5</sup>

Scheduled Unfreezing Methods. We apply and analyze two scheduled unfreezing methods from research in other areas of NLP and computer vision to the task of adapter-based cross-lingual transfer. LPFT (Kumar et al., 2022, +LPFT) first trains the classifier head (linear probing, LP) with the base model frozen, then unfreezes the entire model for fine-tuning (FT). In our setup, we first fine-tune the classifier, then followed by a step of unfreezing all the adapters for fine-tuning.

Gradual Unfreezing (Howard and Ruder, 2018, +GU) performs top-down unfreezing during training or fine-tuning. We fine-tune with the classifier and the top-most adapter unfrozen, and for every k steps we unfreeze the next adapter and continue.

## 4.2 Datasets and Hyperparameters

We conduct experiments on a diverse set of tasks and target languages. We use MLQA (Lewis et al., 2020) and XQuAD (Artetxe et al., 2020) for question answering (SQuAD, Rajpurkar et al., 2016 for training). We use XNLI (Conneau et al., 2018) for natural language inference (training on MNLI, Williams et al., 2018), and we use XCOPA (Ponti et al., 2020) for evaluating causal commonsense reasoning (training on COPA, Roemmele et al., 2011). The data statistics and language codes are summarized in Appendix C. We experiment in the zero-shot setting with English-only task data for training and validation.

Hyperparameters. We perform a hyperparameter search with the learning rates of [1e-4, 2e-4, 5e-4, 8e-4] for our main experiments on all datasets except COPA. For COPA, we found a smaller learning rate (1e-5) is better for scheduled unfreezing methods. See Appendix B for detailed hyperparameters per task per model. For scheduled unfreezing experiments, we search for the hyperparameter k in the following range [25, 50, 100, 800, 1000]. The reported results are averaged across 4 runs on A100 or V100 GPUs.

## 5 Results and Analysis

## 5.1 Scheduled Unfreezing Improves Cross-Lingual Generalization

Figure 2 shows the relative performance of (a) task adapters fine-tuned in the standard way $( ^ { A d a } ) , ( \mathsf { b } ) \bar { \mathbf { G } } \mathsf { U } - ( ^ { A d a } + \mathbf { G } \mathsf { U } )$ and LPFT-tuned adapters (<sup>Ada</sup>+LPFT) compared to full fine-tuning with adapters following the AdapterHub recommendations for hyperparameter values. Please see Appendix B for details.

mBERT and XLM-R on all datasets.<sup>6</sup> We report the results averaged across all target languages in the respective datasets. Our experiments show that both LPFT and GU are effective in closing the gap to full fine-tuning across all tasks and models. Moreover, GU-trained task adapters perform better, even exceeding the performance of full fine-tuning in some cases.<sup>7</sup>

Our results suggest that scheduled unfreezing can do more than just mitigate catastrophic forgetting. Even in a CF-free setting like ours, they achieve better generalization for cross-lingual transfer. We focus on GU as the scheduled unfreezing method for further analyses, since it produced better empirical results than LPFT.

Since the training data for both XQuAD and XNLI are well over 50k instances; this amount of annotated task data might be unrealistic for many tasks in practice, even in English. We simulated low-data training settings (details and results are in Appendix G, Table 11), and found that even with a smaller amount of training data, we still observe the advantages of GU over standard task adapter fine-tuning.

## 5.2 Scheduled Unfreezing Beyond Mitigating Catastrophic Forgetting

To understand why scheduled unfreezing helps even in the CF-free setting, we examine the learning dynamics during the training of task adapters.

Due to the unfreezing of task adapters at different times, the model has access to a different number of trainable parameters, which affects the optimization and information encoding for adapters differently under scheduled unfreezing than in standard fine-tuning. Hence, we draw our attention to higher-order metrics, captured in tr(F).

GU Changes Learning Dynamics. We plot tr(F) (moving average, normalized by the number of trainable adapters at the given step) during optimization. Figure 3 shows tr(F) during training on three datasets.<sup>8</sup> The plots indicate that GU significantly changes the learning dynamics of adapters compared to their standard fine-tuning. With GU, due to the model having fewer parameters to encode the same amount of data initially, the tr(F) curve is higher than in standard fine-tuning. The training process induces a tr(F) curve that has a distinctive “hill” shape (i.e., a “learning period", with fast changes of gradients, and hence weights).

Effects of the Unfreezing Schedule on tr(F) and Generalization. While we have empirically validated that scheduled unfreezing changes tr(F) during learning, it is unclear which factors are the main drivers that impact tr(F), and potentially improve generalization. Previous work has studied factors such as learning rate, weight decay, optimizer and loss functions (Golatkar et al., 2019; Jastrzebski et al., 2021). However, based on this work, another novel relevant factor is the unfreezing schedule (i.e., the number of parameters to update at each optimization step). Here, we aim to further understand how unfreezing schedules change the learning dynamics and generalization.

We found that the sensitivity of the learning dynamics to schedules depends on the dataset and the base model. Hence, we focus on XLM-R for MLQA/XQuAD and mBERT for XNLI. These two settings are the most sensitive to schedules and best illustrate the effects, but they are different in terms of models and tasks to examine general patterns.

We randomly sampled 9 schedules (which are effectively permutations of layer indices, where we also treat the standard top-down order as the 10th schedule) that start unfreezing at either layer 10 or 9. The remaining experimental conditions are unchanged.<sup>9</sup> We then aggregate runs with similar cross-lingual transfer results.<sup>10</sup>

We plot the tr(F) along with the validation metrics in Figure 4. We observe that decreases in tr(F) from the peak value are associated with rapid generalization (cf., a drastic increase of the validation metrics) of the network. Previous work points out that the peak of tr(F) – with contradicting claims from different works (Golatkar et al., 2019; Achille et al., 2019; Jastrzebski et al., 2021) – correlates or anti-correlates with generalization, which indicates the underlining relationship may be more complex than just the peak value of the tr(F) curve.

![](images/12fefd66337ab716f57f044d1e477e183ee094c8ef08df165e13a0b4ee444d5d.jpg)  
Figure 2: The relative performance of adapters fine-tuned with scheduled unfreezing (i.e., GU-based and LPFT-based task adapters) and standard fine-tuned task adapters with full fine-tuning of mBERT and XLM-R.

![](images/99b23956a9fe2155efc69001971578bd1fa379a5eb4b99b686aae5373ab9735d.jpg)  
(a) mBERT SQuAD

![](images/2e0a2e2ac4ca1d683a3f50401b98ac786b547d7b33c3b502bab0d739010bcb5e.jpg)  
(b) mBERT COPA

![](images/27d0e89ba51342db915fe299e6f3ac0e011eb6bcca53b8735de748de6909c39e.jpg)  
(c) mBERT MNLI

![](images/41d9ba55389b3c20900cc08866be80766800f76080b9aa236cd28c6955593833.jpg)  
(d) XLM-R SQuAD

![](images/38600681d2962a2bd481728b28ae40e2864d29f0feec0937c564dab9c8c0553f.jpg)  
(e) XLM-R COPA

![](images/c75fb9a89730bc249133e5256a9aab26fac98603bb9b2945e0a45aeaaea10852.jpg)  
(f) XLM-R MNLI  
Figure 3: Average tr(F) per adapter during standard training versus using gradual unfreezing. Every point on the horizontal axis is 100 training steps for all datasets (except for XCOPA which is 50 steps).

Indeed, Figure 4 shows that a wider tr(F) curve (a longer learning period) often accompanies a large peak tr(F) value during the early phase of learning, and leads to better cross-lingual generalization during the test time. This could potentially lead to an additional new avenue of manipulating optimization with a regularization loss for crosslingual transfer, which we leave for future work.

To the best of our knowledge, this is the first evidence indicating that tr(F) correlates with crosslingual transfer performance in parameter-efficient fine-tuning and text-based Transformers. These results suggest that early-phase learning dynamics affect the generalization cross-lingually later on, and tr(F) can be a potential measurement to study cross-lingual generalization. From the above results, we conjecture that inducing large tr(F) and a longer learning period would improve generalization in the CF-free setting,<sup>11</sup> which motivates our experiments in the next section.

## 5.3 Auto-Scheduling by tr(F)

We have observed that scheduled unfreezing effectively changes the learning dynamics of the task adapters. A natural follow-up question is: Iffreezing certain parameters changes the learning dynamics, can we systematically and automatically select which task adapter to unfreeze next?

To answer this question, we propose to select the next layer to unfreeze based on ranked tr(F)

<table><tr><td colspan="4">MLQA (F1 / EM)</td><td colspan="3">XQuAD (F1 / EM)</td></tr><tr><td>Method</td><td>En</td><td>Lowest (Ar)</td><td>Average</td><td>En</td><td>Lowest (Th)</td><td>Average</td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a }$ </td><td>78.99/65.85</td><td>45.76/28.77</td><td> $5 5 . 4 0 { \pm } 0 . 9 4 / 3 7 . 0 7 { \pm } 0 . 7 2$ </td><td></td><td>83.58/71.74 34.53/38.89</td><td> $6 0 . 6 3 { \pm } 1 . 0 4 / 4 3 . 9 0 { \pm } 0 . 8 5$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { R a n d }$ </td><td></td><td>79.22/65.9446.65/29.84</td><td> $5 5 . 9 3 { \pm } 0 . 2 1 / 3 7 . 5 4 { \pm } 0 . 3 1$ </td><td>83.86/72.31</td><td>38.91/38.72</td><td> $6 1 . 6 8 { \pm } 0 . 3 3 / \underline { { 4 7 . 4 2 } } { \pm } 0 . 5 5$ </td></tr><tr><td> $\mathrm { \ m B E R T } ^ { A d a } \mathrm { _ { + G U } }$ </td><td>78.04/64.20</td><td>47.96/29.30</td><td>57.37±0.32 /38.27±0.27</td><td>83.21/71.55</td><td>44.08/35.46</td><td> $6 3 . 4 8 { \pm } 0 . 2 2 / 4 6 . 7 6 { \pm } 0 . 4 4$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { F U N }$ </td><td>78.82/65.29</td><td>48.20/30.90</td><td> $5 7 . 3 3 { \pm } 0 . 5 1 / 3 8 . 2 9 { \pm } 0 . 6 3$ </td><td></td><td>83.71/71.83 42.55/42.84</td><td> $6 3 . 2 5 { \pm } 0 . 2 6 / 4 9 . 0 9 { \pm } 0 . 4 8$ </td></tr><tr><td>Method</td><td>En</td><td>Lowest (Ar)</td><td>Average</td><td> $\mathtt { E n }$ </td><td>Lowest (Ar)</td><td>Average</td></tr><tr><td>XLM-R Ada</td><td>79.52/65.99</td><td>51.74/33.33</td><td> $6 1 . 3 1 { \pm } 0 . 4 6 / 4 2 . 1 0 { \pm } 0 . 4 2$ </td><td>83.48/72.69</td><td>65.47/48.89</td><td> $7 0 . 0 9 { \pm } 0 . 6 0 / 5 3 . 7 7 { \pm } 0 . 4 0$ </td></tr><tr><td> $\mathbf { \boldsymbol { X } } \mathbf { \boldsymbol { L } } \mathbf { \boldsymbol { M } } \mathbf { - } \mathbf { \boldsymbol { R } } ^ { A d a } + \mathbf { \boldsymbol { R } } \mathbf { \boldsymbol { a n d } }$ </td><td>80.32/67.01</td><td>50.33/36.29</td><td> $6 1 . 3 6 { \pm } 1 . 6 9 / 4 1 . 5 9 { \pm } 1 . 9 6$ </td><td>84.76/73.74</td><td>63.69/46.30</td><td> $6 9 . 9 9 \pm 1 . 4 7 / 5 2 . 0 6 \pm 2 . 0 4$ </td></tr><tr><td> $\bf { \cal X } \bf { L } \bf { M } \bf { - } \bf { R }  ^ { \bf { \cal A } d a } \bf { + } \bf { G } \bf { U }$ </td><td>80.37/66.77</td><td>55.16/35.49</td><td> $6 3 . 4 7 { \pm } 0 . 1 2 / 4 3 . 5 5 { \pm } 0 . 1 1$ </td><td>84.49/73.57</td><td>67.83/50.80</td><td> $\mathbf { 7 3 . 0 4 } \pm 0 . 2 2 / 5 5 . 9 3 \pm 0 . 1 5$ </td></tr><tr><td> $\mathrm { X L M - R } ^ { A d a } + \mathrm { F U N }$ </td><td></td><td>80.92/66.7053.17/37.73</td><td> $6 3 . 1 0 { \pm } 0 . 7 9 / \underline { { 4 3 . 3 7 } } { \pm } 0 . 5 1$ </td><td>84.91/73.80</td><td>66.69/49.24</td><td> $7 2 . 3 4 { \pm } 0 . 4 0 / \underline { { 5 5 . 2 1 } } { \pm } 0 . 6 3$ </td></tr><tr><td colspan="5">XCOPA (Accuracy)</td><td>XNLI (Accuracy)</td><td></td></tr><tr><td>Method</td><td>En</td><td>Lowest (It)</td><td>Average</td><td>En</td><td>Lowest (Sw)</td><td>Average</td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a }$ </td><td>63.80</td><td>50.16</td><td> $5 3 . 9 9 { \pm } 0 . 4 9 $ </td><td>82.05</td><td>37.45</td><td> $5 7 . 7 8 \pm 1 . 6 8$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { R a n d }$ </td><td>65.00</td><td>50.96</td><td> $5 3 . 8 4 \pm 0 . 7 1$ </td><td>81.64</td><td>45.95</td><td>59.87±0.96</td></tr><tr><td> $\mathrm { \ m B E R T } ^ { A d a } \mathrm { _ { + G U } }$ </td><td>66.60</td><td>50.00</td><td>54.29±0.60</td><td>81.79</td><td>54.06</td><td>61.67±1.04</td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { F U N }$ </td><td>66.40</td><td>50.68</td><td>53.98±0.64</td><td>81.70</td><td>53.73</td><td>61.36±0.51</td></tr><tr><td>Method</td><td>En</td><td>Lowest (Ht)</td><td>Average</td><td>En</td><td>Lowest (Sw)</td><td>Average</td></tr><tr><td> $\mathbf { X L M - R } ^ { A d a }$ </td><td>65.20</td><td>51.28</td><td>55.93±1.58</td><td>84.31</td><td>68.16</td><td>73.31±0.44</td></tr><tr><td> $\mathbf { \boldsymbol { X } } \mathbf { \boldsymbol { L } } \mathbf { \boldsymbol { M } } \mathbf { - } \mathbf { \boldsymbol { R } } ^ { A d a } + \mathbf { \boldsymbol { R } } \mathbf { \boldsymbol { a n d } }$ </td><td>67.20</td><td>52.12</td><td>57.05±0.42</td><td>84.52</td><td>67.29</td><td>72.68±0.56</td></tr><tr><td> $\bf { \cal X } \bf { L } \bf { M } \bf { - } \bf { R }  ^ { \bf { \cal A } d a } \bf { + } \bf { G } \bf { U }$ </td><td>66.00</td><td>52.52</td><td>58.24±1.11</td><td>84.24</td><td>68.24</td><td>73.44±0.24</td></tr><tr><td> $\mathrm { X L M - R } ^ { A d a } + \mathrm { F U N }$ </td><td>67.80</td><td>52.08</td><td> $5 8 . 1 1 \pm 0 . 9 4$ </td><td>84.72</td><td>67.48</td><td>73.13±0.53</td></tr></table>

Table 1: Zero-shot transfer results across four datasets: MLQA, XQuAD, XCOPA and XNLI. Average is the cross-lingual average without English. We bold the highest and underline the second-highest average value. Lowest denotes the task performance for the lowest-performing target language per each evaluation dataset and base model.

![](images/bcde3201a615167623a2500f29b2aaa3c85ddb5f5bb795737fbfbc51d31da60d.jpg)  
(a) SQuAD

![](images/a358e512718ab3ecfb890fd97e3a385368d52e31e4ac12ceaf816cb3312d3702.jpg)  
(b) MNLI  
Figure 4: Average tr(F) per adapter (normalized between 0-1 to plot together with the validation curve) and validation F1 or accuracy using a randomly sampled schedule. The average results indicated in the legend are the averaged cross-lingual transfer results. a) averaged F1 of MLQA and XQuAD, b) XNLI.

during training (i.e., Figure 1c). According to our hypothesis, an induced large and wide tr(F) during learning leads to better generalization (discussed in Section 5.2).<sup>12</sup>

tr(F)-based Scheduled Unfreezing (FUN) Recovers GU. Surprisingly, the $\operatorname { t r } ( F )$ -based schedules recover the transfer performance as well as the top-down heuristic schedule (i.e., GU) in many cases. To illustrate this, we plot the unfreezing schedules generated by FUN along with the topdown schedule of GU (diagonal line) for all our experiments in Figure 5. From the figure, we can see that FUN nearly perfectly recovers the top-down schedule of GU for mBERT. We conjecture that GU is implicitly maximizing $\operatorname { t r } ( F )$ at every unfreezing step. The XLM-R-based experiments largely follow the top-down order with more variance, which is likely due to noise in tr(F) estimation (for efficiency, we don’t use the entire training data for estimation).

We show the results across all datasets in Table 1 and include an additional baseline (+Rand) that randomly selects the next layer to unfreeze at every time interval k (see Appendix F for detailed results). Table 1 shows that the FUN scheduler achieves near-identical results as GU with the mBERT model, which matches the observations in Figure 5. We also achieve comparable results as GU with XLM-R. The results are well above the random unfreezing baselines and standard training (e.g., improving mBERT from standard training on XNLI for 3.58 points, or the average of 2.03 points across all 4 datasets etc.). Although the source English performance is not the focus of our work, we also find that FUN achieves better English results in most of the cases (e.g., XLM-R with FUN is better than GU in English for 6 out of 8 cases, which shows the potential of FUN beyond the context of cross-lingual transfer explored here).

![](images/03d9e03a3b94269fe184106b64263ef195fa666fe009966ec078379187454370.jpg)  
(a) SQuAD

![](images/d78676b5809bb8902f2b10c1e30b5502a96b0a3338fce948bde5d20e7ba76f28.jpg)  
(b) COPA

![](images/910657151f1a6897ff1d19c1f674ab943eb5abdaf97673e9e6054082a440e06b.jpg)  
(c) MNLI  
Figure 5: Averaged unfreezing schedules for GU and FUN with different base models.

In addition, we highlight that scheduled unfreezing improved transfer results for the lowestperforming language across the board. For example, FUN improved the lowest-performing language under standard adapter training in all 4 cases for the mBERT model, and in 3 out of 4 cases for XLM-R. Some gains are quite substantial, e.g., Thai with mBERT increased from 34.53 to 42.55 (see Table 1, the Lowest column for XQuAD).

Beyond MAD-X. Although less studied in the cross-lingual transfer setting, to show the generality of our findings, we perform additional experiments with LoRA adapters (Hu et al., 2022, see the original paper for details). From Table 2, we observe that GU and FUN have comparable performance on average (FUN is better than GU for MLQA and GU is better than FUN for XQuAD with mBERT, and comparable for XLM-R) for cross-lingual transfer. More importantly, both unfreezing methods consistently outperform the standard tuning of LoRA. Additional results for other tasks with LoRA can be found in Table 14 in Appendix G.4, which also demonstrates comparable results for GU and FUN. Our results indicate that scheduled unfreezing is a general method applicable to different adapter

types and improves generalization.
<table><tr><td colspan="4">XQuAD (F1 / EM)</td></tr><tr><td>Method</td><td>En</td><td>Lowest (Th)</td><td>Average</td></tr><tr><td>mBERT LoRA</td><td>83.68/71.50</td><td>37.56/29.44</td><td>61.53±0.37/45.01±0.44</td></tr><tr><td>mBERTLoRA +GU</td><td>84.24/72.75</td><td>40.73/31.41</td><td>63.12±0.20/45.96±0.38</td></tr><tr><td>mBERT LoRA +FUN</td><td>83.58/71.83</td><td>39.88/28.59</td><td>62.92±0.46/45.80±0.42</td></tr><tr><td>XLM-RLoRA</td><td>83.35/72.27</td><td>58.09/42.10</td><td>68.92±1.16/51.50±1.53</td></tr><tr><td> $\mathbf { X L M - R } ^ { L o R A } \mathbf { + G U }$ </td><td>84.70/73.12</td><td>66.72/49.50</td><td>72.27±0.12/ 54.67±0.34</td></tr><tr><td> $\mathbf { \mathrm { X L M - R } } ^ { L o R A _ { \mathrm { + F U N } } }$ </td><td>84.22/72.29</td><td>65.69/48.62</td><td>72.13±0.18/54.53±0.14</td></tr><tr><td colspan="4">MLQA (F1 / EM)</td></tr><tr><td>Method</td><td>En</td><td>Lowest (Ar)</td><td>Average</td></tr><tr><td>mBERTLoRA</td><td>79.32/65.99</td><td>46.17/29.67</td><td>55.55±0.60 /37.54±0.63</td></tr><tr><td>mBERTLoRA +GU</td><td>78.63/65.03</td><td>47.70/30.21</td><td>56.52±0.78/37.72±0.79</td></tr><tr><td>mBERT LoRA +FUN</td><td>78.80/65.40</td><td>47.12/29.98</td><td>56.65±0.79/ 38.02±0.97</td></tr><tr><td colspan="2">XLM-RLoRA</td><td>46.19/28.92</td><td></td></tr><tr><td> $\mathbf { X L M - R } ^ { L o R A } \mathbf { + G U }$ </td><td>80.04/67.27</td><td></td><td>59.40±0.61 /40.36±0.38</td></tr><tr><td></td><td>80.27/66.68</td><td>52.35/34.36</td><td>63.11±0.35/43.43±0.13</td></tr><tr><td> $\mathbf { \mathrm { X L M - R } } ^ { L o R A _ { \mathrm { + F U N } } }$ </td><td>80.51/67.18</td><td>52.39/33.85</td><td>62.62±0.50/43.21±0.45</td></tr></table>

Table 2: Zero-shot transfer results across MLQA and XQuAD, with the LoRA adapter. We bold the highest and underline the second-highest average value.

Future Perspectives of FUN. While GU remains effective, FUN offers the opportunity to potentially break away from the strict top-down unfreezing schedule and experiment with networks that have parallel structures (e.g., dual-network structures) in future work. Nonetheless, as the main finding in this work, FUN (i) provides evidence for a theorybased justification of heuristic unfreezing schedules from prior work, and (ii) it leads us to extend our understanding of learning dynamics during finetuning with scheduled unfreezing.

## 6 Conclusion

In this work, we first investigated whether scheduled unfreezing algorithms can help with generalization in the zero-shot cross-lingual transfer setting, and close the gap between parameter-efficient task-adapter training and full fine-tuning. Our experiments showed that scheduled unfreezing was indeed successful in closing this gap. Next, we investigated the training dynamics of scheduled unfreezing using the trace of the Fisher Information

Matrix (tr(F)). Our experiments revealed a link between scheduled unfreezing, tr(F) and generalization performance. Finally, we proposed a general scheduled unfreezing algorithm (tr(F)-based scheduled unfreezing, FUN) that achieves performance comparable to existing heuristic variants, across multiple models and datasets. We hope to look into utilizing scheduled unfreezing to improve the cross-lingual generalization capabilities of large language models in the future.

## 7 Limitations

In this paper, we work with the trace of the Fisher Information Matrix as the metric for studying learning dynamics. While we believe our experiments and conclusions are widely applicable, there may be other complex factors and theoretical metrics (such as the eigenvalue spectrum of F or other matrix norms, etc.) that we could potentially investigate. Furthermore, our use of tr(F) is connected to prior work on the “critical learning period” during the early stages of training neural networks (Achille et al., 2019; Kleinman et al., 2023), which could help us gain deeper theoretical insights into parameter-efficient training methods. We also speculate GU may not degrade the performance of full parameter fine-tuning (Raffel et al., 2022) if it is done early in the training (i.e., kL  N) rather than evenly unfrozen throughout the entire training process (i.e. $k = N / L )$ . However, such investigations are outside of the scope of this work, and we leave it for future work.

## Acknowledgments

This work was funded by the German Federal Ministry of Education and Research (BMBF) under the promotional reference 13N15897 (MISRIK), and the LOEWE initiative (Hesse, Germany) within the emergenCITY center. The work was also supported in part by a personal Royal Society University Research Fellowship (no 221137; 2022-) awarded to Ivan Vulic.´

We thank Sebastian Ruder, Ji-Ung Lee, Nico Daheim, and Tim Baumgärtner for their valuable feedback and suggestions on a draft of this paper. We thank Kexin Wang and Alan Ansell for discussions about training adapters.

## References

Alessandro Achille, Matteo Rovere, and Stefano Soatto. 2019. Critical learning periods in deep networks. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019.

Shun-ichi Amari. 1998. Natural gradient works efficiently in learning. Neural Computation, 10(2):251– 276.

Alan Ansell, Edoardo Maria Ponti, Jonas Pfeiffer, Sebastian Ruder, Goran Glavaš, Ivan Vulic, and Anna´ Korhonen. 2021. MAD-G: Multilingual adapter generation for efficient cross-lingual transfer. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 4762–4781, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2020. On the cross-lingual transferability of monolingual representations. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4623–4637, Online. Association for Computational Linguistics.

Ankur Bapna and Orhan Firat. 2019. Simple, scalable adaptation for neural machine translation. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1538– 1548, Hong Kong, China. Association for Computational Linguistics.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics.

Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. XNLI: Evaluating crosslingual sentence representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2475–2485, Brussels, Belgium. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Rory A. Fisher. 1925. Theory of statistical estimation. Mathematical Proceedings of the Cambridge Philosophical Society, 22:700 – 725.

Aditya Golatkar, Alessandro Achille, and Stefano Soatto. 2019. Time matters in regularizing deep networks: Weight decay and data augmentation affect early learning dynamics, matter little near convergence. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 10677–10687. Curran Associates, Inc.

Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2021. DeBERTa: decoding-enhanced bert with disentangled attention. In 9th International Conference on Learning Representations, ICLR 2021, Online, Austria, May 3-7, 2021.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for NLP. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2790–2799. PMLR.

Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification. In Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 328–339, Melbourne, Australia. Association for Computational Linguistics.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. 2020. XTREME: A massively multilingual multitask benchmark for evaluating cross-lingual generalisation. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, pages 4411–4421.

Stanislaw Jastrzebski, Devansh Arpit, Oliver Astrand, Giancarlo B Kerg, Huan Wang, Caiming Xiong, Richard Socher, Kyunghyun Cho, and Krzysztof J Geras. 2021. Catastrophic fisher explosion: Early phase fisher matrix impacts generalization. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 4772–4784. PMLR.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell.

2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521–3526.

Michael Kleinman, Alessandro Achille, and Stefano Soatto. 2023. Critical learning periods for multisensory integration in deep networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 24296–24305. IEEE.

Ananya Kumar, Aditi Raghunathan, Robbie Matthew Jones, Tengyu Ma, and Percy Liang. 2022. Finetuning can distort pretrained features and underperform out-of-distribution. In 10th International Conference on Learning Representations, ICLR 2019, Online, Apr 25-29, 2022.

Frederik Kunstner, Philipp Hennig, and Lukas Balles. 2019. Limitations of the empirical fisher approximation for natural gradient descent. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 4158–4169. Curran Associates, Inc.

Yoonho Lee, Annie S. Chen, Fahim Tajwar, Ananya Kumar, Huaxiu Yao, Percy Liang, and Chelsea Finn. 2023. Surgical fine-tuning improves adaptation to distribution shifts. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Patrick Lewis, Barlas Oguz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk. 2020. MLQA: Evaluating cross-lingual extractive question answering. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 7315– 7330, Online. Association for Computational Linguistics.

Abhilasha Lodha, Gayatri Belapurkar, Saloni Chalkapurkar, Yuanming Tao, Reshmi Ghosh, Samyadeep Basu, Dmitrii Petrov, and Soundararajan Srinivasan. 2023. On surgical fine-tuning for language encoders. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 3105–3113, Singapore. Association for Computational Linguistics.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019.

James Martens. 2020. New insights and perspectives on the natural gradient method. Journal ofMachine Learning Research, 21(146):1–76.

Michael McCloskey and Neal J. Cohen. 1989. Catastrophic interference in connectionist networks: The sequential learning problem. volume 24 of Psychology of Learning and Motivation, pages 109–165. Academic Press.

Marinela Parovic, Goran Glavaš, Ivan Vuli´ c, and Anna´ Korhonen. 2022. BAD-X: Bilingual adapters improve zero-shot cross-lingual transfer. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1791–1799, Seattle, United States. Association for Computational Linguistics.

Jonas Pfeiffer, Ivan Vulic, Iryna Gurevych, and Se-´ bastian Ruder. 2020. MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7654–7673, Online. Association for Computational Linguistics.

Jonas Pfeiffer, Ivan Vulic, Iryna Gurevych, and Sebas-´ tian Ruder. 2021. UNKs everywhere: Adapting multilingual language models to new scripts. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 10186–10203, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Edoardo Maria Ponti, Goran Glavaš, Olga Majewska, Qianchu Liu, Ivan Vulic, and Anna Korhonen. 2020.´ XCOPA: A multilingual dataset for causal commonsense reasoning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2362–2376, Online. Association for Computational Linguistics.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2022. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofMachine Learning Research, 21(1).

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas. Association for Computational Linguistics.

Alan Ramponi and Barbara Plank. 2020. Neural unsupervised domain adaptation in NLP—A survey. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6838–6855, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Melissa Roemmele, Cosmin A. Bejan, and Andrew S. Gordon. 2011. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In AAAI Spring Symposium on Logical Formalizations ofCommonsense Reasoning, Stanford University.

Amari Shun-ichi and Nagaoka Hiroshi. 2000. Methods of information geometry. volume 191 of Translations of Mathematical Monographs.

Sidak Pal Singh and Dan Alistarh. 2020. Woodfisher: Efficient second-order approximation for neural network compression. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, Online. Curran Associates, Inc.

Asa Cooper Stickland and Iain Murray. 2019. BERT and PALs: Projected attention layers for efficient adaptation in multi-task learning. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pages 5986–5995. PMLR.

Yi-Lin Sung, Varun Nair, and Colin Raffel. 2021. Training neural networks with fixed sparse masks. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6- 14, 2021, Online, pages 24193–24205. Curran Associates, Inc.

Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith, and Yejin Choi. 2020. Dataset cartography: Mapping and diagnosing datasets with training dynamics. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9275–9293, Online. Association for Computational Linguistics.

Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings ofthe 2018 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1112–1122, New Orleans, Louisiana. Association for Computational Linguistics.

Runxin Xu, Fuli Luo, Zhiyuan Zhang, Chuanqi Tan, Baobao Chang, Songfang Huang, and Fei Huang. 2021. Raise a child in large language model: Towards effective and generalizable fine-tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 9514– 9528, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Zhuoyi Yang, Ming Ding, Yanhui Guo, Qingsong Lv, and Jie Tang. 2022. Parameter-efficient tuning makes a good classification head. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, Abu Dahbi, United Arab Emirates. Association for Computational Linguistics.

![](images/1a9a3c5c7a92e07dbd8b5caa991fe99cc6db5cada3f833ea1a72fd24ac218009.jpg)  
Figure 6: Adapter Architecture.

## A Adapter Architecture

Figure 6 shows the adapter architecture for crosslingual transfer based on MAD-X. Each adapter consists of a down-projection followed by a ReLU activation and an up-projection, inserted after the feed-forward layer in every Transformer layer.

## B Hyperparameters

We include the detailed hyperparameters used in our experiments in Table 3 and 4. We use the AdamW optimizer (Loshchilov and Hutter, 2019) for all experiments, without warmups and a maximum gradient norm of 1 in all models.

For COPA we train the model with longer epochs for scheduled unfreezing. since the training data for COPA only contains 400 instances, the training time is still very small (<1 hour). We found standard training for task adapters for COPA does not benefit from longer training time and a smaller learning rate.

In addition, some of the language adapters are missing from the AdapterHub (just for mBERT or both for mBERT and XLM-R). We follow the same adapter configuration from (Pfeiffer et al., 2020), and train those language adapters that are missing for mBERT (Italian, Tamil and Thai) with the Wikipedia data, a learning rate of 1e-4 , batch size of 64, sequence length of 512, and a maximum 100 epochs/training budget or 24 hours (whichever is reached first).

The task adapters have a reduction factor of 16 as indicated in MAD-X.

<table><tr><td>Model</td><td>Epochs</td><td>lr</td><td colspan="3">batchsize k-LPFT k-GU k-FUN</td></tr><tr><td colspan="6">SQuAD - LR Schedule: linear</td></tr><tr><td>mBERT</td><td>5</td><td>5e-4</td><td>32</td><td>800 800</td><td>800</td></tr><tr><td>XLM-R</td><td>15</td><td>2e-4/5e-4</td><td>32</td><td>800</td><td>800 100</td></tr><tr><td colspan="6">COPA- LR Schedule: constant</td></tr><tr><td>mBERT XLM-R</td><td>500/50001e-4/1e-5</td><td>|500/50001e-4/1e-5</td><td>64 64</td><td>800 800</td><td>50 50 1000 1000</td></tr><tr><td colspan="6">MNLI- LR Schedule: linear</td></tr><tr><td>mBERT|</td><td>15</td><td>5e-4</td><td>128</td><td>25</td><td>800 50</td></tr><tr><td>XLM-R</td><td>15</td><td>5e-4</td><td>128</td><td>800</td><td>800 800</td></tr></table>

Table 3: Hyperparameters used in the main experiments. x/y denotes: x for standard adapter training and y for all scheduled unfreezing experiments.

<table><tr><td>Model Epochs lr</td><td colspan="4">batchsize k-1k k-5k k-10k</td></tr><tr><td colspan="6">SQuAD</td></tr><tr><td>mBERT| XLM-R</td><td>20 5e-4 20 5e-4</td><td>32 32</td><td>10 10</td><td>50 50</td><td>50 50</td></tr><tr><td colspan="6">MNLI</td></tr><tr><td>mBERT</td><td>50</td><td>5e-4 128</td><td>1</td><td>25</td><td>25</td></tr><tr><td>XLM-R</td><td>50</td><td>5e-4 128</td><td>1</td><td>25</td><td>10</td></tr></table>

Table 4: Hyperparameters used in the experiments with reduced task training data.

## C Dataset Statistics

We include the dataset statistics in Table 5. The training data for XQuAD and MLQA are SQuAD (Rajpurkar et al., 2016). The training data for XCOPA is COPA (Roemmele et al., 2011) and the training data for XNLI is MNLI (Williams et al., 2018). All datasets used are available on Hugging-Face. The language names and codes in our experiments are in Table 6.

<table><tr><td>Train data / Test data|1</td><td>|n. train (En) n. val (En)</td><td></td><td>n. test</td><td>n. lang</td></tr><tr><td>SQuAD / XQuAD</td><td>87599</td><td>10570</td><td>1190</td><td>11</td></tr><tr><td>SQuAD / MLQA</td><td>87599</td><td>10570</td><td>4517-5495</td><td>5</td></tr><tr><td>COPA / XCOPA</td><td>400</td><td>100</td><td>500</td><td>11</td></tr><tr><td>MNLI/XNLI</td><td>392702</td><td>2490</td><td>5010</td><td>14</td></tr></table>

Table 5: Dataset statistics.

## D Pseudo Code for tr(F) calculation

We provide the pseudo code for tr(F) calculation in Algorithm 2. Alternatively, please see our code at URL. Let FORWARD( ) be the standard forward pass operation, SAMPLE( ) be a function sampling labels from the label distribution of data, and AGGAVG( ) the function that aggregates tr(F) by task adapter blocks then taking the average over the number of trainable layers.

<table><tr><td>Language</td><td></td><td>Code|Language Code|Language</td><td></td><td>Code</td></tr><tr><td>Arabic</td><td>Ar</td><td>|German De</td><td>Greek</td><td>El</td></tr><tr><td>Spanish</td><td>Es</td><td>Hindi Hi</td><td>Russian</td><td>Ru</td></tr><tr><td>Thai</td><td>Th</td><td>Turkish Tr</td><td>Vietnamese</td><td>Vi</td></tr><tr><td>Estonian</td><td>Et</td><td>Haitian Ht</td><td>Italian</td><td>It</td></tr><tr><td>Indonesian</td><td>Id</td><td>Quechua Qu</td><td>Swahili</td><td>Sw</td></tr><tr><td>Chinese</td><td>Zh</td><td>Tamil</td><td>Ta</td><td></td></tr></table>

Table 6: Language code.

Algorithm 2 tr(F) Calculation   
Require: Number of batches N to sample from training data   
D for computing tr(F), trainable parameters.   
1: Copy the model. ▷ Avoid interference to the standard   
optimization.   
2: for i = 1 . . . N do   
3: Sample a data batch b D   
4: outputs = FORWARD( )   
5: labels = SAMPLE( )   
6: prob = LogSoftmax(outputs)   
7: loss = NLL(prob; labels)   
8: loss.backward()   
9: for $p _ { j } = 1 \ldots | \mathcal { P } |$ do   
10: $\mathrm { t r } ( F ) _ { j } = \dot { p } _ { j } . \dot { g } r a d ^ { 2 } / \left| b \right|$   
11: end for   
12: tr(F) = AGGAVG( )   
13: end for

## E Additional Discussion on Combining Scheduled Unfreezing with a Regularizer

Prior work (Jastrzebski et al., 2021) proposed a FIM regularizer that relies on an artificially low learning rate (i.e., 10% of the optimal learning rate for standard training). Adapter training requires a higher learning rate in general (e.g., learning rate values in Table 3), hence it is not straightforward to use an FIM regularizer directly. In addition, the generalization experiments in Jastrzebski et al. (2021) are considered in-distribution (ID), whereas we focus on distribution shifts (i.e., cross-lingual and OOD).

Recent work such as LPFT (Kumar et al., 2022) shows that ID and OOD performances do not correlate well, which we also observe in cross-lingual evaluation, especially when only English data is available for training and validation in zero-shot cross-lingual transfer (trends in English monolingual results can be inconsistent with trends in cross-

<table><tr><td>Model</td><td>MLQA (F1)</td><td>XQuAD (F1)</td></tr><tr><td>mBERT XLM-R</td><td>56.85 62.59</td><td>63.33 71.98</td></tr><tr><td>Model</td><td>|XCOPA (Acc.)</td><td>XNLI (Acc.)</td></tr><tr><td>mBERT</td><td>53.39</td><td>63.60</td></tr><tr><td>XLM-R</td><td>54.99</td><td>73.43</td></tr></table>

Table 7: Baselines (full fine-tuning) results for crosslingual transfer.

lingual results).

Hence, although it is possible to combine scheduled unfreezing with a regularization method, it is not obvious how to do so in the best way. A more extensive investigation is outside of the scope of this paper and we reserve this topic for future work.

## F Detailed Results for Experiments

We show the detailed per-language experimental results in Tables 8 to 10.

The baselines (full parameter fine-tuning) results for plotting Figure 2 are in Table 7.

## G Additional Experiments

## G.1 Simulated Low-data Scenario

In order to simulate setups with fewer annotated data for training and analyze the impact of scheduled unfreezing in those setups, we sample 1k, 5k, and 10k training examples from SQuAD and MNLI. We evaluate GU against the standard adapter training baseline (Table 11). With a smaller amount of training data, we still observe the advantages of GU over standard task adapter fine-tuning.

## G.2 Reverse Gradual Unfreezing

We briefly experimented (2 runs) with the reverse order (bottom-up) of gradual unfreezing on MLQA, XQuAD and XNLI. We include the results for bottom-up GU (rev) in Figure 12 and included the numbers from standard GU for reference. The cross-lingual transfer results are significantly lower than the standard GU; however, the English results are similar.

## G.3 Experiments on Smaller Task Training Data with tr(F)-based Scheduling

We additionally performed the same experiments with smaller training data as described in our main paper, but now with tr(F)-based scheduling (+FUN). The results are in Table 13. Our results show that FUN is also comparable to GU when there are fewer training instances available. The k used in our experiment for FUN are (for 1k/5k/10k ${ \mathrm { c o r r e s p o n d i n g l y } } ) \colon { \mathrm { m B E R T - X Q u A D } } = [ 1 0 , 5 0 , 5 0 ] ,$ $\mathrm { X L M \mathrm { - } R \mathrm { - } X Q u A D } = [ 1 0 , 5 0 , 2 5 ] , \mathrm { \ m B E R T \mathrm { - } X N L I } =$ [10,50,25], $\mathrm { X L M - R - X N L I } = [ 1 , 2 5 , 1 0 ]$ . The remaining hyperparameters are the same as in all other experiments.

<table><tr><td>MLQA</td><td>| En (F1 / EM) |</td><td>Ar</td><td>De</td><td>El</td><td>Es</td><td>Hi</td><td>Ru</td><td>Th</td><td>Tr</td><td>Vi</td><td>Zh</td><td>Avg. F1 / EM</td></tr><tr><td>mBERTAda</td><td>78.99/65.85</td><td>45.7659.47</td><td></td><td></td><td>64.60</td><td>46.91</td><td></td><td></td><td></td><td>57.01</td><td></td><td>58.6455.40±0.94/37.07±0.72</td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { R a n d }$ </td><td>79.22/65.94</td><td>46.65 57.92</td><td></td><td></td><td>67.91 46.03</td><td></td><td></td><td></td><td></td><td>57.75</td><td>59.34</td><td> $5 5 . 9 3 { \pm } 0 . 2 1 / 3 7 . 5 4 { \pm } 0 . 3 1$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } \mathrel { + } \mathrm { G U }$ </td><td>78.04/64.20</td><td>47.9657.95</td><td></td><td></td><td>68.6451.75</td><td></td><td></td><td></td><td></td><td>58.95</td><td>58.95</td><td> $5 7 . 3 7 { \pm } 0 . 3 2 / 3 8 . 2 7 { \pm } 0 . 2 7$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { F U N }$ </td><td>78.82/65.29</td><td>48.2058.96</td><td></td><td></td><td>67.19</td><td>51.51</td><td></td><td></td><td></td><td>59.47</td><td>58.64</td><td> $5 7 . 3 3 { \pm } 0 . 5 1 / 3 8 . 2 9 { \pm } 0 . 6 3$ </td></tr><tr><td> $\mathbf { X L M - R } ^ { A d a }$ </td><td>79.52/65.99</td><td>51.74 59.64</td><td></td><td></td><td>68.33</td><td>61.77</td><td></td><td></td><td></td><td>64.85</td><td>61.88</td><td>61.31±0.46/42.10±0.42</td></tr><tr><td> $\mathbf { \boldsymbol { X } } \mathbf { \boldsymbol { L } } \mathbf { \boldsymbol { M } } \mathbf { - } \mathbf { \boldsymbol { R } } ^ { A d a } + \mathbf { \boldsymbol { R } } \mathbf { \boldsymbol { a n d } }$ </td><td>80.32/67.01</td><td>50.33</td><td>61.72</td><td></td><td>69.98</td><td>60.08</td><td></td><td></td><td></td><td>63.81</td><td>62.22</td><td>61.36±1.69/41.59±1.96</td></tr><tr><td> $\bf { \cal X } \mathrm { L } { \bf M } { \bf - R } ^ { \cal A d a } { \bf + G } { \bf U }$ </td><td>80.37/66.77</td><td>55.16</td><td>61.30</td><td></td><td>70.36</td><td>63.75</td><td></td><td></td><td></td><td>66.45</td><td>63.79</td><td> $6 3 . 4 7 { \pm } 0 . 1 2 / 4 3 . 5 5 { \pm } 0 . 1 1$ </td></tr><tr><td> $\mathrm { X L M - R } ^ { A d a } \mathrm { + F U N }$ </td><td>80.92/66.70</td><td>53.17</td><td>62.38</td><td></td><td>70.04</td><td>63.77</td><td></td><td></td><td></td><td>65.38</td><td>63.86</td><td> $6 3 . 1 0 { \pm } 0 . 7 9 / 4 3 . 3 7 { \pm } 0 . 5 1$ </td></tr><tr><td>XQuAD</td><td>| En (F1 / EM)</td><td>Ar</td><td>De</td><td>El</td><td>Es</td><td>Hi</td><td>Ru</td><td>Th</td><td>Tr</td><td>Vi</td><td>Zh</td><td>Avg. F1 / EM</td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a }$ </td><td>83.58/71.74</td><td>57.95</td><td>71.20</td><td>57.60</td><td>73.11</td><td>53.75</td><td>70.05</td><td>34.53</td><td>50.15</td><td>68.38</td><td>69.56</td><td> $6 0 . 6 3 { \pm } 1 . 0 4 / 4 3 . 9 0 { \pm } 0 . 8 5$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { R a n d }$ </td><td>83.86/72.31</td><td>59.09</td><td>71.58</td><td>62.06</td><td>74.84</td><td>52.61</td><td>70.00</td><td>38.91</td><td>48.49</td><td>69.29</td><td>69.92</td><td> $6 1 . 6 8 { \pm } 0 . 3 3 / 4 7 . 4 2 { \pm } 0 . 5 5$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } \mathrel { + } \mathrm { G U }$ </td><td>83.21/71.55</td><td>62.90</td><td>72.39</td><td>62.38</td><td>74.43</td><td>56.28</td><td>69.46</td><td>44.08</td><td>53.39</td><td>70.10</td><td>69.37</td><td> $6 3 . 4 8 { \pm } 0 . 2 2 / 4 6 . 7 6 { \pm } 0 . 4 4$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { F U N }$ </td><td>83.71/71.83</td><td>62.24</td><td>72.74</td><td>62.19</td><td>74.05</td><td>56.92</td><td>69.74</td><td>42.55</td><td>53.40</td><td>69.56</td><td>69.13</td><td> $6 3 . 2 5 { \pm } 0 . 2 6 / 4 9 . 0 9 { \pm } 0 . 4 8$ </td></tr><tr><td> $\mathbf { \boldsymbol { X } } \mathbf { \boldsymbol { L } } \mathbf { \boldsymbol { M } } \mathbf { \cdot } \mathbf { \boldsymbol { R } } ^ { A d a }$ </td><td>83.48/72.69</td><td>65.47</td><td>72.74</td><td>71.92</td><td>74.88</td><td>67.70</td><td>73.58</td><td>66.53</td><td>66.36</td><td>72.38</td><td>69.36</td><td> $7 0 . 0 9 { \pm } 0 . 6 0 / 5 3 . 7 7 { \pm } 0 . 4 0$ </td></tr><tr><td> $\mathbf { \boldsymbol { X } } \mathbf { \boldsymbol { L } } \mathbf { \boldsymbol { M } } \mathbf { - } \mathbf { \boldsymbol { R } } ^ { A d a } + \mathbf { \boldsymbol { R } } \mathbf { \boldsymbol { a n d } }$ </td><td>84.76/73.74</td><td>63.69</td><td>74.40</td><td>71.25</td><td>76.28</td><td>65.09</td><td>73.77</td><td>64.93</td><td>64.85</td><td>72.06</td><td>73.54</td><td> $6 9 . 9 9 \pm 1 . 4 7 / 5 2 . 0 6 \pm 2 . 0 4$ </td></tr><tr><td> $\mathrm { X L M - R } ^ { A d a } \mathrm { + G U }$ </td><td>84.49/73.57</td><td>67.83</td><td>75.55</td><td>74.26</td><td>77.42</td><td>70.46</td><td>75.52</td><td>69.52</td><td>68.53</td><td>75.88</td><td>75.39</td><td> $7 3 . 0 4 { \pm } 0 . 2 2 / 5 5 . 9 3 { \pm } 0 . 1 5$ </td></tr><tr><td> $\mathrm { X L M - R } ^ { A d a } + \mathrm { F U N }$ </td><td>84.91/73.80</td><td>66.69</td><td>75.94</td><td>74.07</td><td>76.58</td><td>69.59</td><td>75.48</td><td>67.59</td><td>68.19</td><td></td><td>74.52 74.77</td><td> $7 2 . 3 4 { \pm } 0 . 4 0 / 5 5 . 2 1 { \pm } 0 . 6 3$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 8: Zero-shot transfer results (F1) for MLQA and XQuAD. Average is the cross-lingual average without English.
<table><tr><td>XCOPA</td><td>En</td><td>Et</td><td>Ht</td><td>It</td><td>Id</td><td>Qu</td><td>Sw</td><td>Zh</td><td>Ta</td><td>Th</td><td>Tr</td><td>Vi</td><td>Avg. Acc.</td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a }$ </td><td>63.80</td><td>54.20</td><td>53.04</td><td>50.16</td><td>53.84</td><td>53.12</td><td>54.16</td><td>59.08</td><td>52.56</td><td>51.68</td><td></td><td>54.5257.56</td><td> $5 3 . 9 9 \pm 0 . 4 9$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { R a n d }$ </td><td>65.00</td><td>53.36</td><td>52.32</td><td>50.96</td><td>53.60</td><td>54.00</td><td>53.44</td><td>58.64</td><td>50.92</td><td>51.96</td><td>55.0457.96</td><td></td><td> $5 3 . 8 4 \pm 0 . 7 1$ </td></tr><tr><td> $\mathrm { \ m B E R T } ^ { A d a } \mathrm { _ { + G U } }$ </td><td>66.60</td><td>54.44</td><td>52.60</td><td>50.00</td><td>54.88</td><td>53.52</td><td>53.52</td><td>59.76</td><td>52.12</td><td>52.36</td><td>55.44 58.60</td><td></td><td> $5 4 . 2 9 { \pm } 0 . 6 0$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { F U N }$ </td><td>66.40</td><td>53.44</td><td>52.92</td><td>50.68</td><td>54.76</td><td>53.48</td><td>54.40</td><td>59.00</td><td>51.36</td><td>51.08</td><td>54.3258.36</td><td></td><td> $5 3 . 9 8 { \pm } 0 . 6 4$ </td></tr><tr><td> $\mathbf { \boldsymbol { X } } \mathbf { \boldsymbol { L } } \mathbf { \boldsymbol { M } } \mathbf { \cdot } \mathbf { \boldsymbol { R } } ^ { A d a }$ </td><td>65.20</td><td>56.1651.28</td><td></td><td>56.72</td><td>58.00</td><td>51.80</td><td>55.60</td><td>59.12</td><td>56.44</td><td>57.72</td><td>56.48 55.96</td><td></td><td></td></tr><tr><td> $\mathrm { X L M - R } ^ { A d a } + \mathrm { R a n d }$ </td><td>67.20</td><td>57.08</td><td>52.12</td><td>57.80</td><td>60.72</td><td>53.32</td><td>56.24</td><td>60.08</td><td>57.36</td><td>58.20</td><td>56.7657.88</td><td></td><td> $5 5 . 9 3 { \pm } 1 . 5 8 $   $5 7 . 0 5 { \pm } 0 . 4 2 $ </td></tr><tr><td> $\mathrm { X L M - R } ^ { A d a } \substack { + \mathrm { G U } }$ </td><td>66.00</td><td>58.5652.52</td><td></td><td>58.24</td><td>62.04</td><td></td><td>53.9656.88</td><td>61.36</td><td>59.00</td><td></td><td>60.08 58.52 59.52</td><td></td><td> $5 8 . 2 4 \pm 1 . 1 1 $ </td></tr><tr><td> $\mathrm { X L M - R } ^ { A d a } + \mathrm { F U N }$ </td><td>67.80</td><td>58.16 52.08</td><td></td><td>57.44</td><td>61.28</td><td></td><td>55.04 56.36</td><td>61.64</td><td>57.84</td><td></td><td>61.04 58.80 59.48</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $5 8 . 1 1 \pm 0 . 9 4$ </td></tr></table>

Table 9: Zero-shot transfer results (Accuracy) for XCOPA. Average is the cross-lingual average without English.

<table><tr><td>XNLI</td><td>En</td><td>Ar</td><td>De</td><td>El</td><td>Es</td><td>Hi</td><td>Ru</td><td>Sw</td><td>Th</td><td>Tr</td><td>Vi</td><td>Zh</td><td> $\mathbf { A v g . A c c . }$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a }$ </td><td>82.05</td><td>42.09</td><td>65.81</td><td>62.16</td><td>70.84</td><td>57.92</td><td>63.76</td><td>37.45</td><td>40.89</td><td>61.53</td><td>68.01</td><td>65.08</td><td> $5 7 . 7 8 { \pm } 1 . 6 8 $ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { R a n d }$ </td><td>81.64</td><td>53.98</td><td>66.32</td><td>62.85</td><td>70.33</td><td>58.15</td><td>65.08</td><td>45.95</td><td>41.80</td><td>61.25</td><td>67.73</td><td>65.12</td><td> $5 9 . 8 7 { \pm } 0 . 9 6 $ </td></tr><tr><td> $\mathrm { \ m B E R T } ^ { A d a } \mathrm { _ { + G U } }$ </td><td>81.79</td><td>62.78</td><td>66.25</td><td>63.51</td><td>70.28</td><td>58.89</td><td>65.74</td><td>54.06</td><td>38.97</td><td>62.25</td><td>68.52</td><td>67.17</td><td> $6 1 . 6 7 { \pm } 1 . 0 4 $ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { F U N }$ </td><td>81.70</td><td>58.48</td><td>66.32</td><td>63.87</td><td>70.98</td><td>59.08</td><td>65.45</td><td>53.73</td><td>41.13</td><td>61.89</td><td>68.25</td><td>65.75</td><td> $6 1 . 3 6 { \pm } 0 . 5 1$ </td></tr><tr><td> $\mathbf { \boldsymbol { X } } \mathbf { \boldsymbol { L } } \mathbf { \boldsymbol { M } } \mathbf { \cdot } \mathbf { \boldsymbol { R } } ^ { A d a }$ </td><td></td><td>70.42</td><td></td><td></td><td></td><td></td><td>75.14</td><td>68.16</td><td>71.14</td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { X L M - R } ^ { A d a } + \mathrm { R a n d }$ </td><td>84.31 84.52</td><td>69.91</td><td>76.16 75.91</td><td>75.80 75.05</td><td>78.85 78.04</td><td>70.16 69.56</td><td>74.51</td><td>67.29</td><td>70.40</td><td>72.33 72.05</td><td>74.2572.48</td><td>75.0673.24</td><td> $7 3 . 3 1 { \pm } 0 . 4 4$ </td></tr><tr><td> $\mathrm { X L M - R } ^ { A d a } \mathrm { + G U }$ </td><td>84.24</td><td>70.22</td><td>75.92</td><td>75.7</td><td>78.32</td><td>70.61</td><td>75.70</td><td>68.24</td><td>71.99</td><td></td><td>71.97 75.3673.82</td><td></td><td> $7 2 . 6 8 { \pm } 0 . 5 6 $   $7 3 . 4 4 \pm 0 . 2 4$ </td></tr><tr><td></td><td></td><td></td><td>70.5876.17</td><td>75.68</td><td>78.29</td><td>69.75</td><td>75.42</td><td></td><td>67.4871.4471.71 74.7573.20</td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { X L M - R } ^ { A d a } \mathrm { + F U N }$ </td><td>84.72</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $7 3 . 1 3 { \pm } 0 . 5 3 $ </td></tr></table>

Table 10: Zero-shot transfer results (Accuracy) for XNLI. Average is the cross-lingual average without English.

<table><tr><td>XQuAD (F1) 一</td><td>1K</td><td>5K</td><td>10K</td></tr><tr><td>mBERTAda</td><td> $4 5 . 2 7 { \pm } 0 . 5 9$ </td><td> $5 2 . 5 8 { \pm } 0 . 8 1 $ </td><td> $5 5 . 8 9 { \pm } 1 . 0 8 $ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { G U }$ </td><td> ${ \pm } 5 . 9 3 { \pm } 0 . 5 0 $ </td><td> ${ \pm 3 . 1 0 \pm 0 . 3 5 }$ </td><td> ${ \pm 6 . 4 7 \pm 0 . 7 4 }$ </td></tr><tr><td>XLM-RAda</td><td> $4 4 . 1 1 \pm 1 . 4 3 $ </td><td> $5 7 . 2 0 { \pm } 0 . 3 6 $ </td><td> $6 1 . 7 5 { \pm } 0 . 6 8$ </td></tr><tr><td> $\bf { X L M - R } ^ { \bf { . } \bf { A } \it { d } a } \mathrm { { _ { + G U } } }$ </td><td> $4 8 . 4 2 \pm 1 . 2 0 $ </td><td> ${ \pm } 9 . 8 8 { \pm } 1 . 5 1 $ </td><td> ${ \bf 6 5 . 2 8 { \pm } 0 . 7 6 }$ </td></tr><tr><td>XNLI (Accuracy)|</td><td>1K</td><td>5K</td><td>10K</td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a }$ </td><td> $4 3 . 8 6 \pm 1 . 4 3$ </td><td> $4 9 . 6 8 \pm 0 . 7 3$ </td><td> $5 2 . 3 4 { \pm } 0 . 4 0 $ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { G U }$ </td><td> ${ \pm 4 . 6 9 } \mathrm { \pm 0 . 6 1 }$ </td><td> ${ \bf 5 1 . 6 7 } \pm 0 . 4 3$ </td><td> ${ \pm } 3 . 9 5 { \pm } 1 . 4 7$ </td></tr><tr><td> $\mathbf { X L M - R } ^ { A d a }$ </td><td> $5 2 . 7 5 { \pm } 2 . 0 3 $ </td><td> ${ \bf 6 4 . 2 2 \pm 1 . 0 4 }$ </td><td> $6 5 . 8 0 { \pm } 0 . 6 1 $ </td></tr><tr><td> $\bf { X L M - R } ^ { \bf { . } \bf { A } \it { d } a } \mathrm { { + } \bf { G U } }$ </td><td> ${ \pm 1 . 3 6 \pm 1 . 3 8 }$ </td><td> $6 4 . 1 5 { \pm } 0 . 3 5 $ </td><td> ${ \bf 6 5 . 9 1 \pm 0 . 5 6 }$ </td></tr></table>

Table 11: Cross-lingual transfer performance with subsampled English task data for task fine-tuning.

Table 12: Zero-shot transfer results of gradual unfreezing in reverse order across three datasets: MLQA, XQuAD, and XNLI. Average is the cross-lingual average without English.
<table><tr><td>MLQA</td><td>En (F1)</td><td>Avg. F1</td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } \substack { + \mathrm { G U } }$ </td><td>78.04</td><td>57.37</td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } { + } \mathrm { G U ( r e v ) }$ </td><td>78.71</td><td>49.09</td></tr><tr><td> $\mathrm { X L M - R } ^ { A d a } \substack { + } \mathrm { G U }$   $\mathrm { X L M - R } ^ { A d a } { + } \mathrm { G U } \left( \mathrm { r e v } \right)$ </td><td>80.37 81.38</td><td>63.47 57.59</td></tr><tr><td>XQuAD</td><td>En (F1)</td><td>Avg. F1</td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } \substack { + \mathrm { G U } }$ </td><td>83.21</td><td>63.48</td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } { + } \mathrm { G U ( r e v ) }$ </td><td>82.17</td><td>53.43</td></tr><tr><td> $\mathrm { X L M - R } ^ { A d a } \substack { + } \mathrm { G U }$ </td><td>84.49</td><td>73.04</td></tr><tr><td> $\mathrm { X L M - R } ^ { A d a } { + } \mathrm { G U } \left( \mathrm { r e v } \right)$ </td><td>84.12</td><td>65.44</td></tr><tr><td>XNLI</td><td></td><td>| En (Acc.) Avg. Acc.</td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } \substack { + \mathrm { G U } }$ </td><td></td><td></td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } \substack { + \mathrm { G U } }$ </td><td>81.79 81.43</td><td>61.67 55.67</td></tr><tr><td></td><td></td><td></td></tr><tr><td> $\mathrm { X L M - R } ^ { A d a } \substack { + } \mathrm { G U }$   $\mathrm { X L M - R } ^ { A d a } { + } \mathrm { G U } \left( \mathrm { r e v } \right)$ </td><td>84.24 84.23</td><td>73.44 72.50</td></tr></table>

<table><tr><td>XQuAD ((F1)) 一</td><td>1K</td><td>5K</td><td>10K</td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a }$ </td><td> $4 5 . 2 7 { \pm } 0 . 5 9$ </td><td> $5 2 . 5 8 { \pm } 0 . 8 1 $ </td><td> $5 5 . 8 9 \pm 1 . 0 8$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { G U }$ </td><td> $4 5 . 9 3 { \pm } 0 . 5 0 $ </td><td> $5 3 . 1 0 { \pm } 0 . 3 5 $ </td><td> $5 6 . 4 7 { \scriptstyle \pm 0 . 7 4 }$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { F U N }$ </td><td> $4 6 . 3 0 { \pm } 0 . 7 1$ </td><td> $5 3 . 6 8 { \pm } 0 . 3 8 $ </td><td> $5 6 . 5 0 { \scriptstyle \pm 0 . 9 7 }$ </td></tr><tr><td> $\mathbf { X L M - R } ^ { A d a }$ </td><td> $4 4 . 1 1 \pm 1 . 4 3 $ </td><td> $5 7 . 2 0 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $6 1 . 7 5 { \scriptstyle \pm 0 . 6 8 }$ </td></tr><tr><td> $\mathbf { \mathbf { \mathbf { X } } \mathbf { \mathbf { L } } \mathbf { \mathbf { M } } \mathbf { - } \mathbf { R } ^ { A d a } \mathbf { \mathbf { + } } \mathbf { G } \mathbf { U } }$ </td><td> $4 8 . 4 2 \pm 1 . 2 0 $ </td><td> $5 9 . 8 8 \pm 1 . 5 1 $ </td><td> $6 5 . 2 8 { \pm } 0 . 7 6$ </td></tr><tr><td> $\mathbf { \boldsymbol { X } } \mathbf { \boldsymbol { L } } \mathbf { \boldsymbol { M } } \mathbf { \boldsymbol { - R } } ^ { A d a } \mathbf { \boldsymbol { + F } } \mathbf { \boldsymbol { U } } \mathbf { \boldsymbol { N } }$ </td><td> $4 7 . 6 7 { \pm } 1 . 7 3 $ </td><td> $6 0 . 7 2 { \scriptstyle \pm 1 . 0 7 }$ </td><td> $6 5 . 1 6 { \pm } 1 . 1 6$ </td></tr><tr><td> $\mathrm { X N L I } \left( \mathrm { A c c u r a c y } \right)$  一</td><td>1K</td><td>5K</td><td>10K</td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a }$ </td><td> $4 3 . 8 6 \pm 1 . 4 3$ </td><td> $4 9 . 6 8 \pm 0 . 7 3$ </td><td> $5 2 . 3 4 \pm 0 . 4 0$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { G U }$ </td><td> $4 4 . 6 9 { \pm } 0 . 6 1 $ </td><td> $5 1 . 6 7 { \pm } 0 . 4 3 $ </td><td> $5 3 . 9 5 { \pm } 1 . 4 7 $ </td></tr><tr><td> $\mathrm { m B E R T } ^ { A d a } + \mathrm { F U N }$ </td><td> $4 4 . 8 6 \pm 0 . 4 9$ </td><td> $5 1 . 6 1 { \pm } 0 . 5 8 $ </td><td> $5 3 . 4 0 { \pm } 0 . 9 3 $ </td></tr><tr><td> $\mathbf { X L M - R } ^ { A d a }$ </td><td> $5 2 . 7 5 { \pm } 2 . 0 3 $ </td><td> $6 4 . 2 2 \pm 1 . 0 4$ </td><td> $6 5 . 8 0 { \pm } 0 . 6 1 $ </td></tr><tr><td> $\bf { X L M - R } ^ { \bf { \delta A d a } } \mathrm { _ { + G U } }$ </td><td> $5 2 . 8 6 \pm 1 . 3 8 $ </td><td> $6 4 . 1 5 { \pm } 0 . 3 5 $ </td><td> $6 5 . 9 1 { \scriptstyle \pm 0 . 5 6 }$ </td></tr><tr><td> $\mathbf { \boldsymbol { X } } \mathbf { \boldsymbol { L } } \mathbf { \boldsymbol { M } } \mathbf { \boldsymbol { - R } } ^ { A d a } \mathbf { \boldsymbol { + F } } \mathbf { \boldsymbol { U } } \mathbf { \boldsymbol { N } }$ </td><td> $5 2 . 2 9 { \pm } 1 . 8 5 $ </td><td></td><td></td></tr><tr><td></td><td></td><td> $6 4 . 1 0 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $6 6 . 2 6 { \scriptstyle \pm 0 . 8 7 }$ </td></tr></table>

Table 13: Cross-lingual transfer performance with subsampled English task data for task fine-tuning.

## G.4 Additional Results with LoRA Adapters

We provide additional results with LoRA adapters in Table 14. For simplicity and to draw comparisons to our experiments with MAD-X, we couple the unfreezing for query and value LoRAs (applied to ‘q’ and ‘v’ attentions) and use the default LoRA configuration [lora\_r=8, lora\_alpha=8] from AdapterHub.

From the results, we can see that GU and FUN are consistently comparable. GU is slightly worse than the standard training likely due to the extremely small training data size of COPA. Overall, scheduled unfreezing algorithms can be easily applied to different adapter architectures to provide a performance boost.

The hyperparameters used in this experiment are in Table 15, where we kept the number of epochs for training and batch size the same as in our main experiments (Table 3).

## G.5 Preliminary Results with mDeBERTa

We include additional experiments on another, more recent base model, mDeBERTa (He et al., 2021). mDeBERTa is the multilingual version of the recently proposed DeBERTa (He et al., 2021) model with disentangled attention to its word content and position representations. We used the (mdeberta-v3-base) model for all our experiments.

Note that we trained language adapters using MLM loss for the mDeBERTa model according to the setup described in (Pfeiffer et al., 2020). However, we see very large discrepancies in terms of transfer results for both XCOPA and XNLI when compared to the standard fine-tuning (the gaps are also much larger than the gaps for mBERT or XLM-R). We hypothesize that the discrepancies may be because mDeBERTa uses different attentions and the adapters we studied here are not designed for mDeBERTa (both in their architecture and in their training method). However, as tuning adapter architectures is beyond the scope of our study, we include the results here for completeness only.

<table><tr><td></td><td colspan="3">MLQA (F1 / EM)</td><td colspan="3">XQuAD (F1 / EM)</td></tr><tr><td>Method</td><td>En</td><td>Lowest (Th)</td><td>Average</td><td>En</td><td>Lowest (Ar)</td><td>Average</td></tr><tr><td> $\mathrm { m B E R T } ^ { L o R A }$ </td><td></td><td>79.32/65.99 46.17/29.67</td><td> $5 5 . 5 5 { \pm } 0 . 6 0 / 3 7 . 5 4 { \pm } 0 . 6 3$ </td><td>83.68/71.50</td><td>37.56/29.44</td><td> $6 1 . 5 3 { \pm } 0 . 3 7 / 4 5 . 0 1 { \pm } 0 . 4 4$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { L o R A } \mathrm { + G U }$ </td><td></td><td>78.63/65.03 47.70/30.21</td><td> $5 6 . 5 2 { \pm } 0 . 7 8 / \underline { { 3 7 . 7 2 } } { \pm } 0 . 7 9$ </td><td>84.24/72.75</td><td>40.73/31.41</td><td> ${ 6 3 . 1 2 \pm 0 . 2 0 / 4 5 . 9 6 \pm 0 . 3 8 }$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { L o R A } \mathrm { + F U N }$ </td><td>78.80/65.40</td><td>47.12/29.98</td><td> ${ \bf 5 6 . 6 5 \pm 0 . 7 9 / 3 8 . 0 2 \pm 0 . 9 7 }$ </td><td>83.58/71.83</td><td>39.88/28.59</td><td> $6 2 . 9 2 { \pm } 0 . 4 6 / \underline { { 4 5 . 8 0 { \pm } 0 . 4 2 } }$ </td></tr><tr><td>Method</td><td>En</td><td>Lowest (Ar)</td><td>Average</td><td>En</td><td>Lowest (Ar)</td><td>Average</td></tr><tr><td> $\mathbf { \boldsymbol { X } } \mathbf { \boldsymbol { L } } \mathbf { \boldsymbol { M } } \mathbf { \mathbf { - } } \mathbf { \boldsymbol { R } } ^ { L o R A }$ </td><td>80.04/67.27</td><td>46.19/28.92</td><td> $5 9 . 4 0 { \pm } 0 . 6 1 / 4 0 . 3 6 { \pm } 0 . 3 8$ </td><td>83.35/72.27</td><td>58.09/42.10</td><td> $6 8 . 9 2 { \pm } 1 . 1 6 / 5 1 . 5 0 { \pm } 1 . 5 3$ </td></tr><tr><td> $\mathrm { X L M - R } ^ { L o R A } \mathrm { + G U }$ </td><td>80.27/66.68</td><td>52.35/34.36</td><td> ${ \bf 6 3 . 1 1 \pm 0 . 3 5 / 4 3 . 4 3 \pm 0 . 1 3 }$ </td><td>84.70/73.12</td><td>66.72/49.50</td><td> $7 2 . 2 7 { \pm } 0 . 1 2 / 5 4 . 6 7 { \pm } 0 . 3 4$ </td></tr><tr><td> $\mathbf { \mathrm { X L M - R } } ^ { L o R A _ { \mathrm { + F U N } } }$ </td><td>80.51/67.18</td><td>52.39/33.85</td><td> $6 2 . 6 2 { \pm } 0 . 5 0 / \underline { { 4 3 . 2 1 } } { \pm } 0 . 4 5$ </td><td>84.22/72.29</td><td>65.69/48.62</td><td> $7 2 . 1 3 { \pm } 0 . 1 8 / \underline { { 5 4 . 5 3 } } { \pm } 0 . 1 4$ </td></tr><tr><td colspan="5">XCOPA (Accuracy)</td><td>XNLI (Accuracy)</td><td></td></tr><tr><td>Method</td><td>En</td><td>Lowest</td><td>Average</td><td>En</td><td>Lowest (Sw)</td><td>Average</td></tr><tr><td> $\mathrm { m B E R T } ^ { L o R A }$ </td><td>68.00</td><td>50.76 (Et)</td><td> $5 3 . 4 3 { \pm } 1 . 0 5 $ </td><td>81.86</td><td>49.96</td><td> $6 5 . 2 7 { \pm } 0 . 0 2$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { L o R A } \mathrm { + G U }$ </td><td>66.00</td><td>52.04 (Ta)</td><td> $\underline { { 5 4 . 6 7 } } \pm 0 . 4 5$ </td><td>81.27</td><td>49.93</td><td> $6 5 . 3 3 \pm 0 . 1 5$ </td></tr><tr><td> $\mathrm { m B E R T } ^ { L o R A } \mathrm { + F U N }$ </td><td>66.80</td><td>51.84 (Sw)</td><td> ${ \pm } 0 . 8 3 { \pm } 0 . 2 6$ </td><td>81.05</td><td>50.23</td><td> $6 5 . 3 2 { \pm } 0 . 1 5$ </td></tr><tr><td>Method</td><td>En</td><td>Lowest</td><td>Average</td><td>En</td><td>Lowest (Sw)</td><td>Average</td></tr><tr><td> $\mathbf { \boldsymbol { X } } \mathbf { \boldsymbol { L } } \mathbf { \boldsymbol { M } } \mathbf { \mathbf { - } } \mathbf { \boldsymbol { R } } ^ { L o R A }$ </td><td>63.60</td><td>50.35 (Ht)</td><td> $5 5 . 4 6 { \pm } 0 . 8 8 $ </td><td>84.26</td><td>64.79</td><td> $7 2 . 9 8 \pm 0 . 3 3$ </td></tr><tr><td> $\mathrm { X L M - R } ^ { L o R A } \mathrm { + G U }$ </td><td>64.00</td><td>51.10 (Ht)</td><td> $5 5 . 0 7 \pm 1 . 3 7$ </td><td>84.04</td><td>65.29</td><td> ${ \underline { { 7 3 . 4 3 } } } \pm 0 . 2 0$ </td></tr><tr><td> $\mathbf { \mathrm { X L M - R } } ^ { L o R A _ { \mathrm { + F U N } } }$ </td><td>64.40</td><td>49.04 (Qu)</td><td> ${ \pm } 5 5 . 6 2 { \pm } 0 . 9 9$ </td><td>84.10</td><td>65.88</td><td> ${ \bf 7 3 . 6 5 \pm 0 . 5 7 }$ </td></tr></table>

Table 14: Zero-shot transfer results with LoRA adapters. Average is the cross-lingual average without English. We bold the highest and underline the second-highest average value. Lowest denotes the task performance for the lowest-performing target language per each evaluation dataset and base model.

<table><tr><td>Model</td><td>Epochs lr</td><td>batchsize k-GU k-FUN</td><td></td></tr><tr><td colspan="4">SQuAD - LR Schedule: linear</td></tr><tr><td>mBERT|</td><td>5  $5 \mathrm { e } { - } 4 / 8 \mathrm { e } { - } 4$ </td><td>32 800</td><td>800</td></tr><tr><td>XLM-R</td><td>15 5e-4</td><td>32</td><td>800 800</td></tr><tr><td colspan="4">COPA - LR Schedule: constant</td></tr><tr><td>mBERT</td><td>500/5000 1e-4</td><td>64</td><td>50 100</td></tr><tr><td>XLM-R</td><td>500/5000 1e-4</td><td>64 100</td><td>200</td></tr><tr><td colspan="4">MNLI - LR Schedule: linear</td></tr><tr><td>mBERT</td><td>15  $5 \mathrm { e } { \mathrm { - } } 4$ </td><td>128 50</td><td>50</td></tr><tr><td>XLM-R</td><td>15 5e-4</td><td>128 800</td><td>800</td></tr></table>

Table 15: Hyperparameters used in the experiments with LoRA adapters. $x / y$ denotes: x for standard adapter training and y for all scheduled unfreezing experiments.

<table><tr><td>MLQA</td><td>| En (F1 / EM) |</td><td>Ar</td><td>De</td><td>El</td><td>Es</td><td>Hi</td><td>Ru</td><td>Th</td><td>Tr</td><td>Vi</td><td>Zh</td><td>Avg. F1 /EM</td></tr><tr><td> $\mathrm { \ m D e B E R T a } ^ { A d a }$ </td><td>82.31</td><td>58.50</td><td>67.56</td><td>1</td><td>73.00</td><td>64.27</td><td></td><td>1</td><td></td><td>64.92</td><td>65.27</td><td>65.59±0.22/46.19±0.29</td></tr><tr><td> $\mathrm { \ m D e B E R T a } ^ { A d a } + \mathrm { G U }$ </td><td>82.27</td><td>61.19</td><td>67.01</td><td>=</td><td>73.26</td><td>65.71</td><td>=</td><td>=</td><td>=</td><td>65.71</td><td>67.44</td><td>67.08±0.38 /47.22±0.32</td></tr><tr><td>XQuAD</td><td>| En (F1 / EM) |</td><td>Ar</td><td>De</td><td>El</td><td>Es</td><td>Hi</td><td>Ru</td><td>Th</td><td>Tr</td><td>Vi</td><td>Zh</td><td>Avg. F1 / EM</td></tr><tr><td> $\mathrm { \ m D e B E R T a } ^ { A d a }$ </td><td>86.30</td><td>72.33</td><td>80.38</td><td>76.89</td><td>79.93</td><td>72.77</td><td>77.27</td><td>69.97</td><td>71.51</td><td>74.99</td><td></td><td>78.98 75.50±0.29 /58.66±0.19</td></tr><tr><td> $\mathrm { \ m D e B E R T a } ^ { A d a } + \mathrm { G U }$ </td><td>85.52</td><td>73.31 79.59</td><td></td><td>78.41</td><td>79.77</td><td>73.58</td><td>77.87</td><td>70.98</td><td>72.66</td><td>75.42 79.15</td><td></td><td> $7 6 . 0 7 { \pm } 0 . 1 3 / 5 8 . 9 0 { \pm } 0 . 1 2$ </td></tr></table>

Table 16: mDeBERTa: Zero-shot transfer results (F1) XQUAD and MLQA. Average is the cross-lingual average without English.

<table><tr><td>XCOPA</td><td>En</td><td>Et</td><td>Ht</td><td>It</td><td>Id</td><td>Qu</td><td> $\operatorname { S w }$ </td><td>Zh</td><td>Ta</td><td>Th</td><td>Tr</td><td>Vi</td><td>Avg. Acc.</td></tr><tr><td> $\scriptstyle { \mathrm { m D e B E R T a } } ^ { A d a }$ </td><td>65.75</td><td></td><td></td><td>55.32 55.68 58.64 61.00</td><td></td><td></td><td>51.40 56.48</td><td>62.7657.1657.2855.7659.04</td><td></td><td></td><td></td><td></td><td> $5 7 . 3 2 { \pm } 4 . 4 6 $ </td></tr><tr><td> $\mathrm { \ m D e B E R T a } ^ { A d a } + \mathrm { G U }$ </td><td>65.80</td><td></td><td></td><td>57.96 56.60 59.36 60.12 52.36 55.96 63.28 57.88 57.48 58.12 58.76</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $5 7 . 9 9 \pm 3 . 5 7$ </td></tr></table>

Table 17: mDeBERTa: Zero-shot transfer results (Accuracy) XCOPA. Average is the cross-lingual average without English.

<table><tr><td>XNLI</td><td>En</td><td>Ar</td><td>De</td><td>El</td><td>Es</td><td>Hi</td><td>Ru</td><td>Sw</td><td>Th</td><td>Tr</td><td>Vi</td><td>Zh</td><td>Avg. Acc.</td></tr><tr><td> $\scriptstyle { \mathrm { m D e B E R T a } } ^ { A d a }$ </td><td>86.94</td><td>72.09</td><td>78.57</td><td>76.03</td><td>80.92</td><td>69.41</td><td>76.98</td><td>68.45</td><td>68.73</td><td>75.93</td><td>74.62 73.30</td><td></td><td> $7 4 . 0 9 { \pm } 0 . 7 7$ </td></tr><tr><td>mDeBERTaAda+GU</td><td>86.48</td><td></td><td>71.4677.90</td><td>75.50</td><td>79.23</td><td></td><td>67.3075.84</td><td>68.64</td><td></td><td>69.68 74.51 74.05 74.85</td><td></td><td></td><td> $7 3 . 5 4 \pm 0 . 5 8$ </td></tr></table>

Table 18: mDeBERTa: Zero-shot transfer results (Accuracy) XNLI. Average is the cross-lingual average without English.