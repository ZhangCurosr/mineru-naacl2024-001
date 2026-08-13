# R-Spin: Efficient Speaker and Noise-invariant Representation Learning with Acoustic Pieces

Heng-Jui Chang and James Glass MIT CSAIL hengjui@mit.edu

## Abstract

This paper introduces Robust Spin (R-Spin), a data-efficient domain-specific self-supervision method for speaker and noise-invariant speech representations by learning discrete acoustic units with speaker-invariant clustering (Spin). R-Spin resolves Spin's issues and enhances content representations by learning to predict acoustic pieces. R-Spin offers a 12X reduction in computational resources compared to previous state-of-the-art methods while outperforming them in severely distorted speech scenarios. This paper provides detailed analyses to show how discrete units contribute to speech encoder training and improving robustness in diverse acoustic environments.

## 1 Introduction

Self-supervised learning (SSL) for encoder pretraining has emerged as a foundational element in speech processing, outperforming conventional approaches across various applications (Mohamed et al., 2022; Liu et al., 2022). Given the substantial cost associated with human annotation of speech data, SSL methods leverage unlabeled audio data to pre-train encoders, generating good representations for downstream tasks like automatic speech recognition (ASR) and speaker identification (Yang et al., 2021; Tsai et al., 2022). The application of SSL models has notably concentrated on ASR, aiming to mitigate the dependence on large transcribed corpora (Hsu et al., 2021a; Baevski et al., 2022; Liu et al., 2023). Thus, extracting content representations has become a crucial aspect of speech SSL research (Tjandra et al., 2021; Chan and Ghosh, 2022; Peyser et al., 2022; Williams, 2022). Prior studies have devised objective functions to disentangle content from speech, fostering the ability of SSL models to generate speakerinvariant representations through domain-specific self-supervision (DS). In DS, a pre-trained SSL model is fine-tuned with unlabeled data for specific applications. Qian et al. (2022) propose ContentVec by disentangling speaker and content information, demonstrating promising results. However, ContentVec suffers from the requirement of a voice conversion model and substantial computational costs exceeding 600 GPU hours. Alternatively, Chang et al. (2023) propose Speaker-invariant Clustering (Spin) to produce content representations with minimal fine-tuning resources. Nonetheless, Spin is constrained to fine-tuning only the top layers, thereby lacking the flexibility to adapt to diverse acoustic domains.

Parallel to modeling content information in speech, numerous studies are dedicated to investigating the robustness of speech SSL. While current methods perform well on clean speech datasets, they are vulnerable to out-of-domain data like distorted audio signals (Hsu et al., 2021b). To mitigate this vulnerability, researchers have proposed noise-invariant training techniques. Huang et al. (2022a) proposes HuBERT-MGR via domain adversarial training to render the fine-tuned HuBERT model (Hsu et al., 2021a) invariant to domain shifts. WavLM (Chen et al., 2022) integrates denoising with the HuBERT pre-training framework, achieving state-of-the-art performance in many speech processing downstream tasks. Similarly, Zhu et al. (2023) propose Robust data2vec, introducing perturbations to the input to predict the exponential moving average teacher model's representations. In deHuBERT (Ng et al., 2023), the Barlow Twins loss (Zbontar et al., 2021) is applied to encourage representation invariability to input perturbations. Although many methods have shown success in noisy speech recognition (Wang et al., 2022; Zhu et al., 2022; Huang et al., 2022b; Hu et al., 2023), to our knowledge, none have concurrently addressed the disentanglement of speaker and noise while enhancing content information. Furthermore, these approaches exhibit inefficiency, often requiring high computation costs and iterating large corpora over numerous epochs.

![](images/6a901a4fa3ecb0817da5d2f964425554fbd197f5676abfc21820b60cf909b323.jpg)  
Figure 1: The proposed R-Spin domain-specific self-supervision framework. The input utterance is perturbed into a different voice and distorted with random noise. Both the original and perturbed views are fed into an encoder initialized with an SSL pre-trained model. The model is optimized with Speaker-invariant Clustering (Spin) (Chang et al., 2023) objective $( { \mathcal { L } } _ { \mathrm { s p i n } } )$ and frame-wise pseudo-label prediction loss $( \mathcal { L } _ { \mathrm { A u x } } )$

To effectively acquire high-quality content and robust representations for real-world applications, this paper extends Spin with noise-invariant training and acoustic piece pseudo-label learning, coined Robust Spin (R-Spin). During training, two utterances of the same content with different distortions are fed into a speech SSL encoder. The outputs are frame-wise vector-quantized with a learnable codebook via online clustering, as in Spin. The model is trained to match cluster ID distributions between the utterances. To prevent codebook collapse, an additional pseudo-label prediction loss is introduced. The pseudo-labels are generated by learning acoustic pieces (Ren et al., 2022) on top of the discrete units produced by a pre-trained Spin model, offering better training targets that closely align with phonemes and characters. Within this framework, the speech encoder learns speaker and noise-invariant representations, benefiting robustness and content extraction simultaneously. The contributions are summarized as follows:

1. We integrate predicting acoustic pieces into Spin, enabling fine-tuning all parameters without collapsing, which allows the processing of more complex speech recordings.

2. R-Spin inherits the benefit of low training costs from Spin, requiring 12X less computation than prior art.

3. With noise-invariant training, R-Spin outperforms other DS approaches in distorted speech and phoneme recognition tasks like the CHiME-4 challenge (Vincent et al., 2017).

4. We inspect the hidden representations of speech SSL models to quantify the speaker

and noise invariability.

5. We offer in-depth analyses of discrete acoustic units to understand how these units help speech encoder training.

## 2 Method

## 2.1 Overview

The proposed R-Spin framework is shown in Fig. 1. R-Spin is based on Speaker-invariant Clustering (Spin) (Chang et al., 2023), a domain-specific self-supervision method with online clustering and swapped prediction for capturing content representations (Sec. 2.2). We introduce noise-invariant training by perturbing inputs to improve robustness (Sec. 2.3). Moreover, an auxiliary pseudolabel prediction loss enables fine-tuning the entire model without collapsing (Sec. 2.4). Acoustic Piece is incorporated with the auxiliary loss to improve performance further (Sec. 2.5).

## 2.2 Speaker-invariant Clustering

Spin is an efficient DS method for improving content representations inspired by Swapping Assignments between Views (SwAV) (Caron et al., 2020). We briefly introduce Spin and suggest readers refer to the original paper for further details.

For each utterance in a mini-batch, the F0 frequency and the relative ratio between formant frequencies are randomly perturbed to mimic the same sentence spoken by a different speaker (Choi et al. 2021; Qian et al., 2022). The original and perturbed views are fed into a transformer encoder (Vaswani et al., 2017) initialized with an SSL model like HuBERT (Hsu et al., 2021a). The output representations $\textbf { H } = \mathbf { \Psi } [ \pmb { h } _ { 1 } \dots \pmb { h } _ { B } ] ^ { \intercal }$ of the original view are linearly projected and L2-normalized to $\mathbf { Z } = [ z _ { 1 } \ldots z _ { B } ] ^ { \mathsf { T } }$ , where B is the number of frames in a batch. We then use the representations to compute a probability distribution over a learnable codebook as $p \left( \cdot | z _ { b } \right)$ . We perform the same operations to the perturbed utterance, resulting in another distribution $q \left( \cdot | \tilde { z } _ { b } \right)$ , where \~ denotes features from the speaker-perturbed view. Next, q is smoothed by solving an optimal transport problem to enforce full codebook usage. Finally, the model is trained to perform swapped predictions between views by minimizing the cross-entropy loss

$$
\begin{array} { r l } {  { \mathcal { L } _ { \mathrm { { S p i n } } } = - \frac { 1 } { 2 B } \sum _ { b \in [ B ] } \sum _ { k \in [ K ] } q ( k | \tilde { \mathbf { z } } _ { b } ) \log p ( k | \boldsymbol { z } _ { b } ) } } \\ & { - \frac { 1 } { 2 B } \sum _ { b \in [ B ] } \sum _ { k \in [ K ] } q ( k | \boldsymbol { z } _ { b } ) \log p ( k | \tilde { \mathbf { z } } _ { b } ) , } \end{array}\tag{1}
$$

where K is the size of the learnable codebook, and the second term emerges from the interchangeability of the role of the perturbed and original speech.¹ Under this DS framework, the fine-tuned model produces speaker-invariant representations, making the content of speech signals more accessible to downstream applications.

## 2.3 Noise-invariant Training

To improve the robustness of SSL models, we introduce noise-invariant training by including audio distortions to both views of the input. We anticipate the model will be able to concurrently eliminate noise and speaker-related information, thereby enabling the trained model to generate robust content representations.

## 2.4 Auxiliary Pseudo-label Prediction Loss

As noted by Chang et al. (2023), Spin is constrained to fine-tuning solely the top layers of pre-trained SSL encoders. Otherwise, the model converges towards a trivial solution, yielding outputs irrelevant to the corresponding inputs. This limitation may not be problematic when the application domain closely aligns with the pre-training data. However, given that the bottom layers are associated with low-level signal processing like denoising (Chang et al., 2021; Gong et al., 2023), subjecting these layers to fine-tuning is imperative. This adjustment is particularly beneficial in enhancing the model's robustness to out-of-domain data. Consequently, we propose a pseudo-label prediction loss to prevent models from collapsing.

The pseudo-label prediction is a frame-wise classification problem with a loss function of

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { A u x } } = - \frac { 1 } { 2 B } \sum _ { b \in [ B ] } \log p \left( y _ { b } \big | \pmb { h } _ { b } \right) } \\ { \displaystyle - \frac { 1 } { 2 B } \sum _ { b \in [ B ] } \log p \left( y _ { b } \Big | \tilde { h } _ { b } \right) , } \end{array}\tag{2}
$$

where $y _ { b }$ is the pseudo-label at frame b. The probability distributions are computed by projecting h with a fully connected layer followed by a softmax. The choice of pseudo-labels is flexible, including Kmeans clusters of acoustic features and codewords produced by Spin. With this loss, the fine-tuned models are expected to preserve content even when all layers are fine-tuned. Combining Eqs. 1 and 2, the overall loss function is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { S p i n } } + \lambda \mathcal { L } _ { \mathrm { A u x } } , } \end{array}\tag{3}
$$

where $\lambda > 0$ is a hyper-parameter. ${ \mathcal { L } } _ { \mathrm { A u x } }$ has learning targets independent of the model, regularizing and stabilizing the training process. Meanwhile, ${ \mathcal { L } } _ { \mathrm { S p i n } }$ optimizes on varying labels from a dynamically changing codebook, offering flexibility to improve upon the pseudo-labels in ${ \mathcal { L } } _ { \mathrm { A u x } }$ . Therefore, the combined loss function is expected to enhance pre-trained speech SSL encoders and mitigate Spin's limitations.

## 2.5 Acoustic Piece

This section introduces acoustic pieces (Ren et al., 2022) to ${ \mathcal { L } } _ { \mathrm { A u x } }$ to further improve R-Spin. APs are learned by applying byte-pair encoding (BPE) (Sennrich et al., 2016) to discrete acoustic units like K-means clusters of HuBERT representations. AP captures high-level units close to phonemes and characters, useful for pre-training (Wu et al., 2023) and generation (Shen et al., 2024). Hence, we propose to set AP as the target of $\mathcal { L } _ { \mathrm { A u x } }$ to extract better content representations.

Following Ren et al. (2022), we first merge identical consecutive units in time for each utterance. The BPE algorithm is then applied to the reduced sequences to learn acoustic pieces. Next, we encode the entire training corpus into APs and duplicate the encoded units to the original utterance length. The encoded corpus is then used as the pseudo-labels for Eq. 2, expecting to encourage the fine-tuned SSL model to encode better phoneme and character representations.

## 3 Experiments

## 3.1 Data

The 960 hours of unlabeled English speech in LibriSpeech is used for R-Spin training (Panayotov et al., 2015).2 Audio distortions are generated with torch-audiomentations.3 Following Zhu et al. (2023), background noises are sampled from MU-SAN (Snyder et al., 2015) and CHiME-4 (Vincent et al., 2017), covering music, speech, and outdoor noise.4 Signal-to-noise ratios (SNR) are uniformly sampled from [—10, 10] during training. We add distortions to each utterance during evaluation, including Gaussian noise, MUSAN noise, and reverberation (Appendix A.5).

## 3.2 Implementation

The DS experiments are mostly based on WavLM (Chen et al., 2022) because WavLM is pretrained with a denoising objective, offering a good initialization. HuBERT (Hsu et al., 2021a) is also considered to demonstrate R-Spin's generalizability to SSL models trained with clean speech. We follow the implementations by Chang et al. (2023), which uses PyTorch (Paszke et al., 2019), PyTorch-Lightning (Falcon, 2019), and torchaudio (Yang et al., 2022).⁵ The acoustic pieces are generated by learning BPE tokens on top of a HuBERT + Spin2048 model (Appendix A.2). Further details can be found in Appendix A.3.6

## 3.3 Notations

We denote an SSL model X fine-tuned with Spin and K codewords with X + SpinK. In X + R-$\mathrm { S p i n } _ { K _ { 1 } , K _ { 2 } } , K _ { 1 }$ and $K _ { 2 }$ are respectively the codebook size of ${ \mathcal { L } } _ { \mathrm { S p i n } }$ and the number of classes of pseudo-labels for ${ \mathcal { L } } _ { \mathrm { A u x } }$ . If the pseudo-labels are acoustic pieces, $\mathbf { \ddot { s } } 6 \mathbf { P } ^ { \prime }$ is added to $K _ { 2 }$ . Unless specified otherwise, R-Spin denotes R-Spin32, AP40k.

## 3.4 Noisy Phoneme Recognition

We compare the phoneme recognition performance of SSL and DS methods under noisy conditions. The training setup is similar to the SUPERB phoneme recognition task (Yang et al., 2021), where the SSL models are frozen and only a lightweight prediction head is fine-tuned (Appendix $\mathsf { A } . 5 ) . ^ { 7 }$ We apply some classes of distortions only to testing data to obtain phoneme error rates (PER). We divide results by budget, which is the amount of speech processed during DS, proportional to the computational resources required (Sec. 3.6).

![](images/2630980ed263a1692502c4bca579f6ca400a1737a20b7a71c0598c8e2a336042.jpg)  
(a) Gaussian Noise

![](images/d887d67573aa869bdadf4356d3c13008a23d77162b1d0473a4e1a0ae01fcba5a.jpg)  
(b) MUSAN Noise  
Figure 2: Phoneme error rates (PER) under different noise types and SNRs. R-Spin32, AP40k is used here.

As shown in the middle columns of Table 1, R-Spin outperforms low and high-budget methods in all conditions. WavLM + R-Spin has the best overall PERs because WavLM is pre-trained with a denoising task, showing that model initialization contributes largely to the recognition performance after DS. Next, R-Spin improves unseen tasks like Gaussian noise and reverberation, indicating that noise-invariant training generalizes to some outof-domain perturbations. Furthermore, comparing Robust data2vec with R-Spin is unfair since the training costs are 12 times apart, so we train a lowbudget Robust data2vec (Appendix A.4). The noticeable performance drop in the low-budget model implies Robust data2vec requires high computation resources, but our approach offers competitive results with fewer training data.8

We plot PERs under different SNRs in Fig. 2 for a detailed comparison. Overall, R-Spin achieves the lowest PERs even when the SNR is high.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Processed Speech (hours)</td><td colspan="4">LibriSpeech test-other Phoneme Recognition (PER↓)</td><td colspan="2">CHiME-4 ASR (WER↓)</td></tr><tr><td></td><td></td><td>Clean Gaussian† MuSAN Reverb†</td><td></td><td>Real</td><td>Sim</td></tr><tr><td>No DS Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HuBERT (Hsu et al., 2021a)</td><td>0</td><td>10.7</td><td>74.5</td><td>50.2</td><td>23.2</td><td>72.7</td><td>63.1</td></tr><tr><td>WavLM (Chen et al., 2022)</td><td>0</td><td>10.3</td><td>59.9</td><td>45.1</td><td>19.4</td><td>52.4</td><td>46.4</td></tr><tr><td>DS Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HuBERT + Spin2048 (Chang et al., 2023)</td><td>0.4k</td><td>8.4</td><td>70.8</td><td>47.8</td><td>18.4</td><td>71.3</td><td>62.0</td></tr><tr><td>WavLM + Spin2048 (Chang et al., 2023)</td><td>0.4k</td><td>8.2</td><td>59.2</td><td>41.2</td><td>16.7</td><td>52.1</td><td>46.6</td></tr><tr><td>Robust data2vec (Low-budget)</td><td>10.4k</td><td>38.8</td><td>68.2</td><td>52.9</td><td>53.7</td><td>80.9</td><td>78.2</td></tr><tr><td>Proposed</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HuBERT + R-Spin32, AP40k</td><td>8.2k</td><td>8.3</td><td>36.4</td><td>18.2</td><td>16.3</td><td>34.3</td><td>34.1</td></tr><tr><td>WavLM + R-Spin32, AP40k</td><td>8.2k</td><td>8.2</td><td>33.7</td><td>16.7</td><td>14.9</td><td>26.4</td><td>26.6</td></tr><tr><td>High-budget DS Toplines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ContentVec500 (Qian et al., 2022)</td><td>76k</td><td>8.7</td><td>71.4</td><td>47.2</td><td>16.8</td><td>61.4</td><td>55.1</td></tr><tr><td>HuBERT-MGR (Huang et al., 2022a)</td><td>78k</td><td>9.5</td><td>37.1</td><td>36.3</td><td>18.3</td><td>49.7</td><td>44.3</td></tr><tr><td>Robust data2vec (Zhu et al., 2023)</td><td>105k</td><td>6.5</td><td>56.7</td><td>27.7</td><td>19.2</td><td>17.5</td><td>20.1</td></tr><tr><td>Supervised Toplines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Whisper Base (Radford et al., 2022)</td><td>一</td><td></td><td></td><td></td><td></td><td>17.9</td><td>23.3</td></tr><tr><td>Whisper Small (Radford et al., 2022)</td><td>一</td><td>一</td><td></td><td></td><td></td><td>10.8</td><td>14.3</td></tr></table>

†Unseen perturbation types for R-Spin and Robust data2vec.  
Table 1: Phoneme recognition on LibriSpeech and ASR on CHiME-4 test sets. Gaussian noise, MUSAN background noise, and reverberation (Reverb) are respectively added to simulate noisy conditions, where the SNRs are fixed to OdB. The calculation of the number of hours of processed speech during DS follows Chang et al. (2023).

HuBERT-MGR excels in Gaussian noise because it is the only model trained with this noise type. Nevertheless, R-Spin offers a similar performance in Gaussian noise across different SNRs, aligning with the results in Table 1.

## 3.5 Noisy Speech Recognition

This section assesses R-Spin with a noisy ASR task. We adopt the SUPERB ASR setup with the CHiME-4 corpus (Vincent et al., 2017) to evaluate the models in more realistic noisy recordings (Appendix A.6). The results in the right columns of Table 1 reveal that R-Spin surpasses low-budget baseline models. While R-Spin demonstrates commendable performance on CHiME-4, this method falls short compared to Robust data2vec, which benefits from training with a substantially higher budget. Furthermore, we set Whisper Base and Small as toplines due to their robustness demonstrated through large-scale weaklysupervised learning (Radford et al., 2022). R-Spin successfully mitigates the performance gap between WavLM and the Whisper toplines by over 60%. Combining phoneme and speech recognition findings, we conclude that R-Spin effectively enhances pre-trained SSL models in capturing robust content representations.

## 3.6 Data Efficiency

Developing R-Spin aims to enhance speech SSL models with minimal resources, including improving data efficiency. Following Chang et al. (2023), an analysis of the duration of speech data processed during training is undertaken to quantify the computational expenses associated with each method. As depicted in the second column of Table 1, these values are derived by multiplying the number of training updates and the effective batch size for each update. Compared with the high-budget methods, R-Spin requires significantly lower training costs, concurrently exhibiting superior performance across diverse conditions. A complete comparison of the costs can be found in Appendix C.

## 3.7 Representation Invariability

This section explores the robustness of models regarding representation invariability by examining their characteristics under diverse perturbations.

## 3.7.1 Speaker Invariability

We first inspect each model's invariability to speaker changes by computing the speaker identification (SID) accuracy with different hidden layer representations. The SID task follows SUPERB's setup but with 50k training updates. As shown in Fig. 5, R-Spin has a significantly lower SID accuracy for the top layers, demonstrating the effect of fine-tuning the whole model with a speakerinvariant objective. Moreover, requiring 9X less training costs, our method produces representations with less speaker information than ContentVec. Therefore, the proposed method outperforms prior speaker-invariant self-supervision approaches in removing speaker ID.

![](images/7d269168b0e3784c668075e45df008353ab26e126fb1cb8c4a99ee4c101d677a.jpg)  
(a) HuBERT CNN (Layer 0)

![](images/da40003a4eecbca9e6b25536e2ecc966520b32a382cde64d87aee4c2daf0f4db.jpg)  
(b) HuBERT + R-Spin CNN (Layer 0)

![](images/7340ef45f4889a7c191bdafcb79a2046f3e11d76e9770278f7c2b52574708fe3.jpg)  
(c) HuBERT Best Layer (9)

![](images/ae1cbec8d600498421fc660a00c235eaf9baa11ff7c1dd789a79529d3b95b755.jpg)  
(d) HuBERT + R-Spin Best Layer (12)

Figure 3: t-SNE (Van der Maaten and Hinton, 2008) visualization of the CNN and the layer with the lowest speaker identification rate given the same clean utterance spoken by three speakers from TIMIT (Garofolo, 1993). Each color represents a speaker, while each label visualizes a frame and the corresponding phoneme label. The transcription is “Don't ask me to carry an oily rag like that." The silence frames are omitted for clarity.  
![](images/7b593ba5d4b9dccfacc01e7cae9880e7d9fc1cd9bcf1c6711b4037f92c6a9ba6.jpg)  
(a) Gaussian Noise (SNR = 0dB)

![](images/8baf6811ae9daa4f2ca4fc4c6b758fe10315172b56d788511d17d788526a5eff.jpg)  
(b) Reverberation (real room impulse response)

Figure 4: Layer-wise perturbation invariability analyses with Linear CKA, where higher values indicate higher invariability to perturbations. The zeroth layer denotes the CNN feature extractor.  
![](images/46c47bb67b9b428bf7f8583abc614565876f62afeba2f42a2f713161dce2666a.jpg)  
Figure 5: Layer-wise speaker identification accuracy.

Next, we use t-SNE (Van der Maaten and Hinton, 2008) to visualize representations articulated by distinct speakers. We show the layer with the lowest SID rate according to Fig. 5. In Figs. 3a and 3b, there is a discernible clustering of frames uttered by the same speaker, suggesting that lower layers retain more speaker-specific information. Conversely, Figs. 3c and 3d illustrate that top layer features are grouped according to phonemes rather than speakers. Moreover, the top layer representations are context-dependent, as exemplified by the spatial arrangement of phonemes such as “carry" (k eh r iy) and the same phoneme /iy/ in the word "oily" (oy 1 iy). Besides, a comparative analysis between Figs. 3c and 3d reveals that R-Spin features exhibit a more prominent overlap among speakers than HuBERT. As a result, this section substantiates the speaker-invariability of R-Spin. Detailed visualization can be found in Appendix D.

## 3.7.2 Noise Invariability

We examine the response of continuous representations to input distortions. We compute linear centered kernel alignment (CKA) similarities (Kornblith et al., 2019) of frame-wise features with and without noisy inputs, where a higher similarity indicates a higher invariability to distortions. The evaluation involves building datasets derived from the LibriSpeech dev-clean and dev-other sets augmented with various distortions. Fig. 4 illustrates that R-Spin exhibits superior noise invariability for the upper layers than other models, indicating the efficacy of noise-invariant training even if the noise types are unseen. Lower layers tend to have lower similarities, suggesting a higher sensitivity to perturbations. This observation aligns with existing research discussed in Sec. 2.4, which associates lower layers with fundamental signal processing functions. In contrast, Robust data2vec has a greater noise invariability starting from the bottom layers because of the data2vec training strategy.

![](images/9b364630eba35cfeb650803c69d563ac365dfed2fa5643711053e9e39fbf2157.jpg)  
(a)

![](images/bac7aad013fbbcd22e8ac36102c53894295f0aa715a4cbb4bc641d18b74337d3.jpg)  
(b)

![](images/15ac9378efe947fc489326e0807a4cb58648ed163f47cc962611f4a69b6ca569.jpg)  
(c)

Figure 6: WavLM + R-Spin with different (a) codebook and (b)(c) AP vocabulary sizes. (b) and (c) depict the phoneme and character segmentation R-values, where the dotted curves are the baselines by segmenting each utterance with equal-length segments given the number of boundaries obtained by the APs. The PERs are calculated by averaging over different noise conditions on LibriSpeech test-other. The WERs are the averaged scores of the real and simulated evaluation sets of CHiME-4.  
![](images/17f21a2ce0dd27fe0cd51ed100bad839e756d48528ad6a15daeef48206faf2ee.jpg)

![](images/9ee524d5db5980e9cb2dc7258ddb603ee4a6c3c869ba0b6a0eb5db66ab11b7e8.jpg)  
(b) HuBERT + R-Spin Best Layer (12)  
Figure 7: t-SNE visualization of hidden representations of the same utterance in Fig. 3 with different distortions indicated by colors, where SNR = OdB.

We employ t-SNE to explore representations under distortions. As shown in Fig. 7, the R-Spin features exhibit a more pronounced overlap than HuBERT, suggesting that R-Spin improves robustness to noise, aligning with the observations in Fig. 4. Fig. 7b reveals that R-Spin features exposed to MUSAN noise exhibit a high degree of overlap with unperturbed ones, whereas the other two perturbation types diverge slightly because Gaussian noise and reverberation are unseen for R-Spin. Overall, the analysis underscores the notable noise invariability offered by R-Spin.

## 3.8 Importance of Discrete Units

This section analyzes the efficacy of APs and their relation to phonemes and characters.

## 3.8.1 Codebook and Acoustic Pieces Size

We inspect the importance of the codebook size in Spin. As highlighted by Chang et al. (2023), the codebook size positively correlates with phoneme recognition. A similar trend can be found in Fig. 6a but has an inverted trend for ASR. However, the observed performance discrepancy is less than 1% absolute, suggesting that codebook size's impact on R-Spin is marginal. In contrast, substantial improvements in ASR are observed with more APs, but not in phoneme recognition, as evidenced by Figs. 6c and 6b. To analyze this phenomenon, we investigate R-Spin's phoneme and character segmentation capabilities using discrete units.

## 3.8.2 Phoneme and Character Segmentation

We segment speech with acoustic pieces and show the R-values in Figs. 6b and 6c. R-value, a metric for evaluating word or phoneme segmentation quality (Räsänen et al., 2009), is robust to oversegmentation, an issue that plagues F1. The boundaries are predicted by locating differing adjacent discrete units. We evaluate on force-aligned LibriSpeech dev-clean and dev-other sets (Lugosch et al., 2019; McAuliffe et al., 2017).9 The character boundaries are obtained by dividing each forcealigned word segment into equal-length segments corresponding to individual characters within the word. More accurate boundaries can be computed with character-based aligners, but we only need a rough estimation of the segmentation quality.

As depicted in both Figs. 6b and 6c, larger AP vocabulary sizes have superior segmentation, indicating that more APs form units that closely resemble linguistic units. The baseline, which involves uniformly segmenting utterances based on the number

![](images/a78df0fe30933e209e1478ae418b976729242082e52f7ccf697e158d2fd12a61.jpg)  
Figure 8: An example of phoneme alignment of an utterance “This had some effect in calming him." from LibriSpeech dev-clean. The black lines indicate the force-aligned boundaries, while the red dashed lines are the predicted boundaries with AP40k.

<table><tr><td>Method</td><td colspan="2">CHiME-4 Real Sim</td></tr><tr><td>Spin2048 (Chang et al., 2023) R-Spin32, AP40k (Proposed)</td><td>52.1</td><td>46.6 26.6</td></tr><tr><td>no  ${ \mathcal { L } } _ { \mathrm { A u x } }$  no  ${ \mathcal { L } } _ { \mathrm { S p i n } }$ </td><td>26.4 47.8 31.9</td><td>45.6 32.4</td></tr><tr><td>no speaker perturbation no additive noise</td><td>28.3 49.4</td><td>28.0 46.8</td></tr><tr><td>Pseudo Label for  $\mathcal { L } _ { \mathrm { A u x } }$  Spin2048 codebook MFCC (K-means 512) MFCC (K-means 2048)  $5 1 2 ) ^ { \bullet }$ </td><td>28.3 46.9</td><td>29.1 45.4</td></tr></table>

Pairwise t-tests between these results all have $p > 0 . 0 5$

Also, $p < 0 . 0 5$ when they are compared with R-Spin32, AP40k.

Table 2: CHiME-4 ASR results for ablation studies based on fine-tuned WavLM models.

of boundaries derived from APs, underscores the non-random nature of AP boundaries. Although the segmentation capability of APs is incomparable with other unsupervised speech segmentation algorithms (Kreuk et al., 2020), they present significantly improved targets for ${ \mathcal { L } } _ { \mathrm { A u x } } .$ consequently enhancing the accuracy of ASR. A full comparison of unsupervised phoneme segmentation can be found in Appendix B.3.

Furthermore, we provide an example of segmenting an utterance with 40k APs in Fig. 8. The red dashed stripes depict that the boundaries of APs are mostly aligned with phoneme boundaries. Notably, the predicted boundaries occasionally exhibit a slight temporal lag compared to the ground truth, like the first /ah/ and $/ \mathfrak { m } /$ . We suspect the 50Hz framerate of HuBERT or the Spin training objective causes this phenomenon since they could reduce time resolution and introduce temporal shifts. Still, the actual cause remains a subject for future investigation. To summarize, APs effectively learn discrete acoustic units that benefit ASR performance.

## 3.9 Ablation Studies

Under the same CHiME-4 ASR setup in Sec. 3.5, we conduct ablation studies to analyze the proposed methods. As shown in Table 2, the WERs increase significantly without ${ \mathcal { L } } _ { \mathrm { A u x } } )$ , showing that the auxiliary loss helps ASR performance and mitigates collapsing. Second, WERs increase by about 5% without ${ \mathcal { L } } _ { \mathrm { S p i n } }$ , indicating the necessity of this loss for achieving perturbation-invariant representations. Speaker perturbation also plays an important role in offering good content representations according to the degraded WERs. Moreover, the fine-tuned model exhibited suboptimal performance when trained without noise, emphasizing the importance of noise-invariant training for improving robustness. The above findings verify the necessity of the proposed approaches.

We inspect the effect of choosing different pseudo-labels for ${ \mathcal { L } } _ { \mathrm { A u x } }$ First, APs are helpful for R-Spin since learning from the original Spin model's codeword labels increases WERs by over 2%. Next, we replace the pseudo-labels with the more commonly used K-means clustered representations (Hsu et al., 2021a). Clustered MFCC features degrade R-Spin the most, no matter the number of clusters used. In contrast, clustered HuBERT representations from layer 9 (L9) yield results comparable to $\mathrm { S p i n } _ { 2 0 4 8 }$ , and the t-test suggests the disparities between pseudo-labels are statistically insignificant. Thus, using clustered features from an SSL model is a viable alternative. Additional ablation studies are available in Appendix B.1.

## 4 Conclusion

This paper proposes R-Spin, a domain-specific self-supervision method with speaker and noiseinvariant clustering for robust content representations. Results illustrate the efficacy and broad applicability of R-Spin across various acoustic scenarios, even within constrained computation budgets. The acoustic analyses presented in this study offer insights into the characteristics of discrete units of this nature and strategies for their utilization. Future directions involve scaling to larger models and exploring its application in diverse downstream tasks like robust voice conversion.

## Acknowledgements

We thank Alexander H. Liu, Saurabhchand Bhati, Nauman Dawalatabad, and Yuan Gong for their insightful feedback.

## Limitations

This paper faces four primary limitations due to constrained computation resources and available data. First, we consider background noises including human speech, music, and natural noises, and evaluate the proposed methods with similar noise types and reverberation, covering many real-world conditions. However, the trained models may encounter challenges in processing more severely distorted audio data, such as air traffic control communications. Second, the dataset employed consists of English utterances spoken by native speakers, predominantly of North American dialects, leaving the performance in other languages and accents unexplored. Third, the experiments are conducted on 95M-parameter models, so the scalability of R-Spin remains unknown. Last, to fully comprehend the capabilities of the proposed method, further analyses and extensions to other applications are recommended for future exploration (Sicherman and Adi, 2023). These questions can be answered by experimenting with diverse datasets and more computation resources.

## Ethics Statement

Our models inherit the biases of SSL models (Hu-BERT and WavLM) pre-trained on the LibriSpeech corpus. This corpus contains read English audio recordings derived from audiobooks. Limitations arise when confronted with accents and topic domains outside the corpus scope, potentially diminishing the effectiveness of the proposed methods. Thus, the direct application of our models to real-world scenarios may result in increased speech recognition error rates. These errors, if unaddressed, can propagate through downstream applications like natural language processing systems, leading to potential risks for users, such as the misinterpretation of voice commands.

## References

Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. 2022. data2vec: A general framework for self-supervised learning in speech, vision and language. In ICML.

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. wav2vec 2.0: A framework for self-supervised learning of speech representations. In NeurIPS.

Saurabhchand Bhati, Jesús Villalba, Piotr Żelasko, Laureano Moro-Velazquez, and Najim Dehak. 2021. Segmental contrastive predictive coding for unsupervised word segmentation. In Interspeech.

Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. 2020. Unsupervised learning of visual features by contrasting cluster assignments. In NeurIPS.

David M Chan and Shalini Ghosh. 2022. Contentcontext factorized representations for automated speech recognition. In Interspeech.

Heng-Jui Chang, Alexander H. Liu, and James Glass. 2023. Self-supervised Fine-tuning for Improved Content Representations by Speaker-invariant Clustering. In Interspeech.

Heng-Jui Chang, Alexander H Liu, Hung-yi Lee, and Lin-shan Lee. 2021. End-to-end whispered speech recognition with frequency-weighted approaches and pseudo whisper pre-training. In SLT.

Heng-Jui Chang, Shu-wen Yang, and Hung-yi Lee. 2022. DistilHuBERT: Speech representation learning by layer-wise distillation of hidden-unit bert. In ICASSP.

Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, et al. 2022. Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE JSTSP, 16.

Hyeong-Seok Choi, Juheon Lee, Wansoo Kim, Jie Lee, Hoon Heo, and Kyogu Lee. 2021. Neural analysis and synthesis: Reconstructing speech from selfsupervised representations. In NeurIPS.

William A Falcon. 2019. Pytorch lightning. GitHub, 3.

John S Garofolo. 1993. Timit acoustic phonetic continuous speech corpus. LDC.

Yuan Gong, Sameer Khurana, Leonid Karlinsky, and James Glass. 2023. Whisper-at: Noise-robust automatic speech recognizers are also strong audio event taggers. In Interspeech.

Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. 2021a. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. TASLP, 29.

Wei-Ning Hsu, Anuroop Sriram, Alexei Baevski, Tatiana Likhomanenko, Qiantong Xu, Vineel Pratap, Jacob Kahn, Ann Lee, Ronan Collobert, Gabriel Synnaeve, et al. 2021b. Robust wav2vec 2.0: Analyzing domain shift in self-supervised pre-training. arXiv.

Yuchen Hu, Chen Chen, Qiushi Zhu, and Eng Siong Chng. 2023. Wav2code: Restore clean speech representations via codebook lookup for noise-robust asr. arXiv.

Kuan Po Huang, Yu-Kuan Fu, Yu Zhang, and Hungyi Lee. 2022a. Improving distortion robustness of self-supervised speech processing tasks with domain adaptation. In Interspeech.

Wenyong Huang, Zhenhe Zhang, Yu Ting Yeung, Xin Jiang, and Qun Liu. 2022b. Spiral: Self-supervised perturbation-invariant representation learning for speech pre-training. In ICLR.

Tom Ko, Vijayaditya Peddinti, Daniel Povey, Michael L Seltzer, and Sanjeev Khudanpur. 2017. A study on data augmentation of reverberant speech for robust speech recognition. In ICASSP.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. 2019. Similarity of neural network representations revisited. In ICML.

Felix Kreuk, Joseph Keshet, and Yossi Adi. 2020. Self-supervised contrastive learning for unsupervised phoneme segmentation. In Interspeech.

Alexander H Liu, Heng-Jui Chang, Michael Auli, Wei-Ning Hsu, and James R Glass. 2023. Dinosr: Selfdistillation and online clustering for self-supervised speech representation learning. In NeurIPS.

Shuo Liu, Adria Mallol-Ragolta, Emilia Parada-Cabaleiro, Kun Qian, Xin Jing, Alexander Kathan, Bin Hu, and Bjoern W Schuller. 2022. Audio selfsupervised learning: A survey. Patterns, 3(12).

Loren Lugosch, Mirco Ravanelli, Patrick Ignoto, Vikrant Singh Tomar, and Yoshua Bengio. 2019. Speech model pre-training for end-to-end spoken language understanding. In Interspeech.

Michael McAuliffe, Michaela Socolof, Sarah Mihuc, Michael Wagner, and Morgan Sonderegger. 2017. Montreal forced aligner: Trainable text-speech alignment using kaldi. In Interspeech.

Abdelrahman Mohamed, Hung-yi Lee, Lasse Borgholt, Jakob D Havtorn, Joakim Edin, Christian Igel, Katrin Kirchhoff, Shang-Wen Li, Karen Livescu, Lars Maaløe, et al. 2022. Self-supervised speech representation learning: A review. IEEE JSTSP.

Dianwen Ng, Ruixi Zhang, Jia Qi Yip, Zhao Yang, Jinjie Ni, Chong Zhang, Yukun Ma, Chongjia Ni, Eng Siong Chng, and Bin Ma. 2023. De'hubert: Disentangling noise in a self-supervised model for robust speech recognition. In ICASSP.

Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. fairseq: A fast, extensible toolkit for sequence modeling. In NAACL-HLT: Demonstrations.

Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. 2015. Librispeech: An ASR corpus based on public domain audio books. In ICASSP.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS.

Douglas B. Paul and Janet M. Baker. 1992. The design for the Wall Street Journal-based CSR corpus. In HLT.

Cal Peyser, W. Ronny Huang, Andrew Rosenberg, Tara Sainath, Michael Picheny, and Kyunghyun Cho. 2022. Towards disentangled speech representations. In Interspeech.

Kaizhi Qian, Yang Zhang, Heting Gao, Junrui Ni, Cheng-I Lai, David Cox, Mark Hasegawa-Johnson, and Shiyu Chang. 2022. Contentvec: An improved self-supervised speech representation by disentangling speakers. In ICML.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2022. Robust speech recognition via large-scale weak supervision. arXiv.

Okko Johannes Räsänen, Unto Kalervo Laine, and Toomas Altosaar. 2009. An improved speech segmentation quality measure: the r-value. In Interspeech.

Shuo Ren, Shujie Liu, Yu Wu, Long Zhou, and Furu Wei. 2022. Speech pre-training with acoustic piece. In Interspeech.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In ACL.

Feiyu Shen, Yiwei Guo, Chenpeng Du, Xie Chen, and Kai Yu. 2024. Acoustic bpe for speech generation with discrete tokens. In ICASSP.

Amitay Sicherman and Yossi Adi. 2023. Analysing discrete self supervised speech representation for spoken language modeling. In ICASSP.

David Snyder, Guoguo Chen, and Daniel Povey. 2015. Musan: A music, speech, and noise corpus. arXiv.

Luke Strgar and David Harwath. 2022. Phoneme segmentation using self-supervised speech models. In SLT.

Andros Tjandra, Ruoming Pang, Yu Zhang, and Shigeki Karita. 2021. Unsupervised learning of disentangled speech content and style representation. In Interspeech.

Hsiang-Sheng Tsai, Heng-Jui Chang, Wen-Chin Huang, Zili Huang, Kushal Lakhotia, Shu-wen Yang, Shuyan Dong, Andy Liu, Cheng-I Lai, Jiatong Shi, Xuankai

Chang, Phil Hall, Hsuan-Jui Chen, Shang-Wen Li, Shinji Watanabe, Abdelrahman Mohamed, and Hungyi Lee. 2022. SUPERB-SG: Enhanced speech processing universal PERformance benchmark for semantic and generative capabilities. In ACL.

Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-sne. Journal of machine learning research, 9(11).

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In NIPS.

Emmanuel Vincent, Shinji Watanabe, Aditya Arie Nugraha, Jon Barker, and Ricard Marxer. 2017. An analysis of environment, microphone and data simulation mismatches in robust speech recognition. Computer Speech & Language, 46:535–557.

Yiming Wang, Jinyu Li, Heming Wang, Yao Qian, Chengyi Wang, and Yu Wu. 2022. Wav2vec-switch: Contrastive learning from original-noisy speech pairs for robust speech recognition. In ICASSP.

Jennifer Williams. 2022. Learning disentangled speech representations. Ph.D. thesis, The University of Edinburgh.

Felix Wu, Kwangyoun Kim, Shinji Watanabe, Kyu J Han, Ryan McDonald, Kilian Q Weinberger, and Yoav Artzi. 2023. Wav2seq: Pre-training speech-totext encoder-decoder models using pseudo languages. In ICASSP.

Shu-wen Yang, Po-Han Chi, Yung-Sung Chuang, Cheng-I Jeff Lai, Kushal Lakhotia, Yist Y Lin, Andy T Liu, Jiatong Shi, Xuankai Chang, Guan-Ting Lin, et al. 2021. SUPERB: Speech processing universal performance benchmark. In Interspeech.

Yao-Yuan Yang, Moto Hira, Zhaoheng Ni, Artyom Astafurov, Caroline Chen, Christian Puhrsch, David Pollack, Dmitriy Genzel, Donny Greenberg, Edward Z Yang, et al. 2022. Torchaudio: Building blocks for audio and speech processing. In ICASSP.

Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. 2021. Barlow twins: Self-supervised learning via redundancy reduction. In ICML.

Qiu-Shi Zhu, Jie Zhang, Zi-Qiang Zhang, Ming-Hui Wu, Xin Fang, and Li-Rong Dai. 2022. A noise-robust self-supervised pre-training model based speech representation learning for automatic speech recognition. In ICASSP.

Qiu-Shi Zhu, Long Zhou, Jie Zhang, Shu-Jie Liu, Yu-Chen Hu, and Li-Rong Dai. 2023. Robust data2vec: Noise-robust speech representation learning for asr by combining regression and improved contrastive learning. In ICASSP.

## A Implementation Details

## A.1 Speech SSL Models

Each SSL model used in this paper has a 7-layer CNN feature extractor and a 12-layer transformer encoder (Vaswani et al., 2017), having roughly 95M parameters in total. All SSL models are pretrained with 960 hours of unlabeled speech in the LibriSpeech corpus. HuBERT (Hsu et al., 2021a) is pre-trained in two iterations. In the first iteration, the encoder model learns to predict each masked frame's K-means cluster ID of MFCC features. The second iteration model has learning targets obtained by clustering hidden representations of the first iteration model. WavLM (Chen et al., 2022) follows the second iteration of HuBERT, but the training process involves a denoising task by adding random noises to the input to increase robustness. ContentVec (Qian et al., 2022) is finetuned on top of a pre-trained HuBERT model, but the inputs are augmented with speaker perturbation so that the model learns to produce representations invariant of the speaker. ContentVec's learning targets are obtained by converting all LibriSpeech data into the same speaker with a voice conversion model and then applying K-means clustering to the hidden features of HuBERT, given the converted inputs. HuBERT-MGR (Huang et al., 2022a) continues the HuBERT pre-training process with noisy speech and an auxiliary domain adversarial training objective to enhance robustness. HuBERT-MGR is trained with a mix of clean and distorted speech, where the distortions include MUSAN background noise, Gaussian noise, and reverberation. Robust data2vec (Zhu et al., 2023) fine-tunes a pre-trained data2vec model. Unlike data2vec, the inputs to the student model include background noise so that the model learns denoising. An additional contrastive learning objective is incorporated to enhance robustness. The pre-trained model weights are obtained from the s3prl toolkit.10

## A.2 Spin

Since R-Spin is trained with 960 hours of speech in LibriSpeech, the pseudo-labels for ${ \mathcal { L } } _ { \mathrm { A u x } }$ should be generated for all those data with Spin. To avoid labeling unseen data with Spin, we train another Hu-$\mathrm { B E R T } + \mathrm { S p i n } _ { 2 0 4 8 }$ model with the same data (originally 100 hours in Chang et al., 2023). Each minibatch before data perturbation has 2560 seconds of speech, equivalent to 32k frames after downsampling. The learning rate first linearly increases from $1 0 ^ { - 6 }$ to $1 0 ^ { - 4 }$ in the first 2.5k updates, then linearly decreases to $1 0 ^ { - 6 }$ in the last 7.5k updates. The implementation of the Spin loss follows Caron et al. (2020).11 This model takes four hours of training time on four RTX A6000 GPUs. Models trained with all 10k updates are used to generate pseudo-labels. In total, roughly 7.1k hours of unlabeled speech data are processed. Compared with the model in Chang et al. (2023), a similar performance is achieved on phoneme recognition.

## A.3 R-Spin

Each mini-batch before perturbation has 384 seconds of speech, equivalent to 19.2k frames in each view. Each utterance is first speaker-perturbed to generate a second view. All utterances are added with noise from MUSAN and CHiME-4 with an SNR in [–10, 10] dB. The noise for each view is independent. The learning rate first linearly increases from $1 0 ^ { - 6 }$ to $1 0 ^ { - 4 }$ in the first 4k updates, then linearly decreases to $1 0 ^ { - 6 }$ in the last 6k updates. λ in Eq. 3 is set to 5. Each R-Spin DS training takes less than eight hours on two RTX A6000 GPUs. Models trained with all 10k updates are used for evaluation. For the R-Spin training, 1.1k hours of unlabeled speech data are processed. Combined with the Spin training in Appendix A.2, 8.2k hours of data are used during DS.

## A.4 Low-budget Robust data2vec

We follow the implementation of Zhu et al. (2023) with fairseq (Ott et al., 2019).12 We changed the training data from CHiME-4 to LibriSpeech for a fair comparison with our method. Because we found a long training schedule is necessary for Robust data2vec converge, the number of updates is the same as the original implementation (100k). Meanwhile, the mini-batch size is reduced from 63 to 6.25 minutes so that the amount of speech data processed is similar to R-Spin. The rest of the hyperparameters remain the same since we found the original ones are sufficiently good. As shown in Table 1, the low-budget Robust data2vec model has a significant performance degradation compared with the fully-trained version, implying the necessity to train this model with a large batch size. Concurrently, R-Spin achieves superior results under the same budget, indicating that our approach is more data-efficient.

We found that the low-budget Robust data2vec is particularly difficult to train. First, Hsu et al. (2021a) has shown that large batch sizes favor speech SSL model training, which applies to Robust data2vec. Second, we found that the lowbudget model fails when trained with CHiME-4, indicating the training corpus highly affects model convergence. Third, we did not have the resources to find the optimal hyperparameters. However, the hyperparameters for data2vec must be carefully determined to let the model converge (Baevski et al., 2022), which is amplified when the training scale is reduced. In conclusion, we were unable to find a setup where a comparable computational budget is used for both R-Spin and Robust data2vec. Nonetheless, the results demonstrate that the proposed R-Spin is much easier to operate under lowbudget scenarios.

## A.5 Phoneme Recognition

We follow the setup in SUPERB (Yang et al., 2021), which freezes each SSL model and uses a set of learnable weights to weighted-sum hidden features of all layers. The aggregated frame-wise features are fed into a lightweight linear prediction head to perform downstream tasks. Only the prediction head and the weighted-sum mechanism are finetuned with clean and labeled speech data to reveal the capabilities of SSL models. The LibriSpeech train-clean-100 and the test-other subsets are used as the training and evaluation datasets, respectively. The prediction head projects features to phoneme labels. Unlike the SUPERB setup, the learning rate is $5 \times 1 0 ^ { - 4 }$ (originally $1 0 ^ { - 2 } )$ to obtain a better performance, and the number of training updates is 30k (originally 100k). The noise and perturbation data sources are listed as follows.

1. Gaussian Noise: The Gaussian noises are generated with PyTorch.

2. Background Noise: The background noises are sampled from the MUSAN dataset. We duplicate the noise recording when it is shorter than the input. Otherwise, we randomly crop the recording to match the utterance length.

3. Reverberation: We filter waveforms with real and simulated impulse responses in RIRS (Ko et al., 2017).13 The scores for the real and simulated reverberation are averaged.

![](images/59e4bffae7bdc7ebf51b175b3191e77c79a737ef7061b54231de050987d3d0e9.jpg)  
Figure 9: AP size vs. actual vocabularies used.

## A.6 CHiME-4 ASR

We follow the ASR task of SUPERB, but the prediction heads (two-layer BLSTM) are trained with the clean portion of the CHiME-4 speech corpus obtained from the WSJO corpus (Paul and Baker, 1992), consisting of 14 hours of clean English speech. The number of training updates is 100k (originally 200k). The trained ASR models are evaluated on the 1-channel track of the CHiME-4 challenge. We report the averaged WERs over each subset (real and simulated data). We apply Whisper normalization to all ASR results for a fair comparison with the Whisper toplines.14

## A.7 Acoustic Pieces

We implemented the BPE algorithm in Python. The AP vocabulary sizes vs. the actual APs used are shown in Fig. 9. Since some merging operations in BPE replace previously learned BPE vocabularies with new ones, the number of used BPEs in the encoded LibriSpeech corpus is smaller than the learned BPE vocabularies. E.g., we have a sentence “a a b b a" and the learned BPE vocabularies a, b, aa, bb, and aabb. Then, the encoded sentence is “aabb a,"eliminating the intermediate vocabularies b, aa, and bb. Thus, the number of classes in the linear prediction head for ${ \mathcal { L } } _ { \mathrm { A u x } }$ is adjusted accordingly. E.g., the prediction head's output for R-Spin32, 40k is 19857 instead of 40000.

## B Additional Experiments

## B.1 Ablation Studies

## B.1.1 Hyperparameters

To examine the impact of the auxiliary loss, we change the value of λ in Eq. 3. As shown in Table 3, the differences of ASR WERs between different λ's are negligible. We can conclude that combining ${ \mathcal { L } } _ { \mathrm { S p i n } }$ and ${ \mathcal { L } } _ { \mathrm { A u x } }$ is necessary, and the ratio between the two objectives is robust.

<table><tr><td></td><td colspan="2">CHiME-4</td></tr><tr><td>Method</td><td>Real</td><td>Sim</td></tr><tr><td>Spin2048 (Chang et al., 2023)</td><td>52.1</td><td>46.6</td></tr><tr><td>R-Spin32, BPE40k</td><td>26.4</td><td>26.6</td></tr><tr><td>Hyperparameters</td><td></td><td></td></tr><tr><td>λ = 1</td><td>26.3</td><td>27.7</td></tr><tr><td>λ = 0.5</td><td>26.6</td><td>27.3</td></tr><tr><td>Layer to Apply  ${ \mathcal { L } } _ { \mathrm { A u x } }$ </td><td></td><td></td></tr><tr><td>Layer 11</td><td>28.1</td><td>28.8</td></tr><tr><td>Layer 10</td><td>34.7</td><td>33.8</td></tr><tr><td>Layer to Apply  ${ \mathcal { L } } _ { \mathrm { S p i n } }$ </td><td></td><td></td></tr><tr><td>Layer 11</td><td>27.2</td><td>27.9</td></tr><tr><td>Layer 10</td><td>27.0</td><td>27.8</td></tr><tr><td>Fine-tuned Layers</td><td></td><td></td></tr><tr><td>Top 10 Layers</td><td>29.7</td><td>30.0</td></tr><tr><td>Top 6 Layers</td><td>39.4</td><td>37.5</td></tr><tr><td>Dataset</td><td></td><td></td></tr><tr><td>LibriSpeech 100h</td><td>27.2</td><td>27.6</td></tr><tr><td>LibriSpeech 360h</td><td>26.6</td><td>27.6</td></tr><tr><td>Noise SNR Range</td><td></td><td></td></tr><tr><td>0 - 20dB</td><td>29.0</td><td>28.6</td></tr></table>

Table 3: CHiME-4 ASR results for additional ablation studies based on fine-tuned WavLM models.

## B.1.2 Layer to Apply $\mathcal { L } _ { \mathbf { A u x } }$

In the R-Spin design, ${ \mathcal { L } } _ { \mathrm { A u x } }$ is applied to the last layer. Here, we apply ${ \mathcal { L } } _ { \mathrm { A u x } }$ to other hidden layers to verify that our approach leads to the best overall result. When we move the auxiliary loss ${ \mathcal { L } } _ { \mathrm { A u x } }$ to lower layers, the performance degrades significantly, showing that this loss should regularize the entire model. Otherwise, the Spin loss still makes the codebook collapse.

## B.1.3 Layer to Apply $\mathcal { L } _ { \mathbf { S p i n } }$

Similar to the previous experiments, we apply ${ \mathcal { L } } _ { \mathrm { S p i n } }$ to lower layers to find the optimal design. The ASR performance degrades slightly when we move the Spin objective function to lower layers. With the results of ${ \mathcal { L } } _ { \mathrm { A u x } }$ , we conclude that a relatively good strategy for applying the two proposed loss functions is adding both to the top layer.

## B.1.4 Fine-tuned Layers

Here, we inspect the benefits of fine-tuning SSL models entirely in contrast to Spin, which finetunes only the top two layers. Hence, we reduce the number of fine-tuned layers. The results indicate that the model cannot adapt to noisy scenarios by fine-tuning fewer top layers. Thus, R-Spin is beneficial since we can now fine-tune the entire model for noisy conditions.

## B.1.5 Data

We further changed the data for R-Spin DS to reveal the impact of training corpora on performance. We found that WERs degrade slightly when the training corpus size is reduced. Moreover, the ASR performance degrades prominently by increasing the SNRs of the background noise for the noiseinvariant training. Hence, the choice of noise data and SNRs has a greater impact on the downstream performance than that of the clean speech corpus.

## B.2 Importance of Hidden Representations

We visualize the weighted sum mechanism for phoneme and speech recognition to understand the importance of each layer. The weights form a probability distribution over all layers (including the CNN feature extractor). The features of each layer are weighted and summed with these weights. However, the scale of the embedding spaces differs between layers. Suppose the weight of a layer is small, but the norm of the corresponding hidden vectors is large. That layer might contribute significantly to the downstream task. Consequently, we normalize each weight by multiplying with the averaged L2 norm of the corresponding layer embedding, which is written as

$$
\hat { w } _ { l } = w _ { l } \cdot \mathbb { E } \left[ \left. h ^ { ( l ) } \right. _ { 2 } \right] ,
$$

where $w _ { l }$ and $\mathbf { \Omega } _ { h } ( l )$ are respectively the unnormalized weight and hidden features of layer l, and E is the expectation over all samples from the LibriSpeech dev-clean and dev-other sets (Chang et al. 2022). Next, the new set of weights $\hat { w } _ { l }$ is normalized to sum to one. As shown in Fig. 10, the last layer of R-Spin has the least speaker and noise information, but the second last layer offers the best phoneme representations. In contrast, when R-Spin is applied, the best ASR layers tend to shift towards the last layer.

![](images/283dc82b67fbc2be8640670391ba23e38edd40a66b1167c9d3f1d0542aab61a4.jpg)  
(a) Phoneme Recognition

![](images/ad98e764e1c8e8b92a1bcab8d01664b68330d6d3396b1489b4ee275d26771b62.jpg)  
(b) Automatic Speech Recognition

Figure 10: Normalized weights of the weighted sum mechanism in the SUPERB PR and ASR.
<table><tr><td>Method</td><td>Precision↑</td><td>Recall↑</td><td>F1↑</td><td>OS→0</td><td>R-val↑</td></tr><tr><td colspan="6">Baseline</td></tr><tr><td>Oracle Uniform</td><td>56.49</td><td>62.99</td><td>59.56</td><td>11.50</td><td>63.47</td></tr><tr><td colspan="6">Unsupervised</td></tr><tr><td>CPC (Kreuk et al., 2020)</td><td>83.89</td><td>83.55</td><td>83.71</td><td></td><td>86.02</td></tr><tr><td>SCPC (Bhati et al., 2021)</td><td>84.63</td><td>86.04</td><td>85.33</td><td></td><td>87.44</td></tr><tr><td>HuBERT readout (Strgar and Harwath, 2022)</td><td>90.98</td><td>88.48</td><td>89.71</td><td></td><td>90.98</td></tr><tr><td colspan="6">Spin Codebook</td></tr><tr><td>HuBERT + Spin128 (Chang et al., 2023)</td><td>64.76</td><td>87.87</td><td>74.56</td><td>35.69</td><td>64.25</td></tr><tr><td>HuBERT + Spin256 (Chang et al., 2023)</td><td>61.71</td><td>90.84</td><td>73.49</td><td>47.22</td><td>56.02</td></tr><tr><td>HuBERT + Spin512 (Chang et al., 2023)</td><td>60.78</td><td>95.46</td><td>74.27</td><td>57.07</td><td>49.60</td></tr><tr><td>HuBERT + Spin1024 (Chang et al., 2023)</td><td>59.93</td><td>97.95</td><td>74.36</td><td>63.44</td><td>45.11</td></tr><tr><td>HuBERT + Spin2048 (Chang et al., 2023)</td><td>58.58</td><td>99.46</td><td>73.73</td><td>69.77</td><td>40.26</td></tr><tr><td>HuBERT + Spin2048 (for AP)</td><td>61.31</td><td>96.87</td><td>75.09</td><td>58.00</td><td>49.34</td></tr><tr><td>HuBERT + R-Spin32, AP40k</td><td>64.73</td><td>71.47</td><td>67.93</td><td>10.41</td><td>71.05</td></tr><tr><td> $\mathrm { W a v L M + R \mathrm { - } S p i n _ { 1 6 , A P 4 0 k } }$ </td><td>60.73</td><td>68.22</td><td>64.26</td><td>12.32</td><td>67.36</td></tr><tr><td>WavLM + R-Spin32, AP40k</td><td>65.12</td><td>73.76</td><td>69.17</td><td>13.28</td><td>71.33</td></tr><tr><td> $\mathrm { W a v L M + R \mathrm { - } S p i n _ { 6 4 , A P 4 0 k } }$ </td><td>63.02</td><td>73.63</td><td>67.91</td><td>16.83</td><td>69.08</td></tr><tr><td> $\mathrm { W a v L M + R – S p i n _ { 1 2 8 , A P 4 0 k } }$ </td><td>61.44</td><td>72.42</td><td>66.48</td><td>17.88</td><td>67.49</td></tr><tr><td> $\mathrm { W a v L M + R – S p i n } _ { 2 5 6 , \mathrm { A P 4 0 k } }$ </td><td>60.80</td><td>78.61</td><td>68.57</td><td>29.28</td><td>63.95</td></tr><tr><td> $\mathrm { W a v L M + R – S p i n } _ { 5 1 2 , \mathrm { A P 4 0 k } }$ </td><td>59.66</td><td>82.94</td><td>69.40</td><td>39.01</td><td>58.89</td></tr><tr><td> $\mathrm { W a v L M + R { \mathrm { - } } S p i n _ { 1 0 2 4 , A P 4 0 k } }$ </td><td>59.03</td><td>89.09</td><td>71.01</td><td>50.94</td><td>52.09</td></tr><tr><td> $\mathrm { W a v L M + R { \cdot } S p i n _ { 2 0 4 8 , A P 4 0 k } }$ </td><td>58.47</td><td>94.19</td><td>72.15</td><td>61.08</td><td>45.67</td></tr><tr><td colspan="6">Acoustic Pieces</td></tr><tr><td> $\mathrm { H u B E R T + S p i n _ { 2 0 4 8 } \ A P 5 k }$ </td><td>60.80</td><td>71.64</td><td>65.77</td><td>17.83</td><td>66.92</td></tr><tr><td>HuBERT + Spin2048 AP10k</td><td>61.37</td><td>68.51</td><td>64.74</td><td>11.64</td><td>67.97</td></tr><tr><td> $\mathrm { H u B E R T + S p i n _ { 2 0 4 8 } \ A P 2 0 k }$ </td><td>61.76</td><td>68.74</td><td>65.06</td><td>11.29</td><td>68.34</td></tr><tr><td> $\mathrm { H u B E R T + S p i n _ { 2 0 4 8 } \ A P 4 0 k }$ </td><td>62.10</td><td>68.54</td><td>65.16</td><td>10.37</td><td>68.65</td></tr></table>

Table 4: Unsupervised phoneme segmentation on TIMIT test set. OS and R-val respectively denote the oversegmentation rate and R-value (Räsänen et al., 2009). Oracle uniform is a segmentation method that splits speech into equal-length segments, given the ground truth number of phoneme boundaries. Unknown results are left blank.

## B.3 Unsupervised Phoneme Segmentation

This section inspects the phoneme segmentation capability of the proposed methods. As shown in Table 4, segmenting speech with Spin codebook or acoustic pieces is inferior to prior methods specifically designed for phoneme segmentation because no explicit constraints are added to encourage phoneme boundary detection. Still, some R-Spin discrete units like R-Spin32, AP40k surpass the oracle uniform baseline, indicating that the discrete unit boundaries are close to phoneme boundaries. The results align with the findings in Fig. 6b.

<table><tr><td>Model</td><td>Init</td><td>Updates</td><td>Batch Size (minutes)</td><td>Processed Speech (hours)</td><td>#GPUs Hours Model</td><td>GPU</td><td>Open</td></tr><tr><td>Self-supervised Pre-training (Clean Speech)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>wav2vec 2.0 (Baevski et al., 2020)</td><td></td><td>400k</td><td>96</td><td>640k</td><td>64</td><td>2458</td><td>√</td></tr><tr><td>HuBERT (Hsu et al., 2021a)</td><td></td><td>250k + 400k</td><td>47</td><td>505k</td><td>32</td><td>1976</td><td>√</td></tr><tr><td>data2vec (Baevski et al., 2022)</td><td></td><td>400k</td><td>63</td><td>420k</td><td>16</td><td></td><td>√</td></tr><tr><td>DinoSR (Liu et al., 2023)</td><td></td><td>400k</td><td>63</td><td>420k</td><td>16</td><td>2880</td><td>√</td></tr><tr><td colspan="8">Self-supervised Pre-training (Noisy Speech)</td></tr><tr><td>WavLM (Chen et al., 2022)</td><td></td><td>250k + 400k</td><td>187</td><td>1439k</td><td>32</td><td></td><td>√</td></tr><tr><td>wav2vec-Switch (Wang et al., 2022)</td><td></td><td>400k</td><td>96</td><td>640k</td><td>32</td><td></td><td>X</td></tr><tr><td>SPIRAL (Huang et al., 2022b)</td><td></td><td>200k</td><td>100</td><td>333k</td><td>16</td><td>499</td><td>√</td></tr><tr><td colspan="8">Domain-specific Self-supervision</td></tr><tr><td>ContentVec (Qian et al., 2022)</td><td>HuBERT</td><td>100k</td><td>46</td><td>76k</td><td>36</td><td>684</td><td>√</td></tr><tr><td>HuBERT-MGR (Huang et al., 2022a)</td><td>HuBERT</td><td>400k</td><td>12</td><td>78k</td><td>8</td><td>768</td><td>√</td></tr><tr><td>Robust data2vec (Zhu et al., 2023)</td><td>data2vec</td><td>100k</td><td>63</td><td>105k</td><td>16</td><td></td><td>√</td></tr><tr><td>deHuBERT (Ng et al., 2023)</td><td>HuBERT</td><td>250k</td><td></td><td></td><td></td><td></td><td>X</td></tr><tr><td>Spin2048 (Chang et al., 2023)</td><td>HuBERT</td><td>5k</td><td>43</td><td>0.4k</td><td>1</td><td>1</td><td>√</td></tr><tr><td colspan="8">This Paper</td></tr><tr><td>Robust data2vec (low budget)</td><td>data2vec</td><td>100k</td><td>6.3</td><td>10.4k</td><td>2</td><td>44</td><td>△</td></tr><tr><td>Spin2048 (for AP40k)</td><td>HuBERT</td><td>10k</td><td>43</td><td>7.1k</td><td>2</td><td>8</td><td>△</td></tr><tr><td>R-Spin32, AP40k</td><td>HuBERT</td><td>10k</td><td>6.4</td><td>1.1k</td><td>2</td><td>16</td><td> $\bigtriangleup$ </td></tr></table>

Table 5: SSL and DS costs of models with 95M parameters. The “Init" column shows the pre-trained models used for initialization. △ denotes models in this paper, which will be made publicly available in the near future.15 Note that duplicate input utterances by data augmentation are not included when calculating the hours of speech processed. The number of GPU hours required for training is roughly estimated, so the true values might differ slightly. The availability of the models listed was updated in March 2024. Unknown data are left blank

<table><tr><td>Task</td><td>Updates</td><td>Hours</td><td>GPU</td></tr><tr><td>(A.2) Spin</td><td>10k</td><td>4</td><td>A6000×2</td></tr><tr><td>(A.3) R-Spin</td><td>10k</td><td>8</td><td>A6000×2</td></tr><tr><td>(A.4) Robust data2vec</td><td>100k</td><td>22</td><td>A6000×2</td></tr><tr><td>(A.5) SUPERB PR</td><td>30k</td><td>10</td><td>2080 Ti</td></tr><tr><td>(A.6) SUPERB ASR</td><td>100k</td><td>20</td><td>A5000</td></tr><tr><td>(3.7.1) SUPERB SID</td><td>50k</td><td>4</td><td>A6000</td></tr></table>

Table 6: Computation resources used in the experiments.

## C Computation Resources

The costs of self-supervised pre-training and domain-specific self-supervision methods are shown in Table 5. The required computation resources for each training task in this paper are listed in Table 6. Note that all results in this paper are obtained with a single run.

## D t-SNE of Hidden Representations

We plot more t-SNE visualization of hidden representations in Figs. 11, 12, 13, 14, 15, and 16.

![](images/98a195f8831bdc8a9b319d1cc61aa3f9b75a59cb267c661d22ea1f5d824eb0ec.jpg)

![](images/7f8662a81b6173ceba91ba00b8bfa15fc0106d40ae565ab8f27658cd4efc65fe.jpg)  
(b) HuBERT + R-Spin Layer 12  
Figure 11: t-SNE visualization of three speakers with different English dialects (see Fig. 3 for details).

![](images/7925c13a1c2f00a34d0021fbb9061bccebd2bf016c3e9bd2ed7a8296b2c9ea5b.jpg)

![](images/46a0b4d6efadcef630df693bd74329fe8db213210ca330365a2c4418785b13e8.jpg)  
(b) HuBERT + R-Spin Layer 12  
Figure 12: t-SNE visualization of eight speakers with different English dialects (see Fig. 3 for details).

![](images/311950e0fba0c9e7b47d6b35d8b689838c79399950069df293e0a416abf8521c.jpg)  
Figure 13: t-SNE visualization of HuBERT representations of the same utterance spoken by three speakers (see Fig. 3 for details).

![](images/98695fb0b34b95e87adb96edc68aaddf0db78c5e891705413e922d345ed021cc.jpg)  
(k) Layer 11  
(1) Layer 12  
Figure 14: t-SNE visualization of HuBERT + R-Spin representations of the same utterance spoken by three speakers (see Fig. 3 for details).

![](images/f3d10b67ff170390b4ee6f0cb577fdbfa5abbc29e0d4e38f45d1400906a56e13.jpg)  
Figure 15: t-SNE visualization of HuBERT representations of the same utterance under different distortions (see Fig. 7 for details).

![](images/75a020455f7fb764c08b07acbbfcae73de50db5c0947657673b8a83355627ba9.jpg)  
Figure 16: t-SNE visualization of HuBERT + R-Spin representations of the same utterance under different distortions (see Fig. 7 for details).