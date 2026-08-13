# Subspace Representations for Soft Set Operations and Sentence Similarities

Yoichi Ishibashi<sup>1,2</sup>∗ Sho Yokoi<sup>3,4</sup> Katsuhito Sudoh<sup>1</sup> Satoshi Nakamura<sup>1</sup>

<sup>1</sup> Nara Institute of Science and Technology <sup>2</sup> Kyoto University

<sup>3</sup> Tohoku University <sup>4</sup> RIKEN

{ishibashi.yoichi.ir3, sudoh, s-nakamura}@is.naist.jp yokoi@tohoku.ac.jp

## Abstract

In the field of natural language processing (NLP), continuous vector representations are crucial for capturing the semantic meanings of individual words. Yet, when it comes to the representations of sets of words, the conventional vector-based approaches often struggle with expressiveness and lack the essential set operations such as union, intersection, and complement. Inspired by quantum logic, we realize the representation of word sets and corresponding set operations within pre-trained word embedding spaces. By grounding our approach in the linear subspaces, we enable efficient computation of various set operations and facilitate the soft computation of membership functions within continuous spaces. Moreover, we allow for the computation of the F-score directly within word vectors, thereby establishing a direct link to the assessment of sentence similarity. In experiments with widely-used pre-trained embeddings and benchmarks, we show that our subspace-based set operations consistently outperform vector-based ones in both sentence similarity and set retrieval tasks. 1

## 1 Introduction

Embedding-based word representations have become fundamental in the field of natural language processing (NLP). Models like word2vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014), along with recent Transformer-based architectures (Vaswani et al., 2017; Devlin et al., 2019), have underscored the significance of embeddings in capturing the complexities of linguistic semantics.

The importance of representing collections of words is pivotal in understanding concepts and relationships within language contexts (Zaheer et al., 2017; Zhelezniak et al., 2019). For instance, while words like “apple” and “orange” each carry their distinct meanings, together they represent the broader concept of fruits. Another example of important application is a sentence representation (Zaheer et al., 2017). The set of words in a sentence captures the overall meaning, allowing for computations such as text similarity (Agirre et al., 2012).

![](images/867375111e02169819ab002a21f41f3883e98b06ad5b974df1c0615d7ddbe197.jpg)

![](images/2a24cb7e358b5d191ba754fed79cbff286b5956517109c91aa8a203197d11730.jpg)

![](images/6aef39a15a211aa943b9d590d073e8fc01e518892bcdeb9d9691a480de919d2c.jpg)

![](images/2cb4370b7dc21828434b9f5839ee911afa3d128e8f357e5293199474812a561a.jpg)  
Figure 1: Superiority of subspace representations: Our subspace representation (blue) surpasses the traditional vector set representation (gray) in both text similarity and text concept set retrieval tasks.

Against this backdrop, our research recognizes the significance of applying set operations in NLP and explores a new approach. Set operations enable a richer representation of relationships between collections of words, leading to more accurate semantic analysis based on context. For example, employing set operations allows for a clearer understanding of shared semantic features and differences among word groups within a text. This directly benefits tasks like determining semantic similarity and expanding word sets.

In response to these challenges, our study introduces a novel methodology that exploits the principles of quantum logic (Birkhoff and Von Neumann, 1936), applied within embedding spaces to define set operations. Our proposed framework adopts a subspace-based approach for representing word sets, aiming to maintain the intricate semantic relationships within these sets. We represent a word set as a subspace which is spanned by pretrained embeddings. Additionally, it adheres to the foundational laws of set theory as delineated in the framework of quantum logic. This compliance ensures that our set operations, such as union, intersection, and complement, are not only mathematically robust but also linguistically meaningful when applied them in pre-trained embedding space.

We first introduce a subspace set representation along with basic operations ( , , and ). Subsequently, to highlight the usefulness of our proposed framework, we introduce two core set computations: text similarity and set membership. The empirical results consistently point towards the notable superiority of our approach; our straightforward approach of spanning subspaces with pretrained embedding sets enables a rich set representation, and we demonstrated its consistent performance enhancement in downstream tasks (Figure 1). Our research contributions include:

1. The introduction of continuous set representations and a framework for set operations, enabling more effective manipulation of word embedding sets (§4).

2. We propose SubspaceBERTScore, an extension of the embedding set-based text similarity method, BERTScore (Zhang et al., 2020). By simply transitioning from a vector set representation to a subspace, and incorporating a subspace-based indicator function, we observe a salient improvement in performance across all text similarity benchmarks (§5).

3. We apply subspace-based basic operations ( , , and ) to set expansion task and achive high performance (§6).

## 2 Preliminaries

To make the following discussion clear, we define several symbols. The sets of tokens in two sentences (A and B) are denoted as $A \ =$ $\{ a _ { 1 } , a _ { 2 } , \ldots \} , B = \{ b _ { 1 } , b _ { 2 } , \ldots \}$ respectively. The sets of contextualized token vectors are denoted as $\mathbf { A } = \{ \pmb { a } _ { 1 } , \pmb { a } _ { 2 } , \ldots \} , \mathbf { B } = \{ \pmb { b } _ { 1 } , \pmb { b } _ { 2 } , \ldots \}$ , where a and b are token vectors generated by the pretrained embedding model such as BERT. The subspace spanned by A is denoted as $\mathbb { S } _ { A } ~ =$ span $( \pmb { a } _ { 1 } , \pmb { a } _ { 2 } , \dots )$ . Note that the bases of the subspace is orthonormalized.

## 3 Symbolic Set Operations

We first formulate various set operations in a pretrained embedding space. Among many types of operations for practical NLP applications, this work focuses on set similarity:

$$
\begin{array} { r l } & { A = \{ A , b o y , w a l k s , i n , t h i s , p a r k \} , } \\ & { B = \{ T h e , k i d , r u n s , i n , t h e , s q u a r e \} , } \\ & { \mathrm { S i m i l a r i t y } ( A , B ) , } \end{array}\tag{1}
$$

set membership ( ) and basic operations ( , ):

$$
\begin{array} { r l } & { C o l o r = \{ r e d , b l u e , g r e e n , o r a n g e , \ldots \} , } \\ & { F r u i t = \{ a p p l e , o r a n g e , p e a c h , \ldots \} , } \\ & { o r a n g e \in C o l o r \cap F r u i t . } \end{array}\tag{2}
$$

For this purpose, we need following representations on a pre-trained embedding space<sup>2</sup>:

An element and a set of elements The representations of an element and a set of elements are the most basic ones. To exploit word embeddings, we represent a word (e.g., orange) as an element and a group of words (e.g., red, blue, green, orange, . . . ) as a word set.

Quantification of set membership (indicator function) Membership denotes a relation in which word w is an element of set A, i.e., w A. We quantify it based on vector representations. Although the membership is typically a binary decision identical to that in a symbolic space, it can also be measured by the degree of closeness in a continuous vector space. Membership can be computed as an indicator function. The indicator function $\mathbb { 1 } _ { \mathrm { s e t } }$ quantifies whether the word w is included (1) or not (0) in the set in a discrete manner:

$$
\mathbb { 1 } _ { \mathrm { s e t } } [ w \in A ] = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } w \in A , } \\ { 0 } & { { \mathrm { i f ~ } } w \notin A . } \end{array} \right. }\tag{3}
$$

Similarity between discrete symbol sets Set similarity, such as recall and precision, is an essential operation when calculating the similarity of texts. Despite its simplicity, the word overlapbased sentence similarity serves as a remarkably effective approximation and has found widespread practical application, as evidenced by numerous studies(Bojar et al., 2018; Zhang et al., 2020; Cer et al., 2017; Zhelezniak et al., 2019). They stand out as excellent similarity metrics based on embeddings. BERTScore (Zhang et al., 2020), which utilizes embeddings for its computation, is grounded in recall and precision <sup>3</sup>. The typical computations for recall (R) and precision (P) are as follows<sup>4</sup>:

$$
R = \frac { 1 } { | A | } \sum _ { a _ { i } \in A } \mathbb { 1 } _ { \mathrm { s e t } } [ a _ { i } \in B ] ,\tag{4}
$$

$$
P = \frac { 1 } { | B | } \sum _ { b _ { i } \in B } \mathbb { 1 } _ { \mathrm { s e t } } [ b _ { i } \in A ] ,\tag{5}
$$

Basic set operations We need three basic set operations: intersection $( A \ \cap \ B )$ , union $( A \cup B )$ and complement $( { \overline { { A } } } )$ . They allow us to represent various sets using such different combinations as Color Fruit.

## 4 Subspace-based Set Representations

We propose the representations of a word set and set operations based on quantum logic (Birkhoff and Von Neumann, 1936). They hold advantages of geometric properties in an embedding space, and the set operations are guaranteed to hold for the laws of a set defined in quantum logic.

## 4.1 Quantum logic

While word embedding represents a word’s meaning as a vector in linear space, quantum mechanics similarly represents a quantum state as a vector in linear space. These two intuitively different fields are very close to each other in terms of the representation and the operation of information.

Quantum logic (Birkhoff and Von Neumann, 1936) theory describes quantum mechanical phenomena. Intuitively, it is a framework for set operations in a vector space. In quantum logic, a set of vectors is represented as a linear subspace in a Hilbert space, and such set operations as union, intersection, and complement are defined as operations on subspaces. Quantum logic, which employs a complete orthomodular lattice as its system of truth values, guarantees to hold various set operations, such as De Morgan’s laws ${ \overline { { ( A \cap B ) } } } = { \overline { { A } } } \cup { \overline { { B } } }$ and ${ \overline { { ( A \cup B ) } } } = { \overline { { A } } } \cap { \overline { { B } } } .$ , idempotent law: $A \cap A = A ,$ and double complement: ${ \overline { { \overline { { A } } } } } = A$

Algorithm 1 Computing basis of a subspace   
Input: $\{ \pmb { v } ^ { ( 1 ) } , \dots , \pmb { v } ^ { ( k ) } \} \subseteq \mathbb { R } ^ { 1 \times d } ;$ Word embed  
dings to span subspace $\mathbb { S } _ { A }$   
Output: $\mathbf { S } _ { \mathrm { A } } ^ { \mathrm { ~ - ~ } } \in \mathbb { R } ^ { r \times \bar { d } } ;$ : Bases of $\mathbb { S } _ { A }$   
$\bar { \mathbf { A } ^ { \prime } } \in \mathbb { R } ^ { k \times d }  \operatorname { s T a C K \_ R O W S } ( \pmb { v } ^ { ( 1 ) } , \dots , \pmb { v } ^ { ( k ) } )$   
$\mathbf { S } _ { \mathrm { A } } \in \mathbb { R } ^ { r \times d }  ( \mathrm { O R T H O \_ N O R M A L } ( \mathbf { A } ^ { \top } ) ) ^ { \top } \quad \triangleright$   
Orthonormalize the bases. r is the rank of A   
return $\mathbf { S } _ { \mathrm { A } }$

## 4.2 Set Operations in an Embedding Space

The representations of an element, a set, and such set operations as union, intersection, and complement in quantum logic can be applied directly in a word embedding space because it is a Euclidean space and therefore also a Hilbert space. However, since set similarity and set membership for a word embedding space are still missing in quantum logic, we propose a novel formulation of those operations using subspace-based representations, which is consistent with quantum logic. The correspondence between symbolic and subspace-based set operations is shown in Table 1.

Set and elements Let $\mathbb { R } ^ { n }$ be a n-dimensional embedding space (Euclidean space), let $A \ =$ $\{ w _ { 1 } , w _ { 2 } , . . . \}$ be a set of words, and let $\pmb { v } _ { w } \in \mathbb { R } ^ { n }$ be a word (token) vector corresponding to w. As discussed in §3, we first formulate the representation of a word and a word set. In quantum logic, an element is represented by a vector, and a set is represented by a subspace spanned by the vectors corresponding to its elements. Here we assume an element, i.e., word $w ,$ is represented by vector ${ \pmb v } _ { w } .$ and a word set is represented by linear subspace $\mathbb { S } _ { A } \subset \mathbb { R } ^ { n }$ spanned by word vectors:

$$
\mathbb { S } _ { A } : = \operatorname { s p a n } ( \mathbf { A } ) : = \operatorname { s p a n } ( \pmb { a } _ { 1 } , \pmb { a } _ { 2 } , . . . ) .\tag{6}
$$

Hereinafter we simply refer to linear subspace as subspace. Algorithm 1 is the pseudocode for computing the basis of the subspace.

Basic set operations The complement of set A, denoted by A, is represented by the orthogonal complement of subspace $\mathbb { S } _ { A }$ :

$$
\mathbb { S } _ { \overline { { A } } } : = ( \mathbb { S } _ { A } ) ^ { \perp } = \{ \pmb { v } \mid \exists \pmb { a } \in \mathbb { S } _ { A } , \pmb { v } \cdot \pmb { a } = 0 \} .\tag{7}
$$

The union of two sets, A and B, denoted by $A \cup B .$ , is represented by the sum space of two

<table><tr><td colspan="2">Symbolic Set Representation</td><td colspan="2">Subspace-based Set Representation</td></tr><tr><td>Element king</td><td><img src="images/9ef7f9fe17533de8ca6c897fe56fe92b9c74e5118363925732964474887fde53.jpg"/></td><td>Vector  ${ \pmb v } _ { k i n g }$ </td><td></td></tr><tr><td>Set  $M a l e = \left\{ k i n g , m a n , \ldots \right\}$ </td><td></td><td>Subspace  $\mathbb { S } _ { M a l e } = \operatorname { s p a n } ( \pmb { v } _ { k i n g } , \pmb { v } _ { m a n } , \dots )$ </td><td><img src="images/c3b192f2348bad271d6826034fe80399b3ba954b84400750f51710fb8d054d40.jpg"/></td></tr><tr><td>Complement Male</td><td><img src="images/7b43b62e07853ef123ecde7207b4f46b816d70d088ec0ac63a5992dae65ada56.jpg"/></td><td>Orthogonal complement  $( \mathbb { S } _ { M a l e } ) ^ { \perp }$ </td><td><img src="images/3d5601f7623aceff813035da47482c0a80d60e913f9d865e74453a9bb6ceae7b.jpg"/></td></tr><tr><td>Union Male ∪ Female</td><td><img src="images/eb5e4a468c2cb34180bd7467ba37eac522eddb071d241609c67797e368efcfed.jpg"/></td><td>Sum space  $\mathbb { S } _ { M a l e } + \mathbb { S } _ { F e m a l e }$ </td><td><img src="images/bd8981dd083205f7fbaad5d05a1c468c0f8fa4edfd44079e67fbc5639c1bb0d1.jpg"/></td></tr><tr><td>Intersection Color ∩ Fruit</td><td><img src="images/6a5ba20460b15c8086b00216b0155a7020e98bb71d15f013992da5c32451109a.jpg"/></td><td>Intersection  $\mathbb { S } _ { C o l o r } \cap \mathbb { S } _ { F r u i t }$ </td><td><img src="images/42aefc6250fd7e480bdd1f7098c2b407958eb7861e52ef4ea9b239ca4cd67083.jpg"/></td></tr><tr><td>Set membership  $\mathbb { 1 } _ { \mathrm { s e t } } [ b o y \in M a l e ]$ </td><td><img src="images/89c13b7b561a845932f836dd95cccff7d456ee57fbe19ba85bcca05bddd14346.jpg"/></td><td>Subspace indicator function  $\mathbb { 1 } _ { \mathrm { s u b s p a c e } } ( \pmb { v } _ { b o y } , \mathbb { S } _ { M a l e } )$ </td><td><img src="images/9d6d899eb02977579432ed79a2becc07c08460c9912d9ac1166d1de2a1f8f012.jpg"/></td></tr><tr><td>Set similarity</td><td></td><td>SubspaceBERTScore</td><td></td></tr><tr><td>Recall</td><td>0</td><td> $R _ { \mathrm { s u b s p a c e } }$ </td><td> $\frac { 6 4 } { 1 0 }$ </td></tr><tr><td>Precision</td><td></td><td> $P _ { \mathrm { s u b s p a c e } }$ </td><td> $\frac { 1 4 } { 4 }$ </td></tr><tr><td>F-score</td><td> $2 \frac { \frac { \textcircled { \times } } { \textcircled { \times } } \times \frac { \textcircled { \times } } { \textcircled { \times } } } { \frac { \textcircled { \times } } { \textcircled { \times } } + \frac { \textcircled { \times } } { \textcircled { \times } } }$ </td><td> $F _ { \mathrm { s u b s p a c e } }$  2-</td><td> $\frac { a } { \textcircled { 1 } } \times \frac { a } { \textcircled { 1 } }$   $\frac { a } { \textcircled { 1 } } + \frac { a } { \textcircled { 1 } }$ </td></tr></table>

Table 1: Correspondence between symbolic set representations and subspace-based set representations: We demonstrate that union, intersection, and complement, which are formulated in quantum logic, and our new formulations of set membership and word set similarity hold in pre-trained word embedding space.

subspaces, $\mathbb { S } _ { A }$ and $\mathbb { S } _ { B } \colon$

$$
\mathbb { S } _ { A \cup B } : = \mathbb { S } _ { A } + \mathbb { S } _ { B } = \{ \pmb { a } + \pmb { b } | \pmb { a } \in \mathbb { S } _ { A } , \pmb { b } \in \mathbb { S } _ { B } \} . ( 8 )
$$

The intersection of two sets, A and B, denoted by A B, is represented by the intersection of two subspaces, $\mathbb { S } _ { A }$ and $\mathbb { S } _ { B }$ :

$$
\mathbb { S } _ { A \cap B } : = \mathbb { S } _ { A } \cap \mathbb { S } _ { B } = \{ \pmb { v } \mid \pmb { v } \in \mathbb { S } _ { A } , \pmb { v } \in \mathbb { S } _ { B } \} .\tag{9}
$$

The basis of the intersection can be computed based on singular value decomposition (SVD). The bases are the vectors shared by the two subspaces.

Hard membership The set membership in the embedding space (e.g., boy $\in M a l e )$ can be represented by the inclusion of a vector into a subspace

(e.g., ${ \pmb v } _ { b o y } \in \mathbb { S } _ { M a l e } )$ and given by the following indicator function:

$$
\mathbb { 1 } _ { \mathrm { h a r d } } ( \pmb { v } , \mathbb { S } _ { \cal A } ) : = \left\{ \begin{array} { l l } { 1 } & { ( \pmb { v } \in \mathbb { S } _ { \cal A } ) , } \\ { 0 } & { ( \pmb { v } \notin \mathbb { S } _ { \cal A } ) . } \end{array} \right.\tag{10}
$$

However, this binary decision fails to exploit the geometric properties of the word embedding space regarding semantic similarity. Suppose we quantify the degree of membership of word boy for word set Male consisting of many masculine nouns other than boy. Even if ${ \pmb v } _ { b o y }$ is located very close to $\mathbb { S } _ { M a l e }$ due to its semantic similarity to masculine nouns, $\mathbb { 1 } _ { \mathrm { h a r d } } ( \pmb { v } _ { b o y } , \mathbb { S } _ { M a l e } )$ must return 0 because ${ \pmb v } _ { b o y }$ must not be located exactly on subspace $\mathbb { S } _ { M a l e }$ defined by Male. It must return 1 based on the masculine property of word boy. Such hard membership defined by Eq. (10) is incompatible with an embedding space.

Soft membership: Subspace indicator function Instead, we define another membership function called subspace indicator function $\mathbb { 1 } _ { \mathrm { s u b s p a c e } }$ that returns continuous values from 0 to 1 depending on the following minimum angle between vector ${ \pmb v } _ { w }$ and subspace $\mathbb { S } _ { A }$ (the first canonical angle):

$$
\mathbb { 1 } _ { \mathrm { s u b s p a c e } } ( \pmb { v } , \mathbb { S } _ { A } ) : = \operatorname* { m a x } \bigg \{ \frac { | \pmb { a } \cdot \pmb { v } | } { \| \pmb { a } \| \| \pmb { v } \| } \bigg | \pmb { a } \in \mathbb { S } _ { A } \bigg \} .\tag{11}
$$

This captures the degree of membership between a word and a word set, represented by the angle between a word vector and a subspace. It is a natural extension of $\mathbb { 1 } _ { \mathrm { h a r d } }$ , i.e., 1<sub>subspace</sub> returns 1 when $\pmb { v } _ { w } \in \mathbb { S } _ { A }$ and 0 when $\pmb { v } _ { w } \in \mathbb { S } _ { \overline { { A } } }$

The key distinction of our subspace indicator function approach lies in its ability to leverage the comprehensive information encapsulated within pre-trained word embedding space. The subspace indicator function does not simply find the nearest individual word from the set. Instead, we consider the closeness of the query word to the entire set as a whole, by projecting the query word into the subspace spanned by the pre-trained embeddings (as illustrated in the figure of the subspace indicator function function in Table 1). This way, we account not just for the individual word similarities, but also for the overall semantic coherence of the word set. The detailed process for computing this subspace indicator function can be found in Algorithm 2.

## 4.3 Set Similarity

Limitation of symbolic set similarities Suppose we quantify the set similarity between $A = \{ A$ , boy, walks, in, this, park and $B = \{ T h e , k i d , r u n s , i n$ the, square , which represent semantically similar sentences. The challenge with traditional symbolic set similarities, such as recall, is that they primarily rely on the exact overlap of words between the sets. These semantically similar sentences share only one word in between A and $B \colon | A \cap B | = 1$ . To address this shortcoming, it is essential to compute vector-based set similarity, such as BERTScore.

Three types of similarity in BERTScore BERTScore is a method that uses embeddings to approximately calculate $R , P ;$ , and the F-score:

$$
R _ { \mathrm { B E R T } } = \frac { 1 } { | A | } \sum _ { \pmb { a } _ { i } \in \mathbf { A } } \mathbb { 1 } _ { \mathrm { v e c t o r s } } ( \pmb { a } _ { i } , \mathbf { B } ) ,\tag{12}
$$

$$
P _ { \mathrm { B E R T } } = \frac { 1 } { | B | } \sum _ { b _ { i } \in \mathbf { B } } \mathbb { 1 } _ { \mathrm { v e c t o r s } } ( b _ { i } , \mathbf { A } ) ,\tag{13}
$$

$$
F _ { \mathrm { B E R T } } = 2 \frac { P _ { \mathrm { B E R T } } \cdot R _ { \mathrm { B E R T } } } { P _ { \mathrm { B E R T } } + R _ { \mathrm { B E R T } } } ,\tag{14}
$$

where a sentence is represented as a set of token vectors A and B. $\mathbb { 1 } _ { \mathrm { v e c t o r s } }$ is the indicator function for vector sets. Intuitively, this indicator function represents the calculation of selecting one token from the sentence and serves as a flexible extension of the binary indicator function $\mathbb { 1 } _ { \mathrm { s e t } }$ . It returns a continuous similarity score between 1 and 1 for a token, depending on its similarity with the tokens in the sentence. Specifically, $\mathbb { 1 } _ { \mathrm { v e c t o r s } } ( \mathbf { \boldsymbol { a } } _ { i } , \mathbf { \boldsymbol { B } } )$ quantifies to what extent the i-th token vector $\mathbf { a } _ { i }$ in sentence A is semantically included in sentence B by taking the maximum cosine similarity between $\mathbf { a } _ { i }$ and the token vectors in $B { : }$

$$
\mathbb { 1 } _ { \mathrm { v e c t o r s } } ( \pmb { a } _ { i } , \mathbf { B } ) = \operatorname* { m a x } _ { { b _ { j } \in \mathbf { B } } } \cos ( \pmb { a } _ { i } , { b } _ { j } ) \in [ - 1 , 1 ] ,\tag{15}
$$

where cos $( a _ { i } , b _ { j } )$ is the cosine similarity between $\mathbf { \alpha } _ { \mathbf { \alpha } } \mathbf { a } _ { i }$ and $b _ { j }$ .

Limitations of BERTScore’s Indicator Function The indicator function $\mathbb { 1 } _ { \mathrm { v e c t o r } }$ lies at the heart of BERTScore, playing a crucial role in the computation of P<sub>BERT</sub>, R<sub>BERT</sub>, and $F _ { \mathrm { B E R T } }$ . However, a critical limitation arises from BERTScore’s reliance on maximum cosine similarity for its indicator function, which severely hinders its ability to capture the rich and nuanced meanings conveyed by a sentence. Figure 2 starkly illustrates this limitation through the visualization of BERTScore’s $\mathbb { 1 } _ { \mathrm { v e c t o r s } }$ calculation. Consider the sentence We are the king and queen, which evokes a broader, more abstract concept of royalty through the cooccurrence of king and queen. When seeking an alignment for royalty, BERTScore’s indicator function heavily favors the token with the highest cosine similarity — in this case, king. This approach leads to a severe alignment bias towards the meaning of a single word, while the complex and implicit meanings conveyed by the entire sentence are overlooked.

SubspaceBERTScore To overcome the limitations of BERTScore regarding the expressiveness of its indicator function, we propose Subspace-BERTScore. This method extends BERTScore by employing the concept of subspace-based sentence representation and indicator function.

![](images/b83ff147cd3b54827f0ebd530bf573de78719717ab725ca6cb9b0ce010f327d8.jpg)  
Figure 2: Comparison between the proposed SubspaceBERTScore and BERTScore. We visualize the alignment process between the word royalty and the words in the sentence $B .$ SubspaceBERTScore represents B as the subspace $\mathbb { S } _ { B }$ and calculates the similarity (canonical angle) between $\mathbb { S } _ { B }$ and the royalty vector $( \pmb { a } _ { 4 } )$ . Our approach provides a “softer” alignment, capturing the overall semantic context of the sentence. On the other hand, BERTScore adopts a “harder” alignment strategy, selecting only the word from the sentence with the maximum cosine similarity.

Extension of P, R, F with Subspaces Based on the above discussions, we propose Subspace-BERTScore, which calculates BERTScore’s R, P, F using the subspace representation of sentences and the subspace indicator function:

$$
R _ { \mathrm { s u b s p a c e } } = { \frac { 1 } { | A | } } \sum _ { \mathbf { a } _ { i } \in A } \mathbb { 1 } _ { \mathrm { s u b s p a c e } } ( \mathbf { a } _ { i } , \mathbb { S } _ { B } ) ,\tag{16}
$$

$$
P _ { \mathrm { s u b s p a c e } } = \frac { 1 } { | B | } \sum _ { b _ { i } \in B } \mathbb { 1 } _ { \mathrm { s u b s p a c e } } ( \boldsymbol { b } _ { i } , \mathbb { S } _ { A } ) ,\tag{17}
$$

$$
F _ { \mathrm { s u b s p a c e } } = 2 { \frac { P _ { \mathrm { s u b s p a c e } } \cdot R _ { \mathrm { s u b s p a c e } } } { P _ { \mathrm { s u b s p a c e } } + R _ { \mathrm { s u b s p a c e } } } } ,\tag{18}
$$

where $R _ { \mathrm { s u b s p a c e } } , P _ { \mathrm { s u b s p a c e } }$ , and $F _ { \mathrm { s u b s p a c e } }$ are the final evaluation measures of SubspaceBERTScore.

Weighting by Importance Previous study (Banerjee and Lavie, 2005; Vedantam et al., 2015) has shown that infrequently occurring words play a more important role in sentence similarity than general words. We apply importance weightings to our method as follows:

$$
R _ { \mathrm { s u b s p a c e } } = \frac { \sum _ { { \pmb a } _ { i } \in A } \mathrm { w e i g h t } ( { \pmb a } _ { i } ) \mathbb { 1 } _ { \mathrm { s u b s p a c e } } ( { \pmb a } _ { i } , \mathbb { S } _ { B } ) } { \sum _ { { \pmb a } _ { i } \in A } \mathrm { w e i g h t } ( { \pmb a } _ { i } ) } ,\tag{19}
$$

$$
P _ { \mathrm { s u b s p a c e } } = \frac { \sum _ { b _ { i } \in \cal B } \mathrm { w e i g h t } ( b _ { i } ) \mathbb { 1 } _ { \mathrm { s u b s p a c e } } ( b _ { i } , \mathbb { S } _ { \cal A } ) } { \sum _ { b _ { i } \in \cal B } \mathrm { w e i g h t } ( b _ { i } ) } ,\tag{20}
$$

where weight( ) is a weighting function. We use the L2 norm of the vector (Yokoi et al., 2020; Oyama et al., 2023).

Algorithm 2 Subspace indicator function   
$\mathbb { 1 } _ { \mathrm { s u b s p a c e } } ( \pmb { v } _ { w } , \mathbb { S } _ { A } )$   
Input: $\mathbf { S } _ { \mathrm { A } } \in \mathbb { R } ^ { k \times d } \colon$ Bases of $\mathbb { S } _ { A }$   
Input: $\pmb { v } _ { w } \in \mathbb { R } ^ { 1 \times d } ;$ A word vector   
Output: $\sigma \in \mathbb { R } :$ Membership degree   
if $k = 0$ then   
return 0   
else   
$\begin{array} { r } { \widetilde { \pmb { v } } _ { w }  \frac { \pmb { v } _ { w } } { \vert \vert \pmb { v } _ { w } \vert \vert } \in \mathbb { R } ^ { 1 \times d } } \end{array}$   
${ \bf U } ^ { \top } \in \mathbb { R } ^ { k \times k } , \sigma \in \mathbb { R } , { \bf V } \in \mathbb { R }  $   
$\mathrm { S V D } ( \mathbf { S } _ { \mathrm { A } } \widetilde { \pmb { v } } _ { w } ^ { \top } )$   
ereturn $\sigma \in \mathbb { R }$ ▷   
The output of $\mathbb { 1 } _ { \mathrm { s u b s p a c e } } ( \pmb { v } _ { w } , \mathbb { S } _ { A } )$ is always non  
negative because it is a singular value   
end if

## 5 Semantic Textual Similarity Task

In this section, we examine the effectiveness of SubspaceBERTScore through the semantic textual similarity task (STS; Agirre et al., 2012).

Task An STS task calculates the similarity between two sentences. For the STS evaluation protocol, we follow Gao et al. (2021). Its evaluation is based on the correlation between the similarity calculated by the model and corresponding human judgments. We used datasets from the SemEval shared task 2012-2016 (Agirre et al., 2012, 2013, 2014, 2015, 2016), STS benchmark (STS-B; Cer et al., 2017), and SICK-Relatedness (SICK-R; Marelli et al., 2014). We used Spearman’s $\rho .$

Embeddings We used 768-dimensional $\mathbf { B E R T _ { b a s e } } ^ { 5 }$ (Devlin et al., 2019), which was pre-trained with BookCorpus and Wikipedia. We used hidden states in the last layer.

<table><tr><td>Method</td><td>Metric</td><td>Weighting</td><td>STS12</td><td>STS13</td><td>STS14</td><td>STS15</td><td>STS16</td><td>STS-B</td><td>SICK-R</td><td>Avg.</td></tr><tr><td>CLS-cos</td><td></td><td></td><td>.215</td><td>.321</td><td>.213</td><td>.379</td><td>.442</td><td>.203</td><td>.424</td><td>.314</td></tr><tr><td>Avg-cos</td><td></td><td></td><td>.309</td><td>.599★</td><td>.477</td><td>.603</td><td>.637</td><td>.473</td><td>.582★</td><td>.526</td></tr><tr><td>WMD</td><td></td><td></td><td>.238</td><td>.443</td><td>.389</td><td>.531</td><td>.532</td><td>.384</td><td>.509</td><td>.432</td></tr><tr><td>WRD</td><td></td><td></td><td>.241</td><td>.502</td><td>.410</td><td>.573</td><td>.573</td><td>.421</td><td>.527</td><td>.464</td></tr><tr><td>DynaMax</td><td></td><td></td><td>.322</td><td>.518</td><td>.432</td><td>.616</td><td>.639</td><td>.452</td><td>.560</td><td>.506</td></tr><tr><td rowspan="3">BERTScore</td><td>F</td><td></td><td>.312</td><td>.546</td><td>.450</td><td>.602</td><td>.636</td><td>.446</td><td>.553</td><td>.506</td></tr><tr><td>P</td><td></td><td>.261</td><td>.532</td><td>.462</td><td>.576</td><td>.622</td><td>.443</td><td>.559</td><td>.494</td></tr><tr><td>R</td><td></td><td>.350</td><td>.527</td><td>.416</td><td>.602</td><td>.623</td><td>.430</td><td>.522</td><td>.496</td></tr><tr><td rowspan="3">SubspaceBERTScore</td><td>F</td><td></td><td>.335</td><td>.573</td><td>.476</td><td>.610</td><td>.650</td><td>.479</td><td>.562</td><td>.526</td></tr><tr><td>P</td><td></td><td>.282</td><td>.550</td><td>.488</td><td>.580</td><td>.630</td><td>.475</td><td>.568</td><td>.511</td></tr><tr><td>R</td><td></td><td>.369★</td><td>.552</td><td>.436</td><td>.611</td><td>.639</td><td>.462</td><td>.530</td><td>.514</td></tr><tr><td rowspan="3">BERTScore</td><td>F</td><td>L2</td><td>.321</td><td>.540</td><td>.452</td><td>.613</td><td>.640</td><td>.454</td><td>.558</td><td>.511</td></tr><tr><td>P</td><td>L2</td><td>.274</td><td>.529</td><td>.468</td><td>.589</td><td>.627</td><td>.450</td><td>.565</td><td>.500</td></tr><tr><td>R</td><td>L2</td><td>.348</td><td>.520</td><td>.414</td><td>.610</td><td>.624</td><td>.437</td><td>.524</td><td>.497</td></tr><tr><td rowspan="3">SubspaceBERTScore</td><td>F</td><td>L2</td><td>.342</td><td>.568</td><td>.477</td><td>.621</td><td>.653★</td><td>.486*</td><td>.568</td><td>.531*</td></tr><tr><td> $P$ </td><td>L2</td><td>.292</td><td>.547</td><td>.492*</td><td>.592</td><td>.634</td><td>.479</td><td>.574</td><td>.516</td></tr><tr><td>R</td><td>L2</td><td>.367</td><td>.544</td><td>.434</td><td>.620*</td><td>.640</td><td>.468</td><td>.532</td><td>.515</td></tr></table>

Table 2: A comprehensive comparison of similarity metrics in the STS task. The scores are Spearman’s ρ. The methods with the highest values, using the same pre-trained embeddings, are highlighted in ⋆. Scores that showed improvement from BERTScore are denoted in bold.

Baselines We compared our method Subspace-BERTScore with other baseline similarity metrics. The baselines included Avg-cos (Arora et al., 2017), the cosine similarity between the averaged vectors, CLS-cos (Gao et al., 2021), the cosine similarity between the [CLS] representations of the pre-trained language model, DynaMax (Zhelezniak et al., 2019), a set similarity based on fuzzy sets, Word Mover’s Distance (WMD; Kusner et al., 2015), a metric based on optimal transport cost, and Word Rotator’s Distance (WMD; Yokoi et al., 2020), an optimal transport-based metric that improves WMD.

Main results The results are shown in Table 2. In comparison to BERTScore, our method achieves superior correlation with human judgments across all three key metrics: F-score, precision, and recall. An important observation is that the performance consistently improves by subspace representation of the set. The results suggest that simply replacing the representation of embedding sets and the indicator function with subspace-based alternatives significantly enhances our ability to capture and express the depth of linguistic semantics.

We also conduct an experiment using L2 norm as a weighting factor for the indicator function. This method has previously been proven effective in the STS task (Yokoi et al., 2020). We see that both our proposed method and BERTScore improve their performance underlining the effectiveness of this weighting approach in both cases. Notably, our proposed method continued to outperform BERTScore even when L2 norm was used for weighting.

Our similarity also outperforms the fuzzy-set based similarity of DynaMax. This result suggests that the proposed subspace-based approach represents a set and set operations better than the fuzzy set-based approach in embedding space.

## 6 Text Concept Set Retrieval Task

In this section, we evaluate the capability of our proposed set operations ( , , and ) in effectively representing word sets.

Task We evaluate our set operations by the set expansion task introduced by Zaheer et al. (2017). In this task, the model is given a set of words that share a common concept or theme. The objective is to expand this set by retrieving relevant words from a vocabulary that fit the same concept. For instance, if the initial set includes words like “apple”, “banana”, and “peach”, the task would be to identify and add other fruit names (e.g., “orange”) to this set. For the evaluation, we follow Zaheer et al. (2017). We report recall (R@k) and Median, that indicate whether the words in the test set can be ranked higher.

Embeddings We used the most standard pretrained word embeddings in all of our experiments:

300-dimensional $\mathrm { G l o V e } ^ { 6 }$ (Pennington et al., 2014), which was pre-trained with Common Crawl, and 300-dimensional word2vec<sup>7</sup> (Mikolov et al., 2013), which was pre-trained with Google News.

Set Expansion with Subspace Indicator Function To illustrate our subspace-based set expansion method (Subspace Set), we consider a set of fruit-related words. For example, let’s take $S _ { \mathrm { f r u i t } } ~ = ~ \{ a p p l e , b a n a n a , . . . \}$ This set is divided into two subsets: a ’span’ subset used for creating a subspace representation, and a ’test’ subset for evaluation. Let’s assume orange $\notin \ S _ { \mathrm { f r u i t \_ s p a n } }$ is a target word for testing. (1) From the ’span’ subset, we generate a subspace: $\mathbb { S } _ { F r u i t } \ = \ \mathrm { s p a n } ( S _ { \mathrm { f r u i t \_ s p a n } } )$ For instance, if $S _ { \mathrm { f r u i t \_ s p a n } } = \{ a p p l e , b a n a n a , \ldots \}$ , then $\mathbb { S } _ { F r u i t } = \mathrm { s p a n } ( \pmb { v } _ { a p p l e } , \pmb { v } _ { b a n a n a } , \dots )$ . (2) We define a subspace indicator function, which computes the degree to which a word vector belongs to the subspace. For a word w, the membership score is calculated as $\mathrm { s c o r e } = \mathbb { 1 } _ { \mathrm { s u b s p a c e } } ( \pmb { v } _ { o r a n g e } , \mathbb { S } _ { F r u i t } )$ This score reflects the extent to which w aligns with the semantic characteristics of the subspace. This method effectively expand the set $S _ { \mathrm { f r u i t } }$ by identifying words that share semantic properties with the subspace defined by the initial set.

Baselines We compared several baselines, which don’t require training on word sets, to our method. Random just selects words randomly from the dataset’s vocabulary. A simple unsupervised baseline with word embeddings uses the nearest neighbors in the embedding space (Near)<sup>8</sup>. We also compare a method based on fuzzy sets (Fuzzy set; Zhelezniak et al., 2019) with our method. Similar to our method, their method is designed to exploit both the flexibility of word vectors and rich set operations. Fuzzy Set represents word set A by max-pooled word vectors $\pmb { s } = \operatorname* { m a x } _ { \pmb { w } \in A } \pmb { v } _ { w }$ . One major difference from our method is that Fuzzy set represents a set of word vectors by compressing them into a vector of fixed dimensions. Although the Text Concept Set Retrieval task requires computing the degree of a word’s membership for a word set, their method does not provide it. We instead used cosine similarity $\cos ( \boldsymbol { v } _ { w } , \boldsymbol { s } )$ between word vector ${ \pmb v } _ { w }$ of word $w \in V$ and s as the degree of membership to apply fuzzy sets to the task.

<table><tr><td rowspan="2">Dataset (# Set)</td><td colspan="5">Example</td></tr><tr><td>Set</td><td></td><td>Words (set elements)</td><td></td><td></td></tr><tr><td> $\mathbf { D } ^ { \mathrm { S e t } } \left( 1 0 0 \right)$ </td><td> $S _ { 3 }$ </td><td>daily</td><td>news</td><td>paper</td><td></td></tr><tr><td rowspan="3"> $\mathbf { D } ^ { \mathrm { U n i o n } } \left( 1 0 0 \right)$ </td><td> $S _ { 1 2 }$ </td><td>rider bike</td><td></td><td>bicycle</td><td></td></tr><tr><td> $S _ { 5 1 }$ </td><td></td><td>island fishing sea</td><td></td><td></td></tr><tr><td> $S _ { 1 2 } \cup S _ { 5 1 }$ </td><td></td><td></td><td>races cycling islands</td><td></td></tr><tr><td rowspan="3">DIntersect (100)</td><td> $S _ { 9 }$ </td><td>tour</td><td>open</td><td>golf</td><td></td></tr><tr><td> $S _ { 7 2 }$ </td><td></td><td></td><td>poker casino gambling . ..</td><td></td></tr><tr><td> $S _ { 9 } \cap S _ { 7 2 }$ </td><td>money won</td><td></td><td>player</td><td></td></tr></table>

Table 3: Examples from original dataset (denoted as $\mathbf { D } ^ { \mathrm { S e t } } )$ and additional $\mathbf { D } ^ { \mathrm { U n i o n } }$ and D<sup>Intersect</sup> sets.

<table><tr><td>Method</td><td>Emb.</td><td></td><td>Set R@100</td><td>R@1k</td><td>Med.</td></tr><tr><td rowspan="5"> $\operatorname { R a n d } ^ { \bigstar }$   $\mathrm { N e a r } ^ { \bigstar }$  Se D</td><td></td><td>x</td><td>0.6</td><td>5.9</td><td rowspan="2">8520 641</td></tr><tr><td>word2vec</td><td>x</td><td>28.1</td><td>54.7</td></tr><tr><td>Fuzzy set Fuzzy set</td><td>word2vec</td><td>19.9</td><td>47.2 69.0</td><td>1240 320</td></tr><tr><td>GloVe Subspace set</td><td></td><td>30.9 29.7</td><td>58.9</td><td>478</td></tr><tr><td>word2vec Subspace set GloVe</td><td>V</td><td>35.7</td><td>72.7</td><td>246</td></tr><tr><td rowspan="6">Don Fuzzy set Subspace set Rand</td><td>Rand Near</td><td></td><td>x</td><td>0.6</td><td>6.0 8422</td></tr><tr><td>word2vec Fuzzy set</td><td>x</td><td>17.5</td><td>34.3</td><td>3270</td></tr><tr><td>word2vec</td><td></td><td>2.8</td><td>17.1</td><td>4426</td></tr><tr><td>GloVe</td><td></td><td>5.4</td><td>32.0</td><td>2347</td></tr><tr><td>word2vec</td><td></td><td>18.4</td><td>46.9</td><td>1202</td></tr><tr><td>Subspace set GloVe</td><td>L</td><td>24.4</td><td>68.3</td><td>407</td></tr><tr><td rowspan="5">Dntsect Near Fuzzy set Fuzzy set</td><td></td><td>x</td><td>0.2</td><td>6.6</td><td>7929</td></tr><tr><td>word2vec</td><td>x</td><td>23.5</td><td>40.8</td><td>3304</td></tr><tr><td>word2vec</td><td>V</td><td>4.7</td><td>20.9</td><td>3420</td></tr><tr><td>GloVe</td><td></td><td>32.5</td><td>75.0</td><td>255</td></tr><tr><td>Subspace set word2vec</td><td></td><td>25.7</td><td>45.7</td><td>1445</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Subspace set</td><td>GloVe</td><td>V</td><td>44.2</td><td>83.7</td><td>149</td></tr></table>

Table 4: Results of set retrieval task on $\mathbf { D } ^ { \mathrm { U n i o n } }$ (top half) and D<sup>Intersect</sup> (bottom half). The “Emb.” column indicates which pre-trained embedding is used. The “Set” column indicates whether each method is based on set computations: $\pmb { \nu }$ for incorporating set operations.

Dataset We used a previously created dataset (Zaheer et al., 2017), which was denoted by “LDA-1k, $\mathrm { V o c a b } = 1 7 \mathrm { k } . ^ { \prime \prime }$ in the paper. The dataset $( \mathbf { D } ^ { \mathrm { S e t } } )$ contains 100 word sets, each of which consists of 50 words sampled from a common topic<sup>9</sup>. Five pre-determined words from each set were used as the word set S. An additional 800 word sets were used to train the models that require training on word sets. Table 3 shows an example of the data and the number of test sets.

To evaluate the union and intersection sets, we prepared additional data through the union and intersection operations on two randomly-selected word sets from the original word sets (D<sup>Set</sup>)<sup>10</sup>. The number of words in each set in D<sup>Union</sup> was limited to 50 to match the original dataset (D<sup>Set</sup>). The number of words in each set in D<sup>Intersect</sup> was set to a minimum of 10. Finally, 100 unions and intersections were randomly selected from these word sets with zero elements excluded. See Table 3 for examples and statistics of the datasets.

Results In experiments on union and intersection, we compared our method only with Fuzzy Set. The proposed method and Fuzzy Set can induce representations for the union and intersection using set operations defined in the word embedding space; the others cannot do so directly. Table 4 shows the experimental results. Here our subspace-based set operation method (Subspace set) is the best among the methods that did not require training. The results suggest that combining off-the-shelf pretrained embeddings with appropriate set-oriented operations makes linguistic computation on sets feasible without additional training. The results in D<sup>Union</sup> and D<sup>Intersect</sup> show that the our method outperform Fuzzy Set in most metrics. As methods for achieving set operations in vector spaces, the proposed method is empirically more promising than the existing fuzzy set-based method.

## 7 Related Work

Symbol-based similarities between word sets have been proposed, such as Jaccard coefficient (Jaccard, 1901; Manning and Schütze, 2001; Thada and Jaglan, 2013) and TF-IDF-based similarity (Jurafsky, 2000). Unfortunately, symbol-based methods cannot capture the semantic similarity of similar sets or words when the symbols are different.

While many studies have explored representing word sets in pre-trained embedding spaces (Kusner et al., 2015; Yokoi et al., 2020), they primarily focus on set similarity. Our approach, however, extends beyond this by developing a comprehensive framework for various set operations within these spaces. Utilizing subspace properties, our method not only represents word sets but also performs a range of versatile operations, such as calculating textual similarities and membership degrees.

Many methods for learning the representation of sets have been proposed because of the wide range of possible applications (Zaheer et al., 2017; Pellegrini et al., 2021; Lee et al., 2019; Vilnis and Mc-Callum, 2015; Athiwaratkun and Wilson, 2017). In contrast, our approach does not require additional training. This enables us to compute set representations and operations using popular general-purpose language models, which are trained on the general domain (Brown et al., 2020).

## 8 Conclusion

This study introduces a novel framework for set representation and operations within pre-trained embedding spaces, employing linear subspaces grounded in quantum logic. This approach extends the scope of conventional embedding set operations by incorporating vector-based representations.

## Ethical Considerations

We recognize the importance of addressing the inherent biases in pre-trained models, such as gender stereotypes. In our experiment, we used RoBERTa, which has gender biases (Sharma et al., 2021). We used this model in its original state to preserve the experimental conditions of BERTScore, acknowledging that such biases may influence our results. However, we would like to emphasize that the focus of our work, which lies in sentence similarity, does not inherently add to or magnify these ethical concerns.

## Limitations

Our SubspaceBERTScore is built upon the foundation of BERTScore, which presents a limitation in that our results and findings are inherently dependent on the characteristics and performance of BERTScore. While we chose BERTScore due to its robustness and popularity in the field, potential biases or shortcomings intrinsic to BERTScore might be incorporated into our extension. Nevertheless, this constraint also suggests future research possibilities, such as applying our subspace-based approach to other base sentence similarity metrics, further expanding the versatility and applicability of our method.

The experiments we conducted were exclusive to BERT and RoBERTa. Testing our methodology with other pre-trained models, like GPT-3 (Brown et al., 2020), could broaden its applicability and establish its robustness across various pre-trained models.

We evaluated our methodology primarily using English datasets. This decision was made to streamline our initial explorations rather than due to an inherent language-specific bias in our approach. We expect that our subspace-based methodology will be effective across various languages.

## Acknowledgments

This work was supported by JSPS KAKENHI Grant Number 22H03654, 22H03651, and JST, ACT-X Grant Number JPMJAX200S, Japan.

## References

Eneko Agirre, Carmen Banea, Claire Cardie, Daniel M. Cer, Mona T. Diab, Aitor Gonzalez-Agirre, Weiwei Guo, Iñigo Lopez-Gazpio, Montse Maritxalar, Rada Mihalcea, German Rigau, Larraitz Uria, and Janyce Wiebe. 2015. SemEval-2015 Task 2: Semantic Textual Similarity, English, Spanish and Pilot on Interpretability. In Proceedings of the 9th International Workshop on Semantic Evaluation, SemEval@NAACL-HLT 2015, Denver, Colorado, USA, June 4-5, 2015, pages 252–263. The Association for Computer Linguistics.

Eneko Agirre, Carmen Banea, Claire Cardie, Daniel M. Cer, Mona T. Diab, Aitor Gonzalez-Agirre, Weiwei Guo, Rada Mihalcea, German Rigau, and Janyce Wiebe. 2014. SemEval-2014 Task 10: Multilingual Semantic Textual Similarity. In Proceedings ofthe 8th International Workshop on Semantic Evaluation, SemEval@COLING 2014, Dublin, Ireland, August 23-24, 2014, pages 81–91. The Association for Computer Linguistics.

Eneko Agirre, Carmen Banea, Daniel M. Cer, Mona T. Diab, Aitor Gonzalez-Agirre, Rada Mihalcea, German Rigau, and Janyce Wiebe. 2016. Semeval-2016 task 1: Semantic textual similarity, monolingual and cross-lingual evaluation. In Proceedings ofthe 10th International Workshop on Semantic Evaluation, SemEval@NAACL-HLT 2016, San Diego, CA, USA, June 16-17, 2016, pages 497–511. The Association for Computer Linguistics.

Eneko Agirre, Daniel M. Cer, Mona T. Diab, and Aitor Gonzalez-Agirre. 2012. SemEval-2012 Task 6: A Pilot on Semantic Textual Similarity. In Proceedings ofthe 6th International Workshop on Semantic Evaluation, SemEval@NAACL-HLT 2012, Montréal, Canada, June 7-8, 2012, pages 385–393. The Association for Computer Linguistics.

Eneko Agirre, Daniel M. Cer, Mona T. Diab, Aitor Gonzalez-Agirre, and Weiwei Guo. 2013. \*SEM 2013 shared task: Semantic Textual Similarity. In Proceedings ofthe Second Joint Conference on Lexical and Computational Semantics, \*SEM 2013, June 13-14, 2013, Atlanta, Georgia, USA, pages 32–43. Association for Computational Linguistics.

Sanjeev Arora, Yingyu Liang, and Tengyu Ma. 2017. A Simple but Tough-to-Beat Baseline for Sentence Embeddings. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net.

Ben Athiwaratkun and Andrew Gordon Wilson. 2017. Multimodal Word Distributions. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers, pages 1645–1656.

Satanjeev Banerjee and Alon Lavie. 2005. METEOR: an automatic metric for MT evaluation with improved correlation with human judgments. In Proceedings of the Workshop on Intrinsic and Extrinsic Evaluation Measuresfor Machine Translation and/or Summarization@ACL 2005, Ann Arbor, Michigan, USA, June 29, 2005, pages 65–72. Association for Computational Linguistics.

Garrett Birkhoff and John Von Neumann. 1936. The logic of quantum mechanics. Annals of mathematics, pages 823–843.

David M. Blei, Andrew Y. Ng, and Michael I. Jordan. 2003. Latent Dirichlet Allocation. J. Mach. Learn. Res., 3:993–1022.

Ondv ej Bojar, Christian Federmann, Mark Fishel, Yvette Graham, Barry Haddow, Matthias Huck, Philipp Koehn, and Christof Monz. 2018. Findings of the 2018 conference on machine translation (WMT18). In Proceedings ofthe Third Conference on Machine Translation: Shared Task Papers, pages 272–303, Belgium, Brussels. Association for Computational Linguistics.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Daniel M. Cer, Mona T. Diab, Eneko Agirre, Iñigo Lopez-Gazpio, and Lucia Specia. 2017. SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation. In Proceedings ofthe 11th International Workshop on Semantic Evaluation, SemEval@ACL 2017, Vancouver, Canada, August 3-4, 2017, pages 1–14. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 4171–4186. Association for Computational Linguistics.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple Contrastive Learning of Sentence Embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 6894– 6910. Association for Computational Linguistics.

Paul Jaccard. 1901. Distribution de la flore alpine dans le bassin des dranses et dans quelques régions voisines. Bulletin de la Societe Vaudoise des Sciences Naturelles, 37:241–72.

Dan Jurafsky. 2000. Speech & language processing. Pearson Education India.

Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, and Kilian Q. Weinberger. 2015. From Word Embeddings To Document Distances. In Proceedings ofthe 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015, volume 37 of JMLR Workshop and Conference Proceedings, pages 957–966. JMLR.org.

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, and Yee Whye Teh. 2019. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. In Proceedings ofthe 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pages 3744–3753. PMLR.

Qingsong Ma, Ondrej Bojar, and Yvette Graham. 2018. Results of the WMT18 metrics shared task: Both characters and embeddings achieve good performance. In Proceedings of the Third Conference on Machine Translation: Shared Task Papers, WMT 2018, Belgium, Brussels, October 31 - November 1, 2018, pages 671–688. Association for Computational Linguistics.

Christopher D. Manning and Hinrich Schütze. 2001. Foundations of statistical natural language processing. MIT Press.

Marco Marelli, Stefano Menini, Marco Baroni, Luisa Bentivogli, Raffaella Bernardi, and Roberto Zamparelli. 2014. A SICK cure for the evaluation of compositional distributional semantic models. In Proceedings of the Ninth International Conference on Language Resources and Evaluation, LREC 2014, Reykjavik, Iceland, May 26-31, 2014, pages 216–223. European Language Resources Association (ELRA).

Tomas Mikolov, Ilya Sutskever, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. 2013. Distributed Representations of Words and Phrases and their Compositionality. In Advances in Neural Information Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems 2013. Proceedings ofa meeting held December 5-8, 2013, Lake Tahoe, Nevada, United States., pages 3111–3119.

Momose Oyama, Sho Yokoi, and Hidetoshi Shimodaira. 2023. Norm of word embedding encodes information gain. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 2108–2130. Association for Computational Linguistics.

Giovanni Pellegrini, Alessandro Tibo, Paolo Frasconi, Andrea Passerini, and Manfred Jaeger. 2021. Learning Aggregation Functions. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI 2021, Virtual Event / Montreal, Canada, 19-27 August 2021, pages 2892–2898. ijcai.org.

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. Glove: Global Vectors for Word Representation. In Proceedings ofthe 2014 Conference on Empirical Methods in Natural Language Processing, EMNLP 2014, October 25-29, 2014, Doha, Qatar, A meeting of SIGDAT, a Special Interest Group ofthe ACL, pages 1532–1543.

Shanya Sharma, Manan Dey, and Koustuv Sinha. 2021. Evaluating gender bias in natural language inference. CoRR, abs/2105.05541.

Vikas Thada and Vivek Jaglan. 2013. Comparison of jaccard, dice, cosine similarity coefficient to find best fitness value for web retrieved documents using genetic algorithm. International Journal of Innovations in Engineering and Technology, 2(4):202–205.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008.

Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. 2015. Cider: Consensus-based image description evaluation. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015, pages 4566–4575. IEEE Computer Society.

Luke Vilnis and Andrew McCallum. 2015. Word Representations via Gaussian Embedding. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings.

Sho Yokoi, Ryo Takahashi, Reina Akama, Jun Suzuki, and Kentaro Inui. 2020. Word Rotator’s Distance. In

Proceedings of the 2020 Conference on Empirical   
Methods in Natural Language Processing, EMNLP   
2020, Online, November 16-20, 2020, pages 2944–   
2960. Association for Computational Linguistics.   
Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh,   
Barnabás Póczos, Ruslan Salakhutdinov, and Alexan  
der J. Smola. 2017. Deep Sets. In Advances in   
Neural Information Processing Systems 30: Annual   
Conference on Neural Information Processing Sys  
tems 2017, December 4-9, 2017, Long Beach, CA,   
USA, pages 3391–3401.   
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.   
Weinberger, and Yoav Artzi. 2020. BERTScore:   
Evaluating Text Generation with BERT. In 8th Inter  
national Conference on Learning Representations,   
ICLR 2020, Addis Ababa, Ethiopia, April 26-30,   
2020. OpenReview.net.   
Vitalii Zhelezniak, Aleksandar Savkov, April Shen,   
Francesco Moramarco, Jack Flann, and Nils Y. Ham  
merla. 2019. Don’t Settle for Average, Go for the   
Max: Fuzzy Sets and Max-Pooled Word Vectors. In   
7th International Conference on Learning Represen  
tations, ICLR 2019, New Orleans, LA, USA, May 6-9,   
2019. OpenReview.net.

## A Pseudocodes

The pseudocodes for our SubspaceBERTScore and basic set operations are shown in Algorithms 3, 4, 5, and 6.

Algorithm 3 Union   
Input: $\mathbf { S } _ { \mathrm { A } } \in \mathbb { R } ^ { k \times d } ;$ : Bases of $\mathbb { S } _ { A }$   
Input: $\mathbf { S } _ { \mathrm { B } } \in \mathbb { R } ^ { \ell \times d } \colon$ Bases of $\mathbb { S } _ { B }$   
Output: $\mathbf { S } _ { \mathrm { A U B } } \in \mathbb { R } ^ { r \times d } \colon$ Bases of $\mathbb { S } _ { A } \cup \mathbb { S } _ { B }$   
M $\in \mathbb { R } ^ { ( k + \ell ) \times d } \gets \mathrm { C O N C A T \_ R O W S } ( \mathbf { S } _ { \mathrm { A } } , \mathbf { S } _ { \mathrm { B } } )$   
$\mathbf { S } _ { \mathrm { A U B } } \in \mathbb { R } ^ { r \times d }  \mathrm { O R T H O \_ N O R M A L } ( \mathbf { M } ) \triangleright r$ is   
rank of M   
return $\mathbf { S } _ { \mathrm { A U B } }$

Algorithm 4 Intersection   
Input: $\mathbf { S } _ { \mathrm { A } } \in \mathbb { R } ^ { k \times d } \mathrm { : }$ Bases of $\mathbb { S } _ { A }$   
Input: $\mathbf { S } _ { \mathrm { B } } \in \mathbb { R } ^ { \ell \times d } \colon$ Bases of S $( \ell \geq k )$   
Input: α: Threshold below which the cosine of   
canonical angles is considered zero.   
Output: $\mathbf { S } _ { \mathrm { A } \cap \mathrm { B } } \in \mathbb { R } ^ { m \times d } \colon$ Bases of $\mathbb { S } _ { A } \cap \mathbb { S } _ { B }$   
M $\ u \in \mathbb { R } ^ { k \times \ell } \gets \mathbf { S } _ { \mathrm { { A } } } \mathbf { S } _ { \mathrm { { B } } } ^ { \top }$   
U $\begin{array} { r l r } { \in } & { { } \mathbb { R } ^ { k \times \ell } , \sum ^ { \sim \mathrm { ~ \infty ~ } } \in \mathrm { ~ \mathbb { R } ~ } ^ { \ell \times \ell } , { \bf V } ^ { \top } } & { \in } & { { } \mathbb { R } ^ { \ell \times \ell } \mathrm { ~  ~ } } \end{array}$   
SVD(M)   
△ $\cdot \Sigma = \operatorname { d i a g } ( \sigma _ { 1 } , \dots , \sigma _ { \ell } ) ( \sigma _ { 1 } \geq \cdot \cdot \cdot \geq \sigma _ { \ell } )$ has   
cosines of the canonical angles between $\mathbb { S } _ { A }$ and   
$\mathbb { S } _ { B }$   
$\mathbf { W } \in \mathbb { R } ^ { m \times d }  \mathbf { U } [ : , 1 : \sigma _ { m } ]$   
▷ m is the maximum index that satisfies   
$| \sigma _ { i } - 1 | \leq \alpha .$   
return $\mathbf { S } _ { \mathrm { A } \cap \mathrm { B } }$

Algorithm 5 Complement   
Input: $\mathbf { S } _ { \mathrm { A } } \in \mathbb { R } ^ { k \times d } \mathrm { : }$ Bases of $\mathbb { S } _ { A }$   
Output: $\mathbf { S } _ { \overline { { \mathsf { A } } } } \in \mathbb { R } ^ { ( d - k ) \times d } ;$ : Bases of $\mathbb { S } _ { \overline { { A } } }$   
$\mathbf { U } ~ \in ~ \mathbb { R } ^ { \hat { d } \times d } , \pmb { \Sigma } ~ \in ~ \mathbb { R } ^ { d \times k } , \mathbf { V } ^ { \top } ~ \in ~ \mathbb { R } ^ { k \times k } ~ $   
$\mathrm { S V D } ( \mathbf { S } _ { \mathrm { A } } ^ { \top } )$   
$\begin{array} { r } { \mathbf { S } _ { \overline { { \mathrm { A } } } } \in \dot { \mathbb { R } } ^ { ( \dot { d } - k ) \times d }  ( \mathbf { U } [ : , k : d ] ) ^ { \top } } \end{array}$   
return $\mathbf { S } _ { \overline { { \mathrm { A } } } }$

## B WMT Results

An example of a practical task where our proposed method can be directly utilized is the automatic evaluation of machine translation systems. We applied our SubspaceBERTScore to the WMT18 (Ma et al., 2018). The table 5 presents the results of applying both BERTScore and our proposed SubspaceBERTScore to various X-to-English translation settings in the WMT18 competition. Our proposed method consistently outperforms BERTScore. Specifically, for all X-to-English translation settings, we observed an increase in Kendall’s τ for F-score, Precision (P), and Recall (R). It is evident that SubspaceBERTScore shows an improvement across all the considered metrics, underscoring the effectiveness of our proposed method in evaluating machine translation systems.

<table><tr><td>Method</td><td>Metric</td><td>cs-en</td><td>de-en</td><td>et-en</td><td>fi-en</td><td>ru-en</td><td>tr-en</td><td>zh-en</td><td>Avg.</td></tr><tr><td rowspan="3">BERTScore</td><td>F</td><td>.404</td><td>.550</td><td>.397</td><td>.296</td><td>.353</td><td>.292</td><td>.264</td><td>.365</td></tr><tr><td>P</td><td>.387</td><td>.541</td><td>.389</td><td>.283</td><td>.345</td><td>.280</td><td>.248</td><td>.353</td></tr><tr><td>R</td><td>.388</td><td>.546</td><td>.391</td><td>.304</td><td>.343</td><td>.290</td><td>.255</td><td>.360</td></tr><tr><td rowspan="3">SubspaceBERTScore</td><td>F</td><td>.411</td><td>.557</td><td>.403</td><td>.309</td><td>.358</td><td>.303</td><td>.264</td><td>.372</td></tr><tr><td>P</td><td>.382</td><td>.548</td><td>.393</td><td>.290</td><td>.352</td><td>.294</td><td>.248</td><td>.358</td></tr><tr><td>R</td><td>.391</td><td>.547</td><td>.392</td><td>.313</td><td>.358</td><td>.292</td><td>.259</td><td>.365</td></tr></table>

Table 5: Comparison of BERTScore and SubspaceBERTScore on WMT18 for X-to-English translation tasks.

Algorithm 6 SubspaceBERTScore   
Input: $\{ \pmb { a } ^ { ( 1 ) } , \dots , \pmb { a } ^ { ( k ) } \} \subseteq \mathbb { R } ^ { 1 \times d } \colon$ Token vectors   
of first sentence   
Input: $\{ b ^ { ( 1 ) } , \dots , b ^ { ( \ell ) } \} \subseteq \mathbb { R } ^ { 1 \times d } \colon$ Token vectors of   
second sentence   
Output: $P \in \mathbb { R } , R \in \mathbb { R } , F \in \mathbb { R } \mathrm { : }$ : Similarity score   
$\mathbf { A } \in \mathbb { R } ^ { k \times d }  \operatorname { s T a C K \_ R O W S } ( \pmb { a } ^ { ( 1 ) } , \dots , \pmb { a } ^ { ( k ) } )$   
$\mathbf { B } \in \mathbb { R } ^ { \ell \times d }  \mathrm { s T a C K \_ R O W S } ( \lfloor ^ { ( 1 ) } , \dots , \pmb { b } ^ { ( \ell ) } )$   
$\mathbf { A } _ { \mathrm { { o r t h } } } \in \mathbb { R } ^ { n \times d }  ( \mathrm { { O R T H O \_ N O R M A L } } ( \mathbf { A } ^ { \top } ) ) ^ { \top }$   
$\mathbf { B } _ { \mathrm { o r t h } } \in \mathbb { R } ^ { m \times d }  ( \mathrm { o R T H O \_ N O R M A L } ( \mathbf { B } ^ { \top } ) ) ^ { \top }$   
▷ Orthonormalize the bases. n and m are the   
ranks.   
k k   
$R  \sum f ( a _ { i } ) \mathbb { 1 } _ { \mathrm { s u b s p a c e } } ( { \pmb a } ^ { ( i ) } , { \bf B } _ { \mathrm { o r t h } } ) \bigg / \sum f ( a _ { i } )$   
i=1 i=1   
  
$P \gets \sum _ { j = 1 } ^ { \varepsilon } f ( b _ { i } ) \mathbb { 1 } _ { \mathrm { s u b s p a c e } } ( \pmb { b } ^ { ( j ) } , \mathbf { A } _ { \mathrm { o r t h } } ) \Big / \sum _ { j = 1 } ^ { \varepsilon } f ( b _ { i } )$   
▷ f( ) is a weighting function.   
$F \gets 2 ( P \cdot R ) / ( P + R )$   
return $P , R , F$

## C Vector vs. Set of Vectors

Table 6 presents the comparison results on the STS-B dataset. We utilized embeddings from BERT and SimCSE (Gao et al., 2021). Avg-cos and CLScos are based on the cosine similarity of vectors, employing methods that compress information into a single vector. In contrast, BERTScore among other methods is based on sets of vectors. A key takeaway from these results is the suggestion that methods using sets of vectors outperform those using a single vector in evaluating similarity. It is particularly interesting that even SimCSE, which aims to optimize average cosine similarity, shows superior score when using set-based similarity.

<table><tr><td>Emb.</td><td>Method</td><td>STS-B</td></tr><tr><td rowspan="4">BERT-base</td><td>CLS-cos</td><td>.424</td></tr><tr><td>Avg-cos</td><td>.582</td></tr><tr><td>BERTScore†</td><td>.446</td></tr><tr><td>SubspaceBERTScore†</td><td>.478</td></tr><tr><td rowspan="4">SimCSE</td><td>CLS-cos</td><td>.769</td></tr><tr><td>Avg-cos</td><td>.786</td></tr><tr><td> $\mathrm { \mathbf { B E R T S c o r e } ^ { \dagger } }$ </td><td>.798</td></tr><tr><td> $\mathrm { S u b s p a c e B E R T S c o r e } ^ { \dagger }$ </td><td>.801</td></tr></table>

Table 6: Comparison of vector-based similarity and setbased similarity metrics (†) on the STS-B dataset.