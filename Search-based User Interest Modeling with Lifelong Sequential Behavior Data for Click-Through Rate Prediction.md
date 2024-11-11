## ABSTRACT
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
Rich user behavior data has been proven to be of great value for click-through rate prediction tasks, especially in industrial applications such as recommender systems and online advertising. Both industry and academy have paid much attention to this topic and propose different approaches to modeling with long sequential user behavior data. Among them, memory network based model MIMN[8] proposed by Alibaba, achieves SOTA with the co-design of both learning algorithm and serving system. MIMN is the first industrial solution that can model sequential user behavior data with length scaling up to 1000. However, MIMN fails to precisely capture user interests given a specific candidate item when the length of user behavior sequence increases further, say, by 10 times or more. This challenge exists widely in previously proposed approaches.    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
In this paper, we tackle this problem by designing a new modeling paradigm, which we name as Search-based Interest Model (SIM). SIM extracts user interests with two cascaded search units: (i) General Search Unit (GSU) acts as a general search from the raw and arbitrary long sequential behavior data, with query information from candidate item, and gets a Sub user Behavior Sequence (SBS) which is relevant to candidate item; (ii) Exact Search Unit (ESU) models the precise relationship between candidate item and SBS. This cascaded search paradigm enables SIM with a better ability to model lifelong sequential behavior data in both scalability and accuracy. Apart from the learning algorithm, we also introduce our hands-on experience on how to implement SIM in large scale industrial systems. Since 2019, SIM has been deployed in the display advertising system in Alibaba, bringing 7.1% CTR and 4.4% RPM lift, which is significant to the business. Serving the main traffic in our real system now, SIM models sequential user behavior data with maximum length reaching up to 54000, pushing SOTA to 54x.

## 摘要
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
丰富的用户行为数据在点击率预测任务中被证明具有极大的价值，尤其是在推荐系统和在线广告等工业应用中。无论是业界还是学术界，都对这一主题给予了大量关注，并提出了多种长序列用户行为数据建模的方法。其中，阿里巴巴提出的基于记忆网络的模型MIMN[8]，通过学习算法与服务系统的联合设计达到了当前最优（SOTA）的效果。MIMN是第一个能够处理长度扩展到1000的用户行为序列数据的工业解决方案。然而，当用户行为序列长度进一步增加，比如扩大到10倍或更多时，MIMN在准确捕捉特定候选项对应的用户兴趣方面表现不佳。这一挑战在此前提出的多种方法中普遍存在。     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
在本文中，我们通过设计一种新的建模范式来解决这一问题，该方法被称为基于搜索的兴趣模型（Search-based Interest Model, SIM）。SIM通过两个级联的搜索单元提取用户兴趣：（i）通用搜索单元（General Search Unit, GSU）利用候选项的查询信息，从原始且任意长的用户行为序列中进行通用搜索，获取与候选项相关的子用户行为序列（Sub user Behavior Sequence, SBS）；（ii）精确搜索单元（Exact Search Unit, ESU）建模候选项与SBS之间的精确关系。该级联搜索范式使得SIM在可扩展性和精确性方面具备了更强的能力，从而能够更好地建模终身的用户行为数据序列。除了学习算法外，我们还介绍了如何在大规模工业系统中实现SIM的实践经验。自2019年以来，SIM已部署在阿里巴巴的展示广告系统中，带来了7.1%的CTR提升和4.4%的RPM提升，这对业务来说是非常显著的。如今，SIM已在实际系统中服务主流流量，能够处理长度达到54000的用户行为数据序列，是当前最优水平的54倍


## 1 INTRODUCTION

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
Click-Through Rate (CTR) prediction modeling plays a critical role in industrial applications such as recommender systems and online advertising. Due to the rapid growth of user historical behavior data, user interest modeling, which focuses on learning the intent representation of user interest, has been widely introduced in the CTR prediction model [2, 8, 19, 20]. However, most of the proposed approaches can only model sequential user behavior data with length scaling up to hundreds, limited by the burden of computation and storage in real online systems [19, 20]. Rich user behavior data is proven to be of great value [8]. For example, 23% of users in Taobao, one of the world’s leading e-commerce site, click with more than 1000 products in last 5 months[8, 10]. How to design a feasible solution to model the long sequential user behavior data has been an open and hot topic, attracting researchers from both industry and academy.
## 1 介绍：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
点击率（CTR）预测建模在推荐系统和在线广告等工业应用中起着关键作用。随着用户历史行为数据的迅速增长，专注于学习用户兴趣意图表示的用户兴趣建模已广泛引入CTR预测模型中 [2, 8, 19, 20]。然而，由于实时在线系统中计算和存储的负担，大多数提出的方法只能处理长度最多为几百的用户行为序列数据 [19, 20]。丰富的用户行为数据被证明具有极高的价值 [8]。例如，在全球领先的电商平台淘宝中，23%的用户在过去五个月中点击了超过1000个商品 [8, 10]。如何设计一个可行的解决方案来建模长序列的用户行为数据一直是一个开放且热门的话题，吸引了来自工业界和学术界的研究者的关注。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
A branch of research, which borrows ideas from the area of NLP, proposes to model long sequential user behavior data with mem- ory network and makes some breakthroughs. MIMN[8] proposed by Alibaba, is one typical work, which achieves SOTA with the co-design of both learning algorithm and serving system. MIMN is the first industrial solution which can model sequential user behavior data with length scaling up to 1000. Specifically, MIMN incrementally embeds diverse interest of one user into a fixed size memory matrix which will be updated by each new behavior. In that way, the computation of user modeling is decoupled from CTR prediction. Thus for online serving, latency will not be a problem and the storage cost depends on the size of the memory matrix which is much less than the raw behavior sequence. A similar idea can be found in long-term interest modeling[10]. However, it is still challenging for memory network based approaches to model arbitrary long sequential data. Practically, we find that MIMN fails to precisely capture user interest given a specific candidate item when the length of user behavior sequence increases further, say, up to 10000 or more. This is because encoding all user historical behaviors into a fixed size memory matrix causes massive noise to be contained in the memory units.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
有一类研究借鉴了自然语言处理（NLP）领域的思路，提出使用记忆网络来建模长序列用户行为数据，并取得了一些突破。阿里巴巴提出的MIMN[8]是一个典型的工作，通过学习算法和服务系统的联合设计实现了当前最优（SOTA）的效果。MIMN是首个能够建模长度达1000的用户行为序列的工业解决方案。具体来说，MIMN将用户的多样兴趣逐步嵌入一个固定大小的记忆矩阵中，每次新的行为都会对该矩阵进行更新。这样，用户建模的计算与CTR预测解耦，因此在在线服务中，延迟不会成为问题，存储成本取决于记忆矩阵的大小，而非原始行为序列的长度。类似的想法也可见于长期兴趣建模中[10]。然而，对于基于记忆网络的方法而言，仍然难以处理任意长的序列数据。在实际应用中，我们发现当用户行为序列长度进一步增加到10000或更多时，MIMN无法准确捕捉特定候选项的用户兴趣。这是因为将所有用户的历史行为编码到一个固定大小的记忆矩阵中会导致记忆单元中包含大量噪声.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
On the other hand, as pointed out in the previous work of DIN[20], the interest of one user is diverse and varies when facing different candidate items. The key idea of DIN is searching the ef- fective information from user behaviors to model special interest of user, facing different candidate items. In this way, we can tackle the challenge of encoding all user interest into fixed-size parameters. DIN does bring a big improvement for CTR modeling with user behavior data. But the searching formula of DIN costs an unac- ceptable computation and storage facing the long sequential user behavior data as we mentioned above. So, can we apply a similar search trick and design a more efficient way to extract knowledge from the long sequential user behavior data?

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
另一方面，正如DIN[20]的前期研究所指出的，用户的兴趣是多样的，并会在面对不同的候选项时发生变化。DIN的关键思路是从用户行为中搜索有效信息，以建模用户在面对不同候选项时的特定兴趣。通过这种方式，我们可以应对将所有用户兴趣编码到固定大小参数中的挑战。DIN确实在结合用户行为数据的CTR建模中带来了显著的提升。然而，如上所述，DIN的搜索公式在处理长序列用户行为数据时带来了不可接受的计算和存储成本。那么，是否可以应用类似的搜索技巧，并设计一种更高效的方式从长序列用户行为数据中提取知识呢？

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
In this paper, we tackle the challenge by designing a new modeling paradigm, which we name as Search-based Interest Model (SIM). SIM adopts the idea of DIN [20] and captures only relevant user interest with respect to specific candidate items. In SIM, user interest is extracted with two cascaded search units:(i) General Search Unit (GSU) acts as a general search from the raw and arbi- trary long sequential behavior data, with query information from candidate item, and gets a Sub user Behavior Sequence (SBS) which is relevant to candidate item. In order to meet the strict limitation of latency and computation resources, general but effective methods are utilized in the GSU. To our experience, the length of SBS can be cut down to hundreds and most of the noise information in raw long sequential behavior data could be filtered. (ii) Exact Search Unit (ESU) models the precise relationship between the candidate item and SBS. Here we can easily apply similar methods proposed by DIN[20] or DIEN[19].

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
在本文中，我们通过设计一种新的建模范式来应对这一挑战，该方法被称为基于搜索的兴趣模型（Search-based Interest Model, SIM）。SIM借鉴了DIN[20]的思想，仅捕捉与特定候选项相关的用户兴趣。在SIM中，用户兴趣通过两个级联的搜索单元进行提取：（i）通用搜索单元（General Search Unit, GSU）从原始且任意长的用户行为序列中进行通用搜索，利用候选项的查询信息获取与候选项相关的子用户行为序列（Sub user Behavior Sequence, SBS）。为了满足延迟和计算资源的严格限制，GSU中采用了通用但有效的方法。根据我们的经验，SBS的长度可以减少到几百，并且大部分原始长序列行为数据中的噪声信息可以被过滤掉。（ii）精确搜索单元（Exact Search Unit, ESU）建模候选项与SBS之间的精确关系。在此我们可以轻松地应用DIN[20]或DIEN[19]提出的类似方法。

The main contributions of this work are summarized as follows:
 - We propose a new paradigm SIM for modeling long sequential user behavior data. The design of a cascaded two-stage search mechanism enables SIM with a better ability to model lifelong sequential behavior data in both scalability and ac- curacy.
 - We introduce our hands-on experience of implementing SIM in large scale industrial systems. Since 2019, SIM has been deployed in the display advertising system in Alibaba, bringing 7.1% CTR and 4.4% RPM lift. Now, SIM is serving the main traffic. 
 - We push the maximum length for modeling with long sequential user behavior data up to 54000, 54x larger than MIMN, the published SOTA industry solution for this task.

该工作的主要贡献总结如下：

 - 我们提出了一种新的范式SIM用于建模长序列用户行为数据。级联的两阶段搜索机制设计使得SIM在可扩展性和准确性方面具备更强的能力，以更好地建模终身用户行为数据。
 - 我们介绍了在大规模工业系统中实现SIM的实践经验。自2019年以来，SIM已部署在阿里巴巴的展示广告系统中，带来了7.1%的CTR提升和4.4%的RPM提升。目前，SIM已服务于主要流量。
 - 我们将长序列用户行为数据建模的最大长度提升至54000，是现有发表的工业解决方案MIMN的54倍。




## 2  RELATED WORK
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
__User Interest Model.__ Deep learning based methods have achieved great success in CTR prediction task[1, 11, 18]. In early age, most pioneer works[1, 4, 7, 9, 15] use a deep neural network to capture interactions between features from different fields so that engineers could get rid of boring feature engineering works. Recently, a series of works, which we called User Interest Model, focus on learning the representation of latent user interest from historical behaviors, using different neural network architecture such as CNN[14, 17], RNN[5, 19], Transformer[3, 13] and Capsule[6], etc. DIN[20] emphasizes that user interest are diverse and an attention mechanism is introduced in DIN to capture users’ diverse interest on the differ- ent target items. DIEN[19] points out that the temporal relationship between historical behaviors matters for modeling users’ drifting interest. An interest extraction layer based on GRU with auxiliary loss is designed in DIEN. MIND[6] argues that using a single vector to represent one user is insufficient to capture the varying nature of the user’s interest. Capsule network and dynamic routing method are introduced in MIND to learn the representation of user interest as multiple vectors. Moreover, inspired by the success of the self-attention architecture in the tasks of sequence to sequence learning, Transformer is introduced in [3] to model user cross-session and in-session interest.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
## 2  相关工作
__用户兴趣模型.__ 基于深度学习的方法在CTR预测任务中取得了巨大成功[1, 11, 18]。在早期，大多数先驱性工作[1, 4, 7, 9, 15]使用深度神经网络来捕捉不同字段特征之间的交互，以便工程师可以摆脱繁琐的特征工程工作。最近，一系列我们称之为用户兴趣模型的工作，专注于通过历史行为学习潜在用户兴趣的表示，采用不同的神经网络架构，如CNN[14, 17]、RNN[5, 19]、Transformer[3, 13]和Capsule网络[6]等。
DIN[20]强调用户兴趣是多样的，并在DIN中引入了注意力机制，以捕捉用户对不同目标项的多样兴趣。DIEN[19]指出历史行为之间的时间关系对于建模用户的兴趣漂移非常重要，并在DIEN中设计了基于GRU的兴趣提取层，辅以辅助损失。MIND[6]认为，使用单一向量表示用户不足以捕捉用户兴趣的多变性，因此在MIND中引入了胶囊网络和动态路由方法，以将用户兴趣表示为多个向量。此外，受自注意力架构在序列到序列学习任务中成功的启发，Transformer在[3]中被引入，用于建模用户的跨会话和会话内兴趣。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
__Long-term User Interest.__ [8] shows that considering long- term historical behavior sequences in the user interest model can significantly improve CTR model performance. Although longer user behavior sequences bring in more useful information for user interest modeling, it extremely burdens the latency and storage of an online serving system and contains massive noise for point-wise CTR prediction at the same time. A series of works focus on tackling the challenge in long-term user interest modeling, which usually learns user interest representation based on historical behavior sequences with extremely large length even lifelong. [10] proposes a Hierarchical Periodic Memory Network for lifelong sequential modeling with personalized memorization of sequential patterns for each user. [16] choose an attention-based framework to combine users’ long-term and short-term preferences. And they adopt the attentive Asymmetric-SVD paradigm to model long-term interest. A memory-based architecture named MIMN is proposed in [8], which embedded user long-term interest into fixed-sized memory network to solve the problem of large storage of user behavior data. And a UIC module is designed to record the new user behaviors incrementally to deal with the latency limitation. But MIMN abandons the information from the target item in the memory network which has been proved to be important for user interest modeling.


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
__长期用户兴趣__ [8] 表明，在用户兴趣模型中考虑长期历史行为序列可以显著提高CTR模型的性能。尽管较长的用户行为序列为用户兴趣建模带来了更多有用信息，但它极大地增加了在线服务系统的延迟和存储负担，同时对于点对点的CTR预测也包含了大量噪声。一系列工作致力于解决长期用户兴趣建模中的挑战，这些方法通常基于极长甚至终身的历史行为序列来学习用户兴趣表示。[10] 提出了一种分层周期记忆网络，用于终身序列建模，以为每个用户个性化地记忆其序列模式。[16] 选择了一种基于注意力的框架，结合用户的长期和短期偏好，并采用注意力不对称SVD范式来建模长期兴趣。[8] 提出了一种基于记忆的架构MIMN，它将用户的长期兴趣嵌入到固定大小的记忆网络中，以解决用户行为数据的大存储问题。同时设计了一个UIC模块，以增量记录新的用户行为来应对延迟限制。但MIMN在记忆网络中舍弃了被推荐物品的信息，而这一信息已被证明对用户兴趣建模至关重要。

## 3 SEARCH-BASED INTEREST MODEL
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
It has been proven to be effective by modeling user behavior data for CTR prediction modeling. Typically, attention-based CTR models, such as DIN[20] and DIEN[19], design complex model structure and involve attention mechanism to capture user diverse interest by searching effective knowledge from user behavior sequence, with inputs from different candidate items. But in a real-world system, these models can only handle short-term behavior sequence data, of which the length is usually less than 150. On the other hand, the long-term user behavior data is valuable, and modeling the long- term interest of users may bring more diverse recommendation results for users. It seems we are on the horns of a dilemma: we cannot handle the valuable life-long user behavior data with the effective but complex methods in a real-world system. To tackle this challenge, we propose a new modeling paradigm, which is named as Search-based Interest Model (SIM). SIM follows a two-stage search strategy and can handle long user behavior sequences in an efficient way. In this section, we will introduce the overall workflow of SIM first and then introduce the two proposed search units in detail

## 3 搜索式兴趣模型
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
通过对用户行为数据进行CTR预测建模已被证明是有效的。通常，基于注意力的CTR模型（如DIN[20]和DIEN[19]）设计了复杂的模型结构，并引入注意力机制，通过从用户行为序列中搜索有效信息来捕捉用户的多样兴趣，借助不同候选项作为输入。然而，在真实系统中，这些模型只能处理短期行为序列数据，通常长度不超过150。另一方面，长期用户行为数据具有重要价值，对用户的长期兴趣进行建模可能会为用户带来更多样化的推荐结果。我们似乎陷入了两难境地：在真实系统中无法用有效但复杂的方法处理有价值的终身用户行为数据。
为了解决这一挑战，我们提出了一种新的建模范式，称为基于搜索的兴趣模型（Search-based Interest Model, SIM）。SIM遵循一个两阶段的搜索策略，能够高效地处理长用户行为序列。在本节中，我们将首先介绍SIM的整体工作流程，然后详细介绍所提出的两个搜索单元。

### 3.1 Overall Workflow

The overall workflow of SIM is shown in Figure 1. SIM follows a cascaded two-stage search strategy with two corresponding units: the General Search Unit (GSU) and the Exact Search Unit (ESU).

**In the first stage**, we utilize General Search Unit (GSU) to seek top-K relevant sub behavior sequences from original long-term behavior sequences with sub-linear time complexity. Here K is generally much shorter than the original length of behavior sequences. An efficient search method can be conducted if the relevant behaviors can be searched under the limitation of time and computation resources. In section 3.2 we provide two straightforward implementations of GSU: soft-search and hard-search. GSU takes a general but effective strategy to cut off the length of raw sequential behaviors to meet the strict limitation of time and computation resources.
Meanwhile, the massive noise that exists in the long-term user behavior sequence, which may undermine user interest modeling, can be filtered by the search strategy in the first stage.

**In the second stage**, the Exact Search Unit (ESU), which takes the filtered sub-sequential user behaviors as input, is introduced to further capture the precise user interest. Here a sophisticated model with complex architecture can be applied, such as DIN[20] and DIEN[19], as the length of long-term behaviors has been reduced to hundreds.

Note that although we introduce the two stages separately, actually they are trained together.


#### 3.1 整体工作流程

SIM 的整体工作流程如图 1 所示。SIM 遵循一个级联的两阶段搜索策略，包含两个对应的单元：通用搜索单元（General Search Unit, GSU）和精确搜索单元（Exact Search Unit, ESU）。

**在第一阶段**，我们利用通用搜索单元（GSU）从原始的长期行为序列中以亚线性时间复杂度寻找前 K 个相关的子行为序列。这里 K 通常远小于行为序列的原始长度。如果在时间和计算资源的限制下可以搜索到相关行为，则可以进行高效的搜索方法。在第 3.2 节中，我们提供了 GSU 的两种简单实现：软搜索和硬搜索。GSU 采取了一种通用但有效的策略来缩短原始序列行为的长度，以满足时间和计算资源的严格限制。
与此同时，存在于长期用户行为序列中的大量噪声可能会削弱用户兴趣建模的效果，这些噪声可以通过第一阶段的搜索策略进行过滤。

**在第二阶段**，引入了精确搜索单元（Exact Search Unit, ESU），该单元以过滤后的子序列用户行为作为输入，进一步捕捉精确的用户兴趣。此时可以应用复杂架构的精细模型，如DIN[20]和DIEN[19]，因为长期行为的长度已经减少到数百。

请注意，尽管我们分别介绍了这两个阶段，实际上它们是一起训练的。

### 3.2 General Search Unit

Given a candidate item (the target item to be scored by CTR model), only a part of user behaviors are valuable. This part of user behaviors are closely related to final user decision. Picking out these relevant user behaviors is helpful in user interest modeling. However, using the whole user behavior sequence to directly model the user interest will bring enormous resource usage and response latency, which is usually unacceptable in practical applications. To this end, we propose a general search unit to cut down the input number of user behaviors in user interest modeling. Here we introduce two kinds of general search unit: hard-search and soft-search.

Given the list of user behaviors B = [b₁; b₂; ⋯ ; bₜ], where bᵢ is the i-th user behavior and T is the length of user behaviors. The general search unit can calculate relevant score rᵢ for each behavior bᵢ with the candidate item and then select the Top-K relevant behaviors with score rᵢ as sub behavior sequence B*. The difference between hard-search and soft-search is the formulation of relevant score rᵢ:

![image.png](https://s2.loli.net/2024/11/11/AGxpevWoZdaMUPm.jpg)

**Hard-search**. The hard-search model is non-parametric. Only behavior belongs to the same category as the candidate item will be selected and aggregated as a sub behavior sequence to be sent to the exact search unit. Here Cₐ and Cᵢ denote the categories of target item and the i-th behavior bᵢ that belong to correspondingly. Hard-search is straightforward but later in section 4 we will show how it is quite suitable for online serving.

**Soft-search**. In the soft-search model, bᵢ is first encoded as one-hot vector and then embedded into low-dimensional vectors E = [e₁; e₂; ⋯ ; eₜ], as shown in Figure 1. Wᵦ and Wᵥ are the parameters of weight. eₐ and eᵢ denote the embedding vectors of target item and i-th behavior bᵢ, respectively. To further speed up the top-K search over ten thousands length of user behaviors, sublinear time maximum inner product search method ALSH[12] is conducted based on the embedding vectors E to search the related top-K behaviors with target item. With the well-trained embedding and Maximum Inner Product Search (MIPS) method, over ten thousands of user behaviors could be reduced to hundreds.

It should be noticed that distributions of long-term and short-term data are different. Thus, directly using the parameters learned from short-term user interest modeling in soft-search model may mislead the long-term user interest modeling. In this paper, the parameters of soft-search model is trained under an auxiliary CTR prediction task based on long-term behavior data, illustrated as soft search training in the left of Figure 1. The behaviors representation Uᵣ is obtained by multiplying the rᵢ and eᵢ:

![image.png](https://s2.loli.net/2024/11/11/9SzB3TkaOn2IKAV.jpg)

The behaviors representation Uᵣ and the target vector eₐ are then concatenated as the input of following MLP (Multi-Layer Perception). Note that if the user behavior grows to a certain extent, it is impossible to directly fed the whole user behaviors into the model. In that situation, one can randomly sample sets of sub-sequence from the long sequential user behaviors, which still follows the same distribution of the original one.


### 3.2 通用搜索单元

给定一个候选项（即CTR模型中要评分的目标项），只有部分用户行为是有价值的。这部分用户行为与最终用户决策密切相关。挑选出这些相关的用户行为有助于用户兴趣建模。然而，使用整个用户行为序列直接建模用户兴趣会带来巨大的资源消耗和响应延迟，这在实际应用中通常是不可接受的。为此，我们提出了一个通用搜索单元，以减少用户兴趣建模中输入的用户行为数量。在这里，我们引入了两种通用搜索单元：硬搜索和软搜索。

给定用户行为列表 B = [b₁; b₂; ⋯ ; bₜ]，其中 bᵢ 是第 i 个用户行为，T 是用户行为的长度。通用搜索单元可以计算每个行为 bᵢ 和候选项之间的相关性得分 rᵢ，然后选择得分 rᵢ 最高的 Top-K 个相关行为作为子行为序列 B*。硬搜索和软搜索的区别在于相关性得分 rᵢ 的计算公式：

![image.png](https://s2.loli.net/2024/11/11/AGxpevWoZdaMUPm.jpg)

**硬搜索**。硬搜索模型是非参数的。只有属于与候选项相同类别的行为会被选中并汇总为一个子行为序列，然后发送到精确搜索单元。这里，Cₐ 和 Cᵢ 分别表示目标项和第 i 个行为 bᵢ 所属于的类别。硬搜索实现简单，但在第4节中我们将展示其在在线服务中的适用性。

**软搜索**。在软搜索模型中，首先将 bᵢ 编码为一个独热向量，然后嵌入到低维向量 E = [e₁; e₂; ⋯ ; eₜ] 中，如图1所示。Wᵦ 和 Wᵥ 是权重参数。eₐ 和 eᵢ 分别表示目标项和第 i 个行为 bᵢ 的嵌入向量。为了加速在成千上万个用户行为中的Top-K搜索，基于嵌入向量 E 使用亚线性时间的最大内积搜索方法 ALSH[12] 来搜索与目标项相关的 Top-K 行为。通过训练良好的嵌入和最大内积搜索（MIPS）方法，可以将成千上万的用户行为减少到几百个。

需要注意的是，长期和短期数据的分布是不同的。因此，直接在软搜索模型中使用从短期用户兴趣建模中学习到的参数可能会误导长期用户兴趣建模。在本文中，软搜索模型的参数是在一个基于长期行为数据的辅助CTR预测任务下训练的，如图1左侧所示。行为表示 Uᵣ 通过 rᵢ 和 eᵢ 相乘得到：

![image.png](https://s2.loli.net/2024/11/11/9SzB3TkaOn2IKAV.jpg)

然后，将行为表示 Uᵣ 和目标向量 eₐ 连接起来，作为后续多层感知器（MLP）的输入。请注意，如果用户行为增长到一定程度，就不可能将所有用户行为直接输入模型。在这种情况下，可以从长序列用户行为中随机采样子序列集，其分布仍与原始数据相同。
