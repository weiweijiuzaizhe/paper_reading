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
## 1 介绍：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
Click-Through Rate (CTR) prediction modeling plays a critical role in industrial applications such as recommender systems and online advertising. Due to the rapid growth of user historical behavior data, user interest modeling, which focuses on learning the intent representation of user interest, has been widely introduced in the CTR prediction model [2, 8, 19, 20]. However, most of the proposed approaches can only model sequential user behavior data with length scaling up to hundreds, limited by the burden of computation and storage in real online systems [19, 20]. Rich user behavior data is proven to be of great value [8]. For example, 23% of users in Taobao, one of the world’s leading e-commerce site, click with more than 1000 products in last 5 months[8, 10]. How to design a feasible solution to model the long sequential user behavior data has been an open and hot topic, attracting researchers from both industry and academy.

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










