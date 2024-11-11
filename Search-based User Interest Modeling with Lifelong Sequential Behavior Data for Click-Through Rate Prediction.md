
## 心得
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
MIMN 被反复提起



## 摘要
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
丰富的用户行为数据在点击率预测任务中被证明具有极大的价值，尤其是在推荐系统和在线广告等工业应用中。无论是业界还是学术界，都对这一主题给予了大量关注，并提出了多种长序列用户行为数据建模的方法。其中，阿里巴巴提出的基于记忆网络的模型MIMN[8]，通过学习算法与服务系统的联合设计达到了当前最优（SOTA）的效果。MIMN是第一个能够处理长度扩展到1000的用户行为序列数据的工业解决方案。然而，当用户行为序列长度进一步增加，比如扩大到10倍或更多时，MIMN在准确捕捉特定候选项对应的用户兴趣方面表现不佳。这一挑战在此前提出的多种方法中普遍存在。     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
在本文中，我们通过设计一种新的建模范式来解决这一问题，该方法被称为基于搜索的兴趣模型（Search-based Interest Model, SIM）。SIM通过两个级联的搜索单元提取用户兴趣：（i）通用搜索单元（General Search Unit, GSU）利用候选项的查询信息，从原始且任意长的用户行为序列中进行通用搜索，获取与候选项相关的子用户行为序列（Sub user Behavior Sequence, SBS）；（ii）精确搜索单元（Exact Search Unit, ESU）建模候选项与SBS之间的精确关系。该级联搜索范式使得SIM在可扩展性和精确性方面具备了更强的能力，从而能够更好地建模终身的用户行为数据序列。除了学习算法外，我们还介绍了如何在大规模工业系统中实现SIM的实践经验。自2019年以来，SIM已部署在阿里巴巴的展示广告系统中，带来了7.1%的CTR提升和4.4%的RPM提升，这对业务来说是非常显著的。如今，SIM已在实际系统中服务主流流量，能够处理长度达到54000的用户行为数据序列，是当前最优水平的54倍

## 1 介绍：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
点击率（CTR）预测建模在推荐系统和在线广告等工业应用中起着关键作用。随着用户历史行为数据的迅速增长，专注于学习用户兴趣意图表示的用户兴趣建模已广泛引入CTR预测模型中 [2, 8, 19, 20]。然而，由于实时在线系统中计算和存储的负担，大多数提出的方法只能处理长度最多为几百的用户行为序列数据 [19, 20]。丰富的用户行为数据被证明具有极高的价值 [8]。例如，在全球领先的电商平台淘宝中，23%的用户在过去五个月中点击了超过1000个商品 [8, 10]。如何设计一个可行的解决方案来建模长序列的用户行为数据一直是一个开放且热门的话题，吸引了来自工业界和学术界的研究者的关注。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
有一类研究借鉴了自然语言处理（NLP）领域的思路，提出使用记忆网络来建模长序列用户行为数据，并取得了一些突破。阿里巴巴提出的MIMN[8]是一个典型的工作，通过学习算法和服务系统的联合设计实现了当前最优（SOTA）的效果。MIMN是首个能够建模长度达1000的用户行为序列的工业解决方案。具体来说，MIMN将用户的多样兴趣逐步嵌入一个固定大小的记忆矩阵中，每次新的行为都会对该矩阵进行更新。这样，用户建模的计算与CTR预测解耦，因此在在线服务中，延迟不会成为问题，存储成本取决于记忆矩阵的大小，而非原始行为序列的长度。类似的想法也可见于长期兴趣建模中[10]。然而，对于基于记忆网络的方法而言，仍然难以处理任意长的序列数据。在实际应用中，我们发现当用户行为序列长度进一步增加到10000或更多时，MIMN无法准确捕捉特定候选项的用户兴趣。这是因为将所有用户的历史行为编码到一个固定大小的记忆矩阵中会导致记忆单元中包含大量噪声.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
另一方面，正如DIN[20]的前期研究所指出的，用户的兴趣是多样的，并会在面对不同的候选项时发生变化。DIN的关键思路是从用户行为中搜索有效信息，以建模用户在面对不同候选项时的特定兴趣。通过这种方式，我们可以应对将所有用户兴趣编码到固定大小参数中的挑战。DIN确实在结合用户行为数据的CTR建模中带来了显著的提升。然而，如上所述，DIN的搜索公式在处理长序列用户行为数据时带来了不可接受的计算和存储成本。那么，是否可以应用类似的搜索技巧，并设计一种更高效的方式从长序列用户行为数据中提取知识呢？


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nb sp;&nbsp; 
在本文中，我们通过设计一种新的建模范式来应对这一挑战，该方法被称为基于搜索的兴趣模型（Search-based Interest Model, SIM）。SIM借鉴了DIN[20]的思想，仅捕捉与特定候选项相关的用户兴趣。在SIM中，用户兴趣通过两个级联的搜索单元进行提取：（i）通用搜索单元（General Search Unit, GSU）从原始且任意长的用户行为序列中进行通用搜索，利用候选项的查询信息获取与候选项相关的子用户行为序列（Sub user Behavior Sequence, SBS）。为了满足延迟和计算资源的严格限制，GSU中采用了通用但有效的方法。根据我们的经验，SBS的长度可以减少到几百，并且大部分原始长序列行为数据中的噪声信息可以被过滤掉。（ii）精确搜索单元（Exact Search Unit, ESU）建模候选项与SBS之间的精确关系。在此我们可以轻松地应用DIN[20]或DIEN[19]提出的类似方法。


该工作的主要贡献总结如下：    
 - 我们提出了一种新的范式SIM用于建模长序列用户行为数据。级联的两阶段搜索机制设计使得SIM在可扩展性和准确性方面具备更强的能力，以更好地建模终身用户行为数据。
 - 我们介绍了在大规模工业系统中实现SIM的实践经验。自2019年以来，SIM已部署在阿里巴巴的展示广告系统中，带来了7.1%的CTR提升和4.4%的RPM提升。目前，SIM已服务于主要流量。
 - 我们将长序列用户行为数据建模的最大长度提升至54000，是现有发表的工业解决方案MIMN的54倍。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
## 2  相关工作 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
__用户兴趣模型.__ 基于深度学习的方法在CTR预测任务中取得了巨大成功[1, 11, 18]。在早期，大多数先驱性工作[1, 4, 7, 9, 15]使用深度神经网络来捕捉不同字段特征之间的交互，以便工程师可以摆脱繁琐的特征工程工作。最近，一系列我们称之为用户兴趣模型的工作，专注于通过历史行为学习潜在用户兴趣的表示，采用不同的神经网络架构，如CNN[14, 17]、RNN[5, 19]、Transformer[3, 13]和Capsule网络[6]等。
DIN[20]强调用户兴趣是多样的，并在DIN中引入了注意力机制，以捕捉用户对不同目标项的多样兴趣。DIEN[19]指出历史行为之间的时间关系对于建模用户的兴趣漂移非常重要，并在DIEN中设计了基于GRU的兴趣提取层，辅以辅助损失。MIND[6]认为，使用单一向量表示用户不足以捕捉用户兴趣的多变性，因此在MIND中引入了胶囊网络和动态路由方法，以将用户兴趣表示为多个向量。此外，受自注意力架构在序列到序列学习任务中成功的启发，Transformer在[3]中被引入，用于建模用户的跨会话和会话内兴趣。




&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
__长期用户兴趣__ [8] 表明，在用户兴趣模型中考虑长期历史行为序列可以显著提高CTR模型的性能。尽管较长的用户行为序列为用户兴趣建模带来了更多有用信息，但它极大地增加了在线服务系统的延迟和存储负担，同时对于点对点的CTR预测也包含了大量噪声。一系列工作致力于解决长期用户兴趣建模中的挑战，这些方法通常基于极长甚至终身的历史行为序列来学习用户兴趣表示。[10] 提出了一种分层周期记忆网络，用于终身序列建模，以为每个用户个性化地记忆其序列模式。[16] 选择了一种基于注意力的框架，结合用户的长期和短期偏好，并采用注意力不对称SVD范式来建模长期兴趣。[8] 提出了一种基于记忆的架构MIMN，它将用户的长期兴趣嵌入到固定大小的记忆网络中，以解决用户行为数据的大存储问题。同时设计了一个UIC模块，以增量记录新的用户行为来应对延迟限制。但MIMN在记忆网络中舍弃了被推荐物品的信息，而这一信息已被证明对用户兴趣建模至关重要。



## 3 搜索式兴趣模型
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
通过对用户行为数据进行CTR预测建模已被证明是有效的。通常，基于注意力的CTR模型（如DIN[20]和DIEN[19]）设计了复杂的模型结构，并引入注意力机制，通过从用户行为序列中搜索有效信息来捕捉用户的多样兴趣，借助不同候选项作为输入。然而，在真实系统中，这些模型只能处理短期行为序列数据，通常长度不超过150。另一方面，长期用户行为数据具有重要价值，对用户的长期兴趣进行建模可能会为用户带来更多样化的推荐结果。我们似乎陷入了两难境地：在真实系统中无法用有效但复杂的方法处理有价值的终身用户行为数据。
为了解决这一挑战，我们提出了一种新的建模范式，称为基于搜索的兴趣模型（Search-based Interest Model, SIM）。SIM遵循一个两阶段的搜索策略，能够高效地处理长用户行为序列。在本节中，我们将首先介绍SIM的整体工作流程，然后详细介绍所提出的两个搜索单元。



### 3.1 整体工作流程
<p align="center">
  <img src="https://s2.loli.net/2024/11/11/DMxZcYqsnQkfuLK.jpg" alt="image.png" width="2000"/>
</p>

图1：基于搜索的兴趣模型（SIM）。SIM采用两阶段搜索策略，包含两个单元：（i）通用搜索单元（General Search Unit, GSU），从超过一万条用户行为中找到最相关的K条行为；（ii）精确搜索单元（Exact Search Unit, ESU），利用多头注意力机制捕捉用户的多样兴趣。然后，它采用传统的嵌入和MLP（Embedding&MLP）范式，将精确的长期用户兴趣输出和其他特征作为输入。在本文中，我们为GSU引入了硬搜索和软搜索两种方法。硬搜索意味着选择与候选项属于同一类别的行为数据。软搜索则基于嵌入向量对每个用户行为数据进行索引，并使用最大内积搜索（MIPS）来寻找Top-K行为。在软搜索中，GSU和ESU共享相同的嵌入参数，这些参数在学习过程中同步训练，Top-K行为序列基于最新参数生成。

SIM的整体工作流程如图1所示。SIM遵循一个级联的两阶段搜索策略，包含两个对应的单元：通用搜索单元（General Search Unit, GSU）和精确搜索单元（Exact Search Unit, ESU）。

**在第一阶段**，我们利用通用搜索单元（GSU）从原始的长期行为序列中以亚线性时间复杂度寻找前K个相关的子行为序列。这里K通常远小于行为序列的原始长度。如果在时间和计算资源的限制下可以搜索到相关行为，则可以进行高效的搜索方法。在第3.2节中，我们提供了GSU的两种简单实现：软搜索和硬搜索。GSU采取了一种通用但有效的策略来缩短原始序列行为的长度，以满足时间和计算资源的严格限制。
与此同时，存在于长期用户行为序列中的大量噪声可能会削弱用户兴趣建模的效果，这些噪声可以通过第一阶段的搜索策略进行过滤。

**在第二阶段**，引入了精确搜索单元（Exact Search Unit, ESU），该单元以过滤后的子序列用户行为作为输入，进一步捕捉精确的用户兴趣。此时可以应用复杂架构的精细模型，如DIN[20]和DIEN[19]，因为长期行为的长度已经减少到数百。

请注意，尽管我们分别介绍了这两个阶段，实际上它们是一起训练的。

### 3.2 通用搜索单元

给定一个候选项（即CTR模型中要评分的目标项），只有部分用户行为是有价值的。这部分用户行为与最终用户决策密切相关。挑选出这些相关的用户行为有助于用户兴趣建模。然而，使用整个用户行为序列直接建模用户兴趣会带来巨大的资源消耗和响应延迟，这在实际应用中通常是不可接受的。为此，我们提出了一个通用搜索单元，以减少用户兴趣建模中输入的用户行为数量。在这里，我们引入了两种通用搜索单元：硬搜索和软搜索。

给定用户行为列表 B = [b₁; b₂; ⋯ ; bₜ]，其中 bᵢ 是第 i 个用户行为，T 是用户行为的长度。通用搜索单元可以计算每个行为 bᵢ 和候选项之间的相关性得分 rᵢ，然后选择得分 rᵢ 最高的 Top-K 个相关行为作为子行为序列 B*。硬搜索和软搜索的区别在于相关性得分 rᵢ 的计算公式：

<p align="center">
  <img src="https://s2.loli.net/2024/11/11/AGxpevWoZdaMUPm.jpg" alt="image.png" width="500"/>
</p>

**硬搜索**。硬搜索模型是非参数的。只有属于与候选项相同类别的行为会被选中并汇总为一个子行为序列，然后发送到精确搜索单元。这里，Cₐ 和 Cᵢ 分别表示目标项和第 i 个行为 bᵢ 所属于的类别。硬搜索实现简单，但在第4节中我们将展示其在在线服务中的适用性。

**软搜索**。在软搜索模型中，首先将 bᵢ 编码为一个独热向量，然后嵌入到低维向量 E = [e₁; e₂; ⋯ ; eₜ] 中，如图1所示。Wᵦ 和 Wᵥ 是权重参数。eₐ 和 eᵢ 分别表示目标项和第 i 个行为 bᵢ 的嵌入向量。为了加速在成千上万个用户行为中的Top-K搜索，基于嵌入向量 E 使用亚线性时间的最大内积搜索方法 ALSH[12] 来搜索与目标项相关的 Top-K 行为。通过训练良好的嵌入和最大内积搜索（MIPS）方法，可以将成千上万的用户行为减少到几百个。

需要注意的是，长期和短期数据的分布是不同的。因此，直接在软搜索模型中使用从短期用户兴趣建模中学习到的参数可能会误导长期用户兴趣建模。在本文中，软搜索模型的参数是在一个基于长期行为数据的辅助CTR预测任务下训练的，如图1左侧所示。行为表示 Uᵣ 通过 rᵢ 和 eᵢ 相乘得到：

<p align="center">
  <img src="https://s2.loli.net/2024/11/11/9SzB3TkaOn2IKAV.jpg" alt="image.png" width="500"/>
</p>

然后，将行为表示 Uᵣ 和目标向量 eₐ 连接起来，作为后续多层感知器（MLP）的输入。请注意，如果用户行为增长到一定程度，就不可能将所有用户行为直接输入模型。在这种情况下，可以从长序列用户行为中随机采样子序列集，其分布仍与原始数据相同。

### 3.3 精确搜索单元

在第一阶段搜索中，从长期用户行为中选择与目标项相关的前K个子用户行为序列 B*。为了进一步建模这些相关行为中的用户兴趣，我们引入了精确搜索单元（Exact Search Unit, ESU），它是一个以 B* 为输入的基于注意力的模型。

考虑到这些选定的用户行为跨越较长时间，因此每个行为的贡献是不同的，我们为每个行为加入了时间序列属性。具体而言，目标项和选定的K个用户行为之间的时间间隔 D = [Δ₁; Δ₂; ⋯; Δₖ] 用于提供时间距离信息。B* 和 D 分别被编码为嵌入 E* = [e₁*; e₂*; ⋯; eₖ*] 和 Eₜ = [e₁ᵗ; e₂ᵗ; ⋯; eₖᵗ]。eⱼ* 和 eⱼᵗ 被连接为用户行为的最终表示，表示为 zⱼ = concat(eⱼ*, eⱼᵗ)。我们利用多头注意力机制来捕捉多样的用户兴趣：

<p align="center">
  <img src="https://s2.loli.net/2024/11/11/hm9IxcC3fdzKra8.jpg" alt="image.png" width="500"/>
</p>

其中 ![image.png](https://s2.loli.net/2024/11/11/mpC59KrPwUueQVH.jpg) 是第 i 个注意力得分，headᵢ 是多头注意力中的第 i 个头。Wᵦᵢ 和 Wₐᵢ 是第 i 个权重参数。用户的最终长期多样兴趣表示为 Uₗₜ = concat(head₁; ⋯; headq)，然后输入到MLP中进行CTR预测。

最终，通用搜索单元和精确搜索单元在交叉熵损失函数下同时训练。
<p align="center">
  <img src="https://s2.loli.net/2024/11/11/hl1McTCFLt7oRqE.jpg" alt="image.png" width="500"/>
</p>
其中 α 和 β 是控制损失权重的超参数。在我们的实验中，如果GSU使用软搜索模型，则 α 和 β 都设为1。GSU使用硬搜索模型时为非参数化，因此 α 设为0。

## 4 实时服务的实现

在本节中，我们介绍了在阿里巴巴展示广告系统中实现SIM的实践经验。

<p align="center">
  <img src="https://s2.loli.net/2024/11/11/LcAXOlae3QJprxV.jpg" alt="image.png" width="2000"/>
</p>

图2：用于CTR任务的实时预测（RTP）系统在我们的工业展示广告系统中。它包含两个关键组件：计算节点和预测服务器。处理包含长序列用户行为数据的RTP在线系统会带来巨大的存储和延迟压力。

#### 4.1 终身用户行为数据在线服务的挑战

工业推荐系统或广告系统需要每秒处理大量流量请求，这要求CTR模型能够实时响应。通常，服务延迟应少于30毫秒。图2简要展示了我们在线展示广告系统中用于CTR任务的实时预测（RTP）系统。

在考虑终身用户行为的情况下，让长期用户兴趣模型在实时工业系统中运行更加困难。存储和延迟的限制可能成为长期用户兴趣模型的瓶颈[8]。随着用户行为序列长度的增加，流量会线性增长。此外，在流量高峰期，我们的系统每秒服务超过100万用户。因此，将长期模型部署到在线系统中是一项巨大的挑战。

#### 4.2 面向在线服务系统的基于搜索的兴趣模型

<p align="center">
  <img src="https://s2.loli.net/2024/11/11/hCFkGAPp5uKgB1V.jpg" alt="image.png" width="2000"/>
</p>

图3：采用提议的SIM模型的CTR预测系统。新系统增加了一个硬搜索模块，以从长序列行为数据中提取与目标项相关的有效行为。用户行为树的索引提前离线构建，从而节省了在线服务的大部分延迟成本。

在第3.2节中，我们提出了两种通用搜索单元，软搜索模型和硬搜索模型。对于软搜索和硬搜索模型，我们在阿里巴巴的工业数据上进行了广泛的离线实验，数据来自在线展示广告系统。我们观察到，软搜索模型生成的Top-K行为与硬搜索模型的结果极为相似。换句话说，软搜索中的大多数Top-K行为通常属于目标项的类别。这是我们场景中数据的特性。在电子商务网站中，同一类别的商品在大多数情况下是相似的。因此，尽管在离线实验中软搜索模型的表现略优于硬搜索模型（详情见表4），在平衡性能收益和资源消耗后，我们选择在广告系统中部署硬搜索模型来实现SIM。

对于硬搜索模型，包含所有长序列行为数据的索引是一个关键组件。我们观察到，可以根据行为所属类别自然地实现行为分类。因此，我们为每个用户构建了一个两级结构的索引，称为用户行为树（UBT），如图3所示。简而言之，UBT遵循Key-Key-Value数据结构：第一个键是用户ID，第二个键是类别，最后的值是属于该类别的具体行为项。UBT实现为一个分布式系统，大小达到22 TB，并且足够灵活，可以提供高吞吐量的查询。然后，我们将目标项的类别作为硬搜索查询。经过通用搜索单元处理后，用户行为的长度可以从超过一万条减少到几百条。因此，在线系统中终身行为的存储压力得以释放。图3展示了带有基于搜索兴趣模型的新的CTR预测系统。

请注意，通用搜索单元的用户行为树索引可以离线预先构建。这样一来，在线系统中通用搜索单元的响应时间可以非常短，并且相对于GSU的计算可以忽略不计。此外，其他用户特征可以并行计算。

### 5 实验

在本节中，我们详细介绍了实验过程，包括数据集、实验设置、模型比较及相关分析。所提出的搜索模型在两个公共数据集和一个工业数据集上与几种最先进的工作进行了比较，结果如表1所示。由于SIM已经部署在我们的在线广告系统中，我们还进行了严格的在线A/B测试，并与一些知名的行业模型进行了比较。

#### 5.1 数据集
**表1：本文使用的数据集统计信息**

| Dataset        | Users     | Items<sup>a</sup>     | Categories | Instances   |
|----------------|-----------|-----------------------|------------|-------------|
| Amazon (Book)  | 75053     | 358367                | 1583       | 150016      |
| Taobao         | 7956431   | 34196612              | 5597       | 7956431     |
| Industrial     | 0.29 billion | 0.6 billion       | 100,000    | 12.2 billion|

<sup>a</sup> 对于工业数据集，Items指的是广告。


模型比较在两个公共数据集以及从阿里巴巴在线展示广告系统收集的工业数据集上进行。表1展示了所有数据集的统计信息。

- **Amazon Dataset** 由亚马逊的产品评论和元数据组成。我们使用Amazon数据集的Books子集，其中包含75053名用户、358367个商品和1583个类别。在该数据集中，我们将评论视为一种交互行为，并按时间排序一个用户的所有评论。Amazon Book数据集的最大行为序列长度为100。我们将最近10条用户行为作为短期用户序列特征，将最近90条用户行为作为长期用户序列特征。这些预处理方法已广泛用于相关研究中。

- **Taobao Dataset** 是来自淘宝推荐系统的用户行为集合。该数据集包含了点击、购买等多种用户行为，约有八百万用户的行为序列。我们提取每个用户的点击行为并按时间排序，以构建行为序列。Taobao数据集的最大行为序列长度为500。我们将最近100条用户行为作为短期用户序列特征，将最近400条用户行为作为长期用户序列特征。该数据集将很快发布。

- **Industrial Dataset** 来自阿里巴巴的在线展示广告系统。样本由曝光日志构建，以“点击”或“未点击”作为标签。训练集由过去49天的样本组成，测试集来自接下来的1天，这是工业建模的经典设置。在该数据集中，每天样本中的用户行为特征包含了前180天的历史行为序列作为长期行为特征，以及前14天的行为作为短期行为特征。超过30%的样本包含长度超过10000的序列行为数据。此外，行为序列的最大长度达到54000，是MIMN [8]中的54倍。

#### 5.2 竞争对手和实验设置

我们将SIM与以下主流CTR预测模型进行了比较。

- **DIN** [20] 是用户行为建模的早期工作，提出了相对于候选项进行软搜索的用户行为建模方法。与其他长期用户兴趣模型相比，DIN仅将短期用户行为作为输入。

- **Avg-Pooling Long DIN** 为了比较模型在长期用户兴趣方面的性能，我们对长期行为应用平均池化操作，并将长期嵌入与其他特征嵌入连接。

#### 5.3 公共数据集上的结果
**表2：公共数据集上的模型性能（AUC）**

| Model                   | Taobao (mean ± std)      | Amazon (mean ± std)    |
|-------------------------|--------------------------|-------------------------|
| DIN                     | 0.9214 ± 0.00017         | 0.7276 ± 0.00051       |
| Avg-Pooling Long DIN    | 0.9281 ± 0.00025         | 0.7280 ± 0.00012       |
| MIMN                    | 0.9278 ± 0.00035         | 0.7396 ± 0.00037       |
| SIM (soft)<sup>a</sup>  | 0.9416 ± 0.00049         | 0.7510 ± 0.00052       |
| SIM (soft) with Timeinfo | **0.9501 ± 0.00017**    | -<sup>b</sup>          |

<sup>a</sup> SIM (soft) 是不包含时间间隔嵌入的软搜索SIM  
<sup>b</sup> 我们没有在Amazon数据集上进行实验，因为其中没有时间戳特征

---

**表3：两阶段搜索架构在长期用户兴趣建模上的效果评估**

| Operations                    | Taobao (mean ± std)      | Amazon (mean ± std)    |
|-------------------------------|--------------------------|-------------------------|
| Avg-Pooling without Search    | 0.9281 ± 0.00025         | 0.7280 ± 0.00012       |
| Only First Stage (hard)       | 0.9330 ± 0.00031         | 0.7365 ± 0.00022       |
| Only First Stage (soft)       | 0.9357 ± 0.00025         | 0.7342 ± 0.00012       |
| SIM (hard)                    | 0.9332 ± 0.00008         | 0.7413 ± 0.00016       |
| SIM (soft)                    | 0.9416 ± 0.00049         | 0.7510 ± 0.00052       |
| SIM (soft) with Timeinfo      | **0.9501 ± 0.00017**     | -                      |

表2展示了所有比较模型的结果。与DIN相比，其他采用长期用户行为特征的模型表现更佳，这表明长期用户行为对CTR预测任务有帮助。与MIMN相比，SIM显著提升了性能，因为MIMN将所有未经筛选的用户历史行为编码为固定长度的记忆，使得难以捕捉多样化的长期兴趣。SIM使用两阶段搜索策略，从大量历史行为序列中搜索相关行为，并根据不同的目标项建模多样化的长期兴趣。实验结果表明，SIM优于其他所有长期兴趣模型，强有力地证明了所提出的两阶段搜索策略在长期用户兴趣建模中的有效性。此外，引入时间嵌入还能进一步提升性能。

#### 5.4 消融研究

**两阶段搜索的有效性**。如上所述，所提出的兴趣搜索模型使用了两阶段搜索策略。第一阶段采用通用搜索策略，以过滤出与目标项相关的历史行为。第二阶段对第一阶段的行为进行基于注意力的精确搜索，以准确捕捉用户在目标项上的多样化长期兴趣。在本节中，我们通过对长期历史行为的不同操作实验来评估所提出的两阶段搜索架构的有效性。如表3所示，**Avg-Pooling without Search** 仅使用平均池化整合长期行为嵌入而不做任何过滤，类似于Avg-Pooling Long DIN。**Only First Stage (hard)** 是在第一阶段对长期历史行为进行硬搜索，并通过平均池化将过滤后的嵌入整合为固定大小的向量，作为MLP的输入。**Only First Stage (soft)** 与Only First Stage (hard)基本相同，但在第一阶段使用的是参数化的软搜索而非硬搜索。在第三个实验中，我们基于预训练的嵌入向量计算目标项与长期用户行为之间的内积相似度得分，并根据相似度得分选择Top 50相关行为进行软搜索。最后三个实验是带有两阶段搜索架构的提出的搜索模型。

如表3所示，使用过滤策略的所有方法相比简单平均池化嵌入极大地提升了模型性能。这表明原始长期行为序列中确实存在大量噪声，可能会削弱长期用户兴趣学习。与仅使用一阶段搜索的模型相比，所提出的两阶段搜索策略通过在第二阶段引入基于注意力的搜索取得了进一步进展。这表明精确建模用户在目标项上的多样化长期兴趣对CTR预测任务是有帮助的。而且经过第一阶段搜索后，过滤后的行为序列通常比原始序列短得多，注意力操作不会对实时RTP系统造成负担。

引入时间嵌入可以进一步提升性能，这表明用户行为在不同时间段的贡献是不同的。

<p align="center">
  <img src="https://s2.loli.net/2024/11/11/t4YIzrA7NGwmfjT.jpg" alt="image.png" width="500"/>
</p>

**图4：来自DIEN和SIM的点击样本分布**。点击被分为两部分：长期（>14天）和短期（≤14天），并根据提出的同类别行为上次发生的天数（d<sub>category</sub>）指标进行聚合。短期的聚合单位为2天，长期的聚合单位为20天。方框展示了SIM在不同d<sub>category</sub>上的点击比例提升。

#### 5.5 工业数据集上的结果

**表4：工业数据集上的模型性能（AUC）**

| Model                    | AUC     |
|--------------------------|---------|
| DIEN                     | 0.6452  |
| MIMN                     | 0.6541  |
| SIM (hard)               | 0.6604  |
| SIM (soft)               | 0.6625  |
| SIM (hard) with timeinfo<sup>a</sup> | 0.6624  |

<sup>a</sup> 该模型已部署在我们的在线服务系统中，并正在服务主要流量。


我们进一步在从阿里巴巴在线展示广告系统收集的数据集上进行了实验。表4显示了结果。与第一阶段的硬搜索相比，软搜索表现更好。同时，我们注意到在第一阶段两种搜索策略之间只有轻微的差距。第一阶段应用软搜索需要更多的计算和存储资源，因为在线服务会利用最近邻搜索方法，而硬搜索只需从一个离线构建的两级索引表中进行搜索。因此，硬搜索更加高效和系统友好。此外，对于两种不同的搜索策略，我们对工业数据集中超过100万个样本和10万名拥有长期历史行为的用户进行了统计。结果显示，通过硬搜索策略搜索到的用户行为可以覆盖软搜索策略的75%。最终，为了在效率和性能之间取得平衡，我们在第一阶段选择了更简单的硬搜索策略。SIM相较于MIMN提升了0.008的AUC，这对于我们的业务而言是显著的。

**在线A/B测试**

**表5：从2020年1月7日至2020年2月7日，SIM与MIMN在淘宝APP首页“猜你想要”栏目中的线上表现提升率**

| Metric     | CTR   | RPM   |
|------------|-------|-------|
| 提升率     | 7.1%  | 4.4%  |

自2019年以来，我们已在阿里巴巴的展示广告系统中部署了所提出的解决方案。从2020年1月7日至2020年2月7日，我们进行了严格的在线A/B测试实验，以验证所提出的SIM模型。与MIMN（我们的上一代产品模型）相比，SIM在阿里巴巴展示广告场景中取得了巨大提升，结果如表5所示。现在，SIM已在线部署并每天服务主要场景流量，为业务收入增长作出了重要贡献。

**重新思考搜索模型** 我们在用户长期兴趣建模上投入了大量精力，所提出的SIM在离线和在线评估中都表现良好。但是，SIM是否因精确的长期兴趣建模而表现更佳？SIM是否更倾向于推荐与用户长期兴趣相关的商品？为了回答这两个问题，我们制定了一个新的指标。**同类别行为上次发生的天数**（dₐcₐₜₑgₒᵣy）定义为点击样本的点击事件发生之前，用户在相同类别的商品上产生上次行为的天数。例如，用户u₁点击了类别为c₁的商品i₁，且用户u₁在5天前点击了另一个同类别的商品j，则点击事件s₃的dₐcₐₜₑgₒᵣy为5。如果用户u₁从未在类别c₁上产生过行为，我们将dₐcₐₜₑgₒᵣy设为-1。对于特定模型，dₐcₐₜₑgₒᵣy可用于评估模型在长期或短期兴趣上的选择偏好。

在在线A/B测试后，我们根据提出的dₐcₐₜₑgₒᵣy指标分析了SIM和DIEN（上一版本短期CTR预测模型）的点击样本。图4展示了点击在dₐcₐₜₑgₒᵣy上的分布。可以发现，在短期部分（dₐcₐₜₑgₒᵣy < 14）两种模型几乎没有差异，因为SIM和DIEN在最近14天内均包含了短期用户行为特征。而在长期部分，SIM占据了更大比例。此外，我们统计了工业数据集上dₐcₐₜₑgₒᵣy的平均值以及用户在目标项上有历史类别行为的概率p(dₐcₐₜₑgₒᵣy > -1)，结果如表6所示。工业数据集的统计结果证明了SIM的改进确实来源于更好的长期兴趣建模，且与DIEN相比，SIM更倾向于推荐与用户长期行为相关的商品。

**表6：工业数据集中d<sub>category</sub>的统计信息**

| Model | Average d<sub>category</sub> | p(d<sub>category</sub> > -1) |
|-------|------------------------------|------------------------------|
| DIEN  | 11.2                         | 0.91                         |
| SIM   | 13.3                         | 0.94                         |

**部署的实践经验** 这里我们介绍在在线服务系统中实现SIM的实践经验。众所周知，阿里巴巴的高流量场景在高峰时每秒服务超过100万用户。此外，对于每个用户，RTP系统需要计算数百个候选项的CTR预测。我们为所有用户行为数据离线构建了两级索引，并将其每日更新。第一级索引为用户ID，第二级索引为用户终身行为数据按类别进行索引。尽管候选项的数量是数百个，但这些项的类别通常少于20。同时，GSU为每个类别截断的子行为序列长度为200（原始长度通常小于150）。这样，每个用户请求的流量被限制在可接受的范围内。此外，我们通过深度核融合优化了ESU中多头注意力的计算。

<p align="center">
  <img src="https://s2.loli.net/2024/11/11/o4BZbsCEL71DuAn.jpg" alt="image.png" width="500"/>
</p>

图5：不同吞吐量下实时CTR预测系统的系统性能。MIMN和DIEN的用户行为长度截断至1000，而SIM的用户行为长度可以扩展到上万。DIEN的最大吞吐量为200，因此图中仅有一个点。

图5展示了我们实时CTR预测系统在不同吞吐量下的延迟性能，分别对比了DIEN、MIMN和SIM。值得注意的是，MIMN能处理的最大用户行为长度为1000，所展示的性能基于截断后的行为数据。而在SIM中用户行为长度未被截断，可扩展到54000，将最大长度提升至54倍。SIM在处理上万条行为时与MIMN处理截断后的用户行为相比，延迟仅增加5毫秒。

### 6 结论

在本文中，我们致力于在真实工业环境中利用超过数万条的用户行为序列数据。我们提出了基于搜索的兴趣模型（SIM），以捕捉用户在目标项上的多样化长期兴趣。在第一阶段中，我们提出了一个通用搜索单元，将数万条行为缩减至几百条。在第二阶段，一个精确搜索单元利用这些相关的行为来建模精确的用户兴趣。我们已在阿里巴巴的展示广告系统中实现了SIM。SIM带来了显著的业务提升，并正在服务主要流量。

SIM相比以往的方法引入了更多的用户行为数据，实验结果表明，SIM更加关注长期兴趣。然而，搜索单元在所有用户中共享相同的公式和参数。未来，我们将尝试构建用户特定的模型，以根据个人偏好来组织每个用户的终身行为数据。这样，每个用户将拥有自己的个性化模型，以持续建模其不断变化的兴趣。

### REFERENCES

[1] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In *Proceedings of the 1st Workshop on Deep Learning for Recommender Systems.* ACM, 7–10.

[2] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In *Proceedings of the 10th ACM Conference on Recommender Systems.* ACM, 191–198.

[3] Yufei Feng, Fuyu Lv, Weichen Shen, Menghan Wang, Fei Sun, Yu Zhu, and Keping Yang. 2019. Deep session interest network for click-through rate prediction. *arXiv preprint arXiv:1905.06482* (2019).

[4] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. Deepfm: a factorization-machine based neural network for ctr prediction. In *Proceedings of the 26th International Joint Conference on Artificial Intelligence.* Melbourne, Australia., 2782–2788.

[5] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2015. Session-based recommendations with recurrent neural networks. *arXiv preprint arXiv:1511.06939* (2015).

[6] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-interest network with dynamic routing for recommendation at Tmall. In *Proceedings of the 28th ACM International Conference on Information and Knowledge Management.*, 2615–2623.

[7] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.* London, United Kingdom.

[8] Qi Pi, Weijie Bian, Guorui Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Practice on Long Sequential User Behavior Modeling for Click-through Rate Prediction. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.* ACM, 1059–1068.

[9] Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang. 2016. Product-Based Neural Networks for User Response Prediction. (2016), 1149–1154.

[10] Kan Ren, Jiarui Qin, Yuchen Fang, Weinan Zhang, Lei Zheng, Weijie Bian, Guorui Zhou, Jian Xu, Yong Yu, Xiaoqiang Zhu, et al. 2019. Lifelong Sequential Modeling with Personalized Memorization for User Response Prediction. In *Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval.*, 565–574.

[11] Ying Shan, T Ryan Hoens, Jian Jiao, Haijung Wang, Dong Yu, and JC Mao. [n.d.]. Deep Crossing: Web-scale modeling without manually crafted combinatorial features.

[12] Anshumali Shrivastava and Ping Li. 2014. Asymmetric LSH (ALSH) for sublinear time maximum inner product search (MIPS). In *Advances in Neural Information Processing Systems.*, 2321–2329.

[13] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In *Proceedings of the 28th ACM International Conference on Information and Knowledge Management.*, 1441–1450.

[14] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In *Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining.*, 565–573.

[15] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & Cross Network for Ad Click Predictions. (2017), 12.

[16] Zeping Yu, Jianxun Lian, Ahmad Mahmood, Gongshen Liu, and Xing Xie. 2019. Adaptive user modeling with long and short-term preferences for personalized recommendation. In *Proceedings of the 28th International Joint Conference on Artificial Intelligence.* AAAI Press, 4213–4219.

[17] Fajie Yuan, Alexandros Karatzoglou, Ioannis Arapakis, Joemon M Jose, and Xiangnan He. 2019. A simple convolutional generative network for next item recommendation. In *Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining.*, 582–590.

[18] Shuangfei Zhai, Keng-hao Chang, Ruofei Zhang, and Zhongfei Mark Zhang. 2016. Deepintent: Learning attentions for online advertising with recurrent neural networks. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.* ACM, 1295–1304.

[19] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep interest evolution network for click-through rate prediction. In *Proceedings of the 33nd AAAI Conference on Artificial Intelligence.* Honolulu, USA.

[20] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.* ACM, 1059–1068.
