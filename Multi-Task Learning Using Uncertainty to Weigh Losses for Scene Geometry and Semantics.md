## Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics


| Name          | Affiliation           | Email                |
|---------------|-----------------------|-----------------------|
| Alex Kendall   | University of Cambridge | agk34@cam.ac.uk    |
| Yarin Gal    | University of Oxford     | yarin@cs.ox.ac.uk  |
| Roberto Cipolla | University of Cambridge | rc10001@cam.ac.uk  |

### Abstract

Numerous deep learning applications benefit from multi-task learning with multiple regression and classification objectives. In this paper we make the observation that the performance of such systems is strongly dependent on the relative weighting between each task's loss. Tuning these weights by hand is a difficult and expensive process, making multi-task learning prohibitive in practice. We propose a principled approach to multi-task deep learning which weighs multiple loss functions by considering the homoscedastic uncertainty of each task. This allows us to simultaneously learn various quantities with different units or scales in both classification and regression settings. We demonstrate our model learning per-pixel depth regression, semantic and instance segmentation from a monocular input image. Perhaps surprisingly, we show our model can learn multi-task weightings and outperform separate models trained individually on each task.

### 1. Introduction

Multi-task learning aims to improve learning efficiency and prediction accuracy by learning multiple objectives from a shared representation [7]. Multi-task learning is prevalent in many applications of machine learning - from computer vision [27] to natural language processing [11] to speech recognition [23].

We explore multi-task learning within the setting of visual scene understanding in computer vision. Scene understanding algorithms must understand both the geometry and semantics of the scene at the same time. This forms an interesting multi-task learning problem because scene understanding involves joint learning of various regression and classification tasks with different units and scales. Multi-task learning of visual scene understanding is of crucial importance in systems where long computation run-time is prohibitive, such as the ones used in robotics. Combining all tasks into a single model reduces computation and allows...

#### 2. Related Work

Multi-task learning aims to improve learning efficiency and prediction accuracy for each task, when compared to training a separate model for each task[40,5]. It can be considered an approach to inductive knowledge transfer which improves generalisation by sharing the domain information between complimentary tasks. It does this by using a shared representation to learn multiple tasks - what is learned from one task can help learn other tasks[7].

Fine-tuning[1, 36] is a basic example of multi-task learning, where we can leverage different learning tasks by considering them as a pre-training step. Other models alternate learning between each training task, for example in natural language processing[11]. Multi-task learning can also be used in a data streaming setting[40], or to prevent forgetting previously learned tasks in reinforcement learning[26]. It can also be used to learn unsupervised features from various data sources with an auto-encoder[35].

In computer vision there are many examples of methods for multi-task learning. Many focus on semantic tasks, such as classification and semantic segmentation[30] or classification and detection[38]. MultiNet[39] proposes an architecture for detection, classification and semantic segmentation. CrossStitch networks[34] explore methods to combine multi-task neural activations. Uhrig et al.[41] learn semantic and instance segmentations under a classification setting. Multi-task deep learning has also been used for geometry and regression tasks. [15] show how to learn semantic segmentation, depth and surface normals. PoseNet[25] is a model which learns camera position and orientation. UberNet[27] learns a number of different regression and classification tasks under a single architecture. In this work we are the first to propose a method for jointly learning depth regression, semantic and instance segmentation.

Like the model of [15], our model learns both semantic and geometry representations, which is important for scene understanding. However, our model learns the much harder task of instance segmentation which requires knowledge of both semantics and geometry. This is because our model must determine the class and spatial relationship for each pixel in each object for instance segmentation.
## 3. Multi Task Learning with Homoscedastic Uncertainty

Multi-task learning concerns the problem of optimising a model with respect to multiple objectives. It is prevalent in many deep learning problems. The naive approach to combining multi objective losses would be to simply perform a weighted linear sum of the losses for each individual task:

$$
L_{\text{total}} = \sum_i w_i L_i. \qquad\qquad\qquad\qquad (1)
$$

This is the dominant approach used by prior work[39, 38, 30, 41], for example for dense prediction tasks[27], for scene understanding tasks[15] and for rotation (in quaternions) and translation (in meters) for camera pose[25]. However, there are a number of issues with this method. Namely, model performance is extremely sensitive to weight selection, $w_i$, as illustrated in Figure 2. These weight hyper-parameters are expensive to tune, often taking many days for each trial. Therefore, it is desirable to find a more convenient approach which is able to learn the optimal weights.

More concretely, let us consider a network which learns to predict pixel-wise depth and semantic class from an input image. In Figure 2 the two boundaries of each plot show models trained on individual tasks, with the curves showing performance for varying weights $w_i$ for each task. We observe that at some optimal weighting, the joint network performs better than separate networks trained on each task individually (performance of the model in individual tasks is seen at both edges of the plot: $w=0$ and $w=1$). At nearby values to the optimal weight the network performs worse on one of the tasks. However, searching for these optimal weightings is expensive and increasingly difficult with large models with numerous tasks. Figure 2 also shows a similar result for two regression tasks; instance segmentation and depth regression. We next show how to learn optimal task weightings using ideas from probabilistic modelling.

### 3.1. Homoscedastic Uncertainty as Task-Dependent Uncertainty

In Bayesian modeling, there are two main types of uncertainty one can model[24].

- **Epistemic uncertainty** is uncertainty in the model, which captures what our model does not know due to lack of training data. It can be explained away with increased training data.
- **Aleatoric uncertainty** captures our uncertainty with respect to information which our data cannot explain. Aleatoric uncertainty can be explained away with the ability to observe all explanatory variables with increasing precision.

Aleatoric uncertainty can again be divided into two sub-categories.

- **Data-dependent or Heteroscedastic uncertainty** is aleatoric uncertainty which depends on the input data and is predicted as a model output.
- **Task-dependent or Homoscedastic uncertainty** is aleatoric uncertainty which is not dependent on the input data. It is not a model output, rather it is a quantity which stays constant for all input data and varies between different tasks. It can therefore be described as task-dependent uncertainty.

In a multi-task setting, we show that the task uncertainty captures the relative confidence between tasks, reflecting the uncertainty inherent to the regression or classification task. It will also depend on the task's representation or unit of measure. We propose that we can use homoscedastic uncertainty as a basis for weighting losses in a multi-task learning problem.

### 3.2. Multi-task Likelihoods

In this section, we derive a multi-task loss function based on maximizing the Gaussian likelihood with homoscedastic uncertainty. Let \( f^W(x) \) be the output of a neural network with weights \( W \) on input \( x \). We define the following probabilistic model. For regression tasks, we define our likelihood as a Gaussian with mean given by the model output:

$$  p(y|f^W(x)) = \mathcal{N}(f^W(x), \sigma^2) \quad \qquad(2)  $$

with an observation noise scalar \( \sigma \). For classification, we often squash the model output through a softmax function, and sample from the resulting probability vector:

$$ p(y|f^W(x)) = \text{Softmax}(f^W(x)) \quad \qquad(3) $$

In the case of multiple model outputs, we often define the likelihood to factorize over the outputs, given some sufficient statistics. We define \( f^W(x) \) as our sufficient statistics, and obtain the following multi-task likelihood:

$$ p(y_1, \ldots, y_K|f^W(x)) = p(y_1|f^W(x)) \cdots p(y_K|f^W(x)) \quad (4) $$

with model outputs \( y_1, \ldots, y_K \) (such as semantic segmentation, depth regression, etc.).

In maximum likelihood inference, we maximize the log likelihood of the model. In regression, for example, the log likelihood can be written as:

$$ \log p(y|f^W(x)) \propto -\frac{1}{2\sigma^2} \|y - f^W(x)\|^2 - \log \sigma \quad (5) $$

for a Gaussian likelihood (or similarly for a Laplace likelihood) with $\sigma$ the model's observation noise parameter—capturing how much noise we have in the outputs. We then maximize the log likelihood with respect to the model parameters $W$ and observation noise parameter  $\sigma$ .

Let us now assume that our model output is composed of two vectors $y_1$ and $y_2$, each following a Gaussian distribution:

$$ p(y_1, y_2|f^W(x)) = p(y_1|f^W(x)) \cdot p(y_2|f^W(x)) = \mathcal{N}(y_1; f^W(x), \sigma_1^2) \cdot \mathcal{N}(y_2; f^W(x), \sigma_2^2)  \qquad\qquad  (6) $$

This leads to the minimization objective, \( \mathcal{L}(W, \sigma_1, \sigma_2) \) (our loss) for our multi-output model:

$$-\log p(y_1, y_2|f^W(x)) \propto \frac{1}{2\sigma_1^2} \|y_1 - f^W(x)\|^2 + \frac{1}{2\sigma_2^2} \|y_2 - f^W(x)\|^2 + \log \sigma_1 \sigma_2 $$

$$= \frac{1}{2\sigma_1^2} \mathcal{L}_1(W) + \frac{1}{2\sigma_2^2} \mathcal{L}_2(W) + \log \sigma_1 \sigma_2                                        
\qquad\qquad(7) $$

Where we wrote $\mathcal{L}_1(W) = \|y_1 - f^W(x)\|^2$  for the loss of the first output variable, and similarly for  $\mathcal{L}_2(W)$

We interpret minimizing this last objective with respect to \( \sigma_1 \) and \( \sigma_2 \) as learning the relative weight of the losses $\mathcal{L}_{1}(W)$ and $\mathcal{L}_{2}(W)$ adaptively, based on the data. As $\sigma_{1}$ - the noise parameter for the variable $y_1$ - increases, we have that the weight of $\mathcal{L}_{1}(W)$ decreases. On the other hand, as the noise decreases, we have that the weight of the respective objective increases. The noise is discouraged from increasing too much (effectively ignoring the data) by the last term in the objective, which acts as a regulariser for the noise terms.

This construction can be trivially extended to multiple regression outputs. However, the extension to classification likelihoods is more interesting. We adapt the classification likelihood to squash a scaled version of the model output through a softmax function:

$$
p(y \mid f^{W}(x), \sigma) = \text{Softmax}\left(\frac{1}{\sigma^2} f^{W}(x)\right) \quad (8)
$$

with a positive scalar $\sigma$. This can be interpreted as a Boltzmann distribution (also called Gibbs distribution) where the input is scaled by $\sigma^2$ (often referred to as temperature). This scalar is either fixed or can be learnt, where the parameter's magnitude determines how 'uniform' (flat) the discrete distribution is. This relates to its uncertainty, as measured in entropy. The log likelihood for this output can then be written as

$$
\begin{align*}
\log p\left(y=c\mid f^W(x),\sigma\right) &= \frac{1}{\sigma^2} f_c^W(x) \\
&\quad - \log\sum_{c^{\prime}}\exp\left(\frac{1}{\sigma^2} f_{c^{\prime}}^W(x)\right)
\end{align*}
$$

with $f_{c}^{W}(x)$ the c'th element of the vector $f^{W}(x)$.

Next, assume that a model's multiple outputs are composed of a continuous output $y_1$ and a discrete output $y_{2}$, modelled with a Gaussian likelihood and a softmax likelihood, respectively. Like before, the joint loss, $\mathcal{L}\left(W,\sigma_{1},\sigma_{2}\right)$, is given as:

$$
\mathcal{L}(W, \sigma_1, \sigma_2) = -\log p(y_1, y_2 = c \mid f^{W}(x)) 
$$

$$
= -\log\mathcal{N}(y_1; f^{W}(x), \sigma_1^2) \cdot \text{Softmax}(y_2 = c; f^{W}(x), \sigma_2) 
$$

$$
= \frac{1}{2\sigma_1^2} \left|y_1 - f^{W}(x)\right|^2 + \log \sigma_1 - \log p(y_2 = c \mid f^{W}(x), \sigma_2)
$$

$$
= \frac{1}{2\sigma_1^2} \mathcal{L}_1(W) + \frac{1}{\sigma_2^2} \mathcal{L}_2(W) + \log \sigma_1 
$$

$$
\quad + \log \frac{\sum_{c'} \exp \left( \frac{1}{\sigma_2^2} f_{c'}^W(x) \right)}{\left( \sum_{c'} \exp \left( f_{c'}^W(x) \right) \right)^{\frac{1}{\sigma_2^2}}}
$$

$$
\approx \frac{1}{2\sigma_1^2} \mathcal{L}_1(W) + \frac{1}{\sigma_2^2} \mathcal{L}_2(W) + \log \sigma_1 + \log \sigma_2
$$

where again we write $\mathcal{L}_1(W) = \|y_1 - f^W(x)\|^2$ for the Euclidean loss of $y_1$, and $\mathcal{L}_2(W)$ for the cross entropy loss of $y_2$ (with $f^W(x)$ not scaled), and optimize with respect to $W$ as well as $\sigma_1, \sigma_2$. In the last transition, we introduced the explicit simplifying assumption: 

$$
\frac{1}{\sigma_2} \sum_{c'} \exp \left( \frac{1}{\sigma_2^2} f_{c'}^W(x) \right) \approx \left( \sum_{c'} \exp \left( f_{c'}^W(x) \right) \right)^{\frac{1}{\sigma_2^2}}
$$

which becomes an equality when $\sigma_2 \rightarrow 1$. This has the advantage of simplifying the optimization objective, as well as empirically improving results.


This last objective can be seen as learning the relative weights of the losses for each output. Large scale values $\sigma_2$ will decrease the contribution of $\mathcal{L}_2(W)$, whereas small scale $\sigma_2$ will increase its contribution. The scale is regulated by the last term in the equation. The objective is penalized when setting $\sigma_2$ too large.

This construction can be trivially extended to arbitrary combinations of discrete and continuous loss functions, allowing us to learn the relative weights of each loss in a principled and well-founded way. This loss is smoothly differentiable, and is well formed such that the task weights will not converge to zero. In contrast, directly learning the weights using a simple linear sum of losses (1) would result in weights which quickly converge to zero. In the following sections, we introduce our experimental model and present empirical results.

In practice, we train the network to predict the log variance, $s := \log \sigma^2$. This is because it is more numerically stable than regressing the variance, $\sigma^2$, as the loss avoids any division by zero. The exponential mapping also allows us to regress unconstrained scalar values, where $\exp(-s)$ is resolved to the positive domain giving valid values for variance.

## 4. Scene Understanding Model

To understand semantics and geometry, we first propose an architecture which can learn regression and classification outputs, at a pixel level. Our architecture is a deep convolutional encoder-decoder network [3]. Our model consists of a number of convolutional encoders which produce a shared representation, followed by a corresponding number of task-specific convolutional decoders. A high-level summary is shown in Figure 1.

The purpose of the encoder is to learn a deep mapping to produce rich, contextual features, using domain knowledge from a number of related tasks. Our encoder is based on DeepLabV3 [10], which is a state-of-the-art semantic segmentation framework. We use ResNet101 [20] as the base feature encoder, followed by an Atrous Spatial Pyramid Pooling (ASPP) module [10] to increase contextual awareness. We apply dilated convolutions in this encoder, such that the resulting feature map is sub-sampled by a factor of 8 compared to the input image dimensions.

We then split the network into separate decoders (with separate weights) for each task. The purpose of the decoder is to learn a mapping from the shared features to an output. Each decoder consists of a 3 × 3 convolutional layer with output feature size 256, followed by a 1 × 1 layer regressing the task’s output. Further architectural details are described in Appendix A.

### Semantic Segmentation
We use the cross-entropy loss to learn pixel-wise class probabilities, averaging the loss over the pixels with semantic labels in each mini-batch.

### Instance Segmentation
An intuitive method for defining which instance a pixel belongs to is an association to the instance’s centroid. We use a regression approach for instance segmentation [29]. This approach is inspired by [28] which identifies instances using Hough votes from object parts. In this work, we extend this idea by using votes from individual pixels using deep learning. We learn an instance vector, $\hat{x}_n$, for each pixel coordinate, $c_n$, which points to the centroid of the pixel’s instance, $i_n$, such that

$$
i_n = \hat{x}_n + c_n
$$

We train this regression with an $L_1$ loss using ground truth labels $x_n$, averaged over all labelled pixels, $N_I$, in a mini-batch:

$$
L_{\text{Instance}} = \frac{1}{|N_I|} \sum_{n \in N_I} \left| x_n - \hat{x}_n \right|_1
$$


**Figure 3** details the representation we use for instance segmentation. **Figure 3(a)** shows the input image and a mask of the pixels which are of an instance class (at test time inferred from the predicted semantic segmentation). **Figure 3(b)** and **Figure 3(c)** show the ground truth and predicted instance vectors for both $x$ and $y$ coordinates. We then cluster these votes using OPTICS [2], resulting in the predicted instance segmentation output in **Figure 3(d)**.

One of the most difficult cases for instance segmentation algorithms to handle is when the instance mask is split due to occlusion. **Figure 4** shows that our method can handle these situations, by allowing pixels to vote for their instance centroid with geometry. Methods which rely on watershed approaches [4], or instance edge identification approaches fail in these scenarios.

To obtain segmentations for each instance, we now need to estimate the instance centres, $\hat{i}_n$. We propose to consider the estimated instance vectors, $\hat{x}_n$, as votes in a Hough parameter space and use a clustering algorithm to identify these instance centres. **OPTICS** [2] is an efficient density-based clustering algorithm. It is able to identify an unknown number of multi-scale clusters with varying density from a given set of samples. We chose **OPTICS** for two reasons. Crucially, it does not assume knowledge of the number of clusters like algorithms such as k-means [33]. Secondly, it does not assume a canonical instance size or density like discretized binning approaches [12]. Using **OPTICS**, we cluster the points $c_n + \hat{x}_n$ into a number of estimated instances, $i$. We can then assign each pixel, $p_n$, to the instance closest to its estimated instance vector, $c_n + \hat{x}_n$.

### Depth Regression
We train with supervised labels using pixel-wise metric inverse depth using an $L_1$ loss function:


$$
L_{\text{Depth}} = \frac{1}{|N_D|} \sum_{n \in N_D} | d_n - \hat{d}_n |_1
$$

Our architecture estimates inverse depth, $\hat{d}_n$, because it can represent points at infinite distance (such as sky). We can obtain inverse depth labels, $d_n$, from an RGBD sensor or stereo imagery. Pixels which do not have an inverse depth label are ignored in the loss.

## 5. Experiments

We demonstrate the efficacy of our method on CityScapes [13], a large dataset for road scene understanding. It comprises of stereo imagery, from automotive grade stereo cameras with a 22cm baseline, labelled with instance and semantic segmentations from 20 classes. Depth images are also provided, labelled using SGM [22], which we treat as pseudo ground truth. Additionally, we assign zero inverse depth to pixels labelled as sky. The dataset was collected from a number of cities in fine weather and consists of 2,975 training and 500 validation images at 2048 × 1024 resolution. 1,525 images are withheld for testing on an online evaluation server.

Further training details, and optimisation hyperparameters, are provided in Appendix A.

### 5.1. Model Analysis

In Table 1 we compare individual models to multi-task learning models using a naive weighted loss or the task uncertainty weighting we propose in this paper. To reduce the computational burden, we train each model at a reduced resolution of 128 × 256 pixels, over 50,000 iterations. When we downsample the data by a factor of four, we also need to scale the disparity labels accordingly. Table 1 clearly illustrates the benefit of multi-task learning, which obtains significantly better performing results than individual task models. For example, using our method we improve classification results from 59.4% to 63.4%.

We also compare to a number of naive multi-task losses. We compare weighting each task equally and using approximately optimal weights. Using a uniform weighting results in poor performance, in some cases not even improving on the results from the single task model. Obtaining approximately optimal weights is difficult with increasing number of tasks as it requires an expensive grid search over parameters. However, even these weights perform worse compared with our proposed method. Figure 2 shows that using task uncertainty weights can even perform better compared to optimal weights found through fine-grained grid search.

Second, optimising the task weights using a homoscedastic noise term allows for the weights to be dynamic during training. In general, we observe that the uncertainty term decreases during training which improves the optimisation process.

In Appendix B we find that our task-uncertainty loss is robust to the initialisation chosen for the parameters. These quickly converge to a similar optima in a few hundred training iterations. We also find the resulting task weightings varies throughout the course of training. For our final model (in Table 2), at the end of training, the losses are weighted with the ratio 43 : 1 : 0.16 for semantic segmentation, depth regression and instance segmentation, respectively.

Finally, we benchmark our model using the full-size CityScapes dataset. In Table 2 we compare to a number of other state of the art methods in all three tasks. Our method is the first model which completes all three tasks with a single model. We compare favourably with other approaches, outperforming many which use comparable training data and inference tools. Figure 5 shows some qualitative examples of our model.

## 6. Conclusions

We have shown that correctly weighting loss terms is of paramount importance for multi-task learning problems. We demonstrated that homoscedastic (task) uncertainty is an effective way to weight losses. We derived a principled loss function which can learn a relative weighting automatically from the data and is robust to the weight initialisation. We showed that this can improve performance for scene understanding tasks with a unified architecture for semantic segmentation, instance segmentation and per-pixel depth regression. We demonstrated modelling task-dependent homoscedastic uncertainty improves the model's representation and each task's performance when compared to separate models trained on each task individually.

There are many interesting questions left unanswered. Firstly, our results show that there is usually not a single optimal weighting for all tasks. Therefore, what is the optimal weighting? Is multitask learning an ill-posed optimisation problem without a single higher-level goal?

A second interesting question is where the optimal location is for splitting the shared encoder network into separate decoders for each task? And, what network depth is best for the shared multi-task representation?

Finally, why do the semantics and depth tasks outperform the semantics and instance tasks results in Table 1? Clearly the three tasks explored in this paper are complimentary and useful for learning a rich representation about the scene. It would be beneficial to be able to quantify the relationship between tasks and how useful they would be for multitask representation learning.

