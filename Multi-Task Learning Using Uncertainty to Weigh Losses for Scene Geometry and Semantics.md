# Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics

**Alex Kendall**  
University of Cambridge  
agk34@cam.ac.uk  

**Yarin Gal**  
University of Oxford  
yarin@cs.ox.ac.uk  

**Roberto Cipolla**  
University of Cambridge  
rc10001@cam.ac.uk  

---

## Abstract

Numerous deep learning applications benefit from multi-task learning with multiple regression and classification objectives. In this paper we make the observation that the performance of such systems is strongly dependent on the relative weighting between each task’s loss. Tuning these weights by hand is a difficult and expensive process, making multi-task learning prohibitive in practice. We propose a principled approach to multi-task deep learning which weighs multiple loss functions by considering the *homoscedastic uncertainty of each task*. This allows us to simultaneously learn various quantities with different units or scales in both classification and regression settings. We demonstrate our model learning per-pixel depth regression, semantic and instance segmentation from a monocular input image. Perhaps surprisingly, we show our model can learn multi-task weightings and outperform separate models trained individually on each task.

---

## 1. Introduction

Multi-task learning aims to improve learning efficiency and prediction accuracy by learning multiple objectives from a shared representation [7]. Multi-task learning is prevalent in many applications of machine learning – from computer vision [27] to natural language processing [11] to speech recognition [23].

We explore multi-task learning within the setting of visual scene understanding in computer vision. Scene understanding algorithms must understand both the geometry and semantics of the scene at the same time. This forms an interesting multi-task learning problem because scene understanding involves joint learning of various regression and classification tasks with different units and scales. Multi-task learning of visual scene understanding is of crucial importance in systems where long computation run-time is prohibitive, such as the ones used in robotics. Combining all tasks into a single model reduces computation and allows these systems to run in real-time.

Prior approaches to simultaneously learning multiple tasks use a naïve weighted sum of losses, where the loss weights are uniform, or manually tuned [38, 27, 15]. However, we show that performance is highly dependent on an appropriate choice of weighting between each task’s loss. Searching for an optimal weighting is prohibitively expensive and difficult to resolve with manual tuning. We observe that the optimal weighting of each task is dependent on the measurement scale (e.g., meters, centimeters or millimeters) and ultimately the magnitude of the task’s noise.

In this work we propose a principled way of combining multiple loss functions to simultaneously learn multiple objectives using homoscedastic uncertainty. We interpret *homoscedastic uncertainty* as task-dependent weighting and show how to derive a principled multi-task loss function which can learn to balance various regression and classification losses. Our method can learn to balance these weightings optimally, resulting in superior performance, compared with learning each task individually.

Specifically, we demonstrate our method in learning scene geometry and semantics with three tasks. Firstly, we learn to classify objects at a pixel level, also known as semantic segmentation [32, 3, 42, 8, 45]. Secondly, our model performs instance segmentation, which is the harder task of segmenting separate masks for each individual object in an image (for example, a separate, precise mask for each individual car on the road) [37, 18, 14, 4]. This is a more difficult task than semantic segmentation, as it requires not only an estimate of each pixel’s class, but also which object that pixel belongs to. It is also more complicated than object detection, which often predicts object bounding boxes alone [17]. Finally, our model predicts pixel-wise metric depth. Depth by recognition has been demonstrated using dense prediction networks with supervised [15] and unsupervised [16] deep learning. However it is very hard to estimate depth in a way which generalises well. We show that we can improve our estimation of geometry and depth by using semantic labels and multi-task deep learning.

In existing literature, separate deep learning models would be used to learn depth regression, semantic segmentation and instance segmentation to create a complete scene understanding system. Given a single monocular input image, our system is the first to produce a semantic segmentation, a dense estimate of metric depth and an instance level segmentation jointly (Figure 1). While other vision models have demonstrated multi-task learning, we show how to learn to combine semantics and geometry. Combining these tasks into a single model ensures that the model agrees between the separate task outputs while reducing computation. Finally, we show that using a shared representation with multi-task learning improves performance on various metrics, making the models more effective.

In summary, the key contributions of this paper are:

1. a novel and principled multi-task loss to simultaneously learn various classification and regression losses of varying quantities and units using homoscedastic task uncertainty,
2. a unified architecture for semantic segmentation, instance segmentation and depth regression,
3. demonstrating the importance of loss weighting in multi-task deep learning and how to obtain superior performance compared to equivalent separately trained models.

---

## 2. Related Work

Multi-task learning aims to improve learning efficiency and prediction accuracy for each task, when compared to training a separate model for each task [40, 5]. It can be considered an approach to inductive knowledge transfer which improves generalisation by sharing the domain information between complimentary tasks. It does this by using a shared representation to learn multiple tasks – what is learned from one task can help learn other tasks [7].

Fine-tuning [1, 36] is a basic example of multi-task learning, where we can leverage different learning tasks by considering them as a pre-training step. Other models alternate learning between each training task, for example in natural language processing [11]. Multi-task learning can also be used in a data streaming setting [40], or to prevent forgetting previously learned tasks in reinforcement learning [26]. It can also be used to learn unsupervised features from various data sources with an auto-encoder [35].

In computer vision there are many examples of methods for multi-task learning. Many focus on semantic tasks, such as classification and semantic segmentation [30] or classification and detection [38]. MultiNet [39] proposes an architecture for detection, classification and semantic segmentation. CrossStitch networks [34] explore methods to combine multi-task neural activations. Uhrig et al. [41] learn semantic and instance segmentations under a classification setting. Multi-task deep learning has also been used for geometry and regression tasks. [15] show how to learn semantic segmentation, depth and surface normals. PoseNet [25] is a model which learns camera position and orientation. UberNet [27] learns a number of different regression and classification tasks under a single architecture. In this work we are the first to propose a method for jointly learning depth regression, semantic and instance segmentation. Like the model of [15], our model learns both semantic and geometry representations, which is important for scene understanding. However, our model learns the much harder task of instance segmentation which requires knowledge of both semantics and geometry. This is because our model must determine the class and spatial relationship for each pixel in each object for instance segmentation.


More importantly, all previous methods which learn multiple tasks simultaneously use a naïve weighted sum of losses, where the loss weights are uniform, or crudely and manually tuned. In this work we propose a principled way of combining multiple loss functions to simultaneously learn multiple objectives using homoscedastic task uncertainty. We illustrate the importance of appropriately weighting each task in deep learning to achieve good performance and show that our method can learn to balance these weightings optimally.

### 3. Multi Task Learning with Homoscedastic Uncertainty

Multi-task learning concerns the problem of optimising a model with respect to multiple objectives. It is prevalent in many deep learning problems. The naive approach to combining multi objective losses would be to simply perform a weighted linear sum of the losses for each individual task:

$$
L_{total} = \sum_i w_i L_i.
$$

This is the dominant approach used by prior work [39, 38, 30, 41], for example for dense prediction tasks [27], for scene understanding tasks [15] and for rotation (in quaternions) and translation (in meters) for camera pose [25]. However, there are a number of issues with this method. Namely, model performance is extremely sensitive to weight selection, $w_i$, as illustrated in Figure 2. These weight hyper-parameters are expensive to tune, often taking many days for each trial. Therefore, it is desirable to find a more convenient approach which is able to learn the optimal weights.

More concretely, let us consider a network which learns to predict pixel-wise depth and semantic class from an input image. In Figure 2 the two boundaries of each plot show models trained on individual tasks, with the curves showing performance for varying weights $w_i$ for each task. We observe that at some optimal weighting, the joint network performs better than separate networks trained on each task individually (performance of the model in individual tasks is seen at both edges of the plot: $w = 0$ and $w = 1$). At nearby values to the optimal weight the network performs worse on one of the tasks. However, searching for these optimal weightings is expensive and increasingly difficult with large models with numerous tasks. Figure 2 also shows a similar result for two regression tasks; instance segmentation and depth regression. We next show how to learn optimal task weightings using ideas from probabilistic modelling.

#### 3.1. Homoscedastic uncertainty as task-dependent uncertainty

In Bayesian modelling, there are two main types of uncertainty one can model [24].

- **Epistemic uncertainty** is uncertainty in the model, which captures what our model does not know due to lack of training data. It can be explained away with increased training data.
  
- **Aleatoric uncertainty** captures our uncertainty with respect to information which our data cannot explain. Aleatoric uncertainty can be explained away with the ability to observe all explanatory variables with increasing precision.

Aleatoric uncertainty can again be divided into two sub-categories.

- **Data-dependent or Heteroscedastic uncertainty** is aleatoric uncertainty which depends on the input data and is predicted as a model output.
  
- **Task-dependent or Homoscedastic uncertainty** is aleatoric uncertainty which is not dependent on the input data. It is not a model output, rather it is a quantity which stays constant for all input data and varies between different tasks. It can therefore be described as task-dependent uncertainty.

In a multi-task setting, we show that the task uncertainty captures the relative confidence between tasks, reflecting the uncertainty inherent to the regression or classification task. It will also depend on the task’s representation or unit of measure. We propose that we can use homoscedastic uncertainty as a basis for weighting losses in a multi-task learning problem.

### 3.2. Multi-task likelihoods

In this section we derive a multi-task loss function based on maximising the Gaussian likelihood with homoscedastic uncertainty. Let $f_W(x)$ be the output of a neural network with weights $W$ on input $x$. We define the following probabilistic model. For regression tasks we define our likelihood as a Gaussian with mean given by the model output:

$$
p(y | f_W(x)) = \mathcal{N}(f_W(x); \sigma^2)
$$

with an observation noise scalar $\sigma$. For classification we often squash the model output through a softmax function, and sample from the resulting probability vector:

$$
p(y | f_W(x)) = \text{Softmax}(f_W(x)).
$$

In the case of multiple model outputs, we often define the likelihood to factorise over the outputs, given some sufficient statistics. We define $f_W(x)$ as our sufficient statistics, and obtain the following multi-task likelihood:

$$
p(y_1, ..., y_K | f_W(x)) = p(y_1 | f_W(x)) \cdots p(y_K | f_W(x))
$$

with model outputs $y_1, ..., y_K$ (such as semantic segmentation, depth regression, etc).

In maximum likelihood inference, we maximise the log likelihood of the model. In regression, for example, the log likelihood can be written as

$$
\log p(y | f_W(x)) \propto -\frac{1}{2\sigma^2} || y - f_W(x) ||^2 - \log \sigma
$$

for a Gaussian likelihood (or similarly for a Laplace likelihood) with $\sigma$ the model’s observation noise parameter – capturing how much noise we have in the outputs. We then maximise the log likelihood with respect to the model parameters $W$ and observation noise parameter $\sigma$.

Let us now assume that our model output is composed of two vectors $y_1$ and $y_2$, each following a Gaussian distribution:

$$
p(y_1, y_2 | f_W(x)) = p(y_1 | f_W(x)) \cdot p(y_2 | f_W(x)) = \mathcal{N}(y_1; f_W(x); \sigma_1^2) \cdot \mathcal{N}(y_2; f_W(x); \sigma_2^2).
$$

This leads to the minimisation objective, $L(W, \sigma_1, \sigma_2)$, (our loss) for our multi-output model:

$$
= -\log p(y_1, y_2 | f_W(x)) \propto \frac{1}{2\sigma_1^2} || y_1 - f_W(x) ||^2 + \frac{1}{2\sigma_2^2} || y_2 - f_W(x) ||^2 + \log \sigma_1 \sigma_2.
$$

Therefore, the loss function can be decomposed as:

$$
L(W; \sigma_1, \sigma_2) = \frac{1}{2\sigma_1^2} L_1(W) + \frac{1}{2\sigma_2^2} L_2(W) + \log (\sigma_1 \sigma_2).
$$

Where we wrote $L_1(W) = || y_1 - f_W(x) ||^2$ for the loss of the first output variable, and similarly for $L_2(W)$.

We interpret minimising this last objective with respect to $\sigma_1$ and $\sigma_2$ as learning the relative weight of the losses 
$L_1(W)$ and $L_2(W)$ adaptively, based on the data. As $\sigma_1$ – the noise parameter for the variable $y_1$ – increases, we have that the weight of $L_1(W)$ decreases. On the other hand, as the noise decreases, we have that the weight of the respective objective increases. The noise is discouraged from increasing too much (effectively ignoring the data) by the last term in the objective, which acts as a regulariser for the noise terms.

This construction can be trivially extended to multiple regression outputs. However, the extension to classification likelihoods is more interesting. We adapt the classification likelihood to squash a scaled version of the model output through a softmax function:

$$
p(y|f_W(x), \sigma) = \text{Softmax} \left( \frac{1}{\sigma^2} f_W(x) \right)
$$

with a positive scalar $\sigma$. This can be interpreted as a Boltzmann distribution (also called Gibbs distribution) where the input is scaled by $\sigma^2$ (often referred to as temperature). This scalar is either fixed or can be learnt, where the parameter’s magnitude determines how ‘uniform’ (flat) the discrete distribution is. This relates to its uncertainty, as measured in entropy. The log likelihood for this output can then be written as

$$
\log p(y = c|f_W(x), \sigma) = \frac{1}{\sigma^2} f_W^c(x) - \log \sum_{c'} \exp \left( \frac{1}{\sigma^2} f_W^{c'}(x) \right)
$$

with $f_W^c(x)$ the $c$'th element of the vector $f_W(x)$.

Next, assume that a model’s multiple outputs are composed of a continuous output $y_1$ and a discrete output $y_2$, modelled with a Gaussian likelihood and a softmax likelihood, respectively. Like before, the joint loss, $L(W, \sigma_1, \sigma_2)$, is given as:

$$
= -\log p(y_1, y_2 = c|f_W(x)) = -\log \mathcal{N}(y_1; f_W(x); \sigma_1^2) \cdot \text{Softmax}(y_2 = c; f_W(x); \sigma_2)
$$

$$
= \frac{1}{2\sigma_1^2} ||y_1 - f_W(x)||^2 + \log \sigma_1 - \log p(y_2 = c|f_W(x); \sigma_2)
$$

$$
= \frac{1}{2\sigma_1^2} L_1(W) + \frac{1}{\sigma_2^2} L_2(W) + \log \sigma_1 + \log \sigma_2 + \sum_{c'} \exp \left( \frac{1}{\sigma_2^2} f_W^{c'}(x) \right)^{-\frac{1}{\sigma_2}}
$$

where again we write $L_1(W) = ||y_1 - f_W(x)||^2$ for the Euclidean loss of $y_1$, write $L_2(W)$ for the cross entropy loss of $y_2$ (with $f_W(x)$ not scaled), and optimise with respect to $W$ as well as $\sigma_1, \sigma_2$.

This last objective can be seen as learning the relative weights of the losses for each output. Large scale values $\sigma_2$ will decrease the contribution of $L_2(W)$, whereas small scale $\sigma_2$ will increase its contribution. The scale is regulated by the last term in the equation. The objective is penalised when setting $\sigma_2$ too large.

This construction can be trivially extended to arbitrary combinations of discrete and continuous loss functions, allowing us to learn the relative weights of each loss in a principled and well-founded way. This loss is smoothly differentiable, and is well formed such that the task weights will not converge to zero. In contrast, directly learning the weights using a simple linear sum of losses (1) would result in weights which quickly converge to zero. In the following sections we introduce our experimental model and present empirical results.

In practice, we train the network to predict the log variance, $s := \log \sigma^2$. This is because it is more numerically stable than regressing the variance, $\sigma^2$, as the loss avoids any division by zero. The exponential mapping also allows us to regress unconstrained scalar values, where $\exp(-s)$ is resolved to the positive domain giving valid values for variance.

---

### 4. Scene Understanding Model

To understand semantics and geometry we first propose an architecture which can learn regression and classification outputs, at a pixel level. Our architecture is a deep convolutional encoder decoder network [3]. Our model consists of a number of convolutional encoders which produce a shared representation, followed by a corresponding number of task-specific convolutional decoders. A high level summary is shown in Figure 1.

The purpose of the encoder is to learn a deep mapping to produce rich, contextual features, using domain knowledge from a number of related tasks. Our encoder is based on DeepLabV3 [10], which is a state of the art semantic segmentation framework. We use ResNet101 [20] as the base feature encoder, followed by an Atrous Spatial Pyramid Pooling (ASPP) module [10] to increase contextual awareness. We apply dilated convolutions in this encoder, such that the resulting feature map is sub-sampled by a factor of 8  compared to the input image dimensions.

We then split the network into separate decoders (with separate weights) for each task. The purpose of the decoder is to learn a mapping from the shared features to an output. Each decoder consists of a $3 \times 3$ convolutional layer with output feature size 256, followed by a $1 \times 1$ layer regressing the task’s output. Further architectural details are described in Appendix A.

**Semantic Segmentation.** We use the cross-entropy loss to learn pixel-wise class probabilities, averaging the loss over the pixels with semantic labels in each mini-batch.

**Instance Segmentation.** An intuitive method for defining which instance a pixel belongs to is an association to the instance’s centroid. We use a regression approach for instance segmentation [29]. This approach is inspired by [28], which identifies instances using Hough votes from object parts. In this work we extend this idea by using votes from individual pixels using deep learning. We learn an instance vector, $\hat{x}_n$, for each pixel coordinate, $c_n$, which points to the centroid of the pixel’s instance, $i_n$, such that $i_n = \hat{x}_n + c_n$. We train this regression with an $L_1$ loss using ground truth labels $x_n$, averaged over all labelled pixels, $N_I$, in a mini-batch:

$$
\mathcal{L}_{\text{Instance}} = \frac{1}{|N_I|} \sum_{N_I} \| x_n - \hat{x}_n \|_1.
$$



Figure 3 details the representation we use for instance segmentation. Figure 3(a) shows the input image and a mask of the pixels which are of an instance class (at test time inferred from the predicted semantic segmentation). Figure 3(b) and Figure 3(c) show the ground truth and predicted instance vectors for both $x$ and $y$ coordinates. We then cluster these votes using OPTICS [2], resulting in the predicted instance segmentation output in Figure 3(d).

One of the most difficult cases for instance segmentation algorithms to handle is when the instance mask is split due ...


