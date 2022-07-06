---
title: "contrastive learning"
date: 2022-05-09T06:00:20+06:00
menu:
  sidebar:
    name: contrastive learning
    identifier: cv-backbone-contrastive
    parent: cv-contrastive-learning
    weight: 50
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["backbone","contrastive learning"]
categories: ["Basic"]
math: true
---

从2019年中~2020年中，对比学习火了一段时间，到ViT出来后，大量的研究这才投身于ViT。

## 一、简介
什么是对比学习？<br>
简单来说就是，只要模型把相似的数据跟其他不相似的数据区分开就可以。比如：$A_1, A_2, ...$ 是狗，$B_1, B_2, ...$ 是猫，只要模型能把这两批数据区分开就行。<br>
所以，训练集中不需要明确的标签，只要能区分出那些数据之间是相似的，那些是与它们不相似的。<br>
所以，训练集中不必人为标注，只需要设计一些规则生产出这种类型的训练集就行。<br>

看下Hinton老爷子的<a href="http://www.cs.toronto.edu/~fritz/absps/naturebecker.pdf" target="blank">《Self-organizing neural network that discovers surfaces in random-dot stereograms》</a> 和 LeCun的<a href="http://www.cs.toronto.edu/~hinton/csc2535/readings/hadsell-chopra-lecun-06-1.pdf" target="blank">《Dimensionality reduction by learning an invariant mapping》</a> <br>

对比学习为啥在cv领域被认为是无监督呢？：
1. 通过设计一些巧妙的代理任务，就是pretext task：人为的定义一些规则，这些规则可以用来定义那些图片是相似的，那些图片是不相似的。<br>
例如：instance discrimination：如果有N张图片的数据集，随机一张图片$x_i$，对这个图片随机裁剪+数据增广，从同一张图片中通过裁剪+增广产生的数据，虽然有差异但是语义信息是一样的，所以是正样本(它们之间是相似的)，负样本就是除了图$x_i$之外的所有样本。

### 1、代理任务
代理任务(pretext task)的目的: 生成一个自监督的信号，从而充当ground truth这个标签信息<br>
有监督学习：训练时比较输出 $\hat{Y}$ 和 groud truth $Y$；<br>
自监督学习：因为缺少groud truth，所以需要代理任务自己创建类似groud truth的信号。


### 2、对比学习的loss
#### 1)、InfoNCE loss 
noise contrastive estimation loss：其实就是一个交叉熵
$$
L_q = -log\frac{exp(q\cdot k_+ / \tau)}{\sum_{i=0}^{K} exp(q\cdot k_i / \tau)}
$$
分母：一个正样本，K个负样本；$\tau$：温度超参数，值越大分布就越平缓，表示对每种的关注度越相似；值越小分布就越陡峭，表示比较关注比较困难的case，不容易收敛。

### 3、数据

数据处理流程：
1. 图片$x_1$经过不同的变换分别生成了不同的图片$x_1^1, x_1^2$，一般$x_1^1$为锚点作为基准；$x_1^2$是$x_1^1$的正样本；剩余的$x_2, x_3, ..., x_N$是$x_1^1$的负样本。
2. 有了正负样本数据后，就是把这些数据丢进编码器提取特征；锚点的特征：$f_{11}$，正样本：$f_{12}$，负样本：$f_2, f_3, ..., f_N$
3. 对比学习的目的：在特征空间里，让锚点的特征$f_{11}$与正样本的特征$f_{12}$尽量靠近；与负样本的特征$f_2, f_3, ..., f_N$尽量远离。

<p align="center"><img src="/datasets/posts/cnn/contrast_data.jpg" width="70%" height="70%" title="data" alt="data"></p>


## 二、初代对比网络

### 1、InstDisc
<a href="https://arxiv.org/abs/1805.01978" target="blank">《Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination》2018</a>  缺点是：字典特征不一致。

1. 提出了个体判别这个代理任务。
2. 把ImageNet的每张图片，表示层128维特征，提前算好存起来(memory bank)
3. 对memory bank的特征进行动态更新
4. 在计算loss时使用动量，来弥补字典特征的不一致性

### 2、InvaSpread
<a href="https://arxiv.org/abs/1904.03436" target="blank">《Unsupervised Embedding Learning via Invariant and Spreading Instance Feature》2019</a>：Invariant：对于相似的图片，特征尽量不变；Spreading：对于不相似的图片，特征尽量分散。 缺点：字典太小。

1. 个体判别代理任务
2. 在mni-batch内选择正样本和负样本：比如batch size = 256，正样本就是mini-batch内的每个样本，负样本就是除去该正样本后的所有样本和其数据增强后的样本。
3. 目标函数为 NCE Loss的一个变体

### 3、CPC
<a href="https://arxiv.org/abs/1807.03748" target="blank">《Representation Learning with Contrastive Predictive Coding》2018</a>

1. 预测未来的代理任务(生成式)：未来输入，通过已训的网络后的输出特征作为正样本；其他任意输入通过已训的网络后的输出特征作为负样本。


### 4、CMC
<a href="https://arxiv.org/abs/1906.05849" target="blank">《Contrastive Multiview Coding》2019</a>

1. 多视角的代理任务：同一事物的不同视角的表征，是正样本；其他事物的表征，是负样本
2. 不同视角会有不同的编码器
3. 证明了多模态融合的可能性


## 三、二代目对比网络

### 1、MoCo
<a href="https://arxiv.org/abs/1911.05722" target="blank">MoCo</a>(2020) 作为一个无监督的表征学习工作，不仅在分类领域逼近有监督的模型，还在检测、分割、人体关键点检测都超越了有监督的预训练模型，MoCo的出现证明了在视觉领域无监督训练是有前途的。<br>

{{< alert type="success">}}
问题：为什么无监督学习在NLP领域表现较好，在视觉领域效果不好呢？<br>
作者认为：NLP领域，每个token是一个独立的语义信息，其分布是一个离散的信号空间，由于token的独立性，在字典集合中，其就是一个分类任务，可以有类似标签的形式帮助训练；视觉是一个高维连续空间，不像token有很强的语义信息而且浓缩的比较好，导致视觉不能创建一个类似NLP的字典，没有这个字典就不容易建模，所以在视觉领域无监督学习还不如有监督学习。

问题：作者设计动机？<br>
以往的工作，会受限于：`a. 字典的大小；b. 字典内的一致性`。
1. 训练一个encoder，从图像数据里抽样出特征，由这些特征组成一个动态字典；
    1. 这个字典一定要大：字典越大，就可以表征更多的视觉信息。
    2. 在训练的时候，字典内要有一致性：各个key的尽量是通过相同的编码器产出的，不然query可能选择与自己的编码器相同的key，而不是真的和它含有相同语义信息的那个key。
2. 对比学习使得正样本间距离尽量小，负样本距离尽量大

{{< /alert >}}

{{< split 6 6>}}
作者怎么设计的：
1. 怎么构建大字典？所有数据集组成一个字典肯定不行，计算一次要花费很长时间，而且内存也不够。作者使用了队列这种数据结构，队列的大小就是字典的大小；队列可以很大(字典很大)，每次训练的mini-batch很小；队列中的元素不是每次都需要更新，每次更新mini-batch大小的数据。新的数据入队，最久的数据出队。字典大小(作者默认：65536)是一个超参数，可以是几千上万，训练时间都差不多。
2. 怎么保持一致性呢？作者设计了动量编码器：例如：query的编码器为 $\theta_q$，动量编码器为：$\theta_k \gets m \theta_{k} + (1-m)\theta_q$。设计一个较大的动量参数m(e.g., m=0.999)，使得动量编码器更新的`比较缓慢`，不会因为 $\theta_q$引起太多的改变，所以近似保持一致性。只有 $\theta_q$ 参与训练，$\theta_k$ 是不参与训练的，是由0.999的上一个 $\theta_k$+0.001的当前 $\theta_q$组合而成。
---
<p align="center"><img src="/datasets/posts/cnn/moco.jpg" width="90%" height="90%" title="moco" alt="moco"></p>

{{< /split >}}

```python
# f_q, f_k: encoder networks for query and key
# queue: dictionary as a queue of K keys (CxK)
# m: momentum
# t: temperature
f_k.params = f_q.params # initialize
for x in loader: # load a minibatch x with N samples
    x_q = aug(x) # a randomly augmented version
    x_k = aug(x) # another randomly augmented version
    q = f_q.forward(x_q) # queries: NxC
    k = f_k.forward(x_k) # keys: NxC
    k = k.detach() # no gradient to keys
    # positive logits: Nx1
    l_pos = bmm(q.view(N,1,C), k.view(N,C,1))
    # negative logits: NxK
    l_neg = mm(q.view(N,C), queue.view(C,K))
    # logits: Nx(1+K)
    logits = cat([l_pos, l_neg], dim=1)
    # contrastive loss, Eqn.(1)
    labels = zeros(N) # positives are the 0-th
    loss = CrossEntropyLoss(logits/t, labels)
    # SGD update: query network
    loss.backward()
    update(f_q.params)
    # momentum update: key network
    f_k.params = m*f_k.params+(1-m)*f_q.params
    # update dictionary
    enqueue(queue, k) # enqueue the current minibatch
    dequeue(queue) # dequeue the earliest minibatch

```

### 2、SimCLR
<a href="https://arxiv.org/abs/2002.05709" target="blank">《A simple framework for contrastive learning of visual representations》2020</a>很简单。缺点就是 字典比较小。
<p align="center"><img src="/datasets/posts/cnn/simclr_arch.png" width="70%" height="70%" title="SimCLR" alt="SimCLR"></p>

1. 在mni-batch内选择正样本和负样本：比如batch size = 256，正样本就是mini-batch内的每个样本，负样本就是除去该正样本后的所有样本和其数据增强后的样本。
2. 训练时添加了一个mlp全连接层，就是那个 $g(\sdot)$，这个简单的操作，在ImageNet数据集上提升了10个点。
3. 更丰富的数据增强

### 3、MoCo v2
<a href="https://arxiv.org/abs/2003.04297" target="blank">MoCo V2</a> 基于初版MoCo和SimCLR的全连接层，做了一些优化，发现添加mlp全连接层真香
<p align="center"><img src="/datasets/posts/cnn/mocov2.jpg" width="70%" height="70%" title="MoCoV2" alt="MoCoV2"></p>

1. 加了更丰富的数据增强  `提升3个点`
2. 加了MLP  `提升了6个点`
3. 加了cosine 的学习率
4. 训练更多个epoch  `提升4个点`

### 4、SimCLR v2
<a href="https://arxiv.org/abs/2003.04297" target="blank">《Big Self-Supervised Models are Strong Semi-Supervised Learners》</a>

1. 采用更大的模型，152-ResNet
2. 添加了2层mlp
3. 使用动量编码器

### 5、SwAV(swap assignment views)
<a href="https://arxiv.org/abs/2006.09882" target="blank">《Unsupervised Learning of Visual Features by Contrasting Cluster Assignments》2020</a>：给定同样一张图片，生成不同的视角；希望可以用一个视角得到的特征去预测另外一个视角得到的特征。

1. multi crop 技术：全局和局部的特征都需要关注


## 四、三代目对比网络
不使用负样本
### 1、BYOL
<a href="https://arxiv.org/abs/2006.07733" target="blank">《Bootstrap your own latent: A new approach to self-supervised Learning》2020</a>

<p align="center"><img src="/datasets/posts/cnn/byol.jpg" width="70%" height="70%" title="byol" alt="byol"></p>
<p align="center"><img src="/datasets/posts/cnn/byol_2.jpg" width="70%" height="70%" title="byol" alt="byol"></p>

1. 对输入图片x，锚点通过一系列的变换，最后是 $q_{\theta}(z_{\theta})$；正样本通过一些列的变换，最后是 $sg(z'_{\xi})$。这两个是输入图片的近似表示
2. 让 $sg(z'_{\xi})$ 做target，计算这两个的MSE-loss
3. 模型最后就训练了编码器 $f_{\theta}$，正样本的编码器 $f_{\xi}$ 只是 $f_{\theta}$ 的平均，也就是动量编码器。
4. 模型没有使用负样本，只是用自己预测自己，为啥没有出现`模型坍塌`呢？
    1. 参考 这个 <a href="https://generallyintelligent.ai/blog/2020-08-24-understanding-self-supervised-contrastive-learning/" target="blank">Blog</a> 博主，通过一系列的实验，得出他的结论：BYOL能够学到东西，主要是因为Batch normalization。通过BN的操作，用整个batch的样本计算 均值、方差，然后用在batch内的各个样本上；这个操作相当于存在信息泄露，一个样本在计算是也能看到整个batch的信息，相当于一个平均的信息作为负样本；即使没有刻意提供负样本，但通过BN的操作也有了负样本的作用。
    2. BYOL的作者听到后就不同意了，他也做了一些列的实验 <a href="https://arxiv.org/abs/2010.10241" target="blank">《BYOL works even without batch statistics》</a>，发现BN确实很香；不过，他认为是BN只是使得模型能够稳定训练，真正起作用的是一个很好的初始化。

### 2、SimSiam
<a href="https://arxiv.org/abs/2011.10566" target="blank">《Exploring Simple Siamese Representation Learning》2020</a>

{{< split 6 6>}}
作者怎么设计的：
1. 不需要负样本
2. 不需要大的batch size
3. 不需要动量编码器
```python
# f: backbone + projection mlp
# h: prediction mlp
for x in loader: # load a minibatch x with n samples
    x1, x2 = aug(x), aug(x) # random augmentation
    z1, z2 = f(x1), f(x2) # projections, n-by-d
    p1, p2 = h(z1), h(z2) # predictions, n-by-d
    L = D(p1, z2)/2 + D(p2, z1)/2 # loss
    L.backward() # back-propagate
    update(f, h) # SGD update
def D(p, z): # negative cosine similarity
    z = z.detach() # stop gradient
    p = normalize(p, dim=1) # l2-normalize
    z = normalize(z, dim=1) # l2-normalize
    return -(p*z).sum(dim=1).mean()

```
---
<p align="center"><img src="/datasets/posts/cnn/simSiam.jpg" width="90%" height="90%" title="moco" alt="moco"></p>

{{< /split >}}

---
结论：类似于EM(Expectation-Maximization)算法。

## 五、四代目对比网络
Transformer在对比学习上的应用

### 1、MoCo V3
<a href="https://arxiv.org/abs/2104.02057" target="blank">《An Empirical Study of Training Self-Supervised Vision Transformers》2021</a> 相当于 MoCo V2 和SimSiam的合体

{{< split 6 6>}}

1. 不训练 patch projection层。 
2. 把backbone有ResNet换成ViT；在训练时作者发现，准确度会时不时的塌陷。这个原因是什么呢？
    1. 作者通过观察回传梯度，在第一层梯度波动较大，说明这一层梯度不正常。作者设想：梯度既然不正常还不如直接不用梯度更新第一层，作者在第一层初始化后就直接冻住，不再更新第一层，实验后发现问题解决了。
    2. 所以，在ViT的第一层patch projection 还是有问题的，后续会被继续研究

---

```python
# f_q: encoder: backbone + proj mlp + pred mlp
# f_k: momentum encoder: backbone + proj mlp
# m: momentum coefficient
# tau: temperature
for x in loader: # load a minibatch x with N samples
    x1, x2 = aug(x), aug(x) # augmentation
    q1, q2 = f_q(x1), f_q(x2) # queries: [N, C] each
    k1, k2 = f_k(x1), f_k(x2) # keys: [N, C] each
    loss = ctr(q1, k2) + ctr(q2, k1) # symmetrized
    loss.backward()
    update(f_q) # optimizer update: f_q
    f_k = m*f_k + (1-m)*f_q # momentum update: f_k
# contrastive loss
def ctr(q, k):
    logits = mm(q, k.t()) # [N, N] pairs
    labels = range(N) # positives are in diagonal
    loss = CrossEntropyLoss(logits/tau, labels)
    return 2 * tau * loss

```
{{< /split >}}


### 2、DINO

<a href="https://arxiv.org/abs/2104.14294" target="blank">《Emerging Properties in Self-Supervised Vision Transformers》2021</a> 跟BYOL、SimSiam类似，也是预测型。

{{< split 6 6>}}
跟MoCo V3 非常类似<br>

总结一下：
<p align="center"><img src="/datasets/posts/cnn/craster_all.jpg" width="90%" height="90%" title="contrastive" alt="contrastive"></p>

---
```python
# gs, gt: student and teacher networks
# C: center (K)
# tps, tpt: student and teacher temperatures
# l, m: network and center momentum rates
gt.params = gs.params
for x in loader: # load a minibatch x with n samples
    x1, x2 = augment(x), augment(x) # random views
    s1, s2 = gs(x1), gs(x2) # student output n-by-K
    t1, t2 = gt(x1), gt(x2) # teacher output n-by-K
    loss = H(t1, s2)/2 + H(t2, s1)/2
    loss.backward() # back-propagate
    # student, teacher and center updates
    update(gs) # SGD
    gt.params = l*gt.params + (1-l)*gs.params
    C = m*C + (1-m)*cat([t1, t2]).mean(dim=0)
def H(t, s):
    t = t.detach() # stop gradient
    s = softmax(s / tps, dim=1)
    t = softmax((t - C) / tpt, dim=1) # center + sharpen
    return - (t * log(s)).sum(dim=1).mean()


```
{{< /split >}}

