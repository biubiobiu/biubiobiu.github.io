---
title: "深度学习-结构"
date: 2021-08-05T12:30:40+08:00
description: Markdown rendering samples
menu:
  sidebar:
    name: 深度学习-结构
    identifier: deep-learning-problem
    parent: deep-learning-summary
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["机器学习","深度学习","网络结构"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、激活函数

> 为什么需要激活函数？<br>
> 例如：两个感知机。 $h_1 = W_1 x + b_1, h_2 = W_2 h_1 + b_2$ <br>
> 如果没有激活函数这个 非线性变换，由于感知机的计算时线性变换，可以转换为：$h_2 = W_2 W_1 x + W_2 b_1 + b_2$ <br>
> 就是说：<font color=#f00000>如果没有激活函数，模型就做不了太深。两层的权重完全可以用一层的权重来表示。</font>

### 1、Sigmoid函数

**logistic函数**<br>
$$
\varphi(v) = \frac{1}{1+e^{-av}}
$$

**tanh函数**<br>
$$
\varphi(v) = tanh(v) = \frac{1-e^{-v}}{1+e^{-v}}
$$

**分段线性函数**<br>
$ \varphi(v) = \begin{cases} 1 &\text{if } v \geqslant \theta \\\ kv &\text{if } - \theta < v < \theta \\\ 0 &\text{if } v \leqslant 0 \end{cases}$

**概率型函数**<br>
$$
P(1) = \frac{1}{1+e^{-\frac{x}{T}}}
$$

### 2、ReLU函数
relu函数有助于梯度收敛，收敛速度快了6倍。但仍然有缺陷：<br>
在x<0是，梯度为0，一旦变成负将无法影响训练，这种现象叫做死区。如果学习率较大，会发现40%的死区。如果有一个合适的学习率，死区会大大减少。<br>

$ ReLU(x) = max(0, x) = \begin{cases} x &\text{if } x \geqslant 0 \\\ 0 &\text{if } x < 0 \end{cases}$

**带滞漏的ReLU**<br>
$ LeakyReLU(x) = \begin{cases} x &\text{if } x \geqslant 0 \\\ \gamma x &\text{if } x < 0 \end{cases}$

缓解了死区，不过 $\gamma$ 是个超参，人为设定的不准，调参影响较大。<br>
$\gamma = 0.01$ ，当神经元处于非激活状态时，也能有一个非零的梯度可以更新参数，避免永远不能被激活。

**带参数的ReLU**<br>
$ PReLU(x) = \begin{cases} x &\text{if } x \geqslant 0 \\\ \gamma_i x &\text{if } x < 0 \end{cases}$

引入一个可学习的参数 $\gamma_i$ ，不同神经元可以有不同的参数。

**ELU函数**：Exponential Linear Unit 指数线性单元<br>

在小于0的部分使用指数，具备relu的优点，同时ELU也解决了relu函数自身死区的问题。不过ELU函数指数操作稍稍增大了工作量<br>

优点：
1. 在所有点上都是连续的可微的
2. 与ReLU不同，它没有死区问题
3. 与ReLU相比，实现了更高的准确性

缺点：
1. 计算速度慢，由于负输入设计非线性

$ ELU(x) = \begin{cases} x &\text{if } x \geqslant 0 \\\ \gamma(e^x-1) &\text{if } x < 0 \end{cases}$
<p align="center"><img src="/datasets/posts/nlp/elu_0.png" width=50% height=50%></p>

**Softplus函数**<br>
$$
Softplus(x) = log(1+e^x)
$$
Softplus函数可以看做ReLU函数的平滑版本，其导数刚好是logistic函数。Softplus函数虽然也具有单侧抑制、宽兴奋边界的特性，但没有稀疏激活性。

**swish**<br>
该函数是google大脑提出的一个新的激活函数，从图像上来看，与relu差不多，唯一区别较大的是接近0的负半轴区域。
$$
swish(x) = \frac{x}{1+e^{-x}}
$$
<p align="center"><img src="/datasets/posts/nlp/swish_0.png" width=50% height=50%></p>

**GELU**<br>
高斯误差线性单元（Gaussian Error Linear Unit）激活函数，公式如下：
$$
GELU(x) = 0.5 x (1+tanh(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)))
$$
<p align="center"><img src="/datasets/posts/nlp/gelu_0.png" width=50% height=50%></p>
在大于0时，输出基本上是x。GELU激活函数的微分：
<p align="center"><img src="/datasets/posts/nlp/gelu_1.png" width=50% height=50%></p>

优点：
1. 平滑性：所有点上都可导，没有梯度截断问题。这使得梯度优化更加稳定，有助于提高神经网路的训练效率
2. 近似性：对于较大的输入，GELU 接近线性函数 x，这使得它在这些情况下起到线性激活函数的作用
3. 高斯分布：GELU 在0附近，近似微高斯分布，这有助于提高网络的泛化能力，使得模型更容易适应不同的数据分布。

缺点：
1. 计算量大

### 3、GLU函数
<a href="https://arxiv.org/pdf/1612.08083.pdf" target="bland">GLU（Gated Linear Unit）</a>激活函数是2017年提出的，其实不算是一种激活函数，而是一种神经网络层。它是一个线性变换后面接门控机制的结构。其中门控机制是一个sigmoid函数用来控制信息能够通过多少。<br>
它的门控机制，可以帮助网络更好地捕捉序列数据中的长期依赖关系。GLU激活函数最初在自然语言处理（NLP）任务中提出，并在机器翻译、语音识别等领域取得了良好的效果。<br>

$$
GLU(x) = (Vx+c) \otimes \sigma{(Wx+b)}
$$
其中，$x$：表示输入向量；$\otimes$：表示逐元素相乘；$\sigma()$：表示Sigmoid函数；$W, V, b, c$：是需要学习的参数。<br>

理解GLU激活函数的关键在于它的门控机制。门控机制使得GLU能够选择性地过滤输入向量的某些部分，并根据输入的上下文来调整输出。门控部分的作用是将输入进行二分类，决定哪些部分应该被保留，哪些部分应该被抑制。例如，在语言模型中，GLU激活函数可以帮助网络根据上下文选择性地关注某些单词或短语，从而更好地理解句子的语义。门控机制可以有效地减少噪声和不相关信息的影响，提高网络的表达能力和泛化能力。

```python
class GluLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # 第一个线性层
        self.fc1 = nn.Linear(input_size, output_size)
        # 第二个线性层
        self.fc2 = nn.Linear(input_size, output_size)
        # pytorch的GLU层
        self.glu = nn.GLU()
    
    def forward(self, x):
        # 先计算第一个线性层结果
        a = self.fc1(x)
        # 再计算第二个线性层结果
        b = self.fc2(x)
        # 拼接a和b，水平扩展的方式拼接
        # 然后把拼接的结果传给glu
        return self.glu(torch.cat((a, b), dim=1)) 
```

### 1、GLU的变体

可以把 $\sigma()$ 换成其他的激活函数，例如：Relu、swish、tanh、GELU等等。<br>

替换为tanh：$ GTU(x, W, V, b, c) = tanh(xW + b) \otimes \sigma{(xV+c)}$ <br>
替换为ReLU：$ReGLU(x, W, V, b, c) = ReLU(xW + b) \otimes \sigma{(xV+c)}$ <br>
替换为Swish：$SwiGLU(x, W, V, b, c) = Swish(xW + b) \otimes \sigma{(xV+c)}$ <br>
替换为GELU：$GEGLU(x, W, V, b, c) = GELU(xW + b) \otimes \sigma{(xV+c)}$ <br>
替换为Bilinear：$Bilinear(x, W, V, b, c) = (xW + b) \otimes \sigma{(xV+c)}$ <br>

## 二、损失函数

损失函数（Loss Function），是用来评价模型的预测值和真实值不一样的程度。常见的损失函数有：
1. 均方差损失函数
$$
MSE = \frac{1}{2N}\sum^{N}_{i=1}(t_i-y_i)^2
$$
2. 平均绝对误差损失函数
$$
MAE = \frac{1}{N}\sum^{N}_{i=1}|t_i-y_i|
$$
3. 交叉熵损失函数（Cross Entropy Loss Function）<br>
二分类任务：
$$
E = - \frac{1}{N}\sum^N_{i=1}(t_i \log y_i + (1-t_i) \log (1-y_i))
$$
$N$：表示样本数量；$y_i$：表示样本 $i$ 预测为正类的概率；$t_i$：表示样本 $i$ 的目标值。<br>
在多分类任务时，可以看成二分类的扩展：
$$
E = - \frac{1}{N}\sum_i^N\sum^{M}_{c=1}(t_i \log y_i)
$$
$N$：表示样本数量；$M$：表示类别的数量；$y_i$：表示样本 $i$ 属于类别 $c$ 的概率；$t_i$：表示样本 $i$ 的目标值(只有目标值为1，其余都是0)。<br>
**熵**：表征的是期望的稳定性，其值越小越稳定；熵越大，表示该事件发生的可能性越小，风险度会越大。<br>
**交叉熵**：主要应用于度量两个概率分布之间的差异性。
    * 当两个分布完全相同时，交叉熵的取值最小
    * 交叉熵的值是非负的，并且预测值于目标值越接近，交叉熵的值就越小
    * 对于损失函数而言，如果模型输出预测值于目标值相同，那么损失函数就越小。


## 三、学习规则
神经网络的学习规则：希望模型的损失函数最小，即：风险最小化准则。<br>

对于训练的评价准确：有一个较小的期望风险。但是由于不知道真实的数据分布和映射函数，所以实际上无法计算模型的期望风险：$R(\bold \theta)$。给定一个训练集 $\bold D$，可以计算的是 <font color=#a00000>经验风险（Empirical Risk）</font> ，即：在训练集上的平均损失。根据<font color=#a00000>大数定理</font> 当训练集大小趋于无穷大时，经验风险就趋于期望风险。然而，在通常情况下，我们无法获取无限的训练样本，并且训练样本往往是真实数据的一个很小的子集（或许包含了一定的噪声数据）。因此，最优的学习规则：就是能够找到一组参数 $\hat{\bold \theta}$，使得经验风险最小，这就是经验风险最小化准则(Empirical Risk Minimization)。


### 1、极大似然估计
估计类条件概率（似然）的一种常用策略是：先假定其具有某种确定的概率分布形式，再基于训练样本对概率分布的参数进行估计。<br>
似然函数（Likelihood Function）：是统计模型中参数的函数。当给定 联合样本值 $\bold x$ 时，关于参数 $\bold \theta$ 的似然函数 $L(\bold \theta | \bold x)$，在数值上等于给定参数 $\bold \theta$ 后变量 $\bold x$ 的概率。
$$
L(\bold \theta | \bold x) = P(\bold X = \bold x|\bold \theta)
$$
实际上，概率模型的训练过程就是参数估计过程。对于参数估计，统计学界的两个学派分别提供了不同的解决方案：
1. 频率主义学派（Frequentist）：认为参数虽然未知，但客观存在固定值。因此，可以通过优化似然函数等准则来确定参数值。
2. 贝叶斯学派（Batesian）：认为参数是为观测到的随机变量，其本身也可有分布，因此，可先假定参数服从一个先验分布，然后基于观测到的数据计算参数的后验分布。


### 2、欠/过拟合

#### 1. 误差

训练误差(training error): 训练模型在训练数据集(training set)上表现出的误差。 </p>
泛化误差(generalization error)：模型在任意一个测试数据集(test set)上表现出的误差的期望。</p>
训练集(training set)：用来产出模型参数。</p>
验证集(validation set)：由于无法从训练误差评估泛化误差，因此从训练集中预留一部分数据作为验证集，主要用来选择模型。 </p>
测试集(test set)：在模型参数选定后，实际使用。</p>

#### 2. 欠/过拟合

欠拟合(underfitting)：`模型的表现能力不足。`</p>
1. 训练样本足够，模型参数不足

过拟合(overfitting)：`模型的表现能力过剩。`</p>
1. 训练样本不足，模型参数足够：样本不足导致特征较少，相当于模型足够表征数据的特征，产生过拟合现象。


#### 3. 优化过拟合

增大训练集可能会减轻过拟合，但是获取训练数据往往代价很高。可以在模型方面优化一下，减轻过拟合现象。
1. 权重衰减(weight decay)：
对模型参数计算 $L_2$ 范数正则化。即：在原Loss中添加对模型参数的惩罚。使得模型学到的权重参数较接近于0。`权重衰减`通过惩罚绝对值较大的模型参数，为需要学习的模型增加了限制。这可能对过拟合有效。

2. 丢弃法(dropout)：针对隐藏层中的各个神经元，以概率*p*随机丢弃，有可能该成神经元被全部清零。这样，下一层的计算无法过渡依赖该层的任意一个神经元，从而在训练中可以用来对付过拟合。在测试中，就不需要丢弃了。</p>
例如：对隐藏层使用丢弃法，丢弃概率: *p*，那么h<sub>i</sub> 有*p*的概率被清零；不丢弃概率: 1-*p*，为了保证隐藏层的期望值不变*E(p')=E(p)*，需要对不丢弃的神经元做拉伸，即：$$h'_i = \frac{\xi_i} {1-p} h_i$$ 其中：随机变量*ξ<sub>i<sub>* 为0和1的概率分别为*p*和1-*p*
![DropOut](/datasets/posts/dp_summary/mlp_dropout.jpg)

3. 权重衰减（Weight Decay）：在每次参数更新时，引入一个衰减系数：
$$ \theta_t \gets (1-\beta)\theta_{t-1}-\eta g_t $$
其中 $\beta$ 为权重衰减系数，一般取值比较小，比如：0.0005。在标准的随机梯度下降中，权重衰减正则化和 $L_2$ 正则化的效果相同；但是在较为复杂的优化方法(比如：Adam)中，权重衰减正则化和 $L_2$ 正则化并不等价。

4. 数据增强（Data Augmentation）：可以减轻网络的过拟合现象。通过对训练数据进行交换可以得到泛化能力更强的网络。


## 四、优化方法


在深度学习中，通过最小化损失函数使得训练误差最小化，由于损失函数一般都会比较复杂，很难直接求解析解，而是需要基于数值方法的优化算法找到近似解，即：数值解。在局域数值方法的优化算法中，损失函数就是目标函数(Objective Function)，


### 1. 梯度下降法

梯度下降(gradient descent)的工作原理，以一维为例：
假设连续可导的函数 $f:\Reals \to \Reals$ 的输入和输出都是标量，给定绝对值足够小的数 $\epsilon$ ，根据泰勒展开式，近似：
$$
f(x+\epsilon) \approx f(x) + \epsilon f'(x)
$$
其中 $f'(x)$ 表示函数在x处的梯度。找到一个常数 $\eta > 0$，使得 $\lvert \eta f'(x) \rvert$ 足够小，那么可以将 $\epsilon$ 提换为 $-\eta f'(x)$，得到：
$$
f(x-\eta f'(x)) \approx f(x) - \eta f'(x)^{2}
$$
所以
$$
f(x-\eta f'(x)) \lesssim f(x)
$$
这就意味着，可以通过 $x \gets x-\eta f'(x)$ 来迭代x，函数 $f(x)$ 的值可能会降低。在梯度下降中，先取一个初始值 $x_0$ 和学习率 $\eta>0$，然后不断通过上式迭代x，直到停止条件。学习率 $\eta$ 是一个超参数，需要人工设定，如果学习率过小：会导致x更新缓慢从而需要更多的迭代次数；如果学习率过大，泰勒展开式不再成立，可能会出现振荡，无法保证会迭代出近似最优解。

在每次迭代中，由于训练集较大，不可能把所有样本都加载到内存中，通常是随机均匀采样多个样本组成一个小批量，然后使用这个小批量来计算梯度，完成一次迭代，即：小批量随机梯度下降(batch gradient descent)。<br>
设：目标函数 $f(x): \Reals^{d} \to \Reals$ <br> 
　　小批量数据集 $\text{\ss}$ <br> 
梯度计算：
$$
g_t \gets \nabla f_{\text{\ss}_{t}} 
$$

$$
= \frac {1} {\lvert \text{\ss} \rvert} \sum_{i \in \text{\ss}_t} \nabla f_i(x_t)
$$

$$
x_t \gets x_{t-1} - \eta_t g_t
$$

其中，$ \lvert \text{\ss} \rvert $ 表示批量大小，$\eta_t$ 表示学习率，这两个都是超参数。  


### 2. 动量法

**问题**：自变量的梯度代表了目标函数在当前位置下降最快的方向，沿着该方向更新自变量，可能还是会有一些问题。例如：类似峡谷的函数，在有些方向上的梯度比较缓慢，在有些方向上梯度比较陡峭，在相同的学习率下，容易导致在梯度缓慢的方向收敛太慢；如果调大学习率，容易导致在梯度陡峭的方向上振荡。如下图，梯度在水平方向上为正，而在竖直方向上时上时下：

<p align="center"><img src="/datasets/posts/cnn/sgd_failed.png" width="30%" height="30%" title="" alt=""></p>

**动量法**：动量法在迭代自变量时，不仅仅是利用当前的梯度值，而是利用过去一段时间的梯度值的平均。新的梯度更迭方向，不再是指下降最陡峭的方向，而是指向过去梯度的加权平均值的方向，越靠近当前时刻权重越重。

$$
\upsilon_t \gets \gamma \upsilon_{t-1} + \eta_t g_t
$$

$$
x_t \gets x_{t-1} - \upsilon_t
$$

其中，$0 \leqslant \gamma < 1$，当$\gamma = 0$时，动量法等价于小批量随机梯度下降法。

**证明**：  
我们先解释**指数加权移动平均**(exponentially weighted moving average)，然后在类比到动量法。  
$$
y_t = \gamma y_{t-1} + (1-\gamma)x_t
$$
其中，$0 \leqslant \gamma < 1$，在当前时间步$t$的变量$y_t$可以展开（类似信号系统中的激励与响应）：  

$$
\begin{array}{cc}
y_t & = (1-\gamma)x_t + \gamma y_{t-1} \\\ & = (1-\gamma)x_t + (1-\gamma)\gamma x_{t-1} + \gamma^2 y_{t-2} \\\ & = (1-\gamma)x_t + (1-\gamma)\gamma x_{t-1} + \dots + (1-\gamma)\gamma^{t-1} x_{1} + \gamma^t y_{0}
\end{array}
$$

令$n=\frac {1} {1-\gamma}$，那么$(1-\frac {1} {n})^n = \gamma^{\frac {1} {1-\gamma}}$。  
有极限：$\lim\limits_{n \to \infty} (1-\frac {1} {n})^n =\lim\limits_{\gamma \to 1} \gamma^{\frac {1} {1-\gamma}}= exp(-1) \approx 0.3679$  

对于$y_t$，可以看做是对最近$\frac {1} {1-\gamma}$个时间步的加权平均；忽略含有$\gamma^{\frac {1} {1-\gamma}}$和比$\gamma^{\frac {1} {1-\gamma}}$更高阶系数的项，即：当$\gamma=0.95$时，可以看成对最近20时间步的$x_i$值的加权平均  
$$
y_t \approx 0.05\displaystyle\sum_{i=0}^{19} 0.95^i x_{t-i}
$$

**类比向量法**  

$$
\upsilon_t \gets \gamma \upsilon_{t-1} + (1-\gamma)\frac {\eta_t} {1-\gamma} g_t
$$

$$
x_t \gets x_{t-1} - \upsilon_t
$$

> 所以：向量$\upsilon_t$实际上是对序列$\frac {\eta_{t-i}} {1-\gamma} g_{t-i}$做指数加权移动平均；也就是说：动量法在每个时间步的自变量更新量近似于将最近的$\frac {1} {1-\gamma}$个时间步的更新量做指数加权移动平均。<mark>动量法中，自变量在各个方向上的移动幅度，不仅取决于当前梯度，还取决于历史各个梯度在各个方向上是否一致。</mark>如果在某个方向上时正时负，说明在该方向上有振荡，通过动量的向量相加，对于该情况会降低每次的更新量，使得梯度在该方向上不发散。

### 3. AdaGrad算法

**问题**：在统一学习率的情况下，梯度值较大的维度可能会振荡，梯度值较小的维度收敛可能会过慢。  

**AdaGrad算法**：根据自变量在每个维度的梯度值大小来调整各个维度上的学习率，从而避免统一的学习率难以适应所有维度的问题。

$$
s_t \gets s_{t-1} + g_t \odot g_t
$$

$$
x_t \gets x_{t-1} - \frac {\eta} {\sqrt{s_t + \epsilon}} \odot g_t
$$

其中，$\odot$表示按元素相乘，$\eta$表示学习率。目标函数自变量中每个元素的学习率通过按元素运算重新调整一下，每个元素都分别拥有自己的学习率。由于$s_t$一直在累加，所以每个元素的学习率在迭代过程中一直在降低，当学习率在迭代早期降得比较快且当前解依然不佳时，AdaGrad算法在迭代后期由于学习率过小，可能较难找到一个有用的解。


### 4. RMSProp算法

**问题**：AdaGrad算法在迭代后期由于学习率过小，可能较难找到一个有用的解。为了解决这一问题，RMSProp算法对AdaGrad算法做了一些修改。

**RMSProp算法**：只是在AdaGrad算法中添加了 指数加权移动平均。

$$
s_t \gets \gamma s_{t-1} + (1-\gamma)g_t \odot g_t
$$

$$
x_t \gets x_{t-1} - \frac {\eta} {\sqrt{s_t + \epsilon}} \odot g_t
$$

其中，$\eta$是学习率，RMSProp算法的状态变量是对平方项$g_t \odot g_t$的指数加权移动平均，所以可以看作最近$\frac {1} {1-\gamma}$个时间步的加权平均。如此一来，自变量每个元素的学习率在迭代过程中就不再一直降低。

### 5. AdaDelta算法

**AdaDelta算法**：是另一个针对AdaGrad算法优化的算法，不过没有学习率这个超参数。

$$
s_t \gets \gamma s_{t-1} + (1-\gamma)g_t \odot g_t
$$

$$
g_t' \gets \sqrt{\frac {\Delta x_{t-1} + \epsilon} {s_t + \epsilon}} \odot g_t
$$

$$
\Delta x_t \gets \gamma \Delta x_{t-1} + (1-\gamma)g_t' \odot g_t'
$$

RMSProp算法，还维护一个额外的状态变量$\Delta x_t$，用来记录自变量变化量$g_t'$按元素平方的指数加权移动平均。

$$
x_t \gets x_{t-1} - g_t'
$$


### 6. Adam算法

**Adam算法**：结合了动量变量$\upsilon_t$ 和 RMSProp算法的梯度按元素平方和的指数加权移动平均。

$$
\upsilon_t \gets \beta_1 \upsilon_{t-1} + (1-\beta_1) g_t
$$

其中，$0 \leqslant \beta_1 < 1$（建议0.9），$\upsilon_0$初始化为0，则：$\upsilon_t = (1-\beta_1)\displaystyle\sum_{i=1}^t \beta_1^{t-i} g_i$  
将过去各时间步小批量随机梯度的<mark>权值</mark>相加：$(1-\beta_1)\displaystyle\sum_{i=1}^t \beta_1^{t-i}=1-\beta_1^t$，当t较小时，过去各时间步梯度权值之和会较小，为了消除这样的影响，对任意时间步t，可以将向量$\upsilon_t$再除以$1-\beta_1^t$ ：

$$
\hat{\upsilon}_t \gets \frac {\upsilon_t} {1-\beta_1^t}
$$

$$
s_t \gets \beta_2 s_{t-1} + (1-\beta_2)g_t \odot g_t
$$

$$
\hat{s}_t \gets \frac {s_t} {1-\beta_2^t}
$$

其中，$0 \leqslant \beta_2 < 1$（建议0.999），$s_0$初始化为0

Adam算法使用修正后的变量$\hat{\upsilon}_t, \hat{s}_t$，将模型参数中每个元素的学习率通过按元素运算重新调整。

$$
g_t' \gets \frac {\eta \hat{\upsilon}_t} {\sqrt{\hat{s}_t + \epsilon}}
$$
其中，$\eta$是学习率，$\epsilon$是为了维持数值稳定性而添加的常数，例如$10^{-8}$。<mark>分子：是动量，可以在方向上消除发散；分母：在幅度上修改每个元素的学习率。</mark>

$$
x_t \gets x_{t-1} - g_t'
$$


## 五、模型评估

在深度学习中，用来衡量模型的好坏标准有很多，比如：混淆矩阵、准确率、精确率、召回率、F值等

### 1、混淆矩阵

||预测类别|
|:--|:--|
|真实类别||

||P|N|
|:--|:--|:--|
|T|TP|TN|
|F|FP|FN|

### 2、准确率、精确率、召回率

1. 准确率（Accuracy）：正确预测的各个类别的数量 / 总数
$$
A_c = \frac {TP + FN} {TP + TN + FP + FN}
$$

2. 精确率（Precision）：单个类别的正确预测数量 / 所有预测为这类的数量
$$
P_c = \frac {TP} {TP + FP}
$$

3. 召回率（Recall）：单个类别的正确预测数量 / 所有这类真实值的数量
$$
R_c = \frac {TP} {TP + TN}
$$

4. F值（F Measure）：是一个综合指标，为精确率和召回率的调和平均
$$
F = \frac {(1+\beta^2)P_c R_c} {\beta^2 P_c + R_c}
$$
$\beta$：用于平衡精确率和召回率的重要性，一般取值为 1。$\beta = 1$时，$F$ 值称为 $F_1$ 值，是精确率和召回率的调和平均，表示：<font color=#a00000>精确率和召回率同等重要</font>。$\beta < 1$ 表示：精确率更重要一些; $\beta > 1$ 表示：召回率更重要一些。

### 3、ROC/AUC/PR曲线

1. ROC曲线（Receiver Operating Characteristic）称为接受者操作特征曲线。
    * 以 “真正例”（TPR）为y轴；以“假正例”（FPR）为x轴
    * (0, 1): 表示所有的样本都正确分类
    * (1, 0): 表示避开了所有正确的答案
    * (0, 0): 表示分类器把每个样本都预测为负例
    * (1, 1): 表示分类器把所有样本都预测为正例
    * ROC曲线越靠近左上角，模型的准确性就越高，一般ROC曲线是光滑的，那么基本上可以判断模型没有太大的过拟合。



2. AUC曲线（Area Under Curve）的值为ROC曲线下面的面积，AUC的值越大，表示模型的性能越好。若分类器的性能极好，则AUC=1。但是在现实中，没有如此完美的模型，一般 $AUC \in (0.5, 1)$
    * AUC=1：完美预测
    * $0.5 < AUC < 1$：优于随机猜测
    * $AUC = 0.5$：与随机猜测一样
    * $AUC < 0.5$：比随机猜测还差

3. PR曲线（Precision Recall）：表示精确率和召回率的曲线
    * 以精确率为y轴；以召回率为x轴。



