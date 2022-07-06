---
title: "optimizer"
date: 2021-09-09T06:00:20+06:00
menu:
  sidebar:
    name: optimizer
    identifier: cv-backbone-optimizer
    parent: cv-backbone
    weight: 20
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["optimizer"]
categories: ["Basic"]
math: true
---

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
g_t \gets \nabla f_{\text{\ss}_{t}}=\frac {1} {\lvert \text{\ss} \rvert} \displaystyle\sum_{i \in \text{\ss}_{t}} \nabla f_i(x_{t-1})
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
