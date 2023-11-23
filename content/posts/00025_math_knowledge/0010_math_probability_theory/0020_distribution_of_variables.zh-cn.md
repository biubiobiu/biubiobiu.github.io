---
title: "随机变量及其分布"
date: 2023-08-01T06:00:20+08:00
menu:
  sidebar:
    name: 随机变量及其分布
    identifier: math-distribution_variables
    parent: math-probability-theory
    weight: 20
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["概率论","随机变量","分布"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、一维随机变量

> 如何引入一个法则，将随机试验的每个结果（即：$S$ 中的每个元素 $e$）与实数 $x$ 对应起来 <br>
> 从而引入了<font color=#f00000>随机变量</font>的概念，即：定义域是：样本空间$S$，值域是：实数。<br>
> **定义**：设随机试验的样本空间为 $S=\\{e\\}, X=X(e)$ 是定义在样本空间 $S$ 上的实值单值函数。称 $X = X(e)$ 为随机变量。<br>
> **例如**：以 $X$ 记录三次投掷硬币得到正面的次数。$P(X=2) = 3/8$ 就表示：随机变量 $X=2$ 的概率，就是 $A=\\{HHT,HTH,THH\\}$ 这个事件的概率。<br>

随机变量的引入，使得我们能用随机变量来描述各种随机现象，并能利用数学分析的方法对随机试验的结果进行深入广泛的研究和讨论。

### 1、离散型随机变量
{{< alert type="info" >}}
**离散型随机变量**：随机变量，它全部可能取到的值是有限个或者可列无限多个。 可以用 <font color=#f00000>分布律</font> 来描述。<br>

**常见离散型随机变量**：<br>

1. (0-1)分布 <font color=#f00000>期望：p，方差：p(1-p)</font> <br>
随机变量 $X$ 只能取 0 与 1 两个值。

2. 伯努利实验、二项分布，记：$ \textcolor{#f00000} {X \sim b(n, p)，期望：np，方差：np(1-p)}$<br>
设实验 $E$ 只有两个可能结果：$A$ 和 $\bar A$ ，则称 $E$ 为伯努利实验。<br>
将实验 $E$ 独立重复地进行 $n$ 次，则称这一串重复的独立实验为 $n$ 重伯努利实验。<br>
设：$P(A) = p, P(\bar A) = 1 - p = q$ ，显然：<br> 
$$P(X=k) = C^k_n p^kq^{n-k}$$ <br>
$ \sum^n_{k=0} P(X=k) = \sum^n_{k=0} C^k_n p^k q^{n-k} = (p+q)^n = 1$

3. 泊松分布，记：$\textcolor{#f00000} {X \sim \pi(\lambda)，期望：\lambda，方差：\lambda}$ <br>
设：随机变量 $X$ 所有可能取的值为 $0, 1, 2, ...$ ，而取各个值的概率为：<br>
$$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$ <br>
其中 $\lambda > 0$ 是常数，则称 $X$ 服从参数为 $\lambda$ 的泊松分布，记为 $X \sim \pi(\lambda)$ <br>
由于：
$$
\sum^{\infty}_{k=0} \frac{\lambda^k}{k!} = e^{\lambda}
$$ 
服从泊松分布的，例如：一本书一页中的印刷错误数、某医院在一天内的急诊病人数、某地区一个时间间隔内发生交通事故的次数。

{{< /alert >}}

---

{{< alert type="success" >}}
**泊松定理**：设 $\lambda > 0$ 是一个常数，$n$ 是任意正整数，设 $np_n = \lambda$，则对于任一固定的非负整数 $k$，有
$$
lim_{x \to \infty} C^k_n p^k_n(1-p_n)^{n-k} = \frac{\lambda^k e^{-\lambda}}{k!}
$$
定理的条件：$np_n = \lambda$，意味着当 $n$ 很大时（$\lambda$ 是常数），$p_n$ 必定很小。因此，上述定理表明当 $n$ 很大，$p$ 很小时，有以下近似式
$$
C^k_n p^k (1-p)^{n-k} \approx \frac{\lambda^k e^{-\lambda}}{k!}
$$
也就是说以 $n, p$ 为参数的二项分布的概率值可以由参数为 $\lambda = np$ 的泊松分布的概率值近似。

{{< /alert >}}

### 2、非离散型

对于非离散型随机变量 $X$ ，由于其值不能一一列举出来，因而不能用分布律来描述它。<font color=#f00000>另外，通常所遇到的非离散型随机变量取任一指定的实数值的概率都是0。再者，我们不会对 $X$ 等于具体的值的概率感兴趣，而是对 随机变量 $X$ 落在一个区间 $(x_1, x_2)$ 的概率 感兴趣。</font> <br>
所以，我们只需要知道 $P(X <= x_2)$ 和 $P(X <= x_1)$ 就可以了，即：<font color=#f00000>分布函数</font>。

{{< alert type="info" >}}
**分布函数**：设 $X$ 是一个随机变量，$x$ 是任意实数，函数 $F(x) = P(X <= x), -\infty < x < \infty$ 称为 $X$ 的分布函数。

{{< /alert >}}

### 3、连续型随机变量
{{< alert type="info" >}}
**概率密度**：对于随机变量 $X$ 的分布函数 $F(x)$，若存在非负函数 $f(x)$，使得对于任意实数 $x$ 有
$$
F(x) = \int^x_{-\infty} f(x) dt
$$
则称 $X$ 为连续型随机变量，其中函数 $f(x)$ 称为 $X$ 的概率密度函数，简称 概率密度。<br>

给定 $X$ 的概率密度 $f(x)$ 就能确定 $F(x)$，由于 $f(x)$ 位于积分号之内，故改变 $f(x)$ 在个别点上的函数值并不改变 $F(x)$ 的值，因此，改变 $f(x)$ 在个别点上的值，是无关紧要的。

**常见连续型随机分布** <br>

1. 均匀分布 $  \textcolor{#f00000} {X \sim U(a, b)，期望：\frac{a+b}{2}，方差：\frac{(b-a)^2}{12}}$<br>
$$
f(x) = \begin{cases} \frac{1}{b-a} & a<x<b \\\ 0 & 其他 \end{cases}
$$

2. 指数分布 <font color=#f00000>期望：$\theta，方差：\theta^2$</font><br>
若连续型随机变量 $X$ 的概率密度为：<br>
$$
f(x) = \begin{cases} \frac{1}{\theta} e^{-x/\theta} & x > 0 \\\ 0 & 其他 \end{cases}
$$

其中，$\theta > 0$ 为常数，则称 $X$ 服从参数为 $\theta$ 的指数分布。<br>
<font color=#f00000>属性</font>：无记忆性。$P(X > s+t | X > s) = P(X > t)$，即：$X$ 表示一元件的寿命。已知元件已经使用 s 小时，它总共能使用至少 s+t 小时的条件概率，与从开始使用时算起它至少能使用 t 小时的概率相等。也就是说，元件对它已使用过 s 小时没有记忆。

3. 正态分布 $ \textcolor{#f00000} {X \sim N(\mu, \sigma^2)}$ <br>
若连续型随机变量 $X$ 的概率密度为：<br>
$$
f(x) = \frac{1}{\sqrt{2\pi} \sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}, -\infty < x < \infty
$$
其中，$\mu, \sigma > 0$ 为常数，则称 $X$ 服从参数为 $\mu, \sigma$ 的正态分布或者高斯分布，记为 $X \sim N(\mu, \sigma^2)$ <br>
<font color=#f00000>属性</font>：
    * 曲线关于 $x=\mu$ 对称
    * 当 $x=\mu$ 时，取到最大值，$x$ 离 $\mu$ 越远，$f(x)$ 的值越小。
    * 如果固定 $\mu$，改变 $\sigma$ ，当 $\sigma$ 越小时，图形变得越尖，因而 $X$ 落在 $\mu$ 附近的概率越大。
  <p align="center"><img src="/datasets/posts/maths/gass_n.png" width=100% height=100%></p>

4. $\Gamma$ 分布 $\textcolor{#f00000} {X \sim \Gamma(\alpha, \theta)}$ <br>
$$
f(x) = \begin{cases} \frac{1}{\theta^\alpha \Gamma(\alpha)} x^{\alpha-1} e^{-x/\theta} & x > 0, \alpha > 0, \theta > 0 \\\ 0 & 其他
\end{cases}
$$

{{< /alert >}}


## 二、一个随机变量的函数
在实际中，我们感兴趣的随机变量不能直接测量得到，而它却是某个能直接测量的随机变量的函数。比如：我们不能直接测量圆的面积，我们可以直接测量直径 $d$，面积可以由直径计算得到：$A = \frac{1}{4} \pi d^2$。<br>
所以，需要研究如何由已知的随机变量 $X$ 的概率分布去求得它的函数 $Y = g(X)$ 的概率分布。

设 $y = g(x), x = h(y)$ <br>
需要计算：$F_Y(y) = P(Y <= y) = P(g(x) <= y)$

{{< alert type="info" >}}
**定理**：<br>
设随机变量 $X$ 具有概率密度 $f_x(x), -\infty < x < \infty$，又设函数 $y = g(x)$ 处处可导且<font color=#f00000>恒有 $g'(x) > 0$ 或者 恒有 $g'(x) < 0$，</font>则 $Y = g(X)$ 是连续型随机变量，其概率密度为<br>
$$
f_y(y) = \begin{cases} f_x[h(y)]|h'(y)| & a < y < b> \\\ 0 & 其他 \end{cases}
$$ 
其中，$a = min(g(-\infty), g(\infty)), b = max(g(-\infty), g(\infty))$，$h(y)$ 是 $g(x)$ 的反函数。


{{< /alert >}}

## 三、多维随机变量及其分布

比如炮弹弹着点的位置需要由它的横坐标，纵坐标来确定，而横坐标和纵坐标是定义在同一个样本空间的两个随机变量。<br>

二维随机变量 $(X, Y)$ 的性质不仅与 $X$、$Y$ 有关，而且还依赖于这两个随机变量的相互关系，因此，逐个来研究 $X$、$Y$ 的性质是不够的，还需要将 $(X, Y)$ 作为一个整体来进行研究。

{{< alert type="info" >}}
**定义**：设 $(X, Y)$ 是二维随机变量，对于任意实数 $x, y$ 二元函数：
$$
F(x, y) = P[(X<=x) \cap (Y<=y)] \overset{\underset{\mathrm{记作}}{}}{==} P(X<=x, Y<=y)
$$
称为二维随机变量 $(X, Y)$ 的分布函数，或称为随机变量 $X$ 和 $Y$ 的联合分布函数。<br>

二维随机变量 $(X, Y)$ 作为一个整体，具有分布函数 $F(x, y)$，而 $X$ 和 $Y$ 都是随机变量，各自也有分布函数，将它们分别记为 $F_X(x), F_Y(y)$，依次称为二维随机变量 $(X, Y)$ 关于  $X$ 和 $Y$ 的 <font color=#f00000>边缘分布函数</font>，边缘分布函数可以由 $(X, Y)$ 的分布函数 $F(x, y)$ 所确定
$$
F_X(x) = P(X<=x) = P(X<=x, Y < \infty) = F(x, \infty) \\\
f_X(x) = \int^\infty_{-\infty} f(x, y) dy \\\
f_Y(y) = \int^\infty_{-\infty} f(x, y) dx
$$
分别称 $f_X(x), f_Y(y)$ 为 $(X, Y)$ 关于 $X$ 和 $Y$ 的边缘概率密度

**条件分布** <br>
1. 离散型 - 条件分布律 <br>
设 $(X, Y)$ 是二维离散型随机变量，对于固定的 $j$，若 $P(Y = y_j)$ > 0，则称
$$
P(X=x_i | Y=y_j) = \frac{P(X=x_i, Y=y_j)}{P(Y = y_j)} = \frac{p_{ij}}{p_{·j}}
$$
为在 $Y = y_j$ 条件下随机变量 $X$ 的条件分布律

2. 连续型 - 条件概率密度 <br>
设二维随机变量 $(X, Y)$ 的概率密度为 $f(x, y)$，$(X, Y)$ 关于 $Y$ 的边缘概率密度为 $f_Y(y)$，若对于固定的 $y$，$f_Y(y) > 0$，则称 $\frac{f(x, y)}{f_Y(y)}$ 为在 $Y=y$ 的条件下 $X$ 的条件概率密度，记为：
$$
f_{X|Y}(x|y) = \frac{f(x, y)}{f_Y(y)}
$$
称 $\int^x_{-\infty} f_{X|Y}(x|y) dx = \int^x_{-\infty} \frac{f(x, y)}{f_Y(y)} dx$ 为在 $Y=y$ 的条件下 $X$ 的条件分布函数，记为 $P(X<=x | Y=y)$ 或 $F_{X|Y}(x|y)$


**常见二维随机变量**：<br>

1. 二维正态分布 $\textcolor{#f00000} {(X, Y) \sim N(\mu_1, \mu_2, \sigma^2_1, \sigma^2_2, \rho)}$ <br>
$$
f(x, y) = \frac{1}{2\pi\sigma_1 \sigma_2 \sqrt{1-\rho^2}} e ^{-\frac{1}{2(1-\rho^2)}[\frac{(x-\mu_1)^2}{\sigma^2_1}-2\rho \frac{(x-\mu_1)(y-\mu_2)}{\sigma_1 \sigma_2} + \frac{(y-\mu_2)^2}{\sigma^2_2}]}
$$
其中，$\mu_1, \mu_2, \sigma_1, \sigma_2, \rho$ 都是常数，且 $\sigma_1 > 0, \sigma_2 > 0, -1 < \rho < 1$。 <br>
x 的边缘概率密度：
$$
f_X(x) = \frac{1}{\sqrt{2\pi} \sigma_1} e^{-\frac{(x-\mu_1)^2}{2\sigma_1^2}}, -\infty < x < \infty
$$
y 的边缘概率密度：
$$
f_Y(y) = \frac{1}{\sqrt{2\pi \sigma_2}} e^{-\frac{(y-\mu_2)^2}{2\sigma_2^2}}, -\infty < x < \infty
$$
<font color=#f00000>可以看到二维正态分布的两个边缘分布都是一维正态分布，并且都不依赖于参数 $\rho$。</font> <br>
对于给定的$\mu_1, \mu_2, \sigma_1, \sigma_2$，不同的 $\rho$ 对应不同的二维正态分布，它们的边缘分布却都一样。这说明：<font color=#f00000>单由 $X$ 和 $Y$ 的边缘分布，一般来说不能确定随机变量 $X$ 和 $Y$ 的联合分布。</font>


{{< /alert >}}

## 四、两个随机变量的函数
两个随机变量组成的函数，是否也是个随机变量呢？
### 1、$Z=X+Y$ 的分布
设 $(X, Y)$ 是二维连续型随机变量，它具有概率密度 $f(x, y)$，则 $Z=X+Y$ 仍然为连续型随机变量，其概率密度为：
$$
f_{X+Y}(z) = \int^\infty_{-\infty} f(z-y, y) dy \\\
f_{X+Y}(z) = \int^\infty_{-\infty} f(x, z-x) dy
$$
如果 $X, Y$ 相互独立，设 $(X, Y)$ 关于 $X, Y$ 的边缘密度分别为 $f_X(x), f_Y(y)$，则
$$
f_{X+Y}(z) = \int^\infty_{-\infty} f_X(z-y) f_Y(y) dy \\\
f_{X+Y}(z) = \int^\infty_{-\infty} f_X(x)f_Y(z-x) dx
$$
这两个公式称为 $f_X$ 和 $f_Y$ 的卷积公式，记为 $f_X * f_Y$ 即：
$$
f_X * f_Y = \int^\infty_{-\infty} f_X(z-y) f_Y(y) dy = \int^\infty_{-\infty} f_X(x)f_Y(z-x) dx
$$
<font color=#f00000>一般，设 $X, Y$ 相互独立，且 $X \sim N(\mu_1, \sigma_1^2), Y \sim N(mu_2, \sigma_2^2)$。由于 $Z=X+Y$ 仍然服从正态分布，且有 $Z \sim N(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$。这个结论还能推广到 n 个独立正态随机变量之和的情况，即：若 $X_i \sim N(\mu_i, \sigma_i^2)$ 且它们相互独立，则 它们的和 $Z = X_1 + X_2 + ... + X_n$ 仍然服从正态分布，且有 $Z \sim N(\mu_1+\mu_2+...+\mu_n, \sigma_1^2+\sigma_1^2+...+\sigma_n^2)$</font> <br>
<font color=#a020f0>更一般：有限个相互独立的正态随机变量的线性组合，仍然服从正态分布。</font> <br>

<font color=#f00000>如果是，$X, Y$ 都是 $\Gamma(\alpha, \theta)$ 分布，且相互独立</font>：<br>
$X$ 的分布 <br>
$
f(x) = \begin{cases}
   \frac{1}{\theta^\alpha \Gamma(\alpha)} x^{\alpha-1} e^{-x/\theta} & x > 0, \alpha > 0, \theta > 0 \\\
   0 & 其他
\end{cases}
$ <br>
$Y$ 的分布 <br>
$
f(y) = \begin{cases}
   \frac{1}{\theta^\beta \Gamma(\beta)} y^{\beta-1} e^{-y/\theta} & x > 0, \beta > 0, \theta > 0 \\\
   0 & 其他
\end{cases}
$ <br>
有结论：$Z = X + Y$ 服从参数为 $\alpha + \beta, \theta$ 的 $\Gamma$ 分布，即 $X+Y \sim \Gamma(\alpha+\beta, \theta)$ <br>
这个结论还能推广到 $n$ 个相互独立的 $\Gamma$ 分布变量之和的情况，这一性质称为 $\Gamma$ 分布的可加性。

### 2、$Z=\frac{Y}{X}, Z=XY$的分布
设 $(X, Y)$ 是二维连续型随机变量，它具有概率密度 $f(x, y)$，则 $Z = \frac{Y}{X}$，$Z = YX$ 仍为连续型随机变量，其概率密度分别为：
$$
f_{Y/X} = \int^\infty_{-\infty} |x| f(x, xz) dx \\\
f_{XY} = \int^\infty_{-\infty} \frac{1}{|x|} f(x, \frac{z}{x}) dx
$$


### 3、$M=max(X, Y); N=min(X, Y)$的分布
设 $X, Y$ 是两个相互独立的随机变量，它们的分布函数分别为 $F_X(x)$ 和 $F_Y(y)$，现在来求 $M=max(X, Y); N=min(X, Y)$ 的分布函数。<br>
$$
P(M <= z) = P(X <= z, Y <= z) \overset{\mathrm{相互独立}}{===} P(X <= z) P(Y <= z) \\\
即: F_{max}(z) = F_X(z)F_Y(z)
$$
类似的：
$$
F_{min}(z) = p(N<=z) = 1-P(N>z) = 1 - [1-F_X(z)][1-F_Y(z)]
$$

{{< split 6 6>}}
<p align="center"><img src="/datasets/posts/maths/dim2_fun.png" width=60% height=60%></p>
---
情况1：串联的情况： $Z = min(X, Y)$ <br>
情况2：并联的情况： $Z = max(X, Y)$ <br>
情况3：备用的情况： $Z = X+Y$
{{< /split >}}

## 五、随机变量的数字特征

虽然有分布函数、概率密度 可以完整地描述随机变量，但是某些实际的问题中，人们感兴趣于某些能描述随机变量某一种特征的常数，比如：运动员的平均身高、棉花纤维的平均长度、以及偏离程度。<br>
所以，要介绍几个重要的数字特征：<font color=#f00000>数学期望、方差、相关系数、矩</font> <br>

### 1、数学期望
> **离散型**：<br>
> **定义**：设离散型随机变量 $X$ 的分布律 为 
> $$
> P(X = x_k) = p_k, k = 1, 2, ...
> $$
> 若级数 $\sum^\infty_{k=1} x_k p_k$ 绝对收敛，则称级数 $\sum^\infty_{k=1} x_k p_k$ 的和为随机变量 $X$ 的 数学期望，记为 $E(X)$
> $$
> E(X) = \sum^\infty_{k=1} x_k p_k
> $$
> <br>
> **连续型**：<br>
> 设连续型随机变量 $X$ 的概率密度为 $f(x)$ 若积分 $\int^\infty_{-\infty} xf(x) dx$ 绝对收敛，则称积分 $\int^\infty_{-\infty} xf(x) dx$ 的值为随机变量 $X$ 的数学期望，记为 $E(X)$ <br>
> $$
> E(X) = \int^\infty_{-\infty} xf(x) dx
> $$


{{< alert type="info" >}}
**随机变量的函数的期望**: <br>
定理：设 $Y$ 是随机变量 $X$ 的函数：$Y = g(X)$ <br>
1. 如果 $X$ 是离散型随机变量，则有
$$
E(Y) = E[g(X)] = \sum^\infty_{k=1} g(x_k)p_k
$$

2. 如果 $X$ 是连续型随机变量，它的概率密度为 $f(x)$，则有
$$
E(Y) = E[g(X)] = \int^\infty_{-\infty} g(x)f(x)dx
$$

<font color=#f00000>定理的重要意义：当我们求 $E(Y)$ 时，不必计算 $Y$ 的分布律或者概率密度，而只需要利用 $X$ 的分布律或者概率密度就可以了。</font>


**重要性质**：<br>
1. 设 $C$ 是常数，则有 $E(C) = C$
2. 设 $X$ 是一个随机变量，$C$ 是常数，则有
$$
E(CX) = CE(X)
$$
3. 设 $X, Y$ 是两个随机变量，则有
$$
E(X+Y) = E(X) + E(Y)
$$
4. 设 $X, Y$ 是相互独立的随机变量，则有
$$
E(XY) = E(X) E(Y)
$$


{{< /alert >}}

### 2、方差
{{< alert type="info" >}}
定义：设 $X$ 是一个随机变量，若 $E([X-E(X)]^2)$ 存在，则称 $E([X-E(X)]^2)$ 为 $X$ 的方差，记：$D(X) 或 Var(X)$，即：
$$
D(X) = Var(X) = E([X-E(X)]^2) = E(X^2) - [E(X)]^2
$$
来度量随机变量 $X$ 与 其均值 $E(X)$ 的偏离程度。 $D(X)$ 是刻画 $X$ 取值分散程度的一个量，它是衡量 $X$ 取值分散程度的一个尺度。<br>

方差的重要性质：
1. 设 $C$ 为常数，则 $D(C) = 0$
2. 设 $X$ 是随机变量，$C$ 是常数，则有
$$
D(CX) = C^2 D(X), D(X+C) = D(X)
$$
3. 设 $X, Y$ 是两个随机变量，则有
$$
D(X+Y) = D(X) + D(Y) + 2E[(X-E(X))(Y-E(Y))]
$$
若 $X, Y$ 相互独立，则有
$$
D(X+Y) = D(X) + D(Y)
$$
4. $D(X) = 0$ 的充要条件是 $X$ 以概率1取常数 $E(X)$，即
$$
P(X=E(X)) = 1
$$


{{< /alert >}}

---

{{< alert type="success" >}}

有些分布，当拿到 期望、方差 这两个数字特征，可以完全确定整个分布：
1. 正态分布，完全可以由它的数学期望、方差确定
2. 指数分布，$E(X) = \theta, D(X) = \theta^2$，完全可以根据 $\theta$ 完全确定整个分布
3. 泊松分布，$E(X) = \lambda, D(X) = \lambda$，完全可以根据 $\lambda$ 完全确定整个分布

<font color=#f00000>切比雪夫(Chebyshev)不等式</font>：<br>
设随机变量 $X$ 具有数学期望 $E(X) = \mu$，方差 $D(X) = \sigma^2$，则对于任意整数 $\varepsilon$，不等式成立：
$$
P[|X-\mu| \geqslant \varepsilon] \leqslant \frac{\sigma^2}{\varepsilon^2}
$$
切比雪夫不等式给出了在随机变量的<font color=#f00000>分布未知</font>，而只知道 $E(X), D(X)$ 的情况下，估计概率 $P[|X-\mu| \geqslant \varepsilon]$ 的界限。<br>

**证明**：<br>
$$
P[|X-\mu| \geqslant \varepsilon] = \int_{|X-\mu| \geqslant \varepsilon} f(x) dx \leqslant \int_{|X-\mu| \geqslant \varepsilon} \frac{|x-\mu|^2}{\varepsilon^2} f(x) dx \leqslant \frac{1}{\varepsilon^2} \int^\infty_{-\infty} (x-\mu)^2 f(x)dx = \frac{\sigma^2}{\varepsilon^2}
$$

{{< /alert >}}

### 3、相关系数
描述 $X, Y$ 之间相互关系的数字特征。前面计算方差时，如果相互独立，则 $E([X-E(X)][Y-E(Y)]) = 0$，如果不相互独立，则 $X, Y$ 之间存在一定的关系。

{{< alert type="info" >}}
**定义**： $E([X-E(X)][Y-E(Y)])$ 称为随机变量 $X, Y$ 的协方差，记为：$Cov(X, Y)$，即：
$$
Cov(X, Y) = E([X-E(X)][Y-E(Y)]) = E(XY) - E(X)E(Y)
$$
而
$$
\rho = \frac{Cov(X, Y)}{\sqrt{D(X)} \sqrt{D(Y)}}
$$
称为随机变量 $X, Y$ 的相关系数。<br>

<font color=#f00000>解释</font>：$|\rho|$ 较大时，表明 $X, Y$ （就线性关系来说）联系较紧密，特别当 $|\rho|=1$ 时，$X, Y$ 之间存在线性关系。于是 $\rho$ 是一个可以用来表征 $X, Y$ 之间线性关系紧密程度的量。<br>

<font color=#f00000>$\rho = 0$ 称 $X, Y$ 不相关。当$X, Y$相互独立时，有：$Cov(X, Y) = 0, \rho = 0$。反之，若 $\rho=0$，$X, Y$不相关，$X, Y$ 不一定相互独立。<br>
因为：相关性 只是从线性关系来说的，而互相独立 是就更一般的关系而言。
</font>


<font color=#a020f0>对于正态分布而言，不相关 与 相互独立 是等价的。</font>

根据定义，可以推理出：
1. $Cov(X, Y) = Cov(Y, X), Cov(X, X) = D(X)$
2. $D(X+Y) = D(X) + D(Y) + 2Cov(X, Y)$
3. $Cov(aX, bY) = ab Cov(X, Y)$
4. $Cov(X_1+X_2, Y) = Cov(X_1, Y) + Cov(X_2, Y)$


**实例**：<br>
1. 正态分布 <br>
设 $X \sim N(\mu_1, \sigma^2_1), Y \sim N(\mu_2, \sigma^2_2)$，故
$$
Cov(X, Y) = \rho \sigma_1 \sigma_2
$$
这就是说，二维正态随机变量 $(X, Y)$ 的概率密度中的参数 $\rho$ 就是 $X, Y$ 的相关系数，因而二维正态随机变量的分布完全可由 $X, Y$ 各自的数学期望、方差以及它们的相关系数锁确定。<br>
<font color=#a020f0>若 $X, Y$ 服从二维正态分布，那么 $X, Y$ 相互独立的充要条件是：$\rho = 0$。</font>

{{< /alert >}}

### 4、矩

{{< alert type="info" >}}
设 $X, Y$ 是二维随机变量 <br>
**定义**：设 $X, Y$ 是随机变量，若
$$
E(X^k), k = 1, 2, ...
$$
存在，称它为 $X$ 的 <font color=#f00000>$k$ 阶原点矩</font>，简称 $k$ 阶矩。若
$$
E([X-E(X)]^k), k = 2, 3, ...
$$
存在，称它为 $X$ 的 <font color=#f00000>$k$ 阶中心矩</font>。若
$$
E([X-E(X)]^k [Y-E(Y)]^l), k,l = 1, 2, 3, ...
$$
存在，称它为 $X, Y$ 的 <font color=#f00000>$k+l$ 阶混合中心矩</font>。<br>

显然，$X$ 的数学期望 $E(X)$ 是 $X$ 的一阶原点矩，方差 $D(X)$ 是 $X$ 的二阶中心距，协方差 $Cov(X, Y)$ 是 $X, Y$ 的二阶混合中心距。<br>


二维随机变量 $(X_1, X_2)$ 有四个二阶中心矩，设它们都存在，分别记为：<br>
$c_{11} = E([X_1 - E(X_1)]^2)$ <br>
$c_{12} = E([X_1 - E(X_1)][X_2 - E(X_2)])$ <br>
$c_{21} = E([X_2 - E(X_2)][X_1 - E(X_1)])$ <br>
$c_{22} = E([X_2 - E(X_2)]^2)$ <br>
将它们排成矩阵的形式：
$$
\begin{pmatrix}
   c_{11} & c_{12} \\\
   c_{21} & c_{22}
\end{pmatrix}
$$
这个矩阵称为随机变量 $(X_1, X_2)$ 的 <font color=#f00000>协方差矩阵</font>。 <br>

更为一般的 $n$ 维随机变量 $X_1, X_2, ..., X_n$ 的协方差矩阵：
$$
\begin{pmatrix}
   c_{11} & c_{12} & ... & c_{1n} \\\
   c_{21} & c_{22} & ... & c_{2n} \\\
   \vdots & \vdots & ... & \vdots \\\
   c_{n1} & c_{n2} & ... & c_{nn}
\end{pmatrix}
$$
由于 $c_{ij} = c_{ji}, i \ne j$，所以 上述矩阵式一个对称矩阵。<br>

<font color=#c020f0>一般，$n$ 维随机变量的分布是不知道的，或者太复杂，以至在数学上不易处理，因此，在实际应用中协方差矩阵就显得重要了。</font> <br> 
比如：$n$ 维正态随机变量的概率密度：<br>
先看二维正态随机变量的概率密度，改写成另一种形式，以便将它推广到 $n$ 维随机变量的场合中区。
$$
f(x_1, x_2) = \frac{1}{2\pi\sigma_1\sigma_2 \sqrt{1-\rho^2}} exp\\{ -\frac{\frac{(x_1-\mu_1)^2}{\sigma_1^2} - 2\rho \frac{(x_1 - \mu_1)(x_2 - \mu_2)}{\sigma_1 \sigma_2} + \frac{(x_2 - \mu_2)^2}{\sigma_2^2}}{2(1-\rho^2)} \\}
$$
写成矩阵形式：
$$
\mathbf{X} = \begin{pmatrix}
x_1 \\\
x_2 
\end{pmatrix}, \mathbf{\mu} = \begin{pmatrix} \mu_1 \\\ \mu_2 \end{pmatrix}, 
\mathbf{C} = \begin{pmatrix} c_{11} & c_{12} \\\ c_{21} & c_{22} \end{pmatrix} = \begin{pmatrix} \sigma_1^2 & \rho \sigma_1 \sigma_2 \\\ \rho \sigma_1 \sigma_2 & \sigma_2^2 \end{pmatrix}
$$
行列式 $det \mathbf{C} = \sigma_1^2 \sigma_2^2 (1-\rho^2)$，$\mathbf{C}$ 的逆矩阵：
$$
\mathbf{C^{-1}} = \frac{1}{det \mathbf{C}} \begin{pmatrix} \sigma_2^2 & -\rho \sigma_1 \sigma_2 \\\ -\rho \sigma_1 \sigma_2 & \sigma_1^2 \end{pmatrix}
$$
开始变换矩阵形式：
$$
(\mathbf{X - \mu})^T \mathbf{C}^{-1} (\mathbf{X - \mu}) \\\
= \frac{1}{det \mathbf{C}} \begin{pmatrix} x_1 - \mu_1 & x_2 - \mu_2 \end{pmatrix} \begin{pmatrix} \sigma_2^2 & -\rho \sigma_1 \sigma_2 \\\ -\rho \sigma_1 \sigma_2 & \sigma_1^2 \end{pmatrix} \begin{pmatrix} x_1 - \mu_1 \\\ x_2 - \mu_2 \end{pmatrix} \\\
= \frac{1}{1-\rho^2} [\frac{(x_1-\mu_1)^2}{\sigma_1^2} - 2\rho \frac{(x_1 - \mu_1)(x_2 - \mu_2)}{\sigma_1 \sigma_2} + \frac{(x_2 - \mu_2)^2}{\sigma_2^2}]
$$
于是 $X_1, X_2$ 的概率密度可以写成：
$$
f(x_1, x_2) = \frac{1}{(2\pi)^{2/2} (det \mathbf{C})^{1/2}} exp\\{ -\frac{1}{2} (\mathbf{X - \mu})^T \mathbf{C}^{-1} (\mathbf{X - \mu}) \\}
$$

<font color=#f00000>推广到 $n$ 维正态随机变量</font>：
$$
f(x_1, x_1, ..., x_n) = \frac{1}{(2\pi)^{n/2} (det \mathbf{C})^{1/2}} exp\\{ -\frac{1}{2} (\mathbf{X - \mu})^T \mathbf{C}^{-1} (\mathbf{X - \mu}) \\}
$$
其中
$$
\mathbf{X} = \begin{pmatrix} x_1 \\\ x_2 \\\ \vdots \\\ x_n \end{pmatrix}, 
\mathbf{\mu} = \begin{pmatrix} \mu_1 \\\ \mu_2 \\\ \vdots \\\ \mu_n \end{pmatrix}, 
\mathbf{C} = \begin{pmatrix}
   c_{11} & c_{12} & ... & c_{1n} \\\
   c_{21} & c_{22} & ... & c_{2n} \\\
   \vdots & \vdots & ... & \vdots \\\
   c_{n1} & c_{n2} & ... & c_{nn}
\end{pmatrix}
$$

{{< /alert >}}

---

{{< alert type="success" >}}
$n$ 维正态随机变量具有以下四条重要性质：
1. $n$ 维正态随机变量 $X_1, X_2, ..., X_n$ 的每一个分量 $X_i$ 都是正态随机变量，反之，若 $X_1, X_2, ..., X_n$ 都是正态随机变量，且相互独立，则$X_1, X_2, ..., X_n$ 是 $n$ 维正态随机变量。

2. $n$ 维正态随机变量 $X_1, X_2, ..., X_n$ 服从 $n$ 维正态分布的充要条件是 $X_1, X_2, ..., X_n$ 的任意的线性组合
$$
l_1 X_1 + l_2 X_2 + ... + l_n X_n
$$
服从一维正态分布（其中 $l_1, l_2, ..., l_n$ 不全为零）

3. 若 $X_1, X_2, ..., X_n$ 服从 $n$ 维正态分布，设 $Y_1, Y_2, ..., Y_k$ 是 $X_j (j = 1, 2, ..., n)$ 的线性函数，则 $Y_1, Y_2, ..., Y_k$ 也服从多维正态分布。 这一性质称为 <font color=#f00000>正态变量的线性变换不变性</font>。

4. 若 $X_1, X_2, ..., X_n$ 服从 $n$ 维正态分布，则 <font color=#a020c0> $X_1, X_2, ..., X_n$ 相互独立</font> 与 <font color=#a020c0> $X_1, X_2, ..., X_n$ 两两不相关</font> 是等价的。

{{< /alert >}}

## 参考

### 1、极限
1. e
$$
lim_{n \to \infty} (1-\frac{\lambda}{n})^n = e^{-\lambda}
$$




### 2、积分

1. 根据泰勒公式证明
$$
\sum^{\infty}_{k=0} \frac{\lambda^k}{k!} = e^{\lambda}
$$ 

2. 积分
$$
\int^\infty_{-\infty} e^{-x^2} dx = \sqrt{\pi}
$$

3. 利用极坐标系 证明：$dxdy = rdrd\theta$ 即：
$$
\int^\infty_{-\infty} e^{- \frac{t^2}{2}} dt = \sqrt{2\pi}
$$

4. 求积分
$$
\int^\infty_{-\infty} x^2 e^{-\frac{x^2}{2}} dx = \sqrt{2\pi}
$$

5. 求积分
$$
\int^\infty_0 x e^{-x} dx = 1
$$

