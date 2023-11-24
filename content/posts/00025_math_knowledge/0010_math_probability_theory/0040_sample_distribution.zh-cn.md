---
title: "样本及抽样分布"
date: 2023-08-01T06:00:20+08:00
menu:
  sidebar:
    name: 样本及抽样分布
    identifier: math-sample-distribution
    parent: math-probability-theory
    weight: 40
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["概率论","采样","分布"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

<!--
{{< alert type="success" >}}
{{< /alert >}}

<font color=#f00000> </font>

-->

前面介绍了概率论的基本内容，接下来以概率论为基础，根据试验或观察得到的数据，来研究随机现象，对研究对象的客观规律性做出种种合理的估计和判断。

<font color=#f00000>数理统计</font>：的内容包括：如何收集、整理数据资料；如何对所得的数据资料进行分析、研究，从而对研究的对象的性质、特点做出判断。在数理统计中，我们研究的随机变量，其分布是未知的，或者是不完全知道的。人们是通过对所研究的随机变量进行重复独立的观察，得到许多观察值，对这些数据进行分析，从而对所研究的随机变量的分布作出种种推断。

## 一、随机样本

{{< alert type="info" >}}
**总体**：试验的全部可能的观察值。<br>
**个体**：每一个可能观察值。<br>
**容量**：总体中所包含的个体的个数。<br>

**定义**：<br>
设 $X$ 是具有分布函数 $F$ 的随机变量，若 $X_1, X_2, ..., X_n$ 是具有同一分布函数 $F$ 的相互独立的随机变量，则称 $X_1, X_2, ..., X_n$ 为从分布函数 $F$ (或者总体 $F$、或 总体 $X$) 得到的<font color=#f00000>容量为 $n$ 的简单随机样本</font>，简称 <font color=#f00000>样本</font>。它们的观察值 $x_1, x_2, ..., x_n$ 称为<font color=#f00000>样本值</font>，又称为 $X$ 的 $n$ 个独立的<font color=#f00000>观察值</font>。<br>

也可以将样本看成是一个随机向量，写成 $(X_1, X_2, ..., X_n)$。由定义得：若 $X_1, X_2, ..., X_n$ 为 $F$ 的一个样本，则 $X_1, X_2, ..., X_n$ 相互独立，且它们的分布函数都是 $F$ ，所以 $(X_1, X_2, ..., X_n)$ 的分布函数为：
$$
F^*(x_1, x_2, ..., x_n) = \prod^n_{i=1} F(x_i)
$$

概率密度函数：
$$
f^*(x_1, x_2, ..., x_n) = \prod^n_{i=1} f(x_i)
$$

{{< /alert >}}


## 二、抽样分布

样本是进行统计推断的依据，在应用时，往往不是直接使用样本本身，而是针对不同的问题构造样本的适当函数，利用这些样本的函数进行统计推断。<br>

{{< alert type="info" >}}
<font color=#a020f0>定义</font>: <br>
设 $X_1, X_2, ..., X_n$ 是来自总体 $X$ 的一个样本，$g(X_1, X_2, ..., X_n)$ 是 $X_1, X_2, ..., X_n$ 的函数，若 $g$ 函数中不含未知参数，则称 $g(X_1, X_2, ..., X_n)$ 是一<font color=#f00000>统计量</font> <br>

即：统计量 $g(X_1, X_2, ..., X_n)$ 是随机变量 $X_1, X_2, ..., X_n$ 的函数，因此，统计量是一个随机变量。<br>

设：$x_1, x_2, ..., x_n$ 是相应样本 $X_1, X_2, ..., X_n$ 的样本值，则称 $g(x_1, x_2, ..., x_n)$ 是 $g(X_1, X_2, ..., X_n)$ 的观察值。<br>


<font color=#a020f0>常用的统计量</font>: <br>

<font color=#f00000>样本均值</font>
$$
\bar{X} = \frac{1}{n} \sum^n_{i=1} X_i
$$

<font color=#f00000>样本方差</font>
$$
S^2 = \frac{1}{n-1} \sum^n_{i=1} (X_i - \bar{X})^2 = \frac{1}{n-1} (\sum^n_{i=1} X^2_i - n \bar{X}^2)
$$

<font color=#f00000>样本标准层</font>
$$
S = \sqrt{S^2} = \sqrt{\frac{1}{n-1} (\sum^n_{i=1} X^2_i - n \bar{X}^2)}
$$

<font color=#f00000>样本 $k$ 阶(原点)矩</font>
$$
A_k = \frac{1}{n} \sum^n_{i=1} X^k_i, k = 1, 2, ...
$$
若：总体 $X$ 的 $k$ 阶矩 $E(X^k) \overset{\mathrm{记作}}{==} \mu_k$ <br>
则：由辛钦大数定理知：当 $n \to \infty$ 时，$A_k \overset{\mathrm{P}}{\to} \mu_k, k=1, 2, ...$ <br>
依概率收敛的序列的性质知道：$g(A_1, A_2, ..., A_k) \overset{\mathrm{P}}{\to} g(\mu_1, \mu_2, ..., \mu_k)$。其中，$g$ 为连续函数。这就是 <font color=#f00000>矩估计法</font> 的理论基础。

<font color=#f00000>样本 $k$ 阶中心矩</font>
$$
B_k = \frac{1}{n} \sum^n_{i=1}(X_i - \bar{X})^k, k = 2, 3, ...
$$

{{< /alert >}}

---

{{< alert type="success" >}}
<font color=#a020f0>经验分布函数</font> ：与 总体分布函数$F(x)$ 相应的统计量 <br>
它的作法如下：<br>
设：$X_1, X_2, ..., X_n$ 是 总体 $F$ 的一个样本，用 $S(x), -\infty < x < \infty$ 表示 $X_1, X_2, ..., X_n$ 中不大于 $x$ 的随机变量的个数。定义 经验分布函数 $F_n(x)$ 为
$$
F_n(x) = \frac{1}{n} S(x), -\infty < x < \infty
$$

对于一个样本值，其经验分布函数 $F_n(x)$ 的观察值是很容易得到的。<br>

格里汶科（Glivenko）在1933年证明了一下结论：<br>
对于任一实数 $x$，当 $n \to \infty$ 时，$F_n(x)$ 以概率 1 一致收敛于分布函数 $F(x)$，即：
$$
P\\{ \lim\limits_{n \to \infty} sup_{-\infty < x < \infty} |F_n(x) - F(x)| = 0 \\} = 1
$$

因此，对于任一实数 $x$ 当 $n$ 充分大时，经验分布函数的任一个观察值 $F_n(x)$ 与总体分布函数 $F(x)$ 只有微小的差别，从而在实际上可当做 $F(x)$ 来使用。
{{< /alert >}}

---

{{< alert type="info" >}}
统计量的分布 称为 <font color=#f00000>抽样分布</font> <br>
1. 当总体的分布函数已知时，抽样分布是确定的，然而要求出统计量的精确分布，一般来说比较困难。
2. 使用统计量进行统计推断时，常需知道它的分布。

下面介绍来自正态总体的几个常用统计量的分布：

{{< /alert >}}

### 1、$\chi^2$分布

设 $X_1, X_2, ..., X_n$ 是来自总体 $N(0, 1)$ 的样本，则称统计量：
$$
\chi^2 = X_1^2 + X_2^2 + ... + X_n^2
$$

服从自由度为 $n$ 的 $\chi^2$ 分布，记为 $\chi^2 \sim \chi^2(n)$ <br>

$\chi^2(n)$ 分布的概率密度为
$$
f(y) = \begin{cases} \frac{1}{2^{\frac{n}{2}} \Gamma(\frac{n}{2})} y^{\frac{n}{2}-1} e^{-\frac{y}{2}} & y > 0 \\\ 0 & 其他 \end{cases}
$$

图形如下图：
<p align="center"><img src="/datasets/posts/maths/chi_n.png" width=60% height=60%></p>

{{< alert type="success" >}}

由：$\chi^2(1)$ 分布，就是 $\Gamma(\frac{1}{2}, 2)$ 分布 <br>

由：定义 $X_i^2 \sim \chi^2(1)$，即：$X_i^2 \sim \Gamma(\frac{1}{2}, 2), i = 1, 2, ..., n$ <br>

由：$X_1, X_2, ..., X_n$ 的独立性知，$X^2_1, X^2_2, ..., X^2_n$ 相互独立 <br>

由：$\Gamma$ 分布的可加性。<br>

可得：
$$
\chi^2 = \sum^n_{i=1} X^2_i \sim \Gamma(\frac{n}{2}, 2)
$$

根据 $\Gamma$ 分布的可加性，得 $\chi^2$ 分布的可加性如下：
1. <font color=#f00000> $\chi^2$ 分布的可加性</font>：设 $\chi^2_1 \sim \chi^2(n_1), \chi^2_2 \sim \chi^2(n_2)$，并且 $\chi^2_1, \chi^2_2$ 相互独立，则有：$\chi^2_1 + \chi^2_2 \sim \chi^2(n_1 + n_2) \sim \Gamma(\frac{n_1+n_2}{2}, 2)$

2. <font color=#f00000> $\chi^2$ 分布的期望和方差</font>：若 $\chi^2 \sim \chi^2(n)$，则有 <br>
$E(\chi^2) = n, D(\chi^2) = 2n$

3. <font color=#f00000> $\chi^2$ 分布的分位点</font>：对于给定的正数 $\alpha, 0 < \alpha < 1$ ，如果满足条件
$$
P\\{ \chi^2 > \chi^2_\alpha(n) \\} = \int^\infty_{\chi^2_\alpha(n)} f(y) dy = \alpha
$$
则，点 $\chi^2_\alpha(n)$ 为 $\chi^2(n)$ 分布的上 $\alpha$ 分位点，如图所示：<p align="center"><img src="/datasets/posts/maths/chi_0.png" width=40% height=40%></p>
费希尔(R.A.Fisher)曾证明，当 $n$ 充分大时，近似地有
$$
\chi^2_\alpha(n) \approx \frac{1}{2}(z_\alpha + \sqrt{2n-1})^2
$$
其中，$z_\alpha$ 是标准正态分布的上 $\alpha$ 分位点。


{{< /alert >}}

### 2、$t$ 分布

设 $X \sim N(0, 1), Y \sim \chi^2(n)$ ，且 $X, Y$ 相互独立，则称<font color=#f00000>随机变量 $t = \frac{X}{\sqrt{Y/n}}$ 服从自由度为 $n$ 的 $t$ 分布</font>，记为 $t \sim t(n)$ <br>

$t$ 分布 又称学生式(Student) 分布，$t(n)$ 分布的概率密度函数为：
$$
h(t) = \frac{\Gamma[\frac{n+1}{2}]}{\sqrt{n\pi} \Gamma[\frac{n}{2}]} (1+\frac{t^2}{n})^{-\frac{n+1}{2}}, -\infty < t < \infty
$$

<p align="center"><img src="/datasets/posts/maths/t_nor.png" width=50% height=50%></p>

从图中可以看到，$h(t)$ 的图形是关于 $t=0$ 对称，当$n$ 充分大时，$t$ 分布近似于 $N(0, 1)$ 分布。即：
$$
\lim\limits_{n \to \infty} h(t) = \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}}
$$

<font color=#f00000> $t$ 分布的分位点</font>：当 $n > 45$时，对于常数为 $\alpha$ 的值，就用正态近似
$$
t_\alpha(n) \approx z_\alpha
$$
<p align="center"><img src="/datasets/posts/maths/t_nor_0.png" width=50% height=50%></p>


### 3、$F$ 分布

设 $U \sim \chi^2(n_1), V \sim \chi^2(n_2)$，且 $U, V$ 相互独立，则称随机变量 $F = \frac{U/n_1}{V/n_2}$ 服从自由度为 $(n_1, n_2)$ 的 $F$ 分布，记为 $F \sim F(n_1, n_2)$。<br>

$F(n_1, n_2)$ 分布的概率密度为：
$$
\psi(y) = \begin{cases} \frac{\Gamma[\frac{n_1+n_2}{2}] \frac{n_1}{n_2} y^{\frac{n_1}{2}-1}}{\Gamma(\frac{n_1}{2}) \Gamma(\frac{n_2}{2}) [1+\frac{n_1y}{n_2}]^{\frac{n_1+n_2}{2}}}
& y > 0 \\\ 0 & 其他 \end{cases}
$$

画出 $\psi(y)$ 的图形：
<p align="center"><img src="/datasets/posts/maths/psi_0.png" width=50% height=50%></p>

<font color=#f00000> $F$ 分布的分位点</font>：有个重要额性质
$$
F_{1-\alpha}(n_1, n_2) = \frac{1}{F_\alpha(n_2, n_1)}
$$

<p align="center"><img src="/datasets/posts/maths/psi_1.png" width=50% height=50%></p>


### 4、正态总体的样本均值/样本方差的分布

设 总体 $X$ （不管服从什么分布，只要均值和方差都存在）的均值为 $\mu$，方差为 $\sigma^2$。<br>
$X_1, X_2, ..., X_n$ 是来自 $X$ 的一个样本，其中，$\bar{X}$ ：样本均值；$S^2$ ：样本方差。<br>
则：
$$
E(\bar{X}) = \mu, D(\bar{X}) = \frac{\sigma^2}{n}
$$
而
$$
E(S^2) = E[\frac{1}{n-1}(\sum^n_{i=1}X_i^2 - n\bar{X}^2)] = \frac{1}{n-1} [\sum^n_{i=1}E(X^2_i) - n E(\bar{X}^2)] = \frac{1}{n-1} [\sum^n_{i=1}(\sigma^2 + \mu^2) - n(\frac{\sigma^2}{n} + \mu^2)] = \sigma^2
$$

{{< alert type="success" >}}
设 $X \sim N(\mu, \sigma^2)$ ，则 $\bar{X} = \frac{1}{n} \sum^n_{i=1}X_i$ 也服从正态分布，有以下定理：<br>

<font color=#a020f0>定理一</font>：<br>
设 $X_1, X_2, ..., X_n$ 是来自正态总体 $N(\mu, \sigma^2)$ 的样本，$\bar{X}$ 是样本均值，则有
$$
\bar{X} \sim N(\mu, \frac{\sigma^2}{n})
$$

<font color=#a020f0>定理二</font>：<br>
设 $X_1, X_2, ..., X_n$ 是来自正态总体 $N(\mu, \sigma^2)$ 的样本，$\bar{X}$ 是样本均值，$S^2$ 是样本方差，则  $\bar{X}$ 与 $S^2$ 相互独立。 且
$$
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)
$$


<font color=#a020f0>定理三</font>：<br>
设 $X_1, X_2, ..., X_n$ 是来自正态总体 $N(\mu, \sigma^2)$ 的样本，$\bar{X}$ 是样本均值，$S^2$ 是样本方差，则有
$$
\frac{\bar{X} - \mu}{S/\sqrt{n}} \sim t(n-1)
$$
证明 由
$$
\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0, 1), \frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)
$$

<font color=#a020f0>定理四</font>：<br>
设 $X_1, X_2, ..., X_{n_1}$ 与 $Y_1, Y_2, ..., Y_{n_2}$ 分别是来自正态总体 $N(\mu_1, \sigma^2_1)$ 和 $N(\mu_2, \sigma^2_2)$ 的样本，且这两个样本相互独立。<br>

设 样本均值：$\bar{X} = \frac{1}{n_1} \sum^{n_1}_{i=1} X_i$ <br>

样本均值：$\bar{Y} = \frac{1}{n_2} \sum^{n_2}_{i=1} Y_i$ <br>

样本方差：$S^2_1 = \frac{1}{n_1 - 1} \sum^{n_1}_{i=1}(X_i - \bar{X})^2$ <br>

样本方差：$S^2_2 = \frac{1}{n_2 - 1} \sum^{n_2}_{i=1}(Y_i - \bar{Y})^2$  <br>

则有：$\frac{S_1^2 / S_2^2}{\sigma^2_1 / \sigma^2_2} \sim F(n_1 - 1, n_2 - 1)$ <br>

当 $\sigma^2_1 = \sigma^2_2 = \sigma^2$ 时：
$$
\frac{(\bar{X}-\bar{Y})-(\mu_1 - \mu_2)}{S_w \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} \sim t(n_1 + n_2 -2)
$$

其中，$S^2_w = \frac{(n_1 - 1)S^2_1 + (n_2 - 1)S^2_2}{n_1 + n_2 - 2}$

{{< /alert >}}

