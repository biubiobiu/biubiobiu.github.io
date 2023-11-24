---
title: "大数定律"
date: 2023-08-01T06:00:20+08:00
menu:
  sidebar:
    name: 大数定律
    identifier: math-large-numbers
    parent: math-probability-theory
    weight: 30
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["概率论","大数定律","中心极限定理"]
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

## 一、大数定理


{{< alert type="info" >}}
**弱大数定理（辛钦大数定理）**：<br>
设 $X_1, X_2, ...$ 相互独立，服从同一分布的随机变量序列，且具有数学期望 $E(X_k) = \mu , (k = 1, 2, ...)$ ，作前 $n$ 个变量的算数平均 $\frac{1}{n} \sum^n_{k=1} X_k$ ，则对于任意 $\varepsilon > 0$ 有
$$
\lim\limits_{n \to \infty} P\\{ |\frac{1}{n} \sum^n_{k=1} X_k - \mu| < \varepsilon \\} = 1
$$
通俗地说：辛钦大数定理是说，对于<font color=#f00000> 独立同分布且具有均值</font>的随机变量 $X_1, X_2, ..., X_n$ ，当 $n$ 很大时它们的算术平均 $\frac{1}{n} \sum^n_{k=1} X_k$ 很可能接近于 $\mu$ 。<br>
个人理解：$X_1, X_2, ..., X_n$ 这么多的随机变量，当取一个时刻的值时，其每项的值是按照同一分布独立出现的，本身就类似于 在同一个分布里随机取 $n$ 次数，当 $n \to \infty$ 时，均值肯定是 $\mu$

**伯努利大数定律**：<br>
设 $f_A$ 是 $n$ 次独立重复试验中事件 $A$ 发生的次数，$p$ 是事件 $A$ 在每次试验中发生的概率，则对于任意整数 $\varepsilon > 0$ 有
$$
\lim\limits_{n \to \infty} P\\{ |\frac{f_A}{n} - p| < \varepsilon \\} = 1
$$
或者
$$
\lim\limits_{n \to \infty} P\\{ |\frac{f_A}{n} - p| >= \varepsilon \\} = 0
$$

伯努利大数定理说明：在 $n$ 充分大时，事件 <font color=#f00000>频率 $\frac{f_A}{n}$ 与 频率 $p$ 的偏差小于 $\varepsilon$</font> 实际上几乎是必定要发生的，这就是我们所说的 频率稳定性的真正含义。所以，在实际应用中，当试验次数很大时，可以用事件的频率来替代事件的概率。

{{< /alert >}}


## 二、中心极限定理

{{< alert type="success" >}}

**中心极限定理**：<br>
<font color=#a020f0>客观背景</font>：在客观实际中有很多<font color=#f00000>随机变量</font>，它们是由大量的相互独立的随机因素的综合影响所形成的，而其中每一个因素在总的影响中所起的的作用都是微小的，这种<font color=#f00000>随机变量</font>往往近似地服从正态分布。</font> <br>

<font color=#a020f0>定理一（独立同分布的中心极限定理）</font>：<br>
设随机变量 $X_1, X_2, ..., X_n, ...$ 相互独立，服从同一分布，且具有数学期望和方差： $E(X_k) = \mu, D(X_k) = \sigma^2 > 0$，则随机变量之和 $\sum^n_{k=1} X_k$ 的标准化变量：
$$
Y_n = \frac{\sum^n_{k=1} X_k - E(\sum^n_{k=1} X_k)}{\sqrt{D(\sum^n_{k=1}X_k)}} = \frac{\sum^n_{k=1} X_k - n\mu}{\sqrt{n}\sigma}
$$
的分布函数 $F_n(x)$ 对于任意 $x$ 满足
$$
\lim\limits_{n \to \infty} F_n(x) = \lim\limits_{n \to \infty}P\\{ \frac{\sum^n_{k=1} X_k - n\mu}{\sqrt{n}\sigma} <= x \\} = \int^x_{-\infty} \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dt = \Phi(x)
$$
这就是说：独立同分布的随机变量 $X_1, X_2, ..., X_n, ...$ 之和的标准化变量，当 $n$ 充分大时，近似地服从 标准正态分布。
$$
\frac{\sum^n_{k=1} X_k - n\mu}{\sqrt{n}\sigma} \overset{\mathrm{近似}}{\sim} N(0, 1)
$$
<font color=#f00000>好处</font>：一般情况下，很难求出 $n$ 个随机变量之和 $\sum^n_{k=1} X_k$ 的分布函数，通过上面的定理可知，当 $n$ 充分大时，可以通过 $Phi(x)$ 给出其近似的分布，这样，就可以用正态分布对 $\sum^n_{k=1} X_k$ 作理论分析。<br>
<font color=#f00000>可以改写为</font>：
$$
\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \overset{\mathrm{近似}}{\sim} N(0, 1) 或 \bar{X} \overset{\mathrm{近似}}{\sim} N(\mu, \frac{\sigma^2}{n})
$$

<font color=#a020f0>定理二（李雅普诺夫(Lyapunov)定理）</font>：<br>
设随机变量 $X_1, X_2, ..., X_n, ...$ 相互独立，它们具有数学期望和方差：$E(X_k) = \mu_k, D(X_k) = \sigma^2_k > 0$，<br> 记：$B^2_n = \sum^n_{k=1}\sigma^2_k$，<br>

若：存在整数 $\delta$，使得当 $n \to \infty$ 时，
$$
\frac{1}{B^{2+\delta}_n} \sum^n_1 E\\{|X_k - \mu_k|^{2+\delta}\\} \to 0
$$

则：随机变量之和 $\sum^n_{k=1} X_k$ 的标准化变量
$$
Z_n = \frac{\sum^n_{k=1} X_k - E(\sum^n_{k=1} X_k)}{\sqrt{D(\sum^n_{k=1}X_k)}} = \frac{\sum^n_{k=1} X_k - \sum^n_{k=1} \mu_k}{B_n}
$$

的分布函数 $F_n(x)$ 对于任意 $x$ ，满足：
$$
\lim\limits_{n \to \infty} F_n(x) = \lim\limits_{n \to \infty}P\\{ \frac{\sum^n_{k=1} X_k - \sum^n_{k=1}\mu_k}{B_n} <= x \\} = \int^x_{-\infty} \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dt = \Phi(x)
$$

<font color=#f00000>这就是说</font>：无论各个随机变量 $X_1, X_2, ..., X_n, ...$ 服从什么分布，只要满足定理的条件，那么它们的和 $\sum^n_{k=1} X_k$ 当 $n$ 很大时，就近似地服从正态分布。<br>
很多问题中，所考虑的随机变量可以表示成很多个独立的随机变量之和，例如：一个物理实验的测量误差是由很多观察不到的、可加微小误差所合成的，它们往往近似地服从正态分布。

<font color=#a020f0>定理三（棣莫弗-拉普拉斯(De Moivre-Laplace)定理）</font>：<br>
设随机变量 $\eta_n (n=1, 2, ...)$ 服从参数为 $n, p (0 < p < 1)$ 的二项分布，则对于任意 $x$ ，有
$$
\lim\limits_{n \to \infty}P\\{ \frac{\eta_n - np}{\sqrt{np(1-p)}} <= x \\} = \int^x_{-\infty} \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dt = \Phi(x)
$$

证明：<br>
二项分布 $\eta_n$ 可以看成是 $n$ 个独立、同分布的 $(0-1)$ 分布的诸随机变量之和，即：$\eta_n = \sum^n_{jk=1} X_k$。所以，根据上面几个定理，就可以得证。


{{< /alert >}}



