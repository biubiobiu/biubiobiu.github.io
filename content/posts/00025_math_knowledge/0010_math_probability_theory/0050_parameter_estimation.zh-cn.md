---
title: "假设检验"
date: 2023-08-01T06:00:20+08:00
menu:
  sidebar:
    name: 假设检验
    identifier: math-parameter-estimation
    parent: math-probability-theory
    weight: 50
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["概率论","假设检验"]
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

## 一、估计问题

### 1、点估计

> 设总体 $X$ 的分布函数的形式已知，但它的一个或多个参数位置，借助于总体 $X$ 的一个样本来估计未知参数的值的问题，<font color=#f00000>称为参数的点估计问题</font>。<br>

点估计问题的一般提法如下：<br>
设总体 $X$ 的分布函数 $F(x; \theta)$ 的形式为已知，$\theta$ 是待估参数。$X_1, X_2, ..., X_n$ 是 $X$ 的一个样本，$x_1, x_2, ..., x_n$ 是相应的一个样本值 <br>
点估计问题就是要构造一个适当的统计量 $\hat{\theta} (X_1, X_2, ..., X_n)$，用它的观察值 $\hat{\theta}(x_1, x_2, ..., x_n)$ 作为未知参数 $\theta$ 的近似值。我们称 $\hat{\theta} (X_1, X_2, ..., X_n)$ 为 $\theta$ 的估计量，称$\hat{\theta}(x_1, x_2, ..., x_n)$ 为 $\theta$ 的估计值。

<font color=#f00000>个人理解：由于很多分布的均值、方差，这些数字特征就能完全决定分布的情况，所以，在构建统计量时，就可以用样本均值、样本方差 这些统计量。</font>。<br>

#### a. 矩估计法

设 $X$ 为连续型随机变量，其概率密度为 $f(x; \theta_1, \theta_2, ..., \theta_n)$，或 $X$ 为离散型随机变量。其中，$\theta_1, \theta_2, ..., \theta_n$ 为待估参数，$X_1, X_2, ..., X_n$ 是来自 $X$ 的样本，假设总体 $X$ 的前 $k$ 阶矩 存在
$$
\mu_l = E(X^l) = \int^\infty_{-\infty} x^l f(x; \theta_1, \theta_2, ..., \theta_n) dx
$$
离散型：
$$
\mu_l = E(X^l) = \sum_{x \in R_X} x^l p(x; \theta_1, \theta_2, ..., \theta_n)
$$

<font color=#f00000>理论基础：样本矩 $A_l = \frac{1}{n} \sum^n_{i=1}X^l_i$ 依概率收敛于相应的总体矩 $\mu_l$</font>。所以，我们就用样本矩作为相应的总体矩的估计量。<br>

所以，这样就可以有
$$
\begin{cases}
A_1 = \mu_1 & = \mu_1(\theta_1, \theta_2, ..., \theta_k) \\\
A_2 = \mu_2 & = \mu_2(\theta_1, \theta_2, ..., \theta_k) \\\
& \vdots \\\
A_k = \mu_k & = \mu_1(\theta_1, \theta_2, ..., \theta_k)
\end{cases}
$$

这样就可以用 $k$ 个联立方程组，求解 $k$ 个未知参数 $\theta_1, \theta_2, ..., \theta_k$.

> 以样本矩的连续函数作为相应的总体矩的连续函数的估计量，这种估计方法称为 矩估计法。<br>
> $\hat{\theta}_i = \theta(A_1, A_2, ..., A_n)$ 作为 $\theta_i$ 的估计量，这种估计量称为 <font color=#f00000>矩估计量</font>


#### b. 最大似然估计法

若总体 $X$ 是离散型，其分布律 $P(X=x) = p(x; \theta), \theta \in \Theta$ 的形式为已知，$\theta$ 为待估参数，$\Theta$ 是 $\theta$肯能取值的范围，设 $X_1, X_2, ..., X_n$ 是来自$X$ 的样本，则 $X_1, X_2, ..., X_n$ 的联合分布律为
$$
\prod_{i=1}^N p(x_i; \theta)
$$

样本 $X_1, X_2, ..., X_n$ 取观察值 $x_1, x_2, ..., x_n$ 的概率，亦即时间 $ \\{X_1=x_1, X_2=x_2, ..., X_n=x_n \\}$ 发生的概率为：
$$
L(\theta) = L(x_1, x_2, ..., x_n; \theta) = \prod_{i=1}^N p(x_i; \theta)
$$

这一概率随 $\theta$ 的取值而变化，它是 $\theta$ 的函数，$L(\theta)$ 称为样本的 <font color=#f00000>似然函数</font>。<br>

直观想法：现在已经取到样本值 $x_1, x_2, ..., x_n$ 了，这表明取到这一样本值的概率 $L(\theta)$ 比较大，如果已知 $\theta = \theta_0$ 时使得 $L(\theta)$ 取很大值，我们自然认为取 $\theta_0$ 作为未知参数 $\theta$ 的估计值，比较合理。<br>

费希尔(R.A.Fisher) 引进的最大似然估计，就是固定样本观察值 $x_1, x_2, ..., x_n$，在$\theta$ 取值的可能范围内，挑选使得 似然函数 $L(x_1, x_2, ..., x_n; \theta)$ 达到最大的参数值 $\hat{\theta}$，作为参数 $\theta$ 的估计值。使得
$$
L(x_1, x_2, ..., x_n; \hat{\theta}) = max_{\theta \in \Theta} L(x_1, x_2, ..., x_n; \theta)
$$

这样得到的 $\hat{\theta}$ 与 样本值 $x_1, x_2, ..., x_n$ 有关，记 $\hat{\theta}(x_1, x_2, ..., x_n)$ ，称为 参数 $\theta$ 的<font color=#f00000>最大似然估计值</font>。<br>

对于连续型的：
$$
L(\theta) = L(x_1, x_2, ..., x_n; \theta) = \prod_{i=1}^N f(x_i; \theta)
$$

<mark>对数似然方程</mark> <br>
在多数情况下，$ln L(\theta)$ 比 $L(\theta)$ 更容易求解极限。所以，$\theta$ 的最大似然估计 $\hat{\theta}$ ，可以从方程：
$$
\frac{d ln L(\theta)}{d \theta} = 0
$$
求得。

<mark>多个未知参数</mark>： <br>
最大似然估计法也使用与分布中含多个未知参数 $\theta_1, \theta_2, ..., \theta_k$ 的情况，这时，似然函数 $L$ 是这些位置参数的函数，分别令
$$
\frac{\partial L}{\partial \theta_i} = 0, i = 1, 2, ..., k
$$

这样就可以得到 $k$ 个方程 组成的方程组。


### 2、估计量的评选标准

{{< alert type="success" >}}

<font color=#a020f0>无偏性</font> <br>
$$
E(\hat{\theta}) = \theta
$$
称 $\hat{\theta}$ 是 $\theta$ 的无偏估计量。

<font color=#f00000>不论总体服从什么分布，$k$阶样本矩 $A_k = \frac{1}{n} \sum^n_{i=1}X^k_i$ 是 $k$ 阶总体矩 $\mu_k$ 的无偏估计量。</font>

<font color=#a020f0>有效性</font> <br>
设 $\hat{\theta_1}$ 与 $\hat{\theta_2}$ 都是 $\theta$ 的无偏估计量，若对于任意 $\theta \in \Theta$ 有
$$
D(\hat{\hat{\theta_1}}) <= D(\hat{\hat{\theta_2}})
$$

则称 $\hat{\theta_1}$ 比 $\hat{\theta_2}$ 有效。

<font color=#a020f0>相合性</font> <br>
设 $\hat{\theta}$ 为参数 $\theta$ 的估计量，若对于任意 $\theta \in \Theta$ ，当 $n \rarr \infty$ 时，$\hat{\theta}$ 依概率收敛于 $\theta$，则称 $\hat{\theta}$ 为 $\theta$ 的相合估计量。<br>

<font color=#f00000>样本 $k$ 阶矩是总体 $X$ 的 $k$ 阶矩 $\mu_k$ 的相合估计量。</font>

相合性是对一个估计量的基本要求，若估计量不具有相合性，那么不论将样本容量 $n$ 取得多大，都不能将 $\theta$ 估计得足够准确，这样的估计量是不可取的。

{{< /alert >}}


### 3、区间估计



## 二、检验问题


