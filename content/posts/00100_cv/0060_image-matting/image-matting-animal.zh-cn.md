---
title: "animal matting"
date: 2021-09-09T06:00:20+06:00
menu:
  sidebar:
    name: animal matting
    identifier: cv-image-matting-animal-github
    parent: cv-image-matting
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["matting", "CV"]
categories: ["Basic"]
---

## Image Matting

<a href="https://browse.arxiv.org/pdf/2010.16188.pdf" target="bland">《End-to-end Animal Image Matting》</a> <br>

### 1、先前算法的不足

**流水线**：全局分割和局部抠图<br>
其中前者的目标是trimap生成或前景/背景生成，后者是基于从前一阶段生成的trimap或其他先验的图像抠图。这种流水线的不足归因于它的顺序性<br>
1. 因为它们可能产生错误的语义，该错误不能通过后续的抠图步骤来纠正。
2. 两个阶段的单独训练方案可能由于它们之间的不匹配而导致次优解。

### 2、提出新结构

提出了一个新颖的 Glance and Focus Matting network (GFM)，它使用一个共享的编码器和两个独立的解码器来以协作的方式学习两个任务，用于端到端的动物图像抠图。
<p align="center"><img src="/datasets/posts/cnn/animal_matting_0.png" width=90% height=90%></p>

该结构可以粗略地描述为一个粗略的分割阶段和抠图阶段。请注意，这两个阶段可能是交织在一起的，因为在第一个阶段会有来自第二阶段的反馈来纠正错误的决定,将它们集成到单个模型中并明确地为协作建模是合理的。

网络结构是一个编码解码器的结构，编码器、两个平行的解码器（GD和FD）。<br>
然后，以不同的表征域（RoSTa），连接 GD 和 FD 的输出结果。<br>
最后，通过协同合作抠图（CM），将RoSTa中三个不同的表征域的结果，进行合并，获得最终的 alpha 预测<br>

**编码器**：<br>
以在 ImageNet 上预训练的 ResNet-34 或 DenseNet-121 作为编码器。将单个图像作为输入，通过五个 $E_0 - E_4$ 模块进行处理

**解码器**：<br>
1. Glance Decoder（GD）：<br>
旨在识别容易的语义部分，而将其他部分作为未知区域。模型采用了 $D^G_4 -- D^G_0$ ，每层的输出与编码器一一对应。为了进一步扩大感受野，在 $E_4$ 之后增加了一个金字塔汇集模块(PPM)以提取全局上下文。其损失函数为：交叉熵。
$$
L_{CE} = - \sum_{c=1}^C G^c_g log(G_p^c)
$$

2. Focus Decoder (FD)：<br>
FD旨在提取低层结构特征，即：非常有用的过渡区域的细节。模型采用了 $D^F_4 -- D^F_0$ ，每层的输出与编码器一一对应。使用一个bridge block(BB)来代替 $E_4$ 之后的PPM，以在不同的感受野中利用local context。来自 $E_4$ 和 $BB$ 的特征被连接，并馈入 $D^F_4$，遵循U-net风格，在每个编码器块 $E_i$ 和解码器块 $D^F_i$ 之间添加一个快捷方式，以保留精细细节。<br>
在未知的过渡区域中，训练损失由 α 预测损失和拉普拉斯损失组成： $ L_{FD} = L^T_{\alpha} + L^T_{lap} $ <br>
其中，$\alpha$ 预测的损失：
$$
L^T_{\alpha} = \frac{\sum_i \sqrt{((\alpha_i - \alpha_i^F) \times W_i^T)^2+\varepsilon^2}}{\sum_i W_i^T}
$$
拉普拉斯损失：
$$
L^T_{lap} = \sum_i W_i^T \sum_{k=1}^5|Lap^k(\alpha_i) - Lap^k(\alpha_i^F)|_1
$$

**综合**：<br>
Collaborative Matting (CM)：合并来自GD和FD的预测以产生最终的alpha预测。通过这种方式，GD通过学习全局语义特征来识别粗糙的前景和背景，并且FD负责通过学习局部结构特征来解决未知区域中的细节。<br>
协同抠图的训练损失由：$\alpha$ 预测损失 $L_{\alpha}$、拉普拉斯损失 $L_{lap}$和合成损失 $L_{comp}$组成，即: 
$$
L_{CM} = L_{\alpha} + L_{lap} + L_{comp}
$$

其中，合成损失：
$$
L_{comp} = \frac{\sum_i \sqrt{(C(\alpha_i) - C(\alpha_i^{CM}))^2+\varepsilon^2}}{N}
$$

综合以上损失：
$$
L = L_{CE} + L_{FD} + L_{CM}
$$


### 3、评估指标
研究的方法由两种，分别为主观评价方法和客观评价方法。
1. 主观评价方法：<br>
主要依据的是人体视觉感知神经在特定的观测环境下，参考图像亮度、图像噪声和图像模糊等指标，对被测图像进行主观评价打分。
虽然主观评价方法是人体的直观感受，肯定符合人眼对图像质量的评价，但是主观评价方法也有很多缺陷：
    * 首先人体主观评价很容易受到被测人情绪、偏好等个人主观原因的影响；
    * 其次测试环境等客观因素也会影响主观评价方法的结果；
    * 最后，在数据量特别大的时候，组织人员进行主观评价费时费力，图像质量评价的效率会很低下。

2. 客观评价方法<br>
    * 绝对误差和（SAD）<br> $SAD = \sum_i |a_i - a_i^*|$
    * 均方误差（MSE）<br> $MSE = \frac{1}{n} \sum_i(a_i - a_i^*)^q$
    * 梯度误差（Gradient error）<br> $\sum_i(\nabla a_i - \nabla a_i^*)^q$
    * 连通性误差（Connectivity error）<br> $\sum_i| \varphi(a_i, \varOmega) -  \varphi(a_i^*, \varOmega)|$
    * 平均绝对误差（MAD）<br>  $MSE = \frac{1}{n}\sum_i |a_i - a_i^*|$

