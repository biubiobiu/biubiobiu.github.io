---
title: "vision transformer"
date: 2022-05-09T06:00:20+06:00
menu:
  sidebar:
    name: vision transformer
    identifier: cv-backbone-vit
    parent: cv-vision-transformer
    weight: 40
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["backbone","vision transformer"]
categories: ["Basic"]
math: true
---


## 一、简介

### 1、Transformer用在CV领域
在NLP中，Transformer的输入是一个时间步长为T的序列，比如：basic版bert，T=512，每个token embeding为768维特征。如何把二维图片转化为一维呢？<br>
  - $\bf \color{red} \times$ 如果把每个像素点看做是一个样本，铺平后是一维序列。但是，图片大小 224*224=50176，远远大于Transformer的最大序列长度。
  - $\bf \color{red} \times$ 卷积和Transformer一起用：<a href="https://arxiv.org/abs/1711.07971" target="blank">《Non-local Neural Networks》</a>(2018)、<a href="https://arxiv.org/abs/2005.12872" target="blank">《End-to-End Object Detection with Transformers》</a>(2020) 为了减小序列的长度，不直接使用输入图片，而是使用feature map 转换为序列。比如：ResNet50在最后的阶段的输出尺寸为 14x14，拉平后序列长度只有196。
  - $\bf \color{red} \times$ 抛弃卷积使用定制化的自注意力机制：<a href="https://arxiv.org/abs/1906.05909" target="blank">《Stand-Alone Self-Attention in Vision Models》</a>(2019) 采用的是 孤立自注意力。用一个局部的小窗口做自注意力； <a href="https://arxiv.org/abs/2003.07853" target="blank">《Stand-alone axial-attention for panoptic segmentation》</a>(2020) 采用的是轴注意力。在高度的方向上做自注意力、在宽度方向做自注意力。由于这些自注意力机制比较定制化，还没有在硬件上大规模加速计算，所以网络做不大。
  - $\color{green} \checkmark$ 对图片做些预处理，直接使用Transformer：将图片切分成一个个patch，然后每个patch作为一个token输入到Transformer中。
      1. $224 \times 224$ 的图片，切分成一个个 $16 \times 16$ 的patch，最终切分出196个patch；每个patch的大小是：$16 \times 16 \times 3=768$，刚好是basic版bert每个token的维度。
      2. 多头注意力机制，12个头，每个头的k、q、v对应的维度是64维




## 二、网络

### 1、ViT
<a href="https://arxiv.org/abs/2010.11929" target="blank">ViT</a>(2021) 直接把Transformer应用到图像处理，尽量改动最少，所以只对图像做预处理，让其符合NLP的输入形式， 思路：
  1. 图片尺寸 224x224，将图片切分成一个个patch，patch的大小16x16，每个patch作为一个token，即：14x14=196个patch，每个patch长16x16x3=768
  2. 学习一个线性矩阵$E$，尺寸为768x768，对每个patch做线性变换。多头注意力的话，basic版本12个头，所以12个196x64拼接起来，还是196x768。
  3. 位置编码：可学习的位置向量，尺寸为196x768
  4. cls的输出作为提取的图片特征，用于后续的分类操作

<p align="center"><img src="/datasets/posts/cnn/vit.jpg" width="70%" height="70%" title="ViT" alt="ViT"></p>

实验结论：
  1. 额外使用cls做分类，为啥不用GAP呢？作者对比了这两种方式，发现如果参数调整好的话效果是一样的。所以为了在使用Transformer时改动最少，所以继续使用cls作为分类。
  2. 位置编码信息，每个位置编码信息是一个768维度的向量。图像是二维的，所以是否需要在位置编码里体现二维特征呢？作者实验发现，即使是一维的768维度的向量，其中也会学习到二维特征，所以没有必要故意设计二维特征。
  3. 归纳偏置(inductive bias)，在cnn中位置信息是贯穿在所有卷积操作中的，卷积是线性操作，具有偏移不变性。在ViT中除了添加了位置编码信息，是没有其他的空间信息的。所以作者认为是这个原因导致ViT在小规模的数据集上效果不好，在中/大规模的数据集上效果较好。
  4. Transformer有处理长文本的能力，所以ViT能够捕获图片的全局信息，而并不是像cnn只用到感受野区域。
  5. 作者还提出是否可以类比BERT，mask掉一些patch，然后自监督训练，修复出mask掉的区域。这就是后来何大神的MAE。
  6. ViT可以说是打通了CV和NLP的鸿沟，对多模态具有重要的意义。

可视化显示模型能学习的信息：
{{< split 6 6>}}
<p align="center"><img src="/datasets/posts/cnn/vit-1.jpg" title="ViT" alt="ViT"></p>
可学习的线性矩阵$\bf E$，具体学习的信息：跟CNN很像，可以学习到颜色、纹理、等底层信息。
---
<p align="center"><img src="/datasets/posts/cnn/vit-2.jpg" width="70%" height="70%" title="ViT" alt="ViT"></p>
位置信息：发现可以学习到位置信息，同时也可以学习到行、列的信息；这就是为啥没有必要设计2维的位置信息。
{{< /split >}}


{{< split 6 6>}}
<p align="center"><img src="/datasets/posts/cnn/vit-3.jpg" width="80%" height="80%" title="ViT" alt="ViT"></p>
图中，每列都有16个点，就是16个头的输出；纵轴：平均注意力的距离；横轴：网络的层数。图展示了ViT能不能注意到全局信息: a. 在前几层有相近的、距离远的，这说明，在刚开始模型就能注意到较远的像素，而不像CNN前几层因为感受野较小只能注意到相近的像素点；b. 在后几层距离都比较远，说明网络学习到的特征越来越具有语义信息。
---
<p align="center"><img src="/datasets/posts/cnn/vit-4.jpg" width="68%" height="68%" title="ViT" alt="ViT"></p>
作者用网络的最后一层的输出，映射回输入图片上，发现模型是可以获取图像的高阶语义信息，是可以关注到用于分类的图像区域。
{{< /split >}}


### 2、BEiT
<a href="https://arxiv.org/abs/2106.08254" target="blank">《BEiT: BERT Pre-Training of Image Transformers》</a>(2021) 


### 3、PVT
<a href="https://arxiv.org/abs/2102.12122" target="blank">《Pyramid Vision Transformer》</a>(2021) 


### 4、Swin
<a href="https://arxiv.org/abs/2103.14030" target="blank">Swin Transformer</a>(2021) 微软亚研究院发表在ICCV上的一篇文章，获取2021 best paper。
<a href="https://github.com/microsoft/Swin-Transformer" target="blank"> github </a>

把Transformer从NLP应用到vision领域，有两个挑战：
1. 图中物体尺寸不一，比如一个行人、一辆骑车，在图中尺寸不同；就算是两个行人，也有大有小。NLP的同一个词，在图片中尺寸可能差别很大。
2. 图像分辨率太大，如果以像素点为基本单位的话，则序列就非常大。为了减少序列长度，一些方法：使用feature 

作者为了解决这两个问题，提出了Swin：
1. 通过`移动窗口`学习出 序列特征作为Transformer的输入；
2. 移动窗口(shifted window)能够使得相邻的两个窗口有了交互，变相的达到全局建模的能力。
3. 层级结构(hierarchical architecture)，非常灵活，不仅可以提供`不同尺度`的特征信息；而且计算复杂度跟图片大小成线性关系

---
{{< split 6 6>}}
<p align="center"><img src="/datasets/posts/cnn/swin_0.jpg" width="90%" height="90%" title="swin" alt="swin"></p>

1. ViT：每个token代表的尺寸都是一样的，每一层看到的token的尺寸都是一样的；虽说通过多注意力机制能够把握全局信息，但是在多尺寸特征上的把握是不够的。而在vision中多尺寸的的特征是很重要的。
2. ViT：多注意力机制是在全图上计算，所以它的计算复杂度跟图片尺寸成平方的关系；
3. Swin：在小窗口内计算多注意力，因为在视觉里有这样的先验：相邻的区域大概率是相似物体。对于视觉直接做全局注意力是浪费的
4. Swin：在卷积操作中，由于pooling操作提供了不同尺寸的特征；类比pooling，作者提出了patch merging：把相邻的小patch合并成一个大patch

---
<p align="center"><img src="/datasets/posts/cnn/swin_1.jpg" width="90%" height="90%" title="swin" alt="swin"></p>

怎么计算多注意力？滑动窗口的设计？<br>
图中灰框：是 $4\times4$ 的小块；rgb三通道拉直后就是 $1\times48$ <br>
图中红框：是一个包含 $7\times7$ 个小块的窗口，即：$7\times7\times48$。多注意力机制就是在这些窗口中计算的。<br>
滑动窗口：

<p align="center"><img src="/datasets/posts/cnn/swin_win.jpg" width="90%" height="90%" title="swin" alt="swin"></p>

1. 在窗口里做多头注意力计算，只能关注到这个窗口的信息；只这样操作，就违背了Transformer的初衷(把握上下文)，所以作者采用滑动窗口，下一层的窗口与上一层的多个窗口相交，这样多个窗口之间就有了联系。作者提出的patch merging 合并到Transformer最后几层时，就会看到大部分图片的信息了。
2. 为了统一的标准化计算，采用循环移位；
    - 向右下移动两个位置；上面多出的部分循环移动到下面、左边多出的部分循环移动到右边、左上角多出的部分循环移动到右下角
    - 在窗口内，循环移动的这些块之间 在图片中是不相邻的，所以不应该计算它们之间的联系(比如：上面是天空下面是地面)。作者就采用masked MSA
    - 在计算完注意力后，把移动的块复原到原来的位置；保证信息的一致性

{{< /split >}}

---
**网络架构**
<p align="center"><img src="/datasets/posts/cnn/swin_net.jpg" width="70%" height="70%" title="swin net" alt="swin net"></p>

1. 输入图片：$224\times224\times3$，通过patch partition操作，把 $4\times4$ 的小patch拉直后，变成：$56\times56\times48$
2. 通过4个Swin Transformer块，生成不同尺度的特征。
    1. 第一个Swin Tranformer块 没有使用patch merging，尺寸信息为 $56\times56\times96$，其中 $C=96$；剩余的模块，都经过patch merging：让H、W减半，在channel上扩大2倍。即：$56\times56\times96 \rArr 28\times28\times192 \rArr 14\times14\times384 \rArr 7\times7\times768$ (768怎么这么熟悉^_^)
    2. 每个Swin Transformer块，都含有两个Transformer块，第一个采用W-MSA（窗口-多头自注意力），第二个相匹配的采用SW-MSA（滑动窗口-多头自注意力）。

3. 不同变体：
    1. Swin-T：$C=96, layer numbers = {2, 2, 6, 2}$ ，计算复杂度与ResNet50差不多
    2. Swin-S：$C=96, layer numbers = {2, 2, 18, 2}$ ，计算复杂度与ResNet101差不多
    3. Swin-B：$C=128, layer numbers = {2, 2, 18, 2}$
    4. Swin-L：$C=192, layer numbers = {2, 2, 18, 2}$


{{< split 6 6>}}
**patch merging**

其中，patch merging 的操作如下图所示：
<p align="center"><img src="/datasets/posts/cnn/swin_merge.png" width="90%" height="90%" title="patch merging" alt="patch merging"></p>

1. 把相邻 $2*2$ 的patch块，拉伸到channel维度；使得H、W方向降维，C方向升维
2. 拉伸后变成 $4C$，如果希望是 $2C$的话，后续接一个全连接层

---
**masked MSA**
<p align="center"><img src="/datasets/posts/cnn/swin_mask-msa.png" width="90%" height="90%" title="masked MSA" alt="masked MSA"></p>

1. 只要不是自己区域的向量相乘，就需要被mask掉
2. 掩码矩阵：需要mask的区域为：-100，不需要mask掉的区域为：0。
3. softmax(计算好的自注意力矩阵（就是那个权重） + 掩码矩阵)，-100经过softmax后近似为0了。

{{< /split >}}

---

{{< split 6 6>}}

**W-MSA**
<p align="center"><img src="/datasets/posts/cnn/swin-wmsa.png" width="90%" height="90%" title="w-msa" alt="w-msa"></p>

1. 在每个窗口中做多头自注意力计算，各个窗口之间是没有联系的
2. 计算复杂度大概：$4hwC^2+2hwM^2C$，相比于全图片做自注意力计算($4hwC^2+2(hw)^2C$)，计算效率提升不少

---

**SW-MSA**
<p align="center"><img src="/datasets/posts/cnn/swin-sw-msa.png" width="90%" height="90%" title="sw-msa" alt="sw-msa"></p>

1. 如果只在窗口内各自计算注意力，那么就没有整个图片上下文；通过移动窗口来使得相邻的窗口之间有联系
2. 窗口的大小固定为 $7\times7$，由于patch merging类似于pooling操作，所以随着层数的加深，窗口看到的感受野越来越大

{{< /split >}}


### 5、MAE
<a href="https://arxiv.org/abs/2111.06377" target="blank">MAE</a>(2021)
主要思想：随机mask掉一些patch，然后重构这些patch里的所有像素。
  1. 设计非对称的encoder-decoder架构<br>
  encoder：作用在非mask的patch；将这些观察到的信息，映射到一个潜表示（在语义空间上的表示）<br>
  decoder：是一个轻量级的解码器，从这个潜表示中重构原始信号；
  2. 模型架构就是ViT，不一样的是输入是非mask的patchs
      - 高比例的mask是比较有效的，因为低比例的mask可以通过简单的插值就能重组，使得模型学不到什么东西；高比例的mask迫使模型学习更有效的表征；
      - 由于参与计算的是非mask的patch，高比例的mask使得计算速度加快好几倍。
  3. 图片patch与文本token的区别：图像只是被记录下来的光，没有语义分解成视觉上的词。后续的工作可以是：mask掉 不能构建语义段的patchs，就是这些patchs没有主体，只是包含主体的一小部分。
      - 由于patch不是一个word，不是独立的语义，可能跟其他patch构成一个独立的语义word，这就说明图片相对于文本冗余信息太多，即使mask掉好多信息也可以重构出图片
      - 模型学习到图片的全局信息，可以通过一些局部信息重构图片


<p align="center"><img src="/datasets/posts/cnn/mae.jpg" width="50%" height="50%" title="MAE" alt="MAE"></p>



---

### 6、
微软研究院提出的<a href="https://arxiv.org/abs/2107.00641" target="blank">Focal Transformer</a>在分类/检测/分割任务上表现SOTA！在ADE20K 语义分割上高达55.4 mIoU<br>

中科大、微软亚洲研究院提出的<a href="https://arxiv.org/abs/2107.00652" target="blank">CSWin</a>在ImageNet上高达87.5%准确率，在ADE20K上高达55.2mIoU
<a href="https://github.com/microsoft/CSWin-Transformer" target="blank">github</a> <br>


北大提出的<a href="https://arxiv.org/abs/2107.00420" target="blank">CBNetV2</a> 
<a href="https://github.com/VDIGPKU/CBNetV2" target="blank">github</a> <br>
<br>

## 三、CV各领域

### 1、目标检测

#### 1). DETR(Detection Transformer)
<a href="https://arxiv.org/abs/2005.12872" target="blank">《End-to-End Object Detection with Transformers》</a>(2020)，是Transformer在目标检测领域的开山之作。

### 2、图像分割



