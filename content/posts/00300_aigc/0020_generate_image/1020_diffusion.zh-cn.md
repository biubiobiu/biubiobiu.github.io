---
title: "Diffusion"
date: 2023-08-05T12:30:40+08:00
menu:
  sidebar:
    name: Diffusion
    identifier: aigc-image-diffusion
    parent: aigc-image
    weight: 1020
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["AIGC","Diffusion"]
categories: ["Basic"]
math: true
---

## 一、简介

Diffusion 过程：每一步添加一次noise，经过很多步后，图片就接近白噪声了。<br>
<p align="center"><img src="/datasets/posts/vlp/diffusion_process.png" width=100% height=100%></p>

但是，我们的目的是：从白噪声中，根据输入的文字描述，生成一张图片。是上面Diffusion过程的逆过程。<br>
<p align="center"><img src="/datasets/posts/vlp/diffusion_generate_2.png" width=100% height=100%></p>

所以，根据Diffusion过程生成训练数据，然后训练一个：<font color=#f00000>noise 生成器</font>

{{< split 6 6>}}
<p align="center"><img src="/datasets/posts/vlp/diffusion_generate_3.png" width=100% height=100%></p>

---
从noise中生成图片的过程，也是一步一步删除noise的过程。<br>
所以，训练一个<font color=#f00000>noise 生成器</font>，根据输入的图片、步数，生成图片中的噪声，然后减去输入图片中的这部分噪声，就是去除噪声的图片。<br>
为啥设计个 <font color=#f00000>noise 生成器</font>，而不是直接设计个图片生成器？<br>
<font color=#f00000>noise 生成器</font>：生成噪声，总比一步生成最终的优质图片<font color=#a020f0>要容易得多</font>。

{{< /split >}}

需要根据输入的文字描述，来生成相应的图片，那么文字特征是怎么输入的呢？

{{< split 6 6>}}
<p align="center"><img src="/datasets/posts/vlp/diffusion_generate_4.png" width=100% height=100%></p>
---
在 <font color=#f00000>noise 生成器</font> 中添加一个文本输入。

{{< /split >}}




## 二、Diffusion Model

{{< split 6 6>}}
<p align="center"><img src="/datasets/posts/vlp/frame_difusion.png" width=100% height=100%></p>

---

Diffusion 的常用范式：
1. 文本编码器
2. 生成模型：产出中间产物（图片的压缩版本）
3. 图像解码器：生成高质量的图片

{{< /split >}}

**文本编码器**：<br>
可以是 <a href="https://arxiv.org/pdf/2103.00020.pdf" target="bland">《CLIP》</a> <br>


**生成模型**：<br>

在对图片做Diffusion时，是对图片不断地添加noise。<br>

对于生成模型，是对latent representation 不断地添加noise。<br>

<p align="center"><img src="/datasets/posts/vlp/diffusion_generate_0.png" width=100% height=100%></p>

所以，在真实的生成模块里，是对latent representation 不断地删除噪声，生成可用的中间产物。
<p align="center"><img src="/datasets/posts/vlp/diffusion_generate_1.png" width=100% height=100%></p>


**图像编码器**：<br>

<!--
{{< split 6 6>}}

---

{{< /split >}}
-->
图像编码器，可以是把低分辨率的小图，生成高分辨率的大图，如下图：
<p align="center"><img src="/datasets/posts/vlp/decoder-0.png" width=100% height=100%></p>

也可以是，把中间产物，生成高分辨率的大图，如下图：
<p align="center"><img src="/datasets/posts/vlp/decoder-1.png" width=100% height=100%></p>



### 1、Stable Diffusion

<a href="https://arxiv.org/pdf/2112.10752.pdf" target="bland">《Stable Diffusion》</a> <br>

<p align="center"><img src="/datasets/posts/vlp/stable_diffusion_0.png" width=100% height=100%></p>


### 2、DALL-E 


<a href="https://arxiv.org/pdf/2204.06125.pdf" target="bland">《Hierarchical Text-Conditional Image Generation with CLIP Latents》</a> <br>

<a href="https://arxiv.org/pdf/2102.12092.pdf" target="bland">《Zero-Shot Text-to-Image Generation》</a> <br>

<p align="center"><img src="/datasets/posts/vlp/dall-e-0.png" width=100% height=100%></p>


### 3、Imagen

<a href="https://imagen.research.google/" target="bland">Imagen Home</a> <br>

<a href="https://arxiv.org/pdf/2205.11487.pdf" target="bland">《Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding》</a> <br>

<p align="center"><img src="/datasets/posts/vlp/imagen-0.png" width=100% height=100%></p>


## 三、评估指标

### a. FID
<a href="https://arxiv.org/pdf/1706.08500.pdf" target="bland">《FID》</a> <br>


## 四、总结

