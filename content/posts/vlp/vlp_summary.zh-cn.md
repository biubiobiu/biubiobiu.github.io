---
title: "简介"
date: 2022-05-09T06:00:20+06:00
menu:
  sidebar:
    name: 简介
    identifier: vlp-summary
    parent: vlp
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["vlp","summary"]
categories: ["Basic"]
math: true
---

## 一、简介

多模态学习，英文全程MultiModal Machine Learning(MMML)，从1970年 起步，已经经历了多个发展阶段，在2010年后，全面进入深度学习的阶段。多模态机器学习，以机器学习实现处理和理解多源模态信息的能力。图像、视频、音频、语义之间的多模态学习比较热门。比如互联网大型视频平台，都会将多模态技术用于视频理解业务，可以加视频封面、视频抽帧、文本信息融合。当计算机能够看懂视频，就可以做很多事儿了，比如：视频分类、审核、推荐、搜索、特效。<br>

多模态学习有5个研究方向：
1. 多模态表示学习（Multimodal Representation）
2. 模态转化（Translation）
3. 对齐（Alignment）
4. 多模态融合（Multimodal Fusion）
5. 协同学习（Co-learning）

实际应用，比如：
1. 视频网站上进度条，会显示那个时间段是高光时刻
2. 自动驾驶领域，雷达、视觉与多传感器信息融合
3. 视频的分类、审核、推荐、搜索、特效等等

### 1、VLP

微软发表的一篇文章<a href="https://arxiv.org/abs/2111.02387" target="blank">《An Empirical Study of Training End-to-End Vision-and-Language Transformers》</a>进行了大量的实验，对不同VLP模型、各个模块不同配置的效果。<br>
<p align="center"><img src="/datasets/posts/vlp/vlp_s.png" width="70%" height="70%" title="ViT" alt="ViT"></p>

VLP通常都会遵循同一个框架，包含5大模块：
1. **`Vision Encoder`**：主要有3中类型
    1. 使用object detection模型，比如：Faster R-CNN，识别图像中的目标区域，并生成每个目标区域的特征表示，输入到后续模型中
    2. 利用CNN模型提取grid feature作为图像输入
    3. ViT采用的将图像分解成patch，每个patch生成embeding输入到模型。
随着Vision Transformer的发展，ViT的方式逐渐成为主流方式。
2. **`Text Encoder`**：包括BERT、RoBERTa、ELECTRA、ALBERT、DeBERTa等经典预训练语言模型结构。
3. **`Multimodel Fusion`**：主要指如何融合图像、文本，主要有2中：
    1. co-attention：图像、文本分别使用Transformer编码，在每个Transformer模块中加入图像、文本的cross attention
    2. merged attention model，图像、文本在开始就拼接在一起，输入到Transformer
4. **`模型结构`**：主要有2中：
    1. Encoder-only：这种比较常见
    2. Encoder-Decoder
5. **`预训练任务`**：主要有3中：
    1. Masked Language Modeling（MLM）类似BERT，随机mask掉部分token，用剩余的预测出被mask掉的token
    2. Masked Image Modeling，对输入的部分图像patch进行mask，然后预测被mask的patchs
    3. Image-Text Matching（ITM），预测image和text的pair对是否匹配，对比学习的预训练方法可以属于这类。



## 二、网络

Open AI 在2021年1月份发布的<a href="https://openai.com/blog/dall-e/" target="blank">DALL-E</a>和<a href="https://openai.com/blog/clip/" target="blank">CLIP</a>，属于结合图像和文本的多模态模型，其中DALL-E是基于文本来生成模型的模型；CLIP是用文本作为监督信号来训练可迁移的视觉模型，这两个工作带动了一波新的研究高潮。<br>

### 1、CLIP
<a href="https://zhuanlan.zhihu.com/p/493489688" target="blank">参考</a>

<a href="https://arxiv.org/abs/2103.00020" target="blank">CLIP</a>

### 2、DALL-E
<a href="https://arxiv.org/abs/2103.00020" target="blank">DALL-E</a>：通过文本生成图片

### 3、KOSMOS-1

<a href="https://arxiv.org/abs/2302.14045" target="blank">《Language Is Not All You Need: Aligning Perception with Language Models》</a> 代码：<a href="https://github.com/microsoft/unilm" target="blank">github</a> 中介绍了一个多模态大型语言模型(MLLM)——KOSMOS-1。它可以感知一般模态、遵循指令(即零样本学习)以及在上下文中学习(即少样本学习)。研究目标：使感知与LLM保持一致，如此一来模型能够看到(see)和说话(talk)。研究者按照 <a href="https://arxiv.org/abs/2206.06336" target="blank">《Language models are general-purpose interfaces》</a> 的方式从头开始训练KOSMOS-1。



