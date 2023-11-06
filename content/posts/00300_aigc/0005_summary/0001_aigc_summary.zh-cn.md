---
title: "综述"
date: 2023-08-05T12:30:40+08:00
menu:
  sidebar:
    name: 综述
    identifier: aigc-summary-text
    parent: aigc-summary
    weight: 1
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["aigc","summary"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

<p align="center"><img src="/datasets/posts/nlp/evolutionary_tree.png" width="100%" height="100%"></p>

AIGC 的技术分类按照处理的模态来看，可以分为一下几类：
1. 文本类：<br>
主要包括：文章生成、文本风格转换、问答对话等生成或者编辑文本内容的AIGC技术。
2. 音频类：<br>
包括：文本转音频、语音转换、语音属性编辑等生成或者编辑语音内容的AIGC技术；以及音乐合成、场景声音编辑等生成或者编辑非语言内容的AIGC技术。例如：智能配音主播、虚拟歌手演唱、自动配乐、歌曲生成等
3. 图像视频类：<br>
包括：人脸生成、人脸替换、人物属性编辑、人类操控、姿态操控等AIGC技术；以及编辑图像、视频内容、图像生成、图像增强、图像修复等AIGC技术
4. 虚拟空间类：<br>
主要包括：三维重建、数字仿真等AIGC技术，以及编辑数字任务、虚拟场景相关的AIGC技术，例如：元宇宙、数字孪生、游戏引擎、3D建模、VR等。

在大语言模型的训练中，如果增大数据量，相应的应该减少学习率，这个跟原来的经验相反。<br>

## 模型大小与模型效果

<a href="https://arxiv.org/pdf/2206.07682.pdf" target="bland">《Emergent Abilities of Large Language Models》</a> <br>
这篇文章指出：随着模型大小的增大，模型效果先不会有明显提升；增加到一定程度，模型有个突然顿悟时刻。

<p align="center"><img src="/datasets/posts/nlp/ablity_llm.png" width="90%" height="90%"></p>

## 为什么需要预训练

<a href="https://arxiv.org/pdf/1908.05620.pdf" target="bland">《Visualizing and Understanding the Effectiveness of BERT》</a> <br>
<p align="center"><img src="/datasets/posts/nlp/why_pretrain_0.png" width=100% height=100%></p>

> 这篇文章指出:
> 1. 首先，预训练能在下游任务中<font color=#f00000>达到一个良好的初始点</font>，与从头开始训练相比，预训练能带来<font color=#f00000>更宽的最优点，更容易优化</font>。尽管 BERT 对下游任务的参数设置过高，但微调程序对过拟合具有很强的鲁棒性。
> 2. 其次，可视化结果表明，由于<font color=#f00000>最佳值平坦且宽广</font>，以及训练损失面和泛化误差面之间的一致性，微调 BERT 趋向于更好地泛化。
> 3. 第三，在微调过程中，BERT 的低层更具不变性，这表明靠近输入的层学习到了更多可迁移的语言表征。


## 一、文本生成


### 1、GPT

<a href="/zh-cn/posts/00200_nlp/0080_gpt/0010_gpt_summary">参考</a>

**GPT-4**: 参数量1800B，训练集：1.3T token <br>


### 2、PaLM
<a href="https://arxiv.org/pdf/2204.02311.pdf" target="bland">《PaLM: Scaling Language Modeling with Pathways》</a> <br>

PaLM才是真正的“大”模型。它是迄今为止训练的最大的密集语言模型，参数为 540B，需要 6144 个 TPU 来训练（这是 3 个完整的 TPU pod，每个包含 2048 个 TPU）。这太贵了！可能只有谷歌拥有资源+基础设施来做到这一点。使用的Token高达7800亿。PaLM是使用Google新一代PathWay分布式训练框架训练出来。<br>

与GPT-3相比的变化：
> 1. 多查询注意力（Multi-query attention）：在每个注意力头中共享K/V（Key/Value）嵌入，但使用独立的Q（Query）嵌入。这样做可以在推理阶段显著提高模型的速度。
> 2. 并行Transformer块：使用并行的Transformer块来提高训练时间，相较于传统的串行设计，可以减少约15%的训练时间。
> 3. SwiGLU激活函数：与GPT-3使用的GELU激活函数不同，这里采用了SwiGLU激活函数。
> 4. 旋转位置编码RoPE嵌入：使用RoPE（Relative Positional Encodings）嵌入代替学习得到的嵌入方式，在长文本上具有更好的性能 。
> 5. 输入-输出嵌入共享：输入和输出embedding矩阵是共享的。
> 6. 无偏置向量：在mlp、normlayer等算法中，都不使用bias，对于大模型，可以提高训练稳定性。
> 7. SentencePiece与256k标记：使用SentencePiece进行分词处理，标记数量为256k。

所以，有很多变化！同样，其中很多都是常见的，例如使用 GPT-3 的学习嵌入向量已经非常过时了，现在几乎没有人这样做。

<p align="center"><img src="/datasets/posts/nlp/palm_0.png" width="90%" height="90%"></p>
<p align="center"><img src="/datasets/posts/nlp/palm_1.png" width="90%" height="90%"></p>



### 3、ChatGLM
**数据**：
1. 经过了 1.4T 中英文，1:1 比例
2. 130528 词表大小

**模型**
1. Transformer的整体
2. 编码器，采用span-mask，调整span的个数、长度，同时满足NLU和NLG
3. 2D的位置编码，旋转位置编码RoPE
4. Post-DeepNorm
5. 激活函数：GeGLU
6. 上下文长度：32k，基于FlashAttention技术

Layer Normalization的顺序和残差连接被重新排列，
用于输出标记预测的单个线性层；
ReLU s替换为GELU s
二维位置编码

### 4、BLOOM

<a href="https://arxiv.org/pdf/2211.05100.pdf" target="bland">《BLOOM: A 176B-Parameter Open-Access Multilingual
Language Model》</a> <br>

**数据**
1. 使用了 25 万个token的词汇表。 使用字节级 BPE。 这样，标记化永远不会产生未知标记
2. BLOOM是在一个称为ROOTS的语料上训练的，其是一个由498个Hugging Face数据集组成的语料。共计1.61TB的文本，包含46种自然语言和13种编程语言。

**模型**
1. 使用Transformer的decoder
2. 使用 ALiBi 位置嵌入，它根据键和查询的距离直接衰减注意力分数，能够外推至更长的序列。
3. 在BLOOM的第一个embedding层后添加了额外的layer normalization层来避免训练不稳定性

**训练**
1. 模型在Jean Zay上训练，其是由法国政府资助的超级计算机。训练BLOOM花费了3.5个月才完成，并消耗了1082990计算小时。在48个节点上进行训练，每个有8 NVIDIA A100 80GB GPUs(总共384个GPUs)
2. BLOOM使用Megatron-DeepSpeed训练，一个用于大规模分布式训练的框架。其由两部分组成：
    * Megatron-LM提供Transformer实现、张量并行和数据加载原语，
    * DeepSpeed提供ZeRO优化器、模型流水线、通过分布式训练组件。
3. 混合精度训练

**模型版本**
|模型|Layers|Hidden dim|vocab|句长|batch|
|:--|:--|:--|:--|:--|:--|
|BLOOM-560M|24|1024|||256|
|BLOOM-1.1B|24|1536|||256|
|BLOOM-1.7B|24|2048|||512|
|BLOOM-3B|30|2560|||512|
|BLOOM-7.1B|30|4096|||512|
|BLOOM-176B|70|14336|250680|2048|2048|


### 5、LLaMa
<a href="https://arxiv.org/pdf/2307.09288.pdf" target="bland">《Llama 2: Open Foundation and Fine-Tuned Chat Models》</a> <br>
<a href="https://github.com/facebookresearch/llama" target="bland">Github</a> <br>

<p align="center"><img src="/datasets/posts/nlp/llama_scale.png" width="90%" height="90%"></p>


**数据**<br>
1. 2T 的数据，英文为主
2. 32k 的词表

**模型**<br>
LLaMa结合了PaLM和Chinchilla两个模型的最佳特点，并做出了一些改进：

> 1. 预归一化（Pre-normalize）：在每个Transformer子层之前对输入进行预归一化。
> 2. 使用RMSNorm：使用RMSNorm代替LayerNorm，与Gopher模型中一样。
> 3. SwiGLU激活函数：使用了PaLM中的SwiGLU激活函数，但是维度从PaLM的值改为了新的值。
> 4. 旋转位置嵌入（Rotary positional embeddings）：采用RoPE（相对位置编码）替代了PaLM中的绝对位置嵌入法。
> 5. 使用AdamW：与Chinchilla模型一样，使用AdamW优化算法。
> 6. 上下文长度：4096

在计算方面的变化有：
> 1. 使用高效的注意力机制（Rabe & Staats, FlashAttention）。
> 2. 梯度检查点（Gradient checkpointing）。

作者唯一的抱怨是他希望他们能够将模型训练更长时间，因为学习曲线与收敛相差甚远！<br>


<font color=#f00000>基于LLaMa的衍生模型</font>：
|模型|介绍|
|:--|:--|
|<a href="https://github.com/tatsu-lab/stanford_alpaca" target="bland">Alpaca</a>|斯坦福大学在52k条英文指令遵循数据集上微调了7B规模的LLaMA。|
|<a href="https://github.com/lm-sys/FastChat" target="bland">Vicuna</a>|加州大学伯克利分校在ShareGPT收集的用户共享对话数据上，微调了13B规模的LLaMA。|
|<a href="https://github.com/project-baize/baize-chatbot" target="bland">Baize</a>|在100k条ChatGPT产生的数据上，对LLaMA通过LoRA微调得到的模型。|
|<a href="https://github.com/Stability-AI/StableLM" target="bland">StableLM</a>|Stability AI在LLaMA基础上微调得到的模型。|
|<a href="https://github.com/LianjiaTech/BELLE" target="bland">BELLE</a>|链家仅使用由ChatGPT生产的数据，对LLaMA进行了指令微调，并针对中文进行了优化。|


### 6、Claude

<a href="https://claude.ai/chat/" target="bland">Claude Chat API</a> <br>

特点：
1. 输入序列长度可达：100k
2. 

### 7、Cohere

### 8、Falcon


### 9、Vicuna


### 10、Guanaco


### 11、MPT


### 12、Lazarus


### 13、WizardLM



## 二、图像生成

### 1、GAN
2014年

### 2、CAN
2017年

### 3、DALL-E
2021年2月<br>
根据文本描述绘画，绘画水平一般。

### 4、CLIP+VQGAN
2021年4月<br>
根据文本描述绘画，绘画水平一般。

### 5、Disco Diffusion
2022年2月<br>
根据文本描述绘画，具有原创性，图片精美，渲染时间长。

### 6、Midjourney
2022年3月<br>
根据文本描述绘画，适合人像，细节突出

### 7、DALL-E2
2022年4月，OpenAI发布DALL-E 2，命名来源于著名画家Dali和机器人总动员Wall-E，是DALL-E的升级版，其分辨率是之前版本的4倍。<br>

DALL-E 2 由三个模型组成：<font color=a00000>CLIP模型、先验模型、扩散模型</font>。
1. CLIP模型主要是用来对齐文本和图像特征：获取文本编码
2. 先验模型主要是将文本表征映射为图片表征：将文本编码映射为图片编码
3. 扩散模型是根据图片表征来完成完整的图像：用图片编码生成完整的图片。

根据文本描述绘画，限制较多，对复杂文字理解准确，渲染快

### 8、Stable Diffusion
2022年8月，慕尼黑大学的Robin Rombach和Patrick Esser的团队提出的文本生成图像模型，交互简单，生成速度快。Stable Diffusion主要由三部分组成，分别是 <font color=#a00000>VAE、U-Net、CLIP文本编码器</font>：
1. 首先使用CLIP模型将文本转换为表征形式
2. 然后引导扩散模型U-Net在低维表征上进行扩散
3. 最后将扩散后的低维表征送入VAE中的解码器，从而生成图像。

在GAN和CLIP的基础上，Stable Diffusion模型开源，直接推动了AIGC技术的突破性发展。<br>
Stable Diffusion 扩散模型的原理是：先添加噪声后降噪。即：给现有的图像逐步添加噪声，直到图像被完全破坏，然后根据给定的高斯噪声，逆向逐步还原出原图。在模型训练完毕后，只需要输入一段随机的高斯噪声，就能生成一张图像。<br>
根据文本描述绘画，具有原创性，灵活度高，图片精美，具有真实感，渲染快。<br>



### 9、Imagen
2022年11月<br>
优先开源，效果好于DALL-E


## 三、国内

### 1、太极

腾讯基于自身在自然语言处理和图像多模态等方面积累的经验，打造了通用场景模型——<font color=#f00000>太极文生图大模型</font>。太极文生图采用了Diffusion路线

### 2、文心一格
百度提出的AIGC大模型——<font color=#f00000>ERNIE-ViLG 文生图模型</font>，包括：工业设计、游戏制作、服装设计、Logo设计、盆栽设计、动漫设计、珠宝设计、传统艺术等领域。ERNIE-ViLG模型能够深刻地理解中文语境，更了解中国化。

### 3、太乙
IDEA研究院开源的第一个中文版Stable Diffusion模型——<font color=#f00000>太乙 Stable Diffusion</font>，该模型基于0.2亿筛选过的中文图文对进行训练，从而实现了具备中文内核的AIGC模型。

### 4、CogView

智源研究院于2022年上半年，推出的CogView 2.0和 CogVideo

### 5、MSRA

2021年11月微软亚洲研究院与北京大学联合发布了女娲模型，女娲模型用来从输入的文本、图像、视频中生成图像或者视频。

### 6、MagicMix

字节跳动公司发布了MagicMix模型，模型可以将任意两个语义进行组合，生成全新的概念，再基于新概念进行图像生成。

### 7、DPM-Solver

清华大学的朱军教授团队提出的DPM-Solver，是一种针对扩散模型特殊设计的高效求解器。
