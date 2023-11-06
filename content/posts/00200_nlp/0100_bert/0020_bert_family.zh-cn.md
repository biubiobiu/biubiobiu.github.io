---
title: "Bert家族"
date: 2021-09-08T06:00:20+08:00
menu:
  sidebar:
    name: Bert家族
    identifier: bert-family
    parent: bert-github
    weight: 20
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["BERT", "Family"]
categories: ["Basic"]
---

<p align="center"><img src="/datasets/posts/nlp/bert_family_0.png" width=100% height=100%></p>

## 一、简介


### 1、为什么需要预训练

<a href="https://arxiv.org/pdf/1908.05620.pdf" target="bland">《Visualizing and Understanding the Effectiveness of BERT》</a> <br>
<p align="center"><img src="/datasets/posts/nlp/why_pretrain_0.png" width=100% height=100%></p>

> 这篇文章指出:
> 1. 首先，预训练能在下游任务中<font color=#f00000>达到一个良好的初始点</font>，与从头开始训练相比，预训练能带来<font color=#f00000>更宽的最优点，更容易优化</font>。尽管 BERT 对下游任务的参数设置过高，但微调程序对过拟合具有很强的鲁棒性。
> 2. 其次，可视化结果表明，由于<font color=#f00000>最佳值平坦且宽广</font>，以及训练损失面和泛化误差面之间的一致性，微调 BERT 趋向于更好地泛化。
> 3. 第三，在微调过程中，BERT 的低层更具不变性，这表明靠近输入的层学习到了更多可迁移的语言表征。


### 2、下游任务怎么Fine-tune

我们希望有一个预训练的模型，输入一串单词，输出一串嵌入向量，并且希望这些向量是可以考虑上下文的。那么要怎么做呢？<br>

最早是由CoVe提出用翻译的方法，来得到可以考虑上下文的向量。那如何通过翻译的方法来得到这个预训练模型呢，就是把该模型当成翻译的编码器，输入一个A语言的序列，然后有一个解码器，结合编码器的注意力，得到B语言的输出。<br>
虽然可以做到这件事，但是翻译任务需要大量的语言对数据，收集这么多语言对数据是比较困难的，所以我们期望可以用很容易得到的<font color=#f00000>无标签文本</font>得到一个这样的预训练模型。<br>
过去这样的方法被叫做无监督学习，不过现在通常叫做自监督学习。在自监督学习中，模型学会用部分输入去预测另外一部分输入。换句话说，就是输入的一部分用于预测输入中的其他部分。这种预测下一个单词的方法就是我们训练语言模型的方式。那么要用什么样的网络结构来训练这个模型呢？<br>
最早用的就是LSTM，比较知名的使用LSTM进行预训练的模型，就是​ <a href="https://arxiv.org/pdf/1802.05365.pdf" target="bland">​ELMo​</a>​。随着自注意的流行，很多人把LSTM换成Transformer。


**问题**：
- [x] 为什么预训练下一个单词的方法，能让我们得到代表单词意思的嵌入向量呢？<br>
> 语言学家John Rupert Firth说过，你想要知道某个单词的意思，只要知道它和哪些单词一起出现。预测下一个单词其实做的是类似的事情。
- [x] 假设我们有一些特定任务的标签数据，那如何微调模型呢？<br>
> 1. 一种做法是预选练的模型训练好后就固定了，变成一个特征Extrator。输入一个单词序列，通过这个预训练模型抽取一大堆特征，把这些特征丢到特征任务模型中，然后进行微调；
> 2. 另外一种做法是把预训练的模型和特定任务的模型接在一起，在微调的时候，同时微调预训练模型和特定任务的模型。

- [x] 如果微调整个模型，会遇到什么问题呢? <br>
现在有三个不同的任务，每个任务中都有一个预训练好的模型，然后都微调整个模型。<br>
这三个预训练好的模型，在不同的任务微调里面，它们会变得不一样。每一个任务都需要存一个新的模型，包含微调的预训练模型和特定任务模型。这样的模型往往非常巨大，其中的参数非常多，导致需要占用特别多的空间。<br>


**怎么解决这个问题呢** <br>
有人提出 <a href="https://arxiv.org/pdf/1902.00751.pdf" target="bland">Adaptor</a> 的概念，在预训练的模型中加入一些叫Apt(Adaptor)的层，在微调的时候，只微调Apt层。这篇文章中，将Adapter插在Feed-forward层之后，在预训练的时候是没有Adapter的，只有在微调的时候才插进去。并且在微调的时候，只调整Adapter层的参数。
<p align="center"><img src="/datasets/posts/nlp/adaptor_0.png" width=70% height=70%></p>


## 二、bert家族

### 1、修改Mask范围

那在BERT里面，要盖住哪些单词呢，原始的BERT里面是随机的。也许随机的不够好，尤其对于中文来说，如果盖住中文中的某个字，还是很容易从它附近的字猜出，比如“奥x会”，只要看到“奥”和“会”就可以猜到中间是”运”了。所以
1. 有人提出 <a href="https://arxiv.org/pdf/1906.08101.pdf" target="bland">Whole Word Masking</a> ​​盖住整个单词(中文里的词语)的方法，这样得到的模型可以学到更长的依赖关系。
2. 可能只是盖住几个单词还不够好，<a href="https://arxiv.org/pdf/1904.09223.pdf" target="bland">ERNIE​(Baidu)</a> ​​就提出了盖住短语级别(多个单词组成一个短语)和实体级别(需要识别出实体，然后盖住)。
3. 还有一种Masking的方法，<a href="https://arxiv.org/pdf/1907.10529.pdf" target="bland">SpanBert</a>​​​，思想很简单，一次盖住一排单词(token)。不用考虑什么短语啊、单词啊、实体啊。在SpanBert里面还提出了一种训练方法，叫SBO(Span Boundary Objective)，一般我们盖住了一些单词后，我们要把盖住的部分预测出现。而SBO通过被盖住范围的左右两边的向量，然后给定一个数值，比如3，代表要还原被盖住的第3个单词。然后SBO就知道，现在要还原3个位置。
4. 还有一种方法，<a href="https://arxiv.org/pdf/1906.08237.pdf" target="bland">XLNet</a>，从输入的文本序列中，随机一部分，去预测mask的结果，就是让各种各样不同的信息去预测一个单词，模型可以学到比较多的依赖关系。具体
    * 在预训练阶段，引入permutation language model 的训练目标，对句子中单词排列组合，把一部分下文单词排列到上文位置中。这种做法是采用 attention掩码的机制来实现的：当前输入句子是X，要预测的第i个单词，i前面的单词位置不变，但是在transformer内部，通过attention mask，把其他没有被选到的单词mask掉，不让他们在预测单词的时候发生作用，看上去就是把这些被选中用来做预测的单词放在了上文位置了。

### 2、生成式任务
一般讲到BERT，大家都会说BERT不适于用来做生成任务，因为BERT训练的时候，会看到MASK左右两边的单词，而在生成任务中，只能看到左边已经生成出来的单词，然后BERT就表现不好了。<br>

但是这种讨论只局限在autoregressive模型，即由左到右生成单词的模型，这符合我们的写字方式。<br>
但是non-autoregressive模型，不需要一定要由左到右生成序列，也许这种情况下BERT就比较适用了。<br>

有人提出把输入序列做一点破坏，然后希望输出序列能复制出输入序列，通过这种方法来对seq2seq模型进行预训练。

那如何破坏输入序列呢，MASS和BART都探讨如何对输入进行破坏。
1. <a href="https://arxiv.org/pdf/1905.02450.pdf" target="bland">MASS</a>的想法和原来的BERT很像，通过把序列中一些单词随机盖住。MASS的目的只要还原出盖住的部分就可以了。
2. <a href="https://arxiv.org/pdf/1910.13461.pdf" target="bland">BART</a>提出了各种各样的破坏方法，比如：
    * 删掉某些单词(Delete)；
    * 打乱输入多个句子的顺序(permutation)； (❌: 效果不好)
    * 交换序列中单词的位置(rotation)； (❌: 效果不好)
    * 随机插入MASK(比如：原来AB单词之间没有其他单词，故意插入一个MASK去误导模型)或一个MASK盖多个单词(误导模型这里只有一个单词)(Text Infilling)。 (✅: 效果最好)

3. <a href="https://arxiv.org/pdf/1905.03197.pdf" target="bland">UniLM</a> ，它是一个神奇的模型，能同时充当编码器、解码器和seq2seq模型的角色。它是一个有很多个自注意的(这里是Transformer)模型，没有分编码器和解码器。然后让这个模型同时训练三种任务，训练的时候，整个输入分成两部分，第一个部分，像编码器一样，能看整个部分的单词；而第二个部分，只能看到输出的单词(解码器)。
    * 第一种：和BERT一样，把某些单词MASK起来；
    * 第二种：类似GPT的训练，即把它当成语言模型来用；
    * 第三种：就像BART和MASS一样，当成seq2seq来用。

### 3、YES/NO

<a href="https://arxiv.org/pdf/2003.10555.pdf" target="bland">ELECTRA</a> <br>
上面的预训练的方法，都需要预测一些部分。或者是预测下一个单词，或者是预测被盖住的部分。其实预测的模型需要的训练量是很大的，ELECTRA不做预测，只回答是或者否。<br>
比如: 上面原来的句子是“the chef cooked the meal”，现在把“cooked”换成了“ate”。ELECTRA需要判断输入的单词中，哪些被替换了。<br>
这样的好处是：预测Y/N简单；并且每个输出都被用到，可以计算损失。不像训练BERT时，只要mask的部分才计算loss。<br>
<p align="center"><img src="/datasets/posts/nlp/ELECTRA.png" width=70% height=70%></p>
ELECTRA的效果还比较不错，从上图可以看到，在同样的运算量下，它的表现比其他模型要好，并且能更快地达到较好的效果。<br>


如何输入一个句子，得到这个句子的向量呢？<br>
我们之前说过，如何了解一个单词的意思，要看这个单词与哪些单词相邻。如我们是不是可以看某个句子的相邻句子，来猜测这个句子的意思呢？Skip Thought就是基于这个想法，通过一个seq2seq模型，输入某个句子，来预测它的下一个句子。如果有两个不同的句子，它们的下一个句子很像，那么这两个句子就会有类似的嵌入向量。不过Skip Thought难以训练，有一个升级版——Quick Thought。这个模型的思想是，<font color=#f00000>有两个句子，分别通过编码器得到句向量，如果这两个句子是相邻的，那么就让这两个句向量越接近越好</font>。<br>

<a href="https://arxiv.org/pdf/1907.11692.pdf" target="bland">RoBERTa</a> <br>
<a href="https://arxiv.org/pdf/1909.11942.pdf" target="bland">ALBERT</a> <br>

如何判断两个句子是否相似？
1. 在原始bert的输入中，有判断下一句的逻辑。[CLS]句子1[SEP]句子2[SEP]，在输出层有二分类判断。这种方法叫：NSP（Next Sentence Prediction）
2. 还有一种方法叫SOP(Sentence order prediction)，输入两个相邻的句子，模型要输出YES；如果把两个句子反向，那么BERT要输出NO。ALBERT采用了这种思想。

### 4、知识图谱

在预训练的时候，加入外部知识，比如知识图谱。<a href="https://arxiv.org/pdf/1905.07129.pdf" target="bland">ERNIE(Tsinghua)</a> <br>


## RoBERTa

改进处RoBERTa是在论文《RoBERTa: A Robustly Optimized BERT Pretraining Approach》中被提出的。此方法属于BERT的强化版本，也是BERT模型更为精细的调优版本。<br>

在模型层面的改进：
1. 去掉下一句预测任务 NSP
2. 动态掩码：bert依赖随机掩码和预测token。原版bert在数据预处理期间执行一次掩码，得到一个静态掩码，而RoBERTa 使用了动态掩码：
    * 动态掩码：每次向模型输入一个序列时，会生成新的掩码模式。这样，在大量数据不打断输入的过程中，模型会逐渐适应不同的掩码策略，学习不同的语音表征。
    * 静态掩码：在准备数据时，每个样本会进行一次随机mask，因此每个epoch都是重复的。后续每个训练步都是采用相同的mask。
3. 文本掩码：Byte-Pair Encoding（BPE）是字符级和词级别表征的混合，支持处理自然语言语料库中的众多常见词汇。原版的 BERT 实现使用字符级别的 BPE 词汇，大小为 30K，是在利用启发式分词规则对输入进行预处理之后学得的。Facebook 研究者没有采用这种方式，而是考虑用更大的 byte 级别 BPE 词汇表来训练 BERT，这一词汇表包含 50K 的 subword 单元，且没有对输入作任何额外的预处理或分词。


在数据层面的优化：
1. 将16G的数据集提升到了160G
2. 采用bytes-leval的BPE后，词表从3万增加到5w。


## ALBERT

A Lite BERT(ALBERT) 的参数量只有BERT的70%，性能却能够显著超越BERT。ALBERT 采用两种参数精简技术来降低内存消耗，加快训练速度。
1. factorized embedding parameterization（词嵌入的因式分解）：<br>
对嵌入参数进行因式分解，将一个大的词汇嵌入矩阵分解为两个小矩阵，从而将隐藏层的大小与词汇嵌入的大小分离开来。这种分离便于后续隐藏层单元数量的增加，怎么说呢？就是增加隐藏层单元数量，并不显著增加词汇嵌入的参数量。
2. cross-layer parameter sharing（交叉层的参数共享）：<br>
这一技术可以避免参数量随着网络深度的增加而增加。
3. 放弃NSP（下一句预测），引入SOP（sentence order prediction 句子顺序预测），有利于学习句子间的连贯性

这两项技术显著降低了bert的参数量，同时不显著损坏其性能。


**SOP**：SOP关注于句子间的连贯性，而非句子间的匹配性。SOP正样本也是从原始语料中获得，负样本是原始语料的句子A和句子B交换顺序。

**跨层参数共享**<br>

