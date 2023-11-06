---
title: "Word Embedding综述"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: Word Embedding综述
    identifier: word_embedding-summary-github
    parent: word_embedding-github
    weight: 1
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["word embedding"]
categories: ["Basic"]
---

<p align="center"><img src="/datasets/posts/nlp/word_embeding.png" width=80% height=80%></p>

## 一、word embedding

词向量：是用来表示词的向量或者表征，也可被认为是词的特征向量。把词映射为实数域向量的技术 -- 词嵌入(word embedding)<br>
最简单的方式：one-hot向量。
1. 词库里假设有N个词，对所有词排序，用0~N-1作为每个词的索引。
2. 每个词的one-hot向量：长度为N，在该次索引的位置为1，其他位置都是0

缺点：one-hot向量，不能表征两个词的相似度。比如我们常用余弦相似度，one-hot的向量都是相互垂直的。


词嵌入是一种无监督学习。机器通过阅读大量的文章来学习的单词的意思，通过上下文信息来理解一个单词。怎么挖掘上下文信息：
1. Count-based method。认为如果两个单词一起出现的频率很高，那么它们的word embedding feature应该很接近彼此，二者的内积就越接近这两个的单词在同一篇文章中出现的次数。GloVe 就是一种count-based的算法。
2. prediction-based method. ski-gram 和 CBOW 就是这种算法。

## 二、Count-based method

### 1、GloVe

<a href="https://aclanthology.org/D14-1162.pdf" target="bland">《GloVe: Global Vectors for Word Representation》</a> <br>
上下文窗口内的词共现可以携带丰富的语义信息。例如，在一个大型语料库中，“固体”比“气体”更有可能与“冰”共现，但“气体”一词与“蒸汽”的共现频率可能比与“冰”的共现频率更高。此外，可以预先计算此类共现的全局语料库统计数据：这可以提高训练效率。<br>

GloVe模型基于平方损失 (Pennington et al., 2014)对跳元模型做了三个修改：
1. 使用变量 $p_{ij} = x_{ij}$ 和 $q_{ij} = e^{(u^T_j v_i)}$ 而非概率分布，并取两者的对数。所以平方损失项是 $(log p_{ij} - log q_{ij})^2 = (u^T_j v_i - log x_{ij})^2$
2. 为每个词 $w_i$ 添加两个标量模型参数：中心词偏置 $b_i$ 和上下文词偏置 $c_i$。
3. 用权重函数 $h(x_{ij})$ 替换每个损失项的权重，其中 $h(x)$ 在 $[0, 1]$ 的间隔内递增。

整合代码，训练GloVe是为了尽量降低以下损失函数：
$$
\sum_{i \in V} \sum_{j \in V} h(x_{ij})(u^T_j v_i + b_i + c_j - log x_{ij})^2
$$

## 三、prediction-based method

根据语言学家的理论：<font color=#f00000> 一个词的含义可以有其周围的词来决定。 </font>

word2vec工具，将每个词表示成一个定长的向量，并使得这些向量能较好地表达不同词之间的相似和类比关系。包含了两个模型，
1. <a href="https://arxiv.org/pdf/1310.4546.pdf" target="bland">跳字模型(skip-gram)</a>
2. <a href="https://arxiv.org/pdf/1301.3781.pdf" target="bland">连续词袋模型(continuous bag of word, CBOW)</a>

对于在语义上有意义的表示，它们的训练依赖于条件概率，条件概率可以被看作使用语料库中一些词来预测另一些单词。由于是不带标签的数据，因此跳元模型和连续词袋都是自监督模型。

### 1、跳字模型
跳元模型假设一个词可以用来在文本序列中生成其周围的单词。以文本序列 “the”“man”“loves”“his”“son” 为例。假设中心词选择 “loves”，并将上下文窗口设置为2，如图所示，给定中心词 “loves”，跳元模型考虑生成上下文词“the”“man”“him”“son”的条件概率：
$$
P(the, man, his, son | loves)
$$
假设上下文词是在给定中心词的情况下独立生成的（即条件独立性）。在这种情况下，上述条件概率可以重写为
$$
P(the | loves) · P(man | loves) · P(his | loves) · P(son | loves)
$$
<p align="center"><img src="/datasets/posts/nlp/slip-gram.png" width=50% height=50%></p>

在跳元模型中，每个词都有两个 $d$ 维向量表示，用于计算条件概率。更具体地说，对于词典中索引为 $i$ 的任何词，分别用 $v_i \in \R^d$ 和 $u_i \in \R^d$ 表示其用作中心词和上下文词时的两个向量。给定中心词 $w_c$ （词典中的索引 $c$），生成任何上下文词 $w_o$（词典中的索引 $o$）的条件概率可以通过对向量点积的softmax操作来建模：
$$
P(w_o|w_c) = \frac{e^{u^T_o v_c}}{\sum_{i \in V}e^{u^T_i v_c}}
$$

其中词表索引集 $V = {0, 2, ..., |V|-1}$。给定长度为 $T$ 的文本序列，其中时间步 $t$ 处的词表示为 $w^{(t)}$。假设上下文词是在给定任何中心词的情况下独立生成的。对于上下文窗口 $m$，跳元模型的似然函数是在给定任何中心词的情况下生成所有上下文词的概率：
$$
\prod^T_{t=1} \prod_{-m \le j \le m, j \ne 0} P(w^{(t+j)}|w^{(t)})
$$
其中可以省略小于1或大于 $T$的任何时间步。

### 2、连续词袋模型

连续词袋（CBOW）模型类似于跳元模型。与跳元模型的主要区别在于，连续词袋模型假设中心词是基于其在文本序列中的周围上下文词生成的。例如，在文本序列“the”“man”“loves”“his”“son”中，在“loves”为中心词且上下文窗口为2的情况下，连续词袋模型考虑基于上下文词“the”“man”“him”“son”（如图所示）生成中心词“loves”的条件概率，即：
<p align="center"><img src="/datasets/posts/nlp/cbow_0.png" width=50% height=50%></p>
由于连续词袋模型中存在多个上下文词，因此在计算条件概率时对这些上下文词向量进行平均。具体地说，对于字典中索引 $i$ 的任意词，分别用 $v_i \in \R^d$ 和 $u_i \in \R^d$ 表示用作上下文词和中心词的两个向量（符号与跳元模型中相反）。给定上下文词 $w_{o1}, ..., w_{o2m}$（在词表中索引是 $o_1, ..., o_{2m}$ ）生成任意中心词  $w_c$ （在词表中索引是  $c$）的条件概率可以由以下公式建模:
$$
P(w_c|w_{o1}, ..., w_{o2m}) = \frac{e^{\frac{1}{2m}u^T_c(v_{o1}+...+v_{o2m})}}{\sum_{i \in V}e^{\frac{1}{2m}u_i^T(v_{o1}+...+v_{o2m})}}
$$
给定长度为 $T$的文本序列，其中时间步 $t$ 处的词表示为 $w^{(t)}$。对于上下文窗口 $m$ ，连续词袋模型的似然函数是在给定其上下文词的情况下生成所有中心词的概率：
$$
\prod^T_{t=1} P(w^{(t)}|w^{(t-m)},...,w^{(t-1)},w^{(t+1)},...,w^{(t+m)})
$$


## 四、word2vec训练
不论是跳字模型还是连续词袋模型，由于条件概率使用了softmax运算，每一步的梯度计算都包含字典大小数目的项的累加。
对含几十万或者上百万词的较大词典来说，每次的梯度计算开销可能过大。为了降低计算复杂度，使用两种近似训练方法：
1. 负采样(negative sampling)
2. 层序(hierarchical softmax)



## 五、子词嵌入/fastText


### 1、子词嵌入（subword）
问题：通常w2v模型使用向量表示词信息，忽略了词构造(词性)特点，词词之间的信息没有被联系起来。

编码方式介绍：
> 1. 传统方式：<br>
> 用空格作为划分符，不利于模型学习词缀信息，不能处理 OOV（罕见词、训练集之外的词）问题。
> 2. 字节对编码(BPE)：<br>
> 一种数据压缩方式，词库大小可控，对单词拆分。减少词数，可以加快训练。缺点：基于贪心算法和确定串的匹配，不能提供概率的多个分片结果。
> 3. wordpiece：<br>
> BPE的变种，过程类似BPE，但是基于概率确定subword，而不是基于下一个最高频次字节对。在bert模型预处理中被使用。
> 4. n-gram子词编码：<br>
> 和wordpiece一样，利用模型语音建立subword

<a href="https://cloud.tencent.com/developer/article/1593466?areaSource=102001.3&traceId=P9zvRUxQMHvv5MPUUhjUB" target="bland">对比：BPE、wordpiece、</a> <br>

#### a、BPE(Byte Pair Encoding)

<a href="https://zhuanlan.zhihu.com/p/424631681" target="bland">参考1</a> <br>


1. 优点 <br>
可以有效地平衡词汇表大小和对未知词的覆盖。
2. 缺点 <br>
基于贪婪和确定的符号替换，不能提供带概率的多个分片结果。

#### b、wordpiece
WordPiece算法可以看作是BPE的变种。不同点在于，WordPiece基于概率生成新的subword而不是下一最高频字节对。

算法:
1. 准备足够大的训练语料
2. 确定期望的subword词表大小
3. 将单词拆分成字符序列
4. 基于第3步数据训练语言模型
5. 从所有可能的subword单元中选择加入语言模型后能最大程度地增加训练数据概率的单元作为新的单元
6. 重复第5步直到达到第2步设定的subword词表大小或概率增量低于某一阈值

#### c、ULM
ULM是另外一种subword分隔算法，它能够输出带概率的多个子词分段。它引入了一个假设：所有subword的出现都是独立的，并且subword序列由subword出现概率的乘积产生。WordPiece和ULM都利用语言模型建立subword词表。


ULM(unigram Language Model) 算法：
1. 准备足够大的训练语料
2. 确定期望的subword词表大小
3. 给定词序列优化下一个词出现的概率
4. 计算每个subword的损失
5. 基于损失对subword排序并保留前X%。为了避免OOV，建议保留字符级的单元
6. 重复第3至第5步直到达到第2步设定的subword词表大小或第5步的结果不再变化


### 2、FastText
FastText 是2013年Facebook开源的计算词向量及高效的文本分类工具。<br>

问题：在上述模型中，将形状不同的单词用不同的向量来表示。例如：dog和dogs分别用不同的向量表示，模型中没有直接表示这两个向量的关系。鉴于此，fastText提出了子词嵌入(subword embedding)的方法，从而试图将构词信息引入Word2vec中的跳字模型。

改进：
1. 在fastText中，每个中心词：表示成子词的集合。 例如 用where来构建子词：
    * 在单词的首尾添加特殊字符<>,以区分作为前后缀的子词
    * 将单词当成一个由字符构成的序列来提取n元语法。例如n=3，得到所有长度为3的子词：`<wh  whe  her  ere  re> `
2. 词典：所有词的子集的并集
3. 模型中词w作为中心词的向量 $v_w$  则表示成 $v_w = \sum_{g \in G_w} z_g$



## 六、git工程

参考：
1. <a href="https://www.jianshu.com/p/d4de091d1367" target="bland">BPE、WordPiece和SentencePiece</a>
2. <a href="https://zhuanlan.zhihu.com/p/86965595" target="bland">深入理解NLP Subword算法：BPE、WordPiece、ULM</a>

如果自己想添加自己的词表：可以用 <a href="https://github.com/taishan1994/sentencepiece_chinese_bpe" target="bland">使用sentencepiece中BPE训练中文词表</a> <br>
然后，<a href="https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_tokenizer/merge_tokenizers.py" target="bland">再添加到词表中</a>
