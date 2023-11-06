---
title: "Bert综述"
date: 2021-09-08T06:00:20+08:00
menu:
  sidebar:
    name: Bert综述
    identifier: bert-summary-github
    parent: bert-github
    weight: 1
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["BERT"]
categories: ["Basic"]
---


## 一、背景
在使用预训练模型，处理下游任务时，有两类策略：基于特征(feature-based)、基于微调(fine-tuning)
  - 基于特征：比如：ELMo，在使用时，对每个下游任务，创建一个跟这个任务相关的神经网络；预训练作为额外的特征跟输入一起输入到模型，预训练的额外特征可能会对要训练的模型有指导作用。
  - 基于微调：比如：GPT，预训练模型在下游使用时，不需要改动太多，类似于视觉模型的fine-tuning，预训练完成特征提取，预训练模型后面添加个简单的网络用于实现具体任务。

### 1、上下文敏感
在自然语言中，有丰富的多义现象，一个词到底是什么意思，需要参考上下文才能判断。流行的上下文敏感表示：
  - <a href="https://arxiv.org/abs/1705.00108" target="blank">TagLM</a>(language-model-augmented sequence tagger 语言模型增强的序列标记器)
  - <a href="https://arxiv.org/abs/1708.00107" target="blank">CoVe</a>(Context Vectors 上下文向量)
  - <a href="https://arxiv.org/abs/1802.05365" target="blank">ELMo</a>(Embeddings from Language Models 来自语言模型的嵌入)
    1. ELMo 将来自预训练LSTM的所有中间层表示组合为输出表示
    2. ELMo的表示，将作为添加特征添加到下游任务的有监督模型中

### 2、从特定任务到通用任务
ELMo显著改进了自然语言任务，但每个解决方案仍然依赖于一个特定的任务架构。怎么设计一个模型，让各个自然语言任务通用呢？<br>
<a href="https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf" target="blank">GPT</a>(Generative Pre Training 生成式预训练)：在Transformer的基础上，为上下文敏感设计了通用的模型。

  1. 预训练一个用于表示文本序列的语言模型
  2. 当将GPT应用于下游任务时，语言模型的后面接一个线性输出层，以预测任务的标签。GPT的下游任务的监督学习过程，只对预训练Transformer解码器中的所有参数做微调。
  3. GPT只能从左到右


## 二、BERT
<a href="https://arxiv.org/abs/1810.04805" target="blank">BERT</a>的全称是Bidirectional Encoder Representation from Transformers, 即双向Transformer的Encoder。Bert结合了ELMo和GPT的有点，其主要贡献：

  1. 双向的重要性
  2. 基于微调的掩码语言模型(Masked Language Modeling)：BERT随机遮掩词元，并使用来自双向上下文的词元以自监督的方式预测该遮掩词元。

### 1、**构造输入**
token embedding: 格式：`<CLS>`第一个文本序列`<SEP>`第二个文本序列`<SEP>`<br>
segment embedding: 用来区分句子<br>
position embedding: 在bert中 位置嵌入 是可学习的<br>
```python
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```
<p align="center"><img src="https://s2.loli.net/2022/05/21/fzAiG9ZIhB3X1c4.jpg" width="70%" height="70%" title="input" alt="input"></p>


### 2、**MLM**
**词元维度**<br>
在预训练任务中，随机选择15%的词元作为预测的遮掩词元。
  - 80%的概率 替换为特殊词元 `<mask>` （填词）
  - 10%的概率 替换为 随机词元 （纠错）
  - 10%的概率 不做任何处理 （作弊）

> 为什么选择15%的mask量? <br>
> 
> 在这15%中，为啥有3种类型？ <br>
> 1. 由于 预训练过程中有mask，在fine-tuning、推理的时候是没有mask的。为了能保持一致，保持了80%是 token <mask> ，其余的是有对于的值是其他token。
> 2. 10%为错误的token，这个是为了保持模型的纠错能力
> 3. 10%为正确的token，为了避免让模型认为：给出的token都是错误的，所以一部分是正确的token

```python
class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        # batch_idx: batch * 序列大小
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        # masked_x的形状：（batch, 每个序列中被mask词的个数, 词元特征维度）
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        # 输出mlm_Y_hat形状：（batch, 每个序列中被mask词的个数, vocab_size）
        return mlm_Y_hat

```


### 3、预测下一句
**句子维度**<br>
尽管MLM能够使用上下文来表示词元，但它不能显式地建模文本对之间的逻辑关系，为了帮助理解两个文本序列之间的关系，BERT在预训练中考虑了一个二元分类：预测下一句。<br>
  - 在为预训练构建句子对儿时，50%的概率 句子对儿是连续句子；50%的概率 句子对儿不是连续句子。
```python
class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)
```


### 4、bert模型

位置编码，是可学习的。<font color=f00000>nn.Parameter()</font>


```python

class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens) 
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", 
                                 d2l.EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, 
                                                  ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768, nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size, query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens), 
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        # encoded_X 的形状：（批量大小，最大序列长度，num_hiddens）
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

## 三、各式各样的Bert

### 1、Bert的问题

> 1. 预训练 与 fine-tuning 之间会有差异。<mask> 在预训练里存在，在fine-tuning是不存在的
> 2. 预训练的效率是比较低的，训练数据中只预测了15%的量   <font color=#f00000>ELECTA：针对这个问题改进。输出是判断这个token是否被mask</font>
> 3. 上下文长度只有512

<a href="http://mitchgordon.me/machine/learning/2019/11/18/all-the-ways-to-compress-BERT.html" target="bland">《All The Ways You Can Compress BERT》</a>

## 总结

BERT虽然对上下文有很强的编码能力，但是缺乏细粒度语义的表示。比如：</br>
  - The sky is blue today.
  - The sea is blue today.
sky 和sea 明明是天和海的区别，却因为上下文一样而得到极为相似的编码。细粒度表示能力的缺失会对真实任务造成很大的影响。

## 参考



