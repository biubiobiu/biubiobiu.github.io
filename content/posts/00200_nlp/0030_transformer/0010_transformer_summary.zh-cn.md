---
title: "Transformer"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: Transformer
    identifier: transformer-summary-github
    parent: transformer-github
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["NLP", "Transformer"]
categories: ["Basic"]
---


## 一、简介

谷歌大脑、谷歌研究院等团队于2017年联合发表文章<a href="https://arxiv.org/abs/1706.03762" target="blank">《Attention Is All You Need》</a>，提出了一种新的注意力 Seq2Deq 模型，以取代之前以RNN作为编/解码器实现的 Seq2Seq 模型。模型采用的也是编码器-解码器架构，但是在该模型中，编码器和解码器不再是 RNN结构，取而代之的是编码器栈（encoder stack）和解码器栈（decoder stack）（注：所谓的“栈”就是将同一结构重复多次，“stack”翻译为“堆叠”更为合适）。编码器栈和解码器栈中分别为连续N个具有相同结构的编码器和解码器。

> `编码器：由两部分组成（自注意力模块 + 前馈神经网络）`<br>
> 自注意力模块：具体来说是“Multi-Head Attention”，即“多头注意力”模块<br>
> 全连接前馈网络 <br>
> 每个子网络都具有残差连接，其输出形式为 $LayerNorm(Sublayer(x)+x)$ ，其中 $Sublayer(x)$ 表示子网络对输入特征x进行的具体映射操作；$LayerNorm()$ 表示归一化操作。

> `解码器：由三部分组成（自注意力模块 + 编码-解码注意力模块 + 前馈神经网络）`<br>
> 解码器中多了一个编码-解码注意力模块，用来利用当前已有的输出，来匹配输入特征（即：attention操作），然后拿计算出的新特征来计算当前时间步的输出。解码器中的自注意力模块与编码器不同是：这里只能看到当前时间步之前的输入，而不是全部的输入，所以需要有mask的操作。

论文中图：
{{< split 6 6 >}}
<p align="center"><img src="/datasets/posts/nlp/attention.png" width="80%" height="80%" title="attention" alt="attention"></p>
---
<p align="center"><img src="https://s2.loli.net/2022/05/19/9nCzrTwESlBRfmy.jpg" width="100%" height="100%" title="attention" alt="attention"></p>
{{< /split >}}


## 二、Transformer
输入：序列的embeding表示 + 位置编码

编码器：
  1. 多头注意力 + 残差连接(residual connection) --> 层归一化(layer normalization)
  2. 基于位置的前馈网络(positionwise feed-forward network) + 残差连接(residual connection) --> 层归一化(layer normalization)

```python
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

解码器：
  1. 解码器自注意力：解码器中每个位置只能考虑该位置之前的所有位置，所以添加`掩码`
  2. 编码器-解码器注意力：query：前一个解码器层的输出；k和v：整个编码器的输出。目的是捕获与当前解码最相关的编码状态，而不是所有的编码状态都是同等重要。
  3. 基于位置的前馈网络(positionwise feed-forward network) + 残差连接(residual connection) --> 层归一化(layer normalization)

### 1、位置编码

与CNN/RNN不同，自注意力是`没有记录位置信息的`。可以回顾自注意力的计算过程，在$q_j$ 跟 $\bold{K}$、$\bold{V}$ 计算后，生成一个把$\bold{V}$各个向量加权后的向量，这里面是没有位置信息的，也就是说不管输入的 $\bold{V}$ 的向量顺序如何变化，自注意力的输出是不会改变的。<br>
所以：`需要位置编码将位置信息注入到输入里。`<br>

输入：$\bold{X} \in \R^{n \times d}$ 包含一个序列中n个词元的d维嵌入表示。<br>
位置编码：$\bold{P} \in \R^{n \times d}$, 矩阵第i行 偶数列、奇数列：用不同的频率、偏移来记录位置信息。
$$p_{i,2j} = sin(\frac{i}{10000^{\frac{2j}{d}}})$$
$$p_{i,2j+1} = cos(\frac{i}{10000^{\frac{2j}{d}}})$$

在 $\bold{X} + \bold{P}$ 时，当$\bold{X}$的幅度值比$\bold{P}$小或者差不多时，可以增大$\bold{X}$的幅度值，以保证$\bold{X}$的主导性。
$$
\bold{X} \times M + \bold{P}
$$

<p align="center"><img src="/datasets/posts/nlp/position_em.png" width="50%" height="50%" title="position" alt="position"></p>

### 2、层归一化

**层归一化**：<a href="https://arxiv.org/abs/1607.06450" target="blank">《Layer Normalization》</a>，在一个输入序列中，做归一化。
  - 由于输入序列的长度是不确定的

**批归一化**：<a href="https://arxiv.org/abs/1502.03167" target="blank">《Batch normalization》</a>，在一个batch中，在通道维度 做归一化。
  - 避免梯度消失/爆炸：这是因为通过归一化(偏移、拉伸)，把原来可能波动较大的数据，限制在一定的范围内
  - 为啥有效：有的解释是：通过(偏移、拉伸)，相当于添加了一个随机噪声，因为 均值、方差是在当前小批量样本上算出来的，包含了随机性。
  - 批归一化，限制波动的范围，所以可以调大学习率，可以加速收敛。

```python
# 批归一化
mean = X.mean(dim=(0, 2, 3), keepdim=True)
var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
X_hat = (X - mean) / torch.sqrt(var + eps)
```

{{< alert type="info" >}}
**图像与文本**<br>
  - 图像: (batch, c, h, w)  ： (h, w: 图像的高和宽)、(c: 通道数)
  - 文本: (batch, T, d) ：(T：序列长度)、(d: 每个词元embeding表示的维度)

类比一下：
  - 每个图像中包含 $h \times w$ 个像素点，这个数目 类似 文本的长度。
  - 每个像素点在channel方向是一个向量，这个向量 类似 词元的embeding

也就是说：图像的每个像素点表示一个样本点，channel方向 表示该样本点的特征表示。这也就是为什么说 $1 \times 1$ 的卷积核的作用相当于全连接层。

{{< /alert >}}

```python
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

<p align="center"><img src="https://s2.loli.net/2022/05/19/PGuYO7FWjdUTIeE.jpg" width="50%" height="50%" title="layer NM" alt="layer NM"></p>


### 3、基于位置的前馈网络

输入：(batch, 序列长度, embeding维度)<br>
输出：(batch, 序列长度, 新特征维度)<br>
作用：类似于卷积中的 $1 \times 1$ 卷积核，就是转换一下特征的维度，样本个数不变。
```python
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

### 4、编码器中的attention
在编码器中使用VVV模式，该自注意力模块为 MHA 结构，其中`V`为上一个编码器对输入句子中的每个词的编码（这里的编码可以理解为 RNN 中的隐变量向量，即输入句子中每个词的内部表示。如果是第一个编码器这里的编码即每个词的嵌入向量）。编码器自注意力模块用来捕捉输入句子中词与词之间的关系。例如翻译句子“The dog is barking at the bird because it is angry”，这里的“it”到底说的是狗还是鸟？编码器自注意力模块就是为了在对“it”进行编码时，尽量使得 “dog”对其具有更高的影响力——即注意力权重。
下图为针对上述翻译问题的一个自注意力模块示意，其中 $x_1, \dotsb, x_{11}$ 分别代表句子中每个词的编码（为简化起见不考虑结束符等辅助符号），$y_1, \dotsb, y_{11}$ 分别为自注意力模块对每个词的输出， $e_{ij}$ 即为自注意力模型输出 时在输入 $x_i$ 上投射的注意力权重。下图以 $y_9$（即针对“it”一词）为例，示意了各个输入编码上的注意力权重分配。
<p align="center"><img src="/datasets/posts/nlp/encoder_vvv.jpg" width="60%" height="60%" title="encoder" alt="encoder"></p>

```python
class EncoderBlock(nn.Module):
    """transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(d2l.Encoder):
    """transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

### 5、解码器中的attention
在解码器中先使用VVV模式，该自注意力模块与编码器自注意力模块的结构非常类似，唯一不同的是添加了掩膜机制，这是由于在解码器中，自注意力模块只被允许处理当前项之前的那些项，这一点与编码器需要“看到”所有项是不同的。上述目标的实现方式非常简单，只要在softmax 注意力权重概率化之前，用掩膜将当前处理项之后的所有项隐去即可，即将注意力的计算改为如下具有掩膜的形式

<p align="center"><img src="/datasets/posts/nlp/decoder_qvv.jpg" width="60%" height="60%" title="decoder" alt="decoder"></p>

解码器自注意力模块之后是编码-解码注意力模块：该模块也被构造为MHA结构、QVV模式。其中`Q`来自于上一个解码器的输出，而`V`来自于最后一个编码器输出（即也是编码器栈的最终输出）。该注意力模块能够使得解码器中的每个项都能够有重点的“看一遍”输入序列中的每一个词，这一点与基于 RNN结构的 Seq2Seq 结构类似。

```python
class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```


## 三、参数量计算

设：$V$ : 词表量；$H$ : embedding；$L$: Layers ；$S$：输入句子最大长度；$B$：batch size<br>

$ \frak{L}$：损失函数 <br>

$ FLOPs $ ： 计算量

### 1、参数量

1. Embedding 层： $VH$
2. 每层Transformer Block: $ VH + L(12H^2+13H)$
    * 两个layer normalization，总参数量为 : $4H$。<br>
    层归一化的公式：$y = w \odot \frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}} + b$ ，即：每个LN有两个 $H$ 维度的参数。<br>
    需要计算3个梯度：$\frac{\partial\frak{L}}{\partial w}$，$\frac{\partial\frak{L}}{\partial b}$， $\frac{\partial\frak{L}}{\partial x}$
    * attention部分有 $Q, K, V$ 和输出的权重矩阵以及偏置，总参数量 $4H^2+4H$
    * MLP部分由两个线性层组成，分别从 $ H \rarr 4H$， $4H \rarr H$，总参数量为 $8H^2+5H$

### 2、计算量

已知：对于形状为 $m \times k, k \times n$ 的两个矩阵相乘，其浮点数运算次数为：$2mkn$

1. 每个Tranformer Block的前向过程，有 $24BSH^2 + 4BS^2H$ 的$FLOPs$
    * attention 中计算 $Q, K, V$，计算次数：$8BSH^2$
    * 计算attention score时，$QK^T$，计算次数：$2BS^2H$
    * attention score 作用到 $V$ 上，计算次数：$2BS^2H$
    * MLP 的 两个全连接层，分别从 $ H \rarr 4H$， $4H \rarr H$，计算次数：$16BSH^2$

2. 每个Transformer Block的反向过程，有 $48BSH^2 + 8BS^2H$ 的 $FLOPs$
    * 简单来说，对于matmul和conv算子而言，它们额反向计算过程需要对两个输入求导，可以粗略估计反向的计算量就是前向的两倍。

### 3、通信量

1. 数据并行（Data Parallel）
假设：
    * 把数据分成 $dps$ 份。模型就会copy $dps$ 份，每次更新模型参数时，需要把更新值合并到主节点中，计算完后，然后再分发给各个节点中。<br>
    * 模型大小为6B，且梯度为FP16模式，则模型的梯度大小一共为12GB。 <br>


一次all_reduce的通信量：$\frac{model}{dps} \times (dps - 1) \times 2$：
<br>
   * 收集：每个节点，receive $dps-1$ 次数据，每次receive数据的大小 $\frac{model}{dps}$
   * 分发：每个节点，receive $dps-1$ 次数据，每次receive数据的大小 $\frac{model}{dps}$

2. Pipeline Parallel


