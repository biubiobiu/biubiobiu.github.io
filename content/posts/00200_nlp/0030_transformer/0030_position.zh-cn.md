---
title: "位置编码"
date: 2021-09-08T12:30:40+08:00
menu:
  sidebar:
    name: 位置编码
    identifier: transformer-summary-position
    parent: transformer-github
    weight: 30
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["NLP", "Transformer", "位置编码"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 位置编码

### 1、绝对位置编码

最早出现于Transformer文章中，目的是为了弥补模型中位置信息的缺失。<br>
输入：$\bold{X} \in \R^{n \times d}$ 包含一个序列中n个词元的d维嵌入表示。<br>
位置编码：$\bold{P} \in \R^{n \times d}$, 矩阵第i行 偶数列、奇数列：用不同的频率、偏移来记录位置信息。
$$p_{i,2j} = sin(\frac{i}{10000^{\frac{2j}{d}}})$$
$$p_{i,2j+1} = cos(\frac{i}{10000^{\frac{2j}{d}}})$$

在 $\bold{X} + \bold{P}$ 时，当$\bold{X}$的幅度值比$\bold{P}$小或者差不多时，可以增大$\bold{X}$的幅度值，以保证$\bold{X}$的主导性。
$$
\bold{X} \times M + \bold{P}
$$

<p align="center"><img src="/datasets/posts/nlp/position_em.png" width="50%" height="50%" title="position" alt="position"></p>


### 2、相对位置编码

Google于2018年提出的 <a href="https://arxiv.org/pdf/1803.02155.pdf" target="bland">《Self-Attention with Relative Position Representations》</a> 。该方法出自Transformer的原班人马，通过在attention模块中加入可训练的参数，帮助模型来记住输入中的相对位置。

### 3、ALiBi

<a href="https://arxiv.org/pdf/2108.12409.pdf" target="bland">ALiBi</a>

### 4、旋转位置编码(RoPE)

<a href="https://arxiv.org/pdf/2104.09864.pdf" target="bland">RoPE</a>

个人理解：对embedding向量做一个角度旋转。由于d维的向量旋转太复杂，只对2维的向量做旋转。所以d维的向量，有d/2个小向量。
旋转的基本角度：

<p align="center"><img src="/datasets/posts/nlp/RoPE_0.png" width="90%" height="90%" title="position" alt="position"></p>
<p align="center"><img src="/datasets/posts/nlp/RoPE_1.png" width="90%" height="90%" title="position" alt="position"></p>
参考：<a href="https://spaces.ac.cn/archives/8265" target="bland">苏剑林的blog</a>

```python
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()
    # torch.polar 的文档
    # https://pytorch.org/docs/stable/generated/torch.polar.html
    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.wq = Linear(...)
        self.wk = Linear(...)
        self.wv = Linear(...)
        
        self.freqs_cis = precompute_freqs_cis(dim, max_seq_len * 2)

    def forward(self, x: torch.Tensor):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seq_len, dim)
        xk = xk.view(batch_size, seq_len, dim)
        xv = xv.view(batch_size, seq_len, dim)

        # attention 操作之前，应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # scores.shape = (bs, seqlen, seqlen)
        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(dim)
        scores = F.softmax(scores.float(), dim=-1)
        output = torch.matmul(scores, xv)  # (batch_size, seq_len, dim)
  # ......
```