---
title: "大模型训练框架"
date: 2023-08-05T12:30:40+08:00
menu:
  sidebar:
    name: 大模型训练框架
    identifier: aigc-summary-train
    parent: aigc-summary
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["aigc","大模型", "训练框架"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

目前训练超大规模语言模型主要有两条技术路线：
1. TPU + XLA + TensorFlow/JAX 
2. GPU + PyTorch + Megatron-LM + DeepSpeed。

前者由Google主导，由于TPU和自家云平台GCP深度绑定，对于非Googler来说， 只可远观而不可把玩，后者背后则有NVIDIA、Meta、MS大厂加持，社区氛围活跃，也更受到群众欢迎。


## 一、简介

### 1、并行计算
模型并行：将模型参数分布到多个GPU上
1. 数据并行(Data parallelism, DP)：复制多份模型，每个副本被放置在不同设备上，并输入数据分片。该过程是并行完成的，所有模型副本在每个训练step结束时同步。
2. 张量并行(Tensor parallelism, TP)：这种方式，我们不把整个激活张量或者梯度张量放在单个GPU上，而是<font color=#f00000>切分参数矩阵，每个GPU计算一部分</font>。该技术有时被称为水平并行或者层内模型并行。缺点是：需要额外通信，降低计算粒度
3. 流水线并行(Pipeline parallelism, PP)：将网络分成多段并行。这有时也称为垂直并行。缺点是：引入流水线气泡
4. Zero Redundancy Optimizer(ZeRO)：将参数分布到数据并行组中，计算之前先获取模型参数。缺点是：需要额外通信


为了能够提升训练的效率，目前都采用混合精度训练，然而混合精度训练，是非常不稳定的，很容易导致梯度爆炸。这个原因是：<font color=#f00000>在做Forword或者Backword的时候，需要把FP32位，降低到FP16位。这个操作有可能会导致精度溢出，从而导致loss爆炸</font>。<br>

### 2、混合精度(AMP)
混合精度 (Automatically Mixed Precision, AMP)

1. 为加速训练，模型的参数是以FP16半精度存储的；
2. 然后，输入数据也是 FP16半精度，与模型参数 foreword计算，激活结果也是FP16半精度；
3. 计算loss，然后backword。在backword之前，需要对loss进行缩放，让他变成Fp32位


### 3、训练时的空间量

#### a. 模型参数（parameter）
需要的空间大小：跟模型大小一致。

#### b. 梯度（gradient）
需要的空间大小：跟模型大小一致。

#### c. 中间状态
以线性层为例：
1. Forword: $y = Wx$
2. Backword: $\nabla x = W^T \nabla y, \nabla W = \nabla y x^T$ <br>
利用梯度更新模型参数时，需要用到：模型输入、输出。所以这些数据是要一直保存，直到参数更新完毕。

需要的空间大小：

#### d. 优化器（Optimizer）
例如：adam。需要保存
1. 模型梯度：$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
2. 模型梯度二次项相关的一些历史信息：$v_t = \beta_2 v_{t-1} + (1-\beta_2)g^2_t$

需要的空间大小：至少2倍的模型参数量。

## 二、Deepspeed

<a href="https://deepspeed.readthedocs.io/en/latest/" target="bland">使用文档</a> <br>

Deepspeed是微软的大规模分布式训练工具。专门用于训练超大模型。增加的功能主要有：
> 1. 3个维度并行化实现万亿参数模型训练
> 2. ZeRO-Offload 使 GPU 单卡能够训练 10 倍大的模型
> 3. 通过 DeepSpeed Sparse Attention 用6倍速度执行10倍长的序列
> 4. 1 比特 Adam 减少 5 倍通信量

DeepSpeed 是一个微软开发的开源深度学习优化库，它通过多种技术手段来加速训练，包括：<font color=#f00000>模型并行化、梯度累积、动态精度缩放、本地模式混合精度等。</font> DeepSpeed基于pytorch构建，只需要简单修改即可迁移。<br>

**DeepSpeed主要包含三部分**：
> 1. Apis：提供易用的api接口，训练、推理只需要简单调用几个api接口即可。<br>
> 最重要的是initialize接口：用来初始化引擎，配置训练参数以及优化技术。配置参数一般保存在config.json文件中。
> 2. runtime：是deepspeed管理、执行、性能优化的核心组件。是用python语言实现的。<br>
> 例如：部署训练任务到分布式设备、数据分区、模型分区、系统优化、微调、故障检测、checkpoints保存和加载。
> 3. ops：用c++和cuda实现底层内核，优化计算和通信。<br>

---
**核心技术**：ZeRO(零冗余优化器)<br>
> 1. ZeRO克服数据并行和模型并行的局限性，同时实现两者的优点。
> 2. 通过在数据并行进程之间，划分：<font color=#f00000>模型状态、梯度、优化器状态</font> 来消除数据并行进程中的内存冗余。
> 3. 在训练期间使用 <font color=#f00000>动态通信调度</font> 来在分布式设备之间共享必要的状态
---

DeepSpeed的核心就在于：<font color=#f00000>GPU显存不够，CPU内存来凑</font>。比方说，我们只有一张10GB的GPU，那么我们很可能需要借助80GB的CPU，才能够训练一个大模型。<br>

**具体点说**，DeepSpeed将当前时刻，训练模型用不到的参数，缓存到CPU中，等到要用到了，再从CPU挪到GPU。这里的“参数”，不仅指的是模型参数，还指optimizer、梯度等。<br>

越多的参数挪到CPU上，GPU的负担就越小；但随之的代价就是，更为频繁的CPU，GPU交互，极大增加了训练推理的时间开销。因此，DeepSpeed使用的一个核心要义是：<font color=#f00000>时间开销和显存占用的权衡</font>。

### 1、使用DeepSpeed

```python
deepspeed --master_port 29500 --num_gpus=2 run_s2s.py --deepspeed ds_config.json
```
<font color=#f00000>--master_port</font>：端口号。最好显示指定，默认为29500，可能会被占用（i.e., 跑了多个DeepSpeed进程）。<br>
<font color=#f00000>--num_gpus</font>: GPU数目，默认会使用当前所见的所有GPU。<br>
<font color=#f00000>--deepspeed</font>: 提供的config文件，用来指定许多DeepSpeed的重要参数。<br>

使用DeepSpeed的一个核心要点，就在于写一个config文件（可以是.json，也可以是类json格式的配置文件），在这个配置文件中，你可以指定你想要的参数，例如，权衡时间和显存。因此，上面几个参数里，最重要的便是--deepspeed，即你提供的config文件，即ZeRO。<br>

### 2、ZeRO
Zero Redundancy Optimizer (ZeRO)是DeepSpeed的workhorse. 用户可以提供不同的ZeRO config文件，来实现DeepSpeed的不同功能特性。<br>

即，传统的深度学习，模型训练并行，是将模型参数复制多份到多张GPU上，只将数据拆分（如，torch的Dataparallel），这样就会有大量的显存冗余浪费。而ZeRO就是为了消除这种冗余，提高对memory的利用率。注意，这里的“memory”不仅指多张GPU memory，还包括CPU。<br>

而ZeRO的实现方法，就是把参数占用，逻辑上分成三种类型。将这些类型的参数划分：
1. <font color=#f00000>optimizer states</font>：即优化器的参数状态。例如，Adam的动量参数。
2. <font color=#f00000>gradients</font>：梯度缓存，对应于optimizer。
3. <font color=#f00000>parameters</font>：模型参数。

对应的，DeepSpeed的ZeRO config文件就可以分为如下几类：
1. <font color=#f00000>ZeRO Stage 1</font>: 划分optimizer states。优化器参数被划分到多个memory上，每个momoey上的进程只负责更新它自己那部分参数。
2. <font color=#f00000>ZeRO Stage 2</font>: 划分gradient。每个memory，只保留它分配到的optimizer state所对应的梯度。这很合理，因为梯度和optimizer是紧密联系在一起的。只知道梯度，不知道optimizer state，是没有办法优化模型参数的。
3. <font color=#f00000>ZeRO Stage 3</font>: 划分模型参数，或者说，不同的layer. ZeRO-3会在forward和backward的时候，自动将模型参数分配到多个memory。


**示例**:
```python
{
    "bfloat16": {
        "enabled": "auto"
    },
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "steps_per_print": 1e5
}
```


## 三、Megatron-LM

<a href="https://arxiv.org/pdf/1909.08053v4.pdf" target="bland">《Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism》</a> <br>
Megatron 是一篇极具影响力的论文，介绍了高效的模型并行架构。Megatron引入了张量并行(tensor parallelism)，这是一种模型并行的变体，它将模型分割成多块，以实现层内模型并行，从而达到与单个GPU基准线76%效率相当的水平（尽管基准线只有峰值FLOPS的30%）。<br>

Megatron意识到如果，你有一个网络模型 $Y=f(XW)$，你沿着列拆分开了 $W=[W1, W2]$ ，然后 $Y=[f(XW1), f(XW2)]$，所以你不需要做任何操作来同步 $Y$，transformer中唯一需要同步（all-reduce）的点是：
1. 正向传播中，在MLP块后拼接模型激活值之前添加dropout时需要同步。
2. 反向传播中，在self-attention块的开始处需要进行同步。

通过在这两个关键点进行同步操作，可以保证Transformer模型在计算过程中的正确性和一致性。

<p align="center"><img src="/datasets/posts/nlp/Megatron.png" width=60% height=60%></p>

<a href="https://arxiv.org/pdf/2201.11990v3.pdf" target="bland">《Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model》</a> <br>


