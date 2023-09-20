---
title: "cuda"
date: 2021-12-08T16:00:20+08:00
menu:
  sidebar:
    name: cuda
    identifier: python-env-cuda
    parent: python-env
    weight: 20
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["python","cuda"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、简介

### 1、CUDA
CUDA：英伟达开发的一个通用并行计算平台和编程模型，能让你调用GPU的指令集及其并行计算单元。基于cuda编程可以利用GPU的并行计算引擎来更高效地计算。<br>
特点：
1. GPU有更多的计算核心，适合数据并行的<font color=#a020f0>计算密集型任务</font>。
2. CPU有较少的运算核心，适合实现复杂的逻辑计算，用于<font color=#a020f0>控制</font>密集型任务。
3. 对比一下：
   * CPU -- 线程是重量级的，上下文切换开销较大。 负责处理逻辑复杂的串行程序
   * GPU -- 由于存在较多核心，线程是轻量级的。负责处理数据密集型的并行机选程序

### 2、CUDA编程模型
CUDA编程模型是一个异构模型，需要CPU和GPU协同工作，在CUDA中有两个重要的概念：host和device。<br>
host: CPU + 其内存<br>
device: GPU + 其内存<br>

典型的CUDA程序的执行流程：
1. 分配host内存，并进行数据初始化
2. 分配device内存，并从host将数据copy到device上
3. 调用CUDA的核函数在device上完成指定的运算
4. 将device上的运算结果copy到host上
5. 释放device和host上分配的内存。


### 3、cuDNN

cuDNN: CUDA Deep Neural Network软件库，是一个用于深度神经网络的GPU加速原语库。<br>
TensorRT: 是一套用于高性能深度学习接口的SDK，其包含深度学习接口优化器、运行时优化器，能为深度学习接口提供低延迟和高通量的特性。

## 二、CUDA安装

### 1、驱动安装

<a href="https://www.nvidia.cn/geforce/drivers/" target="bland">NVIDIA驱动</a><br>

关键点：CUDA和显卡驱动没有一一对应的关系，一般情况下安装最新的驱动。

### 2、CUDA安装

<a href="https://developer.nvidia.com/cuda-toolkit-archive" target="bland">CUDA下载</a><br>

CUDA: 只是一个工具包，在同一设备上可以安装多个不同的版本，比如：9.0，10.0，11.0。一般情况下安装最新的驱动，然后根据自己的需求选择不同CUDA工具包就行了。但在离线安装CUDA时会绑定CUDA和驱动程序，所以在使用多个CUDA的时候就不要选择离线安装CUDA了。<br>

安装步骤:<br>
1. 不用选择太高的cuda版本，太高反而兼容性不好，要兼顾Tensorflow等架构的版本
2. 安装包下载后，一路默认安装就好。检查是否安装成功：nvcc -V
3. cuda的安装包中包含NVIDIA驱动，安装时取消勾选安装驱动，只安装工具包就行

<p align="center"><img src="/datasets/posts/language/cuda_install.png" width="90%" height="90%" title="cuda" alt="cuda"></p>

CUDA安装后，配置环境变量：
<p align="center"><img src="/datasets/posts/language/cuda_path.png" width="90%" height="90%" title="cuda" alt="cuda"></p>

CUDA10.1是之前安装的，CUDA11.1是之后安装的，所以默认CUDA10.1的环境变量在CUA11.1之前，CUDA_PATH环境变量被CUDA11.1覆盖
<p align="center"><img src="/datasets/posts/language/cuda_version_1.png" width="90%" height="90%" title="cuda" alt="cuda"></p>

**CUDA版本切换：**<br>
切换CUDA版本时，只需要切换环境变量中CUDA的顺序即可，比如让CUDA11.1生效，则CUDA11.1环境变量在CUDA10.1之前。
<p align="center"><img src="/datasets/posts/language/cuda_version_check.png" width="90%" height="90%" title="cuda" alt="cuda"></p>

### 3、cuDNN安装
<a href="https://developer.nvidia.com/rdp/cudnn-archive" target="bland">cuDNN下载</a><br>
cuDNN：是一个SDK，是一个专门用于神经网路的加速包，它跟CUDA没有一一对应的关系。即：每个CUDA版本可能有好几个cuDNN版本，一般有一个最新版本的cuDNN版本与CUDA对应更好。<br>

安装步骤：<br>
1. 根据cuda版本选择对应的cudnn版本；不同系统，选择不同的型号。
2. 下载的不是安装包，而是压缩文件，解压后将对应的文件拷贝到cuda安装路径对应的目录中。默认安装的路径：  
    1. 复制 cudnn\bin\cudnn64_5.dll 到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin\
    2. 复制 cudnn\include\cudnn.h   到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include\
    3. 复制 cudnn\lib\x64\cudnn.lib 到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\

### 4、CUDA版本切换

1. 需要哪个版本时，就把环境变量中 CUDA_PATH、NVCUDASAMPLES_ROOT修改成相应的路径<br>
   不用哪个版本时，就把环境变量中的path路径，修改为非实际路径
2. 创建不同的虚拟环境，在虚拟环境中分别安装不同版本的TensorFlow，TensorFlow会根据自身版本的需求找到对应的cuda版本。在需要使用哪个版本时，激活哪个虚拟环境。
    1. 创建虚拟环境： conda create -n py37 python=3.7
    2. 进入该虚拟环境 --> 进入该虚拟环境的路径：cd E:\ProgramFiles\anaconda3\envs\py37
    3. mkdir .\etc\conda\activate.d<br>
       mkdir .\etc\conda\deactivate.d
    4. 在activate.d中创建env_vars.bat，内容<br>
       @set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin<br>
       @set CUDA_INCLUDE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include<br>
       @set CUDA_LIB64=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\lib64<br>
       @set CUDA_NVVP=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\libnvvp<br>
       @set CUDA_lib=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64<br>
       @set OLD_PATH=%PATH%<br>
       @set PATH=%CUDA_PATH%;%CUDA_NVVP%;%CUDA_lib%;%PATH%;%CUDA_INCLUDE%;%CUDA_LIB64%;
    5. 在deactivate.d中创建同名文件env_vars.bat，内容<br>
       @set PATH=%OLD_PATH%


### 5、TF/torch版本 & CUDA版本 & cuDNN版本
<a href="https://pytorch.org/get-started/previous-versions/" target="bland">PyTorch 版本与CUDA的对应关系</a><br>

|torch版本||示例|
|:--|:--|:--|
|v1.8.0|conda 安装|# CUDA 10.2<br>conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch<br># CUDA 11.1<br>conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge<br># CPU Only<br>conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch<br>|
||pip 安装|# CUDA 11.0<br>pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html<br># CUDA 10.2<br>pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0<br># CPU only<br>pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html<br>|
|v1.7.1|conda 安装|# CUDA 9.2<br>conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=9.2 -c pytorch<br># CUDA 10.1<br>conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch<br># CUDA 10.2<br>conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch<br># CUDA 11.0<br>conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch<br># CPU Only<br>conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch<br>|
||pip 安装|# CUDA 11.0<br>pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html<br># CUDA 10.2<br>pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2<br># CUDA 10.1<br>pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html<br># CUDA 9.2<br>pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html<br># CPU only<br>pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html<br>|
|v1.7.0|conda 安装|# CUDA 9.2<br>conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=9.2 -c pytorch<br># CUDA 10.1<br>conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch<br># CUDA 10.2<br>conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch<br># CUDA 11.0<br>conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch<br># CPU Only<br>conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cpuonly -c pytorch<br>|
||pip 安装|# CUDA 11.0<br>pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html<br># CUDA 10.2<br>pip install torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0<br># CUDA 10.1<br>pip install torch==1.7.0+cu101 torchvision==0.8.0+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html<br># CUDA 9.2<br>pip install torch==1.7.0+cu92 torchvision==0.8.0+cu92 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html<br># CPU only<br>pip install torch==1.7.0+cpu torchvision==0.8.0+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html<br>|
||||


|TF版本|Python版本|编译器|构建工具|CUDA|cnDNN|
|:--|:--|:--|:--|:--|:--|
|tensorflow-gpu-2.4.0|3.6~3.8|MSVC 2019|Bazel 3.1.0|8.0|11.0|
|tensorflow-gpu-2.3.0|3.5~3.8|MSVC 2019|Bazel 3.1.0|7.6|10.1|
|tensorflow-gpu-2.2.0|3.5~3.8|MSVC 2019|Bazel 2.0.0|7.6|10.1|
|tensorflow-gpu-2.1.0|3.5~3.7|MSVC 2019|Bazel 0.29.1|7.6|10.1|
|tensorflow-gpu-2.0.0|3.5~3.7|MSVC 2017|Bazel 0.26.1|7.4|10|
|tensorflow-gpu-1.15.0|3.5~3.7|MSVC 2017|Bazel 0.26.1|7.4|10|
|tensorflow-gpu-1.14.0|3.5~3.7|MSVC 2017|Bazel 0.26.1|7.4|10|
|tensorflow-gpu-1.13.0|3.5~3.7|MSVC 2015|Bazel 0.21.0|7.4|10|
|tensorflow-gpu-1.12.0|3.5~3.6|MSVC 2015|Bazel 0.15.0|7.2|9|
|tensorflow-gpu-1.11.0|3.5~3.6|MSVC 2015|Bazel 0.15.0|7|9|
|tensorflow-gpu-1.10.0|3.5~3.6|MSVC 2015|Cmake v3.6.3|7|9|
|tensorflow-gpu-1.9.0|3.5~3.6|MSVC 2015|Cmake v3.6.3|7|9|
|tensorflow-gpu-1.8.0|3.5~3.6|MSVC 2015|Cmake v3.6.3|7|9|
|tensorflow-gpu-1.7.0|3.5~3.6|MSVC 2015|Cmake v3.6.3|7|9|
|tensorflow-gpu-1.6.0|3.5~3.6|MSVC 2015|Cmake v3.6.3|7|9|
|tensorflow-gpu-1.5.0|3.5~3.6|MSVC 2015|Cmake v3.6.3|7|9|
|tensorflow-gpu-1.4.0|3.5~3.6|MSVC 2015|Cmake v3.6.3|6|8|
|tensorflow-gpu-1.3.0|3.5~3.6|MSVC 2015|Cmake v3.6.3|6|8|
|tensorflow-gpu-1.2.0|3.5~3.6|MSVC 2015|Cmake v3.6.3|5.1|8|
|tensorflow-gpu-1.1.0|3.5|MSVC 2015|Cmake v3.6.3|5.1|8|
|tensorflow-gpu-1.0.0|3.5|MSVC 2015|Cmake v3.6.3|5.1|8|

## 三、查看版本

### 1、CUDA版本查看

查看已安装CUDA版本<br>
> 1. 直接在NVIDIA的控制面板里查看NVCUDA.DLL的版本<br>
>   注意：这个版本并不能绝对说明自己安装的CUDA工具包一定是这个版本
> 2. 通过命令：nvcc -V  或者 nvcc --version 
> 3. 直接通过文件查看<br>
   Linux：进入安装目录，然后执行 cat version.txt <br>
   win：CUDA的安装目录中，比如：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2 里面version.txt

### 2、cuDNN版本查看

cuDNN本质上就是一个C语言的H头文件。<br>
cudnn.h的头文件，直接打开查看，在最开始的部分有如下定义：
```cpp
# define CUDNN_MAJOR 7
# define CUDNN_MINOR 5
# define CUDNN_PATCHLEVEL 0

# define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 500 + CUDNN_PATCHLEVEL)
```
即：7500，也就是cudnn的版本为7.5.0

> 1. win：进入安装目录C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include 里 cudnn.h 打开查看
> 2. Linux：进入安装目录 cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2 


