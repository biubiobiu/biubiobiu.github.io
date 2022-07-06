---
title: "backbone net"
date: 2021-09-09T06:00:20+06:00
menu:
  sidebar:
    name: backbone net
    identifier: cv-backbone-net
    parent: cv-backbone
    weight: 30
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["backbone","卷积神经网络"]
categories: ["Basic"]
math: true
---

卷积神经网络的发展历程：  
<p align="center"><img src="/datasets/posts/cnn/cnn_net_summary.jpg" title="卷积神经网络" width="80%" height="80%" alt="卷积神经网络"></p>

## 一、Backbone
### 1. LeNet

<a href="https://www.researchgate.net/publication/2985446_Gradient-Based_Learning_Applied_to_Document_Recognition" target="blank">论文</a>  

LeNet：名字来源于第一作者Yann LeCun。是一个奠基性的网络，第一次将卷积神经网络推上舞台。  

- 卷积层+最大池化：卷积层用来识别图像里的空间模式；最大池化用来降低卷积层对位置的敏感度。卷积层块由两个这样的基本单位重复堆叠构成。
- LeNet可以在早起的小数据集上取得较好的效果，但是在更大的真实数据集上表现并不如人意。一方面：神经网络计算复杂，在GPU没有大量普及的20世纪90年代，训练一个多通道、多层、含有大量参数的卷积神经网络是很难完成的；另一方面：当年并没有深入研究参数初始化和非凸优化算法，导致复杂的神经网络的训练通常比较困难。
- 特征本身是由学习得来的，为了表征足够复杂的输入，`特征本身应该分级表示`。想要学习到复杂的多级特征，需要大量的带有标签的数据，这样才能表现得比其他经典方法要好。早期研究只基于小的公开数据集，自2009年ImageNet数据集创建以来，传统方法不再有优势。

<p align="center"><img src="/datasets/posts/cnn/lenet.jpg" title="LeNet" alt="LeNet"></p>
输入：32*32  

- C1-卷积层：卷积层尺寸：6 * 28 * 28；卷积核尺寸：6 * 1 * 5 * 5；可训练参数：(5 * 5 + 1) * 6 
- S2-池化层：池化尺寸：2 * 2；步幅：2；方式：4个输入相加，然后乘以个可训练参数，加上个可训练参数，最后通过sigmoid；输出尺寸：6 * 14 * 14；可训练 参数：2 * 6 
- C3-卷积层：输出尺寸：16 * 10 * 10；卷积核尺寸: 16 * 6 * 5 * 5;  
　　　　　　组合方式：前6个map - 以S2中3个相邻的feature map  
　　　　　　　　　　　再6个map - 以S2中4个相邻的feature map  
　　　　　　　　　　　再3个map - 以S2中不相邻的4个feature map  
　　　　　　　　　　　再1个map - 以S2中所有feature map  
<p align="center"><img src="/datasets/posts/cnn/lenet_1.png" title="组合方式" alt="组合方式"></p>

- S4-池化层：输出尺寸：16 * 5 * 5；池化尺寸：2 * 2；步幅：2  
　　　　　采样方式：4个输入相加，然后乘个可训参数，加上个可训参数，最后通过sigmoid 
- C5-卷积层：输出尺寸：120*1；卷积核：120 * 16 * 5 * 5；可训参数：120 * (16 * 5 * 5 + 1)
- F6-全连接层：输出尺寸：84；对应一个 7 * 12的比特图；可训参数：84 * (120+1)
- Output层-全连接层：输出尺寸：10；分别代表数字0~9


### 2. AlexNet

<a href="https://dl.acm.org/doi/pdf/10.1145/3065386" target="blank">论文</a>  

2012年AlexNet横空出世，这个名字来源于一作的姓名(Alex Krizhevsky)，是Hinton实验室提出的，是卷积神经网络在大规模数据集上的开篇巨作。它赢得了2012年ImageNet图像识别挑战赛，首次证明了学习到的特征可以超越手工设计的特征，使视觉从业者从人工提取特征的特征工程中解脱出来，转向 从数据中自动提取需要的特征，做数据驱动。

<p align="center"><img src="/datasets/posts/cnn/alexnet.png" title="AlexNet" alt="AlexNet"></p>

卷积层：卷积核11 * 11；步幅4；输出：96 * 54  *54  
pool层：pool尺寸3 * 3；步幅2；输出：96 * 26 * 26  
卷积层：卷积核5 * 5；填充2；输出：256 * 26 * 26  
pool层：pool尺寸3 * 3；步幅2；输出：265 * 12 * 12  
卷积层：卷积核3 * 3；填充1；输出：384 * 12 * 12  
卷积层：卷积核3 * 3；填充1；输出：384 * 12 * 12  
卷积层：卷积核3 * 3；填充1；输出：256 * 12 * 12  
pool层：pool尺寸3 * 3；步幅2；输出：256 * 5 * 5  
全连接层：4096；DropOut(0.5)  
全连接层：4096；DropOut(0.5)  
输出层：1000  

创新点：  
1. 加深了网络深度  
2. 激活函数由sigmoid转换为Relu，作用：加快收敛速度；引入非线性，增强非线性的映射能力
3. 使用DropOut，控制模型复杂度
4. 数据增广：翻转、裁剪、颜色变化，进一步扩大数据集来缓解过拟合。
5. 使用GPU计算
6. 局部相应归一化(LRN)：N 表示通道数，在通道维度做局部归一化。比如在第i通道(x,y)像素点做归一化：前后n/2个通道上做局部归一化。


### 3. VGG

<a href="https://arxiv.org/abs/1409.1556" target="blank">论文</a>

AlexNet：指明了深度卷积神经网络可以取得出色的结果，但并没有提供简单的规则，没有指导后来者如何设计新的网络。  
VGG(牛津Visual Geometry Group 实验室)：提供了可以通过`重复使用简单的基础块来构建深度模型`的思路。证明了网络深度对性能的影响。在
LSVRC2014比赛分类项目的第二名。

<p align="center"><img src="/datasets/posts/cnn/vgg.png" title="VGG" alt="VGG"></p>

创新点：  
1. 使用3*3的卷积核替换大尺寸卷积核。增加了网络深度，证明了网络深度对精度的影响 
2. 使用基础块 搭建网络的 思路


### 4. NiN

<a href="https://arxiv.org/abs/1312.4400" target="blank">论文</a>

NiN(network in network)：提出了一个新思路。串联多个(卷积层+全连接层构成的小网络) 来构建一个深层网络。使用1*1的卷积层来替代全连接层，从而使得空间信息能够传递到后面层。  NiN基础块如下：

<p align="center"><img src="/datasets/posts/cnn/nin.png" title="NiN" alt="NiN"></p>

模块堆叠：
|NiN基础块|池化层|NiN基础块|池化层|NiN基础块|池化层|NiN基础块|全局平均池化层|Flatten|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|  

创新点：
1. 使用1*1的卷积层 来替换全连接层，对准确率提升有效果。
2. 串联多个小网络，搭建一个深层网络


### 5. GoogLeNet

<a href="https://arxiv.org/abs/1409.4842" target="blank">论文</a> ImageNet分类Top5错误率：6.67% <br>

GoogLeNet 在ImageNet LSVRC2014比赛分类项目的第一名，名字上向LeNet致敬。GoogLeNet吸收了NiN网络串联的思想，并在此基础上做了很大改进。GoogLeNet采用了模块化的结构(Inception结构，名字与电影《盗梦空间》Inception同名)，方便添加和修改。

<p align="center"><img src="/datasets/posts/cnn/googlenet.png" title="GoogLeNet" alt="GoogLeNet"></p>

模块堆叠：
|Conv2D  MaxPool2D|Conv2D  Conv2D  MaxPool2D|Inception  Inception  MaxPool2D|Inception  Inception  Inception  Inception    Inception  MaxPool2D|Inception  Inception  全局平均池化层|
|:--|:--|:--|:--|:--|


创新点：
1. Inception模块里有4条并行的线路：1 * 1、3 * 3、5 * 5 的卷积层是用来抽取不同空间尺寸下的特征信息，其中中间两条线路会对输入先做1*1的卷积，是为了减少输入通道数，减低复杂度；第四条线路则使用3 * 3最大池化层，后接1 * 1卷积层来改变通道数。4条线路使用了合适的填充来保障`输入与输出的尺寸一致`。
2. 采用4条线路并行，是想提取不同空间尺寸的特征信息，那种信息更有用，在模型训练时让数据自己选择。
3. GoogLeNet跟VGG一样，在主体卷积部分使用5个模块(block)，每个模块之间使用步幅为2的3 * 3最大池化层来减小输出宽高。

### 6. ResNet

<a href="https://arxiv.org/abs/1512.03385" target="blank">论文</a> ImageNet分类Top5错误率：3.57% <br>

随着网络逐渐加深，模型的误差不降反曾。ResNet针对这个问题，在2015年的ImageNet LSVRC-2015比赛中夺魁。以前的网络，网络层拟合的是映射f(x)，而ResNet的网络层拟合的是残差：f(x)-x。残差映射更易于捕捉恒等映射的细微波动。  

<p align="center"><img src="/datasets/posts/cnn/resnet.png" title="ResNet" alt="ResNet"></p>

### 7. ResNeXt

<a href="https://arxiv.org/abs/1611.05431" target="blank">ResNeXt</a>是ResNet和Inception的结合体，是2016年的ImageNet LSVRC-2016比赛的亚军。<br>


### 8. SENet
<a href="https://arxiv.org/abs/1709.01507" target="blank">SENet</a>(2017)是ImageNet 2017（ImageNet收官赛）的冠军模型，ImageNet分类Top5错误率：2.25% <br>
<p align="center"><img src="/datasets/posts/cnn/SEnet.png" width="70%" height="70%" title="SEnet" alt="SEnet"></p>

主要思想：
  1. Squeeze部分。即为压缩部分，原始feature map的维度为 H x W x C，Squeeze做的事情是把H x W x C压缩为1 x 1 x C，相当于把H x W压缩成一维了，实际中一般是用global average pooling实现的。H x W压缩成一维后，相当于这一维参数获得了之前H x W全局的视野，感受区域更广。
  2. Excitation部分。得到Squeeze的1 x 1 x C的表示后，加入一个FC全连接层（Fully Connected），对每个通道的重要性进行预测，得到不同channel的重要性大小后再作用（激励）到之前的feature map的对应channel上，再进行后续操作

提升很大，并且代价很小，通过对通道进行加权，强调有效信息，抑制无效信息，注意力机制，并且是一个通用方法。


### 9. DenseNet

<a href="https://arxiv.org/abs/1608.06993" target="blank">论文</a>(2017)

受ResNet影响，DenseNet将输入和输出拼接在一起；ResNet是：输入+输出。DenseNet的主要构建模块：稠密快(dense block)和过渡层(transition layer)  
稠密块：主要定义输入和输出是如何连接的  
过渡层：用来调控通道数，h/w 尺寸。由于输入和输出拼接在一起，通道数增加，需要过渡层来调控。  

<p align="center"><img src="/datasets/posts/cnn/densenet.png" title="DenseNet" alt="DenseNet"></p>

主要结论：
1. 一些较早层提取出的特征仍然可能被较深层直接使用
2. 过渡层  输出大量冗余特征
3. 最后的分类层，虽然使用了之前的多层信息，但更偏向于使用最后几个feature map，说明在网络的最后几层，某些high-level的特征可能被产生

### 10. SKNet
<a href="https://arxiv.org/abs/1903.06586" target="blank">SKNet</a>(2019) 是对SENet的改进。
SENet 在channel维度上做attention，而SKNet在SENet的基础上又引入了kernel维度上的attention，除此之外，还利用诸如分组卷积和多路卷积的trike来平衡计算量。

### 11. CSPNet
<a href="https://arxiv.org/abs/1911.11" target="blank">CSPNet</a>(2019)
作者想提出一个计算量小效果还好的网络结构。具体来说作者希望：
  - 增强CNN的学习能力
  - 减少计算量
  - 降低内存占用

作者把 CSPNet 应用到分类和检测任务中，发现性能都有所提升，特别是在检测任务中提升更为明显。这也是为什么后续的 YOLOv4 和 YOLOv5 的 backbone 都是基于 CSPNet 修改的

### 12. EfficientNet
<a href="https://arxiv.org/abs/1905.11946" target="blank">EfficientNet</a>(2019)

### 12. VoVNet
<a href="https://arxiv.org/abs/1904.09730" target="blank">VoVNet</a>(2019) 基于DenseNet，实现实时目标检测。 <br>
<a href="https://arxiv.org/abs/1911.06667" target="blank">VoVNet2</a>(2020)

### 13. RepVGG
<a href="https://arxiv.org/abs/2101.03697" target="blank">RepVGG</a>(2021)

### 14. ViT
<a href="https://arxiv.org/abs/2010.11929" target="blank">ViT</a>(2021) 

## 二、各领域的Backbone
### 1、 提升速度的Backbone

#### 1. SqueezeNet
<a href="https://arxiv.org/abs/1602.07360" target="blank">SqueezeNet</a>(2016) 的主要思想：
  1. 多用 1x1 的卷积核，而少用 3x3 的卷积核
  2. 在用 3x3 卷积的时候尽量减少 channel 的数量，从而减少参数量
  3. 延后用 pooling，因为 pooling 会减小 feature map size，延后用 pooling， 这样可以使 size 到后面才减小，而前面的层可以保持一个较大的 size，从而起到提高精度的作用。

#### 2. MobileNet
<a href="https://arxiv.org/abs/1704.04861" target="blank">MobileNet </a>(2017) 是通过优化卷积操作来达到轻量化的目的的，具体来说，文中通过 Deepwise Conv（其实是Deepwise Conv + Pointwise Conv）代替原始的卷积操作实现，从而达到减少计算的目的（通常所使用的是 3×3 的卷积核，计算量会下降到原来的九分之一到八分之一）<br>

#### 3. ShuffleNet
<a href="https://arxiv.org/abs/1707.01083" target="blank">ShuffleNet</a>(2017) 的核心思想是对卷积进行分组，从而减少计算量，但是由于分组相当于将卷积操作局限在某些固定的输入上，为了解决这个问题采用 shuffle 操作将输入打乱，从而解决这个问题。<br>



### 2、 目标检测的Backbone

#### 1. 

#### 2. Darknet
YOLO作者自己写的一个深度学习框架叫<a href="https://arxiv.org/abs/1506.02640" target="blank">Darknet</a>

#### 3.DetNet
旷视2018年提出的<a href="https://arxiv.org/abs/1804.06215" target="blank">DetNet</a>，是一个目标检测的backbone


### 3、 姿态识别的Backbone

<a href="https://arxiv.org/abs/1603.06937" target="blank">《Stacked Hourglass Networks for Human Pose Estimation》</a>(2016)



