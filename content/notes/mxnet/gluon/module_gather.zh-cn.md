---
title: Gluon模块简介
weight: 220
menu:
  notes:
    name: Gluon模块简介
    identifier: notes-mxnet-module-gather
    parent: notes-mxnet-gluon
    weight: 20
---

{{< note title="gluon模块-结构" >}}
路径.mxnet/gluon/下的树状结构:  
│　　block.py `类：Block, HybridBlock`  
│　　loss.py `各种loss函数`  
│　　parameter.py `类：Parameter, Constant, ParameterDict`  
│　　trainer.py `类：Trainer`  
│　　utils.py `优化操作`  
│　　__init__.py  
│  
├─contrib  
│　　│  
│　　├─cnn  
│　　│　　└─ conv_layers.py  
│　　├─data  
│　　│　　└─ sampler.py  
│　　│  
│　　├─estimator  
│　　│　　│　　estimator.py  
│　　│　　└─ event_handler.py  
│　　│  
│　　├─nn  
│　　│　　└─ basic_layers.py  
│　　│  
│　　└─rnn  
│　　　　 │　　conv_rnn_cell.py  
│　　　　 └─ rnn_cell.py  
│  
├─data `主要是数据处理操作`  
│　　│　　dataloader.py `类：DataLoader`  
│　　│　　dataset.py `常用类: ArrayDataset`  
│　　│　　sampler.py  
│　　│  
│　　└─vision  
│　　　　 │　　datasets.py `可用的数据集-各个类`  
│　　　　 └─ transforms.py `数据预处理-各个类`   
│  
├─model_zoo  
│　　│　　model_store.py  
│　　│  
│　　└─vision  
│　　　　 │　　alexnet.py  
│　　　　 │　　densenet.py  
│　　　　 │　　inception.py  
│　　　　 │　　mobilenet.py  
│　　　　 │　　resnet.py  
│　　　　 │　　squeezenet.py  
│　　　　 └─ vgg.py  
├─nn `网络结构`  
│　　│　　activations.py `定义了各种激活层`  
│　　│　　basic_layers.py `定义了网络的基础层,例如：BN,Dropout等`   
│　　└─  conv_layers.py `定义了各种卷积池化层等`  
│  
└─rnn  
　　 │　　rnn_cell.py  
　　 └─ rnn_layer.py  
{{< /note >}}


{{< note title="gluon模块-导入" >}}

```python
# data
from mxnet.gluon.data import ArrayDataset, DataLoader
from mxnet.gluon.data.vision.transforms import ToTensor, Normalize
# nn
from mxnet.gluon.nn import Block, HybridBlock, Sequential, HybridSequential, Dropout, BatchNorm, Dense, PReLU, Conv2D
# 模型参数
from mxnet.gluon.parameter import Parameter, Constant, ParameterDict
# 训练
from mxnet.gluon.trainer import Trainer
# 损失函数
from mxnet.gluon. import loss        # 损失函数 ['Loss', 'L2Loss', 'L1Loss', 'SigmoidBinaryCrossEntropyLoss', 'SigmoidBCELoss', 'SoftmaxCrossEntropyLoss', 'SoftmaxCELoss', 'KLDivLoss', 'CTCLoss', 'HuberLoss', 'HingeLoss', 'SquaredHingeLoss', 'LogisticLoss', 'TripletLoss', 'PoissonNLLLoss', 'CosineEmbeddingLoss']

```
{{< /note >}}


{{< note title="数据集 - data" >}}
`ToTensor`：将图像数据从uint8格式变换成32位浮点数格式，并除以255使得所有像素的数值均在0到1之间 </p>
`transform_first函数`：数据集的函数。将`ToTensor`的变换应用在每个数据样本（图像和标签）的第一个元素，即图像之上.
```python
from mxnet.gluon import data as gdata
batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size, 
                              shuffle=True,
                              num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size, 
                             shuffle=False,
                             num_workers=num_workers)

```
{{< /note >}}


{{< note title="模型初始化 - init" >}}

```python
from mxnet import init
```
{{< /note >}}

{{< note title="损失函数 - loss" >}}

```python
from mxnet.gluon import loss as gloss
# 平方损失又称L2范数损失
loss = gloss.L2Loss()
# 包含了softmax运算和交叉熵损失运算
loss = gloss.SoftmaxCrossEntropyLoss()
```
{{< /note >}}

{{< note title="优化算法 - Trainer" >}}

```python
from mxnet.gluon import Trainer
```
{{< /note >}}

