---
title: "模型训练"
date: 2022-04-08T06:00:20+06:00
menu:
  sidebar:
    name: 模型训练
    identifier: torch-model
    parent: torch
    weight: 50
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["torch","训练","模型"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、数据预处理

```python
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np

class MyDataset(Dataset):
    """
    下载数据、初始化数据，都可以在这里完成
    """
    def __init__(self):
        xy = np.loadtxt('../dataSet/diabetes.csv.gz', delimiter=',', dtype=np.float32) # 使用numpy读取数据
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.len = xy.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# 创建Dataset对象
dataset = MyDataset()
# 创建DataLoadder对象
dataloader = DataLoader(dataset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=2)
# 循环DataLoader对象
num_epoches = 100
for epoch in range(num_epoches)
    for img, label in dataloader:
        # 将数据从dataloader中读取出来，一次读取的样本数为32个


# class torch.utils.data.DataLoader(dataset,            # 加载数据的数据集
#                                   batch_size=24,      # 每批加载多少个样本
#                                   shuffle=False,      # Ture：每个epoch对数据打乱
#                                   sampler=None,       # 定义从数据集中提取样本的策略，返回一个样本
#                                   batch_sampler=None, #
#                                   num_workers=12,     # 0：表示数据将在主进程中加载,12表示开12个进程
#                                   collate_fn=, 
#                                   pin_memory=False, 
#                                   drop_last=False, 
#                                   timeout=0, 
#                                   worker_init_fn=None)

```
> DataLoader 中：
> 1. 其中: `__getitem__()`的含义： <br>
>    如果在类中定义了`__getitem__()`方法，那么类的实例对象P就可以实现P[key]取值。当实例对象P[key]运算时，就会调用类中的`__getitem__()`方法
> 2. 其中：`__iter__()`的含义：<br>
>    每次迭代，都会执行`__iter__()`，返回的值放在data_r中。每次迭代返回的是一个batch的数据。

|torchvision的包||
|:--|:--|
|torchvision.datasets||
|torchvision.models|包含了常用的网络结构，并提供了预训练模型|
|torchvision.transforms|提供了一般的图像转换操作类|

> <font color=#a020f0>torchvision.transforms</font>：提供了一般的图像转换操作类：
> 1. [x] <font color=#a020f0>torchvision.transforms.ToTensor()</font><br>
    功能：<br>
    输入的Tensor数据类型必须是float32，才有这样的功能：把shape=(H,W,C)像素值范围为[0,255]的PIL.Image或者numpy.ndarrsy，转换为shape=(C,H,W)像素值范围为[0.0,1.0]的torch.FloatTensor
> 2. [x] <font color=#a020f0>torchvision.transforms.Normmalize(mean, std)</font><br>
    功能：<br>
    mean=(R,G,B), std=(R,G,B), 公式channel=(channel-mean)/std进行规范化
> 3. [x] <font color=#a020f0>torchvision.transforms.RandomCrop(size, padding=0)</font><br>
    功能：裁剪<br>
> 4. [x] <font color=#a020f0>torchvision.transforms.RandomSizedCrop(size, interpolation=2)</font><br>
    功能：裁剪<br>
> 5. [x] <font color=#a020f0>torchvision.transforms.Compose()</font><br>
    功能：把多个transform组合起来使用<br>
    示例：<br>
    from torchvision import transforms as transforms <br>
    transform = transforms.Compose([ <br>
    　　transforms.Resize(96), <br>
    　　transforms.ToTensor(), <br>
    　　transforms.Normalize((0.5,0.5,0.5), (0.1,0.1,0.1)) <br>
    ])<br>
    train_dataset = torchvision.datasets.MNIST(root='./data/', train=True, transform=transform, download=True)<br>

## 二、模型

### 1、模型的定义

> **模型定义**
> 1. 模型定义时：需要进行更新的参数注册为<font color=#a020f0>Parameter</font>，不需要进行更新的参数注册为<font color=#a020f0>buffer</font>
>    1. 网络中的参数保存成<font color=#00ffa0>OrderedDict</font>形式，这些参数包括2种：nn.Parameter 和 buffer
>    2. <font color=#a020f0>torch.nn.register_parameter()</font> 用于注册Parameter实例到当前Module; <br> <font color=#a020f0>Module.parameters()</font>函数会返回当前Module中注册的所有Parameter迭代器。<br>
>     创建：<br>
>       * 将模型的成员变量(self.xxx)通过 nn.Parameter()创建，会自动注册到parameters中
>       * 通过nn.Parameter()创建普通的Parameter对象，不作为模型的成员变量，然后将Parameter对象通过register_parameter()进行注册
>    3. <font color=#a020f0>torch.nn.register_buffer()</font> 用于注册Buffer实例到当前Module中; <br>
>     <font color=#a020f0>Module.buffers()</font>函数会返回当前Module中注册的Buffer迭代器
> 
> 2. 模型保存的参数是 Model.state_dict() 返回的<font color=#00ffa0>OrderedDict</font>，包含当前Module中注册的所有Parameter和Buffer
> 3. 模型进行设备移动时，模型中注册的参数(parameter和buffer)会同时进行移动

> 1. [x] <font color=#a020f0>torch.nn.Parameter()</font>。<br>
     功能：将一个不可训练的类型Tensor转换为可训练的类型。并将这个parameter绑定到这个module里。


> **定义MyModel**
> 1. 必须继承nn.Module这个类，要让pytorch知道这个类是一个Module
> 2. 在`__init__`(self)中，设置好需要的组件
> 3. 在forward(self, x)中，用定义好的组建进行组装，像搭建积木一样把网络结果搭建出来。


```python
# Model 模块
class Module(object):
    dump_patches = False
    _version = 1

    def __init__(self):
        """
        Initializes internal Module state, shared by both nn.Module and ScriptModule.
        """
        torch._C._log_api_usage_once("python.nn_module")

        self.training = True
        self._parameters = OrderedDict()                 # 2.
        self._buffers = OrderedDict()                    # 1. 
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._state_dict_hooks = OrderedDict()
        self._load_state_dict_pre_hooks = OrderedDict()
        self._modules = OrderedDict()

    def forward(self, *input):
    	raise NotImplementedError

    # 1. add a persistent buffer to the module. 
    # >>> self.register_buffer('running_mean', torch.zeros(num_features))
    def register_buffer(self, name, tensor):
    	'''...'''
    	self._buffers[name] = tensor

    # 2. add a parameter to the module.
    def register_parameter(self, name, param):
    	'''...'''
    	self._parameters[name] = param

    # 3. add a child module to the current module.
    def add_module(self, name, module):
    	'''...'''
    	self._modules[name] = module

    # Typical use includes initializing the parameters of a model
    def apply(self, fn):
    	return self
    
    # Moves all model parameters and buffers to the GPU
    def cuda(self, device=None):
    	return self

    # Moves all model parameters and buffers to the CPU
    def cpu(self):
    	return self

    # cast all parameters and buffers to :attr: `dst_type`
    def type(self, dst_type):
    	return self

    # Moves and/or casts the parameters and buffers
    # args: device
    #       dtype
    #       tensor
    def to(self, *args, **kwargs):
    	return self

    # Registers a backward hook on the module
    def register_backward_hook(self, hook):
    def register_forward_pre_hook(self, hook):
    def register_forward_hook(self, hook):
    # Returns a dictionary containing a whole state of the module.
    def state_dict(self, destination=None, prefix='', keep_vars=False):
    	
    # 
    def load_state_dict(self, state_dict, strict=True):
    # Returns an iterator over module parameters.
    def parameters(self, recurse=True):
    	for name, param in self.named_parameters(recurse=recurse):
            yield param
    # Returns an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.
    def named_parameters(self, prefix='', recurse=True):
    	yield elem
    # Returns an iterator over module buffers.
    def buffers(self, recurse=True):
    	for name, buf in self.named_buffers(recurse=recurse):
            yield buf
    # Returns an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.
    def named_buffers(self, prefix='', recurse=True):
    	yield elem
    # Returns an iterator over all modules in the network.
    def modules(self):
    	for name, module in self.named_modules():
            yield module
    # Returns an iterator over all modules in the network, yielding both the name of the module as well as the module itself.
    def named_modules(self, memo=None, prefix=''):
    	yield m 
    # Sets the module in training mode.
    def train(self, mode=True):
    	return self
    # Sets the module in evaluation mode.
    def eval(self):
    	return self.train(False)
    # Change if autograd should record operations on parameters in this module.
    def requires_grad_(self, requires_grad=True):
    	return self
    # Sets gradients of all model parameters to zero.
    def zero_grad(self):
    #
    def _get_name(self):
    	return self.__class__.__name__
    #


```

### 2、模型的保存域加载
这主要有两种方法序列化和恢复模型。

1. 第一种（推荐）只保存和加载模型参数：<br>
**保存**：<br>
torch.save(<font color=#a020f0>the_model.state_dict()</font>, PATH) <br>
**读取**：先读取Model的网络定义，在读取模型参数<br>
the_model = TheModelClass(*args, **kwargs) <br>
the_model.load_state_dict(torch.load(PATH)) <br>

2. 第二种保存和加载整个模型：<br>
**保存**：<br>
torch.save(the_model, PATH)<br>
**读取**：因为保存了整个模型，可以直接加载<br>
the_model = torch.load(PATH)<br>

### 3、模型初始化

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

net = BuildModle()
for m in net.modules():
    m.weight.data.normal_(0,math.sqrt(2./n))
    m.weight.data.fill_(1)
    m.bias.data.zero_()

```

初始化的方法：<font color=#a020f0>torch.nn.init</font><br>

1. [x] <font color=#a020f0>torch.nn.init.xavier_normal_(tensor,)</font>
2. [x] <font color=#a020f0>torch.nn.init.xavier_uniform(tensor,)</font>
3. [x] <font color=#a020f0>torch.nn.init.kaiming_normal_()</font>
4. [x] <font color=#a020f0>torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')</font>
5. [x] <font color=#a020f0>torch.nn.init.uniform_(tensor, a=0, b=1)</font>。均匀分布 U(a,b)
6. [x] <font color=#a020f0>torch.nn.init.normal_(tensor, mean=0, std=1)</font>。正态分布
7. [x] <font color=#a020f0>torch.nn.init.constant_(tensor, val)</font>。常数
8. [x] <font color=#a020f0>torch.nn.init.eye_(tensor)</font>。单位矩阵
9. [x] <font color=#a020f0>torch.nn.init.orthogonal_(tensor, gain=1)</font>。正交初始化
10. [x] <font color=#a020f0>torch.nn.init.sparse_(tensor, sparsity, std=0.01)</font>。从正态分布 N~(0, std)中进行稀疏化。<br>
sparsity：每个column稀疏的比例


## 三、学习率

> 1. [x] 自定义调整：<br>
    示例：<font color=#a020f0>torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)</font>
    参数：<br>
    optimizer: 优化器 <br>
    lr_lambda: 为 optimizer.param_groups中的每个组计算一个乘法因子 <br>
    last_epoch: 是从last_start开始后已经记录了多少个epoch <br>
> 2. [x] 有序调整-StepLR：<br>
    示例：<font color=#a020f0>torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)</font>
    参数：<br>
    optimizer: 优化器 <br>
    step_size(int): 学习率下降间隔数。将学习率调整为 lr*gamma <br>
    gamma(float): 学习率调整倍数，默认为0.1 <br>
    last_epoch: 是从last_start开始后已经记录了多少个epoch <br>
> 3. [x] 有序调整-MultiStepLR：<br>
    示例：<font color=#a020f0>torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)</font>
    参数：<br>
    optimizer: 优化器 <br>
    milestones(list): lr 改变时的epoch数，比如[10,15,20,22,]。将学习率调整为 lr * gamma <br>
    gamma(float): 学习率调整倍数，默认为0.1 <br>
    last_epoch: 是从last_start开始后已经记录了多少个epoch <br>
> 4. [x] 有序调整-ExponentialLR：<br>
    示例：<font color=#a020f0>torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)</font>
    参数：<br>
    optimizer: 优化器 <br>
    gamma(float): 学习率调整倍数，默认为0.1。 每个epoch都衰减 lr = lr * gamma <br>
    last_epoch: 是从last_start开始后已经记录了多少个epoch <br>

## 四、优化器

|优化步骤|解释|
|:--|:--|
|class torch.optim.Optimizer(params, defaults)|params: Variable或者dict的iterable。指定了什么参数应当被优化<br>defaults：包含了优化选项默认值的字典|
|load_stat_dict(state_dict)|加载optimizer状态<br>state_dict: optimizer的状态。应当是一个调用<br>state_dict()所返回的对象|
|state_dict()|以dict返回optimizer的状态。包含两项：<br>1、state：一个保存了当前优化状态的dict<br>2、param_groups：一个包含了全部参数组的dict|
|step(closure)|进行单次优化(参数更新)<br>closure(一个函数callable)：一个重新评价模型并返回loss的闭包。|
|zero_grad()|清空所有被优化过的Variable的梯度|

```python
# 实例
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer.zero_grad()
loss_fn(model(input), target).backward()
optimizer.step()
```

参数组(param_groups): 在finetune时，某层定制学习率、某层学习率置零操作中，都会涉及参数组的概念。
1. optimizer对参数的管理是基于组的概念，可以为每一组参数配置特定的 lr, momentum, weight_decay等
2. 参数组在optimizer中表现为一个list(self.param_groups), 其中每个元素是一个dict，表示一个参数及其相应的配置
dict中包括 params, weight_decay, lr, momentum 等字段。

> 常用优化器：
> 1. [x] <font color=#a020f0>torch.optim.SGD()</font> <br>
    示例：optimizer = torch.optim.SGD(model.parameters(), lr=0.07) <br>
    参数: <br>
    params:  待优化参数的iterable或者是定义了参数组的dict <br>
    lr=1e-5:      学习率 <br>
    momentum=0 ： 动量因子 <br>
    dampening=0  ：动量的抑制因子 <br>
    weight_decay=0：权重衰减 <br>
    nesterov=False：使用nesterov动量

> 2. [x] <font color=#a020f0>torch.optim.Rprop()</font> <br>
    示例：optimizer = torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50)) <br>
    参数: <br>
    params (iterable): – 待优化参数的iterable或者是定义了参数组的dict <br>
    lr (float, 可选): – 学习率（默认：1e-2） <br>
    etas (Tuple[float, float], 可选): – 一对（etaminus，etaplis）, 它们分别是乘法的增加和减小的因子（默认：0.5，1.2） <br>
    step_sizes (Tuple[float, float], 可选): – 允许的一对最小和最大的步长（默认：1e-6，50） <br>
      
> 3. [x] <font color=#a020f0>torch.optim.RMSprop()</font> <br>
    示例：optimizer = torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False) <br>
    参数: <br>
    params (iterable): – 待优化参数的iterable或者是定义了参数组的dict <br>
    lr (float, 可选): – 学习率（默认：1e-2） <br>
    momentum (float, 可选): – 动量因子（默认：0） <br>
    alpha (float, 可选): – 平滑常数（默认：0.99） <br>
    eps (float, 可选): – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8） <br>
    centered (bool, 可选): – 如果为True，计算中心化的RMSProp，并且用它的方差预测值对梯度进行归一化 <br>
    weight_decay (float, 可选): – 权重衰减（L2惩罚）（默认: 0） <br>

> 4. [x] <font color=#a020f0>torch.optim.ASGD()</font> <br>
    示例：optimizer = torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0) <br>
    参数: <br>
    params (iterable): – 待优化参数的iterable或者是定义了参数组的dict <br>
    lr (float, 可选): – 学习率（默认：1e-2） <br>
    lambd (float, 可选): – 衰减项（默认：1e-4） <br>
    alpha (float, 可选): – eta更新的指数（默认：0.75） <br>
    t0 (float, 可选): – 指明在哪一次开始平均化（默认：1e6） <br>
    weight_decay (float, 可选): – 权重衰减（L2惩罚）（默认: 0） <br>

> 5. [x] <font color=#a020f0>torch.optim.Adamax()</font> <br>
    示例：optimizer = torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) <br>
    参数: <br>
    params (iterable): – 待优化参数的iterable或者是定义了参数组的dict <br>
    lr (float, 可选): – 学习率（默认：2e-3） <br>
    betas (Tuple[float, float], 可选): – 用于计算梯度以及梯度平方的运行平均值的系数 <br>
    eps (float, 可选): – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8） <br>
    weight_decay (float, 可选): – 权重衰减（L2惩罚）（默认: 0） <br>

> 6. [x] <font color=#a020f0>torch.optim.Adam()</font> <br>
    示例：optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) <br>
    参数: <br>
    params (iterable): – 待优化参数的iterable或者是定义了参数组的dict <br>
    lr (float, 可选): – 学习率（默认：1e-3） <br>
    betas (Tuple[float, float], 可选): – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999） <br>
    eps (float, 可选): – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8） <br>
    weight_decay (float, 可选): – 权重衰减（L2惩罚）（默认: 0） <br>

> 7. [x] <font color=#a020f0>torch.optim.Adagrad()</font> <br>
    示例：optimizer = torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0) <br>
    参数: <br>
    params (iterable): – 待优化参数的iterable或者是定义了参数组的dict <br>
    lr (float, 可选): – 学习率（默认: 1e-2） <br>
    lr_decay (float, 可选): – 学习率衰减（默认: 0） <br>
    weight_decay (float, 可选): – 权重衰减（L2惩罚）（默认: 0） <br>

> 8. [x] <font color=#a020f0>torch.optim.Adadelta()</font> <br>
    示例：optimizer = torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0) <br>
    参数: <br>
    params (iterable): – 待优化参数的iterable或者是定义了参数组的dict <br>
    rho (float, 可选): – 用于计算平方梯度的运行平均值的系数（默认：0.9） <br>
    eps (float, 可选): – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-6） <br>
    lr (float, 可选): – 在delta被应用到参数更新之前对它缩放的系数（默认：1.0） <br>
    weight_decay (float, 可选): – 权重衰减（L2惩罚）（默认: 0） <br>



```python

class Optimizer(object):
	def __init__(self, params, defaults):
		torch._C._log_api_usage_once("python.optimizer")
	    self.defaults = defaults

	    self.state = defaultdict(dict)
	    self.param_groups = [] # 元素: {"params": [torch.nn.parameter.Parameter, ...]}

	    param_groups = list(params)
	    if not isinstance(param_groups[0], dict):
	        param_groups = [{'params': param_groups}]

	    for param_group in param_groups:
	        self.add_param_group(param_group)
	# Returns the state of the optimizer as a :class:`dict`.
    def state_dict(self):
    	return {
            'state': packed_state,
            'param_groups': param_groups,
        }
    # Loads the optimizer state.
    def load_state_dict(self, state_dict):
    # Clears the gradients of all optimized :class:`torch.Tensor`
    def zero_grad(self):
    # Performs a single optimization step (parameter update)
    def step(self, closure):
    	  raise NotImplementedError
    # Add a param group to the :class:`Optimizer` s `param_groups`.
    def add_param_group(self, param_group):

```


## 五、loss

## 六、

