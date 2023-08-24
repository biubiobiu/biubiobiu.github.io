---
title: Gluon-nn模块
weight: 220
menu:
  notes:
    name: Gluon-nn模块
    identifier: notes-mxnet-module-gluon-nn
    parent: notes-mxnet-gluon
    weight: 30
---

{{< note title="模型基类-Block" >}}

```python
from mxnet.gluon import Block, nn
from mxnet import ndarray as F

class Model(Block):
	def __init__(self, **kwargs):
		super(Model, self).__init__(**kwargs)
		# use name_scope to give child Blocks appropriate names.
		with self.name_scope():
			self.dense0 = nn.Dense(20)
			self.dense1 = nn.Dense(20)

	def forward(self, x):
		x = F.relu(self.dense0(x))
		return F.relu(self.dense1(x))

model = Model()
model.initialize(ctx=mx.cpu(0))
model(F.zeros((10, 10), ctx=mx.cpu(0)))

```
class Block(builtins.object)   
网络的最基础的类，搭建网络时必须继承此Block类 <br>
—————————————————  
Block的两个参数：<br>
- `prefix` : str; 前缀的作用就像一个命名空间。在父模块的作用域下创建的子模块都有父模块的前缀(prefix).  
- `params` : ParameterDict or None; 共享参数。  
　　例如：dense1共享dense0的参数。  
　　　　dense0 = nn.Dense(20)  
　　　　dense1 = nn.Dense(20, params=dense0.collect_params())   

—————————————————   
Block的方法：  
- `collect_params`(self, select=None) 返回一个`ParameterDict`类。默认包含所有的参数；同时也可以正则匹配:  
例如：选出特定的参数 ['conv1_weight', 'conv1_bias', 'fc_weight', 'fc_bias']  
即：model.collect_params('conv1_weight|conv1_bias|fc_weight|fc_bias')  
　　`Parameters`：空 或者 正则表达式  
　　`Returns`: py:class:ParameterDict  
- `forward`(self, *args) 完成前向计算，输入是NDArray列表  
　　Parameters：*args : list of NDArray  
- `hybridize`(self, active=True, **kwargs) 激活/不激活HybridBlock的递归  
　　Parameters： bool, default True  
- `initialize`(self, init=<mxnet.initializer.Uniform object>, ctx=None, verbose=False, force_reinit=False)  
　　对模型的参数初始化，默认是均匀分布。  
　　等价于：block.collect_params().initialize(...)  
　　Parameters：  
　　　　init : Initializer 初始化方法  
　　　　ctx : 设备 或者 设备列表。会把模型copy到所有指定的设备上  
　　　　verbose : bool, default False 是否在初始化时粗略地打印细节。  
　　　　force_reinit : bool, default False 是否重新初始化，即使已经初始化  
- `load_parameters`(self, filename, ctx=None, allow_missing=False, ignore_extra=False, cast_dtype=False, dtype_source='current')  
　　加载模型参数从 用`save_parameters`保存的模型文件中。  
　　Parameters：  
　　　　filename : str 模型文件路径  
　　　　ctx : 设备 或者设备列表。默认使用CPU  
　　　　allow_missing : bool, default False 是否默默跳过模型文件中不存  
　　　　　　在的模型参数。   
　　　　ignore_extra : bool, default False 是否默默忽略模型中不存在的参  
　　　　　　数(模型文件中有，模型定义中没有)  
　　　　cast_dtype : bool, default False 从checkpointload模型时，是否根  
　　　　　　据传入转换NDArray的数据类型  
　　　　dtype_source : str, default 'current' 枚举值：{'current', 'saved'}   
　　　　　　只有再cast_dtype=True时有效，指定模型参数的数据类型  
- `name_scope`(self) 返回一个命名空间，用来管理Block和参数names。  
　　必须在with语句中使用：  
　　with self.name_scope():  
　　　　self.dense = nn.Dense(20)  
- `register_child`(self, block, name=None) 将block注册为子节点，block的属  
　　性将自动注册。  
- `save_parameters`(self, filename)  保持模型参数到磁盘。该方法只保存模型  
　　参数的权重，不保存模型的结构。如果想要保存模型的结构，请使  
　　用:py:meth:`HybridBlock.export`.  
　　Parameters：Path to file.  
- `summary`(self, *inputs) 打印模型的输出和参数的摘要。模型必须被初始化  

—————————————————  
数据描述：  
- `name` :py:class:Block 的名字  
- `params`：返回一个参数字典（不包含子节点的参数）  
- `prefix`：返回py:class:Block的前缀  

{{< /note >}}

{{< note title="模型参数-Parameter" >}}
```python
ctx = mx.gpu(0)
x = mx.nd.zeros((16, 100), ctx=ctx)
w = mx.gluon.Parameter('fc_weight', shape=(64, 100), init=mx.init.Xavier())
b = mx.gluon.Parameter('fc_bias', shape=(64,), init=mx.init.Zero())
w.initialize(ctx=ctx)
b.initialize(ctx=ctx)
out = mx.nd.FullyConnected(x, w.data(ctx), b.data(ctx), num_hidden=64)
```

class:Parameter 一个存放Blocks的参数的权重的容器。初始化后`Parameter.initialize(...)`，会copy所有参数权重到每个设备上。如果`grad_req`不为null，在每个设备上，该容器会拥有一个梯度向量。  

Parameter(name,  
　　　　　grad_req='write',   
　　　　　shape=None,  
　　　　　dtype=<class 'numpy.float32'>,  
　　　　　lr_mult=1.0,  
　　　　　wd_mult=1.0,  
　　　　　init=None,  
　　　　　allow_deferred_init=False,  
　　　　　differentiable=True,  
　　　　　stype='default',  
　　　　　grad_stype='default')  

形参：  
——————————  
- `name` : 
str类型；参数的名字。  
- `grad_req` : 枚举值：{'write', 'add', 'null'}, 默认值：'write'。指定怎么更新梯度到梯度向量。  
　　`'write'`:每次把梯度值写到 梯度向量中  
　　`'add'`: 每次把计算的梯度值add到梯度向量中. 在每次迭代之前，  
　　　　你需要手动调用`zero_grad()`来清理梯度缓存。  
　　`'null'`: 参数不需要计算梯度，不会分配梯度向量。  
- `shape` : int or tuple of int, default None. 参数的尺寸.  
- `dtype` : numpy.dtype or str, default 'float32'. 参数的数据类型  
- `lr_mult` : float, default 1.0, 学习率.  
- `wd_mult` : float, default 1.0, 权重衰减率 L2  
- `init` : Initializer, default None. 参数的初始化，默认全局初始化  
- `stype`: 枚举值: {'default', 'row_sparse', 'csr'}, defaults to 'default'. 参数的存储类型。  
- `grad_stype`: 枚举值: {'default', 'row_sparse', 'csr'}, defaults to 'default'. 参数梯度的存储类型  

属性:  
——————————  
- `grad_req` : 枚举值:{'write', 'add', 'null'}  可以在初始化之前/之后设置。当不需要计算参数的梯度时，设置为`null`，以节省内存和计算量。  
- `lr_mult` : float 学习率  
- `wd_mult` : float 权重衰减率  

定义的函数：  
——————————  
- `cast(self, dtype)`  转换参数的值/梯度的数据类型。  
　　dtype : str or numpy.dtype  新的数据类型
- `data(self, ctx=None)` 获取这个参数在设备`ctx`上的值，参数必须已经初始化了。  
　　ctx : 指定设备  
　　Returns：NDArray on ctx  
- `grad(self, ctx=None)` 获取这个参数在设备`ctx`上的梯度值。  
　　ctx : 指定设备

- `initialize`(self, init=None, ctx=None,  
　　default_init=<mxnet.initializer.Uniform>,  
　　force_reinit=False) 初始化参数和梯度向量  
　　`init` : Initializer 初始化参数的值  
　　`ctx` : 设备/设备列表, 默认使用:py:meth:`context.current_context()`.  
　　`default_init` : Initializer  当:py:func:`init`和:py:meth:`Parameter.init`都为none时，使用该默认的初始化.  
　　`force_reinit` : bool, default False 当参数已经被初始化，是否再次初始化。  
```python
weight = mx.gluon.Parameter('weight', shape=(2, 2))
weight.initialize(ctx=mx.cpu(0))
weight.data()
#　　[[-0.01068833  0.01729892]
#　　 [ 0.02042518 -0.01618656]]
#　　<NDArray 2x2 @cpu(0)>
weight.grad()
#　　[[ 0.  0.]
#　　 [ 0.  0.]]
#　　<NDArray 2x2 @cpu(0)>
weight.initialize(ctx=[mx.gpu(0), mx.gpu(1)])
weight.data(mx.gpu(0))
#　　[[-0.00873779 -0.02834515]
#　　 [ 0.05484822 -0.06206018]]
#　　<NDArray 2x2 @gpu(0)>
weight.data(mx.gpu(1))
#　　[[-0.00873779 -0.02834515]
#　　 [ 0.05484822 -0.06206018]]
#　　<NDArray 2x2 @gpu(1)>
```
- `list_ctx(self)`  返回参数初始化在那些设备上  
- `list_data(self)` 按照顺序返回所有设备上的参数值 
　　Returns: list of NDArrays  
- `list_grad(self)` 按照顺序返回所有设备上的梯度值  
- `list_row_sparse_data(self, row_id)` 按照顺序返回所有设备上的 `行稀疏`的参数。  
　　row_id: 指定看哪一行的数据  
　　Returns: list of NDArrays  
- `reset_ctx(self, ctx)` 重新设定设备，把参数copy到该设备上  
　　ctx : Context or list of Context, default `context.current_context()`  
- `row_sparse_data(self, row_id)`  
　　row_id: NDArray 指定看哪一行的数据  
　　Returns: NDArray on row_id's context  
- `set_data(self, data)` 在所有设备上，设置该参数的值。  
- `var(self)` 返回一个代表该参数的符号  
- `zero_grad(self)` 将所有设备上的梯度缓存清零  

数据描述:  
——————————   
- `dtype` 参数的数据类型
- `grad_req`  
- `shape` 参数的尺寸  
{{< /note >}}

{{< note title="模型参数-访问" >}}
`ToTensor`：将图像数据从uint8格式变换成32位浮点数格式，并除以255使得所有像素的数值均在0到1之间 </p>
`transform_first函数`：数据集的函数。将`ToTensor`的变换应用在每个数据样本（图像和标签）的第一个元素，即图像之上.
```python
from mxnet import nd
from mxnet.gluon import nn

```
{{< /note >}}


{{< note title="网络设计" >}}

```python
from mxnet import nd
from mxnet.gluon import nn
```

{{< /note >}}

{{< note title="模型初始化" >}}

```python
from mxnet import nd
from mxnet.gluon import nn
```

{{< /note >}}


{{< note title="模型初始化" >}}

```python
from mxnet import nd
from mxnet.gluon import nn
```

{{< /note >}}

