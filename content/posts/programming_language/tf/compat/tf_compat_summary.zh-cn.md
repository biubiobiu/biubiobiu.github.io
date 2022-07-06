---
title: "静态图"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: 静态图
    identifier: tf-compat-graph
    parent: tf-compat
    weight: 1
author:
  name: biubiobiu
  image: /images/author/john.png
---


在TensorFlow 2中使用兼容性模块，必须使用`tf.compat.v1`替换`tf`，并且在导入TensorFlow软件包后添加一行`tf.compat.v1.disable_eager_execution()`函数来关闭eager执行模式。
```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```

### 简介

数据流是一种编程模型，被广泛地应用于并行计算中。TF使用数据流图来表示计算中各个运算之间的关系，在数据流图中，`节点`：表示计算单元(即：操作tf.Operation)；`边`：表示被计算单元消费/生产的数据(即：tf.Tensor)。</br>
`数据流图，可以被导出成一个可移植的、编程语言不相关的表示(ProtoBuf)，这种表示可以被其他语言使用，来创建一个图并在会话中使用它。`

```python
def graph_demo():
    a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
    b = tf.constant([[10, 0, 0], [0, 0.5, 0], [0, 0, 2]])
    c = tf.constant([[1, -1, 3]], dtype=tf.float32)

    y = tf.add(tf.matmul(a, b), c, name='result')

    writer = tf.summary.FileWriter(os.path.join(root_dir, 'log/matmul'), tf.get_default_graph())
    writer.close()
    return y

# 在终端启动TensorBoard对图进行可视化
tensorboard --logdir log/matmul
```
> 上例中创建一个数据流图，然后用TensorBoard对这个图进行可视化。
> 1. `tf.summary.FileWriter` 创建了一个tf.summary.SummaryWriter来保存一个图像化表示，这个writer对象创建时，初始化参数包括：a.该图像化表示的存储路径；b.一个tf.Graph对象，可以使用`tf.get_default_graph`函数返回默认图
> 2. `tf.get_default_graph` 函数，返回默认图。

在执行时，调用TF API创建数据流图，这个阶段并没有进行计算。

### 1、图-tf.Graph

TF是一个C++库，我们只是用python来用简单的方式来构造数据流图，python简化了数据流图的描述阶段，因为它无须特意显示定义一个图，而是会默认一个tf.Graph。

{{< alert type="success" >}}
图的定义:  
1. 隐式定义：在我们用tf.*搭建一个图时，TensorFlow总是会定义一个默认的`tf.Graph`，可以通过`tf.get_default_graph`访问。隐式定义限制了TF的表示能力，因为它被限制只能使用一个图。
2. 显式定义：可以显式地定义一个计算图，因此每个应用可以有多个图。这种方式的表现能力更强，但并不常用，因为需要多个图的应用不常见。</br>
TF通过`tf.Graph()`创建一个tf.Graph对象，并通过`as_default`方法创建一个上下文管理器，每个上下文中定义的运算都被放进相应的图中。实际上，`tf.Graph()`对象定义了一个它所包含的`tf.Operation`对象的命名空间。
    ```python
    import tensorflow as tf

    def graph_define():
      g1 = tf.Graph()
      g2 = tf.Graph()

      with g1.as_default():
          a = tf.constant([[1, 2, 3], [3, 4, 6], [7, 8, 9]], dtype=tf.float32)
          b = tf.constant([[9, 0, 0], [0, 1, 0], [0, 0, 0.5]])
          c = tf.constant([[1, -1, 3]], dtype=tf.float32)
          y = tf.add(tf.matmul(a, b), c, name='result')

      with g2.as_default():
          with tf.name_scope('scope_2'):
              x = tf.constant(1, name='x')
              print(x) # Tensor("scope_2/x:0", shape=(), dtype=int32)
          with tf.name_scope('scope_3'):
              x = tf.constant(10, name='x')
              print(x) # Tensor("scope_3/x:0", shape=(), dtype=int32)
          y = tf.constant(12)
          z = x*y

      writer = tf.summary.FileWriter(os.path.join(root_dir, 'log/two_graphs/g1'), g1)
      writer = tf.summary.FileWriter(os.path.join(root_dir, 'log/two_graphs/g2'), g2)
      writer.close()

    ```
图的集合：  
每个tf.Graph，用集合机制 来存储与图结构相关的元数据，一个集合由一个键值唯一标识，其内容是一个对象/运算的列表。使用者通常不需要关注集合是否存在，因为它们是TF为了正确定义一个图所使用的。

图中节点名： 
1. 后缀：图中每个节点的名字都是唯一的，如果有重复，为了避免重复，TF会添加:id形式的后缀。</br>
在定义时如果没有指定节点的name，TF就会用Operation（操作）的名字来命名，输出的tf.Tensor和其相关的tf.Operation名字相同，只是可能会加上后缀。
2. 前缀：可以通过`tf.name_scope`函数定义一个上下文，为该上下文中所有的运算添加命名范围前缀。

图中的计算：
1. 作为一个C++库，TF数据类型是严格的静态类型，这意味着在图定义阶段必须知道每个运算/张量的类型，且参与运算的数据类型必须相同。
2. 可以使用运算符重载，来简化一些常用的数学运算。运算符重载使得图定义更便捷，并且与tf.*的API调用完全等价，只是有一点：不能给运算指定名字。
    ```python
    y = tf.add(tf.matmul(A, x), b, name='result')
    # 等价
    y = A @ x + b
    ```

    |运算符|操作名|运算符|操作名|运算符|操作名|运算符|操作名|
    |:---|:----|:---|:----|:---|:----|:---|:----|
    |`__neg__`|unary -|`__abs__`|abs()|`__invert__`|unary ~|`__add__`|binary +|
    |`__sub__`|binary -|`__mul__`|binary 元素*|`__floordiv__`|binary //|`__truediv__`|binary /|
    |`__mod__`|binary %|`__pow__`|binary **|`__and__`|binary &|`__or__`|binary \||
    |`__xor__`|binary ^|`__le__`|binary <|`__lt__`|binary <=|`__gt__`|binary >|
    |`__ge__`|binary >=|`__matmul__`|binary @|
{{< /alert >}}

### 2、图放置-tf.device

`tf.device`创建一个和设备相符的上下文管理器，这个函数运行使用者将同一个上下文下的所有运算放置在相同的设备上。tf.device指定的设备不仅仅是物理设备，它能指定远程服务器、远程设备、远程工作者、不同种类的物理设备(GPU、CPU、TPU)。</br>
> 格式：/job:<JOB_NAME>/task:<TASK_INDEX>/device:<DEVICE_TYPE>:<DEVICE_INDEX>
> 1. <JOB_NAME>：是一个由字母和数字构成的字符串，首个字符不能是数字
> 2. <TASK_INDEX>：是一个非负整数，代表在名为<JOB_NAME>的工作中的任务编号
> 3. <DEVICE_TYPE>：是一个已经注册过的设备类型(CPU或者GPU)
> 4. <DEVICE_INDEX>：是一个非负整数，代表设备的索引号

```python
def device_demo():
    with tf.device('/CPU:0'):
        a = tf.constant([[1, 2, 3], [3, 4, 6], [7, 8, 9]], dtype=tf.float32)
        b = tf.constant([[9, 0, 0], [0, 1, 0], [0, 0, 0.5]])
        c = tf.constant([[1, -1, 3]], dtype=tf.float32)
    with tf.device('/GPU:0'):
        mul = tf.matmul(a, b, name='mul_result')

    y = tf.add(mul, c, name='add_result')

    writer = tf.summary.FileWriter(os.path.join(root_dir, 'log/device'), tf.get_default_graph())
    writer.close()

```

### 3、图执行-tf.Session

静态图，图的定义与执行完全分离，在eager执行模式中不是这样。`tf.Session`：是一个TF提供的类，用来表示Python程序与C++运算库之间的联系，是唯一能直接与硬件通信、将运算放置到指定的设备上、使用本地和分布式TF运行库的类。它的主要目的：根据定义的图，具体地实现各个计算。</br>
`tf.Session`对象是高度优化过的，一旦被正确构建，它会将`tf.Graph`缓存起来以加速其执行，`tf.Session`作为物理资源的拥有者，必须以一个文件描述符的方式来做下面的工作：
1. 通过创建tf.Session来获取资源（等价于open操作系统调用）
2. 使用这些资源（等价于在文件描述符上使用 读/写 操作）
3. 使用tf.Session.close释放资源（通常会使用一个上下文管理器，不需要手动销毁释放资源）

#### 1). `tf.Session`的三个参数
Session(target='', graph=None, config=None)
1. `target`：配置执行引擎
    常见的场景：
    1. 使用当前的本地的硬件来执行图
        ```python
        with tf.Session() as sess:
            # 使用session去执行 某些操作
            sess.run(...)
        ```
    
    2. 一些更复杂的场景：使用一个远程TensorFlow服务器，可以通过使用服务器的url(grpc://)来指定`tf.Session的target参数`
        ```python
        # TensorFlow服务器的 ip和port
        ip = '192.168.1.90'
        port = '9877'
        with tf.Session(target=f'grpc://{ip}:{port}') as sess:
            sess.run(...)
        ```
2. `graph`: 指定需要使用的图。tf.Session会使用默认的图对象，在需要运算多个图时，可以指定需要使用的图。tf.Session对象每次只能处理一个图。

3. `config`: 硬件/网络配置，这个配置通过`tf.ConfigProto`对象来指定，用来控制Session的行为。`tf.ConfigProto`比较复杂，选项也比较多，最常用的选项有下面两个：
    1. `allow_soft_placement`：当为True时：启动软设备安排，即：不是所有运算都会按照图定义的那样，被安排在指定的设备上。这是为了防止这种情况：比如GPU不存在，或者原来存在现在出了些问题，TensorFlow没有检测到该设备，就可以把指定给这个设备的运算，安排到其他正确的设备上。
    2. `gpu_options.allow_growth`：当为True时：会改变GPU显存分配器的工作方式。分配器默认的工作方式：tf.Session被创建时就会分配所有可用的GPU显存。当allow_growth=True时，分配器会以`逐步递增的方式`分配显存。这是为了适应这种情况：在研究环境下，GPU资源是共享的，当一个tf.Session执行时，不能占用所有资源，其他人也还在使用。
    3. `per_process_gpu_memory_fraction`：手动限定显存的使用量
    4. `log_device_placement`：当为True时，会获取Operations和Tensor被指派到的设备号，在终端会打印出各个操作是在那些设备上运行的。

    ```python
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth=True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4  #占用40%显存
        with tf.Session(config=config) as sess:
            # 使用session去执行 某些操作
            sess.run(...)

    ```
#### 2). sess.run()
`sess.run(y)`的工作方式如下：
1. y是一个运算的输出节点，回溯y的输入
2. 递归的回溯所有节点，直到无法找到父节点
3. 评估输入
4. 跟踪依赖图：分析各个节点的关系
5. 执行计算

`feed_dict`：可以把外部的数据，注入计算图中，相当于重写计算图里的某个值。跟`tf.placeholder`配合使用，完成外部的数据流入计算图。
`tf.placeholder`：重写运算符。其创建的目的就是：当外面的值没有注入图中时，就会抛出一个错误。


```python
def session_demo():
    a = tf.constant([[1, 2, 3], [3, 4, 6], [7, 8, 9]], dtype=tf.float32)
    b = tf.constant([[9, 0, 0], [0, 1, 0], [0, 0, 0.5]])
    c = tf.constant([[1, -1, 3]], dtype=tf.float32)
    y = tf.add(tf.matmul(a, b), c, name='result')

    with tf.Session() as sess:
        a_value, b_value, c_value = sess.run([a, b, c])
        y_value = sess.run(y)

        # 重写
        y_new = sess.run(y, feed_dict={c: np.zeros((1, 3))})

    print(f'a: {a_value}\nb: {b_value}\nc: {c_value}\ny: {y_value}')
    print(f'y_new: {y_new}')
```

### 4、图中的变量

一个变量是一个tf.Variable对象，用于维护图的状态，作为图中其他节点的输入。`tf.Tensor`和`tf.Variable`对象可以用相同的方式使用，不过`tf.Variable`拥有更多的属性：
1. 一个变量必须要被初始化
2. 一个变量默认被加到`全局变量`和`可训练变量`集合中

#### 1. 变量声明
声明变量的两种方式：需要(type, shape)
1. `tf.Variable`：是一个类，创建一个变量，同时它需要指定一个初始值。  
变量的赋值，可以使用assign函数：比如：`w.assign(w+0.1)`等价于`w.assign_add(0.1)`。其实，变量的初始化操作，就是把初始值assign给每个变量。
    > tf.Variable是一个类  
    >`__init__`(
    initial_value=None,  
    trainable=True, # 是否可训练  
    collections=None,  
    validate_shape=True,  
    caching_device=None,  
    name=None,  
    variable_def=None,  
    dtype=None,  
    expected_shape=None,  
    import_scope=None,  
    constraint=None)

    ```python
    size_in = 100
    size_out = 100
    # w的初始值是有tf.truncated_normal运算产生，服从正太分布N(0, 0.1)
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name='w')
    # b的初始值是有tf.constant运算产生的常量来初始化
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name='b')

    with tf.Session() as sess:
        # 变量的初始化
        sess.run(w.initializer)

    ```

2. `tf.get_variable`：更复杂，但拥有更强的表现能力。例如：如果我们需要`变量共享`，就不能使用tf.Variable定义，只能使用tf.get_variable。tf.get_variable和tf.variable_scope一起使用，通过它的reuse参数，实现tf.get_variable的变量共享能力。其中，tf.get_variable不受tf.name_scope的影响。</br>
TensorFlow提供的`tf.layers`模块，包含了几乎所有常用的层，这些层内部都是用`tf.get_variable`来定义的，因此，这些层可以和`tf.variable_scope`一起使用来共享它们的变量。
    > tf.get_variable是一个函数：  
    > (  
        name,  
        shape=None,  
        dtype=None,  
        initializer=None,  
        regularizer=None,  
        trainable=True,  
        collections=None,  
        caching_device=None,  
        partitioner=None,  
        validate_shape=None,  
        use_resource=None,  
        custom_getter=None,  
        constraint=None  
    )

    ```python
    with tf.variable_scope('scope'):
        a = tf.get_variable('v', [1])

    with tf.variable_scope('scope', reuse=True):
        b = tf.get_variable('v', [1])

    print(a.name, b.name) # scope/v:0 scope/v:0
    ```

#### 2. 变量初始化

TensorFlow变量随机初始化，例如：`w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name='w')`</br>
常见的随机函数：
|操作|功能|
|:---|:---|
|`tf.random_normal()`|正态分布，参数:(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)|
|`tf.truncated_normal`|正态分布，但如果随机出来的值偏离平均值超过了2个标准差，那么这个数将会被重新随机, 参数如上|
|`tf.random_uniform`|平均分布，参数：([m, n], 最小值, 最大值, 取值类型)|
|`tf.random.gamma`|Gamma分布，参数：([m, n], 形状参数 $\alpha$，尺度参数 $\beta$, 取数类型)|
|`常数函数`||
|`tf.zeros()`|参数：(shape, dtype=tf.float32, name=None)， shape的格式: [m, n]|
|`tf.ones()`|参数：(shape, dtype=tf.float32, name=None), shape的格式: [m, n]|
|`tf.fill()`|参数：(shape, value, name=None), shape的格式: [m, n]|
|`tf.constant()`|参数：(value, dtype=None, shape=None, name='Const', verify_shape=False)</br>例如：tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]</br>tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.],[-1. -1. -1.]]|
|`tf.range()`|参数：tf.range(start, limit, delta)|
|`tf.linspace()`|参数：(start, stop, num)  功能： (stop - start)/(num - 1)|

1. 传入初始值  
    > 在session中执行时，变量必须要初始化：  
    > a. 全部变量初始化： `tf.global_variables_initializers():其实内部实现：=tf.variabels_initializer(tf.global_variables())`  
    > b. 部分变量初始化：`tf.variables_initializer([变量])`  
    > c. 检查变量是否初始化成功：`tf.is_variable_initialized`：检查变量是否初始化；`tf.report_uninitialized_variables`：获取未初始化的变量集合；`tf.assert_variables_initialized`：断言变量已经初始化。

    ```python
    with tf.Session() as sess:
        sess.run(tf.global_variable_initializer()) # 初始化所有变量
    ```


2. 从checkpoint文件中恢复变量的值  
当我们创建Saver实例时，它的构造方法会向当前的数据流图中添加一对操作：SaveOp和RestoreOp
    - SaveOp负责向checkpoint文件中写入变量  

        ```python
        saver = tf.train.Saver()
        saver.save(sess, '/tmp/summary/test.ckpt')
        ```

    - RestoreOp负责从checkpoint文件中恢复变量

        ```python
        saver = tf.train.Saver()
        saver.restore(sess, '/tmp/summary/test.ckpt')
        ```


#### 3. 变量的访问

1. 通过在`tf.global_variable()`变量表中，根据变量名进行匹配查找
```python
x = tf.Variable(1,name='x')
y = tf.get_variable(name='y',shape=[1,2])
for var in tf.global_variables():              #返回全部变量列表
    if var.name == 'x:0':
        print(var)
```

2. 利用`tf.get_tensor_by_name`，在图中根据name查找
```python
import tensorflow as tf

x = tf.Variable(1,name='x')
y = tf.get_variable(name='y',shape=[1,2])

graph = tf.get_default_graph()
x1 = graph.get_tensor_by_name("x:0")
y1 = graph.get_tensor_by_name("y:0")

```
