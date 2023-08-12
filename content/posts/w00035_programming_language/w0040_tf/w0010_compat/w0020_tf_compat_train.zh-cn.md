---
title: "模型训练"
date: 2021-09-08T06:00:20+08:00
menu:
  sidebar:
    name: 模型训练
    identifier: tf-compat-train
    parent: tf-compat
    weight: 20
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["TF","训练"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

### 一、tf.layers

`tf.layers`模块在TensorFlow2.0中已经被完全移除了，用`tf.keras.layers`定义层是新的标准。</br>




### 二、tf.losses

`tf.losses`模块包含了经常使用的、能够实现独热编码的损失函数。



### 三、tf.train

#### 1. Optimizer

**TensorFlow提供的优化器**

|优化器|功能|
|:---|:---|
|tf.train.Optimizer                       ||
|tf.train.GradientDescentOptimizer        ||
|tf.train.AdadeltaOptimizer               ||
|tf.train.AdagtadOptimizer                ||
|tf.train.AdagradDAOptimizer              ||
|tf.train.MomentumOptimizer               ||
|tf.train.AdamOptimizer                   ||
|tf.train.FtrlOptimizer                   ||
|tf.train.ProximalGradientDescentOptimizer||
|tf.train.ProximalAdagradOptimizer        ||
|tf.train.RMSProOptimizer                 ||

{{< split 6 6>}}
Optimizer类与其子类的继承关系：
<p align="center"><img src="/datasets/posts/tf/tf_optimizer_summary.jpg" width="100%" height="100%" title="optimizer" alt="optimizer"></p>

---

```python 
def minimize(self,
  loss,   # 损失值， tensor
  # 全局训练步数，随着模型迭代优化自增， variable
  global_step=None, 
  # 待训练模型参数的列表， list
  var_list=None,    
  # 计算梯度和更新参数模型时的并行化程度，可选值GATE_OP,GATE_NONE,GATE_GRAPH
  #   GATE_NONE 无同步，最大化并行执行效率，将梯度计算和模型参数更新完全并行化。
  #   GATE_OP，操作级同步，对于每个操作，分别确保所有梯度在使用前都计算完成。
  #   GATE_GRAPH，图级同步，最小化并行执行效率，确保所有模型参数的梯度计算完成。
  gate_gradients=GATE_OP,
  # 聚集梯度值的方法， Enum
  aggregation_methed=None,  
  # 是否将梯度计算放置到对应操作所在同一个设备，默认否，Boolean
  colocate_gradients_with_ops=False, 
  # 优化器在数据流图中的名称，string
  nmae=None,    
  # 损失值的梯度            
  grad_loss=None)           
```


{{< /split >}}

|属性|功能介绍|
|:---|:---|
|_name|表示优化器的名称|
|_use_locking|表示是否在并发更新模型参数时加锁|
|minimize|最小化损失函数，该方法会依次调用compute_gradients和apply_gradients |
|compute_gradients|计算模型所有参数的梯度值,返回`<梯度，响应参数>`的键值对列表|
|apply_gradients|将梯度值更新到对应的模型参数，优化器的apply_gradients成员方法内部会调用tf.assign，tf.assign_add,tf.assign_sub方法完成模型参数的更新。|


**自定义优化器**
> 分为三步骤：
> 1. 计算梯度：调用compute_gradients方法，依据指定的策略求得梯度值。
> 2. 处理梯度：用户按照自己的需求处理梯度值，例如：进行梯度裁剪和梯度加权
> 3. 应用梯度：调用apply_gradients方法，将处理后的梯度值应用到模型参数，实现模型更新。

```python
def define_optimizer(learning_rate,loss):
	# 定义优化器
	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5)
	# 计算梯度
	grads_and_vars=optimizer.compute_gradients(loss=loss)
	# 处理梯度
	for i,(g,v) in enumerate(grads_and_vars):
		if g is not None:
			grads_and_vars[i]=(tf.clip_by_norm(g,5),v)
	# 应用梯度
  return optimizer.apply_gradients(grads_and_vars)

```

#### 2. Saver

保存模型参数很重要，训练中断后，可以根据保存的参数继续迭代。Saver 是TensorFlow Python API提供的、能够保存当前模型变量的对象，</br>`Saver对象：只能保存变量，不能保存图结构，所以更常用于训练迭代过程，防止中断重启。` </br>`SavedModel对象：可以同时保存图结构和变量，所以Saved Model对象与(将训练好的模型用在生产中)的行为联系更紧密。`

`tf.train.Saver()`


### 四、tf.summary

可以记录数据流图、直方图、标量值、分布、日志图和其他多种数据类型。

|操作|解释|功能|
|:---|:---|:---|
|tf.summary.scalar()|例如：</br>tf.summary.scalar('loss', loss)|记录标量值|
|tf.summary.Filewirter()|可以关联不同的路径，这样可以可视化不同阶段的数据情况||


