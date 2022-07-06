---
title: "简介"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: 简介
    identifier: torch-summary
    parent: torch
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
mermaid: true
enableEmoji: true
---

<a href="https://pytorch.org/docs/stable/index.html" target="blank">官方文档</a>  

{{< split 6 6>}}

torch目录下，树状图:  
├── quasirandom.py  
├── random.py `random模块`  
├── serialization.py  
├── storage.py  
├── tensor.py `Tensor模块`  
├── functional.py  
│  
├── cuda  
│　　├── comm.py  
│　　├── error.py  
│　　├── memory.py  
│　　├── nccl.py  
│　　├── nvtx.py  
│　　├── profiler.py  
│　　├── random.py  
│　　├── sparse.py  
│　　└── streams.py  
│  
├── nn  
│　　├── backends  
│　　├── cpp.py  
│　　├── functional.py  
│　　├── grad.py  
│　　├── init.py  
│　　├── intrinsic  
│　　│　　├── modules  
│　　│　　│　　└── fused.py  
│　　│　　├── qat  
│　　│　　│　　└── modules  
│　　│　　│　　　　├── conv_fused.py  
│　　│　　│　　　　└── linear_relu.py  
│　　│　　└── quantized  
│　　│　　　　└── modules  
│　　│　　　　├── conv_relu.py  
│　　│　　　　└── linear_relu.py  
│　　├── modules  
│　　│　　├── activation.py  
│　　│　　├── adaptive.py  
│　　│　　├── container.py  
│　　│　　├── conv.py  
│　　│　　├── distance.py  
│　　│　　├── dropout.py  
│　　│　　├── flatten.py  
│　　│　　├── fold.py  
│　　│　　├── instancenorm.py  
│　　│　　├── linear.py  
│　　│　　├── loss.py  
│　　│　　├── module.py  
│　　│　　├── normalization.py  
│　　│　　├── padding.py  
│　　│　　├── pooling.py  
│　　│　　├── rnn.py  
│　　│　　├── sparse.py  
│　　│　　├── transformer.py  
│　　│　　├── upsampling.py  
│　　│　　└── utils.py  
│　　├── parallel  
│　　│　　├── data_parallel.py  
│　　│　　├── distributed.py  
│　　│　　├── parallel_apply.py  
│　　│　　├── replicate.py  
│　　├── parameter.py  
│　　├── qat  
│　　│　　└── modules  
│　　│　　　　├── conv.py  
│　　│　　　　└── linear.py  
│　　├── quantized  
│　　│　　├── dynamic  
│　　│　　│　　└── modules  
│　　│　　│　　　　├── linear.py  
│　　│　　│　　　　└── rnn.py  
│　　│　　├── functional.py  
│　　│　　└── modules  
│　　│　　　　├── activation.py  
│　　│　　　　├── conv.py  
│　　│　　　　├── functional_modules.py  
│　　│　　　　├── linear.py  
│　　│　　　　└── utils.py  
│　　└── utils  
│　　　　├── clip_grad.py  
│　　　　├── convert_parameters.py  
│　　　　├── fusion.py  
│　　　　├── prune.py  
│　　　　├── rnn.py  
│　　　　└── spectral_norm.py  
│  
├── optim  
│　　├── adadelta.py  
│　　├── adagrad.py  
│　　├── adam.py  
│　　├── adamax.py  
│　　├── adamw.py  
│　　├── asgd.py  
│　　├── lbfgs.py  
│　　├── optimizer.py  
│　　├── rmsprop.py  
│　　├── rprop.py  
│　　├── sgd.py  
│　　└── sparse_adam.py  
│  
├── autograd  
│　　├── anomaly_mode.py  
│　　├── function.py  
│　　├── grad_mode.py  
│　　├── profiler.py  
│　　└── variable.py  
│  
├── distributed  
│　　├── autograd  
│　　├── distributed_c10d.py  
│　　├── optim  
│　　│　　└── optimizer.py  
│　　├── rendezvous.py  
│　　└── rpc  
│　　　　├── api.py  
│　　　　├── backend_registry.py  
│　　　　├── constants.py  
│　　　　└── internal.py  
│  
├── distributions  
│　　├── bernoulli.py  
│　　├── beta.py  
│　　├── binomial.py  
│　　├── categorical.py  
│　　├── constraint_registry.py  
│　　├── constraints.py  
│　　├── distribution.py  
│　　├── exp_family.py  
│　　├── exponential.py  
│　　├── gamma.py  
│　　├── geometric.py  
│　　├── gumbel.py  
│　　├── independent.py  
│　　├── kl.py  
│　　├── laplace.py  
│　　├── log_normal.py  
│　　├── logistic_normal.py  
│　　├── lowrank_multivariate_normal.py  
│　　├── multinomial.py  
│　　├── multivariate_normal.py  
│　　├── negative_binomial.py  
│　　├── normal.py  
│　　├── pareto.py  
│　　├── poisson.py  
│　　├── relaxed_bernoulli.py  
│　　├── relaxed_categorical.py  
│　　├── studentT.py  
│　　├── transformed_distribution.py  
│　　├── transforms.py  
│　　├── uniform.py  
│　　├── utils.py  
│　　└── weibull.py  
│  
├── jit  
│　　├── annotations.py  
│　　├── frontend.py  
│　　├── quantized.py  
│　　└── supported_ops.py  
│  
├── multiprocessing  
│　　├── pool.py  
│　　├── queue.py  
│　　├── reductions.py  
│　　└── spawn.py  
│  
├── quantization  
│　　├── default_mappings.py  
│　　├── fake_quantize.py  
│　　├── fuse_modules.py  
│　　├── observer.py  
│　　├── qconfig.py  
│　　├── quantize.py  
│　　└── stubs.py  
│  
├── onnx  
│　　├── operators.py  
│　　├── symbolic_caffe2.py  
│　　├── symbolic_opset10.py  
│　　├── symbolic_opset11.py  
│　　├── symbolic_opset7.py  
│　　├── symbolic_opset8.py  
│　　├── symbolic_opset9.py  
│　　├── symbolic_registry.py  
│　　└── utils.py  
│  
├── `utils: 辅助模块`  
│　　├── backcompat  
│　　├── bottleneck  
│　　├── collect_env.py  
│　　├── cpp_extension.py  
│　　├── data  
│　　│　　│　　├── collate.py  
│　　│　　│　　├── pin_memory.py  
│　　│　　│　　└── worker.py  
│　　│　　├── dataloader.py  
│　　│　　├── dataset.py  
│　　│　　├── distributed.py  
│　　│　　├── sampler.py  
│　　├── dlpack.py  
│　　├── file_baton.py  
│　　│　　└── constants.py  
│　　├── mkldnn.py  
│　　├── model_zoo.py  
│　　├── tensorboard  
│　　│　　├── summary.py  
│　　│　　└── writer.py  
│  
└── version.py  

---


PyTorch主要包括一下16个模块：  
1. `torch模块`  
    + torch本身包含了PyTorch经常使用的激活函数：torch.sigmoid, torch.relu, torch.tanh  
    + 一些张量操作：torch.mm()(矩阵的乘法), torch.select()(张量元素的选择)等操作
    + 生成张量：torch.zeros(), torch.randn()等操作。  

2. `torch.Tensor模块`  
torch.Tensor模块：定义了torch中的张量类型。`张量`：一定维度的矩阵。
   - 张量类中包含着一些列的方法，返回新的张量或者更改当前的张量：根据PyTorch的命名规则，如果张量方法后缀带有下划线，该方法会修改张量本身的数据；反之则会返回新的张量。例如：Torch.add方法：返回新的张量；Torch.add_方法：修改当前张量的值。  
   - torch.Storage负债torch.Tensor底层的数据存储，即：为一个张量分配连续的一维内存地址。  

3. `torch.sparse模块`  
torch.sparse模块：定义了稀疏张量，其中构造的稀疏张量采用的是COO格式(Coordinate)，用一个长整形定义非零元素的位置，用浮点数张量定义对应非零元素的值。稀疏张量之间可以做元素的算术运算和矩阵运算。  

4. `torch.cuda模块`  
torch.cuda模块：定义了与CUDA运算相关的一些列函数，包括：检测系统的CUDA是否可用、当前进程对应的GPU序号、清除GPU上的缓存、设置GPU的计算流、同步GPU上执行的所有核函数等。  

5. `torch.nn模块`  
torch.nn模块：是神经网络模块化的核心模块，该模块定义了一些神经网络的计算模块：nn.ConvNd(卷积层，其中N=1,2,3)、nn.Linear(全连接层)等。构建深度学习模型的时候，可以通过继承nn.Module类并重写forward方法来实现一个新的神经网络。  
   - torch.nn.functional模块：定义一些和神经网络相关的函数，包括卷积函数和池化函数等，这些函数也是深度学习网络构建的基础。需要指出的是：torch.nn中定义的模块一般会调用torch.nn.functional里的函数，比如：nn.ConvNd模块会调用torch.nn.functional.convNd函数。另外，torch.nn.functional里面还定义了一些不常用的激活函数：torch.nn.functional.relu6、torch.nn.functional.elu等。  
   - torch.nn.init模块：定义了神经网络权重的初始化。  

6. `torch.optim模块`  
   - 定义了一系列的优化器。比如：torch.optim.SGD(随机梯度下降法)、torch.optim.Adagrad、torch.optim.RMSprop、torch.optim.Adam等。  
   - 定义了一些学习率衰减的算法的子模块：torch.optim.lr_scheduler，这个子模块中包含了：torch.optim.lr_scheduler.StepLR(学习率阶梯下降算法)、torch.optim.lr_scheduler.CosineAnnealingLR(余弦退火算法)等学习率衰减算法。  

7. `torch.autograd模块`  
torch.autograd模块：是PyTorch的自动微分模块，定义了一系列的自动微分函数，包括torch.autograd.backward函数，主要用于：在求得损失函数之后进行反向梯度传播。torch.autograd.grad函数：用于一个标量张量对一个另一个张量求导(在代码中设置不参与求导的部分参数)。另外，这个模块还内置了数值梯度功能和检查自动微分引擎是否输出正确结果的功能。  

8. `torch.distribute模块`  
torch.distributed模块：是PyTorch的分布式计算模块，主要功能是提供PyTorch并行运行环境，其主要支持的后端有MPI、Gloo、NCCL三种。PyTorch分布式工作原理：启动多个并行的进程，每个进程(都拥有一个模型的备份，然后输入不同的训练数据)，计算损失函数，每个进程独立地做反向 传播，最后对所有进行权重张量的梯度做归约(Reduce)。用到后端的部分主要是：数据的广播(Broadcast)和数据的收集(Gather)  
   - Broadcast：把数据从一个节点(进程)传播到另一个节点(进程)，比如：归约后梯度张量的传播  
   - Gather：把数据从其他节点(进程)转移到当前节点(进程)，比如：把梯度张量从其他节点转移到某个特定的节点，然后求梯度平均。  
PyTorch的分布式计算模块不但踢动了后端的一个包装，还提供了一些启动方式来启动多个进程，包括：通过网络(TCP)方式、通过环境变量方法、通过共享文件方式等。  

9. `torch.distributions模块`  
torch.distributions模块：提供了一系列类，使得PyTorch能够对不同的分布进行采样，并生成概率采样过程的计算图。在一些应用过程中，比如强化学习(Reinforcement Learning)，经常会使用一个深度学习模型来模拟在不同环境条件下采取的策略，其最后的输出是不同动作的概率。当深度学习模型输出概率之后，需要根据概率对策略进行采样来模拟当前的策略概率分布，最后用梯度下降法来让最优策略的概率最大(策略梯度算法PolicyGradient)。实际上，因为采样的输出结果是离散的，无法直接求导，所以不能使用反向传播的方法来优化网络。torch.distributions模块的存在就是为了解决这个问题。可以结合torch.distributions.Categorical进行采样，然后使用对数求导来规避这个问题。当然，除了服从多项式分布的torch.distributions.Categorical类，PyTorch还支持其他的分布(包括连续分布和离散分布)，比如torch.distributions.Normal类支持连续的正态分布的采样，可以用于连续的强化学习的策略。  

10. `torch.hub模块`  
torch.hub模块：提供了一系列预训练的模型，比如：torch.hub.list函数可以获取某个模型镜像站点的模型信息。通过torch.hub.load来加载预训练模型，载入后的模型可以保存到本地，并可以看到这些模型对应类支持的方法。  

11. `torch.jit模块`  
torch.jit模块：是PyTorch的即时编译器，这个模块存在的意义是把PyTorch的动态图换成可以优化和序列化的静态图，工作原理：通过输入预先定义好的张量，追踪整个动态图的构建过程，得到最终构建出来的动态图，然后转换为静态图(通过中间表示：IntermediateRepresentation，来描述最后得到的图)。通过JIT得到的静态图可以被保存，并且被PyTorch其他的前端支持。另外，JIT可以用来生成其他格式的神经网络描述文件，比如ONNX。torch.jit支持两种模式，即：脚本模式(ScriptModule)和追踪模式(Tracing)，这两个都能构建静态图，区别在于脚本模式支持控制流，追踪模式不支持，不过前者支持的神经网络模块比后者少。  

12. `torch.multiprocessing模块`  
torch.multiprocessing模块：定义PyTorch中的多进程API。通过这个模块可以启动不同的进程，每个进程运行不同的深度学习模型，并且能够在进程间共享张量，共享的张量可以在CPU上，也可在GPU上。多进程API还提供了与Python原生的多进程API相同的一系列函数，包括锁(Lock)和队列(Queue)等。  

13. `torch.random模块`  
torch.random模块：提供了一系列的方法来保存和设置随机数生成器的状态。因为神经网络的训练是一个随机过程，包括数据的输入、权重的初始化都具有一定的随机性。设置一个统一的随机种子可以有效地帮助我们测试不同结构神经网络的表现，有助于调试神经网络的结构。  
    - get_rng_state函数获取当前随机数生成器状态
    - set_rng_state函数：设置当前随机数生成器状态
    - manual_seed函数：设置随机种子
    - initial_seed函数：得到程序初始的随机种子。  

14. `torch.onnx模块`  
torch.onnx模块：定义了PyTorch导出和载入ONNX格式的深度学习模型描述文件。ONNX格式的存在：为了方便不同深度学习框架之间交换模型。引入这个模块可以方便PyTorch导出模型给其他深度学习架构使用，或者让PyTorch可以载入其他深度学习框架构建的深度学习模型。  

15. `PyTorch的辅助模块`  
    - torch.utils.bottleneck模块：可以用来检测深度学习模型中模块的运行时间，从而可以找到导致性能瓶颈的那些模块，通过优化这些模块的运行时间，优化整个深度学习模型的性能。  
    - torch.utils.checkpoint模块：可以用来节约深度学习使用的内存。因为梯度反向传播，在构建计算图的时候需要保存中间数据，而这些数据大大增加了深度学习的内存消耗。为了减少内存消耗，让迷你批次的大小得到提高，从而提升深度学习模型的性能和优化时的稳定性，可以通过这个模块记录中间数据的计算过程，然后丢掉这些中间数据，等需要用到的时候再从新计算这些数据，这个模块设计的核心思想是以计算时间换存储空间。  
    - torch.utils.cpp_extension模块：定义了PyTorch的C++扩展。 
    - torch.utils.data模块：引入了数据集和数据载入器的概念，前者代表包含了所有数据的数据集，通过索引能够得到某条特定的数据，后者通过对数据集的包装，可以对数据集进行随机排列和采样，得到一些列打乱顺序的批次。  
    - torch.utils.dlpacl模块：定义了PyTorch张量和DLPack张量存储格式之间的转换，用于不同框架之间张量数据的交换。  
    - torch.utils.tensorboard模块：是PyTorch对TensorBoard数据可视化工具的支持。TensorBoard原来是TF自带的数据可视化工具，能够显示深度学习模型在训练过程中损失函数、张量权重的直方图，以及模型训练过程中输出的文本、图像、视频等。TensorBoard的功能非常强大，而且是基于可交互的动态网页设计的，使用者可以通过预先提供的一系列功能来输出特定的训练过程细节。PyTorch支持TensorBoard可视化后，在训练过程中，可以方便地观察中间输出的张量，可以方便地调试深度学习模型。  

{{< /split >}}
