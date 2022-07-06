---
title: NdArray技巧搜集
weight: 210
menu:
  notes:
    name: NdArray技巧搜集
    identifier: notes-mxnet-technic-gather
    parent: notes-mxnet-ndarray
    weight: 20
---

{{< note title="sum/mean等操作 - 保留原维度数" >}}
`keepdims`: 保留原维度数。例如： 
```python
from mxnet import nd
def softmax(X):
    X_exp = X.exp()  # shape = (n, m)
    # shape = (n, 1) 而并不是 (n,)
    partition = X_exp.sum(axis=1, keepdims=True)  
    return X_exp / partition  # 这里应用了广播机制
X = nd.random.normal(shape=(2, 5))
X_prob = softmax(X)

```
{{< /note >}}


{{< note title="B的值作为A的索引 - 取值" >}}

```python
from mxnet import nd
y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = nd.array([0, 2], dtype='int32')
nd.pick(y_hat, y)
# 结果: [0.1, 0.5]

# 应用的实例：交叉熵的实现
def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()
```
{{< /note >}}


{{< note title="样例" >}}

```python
from mxnet import nd
```
{{< /note >}}
