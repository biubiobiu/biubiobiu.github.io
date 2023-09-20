---
title: "数学计算"
date: 2022-04-08T06:00:20+06:00
menu:
  sidebar:
    name: 数学计算
    identifier: torch-mathematical
    parent: torch
    weight: 30
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["torch","数学计算"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

## 一、数学计算

1. [x] <font color=#a020f0>torch.abs(input)</font>。 数学---绝对值
2. [x] <font color=#a020f0>torch.add(input, value)</font>。数学---对张量的每个元素加value值
3. [x] <font color=#a020f0>torch.div(input, value)</font>。数学---逐元素除法，将input逐元素除以标量value
4. [x] <font color=#a020f0>torch.div(input, other)</font>。数学---逐元素除法。<br>
两个张量input和other逐元素相除.这两个维度可以不同，但元素数量一定要一致。输出: 与input维度一致
5. [x] <font color=#a020f0>torch.mul(input, value)</font>。数学---逐元素乘法
6. [x] <font color=#a020f0>torch.mul(input, other)</font>。数学---逐元素乘法
7. [x] <font color=#a020f0>torch.fmod(inpur, divisor, out)</font>。数学---取余
8. [x] <font color=#a020f0>torch.remainder(input, divisor, out)</font>。数学---取余 相当于 %。<br>
divisor: 标量或者张量 逐元素
9. [x] <font color=#a020f0>torch.addcdiv(tensor, value=1, tensor1, tensor2, out=None)</font>。数学--- 像素点相除后相加。<br>
out = tensor .+ value*(tensor1./tensor2)
10. [x] <font color=#a020f0>torch.addcmul(tensor, value=1, tensor1, tensor2, out=None)</font>。数学--- 像素点相乘后相加。<br>
out = tensor .+ value*(tensor1 .* tensor2)
11. [x] <font color=#a020f0>torch.neg(input)</font>。数学---取负。out = -1 * input。
12. [x] <font color=#a020f0>torch.reciprocal(input)</font>。数学---倒数。out = 1.0 / input。
13. [x] <font color=#a020f0>torch.sign(input)</font>。数学---取正负符号
14. [x] <font color=#a020f0>torch.sin(Tensor)</font>。数学---正弦
15. [x] <font color=#a020f0>torch.cos(Tensor)</font>。数学---余弦
16. [x] <font color=#a020f0>torch.tan(Tensor)</font>。数学---正切
17. [x] <font color=#a020f0>torch.sinh(Tensor)</font>。数学---双曲正弦
18. [x] <font color=#a020f0>torch.cosh(Tensor)</font>。数学---双曲余弦
19. [x] <font color=#a020f0>torch.tanh(Tensor)</font>。数学---双曲正切
20. [x] <font color=#a020f0>torch.asin(Tensor)</font>。数学---反正弦
21. [x] <font color=#a020f0>torch.acos(input)</font>。数学---反余弦
22. [x] <font color=#a020f0>torch.atan(Tensor)</font>。数学---反正切
23. [x] <font color=#a020f0>torch.atan2(input1, input2, out=None)</font>。数学---
24. [x] <font color=#a020f0>torch.ceil(input)</font>。数学---向上取整
25. [x] <font color=#a020f0>torch.floor(input)</font>。数学---向下取整
26. [x] <font color=#a020f0>torch.round(input, out)</font>。数学---四舍五入
27. [x] <font color=#a020f0>torch.clamp(input, min, max, out=None)</font>。数学---销掉最小最大。将input张量每个元素，夹在[min,max]之间
28. [x] <font color=#a020f0>torch.exp(tensor)</font>。数学---指数
29. [x] <font color=#a020f0>torch.pow(input, exponent, out)</font>。数学---逐元素求exponent次幂
30. [x] <font color=#a020f0>torch.rsqrt(tensor)</font>。数学---平方根倒数。out = 1.0 / input^0.5
31. [x] <font color=#a020f0>torch.frac(tensor)</font>。数学---返回逐元素的小数部分
32. [x] <font color=#a020f0>torch.sqrt(tensor)</font>。数学---平方根。out = input^0.5
33. [x] <font color=#a020f0>torch.lerp(start, end, weight, out)</font>。数学---线性插值。out = start + weight(end-start)
34. [x] <font color=#a020f0>torch.log(tensor, out=None)</font>。数学---自然对数。out = log(input)
35. [x] <font color=#a020f0>torch.loglp(tensor, out=None)</font>。数学---input+1的自然对数。out = log(input+1)
36. [x] <font color=#a020f0>torch.sigmoid(input)</font>。数学---sigmoid
37. [x] <font color=#a020f0>torch.cumsum(input, dim, out)</font>。数学---累加。沿指定维度的累加和
38. [x] <font color=#a020f0>torch.cumprod(input, dim, out)</font>。数学---累积。沿指定维度累积
39. [x] <font color=#a020f0>torch.dist(input, other, p=2, out)</font>。数学---两个Tensor之间的范数
40. [x] <font color=#a020f0>torch.norm(input, p=2, dim, out=None)</font>。数学---单个Tensor的范数。返回输入张量的p的范数
41. [x] <font color=#a020f0>torch.mean(input, dim, out=None)</font>。数学---均值
42. [x] <font color=#a020f0>torch.std(input, dim, out=None)</font>。数学---标准差。返回张量在指定维度上的标准差
43. [x] <font color=#a020f0>torch.var(input, dim, out=None)</font>。数学---方差
44. [x] <font color=#a020f0>torch.sum(input, dim, out=None)</font>。数学---和
45. [x] <font color=#a020f0>torch.median(input, dim=-1, values=None, indices=None)</font>。数学---中位数
46. [x] <font color=#a020f0>torch.mode(input, dim=-1, values=None, indices=None)</font>。数学---众数
47. [x] <font color=#a020f0>torch.prod(input, dim, out=None)</font>。数学---所有元素的积。输出张量在指定维度上所有元素的积
48. [x] <font color=#a020f0>torch.cross(input, other, dim=-1, out=None)</font>。数学---叉积。输出两个张量的向量积,dim维上size必须为3


## 二、逻辑计算

1. [x] <font color=#a020f0>torch.eq(input, other, out=None)</font>。比较--- 等于 像素级
2. [x] <font color=#a020f0>torch.equal(tensor1, tensor2)</font>。比较--- Tensor，是否具有相同的形状和元素值
3. [x] <font color=#a020f0>torch.ge(input, other, out=None)</font>。比较--- 大于等于 像素级
4. [x] <font color=#a020f0>torch.gt(input, other, out=None)</font>。比较--- 大于 像素级
5. [x] <font color=#a020f0>torch.le(input, other, out=None)</font>。比较--- 小于等于 像素级
6. [x] <font color=#a020f0>torch.lt(input, other, out=None)</font>。比较--- 小于 像素级
7. [x] <font color=#a020f0>torch.ne(input, other, out=None)</font>。比较--- 不等于 像素级
8. [x] <font color=#a020f0>torch.max(input, dim, max=None, max_indices=None)</font>。比较--- 取最大值。在指定维度上取最大值
9. [x] <font color=#a020f0>torch.min(input, dim, min=None, min_indices=None)</font>。比较--- 取最小值。在指定维度上取最小值
10. [x] <font color=#a020f0>torch.kthvalue(input, k, dim=None, out=None)</font>。比较--- 取第k个最小值。取张量在指定维度上第k个最小值，默认最后一维
11. [x] <font color=#a020f0>torch.sort(input, dim=None, descending=False, out=None)</font>。比较--- 排序
12. [x] <font color=#a020f0>torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)</font>。比较--- 取第k个最大值

## 三、复数域
1. [x] <font color=#a020f0>torch.view_as_complex()</font>。将实数 [a, b] 转为复数域 [a, bj]。复数域
2. [x] <font color=#a020f0>torch.view_as_real()</font>。将复数 [a, bj] 转为实数域 [a, b]
3. [x] <font color=#a020f0>torch.flatten(input, start_dim=0, end_dim=1)</font>。默认将张量拉成一维的向量

## 四、矩阵操作

1. [x] <font color=#a020f0>torch.matmul(input, other, out)</font>。矩阵--- 矩阵相乘
2. [x] <font color=#a020f0>torch.mm(input, other, out)</font>。矩阵--- 矩阵相乘
3. [x] <font color=#a020f0>toch.mv(mat, vec, out=None)</font>。矩阵*向量。矩阵--- 矩阵*向量
4. [x] <font color=#a020f0>torch.addbmm(beta=1, mat, alpha=1, batch1, batch2, out=None)</font>。矩阵--- batch 相乘后相加。<br>
batch1:  $ b·n·m $ ;  batch2: $b·m·p$。<br>
mat: $n·p$; out: $n·p$;。<br>
res = $beta·mat + alpha·sum(batch1_i·batch2_i),i\in[0~b]$。<br>
out: batch个2维矩阵相乘后,再相加
5. [x] <font color=#a020f0>torch.addmm(beta=1, mat, alpha=1, mat1, mat2, out=None)</font>。矩阵--- 单个矩阵 相乘后相加。<br>
$ out=beta·mat + alpha·mat_1·mat_2 $
6. [x] <font color=#a020f0>torch.addmv(beta=1, tensor, alpha=1, mat, vec, out=None)</font>。矩阵--- 单个矩阵 矩阵*向量后相加。<br>
vec: 向量。mat: 矩阵。$out=beta·tensor + alpha·(mat·vec)$
7. [x] <font color=#a020f0>torch.addr(beta=1, mat, alpha=1, vec1, vec2, out)</font>。矩阵--- 向量*向量后相加。<br>
$out=beta·mat + alpha·(vec_1·vec_2)$
8. [x] <font color=#a020f0>torch.baddbmm(beta=1, mat, alpha=1, batch1, batch2, out=None)</font>。矩阵--- batch 单个矩阵相乘后单个相加。<br>
$out=beta·mat_i + alpha·(batch1_i·batch2_i)$
9. [x] <font color=#a020f0>torch.bmm(batch1, batch2, out=None)</font>。矩阵--- batch 单个矩阵相乘。<br>
$out=batch1_i·batch2_i$
10. [x] <font color=#a020f0>torch.ger(vec1, vec2, out=None)</font>。矩阵--- 向量相乘生成矩阵
11. [x] <font color=#a020f0>torch.inverse(input, out=None)</font>。矩阵--- 取逆
12. [x] <font color=#a020f0>torch.dot(tensor1, tensor2)</font>。矩阵--- 内积。计算两个张量的内积, 两个张量都是一维向量
13. [x] <font color=#a020f0>torch.eig(a, eigenvectors=False, out=None)</font>。矩阵--- 特征值+特征向量
14. [x] <font color=#a020f0>torch.symeig(input, eigenvectors=False, upper=True, out=None)</font>。矩阵--- 实对称矩阵的特征值+特征向量
15. [x] <font color=#a020f0>torch.qr(input, out=None)</font>。矩阵--- QR分解
16. [x] <font color=#a020f0>torch.svd(input, some=True, out=None)</font>。矩阵--- 奇异值分解
17. [x] <font color=#a020f0>torch.gesv(B, A, out=None)</font>。矩阵--- 线性方程组的解。<br>
$X, Lu = torch.gesv(B, A)$, 返回线性方程$A·x=B$的解
18. [x] <font color=#a020f0>torch.btrifact(A, info=None)</font>。矩阵--- 方程组求解 IntTensor。返回一个元组，包含LU分解和pivots
19. [x] <font color=#a020f0>torch.btrisolve(b, LU_data, LU_pivots)</font>。矩阵--- 方程组求解r。返回线性方程组Ax=b的LU解
20. [x] <font color=#a020f0>torch.diag(input, diagonal=0, out)</font>。矩阵--- 对角线
21. [x] <font color=#a020f0>torch.histc(input, bins=100, min=0, max=0, out=None)</font>。矩阵---直方图。bins(int):直方图分区个数
22. [x] <font color=#a020f0>torch.trace(input)</font>。矩阵--- 对角线和。返回输入2维矩阵对角线元素的和
23. [x] <font color=#a020f0>torch.tril(input, k=0, out)</font>。矩阵--- 下三角
24. [x] <font color=#a020f0>torch.triu(input, k=0, out)</font>。矩阵--- 上三角
25. [x] <font color=#a020f0>torch.gels(B, A, out=None)</font>。矩阵--- 最小二乘解。输出：元组，X: 最小二乘解 qr: QR分解的细节