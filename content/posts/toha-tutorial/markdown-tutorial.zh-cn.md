---
title: "MarkDown入门"
date: 2021-06-08T06:00:20+06:00
menu:
  sidebar:
    name: MarkDown入门
    identifier: markdown-tutorial-github
    parent: toha-tutorial
    weight: 13
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["MarkDown","教程"]
categories: ["Basic"]
math: true
---

## 一、小技巧

<a href="https://www.w3school.com.cn/tags/index.asp" target="blank">可以使用html的标签</a>  
markdown中常用的html标签：  
| 操作 | 标签 |
| :-------- | :-- |
| 换行 | `测试<br>一下`|
| 标记 | `<mark>测试一下</mark>`|
| 引用 | `<cite>引用[^1]</cite>`|
| 空格 | `&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;测试一下`|
| 删除线 | `<s>测试一下</s>`|
| 下划线 | `<u>测试一下</u>`|
| 字体增大 | `<big>测试一下</big>`|
| 字体减小 | `<small>测试一下</small>`|
| 文字上标 | `测试<sup>一下</sup>`|
| 文字下标 | `测试<sub>一下</sub>`|
| 右对齐 | `<p align=right>测试一下</p>`|
| 文字居中 | `<center>测试一下</center>`|
| 图片居中 | `<p align="center"><img src="***.jpg" width="60%"></p>`|
| 超链接 | `<a href="center" target="blank">文本</a>` <br> href指定跳转的目标路径；<br>target属性指定超链接打开的位置，<br>　　值blank: 表示在一个新的页面中打开；<br>　　默认值self: 在当前页面中打开超链接|
| 图片 | `<img src="***.jpg" width="60%" height="图片高度(单位是像素级)" alt="图片描述，当图片加载失败时显示">`|
| 音频 | `<audio src="音频的url" controls="是否允许用户控制播放" autoplay="音频文件是否自动播放" `<br>`loop="音频是否循环播放" preload="音频在页面加载时进行加载-如果设置了autoplay则忽略该属性">`|
| 视频 | 跟音频一样，只是多了width和height |

| 操作 | 需求 |
| :-- | :-- |
| Markdown只能识别一个空格(在半角输入状态下)。有两种方法插入更多空格</p> 方法一：手动输入空格(半个空格`&nbsp;`)(半角相当于1个空格`&ensp;`)(全角相当于2个空格`&emsp;`) <br> 方法二：使用权角空格即：在全角状态下直接使用空格键就ok了  | 添加空格 |
| 如果行与行之间没有空行，则会被视为同一段落。</p> 方法一：段内换行，在上一行的结尾插入两个以上的空格然后回车; 或者直接用 `<br>` <br>方法二：新起一段，在上一行的结尾插入两个以上的空格然后回车+空行；或者直接用`</p>`| 换行 |



## 二、基本语法

<a href="https://www.runoob.com/markdown/md-tutorial.html" target="blank">教程</a>  

### 1. 代码块

```python
​```语言名称```
```

### 2. 标题

```python
# 一阶标题 
## 二阶标题 
### 三阶标题 
#### 四阶标题 
##### 五阶标题
###### 六阶标题
```

### 3. 字体

1. 斜体： 格式：`*文本*` </br>  示例： `*斜体*`： *斜体*
2. 加粗： 格式：`**文本**` </br> 示例： `**加粗**`： **加粗**
3. 斜体+加粗： 格式：`***文本***` </br> 示例：`***斜体加粗***`：***斜体加粗***
4. 删除线： 格式：`~~文本~~或者<s>文本</s>` </br> 示例：`<s>删除线</s>`： <s>删除线</s>
5. 背景高亮：格式：`<mark>文本</mark>` </br> 示例：`<mark>高亮</mark>`：<mark>高亮</mark>
6. 背景按钮形式：格式：`<kbd>文本</kbd>` </br> 示例：`<kbd>按钮</kbd>`：<kbd>按钮</kbd>
7. 上标：格式：`<sup>文本</sup>` </br> 示例：`x<sup>20</sup>y`：x<sup>20</sup>y
8. 下标：格式：`<sub>文本</sub>` </br> 示例：`H<sub>2</sub>O`：H<sub>2</sub>O


### 4. 引用

```python
语法：>在引用的文字前加>即可。引用也可嵌套，比如加两个>>   三个>>>
引用文献 语法：
<cite>论文名[^1]</cite>
[^1]: 详细的内容
```

> Don't communicate by sharing memory, share memory by communicating.</br>
> — <cite>Rob Pike[^1]</cite> </br> 引用第二篇论文 <cite>Matting[^2]</cite>


[^1]: The above quote is excerpted from Rob Pike's [talk](https://www.youtube.com/watch?v=PAAkCSZUG1c) during Gopherfest, November 18, 2015.
[^2]: 这是第二个引用的详细内容


### 5. 分割线

```python
语法：两种表达方式
---
***
```
下面是分割线： `---`

---

下面是分割线： `***`

***


### 6. 插入图片

> **markdown语法**：`![图片名称](图片地址)` </br>
> []: 里面的内容表示图片未加载时的提示文字 </br>
> (): 表示图片地址 </br>

> **html语法**：</br>
> 插入图片：`<img src="***.jpg" width="60%" height="图片高度(单位是像素级)" alt="图片描述，当图片加载失败时显示">` </br>
> 居中：`<p align="center"><img src="***.jpg" width="60%"></p>` </br>

1. 本项目中，图片统一放在根目录下的 /static/ 路径下：例如：图片路径: /static/datasets/moon.jpg</br>
`<p align="center"><img src="/datasets/moon.jpg" width="30%" height="30%" title="moon" alt="moon"></p>` </br>
<p align="center"><img src="/datasets/moon.jpg" width="30%" height="30%" title="moon" alt="moon"></p>

2. 如果图片与本文放在同一个路径下，例如：图片路径: /content/posts/***/moon.jpg </br>
`<p align="center"><img src="/zh-cn/posts/***/moon.jpg" width="30%" height="30%" title="moon" alt="moon"></p>` </br>
<p align="center"><img src="/zh-cn/posts/toha-tutorial/datasets/toha/moon.jpg" width="30%" height="30%" title="moon" alt="moon"></p>


### 7. 超链接

> **markdown语法**：`[名称](url地址/本地地址)`

> **html语法**：`<a href="目标路径" target="blank">文本</a>`

1. 本项目的地址，例如本地地址: /content/posts/***/latax_formula.zh-cn.md
例如：`<a href="/zh-cn/posts/***/latax_formula" target="bland">katex</a>` </br>
<a href="/zh-cn/posts/toha-tutorial/latax_formula" target="bland">本地路径：katex</a>

2. 外网地址，例如：`<a href="https://www.baidu.com/" target="blank">百度一下</a>`</br>
<a href="https://www.baidu.com/" target="blank">百度一下</a>


### 8. 表格

```python
语法：
|表头|表头|表头|
|:--|:--:|--:|
|内容|内容|内容|
|内容|内容|内容|
```

| 表头 | 表头 | 表头 |
| :--- | :--: | ---: |
| 内容 | 内容 | 内容 |
| 内容 | 内容 | 内容 |

### 9. 列表

#### 1.无序列表

```python
- 列表内容
+ 列表内容
* 列表内容

```

+ 效果一样   

  - 二级

    + 三级

      * 四级

        

#### 2. 有序列表

```python
数字加点
例如：1. 有序列表内容
```

1. 一级有序列表内容
   1. 二级有序列表
      1. 三级有序列表
         1. 四级有序列表
2. 一级有序列表内容

### 10. 流程图

```python
st=>start: 开始
```



### 11. 注释

```python
语法: <!-- this is a comment -->

```

<!--This is a commet-->



### 12. 公式

markdown的公式: 可以使用两个美元符 `$$` 包裹 TeX 或 LaTeX 格式的数学公式来实现。提交后，问答和文章页会根据需要加载 Mathjax 对数学公式进行渲染，例如：

<a href="https://katex.org/docs/supported.html" target="blank">公式katex文档</a>  

| 希腊       | 转义     | 希腊       | 转义     | 希腊       | 转义     | 希腊          | 转义        |
| ---------- | -------- | ---------- | -------- | ---------- | -------- | ------------- | ----------- |
| $$\alpha$$   | \alpha   | $$\kappa$$   | \kappa   | $$\psi$$     | \psi     | $$\digamma$$    | \digamma    |
| $$\beta$$    | \beta    | $$\lambda$$  | \lambda  | $$\rho$$     | \rho     | $$\varepsilon$$ | \varepsilon |
| $$\chi$$     | \chi     | $$\mu$$      | \mu      | $$\sigma$$   | \sigma   | $$\varkappa$$   | \varkappa   |
| $$\delta$$   | \delta   | $$\nu$$     | \nu      | $$\tau$$     | \tau     | $$\varphi$$     | \varphi     |
| $$\epsilon$$ | \epsilon | $$\omicron$$ | \omicron | $$\theta$$   | \theta   | $$\varpi$$      | \varpi      |
| $$\eta$$     | \eta     | $$\omega$$   | \omega   | $$\upsilon$$ | \upsilon | $$\varrho$$     | \varrho     |
| $$\gamma$$   | \gamma   | $$\phi$$     | \phi     | $$\xi$$      | \xi      | $$\varsigma$$   | \varsigma   |
| $$\iota$$    | \iota    | $$\pi$$      | \pi      | $$\zeta$$    | \zeta    | $$\vartheta$$   | \vartheta   |
| $$\Delta$$   | \Delta   | $$\Theta$$   | \Theta   | $$\Lambda$$  | \Lambda  | $$\Xi$$        | \Xi         |
| $$\Gamma$$   | \Gamma   | $$\Upsilon$$ | \Upsilon | $$\Omega$$   | \Omega   | $$\Phi$$        | \Phi        |
| $$\Pi$$      | \Pi      | $$\Psi$$     | \Psi     | $$\Sigma$$   | \Sigma   | $$\aleph$$      | \aleph      |
| $$\beth$$    | \beth    | $$\gimel$$   | \gimel   | $$\daleth$$  | \daleth  |               |             |



我是一个公式 `$$\Gamma(n) = (n-1)!$$`：$$\Gamma(n) = (n-1)!$$

Block math: `$$ \varphi = 1+\frac{1} {1+\frac{1} {1+\frac{1} {1+\cdots} } } $$`
$$
 \varphi = 1+\frac{1} {1+\frac{1} {1+\frac{1} {1+\cdots} } } 
$$

$$\alpha = \frac a b$$

### 13. 切割成列

这个主题支持将页面分割成尽可能多的列。

```markdown
{< split 6 6>}
```
