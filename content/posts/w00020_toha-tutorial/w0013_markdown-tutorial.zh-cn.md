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
mermaid: true
enableEmoji: true
---

## 一、小技巧

<a href="https://www.w3school.com.cn/tags/index.asp" target="blank">可以使用html的标签</a>  
markdown中常用的html标签：  
| 操作 | 标签 |
| :-------- | :-- |
| 换行 | `测试<br>一下`|
| <mark>标记</mark> | `<mark>测试一下</mark>`|
| <kbd>按钮</kbd> | `<kbd>测试一下</kbd>`|
| <font color="#A020F0">颜色</font>|`<font color="#A020F0">颜色</font>`|
| <font size="4">四号文字</font>|`<font size="4">四号文字</font>`|
| <cite>引用[^1]</cite> | `<cite>引用[^1]</cite>`|
| 空格 | `&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;测试一下`|
| <s>删除线</s> | `<s>测试一下</s>`|
| <u>下划线</u> | `<u>测试一下</u>`|
| <big>字体增大</big> | `<big>测试一下</big>`|
| <small>字体减小</small> | `<small>测试一下</small>`|
| 文字<sup>上标</sup> | `测试<sup>一下</sup>`|
| 文字<sub>下标</sub> | `测试<sub>一下</sub>`|
|加n个空行|`{`{< vs n>}`}`|
| <p align=right>右对齐</p>| `<p align=right>测试一下</p>`|
| <center>文字居中</center> | `<center>测试一下</center>`|
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

> 并不一定真要放代码时才用这个标签，比如：
>* 如果要重点突出某个字，可以用行内代码标签
>* 如果不想让Markdown渲染某段文字，可以用代码块标签进行包裹 
> 
> 1. 行内代码标签： \`行内代码\`
> 2. 代码块：通过一对 ```包裹


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

#### a. 引用文本

> **语法**：`>` 加一个空格。 多级引用是可以嵌套的。<br>
> `> ` 一级引用文本 <br>
>> `>> ` 二级引用文本 <br>
>>> `>>> ` 三级引用文本 <br>
>>>* `>>> * ` 三级引用，无序列表 <br>

#### b. 引用参考文献

> **语法**：<br>
> `<cite>论文名[^1]</cite>` <br>
> `[^1]: 详细的内容` <br>

> Don't communicate by sharing memory, share memory by communicating.</br>
> — <cite>Rob Pike[^2]</cite> </br> 引用第二篇论文 <cite>Matting[^3]</cite>


[^1]: 测试
[^2]: The above quote is excerpted from Rob Pike's [talk](https://www.youtube.com/watch?v=PAAkCSZUG1c) during Gopherfest, November 18, 2015.
[^3]: 这是第二个引用的详细内容



### 5. 分割线

> 下面是分割线： `---`

---

> 下面是分割线： `***`

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

---
> 图文混排  
```
左图右文，例如：
<p>  
<img src="/datasets/moon.jpg" width="30%" height="30%" align="left" />  
文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。  
</p>  
```
> 左文右图  
<p>  
<img src="https://img2.baidu.com/it/u=638285213,1746517464&fm=253&fmt=auto&app=120&f=JPEG?w=1422&h=800" width="50%" height="50%" align="left" />  
文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。  文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。  文字在右边，图片在左边。文字在右边，图片在左边。文字在右边，图片在左边。
</p>  

---

### 7. 多媒体

> 视频语法：`{`{< video src="/videos/sample.mp4" >}`}`  
```
这个没啥用：
<video id="video" controls="" preload="none" poster="封面图链接"><source id="mp4" src="视频地址" type="video/mp4"></video>

这个有用：
{ {< video src="/videos/sample.mp4" >} }
```
{{< video src="/videos/sample.mp4" >}}

<!-- markdown-link-check-disable-next-line -->
Video by [Rahul Sharma](https://www.pexels.com/@rahul-sharma-493988) from [Pexels](https://www.pexels.com).


> 音频：
```
<audio id="audio" controls="" preload="none"><source id="mp3" src="音频地址"></audio>
```


### 8. 超链接

> **markdown语法**：`[描述](https://xxxx.com)`  
如果让项目默认：点击超链接，重新打开网页。  
可以在 themes/toha/layouts/_default/baseof.html 中的`<head>中添加<base target="_blank">`  

> **html语法**：`<a href="目标路径" target="blank">文本</a>`

1. 本项目的地址，例如本地地址: /content/posts/***/latax_formula.zh-cn.md
例如：`<a href="/zh-cn/posts/***/latax_formula" target="bland">katex</a>` </br>
<a href="/zh-cn/posts/toha-tutorial/latax_formula" target="bland">本地路径：katex</a>

2. 外网地址，例如：`<a href="https://www.baidu.com/" target="blank">百度一下</a>`</br>
<a href="https://www.baidu.com/" target="blank">百度一下</a>


### 9. 表格

> `语法：` <br>
> `|表头|表头|表头|` <br>
> `|:--|:--:|--:|` <br>
> `|内容|内容|内容|` <br>
> `|内容|内容|内容|` <br>


| 表头 | 表头 | 表头 |
| :--- | :--: | ---: |
| 内容 | 内容 | 内容 |
| 内容 | 内容 | 内容 |

### 10. 列表

#### a. 无序列表

> `markdown语法：`<br>
> `- 列表内容` <br>
> `+ 列表内容` <br>
> `* 列表内容` <br>


+ 效果一样   

  - 二级

    + 三级

      * 四级

> `html语法：太复杂` <br>
> `<ul><li>内容</li></ul>`
```
<ul> 
  <li>书籍
    <ul>
      <li>道德经</li>
    </ul>
  </li>
</ul>
```
效果：<br>
<ul> 
  <li>书籍
    <ul>
      <li>道德经</li>
    </ul>
  </li>
</ul>


#### b. 有序列表

> `markdown语法：数字加点，加空格`<br>
> `例如：1. 有序列表内容` <br>


1. 一级有序列表内容
   1. 二级有序列表
      1. 三级有序列表
         1. 四级有序列表
2. 一级有序列表内容

> `html语法：太复杂` <br>
> `用 <ol></ol> 和 <li></li>` <br>
```
<ol> 
  <li>书籍
    <ol>
      <li>道德经</li>
    </ol>
  </li>
</ol>
```
效果：<br>
<ol> 
  <li>书籍
    <ol>
      <li>道德经</li>
    </ol>
  </li>
</ol>

### 11. 流程图

```python
st=>start: 开始
```
跳转到：<a href="#三流程图">三、流程图</a>


### 12. 注释

被注释的文字不会显示出来。

> html注释：`<!-- this is a comment -->` <br>
> 例如：<br>
`<!--`<br>
`我是多行`<br>
`段落注释`<br>
`渲染时不会显示`<br>
`-->`<br>


> html标签：`style='display: none'`

> markdown注释：`[](注释内容，渲染时不会显示)`

### 13. 特殊字符

|特殊字符|语法|字符|
|:--|:--|:--|
|空格符| `&nbsp;` |&nbsp;|
|小于号| `&lt;` |&lt;|
|大于号| `&gt;` |&gt;|
|和号| `&amp;` |&amp;|
|人民币| `&yen;` |&yen;|
|版权| `&copy;` |&copy;|
|注册商标| `&reg;` |&reg;|
|摄氏度| `&deg;` |&deg;|
|正负号| `&plusmn;` |&plusmn;|
|乘号| `&times;` |&times;|
|除号| `&divide;` |&divide;|
|平方（上标²）| `&sup2;` |&sup2;|
|立方（上标³）| `&sup3;` |&sup3;|


### 14. 公式

markdown的公式: 可以使用两个美元符 `$$` 包裹 TeX 或 LaTeX 格式的数学公式来实现。提交后，问答和文章页会根据需要加载 Mathjax 对数学公式进行渲染，例如：

<a href="https://katex.org/docs/supported.html" target="blank">公式katex文档</a>  

|序号|大写|大写|小写|小写|英文|英语音标注音|汉语名称|常用指代意义|
|:--:|:--:|:--|:--:|:--|:--|:--|:--:|:--|
|1|$$\Alpha$$|\Alpha|$$\alpha$$|\alpha|alpha|/'ælfə/|阿尔法|角度、系数、角加速度、第一个、电离度、转化率|
|2|$$\Beta$$|\Beta|$$\beta$$|\beta|beta|/'beɪtə/|贝塔|角度、系数、磁通系数|
|3|$$\Gamma$$|\Gamma|$$\gamma$$|\gamma|gamma|/'gæmə/|伽玛|电导系数、角度、比热容比|
|4|$$\Delta$$|\Delta|$$\delta$$|\delta|delta|/'deltə/|德尔塔|变化量、焓变、熵变、屈光度、一元二次方程中的判别式、化学位移|
|5|$$\Epsilon$$|\Epsilon|$$\epsilon, \varepsilon$$|\epsilon, \varepsilon|epsilon|/'epsɪlɒn/|艾普西隆|对数之基数、介电常数、电容率、应变|
|6|$$\Zeta$$|\Zeta|$$\zeta$$|\zeta|zeta|/'zi:tə/|泽塔|系数、方位角、阻抗、相对黏度|
|7|$$\Eta$$|\Eta|$$\eta$$|\eta|eta|/'i:tə/|伊塔|迟滞系数、机械效率|
|8|$$\Theta$$|\Theta|$$\theta, \vartheta$$|\theta, \vartheta|theta|/'θi:tə/|西塔|温度、角度|
|9|$$\Iota$$|\Iota|$$\iota$$|\iota|iota|/aɪ'əʊtə/|约(yāo)塔|微小、一点|
|10|$$\Kappa$$|\Kappa|$$\kappa, \varkappa$$|\kappa, \varkappa|kappa|/'kæpə/|卡帕|介质常数、绝热指数|
|11|$$\Lambda$$|\Lambda|$$\lambda$$|\lambda|lambda|/'læmdə/|拉姆达|波长、体积、导热系数|
|12|$$\Mu$$|\Mu|$$\mu$$|\mu|mu|/mju:/|谬|磁导率、微、动摩擦系（因）数、流体动力黏度、货币单位、莫比乌斯函数|
|13|$$\Nu$$|\Nu|$$\nu$$|\nu|nu|/nju:/|纽|磁阻系数、流体运动粘度、光波频率、化学计量数|
|14|$$\Xi$$|\Xi|$$\xi$$|\xi|xi|/ksi/|克西|随机变量、（小）区间内的一个未知特定值|
|15|$$\Omicron$$|\Omicron|$$\omicron$$|\omicron|omicron|/əuˈmaikrən/|奥米克戎|高阶无穷小函数|
|16|$$\Pi$$|\Pi|$$\pi, \varpi$$|\pi, \varpi|pi|/paɪ/|派|圆周率、π(n)表示不大于n的质数个数、连乘|
|17|$$\Rho$$|\Rho|$$\rho, \varrho$$|\rho, \varrho|rho|/rəʊ/|柔|电阻率、柱坐标和极坐标中的极径、密度、曲率半径|
|18|$$\Sigma$$|\Sigma|$$\sigma, \varsigma$$|\sigma, \varsigma|sigma|/'sɪɡmə/|西格马|总和、表面密度、跨导、应力、电导率|
|19|$$\Tau$$|\Tau|$$\tau$$|\tau|tau|/taʊ/|陶|时间常数、切应力、2π（两倍圆周率）|
|20|$$\Upsilon$$|\Upsilon|$$\upsilon$$|\upsilon|upsilon|/ˈipsɪlon/|宇普西隆 |位移|
|21|$$\Phi$$|\Phi|$$\phi, \varphi$$|\phi, \varphi|phi|/faɪ/|斐|磁通量、电通量、角、透镜焦度、热流量、电势、直径、欧拉函数、相位、孔隙度|
|22|$$\Chi$$|\Chi|$$\chi$$|\chi|chi|/kaɪ/|希 /恺|统计学中有卡方(χ^2)分布|
|23|$$\Psi$$|\Psi|$$\psi$$|\psi|psi|/psaɪ/|普西|角速、介质电通量、ψ函数、磁链|
|24|$$\Omega$$|\Omega|$$\omega$$|\omega|omega|/'əʊmɪɡə/|欧米伽|欧姆、角速度、角频率、交流电的电角度、化学中的质量分数、有机物的不饱和度|

{{< vs 2>}}

我是一个公式 `$$\Gamma(n) = (n-1)!$$`：$$\Gamma(n) = (n-1)!$$

Block math: `$$ \varphi = 1+\frac{1} {1+\frac{1} {1+\frac{1} {1+\cdots} } } $$`
$$
 \varphi = 1+\frac{1} {1+\frac{1} {1+\frac{1} {1+\cdots} } } 
$$

`$$\alpha = \frac a b$$`: 
$$\alpha = \frac a b$$

### 15. 切割成列

这个主题支持将页面分割成尽可能多的列。

```markdown
{< split 6 6>}
```

### 16. 任务清单

> 语法实例 <br>
`- [ ] 未完成` <br>
`- [x] 已完成` <br>

- [ ] 未完成
- [x] 已完成

### 17. Markdown 变量

> Markdown中文持变量定义和变量引用，且支持中文。一处定义，处处使用，而且方便，统一修改。<br>
语法：<br>
步骤1：定义链接：`[百度]:https://www.baidu.com`  
步骤2：引用链接：`[自定义文本][百度]`  


[百度]:https://www.baidu.com

[自定义文本][百度]  

### 18. Markdown 锚点
> 场景：现在在写一篇博客，内容牵涉到以前的博文或者博文前面的章节。想设置一个超链接，跳转到前面博文的具体位置。  
> 步骤一： 在 需要跳至的位置 设置锚点(或者是前面的 标题)：`<a id="锚点1-id">跳到此处</a>`  
> 步骤二： 从该位置调到 锚点位置：`<a href="#锚点1-id">请看前博文</a>`  

例如：Markdown语法，参考：<a href="#二基本语法">基本语法</a>



## 三、流程图

### 1、设置
要是用流程图时，需要添加：`mermaid: true`

```python
title: "Mermaid Support"
date: 2022-03-14T06:15:35+06:00
menu:
  sidebar:
    name: Mermaid
    identifier: writing-post-mermaid
    parent: writing-post
    weight: 60
mermaid: true
```
### 2、语法
{{< alert type="info">}}
`{`{< mermaid align="left" >}`}`<br>
內容<br>
`{`{< /mermaid >}`}`

参数：
1. `align`：让您将图表对齐到左边、右边或中间(left, right, center)。默认对齐方式为居中。
2. `background`：让您更改图表的背景颜色。

{{< /alert >}}

### 3、实例
#### 1）Graph
`[]`：表示矩形框 <br>
`()`：表示圆角矩形框<br>
`{}`：表示菱形框<br>

```
`{`{< mermaid align="left" >}}
graph LR;
    A[Hard edge] -->|Link text| B(Round edge)
    B --> C{Decision}
    C -->|One| D[Result one]
    C -->|Two| E[Result two]
`{`{< /mermaid >}}
```

{{< mermaid align="left" >}}
graph LR;
    A[Hard edge] -->|Link text| B(Round edge)
    B --> C{Decision}
    C -->|One| D[Result one]
    C -->|Two| E[Result two]
{{< /mermaid >}}

#### b）序列图(Sequence Diagram)
```
`{`{< mermaid >}}
sequenceDiagram
    participant Alice
    participant Bob
    Alice->>John: Hello John, how are you?
    loop Healthcheck
        John->>John: Fight against hypochondria
    end
    Note right of John: Rational thoughts <br/>prevail!
    John-->>Alice: Great!
    John->>Bob: How about you?
    Bob-->>John: Jolly good!
`{`{< /mermaid >}}
```

{{< mermaid >}}
sequenceDiagram
    participant Alice
    participant Bob
    Alice->>John: Hello John, how are you?
    loop Healthcheck
        John->>John: Fight against hypochondria
    end
    Note right of John: Rational thoughts <br/>prevail!
    John-->>Alice: Great!
    John->>Bob: How about you?
    Bob-->>John: Jolly good!
{{< /mermaid >}}

#### c）甘特图 (Gantt diagram)

```
`{`{< mermaid >}}
gantt
  dateFormat  YYYY-MM-DD
  title Adding GANTT diagram to mermaid
  excludes weekdays 2014-01-10

section A section
  Completed task            :done,    des1, 2014-01-06,2014-01-08
  Active task               :active,  des2, 2014-01-09, 3d
  Future task               :         des3, after des2, 5d
  Future task2               :         des4, after des3, 5d
`{`{< /mermaid >}}
```

{{< mermaid >}}
gantt
  dateFormat  YYYY-MM-DD
  title Adding GANTT diagram to mermaid
  excludes weekdays 2014-01-10

section A section
  Completed task            :done,    des1, 2014-01-06,2014-01-08
  Active task               :active,  des2, 2014-01-09, 3d
  Future task               :         des3, after des2, 5d
  Future task2               :         des4, after des3, 5d
{{< /mermaid >}}


#### 4）类图(class diagram)
```
`{`{< mermaid >}}
classDiagram
  Class01 <|-- AveryLongClass : Cool
  Class03 *-- Class04
  Class05 o-- Class06
  Class07 .. Class08
  Class09 --> C2 : Where am i?
  Class09 --* C3
  Class09 --|> Class07
  Class07 : equals()
  Class07 : Object[] elementData
  Class01 : size()
  Class01 : int chimp
  Class01 : int gorilla
  Class08 <--> C2: Cool label
`{`{< /mermaid >}}
```

{{< mermaid >}}
classDiagram
  Class01 <|-- AveryLongClass : Cool
  Class03 *-- Class04
  Class05 o-- Class06
  Class07 .. Class08
  Class09 --> C2 : Where am i?
  Class09 --* C3
  Class09 --|> Class07
  Class07 : equals()
  Class07 : Object[] elementData
  Class01 : size()
  Class01 : int chimp
  Class01 : int gorilla
  Class08 <--> C2: Cool label
{{< /mermaid >}}

#### 5）git图(git graph)
```
`{`{< mermaid background="black" align="right" >}}
gitGraph:
options
{
    "nodeSpacing": 150,
    "nodeRadius": 10
}
end
commit
branch newbranch
checkout newbranch
commit
commit
checkout master
commit
commit
merge newbranch
`{`{< /mermaid >}}
```

{{< mermaid background="black" align="right" >}}
gitGraph:
options
{
    "nodeSpacing": 150,
    "nodeRadius": 10
}
end
commit
branch newbranch
checkout newbranch
commit
commit
checkout master
commit
commit
merge newbranch
{{< /mermaid >}}

#### 6）ER图(ER Diagram)
```
`{`{< mermaid >}}
erDiagram
    CUSTOMER ||--o{ ORDER : places
    ORDER ||--|{ LINE-ITEM : contains
    CUSTOMER }|..|{ DELIVERY-ADDRESS : uses
`{`{< /mermaid >}}
```

{{< mermaid >}}
erDiagram
    CUSTOMER ||--o{ ORDER : places
    ORDER ||--|{ LINE-ITEM : contains
    CUSTOMER }|..|{ DELIVERY-ADDRESS : uses
{{< /mermaid >}}
