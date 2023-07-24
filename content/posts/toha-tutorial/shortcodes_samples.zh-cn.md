---
title: "区域块-实例"
date: 2021-06-08T06:00:20+06:00
menu:
  sidebar:
    name: 区域块-实例
    identifier: shortcodes
    parent: toha-tutorial
    weight: 20
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["shortcodes"]
categories: ["Basic"]
math: true
mermaid: true
enableEmoji: true
---

:money_mouth_face:

This is a sample post intended to test the followings:

- Default hero image.
- Different shortcodes.

## 一、报警(Alert)

The following alerts are available in this theme.

{{< alert type="success" >}}
这是 `type="success"`的报警样例.  
格式:  
` {`{< alert type="success" > }`}`  
`内容`   
` {`{< /alert > }`}`
{{< /alert >}}


{{< alert type="danger" >}}
这是 `type="danger"`的报警样例.  
格式:  
` {`{< alert type="danger" > }`}`  
`内容`   
` {`{< /alert > }`}`
{{< /alert >}}

{{< alert type="warning" >}}
这是 `type="warning"`的报警样例.  
格式:  
` {`{< alert type="warning" > }`}`  
`内容`   
` {`{< /alert > }`}`
{{< /alert >}}

{{< alert type="info" >}}
这是 `type="info"`的报警样例.  
格式:  
` {`{< alert type="info" > }`}`  
`内容`   
` {`{< /alert > }`}`
{{< /alert >}}

{{< alert type="dark" >}}
这是 `type="dark"`的报警样例.  
格式:  
` {`{< alert type="dark" > }`}`  
`内容`   
` {`{< /alert > }`}`
{{< /alert >}}

{{< alert type="primary" >}}
这是 `type="primary"`的报警样例.  
格式:  
` {`{< alert type="primary" > }`}`  
`内容`   
` {`{< /alert > }`}`
{{< /alert >}}

{{< alert type="secondary" >}}
这是 `type="secondary"`的报警样例.  
格式:  
` {`{< alert type="secondary" > }`}`  
`内容`   
` {`{< /alert > }`}`
{{< /alert >}}


## 二、插入图片(Image)
语法格式:  
> `{`{< img src="/zh-cn/posts/toha-tutorial/datasets/toha/moon.jpg" `title`="鼠标停在图片上显示的" `height`="尺寸" `width`="尺寸" `align`="center" `float`="right">}`}`  

属性设置：  
> `float`: 图片与文本内容之间的关系，值: right：表示图片在文本的右边(文本中插入图片，图片放在右边)  
> `align`: 图片排版方式，值：center：表示居中放置  

例如：`{`{< img src="/zh-cn/posts/toha-tutorial/datasets/toha/moon.jpg" height="200" width="300" float="left" title="A boat at the sea" >}`}`  

{{< img src="/zh-cn/posts/toha-tutorial/datasets/toha/moon.jpg" height="200" width="300" float="left" title="A boat at the sea" >}}

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Cras egestas lectus sed leo ultricies ultricies. Praesent tellus risus, eleifend vel efficitur ac, venenatis sit amet sem. Ut ut egestas erat. Fusce ut leo turpis. Morbi consectetur sed lacus vitae vehicula. Cras gravida turpis id eleifend volutpat. Suspendisse nec ipsum eu erat finibus dictum. Morbi volutpat nulla purus, vel maximus ex molestie id. Nullam posuere est urna, at fringilla eros venenatis quis.

Fusce vulputate dolor augue, ut porta sapien fringilla nec. Vivamus commodo erat felis, a sodales lectus finibus nec. In a pulvinar orci. Maecenas suscipit eget lorem non pretium. Nulla aliquam a augue nec blandit. Curabitur ac urna iaculis, ornare ligula nec, placerat nulla. Maecenas aliquam nisi vitae tempus vulputate.  

## 三、页面分割(split)

这个主题支持将页面分割成你想要的任意多列。

### 1. 分割成两列
语法格式：  
> `{`{< split 6 6>}`}`  
> `---`  
> `{`{< /split >}`}`

例如：  
> `{`{< split 6 6>}`}`  
> `这是左边列`  
> `---`  
> `这是右边列`  
> `{`{< /split >}`}`  

结果样式：  
{{< split 6 6>}}
这是左边列
---
这是右边列
{{< /split >}}

### 2. 分割3列
语法格式：  

> `{`{< split 4 4 4 >}`}`  
> `---`  
> `---`  
> `{`{< /split >}`}`  

例如：  
> `{`{< split 4 4 4 >}`}`  
> 这是左边列  
> `---`  
> 这是中间列  
> `---`  
> 这是右边列  
> `{`{< /split >}`}`  

结果样式：  
{{< split 4 4 4 >}}
这是左边列
---
这是中间列
---
这是右边列
{{< /split >}}

## 四、垂直方向-空行
在两行之间加入`空行`  
语法格式:   
> `{`{< vs 4>}`}` ： 表示加入4个`空行`


## 五、视频
语法格式：  
> `{`{< video src="/videos/sample.mp4" >}`}`  

例如：  

{{< video src="/videos/sample.mp4" >}}

<!-- markdown-link-check-disable-next-line -->
Video by [Rahul Sharma](https://www.pexels.com/@rahul-sharma-493988) from [Pexels](https://www.pexels.com).


## 六、hero

要显示自己的hero图：  

例如： 在一个块(CV)下创建子块(cv_sub)，路径如下：  
├── _index.en.md  
├── _index.zh-cn.md `identifier: cv`  
├── cv_sub  
│　├── _index.zh-cn.md `identifier: cv_sub; parent: cv`  
│　└── rich_content  
│　　　├── images `hero图片位置`  
│　　　│　├── forest.jpg  
│　　　│　└── hero.svg  
│　　　└── index.md `真正的博文内容，必须命名为index.md。identifier: rich-content; parent: cv_sub`  


## 七、流程图

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