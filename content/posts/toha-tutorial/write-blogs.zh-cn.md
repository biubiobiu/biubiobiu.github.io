---
title: "撰写文章"
date: 2020-06-08T06:00:20+06:00
menu:
  sidebar:
    name: 撰写文章
    identifier: write-blogs-github
    parent: toha-tutorial
    weight: 11
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["博文路径"]
categories: ["Basic"]
---

## 一、创建类别

### 1、创建文章

在`content`文件夹中创建`posts`文件夹，在该文件夹中创建一个`_index.zh-cn.md`文件(中文环境)/`_index.en.md`(英文环境)。在里面添加如下内容：

``` markdown
---
title: Posts
---
```

 现在，假设你想写一篇文章。首先，创建一个文件，在末尾用markdown扩展名命名它。例如:我们创建了一个名为analytics-and-comments.en.md，并添加以下几行内容。如果在中文环境下创建，名字应该是analytics-and-comments.zh-cn.md: 

``` markdown
---
title: "Analytics and Comments"
date: 2020-06-08T06:00:23+06:00
hero: /images/posts/writing-posts/analytics.svg
description: Adding analytics and disquss comment in hugo 
theme Toha
menu:
  sidebar:
    name: Analytics & Comments
    identifier: analytics-and-comments
    weight: 500
---

### Complete Post Coming Soon...
```

在文件的头部以3个-开始和结束，称为`前置内容`。我们写得每一篇博客文章都需要有前置内容，在前置内容之后，可以开始写文章内容了，前置内容的参数有：

| 参数        | 解释                                                         |
| :---------- | ------------------------------------------------------------ |
| title       | 贴子的标题                                                   |
| date        | 显示博客发布时间，第一部分 `year-month-date format`        |
| hero        | 文章封面图的位置路径。创建路径`static/images/posts/writingposts/` 在其中放置图片文件 |
| description | 添加任意你喜欢的描述                                         |
| menu        | 这个部分包含了另一个sidebar参数，该参数定义了`侧边栏`中文件结构的样子。该参数的子参数有：`name`,`identifier`,`weight` |
|             | `name`: 定义了侧边栏文件层次结构中，文档的名称                 |
|             | `identifier`: 标识符。有助于将文件与其他文件区分开来，有助于分类 |
|             | `weight`: 权重值，对于多个文件，文档将基于该权重值以升序出现在文件层次结构中。 |
|             | `parent`:                                                      |
|             |                                                              |

![image error](/zh-cn/posts/toha-tutorial/datasets/toha/blog_create_1.png "image error")

### 2、创建子类

刚刚我们创建了一个`_index.zh-cn.md`文件和一个博客文章的markdown文件，现在我们创建一个子类。创建一个文件夹 `getting-started/_index.zh-cn.md`，该文件中包含下面的前置内容:

```markdown
---
title: Deploy Site
menu:
  sidebar:
    name: Deploy Site
    identifier: getting-started
    weight: 300
---
```

上述代码块中各个参数的含义前面已经讨论过了。 只是，暂时请记住，我们将创建类别名称作为getting-started，这就是我们将其作为标识符包含在此 _index.md 中的原因。 接下来，我们将创建一个名为 github-pages.md 的 Markdown 文件。这将是我们此文件夹的博客文章文件。 github -pages.md 包括以下几行：

```markdown
---
title: "Deploy site in Github Pages"
date: 2020-06-08T06:00:20+06:00
hero: /images/posts/writing-posts/git.svg
menu:
  sidebar:
    name: Github Pages
    identifier: getting-started-github
    parent: getting-started
    weight: 10
---
目录关系如下：
getting-started
|__ _index.md_
|__ github-pages.md
```

一个新参数：parent：该参数的值一定要与上一级的 标签(identifier)相匹配。

![](/zh-cn/posts/toha-tutorial/datasets/toha/blog_create_2.png)



### 3、作者信息

在默认情况下，文章的作者信息用的是 `config.yaml`文件中的相关信息。如果想修改作者信息，可以在`前置内容`中添加 author 块：

```markdown
author:
  name: Md.Habibur
  image: /images/authors/habib.jpg
```



##  二、创建子类


It's coming soon ...