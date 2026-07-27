---
title: 文档写作说明
description: 新增和组织站内文档的通用格式。
slug: writing-docs
weight: 10
menu:
  docs:
    name: 文档写作说明
    identifier: docs-writing-guide
    weight: 10
enableTOC: true
---

## 文档与博文的区别

文档页面专注于内容本身，不显示博文的封面、头像、作者、发布日期、Tags 和分享按钮。

文档仍然保留以下能力：

- 左侧文档目录
- Markdown 正文
- 右侧页内目录
- 代码高亮
- 数学公式和 Mermaid 图表
- 左侧目录缩放、收起与展开
- 右侧目录收起与展开

## 新增一篇文档

在 `content/docs/` 下创建一个 Markdown 文件，例如：

```text
content/docs/qwen-deployment.zh-cn.md
```

文件头部使用下面的通用格式：

```yaml
---
title: Qwen 部署手册
description: Qwen 服务的安装、配置和部署过程。
slug: qwen-deployment
weight: 20
menu:
  docs:
    name: Qwen 部署手册
    identifier: docs-qwen-deployment
    weight: 20
enableTOC: true
---
```

Front Matter 结束后即可直接编写 Markdown 正文。

## 字段说明

### title

页面显示的文档标题。

### description

文档简介，会显示在文档首页列表和文档标题下方。

### slug

公开 URL 使用的名称，与真实文件名和文件夹结构解耦。

### menu.docs

控制文档在左侧目录中的名称、层级和顺序。

### weight

数字越小，文档在列表和目录中的位置越靠前。
