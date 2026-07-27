# HTML 博文添加规范

本文档记录当前项目中已经验证可用的 HTML 博文接入方式。以后新增 HTML 博文时，统一按照本规范处理。

## 一、基本原则

1. HTML 原稿、共享封面和配套脚本统一保存在根目录的 `datasets` 中。
2. Hugo 的 `content` 目录只保存栏目配置和轻量的 Markdown 博文入口。
3. 每个 HTML 文件对应一篇博文，也对应一个 Markdown 入口。
4. 同一专题的封面和 `annotator.js` 只保留一份，不复制到每篇博文目录。
5. 博文卡片只展示封面、标题、日期和阅读按钮，不展示简介。
6. 博文 URL 使用 `slug`，不暴露 `content` 中的真实文件结构。
7. 日期和时间统一使用北京时间 `+08:00`。

## 二、文件目录

HTML 原稿按照内容类型分类：

```text
datasets/
├── posts/                       # 博文 HTML
│   └── mmlm/
│       ├── qwen/
│       │   ├── qwen.html
│       │   └── qwen.png         # Qwen 专题共享封面
│       └── hunyuan/
│           ├── HunyuanImage_3_bilingual.html
│           ├── hunyuan-image-3.html
│           ├── hunyuan-image-3-vit.html
│           ├── hunyuan-image-3-vae.html
│           ├── hunyuan-image-3-backbone.html
│           ├── hunyuan-image-3-flow-matching.html
│           ├── annotator.js     # 混元专题共享批注脚本
│           └── hunyuan.png      # 混元专题共享封面
├── notes/                       # 笔记 HTML
└── docs/                        # 文档 HTML
```

对应的 Hugo 博文入口：

```text
content/posts/
└── 00400_vlp/
    ├── _index.zh-cn.md          # “多模态”栏目
    ├── 0100_qwen/
    │   ├── _index.zh-cn.md      # “Qwen”子栏目
    │   └── 0010_overview/
    │       └── index.zh-cn.md
    └── 0200_hunyuan/
        ├── _index.zh-cn.md      # “混元”子栏目
        ├── 0010_report/
        │   └── index.zh-cn.md
        ├── 0020_overview/
        │   └── index.zh-cn.md
        └── ...
```

每篇博文使用独立文件夹和 `index.zh-cn.md`，形成 Hugo Leaf Bundle。这样以后某一篇文章需要独有附件时，可以直接放进对应文件夹；共享资源仍然放在 `datasets` 中。

## 三、新建子栏目

如果专题栏目不存在，先创建：

```text
content/posts/<一级栏目>/<专题目录>/_index.zh-cn.md
```

模板：

```yaml
---
title: "专题名称"
url: "/zh-cn/topics/topic-slug/"
aliases:
  - "/zh-cn/posts/真实目录路径/"
menu:
  sidebar:
    name: 专题名称
    identifier: 唯一的专题标识
    parent: 一级栏目标识
    weight: 200
---
```

注意：

- `identifier` 在整个侧边栏菜单中必须唯一。
- `parent` 指向上一级目录的 `identifier`。
- `weight` 控制左侧目录中的显示顺序。
- `url` 使用对外公开的简洁地址。
- `aliases` 保留旧路径兼容，但页面主要地址不能使用真实目录结构。

## 四、新建 HTML 博文入口

每个 HTML 对应一个 `index.zh-cn.md`：

```yaml
---
title: "博文标题"
slug: public-post-slug
date: 2026-07-27T22:50:00+08:00
hero: datasets/posts/mmlm/topic/cover.png
menu:
  sidebar:
    name: 左侧目录名称
    identifier: 唯一的博文标识
    parent: 所属专题标识
    weight: 10
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["模型名称", "多模态", "技术主题"]
categories: ["AIGC"]
enableTOC: true
aliases:
  - /zh-cn/posts/原始目录路径/
---

{{< embed-html
    src="datasets/posts/mmlm/topic/article.html"
    script="datasets/posts/mmlm/topic/annotator.js"
    title="iframe 无障碍标题"
>}}
```

如果该 HTML 不需要批注功能，删除 `script` 参数：

```text
script="datasets/posts/mmlm/topic/annotator.js"
```

## 五、共享封面

同一专题共用一张封面时，只在 `datasets` 中保存一份：

```text
datasets/posts/mmlm/hunyuan/hunyuan.png
```

所有相关博文的 Front Matter 直接引用它：

```yaml
hero: datasets/posts/mmlm/hunyuan/hunyuan.png
```

不要把相同图片复制到每个博文文件夹中。

主题中的 `get-hero.html` 已支持从 `datasets` 读取共享图片，并在 Hugo 构建时发布到网站对应路径。

如果某篇博文需要独有封面，也可以把图片放在该博文的 Leaf Bundle 中：

```text
content/posts/.../article/
├── index.zh-cn.md
└── hero.png
```

此时使用：

```yaml
hero: hero.png
```

## 六、annotator.js

`embed-html` 短代码支持以下参数：

```text
script="datasets/posts/mmlm/topic/annotator.js"
```

构建时会：

1. 读取同一专题共享的 `annotator.js`。
2. 移除 HTML 中原有的 `<script defer src="annotator.js"></script>`。
3. 将脚本安全地内联进 `srcdoc`。
4. 即使原 HTML 没有写 `<script>` 标签，只要 Markdown 传入了 `script` 参数，也会注入批注功能。

因此：

- 一个专题只保留一份 `annotator.js`。
- 不要把脚本复制到各博文目录。
- 不需要把 `annotator.js` 放进 `static`。
- HTML 原稿中可以保留原来的脚本标签，构建时会自动替换。

## 七、HTML 内部专题链接

HTML 通过 iframe 的 `srcdoc` 展示。相对链接如：

```html
<a href="hunyuan-image-3-vit.html">ViT 专题</a>
```

无法正确跳转到 Hugo 博文，必须改成公开博文 URL：

```html
<a href="/zh-cn/posts/hunyuan-image-3-vit/">ViT 专题</a>
```

`embed-html` 会自动注入：

```html
<base target="_top">
```

因此点击链接时会跳转整个页面，不会在 iframe 中再次嵌套一个完整网站。

## 八、命名和排序

推荐目录使用十进制间隔，便于以后插入新文章：

```text
0010_report/
0020_overview/
0030_vit/
0040_vae/
```

左侧目录顺序由菜单 `weight` 控制：

```yaml
weight: 10
weight: 20
weight: 30
```

博文卡片默认按照日期倒序显示。需要固定卡片顺序时，让希望排在前面的文章使用稍晚的时间。

## 九、添加完成后的检查

每次新增 HTML 博文后至少检查：

1. `hugo --cleanDestinationDir` 构建成功。
2. 专题目录页能够找到所有新增博文。
3. 左侧目录的父子层级和顺序正确。
4. 博文 URL 使用 `slug`，没有暴露真实目录。
5. 卡片封面引用共享图片，没有重复复制。
6. HTML 内容完整显示，iframe 高度能够自动适应。
7. `annotator.js` 已注入且没有相对路径加载错误。
8. HTML 内部专题链接跳转到对应的 Hugo 博文。
9. 原始真实路径通过 `aliases` 仍可兼容访问。
10. 博文卡片中不显示简介。

## 十、当前示例

Qwen：

```text
HTML：datasets/posts/mmlm/qwen/qwen.html
封面：datasets/posts/mmlm/qwen/qwen.png
博文：/zh-cn/posts/qwen-image-edit/
```

混元：

```text
HTML：datasets/posts/mmlm/hunyuan/*.html
脚本：datasets/posts/mmlm/hunyuan/annotator.js
封面：datasets/posts/mmlm/hunyuan/hunyuan.png
栏目：/zh-cn/topics/hunyuan/
```

以后添加 HTML 博文时，以这两个专题的实现为标准。
