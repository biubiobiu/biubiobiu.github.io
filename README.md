# biubiobiu.github.io

An example hugo static site with Toha theme.

1. archetypes： 原型（创建新内容时使用的模板）；
2. assets： 存储 Hugo Pipes 需要处理的所有文件。只有使用 .Permalink 或的文件 .RelPermalink 才会发布到 public 目录中。注意：默认情况下未创建资产目录；
3. config： Hugo附带了大量的配置指令。在config目录正是这些指令被存储为JSON，YAML，或TOML文件。每个根设置对象都可以作为自己的文件站立，并可以按环境进行结构化。设置最少的项目且不需要环境意识的项目可以config.toml在其根目录使用单个文件；
4. content： 网站的所有内容都将位于此目录中；
5. data： 存储生成网站时 Hugo 可以使用的配置文件；
6. layouts： 以 .html 文件形式存储模板，这些模板指定如何将内容视图呈现到静态网站中。模板包括列表页面，主页，分类模板，partials，单页模板等；
7. static： 存储所有静态内容：图像，CSS，JavaScript 等；
8. resources： 缓存一些文件以加快生成速度；
9. themes： 当前应用的主题文件；
10. public： 生成的用于发布的网站资源。
