---
title: "Katex公式"
date: 2021-06-08T06:00:20+06:00
menu:
  sidebar:
    name: Katex公式
    identifier: katex-formula
    parent: toha-tutorial
    weight: 15
author:
  name: biubiobiu
  image: /images/author/john.png
tags: ["Latex","公式"]
categories: ["Basic"]
math: true
hero: datasets/toha/moon.jpg
---

<a href="https://katex.org/docs/supported.html" target="blank">官方文档</a>  

<a href="https://katex.org/" target="blank">线上工具</a>

## 一、基础篇

### 1. 输入公式
1. 行内公式：
格式：`$数学公式$`  例如：`$x^2=1$ : ` $x^2=1$

2. 行间公式：  
`$$`  
`数学公式`  
`$$`  
例如: `$$f(x)=\int_{-\infty}^\infty\widehat f\xi\ e^{2\pi i\xi x}\ d\xi$$`
$$f(x)=\int_{-\infty}^\infty\widehat f\xi\ e^{2\pi i\xi x}\ d\xi$$


## 二、进阶篇

### 1. 声调/变音符号
`\dot{a}, \ddot{a}, \acute{a}, \grave{a}`  
$\dot{a}, \ddot{a}, \acute{a}, \grave{a}$

`\check{a}, \breve{a}, \tilde{a}, \bar{a}`  
$\check{a}, \breve{a}, \tilde{a}, \bar{a}$

`\hat{a}, \widehat{a}, \vec{a}, \tilde{a}, \widetilde{a}`  
$\hat{a}, \widehat{a}, \vec{a}, \tilde{a}, \widetilde{a}$

`a', a'' `  
$a', a''$

### 2. 标准函数

  1. 指数/上下标  
      - 指数：`\exp_a b = a^b, \exp b = e^b, 10^m`  
    $\exp_a b = a^b, \exp b = e^b, 10^m$  
      - 前置上下标：`{}_1^2\!X_3^4`  
    ${}_1^2\!X_3^4$  
      - 导数：`x', \dot{x}, \ddot{y}`  
    $x', \dot{x}, \ddot{y}$

  2. 对数  
`\ln c, \lg d = \log e, \log_{10} f`  
$\ln c, \lg d = \log e, \log_{10} f$

  3. 三角函数  
     - `\sin a, \cos b, \tan c, \cot d, \sec e, \csc f`  
     $\sin{a}, \cos b, \tan c, \cot d, \sec e, \csc f$  
     - `\arcsin a, \arccos b, \arctan c`  
     $\arcsin a, \arccos b, \arctan c$

  4. 绝对值  
     - `\left\vert` s `\right\vert`  
     $\left\vert s \right\vert$
     - `\lVert z \rVert`  
     $\lVert z \rVert$

  5. 最大值/最小值  
`\min(x,y), \max(x,y)`  
$\min(x,y), \max(x,y)$

### 3. 界限/极限
`\min x, \max y, \inf s, \sup t, `  
$\min x, \max y, \inf s, \sup t$

`\lim_{x \to \infty} \frac{1}{n(n+1)}`  
$\lim_{x \to \infty} \frac{1}{n(n+1)}$

### 4. 微分/导数
`dt, \mathrm{d}t, \partial t, \nabla\psi`  
$dt, \mathrm{d}t, \partial t, \nabla\psi$

`dy/dx, \mathrm{d}y/\mathrm{d}x`   
$dy/dx, \mathrm{d}y/\mathrm{d}x$

`\frac{dy}{dx}, \frac{\mathrm{d}y}{\mathrm{d}x}, \frac{\partial^2}{\partial x_1\partial x_2}y`  
$\frac{dy}{dx}, \frac{\mathrm{d}y}{\mathrm{d}x}, \frac{\partial^2}{\partial x_1\partial x_2}y$

`\prime, \backprime, f^\prime, f', f'', f^{(3)}, \dot y, \ddot y`  
$\prime, \backprime, f^\prime, f', f'', f^{(3)}, \dot y, \ddot y$


### 5. 根号/分数
`\surd, \sqrt{2}, \sqrt[n]{}, \sqrt[3]{\frac{x^3+y^3}{2}}`  
$\surd, \sqrt{2}, \sqrt[n]{}, \sqrt[3]{\frac{x^3+y^3}{2}}$ 


### 6. 运算符

|||||
|------------------|-------------------------|--------------------------|--------------|
| $\sum$ `\sum`    | $\prod$ `\prod`         | $\bigotimes$ `\bigotimes`| $\bigvee$ `\bigvee`
| $\int$ `\int`    | $\coprod$ `\coprod`     | $\bigoplus$ `\bigoplus`  | $\bigwedge$ `\bigwedge`
| $\iint$ `\iint`  | $\intop$ `\intop`       | $\bigodot$ `\bigodot`    | $\bigcap$ `\bigcap`
| $\iiint$ `\iiint`| $\smallint$ `\smallint` | $\biguplus$ `\biguplus`  | $\bigcup$ `\bigcup`
| $\oint$ `\oint`  | $\oiint$ `\oiint`       | $\oiiint$ `\oiiint`      | $\bigsqcup$ `\bigsqcup`
|||||
| $+$ `+`| $\cdot$ `\cdot`  | $\gtrdot$ `\gtrdot`| $x \pmod a$ `x \pmod a`|
| $-$ `-`| $\cdotp$ `\cdotp` | $\intercal$ `\intercal` | $x \pod a$ `x \pod a` |
| $/$ `/`| $\centerdot$ `\centerdot`| $\land$ `\land`  | $\rhd$ `\rhd` |
| $*$ `*`| $\circ$ `\circ`  | $\leftthreetimes$ `\leftthreetimes` | $\rightthreetimes$ `\rightthreetimes` |
| $\amalg$ `\amalg` | $\circledast$ `\circledast`  | $\ldotp$ `\ldotp` | $\rtimes$ `\rtimes` |
| $\And$ `\And`| $\circledcirc$ `\circledcirc` | $\lor$ `\lor`| $\setminus$ `\setminus`  |
| $\ast$ `\ast`| $\circleddash$ `\circleddash` | $\lessdot$ `\lessdot`  | $\smallsetminus$ `\smallsetminus`|
| $\barwedge$ `\barwedge` | $\Cup$ `\Cup`| $\lhd$ `\lhd`| $\sqcap$ `\sqcap`  |
| $\bigcirc$ `\bigcirc`  | $\cup$ `\cup`| $\ltimes$ `\ltimes`| $\sqcup$ `\sqcup`  |
| $\bmod$ `\bmod`  | $\curlyvee$ `\curlyvee` | $x \mod a$ `x\mod a`| $\times$ `\times`  |
| $\boxdot$ `\boxdot`| $\curlywedge$ `\curlywedge`  | $\mp$ `\mp` | $\unlhd$ `\unlhd`  |
| $\boxminus$ `\boxminus` | $\div$ `\div`| $\odot$ `\odot`  | $\unrhd$ `\unrhd`  |
| $\boxplus$ `\boxplus`  | $\divideontimes$ `\divideontimes`  | $\ominus$ `\ominus`| $\uplus$ `\uplus`  |
| $\boxtimes$ `\boxtimes` | $\dotplus$ `\dotplus`  | $\oplus$ `\oplus` | $\vee$ `\vee` |
| $\bullet$ `\bullet`| $\doublebarwedge$ `\doublebarwedge` | $\otimes$ `\otimes`| $\veebar$ `\veebar` |
| $\Cap$ `\Cap`| $\doublecap$ `\doublecap`| $\oslash$ `\oslash`| $\wedge$ `\wedge`  |
| $\cap$ `\cap`| $\doublecup$ `\doublecup`| $\pm$ `\pm` or `\plusmn` | $\wr$ `\wr`  |

直接输入: $∫ ∬ ∭ ∮ ∏ ∐ ∑ ⋀ ⋁ ⋂ ⋃ ⨀ ⨁ ⨂ ⨄ ⨆$ ∯ ∰  
$+ - / * ⋅ ± × ÷ ∓ ∔ ∧ ∨ ∩ ∪ ≀ ⊎ ⊓ ⊔ ⊕ ⊖ ⊗ ⊘ ⊙ ⊚ ⊛ ⊝ ◯$

### 7. 关系符号
`=, \ne, \neq, \equiv, \not\equiv`  
$=, \ne, \neq, \equiv, \not\equiv$

`\doteq, \doteqdot, \overset{\underset{\mathrm{def}}{}}{=}`  
$\doteq, \doteqdot, \overset{\underset{\mathrm{def}}{}}{=}$

`\sim, \nsim, \backsim, \thicksim, \simeq, \backsimeq, \eqsim, \cong, \ncong`  
$\sim, \nsim, \backsim, \thicksim, \simeq, \backsimeq, \eqsim, \cong, \ncong$

`\approx, \thickapprox, \approxeq, \asymp, \propto, \varpropto`  
$\approx, \thickapprox, \approxeq, \asymp, \propto, \varpropto$

`<, \nless, \ll, \not\ll, \lll, \not\lll, \lessdot`  
$<, \nless, \ll, \not\ll, \lll, \not\lll, \lessdot$

`>, \ngtr, \gg, \not\gg, \ggg, \not\ggg, \gtrdot`  
$>, \ngtr, \gg, \not\gg, \ggg, \not\ggg, \gtrdot$

`\le, \leq, \lneq, \leqq, \nleq, \nleqq, \lneqq, \lvertneqq`  
$\le, \leq, \lneq, \leqq, \nleq, \nleqq, \lneqq, \lvertneqq$

`\ge, \geq, \gneq, \geqq, \ngeq, \ngeqq, \gneqq, \gvertneqq`  
$\ge, \geq, \gneq, \geqq, \ngeq, \ngeqq, \gneqq, \gvertneqq$

`\leqslant, \nleqslant, \eqslantless`  
$\leqslant, \nleqslant, \eqslantless$

`\geqslant, \ngeqslant, \eqslantgtr`  
$\geqslant, \ngeqslant, \eqslantgtr$

`\lesssim, \lnsim, \lessapprox, \lnapprox`  
$\lesssim, \lnsim, \lessapprox, \lnapprox$

`\gtrsim, \gnsim, \gtrapprox, \gnapprox`  
$\gtrsim, \gnsim, \gtrapprox, \gnapprox$

### 8. 集合
`\empty \emptyset, \varnothing`  
$\empty, \emptyset, \varnothing$

`\in, \notin \not\in, \ni, \not\ni`  
$\in, \notin \not\in, \ni, \not\ni$

`\cap, \Cap, \sqcap, \bigcap`  
$\cap, \Cap, \sqcap, \bigcap$

`\cup, \Cup, \sqcup, \bigcup, \bigsqcup, \uplus, \biguplus`  
$\cup, \Cup, \sqcup, \bigcup, \bigsqcup, \uplus, \biguplus$

`\subset, \Subset, \sqsubset`  
$\subset, \Subset, \sqsubset$

`\supset, \Supset, \sqsupset`  
$\supset, \Supset, \sqsupset$

`\subseteq, \nsubseteq, \subsetneq, \varsubsetneq, \sqsubseteq`  
$\subseteq, \nsubseteq, \subsetneq, \varsubsetneq, \sqsubseteq$

`\supseteq, \nsupseteq, \supsetneq, \varsupsetneq, \sqsupseteq`  
$\supseteq, \nsupseteq, \supsetneq, \varsupsetneq, \sqsupseteq$

`\subseteqq, \nsubseteqq, \subsetneqq, \varsubsetneqq`  
$\subseteqq, \nsubseteqq, \subsetneqq, \varsubsetneqq$

`\supseteqq, \nsupseteqq, \supsetneqq, \varsupsetneqq`  
$\supseteqq, \nsupseteqq, \supsetneqq, \varsupsetneqq$



### 9. 几何符号

||||
|:----------|:----------|:----------|
|`% comment`|$\dots$ `\dots`|$\KaTeX$ `\KaTeX`
|$\\%$ `\\%`|$\cdots$ `\cdots`|$\LaTeX$ `\LaTeX`
|$\\#$ `\\#`|$\ddots$ `\ddots`|$\TeX$ `\TeX`
|$\\&$ `\\&`|$\ldots$ `\ldots`|$\nabla$ `\nabla`
|$\\_$ `\\_`|$\vdots$ `\vdots`|$\infty$ `\infty`
|$\text{\textunderscore}$ `\text{\textunderscore}`|$\dotsb$ `\dotsb`|$\infin$ `\infin`
|$\text{--}$ `\text{--}`|$\dotsc$ `\dotsc`|$\checkmark$ `\checkmark`
|$\text{\textendash}$ `\text{\textendash}`|$\dotsi$ `\dotsi`|$\dag$ `\dag`
|$\text{---}$ `\text{---}`|$\dotsm$ `\dotsm`|$\dagger$ `\dagger`
|$\text{\textemdash}$ `\text{\textemdash}`|$\dotso$ `\dotso`|$\text{\textdagger}$ `\text{\textdagger}`
|$\text{\textasciitilde}$ `\text{\textasciitilde}`|$\sdot$ `\sdot`|$\ddag$ `\ddag`
|$\text{\textasciicircum}$ `\text{\textasciicircum}`|$\mathellipsis$ `\mathellipsis`|$\ddagger$ `\ddagger`
||$\text{\textellipsis}$ `\text{\textellipsis}`|$\text{\textdaggerdbl}$ `\text{\textdaggerdbl}`
|$\text{\textquoteleft}$ `text{\textquoteleft}`|$\Box$ `\Box`|$\Dagger$ `\Dagger`
|$\lq$ `\lq`|$\square$ `\square`|$\angle$ `\angle`
|$\text{\textquoteright}$ `\text{\textquoteright}`|$\blacksquare$ `\blacksquare`|$\measuredangle$ `\measuredangle`
|$\rq$ `\rq`|$\triangle$ `\triangle`|$\sphericalangle$ `\sphericalangle`
|$\text{\textquotedblleft}$ `\text{\textquotedblleft}`|$\triangledown$ `\triangledown`|$\top$ `\top`
|$"$ `"`|$\triangleleft$ `\triangleleft`|$\bot$ `\bot`
|$\text{\textquotedblright}$ `\text{\textquotedblright}`|$\triangleright$ `\triangleright`|$\$$ `\$`
|$\colon$ `\colon`|$\bigtriangledown$ `\bigtriangledown`|$\text{\textdollar}$ `\text{\textdollar}`
|$\backprime$ `\backprime`|$\bigtriangleup$ `\bigtriangleup`|$\pounds$ `\pounds`
|$\prime$ `\prime`|$\blacktriangle$ `\blacktriangle`|$\mathsterling$ `\mathsterling`
|$\text{\textless}$ `\text{\textless}`|$\blacktriangledown$ `\blacktriangledown`|$\text{\textsterling}$ `\text{\textsterling}`
|$\text{\textgreater}$ `\text{\textgreater}`|$\blacktriangleleft$ `\blacktriangleleft`|$\yen$ `\yen`
|$\text{\textbar}$ `\text{\textbar}`|$\blacktriangleright$ `\blacktriangleright`|$\surd$ `\surd`
|$\text{\textbardbl}$ `\text{\textbardbl}`|$\diamond$ `\diamond`|$\degree$ `\degree`
|$\text{\textbraceleft}$ `\text{\textbraceleft}`|$\Diamond$ `\Diamond`|$\text{\textdegree}$ `\text{\textdegree}`
|$\text{\textbraceright}$ `\text{\textbraceright}`|$\lozenge$ `\lozenge`|$\mho$ `\mho`
|$\text{\textbackslash}$ `\text{\textbackslash}`|$\blacklozenge$ `\blacklozenge`|$\diagdown$ `\diagdown`
|$\text{\P}$ `\text{\P}` or `\P`|$\star$ `\star`|$\diagup$ `\diagup`
|$\text{\S}$ `\text{\S}` or `\S`|$\bigstar$ `\bigstar`|$\flat$ `\flat`
|$\text{\sect}$ `\text{\sect}`|$\clubsuit$ `\clubsuit`|$\natural$ `\natural`
|$\copyright$ `\copyright`|$\clubs$ `\clubs`|$\sharp$ `\sharp`
|$\circledR$ `\circledR`|$\diamondsuit$ `\diamondsuit`|$\heartsuit$ `\heartsuit`
|$\text{\textregistered}$ `\text{\textregistered}`|$\diamonds$ `\diamonds`|$\hearts$ `\hearts`
|$\circledS$ `\circledS`|$\spadesuit$ `\spadesuit`|$\spades$ `\spades`
|$\text{\textcircled a}$ `\text{\textcircled a}`|$\maltese$ `\maltese`||

Direct Input: § ¶ $ £ ¥ ∇ ∞ · ∠ ∡ ∢ ♠ ♡ ♢ ♣ ♭ ♮ ♯ ✓ …  ⋮  ⋯  ⋱  !$ ‼ ⦵

### 10. 逻辑符号
`\forall, \exists, \nexists`  
$\forall, \exists, \nexists$

`\therefore, \because, \And`  
$\therefore, \because, \And$

`\lor, \vee, \curlyvee, \bigvee`  
$\lor, \vee, \curlyvee, \bigvee$

`\land, \wedge, \curlywedge, \bigwedge`  
$\land, \wedge, \curlywedge, \bigwedge$

`\bar{q}, \bar{abc}, \overline{q}, \overline{abc}, \lnot \neg, \not\operatorname{R}, \bot, \top`  
$\bar{q}, \bar{abc}, \overline{q}, \overline{abc}, \lnot \neg, \not\operatorname{R}, \bot, \top$

### 11. 箭头

||||
|:----------|:----------|:----------|
|$\circlearrowleft$ `\circlearrowleft`|$\leftharpoonup$ `\leftharpoonup`|$\rArr$ `\rArr`
|$\circlearrowright$ `\circlearrowright`|$\leftleftarrows$ `\leftleftarrows`|$\rarr$ `\rarr`
|$\curvearrowleft$ `\curvearrowleft`|$\leftrightarrow$ `\leftrightarrow`|$\restriction$ `\restriction`
|$\curvearrowright$ `\curvearrowright`|$\Leftrightarrow$ `\Leftrightarrow`|$\rightarrow$ `\rightarrow`
|$\Darr$ `\Darr`|$\leftrightarrows$ `\leftrightarrows`|$\Rightarrow$ `\Rightarrow`
|$\dArr$ `\dArr`|$\leftrightharpoons$ `\leftrightharpoons`|$\rightarrowtail$ `\rightarrowtail`
|$\darr$ `\darr`|$\leftrightsquigarrow$ `\leftrightsquigarrow`|$\rightharpoondown$ `\rightharpoondown`
|$\dashleftarrow$ `\dashleftarrow`|$\Lleftarrow$ `\Lleftarrow`|$\rightharpoonup$ `\rightharpoonup`
|$\dashrightarrow$ `\dashrightarrow`|$\longleftarrow$ `\longleftarrow`|$\rightleftarrows$ `\rightleftarrows`
|$\downarrow$ `\downarrow`|$\Longleftarrow$ `\Longleftarrow`|$\rightleftharpoons$ `\rightleftharpoons`
|$\Downarrow$ `\Downarrow`|$\longleftrightarrow$ `\longleftrightarrow`|$\rightrightarrows$ `\rightrightarrows`
|$\downdownarrows$ `\downdownarrows`|$\Longleftrightarrow$ `\Longleftrightarrow`|$\rightsquigarrow$ `\rightsquigarrow`
|$\downharpoonleft$ `\downharpoonleft`|$\longmapsto$ `\longmapsto`|$\Rrightarrow$ `\Rrightarrow`
|$\downharpoonright$ `\downharpoonright`|$\longrightarrow$ `\longrightarrow`|$\Rsh$ `\Rsh`
|$\gets$ `\gets`|$\Longrightarrow$ `\Longrightarrow`|$\searrow$ `\searrow`
|$\Harr$ `\Harr`|$\looparrowleft$ `\looparrowleft`|$\swarrow$ `\swarrow`
|$\hArr$ `\hArr`|$\looparrowright$ `\looparrowright`|$\to$ `\to`
|$\harr$ `\harr`|$\Lrarr$ `\Lrarr`|$\twoheadleftarrow$ `\twoheadleftarrow`
|$\hookleftarrow$ `\hookleftarrow`|$\lrArr$ `\lrArr`|$\twoheadrightarrow$ `\twoheadrightarrow`
|$\hookrightarrow$ `\hookrightarrow`|$\lrarr$ `\lrarr`|$\Uarr$ `\Uarr`
|$\iff$ `\iff`|$\Lsh$ `\Lsh`|$\uArr$ `\uArr`
|$\impliedby$ `\impliedby`|$\mapsto$ `\mapsto`|$\uarr$ `\uarr`
|$\implies$ `\implies`|$\nearrow$ `\nearrow`|$\uparrow$ `\uparrow`
|$\Larr$ `\Larr`|$\nleftarrow$ `\nleftarrow`|$\Uparrow$ `\Uparrow`
|$\lArr$ `\lArr`|$\nLeftarrow$ `\nLeftarrow`|$\updownarrow$ `\updownarrow`
|$\larr$ `\larr`|$\nleftrightarrow$ `\nleftrightarrow`|$\Updownarrow$ `\Updownarrow`
|$\leadsto$ `\leadsto`|$\nLeftrightarrow$ `\nLeftrightarrow`|$\upharpoonleft$ `\upharpoonleft`
|$\leftarrow$ `\leftarrow`|$\nrightarrow$ `\nrightarrow`|$\upharpoonright$ `\upharpoonright`
|$\Leftarrow$ `\Leftarrow`|$\nRightarrow$ `\nRightarrow`|$\upuparrows$ `\upuparrows`
|$\leftarrowtail$ `\leftarrowtail`|$\nwarrow$ `\nwarrow`|
|$\leftharpoondown$ `\leftharpoondown`|$\Rarr$ `\Rarr`|
|$\xleftarrow{abc}$ `\xleftarrow{abc}`                |$\xrightarrow[under]{over}$ `\xrightarrow[under]{over}`
|$\xLeftarrow{abc}$ `\xLeftarrow{abc}`                |$\xRightarrow{abc}$ `\xRightarrow{abc}`
|$\xleftrightarrow{abc}$ `\xleftrightarrow{abc}`      |$\xLeftrightarrow{abc}$ `\xLeftrightarrow{abc}`
|$\xhookleftarrow{abc}$ `\xhookleftarrow{abc}`        |$\xhookrightarrow{abc}$ `\xhookrightarrow{abc}`
|$\xtwoheadleftarrow{abc}$ `\xtwoheadleftarrow{abc}`  |$\xtwoheadrightarrow{abc}$ `\xtwoheadrightarrow{abc}`
|$\xleftharpoonup{abc}$ `\xleftharpoonup{abc}`        |$\xrightharpoonup{abc}$ `\xrightharpoonup{abc}`
|$\xleftharpoondown{abc}$ `\xleftharpoondown{abc}`    |$\xrightharpoondown{abc}$ `\xrightharpoondown{abc}`
|$\xleftrightharpoons{abc}$ `\xleftrightharpoons{abc}`|$\xrightleftharpoons{abc}$ `\xrightleftharpoons{abc}`
|$\xtofrom{abc}$ `\xtofrom{abc}`                      |$\xmapsto{abc}$ `\xmapsto{abc}`
|$\xlongequal{abc}$ `\xlongequal{abc}`

Direct Input: $← ↑ → ↓ ↔ ↕ ↖ ↗ ↘ ↙ ↚ ↛ ↞ ↠ ↢ ↣ ↦ ↩ ↪ ↫ ↬ ↭ ↮ ↰ ↱↶ ↷ ↺ ↻ ↼ ↽ ↾ ↾ ↿ ⇀ ⇁ ⇂ ⇃ ⇄ ⇆ ⇇ ⇈ ⇉ ⇊ ⇋ ⇌⇍ ⇎ ⇏ ⇐ ⇑ ⇒ ⇓ ⇔ ⇕ ⇚ ⇛ ⇝ ⇠ ⇢ ⟵ ⟶ ⟷ ⟸ ⟹ ⟺ ⟼$ ↽

### 12. 上下标

|功能|语法|效果|
|:-----------------|:-------------------------------------------------|:-----------------------|
|上标|`a^2`|$a^2$|
|下标|`a_2`|$a_2$|
|组合|`a^{2+2}`|$a^{2+2}$|
|结合上下标|`x_2^3`|$x_2^3$|
|前置上下标|`{}_1^2\!X_3^4`|${}_1^2\!X_3^4$|
|导数|`x', \dot{x}, \ddot{x}`|$x', \dot{x}, \ddot{x}$|
|向量|`\vec{c}, \overleftarrow{a b}, \overrightarrow{c d}, \overleftrightarrow{a b}`|$\vec{c}, \overleftarrow{a b}, \overrightarrow{c d}, \overleftrightarrow{a b}$|
|弧线|`\widehat{e f g}, \overset{\frown} {AB}`|$\widehat{e f g}, \overset{\frown} {AB}$|
|上/下划线|`\overline{h i j}, \underline{k l m}`|$\overline{h i j}, \underline{k l m}$|
|上括号|`\overbrace{1+2+\cdots+100}`<br>`\begin{matrix} 5050 \\ \overbrace{ 1+2+\cdots+100 } \end{matrix}`|$\overbrace{1+2+\cdots+100}$<br>$\begin{matrix} 5050 \\ \overbrace{ 1+2+\cdots+100 } \end{matrix}$|
|下括号|`\underbrace{a+b+\cdots+z}`<br>`\begin{matrix} \underbrace{ a+b+\cdots+z } \\ 26 \end{matrix}`|$\underbrace{a+b+\cdots+z}$<br>$\begin{matrix} \underbrace{ a+b+\cdots+z } \\ 26 \end{matrix}$|
|累加|`\sum_{k=1}^N k^2`<br>`\begin{matrix} \sum_{k=1}^N k^2 \end{matrix}`|$\sum_{k=1}^N k^2$<br>$\begin{matrix} \sum_{k=1}^N k^2 \end{matrix}$|
|累加-格式|`\displaystyle\sum_{\substack{0<i<m\\\\0<j<n}}`</br>`\textstyle\sum_{\substack{0<i<m\\\\0<j<n}}`|$\displaystyle\sum_{\substack{0<i<m\\\\0<j<n}}$</br>$\textstyle\sum_{\substack{0<i<m\\\\0<j<n}}$|
|累乘|`\prod_{i=1}^N x_i`|$\prod_{i=1}^N x_i$|
|上积|`\coprod_{i=1}^N x_i`|$\coprod_{i=1}^N x_i$|
|极限|`\lim_{n \to \infty}x_n`|$\lim_{n \to \infty}x_n$|
|极限-格式|`\lim\limits_{n \to \infty}x_n`</br>`\lim\nolimits_{n \to \infty}x_n`|$\lim\limits_{n \to \infty}x_n$<br>$\lim\nolimits_{n \to \infty}x_n$|
|积分|`\int_{-N}^{N} e^x\, {\rm d}x`|$\int_{-N}^{N} e^x\, {\rm d}x$|
|双重积分|`\iint_{D}^{W} \, \mathrm{d}x\,\mathrm{d}y`|$\iint_{D}^{W} \, \mathrm{d}x\,\mathrm{d}y$|
|三重积分|`\iiint_{E}^{V} \, \mathrm{d}x\,\mathrm{d}y\,\mathrm{d}z`|$\iiint_{E}^{V} \, \mathrm{d}x\,\mathrm{d}y\,\mathrm{d}z$|
|闭合|`\oint_{C} x^3\, \mathrm{d}x + 4y^2\, \mathrm{d}y`|$\oint_{C} x^3\, \mathrm{d}x + 4y^2\, \mathrm{d}y$|
|交集|`\bigcap_1^{n} p`|$\bigcap_1^{n} p$|
|并集|`\bigcup_1^{k} p`|$\bigcup_1^{k} p$|

### 13. 分式
通常使用`\frac {分子} {分母}` 来生成一个分数。如果分式比较复杂，也可以使用 `分子 \over 分母`

|功能|语法|效果|
|:--|:--|:--|
|分数|`\frac{2}{4}=0.5`|$\frac{2}{4}=0.5$|
|小型分数|`\tfrac{2}{4} = 0.5`|$\tfrac{2}{4} = 0.5$|
|连分式|`\cfrac{2}{c + \cfrac{2}{d + \cfrac{2}{4}}} = a`|$\cfrac{2}{c + \cfrac{2}{d + \cfrac{2}{4}}} = a$|
||`\binom{n}{k}, \dbinom{n}{k}, \tbinom{n}{k}`|$\binom{n}{k}, \dbinom{n}{k}, \tbinom{n}{k}$|
||`{n \choose k}, {n\brace k}, {n\brack k}`|${n \choose k}, {n\brace k}, {n\brack k}$|
|二项式系数|`\dbinom{n}{r}=\binom{n}{n-r}=\mathrm{C}_n^r=\mathrm{C}_n^{n-r}`|$\dbinom{n}{r}=\binom{n}{n-r}=\mathrm{C}_n^r=\mathrm{C}_n^{n-r}$|
|小型二项式系数|`\tbinom{n}{r}=\tbinom{n}{n-r}=\mathrm{C}_n^r=\mathrm{C}_n^{n-r}`|$\tbinom{n}{r}=\tbinom{n}{n-r}=\mathrm{C}_n^r=\mathrm{C}_n^{n-r}$|
|大型二项式系数|`\binom{n}{r}=\dbinom{n}{n-r}=\mathrm{C}_n^r=\mathrm{C}_n^{n-r}`|$\binom{n}{r}=\dbinom{n}{n-r}=\mathrm{C}_n^r=\mathrm{C}_n^{n-r}$|


### 14. 矩阵


|效果|语法|效果|语法|
|:---------------------|:---------------------|:---------------------|:---------------------------------|
|$\begin{matrix} a & b \\\ c & d \end{matrix}$ | `\begin{matrix}`<br>&nbsp;&nbsp;&nbsp;`a & b \\\`<br>&nbsp;&nbsp;&nbsp;`c & d`<br>`\end{matrix}` |$\begin{array}{cc}a & b\\\c & d\end{array}$ | `\begin{array}{cc}`<br>&nbsp;&nbsp;&nbsp;`a & b \\\`<br>&nbsp;&nbsp;&nbsp;`c & d`<br>`\end{array}`
|$\begin{pmatrix} a & b \\\ c & d \end{pmatrix}$ |`\begin{pmatrix}`<br>&nbsp;&nbsp;&nbsp;`a & b \\\`<br>&nbsp;&nbsp;&nbsp;`c & d`<br>`\end{pmatrix}` |$\begin{bmatrix} a & b \\\ c & d \end{bmatrix}$ | `\begin{bmatrix}`<br>&nbsp;&nbsp;&nbsp;`a & b \\\`<br>&nbsp;&nbsp;&nbsp;`c & d`<br>`\end{bmatrix}`
|$\begin{vmatrix} a & b \\\ c & d \end{vmatrix}$ |`\begin{vmatrix}`<br>&nbsp;&nbsp;&nbsp;`a & b \\\`<br>&nbsp;&nbsp;&nbsp;`c & d`<br>`\end{vmatrix}` |$\begin{Vmatrix} a & b \\\ c & d \end{Vmatrix}$ |`\begin{Vmatrix}`<br>&nbsp;&nbsp;&nbsp;`a & b \\\`<br>&nbsp;&nbsp;&nbsp;`c & d`<br>`\end{Vmatrix}`
|$\begin{Bmatrix} a & b \\\ c & d \end{Bmatrix}$ |`\begin{Bmatrix}`<br>&nbsp;&nbsp;&nbsp;`a & b \\\`<br>&nbsp;&nbsp;&nbsp;`c & d`<br>`\end{Bmatrix}`|$\def\arraystretch{1.5}\begin{array}{c:c:c} a & b & c \\\ \hline d & e & f \\\ \hdashline g & h & i \end{array}$|`\def\arraystretch{1.5}`<br>&nbsp;&nbsp;&nbsp;`\begin{array}{c:c:c}`<br>&nbsp;&nbsp;&nbsp;`a & b & c \\\ \hline`<br>&nbsp;&nbsp;&nbsp;`d & e & f \\\`<br>&nbsp;&nbsp;&nbsp;`\hdashline`<br>&nbsp;&nbsp;&nbsp;`g & h & i`<br>`\end{array}`
|$x = \begin{cases} a &\text{if } b \\\ c &\text{if } d \end{cases}$ |`x = \begin{cases}`<br>&nbsp;&nbsp;&nbsp;`a &\text{if } b  \\\`<br>&nbsp;&nbsp;&nbsp;`c &\text{if } d`<br>`\end{cases}`|无效呢？|`\begin{rcases}`<br>&nbsp;&nbsp;&nbsp;`a &\text{if } b  \\\`<br>&nbsp;&nbsp;&nbsp;`c &\text{if } d`<br>`\end{rcases}⇒…`|
|$\begin{smallmatrix} a & b \\\ c & d \end{smallmatrix}$ | `\begin{smallmatrix}`<br>&nbsp;&nbsp;&nbsp;`a & b \\\`<br>&nbsp;&nbsp;&nbsp;`c & d`<br>`\end{smallmatrix}` |$$\sum_{\begin{subarray}{l} i\in\Lambda\\\  0<j<n\end{subarray}}$$ | `\sum_{`<br>`\begin{subarray}{l}`<br>&nbsp;&nbsp;&nbsp;`i\in\Lambda\\\`<br>&nbsp;&nbsp;&nbsp;`0<j<n`<br>`\end{subarray}}`|


### 15. 希腊字母

直接输入: $Α Β Γ Δ Ε Ζ Η Θ Ι \allowbreak Κ Λ Μ Ν Ξ Ο Π Ρ Σ Τ Υ Φ Χ Ψ Ω$
$\allowbreak α β γ δ ϵ ζ η θ ι κ λ μ ν ξ o π \allowbreak ρ σ τ υ ϕ χ ψ ω ε ϑ ϖ ϱ ς φ ϝ$

|||||
|---------------|-------------|-------------|---------------|
| $\Alpha$ `\Alpha` | $\Beta$ `\Beta` | $\Gamma$ `\Gamma`| $\Delta$ `\Delta`
| $\Epsilon$ `\Epsilon` | $\Zeta$ `\Zeta` | $\Eta$ `\Eta` | $\Theta$ `\Theta`
| $\Iota$ `\Iota` | $\Kappa$ `\Kappa` | $\Lambda$ `\Lambda` | $\Mu$ `\Mu`
| $\Nu$ `\Nu` | $\Xi$ `\Xi` | $\Omicron$ `\Omicron` | $\Pi$ `\Pi`
| $\Rho$ `\Rho` | $\Sigma$ `\Sigma` | $\Tau$ `\Tau` | $\Upsilon$ `\Upsilon`
| $\Phi$ `\Phi` | $\Chi$ `\Chi` | $\Psi$ `\Psi` | $\Omega$ `\Omega`
| $\varGamma$ `\varGamma`| $\varDelta$ `\varDelta` | $\varTheta$ `\varTheta` | $\varLambda$ `\varLambda`  |
| $\varXi$ `\varXi`| $\varPi$ `\varPi` | $\varSigma$ `\varSigma` | $\varUpsilon$ `\varUpsilon` |
| $\varPhi$ `\varPhi`  | $\varPsi$ `\varPsi`| $\varOmega$ `\varOmega` ||
| $\alpha$ `\alpha`| $\beta$ `\beta`  | $\gamma$ `\gamma` | $\delta$ `\delta`|
| $\epsilon$ `\epsilon` | $\zeta$ `\zeta`  | $\eta$ `\eta`| $\theta$ `\theta`|
| $\iota$ `\iota` | $\kappa$ `\kappa` | $\lambda$ `\lambda`| $\mu$ `\mu`|
| $\nu$ `\nu`| $\xi$ `\xi` | $\omicron$ `\omicron`  | $\pi$ `\pi`|
| $\rho$ `\rho`  | $\sigma$ `\sigma` | $\tau$ `\tau`| $\upsilon$ `\upsilon` |
| $\phi$ `\phi`  | $\chi$ `\chi`| $\psi$ `\psi`| $\omega$ `\omega`|
| $\varepsilon$ `\varepsilon` | $\varkappa$ `\varkappa` | $\vartheta$ `\vartheta` | $\thetasym$ `\thetasym`
| $\varpi$ `\varpi`| $\varrho$ `\varrho`  | $\varsigma$ `\varsigma` | $\varphi$ `\varphi`
| $\digamma $ `\digamma`

**Other Letters**

||||||
|:----------|:----------|:----------|:----------|:----------|
|$\imath$ `\imath`|$\nabla$ `\nabla`|$\Im$ `\Im`|$\Reals$ `\Reals`|$\text{\OE}$ `\text{\OE}`
|$\jmath$ `\jmath`|$\partial$ `\partial`|$\image$ `\image`|$\wp$ `\wp`|$\text{\o}$ `\text{\o}`
|$\aleph$ `\aleph`|$\Game$ `\Game`|$\Bbbk$ `\Bbbk`|$\weierp$ `\weierp`|$\text{\O}$ `\text{\O}`
|$\alef$ `\alef`|$\Finv$ `\Finv`|$\N$ `\N`|$\Z$ `\Z`|$\text{\ss}$ `\text{\ss}`
|$\alefsym$ `\alefsym`|$\cnums$ `\cnums`|$\natnums$ `\natnums`|$\text{\aa}$ `\text{\aa}`|$\text{\i}$ `\text{\i}`
|$\beth$ `\beth`|$\Complex$ `\Complex`|$\R$ `\R`|$\text{\AA}$ `\text{\AA}`|$\text{\j}$ `\text{\j}`
|$\gimel$ `\gimel`|$\ell$ `\ell`|$\Re$ `\Re`|$\text{\ae}$ `\text{\ae}`
|$\daleth$ `\daleth`|$\hbar$ `\hbar`|$\real$ `\real`|$\text{\AE}$ `\text{\AE}`
|$\eth$ `\eth`|$\hslash$ `\hslash`|$\reals$ `\reals`|$\text{\oe}$ `\text{\oe}`

直接输入: $∂ ∇ ℑ Ⅎ ℵ ℶ ℷ ℸ ⅁ ℏ ð − ∗$
ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖÙÚÛÜÝÞßàáâãäåçèéêëìíîïðñòóôöùúûüýþÿ

### 16. 字体大小

|||
|:----------------------|:-----
|$\Huge AB$ `\Huge AB`  |$\normalsize AB$ `\normalsize AB`
|$\huge AB$ `\huge AB`  |$\small AB$ `\small AB`
|$\LARGE AB$ `\LARGE AB`|$\footnotesize AB$ `\footnotesize AB`
|$\Large AB$ `\Large AB`|$\scriptsize AB$ `\scriptsize AB`
|$\large AB$ `\large AB`|$\tiny AB$ `\tiny AB`

### 17. 颜色

|语法|效果|
|:---------------|:------------------|
|`\color{blue} F=ma`|$\color{blue} F=ma$|
|`\textcolor{blue}{F=ma}`|$\textcolor{blue}{F=ma}$|
|`\textcolor{#228B22}{F=ma}`|$\textcolor{#228B22}{F=ma}$|
|`\colorbox{aqua}{$F=ma$}`|$\colorbox{aqua}{$F=ma$}$|
|`\fcolorbox{red}{aqua}{$F=ma$}`|$\fcolorbox{red}{aqua}{$F=ma$}$|

### 18. 字体

||||
|:------------------------------|:------------------------------|:-----
|$\mathrm{Ab0}$ `\mathrm{Ab0}`  |$\mathbf{Ab0}$ `\mathbf{Ab0}`  |$\mathit{Ab0}$ `\mathit{Ab0}`
|$\mathnormal{Ab0}$ `\mathnormal{Ab0}`|$\textbf{Ab0}$ `\textbf{Ab0}`  |$\textit{Ab0}$ `\textit{Ab0}`
|$\textrm{Ab0}$ `\textrm{Ab0}`  |$\bf Ab0$ `\bf Ab0`            |$\it Ab0$ `\it Ab0`
|$\rm Ab0$ `\rm Ab0`            |$\bold{Ab0}$ `\bold{Ab0}`      |$\textup{Ab0}$ `\textup{Ab0}`
|$\textnormal{Ab0}$ `\textnormal{Ab0}`|$\boldsymbol{Ab0}$ `\boldsymbol{Ab}`|$\Bbb{AB}$ `\Bbb{AB}`
|$\text{Ab0}$ `\text{Ab0}`      |$\bm{Ab0}$ `\bm{Ab0}`          |$\mathbb{AB}$ `\mathbb{AB}`
|$\mathsf{Ab0}$ `\mathsf{Ab0}`  |$\textmd{Ab0}$ `\textmd{Ab0}`  |$\frak{Ab0}$ `\frak{Ab0}`
|$\textsf{Ab0}$ `\textsf{Ab0}`  |$\mathtt{Ab0}$ `\mathtt{Ab0}`  |$\mathfrak{Ab0}$ `\mathfrak{Ab0}`
|$\sf Ab0$ `\sf Ab0`            |$\texttt{Ab0}$ `\texttt{Ab0}`  |$\mathcal{AB0}$ `\mathcal{AB0}`
|                               |$\tt Ab0$ `\tt Ab0`            |$\cal AB0$ `\cal AB0`
|                               |                               |$\mathscr{AB}$ `\mathscr{AB}`

### 19. 括号

||||||
|:-----------------------------------|:---------------------------------------|:----------|:-------------------------------------------------------|:-----
|$(~)$ `( )` |$\lparen~\rparen$ `\lparen`<br>$~~~~$`\rparen`|$⌈~⌉$ `⌈ ⌉`|$\lceil~\rceil$ `\lceil`<br>$~~~~~$`\rceil`  |$\uparrow$ `\uparrow`
|$[~]$ `[ ]` |$\lbrack~\rbrack$ `\lbrack`<br>$~~~~$`\rbrack`|$⌊~⌋$ `⌊ ⌋`|$\lfloor~\rfloor$ `\lfloor`<br>$~~~~~$`\rfloor` |$\downarrow$ `\downarrow`
|$\{ \}$ `\{ \}`|$\lbrace \rbrace$ `\lbrace`<br>$~~~~$`\rbrace`|$⎰⎱$ `⎰⎱`  |$\lmoustache \rmoustache$ `\lmoustache`<br>$~~~~$`\rmoustache`|$\updownarrow$ `\updownarrow`
|$⟨~⟩$ `⟨ ⟩` |$\langle~\rangle$ `\langle`<br>$~~~~$`\rangle`|$⟮~⟯$ `⟮ ⟯`|$\lgroup~\rgroup$ `\lgroup`<br>$~~~~~$`\rgroup` |$\Uparrow$ `\Uparrow`
|$\vert$ <code>&#124;</code> |$\vert$ `\vert` |$┌ ┐$ `┌ ┐`|$\ulcorner \urcorner$ `\ulcorner`<br>$~~~~$`\urcorner`  |$\Downarrow$ `\Downarrow`
|$\Vert$ <code>&#92;&#124;</code> |$\Vert$ `\Vert` |$└ ┘$ `└ ┘`|$\llcorner \lrcorner$ `\llcorner`<br>$~~~~$`\lrcorner`  |$\Updownarrow$ `\Updownarrow`
|$\lvert~\rvert$ `\lvert`<br>$~~~~$`\rvert`|$\lVert~\rVert$ `\lVert`<br>$~~~~~$`\rVert` |`\left.`|  `\right.` |$\backslash$ `\backslash`
|$\lang~\rang$ `\lang`<br>$~~~~$`\rang`|$\lt~\gt$ `\lt \gt`|$⟦~⟧$ `⟦ ⟧`|$\llbracket~\rrbracket$ `\llbracket`<br>$~~~~$`\rrbracket`|$\lBrace~\rBrace$ `\lBrace \rBrace`

**调整尺寸**

$\left(\LARGE{AB}\right)$ `\left(\LARGE{AB}\right)`

$( \big( \Big( \bigg( \Bigg($ `( \big( \Big( \bigg( \Bigg(`

||||||
|:--------|:------|:--------|:-------|:------|
|`\left`  |`\big` |`\bigl`  |`\bigm` |`\bigr`
|`\middle`|`\Big` |`\Bigl`  |`\Bigm` | `\Bigr`
|`\right` |`\bigg`|`\biggl` |`\biggm`|`\biggr`
|         |`\Bigg`|`\Biggl` |`\Biggm`|`\Biggr`


