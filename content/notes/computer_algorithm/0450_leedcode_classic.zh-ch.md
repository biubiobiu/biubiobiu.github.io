---
title: LeedCode-经典
weight: 220
menu:
  notes:
    name: LeedCode-经典
    identifier: notes-leedcode-classic
    parent: notes-algorithm
    weight: 450
math: true

---

<!--
{{< note title="标题" >}}
{{< /note >}}
-->

{{< note title="数组" >}}

1. [x] <a href="https://leetcode.cn/problems/gas-station/?envType=study-plan-v2&envId=top-interview-150" target="blank">134. 加油站 </a> <br>
**关键点**：每个加油站的 剩余=添加-消耗。 累积每个加油站的剩余量，剩余累积量达到最小值时（升高的拐点，注意累积量保持最小值不变的情况。），下一个加油站就是起点。
<p align="center"><img src="/datasets/note/leecode-134.png" width="100%" height="100%" title="merge_sort" alt="merge_sort"></p>

2. [x] <a href="https://leetcode.cn/problems/product-of-array-except-self/?envType=study-plan-v2&envId=top-interview-150" target="blank">238. 除自身外数组的乘积 </a> <br>
**关键点**：从左到右累积相乘，从右到左累积相乘。这样就不用除法，且避免了重复的计算。

3. [x] <a href="https://leetcode.cn/problems/rotate-array/?envType=study-plan-v2&envId=top-interview-150" target="blank">189. 轮转数组 </a> <br>
**关键点**：
    * **方法一**：环状替换：替换到下一个位置，直到回到原位置，完成一轮。如果有没有遍历的元素，偏移一个位置，继续环状替换操作。 
    <p align="center"><img src="/datasets/note/list_rotate.png" width="100%" height="100%" title="list_rotate" alt="list_rotate"></p>

    * **方法二**：多次翻转 --达到-->  旋转的效果


{{< /note >}}



{{< note title="图" >}}

1. [x] <a href="https://leetcode.cn/problems/word-ladder/description/?envType=study-plan-v2&envId=top-interview-150" target="blank">127. 单词接龙 </a> <br>
**关键点**：单词与单词之间用“中间单词”连接。这样的设计是 <font color=#a00000>降低了计算复杂度</font>。
    * 每个单词mask掉一个字母，单词与单词之间没有连接，是通过中间的单词相互连接
    * 通过广度优先遍历，从起始单词开始，直到结束单词。由于路径中有一半的量是“中间单词”，所以总的步数N，应该缩小：N//2+1

    <p align="center"><img src="/datasets/note/127_words.png" width="100%" height="100%" title="list_rotate" alt="list_rotate"></p>
```python
class Solution(object):
    def __init__(self):
        self.nodeNum = 0
    
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        def addWord(word):
            if word not in wordId:
                wordId[word] = self.nodeNum
                self.nodeNum += 1
        
        def addEdge(word):
            addWord(word)
            id1 = wordId[word]
            chars = list(word)
            for i in range(len(chars)):
                tmp = chars[i]
                chars[i] = "*"
                newWord = "".join(chars)
                addWord(newWord)
                id2 = wordId[newWord]
                edge[id1].append(id2)
                edge[id2].append(id1)
                chars[i] = tmp

        wordId = dict()
        edge = collections.defaultdict(list)

        for word in wordList:
            addEdge(word)
        
        addEdge(beginWord)
        if endWord not in wordId:
            return 0
        
        dis = [float("inf")] * self.nodeNum
        beginId, endId = wordId[beginWord], wordId[endWord]
        dis[beginId] = 0

        que = collections.deque([beginId])
        while que:
            x = que.popleft()
            if x == endId:
                return dis[endId] // 2 + 1
            for it in edge[x]:
                if dis[it] == float("inf"):
                    dis[it] = dis[x] + 1
                    que.append(it)
        
        return 0
```



{{< /note >}}



{{< note title="计算器" >}}

<a href="https://leetcode.cn/problems/basic-calculator-iii/" target="bland">基本计算器 III</a> <br>

**通用思路**：<br>

双栈法，stack_num记录数字，stack_opt记录操作符。<br>
首先来看一种最基本的计算操作：
1. stack_num中pop出来两个数A，B, 
2. stack_opt中pop出来一个操作符opt, 计算结果为B opt A, 将结果存到stack_num中。


字符有以下几种情况：
1. 空格，直接下一个字符。
2. 数字，直接入stack_num栈。需要注意处理多位数。
3. '(', 直接入stack_opt栈
4. ')', 重复最基本的计算操作，直到stack_opt栈栈顶为 ')'。然后stack_opt栈pop出来 ')'。
5. 操作符，
    * 如果栈顶操作符的优先级大于等于当前操作符的优先级，则重复最基本的计算操作，直到stack_opt栈空或者其栈顶操作符的优先级小于当前操作符的优先级，再将当前操作符入stack_num栈。
    * 优先级定义priorty = {'(':0,')':0,'+':1,'-':1,'*':2,'/':2}。数字越大，优先级与越高。

最后计算：<br>
在遍历完所有字符后，判断当前操作栈stack_opt是否为空，如果不为空，重复最基本的计算操作，直到操作栈为stack_opt为空。最后结果就是stack_num中剩余的唯一一个元素。

```python
class Solution(object):
    def calc(self, num1, num2, opt):
        if opt == '+':
            return int(num1) + int(num2)
        elif opt == '-':
            return int(num1) - int(num2)
        elif opt == '*':
            return int(num1) * int(num2)
        elif opt == '/':
            return int(int(num1) / int(num2))

    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        stack_num = []
        stack_opt = []
        i = 0
        priorty = {'(': 0, ')': 0, '+': 1, '-': 1, '*': 2, '/': 2}

        while i < len(s):
            if s[i] == ' ':
                i += 1
                continue
            # 数
            if '0' <= s[i] <= '9':
                j = i 
                while i+1 < len(s) and '0' <= s[i+1] <= '9':
                    i += 1
                num = int(s[j:i+1])
                stack_num.append(num)
            elif s[i] == '(':
                stack_opt.append(s[i])
            # 消除 ()
            elif s[i] == ')':
                while stack_opt[-1] != '(':
                    opt = stack_opt.pop()
                    B = stack_num.pop()
                    A = stack_num.pop()
                    res = self.calc(A, B, opt)
                    stack_num.append(res)
                stack_opt.pop()
            # 计算优先级
            else:
                while stack_opt and priorty[stack_opt[-1]] >= priorty[s[i]]:
                    opt = stack_opt.pop()
                    B = stack_num.pop()
                    A = stack_num.pop()
                    res = self.calc(A, B, opt)
                    stack_num.append(res)
                stack_opt.append(s[i])
            i += 1

        while stack_opt:
            opt = stack_opt.pop()
            B = stack_num.pop()
            A = stack_num.pop()
            res = self.calc(A, B, opt)
            stack_num.append(res)

        return stack_num[-1]
```

{{< /note >}}