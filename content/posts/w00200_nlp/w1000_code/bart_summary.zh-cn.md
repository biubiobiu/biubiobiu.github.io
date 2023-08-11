---
title: "code解析"
date: 2021-09-08T06:00:20+06:00
menu:
  sidebar:
    name: code解析
    identifier: code-summary-github
    parent: code-github
    weight: 1
author:
  name: biubiobiu
  image: /images/author/john.png
math: true
tags: ["Transformer"]
categories: ["Basic"]
---


## 一、transformers
Hugging Face公司发布的transformers包，能够超级方便的引入训练模型：BERT、GPT2、...
<a href="https://huggingface.co/docs/transformers/index" target="blank">transformers英文文档</a>
<a href="https://huggingface.co/docs/transformers/main/zh/index" target="blank">transformers中文文档</a>



## 二、Tokenizer

```python
from transformers import BertTokenizerFast, BertTokenizer
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
# 初始化tokenizer
tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")

# 对比 tokenizer.encode() 与 tokenizer.tokenize()
sentence = "Hello, my son is cuting."
input_ids_1 = tokenizer.encode(sentence, add_special_tokens=False)
# add_special_tokens=True 将句子转换成对应模型的输入形式，默认开启。就是首尾加上[cls]、[sep]。即：tensor([ 101, 7592, 1010, 2026, 2365, 2003, 3013, 2075, 1012,  102])
# add_special_tokens=False 首尾先不加[cls]、[sep]

input_tokens = tokenizer.tokenize(sentence)
# ['hello', ',', 'my', 'son', 'is', 'cut', '##ing', '.']
input_ids_2 = tokenizer.convert_tokens_to_ids(input_tokens)
# tensor([7592, 1010, 2026, 2365, 2003, 3013, 2075, 1012])
# 并没有开头和结尾的标记：[cls]、[sep]
```
其中tokenizer.encode()的参数
  1. add_special_tokens=True, 首尾是否添加[cls]、[sep]
  2. max_length=512, 设置最大长度，如果不设置的话，模型设置的最大长度为512，如果超过512会报错。所以启用这个参数，设置想要的最大长度，这样函数将只保留长度-2个token并转化成id。
  3. pad_to_max_length=False, 是否按照最长长度补齐，默认关闭。此处可以通过tokenizer.padding_side='left'设置补齐的位置在左边插入
  4. truncation_strategy='longest_first', 截断机制，有四种方式
    - 'longest_first' 默认，读到不能再读，读满为止
    - 'only_first', 只读入第一个序列
    - 'only_second', 只读入第二个序列
    - 'do_not_truncate', 不做截取，长了就报错
  5. return_tensors=None, 返回的数据类型，默认是None, 可以选择TensorFlow版本('tf')和pytorch版本('pt')




## 三、总结


