# Bert及Attention原理知识

Transformer网络由多个层组成，每个层都由多头注意力机制和前馈网络构成。由于在全局进行注意力机制的计算，忽略了序列中最重要的位置信息。Transformer为输入添加了位置编码（Positional Encoding），使用正弦函数完成，为每个部分的位置生成位置向量，不需要学习，用于帮助网络学习其位置信息。其示意如下图所示：

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200225213013.png)



另外一经典Blog：https://jalammar.github.io/illustrated-transformer/

上面经典blog的翻译：https://zhuanlan.zhihu.com/p/59629215

李文哲的知乎专栏：https://zhuanlan.zhihu.com/p/84559048