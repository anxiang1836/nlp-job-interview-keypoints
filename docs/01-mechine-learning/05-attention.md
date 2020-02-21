# Attention考点

> 2020.02.21更新：初始创建

## 问题1：为什么会有attention？

attention创造之初，是为了翻译任务（Seq2Seq）而生的，但最后又不局限于翻译任务。

Seq2Seq模型存在的缺点：

- 所有的输入都编码成一个固定长度的context vector，而固定长度的vector并不能很好的编码所有的上下文信息，导致长距离依赖关系信息都消失了
- 很难对输入和输出序列之间的词语对齐进行建模，这是结构化输出任务（如翻译或摘要）的重要任务之一

- 从传统经验来看，Seq2Seq任务的输出会受到输入的某些特定部分影响，显然Seq2Seq模型并没有建模

因此，最早是由[Bahdanau[2014]](https://arxiv.org/pdf/1409.0473))在做机器翻译时提出的在Encoder和Decoder之间，增加追加attention block，最主要就是解决编/解码器之间匹配问题。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200221165932.png)

## 问题2：Attention机制的本质思想？

参考资料：https://cloud.tencent.com/developer/article/1143127

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200221170733.png)

把Attention机制从上文讲述例子中的Encoder-Decoder框架中剥离，并进一步做抽象，可以更容易看懂Attention机制的本质思想。

> 将Source中的构成元素想象成是由一系列的<Key,Value>数据对构成。
>
> 给定Target中的某个元素Query，通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。
>
> 所以本质上Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数。

$$
Attention(Query,Source)=\sum_{i=1}^{L_x} SoftMax \Big(Sim(Query,Key_i)\Big) *Value_i
$$

其中Lx代表Source的长度。

听起来简单，实现起来其实也简单，我认为对于Attention可以有三种理解：

- **首先**，从数学公式上和代码实现上Attention可以理解为**加权求和**。
- **其次**，从形式上Attention可以理解为**键值查询**。
- **最后**，从物理意义上Attention可以理解为**相似性度量**。

> 对应到具体任务上，可能会更加清晰一点：
>
> 在机器翻译任务中，Query可以定义成decoder中某一步的hidden state，Key是encoder中每一步的hidden state，我们用每一个Query对所有Key都做一个对齐，decoder每一步都会得到一个不一样的对齐向量。
>
> 在文本分类任务中，Query可以定义成一个可学的随机变量（参数），Key就是输入文本每一步的hidden state，通过加权求和得到句向量，再进行分类。
>
> PS：一般情况下，K和V是相同的。。

## 问题3：Attention中相似度计算方式？

在做attention的时候，我们需要计算query和某个key的分数（相似度），常用方法有：

1）点乘：最简单的方法：
$$
sim(q,k)=q^Tk
$$
2）矩阵相乘： 
$$
sim(q,k)=q^TWk
$$
3）cos相似度：
$$
sim(q,k)=\frac{q^Tk}{||q|| \cdot ||k||}
$$
5）用多层感知机：
$$
attention = v^Ttanh(Wq+Uk)
$$

## 问题4：都有哪些种类的Attention？

参考资料： [Chaudhari, S., An attentive survey of attention models, 2019](https://arxiv.org/pdf/1904.02874.pdf) （非常精彩的一篇综述）