# Word2Vec原理知识

## 问题1：图解Word2Vec模型

整理摘自：https://www.jianshu.com/p/a92475dfb99a

首先，用**Skip-Gram**模型进行详细展开，然后，在补充上CBOW在Skip-Gram上具体操作上的区别之处。

### 1.生成训练数据

对于一句话，用window_size大小为2的滑窗进行切分句子（目标单词的前2个词+后2个词）。

例如：“natural language processing and machine learning is fun and exciting”。一个有有10个单词的一句话，其中有9个不重复的单词，用滑动窗口可以生成10组训练数据。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200131214949.png)

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200131234233.png)

### 2.模型训练

这样对于每一个词，都可以表示成1个9维的One-Hot向量。Word2Vec2模型有两个权重矩阵(w1和w2)，为了展示，我们把值初始化到形状分别为(9x10)和(10x9)的矩阵。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200131215227.png)

模型训练总共分为4个步骤：

1. 前向计算
2. 计算误差（EI）
3. 计算Weight的修改量
4. 修改Weight的值

#### 2.1 Forward-process

以第一窗口（#1）中作为训练样本来展示计算过程：

其中目标词是“natural”，上下文单词是“language”和“processing”。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200131215510.png)

#### 2.2 计算误差

注意，这里Skip-gram是用1个词来预测上下文的，所以在EI计算中，**是对上下文词全都计算diff**，然后进行加和，作为总的EI。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200131220320.png)



#### 2.3 计算Weight1和Weight2的修改量

对于Weight2的修改量：用Hidden Layer和EI进行外积运算，得到10*9维的修改量矩阵；

对于Weight1的修改量：首先，将EI（转置）与Weight2进行相乘，得到Hidden Layer的Loss值；然后，用Input与Hidden Layer的Loss值进行外积运算，得到9*10维的修改量矩阵；

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200131233955.png)



#### 2.4 修改Weight1与Weight2

这里就将修改量与学习率进行相乘，从Weight中减下去就OK咯！

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200131234033.png)

### 3. CBOW的区别之处

CBOW模型中，仍然是两个权重矩阵，Weight1和Weight2。主要区别就存在于对于Weight1权重矩阵这里。

#### 3.1 Forward-Process

由于CBOW中是当前词的上下文来预测当前词，所以，就是用Weight1分别与当前词的所有上下文词进行相乘后，取平均，即为：
$$
h=\frac{1}{C} W\cdot(\sum^{C}_{i=1}x_i)
$$

#### 3.2 Backward-Process

- 在计算误差时：softMax之后的结果只与预测词进行计算，得到EI（diff）。

- 在计算Weight1的修正量时：依然是计算所有的上下文词与Hidden Layer的Loss外积，并乘以1/C，得到Weight1的修正量，这里就不详细列出计算公式了。

### 4.最终训练得到的词向量

最终，取Weight1的每一行，作为每一个词的词嵌入（一般称：Weight1为Input Embedding；Weight2为Output Embedding）。

## 问题2：Word2Vec中2种预测方式的区别？

先上图再说：

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200131210802.png)

Word2Vec模型：是基于分布假说，其认为**每个单词的上下文都在其附近的单词中**。通过查看它的相邻单词我们可以尝试对目标单词进行预测。

二者在模型预测上形式的区别：

Continuous Bag-of-Words(CBOW)：尝试从相邻单词（上下文单词）猜测输出（目标单词）；

Skip-gram(SG)：从目标单词猜测上下文单词。

二者在模型效果上的区别：

CBOW：比skip-gram训练快几倍，对出现频率高的单词的准确度稍微更好一些

Skip-gram：能够很好地处理少量的训练数据，而且能够很好地表示不常见的单词或短语

##问题3：为什么skip-gram能够很好表示不常见的单词？

由于Skip-gram学习用给定单词来预测上下文单词，所以万一两个单词（一个出现频率较低，另一个出现频率较高）放在一起，那么当最小化loss值时，**两个单词将进行有相同的处理**，因为每个单词都将被当作目标单词和上下文单词。与CBOW相比，不常见的单词将只是用于预测目标单词的上下文单词集合的一部分。

上面说的可能比较干巴巴的，下面用一个比较生动的例子进行进一步的解释：

CBOW是根据上下文来预测当前词。在CBOW中，先从反向思维来看：

> 比如这样一个句子`yesterday was really a [...] day`，中间可能是`good`也可能是`nice`，比较生僻的词是`delightful`。当CBOW去预测中间的词的时候，它只会考虑模型最有可能出现的结果，比如`good`和`nice`，生僻词`delightful`就被忽略了。

再从正向思维来看：

> 对于`[...] was really a delightful day`这样的句子，每个词在进入模型后，都相当于进行了均值处理，`delightful`本身因为生僻，出现得少，所以在进行加权平均后，也容易被忽视。

而Skip-Gram是根据一个词预测它的上下文。也就是用一个词预测多个词，每个词都会被单独得训练，较少受其他高频的干扰。所以对于生僻词Skip-Gram的word2vec更占优。

## 问题4：为什么CBOW训练速度比skip-gram训练速度快？

CBOW预测行为的次数跟整个文本的词数几乎是相等的，因为是对上下文词的向量进行进行了加和平均正向/反向传播的，复杂度大概是`O(V)`; 

skip-gram进行预测的次数是要多于cbow的：因为每个词在作为中心词时，都要使用周围词进行预测一次，即上下文词都要对正向传播结果进行运算得到diff，再加和得到EI。这样相当于比cbow的方法多进行了K次（假设K为窗口大小），因此时间的复杂度为O(KV)，训练时间要比cbow要长。

## 问题5：负采样是如何优化训练的？



## 问题6：层次化SoftMax是如何工作的？

（Huffman树相关知识）

## 问题7：什么情况下更倾向于使用skip-gram？什么情况下更倾向于CBOW？



## 问题8：FastText与Word2Vec的区别？

（这个问题详见FastText知识章节）

