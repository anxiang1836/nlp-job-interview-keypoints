# XLNet

> 感谢[贪心科技](https://www.greedyai.com/)，在2020年2月组织了一场NLP的公开课，感谢[李文哲](https://www.zhihu.com/people/wenzhe-li/posts)老师，非常清晰滴剖析了XLNet中的1个重要的创新点——Permutation LM及Two-Stream Self-Attention。
>
> [论文原文：XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://papers.nips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)

## SUMMARY

**XLNet的创新点在于什么？**

* **创新点1.1：提出排列语言模型。**

  > 针对BERT中的2个问题：\(1\)Train与Test分布不一致；\(2\)MASK预测的是条件独立假设；
  >
  > XLNet在Auto-Regressive LM的基础上，加入排列组合，即为Permutation LM，解决BERT存在的上述2个问题，同时又使得Auto-Regressive LM能够同时学习上下文的信息。

* **创新点1.2：提出Two-Stream Self-Attention机制。**

  > 与Permutation LM相配合的，XLNet提出Two-Stream Self-Attention机制，以解决在乱序的排列组合的预测序列中，同时满足部分保留全部信息，部分只保留位置信息。（attention-mask矩阵的对角线是否为1）

* **创新点2：引入使用Transformer-XL结构，以解决建模序列更长依赖的问题。**

## 创新点1：Permutation LM

### 问题1.1：DAE解决什么问题？

**DAE**（Denoising AutoEncoders）：Vincent在2008年的论文《Extracting and Composing Robust Features with Denoising Autoencoders》，在输入中加入随机噪声，来缓解经典AutoEncoder容易过拟合的问题。

> 模型的过程如下：
>
> 1. 就是以一定概率分布（通常使用二项分布）去擦除原始input矩阵，即每个值都随机置0,  这样看起来部分数据的部分特征是丢失了。
> 2. 以这丢失的数据去计算hat\(x\)，计算y、z，并将z与原始x做误差迭代，这样，网络就学习了这个破损（Corruputed）的数据。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200225111034.png)

破坏原始数据的意义在于：

* **通过与非破损数据训练的对比，破损数据训练出来的Weight噪声比较小。** 降噪因此得名。原因不难理解，因为擦除的时候不小心把输入噪声给擦掉了。
* **破损数据一定程度上减轻了训练数据与测试数据的代沟。** 由于数据的部分被擦掉了，因而这破损数据一定程度上比较接近测试数据。（训练、测试肯定有同有异，当然我们要求同舍异）。这样训练出来的Weight的鲁棒性就提高了。

### 问题1.2：BERT与DAE的关系？

BERT是一种基于Transformer Encoder来构建的一种模型，它整个的架构其实是基于DAE（Denoising Autoencoder，去噪自编码器）的，这部分在BERT文章里叫作Masked Lanauge Model（MLM）。BERT随机把一些单词通过MASK标签来代替，并接着去预测被MASK的这个单词，**过程其实就是DAE的过程。**

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200222201857.png)

### 问题1.3：Autoregressive VS AutoEncoding？

> **自回归语言模型（Autoregressive LM）**
>
> 根据上文内容预测下一个可能跟随的单词，就是常说的自左向右的语言模型任务，或者反过来也行，就是根据下文预测前面的单词，这种类型的LM被称为自回归语言模型。
>
> PS：ELMO是做了两个方向（从左到右以及从右到左两个方向的语言模型），但是是分别有两个方向的自回归LM，然后把LSTM的两个方向的隐节点状态拼接到一起，所以并没有同时利用到上文、下文。
>
> **自编码语言模型（Autoencoder LM）**
>
> 根据上下文单词来预测这些被Mask掉的单词，也就是典型的DAE的思路。那些被Mask掉的单词就是在输入侧加入的所谓噪音。类似Bert这种预训练模式，被称为自编码语言模型。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200222203139.png)

下面对比比较一下，**Auto-Regressive**与**Auto-Encoding**的优劣：

|  | Auto-regressive LM | Auto-encoding LM |
| :--- | :--- | :--- |
| **优点** | 1.Train与Test分布一致；2.考虑依赖关系 | 1.能够同时考虑上下文信息 |
| **缺点** | 不能同时考虑双向 | 1.Train与Test分布不一致（Test中无Mask）;2.条件独立假设 |
| **模型代表** | ELMo | BERT |

补充解释一下：

1. ELMo虽然建模了双向的，但是是分别建立了2个，所以不能算同时考虑双向。
2. BERT的条件独立假设：指的是在预测时，对于Mask的预测是独立的，也就是说默认Mask词间无关联

> 比如，New York is a city和Los Angles is a city。现在对前2个词Mask掉，来预测：
>
> $$
> P(New\_York|is\_a\_city) = P(New|is\_a\_city)P(York|is\_a\_city)
> $$
>
> 这样就孤立了被预测词间的依赖关系，这显然是我们所不希望的。

### 问题1.4：XLNet在AutoRegressive LM上做了什么样的变动？

XLNet的思路采用的是AutoRegressive LM模型，根据上文来预测下一个单词；但是为了能够同时保证引入上文和下文的信息，这里对句子进行乱序排列组合，**排列语言建模**（Permutation Language Modeling）。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200225211501.png)

比如，输入句子为x1-&gt;x2-&gt;x3-&gt;x4：预测x3的时候，只知道x1和x2的信息；

排列组合x2-&gt;x4-&gt;x3-&gt;x1：这样预测三的时候，就是能看到上文x2的信息和下文x4的信息了。

因此这就弥补了Auto-Regressive的缺点：**做到能够同时考虑上文和下文的信息**。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200222204831.png)

### 问题1.5：XLNet的Two-Stream Self-Attention

#### 1.5.1 为什么需要Two-Stream？

我们来看这样的例子，预测序列为x3-&gt;x2-&gt;x4-&gt;x1，如下图的右边。在预测x4的时候，我们需要知道：

1. x2与x3全部信息（位置+内容）
2. x4的位置信息
3. （不能包含x4的内容信息）

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/image-20200222213905215.png)

于是就出现了矛盾点：

* 有的时候，我只想要位置
* 有的时候，我既想要位置，又想要内容信息

这种矛盾通过一种策略显然是无法同时满足的，所以Two-Stream的概念就很自然出现。

#### 1.5.2 双流自注意力机制

先通俗的理解：

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200225173601.png)

在这个例子中，2个AttentionMask矩阵区别就在于主对角线元素，红色的为可见，白色为不可见。

> 以第一行为例：第1行表示预测1。
>
> 在这个序列中3-&gt;2-&gt;4-&gt;1。
>
> 1在最后，则其应该看见3，2，4的全部信息，并且看不到自身的内容信息。
>
> 可以看到：
>
> * 【Content Stream】在第1个矩阵的第1行全都为红色，即全可见；
> * 【Query Stream】第2矩阵的第1行，除第1个位置其他全为红色。

所以在2个矩阵中，第1个矩阵就表示**位置信息**的Mask矩阵，第2个矩阵就表示**内容信息**的Mask矩阵。

然后我们用数学的形式进行表达：

论文中提出来新的分布计算方法，来实现目标位置感知：

$$
p_\theta(X_{z_t}=x|x_{Z<t})=\frac{exp\bigg(e(x)^Tg_\theta(x_{Z<t},z_t) \bigg)}
                                                                 {\sum_{x'}exp\bigg(e(x')^Tg_{\theta}(x_{Z<t},z_t)\bigg)}
$$

可以看到，这里的g是把位置信息zt考虑进去的，作为其输入。

**下面还是以序列3-&gt;2-&gt;4-&gt;1为例，从数学的角度来分别理解Content Representation和Query Representation。**

* **Content Representation**内容表述

  该表述和传统的transformer一样，即同时编码上下文的内容和自身内容，如下图：

$$
h_{z_t}^{(m)}\gets Attention(Q=h_{z_t}^{(m-1)},KV=h_{Z \le t}^{(m-1)};\theta),(use \ both \ z_t \ and \ x_{z_t})
$$

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200225205828.png)

* **Query Representation**查询表述

  包含上下文的内容信息和目标的位置信息，但不包括目标的内容信息，如下图：

$$
g_{z_t}^{(m)}\gets Attention(Q=g_{z_t}^{(m-1)},KV=h_{Z<t}^{(m-1)};\theta),(use \ z_t \ but \ cannot \ see \ x_{z_t})
$$

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200225210719.png)

* **总的计算过程**

首先，第一层的有如下的初始化：

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200225191750.png)

然后，在计算中，self-attention的计算过程中两个流的网络权重是共享的。

最后，在微调阶段，只需要简单的把query stream移除，只采用content stream即可。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200225211243.png)

## 创新点2：并入Transformer-XL

> Transformer-XL是在下面这篇文章首次提出的，作者团队就是XLNet的同一班人马
>
> [《Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context》](https://arxiv.org/pdf/1901.02860.pdf)

（未完待续）

[https://zhuanlan.zhihu.com/p/84159401](https://zhuanlan.zhihu.com/p/84159401)

[https://blog.csdn.net/magical\_bubble/article/details/89060213](https://blog.csdn.net/magical_bubble/article/details/89060213)

vanilla Transformer：[https://arxiv.org/pdf/1808.04444.pdf](https://arxiv.org/pdf/1808.04444.pdf)

