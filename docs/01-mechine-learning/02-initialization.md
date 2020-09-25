# 初始化

> 精品Blog：[https://towardsdatascience.com/how-to-initialize-a-neural-network-27564cfb5ffc](https://towardsdatascience.com/how-to-initialize-a-neural-network-27564cfb5ffc)

## 1. Xavier初始化（2010年）

> 参考资料：[https://blog.csdn.net/weixin\_35479108/article/details/90694800](https://blog.csdn.net/weixin_35479108/article/details/90694800)
>
> 论文原文：[Understanding the difficulty of training deep feedforward neural networks](https://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf)

### 1.1 参数初始化的意义

参数初始化的目的是为了让神经网络在训练过程中学习到有用的信息，这意味着参数梯度不应该为0。而我们知道在全连接的神经网络中，参数梯度和反向传播得到的状态梯度以及入激活值有关：

* 激活值饱和会导致该层状态梯度信息为0，然后导致下面所有层的参数梯度为0；
* 入激活值为0会导致对应参数梯度为0。

所以如果要保证参数梯度不等于0，那么参数初始化应该使得各层激活值不会出现饱和现象且激活值不为0。

### 1.2 Glorot条件

在Glorot提出的Xavier初始化的文章中，给出了参数初始化的必要条件，其认为：

优秀的初始化应该使得各层的**激活值**和**状态梯度**的**方差**在传播过程中的方差保持一致：

$$
\begin{align}
\forall(i,j),& Var(h^i)=Var(h^j) \\\\
\forall(i,j),& Var(\frac{\partial cos \space t}{\partial z^i}) = Var(\frac{\partial cos \space t}{\partial z^j})
\end{align}
$$

> 其中：
>
> $$h^i$$表示第i层的激活值
>
> $$\frac{\partial cos \space t}{\partial z^i}$$表示第i层反向传播的梯度

那么，根据Glorot条件，一顿推导，可以得到满足Glorot条件的初始化的形式，应该是如下分布：

$$
W \thicksim U\bigg[ -\frac{\sqrt 6}{\sqrt{n_i+n_{i+1}}},\frac{\sqrt 6}{\sqrt{n_i+n_{i+1}}} \bigg]
$$

### 1.3 Xavier初始化的缺点

因为Xavier的推导过程是基于几个假设的：

* 一个是激活函数是线性的，这并不适用于ReLU激活函数；
* 另一个是激活值关于原点对称，这个不适用于sigmoid函数和ReLU函数。所以作者并没有对sogmoid网络应用Xavier初始化。

也就是说，Xavier初始化适用于tanh和Softsign这种关于原点对称的激活函数

## 2. Kaiming初始化（2015年）

> 参考资料：[https://www.jianshu.com/p/f2d800388d1c](https://www.jianshu.com/p/f2d800388d1c)
>
> 论文原文：[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

实际上，对于深层神经网络来说，线性激活函数是没有价值，神经网络需要非线性激活函数来构建复杂的非线性系统。今天的神经网络普遍使用relu激活函数。所以Kaiming大神就针对Relu提出Kaiming初始化。

### 2.1 Kaiming原理

因为Relu在处理输入状态时，只有一半的激活，另一半是抑制不激活的，那么，在计算方差时，假设均值为0的Data通过Relu激活后，会在Output产生期望的偏移，这便不满足Xavier中的推导过程了。

> 全部推导过程参见原论文的第2章

最终得到最新的rescale系数：$$\sqrt{\frac{2}{n}}$$，同时，这里Kaiming初始化不同于Xavier初始化的均匀分布，而是正态分布。

也就是说，假如对于某一层的输入神经元的size是512，那么，对于该层的神经元的初始化应该就是：

$$
Normalized \space random \space weights \times \sqrt{\frac{2}{512}}
$$

