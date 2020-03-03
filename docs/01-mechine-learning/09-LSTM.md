# LSTM

> 参考资料：\[1\][Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
>
> PS：不看不知道，原来RNNs和LSTM都是上个世纪诞生的产物，惊人啊！

## 问题1：传统RNNs的问题？

传统RNNs模型一定程度上可以建立时间序列上的数据关系；但是，其面对较长的依赖关系，就很难处理了，因为会有梯度消失问题呀！

> [Hochreiter (1991) [German\]](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf) 和 [Bengio, et al. (1994)](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf) 曾经对这个问题进行过深入的研究，发现 RNNs 的确很难解决这个问题。
>
> 我嘞个亲娘啊，30年前~~惊叹了。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200303221040.png)

## 问题2：LSTM的模型结构？

> LSTM由[Hochreiter & Schmidhuber (1997)](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)提出，许多研究者进行了一系列的工作对其改进并使之发扬光大。

LSTM的关键是元胞状态(Cell State)，下图中横穿整个元胞顶部的水平线。

Cell State有点像是传送带，它直接穿过整个链，同时只有一些较小的线性交互。上面承载的信息可以很容易地流过而不改变。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200303222001.png)

然后，在整个Cell的内部，通过门的结构来控制Cell对Cell State的信息添加或删除：

- Forget Gate（遗忘门）
  $$
  f_t = \sigma(W_f\cdot[h_{t-1};x_t]+b_f)
  $$

  > 理解：选择性的对上一状态的信息进行遗忘。
  >
  > 例如：元胞状态可能包含当前主语的性别信息，以用来选择正确的物主代词。当我们遇到一个新的主语时，我们就需要把旧的性别信息遗忘掉。

- Input Gate（传入门）
  $$
  \begin{align}
  i_t &= \sigma(W_i\cdot[h_{t-1},x_t]+b_i)\\
  \\
  \tilde{C_t} &= tanh(W_C\cdot[h_{t-1},x_t]+b_C)
  \end{align}
  $$

  > 理解：决定让多少新的信息加入到Cell状态中来。实现这个需要包括两个步骤：首先，Input Gate决定哪些信息需要更新；然后tanh生成一个向量，也就是备选的用来更新的内容。

- Update Cell State
  $$
  C_t=f_t*C_{t-1}+i_t*\tilde{C_t}
  $$

  > 理解：就是融上一步的部分记忆，并更新这一步的新信息进来

- Output Gate（输出门）
  $$
  \begin{align}
  o_t &= \sigma(W_o[h_{t-1},x_t]+b_o)\\
  \\
  h_t &= o_t*tanh(C_t)
  \end{align}
  $$

  > 理解：根据当前的输入xt和前step的输出ht-1，来计算得到当前要输出多少内容（可理解为权重）；然后，将得到的权重作用于激活后的当前Cell State，得到当前Cell的输出。

## 问题3：为什么LSTM可以轻松处理长时依赖？

主要是LSTM的Cell State的的核心思路起的作用。

在LSTM中，有两个通道在保持记忆：

- 短期记忆h，保持非线性操作，通过门结构实现；
- 长期记忆C，保持线性操作，通过Cell State实现。

因为线性操作是比较稳定的，那么C的变化也就是相对稳定，因此，LSTM可以很好保持长期记忆。

## 问题4：LSTM该不该使用Relu激活？

**答：可以，但需要付出额外考虑梯度爆炸时的梯度裁剪。**

激活函数的作用其实是在为神经网络增加非线性，tanh是LSTM的默认激活函数。

在RNN的层数较深的时候，这时为了避免产生梯度消失的风险，可以考虑使用Relu替换LSTM的激活函数，但是ReLUs在RNNs中，可能会出现另外一个问题：

> 因为RNNs的Cell在时间序列上是参数共享的，且ReLUs的导数要么为0要么为1，那么在BP过程，从某种程度上，可以近似看成权重矩阵的连乘（当然有的会被抑制掉），这就会很大风险出现可怕的梯度爆炸的问题。
>
> 也就是当替换tanh为ReLUs时候要考虑的问题：gradient clip（梯度裁剪）。

### 补充问题4.+：如何在Keras下做梯度裁剪呢？

在keras所有的优化器中，直接设置`clipvalue`参数或者是`clipnorm`参数，如下图：

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200304012351.png)

## 问题5：LSTM的门可否用Relu激活？

**答：显然不能！！！**

首先我们要明白LSTM中的门的在实际工作中的意义是在做选择（可以类比考虑一下逻辑电路的思想）：例如选择遗忘多少，选择记忆多少，选择更新多少。

根据这种实际的数学意义，显然是：

我们是必须要使用值域为[0,1]的饱和激活函数来实现门的选择功能，同时，又可以很好的兼顾满足非线性的激活需要。