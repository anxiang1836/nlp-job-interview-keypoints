# Dropout

> 在面试中，面到了Dropout的内容，我对于他的印象是仅在训练时生效，预测时并不生效，但是对于底层的原理和探究并没有掌握的很透彻，导致面试问题直接GG。

## 问题1：Inverted Dropout是如何操作的？

> dropout 有两种实现方式：**Vanilla Dropout** 和 **Inverted Dropout**。
>
> 前者是 [原始论文](https://link.zhihu.com/?target=http%3A//www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) 中的朴素版，后者在 Andrew Ng 的 [cs231](https://link.zhihu.com/?target=http%3A//cs231n.github.io/neural-networks-2/%23init) 课程中有介绍。

* Vanilla Dropout

  在vanilla Dropout中，因为使用**伯努利二项分布**的随机变量使得有1-p的神经元失活，在梯度更新的中，会对保留下来的占比为p的神经元参与了训练，而在预测时是由全部神经元参与预测的，从概率统计上来看，在预测时期望会比训练时放大了1/p倍，所以会在预测时根据dropout策越做调整，很麻烦。

* Inverted Dropout

  在Inverted Dropout中，是在训练阶段进行了缩放操作。在选择1-p的神经元失活的同时，会对再除掉一个p用于数据缩放，然后再参与正向传播和反向更新。这样就会保证训练和与预测时，不会因Dropout带来的神经元失活而导致数学期望发生变化（但是方差会变的哦~~）。

## 问题2：Dropout在remain-rate为0.5和1时，导致预测结果不同的原理分析？

> 考察的内容其实在于：Inverted Dropout在做缩放的时候，虽然保证了期望不变，但是会有方差的偏移。
>
> 下图取自于论文：[《Understanding the disharmony between dropout and batch normalization by variance shift》](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Understanding_the_Disharmony_Between_Dropout_and_Batch_Normalization_by_Variance_CVPR_2019_paper.pdf)

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200322233622.png)

我们知道，在Inverted Dropout在训练时，对Dropout后的数据进行了1/p倍的放大，用来保证期望不变。但是方差呢？

我们另$$X$$表示$$\alpha \sim Bernoulli(p)$$，另$$Y$$表示$$x \sim N(0,1)$$。

在$$X$$与$$Y$$独立时，有$$E(XY)=E(X)E(Y)$$。下面对于$$D(XY)$$进行一步整理。

$$
\begin{align}
D(XY)&= E(X^2Y^2)-E^2(XY)\\\\
         &= E(X^2)E(Y^2)-E^2(XY)\\\\
         &= \big[D(X)+E^2(X)\big]\big[D(Y)+E^2(Y)\big]-E^2(XY)\\\\
         &= D(X)D(Y)+D(X)E^2(Y)+D(Y)E^2(X)+E^2(X)E^2(Y)-E^2(XY)\\\\
         &= D(X)D(Y)+D(X)E^2(Y)+D(Y)E^2(X)
\end{align}
$$

再来看，$$X$$为伯努利二项分布，$$E(X)=p$$，$$D(X)=p(1-p)$$。$$Y$$为0-1正态分布，$$E(Y)=0$$，$$D(Y)=1$$。

则有：

$$
\begin{align}
D(\frac{1}{p}XY)&=\frac{1}{p^2}\big(D(X)D(Y)+D(X)E^2(Y)+D(Y)E^2(X)\big)\\\\
            &= \frac{1}{p^2}\big(p(1-p)+1\times p^2\big)\\\\
            &= \frac{1}{p}
\end{align}
$$

这回我们再回到问题本身上来看：

在使用remain-rate为0.5和1时，训练时的期望相同，方差不同，一个是2一个是1；这样会导致训练学习得到的模型参数产生变化，数据分布在方差上存在差异，所以最后，会导致预测结果在概率上会存在差异。

## 问题3：为什么Dropout会缓解过拟合？

> 参考资料：[https://blog.csdn.net/program\_developer/article/details/80737724](https://blog.csdn.net/program_developer/article/details/80737724)
>
> 先来比较口语化的解释吧，后面找到比较数学再补充过来吧！

1. **取平均的作用：**

   dropout掉不同的隐藏神经元就类似在训练不同的网络，随机删掉一半隐藏神经元导致网络结构已经不同，整个dropout过程就相当于对很多个不同的神经网络的综合结果取平均。而不同的网络产生不同的过拟合，一些互为“反向”的拟合相互抵消就可以达到整体上减少过拟合。

2. **减少神经元之间复杂的共适应关系：**

   dropout强迫神经单元和随机挑选出来的其他神经单元共同工作，达到好的效果，这样权值的更新不再依赖于有固定关系的隐含节点的共同作用，阻止了某些特征仅仅在其它特定特征下才有效果的情况 。消除减弱了神经元节点间的联合适应性，增强模型的鲁棒性。

## 问题4：为什么加Dropout训练时间要比不加时间长？

Dropout因为在每次更新时，是只更新到了

（待定）

