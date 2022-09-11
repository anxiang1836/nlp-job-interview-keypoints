# 正则化

## 问题1：L1正则化和L2正则化？

> 参考资料：[https://blog.csdn.net/jinping\_shi/article/details/52433975](https://blog.csdn.net/jinping_shi/article/details/52433975)

### 1.1 处理方式

L1-norm和L2-norm都是针对于机器学习中的常见处理方式：

* L1正则化是指权值向量w中各个元素的绝对值之和
* L2正则化是指权值向量w中各个元素的平方和然后再求平方根

对应线性回归模型：

* L1正则化的模型建模叫做Lasso回归

  $$
  \mathop{min}_w \frac{1}{2n_{sample}}||X_w-y||^2_2+\alpha||w||_1
  $$

* L2正则化的模型建模叫做Ridge回归（岭回归）

  $$
  \mathop{min}_w \frac{1}{2n_{sample}}||X_w-y||^2_2+\alpha||w||^2_2
  $$

### 1.2 直观理解与区别

（待补充）

## 问题2：Batch Normalization

> 参考资料：[https://blog.csdn.net/Taiyang625/article/details/89245907](https://blog.csdn.net/Taiyang625/article/details/89245907)

是对于Mini-Batch进行的，作用是将每一个Batch的输入值的分布拉回到N\(0,1\)的正态分布上

$$
\hat{x}^{(k)} = \frac{x^{(k)}-E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}
$$

**BN的位置**：一般是BN在激活函数前，在卷积之后。

缺点是：

* 对于Batch-size过于敏感，目的就是尽量让每一个batch的分布于训练样本的整个分布近乎相同，而batch太小的话，则可能不具有统计意义了
* 对于DNN、CNN这种深度固定的还好，但是对于RNN深度不固定的，处理起来会很麻烦（待补充原理分析）

## 问题3：Layer Normalization

是对于同一层的神经元进行的，将统计值归到相同的均值方差上

$$
\begin{align}
\mu^l &= \frac{1}{H} \sum^H_{i=1}\alpha_i^l \\\\
\sigma^l &= \sqrt{\frac{1}{H}\sum^H_{i=1}(\alpha_i^l - \mu^l)^2}
\end{align}
$$

同时，LN用于RNN的效果比较明显，但在CNN上表现不如BN。

> PS：在Transformer的FFN中加入的就是LN，而且激活函数用的是gelu。

