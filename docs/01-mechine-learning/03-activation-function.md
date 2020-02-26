# 激活函数

> 2020.02.26 文本创建 （参考资料：[从ReLU到GELU，一文概览神经网络的激活函数](https://mp.weixin.qq.com/s/PMPG804-zMDHUzH9Gy4NLw)）

## 问题1：激活函数的意义？

在没有激活函数的时候，网络的表达能力有限，无论多少层的网络结构都可以退化表示成一个单层的线性网络。

**那么它的意义在于：**

> 1. 模拟生物神经元特性，接受输入后通过一个阈值模拟神经元的激活和兴奋并产生输出。
> 2. 为神经网络引入非线性，增强神经网络的表达能力。

## 问题2：Sigmoid函数？

sigmoid 函数是一个 logistic 函数，意思就是说：不管输入是什么，得到的输出都在 0 到 1 之间。也就是说，你输入的每个神经元、节点或激活都会被缩放为一个介于 0 到 1 之间的值。
$$
sigmoid(x)=\frac{1}{1+e^{-x}}
$$
他的函数图像如下图：

<img src="https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200226145458.png" style="zoom:50%;" />

### 2.1 Sigmoid缺点

【梯度消失】Sigmiod问题出在了当x较大时，其导数近乎为0，那么在反向传播时，每一个参数的更新量都会去乘上这个近乎为0的值，也就几乎不会产生什么太多的变化，即为梯度消失。

因为其导数为：
$$
\sigma'(x)=\sigma(x)(1-\sigma(x))=\frac{e^{-x}}{(e^{-x}+1)}
$$
<img src="https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200226155246.png" style="zoom:50%;" />

## 问题3：ReLU函数？

整流线性单元是解决梯度消失问题的方法，但是ReLU还是存在一定问题的。

ReLU的公式如下：
$$
ReLU(0,x)=max(0,x)
$$
其公式的含义为：

- 如果输入 x 小于 0，则令输出等于 0；
- 如果输入 x 大于 0，则令输出等于输入。

那这跟解决梯度消失有什么关系呢？
$$
ReLU'(x)=\left\{\begin{align}0,x<0\\\\1,x\ge0\end{align}\right.
$$
<img src="https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200226160526.png" style="zoom:50%;" />

答案是：

> 当使用 ReLU 激活函数时，我们不会得到非常小的值（比如前面 sigmoid 函数得到的 0.0000000438），因为ReLU的导函数的值要么是 0（导致某些梯度不返回任何东西），要么是 1。

### 3.1 ReLU的缺点

【死亡ReLU问题】

从上面的解释可以得到，尽管通过ReLU避免了会得到很小的值，但是会引起另外一个很严重的问题：

>  当输入小于0的时候，会得到0更新。那么，可以想象，当如果输入存在较多的小于0的值时，这就会导致网络学习陷入了停滞的状态，网络不再更新。

### 3.2 ReLU特点总结

优点：

- 相比于Sigmoid，由于稀疏性，时间和空间复杂度更低，不涉及到过于复杂的指数运算；
- 避免了梯度消失的问题

缺点：

- 引入了死亡ReLU，即网络内的部分参数永远都不会更新了；（但有的时候也是一个优势）
- ReLU不能避免梯度爆炸的问题

## 问题4：ELU函数？

**指数线性单元（ELU）**：指数线性单元激活函数解决了 ReLU 的一些问题，同时也保留了一些好的方面。这种激活函数要选取一个 α 值；常见的取值是在 0.1 到 0.3 之间。
$$
ELU(x)=\left\{\begin{align}\alpha (e^x - 1),x\le 0\\\\x,x > 0\end{align}\right.
$$
<img src="https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200226221755.png" style="zoom:50%;" />

也就是说：如果输入的 x 值大于 0，则结果与 ReLU 一样——即 y 值等于 x 值；但如果输入的 x 值小于 0，则我们会得到一个稍微小于 0 的值。

### 4.1 ELU的导函数

$$
ELU'(x)=\left\{\begin{align}\alpha + ELU(x),x\le 0\\\\1,x > 0\end{align}\right.
$$

看起来很简单。如果输入 x 大于 0，则 y 值输出为 1；如果输入 x 小于或等于 0，则输出是 ELU 函数（未微分）加上 α 值。

<img src="https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200226222343.png" style="zoom:50%;" />

【闪光之处】：

从函数图上可以直观的看出来，成功的避免了死亡ReLU的问题，但与此同时又保持了ReLU 激活函数的一些计算速度增益——也就是说，网络中仍还有一些死亡的分量

### 4.2 ELU特点总结

优点：

- 能避免死亡 ReLU 问题，在计算梯度时能得到激活，而不是让它们等于 0；
- 能得到负值输出，这能帮助网络向正确的方向推动权重和偏置变化；

缺点：

- 由于包含指数运算，所以计算时间更长；
- 无法避免梯度爆炸问题；
- 神经网络不学习 α 值。

## 问题5：Leaky ReLU？

Leaky ReLU激活函数很常用，但相比于 ELU 它也有一些缺陷，但也比 ReLU 具有一些优势。

Leaky ReLU的数学形式如下：
$$
LReLU(x)=\left\{\begin{align}\alpha x,x\le 0\\\\x,x > 0\end{align}\right.
$$
<img src="https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200226181641.png" style="zoom:50%;" />

这里有一个参数值α，通常取值在0.1-0.3之间。这样的话，Leaky ReLU的微分值在输入 x 大于或小于 0时，各为一个常量（为正的时候，微分值为1；为负的时候，微分值为α）

### 5.1 Leaky ReLU特点总结

优点：

- Leaky ReLU能避免死亡 ReLU 问题，因为其在计算导数时，在输入为负时允许较小的梯度；
- 由于不包含指数运算，所以计算速度比 ELU 快。

缺点：

- 无法避免梯度爆炸问题；
- 神经网络不学习 α 值；
- 在微分时，两部分都是线性的；而 ELU 的一部分是线性的，一部分是非线性的。

## 问题6：GeLU函数？

> 论文原文：[GAUSSIAN ERROR LINEAR UNITS (GELUS)](https://arxiv.org/pdf/1606.08415.pdf)

### 6.1 GELU表达式

将input x 乘以一个服从伯努利分布的m，而该伯努利分布又是依赖于输入Input x的。

> 这里X是选择标准正态分布的，原因是：一般神经元的输入数据的分布倾向于服从正态分布，尤其是进行了BatchNorm之后。

$$
m\thicksim Bernoulli(\Phi(x)),\ where \ \Phi(x)=P(X\le x)
$$

其中，X满足标准正态分布：
$$
\Phi(x)=F(x;0,1)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^x e^{-\frac{t^2}{2}}dt
$$
这样，GeLU的表示为：
$$
GELU(x)=xP(X\le x)=x\Phi(x)
$$
其中，用[误差函数erf(x)](https://zh.wikipedia.org/wiki/%E8%AF%AF%E5%B7%AE%E5%87%BD%E6%95%B0)表示标准正态分布：
$$
\begin{align}
\Phi(x)&=\frac{1}{2}+\frac{1}{2}erf\big(\frac{x}{\sqrt 2}\big)\\
erf(x)&=\frac{1}{\sqrt\pi}\int_{-x}^x e^{-t^2}dt=\frac{2}{\sqrt\pi}\int_0^x e^{-t^2}dt
\end{align}
$$


> 如何理解呢？
>
> 