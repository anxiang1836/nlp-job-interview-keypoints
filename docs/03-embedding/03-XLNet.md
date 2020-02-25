# XLNet

> 很有幸能够在疫情中，参加了贪心学院组织的公开课，非常喜欢李文哲老师的讲课风格，而且思路清晰，点赞。

## 问题1：DAE解决什么问题？

DAE（Denoising Autoencoders）：Vincent在2008年的论文《Extracting and Composing Robust Features with Denoising Autoencoders》，是输入中加入随机噪声的AutoEncoders，来缓解经典AutoEncoder容易过拟合的问题。

>模型的过程如下：
>
>1. 就是以一定概率分布（通常使用二项分布）去擦除原始input矩阵，即每个值都随机置0,  这样看起来部分数据的部分特征是丢失了。
>2. 以这丢失的数据去计算hat(x)，计算y、z，并将z与原始x做误差迭代，这样，网络就学习了这个破损（Corruputed）的数据。
>
>![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200225111034.png)

破坏原始数据的意义在于：

- **通过与非破损数据训练的对比，破损数据训练出来的Weight噪声比较小。** 降噪因此得名。原因不难理解，因为擦除的时候不小心把输入噪声给擦掉了。

- **破损数据一定程度上减轻了训练数据与测试数据的代沟。** 由于数据的部分被擦掉了，因而这破损数据一定程度上比较接近测试数据。（训练、测试肯定有同有异，当然我们要求同舍异）。这样训练出来的Weight的鲁棒性就提高了。

## 问题2：BERT与DAE的关系？

BERT是一种基于Transformer Encoder来构建的一种模型，它整个的架构其实是基于DAE（Denoising Autoencoder，去噪自编码器）的，这部分在BERT文章里叫作Masked Lanauge Model（MLM）。BERT随机把一些单词通过MASK标签来代替，并接着去预测被MASK的这个单词，**过程其实就是DAE的过程。**

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200222201857.png)

## 问题3：Autoregressive VS AutoEncoding？

### 3.1 模型含义

> 自回归语言模型（Autoregressive LM）**
>
> 根据上文内容预测下一个可能跟随的单词，就是常说的自左向右的语言模型任务，或者反过来也行，就是根据下文预测前面的单词，这种类型的LM被称为自回归语言模型。
>
> PS：ELMO是做了两个方向（从左到右以及从右到左两个方向的语言模型），但是是分别有两个方向的自回归LM，然后把LSTM的两个方向的隐节点状态拼接到一起，所以并没有同时利用到上文、下文。
>
> **自编码语言模型（Autoencoder LM）**
>
> 根据上下文单词来预测这些被Mask掉的单词，也就是典型的DAE的思路。那些被Mask掉的单词就是在输入侧加入的所谓噪音。类似Bert这种预训练模式，被称为自编码语言模型。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200222203139.png)

### 3.2 模型优劣

先来说Autoregressive模型：

- 优点：
  - Train与Text一致
  - 考虑依赖关系
- 缺点：
  - 不能同时考虑双向

#### Auto-encoding

- 优点：
  - 考虑双向
- 缺点：
  - Train与Text不一致（因为Train的时候是有Mask的，Text不Mask）
  - 独立假设（比如说，New York is a city|Los Angles is a city，因为独立预测Mask的部分，可能会预测出NewAngles和LosYork的这种）

## 问题4：XLNet是什么LM模型？做了什么样的变动？

XLNet的思路采用的是AutoRegressive LM模型，根据上文来预测下一个单词；但是为了能够同时保证引入上文和下文的信息，这里对句子进行乱序排列组合：

比如，输入句子为x1->x2->x3->x4：预测x3的时候，只知道x1和x2的信息；

排列组合x2->x4->x3->x1：这样预测三的时候，就是能看到上文x2的信息和下文x4的信息了。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200222204831.png)

### 问题5：XLNet是怎么控制实现Permutation LM的？

PermutationLM通过Attention-Mask来实现的。



## (未完，待完善)

