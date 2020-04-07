# Ranking Loss

> ranking loss在很多不同的领域，任务和神经网络结构（比如siamese net或者Triplet net）中被广泛地应用。其广泛应用但缺乏对其命名标准化导致了其拥有很多其他别名，比如对比损失Contrastive loss，边缘损失Margin loss，铰链损失hinge loss和我们常见的三元组损失Triplet loss等。

## 1. 度量学习 （Metric Learning）

- 一般的损失函数：

  e.g. 交叉熵损失和均方差损失函数，这些损失的设计目的就是学习如何去直接地预测标签，或者回归出一个值，又或者是在给定输入的情况下预测出一组值，这是在传统的分类任务和回归任务中常用的。

- ranking loss：

  目的是去预测输入样本之间的相对距离。这个任务经常也被称之为**度量学习**(metric learning)。

> 在训练集上使用ranking loss函数是非常灵活的，我们只需要一个可以衡量数据点之间的相似度度量就可以使用这个损失函数了。
>
> 这个度量可以是二值的（相似/不相似）：比如，在一个人脸验证数据集上，我们可以度量某个两张脸是否属于同一个人（相似）或者不属于同一个人（不相似）。通过使用ranking loss函数，我们可以训练一个CNN网络去对这两张脸是否属于同一个人进行推断。（当然，这个度量也可以是连续的，比如余弦相似度。）

在使用ranking loss的过程中，

首先，从这两张（或者三张，见下文）输入数据中提取出特征，并且得到其各自的Embedding；

然后，我们定义一个距离度量函数用以度量这些表达之间的相似度，比如说欧式距离；

最终，我们训练这个特征提取器，以对于特定的样本对（sample pair）产生特定的相似度度量。

尽管我们并不需要关心这些表达的具体值是多少，只需要关心样本之间的距离是否足够接近或者足够远离，但是这种训练方法已经被证明是可以在不同的任务中都产生出足够强大的表征的。

## 2. ranking loss的表达式

Ranking loss主要针对以下两种不同的设置进行：

1. 二元组的训练数据点（正例 + 负例）
2. 三元组的训练数据点（Anchor + 正例 + 负例）

### 2.1 二元组的Loss

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200407233703.png)









> `Triplet loss`最初是在 [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) 论文中提出的，可以学到较好的人脸的`embedding`