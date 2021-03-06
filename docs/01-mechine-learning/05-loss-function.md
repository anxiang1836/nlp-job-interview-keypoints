# 损失函数

## 问题1：常见的损失函数？

* 平方损失函数

$$
L(y,f(x))=(y-f(x))^2
$$

> 使用平方损失来计算Loss的模型：
>
> 线性回归

* 交叉熵损失函数

$$
H(p,q)=-\sum p(x)logq(x)
$$

其中，p\(x\)​表示正确答案的概率分布，q\(x\)​是表示预测值的概率分布。举个例子：

比如说，一个三分类的问题，某一个样例的正确答案是\(1,0,0\)。模型经过SoftMax之后，预测概率为\(0.5,0.4,0.1\)。那么这个预测结果和正确答案的交叉熵是：

$$
H((1,0,0),(0.5,0.4,0.1))=-(1*log0.5+0*log0.4+0*log0.1)=0.3
$$

> 使用交叉熵来计算Loss的模型：
>
> 神经网络、逻辑回归

* Hinge Loss

$$
L(w,b)=max\{0,1-yf(x)\}
$$

其中，y=1或-1​，f\(x\)=wx+b，当SVM的核函数为f\(x\)时。

> 使用Hinge Loss来计算Loss的模型：
>
> 支持向量机

* Focal Loss

> （KaiMing团队在2017年提出的[《Focal Loss for Dense Object Detection》](https://arxiv.org/abs/1708.02002)）
>
> Focal Loss是针对**样本不均衡**而修正的交叉熵损失。
>
> 下面问题2进行详细展开。

## 问题2：Focal Loss是什么？解决的什么问题？

答：是在解决样本不均衡的问题基础上，进一步解决了容易区分和、难以区分的样本权重自适应的问题。

> 样本不均衡的解决办法：Weighted cross-entropy。
>
> 【本质】：在训练神经网络时，面对样本不均衡问题，我们是可以在神经网络计算Loss时，添加样本的权重的，以平衡不同数量的样本对于Loss的贡献程度。
>
> 【如何实现呢？】
>
> ```python
> # keras已经在新版本中加入了 class_weight = 'auto'。
> # 设置了这个参数后，keras会自动设置class weight让每类的sample对损失的贡献相等
>
> clf.fit([X_head_train,X_body_train], y_train_embedding, 
>         epochs=10, batch_size=128, 
>         class_weight = 'auto', 
>         validation_data= [[X_head_validate, X_body_validate], y_validate_embedding], 
>         callbacks = [tsb])
> ```
>
> 参数说明：[https://blog.csdn.net/weixin\_40755306/article/details/82290033](https://blog.csdn.net/weixin_40755306/article/details/82290033)

Focal Loss函数的主要出发点为：**通过减少易分类样本的权重，使得模型在训练时更专注于难分类的样本**。

以二分类为例，绘制Loss的函数图像：

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200130214251.png)

> 首先在原有的Cross Entropy的基础上加了一个因子，其中γ&gt;0使得减少易分类样本的损失。使得更关注于困难的、错分的样本。
>
> ![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200130221931.png)
>
> 此外，加入平衡因子α，用来平衡正负样本本身的比例不均：文中α取0.25，即正样本要比负样本占比小，这是因为负例易分。

【与Weighted CrossEntropy的区别】

答：只添加α虽然可以平衡正负样本的重要性，但是无法解决简单与困难样本的问题。

> Focal loss 核心参数有两个，一个是α，一个是γ。
>
> 其中γ是类别无关的，而α是类别相关的。
>
> γ根据真实标签对应的输出概率来决定此次预测loss的权重，概率大说明这是简单任务，权重减小，概率小说明这是困难任务，权重加大。（这是Focal loss的核心功能）
>
> α是给数量少的类别增大权重，给数量多的类别减少权重。
>
> 多分类时，可以不使用α，因为其一，论文中讲α的使用带来微弱提升，其二α要为每个类别指定一个权重，增加太多超参（亦或者用样本中的类别统计数来自动计算各个类别的比例，作为α值？）

## 问题3：是否了解其他因样本失衡而设计的Loss函数？

在苏剑林老师的[《用Bert4keras做三元组抽取》](https://spaces.ac.cn/archives/7161)的帖子中，提到了一种对于输出概率做n次方运算的方法。（在抽取任务下，标记为1的标签会远小于标记为0的，因此需要有针对性的提高标记为1的对权重的影响，降低标记为0对权重的影响）

【解决方案】

具体来说，原来输出一个概率值p，代表类别1的概率是p。现在将它变为p^n，此外，Loss还是用正常的二分类交叉熵loss。

> 由于原来就有0≤p≤1，所以p^n​整体会更接近于0，因此初始状态就符合目标分布了，所以最终能加速收敛。

从loss角度也可以比较两者的差异。假设t为\(0,1\)的概率值，那么原来的Loss是：

$$
−tlogp−(1−t)log(1−p)
$$

而次方之后的loss就变成了:

$$
−tlogp^n−(1−t)log(1−p^n)
$$

注意到，当标签为1时，相当于放大了loss的权重，而标签为0时，1-p^n​更接近于1，因而得到的Loss会更小。

相比于focal loss或人工调节类权重，这种方法的好处是不改变原来内积（$p$通常是内积加sigmoid得到的）的分布就能使得分布更加贴近目标，而不改变内积分布通常来说对优化更加友好。

## 问题4：Ranking-Loss解决什么问题？如何起到效果的？

（在关系抽取的部分，提到了对CNN模型做出了改进的！）——后面补充上的

