# 其他知识图谱

## 1.事件图谱

参考资料：《[中科院自动化所陈玉博：事件抽取与金融事件图谱构建](https://zhuanlan.zhihu.com/p/45583498)》、《[最新事件抽取技术进展](https://zhuanlan.zhihu.com/p/79735678)》、《[事件知识图谱构建研究进展与趋势](https://www.secrss.com/articles/15676)》、《[事理图谱，下一代知识图谱](https://www.jiqizhixin.com/articles/2018-12-29-23)》

### 问题1.1：事件图谱的意义？

**实体知识图谱**的优点是构建相对简单、不需要深层语义理解，缺点是实体信息脱离具体的语境存在、存在语义信息的片面性，从而缺乏足够的深层语义信息。

实际上，语义理解的知识来源除了实体以外，还有更重要的与实体相关的行为、状态、转换等具体动作信息。作为一种更高层次的语义单位，事件表达了特定人、物、事在特定时间和特定地点相互作用的客观事实。与实体相比，**事件能够更加清晰、精确表示发生的各种事实信息**。

因此，和实体知识图谱相比，事件知识图谱具有更深入、丰富、精确的语义表示能力，可广泛应用于各种知识的学习、推理和理解。

我们可以把**实体**和**事件**看作两类，实体是一种静态的，而事件偏动态。

下面给出一个事件知识图谱的实例：

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200222010407.png)

> 图是一个“金正男遇害”话题的事件知识图谱（部分）实例，该图谱展示了“金正男遇害”话题早期的发展过程。
>
> 其中，圆圈表示事件，边表示事件关系。
>
> 由于篇幅关系，图中省略了每个事件的事件类型、事件的参与者、发生时间、发生地点、关系的可信度等具体信息。

事件知识图谱可以广泛应用于情报分析、信息检索、自动文摘和舆情分析等多个应用领域。例如在情报分析中，可以帮助情报分析员从海量信息中快速获取所关注的某个话题（如：第五代战机研发）相关的事件知识图谱，不仅为情报分析人员节省大量时间减轻工作量，而且更快速、高效和全面。

### 问题1.2：事件图谱构建关键？

**构建一个事件图谱有两项关键技术：第一是事件抽取，第二是事件关系抽取。**

#### 1.2.1 事件抽取

> 事件抽取分两个步骤：第一步就是事件的发现和抽取，第二个是事件元素的抽取。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200222144021.png)

- **事件发现：**（应该也叫做“触发词抽取”）是你要让计算机知道读完这一句话，是哪一个词触发了这个类型的事件并且判断它触发什么类型。

  > 触发词(Trigger)：用于标识事件的谓词（一般为动词和名词），又称为锚，是事件的基本要素之一，例如“生于”、“出生”等就是出生事件的触发词。

- **事件元素抽取：**要让计算机判断出参与这个事件所有的元素是什么，并且它们在这个事件当中扮演一个什么角色，比如说美团和大众点评合并这样一个事件，其实它描述的就是一个公司合并事件。

#### 1.2.2 事件关系抽取

> 事件关系主要包括四类：共指关系，时序关系，因果关系和子事件关系

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200222144644.png)

- 共指关系：比如说一个事件你会有不同的新闻来源，去描述它不同的侧面，如果识别出这些不同的描述，描述的都是同一个事件，就能判断这是共指关系，这样你就能从各个角度全方位地去认识这个事件，并且追踪这个事件。
- 因果关系：有了因果关系以后，就可以做很多风控或者是预测。比如日本大地震导致了海啸，最后导致了核泄露。类似这样的因果关系能对风控和预测有参考价值。
- 时序关系：只要把一个事件结构化，就有时间信息，时序就比较清晰了。

### 问题1.3：句子级事件抽取的方法？

> 秦彦霞, 张民, 郑德权. 神经网络事件抽取技术综述[J]. 智能计算机与应用, 2018, 8(3): 1-5.
>
> 数据集为：ACE2005

#### 1.3.1 传统语法分析方法

（待补充完善）

#### 1.3.2 深度学习方法

> 参考资料：https://zhuanlan.zhihu.com/p/79735678

（待补充完善）

### 问题1.4：篇章级事件抽取的方法？

> 仲伟峰, 杨航, 陈玉博, 等. 基于联合标注和全局推理的篇章级事件抽取[J]. 中文信息学报, 2019, 33(9): 88-95,106.
>
> 数据集为：ACE2005

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200222151800.png)

模型组成：采用Pipeline的方法将问题分为3个子问题：

1. 利用序列标注模型对句子进行实体和事件的联合标注
2. 利用多层感知机对事件中的元素进行分类
3. 基于整数线性规划做全局推理，得到篇章级结构化事件信息

#### 1.4.1 联合标注

使用的是char的Embedding，biLSTM+Attention+CRF的模型结构，进行实体抽取

#### 1.4.2 元素分类

构建了一个多特征拼接的特征向量，然后送入多层感知机进行Softmax分类。特征向量的组成为：
$$
X=[EV,ET,TV,TT,CF,PF]
$$
![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200222154601.png)

#### 1.4.3 全局推理

从句子级抽取了若干个事件，因为在一篇报道中，事件会以不同的角度进行重复，所以，目标函数就是最小化这若干个事件的相似程度。

输入为：
$$
Event=\{e_1,e_2,...,e_l\}
$$
目标函数为：
$$
obj = \sum_i^l \sum_j^l sim(e_i,e_j)*var\_refer_{i,j}
$$
其中，var_refer表示两个事件的类型是否为同一类型，如果不是同一类型的，则表示为0。
$$
s.t. var\_refer_{i,j} \in {0,1}
$$

#### 1.4.4 抽取效果

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200222155910.png)

其实可以看得出，这个效果也才达到35%左右的P和R，这个效果指标其实还是比较差的，主要是受到句子级事件抽取的结果影响的。

## 2.事理图谱

> 哈工大社会计算与信息检索研究中心提出了事理图谱的概念，并基于大规模财经新闻文本构建了一个金融领域事理图谱。本文中我中心正式对外发布该金融事理图谱Demo（http://eeg.8wss.com）
>
> 参考资料：《[事理图谱，下一代知识图谱](https://www.jiqizhixin.com/articles/2018-12-29-23)》、《[从工业应用角度解析事理图谱](https://zhuanlan.zhihu.com/p/53699796)》、

### 问题2.1：事理图谱与实体知识图谱的区别？

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200222004754.png)

**事理图谱：**是以事件为节点，事件间的关系为边的图谱网络；

**知识图谱**：是以实体为节点，实体间的关系为边的图谱网络。

知识（实体）图谱中实体及关系大多是稳定的；事理图谱中关系大多是不确定的，以一定的概率进行转移。

> 近些年知识图谱的火热也带动起了事理图谱的研究，知识图谱的技术及应用相对已经开始成熟。但只是基于实体知识库，并不足以描述事件之间的演化规律，而事理逻辑对现实世界的行为推演尤其重要。

### 问题2.2：事理图谱与事件图谱的区别？

**事件图谱 + 本体 = 事理图谱**

简单说来，事件图谱是不含本体的事理图谱，是事理图谱的初级阶段。

> 在2019年7月20日，哈尔滨举办的“首届事理图谱研讨会”上，对于事理图谱的问答表示：
>
> 事件图谱作为事件表示、演化和推理机制的初级阶段的产物，或许只是一个短暂的存在，尔后大家就会殊途同归，快速走向高级阶段。
>
> 既然事件表示、演化和推理的最终归宿就是事理图谱，那么我们也可以一步到位，直接把这个研究领域和研究对象命名为 EventGraph，与国际接轨。

本人的理解：本体其实就是对于领域的概念抽象，现在事件图谱构建的是具体的；那么，在具体的事件上，进一步挖掘提出出一般的抽象的概念，就是事理图谱。这就很有马克思主义哲学的意味了。

因为这个事理图谱本就是哈工大新提出的概念，建立在事件图谱上的，事件图谱现在都还没有完全的攻克解决，更没办法谈及更高层次的抽象概念了！