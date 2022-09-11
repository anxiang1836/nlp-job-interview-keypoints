# FLAT

> **paper :** [FLAT: Chinese NER Using Flat-Lattice Transformer](https://arxiv.org/pdf/2004.11795.pdf)
>
> **source:** ACL 2020
>
> **code:** [FLAT](https://github.com/LeeSureman/Flat-Lattice-Transforme)

模型当时刷新了中文 NER任务的SOTA，模型还是很值得去学习的。。

![](https://pictrue-bed.oss-cn-beijing.aliyuncs.com/20220912015136.png)

模型主要有 2方面的贡献：

- 无损引入词汇信息：（与SoftLexicon的垂直展开不同，FLAT是将词展平的方式放在query句尾，用attention与原query的字进行交互）
- 改进Transformer的原有相对位置编码，使得展平的词与字能够进行计算位置信息，更适用于NER任务

## 1:  "展平式"无损引入此信息

受到位置向量表征的启发，FLAT设计了一种position encoding来融合Lattice 结构，对于每一个字符和词汇都构建两个head position encoding 和tail position encoding，这种方式可以重构原有的Lattice结构，如下图。

<img src="https://pictrue-bed.oss-cn-beijing.aliyuncs.com/20220912015715.png" style="zoom:50%;" />

这样铺平之后，直接做self-attention，建模字符与所有词汇信息间的交互。将Lattice结构展平，将其从一个有向无环图展平为一个平面的Flat-Lattice Transformer结构，由多个span构成：每个字符的head和tail是相同的，每个词汇的head和tail是skipped的。

## 2: 4组相对位置编码信息

**原生Transformer的绝对位置编码本身缺乏方向性，虽然具备距离感知，但还是被self-attention机制打破了**。

> 仔细分析，BiLSTM在NER任务上的成功，一个关键就是BiLSTM能够区分其上下文信息的方向性，来自左边还是右边。而对于Transformer，其区分上下文信息的方向性是困难的。
>
> 因此，作者决定提升Transformer的位置感知和方向感知

每个token都有head和tail这样2个位置信息，所以，作者设计：每2个token间，由4种相对距离表示$x_i$和$x_j$之间的关系：head-head，head-tail，tail-head，head-tail这样的相对位置信息，如下图：

<img src="https://pictrue-bed.oss-cn-beijing.aliyuncs.com/20220912020717.png" style="zoom:50%;" />
$$
d_{ij}^{(hh)}=head[i]-head[j]
$$

$$
d_{ij}^{(ht)}=head[i]-tail[j]
$$

$$
d_{ij}^{(th)}=tail[i]-head[j]
$$

$$
d_{ij}^{(tt)}=tail[i]-tail[j]
$$

那么相对位置emb可以表示为：
$$
R_{ij}=ReLU(
				W_r(
					p_{d_{ij}^{hh}}\oplus p_{d_{ij}^{ht}} \oplus p_{d_{ij}^{th}}\oplus p_{d_{ij}^{tt}}))
$$
$p_d$的计算方式与Transformer是相同的，三角函数：
$$
p_d^{(2k)}=sin(d/10000^{2k/d_{model}})
$$

$$
p_d^{(2k+1)}=cos(d/10000^{2k/d_{model}})
$$

