# NER

## 模型1：biLSTM-CRF

参考资料：[https://createmomo.github.io/2019/11/13/Table-of-Contents/](https://createmomo.github.io/2019/11/13/Table-of-Contents/)

> LSTM-softmax也可以做序列预测问题，但是为什么不行的呢？
>
> 答：softmax层的输出是相互独立的，即虽然BiLSTM学习到了上下文的信息，但是输出相互之间并没有影响，它只是在每一步挑选一个最大概率值的label输出。这样就会导致如B-person后再接一个B-person的问题。而crf中有转移特征，即它会考虑输出label之间的顺序性，所以考虑用crf去做BiLSTM的输出层。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200311234228.png)

**模型结构**

第1层：Embedding层，将输入转换为字向量

第2层：双向LSTM层，提取序列的上下文语义特征，并且return-sequence=True，返回完整的隐状态序列

第3层：全连接层，将隐状态向量映射到与k维（k是标签数量），用于表示各个位置对于每种label的预测概率

第4层：CRF层，训练的是一个参数为\(k+2\)\*\(k+2\)的矩阵，表示状态转移矩阵，加的2即表示句子的首部与尾部的状态符

> 模型的打分由2部分组成：一部分为LSTM输出的pi，一部分为CRF的转移矩阵的转移概率（这也对应到CRF的2个特征函数：\(1\)节点特征函数；\(2\)局部特征函数）

### 问题1.1：biLSTM-CRF中的CRF的输入内容是什么？

BiLSTM的每个step上的输出映射到等同于标签类别数的维度，比如用BIO标注体系的话，有2中实体，那么，相当与对于标注是有7种标注，那么在biLSTM的输出就映射到7维，表示对于不同类别标签的预测可能概率。

### 问题1.2：为什么CRF层接在biLSTM后可以学习到约束？

## 模型2：biLSTM-LAN

[http://www.dataguru.cn/article-15211-1.html](http://www.dataguru.cn/article-15211-1.html)

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200311235155.png)

## 模型3：[TENER](https://arxiv.org/pdf/1911.04474.pdf)

参考资料：[https://mp.weixin.qq.com/s/lmIau07zaGrr6rZs6XJmJA](https://mp.weixin.qq.com/s/lmIau07zaGrr6rZs6XJmJA)

（改进Transformer的NER识别模型）

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200221235626.png)

