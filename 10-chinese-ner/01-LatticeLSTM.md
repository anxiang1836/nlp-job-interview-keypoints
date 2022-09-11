# 中文NER鼻祖: LatticeLSTM

> LatticeLSTM是"词汇增强"方法的中文NER的开篇之作，提出了一种Lattice LSTM以融合词汇信息。
>
> **paper :** [Chinese NER Using Lattice  LSTM](https://arxiv.org/pdf/1805.02023.pdf)
>
> **source:** ACL-2018
>
> **code:** [Lattice-LSTM](https://github.com/jiesutd/LatticeLSTM)

主要的核心思想：**将潜在词信息整合到基于字符的 LSTM-CRF 中**

由于在网格中存在指数级数量的词-字符路径，因此研究者利用 lattice LSTM 结构自动控制从句子开头到结尾的信息流。

如下图所示，门控单元用于将来自不同路径的信息动态传送到每个字符，不会受到分词偏差的影响。

<img src="https://pictrue-bed.oss-cn-beijing.aliyuncs.com/20220911224720.png" style="zoom:85%;" />

## 1: 整体模型构成

当我们通过词汇信息（词典）匹配一个句子时，可以获得一个类似Lattice的结构。

Lattice是一个有向无环图，词汇的开始和结束字符决定了其位置。Lattice LSTM结构则融合了词汇信息到原生的LSTM中。

<img src="https://pictrue-bed.oss-cn-beijing.aliyuncs.com/20220911231755.png" style="zoom:100%;" />

Lattice LSTM引入了一个word cell结构，对于当前的字符，融合以该字符结束的所有word信息，如对于「店」融合了「人和药店」和「药店」的信息。 

Lattice LSTM在基于字符的模型上附加了基于词汇对的cells和控制信息流的附加门。

## 2: 特点总结

- 优点

Lattice LSTM 的提出，将词汇信息引入，有效提升了NER性能；

- 缺点

计算性能低下，不能batch并行化：

> 究其原因主要是每个字符之间的增加word cell（看作节点）数目不一致；

信息损失：

> 1）每个字符只能获取以它为结尾的词汇信息，对于其之前的词汇信息也没有持续记忆。
>
> 2）由于RNN特性，采取BiLSTM时其前向和后向的词汇信息不能共享。

可迁移性差：

> 只适配于LSTM，不具备向其他网络迁移的特性