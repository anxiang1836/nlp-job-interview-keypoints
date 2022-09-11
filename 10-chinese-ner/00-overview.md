# 中文NER

> 命名实体识别任务（NER）是几乎所有NLP应用场景下的最核心任务之一，覆盖到：会话系统、智能客服、知识图谱、细粒度情感分析、事件抽取。

## 1：中文NER任务的特点

* 中文无分词边界，分词存在误差

  > 与英文NER相比，中文NER通常采取基于字符的方式。究其缘由，由于中文分词存在误差，所以基于字符的NER系统通常好于基于词汇（经过分词）的方法。

* 字粒度信息缺乏语义信息

  > 虽然字粒度通常好于基于词汇的的方法，但是，从中文的角度，使用字符的NER是没有使用到词汇信息的，不成词的单字没有严格的语义含义，而NER任务又是对于实体边界非常敏感的任务。

在基于字符的NER系统中引入词汇信息，是近年来NER的研究重点。这种引入词汇的方法一般称为"**词汇增强**"，即引入词汇信息可以增强NER性能。

## 2：近些年的中文NER的经典佳作

这里整理了共计6篇的经典paper，按照会议的录用时间，顺序排序如下：

> - Lattice LSTM: Chinese NER Using Lattice  LSTM(ACL 2018)
> - WC-LSTM: An Encoding Strategy Based Word-Character LSTM for Chinese NER Lattice LSTM(NAACL 2019)
> - Simple-Lexicon: Simplify the Usage of Lexicon in Chinese NER(ACL 2020)
> - FLAT: Chinese NER Using Flat-Lattice Transformer(ACL 2020)
> - Porous Lattice Transformer Encoder for Chinese NER(ACL 2020)
> - LEBERT: Lexicon Enhanced Chinese Sequence Labelling Using BERT Adapter(ACL2021)

<img src="https://pictrue-bed.oss-cn-beijing.aliyuncs.com/20220911214857.png" style="zoom:85%;" />

PS：我排除了一些Graph NetWork有关的paper（主要是感觉模型略微有一些花哨，我又不是很涉略图神经网络）后面有精力想拓展的可以再进行展开：

> - CGN: Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network（ EMNLP2019）
> -  LGN: A Lexicon-Based Graph Neural Network for Chinese NER(EMNLP2019)
> - Multi-digraph: A Neural Multi-digraph Model for Chinese NER with Gazetteers（ACL2019）

