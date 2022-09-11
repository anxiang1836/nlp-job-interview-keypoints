# ELMO

## 问题1：请解释一下ELMO？

ELMO是用自回归进行建模的语言模型，即给定文本的Context，预测下一个词。因为其用bi-LSTM进行建模，所以，Embedding中会包含上下文的信息，对每个单词会给出3个

