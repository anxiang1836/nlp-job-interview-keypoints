# WC-LSTM

> **paper :** [An Encoding Strategy Based Word-Character LSTM for Chinese NER](https://www.aclweb.org/anthology/N19-1247.pdf)
>
> **source:** NAACL-2019
>
> **code:** [WC-LSTM](https://github.com/liuwei1206/CCW-NER)

主要的核心思想：它是Lattice-LSTM模型的改进版，WC-LSTM解决了**Lattice-LSTM不支持batch**的问题；正向和反向的LSTM分别挂载不同的词信息。

Lattice-LSTM是否增加远程信息取决于是否有词信息，一个节点可能有零个或多个远程词信息，所以没有办法批处理；

而WC-LSTM为每个节点都添加远程词信息后，做了对应的挂词策略（4种），所以，每个input都会有相同的shape。

模型结构如下：

![](https://pictrue-bed.oss-cn-beijing.aliyuncs.com/20220912002327.png)

##  1: 引入“字符-单词对”概念

用$$s={c_1, c_2, ..., c_n}$$表示中文句子，其中$$c_i$$表示第$i$个字。

用$$\stackrel{\longrightarrow}{ws_i}$$表表示分配给第$$i$$个字符的候选词集，其中的词语用词典 D 在句子中遍历得到，候选词集的词是以第$i$个字符为**结尾**的词；

用$$\stackrel{\longleftarrow}{ws_i}$$表表示分配给第$$i$$个字符的候选词集，其中的词语用词典 D 在句子中遍历得到，候选词集的词是以第$$i$$个字符为**开头**的词； 

> 这样，对于同一个char，在正向LSTM和反向LSTM的时候 ，会分别挂上 不同的词；

那么原句子的**正向**和**反向**分别可以表示为：
$$
\stackrel{\longrightarrow}{rs}=
\{(c_1,\stackrel{\longrightarrow}{ws_1}),
  (c_2,\stackrel{\longrightarrow}{ws_2}),...,
  (c_n,\stackrel{\longrightarrow}{ws_n})\}
$$

$$
\stackrel{\longleftarrow}{rs}=
\{(c_1,\stackrel{\longleftarrow}{ws_1}),
  (c_2,\stackrel{\longleftarrow}{ws_2}),...,
  (c_n,\stackrel{\longleftarrow}{ws_n})\}
$$



## 2: 4种不同的挂词策略

WC-LSTM建模词信息有四种策略：

- Shortest Word First

  > 直接取词语集中，长度最短的词，所对应的word embedding

- Longest Word First

  > 取词语集中长度最长的词所对应的word embedding

- Average

  > 取词语集中所有词对应的word embedding，求和取平均。如果词语集长度为0，即全是\<PAD\>，则取\<PAD\>的求和平均

- Self-Attention

  > 取词语集中所有词对应的word embedding，的self-attention，然后再对词emb进行加权

> WC-LSTM由于是直接引入词信息理论上应该也不能解决新词问题。

## 3: 特点总结

- 优点

改进了Lattice LSTM 的不能batch训练的问题，训练效率得到了提高；

- 缺点

仍然存在信息：

> 比如，"人和药店"，这个词只会挂在到"人"和"店"这2个字上，中间的2个字不会挂载上这个词信息

仍采取LSTM进行编码：

> 建模能力有限、存在效率问题