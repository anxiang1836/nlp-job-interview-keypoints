# 关系抽取理论知识

参考资料：https://shomy.top/2018/02/28/relation-extraction/

## 问题1：远程监督VS多示例学习？

**远程监督：**即为利用知识库数据标注语料库，生成大量的训练数据。

【强假设】：[Mintz et al.(2009)《Distant supervision for relation extraction without labeled data》](https://www.aclweb.org/anthology/P09-1113.pdf)

Mintz提出的假设：**如果知识库中的实体对存在某种关系，那么包含该实体对的每篇文档都存在该关系。** 

> 显然是太过绝对了！！
>
> 例如，知识库中的三元组(Bill Gatess, Founder of, Microsoft)，文档“Bill Gatess turn to philanthropy was linked to the antitrust problems Microsoft had in the U.S. and the European union”，该文档并没有表示“Founder of”关系，即使存在该实体对。

****

【弱假设】：[Riedel et al.(2010)《Modeling Relations and Their Mentions without Labeled Text》](https://link.springer.com/content/pdf/10.1007/978-3-642-15939-8_10.pdf)

**多示例学习**是有监督学习的一种形式，对一个包进行标注，而不是对一个实例。在关系抽取中，每个实体对定义为一个包，包由存在该实体对的所有句子组成。并不是对每个句子打关系标签，而是对整个包。**如果实体对存在某种关系，那么包中至少有一个句子反映了该关系。**

## 问题2：PCNN模型

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200202173914.png)

[Zeng et al.,2015《Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks》](https://www.aclweb.org/anthology/D15-1203.pdf)



## 问题3：PECNN模型

参考资料：https://cloud.tencent.com/developer/news/197006

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200202211731.png)

[Yunlun Yang et al.,2016 《A Position Encoding Convolutional Neural Network Based on Dependency Tree for Relation Classification》](https://www.aclweb.org/anthology/D16-1007.pdf)

