# 镁佳科技面试

## 一面

> 一面是个小哥儿，哈工大的硕士，性情很温和，聊得还算不错的。

1. 可以讲讲AUC和ROC么？

2. 用过GBDT么，用到什么场景的？可以讲讲GBDT的原理么？

3. 也就是说GBDT比较擅长处理特征咯。（忘记是怎么问的了）

   > 我大概get到应该想问，GBDT的特征选择和feature_importance的事情吧，就又补充说了一下。

4. 那对于集成学习，除了Boosting之外，还有一类是什么呢？两者之间的区别和优劣嘞？

5. 能说说Attention的原理以及是如何计算的么？

6. 看你在做Embedding基本上都是用w2v的，可以说一下w2v的原理？还有2种优化方式？可以详细说一下负采样么？

7. 看你用到了Albert，能说说和原始Bert的区别么？那你是用的什么做的Albert？

   > 主要是在输入的Embedding上的矩阵分解和Attention Block的参数共享，也说到了SOP替换NSP，并且加宽了维度。我说的是用苏神的bert4Keras。

8. 在项目中你这个用到了2个做NER的模型，有啥区别？效果表现不明显为什么呢？那训练时间上有啥差异么？

   > 主要是比较了一下biLSTM-CRF和biLSTM-LAN的差异，重点说了LAN的Attention设计的巧妙之处。
   >
   > 效果不明显主要是从英文的有着天然的分词表现这个角度分析的。训练时间这块，在原论文中，我大概是是有印象说CRF的比较复杂，而且没有那么大的必要通过CRF来进行条件约束的，所以大概分析了一块儿这里。

9. 在情感分类和文本分类的项目上，感觉关注点在落在了为什么这样设计特征工程，也提出了几点他的想法，一起在这块算是有一段讨论吧。没有太过于切入知识点的询问。

10. 相似句匹配的项目主要其实也还是一样，询问了特征工程做了这么多分别都是怎么再加到网络里面的，以及其中的一些细微细节。

    > 我在介绍项目的时候，就直接说了数据增强的过程，并交代了用什么做的baseline，同时也解释了一下为什么选孪生网络而不是交互式的网络。

11. 算法编程题：

    - [删除链表的倒数第N个节点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)
    - [乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

    （没有线上评测系统，我是直接用SubLimit写的，关键还是要**注意异常值和边界值的判定**，面试官还是比较注重这块儿。写完之后，交代一下算法题解的思考过程。PS：我就忘了分析复杂度了，下回应该主动再加上去分析）