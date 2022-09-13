# HM医疗

面试环节分成3个部分：

## 第1部分

> 简单自我介绍后，让我讲了一下知识图谱构建的项目。

1. 这里面你都用到了什么模型？（我主要讲了NER的模型，然后提到了BiLSTM-LAN）
2. LAN模型的结构是什么样的？（我又额外回答并分析了为什么LAN并没有产生像论文那么好的结果）
3. 对于模型的改进方面，你都做了哪些工作呢？（我回答的是通过加入了Attention，确实获得一些效果上的提升）
4. Attention加在了哪里？Attention加入后对于效率的影响有多少？（我回答的是并没有注意到效率的问题，更关注了精度）

## 第2部分

> 上来就进入了算法环节，当时慌得很。

1. 一个N\*M的矩阵，从左上角走到右下角，每次只能向下或者向右，走到每个格子的权值不同，请找到最短路径？

   > 经典动态规划问题，此外还让分析了一下时间复杂度，额，因为dp当时还没怎么刷题，更别说分析时间复杂度了。

2. 一个字符串和一个字母列表，问字母列表是否存在子集能够构成字符串的子串？

   > 思考了1分钟左右，完全没有什么想法。说到了暴力搜索，但是好像也没有阐述的太清晰。后面问了我是否需要思考时间，我想了一下，我就说这个题先过吧。

## 第3部分

> 主要是ML和DL的部分。

1. ML的模型你常用哪些？（我回答到LR和树模型）
2. 那问一下LR模型针对文本这种非结构化数据，在遇到效果不好的时候，你通常是怎么做的？
3. （接着LR的问题）调参一般你是怎么调的？
4. 在DNN网络中我们会常用的Dropout，请问，对于其他参数相同，dropout的rate不同的情况下，比如0.5和1，得到的结果是否相同？
5. 不对吧，据我所知，这个是不同的。那你能从原理上分析一下，为什么不同么？
6. 我们对于网络初始化，都有哪些方法呢？
7. 能跟我说一下不同初始化的方法对于网络有什么样的影响呢？（这个问题我没有回答上来，回答到为什么需要合理的初始化了）
8. 基本上他想问的就这些，问我有什么想问他的。

> 点评环节，说整体上还凑合吧，他比较注重比较偏基础的东西，（估计我的基础表现没能让他满意），确实算法部分和后面的DL的2个问题基本上回答的都不好。
