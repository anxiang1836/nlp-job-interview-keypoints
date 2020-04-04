# 标注Brat工具

## 1.安装过程

**Step1：**进入brat的主页，下载安装包。链接为：http://brat.nlplab.org/index.html

**Step2：**解压后，进行安装

```bash
./install.sh
```

**Step3：**配置apache2的内容来支持brat（macos和linux的配置不同，详见）

1. 进入apache2的目录：

   ```bash
   cd /private/etc/apache2
   ```

2. 编辑httpd.conf文件，在最末尾添加如下内容：

   ```bash
   <Directory "/opt/brat-v1.3_Crunchy_Frog">
      Options Indexes MultiViews FollowSymLinks
      Order allow,deny
      Allow from all
   
      AllowOverride Options Indexes FileInfo Limit
      AddHandler cgi-script .cgi
      AddType application/xhtml+xml .xhtml
      AddType font/ttf .ttf
   </Directory>
   ```

**Step4：**每次运行需要进入brat的根目录，然后运行：

```bash
python2 standalone.py
```

> 【Tips】
>
> 为了方便，我在oh my zsh的配置文件配置了便捷命令：
>
> ```bash
> alias cdbrat="cd /opt/brat-v1.3_Crunchy_Frog"
> alias runbrat="python2 standalone.py"
> ```
>
> 这样的话，每次再登入brat就OK了：
>
> ```bash
> cdbrat && runbrat
> ```

## 2.Brat使用指北

### 2.0 标注过程

1. 将待标注的文本数据copy到Brat的目录的data下：`/opt/brat-v1.3_Crunchy_Frog/data/medecial`

2. cd到上述文件夹中，然后对应生成空的`ann`文件（或者是运行第3章的预标注的脚本来生成）：

   ```bash
   find medical -name '*.txt' |sed -e 's|\.txt|.ann|g' |xargs touch
   ```

3. 修改支持中文，记住待标注的文件名别有中文，要不然brat是不识别的。到./server/src/projectconfig.py第163行，修改为如下：

   ```python
   n = re.sub(u'[^a-zA-Z\u4e00-\u9fa5<>,0-9_-]', '_', n)
   ```

4. 在`medecial`文件夹下，创建`annotation.conf`和`visual.conf`。

   `annotation.conf`：用于定义Schema

   `visual.conf`：用于定义可视化的颜色和显示的别称

### 2.1  标注Schema

**实体类别**

- 身体部位 Body
- 症状和体征 Symptom
- 疾病和诊断 Disease
- 检查和检验 Examine
- 治疗 Treatment

**关系类型**

- R1   Arg1:Body, Arg2:Symptom
- R2   Arg1:Examine,Arg2:Disease
- R3   Arg1:Body, Arg2:Disease
- R4   Arg1:Body, Arg2:Examine

### 2.2 标注中遇到的问题

#### 问题1：当2个实体存在包含关系时，怎么办？

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/image-20200328170440499.png)

标注规则规定如下：

当`实体I`$$\subset$$`实体J`，标注为`实体J`

> 注：这个规则可以合并入下面的规则3

#### 问题2：出现标注标准不同的情况，怎么办？

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200328231652.png)

规定标注规则如下：

- **规则1：**N*`实体A`+`实体B`，标注为`实体B`
- **规则2：**`实体A`+Others+`实体A`+`实体C`，分开标注；并分别标注`实体A`与`实体C`的关系。
- **规则3：**当`实体A`+`实体C`可构成实体`实体D`时，标注为`实体D`

> 上述规则中：
>
> `实体A`为`身体部位`
>
> `实体B`为`检查和检验`
>
> `实体C`为`症状与体征`
>
> `实体D`为`疾病和诊断`

## 3.根据词典预标注

> 这块主要思想是：用已知的部分领域词在语料中进行预标注，预生成ann文件，用以起到部分辅助标注的工作。
>
> > 在甲方爸爸这里拿到了一小部分的领域词，词数为1350词，主要包括：作战单元的类型与命名、部分作战区域的名称、常见气象名称。
>
> 预标注后需要人工来检测验证的问题：
>
> 1. 标注的词在文章出现重叠/交叉（根据标注规则，来选择是分开标注or最长标注）
> 2. 标注中出现错误（像比如把`武大靖`中的`武大`给标记出来了）
> 3. 标注中未标注的（在词典中未出现过的词，以及部分词的缩略称/别称）

### 3.1 预标注伪代码描述

1. **输入**

   - `enti_list`，其中每个元素为`(enti_name,enti_type)`
   - `unlabeled_txt_list`，其中每个元素为`txt`

2. **预标注伪代码**

   ```python
   def mark_ann(enti_list,txt_list):
   	for txt in txt_list:
       txt_name = txt.get_name # 获取txt的名字
       
       result = []
       count = 0
       for enti in enti_list: # (enti_name,enti_type)
         split_list = txt.split(enti)
         pre = -1
         
         enti_name = enti[0]
         enti_type = enti[1]
         len_word = len(enti_name)
         
         for idx,words in enumerate(split_list):
           if idx < len(split_list) - 1:
             start = pre + len(words) + 1
             end = start + len_word - 1
             pre = end
             count += 1
             result.append(("T" + str(count),enti_type,str(start),str(end),enti_name))
        
       with open("txt_name" + ".ann","w") as f:
         for idx,t in enumerate(result):
           f.write(" ".join(t))
           if idx < len(result) -1:
           	f.write("\n")
   ```

3. **时间复杂度**

设文本集$$T={t_1,t_2,...,t_N}$$中文本数量为`N`，已有词典$$D={d_1,d_2,...,d_K}$$，词数为`K`。

- 对于任意一篇文章$$t_i$$而言，复杂度主要取决于词典D中的每个词出现的频数：那么时间复杂度为`O(CK)`

  > `O(CK)`：`C`与文章的总长度和词在文章的频数有关

- 对于整个文本数据集T，时间复杂度为`O(N*CK)`

效率随着每篇文章的长度、词典的长度、文本集中文本个数增加而增加。

### 3.2 AC自动机算法原理

> Aho-Corasick算法是**多模式匹配**中的经典算法，该算法在1975年产生于贝尔实验室，是著名的多模匹配算法，目前在实际应用中较多。
>
> AC自动机的基本算法原理是基于：**Trie树数据结构** + **确定性有穷自动机**。这样，AC自动机能够在一次运行中找到给定集合所有字符串。

#### 3.2.1 算法步骤及原理

它大概分为三个步骤：**构建前缀树（生成goto表），添加失配指针（生成fail表），模式匹配（构造output表）**。下面，我们拿模式集合[say, she, shr, he, her]为例，构建一个AC 自动机。

##### 1. 构建Trie树

将模式串逐字符放进Trie树，如下图。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200402162147.png)

```javascript
class Trie {
      constructor() {
          this.root = new Node("root");
      }
      insert(word) {
          var cur = this.root;
          for (var i = 0; i < word.length; i++) {
              var c = word[i];
              var node = cur.children[c];
              if (!node) {
                  node = cur.children[c] = new Node(word[i]);
              }
              cur = node;
          }
          cur.pattern = word; //防止最后收集整个字符串用
          cur.endCount++; //这个字符串重复添加的次数
      }
  }
  function createGoto(trie, patterns) {
      for (var i = 0; i < patterns.length; i++) {
          trie.insert(patterns[i]);
      }
  }
```

我们尝试用它处理字符串sher。理想情况下是这样：

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200403114045.png)

> 很遗憾，前缀树只会顺着某一路径往下查找，最多到叶子节点折回树节点，继续选择另一条路径。
>
> 因此我们需要添加一些横向的路径，在失配时，跳到另一个分支上继续查找，保证搜索过的节点不会冗余搜索。

##### 2. 添加失配指针

很显然，对于每个节点，其失配指针应该指向其他子树中的表示同一字符的那些节点，并且它与其子树能构成剩下的最长后缀。

> 即，我们要匹配`sher`, 我们已经在某一子树中命中了`sh`，那么我们希望能在另一个子树中命中`er`。

![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/20200404092715.png)

现在的问题是，如何求fail指针？

我们发现`root`的每个儿子的fail都指向`root`（前缀和后缀是不会包含整个串的）。

> 也就是上图中root所连的`s`和`h`的fail都指向root。若已经求得`sh`所在点的fail，我们来考虑如何求`she`所在点的fail。
>
> 根据`sh`所在点的fail得到`h`是`sh`的最长后缀，而`h`又有儿子`e`，因此`she`的最长后缀应该是`he`，其fail指针就指向`he`所在点。

**概括AC自动机求fail指针的过程：**

1. 1.对整个字典树进行宽度优先遍历。

2. 若当前搜索到点`x`，那么对于`x`的第`i`个儿子(也就是代表字符`i`的儿子)：循环迭代`x`的兄弟节点，直到跳到某个点也有`i`这个儿子，`x`的第`i`个儿子的fail就指向这个点的儿子`i`。

   ```javascript
   function createFail(ac) {
       var root = ac.root;
       var queue = [root]; //root所在层为第0层
       while (queue.length) {
           //广度优先遍历
           var node = queue.shift();
           if (node) {
               //将其孩子逐个加入列队
               for (var i in node.children) {
                   var child = node.children[i];
                   if (node === root) {
                       child.fail = root; //第1层的节点的fail总是指向root
                   } else {
                       var p = node.fail; //第2层以下的节点, 其fail是在另一个分支上
                       while (p) {
                           //遍历它的孩子，看它们有没与当前孩子相同字符的节点
                           if (p.children[i]) {
                               child.fail = p.children[i];
                               break;
                           }
                           p = p.fail;
                       }
                       if (!p) {
                           child.fail = root;
                       }
                   }
                   queue.push(child);
               }
           }
       }
   }
   ```

##### 3. 模式匹配

我们从根节点开始查找，如果它的孩子能命中目标串的第1个字符串，那么我们就从这个孩子的孩子中再尝试命中目标串的第2个字符串。否则，我们就顺着它的失配指针，跳到另一个分支，找其他节点。

如果都没有命中，就从根节点重头再来。

> 当我们节点存在表示有字符串在它这里结束的标识时（如endCound, isEnd），我们就可以确认这字符串已经命中某一个模式串，将它放到结果集中。
>
> 如果这时长字符串还没有到尽头，我们继续收集其他模式串。

### 3.3 ahocorasick库

pyahocorasick是AC算法在python实现的工具库，直接pip install安装可能会有问题，用conda安装实现吧。

```bash
conda install -c https://conda.anaconda.org/conda-forge pyahocorasick
```

具体的食用方法如下：

```python
import ahocorasick
# https://blog.csdn.net/u010569893/article/details/97136696
def make_AC(AC, word_set):
    for word in word_set:
        AC.add_word(word,word)
    return AC
key_list = ["我爱你","爱你"]
AC_KEY = ahocorasick.Automaton()
AC_KEY = make_AC(AC_KEY, set(key_list))
AC_KEY.make_automaton()

content = "我爱你，塞北的雪，爱你，我爱你！"
for item in AC_KEY.iter(content):
    word = item[1]
    end = item[0]
    start = end - len(word) + 1
    print(start,end,word)
```

输出结果为：

```bash
0 2 我爱你
1 2 爱你
9 10 爱你
12 14 我爱你
13 14 爱你
```









