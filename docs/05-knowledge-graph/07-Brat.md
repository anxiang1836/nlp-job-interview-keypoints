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

3. 修改支持中文，记住待标注的文件名别有中文，要不然brat是不识别的。

   到./server/src/projectconfig.py第163行，修改为如下：

   ```python
   n = re.sub(u'[^a-zA-Z\u4e00-\u9fa5<>,0-9_-]', '_', n)
   ```

4. 在`medecial`文件夹下，创建`annotation.conf`和`visual.conf`。

   - `annotation.conf`：用于定义Schema
   - `visual.conf`：用于定义可视化的颜色和显示的别称

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

- 问题1：当2个实体存在包含关系时，怎么办？

  ![](https://raw.githubusercontent.com/anxiang1836/FigureBed/master/img/image-20200328170440499.png)

标注规则规定如下：

当`实体I`$$\subset$$`实体J`，标注为`实体J`

> 注：这个规则可以合并入下面的规则3

- 问题2：出现标注标准不同的情况，怎么办？

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