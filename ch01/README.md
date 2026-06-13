# 第1章 向量——从GPS坐标到词向量

你打开手机上的地图应用，搜索"天安门"，屏幕上跳出一串数字：北纬39.9042°，东经116.4074°。恭喜，你刚才使用了一个二维向量。

这个向量——$[39.9042, 116.4074]$——精确地定位了地球表面上的一个点。它只有两个维度（纬度和经度），但已经足以让你在十亿人口的城市里找到一座特定的建筑。如果向量能做这件事，那一个768维的向量能做什么？一个4096维的向量呢？

答案是：它能表示一个词的含义，一句话的情感，甚至一整篇文章的主题。欢迎来到向量的世界——这是理解大语言模型（LLM）所需的第一块，也是最关键的一块积木。

## 1.1 什么是向量——从GPS坐标说起

### 从坐标到向量：只换一个名字

我们继续从GPS坐标说起。假设你在规划一次旅行，要从北京去上海。北京的位置可以用一个有序的数字对来表示：

$$\mathbf{b} = [39.9042, 116.4074]$$

上海的位置是：

$$\mathbf{s} = [31.2304, 121.4737]$$

把这两个"有序的数字对"画在纸上，它们就是平面上的两个点。但如果你从北京向上海画一条带箭头的线段——这就变成了一个**向量**。

严格地说，向量（vector）是一个既有**大小**（length/magnitude）又有**方向**（direction）的量。GPS坐标对本身严格来说是一个"点"（point），但当我们把它放在坐标系中，从原点 $[0, 0]$ 指向这个点的那条箭头线段，就构成了一个向量。

在北京和上海的例子中，从北京到上海的位移向量是：

$$\vec{d} = \mathbf{s} - \mathbf{b} = [31.2304 - 39.9042, \; 121.4737 - 116.4074] = [-8.6738, \; 5.0663]$$

这个向量 $\vec{d}$ 告诉你：从北京出发，向南走约8.67度、向东走约5.07度，就到上海了。它同时编码了"多远"（大小）和"哪个方向"（方向）。

### 向量的基本运算

向量的运算规则简单得令人愉悦。让我们用更一般的记号：一个 $n$ 维向量是一个有序的数字列表

$$\mathbf{v} = [v_1, v_2, \ldots, v_n]$$

**加法**就是把对应位置的数字加起来。想象两个人同时推一张桌子：一个人向北推3牛顿，向东推1牛顿（力向量 $\mathbf{f_1} = [3, 1]$）；另一个人向北推1牛顿，向东推2牛顿（力向量 $\mathbf{f_2} = [1, 2]$）。合力就是：

$$\mathbf{f_1} + \mathbf{f_2} = [3+1, \; 1+2] = [4, 3]$$

几何上看，加法就是"把第二个向量的起点平移到第一个向量的终点"——也就是经典的平行四边形法则。

**数乘**（scalar multiplication）就是用一个标量（普通数字）乘以向量的每个分量。比如 $2 \cdot [3, 1] = [6, 2]$，相当于把力放大了两倍。$-1 \cdot [3, 1] = [-3, -1]$，方向反转了180度。数乘改变向量的大小，不改变方向（乘以负数除外，那会翻转方向）。

### 点积：两个向量有多"相似"

现在到了一个关键概念——**点积**（dot product），也叫内积。它有两种等价的定义方式。

代数定义：逐位相乘再求和。

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

几何定义：

$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \cdot \|\mathbf{b}\| \cdot \cos\theta$$

其中 $\|\mathbf{a}\|$ 是向量 $\mathbf{a}$ 的长度（模），$\theta$ 是两个向量之间的夹角。

后面这个几何定义非常重要。当你把点积除以两个向量的长度后，得到的 $\cos\theta$ 衡量的是两个向量的**方向一致性**：

- $\cos\theta = 1$：方向完全相同（最相似）
- $\cos\theta = 0$：互相垂直（毫无关系）
- $\cos\theta = -1$：方向完全相反（最不相似）

这就是机器学习中广泛使用的**余弦相似度**（cosine similarity）的数学根源：

$$\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$$

当你听到ChatGPT在"计算两个词的相似度"时，它的底层操作之一就是点积——在几万维的空间里做点积。

### 点积为什么能衡量相似度：投影的视角

代数定义（逐位相乘求和）和几何定义（$\|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$）为什么是同一件事？又为什么点积能衡量"相似度"？答案藏在一个被跳过的概念里——**投影**。

**几何含义：** 点积 $\mathbf{a}\cdot\mathbf{b}$ 等于 $\mathbf{b}$ 的长度，乘以 $\mathbf{a}$ 在 $\mathbf{b}$ 方向上的投影长度（带正负号）：

$$\mathbf{a}\cdot\mathbf{b} = \|\mathbf{b}\| \times (\mathbf{a}\text{ 在 }\mathbf{b}\text{ 上的投影长度})$$

把 $\mathbf{a}$ 想象成一束光，$\mathbf{b}$ 是地面——$\mathbf{a}$ 投在地面上的影子长度，就是投影长度。

**搭桥：** 投影长度正是 $\|\mathbf{a}\|\cos\theta$（$\mathbf{a}$ 越长、和 $\mathbf{b}$ 夹角越小，影子越长）。把它代回上式：

$$\mathbf{a}\cdot\mathbf{b} = \|\mathbf{b}\| \cdot \|\mathbf{a}\|\cos\theta = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$$

这就是两个定义等价的真正原因——**点积的几何本质是投影**，而 $\cos\theta$ 只是投影长度的另一种写法。这也解释了相似度：两个向量方向越一致（$\theta$ 越小，投影越长），点积越大；互相垂直时投影为零，点积为零。注意力机制用 $\mathbf{q}\cdot\mathbf{k}$ 衡量一个 token 对另一个 token 的"关注度"，本质上就是在算 query 在 key 方向上的投影有多大。

**一个 2 维小例：** 取 $\mathbf{a}=[3,1]$，$\mathbf{b}=[2,4]$。

- 代数：$\mathbf{a}\cdot\mathbf{b} = 3\times2 + 1\times4 = 10$
- 投影：$\mathbf{a}$ 在 $\mathbf{b}$ 上的投影长度 $= \dfrac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{b}\|} = \dfrac{10}{\sqrt{2^2+4^2}} = \dfrac{10}{\sqrt{20}} \approx 2.24$
- 夹角：$\cos\theta = \dfrac{10}{\sqrt{10}\cdot\sqrt{20}} \approx 0.707$，即 $\theta = 45°$

```python
import numpy as np
a, b = np.array([3., 1.]), np.array([2., 4.])
print("点积 =", a @ b)                       # 10.0
print("a 在 b 上的投影长度 =", a @ b / np.linalg.norm(b))  # 2.236
print("夹角(度) =", np.degrees(np.arccos(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)))))  # 45.0
```

> **一句话锚点：** 点积 = 一个向量在另一个向量方向上的"分量大小"；分量越大越相似，垂直则为零。

## 1.2 向量的维度——从音乐特征到人格画像

### Spotify怎么知道你想听什么

当你打开Spotify的"发现"页面，它为什么能推荐你从没听过却很喜欢的歌？秘密之一是：Spotify为每首歌都维护了一个高维**特征向量**。

通过Spotify的音频分析API，每首歌可以被提取出如下特征：

| 维度 | 含义 | 范围 |
|------|------|------|
| acousticness | 原声程度 | 0~1 |
| danceability | 适合跳舞的程度 | 0~1 |
| energy | 能量感 | 0~1 |
| instrumentalness | 纯器乐程度 | 0~1 |
| liveness | 现场感 | 0~1 |
| loudness | 响度 | -60~0 dB |
| speechiness | 人声占比 | 0~1 |
| tempo | 节奏速度 | 0~300 BPM |
| valence | 情绪积极性 | 0~1 |

一首歌可以被表示为一个9维向量。比如，贝多芬的《月光奏鸣曲》可能是：

$$\mathbf{moonlight} = [0.95, \; 0.12, \; 0.08, \; 0.92, \; 0.05, \; -18, \; 0.03, \; 60, \; 0.10]$$

而一首欢快的流行舞曲可能是：

$$\mathbf{dance\_pop} = [0.05, \; 0.92, \; 0.88, \; 0.00, \; 0.10, \; -4, \; 0.35, \; 128, \; 0.85]$$

如果计算这两个向量的余弦相似度，你会发现它们几乎垂直——完全不同的歌。但如果拿两首民谣来算，它们的向量就会非常接近，余弦相似度接近1。

这就是推荐系统的核心直觉：**把物品表示为向量，然后通过向量间的距离或角度来度量相似性**。

### 维度越高，表达越丰富

9维向量已经能区分歌曲的风格了。但如果我们想表达更复杂的东西呢？

考虑"描述一个人的性格"这件事。大五人格模型（Big Five）用5个维度：开放性、尽责性、外向性、宜人性、神经质。5个数字就能给一个人的性格画出一个粗略的画像。

但如果想要更精细的画像呢？你可能需要加上：幽默感、好奇心、冒险精神、同理心、审美偏好、社交风格……每多一个维度，你的描述就更精确一分。

数学上也是一样的道理。在向量空间中，增加维度意味着增加了更多的"自由度"。一个 $n$ 维向量可以区分的空间中的点数，随着 $n$ 的增长呈指数级增加。这意味着高维向量能够编码极其丰富的信息——这也正是LLM使用高维向量的原因。GPT-3的隐藏状态维度是12288，也就是说，在模型的"大脑"中，每一个词都被表示为一个12288维的向量。

### 维度灾难：更多的维度不总是更好

但维度也不是越高越好。当我们把向量放到很高维的空间中时，一个反直觉的现象出现了：**在高维空间中，几乎所有点之间的距离都差不多远**。

举个例子：在一个单位正方体中随机撒两个点。在1维中（单位线段），两点期望距离是1/3。在2维中（单位正方形），期望距离约0.52。到了100维，期望距离变成了约4.08——而正方体的对角线长度为 $\sqrt{100} = 10$。也就是说，任意两点的距离都挤在4~5之间，相对于对角线长度10来说，差距变得很小，很难区分"近"和"远"了。

这意味着：如果盲目增加维度，向量之间的差异性反而会被稀释，计算相似度就变得不可靠了。这就是所谓的**维度灾难**（curse of dimensionality）。LLM的设计者需要在"足够高的维度以表达丰富的语义"和"不要高到丧失区分度"之间找到平衡——这就是为什么实践中常用的维度是768、1024、4096这样的数字，而不是一百万。

## 1.3 词向量：语言在计算机中的灵魂

### 一个改变NLP历史的直觉

2013年，Google的Tomas Mikolov等人发表了一篇论文《Efficient Estimation of Word Representations in Vector Space》，提出了Word2Vec。这个方法的核心直觉极其优雅：

**语义相近的词，出现在相似的上下文中；出现在相似上下文中的词，它们的向量表示应该相近。**

具体来说，Word2Vec有两种训练方式。我们以**Skip-gram**模型为例：给定一个词，预测它的上下文。假设训练语料中有一句话"猫坐在垫子上"，当模型看到"坐"这个词时，它需要学会预测"猫""在""垫子""上"这些周围词。

训练目标是最大化：

$$\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \le j \le c, j \ne 0} \log p(w_{t+j} \mid w_t)$$

其中 $T$ 是语料长度，$c$ 是上下文窗口大小，$p(w_{t+j} \mid w_t)$ 是给定中心词 $w_t$ 预测上下文词 $w_{t+j}$ 的概率。这个概率用softmax计算：

$$p(w_O \mid w_I) = \frac{\exp(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I})}{\sum_{w=1}^{W} \exp(\mathbf{v}'_w \cdot \mathbf{v}_{w_I})}$$

其中 $\mathbf{v}_{w_I}$ 是输入词的向量，$\mathbf{v}'_{w_O}$ 是输出词的向量，$W$ 是词表大小。

训练完成后，每个词就获得了一个固定维度的向量（通常是100~300维）。这些向量有什么神奇的特性呢？

### 国王减去男人加上女人等于女王

Word2Vec最著名的例子就是：

$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$

这个等式说的是："国王"向量减去"男人"向量，提取出的是"royalty"（王权）这个语义成分。再加上"女人"向量，得到的最接近的词就是"女王"。

这不是被硬编码进去的——这是模型通过大量文本自动学到的。它在向量空间中发现了一个"性别"方向：从"男人"指向"女人"的方向，和从"国王"指向"女王"的方向，大致相同。

类似的关系还有：

$$\mathbf{v}_{\text{Paris}} - \mathbf{v}_{\text{France}} + \mathbf{v}_{\text{Italy}} \approx \mathbf{v}_{\text{Rome}}$$

$$\mathbf{v}_{\text{walked}} - \mathbf{v}_{\text{walk}} + \mathbf{v}_{\text{swim}} \approx \mathbf{v}_{\text{swam}}$$

这意味着词向量空间具有某种**代数结构**——语义关系被编码为向量空间中的方向和偏移量。这是一个深刻的发现：语言的语义结构，在足够大的数据上训练后，可以被几何结构所捕捉。

### 用Python看看词向量长什么样

下面用简单的代码演示词向量的基本操作。这里我们不加载完整的Word2Vec模型，而是用模拟数据来演示核心概念（因为加载真实模型需要下载几百MB的文件）。如果你有gensim和真实的词向量，只需取消注释相关行即可。

```python
import numpy as np

# --- 模拟一组简化的词向量（300维，这里只展示原理）---
np.random.seed(42)
dim = 300

# 模拟词向量：我们手动构造一些语义关系
def make_word_vector(base, noise_scale=0.1):
    """基于一个基底向量加噪声生成一个词向量"""
    return base + np.random.randn(dim) * noise_scale

# 语义基底向量
royalty = np.random.randn(dim)
male = np.random.randn(dim)
female = np.random.randn(dim)
country = np.random.randn(dim)
capital = np.random.randn(dim)

# 构造词向量
vectors = {
    "king":   make_word_vector(royalty + male, 0.05),
    "queen":  make_word_vector(royalty + female, 0.05),
    "man":    make_word_vector(male, 0.05),
    "woman":  make_word_vector(female, 0.05),
    "france": make_word_vector(country, 0.05),
    "paris":  make_word_vector(country + capital, 0.05),
    "italy":  make_word_vector(country + 0.1 * np.random.randn(dim), 0.05),
    "rome":   make_word_vector(country + capital + 0.1 * np.random.randn(dim), 0.05),
}

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 经典的 king - man + woman ≈ queen 实验
result = vectors["king"] - vectors["man"] + vectors["woman"]

# 找到与结果最接近的词（排除 king, man, woman 本身）
exclude = {"king", "man", "woman"}
similarities = {
    word: cosine_similarity(result, vec)
    for word, vec in vectors.items()
    if word not in exclude
}

# 按相似度排序
sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
print("king - man + woman 最接近的词：")
for word, sim in sorted_words[:3]:
    print(f"  {word}: {sim:.4f}")

# 打印一些词之间的余弦相似度
print("\n余弦相似度对比：")
print(f"  king vs queen:  {cosine_similarity(vectors['king'], vectors['queen']):.4f}")
print(f"  king vs man:    {cosine_similarity(vectors['king'], vectors['man']):.4f}")
print(f"  king vs france: {cosine_similarity(vectors['king'], vectors['france']):.4f}")
```

运行结果类似：

```
king - man + woman 最接近的词：
  queen: 0.9975
  paris: 0.0884
  rome:  0.0882

余弦相似度对比：
  king vs queen:  0.4645
  king vs man:    0.6852
  king vs france: 0.0185
```

可以看到，"king - man + woman"的结果向量与"queen"的相似度远高于其他词——这正是Word2Vec在真实数据上展现的语义算术特性。如果你用真实的预训练词向量运行这段代码，效果会更加明显。

### 词嵌入空间中的几何关系

词向量空间中蕴含着丰富的几何结构。除了上面展示的"类比"关系（平行四边形结构），还有一些有趣的几何特性：

- **聚类**：语义相近的词在空间中聚在一起。所有表示颜色的词（红、蓝、绿）形成一个簇，所有表示动物的词（猫、狗、兔）形成另一个簇。
- **方向**：空间中某些特定方向编码了特定的语义属性。比如存在一个"性别"方向，沿着它走，词就从男性变女性；存在一个"时态"方向，沿着它走，词就从过去变现在。
- **相对位置**：两个词之间的差异向量（如 $\mathbf{v}_{\text{queen}} - \mathbf{v}_{\text{king}}$）编码了它们之间的语义关系。

这些几何结构并非人为设计，而是模型从海量文本中自动发现的。这也解释了为什么词向量可以被迁移到各种下游任务——因为它们捕获了语言的深层结构。

## 1.4 向量与LLM

### 一切皆向量：LLM的数据流

理解LLM的一个关键洞察是：**在LLM的整个处理流程中，信息始终以向量的形式存在。**

让我们追踪一个输入——比如句子"我爱北京"——在LLM中的旅程：

1. **分词**（Tokenization）：句子被切分成token，比如 `["我", "爱", "北京"]`。
2. **查找Embedding表**：每个token被映射为一个高维向量。比如"我"变成 $\mathbf{e}_{\text{我}} \in \mathbb{R}^{d}$，其中 $d$ 是模型的嵌入维度。
3. **Transformer层**：向量通过一层又一层的Transformer block，每层都在做"向量运算"——自注意力机制本质上是在计算向量之间的相关性（用点积！），前馈网络本质上是在做矩阵乘向量。
4. **输出**：最后一层输出的仍然是向量，这个向量被乘以一个矩阵（"unembedding matrix"），映射回词表空间，得到每个词作为下一个词的概率。

用一句话概括：**输入是向量，中间是向量，输出也是向量。** 离散的文本符号只在最外面存在了一瞬间，进入模型后，一切都是连续的向量世界。

### Embedding层：把离散的词变成连续的向量

Embedding层本质上是一个**查找表**（lookup table）。假设词表大小为 $V$，嵌入维度为 $d$，那么Embedding层就是一个 $V \times d$ 的矩阵 $E$。

$$E = \begin{bmatrix} \text{---} & \mathbf{e}_1 & \text{---} \\ \text{---} & \mathbf{e}_2 & \text{---} \\ & \vdots & \\ \text{---} & \mathbf{e}_V & \text{---} \end{bmatrix} \in \mathbb{R}^{V \times d}$$

当输入token的ID是 $k$ 时，Embedding层就返回矩阵 $E$ 的第 $k$ 行：$\mathbf{e}_k \in \mathbb{R}^d$。这个操作可以写为：

$$\mathbf{e}_k = E[k, :]$$

其中 $E[k, :]$ 表示取矩阵 $E$ 的第 $k$ 行。

这就是从离散到连续的桥梁——一个整数ID（token编号）变成了一个浮点数向量。这个向量在训练开始时是随机的，随着训练的进行，它被逐渐调整，使得语义相近的词拥有相近的向量。

在Transformer论文《Attention Is All You Need》中，作者还提到了一个细节：他们将输入Embedding乘以 $\sqrt{d}$（$d$ 是模型维度），目的是让Embedding的数值范围与位置编码（Positional Encoding）的数值范围匹配：

$$\text{Input} = \text{Embedding}(x) \times \sqrt{d_{\text{model}}} + \text{PositionalEncoding}$$

### 为什么LLM的隐藏维度是4096、8192这样的数字

如果你留意过各种LLM的参数配置，会发现隐藏维度（hidden dimension / $d_{\text{model}}$）通常是特定的数字：GPT-2是768/1024/1280/1600，GPT-3是12288，LLaMA-7B是4096，LLaMA-70B是8192。

这些数字不是随机的。它们受到以下几个因素的制约：

**1. 多头注意力的整除约束。** Transformer使用多头注意力（Multi-Head Attention），$d_{\text{model}}$ 必须能被注意力头数 $h$ 整除，这样每个头的维度 $d_k = d_{\text{model}} / h$ 才是整数。比如 $d_{\text{model}} = 4096$，$h = 32$，则 $d_k = 128$。

**2. 硬件对齐。** GPU的计算效率高度依赖于数据对齐。NVIDIA GPU的Tensor Core在矩阵维度是8的倍数（甚至64的倍数）时效率最高。所以 $d_{\text{model}}$ 通常取 $256 \times k$（256, 512, 768, 1024, 2048, 4096, 8192, ...）。

**3. 参数预算与模型能力的平衡。** 在总参数量固定的情况下，增加 $d_{\text{model}}$ 意味着模型能编码更丰富的信息，但也意味着每一层的矩阵更大、计算量更高。实践中，研究者发现 $d_{\text{model}}$ 在几千到一万这个范围内取得了好的性价比。

用一个简单的Python脚本来感受一下维度对参数量的影响：

```python
def count_transformer_params(d_model, n_heads, n_layers, vocab_size=50000, ffn_multiplier=4):
    """估算一个Transformer语言模型的参数量"""
    d_ff = d_model * ffn_multiplier  # FFN中间层维度

    # 每一层的参数
    # 自注意力: Q, K, V, O 四个矩阵
    attn_params = 4 * d_model * d_model
    # FFN: 两个线性层
    ffn_params = d_model * d_ff + d_ff * d_model  # = 2 * d_model * d_ff
    # LayerNorm: 两个，各 d_model 个参数
    ln_params = 2 * 2 * d_model

    params_per_layer = attn_params + ffn_params + ln_params
    total = n_layers * params_per_layer

    # Embedding + 输出头（通常共享权重）
    embedding_params = vocab_size * d_model

    total += embedding_params
    return total

# 对比不同隐藏维度
configs = [
    ("GPT-2 Small",    768,  12, 12),
    ("GPT-2 Medium",   1024, 16, 24),
    ("GPT-2 Large",    1280, 20, 36),
    ("LLaMA-7B",       4096, 32, 32),
    ("LLaMA-70B",      8192, 64, 80),
]

print(f"{'模型':<16} {'d_model':>8} {'n_heads':>8} {'n_layers':>9} {'参数量':>12}")
print("-" * 58)
for name, d, h, l in configs:
    p = count_transformer_params(d, h, l)
    print(f"{name:<16} {d:>8} {h:>8} {l:>9} {p/1e9:>10.2f}B")
```

输出类似：

```
模型               d_model   n_heads   n_layers         参数量
----------------------------------------------------------
GPT-2 Small          768       12        12       0.12B
GPT-2 Medium        1024       16        24       0.35B
GPT-2 Large         1280       20        36       0.77B
LLaMA-7B            4096       32        32       6.65B
LLaMA-70B           8192       64        80      64.84B
```

注意参数量随 $d_{\text{model}}$ 的平方增长——因为每一层的注意力矩阵是 $d \times d$，FFN是 $d \times 4d$。这就是为什么从7B到70B，维度"只"翻了一倍（4096到8192），但层数也从32涨到了80——参数量的增长来自维度和层数的双重增加。

### 向量在Transformer中的完整旅程

让我们把这一章的内容串起来，看看向量在Transformer中的一个完整计算步骤。对于输入序列中的某一个位置 $t$ 的词 $x_t$：

**Step 1: Embedding + 位置编码**

$$\mathbf{h}_t^{(0)} = \mathbf{e}_{x_t} \cdot \sqrt{d_{\text{model}}} + \mathbf{p}_t$$

其中 $\mathbf{e}_{x_t}$ 是token的embedding向量，$\mathbf{p}_t$ 是位置 $t$ 的位置编码向量（也是 $d_{\text{model}}$ 维）。两者相加（都是同维向量相加），得到了该位置的初始表示。

**Step 2: 自注意力（Self-Attention）**

在每一层 $l$，首先计算Query、Key、Value三个向量：

$$\mathbf{q}_t = \mathbf{h}_t^{(l-1)} W^Q, \quad \mathbf{k}_t = \mathbf{h}_t^{(l-1)} W^K, \quad \mathbf{v}_t = \mathbf{h}_t^{(l-1)} W^V$$

其中 $W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$ 是可学习的投影矩阵。注意：这就是用矩阵乘法将一个 $d_{\text{model}}$ 维向量投影为 $d_k$ 维向量。

然后计算注意力权重：

$$\alpha_{t,s} = \frac{\exp(\mathbf{q}_t \cdot \mathbf{k}_s / \sqrt{d_k})}{\sum_{s'} \exp(\mathbf{q}_t \cdot \mathbf{k}_{s'} / \sqrt{d_k})}$$

看到了吗？分子上就是一个**点积**——我们在1.1节学的那个点积。它衡量的是：位置 $t$ 的query向量和位置 $s$ 的key向量有多"相似"。越相似，注意力权重越大，意味着位置 $t$ 越关注位置 $s$ 的信息。

最终，自注意力的输出是所有位置的value向量的加权和：

$$\text{Attention}(\mathbf{q}_t) = \sum_s \alpha_{t,s} \cdot \mathbf{v}_s$$

**Step 3: 前馈网络（FFN）**

注意力层之后是一个两层的前馈网络：

$$\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x} W_1 + \mathbf{b}_1) W_2 + \mathbf{b}_2$$

其中 $W_1 \in \mathbb{R}^{d_{\text{model}} \times 4d_{\text{model}}}$，$W_2 \in \mathbb{R}^{4d_{\text{model}} \times d_{\text{model}}}$。输入一个 $d_{\text{model}}$ 维向量，先"升维"到 $4 \cdot d_{\text{model}}$，经过ReLU激活，再"降维"回 $d_{\text{model}}$。

每一步操作——矩阵乘向量、向量加法、逐元素非线性——本质上都在对向量进行变换，逐步提取和组合信息。

---

从GPS坐标到词向量，从二维空间到4096维空间，向量的核心思想始终如一：**用一个有序的数字列表来表示事物的特征，用向量之间的运算来衡量和操作这些特征之间的关系。** 当你下次听到"大语言模型"时，不妨把它想象成一台在超 high 维空间中做向量运算的机器——输入一句话，它就在这个空间中导航，最终在正确的方向上找到下一个词。

下一章我们将深入矩阵——向量运算的"批量版"，看看Transformer中的矩阵乘法如何让一切高效运转。
