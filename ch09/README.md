# 第9章 注意力机制的线性代数

2017年，Google的一篇论文《Attention Is All You Need》横空出世，把一个叫做"注意力机制"的东西推上了舞台中央。九年过去了，这篇论文催生了GPT、BERT、Claude等一系列改变世界的大语言模型。而这一切的核心——注意力公式——本质上就是几个矩阵乘法。

你没有看错。ChatGPT背后那个被认为"涌现了智能"的模型，其最核心的运算，就是你在第一章学过的点积、第二章学过的矩阵乘法和第三章学过的线性变换。今天我们要做的事情，就是把 Transformer 论文里那行看似复杂的注意力公式，一个符号一个符号地拆开，让你看到它优雅而朴素的线性代数内核。

---

## 9.1 注意力：从"图书馆检索"说起

### 一个生活类比

想象你走进一家图书馆，手里拿着一个问题："如何做红烧肉？"

这家图书馆有几万本书。每本书的封面上有一个标题（Key），书里有具体内容（Value）。你不会真的去翻每一本书——你的眼睛会快速扫过那些标题，和你的问题做比较。哪些标题跟"红烧肉"最相关，你的注意力就集中在那些书上，然后你去翻阅它们的内容。

这个过程中有三个角色：

- **Query（Q）**：你脑子中的问题——"如何做红烧肉？"
- **Key（K）**：每本书的标题——"中式家常菜100道"、"量子力学导论"、"红烧肉秘方大全"……
- **Value（V）**：每本书的具体内容

你的大脑会计算 Query 和每个 Key 之间的**匹配程度**，然后按匹配程度的高低去"提取"对应的 Value。这就是注意力机制的核心思想：

> **注意力 = 一种加权检索机制：用 Query 和 Key 算匹配度，用匹配度作为权重去聚合 Value。**

在深度学习里，Q、K、V 不是人工设计的，而是从输入数据中自动学出来的。怎么学？靠的就是矩阵乘法。

---

## 9.2 Query、Key、Value 的生成

### 从输入到 Q、K、V：三次线性变换

假设我们有一句输入："猫坐在垫子上"。经过词嵌入（embedding）后，这句话变成了一个矩阵 $X$，形状为 $(n, d_{\text{model}})$，其中 $n$ 是词的数量（序列长度），$d_{\text{model}}$ 是每个词向量的维度。

要从 $X$ 得到 Q、K、V，我们只需要做三次矩阵乘法——对，就是第三章学的线性变换：

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

其中 $W_Q, W_K, W_V$ 是三个可学习的权重矩阵。它们的形状是：

| 符号 | 形状 | 含义 |
|------|------|------|
| $X$ | $(n, d_{\text{model}})$ | 输入序列，每行是一个词的嵌入向量 |
| $W_Q$ | $(d_{\text{model}}, d_k)$ | Query 的投影矩阵 |
| $W_K$ | $(d_{\text{model}}, d_k)$ | Key 的投影矩阵 |
| $W_V$ | $(d_{\text{model}}, d_v)$ | Value 的投影矩阵 |
| $Q$ | $(n, d_k)$ | 每个词的 Query 向量 |
| $K$ | $(n, d_k)$ | 每个词的 Key 向量 |
| $V$ | $(n, d_v)$ | 每个词的 Value 向量 |

在原始 Transformer 论文中，$d_{\text{model}} = 512$，$d_k = d_v = 64$。

这三步做的事情很简单：**把每个词的通用嵌入向量，分别投影到三个不同的"角色空间"**。就像一个演员可以穿上不同的戏服，同一个词向量经过不同的权重矩阵，就分别扮演了"提问者"（Query）、"被检索者"（Key）和"内容提供者"（Value）三个角色。

这三组权重矩阵是模型在训练过程中自动学出来的。训练完成后，模型就"知道"应该怎样把一个词映射到三个不同的子空间，使得注意力机制能最好地捕捉词与词之间的关系。

```python
import numpy as np

# 模拟输入：3个词，每个词512维嵌入
n, d_model, d_k, d_v = 3, 512, 64, 64
X = np.random.randn(n, d_model)

# 三个可学习的权重矩阵
W_Q = np.random.randn(d_model, d_k) * 0.01
W_K = np.random.randn(d_model, d_k) * 0.01
W_V = np.random.randn(d_model, d_v) * 0.01

# 生成 Q, K, V
Q = X @ W_Q   # (3, 64)
K = X @ W_K   # (3, 64)
V = X @ W_V   # (3, 64)

print(f"X shape: {X.shape}")  # (3, 512)
print(f"Q shape: {Q.shape}")  # (3, 64)
print(f"K shape: {K.shape}")  # (3, 64)
print(f"V shape: {V.shape}")  # (3, 64)
```

---

## 9.3 Self-Attention 公式的逐项拆解

现在我们来看 Transformer 论文中那个著名的公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

短短一行，却浓缩了四个步骤。让我们一个一个拆开。

### 第一步：$QK^T$——计算相似度矩阵

$Q$ 的形状是 $(n, d_k)$，$K^T$ 的形状是 $(d_k, n)$，所以 $QK^T$ 的形状是 $(n, n)$。

这个 $(n, n)$ 的矩阵是什么意思？它的第 $i$ 行第 $j$ 列的元素，就是第 $i$ 个词的 Query 向量和第 $j$ 个词的 Key 向量的**点积**：

$$[QK^T]_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j = \sum_{l=1}^{d_k} q_{il} \cdot k_{jl}$$

还记得第一章学的点积吗？点积衡量的是两个向量的"相似度"。点积越大，说明 Query 和 Key 越"匹配"。所以 $QK^T$ 这个矩阵的每一行，描述了某一个位置和序列中所有位置的匹配程度。

回到图书馆的类比：$QK^T$ 就好比你在图书馆里，把你的问题和每一本书的标题都比对了一遍，给每本书打了一个"匹配分数"。

```python
# 第一步：计算 QK^T
scores = Q @ K.T   # (3, 3)
print("相似度矩阵 scores:")
print(np.round(scores, 2))
# scores[i][j] = 第i个词对第j个词的"关注度分数"
```

### 第二步：除以 $\sqrt{d_k}$——缩放

为什么要除以 $\sqrt{d_k}$？这和 softmax 的数值性质有关。

当 $d_k$ 比较大时（比如 64），$QK^T$ 的元素值也会比较大。原因很简单：点积是 $d_k$ 个乘积项的求和。假设 $q_i$ 和 $k_j$ 的每个分量都是均值为 0、方差为 1 的独立随机变量，那么 $\mathbf{q}_i \cdot \mathbf{k}_j$ 的方差就是 $d_k$。也就是说，随着维度增大，点积的绝对值倾向于增大。

这不是什么好事。因为接下来要对这些分数做 softmax，而 softmax 遇到很大的输入值时，梯度会变得极小（这叫"梯度消失"）。直观地说，如果某个分数比其他分数大很多，softmax 就会输出一个"几乎 one-hot"的分布——只有最大值那个位置接近 1，其他全接近 0。这个分布的梯度几乎为零，模型就学不动了。

除以 $\sqrt{d_k}$ 就是在做**方差归一化**。除完之后，点积的方差回到了 1 左右，softmax 就能健康地工作了。这就是为什么 Transformer 的注意力叫做"缩放点积注意力"（Scaled Dot-Product Attention）。

$$\text{scores} = \frac{QK^T}{\sqrt{d_k}}$$

从概率论的角度理解：如果 $q$ 和 $k$ 的每个分量独立同分布，均值为 0，方差为 1，那么点积 $q \cdot k$ 的均值为 0，方差为 $d_k$。除以 $\sqrt{d_k}$ 后方差变为 1，消除了维度对数值范围的影响。

```python
# 第二步：缩放
d_k = Q.shape[-1]
scaled_scores = scores / np.sqrt(d_k)
print(f"\nd_k = {d_k}, sqrt(d_k) = {np.sqrt(d_k):.2f}")
print("缩放后:", np.round(scaled_scores, 2))
```

### 第三步：softmax——归一化为概率分布

softmax 函数把一个实数向量变换成一个概率分布（所有元素为正且和为 1）。对矩阵的每一行分别做 softmax：

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

对于我们的注意力分数矩阵，softmax 作用在每一行上：第 $i$ 行经过 softmax 之后，就变成了第 $i$ 个位置对所有位置的"注意力权重"。这些权重都是正数，且和为 1，可以理解为一个概率分布。

```python
def softmax(x, axis=-1):
    """数值稳定的 softmax 实现"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)   # 减去最大值，防止数值溢出
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# 第三步：softmax 归一化
attn_weights = softmax(scaled_scores)
print("\n注意力权重矩阵:")
print(np.round(attn_weights, 3))
print("每行之和:", np.round(attn_weights.sum(axis=1), 4))  # 每行都是 1.0
```

注意 softmax 的实现中有一个细节：先减去最大值再做指数运算。这纯粹是数值稳定的考虑——$e^{100}$ 会溢出，但 $e^{100 - 100} = 1$ 不会。减去最大值不影响 softmax 的结果（因为分子分母同时除以 $e^{\max}$），但避免了数值溢出。

### 第四步：乘以 V——加权聚合

最后一步：用注意力权重去加权 Value 矩阵。

$$\text{output} = \text{Attention\_Weights} \times V$$

注意力权重矩阵的形状是 $(n, n)$，$V$ 的形状是 $(n, d_v)$，所以输出的形状是 $(n, d_v)$。

输出的第 $i$ 行是：

$$\text{output}_i = \sum_{j=1}^{n} \text{attn\_weight}_{ij} \cdot \mathbf{v}_j$$

这就是一个加权求和：第 $i$ 个位置的新表示，是所有位置的 Value 向量的加权平均，权重就是注意力分数。哪个位置获得了更高的注意力，它的 Value 就对输出贡献更大。

回到图书馆：你根据匹配度确定了每本书的"关注度"，然后按照关注度去阅读这些书的内容。关注度高的书你仔细读了（权重大），关注度低的书你只是扫了一眼（权重小）。

```python
# 第四步：加权聚合
output = attn_weights @ V   # (3, 3) @ (3, 64) = (3, 64)
print(f"\n输出形状: {output.shape}")  # (3, 64)
print("输出的第一行（第一个词的新表示）:", np.round(output[0, :5], 3), "...")
```

### 合在一起：完整的 Self-Attention

把四步合在一起，就是论文里的公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

输入一个 $(n, d_{\text{model}})$ 的矩阵 $X$，经过三次线性变换得到 Q、K、V，然后通过上述四步运算，输出一个 $(n, d_v)$ 的矩阵。整个过程就是一连串矩阵乘法和逐元素运算——纯粹的线性代数。

---

## 9.4 注意力矩阵的性质

理解了计算过程之后，我们来看看注意力权重矩阵（softmax 之后的那个 $(n, n)$ 矩阵）有哪些有趣的数学性质。

### 每一行是一个概率分布

因为 softmax 的缘故，注意力矩阵的每一行元素都为正且和为 1。这使它成为一个**行随机矩阵**（row-stochastic matrix）。每一行可以理解为：对于序列中的第 $i$ 个位置，它将"注意力预算"如何分配给所有位置。

### 不对称性

Self-Attention 的相似度矩阵 $QK^T$ 一般**不是对称矩阵**。这是因为 $Q$ 和 $K$ 是由不同的权重矩阵 $W_Q$ 和 $W_K$ 生成的。$\mathbf{q}_i \cdot \mathbf{k}_j$ 通常不等于 $\mathbf{q}_j \cdot \mathbf{k}_i$。

这很好理解：在"猫坐在垫子上"这句话中，"猫"关注"坐"的程度（"猫"是 Query，"坐"是 Key），和"坐"关注"猫"的程度（"坐"是 Query，"猫"是 Key），完全可能是不同的。注意力是有方向的。

### 秩与表达能力

注意力矩阵的秩最多为 $\min(n, d_k)$。在实践中 $d_k$ 通常远小于序列长度 $n$（例如 $d_k = 64$ 而 $n$ 可以是几千），这意味着注意力矩阵是一个**低秩矩阵**。这带来一个好处：注意力矩阵可以用更少的参数来近似，这也是一些高效注意力方法的理论基础。

### 注意力往往集中在少数位置

虽然注意力矩阵的每一行都是一个"满"的概率分布，但在实际训练好的模型中，你会发现大部分注意力权重集中在少数几个位置上，其余位置的权重接近于 0。这和人类阅读时的直觉一致——我们不会均匀地关注句子中的每一个词，而是把注意力聚焦在少数关键信息上。

---

## 9.5 多头注意力：多视角的线性代数

### 为什么要多头

单头注意力只有一个"视角"。但语言中的关系是多样的：一句话中可能同时存在语法关系（主语-谓语）、语义关系（同义词）、位置关系（相邻词）。我们希望模型能同时捕捉多种不同类型的关系。

解决方案很直觉：**让多组不同的 Q、K、V 同时工作**。每一组叫做一个"头"（head），每个头可以学会关注不同类型的关系。然后在最后把所有头的结果拼在一起，再做一次线性投影融合信息。

### 多头注意力的矩阵运算

Transformer 原始论文使用 $h = 8$ 个头，每个头的维度 $d_k = d_v = d_{\text{model}} / h = 512 / 8 = 64$。

对于第 $i$ 个头：

$$\text{head}_i = \text{Attention}(QW_Q^{(i)},\ KW_K^{(i)},\ VW_V^{(i)})$$

每个头的输出形状是 $(n, d_v) = (n, 64)$。然后把 $h = 8$ 个头的输出拼接（concatenate）起来：

$$\text{MultiHead} = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)$$

拼接后的形状是 $(n, h \cdot d_v) = (n, 512)$，正好等于 $d_{\text{model}}$。最后再做一次线性投影：

$$\text{Output} = \text{MultiHead} \cdot W_O$$

其中 $W_O$ 的形状是 $(h \cdot d_v, d_{\text{model}}) = (512, 512)$。

### 形状变换的完整追踪

让我们追踪一个具体例子：$n = 4$（四个词），$d_{\text{model}} = 512$，$h = 8$，$d_k = d_v = 64$。

```
输入 X:          (4, 512)
                    |
        +-----------+-----------+
        |           |           |
    X @ W_Q    X @ W_K    X @ W_V      各 (512, 512)
        |           |           |
    Q (4,512)  K (4,512)  V (4,512)
        |           |           |
    按头切分    按头切分    按头切分       8组，每组 (4, 64)
        |           |           |
    Q_i(4,64)  K_i(4,64)  V_i(4,64)     对每个头 i
        |           |           |
        +-----+-----+           |
              |                 |
         softmax(Q_i K_i^T / sqrt(64))  →  attn_i (4, 4)
              |                 |
              +-------+---------+
                      |
                attn_i @ V_i    →  head_i (4, 64)
                      |
              [对8个头重复上述过程]
                      |
        Concat(head_1, ..., head_8)     →  (4, 512)
                      |
                @ W_O (512, 512)
                      |
                Output (4, 512)
```

总计算量看起来很多，但实际上和单头注意力（维度 512）的计算量基本相当——只是把一个大注意力拆成了多个小注意力，每个小注意力的矩阵更小，但总元素数一样。

```python
def multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads=8):
    """
    多头注意力（简化版，权重矩阵已包含所有头）
    W_Q, W_K, W_V: (d_model, d_model)
    W_O: (d_model, d_model)
    """
    n, d_model = X.shape
    d_k = d_model // n_heads

    # 生成 Q, K, V
    Q = X @ W_Q   # (n, d_model)
    K = X @ W_K
    V = X @ W_V

    # 重塑为 (n_heads, n, d_k) 方便并行计算
    Q = Q.reshape(n, n_heads, d_k).transpose(1, 0, 2)   # (8, n, 64)
    K = K.reshape(n, n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(n, n_heads, d_k).transpose(1, 0, 2)

    # 每个头独立计算注意力
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)     # (8, n, n)
    attn = softmax(scores, axis=-1)                        # (8, n, n)
    heads = attn @ V                                       # (8, n, 64)

    # 拼接所有头
    heads = heads.transpose(1, 0, 2).reshape(n, d_model)  # (n, 512)

    # 最终线性投影
    output = heads @ W_O   # (n, 512)
    return output, attn
```

---

## 9.6 因果注意力：为什么要有 Mask

### 自回归生成不能"偷看未来"

GPT 这类模型是**自回归**的：生成第 $t$ 个词时，只能看到第 $1$ 到 $t-1$ 个词，不能偷看第 $t+1$ 个词及之后的"未来"信息。这在训练时也需要遵守——否则模型就学会了作弊，生成质量会很差。

但 Self-Attention 默认是让每个位置都能看到所有位置的。怎么办？加一个 **Mask**。

### Mask：一个下三角矩阵

因果 Mask 是一个下三角矩阵，形状为 $(n, n)$：

$$M = \begin{pmatrix} 1 & 0 & 0 & \cdots & 0 \\ 1 & 1 & 0 & \cdots & 0 \\ 1 & 1 & 1 & & 0 \\ \vdots & & & \ddots & \vdots \\ 1 & 1 & 1 & \cdots & 1 \end{pmatrix}$$

位置 $i$ 只能看到位置 $1$ 到 $i$（含），不能看到 $i+1$ 及之后的位置。Mask 中 1 表示"允许看"，0 表示"遮挡"。

### 应用 Mask 的方式

在计算注意力时，Mask 的应用发生在 softmax **之前**：把 $QK^T / \sqrt{d_k}$ 中那些对应 Mask 为 0 的位置，替换为一个非常大的负数（比如 $-10^9$）。经过 softmax 之后，这些位置的权重就会变成几乎为 0。

$$\text{scores}_{ij} = \begin{cases} \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} & \text{if } M_{ij} = 1 \\ -\infty & \text{if } M_{ij} = 0 \end{cases}$$

从线性代数的角度看，这就是把缩放后的分数矩阵与 Mask 矩阵做一次**逐元素操作**（Hadamard 积，第二章学的），然后把 0 位置替换为 $-\infty$：

$$\text{masked\_scores} = (S \odot M) + (1 - M) \cdot (-10^9)$$

其中 $S = QK^T / \sqrt{d_k}$，$\odot$ 是 Hadamard 积（逐元素乘法）。

### 完整的 Masked Self-Attention

```python
def masked_self_attention(X, W_Q, W_K, W_V, d_k):
    """因果注意力：不允许偷看未来"""
    n = X.shape[0]

    # 生成 Q, K, V
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    # 计算缩放点积分数
    scores = (Q @ K.T) / np.sqrt(d_k)   # (n, n)

    # 构造因果 Mask（下三角矩阵）
    mask = np.tril(np.ones((n, n)))      # 下三角全1
    print("因果 Mask:")
    print(mask)

    # 应用 Mask：把上三角位置设为 -inf
    scores = scores * mask + (1 - mask) * (-1e9)

    # softmax
    attn = softmax(scores)

    print("\nMasked 注意力权重:")
    print(np.round(attn, 3))
    # 上三角全为 0，每个位置只关注自己和之前的位置

    # 加权聚合
    output = attn @ V
    return output
```

看一眼输出就会发现，注意力矩阵的上三角部分全是 0——每个位置都严格地只看"过去"，不会偷看"未来"。第一个位置只能看自己（注意力权重为 1.0），最后一个位置能看到所有人。

---

## 9.7 Python 实战：从零实现 Self-Attention

让我们把前面的所有内容整合在一起，用纯 NumPy 实现一个完整的 Self-Attention 模块，并可视化注意力矩阵。

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def self_attention(X, W_Q, W_K, W_V, mask=None):
    """
    完整的 Self-Attention 实现
    X: (n, d_model) — 输入序列
    W_Q, W_K, W_V: 投影矩阵
    mask: 可选的 (n, n) 因果 mask
    """
    n, d_model = X.shape
    d_k = W_Q.shape[1]

    # Step 1: 生成 Q, K, V
    Q = X @ W_Q   # (n, d_k)
    K = X @ W_K   # (n, d_k)
    V = X @ W_V   # (n, d_v)

    # Step 2: 计算缩放点积分数
    scores = (Q @ K.T) / np.sqrt(d_k)   # (n, n)

    # Step 3: 应用 mask（如果有）
    if mask is not None:
        scores = scores * mask + (1 - mask) * (-1e9)

    # Step 4: softmax 归一化
    attn_weights = softmax(scores)       # (n, n)

    # Step 5: 加权聚合 Value
    output = attn_weights @ V            # (n, d_v)

    return output, attn_weights

def multi_head_attention(X, n_heads, W_Q, W_K, W_V, W_O, mask=None):
    """
    多头注意力实现
    W_Q, W_K, W_V: (d_model, d_model) — 包含所有头的投影矩阵
    W_O: (d_model, d_model) — 输出投影矩阵
    """
    n, d_model = X.shape
    d_k = d_model // n_heads

    Q = X @ W_Q   # (n, d_model)
    K = X @ W_K
    V = X @ W_V

    # 重塑为多头格式: (n, d_model) -> (n_heads, n, d_k)
    Q = Q.reshape(n, n_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(n, n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(n, n_heads, d_k).transpose(1, 0, 2)

    # 各头独立计算注意力
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k)

    if mask is not None:
        scores = scores * mask + (1 - mask) * (-1e9)

    attn_weights = softmax(scores)
    heads = attn_weights @ V   # (n_heads, n, d_k)

    # 拼接并投影
    heads = heads.transpose(1, 0, 2).reshape(n, d_model)
    output = heads @ W_O

    return output, attn_weights


# ---- 演示 ----
np.random.seed(42)

# 参数设置
n = 6            # 序列长度（6个词）
d_model = 64     # 嵌入维度
d_k = d_v = 32   # Q, K, V 的维度

# 模拟输入
X = np.random.randn(n, d_model) * 0.1
W_Q = np.random.randn(d_model, d_k) * 0.01
W_K = np.random.randn(d_model, d_k) * 0.01
W_V = np.random.randn(d_model, d_v) * 0.01

# ===== 单头 Self-Attention =====
output, attn = self_attention(X, W_Q, W_K, W_V)
print("=== 单头 Self-Attention ===")
print(f"输入形状: {X.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力矩阵形状: {attn.shape}")
print(f"注意力矩阵每行之和: {attn.sum(axis=1).round(4)}")

# 可视化注意力矩阵
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 单头注意力（无 mask）
im0 = axes[0].imshow(attn, cmap='Blues', vmin=0, vmax=1)
axes[0].set_title('Single-Head Attention (No Mask)')
axes[0].set_xlabel('Key Position')
axes[0].set_ylabel('Query Position')
plt.colorbar(im0, ax=axes[0])

# ===== 因果 Masked Self-Attention =====
causal_mask = np.tril(np.ones((n, n)))
_, attn_masked = self_attention(X, W_Q, W_K, W_V, mask=causal_mask)
print(f"\n因果 Mask:\n{causal_mask}")
print(f"因果注意力矩阵:\n{np.round(attn_masked, 3)}")

im1 = axes[1].imshow(attn_masked, cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('Masked Attention (Causal)')
axes[1].set_xlabel('Key Position')
axes[1].set_ylabel('Query Position')
plt.colorbar(im1, ax=axes[1])

# ===== 多头注意力 =====
n_heads = 4
d_model_multi = 64
X2 = np.random.randn(n, d_model_multi) * 0.1
W_Q2 = np.random.randn(d_model_multi, d_model_multi) * 0.01
W_K2 = np.random.randn(d_model_multi, d_model_multi) * 0.01
W_V2 = np.random.randn(d_model_multi, d_model_multi) * 0.01
W_O = np.random.randn(d_model_multi, d_model_multi) * 0.01

output_multi, attn_multi = multi_head_attention(
    X2, n_heads, W_Q2, W_K2, W_V2, W_O
)
print(f"\n=== 多头注意力 ({n_heads} heads) ===")
print(f"输入形状: {X2.shape}")
print(f"输出形状: {output_multi.shape}")
print(f"注意力矩阵形状: {attn_multi.shape}  (n_heads, n, n)")

# 可视化多个头的注意力
avg_attn = attn_multi.mean(axis=0)
im2 = axes[2].imshow(avg_attn, cmap='Blues', vmin=0, vmax=1)
axes[2].set_title(f'Multi-Head Attention (avg of {n_heads} heads)')
axes[2].set_xlabel('Key Position')
axes[2].set_ylabel('Query Position')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('/root/gitbook/ch09/attention_visualization.png', dpi=150)
plt.close()
print("\n注意力可视化已保存。")
```

运行这段代码，你会看到三个注意力矩阵的热力图。左边的无 Mask 注意力中，每个位置可以自由地关注所有位置；中间的因果注意力中，上三角被严格遮蔽；右边是多头的平均注意力，展示了一种更丰富的关注模式。

---

## 小结

这一章我们把 Transformer 注意力机制的公式拆到了每一个运算：

| 步骤 | 运算 | 对应的线性代数知识 |
|------|------|-------------------|
| 生成 Q、K、V | $XW_Q, XW_K, XW_V$ | 第三章：线性变换（矩阵乘法） |
| 计算相似度 | $QK^T$ | 第一章：点积 + 矩阵乘法 |
| 缩放 | 除以 $\sqrt{d_k}$ | 概率论：方差归一化 |
| 归一化 | softmax | 把分数变成概率分布 |
| 加权聚合 | 权重 $\times$ $V$ | 矩阵乘法：加权求和 |
| 多头 | 切分、拼接、投影 | 矩阵重塑和线性投影 |
| 因果 Mask | 下三角矩阵 | 第二章：Hadamard 积 |

核心公式从头到尾就是：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

三个矩阵乘法，一次缩放，一次 softmax。就这么简单，却撑起了整个大语言模型的"智能"。线性代数的威力，在这里展现得淋漓尽致。
