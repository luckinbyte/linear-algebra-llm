# 第10章 位置编码——从音乐节拍到RoPE

你有没有想过，为什么一首歌的旋律听起来优美动人，但如果把所有音符随机打乱重新排列，哪怕用的是完全相同的音符，结果也只是一堆噪音？

这是因为音乐的本质不仅仅在于"有哪些音符"，更在于"这些音符以什么顺序出现"。C-E-G 是明亮的大三和弦，G-E-C 是倒过来的分解和弦，而 G-C-E 则又是另一种色彩。同样的三个音，不同的排列，带来完全不同的听觉体验。

语言也是如此。"国王吃了一只猫"和"一只猫吃了国王"，用的是完全相同的五个词，但意思截然相反。词语的**顺序**，或者说**位置**，携带了至关重要的信息。

然而，Transformer 架构最核心的 Self-Attention 机制，天生就**不敏感于顺序**。这意味着如果我们不给它额外的帮助，它分不清"国王吃猫"和"猫吃国王"。这一章，我们就来聊聊 Transformer 如何解决这个问题——从最经典的正弦编码，到当今大模型标配的 RoPE（旋转位置编码），看看线性代数如何帮助模型"记住"词语的位置。

---

## 10.1 为什么需要位置信息

### Self-Attention 的置换不变性

在上一章，我们学习了 Self-Attention 的核心公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

这里有一个隐藏的数学性质：**Self-Attention 是置换不变的（permutation invariant）**。

什么意思？如果你把输入序列的顺序打乱——比如把 $[x_1, x_2, x_3]$ 变成 $[x_3, x_1, x_2]$——Self-Attention 的输出也只是相应地换了一下行，本质上没有变化。打个比方，Self-Attention 就像一个**集合运算**：它知道"这个句子里有哪些词"，也计算了词和词之间的关联，但它不知道"哪个词在前，哪个词在后"。

用更数学化的语言来说：设 $P$ 是一个置换矩阵（permutation matrix），它只重新排列向量的顺序。那么对于 Self-Attention，有：

$$\text{Attention}(PX, PX, PX) = P \cdot \text{Attention}(X, X, X)$$

输出只是跟着输入一起换了个顺序，内部的计算逻辑完全不变。这就是置换不变性。

这对语言模型来说是个大问题。自然语言是高度依赖顺序的。我们需要一种机制，把位置信息"注入"到模型中。

### 位置编码的基本思路

解决方案出奇地简单直接：**给每个位置一个独特的向量，然后把它加到对应位置的词向量上。**

设 $x_i$ 是第 $i$ 个词的嵌入向量，$\mathbf{p}_i$ 是第 $i$ 个位置的"位置向量"，那么模型实际看到的输入是：

$$\hat{x}_i = x_i + \mathbf{p}_i$$

这样一来，即使两个不同位置出现了相同的词，它们的输入向量也不同（因为加上了不同的位置编码）。模型就能区分"出现在句首的'猫'"和"出现在句尾的'猫'"了。

接下来的问题就变成了：**如何设计这些位置向量 $\mathbf{p}_i$？** 这个问题的不同答案，催生了不同的位置编码方案。

---

## 10.2 正弦位置编码

### Transformer 原论文的方案

2017 年的 Transformer 原论文《Attention Is All You Need》提出了第一种位置编码方案。思路非常优雅：用不同频率的正弦和余弦函数来生成位置向量。

对于位置 $pos$ 和维度 $i$，位置编码定义为：

$$PE_{pos, 2i} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{pos, 2i+1} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

其中 $d_{\text{model}}$ 是模型的维度（比如 512），$i$ 从 $0$ 取到 $d_{\text{model}}/2 - 1$。

这意味着位置向量的偶数维用正弦函数，奇数维用余弦函数。不同的维度对应不同的频率——低维度频率高（变化快），高维度频率低（变化慢）。

### 为什么选择正弦函数？

你可能会问：为什么不用线性增长（$1, 2, 3, \ldots$）或者 one-hot 编码？正弦函数有什么特别的？

首先，正弦和余弦函数的值域是 $[-1, 1]$，这意味着位置编码的值和词嵌入的值在同一个数量级上，可以直接相加，不会发生数值上的"淹没"问题。

其次，正弦函数具有**周期性**。周期函数有一个美妙的性质：对于任意固定的频率 $\omega$，存在一个线性变换，可以将 $\sin(\omega(pos + k))$ 表示为 $\sin(\omega \cdot pos)$ 和 $\cos(\omega \cdot pos)$ 的线性组合。具体来说，利用三角恒等式：

$$\sin(\omega(pos + k)) = \sin(\omega \cdot pos)\cos(\omega \cdot k) + \cos(\omega \cdot pos)\sin(\omega \cdot k)$$

$$\cos(\omega(pos + k)) = \cos(\omega \cdot pos)\cos(\omega \cdot k) - \sin(\omega \cdot pos)\sin(\omega \cdot k)$$

用矩阵形式写出来就是：

$$\begin{pmatrix} \sin(\omega(pos+k)) \\ \cos(\omega(pos+k)) \end{pmatrix} = \begin{pmatrix} \cos(\omega k) & \sin(\omega k) \\ -\sin(\omega k) & \cos(\omega k) \end{pmatrix} \begin{pmatrix} \sin(\omega \cdot pos) \\ \cos(\omega \cdot pos) \end{pmatrix}$$

这意味着模型理论上可以通过学习一个线性变换来捕捉相对位置关系。

第三，不同频率的设置让每个位置都有一个"唯一的指纹"。低频率的维度在位置之间变化缓慢，捕捉长程位置信息；高频率的维度变化迅速，精确区分相邻位置。

### 位置编码矩阵的形状

如果序列长度为 $n$，模型维度为 $d_{\text{model}}$，那么位置编码矩阵 $PE$ 的形状是 $(n, d_{\text{model}})$。每一行对应一个位置，每一列对应一个维度。当你用热力图可视化这个矩阵时，会看到一系列从左到右波长逐渐增大的"波浪纹"——这就是不同频率的正弦和余弦函数。

```python
import numpy as np

def sinusoidal_position_encoding(max_len, d_model):
    """生成正弦位置编码矩阵"""
    PE = np.zeros((max_len, d_model))
    position = np.arange(max_len).reshape(-1, 1)  # (max_len, 1)
    
    # 计算频率的分母：10000^(2i/d_model)
    div_term = np.power(10000, np.arange(0, d_model, 2) / d_model)
    
    # 偶数维度用 sin，奇数维度用 cos
    PE[:, 0::2] = np.sin(position / div_term)  # (max_len, d_model/2)
    PE[:, 1::2] = np.cos(position / div_term)  # (max_len, d_model/2)
    
    return PE

# 生成位置编码
pe = sinusoidal_position_encoding(max_len=100, d_model=64)
print(f"位置编码矩阵形状: {pe.shape}")  # (100, 64)
print(f"位置 0 的前 8 维: {np.round(pe[0, :8], 4)}")
print(f"位置 1 的前 8 维: {np.round(pe[1, :8], 4)}")
```

---

## 10.3 学习式位置编码

### BERT 的方案：让模型自己学

Transformer 原论文用数学公式固定了位置编码，但还有另一种思路：**把位置向量当作可训练参数，让模型自己学。**

BERT（Bidirectional Encoder Representations from Transformers）就采用了这种方案。具体做法很简单：

- 设定一个最大序列长度 $n_{\max}$（比如 512）
- 初始化一个位置嵌入矩阵 $P \in \mathbb{R}^{n_{\max} \times d_{\text{model}}}$，和词嵌入矩阵一样，参数是随机初始化的
- 在训练过程中，通过反向传播不断调整这个矩阵的值

从线性代数的角度看，这相当于为每个位置学了一个 $d_{\text{model}}$ 维的向量。没有任何约束（比如周期性、连续性），完全由数据驱动。

### 优缺点对比

| 方面 | 正弦编码（固定） | 学习式编码（可训练） |
|------|-------------------|----------------------|
| 灵活性 | 受数学公式约束 | 完全自由，数据驱动 |
| 外推能力 | 理论上可处理训练时未见过的长度 | 严格限制在训练长度内 |
| 参数量 | 零参数 | 增加参数量 |
| 实际效果 | 在大多数任务上与学习式持平 | 在某些任务上略优 |

关键的一点是外推能力。正弦编码是由公式定义的，理论上你可以计算任意位置的编码——即使训练时只见过长度为 512 的序列，推理时遇到位置 600 也能直接算出来。而学习式编码只有前 $n_{\max}$ 个位置的向量，超出范围就"没学过"了。

不过在实践中，正弦编码的外推优势并没有想象中那么大。当序列长度远超训练长度时，模型的表现依然会下降。这也催生了后来更多针对长度外推的位置编码方案。

---

## 10.4 旋转位置编码（RoPE）：当前最主流的方案

### 核心思想：用旋转编码相对位置

2021 年，苏剑等人在论文《RoFormer》中提出了旋转位置编码（Rotary Position Embedding，RoPE），这是一种精妙的位置编码方案，目前已成为 Llama、Mistral、Qwen 等几乎所有主流大语言模型的标配。

RoPE 的核心思想可以用一句话概括：**不是给位置向量加上什么，而是根据位置去旋转特征向量。**

回忆一下第 2 章学的正交矩阵：旋转矩阵是正交矩阵的一种，它保持向量的长度不变，只改变方向。RoPE 正是利用了这个性质——用旋转角度来编码位置信息。

### 二维情况的直觉

先从最简单的二维情况开始。假设我们有一个二维特征向量 $\mathbf{z} = (z_1, z_2)^T$，位置为 $m$。

RoPE 对它做的操作是：**按照角度 $m\theta$ 进行旋转。**

$$f(\mathbf{z}, m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} z_1 \\ z_2 \end{pmatrix}$$

这不就是我们第 2 章学的旋转矩阵吗？角度为 $m\theta$ 的旋转矩阵作用于向量 $\mathbf{z}$。

现在关键来了。假设有两个位置 $m$ 和 $n$ 上的向量 $\mathbf{q}_m$ 和 $\mathbf{k}_n$（分别是 Query 和 Key），我们想计算它们的点积来衡量注意力。在 RoPE 的变换下：

$$\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle = \text{Re}\left[(q_1 + iq_2)(k_1 - ik_2)e^{i(m-n)\theta}\right]$$

点积的结果只依赖于**相对位置** $(m - n)$！这正是 RoPE 的魔力所在：它通过旋转操作，天然地在注意力分数中编码了相对位置信息。

具体推导如下。设 $\mathbf{q} = (q_1, q_2)^T$，旋转后的 Query 为：

$$\mathbf{q}_m = R(m\theta)\mathbf{q} = \begin{pmatrix} q_1\cos m\theta - q_2\sin m\theta \\ q_1\sin m\theta + q_2\cos m\theta \end{pmatrix}$$

同理，旋转后的 Key 为：

$$\mathbf{k}_n = R(n\theta)\mathbf{k} = \begin{pmatrix} k_1\cos n\theta - k_2\sin n\theta \\ k_1\sin n\theta + k_2\cos n\theta \end{pmatrix}$$

它们的点积为：

$$\mathbf{q}_m \cdot \mathbf{k}_n = (q_1 k_1 + q_2 k_2)\cos((m-n)\theta) + (q_1 k_2 - q_2 k_1)\sin((m-n)\theta)$$

结果只与相对位置 $m - n$ 有关，与绝对位置 $m$ 和 $n$ 分别是多少无关。

### 推广到高维：分块对角旋转矩阵

实际的模型维度远不止二维。对于 $d$ 维的向量（$d$ 是偶数），RoPE 的做法是：

1. 把 $d$ 维向量分成 $d/2$ 个二维块
2. 每个块用不同的频率 $\theta_i$ 进行旋转

旋转矩阵是一个**分块对角矩阵**：

$$R(\Theta, m) = \begin{pmatrix} \cos m\theta_0 & -\sin m\theta_0 & & & \\ \sin m\theta_0 & \cos m\theta_0 & & & \\ & & \cos m\theta_1 & -\sin m\theta_1 & \\ & & \sin m\theta_1 & \cos m\theta_1 & \\ & & & & \ddots \\ & & & & & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\ & & & & & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \end{pmatrix}$$

其中频率 $\theta_i$ 通常设为 $\theta_i = 10000^{-2i/d}$。

这又是一个正交矩阵！回忆第 2 章的知识：分块对角矩阵的每个块都是正交矩阵，所以整个矩阵也是正交矩阵。这意味着：

- **旋转不改变向量的范数（长度）**：$\|R\mathbf{x}\| = \|\mathbf{x}\|$
- 旋转是可逆的，逆变换就是反向旋转

从计算效率的角度看，实际实现中不会真的构造这么大的稀疏矩阵。而是把向量看成复数，用逐元素乘法来等价实现：

$$f(\mathbf{x}, m) = \begin{pmatrix} x_1 \\ x_3 \\ \vdots \\ x_{d-1} \end{pmatrix} \odot \cos(m\boldsymbol{\theta}) - \begin{pmatrix} x_2 \\ x_4 \\ \vdots \\ x_d \end{pmatrix} \odot \sin(m\boldsymbol{\theta}) + i\left(\begin{pmatrix} x_1 \\ x_3 \\ \vdots \\ x_{d-1} \end{pmatrix} \odot \sin(m\boldsymbol{\theta}) + \begin{pmatrix} x_2 \\ x_4 \\ \vdots \\ x_d \end{pmatrix} \odot \cos(m\boldsymbol{\theta})\right)$$

但本质上，这些都是同一个分块对角旋转矩阵的等价实现。

### 为什么 RoPE 有效

RoPE 之所以成为当前主流方案，有几个关键优势：

1. **天然编码相对位置**：如前所述，两个位置的特征向量做内积时，结果自动只依赖于相对位置差，无需模型额外学习这个关系。

2. **旋转保持范数不变**：因为旋转矩阵是正交矩阵，它不改变向量的长度。这意味着位置编码不会扭曲特征空间的几何结构——距离和角度的度量被完整保留。

3. **远程衰减特性**：随着相对距离增大，RoPE 编码后的点积倾向于衰减。这符合语言学的直觉：通常来说，距离较远的词之间的直接关联较弱。

4. **计算高效**：RoPE 的实现只需要逐元素的三角函数运算，不需要额外的参数或复杂的矩阵运算。

### RoPE 在大模型中的应用

从 2023 年开始，几乎所有重要的开源大语言模型都采用了 RoPE：

- **Llama 系列**（Meta）：从 Llama 1 到 Llama 3，都使用 RoPE 作为位置编码
- **Mistral / Mixtral**：RoPE 配合滑动窗口注意力
- **Qwen 系列**（阿里）：RoPE + YaRN 外推策略
- **DeepSeek**：RoPE + 多种长度扩展技术

RoPE 还催生了一系列长度外推技术（如 NTK-aware Scaling、YaRN 等），使得模型可以在更长的序列上推理而无需重新训练。这些技术的核心思想都是调整 RoPE 的频率基数（10000），让旋转角度的变化更加平缓。

---

## 10.5 其他位置编码方案简述

### ALiBi：通过注意力偏置编码位置

ALiBi（Attention with Linear Biases，2021 年）走了一条完全不同的路：它不改词向量，而是在注意力分数上直接加一个与距离相关的偏置。

$$\text{score}(i, j) = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} - m \cdot |i - j|$$

其中 $m$ 是一个固定的斜率（不同的注意力头使用不同的 $m$ 值）。距离越远，惩罚越大。

ALiBi 的优势是极其简单，且外推能力很强——训练时用短序列，推理时可以直接用于长序列。BLOOM、MPT 等模型采用了 ALiBi。

### 相对位置编码

相对位置编码（Relative Position Representation，Shaw et al., 2018）认为，重要的不是"第几个位置"，而是"两个词之间的距离"。它给每个相对距离 $k$ 学习一个向量，在计算注意力时加进去：

$$e_{ij} = \frac{\mathbf{x}_i W_Q ( \mathbf{x}_j W_K + \mathbf{a}_{ij}^K )^T}{\sqrt{d_k}}$$

其中 $\mathbf{a}_{ij}^K$ 是与相对距离 $i - j$ 相关的可学习向量。

### 各方案的线性代数本质对比

| 方案 | 线性代数操作 | 作用位置 |
|------|------------|---------|
| 正弦编码 | 固定向量加法 | 输入层 |
| 学习式编码 | 可训练向量加法 | 输入层 |
| RoPE | 正交旋转（矩阵乘法） | 注意力层 |
| ALiBi | 标量偏置加法 | 注意力分数 |
| 相对位置编码 | 向量加法到 Key | 注意力层 |

从线性代数的角度看，这些方案的演进有一个清晰的趋势：从简单的向量加法（正弦/学习式），到优雅的正交旋转（RoPE），核心思想都是用线性变换来编码位置信息。而 RoPE 之所以脱颖而出，正是因为它用最"干净"的线性代数操作（正交旋转）实现了最理想的效果（自然编码相对位置，且不破坏特征空间结构）。

---

## 10.6 Python 实战

下面我们用 Python 实现正弦位置编码和 RoPE，并做一些简单的可视化分析。

### 实现正弦位置编码

```python
import numpy as np

def sinusoidal_pe(max_len, d_model):
    """
    生成正弦位置编码矩阵
    返回形状: (max_len, d_model)
    """
    PE = np.zeros((max_len, d_model))
    position = np.arange(max_len).reshape(-1, 1)  # (max_len, 1)
    
    # 频率: 1 / 10000^(2i/d_model)
    div_term = np.power(10000.0, np.arange(0, d_model, 2).astype(float) / d_model)
    
    PE[:, 0::2] = np.sin(position / div_term)  # 偶数维: sin
    PE[:, 1::2] = np.cos(position / div_term)  # 奇数维: cos
    
    return PE

# 测试
pe = sinusoidal_pe(max_len=50, d_model=64)
print(f"正弦位置编码形状: {pe.shape}")  # (50, 64)

# 观察不同位置编码的点积
# 如果位置编码只依赖于绝对位置，相邻位置的编码应该更相似
dot_products = pe @ pe.T  # (50, 50)
print(f"位置 0 和位置 1 的相似度: {dot_products[0, 1]:.4f}")
print(f"位置 0 和位置 10 的相似度: {dot_products[0, 10]:.4f}")
print(f"位置 0 和位置 49 的相似度: {dot_products[0, 49]:.4f}")
```

### 实现 RoPE

```python
def apply_rope(x, positions, theta_base=10000.0):
    """
    对输入向量 x 应用旋转位置编码 (RoPE)
    
    参数:
        x: 输入向量, 形状 (..., d)，d 必须是偶数
        positions: 位置索引, 形状 (...,)
        theta_base: 频率基数, 默认 10000
    
    返回:
        旋转后的向量, 形状与 x 相同
    """
    d = x.shape[-1]
    assert d % 2 == 0, "维度必须是偶数"
    
    # 频率: theta_i = 1 / base^(2i/d)
    freqs = 1.0 / np.power(theta_base, np.arange(0, d, 2).astype(float) / d)
    # freqs 形状: (d/2,)
    
    # 计算每个位置的旋转角度: pos * theta_i
    # positions 形状: (...,) -> (..., 1) 以便广播
    angles = positions[..., np.newaxis] * freqs  # (..., d/2)
    
    # 把 x 拆成奇偶两组
    x_even = x[..., 0::2]  # (..., d/2)
    x_odd  = x[..., 1::2]  # (..., d/2)
    
    # 旋转: 二维旋转矩阵的简化实现
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    out_even = x_even * cos_angles - x_odd * sin_angles
    out_odd  = x_even * sin_angles + x_odd * cos_angles
    
    # 交错合并
    out = np.stack([out_even, out_odd], axis=-1)  # (..., d/2, 2)
    out = out.reshape(x.shape)  # (..., d)
    
    return out

# 测试 RoPE
d = 64
q = np.random.randn(d)
k = np.random.randn(d)

# 模拟位置 0 和位置 5 的 Query 和 Key
q_pos0 = apply_rope(q, np.array([0]))
q_pos5 = apply_rope(q, np.array([5]))
k_pos3 = apply_rope(k, np.array([3]))
k_pos8 = apply_rope(k, np.array([8]))

# 计算注意力分数
score_same_rel_1 = np.dot(q_pos0.flatten(), k_pos3.flatten())  # 相对距离 = 3
score_same_rel_2 = np.dot(q_pos5.flatten(), k_pos8.flatten())  # 相对距离 = 3

print(f"q(pos=0) · k(pos=3) = {score_same_rel_1:.4f}  (相对距离=3)")
print(f"q(pos=5) · k(pos=8) = {score_same_rel_2:.4f}  (相对距离=3)")
print(f"两者接近程度: |差值| = {abs(score_same_rel_1 - score_same_rel_2):.6f}")
print("=> 相同相对距离的注意力分数几乎相等，验证了 RoPE 的相对位置编码性质")
```

### 可视化不同位置编码的效果

```python
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 左图：正弦位置编码的热力图 ---
pe = sinusoidal_pe(max_len=50, d_model=64)
im1 = axes[0].imshow(pe.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_xlabel('Position')
axes[0].set_ylabel('Dimension')
axes[0].set_title('Sinusoidal Position Encoding')
plt.colorbar(im1, ax=axes[0])

# --- 右图：RoPE 注意力分数随相对距离的变化 ---
d = 64
np.random.seed(42)
q = np.random.randn(1, d)
k = np.random.randn(1, d)

max_dist = 40
scores = []
for dist in range(max_dist + 1):
    q_rot = apply_rope(q, np.array([0]))
    k_rot = apply_rope(k, np.array([dist]))
    scores.append(np.dot(q_rot.flatten(), k_rot.flatten()))

axes[1].plot(range(max_dist + 1), scores, 'b-o', markersize=3)
axes[1].axhline(y=np.dot(q.flatten(), k.flatten()), color='r', linestyle='--', 
                label='Without RoPE')
axes[1].set_xlabel('Relative Distance')
axes[1].set_ylabel('Dot Product')
axes[1].set_title('RoPE: Attention Score vs Relative Distance')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/gitbook/ch10/position_encoding_viz.png', dpi=150, bbox_inches='tight')
print("可视化图表已保存到 position_encoding_viz.png")
```

左图展示了正弦位置编码矩阵的"波浪纹"结构——你可以清楚地看到不同维度的频率从高（底部）到低（顶部）的变化。右图展示了 RoPE 的远程衰减特性：随着两个位置之间的相对距离增大，注意力分数呈震荡衰减的趋势，这与语言中"近处的词通常关联更强"的直觉一致。

---

## 小结

这一章我们从"为什么 Self-Attention 需要位置信息"出发，走过了位置编码方案的演进之路：

1. **正弦位置编码**（Transformer 原论文）：用不同频率的三角函数为每个位置生成固定向量，简洁优雅，具有一定的外推能力。

2. **学习式位置编码**（BERT）：把位置向量当作可训练参数，灵活但无法处理超长序列。

3. **RoPE**（Llama、Mistral 等）：利用第 2 章学的正交旋转矩阵，通过旋转操作天然编码相对位置，保持了特征空间的几何结构，成为当前大模型的事实标准。

4. **ALiBi 等**：通过在注意力分数上加偏置来编码位置，方案简单，外推能力强。

从线性代数的角度看，位置编码的演进就是从"向量加法"到"矩阵乘法（正交旋转）"的过程——用越来越精巧的线性变换来编码位置信息。而 RoPE 的成功也再次印证了一个道理：好的数学设计（正交旋转天然保持距离和角度）往往比暴力学习更有效。

在下一章，我们将进入 Transformer 的另一个核心组件——层归一化（Layer Normalization），看看它是如何稳定训练过程的，以及它背后的线性代数原理。
