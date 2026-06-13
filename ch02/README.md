# 第2章 矩阵——从Excel到权重矩阵

上一章我们认识了向量——一个有序的数字列表，它是LLM中表示信息的基本单位。但LLM从来不会只处理一个向量。当你输入一句话"今天天气真好"，模型同时处理的是多个token的向量；当注意力机制计算相关性时，它需要对所有位置的向量做批量变换。这种"对一批向量做同样的操作"，就是矩阵的舞台。

如果你已经理解了向量，那么矩阵不过是自然的下一步：**把很多向量排列在一起，就成了矩阵。**

## 2.1 矩阵是什么：从Excel表格到权重矩阵

### 矩阵就是"一张数字表格"

打开Excel，你看到的是什么？一个由行和列组成的网格，每个格子里填着一个数字。这就是矩阵。

用数学语言说，矩阵（matrix）是一个由 $m$ 行 $n$ 列数字排列成的矩形阵列：

$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

其中 $a_{ij}$ 表示第 $i$ 行第 $j$ 列的元素。我们记作 $A \in \mathbb{R}^{m \times n}$，读作"$A$ 是一个 $m$ 乘 $n$ 的实数矩阵"。

你可以把矩阵的每一行看作一个行向量，每一列看作一个列向量。所以一个 $m \times n$ 的矩阵，既可以说是 $m$ 个 $n$ 维行向量的集合，也可以说是 $n$ 个 $m$ 维列向量的集合。这个视角在后面会反复用到。

### 生活中的矩阵

矩阵无处不在，只是我们很少用"矩阵"这个词来称呼它们：

**成绩单。** 30个学生、5门课的成绩表，就是一个 $30 \times 5$ 的矩阵。第 $i$ 行第 $j$ 列的数字，就是第 $i$ 个学生在第 $j$ 门课上的分数。每一行是一个学生的"成绩画像"（5维向量），每一列是一门课的成绩分布。

**航班调度表。** 一个航空公司有200个航班、一周7天的排班表，就是一个 $200 \times 7$ 的矩阵。每个格子填的是执行该航班的飞机编号（或者0表示不排班）。

**像素矩阵。** 一张 $1080 \times 1920$ 的灰度照片，本质上就是一个 $1080 \times 1920$ 的矩阵，每个元素是0到255之间的整数，表示该像素的亮度。如果是彩色照片，那就是三个这样的矩阵叠在一起（RGB三通道），形成一个 $1080 \times 1920 \times 3$ 的**张量**（tensor）——矩阵的三维推广。

### LLM中的矩阵：权重矩阵是AI学到的"知识"

在LLM中，矩阵扮演着至关重要的角色：**模型的"知识"就存储在矩阵里。**

回忆上一章提到的Embedding层：一个词表大小为 $V$、嵌入维度为 $d$ 的Embedding层，就是一个 $V \times d$ 的矩阵 $E$。查某个词的向量，就是取这个矩阵的某一行。

但Embedding只是冰山一角。Transformer中几乎每一个可学习的参数都是以矩阵的形式存在的：

- **Query/Key/Value投影矩阵** $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$：把输入向量投影成查询、键、值向量。
- **输出投影矩阵** $W^O \in \mathbb{R}^{d_v \times d}$：把多头注意力的结果投影回原始维度。
- **前馈网络的权重矩阵** $W_1 \in \mathbb{R}^{d \times 4d}$ 和 $W_2 \in \mathbb{R}^{4d \times d}$：先升维再降维。
- **输出层的语言模型头** $W_{\text{out}} \in \mathbb{R}^{d \times V}$：把隐藏状态映射回词表空间。

一个LLaMA-7B模型大约有70亿个参数。这些参数几乎全部存在于各种大小的矩阵中。训练的过程，就是不断调整这些矩阵中的数字，使得模型在预测下一个词时越来越准。**矩阵中的每一个数字，都是模型从数据中学到的一丁点"知识"。**

## 2.2 矩阵的基本运算

### 加法（类比：两张图片叠加）

如果你有两张同样大小的灰度照片（两个同尺寸矩阵），把它们"叠加"在一起——每个像素的亮度值相加——就得到了矩阵加法。

$$C = A + B, \quad \text{其中} \; c_{ij} = a_{ij} + b_{ij}$$

矩阵加法要求两个矩阵**形状相同**（都是 $m \times n$）。这在LLM中随处可见：残差连接（residual connection）的核心操作就是矩阵加法——把Transformer某一层的输入直接加到输出上：

$$\mathbf{output} = \mathbf{input} + \text{SubLayer}(\mathbf{input})$$

后面会详细说，但直觉很简单：如果子层学到了有用的变换就保留，没学到就退回原值（因为加上了原始输入）。

### 标量乘法（类比：调亮/调暗图片）

用一个数字乘以矩阵，就是把矩阵的每个元素都乘以这个数字。就像调节照片的亮度滑块——乘以1.5就是整体变亮50%，乘以0.5就是整体变暗。

$$c \cdot A = cA, \quad \text{其中} \; (cA)_{ij} = c \cdot a_{ij}$$

在Transformer论文中，Embedding向量会乘以 $\sqrt{d_{\text{model}}}$，这就是一个标量乘法操作。

### 矩阵乘法——最重要的运算

矩阵乘法是这一章最重要的内容，也是理解LLM计算流程的核心。但在讲规则之前，先建立一个直觉。

**类比：工厂流水线的信息加工。** 想象一条工厂流水线。原材料（输入向量）进入流水线，经过一道道工序（矩阵乘法），每一道工序都会对原材料进行某种变换——拉伸、旋转、投影——最终产出成品（输出向量）。每一道工序对应一个矩阵，矩阵的数值决定了变换的方式。

**矩阵乘向量：一条流水线。** 最基本的形式是一个矩阵乘以一个向量。设 $A \in \mathbb{R}^{m \times n}$，$\mathbf{x} \in \mathbb{R}^n$，则：

$$\mathbf{y} = A\mathbf{x} \in \mathbb{R}^m$$

结果 $\mathbf{y}$ 的第 $i$ 个分量是 $A$ 的第 $i$ 行与 $\mathbf{x}$ 的点积：

$$y_i = \sum_{j=1}^{n} a_{ij} x_j = a_{i1}x_1 + a_{i2}x_2 + \cdots + a_{in}x_n$$

换句话说，$A$ 的每一行定义了一个"检测器"，它通过点积来衡量输入向量 $\mathbf{x}$ 与这个检测器的匹配程度。$m$ 行就产生 $m$ 个输出值——输入被从 $n$ 维空间映射到了 $m$ 维空间。

在LLM中，这就是一个线性层（linear layer / fully connected layer）做的事。比如，前馈网络的第一步 $\mathbf{h} = \mathbf{x}W_1$，就是把一个 $d$ 维输入向量通过 $d \times 4d$ 的矩阵 $W_1$ 映射为 $4d$ 维输出向量。

**两种更深的读法：列的线性组合 / 变换的复合。** 把 $\mathbf{y}=A\mathbf{x}$ 的第 $i$ 行展开 $y_i=\sum_j a_{ij}x_j$，你会自然得到一个等价但更深刻的写法——**$A\mathbf{x}$ 是 $A$ 各列的线性组合**：

$$A\mathbf{x} = x_1\begin{bmatrix}a_{11}\\ \vdots \\ a_{m1}\end{bmatrix} + x_2\begin{bmatrix}a_{12}\\ \vdots \\ a_{m2}\end{bmatrix} + \cdots + x_n\begin{bmatrix}a_{1n}\\ \vdots \\ a_{mn}\end{bmatrix}$$

$\mathbf{x}$ 的每个分量 $x_j$，就是"第 $j$ 列取多少份"。这个视角带来一个关键推论：**$A\mathbf{x}$ 的所有可能输出，永远落在 $A$ 的列向量所能张成的空间里**（这个空间叫**列空间**）。后面讲秩时，这会是理解"信息能活下来几维"的钥匙。

更一般地，矩阵乘矩阵 $AB$ 也有两层读法：既是"行×列的点积"，也等于"$A$ 依次作用在 $B$ 的每一列上"——也就是**先做 $B$ 再做 $A$ 的复合变换**。多层 $\mathbf{y}=A_2(A_1\mathbf{x})$ 能合并成 $(A_2A_1)\mathbf{x}$，正是这个"复合"性质。

**一个 2 维小例（同一乘法、两种读法）：** $A=\begin{bmatrix}1&2\\3&4\end{bmatrix}$，$\mathbf{x}=[1,1]$。

- 行视角（点积）：$A\mathbf{x} = [1\cdot1+2\cdot1,\; 3\cdot1+4\cdot1] = [3,7]$
- 列视角（线性组合）：$A\mathbf{x} = 1\cdot\begin{bmatrix}1\\3\end{bmatrix} + 1\cdot\begin{bmatrix}2\\4\end{bmatrix} = [3,7]$

两种读法结果完全一致——区别在于"列视角"让你看到输出 $[3,7]$ 是怎么由 $A$ 的两列拼出来的。

```python
import numpy as np
A = np.array([[1., 2.], [3., 4.]]); x = np.array([1., 1.])
print("行视角 A@x =", A @ x)                 # [3. 7.]
print("列视角    =", 1*A[:,0] + 1*A[:,1])    # [3. 7.]
```

> **一句话锚点：** $A\mathbf{x}$ 既是"每行与 $\mathbf{x}$ 的点积"，也是"$A$ 各列的线性组合"——后者揭示了输出总落在列空间里。

**矩阵乘矩阵：批量流水线。** 更一般地，设 $A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$，则乘积 $C = AB \in \mathbb{R}^{m \times n}$：

$$c_{ij} = \sum_{l=1}^{k} a_{il} b_{lj}$$

直觉：$C$ 的第 $(i,j)$ 个元素，是 $A$ 的第 $i$ 行和 $B$ 的第 $j$ 列做点积的结果。

**关键约束**：$A$ 的列数必须等于 $B$ 的行数（即内维度必须匹配）。如果 $A$ 是 $m \times k$，$B$ 是 $k \times n$，中间的 $k$ 被"消耗"掉了，结果是 $m \times n$。记忆口诀：$(m \times k) \times (k \times n) = m \times n$。

**批量视角：** 另一个有用的视角是，$B$ 的每一列都是一个输入向量，矩阵乘法 $AB$ 相当于用 $A$ 对 $B$ 的所有列同时做变换。这就是为什么矩阵乘法能实现"批量计算"——GPU之所以擅长矩阵乘法，正是因为它可以并行地对所有列做相同的操作。

在Transformer中，注意力分数的计算就是一个典型的矩阵乘法：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

这里 $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{n \times d_k}$，所以 $QK^T \in \mathbb{R}^{n \times n}$。这个结果矩阵的第 $(i,j)$ 个元素就是第 $i$ 个位置的query和第 $j$ 个位置的key的点积——一次矩阵乘法就计算出了所有位置对之间的相关性。

### 结合律与分配律：为什么多层可以合并计算

矩阵乘法满足两个重要性质：

**结合律：** $(AB)C = A(BC)$。这意味着多个矩阵相乘时，计算顺序不影响结果。在实践中，如果多个线性层之间没有非线性激活函数，它们可以合并为一个矩阵乘法：

$$\mathbf{y} = (A_2 A_1)\mathbf{x} = A_2(A_1 \mathbf{x})$$

这也是为什么Transformer的前馈网络在两层线性变换之间必须有ReLU或SiLU等非线性激活——没有非线性，两层就退化成一层，模型就失去了表达力。

**分配律：** $A(B + C) = AB + AC$ 以及 $(A + B)C = AC + BC$。

这两个性质在模型优化和量化中非常重要。例如，当你想把一个 $d \times 4d$ 的矩阵拆分成两个小矩阵来节省内存时，分配律保证了这种拆分是合法的。

### 转置（类比：把Excel表格行列互换）

矩阵的转置（transpose），就是把行和列互换。就像在Excel中选中整个表格，复制，然后"选择性粘贴-转置"——原来的行变成了列，原来的列变成了行。

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \quad A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

如果 $A \in \mathbb{R}^{m \times n}$，则 $A^T \in \mathbb{R}^{n \times m}$，且 $(A^T)_{ij} = a_{ji}$。

转置的一个重要性质是**乘积的转置 reversal 法则**：

$$(AB)^T = B^T A^T$$

注意顺序反过来了。这在注意力计算中直接体现：$QK^T$ 的转置是 $(QK^T)^T = (K^T)^T Q^T = KQ^T$，交换了Query和Key的角色。

在LLM中，转置操作频繁出现。最典型的就是注意力机制中的 $QK^T$——Query矩阵和Key矩阵的转置相乘，一行代码就能算出所有位置对的相似度。

### Hadamard积/逐元素乘法（类比：给照片加滤镜蒙版）

Hadamard积（Hadamard product），也叫逐元素乘法（element-wise multiplication），是两个同形状矩阵对应位置元素直接相乘：

$$C = A \odot B, \quad c_{ij} = a_{ij} \cdot b_{ij}$$

**类比：** 在Photoshop中，给一张照片加一个黑白蒙版（mask）。蒙版白色的区域，照片原样保留（乘以1）；蒙版黑色的区域，照片变成全黑（乘以0）；灰色区域则半透明（乘以0~1之间的值）。这就是Hadamard积的效果。

在LLM中，Hadamard积最经典的应用是**注意力掩码（Attention Mask）**。在自回归语言模型中，当前位置不能看到未来的信息——这就需要一个因果掩码（causal mask）。这个掩码是一个下三角矩阵：

$$M = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 1 \end{bmatrix}$$

在注意力计算中，先用掩码把不该看到的注意力权重清零（通过减去一个很大的负数，使softmax后接近0），等价于对注意力权重矩阵做Hadamard积。具体实现通常是这样的：

```python
# causal mask: 将上三角设为 -inf，softmax 后变为 0
mask = torch.triu(torch.ones(n, n), diagonal=1) * float('-inf')
attn_weights = torch.softmax(Q @ K.T / math.sqrt(d_k) + mask, dim=-1)
```

## 2.3 特殊矩阵家族

矩阵中有一些"特殊公民"，它们具有独特的结构，在LLM中以各种形式出现。认识它们，就像认识工具箱里的各种专用工具。

### 单位矩阵 I："什么都不做"的矩阵

单位矩阵（identity matrix）$I_n$ 是一个 $n \times n$ 的矩阵，对角线上全是1，其余全是0：

$$I_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

它的核心性质是：对于任何矩阵 $A$，都有 $AI = IA = A$。乘以单位矩阵就像乘以数字1——什么都不变。

**残差连接的直觉。** 在Transformer中，残差连接（residual connection）做的事情就是 $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$。可以把它理解为：把操作"默认设为什么都不做"（即乘以 $I$），然后在这个基础上叠加学到的变换 $F(\mathbf{x})$。如果 $F$ 学到的是全零（等于 $I$ 的"什么都不做"），输出就等于输入。这保证了即使网络很深，信息至少能原封不动地传递过去，不会丢失。

这个设计来自He等人2016年的ResNet论文。原本是给图像网络用的，但Transformer全盘继承了这一设计。事实证明，没有残差连接，深度Transformer根本无法训练——梯度在几十层矩阵乘法中会迅速消失。

### 对角矩阵：只管自己那行

对角矩阵（diagonal matrix）是只有对角线上有非零元素的方阵：

$$D = \begin{bmatrix} d_1 & 0 & 0 \\ 0 & d_2 & 0 \\ 0 & 0 & d_3 \end{bmatrix}$$

对角矩阵乘以向量，就是对向量的每个分量独立缩放：$D\mathbf{x} = [d_1 x_1, d_2 x_2, d_3 x_3]^T$。每个维度只关心自己，互不干扰。

在LLM中，对角矩阵的一个直接应用是**门控机制**（gating）。比如在GLU（Gated Linear Unit）变体中，一个分支产生"门控信号"（经过sigmoid，范围在0~1之间），对另一个分支做逐元素乘法——这等价于用一个对角矩阵（对角线元素就是门控信号）来筛选信息。

另外，Transformer中广泛使用的RMSNorm也可以用对角矩阵来理解：计算每个向量各分量的均方根，然后逐分量缩放——本质上就是对角矩阵乘以向量。

### 对称矩阵：$A = A^T$

对称矩阵（symmetric matrix）满足 $A = A^T$，即 $a_{ij} = a_{ji}$。它关于主对角线"镜面对称"：

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 5 & 6 \\ 3 & 6 & 9 \end{bmatrix}$$

对称矩阵在LLM中最自然的体现是**注意力矩阵**。在双向注意力（如BERT使用的）中，位置 $i$ 对位置 $j$ 的注意力分数和位置 $j$ 对位置 $i$ 的注意力分数虽然不完全相同（因为经过了softmax），但在原始的 $QK^T$ 阶段，如果 $Q = K$，那么 $QK^T$ 就是对称的。即使 $Q \ne K$，注意力矩阵也具有某种"近似对称"的结构。

对称矩阵有一个优美的数学性质：它的所有特征值都是实数，且可以被正交对角化。这个性质在理解注意力机制的信息传递时非常有用——我们在后续章节讲特征分解时还会回来。

### 正交矩阵：长度不变的旋转

正交矩阵（orthogonal matrix）满足 $Q^T Q = I$，即它的转置就是它的逆。它的核心性质是：**向量经过正交矩阵变换后，长度不变**——只可能旋转或翻转，不会拉伸或压缩。

$$\|Q\mathbf{x}\| = \|\mathbf{x}\|, \quad \forall \mathbf{x}$$

**RoPE预告。** 在LLaMA等现代模型中使用的旋转位置编码（Rotary Position Embedding, RoPE），本质上就是把位置信息编码为一个正交矩阵变换——通过旋转向量来注入位置信息。因为旋转不改变向量的长度，所以不会干扰向量本身所承载的语义信息，只改变其"方向"来编码位置。我们会在位置编码章节详细展开这个优美的设计。

正交矩阵在模型训练中还有一个重要应用：**正交初始化**。如果权重矩阵被初始化为近似的正交矩阵，训练初期梯度会流动得更均匀（不会在某些方向上爆炸或消失），这对于深层网络的训练至关重要。

## 2.4 逆矩阵与行列式

### 逆矩阵：操作的"撤销"按钮

如果说矩阵 $A$ 是一个变换操作（把 $\mathbf{x}$ 变成 $A\mathbf{x}$），那么逆矩阵 $A^{-1}$ 就是这个操作的"撤销"按钮：

$$A^{-1}(A\mathbf{x}) = \mathbf{x}$$

即 $A^{-1}A = I$。

不是所有矩阵都可逆。只有**方阵**（行数等于列数），且行列式不为零，才存在逆矩阵。

在LLM的语境中，我们几乎不会显式计算逆矩阵。为什么？因为LLM的权重矩阵从来不需要"撤销"——训练是通过梯度下降来调整参数的，不是通过求逆来求解方程组的。但逆矩阵的概念对于理解"信息是否可恢复"非常重要。

### 行列式：变换的"体积缩放因子"

行列式（determinant）是一个把方阵映射为一个标量的函数，记作 $\det(A)$ 或 $|A|$。

对于一个 $2 \times 2$ 矩阵：

$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

几何直觉：如果把矩阵的每一行（或每一列）看作空间中的一个向量，那么行列式的绝对值就是这些向量所张成的平行多面体的**体积**。对于 $2 \times 2$ 矩阵，就是两个行向量围成的平行四边形面积。

如果 $\det(A) = 2$，意味着这个变换把空间放大了2倍。如果 $\det(A) = -1$，意味着变换翻转了方向但保持了体积。

**为什么 $ad-bc$ 恰好是面积？** 把单位正方形（四个顶点 $[0,0],[1,0],[0,1],[1,1]$）经过矩阵 $A=\begin{bmatrix}a&b\\c&d\end{bmatrix}$ 变换。$[1,0]$ 变成 $A$ 的第一列 $\mathbf{c}_1=[a,c]$，$[0,1]$ 变成第二列 $\mathbf{c}_2=[b,d]$——单位正方形被"揉"成了由 $\mathbf{c}_1,\mathbf{c}_2$ 张成的平行四边形。这个平行四边形的面积，正是 $|\det A|=|ad-bc|$。而 $A$ 的**符号**记录了朝向：正号表示保持原来的手性（朝向不变），负号表示被镜像翻转了一次。

**搭桥（用具体数字验证）：** 取 $A=\begin{bmatrix}2&1\\1&3\end{bmatrix}$，两列为 $\mathbf{c}_1=[2,1]$、$\mathbf{c}_2=[1,3]$。

- 行列式：$\det A = 2\times3 - 1\times1 = 5$
- 用底×高独立算面积：$\|\mathbf{c}_1\|=\sqrt5$，$\|\mathbf{c}_2\|=\sqrt{10}$，$\cos\theta=\dfrac{\mathbf{c}_1\cdot\mathbf{c}_2}{\|\mathbf{c}_1\|\|\mathbf{c}_2\|}=\dfrac{5}{\sqrt{50}}\approx0.707$，故 $\sin\theta\approx0.707$，面积 $=\|\mathbf{c}_1\|\|\mathbf{c}_2\|\sin\theta=\sqrt5\cdot\sqrt{10}\cdot0.707=5$。

两条路殊途同归，都得到 5——这就是"行列式 = 面积缩放"的可信来源：变换把面积为 1 的单位正方形，放大成了面积为 5 的平行四边形。

**行列式为零 = 被压扁：** 再看 $B=\begin{bmatrix}1&2\\2&4\end{bmatrix}$，第二列 $\mathbf{c}_2=[2,4]$ 恰好是 $\mathbf{c}_1=[1,2]$ 的 2 倍——两列共线，平行四边形退化成一条线段，面积 $=0=\det B$。空间被压扁了，这一维的信息彻底丢失（这正是下一节"不可逆/信息丢失"的几何根源）。

```python
import numpy as np
A = np.array([[2., 1.], [1., 3.]]); B = np.array([[1., 2.], [2., 4.]])
print("det A =", np.linalg.det(A))   # 5.0
print("det B =", np.linalg.det(B))   # 0.0  (列共线 -> 压扁)
```

> **一句话锚点：** 行列式 = 变换对空间的体积（面积）缩放倍数；为 0 意味着空间被压扁、信息丢失。

### 行列式为零 = 不可逆 = 信息丢失

行列式最重要的性质是：

$$\det(A) = 0 \iff A \text{ 不可逆}$$

直觉：如果行列式为零，说明矩阵把空间"压缩"了——某个维度被压扁了。信息被丢掉了，自然无法恢复。就像把一张3D的照片压成2D——你不可能从2D图完美还原出3D信息。

在LLM中，虽然我们不直接计算行列式，但"信息压缩"的概念无处不在。比如，前馈网络的第一层把 $d$ 维向量映射到 $4d$ 维（信息"展开"），第二层又把 $4d$ 维映射回 $d$ 维（信息"压缩"）。如果中间没有非线性激活，这两步就等价于一个 $d \times d$ 矩阵——而且是一个秩可能很低的矩阵，意味着信息经过"展开再压缩"后大量丢失了。非线性激活函数的存在就是为了打破这种线性退化。

### 为什么这些在LLM中几乎不直接出现

你可能会问：既然逆矩阵和行列式这么重要，为什么在LLM的代码里几乎看不到它们？原因有三：

1. **LLM是"前向"的。** 模型只需要做 $A\mathbf{x}$（矩阵乘向量），从不需要做 $A^{-1}\mathbf{y}$（求解方程组）。训练中用的反向传播虽然名字里有"逆"，但它是通过链式法则计算梯度，不是求逆矩阵。

2. **LLM的矩阵通常是矩形的。** 比如 $W_1 \in \mathbb{R}^{d \times 4d}$，不是方阵，根本没有行列式和逆。

3. **计算代价太高。** 求逆矩阵的复杂度是 $O(n^3)$，对于 $4096 \times 4096$ 的矩阵来说，每层都求逆是不可接受的。

但理解这些概念仍然重要——它们帮助你理解"为什么需要非线性"、"为什么残差连接能防止信息丢失"、"为什么某些初始化方式更好"。

## 2.5 范数：衡量向量和矩阵的"大小"

### L1范数和L2范数

范数（norm）是衡量向量"大小"的数学工具。最常用的两个：

**L1范数**（曼哈顿距离）：所有分量的绝对值之和。

$$\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|$$

名字来自曼哈顿的街道——出租车从A到B的距离是各方向路程的绝对值之和（因为不能走对角线）。

**L2范数**（欧氏距离）：所有分量平方和的平方根。

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$$

这就是我们通常说的"向量长度"。点积公式中的 $\|\mathbf{a}\|$ 用的就是L2范数。

### 矩阵范数

矩阵也有范数。最常用的两个：

**Frobenius范数**：所有元素的平方和的平方根。可以理解为把矩阵"拉平"成一个长向量后取L2范数。

$$\|A\|_F = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n} a_{ij}^2}$$

在LLM的模型量化中，Frobenius范数用来衡量量化误差——量化后的权重矩阵与原始矩阵的Frobenius范数差，直接反映了信息损失的程度。

**谱范数**（2-范数）：矩阵的最大奇异值。它衡量的是矩阵对向量的"最大拉伸能力"：

$$\|A\|_2 = \max_{\mathbf{x} \ne 0} \frac{\|A\mathbf{x}\|}{\|\mathbf{x}\|}$$

如果谱范数大于1，某些方向的信号会在多层传播中被不断放大；如果小于1，信号会不断衰减。这直接关联到训练稳定性——梯度爆炸或消失，本质上就是矩阵的谱范数失控。

### LayerNorm/RMSNorm为什么叫"范数"

Transformer中广泛使用的层归一化（LayerNorm）和均方根归一化（RMSNorm），名称中都带有"Norm"，这绝非巧合。

**RMSNorm** 的计算过程：

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \mathbf{g}, \quad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$$

其中 $\mathbf{g}$ 是可学习的缩放参数。注意RMS的计算：先求所有分量平方的平均，再开根号——这其实就是L2范数除以 $\sqrt{d}$。

所以RMSNorm的本质是：先计算向量的（归一化版本的）L2范数，然后用它来缩放向量，使得向量的"大小"变成一个标准值，再通过可学习参数 $\mathbf{g}$ 恢复表达能力。

**为什么要这样做？** 在深层网络中，经过多层矩阵乘法，向量的各分量可能会变得很大或很小（因为每乘一次矩阵，数值范围都会变化）。如果不加控制，数值会指数级增长或衰减。归一化就像是给信号加了一个"音量自动调节器"——不管输入多大多小，输出都维持在一个稳定的范围内。

## 2.6 从矩阵到LLM：一次完整的"一层"长什么样

现在让我们把这一章学到的所有概念串起来，用Python代码实现一个简化版的Transformer层，并标注每一个矩阵运算。

```python
import numpy as np

# ============================================================
# 简化版 Transformer 层 —— 矩阵运算全景图
# ============================================================

# 超参数
d_model = 512    # 模型维度
n_heads = 8      # 注意力头数
d_k = d_model // n_heads  # 每个头的维度 = 64
d_ff = 4 * d_model        # FFN 中间维度 = 2048
seq_len = 10     # 序列长度

np.random.seed(42)

# ----- 可学习参数（矩阵）-----
# 自注意力的四个投影矩阵 [2.1节: 权重矩阵]
W_Q = np.random.randn(d_model, d_model) * 0.02  # Query 投影
W_K = np.random.randn(d_model, d_model) * 0.02  # Key 投影
W_V = np.random.randn(d_model, d_model) * 0.02  # Value 投影
W_O = np.random.randn(d_model, d_model) * 0.02  # 输出投影

# 前馈网络的两个权重矩阵
W_1 = np.random.randn(d_model, d_ff) * 0.02     # 升维: 512 -> 2048
W_2 = np.random.randn(d_ff, d_model) * 0.02     # 降维: 2048 -> 512

# LayerNorm 的缩放参数（对角矩阵的等价物 [2.3节]）
gamma_attn = np.ones(d_model)    # 注意力后的 LN 缩放
gamma_ffn = np.ones(d_model)     # FFN 后的 LN 缩放

# ----- 输入：seq_len 个 d_model 维向量组成的矩阵 -----
X = np.random.randn(seq_len, d_model)  # [seq_len, d_model]

print(f"输入矩阵 X 的形状: {X.shape}")
print(f"等于 {seq_len} 个 {d_model} 维向量\n")

# ============================================================
# 第一步：自注意力 (Self-Attention)
# ============================================================

# --- 矩阵乘法 [2.2节]: 输入投影为 Q, K, V ---
# X @ W_Q: [seq_len, d_model] × [d_model, d_model] -> [seq_len, d_model]
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print(f"Q, K, V 的形状: {Q.shape}")  # [10, 512]

# --- 转置 [2.2节] + 矩阵乘法: 计算注意力分数 ---
# Q @ K^T: [seq_len, d_model] × [d_model, seq_len] -> [seq_len, seq_len]
attn_scores = Q @ K.T / np.sqrt(d_k)

print(f"注意力分数矩阵的形状: {attn_scores.shape}")  # [10, 10]
print(f"(i,j) 元素 = 位置 i 的 query 与位置 j 的 key 的点积\n")

# --- Hadamard 积 [2.2节]: 因果掩码（自回归模型不能看未来） ---
causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
attn_scores = attn_scores + causal_mask  # 等价于 mask 后做 Hadamard 积

# Softmax（按行归一化，使每行和为1）
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

attn_weights = softmax(attn_scores)
print(f"注意力权重矩阵: 对称矩阵 [2.3节]? 近似，但 softmax 打破了严格对称性")
print(f"对角线附近权重较大 (就近关注)\n")

# --- 矩阵乘法: 加权求和 value ---
# attn_weights @ V: [seq_len, seq_len] × [seq_len, d_model] -> [seq_len, d_model]
attn_output = attn_weights @ V

# --- 矩阵乘法: 输出投影 ---
# attn_output @ W_O: [seq_len, d_model] × [d_model, d_model] -> [seq_len, d_model]
attn_output = attn_output @ W_O

# ============================================================
# 第二步：残差连接 + LayerNorm
# ============================================================

# --- 矩阵加法 [2.2节]: 残差连接 ---
# 单位矩阵 I 的直觉 [2.3节]: 如果子层输出为 0, 信息原封不动传递
residual = X + attn_output
print(f"残差连接: output = X + Attention(X), 形状 {residual.shape}")

# --- 范数 [2.5节]: RMSNorm ---
rms = np.sqrt(np.mean(residual ** 2, axis=-1, keepdims=True))
normed = residual / rms
attn_out = gamma_attn * normed  # 可学习缩放（对角矩阵操作）

# ============================================================
# 第三步：前馈网络 (FFN) + 残差 + LayerNorm
# ============================================================

# --- 矩阵乘法 [2.2节]: 升维 ---
# attn_out @ W_1: [seq_len, d_model] × [d_model, d_ff] -> [seq_len, d_ff]
hidden = attn_out @ W_1
print(f"FFN 升维: {d_model} -> {d_ff}")

# 非线性激活 (SiLU, LLaMA 风格)
# 没有 非线性激活，两层线性变换等价于一层 [2.2节 结合律]
hidden = hidden * (1 / (1 + np.exp(-hidden)))  # SiLU = x * sigmoid(x)

# --- 矩阵乘法: 降维 ---
# hidden @ W_2: [seq_len, d_ff] × [d_ff, d_model] -> [seq_len, d_model]
ffn_output = hidden @ W_2
print(f"FFN 降维: {d_ff} -> {d_model}")

# 残差连接 + LayerNorm
residual2 = attn_out + ffn_output
rms2 = np.sqrt(np.mean(residual2 ** 2, axis=-1, keepdims=True))
output = gamma_ffn * (residual2 / rms2)

print(f"\n=== 一层 Transformer 的矩阵运算总结 ===")
print(f"矩阵乘法: 8 次")
print(f"  1. Q = X @ W_Q    (Query投影)")
print(f"  2. K = X @ W_K    (Key投影)")
print(f"  3. V = X @ W_V    (Value投影)")
print(f"  4. scores = Q @ K^T (注意力分数)")
print(f"  5. context = attn_weights @ V (加权取值)")
print(f"  6. output = context @ W_O (输出投影)")
print(f"  7. hidden = attn_out @ W_1 (FFN第一层)")
print(f"  8. result = hidden @ W_2    (FFN第二层)")
print(f"矩阵加法: 2 次 (两次残差连接)")
print(f"Hadamard 积: 1 次 (因果掩码, 等) + 激活函数")
print(f"范数运算: 2 次 (两次 RMSNorm)")
print(f"转置: 1 次 (K^T)")
print(f"\n最终输出形状: {output.shape}")
```

运行这段代码，你会看到一整层Transformer中的矩阵运算全貌。让我们总结一下这一层里出现的所有本章概念：

| 概念 | 出现位置 | 对应章节 |
|------|----------|----------|
| 矩阵乘法 | $Q=XW_Q$, $K=XW_K$, $V=XW_V$, $QK^T$, $\text{attn} \cdot V$, $W_O$, $W_1$, $W_2$ 共8处 | 2.2 |
| 矩阵加法 | 残差连接 $X + \text{SubLayer}(X)$ 共2处 | 2.2 |
| 转置 $K^T$ | 注意力分数 $QK^T$ | 2.2 |
| Hadamard积 | 因果掩码、SiLU激活 | 2.2 |
| 对角矩阵 | LayerNorm的缩放参数 $\gamma$ | 2.3 |
| 对称矩阵 | 注意力权重矩阵（近似对称） | 2.3 |
| 范数 | RMSNorm中的RMS计算 | 2.5 |
| 结合律 | 多个矩阵乘法可合并计算（实际不合并，因中间有非线性） | 2.2 |

这就是Transformer一层中的全部矩阵运算。一个LLaMA-7B模型有32层这样的结构，每层都在重复类似的操作。如果把每一层的矩阵乘法次数加起来（8次 $\times$ 32层 = 256次矩阵乘法），再加上Embedding和输出层，一次前向传播（生成一个token）大约需要260次矩阵乘法。

而每一次矩阵乘法，GPU都在并行执行数十亿次浮点运算。这就是为什么LLM需要如此强大的算力——不是因为单个操作有多复杂，而是因为操作的数量实在太多。

---

从Excel表格到注意力分数矩阵，从成绩单到权重矩阵，矩阵的核心思想同样简洁：**把很多向量排列在一起，用统一的规则进行批量运算。** 当你下次听到"矩阵乘法"时，不妨把它想象成一条工厂流水线——原材料（输入向量）经过一道道矩阵变换的加工，最终变成成品（输出向量）。而流水线上每一道工序的参数（矩阵中的数字），都是模型从海量文本中学到的。

下一章我们将继续深入，看看当矩阵有了"方向"和"基底"——线性空间与线性变换——向量在变换后会发生什么，以及特征值和特征向量如何帮助我们理解Transformer的信息流。
