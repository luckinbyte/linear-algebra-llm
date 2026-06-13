# 第3章 线性变换——从照片滤镜到神经网络层

你打开手机修图软件，随手拉伸一张照片、旋转一个角度、或者套上一个素描滤镜。这些操作的底层，其实都在做同一件事：把图像中的每一个像素点，从原来的位置"搬"到新的位置。而负责描述这种搬运规则的数学工具，叫做**线性变换**。

别被"线性"两个字吓到。你十年前学过矩阵乘法，那个操作本身就是线性变换。今天我们要做的事情，就是把你记忆中那个干巴巴的"矩阵乘向量"，还原成一幅幅生动的几何画面，然后直接连接到 Transformer 里每一个"线性投影"操作。

---

## 3.1 什么是线性变换

### 从 Photoshop 的拉伸、旋转、倾斜说起

想象你在 Photoshop 里有一张正方形的图片。你做了三个操作：

1. **水平拉伸两倍**——正方形变成长方形
2. **旋转 45 度**——长方形歪了
3. **倾斜（剪切）**——变成平行四边形

这三个操作有一个共同特点：**图片上原来的直线，变换之后还是直线；原来平行的线，变换之后还是平行的。** 更关键的是，**原点不动**——图片的中心（如果你把中心设为原点）始终在原地。

这就是线性变换的全部几何直觉：

> **线性变换 = 保持直线仍是直线、原点不动的空间变形。**

### 两个数学性质

用数学语言精确描述，线性变换 $T$ 满足两条性质，对任意向量 $\mathbf{u}, \mathbf{v}$ 和标量 $c$：

1. **可加性**：$T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. **齐次性**：$T(c\mathbf{u}) = cT(\mathbf{u})$

两条合在一起就是**叠加原理**：先加再变换 = 先变换再加，先乘再变换 = 先变换再乘。这意味着线性变换是"可预测的"——你只需要知道基向量变换到了哪里，就能推断出空间中任何一个向量的去向。

### 矩阵就是线性变换的"配方"

这是整章最重要的一个认知飞跃：

> **一个矩阵 $A$ 就是一个线性变换的完整描述。矩阵的每一列，就是原来的基向量变换后的位置。**

在二维空间里，原来的基向量是 $\mathbf{e}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ 和 $\mathbf{e}_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$。如果某个线性变换把 $\mathbf{e}_1$ 变到了 $\begin{pmatrix} 2 \\ 1 \end{pmatrix}$，把 $\mathbf{e}_2$ 变到了 $\begin{pmatrix} -1 \\ 3 \end{pmatrix}$，那么这个变换的矩阵就是：

$$A = \begin{pmatrix} 2 & -1 \\ 1 & 3 \end{pmatrix}$$

左列是 $\mathbf{e}_1$ 变换后的位置，右列是 $\mathbf{e}_2$ 变换后的位置。就这么简单。当你做矩阵乘法 $A\mathbf{x}$ 时，本质上就是在问："空间经过 $A$ 的变形后，原来在 $\mathbf{x}$ 位置的点，现在跑到了哪里？"

---

## 3.2 几何直觉：拉伸、旋转、投影、剪切

理解了"矩阵 = 变换配方"之后，我们来看几种经典变换，建立几何直觉。

### 拉伸：对角矩阵

最简单的矩阵是对角矩阵：

$$D = \begin{pmatrix} a & 0 \\ 0 & d \end{pmatrix}$$

它做的事情非常直白：$x$ 轴方向拉伸 $a$ 倍，$y$ 轴方向拉伸 $d$ 倍。如果 $a = 2, d = 0.5$，那就是横向拉宽、纵向压扁。如果某个对角元素是负数，比如 $a = -1$，那就是沿该轴翻转。

### 旋转：正交矩阵

旋转矩阵长这样（逆时针旋转 $\theta$ 角）：

$$R = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

你可以验证：$R$ 的两个列向量长度都是 1，而且互相垂直——这就是**正交矩阵**的特点。正交矩阵做变换时，不改变向量的长度，只改变方向。它是一种"刚体旋转"，不会拉伸或压缩空间。

在 Transformer 的多头注意力中，某些位置编码方案（如 RoPE）就是用旋转矩阵来编码位置信息——在二维子空间里做旋转变换，让相邻位置的向量角度略有不同。

### 投影：把高维压到低维

投影矩阵把空间"压扁"。比如：

$$P = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$

这个矩阵把所有向量压到 $x$ 轴上——$y$ 坐标变成 0。你丢掉了维度信息。

在神经网络里，降维投影无处不在。Transformer 的注意力机制用 $W_V$ 矩阵把高维向量投影到 $d_v$ 维的 Value 空间，本质上就是一种信息压缩——只保留对当前任务有用的特征。

### 剪切：非对角元素的效果

剪切矩阵：

$$S = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$$

它的效果是：把正方形变成平行四边形——顶部向右平移，底部不动。非对角线上的元素 $s_{12} = 1$ 意味着 $y$ 方向的坐标会"溢出"影响 $x$ 方向。

**这个概念在神经网络中极其重要。** 当权重矩阵的非对角元素不为零时，不同的输入特征之间就会产生混合和交互——这正是神经网络学习特征组合的方式。

### 用 Python 可视化这些变换

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_transform(A, title, ax):
    """画出单位正方形经过矩阵 A 变换后的形状"""
    # 单位正方形的四个顶点
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    # 变换后的正方形
    transformed = A @ square

    ax.plot(square[0], square[1], 'b--', alpha=0.5, label='原始')
    ax.fill(square[0], square[1], alpha=0.1, color='blue')
    ax.plot(transformed[0], transformed[1], 'r-', linewidth=2, label='变换后')
    ax.fill(transformed[0], transformed[1], alpha=0.2, color='red')
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 拉伸
plot_transform(np.array([[2, 0], [0, 0.5]]), '拉伸 (对角矩阵)', axes[0, 0])
# 旋转 45°
theta = np.pi / 4
plot_transform(np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta),  np.cos(theta)]]),
               '旋转 45° (正交矩阵)', axes[0, 1])
# 投影到 x 轴
plot_transform(np.array([[1, 0], [0, 0]]), '投影到 x 轴', axes[1, 0])
# 剪切
plot_transform(np.array([[1, 1], [0, 1]]), '剪切', axes[1, 1])

plt.tight_layout()
plt.savefig('transforms.png', dpi=150)
plt.show()
```

运行这段代码，你会看到一个蓝色虚线的单位正方形，和经过变换后的红色实心区域。拉伸、旋转、投影、剪切——四种基本操作，一目了然。而任意一个复杂的矩阵变换，本质上都是这几种基本操作的组合。

---

## 3.3 为什么要"线性"——非线性激活函数的角色

### 线性变换的致命局限

线性变换有一个"看起来是优点，其实是致命缺陷"的性质：

> **多个线性变换叠加在一起，等价于一个线性变换。**

数学上表达就是：$A(B\mathbf{x}) = (AB)\mathbf{x}$。你用两个矩阵依次变换，和用一个矩阵 $C = AB$ 变换一次，效果完全相同。

这意味着什么？**如果你把 100 层线性变换叠在一起，它的表达能力不比一层更强。** 100 层矩阵乘法可以合并成一个矩阵乘法。深度就失去了意义。

### 为什么需要非线性

为了让多层网络的表达力真正增长，我们必须在层与层之间引入**非线性**——打破线性叠加的等价性。

最常见的非线性激活函数：

- **ReLU**：$\text{ReLU}(x) = \max(0, x)$。简单粗暴：负数变零，正数不变。看起来很简陋，但它打破了线性叠加，而且计算极快。AlexNet (2012) 首次大规模使用 ReLU，直接引爆了深度学习革命。

- **GELU**：$\text{GELU}(x) = x \cdot \Phi(x)$，其中 $\Phi(x)$ 是标准正态分布的累积分布函数。GELU 是一种"平滑版的 ReLU"——不是硬截断，而是按概率平滑过渡。GPT、BERT、以及几乎所有 Transformer 模型都用 GELU。

- **Sigmoid / Tanh**：早期神经网络的主力，现在主要用在门控机制中（如 LSTM 的遗忘门、Transformer 的某些变体）。

### 神经网络 = 线性变换 + 非线性激活，反复叠加

一个典型的全连接层：

$$\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$$

其中 $W\mathbf{x}$ 是线性变换，$\mathbf{b}$ 是偏置（平移），$\sigma$ 是非线性激活函数。多层网络就是反复执行这个操作：

$$\mathbf{x} \xrightarrow{W_1, b_1, \sigma} \mathbf{h}_1 \xrightarrow{W_2, b_2, \sigma} \mathbf{h}_2 \xrightarrow{W_3, b_3, \sigma} \cdots \xrightarrow{W_L, b_L} \mathbf{y}$$

每一层先用线性变换在空间中"扭曲"数据，再用激活函数"弯折"空间。一层做不出来的复杂形状，多层叠加就能做出来。这就是深度的力量。

---

## 3.4 线性变换与神经网络层

### 每一个全连接层就是矩阵乘法 + 偏置

拆开任何一个全连接层（Fully Connected Layer / Linear Layer），核心操作就是：

$$\mathbf{y} = W\mathbf{x} + \mathbf{b}$$

其中 $W$ 是 $d_{out} \times d_{in}$ 的权重矩阵，$\mathbf{x}$ 是 $d_{in}$ 维输入向量，$\mathbf{b}$ 是 $d_{out}$ 维偏置向量，$\mathbf{y}$ 是输出。

几何意义上，$W\mathbf{x}$ 做的是线性变换（旋转、拉伸、投影等），$\mathbf{b}$ 做的是平移。合在一起叫做**仿射变换**（Affine Transformation）。线性变换保持原点不动，加上偏置后原点也可以移动了——这给了网络额外的自由度。

在 PyTorch 中，这就是 `nn.Linear(in_features, out_features)`。底层调用就是 `output = input @ weight.T + bias`，一个矩阵乘法加上一个向量加法。

### 为什么 Transformer 中叫"线性投影"

在 Transformer 原始论文《Attention Is All You Need》中，你会反复看到"线性投影"和"线性变换"这两个词。具体来说，注意力机制中的 Query、Key、Value 是这样生成的：

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

其中 $X$ 是输入序列的嵌入矩阵（$n \times d_{model}$），$W^Q, W^K, W^V$ 是三个投影矩阵（$d_{model} \times d_k$）。

这里叫"投影"是因为通常 $d_k < d_{model}$——从高维空间（比如 768 维）投影到低维空间（比如 64 维）。这就像把三维物体的影子投射到二维墙上，保留了一些信息，也丢掉了一些。

多头注意力（Multi-Head Attention）中，每一头有自己的 $W^Q, W^K, W^V$，各自投影到不同的低维子空间，在不同的子空间里关注不同的模式。最后再把所有头的结果拼起来，经过一个输出投影矩阵 $W^O$ 变回 $d_{model}$ 维。

**从线性代数的视角看，多头注意力的每一步都是矩阵乘法——都是线性变换。** 整个 Transformer 模型中绝大多数参数和计算都在做线性变换，只有激活函数和 Softmax 提供了非线性。

---

## 3.5 多层变换的叠加

### 深度网络的本质：变换的复合

一个 $L$ 层的神经网络做的事情，可以用一个巨大的复合函数来描述：

$$f(\mathbf{x}) = W_L \cdot \sigma(W_{L-1} \cdot \sigma(\cdots \sigma(W_1 \mathbf{x} + \mathbf{b}_1) \cdots) + \mathbf{b}_{L-1}) + \mathbf{b}_L$$

从几何视角看，这就是**一系列空间变形的叠加**。每一层把数据空间扭曲一次，激活函数再折叠一次。经过多层处理后，原本纠缠在一起的数据被"展开"了——同类数据聚在一起，不同类数据被推开。

在 Transformer 中，一个典型的编码器层包含：自注意力（涉及 $Q, K, V$ 的线性投影）$\to$ 残差连接 $\to$ 层归一化 $\to$ 前馈网络（两层线性变换夹一个 GELU）$\to$ 残差连接 $\to$ 层归一化。GPT-3 有 96 个这样的层，意味着输入文本的向量表示经历了 96 轮精心调校的空间变换，最终变成预测下一个词的概率分布。

### 残差连接：跳过一层变换

Transformer 论文中一个关键设计是**残差连接**（Residual Connection）：

$$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$

其中 $F(\mathbf{x})$ 是某一层（或几层）的变换，$\mathbf{x}$ 是直接从输入"跳"过来的。

### 为什么残差连接有效——从线性代数的角度

假设我们要学习的理想变换是恒等变换（什么都不做），即 $F(\mathbf{x}) = \mathbf{x}$。如果没有残差连接，网络需要学 $W \approx I$（单位矩阵），这在优化上并不容易——随机初始化的 $W$ 离 $I$ 可能很远。

有了残差连接，网络只需要学 $F(\mathbf{x}) = \mathbf{x} - \mathbf{x} = \mathbf{0}$，即 $F$ 接近零映射。让权重矩阵接近零矩阵，比让它接近单位矩阵容易得多——初始化时权重已经很小了。

更一般地说，残差连接把学习目标从"学习完整的变换"变成了"学习对恒等变换的修正"。这在数学上等价于说：输出是单位变换（$I$）加上一个小修正（$F$）。当你靠近单位变换时，梯度流动更稳定，优化更顺畅。

在 Transformer 的每一层中，你都会看到：

$$\text{Output} = \text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$$

那个 $\mathbf{x} +$ 就是残差连接——把输入直接加到输出上。这使得即使 Transformer 堆叠了几十层甚至上百层，梯度仍然可以通过残差路径顺畅地回传。

---

## 3.6 Python 实战：可视化神经网络的变换

理论讲了这么多，不如用代码亲手感受一下。我们来做一个直观的实验：生成一些二维数据点，看神经网络如何通过线性变换 + 非线性激活来扭曲空间，把不同类别的数据分离开。

### 可视化基本矩阵变换的效果

首先，让我们生成一组二维点，然后观察它们经过不同矩阵变换后的位置变化：

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成两类二维数据
np.random.seed(42)
n = 100
class_A = np.random.randn(n, 2) + np.array([2, 2])
class_B = np.random.randn(n, 2) + np.array([-2, -2])

# 定义不同的变换矩阵
transforms = {
    '原始数据': np.eye(2),
    '拉伸 (x2, x0.5)': np.array([[2, 0], [0, 0.5]]),
    '旋转 30°': lambda: (lambda t: np.array([[np.cos(t), -np.sin(t)],
                                               [np.sin(t),  np.cos(t)]]))(np.pi/6),
    '剪切': np.array([[1, 0.8], [0, 1]]),
}

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for ax, (title, A) in zip(axes, transforms.items()):
    if callable(A):
        A = A()
    tA = class_A @ A.T
    tB = class_B @ A.T
    ax.scatter(tA[:, 0], tA[:, 1], c='blue', alpha=0.6, label='类 A')
    ax.scatter(tB[:, 0], tB[:, 1], c='red', alpha=0.6, label='类 B')
    ax.set_title(title)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('data_transforms.png', dpi=150)
plt.show()
```

你会发现，线性变换可以拉伸和旋转数据，但**无法让两类数据从线性不可分变成线性可分**——如果画一条直线不能分开它们，那任何线性变换之后依然不能。这就是我们需要非线性激活的原因。

### 可视化两层神经网络如何扭曲空间

现在我们训练一个极简的 2 层网络，观察它是如何扭曲空间来分类数据的：

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成 XOR 型数据（线性不可分）
np.random.seed(0)
torch.manual_seed(0)
n = 200
X = np.random.randn(n, 2) * 1.5
y = (X[:, 0] * X[:, 1] > 0).astype(float)  # XOR 模式：一三象限为 1

X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).unsqueeze(1)

# 极简 2 层网络
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.BCELoss()

for epoch in range(300):
    optimizer.zero_grad()
    pred = model(X_tensor)
    loss = loss_fn(pred, y_tensor)
    loss.backward()
    optimizer.step()

# 可视化决策边界
xx, yy = np.meshgrid(np.linspace(-4, 4, 200),
                       np.linspace(-4, 4, 200))
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
with torch.no_grad():
    zz = model(grid).numpy().reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：原始数据
axes[0].scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.6, label='类别 0')
axes[0].scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.6, label='类别 1')
axes[0].set_title('原始数据（XOR 模式，线性不可分）')
axes[0].legend()
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# 右图：网络学到的决策边界
axes[1].contourf(xx, yy, zz, levels=20, cmap='RdBu', alpha=0.7)
axes[1].scatter(X[y==0, 0], X[y==0, 1], c='blue', edgecolors='k', s=30)
axes[1].scatter(X[y==1, 0], X[y==1, 1], c='red', edgecolors='k', s=30)
axes[1].set_title('神经网络学到的决策边界')
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('nn_decision_boundary.png', dpi=150)
plt.show()
```

左图中，两类数据呈 XOR 分布（对角线上的点同色），你不可能画一条直线把它们分开。右图中，神经网络通过两层线性变换加 ReLU 激活，学会了弯曲决策边界，完美地将两类数据分开。

**这就是线性变换 + 非线性的魔力。** 每一层线性变换旋转和拉伸空间，ReLU 把空间"折弯"，多层叠加后，空间被扭曲成了足够复杂的形状，可以包围住任意分布的数据。

### 更进一步：观察隐藏层的空间变换

让我们把第一层隐藏层的输出取出来，看看线性变换 + ReLU 做了什么：

```python
# 提取第一层变换后的隐藏表示
with torch.no_grad():
    hidden = torch.relu(model[0](X_tensor)).numpy()

# 可视化前两个隐藏维度
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.scatter(hidden[y==0, 0], hidden[y==0, 1], c='blue', alpha=0.6, label='类别 0')
ax.scatter(hidden[y==1, 0], hidden[y==1, 1], c='red', alpha=0.6, label='类别 1')
ax.set_title('第一层隐藏表示（前两个维度）')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hidden_representation.png', dpi=150)
plt.show()
```

你会发现，经过第一层的 $W\mathbf{x} + \mathbf{b}$ 和 ReLU 之后，原本纠缠在一起的 XOR 数据已经被"展开"了——两类数据在新的空间里变得更容易分开。这就是神经网络每一层在做的事情：**逐步将数据变换到更容易处理的空间**。

---

## 小结

让我们把这一章的核心思想串起来：

1. **矩阵就是线性变换的配方**——矩阵的每一列告诉你基向量变换到了哪里。
2. **四种基本变换**——拉伸（对角矩阵）、旋转（正交矩阵）、投影（降维）、剪切（非对角元素）——任何矩阵变换都是它们的组合。
3. **线性变换的局限**——叠加多次线性变换等价于一次线性变换，所以深度没有意义。必须加入非线性激活函数（ReLU、GELU）。
4. **神经网络层 = 线性变换 + 偏置 + 非线性激活**，即 $h = \sigma(Wx + b)$。Transformer 中的 $Q, K, V$ 投影本质上就是线性变换。
5. **残差连接**让网络靠近单位变换，使优化更稳定。
6. **多层叠加**的效果是逐步扭曲空间，让数据变得可分——这正是深度学习能奏效的根本原因。

下一章，我们将深入"特征值与特征向量"——那些在变换中方向不变的"特殊向量"，以及它们如何帮助我们理解 Transformer 的注意力模式。
