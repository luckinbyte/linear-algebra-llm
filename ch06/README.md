# 第6章 特征值与特征向量——从桥梁共振到语义骨架

> 如果矩阵是一个变换的"配方"，那特征向量就是配方里最不可或缺的那几味主料——它们抓住了变换的本质。

## 6.1 什么是特征值和特征向量

### 从一个反直觉的现象说起

想象你拿着一张照片，对它做一个线性变换——拉伸、旋转、剪切……大多数向量被变换后，方向都变了。但在几乎每个变换里，都藏着一些"倔强"的向量：它们只改变长度，不改变方向。

这就是**特征向量**（eigenvector）。

数学上，如果一个方阵 $A$ 作用在向量 $v$ 上，结果只是把 $v$ 乘了一个标量：

$$A v = \lambda v$$

那 $v$ 就叫 $A$ 的特征向量，$\lambda$ 就是对应的**特征值**（eigenvalue）。

注意这里 $v \neq 0$——零向量当然满足等式，但那太无聊了，不算。

### 生活中的类比

**旋转陀螺**：陀螺高速旋转时，旋转轴的方向始终不变（忽略进动），这个方向就是旋转矩阵的一个特征向量。而旋转的角速度，则对应特征值的大小。当然，严格来说三维旋转矩阵的特征值可能是复数，后面我们会解释。

**吉他弦的振动**：拨动吉他弦时，弦不是随机乱抖的。它会在几个特定的"驻波模式"上振动——基频、二次谐波、三次谐波……每一个驻波模式就是一个特征向量，对应的振动频率就是特征值。最低频的那个模式最显著（声音最大），对应最大的特征值。工程师把这个叫做**模态分析**。

**桥的共振**：塔科马海峡大桥在1940年因风振而坍塌。风吹对桥梁施加的力，恰好在桥梁某个"特征振动模式"（特征向量）的频率（特征值）附近产生了共振，导致振幅越来越大。如果你去问任何一个土木工程师"特征值是什么"，他会告诉你：那是我必须避免共振的东西。

## 6.2 几何直觉

### 特征向量是变换中最"稳定"的方向

让我们在二维平面上建立一个直觉。考虑矩阵：

$$A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$$

这个矩阵把单位圆变成一个椭圆。大多数向量在变换后方向都变了，但有两条线上的向量只被拉伸了——它们就是特征向量。

对于上面这个矩阵，可以算出：

- $\lambda_1 = 3$，对应的特征向量 $v_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$
- $\lambda_2 = 2$，对应的特征向量 $v_2 = \begin{pmatrix} -1 \\ 1 \end{pmatrix}$

沿着 $v_1$ 方向的向量被拉长了 3 倍，沿着 $v_2$ 方向的向量被拉长了 2 倍。其他所有方向都被"扭"了一下。

### 特征值的含义

特征值 $\lambda$ 的**绝对值**告诉你变换在那个方向上拉伸或压缩了多少：

- $|\lambda| > 1$：这个方向被**拉伸**
- $|\lambda| < 1$：这个方向被**压缩**
- $|\lambda| = 1$：这个方向长度**不变**

而 $\lambda$ 的**符号**和**虚实**则告诉你更深层的信息：

| 特征值类型 | 几何含义 |
|---|---|
| 正实数 $\lambda > 0$ | 纯拉伸，方向不变 |
| 负实数 $\lambda < 0$ | 拉伸 + 翻转（方向反转180度） |
| 复数 $\lambda = a + bi$ | 包含旋转成分，$b$ 决定旋转的"角度感" |

一个经典例子：二维旋转矩阵

$$R = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

它的特征值是 $\lambda = \cos\theta \pm i\sin\theta = e^{\pm i\theta}$，是一对共轭复数。这完全合理——旋转就是没有哪个方向是"纯粹拉伸"的，所以没有实数特征向量。

### Python可视化：看特征向量的方向

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_eigenvectors(A, title):
    """可视化矩阵变换和特征向量"""
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # 画单位圆
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])

    # 变换后的椭圆
    transformed = A @ circle

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, label in zip(axes, [circle, transformed], ['变换前', '变换后']):
        ax.plot(data[0], data[1], 'b-', alpha=0.5)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(label)

        # 画特征向量
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i].real
            if np.linalg.norm(v) > 1e-10:
                ax.arrow(0, 0, v[0], v[1],
                        head_width=0.15, head_length=0.1,
                        color=['red', 'green'][i], linewidth=2,
                        label=f'$v_{i+1}$, $\\lambda$={eigenvalues[i]:.2f}')

        ax.legend()
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('eigenvectors_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

# 示例1：对称矩阵（拉伸）
A1 = np.array([[3, 1],
               [1, 2]])
plot_eigenvectors(A1, '对称矩阵的特征向量（正交）')

# 示例2：非对称矩阵（剪切 + 拉伸）
A2 = np.array([[2, 1],
               [0, 3]])
plot_eigenvectors(A2, '非对称矩阵的特征向量（不正交）')
```

运行这段代码，你会看到红色和绿色箭头就是特征向量的方向——变换前后它们始终指向同一个方向（或相反方向），而圆上的其他点都被"扭"到了新位置。

## 6.3 特征值分解：EVD

### 矩阵的"骨架"——对角化

如果一个 $n \times n$ 矩阵 $A$ 有 $n$ 个线性无关的特征向量（不是所有矩阵都有），我们可以把它们排成矩阵 $Q$ 的列：

$$Q = \begin{pmatrix} | & | & & | \\ v_1 & v_2 & \cdots & v_n \\ | & | & & | \end{pmatrix}$$

然后把对应的特征值放在对角矩阵 $\Lambda$ 上：

$$\Lambda = \begin{pmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_n \end{pmatrix}$$

于是矩阵 $A$ 可以被分解为：

$$A = Q \Lambda Q^{-1}$$

这就是**特征值分解**（Eigenvalue Decomposition，EVD）。

这意味着什么？$A$ 作用在任意向量 $x$ 上：

$$Ax = Q \Lambda Q^{-1} x$$

可以读作三步操作：
1. $Q^{-1} x$：把 $x$ 变换到特征向量构成的坐标系
2. $\Lambda(\cdot)$：在每个特征方向上独立地拉伸/压缩
3. $Q(\cdot)$：变回原来的坐标系

**EVD的本质就是找到矩阵变换的"骨架"——在特征向量的坐标系里，变换变成了最简单的对角缩放。**

### 对称矩阵的特殊性质

对称矩阵（$A = A^T$）在工程和数据科学中无处不在，它有几个特别好的性质：

1. **特征值全为实数**——不用操心复数
2. **特征向量互相正交**——$v_i \perp v_j$（当 $i \neq j$）
3. **一定可以对角化**——永远有 $n$ 个线性无关的特征向量

因此对称矩阵的 EVD 变得更加优雅：

$$A = Q \Lambda Q^T$$

注意 $Q^{-1} = Q^T$（因为 $Q$ 是正交矩阵），不需要求逆！

这也是为什么**协方差矩阵**（永远对称）做 PCA 那么方便——它的特征向量天然正交，每个方向的信息完全不重叠。

### 不是所有矩阵都能对角化

一个反例：

$$B = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$$

这个矩阵只有一个特征值 $\lambda = 1$（重根），但只找到一个线性无关的特征向量 $v = (1, 0)^T$。它不能被对角化。这类矩阵需要用 **Jordan 标准形**来处理，不过在实际的机器学习场景中，这种情况不太常见。

## 6.4 从桥梁到语义：PCA（主成分分析）

### 桥梁共振与特征值

回到桥梁的例子。工程师用**刚度矩阵** $K$ 和**质量矩阵** $M$ 建模桥梁的振动，求解广义特征值问题：

$$K v = \lambda M v$$

得到的每个特征向量 $v_i$ 代表一种振动模式（也叫"模态"），对应的特征值 $\lambda_i$ 就是该模式的固有频率的平方 $\omega_i^2$。

**最大的特征值对应最高频的振动模式，最小的特征值对应最低频（最容易激发的）模式。** 塔科马大桥的悲剧正是因为风力激励恰好匹配了某个低频模态。

### PCA：数据的"振动模式"

PCA（主成分分析）做的事情和模态分析如出一辙，只是把物理结构换成了数据。

给定一组数据中心（已去均值），我们计算**协方差矩阵**：

$$C = \frac{1}{n-1} X^T X$$

其中 $X$ 是 $n \times d$ 的数据矩阵（$n$ 个样本，$d$ 个特征）。

$C$ 是对称半正定矩阵，所以一定可以 EVD：

$$C = Q \Lambda Q^T$$

特征向量 $q_1, q_2, \ldots, q_d$ 就是数据的**主成分**——它们是数据变化最大、次大、……最小的方向。对应的特征值 $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_d \geq 0$ 告诉你每个方向上的方差（信息量）。

**降维的核心思想**：只保留前 $k$ 个（最大特征值对应的）主成分，把数据从 $d$ 维投影到 $k$ 维：

$$X_{\text{reduced}} = X \cdot Q_k$$

其中 $Q_k$ 只包含前 $k$ 个特征向量。你丢弃的维度是方差最小的方向——也就是信息量最少的方向。

### 在NLP中：语义空间的骨架

想象你有一个 300 维的词向量空间。词与词之间的关系千丝万缕，但也许真正重要的语义维度只有几十个——"性别"、"年龄"、"褒贬"、"具体/抽象"……

PCA 告诉我们：计算词向量协方差矩阵的特征值分解，前几个特征向量往往对应着最显著的语义区分维度。虽然这些维度不一定能被人类直接命名，但它们确实捕捉了语义空间中"变化最大"的方向。

这就是为什么把 300 维的词向量用 PCA 压缩到 50 维，语义信息损失很小——因为 300 维中的大部分维度方差极小，几乎不携带有用信息。

```python
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_20newsgroups
import numpy as np

# 模拟：假设我们有一些词向量（这里用随机数据演示流程）
np.random.seed(42)
n_words, dim = 1000, 300
word_vectors = np.random.randn(n_words, dim)

# 真实数据中，word_vectors 来自 Word2Vec / GloVe 等模型

# PCA 降维
pca = PCA()
pca.fit(word_vectors)

# 看特征值（解释方差）的分布
explained_var_ratio = pca.explained_variance_ratio_
cumulative = np.cumsum(explained_var_ratio)

print(f"前 10 个主成分解释了 {cumulative[9]*100:.1f}% 的方差")
print(f"前 50 个主成分解释了 {cumulative[49]*100:.1f}% 的方差")

# 降到 50 维
pca_50 = PCA(n_components=50)
word_vectors_50d = pca_50.fit_transform(word_vectors)
print(f"降维后形状: {word_vectors_50d.shape}")
```

## 6.5 特征值与 LLM

现在我们把目光转向大语言模型。特征值在 Transformer 的运作中扮演着至关重要的角色。

### 注意力矩阵的特征值分布

Transformer 的自注意力机制计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中 $QK^T$ 是一个 $n \times n$ 的注意力得分矩阵（$n$ 是序列长度）。经过 softmax 后，每一行加起来等于 1，所以它是一个随机矩阵（行随机矩阵）。

这个矩阵的**特征值分布**决定了信息如何在 token 之间流动：

- **一个大的特征值（接近 1）**：对应"共识方向"，所有 token 在这个方向上趋同——模型捕捉到了全局语义
- **衰减的特征值谱**：其余特征值越小，说明注意力越集中在少数 token 上（接近"硬注意力"）
- **均匀分布的特征值**：每个方向权重差不多，说明注意力很分散（接近均匀注意力）

研究表明，训练良好的 Transformer 倾向于形成**低秩的注意力模式**——即少数几个大的特征值主导了信息流动。这不是巧合，而是自然语言本身的特性决定的：一段文本的语义通常可以用少数几个"主题方向"来概括。

### 谱归一化（Spectral Normalization）

在 GAN 的训练中，谱归一化是一种极其有效的稳定训练的技术。它的核心思想用一句话就能概括：**用矩阵的最大奇异值（即最大特征值的平方根，对于 $A^TA$ 而言）来归一化权重矩阵。**

虽然严格来说这里用的是奇异值而非特征值（因为权重矩阵通常不是方阵），但思想是相通的：

$$W_{\text{SN}} = \frac{W}{\sigma_{\max}(W)}$$

其中 $\sigma_{\max}(W)$ 是 $W$ 的最大奇异值（即 $W^T W$ 最大特征值的平方根）。

这保证了权重矩阵的**谱范数**（spectral norm）为 1，从而限制了每一层对输入的放大倍数，防止梯度爆炸。

在 LLM 的语境下，类似的思路也被用于控制 Transformer 层的**Lipschitz 常数**，确保信息逐层传播时不会指数级放大或衰减。

### 训练稳定性与特征值的深层关系

训练深度网络时，如果权重矩阵 $W$ 的最大特征值 $|\lambda_{\max}| > 1$，信号在前向传播时会被逐层放大，最终爆炸。如果 $|\lambda_{\max}| < 1$，信号会逐层衰减为噪声，最终梯度消失。

理想的训练状态是权重矩阵的谱（所有特征值的集合）保持在 1 附近——这正是残差连接（Residual Connection）和层归一化（Layer Normalization）的设计动机之一。

残差连接把 $f(x)$ 变成了 $x + f(x)$。从特征值的角度看，即使 $f$ 的特征值接近 0，$x + f(x)$ 的特征值仍然在 1 附近——信息至少能通过"恒等映射"这条路径传递下去。这也是为什么 Transformer 中每个子层都有残差连接：

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

从特征值的视角，这确保了无论 $\text{Sublayer}$ 学到了什么，信息都不会完全丢失。

## 6.6 Python 实战

让我们把本章的核心概念全部用代码实现一遍。

### 用 NumPy 计算特征值和特征向量

```python
import numpy as np

# 定义一个矩阵
A = np.array([[4, 1],
              [2, 3]])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print("特征值:", eigenvalues)
# 输出: 特征值: [5. 2.]

print("特征向量 (每列一个):")
print(eigenvectors)

# 验证 A @ v = lambda * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lambda_v = eigenvalues[i]
    left = A @ v
    right = lambda_v * v
    print(f"特征值 {lambda_v:.2f}: Av = {left}, λv = {right}")
    print(f"  差异: {np.allclose(left, right)}")
```

### 对称矩阵的 EVD

```python
# 对称矩阵
S = np.array([[2, 1, 0],
              [1, 3, 1],
              [0, 1, 2]])

eigenvalues, Q = np.linalg.eigh(S)  # eigh 专用于对称矩阵，更高效

print("对称矩阵的特征值:", eigenvalues)
print("特征向量矩阵 Q:")
print(Q)

# 验证 Q 是正交矩阵
print("\nQ^T @ Q (应该接近单位矩阵):")
print(np.round(Q.T @ Q, 10))

# 验证 EVD: S = Q @ diag(eigenvalues) @ Q^T
Lambda = np.diag(eigenvalues)
S_reconstructed = Q @ Lambda @ Q.T
print("\n重构的 S (应该与原矩阵一致):")
print(np.round(S_reconstructed, 10))
```

### 对词向量矩阵做 PCA 降维并可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 模拟词向量数据
# 假设词向量有内在的低维结构
np.random.seed(42)

# 构造 5 个语义簇
n_per_cluster = 50
dim = 100
centers = np.random.randn(5, dim) * 5  # 5 个簇中心，间隔较远
labels = []
vectors = []

for i, center in enumerate(centers):
    noise = np.random.randn(n_per_cluster, dim) * 0.5
    cluster = center + noise
    vectors.append(cluster)
    labels.extend([f'簇{i+1}'] * n_per_cluster)

X = np.vstack(vectors)  # shape: (250, 100)

# ---- 手动实现 PCA ----
# 1. 中心化
X_centered = X - X.mean(axis=0)

# 2. 计算协方差矩阵
cov_matrix = (X_centered.T @ X_centered) / (len(X) - 1)

# 3. 特征值分解
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 4. 按特征值从大到小排序
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 5. 取前 2 个主成分
PC2 = eigenvectors[:, :2]
X_pca_manual = X_centered @ PC2

# ---- 用 sklearn 验证 ----
pca = PCA(n_components=2)
X_pca_sklearn = pca.fit_transform(X)

print(f"前 2 个主成分解释方差比: {pca.explained_variance_ratio_}")
print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"  合计: {sum(pca.explained_variance_ratio_)*100:.1f}%")

# 可视化
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
for i in range(5):
    mask = [l == f'簇{i+1}' for l in labels]
    ax.scatter(X_pca_sklearn[mask, 0], X_pca_sklearn[mask, 1],
              c=colors[i], label=f'簇{i+1}', alpha=0.7, s=30)

ax.set_xlabel('第一主成分')
ax.set_ylabel('第二主成分')
ax.set_title('PCA 降维可视化：从 100 维到 2 维')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_word_vectors.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 可视化特征值谱

```python
# 展示特征值谱——理解 PCA 保留多少信息
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 特征值（方差）
axes[0].bar(range(1, 21), eigenvalues[:20], color='steelblue')
axes[0].set_xlabel('主成分编号')
axes[0].set_ylabel('特征值（方差）')
axes[0].set_title('前 20 个特征值')

# 累计解释方差比
cumulative_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
axes[1].plot(range(1, 101), cumulative_var, 'o-', markersize=2)
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% 方差')
axes[1].set_xlabel('主成分数量')
axes[1].set_ylabel('累计解释方差比')
axes[1].set_title('累计解释方差比')
axes[1].legend()

plt.tight_layout()
plt.savefig('eigenvalue_spectrum.png', dpi=150, bbox_inches='tight')
plt.show()

# 找到解释 95% 方差需要多少主成分
n_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"解释 95% 方差只需要 {n_95} 个主成分（原始维度: {dim}）")
print(f"压缩比: {dim/n_95:.1f}x")
```

---

## 小结

特征值和特征向量是线性代数中最有"实用气质"的概念之一。它们不是抽象的数学游戏，而是工程和 AI 中的核心工具：

- **定义**：$Av = \lambda v$，特征向量是变换中方向不变的向量，特征值是拉伸/压缩的倍数
- **EVD**：$A = Q\Lambda Q^{-1}$，把矩阵变换分解到最简洁的对角形式
- **PCA**：协方差矩阵的特征值分解，找到数据中最重要的方向
- **LLM**：注意力矩阵的谱分布影响信息流动，谱归一化用最大特征值稳定训练，残差连接用特征值视角解释为什么深度网络能训练

下一章我们将看到特征值分解的一个强大推广——奇异值分解（SVD），它能处理非方阵，是推荐系统和语言模型的底层引擎之一。
