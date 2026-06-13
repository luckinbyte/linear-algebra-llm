# 第5章 矩阵分解——从密码学到知识压缩

你小时候有没有拆过家里的收音机？把后盖撬开，里面是密密麻麻的电路板、电容、电阻和线圈。那些零件单独看并不复杂，但它们组合在一起，就能接收无线电波、播放出音乐。如果你只看收音机的"黑盒子"外观——一个旋钮、一个喇叭——你永远不会理解它是怎么工作的。拆开它，理解每个零件的功能，你才真正掌握了这台机器。

矩阵分解做的事，本质上就是"拆矩阵的壳"。

一个矩阵放在你面前，就像一个密封的黑盒子。你知道它能做什么——乘以一个向量，输出另一个向量——但你不知道它内部的结构。矩阵分解就是把一个复杂的矩阵"拆"成几个简单矩阵的乘积，就像拆解一台机器，让你看清内部的齿轮、弹簧和杠杆是如何协同工作的。

如果你更喜欢数学的类比：还记得小学学因式分解吗？$x^2 + 5x + 6$ 看上去是一个完整的二次多项式，但把它因式分解成 $(x + 2)(x + 3)$ 之后，你一眼就能看出它的根是 $-2$ 和 $-3$。因式分解让一个复杂的表达式变得透明。矩阵分解做的也是同样的事，只是对象从多项式变成了矩阵。

这一章，我们要学习三种最重要的矩阵分解方法：LU分解、QR分解和Cholesky分解。每一种都像是拆解矩阵的一把不同的"螺丝刀"，适用于不同类型的矩阵，揭示不同类型的内部结构。更重要的是，我们会看到这些分解技术与大语言模型（LLM）之间的深层联系——当你理解了矩阵可以怎样被拆开，你也就理解了LLM中的知识可以怎样被压缩。

---

## 5.1 为什么要分解矩阵

### 黑盒子与透明盒子

矩阵乘法是一个"信息融合"的过程。当你计算 $\mathbf{y} = A\mathbf{x}$ 时，输入向量 $\mathbf{x}$ 的每个分量被混合、重组，变成输出向量 $\mathbf{y}$。这个混合过程发生在一个黑盒子里——你看不到 $A$ 是怎么工作的。

但如果你能把 $A$ 拆开呢？比如，如果你发现：

$$A = BC$$

那么 $\mathbf{y} = A\mathbf{x}$ 就变成了两步操作：

$$\mathbf{z} = C\mathbf{x}, \quad \mathbf{y} = B\mathbf{z}$$

每一步都比你直接计算 $A\mathbf{x}$ 更简单、更透明。这就是矩阵分解的核心价值：**用简单操作的组合来替代复杂操作。**

### 三大收益

矩阵分解带来三个直接好处：

**计算加速。** 假设你要用矩阵 $A$ 解方程 $A\mathbf{x} = \mathbf{b}$。直接求解一个 $n \times n$ 的方程组需要 $O(n^3)$ 次运算。但如果你已经知道 $A = LU$（一个下三角矩阵乘以上三角矩阵），那就可以分两步求解，每步只需要 $O(n^2)$。对于需要反复用同一个 $A$ 解不同 $\mathbf{b}$ 的问题，分解一次、反复使用，效率提升巨大。

**结构揭示。** 一个普通的矩阵看起来杂乱无章，但分解后的部件有明确的几何意义。比如QR分解告诉你，这个线性变换可以被理解为"先旋转、再拉伸"——一个清晰的几何故事。

**数值稳定。** 在计算机里用浮点数做矩阵运算，不可避免地会有舍入误差。直接操作一个"病态"的矩阵可能导致灾难性的精度损失，而分解后的部件通常更"温顺"，更容易做数值控制。

### 分解的多样性

值得强调的是，同一个矩阵可以被不同方式分解，就像同一台机器可以用不同方式拆解——按功能模块拆、按物理位置拆、按电路连接拆。常见的矩阵分解包括：

- **LU分解**：$A = LU$（下三角 × 上三角）——适合解方程组
- **QR分解**：$A = QR$（正交 × 上三角）——适合最小二乘
- **Cholesky分解**：$A = LL^T$（下三角 × 其转置）——对称正定矩阵的专属
- **特征值分解**：$A = P\Lambda P^{-1}$（下一章详细讲）
- **SVD**：$A = U\Sigma V^T$（下一章详细讲）

这一章我们聚焦前三种，后两种留到下一章。原因很简单：LU、QR、Cholesky 是"计算工具型"分解，是数值线性代数的基础设施；特征值分解和SVD是"结构洞察型"分解，是理解矩阵灵魂的钥匙。两种类型都很重要，但学习上有先后顺序。

---

## 5.2 LU分解

### 密码学的类比：把一步加密变成两步

想象你是一个密码学家。你的任务是设计一个加密函数 $E$，把明文 $\mathbf{x}$ 变成密文 $\mathbf{y}$：

$$\mathbf{y} = E(\mathbf{x})$$

如果 $E$ 是一个极其复杂的变换，直接实现它既慢又容易出错。但如果你发现 $E$ 可以拆成两步——先做一个变换 $L$，再做一个变换 $U$：

$$\mathbf{y} = U(L(\mathbf{x}))$$

每一步都相对简单，那就好办了。你只需要实现两个简单的变换，串联起来就等效于那个复杂的 $E$。而且，如果需要解密（做逆操作），你只需要分别对 $U$ 和 $L$ 求逆，然后反着来。

LU分解做的就是这个事。

### $A = LU$：一个矩阵 = 一个下三角 × 一个上三角

**LU分解**将矩阵 $A$ 分解为一个下三角矩阵 $L$（Lower triangular）和一个上三角矩阵 $U$（Upper triangular）的乘积：

$$A = LU$$

其中：

$$L = \begin{pmatrix} 1 & 0 & 0 \\ l_{21} & 1 & 0 \\ l_{31} & l_{32} & 1 \end{pmatrix}, \quad U = \begin{pmatrix} u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33} \end{pmatrix}$$

$L$ 的对角线上通常放 1（这叫 Doolittle 分解），$L$ 的对角线下方有数字，上方全是 0；$U$ 的对角线上方有数字，下方全是 0。

为什么三角矩阵"简单"？因为它们代表了一种**逐层推进**的操作。来看一个 $3 \times 3$ 的下三角方程组 $L\mathbf{z} = \mathbf{b}$：

$$\begin{pmatrix} l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33} \end{pmatrix} \begin{pmatrix} z_1 \\ z_2 \\ z_3 \end{pmatrix} = \begin{pmatrix} b_1 \\ b_2 \\ b_3 \end{pmatrix}$$

第一行直接告诉你 $l_{11} z_1 = b_1$，所以 $z_1 = b_1 / l_{11}$。知道了 $z_1$，代入第二行就能求 $z_2$。知道了 $z_1$ 和 $z_2$，代入第三行就能求 $z_3$。这种从上往下一步一步求解的过程叫**前向替换**（forward substitution），复杂度只有 $O(n^2)$，远低于一般方程组的 $O(n^3)$。

上三角方程组 $U\mathbf{x} = \mathbf{z}$ 同理，只是从下往上算，叫**后向替换**（backward substitution）。

### 解线性方程组的核心算法

LU分解最经典的应用就是解方程组 $A\mathbf{x} = \mathbf{b}$。思路是：

1. **分解**：把 $A = LU$ 算出来（只做一次）
2. **前向替换**：解 $L\mathbf{z} = \mathbf{b}$，求 $\mathbf{z}$
3. **后向替换**：解 $U\mathbf{x} = \mathbf{z}$，求 $\mathbf{x}$

为什么这样做？因为 $A\mathbf{x} = \mathbf{b}$ 等价于 $LU\mathbf{x} = \mathbf{b}$。令 $\mathbf{z} = U\mathbf{x}$，则先解 $L\mathbf{z} = \mathbf{b}$，再解 $U\mathbf{x} = \mathbf{z}$。两个三角方程组，各 $O(n^2)$。

这在实际中意味着什么？如果你有1000个不同的 $\mathbf{b}$，但矩阵 $A$ 不变，你只需要做一次LU分解（$O(n^3)$），然后对每个 $\mathbf{b}$ 做 $O(n^2)$ 的前向和后向替换。总代价从 $1000 \times O(n^3)$ 降到了 $O(n^3) + 1000 \times O(n^2)$。

在 LLM 的训练中，每次迭代都涉及大量线性方程求解和矩阵求逆操作。底层库（如 PyTorch）使用的高效线性求解器，背后几乎都依赖 LU 分解或其变种。理解了 LU 分解，你就理解了为什么 GPU 能在毫秒级别完成这些运算——因为三角矩阵的求解几乎可以被完美并行化。

### PA = LU：加入置换矩阵

并非所有矩阵都能直接做 LU 分解。有时你在分解过程中需要交换行，否则会遇到主元为零的情况（除以零！）。这时需要引入一个**置换矩阵** $P$（Permutation），它做的只是把矩阵的行重新排列：

$$PA = LU$$

$P$ 的每一行和每一列都恰好有一个 1，其余都是 0。你可以把 $P$ 理解为一个"打乱顺序"的操作。数值稳定的 LU 分解算法（如部分主元选取）总是输出 $PA = LU$ 的形式。

### Python演示：LU分解

```python
import numpy as np
from scipy.linalg import lu

# 一个 3x3 矩阵
A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)

# PA = LU 分解
P, L, U = lu(A)

print("P (置换矩阵):")
print(np.round(P, 4))
print("\nL (下三角矩阵):")
print(np.round(L, 4))
print("\nU (上三角矩阵):")
print(np.round(U, 4))
print("\n验证 P @ A ≈ L @ U:")
print(np.round(P @ A - L @ U, 10))  # 应该接近零矩阵

# 用 LU 分解解方程组 A @ x = b
b = np.array([1, 1, 1], dtype=float)

# 前向替换解 L @ z = P @ b
Pb = P @ b
z = np.linalg.solve(L, Pb)

# 后向替换解 U @ x = z
x = np.linalg.solve(U, z)

print(f"\n解 x = {x}")
print(f"验证 A @ x = {A @ x}（应该接近 {b}）")
```

运行后你会看到 $P$、$L$、$U$ 三个矩阵，以及验证 $PA = LU$ 的残差接近零。用分解后的矩阵解方程，结果与直接求解完全一致。

---

## 5.3 QR分解

### "歪斜"操作 = "旋转" + "拉伸"

想象你在健身房拉伸一根弹力绳。你先把绳子转到某个角度（旋转），然后再拉长它（拉伸）。这两个动作是独立的，先转后拉，效果和直接"歪斜着拉"一样——但拆成两步后，每一步都更容易理解和控制。

QR分解的直觉就在于此。任何一个矩阵 $A$ 代表的线性变换，都可以被拆成两步：先做一个正交变换 $Q$（"旋转/反射"），再做上三角变换 $R$（"拉伸/剪切"）。

### $A = QR$：正交矩阵 × 上三角矩阵

$$A = QR$$

其中 $Q$ 是正交矩阵（$Q^TQ = I$），$R$ 是上三角矩阵。

正交矩阵 $Q$ 有一个超级好的性质：**它的逆就是它的转置。** $Q^{-1} = Q^T$。这意味着什么？正交变换不改变向量的长度和角度——它只做旋转或镜像翻转。没有任何"压缩"或"拉伸"，信息量完美保留。

正交矩阵的列向量 $\mathbf{q}_1, \mathbf{q}_2, \ldots, \mathbf{q}_n$ 满足两两正交且单位长度：

$$\mathbf{q}_i^T \mathbf{q}_j = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

这正是 Gram-Schmidt 正交化的产物——把一组任意的向量，逐步正交化和单位化，变成一组"完美对齐"的坐标系。QR分解本质上就是记录了这个正交化的过程：$Q$ 是正交化后的新坐标系，$R$ 是从旧坐标系到新坐标系的"变换参数"。

### 最小二乘法中的应用

QR分解最重要的应用是**最小二乘问题**。当你有一个超定方程组（方程个数多于未知数个数）$A\mathbf{x} = \mathbf{b}$，通常没有精确解。你转而寻找让残差 $\|A\mathbf{x} - \mathbf{b}\|$ 最小的 $\mathbf{x}$。

直接用正规方程 $\hat{\mathbf{x}} = (A^TA)^{-1}A^T\mathbf{b}$ 在理论上是对的，但数值上经常翻车——$A^TA$ 的条件数是 $A$ 的条件数的平方，精度灾难。

QR分解给出了更优雅的解法：

$$A\mathbf{x} \approx \mathbf{b}$$
$$QR\mathbf{x} \approx \mathbf{b}$$
$$R\mathbf{x} \approx Q^T\mathbf{b}$$（左乘 $Q^T$，利用 $Q^TQ = I$）

因为 $R$ 是上三角矩阵，所以 $R\mathbf{x} = Q^T\mathbf{b}$ 可以用后向替换直接求解，不需要计算任何逆矩阵，数值稳定性远优于正规方程。

**在 LLM 中**，QR分解的身影出现在模型训练的深层。当你用最小二乘法拟合线性层（比如某些高效微调方法中的适配器权重），底层的求解器很可能用的就是基于QR分解的方法。此外，在 Transformer 的注意力机制中，$Q$（Query）矩阵的命名虽然巧合，但其背后的正交性思想——让查询方向尽量不重叠——与 QR 分解中正交化的精神一脉相承。

### Python演示：QR分解

```python
import numpy as np

# 一个 4x3 矩阵（行数 > 列数，典型的最小二乘场景）
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1],
              [1, 1, 1]], dtype=float)

# QR 分解
Q, R = np.linalg.qr(A, mode='reduced')

print("Q (正交矩阵，列向量两两正交且单位长度):")
print(np.round(Q, 4))
print(f"\n验证 Q^T @ Q ≈ I:")
print(np.round(Q.T @ Q, 4))

print("\nR (上三角矩阵):")
print(np.round(R, 4))
print(f"\n验证 Q @ R ≈ A:")
print(np.round(Q @ R, 4))

# 最小二乘：求解 A @ x ≈ b
b = np.array([1, 2, 3, 4], dtype=float)

# 用 QR 分解求解
x_qr = np.linalg.solve(R, Q.T @ b)

# 对比 numpy 内置的最小二乘
x_lstsq, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

print(f"\nQR分解解: {np.round(x_qr, 4)}")
print(f"numpy lstsq 解: {np.round(x_lstsq, 4)}")
print(f"两种方法结果一致: {np.allclose(x_qr, x_lstsq)}")
```

你会看到 $Q$ 的列确实两两正交，$R$ 是上三角的，两种方法给出了完全一致的最小二乘解。

---

## 5.4 Cholesky分解

### 对称正定矩阵的"专属通道"

想象一个VIP通道。不是所有人都能走，只有满足特定条件的人——持VIP卡的贵宾——才能进入。这个通道对贵宾来说超级快，几乎不需要排队。

Cholesky分解就是矩阵世界的VIP通道。它只为一种特殊的矩阵服务：**对称正定矩阵**（Symmetric Positive Definite, SPD）。但只要是这种矩阵，Cholesky分解的效率比LU分解快一倍，数值稳定性也更好。

什么是正定矩阵？一个矩阵 $A$ 是正定的，意味着对任意非零向量 $\mathbf{x}$，都有：

$$\mathbf{x}^T A \mathbf{x} > 0$$

直观地说，正定矩阵对应的二次函数 $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ 是一个"碗"形——在任意方向上都是向上弯曲的，有且只有一个最低点。

### $A = LL^T$：极致简洁的分解

Cholesky分解把对称正定矩阵 $A$ 分解为一个下三角矩阵 $L$ 和其转置 $L^T$ 的乘积：

$$A = LL^T$$

以 $3 \times 3$ 为例：

$$\begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{12} & a_{22} & a_{23} \\ a_{13} & a_{23} & a_{33} \end{pmatrix} = \begin{pmatrix} l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33} \end{pmatrix} \begin{pmatrix} l_{11} & l_{21} & l_{31} \\ 0 & l_{22} & l_{32} \\ 0 & 0 & l_{33} \end{pmatrix}$$

注意右边的两个矩阵互为转置！这意味着 Cholesky 分解只需要存储 $L$ 一个矩阵，存储量只有 LU 分解的一半。

计算 $L$ 的过程异常简洁。以第一行为例：

- $l_{11} = \sqrt{a_{11}}$
- $l_{21} = a_{12} / l_{11}$
- $l_{31} = a_{13} / l_{11}$

第二行：

- $l_{22} = \sqrt{a_{22} - l_{21}^2}$
- $l_{32} = (a_{23} - l_{21}l_{31}) / l_{22}$

每一步只需要用到前面已经算出来的元素，总共只需 $O(n^3/3)$ 次运算（LU分解需要 $O(n^3/3)$ 的两倍，因为LU不需要利用对称性）。没有部分主元选取的麻烦——正定矩阵保证了所有主元都是正的，永远不会除以零。

### 为什么协方差矩阵能做 Cholesky 分解

在实际应用中，最常遇到的对称正定矩阵就是**协方差矩阵**。

协方差矩阵 $\Sigma$ 衡量多维数据各维度之间的相关性。它的对角线是各维度的方差（恒正），非对角线是维度间的协方差。一个合法的协方差矩阵必然是对称正定的。

为什么？因为 $\Sigma = \frac{1}{n}X^TX$，其中 $X$ 是数据矩阵（已经中心化）。对任意非零向量 $\mathbf{v}$：

$$\mathbf{v}^T \Sigma \mathbf{v} = \frac{1}{n}\mathbf{v}^T X^T X \mathbf{v} = \frac{1}{n}\|X\mathbf{v}\|^2 > 0$$

$\|X\mathbf{v}\|^2$ 是一个范数的平方，恒大于零。所以 $\Sigma$ 永远是正定的。

Cholesky分解在概率和统计中有一个特别优雅的应用：**从相关到独立。** 如果 $\mathbf{z} \sim \mathcal{N}(0, I)$ 是标准正态分布（各分量独立），那么 $\mathbf{x} = L\mathbf{z}$ 服从 $\mathcal{N}(0, \Sigma)$，因为 $\text{Cov}(L\mathbf{z}) = L \cdot I \cdot L^T = LL^T = \Sigma$。通过 Cholesky分解，你可以把"独立"的标准正态随机变量"编织"成具有任意相关结构的多元正态分布。

**在LLM中**，Transformer的注意力权重矩阵 $W_Q$、$W_K$、$W_V$ 虽然不一定是正定的，但模型中的归一化层、层归一化（LayerNorm）的统计量计算、以及训练过程中梯度的协方差估计，都涉及正定矩阵的运算，底层可能调用 Cholesky 分解来保证数值稳定性。

### Python演示：Cholesky分解

```python
import numpy as np

# 构造一个对称正定矩阵
# 最简单的方法：随机矩阵 × 其转置 + 单位矩阵
np.random.seed(42)
B = np.random.randn(3, 3)
A = B @ B.T + np.eye(3)  # 加单位矩阵确保正定性

print("A (对称正定矩阵):")
print(np.round(A, 4))
print(f"\nA 的特征值（全正才是正定）: {np.round(np.linalg.eigvalsh(A), 4)}")

# Cholesky 分解
L = np.linalg.cholesky(A)

print("\nL (下三角矩阵):")
print(np.round(L, 4))
print(f"\n验证 L @ L^T ≈ A:")
print(np.round(L @ L.T, 4))
print(f"残差: {np.round(np.linalg.norm(A - L @ L.T), 12)}")

# 用 Cholesky 分解采样多元正态分布
mean = np.zeros(3)
n_samples = 10000
z = np.random.randn(n_samples, 3)  # 标准正态
x = z @ L.T  # 变换后的样本服从 N(0, A)

print(f"\n采样协方差矩阵:")
print(np.round(np.cov(x, rowvar=False), 4))
print(f"原始矩阵 A:")
print(np.round(A, 4))
print("两者应该非常接近")
```

你会看到采样的协方差矩阵和原始矩阵 $A$ 非常接近——10000个样本足以精确还原相关结构。

---

## 5.5 矩阵分解在LLM中的角色

### 权重矩阵的内部结构

在大语言模型中，知识被存储在大量的权重矩阵里。一个 Transformer 层的核心操作是：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q = XW_Q$，$K = XW_K$，$V = XW_V$，$X$ 是输入，$W_Q, W_K, W_V$ 是可学习的权重矩阵。一个大型模型的每一层都有多个这样的权重矩阵，总计参数量可达数十亿甚至千亿。

这些权重矩阵并不是随机初始化后保持不变的。经过大量数据的训练，它们内部形成了复杂的结构——某些行高度相关，某些列几乎线性依赖，有效秩远小于矩阵的物理尺寸。

矩阵分解给了我们一种"透视"这些结构的工具。如果你对权重矩阵 $W$ 做分解，你可能会发现：

- $W$ 的有效秩 $r$ 远小于其维度——这意味着大部分参数其实是"冗余"的
- $W$ 中只有少数几个"重要"的成分承载了关键知识，其余成分的作用微乎其微
- 不同的权重矩阵有不同的结构特征——有些更"稀疏"，有些更"低秩"

### 模型压缩的直觉

理解了矩阵的分解结构，压缩模型的直觉就呼之欲出了：

**保留重要的分解部分，丢弃不重要的。**

这就像MP3压缩音乐。一首歌的原始波形数据非常庞大，但经过傅里叶变换分解成不同频率的成分后，你会发现人耳对某些频率极其不敏感。丢弃这些"不重要"的频率分量，文件大小可以缩减到原来的十分之一，但听感几乎没有变化。

矩阵压缩的思路类似。如果你把一个权重矩阵 $W$ 分解成：

$$W = \sigma_1 \mathbf{u}_1\mathbf{v}_1^T + \sigma_2 \mathbf{u}_2\mathbf{v}_2^T + \cdots + \sigma_r \mathbf{u}_r\mathbf{v}_r^T$$

其中 $\sigma_1 \gg \sigma_2 \gg \cdots \gg \sigma_r$（奇异值按重要性递减），那么保留前 $k$ 项、丢弃后 $r-k$ 项，就得到了 $W$ 的一个低秩近似：

$$W_k = \sigma_1 \mathbf{u}_1\mathbf{v}_1^T + \cdots + \sigma_k \mathbf{u}_k\mathbf{v}_k^T$$

参数量从 $m \times n$ 降到 $k(m + n + 1)$。当 $k \ll \min(m, n)$ 时，压缩比非常可观。

这个思想直接催生了**LoRA**（Low-Rank Adaptation）等高效微调技术。LoRA 的核心假设是：微调过程中权重矩阵的变化量 $\Delta W$ 是低秩的，不需要学习一个完整的 $m \times n$ 矩阵，只需要学习两个小矩阵 $B$ 和 $A$（$B$ 是 $m \times r$，$A$ 是 $r \times n$，$r \ll \min(m, n)$），让 $\Delta W = BA$。这样，需要训练的参数从 $m \times n$ 降到了 $r(m + n)$，减少了几个数量级。

### 为下一章做铺垫

这一章我们学了三种"计算工具型"分解：LU、QR、Cholesky。它们解决的是"如何高效、稳定地计算"的问题。

下一章我们要进入更深的水域——**特征值分解**和**奇异值分解（SVD）**。如果说LU、QR、Cholesky是在"拆机器"，那特征值分解和SVD就是在"解读机器的灵魂"。它们揭示的不只是矩阵的计算结构，而是矩阵的**本质信息**——哪些方向重要，哪些方向不重要，矩阵的"能量"分布在哪些维度上。

SVD 尤其关键。它是矩阵分解的"瑞士军刀"，适用于任意矩阵（不管方阵还是矩形，不管满秩还是亏秩），并且在 LLM 的量化、压缩、加速中无处不在。理解了SVD，你就理解了现代AI模型压缩的数学基础。

---

## 5.6 Python实战

让我们用一个完整的例子，演示这一章学到的所有分解方法，并可视化分解前后矩阵的结构。

```python
import numpy as np
from scipy.linalg import lu, cholesky
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# ==========================================
# 构造三种类型的矩阵
# ==========================================

# 1. 一般方阵（用于LU分解）
A_general = np.array([[2, 1, 1, 0],
                       [4, 3, 3, 1],
                       [8, 7, 9, 5],
                       [6, 7, 9, 8]], dtype=float)

# 2. 矩形矩阵（用于QR分解）
A_rect = np.random.randn(6, 3)

# 3. 对称正定矩阵（用于Cholesky分解）
B = np.random.randn(4, 4)
A_spd = B @ B.T + 3 * np.eye(4)

# ==========================================
# LU分解
# ==========================================
print("=" * 50)
print("LU 分解")
print("=" * 50)
P, L_lu, U_lu = lu(A_general)
print(f"残差 ||PA - LU|| = {np.linalg.norm(P @ A_general - L_lu @ U_lu):.2e}")
print(f"L 的条件数: {np.linalg.cond(L_lu):.2f}")
print(f"U 的条件数: {np.linalg.cond(U_lu):.2f}")
print(f"A 的条件数: {np.linalg.cond(A_general):.2f}")

# ==========================================
# QR分解
# ==========================================
print("\n" + "=" * 50)
print("QR 分解")
print("=" * 50)
Q, R_qr = np.linalg.qr(A_rect, mode='reduced')
print(f"残差 ||A - QR|| = {np.linalg.norm(A_rect - Q @ R_qr):.2e}")
print(f"Q^TQ 和单位矩阵的偏差: {np.linalg.norm(Q.T @ Q - np.eye(3)):.2e}")

# ==========================================
# Cholesky分解
# ==========================================
print("\n" + "=" * 50)
print("Cholesky 分解")
print("=" * 50)
L_chol = np.linalg.cholesky(A_spd)
print(f"残差 ||A - LL^T|| = {np.linalg.norm(A_spd - L_chol @ L_chol.T):.2e}")
print(f"L 的对角线元素（全正）: {np.round(np.diag(L_chol), 4)}")

# ==========================================
# 可视化：分解前后矩阵的热力图
# ==========================================
fig, axes = plt.subplots(3, 3, figsize=(15, 13))

titles = [
    ['A (一般方阵)', 'L (下三角)', 'U (上三角)'],
    ['A (矩形矩阵)', 'Q (正交矩阵)', 'R (上三角)'],
    ['A (对称正定)', 'L (下三角)', r'$L^T$ (上三角)'],
]

matrices = [
    [A_general, L_lu, U_lu],
    [A_rect, Q, R_qr],
    [A_spd, L_chol, L_chol.T],
]

for i in range(3):
    for j in range(3):
        ax = axes[i][j]
        im = ax.imshow(matrices[i][j], cmap='RdBu_r', aspect='auto')
        ax.set_title(titles[i][j], fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('矩阵分解可视化：三种分解方法对比', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('/root/gitbook/ch05/matrix_decomposition.png', dpi=150, bbox_inches='tight')
print("\n可视化已保存到 /root/gitbook/ch05/matrix_decomposition.png")

# ==========================================
# 彩蛋：观察分解的"信息压缩"效果
# ==========================================
print("\n" + "=" * 50)
print("信息压缩效果：对比存储开销")
print("=" * 50)
m, n = A_general.shape
print(f"原矩阵 A: {m}x{n} = {m*n} 个元素")
print(f"LU分解: L({m}x{n}) + U({m}x{n}) = {m*n*2} 个元素")
print(f"  但 L 和 U 都是三角矩阵，实际非零元素约 {m*(m+1)//2 + m*(m+1)//2} 个")
print(f"  对称正定矩阵的Cholesky: 只需存储 L = {m*(m+1)//2} 个元素")
print(f"  压缩比: {m*n} -> {m*(m+1)//2} ({m*(m+1)//2/(m*n)*100:.1f}%)")
```

运行这段代码，你会看到三种分解方法的数值验证（残差都接近机器精度 $10^{-15}$），以及一张热力图，直观展示了每个分解把原始矩阵"拆"成了什么样的部件。注意看下三角矩阵 $L$ 的上半部分全是零，上三角矩阵 $U$ 的下半部分全是零——这就是三角矩阵"简单"的来源。

最后的信息压缩统计揭示了一个重要的事实：一个 $4 \times 4$ 的对称正定矩阵有16个元素，但 Cholesky 分解只需要存储10个元素（$L$ 的非零部分）。当矩阵维度变大时，这种压缩效果会更加显著。对于 $1000 \times 1000$ 的对称矩阵，原始元素有100万个，但 $L$ 只需要约50万个——减半。这只是一个开始，下一章我们会看到 SVD 如何实现更激进的压缩。

---

**小结。** 这一章我们学习了三种基本的矩阵分解方法。LU分解把矩阵拆成"下三角 × 上三角"，让解方程变成两次简单的替换操作；QR分解把矩阵拆成"正交 × 上三角"，让最小二乘问题变得数值稳定；Cholesky分解是对称正定矩阵的专属工具，效率更高、存储更省。这三种分解是数值计算的基石，也是理解LLM内部权重结构和模型压缩技术的第一步。下一章，我们将进入特征值分解和SVD——矩阵分解的"终极武器"。
