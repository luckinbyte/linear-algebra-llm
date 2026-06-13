# 第7章 奇异值分解（SVD）——从推荐系统到低秩近似

> 上一章我们学会了特征值分解（EVD），它像一把手术刀，把方阵解剖成特征向量构成的"骨架"。但现实世界从不规规矩矩地给你一个方阵——你的数据矩阵往往是 1000 行 200 列的"矩形"，既不对称也不是方的。EVD 对此束手无策。而 SVD 说：**管你什么形状，我都能拆。**

## 7.1 SVD 是什么：矩阵的"终极 X 光"

### 从 EVD 的局限说起

上一章我们学过，特征值分解 $A = Q\Lambda Q^{-1}$ 要求矩阵 $A$ 是方阵。但在实际应用中，你面对的矩阵几乎都不是方的：

- 一个推荐系统里，用户-电影评分矩阵是 10000（用户）x 500（电影）
- 一张 1080p 灰度图是 1080 x 1920
- Transformer 的权重矩阵 $W_{ff}$ 是 4096 x 11008

这些矩阵统统不是方阵，EVD 直接罢工。更糟糕的是，即使你有一个方阵，如果它不可对角化（比如上一章提到的 Jordan 块），EVD 也无能为力。

**奇异值分解（Singular Value Decomposition, SVD）**就是为了解决这个痛点而生的。它有一个令人震撼的性质：

**任意矩阵——不管是不是方阵、不管元素是正是负、不管秩是多少——都可以做 SVD。**

没有例外。没有前提条件。这就是为什么数学家们把它称为"线性代数中最优美的定理之一"。

### SVD 的数学表述

对于任意 $m \times n$ 的实数矩阵 $A$，存在分解：

$$A = U \Sigma V^T$$

其中：
- $U$ 是 $m \times m$ 的正交矩阵（满足 $U^TU = I$）
- $\Sigma$ 是 $m \times n$ 的"对角"矩阵（只有对角线上有值，其余全为 0）
- $V$ 是 $n \times n$ 的正交矩阵（满足 $V^TV = I$）

对角矩阵 $\Sigma$ 上的元素 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$（$r = \text{rank}(A)$）叫做**奇异值**（singular values），它们从大到小排列。

### "三步操作"的直觉

如果矩阵 $A$ 代表一个线性变换，那 SVD 把这个变换拆成了三步：

$$A\mathbf{x} = U(\Sigma(V^T\mathbf{x}))$$

1. **$V^T\mathbf{x}$**：在输入空间做一次旋转（或反射）
2. **$\Sigma(\cdot)$**：沿坐标轴做缩放（每个轴缩放的比例就是对应的奇异值）
3. **$U(\cdot)$**：在输出空间做一次旋转（或反射）

类比：想象你在寄一个快递。第一步，把物品摆放到标准姿势（$V^T$ 旋转）；第二步，按不同方向压缩到合适大小（$\Sigma$ 缩放）；第三步，转成收件人期望的朝向（$U$ 旋转）。任何形状的物品，都能通过这三步完成打包。

这比 EVD 优雅太多了——EVD 只能做到 $Q\Lambda Q^{-1}$，而 $Q^{-1}$ 不一定是正交矩阵，所以那三步操作并不都是"旋转"。SVD 的三步全都是正交变换（纯旋转/反射）加缩放，几何意义更加清晰。

### 与 EVD 的关系

SVD 和 EVD 之间有一个精妙的数学联系。如果你对 $A^TA$ 做 EVD：

$$A^TA = (U\Sigma V^T)^T(U\Sigma V^T) = V\Sigma^T\Sigma V^T = V\Sigma^2 V^T$$

看！$V$ 就是 $A^TA$ 的特征向量矩阵，而 $\sigma_i^2$ 就是 $A^TA$ 的特征值。同理，$U$ 是 $AA^T$ 的特征向量矩阵。所以 SVD 本质上是把 $A^TA$ 和 $AA^T$ 这两个对称矩阵的 EVD 统一到了一个框架里。

```python
import numpy as np

# 任意矩阵——不需要方阵！
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

# SVD 分解
U, sigma, Vt = np.linalg.svd(A, full_matrices=True)

print(f"A 的形状: {A.shape}")
print(f"U 的形状: {U.shape}")
print(f"Σ 的形状: {sigma.shape}（一维数组，对角线元素）")
print(f"V^T 的形状: {Vt.shape}")
print(f"\n奇异值: {sigma}")

# 重构验证
Sigma = np.zeros_like(A, dtype=float)
for i in range(len(sigma)):
    Sigma[i, i] = sigma[i]
A_reconstructed = U @ Sigma @ Vt
print(f"\n重构误差: {np.linalg.norm(A - A_reconstructed):.2e}")
```

## 7.2 SVD 的直觉理解

### 三个矩阵各自的含义

让我们逐一理解 $U$、$\Sigma$、$V^T$ 分别代表什么。

**$U$：输出空间的"坐标轴"。** $U$ 的每一列 $\mathbf{u}_i$ 是 $m$ 维输出空间中的一个正交方向。这些方向是 $AA^T$ 的特征向量——也就是说，它们是数据经过 $A$ 变换后，在输出空间中最"活跃"的方向。

**$\Sigma$：每个坐标轴的"重要程度"。** 奇异值 $\sigma_1 \geq \sigma_2 \geq \cdots$ 按从大到小排列，第一个奇异值最大，最后一个（如果有非零的话）最小。$\sigma_i$ 越大，说明沿着 $\mathbf{u}_i$ 和 $\mathbf{v}_i$ 这对方向，矩阵传递的信息量越多。

**$V^T$：输入空间的"坐标轴"。** $V$ 的每一列 $\mathbf{v}_i$ 是 $n$ 维输入空间中的一个正交方向。它们是 $A^TA$ 的特征向量——输入空间中数据变化最显著的方向。

用一个公式可以把 SVD 写成"秩一矩阵之和"的形式：

$$A = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T + \sigma_2 \mathbf{u}_2 \mathbf{v}_2^T + \cdots + \sigma_r \mathbf{u}_r \mathbf{v}_r^T$$

每一项 $\sigma_i \mathbf{u}_i \mathbf{v}_i^T$ 都是一个秩一矩阵（因为它是列向量乘行向量），而 $\sigma_i$ 就是这一项的"权重"。奇异值从大到小排列，意味着**第一项最重要，第二项次之，越往后越不重要**。

### 推荐系统的类比

这是 SVD 最经典的直觉。

想象 Netflix 有 10000 个用户和 5000 部电影。每个用户对看过的电影打 1-5 分，构成一个 $10000 \times 5000$ 的评分矩阵 $R$。这个矩阵极度稀疏——大多数人只看过几十部电影。但如果我们对 $R$ 做 SVD：

- **$V$ 的每一列**（对应 $\mathbf{v}_i$）代表一种**电影特征**。比如 $\mathbf{v}_1$ 可能代表"动作片 vs 文艺片"这个维度，$\mathbf{v}_2$ 可能代表"喜剧 vs 悲剧"。当然，这些维度是 SVD 自动发现的，不一定能用人类语言描述——但它们确实捕捉了电影之间的潜在关联。

- **$U$ 的每一列**（对应 $\mathbf{u}_i$）代表一种**用户偏好**。$\mathbf{u}_1$ 对应的用户偏好维度，恰好与 $\mathbf{v}_1$ 对应的电影特征维度相匹配。

- **$\sigma_i$** 代表第 $i$ 组"用户偏好-电影特征"对的**关联强度**。如果 $\sigma_1$ 很大，说明第一个偏好-特征维度对评分的影响很大。

而那些 $\sigma_i$ 接近 0 的维度，代表"几乎不影响评分的微弱因素"——这就是我们可以丢弃的部分。保留前 $k$ 个奇异值，我们就能用一个紧凑的模型预测用户对没看过的电影的评分。

这就是 2006 年 Netflix Prize 竞赛中夺冠算法的核心思想，也是今天推荐系统（协同过滤）的数学基础。

## 7.3 低秩近似：SVD 的超能力

### 截断 SVD

前面我们写了 SVD 的完整展开：

$$A = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T + \sigma_2 \mathbf{u}_2 \mathbf{v}_2^T + \cdots + \sigma_r \mathbf{u}_r \mathbf{v}_r^T$$

如果只保留前 $k$ 项，丢弃后面的 $r - k$ 项，就得到**截断 SVD**（Truncated SVD）：

$$A_k = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T + \cdots + \sigma_k \mathbf{u}_k \mathbf{v}_k^T$$

$A_k$ 是一个秩为 $k$ 的矩阵，是 $A$ 的近似。

为什么这样做合理？因为奇异值从大到小排列，后面那些被丢弃的项对应的 $\sigma_i$ 很小，对 $A$ 的贡献微乎其微。就像 JPEG 图片压缩保留低频分量、丢弃高频细节一样——人的眼睛对高频细节不敏感，丢了也看不太出来。

### Eckart-Young 定理：最优低秩近似

截断 SVD 不仅"听起来合理"，它还有一个坚实的数学保证——**Eckart-Young-Mirsky 定理**：

在所有秩不超过 $k$ 的矩阵中，$A_k$（截断 SVD 的结果）是 $A$ 在 Frobenius 范数下的**最优近似**：

$$\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_r^2}$$

而且，这个近似误差恰好等于被丢弃的奇异值的平方和的平方根。

**没有其他秩为 $k$ 的矩阵能比 $A_k$ 更好地近似 $A$。** 这就是 SVD 的超能力——它给了你一个理论上最优的压缩方案。

类比：如果矩阵是一本厚达 1000 页的百科全书，低秩近似就像是用 50 页的"精华摘要"来替代它。Eckart-Young 定理告诉你：SVD 给你的是所有 50 页摘要中**信息损失最小**的那一份。

### Python 演示：对图像做 SVD 压缩

```python
import numpy as np
import matplotlib.pyplot as plt

# 构造一张有意义的灰度图（渐变 + 图案）
x = np.linspace(0, 2 * np.pi, 200)
y = np.linspace(0, 2 * np.pi, 300)
X, Y = np.meshgrid(x, y)
image = np.sin(X) * np.cos(Y) + 0.5 * np.sin(3 * X + Y) + 0.3 * np.cos(X - 2 * Y)

# SVD 分解
U, sigma, Vt = np.linalg.svd(image, full_matrices=False)

print(f"原图大小: {image.shape}")
print(f"奇异值个数: {len(sigma)}")
print(f"前 5 个奇异值: {sigma[:5]}")
print(f"奇异值总和: {sigma.sum():.2f}")
print(f"前 10 个奇异值占比: {sigma[:10].sum() / sigma.sum() * 100:.1f}%")
print(f"前 50 个奇异值占比: {sigma[:50].sum() / sigma.sum() * 100:.1f}%")

# 不同 k 值的压缩效果
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

k_values = [1, 5, 10, 30, 100, 300]
for idx, k in enumerate(k_values):
    # 截断 SVD 重构
    compressed = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
    
    # 计算压缩比和误差
    original_size = image.shape[0] * image.shape[1]
    compressed_size = k * (image.shape[0] + image.shape[1] + 1)
    ratio = compressed_size / original_size
    error = np.linalg.norm(image - compressed, 'fro')
    
    axes[idx].imshow(compressed, cmap='gray', vmin=image.min(), vmax=image.max())
    axes[idx].set_title(f'k={k}, 压缩比={ratio:.2f}, 误差={error:.2f}')
    axes[idx].axis('off')

plt.suptitle('截断 SVD 图像压缩：不同 k 值的效果', fontsize=14)
plt.tight_layout()
plt.savefig('svd_image_compression.png', dpi=150, bbox_inches='tight')
plt.show()
```

你会看到：当 $k = 1$ 时图像模糊得只剩一个轮廓，$k = 10$ 时基本结构已经清晰，$k = 50$ 时人眼几乎看不出和原图的差别。而原图有 $300 \times 200 = 60000$ 个数值，$k = 50$ 的截断 SVD 只需要存储 $50 \times (300 + 200 + 1) = 25050$ 个数值——不到原始数据的一半。

## 7.4 SVD 与第 4 章（秩）的连接

### 秩 = 非零奇异值的个数

还记得第 4 章我们说"秩是矩阵中独立信息的数量"吗？SVD 给了这句话一个精确的数学表达：

$$\text{rank}(A) = \text{非零奇异值的个数}$$

就这么简单。如果一个 $100 \times 100$ 的矩阵只有 3 个非零奇异值，那它的秩就是 3——不管它表面上有多少个非零元素。

```python
import numpy as np

# 构造一个"看起来复杂但其实很简单"的矩阵
u = np.array([1, 2, 3, 4, 5])
v = np.array([1, 0, -1, 2, 1])
A = np.outer(u, v)  # 秩一矩阵

print("矩阵 A:")
print(A)
print(f"非零元素个数: {np.count_nonzero(A)}")
print(f"秩: {np.linalg.matrix_rank(A)}")

U, sigma, Vt = np.linalg.svd(A)
print(f"奇异值: {sigma}")
print(f"非零奇异值个数: {np.count_nonzero(sigma > 1e-10)}")
```

这个 $5 \times 5$ 的矩阵有 20 个非零元素（第二列全为零），但只有一个非零奇异值——秩为 1。那些"额外"的非零元素都是冗余的，它们只是同一行信息的线性组合。

### 秩一矩阵 = SVD 的一个"原子"

回顾 SVD 的展开式：

$$A = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T + \sigma_2 \mathbf{u}_2 \mathbf{v}_2^T + \cdots + \sigma_r \mathbf{u}_r \mathbf{v}_r^T$$

每一项 $\sigma_i \mathbf{u}_i \mathbf{v}_i^T$ 都是秩一矩阵。$\mathbf{u}_i \mathbf{v}_i^T$ 是一个 $m \times n$ 矩阵，它的每一行都是 $\mathbf{v}_i^T$ 的倍数（倍数由 $\mathbf{u}_i$ 的对应元素决定），所以秩只能是 1。

**这就是第 4 章承诺的"原子拆解"。** 第 4 章我们说：秩一矩阵是矩阵世界的"原子"，任意矩阵都可以拆成秩一矩阵的加权和。SVD 不仅实现了这个拆解，而且做到了最优——它按照重要性排序，最重要的"原子"排在最前面。

用一个比喻：如果说矩阵是一篇文章，那秩一矩阵就是其中的句子。有些句子是中心论点（大奇异值对应的项），有些句子是细节补充（小奇异值对应的项），还有些句子纯属废话（零奇异值对应的项，根本不该存在）。SVD 不但帮你把文章拆成了一句一句的，还按重要性排好了序。

### 数值秩：现实世界的模糊地带

在理论中，奇异值要么非零要么是零。但在计算机里，由于浮点误差，很多本该是零的奇异值会变成极小的正数（比如 $10^{-15}$）。这就引入了**数值秩**的概念——如果你设定一个阈值 $\epsilon$，把所有小于 $\epsilon$ 的奇异值当成零，那"数值秩"就是大于 $\epsilon$ 的奇异值的个数。

```python
# 一个"几乎秩亏"的矩阵
A = np.array([[1.0, 2.0],
              [1.0001, 2.0003]])

_, sigma, _ = np.linalg.svd(A)
print(f"奇异值: {sigma}")
print(f"理论秩: 2")
print(f"数值秩 (eps=1e-4): {np.sum(sigma > 1e-4)}")
print(f"数值秩 (eps=1e-5): {np.sum(sigma > 1e-5)}")
```

这个矩阵的第二个奇异值很小（约 $3 \times 10^{-5}$），说明第二行几乎和第一行线性相关。在不同的阈值下，它可能被认为是秩 2（精确数学）或秩 1（数值实践）。在 LLM 训练中，这种"几乎秩亏"的权重矩阵正是低秩近似（如 LoRA）可以大显身手的场景。

## 7.5 SVD 在 LLM 中的应用

### 模型压缩：对权重矩阵做 SVD

一个 LLM 有数十亿甚至数千亿个参数。这些参数中很多是"冗余"的——它们对模型的输出影响很小，可以被压缩掉。

SVD 提供了一个自然的压缩方案。假设 Transformer 中有一个权重矩阵 $W \in \mathbb{R}^{m \times n}$，我们对它做 SVD：

$$W = U\Sigma V^T \approx U_k \Sigma_k V_k^T$$

其中 $U_k$ 取 $U$ 的前 $k$ 列，$\Sigma_k$ 取前 $k$ 个奇异值，$V_k^T$ 取 $V^T$ 的前 $k$ 行。

原始矩阵需要存储 $m \times n$ 个参数。截断后需要存储 $U_k$（$m \times k$）、$\Sigma_k$（$k$）、$V_k^T$（$k \times n$），共 $k(m + n + 1)$ 个参数。当 $k \ll \min(m, n)$ 时，压缩比非常可观。

举个例子：GPT-2 的 FFN 层有一个 $768 \times 3072$ 的权重矩阵。如果奇异值衰减很快（经验上确实如此），我们可能只需要 $k = 128$ 就能保留 95% 以上的信息量。存储量从 $768 \times 3072 = 2,359,296$ 降到 $128 \times (768 + 3072 + 1) = 491,648$——压缩了近 5 倍。

### 知识蒸馏的数学基础

知识蒸馏（Knowledge Distillation）是把大模型的知识转移到小模型中。从 SVD 的角度看，大模型的权重矩阵包含了大量信息，但其中很多对应小奇异值的维度是"噪声"或"冗余知识"。截断 SVD 恰好提取了大模型中最重要的知识成分（大奇异值对应的子空间），丢弃了不重要的噪声（小奇异值对应的子空间）。

最近的研究（如 2023 年的 GPT-Zip 等工作）直接用截断 SVD 压缩 LLM 的权重，发现在压缩 3-5 倍的情况下，模型在下游任务上的性能下降不到 2%。这证实了 LLM 的权重矩阵确实存在显著的低秩结构。

### 为 LoRA 铺垫：低秩意味着用小矩阵近似大矩阵

如果你已经听说过 **LoRA**（Low-Rank Adaptation），那么你已经接触过 SVD 的思想了，只是可能还不知道。

LoRA 的核心假设是：虽然预训练的权重矩阵 $W_0$ 很大（比如 $4096 \times 4096$），但**微调时的权重变化量** $\Delta W$ 是低秩的。也就是说，$\Delta W$ 可以用两个小矩阵的乘积来近似：

$$\Delta W = BA$$

其中 $B \in \mathbb{R}^{m \times r}$，$A \in \mathbb{R}^{r \times n}$，且 $r \ll \min(m, n)$。

为什么这个假设成立？因为微调只是在预训练的基础上做小幅调整——新的知识不需要用完整的 $m \times n$ 个自由度来表达。如果对 $\Delta W$ 做 SVD，你会发现只有前几个奇异值比较大，后面的迅速衰减到接近零。

从 SVD 的角度看，LoRA 本质上就是在说：**微调的过程只改变了权重矩阵在奇异值最大的几个方向上的分量。** 微调结束时学到的 $\Delta W$，近似等于截断 SVD 中被保留的那部分。

这就是为什么 LoRA 能用 0.1% 的参数量达到全参数微调 90% 以上的效果——因为那 99.9% 的参数本来就在做"无用功"，SVD 告诉我们它们对应的奇异值微乎其微。

我们会在第 12 章深入展开 LoRA 的细节。但此刻你应该已经有了坚实的数学基础：**低秩近似 = 截断 SVD = 保留重要信息 + 丢弃冗余**。

## 7.6 Python 实战

### 对模拟权重矩阵做 SVD 并分析

```python
import numpy as np
import matplotlib.pyplot as plt

# 模拟一个 LLM FFN 层的权重矩阵
np.random.seed(42)
m, n = 512, 2048

# 构造一个"有低秩结构"的权重矩阵
# 真实 LLM 的权重矩阵通常有这种性质
true_rank = 50
U_true = np.random.randn(m, true_rank)
V_true = np.random.randn(true_rank, n)
W = U_true @ V_true + 0.1 * np.random.randn(m, n)  # 低秩部分 + 小噪声

print(f"权重矩阵 W 的形状: {W.shape}")
print(f"W 的数值秩: {np.linalg.matrix_rank(W)}")

# SVD 分解
U, sigma, Vt = np.linalg.svd(W, full_matrices=False)

print(f"\n奇异值统计:")
print(f"  总数: {len(sigma)}")
print(f"  最大值: {sigma[0]:.2f}")
print(f"  最小值: {sigma[-1]:.2f}")
print(f"  前 10 个: {sigma[:10].round(2)}")
```

### 不同截断程度的近似效果

```python
# 计算不同 k 值下的重构误差和信息保留率
total_energy = np.sum(sigma ** 2)
k_values = [5, 10, 20, 50, 100, 200, 500]

print(f"{'k':>5} | {'累积能量占比':>12} | {'Frobenius误差':>14} | {'相对误差':>10}")
print("-" * 55)

for k in k_values:
    # 截断 SVD
    W_k = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
    error = np.linalg.norm(W - W_k, 'fro')
    relative_error = error / np.linalg.norm(W, 'fro')
    energy_ratio = np.sum(sigma[:k] ** 2) / total_energy
    
    print(f"{k:5d} | {energy_ratio:12.4%} | {error:14.2f} | {relative_error:10.4%}")
```

### 可视化奇异值的分布

这是实践中最重要的诊断工具之一——看奇异值的"衰减曲线"。

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1：奇异值本身（线性尺度）
axes[0].plot(range(1, len(sigma) + 1), sigma, 'b-', linewidth=1)
axes[0].set_xlabel('奇异值编号 i')
axes[0].set_ylabel(r'$\sigma_i$')
axes[0].set_title('奇异值分布（线性尺度）')
axes[0].grid(True, alpha=0.3)

# 图2：奇异值（对数尺度）——能更清楚看到衰减
axes[1].semilogy(range(1, len(sigma) + 1), sigma, 'r-', linewidth=1)
axes[1].set_xlabel('奇异值编号 i')
axes[1].set_ylabel(r'$\sigma_i$ (log scale)')
axes[1].set_title('奇异值分布（对数尺度）')
axes[1].grid(True, alpha=0.3)

# 图3：累计能量占比——决定保留多少奇异值
cumulative_energy = np.cumsum(sigma ** 2) / total_energy
axes[2].plot(range(1, len(sigma) + 1), cumulative_energy, 'g-', linewidth=1.5)
axes[2].axhline(y=0.95, color='r', linestyle='--', label='95% 能量')
axes[2].axhline(y=0.99, color='orange', linestyle='--', label='99% 能量')
axes[2].set_xlabel('保留的奇异值个数 k')
axes[2].set_ylabel('累计能量占比')
axes[2].set_title('累计能量占比')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# 找到保留 95% 和 99% 能量需要的 k
k95 = np.argmax(cumulative_energy >= 0.95) + 1
k99 = np.argmax(cumulative_energy >= 0.99) + 1
axes[2].axvline(x=k95, color='r', linestyle=':', alpha=0.5)
axes[2].axvline(x=k99, color='orange', linestyle=':', alpha=0.5)
axes[2].annotate(f'k={k95} (95%)', xy=(k95, 0.95), fontsize=10)
axes[2].annotate(f'k={k99} (99%)', xy=(k99, 0.99), fontsize=10)

plt.tight_layout()
plt.savefig('svd_spectrum_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n关键发现:")
print(f"  保留 95% 能量只需 k = {k95} / {min(m,n)} ({k95/min(m,n)*100:.1f}%)")
print(f"  保留 99% 能量只需 k = {k99} / {min(m,n)} ({k99/min(m,n)*100:.1f}%)")
print(f"  压缩比 (k={k95}): {(m*n) / (k95*(m+n+1)):.1f}x")
```

运行这段代码，你会看到一条典型的奇异值衰减曲线：前几十个奇异值很大，然后迅速"悬崖式"下降，最后变成几乎为零的长尾巴。这条曲线的形状直接决定了模型的可压缩性——悬崖越陡峭，可以丢弃的维度越多，压缩效果越好。

### 实战：用截断 SVD 压缩一个真实的权重矩阵

```python
# 模拟更真实的场景：对一个"预训练权重"做 SVD 压缩
np.random.seed(2024)

# 构造一个有明确低秩结构的权重矩阵
# 这模拟了 LLM 中常见的模式：大部分信息集中在少数方向
m, n, latent_rank = 1024, 4096, 32

# "信号"部分（低秩）
U_signal = np.linalg.qr(np.random.randn(m, latent_rank))[0]
V_signal = np.linalg.qr(np.random.randn(n, latent_rank))[0]
signal_strengths = np.exp(-np.arange(latent_rank) * 0.3) * 100
W_signal = U_signal @ np.diag(signal_strengths) @ V_signal.T

# "噪声"部分（小幅度）
W_noise = np.random.randn(m, n) * 0.5

W_total = W_signal + W_noise

# SVD 分析
U, sigma, Vt = np.linalg.svd(W_total, full_matrices=False)
total_energy = np.sum(sigma ** 2)
cum_energy = np.cumsum(sigma ** 2) / total_energy

# 找到"拐点"——奇异值从信号主导变成噪声主导的位置
# 经验方法：看 sigma 的对数图中的"拐弯处"
log_sigma = np.log10(sigma)
diff_log_sigma = np.diff(log_sigma)
# 拐点处二阶差分最大
inflection = np.argmax(np.abs(np.diff(diff_log_sigma))) + 2

print(f"权重矩阵: {m} x {n}")
print(f"真实潜在秩: {latent_rank}")
print(f"SVD 估计的拐点: k ≈ {inflection}")
print(f"保留 95% 能量: k = {np.argmax(cum_energy >= 0.95) + 1}")
print(f"保留 99% 能量: k = {np.argmax(cum_energy >= 0.99) + 1}")

# 不同 k 值的近似误差
for k in [16, 32, 64, 128]:
    W_k = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
    rel_err = np.linalg.norm(W_total - W_k, 'fro') / np.linalg.norm(W_total, 'fro')
    params_original = m * n
    params_compressed = k * (m + n + 1)
    print(f"  k={k:3d}: 相对误差 {rel_err:.4f}, "
          f"参数量 {params_compressed/params_original*100:.1f}% "
          f"(压缩 {params_original/params_compressed:.1f}x)")
```

这段代码模拟了 LLM 权重矩阵的一个核心特征：**有效秩（有效信号对应的奇异值个数）远小于矩阵的维度**。这是 LoRA、模型剪枝、量化等技术的共同数学基础——参数矩阵中真正有用的信息只占一小部分，其余要么是冗余，要么是噪声。

---

## 小结

奇异值分解是线性代数的"瑞士军刀"——它适用于任意矩阵，给出了最优的低秩近似，并且在从推荐系统到大语言模型的广阔领域中都有核心应用：

- **定义**：$A = U\Sigma V^T$，把任意矩阵分解为"输入旋转 + 缩放 + 输出旋转"三步
- **直觉**：$U$ 是输出空间的坐标轴，$\Sigma$ 是各轴的重要性，$V$ 是输入空间的坐标轴
- **低秩近似**：截断 SVD 给出最优的秩 $k$ 近似（Eckart-Young 定理），保留重要信息、丢弃冗余
- **与秩的关系**：秩 = 非零奇异值个数，矩阵 = 秩一矩阵的加权和，这就是第 4 章承诺的"原子拆解"
- **LLM 应用**：权重矩阵的 SVD 压缩、知识蒸馏、LoRA 的数学基础——都建立在"LLM 权重具有低秩结构"这个事实之上

下一章我们将进入概率与统计的世界——看看线性代数如何与不确定性对话，以及为什么高斯分布是连接两者的桥梁。
