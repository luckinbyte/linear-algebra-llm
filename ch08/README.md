# 第8章 张量——从RGB图片到批量推理

你打开手机拍了一张照片。在屏幕上，它是一幅色彩斑斓的画面；在计算机眼里，它是一个三维数字阵列——高 $\times$ 宽 $\times$ 3个颜色通道。这个三维阵列，就是一个张量（Tensor）。

如果你打开的不是一张照片而是一段视频，那它变成了四维阵列——帧数 $\times$ 高 $\times$ 宽 $\times$ 3。如果你把一个 batch 的视频喂给神经网络，恭喜，五维了。

张量这个名字听起来很唬人，但它的本质极其朴素：**就是多维数组**。向量是一维数组，矩阵是二维数组，张量是它们向更高维度的自然推广。在深度学习的世界里，张量是一切运算的载体——模型的输入是张量，权重是张量，中间的激活值是张量，最终的输出也是张量。理解张量，就是理解神经网络如何"看"世界。

## 8.1 什么是张量：矩阵的"升维打击"

### 从标量到张量：维度阶梯

让我们从一个简单的阶梯开始攀登：

| 名称 | 维度 | 形状示例 | 生活类比 |
|------|------|----------|----------|
| 标量（Scalar） | 0D | `()` | 温度：25°C |
| 向量（Vector） | 1D | `(3,)` | RGB颜色值：`(255, 128, 0)` |
| 矩阵（Matrix） | 2D | `(28, 28)` | 灰度图：28×28像素 |
| 张量（Tensor） | 3D | `(224, 224, 3)` | RGB图片 |
| 张量（Tensor） | 4D | `(16, 224, 224, 3)` | 一个batch的RGB图片 |
| 张量（Tensor） | 5D | `(4, 16, 224, 224, 3)` | 多段视频 |

严格来说，标量是0阶张量，向量是1阶张量，矩阵是2阶张量。但在日常使用中，我们习惯把3阶及以上的数组称为"张量"。这就好比"汽车"这个概念——自行车也是车，但没人会把它叫"车"而不加前缀。

数学上，一个 $n$ 阶张量可以表示为：

$$T \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_n}$$

其中每个 $d_i$ 是第 $i$ 个维度的大小。张量中的每一个元素可以用 $n$ 个索引来定位：$T_{i_1, i_2, \ldots, i_n}$。

### RGB图片：你每天都在接触的3D张量

一张 $224 \times 224$ 的 RGB 图片，在计算机中存储为形状 `(224, 224, 3)` 的张量。具体来说：

- 第一个维度（224）：图片的高度，有224行像素
- 第二个维度（224）：图片的宽度，有224列像素
- 第三个维度（3）：三个颜色通道，分别是 R（红）、G（绿）、B（蓝）

所以 `image[100, 50, 0]` 表示第100行、第50列像素的红色通道值，取值范围是 0-255（8位无符号整数）。

```python
import torch

# 模拟一张 224x224 的 RGB 图片
image = torch.randn(224, 224, 3)
print(f"图片形状: {image.shape}")       # torch.Size([224, 224, 3])
print(f"像素(0,0)的RGB值: {image[0, 0]}")  # 三个值
```

### 视频：4D张量

视频是什么？就是一系列连续的图片。一段 30 帧、$224 \times 224$ 分辨率的 RGB 视频：

$$\text{video} \in \mathbb{R}^{30 \times 224 \times 224 \times 3}$$

形状为 `(30, 224, 224, 3)`。如果批量处理 8 段视频，就是 `(8, 30, 224, 224, 3)`——一个五维张量。

### 为什么需要张量

你可能会问：矩阵不够用吗？为什么要搞到三维、四维甚至更高？

因为**世界不是平的**。一个灰度图可以用矩阵表示，但彩色图天然就有通道维度；一个样本可以用向量表示，但一批样本天然就有 batch 维度；一步计算可以用矩阵乘法描述，但多头注意力天然就需要在 head 维度上独立运算。

张量不是一个故弄玄虚的概念，它是自然界多维度信息的忠实表示。你之所以觉得它陌生，只是因为"张量"这个词太数学了，而它的实际含义——多维数组——简单得像呼吸。

## 8.2 张量的基本运算

### 形状（Shape）：最重要的属性

在深度学习中，张量最重要的属性不是它的值，而是它的**形状**（shape）。形状告诉你每个维度有多大，决定了张量之间能进行哪些运算。

```python
x = torch.randn(2, 3, 4)
print(f"形状: {x.shape}")        # torch.Size([2, 3, 4])
print(f"维度数: {x.dim()}")       # 3
print(f"元素总数: {x.numel()}")   # 24 = 2 * 3 * 4
```

深度学习中有一个经验法则：**80%的 bug 都是形状不匹配造成的**。你看到一个报错 `RuntimeError: shape mismatch`，不要慌，逐个检查每一层的输入输出形状，问题总能找到。

### 重塑（Reshape）：换个角度看数据

重塑操作不改变张量中的数据，只改变我们"看待"数据的视角。就像同一堆乐高积木，你可以搭成城堡，也可以拆了搭成飞船——积木本身没变，只是组织方式变了。

```python
x = torch.arange(12)  # [0, 1, 2, ..., 11]
print(f"原始: {x.shape}")          # torch.Size([12])

# 重塑为 3x4 矩阵
y = x.reshape(3, 4)
print(f"重塑为 3x4: {y.shape}")    # torch.Size([3, 4])

# 重塑为 2x2x3 张量
z = x.reshape(2, 2, 3)
print(f"重塑为 2x2x3: {z.shape}")  # torch.Size([2, 2, 3])

# 用 -1 自动推断某个维度
w = x.reshape(3, -1)  # 自动算出第二维是 4
print(f"自动推断: {w.shape}")      # torch.Size([3, 4])
```

重塑的前提是元素总数不变：$2 \times 3 \times 4 = 24$ 个元素，你只能重塑成同样是 24 个元素的形状，比如 `(4, 6)`、`(2, 12)`、`(2, 2, 6)` 等等。

一个极其重要且容易混淆的重塑操作是 `view` 和 `reshape` 的区别：`view` 要求张量在内存中是连续的（contiguous），而 `reshape` 不要求。在实际使用中，用 `reshape` 更安全，性能差异可以忽略。

### 广播（Broadcasting）：小张量的"自动扩展"

广播是张量运算中最优雅的机制之一。当你对一个形状为 `(3, 4)` 的张量加上一个形状为 `(4,)` 的向量时，不需要手动把这个向量复制3次——PyTorch 会自动"广播"它：

$$\begin{pmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \end{pmatrix} + \begin{pmatrix} 100 & 200 & 300 & 400 \end{pmatrix} = \begin{pmatrix} 101 & 202 & 303 & 404 \\ 105 & 206 & 307 & 408 \\ 109 & 210 & 311 & 412 \end{pmatrix}$$

广播的规则很简单：

1. **从右向左**逐维比较两个张量的形状
2. 如果某维大小相同，直接运算
3. 如果某维其中一个为1，该维被扩展为另一个的大小
4. 如果某维大小不同且都不为1，报错

```python
A = torch.ones(3, 4)       # shape: (3, 4)
b = torch.tensor([1, 2, 3, 4])  # shape: (4,)

C = A * b  # 广播！b 被扩展为 (3, 4)
print(f"结果形状: {C.shape}")  # (3, 4)
```

在 Transformer 中，广播无处不在。比如位置编码是形状 `(1, seq_len, dim)` 的张量，输入是形状 `(batch, seq_len, dim)` 的张量，两者相加时，位置编码会自动广播到整个 batch——你不需要为每个样本复制一份位置编码。

### 张量乘法：矩阵乘法的推广

矩阵乘法 $\mathbf{C} = \mathbf{A}\mathbf{B}$ 是深度学习的计算核心。在张量世界，矩阵乘法被推广为**批量矩阵乘法**（batched matmul）：

如果 $\mathbf{A}$ 的形状是 `(batch, m, k)`，$\mathbf{B}$ 的形状是 `(batch, k, n)`，那么：

$$\mathbf{C}[i] = \mathbf{A}[i] \times \mathbf{B}[i], \quad i = 0, 1, \ldots, \text{batch}-1$$

结果 $\mathbf{C}$ 的形状是 `(batch, m, n)`。这不是一个循环，而是一次并行计算——这正是 GPU 擅长的。

```python
# 批量矩阵乘法
A = torch.randn(8, 4, 16)   # (batch, m, k)
B = torch.randn(8, 16, 32)  # (batch, k, n)
C = torch.bmm(A, B)          # 或 A @ B
print(f"结果形状: {C.shape}")  # (8, 4, 32)
```

### 转置与维度交换（Transpose / Permute）

有时候你需要重新排列张量的维度。比如 PyTorch 默认图片格式是 `(C, H, W)`（通道在前），而 matplotlib 显示图片需要 `(H, W, C)`（通道在后）：

```python
image = torch.randn(3, 224, 224)  # CHW 格式

# 方法1：permute 指定新维度顺序
image_hwc = image.permute(1, 2, 0)  # 原来的第1维→第0维, 第2维→第1维, 第0维→第2维
print(f"转换后: {image_hwc.shape}")  # (224, 224, 3)

# 方法2：对于2D张量，可以用 transpose 交换两个维度
matrix = torch.randn(3, 4)
transposed = matrix.transpose(0, 1)
print(f"转置后: {transposed.shape}")  # (4, 3)
```

在 Transformer 的多头注意力中，`permute` 是家常便饭。你需要把 head 维度从隐藏维度中"拆出来"，变成一个独立的维度，然后才能并行计算每个 head 的注意力。

## 8.3 GPU为什么擅长张量运算

### 矩阵乘法 = 大量独立的乘加运算

让我们拆解一个矩阵乘法。计算 $\mathbf{C} = \mathbf{A}\mathbf{B}$，其中 $\mathbf{A} \in \mathbb{R}^{m \times k}$，$\mathbf{B} \in \mathbb{R}^{k \times n}$：

$$C_{ij} = \sum_{l=1}^{k} A_{il} \cdot B_{lj}$$

每一个 $C_{ij}$ 需要 $k$ 次乘法和 $k-1$ 次加法。总共 $m \times n$ 个结果元素，互不依赖——计算 $C_{0,0}$ 不需要等 $C_{0,1}$ 算完。这种"大量独立、结构相同"的运算模式，正是并行计算的完美场景。

### GPU的并行架构：数千个核心同时计算

CPU 和 GPU 的设计哲学截然不同：

| 特性 | CPU | GPU |
|------|-----|-----|
| 核心数 | 8-24个 | 数千到数万个 |
| 每个核心 | 强大、通用 | 简单、专用 |
| 擅长 | 复杂逻辑、分支 | 大规模重复计算 |
| 类比 | 8个博士 | 10000个高中生 |

深度学习的核心运算就是矩阵乘法，而矩阵乘法的本质就是大量重复的"乘加"操作。一个 $(1024 \times 1024) \times (1024 \times 1024)$ 的矩阵乘法需要约 $10^9$ 次乘加运算。CPU 串行执行需要很长时间，GPU 的数千个核心可以并行处理，速度提升几十到上百倍。

### 为什么深度学习推动了GPU产业的发展

2012年，Alex Krizhevsky 用两个 GTX 580 GPU 训练了 AlexNet，在 ImageNet 竞赛中以巨大优势夺冠。这件事向整个AI社区证明了一件事：**GPU + 大数据 + 深度网络 = 惊人的效果**。

此后，NVIDIA 做了一个关键决策——投资 CUDA 生态。CUDA 让研究者可以用 C/C++ 的语法直接调用 GPU 的并行计算能力，而不需要懂图形学。这降低了一大批人使用 GPU 的门槛。

到今天，NVIDIA 的市值很大程度上建立在深度学习对 GPU 的巨大需求上。而 Google 的 TPU、华为的昇腾等芯片，本质上都是在追求同一件事：**更快地做矩阵乘法**。

### CUDA和张量核心（Tensor Core）

传统的 GPU 计算单元每次执行一次乘加操作。NVIDIA 从 Volta 架构（2017年）开始引入**张量核心**（Tensor Core），可以在一个时钟周期内完成一个 $4 \times 4$ 矩阵的乘加运算：

$$\mathbf{D} = \mathbf{A} \times \mathbf{B} + \mathbf{C}$$

其中 $\mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{D}$ 都是 $4 \times 4$ 矩阵。这意味着单次操作完成 128 次乘加。到 H100（2022年）和 B200（2024年），张量核心经过多次迭代，对各种精度（FP16、BF16、FP8、INT8）的支持越来越完善，吞吐量也在持续提升。

这就是为什么现代 GPU 训练大语言模型如此高效——硬件层面，它就是为张量运算量身定制的。

## 8.4 张量在LLM中的角色

如果说前面的内容是"认识张量"，那这一节就是"张量在战场上的实战"。让我们追踪张量在大语言模型中的完整旅程。

### 输入张量：(batch_size, seq_len, hidden_dim)

当你给 LLM 输入一段文本，比如 "Hello world"，模型的第一步是把它变成张量：

1. **分词**：`["Hello", " world"]` → token IDs `[15496, 995]`
2. **查词嵌入表**：每个 token ID 映射为一个 `hidden_dim` 维的向量
3. **组装为输入张量**：形状 `(batch_size, seq_len, hidden_dim)`

```python
batch_size = 4    # 同时处理4个序列
seq_len = 128     # 每个序列128个token
hidden_dim = 768  # 每个token用768维向量表示

input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
print(f"输入张量形状: {input_tensor.shape}")  # (4, 128, 768)
```

这三个维度各有含义：

- **batch_size**（4）：并行处理多少个序列，越大吞吐量越高
- **seq_len**（128）：每个序列有多长，决定了上下文窗口
- **hidden_dim**（768）：每个 token 的表示向量有多"厚"，决定模型的表达能力

### 权重张量的形状

Transformer 中每一层都有参数，它们也是张量。以 GPT-2 Small（768维）为例：

| 参数 | 形状 | 元素数 | 含义 |
|------|------|--------|------|
| Token Embedding | `(50257, 768)` | 38.6M | 每个token的嵌入向量 |
| Position Embedding | `(1024, 768)` | 0.8M | 每个位置的编码 |
| Q/K/V 权重 | `(768, 768)` × 3 | 1.8M | 查询/键/值的线性变换 |
| Attention 输出 | `(768, 768)` | 0.6M | 注意力输出的线性变换 |
| FFN 第一层 | `(768, 3072)` | 2.4M | 前馈网络扩展 |
| FFN 第二层 | `(3072, 768)` | 2.4M | 前馈网络压缩 |
| LayerNorm | `(768,)` × 2 | — | 每层两个，含 γ 和 β |

一个 Transformer 层的总参数量约为 700 万，GPT-2 Small 有 12 层，总共约 1.24 亿参数。

### 多头注意力中的张量操作

这是张量操作最密集的地方。让我们一步步拆解。

输入 $\mathbf{X}$ 形状为 `(batch, seq_len, hidden_dim)`，即 `(B, S, D)`。设 `num_heads = 12`，`head_dim = D / num_heads = 64`。

**第一步：计算 Q、K、V**

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

每一步是 `(B, S, D) × (D, D) → (B, S, D)`。

**第二步：拆分头**

把 `D` 拆成 `num_heads × head_dim`：

$$\mathbf{Q} \to \text{reshape} \to (B, S, H, D_h) \to \text{permute} \to (B, H, S, D_h)$$

其中 $H$ 是 head 数，$D_h$ 是每个 head 的维度。

**第三步：计算注意力分数**

$$\text{scores} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{D_h}}$$

这是 `(B, H, S, D_h) × (B, H, D_h, S) → (B, H, S, S)`。每个 head 独立计算 $S \times S$ 的注意力矩阵。

**第四步：Softmax 和加权求和**

$$\text{attn} = \text{softmax}(\text{scores}) \cdot \mathbf{V}$$

`(B, H, S, S) × (B, H, S, D_h) → (B, H, S, D_h)`

**第五步：合并头**

$$\text{permute} \to (B, S, H, D_h) \to \text{reshape} \to (B, S, D)$$

然后过输出线性层：`(B, S, D) × (D, D) → (B, S, D)`。

```
输入 (B, S, D)
  │
  ├── Q = X @ Wq  (B, S, D)
  ├── K = X @ Wk  (B, S, D)
  └── V = X @ Wv  (B, S, D)
  │
  reshape → (B, S, H, Dh)
  permute → (B, H, S, Dh)
  │
  scores = Q @ K^T  (B, H, S, S)
  attn = softmax(scores) @ V  (B, H, S, Dh)
  │
  permute → (B, S, H, Dh)
  reshape → (B, S, D)
  │
  output = attn @ Wo  (B, S, D)
```

### 一次前向推理中的张量形状变化追踪

让我们完整追踪一个 token 经过 GPT-2 的一层 Transformer：

```python
B, S, D = 2, 64, 768
H, Dh = 12, 64

x = torch.randn(B, S, D)  # 输入

# 自注意力
Q = x @ Wq  # (2, 64, 768) @ (768, 768) → (2, 64, 768)
Q = Q.reshape(B, S, H, Dh).permute(0, 2, 1, 3)  # → (2, 12, 64, 64)
scores = Q @ K.transpose(-2, -1)  # (2, 12, 64, 64) @ (2, 12, 64, 64) → (2, 12, 64, 64)
attn = softmax(scores) @ V  # (2, 12, 64, 64) @ (2, 12, 64, 64) → (2, 12, 64, 64)
attn = attn.permute(0, 2, 1, 3).reshape(B, S, D)  # → (2, 64, 768)
attn_out = attn @ Wo  # (2, 64, 768) @ (768, 768) → (2, 64, 768)

# 残差连接 + LayerNorm
x = layernorm(x + attn_out)  # (2, 64, 768)

# 前馈网络
ff = x @ W1  # (2, 64, 768) @ (768, 3072) → (2, 64, 3072)  扩展4倍
ff = gelu(ff)
ff = ff @ W2  # (2, 64, 3072) @ (3072, 768) → (2, 64, 768)  压缩回来

# 残差连接 + LayerNorm
x = layernorm(x + ff)  # (2, 64, 768) — 形状不变！
```

注意一个关键特征：**张量形状在整个 Transformer 层中保持不变**。输入 `(B, S, D)`，输出还是 `(B, S, D)`。这使得 Transformer 可以像搭积木一样堆叠任意多层。

## 8.5 Python实战：用PyTorch追踪张量形状

理论讲完了，让我们动手写代码，从头模拟一次 Transformer 的前向推理，重点追踪张量形状的变化。

### 创建模拟输入，逐步追踪形状变化

```python
import torch
import torch.nn.functional as F
import math

# ========== 模型超参数 ==========
batch_size = 2
seq_len = 8
hidden_dim = 64
num_heads = 4
head_dim = hidden_dim // num_heads  # 16
ffn_dim = hidden_dim * 4            # 256

# ========== 模拟输入 ==========
# 假设 batch_size=2, seq_len=8, 词表大小=1000
token_ids = torch.randint(0, 1000, (batch_size, seq_len))
print(f"Token IDs 形状: {token_ids.shape}")  # (2, 8)

# 词嵌入
embedding = torch.nn.Embedding(1000, hidden_dim)
x = embedding(token_ids)
print(f"嵌入后形状: {x.shape}")  # (2, 8, 64)

# 位置编码（简化版：随机生成）
pos_encoding = torch.randn(1, seq_len, hidden_dim)
x = x + pos_encoding  # 广播：(2,8,64) + (1,8,64) → (2,8,64)
print(f"加位置编码后: {x.shape}")  # (2, 8, 64)

# ========== 单层 Transformer ==========
# Q, K, V 权重
Wq = torch.randn(hidden_dim, hidden_dim)
Wk = torch.randn(hidden_dim, hidden_dim)
Wv = torch.randn(hidden_dim, hidden_dim)
Wo = torch.randn(hidden_dim, hidden_dim)

# 计算 Q, K, V
Q = x @ Wq  # (2, 8, 64) @ (64, 64) → (2, 8, 64)
K = x @ Wk  # (2, 8, 64)
V = x @ Wv  # (2, 8, 64)
print(f"Q 形状: {Q.shape}")  # (2, 8, 64)

# 拆分多头
Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
K = K.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
V = V.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
print(f"拆分头后 Q 形状: {Q.shape}")  # (2, 4, 8, 16)

# 注意力分数
scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)
print(f"注意力分数形状: {scores.shape}")  # (2, 4, 8, 8)

# Softmax
attn_weights = F.softmax(scores, dim=-1)

# 加权求和
attn_output = attn_weights @ V  # (2, 4, 8, 16)
print(f"注意力输出形状: {attn_output.shape}")  # (2, 4, 8, 16)

# 合并多头
attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)
print(f"合并头后形状: {attn_output.shape}")  # (2, 8, 64)

# 输出线性层
attn_output = attn_output @ Wo  # (2, 8, 64)

# 残差连接
x = x + attn_output

# LayerNorm
ln = torch.nn.LayerNorm(hidden_dim)
x = ln(x)
print(f"LayerNorm 后形状: {x.shape}")  # (2, 8, 64)

# 前馈网络
W1 = torch.randn(hidden_dim, ffn_dim)
W2 = torch.randn(ffn_dim, hidden_dim)

ff = x @ W1         # (2, 8, 64) @ (64, 256) → (2, 8, 256)
ff = F.gelu(ff)     # 激活函数，形状不变
ff = ff @ W2        # (2, 8, 256) @ (256, 64) → (2, 8, 64)
print(f"FFN 输出形状: {ff.shape}")  # (2, 8, 64)

# 残差 + LayerNorm
x = ln(x + ff)
print(f"一层 Transformer 输出形状: {x.shape}")  # (2, 8, 64)
```

运行这段代码，你会看到张量的形状像一条河流，在 Transformer 层中蜿蜒流淌，中间可能分支、变宽、合并，但最终汇入终点时，形状和起点一模一样。

### 理解"形状不匹配"是最常见的bug

当你开始自己实现 Transformer 组件时，你会频繁遇到这样的报错：

```
RuntimeError: Expected size for first two dimensions of batch2 (B) to be [64, 64] but got [64, 16].
```

**调试形状不匹配的策略**：

1. **像刚才那样，在每一步后打印形状**。找到形状第一次"跑偏"的地方
2. **最常见的错误**：忘记 `permute`/`transpose`，或者 `reshape` 的维度算错了
3. **黄金法则**：在纸上画出形状的变化，然后在代码中逐步验证

```python
# 常见错误：reshape 时维度顺序搞错
Q_wrong = Q.reshape(batch_size, seq_len, num_heads, head_dim)
# 这样 reshape 后直接做矩阵乘法，语义是错的
# 正确的做法是先 reshape 再 permute，把 head 维度放到 batch 后面

Q_right = Q.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
# permute 之后，head 维度在第二位，seq_len 在第三位
# 这样矩阵乘法才是"每个 head 独立地做 seq_len × head_dim 的运算"
```

### einsum：用爱因斯坦求和约定简化张量运算

当你熟悉了基本的张量操作后，一定会遇到 `torch.einsum`。它基于爱因斯坦求和约定，用一个简洁的字符串描述复杂的张量运算。

**核心规则**：出现在箭头右侧但不在左侧的下标，会被求和消除。

```python
# 矩阵乘法：C_ij = sum_k A_ik * B_kj
C = torch.einsum('ik,kj->ij', A, B)

# 批量矩阵乘法：C_bij = sum_k A_bik * B_bkj
C = torch.einsum('bik,bkj->bij', A, B)

# 多头注意力的 QK^T：scores_bhij = sum_d Q_bhid * K_bhjd
scores = torch.einsum('bhid,bhjd->bhij', Q, K)

# 等价于:
# scores = Q @ K.transpose(-2, -1)

# 更复杂的：同时做注意力计算和加权求和
# output_bhie = sum_j (softmax(scores_bhij) * V_bhje)
output = torch.einsum('bhij,bhje->bhie', attn_weights, V)
```

`einsum` 的优势在于：

- **可读性**：一眼看出哪些维度在参与运算，哪些被求和了
- **安全性**：维度名字必须匹配，减少了"维度顺序搞错"这类 bug
- **通用性**：一个函数搞定转置、矩阵乘法、批量运算、内积等几乎所有操作

在阅读 Transformer 论文和代码时，你会越来越频繁地看到 `einsum`。它看起来像天书，但一旦习惯了，你会发现它是最精确的张量运算语言。

---

**本章小结**：张量就是多维数组。它的形状（shape）是最重要的属性——搞清形状，就搞清了运算。在 Transformer 中，张量的形状变化就像一条河流：从输入 `(batch, seq_len, hidden_dim)` 出发，经过注意力的拆头、计算、合并，再经过前馈网络的扩展和压缩，最终以相同的形状汇入下一层。GPU 之所以是深度学习的核心硬件，正是因为张量运算（尤其是矩阵乘法）天生适合并行计算。掌握张量，你就掌握了理解神经网络内部运作的万能钥匙。
