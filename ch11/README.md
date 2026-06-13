# 第11章 Softmax与概率矩阵

## 11.1 从"排名"到"概率"：Softmax在做什么

想象你是一位老师，刚批完一场考试。五个学生的原始分数分别是：

| 学生 | 原始分 |
|------|--------|
| 小明 | 85 |
| 小红 | 72 |
| 小刚 | 90 |
| 小李 | 65 |
| 小王 | 78 |

如果你想把这些分数变成"相对实力占比"——也就是说，把每个人的分数换算成一个百分比，让所有百分比加起来刚好等于 100%——你会怎么做？

最直觉的做法是：每个人的分数除以总分。

$$p_i = \frac{x_i}{\sum_j x_j}$$

这看起来很合理。85 / (85+72+90+65+78) = 85/390 ≈ 21.8%。

但问题来了：**如果原始分数里有负数怎么办？**

在神经网络的输出中，这是常态。一个全连接层吐出来的 logits 可能是 [-3.2, 1.5, 0.8, -0.1]，里面有正有负。你没法对一个负值做"除以总和"来得到概率——概率不能是负数。

更糟糕的是，就算全是正数，如果总和恰好是零呢？（虽然在实数域不太可能，但理论上你没法保证。）

所以我们需要一个函数，它能把**任意一组实数**映射成一个合法的概率分布。这个函数就是 Softmax。

它的核心思路极其朴素：**先把每个数变成正数，再做归一化。**

怎么把任意实数变成正数？指数函数 $e^x$ 是最自然的选择——不管你给它什么实数，吐出来的永远是正数。

---

## 11.2 Softmax的数学定义

Softmax 函数的定义如下：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

对向量 $\mathbf{x} = [x_1, x_2, \dots, x_n]$ 中的每一个元素 $x_i$，计算 $e^{x_i}$，然后除以所有 $e^{x_j}$ 的总和。

让我们用上面的 logits 验证一下：

$$\mathbf{x} = [-3.2, \; 1.5, \; 0.8, \; -0.1]$$

先算指数：

$$e^{-3.2} \approx 0.041, \quad e^{1.5} \approx 4.482, \quad e^{0.8} \approx 2.225, \quad e^{-0.1} \approx 0.905$$

总和：$0.041 + 4.482 + 2.225 + 0.905 = 7.653$

概率分布：

$$p \approx [0.005, \; 0.586, \; 0.291, \; 0.118]$$

验证：$0.005 + 0.586 + 0.291 + 0.118 = 1.000$ ✓

每个值都在 0 和 1 之间，总和恰好为 1。完美。

### 为什么用 $e^x$？

三个理由：

1. **单调递增**：$e^x$ 严格单调递增，所以输入的大小关系在输出中被完美保留——原来排第一的，softmax 之后还是排第一。
2. **处处可微**：$e^x$ 的导数就是 $e^x$ 本身，反向传播时极其方便。
3. **放大差异**：$e^x$ 的增长是超线性的。比如 1.5 和 0.8 的差距是 0.7，但 $e^{1.5}$ 和 $e^{0.8}$ 的比值约是 2:1。Softmax 会让强者更强、弱者更弱——这正是我们想要的"聚焦"效果。

### 温度参数

Softmax 有一个重要的变体，引入温度参数 $T$：

$$\text{softmax}(x_i / T) = \frac{e^{x_i / T}}{\sum_{j} e^{x_j / T}}$$

温度 $T$ 控制输出分布的"尖锐度"：

- **$T \to 0$**：趋向 one-hot 分布。最大的那个值会独占几乎全部概率，其他值趋近于零。这表示模型"非常确定"自己的选择。
- **$T \to \infty$**：趋向均匀分布。所有值的概率趋于相等，即 $1/n$。这表示模型"完全不确定"。
- **$T = 1$**：标准 softmax。

在 LLM 的语境里，温度直接控制"创造力"。低温生成保守、确定的文本；高温生成多样、出人意料（但可能不连贯）的文本。

用前面的例子，当 $T = 0.5$ 时：

$$\mathbf{x}/0.5 = [-6.4, \; 3.0, \; 1.6, \; -0.2]$$

差异被放大了，原本 0.586 的最大概率会变得更加突出。当 $T = 2.0$ 时，$\mathbf{x}/2.0 = [-1.6, \; 0.75, \; 0.4, \; -0.05]$，差异被缩小，分布变得更平坦。

---

## 11.3 Softmax的几何直觉

从几何的角度看 Softmax，它在做一件非常优雅的事情。

首先，指数函数 $e^x$ 把实数轴 $\mathbb{R}$ 映射到正半轴 $\mathbb{R}_{>0}$。然后，除以总和这一步是投影——它把一个 $n$ 维正象限中的点，投影到一个特殊的几何体上。

这个几何体叫做**单纯形**（simplex）。

$n$ 维单纯形的定义是：

$$\Delta^{n-1} = \left\{ \mathbf{p} \in \mathbb{R}^n \;\middle|\; p_i \geq 0, \; \sum_{i=1}^{n} p_i = 1 \right\}$$

换句话说，所有分量非负且总和为 1 的向量构成的集合。这恰好是概率分布的数学定义。

### 二维可视化

考虑 $n = 3$ 的情况（三个类别的分类）。输入是三维空间中的一个点，经过 softmax 之后，输出落在三维空间中的一块三角形上——这个三角形的三个顶点是 $(1,0,0)$、$(0,1,0)$ 和 $(0,0,1)$。

对于 $n = 2$ 的情况就更容易想象了。输入 $(x_1, x_2)$ 是二维平面上的一个点，经过 softmax 之后：

$$p_1 = \frac{e^{x_1}}{e^{x_1} + e^{x_2}}, \quad p_2 = \frac{e^{x_2}}{e^{x_1} + e^{x_2}}$$

输出 $(p_1, p_2)$ 永远满足 $p_1 + p_2 = 1$，也就是说它落在从 $(1, 0)$ 到 $(0, 1)$ 的线段上。Softmax 把整个二维平面压缩到了一条一维线段上。

这个映射有一个有趣的性质：它保持了顺序关系。如果 $x_1 > x_2$，那么 $p_1 > p_2$。但距离关系没有被保持——softmax 会扭曲距离，而且扭曲的方式是非线性的。

---

## 11.4 数值稳定性：LogSumExp技巧

Softmax 的公式看起来简单，但在实际编程中有一个大坑：**溢出**。

假设你的 logits 是 $[1000, \; 1001, \; 999]$。直接算 $e^{1000}$ 会怎样？

大多数编程语言会返回 `inf`（无穷大）。然后你拿 `inf` 除以 `inf`，得到 `NaN`。程序崩溃。

这不是理论上的极端案例。在深层网络中，未归一化的 logits 很容易达到几百甚至上千的量级。

### 解法：减去最大值

关键观察：给所有 $x_i$ 同时加上或减去同一个常数 $c$，softmax 的结果不变。

$$\frac{e^{x_i + c}}{\sum_j e^{x_j + c}} = \frac{e^{x_i} \cdot e^c}{\sum_j e^{x_j} \cdot e^c} = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

$e^c$ 在分子分母中同时出现，直接约掉了。

所以我们可以放心地令 $c = -\max(\mathbf{x})$，把最大值平移到零：

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(\mathbf{x})}}{\sum_j e^{x_j - \max(\mathbf{x})}}$$

现在最大的指数变成 $e^0 = 1$，其他都小于 1。完全没有溢出风险。

这个技巧有一个名字，叫做 **LogSumExp 技巧**。在计算 $\log \sum_i e^{x_i}$（这个量在很多地方出现，比如交叉熵损失函数）时，同样的 trick 适用：

$$\log \sum_i e^{x_i} = \max(\mathbf{x}) + \log \sum_i e^{x_i - \max(\mathbf{x})}$$

这不是一个可选的优化，而是工程实践中**必须处理**的问题。任何声称"从零实现 softmax"但没加这个的代码，都是不正确的。

---

## 11.5 采样策略：从概率到选择

Softmax 给你一个概率分布，但概率分布本身不是选择。要让模型"说"出下一个词，你需要从概率分布中**采样**——也就是根据概率随机挑一个。

怎么挑？看似简单，实际上大有学问。

### 贪心采样（Greedy）

每次选概率最高的那个。

简单粗暴，但它有一个致命缺点：生成的文本高度确定性。同一个 prompt 永远得到同样的输出。更重要的是，它容易陷入重复循环——"我很开心我很开心我很开心"。

### 温度采样（Temperature Sampling）

就是前面说的温度参数。$T < 1$ 让分布更尖锐（趋近贪心），$T > 1$ 让分布更平坦（更多样）。$T = 1$ 就是原始 softmax 分布。

在实践中，温度通常设为 0.7 到 1.0 之间，这是一个平衡"合理性"和"多样性"的甜蜜区间。

### Top-K 采样

只在概率最高的 $K$ 个候选中采样，把其余的全部置零。

比如 $K = 50$，意思是只考虑概率排名前 50 的词。这能有效过滤掉那些概率极低但偶尔被采到的"胡言乱语"。

数学上说，就是先把概率向量中除了前 $K$ 大之外的所有元素设为 0，然后重新归一化。这是一个截断操作。

### Top-P 采样（核采样，Nucleus Sampling）

Top-K 的问题在于 $K$ 是固定的。有时候前 10 个词就占了 95% 的概率，有时候前 100 个才占 60%。

Top-P 解决了这个问题。它设定一个累积概率阈值 $P$（通常 0.9 或 0.95），然后把概率从大到小排序、依次累加，直到总和超过 $P$ 为止。只保留这些词，其余置零，再重新归一化。

这样，当分布"尖锐"时（少数词占大头），候选集很小；当分布"平坦"时（概率分散在大量词上），候选集自动扩大。

### 线性代数视角

所有这些采样策略，本质上都是对概率向量 $\mathbf{p}$ 做变换：

| 策略 | 变换 |
|------|------|
| 温度 | $\mathbf{p} = \text{softmax}(\mathbf{x} / T)$，在 softmax 之前修改输入 |
| Top-K | $\mathbf{p} \leftarrow \text{topK}(\mathbf{p})$ 然后重新归一化，是一个线性截断 + 缩放 |
| Top-P | $\mathbf{p} \leftarrow \text{topP}(\mathbf{p})$ 然后重新归一化，同上 |

它们都是"修改概率分布的形状"，然后再从修改后的分布中采样。在实际系统中，这些策略常常组合使用——比如同时设温度 0.8 和 Top-P 0.9。

---

## 11.6 Softmax在LLM中的三重角色

Softmax 在 Transformer 架构中出现了不止一次，而是三次，扮演三个不同的角色。理解这三重角色，是理解 LLM 内部运作的关键。

### 角色一：注意力分数的归一化

在第 9 章我们看到了注意力机制的核心计算：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

这里的 softmax 作用在注意力分数矩阵上。对于每个 query，softmax 把它与所有 key 的点积分数归一化为一个概率分布。这个分布决定了：在当前这个位置，应该给序列中其他每个位置多少"注意力"。

注意这里 softmax 的维度——它是沿着 key 的维度（即矩阵的每一行）做 softmax，而不是对整个矩阵做。每一行独立归一化。

### 角色二：输出层的词概率分布

在 Transformer 的最后一层，有一个线性变换把隐藏状态投影到词表大小 $V$ 的维度上，产生 logits 向量 $\mathbf{z} \in \mathbb{R}^V$。然后 softmax 把它变成概率：

$$P(w_t | w_{<t}) = \text{softmax}(\mathbf{z})$$

这个概率分布覆盖整个词表，告诉我们在给定上下文的情况下，每个词作为下一个词的可能性。

$V$ 的大小通常是 3 万到 15 万。所以这里的 softmax 是在非常大的向量上操作的，计算量不小。这也是为什么有各种近似 softmax（如分层 softmax、负采样等）的研究。

### 角色三：门控机制

在某些高级架构中，softmax 被用作"选择器"。

最典型的例子是混合专家模型（Mixture of Experts, MoE）。在 MoE 层中，有多个并行的前馈网络（称为"专家"），softmax 被用来计算每个专家的权重：

$$\text{gate}(\mathbf{x}) = \text{softmax}(\mathbf{W}_g \mathbf{x})$$

然后只选择概率最高的 $K$ 个专家（通常 $K = 1$ 或 $2$）来处理输入，其余专家被跳过。这大大减少了计算量，同时保持了模型容量。

三重角色，同一个函数。Softmax 的本质就是：**把一组任意实数变成一个"选择权重"**——无论选择的是注意力位置、词表中的词、还是专家网络。

---

## 11.7 Python实战

下面我们从零实现 Softmax 以及相关的采样策略，并演示它们的效果。

```python
import numpy as np

# --- 1. 数值稳定的 Softmax ---

def softmax(logits, temperature=1.0):
    """
    数值稳定的 softmax 实现。
    
    参数:
        logits: 形状为 (n,) 或 (batch, n) 的 numpy 数组
        temperature: 温度参数，默认 1.0
    返回:
        概率分布，形状与 logits 相同
    """
    # 温度缩放
    scaled = logits / temperature
    
    # 数值稳定性：减去最大值（关键步骤！）
    # axis=-1 表示沿最后一个轴操作，keepdims 保持维度
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    
    # 计算指数
    exp_scores = np.exp(shifted)
    
    # 归一化
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


# 验证：基本功能
logits = np.array([-3.2, 1.5, 0.8, -0.1])
probs = softmax(logits)
print("概率分布:", probs)
print("总和:", np.sum(probs))
# 输出: 概率分布: [0.005  0.586  0.291  0.118]
#       总和: 1.0

# 验证：数值稳定性（没有减最大值会溢出的情况）
big_logits = np.array([1000.0, 1001.0, 999.0])
print("大数 softmax:", softmax(big_logits))
# 输出: 大数 softmax: [0.245  0.665  0.090]  —— 没有溢出！


# --- 2. 温度参数的效果 ---

logits = np.array([1.0, 2.0, 3.0, 4.0])

for T in [0.1, 0.5, 1.0, 2.0, 5.0]:
    probs = softmax(logits, temperature=T)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    print(f"T={T:4.1f}: {probs}  熵={entropy:.3f}")
# T= 0.1: [0.000  0.000  0.000  1.000]  熵≈0    (几乎确定)
# T= 0.5: [0.002  0.016  0.117  0.865]  熵≈0.455 (偏确定)
# T= 1.0: [0.032  0.087  0.237  0.644]  熵≈0.948 (标准)
# T= 2.0: [0.102  0.167  0.276  0.455]  熵≈1.245 (偏均匀)
# T= 5.0: [0.181  0.221  0.270  0.329]  熵≈1.362 (接近均匀)


# --- 3. Top-K 采样 ---

def top_k_sample(logits, k=5, temperature=1.0):
    """
    Top-K 采样：只在概率最高的 K 个中选。
    """
    probs = softmax(logits, temperature)
    
    # 找到排名前 K 的索引
    top_k_indices = np.argsort(probs)[-k:]
    
    # 把不在 Top-K 中的概率设为 0
    mask = np.zeros_like(probs)
    mask[top_k_indices] = 1.0
    filtered_probs = probs * mask
    
    # 重新归一化
    filtered_probs /= np.sum(filtered_probs)
    
    return filtered_probs

logits = np.array([0.1, 0.5, 2.0, 3.0, 1.0, 0.3, 0.01, 0.8])
print("\nTop-K=3:", top_k_sample(logits, k=3))
# 只有排名前 3 的位置有非零概率


# --- 4. Top-P（核采样） ---

def top_p_sample(logits, p=0.9, temperature=1.0):
    """
    Top-P（核采样）：累积概率达到 P 就停。
    """
    probs = softmax(logits, temperature)
    
    # 按概率从大到小排序
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # 累积概率
    cumulative = np.cumsum(sorted_probs)
    
    # 找到累积概率刚超过 p 的位置
    cutoff = np.searchsorted(cumulative, p) + 1
    
    # 只保留前 cutoff 个
    mask = np.zeros_like(probs)
    mask[sorted_indices[:cutoff]] = 1.0
    filtered_probs = probs * mask
    
    # 重新归一化
    filtered_probs /= np.sum(filtered_probs)
    
    return filtered_probs

logits = np.array([0.1, 0.5, 2.0, 3.0, 1.0, 0.3, 0.01, 0.8])
print("\nTop-P=0.9:", top_p_sample(logits, p=0.9))
# 累积概率达到 90% 后，剩余词被过滤掉


# --- 5. 完整的生成采样函数 ---

def generate_token(logits, temperature=0.8, top_k=50, top_p=0.9):
    """
    组合策略：温度 + Top-K + Top-P，从 logits 中采样一个 token。
    这接近 GPT 类模型实际使用的采样流程。
    """
    # Step 1: 温度缩放后的 softmax
    probs = softmax(logits, temperature)
    
    # Step 2: Top-K 过滤
    if top_k is not None and top_k < len(probs):
        top_k_indices = np.argsort(probs)[-top_k:]
        mask = np.zeros_like(probs)
        mask[top_k_indices] = 1.0
        probs = probs * mask
    
    # Step 3: Top-P 过滤
    if top_p is not None:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative, top_p) + 1
        keep_indices = sorted_indices[:cutoff]
        mask = np.zeros_like(probs)
        mask[keep_indices] = 1.0
        probs = probs * mask
    
    # Step 4: 重新归一化
    probs = probs / np.sum(probs)
    
    # Step 5: 按概率采样
    return np.random.choice(len(probs), p=probs)

# 模拟一个小型词表上的采样
np.random.seed(42)
fake_logits = np.random.randn(1000)  # 假设有 1000 个词
token_id = generate_token(fake_logits, temperature=0.8, top_k=50, top_p=0.9)
print(f"\n采样得到的 token ID: {token_id}")
```

这段代码涵盖了 Softmax 的完整工程实现，以及从温度调节到 Top-K、Top-P 采样的全流程。注意几个要点：

1. **减最大值不是可选的**——没有它，大数会直接溢出。
2. **温度在 softmax 之前应用**——它是修改输入，不是修改输出。
3. **Top-K 和 Top-P 是后处理**——它们直接修改概率分布，然后重新归一化。
4. **`np.random.choice(p=...)`** 是最终的采样步骤——它根据概率分布随机选一个元素。

---

## 小结

Softmax 做的事情看起来简单——把一组数变成概率。但这个简单的操作在 LLM 中无处不在：

- 它把注意力分数变成注意力权重
- 它把输出层的 logits 变成下一个词的概率分布
- 它在门控机制中充当"路由选择器"

而围绕 softmax 的采样策略——温度、Top-K、Top-P——直接决定了 LLM 生成文本的风格和多样性。理解了 softmax，你就理解了 LLM "从思考到说话"的最后一公里。

下一章我们将探索低秩近似与 LoRA，看看如何用极少的参数高效微调大语言模型——矩阵分解的数学原理与工程实践。
