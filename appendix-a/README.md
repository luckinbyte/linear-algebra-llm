# 附录 A：核心概念地图与速览

本书反复强调一个观点：**矩阵就是线性变换**。下面的地图，把全书最重要的几个"度量"概念，统一挂到"线性变换"这一个根上——它们各自回答的是同一个问题的不同侧面：*这个变换，到底对空间做了什么？*

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 7))
ax.axis('off')

# 中心节点：线性变换
ax.text(0.5, 0.85, '线性变换 $A\\mathbf{x}$', ha='center', va='center',
        fontsize=15, bbox=dict(boxstyle='round', facecolor='#ffd966', edgecolor='black'))

# 五个度量概念
nodes = [
    (0.10, 0.40, '点积',         '向量间关系\n(相似度)'),
    (0.30, 0.15, '行列式',       '体积缩放\n(det=0→压扁)'),
    (0.50, 0.40, '秩',           '输出空间维数\n(几维能活下来)'),
    (0.70, 0.15, '特征值',       '只拉伸的方向\n及倍数'),
    (0.90, 0.40, '奇异值',       '最大拉伸量\n(任意矩阵)'),
]
for x, y, name, desc in nodes:
    ax.text(x, y, f'{name}\n\n{desc}', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#cfe2f3', edgecolor='gray'))
    ax.annotate('', xy=(x, y+0.07), xytext=(0.5, 0.80),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))

ax.set_title('一张图：所有核心概念都是"线性变换"的一个侧面', fontsize=13)
plt.tight_layout()
plt.savefig('concept_map.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 核心概念速览表

| 概念 | 一句话含义 | 几何图像 | 一行代码 | 易错点 |
|------|-----------|----------|----------|--------|
| 点积 | 向量在另一向量方向上的分量大小 | 投影影子 | `a @ b` | 点积大≠向量长，是方向一致 |
| 行列式 | 变换对体积/面积的缩放倍数 | 单位方块→平行四边形 | `np.linalg.det(A)` | 只对方阵有定义；为0=压扁 |
| 秩 | 变换后输出空间的维数 | 3D 空间被压成平面 | `np.linalg.matrix_rank(A)` | 秩≠非零元素个数 |
| 线性相关 | 某向量落在其余张成的空间里 | 第三向量落在前两者平面内 | — | 2D 中任意 3 向量必相关 |
| 特征值/向量 | 变换中只拉伸、不旋转的方向及倍数 | 圆→椭圆，长短轴 | `np.linalg.eig(A)` | 非方阵/旋转矩阵可能无实特征值 |
| 奇异值 | 沿各正交方向的拉伸量（排序） | 圆→椭圆，半轴=σ | `np.linalg.svd(A)` | 对任意矩阵都成立；恒非负 |

> 遇到任何一个概念想不起来"它到底是什么意思"，就回这张图、这张表——它们都是同一个变换的不同侧面。
