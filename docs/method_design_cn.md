# 方法设计文档（中文底稿）

## 1. 问题定义与目标

本文关注的问题不是全局场预测，也不是规则网格上的算子学习，而是一个**局部 meshfree shape function 生成问题**。给定查询点周围的一小块点云 patch，我们希望直接输出该 patch 上每个邻居节点在查询点处的 shape function 值。

形式化地，目标映射可以写成：

$$
\mathcal{G}_{\theta}:\left(\mathbf{x}_q,\{\mathbf{X}_i\}_{i=1}^{k},\beta,\rho_q\right)\mapsto \{\phi_i(\mathbf{x}_q)\}_{i=1}^{k}.
$$

其中：

- $\mathbf{x}_q \in \mathbb{R}^d$：查询点。
- $\mathbf{X}_i \in \mathbb{R}^d$：查询点局部支撑域中的第 $i$ 个邻居节点。
- $k$：局部 patch 的邻居个数。当前实现固定为 $k=16$。
- $\beta \in \mathbb{R}^+$：teacher max-ent 参考解中的尺度/热力学参数。
- $\rho_q$：查询点处的局部几何上下文特征。
- $\phi_i(\mathbf{x}_q)$：第 $i$ 个邻居节点在查询点 $\mathbf{x}_q$ 处的 shape function 值。

本工作要学到的不是一个“节点坐标到标量”的普通回归器，而是一个满足数值结构约束的**局部权函数生成器**。它需要同时兼顾以下目标：

1. 与参考 max-ent shape function 尽量接近。
2. 保持局部性，即只在给定 patch 上起作用。
3. 保证 partition of unity。
4. 保证一阶线性再现性。
5. 对局部点云几何具有稳定响应，而不是依赖固定网格拓扑。

从力学与数值分析的角度看，这个问题的意义在于：shape function 是 meshfree 离散中的基本构件。若能快速近似生成局部 shape function，就可以为大变形 meshfree 分析提供快速初始化、warm start，甚至直接生成近似可用的局部离散权重。

---

## 2. 方法总览

整个方法被刻意拆成两个职责不同的部分：

- **Local Geometric Encoder / Backbone**：负责从局部几何中学习“哪些节点更重要、节点间如何相互影响”。
- **Structure-Preserving Output Head**：负责把 backbone 输出的原始 logits 转成满足数值结构约束的 shape function。

总流程可以表示为：

$$
\left(\mathbf{x}_q,\{\mathbf{X}_i\},\beta,\rho_q\right)
\xrightarrow{\text{feature construction}}
\{\mathbf{f}_i\}
\xrightarrow{\text{backbone}}
\{l_i\}
\xrightarrow{\text{B1--B4 head}}
\{\phi_i(\mathbf{x}_q)\}.
$$

对应的数据流可以用 ASCII 图表示：

```text
patch sample
  ├── query point x_q
  ├── neighbor nodes X
  └── beta
        ↓
feature construction
  ├── query-centered relative coordinates
  ├── scale normalization
  └── node features + rho_q
        ↓
backbone
  ├── kernel_operator (primary)
  └── mlp_baseline (control)
        ↓
raw logits l_i
        ↓
structure-preserving head
  ├── B1: softplus
  ├── B2: window modulation
  ├── B3: normalization
  └── B4: reproducing correction
        ↓
phi_corr
```

这种分工的核心思想是：**“表达能力”和“结构合法性”分离**。backbone 不直接承担所有数值约束；它只学习一个尽可能好的基底权重排序。真正的结构合法性由 output head 强制构造出来。这样做的好处是：

- backbone 可以专注于几何编码和局部模式学习；
- 数值约束不依赖训练过程“慢慢学出来”，而是由架构直接保证；
- 不同 backbone 可以在同一个结构头下公平比较。

---

## 3. 局部几何编码

### 3.1 Query-centered normalization

对每个邻居节点，以查询点为参考原点构造相对坐标：

$$
\mathbf{r}_i=\mathbf{X}_i-\mathbf{x}_q,\qquad i=1,\dots,k.
$$

这一步的目的是去掉平移自由度。对于局部 shape function 而言，真正重要的是邻居节点相对于查询点的几何结构，而不是绝对坐标。

### 3.2 Scale normalization

当前实现使用 patch 内最远邻居距离作为尺度：

$$
r_{\max}=\max_{i}\|\mathbf{r}_i\|,\qquad
\hat{\mathbf{r}}_i=\frac{\mathbf{r}_i}{r_{\max}}.
$$

这一步的动机是把不同 patch 统一到无量纲坐标系中，使网络处理的是“相对几何形状”，而不是具体长度尺度。

需要强调的是，当前实现已经显式区分了归一化尺度与窗口支撑半径。`r_{\max}` 继续只承担 backbone 输入归一化的职责；而在 output head 的 B2 中，实际使用的是

$$
r_{\mathrm{support}}=\alpha r_{\max},\qquad \alpha>1.
$$

当前代码中的默认值为 `support_radius_scale = 1.05`。这样做的直接原因是：若把紧支撑窗口的边界半径硬绑在 $r_{\max}$ 上，则最远邻居会系统性落在支撑边界上并被压成零；而把窗口半径略微放大后，可以保留同样的无量纲归一化方式，同时避免这一硬截断效应。

从数值角度看，这相当于把局部支撑域压缩到一个标准化半径附近，从而提高：

- 不同 patch 之间的可比性；
- 网络训练的稳定性；
- 窗口函数和 kernel aggregation 的尺度一致性。

### 3.3 节点特征构造

当前最小特征模式为：

$$
\mathbf{f}_i =
\left[
\hat{\mathbf{r}}_i,\;
\|\hat{\mathbf{r}}_i\|,\;
\beta
\right].
$$

在二维情况下，这一最小特征的维度是 4：

- 两个归一化相对坐标分量；
- 一个归一化距离；
- 一个 $\beta$ 标量。

在增强模式下，还会额外引入局部几何上下文 $\rho_q$。当前实现中的 $\rho_q$ 包括：

1. 平均最近邻距离（归一化后）。
2. 当前 patch 节点数占 $k_{\max}$ 的比例。
3. 局部协方差特征值比 $\lambda_{\min}/\lambda_{\max}$。

因此增强特征为：

$$
\mathbf{f}_i =
\left[
\hat{\mathbf{r}}_i,\;
\|\hat{\mathbf{r}}_i\|,\;
\beta,\;
\rho_q
\right].
$$

它的目的不是让网络显式记住某个 patch 类型，而是把局部密度与各向异性等低维几何统计直接提供给模型，减轻 backbone 自己从原始点云中反推这些量的负担。

### 3.4 几何意义与力学意义

从力学与近似理论角度看，这些特征并非任意堆叠：

- $\hat{\mathbf{r}}_i$ 对应邻居在 query-centered 局部参考系中的位置；
- $\|\hat{\mathbf{r}}_i\|$ 对应支撑域中的相对半径位置；
- $\beta$ 控制 reference max-ent 解的扩散/集中特性；
- $\rho_q$ 则刻画局部点云是否稠密、是否各向异性、是否接近退化几何。

这类特征组合本质上是在逼近这样一个观点：局部 shape function 的主要决定因素，不是全局问题规模，而是**局部归一化几何 + 局部尺度参数 + 局部几何统计**。

---

## 4. Backbone 设计

### 4.1 Kernel-Integral Operator Backbone

这是当前方法的主 backbone。它的思想是：让每个节点通过局部 kernel-like aggregation 感受到其他节点的影响，从而得到对局部 patch 几何更敏感的表示。

首先对节点特征做 lifting：

$$
\mathbf{h}_i^{(0)}=\mathrm{MLP}_{\text{lift}}(\mathbf{f}_i).
$$

之后在每一层中，对节点对 $(i,j)$ 构造 pair feature：

$$
\mathbf{g}_{ij}^{(\ell)}=
\left[
\mathbf{h}_i^{(\ell)}-\mathbf{h}_j^{(\ell)},
\hat{\mathbf{r}}_i,
\hat{\mathbf{r}}_j,
\|\hat{\mathbf{r}}_i-\hat{\mathbf{r}}_j\|
\right].
$$

再通过一个小型 modulation network 输出标量调制项：

$$
m_{ij}^{(\ell)}=\mathrm{softplus}\!\left(
\mathrm{MLP}_{\text{mod}}^{(\ell)}(\mathbf{g}_{ij}^{(\ell)})
\right).
$$

基础 kernel 使用紧支撑的 Wendland C2：

$$
k_{ij}^{(\ell)} = w_{\text{Wendland}}\!\left(
\frac{\|\hat{\mathbf{r}}_i-\hat{\mathbf{r}}_j\|}{\gamma}
\right),
$$

其中 $\gamma$ 是 kernel radius scale。

于是 pairwise aggregation 权重为：

$$
\alpha_{ij}^{(\ell)} = m_{ij}^{(\ell)}\,k_{ij}^{(\ell)},
\qquad
\bar{\alpha}_{ij}^{(\ell)}=
\frac{\alpha_{ij}^{(\ell)}}
{\sum_{j=1}^{k}\alpha_{ij}^{(\ell)}+\varepsilon}.
$$

节点信息聚合为：

$$
\mathbf{m}_i^{(\ell)}=
\sum_{j=1}^{k}\bar{\alpha}_{ij}^{(\ell)}\,
\mathbf{V}^{(\ell)}\mathbf{h}_j^{(\ell)}.
$$

最后更新：

$$
\mathbf{h}_i^{(\ell+1)}=
\mathbf{h}_i^{(\ell)}+
\mathrm{GELU}\!\left(
\mathbf{U}^{(\ell)}\mathbf{h}_i^{(\ell)}+\mathbf{m}_i^{(\ell)}
\right).
$$

最终输出 raw logit：

$$
l_i=\mathbf{w}_{\text{out}}^\top \mathbf{h}_i^{(L)}+b_{\text{out}}.
$$

#### 设计动机

这个 backbone 的设计思路贴近 meshfree 数值方法的语言习惯：

- 邻居作用是局部的；
- 节点间相互作用随距离衰减；
- 作用大小既受几何位置影响，也受节点表征差异影响；
- 聚合过程应保持对 patch 内部节点排列顺序的不敏感。

因此它不是简单的注意力，也不是简单图网络，而是一个更接近“带学习调制的局部核积分算子”的结构。

#### 力学/数值视角

在 meshfree 中，局部支撑域上的函数近似天然带有 kernel weighting、support truncation 和局部重构的味道。`kernel_operator` 的意义在于：

- 用局部核衰减表达“远处节点贡献较弱”的数值先验；
- 用可学习调制表达“同样的距离，不同局部几何下贡献可能不同”；
- 用层叠聚合表达 patch 内多节点关系，而不是只看 query-to-node 的单边关系。

这使它比平铺 MLP 更有机会学到真正与局部几何结构相关的模式。

### 4.2 MLP Baseline

作为最小对照模型，MLP baseline 做法更直接：先按查询点距离对邻居排序，再把所有节点特征拉平成一个向量送入多层感知机。

记按距离排序后的节点特征为：

$$
\mathbf{f}_{\pi(1)},\dots,\mathbf{f}_{\pi(k)},
$$

则输入为：

$$
\mathbf{z}=
\left[
\mathbf{f}_{\pi(1)},
\mathbf{f}_{\pi(2)},
\dots,
\mathbf{f}_{\pi(k)}
\right].
$$

输出为：

$$
\mathbf{l}=\mathrm{MLP}(\mathbf{z})\in\mathbb{R}^{k}.
$$

#### 设计动机

这个 baseline 的目的不是追求最优，而是回答一个核心问题：

> 若只给一个固定维度的普通 MLP，它能否仅靠排序后的局部特征学到足够好的 shape function 近似？

如果不能，而 `kernel_operator` 可以，就说明几何归纳偏置确实有价值。

#### 局限性

它依赖固定 $k$，而且对排序策略敏感，因此不适合作为最终方法，只适合作为低基准线。

---

## 5. Structure-Preserving Head（B1--B4）

本方法最关键的贡献不只是 backbone，而是一个把 raw logits 转成结构合法 shape function 的 output head。整个 pipeline 是：

$$
\{l_i\}
\xrightarrow{\text{B1}}
\{a_i\}
\xrightarrow{\text{B2}}
\{\tilde{a}_i\}
\xrightarrow{\text{B3}}
\{\phi_i^{\text{base}}\}
\xrightarrow{\text{B4}}
\{\phi_i\}.
$$

### 5.1 B1：Softplus

定义：

$$
a_i = \mathrm{softplus}(l_i)=\log(1+e^{l_i}).
$$

目的很明确：把 backbone 输出的任意实数 logits 转成非负原始权重。

之所以使用 softplus 而不是 ReLU，是因为：

- softplus 处处可导；
- 梯度更平滑；
- 对训练中的小 logit 更稳定，不容易出现“死区”。

### 5.2 B2：Window modulation

定义查询点到邻居节点的距离：

$$
d_i=\|\mathbf{X}_i-\mathbf{x}_q\|.
$$

再构造窗口调制：

$$
\tilde{a}_i = a_i\,w\!\left(\frac{d_i}{r_{\mathrm{support}}}\right),
\qquad
r_{\mathrm{support}}=\alpha r_{\max}.
$$

当前支持两类窗口：

1. Quartic spline

$$
w(s)=(1-s^2)_+^2.
$$

2. Wendland C2

$$
w(s)=\max(0,1-s)^4(4s+1).
$$

#### 设计目的

B2 的作用是显式施加 locality。它表达的是：

- patch 内不是所有邻居都同等重要；
- 越靠近支撑域边界，贡献应越弱；
- 这种局部性应由架构直接体现，而不是全靠 loss 去学。

#### 数值视角

在 meshfree 中，紧支撑窗口函数的引入很自然，因为局部近似本身依赖支撑域截断。B2 相当于在 backbone 学出的“原始偏好”之上，再叠加一个数值上合理的局部衰减。

当前实现进一步采用 $\alpha=1.05$ 的轻微支撑半径放大，这一设计不是为了放弃 locality，而是为了避免“最远邻居必为零权重”的机械性边界效应。也就是说，B2 现在表达的是“边界附近贡献快速衰减”，而不是“patch 中最远邻居必须被硬截断”。

### 5.3 B3：Normalization

将加窗后的权重归一化为 base shape function：

$$
\phi_i^{\text{base}}=
\frac{\tilde{a}_i}
{\sum_{j=1}^{k}\tilde{a}_j+\varepsilon}.
$$

这一层直接保证：

$$
\sum_{i=1}^{k}\phi_i^{\text{base}}=1,\qquad
\phi_i^{\text{base}}\ge 0.
$$

因此在 B3 之后，已经获得了 partition of unity 的 base 权重。

### 5.4 B4：Linear reproducing correction

为了进一步保证一阶线性再现性，引入基于 $\phi_i^{\text{base}}$ 的 reproducing correction。

二维线性基底取为：

$$
\mathbf{p}(\mathbf{x})=
\begin{bmatrix}
1\\x_1\\x_2
\end{bmatrix}.
$$

构造 moment matrix：

$$
\mathbf{M}(\mathbf{x}_q)=
\sum_{j=1}^{k}\phi_j^{\text{base}}\,
\mathbf{p}(\mathbf{X}_j)\mathbf{p}(\mathbf{X}_j)^\top.
$$

求校正系数：

$$
\mathbf{c}(\mathbf{x}_q)=
\mathbf{M}(\mathbf{x}_q)^{-1}\mathbf{p}(\mathbf{x}_q).
$$

最终校正后的 shape function 为：

$$
\phi_i(\mathbf{x}_q)=
\phi_i^{\text{base}}\,
\mathbf{p}(\mathbf{X}_i)^\top \mathbf{c}(\mathbf{x}_q).
$$

#### 设计目的

B4 的核心不是“再做一次归一化”，而是用一个闭式修正把 base 权重提升到满足一阶一致性的权函数族。

因此 B4 保证：

$$
\sum_{i=1}^{k}\phi_i(\mathbf{x}_q)=1,
$$

以及

$$
\sum_{i=1}^{k}\phi_i(\mathbf{x}_q)\mathbf{X}_i=\mathbf{x}_q.
$$

这两条正是 meshfree 近似中最重要的两个低阶 reproducing 条件。

#### 与 RK / MLS 的关系

这一修正不是一般意义上的黑箱后处理，而是和 reproducing kernel、MLS 思想一致的结构性修正。它使用 base 权重构造 moment matrix，再通过解析校正因子恢复线性再现性。因此它既保留了局部权重的信息，又显式嵌入了经典 meshfree 的 reproducing 思想。

### 5.5 为什么不强制严格非负

虽然 B3 保证了 $\phi_i^{\text{base}}\ge 0$，但 B4 之后最终 $\phi_i$ 可能出现小的负值。这一点在本方法中是**被允许的**。

原因是：

1. reproducing correction 的目标是保证一致性，不是保证逐点非负；
2. 若在 B4 后再对负值做裁切或再归一化，会破坏一阶再现性；
3. 对当前目标应用而言，一致性和快速近似比严格逐点非负更重要。

因此本方法的默认立场是：

- 强制保证 partition of unity；
- 强制保证一阶一致性；
- 允许小幅负值，但通过损失中的 `loss_neg` 进行软惩罚。

### 5.6 B4 后不能再归一化

这一点必须单独强调。B4 的闭式修正已经同时编码了零阶与一阶一致性。若在 B4 后再做：

$$
\phi_i \leftarrow \frac{\phi_i}{\sum_j \phi_j},
$$

虽然仍能保住零阶一致性，但会破坏：

$$
\sum_i \phi_i \mathbf{X}_i = \mathbf{x}_q.
$$

因此 B4 后绝不能再做额外 normalization。

---

## 6. Teacher 参考解与监督目标

### 6.1 参考解角色

当前训练并不是直接从力学残差出发，而是采用**reference-guided warm start** 思路：先用 patch-level max-ent 求解器构造一个高质量 reference shape function，再让神经网络去拟合它。

因此 teacher 的定位是：

- 为网络提供稳定监督；
- 提供可解释的目标函数空间；
- 把“学习局部权函数”的问题先简化为“逼近参考构造器”。

它不是最终物理求解器的替代，而是当前学习阶段的参考标准。

### 6.2 Patch-level max-ent 求解

对每个 patch，定义相对坐标：

$$
\mathbf{r}_i=\mathbf{X}_i-\mathbf{x}_q.
$$

给定 prior 权重 $w_i$ 与 Lagrange multiplier $\boldsymbol{\lambda}$，定义：

$$
z_i = w_i\exp(-\mathbf{r}_i^\top\boldsymbol{\lambda}),
\qquad
Z=\sum_i z_i.
$$

则 teacher shape function 为：

$$
\phi_i^{\text{ref}}=\frac{z_i}{Z}.
$$

当前实现使用 Newton 迭代去逼近使一阶矩残差最小的 $\boldsymbol{\lambda}$。

### 6.3 两类 prior

当前支持两类 prior：

1. Gaussian prior

$$
w_i=\exp\!\left(-\beta \left(\frac{\|\mathbf{r}_i\|}{r_{\max}}\right)^2\right).
$$

2. Quartic spline prior

$$
w_i=
\left(1-\left(\frac{\|\mathbf{r}_i\|}{(1+\beta)r_{\max}}\right)^2\right)_+^2.
$$

它们分别对应：

- Gaussian：更平滑、非紧支撑、训练参考更连续；
- Quartic spline：更接近紧支撑局部权函数的思路。

当前 CLI 默认仍使用 Gaussian teacher。

### 6.4 teacher 的诊断量

除了 $\phi_i^{\text{ref}}$，当前 teacher 还会记录：

- 求解是否成功；
- 迭代次数；
- 目标函数值；
- partition of unity residual；
- linear residual；
- Hessian 条件数；
- support mask。

这些量不仅用于过滤失败 patch，也为后续 failure analysis 提供了基础。

---

## 7. 训练目标与评价指标

### 7.1 总损失

当前总损失定义为：

$$
\mathcal{L}=
\mathcal{L}_{\text{data}}+
\lambda_{\text{cons}}\mathcal{L}_{\text{cons}}+
\lambda_{\text{neg}}\mathcal{L}_{\text{neg}}.
$$

其中：

1. 数据项

$$
\mathcal{L}_{\text{data}}=
\frac{1}{k}\sum_{i=1}^{k}
\left(\phi_i-\phi_i^{\text{ref}}\right)^2.
$$

2. 约束项

$$
\mathcal{L}_{\text{cons}}=
\left(\sum_i \phi_i-1\right)^2+
\left\|
\sum_i \phi_i\mathbf{X}_i-\mathbf{x}_q
\right\|^2.
$$

3. 负值惩罚项

$$
\mathcal{L}_{\text{neg}}=
\frac{1}{k}\sum_{i=1}^{k}\max(0,-\phi_i).
$$

### 7.2 各项损失的意义

- `loss_data`：要求网络输出尽量逼近 reference max-ent 解。
- `loss_cons`：虽然 B4 理论上已经保证一致性，但训练中保留这项可以额外监控数值稳定性，并对有限精度误差形成软约束。
- `loss_neg`：不禁止负值，但惩罚过大的负值，使最终 shape function 更接近数值上可接受的状态。

### 7.3 评价指标

当前主要监控以下指标：

1. `relative_l2`

$$
\frac{\|\boldsymbol{\phi}-\boldsymbol{\phi}^{\text{ref}}\|_2}
{\|\boldsymbol{\phi}^{\text{ref}}\|_2}.
$$

2. `global_linf`

$$
\|\boldsymbol{\phi}-\boldsymbol{\phi}^{\text{ref}}\|_{\infty}.
$$

3. `mean_pou_residual`

$$
\left|\sum_i \phi_i-1\right|.
$$

4. `mean_linear_residual`

$$
\left\|
\sum_i \phi_i\mathbf{X}_i-\mathbf{x}_q
\right\|.
$$

5. `negative_fraction`

表示 patch 中有多少比例的条目出现了负值。

6. `max_negative_magnitude`

表示最终 shape function 中最严重的负值幅度。

7. `cond_M`

用于衡量 reproducing correction 中 moment matrix 的病态程度。

这些指标共同构成一个判断框架：不仅要看“拟合是否好”，还要看“结构是否稳”“负值是否可控”“局部几何退化时是否病态”。

---

## 8. 方法假设、边界与当前未决项

### 8.1 当前稳定设计

截至当前实现，可以视为稳定的部分包括：

- 二维固定 $k=16$ patch 学习框架；
- `kernel_operator` 与 `mlp_baseline` 两类 backbone；
- B1--B4 structure-preserving head；
- patch-level max-ent teacher；
- 基于 reference-guided warm start 的训练范式；
- CLI 驱动的训练、评估与产物保存闭环。

### 8.2 当前边界

当前方法仍然有明确边界：

- 仅支持 2D；
- 默认固定 $k=16$；
- 默认快速配置 `data.yaml` 仍是 scaffold 规模；
- 默认快速配置中的 patch types 仍未覆盖全部病态分布；
- 虽然已经补充生产配置与 trainer 闭环，但生产规模实验结果尚未沉淀。

### 8.3 当前未决项 1：`support radius / teacher` 一致性仍待验证

当前实现已经不再直接使用

$$
w\!\left(\frac{d_i}{r_{\max}}\right),
$$

而是改为

$$
w\!\left(\frac{d_i}{r_{\mathrm{support}}}\right),
\qquad
r_{\mathrm{support}}=\alpha r_{\max},
$$

其中当前默认 $\alpha=1.05$。这已经修复了“最远邻居系统性被压成零”的实现问题，并且使 head 的 locality 与 Gaussian teacher 的平滑尾部更接近。

但这一问题并没有在方法层面被彻底封口，仍然有两点待验证：

- 默认 $\alpha=1.05$ 是否已经足够，还是应作为可调超参数进入后续实验；
- Gaussian teacher 与紧支撑 head 之间是否仍存在可观测的分布失配，尤其是在病态 patch 上。

因此，这一项的性质已经从“实现缺陷”转为“方法超参数与表征一致性的实验问题”。

### 8.4 当前未决项 2：trainer 闭环已补齐，但长程训练尚未验证

当前 trainer 已经具备：

- best validation checkpoint 保存；
- 训练结束后恢复 best 权重；
- cosine annealing learning-rate scheduler；
- 学习率曲线记录与 best-model artifact 导出。

因此从工程定义上说，训练闭环已经补齐。当前真正未完成的部分，不再是“有没有这些机制”，而是“这些机制在生产规模训练中是否带来稳定收益、是否需要进一步调参”。这需要真实训练结果来回答，而不能只靠单元测试结论。

### 8.5 当前未决项 3：配置分层已经建立，但生产实验尚未真正启动

当前仓库中已经同时存在：

- `configs/data.yaml`：用于快迭代、CLI 冒烟与单测联调；
- `configs/data_production.yaml`：用于生产规模训练，默认包含全部 7 类 patch type。

其中默认快速配置仍为：

- `num_train = 256`
- `num_val = 64`

而生产配置默认提升到：

- `num_train = 20000`
- `num_val = 2000`

这意味着“配置层面缺 production preset”的问题已经解决；接下来真正缺的是首轮真实训练、结果归档与 failure analysis，而不是继续停留在配置设计层面。

### 8.6 本方法的定位

总的来说，本方法不是要替代经典 meshfree 理论，而是把它的结构约束嵌入神经网络近似器中，形成一个：

- 具有局部几何归纳偏置；
- 具有结构合法性；
- 能够被 teacher reference 稳定监督；
- 面向后续大变形 meshfree warm start 的局部 shape function 生成器。

这也是它与普通 operator learning 模型最根本的区别。
