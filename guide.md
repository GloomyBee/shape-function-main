好，那我直接往前推进到**“实现蓝图版”**。
下面我不再停留在选题和框架层，而是把你的 v2 方案压成一个**可以开始编码的工程说明**：先做什么、每个模块输入输出是什么、关键实现细节在哪里、训练时该记录什么、先跑哪些最小实验。

我会默认你当前的主线不变：
**kernel-integral operator 作为首选 backbone，structure-preserving head 作为方法核心，reference-guided warm start 作为主训练路线。** 这和你最新版技术文档一致。

---

# 一、整体实施顺序

我建议你不要同时铺开所有模块，而是按下面顺序推进。

## Phase A：先把数据工厂和输出头做稳

这是地基。

先完成：

1. patch 采样器
2. reference max-ent 生成器
3. B1–B4 输出头
4. 输出头数值验证脚本

只要这四个没稳住，后面任何 backbone 训练结果都不可信。

## Phase B：先上两个 backbone

先做：

1. kernel-integral operator
2. MLP baseline

原因很简单：

* 一个是你最看好的主方案；
* 一个是最低基准。

这样最快能形成第一轮“有意义结果”。

## Phase C：再补完整比较

再加：

1. DeepSets / Set Transformer-lite
2. Transolver-lite

## Phase D：最后嵌入求解器

等你确定最优 backbone + 输出头稳定后，再接 updated-Lagrangian meshfree solver。

---

# 二、工程目录建议

建议你一开始就把项目结构搭清楚，不然后面会乱。

```text
project_root/
│
├─ configs/
│   ├─ data.yaml
│   ├─ train_kernel_operator.yaml
│   ├─ train_mlp.yaml
│   ├─ train_deepsets.yaml
│   └─ train_transolver_lite.yaml
│
├─ data/
│   ├─ raw_patches/
│   ├─ processed/
│   ├─ splits/
│   └─ stats/
│
├─ src/
│   ├─ data/
│   │   ├─ patch_sampler.py
│   │   ├─ patch_validator.py
│   │   ├─ maxent_solver.py
│   │   ├─ dataset_builder.py
│   │   └─ dataloader.py
│   │
│   ├─ models/
│   │   ├─ heads/
│   │   │   ├─ structure_head.py
│   │   │   └─ reproducing_correction.py
│   │   │
│   │   ├─ backbones/
│   │   │   ├─ kernel_operator.py
│   │   │   ├─ mlp_baseline.py
│   │   │   ├─ deepsets.py
│   │   │   └─ transolver_lite.py
│   │   │
│   │   ├─ full_model.py
│   │   └─ feature_builder.py
│   │
│   ├─ train/
│   │   ├─ losses.py
│   │   ├─ metrics.py
│   │   ├─ trainer.py
│   │   └─ callbacks.py
│   │
│   ├─ eval/
│   │   ├─ eval_backbones.py
│   │   ├─ eval_ood.py
│   │   ├─ eval_head_ablation.py
│   │   └─ visualize_patches.py
│   │
│   └─ utils/
│       ├─ geometry.py
│       ├─ masking.py
│       ├─ logging.py
│       └─ seed.py
│
├─ notebooks/
│   ├─ 01_patch_statistics.ipynb
│   ├─ 02_output_head_validation.ipynb
│   └─ 03_failure_case_analysis.ipynb
│
└─ runs/
```

---

# 三、数据工厂：你最先该做的模块

这部分是成败关键。

---

## 3.1 patch 的基本数据结构

建议每个 patch 保存成统一字典格式：

```python
patch = {
    "x_q": np.ndarray,          # shape: (d,)
    "X": np.ndarray,            # shape: (k, d)
    "beta": float,
    "rho_q": np.ndarray,        # shape: (n_rho,)
    "phi_ref": np.ndarray,      # shape: (k,)
    "r_max": float,
    "cond_M_ref": float,        # 可选：基于 phi_ref 或 phi_base 统计
    "patch_type": str,          # e.g. uniform / clustered / boundary_truncated
    "meta": {...}
}
```

如果你之后支持 variable-(k)，那 `X.shape[0]` 就是当前 patch 的真实 (k)。

---

## 3.2 patch 采样器怎么写

你要做的不是简单随机点，而是**可控生成多类局部几何**。

建议先实现以下 patch 类型生成器：

### A. uniform

在单位圆或单位方域中均匀采样，再以 query 为中心截取局部邻域。

### B. mildly_perturbed

从规则点阵出发，加小随机扰动。

### C. highly_random

纯随机散点。

### D. clustered

先采样 2–3 个 cluster center，再在 cluster 周围采样。

### E. boundary_truncated

先构造完整 patch，再用边界平面/曲线裁掉一部分邻域。

### F. anisotropic

对坐标施加仿射拉伸：
[
X \leftarrow A X,\quad A=\mathrm{diag}(a,b),\ a\neq b
]

### G. sparse_dense_transition

一部分邻域高密，一部分低密。

---

## 3.3 query 点怎么选

不要只把 query 固定在 patch 中心。
建议 query 的相对位置也多样化，否则网络会过拟合到“中心点情形”。

建议三类 query：

1. **interior-centered**：在 patch 中央附近
2. **off-center**：偏心位置
3. **near-boundary**：靠近局部支撑边界

你可以把 query 作为局部 patch 几何的一部分，而不是固定参考点。

---

## 3.4 邻域搜索策略

建议一开始就明确两种模式：

### 模式 1：固定 k

最适合先做主实验。

* 2D 先取 (k=16)
* 再扩到 (k=8,12,20,25)

### 模式 2：variable-k

用于后续扩展与 generalization 测试。

我的建议是：
**训练主方案先固定 k=16 跑通。**
这是最快形成第一轮稳定结果的方式。
等输出头和主 backbone 稳了，再上 variable-(k)。

虽然 v2 文档已经支持 variable-(k)，但实现上你完全可以先固定 k。这样更稳。

---

## 3.5 reference max-ent 求解器

你需要一个离线 reference generator。

### 它的功能：

给定：

* query 点 (x_q)
* 邻域节点 (X_i)
* 参数 (\beta)

输出：

* (\phi_i^{ref}(x_q))

### 建议要求：

1. 先只支持 2D
2. 只做 patch 级求解
3. 支持批量离线生成
4. 生成失败时返回状态码

### 额外建议保存：

* solve success / fail
* 迭代次数
* reference objective 值
* reference consistency residual

这些信息后面做 failure analysis 很有用。

---

## 3.6 patch 质检器

这一步很关键，建议你单独做 `patch_validator.py`。

### 至少检查：

1. 邻域节点是否重复
2. 是否存在几乎重合的点
3. convex hull 是否包含 query（可选）
4. patch 协方差矩阵特征值比
5. (k) 是否足够支撑线性 reproducing
6. 参考权重是否有 NaN / inf
7. B4 的 moment matrix 条件数是否过大

### 建议输出：

* `is_valid`
* `warnings`
* `patch_type`
* `cond_geom`
* `cond_M_candidate`

这样你后面能明确知道是模型坏了，还是 patch 本来就病态。

---

# 四、特征构造模块：feature_builder

---

## 4.1 最小特征集

按 v2 文档，最小特征集可取：

[
f_i = [\hat r_{ix}, \hat r_{iy}, |\hat r_i|, \beta]
]

2D 下 shape 为 `(k, 4)`。

建议你把它实现成配置项：

```python
feature_mode = "minimal"
```

---

## 4.2 增强特征集

按 v2，再加：

* mean nearest-neighbor distance
* (k/k_{max})
* covariance eigenvalue ratio

则 2D 下变为 `(k, 7)`。

建议 feature_builder 输出两部分：

```python
node_features: (k, d_f)
global_context: (n_rho,)
```

虽然目前你把 (\rho_q) 复制进每个节点特征也可以，但工程上分开更清楚。
后面不同 backbone 也更容易处理。

---

# 五、Module B：structure-preserving head 的实现蓝图

这是你最先该独立验证的模块。

---

## 5.1 输入与输出

### 输入

* `logits`: `(B, k)` 或 `(k,)`
* `X`: `(B, k, d)` 或 `(k, d)`
* `x_q`: `(B, d)` 或 `(d,)`
* `mask`: `(B, k)` 可选
* `r_max`: `(B, 1)` 或标量

### 输出

* `phi_base`: `(B, k)`
* `phi_corr`: `(B, k)`
* diagnostics:

  * `sum_phi`
  * `lin_residual`
  * `neg_fraction`
  * `cond_M`

---

## 5.2 B1: softplus

```python
a = F.softplus(logits)
```

如果有 mask：

```python
a = a * mask
```

---

## 5.3 B2: window modulation

建议先实现两种 window：

### quartic spline

[
w(s) = (1-s^2)^2_+
]

### Wendland C2

[
w(s)=\max(0,1-s)^4(4s+1)
]

代码层面：

```python
s = norm(X - x_q[:, None, :], dim=-1) / r_max
w = quartic_or_wendland(s)
a_tilde = a * w
```

如果 padding 节点，强制 `w=0`。

---

## 5.4 B3: normalization

```python
den = a_tilde.sum(dim=-1, keepdim=True) + eps
phi_base = a_tilde / den
```

这里要做一个 sanity check：

* 如果某个 patch 的 `den` 很小，说明窗口和 logits 组合出问题了，应该报警。

---

## 5.5 B4: reproducing correction

这是最核心的部分。

### 2D 情形

[
p(X_i) = [1, X_i^x, X_i^y]^T
]

代码建议：

```python
def build_basis(X):  # X: (B, k, 2)
    ones = torch.ones_like(X[..., :1])
    return torch.cat([ones, X], dim=-1)  # (B, k, 3)
```

然后：

```python
P = build_basis(X)                  # (B, k, 3)
pxq = build_basis(x_q[:, None, :])  # (B, 1, 3) -> squeeze to (B, 3)

# M = sum_j phi_base_j * p(X_j) p(X_j)^T
M = torch.einsum("bk,bki,bkj->bij", phi_base, P, P)
M = M + eps_reg * I

c = torch.linalg.solve(M, pxq.squeeze(1))      # (B, 3)
corr = torch.einsum("bki,bi->bk", P, c)        # p(X_i)^T c(x_q)
phi_corr = phi_base * corr
```

### 注意

* 不要在 B4 后再归一化
* `eps_reg` 建议可配置
* 输出 `cond_M`

---

## 5.6 B 模块的单元测试

这部分必须独立跑，不依赖 backbone。

### 测试 1：随机 logits + 合理 patch

输入随机 logits，检查：

* ( \sum_i \phi_i \approx 1 )
* ( \sum_i \phi_i X_i \approx x_q )

### 测试 2：平移不变性

整体平移 patch 和 query，检查结果是否正确平移响应。

### 测试 3：缩放归一化一致性

改 (r_{max}) 和归一化后，检查数值行为。

### 测试 4：病态 patch

几乎共线点，检查：

* `cond_M`
* 负值大小
* residual 是否爆炸

### 测试 5：反向传播

对 logits 求梯度，检查无 NaN。

---

# 六、首选 backbone：kernel-integral operator 的实现蓝图

---

## 6.1 输入输出

### 输入

* `F`: `(B, k, d_f)`
* `R`: `(B, k, d)`  即 (\hat r_i)
* `mask`: `(B, k)` 可选

### 输出

* `logits`: `(B, k)`

---

## 6.2 网络结构建议

### Lift

```python
MLP_lift: d_f -> 64 -> 64
```

### Message passing layers

先做 2 层就够。

每层：

1. 构造 pairwise 几何量
2. 构造 pair feature
3. 通过小 MLP 得到 modulation scalar
4. 乘 base kernel
5. 聚合消息
6. 残差更新

---

## 6.3 pairwise 特征

按 v2 建议：

[
[\mathbf h_i - \mathbf h_j,\ \hat r_i,\ \hat r_j,\ |\hat r_i-\hat r_j|]
]

如果 (C=64)，2D 下 pair feature 维度为：

* (h_i-h_j): 64
* (\hat r_i): 2
* (\hat r_j): 2
* distance: 1

共 69 维。

实现时可以这样：

```python
h_i = h.unsqueeze(2)   # (B,k,1,C)
h_j = h.unsqueeze(1)   # (B,1,k,C)
r_i = R.unsqueeze(2)   # (B,k,1,d)
r_j = R.unsqueeze(1)   # (B,1,k,d)

pair_diff_h = h_i - h_j
pair_dist = torch.norm(r_i - r_j, dim=-1, keepdim=True)

pair_feat = torch.cat([pair_diff_h, r_i.expand_as(r_j), r_j.expand_as(r_i), pair_dist], dim=-1)
```

---

## 6.4 base kernel

建议先做两种：

### A. Wendland C2

### B. Gaussian truncated

因为你最终可能会发现：

* 紧支撑核更贴 meshfree 语言；
* Gaussian 在训练初期更平滑。

可以作为一个小 ablation。

---

## 6.5 modulation network

```python
g_theta: 69 -> 64 -> 1
```

输出建议用：

* `softplus`
  或
* `sigmoid * scale`

确保 modulation 不出现过度爆炸。

然后：
[
\alpha_{ij}=k_{base}(r_{ij})\cdot g_\theta(\cdot)
]

可以再加一层 row-normalization，使消息聚合稳定：

```python
alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + eps)
```

虽然文档没强制要求，但工程上通常更稳。

---

## 6.6 update

```python
v = linear_v(h)
m = torch.einsum("bij,bjc->bic", alpha, v)
h = gelu(W(h) + m)
```

建议加 residual：

```python
h = h + gelu(W(h) + m)
```

或者用：

```python
h_new = gelu(W(h) + m)
h = h + h_new
```

小网络里 residual 很有帮助。

---

## 6.7 readout

```python
logits = linear_out(h).squeeze(-1)
```

mask 节点 logits 可以直接置 0 或很小值。

---

# 七、最小可运行模型：先不要一次做四个 backbone

我建议你真正开始时，只做：

## Model A

kernel-integral backbone + structure head

## Model B

MLP baseline + structure head

这是最快能产出第一轮结论的组合。

等这两个都跑通了，再加：

* DeepSets
* Transolver-lite

这样开发效率最高。

---

# 八、训练器怎么写

---

## 8.1 forward pipeline

```python
logits = backbone(F, R, mask)
phi_base, phi_corr, aux = structure_head(logits, X, x_q, mask, r_max)
loss = loss_fn(phi_corr, phi_ref, aux)
```

---

## 8.2 loss_fn

按你当前 v2：

[
\mathcal L = \mathcal L_{data} + \lambda_c \mathcal L_{cons} + \lambda_{neg}\mathcal L_{neg}
]

建议实现成：

```python
loss_data = mse(phi_corr[mask], phi_ref[mask])
loss_cons = ((sum_phi - 1)**2 + lin_residual_sq).mean()
loss_neg = relu(-phi_corr).mean()
loss = loss_data + lambda_c * loss_cons + lambda_neg * loss_neg
```

其中：

* `sum_phi`
* `lin_residual_sq`
* `neg_fraction`
* `max_neg_mag`
  都从 `aux` 里拿。

---

## 8.3 日志指标

每个 epoch 至少记录：

* train/val loss_data
* train/val loss_cons
* train/val loss_neg
* mean relative L2
* max pointwise error
* mean PoU residual
* mean linear residual
* negative fraction
* max negative magnitude
* mean cond(M)
* worst cond(M)

这些指标是你后面写论文最需要的，不要等实验结束再补。

---

# 九、第一批必须跑的 sanity experiments

在大规模训练前，先跑下面这些小实验。

---

## sanity 1：输出头不依赖 backbone 是否可正常工作

随机 logits 下跑 1000 个 patch，验证：

* residual 是否小
* 是否出现大负值
* cond(M) 分布

## sanity 2：在 1000 个 patch 上过拟合

用 kernel-integral backbone 训练到几乎过拟合，确认：

* loss_data 真能降下去
* 不是 pipeline 某处有 bug

## sanity 3：固定 k=16，小数据集

训练 1 万 patch，快速看：

* kernel-integral vs MLP 的差距
* output head ablation 是否符合预期

## sanity 4：boundary-truncated patch 单独测试

因为这类 patch 最容易出问题，建议单独可视化。

---

# 十、我建议的实际开发时间表

## 第 1 周

* patch 采样器
* reference 求解器
* patch validator
* 数据可视化

## 第 2 周

* structure head
* structure head 单元测试
* cond(M)、negative fraction 诊断工具

## 第 3 周

* kernel-integral backbone
* MLP baseline
* 最小训练 loop

## 第 4 周

* 固定 k=16 主实验
* 输出头消融
* failure case 可视化

## 第 5–6 周

* DeepSets / Set Transformer-lite
* Transolver-lite
* OOD geometry / OOD beta

## 第 7–8 周

* solver embedding
* KAN head-to-head
* 论文图表整理

---

# 十一、我的最终建议

你现在已经不需要再反复改方案了。
这一步之后，最正确的动作就是：

> **先做数据工厂和 B 模块，再做 kernel-integral + MLP 两个 backbone，形成第一轮硬结果。**

这是风险最低、信息增益最高的路径。

如果你愿意，我下一步可以直接继续往前推到更具体的一层：
**我把 `structure_head.py` 和 `kernel_operator.py` 的伪代码骨架直接写出来，连张量维度和关键注释一起给你。**
