# 子线 A v4 DeepSets 方案：局部 patch 算子化 backbone

## 0. 摘要

本方案把 DeepSets 引入当前 shape-function generator，不是为了单纯增加模型容量，而是为了让 backbone 的结构与无网格局部 patch 的数学对象一致。局部邻域节点天然是一个无序集合，且在真实点云与更新拉格朗日分析中邻域大小 \(k\) 不应被固定为单一常数。因此，v4 阶段应新增一个 **permutation-equivariant DeepSets backbone**，作为“局部 patch 到 base measure logits”的集合型生成器。

DeepSets 在本阶段的角色不是替代 B4，也不是替代传统 RKPM / MLS / max-ent 的数值结构。它只负责从局部几何中生成更合理的 base measure / warm-start logits；partition of unity、一阶或二阶 reproducing consistency 仍由 structure-preserving head 闭式保证。

本方案延续此前三条核心定位：

1. 本工作面向更新拉格朗日无网格分析中的高频局部形函数重构瓶颈，目标是构造局部形函数快速生成器 / warm starter。
2. 模型应学习“局部几何到 shape function 或 base measure 的规则”，而不是记忆某批固定点云样本。
3. 目标不是训练一个“适配所有点云实例的全局网络”，而是训练一个对广泛局部 patch 分布族可泛化的局部生成器 / warm starter。

DeepSets 的价值正好落在第 2 和第 3 点之间：它把局部 patch 明确建模为集合，从结构上消除固定邻居顺序和固定 \(k\) 带来的错误归纳偏置。

---

## 1. 背景与问题

### 1.1 当前主线的真实目标

当前项目不应被表述为“用神经网络重新发明 shape function”。更准确的定位是：

> 在更新拉格朗日无网格分析中，局部几何随时间步、Newton 迭代和积分点不断变化，shape function 及其导数需要反复重构。本方法试图把这种高频局部重构摊销为一次离线训练后的快速前向推理，或至少为传统局部构造器提供高质量 warm start。

因此，神经网络存在的硬理由不是“它能拟合”，而是：

- 局部构造被海量重复调用；
- 传统构造或 support search / moment solve / boundary query 成本明显；
- 网络推理加结构修正后仍更快；
- 误差和负值对下游装配可控。

如果这些条件不能在后续实验中成立，该方向只能停留在 method demo，而不能声称是计算力学工具。

### 1.2 为什么“换点云要重训”会摧毁主张

如果每换一种点云都要重新训练，模型学到的就不是局部 shape-function operator，而是某个点云族上的 dataset-specific surrogate。这会带来三个问题：

1. 方法无法服务真实无网格计算，因为实际点云会随问题、边界、变形和自适应更新而变化。
2. 论文叙事会退化成“在某个数据集上拟合 teacher 输出”，缺少计算力学意义。
3. warm-start / accelerator 的摊销逻辑不成立，因为重训成本会吞掉在线收益。

因此 v4 的核心任务不是“让网络更大”，而是把任务定义成：

$$
\mathcal{G}_\theta:
\left(\mathcal{P}(x_q), \mathrm{aux}\right)
\mapsto
\{l_i\}_{i=1}^{k}
\mapsto
\{\phi_i(x_q)\}_{i=1}^{k},
$$

其中 \(\mathcal{P}(x_q)=\{\mathbf{X}_i-\mathbf{x}_q\}_{i=1}^{k}\) 是 query-centered 的局部点集，\(l_i\) 是每个支撑节点的 base logit，最终 \(\phi_i\) 由 B1--B4 head 生成。

### 1.3 当前 v3/v4 memo 已识别的关键 gap

当前方案已经具备局部化和结构头两个正确底座，但距离“广泛 patch 分布族泛化”仍有几个缺口：

1. **变 \(k\) 缺失**：固定 \(k=16\) 会把问题变成“16 节点 patch 模板学习”，不是真正的可变局部集合算子。
2. **旋转鲁棒性缺失**：普通坐标输入不自动保证旋转等变或不变，v4 只能先用 rotation augmentation for robustness。
3. **尺度归一不彻底**：B4 使用 \(r_{\max}\) 归一化，但 backbone 入口也应看到无量纲坐标。
4. **边界信息缺失**：边界附近 patch 与 interior patch 的几何含义不同，至少需要最小边界上下文。
5. **缺少 OOD 协议**：没有跨 patch family、尺度、\(k\)、旋转、边界的评测，就不能声称“换点云不重训”。

DeepSets 主要解决第 1 个 gap，并部分支撑第 5 个 gap 的实验设计。它不能单独解决旋转、尺度和边界问题。

---

## 2. 为什么 DeepSets 契合本方案

### 2.1 局部 meshfree patch 是无序集合

无网格局部邻域没有天然编号。对于同一个 patch，如果支撑节点顺序被置换：

$$
\mathbf{X}' = \Pi \mathbf{X},
$$

合理的 backbone 输出也应满足：

$$
\mathbf{l}' = \Pi \mathbf{l}.
$$

最终 shape function 也应按相同方式置换：

$$
\boldsymbol{\phi}' = \Pi \boldsymbol{\phi}.
$$

这不是可选性质，而是局部 patch 算子学习的基本一致性要求。固定长度 MLP 即使按距离排序，也会把“第几个邻居”当成有固定语义的位置；这种归纳偏置不适合支撑“任意点云不重训”的主张。

### 2.2 需要的是 permutation-equivariant DeepSets

经典 DeepSets 常写成：

$$
f(\{\mathbf{z}_i\})=
\rho\left(\sum_i \eta(\mathbf{z}_i)\right),
$$

这是 permutation-invariant 形式，适合输出 patch 级整体量。但本项目需要每个节点一个 logit，因此需要 permutation-equivariant 形式：

$$
\mathbf{h}_i = \eta(\mathbf{z}_i),
$$

$$
\mathbf{g} =
\operatorname{Pool}_{j=1}^{k}\mathbf{h}_j,
$$

$$
l_i =
\psi\left(\mathbf{z}_i,\mathbf{h}_i,\mathbf{g}\right).
$$

其中：

- \(\mathbf{z}_i\)：第 \(i\) 个节点的局部特征；
- \(\eta\)：节点 encoder；
- \(\mathbf{g}\)：整个 patch 的集合 summary；
- \(\psi\)：节点 decoder；
- \(l_i\)：第 \(i\) 个支撑节点的 base logit。

节点顺序改变时，\(\mathbf{g}\) 不变，每个节点的 \((\mathbf{z}_i,\mathbf{h}_i)\) 随置换移动，因此输出 logits 也随置换等变。

### 2.3 DeepSets 与 B1--B4 的职责分工

DeepSets 不直接输出最终合法 shape function，而是输出 base logits：

```text
query-centered patch features
      ↓
DeepSetsBackbone
      ↓
per-node logits l_i
      ↓
B1/B2/B3
      ↓
phi_base
      ↓
B4 reproducing correction
      ↓
phi_final
```

职责分工固定为：

- DeepSets：学习局部几何中“哪些节点应有更大 base measure”。
- B1/B2/B3：生成非负、局部、归一化的 base weight。
- B4：闭式保证一阶或二阶 reproducing consistency，并在病态时 fallback。

因此 DeepSets 的目标不是让 B4 消失，而是让 B4 的修正幅度变小、条件数更稳定、负值副作用更可控。

### 2.4 它和当前 kernel operator 的关系

DeepSets 不一定替代当前 kernel operator。更合理的定位是新增一个 v4 backbone baseline：

```text
MLPBaselineBackbone
    固定 k，固定顺序/排序依赖
    只作为弱 baseline，不承担泛化主张

DeepSetsBackbone
    permutation-equivariant
    variable-k ready
    v4 集合归纳偏置 baseline

KernelOperatorBackbone
    更强局部 interaction
    用于验证 pairwise geometry 是否带来额外收益
```

如果 DeepSets 在 OOD 协议上接近或超过 kernel operator，说明集合归纳偏置是主要收益来源。如果 kernel operator 明显更好，说明 pairwise / higher-order interaction 对 shape-function base measure 很关键。两种结果都能服务论文叙事。

---

## 3. 推荐架构

### 3.1 输入特征

DeepSets 不应直接吃绝对坐标。推荐输入为：

$$
\tilde{\mathbf{r}}_i =
\frac{\mathbf{X}_i-\mathbf{x}_q}{r_{\max}},
\qquad
\tilde{d}_i = \|\tilde{\mathbf{r}}_i\|_2.
$$

基础特征：

$$
\mathbf{z}_i =
[
\tilde{x}_i,
\tilde{y}_i,
\tilde{d}_i,
\beta,
\mathrm{boundary\ features}
].
$$

其中 v4 最小边界特征建议为：

$$
[sdf_q, n_x, n_y],
$$

并广播到每个节点。这里 \(sdf_q\) 与 \(n\) 是 query 级上下文，不是节点级几何。

**特征维度口径**：必须和 v4 主 memo `feature_dim_for_mode` 统一。v3 阶段 `"minimal" = 4`（`[rel_hat, dist, β]`），v4 Gap-4 把 minimal 扩到 `4 + 3 = 7`（加 `[sdf_q, n_x, n_y]`）。DeepSets backbone 的 `input_dim` 直接等于此值；`ρ_q` 作为可选 context 特征，按 v4 主 memo §5 未决项处理，保留现状（可传可不传），若启用则在 enhanced mode 里加维度，不在 minimal 中。

### 3.2 基础 DeepSets backbone

最小实现：

$$
\mathbf{h}_i = \eta(\mathbf{z}_i),
$$

$$
\mathbf{g} =
\frac{1}{\sum_i m_i}
\sum_{i=1}^{K_{\max}} m_i \mathbf{h}_i,
$$

$$
l_i =
\psi([\mathbf{z}_i,\mathbf{h}_i,\mathbf{g}]).
$$

其中 \(m_i\in\{0,1\}\) 是 padding mask，用于支持变 \(k\)。无效 padding 节点必须：

- 不参与 pooling；
- 不参与 B1--B4；
- 不计入 loss；
- 不计入 metrics。

### 3.3 DeepSets++：加入显式低阶几何 summary

朴素 DeepSets 只通过 learned pooling 表达整体 patch 几何，可能不足以捕捉 moment 病态、各向异性和密度梯度。建议 v4 实现时预留一个 `use_moment_summary` 选项，显式加入低阶几何矩：

$$
\mathbf{s}_{\mathrm{geom}}
=
\operatorname{mean}_i
[
\tilde{x}_i,
\tilde{y}_i,
\tilde{x}_i^2,
\tilde{x}_i\tilde{y}_i,
\tilde{y}_i^2,
\tilde{d}_i
].
$$

decoder 改为：

$$
l_i =
\psi([\mathbf{z}_i,\mathbf{h}_i,\mathbf{g},\mathbf{s}_{\mathrm{geom}}]).
$$

这与 B4 moment correction 的数学结构更一致，也能帮助 backbone 提前感知局部各向异性和支撑退化。

### 3.4 不在本阶段实现的增强

本阶段不做以下内容：

- SO(2)-equivariant DeepSets；
- Set Transformer / Point Transformer；
- 图网络或局部 attention 替代；
- 直接输出 max-ent dual variable；
- warm-start Newton unroll；
- 学习最终梯度或 SCNI 积分权重。

这些都可以作为 v4 成功后的后续路线。当前阶段先验证最小集合归纳偏置是否对 OOD 泛化有实质收益。

---

## 4. 代码改动方案

### 4.1 新增 backbone 文件

新增：

```text
src/shape_function/models/backbones/deepsets.py
```

建议接口：

```python
class DeepSetsBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        pooling: str = "mean",
        use_moment_summary: bool = True,
    ) -> None:
        ...

    def forward(
        self,
        node_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
```

输入输出约定：

- `node_features`: `[B, K_max, F]`
- `mask`: `[B, K_max]`，`True` 表示有效节点
- `logits`: `[B, K_max]`

### 4.2 修改模型 factory

在现有 model factory 中新增 backbone 类型：

```yaml
model:
  backbone: deepsets
  hidden_dim: 128
  num_layers: 2
  use_moment_summary: true
```

factory 需要支持：

```python
if backbone == "deepsets":
    return DeepSetsBackbone(...)
```

### 4.3 支持 mask 贯穿主链

为了让 DeepSets 真正支持变 \(k\)，mask 必须贯穿：

```text
dataset_builder
    ↓
dataloader collate_fn
    ↓
ShapeFunctionModel.forward
    ↓
backbone.forward(..., mask)
    ↓
structure_head(..., mask)
    ↓
losses / metrics
```

当前如果主链仍默认 `[B, K, ...]` 且无 mask，DeepSets 只能做“固定 k 的置换等变 backbone”，不能完成 v4 Gap-1。

### 4.4 structure head 的 mask 约束

B1--B4 需要明确忽略 padding 节点：

- B1 softplus 后，无效节点权重置零；
- B2 window 后，无效节点仍为零；
- B3 normalization 只对有效节点求和；
- B4 moment matrix 只对有效节点求和；
- reproducing residual 只统计有效节点；
- fallback 与条件数按 batch patch 计算。

关于节点数与 fallback：**不**以"有效节点数 < 6"作为 fallback 判据。6 个共线节点的 moment matrix 仍然奇异，节点数检查无法捕捉。正确做法是继续使用现有 `κ(M₂) > κ_max` 的硬门控——它是充分条件。节点数检查只用于极端情形：有效节点数 < 3 时标记为 invalid patch，不进入训练（一阶基底本身都无法闭合）。

### 4.5 配置新增项

建议新增配置：

```yaml
model:
  backbone: deepsets
  hidden_dim: 128
  num_encoder_layers: 2
  num_decoder_layers: 2
  pooling: mean
  use_moment_summary: true

data:
  k_mode: variable
  k_train_min: 12
  k_train_max: 24
  k_pad_to: 32

eval:
  k_protocols: [16, 24, 32, 48]
```

配置校验规则：

- `backbone=deepsets` 时允许 `k_mode=variable`；
- `backbone=mlp_baseline` 时不允许 `k_mode=variable`，除非实现 padding flatten 版本；
- `k_pad_to >= k_train_max`；
- `basis_order=2` 时应保证训练采样中 \(k_{\min}\ge 6\)，实际建议 \(k_{\min}\ge 12\)。

---

## 5. 测试计划

### 5.1 置换等变单测

构造同一个 batch，随机置换节点顺序：

$$
\mathbf{X}' = \Pi \mathbf{X}.
$$

检查：

$$
\mathrm{DeepSets}(\Pi \mathbf{Z}) = \Pi \mathrm{DeepSets}(\mathbf{Z}).
$$

进一步检查整个模型：

$$
\mathrm{Model}(\Pi \mathbf{Z}) = \Pi \mathrm{Model}(\mathbf{Z}).
$$

**mask 同步置换（关键）**：如果 batch 中包含 padding，pooling 是 masked mean / sum。测试时必须把 mask 和节点一起置换，否则 masked pooling 的结果会被破坏：

```python
Z_perm = Z[:, perm]
mask_perm = mask[:, perm]
out_perm = DeepSets(Z_perm, mask_perm)
out_ref  = DeepSets(Z, mask)[:, perm]
assert torch.allclose(out_perm, out_ref, atol=1e-6)
```

容差建议：

```text
max_abs_error < 1e-6
```

### 5.2 mask / padding 单测

同一个真实 patch 分别 padding 到 `K=24` 和 `K=32`，有效节点完全一致。检查：

- 有效节点 logits 一致；
- 有效节点 `phi_corr` 一致；
- padding 节点输出不参与归一化；
- padding 节点不影响 `cond_M` 与 reproducing residual。

### 5.3 变 k 冒烟测试

使用极小配置训练：

```text
k_train_min = 12
k_train_max = 24
k_pad_to = 32
num_train = 128
num_val = 32
epochs = 2
```

验收：

- CLI 可跑完；
- `curves.npz` 有 base-linear、fallback、negative、cond 指标；
- 无 NaN；
- `eval_metrics.json` 能按不同 \(k\) 协议输出。

### 5.4 不回归测试

现有路径必须保持：

- `kernel_operator + fixed k` 能跑；
- `legacy_teacher_baseline` 能跑；
- `basis_order=1/2` head 单测仍过；
- visualizer 不依赖 DeepSets 特定字段。

---

## 6. 实验设计

**与 v4 主 memo §5 实验矩阵的协调**：
v4 主 memo §5 定义了 4 组消融（v4-A/B/C/Full）× 6 项 OOD 协议。本节不重复定义消融轴，而是补充 **backbone 对照轴**。最终联合矩阵为：

- **Backbone 对照**（固定为 v4-Full 数据管线配置）：`MLP / DeepSets / KernelOp`，各跑 6 项 OOD 协议，共 18 次评测。
- **消融链**（v4 主 memo §5 的 v4-A/B/C/Full）：只挂在 DeepSets 上跑，共 24 次评测。
- **合计**：18 + 24 = 42 次评测。

不在每个 backbone 上重复 4 组消融——消融的目的是验证数据管线各项改造的独立收益，一次跑清楚即可；backbone 对照只需要"相同数据 + 不同架构"的单一对比。

本节剩余小节（6.1–6.5）是针对 backbone 对照轴的细化设计，不与主 memo §5 冲突。

### 6.1 第一组：固定 k 下的架构对照

目的：先隔离“集合等变结构”是否有收益，不马上混入变 \(k\) 难度。

训练组：

```text
MLPBaselineBackbone, k=16
KernelOperatorBackbone, k=16
DeepSetsBackbone, k=16
```

评测：

- IID patch；
- node permutation test；
- rotation augmentation off/on；
- A2/A3 已有 patch types。

关注指标：

- `base_linear_residual`
- `mean_quad_residual`
- `max_negative_magnitude`
- `fallback_rate`
- `mean_cond_M`
- `p95_cond_M`
- `correction_strength = ||phi_corr - phi_base||`

预期解释：

- 如果 DeepSets 在置换测试中严格通过，而 MLP 不通过，则证明固定顺序 MLP 不适合泛化叙事。
- 如果 DeepSets 的 base residual 低于 MLP，说明集合归纳偏置有实际训练收益。

### 6.2 第二组：变 k 插值

训练：

$$
k\in[12,24].
$$

测试：

$$
k\in\{12,16,20,24\}.
$$

目的：验证 DeepSets 在训练区间内部的 variable-k 插值能力。

通过标准：

- 指标不随 \(k\) 剧烈漂移；
- `fallback_rate` 不系统性上升；
- `max_negative_magnitude` 不超过 v4 设定阈值。

### 6.3 第三组：变 k 外推

训练：

$$
k\in[12,24].
$$

测试：

$$
k\in\{32,48\}.
$$

目的：验证是否具备超出训练邻居数的弱外推能力。

这里不能预设一定成功。合理口径是：

> DeepSets 结构允许可变 \(k\)，但数值泛化到更大邻域仍需实验验证。

如果外推失败，应区分：

- pooling summary 是否随 \(k\) 变化失真；
- B4 moment 条件数是否变差；
- base measure 是否过度稀释；
- window/support radius 是否需要随 \(k\) 调整。

### 6.4 第四组：跨 patch family OOD

训练时保留主要 patch family，测试时做 hold-out：

```text
train: uniform + mildly_perturbed + highly_random + boundary_truncated
test: clustered
test: anisotropic
test: sparse_dense_transition
```

目的：证明模型不是记忆某几个 patch 模板。

重点比较：

- DeepSets vs KernelOperator；
- fixed-k vs variable-k；
- with/without boundary features；
- with/without moment summary。

### 6.5 第五组：UL warm-start 前置评估

在真正嵌入 UL 前，先做 patch-level 成本评估：

```text
teacher / RKPM direct solve time
neural forward time
neural + B4 correction time
neural warm start + few-step correction time
```

同时报告：

- 单 patch 平均耗时；
- batch 推理吞吐；
- CPU / GPU 利用率；
- support search 是否仍是瓶颈；
- 误差和负值是否可控。

这一步用于回答项目生死线问题：

> 学习方法是否真的能摊薄局部重构成本？

---

## 7. 指标与判据

### 7.1 结构合法性

必须报告：

- `mean_pou_residual`
- `mean_linear_residual`
- `mean_quad_residual`
- `fallback_rate`
- `mean_cond_M`
- `p95_cond_M`
- `worst_cond_M`

B4 后这些指标可以很小，但不能只看最终值，因为结构头可能掩盖 backbone 的问题。

### 7.2 backbone 是否真的学到更好 base measure

必须新增或重点报告：

$$
\mathrm{base\_linear\_residual}
=
\left\|
\sum_i \phi_i^{base}
(\mathbf{X}_i-\mathbf{x}_q)
\right\|_2.
$$

建议新增：

$$
\mathrm{correction\_strength}
=
\|\boldsymbol{\phi}^{corr}-\boldsymbol{\phi}^{base}\|_2.
$$

以及：

$$
\mathrm{relative\_correction\_strength}
=
\frac{
\|\boldsymbol{\phi}^{corr}-\boldsymbol{\phi}^{base}\|_2
}{
\|\boldsymbol{\phi}^{base}\|_2+\epsilon
}.
$$

如果 DeepSets 有用，理想现象不是“最终 B4 residual 更小”，因为 B4 本来就能强行修到很小；理想现象应是：

- `base_linear_residual` 更低；
- `correction_strength` 更小；
- `cond_M2` 更稳定；
- `max_negative_magnitude` 更小；
- OOD 下退化更慢。

### 7.3 负值副作用

继续报告：

- `negative_fraction`
- `max_negative_magnitude`
- `negative_fraction_2nd`
- `max_negative_magnitude_2nd`

当前 v3 Target 已暴露 `max_negative_magnitude` 过大的问题。DeepSets 的成功不应只看二阶 residual，而应看它是否能减少 B4 强投影导致的负值放大。

### 7.4 泛化判据

量化阈值统一引用 v4 主 memo §6（7 条，含 `relative_correction_strength < 0.3`）。本节不重复定义阈值，避免后续漂移。

定性上，v4-Full 最终主张只有在以下条件同时满足时才成立：

1. 在 OOD patch family 上不需要重训；
2. 在 \(k\) 插值协议上稳定；
3. 在有限 \(k\) 外推协议上不崩；
4. 旋转增强后旋转测试退化可控；
5. 边界 patch 上指标不系统性劣化；
6. patch-level 成本相对传统构造器有实际收益。

每一条的量化成功判据（哪些指标、阈值多少、退化倍数允许多少）都落在 v4 主 memo §6 的 7 条阈值上——本文件不另行规定。

---

## 8. 与论文叙事的关系

### 8.1 可以主张的内容

如果 DeepSets 路径实验成功，可以写：

> We model the local meshfree support as an unordered set and use a permutation-equivariant DeepSets-style backbone to generate nodal base-measure logits. This design aligns the neural architecture with the patch-level operator formulation and avoids relying on a fixed ordering of support nodes.

中文底稿：

> 我们将局部无网格支撑域视为无序点集，并使用 permutation-equivariant DeepSets-style backbone 生成节点级 base-measure logits。该设计使神经网络结构与 patch-level operator formulation 对齐，避免把支撑节点的任意编号或距离排序误当成物理语义。

### 8.2 不能提前主张的内容

在实验闭合前，不能写：

- “模型已经适用于任意点云”；
- “DeepSets 保证旋转不变”；
- “DeepSets 保证所有 \(k\) 外推稳定”；
- “网络替代传统 shape function 构造器”；
- “二阶一致性由网络学出”。

更稳的写法是：

> DeepSets provides the architectural precondition for permutation-equivariant and variable-size patch processing, while actual OOD generalization remains an empirical claim tested by the v4 protocol.

### 8.3 与 warm starter 的关系

DeepSets 输出的是 base measure logits，可被理解为传统局部构造器的 warm-start 参数之一。后续如果接入 RKPM / MLS / max-ent / Newton correction，可以有三种路线：

1. **直接生成 base weights**：DeepSets 输出 logits，B4 修正后直接用于装配。
2. **warm-start 传统构造器**：DeepSets 输出初始权重或 dual variable 初值，再跑少步局部迭代。
3. **预测中间量**：DeepSets 输出 moment preconditioner、support scale 或 correction strength，用于减少传统构造器求解成本。

本阶段只做第 1 条，保留第 2 和第 3 条作为后续 UL 工程路线。

---

## 9. 分阶段执行计划

### Phase D0：文档与接口冻结

目标：

- 确认 DeepSets 是 backbone，不改 B1--B4；
- 确认输出是 per-node logits，不是最终 shape function；
- 确认 mask 是 variable-k 的必要条件；
- 确认实验先从小规模 OOD matrix 开始。

交付：

- 本文档；
- v4 memo 中增加 DeepSets 方案引用；
- issue / TODO 列表。

### Phase D0'：OOD 协议骨架（必须先于 D1）

目标（采纳 v4 主 memo §4.1 "先定义要证明什么，再改模型"）：

- 新增 `src/shape_function/eval/ood_eval.py`，定义 v4 主 memo §4.1 的 6 项协议（IID / LOTO / Rotation / k-OOD / Scale / Boundary）；
- 新增 `configs/ood_eval.yaml`，为每项协议配置数据生成参数；
- 每项协议产出 `ood_eval_metrics.json`，字段与 A3 `eval_metrics.json` 对齐；
- 指标字段必须包含 v4 主 memo §6 的全部 7 个量（含 `relative_correction_strength`）。

交付：

- `ood_eval.py` 可以 stub 一个 "evaluate existing v3 checkpoint" 的路径（暂不依赖 DeepSets），验证协议接口打通；
- D1 实现 DeepSets 时直接可用这套评测 harness，不用等 D3 才发现接口不兼容。

这一步把"泛化证据"从口号变成可 CI 的测试目标。

### Phase D1：固定 k DeepSets

目标：

- 实现 `DeepSetsBackbone`；
- 暂时固定 \(k=16\)；
- 先验证 permutation-equivariance；
- 与 MLP / kernel operator 做最小对照。

验收：

- 单测通过；
- CLI 可训练；
- O1/O2 head 不回归；
- DeepSets 在节点置换下输出严格等变。

### Phase D2：mask 与 variable-k

目标：

- 数据生成支持 \(k\in[k_{\min},k_{\max}]\)；
- collate padding + mask；
- backbone/head/loss/metrics 全链路支持 mask。

验收：

- 同一 patch padding 到不同长度输出一致；
- variable-k 小训练可跑完；
- B4 对无效 padding 节点无感。

### Phase D3：v4 OOD 小矩阵

目标：

- 跑 3 个 backbone；
- 跑 6 个 OOD 协议；
- 汇总 heatmap。

最小矩阵：

```text
Backbone: MLP / DeepSets / KernelOperator
Protocol: IID / rotation / k-interp / k-extra / patch-family OOD / boundary
```

验收：

- 每组有 `eval_metrics.json`；
- 有 summary heatmap；
- 能回答 DeepSets 是否改善 OOD 稳定性。

### Phase D4：UL warm-start 前置成本评估

目标：

- 不急着接完整 Fortran/UL；
- 先在 patch-level 测成本和精度；
- 判断是否值得进入下游装配。

验收：

- teacher/RKPM 与 DeepSets+B4 的 wall-clock 对比；
- batch 推理吞吐；
- 准确性与负值副作用报告。

---

## 10. 风险与应对

### R1：DeepSets 表达力不足

征兆：

- OOD 稳定但精度弱；
- base residual 降不下去；
- correction strength 仍大。

应对：

- 开启 moment summary；
- 增加 encoder/decoder 深度；
- 与 kernel operator 对比，不强行把 DeepSets 设为最终主模型。

### R2：变 k 后 B4 条件数变差

征兆：

- \(k\) 外推时 `cond_M2` 急剧升高；
- fallback_rate 上升；
- 负值放大。

应对：

- 按 \(k\) 分层统计；
- 缩紧 `kappa_max`；
- 调整 support radius；
- 对极端 patch fallback 到一阶或传统构造器。

### R3：旋转 OOD 仍失败

征兆：

- rotation test 下 base residual 明显劣化；
- DeepSets 与 kernel operator 都失败。

应对：

- 增强 rotation augmentation；
- 引入旋转不变量 summary；
- 后续再考虑 SO(2)-equivariant backbone，不在本阶段硬做。

### R4：边界特征不足

征兆：

- interior 指标好，boundary 指标系统性差。

应对：

- 保留 `[sdf_q, n_x, n_y]` 最小特征；
- 后续扩展 corner / concave / visibility；
- 不在 v4 一次性塞过多边界工程。

### R5：推理不比传统构造快

征兆：

- DeepSets+B4 单 patch 成本接近或超过 teacher/RKPM；
- GPU batch 吞吐无法覆盖 support search 成本。

应对：

- 把网络定位降级为 warm-start；
- 预测中间变量而非最终 \(\phi\)；
- 优化 batch support construction；
- 如果仍无收益，则停止下游 UL 集成。

---

## 11. 推荐结论

DeepSets 值得加入 v4，但它的定位必须精确：

- 它是 backbone 层的集合等变改造；
- 它主要解决 permutation 和 variable-k；
- 它不替代 B4；
- 它不自动解决旋转、尺度和边界；
- 它应作为 v4 的关键 baseline，而不是一开始就被包装为最终最优模型。

最稳的执行顺序是：

1. 先实现固定 \(k\) 的 `DeepSetsBackbone` 和置换等变单测。
2. 再接 padding/mask，打通 variable-k。
3. 再跑 OOD 小矩阵，判断集合归纳偏置是否确实提升泛化。
4. 最后再做 patch-level wall-clock 和 warm-start 价值评估。

一句话总结：

> DeepSets 与本项目的 patch-level operator framing 高度契合。它不是因为“更复杂”而有用，而是因为它把局部无网格 patch 的无序集合属性写进了模型结构，是 v4 从固定数据集拟合走向跨点云泛化必须补的一块结构底座。
