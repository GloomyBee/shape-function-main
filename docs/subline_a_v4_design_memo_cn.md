# 子线 A v4 设计备忘：Patch-Level Shape-Function Operator

**日期**：2026-04-21
**作用**：对 v3（teacher-free + 二阶 B4）的扩展方案，把"跨点云不重训"从主张变证据。
**不做**的事情：在线 patch 生成器、SO(2)-equivariant 架构、full amortized max-ent、改 backbone 容量。

---

## 1. 目标重述

把本项目学习对象从"拟合 teacher 输出"重述为：

> 学习一个 **patch-level shape-function operator**
>
> $$\mathcal{G}_\theta:\ (\mathcal{P}(x_q),\text{aux})\ \longmapsto\ \{\phi_i(x_q)\}_{i=1}^{k},$$
>
> 其中 backbone 负责把局部几何编码为 base measure，head 通过解析结构严格闭合再生性。网络不是直接替代传统形函数公式，而是在局部 patch 上提供快速 warm start；最终数值性质由结构层兜底。

这一重述直接决定后续所有设计：只要 patch 的输入表示足够"算子化"（平移 / 尺度 / 置换 / 邻域大小 / 边界几何 上都无偏依赖），backbone 就有可能在一个广域 patch 分布族上训练一次、在新的点云离散上按 query 调用。

这里的口径需要收紧为四句话，后文统一按这四层含义展开：

1. **项目定位**：本工作面向更新拉格朗日无网格分析中的局部形函数重构瓶颈，目标是构造一个局部 shape-function 快速生成器 / warm starter，而不是再发明一个独立于传统数值法的新形函数理论。
2. **核心假设**：局部几何到 shape function 的映射具有可学习的 patch-level 规律；模型应学习这种局部规则，而不是记忆某一批固定点云样本、固定编号或固定全局域上的数值模板。
3. **方法目标**：我们不追求"一个适配所有点云实例的全局网络"，而是追求"一个对广泛局部 patch 分布族可泛化的局部生成器 / warm starter"。这里明确写"广泛 patch 分布族"，不写"任意 patch 全覆盖"，因为病态 patch 仍应由检测、fallback 或拒判机制处理。
4. **当前尚待证明**：上述三点目前是方法定位与研究假设，不是既成结论。要让它们成立，必须通过 OOD 协议证明跨 patch 分布泛化，通过下游 wall-clock 与收敛实验证明 warm-start / accelerator 的实际价值。

DeepSets-style backbone 是这一重述下的一个具体落地选项：它把局部 patch 视为无序集合，输出 permutation-equivariant 的 per-node logits，从结构上服务于“固定点云记忆”到“局部 patch 算子学习”的转向。详细方案见 `docs/subline_a_v4_deepsets_plan_cn.md`。

不回答的问题：二阶一致性能否通过网络学出来。本阶段仍然把二阶交给 B4 结构层。

---

## 2. 现有底座

v3 阶段已经落地的正确决策：

1. **局部化**：`ShapeFunctionModel.forward` 只取 query 邻域 `rel_coords_and_radius_torch(X, x_q)`，不看全局点云。
2. **结构化 head**：B1 softplus → B3 normalize → B4 reproducing correction；一阶严格、二阶结构性闭合；hard fallback on `κ(M₂)`。
3. **Patch family 多样性**：7 类 patch types（uniform / mildly_perturbed / highly_random / clustered / boundary_truncated / anisotropic / sparse_dense_transition），A2 扫描 7000 patch 已给出 `κ_max` 的分布证据。
4. **Teacher-free 训练范式可跑通**：`unsupervised_v1 = L_base_lin + λ_ent·KL(phi_base‖prior) + λ_neg·ReLU(-φ)` 在 A3 短对照中无崩溃、无 NaN。

这两点（局部化 + 硬约束 head）是泛化的必要条件，本 memo 不再改动。

---

## 3. 五个关键 Gap

按"对跨点云证据链的致命程度"排序。注意顺序相对 v3 review 调整：**变 k 比旋转更致命**。

### Gap-1：邻域大小固定 k=16（最致命）

- **现象**：`configs/*.yaml` 全部 `k_neighbors: 16`；`MLPBaselineBackbone` 直接把 k 焊进权重 shape；`KernelOperatorBackbone` 虽然支持 mask，但训练从未见过变长样本。
- **后果**：学到的其实是"16-体局部算子"，不是"任意局部 patch 算子"。真实无网格场景下，内部 / 边界 / 密度梯度区的 k 天然不等。这条不补齐，"任意点云"就是伪命题。
- **改动边界**：本阶段只改 `KernelOperatorBackbone` 的训练分布 + dataloader，**不动** `MLPBaselineBackbone`（后者明确标注为"仅同构对照，不泛化"）。

### Gap-2：旋转鲁棒性缺失

- **现象**：所有 patch 都落在 `[0,1]²` 单位方格，训练时无随机旋转增强；输入 `rel_hat` 是带方向的 2D 向量；`KernelOperatorBackbone` 无 SO(2)-equivariant 结构。
- **后果**：推理时点云整体旋转后特征分布变化，backbone 没理由保持一致。
- **表述纪律**：本 memo 采用 "rotation augmentation for robustness"，**不**主张 "rotation invariance achieved"。后者需要等变结构，留作未来工作。

### Gap-3：尺度归一不够"算子化"（隐藏 gap）

- **现象**：B4 用 `r_max` 归一化基底；但 backbone 看到的特征是 `[rel_hat, |rel_hat|, β, (rho_q)]`，其中 `rel_hat = X - x_q` **并未用 `r_max` 或局部平均节点间距归一**。β 虽然传入，但它是 teacher max-ent 的温度参数，不是长度尺度的无量纲化量。
- **后果**：backbone 仍然隐式看到绝对尺度。换一个支撑半径或密度的点云，同一个局部几何关系会产生不同的 `rel_hat` 数值，网络只能靠训练分布内的点云尺度记忆去对应。
- **与 OOD 的呼应**：这是"尺度外推 OOD 实验"之所以必要的直接原因。

### Gap-4：边界信息零输入

- **现象**：`build_node_features_torch` 只拼 `[rel_hat, dist, β, (rho_q)]`；`boundary_truncated` patch 类型只是数据分布标签，**模型看到的是"截断后的点集结果"，但没看到"导致截断的几何原因"**。
- **后果**：边界 patch 和内部稀疏 patch 在表征上不可区分，模型只能靠点分布暗示去猜。
- **补强的最小集合**：query 到边界的 signed distance、最近边界法向 `(n_x, n_y)`；暂不引入 visibility mask / level set。先少而硬。

### Gap-5：缺少 OOD 评测协议

- **现象**：当前 `train/val` 走随机 split，评测时 `train` 分布 = `val` 分布。
- **后果**：IID metrics 无法支持"不重训可用"的主张。
- **重要性升级**：先定义要证明什么，再改模型。OOD 协议定义应该**先于** Gap-1/2/3/4 的代码实现。

---

## 4. 最小改动方案

每条给出改动文件、接口、预期影响、风险。

### 4.1 Gap-5：先定义 OOD 协议（必须第一个做）

**新增文件**：
- `src/shape_function/eval/ood_eval.py`
- `configs/ood_eval.yaml`

**协议定义**（6 项）：

| 协议 | 训练分布 | 测试分布 | 预期成功判据 |
|------|---------|---------|-----------|
| IID | 7 类 patch random split | 同分布 val | 与 A3 可比 |
| LOTO（leave-one-type-out） | 6 类 patch | 留出第 7 类 | 所有指标退化 <3× IID |
| Rotation-OOD | 未旋转 | 随机旋转 30°/60°/90° | base_linear_residual 退化 <2× IID |
| k-OOD | k ∈ [12, 24] | k = 28, 32 | reproducing_residual 退化 <3× IID |
| Scale-OOD | β ∈ [0.5, 8.0] | β ∈ [8.0, 16.0] | mean_quad_residual 维持 <1e-6 |
| Boundary-OOD | 内部 patch 主导 | 纯 boundary_truncated + 新的 corner / concave | neg_fraction 不超 Target 基线 1.5× |

**新增 metrics**：每项协议都要单独输出 `ood_eval_metrics.json`，字段与 A3 eval_metrics 对齐，便于直接对比。

### 4.2 Gap-1：可变 k

**改动**：
- `src/shape_function/data/patch_sampler.py` 的 `sample_patches` 新增 `k_range: tuple[int, int] | None = None` 参数；`None` 时沿用当前固定 k 行为。
- `src/shape_function/data/dataloader.py` 的 batch collate 必须使用 padded tensor + `mask`（而不是 stack）。
- `configs/train_kernel_operator.yaml` 新增 `data.k_range: [12, 24]`。
- `MLPBaselineBackbone` 保留但**不**接入变 k，配置层面锁 `k=16` 并加注释。

**分层训练 / 测试区间**（采纳老师建议）：
- 训练主区间：`k ∈ [12, 24]` 均匀采样
- 测试插值区间：`k ∈ [14, 20]`（落在训练区间内）
- 测试外推区间：`k ∈ {28, 32}`（明确外推）

**风险**：
- B4 `eps_reg` 在小 k（k=12）下可能条件数变差。需要 A2 scan 扩展到 `k_range=[12, 24]` 重新定 `κ_max`。
- `MLPBaselineBackbone` 对照变失效。可接受：它本身就是不泛化的 ablation。

### 4.3 Gap-2：旋转增强

**改动**：
- `src/shape_function/data/patch_sampler.py` 新增 `rotation_augment: bool = False`；开启时每个 patch 在生成后、写入 pickle 前做一次随机 `θ ∈ U[0, 2π]` 的旋转（作用于 `X - x_q` 和 `x_q`）。
- `configs/train_kernel_operator.yaml` 设 `data.rotation_augment: true`。
- OOD 协议 `Rotation-OOD` 走 `rotation_augment=false` 生成，再在测试时后处理旋转指定角度。

**成本**：低（10 行代码）。

**风险**：无。这是纯数据增强。

### 4.4 Gap-3：尺度归一 backbone 输入

**改动**（`src/shape_function/models/full_model.py:14`）：

```python
def build_node_features_torch(rel_hat, beta, r_max, rho_q=None):
    scale = r_max.clamp_min(1e-12).unsqueeze(-1)  # [B, 1, 1]
    rel_tilde = rel_hat / scale                    # 无量纲相对坐标
    dist_tilde = torch.linalg.norm(rel_tilde, dim=-1, keepdim=True)
    features = [rel_tilde, dist_tilde, beta_feat]
    ...
```

命名上把 `rel_hat` 保留给"带量纲相对坐标"，`rel_tilde` 表示"无量纲"。head 的 B4 本来就用 `r_max` 归一化，这一改动是把这种做法**向前推到 backbone 入口**。

**一致性检查**：训练时 `r_max` 来自 patch 生成时保存的字段；推理时走 `rel_coords_and_radius_torch` 同一公式。单位应该唯一。

**风险**：会改变 backbone 的输入分布，**A3 三组必须重跑**。

### 4.5 Gap-4：边界特征（三维最小集）

**改动**：
- `src/shape_function/data/patch_sampler.py` 在生成 `boundary_truncated` 和其它类时，额外保存 `sdf_q`（query 到边界的 signed distance，取 `min(x_q, 1-x_q)` 等）、`boundary_normal`（2D 单位向量，指向边界外法向）。对纯内部 patch 写入一个 sentinel（如 `sdf_q = +∞`、`normal = [0, 0]`）。
- `build_node_features_torch` 追加 3 维：`[sdf_q, n_x, n_y]`（全 batch 广播到每个支撑节点）。
- `feature_dim_for_mode("minimal")` 从 4 → 7；`"enhanced"` 对应改。
- 所有 backbone 的 `input_dim` 配置需要同步更新。

**限制**：本阶段只处理简单凸域（`[0,1]²` 方形），不做 corner / concave。corner / concave 留作 Boundary-OOD 测试集扩展。

**风险**：老 pickle 数据集不带这些字段，需要重新生成或做懒惰兼容。建议一次性重跑数据生成。

---

## 5. 实验矩阵

在 Gap-1~4 全部改完后，最终实验矩阵如下。所有训练走 `unsupervised_v1` + 二阶 B4（v3 Target 设置）。

**训练组**（每个训练组跑 1 次、固定随机种子）：

| 组 | 旋转增强 | k 分布 | 输入特征 | 备注 |
|---|---------|-------|---------|-----|
| v4-A 消融：仅尺度归一 | ❌ | k=16 固定 | 含 rel_tilde，无边界 | 验证 Gap-3 独立收益 |
| v4-B 消融：加旋转 | ✅ | k=16 固定 | 含 rel_tilde，无边界 | 叠加 Gap-2 |
| v4-C 消融：加变 k | ✅ | k ∈ [12,24] | 含 rel_tilde，无边界 | 叠加 Gap-1 |
| **v4-Full** | ✅ | k ∈ [12,24] | 含 rel_tilde + 边界特征 | 最终版本 |

**评测**：每组都跑 Section 4.1 的 6 项 OOD 协议，产出 4 × 6 = 24 份 `eval_metrics.json`。

**关键汇总图**：
- Heatmap：行 = 训练组，列 = 测试协议，格子 = `base_linear_residual`（+ 另一张用 `neg_fraction`）。
- 预期看到 Full 组在 OOD 列上的退化比消融组轻。

---

## 6. 成功判据

不只是 MSE。本阶段要同时报以下 7 个量，每一项都有明确阈值。

### 数值性质（结构约束是否仍然成立）
1. `mean_pou_residual` < 1e-8（所有组、所有协议）
2. `mean_linear_residual` < 1e-8
3. `mean_quad_residual` < 1e-6（仅二阶 B4 组）
4. `fallback_rate` < 5%（按 patch_type 分层报告）

### 训练有效性（backbone 是否真的学到好的 base measure）
5. `base_linear_residual` 在 Full 组 IID 上 < 1e-2（当前 v3 Target 是 3.2e-2；这是泛化主张的"底力"）
6. `relative_correction_strength = ‖φ_corr − φ_base‖₂ / (‖φ_base‖₂ + ε)` 在 Full 组 IID 上 < 0.3

第 6 条的理由：B4 总能把最终 residual 修到数值零，只看 `mean_quad_residual` 无法区分"base 已经好"和"base 很差但被硬投影"。`relative_correction_strength` 直接测量"backbone 有多依赖 B4 兜底"——v3 Target 的 `max_neg = 1.777` 现象本质就是 correction 过强，这个量会比 residual 更早暴露 backbone 的学习质量问题。

### 负值可控性
7. `max_negative_magnitude` < 0.5（当前 v3 Target 是 1.78，是主要副作用）

**关键判据（最终主张）**：
> v4-Full 在所有 6 项 OOD 协议下，上述 7 个指标的退化均未超出阈值，且明显优于 v4-A/B/C 消融组。

只有这一条满足，才能在论文里写"该 patch-level operator 无需重训即可应用到 OOD 点云"。

---

## 7. 阶段边界

本 memo 明确 **不做** 的事：

1. **在线 patch 生成器**：离线大池子 + 强随机化足够本阶段。等 v4 跑通、论文主张落地后再考虑。
2. **SO(2)-equivariant 架构**：数据增强顶住旋转；等变结构工作量大、风险高、对当前证据链边际贡献有限。
3. **Full amortized max-ent (路线 B)**：留作 subline B。
4. **引入 warm-start Newton 修正层 (路线 C 的完整版)**：当前 B4 已经是"解析 correction 层"的简化版本，路线 C 的少步 Newton 迭代留作未来工作。
5. **更大 backbone / transformer 替换**：不在本阶段讨论；泛化瓶颈不在表达容量。
6. **非凸 / 曲边界 / corner patch**：本阶段只做凸方域边界。corner / concave 作为 Boundary-OOD 的"极端测试集"，不作为训练分布。
7. **`MLPBaselineBackbone` 泛化**：该 backbone 明确放弃泛化主张，仅作"同构对照"，锁 k=16。

---

## 附：叙事配套修改

`docs/method_design_cn.md` 需要补一段"方法定位"：

> 本方法的**网络部分不负责满足数值约束**。
> backbone 只学习"给定局部几何，什么是一个好的 base measure"；
> 结构化 head（B1-B4）通过闭式 reproducing correction 保证 PoU、一阶再生性、（可选）二阶再生性。
> 因此本方法不是"用神经网络替代 max-ent / RKPM"，而是"为 max-ent / RKPM 类方法提供一个 patch-level warm-start 算子"。

这段话不进 memo 正文，但写论文时应该放在方法论动机小节的开头。
