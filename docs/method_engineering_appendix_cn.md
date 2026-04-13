# 方法工程附录（实现映射、开放问题与推进顺序）

## 1. 文档目的

本附录不是论文正文，而是主方法文档的工程支撑材料。它回答四类问题：

1. 当前方法在代码里分别落在哪些模块。
2. 当前实现与技术规范相比有哪些差异。
3. 还存在哪些未决设计点与待验证假设。
4. 后续工作应按什么顺序推进，为什么这样排。

主文档负责讲“方法是什么、为什么这样设计”；本附录负责讲“当前代码怎么实现、哪里还不闭合、接下来先修什么”。

---

## 2. 代码映射

### 2.1 数据工厂与 teacher

- `src/shape_function/data/patch_sampler.py`
  - 负责局部 patch 采样。
  - 当前实现了 7 类 patch family。
  - 生成 `x_q, X, beta, r_max, patch_type` 等基础字段。

- `src/shape_function/data/patch_validator.py`
  - 负责剔除几何退化 patch。
  - 检查近重复节点、秩亏、病态协方差等问题。

- `src/shape_function/data/feature_builder.py`
  - 负责构造 `node_features` 与 `rho_q`。
  - 支持 `minimal` 与 `enhanced` 两种特征模式。

- `src/shape_function/data/maxent_solver.py`
  - 负责 patch-level max-ent teacher 求解。
  - 当前支持 `gaussian` 与 `quartic_spline` prior。
  - 输出 `phi_ref`、状态码、诊断量与残差。

- `src/shape_function/data/dataset_builder.py`
  - 串联采样、teacher 和 feature construction。
  - 当前已支持从配置传入 `patch_types` 与 `beta_range`。

### 2.2 模型层

- `src/shape_function/models/full_model.py`
  - 总装模型。
  - 负责 query-centered 几何归一化、节点特征构造、backbone 调用与 head 调用。

- `src/shape_function/models/backbones/kernel_operator.py`
  - 当前主 backbone。
  - 实现 kernel-modulated message passing。

- `src/shape_function/models/backbones/mlp_baseline.py`
  - 当前对照 backbone。
  - 实现基于距离排序的固定长度 MLP 回归。

- `src/shape_function/models/heads/structure_head.py`
  - 实现 B1--B4 output head 主流程。

- `src/shape_function/models/heads/reproducing_correction.py`
  - 实现基于 `phi_base` 的线性 reproducing correction。

- `src/shape_function/models/heads/windows.py`
  - 实现 quartic spline 与 Wendland C2 窗口函数。

### 2.3 训练、评估与 CLI

- `src/shape_function/train/losses.py`
  - 实现 `loss_data + loss_cons + loss_neg`。

- `src/shape_function/train/metrics.py`
  - 实现误差、一致性、负值比例与条件数诊断。

- `src/shape_function/train/trainer.py`
  - 实现 epoch 级 train/val pass、best-model 保存与 best 权重恢复。
  - 当前已接入 cosine annealing scheduler，并记录学习率曲线。

- `src/shape_function/eval/eval_backbones.py`
  - 对训练后的模型做指标汇总。

- `src/shape_function/cli/config.py`
  - 负责 YAML 配置解析、校验与合并。

- `src/shape_function/cli/train.py`
  - 负责 CLI 训练入口。
  - 串起配置加载、数据构建、训练、评估、checkpoint 和 config snapshot 保存。

### 2.4 配置与产物

- `configs/data.yaml`
  - 当前默认数据配置。
  - 仍然是 scaffold 规模。

- `configs/data_production.yaml`
  - 当前生产训练配置。
  - 默认包含全部 7 类 patch type 与 20k/2k 的 train/val 规模。

- `configs/train_kernel_operator.yaml`
  - kernel operator 训练配置。

- `configs/train_mlp.yaml`
  - MLP baseline 训练配置。

- `runs/<run_name>/`
  - 当前 run artifact 目录。
  - 默认已包含 `metrics.json`, `summary.txt`, `curves.npz`, `checkpoint.pt`, `best_model.pt`, `config_snapshot.yaml`, `eval_metrics.json`。

---

## 3. 当前实现与规范差异

### 3.1 CLI 闭环已补齐

与早期仅有函数库的状态相比，当前实现已经具备完整的 CLI 训练入口：

- 支持 `python -m shape_function.cli train ...`
- 支持 `shape-function train ...`
- 支持 YAML 配置解析、run_name 生成、artifact 落盘

这意味着“如何端到端使用仓库”这一工程缺口已经补上，不再是当前主要问题。

### 3.2 trainer 闭环已补齐

当前 `trainer.py` 已经具备：

1. best validation checkpoint 保存；
2. 训练结束后恢复 best 权重；
3. cosine annealing learning-rate scheduler；
4. `history["lr"]` 学习率曲线记录。

因此当前差异不再是“trainer 缺关键机制”，而是“这些机制是否已在生产规模训练中得到充分验证”。

### 3.3 默认数据配置仍是 scaffold

当前 `configs/data.yaml`：

- `num_train = 256`
- `num_val = 64`
- 默认 patch type 只有 5 类

这套配置适合：

- 单元测试；
- 快速 smoke training；
- CLI 功能验证。

但它不适合：

- backbone 真正对比；
- 论文结果生成；
- failure case 研究。

为此，当前仓库已经新增 `configs/data_production.yaml`，作为生产规模训练入口。也就是说，配置分层已经建立，但真实实验结果还未生成。

### 3.4 快速配置与生产配置的 patch 覆盖范围不同

虽然采样器中已实现：

- `anisotropic`
- `sparse_dense_transition`

但默认快速配置 `data.yaml` 尚未纳入这两类。这会导致默认训练分布偏乐观，不能完整覆盖当前最值得关心的困难 patch。相对地，`data_production.yaml` 已经覆盖全部 7 类 patch，用于后续真实训练。

### 3.5 方法规范与当前 head 行为间的张力

从规范角度看，B2 使用紧支撑窗口函数是合理的；从实现角度看，当前已经把窗口半径与 backbone 归一化尺度显式拆开，因此：

- `r_max` 继续只承担输入归一化；
- `support_radius_scale` 默认取 `1.05`，用于避免最远邻居系统性零权重；
- 但默认 Gaussian teacher 与紧支撑 head 之间仍可能存在轻微可表示族失配。

这意味着最突出的实现缺陷已经修正，但仍保留一个需要实验验证的方法一致性问题。

---

## 4. 开放问题与待验证假设

### 4.1 `support radius` 是否应放大

当前状态：

- backbone 归一化尺度与 B2 窗口支撑半径已经拆开；
- 当前默认 `support_radius_scale = 1.05`；
- 最远邻居不再被系统性丢弃。

当前待验证方向：

1. 默认 `\alpha = 1.05` 是否足够；
2. 是否需要把 `support_radius_scale` 暴露到配置层；
3. 不同 window type 下最优 `\alpha` 是否一致。

待验证标准：

- 最远邻居不再系统性恒为 0；
- POU 和线性一致性不退化；
- relative L2 改善或至少不恶化；
- `cond_M` 和负值比例不显著变差。

### 4.2 teacher prior 与 head locality 是否需要统一

当前默认组合是：

- teacher：Gaussian prior
- head：紧支撑 window

潜在问题：

- teacher 与 student 的可表示族不完全一致

可选方向：

1. 不改 teacher，只放宽 head 的支撑半径；
2. 改 teacher prior，使之更接近紧支撑局部权函数；
3. 二者都保持现状，但把 mismatch 作为已知设计选择。

当前推荐方向是优先采用第 1 种，因为它改动最小，也最不影响现有训练叙事。

### 4.3 数据缓存是否应进入 CLI v2

当前 CLI 每次训练都会在线生成 train/val 数据。这在 scaffold 规模下没有问题，但在生产规模下会变成：

- 启动成本增加；
- 结果复现实验需要重复采样；
- sampler 代码变动会影响“同一配置”的数据分布。

待决策点：

- 是新增 `generate` 子命令；
- 还是给 `train` 增加 `--data-cache` 选项。

当前判断：它是重要优化项，但优先级晚于真实训练与基础对比实验。

### 4.4 OOD 数据生成应放在哪一阶段

OOD 数据集在论文里很重要，但当前是否立刻实现仍有策略问题。

更合理的顺序是：

1. 先让主训练闭环与基础方法定义稳定；
2. 再单独生成几何 OOD 与 $\beta$ OOD；
3. 再进入更系统的泛化分析与可视化整理。

因此 OOD 目前应被视作第二阶段实验任务，而不是第一批代码修正任务。

---

## 5. 推进优先级

### Priority 1：生产训练配置与首轮真实训练

目标：

- 把已经补齐的 CLI、head 与 trainer 真正推到生产规模数据上；
- 形成第一批可分析的实验信号。

动作：

1. 使用 `configs/data_production.yaml` 跑 `kernel_operator + full head`；
2. 记录 train/val 曲线、best checkpoint 与评估指标；
3. 确认 `loss_data`、`loss_cons`、`relative_l2` 的基本走势。

验收标准：

- 训练过程无 NaN / 发散；
- `loss_cons` 保持近零；
- run artifact 完整可用。

### Priority 2：基线对比与方法验证

目标：

- 建立第一轮 backbone 对比结论；
- 验证当前方法设计是否有实际收益。

动作：

1. 运行 `mlp_baseline + full head`；
2. 与 `kernel_operator + full head` 对比；
3. 检查相对误差、负值比例与条件数诊断。

验收标准：

- 至少保留两组可复现实验 artifact；
- 能形成初步的 backbone 优劣判断。

### Priority 3：方法细化实验

目标：

- 把“实现已闭环”进一步推进到“方法叙事已验证”。

动作：

1. 做 head ablation；
2. 检查 `support_radius_scale` 对误差与诊断量的影响；
3. 评估 teacher/head mismatch 是否仍显著。

验收标准：

- 能回答“B4 是否必要”“当前 support radius 是否合理”这两类方法问题。

### Priority 4：增强工程与评估

1. OOD 数据与评估；
2. dataset caching / generate 子命令；
3. visualization。

原因：

这些工作会增强实验完整性，但不是当前最早的科学风险控制项。

---

## 6. 文档与实现如何协同

本附录的用法应当固定：

- 主文档 `method_design_cn.md` 是论文写作底稿；
- 本附录是开发与实验路线图；
- 每次方法定义发生变化时，先更新主文档中的方法叙事，再在本附录中更新“实现差异”和“未决项”；
- 每次阶段性工程完成后，在本附录中移动优先级顺序，而不是反复重写主文档。

这样做的好处是：

- 论文叙事和工程现状不会混在一起；
- 文档不会随着实现细节波动而失去稳定性；
- 任何时候都能清楚地区分“已稳定的方法定义”和“仍在推进的工程任务”。
