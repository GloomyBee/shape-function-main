# shape_function CLI 命令文档

本文档记录当前仓库里实际可用的命令入口，重点面向 Windows PowerShell 使用。

建议所有命令都从仓库根目录执行：

```powershell
cd "D:\ProjectD\papers code\shape_function-main"
$env:PYTHONPATH="src"
```

如果已经用 editable install 安装过包，也可以使用短命令 `shape-function`。如果不确定安装状态，优先使用 `python -m` 形式。

---

## 1. 训练入口

当前正式 CLI 入口只有一个子命令：`train`。

两种等价调用方式：

```powershell
.venv\Scripts\python.exe -m shape_function.cli train `
  --data-config configs\data.yaml `
  --train-config configs\train_kernel_operator.yaml `
  --run-name my_run `
  --device cuda
```

或安装后：

```powershell
shape-function train `
  --data-config configs\data.yaml `
  --train-config configs\train_kernel_operator.yaml `
  --run-name my_run `
  --device cuda
```

### 1.1 参数说明

必填参数：

- `--data-config`：数据配置，例如 `configs\data.yaml`。
- `--train-config`：训练配置，例如 `configs\train_kernel_operator.yaml`。

常用可选参数：

- `--run-name`：输出目录名。若不传，CLI 自动生成。若目录已存在，训练会直接失败，不覆盖。
- `--device`：`auto | cpu | cuda`。推荐有 GPU 时显式用 `cuda`。
- `--seed`：覆盖 `data.yaml` 里的 seed。

输出目录：

```text
runs\<run-name>\
```

默认核心产物：

```text
checkpoint.pt
curves.npz
eval_metrics.json
config_snapshot.yaml
figures\
```

---

## 2. 常用训练配置

### 2.1 默认二阶无 teacher Target

配置：

```text
configs\train_kernel_operator.yaml
```

含义：

- backbone：`kernel_operator`
- loss：`unsupervised_v1`
- head：`basis_order=2`
- `kappa_max=1e8`

命令：

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m shape_function.cli train `
  --data-config configs\data.yaml `
  --train-config configs\train_kernel_operator.yaml `
  --run-name target_unsup_o2 `
  --device cuda
```

### 2.2 legacy teacher Baseline

配置：

```text
configs\train_kernel_operator_legacy_teacher.yaml
```

含义：

- teacher：max-ent reference
- loss：`loss_data + loss_cons + loss_neg`
- head：一阶 B4
- 用途：只作为 A3 Baseline 对照，不是当前默认主线。

命令：

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m shape_function.cli train `
  --data-config configs\data.yaml `
  --train-config configs\train_kernel_operator_legacy_teacher.yaml `
  --run-name baseline_legacy_teacher `
  --device cuda
```

### 2.3 一阶无 teacher Abl-1

配置：

```text
configs\train_kernel_operator_unsup_o1.yaml
```

含义：

- loss：`unsupervised_v1`
- head：`basis_order=1`
- 用途：隔离“去 teacher loss”本身的影响。

命令：

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m shape_function.cli train `
  --data-config configs\data.yaml `
  --train-config configs\train_kernel_operator_unsup_o1.yaml `
  --run-name abl1_unsup_o1 `
  --device cuda
```

---

## 3. 当前推荐短实验命令

### 3.1 修复 metrics 后的 A3 三组对照

这三组用于公平比较：

- Baseline：teacher + O1
- Abl-1：unsupervised + O1
- Target：unsupervised + O2

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m shape_function.cli train `
  --data-config configs\data.yaml `
  --train-config configs\train_kernel_operator_legacy_teacher.yaml `
  --run-name a3_baseline_legacy_teacher_quadfix `
  --device cuda
```

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m shape_function.cli train `
  --data-config configs\data.yaml `
  --train-config configs\train_kernel_operator_unsup_o1.yaml `
  --run-name a3_abl1_unsup_o1_quadfix `
  --device cuda
```

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m shape_function.cli train `
  --data-config configs\data.yaml `
  --train-config configs\train_kernel_operator.yaml `
  --run-name a3_target_unsup_o2_quadfix `
  --device cuda
```

### 3.2 `kappa_max` 敏感性短对照

这三组只改 `kappa_max`，用于观察 fallback 是否能压住二阶 B4 的负值放大。

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m shape_function.cli train `
  --data-config configs\data.yaml `
  --train-config configs\train_kernel_operator_kappa_k1e4.yaml `
  --run-name a3_target_unsup_o2_kappa_1e4_quadfix `
  --device cuda
```

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m shape_function.cli train `
  --data-config configs\data.yaml `
  --train-config configs\train_kernel_operator_kappa_k1e5.yaml `
  --run-name a3_target_unsup_o2_kappa_1e5_quadfix `
  --device cuda
```

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m shape_function.cli train `
  --data-config configs\data.yaml `
  --train-config configs\train_kernel_operator_kappa_k1e6.yaml `
  --run-name a3_target_unsup_o2_kappa_1e6_quadfix `
  --device cuda
```

---

## 4. 指标汇总命令

### 4.1 汇总 A3 三组和 kappa 敏感性结果

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -c "import json, pathlib; runs=['a3_baseline_legacy_teacher_quadfix','a3_abl1_unsup_o1_quadfix','a3_target_unsup_o2_quadfix','a3_target_unsup_o2_kappa_1e4_quadfix','a3_target_unsup_o2_kappa_1e5_quadfix','a3_target_unsup_o2_kappa_1e6_quadfix']; keys=['relative_l2','base_linear_residual','mean_linear_residual','mean_quad_residual','negative_fraction','max_negative_magnitude','fallback_rate','mean_cond_M','p95_cond_M','worst_cond_M','teacher_quad_residual','quad_gain'];
for r in runs:
    p=pathlib.Path('runs')/r/'eval_metrics.json'
    print('\n'+r)
    d=json.loads(p.read_text())
    for k in keys:
        if k in d: print(f'{k}: {d[k]}')"
```

重点看：

- `mean_quad_residual`：二阶一致性误差。
- `negative_fraction`：负值比例。
- `max_negative_magnitude`：最大负值幅度。
- `fallback_rate`：二阶 B4 回退到一阶 B4 的比例。
- `mean_cond_M / p95_cond_M / worst_cond_M`：moment matrix 条件数。

---

## 5. A2 条件数扫描

`scan_cond_m2.py` 当前是 helper 模块，不是 argparse CLI。使用 `python -c` 调用。

### 5.1 小扫描

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -c "from pathlib import Path; from shape_function.eval.scan_cond_m2 import scan_cond_m2, save_scan_results; payload = scan_cond_m2(num_patches=700, seed=42, k_neighbors=16, beta_range=(1.0, 1.0), kappa_max=1.0e8); save_scan_results(Path('runs/a2_cond_m2_scan_small/cond_m2_stats.json'), payload); print(payload['overall']); print(payload['by_patch_type'])"
```

### 5.2 7k 扫描

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -c "from pathlib import Path; from shape_function.eval.scan_cond_m2 import scan_cond_m2, save_scan_results; payload = scan_cond_m2(num_patches=7000, seed=42, k_neighbors=16, beta_range=(1.0, 1.0), kappa_max=1.0e8); save_scan_results(Path('runs/a2_cond_m2_scan_7k/cond_m2_stats.json'), payload); print(payload['overall']); print(payload['by_patch_type'])"
```

输出文件：

```text
runs\a2_cond_m2_scan_7k\cond_m2_stats.json
```

---

## 6. 绘图命令

### 6.1 训练曲线

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m shape_function.eval.visualize_run `
  --run-dir runs\a3_target_unsup_o2_quadfix `
  --device cuda `
  --view training_curves
```

输出：

```text
runs\a3_target_unsup_o2_quadfix\figures\training_curves\training_curves.png
```

### 6.2 basis contour 云图

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m shape_function.eval.visualize_run `
  --run-dir runs\a3_target_unsup_o2_quadfix `
  --device cuda `
  --view basis_contour `
  --selection spread `
  --num-patches 3
```

### 6.3 basis hybrid3d 三维图

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m shape_function.eval.visualize_run `
  --run-dir runs\a3_target_unsup_o2_quadfix `
  --device cuda `
  --view basis_hybrid3d `
  --selection spread `
  --num-patches 3
```

### 6.4 其它 view

`visualize_run` 支持以下 view：

```text
training_curves
bars
contour
hybrid3d
basis_contour
basis_hybrid3d
```

当前推荐优先使用：

```text
basis_contour
basis_hybrid3d
training_curves
```

注意：无 teacher run 的 patch 图会在绘图阶段临时补算 teacher reference；若个别 teacher solve 失败，脚本会跳过对应 patch，并在 summary 里记录。

---

## 7. 均布点 RKPM 对比图

用于画：

- uniform 4x4 patch
- RKPM quadratic reference
- Neural O2
- basis surface 对比
- 二阶 residual 云图

命令：

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m shape_function.eval.visualize_uniform_rkpm_comparison `
  --run-dir runs\a3_target_unsup_o2_quadfix `
  --device cuda `
  --beta 3.0 `
  --grid-size 101 `
  --batch-size 1024
```

输出：

```text
runs\a3_target_unsup_o2_quadfix\figures\uniform_rkpm_comparison\uniform_rkpm_comparison.png
runs\a3_target_unsup_o2_quadfix\figures\uniform_rkpm_comparison\summary.json
```

可选参数：

- `--beta`：控制 RKPM / neural 输入里的 beta。
- `--grid-size`：绘图网格密度，越大越细，但越慢。
- `--node-index`：指定画哪个节点的 basis function；不传则自动选代表性节点。

---

## 8. 生产配置训练

生产数据配置：

```text
configs\data_production.yaml
```

当前不建议在短诊断没完成前直接长跑 production。若要跑：

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m shape_function.cli train `
  --data-config configs\data_production.yaml `
  --train-config configs\train_kernel_operator.yaml `
  --run-name prod_target_unsup_o2_kernel `
  --device cuda
```

注意：

- production 数据量更大，构数和训练都会更慢。
- 若 run 目录已存在，CLI 会失败，不会覆盖。
- 建议先完成 `lambda_neg` 和 `kappa_max` 的短对照后再跑 production。

---

## 9. 常见问题

### 9.1 PowerShell 多行命令能直接复制吗？

可以。PowerShell 使用反引号 `` ` `` 续行。注意反引号后面不要有空格。

### 9.2 run 目录已存在怎么办？

CLI 不覆盖已有目录。换一个 `--run-name`，例如加后缀：

```text
_run2
_quadfix
_kappa_1e4
```

不要手动删除重要 run，除非确认里面没有要保留的结果。

### 9.3 为什么训练输出路径有时不在仓库 runs 里？

如果在 sandbox 环境里运行且没有提升权限，产物可能写到 sandbox cwd。正常用户自己在终端运行不会有这个问题。Codex 代跑时需要确认 artifacts_dir 是否是：

```text
D:\ProjectD\papers code\shape_function-main\runs\...
```

### 9.4 当前最推荐跑什么？

当前阶段推荐优先跑：

1. A3 三组 quadfix 对照
2. `kappa_max` 敏感性
3. `lambda_neg` 敏感性
4. 中等规模训练
5. production 训练

---

## 10. 当前已跑过的关键结果目录

```text
runs\a3_baseline_legacy_teacher_quadfix
runs\a3_abl1_unsup_o1_quadfix
runs\a3_target_unsup_o2_quadfix
runs\a3_target_unsup_o2_kappa_1e4_quadfix
runs\a3_target_unsup_o2_kappa_1e5_quadfix
runs\a3_target_unsup_o2_kappa_1e6_quadfix
runs\a3_quadfix_summary
```

汇总图：

```text
runs\a3_quadfix_summary\a3_quadfix_summary.png
```

汇总 JSON：

```text
runs\a3_quadfix_summary\summary_metrics.json
```
