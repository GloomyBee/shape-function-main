# Shape Function 汇报 PPT

该目录是基于 `mtheme` 模板单独开出的汇报目录，不修改模板根目录文件。

## 结构

- `shape_function_report.tex`：主 PPT 文件
- `figures/`：从 `runs/` 复制过来的汇报图
- `build.ps1`：把编译产物定向到本目录 `build/`

## 编译

在 PowerShell 下执行：

```powershell
cd docs\mtheme-master\demo\shape_function_report
.\build.ps1
```

输出目录：

```text
docs\mtheme-master\demo\shape_function_report\build
```

主 PDF：

```text
docs\mtheme-master\demo\shape_function_report\build\shape_function_report.pdf
```
