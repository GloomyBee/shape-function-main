# Shape Function 汇报 PPT

这里是整理后的正式汇报目录，入口已经从旧的
`demo/shape_function_report/...` 深层路径收口到当前目录。

## 目录结构

- `shape_function_report.tex`
  LaTeX 主文件
- `figures/`
  汇报所用图片素材
- `build.ps1`
  编译脚本，中间文件只写到 `build/`
- `shape_function_report.pdf`
  最终导出的 PDF，直接打开这个文件即可

## 编译方式

在 PowerShell 中运行：

```powershell
cd docs\mtheme-master\shape_function_report
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

## 输出约定

- 临时文件目录：

```text
docs\mtheme-master\shape_function_report\build
```

- 最终 PDF：

```text
docs\mtheme-master\shape_function_report\shape_function_report.pdf
```

脚本会在编译成功后把最终 PDF 复制到当前目录，并清空 `build/`
里的中间文件，避免再去深层目录找结果。
