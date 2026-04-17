param(
    [string]$MainFile = "shape_function_report.tex"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildDir = Join-Path $root "build"

New-Item -ItemType Directory -Force $buildDir | Out-Null

Push-Location $root
try {
    latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir="$buildDir" $MainFile
    $generatedPdf = Join-Path $buildDir "shape_function_report.pdf"
    $finalPdf = Join-Path $root "shape_function_report.pdf"
    if (Test-Path -LiteralPath $generatedPdf) {
        Copy-Item -LiteralPath $generatedPdf -Destination $finalPdf -Force
        Get-ChildItem -LiteralPath $buildDir -Force |
            Remove-Item -Recurse -Force
    }
}
finally {
    Pop-Location
}
