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
}
finally {
    Pop-Location
}
