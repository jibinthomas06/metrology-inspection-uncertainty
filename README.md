## Results (MVTec AD, PatchCore baseline)

Dataset-wide evaluation across 15 MVTec AD categories (backbone: resnet18). Metrics are image-level AUROC and pixel-level AUROC.

Artifacts:
- Summary table: `reports/summary_eval.md`
- Raw table (CSV): `reports/metrics_table.csv`
- Plots:
  - `reports/figures/image_auroc_by_category.png`
  - `reports/figures/pixel_auroc_by_category.png`

Reproduce (Windows / PowerShell):

```powershell
# 1) Train + eval across all categories (train is skipped if model exists)
$cats = @("bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper")
$backbone = "resnet18"
$galleryN = 12

New-Item -ItemType Directory -Force -Path .\configs\_runs | Out-Null
New-Item -ItemType Directory -Force -Path .\reports\models | Out-Null

foreach ($c in $cats) {
  $cfgPath = ".\configs\_runs\run_$c.yaml"
  Copy-Item .\configs\default.yaml $cfgPath -Force

  $content = Get-Content $cfgPath -Raw
  $pattern = '(?ms)^mvtec:\s*\r?\n(\s+.*\r?\n)*?\s+category:\s*.*$'
  $replacement = { param($m) ($m.Value -replace '(?m)^\s+category:\s*.*$', "  category: $c") }
  $content = [regex]::Replace($content, $pattern, $replacement)
  Set-Content -Path $cfgPath -Value $content -NoNewline

  $modelPath = ".\reports\models\patchcore_$c" + "_$backbone.pt"

  if (!(Test-Path $modelPath)) {
    metinspect train --config $cfgPath --backbone $backbone
  }

  metinspect eval --config $cfgPath --backbone $backbone --gallery-n $galleryN
}

# 2) Aggregate + plot
python .\scripts\aggregate_metrics.py
python .\scripts\plot_metrics.py
