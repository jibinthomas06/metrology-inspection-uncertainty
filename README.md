# metrology-inspection-uncertainty

Repo: metrology-inspection-uncertainty  
Python package: metinspect

Goal (MVTec AD first):
1) anomaly/defect heatmap + mask  
2) defect geometry (area, length/width, centroid, bbox) in pixels and millimeters  
3) uncertainty (confidence calibration + measurement uncertainty)  
4) reject option: OK / NOK / UNCERTAIN




## Results (MVTec AD, PatchCore baseline)

Dataset-wide evaluation across 15 MVTec AD categories (backbone: resnet18). Metrics are image-level AUROC and pixel-level AUROC.

Artifacts:
- Summary table: `reports/summary_eval.md`
- Raw table (CSV): `reports/metrics_table.csv`
- Plots:
  - `reports/figures/image_auroc_by_category.png`
  - `reports/figures/pixel_auroc_by_category.png`

Reproduce:
```bash
# train + eval (per category)
metinspect train --config configs/default.yaml --backbone resnet18
metinspect eval  --config configs/default.yaml --backbone resnet18 --gallery-n 12

# aggregate + plot
python scripts/aggregate_metrics.py
python scripts/plot_metrics.py

## Setup (Windows 11 + PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
