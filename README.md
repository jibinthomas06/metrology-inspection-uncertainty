# metrology-inspection-uncertainty

Repo: metrology-inspection-uncertainty  
Python package: metinspect

Goal (MVTec AD first):
1) anomaly/defect heatmap + mask  
2) defect geometry (area, length/width, centroid, bbox) in pixels and millimeters  
3) uncertainty (confidence calibration + measurement uncertainty)  
4) reject option: OK / NOK / UNCERTAIN

## Setup (Windows 11 + PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
