# Interpretation (PatchCore baseline on MVTec AD)

Backbone: resnet18  
Metrics: image AUROC (detection) and pixel AUROC (localization)

## What looks strong
- **bottle, hazelnut, wood, toothbrush**: very high image AUROC; defects are visually distinct relative to normal texture/shape.

## What looks hard (and why)
- **grid (image AUROC 0.6959)**: repetitive texture; small defects can be confused with normal variation.
- **pill (0.7490)**: subtle anomalies + low contrast; “normal” variation across samples is high.
- **screw (0.7438)**: small objects; defects occupy tiny area, strong sensitivity to alignment/pose.
- **tile (0.9134 image, pixel 0.8175)**: detection good but localization weaker; boundaries can be ambiguous.
- **zipper (0.8361)**: thin structures, occlusions, and fine-grained defects.

## Where to look (qualitative galleries)
- `reports/figures/gallery_bottle_resnet18/`
- `reports/figures/gallery_grid_resnet18/`
- `reports/figures/gallery_pill_resnet18/`
- `reports/figures/gallery_screw_resnet18/`
- `reports/figures/gallery_tile_resnet18/`
- `reports/figures/gallery_zipper_resnet18/`

## Next improvement to try
- Add a simple calibration step on anomaly scores (temperature scaling or isotonic) and report:
  - expected calibration error (ECE)
  - reliability diagram
- Add “reject option”: OK / NOK / UNCERTAIN based on calibrated probability bands.
