from pathlib import Path

p = Path("src/metinspect/cli.py")
lines = p.read_text(encoding="utf-8").splitlines()

out = []
for line in lines:
    if 'typer.echo(f"Training PatchCore baseline on category=' in line:
        out.append("    typer.echo(")
        out.append('        f"Training PatchCore baseline on category={cfg.category} "')
        out.append('        f"with {len(train_paths)} train images"')
        out.append("    )")
        continue
    if 'heat = cv2.resize(score_map.astype(np.float32)' in line:
        out.append("        heat = cv2.resize(")
        out.append("            score_map.astype(np.float32),")
        out.append("            (cfg.image_size, cfg.image_size),")
        out.append("            interpolation=cv2.INTER_LINEAR,")
        out.append("        )")
        continue
    out.append(line)

p.write_text("\n".join(out) + "\n", encoding="utf-8")
print("patched cli.py")
