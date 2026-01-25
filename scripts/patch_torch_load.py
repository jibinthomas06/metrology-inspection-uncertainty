from pathlib import Path

p = Path("src/metinspect/models/patchcore.py")
s = p.read_text(encoding="utf-8")

old = 'torch.load(path, map_location="cpu")'
new = 'torch.load(path, map_location="cpu", weights_only=False)'

if old not in s:
    raise SystemExit("pattern not found, aborting")

p.write_text(s.replace(old, new), encoding="utf-8")
print("patched patchcore.py")
