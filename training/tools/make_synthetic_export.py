"""Creates a small fake app export (coloured pill shapes) to smoke-test the training scripts.

python tools/make_synthetic_export.py <output-folder>
"""

import json
import random
import sys
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

out = Path(sys.argv[1])
(out / "crops").mkdir(parents=True, exist_ok=True)
rng = random.Random(0)
meds = {
    "metformin-500": ((240, 240, 235), "oval"),
    "atorvastatin-20": ((250, 250, 250), "round"),
    "aspirin-100": ((235, 180, 60), "round"),
    "perindopril-5": ((120, 200, 120), "oval"),
    "amlodipine-5": ((245, 245, 245), "capsule"),
}
lines = []
for med, (colour, shape) in meds.items():
    for _ in range(3):
        group = str(uuid.uuid4())
        for _ in range(6):
            img = Image.new("RGB", (96, 96), (30, 32, 36))
            d = ImageDraw.Draw(img)
            jitter = tuple(max(0, min(255, c + rng.randint(-12, 12))) for c in colour)
            w, h = {"oval": (60, 36), "round": (44, 44), "capsule": (70, 28)}[shape]
            x0, y0 = 48 - w // 2 + rng.randint(-4, 4), 48 - h // 2 + rng.randint(-4, 4)
            d.ellipse([x0, y0, x0 + w, y0 + h], fill=jitter)
            img = img.rotate(rng.uniform(0, 360)).filter(ImageFilter.GaussianBlur(rng.uniform(0, 1)))
            name = f"{uuid.uuid4()}.jpg"
            img.save(out / "crops" / name, quality=90)
            lines.append(
                {
                    "crop": f"crops/{name}",
                    "medicationID": med,
                    "source": "teaching",
                    "groupID": group,
                    "createdAt": "2026-10-09T00:00:00.000Z",
                }
            )
(out / "labels.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")
print(len(lines), "crops")
