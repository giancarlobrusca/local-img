"""Render the desktop icon source, so the icon set is reproducible.

    source .venv/bin/activate
    python desktop/icon-source.py
    cd desktop/src-tauri && cargo tauri icon ../icon-source.png

Pillow is already a dependency of the app, so this needs nothing new.
"""

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 1024
BACKGROUND = (12, 13, 16, 255)      # --bg
ACCENT = (124, 140, 255, 255)       # --accent

image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)
draw.rounded_rectangle([0, 0, SIZE - 1, SIZE - 1], radius=224, fill=BACKGROUND)
# An aperture: a ring with a solid centre. Legible at 32px, which is the only
# size that really has to work.
draw.ellipse([232, 232, SIZE - 232, SIZE - 232], outline=ACCENT, width=56)
draw.ellipse([412, 412, SIZE - 412, SIZE - 412], fill=ACCENT)

out = Path(__file__).parent / "icon-source.png"
image.save(out)
print(f"wrote {out}")
