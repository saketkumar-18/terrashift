# Generate TerraShift PWA icons (192/512/maskable/apple-touch/favicon) via Pillow.
import os
import sys

from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
os.makedirs(OUT, exist_ok=True)

BG = (11, 15, 20, 255)        # #0b0f14
ACC = (76, 201, 240, 255)     # #4cc9f0 water blue
ACC2 = (255, 107, 53, 255)    # #ff6b35 change orange
GREEN = (26, 122, 60, 255)    # forest green


def draw_icon(size, maskable=False):
    pad = int(size * 0.12) if maskable else 0
    s = size - 2 * pad
    im = Image.new("RGBA", (size, size), BG)
    d = ImageDraw.Draw(im)
    # globe: circle outline
    cx, cy, r = size // 2, size // 2, int(s * 0.36)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ACC, width=max(2, size // 40))
    # meridians (arc hints)
    d.arc([cx - r, cy - r, cx + r, cy + r], 90, 270, fill=ACC, width=max(1, size // 60))
    # change arrow: rising zigzag across the globe
    w = max(3, size // 22)
    pts = [(cx - int(r * 0.7), cy + int(r * 0.55)),
           (cx - int(r * 0.1), cy - int(r * 0.05)),
           (cx + int(r * 0.25), cy + int(r * 0.35)),
           (cx + int(r * 0.8), cy - int(r * 0.65))]
    d.line(pts, fill=ACC2, width=w, joint="curve")
    # arrowhead
    hx, hy = pts[-1]
    d.polygon([(hx, hy), (hx - int(size * 0.09), hy + int(size * 0.02)),
               (hx - int(size * 0.03), hy + int(size * 0.09))], fill=ACC2)
    # small forest triangle at bottom-left
    fx, fy = cx - int(r * 1.05), cy + int(r * 0.8)
    d.polygon([(fx, fy), (fx + int(size * 0.07), fy - int(size * 0.12)),
               (fx + int(size * 0.14), fy)], fill=GREEN)
    return im


for name, size, maskable in [
    ("icon-192.png", 192, False),
    ("icon-512.png", 512, False),
    ("icon-maskable-512.png", 512, True),
    ("apple-touch-icon.png", 180, False),
    ("favicon.png", 64, False),
]:
    draw_icon(size, maskable).save(os.path.join(OUT, name))
    print("wrote", name)

# favicon.ico (16/32/48)
ico = draw_icon(48, False)
ico.save(os.path.join(OUT, "favicon.ico"), sizes=[(16, 16), (32, 32), (48, 48)])
print("wrote favicon.ico")
