# -*- coding: utf-8 -*-
"""重新生成透明底 Logo: 三层荧光切片 + 蓝色轨道环。输出 assets/logo.png(透明) + logo.ico。"""
import math, os
from PIL import Image, ImageDraw, ImageFilter

BASE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE, "assets")
os.makedirs(ASSETS, exist_ok=True)

W, H = 1024, 1024
img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(img, "RGBA")
cx, cy = W // 2, H // 2

def ring(draw, rx, ry, width, gap_angles, color1, color2, glow=True):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g = ImageDraw.Draw(layer)
    for a in range(0, 360):
        skip = any(s <= a <= e for s, e in gap_angles)
        if skip:
            continue
        t = a / 360.0
        c = tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))
        x = cx + rx * math.cos(math.radians(a))
        y = cy + ry * math.sin(math.radians(a))
        g.ellipse([x - width, y - width, x + width, y + width], fill=c + (255,))
    if glow:
        layer = layer.filter(ImageFilter.GaussianBlur(3))
    draw.bitmap((0, 0), layer)

# 外圈纵环 + 横向椭圆轨道(带缺口, 还原原 Logo)
ring(d, 400, 430, 22, [(40, 70), (200, 225)], (20, 90, 255), (40, 200, 255))
ring(d, 440, 190, 26, [(150, 200), (330, 30)], (15, 80, 230), (60, 210, 255))

# 三层切片
panels = [
    {"x": 200, "edge": (200, 200, 205, 255), "fill": (12, 12, 16, 255), "glow": (160, 160, 165, 255)},
    {"x": 400, "edge": (255, 90, 90, 255), "fill": (90, 8, 10, 255), "glow": (255, 70, 40, 255)},
    {"x": 600, "edge": (255, 70, 80, 255), "fill": (110, 10, 18, 255), "glow": (255, 40, 60, 255)},
]
for p in panels:
    x0, y0, x1, y1 = p["x"], 250, p["x"] + 230, 790
    # 投影
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([x0 + 10, y0 + 16, x1 + 10, y1 + 16], 28, fill=(0, 0, 0, 110))
    img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(10)))
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle([x0, y0, x1, y1], 28, fill=p["fill"], outline=p["edge"], width=7)
    # 内部荧光纹理(随机亮斑, 模拟组织)
    import random
    random.seed(hash(p["x"]) % 999)
    for _ in range(90):
        bx = random.randint(x0 + 25, x1 - 25)
        by = random.randint(y0 + 25, y1 - 25)
        r = random.randint(6, 26)
        alpha = random.randint(60, 190)
        ImageDraw.Draw(img, "RGBA").ellipse([bx - r, by - r, bx + r, by + r], fill=p["glow"][:3] + (alpha,))

out_png = os.path.join(ASSETS, "logo.png")
img.save(out_png)
img.save(os.path.join(ASSETS, "logo.ico"), sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])
print(f"Saved {out_png} (transparent RGBA, {img.size})")
