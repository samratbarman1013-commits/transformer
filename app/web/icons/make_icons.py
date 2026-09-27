"""Generate PWA icons (gradient rounded square with a white 'T')."""
from PIL import Image, ImageDraw

C1, C2 = (59, 48, 210), (168, 85, 247)  # deep indigo -> bright purple


def gradient(size):
    img = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(img)
    for y in range(size):
        t = y / (size - 1)
        color = tuple(round(a + (b - a) * t) for a, b in zip(C1, C2))
        d.line([(0, y), (size, y)], fill=color)
    return img


def draw_T(d, box, thick, color=(255, 255, 255)):
    x0, y0, x1, y1 = box
    w = x1 - x0
    d.rectangle([x0, y0, x1, y0 + thick], fill=color)                      # top bar
    d.rectangle([x0 + w / 2 - thick / 2, y0, x0 + w / 2 + thick / 2, y1], fill=color)  # stem


def rounded(img, radius):
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, *img.size], radius=radius, fill=255)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


base = gradient(512)

# Regular icons: rounded square, T occupying most of the canvas.
regular = base.copy()
draw_T(ImageDraw.Draw(regular), (96, 120, 416, 392), 56)
for size, name in [(192, "icon-192.png"), (512, "icon-512.png")]:
    img = regular.resize((size, size), Image.LANCZOS)
    img = rounded(img, round(size * 110 / 512))
    img.save(f"icons/{name}")

# Maskable icon: full-bleed (no rounding), T inside the safe zone.
maskable = gradient(512)
draw_T(ImageDraw.Draw(maskable), (128, 160, 384, 360), 48)
maskable.save("icons/icon-maskable-512.png")

print("icons written")
