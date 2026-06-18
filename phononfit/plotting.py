from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .fitting import LorentzFit


INK = (31, 41, 55)
MUTED = (92, 107, 125)
GRID = (224, 230, 238)
SPECTRUM = (44, 95, 138)
FIT = (226, 99, 35)
ACCENT = (89, 105, 125)


def plot_fit(
    frequency: np.ndarray,
    intensity: np.ndarray,
    fit: LorentzFit,
    output: str | Path,
    *,
    title: str = "Lorentz fit of a synthetic SED peak",
) -> Path:
    """Write a polished spectrum/fit diagnostic figure with Pillow."""
    x = np.asarray(frequency, dtype=float)
    y = np.asarray(intensity, dtype=float)
    fitted = fit.evaluate(x)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1400, 820
    image = Image.new("RGB", (width, height), (247, 249, 252))
    draw = ImageDraw.Draw(image)
    plot = (135, 175, 1320, 665)
    draw.rounded_rectangle((70, 140, 1350, 715), radius=24, fill="white", outline=(235, 239, 244), width=2)

    xmin, xmax = float(x.min()), float(x.max())
    ymin = float(min(y.min(), fitted.min()))
    ymax = float(max(y.max(), fitted.max()))
    padding = max((ymax - ymin) * 0.08, 1e-12)
    ymin -= padding
    ymax += padding

    def point(xvalue: float, yvalue: float) -> tuple[float, float]:
        px = plot[0] + (xvalue - xmin) / (xmax - xmin) * (plot[2] - plot[0])
        py = plot[3] - (yvalue - ymin) / (ymax - ymin) * (plot[3] - plot[1])
        return px, py

    x_ticks = np.linspace(xmin, xmax, 6)
    y_ticks = np.linspace(ymin, ymax, 5)
    for xvalue in x_ticks:
        px, _ = point(float(xvalue), ymin)
        draw.line((px, plot[1], px, plot[3]), fill=GRID, width=2)
        label = f"{xvalue:.2f}"
        bbox = draw.textbbox((0, 0), label, font=_font(20))
        draw.text((px - (bbox[2] - bbox[0]) / 2, plot[3] + 16), label, fill=MUTED, font=_font(20))
    for yvalue in y_ticks:
        _, py = point(xmin, float(yvalue))
        draw.line((plot[0], py, plot[2], py), fill=GRID, width=2)
        label = f"{yvalue:.2f}"
        bbox = draw.textbbox((0, 0), label, font=_font(20))
        draw.text((plot[0] - (bbox[2] - bbox[0]) - 16, py - 11), label, fill=MUTED, font=_font(20))

    draw.line((plot[0], plot[1], plot[0], plot[3]), fill=INK, width=3)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill=INK, width=3)

    spectrum_points = [point(float(xv), float(yv)) for xv, yv in zip(x, y)]
    fit_points = [point(float(xv), float(yv)) for xv, yv in zip(x, fitted)]
    draw.line(spectrum_points, fill=SPECTRUM, width=3)
    draw.line(fit_points, fill=FIT, width=6)

    center_x, _ = point(fit.center, ymin)
    _dashed_vertical(draw, center_x, plot[1], plot[3], fill=ACCENT)

    draw.text((72, 38), title, fill=INK, font=_font(42, bold=True))
    draw.text(
        (74, 92),
        "Single-peak spectral analysis  •  synthetic example",
        fill=MUTED,
        font=_font(23),
    )

    legend_y = 195
    draw.line((165, legend_y + 11, 215, legend_y + 11), fill=SPECTRUM, width=4)
    draw.text((228, legend_y), "Synthetic spectrum", fill=INK, font=_font(21))
    draw.line((415, legend_y + 11, 465, legend_y + 11), fill=FIT, width=6)
    draw.text((478, legend_y), "Lorentz fit", fill=INK, font=_font(21))

    card = (1020, 205, 1288, 380)
    draw.rounded_rectangle(card, radius=18, fill=(248, 250, 253), outline=(220, 227, 236), width=2)
    draw.text((1045, 225), "FIT SUMMARY", fill=MUTED, font=_font(17, bold=True))
    metrics = [
        ("Center", f"{fit.center:.4f} THz"),
        ("HWHM", f"{fit.hwhm:.4f} THz"),
        ("R²", f"{fit.r2:.4f}"),
        ("Lifetime", f"{fit.lifetime_ps:.3f} ps"),
    ]
    for index, (label, value) in enumerate(metrics):
        yy = 265 + index * 28
        draw.text((1045, yy), label, fill=MUTED, font=_font(18))
        value_box = draw.textbbox((0, 0), value, font=_font(18, bold=True))
        draw.text((1264 - (value_box[2] - value_box[0]), yy), value, fill=INK, font=_font(18, bold=True))

    x_label = "Frequency (THz)"
    x_box = draw.textbbox((0, 0), x_label, font=_font(24, bold=True))
    draw.text(((plot[0] + plot[2] - (x_box[2] - x_box[0])) / 2, 746), x_label, fill=INK, font=_font(24, bold=True))
    _rotated_label(image, "Intensity (a.u.)", (42, (plot[1] + plot[3]) // 2))

    image.save(output_path, dpi=(180, 180))
    return output_path


def _dashed_vertical(
    draw: ImageDraw.ImageDraw,
    x: float,
    y0: float,
    y1: float,
    *,
    fill: tuple[int, int, int],
) -> None:
    y = y0
    while y < y1:
        draw.line((x, y, x, min(y + 12, y1)), fill=fill, width=2)
        y += 22


def _rotated_label(image: Image.Image, text: str, center: tuple[int, int]) -> None:
    layer = Image.new("RGBA", (330, 60), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    font = _font(24, bold=True)
    bbox = layer_draw.textbbox((0, 0), text, font=font)
    layer_draw.text(((330 - (bbox[2] - bbox[0])) / 2, 12), text, fill=INK, font=font)
    layer = layer.rotate(90, expand=True)
    image.paste(layer, (center[0] - layer.width // 2, center[1] - layer.height // 2), layer)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold else ["DejaVuSans.ttf", "arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()
