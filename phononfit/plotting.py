from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .fitting import LorentzFit


def plot_fit(
    frequency: np.ndarray,
    intensity: np.ndarray,
    fit: LorentzFit,
    output: str | Path,
    *,
    title: str = "Local Lorentz fit",
) -> Path:
    """Write a compact, dependency-light spectrum/fit diagnostic figure."""
    x = np.asarray(frequency, dtype=float)
    y = np.asarray(intensity, dtype=float)
    fitted = fit.evaluate(x)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1200, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    box = (125, 95, 1130, 620)
    draw.rectangle(box, fill=(250, 250, 250), outline=(35, 35, 35), width=2)

    xmin, xmax = float(x.min()), float(x.max())
    ymin = float(min(y.min(), fitted.min()))
    ymax = float(max(y.max(), fitted.max()))
    padding = max((ymax - ymin) * 0.08, 1e-12)
    ymin -= padding
    ymax += padding

    def point(xvalue: float, yvalue: float) -> tuple[float, float]:
        px = box[0] + (xvalue - xmin) / (xmax - xmin) * (box[2] - box[0])
        py = box[3] - (yvalue - ymin) / (ymax - ymin) * (box[3] - box[1])
        return px, py

    for fraction in np.linspace(0.0, 1.0, 6):
        xvalue = xmin + fraction * (xmax - xmin)
        px, _ = point(xvalue, ymin)
        draw.line((px, box[3], px, box[3] + 8), fill=(30, 30, 30), width=2)
        draw.text((px - 22, box[3] + 14), f"{xvalue:.2f}", fill=(30, 30, 30), font=_font(18))
    for fraction in np.linspace(0.0, 1.0, 5):
        yvalue = ymin + fraction * (ymax - ymin)
        _, py = point(xmin, yvalue)
        draw.line((box[0] - 8, py, box[0], py), fill=(30, 30, 30), width=2)
        draw.text((35, py - 10), f"{yvalue:.2f}", fill=(30, 30, 30), font=_font(18))

    draw.line([point(float(xv), float(yv)) for xv, yv in zip(x, y)], fill=(49, 91, 125), width=3)
    draw.line(
        [point(float(xv), float(yv)) for xv, yv in zip(x, fitted)],
        fill=(217, 95, 2),
        width=4,
    )
    center_x, _ = point(fit.center, ymin)
    draw.line((center_x, box[1], center_x, box[3]), fill=(90, 90, 90), width=2)

    draw.text((125, 30), title, fill=(20, 20, 20), font=_font(32))
    draw.text((470, 685), "Frequency (THz)", fill=(20, 20, 20), font=_font(24))
    draw.text((box[0] + 18, box[1] + 16), "spectrum", fill=(49, 91, 125), font=_font(20))
    draw.text((box[0] + 18, box[1] + 46), "Lorentz fit", fill=(217, 95, 2), font=_font(20))
    summary = (
        f"center = {fit.center:.4f} THz\n"
        f"HWHM = {fit.hwhm:.4f} THz\n"
        f"R2 = {fit.r2:.4f}\n"
        f"tau = {fit.lifetime_ps:.3f} ps"
    )
    draw.multiline_text((885, 120), summary, fill=(30, 30, 30), font=_font(19), spacing=7)
    image.save(output_path, dpi=(180, 180))
    return output_path


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()
