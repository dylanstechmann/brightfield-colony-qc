"""JSON QC report. The disclaimer is part of the schema, not a comment."""

from __future__ import annotations

from colonyqc.features import FEATURE_NAMES

DISCLAIMER = (
    "Research morphology triage from a brightfield field. "
    "Not a karyotype, STR identity assay, residual-reprogramming-factor assay, "
    "mycoplasma test, potency assay, or lot-release criterion. "
    "A contamination flag means 'look at the plate and run a validated assay', "
    "not a diagnosis. Not for administration of any cell product to humans."
)


def build_report(label: str, proba: dict[str, float], features: list[float]) -> dict:
    flags = []
    if proba.get("contamination_suspect", 0.0) >= 0.35:
        flags.append("contamination_triage")
    if proba.get("debris", 0.0) >= 0.5:
        flags.append("mostly_debris")
    if label == "differentiating":
        flags.append("morphology_not_undifferentiated")
    return {
        "disclaimer": DISCLAIMER,
        "call": label,
        "probabilities": {k: round(float(v), 4) for k, v in proba.items()},
        "features": {name: round(float(val), 5) for name, val in zip(FEATURE_NAMES, features)},
        "flags": flags,
        "next_human_step": _next(flags),
    }


def _next(flags: list[str]) -> str:
    if "contamination_triage" in flags:
        return "Quarantine the culture. Inspect under phase contrast. Run a validated mycoplasma assay before any further use."
    if "mostly_debris" in flags:
        return "Field looks empty or full of debris. Check focus, seeding, and whether the vessel was fed."
    if "morphology_not_undifferentiated" in flags:
        return "Morphology is not a compact undifferentiated colony. Confirm the intended fate with a marker assay before expanding."
    return "Morphology is consistent with a compact colony on this model. Still confirm identity and sterility on the lab's schedule."


def _image_to_base64_png(img) -> str | None:
    try:
        from PIL import Image
        import base64
        import io
        import numpy as np
        arr = np.clip(np.asarray(img, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)
        pil_im = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        pil_im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def generate_html_report(
    report: dict,
    image_array=None,
    image_name: str = "Input Brightfield Field",
) -> str:
    """Generate a self-contained, visual HTML QC triage report."""
    import html

    call = report.get("call", "unknown")
    proba = report.get("probabilities", {})
    features = report.get("features", {})
    flags = report.get("flags", [])
    next_step = report.get("next_human_step", "")
    disclaimer = report.get("disclaimer", DISCLAIMER)

    # Status color schemes
    color_map = {
        "undifferentiated": {"bg": "#064e3b", "border": "#059669", "text": "#34d399", "badge": "Nominal / Undifferentiated"},
        "differentiating": {"bg": "#78350f", "border": "#d97706", "text": "#fbbf24", "badge": "Warning: Differentiating"},
        "debris": {"bg": "#312e81", "border": "#6366f1", "text": "#a5b4fc", "badge": "Review: Mostly Debris"},
        "contamination_suspect": {"bg": "#7f1d1d", "border": "#dc2626", "text": "#f87171", "badge": "CRITICAL: Contamination Suspect"},
    }
    status = color_map.get(call, {"bg": "#1e293b", "border": "#475569", "text": "#94a3b8", "badge": call})

    # Encode image if provided
    img_html = ""
    if image_array is not None:
        b64 = _image_to_base64_png(image_array)
        if b64:
            shape_str = f"{image_array.shape[0]} × {image_array.shape[1]} px" if hasattr(image_array, "shape") else ""
            img_html = f"""
            <div class="card">
                <div class="card-header">Input Field Preview ({shape_str})</div>
                <div style="display:flex; justify-content:center; align-items:center; background:#0b0f19; border-radius:8px; padding:16px;">
                    <img src="data:image/png;base64,{b64}" alt="{html.escape(image_name)}" style="max-width:100%; height:auto; border-radius:4px; image-rendering:pixelated; border:1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5);" />
                </div>
                <div style="font-size:12px; color:#64748b; margin-top:8px; text-align:center;">8-bit grayscale brightfield field representation</div>
            </div>
            """

    # Probability bars
    proba_bars = []
    for cls_name, p in sorted(proba.items(), key=lambda kv: kv[1], reverse=True):
        pct = round(p * 100, 1)
        cls_color = color_map.get(cls_name, {}).get("border", "#38bdf8")
        proba_bars.append(f"""
            <div style="margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:13px;">
                    <span style="font-weight:600; color:#e2e8f0;">{html.escape(cls_name)}</span>
                    <span style="font-family:monospace; color:{cls_color};">{pct:.1f}% ({p:.4f})</span>
                </div>
                <div style="background:#1e293b; border-radius:6px; height:12px; overflow:hidden; border:1px solid #334155;">
                    <div style="background:{cls_color}; width:{pct}%; height:100%; border-radius:4px; transition:width 0.3s ease;"></div>
                </div>
            </div>
        """)
    proba_html = "".join(proba_bars)

    # Flags HTML
    if flags:
        flag_tags = "".join(f'<span class="flag-badge">{html.escape(f)}</span>' for f in flags)
    else:
        flag_tags = '<span style="color:#10b981; font-size:13px; font-weight:500;">✓ No morphological anomaly flags</span>'

    # Features table
    feature_descriptions = {
        "fg_fraction": "Fraction of field occupied by colonies / foreground pixels",
        "n_components": "Number of connected foreground components (objects)",
        "largest_area_frac": "Area fraction of the largest component relative to total field",
        "largest_circularity": "Isoperimetric circularity quotient (4*pi*area / perimeter^2)",
        "halo": "Phase-contrast bright halo minus interior intensity difference",
        "interior_std": "Standard deviation of pixel intensities within largest object",
        "thin_fraction": "Foreground pixels with <= 2 neighbors (hyphae/filament indicator)",
        "small_component_frac": "Fraction of components with area < 12 pixels (debris fraction)",
        "bright_near_fg": "Fraction of bright background pixels near foreground boundary",
    }
    feat_rows = []
    for name, val in features.items():
        desc = feature_descriptions.get(name, "Morphological feature measurement")
        feat_rows.append(f"""
            <tr>
                <td style="font-family:monospace; font-weight:600; color:#38bdf8;">{html.escape(name)}</td>
                <td style="font-family:monospace; text-align:right; font-weight:700; color:#f8fafc;">{val:.5f}</td>
                <td style="color:#94a3b8; font-size:12px;">{html.escape(desc)}</td>
            </tr>
        """)
    feat_html = "".join(feat_rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brightfield Colony QC Triage Report</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --text-muted: #94a3b8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.5;
            padding: 32px 16px;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
        }}
        .header {{
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .title {{
            font-size: 24px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.025em;
        }}
        .subtitle {{
            font-size: 14px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        .badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.025em;
            text-transform: uppercase;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}
        .card-header {{
            font-size: 15px;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 16px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            padding-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .triage-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }}
        .callout {{
            background: {status['bg']};
            border: 1px solid {status['border']};
            color: {status['text']};
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
        }}
        .callout-title {{
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .flag-badge {{
            display: inline-block;
            background: #991b1b;
            color: #fecaca;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.05em;
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        .disclaimer-banner {{
            background: #18181b;
            border: 1px solid #3f3f46;
            border-radius: 8px;
            padding: 14px 18px;
            font-size: 12px;
            color: #a1a1aa;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 class="title">Colony Morphology QC Triage</h1>
                <div class="subtitle">Sample: {html.escape(image_name)}</div>
            </div>
            <div>
                <span class="badge" style="background:{status['bg']}; border:1px solid {status['border']}; color:{status['text']};">
                    {html.escape(status['badge'])}
                </span>
            </div>
        </div>

        <div class="callout">
            <div class="callout-title">Recommended Next Laboratory Step</div>
            <div style="font-size:14px; font-weight:500;">{html.escape(next_step)}</div>
            <div style="margin-top:10px; font-size:13px;">
                <span style="font-weight:600;">Active Flags: </span>{flag_tags}
            </div>
        </div>

        <div class="triage-grid">
            {img_html}
            <div class="card">
                <div class="card-header">Classification Probabilities</div>
                {proba_html}
            </div>
        </div>

        <div class="card">
            <div class="card-header">Extracted Morphological Features</div>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th style="text-align:right;">Extracted Value</th>
                        <th>Feature Interpretation</th>
                    </tr>
                </thead>
                <tbody>
                    {feat_html}
                </tbody>
            </table>
        </div>

        <div class="disclaimer-banner">
            <strong style="color:#e4e4e7;">Regulatory & Research Disclaimer:</strong> {html.escape(disclaimer)}
        </div>
    </div>
</body>
</html>
"""

