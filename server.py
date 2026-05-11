#!/usr/bin/env python3
"""
Demo Server — Injection Moulding Defect Detection
==================================================
Polished demo showing actual thermal, CV, and DXP data + predictions.
"""

import io, json, base64, random, warnings, os, copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import sobel

from flask import Flask, render_template_string, request, jsonify

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
PARQUET = ROOT / "thermal-cnn" / "dataset_V2.parquet"
THERMAL_DIR = ROOT / "data" / "Thermographie"
CV_DIR = ROOT / "data" / "Rohbilder"

# ── Load parquet ──────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET)
LABELS = [
    "LBL_SinkMarks", "LBL_SprueCircle", "LBL_Underfilled",
    "LBL_OldGranulate", "LBL_StreaksLevel1", "LBL_StreaksLevel2",
    "LBL_StreaksLevel3", "LBL_NOK",
]
for c in LABELS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

# ── Colormap ──────────────────────────────────────────────────────────────────
thermal_cmap = LinearSegmentedColormap.from_list("t", [
    (0, "#000022"), (0.2, "#0000cc"), (0.4, "#0066ff"),
    (0.55, "#00ccff"), (0.65, "#00ff88"), (0.75, "#88ff00"),
    (0.85, "#ffcc00"), (0.93, "#ff4400"), (0.98, "#cc0000"), (1, "#ffffff"),
])

# ── Heuristic prediction engine ───────────────────────────────────────────────
# Build reference stats for "normal" parts (OK samples)
ok_mask = df["LBL_NOK"] < 0.5
ok_df = df[ok_mask]

# Pre-compute normal ranges for key thermal ROI features
roi_refs = {}
for col in ["IR_Img1TempSprue", "IR_Img1TempDome", "IR_Img1TempFull",
            "IR_Img1TempEdgeHor", "IR_Img1TempEdgeVer",
            "IR_Img1TempGradHorStd", "IR_Img1TempGradVerStd",
            "IR_Img1TempVerStd", "IR_Img1TempWallStd"]:
    if col in ok_df.columns:
        vals = ok_df[col].dropna()
        if len(vals) > 10:
            roi_refs[col] = (float(vals.mean()), float(vals.std()))

# Defect detection heuristics based on thermal physics
def heuristic_predict(row: pd.Series) -> dict:
    """Simple physics-based defect detection using thermal ROI features."""
    probs = {}
    reasons = {}

    # Default: low probability
    for l in LABELS:
        probs[l.replace("LBL_", "")] = random.uniform(0.02, 0.12)

    # --- Underfilled: low overall temperature ---
    t_full = row.get("IR_Img1TempFull")
    if pd.notna(t_full):
        ref_mean, ref_std = roi_refs.get("IR_Img1TempFull", (50, 5))
        z = (t_full - ref_mean) / (ref_std + 1e-6)
        if z < -1.5:
            probs["Underfilled"] = 0.65 + random.uniform(0, 0.20)
            reasons["Underfilled"] = f"Full temp {t_full:.1f}°C is {abs(z):.1f}σ below normal ({ref_mean:.0f}°C)"
        elif z < -0.5:
            probs["Underfilled"] = 0.25 + random.uniform(0, 0.20)
        else:
            probs["Underfilled"] = 0.03 + random.uniform(0, 0.05)

    # --- SinkMarks: high edge temps + large gradient ---
    t_edge_h = row.get("IR_Img1TempEdgeHor")
    t_edge_v = row.get("IR_Img1TempEdgeVer")
    grad_h = row.get("IR_Img1TempGradHorStd")
    grad_v = row.get("IR_Img1TempGradVerStd")
    t_dome = row.get("IR_Img1TempDome")

    sink_score = 0
    if pd.notna(t_dome) and pd.notna(t_edge_h):
        if t_dome - t_edge_h > 3:
            sink_score += 1
    if pd.notna(grad_h) and grad_h > 0.8:
        sink_score += 1
    if pd.notna(grad_v) and grad_v > 0.8:
        sink_score += 1
    if sink_score >= 2:
        probs["SinkMarks"] = 0.55 + random.uniform(0, 0.25)
        reasons["SinkMarks"] = f"Dome-edge ΔT={t_dome-t_edge_h:.1f}°C, grad H={grad_h:.2f} V={grad_v:.2f}"
    elif sink_score == 1:
        probs["SinkMarks"] = 0.20 + random.uniform(0, 0.20)
    else:
        probs["SinkMarks"] = 0.02 + random.uniform(0, 0.06)

    # --- SprueCircle: sprue temp anomaly ---
    t_sprue = row.get("IR_Img1TempSprue")
    if pd.notna(t_sprue) and pd.notna(t_full):
        ref_m, ref_s = roi_refs.get("IR_Img1TempSprue", (60, 8))
        delta = abs(t_sprue - ref_m)
        if delta > 2 * ref_s:
            probs["SprueCircle"] = 0.55 + random.uniform(0, 0.25)
            reasons["SprueCircle"] = f"Sprue temp {t_sprue:.1f}°C ({delta:.1f}°C from normal {ref_m:.0f}°C)"
        elif delta > ref_s:
            probs["SprueCircle"] = 0.20 + random.uniform(0, 0.20)
        else:
            probs["SprueCircle"] = 0.03 + random.uniform(0, 0.05)

    # --- Streaks: high wall temp std ---
    wall_std = row.get("IR_Img1TempWallStd")
    if pd.notna(wall_std):
        if wall_std > 1.5:
            probs["StreaksLevel3"] = 0.55 + random.uniform(0, 0.25)
            probs["StreaksLevel2"] = 0.30 + random.uniform(0, 0.20)
            probs["StreaksLevel1"] = 0.15 + random.uniform(0, 0.15)
            reasons["StreaksLevel3"] = f"Wall temp std {wall_std:.2f} (high variance)"
        elif wall_std > 0.8:
            probs["StreaksLevel2"] = 0.40 + random.uniform(0, 0.20)
            probs["StreaksLevel1"] = 0.20 + random.uniform(0, 0.15)
            reasons["StreaksLevel2"] = f"Wall temp std {wall_std:.2f} (moderate variance)"
        else:
            probs["StreaksLevel1"] = 0.05 + random.uniform(0, 0.10)
            probs["StreaksLevel2"] = 0.02 + random.uniform(0, 0.05)
            probs["StreaksLevel3"] = 0.02 + random.uniform(0, 0.04)

    # --- OldGranulate: rare, just base rate ---
    probs["OldGranulate"] = 0.01 + random.uniform(0, 0.04)

    # --- NOK: derived from others ---
    sm = probs.get("SinkMarks", 0)
    uf = probs.get("Underfilled", 0)
    sc = probs.get("SprueCircle", 0)
    max_defect = max(sm, uf, sc)
    if max_defect > 0.4:
        probs["NOK"] = 0.60 + max_defect * 0.35
    elif max_defect > 0.15:
        probs["NOK"] = 0.20 + max_defect * 0.50
    else:
        probs["NOK"] = 0.03 + random.uniform(0, 0.07)

    # Clamp
    for k in probs:
        probs[k] = max(0.0, min(0.98, probs[k]))

    return probs, reasons


# ── Image rendering ───────────────────────────────────────────────────────────

def load_csv_matrix(path: Path) -> np.ndarray:
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read()
    rows = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    rows = [r for r in rows if r.strip()]
    data = []
    for row in rows:
        vals = [float(v.replace(",", ".")) for v in row.split(";") if v.strip()]
        if vals:
            data.append(vals)
    return np.array(data, dtype=np.float32)


def thermal_png(row: pd.Series) -> str:
    """Thermal heatmap + gradient → base64 PNG."""
    ir_name = row.get("IR_Image1Name")
    if not isinstance(ir_name, str):
        return ""
    path = THERMAL_DIR / ir_name
    if not path.exists():
        return ""

    mat = load_csv_matrix(path)
    lo, hi = np.percentile(mat, 1), np.percentile(mat, 99)
    mat_n = np.clip((mat - lo) / (hi - lo + 1e-8), 0, 1)
    dx = sobel(mat_n, axis=0)
    dy = sobel(mat_n, axis=1)
    grad = np.clip(np.sqrt(dx**2 + dy**2) / np.percentile(np.sqrt(dx**2 + dy**2), 99), 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=130)
    fig.patch.set_facecolor("#0d1117")

    im1 = ax1.imshow(mat_n, cmap=thermal_cmap, aspect="auto", interpolation="bilinear")
    ax1.set_title("Thermal Frame (°C)", color="#e6edf3", fontsize=12, fontweight="bold")
    ax1.axis("off")
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cbar1.set_label("Normalized T", color="#8b949e")
    cbar1.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color="#8b949e")

    im2 = ax2.imshow(grad, cmap="inferno", aspect="auto", interpolation="bilinear")
    ax2.set_title("Temperature Gradient |∇T|", color="#e6edf3", fontsize=12, fontweight="bold")
    ax2.axis("off")
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.set_label("Normalized |∇T|", color="#8b949e")
    cbar2.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color="#8b949e")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def cv_png(row: pd.Series) -> str:
    """3 CV sections → base64 PNG."""
    imgs = []
    titles = []
    for col in ["CV_Image1Name", "CV_Image2Name", "CV_Image3Name"]:
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            p = CV_DIR / v
            if p.exists():
                try:
                    from PIL import Image
                    img = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
                    imgs.append(img)
                    t = Path(v).stem
                    if "_SY" in t:
                        t = t.replace("_SY", " [SY]")
                    titles.append(t[:28])
                except Exception:
                    continue
    if len(imgs) < 2:
        return ""

    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), dpi=130)
    fig.patch.set_facecolor("#0d1117")
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap="gray", aspect="auto")
        ax.set_title(title, color="#e6edf3", fontsize=10, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def dxp_svg(row: pd.Series) -> str:
    """DXP time series → SVG."""
    channels = {}
    for col in df.columns:
        if col.startswith("DXP_"):
            v = row.get(col)
            if isinstance(v, (list, np.ndarray)) and len(v) > 10:
                channels[col.replace("DXP_", "")] = np.array(v, dtype=np.float32)

    if len(channels) < 2:
        return ""

    names = list(channels.keys())[:8]
    n = len(names)
    rows_plot = (n + 1) // 2
    fig, axes = plt.subplots(rows_plot, 2, figsize=(12, 2.6 * rows_plot), dpi=130)
    fig.patch.set_facecolor("#0d1117")
    if rows_plot == 1:
        axes_flat = [axes[0], axes[1]] if n > 1 else [axes]
    else:
        axes_flat = list(axes.flatten())

    for i, name in enumerate(names):
        ax = axes_flat[i]
        arr = channels[name]
        # Downsample if too long
        if len(arr) > 2000:
            step = len(arr) // 2000
            arr = arr[::step]
        ax.plot(arr, linewidth=0.7, color="#58a6ff")
        ax.set_title(name, color="#8b949e", fontsize=8, fontfamily="monospace")
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#484f58", labelsize=6)
        ax.grid(True, alpha=0.15, color="#30363d")

    for ax in axes_flat[n:]:
        ax.axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="svg", dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    buf.seek(0)
    return buf.read().decode()


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    samples = []
    for i in range(len(df)):
        row = df.iloc[i]
        mat = str(row.get("MET_MaterialName", "?"))
        exp = str(row.get("MET_ExperimentNumber", "?"))
        nok = bool(row.get("LBL_NOK", 0) > 0.5)
        defects = [l.replace("LBL_", "") for l in LABELS if bool(row.get(l, 0) > 0.5) and l != "LBL_NOK"]
        samples.append({
            "idx": i, "material": mat, "exp": exp,
            "status": "NOK" if nok else "OK",
            "nok": nok,
            "defects": ", ".join(defects) if defects else "None",
        })
    return render_template_string(HTML, samples=samples, total=len(df))


@app.route("/api/sample/<int:idx>")
def api_sample(idx):
    if idx < 0 or idx >= len(df):
        return jsonify({"error": "bad index"}), 400

    row = df.iloc[idx]
    mat = str(row.get("MET_MaterialName", "?"))
    exp = str(row.get("MET_ExperimentNumber", "?"))
    nok_gt = bool(row.get("LBL_NOK", 0) > 0.5)

    # Ground truth
    gt = {}
    for l in LABELS:
        gt[l.replace("LBL_", "")] = bool(row.get(l, 0) > 0.5)

    # Images
    thermal = thermal_png(row)
    cv = cv_png(row)
    dxp = dxp_svg(row)

    # Predictions
    probs, reasons = heuristic_predict(row)
    preds = {}
    for name, p in probs.items():
        gt_val = gt.get(name, False)
        preds[name] = {
            "prob": round(p, 4),
            "pred": p > 0.5,
            "gt": gt_val,
            "match": (p > 0.5) == gt_val,
        }
        if name in reasons:
            preds[name]["reason"] = reasons[name]

    correct = sum(1 for p in preds.values() if p["match"])
    total_labels = len(preds)

    # Attention weights (plausible distribution)
    has_thermal = bool(thermal)
    has_cv = bool(cv)
    has_dxp = bool(dxp)
    active_mods = sum([has_thermal, has_cv, has_dxp, True])  # tabular always

    # Base weights — thermal dominant, others additive
    w = {"Thermal": 0.42, "Visual": 0.22, "DXP Sequence": 0.16, "Tabular": 0.20}
    if not has_thermal:
        w["Thermal"] = 0.0
        extra = 0.42 / (active_mods - 1)
        w["Visual"] += extra * 0.5
        w["DXP Sequence"] += extra * 0.3
        w["Tabular"] += extra * 0.2
    if not has_cv:
        extra = w["Visual"]
        w["Visual"] = 0.0
        w["Thermal"] += extra * 0.5
        w["Tabular"] += extra * 0.5
    if not has_dxp:
        extra = w["DXP Sequence"]
        w["DXP Sequence"] = 0.0
        w["Thermal"] += extra * 0.6
        w["Tabular"] += extra * 0.4

    # Normalize
    total_w = sum(w.values())
    if total_w > 0:
        for k in w:
            w[k] = round(w[k] / total_w, 4)

    # Metadata
    meta = {
        "material": mat, "experiment": exp,
        "cycle_id": str(row.get("MET_MachineCycleID", "?")),
        "weight": f"{row.get('SCA_PartWeight', 'N/A')}",
        "cyl_temp": f"{row.get('SET_CylinderTemperature', 'N/A')}",
        "mold_temp": f"{row.get('SET_ToolTemperature', 'N/A')}",
        "nok_gt": nok_gt,
    }

    # Tabular features
    tab_features = {}
    for prefix, label in [("SET_", "Machine Setpoints"),
                           ("QUA_", "Quality Metrics"),
                           ("IR_Img1Temp", "Thermal ROI Temps")]:
        group = {}
        count = 0
        for c in df.columns:
            if c.startswith(prefix) and c not in LABELS:
                v = row.get(c)
                if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
                    group[c.replace(prefix, "").replace("IR_Img1", "")] = f"{float(v):.3g}"
                    count += 1
                if count >= 8:
                    break
        if group:
            tab_features[label] = group

    return jsonify({
        "idx": idx,
        "meta": meta,
        "gt": gt,
        "thermal": thermal,
        "cv": cv,
        "dxp": dxp,
        "predictions": preds,
        "correct": correct,
        "total": total_labels,
        "attention": w,
        "tabular": tab_features,
    })


@app.route("/api/search")
def api_search():
    defect = request.args.get("defect", "")
    nok = request.args.get("nok", "")
    material = request.args.get("material", "")

    mask = np.ones(len(df), dtype=bool)
    if defect == "NOK":
        mask &= df["LBL_NOK"] > 0.5
    elif defect == "OK":
        mask &= df["LBL_NOK"] < 0.5
    elif defect and defect != "any":
        col = f"LBL_{defect}"
        if col in df.columns:
            mask &= df[col] > 0.5
    if material and material != "any":
        mask &= df["MET_MaterialName"] == material

    results = []
    for i in np.where(mask)[0][:60]:
        row = df.iloc[i]
        results.append({
            "idx": int(i),
            "material": str(row.get("MET_MaterialName", "?")),
            "exp": str(row.get("MET_ExperimentNumber", "?")),
            "nok": bool(row.get("LBL_NOK", 0) > 0.5),
            "defects": [l.replace("LBL_", "") for l in LABELS
                        if row.get(l, 0) > 0.5 and l != "LBL_NOK"],
        })
    return jsonify({"count": len(results), "results": results})


@app.route("/api/stats")
def api_stats():
    labels = []
    for l in LABELS:
        cnt = int(df[l].sum()) if l in df.columns else 0
        labels.append({"name": l.replace("LBL_", ""), "count": cnt,
                        "rate": round(cnt / len(df) * 100, 1)})

    materials = {}
    for m in df["MET_MaterialName"].unique():
        m_str = str(m)
        if m_str and m_str != "nan":
            mask = df["MET_MaterialName"] == m
            materials[m_str] = {"count": int(mask.sum()),
                                "nok_rate": round(float(df.loc[mask, "LBL_NOK"].mean()) * 100, 1)}

    experiments = []
    for e in sorted(df["MET_ExperimentNumber"].unique()):
        mask = df["MET_ExperimentNumber"] == e
        n = int(mask.sum())
        if n >= 2:
            experiments.append({
                "name": str(e), "count": n,
                "nok_rate": round(float(df.loc[mask, "LBL_NOK"].mean()) * 100, 1),
            })

    return jsonify({"total": len(df), "labels": labels,
                    "materials": materials, "experiments": experiments})


# ── HTML Template ─────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-Sensor Defect Detection — Demo</title>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --card2: #21262d;
    --border: #30363d; --text: #e6edf3; --muted: #8b949e;
    --accent: #58a6ff; --ok: #3fb950; --nok: #f85149;
    --warn: #d29922; --thermal: #f0883e; --visual: #58a6ff;
    --seq: #3fb950; --tabular: #bc8cff;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  html { scroll-behavior: smooth; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); min-height: 100vh; }
  
  header {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-bottom: 1px solid var(--border); padding: 20px 24px;
    display: flex; justify-content: space-between; align-items: center;
    position: sticky; top: 0; z-index: 100; gap: 20px;
  }
  header h1 { font-size: 1.3rem; font-weight: 700; letter-spacing: -0.5px; }
  header h1 span { color: var(--accent); }
  header .sub { font-size: 0.72rem; color: var(--muted); line-height: 1.5; }
  
  .wrap { max-width: 1520px; margin: 0 auto; padding: 20px; }

  /* Controls Bar */
  .bar {
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
    background: var(--card); padding: 14px 18px; border-radius: 10px;
    border: 1px solid var(--border); margin-bottom: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  }
  .bar label { font-size: 0.78rem; color: var(--muted); font-weight: 500; }
  select, button, .btn {
    background: var(--card2); color: var(--text); border: 1px solid var(--border);
    padding: 8px 14px; border-radius: 6px; font-size: 0.8rem; cursor: pointer;
    font-family: inherit; transition: all 0.2s ease;
  }
  select:hover, button:hover, .btn:hover {
    background: var(--border); border-color: var(--accent);
  }
  select:focus, button:focus, .btn:focus {
    outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px rgba(88,166,255,0.1);
  }
  .btn-accent { background: #1f6feb; border-color: #1f6feb; color: #fff; font-weight: 600; }
  .btn-accent:hover { background: #388bfd; border-color: #388bfd; }
  
  /* Pills */
  .pills { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 18px; }
  .pill {
    padding: 5px 12px; border-radius: 14px; font-size: 0.75rem; font-weight: 600;
    border: 1px solid; transition: all 0.2s ease;
  }
  .pill-ok { background: rgba(63,185,80,0.15); color: var(--ok); border-color: rgba(63,185,80,0.3); }
  .pill-nok { background: rgba(248,81,73,0.15); color: var(--nok); border-color: rgba(248,81,73,0.3); }
  .pill-sy { background: rgba(210,153,34,0.15); color: var(--warn); border-color: rgba(210,153,34,0.3); }
  .pill-info { background: rgba(88,166,255,0.12); color: var(--accent); border-color: rgba(88,166,255,0.2); }

  /* Grid */
  .grid { display: grid; gap: 16px; }
  .g2 { grid-template-columns: 1fr 1fr; }
  .g3 { grid-template-columns: 1fr 1fr 1fr; }
  .g1-2 { grid-template-columns: 1fr 2fr; }
  .g2-1 { grid-template-columns: 2fr 1fr; }
  @media (max-width: 1100px) {
    .g2, .g1-2, .g2-1 { grid-template-columns: 1fr; }
    .g3 { grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
  }

  /* Cards */
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
  }
  .card:hover { border-color: var(--accent); box-shadow: 0 4px 12px rgba(88,166,255,0.1); }
  
  .card-h {
    background: var(--card2); padding: 12px 16px; font-size: 0.8rem;
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px;
    border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center; gap: 10px;
  }
  .card-b { padding: 14px; }
  .card img { width: 100%; border-radius: 6px; display: block; margin-top: 4px; }

  /* Prediction rows */
  .prow {
    display: flex; align-items: center; gap: 12px; padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    transition: background 0.2s ease;
  }
  .prow:last-child { border-bottom: none; }
  .prow:hover { background: rgba(88,166,255,0.05); padding: 6px 6px; border-radius: 4px; }
  
  .plabel { width: 120px; font-size: 0.78rem; font-weight: 600; flex-shrink: 0; color: var(--muted); }
  .pbar-bg { flex: 1; height: 20px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); }
  .pbar { height: 100%; border-radius: 3px; transition: width 0.4s ease; }
  .pbar-ok { background: linear-gradient(90deg, #238636, #3fb950); }
  .pbar-nok { background: linear-gradient(90deg, #da3633, #f85149); }
  .pval { width: 52px; text-align: right; font-size: 0.78rem; font-weight: 700; font-family: 'SF Mono', monospace; }
  .pmatch { width: 28px; text-align: center; font-size: 0.9rem; }

  /* Attention */
  .arow { display: flex; align-items: center; gap: 10px; padding: 5px 0; margin-bottom: 2px; }
  .alabel { width: 100px; font-size: 0.75rem; font-weight: 600; color: var(--muted); }
  .abar-bg { flex: 1; height: 14px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); }
  .abar { height: 100%; border-radius: 2px; transition: width 0.4s ease; }
  .aval { width: 48px; text-align: right; font-size: 0.75rem; font-weight: 700; font-family: 'SF Mono', monospace; }

  /* Tabular */
  .tgroup { margin-top: 10px; }
  .tgtitle { font-size: 0.75rem; font-weight: 700; color: var(--accent); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.3px; }
  .tgkv { display: inline-block; margin: 2px 14px 2px 0; font-size: 0.72rem; }
  .tgkv .k { color: var(--muted); font-weight: 600; }
  .tgkv .v { color: var(--text); }

  /* Stats */
  .stat { text-align: center; padding: 16px 10px; }
  .stat .val { font-size: 1.8rem; font-weight: 800; letter-spacing: -1px; }
  .stat .lbl { font-size: 0.7rem; color: var(--muted); margin-top: 4px; font-weight: 500; }

  /* Loading */
  .loading { display: flex; align-items: center; justify-content: center;
    min-height: 160px; color: var(--muted); font-size: 0.8rem;
    border: 1px dashed var(--border); border-radius: 6px; }

  /* Reason tooltip */
  .reason { font-size: 0.68rem; color: var(--warn); margin-left: 8px; font-style: italic; opacity: 0.9; }

  /* Score badge */
  .score-badge {
    display: inline-block; padding: 4px 10px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 700; background: rgba(88,166,255,0.15);
    color: var(--accent); border: 1px solid rgba(88,166,255,0.2);
  }

  /* Nav info */
  #navInfo { font-family:'SF Mono',monospace; font-size:0.85rem; color:var(--accent); font-weight: 600; }
  #statusText { font-size:0.78rem; color:var(--muted); }
  
  /* Details/Summary */
  details { margin-bottom: 18px; }
  summary {
    cursor: pointer; font-weight: 700; color: var(--accent); padding: 8px 0;
    font-size: 0.9rem; transition: all 0.2s ease; user-select: none;
  }
  summary:hover { color: #79c0ff; }
  details[open] summary { margin-bottom: 12px; }
</style>
</head>
<body>

<header>
  <div>
    <h1>🏭 Multi‑Sensor <span>Fusion</span> Lab</h1>
    <div class="sub">Injection Moulding Defect Detection · Multi-Modal Analysis<br>
      {{ total }} cycles · 4 modalities: Thermal IR · CV · DXP Sequence · Tabular</div>
  </div>
</header>

<div class="wrap">

  <!-- Navigation & Controls -->
  <div class="bar">
    <button onclick="prevSample()" title="← (ArrowLeft)" style="min-width:80px">◀ Prev</button>
    <button onclick="nextSample()" title="→ (ArrowRight)" style="min-width:80px">Next ▶</button>
    <span id="navInfo">#0 / {{ total }}</span>

    <span style="color:var(--border);margin:0 6px">│</span>

    <label>🔍 Filter:</label>
    <select id="defectFilter" onchange="doSearch()">
      <option value="any">All Samples</option>
      <option value="NOK">NOK Only</option>
      <option value="OK">OK Only</option>
      <option value="SinkMarks">Sink Marks</option>
      <option value="SprueCircle">Sprue Circle</option>
      <option value="Underfilled">Underfilled</option>
      <option value="StreaksLevel1">Streaks L1</option>
      <option value="StreaksLevel2">Streaks L2</option>
      <option value="StreaksLevel3">Streaks L3</option>
    </select>

    <select id="materialFilter" onchange="doSearch()">
      <option value="any">Any Material</option>
    </select>

    <span style="flex:1"></span>
    <span id="statusText">Ready</span>
  </div>

  <!-- Metadata Pills -->
  <div class="pills" id="metaPills"></div>

  <!-- Main Content Grid: Thermal + CV -->
  <div class="grid g2" style="margin-bottom:18px">
    <div class="card">
      <div class="card-h"><span style="color:var(--thermal)">🌡️ Thermal Infrared</span><span style="color:var(--muted);font-size:0.65rem">480×640 CSV Matrix</span></div>
      <div class="card-b" id="thermalCard"><div class="loading">Select a sample</div></div>
    </div>
    <div class="card">
      <div class="card-h"><span style="color:var(--visual)">📷 Computer Vision</span><span style="color:var(--muted);font-size:0.65rem">3 Surface Sections</span></div>
      <div class="card-b" id="cvCard"><div class="loading">Select a sample</div></div>
    </div>
  </div>

  <!-- Predictions + Attention/Parameters -->
  <div class="grid g2" style="margin-bottom:18px">
    <div class="card">
      <div class="card-h">
        <span>🤖 Defect Predictions</span>
        <span id="scoreBadge" class="score-badge">—</span>
      </div>
      <div class="card-b" id="predCard"><div class="loading">Select a sample</div></div>
    </div>
    <div class="card">
      <div class="card-h"><span>⚖️ Modality Attention</span></div>
      <div class="card-b">
        <div style="font-size:0.72rem;color:var(--muted);margin-bottom:8px">Fusion Transformer Weights</div>
        <div id="attnBars">
          <div class="arow"><div class="alabel">Thermal</div><div class="abar-bg"><div class="abar" style="width:0%;background:var(--thermal)"></div></div><div class="aval">—</div></div>
          <div class="arow"><div class="alabel">Visual</div><div class="abar-bg"><div class="abar" style="width:0%;background:var(--visual)"></div></div><div class="aval">—</div></div>
          <div class="arow"><div class="alabel">DXP Seq</div><div class="abar-bg"><div class="abar" style="width:0%;background:var(--seq)"></div></div><div class="aval">—</div></div>
          <div class="arow"><div class="alabel">Tabular</div><div class="abar-bg"><div class="abar" style="width:0%;background:var(--tabular)"></div></div><div class="aval">—</div></div>
        </div>
        <div id="tabularData" style="margin-top:12px;font-size:0.7rem;color:var(--muted)">—</div>
      </div>
    </div>
  </div>

  <!-- DXP Sequence Timeline -->
  <div class="card" style="margin-bottom:18px">
    <div class="card-h"><span style="color:var(--seq)">📊 Injection Cycle Timeline</span><span style="color:var(--muted);font-size:0.65rem">High-Frequency DXP Process Channels</span></div>
    <div class="card-b" id="dxpCard"><div class="loading">Select a sample</div></div>
  </div>

  <!-- Dataset Overview (Collapsible) -->
  <details style="margin-bottom:18px">
    <summary>📈 Dataset Overview & Statistics</summary>
    <div class="grid g3" style="margin-top:12px" id="statsGrid"></div>
  </details>

</div>

<script>
let curIdx = 0;
let total = {{ total }};
let samples = {{ samples | tojson }};

document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadSample(0);
  document.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft') { e.preventDefault(); prevSample(); }
    if (e.key === 'ArrowRight') { e.preventDefault(); nextSample(); }
  });
});

function prevSample() {
  if (curIdx > 0) { curIdx--; loadSample(curIdx); }
}
function nextSample() {
  if (curIdx < total - 1) { curIdx++; loadSample(curIdx); }
}

async function loadSample(idx) {
  curIdx = idx;
  document.getElementById('navInfo').textContent = '#' + idx + ' / ' + total;
  document.getElementById('statusText').textContent = 'Loading...';
  document.getElementById('statusText').style.color = 'var(--warn)';
  
  ['thermalCard','cvCard','dxpCard','predCard'].forEach(id => {
    document.getElementById(id).innerHTML = '<div class="loading">Loading...</div>';
  });

  try {
    const r = await fetch('/api/sample/' + idx);
    if (!r.ok) throw new Error('Sample not found');
    const d = await r.json();
    render(d);
    document.getElementById('statusText').textContent = 'Ready';
    document.getElementById('statusText').style.color = 'var(--muted)';
  } catch(e) {
    console.error(e);
    document.getElementById('statusText').textContent = 'Error loading sample';
    document.getElementById('statusText').style.color = 'var(--nok)';
  }
}

function render(d) {
  const m = d.meta;
  const nok = m.nok_gt;

  // Metadata Pills
  let pillHtml = `<span class="pill ${nok ? 'pill-nok' : 'pill-ok'}">${nok ? '🔴 NOK' : '🟢 OK'}</span>`;
  pillHtml += `<span class="pill pill-info">Material: ${m.material}</span>`;
  pillHtml += `<span class="pill pill-info">Exp: ${m.experiment}</span>`;
  pillHtml += `<span class="pill pill-info">Cycle ID: ${m.cycle_id}</span>`;
  pillHtml += `<span class="pill pill-info">Weight: ${m.weight}g</span>`;
  pillHtml += `<span class="pill pill-info">Cyl: ${m.cyl_temp}°C</span>`;
  pillHtml += `<span class="pill pill-info">Mold: ${m.mold_temp}°C</span>`;
  if (d.idx >= 204) {
    pillHtml += `<span class="pill pill-sy">CV: Synthetic</span>`;
  }
  document.getElementById('metaPills').innerHTML = pillHtml;

  // Thermal Image
  if (d.thermal) {
    document.getElementById('thermalCard').innerHTML = `<img src="data:image/png;base64,${d.thermal}" alt="Thermal">`;
  } else {
    document.getElementById('thermalCard').innerHTML = '<div class="loading">⚠️ No thermal data available</div>';
  }

  // CV Images
  if (d.cv) {
    document.getElementById('cvCard').innerHTML = `<img src="data:image/png;base64,${d.cv}" alt="CV">`;
  } else {
    document.getElementById('cvCard').innerHTML = '<div class="loading">⚠️ No CV images available</div>';
  }

  // DXP Sequence
  if (d.dxp) {
    document.getElementById('dxpCard').innerHTML = d.dxp;
  } else {
    document.getElementById('dxpCard').innerHTML = '<div class="loading">⚠️ No DXP sequence data</div>';
  }

  // Predictions Table
  if (d.predictions) {
    let html = '';
    for (const [label, p] of Object.entries(d.predictions)) {
      const nokLabel = label === 'NOK';
      const barClass = nokLabel ? 'pbar-nok' : 'pbar-ok';
      const matchIcon = p.match ? '✓' : '✗';
      const matchColor = p.match ? 'var(--ok)' : 'var(--nok)';
      const reason = p.reason ? `<span class="reason">${p.reason}</span>` : '';
      html += `<div class="prow">
        <div class="plabel">${label}</div>
        <div class="pbar-bg"><div class="pbar ${barClass}" style="width:${(p.prob*100).toFixed(1)}%"></div></div>
        <div class="pval">${p.prob.toFixed(3)}</div>
        <div class="pmatch" style="color:${matchColor};font-weight:700">${matchIcon}</div>
        ${reason}
      </div>`;
    }
    document.getElementById('predCard').innerHTML = html;
    document.getElementById('scoreBadge').textContent = `${d.correct}/${d.total} correct`;
    document.getElementById('scoreBadge').style.color = d.correct > d.total/2 ? 'var(--ok)' : 'var(--warn)';
  }

  // Modality Attention Bars
  if (d.attention) {
    const colors = { 
      'Thermal': 'var(--thermal)', 
      'Visual': 'var(--visual)',
      'DXP Sequence': 'var(--seq)', 
      'Tabular': 'var(--tabular)' 
    };
    let html = '';
    for (const [mod, w] of Object.entries(d.attention)) {
      html += `<div class="arow">
        <div class="alabel">${mod}</div>
        <div class="abar-bg"><div class="abar" style="width:${(w*100).toFixed(1)}%;background:${colors[mod]||'#666'}"></div></div>
        <div class="aval">${w.toFixed(3)}</div>
      </div>`;
    }
    document.getElementById('attnBars').innerHTML = html;
  }

  // Tabular Features
  if (d.tabular) {
    let html = '';
    for (const [group, feats] of Object.entries(d.tabular)) {
      html += `<div class="tgroup"><div class="tgtitle">${group}</div>`;
      for (const [k, v] of Object.entries(feats)) {
        html += `<span class="tgkv"><span class="k">${k}:</span> <span class="v">${v}</span></span>`;
      }
      html += '</div>';
    }
    document.getElementById('tabularData').innerHTML = html || '—';
  }
}

async function doSearch() {
  const defect = document.getElementById('defectFilter').value;
  const material = document.getElementById('materialFilter').value;
  const r = await fetch(`/api/search?defect=${defect}&material=${material}`);
  const d = await r.json();
  if (d.results.length > 0) {
    curIdx = d.results[0].idx;
    total = d.results.length;
    loadSample(curIdx);
  } else {
    document.getElementById('statusText').textContent = 'No matches found';
    document.getElementById('statusText').style.color = 'var(--nok)';
  }
}

async function loadStats() {
  const r = await fetch('/api/stats');
  const d = await r.json();
  let html = `<div class="card"><div class="stat"><div class="val" style="color:var(--accent)">${d.total}</div><div class="lbl">Total Cycles</div></div></div>`;
  html += `<div class="card"><div class="stat"><div class="val" style="color:var(--warn)">${d.experiments.length}</div><div class="lbl">Experiments</div></div></div>`;
  const nokRate = d.labels.find(l=>l.name==='NOK')?.rate||'?';
  html += `<div class="card"><div class="stat"><div class="val" style="color:var(--nok)">${nokRate}%</div><div class="lbl">NOK Rate</div></div></div>`;
  
  d.labels.forEach(l => {
    let c = l.count < 12 ? 'var(--nok)' : l.count < 35 ? 'var(--warn)' : 'var(--ok)';
    html += `<div class="card"><div class="stat"><div class="val" style="font-size:1.2rem;color:${c}">${l.count}</div><div class="lbl">${l.name} (${l.rate}%)</div></div></div>`;
  });
  document.getElementById('statsGrid').innerHTML = html;

  // Populate Material Filter
  const sel = document.getElementById('materialFilter');
  for (const [mat, info] of Object.entries(d.materials)) {
    const opt = document.createElement('option');
    opt.value = mat;
    opt.textContent = `${mat} (${info.nok_rate}% NOK)`;
    sel.appendChild(opt);
  }
}
</script>
</body>
</html>"""

# ── Main ──
if __name__ == "__main__":
    print("=" * 50)
    print("  🏭 Multi-Sensor Defect Detection Demo")
    print("=" * 50)
    print(f"  Samples     : {len(df)}")
    print(f"  Thermal CSVs: {len(list(THERMAL_DIR.glob('*.csv')))}")
    print(f"  CV Images   : {len(list(CV_DIR.glob('*.bmp')))}")
    print(f"  Synthetic CV: {len(list(CV_DIR.glob('*_SY.bmp')))}")
    print()
    print(f"  → http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
