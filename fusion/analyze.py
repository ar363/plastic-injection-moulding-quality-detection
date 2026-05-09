"""
analyze.py — Complexity analysis, evaluation, ablation, and report generation.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as C


# =============================================================================
# Complexity Analysis
# =============================================================================

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def estimate_gflops(tabular_dim=128, seq_len=4096):
    """Approximate theoretical FLOPs per forward pass."""
    flops = 0.0
    flops += 0.4e9                     # Thermal EfficientNet-B0
    flops += 4.1e9 * 3                  # Visual ResNet-50 × 3 sections
    # TCN
    chs, dilations = [64, 128, 256], [1, 2, 4]
    for i, ch in enumerate(chs):
        in_ch = 8 if i == 0 else chs[i-1]
        for _ in dilations:
            flops += 2 * in_ch * ch * 3 * seq_len
    # Tabular MLP
    flops += tabular_dim * 256 + 256 * 256 + 256 * 192
    # Fusion Transformer
    flops += 4 * 2 * 4 * 4 * 256
    flops += 2 * 2 * 4 * 256 * 1024
    # Heads
    flops += 256 * 128 + 128 * 8
    flops += 256 * 128 + 128 * 47
    return flops / 1e9

def tcn_rf():
    k = 3
    total = 1
    for _ in C.TCN_CHANNEL_LIST:
        for d in C.TCN_DILATIONS:
            total += (k-1) * d * 2
    return total

def complexity_report(model, tabular_dim, out_dir=Path("artifacts")):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total, trainable = count_params(model)
    gflops = estimate_gflops(tabular_dim)
    rf = tcn_rf()

    # Per-modality components
    mods = {
        "Thermal CNN (EfficientNet-B0)":        (5.1e6, 0.40),
        "Visual CNN (ResNet-50 × 3 views)":     (25.6e6, 12.3),
        "TCN (Causal dilated conv)":            (1.2e6, 0.20),
        "Tabular MLP (3 layers)":               (0.2e6, 0.001),
        "Cross-modal Fusion Transformer":       (2.1e6, 0.01),
        "Defect + DANN heads":                  (0.1e6, 0.0005),
    }

    lines = [
        "# Algorithmic Complexity Analysis",
        "",
        f"## Summary",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Parameters | {total:,} |",
        f"| Trainable Parameters | {trainable:,} |",
        f"| Model Size (FP32) | {total*4/1024/1024:.2f} MB |",
        f"| Theoretical FLOPs | {gflops:.2f} GFLOPs |",
        f"| TCN Receptive Field | {rf} timesteps |",
        f"| Sequence Length | {C.TCN_TARGET_LEN} |",
        f"| Fusion Token Dim | {C.FUSION_TOKEN_DIM} |",
        f"| Attention Heads | {C.FUSION_N_HEADS} |",
        f"| Transformer Layers | {C.FUSION_N_LAYERS} |",
        "",
        "## Per-Modality Breakdown",
        "| Modality | Params | GFLOPs |",
        "|----------|--------|--------|",
    ]
    for name, (p, f) in mods.items():
        lines.append(f"| {name} | {p:,.0f} | {f:.3f} |")

    total_m = sum(v[0] for v in mods.values())
    total_f = sum(v[1] for v in mods.values())
    lines.append(f"| **Total (theoretical)** | {total_m:,.0f} | {total_f:.3f} |")

    lines += [
        "",
        "## Asymptotic Complexity (Big-O)",
        f"- **Thermal CNN**: O(H·W·C_in·C_out) where H=W={C.THERMAL_INPUT_SIZE}",
        f"- **Visual CNN**: O(3 × H·W·C_in·C_out) per part (3 sections)",
        f"- **TCN**: O(L·C²·K) where L={C.TCN_TARGET_LEN}, kernel K=3",
        f"- **Fusion Transformer**: O(N²·D) where N=4 tokens, D={C.FUSION_TOKEN_DIM}",
        f"- **Overall**: O(W·H·C² + L·C²·K + D²)",
        "",
        "## Memory Complexity",
        f"- Embedding buffer: 4 × {C.FUSION_TOKEN_DIM} = 4KB per sample",
        f"- Transformer KV cache: 2 × {C.FUSION_N_LAYERS} × 4 × {C.FUSION_TOKEN_DIM} × {FUSION_N_HEADS if 'FUSION_N_HEADS' in dir() else 4} = negligible",
        f"- Full precision model weights: ~{total*4/1024/1024:.1f} MB",
    ]

    report = "\n".join(lines)
    (out_dir / "complexity.md").write_text(report)
    print(f"  Complexity report → {out_dir / 'complexity.md'}")
    return {"total_params": total, "trainable": trainable, "gflops": gflops, "rf": rf,
            "mods": mods}


# =============================================================================
# Metrics
# =============================================================================

def find_best_thresholds(probs, targets, grid=None):
    """Grid-search per-label threshold maximizing F1."""
    if grid is None:
        grid = C.THRESHOLD_GRID
    best = np.full(probs.shape[1], 0.5)
    for i in range(probs.shape[1]):
        bf = -1.0
        for t in grid:
            p = (probs[:, i] >= t).astype(int)
            if p.sum() == 0:
                continue
            f = f1_score(targets[:, i], p, zero_division=0)
            if f > bf:
                bf, best[i] = f, t
    return best

def compute_metrics(probs, targets, thresholds=None):
    if thresholds is None:
        thresholds = np.full(probs.shape[1], 0.5)
    # Handle NaN/inf in probs
    probs = np.nan_to_num(probs, nan=0.5, posinf=0.5, neginf=0.5)
    preds = (probs >= thresholds).astype(int)
    macro = f1_score(targets, preds, average="macro", zero_division=0)
    micro = f1_score(targets, preds, average="micro", zero_division=0)
    pl_f1 = f1_score(targets, preds, average=None, zero_division=0)
    rocs, prs = [], []
    for i in range(probs.shape[1]):
        if len(np.unique(targets[:, i])) < 2:
            rocs.append(np.nan); prs.append(np.nan)
        else:
            try:
                rocs.append(roc_auc_score(targets[:, i], probs[:, i]))
                prs.append(average_precision_score(targets[:, i], probs[:, i]))
            except (ValueError, ZeroDivisionError):
                rocs.append(np.nan); prs.append(np.nan)
    return {
        "macro_f1": float(macro), "micro_f1": float(micro),
        "mean_roc_auc": float(np.nanmean(rocs)),
        "mean_pr_auc": float(np.nanmean(prs)),
        "per_label_f1": {k: float(v) for k, v in zip(C.LABEL_COLS, pl_f1)},
        "per_label_auc": {k: float(v) for k, v in zip(C.LABEL_COLS, rocs)},
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_convergence(history, out_dir):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Loss
    axes[0].plot(df["epoch"], df["total"], "b-", lw=2, label="Total Loss")
    axes[0].plot(df["epoch"], df["detection"], "g-", lw=1.5, label="Detection (Focal)")
    axes[0].plot(df["epoch"], df["dann"], "r-", lw=1.5, label="DANN")
    axes[0].plot(df["epoch"], df["consistency"], "m-", lw=1.5, label="Cross-modal")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Convergence"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    # DANN lambda
    axes[1].plot(df["epoch"], df["dann_lambda"], "k-", lw=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("λ_DANN")
    axes[1].set_title("DANN Lambda Schedule (Ganin et al. 2016)"); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Convergence plot → {out_dir / 'convergence.png'}")

def plot_attention_weights(token_weights, out_dir):
    mods = ["Thermal", "Visual", "Sequence", "Tabular"]
    means = token_weights.mean(axis=0)
    stds = token_weights.std(axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    bars = ax.bar(mods, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title("Modality Importance via Attention Weights")
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f"{m:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(means)*1.35)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "attention_weights.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Attention weights → {out_dir / 'attention_weights.png'}")
    return means, stds

def plot_confusion(probs, targets, thresholds, out_dir):
    nok = C.LABEL_COLS.index("LBL_NOK")
    preds = (probs >= thresholds).astype(int)
    cm = confusion_matrix(targets[:, nok], preds[:, nok])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["OK", "NOK"], yticklabels=["OK", "NOK"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — LBL_NOK")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {out_dir / 'confusion.png'}")

def plot_embeddings(embeddings, targets, out_dir):
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=C.SEED, perplexity=30)
        emb = tsne.fit_transform(embeddings)
        nok_idx = C.LABEL_COLS.index("LBL_NOK")
        colors = ["#2ecc71" if t <= 0.5 else "#e74c3c" for t in targets[:, nok_idx]]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(emb[:, 0], emb[:, 1], c=colors, alpha=0.6, s=20, edgecolors="none")
        ax.set_title("Fused Embeddings (t-SNE) — NOK=red, OK=green")
        ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
        plt.tight_layout()
        plt.savefig(out_dir / "embeddings.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Embeddings plot → {out_dir / 'embeddings.png'}")
    except ImportError:
        print("  t-SNE not available, skipping embeddings plot")

def plot_ablation(ablation_results, out_dir):
    names = list(ablation_results.keys())
    f1s = [ablation_results[n]["f1"] for n in names]
    colors = ["#2ecc71" if "All" in n else "#e74c3c" if "only" in n or "alone" in n else "#3498db" for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), f1s, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_title("Ablation Study: Modality Contribution")
    ax.axhline(y=max(f1s), color="gray", ls="--", alpha=0.4)
    for b, f in zip(bars, f1s):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                f"{f:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Ablation plot → {out_dir / 'ablation.png'}")

def plot_per_label_f1(probs, targets, thresholds, out_dir):
    preds = (probs >= thresholds).astype(int)
    f1s = [f1_score(targets[:, i], preds[:, i], zero_division=0) for i in range(probs.shape[1])]
    labels = [c.replace("LBL_", "") for c in C.LABEL_COLS]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if l == "NOK" else "#3498db" for l in labels]
    bars = ax.bar(labels, f1s, color=colors)
    ax.set_ylabel("F1 Score"); ax.set_title("Per-Label F1 Score")
    ax.set_ylim(0, 1.0)
    for b, f in zip(bars, f1s):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                f"{f:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "per_label_f1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-label F1 → {out_dir / 'per_label_f1.png'}")


# =============================================================================
# Ablation Runner
# =============================================================================

def run_ablation(model, loader, device, n_batches=10):
    """Evaluate with each modality individually masked."""
    model.eval()
    configs = [
        ("All 4 modalities",   (True, True, True, True)),
        ("Thermal only",       (True, False, False, False)),
        ("Visual only",        (False, True, False, False)),
        ("Sequence only",      (False, False, True, False)),
        ("Tabular only",       (False, False, False, True)),
        ("Thermal + Visual",   (True, True, False, False)),
        ("Thermal + Sequence", (True, False, True, False)),
        ("Visual + Tabular",   (False, True, False, True)),
    ]
    results = {}
    for name, (use_th, use_vi, use_se, use_ta) in configs:
        all_probs, all_targs = [], []
        count = 0
        with torch.no_grad():
            for batch in loader:
                batch["thermal_valid"]  = torch.tensor([use_th]*len(batch["thermal_valid"]))
                batch["visual_valid"]   = torch.tensor([use_vi]*len(batch["visual_valid"]))
                batch["sequence_valid"] = torch.tensor([use_se]*len(batch["sequence_valid"]))
                batch["tabular_valid"]  = torch.tensor([use_ta]*len(batch["tabular_valid"]))
                out = model(batch)
                all_probs.append(torch.sigmoid(out["defect_logits"]).cpu().numpy())
                all_targs.append(batch["labels"].cpu().numpy())
                count += 1
                if count >= n_batches:
                    break
        probs = np.concatenate(all_probs)
        targs = np.concatenate(all_targs)
        thr = find_best_thresholds(probs, targs)
        m = compute_metrics(probs, targs, thr)
        results[name] = {"f1": m["macro_f1"], "roc_auc": m["mean_roc_auc"]}
        print(f"    {name:25s} → F1={m['macro_f1']:.4f}  AUC={m['mean_roc_auc']:.4f}")
    return results


# =============================================================================
# Full Report
# =============================================================================

def write_report(complexity, history, metrics, token_weights, ablation, out_dir):
    dfh = pd.DataFrame(history)
    loss_best = dfh["total"].min()
    loss_final = dfh["total"].iloc[-1]
    mods = ["Thermal", "Visual", "Sequence", "Tabular"]
    mw = token_weights.mean(axis=0)
    sw = token_weights.std(axis=0)

    attn_rows = "\n".join(f"| {mods[i]:10s} | {mw[i]:.4f} | {sw[i]:.4f} |" for i in range(4))
    lbl_rows = "\n".join(
        f"| {k:20s} | {v:.4f} | {metrics['per_label_auc'].get(k, 0):.4f} |"
        for k, v in metrics["per_label_f1"].items())

    n_epochs = len(dfh)
    n_params = complexity["total_params"]
    gflops = complexity["gflops"]

    report = f"""# Multi-Sensor Fusion System — Demo Report

## 1. Algorithmic Complexity

| Metric | Value |
|--------|-------|
| Total Parameters | {n_params:,} |
| Model Size | {n_params*4/1024/1024:.2f} MB |
| FLOPs per forward | {gflops:.2f} GFLOPs |
| TCN Receptive Field | {complexity['rf']} timesteps |
| Fusion Dim | {C.FUSION_TOKEN_DIM} |

## 2. Training Summary

| Metric | Value |
|--------|-------|
| Epochs | {n_epochs} |
| Best Loss | {loss_best:.4f} |
| Final Loss | {loss_final:.4f} |
| Batch Size | {C.BATCH_SIZE} |
| Optimizer | AdamW (lr_backbone={C.LR_BACKBONE}, lr_head={C.LR_HEAD}) |
| DANN λ_max | {C.DANN_LAMBDA_MAX} |

![Convergence](plots/convergence.png)

## 3. Evaluation Metrics

| Metric | Value |
|--------|-------|
| **Macro F1** | {metrics['macro_f1']:.4f} |
| **Micro F1** | {metrics['micro_f1']:.4f} |
| **Mean ROC AUC** | {metrics['mean_roc_auc']:.4f} |
| **Mean PR AUC** | {metrics['mean_pr_auc']:.4f} |

### Per-Label
| Label | F1 | ROC AUC |
|-------|------|---------|
{lbl_rows}

![Confusion](plots/confusion.png)
![Per-Label F1](plots/per_label_f1.png)

## 4. Modality Importance (Attention Weights)

| Modality | Mean | Std |
|----------|------|-----|
{attn_rows}

![Attention](plots/attention_weights.png)
![Embeddings](plots/embeddings.png)

## 5. Ablation Study

| Configuration | F1 | AUC |
|-------------|------|------|
"""
    for name, r in (ablation or {}).items():
        report += f"| {name:25s} | {r['f1']:.4f} | {r['roc_auc']:.4f} |\n"

    report += """
![Ablation](plots/ablation.png)

## 6. Architecture Details

### Forward Pass Pipeline
1. **Thermal** (B,3,224,224) → EfficientNet-B0 → Linear(1280→448) + ROI(10→64) → **512-dim**
2. **Visual** (B,3,1,224,224) → ResNet-50 (shared weights, 3 section views) → Cross-section CLS-attn → **512-dim**
3. **Sequence** (B,8,4096) → Causal dilated TCN (3 blocks × 3 dilations) → SE-attn → GAP → **256-dim**
4. **Tabular** (B,n_feat) → MLP(→256→256→192) → **192-dim**
5. **Fusion** → Project each to 256 → Add modality embedding → Stack 4 tokens → Transformer(2L,4H) → Attentive pool → **256-dim**
6. **Defect Head** → Linear(256→128→8) → sigmoid → 8 defect probabilities
7. **DANN Head** → GradientReversal → Linear(256→128→30) → experiment prediction

### Missing Modality Handling
Each modality's [MASK] token (learned) replaces missing inputs. The validity mask prevents masked tokens from contributing to the pooled representation.

### DANN Regularization
Domain-adversarial training ensures the fused representation does NOT encode experiment-specific setpoint information. λ ramps 0→{C.DANN_LAMBDA_MAX} over first {C.DANN_WARMUP_EP} epochs (Ganin et al. 2016).
"""
    (out_dir / "report.md").write_text(report)
    print(f"  Report → {out_dir / 'report.md'}")
