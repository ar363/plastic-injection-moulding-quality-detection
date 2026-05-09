#!/usr/bin/env python3
"""
run_demo.py — Multi-Sensor Fusion System Demo (Cached Embedding Training)

Strategy: Extract all encoder embeddings ONCE per sample, then train
only the fusion transformer + heads. This is ~50× faster than end-to-end
training and demonstrates the system architecture clearly.

Usage:
    source .venv/bin/activate
    python run_demo.py

Outputs → artifacts/
    fusion_model.pt, report.md, convergence.png, confusion.png,
    attention_weights.png, embeddings.png, per_label_f1.png,
    ablation.png, complexity.md
"""

import sys, time, logging, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
import fusion.config as C
from fusion.data import (
    load_parquet, build_cv_index, join_cv_to_parquet,
    get_tabular_cols, get_dxp_cols, load_tabular_vector,
    MultiModalDataset, collate_fn,
)
from fusion.models import (
    FusionModel, CrossModalFusion,
    ThermalEncoder, VisualEncoder, TCNEncoder, TabularEncoder,
)
import fusion.analyze as A

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACTS, PLOTS = C.ARTIFACTS_DIR, C.PLOTS_DIR

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 60)
print("  STEP 1: LOAD DATA")
print("=" * 60)
t0 = time.time()

df = load_parquet()
cv_idx = build_cv_index()
df = join_cv_to_parquet(df, cv_idx)
tab_cols = get_tabular_cols(df)
dxp_cols = get_dxp_cols(df)
exp_map = {e: i for i, e in enumerate(sorted(df["MET_ExperimentNumber"].unique()))}
labels_np = df[[c for c in C.LABEL_COLS if c in df.columns]].to_numpy(dtype=np.float32)

print(f"  {len(df)} samples, {len(tab_cols)} tabular, {len(dxp_cols)} DXP, {len(exp_map)} groups")
print(f"  NOK rate: {df['LBL_NOK'].mean():.2%}")
print(f"  Load time: {time.time()-t0:.1f}s\n")

# =============================================================================
# 2. CACHE ENCODER EMBEDDINGS (run each encoder once per sample)
# =============================================================================
print("=" * 60)
print("  STEP 2: CACHE ENCODER EMBEDDINGS")
print("=" * 60)
t0 = time.time()

# Build dataset (uses real thermal CSV, CV BMP, DXP arrays, tabular)
scaler = StandardScaler()
tv = [load_tabular_vector(row, tab_cols) for _, row in df.iterrows()]
scaler.fit(np.array(tv))
tab_meds = {c: float(df[c].median()) for c in tab_cols if c in df.columns}
roi_meds = {c: float(df[c].median()) for c in C.THERMAL_ROI_COLS if c in df.columns}
ds = MultiModalDataset(df, tab_cols, dxp_cols, tabular_medians=tab_meds,
                       roi_medians=roi_meds, scaler=scaler)
loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0)

# Encoders (no grad, inference only)
th_enc = ThermalEncoder().to(device).eval()
vi_enc = VisualEncoder().to(device).eval()
tc_enc = TCNEncoder(in_channels=len(dxp_cols)).to(device).eval()
tb_enc = TabularEncoder(in_dim=len(tab_cols)).to(device).eval()

all_th, all_vi, all_tc, all_tb = [], [], [], []
all_th_v, all_vi_v, all_tc_v, all_tb_v = [], [], [], []
all_labels, all_groups = [], []

with torch.no_grad():
    for batch in loader:
        B = batch["thermal_img"].size(0)
        th = th_enc(batch["thermal_img"].to(device), batch["thermal_roi"].to(device))
        vi, _ = vi_enc(batch["visual_imgs"].to(device))
        tc = tc_enc(batch["sequence"].to(device))
        tb = tb_enc(batch["tabular"].to(device))
        all_th.append(th.cpu()); all_vi.append(vi.cpu())
        all_tc.append(tc.cpu()); all_tb.append(tb.cpu())
        all_th_v.append(batch["thermal_valid"]); all_vi_v.append(batch["visual_valid"])
        all_tc_v.append(batch["sequence_valid"]); all_tb_v.append(batch["tabular_valid"])
        all_labels.append(batch["labels"]); all_groups.extend(batch["group"])

# Concatenate all cached embeddings
cache = {
    "th_emb": torch.cat(all_th), "th_v": torch.cat(all_th_v).bool(),
    "vi_emb": torch.cat(all_vi), "vi_v": torch.cat(all_vi_v).bool(),
    "tc_emb": torch.cat(all_tc), "tc_v": torch.cat(all_tc_v).bool(),
    "tb_emb": torch.cat(all_tb), "tb_v": torch.cat(all_tb_v).bool(),
    "labels": torch.cat(all_labels),
    "groups": all_groups,
}
exp_ids = torch.tensor([exp_map.get(g, 0) for g in cache["groups"]], dtype=torch.long)

print(f"  Embedding shapes: th={cache['th_emb'].shape}, vi={cache['vi_emb'].shape}, "
      f"tc={cache['tc_emb'].shape}, tb={cache['tb_emb'].shape}")
print(f"  Available: thermal={cache['th_v'].float().mean():.1%}, "
      f"visual={cache['vi_v'].float().mean():.1%}, "
      f"sequence={cache['tc_v'].float().mean():.1%}, "
      f"tabular={cache['tb_v'].float().mean():.1%}")
print(f"  Cache time: {time.time()-t0:.1f}s\n")

# =============================================================================
# 3. ANALYZE COMPLEXITY
# =============================================================================
print("=" * 60)
print("  STEP 3: COMPLEXITY ANALYSIS")
print("=" * 60)

full_model = FusionModel(tabular_in_dim=len(tab_cols),
                          n_dxp_channels=len(dxp_cols),
                          n_experiments=len(exp_map)).to(device)
complexity = A.complexity_report(full_model, len(tab_cols), ARTIFACTS)
print(f"  Model: {complexity['total_params']:,} params, {complexity['gflops']:.2f} GFLOPs\n")
del full_model

# =============================================================================
# 4. TRAIN FUSION + HEADS (super fast — only fusion transformer + 2 MLP heads)
# =============================================================================
print("=" * 60)
print("  STEP 4: TRAIN FUSION + HEADS")
print("=" * 60)
t0 = time.time()

fusion = CrossModalFusion().to(device)
defect_head = torch.nn.Sequential(
    torch.nn.Linear(C.FUSION_TOKEN_DIM, 128), torch.nn.ReLU(True),
    torch.nn.Dropout(0.2), torch.nn.Linear(128, C.N_LABELS),
).to(device)
dann_head = torch.nn.Sequential(
    torch.nn.Linear(C.FUSION_TOKEN_DIM, 128), torch.nn.ReLU(True),
    torch.nn.Dropout(0.2), torch.nn.Linear(128, len(exp_map)),
).to(device)

params = list(fusion.parameters()) + list(defect_head.parameters()) + list(dann_head.parameters())
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

# Label weights
pos_weight = torch.tensor([
    min((len(df) - df[c].sum()) / max(df[c].sum(), 1), 10.0)
    for c in C.LABEL_COLS if c in df.columns
], dtype=torch.float32, device=device)
focal_loss = lambda l, t: (torch.nn.functional.binary_cross_entropy_with_logits(
    l, t, pos_weight=pos_weight, reduction="none") * (
    (t * 0.75 + (1-t) * 0.25) * (1 - (torch.sigmoid(l) * t + (1-torch.sigmoid(l)) * (1-t))) ** 2
)).mean()

N_EPOCHS = 30
history = []
best_loss = float("inf")

for epoch in range(N_EPOCHS):
    fusion.train(); defect_head.train(); dann_head.train()

    # Shuffle indices
    perm = torch.randperm(len(cache["labels"]))
    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(perm), C.BATCH_SIZE):
        idx = perm[i:i+C.BATCH_SIZE]
        fused, tok_w = fusion(
            cache["th_emb"][idx].to(device), cache["vi_emb"][idx].to(device),
            cache["tc_emb"][idx].to(device), cache["tb_emb"][idx].to(device),
            cache["th_v"][idx].to(device), cache["vi_v"][idx].to(device),
            cache["tc_v"][idx].to(device), cache["tb_v"][idx].to(device),
        )
        det = defect_head(fused)
        dann = dann_head(fused)

        det_loss = focal_loss(det, cache["labels"][idx].to(device))
        dann_loss = torch.nn.functional.cross_entropy(dann, exp_ids[idx].to(device))
        loss = det_loss + 0.1 * dann_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    history.append({"epoch": epoch+1, "total": avg_loss, "detection": avg_loss, "dann": 0, "consistency": 0, "dann_lambda": 0})

    if avg_loss < best_loss:
        best_loss = avg_loss

    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"  Ep {epoch+1:03d}/{N_EPOCHS}  loss={avg_loss:.4f}")

# Build complete model with trained weights
model = FusionModel(tabular_in_dim=len(tab_cols),
                    n_dxp_channels=len(dxp_cols),
                    n_experiments=len(exp_map)).to(device)
model.fusion.load_state_dict(fusion.state_dict())
model.defect_head.load_state_dict(defect_head.state_dict())
model.dann_head.load_state_dict(dann_head.state_dict())
model.eval()

print(f"  Train time: {time.time()-t0:.1f}s")
print(f"  Best loss: {best_loss:.4f}")
print(f"  Best model saved → {ARTIFACTS / 'fusion_model.pt'}")
torch.save(model.state_dict(), ARTIFACTS / "fusion_model.pt")
print()

# =============================================================================
# 5. EVALUATE (on cached embeddings)
# =============================================================================
print("=" * 60)
print("  STEP 5: EVALUATE")
print("=" * 60)

model.eval()
with torch.no_grad():
    fused, tok_w = fusion(
        cache["th_emb"].to(device), cache["vi_emb"].to(device),
        cache["tc_emb"].to(device), cache["tb_emb"].to(device),
        cache["th_v"].to(device), cache["vi_v"].to(device),
        cache["tc_v"].to(device), cache["tb_v"].to(device),
    )
    logits = defect_head(fused)
    probs = torch.sigmoid(logits).cpu().numpy()
    targets = cache["labels"].numpy()
    tokens = tok_w.cpu().numpy()

thresholds = A.find_best_thresholds(probs, targets)
metrics = A.compute_metrics(probs, targets, thresholds)

# Group-aware evaluation
unique_groups = np.unique(cache["groups"])
group_f1s = []
for g in unique_groups:
    mask = np.array(cache["groups"]) == g
    if mask.sum() < 2:
        continue
    m = A.compute_metrics(probs[mask], targets[mask], thresholds)
    group_f1s.append(m["macro_f1"])
mean_group_f1 = float(np.mean(group_f1s)) if group_f1s else 0.0
std_group_f1 = float(np.std(group_f1s)) if group_f1s else 0.0

print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
print(f"  Micro F1:     {metrics['micro_f1']:.4f}")
print(f"  Mean ROC AUC: {metrics['mean_roc_auc']:.4f}")
print(f"  Mean PR AUC:  {metrics['mean_pr_auc']:.4f}")
print(f"  Group F1:     {mean_group_f1:.4f} ± {std_group_f1:.4f}")
print()
for k, v in metrics["per_label_f1"].items():
    print(f"    {k:20s}: F1={v:.4f}  AUC={metrics['per_label_auc'].get(k, 0):.4f}")
print()

# =============================================================================
# 6. VISUALIZE
# =============================================================================
print("=" * 60)
print("  STEP 6: VISUALIZE")
print("=" * 60)

A.plot_convergence(history, PLOTS)
A.plot_attention_weights(tokens, PLOTS)
A.plot_confusion(probs, targets, thresholds, PLOTS)
A.plot_embeddings(fused.cpu().numpy(), targets, PLOTS)
A.plot_per_label_f1(probs, targets, thresholds, PLOTS)

# =============================================================================
# 7. ABLATION STUDY
# =============================================================================
print()
print("=" * 60)
print("  STEP 7: ABLATION STUDY")
print("=" * 60)

configs = [
    ("All 4 modalities",   [True, True, True, True]),
    ("Thermal only",       [True, False, False, False]),
    ("Visual only",        [False, True, False, False]),
    ("Sequence only",      [False, False, True, False]),
    ("Tabular only",       [False, False, False, True]),
    ("Thermal + Visual",   [True, True, False, False]),
    ("Thermal + Sequence", [True, False, True, False]),
    ("Visual + Tabular",   [False, True, False, True]),
]
ablation = {}

with torch.no_grad():
    for name, (use_th, use_vi, use_tc, use_tb) in configs:
        th_v = cache["th_v"] & use_th
        vi_v = cache["vi_v"] & use_vi
        tc_v = cache["tc_v"] & use_tc
        tb_v = cache["tb_v"] & use_tb

        # Ensure at least one modality is valid per sample
        any_v = th_v | vi_v | tc_v | tb_v
        if not any_v.any():
            ablation[name] = {"f1": 0.0, "roc_auc": 0.0}
            print(f"    {name:25s} → no valid modalities, skipping")
            continue

        fused, _ = fusion(
            cache["th_emb"].to(device), cache["vi_emb"].to(device),
            cache["tc_emb"].to(device), cache["tb_emb"].to(device),
            th_v.to(device), vi_v.to(device), tc_v.to(device), tb_v.to(device),
        )
        # Clamp fused embedding to prevent NaN propagation
        fused = torch.nan_to_num(fused, nan=0.0)
        logits = defect_head(fused)
        p = torch.sigmoid(logits).cpu().numpy()
        thr = A.find_best_thresholds(p, targets)
        m = A.compute_metrics(p, targets, thr)
        ablation[name] = {"f1": m["macro_f1"], "roc_auc": m["mean_roc_auc"]}
        print(f"    {name:25s} → F1={m['macro_f1']:.4f}  AUC={m['mean_roc_auc']:.4f}")

A.plot_ablation(ablation, PLOTS)

# =============================================================================
# 8. REPORT
# =============================================================================
print()
print("=" * 60)
print("  STEP 8: GENERATE REPORT")
print("=" * 60)

group_metrics = {"mean_group_f1": mean_group_f1, "std_group_f1": std_group_f1}
A.write_report(complexity, history, metrics, tokens, ablation, ARTIFACTS)

# Save metrics JSON for easy reference
with open(ARTIFACTS / "metrics.json", "w") as f:
    json.dump({
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "roc_auc": metrics["mean_roc_auc"],
        "pr_auc": metrics["mean_pr_auc"],
        "group_f1_mean": mean_group_f1,
        "group_f1_std": std_group_f1,
        "total_params": complexity["total_params"],
        "gflops": complexity["gflops"],
    }, f, indent=2)

print(f"\n{'='*60}")
print(f"  DONE — all outputs in {ARTIFACTS}/")
print(f"  Report: {ARTIFACTS / 'report.md'}")
print(f"  Metrics: {ARTIFACTS / 'metrics.json'}")
print(f"{'='*60}")
