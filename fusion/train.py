"""
train.py — Training loop, loss functions, and optimizer setup.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from . import config as C
from .data import MultiModalDataset, collate_fn, load_tabular_vector, load_parquet, build_cv_index, join_cv_to_parquet, get_tabular_cols, get_dxp_cols

log = logging.getLogger(__name__)


# =============================================================================
# Loss functions
# =============================================================================

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, pos_weight=None, alpha=C.FOCAL_ALPHA, gamma=C.FOCAL_GAMMA):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - pt)**self.gamma * bce).mean()


class DANNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.ce(logits, labels)


class SensorConsistencyLoss(nn.Module):
    """Cosine similarity between thermal and TCN embeddings after projecting to same dim."""
    def __init__(self, th_dim=512, tc_dim=256, proj_dim=256):
        super().__init__()
        self.proj_tc = nn.Linear(tc_dim, proj_dim)
        self.proj_th = nn.Linear(th_dim, proj_dim)

    def forward(self, th_emb, tc_emb, th_v, tc_v):
        both = th_v & tc_v
        if not both.any():
            return torch.tensor(0.0, device=th_emb.device, requires_grad=True)
        dev = th_emb.device
        t = F.normalize(self.proj_th(th_emb[both].to(dev)), dim=1)
        s = F.normalize(self.proj_tc(tc_emb[both].to(dev)), dim=1)
        return (1.0 - (t * s).sum(dim=1)).mean()


class TotalLoss(nn.Module):
    """Weighted sum: λ_det * focal + λ_dann * dann + λ_cons * consistency."""
    def __init__(self, pos_weight=None, n_experiments=30,
                 lambda_det=1.0, lambda_dann=0.0, lambda_cons=0.1):
        super().__init__()
        self.lambda_det, self.lambda_dann, self.lambda_cons = lambda_det, lambda_dann, lambda_cons
        self.focal = MultiLabelFocalLoss(pos_weight=pos_weight)
        self.dann = DANNLoss()
        self.cons = SensorConsistencyLoss()

    def set_dann_lambda(self, lam):
        self.lambda_dann = lam

    def forward(self, outputs, labels, exp_ids, th_v, tc_v):
        det = self.focal(outputs["defect_logits"], labels)
        dann = self.dann(outputs["dann_logits"], exp_ids) if self.lambda_dann > 0 else \
            torch.tensor(0.0, device=det.device)
        cons = self.cons(outputs["thermal_emb"], outputs["tcn_emb"], th_v, tc_v)
        total = self.lambda_det * det + self.lambda_dann * dann + self.lambda_cons * cons
        return {"total": total, "detection": det.detach(), "dann": dann.detach(), "consistency": cons.detach()}


def compute_pos_weights(df, device):
    weights = []
    for col in C.LABEL_COLS:
        n_pos = df[col].sum()
        n_neg = len(df) - n_pos
        w = min(n_neg / (n_pos + 1e-6), 10.0)  # cap at 10
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# =============================================================================
# DANN schedule
# =============================================================================

def dann_schedule(epoch, total_epochs):
    if epoch < C.DANN_WARMUP_EP:
        return 0.0
    p = (epoch - C.DANN_WARMUP_EP) / (total_epochs - C.DANN_WARMUP_EP + 1e-6)
    return float(C.DANN_LAMBDA_MAX * min(p * 2, 1.0))


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, loss_fn, optimizer, device, exp_label_map, train=True):
    model.train(train)
    losses = {"total": 0, "detection": 0, "dann": 0, "consistency": 0}
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in tqdm(loader, leave=False, desc="train" if train else "val "):
            labels = batch["labels"].to(device)
            exp_ids = torch.tensor([exp_label_map.get(g, 0) for g in batch["group"]],
                                   dtype=torch.long, device=device)
            out = model(batch)
            ls = loss_fn(out, labels, exp_ids,
                         batch["thermal_valid"].to(device),
                         batch["sequence_valid"].to(device))

            if train and optimizer:
                optimizer.zero_grad()
                ls["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                optimizer.step()

            for k in losses:
                losses[k] += ls[k].item()
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in losses.items()}


def prepare_data(df, tabular_cols, dxp_cols):
    """Fit scaler, compute medians, return Dataset + DataLoader."""
    scaler = StandardScaler()
    tab_vals = [load_tabular_vector(row, tabular_cols) for _, row in df.iterrows()]
    arr = np.array(tab_vals)
    # Replace any NaN / inf with column median
    col_med = np.nanmedian(arr, axis=0, keepdims=True)
    arr = np.nan_to_num(arr, nan=col_med, posinf=col_med, neginf=col_med)
    scaler.fit(arr)
    tab_medians = {}
    for c in tabular_cols:
        arr_c = df[c].values
        # Get only valid scalar values
        valid = []
        for v in arr_c:
            if not isinstance(v, (np.ndarray, list)):
                try:
                    valid.append(float(v))
                except (ValueError, TypeError):
                    pass
        tab_medians[c] = float(np.median(valid)) if valid else 0.0
    roi_medians = {c: float(df[c].median()) for c in C.THERMAL_ROI_COLS if c in df.columns}
    ds = MultiModalDataset(df, tabular_cols, dxp_cols,
                           tabular_medians=tab_medians, roi_medians=roi_medians,
                           scaler=scaler)
    loader = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)
    return loader, scaler, tab_medians, roi_medians


def train(model, df, tabular_cols, dxp_cols, exp_label_map, n_epochs=20, device=None):
    """Full training loop. Returns model + history."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader, scaler, tab_meds, roi_meds = prepare_data(df, tabular_cols, dxp_cols)
    pos_weight = compute_pos_weights(df, device)
    loss_fn = TotalLoss(pos_weight=pos_weight, n_experiments=len(exp_label_map)).to(device)

    bp = [p for n, p in model.named_parameters() if "backbone" in n]
    hp = [p for n, p in model.named_parameters() if "backbone" not in n]
    optimizer = torch.optim.AdamW([
        {"params": bp, "lr": C.LR_BACKBONE},
        {"params": hp, "lr": C.LR_HEAD},
    ], weight_decay=C.WEIGHT_DECAY)

    history = []
    best_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        lam = dann_schedule(epoch, n_epochs)
        model.set_dann_lambda(lam)
        loss_fn.set_dann_lambda(lam)
        model.set_epoch(epoch)

        losses = train_epoch(model, loader, loss_fn, optimizer, device, exp_label_map, train=True)
        history.append({**losses, "epoch": epoch + 1, "dann_lambda": lam})

        if losses["total"] < best_loss:
            best_loss = losses["total"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Ep {epoch+1:02d}/{n_epochs}  loss={losses['total']:.4f}  "
                  f"det={losses['detection']:.4f}  dann={losses['dann']:.4f}  "
                  f"cons={losses['consistency']:.4f}  λ={lam:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    return model, history
