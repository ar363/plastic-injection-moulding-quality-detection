"""
models.py — All network components for the 4-modal fusion system.

Architecture:
  ThermalEncoder  (EfficientNet-B0 + ROI physics head) → 512-dim
  VisualEncoder   (ResNet-50 shared, 3 views + cross-attn) → 512-dim
  TCNEncoder      (causal dilated convs + SE-attention) → 256-dim
  TabularEncoder  (3-layer MLP) → 192-dim
  CrossModalFusion (Transformer over 4 tokens) → 256-dim
  DefectHead + DANNHead
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as tvm

from . import config as C


# =============================================================================
# Gradient Reversal (DANN)
# =============================================================================

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam=1.0):
        ctx.lam = lam
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None

def grad_reverse(x, lam=1.0):
    return GradientReversal.apply(x, lam)


# =============================================================================
# Thermal Encoder
# =============================================================================

class ROIPhysicsHead(nn.Module):
    def __init__(self, in_dim=10, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(True),
            nn.Dropout(0.2), nn.Linear(hidden, out_dim), nn.ReLU(True),
        )
    def forward(self, x):
        return self.net(x)

class ThermalEncoder(nn.Module):
    """EfficientNet-B0 → 448 + ROI MLP(10→64) → concat → 512"""
    def __init__(self, freeze_epochs=3):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True,
                                          num_classes=0, in_chans=3)
        self.cnn_head = nn.Sequential(
            nn.Linear(1280, 448), nn.BatchNorm1d(448), nn.ReLU(True), nn.Dropout(0.3))
        self.roi_head = ROIPhysicsHead(in_dim=10, hidden=64, out_dim=64)
        self.fusion = nn.Sequential(
            nn.Linear(448+64, 512), nn.LayerNorm(512), nn.ReLU(True))
        self._freeze_ep = freeze_epochs
        self._ep = 0

    def set_epoch(self, ep):
        self._ep = ep
        for p in self.backbone.parameters():
            p.requires_grad = ep >= self._freeze_ep

    def forward(self, img, roi):
        f = self.backbone(img)
        f = self.cnn_head(f)
        r = self.roi_head(roi)
        return self.fusion(torch.cat([f, r], dim=1))


# =============================================================================
# Visual Encoder
# =============================================================================

class CrossSectionAttention(nn.Module):
    """Self-attn over 3 sections + learnable CLS token."""
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x):
        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.norm(x)
        out, w = self.attn(x, x, x)
        return out[:, 0], w[:, 0, 1:]          # CLS output, CLS→section weights

class VisualEncoder(nn.Module):
    """ResNet-50 shared across 3 sections → cross-section attn → 512."""
    def __init__(self, freeze_epochs=3):
        super().__init__()
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        old = base.conv1
        base.conv1 = nn.Conv2d(1, old.out_channels, old.kernel_size,
                               old.stride, old.padding, bias=False)
        with torch.no_grad():
            base.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.proj = nn.Sequential(
            nn.Flatten(), nn.Linear(2048, 512), nn.BatchNorm1d(512),
            nn.ReLU(True), nn.Dropout(0.3))
        self.pos_emb = nn.Embedding(3, 512)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.cross = CrossSectionAttention(512, n_heads=4)
        self.out = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(True))
        self._freeze_ep = freeze_epochs
        self._ep = 0

    def set_epoch(self, ep):
        self._ep = ep
        for p in self.backbone.parameters():
            p.requires_grad = ep >= self._freeze_ep

    def forward(self, imgs):
        sections = []
        for i in range(3):
            f = self.backbone(imgs[:, i]).squeeze(-1).squeeze(-1)
            e = self.proj(f) + self.pos_emb(torch.tensor(i, device=imgs.device))
            sections.append(e)
        tokens = torch.stack(sections, dim=1)
        pooled, attn = self.cross(tokens)
        return self.out(pooled), attn


# =============================================================================
# TCN Encoder
# =============================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation=1):
        super().__init__()
        self.pad = (kernel-1)*dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=0))

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        h = max(channels//reduction, 4)
        self.fc = nn.Sequential(nn.Linear(channels, h), nn.ReLU(True),
                                 nn.Linear(h, channels))

    def forward(self, x):
        w = torch.sigmoid(self.fc(x.mean(dim=2)))
        return x * w.unsqueeze(2)

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.se = SqueezeExcitation(out_ch)
        self.norm1, self.norm2 = nn.BatchNorm1d(out_ch), nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        r = self.res(x)
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.se(x)
        return F.relu(x + r)

class TCNEncoder(nn.Module):
    """Causal temporal conv net → 256-dim pooled embedding."""
    def __init__(self, in_channels=8, channel_list=None, kernel_size=3,
                 dilations=None, dropout=0.2):
        super().__init__()
        channel_list = channel_list or C.TCN_CHANNEL_LIST
        dilations = dilations or C.TCN_DILATIONS
        blocks = []
        in_ch = in_channels
        for out_ch in channel_list:
            for d in dilations:
                blocks.append(TCNBlock(in_ch, out_ch, kernel_size, d, dropout))
                in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)
        final_ch = channel_list[-1]
        self.proj = nn.Sequential(
            nn.Linear(final_ch, C.TCN_EMB_DIM), nn.LayerNorm(C.TCN_EMB_DIM), nn.ReLU(True))

    def forward(self, x):
        f = self.tcn(x)
        return self.proj(f.mean(dim=2))


# =============================================================================
# Tabular Encoder
# =============================================================================

class TabularEncoder(nn.Module):
    """3-layer MLP: in_dim → 256 → 256 → 192"""
    def __init__(self, in_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(256, C.TABULAR_EMB_DIM), nn.LayerNorm(C.TABULAR_EMB_DIM), nn.ReLU(True),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Cross-Modal Fusion Transformer
# =============================================================================

MOD_THERMAL, MOD_VISUAL, MOD_SEQUENCE, MOD_TABULAR = 0, 1, 2, 3
N_MODALITIES = 4

class CrossModalFusion(nn.Module):
    """Transformer over 4 modality tokens with [MASK] for missing modals."""

    def __init__(self):
        super().__init__()
        d = C.FUSION_TOKEN_DIM
        self.proj_t = nn.Linear(C.THERMAL_EMB_DIM, d)
        self.proj_v = nn.Linear(C.VISUAL_EMB_DIM, d)
        self.proj_s = nn.Linear(C.TCN_EMB_DIM, d)
        self.proj_b = nn.Linear(C.TABULAR_EMB_DIM, d)
        self.mod_emb = nn.Embedding(N_MODALITIES, d)
        nn.init.normal_(self.mod_emb.weight, std=0.02)
        self.mask = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.mask, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=C.FUSION_N_HEADS, dim_feedforward=d*4,
            dropout=C.FUSION_DROPOUT, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=C.FUSION_N_LAYERS)
        self.norm = nn.LayerNorm(d)
        self.attn_head = nn.Linear(d, 1)

    def forward(self, th_emb, vi_emb, tc_emb, tb_emb,
                th_v, vi_v, tc_v, tb_v):
        B = th_v.shape[0]
        device = th_v.device
        proj = [(self.proj_t(th_emb), th_v, MOD_THERMAL),
                (self.proj_v(vi_emb), vi_v, MOD_VISUAL),
                (self.proj_s(tc_emb), tc_v, MOD_SEQUENCE),
                (self.proj_b(tb_emb), tb_v, MOD_TABULAR)]
        tokens, valid = [], []
        for p, valid_mask, mi in proj:
            me = self.mod_emb(torch.tensor(mi, device=device))
            tok = torch.where(valid_mask.unsqueeze(1), p + me, self.mask.squeeze(0) + me)
            tokens.append(tok)
            valid.append(valid_mask)
        tokens = torch.stack(tokens, dim=1)
        vf = torch.stack(valid, dim=1).float()

        out = self.transformer(tokens)
        out = self.norm(out)
        scores = self.attn_head(out).squeeze(-1)
        scores = scores.masked_fill(vf < 0.5, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        fused = (out * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights


# =============================================================================
# Full 4-Modal System
# =============================================================================

class FusionModel(nn.Module):
    """Complete 4-modal fusion with defect head + DANN head."""

    def __init__(self, tabular_in_dim, n_dxp_channels=8, n_experiments=30):
        super().__init__()
        self.thermal = ThermalEncoder()
        self.visual  = VisualEncoder()
        self.tcn     = TCNEncoder(in_channels=n_dxp_channels)
        self.tabular = TabularEncoder(in_dim=tabular_in_dim)
        self.fusion  = CrossModalFusion()
        d = C.FUSION_TOKEN_DIM
        self.defect_head = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(128, C.N_LABELS))
        self.dann_head = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(128, n_experiments))
        self.dann_lambda = 0.0

    def set_epoch(self, ep):
        self.thermal.set_epoch(ep)
        self.visual.set_epoch(ep)

    def set_dann_lambda(self, lam):
        self.dann_lambda = lam

    def forward(self, batch):
        device = next(self.parameters()).device
        t_v = batch["thermal_valid"].to(device)
        v_v = batch["visual_valid"].to(device)
        s_v = batch["sequence_valid"].to(device)
        b_v = batch["tabular_valid"].to(device)

        th = self.thermal(batch["thermal_img"].to(device),
                          batch["thermal_roi"].to(device))
        vi, sec_attn = self.visual(batch["visual_imgs"].to(device))
        tc = self.tcn(batch["sequence"].to(device))
        tb = self.tabular(batch["tabular"].to(device))

        fused, tok_w = self.fusion(th, vi, tc, tb, t_v, v_v, s_v, b_v)

        defect = self.defect_head(fused)
        if self.dann_lambda > 0 and self.training:
            dann = self.dann_head(grad_reverse(fused, self.dann_lambda))
        else:
            dann = self.dann_head(fused)

        return {
            "defect_logits": defect,
            "dann_logits":   dann,
            "fused_emb":     fused,
            "token_weights": tok_w,
            "section_attn":  sec_attn,
            "thermal_emb":   th,
            "tcn_emb":       tc,
        }
