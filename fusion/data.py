"""
data.py — Dataset, collation, and all modality loaders.

Thermal CSV  → float32 (3, H, W)   [frame1, frame2, frame1-frame2]
CV BMP      → float32 (3, 1, H, W)  [3 sections, greyscale]
DXP (array) → float32 (n_channels, TCN_TARGET_LEN)
Tabular     → float32 (n_features,) scalars

Missing-modality handling:
  - Each modality has a boolean "valid" flag
  - Fusion model uses [MASK] tokens for missing modalities
"""

import re, struct, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from PIL import Image

from . import config as C

log = logging.getLogger(__name__)

# =============================================================================
# Parquet loader
# =============================================================================

def load_parquet(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or C.PARQUET_PATH
    df = pd.read_parquet(path)
    log.info(f"Parquet: {df.shape[0]} rows × {df.shape[1]} cols")
    for col in C.LABEL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32)
    log.info(f"  NOK rate: {df[C.PRIMARY_TARGET].mean():.2%}")
    return df


# =============================================================================
# Thermal CSV loader
# =============================================================================

def load_thermal_csv(csv_path: Path) -> np.ndarray:
    """Load Thermographie CSV → (480, 640) float32 °C."""
    with open(csv_path, encoding="utf-8-sig") as fh:
        raw = fh.read()
    rows = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    rows = [r for r in rows if r.strip()]
    data = []
    for row in rows:
        vals = [float(v.replace(",", ".")) for v in row.split(";") if v.strip()]
        if vals:
            data.append(vals)
    return np.array(data, dtype=np.float32)


def load_thermal_pair(row: pd.Series) -> Optional[np.ndarray]:
    """Load 2 frames → (3, H, W) [f1_normed, f2_normed, f1-f2 shift]. Returns None if missing."""
    name1 = row.get(C.IR_IMG1_COL)
    name2 = row.get(C.IR_IMG2_COL)
    if not isinstance(name1, str) or not isinstance(name2, str):
        return None
    p1, p2 = C.THERMAL_CSV_DIR / name1, C.THERMAL_CSV_DIR / name2
    if not p1.exists() or not p2.exists():
        return None

    f1, f2 = load_thermal_csv(p1), load_thermal_csv(p2)

    def norm(x):
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        return np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    f1n, f2n = norm(f1), norm(f2)
    diff = np.clip(f1n - f2n, -1.0, 1.0) * 0.5 + 0.5
    return np.stack([f1n, f2n, diff], axis=0).astype(np.float32)


def extract_thermal_rois(row: pd.Series, medians: Optional[Dict] = None) -> np.ndarray:
    """10-dim ROI vector from parquet row."""
    vals = []
    for col in C.THERMAL_ROI_COLS:
        v = row.get(col, np.nan)
        if pd.isna(v) and medians is not None:
            v = medians.get(col, 0.0)
        vals.append(float(v) if not pd.isna(v) else 0.0)
    return np.array(vals, dtype=np.float32)


# =============================================================================
# CV Visual (BMP) loader
# =============================================================================

def load_bmp(path: Path) -> np.ndarray:
    """BMP → uint8 (H, W) greyscale."""
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def _parse_xml_cycle_id(xml_path: Path) -> Optional[float]:
    try:
        content = xml_path.read_text(errors="ignore")
        m = re.search(r'CycleID.*?>([0-9.]+)<', content)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def build_cv_index() -> pd.DataFrame:
    """Build cycle_id → {1: bmp, 2: bmp, 3: bmp} from XML files."""
    xml_dir = C.CV_IMAGE_DIR
    records: Dict[float, Dict[int, Path]] = {}
    for xml_file in sorted(xml_dir.glob("*.xml")):
        cid = _parse_xml_cycle_id(xml_file)
        if cid is None or cid == 0.0:
            continue
        bmp = xml_file.with_suffix(".bmp")
        if not bmp.exists():
            continue
        try:
            content = xml_file.read_text(errors="ignore")
            pm = re.search(r'Position.*?>(\d)<', content)
            pos = int(pm.group(1)) if pm else 1
        except Exception:
            pos = 1
        records.setdefault(cid, {})[pos] = bmp

    rows = [{"cv_cycle_id": cid,
             "pos1_path": str(m.get(1, "")),
             "pos2_path": str(m.get(2, "")),
             "pos3_path": str(m.get(3, ""))}
            for cid, m in records.items()]
    df = pd.DataFrame(rows).sort_values("cv_cycle_id").reset_index(drop=True)
    log.info(f"CV index: {len(df)} unique cycles")
    return df


def join_cv_to_parquet(parquet_df: pd.DataFrame, cv_index: pd.DataFrame,
                       tolerance: float = 0.02) -> pd.DataFrame:
    """Fuzzy-join CV images ↔ parquet via QUA_CycleTime."""
    if "QUA_CycleTime" not in parquet_df.columns:
        parquet_df["pos1_path"] = ""
        parquet_df["pos2_path"] = ""
        parquet_df["pos3_path"] = ""
        return parquet_df
    left = parquet_df.copy().sort_values("QUA_CycleTime")
    right = cv_index.sort_values("cv_cycle_id")
    merged = pd.merge_asof(left, right, left_on="QUA_CycleTime",
                           right_on="cv_cycle_id", tolerance=tolerance,
                           direction="nearest")
    n = (merged["pos1_path"].fillna("") != "").sum()
    log.info(f"CV join: {n}/{len(merged)} matched")
    return merged


def load_cv_triple(row: pd.Series) -> Optional[np.ndarray]:
    """3 section images → (3, H, W) float32 [0,1]. None if any missing."""
    paths = []
    for k in ["pos1_path", "pos2_path", "pos3_path"]:
        p = row.get(k, None)
        if not isinstance(p, str) or not p.strip():
            return None
        pp = Path(p)
        if not pp.exists():
            return None
        paths.append(pp)
    sections = [load_bmp(p).astype(np.float32) / 255.0 for p in paths]
    return np.stack(sections, axis=0)


# =============================================================================
# Sequence / tabular helpers
# =============================================================================

def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Linear-interp resample to target_len."""
    if seq.ndim == 1:
        x_old = np.linspace(0, 1, len(seq))
        x_new = np.linspace(0, 1, target_len)
        return interp1d(x_old, seq, kind="linear", fill_value="extrapolate")(x_new).astype(np.float32)
    out = np.zeros((seq.shape[0], target_len), dtype=np.float32)
    for i in range(seq.shape[0]):
        out[i] = resample_sequence(seq[i], target_len)
    return out


def load_dxp_tensor(row: pd.Series, channel_cols: List[str]) -> Optional[np.ndarray]:
    """DXP channels → (n_channels, TCN_TARGET_LEN) float32. None if any channel missing."""
    channels = []
    for col in channel_cols:
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        if not hasattr(val, "__len__"):
            return None
        channels.append(resample_sequence(np.array(val, dtype=np.float32), C.TCN_TARGET_LEN))
    return np.stack(channels, axis=0)


def load_tabular_vector(row: pd.Series, tabular_cols: List[str],
                         medians: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Scalar features → (n_features,) float32, imputed with medians."""
    vals = []
    for col in tabular_cols:
        v = row.get(col, np.nan)
        if isinstance(v, (np.ndarray, list)):
            v = np.nan
        try:
            if pd.isna(v) or np.isnan(float(v)):
                v = medians.get(col, 0.0) if medians else 0.0
            vals.append(float(v))
        except (ValueError, TypeError, OverflowError):
            vals.append(0.0)
    arr = np.array(vals, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# =============================================================================
# Column detection
# =============================================================================

def get_tabular_cols(df: pd.DataFrame) -> List[str]:
    """Detect scalar tabular columns (SET_, QUA_, ENV_, CALC_, DOS_, DRY_, SIM_).
    Checks ALL rows: any column that contains a single scalar (not array) value
    in at least 90% of rows is considered tabular."""
    candidates = [c for c in df.columns
                  if any(c.startswith(p) for p in C.TABULAR_PREFIXES)
                  and c not in C.LABEL_COLS]
    scalar_cols = []
    for c in candidates:
        # Count non-array, non-list values
        try:
            vals = df[c]
            n_scalar = 0
            for v in vals:
                if isinstance(v, (np.ndarray, list)):
                    continue
                try:
                    _ = float(v)
                    n_scalar += 1
                except (ValueError, TypeError, OverflowError):
                    continue
            if n_scalar >= len(df) * 0.9:  # 90% threshold
                scalar_cols.append(c)
        except Exception:
            continue
    return scalar_cols


def get_dxp_cols(df: pd.DataFrame) -> List[str]:
    """DXP channel columns."""
    avail = [c for c in C.KEY_DXP_CHANNELS if c in df.columns]
    if len(avail) < 3:
        avail = [c for c in df.columns if c.startswith(C.SEQUENCE_PREFIX)
                 and c not in C.LABEL_COLS]
    return avail


# =============================================================================
# Dataset
# =============================================================================

class MultiModalDataset(Dataset):
    """One injection moulding cycle = one sample.

    Returns dict with keys:
      thermal_img, thermal_roi, thermal_valid
      visual_imgs, visual_valid
      sequence, sequence_valid
      tabular, tabular_valid
      labels, group, cycle_id, material
    """

    def __init__(self, df: pd.DataFrame, tabular_cols: List[str], dxp_cols: List[str],
                 tabular_medians: Optional[Dict] = None,
                 roi_medians: Optional[Dict] = None,
                 scaler=None):
        self.df = df.reset_index(drop=True)
        self.tabular_cols = tabular_cols
        self.dxp_cols = dxp_cols
        self.tab_medians = tabular_medians or {}
        self.roi_medians = roi_medians or {}
        self.scaler = scaler

        h, w = C.THERMAL_INPUT_SIZE, C.THERMAL_INPUT_SIZE
        self._zero_thermal  = torch.zeros(3, h, w)
        self._zero_visual   = torch.zeros(3, 1, C.CV_INPUT_SIZE, C.CV_INPUT_SIZE)
        self._zero_sequence = torch.zeros(len(dxp_cols), C.TCN_TARGET_LEN)
        self._zero_tabular  = torch.zeros(len(tabular_cols))
        self._zero_roi      = torch.zeros(len(C.THERMAL_ROI_COLS))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        labels = torch.tensor([row.get(c, 0.0) for c in C.LABEL_COLS], dtype=torch.float32)

        # --- Tabular ---
        tv = load_tabular_vector(row, self.tabular_cols, self.tab_medians)
        if self.scaler is not None:
            tv = self.scaler.transform(tv.reshape(1, -1)).flatten().astype(np.float32)
        tab = torch.from_numpy(tv)
        tab_valid = torch.isfinite(tab).all().item()

        # --- DXP sequence ---
        seq_arr = load_dxp_tensor(row, self.dxp_cols)
        seq_valid = seq_arr is not None
        seq = torch.from_numpy(seq_arr) if seq_valid else self._zero_sequence.clone()

        # --- Thermal ---
        th_arr = load_thermal_pair(row)
        th_valid = th_arr is not None
        th = torch.from_numpy(th_arr) if th_valid else self._zero_thermal.clone()

        roi = torch.from_numpy(extract_thermal_rois(row, self.roi_medians))

        # --- Visual ---
        cv_arr = load_cv_triple(row)
        cv_valid = cv_arr is not None
        if cv_valid:
            sections = [torch.from_numpy((cv_arr[i] * 255).astype(np.uint8)).unsqueeze(0).float() / 255.0
                        for i in range(3)]
            vis = torch.stack(sections, dim=0)
        else:
            vis = self._zero_visual.clone()

        return {
            "thermal_img":    th,          "thermal_roi":    roi,
            "thermal_valid":  th_valid,     "visual_imgs":    vis,
            "visual_valid":   cv_valid,     "sequence":       seq,
            "sequence_valid": seq_valid,    "tabular":        tab,
            "tabular_valid":  tab_valid,    "labels":         labels,
            "group":          str(row.get(C.GROUP_COL, "unknown")),
            "cycle_id":       int(row.get(C.CYCLE_ID_COL, -1)),
            "material":       str(row.get(C.MATERIAL_COL, "unknown")),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    out = {}
    for k in batch[0].keys():
        vals = [s[k] for s in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        elif isinstance(vals[0], bool):
            out[k] = torch.tensor(vals, dtype=torch.bool)
        elif isinstance(vals[0], (int, float)):
            out[k] = torch.tensor(vals)
        else:
            out[k] = vals
    return out
