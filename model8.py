# -*- coding: utf-8 -*-
"""
Enhanced 7-Model Ensemble (TimesNet, N-HiTS, GRU, DLinear, NLinear, TCN, iTransformer)
+ Horizon-wise Softmax Stacking (OOF-based)
+ Global MA/EMA Rolling Features (leak-safe)
+ Optional Optuna, Pruning, Coordinate-Ascent Weight Tuning
"""

import os, gc, glob, json, math, random, warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ---- Optuna (옵션) ----
try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

warnings.filterwarnings('ignore')

# =====================
# Config
# =====================
@dataclass
class EnhancedNHiTSConfig:
    # 경로
    train_csv: str = "LG_train.csv"
    test_glob: str = "test/*.csv"
    submission_template_csv: str = "sample_submission.csv"
    out_submission_csv: str = "LG_output/0815_enhanced_stacking_submission.csv"
    train_log_csv: str = "LG_output/epoch_log.csv"

    # 컬럼명
    date_col: str = "영업일자"
    item_col: str = "영업장명_메뉴명"
    target_col: str = "매출수량"

    # 윈도우
    in_len: int = 28
    out_len: int = 7

    # 학습 설정
    train_end_date: str = "2024-06-15"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log1p: bool = True

    # 공용 학습 하이퍼파라미터(풀 학습)
    EPOCHS_FULL: int = 100
    BATCH_FULL: int = 256
    BASE_LR_FULL: float = 6e-4
    MAX_LR_FULL: float = 1.5e-3
    WD_FULL: float = 6e-4

    # 튜닝 (옵션)
    USE_OPTUNA: bool = False
    N_TRIALS: int = 30
    EPOCHS_TUNE: int = 40
    BATCH_TUNE: int = 256

    # CV folds(끝나는 주의 토요일 등)
    cv_fold_end_dates: Tuple[str, ...] = ("2024-06-14", "2024-06-07", "2024-05-31")

    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False  # 메모리 안전 우선

    # N-HiTS 구조
    hidden: int = 384
    n_blocks: int = 3
    n_layers: int = 1
    n_pool_kernel_size: Optional[List[int]] = None
    pooling_mode: str = "MaxPool1d"
    interpolation_mode: str = "linear"
    dropout: float = 0.22
    stack_types: Optional[List[str]] = None
    n_freq_downsample: Optional[List[int]] = None

    # Loss
    eps_smape: float = 0.01
    zero_weight: float = 0.01
    hurdle_lambda: float = 0.15

    # AMP/EMA
    use_amp: bool = True
    use_compile: bool = False
    ema_decay: float = 0.9998

    # 가중치/공휴일
    store_weights: Optional[Dict[str, float]] = None
    custom_holidays_list: Optional[List[str]] = None

    # ==== PatchTST ====
    ptst_d_model: int = 192
    ptst_nhead: int = 8
    ptst_num_layers: int = 2
    ptst_patch_len: int = 3
    ptst_stride: int = 2
    ptst_dropout: float = 0.2
    ptst_head_hidden: int = 256
    ptst_ff_mult: float = 1.5

    # ==== TimesNet ====
    tnet_channels: int = 128
    tnet_blocks: int = 3
    tnet_kernels: Tuple[int, int, int] = (3, 5, 7)
    tnet_dropout: float = 0.12
    tnet_head_hidden: int = 256

    # ==== GRU ====
    gru_hidden: int = 256
    gru_layers: int = 2
    gru_dropout: float = 0.1
    gru_bidirectional: bool = False
    gru_head_hidden: int = 256

    # ==== DLinear ====
    dlin_head_hidden: int = 256

    # ==== 훈련 보조 ====
    grad_clip: float = 0.5
    earlystop_patience_ratio: float = 0.12
    earlystop_patience_min: int = 8


DEFAULT_STORE_WEIGHTS = {
    "미라시아": 7.71, "담하": 6.51, "연회장": 3.48, "라그로타": 3.44,
    "느티나무 셀프BBQ": 2.78, "화담숲주막": 1.43, "카페테리아": 1.31,
    "화담숲카페": 1.14, "포레스트릿": 1.00,
}

DEFAULT_CUSTOM_HOLIDAYS = [
    '2023-01-01','2023-01-21','2023-01-22','2023-01-23','2023-01-24','2023-03-01','2023-05-01',
    '2023-05-05','2023-05-27','2023-06-06','2023-08-15','2023-09-28','2023-09-29','2023-09-30',
    '2023-10-02','2023-10-03','2023-10-09','2023-12-25',
    '2024-01-01','2024-02-09','2024-02-10','2024-02-11','2024-02-12','2024-03-01','2024-04-10',
    '2024-05-01','2024-05-05','2024-05-06','2024-05-15','2024-06-06','2024-08-15','2024-09-16',
    '2024-09-17','2024-09-18','2024-10-01','2024-10-03','2024-10-09','2024-12-25',
    '2025-01-01','2025-01-28','2025-01-29','2025-01-30','2025-03-01','2025-03-03','2025-05-01',
    '2025-05-05','2025-05-06','2025-06-06','2025-08-15'
]

# =====================
# Feature Utils
# =====================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def sine_cosine_encoding(value: float, max_val: float):
    return math.sin(2 * math.pi * value / max_val), math.cos(2 * math.pi * value / max_val)

def get_menu_category(menu_name: str) -> str:
    menu_lower = menu_name.lower()
    if any(word in menu_lower for word in ['막걸리','소주','맥주','와인','참이슬','처음처럼','카스','하이네켄','버드와이저','스텔라']):
        return 'alcohol'
    elif any(word in menu_lower for word in ['찌개','탕','국밥','라면','해장국','갈비탕']):
        return 'hot_food'
    elif any(word in menu_lower for word in ['삼겹','갈비','목살','bbq','구이','불고기']):
        return 'bbq'
    elif any(word in menu_lower for word in ['아이스크림','식혜','콜라','스프라이트','에이드']):
        return 'dessert_drink'
    elif any(word in menu_lower for word in ['아메리카노','라떼','커피']):
        return 'coffee'
    elif any(word in menu_lower for word in ['냉면','파스타','스파게티','면','우동']):
        return 'noodles'
    elif any(word in menu_lower for word in ['비빔밥','볶음밥','공깃밥','정식']):
        return 'rice'
    else:
        return 'others'

def get_store_type(store_name: str) -> str:
    if store_name == "느티나무 셀프BBQ": return 'outdoor'
    if store_name in ["라그로타","미라시아"]: return 'fine_dining'
    if store_name == "담하": return 'traditional'
    if store_name == "연회장": return 'event'
    if store_name in ["카페테리아","포레스트릿","화담숲카페"]: return 'casual'
    return 'specialty'

def build_enhanced_features(dates: List[pd.Timestamp], holidays_set: set) -> pd.DataFrame:
    df = pd.DataFrame({"date": dates})
    df["dow"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["week"] = df["date"].apply(lambda d: int(d.isocalendar().week))
    df["quarter"] = df["date"].dt.quarter
    df["is_friday"] = (df["dow"] == 4).astype(int)
    df["is_saturday"] = (df["dow"] == 5).astype(int)
    df["is_sunday"] = (df["dow"] == 6).astype(int)
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
    df["is_holiday"] = df["date"].isin(holidays_set).astype(int)
    df["is_off"] = ((df["is_weekend"]==1) | (df["is_holiday"]==1)).astype(int)
    df["tomorrow"] = df["date"] + pd.Timedelta(days=1)
    df["yesterday"] = df["date"] - pd.Timedelta(days=1)
    df["is_before_holiday"] = df["tomorrow"].isin(holidays_set).astype(int)
    df["is_after_holiday"] = df["yesterday"].isin(holidays_set).astype(int)
    df["is_month_start"] = (df["day"] <= 3).astype(int)
    df["is_month_end"] = (df["day"] >= 28).astype(int)
    df["is_spring"] = df["month"].isin([4,5,6]).astype(int)
    df["is_summer"] = df["month"].isin([7,8]).astype(int)
    df["is_autumn"] = df["month"].isin([9,10,11]).astype(int)
    df["is_winter"] = df["month"].isin([12,1,2,3]).astype(int)
    df["is_summer_vacation"] = df["month"].isin([7,8]).astype(int)
    df["is_winter_vacation"] = df["month"].isin([12,1,2]).astype(int)
    df[["month_sin","month_cos"]] = df["month"].apply(lambda m: pd.Series(sine_cosine_encoding(m,12)))
    df[["dow_sin","dow_cos"]] = df["dow"].apply(lambda d: pd.Series(sine_cosine_encoding(d,7)))
    df[["doy_sin","doy_cos"]] = df["date"].apply(lambda d: pd.Series(sine_cosine_encoding(d.dayofyear,365)))
    df[["week_sin","week_cos"]] = df["week"].apply(lambda w: pd.Series(sine_cosine_encoding(w,52)))
    df[["quarter_sin","quarter_cos"]] = df["quarter"].apply(lambda q: pd.Series(sine_cosine_encoding(q,4)))
    return df.drop(columns=["tomorrow","yesterday"])

def parse_store_name(item_full: str) -> str:
    return item_full.split("_")[0]

def parse_menu_name(item_full: str) -> str:
    return item_full.split("_", 1)[1] if "_" in item_full else item_full

def make_val_mask_by_week(dataset: 'EnhancedNHiTSDataset', end_date_str: str) -> np.ndarray:
    end_date = pd.to_datetime(end_date_str)
    start_date = end_date - pd.Timedelta(days=6)
    ted = dataset.target_end_dates
    return (ted >= start_date) & (ted <= end_date)

# --- 롤링 피처(글로벌), 누수 방지: shift(1) ---
def add_rolling_feats_safe(series: pd.Series, windows=(3,7,14)) -> pd.DataFrame:
    out = {}
    s = series.astype(float).copy()
    for w in windows:
        out[f"ma_{w}"]  = s.rolling(w, min_periods=1).mean().shift(1)
        out[f"ema_{w}"] = s.ewm(span=w, adjust=False).mean().shift(1)
    return pd.DataFrame(out)

# =====================
# Dataset
# =====================
class EnhancedNHiTSDataset(Dataset):
    def __init__(self, cfg: EnhancedNHiTSConfig, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df.copy()
        if cfg.date_col not in self.df.columns or cfg.item_col not in self.df.columns or cfg.target_col not in self.df.columns:
            raise ValueError("입력 데이터에 필요한 컬럼이 없습니다.")
        self.df[cfg.date_col] = pd.to_datetime(self.df[cfg.date_col])
        self.df[cfg.target_col] = pd.to_numeric(self.df[cfg.target_col], errors='coerce').fillna(0.0).clip(lower=0)

        pivot = self.df.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index()
        self.items = list(pivot.columns)
        self.dates = list(pivot.index)
        self.values = pivot.fillna(0.0).values.astype(np.float32)

        holidays_set = set(pd.to_datetime(cfg.custom_holidays_list)) if cfg.custom_holidays_list else set()

        # --- 캘린더 + 글로벌 롤링 특징 ---
        caldf = build_enhanced_features(self.dates, holidays_set)
        global_series = pd.Series(np.nanmean(self.values, axis=1), index=self.dates)
        roll_df = add_rolling_feats_safe(global_series, windows=(3,7,14))
        caldf = caldf.join(roll_df).fillna(method="ffill").fillna(0)

        self.cal_feats = caldf.drop(columns=["date"], errors="ignore").values.astype(np.float32)
        self.cal_feat_names = [c for c in caldf.columns if c != "date"]
        self.roll_idx = [i for i, c in enumerate(self.cal_feat_names)
                 if c.startswith(("ma_", "ema_", "rstd_", "rmean_", "roll_"))]
        name_to_pos = {c: i for i, c in enumerate(self.cal_feat_names)}
        # 필수 컬럼이 없으면 안전하게 -1로 표기
        self.idx_dow   = name_to_pos.get("dow", -1)
        self.idx_month = name_to_pos.get("month", -1)

        stores = [parse_store_name(it) for it in self.items]
        menus = [parse_menu_name(it) for it in self.items]

        self.store2idx = {s: i for i, s in enumerate(sorted(set(stores)))}
        self.item_store_idx = np.array([self.store2idx[s] for s in stores], dtype=np.int64)
        self.n_stores = len(self.store2idx)

        menu_categories = [get_menu_category(m) for m in menus]
        self.cat2idx = {c: i for i, c in enumerate(sorted(set(menu_categories)))}
        self.item_cat_idx = np.array([self.cat2idx[c] for c in menu_categories], dtype=np.int64)
        self.n_categories = len(self.cat2idx)

        store_types = [get_store_type(s) for s in stores]
        self.type2idx = {t: i for i, t in enumerate(sorted(set(store_types)))}
        self.item_type_idx = np.array([self.type2idx[t] for t in store_types], dtype=np.int64)
        self.n_types = len(self.type2idx)

        sw = cfg.store_weights or {}
        self.sample_weights = np.array([sw.get(parse_store_name(it), 1.0) for it in self.items], dtype=np.float32)

        self.indices: List[Tuple[int,int]] = []
        self.target_end_dates: List[pd.Timestamp] = []
        T = len(self.dates)
        Lx, Ly = cfg.in_len, cfg.out_len
        cutoff = pd.to_datetime(cfg.train_end_date)
        max_start = max(0, T - (Lx + Ly))

        for j in range(len(self.items)):
            for t0 in range(0, max_start + 1):
                end_date = self.dates[t0 + Lx + Ly - 1]
                if end_date <= cutoff:
                    self.indices.append((t0, j))
                    self.target_end_dates.append(end_date)

        self.target_end_dates = np.asarray(self.target_end_dates, dtype='datetime64[ns]')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cfg = self.cfg
        t0, j = self.indices[idx]
        Lx, Ly = cfg.in_len, cfg.out_len

        x = self.values[t0:t0 + Lx, j]
        y = self.values[t0 + Lx:t0 + Lx + Ly, j]

        if cfg.log1p:
            x_in = np.log1p(x); y_out = np.log1p(y)
        else:
            x_in, y_out = x.copy(), y.copy()

        past_cal = self.cal_feats[t0:t0 + Lx, :]
        fut_cal = self.cal_feats[t0 + Lx:t0 + Lx + Ly, :]  # 롤링 피처 포함
        if self.roll_idx:
            fut_cal[:, self.roll_idx] = past_cal[-1, self.roll_idx][None, :]

        store_idx = self.item_store_idx[j]
        cat_idx = self.item_cat_idx[j]
        type_idx = self.item_type_idx[j]
        sample_w = self.sample_weights[j]

        zero_mask = (y == 0).astype(np.float32)
        pos_mask = (y > 0).astype(np.float32)

        return {
            "x": torch.from_numpy(x_in).float(),
            "y": torch.from_numpy(y_out).float(),
            "past_cal": torch.from_numpy(past_cal).float(),
            "fut_cal": torch.from_numpy(fut_cal).float(),
            "store_idx": torch.tensor(store_idx, dtype=torch.long),
            "cat_idx": torch.tensor(cat_idx, dtype=torch.long),
            "type_idx": torch.tensor(type_idx, dtype=torch.long),
            "sample_w": torch.tensor(sample_w, dtype=torch.float32),
            "zero_mask": torch.from_numpy(zero_mask).float(),
            "pos_mask": torch.from_numpy(pos_mask).float(),
        }

# =====================
# N-HiTS (간결화된 구현)
# =====================
class NHiTSBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, 
                 n_layers: int, dropout: float, pooling_mode: str = "MaxPool1d",
                 n_pool_kernel_size: int = 2, interpolation_mode: str = "linear"):
        super().__init__()
        if pooling_mode == "MaxPool1d":
            self.pooling_layer = nn.MaxPool1d(kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True)
        else:
            self.pooling_layer = nn.AvgPool1d(kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True)
        pooled_size = math.ceil(input_size / n_pool_kernel_size)
        layers = [nn.Linear(pooled_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_pooled = self.pooling_layer(x).squeeze(1)
        h = self.mlp(x_pooled)
        output = self.output_layer(h)
        return output

class EnhancedNHiTSModel(nn.Module):
    def __init__(self, in_len: int, out_len: int, cal_dim: int, n_stores: int,
                 n_categories: int, n_types: int, cfg: EnhancedNHiTSConfig):
        super().__init__()
        if cfg.n_pool_kernel_size is None or len(cfg.n_pool_kernel_size) < cfg.n_blocks:
            base_kernels = [2, 2, 1]
            cfg.n_pool_kernel_size = [base_kernels[i % len(base_kernels)] for i in range(cfg.n_blocks)]
        if cfg.stack_types is None or len(cfg.stack_types) < cfg.n_blocks:
            base_types = ["identity", "identity", "trend"]
            cfg.stack_types = [base_types[i % len(base_types)] for i in range(cfg.n_blocks)]
        if cfg.n_freq_downsample is None or len(cfg.n_freq_downsample) < cfg.n_blocks:
            base_downsample = [2, 1, 1]
            cfg.n_freq_downsample = [base_downsample[i % len(base_downsample)] for i in range(cfg.n_blocks)]

        self.out_len = out_len
        self.stack_types = cfg.stack_types

        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb = nn.Embedding(n_categories, 32)
        self.type_emb = nn.Embedding(n_types, 16)
        self.cal_proj = nn.Linear(cal_dim, 128)

        self.blocks = nn.ModuleList([
            NHiTSBlock(in_len, out_len, cfg.hidden, cfg.n_layers, cfg.dropout,
                       pooling_mode=cfg.pooling_mode, n_pool_kernel_size=cfg.n_pool_kernel_size[i],
                       interpolation_mode=cfg.interpolation_mode) for i in range(cfg.n_blocks)
        ])

        self.basis_weights = nn.ParameterList([
            nn.Parameter(torch.randn(out_len, max(1, out_len // cfg.n_freq_downsample[i])))
            for i in range(cfg.n_blocks)
        ])

        meta_dim = 64 + 32 + 16 + 128
        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, cfg.hidden),
            nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, out_len)
        )

        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 1, cfg.hidden),
            nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden // 1, out_len)
        )

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx):
        store_emb = self.store_emb(store_idx)
        cat_emb = self.cat_emb(cat_idx)
        type_emb = self.type_emb(type_idx)
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)
        cal_emb = self.cal_proj(cal_all)
        meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb], dim=-1)

        outputs = []
        for i, block in enumerate(self.blocks):
            block_output = block(x)
            if self.stack_types[i] == "trend":
                basis = self.basis_weights[i]  # [out_len, k]
                block_output = block_output @ basis
                block_output = F.interpolate(block_output.unsqueeze(1), size=self.out_len,
                                            mode='linear', align_corners=False).squeeze(1)
            outputs.append(block_output)
        nhits_output = torch.stack(outputs, dim=0).sum(dim=0)

        meta_adjustment = self.meta_integration(meta_feat)
        value_pred = nhits_output + meta_adjustment

        prob_feat = torch.cat([meta_feat, x.mean(dim=1, keepdim=True)], dim=-1)
        prob_logits = self.prob_head(prob_feat)
        return value_pred, prob_logits

# =====================
# PatchTST (개선)
# =====================
def _build_sinusoidal_pos(L: int, d_model: int, device: torch.device, dtype: torch.dtype):
    pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(d_model, device=device, dtype=dtype).unsqueeze(0)
    div = torch.exp(- (i // 2) * math.log(10000.0) / max(1, (d_model // 2)))
    angle = pos * div
    pe = torch.zeros((L, d_model), device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe.unsqueeze(0)

class PatchEmbedding(nn.Module):
    def __init__(self, seq_len: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        assert patch_len > 0 and stride > 0
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x):  # x: [B, L]
        B, L = x.shape
        if L < self.patch_len:
            pad = self.patch_len - L
            last_val = x[:, -1:].repeat(1, pad)
            x = torch.cat([last_val, x], dim=1)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # [B, Np, P]
        Np = patches.size(1)
        x_e = self.proj(patches)
        pe = _build_sinusoidal_pos(Np, self.d_model, x_e.device, x_e.dtype)
        x_e = x_e + pe
        return x_e

class ImprovedPatchTSTTiny(nn.Module):
    def __init__(self, cfg: EnhancedNHiTSConfig, in_len: int, out_len: int,
                 cal_dim: int, n_stores: int, n_categories: int, n_types: int):
        super().__init__()
        patch_len = max(2, min(cfg.ptst_patch_len, max(2, in_len // 8)))
        stride = max(1, patch_len // 2)
        self.embed = PatchEmbedding(in_len, patch_len, stride, cfg.ptst_d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.ptst_d_model,
            nhead=cfg.ptst_nhead,
            dim_feedforward=int(cfg.ptst_d_model * cfg.ptst_ff_mult),
            dropout=cfg.ptst_dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.ptst_num_layers + 1)

        self.multi_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(cfg.ptst_d_model),
                nn.Linear(cfg.ptst_d_model, max(16, cfg.ptst_head_hidden // 4)),
                nn.GELU(),
                nn.Dropout(cfg.ptst_dropout),
                nn.Linear(max(16, cfg.ptst_head_hidden // 4), out_len)
            ) for _ in range(4)
        ])

        self.time_proj = nn.Sequential(
            nn.Linear(cal_dim * 2, max(16, cfg.ptst_d_model // 2)),
            nn.GELU(),
            nn.Dropout(cfg.ptst_dropout)
        )

        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb = nn.Embedding(n_categories, 32)
        self.type_emb = nn.Embedding(n_types, 16)
        self.cal_proj = nn.Linear(cal_dim, 128)

        meta_dim = 64 + 32 + 16 + 128 + max(16, cfg.ptst_d_model // 2)
        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, cfg.ptst_head_hidden),
            nn.GELU(), nn.Dropout(cfg.ptst_dropout),
            nn.Linear(cfg.ptst_head_hidden, max(32, cfg.ptst_head_hidden // 2)),
            nn.GELU(), nn.Dropout(cfg.ptst_dropout),
            nn.Linear(max(32, cfg.ptst_head_hidden // 2), out_len)
        )
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 1, cfg.ptst_head_hidden),
            nn.GELU(), nn.Dropout(cfg.ptst_dropout),
            nn.Linear(cfg.ptst_head_hidden, max(32, cfg.ptst_head_hidden // 2)),
            nn.GELU(), nn.Dropout(cfg.ptst_dropout),
            nn.Linear(max(32, cfg.ptst_head_hidden // 2), out_len)
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx):
        z = self.embed(x)
        z = self.encoder(z)
        z_global = z.mean(dim=1)
        z_last = z[:, -1, :]
        z_max, _ = z.max(dim=1)
        z_first = z[:, 0, :]

        head_preds = []
        for i, head in enumerate(self.multi_heads):
            vec = [z_global, z_last, z_max, z_first][i]
            head_preds.append(head(vec))
        seq_out = torch.stack(head_preds, dim=0).mean(dim=0)

        time_info = torch.cat([past_cal.mean(dim=1), fut_cal.mean(dim=1)], dim=-1)
        time_emb = self.time_proj(time_info)

        store_emb = self.store_emb(store_idx)
        cat_emb = self.cat_emb(cat_idx)
        type_emb = self.type_emb(type_idx)
        cal_emb = self.cal_proj(torch.cat([past_cal, fut_cal], dim=1).mean(dim=1))
        meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb, time_emb], dim=-1)

        meta_adjustment = self.meta_integration(meta_feat)
        value_pred = seq_out + self.residual_weight * meta_adjustment

        prob_logits = self.prob_head(torch.cat([meta_feat, x.mean(dim=1, keepdim=True)], dim=-1))
        return value_pred, prob_logits

# =====================
# Improved TimesNet (경량화 버전)
# =====================
class FrequencyBlock(nn.Module):
    def __init__(self, seq_len: int, top_k: int = 5):
        super().__init__()
        self.seq_len = int(seq_len)
        self.top_k = int(min(max(top_k, 1), max(1, self.seq_len // 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        x_freq = torch.fft.rfft(x32, dim=-1)
        amp = torch.abs(x_freq).mean(dim=1)
        _, top_idx = torch.topk(amp, k=self.top_k, dim=-1)
        x_freq_filt = torch.zeros_like(x_freq)
        for i in range(B):
            x_freq_filt[i, :, top_idx[i]] = x_freq[i, :, top_idx[i]]
        x_filt32 = torch.fft.irfft(x_freq_filt, n=L, dim=-1)
        return x_filt32.to(in_dtype)

class ImprovedTimesBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, seq_len: int, kernels=(3,5,7), top_k: int = 5, dropout: float = 0.1):
        super().__init__()
        self.in_ch, self.out_ch, self.seq_len = int(in_ch), int(out_ch), int(seq_len)
        self.freq_block = FrequencyBlock(seq_len=self.seq_len, top_k=top_k)
        self.period_base = max(2, self.seq_len // 4)
        self.conv2d = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=(3,3), padding=1), nn.GELU(),
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=(3,3), padding=1), nn.GELU(),
        )
        ks = list(kernels); n_br = len(ks)
        per = max(1, self.out_ch // n_br)
        branch_chs = [per] * n_br
        for i in range(self.out_ch - sum(branch_chs)): branch_chs[i % n_br] += 1
        branches = []
        for i, k in enumerate(ks):
            ch = branch_chs[i]
            branches.append(
                nn.Sequential(
                    nn.Conv1d(self.in_ch, ch, kernel_size=k, padding=k//2), nn.GELU(),
                    nn.Conv1d(ch, ch, kernel_size=k, padding=k//2), nn.GELU(),
                )
            )
        self.conv1d_branches = nn.ModuleList(branches)
        self.br_sum_ch = sum(branch_chs)
        self.channel_proj = (nn.Conv1d(self.br_sum_ch, self.out_ch, kernel_size=1) if self.br_sum_ch != self.out_ch else nn.Identity())
        self.norm = nn.LayerNorm([self.out_ch, self.seq_len])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        xf = self.freq_block(x)
        period = self.period_base
        if L % period != 0:
            pad_len = period - (L % period)
            xf = F.pad(xf, (0, pad_len), mode="replicate")
            L_pad = L + pad_len
        else:
            pad_len = 0; L_pad = L
        x2d = xf.reshape(B, C, period, L_pad // period).contiguous()
        z2d = self.conv2d(x2d)
        z2d_flat = z2d.reshape(B, self.out_ch, L_pad)
        if pad_len > 0: z2d_flat = z2d_flat[:, :, :L]
        z1d = torch.cat([br(x) for x, br in [(xf, b) for b in self.conv1d_branches]], dim=1)
        z1d = self.channel_proj(z1d)
        out = z2d_flat + z1d
        out = self.norm(out); out = self.dropout(out)
        return out

class ImprovedTimesNetTiny(nn.Module):
    def __init__(self, cfg: EnhancedNHiTSConfig, in_len: int, out_len: int, cal_dim: int, n_stores: int, n_categories: int, n_types: int):
        super().__init__()
        C = max(32, min(cfg.tnet_channels, 96)); C = (C // 8) * 8
        self.stem = nn.Conv1d(1, C, kernel_size=3, padding=1)
        def fast_block(C: int, p: float):
            return nn.Sequential(
                nn.Conv1d(C, C, kernel_size=3, padding=1, groups=C, bias=True),
                nn.Conv1d(C, C, kernel_size=1, bias=False), nn.GELU(),
                nn.Conv1d(C, C, kernel_size=3, padding=1, groups=C, bias=True),
                nn.Conv1d(C, C, kernel_size=1, bias=False), nn.GELU(),
                nn.Dropout(p)
            )
        self.blocks = nn.ModuleList([fast_block(C, p=max(0.0, min(cfg.tnet_dropout, 0.1))) for _ in range(max(1, cfg.tnet_blocks))])
        head_hidden = max(128, min(cfg.tnet_head_hidden, 192))
        self.head_fc1 = nn.Linear(C, head_hidden); self.head_fc2 = nn.Linear(head_hidden, out_len); self.head_drop = nn.Dropout(cfg.tnet_dropout)
        self.store_emb = nn.Embedding(n_stores, 64); self.cat_emb = nn.Embedding(n_categories, 32); self.type_emb = nn.Embedding(n_types, 16)
        self.cal_proj = nn.Linear(cal_dim, 128)
        meta_dim = 64 + 32 + 16 + 128
        self.meta_integration = nn.Sequential(nn.Linear(meta_dim, head_hidden), nn.ReLU(), nn.Dropout(cfg.tnet_dropout), nn.Linear(head_hidden, out_len))
        self.prob_head = nn.Sequential(nn.Linear(meta_dim + 1, head_hidden), nn.ReLU(), nn.Dropout(cfg.tnet_dropout), nn.Linear(head_hidden, out_len))
        self.meta_res_weight = nn.Parameter(torch.tensor(0.15))

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx):
        z = self.stem(x.unsqueeze(1))
        for blk in self.blocks: z = z + blk(z)
        z_mean = z.mean(dim=-1)
        seq_out = self.head_fc2(self.head_drop(F.gelu(self.head_fc1(z_mean))))
        store_emb = self.store_emb(store_idx); cat_emb = self.cat_emb(cat_idx); type_emb = self.type_emb(type_idx)
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1); cal_emb = self.cal_proj(cal_all)
        meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb], dim=-1)
        value_pred = seq_out + self.meta_res_weight * self.meta_integration(meta_feat)
        prob_logits = self.prob_head(torch.cat([meta_feat, x.mean(dim=1, keepdim=True)], dim=-1))
        return value_pred, prob_logits

# =====================
# GRU / DLinear / NLinear / TCN / iTransformer (생략 없이 사용)
# =====================
class GRUTiny(nn.Module):
    def __init__(self, cfg, in_len, out_len, cal_dim, n_stores, n_categories, n_types):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=cfg.gru_hidden, num_layers=cfg.gru_layers,
                          batch_first=True, dropout=cfg.gru_dropout if cfg.gru_layers>1 else 0.0,
                          bidirectional=cfg.gru_bidirectional)
        d_mul = 2 if cfg.gru_bidirectional else 1
        self.head = nn.Sequential(nn.LayerNorm(cfg.gru_hidden*d_mul), nn.Linear(cfg.gru_hidden*d_mul, out_len))
        self.store_emb = nn.Embedding(n_stores,64); self.cat_emb = nn.Embedding(n_categories,32); self.type_emb = nn.Embedding(n_types,16)
        self.cal_proj = nn.Linear(cal_dim,128)
        meta_dim = 64+32+16+128
        self.meta_integration = nn.Sequential(nn.Linear(meta_dim, cfg.gru_head_hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(cfg.gru_head_hidden, out_len))
        self.prob_head = nn.Sequential(nn.Linear(meta_dim+1, cfg.gru_head_hidden), nn.ReLU(), nn.Dropout(0.1),
                                       nn.Linear(cfg.gru_head_hidden, cfg.gru_head_hidden//2), nn.ReLU(), nn.Dropout(0.1),
                                       nn.Linear(cfg.gru_head_hidden//2, out_len))
    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx):
        z_out,_ = self.gru(x.unsqueeze(-1)); z_last = z_out[:, -1, :]; seq_out = self.head(z_last)
        store = self.store_emb(store_idx); cat = self.cat_emb(cat_idx); typ = self.type_emb(type_idx)
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1); cal = self.cal_proj(cal_all)
        meta = torch.cat([store,cat,typ,cal], dim=-1)
        value_pred = seq_out + self.meta_integration(meta)
        prob_logits = self.prob_head(torch.cat([meta, x.mean(dim=1, keepdim=True)], dim=-1))
        return value_pred, prob_logits

class ImprovedDLinearTiny(nn.Module):
    """
    Decomposition-based linear forecaster with meta features + hurdle logits.
    - Trend/Seasonal linear heads
    - Multi-scale (last 7, 14) linear heads (if available)
    - Calendar effects via safer one-hot using dataset-provided indices (idx_dow, idx_month)
    - Meta integration (store/menu/type/calendar)
    - Prob head for hurdle (zero-inflation)
    """
    def __init__(
        self,
        cfg: EnhancedNHiTSConfig,
        in_len: int,
        out_len: int,
        cal_dim: int,
        n_stores: int,
        n_categories: int,
        n_types: int,
        idx_dow: int = -1,
        idx_month: int = -1,
    ):
        super().__init__()
        self.in_len, self.out_len = int(in_len), int(out_len)

        # --- decomposition (moving average for trend) ---
        self.decomp_kernel = 25
        self.moving_avg = nn.AvgPool1d(
            kernel_size=self.decomp_kernel,
            stride=1,
            padding=self.decomp_kernel // 2,
            count_include_pad=False,
        )

        # --- sequence heads ---
        self.trend_linear = nn.Sequential(
            nn.Linear(in_len, cfg.dlin_head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden, out_len),
        )
        self.seasonal_linear = nn.Sequential(
            nn.Linear(in_len, cfg.dlin_head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden, out_len),
        )

        # --- multi-scale heads (tail windows) ---
        self.ws = [w for w in [7, 14] if w <= in_len]
        self.multi_scale_linears = nn.ModuleList(
            [nn.Linear(w, out_len) for w in self.ws]
        )

        # --- calendar heads (one-hot) ---
        self.idx_dow = int(idx_dow)       # expects 0..6
        self.idx_month = int(idx_month)   # expects 1..12
        self.dow_linear = nn.Linear(7, out_len)
        self.month_linear = nn.Linear(12, out_len)

        # --- meta embeddings ---
        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb = nn.Embedding(n_categories, 32)
        self.type_emb = nn.Embedding(n_types, 16)
        self.cal_proj = nn.Linear(cal_dim, 128)

        meta_dim = 64 + 32 + 16 + 128
        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, cfg.dlin_head_hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden * 2, cfg.dlin_head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden, out_len),
        )

        # --- hurdle probability head (uses simple x stats too) ---
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 3, cfg.dlin_head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden, cfg.dlin_head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden // 2, out_len),
        )

        # --- learnable component weights (trend/seasonal/dow/month + multi-scale) ---
        n_components = 4 + len(self.multi_scale_linears)
        self.component_weights = nn.Parameter(torch.ones(n_components))
        self.component_dropout = nn.Dropout(p=0.15)

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx):
        """
        x:        [B, L]
        past_cal: [B, L, C]
        fut_cal:  [B, H, C]
        returns: value_pred [B, H], prob_logits [B, H]
        """
        B, L = x.size()

        # --- decomposition ---
        # moving average produces trend; AvgPool1d expects [B,1,L]
        trend = self.moving_avg(x.unsqueeze(1)).squeeze(1)
        seasonal = x - trend

        trend_pred = self.trend_linear(trend)          # [B, H]
        seasonal_pred = self.seasonal_linear(seasonal) # [B, H]

        # --- multi-scale tails ---
        multi_scale_preds = []
        for w, linear in zip(self.ws, self.multi_scale_linears):
            # last w time steps → [B, w] → [B, H]
            multi_scale_preds.append(linear(x[:, -w:]))

        # --- calendar effects (safe indexing only) ---
        if self.idx_dow >= 0:
            # dow in [0..6]
            dow_idx = fut_cal[..., self.idx_dow].long().clamp(0, 6)       # [B, H]
            dow_oh = F.one_hot(dow_idx, num_classes=7).float()            # [B, H, 7]
            dow_pred = self.dow_linear(dow_oh.mean(dim=1))                # [B, H]
        else:
            dow_pred = x.new_zeros((B, self.out_len))

        if self.idx_month >= 0:
            # month in [1..12] → shift to [0..11]
            month_idx = (fut_cal[..., self.idx_month] - 1).long().clamp(0, 11)  # [B, H]
            month_oh = F.one_hot(month_idx, num_classes=12).float()             # [B, H, 12]
            month_pred = self.month_linear(month_oh.mean(dim=1))                # [B, H]
        else:
            month_pred = x.new_zeros((B, self.out_len))

        # --- combine components with learned weights ---
        components = [trend_pred, seasonal_pred, dow_pred, month_pred] + multi_scale_preds
        w = F.softmax(self.component_weights[: len(components)], dim=0)          # [K]
        w = self.component_dropout(w)
        w = w / (w.sum() + 1e-8)

        combined_pred = None
        for wi, ci in zip(w, components):
            combined_pred = ci * wi if combined_pred is None else combined_pred + ci * wi  # [B, H]

        # --- meta features ---
        store = self.store_emb(store_idx)     # [B, 64]
        cat = self.cat_emb(cat_idx)           # [B, 32]
        typ = self.type_emb(type_idx)         # [B, 16]
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)  # [B, C]
        cal = self.cal_proj(cal_all)          # [B, 128]
        meta = torch.cat([store, cat, typ, cal], dim=-1)  # [B, meta_dim]

        # --- final value head (additive meta adjustment) ---
        value_pred = combined_pred + self.meta_integration(meta)  # [B, H]

        # --- hurdle probability logits (uses simple series stats) ---
        x_stats = torch.stack(
            [x.mean(dim=1), x.std(dim=1), x.max(dim=1).values], dim=1
        )  # [B, 3]
        prob_logits = self.prob_head(torch.cat([meta, x_stats], dim=-1))  # [B, H]

        return value_pred, prob_logits
    
class NLinearTiny(nn.Module):
    def __init__(self, cfg, in_len, out_len, cal_dim, n_stores, n_categories, n_types):
        super().__init__()
        self.backcast = nn.Linear(in_len, in_len, bias=False)
        self.head = nn.Sequential(nn.LayerNorm(in_len), nn.Linear(in_len, out_len))
        self.store_emb = nn.Embedding(n_stores,64); self.cat_emb = nn.Embedding(n_categories,32); self.type_emb = nn.Embedding(n_types,16)
        self.cal_proj = nn.Linear(cal_dim,128)
        self.meta_head = nn.Sequential(nn.Linear(64+32+16+128, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, out_len))
        self.prob_head = nn.Sequential(nn.Linear(64+32+16+128+1, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, out_len))
    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx):
        z = self.backcast(x) + x; seq_out = self.head(z)
        store = self.store_emb(store_idx); cat = self.cat_emb(cat_idx); typ = self.type_emb(type_idx)
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1); cal = self.cal_proj(cal_all)
        meta = torch.cat([store,cat,typ,cal], dim=-1)
        value_pred  = seq_out + self.meta_head(meta)
        prob_logits = self.prob_head(torch.cat([meta, x.mean(dim=1, keepdim=True)], dim=-1))
        return value_pred, prob_logits

class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, d=1, p=0.1):
        super().__init__()
        pad = (k-1)*d
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad), nn.GELU(), nn.Dropout(p),
            nn.Conv1d(c_out, c_out, kernel_size=k, dilation=d, padding=pad), nn.GELU(),
        )
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in!=c_out else nn.Identity()
        self.norm = nn.LayerNorm(c_out)
    def forward(self, x):
        y = self.net(x) + self.proj(x)
        return self.norm(y.transpose(1,2)).transpose(1,2)

class TCNTiny(nn.Module):
    def __init__(self, cfg, in_len, out_len, cal_dim, n_stores, n_categories, n_types, C=128, depth=4, drop=0.1):
        super().__init__()
        self.stem = nn.Conv1d(1, C, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([_TCNBlock(C, C, k=3, d=2**i, p=drop) for i in range(depth)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(C, out_len))
        self.store_emb = nn.Embedding(n_stores,64); self.cat_emb = nn.Embedding(n_categories,32); self.type_emb = nn.Embedding(n_types,16)
        self.cal_proj = nn.Linear(cal_dim,128)
        self.meta_head = nn.Sequential(nn.Linear(64+32+16+128, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, out_len))
        self.prob_head = nn.Sequential(nn.Linear(64+32+16+128+1, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, out_len))
    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx):
        z = self.stem(x.unsqueeze(1))
        for blk in self.blocks: z = blk(z)
        seq_out = self.head(z)
        store = self.store_emb(store_idx); cat = self.cat_emb(cat_idx); typ = self.type_emb(type_idx)
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1); cal = self.cal_proj(cal_all)
        meta = torch.cat([store,cat,typ,cal], dim=-1)
        value_pred  = seq_out + self.meta_head(meta)
        prob_logits = self.prob_head(torch.cat([meta, x.mean(dim=1, keepdim=True)], dim=-1))
        return value_pred, prob_logits

class ITrEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, ff_mult=2.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=drop, batch_first=True)
        self.drop1 = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, int(d_model*ff_mult)), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(int(d_model*ff_mult), d_model), nn.Dropout(drop))
    def forward(self, x):
        h = self.norm1(x); h,_ = self.attn(h,h,h,need_weights=False); x = x + self.drop1(h)
        h = self.ff(self.norm2(x)); return x + h

class ITransformerTiny(nn.Module):
    def __init__(self, cfg, in_len, out_len, cal_dim, n_stores, n_categories, n_types, d_model=256, nhead=8, nlayers=3, ff_mult=2.0, drop=0.1):
        super().__init__()
        self.prj = nn.Linear(1, d_model)
        self.enc = nn.ModuleList([ITrEncoderLayer(d_model, nhead, ff_mult, drop) for _ in range(nlayers)])
        self.readout = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_len))
        self.store_emb = nn.Embedding(n_stores,64); self.cat_emb = nn.Embedding(n_categories,32); self.type_emb = nn.Embedding(n_types,16)
        self.cal_proj = nn.Linear(cal_dim,128)
        self.meta_head = nn.Sequential(nn.Linear(64+32+16+128, d_model), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_model, out_len))
        self.prob_head = nn.Sequential(nn.Linear(64+32+16+128+1, d_model), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_model, out_len))
    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx):
        z = self.prj(x.unsqueeze(-1))
        for layer in self.enc: z = layer(z)
        z_last = z[:, -1, :]; z_mean = z.mean(dim=1); seq_out = self.readout(0.5*z_last + 0.5*z_mean)
        store = self.store_emb(store_idx); cat = self.cat_emb(cat_idx); typ = self.type_emb(type_idx)
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1); cal = self.cal_proj(cal_all)
        meta = torch.cat([store,cat,typ,cal], dim=-1)
        value_pred  = seq_out + self.meta_head(meta)
        prob_logits = self.prob_head(torch.cat([meta, x.mean(dim=1, keepdim=True)], dim=-1))
        return value_pred, prob_logits

# =====================
# Loss / EMA
# =====================
class UltraEnhancedHurdleLoss(nn.Module):
    def __init__(self, eps: float = 0.01, zero_weight: float = 0.01, lambda_bce: float = 0.15):
        super().__init__()
        self.eps = eps; self.zero_weight = zero_weight; self.lambda_bce = lambda_bce
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, v_pred_log, p_logits, y_true_log, pos_mask, sample_w):
        z = (pos_mask > 0).float()
        bce = self.bce(p_logits, z)
        yp_val = torch.expm1(v_pred_log).clamp_min(0.0)
        yt_val = torch.expm1(y_true_log).clamp_min(0.0)
        ultra_eps = torch.where(yt_val < 0.5, self.eps * 0.2,
                                torch.where(yt_val < 2.0, self.eps * 0.5, self.eps))
        denom = (torch.abs(yp_val) + torch.abs(yt_val)).clamp_min(ultra_eps)
        smape_pos = 2.0 * torch.abs(yp_val - yt_val) / denom
        smape_pos = smape_pos * (yt_val > 0).float()
        p = torch.sigmoid(p_logits)
        y_hat = p * yp_val
        denom2 = (torch.abs(y_hat) + torch.abs(yt_val)).clamp_min(ultra_eps)
        smape_all = 2.0 * torch.abs(y_hat - yt_val) / denom2
        ultra_zero_weight = torch.where(
            yt_val < 0.01, torch.full_like(yt_val, self.zero_weight * 0.1),
            torch.where(yt_val < 0.1, torch.full_like(yt_val, self.zero_weight * 0.3),
                        torch.where(yt_val < 1.0, torch.full_like(yt_val, self.zero_weight * 0.6), torch.ones_like(yt_val)))
        )
        smape_all = smape_all * ultra_zero_weight
        bce_s = bce.mean(dim=1); pos_s = smape_pos.mean(dim=1); all_s = smape_all.mean(dim=1)
        sample_loss = self.lambda_bce * bce_s + 0.4 * pos_s + 0.6 * all_s
        sw = sample_w.view(-1); wsum = sw.sum().clamp_min(1e-8)
        loss = (sample_loss * sw).sum() / wsum
        return loss, sample_loss.detach(), sw.detach()

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
    def apply_to(self, model: nn.Module):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(self.shadow[name])

# =====================
# Trainer (공용) + OOF 예측 수집 지원
# =====================
class GenericTrainer:
    def __init__(self, cfg: EnhancedNHiTSConfig, dataset: EnhancedNHiTSDataset, model: nn.Module,
                 epochs: int, batch_size: int, base_lr: float, max_lr: float, weight_decay: float):
        self.cfg = cfg; self.dataset = dataset; self.model = model.to(cfg.device)
        self.device = torch.device(cfg.device)
        self.criterion = UltraEnhancedHurdleLoss(cfg.eps_smape, cfg.zero_weight, cfg.hurdle_lambda)
        self.ema = EMA(self.model, decay=cfg.ema_decay)
        self.epochs = epochs; self.batch_size = batch_size
        self.base_lr = base_lr; self.max_lr = max_lr; self.weight_decay = weight_decay
        self.history_rows = []

    def _dump_history_csv(self, extra_cols: dict = None):
        if not self.history_rows: return
        import pandas as pd, os
        rows = self.history_rows if extra_cols is None else [{**r, **extra_cols} for r in self.history_rows]
        df = pd.DataFrame(rows); path = self.cfg.train_log_csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = not os.path.exists(path)
        df.to_csv(path, index=False, mode="a", header=header)
        self.history_rows.clear()

    def make_loaders_from_mask(self, mask_val: np.ndarray):
        idx_all = np.arange(len(self.dataset))
        val_idx = idx_all[mask_val]; train_idx = idx_all[~mask_val]
        train_subset = torch.utils.data.Subset(self.dataset, train_idx)
        val_subset = torch.utils.data.Subset(self.dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
                                  persistent_workers=self.cfg.persistent_workers, drop_last=False)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
                                persistent_workers=self.cfg.persistent_workers, drop_last=False)
        return train_loader, val_loader

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, use_ema: bool = True) -> float:
        if loader is None: return float('inf')
        self.model.eval()
        backup = None
        if use_ema:
            backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.ema.apply_to(self.model)

        total_loss = 0.0; total_weight = 0.0
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if (self.cfg.use_amp and torch.cuda.is_available() and self.device.type=="cuda")\
                  else torch.cuda.amp.autocast(enabled=False)
        with amp_ctx:
            for batch in loader:
                x = batch["x"].to(self.device); y = batch["y"].to(self.device)
                past_cal = batch["past_cal"].to(self.device); fut_cal = batch["fut_cal"].to(self.device)
                store_idx = batch["store_idx"].to(self.device); cat_idx = batch["cat_idx"].to(self.device)
                type_idx = batch["type_idx"].to(self.device); sample_w = batch["sample_w"].to(self.device)
                pos_mask = batch["pos_mask"].to(self.device)
                v_pred, p_logits = self.model(x, past_cal, fut_cal, store_idx, cat_idx, type_idx)
                _, sample_loss, sw = self.criterion(v_pred, p_logits, y, pos_mask, sample_w)
                total_loss += (sample_loss * sw).sum().item(); total_weight += sw.sum().item()

        if use_ema and backup is not None:
            self.model.load_state_dict(backup); del backup
            gc.collect(); torch.cuda.empty_cache()
        if total_weight <= 0: return float('inf')
        return total_loss / total_weight

    @torch.no_grad()
    def infer_loader_to_arrays(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """OOF 수집용: y_hat(허들 적용, 원 스케일), y_true(원 스케일)"""
        self.model.eval()
        backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.ema.apply_to(self.model)
        yh_list, yt_list = [], []
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if (self.cfg.use_amp and torch.cuda.is_available() and self.device.type=="cuda")\
                  else torch.cuda.amp.autocast(enabled=False)
        with amp_ctx:
            for batch in loader:
                x = batch["x"].to(self.device); y = batch["y"].to(self.device)
                past_cal = batch["past_cal"].to(self.device); fut_cal = batch["fut_cal"].to(self.device)
                store_idx = batch["store_idx"].to(self.device); cat_idx = batch["cat_idx"].to(self.device); type_idx = batch["type_idx"].to(self.device)
                v_pred, p_logits = self.model(x, past_cal, fut_cal, store_idx, cat_idx, type_idx)
                y_val  = torch.expm1(v_pred).clamp_min(0.0)
                y_prob = torch.sigmoid(p_logits)
                y_hat  = (y_prob * y_val).clamp_min(0.0).detach().cpu().numpy()
                y_true = torch.expm1(y).clamp_min(0.0).detach().cpu().numpy()
                yh_list.append(y_hat); yt_list.append(y_true)
        self.model.load_state_dict(backup)
        return np.concatenate(yh_list, axis=0), np.concatenate(yt_list, axis=0)

    def train_with_loaders(self, train_loader: DataLoader, val_loader: DataLoader,
                           model_name: str = "Model", fold_id: int = None):
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr,
                                  weight_decay=self.weight_decay, betas=(0.9, 0.999))
        steps_per_epoch = max(1, len(train_loader))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=self.max_lr, epochs=self.epochs, steps_per_epoch=steps_per_epoch,
            pct_start=0.05, div_factor=max(1e-8, self.max_lr / max(1e-8, self.base_lr))
        )
        patience = max(self.cfg.earlystop_patience_min, int(self.epochs * self.cfg.earlystop_patience_ratio))
        best_val = float("inf"); best_state = None; no_improve = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if (self.cfg.use_amp and torch.cuda.is_available() and self.device.type=="cuda")\
                      else torch.cuda.amp.autocast(enabled=False)
            batch_losses = []
            for batch in train_loader:
                x = batch["x"].to(self.device); y = batch["y"].to(self.device)
                past_cal = batch["past_cal"].to(self.device); fut_cal = batch["fut_cal"].to(self.device)
                store_idx = batch["store_idx"].to(self.device); cat_idx = batch["cat_idx"].to(self.device)
                type_idx = batch["type_idx"].to(self.device); sample_w = batch["sample_w"].to(self.device)
                pos_mask = batch["pos_mask"].to(self.device)
                with amp_ctx:
                    v_pred, p_logits = self.model(x, past_cal, fut_cal, store_idx, cat_idx, type_idx)
                    loss, _, _ = self.criterion(v_pred, p_logits, y, pos_mask, sample_w)
                optim.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                optim.step(); sched.step(); self.ema.update(self.model)
                batch_losses.append(loss.detach().item())
            train_loss = float(np.mean(batch_losses)) if batch_losses else float('nan')
            val_loss = self.evaluate(val_loader, use_ema=True)
            print(f"[{model_name}] [Epoch {epoch:03d}] train_loss: {train_loss:.5f}  val_loss: {val_loss:.5f}")
            cur_lr = float(optim.param_groups[0].get("lr", self.base_lr))
            self.history_rows.append({
                "model": model_name, "fold": fold_id if fold_id is not None else "", "epoch": epoch,
                "train_loss": train_loss, "val_loss": float(val_loss), "lr": cur_lr,
            })
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[{model_name}] Early stopping at epoch {epoch}")
                    break

        if best_state is not None: self.model.load_state_dict(best_state)
        self.ema.apply_to(self.model)
        self._dump_history_csv()
        return self.model, float(best_val)

# =====================
# Predict Utils (공용)
# =====================
@torch.no_grad()
def predict_one_file_generic(cfg: EnhancedNHiTSConfig, model: nn.Module, test_df: pd.DataFrame,
                             store2idx: Dict, cat2idx: Dict, type2idx: Dict) -> pd.DataFrame:
    device = torch.device(cfg.device)
    if test_df is None or len(test_df) == 0: return pd.DataFrame()
    tdf = test_df.copy()
    need = [cfg.date_col, cfg.item_col, cfg.target_col]
    if any(c not in tdf.columns for c in need): return pd.DataFrame()
    tdf[cfg.date_col] = pd.to_datetime(tdf[cfg.date_col], errors="coerce")
    tdf = tdf.dropna(subset=[cfg.date_col]); 
    if len(tdf) == 0: return pd.DataFrame()
    tdf[cfg.target_col] = pd.to_numeric(tdf[cfg.target_col], errors='coerce').fillna(0.0).clip(lower=0)

    try:
        pivot = tdf.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index().fillna(0.0)
    except Exception:
        return pd.DataFrame()
    if pivot.shape[1] == 0: return pd.DataFrame()
    items = list(pivot.columns); dates = list(pivot.index); values = pivot.values.astype(np.float32)

    # in_len 맞추기용 replicate pad
    Lx = cfg.in_len
    if len(dates) < Lx:
        pad_rows = Lx - len(dates)
        first_row = values[0:1, :] if len(dates) > 0 else np.zeros((1, values.shape[1]), dtype=np.float32)
        pad = np.repeat(first_row, pad_rows, axis=0)
        values = np.vstack([pad, values])
        start_date = dates[0] if len(dates) > 0 else pd.Timestamp("2000-01-01")
        backfill = [start_date - pd.Timedelta(days=i) for i in range(pad_rows, 0, -1)]
        dates = backfill + dates

    last_date = dates[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, cfg.out_len + 1)]
    holidays_set = set(pd.to_datetime(cfg.custom_holidays_list)) if cfg.custom_holidays_list else set()

    # 캘린더 + 글로벌 롤링 (테스트에서도 동일 차원 보장; 미래는 ffill)
    past_cal_df = build_enhanced_features(dates[-Lx:], holidays_set)
    global_series = pd.Series(np.nanmean(values, axis=1), index=dates)
    roll_df = add_rolling_feats_safe(global_series, windows=(3,7,14))
    past_cal_df = past_cal_df.join(roll_df).fillna(method="ffill").fillna(0)
    fut_cal_df  = build_enhanced_features(future_dates, holidays_set)
    # 미래 롤링은 마지막 관측값으로 고정
    last_roll = roll_df.iloc[-1:].reindex(future_dates, method="ffill")
    fut_cal_df = fut_cal_df.join(last_roll).fillna(method="ffill").fillna(0)

    past_cal = past_cal_df.drop(columns=["date"], errors="ignore").values.astype(np.float32)
    fut_cal  = fut_cal_df.drop(columns=["date"], errors="ignore").values.astype(np.float32)

    x_np = values[-Lx:, :].T  # [B,L]
    if cfg.log1p: x_np = np.log1p(x_np)
    B = len(items)
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
    past_cal_b = torch.from_numpy(np.repeat(past_cal[None, :, :], B, axis=0)).to(device=device, dtype=torch.float32)
    fut_cal_b  = torch.from_numpy(np.repeat(fut_cal [None, :, :], B, axis=0)).to(device=device, dtype=torch.float32)

    stores = [parse_store_name(it) for it in items]; menus  = [parse_menu_name(it) for it in items]
    safe_store_default = next(iter(store2idx.values()), 0)
    safe_cat_default   = next(iter(cat2idx.values()), 0)
    safe_type_default  = next(iter(type2idx.values()), 0)
    store_idx = torch.as_tensor([store2idx.get(s, safe_store_default) for s in stores], device=device, dtype=torch.long)
    cat_idx   = torch.as_tensor([cat2idx.get(get_menu_category(m), safe_cat_default) for m in menus], device=device, dtype=torch.long)
    type_idx  = torch.as_tensor([type2idx.get(get_store_type(s), safe_type_default) for s in stores], device=device, dtype=torch.long)

    use_cuda_amp = bool(cfg.use_amp and torch.cuda.is_available() and device.type == "cuda")
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_cuda_amp else torch.cuda.amp.autocast(enabled=False)
    model.eval()
    with amp_ctx:
        v_pred_log, p_logits = model(x, past_cal_b, fut_cal_b, store_idx, cat_idx, type_idx)
        y_val  = torch.expm1(v_pred_log).clamp_min(0.0)
        y_prob = torch.sigmoid(p_logits)
        y_hat  = (y_prob * y_val).clamp_min(0.0).detach().cpu().numpy()  # [B, H]

    out = pd.DataFrame(y_hat.T, index=[f"D+{i}" for i in range(1, cfg.out_len + 1)], columns=items)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out

# =====================
# Ensemble Helper + (선택) Pruning/좌표상승
# =====================
def invloss_weights(vals: List[float], eps: float = 1e-8) -> List[float]:
    arr = np.array(vals, dtype=np.float64); arr[~np.isfinite(arr)] = np.nan
    if np.all(np.isnan(arr)): return (np.ones_like(arr) / len(arr)).tolist()
    safe = np.nan_to_num(arr, nan=np.nanmax(arr) * 1.5)
    inv = 1.0 / np.clip(safe, eps, None)
    if not np.isfinite(inv).any() or inv.sum() <= 0: return (np.ones_like(inv) / len(inv)).tolist()
    w = inv / inv.sum(); return w.tolist()

def prune_by_weight(weights: Dict[str, float], thr: float = 0.05, keep_min: int = 4) -> List[str]:
    sorted_models = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    kept = [m for m,_ in sorted_models[:keep_min]]
    for m,w in sorted_models[keep_min:]:
        if w >= thr: kept.append(m)
    return kept

def coordinate_ascent_weights(model_names: List[str], preds: Dict[str, np.ndarray],
                              y_true: np.ndarray, init_w: Optional[Dict[str, float]] = None,
                              steps: int = 80, smape_eps: float = 1e-6, seed: int = 42) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    names = list(model_names)
    if init_w is None: w = np.array([1/len(names)]*len(names), dtype=np.float64)
    else:
        w = np.array([init_w.get(n, 0.0) for n in names], dtype=np.float64)
        s = w.sum(); w = (w/s) if s>0 else np.array([1/len(names)]*len(names), dtype=np.float64)
    def smape(a, b, eps=smape_eps):
        denom = (np.abs(a)+np.abs(b)).clip(min=eps)
        return (200.0*np.abs(a-b)/denom).mean()
    P = np.stack([preds[n] for n in names], axis=0)  # [M,N,H]
    M = P.shape[0]
    for _ in range(steps):
        order = rng.permutation(M)
        for i in order:
            base = (w[:,None,None]*P).sum(axis=0)
            best_alpha = 0.0; best_loss = smape(base, y_true)
            for alpha in [-0.15,-0.1,-0.05,-0.02,0.02,0.05,0.1,0.15]:
                w_try = w.copy(); w_try[i] = max(0.0, w_try[i]+alpha)
                s = w_try.sum(); 
                if s <= 0: continue
                w_try /= s
                mix = (w_try[:,None,None]*P).sum(axis=0)
                loss = smape(mix, y_true)
                if loss < best_loss: best_loss, best_alpha = loss, alpha
            if best_alpha != 0.0:
                w[i] = max(0.0, w[i]+best_alpha); w /= w.sum()
    return {n: float(w_i) for n, w_i in zip(names, w)}

# =====================
# (NEW) Horizon-wise Softmax Stacker (OOF 기반)
# =====================
class HorizonWiseSoftmaxStacker(nn.Module):
    def __init__(self, n_models: int, out_len: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(out_len, n_models))  # [H, M]
    def forward(self):
        return torch.softmax(self.logits, dim=-1)

def fit_stacker_per_horizon(y_true: np.ndarray, preds: Dict[str, np.ndarray], lr=0.1, steps=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = list(preds.keys()); M = len(names); H = y_true.shape[1]
    X = np.stack([preds[n] for n in names], axis=-1)   # [N, H, M]
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(y_true, dtype=torch.float32, device=device)
    model = HorizonWiseSoftmaxStacker(M, H).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        W = model()                                    # [H, M]
        Y_hat = (X_t * W.unsqueeze(0)).sum(dim=-1)     # [N, H]
        loss = torch.mean(torch.abs(Y_hat - Y_t))      # L1
        opt.zero_grad(); loss.backward(); opt.step()
    W = model().detach().cpu().numpy()                 # [H, M]
    return names, W

# =====================
# Model factory / Optuna glue
# =====================
def build_model_by_name(name: str, cfg: EnhancedNHiTSConfig, ds: EnhancedNHiTSDataset,
                        trial: Optional['optuna.trial.Trial']=None) -> nn.Module:
    if name == "N-HiTS":
        hidden = trial.suggest_categorical("nh_hidden", [256,384,512]) if trial else cfg.hidden
        n_layers = trial.suggest_categorical("nh_layers", [1,2,3]) if trial else cfg.n_layers
        dropout = trial.suggest_float("nh_dropout", 0.0, 0.3) if trial else cfg.dropout
        nh_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__, "hidden": hidden, "n_layers": n_layers, "dropout": dropout})
        return EnhancedNHiTSModel(cfg.in_len, cfg.out_len, ds.cal_feats.shape[1], ds.n_stores, ds.n_categories, ds.n_types, nh_cfg)
    if name == "PatchTST":
        d_model = trial.suggest_categorical("pt_d_model", [192,256,320]) if trial else cfg.ptst_d_model
        nhead   = trial.suggest_categorical("pt_nhead", [8,12]) if trial else cfg.ptst_nhead
        nlayer  = trial.suggest_categorical("pt_nlayers", [2,3,4]) if trial else cfg.ptst_num_layers
        p_len   = trial.suggest_categorical("pt_patch", [2,3,4]) if trial else max(2, cfg.ptst_patch_len // 2)
        stride  = trial.suggest_categorical("pt_stride", [1,2]) if trial else max(1, cfg.ptst_stride // 2)
        drop    = trial.suggest_float("pt_dropout", 0.05, 0.2) if trial else cfg.ptst_dropout
        ff_mult = trial.suggest_float("ptst_ff_mult", 1.5, 3.5) if trial else cfg.ptst_ff_mult
        pt_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__,
            "ptst_d_model": d_model, "ptst_nhead": nhead, "ptst_num_layers": nlayer,
            "ptst_patch_len": p_len, "ptst_stride": stride, "ptst_dropout": drop, "ptst_ff_mult": ff_mult
        })
        return ImprovedPatchTSTTiny(pt_cfg, cfg.in_len, cfg.out_len, ds.cal_feats.shape[1], ds.n_stores, ds.n_categories, ds.n_types)
    if name == "TimesNet":
        channels = trial.suggest_categorical("tn_channels", [96,128,160]) if trial else cfg.tnet_channels
        blocks   = trial.suggest_categorical("tn_blocks", [2,3,4]) if trial else cfg.tnet_blocks
        drop     = trial.suggest_float("tn_dropout", 0.0, 0.3) if trial else cfg.tnet_dropout
        tn_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__, "tnet_channels": channels, "tnet_blocks": blocks, "tnet_dropout": drop})
        return ImprovedTimesNetTiny(tn_cfg, cfg.in_len, cfg.out_len, ds.cal_feats.shape[1], ds.n_stores, ds.n_categories, ds.n_types)
    if name == "GRU":
        hidden = trial.suggest_categorical("gru_hidden", [128,192,256]) if trial else cfg.gru_hidden
        layers = trial.suggest_categorical("gru_layers", [1,2,3]) if trial else cfg.gru_layers
        drop   = trial.suggest_float("gru_dropout", 0.0, 0.3) if trial else cfg.gru_dropout
        bidi   = trial.suggest_categorical("gru_bidi", [False, True]) if trial else cfg.gru_bidirectional
        gr_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__, "gru_hidden": hidden, "gru_layers": layers, "gru_dropout": drop, "gru_bidirectional": bidi})
        return GRUTiny(gr_cfg, cfg.in_len, cfg.out_len, ds.cal_feats.shape[1], ds.n_stores, ds.n_categories, ds.n_types)
    if name == "DLinear":
        head_h = trial.suggest_categorical("dl_head_hidden", [256,384,512]) if trial else cfg.dlin_head_hidden
        dl_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__, "dlin_head_hidden": head_h})
        return ImprovedDLinearTiny(dl_cfg, cfg.in_len, cfg.out_len, ds.cal_feats.shape[1],
                                ds.n_stores, ds.n_categories, ds.n_types,
                               idx_dow=ds.idx_dow, idx_month=ds.idx_month)
    if name == "NLinear":
        return NLinearTiny(cfg, cfg.in_len, cfg.out_len, ds.cal_feats.shape[1], ds.n_stores, ds.n_categories, ds.n_types)
    if name == "TCN":
        C = trial.suggest_categorical("tcn_C", [96,128,160]) if trial else 128
        depth = trial.suggest_categorical("tcn_depth", [3,4,5]) if trial else 4
        drop = trial.suggest_float("tcn_drop", 0.05, 0.25) if trial else 0.1
        return TCNTiny(cfg, cfg.in_len, cfg.out_len, ds.cal_feats.shape[1], ds.n_stores, ds.n_categories, ds.n_types, C=C, depth=depth, drop=drop)
    if name == "iTransformer":
        d_model = trial.suggest_categorical("it_d_model", [192,256,320]) if trial else 256
        nhead   = trial.suggest_categorical("it_nhead", [8,12]) if trial else 8
        nlayers = trial.suggest_categorical("it_layers", [2,3,4]) if trial else 3
        ffm     = trial.suggest_float("it_ff_mult", 1.5, 3.0) if trial else 2.0
        drop    = trial.suggest_float("it_drop", 0.05, 0.2) if trial else 0.1
        return ITransformerTiny(cfg, cfg.in_len, cfg.out_len, ds.cal_feats.shape[1], ds.n_stores, ds.n_categories, ds.n_types,
                                d_model=d_model, nhead=nhead, nlayers=nlayers, ff_mult=ffm, drop=drop)
    raise ValueError(f"Unknown model name: {name}")

def optuna_objective_factory(model_name: str, cfg: EnhancedNHiTSConfig, ds: EnhancedNHiTSDataset, mask_val: np.ndarray):
    def objective(trial: 'optuna.trial.Trial') -> float:
        base_lr = trial.suggest_float("base_lr", 3e-4, 3e-3, log=True)
        max_lr  = trial.suggest_float("max_lr",  8e-4, 6e-3, log=True)
        wd      = trial.suggest_float("weight_decay", 1e-6, 8e-4, log=True)
        grad_clip = trial.suggest_float("grad_clip", 0.3, 1.0)
        pat_ratio = trial.suggest_float("pat_ratio", 0.15, 0.35)
        tmp_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__, "grad_clip": grad_clip, "earlystop_patience_ratio": pat_ratio})
        model = build_model_by_name(model_name, tmp_cfg, ds, trial)
        trainer = GenericTrainer(tmp_cfg, ds, model, epochs=tmp_cfg.EPOCHS_TUNE, batch_size=tmp_cfg.BATCH_TUNE,
                                 base_lr=base_lr, max_lr=max_lr, weight_decay=wd)
        train_loader, val_loader = trainer.make_loaders_from_mask(mask_val)
        _, val_loss = trainer.train_with_loaders(train_loader, val_loader, model_name=f"{model_name}-TUNE")
        trial.set_user_attr("val_loss", float(val_loss))
        return float(val_loss)
    return objective

def run_optuna_for_model(model_name: str, cfg: EnhancedNHiTSConfig, ds: EnhancedNHiTSDataset, mask_val: np.ndarray):
    if not HAS_OPTUNA:
        print(f"[Optuna] optuna가 설치되어 있지 않습니다. '{model_name}' 튜닝을 건너뜁니다.")
        return None, None
    print(f"[Optuna] Start tuning for {model_name} (trials={cfg.N_TRIALS})")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True, seed=cfg.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner, study_name=f"study_{model_name}")
    study.optimize(optuna_objective_factory(model_name, cfg, ds, mask_val), n_trials=cfg.N_TRIALS, show_progress_bar=False)
    print(f"[Optuna] Best {model_name} val_loss: {study.best_value:.6f}")
    best_params = study.best_params
    return study, best_params

def build_final_model_with_best(model_name: str, cfg: EnhancedNHiTSConfig, ds: EnhancedNHiTSDataset, best_params: Optional[dict]) -> nn.Module:
    if best_params is None: return build_model_by_name(model_name, cfg, ds, None)
    tmp = dict(cfg.__dict__)
    if model_name == "N-HiTS":
        if "nh_hidden" in best_params: tmp["hidden"] = best_params["nh_hidden"]
        if "nh_layers" in best_params: tmp["n_layers"] = best_params["nh_layers"]
        if "nh_dropout" in best_params: tmp["dropout"] = best_params["nh_dropout"]
    elif model_name == "PatchTST":
        mapping = {"pt_d_model":"ptst_d_model","pt_nhead":"ptst_nhead","pt_nlayers":"ptst_num_layers",
                   "pt_patch":"ptst_patch_len","pt_stride":"ptst_stride","pt_dropout":"ptst_dropout","ptst_ff_mult":"ptst_ff_mult"}
        for k, v in mapping.items():
            if k in best_params: tmp[v] = best_params[k]
    elif model_name == "TimesNet":
        mapping = {"tn_channels":"tnet_channels","tn_blocks":"tnet_blocks","tn_dropout":"tnet_dropout"}
        for k, v in mapping.items():
            if k in best_params: tmp[v] = best_params[k]
    elif model_name == "GRU":
        mapping = {"gru_hidden":"gru_hidden","gru_layers":"gru_layers","gru_dropout":"gru_dropout","gru_bidi":"gru_bidirectional"}
        for k, v in mapping.items():
            if k in best_params: tmp[v] = best_params[k]
    elif model_name == "DLinear":
        if "dl_head_hidden" in best_params: tmp["dlin_head_hidden"] = best_params["dl_head_hidden"]
    new_cfg = EnhancedNHiTSConfig(**tmp)
    return build_model_by_name(model_name, new_cfg, ds, None)

# =====================
# Main
# =====================
if __name__ == "__main__":
    cfg = EnhancedNHiTSConfig()
    if cfg.store_weights is None: cfg.store_weights = DEFAULT_STORE_WEIGHTS
    if cfg.custom_holidays_list is None: cfg.custom_holidays_list = DEFAULT_CUSTOM_HOLIDAYS
    set_seed(cfg.seed)

    print("=== Enhanced 7-Model Ensemble + Stacking 시작 ===")

    if not os.path.exists(cfg.train_csv):
        raise FileNotFoundError(f"학습 파일을 찾을 수 없습니다: {cfg.train_csv}")
    train_df = pd.read_csv(cfg.train_csv)
    ds = EnhancedNHiTSDataset(cfg, train_df)

    # ===== (옵션) Optuna per-model 튜닝 =====
    best_params_all = {}
    if cfg.USE_OPTUNA and HAS_OPTUNA:
        tune_mask_val = make_val_mask_by_week(ds, cfg.cv_fold_end_dates[0])
        for name in ["N-HiTS", "PatchTST", "TimesNet", "GRU", "DLinear"]:
            study, best_params = run_optuna_for_model(name, cfg, ds, tune_mask_val)
            best_params_all[name] = best_params
        os.makedirs("./data", exist_ok=True)
        with open("./data/optuna_best_params.json", "w", encoding="utf-8") as f:
            json.dump(best_params_all, f, ensure_ascii=False, indent=2)
    elif cfg.USE_OPTUNA and not HAS_OPTUNA:
        print("⚠️ cfg.USE_OPTUNA=True지만 optuna가 설치되어 있지 않습니다. 튜닝은 건너뜁니다.")

    # ===== 학습할 모델 목록 =====
    model_names = ["TimesNet", "N-HiTS", "GRU", "DLinear", "NLinear", "TCN", "iTransformer","PatchTST"]
    trained_models: Dict[str, nn.Module] = {}
    avg_val_losses: Dict[str, float] = {}

    # ===== OOF 수집(스태킹용) =====
    #   각 모델: preds[name] -> [N_total, H]
    #   y_true_oof: [N_total, H] (첫 모델 기준으로만 저장)
    oof_preds: Dict[str, List[np.ndarray]] = {m: [] for m in model_names}
    y_true_oof_parts: List[np.ndarray] = []
    first_model_for_oof = model_names[0]

    def _infer_val_batches_to_arrays(
        model: nn.Module, loader: DataLoader, device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        """검증 로더를 돌려 (y_true, y_pred_hat)를 numpy로 반환
           - y_true: expm1(y) 또는 원 스케일
           - y_pred_hat: sigmoid(p) * expm1(v)
        """
        ys, yhs = [], []
        model.eval()
        amp_ctx = (
            torch.autocast(device_type='cuda', dtype=torch.float16)
            if (cfg.use_amp and torch.cuda.is_available() and device.type == "cuda")
            else torch.cuda.amp.autocast(enabled=False)
        )
        with torch.no_grad(), amp_ctx:
            for batch in loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                past_cal = batch["past_cal"].to(device)
                fut_cal = batch["fut_cal"].to(device)
                store_idx = batch["store_idx"].to(device)
                cat_idx = batch["cat_idx"].to(device)
                type_idx = batch["type_idx"].to(device)

                v_pred, p_logits = model(x, past_cal, fut_cal, store_idx, cat_idx, type_idx)
                y_val = torch.expm1(v_pred).clamp_min(0.0) if cfg.log1p else v_pred.clamp_min(0.0)
                p = torch.sigmoid(p_logits)
                y_hat = (p * y_val).clamp_min(0.0)

                y_true = torch.expm1(y).clamp_min(0.0) if cfg.log1p else y

                ys.append(y_true.detach().cpu().numpy())
                yhs.append(y_hat.detach().cpu().numpy())
        return np.concatenate(ys, axis=0), np.concatenate(yhs, axis=0)

    # ===== 모델별 학습 & OOF 수집 =====
    for name in model_names:
        print(f"📚 {name} 학습...")

        # 0) 초기 스냅샷 확보(항상 같은 초기 가중치에서 시작)
        base_model = build_final_model_with_best(
            name, cfg, ds, best_params_all.get(name) if cfg.USE_OPTUNA else None
        )
        init_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        del base_model
        gc.collect(); torch.cuda.empty_cache()

        fold_vals = []
        best_fold = None
        best_fold_state = None

        for fold_i, end_date in enumerate(cfg.cv_fold_end_dates, start=1):
            mask_val = make_val_mask_by_week(ds, end_date)

            model_f = build_final_model_with_best(
                name, cfg, ds, best_params_all.get(name) if cfg.USE_OPTUNA else None
            )
            model_f.load_state_dict(init_state, strict=True)
            trainer = GenericTrainer(
                cfg, ds, model_f,
                epochs=cfg.EPOCHS_FULL, batch_size=cfg.BATCH_FULL,
                base_lr=cfg.BASE_LR_FULL, max_lr=cfg.MAX_LR_FULL, weight_decay=cfg.WD_FULL
            )
            train_loader, val_loader = trainer.make_loaders_from_mask(mask_val)
            console_name = f"{name}-F{fold_i}"
            model_f, val_loss = trainer.train_with_loaders(
                train_loader, val_loader,
                model_name=console_name,
                fold_id=fold_i
            )
            fold_vals.append(float(val_loss))

            # ---- OOF 수집(검증 로더 전 구간 예측) ----
            y_true_np, y_hat_np = _infer_val_batches_to_arrays(model_f, val_loader, trainer.device)
            oof_preds[name].append(y_hat_np)  # [Nv, H]
            if name == first_model_for_oof:
                y_true_oof_parts.append(y_true_np)  # 첫 모델 기준으로만 저장(모든 모델의 val 순서 동일/shuffle=False)

            # 베스트 폴드 state 보관
            if (best_fold is None) or (val_loss < fold_vals[best_fold]):
                best_fold = fold_i - 1
                best_fold_state = {k: v.detach().cpu().clone() for k, v in model_f.state_dict().items()}

            # 폴드별 객체 정리
            del trainer, train_loader, val_loader, model_f
            gc.collect(); torch.cuda.empty_cache()

        # 2) 폴드 평균 성능
        avg_val = float(np.mean(fold_vals)) if len(fold_vals) > 0 else float('inf')
        avg_val_losses[name] = avg_val
        print(f"[VAL(avg over folds)] {name}={avg_val:.5f}")

        # 3) 추론용 fresh 모델 생성 후 베스트 폴드 state 로드
        final_model = build_final_model_with_best(
            name, cfg, ds, best_params_all.get(name) if cfg.USE_OPTUNA else None
        )
        if best_fold_state is not None:
            final_model.load_state_dict(best_fold_state, strict=True)
        trained_models[name] = final_model.to(cfg.device)

    # ===== (선택) 역손실 기반 초기 가중치 =====
    vals_in_order = [avg_val_losses.get(n, np.nan) for n in model_names]
    init_w = invloss_weights(vals_in_order)
    weights = dict(zip(model_names, init_w))
    print("[Ensemble Weights (init invloss)] " + "  ".join([f"{k}={weights[k]:.3f}" for k in model_names]))

    # ===== 스태킹(호라이즌별 softmax) 학습 =====
    #   - 폴드별 OOF를 concat → [N_total, H]
    #   - 누수 없음(각 fold의 val만 사용)
    stacked_names: Optional[List[str]] = None
    stacked_W: Optional[np.ndarray] = None  # [H, M]

    try:
        if len(y_true_oof_parts) > 0:
            y_true_oof = np.concatenate(y_true_oof_parts, axis=0)  # [N, H]
            preds_for_stack = {m: np.concatenate(oof_preds[m], axis=0) for m in model_names}
            # 안전: NaN/Inf 0으로 치환
            y_true_oof = np.nan_to_num(y_true_oof, nan=0.0, posinf=0.0, neginf=0.0)
            for k in preds_for_stack:
                preds_for_stack[k] = np.nan_to_num(preds_for_stack[k], nan=0.0, posinf=0.0, neginf=0.0)

            # 스태커 학습
            stacked_names, stacked_W = fit_stacker_per_horizon(y_true_oof, preds_for_stack, lr=0.1, steps=400)
            print("✅ 스태킹 가중치 학습 완료 (호라이즌별 softmax)")
        else:
            print("⚠️ OOF 타깃이 비어 있어 스태킹을 건너뜁니다.")
    except Exception as e:
        print(f"⚠️ 스태킹 학습 중 예외 발생: {e}. invloss 가중치로 진행합니다.")
        stacked_names, stacked_W = None, None

    # ===== 제출 생성 =====
    print("🔮 8-Model 앙상블 예측...")
    test_files = sorted(glob.glob(cfg.test_glob))
    if not os.path.exists(cfg.submission_template_csv):
        raise FileNotFoundError(f"제출 템플릿 파일을 찾을 수 없습니다: {cfg.submission_template_csv}")
    sub_template = pd.read_csv(cfg.submission_template_csv)
    all_preds = []

    if len(test_files) == 0:
        print("⚠️ 테스트 파일이 없습니다. 제출 생성을 건너뜁니다.")
    else:
        for test_idx, test_file in enumerate(test_files):
            print(f"  📊 {test_file} 처리 중...")
            try:
                tdf = pd.read_csv(test_file)
            except Exception as e:
                print(f"  ⚠️ 파일 로드 실패: {e}")
                continue

            # 각 모델 예측 수집
            df_preds: Dict[str, pd.DataFrame] = {}
            for name in model_names:
                df = predict_one_file_generic(cfg, trained_models[name], tdf, ds.store2idx, ds.cat2idx, ds.type2idx)
                if df is None or df.empty:
                    print(f"  ⚠️ {name} 예측이 비어 있습니다. 이 모델은 스킵합니다.")
                    continue
                df_preds[name] = df

            non_empty = [df for df in df_preds.values() if df is not None and not df.empty]
            if len(non_empty) == 0:
                print(f"  ⚠️ {test_file} 유효 예측이 없어 스킵합니다.")
                continue

            # 아이템 정렬 통일
            items = list(non_empty[0].columns)
            for k in df_preds:
                df_preds[k] = df_preds[k].reindex(columns=items).fillna(0.0)

            H = cfg.out_len
            # ===== 결합: 스태킹 가중치가 있으면 그걸 사용, 없으면 invloss 가중치 사용 =====
            if (stacked_names is not None) and (stacked_W is not None):
                # 스태킹에 포함된 모델만 사용(혹시 일부 모델이 학습 누락되었을 경우 대비)
                mix = np.zeros_like(df_preds[stacked_names[0]].values, dtype=np.float64)  # [H, I]
                for mi, mname in enumerate(stacked_names):
                    if mname not in df_preds:
                        continue
                    w_h = stacked_W[:, mi].reshape(H, 1)  # [H,1]
                    mix += w_h * df_preds[mname].values
            else:
                # invloss weights
                mix = np.zeros_like(non_empty[0].values, dtype=np.float64)
                for name in model_names:
                    if name in df_preds:
                        mix += weights.get(name, 0.0) * df_preds[name].values

            mix = np.clip(mix, 0, None)

            submit_block = pd.DataFrame(
                mix,
                index=[f"D+{i}" for i in range(1, cfg.out_len + 1)],
                columns=items
            )
            submit_block.index = [f"TEST_{test_idx:02d}+{k}일" for k in range(1, cfg.out_len + 1)]
            all_preds.append(submit_block)

    if len(all_preds) == 0:
        print("⚠️ 유효한 예측 결과가 없어 제출 파일을 생성하지 않았습니다.")
    else:
        final_submit = pd.concat(all_preds, axis=0)
        final_submit.reset_index(inplace=True)
        final_submit.rename(columns={"index": "영업일자"}, inplace=True)

        final_submit = final_submit.reindex(columns=sub_template.columns, fill_value=0)

        # 간단한 상한 제약(극단치 방어)
        num_cols = [c for c in final_submit.columns if c != "영업일자"]
        for col in num_cols:
            Q95 = final_submit[col].quantile(0.95)
            final_submit[col] = np.where(final_submit[col] > Q95 * 2.0, Q95 * 1.5, final_submit[col])

        final_submit[num_cols] = np.rint(np.clip(final_submit[num_cols].values, a_min=0, a_max=None)).astype(int)
        os.makedirs(os.path.dirname(cfg.out_submission_csv), exist_ok=True)
        final_submit.to_csv(cfg.out_submission_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 앙상블 완료! → {cfg.out_submission_csv}")

    print("🏆 8-Model + Stacking 학습/추론 끝")