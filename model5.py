# -*- coding: utf-8 -*-
"""
Enhanced N-HiTS + Improved PatchTST + Improved TimesNet + GRU + Improved DLinear (5-Model Ensemble)
+ Optional Optuna Tuning + Multi-fold CV + Safe AMP + Memory-safe EMA

주요 반영:
- PatchEmbedding 패딩 오류 수정(2D replicate 금지 → 직접 복제)
- predict_one_file_generic DataFrame 전치(shape mismatch) 수정
- 패딩 일관성(훈련/추론: replicate)
- ImprovedTimesNet: FFT 주파수 필터 + 2D Conv
- ImprovedPatchTST: 더 촘촘한 패치, multi-head projection, time/meta 통합 강화
- ImprovedDLinear: 분해+멀티스케일 제한([7,14]), dow/month one-hot 활용, 조합 드롭아웃
- invloss_weights: NaN/Inf 안전화
- EMA evaluate 백업 해제 + empty_cache
- Optuna: 학습 레벨 일부 포함(옵션)
- 하이퍼/얼리스톱 Config화
- 다중 fold 검증 평균으로 앙상블 가중치 산출
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
    train_csv: str = "/content/drive/MyDrive/LG_train.csv"
    test_glob: str = "/content/drive/MyDrive/test/*.csv"
    submission_template_csv: str = "sample_submission.csv"
    out_submission_csv: str = "0814_enhanced_5model_submission.csv"

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
    ptst_ff_mult: float = 1.5  # Config화

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
    if any(word in menu_lower for word in ['막걸리', '소주', '맥주', '와인', '참이슬', '처음처럼', '카스', '하이네켄', '버드와이저', '스텔라']):
        return 'alcohol'
    elif any(word in menu_lower for word in ['찌개', '탕', '국밥', '라면', '해장국', '갈비탕']):
        return 'hot_food'
    elif any(word in menu_lower for word in ['삼겹', '갈비', '목살', 'bbq', '구이', '불고기']):
        return 'bbq'
    elif any(word in menu_lower for word in ['아이스크림', '식혜', '콜라', '스프라이트', '에이드']):
        return 'dessert_drink'
    elif any(word in menu_lower for word in ['아메리카노', '라떼', '커피']):
        return 'coffee'
    elif any(word in menu_lower for word in ['냉면', '파스타', '스파게티', '면', '우동']):
        return 'noodles'
    elif any(word in menu_lower for word in ['비빔밥', '볶음밥', '공깃밥', '정식']):
        return 'rice'
    else:
        return 'others'

def get_store_type(store_name: str) -> str:
    if store_name == "느티나무 셀프BBQ":
        return 'outdoor'
    elif store_name in ["라그로타", "미라시아"]:
        return 'fine_dining'
    elif store_name == "담하":
        return 'traditional'
    elif store_name == "연회장":
        return 'event'
    elif store_name in ["카페테리아", "포레스트릿", "화담숲카페"]:
        return 'casual'
    else:
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
    df[["month_sin", "month_cos"]] = df["month"].apply(lambda m: pd.Series(sine_cosine_encoding(m, 12)))
    df[["dow_sin", "dow_cos"]] = df["dow"].apply(lambda d: pd.Series(sine_cosine_encoding(d, 7)))
    df[["doy_sin", "doy_cos"]] = df["date"].apply(lambda d: pd.Series(sine_cosine_encoding(d.dayofyear, 365)))
    df[["week_sin", "week_cos"]] = df["week"].apply(lambda w: pd.Series(sine_cosine_encoding(w, 52)))
    df[["quarter_sin", "quarter_cos"]] = df["quarter"].apply(lambda q: pd.Series(sine_cosine_encoding(q, 4)))
    return df.drop(columns=["tomorrow", "yesterday"])

def parse_store_name(item_full: str) -> str:
    return item_full.split("_")[0]

def parse_menu_name(item_full: str) -> str:
    return item_full.split("_", 1)[1] if "_" in item_full else item_full

def make_val_mask_by_week(dataset: 'EnhancedNHiTSDataset', end_date_str: str) -> np.ndarray:
    end_date = pd.to_datetime(end_date_str)
    start_date = end_date - pd.Timedelta(days=6)
    ted = dataset.target_end_dates
    return (ted >= start_date) & (ted <= end_date)

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
        caldf = build_enhanced_features(self.dates, holidays_set)
        self.cal_feats = caldf.drop(columns=["date"]).values.astype(np.float32)
        self.cal_feat_names = [c for c in caldf.columns if c != "date"]

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

        # Type-safe datetime64 ns
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
        fut_cal = self.cal_feats[t0 + Lx:t0 + Lx + Ly, :]

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
# N-HiTS
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
            nn.Linear(cfg.hidden, cfg.hidden // 2),
            nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden // 2, out_len)
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
                block_output = block_output @ basis            # <-- basis.T 제거
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
            # replicate: 마지막 값 복제하여 왼쪽에 붙임
            last_val = x[:, -1:].repeat(1, pad)
            x = torch.cat([last_val, x], dim=1)

        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # [B, Np, P]
        Np = patches.size(1)
        x_e = self.proj(patches)
        pe = _build_sinusoidal_pos(Np, self.d_model, x_e.device, x_e.dtype)  # [1,Np,D]
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
# Improved TimesNet (주파수 + 2D Conv)
# =====================
import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyBlock(nn.Module):
    def __init__(self, seq_len: int, top_k: int = 5):
        super().__init__()
        self.seq_len = int(seq_len)
        self.top_k = int(min(max(top_k, 1), max(1, self.seq_len // 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        B, C, L = x.shape
        in_dtype = x.dtype

        # --- FFT만 fp32로 강제 (AMP/half 회피) ---
        x32 = x.to(torch.float32)

        # rFFT: [B, C, L//2 + 1] (complex64)
        x_freq = torch.fft.rfft(x32, dim=-1)

        # 진폭 평균 기반 top-k 주파수 선택
        amp = torch.abs(x_freq).mean(dim=1)                # [B, L//2 + 1]
        _, top_idx = torch.topk(amp, k=self.top_k, dim=-1)

        # 선택 주파수만 보존
        x_freq_filt = torch.zeros_like(x_freq)
        for i in range(B):
            x_freq_filt[i, :, top_idx[i]] = x_freq[i, :, top_idx[i]]

        # iFFT 복원 (float32)
        x_filt32 = torch.fft.irfft(x_freq_filt, n=L, dim=-1)  # [B, C, L]

        # 원래 dtype으로 되돌림 (예: fp16)
        x_filt = x_filt32.to(in_dtype)
        return x_filt

class ImprovedTimesBlock(nn.Module):
    """
    개선된 TimesNet 블록:
      - FrequencyBlock으로 주파수 필터링
      - 시간 × 주파수 2D 컨볼루션 경로
      - 다중 커널 1D 컨볼루션 경로
      - 채널 정합을 위한 1x1 Conv (등록형, 학습 가능)
      - 안정적 정규화/드롭아웃
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        seq_len: int,
        kernels=(3, 5, 7),
        top_k: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert in_ch > 0 and out_ch > 0 and seq_len > 0
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.seq_len = int(seq_len)

        # 주파수 필터링
        self.freq_block = FrequencyBlock(seq_len=self.seq_len, top_k=top_k)

        # period 설정(너무 작아지지 않도록 하한 2)
        self.period_base = max(2, self.seq_len // 4)

        # 2D conv 경로(시간×주파수 지도)
        self.conv2d = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=(3, 3), padding=1),
            nn.GELU(),
        )

        # 1D 다중 커널 경로 (브랜치 합 채널 수가 out_ch와 정확히 일치하도록 분배)
        ks = list(kernels)
        n_br = len(ks)
        per = max(1, self.out_ch // n_br)
        branch_chs = [per] * n_br
        for i in range(self.out_ch - sum(branch_chs)):
            branch_chs[i % n_br] += 1

        branches = []
        for i, k in enumerate(ks):
            ch = branch_chs[i]
            branches.append(
                nn.Sequential(
                    nn.Conv1d(self.in_ch, ch, kernel_size=k, padding=k // 2),
                    nn.GELU(),
                    nn.Conv1d(ch, ch, kernel_size=k, padding=k // 2),
                    nn.GELU(),
                )
            )
        self.conv1d_branches = nn.ModuleList(branches)
        self.br_sum_ch = sum(branch_chs)

        # 채널 정합(학습 가능 1x1 conv, forward에서 새로 만들지 않음)
        self.channel_proj = (
            nn.Conv1d(self.br_sum_ch, self.out_ch, kernel_size=1)
            if self.br_sum_ch != self.out_ch
            else nn.Identity()
        )

        # 정규화/드롭아웃 (출력 텐서의 shape [B, out_ch, L]에 맞춘 LayerNorm)
        self.norm = nn.LayerNorm([self.out_ch, self.seq_len])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L] (C=in_ch, L=seq_len)
        return: [B, out_ch, L]
        """
        B, C, L = x.shape
        assert C == self.in_ch, f"in_ch mismatch: got {C}, expected {self.in_ch}"

        # 1) 주파수 필터링
        xf = self.freq_block(x)  # [B, C, L]

        # 2) 2D 경로 (period 분할 후 Conv2d 적용)
        period = self.period_base
        if L % period != 0:
            pad_len = period - (L % period)
            # 마지막 값을 복제해 패딩(시간축)
            xf = F.pad(xf, (0, pad_len), mode="replicate")
            L_pad = L + pad_len
        else:
            pad_len = 0
            L_pad = L

        # [B, C, L_pad] -> [B, C, period, L_pad//period]
        x2d = xf.reshape(B, C, period, L_pad // period).contiguous()
        z2d = self.conv2d(x2d)  # [B, out_ch, period, L_pad//period]
        z2d_flat = z2d.reshape(B, self.out_ch, L_pad)
        if pad_len > 0:
            z2d_flat = z2d_flat[:, :, :L]  # 원래 길이로 복원

        # 3) 1D 다중 커널 경로
        z1d_list = [branch(x) for branch in self.conv1d_branches]   # 각 [B, ch_i, L]
        z1d = torch.cat(z1d_list, dim=1)                            # [B, br_sum_ch, L]
        z1d = self.channel_proj(z1d)                                 # [B, out_ch, L]

        # 4) 합치기(+ 잔차적 의미)
        out = z2d_flat + z1d                                         # [B, out_ch, L]

        # 5) 정규화 & 드롭아웃
        #   LayerNorm의 normalized_shape는 마지막 두 축([C,L]) 기준이므로 입력 [B,C,L]과 호환
        out = self.norm(out)
        out = self.dropout(out)
        return out
    
class ImprovedTimesNetTiny(nn.Module):
    def __init__(self, cfg: EnhancedNHiTSConfig, in_len: int, out_len: int,
                 cal_dim: int, n_stores: int, n_categories: int, n_types: int):
        super().__init__()
        # 1) 채널은 8의 배수 권장(속도 유리)
        C = max(32, min(cfg.tnet_channels, 96))
        C = (C // 8) * 8  # 8의 배수로 스냅

        self.stem = nn.Conv1d(1, C, kernel_size=3, padding=1)

        # 2) 초고속 블록(Depthwise + Pointwise), bias=False로 연산 약간 절감
        def fast_block(C: int, p: float):
            return nn.Sequential(
                nn.Conv1d(C, C, kernel_size=3, padding=1, groups=C, bias=True),   # depthwise
                nn.Conv1d(C, C, kernel_size=1, bias=False),                      # pointwise
                nn.GELU(),
                nn.Conv1d(C, C, kernel_size=3, padding=1, groups=C, bias=True),
                nn.Conv1d(C, C, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Dropout(p)
            )

        self.blocks = nn.ModuleList([
            fast_block(C, p=max(0.0, min(cfg.tnet_dropout, 0.1)))
            for _ in range(max(1, cfg.tnet_blocks))
        ])

        # 3) 헤드: AdaptiveAvgPool 대신 mean 사용(오버헤드 ↓)
        head_hidden = max(128, min(cfg.tnet_head_hidden, 192))
        self.head_fc1 = nn.Linear(C, head_hidden)
        self.head_fc2 = nn.Linear(head_hidden, out_len)
        self.head_drop = nn.Dropout(cfg.tnet_dropout)

        # meta
        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb   = nn.Embedding(n_categories, 32)
        self.type_emb  = nn.Embedding(n_types, 16)
        self.cal_proj  = nn.Linear(cal_dim, 128)
        meta_dim = 64 + 32 + 16 + 128
        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.tnet_dropout),
            nn.Linear(head_hidden, out_len),
        )
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 1, head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.tnet_dropout),
            nn.Linear(head_hidden, out_len),
        )
        self.meta_res_weight = nn.Parameter(torch.tensor(0.15))

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx):
        # [B,L] -> [B,C,L]
        z = self.stem(x.unsqueeze(1))
        for blk in self.blocks:
            z = z + blk(z)  # residual

        # Global Average Pool (manual mean) -> FC
        z_mean = z.mean(dim=-1)           # [B, C]
        seq_out = self.head_fc2(self.head_drop(F.gelu(self.head_fc1(z_mean))))

        # meta
        store_emb = self.store_emb(store_idx)
        cat_emb   = self.cat_emb(cat_idx)
        type_emb  = self.type_emb(type_idx)
        cal_all   = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)
        cal_emb   = self.cal_proj(cal_all)
        meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb], dim=-1)

        value_pred  = seq_out + self.meta_res_weight * self.meta_integration(meta_feat)
        prob_logits = self.prob_head(torch.cat([meta_feat, x.mean(dim=1, keepdim=True)], dim=-1))
        return value_pred, prob_logits
    
    
# =====================
# GRU (경량)
# =====================
class GRUTiny(nn.Module):
    def __init__(self, cfg: EnhancedNHiTSConfig, in_len: int, out_len: int,
                 cal_dim: int, n_stores: int, n_categories: int, n_types: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
            dropout=cfg.gru_dropout if cfg.gru_layers > 1 else 0.0,
            bidirectional=cfg.gru_bidirectional
        )
        d_mul = 2 if cfg.gru_bidirectional else 1
        self.head = nn.Sequential(nn.LayerNorm(cfg.gru_hidden * d_mul),
                                  nn.Linear(cfg.gru_hidden * d_mul, out_len))
        # meta
        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb = nn.Embedding(n_categories, 32)
        self.type_emb = nn.Embedding(n_types, 16)
        self.cal_proj = nn.Linear(cal_dim, 128)
        meta_dim = 64+32+16+128
        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, cfg.gru_head_hidden),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(cfg.gru_head_hidden, out_len)
        )
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 1, cfg.gru_head_hidden),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(cfg.gru_head_hidden, cfg.gru_head_hidden // 2),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(cfg.gru_head_hidden // 2, out_len)
        )

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx):
        z_in = x.unsqueeze(-1)     # [B,L,1]
        z_out, _ = self.gru(z_in)  # [B,L,H*d]
        z_last = z_out[:, -1, :]   # [B,H*d]
        seq_out = self.head(z_last)
        store_emb = self.store_emb(store_idx)
        cat_emb = self.cat_emb(cat_idx)
        type_emb = self.type_emb(type_idx)
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)
        cal_emb = self.cal_proj(cal_all)
        meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb], dim=-1)
        value_pred = seq_out + self.meta_integration(meta_feat)
        prob_logits = self.prob_head(torch.cat([meta_feat, x.mean(dim=1, keepdim=True)], dim=-1))
        return value_pred, prob_logits

# =====================
# Improved DLinear
# =====================
class ImprovedDLinearTiny(nn.Module):
    def __init__(self, cfg: EnhancedNHiTSConfig, in_len: int, out_len: int,
                 cal_dim: int, n_stores: int, n_categories: int, n_types: int):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.decomp_kernel = 25
        self.moving_avg = nn.AvgPool1d(kernel_size=self.decomp_kernel, stride=1, padding=self.decomp_kernel//2, count_include_pad=False)

        self.trend_linear = nn.Sequential(
            nn.Linear(in_len, cfg.dlin_head_hidden),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden, out_len)
        )
        self.seasonal_linear = nn.Sequential(
            nn.Linear(in_len, cfg.dlin_head_hidden),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden, out_len)
        )

        self.ws = [w for w in [7, 14] if w <= in_len]
        self.multi_scale_linears = nn.ModuleList([nn.Linear(w, out_len) for w in self.ws])

        self.dow_linear = nn.Linear(7, out_len)
        self.month_linear = nn.Linear(12, out_len)

        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb = nn.Embedding(n_categories, 32)
        self.type_emb = nn.Embedding(n_types, 16)
        self.cal_proj = nn.Linear(cal_dim, 128)

        meta_dim = 64 + 32 + 16 + 128
        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, cfg.dlin_head_hidden * 2),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden * 2, cfg.dlin_head_hidden),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden, out_len)
        )
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 3, cfg.dlin_head_hidden),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden, cfg.dlin_head_hidden // 2),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden // 2, out_len)
        )
        self.component_weights = nn.Parameter(torch.ones(4 + len(self.multi_scale_linears)))
        self.component_dropout = nn.Dropout(p=0.15)

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx):
        B, L = x.size()
        trend = self.moving_avg(x.unsqueeze(1)).squeeze(1)  # <-- 사전 F.pad 제거
        seasonal = x - trend

        trend_pred = self.trend_linear(trend)
        seasonal_pred = self.seasonal_linear(seasonal)

        multi_scale_preds = []
        for w, linear in zip(self.ws, self.multi_scale_linears):
            if w <= L:
                pred = linear(x[:, -w:])
                multi_scale_preds.append(pred)

        # === 달/요일 one-hot (미래 캘린더 기반) ===
        # build_enhanced_features 순서 기준: 0:dow, 1:month, ...
        dow_idx   = fut_cal[..., 0].long().clamp(0, 6)
        month_idx = (fut_cal[..., 1] - 1).long().clamp(0, 11)
        dow_oh   = F.one_hot(dow_idx, num_classes=7).float()      # [B, out_len, 7]
        month_oh = F.one_hot(month_idx, num_classes=12).float()   # [B, out_len, 12]
        dow_vec   = dow_oh.mean(dim=1)      # [B,7]
        month_vec = month_oh.mean(dim=1)    # [B,12]
        dow_pred   = self.dow_linear(dow_vec)       # [B,out_len]
        month_pred = self.month_linear(month_vec)   # [B,out_len]

        all_preds = [trend_pred, seasonal_pred, dow_pred, month_pred] + multi_scale_preds
        weights = F.softmax(self.component_weights[:len(all_preds)], dim=0)
        weights = self.component_dropout(weights)
        weights = weights / (weights.sum() + 1e-8)
        combined_pred = sum(w * p for w, p in zip(weights, all_preds))

        store_emb = self.store_emb(store_idx)
        cat_emb = self.cat_emb(cat_idx)
        type_emb = self.type_emb(type_idx)
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)
        cal_emb = self.cal_proj(cal_all)
        meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb], dim=-1)
        meta_adjustment = self.meta_integration(meta_feat)
        value_pred = combined_pred + meta_adjustment

        x_stats = torch.stack([x.mean(dim=1), x.std(dim=1), x.max(dim=1)[0]], dim=1)
        prob_feat = torch.cat([meta_feat, x_stats], dim=-1)
        prob_logits = self.prob_head(prob_feat)

        return value_pred, prob_logits
class SeasonalNaiveModel(nn.Module):
    """
    SeasonalNaive 모델: 주기적 패턴 기반 예측
    - 주별 계절성(7일): 작주 같은 요일 값 사용
    - 월별 패턴: 지난달 같은 시기 값 활용
    - 트렌드 보정: 최근 변화율 반영
    - 메타데이터 통합: 공휴일, 매장 특성 등 고려
    """
    def __init__(self, cfg: EnhancedNHiTSConfig, in_len: int, out_len: int,
                 cal_dim: int, n_stores: int, n_categories: int, n_types: int):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        
        # 메타 임베딩 (다른 모델과 동일)
        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb = nn.Embedding(n_categories, 32)
        self.type_emb = nn.Embedding(n_types, 16)
        self.cal_proj = nn.Linear(cal_dim, 128)
        
        # SeasonalNaive 특화 파라미터 (학습 가능한 보정 계수)
        meta_dim = 64 + 32 + 16 + 128
        
        # 계절성 가중치 (주별/월별)
        self.seasonal_weights = nn.Parameter(torch.tensor([0.7, 0.3]))  # [weekly, monthly]
        
        # 트렌드 보정 계수
        self.trend_correction = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_len)
        )
        
        # 공휴일/특수일 보정
        self.holiday_correction = nn.Sequential(
            nn.Linear(meta_dim + cal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_len)
        )
        
        # 확률 예측 헤드 (다른 모델과 통일)
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_len)
        )

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx):
        B, L = x.shape
        
        # 메타 특성 임베딩
        store_emb = self.store_emb(store_idx)
        cat_emb = self.cat_emb(cat_idx)
        type_emb = self.type_emb(type_idx)
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)
        cal_emb = self.cal_proj(cal_all)
        meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb], dim=-1)
        
        # SeasonalNaive 핵심 로직
        seasonal_preds = []
        
        # 1) 주별 계절성 (7일 주기)
        if L >= 7:
            weekly_vals = []
            for i in range(self.out_len):
                # 7일 전, 14일 전, 21일 전... 의 평균
                lookback_positions = []
                for week_back in [1, 2, 3, 4]:  # 최대 4주 전까지
                    pos = L - (7 * week_back) + (i % 7)
                    if 0 <= pos < L:
                        lookback_positions.append(pos)
                
                if lookback_positions:
                    # 최근일수록 높은 가중치
                    weights = torch.softmax(torch.arange(len(lookback_positions), 0, -1, 
                                          device=x.device, dtype=x.dtype), dim=0)
                    weekly_val = sum(w * x[:, pos] for w, pos in zip(weights, lookback_positions))
                else:
                    weekly_val = x[:, -1]  # fallback
                weekly_vals.append(weekly_val)
            weekly_pred = torch.stack(weekly_vals, dim=1)  # [B, out_len]
        else:
            weekly_pred = x[:, -1:].repeat(1, self.out_len)
        
        # 2) 월별 계절성 (28일 주기, 대략 한달)
        if L >= 28:
            monthly_vals = []
            for i in range(self.out_len):
                # 28일 전 같은 위치
                pos = L - 28 + i
                if 0 <= pos < L:
                    monthly_val = x[:, pos]
                else:
                    monthly_val = x[:, -1]
                monthly_vals.append(monthly_val)
            monthly_pred = torch.stack(monthly_vals, dim=1)
        else:
            monthly_pred = x[:, -1:].repeat(1, self.out_len)
        
        # 3) 가중 결합
        weights = torch.softmax(self.seasonal_weights, dim=0)
        base_pred = weights[0] * weekly_pred + weights[1] * monthly_pred
        
        # 4) 트렌드 보정 (최근 변화율 반영)
        if L >= 3:
            recent_trend = (x[:, -1] - x[:, -3]) / 2  # 최근 3일 평균 변화율
            trend_effect = self.trend_correction(meta_feat)
            trend_adjustment = recent_trend.unsqueeze(1) * trend_effect
        else:
            trend_adjustment = torch.zeros_like(base_pred)
        
        # 5) 공휴일/특수일 보정
        fut_cal_flat = fut_cal.mean(dim=1)  # 미래 캘린더 평균
        holiday_input = torch.cat([meta_feat, fut_cal_flat], dim=-1)
        holiday_adjustment = self.holiday_correction(holiday_input)
        
        # 최종 예측값
        value_pred = base_pred + 0.1 * trend_adjustment + 0.05 * holiday_adjustment
        value_pred = torch.clamp(value_pred, min=0.0)  # 음수 방지
        
        # 확률 예측 (0/양수 분류)
        x_stats = torch.stack([x.mean(dim=1), x.std(dim=1), x.max(dim=1)[0]], dim=1)
        prob_input = torch.cat([meta_feat, x_stats], dim=-1)
        prob_logits = self.prob_head(prob_input)
        
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
                    self.shadow[name].mul_((self.decay)).add_(p.detach(), alpha=1-self.decay)
                    
    def apply_to(self, model: nn.Module):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(self.shadow[name])

# =====================
# Trainer (공용)
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

    def make_loaders_from_mask(self, mask_val: np.ndarray):
        idx_all = np.arange(len(self.dataset))
        val_idx = idx_all[mask_val]; train_idx = idx_all[~mask_val]
        train_subset = torch.utils.data.Subset(self.dataset, train_idx)
        val_subset = torch.utils.data.Subset(self.dataset, val_idx)
        train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers, drop_last=False
        )
        val_loader = DataLoader(
            val_subset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers, drop_last=False
        )
        return train_loader, val_loader

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, use_ema: bool = True) -> float:
        if loader is None:
            return float('inf')
        self.model.eval()

        backup = None
        if use_ema:
            backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.ema.apply_to(self.model)

        total_loss = 0.0
        total_weight = 0.0

        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if (
            self.cfg.use_amp and torch.cuda.is_available() and self.device.type == "cuda"
        ) else torch.cuda.amp.autocast(enabled=False)

        with amp_ctx:
            for batch in loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                past_cal = batch["past_cal"].to(self.device)
                fut_cal = batch["fut_cal"].to(self.device)
                store_idx = batch["store_idx"].to(self.device)
                cat_idx = batch["cat_idx"].to(self.device)
                type_idx = batch["type_idx"].to(self.device)
                sample_w = batch["sample_w"].to(self.device)
                pos_mask = batch["pos_mask"].to(self.device)

                v_pred, p_logits = self.model(x, past_cal, fut_cal, store_idx, cat_idx, type_idx)
                _, sample_loss, sw = self.criterion(v_pred, p_logits, y, pos_mask, sample_w)

                total_loss += (sample_loss * sw).sum().item()
                total_weight += sw.sum().item()

        if use_ema and backup is not None:
            self.model.load_state_dict(backup)
            del backup
            gc.collect(); torch.cuda.empty_cache()

        if total_weight <= 0:
            return float('inf')
        return total_loss / total_weight

    def train_with_loaders(self, train_loader: DataLoader, val_loader: DataLoader, model_name="Model"):
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay, betas=(0.9,0.999))
        steps_per_epoch = max(1, len(train_loader))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=self.max_lr, epochs=self.epochs,
            steps_per_epoch=steps_per_epoch, pct_start=0.05,
            div_factor=max(1e-8, self.max_lr / max(1e-8, self.base_lr))
        )
        patience = max(self.cfg.earlystop_patience_min, int(self.epochs * self.cfg.earlystop_patience_ratio))
        best_val = float("inf"); best_state = None; no_improve=0
        for epoch in range(1, self.epochs+1):
            self.model.train()
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if (self.cfg.use_amp and torch.cuda.is_available() and self.device.type=="cuda") else torch.cuda.amp.autocast(enabled=False)
            running = []
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
                running.append(loss.detach().item())
            val_loss = self.evaluate(val_loader, use_ema=True)
            tr_mean = float(np.mean(running)) if running else float('nan')
            print(f"[{model_name}] [Epoch {epoch:03d}] train_loss: {tr_mean:.5f}  val_loss: {val_loss:.5f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[{model_name}] Early stopping at epoch {epoch}")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.ema.apply_to(self.model)
        return self.model, best_val

# =====================
# Predict Utils (공용)
# =====================
@torch.no_grad()
def predict_one_file_generic(
    cfg: EnhancedNHiTSConfig,
    model: nn.Module,
    test_df: pd.DataFrame,
    store2idx: Dict,
    cat2idx: Dict,
    type2idx: Dict
) -> pd.DataFrame:
    device = torch.device(cfg.device)
    if test_df is None or len(test_df) == 0:
        return pd.DataFrame()

    tdf = test_df.copy()
    if cfg.date_col not in tdf.columns or cfg.item_col not in tdf.columns or cfg.target_col not in tdf.columns:
        return pd.DataFrame()

    tdf[cfg.date_col] = pd.to_datetime(tdf[cfg.date_col], errors="coerce")
    tdf = tdf.dropna(subset=[cfg.date_col])
    if len(tdf) == 0:
        return pd.DataFrame()

    tdf[cfg.target_col] = pd.to_numeric(tdf[cfg.target_col], errors="coerce").fillna(0.0).clip(lower=0)

    try:
        pivot = (
            tdf.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col)
               .sort_index()
               .fillna(0.0)
        )
    except Exception:
        return pd.DataFrame()
    if pivot.shape[1] == 0:
        return pd.DataFrame()

    items = list(pivot.columns)
    dates = list(pivot.index)
    values = pivot.values.astype(np.float32)

    # replicate padding (훈련과 일관)
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

    past_cal = build_enhanced_features(dates[-Lx:], holidays_set).drop(columns=["date"]).values.astype(np.float32)
    fut_cal  = build_enhanced_features(future_dates, holidays_set).drop(columns=["date"]).values.astype(np.float32)

    x_np = values[-Lx:, :].T  # [B,L]
    if cfg.log1p:
        x_np = np.log1p(x_np)

    B = len(items)
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
    past_cal_b = torch.from_numpy(np.repeat(past_cal[None, :, :], B, axis=0)).to(device=device, dtype=torch.float32)
    fut_cal_b  = torch.from_numpy(np.repeat(fut_cal [None, :, :], B, axis=0)).to(device=device, dtype=torch.float32)

    stores = [parse_store_name(it) for it in items]
    menus  = [parse_menu_name(it) for it in items]

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
        v_pred_log, p_logits = model(x, past_cal_b, fut_cal_b, store_idx, cat_idx, type_idx)  # [B, out_len]
        y_val  = torch.expm1(v_pred_log).clamp_min(0.0)
        y_prob = torch.sigmoid(p_logits)
        y_hat  = (y_prob * y_val).clamp_min(0.0).detach().cpu().numpy()  # [B, out_len]

    # rows=아웃스텝, cols=아이템 (전치!)
    out = pd.DataFrame(
        y_hat.T,
        index=[f"D+{i}" for i in range(1, cfg.out_len + 1)],
        columns=items
    )
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out

# =====================
# Ensemble Helper
# =====================
def invloss_weights(vals: List[float], eps: float = 1e-8) -> List[float]:
    arr = np.array(vals, dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    if np.all(np.isnan(arr)):
        return (np.ones_like(arr) / len(arr)).tolist()
    safe = np.nan_to_num(arr, nan=np.nanmax(arr) * 1.5)
    inv = 1.0 / np.clip(safe, eps, None)
    if not np.isfinite(inv).any() or inv.sum() <= 0:
        return (np.ones_like(inv) / len(inv)).tolist()
    w = inv / inv.sum()
    return w.tolist()

# =====================
# Model factory
# =====================
def build_model_by_name(name: str, cfg: EnhancedNHiTSConfig, ds: EnhancedNHiTSDataset,
                        trial: Optional['optuna.trial.Trial']=None) -> nn.Module:
    if name == "N-HiTS":
        hidden = trial.suggest_categorical("nh_hidden", [256, 384, 512]) if trial else cfg.hidden
        n_layers = trial.suggest_categorical("nh_layers", [1, 2, 3]) if trial else cfg.n_layers
        dropout = trial.suggest_float("nh_dropout", 0.0, 0.3) if trial else cfg.dropout
        nh_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__, "hidden": hidden, "n_layers": n_layers, "dropout": dropout})
        return EnhancedNHiTSModel(cfg.in_len, cfg.out_len, ds.cal_feats.shape[1], ds.n_stores, ds.n_categories, ds.n_types, nh_cfg)

    if name == "PatchTST":
        d_model = trial.suggest_categorical("pt_d_model", [192, 256, 320]) if trial else cfg.ptst_d_model
        nhead   = trial.suggest_categorical("pt_nhead", [8, 12]) if trial else cfg.ptst_nhead
        nlayer  = trial.suggest_categorical("pt_nlayers", [2, 3, 4]) if trial else cfg.ptst_num_layers
        p_len   = trial.suggest_categorical("pt_patch", [2, 3, 4]) if trial else max(2, cfg.ptst_patch_len // 2)
        stride  = trial.suggest_categorical("pt_stride", [1, 2]) if trial else max(1, cfg.ptst_stride // 2)
        drop    = trial.suggest_float("pt_dropout", 0.05, 0.2) if trial else cfg.ptst_dropout
        ff_mult = trial.suggest_float("ptst_ff_mult", 1.5, 3.5) if trial else cfg.ptst_ff_mult
        pt_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__,
            "ptst_d_model": d_model, "ptst_nhead": nhead, "ptst_num_layers": nlayer,
            "ptst_patch_len": p_len, "ptst_stride": stride, "ptst_dropout": drop, "ptst_ff_mult": ff_mult
        })
        return ImprovedPatchTSTTiny(pt_cfg, cfg.in_len, cfg.out_len, ds.cal_feats.shape[1], ds.n_stores, ds.n_categories, ds.n_types)

    if name == "TimesNet":
        channels = trial.suggest_categorical("tn_channels", [96, 128, 160]) if trial else cfg.tnet_channels
        blocks   = trial.suggest_categorical("tn_blocks", [2, 3, 4]) if trial else cfg.tnet_blocks
        drop     = trial.suggest_float("tn_dropout", 0.0, 0.3) if trial else cfg.tnet_dropout
        tn_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__,
            "tnet_channels": channels, "tnet_blocks": blocks, "tnet_dropout": drop
        })
        return ImprovedTimesNetTiny(tn_cfg, cfg.in_len, cfg.out_len, ds.cal_feats.shape[1], ds.n_stores, ds.n_categories, ds.n_types)

    if name == "GRU":
        hidden = trial.suggest_categorical("gru_hidden", [128, 192, 256]) if trial else cfg.gru_hidden
        layers = trial.suggest_categorical("gru_layers", [1, 2, 3]) if trial else cfg.gru_layers
        drop   = trial.suggest_float("gru_dropout", 0.0, 0.3) if trial else cfg.gru_dropout
        bidi   = trial.suggest_categorical("gru_bidi", [False, True]) if trial else cfg.gru_bidirectional
        gr_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__,
            "gru_hidden": hidden, "gru_layers": layers, "gru_dropout": drop, "gru_bidirectional": bidi
        })
        return GRUTiny(gr_cfg, cfg.in_len, cfg.out_len, ds.cal_feats.shape[1], ds.n_stores, ds.n_categories, ds.n_types)

    if name == "DLinear":
        head_h = trial.suggest_categorical("dl_head_hidden", [256, 384, 512]) if trial else cfg.dlin_head_hidden
        dl_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__, "dlin_head_hidden": head_h})
        return ImprovedDLinearTiny(dl_cfg, cfg.in_len, cfg.out_len, ds.cal_feats.shape[1], ds.n_stores, ds.n_categories, ds.n_types)

    if name == "SeasonalNaive":
        return SeasonalNaiveModel(cfg, cfg.in_len, cfg.out_len, ds.cal_feats.shape[1], 
                                ds.n_stores, ds.n_categories, ds.n_types)

    raise ValueError(f"Unknown model name: {name}")

def optuna_objective_factory(model_name: str, cfg: EnhancedNHiTSConfig, ds: EnhancedNHiTSDataset, mask_val: np.ndarray):
    def objective(trial: 'optuna.trial.Trial') -> float:
        # 훈련 레벨도 일부 튜닝(옵션)
        base_lr = trial.suggest_float("base_lr", 3e-4, 3e-3, log=True)
        max_lr  = trial.suggest_float("max_lr",  8e-4, 6e-3, log=True)
        wd      = trial.suggest_float("weight_decay", 1e-6, 8e-4, log=True)
        grad_clip = trial.suggest_float("grad_clip", 0.3, 1.0)
        pat_ratio = trial.suggest_float("pat_ratio", 0.15, 0.35)

        tmp_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__,
            "grad_clip": grad_clip, "earlystop_patience_ratio": pat_ratio
        })
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
    if best_params is None:
        return build_model_by_name(model_name, cfg, ds, None)

    tmp = dict(cfg.__dict__)
    if model_name == "N-HiTS":
        if "nh_hidden" in best_params: tmp["hidden"] = best_params["nh_hidden"]
        if "nh_layers" in best_params: tmp["n_layers"] = best_params["nh_layers"]
        if "nh_dropout" in best_params: tmp["dropout"] = best_params["nh_dropout"]

    elif model_name == "PatchTST":
        mapping = {"pt_d_model":"ptst_d_model","pt_nhead":"ptst_nhead","pt_nlayers":"ptst_num_layers",
                   "pt_patch":"ptst_patch_len","pt_stride":"ptst_stride","pt_dropout":"ptst_dropout",
                   "ptst_ff_mult":"ptst_ff_mult"}
        for k, v in mapping.items():
            if k in best_params:
                tmp[v] = best_params[k]

    elif model_name == "TimesNet":
        mapping = {"tn_channels":"tnet_channels","tn_blocks":"tnet_blocks","tn_dropout":"tnet_dropout"}
        for k, v in mapping.items():
            if k in best_params:
                tmp[v] = best_params[k]

    elif model_name == "GRU":
        mapping = {"gru_hidden":"gru_hidden","gru_layers":"gru_layers","gru_dropout":"gru_dropout","gru_bidi":"gru_bidirectional"}
        for k, v in mapping.items():
            if k in best_params:
                tmp[v] = best_params[k]

    elif model_name == "DLinear":
        if "dl_head_hidden" in best_params:
            tmp["dlin_head_hidden"] = best_params["dl_head_hidden"]

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

    print("=== Enhanced 5-Model Ensemble (+ Optuna Tuning) 시작 ===")

    if not os.path.exists(cfg.train_csv):
        raise FileNotFoundError(f"학습 파일을 찾을 수 없습니다: {cfg.train_csv}")
    train_df = pd.read_csv(cfg.train_csv)
    ds = EnhancedNHiTSDataset(cfg, train_df)

    # ===== (옵션) Optuna per-model 튜닝 =====
    best_params_all = {}
    if cfg.USE_OPTUNA:
        if not HAS_OPTUNA:
            print("⚠️ cfg.USE_OPTUNA=True지만 optuna가 설치되어 있지 않습니다. 튜닝은 건너뜁니다.")
        else:
            # 단일 fold로 튜닝 효율화(첫 fold)
            tune_mask_val = make_val_mask_by_week(ds, cfg.cv_fold_end_dates[0])
            for name in ["N-HiTS", "PatchTST", "TimesNet", "GRU", "DLinear"]:
                study, best_params = run_optuna_for_model(name, cfg, ds, tune_mask_val)
                best_params_all[name] = best_params
            os.makedirs("./data", exist_ok=True)
            with open("./data/optuna_best_params.json", "w", encoding="utf-8") as f:
                json.dump(best_params_all, f, ensure_ascii=False, indent=2)

    # ===== 모델 학습 (멀티 폴드 평균 val) =====
    model_names = ["SeasonalNaive", "TimesNet", "GRU", "DLinear","N-HiTS", "PatchTST"]
    trained_models: Dict[str, nn.Module] = {}
    avg_val_losses: Dict[str, float] = {}

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

        # 1) 폴드별로 항상 fresh 모델/트레이너
        for fold_i, end_date in enumerate(cfg.cv_fold_end_dates, start=1):
            mask_val = make_val_mask_by_week(ds, end_date)

            model_f = build_final_model_with_best(
                name, cfg, ds, best_params_all.get(name) if cfg.USE_OPTUNA else None
            )
            model_f.load_state_dict(init_state, strict=True)  # ← 동일 초기값
            trainer = GenericTrainer(
                cfg, ds, model_f,
                epochs=cfg.EPOCHS_FULL, batch_size=cfg.BATCH_FULL,
                base_lr=cfg.BASE_LR_FULL, max_lr=cfg.MAX_LR_FULL, weight_decay=cfg.WD_FULL
            )
            train_loader, val_loader = trainer.make_loaders_from_mask(mask_val)
            model_f, val_loss = trainer.train_with_loaders(
                train_loader, val_loader, model_name=f"{name}-F{fold_i}"
            )
            fold_vals.append(float(val_loss))

            # 베스트 폴드 state 보관
            if (best_fold is None) or (val_loss < fold_vals[best_fold]):
                best_fold = fold_i - 1  # index
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

    # ===== 가중치 산출(역손실, 안전화) =====
    vals_in_order = [avg_val_losses.get(n, np.nan) for n in model_names]
    w = invloss_weights(vals_in_order)
    weights = dict(zip(model_names, w))
    print("[Ensemble Weights] " + "  ".join([f"{k}={weights[k]:.3f}" for k in model_names]))

    # ===== 추론 & 제출 =====
    print("🔮 5-Model 앙상블 예측...")
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

            df_preds = {}
            for name in model_names:
                df = predict_one_file_generic(cfg, trained_models[name], tdf, ds.store2idx, ds.cat2idx, ds.type2idx)
                if df.empty:
                    print(f"  ⚠️ {name} 예측이 비어 있습니다. 이 모델은 스킵합니다.")
                df_preds[name] = df

            # 공통 아이템 집합 정렬
            non_empty = [df for df in df_preds.values() if df is not None and not df.empty]
            if len(non_empty) == 0:
                print(f"  ⚠️ {test_file} 유효 예측이 없어 스킵합니다.")
                continue
            items = list(non_empty[0].columns)
            for k in df_preds:
                if df_preds[k] is not None and not df_preds[k].empty:
                    df_preds[k] = df_preds[k].reindex(columns=items).fillna(0.0)

            # 가중 앙상블
            mix = np.zeros_like(non_empty[0].values, dtype=np.float64)
            for name in model_names:
                df = df_preds.get(name, None)
                if df is None or df.empty:
                    continue
                mix += weights[name] * df.values
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

        num_cols = [c for c in final_submit.columns if c != "영업일자"]
        for col in num_cols:
            Q95 = final_submit[col].quantile(0.95)
            final_submit[col] = np.where(final_submit[col] > Q95 * 2.0, Q95 * 1.5, final_submit[col])

        final_submit[num_cols] = np.rint(np.clip(final_submit[num_cols].values, a_min=0, a_max=None)).astype(int)
        os.makedirs(os.path.dirname(cfg.out_submission_csv), exist_ok=True)
        final_submit.to_csv(cfg.out_submission_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 앙상블 완료! → {cfg.out_submission_csv}")

    print("🏆 N-HiTS + PatchTST + TimesNet + GRU + DLinear 학습/추론 끝")
