# -*- coding: utf-8 -*-
"""
Enhanced N-HiTS (All-Fix + Competition Metric Edition)
- 반영 항목:
  1) EMA 베스트 저장/로드 버그 수정 (검증=EMA → 최종 저장=EMA 스냅샷)
  2) 학습 가중치 정렬: store 가중치 w_s를 |I_s|로 나눠 sample_weight = w_s / |I_s|
  3) 손실/추론 게이팅 제거: 회귀 예측에 발생확률 p를 곱하지 않음 (0일은 평가서 제외되므로 불리)
  4) 제출 전 클리핑/반올림 제거 (SMAPE 유리)
  5) Item Embedding 추가 (품목 고유성)
  6) 간단 backcast 잔차 스택 (N-BEATS/N-HiTS 철학 반영)
  7) 최근성 샘플링(지수감쇠)으로 시즌 변화 대응 (pandas Index → ndarray 변환 고정)
  8) OneCycleLR 파라미터 현실화(div_factor/final_div_factor)
  9) 입력 통계 (x_mean, x7_mean, x14_mean, x7_std) 메타에 주입
 10) “대회 산식 그대로” Weighted-SMAPE를 검증/조기정지 1차 기준으로 사용
환경: Linux, NVIDIA RTX A6000 48GB 가정
"""
import os
import math
import random
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import glob

warnings.filterwarnings('ignore')

# =====================
# Config
# =====================
@dataclass
class EnhancedNHiTSConfig:
    # 경로
    train_csv: str = "./data/train/train.csv"
    test_glob: str = "./data/test/*.csv"
    submission_template_csv: str = "./data/sample_submission.csv"
    out_submission_csv: str = "./data/double_enhanced_nhits_plus_submission.csv"

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

    # A6000에 맞춘 기본 하이퍼파라미터
    EPOCHS_FULL: int = 200
    BATCH_FULL: int = 4096
    BASE_LR_FULL: float = 1e-3
    MAX_LR_FULL: float = 2.5e-3
    WD_FULL: float = 3e-4

    # 튜닝
    USE_OPTUNA: bool = False
    N_TRIALS: int = 25
    EPOCHS_TUNE: int = 35
    BATCH_TUNE: int = 1024

    # CV 설정(롤링)
    cv_fold_end_dates: Tuple[str, str, str] = ("2024-06-14", "2024-06-07", "2024-05-31")

    # DataLoader
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True

    # N-HiTS 파라미터
    hidden: int = 512
    n_blocks: int = 3
    n_layers: int = 2
    n_pool_kernel_size: List[int] = None  # [2,2,1] 기본
    pooling_mode: str = "MaxPool1d"
    interpolation_mode: str = "linear"
    dropout: float = 0.1

    # Multi-scale
    stack_types: List[str] = None  # ["identity","identity","trend"]
    n_freq_downsample: List[int] = None  # [2,1,1]

    # Loss
    eps_smape: float = 0.01
    hurdle_lambda: float = 0.15  # BCE 비중 (보조)

    # AMP/EMA
    use_amp: bool = True
    use_compile: bool = False
    ema_decay: float = 0.9998

    # 가중치/공휴일
    store_weights: Dict[str, float] = None
    custom_holidays_list: List[str] = None

    # 최근성 샘플링(지수감쇠) 강도
    time_decay_alpha: float = 0.002

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
# Utils / Features
# =====================
def round_predictions_frame(df: pd.DataFrame, cols: List[str], method: str = "standard", seed: int = 42):
    """제출 직전 라운딩. method:
       - "standard": np.rint (0.5 반올림), 음수는 0으로 클립
       - "stochastic": 소수부 확률로 올림(큰 값에서 SMAPE 왜곡을 조금 줄일 수 있음)
    """
    arr = df[cols].values.astype(np.float64)
    if method == "stochastic":
        rng = np.random.default_rng(seed)
        frac = arr - np.floor(arr)
        up = rng.random(arr.shape) < frac
        arr = np.floor(arr) + up.astype(np.float64)
    else:
        arr = np.rint(arr)
    arr = np.clip(arr, a_min=0, a_max=None)  # 음수 방지
    df[cols] = arr.astype(np.int64)
    return df

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

# =====================
# Dataset
# =====================
class EnhancedNHiTSDataset(Dataset):
    def __init__(self, cfg: EnhancedNHiTSConfig, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df.copy()
        self.df[cfg.date_col] = pd.to_datetime(self.df[cfg.date_col])
        self.df[cfg.target_col] = self.df[cfg.target_col].clip(lower=0)

        pivot = self.df.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index()
        self.items = list(pivot.columns)
        self.dates = list(pivot.index)
        self.values = pivot.fillna(0.0).values.astype(np.float32)

        holidays_set = set(pd.to_datetime(cfg.custom_holidays_list))
        caldf = build_enhanced_features(self.dates, holidays_set)
        self.cal_feats = caldf.drop(columns=["date"]).values.astype(np.float32)
        self.cal_feat_names = [c for c in caldf.columns if c != "date"]

        # 메타 정보
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

        # 아이템 임베딩용
        self.item2idx = {it: i for i, it in enumerate(self.items)}
        self.item_idx = np.arange(len(self.items), dtype=np.int64)
        self.n_items = len(self.items)

        # 가게 가중치 정렬: w_s / |I_s|
        sw = cfg.store_weights
        store_counts = pd.Series(stores).value_counts().to_dict()
        self.sample_weights = np.array([
            sw.get(parse_store_name(it), 1.0) / max(1, store_counts.get(parse_store_name(it), 1))
            for it in self.items
        ], dtype=np.float32)

        # 윈도우 인덱스(훈련 컷오프 포함)
        self.indices: List[Tuple[int,int]] = []
        self.target_end_dates: List[pd.Timestamp] = []
        T = len(self.dates)
        Lx, Ly = cfg.in_len, cfg.out_len
        cutoff = pd.to_datetime(cfg.train_end_date)
        max_start = T - (Lx + Ly)

        for j in range(len(self.items)):
            for t0 in range(0, max_start + 1):
                end_date = self.dates[t0 + Lx + Ly - 1]
                if end_date <= cutoff:
                    self.indices.append((t0, j))
                    self.target_end_dates.append(end_date)
        self.target_end_dates = np.array(self.target_end_dates)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cfg = self.cfg
        t0, j = self.indices[idx]
        Lx, Ly = cfg.in_len, cfg.out_len

        x = self.values[t0:t0 + Lx, j]
        y = self.values[t0 + Lx:t0 + Lx + Ly, j]

        if cfg.log1p:
            x_in = np.log1p(x)
            y_out = np.log1p(y)
        else:
            x_in, y_out = x.copy(), y.copy()

        past_cal = self.cal_feats[t0:t0 + Lx, :]
        fut_cal = self.cal_feats[t0 + Lx:t0 + Lx + Ly, :]

        store_idx = self.item_store_idx[j]
        cat_idx = self.item_cat_idx[j]
        type_idx = self.item_type_idx[j]
        item_idx = self.item_idx[j]
        sample_w = self.sample_weights[j]

        pos_mask = (y > 0).astype(np.float32)

        return {
            "x": torch.from_numpy(x_in).float(),
            "y": torch.from_numpy(y_out).float(),
            "past_cal": torch.from_numpy(past_cal).float(),
            "fut_cal": torch.from_numpy(fut_cal).float(),
            "store_idx": torch.tensor(store_idx, dtype=torch.long),
            "cat_idx": torch.tensor(cat_idx, dtype=torch.long),
            "type_idx": torch.tensor(type_idx, dtype=torch.long),
            "item_idx": torch.tensor(item_idx, dtype=torch.long),
            "sample_w": torch.tensor(sample_w, dtype=torch.float32),
            "pos_mask": torch.from_numpy(pos_mask).float(),
        }

# =====================
# Model (Backcast Residual + Meta)
# =====================
class NHiTSBlock(nn.Module):
    """간단화된 N-HiTS 스타일 블록: pooling→MLP→backcast/forecast"""
    def __init__(self, input_size: int, output_size: int, hidden_size: int,
                 n_layers: int, dropout: float, pooling_mode: str = "MaxPool1d",
                 n_pool_kernel_size: int = 2):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        if pooling_mode == "MaxPool1d":
            self.pool = nn.MaxPool1d(kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True)
        else:
            self.pool = nn.AvgPool1d(kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True)

        pooled_size = math.ceil(input_size / n_pool_kernel_size)

        layers = [nn.Linear(pooled_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)

        self.backcast_layer = nn.Linear(hidden_size, input_size)
        self.forecast_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [B, in_len]
        x_pooled = self.pool(x.unsqueeze(1)).squeeze(1)  # [B, pooled]
        h = self.mlp(x_pooled)
        backcast = self.backcast_layer(h)   # [B, in_len]
        forecast = self.forecast_layer(h)   # [B, out_len]
        return backcast, forecast

class EnhancedNHiTSModel(nn.Module):
    """Enhanced N-HiTS with item/store/category/type embeddings + calendar + input stats"""
    def __init__(self, in_len: int, out_len: int, cal_dim: int, n_stores: int,
                 n_categories: int, n_types: int, n_items: int, cfg: EnhancedNHiTSConfig):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.cfg = cfg

        # 기본 리스트 채우기
        if cfg.n_pool_kernel_size is None:
            cfg.n_pool_kernel_size = [2, 2, 1]
        if cfg.stack_types is None:
            cfg.stack_types = ["identity", "identity", "trend"]
        if cfg.n_freq_downsample is None:
            cfg.n_freq_downsample = [2, 1, 1]
        while len(cfg.n_pool_kernel_size) < cfg.n_blocks:
            cfg.n_pool_kernel_size.append(cfg.n_pool_kernel_size[-1])
        while len(cfg.stack_types) < cfg.n_blocks:
            cfg.stack_types.append("identity")
        while len(cfg.n_freq_downsample) < cfg.n_blocks:
            cfg.n_freq_downsample.append(1)

        # 임베딩
        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb = nn.Embedding(n_categories, 32)
        self.type_emb = nn.Embedding(n_types, 16)
        self.item_emb = nn.Embedding(n_items, 64)

        # 캘린더
        self.cal_proj = nn.Linear(cal_dim, 128)

        # 블록들 (잔차 스택)
        self.blocks = nn.ModuleList()
        for i in range(cfg.n_blocks):
            self.blocks.append(
                NHiTSBlock(
                    input_size=in_len,
                    output_size=out_len,
                    hidden_size=cfg.hidden,
                    n_layers=cfg.n_layers,
                    dropout=cfg.dropout,
                    pooling_mode=cfg.pooling_mode,
                    n_pool_kernel_size=cfg.n_pool_kernel_size[i],
                )
            )

        # meta 통합 (임베딩+캘린더+입력 통계 4개)
        # meta_dim = 64 + 32 + 16 + 64 + 128 + 4 (x_mean, x7_mean, x14_mean, x7_std)
        meta_dim = 64 + 32 + 16 + 64 + 128 + 4
        self.meta_head = nn.Sequential(
            nn.Linear(meta_dim, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, out_len)
        )

        # 보조 발생확률(head) - 최종 예측에 곱하지 않음 (학습 보조)
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden // 2, out_len)
        )

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx, item_idx):
        # 임베딩
        s_emb = self.store_emb(store_idx)   # [B,64]
        c_emb = self.cat_emb(cat_idx)       # [B,32]
        t_emb = self.type_emb(type_idx)     # [B,16]
        i_emb = self.item_emb(item_idx)     # [B,64]

        # 캘린더 (과거+미래 평균)
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)  # [B, cal_dim]
        cal_emb = self.cal_proj(cal_all)                             # [B,128]

        # 입력 통계
        x_mean = x.mean(dim=1, keepdim=True)
        x7_mean = x[:, -7:].mean(dim=1, keepdim=True)
        x14_mean = x[:, -14:].mean(dim=1, keepdim=True)
        x7_std = x[:, -7:].std(dim=1, keepdim=True).nan_to_num(0.0)

        meta_feat = torch.cat([s_emb, c_emb, t_emb, i_emb, cal_emb,
                               x_mean, x7_mean, x14_mean, x7_std], dim=-1)  # [B, meta_dim]

        # 잔차 스택
        residual = x
        forecasts = []
        for block in self.blocks:
            back, fore = block(residual)
            residual = residual - back
            forecasts.append(fore)
        nhits_out = torch.stack(forecasts, dim=0).sum(dim=0)  # [B, out_len]

        meta_adj = self.meta_head(meta_feat)                  # [B, out_len]
        value_pred = nhits_out + meta_adj                     # [B, out_len]

        prob_logits = self.prob_head(meta_feat)               # [B, out_len] (학습 보조)

        return value_pred, prob_logits

# =====================
# Loss (SMAPE on positive days + BCE auxiliary)
# =====================
class PosSMAPEWithBCE(nn.Module):
    def __init__(self, eps: float = 0.01, lambda_bce: float = 0.15):
        super().__init__()
        self.eps = eps
        self.lambda_bce = lambda_bce
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, v_pred_log, p_logits, y_true_log, pos_mask, sample_w):
        # 회귀 예측
        yp = torch.expm1(v_pred_log).clamp_min(0.0)
        yt = torch.expm1(y_true_log).clamp_min(0.0)

        denom = (yp.abs() + yt.abs()).clamp_min(self.eps)
        smape = 2.0 * (yp - yt).abs() / denom
        smape = smape * pos_mask  # 양수일만

        # 보조 BCE: 양수일(발생) 분류
        z = (pos_mask > 0).float()
        bce = self.bce(p_logits, z).mean(dim=1)

        sample_loss = smape.mean(dim=1) + self.lambda_bce * bce  # [B]
        sw = sample_w.view(-1)
        wsum = sw.sum().clamp_min(1e-8)
        loss = (sample_loss * sw).sum() / wsum
        return loss, sample_loss.detach(), sw.detach()

# =====================
# EMA
# =====================
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
                if p.requires_grad and name in self.shadow:
                    p.copy_(self.shadow[name])

# =====================
# Trainer
# =====================
class EnhancedNHiTSTrainer:
    def __init__(self, cfg: EnhancedNHiTSConfig, dataset: EnhancedNHiTSDataset, epochs: int,
                 batch_size: int, base_lr: float, max_lr: float, weight_decay: float):
        self.cfg = cfg
        self.dataset = dataset
        self.device = torch.device(cfg.device)

        cal_dim = dataset.cal_feats.shape[1]
        model = EnhancedNHiTSModel(
            cfg.in_len, cfg.out_len, cal_dim, dataset.n_stores,
            dataset.n_categories, dataset.n_types, dataset.n_items, cfg
        ).to(self.device)

        if cfg.use_compile and torch.cuda.is_available():
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception as e:
                print("torch.compile 실패 → 비컴파일로 진행:", e)

        self.model = model
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.criterion = PosSMAPEWithBCE(cfg.eps_smape, cfg.hurdle_lambda)
        self.ema = EMA(self.model, decay=cfg.ema_decay)
        self.epochs = epochs
        self.batch_size = batch_size

        if cfg.use_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.set_float32_matmul_precision('high')
            torch.cuda.empty_cache()
            print(f"🚀 GPU: {torch.cuda.get_device_name()}")

    def make_loaders_from_mask(self, mask_val: np.ndarray):
        idx_all = np.arange(len(self.dataset))
        val_idx = idx_all[mask_val]
        train_idx = idx_all[~mask_val]

        train_subset = torch.utils.data.Subset(self.dataset, train_idx)
        val_subset = torch.utils.data.Subset(self.dataset, val_idx)

        # ---- 최근성 샘플링(지수감쇠) - pandas Index -> ndarray 변환 확실히 ----
        times_np = self.dataset.target_end_dates[train_idx]              # np.ndarray of timestamps
        times = pd.to_datetime(pd.Index(times_np))                       # DatetimeIndex
        tmax = times.max()
        dt_days = (tmax - times).days.to_numpy(dtype=np.float32)         # ndarray (days)
        decay = float(self.cfg.time_decay_alpha)
        weights_np = np.exp(-decay * dt_days).astype(np.float32)         # ndarray

        # torch tensor 로 변환
        weights = torch.as_tensor(weights_np, dtype=torch.float32)

        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(train_idx), replacement=True
        )

        pin_mem = self.cfg.pin_memory and (self.cfg.device == "cuda")
        train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, sampler=sampler,
            num_workers=self.cfg.num_workers, pin_memory=pin_mem,
            persistent_workers=self.cfg.persistent_workers, drop_last=False
        )
        val_loader = DataLoader(
            val_subset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=pin_mem,
            persistent_workers=self.cfg.persistent_workers, drop_last=False
        )
        return train_loader, val_loader

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, use_ema: bool = True) -> float:
        """Proxy loss (학습 로스 기반) — 참고 로그용."""
        self.model.eval()
        backup = None
        if use_ema:
            backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.ema.apply_to(self.model)

        per_sample_losses = []
        per_sample_weights = []

        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (
            self.cfg.use_amp and torch.cuda.is_available()
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
                item_idx = batch["item_idx"].to(self.device)
                sample_w = batch["sample_w"].to(self.device)
                pos_mask = batch["pos_mask"].to(self.device)

                v_pred, p_logits = self.model(x, past_cal, fut_cal, store_idx, cat_idx, type_idx, item_idx)
                loss, sample_loss, sw = self.criterion(v_pred, p_logits, y, pos_mask, sample_w)
                per_sample_losses.append(sample_loss)
                per_sample_weights.append(sw)

        if use_ema and backup is not None:
            self.model.load_state_dict(backup)

        if len(per_sample_losses) == 0:
            return 0.0

        sample_loss_all = torch.cat(per_sample_losses)
        sw_all = torch.cat(per_sample_weights)
        val = (sample_loss_all * sw_all).sum().item() / float(sw_all.sum().item() + 1e-8)
        return val

    @torch.no_grad()
    def evaluate_competition_metric(self, loader: DataLoader, use_ema: bool = True) -> float:
        """대회 산식 그대로의 Weighted-SMAPE.
        - 품목 i: 양수일만 t에 대해 SMAPE 평균
        - 가게 s: 품목 평균
        - 최종: sum_s w_s * mean_i(SMAPE_i)
        """
        self.model.eval()
        backup = None
        if use_ema:
            backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.ema.apply_to(self.model)

        # 품목 → 가게 매핑
        store_of_item = {}
        for j, it in enumerate(self.dataset.items):
            sname = parse_store_name(it)
            store_of_item[self.dataset.item2idx[it]] = sname

        # 품목별 smape 리스트 누적
        per_item_smape: Dict[int, list] = {}

        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (
            self.cfg.use_amp and torch.cuda.is_available()
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
                item_idx = batch["item_idx"].to(self.device)
                pos_mask = batch["pos_mask"].to(self.device)

                v_pred, _ = self.model(x, past_cal, fut_cal, store_idx, cat_idx, type_idx, item_idx)

                yt = torch.expm1(y).clamp_min(0.0)
                yp = torch.expm1(v_pred).clamp_min(0.0)

                denom = (yt.abs() + yp.abs()).clamp_min(self.cfg.eps_smape)
                smape = 2.0 * (yt - yp).abs() / denom  # [B, 7]
                smape = smape * pos_mask  # 0일 제외

                Ti = pos_mask.sum(dim=1)       # [B]
                smape_sum = smape.sum(dim=1)   # [B]
                valid = Ti > 0
                if valid.any():
                    smape_i = (smape_sum[valid] / Ti[valid]).detach().cpu().numpy()
                    item_ids = item_idx[valid].detach().cpu().numpy()
                    for k, v in zip(item_ids, smape_i):
                        per_item_smape.setdefault(int(k), []).append(float(v))

        # 가게별 평균 후 가중합
        store_to_itemvals: Dict[str, list] = {}
        for item_id, vals in per_item_smape.items():
            if len(vals) == 0:
                continue
            sname = store_of_item[item_id]
            store_to_itemvals.setdefault(sname, []).append(float(np.mean(vals)))

        # 가게별 평균 SMAPE
        store_means: Dict[str, float] = {}
        for sname, arr in store_to_itemvals.items():
            if len(arr) > 0:
                store_means[sname] = float(np.mean(arr))

        # 가중합
        total = 0.0
        for sname, mean_v in store_means.items():
            w = self.cfg.store_weights.get(sname, 1.0)
            total += w * mean_v

        # EMA 복원
        if use_ema and backup is not None:
            self.model.load_state_dict(backup)

        return float(total)

    def train_with_loaders(self, train_loader: DataLoader, val_loader: DataLoader):
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.base_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        # 현실적인 OneCycle 세팅
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=self.max_lr,
            epochs=self.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            div_factor=25.0,          # initial_lr = max_lr/div_factor
            final_div_factor=1e3      # last_lr = initial_lr/final_div_factor
        )

        best_val = float("inf")  # competition metric (낮을수록 좋음)
        no_improve = 0
        patience = 25

        # EMA 스냅샷을 따로 보관
        best_ema_shadow = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (
                self.cfg.use_amp and torch.cuda.is_available()
            ) else torch.cuda.amp.autocast(enabled=False)

            for batch in train_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                past_cal = batch["past_cal"].to(self.device)
                fut_cal = batch["fut_cal"].to(self.device)
                store_idx = batch["store_idx"].to(self.device)
                cat_idx = batch["cat_idx"].to(self.device)
                type_idx = batch["type_idx"].to(self.device)
                item_idx = batch["item_idx"].to(self.device)
                sample_w = batch["sample_w"].to(self.device)
                pos_mask = batch["pos_mask"].to(self.device)

                with amp_ctx:
                    v_pred, p_logits = self.model(x, past_cal, fut_cal, store_idx, cat_idx, type_idx, item_idx)
                    loss, _, _ = self.criterion(v_pred, p_logits, y, pos_mask, sample_w)

                self.optim.zero_grad(set_to_none=True)
                if self.cfg.use_amp and torch.cuda.is_available():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optim.step()
                self.sched.step()
                self.ema.update(self.model)

            # 1) proxy loss (참고용)
            proxy_loss = self.evaluate(val_loader, use_ema=True)
            # 2) competition metric (조기정지 기준)
            comp_metric = self.evaluate_competition_metric(val_loader, use_ema=True)

            print(f"[Epoch {epoch:03d}] proxy_loss: {proxy_loss:.6f}  comp(W-SMAPE): {comp_metric:.6f}")

            if comp_metric < best_val:
                best_val = comp_metric
                # EMA shadow 스냅샷 저장
                best_ema_shadow = {k: v.clone().cpu() for k, v in self.ema.shadow.items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # EMA 스냅샷을 실제 모델 파라미터로 복원
        if best_ema_shadow is not None:
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if p.requires_grad and name in best_ema_shadow:
                        p.copy_(best_ema_shadow[name].to(p.device))

        return self.model, best_val  # best_val은 competition metric

# =====================
# CV & (Optional) Optuna
# =====================
def make_val_mask_by_week(dataset: EnhancedNHiTSDataset, end_date_str: str) -> np.ndarray:
    end_date = pd.to_datetime(end_date_str)
    start_date = end_date - pd.Timedelta(days=6)
    ted = dataset.target_end_dates
    return (ted >= start_date) & (ted <= end_date)

def make_val_mask_by_end_date(dataset: EnhancedNHiTSDataset, end_date_str: str) -> np.ndarray:
    """fold 마지막 날짜 하루만 검증 — 대회 상황과 더 유사"""
    end_date = pd.to_datetime(end_date_str)
    ted = dataset.target_end_dates
    return (ted == end_date)

def evaluate_cfg_rolling(cfg: EnhancedNHiTSConfig, epochs: int, batch_size: int,
                        base_lr: float, max_lr: float, weight_decay: float) -> float:
    train_df = pd.read_csv(cfg.train_csv)
    ds = EnhancedNHiTSDataset(cfg, train_df)
    fold_vals = []
    for end_date_str in cfg.cv_fold_end_dates:
        trainer = EnhancedNHiTSTrainer(cfg, ds, epochs, batch_size, base_lr, max_lr, weight_decay)
        # 단일 날짜로 검증
        mask_val = make_val_mask_by_end_date(ds, end_date_str)
        train_loader, val_loader = trainer.make_loaders_from_mask(mask_val)
        _, best_val = trainer.train_with_loaders(train_loader, val_loader)
        fold_vals.append(best_val)
        print(f"[CV] fold end={end_date_str} comp(W-SMAPE)={best_val:.6f}")
    cv_mean = float(np.mean(fold_vals))
    print(f"[CV] mean comp(W-SMAPE)={cv_mean:.6f}")
    return cv_mean

def run_optuna(cfg: EnhancedNHiTSConfig):
    import optuna
    def objective(trial: "optuna.trial.Trial"):
        cfg.hidden = trial.suggest_categorical("hidden", [384, 512, 640])
        cfg.n_blocks = trial.suggest_categorical("n_blocks", [2, 3])
        cfg.n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
        cfg.dropout = trial.suggest_float("dropout", 0.05, 0.15)
        cfg.pooling_mode = trial.suggest_categorical("pooling_mode", ["MaxPool1d", "AvgPool1d"])
        cfg.eps_smape = trial.suggest_categorical("eps_smape", [0.005, 0.01, 0.02])
        cfg.hurdle_lambda = trial.suggest_categorical("hurdle_lambda", [0.1, 0.15, 0.2])

        epochs = cfg.EPOCHS_TUNE
        batch_size = cfg.BATCH_TUNE
        base_lr = trial.suggest_float("base_lr", 5e-4, 1.5e-3, log=True)
        max_lr = trial.suggest_float("max_lr", 1e-3, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        val = evaluate_cfg_rolling(cfg, epochs, batch_size, base_lr, max_lr, weight_decay)
        return val

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.N_TRIALS)
    print("[Optuna] Best value:", study.best_value)
    print("[Optuna] Best params:", study.best_trial.params)
    best = study.best_trial.params
    cfg.hidden = best.get("hidden", cfg.hidden)
    cfg.n_blocks = best.get("n_blocks", cfg.n_blocks)
    cfg.n_layers = best.get("n_layers", cfg.n_layers)
    cfg.dropout = best.get("dropout", cfg.dropout)
    cfg.pooling_mode = best.get("pooling_mode", cfg.pooling_mode)
    cfg.eps_smape = best.get("eps_smape", cfg.eps_smape)
    cfg.hurdle_lambda = best.get("hurdle_lambda", cfg.hurdle_lambda)

# =====================
# Predict
# =====================
@torch.no_grad()
def predict_one_file(cfg: EnhancedNHiTSConfig, model: EnhancedNHiTSModel, test_df: pd.DataFrame,
                     store2idx: Dict, cat2idx: Dict, type2idx: Dict, item2idx: Dict) -> pd.DataFrame:
    device = torch.device(cfg.device)
    tdf = test_df.copy()
    tdf[cfg.date_col] = pd.to_datetime(tdf[cfg.date_col])
    tdf[cfg.target_col] = tdf[cfg.target_col].clip(lower=0)

    pivot = tdf.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index().fillna(0.0)
    items = list(pivot.columns)
    dates = list(pivot.index)
    values = pivot.values.astype(np.float32)

    last_date = dates[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, cfg.out_len + 1)]

    holidays_set = set(pd.to_datetime(cfg.custom_holidays_list))
    past_cal = build_enhanced_features(dates[-cfg.in_len:], holidays_set).drop(columns=["date"]).values.astype(np.float32)
    fut_cal = build_enhanced_features(future_dates, holidays_set).drop(columns=["date"]).values.astype(np.float32)

    B = len(items)
    Lx = cfg.in_len
    x = values[-Lx:, :].T
    if cfg.log1p:
        x = np.log1p(x)

    x = torch.from_numpy(x).float().to(device)
    past_cal_b = torch.from_numpy(np.repeat(past_cal[None, :, :], B, axis=0)).float().to(device)
    fut_cal_b = torch.from_numpy(np.repeat(fut_cal[None, :, :], B, axis=0)).float().to(device)

    stores = [parse_store_name(it) for it in items]
    menus = [parse_menu_name(it) for it in items]

    store_idx = torch.tensor([store2idx.get(s, 0) for s in stores], dtype=torch.long, device=device)
    cat_idx = torch.tensor([cat2idx.get(get_menu_category(m), 0) for m in menus], dtype=torch.long, device=device)
    type_idx = torch.tensor([type2idx.get(get_store_type(s), 0) for s in stores], dtype=torch.long, device=device)
    item_idx = torch.tensor([item2idx.get(it, 0) for it in items], dtype=torch.long, device=device)

    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (
        cfg.use_amp and torch.cuda.is_available()
    ) else torch.cuda.amp.autocast(enabled=False)

    model.eval()
    with amp_ctx:
        v_pred, _ = model(x, past_cal_b, fut_cal_b, store_idx, cat_idx, type_idx, item_idx)
        y_val = torch.expm1(v_pred).clamp_min(0.0)  # 게이팅 제거
        y_hat = y_val.cpu().numpy()

    return pd.DataFrame(y_hat, index=items, columns=[f"D+{i}" for i in range(1, cfg.out_len+1)]).T

# =====================
# Main
# =====================
if __name__ == "__main__":
    cfg = EnhancedNHiTSConfig()
    if cfg.store_weights is None:
        cfg.store_weights = DEFAULT_STORE_WEIGHTS
    if cfg.custom_holidays_list is None:
        cfg.custom_holidays_list = DEFAULT_CUSTOM_HOLIDAYS

    set_seed(cfg.seed)

    # 빠른 실행 모드 (A6000이면 여유)
    FAST_MODE = True
    if FAST_MODE:
        cfg.USE_OPTUNA = False
        cfg.EPOCHS_FULL = 80
        cfg.BATCH_FULL = 2048

    print("📚 Load train ...")
    train_df = pd.read_csv(cfg.train_csv)
    ds = EnhancedNHiTSDataset(cfg, train_df)
    trainer = EnhancedNHiTSTrainer(cfg, ds, cfg.EPOCHS_FULL, cfg.BATCH_FULL,
                                   cfg.BASE_LR_FULL, cfg.MAX_LR_FULL, cfg.WD_FULL)

    # 최신 주의 '마지막 날짜 하루'를 검증으로 사용 (대회 상황과 유사)
    mask_val = make_val_mask_by_end_date(ds, cfg.cv_fold_end_dates[0])
    train_loader, val_loader = trainer.make_loaders_from_mask(mask_val)

    print("🚀 Train ...")
    model, final_comp_metric = trainer.train_with_loaders(train_loader, val_loader)
    print(f"🎯 Final competition metric (Weighted-SMAPE sum over stores): {final_comp_metric:.6f}")

    # 저장 (EMA 스냅샷이 이미 주입된 상태)
    os.makedirs(os.path.dirname(cfg.out_submission_csv), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg.__dict__,
        "store2idx": ds.store2idx,
        "cat2idx": ds.cat2idx,
        "type2idx": ds.type2idx,
        "item2idx": ds.item2idx,
        "final_comp_metric": final_comp_metric,
    }, "./data/enhanced_nhits_model.pth")
    print("[저장] ./data/enhanced_nhits_model.pth")

    # 추론 & 제출
    print("🔮 Predict & Make submission ...")
    test_files = sorted(glob.glob(cfg.test_glob))
    sub_template = pd.read_csv(cfg.submission_template_csv)
    all_preds = []

    for test_idx, test_file in enumerate(test_files):
        print(f"  📊 {test_file} ...")
        tdf = pd.read_csv(test_file)
        submit_block = predict_one_file(cfg, model, tdf, ds.store2idx, ds.cat2idx, ds.type2idx, ds.item2idx)
        submit_block.index = [f"TEST_{test_idx:02d}+{k}일" for k in range(1, cfg.out_len+1)]
        all_preds.append(submit_block)

    final_submit = pd.concat(all_preds, axis=0)
    final_submit.reset_index(inplace=True)
    final_submit.rename(columns={"index": "영업일자"}, inplace=True)

    # 제출 포맷 정렬
    final_submit = final_submit.reindex(columns=sub_template.columns, fill_value=0.0)

    num_cols = [c for c in final_submit.columns if c != cfg.date_col]
    final_submit[num_cols] = np.rint(
        np.clip(final_submit[num_cols].values, a_min=0, a_max=None)
    ).astype(np.int64)

    # 후처리: 클리핑/반올림 없음 (SMAPE 유리)
    final_submit.to_csv(cfg.out_submission_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Submission saved to {cfg.out_submission_csv}")
    print("🏁 Done.")
