# -*- coding: utf-8 -*-
"""
Single-Model Enhanced N-HiTS (Conditioned) — Full Pipeline
- Train/Val split by target-window end date:
  * Train(튜닝): target_end <= 2023-12-31
  * Val(튜닝):   2024-01-01 <= target_end <= (train.csv last date)
- Refit(재학습): Train + Val 전체 기간의 모든 윈도우
- Test: 각 파일에서 '마지막 28일' 입력만 사용, 28일 미만이면 스킵(패딩/반복 금지)

Rules Compliance
- 외부 데이터/사전학습 가중치 미사용 (도메인 지식: 요일/공휴일만 사용)
- 평가 입력 28일 외 Lookback 확장/패딩/반복 금지
- 샘플 독립 추론
- 미래 정보 미사용
"""
import os
import math
import copy
import random
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import glob
import json

try:
    import optuna
    HAS_OPTUNA = True
except Exception as e:
    print("[WARN] optuna import 실패:", e)
    HAS_OPTUNA = False

warnings.filterwarnings('ignore')

# =========================
# Config (기본값: 기존 코드 하이퍼 반영)
# =========================
@dataclass
class Config:
    # Paths
    train_csv: str = "./data/train/train.csv"
    test_glob: str = "./data/test/*.csv"
    submission_template_csv: str = "./data/sample_submission.csv"
    out_submission_csv: str = "./data/enhanced_nhits_single_submission.csv"
    checkpoint_dir: str = "./checkpoint/enh_nhits_single"

    # Columns
    date_col: str = "영업일자"
    item_col: str = "영업장명_메뉴명"
    target_col: str = "매출수량"

    # Window
    in_len: int = 28
    out_len: int = 7

    # Split by target-window end date
    train_cutoff: str = "2023-12-31"  # Train(튜닝): target_end <= this
    val_start: str   = "2024-01-01"   # Val(튜닝):   target_end >= this
    # (val_end는 train.csv의 마지막 날짜로 자동 계산)

    # Training common
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log1p: bool = True

    # Epochs/Batch (기본값: 당신이 준 코드 계열)
    EPOCHS_TUNE: int = 120     # 튜닝용 에폭(검증 포함 학습)
    BATCH_TUNE: int = 1024
    EPOCHS_FULL: int = 120     # 재학습(Train+Val) 에폭
    BATCH_FULL: int = 1024

    # Optimizer & Scheduler
    base_lr: float = 1e-3      # 초기 학습률
    max_lr: float = 2.5e-3     # (OneCycle용) — 지금은 cosine이라 사용 안함
    weight_decay: float = 3e-4
    scheduler_type: str = "cosine"  # "cosine" | "onecycle" | "plateau"
    warmup_ratio: float = 0.05       # cosine warmup 비율 (총 step 기준)

    # Mixed Precision / Compile / EMA
    use_amp: bool = True
    use_compile: bool = False
    ema_decay: float = 0.9998
    grad_clip: float = 0.5
    earlystop_patience: int = 25     # 튜닝 단계에서만 사용

    # Model (Enhanced N-HiTS)
    hidden: int = 512
    n_blocks: int = 3
    n_layers: int = 2
    n_pool_kernel_size: Optional[List[int]] = None  # e.g., [2,2,1]
    pooling_mode: str = "MaxPool1d"
    interpolation_mode: str = "linear"  # reserved
    dropout: float = 0.1
    stack_types: Optional[List[str]] = None         # e.g., ["identity","identity","trend"]
    n_freq_downsample: Optional[List[int]] = None   # e.g., [2,1,1]
    activation: str = "silu"                       # "silu" | "relu" | "gelu"

    # Loss (Hurdle + SMAPE)
    eps_smape: float = 0.01
    zero_weight: float = 0.01
    hurdle_lambda: float = 0.15

    # DataLoader
    num_workers: int = 6
    pin_memory: bool = True
    persistent_workers: bool = True

    # Holidays (도메인 지식 OK)
    custom_holidays_list: Optional[List[str]] = None

    # Store weights (β → sample_weight)
    store_weights: Optional[Dict[str, float]] = None

    # Optuna
    USE_OPTUNA: bool = True
    N_TRIALS: int = 60
    # Optuna OFF일 때 수동 주입할 "최적" 파라미터
    FIXED_BEST_PARAMS: Dict[str, float] = None

# 기본 업장 가중치(상대 β) — 손실 가중치로 사용
DEFAULT_STORE_WEIGHTS = {
    "미라시아": 7.71, "담하": 6.51, "연회장": 3.48, "라그로타": 3.44,
    "느티나무 셀프BBQ": 2.78, "화담숲주막": 1.43, "카페테리아": 1.31,
    "화담숲카페": 1.14, "포레스트릿": 1.00,
}

# 도메인 지식 허용: 공휴일 리스트 (과거/미리 아는 공휴일)
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

# (선택) 메뉴 클러스터 — 가벼운 conditioning 예시
cluster_mapping = {
    # Cluster 1
    "카페테리아_수제 등심 돈까스": 1,
    "포레스트릿_생수": 1,
    "포레스트릿_치즈 핫도그": 1,
    "포레스트릿_코카콜라": 1,

    # Cluster 2
    "담하_공깃밥": 2,
    "담하_담하 한우 불고기": 2,
    "미라시아_미라시아 브런치 (패키지)": 2,
    "미라시아_브런치(대인) 주말": 2,
    "미라시아_브런치(대인) 주중": 2,
    "화담숲주막_느린마을 막걸리": 2,
    "화담숲주막_병천순대": 2,
    "화담숲주막_참살이 막걸리": 2,
    "화담숲주막_찹쌀식혜": 2,
    "화담숲카페_메밀미숫가루": 2,
    "화담숲카페_아메리카노 ICE": 2,

    # Cluster 3
    "화담숲주막_해물파전": 3,

    # Cluster 4 (36개)
    "느티나무 셀프BBQ_스프라이트 (단체)": 4,
    "느티나무 셀프BBQ_잔디그늘집 대여료 (6인석)": 4,
    "느티나무 셀프BBQ_콜라 (단체)": 4,
    "담하_담하 한우 불고기 정식": 4,
    "담하_생목살 김치찌개": 4,
    "담하_한우 떡갈비 정식": 4,
    "담하_한우 우거지 국밥": 4,
    "담하_한우 차돌박이 된장찌개": 4,
    "미라시아_브런치(어린이)": 4,
    "연회장_Cass Beer": 4,
    "카페테리아_공깃밥(추가)": 4,
    "카페테리아_구슬아이스크림": 4,
    "카페테리아_돼지고기 김치찌개": 4,
    "카페테리아_새우 볶음밥": 4,
    "카페테리아_새우튀김 우동": 4,
    "카페테리아_아메리카노(HOT)": 4,
    "카페테리아_아메리카노(ICE)": 4,
    "카페테리아_약 고추장 돌솥비빔밥": 4,
    "카페테리아_어린이 돈까스": 4,
    "카페테리아_오픈푸드": 4,
    "카페테리아_짜장면": 4,
    "카페테리아_짬뽕": 4,
    "카페테리아_치즈돈까스": 4,
    "카페테리아_카페라떼(HOT)": 4,
    "포레스트릿_복숭아 아이스티": 4,
    "포레스트릿_스프라이트": 4,
    "포레스트릿_아메리카노(HOT)": 4,
    "포레스트릿_아메리카노(ICE)": 4,
    "포레스트릿_카페라떼(HOT)": 4,
    "포레스트릿_페스츄리 소시지": 4,
    "화담숲주막_단호박 식혜": 4,
    "화담숲주막_스프라이트": 4,
    "화담숲주막_콜라": 4,
    "화담숲카페_아메리카노 HOT": 4,
    "화담숲카페_카페라떼 ICE": 4,
    "화담숲카페_현미뻥스크림": 4,

    # Cluster 5
    "포레스트릿_꼬치어묵": 5,
    "포레스트릿_떡볶이": 5,

    # Cluster 6
    "카페테리아_단체식 13000(신)": 6,
    "카페테리아_단체식 18000(신)": 6,

    # Cluster 7
    "느티나무 셀프BBQ_BBQ55(단체)": 7,
    "느티나무 셀프BBQ_참이슬 (단체)": 7,
    "느티나무 셀프BBQ_카스 병(단체)": 7,
    "담하_(단체) 황태해장국 3/27까지": 7,
    "미라시아_(단체)브런치주중 36,000": 7,
    "연회장_Regular Coffee": 7,
}

# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sine_cosine_encoding(value: float, max_val: float):
    return math.sin(2 * math.pi * value / max_val), math.cos(2 * math.pi * value / max_val)

def build_enhanced_features(dates: List[pd.Timestamp], holidays_set: set) -> pd.DataFrame:
    """평일/주말/공휴일/계절/주기 신호(사인코사인) — 도메인 지식 범위 내 캘린더 피처"""
    df = pd.DataFrame({"date": dates})
    df["dow"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["week"] = df["date"].apply(lambda d: int(d.isocalendar().week))
    df["quarter"] = df["date"].dt.quarter

    df["is_friday"] = (df["dow"] == 4).astype(int)
    df["is_saturday"] = (df["dow"] == 5).astype(int)
    df["is_sunday"] = (df["dow"] == 6).astype(int)
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    df["is_holiday"] = df["date"].isin(holidays_set).astype(int)
    df["is_off"] = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).astype(int)
    df["tomorrow"] = df["date"] + pd.Timedelta(days=1)
    df["yesterday"] = df["date"] - pd.Timedelta(days=1)
    df["is_before_holiday"] = df["tomorrow"].isin(holidays_set).astype(int)
    df["is_after_holiday"] = df["yesterday"].isin(holidays_set).astype(int)
    df["is_month_start"] = (df["day"] <= 3).astype(int)
    df["is_month_end"] = (df["day"] >= 28).astype(int)

    df["is_spring"] = df["month"].isin([4, 5, 6]).astype(int)
    df["is_summer"] = df["month"].isin([7, 8]).astype(int)
    df["is_autumn"] = df["month"].isin([9, 10, 11]).astype(int)
    df["is_winter"] = df["month"].isin([12, 1, 2, 3]).astype(int)

    df["is_summer_vacation"] = df["month"].isin([7, 8]).astype(int)
    df["is_winter_vacation"] = df["month"].isin([12, 1, 2]).astype(int)

    df[["month_sin", "month_cos"]] = df["month"].apply(lambda m: pd.Series(sine_cosine_encoding(m, 12)))
    df[["dow_sin", "dow_cos"]] = df["dow"].apply(lambda d: pd.Series(sine_cosine_encoding(d, 7)))
    df[["doy_sin", "doy_cos"]] = df["date"].apply(lambda d: pd.Series(sine_cosine_encoding(d.dayofyear, 365)))
    df[["week_sin", "week_cos"]] = df["week"].apply(lambda w: pd.Series(sine_cosine_encoding(w, 52)))
    df[["quarter_sin", "quarter_cos"]] = df["quarter"].apply(lambda q: pd.Series(sine_cosine_encoding(q, 4)))

    return df.drop(columns=["tomorrow", "yesterday"])

def parse_store(item_full: str) -> str:
    return item_full.split("_")[0]

# =========================
# Dataset
# =========================
class SlidingDataset(Dataset):
    """
    - 모든 (입력 28일 → 타깃 7일) 슬라이딩 윈도우 생성
    - target_end_date 기준으로 train/val 마스크를 나눌 수 있게 target_end_dates 보관
    - 샘플 웨이트: 업장별 가중치(β)
    """
    def __init__(self, cfg: Config, df: pd.DataFrame):
        self.cfg = cfg
        df = df.copy()
        # 타입 정리
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
        df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors='coerce').fillna(0.0).clip(lower=0.0)

        # 피벗 (Dense하게 만드는 패딩/보간은 절대 하지 않음)
        pivot = df.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index()
        self.items = list(pivot.columns)
        self.dates = list(pivot.index)
        self.values = pivot.fillna(0.0).values.astype(np.float32)

        # 캘린더 피처
        hol = set(pd.to_datetime(cfg.custom_holidays_list)) if cfg.custom_holidays_list else set()
        caldf = build_enhanced_features(self.dates, hol)
        self.cal_feats = caldf.drop(columns=["date"]).values.astype(np.float32)
        self.cal_dim = self.cal_feats.shape[1]

        # 매핑들
        stores = [parse_store(it) for it in self.items]
        self.store2idx = {s: i for i, s in enumerate(sorted(set(stores)))}
        self.item_store_idx = np.array([self.store2idx[s] for s in stores], dtype=np.int64)
        self.n_stores = len(self.store2idx)

        # (선택) 클러스터 — 없으면 0
        clusters = [cluster_mapping.get(it, 0) for it in self.items]
        self.cluster2idx = {c: i for i, c in enumerate(sorted(set(clusters)))}
        self.item_cluster_idx = np.array([self.cluster2idx[c] for c in clusters], dtype=np.int64)
        self.n_clusters = len(self.cluster2idx)

        # 샘플 웨이트(업장 가중치)
        sw = cfg.store_weights or {}
        self.sample_weights = np.array([sw.get(stores[k], 1.0) for k in range(len(stores))], dtype=np.float32)

        # 슬라이딩 인덱스 구성
        Lx, Ly = cfg.in_len, cfg.out_len
        T = len(self.dates)
        max_start = T - (Lx + Ly)
        self.indices: List[Tuple[int, int]] = []
        self.target_end_dates: List[pd.Timestamp] = []

        for j in range(len(self.items)):
            for t0 in range(0, max_start + 1):
                end_date = self.dates[t0 + Lx + Ly - 1]  # 타깃 7일 마지막 날
                self.indices.append((t0, j))
                self.target_end_dates.append(end_date)

        self.target_end_dates = np.array(self.target_end_dates, dtype='datetime64[ns]')

        # 데이터 마지막 날짜(검증 upper bound로 사용)
        self.data_last_date: pd.Timestamp = self.dates[-1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        cfg = self.cfg
        t0, j = self.indices[idx]
        Lx, Ly = cfg.in_len, cfg.out_len

        x = self.values[t0:t0 + Lx, j]
        y = self.values[t0 + Lx:t0 + Lx + Ly, j]
        x_in = np.log1p(x) if cfg.log1p else x.copy()
        y_out = np.log1p(y) if cfg.log1p else y.copy()

        past_cal = self.cal_feats[t0:t0 + Lx, :]
        fut_cal = self.cal_feats[t0 + Lx:t0 + Lx + Ly, :]

        store_idx = self.item_store_idx[j]
        cluster_idx = self.item_cluster_idx[j]
        sample_w = self.sample_weights[j]

        # prob head 보조 통계 (현재 입력 28일 구간에서만 계산)
        mean = x.mean()
        std = x.std()
        slope = (x[-1] - x[0]) / max(1, len(x))

        return {
            "x": torch.from_numpy(x_in).float(),
            "y": torch.from_numpy(y_out).float(),
            "past_cal": torch.from_numpy(past_cal).float(),
            "fut_cal": torch.from_numpy(fut_cal).float(),
            "store_idx": torch.tensor(store_idx, dtype=torch.long),
            "cluster_idx": torch.tensor(cluster_idx, dtype=torch.long),
            "sample_w": torch.tensor(sample_w, dtype=torch.float32),
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
            "slope": torch.tensor(slope, dtype=torch.float32),
        }

# =========================
# Model
# =========================
def _act(name: str):
    return {"silu": nn.SiLU(), "relu": nn.ReLU(), "gelu": nn.GELU()}[name]

class NHiTSBlock(nn.Module):
    """Pooling으로 시계열 downsample → MLP → (trend stack이면) 보간"""
    def __init__(self, input_size:int, output_size:int, hidden:int, n_layers:int, dropout:float,
                 pooling_mode:str="MaxPool1d", n_pool_kernel_size:int=2, activation:str="silu",
                 cond_dim:int=0, interpolation_mode:str="linear"):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.pool = nn.MaxPool1d(kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True) \
            if pooling_mode == "MaxPool1d" else \
            nn.AvgPool1d(kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True)

        pooled_size = math.ceil(input_size / n_pool_kernel_size) + cond_dim
        layers = [nn.Linear(pooled_size, hidden), nn.LayerNorm(hidden), _act(activation), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), _act(activation), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, output_size)

    def forward(self, x, cond=None):
        # x: [B, L]
        z = self.pool(x.unsqueeze(1)).squeeze(1)  # [B, L_pooled]
        if cond is not None:
            z = torch.cat([z, cond], dim=-1)
        h = self.mlp(z)
        return self.out(h)  # [B, out_len]

class EnhancedNHiTS(nn.Module):
    """
    - Block 합성 → base forecast
    - meta(업장/클러스터/캘린더 평균)로 conditioning & residual 보정
    - Prob head(0 발생 확률)로 Hurdle Loss 구성
    """
    def __init__(self, cfg: Config, cal_dim:int, n_stores:int, n_clusters:int):
        super().__init__()
        self.cfg = cfg
        in_len, out_len = cfg.in_len, cfg.out_len

        # defaults
        k_list = cfg.n_pool_kernel_size or [2, 2, 1]
        t_list = cfg.stack_types or ["identity", "identity", "trend"]
        d_list = cfg.n_freq_downsample or [2, 1, 1]
        # extend if shorter
        while len(k_list) < cfg.n_blocks: k_list.append(k_list[-1])
        while len(t_list) < cfg.n_blocks: t_list.append("identity")
        while len(d_list) < cfg.n_blocks: d_list.append(1)
        self.stack_types = t_list
        self.down_list = d_list

        # embeddings & projection
        self.store_emb = nn.Embedding(n_stores, 64)
        self.cluster_emb = nn.Embedding(n_clusters, 32)
        self.cal_proj = nn.Linear(cal_dim, 128)

        cond_dim = 64 + 32 + 128  # store + cluster + calendar(avg)
        self.blocks = nn.ModuleList([
            NHiTSBlock(in_len, out_len, cfg.hidden, cfg.n_layers, cfg.dropout,
                       pooling_mode=cfg.pooling_mode,
                       n_pool_kernel_size=k_list[i],
                       activation=cfg.activation,
                       cond_dim=cond_dim,
                       interpolation_mode=cfg.interpolation_mode)
            for i in range(cfg.n_blocks)
        ])

        # trend stack용 basis (간단히 보관 — 현재 구현에서는 직접 보간은 block 내부 결과 사용)
        self.basis_weights = nn.ParameterList([
            nn.Parameter(torch.randn(out_len, max(1, out_len // max(1, d_list[i]))))
            for i in range(cfg.n_blocks)
        ])

        # meta residual head
        self.meta_head = nn.Sequential(
            nn.Linear(cond_dim, cfg.hidden), nn.LayerNorm(cfg.hidden), _act(cfg.activation), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, out_len)
        )

        # prob head (mean,std,slope 추가)
        self.prob_head = nn.Sequential(
            nn.Linear(cond_dim + 3, cfg.hidden), nn.LayerNorm(cfg.hidden), _act(cfg.activation), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden // 2), nn.LayerNorm(cfg.hidden // 2), _act(cfg.activation), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden // 2, out_len)
        )

    def forward(self, x, past_cal, fut_cal, store_idx, cluster_idx, mean, std, slope):
        # meta/cond
        store = self.store_emb(store_idx)
        clus  = self.cluster_emb(cluster_idx)
        cal_avg = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)
        cal   = self.cal_proj(cal_avg)
        cond  = torch.cat([store, clus, cal], dim=-1)

        # stacked blocks
        outs = []
        for i, blk in enumerate(self.blocks):
            y = blk(x, cond=cond)  # [B, out_len]
            # (선택) stack_types == "trend"일 때 basis 활용하는 변형 가능
            outs.append(y)
        base = torch.stack(outs, dim=0).sum(dim=0)

        # residual meta adjustment
        value_pred = base + self.meta_head(cond)

        # Hurdle prob
        prob_in = torch.cat([cond, mean.unsqueeze(1), std.unsqueeze(1), slope.unsqueeze(1)], dim=-1)
        p_logits = self.prob_head(prob_in)
        return value_pred, p_logits

# =========================
# Loss / EMA
# =========================
class UltraEnhancedHurdleLoss(nn.Module):
    """
    Loss = λ*BCE(prob) + 0.4*SMAPE(positive only) + 0.6*SMAPE(all with prob)
    - SMAPE 계산은 log1p 역변환 값으로
    - 0 근처 가중치/epsilon에 더 세밀한 처리
    """
    def __init__(self, eps=0.01, zero_weight=0.01, lambda_bce=0.15):
        super().__init__()
        self.eps = eps
        self.zero_weight = zero_weight
        self.lambda_bce = lambda_bce
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
        smape_pos = smape_pos * pos_mask

        p = torch.sigmoid(p_logits)
        y_hat = p * yp_val
        denom2 = (torch.abs(y_hat) + torch.abs(yt_val)).clamp_min(ultra_eps)
        smape_all = 2.0 * torch.abs(y_hat - yt_val) / denom2

        ultra_zero_weight = torch.where(
            yt_val < 0.01, torch.full_like(yt_val, self.zero_weight * 0.1),
            torch.where(yt_val < 0.1, torch.full_like(yt_val, self.zero_weight * 0.3),
                        torch.where(yt_val < 1.0, torch.full_like(yt_val, self.zero_weight * 0.6),
                                    torch.ones_like(yt_val)))
        )
        smape_all = smape_all * ultra_zero_weight

        bce_s = bce.mean(dim=1)
        pos_s = smape_pos.mean(dim=1)
        all_s = smape_all.mean(dim=1)

        sample_loss = self.lambda_bce * bce_s + 0.4 * pos_s + 0.6 * all_s
        sw = sample_w.view(-1)
        wsum = sw.sum().clamp_min(1e-8)
        loss = (sample_loss * sw).sum() / wsum
        return loss

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply_to(self, model: nn.Module):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(self.shadow[n])

# =========================
# Trainer
# =========================
class Trainer:
    """
    - Cosine with warmup 스케줄러 (per-step)
    - AMP(bfloat16/cuda), EMA, Gradient Clipping
    - Early stopping (튜닝 단계에서만)
    """
    def __init__(self, cfg: Config, dataset: SlidingDataset, epochs: int, batch_size: int, base_lr: float):
        self.cfg = cfg
        self.dataset = dataset
        self.device = torch.device(cfg.device)

        self.model = EnhancedNHiTS(cfg, dataset.cal_dim, dataset.n_stores, dataset.n_clusters).to(self.device)
        if cfg.use_compile and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
            except Exception as e:
                print("[WARN] torch.compile 실패:", e)

        self.criterion = UltraEnhancedHurdleLoss(cfg.eps_smape, cfg.zero_weight, cfg.hurdle_lambda)
        self.ema = EMA(self.model, decay=cfg.ema_decay)

        self.epochs = epochs
        self.batch_size = batch_size
        self.base_lr = base_lr

        # Optimizer
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=cfg.weight_decay)

        # AMP scaler
        self.use_amp = bool(cfg.use_amp and torch.cuda.is_available())
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # CUDA perf flags
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('high')

    def make_loaders(self, train_mask: np.ndarray, val_mask: Optional[np.ndarray] = None):
        idx_all = np.arange(len(self.dataset))
        tr_idx = idx_all[train_mask]
        tr_loader = DataLoader(
            torch.utils.data.Subset(self.dataset, tr_idx),
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers
        )
        if val_mask is None:
            return tr_loader, None
        va_idx = idx_all[val_mask]
        va_loader = DataLoader(
            torch.utils.data.Subset(self.dataset, va_idx),
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers
        )
        return tr_loader, va_loader

    def _build_cosine_warmup(self, steps_per_epoch: int):
        total = self.epochs * steps_per_epoch
        warmup = int(self.cfg.warmup_ratio * total)
        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            prog = float(step - warmup) / float(max(1, total - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        return torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, use_ema: bool = True) -> float:
        if loader is None: return float('inf')
        self.model.eval()
        bak = None
        if use_ema:
            bak = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.ema.apply_to(self.model)
        losses = []
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp)
        with amp_ctx:
            for b in loader:
                x = b["x"].to(self.device)
                y = b["y"].to(self.device)
                past = b["past_cal"].to(self.device)
                fut = b["fut_cal"].to(self.device)
                s = b["store_idx"].to(self.device)
                c = b["cluster_idx"].to(self.device)
                mean = b["mean"].to(self.device)
                std = b["std"].to(self.device)
                slope = b["slope"].to(self.device)
                sw = b["sample_w"].to(self.device)
                pos = (y > 0).float()

                v, p = self.model(x, past, fut, s, c, mean, std, slope)
                loss = self.criterion(v, p, y, pos, sw)
                losses.append(loss.detach().item())
        if use_ema and bak is not None:
            self.model.load_state_dict(bak)
        return float(np.mean(losses)) if len(losses) else float('inf')

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
            earlystop: bool = False):
        steps_per_epoch = max(1, len(train_loader))
        if self.cfg.scheduler_type == "cosine":
            sched = self._build_cosine_warmup(steps_per_epoch)
            step_on_val = False
        elif self.cfg.scheduler_type == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.5, patience=5)
            step_on_val = True
        else:  # onecycle (사용 비권장 — 요구사항은 cosine)
            sched = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=self.cfg.max_lr,
                                                        epochs=self.epochs, steps_per_epoch=steps_per_epoch)
            step_on_val = False

        best = float('inf')
        best_state = None
        noimp = 0
        patience = self.cfg.earlystop_patience if earlystop else 10**9  # 재학습에서는 사실상 미적용

        global_step = 0
        for ep in range(1, self.epochs + 1):
            self.model.train()
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp)

            for b in train_loader:
                x = b["x"].to(self.device)
                y = b["y"].to(self.device)
                past = b["past_cal"].to(self.device)
                fut = b["fut_cal"].to(self.device)
                s = b["store_idx"].to(self.device)
                c = b["cluster_idx"].to(self.device)
                mean = b["mean"].to(self.device)
                std = b["std"].to(self.device)
                slope = b["slope"].to(self.device)
                sw = b["sample_w"].to(self.device)
                pos = (y > 0).float()

                with amp_ctx:
                    v, p = self.model(x, past, fut, s, c, mean, std, slope)
                    loss = self.criterion(v, p, y, pos, sw)

                self.optim.zero_grad(set_to_none=True)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optim.step()

                # scheduler step
                if not step_on_val:
                    sched.step()
                self.ema.update(self.model)
                global_step += 1

            # validation
            val = self.evaluate(val_loader, use_ema=True) if val_loader is not None else float('inf')
            if step_on_val and val_loader is not None:
                sched.step(val)
            print(f"[Epoch {ep:03d}] val={val:.6f}")

            if val_loader is None:
                # 재학습 단계: 베스트 추적은 하지 않고 전체 학습
                continue

            if val < best:
                best = val
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                noimp = 0
            else:
                noimp += 1
                if noimp >= patience:
                    print(f"Early stopping at epoch {ep}")
                    break

        # 복원
        if val_loader is not None and best_state is not None:
            self.model.load_state_dict(best_state)
        # EMA 최종 적용
        self.ema.apply_to(self.model)
        return self.model, best

# =========================
# Split masks (by target end date)
# =========================
def build_train_val_masks(ds: SlidingDataset, train_cutoff: str, val_start: str):
    ted = pd.to_datetime(ds.target_end_dates)
    train_mask = (ted <= pd.to_datetime(train_cutoff))
    val_mask = (ted >= pd.to_datetime(val_start)) & (ted <= pd.to_datetime(ds.data_last_date))
    return train_mask, val_mask

# --- Optuna objective & run helper ---
def suggest_space(trial):
    return {
        "hidden": trial.suggest_categorical("hidden", [512, 640, 768, 820, 896, 1024]),
        "n_layers": trial.suggest_int("n_layers", 1, 10),
        "n_blocks": trial.suggest_int("n_blocks", 1, 10),
        "dropout": trial.suggest_float("dropout", 0.0, 0.30),
        "base_lr": trial.suggest_float("base_lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.10),
        "grad_clip": trial.suggest_float("grad_clip", 0.3, 1.0),
        # 고정/옵션
        "activation": trial.suggest_categorical("activation", ["silu","relu","gelu"]),
        "scheduler_type": "cosine",
    }

def apply_params_to_cfg(cfg, params: dict):
    # cfg에 반영 (존재하는 항목만)
    for k, v in params.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

def run_one_trial(cfg, ds, tr_mask, va_mask, params):
    # cfg 복사 느낌으로 적용
    tmp_cfg = Config(**{**cfg.__dict__})
    apply_params_to_cfg(tmp_cfg, params)

    trainer = Trainer(tmp_cfg, ds, epochs=tmp_cfg.EPOCHS_TUNE, batch_size=tmp_cfg.BATCH_TUNE, base_lr=tmp_cfg.base_lr)
    tr_loader, va_loader = trainer.make_loaders(tr_mask, va_mask)
    _, best_val = trainer.fit(tr_loader, va_loader, earlystop=True)
    return float(best_val)

def optuna_tune(cfg, ds, tr_mask, va_mask):
    assert HAS_OPTUNA, "optuna 미설치"
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=cfg.seed)
    direction = "minimize"  # val loss 최소화

    def objective(trial):
        # 1) 공간 샘플링
        p = suggest_space(trial)

        # 2) trial 설정을 cfg에 반영
        #    (모델/트레이너가 cfg를 읽어 생성되므로 여기서 값만 바꾸면 됨)
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg.hidden = p["hidden"]
        trial_cfg.n_layers = p["n_layers"]
        trial_cfg.n_blocks = p["n_blocks"]     # ✅ 추가
        trial_cfg.dropout = p["dropout"]
        trial_cfg.base_lr = p["base_lr"]
        trial_cfg.weight_decay = p["weight_decay"]
        trial_cfg.warmup_ratio = p["warmup_ratio"]
        trial_cfg.grad_clip = p["grad_clip"]
        trial_cfg.activation = p["activation"]
        trial_cfg.scheduler_type = p["scheduler_type"]

        # 3) 데이터/트레이너 생성 (train/val 마스크는 기존 로직 재사용)
        ds_local = SlidingDataset(trial_cfg, train_df)
        tr_mask, va_mask = build_train_val_masks(ds_local, trial_cfg.train_cutoff, trial_cfg.val_start)
        trainer_local = Trainer(trial_cfg, ds_local, epochs=trial_cfg.EPOCHS_TUNE,
                                batch_size=trial_cfg.BATCH_TUNE, base_lr=trial_cfg.base_lr)
        tl, vl = trainer_local.make_loaders(tr_mask, va_mask)

        # 4) 학습 & 평가
        _, best_val = trainer_local.fit(tl, vl, earlystop=True)

        # 5) Optuna에 반환 (값이 낮을수록 좋음)
        return float(best_val)


    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=cfg.N_TRIALS, show_progress_bar=False)

    best_params = study.best_trial.params
    best_val = study.best_value
    print("[OPTUNA] best val:", best_val, "best params:", best_params)

    # 필요하면 저장
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    study_path = os.path.join(cfg.checkpoint_dir, "optuna_study.json")
    with open(study_path, "w", encoding="utf-8") as f:
        json.dump({"best_params": best_params, "best_val": best_val}, f, ensure_ascii=False, indent=2)
    print(f"[OPTUNA] saved study → {study_path}")

    return best_params, best_val

def all_full_mask(ds: SlidingDataset):
    return np.ones(len(ds), dtype=bool)

# =========================
# Predict (STRICT: no padding/replication if < in_len)
# =========================
@torch.no_grad()
def predict_one_file(cfg: Config, model: nn.Module, test_df: pd.DataFrame,
                     store2idx: Dict[str, int], cluster2idx: Dict[int, int]) -> pd.DataFrame:
    """
    - 각 test 파일에서 '마지막 28일'만 입력으로 사용
    - 28일 미만이면 스킵(빈 DataFrame 반환)
    - 외부/추가 과거 연결, 패딩, 반복 금지
    """
    device = torch.device(cfg.device)
    t = test_df.copy()
    if cfg.date_col not in t.columns or cfg.item_col not in t.columns or cfg.target_col not in t.columns:
        return pd.DataFrame()

    t[cfg.date_col] = pd.to_datetime(t[cfg.date_col], errors='coerce')
    t = t.dropna(subset=[cfg.date_col])
    if len(t) == 0:
        return pd.DataFrame()
    t[cfg.target_col] = pd.to_numeric(t[cfg.target_col], errors='coerce').fillna(0.0).clip(lower=0.0)

    pivot = t.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index()
    if pivot.shape[1] == 0:
        return pd.DataFrame()

    dates = list(pivot.index)
    values = pivot.values.astype(np.float32)
    items = list(pivot.columns)

    # STRICT: 마지막 28일만 — 부족하면 바로 스킵
    if len(dates) < cfg.in_len:
        print(f"[SKIP] test window has only {len(dates)} days (<{cfg.in_len}).")
        return pd.DataFrame()

    last_28_dates = dates[-cfg.in_len:]
    x_np = values[-cfg.in_len:, :].T  # [B, 28]
    if cfg.log1p:
        x_np = np.log1p(x_np)

    # 미래 7일 캘린더(날짜) 만들기
    last_date = last_28_dates[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, cfg.out_len + 1)]

    hol = set(pd.to_datetime(cfg.custom_holidays_list)) if cfg.custom_holidays_list else set()
    past_cal = build_enhanced_features(last_28_dates, hol).drop(columns=["date"]).values.astype(np.float32)
    fut_cal  = build_enhanced_features(future_dates, hol).drop(columns=["date"]).values.astype(np.float32)

    B = len(items)
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
    past_c = torch.from_numpy(np.repeat(past_cal[None, :, :], B, axis=0)).to(device=device, dtype=torch.float32)
    fut_c  = torch.from_numpy(np.repeat(fut_cal[None, :, :], B, axis=0)).to(device=device, dtype=torch.float32)

    stores = [parse_store(it) for it in items]
    clusters = [cluster_mapping.get(it, 0) for it in items]
    sidx = torch.as_tensor([store2idx.get(s, 0) for s in stores], device=device, dtype=torch.long)
    cidx = torch.as_tensor([cluster2idx.get(c, 0) for c in clusters], device=device, dtype=torch.long)

    mean = x.mean(dim=1)
    std = x.std(dim=1)
    slope = (x[:, -1] - x[:, 0]) / cfg.in_len

    model.eval()
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=bool(cfg.use_amp and torch.cuda.is_available()))
    with amp_ctx:
        v, p = model(x, past_c, fut_c, sidx, cidx, mean, std, slope)
        y = torch.expm1(v).clamp_min(0.0)
        prob = torch.sigmoid(p)
        pred = (y * prob).detach().cpu().numpy()  # [B, 7]

    df = pd.DataFrame(pred.T, index=[f"D+{k}" for k in range(1, cfg.out_len + 1)], columns=items)
    # 후처리(과도치 soft clip + 음수 방지 + 반올림은 메인에서)
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

# =========================
# Main
# =========================
if __name__ == "__main__":
    cfg = Config()

    # 사용자 기본값 반영
    if cfg.custom_holidays_list is None:
        cfg.custom_holidays_list = DEFAULT_CUSTOM_HOLIDAYS
    if cfg.store_weights is None:
        cfg.store_weights = DEFAULT_STORE_WEIGHTS

    # 수동 "최적" 파라미터 주입(예: 튜닝 결과를 여기 넣어 재현)
    if cfg.FIXED_BEST_PARAMS is None:
        cfg.FIXED_BEST_PARAMS = {
            # 예시: 그대로 쓰거나 수정 가능
            # "hidden": 512, "n_layers": 2, "dropout": 0.1, ...
        }

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    set_seed(cfg.seed)

    # 1) 데이터 로드 & Dataset 구성
    assert os.path.exists(cfg.train_csv), f"train_csv not found: {cfg.train_csv}"
    train_df = pd.read_csv(cfg.train_csv)
    ds = SlidingDataset(cfg, train_df)
    print(f"Dates: {ds.dates[0].date()} ~ {ds.dates[-1].date()}  (items={len(ds.items)})")

    # 2) Train/Val mask 생성 (튜닝 단계)
    tr_mask, va_mask = build_train_val_masks(ds, cfg.train_cutoff, cfg.val_start)
    print(f"Train windows: {tr_mask.sum():,}  |  Val windows: {va_mask.sum():,}")

    # 3) Trainer 생성 & 로더 만들기
    trainer = Trainer(cfg, ds, epochs=cfg.EPOCHS_TUNE, batch_size=cfg.BATCH_TUNE, base_lr=cfg.base_lr)
    train_loader, val_loader = trainer.make_loaders(tr_mask, va_mask)

    # 3.5) Optuna 튜닝
    if cfg.USE_OPTUNA:
        if not HAS_OPTUNA:
            raise RuntimeError("cfg.USE_OPTUNA=True 이지만 optuna가 설치되어 있지 않습니다.")
        # 튜닝 에폭은 가볍게 권장: 20~35
        best_params, best_val_from_tune = optuna_tune(cfg, ds, tr_mask, va_mask)
        cfg.FIXED_BEST_PARAMS = best_params
        apply_params_to_cfg(cfg, best_params)

    # 4) 튜닝 단계 학습(early stopping 적용)
    if not cfg.USE_OPTUNA:
        # 4) 튜닝 단계 학습(early stopping 적용)
        model_tuned, best_val = trainer.fit(train_loader, val_loader, earlystop=True)
        print(f"[TUNE DONE] best val = {best_val:.6f}")
    else:
        # 방금 optuna로 얻은 best_params를 cfg에 반영했고,
        # 동일 cfg로 다시 한 번 튜닝용 학습을 재수행하여 ckpt 저장
        trainer = Trainer(cfg, ds, epochs=cfg.EPOCHS_TUNE, batch_size=cfg.BATCH_TUNE, base_lr=cfg.base_lr)
        train_loader, val_loader = trainer.make_loaders(tr_mask, va_mask)
        model_tuned, best_val = trainer.fit(train_loader, val_loader, earlystop=True)
        print(f"[TUNE DONE/OPTUNA] best val = {best_val:.6f}")

    # 체크포인트 저장(튜닝 모델)
    tuned_ckpt = os.path.join(cfg.checkpoint_dir, "nhits_tuned.pth")
    torch.save({
        "model_state": model_tuned.state_dict(),
        "cfg": cfg.__dict__,
        "store2idx": ds.store2idx,
        "cluster2idx": ds.cluster2idx,
        "best_val": best_val
    }, tuned_ckpt)
    print(f"Saved tuned checkpoint → {tuned_ckpt}")

    # 5) Refit(재학습): Train+Val 전체 윈도우로 다시 학습 (에폭은 EPOCHS_FULL)
    print("\n[REFIT] Train on FULL windows (Train + Val all)")
    full_trainer = Trainer(cfg, ds, epochs=cfg.EPOCHS_FULL, batch_size=cfg.BATCH_FULL, base_lr=cfg.base_lr)

    # 수동 최적 파라미터 주입 (필요 시 cfg 값 갱신)
    for k, v in (cfg.FIXED_BEST_PARAMS or {}).items():
        if hasattr(full_trainer.model.cfg, k):
            setattr(full_trainer.model.cfg, k, v)

    full_mask = all_full_mask(ds)
    full_loader, _ = full_trainer.make_loaders(full_mask, None)
    model_full, _ = full_trainer.fit(full_loader, val_loader=None, earlystop=False)

    refit_ckpt = os.path.join(cfg.checkpoint_dir, "nhits_refit_full.pth")
    torch.save({
        "model_state": model_full.state_dict(),
        "cfg": cfg.__dict__,
        "store2idx": ds.store2idx,
        "cluster2idx": ds.cluster2idx,
        "best_val": best_val
    }, refit_ckpt)
    print(f"Saved refit checkpoint → {refit_ckpt}")

    # 6) Test 추론 — STRICT 28일 입력만 사용, 부족하면 스킵
    assert os.path.exists(cfg.submission_template_csv), "submission template not found"
    sub_template = pd.read_csv(cfg.submission_template_csv)
    test_files = sorted(glob.glob(cfg.test_glob))
    print(f"\nFound test files: {len(test_files)}")

    all_blocks = []
    for ti, tf in enumerate(test_files):
        try:
            tdf = pd.read_csv(tf)
        except Exception as e:
            print(f"[SKIP] read fail: {tf} ({e})")
            continue

        block = predict_one_file(cfg, model_full, tdf, ds.store2idx, ds.cluster2idx)
        if block.empty:
            print(f"[SKIP] no valid prediction for {os.path.basename(tf)}")
            continue
        block.index = [f"TEST_{ti:02d}+{k}일" for k in range(1, cfg.out_len + 1)]
        all_blocks.append(block)

    if len(all_blocks) == 0:
        print("[WARN] No valid test predictions. Submission not written.")
    else:
        final_submit = pd.concat(all_blocks, axis=0)
        final_submit.reset_index(inplace=True)
        final_submit.rename(columns={"index": "영업일자"}, inplace=True)
        # 템플릿 컬럼 순서 맞추기 & 결측 0 채움
        final_submit = final_submit.reindex(columns=sub_template.columns, fill_value=0)

        # 소프트 클리핑 + 음수 방지 + 반올림
        num_cols = [c for c in final_submit.columns if c != "영업일자"]
        for col in num_cols:
            Q95 = final_submit[col].quantile(0.95)
            final_submit[col] = np.where(final_submit[col] > Q95 * 2.0, Q95 * 1.5, final_submit[col])
        final_submit[num_cols] = np.rint(np.clip(final_submit[num_cols].values, a_min=0, a_max=None)).astype(int)

        os.makedirs(os.path.dirname(cfg.out_submission_csv), exist_ok=True)
        final_submit.to_csv(cfg.out_submission_csv, index=False, encoding="utf-8-sig")
        print(f"[DONE] Submission saved → {cfg.out_submission_csv}")
