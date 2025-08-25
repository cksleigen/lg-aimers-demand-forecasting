# -*- coding: utf-8 -*-
"""
All-in-One Pipeline (전처리 → Rolling-CV 튜닝(Optuna) → 전체 학습 → 추론/제출)
- 모델: N-HiTS Large + Hurdle (확률×양)
- LightGBM 파이프라인과 동일 경로/제출 포맷
- 딥러닝 친화 인코딩(원시 28일 + 캘린더 + 임베딩)
- A6000 최적화: AMP(bfloat16), torch.compile, EMA, OneCycleLR
- 사용자 제공 일정 반영: 학습 컷오프(2024-06-15), 테스트 블록 주차/요일 정렬
- 개선 적용: 날짜기반 Rolling-CV(3fold), week-of-year sin/cos, 안전한 윈도우 컷오프, 메뉴 임베딩 옵션

사용법
1) 경로/공휴일/가중치는 기본값으로 LightGBM 코드와 동일. 필요 시 Config 수정.
2) 튜닝: Rolling-CV(3fold)로 Optuna가 자동 탐색 후, 최적값으로 전체 학습.
3) 최종 제출 파일: ./data/nhits_hurdle_submission.csv
"""
import os
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =====================
# Config
# =====================
@dataclass
class Config:
    # 경로
    train_csv: str = "./data/train/train.csv"
    test_glob: str = "./data/test/*.csv"
    submission_template_csv: str = "./data/sample_submission.csv"
    out_submission_csv: str = "./data/nhits_hurdle_submission.csv"

    # 컬럼명
    date_col: str = "영업일자"
    item_col: str = "영업장명_메뉴명"
    target_col: str = "매출수량"

    # 윈도우
    in_len: int = 28
    out_len: int = 7

    # 학습 데이터 컷오프(누수 방지)
    train_end_date: str = "2024-06-15"  # 이 날까지의 데이터만 학습 타깃으로 허용

    # 공통 학습
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log1p: bool = True

    # 전체 학습(최종)
    EPOCHS_FULL: int = 120
    BATCH_FULL: int = 1024
    BASE_LR_FULL: float = 1e-3
    MAX_LR_FULL: float = 3e-3
    WD_FULL: float = 5e-4

    # 튜닝(Optuna + Rolling-CV)
    USE_OPTUNA: bool = True
    N_TRIALS: int = 30
    EPOCHS_TUNE: int = 30
    BATCH_TUNE: int = 768
    BASE_LR_TUNE: float = 1e-3
    MAX_LR_TUNE: float = 2e-3
    WD_TUNE: float = 3e-4

    # Rolling-CV fold의 검증 종료일(타깃 마지막 날짜) — 사용자 데이터 기준
    # Fold-3: 2024-06-08~2024-06-14, Fold-2: 06-01~06-07, Fold-1: 05-25~05-31
    cv_fold_end_dates: Tuple[str, str, str] = ("2024-06-14", "2024-06-07", "2024-05-31")

    # DataLoader
    num_workers: int = 12
    pin_memory: bool = True
    persistent_workers: bool = True

    # 모델 기본값 (튜닝으로 덮일 수 있음)
    hidden: int = 512
    scales: Tuple[int, ...] = (1,2,3,4,7,14)
    store_emb_dim: int = 32
    use_menu_embedding: bool = False
    menu_emb_dim: int = 16
    dropout: float = 0.1

    # 허들/손실
    eps_smape: float = 0.1
    zero_weight: float = 0.1
    hurdle_lambda: float = 0.3

    # AMP/compile/EMA
    use_amp: bool = True
    use_compile: bool = True
    ema_decay: float = 0.999

    # 가중치/공휴일
    store_weights: Dict[str, float] = None
    custom_holidays_list: List[str] = None

    # 후처리(업장별 배수 보정)
    apply_postprocess: bool = False
    store_post_scales: Dict[str, float] = None


DEFAULT_STORE_WEIGHTS = {
    "미라시아": 7.71,
    "담하": 6.51,
    "연회장": 3.48,
    "라그로타": 3.44,
    "느티나무 셀프BBQ": 2.78,
    "화담숲주막": 1.43,
    "카페테리아": 1.31,
    "화담숲카페": 1.14,
    "포레스트릿": 1.00,
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

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sine_cosine_day_of_year(d: pd.Timestamp):
    doy = d.dayofyear
    return math.sin(2 * math.pi * doy / 365.0), math.cos(2 * math.pi * doy / 365.0)


def sine_cosine_month(m: int):
    return math.sin(2 * math.pi * m / 12.0), math.cos(2 * math.pi * m / 12.0)


def sine_cosine_week(week: int):
    # ISO week: 1..53 를 52로 정규화 (wrap)
    w = ((week - 1) % 52) + 1
    return math.sin(2 * math.pi * w / 52.0), math.cos(2 * math.pi * w / 52.0)


def season_onehot(m: int):
    if m in [4,5,6]: return [1,0,0,0]
    if m in [7,8]:   return [0,1,0,0]
    if m in [9,10,11]: return [0,0,1,0]
    return [0,0,0,1]


def build_calendar_features(dates: List[pd.Timestamp], holidays_set: set) -> pd.DataFrame:
    df = pd.DataFrame({"date": dates})
    df["dow"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].apply(lambda d: int(d.isocalendar().week))
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
    df["is_holiday"] = df["date"].isin(holidays_set).astype(int)
    df["is_off"] = ((df["is_weekend"]==1) | (df["is_holiday"]==1)).astype(int)

    df[["month_sin","month_cos"]] = df["month"].apply(lambda m: pd.Series(sine_cosine_month(m)))
    df[["doy_sin","doy_cos"]] = df["date"].apply(lambda d: pd.Series(sine_cosine_day_of_year(d)))
    df[["week_sin","week_cos"]] = df["week"].apply(lambda w: pd.Series(sine_cosine_week(w)))
    df[["spring","summer","autumn","winter"]] = df["month"].apply(lambda m: pd.Series(season_onehot(m)))

    df["tomorrow"] = df["date"] + pd.Timedelta(days=1)
    df["is_holiday_eve"] = ((df["tomorrow"].isin(holidays_set)) | (df["tomorrow"].dt.weekday.isin([5,6]))).astype(int)
    df["tomm_is_off"] = df["is_holiday_eve"].astype(int)
    return df.drop(columns=["tomorrow"])  # ['date',..., 'tomm_is_off']


def parse_store_name(item_full: str) -> str:
    return item_full.split("_")[0]

def parse_menu_name(item_full: str) -> str:
    return item_full.split("_", 1)[1] if "_" in item_full else item_full


# =====================
# Dataset (Sliding Window with date cutoff & meta)
# =====================
class SlidingWindowDataset(Dataset):
    def __init__(self, cfg: Config, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df.copy()
        self.df[cfg.date_col] = pd.to_datetime(self.df[cfg.date_col])
        self.df[cfg.target_col] = self.df[cfg.target_col].clip(lower=0)

        pivot = self.df.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index()
        self.items = list(pivot.columns)
        self.dates = list(pivot.index)
        self.values = pivot.fillna(0.0).values.astype(np.float32)

        holidays_set = set(pd.to_datetime(cfg.custom_holidays_list))
        caldf = build_calendar_features(self.dates, holidays_set)
        self.cal_feats = caldf.drop(columns=["date"]).values.astype(np.float32)
        self.cal_feat_names = [c for c in caldf.columns if c != "date"]

        stores = [parse_store_name(it) for it in self.items]
        menus  = [parse_menu_name(it) for it in self.items]
        self.store2idx = {s: i for i, s in enumerate(sorted(set(stores)))}
        self.item_store_idx = np.array([self.store2idx[s] for s in stores], dtype=np.int64)
        self.n_stores = len(self.store2idx)

        self.menu2idx = {m: i for i, m in enumerate(sorted(set(menus)))}
        self.item_menu_idx = np.array([self.menu2idx[m] for m in menus], dtype=np.int64)
        self.n_menus = len(self.menu2idx)

        sw = cfg.store_weights
        self.sample_weights = np.array([sw.get(parse_store_name(it), 1.0) for it in self.items], dtype=np.float32)

        # 윈도우 인덱스 & 타깃 종료일 메타 생성 (cutoff 적용)
        self.indices: List[Tuple[int,int]] = []  # (t0, j)
        self.target_end_dates: List[pd.Timestamp] = []
        T = len(self.dates)
        Lx, Ly = cfg.in_len, cfg.out_len
        cutoff = pd.to_datetime(cfg.train_end_date)
        max_start = T - (Lx + Ly)
        for j in range(len(self.items)):
            for t0 in range(0, max_start + 1):
                end_date = self.dates[t0 + Lx + Ly - 1]
                if end_date <= cutoff:  # 타깃 마지막 날이 컷오프 이내인 샘플만 허용
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
        menu_idx  = self.item_menu_idx[j]
        sample_w = self.sample_weights[j]
        zero_mask = (y == 0).astype(np.float32)
        pos_mask = (y > 0).astype(np.float32)

        return {
            "x": torch.from_numpy(x_in).float(),
            "y": torch.from_numpy(y_out).float(),
            "past_cal": torch.from_numpy(past_cal).float(),
            "fut_cal": torch.from_numpy(fut_cal).float(),
            "store_idx": torch.tensor(store_idx, dtype=torch.long),
            "menu_idx": torch.tensor(menu_idx, dtype=torch.long),
            "sample_w": torch.tensor(sample_w, dtype=torch.float32),
            "zero_mask": torch.from_numpy(zero_mask).float(),
            "pos_mask": torch.from_numpy(pos_mask).float(),
        }


# =====================
# Model (N-HiTS Large + Hurdle) with optional menu embedding
# =====================
class MultiScaleHead(nn.Module):
    def __init__(self, in_len: int, out_len: int, cal_dim: int, hidden: int, scale: int, dropout: float):
        super().__init__()
        self.down = nn.AvgPool1d(kernel_size=scale, stride=scale, ceil_mode=True)
        down_len = math.ceil(in_len / scale)
        self.mlp = nn.Sequential(
            nn.Linear(down_len + cal_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.proj = nn.Linear(hidden, out_len)

    def forward(self, x_seq: torch.Tensor, past_cal: torch.Tensor):
        x = x_seq.unsqueeze(1)
        x_down = self.down(x).squeeze(1)
        cal = past_cal.transpose(1, 2)
        cal_down = self.down(cal).mean(dim=2)
        feat = torch.cat([x_down, cal_down], dim=-1)
        return self.proj(self.mlp(feat))


class NHITS_Hurdle(nn.Module):
    def __init__(self, in_len: int, out_len: int, cal_dim: int, n_stores: int, hidden: int,
                 scales: Tuple[int, ...], dropout: float, store_emb_dim: int,
                 use_menu_embedding: bool = False, n_menus: int = 0, menu_emb_dim: int = 16):
        super().__init__()
        self.use_menu = use_menu_embedding and (n_menus > 0) and (menu_emb_dim > 0)
        self.store_emb = nn.Embedding(n_stores, store_emb_dim)
        self.menu_emb = nn.Embedding(n_menus, menu_emb_dim) if self.use_menu else None
        cat_dim = store_emb_dim + (menu_emb_dim if self.use_menu else 0)

        self.value_heads = nn.ModuleList([MultiScaleHead(in_len, out_len, cal_dim + cat_dim, hidden, s, dropout) for s in scales])
        self.prob_heads  = nn.ModuleList([MultiScaleHead(in_len, out_len, cal_dim + cat_dim, hidden, s, dropout) for s in scales])
        self.future_proj_v = nn.Sequential(nn.Linear(cal_dim, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
        self.future_proj_p = nn.Sequential(nn.Linear(cal_dim, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))

    def forward(self, x: torch.Tensor, past_cal: torch.Tensor, fut_cal: torch.Tensor, store_idx: torch.Tensor, menu_idx: torch.Tensor):
        B, Lx, F = past_cal.shape
        store_e = self.store_emb(store_idx)
        if self.menu_emb is not None:
            menu_e = self.menu_emb(menu_idx)
            cat_e = torch.cat([store_e, menu_e], dim=-1)
        else:
            cat_e = store_e
        cat_feat = cat_e.unsqueeze(1).expand(-1, Lx, -1)
        past_plus = torch.cat([past_cal, cat_feat], dim=-1)
        v = torch.stack([head(x, past_plus) for head in self.value_heads], dim=0).sum(dim=0)
        p = torch.stack([head(x, past_plus) for head in self.prob_heads ], dim=0).sum(dim=0)
        v = v + self.future_proj_v(fut_cal).squeeze(-1)
        p = p + self.future_proj_p(fut_cal).squeeze(-1)
        return v, p  # value(log1p), prob(logits)


# =====================
# Loss / EMA
# =====================
class HurdleLoss(nn.Module):
    def __init__(self, eps: float = 0.1, zero_weight: float = 0.1, lambda_bce: float = 0.3):
        super().__init__()
        self.eps = eps
        self.zero_weight = zero_weight
        self.lambda_bce = lambda_bce
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, v_pred_log, p_logits, y_true_log, pos_mask, sample_w):
        """반환값: (배치 가중 평균 손실, 배치 내 각 샘플(아이템)별 손실, 각 샘플 가중치)
        - 검증 단계에서 β 가중치 직접 반영을 위해 per-sample loss를 함께 리턴
        """
        # 발생 여부 분류 손실
        z = (pos_mask > 0).float()  # [B,L]
        bce = self.bce(p_logits, z)  # [B,L]

        # 값 회귀 손실 (양수일 때만)
        yp_val = torch.expm1(v_pred_log).clamp_min(0.0)
        yt_val = torch.expm1(y_true_log).clamp_min(0.0)
        denom = (torch.abs(yp_val) + torch.abs(yt_val)).clamp_min(self.eps)
        smape_pos = 2.0 * torch.abs(yp_val - yt_val) / denom
        smape_pos = smape_pos * pos_mask  # [B,L]

        # 최종 y_hat 기반 보조 손실(0 타깃 가중 완화)
        p = torch.sigmoid(p_logits)
        y_hat = p * yp_val
        denom2 = (torch.abs(y_hat) + torch.abs(yt_val)).clamp_min(self.eps)
        smape_all = 2.0 * torch.abs(y_hat - yt_val) / denom2
        w_zero = torch.where(yt_val <= 0.0 + 1e-8, torch.full_like(yt_val, self.zero_weight), torch.ones_like(yt_val))
        smape_all = smape_all * w_zero  # [B,L]

        # 시간 축 평균 → 샘플별 스칼라 손실
        bce_s = bce.mean(dim=1)
        pos_s = smape_pos.mean(dim=1)
        all_s = smape_all.mean(dim=1)
        sample_loss = self.lambda_bce * bce_s + pos_s + 0.2 * all_s  # [B]

        # β 가중 평균 (정규화)
        sw = sample_w.view(-1)
        wsum = sw.sum().clamp_min(1e-8)
        loss = (sample_loss * sw).sum() / wsum
        return loss, sample_loss.detach(), sw.detach()


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
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
# Trainer + Rolling-CV helpers
# =====================
class Trainer:
    def __init__(self, cfg: Config, dataset: SlidingWindowDataset, epochs: int, batch_size: int, base_lr: float, max_lr: float, weight_decay: float):
        self.cfg = cfg
        self.dataset = dataset
        self.device = torch.device(cfg.device)
        cal_dim = dataset.cal_feats.shape[1]
        model = NHITS_Hurdle(cfg.in_len, cfg.out_len, cal_dim, dataset.n_stores, cfg.hidden, cfg.scales, cfg.dropout,
                             cfg.store_emb_dim, cfg.use_menu_embedding, dataset.n_menus, cfg.menu_emb_dim).to(self.device)
        if cfg.use_compile and torch.cuda.is_available():
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception as e:
                print("torch.compile 실패 → 비컴파일로 진행:", e)
        self.model = model
        # 옵티마이저/스케줄러는 train_loader 크기에 맞춰 train_with_loaders에서 초기화
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.criterion = HurdleLoss(cfg.eps_smape, cfg.zero_weight, cfg.hurdle_lambda)
        self.ema = EMA(self.model, decay=cfg.ema_decay)
        self.epochs = epochs
        self.batch_size = batch_size

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('high')

    def make_loaders_from_mask(self, mask_val: np.ndarray):
        idx_all = np.arange(len(self.dataset))
        val_idx = idx_all[mask_val]
        train_idx = idx_all[~mask_val]
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
        self.model.eval()
        backup = None
        if use_ema:
            backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.ema.apply_to(self.model)
        per_sample_losses = []
        per_sample_weights = []
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (self.cfg.use_amp and torch.cuda.is_available()) else torch.cuda.amp.autocast(enabled=False)
        with amp_ctx:
            for batch in loader:
                x = batch["x"].to(self.device); y = batch["y"].to(self.device)
                past_cal = batch["past_cal"].to(self.device); fut_cal = batch["fut_cal"].to(self.device)
                store_idx = batch["store_idx"].to(self.device); menu_idx = batch["menu_idx"].to(self.device)
                sample_w = batch["sample_w"].to(self.device); pos_mask = batch["pos_mask"].to(self.device)
                v_pred, p_logits = self.model(x, past_cal, fut_cal, store_idx, menu_idx)
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

    def train_with_loaders(self, train_loader: DataLoader, val_loader: DataLoader):
        # 폴드별로 옵티마이저/스케줄러 초기화 (train_loader 길이 기준)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=self.max_lr,
            epochs=self.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            div_factor=self.max_lr / self.base_lr
        )
        best_val = float("inf"); best_state = None
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (self.cfg.use_amp and torch.cuda.is_available()) else torch.cuda.amp.autocast(enabled=False)
            for batch in train_loader:
                x = batch["x"].to(self.device); y = batch["y"].to(self.device)
                past_cal = batch["past_cal"].to(self.device); fut_cal = batch["fut_cal"].to(self.device)
                store_idx = batch["store_idx"].to(self.device); menu_idx = batch["menu_idx"].to(self.device)
                sample_w = batch["sample_w"].to(self.device); pos_mask = batch["pos_mask"].to(self.device)
                with amp_ctx:
                    v_pred, p_logits = self.model(x, past_cal, fut_cal, store_idx, menu_idx)
                    loss, _, _ = self.criterion(v_pred, p_logits, y, pos_mask, sample_w)
                self.optim.zero_grad(set_to_none=True)
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step(); self.sched.step(); self.ema.update(self.model)
            val_loss = self.evaluate(val_loader, use_ema=True)
            print(f"[Epoch {epoch:03d}] val_masked: {val_loss:.5f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.model, best_val


# =====================
# Rolling-CV (by target end-date weekly blocks)
# =====================

def make_val_mask_by_week(dataset: SlidingWindowDataset, end_date_str: str) -> np.ndarray:
    end_date = pd.to_datetime(end_date_str)
    start_date = end_date - pd.Timedelta(days=6)
    # 타깃 종료일이 [start_date, end_date]에 속한 윈도우들을 검증으로
    ted = dataset.target_end_dates
    return (ted >= start_date) & (ted <= end_date)


def evaluate_cfg_rolling(cfg: Config, epochs: int, batch_size: int, base_lr: float, max_lr: float, weight_decay: float) -> float:
    train_df = pd.read_csv(cfg.train_csv)
    ds = SlidingWindowDataset(cfg, train_df)
    trainer = Trainer(cfg, ds, epochs, batch_size, base_lr, max_lr, weight_decay)

    fold_vals = []
    for end_date_str in cfg.cv_fold_end_dates:
        # 폴드마다 새 Trainer(=새 모델/옵티마이저/스케줄러/EMA)
        trainer = Trainer(cfg, ds, epochs, batch_size, base_lr, max_lr, weight_decay)

        mask_val = make_val_mask_by_week(ds, end_date_str)
        train_loader, val_loader = trainer.make_loaders_from_mask(mask_val)
        _, best_val = trainer.train_with_loaders(train_loader, val_loader)
        fold_vals.append(best_val)
        print(f"[CV] fold end={end_date_str} val={best_val:.5f}")
    cv_mean = float(np.mean(fold_vals))
    print(f"[CV] mean val={cv_mean:.5f}")
    return cv_mean


# =====================
# Optuna objective
# =====================

def run_optuna(cfg: Config):
    import optuna
    def objective(trial: optuna.trial.Trial):
        # 모델/손실/정규화 하이퍼 공간
        cfg.hidden = trial.suggest_categorical("hidden", [384, 512, 640])
        cfg.store_emb_dim = trial.suggest_categorical("store_emb_dim", [16, 32, 48])
        cfg.dropout = trial.suggest_float("dropout", 0.05, 0.2)
        cfg.zero_weight = trial.suggest_categorical("zero_weight", [0.05, 0.1, 0.2])
        cfg.hurdle_lambda = trial.suggest_categorical("hurdle_lambda", [0.2, 0.3, 0.4])
        scales_str = trial.suggest_categorical("scales", ["1,2,4,7,14","1,2,3,4,7,14","1,2,3,5,7,14"])
        cfg.scales = tuple(map(int, scales_str.split(",")))
        cfg.use_menu_embedding = trial.suggest_categorical("use_menu_embedding", [False, True])
        if cfg.use_menu_embedding:
            cfg.menu_emb_dim = trial.suggest_categorical("menu_emb_dim", [16, 24, 32])

        # 최적화 관련(고정/경계)
        epochs = cfg.EPOCHS_TUNE
        batch_size = cfg.BATCH_TUNE
        base_lr = cfg.BASE_LR_TUNE
        max_lr = cfg.MAX_LR_TUNE
        weight_decay = cfg.WD_TUNE

        val = evaluate_cfg_rolling(cfg, epochs, batch_size, base_lr, max_lr, weight_decay)
        return val

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.N_TRIALS)
    print("[Optuna] Best value:", study.best_value)
    print("[Optuna] Best params:", study.best_trial.params)
    best = study.best_trial.params
    # 최적 파라미터 적용
    cfg.hidden = best.get("hidden", cfg.hidden)
    cfg.store_emb_dim = best.get("store_emb_dim", cfg.store_emb_dim)
    cfg.dropout = best.get("dropout", cfg.dropout)
    cfg.zero_weight = best.get("zero_weight", cfg.zero_weight)
    cfg.hurdle_lambda = best.get("hurdle_lambda", cfg.hurdle_lambda)
    cfg.scales = best.get("scales", cfg.scales)
    cfg.use_menu_embedding = best.get("use_menu_embedding", cfg.use_menu_embedding)
    cfg.menu_emb_dim = best.get("menu_emb_dim", cfg.menu_emb_dim)


# =====================
# Prediction utils
# =====================
@torch.no_grad()
def predict_one_file(cfg: Config, model: NHITS_Hurdle, test_df: pd.DataFrame, store2idx: Dict[str,int], menu2idx: Dict[str,int]) -> pd.DataFrame:
    device = torch.device(cfg.device)
    tdf = test_df.copy()
    tdf[cfg.date_col] = pd.to_datetime(tdf[cfg.date_col])
    tdf[cfg.target_col] = tdf[cfg.target_col].clip(lower=0)
    pivot = tdf.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index().fillna(0.0)
    items = list(pivot.columns); dates = list(pivot.index)
    assert len(dates) >= cfg.in_len
    values = pivot.values.astype(np.float32)
    last_date = dates[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, cfg.out_len + 1)]

    holidays_set = set(pd.to_datetime(cfg.custom_holidays_list))
    past_cal = build_calendar_features(dates[-cfg.in_len:], holidays_set).drop(columns=["date"]).values.astype(np.float32)
    fut_cal = build_calendar_features(future_dates, holidays_set).drop(columns=["date"]).values.astype(np.float32)

    B = len(items); Lx = cfg.in_len
    x = values[-Lx:, :].T
    if cfg.log1p: x = np.log1p(x)
    x = torch.from_numpy(x).float().to(device)
    past_cal_b = torch.from_numpy(np.repeat(past_cal[None, :, :], B, axis=0)).float().to(device)
    fut_cal_b  = torch.from_numpy(np.repeat(fut_cal[None, :, :], B, axis=0)).float().to(device)

    stores = [parse_store_name(it) for it in items]
    menus  = [parse_menu_name(it) for it in items]
    store_idx = torch.tensor([store2idx.get(s, 0) for s in stores], dtype=torch.long, device=device)
    menu_idx  = torch.tensor([menu2idx.get(m, 0) for m in menus], dtype=torch.long, device=device)

    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (cfg.use_amp and torch.cuda.is_available()) else torch.cuda.amp.autocast(enabled=False)
    model.eval()
    with amp_ctx:
        v_pred, p_logits = model(x, past_cal_b, fut_cal_b, store_idx, menu_idx)
        y_val = torch.expm1(v_pred).clamp_min(0.0)
        y_prob = torch.sigmoid(p_logits)
        y_hat = (y_prob * y_val).clamp_min(0.0).cpu().numpy()
    # 업장 후처리 배수 보정 (옵션)
    if cfg.apply_postprocess and cfg.store_post_scales:
        stores = [parse_store_name(it) for it in items]
        for i, s in enumerate(stores):
            mul = float(cfg.store_post_scales.get(s, 1.0))
            if abs(mul - 1.0) > 1e-8:
                y_hat[i, :] *= mul
    return pd.DataFrame(y_hat, index=items, columns=[f"D+{i}" for i in range(1, cfg.out_len+1)]).T


# =====================
# Main: Optuna + Full Train + Submit
# =====================
if __name__ == "__main__":
    cfg = Config()
    if cfg.store_weights is None:
        cfg.store_weights = DEFAULT_STORE_WEIGHTS
    if cfg.custom_holidays_list is None:
        cfg.custom_holidays_list = DEFAULT_CUSTOM_HOLIDAYS
    if cfg.store_post_scales is None:
        # 예: 담하/미라시아에 3~7% 상향 보정하고 싶을 때 1.03~1.07로 설정
        cfg.store_post_scales = {"담하": 1.00, "미라시아": 1.00}
    set_seed(cfg.seed)

    # ---- 튜닝 (Rolling-CV) ----
    if cfg.USE_OPTUNA:
        run_optuna(cfg)

    # ---- 전체 학습 ----
    train_df = pd.read_csv(cfg.train_csv)
    ds = SlidingWindowDataset(cfg, train_df)
    trainer = Trainer(cfg, ds, cfg.EPOCHS_FULL, cfg.BATCH_FULL, cfg.BASE_LR_FULL, cfg.MAX_LR_FULL, cfg.WD_FULL)
    # 전체 학습은 최신 주(2024-06-08~06-14)를 검증으로 사용
    mask_val = make_val_mask_by_week(ds, cfg.cv_fold_end_dates[0])
    train_loader, val_loader = trainer.make_loaders_from_mask(mask_val)
    model, _ = trainer.train_with_loaders(train_loader, val_loader)

    # 저장
    os.makedirs(os.path.dirname(cfg.out_submission_csv), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg.__dict__,
        "store2idx": ds.store2idx,
        "menu2idx": ds.menu2idx,
    }, "./data/nhits_hurdle_model.pth")
    print("[Save] ./data/nhits_hurdle_model.pth")

    # ---- 추론/제출 ----
    import glob
    test_files = sorted(glob.glob(cfg.test_glob))
    sub_template = pd.read_csv(cfg.submission_template_csv)
    all_preds = []
    for test_idx, test_file in enumerate(test_files):
        tdf = pd.read_csv(test_file)
        submit_block = predict_one_file(cfg, model, tdf, ds.store2idx, ds.menu2idx)
        submit_block.index = [f"TEST_{test_idx:02d}+{k}일" for k in range(1, cfg.out_len+1)]
        all_preds.append(submit_block)
    final_submit = pd.concat(all_preds, axis=0)
    final_submit.reset_index(inplace=True)
    final_submit.rename(columns={"index": "영업일자"}, inplace=True)
    final_submit = final_submit.reindex(columns=sub_template.columns, fill_value=0)
    # 수치 반올림 및 음수 방지 → 정수 제출
    num_cols = [c for c in final_submit.columns if c != "영업일자"]
    final_submit[num_cols] = np.rint(np.clip(final_submit[num_cols].values, a_min=0, a_max=None)).astype(int)
    final_submit.to_csv(cfg.out_submission_csv, index=False, encoding="utf-8-sig")
    print(f"[Submit] saved → {cfg.out_submission_csv}")
