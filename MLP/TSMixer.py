# -*- coding: utf-8 -*-
"""
TSMixer Model for Resort Sales Prediction
- 친구 N-HiTS 코드 기반으로 TSMixer 구현
- MLP + Mixing 구조로 0값에 robust한 성능
- 동일한 Loss 함수(HurdleLoss), Feature Engineering 적용
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
import torch.nn.functional as F

# =====================
# Config (N-HiTS 기반 + TSMixer 파라미터)
# =====================
@dataclass
class TSMixerConfig:
    # 경로 (동일)
    train_csv: str = "./data/train/train.csv"
    test_glob: str = "./data/test/*.csv"
    submission_template_csv: str = "./data/sample_submission.csv"
    out_submission_csv: str = "./data/tsmixer_submission.csv"

    # 컬럼명 (동일)
    date_col: str = "영업일자"
    item_col: str = "영업장명_메뉴명"
    target_col: str = "매출수량"

    # 윈도우 (동일)
    in_len: int = 28
    out_len: int = 7

    # 학습 데이터 컷오프 (동일)
    train_end_date: str = "2024-06-15"

    # 공통 학습 (동일)
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log1p: bool = True

    # 전체 학습
    EPOCHS_FULL: int = 120
    BATCH_FULL: int = 512  # TSMixer는 메모리 효율적
    BASE_LR_FULL: float = 5e-4
    MAX_LR_FULL: float = 1.5e-3
    WD_FULL: float = 5e-4

    # 튜닝
    USE_OPTUNA: bool = True
    N_TRIALS: int = 30
    EPOCHS_TUNE: int = 30
    BATCH_TUNE: int = 512
    BASE_LR_TUNE: float = 1e-3
    MAX_LR_TUNE: float = 2e-3
    WD_TUNE: float = 3e-4

    # Rolling-CV (동일)
    cv_fold_end_dates: Tuple[str, str, str] = ("2024-06-14", "2024-06-07", "2024-05-31")

    # DataLoader (동일)
    num_workers: int = 12
    pin_memory: bool = True
    persistent_workers: bool = True

    # TSMixer 전용 파라미터
    hidden_dim: int = 512     # MLP 은닉층 차원
    n_mixer_layers: int = 4   # Mixer 레이어 수
    expansion_factor: int = 2 # FFN 확장 비율
    dropout: float = 0.15
    norm_type: str = "layer"  # "layer", "batch", "none"
    activation: str = "gelu"  # "relu", "gelu", "swish"
    
    # 임베딩 (동일)
    store_emb_dim: int = 32
    use_menu_embedding: bool = False
    menu_emb_dim: int = 16

    # 허들/손실 (동일)
    eps_smape: float = 0.1
    zero_weight: float = 0.1
    hurdle_lambda: float = 0.3

    # AMP/compile/EMA (동일)
    use_amp: bool = False
    use_compile: bool = False
    ema_decay: float = 0.999

    # 가중치/공휴일 (동일)
    store_weights: Dict[str, float] = None
    custom_holidays_list: List[str] = None

    # 후처리 (동일)
    apply_postprocess: bool = False
    store_post_scales: Dict[str, float] = None


# 기본값들 (동일)
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
# Utils / Enhanced Features (PatchTST와 동일)
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
    w = ((week - 1) % 52) + 1
    return math.sin(2 * math.pi * w / 52.0), math.cos(2 * math.pi * w / 52.0)

def season_onehot(m: int):
    if m in [4,5,6]: return [1,0,0,0]  # 봄
    if m in [7,8]:   return [0,1,0,0]  # 여름
    if m in [9,10,11]: return [0,0,1,0]  # 가을
    return [0,0,0,1]  # 겨울

def build_enhanced_calendar_features(dates: List[pd.Timestamp], holidays_set: set) -> pd.DataFrame:
    """강화된 캘린더 Feature - 교차 Feature 포함"""
    df = pd.DataFrame({"date": dates})
    df["dow"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].apply(lambda d: int(d.isocalendar().week))
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
    df["is_holiday"] = df["date"].isin(holidays_set).astype(int)
    df["is_off"] = ((df["is_weekend"]==1) | (df["is_holiday"]==1)).astype(int)

    # 기본 sin/cos 인코딩
    df[["month_sin","month_cos"]] = df["month"].apply(lambda m: pd.Series(sine_cosine_month(m)))
    df[["doy_sin","doy_cos"]] = df["date"].apply(lambda d: pd.Series(sine_cosine_day_of_year(d)))
    df[["week_sin","week_cos"]] = df["week"].apply(lambda w: pd.Series(sine_cosine_week(w)))
    df[["spring","summer","autumn","winter"]] = df["month"].apply(lambda m: pd.Series(season_onehot(m)))

    # 내일 휴일 여부
    df["tomorrow"] = df["date"] + pd.Timedelta(days=1)
    df["is_holiday_eve"] = ((df["tomorrow"].isin(holidays_set)) | (df["tomorrow"].dt.weekday.isin([5,6]))).astype(int)
    df["tomm_is_off"] = df["is_holiday_eve"].astype(int)
    
    # 추가 Feature
    df["month_end"] = (df["date"].dt.day >= df["date"].dt.days_in_month - 2).astype(int)
    df["month_start"] = (df["date"].dt.day <= 3).astype(int)
    
    # 연휴 길이
    df["long_holiday"] = 0
    for i in range(len(df)):
        if df.iloc[i]["is_off"] == 1:
            consecutive = 1
            for j in range(i+1, min(i+7, len(df))):
                if df.iloc[j]["is_off"] == 1:
                    consecutive += 1
                else:
                    break
            if consecutive >= 3:
                df.iloc[i, df.columns.get_loc("long_holiday")] = 1
    
    return df.drop(columns=["tomorrow"])

def parse_store_name(item_full: str) -> str:
    return item_full.split("_")[0]

def parse_menu_name(item_full: str) -> str:
    return item_full.split("_", 1)[1] if "_" in item_full else item_full

# =====================
# Enhanced Dataset (PatchTST와 동일)
# =====================
class TSMixerDataset(Dataset):
    def __init__(self, cfg: TSMixerConfig, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df.copy()
        self.df[cfg.date_col] = pd.to_datetime(self.df[cfg.date_col])
        self.df[cfg.target_col] = self.df[cfg.target_col].clip(lower=0)

        # 피벗 테이블 생성
        pivot = self.df.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index()
        self.items = list(pivot.columns)
        self.dates = list(pivot.index)
        self.values = pivot.fillna(0.0).values.astype(np.float32)

        # 강화된 캘린더 Feature
        holidays_set = set(pd.to_datetime(cfg.custom_holidays_list))
        caldf = build_enhanced_calendar_features(self.dates, holidays_set)
        self.cal_feats = caldf.drop(columns=["date"]).values.astype(np.float32)
        self.cal_feat_names = [c for c in caldf.columns if c != "date"]

        # 영업장/메뉴 인덱싱
        stores = [parse_store_name(it) for it in self.items]
        menus = [parse_menu_name(it) for it in self.items]
        self.store2idx = {s: i for i, s in enumerate(sorted(set(stores)))}
        self.item_store_idx = np.array([self.store2idx[s] for s in stores], dtype=np.int64)
        self.n_stores = len(self.store2idx)

        self.menu2idx = {m: i for i, m in enumerate(sorted(set(menus)))}
        self.item_menu_idx = np.array([self.menu2idx[m] for m in menus], dtype=np.int64)
        self.n_menus = len(self.menu2idx)

        # 가중치
        sw = cfg.store_weights
        self.sample_weights = np.array([sw.get(parse_store_name(it), 1.0) for it in self.items], dtype=np.float32)

        # 윈도우 인덱스 생성
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

    def _get_dynamic_cross_features(self, item_idx: int, past_cal: np.ndarray, fut_cal: np.ndarray):
        """동적 교차 Feature 계산"""
        store = parse_store_name(self.items[item_idx])
        
        # 계절 정보
        season_cols = [i for i, name in enumerate(self.cal_feat_names) if name in ['spring', 'summer', 'autumn', 'winter']]
        avg_season = past_cal[:, season_cols].mean(axis=0)
        fut_season = fut_cal[:, season_cols].mean(axis=0)
        
        # 영업장×계절 교차 Feature
        store_season_cross = self._encode_store_season(store, avg_season)
        fut_store_season_cross = self._encode_store_season(store, fut_season)
        
        return np.array(store_season_cross, dtype=np.float32), np.array(fut_store_season_cross, dtype=np.float32)

    def _encode_store_season(self, store: str, season_vec: np.ndarray):
        """영업장×계절 교차 인코딩"""
        store_season_map = {
            "담하": [1.2, 0.8, 1.0, 0.9],
            "미라시아": [1.1, 1.3, 1.0, 0.8],
            "라그로타": [1.0, 1.2, 1.1, 0.9],
            "느티나무 셀프BBQ": [0.9, 1.4, 1.0, 0.7],
            "연회장": [1.0, 1.0, 1.0, 1.0],
            "카페테리아": [1.0, 1.0, 1.0, 1.0],
            "포레스트릿": [0.8, 1.2, 1.0, 0.8],
            "화담숲주막": [1.1, 1.2, 1.1, 0.8],
            "화담숲카페": [1.0, 1.1, 1.0, 0.9],
        }
        
        multiplier = store_season_map.get(store, [1.0, 1.0, 1.0, 1.0])
        return (season_vec * multiplier).tolist()

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

        # 동적 교차 Feature
        past_cross, fut_cross = self._get_dynamic_cross_features(j, past_cal, fut_cal)
        
        # 변동성 Feature
        if len(x) >= 7:
            recent_x = x[-7:]
            cv = np.std(recent_x) / (np.mean(recent_x) + 1e-8)
            max_min_ratio = (np.max(recent_x) + 1e-8) / (np.min(recent_x) + 1e-8)
            volatility_feats = np.array([cv, max_min_ratio], dtype=np.float32)
        else:
            volatility_feats = np.array([0.0, 1.0], dtype=np.float32)

        store_idx = self.item_store_idx[j]
        menu_idx = self.item_menu_idx[j]
        sample_w = self.sample_weights[j]
        zero_mask = (y == 0).astype(np.float32)
        pos_mask = (y > 0).astype(np.float32)

        return {
            "x": torch.from_numpy(x_in).float(),
            "y": torch.from_numpy(y_out).float(),
            "past_cal": torch.from_numpy(past_cal).float(),
            "fut_cal": torch.from_numpy(fut_cal).float(),
            "past_cross": torch.from_numpy(past_cross).float(),
            "fut_cross": torch.from_numpy(fut_cross).float(), 
            "volatility": torch.from_numpy(volatility_feats).float(),
            "store_idx": torch.tensor(store_idx, dtype=torch.long),
            "menu_idx": torch.tensor(menu_idx, dtype=torch.long),
            "sample_w": torch.tensor(sample_w, dtype=torch.float32),
            "zero_mask": torch.from_numpy(zero_mask).float(),
            "pos_mask": torch.from_numpy(pos_mask).float(),
        }


# =====================
# TSMixer Model
# =====================
class MixerBlock(nn.Module):
    """TSMixer의 핵심 Mixing Block"""
    def __init__(self, seq_len: int, hidden_dim: int, expansion_factor: int, 
                 dropout: float, norm_type: str, activation: str):
        super().__init__()
        
        # Time Mixing (시간 축 믹싱) - 수치 안정화
        self.time_mix = nn.Sequential(
            nn.Linear(seq_len, seq_len * expansion_factor),
            nn.LayerNorm(seq_len * expansion_factor),  # 중간에 정규화 추가
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(seq_len * expansion_factor, seq_len),
            nn.LayerNorm(seq_len)  # 출력도 정규화
        )
        
        # Feature Mixing (특성 축 믹싱) - 수치 안정화
        self.feat_mix = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.LayerNorm(hidden_dim * expansion_factor),  # 중간에 정규화 추가
            self._get_activation(activation), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
            nn.LayerNorm(hidden_dim)  # 출력도 정규화
        )
        
        # Normalization (둘 다 hidden_dim으로 통일)
        self.norm1 = self._get_norm(norm_type, hidden_dim)  # seq_len → hidden_dim
        self.norm2 = self._get_norm(norm_type, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def _get_activation(self, activation: str):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.ReLU()
        elif activation == "swish":
            return nn.ReLU()
        else:
            return nn.ReLU()
    
    def _get_norm(self, norm_type: str, dim: int):
        if norm_type == "layer":
            return nn.LayerNorm(dim)  # dim은 항상 hidden_dim이어야 함
        elif norm_type == "batch":
            return nn.LayerNorm(dim)  # LayerNorm 강제 사용
        else:
            return nn.Identity()
    
    def forward(self, x: torch.Tensor):
        # x: [B, seq_len, hidden_dim]
        B, L, D = x.shape
        
        # 🔥 Time Mixing: 시간 축에서 정보 교환 (차원 올바르게 처리)
        x_t = x.transpose(1, 2)  # [B, hidden_dim, seq_len]
        time_mixed = self.time_mix(x_t)  # [B, hidden_dim, seq_len]
        time_mixed = time_mixed.transpose(1, 2)  # [B, seq_len, hidden_dim]
        
        # norm1은 hidden_dim(마지막 차원)을 정규화해야 함
        if isinstance(self.norm1, nn.LayerNorm):
            x_t_norm = self.norm1(time_mixed)  # LayerNorm은 마지막 차원 정규화
        else:
            x_t_norm = time_mixed  # Identity인 경우
            
        x = x + self.dropout(x_t_norm)
        
        # 🔥 Feature Mixing: 특성 축에서 정보 교환
        feat_mixed = self.feat_mix(x)  # [B, seq_len, hidden_dim]
        
        if isinstance(self.norm2, nn.LayerNorm):
            x_f_norm = self.norm2(feat_mixed)  # LayerNorm은 마지막 차원 정규화
        else:
            x_f_norm = feat_mixed  # Identity인 경우
            
        x = x + self.dropout(x_f_norm)
        
        return x


class TSMixer_Hurdle(nn.Module):
    def __init__(self, cfg: TSMixerConfig, cal_dim: int, n_stores: int, n_menus: int):
        super().__init__()
        self.cfg = cfg
        
        # Feature 임베딩
        self.store_emb = nn.Embedding(n_stores, cfg.store_emb_dim)
        self.menu_emb = nn.Embedding(n_menus, cfg.menu_emb_dim) if cfg.use_menu_embedding else None
        
        # 전체 Feature 차원 계산
        emb_dim = cfg.store_emb_dim + (cfg.menu_emb_dim if cfg.use_menu_embedding else 0)
        cross_dim = 4  # 영업장×계절 교차 Feature
        volatility_dim = 2  # CV, max_min_ratio
        total_feat_dim = cal_dim + emb_dim + cross_dim + volatility_dim
        
        # 입력 프로젝션: 시계열 + Feature를 hidden_dim으로
        self.input_proj = nn.Linear(1 + total_feat_dim, cfg.hidden_dim)
        
        # TSMixer Blocks
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(cfg.in_len, cfg.hidden_dim, cfg.expansion_factor, 
                      cfg.dropout, cfg.norm_type, cfg.activation)
            for _ in range(cfg.n_mixer_layers)
        ])
        
        # 출력 헤드들 (Hurdle)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim * cfg.in_len, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, cfg.out_len)
        )
        
        self.prob_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim * cfg.in_len, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, cfg.out_len)
        )
        
        # 미래 캘린더 정보 활용
        self.future_proj_v = nn.Sequential(
            nn.Linear(cal_dim + cross_dim, cfg.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 4, cfg.out_len)
        )
        
        self.future_proj_p = nn.Sequential(
            nn.Linear(cal_dim + cross_dim, cfg.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 4, cfg.out_len)
        )

    def forward(self, x, past_cal, fut_cal, past_cross, fut_cross, volatility, store_idx, menu_idx):
        B, L = x.shape
        
        # 1. Feature 임베딩
        store_e = self.store_emb(store_idx)  # [B, store_emb_dim]
        if self.menu_emb is not None:
            menu_e = self.menu_emb(menu_idx)  # [B, menu_emb_dim]
            emb_feat = torch.cat([store_e, menu_e], dim=-1)
        else:
            emb_feat = store_e
        
        # 2. 과거 캘린더/교차/변동성 Feature 결합
        past_cal_avg = past_cal.mean(dim=1)  # [B, cal_dim]
        combined_feat = torch.cat([past_cal_avg, emb_feat, past_cross, volatility], dim=-1)
        
        # 3. 🔥 TSMixer 핵심: 시계열과 Feature를 각 시점에 결합
        # x: [B, L] -> [B, L, 1], combined_feat를 각 시점에 브로드캐스팅
        x_expanded = x.unsqueeze(-1)  # [B, L, 1]
        feat_expanded = combined_feat.unsqueeze(1).expand(-1, L, -1)  # [B, L, feat_dim]
        
        # 시계열과 Feature 결합
        input_combined = torch.cat([x_expanded, feat_expanded], dim=-1)  # [B, L, 1+feat_dim]
        
        # 4. 입력 프로젝션
        x_proj = self.input_proj(input_combined)  # [B, L, hidden_dim]
        
        # 5. 🔥 TSMixer Blocks: Time & Feature Mixing
        for mixer in self.mixer_blocks:
            x_proj = mixer(x_proj)  # [B, L, hidden_dim]
        
        # 6. 출력을 위한 Flatten
        x_flat = x_proj.reshape(B, -1)  # [B, L*hidden_dim]
        
        # 7. 출력 생성
        value_pred = self.value_head(x_flat)  # [B, out_len]
        prob_pred = self.prob_head(x_flat)    # [B, out_len]
        
        # 8. 미래 정보 활용
        fut_cal_avg = fut_cal.mean(dim=1)  # [B, cal_dim]
        fut_combined = torch.cat([fut_cal_avg, fut_cross], dim=-1)
        
        value_fut = self.future_proj_v(fut_combined)
        prob_fut = self.future_proj_p(fut_combined)
        
        value_pred = value_pred + value_fut
        prob_pred = prob_pred + prob_fut
        
        return value_pred, prob_pred


# =====================
# Loss / EMA (동일)
# =====================
class HurdleLoss(nn.Module):
    def __init__(self, eps: float = 0.1, zero_weight: float = 0.1, lambda_bce: float = 0.3):
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
        denom = (torch.abs(yp_val) + torch.abs(yt_val)).clamp_min(self.eps)
        smape_pos = 2.0 * torch.abs(yp_val - yt_val) / denom
        smape_pos = smape_pos * pos_mask

        p = torch.sigmoid(p_logits)
        y_hat = p * yp_val
        denom2 = (torch.abs(y_hat) + torch.abs(yt_val)).clamp_min(self.eps)
        smape_all = 2.0 * torch.abs(y_hat - yt_val) / denom2
        w_zero = torch.where(yt_val <= 0.0 + 1e-8, torch.full_like(yt_val, self.zero_weight), torch.ones_like(yt_val))
        smape_all = smape_all * w_zero

        bce_s = bce.mean(dim=1)
        pos_s = smape_pos.mean(dim=1)
        all_s = smape_all.mean(dim=1)
        sample_loss = self.lambda_bce * bce_s + pos_s + 0.2 * all_s

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
# Trainer
# =====================
class TSMixerTrainer:
    def __init__(self, cfg: TSMixerConfig, dataset: TSMixerDataset, epochs: int, batch_size: int, 
                 base_lr: float, max_lr: float, weight_decay: float):
        self.cfg = cfg
        self.dataset = dataset
        self.device = torch.device(cfg.device)
        
        cal_dim = dataset.cal_feats.shape[1]
        model = TSMixer_Hurdle(cfg, cal_dim, dataset.n_stores, dataset.n_menus).to(self.device)
        
        if cfg.use_compile and torch.cuda.is_available():
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception as e:
                print("torch.compile 실패 → 비컴파일로 진행:", e)
        
        self.model = model
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
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                past_cal = batch["past_cal"].to(self.device)
                fut_cal = batch["fut_cal"].to(self.device)
                past_cross = batch["past_cross"].to(self.device)
                fut_cross = batch["fut_cross"].to(self.device)
                volatility = batch["volatility"].to(self.device)
                store_idx = batch["store_idx"].to(self.device)
                menu_idx = batch["menu_idx"].to(self.device)
                sample_w = batch["sample_w"].to(self.device)
                pos_mask = batch["pos_mask"].to(self.device)
                
                v_pred, p_logits = self.model(x, past_cal, fut_cal, past_cross, fut_cross, 
                                              volatility, store_idx, menu_idx)
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
        # 옵티마이저/스케줄러 초기화
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.optim, max_lr=self.max_lr, epochs=self.epochs, steps_per_epoch=len(train_loader),
            pct_start=0.1, div_factor=self.max_lr / self.base_lr
        )
        
        best_val = float("inf")
        best_state = None
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (self.cfg.use_amp and torch.cuda.is_available()) else torch.cuda.amp.autocast(enabled=False)
            
            for batch in train_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                past_cal = batch["past_cal"].to(self.device)
                fut_cal = batch["fut_cal"].to(self.device)
                past_cross = batch["past_cross"].to(self.device)
                fut_cross = batch["fut_cross"].to(self.device)
                volatility = batch["volatility"].to(self.device)
                store_idx = batch["store_idx"].to(self.device)
                menu_idx = batch["menu_idx"].to(self.device)
                sample_w = batch["sample_w"].to(self.device)
                pos_mask = batch["pos_mask"].to(self.device)
                
                with amp_ctx:
                    v_pred, p_logits = self.model(x, past_cal, fut_cal, past_cross, fut_cross,
                                                  volatility, store_idx, menu_idx)
                    loss, _, _ = self.criterion(v_pred, p_logits, y, pos_mask, sample_w)
                
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                self.sched.step()
                self.ema.update(self.model)
            
            val_loss = self.evaluate(val_loader, use_ema=True)
            print(f"[TSMixer Epoch {epoch:03d}] val_loss: {val_loss:.5f}")
            
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return self.model, best_val


# =====================
# Rolling-CV & Optuna
# =====================
def make_val_mask_by_week(dataset: TSMixerDataset, end_date_str: str) -> np.ndarray:
    end_date = pd.to_datetime(end_date_str)
    start_date = end_date - pd.Timedelta(days=6)
    ted = dataset.target_end_dates
    return (ted >= start_date) & (ted <= end_date)


def evaluate_cfg_rolling(cfg: TSMixerConfig, epochs: int, batch_size: int, base_lr: float, max_lr: float, weight_decay: float) -> float:
    train_df = pd.read_csv(cfg.train_csv)
    ds = TSMixerDataset(cfg, train_df)
    
    fold_vals = []
    for end_date_str in cfg.cv_fold_end_dates:
        trainer = TSMixerTrainer(cfg, ds, epochs, batch_size, base_lr, max_lr, weight_decay)
        mask_val = make_val_mask_by_week(ds, end_date_str)
        train_loader, val_loader = trainer.make_loaders_from_mask(mask_val)
        _, best_val = trainer.train_with_loaders(train_loader, val_loader)
        fold_vals.append(best_val)
        print(f"[TSMixer CV] fold end={end_date_str} val={best_val:.5f}")
    
    cv_mean = float(np.mean(fold_vals))
    print(f"[TSMixer CV] mean val={cv_mean:.5f}")
    return cv_mean


def run_optuna(cfg: TSMixerConfig):
    import optuna
    
    def objective(trial: optuna.trial.Trial):
        # TSMixer 하이퍼파라미터
        cfg.hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768])
        cfg.n_mixer_layers = trial.suggest_categorical("n_mixer_layers", [2, 4, 6, 8])
        cfg.expansion_factor = trial.suggest_categorical("expansion_factor", [1, 2])
        cfg.dropout = trial.suggest_float("dropout", 0.05, 0.25)
        cfg.norm_type = trial.suggest_categorical("norm_type", ["layer", "none"])
        cfg.activation = trial.suggest_categorical("activation", ["relu", "gelu", "swish"])
        
        # 공통 하이퍼파라미터
        cfg.store_emb_dim = trial.suggest_categorical("store_emb_dim", [16, 32, 48])
        cfg.zero_weight = trial.suggest_categorical("zero_weight", [0.05, 0.1, 0.2])
        cfg.hurdle_lambda = trial.suggest_categorical("hurdle_lambda", [0.2, 0.3, 0.4])
        cfg.use_menu_embedding = trial.suggest_categorical("use_menu_embedding", [False, True])
        if cfg.use_menu_embedding:
            cfg.menu_emb_dim = trial.suggest_categorical("menu_emb_dim", [16, 24, 32])

        val = evaluate_cfg_rolling(cfg, cfg.EPOCHS_TUNE, cfg.BATCH_TUNE, 
                                   cfg.BASE_LR_TUNE, cfg.MAX_LR_TUNE, cfg.WD_TUNE)
        return val

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.N_TRIALS)
    print("[TSMixer Optuna] Best value:", study.best_value)
    print("[TSMixer Optuna] Best params:", study.best_trial.params)
    
    # 최적 파라미터 적용
    best = study.best_trial.params
    cfg.hidden_dim = best.get("hidden_dim", cfg.hidden_dim)
    cfg.n_mixer_layers = best.get("n_mixer_layers", cfg.n_mixer_layers)
    cfg.expansion_factor = best.get("expansion_factor", cfg.expansion_factor)
    cfg.dropout = best.get("dropout", cfg.dropout)
    cfg.norm_type = best.get("norm_type", cfg.norm_type)
    cfg.activation = best.get("activation", cfg.activation)
    cfg.store_emb_dim = best.get("store_emb_dim", cfg.store_emb_dim)
    cfg.zero_weight = best.get("zero_weight", cfg.zero_weight)
    cfg.hurdle_lambda = best.get("hurdle_lambda", cfg.hurdle_lambda)
    cfg.use_menu_embedding = best.get("use_menu_embedding", cfg.use_menu_embedding)
    cfg.menu_emb_dim = best.get("menu_emb_dim", cfg.menu_emb_dim)


# =====================
# Prediction
# =====================
@torch.no_grad()
def predict_one_file(cfg: TSMixerConfig, model: TSMixer_Hurdle, test_df: pd.DataFrame, 
                     store2idx: Dict[str,int], menu2idx: Dict[str,int]) -> pd.DataFrame:
    device = torch.device(cfg.device)
    tdf = test_df.copy()
    tdf[cfg.date_col] = pd.to_datetime(tdf[cfg.date_col])
    tdf[cfg.target_col] = tdf[cfg.target_col].clip(lower=0)
    
    pivot = tdf.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index().fillna(0.0)
    items = list(pivot.columns)
    dates = list(pivot.index)
    assert len(dates) >= cfg.in_len
    values = pivot.values.astype(np.float32)
    
    last_date = dates[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, cfg.out_len + 1)]
    
    # 캘린더 Feature
    holidays_set = set(pd.to_datetime(cfg.custom_holidays_list))
    past_cal = build_enhanced_calendar_features(dates[-cfg.in_len:], holidays_set).drop(columns=["date"]).values.astype(np.float32)
    fut_cal = build_enhanced_calendar_features(future_dates, holidays_set).drop(columns=["date"]).values.astype(np.float32)
    
    B = len(items)
    Lx = cfg.in_len
    x = values[-Lx:, :].T  # [B, Lx]
    
    if cfg.log1p: 
        x = np.log1p(x)
    
    x = torch.from_numpy(x).float().to(device)
    past_cal_b = torch.from_numpy(np.repeat(past_cal[None, :, :], B, axis=0)).float().to(device)
    fut_cal_b = torch.from_numpy(np.repeat(fut_cal[None, :, :], B, axis=0)).float().to(device)
    
    # 교차/변동성 Feature 생성
    past_cross_list = []
    fut_cross_list = []
    volatility_list = []
    
    for i, item in enumerate(items):
        store = parse_store_name(item)
        
        # 계절 Feature 추출 (마지막 4개 컬럼이 계절)
        past_season = past_cal[:, -4:].mean(axis=0)
        fut_season = fut_cal[:, -4:].mean(axis=0)
        
        # 영업장×계절 교차
        store_season_map = {
            "담하": [1.2, 0.8, 1.0, 0.9], "미라시아": [1.1, 1.3, 1.0, 0.8],
            "라그로타": [1.0, 1.2, 1.1, 0.9], "느티나무 셀프BBQ": [0.9, 1.4, 1.0, 0.7],
            "연회장": [1.0, 1.0, 1.0, 1.0], "카페테리아": [1.0, 1.0, 1.0, 1.0],
            "포레스트릿": [0.8, 1.2, 1.0, 0.8], "화담숲주막": [1.1, 1.2, 1.1, 0.8],
            "화담숲카페": [1.0, 1.1, 1.0, 0.9]
        }
        multiplier = store_season_map.get(store, [1.0, 1.0, 1.0, 1.0])
        past_cross = (past_season * np.array(multiplier)).tolist()
        fut_cross = (fut_season * np.array(multiplier)).tolist()
        
        # 변동성 Feature
        recent_x = values[-7:, i] if len(values) >= 7 else values[:, i]
        cv = np.std(recent_x) / (np.mean(recent_x) + 1e-8)
        max_min_ratio = (np.max(recent_x) + 1e-8) / (np.min(recent_x) + 1e-8)
        
        past_cross_list.append(past_cross)
        fut_cross_list.append(fut_cross)
        volatility_list.append([cv, max_min_ratio])
    
    past_cross_b = torch.tensor(past_cross_list, dtype=torch.float32, device=device)
    fut_cross_b = torch.tensor(fut_cross_list, dtype=torch.float32, device=device)
    volatility_b = torch.tensor(volatility_list, dtype=torch.float32, device=device)
    
    # 영업장/메뉴 인덱스
    stores = [parse_store_name(it) for it in items]
    menus = [parse_menu_name(it) for it in items]
    store_idx = torch.tensor([store2idx.get(s, 0) for s in stores], dtype=torch.long, device=device)
    menu_idx = torch.tensor([menu2idx.get(m, 0) for m in menus], dtype=torch.long, device=device)
    
    # 예측
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (cfg.use_amp and torch.cuda.is_available()) else torch.cuda.amp.autocast(enabled=False)
    model.eval()
    
    with amp_ctx:
        v_pred, p_logits = model(x, past_cal_b, fut_cal_b, past_cross_b, fut_cross_b,
                                volatility_b, store_idx, menu_idx)
        y_val = torch.expm1(v_pred).clamp_min(0.0)
        y_prob = torch.sigmoid(p_logits)
        y_hat = (y_prob * y_val).clamp_min(0.0).cpu().numpy()
    
    # 후처리
    if cfg.apply_postprocess and cfg.store_post_scales:
        for i, s in enumerate(stores):
            mul = float(cfg.store_post_scales.get(s, 1.0))
            if abs(mul - 1.0) > 1e-8:
                y_hat[i, :] *= mul
    
    return pd.DataFrame(y_hat, index=items, columns=[f"D+{i}" for i in range(1, cfg.out_len+1)]).T


# =====================
# Main
# =====================
if __name__ == "__main__":
    cfg = TSMixerConfig()
    if cfg.store_weights is None:
        cfg.store_weights = DEFAULT_STORE_WEIGHTS
    if cfg.custom_holidays_list is None:
        cfg.custom_holidays_list = DEFAULT_CUSTOM_HOLIDAYS
    if cfg.store_post_scales is None:
        cfg.store_post_scales = {"담하": 1.00, "미라시아": 1.00}
    
    set_seed(cfg.seed)

    # 튜닝
    if cfg.USE_OPTUNA:
        run_optuna(cfg)

    # 전체 학습
    train_df = pd.read_csv(cfg.train_csv)
    ds = TSMixerDataset(cfg, train_df)
    trainer = TSMixerTrainer(cfg, ds, cfg.EPOCHS_FULL, cfg.BATCH_FULL, 
                             cfg.BASE_LR_FULL, cfg.MAX_LR_FULL, cfg.WD_FULL)
    
    mask_val = make_val_mask_by_week(ds, cfg.cv_fold_end_dates[0])
    train_loader, val_loader = trainer.make_loaders_from_mask(mask_val)
    model, _ = trainer.train_with_loaders(train_loader, val_loader)

    # 모델 저장
    os.makedirs(os.path.dirname(cfg.out_submission_csv), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg.__dict__,
        "store2idx": ds.store2idx,
        "menu2idx": ds.menu2idx,
    }, "./data/tsmixer_model.pth")
    print("[Save] ./data/tsmixer_model.pth")

    # 추론
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
    
    # 후처리
    num_cols = [c for c in final_submit.columns if c != "영업일자"]
    final_submit[num_cols] = np.rint(np.clip(final_submit[num_cols].values, a_min=0, a_max=None)).astype(int)
    final_submit.to_csv(cfg.out_submission_csv, index=False, encoding="utf-8-sig")
    print(f"[TSMixer Submit] saved → {cfg.out_submission_csv}")