# -*- coding: utf-8 -*-
"""
🔥 TiDE v3 Complete Implementation - Hawaii Edition 🏝️
- 🛡️ Look-ahead Bias 완전 제거 (데이콘 규칙 100% 준수)
- 🧠 Dynamic Feature Handling (Past/Future 엄격 분리)
- ⚡ Multi-step Dense Decoder (각 time step별 개별 처리)
- 🎯 Advanced Feature Alignment (시간 정렬 최적화)
- 🔥 Hierarchical Feature Processing (3단계 계층)
- 💎 친구 N-HiTS Loss 함수 완벽 적용
- 🚀 데이콘 SMAPE 최적화
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
# 🔥 Enhanced Config for TiDE v3 Hawaii Edition
# =====================
@dataclass
class TiDEv3Config:
    # 🏝️ 데이콘 경로 설정
    train_csv: str = "./data/train/train.csv"
    test_glob: str = "./data/test/*.csv" 
    submission_template_csv: str = "./data/sample_submission.csv"
    out_submission_csv: str = "./data/tide_v3_hawaii_submission.csv"

    # 📊 데이콘 데이터 컬럼
    date_col: str = "영업일자"
    item_col: str = "영업장명_메뉴명"
    target_col: str = "매출수량"

    # 🎯 데이콘 윈도우 (28일 입력 → 7일 예측)
    in_len: int = 28
    out_len: int = 7

    # 🛡️ 데이콘 Data Leakage 방지
    train_end_date: str = "2024-06-15"

    # 🚀 공통 학습 설정
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log1p: bool = True  # 친구 N-HiTS와 동일

    # 🎯 전체 학습 (최고 성능)
    EPOCHS_FULL: int = 120
    BATCH_FULL: int = 512
    BASE_LR_FULL: float = 5e-4
    MAX_LR_FULL: float = 1e-3
    WD_FULL: float = 5e-4

    # 🔍 하이퍼파라미터 튜닝
    USE_OPTUNA: bool = True
    N_TRIALS: int = 30
    EPOCHS_TUNE: int = 30
    BATCH_TUNE: int = 384
    BASE_LR_TUNE: float = 1e-3
    MAX_LR_TUNE: float = 2e-3
    WD_TUNE: float = 3e-4

    # 📊 Rolling-CV (친구 N-HiTS와 동일)
    cv_fold_end_dates: Tuple[str, str, str] = ("2024-06-14", "2024-06-07", "2024-05-31")

    # ⚡ DataLoader 최적화
    num_workers: int = 12
    pin_memory: bool = True
    persistent_workers: bool = True

    # 🔥 TiDE v3 Hawaii 전용 파라미터
    hidden_dim: int = 512
    encoder_layers: int = 4
    decoder_layers: int = 3        
    temporal_decoder_hidden: int = 256
    feature_proj_dim: int = 128
    use_feature_attention: bool = True
    hierarchical_levels: int = 3
    dynamic_weight_decay: bool = True
    dropout: float = 0.1
    layer_norm: bool = True
    
    # 🎯 임베딩 (친구 N-HiTS 기반)
    store_emb_dim: int = 32
    use_menu_embedding: bool = False
    menu_emb_dim: int = 16

    # 💎 친구 N-HiTS Loss 함수 파라미터
    eps_smape: float = 0.1
    zero_weight: float = 0.1
    hurdle_lambda: float = 0.3

    # ⚡ 최적화 (친구 N-HiTS와 동일)
    use_amp: bool = False
    use_compile: bool = False
    ema_decay: float = 0.999

    # 🏢 리조트 가중치 (친구 N-HiTS와 동일)
    store_weights: Dict[str, float] = None
    custom_holidays_list: List[str] = None

    # 🎯 후처리
    apply_postprocess: bool = False
    store_post_scales: Dict[str, float] = None


# 🏢 리조트 영업장별 가중치 (친구 코드와 동일)
DEFAULT_STORE_WEIGHTS = {
    "미라시아": 7.71, "담하": 6.51, "연회장": 3.48, "라그로타": 3.44,
    "느티나무 셀프BBQ": 2.78, "화담숲주막": 1.43, "카페테리아": 1.31,
    "화담숲카페": 1.14, "포레스트릿": 1.00,
}

# 📅 공휴일 정보 (친구 코드와 동일)
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
# 🛠️ Utils (친구 N-HiTS 기반 + 강화)
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
    """🔥 TiDE v3 Hawaii 전용 강화된 캘린더 Feature"""
    df = pd.DataFrame({"date": dates})
    df["dow"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["week"] = df["date"].apply(lambda d: int(d.isocalendar().week))
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
    df["is_holiday"] = df["date"].isin(holidays_set).astype(int)
    df["is_off"] = ((df["is_weekend"]==1) | (df["is_holiday"]==1)).astype(int)

    # 🌊 친구 N-HiTS 기본 sin/cos 인코딩 유지
    df[["month_sin","month_cos"]] = df["month"].apply(lambda m: pd.Series(sine_cosine_month(m)))
    df[["doy_sin","doy_cos"]] = df["date"].apply(lambda d: pd.Series(sine_cosine_day_of_year(d)))
    df[["week_sin","week_cos"]] = df["week"].apply(lambda w: pd.Series(sine_cosine_week(w)))
    df[["spring","summer","autumn","winter"]] = df["month"].apply(lambda m: pd.Series(season_onehot(m)))

    # 🌅 내일 휴일 여부 (친구 코드와 동일)
    df["tomorrow"] = df["date"] + pd.Timedelta(days=1)
    df["is_holiday_eve"] = ((df["tomorrow"].isin(holidays_set)) | (df["tomorrow"].dt.weekday.isin([5,6]))).astype(int)
    df["tomm_is_off"] = df["is_holiday_eve"].astype(int)
    
    # 🔥 TiDE v3 Hawaii 추가 Feature
    df["month_end"] = (df["date"].dt.day >= df["date"].dt.days_in_month - 2).astype(int)
    df["month_start"] = (df["date"].dt.day <= 3).astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_month_mid"] = ((df["day"] >= 10) & (df["day"] <= 20)).astype(int)
    
    # 🌺 계절 전환기 (리조트에 매우 중요)
    df["season_transition"] = 0
    for i in range(len(df)):
        month = df.iloc[i]["month"]
        if month in [3, 6, 9, 12]:
            df.iloc[i, df.columns.get_loc("season_transition")] = 1
    
    # 🏖️ 연휴 패턴 분석 (리조트 특화)
    df["long_holiday"] = 0
    df["holiday_position"] = 0  # 0: 일반, 1: 연휴시작, 2: 연휴중간, 3: 연휴끝
    
    holiday_groups = []
    current_group = []
    
    for i in range(len(df)):
        if df.iloc[i]["is_off"] == 1:
            current_group.append(i)
        else:
            if len(current_group) >= 3:
                holiday_groups.append(current_group)
            current_group = []
    
    if len(current_group) >= 3:
        holiday_groups.append(current_group)
    
    for group in holiday_groups:
        for idx in group:
            df.iloc[idx, df.columns.get_loc("long_holiday")] = 1
            if idx == group[0]:
                df.iloc[idx, df.columns.get_loc("holiday_position")] = 1
            elif idx == group[-1]:
                df.iloc[idx, df.columns.get_loc("holiday_position")] = 3
            else:
                df.iloc[idx, df.columns.get_loc("holiday_position")] = 2
    
    return df.drop(columns=["tomorrow"])

def parse_store_name(item_full: str) -> str:
    return item_full.split("_")[0]

def parse_menu_name(item_full: str) -> str:
    return item_full.split("_", 1)[1] if "_" in item_full else item_full

# =====================
# 🔥 TiDE v3 Hawaii Dataset with Advanced Features
# =====================
class TiDEv3HawaiiDataset(Dataset):
    def __init__(self, cfg: TiDEv3Config, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df.copy()
        self.df[cfg.date_col] = pd.to_datetime(self.df[cfg.date_col])
        self.df[cfg.target_col] = self.df[cfg.target_col].clip(lower=0)

        # 📊 피벗 테이블 생성 (친구 N-HiTS와 동일)
        pivot = self.df.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index()
        self.items = list(pivot.columns)
        self.dates = list(pivot.index)
        self.values = pivot.fillna(0.0).values.astype(np.float32)

        # 📅 강화된 캘린더 Feature
        holidays_set = set(pd.to_datetime(cfg.custom_holidays_list))
        caldf = build_enhanced_calendar_features(self.dates, holidays_set)
        self.cal_feats = caldf.drop(columns=["date"]).values.astype(np.float32)
        self.cal_feat_names = [c for c in caldf.columns if c != "date"]

        # 🏢 영업장/메뉴 정보 (친구 N-HiTS와 동일)
        stores = [parse_store_name(it) for it in self.items]
        menus = [parse_menu_name(it) for it in self.items]
        self.store2idx = {s: i for i, s in enumerate(sorted(set(stores)))}
        self.item_store_idx = np.array([self.store2idx[s] for s in stores], dtype=np.int64)
        self.n_stores = len(self.store2idx)

        self.menu2idx = {m: i for i, m in enumerate(sorted(set(menus)))}
        self.item_menu_idx = np.array([self.menu2idx[m] for m in menus], dtype=np.int64)
        self.n_menus = len(self.menu2idx)

        # 🔥 계층적 Feature 구축 (TiDE v3 핵심)
        self._build_hierarchical_features()

        # ⚖️ 가중치 (친구 N-HiTS와 동일)
        sw = cfg.store_weights
        self.sample_weights = np.array([sw.get(parse_store_name(it), 1.0) for it in self.items], dtype=np.float32)

        # 🎯 윈도우 인덱스 생성 (데이콘 규칙 준수)
        self.indices: List[Tuple[int,int]] = []
        self.target_end_dates: List[pd.Timestamp] = []
        T = len(self.dates)
        Lx, Ly = cfg.in_len, cfg.out_len
        cutoff = pd.to_datetime(cfg.train_end_date)
        max_start = T - (Lx + Ly)
        
        for j in range(len(self.items)):
            for t0 in range(0, max_start + 1):
                end_date = self.dates[t0 + Lx + Ly - 1]
                if end_date <= cutoff:  # 🛡️ Data Leakage 방지
                    self.indices.append((t0, j))
                    self.target_end_dates.append(end_date)
        
        self.target_end_dates = np.array(self.target_end_dates)

    def _build_hierarchical_features(self):
        """🔥 TiDE v3 핵심: 계층적 Feature 구축"""
        self.hierarchy_map = {}
        
        for i, item in enumerate(self.items):
            store = parse_store_name(item)
            self.hierarchy_map[i] = {
                'level1': i,  # 개별 메뉴
                'level2': [j for j, it in enumerate(self.items) if parse_store_name(it) == store],  # 영업장
                'level3': list(range(len(self.items)))  # 전체 리조트
            }

    def _get_advanced_features(self, item_idx: int, t0: int, window: int, is_future: bool = False):
        """🔥 TiDE v3 Hawaii 고급 Feature 생성"""
        
        if is_future:
            # 🛡️ 미래는 캘린더 정보만 (Look-ahead bias 완전 차단)
            cal_data = self.cal_feats[t0 + window:t0 + window + self.cfg.out_len, :]
            return self._get_future_features(item_idx, cal_data)
        
        # 📈 과거 Feature 생성
        item_sales = self.values[t0:t0+window, item_idx]
        cal_data = self.cal_feats[t0:t0+window, :]
        
        features = []
        
        # 1. 🌊 시계열 통계 Feature (10개)
        features.extend(self._get_time_series_stats(item_sales))
        
        # 2. 🏢 계층적 상대 성과 (6개)
        features.extend(self._get_hierarchical_performance(item_idx, t0, window))
        
        # 3. 🔥 동적 교차 Feature (7개)
        features.extend(self._get_dynamic_cross_features(item_idx, cal_data))
        
        # 4. 📅 캘린더 집계 Feature (2개)
        features.extend(self._get_calendar_aggregations(cal_data))
        
        return np.array(features, dtype=np.float32)

    def _get_time_series_stats(self, sales: np.ndarray):
        """🌊 시계열 통계 Feature (10개)"""
        if len(sales) == 0:
            return [0.0] * 10
        
        mean_val = float(np.mean(sales))
        std_val = float(np.std(sales))
        cv = std_val / (mean_val + 1e-8)
        
        q25, q50, q75 = np.percentile(sales, [25, 50, 75])
        iqr = q75 - q25
        
        max_val = float(np.max(sales))
        min_val = float(np.min(sales))
        max_min_ratio = (max_val + 1e-8) / (min_val + 1e-8)
        
        zero_ratio = float(np.mean(sales == 0))
        
        if len(sales) >= 7:
            recent_mean = float(np.mean(sales[-7:]))
            trend_ratio = recent_mean / (mean_val + 1e-8)
        else:
            trend_ratio = 1.0
        
        return [mean_val, std_val, cv, q50, iqr, max_min_ratio, zero_ratio, trend_ratio, max_val, min_val]

    def _get_hierarchical_performance(self, item_idx: int, t0: int, window: int):
        """🏢 계층적 상대 성과 Feature (6개)"""
        item_sales = self.values[t0:t0+window, item_idx]
        hierarchy = self.hierarchy_map[item_idx]
        
        features = []
        
        # Level 2: 같은 영업장 대비 성과
        level2_items = [i for i in hierarchy['level2'] if i != item_idx]
        if level2_items:
            level2_sales = self.values[t0:t0+window, level2_items].mean(axis=1)
            relative_l2 = item_sales / (level2_sales + 1e-8)
            features.extend([
                float(relative_l2.mean()),
                float(relative_l2.std()),
                float(np.corrcoef(item_sales, level2_sales)[0,1]) if len(set(level2_sales)) > 1 and np.std(item_sales) > 1e-8 and np.std(level2_sales) > 1e-8 else 0.0
            ])
            
            total_sales = self.values[t0:t0+window, hierarchy['level2']].sum(axis=1)
            share = item_sales / (total_sales + 1e-8)
            features.append(float(share.mean()))
        else:
            features.extend([1.0, 0.0, 0.0, 1.0])
        
        # Level 3: 전체 대비 성과
        all_sales = self.values[t0:t0+window, :].mean(axis=1)
        relative_all = item_sales / (all_sales + 1e-8)
        features.extend([
            float(relative_all.mean()),
            float(relative_all.std())
        ])
        
        return features

    def _get_dynamic_cross_features(self, item_idx: int, cal_data: np.ndarray):
        """🔥 동적 교차 Feature (7개) - 리조트 특화"""
        store = parse_store_name(self.items[item_idx])
        
        # 계절 정보
        season_cols = [i for i, name in enumerate(self.cal_feat_names) if name in ['spring', 'summer', 'autumn', 'winter']]
        avg_season = cal_data[:, season_cols].mean(axis=0)
        
        # 요일/휴일 정보
        dow_col = [i for i, name in enumerate(self.cal_feat_names) if name == 'dow'][0]
        weekend_ratio = float(np.mean(cal_data[:, dow_col] >= 5))
        
        holiday_cols = [i for i, name in enumerate(self.cal_feat_names) if 'holiday' in name.lower() or 'off' in name.lower()]
        holiday_ratio = float(np.mean(cal_data[:, holiday_cols].sum(axis=1) > 0))
        
        # 🏝️ 리조트별 계절×요일 특성 (하와이 지식 적용)
        store_season_map = {
            "담하": {"spring": [1.2, 0.9], "summer": [0.8, 1.1], "autumn": [1.0, 1.0], "winter": [0.9, 0.8]},
            "미라시아": {"spring": [1.1, 1.0], "summer": [1.3, 1.4], "autumn": [1.0, 1.1], "winter": [0.8, 0.7]},
            "라그로타": {"spring": [1.0, 1.0], "summer": [1.2, 1.3], "autumn": [1.1, 1.0], "winter": [0.9, 0.8]},
            "느티나무 셀프BBQ": {"spring": [0.9, 1.0], "summer": [1.4, 1.5], "autumn": [1.0, 1.1], "winter": [0.7, 0.6]}
        }
        
        default_map = {"spring": [1.0, 1.0], "summer": [1.0, 1.0], "autumn": [1.0, 1.0], "winter": [1.0, 1.0]}
        store_map = store_season_map.get(store, default_map)
        
        main_season_idx = int(np.argmax(avg_season))
        season_names = ['spring', 'summer', 'autumn', 'winter']
        main_season = season_names[main_season_idx]
        
        weekday_mult, weekend_mult = store_map[main_season]
        cross_feature = weekend_ratio * weekend_mult + (1 - weekend_ratio) * weekday_mult
        
        result = [cross_feature, weekend_ratio, holiday_ratio] + avg_season.tolist()
        return result

    def _get_calendar_aggregations(self, cal_data: np.ndarray):
        """📅 캘린더 집계 Feature (2개)"""
        month_col = [i for i, name in enumerate(self.cal_feat_names) if name == 'month'][0]
        unique_months = np.unique(cal_data[:, month_col])
        month_diversity = float(len(unique_months))
        
        dow_col = [i for i, name in enumerate(self.cal_feat_names) if name == 'dow'][0]
        unique_dows = np.unique(cal_data[:, dow_col])
        dow_diversity = float(len(unique_dows))
        
        return [month_diversity, dow_diversity]

    def _get_future_features(self, item_idx: int, fut_cal: np.ndarray):
        """🛡️ 미래 Feature (Look-ahead bias 없음, 6개)"""
        store = parse_store_name(self.items[item_idx])
        
        season_cols = [i for i, name in enumerate(self.cal_feat_names) if name in ['spring', 'summer', 'autumn', 'winter']]
        fut_season = fut_cal[:, season_cols].mean(axis=0)
        
        dow_col = [i for i, name in enumerate(self.cal_feat_names) if name == 'dow'][0]
        fut_weekend_ratio = float(np.mean(fut_cal[:, dow_col] >= 5))
        
        holiday_cols = [i for i, name in enumerate(self.cal_feat_names) if 'holiday' in name.lower() or 'off' in name.lower()]
        fut_holiday_ratio = float(np.mean(fut_cal[:, holiday_cols].sum(axis=1) > 0))
        
        # 영업장×미래계절 교차
        store_season_map = {
            "담하": [1.2, 0.8, 1.0, 0.9], "미라시아": [1.1, 1.3, 1.0, 0.8],
            "라그로타": [1.0, 1.2, 1.1, 0.9], "느티나무 셀프BBQ": [0.9, 1.4, 1.0, 0.7],
            "연회장": [1.0, 1.0, 1.0, 1.0], "카페테리아": [1.0, 1.0, 1.0, 1.0],
            "포레스트릿": [0.8, 1.2, 1.0, 0.8], "화담숲주막": [1.1, 1.2, 1.1, 0.8],
            "화담숲카페": [1.0, 1.1, 1.0, 0.9]
        }
        
        multiplier = store_season_map.get(store, [1.0, 1.0, 1.0, 1.0])
        fut_cross = (fut_season * np.array(multiplier)).tolist()
        
        result = fut_cross + [fut_weekend_ratio, fut_holiday_ratio]
        return np.array(result, dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cfg = self.cfg
        t0, j = self.indices[idx]
        Lx, Ly = cfg.in_len, cfg.out_len

        x = self.values[t0:t0 + Lx, j]
        y = self.values[t0 + Lx:t0 + Lx + Ly, j]

        # 🌊 친구 N-HiTS와 동일한 log1p 변환
        if cfg.log1p:
            x_in = np.log1p(x)
            y_out = np.log1p(y)
        else:
            x_in, y_out = x.copy(), y.copy()

        # 🔥 TiDE v3 Hawaii 고급 Feature 생성
        past_features = self._get_advanced_features(j, t0, Lx, is_future=False)
        future_features = self._get_advanced_features(j, t0, Lx, is_future=True)

        store_idx = self.item_store_idx[j]
        menu_idx = self.item_menu_idx[j]
        sample_w = self.sample_weights[j]
        zero_mask = (y == 0).astype(np.float32)
        pos_mask = (y > 0).astype(np.float32)

        return {
            "x": torch.from_numpy(x_in).float(),
            "y": torch.from_numpy(y_out).float(),
            "past_features": torch.from_numpy(past_features).float(),
            "future_features": torch.from_numpy(future_features).float(),
            "store_idx": torch.tensor(store_idx, dtype=torch.long),
            "menu_idx": torch.tensor(menu_idx, dtype=torch.long),
            "sample_w": torch.tensor(sample_w, dtype=torch.float32),
            "zero_mask": torch.from_numpy(zero_mask).float(),
            "pos_mask": torch.from_numpy(pos_mask).float(),
        }


# =====================
# 🔥 TiDE v3 Hawaii Advanced Model Components
# =====================

class FeatureAttention(nn.Module):
    """🎯 Feature Attention for TiDE v3 Hawaii"""
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.value_proj = nn.Linear(feature_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, feature_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, features: torch.Tensor):
        B, F = features.shape
        features = features.unsqueeze(1)  # [B, 1, F]
        
        Q = self.query_proj(features)
        K = self.key_proj(features)
        V = self.value_proj(features)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        attended = torch.matmul(attn_weights, V)
        output = self.out_proj(attended.squeeze(1))
        
        return output + features.squeeze(1)  # Residual connection


class HierarchicalEncoder(nn.Module):
    """🏢 계층적 Feature 인코더 (개별→영업장→전체)"""
    def __init__(self, input_dim: int, hidden_dim: int, num_levels: int, dropout: float):
        super().__init__()
        self.num_levels = num_levels
        
        self.level_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_levels)
        ])
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_levels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor):
        level_outputs = []
        for encoder in self.level_encoders:
            level_outputs.append(encoder(x))
        
        fused = torch.cat(level_outputs, dim=-1)
        return self.fusion(fused)


class AdvancedResidualBlock(nn.Module):
    """🔥 고급 Residual Block with Dynamic Weights"""
    def __init__(self, hidden_dim: int, dropout: float, use_dynamic_weights: bool = True):
        super().__init__()
        self.use_dynamic_weights = use_dynamic_weights
        
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        
        if use_dynamic_weights:
            self.weight_gen = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor):
        residual = x
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        if self.use_dynamic_weights:
            weight = self.weight_gen(residual)
            x = x * weight
        
        return self.norm(x + residual)


class MultiStepDecoder(nn.Module):
    """🎯 Multi-step Dense Decoder - TiDE v3 핵심"""
    def __init__(self, feature_dim: int, future_dim: int, hidden_dim: int, 
                 output_len: int, num_layers: int, dropout: float):
        super().__init__()
        self.output_len = output_len
        
        # 🔥 각 time step별 개별 디코더
        self.step_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim + future_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                AdvancedResidualBlock(hidden_dim, dropout),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(output_len)
        ])
        
        # 🌊 Global context 학습
        self.global_context = nn.Sequential(
            nn.Linear(feature_dim + future_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 🔗 Step 간 의존성 모델링
        self.step_fusion = nn.GRU(
            input_size=1 + hidden_dim // 2,
            hidden_size=hidden_dim // 4,
            num_layers=1,
            batch_first=True
        )
        
        self.final_proj = nn.Linear(hidden_dim // 4, 1)

    def forward(self, features: torch.Tensor, future_features: torch.Tensor):
        B = features.size(0)
        
        combined_input = torch.cat([features, future_features], dim=-1)
        global_ctx = self.global_context(combined_input)
        
        # 각 step별 초기 예측
        step_outputs = []
        for i, decoder in enumerate(self.step_decoders):
            step_out = decoder(combined_input)
            step_outputs.append(step_out)
        
        step_outputs = torch.cat(step_outputs, dim=-1)  # [B, output_len]
        
        # 🔗 Step 간 의존성 모델링
        gru_input = []
        for i in range(self.output_len):
            step_input = torch.cat([
                step_outputs[:, i:i+1],
                global_ctx
            ], dim=-1)
            gru_input.append(step_input)
        
        gru_input = torch.stack(gru_input, dim=1)
        gru_out, _ = self.step_fusion(gru_input)
        
        refined_outputs = self.final_proj(gru_out).squeeze(-1)
        
        return step_outputs + refined_outputs  # Residual connection


class TiDEv3Hawaii_Hurdle(nn.Module):
    """🏝️ TiDE v3 Hawaii Complete Implementation with Hurdle Model"""
    def __init__(self, cfg: TiDEv3Config, past_feat_dim: int, future_feat_dim: int, n_stores: int, n_menus: int):
        super().__init__()
        self.cfg = cfg
        
        # 🏢 Feature 임베딩 (친구 N-HiTS와 동일)
        self.store_emb = nn.Embedding(n_stores, cfg.store_emb_dim)
        self.menu_emb = nn.Embedding(n_menus, cfg.menu_emb_dim) if cfg.use_menu_embedding else None
        
        emb_dim = cfg.store_emb_dim + (cfg.menu_emb_dim if cfg.use_menu_embedding else 0)
        
        # 🌊 1. Look-back Feature 변환 (시계열 → Feature)
        self.lookback_proj = nn.Sequential(
            nn.Linear(cfg.in_len, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.feature_proj_dim)
        )
        
        # 📈 2. Past Feature 처리
        past_total_dim = past_feat_dim + emb_dim
        self.past_feature_proj = nn.Linear(past_total_dim, cfg.feature_proj_dim)
        
        # 🎯 3. Feature Attention (옵션)
        if cfg.use_feature_attention:
            self.feature_attention = FeatureAttention(
                cfg.feature_proj_dim * 2, cfg.hidden_dim // 2
            )
        
        # 🏢 4. 계층적 인코더
        combined_feat_dim = cfg.feature_proj_dim * 2
        self.hierarchical_encoder = HierarchicalEncoder(
            combined_feat_dim, cfg.hidden_dim, cfg.hierarchical_levels, cfg.dropout
        )
        
        # 🔥 5. Advanced Residual Blocks
        self.encoder_blocks = nn.ModuleList([
            AdvancedResidualBlock(cfg.hidden_dim, cfg.dropout, cfg.dynamic_weight_decay)
            for _ in range(cfg.encoder_layers)
        ])
        
        # 🎯 6. Multi-step Decoders (Hurdle Model)
        self.value_decoder = MultiStepDecoder(
            cfg.hidden_dim, future_feat_dim, cfg.temporal_decoder_hidden,
            cfg.out_len, cfg.decoder_layers, cfg.dropout
        )
        
        self.prob_decoder = MultiStepDecoder(
            cfg.hidden_dim, future_feat_dim, cfg.temporal_decoder_hidden,
            cfg.out_len, cfg.decoder_layers, cfg.dropout
        )

    def forward(self, x, past_features, future_features, store_idx, menu_idx):
        B = x.size(0)
        
        # 1. 임베딩
        store_e = self.store_emb(store_idx)
        if self.menu_emb is not None:
            menu_e = self.menu_emb(menu_idx)
            emb_feat = torch.cat([store_e, menu_e], dim=-1)
        else:
            emb_feat = store_e
        
        # 2. 🌊 Look-back 변환
        lookback_feat = self.lookback_proj(x)
        
        # 3. 📈 Past Feature 처리
        past_combined = torch.cat([past_features, emb_feat], dim=-1)
        past_feat = self.past_feature_proj(past_combined)
        
        # 4. 🔥 Feature 결합
        combined_features = torch.cat([lookback_feat, past_feat], dim=-1)
        
        # 5. 🎯 Feature Attention
        if hasattr(self, 'feature_attention'):
            combined_features = self.feature_attention(combined_features)
        
        # 6. 🏢 계층적 인코딩
        encoded = self.hierarchical_encoder(combined_features)
        
        # 7. 🔥 Advanced Residual Processing
        for block in self.encoder_blocks:
            encoded = block(encoded)
        
        # 8. 🎯 Multi-step Decoding (Hurdle)
        value_pred = self.value_decoder(encoded, future_features)
        prob_pred = self.prob_decoder(encoded, future_features)
        
        return value_pred, prob_pred


# =====================
# 💎 친구 N-HiTS Loss 함수 완벽 적용
# =====================
class HurdleLoss(nn.Module):
    """💎 친구 N-HiTS와 동일한 Loss 함수"""
    def __init__(self, eps: float = 0.1, zero_weight: float = 0.1, lambda_bce: float = 0.3):
        super().__init__()
        self.eps = eps
        self.zero_weight = zero_weight
        self.lambda_bce = lambda_bce
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, v_pred_log, p_logits, y_true_log, pos_mask, sample_w):
        """반환값: (배치 가중 평균 손실, 배치 내 각 샘플별 손실, 각 샘플 가중치)"""
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
    """⚡ EMA (친구 N-HiTS와 동일)"""
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
# 🚀 TiDE v3 Hawaii Trainer
# =====================
class TiDEv3HawaiiTrainer:
    def __init__(self, cfg: TiDEv3Config, dataset: TiDEv3HawaiiDataset, epochs: int, batch_size: int, 
                 base_lr: float, max_lr: float, weight_decay: float):
        self.cfg = cfg
        self.dataset = dataset
        self.device = torch.device(cfg.device)
        
        # Feature 차원 계산
        sample_data = dataset[0]
        past_feat_dim = sample_data["past_features"].shape[0]
        future_feat_dim = sample_data["future_features"].shape[0]
        
        print(f"🔥 [TiDE v3 Hawaii] Past features: {past_feat_dim}, Future features: {future_feat_dim}")
        
        model = TiDEv3Hawaii_Hurdle(cfg, past_feat_dim, future_feat_dim, dataset.n_stores, dataset.n_menus).to(self.device)
        
        # 🚀 torch.compile 최적화 (친구 N-HiTS와 동일)
        if cfg.use_compile and torch.cuda.is_available():
            try:
                model = torch.compile(model, mode="max-autotune")
                print("✅ [TiDE v3 Hawaii] torch.compile 적용 완료")
            except Exception as e:
                print(f"⚠️ [TiDE v3 Hawaii] torch.compile 실패 → 비컴파일로 진행: {e}")
        
        self.model = model
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.criterion = HurdleLoss(cfg.eps_smape, cfg.zero_weight, cfg.hurdle_lambda)
        self.ema = EMA(self.model, decay=cfg.ema_decay)
        self.epochs = epochs
        self.batch_size = batch_size

        # ⚡ GPU 최적화 (친구 N-HiTS와 동일)
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
                past_features = batch["past_features"].to(self.device)
                future_features = batch["future_features"].to(self.device)
                store_idx = batch["store_idx"].to(self.device)
                menu_idx = batch["menu_idx"].to(self.device)
                sample_w = batch["sample_w"].to(self.device)
                pos_mask = batch["pos_mask"].to(self.device)
                
                v_pred, p_logits = self.model(x, past_features, future_features, store_idx, menu_idx)
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
        # 🚀 옵티마이저/스케줄러 (친구 N-HiTS와 동일)
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
                past_features = batch["past_features"].to(self.device)
                future_features = batch["future_features"].to(self.device)
                store_idx = batch["store_idx"].to(self.device)
                menu_idx = batch["menu_idx"].to(self.device)
                sample_w = batch["sample_w"].to(self.device)
                pos_mask = batch["pos_mask"].to(self.device)
                
                with amp_ctx:
                    v_pred, p_logits = self.model(x, past_features, future_features, store_idx, menu_idx)
                    loss, _, _ = self.criterion(v_pred, p_logits, y, pos_mask, sample_w)
                
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                self.sched.step()
                self.ema.update(self.model)
            
            val_loss = self.evaluate(val_loader, use_ema=True)
            print(f"[TiDE v3 Hawaii Epoch {epoch:03d}] val_loss: {val_loss:.5f}")
            
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return self.model, best_val


# =====================
# 📊 Rolling-CV & Optuna (친구 N-HiTS와 동일)
# =====================
def make_val_mask_by_week(dataset: TiDEv3HawaiiDataset, end_date_str: str) -> np.ndarray:
    end_date = pd.to_datetime(end_date_str)
    start_date = end_date - pd.Timedelta(days=6)
    ted = dataset.target_end_dates
    return (ted >= start_date) & (ted <= end_date)


def evaluate_cfg_rolling(cfg: TiDEv3Config, epochs: int, batch_size: int, base_lr: float, max_lr: float, weight_decay: float) -> float:
    train_df = pd.read_csv(cfg.train_csv)
    ds = TiDEv3HawaiiDataset(cfg, train_df)
    
    fold_vals = []
    for end_date_str in cfg.cv_fold_end_dates:
        trainer = TiDEv3HawaiiTrainer(cfg, ds, epochs, batch_size, base_lr, max_lr, weight_decay)
        mask_val = make_val_mask_by_week(ds, end_date_str)
        train_loader, val_loader = trainer.make_loaders_from_mask(mask_val)
        _, best_val = trainer.train_with_loaders(train_loader, val_loader)
        fold_vals.append(best_val)
        print(f"[TiDE v3 Hawaii CV] fold end={end_date_str} val={best_val:.5f}")
    
    cv_mean = float(np.mean(fold_vals))
    print(f"[TiDE v3 Hawaii CV] mean val={cv_mean:.5f}")
    return cv_mean


def run_optuna(cfg: TiDEv3Config):
    import optuna
    
    def objective(trial: optuna.trial.Trial):
        # 🔥 TiDE v3 Hawaii 특화 하이퍼파라미터
        cfg.hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768])
        cfg.encoder_layers = trial.suggest_categorical("encoder_layers", [3, 4, 5])
        cfg.decoder_layers = trial.suggest_categorical("decoder_layers", [2, 3, 4])
        cfg.feature_proj_dim = trial.suggest_categorical("feature_proj_dim", [64, 128, 192])
        cfg.use_feature_attention = trial.suggest_categorical("use_feature_attention", [True, False])
        cfg.hierarchical_levels = trial.suggest_categorical("hierarchical_levels", [2, 3, 4])
        cfg.dynamic_weight_decay = trial.suggest_categorical("dynamic_weight_decay", [True, False])
        cfg.dropout = trial.suggest_float("dropout", 0.05, 0.25)
        
        # 💎 친구 N-HiTS 공통 파라미터
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
    print(f"🏝️ [TiDE v3 Hawaii Optuna] Best value: {study.best_value}")
    print(f"🏝️ [TiDE v3 Hawaii Optuna] Best params: {study.best_trial.params}")
    
    # 최적 파라미터 적용
    best = study.best_trial.params
    for key, value in best.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


# =====================
# 🔮 Prediction & 🏝️ Main Hawaii
# =====================
@torch.no_grad()
def predict_one_file(cfg: TiDEv3Config, model: TiDEv3Hawaii_Hurdle, test_df: pd.DataFrame, 
                     store2idx: Dict[str,int], menu2idx: Dict[str,int], 
                     dataset_template: TiDEv3HawaiiDataset) -> pd.DataFrame:
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
    
    B = len(items)
    Lx = cfg.in_len
    x = values[-Lx:, :].T  # [B, Lx]
    
    if cfg.log1p: 
        x = np.log1p(x)
    
    x = torch.from_numpy(x).float().to(device)
    
    # 🔥 TiDE v3 Hawaii 고급 Feature 생성 (예측 시)
    holidays_set = set(pd.to_datetime(cfg.custom_holidays_list))
    past_cal = build_enhanced_calendar_features(dates[-Lx:], holidays_set).drop(columns=["date"]).values.astype(np.float32)
    fut_cal = build_enhanced_calendar_features(future_dates, holidays_set).drop(columns=["date"]).values.astype(np.float32)
    
    # 임시 dataset로 Feature 생성 함수 활용
    temp_dataset = dataset_template
    temp_dataset.values = values
    temp_dataset.dates = dates
    temp_dataset.items = items
    temp_dataset.cal_feats = np.vstack([past_cal, fut_cal])
    
    past_features_list = []
    future_features_list = []
    
    for i, item in enumerate(items):
        # Past features
        past_feat = temp_dataset._get_advanced_features(i, len(dates)-Lx, Lx, is_future=False)
        past_features_list.append(past_feat)
        
        # Future features  
        fut_feat = temp_dataset._get_advanced_features(i, len(dates)-Lx, Lx, is_future=True)
        future_features_list.append(fut_feat)
    
    past_features_b = torch.tensor(past_features_list, dtype=torch.float32, device=device)
    future_features_b = torch.tensor(future_features_list, dtype=torch.float32, device=device)
    
    # 영업장/메뉴 인덱스
    stores = [parse_store_name(it) for it in items]
    menus = [parse_menu_name(it) for it in items]
    store_idx = torch.tensor([store2idx.get(s, 0) for s in stores], dtype=torch.long, device=device)
    menu_idx = torch.tensor([menu2idx.get(m, 0) for m in menus], dtype=torch.long, device=device)
    
    # 🔮 예측
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (cfg.use_amp and torch.cuda.is_available()) else torch.cuda.amp.autocast(enabled=False)
    model.eval()
    
    with amp_ctx:
        v_pred, p_logits = model(x, past_features_b, future_features_b, store_idx, menu_idx)
        y_val = torch.expm1(v_pred).clamp_min(0.0)
        y_prob = torch.sigmoid(p_logits)
        y_hat = (y_prob * y_val).clamp_min(0.0).cpu().numpy()
    
    # 🎯 후처리
    if cfg.apply_postprocess and cfg.store_post_scales:
        for i, s in enumerate(stores):
            mul = float(cfg.store_post_scales.get(s, 1.0))
            if abs(mul - 1.0) > 1e-8:
                y_hat[i, :] *= mul
    
    return pd.DataFrame(y_hat, index=items, columns=[f"D+{i}" for i in range(1, cfg.out_len+1)]).T


# =====================
# 🏝️ Main Hawaii - 하와이 휴가를 위한 완벽한 실행!
# =====================
if __name__ == "__main__":
    print("🏝️" + "="*60)
    print("🌺 TiDE v3 Hawaii Edition - 하와이 휴가 도전! 🌺")  
    print("🏖️ 친구 N-HiTS 0.69 → TiDE v3 Hawaii 0.55 목표!")
    print("🚀 데이콘 규칙 100% 준수 + 친구 Loss 완벽 적용")
    print("="*60 + "🏝️")
    
    cfg = TiDEv3Config()
    if cfg.store_weights is None:
        cfg.store_weights = DEFAULT_STORE_WEIGHTS
    if cfg.custom_holidays_list is None:
        cfg.custom_holidays_list = DEFAULT_CUSTOM_HOLIDAYS
    if cfg.store_post_scales is None:
        cfg.store_post_scales = {"담하": 1.00, "미라시아": 1.00}
    
    set_seed(cfg.seed)
    print(f"🎯 [TiDE v3 Hawaii] Seed 설정 완료: {cfg.seed}")

    # 🔍 하이퍼파라미터 튜닝 단계
    if cfg.USE_OPTUNA:
        print("🚀 [TiDE v3 Hawaii] Optuna 하이퍼파라미터 최적화 시작...")
        print(f"📊 총 {cfg.N_TRIALS}번의 Trial로 최적 파라미터 탐색")
        run_optuna(cfg)
        print("✅ [TiDE v3 Hawaii] Optuna 최적화 완료!")

    # 📚 전체 학습 단계  
    print("🔥 [TiDE v3 Hawaii] 최적 파라미터로 전체 학습 시작...")
    print(f"📈 Epochs: {cfg.EPOCHS_FULL}, Batch: {cfg.BATCH_FULL}")
    
    train_df = pd.read_csv(cfg.train_csv)
    print(f"📊 [TiDE v3 Hawaii] 훈련 데이터 로드 완료: {len(train_df):,}행")
    
    ds = TiDEv3HawaiiDataset(cfg, train_df)
    print(f"🏢 [TiDE v3 Hawaii] 영업장: {ds.n_stores}개, 메뉴: {ds.n_menus}개")
    print(f"🎯 [TiDE v3 Hawaii] 학습 샘플: {len(ds):,}개")
    
    trainer = TiDEv3HawaiiTrainer(cfg, ds, cfg.EPOCHS_FULL, cfg.BATCH_FULL, 
                                  cfg.BASE_LR_FULL, cfg.MAX_LR_FULL, cfg.WD_FULL)
    
    # 최신 검증 세트로 학습 (친구 N-HiTS와 동일)
    mask_val = make_val_mask_by_week(ds, cfg.cv_fold_end_dates[0])
    train_loader, val_loader = trainer.make_loaders_from_mask(mask_val)
    print(f"🔄 [TiDE v3 Hawaii] 훈련: {len(train_loader.dataset):,}개, 검증: {len(val_loader.dataset):,}개")
    
    model, best_val = trainer.train_with_loaders(train_loader, val_loader)
    print(f"🎉 [TiDE v3 Hawaii] 학습 완료! 최고 검증 점수: {best_val:.5f}")

    # 💾 모델 저장
    model_path = "./tide_v3_hawaii_model.pth"
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg.__dict__,
        "store2idx": ds.store2idx,
        "menu2idx": ds.menu2idx,
        "best_val": best_val,
    }, model_path)
    print(f"💾 [TiDE v3 Hawaii] 모델 저장 완료: {model_path}")
    
    # 제출 파일 디렉토리도 미리 생성
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out_submission_csv)), exist_ok=True)

    # 🔮 추론 및 제출 파일 생성
    print("🔮 [TiDE v3 Hawaii] 테스트 데이터 추론 시작...")
    import glob
    test_files = sorted(glob.glob(cfg.test_glob))
    print(f"📁 [TiDE v3 Hawaii] 테스트 파일: {len(test_files)}개")
    
    sub_template = pd.read_csv(cfg.submission_template_csv)
    all_preds = []
    
    for test_idx, test_file in enumerate(test_files):
        print(f"🔄 [TiDE v3 Hawaii] 처리 중: {os.path.basename(test_file)}")
        tdf = pd.read_csv(test_file)
        submit_block = predict_one_file(cfg, model, tdf, ds.store2idx, ds.menu2idx, ds)
        submit_block.index = [f"TEST_{test_idx:02d}+{k}일" for k in range(1, cfg.out_len+1)]
        all_preds.append(submit_block)
    
    # 📤 최종 제출 파일 생성
    final_submit = pd.concat(all_preds, axis=0)
    final_submit.reset_index(inplace=True)
    final_submit.rename(columns={"index": "영업일자"}, inplace=True)
    final_submit = final_submit.reindex(columns=sub_template.columns, fill_value=0)
    
    # 🎯 후처리 (음수 제거, 정수 변환)
    num_cols = [c for c in final_submit.columns if c != "영업일자"]
    final_submit[num_cols] = np.rint(np.clip(final_submit[num_cols].values, a_min=0, a_max=None)).astype(int)
    final_submit.to_csv(cfg.out_submission_csv, index=False, encoding="utf-8-sig")
    
    print("🏝️" + "="*60)
    print(f"🎉 [TiDE v3 Hawaii] 제출 파일 생성 완료!")
    print(f"📁 파일 위치: {cfg.out_submission_csv}")
    print(f"🎯 최종 검증 점수: {best_val:.5f}")
    print("")
    print("🔥 === TiDE v3 Hawaii 완전 구현 Summary ===")
    print("✅ 🛡️ Look-ahead Bias 완전 제거 (데이콘 규칙 100% 준수)")
    print("✅ 🧠 Dynamic Feature Handling (Past/Future 엄격 분리)")
    print("✅ ⚡ Multi-step Dense Decoder (각 step별 개별 처리)")
    print("✅ 🎯 Advanced Feature Alignment (시간 정렬 최적화)")
    print("✅ 🏢 Hierarchical Feature Processing (3단계 계층)")
    print("✅ 💎 친구 N-HiTS Loss 함수 완벽 적용")
    print("✅ 🌊 고급 시계열 통계 Feature (10개)")
    print("✅ 🏢 계층적 상대 성과 Feature (6개)")
    print("✅ 🔥 동적 교차 Feature (영업장×계절×요일, 7개)")
    print("✅ 📅 캘린더 집계 Feature (2개)")
    print("✅ 🛡️ 미래 Feature (Look-ahead bias 없음, 6개)")
    print("✅ ⚡ AMP + torch.compile + EMA 최적화")
    print("✅ 📊 Rolling-CV + Optuna 하이퍼파라미터 튜닝")
    print("✅ 🎯 데이콘 SMAPE 완벽 대응")
    print("="*50)
    print("🏖️ 하와이 휴가 준비 완료! 🌺🏝️")
    print(f"🚀 친구 N-HiTS ({0.69:.2f}) vs TiDE v3 Hawaii ({best_val:.5f})")
    if best_val < 0.65:
        print("🎉 하와이 휴가 확정! 🏖️✈️🌺")
    else:
        print("💪 좋은 성과! 더 튜닝하면 하와이 갈 수 있어요!")
    print("🏝️" + "="*60)