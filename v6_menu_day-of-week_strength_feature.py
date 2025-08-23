# -*- coding: utf-8 -*-
"""
메뉴 상관관계 feature, 강세 요일 feature 추가
아래 코드에서 USE_CORR_FEATURES 또는 USE_DOW_STRENGTH 사용 설정하면 됨
다만 USE_DOW_STRENGTH만 사용하는 게 성능이 잘 나올것으로 예측됨 

    # ====== NEW: Feature toggles & params ======
    # Corr-based features
    USE_CORR_FEATURES: bool = False
    CORR_THRESHOLD: float = 0.5     # 상관 임계치
    CORR_TOPN: int = 5              # 메뉴당 상위 파트너 N개 제한 (0 = 제한 없음)
    CORR_LAGS: Tuple[int, ...] = (1, 7)
    CORR_RMEANS: Tuple[int, ...] = (7, 14)  # 모두 shift=1

    # Day-of-week strength
    USE_DOW_STRENGTH: bool = True
    DOW_RATIO_THRESHOLD: float = 2.0    # 평균 대비 2배 이상
    DOW_MIN_SUPPORT: int = 4            # 요일 평균 계산에 필요한 최소 관측일 수

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
    # train_csv: str = "/content/drive/MyDrive/data/train/train_original.csv"
    # test_glob: str = "/content/drive/MyDrive/data/test/*.csv"
    # submission_template_csv: str = "/content/drive/MyDrive/data/sample_submission.csv"
    # out_submission_csv: str = "/content/drive/MyDrive/data/0823_submission.csv"
    # checkpoint_dir: str = "/content/drive/MyDrive/data/checkpoint/enhanced_checkpoints"
    train_csv: str = "LG_train.csv"
    test_glob: str = "test/*.csv"
    submission_template_csv: str = "sample_submission.csv"
    out_submission_csv: str = "./0824_submission.csv"
    checkpoint_dir: str = "./0822_0.495_clu_ratio_4f_checkpoints"


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
    EPOCHS_FULL: int = 150
    BATCH_FULL: int = 256
    BASE_LR_FULL: float = 6e-4
    MAX_LR_FULL: float = 1.5e-3
    WD_FULL: float = 6e-4

    # 튜닝 (옵션)
    USE_OPTUNA: bool = False
    N_TRIALS: int = 30
    EPOCHS_TUNE: int = 1
    BATCH_TUNE: int = 256

    # CV folds(끝나는 주의 토요일 등) # 4-fold
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

    # ==== MLinear ====
    mlin_hidden_dim: int = 256
    mlin_dropout: float = 0.1
    mlin_use_residual: bool = True
    mlin_use_layer_norm: bool = True

    # TSMixer 하이퍼파라미터
    tsm_d_model: int = 192
    tsm_num_layers: int = 2
    tsm_dropout: float = 0.2
    tsm_patch_len: int = 3
    tsm_head_hidden: int = 256

    # ==== 훈련 보조 ====
    grad_clip: float = 0.5
    earlystop_patience_ratio: float = 0.12
    earlystop_patience_min: int = 8
    l1_lambda: float = 1e-7 
    time_mask_p: float = 0.15        # 배치 단위 적용 확률
    time_mask_max_spans: int = 2     # 윈도우 내 최대 마스킹 구간 수
    time_mask_span_frac: float = 0.12  # 각 구간 길이 = in_len * 이 비율
    
    # =======================
    # DOW feature 추가
    # =======================

    # ====== NEW: Feature toggles & params ======
    # Corr-based features
    USE_CORR_FEATURES: bool = False
    CORR_THRESHOLD: float = 0.5     # 상관 임계치
    CORR_TOPN: int = 5              # 메뉴당 상위 파트너 N개 제한 (0 = 제한 없음)
    CORR_LAGS: Tuple[int, ...] = (1, 7)
    CORR_RMEANS: Tuple[int, ...] = (7, 14)  # 모두 shift=1

    # Day-of-week strength
    USE_DOW_STRENGTH: bool = True
    DOW_RATIO_THRESHOLD: float = 2.0    # 평균 대비 2배 이상
    DOW_MIN_SUPPORT: int = 4            # 요일 평균 계산에 필요한 최소 관측일 수

    # Scaling extra features: "none" | "log1p" | "zscore"
    EXTRA_FEAT_SCALING: str = "none"

DEFAULT_STORE_WEIGHTS = {
    "미라시아": 1, "담하": 1, "연회장": 1, "라그로타": 1,
    "느티나무 셀프BBQ": 1, "화담숲주막": 1, "카페테리아": 1,
    "화담숲카페": 1, "포레스트릿": 1,
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

# 매장을 경향성별로 페어링
def get_store_type(store_name: str) -> str:
    if store_name in ["느티나무 셀프BBQ", "연회장"]:
        return 'Special_Occasion'
    elif store_name in ["화담숲주막", "화담숲카페"]:
        return 'Forest'
    elif store_name in ["담하", "미라시아"]:
        return 'Fine_Dining'
    elif store_name in ["카페테리아", "포레스트릿"]:
        return 'Casual'
    else:
        return 'Unique_Venue'

# 주말 상대적 판매량 피처를 계산하는 함수.
def get_weekend_sales_ratio(df: pd.DataFrame) -> pd.DataFrame:

    df_copy = df.copy()
    df_copy['영업일자'] = pd.to_datetime(df_copy['영업일자'])
    df_copy['요일'] = df_copy['영업일자'].dt.weekday  # 월요일=0, 일요일=6

    # 주말(토, 일)과 주중(월~금)으로 데이터 분리
    weekend_df = df_copy[df_copy['요일'].isin([5, 6])]
    weekday_df = df_copy[~df_copy['요일'].isin([5, 6])]

    # 영업장_메뉴명별 주말/주중 평균 매출 계산
    weekend_sales = weekend_df.groupby('영업장명_메뉴명')['매출수량'].mean()
    weekday_sales = weekday_df.groupby('영업장명_메뉴명')['매출수량'].mean()

    # 주말 상대적 판매량 비율 계산
    sales_ratio = ((weekend_sales + 0.5) / (weekday_sales + 0.5)).fillna(1.0).reset_index()
    sales_ratio.rename(columns={'매출수량': 'weekend_sales_ratio'}, inplace=True)

    # 분모(평일 매출)가 0인 경우, 비율이 무한대가 되는 것을 방지
    sales_ratio['weekend_sales_ratio'] = sales_ratio['weekend_sales_ratio'].replace([float('inf'), -float('inf')], 1.0)

    return sales_ratio

# =======================
# DOW feature 추가
# =======================

def _safe_shift(a: pd.Series, n: int) -> pd.Series:
    return a.shift(n)

def _safe_rmean(a: pd.Series, w: int) -> pd.Series:
    return a.shift(1).rolling(window=w, min_periods=1).mean()

def build_corr_feature_bank(
    pivot: pd.DataFrame,              # index=date, columns=item, values=sales
    cutoff_date: pd.Timestamp,        # train_end_date (누수 방지)
    threshold: float = 0.5,
    topn: int = 5,
    lags: Tuple[int, ...] = (1, 7),
    rmeans: Tuple[int, ...] = (7, 14),
) -> Tuple[Dict[str, List[str]], Dict[str, pd.DataFrame]]:
    """상관관계 기반 lag/rolling mean 피처 생성"""
    # train 구간만으로 상관 계산
    pivot_train = pivot.loc[:cutoff_date]
    corr = pivot_train.corr(method="pearson").fillna(0.0)

    corr_map: Dict[str, List[str]] = {}
    for tgt in corr.columns:
        partners_all = corr.index[(corr[tgt] >= threshold) & (corr.index != tgt)].tolist()
        # 상관값 기준 정렬 후 상위 N 제한
        partners_all = sorted(partners_all, key=lambda it: corr.loc[it, tgt], reverse=True)
        if topn and topn > 0:
            partners_all = partners_all[:topn]
        corr_map[tgt] = partners_all

    feat_bank: Dict[str, pd.DataFrame] = {}
    for tgt in pivot.columns:
        df_list = []
        for partner in corr_map.get(tgt, []):
            s = pivot[partner].astype(float)
            for L in lags:
                df_list.append(_safe_shift(s, L).rename(f"{partner}_lag{L}"))
            for W in rmeans:
                df_list.append(_safe_rmean(s, W).rename(f"{partner}_rmean{W}_lag1"))
        if df_list:
            fb = pd.concat(df_list, axis=1)
        else:
            fb = pd.DataFrame(index=pivot.index)
        feat_bank[tgt] = fb
    return corr_map, feat_bank

def compute_dow_strength_flags(df: pd.DataFrame,
                               date_col: str, item_col: str, target_col: str,
                               cutoff_date: pd.Timestamp,
                               ratio_threshold: float = 2.0,
                               min_support: int = 4) -> pd.DataFrame:
    """요일별 강세 플래그 계산"""
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp = tmp.loc[tmp[date_col] <= cutoff_date]  # 누수 방지
    tmp["dow"] = tmp[date_col].dt.weekday

    overall = tmp.groupby(item_col)[target_col].mean()
    by_dow = tmp.groupby([item_col, "dow"])[target_col].agg(['mean','count']).unstack("dow")
    # 구조: columns MultiIndex [('mean',0..6), ('count',0..6)]
    mean_mat = by_dow['mean'].fillna(0.0)
    cnt_mat  = by_dow['count'].fillna(0.0)

    ratio = mean_mat.divide(overall, axis=0).fillna(0.0)
    strong = (ratio >= ratio_threshold) & (cnt_mat >= min_support)

    # 날짜 index × item 행렬 플래그 생성
    all_dates = pd.date_range(tmp[date_col].min(), df[date_col].max())
    items = tmp[item_col].unique().tolist()
    flag = pd.DataFrame(0, index=all_dates, columns=items, dtype=np.int8)
    for it in items:
        strong_dows = set(np.where(strong.loc[it].values)[0]) if it in strong.index else set()
        if not strong_dows:
            continue
        # 날짜별 dow가 strong이면 1
        dows = pd.Series(all_dates.weekday, index=all_dates)
        flag.loc[dows.isin(strong_dows), it] = 1
    return flag

# =============================================================
# cluster dict (사용자 정의 직접 입력)
# =============================================================
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

def get_menu_group(store_name: str, menu_name: str) -> str:
    """하드코딩된 딕셔너리를 사용하여 클러스터 그룹 반환 (MLinear용)"""
    item_full_name = f"{store_name}_{menu_name}"
    cluster_id = cluster_mapping.get(item_full_name, '8')
    return f"cluster_{cluster_id}"

def get_longest_off_season_feature(df: pd.DataFrame) -> pd.DataFrame:
    """비판매 기간 피처 생성"""
    df_copy = df.copy()
    df_copy['영업일자'] = pd.to_datetime(df_copy['영업일자'])

    sales_matrix = df_copy.pivot_table(
        index='영업일자', columns='영업장명_메뉴명',
        values='매출수량', fill_value=0
    )

    longest_zero_streak = {}
    for menu in sales_matrix.columns:
        is_zero_sales = (sales_matrix[menu] == 0)
        zero_streaks = is_zero_sales.cumsum() - is_zero_sales.cumsum().where(~is_zero_sales).ffill().fillna(0)
        longest_streak = zero_streaks.max()
        longest_zero_streak[menu] = int(longest_streak)

    off_season_df = pd.DataFrame(longest_zero_streak.items(),
                                columns=['영업장명_메뉴명', 'longest_off_season_days'])
    off_season_df['is_long_off_season'] = (off_season_df['longest_off_season_days'] >= 90).astype(int)

    return off_season_df

def create_regular_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """정기 휴무일 피처 생성"""
    df_copy = df.copy()
    df_copy['영업일자'] = pd.to_datetime(df_copy['영업일자'])
    df_copy['영업장명'] = df_copy['영업장명_메뉴명'].apply(lambda x: x.split('_', 1)[0])
    df_copy['연도'] = df_copy['영업일자'].dt.year
    df_copy['월'] = df_copy['영업일자'].dt.month
    df_copy['요일'] = df_copy['영업일자'].dt.day_name()
    df_copy['is_regular_holiday'] = 0

    # 라그로타 정기휴무 (월요일)
    cond_lagrotta = (
        (df_copy['영업장명'] == '라그로타') &
        (df_copy['요일'] == 'Monday') &
        (
            ((df_copy['연도'] == 2023) & (df_copy['월'].isin(range(2, 12)))) |
            ((df_copy['연도'] == 2024) & (df_copy['월'].isin(range(3, 7))))
        )
    )

    # 포레스트릿 정기휴무
    cond_forest = (
        (df_copy['영업장명'] == '포레스트릿') &
        (
            ((df_copy['연도'] == 2023) & (df_copy['월'] == 3)) |
            ((df_copy['연도'] == 2023) & (df_copy['월'] == 9) &
             (df_copy['요일'].isin(['Monday', 'Tuesday', 'Wednesday']))) |
            ((df_copy['연도'] == 2024) & (df_copy['월'] == 3)) |
            ((df_copy['영업일자'] >= '2024-05-01') &
             (df_copy['요일'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday'])))
        )
    )

    # 화담숲 계열 휴무 (겨울철 + 평상시 월요일)
    stores_hwadam = ['화담숲주막', '화담숲카페']
    cond_hwadam = (
        (df_copy['영업장명'].isin(stores_hwadam)) &
        (
            (df_copy['월'].isin([12, 1, 2, 3])) |  # 겨울철
            ((~df_copy['월'].isin([12, 1, 2, 3])) & (df_copy['요일'] == 'Monday'))  # 평상시 월요일
        )
    )

    df_copy.loc[cond_lagrotta | cond_forest | cond_hwadam, 'is_regular_holiday'] = 1
    df_copy.drop(columns=['연도', '월', '요일', '영업장명'], inplace=True)

    return df_copy

# =====================
# Feature Utils
# =====================
# build_enhanced_features 함수 수정
def build_enhanced_features(dates: List[pd.Timestamp], holidays_set: set, store_names: Optional[List[str]] = None) -> pd.DataFrame:
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
    
# =======================
# DOW feature 추가
# =======================
    df['is_regular_holiday'] = 0
    df.loc[
        ((df['date'].dt.month.isin(range(2, 12))) & (df['date'].dt.year == 2023) & (df['dow'] == 0)) |
        ((df['date'].dt.month.isin(range(3, 7)))  & (df['date'].dt.year == 2024) & (df['dow'] == 0)) |
        (df['date'].isin([pd.Timestamp('2023-03-01')])) |
        (df['date'].isin([pd.Timestamp('2023-09-01'), pd.Timestamp('2023-09-02'), pd.Timestamp('2023-09-03')]) & (df['dow'].isin([0,1,2]))) |
        (df['date'].isin([pd.Timestamp('2024-03-01')])) |
        ((df['date'] >= pd.Timestamp('2023-05-01')) & (df['dow'].isin([0,1,2,3]))) |
        (df['date'].dt.month.isin([12,1,2,3]) & df['date'].dt.year.isin([2023,2024,2025])) |
        ((~df['date'].dt.month.isin([12,1,2,3])) & (df['dow'] == 0)),
        'is_regular_holiday'
    ] = 1

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

"""### ENhancedNHiTSDataset"""

# =====================
# Dataset (MLinear용 그룹 정보 추가)
# =====================
class EnhancedNHiTSDataset(Dataset):
    def __init__(self, cfg: EnhancedNHiTSConfig, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df.copy()
        if cfg.date_col not in self.df.columns or cfg.item_col not in self.df.columns or cfg.target_col not in self.df.columns:
            raise ValueError("입력 데이터에 필요한 컬럼이 없습니다.")
        self.df[cfg.date_col] = pd.to_datetime(self.df[cfg.date_col])
        self.df[cfg.target_col] = pd.to_numeric(self.df[cfg.target_col], errors='coerce').fillna(0.0).clip(lower=0)

        # 🚨 Dense Dataset 전처리
        all_dates = pd.date_range(start=self.df[cfg.date_col].min(), end=self.df[cfg.date_col].max())
        all_items = self.df[cfg.item_col].unique()
        full_df = pd.MultiIndex.from_product([all_dates, all_items], names=[cfg.date_col, cfg.item_col]).to_frame(index=False)
        self.df = pd.merge(full_df, self.df, on=[cfg.date_col, cfg.item_col], how='left')
        self.df[cfg.target_col] = self.df[cfg.target_col].fillna(0)

        # 🚨 (추가) 주말 상대적 판매량 피처를 계산하여 병합
        weekend_ratio_df = get_weekend_sales_ratio(self.df)
        self.df = pd.merge(self.df, weekend_ratio_df, on=cfg.item_col, how='left')
        self.df['weekend_sales_ratio'].fillna(1.0, inplace=True) # 없는 경우 1.0으로 채움 (패턴 없음)

        # 🚨 (수정) 정기 휴무일 피처를 계산하여 병합
        regular_holiday_df = create_regular_holiday_features(self.df)
        self.df = pd.merge(self.df, regular_holiday_df[['영업일자', '영업장명_메뉴명', 'is_regular_holiday']], on=['영업일자', '영업장명_메뉴명'], how='left')
        self.df['is_regular_holiday'].fillna(0, inplace=True)

        pivot = self.df.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index()
        self.items = list(pivot.columns)
        self.dates = list(pivot.index)
        self.values = pivot.fillna(0.0).values.astype(np.float32)

        holidays_set = set(pd.to_datetime(cfg.custom_holidays_list)) if cfg.custom_holidays_list else set()

# =======================
# DOW feature 추가
# =======================
        # 🚨 build_enhanced_features 함수 호출
        caldf = build_enhanced_features(self.dates, holidays_set)

        # 🚨 (수정) is_regular_holiday 피처를 안전하게 가져와서 캘린더 피처에 통합
        if 'is_regular_holiday' in self.df.columns:
            try:
                pivot_regular_holiday = self.df.pivot(index=cfg.date_col, columns=cfg.item_col, values='is_regular_holiday').sort_index().mean(axis=1)
                # 인덱스 맞추기 및 결측값 처리
                pivot_regular_holiday = pivot_regular_holiday.reindex(caldf['date']).fillna(0.0)
                caldf['is_regular_holiday'] = pivot_regular_holiday.values
                print(f"✅ 정규 휴무일 피처 추가됨. cal_feats 차원: {caldf.shape[1]-1}")
            except Exception as e:
                print(f"⚠️ 정규 휴무일 피처 생성 실패: {e}")
                caldf['is_regular_holiday'] = 0.0  # 기본값으로 설정
        else:
            print("⚠️ is_regular_holiday 컬럼이 없음. 0으로 설정")
            caldf['is_regular_holiday'] = 0.0  # 기본값으로 설정

        self.cal_feats = caldf.drop(columns=["date"]).values.astype(np.float32)
        self.cal_feat_names = [c for c in caldf.columns if c != "date"]
        
        # =======================
        # DOW feature 추가
        # =======================
        

        # ===== Extra features (옵션) =====
        cutoff = pd.to_datetime(cfg.train_end_date)
        self.extra_dim = 0
        self.extra_feats_per_item: List[np.ndarray] = []

        # (A) 요일 강세 플래그
        if cfg.USE_DOW_STRENGTH:
            dow_flag = compute_dow_strength_flags(
                self.df, cfg.date_col, cfg.item_col, cfg.target_col,
                cutoff_date=cutoff, ratio_threshold=cfg.DOW_RATIO_THRESHOLD, min_support=cfg.DOW_MIN_SUPPORT
            )
            dow_flag = dow_flag.reindex(index=pivot.index, columns=pivot.columns).fillna(0).astype(np.float32)
        else:
            dow_flag = pd.DataFrame(index=pivot.index, columns=pivot.columns, data=0.0, dtype=np.float32)

        # (B) 상관 기반 lag/rolling
        if cfg.USE_CORR_FEATURES:
            corr_map, feat_bank = build_corr_feature_bank(
                pivot=pivot, cutoff_date=cutoff,
                threshold=cfg.CORR_THRESHOLD, topn=cfg.CORR_TOPN,
                lags=cfg.CORR_LAGS, rmeans=cfg.CORR_RMEANS
            )
        else:
            corr_map, feat_bank = {}, {it: pd.DataFrame(index=pivot.index) for it in self.items}

        # 아이템별 extra DataFrame 구성 + 요일 강세 1채널 추가
        for it in self.items:
            fb = feat_bank.get(it, pd.DataFrame(index=pivot.index))
            fb = fb.copy()
            fb["_dow_strong"] = dow_flag[it] if it in dow_flag.columns else 0.0
            fb = fb.reindex(index=pivot.index).fillna(0.0).astype(np.float32)

            # 스케일링 옵션
            if fb.shape[1] and self.cfg.EXTRA_FEAT_SCALING.lower() == "log1p":
                fb = np.log1p(fb)
            elif fb.shape[1] and self.cfg.EXTRA_FEAT_SCALING.lower() == "zscore":
                mu = fb.mean(axis=0).replace(0.0, 0.0)
                std = fb.std(axis=0).replace(0.0, 1.0)
                fb = (fb - mu) / std

            self.extra_feats_per_item.append(fb.values)

        if self.extra_feats_per_item and self.extra_feats_per_item[0].shape[1] > 0:
            # 열 수를 모든 아이템에서 동일하게 맞춤
            min_dim = min(arr.shape[1] for arr in self.extra_feats_per_item)
            self.extra_feats_per_item = [arr[:, :min_dim] for arr in self.extra_feats_per_item]
            self.extra_dim = min_dim
        else:
            self.extra_dim = 0

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

        menu_groups = [get_menu_group(s, m) for s, m in zip(stores, menus)]
        self.group2idx = {g: i for i, g in enumerate(sorted(set(menu_groups)))}
        self.item_group_idx = np.array([self.group2idx[g] for g in menu_groups], dtype=np.int64)
        self.n_groups = len(self.group2idx)

        # 🚨 (추가) 주말 상대적 판매량 피처를 배열로 저장
        self.item_weekend_ratio = self.df.drop_duplicates(subset=[cfg.item_col])\
                                     .set_index(cfg.item_col)\
                                     .loc[self.items, 'weekend_sales_ratio'].values.astype(np.float32)

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
        fut_cal = self.cal_feats[t0 + Lx:t0 + Lx + Ly, :]

        store_idx = self.item_store_idx[j]
        cat_idx = self.item_cat_idx[j]
        type_idx = self.item_type_idx[j]
        group_idx = self.item_group_idx[j]
        sample_w = self.sample_weights[j]

        # 🚨 (추가) 주말 상대적 판매량 피처 가져오기
        weekend_ratio = self.item_weekend_ratio[j]

        zero_mask = (y == 0).astype(np.float32)
        pos_mask = (y > 0).astype(np.float32)

        zero_ratio = np.mean(x == 0)
        weekend_factor = np.mean(past_cal[:, 12])
        
        # =======================
        # DOW feature 추가 (return 딕셔너리까지 전부 수정)
        # =======================

        if self.extra_dim > 0:
            ex_full = self.extra_feats_per_item[j]  # [T, E]
            past_ex = ex_full[t0:t0 + Lx, :]
            fut_ex  = np.zeros((Ly, self.extra_dim), dtype=np.float32)  # 미래는 0
        else:
            past_ex = np.zeros((Lx, 0), dtype=np.float32)
            fut_ex  = np.zeros((Ly, 0), dtype=np.float32)

        return {
            "x": torch.from_numpy(x_in).float(),
            "y": torch.from_numpy(y_out).float(),
            "past_cal": torch.from_numpy(past_cal).float(),
            "fut_cal": torch.from_numpy(fut_cal).float(),
            "store_idx": torch.tensor(store_idx, dtype=torch.long),
            "cat_idx": torch.tensor(cat_idx, dtype=torch.long),
            "type_idx": torch.tensor(type_idx, dtype=torch.long),
            "group_idx": torch.tensor(group_idx, dtype=torch.long),
            "sample_w": torch.tensor(sample_w, dtype=torch.float32),
            "zero_mask": torch.from_numpy(zero_mask).float(),
            "pos_mask": torch.from_numpy(pos_mask).float(),
            "past_ex": torch.from_numpy(past_ex).float(),
            "fut_ex": torch.from_numpy(fut_ex).float(),
            "zero_ratio": torch.tensor(zero_ratio, dtype=torch.float32),
            "weekend_factor": torch.tensor(weekend_factor, dtype=torch.float32),
            "weekend_ratio": torch.tensor(weekend_ratio, dtype=torch.float32), # 🚨 새로운 피처 추가
        }

"""### N-HiTS"""

# =====================
# N-HiTS (수정된 코드)
# =====================
class NHiTSBlock(nn.Module):
    """
    단일 N-HiTS 스타일 블록:
      - 입력: x [B, L]
      - Max/Average Pool → MLP → 선형 출력
      - 출력: [B, out_len] (coarse 레벨도 허용)
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,           # out_len 혹은 coarse_len
        hidden_size: int,
        n_layers: int,
        dropout: float,
        pooling_mode: str = "MaxPool1d",
        n_pool_kernel_size: int = 2,
    ):
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

        self.output_layer = nn.Linear(hidden_size, output_size)  # 최종 길이는 호출자가 정의

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L]
        x = x.unsqueeze(1)                                # [B, 1, L]
        x_pooled = self.pooling_layer(x).squeeze(1)       # [B, L']
        h = self.mlp(x_pooled)                            # [B, hidden]
        out = self.output_layer(h)                        # [B, output_size]
        return out


class EnhancedNHiTSModel(nn.Module):
    """
    여러 NHiTSBlock을 쌓고, 특정 스택(type=="trend")은
    basis proj + interpolate로 out_len에 맞춰 복원.
    또한 store/category/type/calendar(+optional weekend_ratio) 메타 피처를 통합.
    """
    def __init__(
        self,
        in_len: int,
        out_len: int,
        cal_dim: int,
        n_stores: int,
        n_categories: int,
        n_types: int,
        cfg,                               # EnhancedNHiTSConfig 호환 객체
        has_weekend_ratio: bool = False,
    ):
        super().__init__()
        self.out_len = out_len
        self.interpolation_mode = cfg.interpolation_mode  # "linear" 권장

        # 스택/다운샘플 설정 채우기
        n_blocks = cfg.n_blocks
        pool_k = (cfg.n_pool_kernel_size or [])
        if len(pool_k) < n_blocks:
            base = [2, 2, 1]
            pool_k = [base[i % len(base)] for i in range(n_blocks)]
        else:
            pool_k = pool_k[:n_blocks]

        stack_types = (cfg.stack_types or [])
        if len(stack_types) < n_blocks:
            base = ["identity", "identity", "trend"]
            stack_types = [base[i % len(base)] for i in range(n_blocks)]
        else:
            stack_types = stack_types[:n_blocks]
        self.stack_types = stack_types

        freq_ds = (cfg.n_freq_downsample or [])
        if len(freq_ds) < n_blocks:
            base = [2, 1, 1]
            freq_ds = [base[i % len(base)] for i in range(n_blocks)]
        else:
            freq_ds = freq_ds[:n_blocks]

        # 메타 임베딩/프로젝션
        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb   = nn.Embedding(n_categories, 32)
        self.type_emb  = nn.Embedding(n_types, 16)
        self.cal_proj  = nn.Linear(cal_dim, 128)

        # 선택: weekend_ratio 피처
        self.has_weekend_ratio = has_weekend_ratio
        meta_dim = 64 + 32 + 16 + 128
        if self.has_weekend_ratio:
            self.weekend_ratio_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            )
            meta_dim += 32

        # NHiTS 블록들
        self.blocks = nn.ModuleList([
            NHiTSBlock(
                input_size=in_len,
                output_size=out_len,                 # trend 스택은 아래에서 coarse로 다시 변환
                hidden_size=cfg.hidden,
                n_layers=cfg.n_layers,
                dropout=cfg.dropout,
                pooling_mode=cfg.pooling_mode,
                n_pool_kernel_size=pool_k[i],
            )
            for i in range(n_blocks)
        ])

        # trend 스택용 basis (out_len → coarse_len)
        # 실제 coarse_len은 out_len // freq_ds[i] 등으로 정의
        self.basis_weights = nn.ParameterList([
            nn.Parameter(torch.randn(out_len, max(1, out_len // max(1, freq_ds[i]))))
            for i in range(n_blocks)
        ])

        # 메타 통합 / hurdle head
        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, cfg.hidden),
            nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, out_len),
        )
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 1, cfg.hidden),   # + mean(x)
            nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden // 2),
            nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden // 2, out_len),
        )

# =======================
# DOW feature 추가
# =======================
    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx, 
            weekend_ratio=None, past_ex=None, fut_ex=None, **kwargs):
    
        # extra concat - 캘린더 피처에 추가 피처 결합
        if (past_ex is not None) and (fut_ex is not None) and (past_ex.numel() > 0):
            past_cal = torch.cat([past_cal, past_ex], dim=-1)
            fut_cal  = torch.cat([fut_cal,  fut_ex ], dim=-1)
            
        # 메타 피처 준비
        store_emb = self.store_emb(store_idx)
        cat_emb   = self.cat_emb(cat_idx)
        type_emb  = self.type_emb(type_idx)
        
        # 캘린더 피처 처리 (past + future 결합 후 평균) - 이 부분이 빠져있었음
        cal_all   = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)  # [B, cal_dim + extra_dim]
        cal_emb   = self.cal_proj(cal_all)

        if self.has_weekend_ratio and (weekend_ratio is not None):
            wr_emb  = self.weekend_ratio_mlp(weekend_ratio.unsqueeze(1))
            meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb, wr_emb], dim=-1)
        else:
            meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb], dim=-1)

        # 스택 출력 합성
        outputs = []
        for i, block in enumerate(self.blocks):
            block_output = block(x)  # [B, out_len] (coarse 아님)

            if self.stack_types[i] == "trend":
                # basis: [out_len, coarse_len]
                basis = self.basis_weights[i]
                # block_output @ basis → [B, coarse_len]
                coarse = block_output @ basis
                # coarse → out_len 로 보간
                if self.interpolation_mode in ("linear", "bilinear", "bicubic", "trilinear"):
                    align = False
                else:
                    align = None
                block_output = F.interpolate(
                    coarse.unsqueeze(1), size=self.out_len,
                    mode=self.interpolation_mode, align_corners=align
                ).squeeze(1)  # [B, out_len]

            outputs.append(block_output)

        nhits_output = torch.stack(outputs, dim=0).sum(dim=0)   # [B, out_len]

        # 메타 보정 및 hurdle 확률
        meta_adjustment = self.meta_integration(meta_feat)       # [B, out_len]
        value_pred = nhits_output + meta_adjustment              # [B, out_len]

        prob_feat = torch.cat([meta_feat, x.mean(dim=1, keepdim=True)], dim=-1)
        prob_logits = self.prob_head(prob_feat)                  # [B, out_len]
        return value_pred, prob_logits

"""### PatchTST"""

# =====================
# PatchTST (수정된 코드)
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
        pe = _build_sinusoidal_pos(Np, self.d_model, x_e.device, x_e.dtype)  # [1,Np,D]
        x_e = x_e + pe
        return x_e

class ImprovedPatchTSTTiny(nn.Module):
    def __init__(self, cfg: EnhancedNHiTSConfig, in_len: int, out_len: int,
                 cal_dim: int, n_stores: int, n_categories: int, n_types: int, has_weekend_ratio: bool = False): # 🚨 (수정) has_weekend_ratio 인자 추가
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

        # 🚨 (추가) weekend_ratio를 위한 MLP를 조건부로 생성
        self.has_weekend_ratio = has_weekend_ratio
        meta_dim = 64 + 32 + 16 + 128 + max(16, cfg.ptst_d_model // 2)
        if self.has_weekend_ratio:
            self.weekend_ratio_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.GELU(),
                nn.Dropout(cfg.ptst_dropout)
            )
            meta_dim += 32

        # 🚨 (수정) meta_dim에 weekend_ratio_mlp의 출력 차원(32) 추가
        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, cfg.ptst_head_hidden),
            nn.GELU(), nn.Dropout(cfg.ptst_dropout),
            nn.Linear(cfg.ptst_head_hidden, max(32, cfg.ptst_head_hidden // 2)),
            nn.GELU(), nn.Dropout(cfg.ptst_dropout),
            nn.Linear(max(32, cfg.ptst_head_hidden // 2), out_len)
        )
        self.prob_head = nn.Sequential(
            # 🚨 (수정) prob_head 입력 차원에 weekend_ratio_mlp의 출력 차원(32) 추가
            nn.Linear(meta_dim + 1, cfg.ptst_head_hidden),
            nn.GELU(), nn.Dropout(cfg.ptst_dropout),
            nn.Linear(cfg.ptst_head_hidden, max(32, cfg.ptst_head_hidden // 2)),
            nn.GELU(), nn.Dropout(cfg.ptst_dropout),
            nn.Linear(max(32, cfg.ptst_head_hidden // 2), out_len)
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx, weekend_ratio=None, past_ex=None, fut_ex=None, **kwargs):
    # =======================
# DOW feature 추가
# =======================
        # extra concat - 캘린더 피처에 추가 피처 결합
        if (past_ex is not None) and (fut_ex is not None) and (past_ex.numel() > 0):
            past_cal = torch.cat([past_cal, past_ex], dim=-1)
            fut_cal  = torch.cat([fut_cal,  fut_ex ], dim=-1)
        
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

        if self.has_weekend_ratio and weekend_ratio is not None:
            weekend_ratio_emb = self.weekend_ratio_mlp(weekend_ratio.unsqueeze(1))
            meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb, time_emb, weekend_ratio_emb], dim=-1)
        else:
            meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb, time_emb], dim=-1)

        meta_adjustment = self.meta_integration(meta_feat)
        value_pred = seq_out + self.residual_weight * meta_adjustment

        prob_logits = self.prob_head(torch.cat([meta_feat, x.mean(dim=1, keepdim=True)], dim=-1))
        return value_pred, prob_logits

"""### TimesNet"""

# =====================
# Improved TimesNet (수정된 코드)
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
        x_filt = x_filt32.to(in_dtype)
        return x_filt

class ImprovedTimesBlock(nn.Module):
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

        self.freq_block = FrequencyBlock(seq_len=self.seq_len, top_k=top_k)
        self.period_base = max(2, self.seq_len // 4)

        self.conv2d = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=(3, 3), padding=1),
            nn.GELU(),
        )

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

        self.channel_proj = (
            nn.Conv1d(self.br_sum_ch, self.out_ch, kernel_size=1)
            if self.br_sum_ch != self.out_ch
            else nn.Identity()
        )

        self.norm = nn.LayerNorm([self.out_ch, self.seq_len])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        assert C == self.in_ch, f"in_ch mismatch: got {C}, expected {self.in_ch}"

        xf = self.freq_block(x)

        period = self.period_base
        if L % period != 0:
            pad_len = period - (L % period)
            xf = F.pad(xf, (0, pad_len), mode="replicate")
            L_pad = L + pad_len
        else:
            pad_len = 0
            L_pad = L

        x2d = xf.reshape(B, C, period, L_pad // period).contiguous()
        z2d = self.conv2d(x2d)
        z2d_flat = z2d.reshape(B, self.out_ch, L_pad)
        if pad_len > 0:
            z2d_flat = z2d_flat[:, :, :L]

        z1d_list = [branch(x) for branch in self.conv1d_branches]
        z1d = torch.cat(z1d_list, dim=1)
        z1d = self.channel_proj(z1d)

        out = z2d_flat + z1d
        out = self.norm(out)
        out = self.dropout(out)
        return out

class ImprovedTimesNetTiny(nn.Module):
    def __init__(self, cfg: EnhancedNHiTSConfig, in_len: int, out_len: int,
                 cal_dim: int, n_stores: int, n_categories: int, n_types: int, has_weekend_ratio: bool = False): # 🚨 (수정) has_weekend_ratio 인자 추가
        super().__init__()
        C = max(32, min(cfg.tnet_channels, 96))
        C = (C // 8) * 8

        self.stem = nn.Conv1d(1, C, kernel_size=3, padding=1)

        def fast_block(C: int, p: float):
            return nn.Sequential(
                nn.Conv1d(C, C, kernel_size=3, padding=1, groups=C, bias=True),
                nn.Conv1d(C, C, kernel_size=1, bias=False),
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

        head_hidden = max(128, min(cfg.tnet_head_hidden, 192))
        self.head_fc1 = nn.Linear(C, head_hidden)
        self.head_fc2 = nn.Linear(head_hidden, out_len)
        self.head_drop = nn.Dropout(cfg.tnet_dropout)

        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb = nn.Embedding(n_categories, 32)
        self.type_emb = nn.Embedding(n_types, 16)
        self.cal_proj = nn.Linear(cal_dim, 128)

        # 🚨 (추가) weekend_ratio를 위한 MLP를 조건부로 생성
        self.has_weekend_ratio = has_weekend_ratio
        meta_dim = 64 + 32 + 16 + 128
        if self.has_weekend_ratio:
            self.weekend_ratio_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Dropout(cfg.tnet_dropout)
            )
            meta_dim += 32

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

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx, weekend_ratio=None, past_ex=None, fut_ex=None, **kwargs):
    # =======================
# DOW feature 추가
# =======================
        # extra concat - 캘린더 피처에 추가 피처 결합
        if (past_ex is not None) and (fut_ex is not None) and (past_ex.numel() > 0):
            past_cal = torch.cat([past_cal, past_ex], dim=-1)
            fut_cal  = torch.cat([fut_cal,  fut_ex ], dim=-1)
        
        z = self.stem(x.unsqueeze(1))
        for blk in self.blocks:
            z = z + blk(z)

        z_mean = z.mean(dim=-1)
        seq_out = self.head_fc2(self.head_drop(F.gelu(self.head_fc1(z_mean))))

        store_emb = self.store_emb(store_idx)
        cat_emb = self.cat_emb(cat_idx)
        type_emb = self.type_emb(type_idx)
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)
        cal_emb = self.cal_proj(cal_all)

        if self.has_weekend_ratio and weekend_ratio is not None:
            weekend_ratio_emb = self.weekend_ratio_mlp(weekend_ratio.unsqueeze(1))
            meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb, weekend_ratio_emb], dim=-1)
        else:
            meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb], dim=-1)

        value_pred = seq_out + self.meta_res_weight * self.meta_integration(meta_feat)
        prob_logits = self.prob_head(torch.cat([meta_feat, x.mean(dim=1, keepdim=True)], dim=-1))
        return value_pred, prob_logits

"""### GRU"""

# =====================
# GRU (수정된 코드)
# =====================
class GRUTiny(nn.Module):
    def __init__(self, cfg: EnhancedNHiTSConfig, in_len: int, out_len: int,
                 cal_dim: int, n_stores: int, n_categories: int, n_types: int, has_weekend_ratio: bool = False): # 🚨 (수정) has_weekend_ratio 인자 추가
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
        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb = nn.Embedding(n_categories, 32)
        self.type_emb = nn.Embedding(n_types, 16)
        self.cal_proj = nn.Linear(cal_dim, 128)

        # 🚨 (추가) weekend_ratio를 위한 MLP를 조건부로 생성
        self.has_weekend_ratio = has_weekend_ratio
        meta_dim = 64 + 32 + 16 + 128
        if self.has_weekend_ratio:
            self.weekend_ratio_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            meta_dim += 32

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

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx, weekend_ratio=None, past_ex=None, fut_ex=None, **kwargs):
    # =======================
# DOW feature 추가
# =======================
        # extra concat - 캘린더 피처에 추가 피처 결합
        if (past_ex is not None) and (fut_ex is not None) and (past_ex.numel() > 0):
            past_cal = torch.cat([past_cal, past_ex], dim=-1)
            fut_cal  = torch.cat([fut_cal,  fut_ex ], dim=-1)
        
        z_in = x.unsqueeze(-1)
        z_out, _ = self.gru(z_in)
        z_last = z_out[:, -1, :]
        seq_out = self.head(z_last)
        
        store_emb = self.store_emb(store_idx)
        cat_emb = self.cat_emb(cat_idx)
        type_emb = self.type_emb(type_idx)
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)
        cal_emb = self.cal_proj(cal_all)

        if self.has_weekend_ratio and weekend_ratio is not None:
            weekend_ratio_emb = self.weekend_ratio_mlp(weekend_ratio.unsqueeze(1))
            meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb, weekend_ratio_emb], dim=-1)
        else:
            meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb], dim=-1)

        value_pred = seq_out + self.meta_integration(meta_feat)
        prob_logits = self.prob_head(torch.cat([meta_feat, x.mean(dim=1, keepdim=True)], dim=-1))
        return value_pred, prob_logits

"""### MLinear"""

# =====================
# MLinear (수정된 코드)
# =====================
class MLinearTiny(nn.Module):
    """MLinear 모델 - 그룹별 선형 변환 + 메타 피처 통합"""

    def __init__(self, cfg: EnhancedNHiTSConfig, in_len: int, out_len: int,
                 cal_dim: int, n_stores: int, n_categories: int, n_types: int, n_groups: int, has_weekend_ratio: bool = False): # 🚨 (수정) has_weekend_ratio 인자 추가
        super().__init__()

        self.in_len = in_len
        self.out_len = out_len
        self.cfg = cfg

        # 임베딩
        self.group_emb = nn.Embedding(n_groups, 32)
        self.store_emb = nn.Embedding(n_stores, 32)
        self.cat_emb = nn.Embedding(n_categories, 16)
        self.type_emb = nn.Embedding(n_types, 16)

        # 캘린더 피처 처리
        self.cal_proj = nn.Sequential(
            nn.Linear(cal_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(cfg.mlin_dropout * 0.5),
            nn.Linear(128, 64)
        )

        # 🚨 (추가) weekend_ratio를 위한 MLP를 조건부로 생성
        self.has_weekend_ratio = has_weekend_ratio
        meta_dim = 32 + 32 + 16 + 16 + 64
        if self.has_weekend_ratio:
            self.weekend_ratio_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Dropout(cfg.mlin_dropout)
            )
            meta_dim += 32

        # 그룹별 MLinear
        self.group_linears = nn.ModuleDict({
            f'group_{i}': nn.Linear(in_len, out_len) for i in range(n_groups)
        })

        # 공통 MLinear
        self.common_linear = nn.Linear(in_len, out_len)

        # 메타 피처 통합
        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, cfg.mlin_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.mlin_dropout),
            nn.Linear(cfg.mlin_hidden_dim, out_len)
        )

        # 가중치 네트워크
        self.weight_net = nn.Sequential(
            nn.Linear(meta_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

        # Hurdle 모델
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 2, cfg.mlin_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.mlin_dropout),
            nn.Linear(cfg.mlin_hidden_dim, cfg.mlin_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.mlin_hidden_dim // 2, out_len)
        )

        # 옵션 기능들
        if cfg.mlin_use_residual:
            self.residual_proj = nn.Linear(in_len, out_len)

        if cfg.mlin_use_layer_norm:
            self.output_norm = nn.LayerNorm(out_len)

# =======================
# DOW feature 추가
# =======================
    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx, group_idx,
            zero_ratio, weekend_factor, weekend_ratio=None, past_ex=None, fut_ex=None, **kwargs):
    
        # extra concat - 캘린더 피처에 추가 피처 결합
        if (past_ex is not None) and (fut_ex is not None) and (past_ex.numel() > 0):
            past_cal = torch.cat([past_cal, past_ex], dim=-1)
            fut_cal  = torch.cat([fut_cal,  fut_ex ], dim=-1)
        
        batch_size = x.size(0)

        # 임베딩
        group_emb = self.group_emb(group_idx)
        store_emb = self.store_emb(store_idx)
        cat_emb = self.cat_emb(cat_idx)
        type_emb = self.type_emb(type_idx)

        # 캘린더 피처
        past_cal_mean = past_cal.mean(dim=1)
        fut_cal_mean = fut_cal.mean(dim=1)
        cal_combined = torch.cat([past_cal_mean, fut_cal_mean], dim=1)
        cal_emb = self.cal_proj(cal_combined)

        if self.has_weekend_ratio and weekend_ratio is not None:
            weekend_ratio_emb = self.weekend_ratio_mlp(weekend_ratio.unsqueeze(1))
            meta_feat = torch.cat([group_emb, store_emb, cat_emb, type_emb, cal_emb, weekend_ratio_emb], dim=-1)
        else:
            meta_feat = torch.cat([group_emb, store_emb, cat_emb, type_emb, cal_emb], dim=-1)

        # 그룹별 MLinear
        group_output = torch.zeros(batch_size, self.out_len, device=x.device)
        for i in range(len(self.group_linears)):
            mask = (group_idx == i).float().unsqueeze(-1)
            group_pred = self.group_linears[f'group_{i}'](x)
            group_output += group_pred * mask

        # 공통 MLinear
        common_output = self.common_linear(x)

        # 메타 조정
        meta_output = self.meta_integration(meta_feat)

        # 가중치 계산
        weight_input = torch.cat([meta_feat, zero_ratio.unsqueeze(-1), weekend_factor.unsqueeze(-1)], dim=-1)
        weights = self.weight_net(weight_input)
        w_group, w_common, w_meta = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        # 최종 예측
        value_pred = w_group * group_output + w_common * common_output + w_meta * meta_output

        # Residual
        if self.cfg.mlin_use_residual:
            value_pred = value_pred + self.residual_proj(x)

        # 정규화
        if self.cfg.mlin_use_layer_norm:
            value_pred = self.output_norm(value_pred)

        # Hurdle 확률
        x_stats = torch.cat([x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True)], dim=-1)
        prob_input = torch.cat([meta_feat, x_stats], dim=-1)
        prob_logits = self.prob_head(prob_input)

        return value_pred, prob_logits

"""### TSMixer"""

# =====================
# TSMixer (수정된 코드)
# =====================
class TSMixerBlock(nn.Module):
    def __init__(self, d_model: int, n_patches: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mixing = nn.Sequential(
            nn.Linear(n_patches, n_patches),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mixing = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        z = self.norm1(x)
        z = self.token_mixing(z.transpose(1, 2)).transpose(1, 2)
        x = x + z
        z = self.norm2(x)
        z = self.channel_mixing(z)
        x = x + z
        return x

class TSMixer(nn.Module):
    """TSMixer 모델 - 패치/채널 MLP 믹싱 + 메타 피처 통합"""

    def __init__(self, cfg: EnhancedNHiTSConfig, in_len: int, out_len: int,
                 cal_dim: int, n_stores: int, n_categories: int, n_types: int,
                 n_groups: int, d_model: int, has_weekend_ratio: bool = False): # 🚨 (수정) has_weekend_ratio 인자 추가
        super().__init__()

        self.in_len = in_len
        self.out_len = out_len
        self.cfg = cfg
        self.d_model = d_model

        patch_len = max(2, min(cfg.tsm_patch_len, in_len))
        stride = max(1, patch_len // 2)
        self.patch_len = patch_len
        self.stride = stride

        if in_len < patch_len:
            self.n_patches = 1
        else:
            self.n_patches = math.floor((in_len - patch_len) / stride) + 1

        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        self.mixer_blocks = nn.ModuleList([
            TSMixerBlock(self.d_model, self.n_patches, cfg.tsm_dropout)
            for _ in range(cfg.tsm_num_layers)
        ])

        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb = nn.Embedding(n_categories, 32)
        self.type_emb = nn.Embedding(n_types, 16)

        self.cal_proj = nn.Sequential(
            nn.Linear(cal_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # 🚨 (추가) weekend_ratio를 위한 MLP를 조건부로 생성
        self.has_weekend_ratio = has_weekend_ratio
        meta_dim = 64 + 32 + 16 + 64
        if self.has_weekend_ratio:
            self.weekend_ratio_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Dropout(cfg.tsm_dropout)
            )
            meta_dim += 32

        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, cfg.tsm_head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.tsm_dropout),
            nn.Linear(cfg.tsm_head_hidden, out_len)
        )

        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 1, cfg.tsm_head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.tsm_dropout),
            nn.Linear(cfg.tsm_head_hidden, out_len)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model * self.n_patches),
            nn.Linear(self.d_model * self.n_patches, max(64, cfg.tsm_head_hidden)),
            nn.GELU(),
            nn.Dropout(cfg.tsm_dropout),
            nn.Linear(max(64, cfg.tsm_head_hidden), out_len)
        )

# =======================
# DOW feature 추가
# =======================
    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx, weekend_ratio=None, past_ex=None, fut_ex=None, **kwargs):
    
        # extra concat - 캘린더 피처에 추가 피처 결합
        if (past_ex is not None) and (fut_ex is not None) and (past_ex.numel() > 0):
            past_cal = torch.cat([past_cal, past_ex], dim=-1)
            fut_cal  = torch.cat([fut_cal,  fut_ex ], dim=-1)
        
        if self.in_len < self.patch_len:
            pad_len = self.patch_len - self.in_len
            x = F.pad(x, (0, pad_len), mode='replicate')

        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        z = self.patch_embedding(patches)

        for block in self.mixer_blocks:
            z = block(z)

        z_global = z.reshape(z.size(0), -1)
        seq_out = self.head(z_global)

        store_emb = self.store_emb(store_idx)
        cat_emb = self.cat_emb(cat_idx)
        type_emb = self.type_emb(type_idx)

        cal_all = torch.cat([past_cal, fut_cal], dim=1)
        cal_all = cal_all.mean(dim=1)
        cal_emb = self.cal_proj(cal_all)

        if self.has_weekend_ratio and weekend_ratio is not None:
            weekend_ratio_emb = self.weekend_ratio_mlp(weekend_ratio.unsqueeze(1))
            meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb, weekend_ratio_emb], dim=-1)
        else:
            meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb], dim=-1)

        meta_adjustment = self.meta_integration(meta_feat)
        value_pred = seq_out + meta_adjustment

        prob_feat = torch.cat([meta_feat, x.mean(dim=1, keepdim=True)], dim=-1)
        prob_logits = self.prob_head(prob_feat)

        return value_pred, prob_logits

class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, d=1, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding='same'), nn.GELU(), nn.Dropout(p),
            nn.Conv1d(c_out, c_out, kernel_size=k, dilation=d, padding='same'), nn.GELU(),
        )
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in!=c_out else nn.Identity()
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x):
        y = self.net(x) + self.proj(x)
        return self.norm(y.transpose(1,2)).transpose(1,2)

class TCNTiny(nn.Module):
    def __init__(self, cfg, in_len, out_len, cal_dim, n_stores, n_categories, n_types,
                 C=128, depth=4, drop=0.1, has_weekend_ratio: bool = False): # 🚨 (수정) has_weekend_ratio 인자 추가
        super().__init__()
        self.stem = nn.Conv1d(1, C, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([_TCNBlock(C, C, k=3, d=2**i, p=drop) for i in range(depth)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(C, out_len))
        
        self.store_emb = nn.Embedding(n_stores,64)
        self.cat_emb = nn.Embedding(n_categories,32)
        self.type_emb = nn.Embedding(n_types,16)
        self.cal_proj = nn.Linear(cal_dim,128)
        
        # 🚨 (추가) weekend_ratio를 위한 MLP를 조건부로 생성
        self.has_weekend_ratio = has_weekend_ratio
        meta_dim = 64 + 32 + 16 + 128
        if self.has_weekend_ratio:
            self.weekend_ratio_mlp = nn.Sequential(
                nn.Linear(1, 32), 
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            meta_dim += 32
        
        # 🚨 (수정) meta_head와 prob_head의 입력 차원을 동적으로 조정
        self.meta_head = nn.Sequential(
            nn.Linear(meta_dim, 256), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(256, out_len)
        )
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 1, 256), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(256, out_len)
        )
        
    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx, weekend_ratio=None, past_ex=None, fut_ex=None, **kwargs):
    # =======================
# DOW feature 추가
# =======================
        # extra concat - 캘린더 피처에 추가 피처 결합
        if (past_ex is not None) and (fut_ex is not None) and (past_ex.numel() > 0):
            past_cal = torch.cat([past_cal, past_ex], dim=-1)
            fut_cal  = torch.cat([fut_cal,  fut_ex ], dim=-1)
        
        z = self.stem(x.unsqueeze(1))
        for blk in self.blocks: z = blk(z)
        seq_out = self.head(z)
        
        store = self.store_emb(store_idx)
        cat = self.cat_emb(cat_idx)
        typ = self.type_emb(type_idx)
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)
        cal = self.cal_proj(cal_all)
        
        if self.has_weekend_ratio and weekend_ratio is not None:
            weekend_ratio_emb = self.weekend_ratio_mlp(weekend_ratio.unsqueeze(1))
            meta = torch.cat([store, cat, typ, cal, weekend_ratio_emb], dim=-1)
        else:
            meta = torch.cat([store, cat, typ, cal], dim=-1)
            
        value_pred  = seq_out + self.meta_head(meta)
        prob_logits = self.prob_head(torch.cat([meta, x.mean(dim=1, keepdim=True)], dim=-1))
        return value_pred, prob_logits
# =====================
# Loss / EMA (기존과 동일)
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
# Trainer (수정된 코드)
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

    # =======================
    # DOW feature 추가
    # =======================
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

                # 🚨 (추가) weekend_ratio 가져오기
                weekend_ratio = batch["weekend_ratio"].to(self.device)

                # MLinear용 추가 입력 (이미 전달한 파라미터 제외)
                excluded_keys = {"x", "y", "past_cal", "fut_cal", "store_idx", "cat_idx", "type_idx", "sample_w", "pos_mask", "weekend_ratio", "past_ex", "fut_ex"}
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items() if k not in excluded_keys}

                # 🚨 (수정) 모델에 weekend_ratio 및 모든 필요한 인수 전달
                v_pred, p_logits = self.model(
                    x, past_cal, fut_cal, store_idx, cat_idx, type_idx,
                    weekend_ratio=weekend_ratio,
                    past_ex=batch.get("past_ex", None).to(self.device) if "past_ex" in batch else None,
                    fut_ex=batch.get("fut_ex", None).to(self.device) if "fut_ex" in batch else None,
                    **batch_dict  # 이제 group_idx, zero_ratio, weekend_factor 등이 포함됨
                )

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
    
    def _apply_time_mask(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]
        if self.cfg.time_mask_p <= 0 or random.random() > self.cfg.time_mask_p:
            return x
        B, L = x.size()
        span_len = max(1, int(L * self.cfg.time_mask_span_frac))
        out = x.clone()
        for b in range(B):
            for _ in range(self.cfg.time_mask_max_spans):
                s = random.randint(0, max(0, L - span_len))
                e = s + span_len
                local_mean = x[b, max(0, s-3):min(L, e+3)].mean()
                out[b, s:e] = local_mean
        return out

    def train_with_loaders(self, train_loader: DataLoader, val_loader: DataLoader, model_name="Model"):
        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.base_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        steps_per_epoch = max(1, len(train_loader))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=self.max_lr,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.05,
            div_factor=max(1e-8, self.max_lr / max(1e-8, self.base_lr)),
        )

        patience = max(self.cfg.earlystop_patience_min, int(self.epochs * self.cfg.earlystop_patience_ratio))
        best_val = float("inf")
        best_state = None
        best_shadow = None
        best_epoch = None
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (
                self.cfg.use_amp and torch.cuda.is_available() and self.device.type == "cuda"
            ) else torch.cuda.amp.autocast(enabled=False)

            running = []
            for batch in train_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                past_cal = batch["past_cal"].to(self.device)
                fut_cal = batch["fut_cal"].to(self.device)
                store_idx = batch["store_idx"].to(self.device)
                cat_idx = batch["cat_idx"].to(self.device)
                type_idx = batch["type_idx"].to(self.device)
                sample_w = batch["sample_w"].to(self.device)
                pos_mask = batch["pos_mask"].to(self.device)
                weekend_ratio = batch["weekend_ratio"].to(self.device)

# =======================
# DOW feature 추가
# =======================

                # MLinear용 추가 입력 (이미 전달한 파라미터 제외)
                excluded_keys = {
                    "x", "y", "past_cal", "fut_cal",
                    "store_idx", "cat_idx", "type_idx",
                    "sample_w", "pos_mask", "weekend_ratio",
                    "past_ex", "fut_ex"  # 이 두 줄을 추가
                }
                batch_dict = {
                    k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items() if k not in excluded_keys
                }

                with amp_ctx:
                    # 간단 증강: 시간 마스킹
                    x_in = self._apply_time_mask(x)

                    # =======================
                    # DOW feature 추가
                    # =======================
                    # Forward
                    v_pred, p_logits = self.model(
                        x_in, past_cal, fut_cal, store_idx, cat_idx, type_idx,
                        weekend_ratio=weekend_ratio,
                        past_ex=batch.get("past_ex", None).to(self.device) if "past_ex" in batch else None,
                        fut_ex=batch.get("fut_ex", None).to(self.device) if "fut_ex" in batch else None,
                        **batch_dict
                    )

                    # 기본 손실
                    base_loss, _, _ = self.criterion(v_pred, p_logits, y, pos_mask, sample_w)

                    # L1 정규화 (bias/Norm 제외, AMP에서도 안전하게 FP32로 누적)
                    if self.cfg.l1_lambda > 0:
                        l1 = torch.zeros((), device=self.device, dtype=torch.float32)
                        for n, p in self.model.named_parameters():
                            if p.requires_grad and ("bias" not in n) and ("norm" not in n) and ("bn" not in n):
                                l1 = l1 + p.float().abs().sum()
                        base_loss = base_loss + self.cfg.l1_lambda * l1.to(base_loss.dtype)

                    loss = base_loss  # ← criterion 재호출 금지

                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                optim.step()
                sched.step()
                self.ema.update(self.model)

                running.append(loss.detach().item())

            # ----- epoch 종료: 검증 & early stopping -----
            val_loss = self.evaluate(val_loader, use_ema=True)
            tr_mean = float(np.mean(running)) if running else float("nan")
            print(f"[{model_name}] [Epoch {epoch:03d}] train_loss: {tr_mean:.5f}  val_loss: {val_loss:.5f}")

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                best_shadow = {k: v.detach().cpu().clone() for k, v in self.ema.shadow.items()}  # EMA도 함께 저장
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[{model_name}] Early stopping at epoch {epoch}")
                    break

        # ----- 학습 종료: best 스냅샷 복구 -----
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if best_shadow is not None:
            # EMA도 best 시점으로 복구
            self.ema.shadow = {k: v.to(self.device) for k, v in best_shadow.items()}
            self.ema.apply_to(self.model)
        else:
            # 베스트 EMA가 없으면 현재 shadow 그대로 적용
            self.ema.apply_to(self.model)

        if best_epoch is not None:
            print(f"[{model_name}] Best val={best_val:.5f} @ epoch {best_epoch}")

        return self.model, best_val

# === H-wise weighting utils ===
def smape(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """horizon별 sMAPE (axis=0이 horizon) -> shape [H]"""
    denom = (np.abs(a) + np.abs(b)).clip(eps, None)
    return (2.0 * np.abs(a - b) / denom).mean(axis=0)

@torch.no_grad()
def eval_hwise(trainer, val_loader) -> np.ndarray:
    """
    현재 trainer.model로 val_loader에서 D+1..H horizon별 sMAPE 계산,
    역손실 정규화로 가중치 벡터 반환. shape [H]
    """
    M = trainer.model; M.eval()
    device = trainer.device
    yh_list, yt_list = [], []

    amp = torch.autocast(device_type='cuda', dtype=torch.float16) if (
        trainer.cfg.use_amp and torch.cuda.is_available()
    ) else torch.cuda.amp.autocast(enabled=False)

    with amp:
        for batch in val_loader:
            x = batch["x"].to(device)
            ylog = batch["y"].to(device)
            past_cal = batch["past_cal"].to(device)
            fut_cal  = batch["fut_cal"].to(device)
            si = batch["store_idx"].to(device)
            ci = batch["cat_idx"].to(device)
            ti = batch["type_idx"].to(device)

            weekend_ratio = batch["weekend_ratio"].to(device) if "weekend_ratio" in batch else None

            excluded = {"x","y","past_cal","fut_cal","store_idx","cat_idx","type_idx","sample_w","pos_mask","weekend_ratio"}
            extra = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k,v in batch.items() if k not in excluded}

            v_log, p_logits = M(x, past_cal, fut_cal, si, ci, ti, weekend_ratio=weekend_ratio, **extra)
            y_val = torch.expm1(ylog).clamp_min(0).cpu().numpy()                    # [B,H]
            y_hat = (torch.sigmoid(p_logits)*torch.expm1(v_log)).clamp_min(0).cpu().numpy()  # [B,H]
            yt_list.append(y_val); yh_list.append(y_hat)

    yt = np.concatenate(yt_list, axis=0)   # [N,H]
    yh = np.concatenate(yh_list, axis=0)   # [N,H]
    h_sm = smape(yt, yh)                   # [H]
    inv  = 1.0 / np.clip(h_sm, 1e-6, None)
    w    = inv / inv.sum()                 # 정규화
    return w.astype(np.float64)

# =====================
# Predict Utils (수정된 코드)
# =====================
# =======================
# DOW feature 추가
# =======================

# predict_one_file_generic 함수 수정 부분만 제공
# DOW_only_0302.py의 해당 함수를 아래 코드로 교체하세요

@torch.no_grad()
def predict_one_file_generic(
    cfg: EnhancedNHiTSConfig,
    model: nn.Module,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    dataset: EnhancedNHiTSDataset,  # ✅ 추가: dataset 객체 전달
    store2idx: Dict,
    cat2idx: Dict,
    type2idx: Dict,
    group2idx: Dict = None
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

    stores = [parse_store_name(it) for it in items]

    # ✅ model5_v6.py 방식으로 캘린더 피처 생성
    past_cal_base = build_enhanced_features(dates[-Lx:], holidays_set)
    fut_cal_base = build_enhanced_features(future_dates, holidays_set)
    
    # 🐛 DEBUG: 기본 캘린더 피처 확인
    # print(f"📅 [DEBUG] past_cal_base shape: {past_cal_base.shape}, columns: {list(past_cal_base.columns)}")
    # print(f"📅 [DEBUG] fut_cal_base shape: {fut_cal_base.shape}, columns: {list(fut_cal_base.columns)}")
    
    # ✅ 훈련 시 사용된 피처 차원 확인
    expected_cal_dim = dataset.cal_feats.shape[1]
    expected_extra_dim = getattr(dataset, 'extra_dim', 0)
    expected_total_dim = expected_cal_dim + expected_extra_dim

    # is_regular_holiday가 기본 캘린더 피처에 포함되어 있는지 확인
    has_reg_holiday_in_base = 'is_regular_holiday' in past_cal_base.columns

    # ✅ 정기 휴무일 피처를 생성 및 병합 (model5_v6.py 방식)
    all_df = pd.concat([train_df, tdf], axis=0)
    
    try:
        regular_holiday_features = create_regular_holiday_features(all_df)

        # is_regular_holiday 값 분포 확인
        if 'is_regular_holiday' in regular_holiday_features.columns:
            holiday_count = regular_holiday_features['is_regular_holiday'].sum()
            total_count = len(regular_holiday_features)
        
        reg_hol_pivot = regular_holiday_features.pivot_table(
            index='영업일자', columns='영업장명_메뉴명', 
            values='is_regular_holiday'
        ).fillna(0)
        
    except Exception as e:
        print(f"❌ [ERROR] 정기 휴무일 피처 생성 실패: {e}")
        print(f"🔄 [DEBUG] Using fallback: setting all is_regular_holiday to 0")
        # Fallback: 모든 아이템에 대해 0으로 설정
        reg_hol_pivot = pd.DataFrame(
            0, 
            index=pd.concat([past_cal_base['date'], fut_cal_base['date']]), 
            columns=items
        )
        regular_holiday_features = None

    # past_cal 확장 및 병합
    past_cal_expanded = past_cal_base.reindex(past_cal_base.index.repeat(len(items)))
    past_cal_expanded['영업장명_메뉴명'] = list(items) * len(past_cal_base)

    # future_cal 확장 및 병합  
    fut_cal_expanded = fut_cal_base.reindex(fut_cal_base.index.repeat(len(items)))
    fut_cal_expanded['영업장명_메뉴명'] = list(items) * len(fut_cal_base)

    # 정기 휴무일 피처를 past_cal에 병합
    try:
        if regular_holiday_features is not None:
            
            # ✅ 충돌 방지: 기존 is_regular_holiday 컬럼 제거
            past_cal_clean = past_cal_expanded.drop(columns=['is_regular_holiday'], errors='ignore')
            
            # ✅ 정교한 정기휴무일 데이터를 다른 이름으로 병합
            regular_features_renamed = regular_holiday_features[['영업일자', '영업장명_메뉴명', 'is_regular_holiday']].copy()
            regular_features_renamed = regular_features_renamed.rename(columns={'is_regular_holiday': 'is_regular_holiday_detailed'})
            
            past_cal_df = pd.merge(
                past_cal_clean, 
                regular_features_renamed,
                left_on=['date', '영업장명_메뉴명'], 
                right_on=['영업일자', '영업장명_메뉴명'], 
                how='left'
            )
            
            # ✅ 병합 후 다시 원래 이름으로 변경
            past_cal_df = past_cal_df.rename(columns={'is_regular_holiday_detailed': 'is_regular_holiday'})
            
            # 병합 결과 확인
            before_fill = past_cal_df['is_regular_holiday'].isna().sum()
            past_cal_df['is_regular_holiday'].fillna(0, inplace=True)

            # is_regular_holiday 분포 확인
            holiday_sum = past_cal_df['is_regular_holiday'].sum()
            # print(f"📈 [DEBUG] past_cal is_regular_holiday sum: {holiday_sum}")
            
            past_cal = past_cal_df.drop(
                columns=["date", "영업장명_메뉴명", "영업일자"]
            ).values.astype(np.float32).reshape(len(items), Lx, -1)
            
        else:
            raise ValueError("regular_holiday_features is None, using fallback")
            
    except Exception as e:
        print(f"❌ [ERROR] past_cal 병합 실패: {e}")
        print(f"🔄 [DEBUG] Using fallback for past_cal")
        # Fallback: build_enhanced_features 결과에 is_regular_holiday=0 추가
        past_cal_base_fallback = past_cal_base.drop(columns=["date"]).copy()
        past_cal_base_fallback['is_regular_holiday'] = 0.0
        print(f"📝 [DEBUG] Fallback past_cal_base shape: {past_cal_base_fallback.shape}")
        print(f"🔍 [DEBUG] Fallback past_cal_base columns: {list(past_cal_base_fallback.columns)}")
        
        past_cal = np.repeat(
            past_cal_base_fallback.values[None, :, :], len(items), axis=0
        ).astype(np.float32)
        
    # ✅ Extra 피처 차원 맞추기 (훈련 시와 동일하게)
    if expected_extra_dim > 0:
        extra_features_past = np.zeros((len(items), Lx, expected_extra_dim), dtype=np.float32)
        past_cal = np.concatenate([past_cal, extra_features_past], axis=2)

    # 정기 휴무일 피처를 fut_cal에 병합
    try:
        if regular_holiday_features is not None:
            
            # ✅ 충돌 방지: 기존 is_regular_holiday 컬럼 제거
            fut_cal_clean = fut_cal_expanded.drop(columns=['is_regular_holiday'], errors='ignore')
            
            # ✅ 정교한 정기휴무일 데이터를 다른 이름으로 병합
            regular_features_renamed = regular_holiday_features[['영업일자', '영업장명_메뉴명', 'is_regular_holiday']].copy()
            regular_features_renamed = regular_features_renamed.rename(columns={'is_regular_holiday': 'is_regular_holiday_detailed'})
            
            fut_cal_df = pd.merge(
                fut_cal_clean, 
                regular_features_renamed,
                left_on=['date', '영업장명_메뉴명'], 
                right_on=['영업일자', '영업장명_메뉴명'], 
                how='left'
            )
            
            # ✅ 병합 후 다시 원래 이름으로 변경
            fut_cal_df = fut_cal_df.rename(columns={'is_regular_holiday_detailed': 'is_regular_holiday'})
            
            # 병합 결과 확인
            before_fill = fut_cal_df['is_regular_holiday'].isna().sum()
            fut_cal_df['is_regular_holiday'].fillna(0, inplace=True)
            
            # is_regular_holiday 분포 확인
            holiday_sum = fut_cal_df['is_regular_holiday'].sum()
            
            fut_cal = fut_cal_df.drop(
                columns=["date", "영업장명_메뉴명", "영업일자"]
            ).values.astype(np.float32).reshape(len(items), cfg.out_len, -1)
            
        else:
            raise ValueError("regular_holiday_features is None, using fallback")
            
    except Exception as e:
        print(f"❌ [ERROR] fut_cal 병합 실패: {e}")
        print(f"🔄 [DEBUG] Using fallback for fut_cal")
        # Fallback: build_enhanced_features 결과에 is_regular_holiday=0 추가
        fut_cal_base_fallback = fut_cal_base.drop(columns=["date"]).copy()
        fut_cal_base_fallback['is_regular_holiday'] = 0.0

        fut_cal = np.repeat(
            fut_cal_base_fallback.values[None, :, :], len(items), axis=0
        ).astype(np.float32)
        
    # ✅ Extra 피처 차원 맞추기 (fut_cal도 동일하게)
    if expected_extra_dim > 0:
        extra_features_fut = np.zeros((len(items), cfg.out_len, expected_extra_dim), dtype=np.float32)
        fut_cal = np.concatenate([fut_cal, extra_features_fut], axis=2)

    x_np = values[-Lx:, :].T  # [B,L]
    if cfg.log1p:
        x_np = np.log1p(x_np)

    B = len(items)
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)

    # ✅ past_cal_b와 fut_cal_b를 올바른 형태로 변환
    past_cal_b = torch.from_numpy(past_cal).to(device=device, dtype=torch.float32)
    fut_cal_b = torch.from_numpy(fut_cal).to(device=device, dtype=torch.float32)

    stores = [parse_store_name(it) for it in items]
    menus = [parse_menu_name(it) for it in items]

    # ✅ 주말 상대적 판매량 피처 계산 및 모델 입력으로 준비
    weekend_ratio_df = get_weekend_sales_ratio(pd.concat([train_df, tdf], axis=0))
    weekend_ratio_map = weekend_ratio_df.set_index(cfg.item_col)['weekend_sales_ratio'].to_dict()
    weekend_ratio_np = np.array([weekend_ratio_map.get(it, 1.0) for it in items], dtype=np.float32)
    weekend_ratio_tensor = torch.from_numpy(weekend_ratio_np).to(device=device, dtype=torch.float32)

    safe_store_default = next(iter(store2idx.values()), 0)
    safe_cat_default = next(iter(cat2idx.values()), 0)
    safe_type_default = next(iter(type2idx.values()), 0)

    store_idx = torch.as_tensor([store2idx.get(s, safe_store_default) for s in stores], device=device, dtype=torch.long)
    cat_idx = torch.as_tensor([cat2idx.get(get_menu_category(m), safe_cat_default) for m in menus], device=device, dtype=torch.long)
    type_idx = torch.as_tensor([type2idx.get(get_store_type(s), safe_type_default) for s in stores], device=device, dtype=torch.long)

    if group2idx is not None:
        safe_group_default = next(iter(group2idx.values()), 0)
        group_idx = torch.as_tensor([group2idx.get(get_menu_group(s, m), safe_group_default)
                                     for s, m in zip(stores, menus)], device=device, dtype=torch.long)
        zero_ratio = torch.tensor([np.mean(x_np[i] == 0) for i in range(B)], device=device, dtype=torch.float32)

    # ✅ weekend_factor 계산 시 피처 인덱스 변경
        try:
            # past_cal_df에서 컬럼 이름 확인
            if 'past_cal_df' in locals() and past_cal_df is not None:
                weekend_factor_idx = past_cal_df.columns.get_loc('is_weekend')
            else:
                # fallback: build_enhanced_features 결과에서 찾기
                cal_feat_names = [c for c in past_cal_base.columns if c != "date"]
                weekend_factor_idx = cal_feat_names.index("is_weekend")
            
            weekend_factor = torch.tensor([np.mean(past_cal[i*Lx:(i+1)*Lx, weekend_factor_idx]) for i in range(B)], device=device, dtype=torch.float32)
            
        except (ValueError, IndexError) as e:
            print(f"❌ [ERROR] weekend_factor computation failed: {e}")
            print(f"🔄 [DEBUG] Using zero fallback for weekend_factor")
            weekend_factor = torch.zeros(B, device=device, dtype=torch.float32)
    else:
        group_idx = torch.zeros(B, device=device, dtype=torch.long)
        zero_ratio = torch.zeros(B, device=device, dtype=torch.float32)
        weekend_factor = torch.zeros(B, device=device, dtype=torch.float32)

    use_cuda_amp = bool(cfg.use_amp and torch.cuda.is_available() and device.type == "cuda")
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_cuda_amp else torch.cuda.amp.autocast(enabled=False)

    model.eval()
    
    # is_regular_holiday 피처가 포함되어 있는지 샘플 확인
    if past_cal_b.shape[-1] > 0:
        if past_cal_b.shape[-1] >= 25:  # is_regular_holiday는 보통 마지막 쪽에 위치
            is_reg_holiday_feature = past_cal_b[:, :, -1]  # 마지막 피처가 is_regular_holiday라고 가정
            holiday_nonzero = (is_reg_holiday_feature > 0).sum().item()
            total_elements = is_reg_holiday_feature.numel()
    
    with amp_ctx:
        v_pred_log, p_logits = model(
            x, past_cal_b, fut_cal_b, store_idx, cat_idx, type_idx,
            weekend_ratio=weekend_ratio_tensor,
            past_ex=None,  # 예측 시에는 extra 피처 없음
            fut_ex=None,   # 예측 시에는 extra 피처 없음
            group_idx=group_idx, zero_ratio=zero_ratio, weekend_factor=weekend_factor
        )
        y_val = torch.expm1(v_pred_log).clamp_min(0.0)
        y_prob = torch.sigmoid(p_logits)
        y_hat = (y_prob * y_val).clamp_min(0.0).detach().cpu().numpy()
        
    # Dynamic cap/floor 적용
    y_hat_before = y_hat.copy()
    y_hat = dynamic_cap_floor_itemwise(
        y_hat, train_df=train_df, item_names=items, cfg=cfg, 
        k_sigma=2.5, q_hi=0.98, min_floor=0.0, lookback_days=120
    )
    
    # Cap/floor 적용 효과 확인
    changed_values = (y_hat != y_hat_before).sum()
    if changed_values > 0:
        print(f"   - Before cap/floor: mean={y_hat_before.mean():.4f}, max={y_hat_before.max():.4f}")
        print(f"   - After cap/floor: mean={y_hat.mean():.4f}, max={y_hat.max():.4f}")

    out = pd.DataFrame(
        y_hat.T,
        index=[f"D+{i}" for i in range(1, cfg.out_len + 1)],
        columns=items
    )
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    
    return out

def dynamic_cap_floor_itemwise(y_hat: np.ndarray,
                               train_df: pd.DataFrame,
                               item_names: List[str],
                               cfg: EnhancedNHiTSConfig,
                               k_sigma: float = 2.5,
                               q_hi: float = 0.98,
                               min_floor: float = 0.0,
                               lookback_days: int = 120) -> np.ndarray:
    """
    y_hat: [B, H] (예측)
    train_df: (cfg.date_col, cfg.item_col, cfg.target_col)
    item_names: 길이 B, y_hat의 열 순서와 동일
    """
    clipped = y_hat.copy()
    tdf = train_df.copy()
    tdf[cfg.date_col] = pd.to_datetime(tdf[cfg.date_col], errors='coerce')
    cutoff = tdf[cfg.date_col].max()
    if pd.isna(cutoff):
        return np.clip(clipped, min_floor, None)

    start = cutoff - pd.Timedelta(days=lookback_days)
    sub = tdf[(tdf[cfg.date_col] >= start) & (tdf[cfg.date_col] <= cutoff)]

    for i, it in enumerate(item_names):
        hist = sub.loc[sub[cfg.item_col] == it, cfg.target_col].astype(float).values
        if hist.size == 0:
            clipped[i, :] = np.clip(clipped[i, :], min_floor, None)
            continue
        pos = hist[hist > 0]
        mu  = pos.mean() if pos.size else 0.0
        sd  = pos.std() if pos.size else 0.0
        q98 = np.quantile(hist, q_hi) if hist.size else 0.0
        upper = max(min(mu + k_sigma * sd, q98 * 1.2), 1.0)
        clipped[i, :] = np.clip(clipped[i, :], min_floor, upper)
    return clipped


def compute_dyn_weights_from_val(val_loss_dict: Dict[str, float], temperature: float = 1.5) -> Dict[str, float]:
    """
    val_loss_dict: {"TimesNet": 0.36, "GRU": 0.32, ...}
    낮은 손실일수록 높은 가중치. temperature(τ)로 쏠림 제어.
    """
    names = list(val_loss_dict.keys())
    losses = np.array([val_loss_dict[n] for n in names], dtype=np.float64)
    losses = np.clip(losses, 1e-12, None)
    scores = -losses / max(1e-8, temperature)
    w = np.exp(scores - scores.max())
    w = w / w.sum()
    return {n: float(w[i]) for i, n in enumerate(names)}

# =====================
# Ensemble Helper (기존과 동일)
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

"""### 모델 팩토리"""

# =====================
# Model factory (수정된 코드)
# =====================
def build_model_by_name(name: str, cfg: EnhancedNHiTSConfig, ds: EnhancedNHiTSDataset,
                        trial: Optional['optuna.trial.Trial']=None) -> nn.Module:
    
    # cal_dim에 extra_dim 포함
    cal_dim = int(ds.cal_feats.shape[1] + getattr(ds, "extra_dim", 0))
    has_weekend_ratio = hasattr(ds, 'item_weekend_ratio')

    if name == "N-HiTS":
        hidden = trial.suggest_categorical("nh_hidden", [256, 384, 512]) if trial else cfg.hidden
        n_layers = trial.suggest_categorical("nh_layers", [1, 2, 3]) if trial else cfg.n_layers
        dropout = trial.suggest_float("nh_dropout", 0.0, 0.3) if trial else cfg.dropout
        nh_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__, "hidden": hidden, "n_layers": n_layers, "dropout": dropout})
        return EnhancedNHiTSModel(cfg.in_len, cfg.out_len, cal_dim, ds.n_stores, 
                                  ds.n_categories, ds.n_types, nh_cfg, 
                                  has_weekend_ratio=has_weekend_ratio)

    if name == "PatchTST":
        d_model = trial.suggest_categorical("pt_d_model", [192, 256, 320]) if trial else cfg.ptst_d_model
        nhead = trial.suggest_categorical("pt_nhead", [8, 12]) if trial else cfg.ptst_nhead
        nlayer = trial.suggest_categorical("pt_nlayers", [2, 3, 4]) if trial else cfg.ptst_num_layers
        p_len = trial.suggest_categorical("pt_patch", [2, 3, 4]) if trial else max(2, cfg.ptst_patch_len // 2)
        stride = trial.suggest_categorical("pt_stride", [1, 2]) if trial else max(1, cfg.ptst_stride // 2)
        drop = trial.suggest_float("pt_dropout", 0.05, 0.2) if trial else cfg.ptst_dropout
        ff_mult = trial.suggest_float("ptst_ff_mult", 1.5, 3.5) if trial else cfg.ptst_ff_mult
        pt_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__,
                                        "ptst_d_model": d_model, "ptst_nhead": nhead, "ptst_num_layers": nlayer,
                                        "ptst_patch_len": p_len, "ptst_stride": stride, "ptst_dropout": drop, "ptst_ff_mult": ff_mult
                                        })
        # 🚨 (수정) weekend_ratio 존재 여부를 전달
        return ImprovedPatchTSTTiny(pt_cfg, cfg.in_len, cfg.out_len, cal_dim, ds.n_stores, ds.n_categories, ds.n_types, has_weekend_ratio=has_weekend_ratio)

    if name == "TimesNet":
        channels = trial.suggest_categorical("tn_channels", [96, 128, 160]) if trial else cfg.tnet_channels
        blocks = trial.suggest_categorical("tn_blocks", [2, 3, 4]) if trial else cfg.tnet_blocks
        drop = trial.suggest_float("tn_dropout", 0.0, 0.3) if trial else cfg.tnet_dropout
        tn_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__,
                                        "tnet_channels": channels, "tnet_blocks": blocks, "tnet_dropout": drop
                                        })
        # 🚨 (수정) weekend_ratio 존재 여부를 전달
        return ImprovedTimesNetTiny(tn_cfg, cfg.in_len, cfg.out_len, cal_dim, ds.n_stores, ds.n_categories, ds.n_types, has_weekend_ratio=has_weekend_ratio)

    if name == "GRU":
        hidden = trial.suggest_categorical("gru_hidden", [128, 192, 256]) if trial else cfg.gru_hidden
        layers = trial.suggest_categorical("gru_layers", [1, 2, 3]) if trial else cfg.gru_layers
        drop = trial.suggest_float("gru_dropout", 0.0, 0.3) if trial else cfg.gru_dropout
        bidi = trial.suggest_categorical("gru_bidi", [False, True]) if trial else cfg.gru_bidirectional
        gr_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__,
                                        "gru_hidden": hidden, "gru_layers": layers, "gru_dropout": drop, "gru_bidirectional": bidi
                                        })
        # 🚨 (수정) weekend_ratio 존재 여부를 전달
        return GRUTiny(gr_cfg, cfg.in_len, cfg.out_len, cal_dim, ds.n_stores, ds.n_categories, ds.n_types, has_weekend_ratio=has_weekend_ratio)

    if name == "MLinear":
        hidden_dim = trial.suggest_categorical("ml_hidden_dim", [256, 384, 512]) if trial else cfg.mlin_hidden_dim
        dropout = trial.suggest_float("ml_dropout", 0.0, 0.3) if trial else cfg.mlin_dropout
        use_residual = trial.suggest_categorical("ml_residual", [True, False]) if trial else cfg.mlin_use_residual
        use_layer_norm = trial.suggest_categorical("ml_layer_norm", [True, False]) if trial else cfg.mlin_use_layer_norm
        ml_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__,
                                        "mlin_hidden_dim": hidden_dim, "mlin_dropout": dropout,
                                        "mlin_use_residual": use_residual, "mlin_use_layer_norm": use_layer_norm
                                        })
        # 🚨 (수정) weekend_ratio 존재 여부를 전달
        return MLinearTiny(ml_cfg, cfg.in_len, cfg.out_len, cal_dim, ds.n_stores, ds.n_categories, ds.n_types, ds.n_groups, has_weekend_ratio=has_weekend_ratio)

    if name == "TSMixer":
        d_model = trial.suggest_categorical("tm_d_model", [192, 256, 320]) if trial else cfg.tsm_d_model
        n_layers = trial.suggest_categorical("tm_n_layers", [2, 3, 4]) if trial else cfg.tsm_num_layers
        drop = trial.suggest_float("tm_dropout", 0.0, 0.3) if trial else cfg.tsm_dropout
        p_len = trial.suggest_categorical("tm_patch", [2, 3, 4]) if trial else max(2, cfg.tsm_patch_len)
        tsm_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__,
                                        "tsm_d_model": d_model, "tsm_num_layers": n_layers, "tsm_dropout": drop,
                                        "tsm_patch_len": p_len
                                        })
        # 🚨 (수정) weekend_ratio 존재 여부를 전달
        return TSMixer(tsm_cfg, cfg.in_len, cfg.out_len, cal_dim,
                       ds.n_stores, ds.n_categories, ds.n_types, ds.n_groups, d_model, has_weekend_ratio=has_weekend_ratio)

    if name == "TCN":
        C = trial.suggest_categorical("tcn_C", [96,128,160]) if trial else 128
        depth = trial.suggest_categorical("tcn_depth", [3,4,5]) if trial else 4
        drop = trial.suggest_float("tcn_drop", 0.05, 0.25) if trial else 0.1
        return TCNTiny(cfg, cfg.in_len, cfg.out_len, cal_dim, ds.n_stores, ds.n_categories, ds.n_types, C=C, depth=depth, drop=drop, has_weekend_ratio=has_weekend_ratio)
    
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

    elif model_name == "MLinear":
        mapping = {"ml_hidden_dim":"mlin_hidden_dim","ml_dropout":"mlin_dropout","ml_residual":"mlin_use_residual","ml_layer_norm":"mlin_use_layer_norm"}
        for k, v in mapping.items():
            if k in best_params:
                tmp[v] = best_params[k]

    elif model_name == "TSMixer":
        # ⭐⭐⭐ 이 부분을 아래와 같이 수정하세요. ⭐⭐⭐
        mapping = {"tm_d_model": "tsm_d_model", "tm_n_layers": "tsm_num_layers", "tm_dropout": "tsm_dropout", "tm_patch": "tsm_patch_len"}
        for k, v in mapping.items():
            if k in best_params:
                tmp[v] = best_params[k]

    new_cfg = EnhancedNHiTSConfig(**tmp)
    return build_model_by_name(model_name, new_cfg, ds, None)


def smooth_horizon_block(block: pd.DataFrame) -> pd.DataFrame:
    # block: index=D+1..D+H, columns=items
    arr = block.values  # [H, I]
    H, I = arr.shape
    sm = arr.copy()
    for j in range(I):
        y = arr[:, j]
        y_pad = np.pad(y, (1, 1), mode="edge")
        y_sm = (y_pad[:-2] + y_pad[1:-1] + y_pad[2:]) / 3.0  # 3-point
        s0, s1 = y.sum(), y_sm.sum()
        if s1 > 0:
            y_sm *= (s0 / s1)  # 총합 보존
        sm[:, j] = y_sm
    return pd.DataFrame(sm, index=block.index, columns=block.columns)

def cap_floor_itemwise(final_submit: pd.DataFrame, train_df: pd.DataFrame, cfg) -> pd.DataFrame:
    num_cols = [c for c in final_submit.columns if c != cfg.date_col]
    for col in num_cols:
        recent = train_df.loc[train_df[cfg.item_col] == col, cfg.target_col].tail(28).to_numpy()
        if recent.size >= 7:
            mu, sigma = float(np.mean(recent)), float(np.std(recent))
            cap_stat   = mu + 3 * sigma
            floor_stat = max(0.0, mu - 3 * sigma)
        else:
            cap_stat, floor_stat = np.inf, 0.0

        cap_q = float(final_submit[col].quantile(0.95) * 1.5)
        cap = min(cap_stat, cap_q)
        final_submit[col] = np.clip(final_submit[col].to_numpy(), floor_stat, cap)
    return final_submit

"""### 메인 실행 (H-wise 제거, 전역 가중치만)"""
if __name__ == "__main__":
    cfg = EnhancedNHiTSConfig()
    if cfg.store_weights is None: cfg.store_weights = DEFAULT_STORE_WEIGHTS
    if cfg.custom_holidays_list is None: cfg.custom_holidays_list = DEFAULT_CUSTOM_HOLIDAYS
    set_seed(cfg.seed)

    print("=== Enhanced 6-Model Ensemble (+ MLinear + TCN) 시작 ===")

    if not os.path.exists(cfg.train_csv):
        raise FileNotFoundError(f"학습 파일을 찾을 수 없습니다: {cfg.train_csv}")

    # train_df 먼저 로드 (predict에 사용)
    train_df = pd.read_csv(cfg.train_csv)
    ds = EnhancedNHiTSDataset(cfg, train_df)

    # 체크포인트 디렉터리
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ===== (옵션) Optuna per-model 튜닝 =====
    best_params_all = {}
    if cfg.USE_OPTUNA:
        if not HAS_OPTUNA:
            print("⚠️ cfg.USE_OPTUNA=True지만 optuna가 설치되어 있지 않습니다. 튜닝은 건너뜁니다.")
        else:
            tune_mask_val = make_val_mask_by_week(ds, cfg.cv_fold_end_dates[0])
            for name in ["MLinear"]:
                study, best_params = run_optuna_for_model(name, cfg, ds, tune_mask_val)
                best_params_all[name] = best_params
            os.makedirs("./data", exist_ok=True)
            with open("./data/optuna_best_params.json", "w", encoding="utf-8") as f:
                json.dump(best_params_all, f, ensure_ascii=False, indent=2)

    # 🚨 수동 주입: Optuna 결과가 없으면 아래 값 사용 (MLinear)
    if ("MLinear" not in best_params_all) or (best_params_all["MLinear"] is None):
        best_params_all["MLinear"] = {
            "base_lr": 0.0005377,
            "max_lr": 0.001667,
            "weight_decay": 4.5283e-05,
            "grad_clip": 0.3176,
            "pat_ratio": 0.1633,
            "ml_hidden_dim": 512,
            "ml_dropout": 0.2586,
            "ml_residual": True,
            "ml_layer_norm": False,
        }

    # ===== 모델 학습 (멀티 폴드 평균 val) =====
    model_names = ["TCN", "MLinear", "TimesNet", "GRU", "N-HiTS", "PatchTST", "TSMixer"]
    trained_models: Dict[str, nn.Module] = {}
    avg_val_losses: Dict[str, float] = {}

    for name in model_names:
        checkpoint_path = os.path.join(cfg.checkpoint_dir, f"{name}_best_fold.pth")
        print(f"📚 {name} 학습...")

        # --- per-model 하이퍼/CFG ---
        cfg_for_model = cfg
        base_lr = cfg.BASE_LR_FULL; max_lr = cfg.MAX_LR_FULL; weight_decay = cfg.WD_FULL
        best_for_model = best_params_all.get(name)
        if name == "MLinear" and best_for_model is not None:
            cfg_for_model = EnhancedNHiTSConfig(**{
                **cfg.__dict__,
                "mlin_hidden_dim": best_for_model.get("ml_hidden_dim", cfg.mlin_hidden_dim),
                "mlin_dropout": best_for_model.get("ml_dropout", cfg.mlin_dropout),
                "mlin_use_residual": best_for_model.get("ml_residual", cfg.mlin_use_residual),
                "mlin_use_layer_norm": best_for_model.get("ml_layer_norm", cfg.mlin_use_layer_norm),
                "grad_clip": best_for_model.get("grad_clip", cfg.grad_clip),
                "earlystop_patience_ratio": best_for_model.get("pat_ratio", cfg.earlystop_patience_ratio),
            })
            base_lr = best_for_model.get("base_lr", base_lr)
            max_lr  = best_for_model.get("max_lr",  max_lr)
            weight_decay = best_for_model.get("weight_decay", weight_decay)

        # 초기 가중치 스냅샷
        base_model = build_final_model_with_best(
            name, cfg_for_model, ds, best_params_all.get(name) if (cfg.USE_OPTUNA or name == "MLinear") else None
        )
        init_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        del base_model
        gc.collect(); 
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        fold_vals = []
        best_fold = None; best_fold_state = None

        for fold_i, end_date in enumerate(cfg.cv_fold_end_dates, start=1):
            mask_val = make_val_mask_by_week(ds, end_date)

            model_f = build_final_model_with_best(
                name, cfg_for_model, ds, best_params_all.get(name) if (cfg.USE_OPTUNA or name == "MLinear") else None
            )
            model_f.load_state_dict(init_state, strict=True)

            trainer = GenericTrainer(
                cfg_for_model, ds, model_f,
                epochs=cfg_for_model.EPOCHS_FULL, batch_size=cfg_for_model.BATCH_FULL,
                base_lr=base_lr, max_lr=max_lr, weight_decay=weight_decay
            )
            train_loader, val_loader = trainer.make_loaders_from_mask(mask_val)
            model_f, val_loss = trainer.train_with_loaders(
                train_loader, val_loader, model_name=f"{name}-F{fold_i}"
            )
            fold_vals.append(float(val_loss))

            if (best_fold is None) or (val_loss < fold_vals[best_fold]):
                best_fold = fold_i - 1
                best_fold_state = {k: v.detach().cpu().clone() for k, v in model_f.state_dict().items()}

            del trainer, train_loader, val_loader, model_f
            gc.collect(); 
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # 모델별 fold 평균 검증
        avg_val = float(np.mean(fold_vals)) if len(fold_vals) > 0 else float('inf')
        avg_val_losses[name] = avg_val
        print(f"[VAL(avg over folds)] {name}={avg_val:.5f}")

        # 베스트 폴드 모델을 보관 + 저장
        final_model = build_final_model_with_best(
            name, cfg_for_model, ds, best_params_all.get(name) if (cfg.USE_OPTUNA or name == "MLinear") else None
        )
        if best_fold_state is not None:
            final_model.load_state_dict(best_fold_state, strict=True)
        trained_models[name] = final_model.to(cfg_for_model.device)

        if best_fold_state is not None:
            torch.save(best_fold_state, checkpoint_path)
            print(f"💾 {name} 베스트 폴드 체크포인트 저장: {checkpoint_path}")

    # ===== 전역 가중치 계산 (H-wise 없애고 전역만) =====
    vals_in_order = [avg_val_losses.get(n, np.nan) for n in model_names]
    weights_arr = np.array(invloss_weights(vals_in_order), dtype=np.float64)  # 낮을수록 가중 ↑
    weights = dict(zip(model_names, weights_arr))
    print("[Ensemble Weights] " + "  ".join([f"{k}={weights.get(k,0.0):.3f}" for k in model_names]))

    # ===== 추론 & 제출 =====
    print("🔮 앙상블 예측...")
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

            df_preds: Dict[str, pd.DataFrame] = {}
            for name in model_names:
                if name not in trained_models: 
                    continue
                group2idx = ds.group2idx if name == "MLinear" else None
                # 기존 코드에서 이렇게 수정하세요:
                df = predict_one_file_generic(
                    cfg, trained_models[name], tdf, train_df, ds,  # ✅ ds (dataset) 추가
                    ds.store2idx, ds.cat2idx, ds.type2idx, group2idx
                )
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
                if k in df_preds:
                    df_preds[k] = df_preds[k].reindex(columns=items).fillna(0.0)

            # ✅ 전역 가중치만 사용한 가중 평균
            models_available = [n for n in model_names if n in df_preds and df_preds[n] is not None and not df_preds[n].empty]
            if len(models_available) == 0:
                print("  ⚠️ 사용 가능한 모델 예측이 없습니다. 스킵")
                continue

            w = np.array([weights.get(n, 0.0) for n in models_available], dtype=np.float64)
            w_sum = w.sum()
            if w_sum <= 0:
                w = np.ones(len(models_available), dtype=np.float64) / len(models_available)
            else:
                w = w / w_sum

            mix = np.zeros_like(df_preds[models_available[0]].values, dtype=np.float64)
            for i, n in enumerate(models_available):
                mix += w[i] * df_preds[n].values
            mix = np.clip(mix, 0, None)

            submit_block = pd.DataFrame(
                mix,
                index=[f"D+{i}" for i in range(1, cfg.out_len + 1)],
                columns=items
            )
            submit_block.index = [f"TEST_{test_idx:02d}+{k}일" for k in range(1, cfg.out_len + 1)]
            submit_block = smooth_horizon_block(submit_block)
            all_preds.append(submit_block)

    if len(all_preds) == 0:
        print("⚠️ 유효한 예측 결과가 없어 제출 파일을 생성하지 않았습니다.")
    else:
        final_submit = pd.concat(all_preds, axis=0)
        final_submit.reset_index(inplace=True)
        final_submit.rename(columns={"index": "영업일자"}, inplace=True)

        # 템플릿 컬럼 순서에 맞추기
        final_submit = final_submit.reindex(columns=sub_template.columns, fill_value=0)

        # (선택) 최근 분포 기반 상한/하한 보정
        final_submit = cap_floor_itemwise(final_submit, train_df, cfg)

        # 정수 반올림/클리핑
        num_cols = [c for c in final_submit.columns if c != cfg.date_col]
        final_submit[num_cols] = np.rint(
            np.clip(final_submit[num_cols].values, a_min=0, a_max=None)
        ).astype(int)

        os.makedirs(os.path.dirname(cfg.out_submission_csv), exist_ok=True)
        final_submit.to_csv(cfg.out_submission_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 앙상블 완료! → {cfg.out_submission_csv}")

    print("🏆 TimesNet + GRU + MLinear + N-HiTS + PatchTST + TSMixer + TCN 학습/추론 끝")