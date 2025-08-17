# -*- coding: utf-8 -*-
"""
Enhanced N-HiTS + Improved PatchTST + Improved TimesNet + GRU + Improved DLinear
(Multi-model Training only, Ensemble/Submission disabled)

적용 개선 사항:
- [N-HiTS] FiLM 조건부 블록, Horizon Embedding(바이어스), Prob Temperature
- [PatchTST] 보조 채널(원신호/7일 이동평균/지난주), Local→Global Cross-Attn, Prob Temperature
- [TimesNet] 트렌드/시즌 분해 입력 + 멀티 주기 블록(7/14/28) 병렬, Prob Temperature
- [GRU] Attention Pooling Head + Meta-gate Blend, Prob Temperature
- [DLinear] 7/14/28 스케일, Horizon-wise 요일/월 계수(스텝별), Prob Temperature
- [Dataset/Pred] cluster 포함, prob-head 보조 통계(mean/std/slope) + 메뉴 메타(출시/단종/계절성) 반영
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
    out_submission_csv: str = "LG_output/0817_5model_submission.csv"

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
    BASE_LR_FULL: float = 3e-4
    MAX_LR_FULL: float = 8e-3
    WD_FULL: float = 1.2e-3

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
    hidden: int = 352
    n_blocks: int = 3
    n_layers: int = 1
    n_pool_kernel_size: Optional[List[int]] = None
    pooling_mode: str = "MaxPool1d"
    interpolation_mode: str = "linear"
    dropout: float = 0.25
    stack_types: Optional[List[str]] = None
    n_freq_downsample: Optional[List[int]] = None

    # Loss
    eps_smape: float = 0.02
    zero_weight: float = 0.01
    hurdle_lambda: float = 0.07

    # AMP/EMA
    use_amp: bool = True
    use_compile: bool = False
    ema_decay: float = 0.9990

    # 가중치/공휴일
    store_weights: Optional[Dict[str, float]] = None
    custom_holidays_list: Optional[List[str]] = None

    # ==== PatchTST ====
    ptst_d_model: int = 192
    ptst_nhead: int = 8
    ptst_num_layers: int = 2
    ptst_patch_len: int = 3
    ptst_stride: int = 2
    ptst_dropout: float = 0.25
    ptst_head_hidden: int = 256
    ptst_ff_mult: float = 1.5  # Config화

    # ==== TimesNet ====
    tnet_channels: int = 128
    tnet_blocks: int = 3
    tnet_kernels: Tuple[int, int, int] = (3, 5, 7)
    tnet_dropout: float = 0.25
    tnet_head_hidden: int = 160

    # ==== GRU ====
    gru_hidden: int = 192
    gru_layers: int = 2
    gru_dropout: float = 0.25
    gru_bidirectional: bool = False
    gru_head_hidden: int = 128

    # ==== DLinear ====
    dlin_head_hidden: int = 256

    # ==== 훈련 보조 ====
    grad_clip: float = 0.5
    earlystop_patience_ratio: float = 0.10
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
# Cluster mapping (수정/확장 가능)
# =====================
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

    # Cluster 4 (샘플)
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

# =====================
# Feature Utils
# =====================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def sine_cosine_encoding(value: float, max_val: float):
    return math.sin(2 * math.pi * value / max_val), math.cos(2 * math.pi * value / max_val)

def get_menu_category(menu_name: str) -> str:
    s = "" if menu_name is None else str(menu_name)
    s = s.strip()
    if not s:
        return "others"

    # 한글/영문 혼재 안전 처리(소문자화는 영문에만 영향)
    s_lower = s.lower()

    alcohol_kw = ['막걸리','소주','맥주','와인','참이슬','처음처럼','카스','하이네켄','버드와이저','스텔라']
    hot_kw     = ['찌개','탕','국밥','라면','해장국','갈비탕']
    bbq_kw     = ['삼겹','갈비','목살','bbq','구이','불고기']
    dd_kw      = ['아이스크림','식혜','콜라','스프라이트','에이드','아이스티']
    coffee_kw  = ['아메리카노','라떼','커피','espresso','latte']
    noodle_kw  = ['냉면','파스타','스파게티','면','우동','라멘']
    rice_kw    = ['비빔밥','볶음밥','공깃밥','정식','덮밥','밥']

    def has_any(keywords):
        return any((kw in s) or (kw.lower() in s_lower) for kw in keywords)

    if has_any(alcohol_kw): return 'alcohol'
    if has_any(hot_kw):     return 'hot_food'
    if has_any(bbq_kw):     return 'bbq'
    if has_any(dd_kw):      return 'dessert_drink'
    if has_any(coffee_kw):  return 'coffee'
    if has_any(noodle_kw):  return 'noodles'
    if has_any(rice_kw):    return 'rice'
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
    df = pd.DataFrame({"date": pd.to_datetime(dates)})
    df["dow"] = df["date"].dt.weekday           # 0=Mon
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["week"] = df["date"].apply(lambda d: int(d.isocalendar().week))
    df["quarter"] = df["date"].dt.quarter

    # 주말/휴일 기본
    df["is_friday"] = (df["dow"] == 4).astype(int)
    df["is_saturday"] = (df["dow"] == 5).astype(int)
    df["is_sunday"] = (df["dow"] == 6).astype(int)
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    holidays = set(pd.to_datetime(list(holidays_set))) if holidays_set else set()
    df["is_holiday"] = df["date"].isin(holidays).astype(int)

    # 전후일
    df["tomorrow"] = df["date"] + pd.Timedelta(days=1)
    df["yesterday"] = df["date"] - pd.Timedelta(days=1)
    df["is_before_holiday"] = df["tomorrow"].isin(holidays).astype(int)
    df["is_after_holiday"] = df["yesterday"].isin(holidays).astype(int)

    # 월초/월말
    df["is_month_start"] = (df["day"] <= 3).astype(int)
    df["is_month_end"] = (df["day"] >= 28).astype(int)

    # 계절/방학(리조트에 유의미)
    df["is_spring"] = df["month"].isin([4,5,6]).astype(int)
    df["is_summer"] = df["month"].isin([7,8]).astype(int)
    df["is_autumn"] = df["month"].isin([9,10,11]).astype(int)
    df["is_winter"] = df["month"].isin([12,1,2,3]).astype(int)
    df["is_summer_vacation"] = df["month"].isin([7,8]).astype(int)
    df["is_winter_vacation"] = df["month"].isin([12,1,2]).astype(int)

    # 월 내 주차
    def week_of_month(d: pd.Timestamp) -> int:
        first = d.replace(day=1)
        return 1 + (d.day + first.weekday() - 1) // 7
    df["week_of_month"] = df["date"].apply(week_of_month).astype(int)
    df["is_week_of_month_end"] = (df["week_of_month"] >= 4).astype(int)

    # 연휴/주말 블록 분석
    df["is_off"] = ((df["is_weekend"]==1) | (df["is_holiday"]==1)).astype(int)

    off = df["is_off"].values.astype(int)
    start = np.zeros_like(off); end = np.zeros_like(off)
    for i in range(len(off)):
        if off[i] == 1 and (i==0 or off[i-1]==0): start[i] = 1
        if off[i] == 1 and (i==len(off)-1 or off[i+1]==0): end[i] = 1
    df["is_holiday_block_start"] = start
    df["is_holiday_block_end"] = end

    # 3일 이상 롱위켄드
    block_len = np.zeros_like(off)
    i = 0
    while i < len(off):
        if off[i] == 1:
            j = i
            while j+1 < len(off) and off[j+1] == 1:
                j += 1
            L = j - i + 1
            block_len[i:j+1] = L
            i = j + 1
        else:
            i += 1
    df["is_long_weekend"] = (block_len >= 3).astype(int)

    # 징검다리 평일
    df["is_bridge_day"] = (((df["yesterday"].isin(holidays)) | (df["tomorrow"].isin(holidays)) | 
                            (df["is_before_holiday"]==1) | (df["is_after_holiday"]==1)) & 
                           (df["is_off"]==0)).astype(int)

    # 휴일까지/이전 휴일로부터 거리(클램프 0~7)
    hol_sorted = sorted(list(holidays))
    def days_to_next(d: pd.Timestamp) -> int:
        for h in hol_sorted:
            if h >= d: return int((h - d).days)
        return 9999
    def days_since_prev(d: pd.Timestamp) -> int:
        prev = [h for h in hol_sorted if h <= d]
        if not prev: return 9999
        return int((d - prev[-1]).days)
    df["days_to_next_holiday"] = df["date"].apply(days_to_next).clip(0, 7)
    df["days_since_prev_holiday"] = df["date"].apply(days_since_prev).clip(0, 7)

    # 주기 인코딩
    def sine_cosine_encoding(value: float, max_val: float):
        return math.sin(2 * math.pi * value / max_val), math.cos(2 * math.pi * value / max_val)
    df[["month_sin", "month_cos"]] = df["month"].apply(lambda m: pd.Series(sine_cosine_encoding(m, 12)))
    df[["dow_sin", "dow_cos"]] = df["dow"].apply(lambda d: pd.Series(sine_cosine_encoding(d, 7)))
    df[["doy_sin", "doy_cos"]] = df["date"].apply(lambda d: pd.Series(sine_cosine_encoding(d.dayofyear, 365)))
    df[["week_sin", "week_cos"]] = df["week"].apply(lambda w: pd.Series(sine_cosine_encoding(w, 52)))
    df[["quarter_sin", "quarter_cos"]] = df["quarter"].apply(lambda q: pd.Series(sine_cosine_encoding(q, 4)))

    return df.drop(columns=["tomorrow", "yesterday"])

def build_menu_meta(train_df: pd.DataFrame, cfg) -> Dict[str, Dict]:
    """
    각 메뉴(= item_col)별로 출시/단종/계절성 메타를 추출.
    반환 예:
    {
      "카페테리아_아메리카노(ICE)": {
         "launch_date": "2023-06-01",
         "discontinue_date": None,
         "active_months": [7,8,9],             # 판매가 관측된 달 집합
         "ever_sold": True
      },
      ...
    }
    """
    if cfg.date_col not in train_df or cfg.item_col not in train_df or cfg.target_col not in train_df:
        raise ValueError("train_df에 필수 컬럼이 없습니다.")

    df = train_df.copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col])
    df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce").fillna(0.0).clip(lower=0)

    # 아이템별 시계열 정렬
    df = df.sort_values([cfg.item_col, cfg.date_col])
    meta: Dict[str, Dict] = {}

    for item, g in df.groupby(cfg.item_col):
        dates = g[cfg.date_col].tolist()
        vals  = g[cfg.target_col].to_numpy()

        # 판매가 있었는가
        sold_mask = vals > 0
        ever_sold = bool(sold_mask.any())

        # 출시/단종 추정
        launch_date = None
        discontinue_date = None
        if ever_sold:
            first_idx = int(np.argmax(sold_mask))
            last_idx  = len(vals) - 1 - int(np.argmax(sold_mask[::-1]))
            launch_date = pd.to_datetime(dates[first_idx]).normalize()
            # 마지막 판매 이후 전부 0이면 단종으로 간주 (train 말단에 0-run)
            if last_idx < len(vals) - 1 and not (sold_mask[last_idx+1:].any()):
                discontinue_date = pd.to_datetime(dates[last_idx]).normalize()

        # 계절성: 판매가 발생한 month 집합
        active_months = sorted(set(int(d.month) for d, v in zip(dates, vals) if v > 0))

        meta[str(item)] = {
            "launch_date": str(launch_date.date()) if launch_date is not None else None,
            "discontinue_date": str(discontinue_date.date()) if discontinue_date is not None else None,
            "active_months": active_months,
            "ever_sold": ever_sold,
        }

    return meta


def parse_store_name(item_full: str) -> str:
    if item_full is None:
        return ""
    s = str(item_full)
    if "_" in s:
        return s.split("_", 1)[0]
    return s  # 구분자가 없을 때 전체를 스토어로 취급

def parse_menu_name(item_full: str) -> str:
    if item_full is None:
        return ""
    s = str(item_full)
    if "_" in s:
        return s.split("_", 1)[1]
    return s  # 구분자가 없을 때 전체를 메뉴로 취급


def make_val_mask_by_week(dataset: 'EnhancedNHiTSDataset', end_date_str: str) -> np.ndarray:
    end_date = pd.to_datetime(end_date_str)
    start_date = end_date - pd.Timedelta(days=6)
    ted = dataset.target_end_dates
    return (ted >= start_date) & (ted <= end_date)

# =====================
# Dataset (+ cluster)
# =====================
class EnhancedNHiTSDataset(Dataset):
    def __init__(self, cfg: EnhancedNHiTSConfig, df: pd.DataFrame, menu_meta: Optional[Dict[str, Dict]] = None):
        self.cfg = cfg
        self.df = df.copy()
        self.menu_meta = menu_meta or {}
        if cfg.date_col not in self.df.columns or cfg.item_col not in self.df.columns or cfg.target_col not in self.df.columns:
            raise ValueError("입력 데이터에 필요한 컬럼이 없습니다.")

        self.df[cfg.date_col] = pd.to_datetime(self.df[cfg.date_col])
        self.df[cfg.target_col] = pd.to_numeric(self.df[cfg.target_col], errors='coerce').fillna(0.0).clip(lower=0)

        pivot = self.df.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index()
        self.items = list(pivot.columns.astype(str))
        self.dates = list(pivot.index)
        self.values = pivot.fillna(0.0).values.astype(np.float32)

        holidays_set = set(pd.to_datetime(cfg.custom_holidays_list)) if cfg.custom_holidays_list else set()
        caldf = build_enhanced_features(self.dates, holidays_set)
        self.base_cal_feats = caldf.drop(columns=["date"]).values.astype(np.float32)   # [T, C_base]
        self.base_cal_feat_names = [c for c in caldf.columns if c != "date"]

        # ---- meta 인덱스들 ----
        stores = [parse_store_name(it) for it in self.items]
        menus  = [parse_menu_name(it) for it in self.items]

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

        clusters = [cluster_mapping.get(it, 0) for it in self.items]
        self.cluster2idx = {c: i for i, c in enumerate(sorted(set(clusters)))}
        self.item_cluster_idx = np.array([self.cluster2idx[c] for c in clusters], dtype=np.int64)
        self.n_clusters = len(self.cluster2idx)

        sw = cfg.store_weights or {}
        self.sample_weights = np.array([sw.get(parse_store_name(it), 1.0) for it in self.items], dtype=np.float32)

        # ---- 슬라이싱 인덱스 생성 ----
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

        # ---- 캘린더 피처 이름/인덱스 (DLinear 안전용) ----
        self.cal_feat_names = list(self.base_cal_feat_names)
        name2idx = {n: i for i, n in enumerate(self.cal_feat_names)}
        self.idx_dow   = name2idx.get("dow",   0)
        self.idx_month = name2idx.get("month", 1)

        # ✅ 모델에 전달될 캘린더 차원 = 기본 + 3(launch/discontinue/season)
        self.cal_dim_for_model = self.base_cal_feats.shape[1] + 3

    def _flags_for_dates(self, item: str, dates: List[pd.Timestamp]) -> np.ndarray:
        """
        주어진 item과 날짜 리스트에 대해 [len(dates), 3] 플래그 생성:
        [is_launched, is_discontinued, is_in_season]
        """
        meta = self.menu_meta.get(str(item), {})
        launch_date_str = meta.get("launch_date")
        discontinue_date_str = meta.get("discontinue_date")
        active_months = set(int(m) for m in meta.get("active_months", []))

        if launch_date_str is not None:
            try:
                launch_date = pd.to_datetime(launch_date_str)
            except Exception:
                launch_date = None
        else:
            launch_date = None

        if discontinue_date_str is not None:
            try:
                discontinue_date = pd.to_datetime(discontinue_date_str)
            except Exception:
                discontinue_date = None
        else:
            discontinue_date = None

        L = len(dates)
        flags = np.zeros((L, 3), dtype=np.float32)
        for i, d in enumerate(dates):
            is_launched = (launch_date is not None) and (d >= launch_date)
            is_discont  = (discontinue_date is not None) and (d > discontinue_date)
            is_in_season = (len(active_months) > 0) and (int(d.month) in active_months)
            flags[i, 0] = 1.0 if is_launched else 0.0
            flags[i, 1] = 1.0 if is_discont else 0.0
            flags[i, 2] = 1.0 if is_in_season else 0.0
        return flags

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

        # ---- 캘린더 (기본 + 메뉴 플래그 3채널) ----
        past_cal_base = self.base_cal_feats[t0:t0 + Lx, :]                  # [Lx, Cb]
        fut_cal_base  = self.base_cal_feats[t0 + Lx:t0 + Lx + Ly, :]        # [Ly, Cb]

        item_name = self.items[j]
        past_dates = self.dates[t0:t0 + Lx]
        fut_dates  = self.dates[t0 + Lx:t0 + Lx + Ly]
        past_flags = self._flags_for_dates(item_name, past_dates)           # [Lx, 3]
        fut_flags  = self._flags_for_dates(item_name, fut_dates)            # [Ly, 3]

        past_cal = np.concatenate([past_cal_base, past_flags], axis=1)      # [Lx, Cb+3]
        fut_cal  = np.concatenate([fut_cal_base,  fut_flags ], axis=1)      # [Ly, Cb+3]

        store_idx = self.item_store_idx[j]
        cat_idx = self.item_cat_idx[j]
        type_idx = self.item_type_idx[j]
        cluster_idx = self.item_cluster_idx[j]
        sample_w = self.sample_weights[j]

        mean = x_in.mean()
        std = x_in.std()
        slope = (x_in[-1] - x_in[0]) / max(1, Lx)
        pos_mask = (y > 0).astype(np.float32)

        return {
            "x": torch.from_numpy(x_in).float(),
            "y": torch.from_numpy(y_out).float(),
            "past_cal": torch.from_numpy(past_cal).float(),
            "fut_cal": torch.from_numpy(fut_cal).float(),
            "store_idx": torch.tensor(store_idx, dtype=torch.long),
            "cat_idx": torch.tensor(cat_idx, dtype=torch.long),
            "type_idx": torch.tensor(type_idx, dtype=torch.long),
            "cluster_idx": torch.tensor(cluster_idx, dtype=torch.long),
            "sample_w": torch.tensor(sample_w, dtype=torch.float32),
            "pos_mask": torch.from_numpy(pos_mask).float(),
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
            "slope": torch.tensor(slope, dtype=torch.float32),
        }

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
    
def safe_cuda_empty_cache():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

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
# Utils for decomposition
# =====================
def moving_average_replicate(x: torch.Tensor, kernel_size: int = 25) -> torch.Tensor:
    """
    x: [B, L], replicate 패딩 후 same-length 평균 이동
    짧은 시퀀스(L <= kernel)에서는 전체 평균으로 대체해 왜곡/패딩 비용 방지.
    """
    B, L = x.shape
    if kernel_size >= L:
        # [B, L]로 브로드캐스트된 전체 평균
        return x.mean(dim=-1, keepdim=True).expand_as(x)
    
    pad = kernel_size // 2
    x_pad = F.pad(x.unsqueeze(1), (pad, pad), mode='replicate')  # [B,1,L+2p]
    weight = torch.ones(1, 1, kernel_size, device=x.device, dtype=x.dtype) / kernel_size
    trend = F.conv1d(x_pad, weight, stride=1).squeeze(1)  # [B,L]
    return trend

def adaptive_ma(x: torch.Tensor, base: int = 25) -> torch.Tensor:
    L = x.size(-1)
    k = min(base, max(3, L // 4))  # 길이가 짧으면 커널도 같이 축소
    return moving_average_replicate(x, kernel_size=k)

# =====================
# N-HiTS (FiLM + Horizon Emb + Temp)
# =====================
class NHiTSBlock(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, dropout,
                 pooling_mode="MaxPool1d", n_pool_kernel_size=2, interpolation_mode="linear",
                 cond_dim=0, activation="silu", use_layernorm=True):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.n_pool_kernel_size = int(n_pool_kernel_size)

        if pooling_mode == "MaxPool1d":
            self.pool = nn.MaxPool1d(kernel_size=self.n_pool_kernel_size,
                                     stride=self.n_pool_kernel_size, ceil_mode=True)
        else:
            self.pool = nn.AvgPool1d(kernel_size=self.n_pool_kernel_size,
                                     stride=self.n_pool_kernel_size, ceil_mode=True)

        act = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
        pooled_size = math.ceil(input_size / self.n_pool_kernel_size) + int(cond_dim)

        layers = [nn.Linear(pooled_size, hidden_size)]
        if use_layernorm: layers += [nn.LayerNorm(hidden_size)]
        layers += [act(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size)]
            if use_layernorm: layers += [nn.LayerNorm(hidden_size)]
            layers += [act(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)

        # FiLM conditioning
        self.film_gamma = nn.Linear(cond_dim, hidden_size) if cond_dim > 0 else None
        self.film_beta  = nn.Linear(cond_dim, hidden_size) if cond_dim > 0 else None

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, cond=None):
        x = x.unsqueeze(1)               # [B,1,L]
        pooled = self.pool(x).squeeze(1) # [B,Lp]
        if cond is not None:
            pooled = torch.cat([pooled, cond], dim=-1)
        h = self.mlp(pooled)

        if cond is not None and self.film_gamma is not None:
            gamma = self.film_gamma(cond)
            beta  = self.film_beta(cond)
            h = h * (1 + gamma) + beta

        return self.output_layer(h)      # [B,out_len]

class EnhancedNHiTSModel(nn.Module):
    def __init__(self, in_len: int, out_len: int, cal_dim: int, n_stores: int,
                 n_categories: int, n_types: int, n_clusters: int, cfg: EnhancedNHiTSConfig):
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

        # meta embeddings (+cluster)
        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb   = nn.Embedding(n_categories, 32)
        self.type_emb  = nn.Embedding(n_types, 16)
        self.cluster_emb = nn.Embedding(n_clusters, 32)
        self.cal_proj  = nn.Linear(cal_dim, 128)
        meta_dim = 64 + 32 + 16 + 32 + 128

        # blocks with conditioning
        self.blocks = nn.ModuleList([
            NHiTSBlock(
                in_len, out_len, cfg.hidden, cfg.n_layers, cfg.dropout,
                pooling_mode=cfg.pooling_mode, n_pool_kernel_size=cfg.n_pool_kernel_size[i],
                interpolation_mode=cfg.interpolation_mode,
                cond_dim=meta_dim, activation="silu", use_layernorm=True
            )
            for i in range(cfg.n_blocks)
        ])

        # trend basis
        self.basis_weights = nn.ParameterList([
            nn.Parameter(torch.randn(out_len, max(1, out_len // max(1, cfg.n_freq_downsample[i]))))
            for i in range(cfg.n_blocks)
        ])

        # meta heads
        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.SiLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, out_len)
        )

        # prob head
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 3, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.SiLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden // 2), nn.LayerNorm(cfg.hidden // 2), nn.SiLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden // 2, out_len)
        )

        # NEW: Horizon embedding bias + temperature for calibration
        self.horizon_emb = nn.Embedding(out_len, cfg.hidden)
        self.register_buffer("h_index", torch.arange(out_len), persistent=False)
        nn.init.zeros_(self.horizon_emb.weight)

        self.temp = nn.Parameter(torch.tensor(1.0))
        self.temp_emb = nn.Embedding(n_clusters, 1)
        nn.init.zeros_(self.temp_emb.weight)

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx, cluster_idx, mean, std, slope):
        store_emb = self.store_emb(store_idx)
        cat_emb   = self.cat_emb(cat_idx)
        type_emb  = self.type_emb(type_idx)
        cl_emb    = self.cluster_emb(cluster_idx)
        cal_emb   = self.cal_proj(torch.cat([past_cal, fut_cal], dim=1).mean(dim=1))
        meta_feat = torch.cat([store_emb, cat_emb, type_emb, cl_emb, cal_emb], dim=-1)

        outs = []
        for i, block in enumerate(self.blocks):
            o = block(x, cond=meta_feat)
            if self.stack_types[i] == "trend":
                basis = self.basis_weights[i]          # [out_len, k]
                o = o @ basis                           # [B, k]
                o = F.interpolate(o.unsqueeze(1), size=self.out_len,
                                  mode='linear', align_corners=False).squeeze(1)
            outs.append(o)
        nhits_out = torch.stack(outs, dim=0).sum(dim=0)

        # horizon bias
        h_emb = self.horizon_emb(self.h_index)         # [out_len, hidden]
        h_bias = h_emb.mean(dim=-1).unsqueeze(0)       # [1, out_len]
        nhits_out = nhits_out + h_bias

        value_pred = nhits_out + self.meta_integration(meta_feat)

        # 확률 헤드 (한 번만 호출 + 전역/클러스터 온도 보정)
        prob_input  = torch.cat([meta_feat, mean.unsqueeze(1), std.unsqueeze(1), slope.unsqueeze(1)], dim=-1)
        base_logits = self.prob_head(prob_input)       # [B, out_len]

        cluster_delta = self.temp_emb(cluster_idx).squeeze(-1)  # [B]
        total_temp = (self.temp + cluster_delta).clamp(0.5, 3.0).to(base_logits.dtype)  # [B]
        prob_logits = base_logits / total_temp.unsqueeze(-1)

        return value_pred, prob_logits

# =====================
# PatchTST (보조채널 + CrossAttn + Temp)
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

class MultiChannelPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.lin = nn.Linear(self.patch_len, d_model)   # P → d_model
        self.channel_weights = nn.Parameter(torch.ones(in_channels))  # [C]

    def forward(self, x):  # x: [B, C, L]
        B, C, L = x.shape
        if L < self.patch_len:
            pad = self.patch_len - L
            pad_val = x[:, :, :1].repeat(1, 1, pad)
            x = torch.cat([pad_val, x], dim=-1)
            L = x.size(-1)

        # [B, C, Np, P]
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # [B, C, Np, d_model]
        tokens = self.lin(patches)

        # 채널별 학습 가중합 → [B, Np, d_model]
        w = F.softmax(self.channel_weights, dim=0).view(1, C, 1, 1)
        tokens = (tokens * w).sum(dim=1)

        pe = _build_sinusoidal_pos(tokens.size(1), tokens.size(-1), tokens.device, tokens.dtype)
        return tokens + pe
    
class ImprovedPatchTSTTiny(nn.Module):
    def __init__(self, cfg: EnhancedNHiTSConfig, in_len: int, out_len: int,
                 cal_dim: int, n_stores: int, n_categories: int, n_types: int, n_clusters: int):
        super().__init__()
        p_len = max(2, min(cfg.ptst_patch_len, max(2, in_len // 8)))
        stride = max(1, p_len // 2)
        self.embed = MultiChannelPatchEmbed(in_channels=3, patch_len=p_len, stride=stride, d_model=cfg.ptst_d_model)

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

        self.cross_q = nn.Linear(cfg.ptst_d_model, cfg.ptst_d_model)
        self.cross_k = nn.Linear(cfg.ptst_d_model, cfg.ptst_d_model)
        self.cross_v = nn.Linear(cfg.ptst_d_model, cfg.ptst_d_model)
        self.cross_attn = nn.MultiheadAttention(cfg.ptst_d_model, cfg.ptst_nhead, batch_first=True)

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
        self.cluster_emb = nn.Embedding(n_clusters, 32)
        self.cal_proj = nn.Linear(cal_dim, 128)

        meta_dim = 64 + 32 + 16 + 32 + 128 + max(16, cfg.ptst_d_model // 2)
        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, cfg.ptst_head_hidden),
            nn.GELU(), nn.Dropout(cfg.ptst_dropout),
            nn.Linear(cfg.ptst_head_hidden, max(32, cfg.ptst_head_hidden // 2)),
            nn.GELU(), nn.Dropout(cfg.ptst_dropout),
            nn.Linear(max(32, cfg.ptst_head_hidden // 2), out_len)
        )
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 3, cfg.ptst_head_hidden),
            nn.GELU(), nn.Dropout(cfg.ptst_dropout),
            nn.Linear(cfg.ptst_head_hidden, max(32, cfg.ptst_head_hidden // 2)),
            nn.GELU(), nn.Dropout(cfg.ptst_dropout),
            nn.Linear(max(32, cfg.ptst_head_hidden // 2), out_len)
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx, cluster_idx, mean, std, slope):
        # 보조채널: 이동평균7, 지난주
        if x.size(1) >= 7:
            ma7 = x.unfold(dimension=1, size=7, step=1).mean(dim=-1)
            ma7 = F.pad(ma7, (x.size(1)-ma7.size(1), 0), mode='replicate')
            lastweek = torch.roll(x, shifts=+7, dims=1)
        else:
            ma7 = x
            lastweek = x
        aux = torch.stack([x, ma7, lastweek], dim=1)  # [B,3,L]

        z = self.embed(aux)           # [B,Np,D]
        z_enc = self.encoder(z)       # [B,Np,D]

        # Local (최근 패치) → Global cross-attn
        num_local = max(1, z_enc.size(1) // 6)
        Q = self.cross_q(z_enc[:, -num_local:, :])
        K = self.cross_k(z_enc)
        V = self.cross_v(z_enc)
        z_cross, _ = self.cross_attn(Q, K, V)   # [B,num_local,D]

        z_global = z_enc.mean(dim=1)
        z_local  = z_cross.mean(dim=1)
        z_last   = z_enc[:, -1, :]
        z_max, _ = z_enc.max(dim=1)
        z_first  = z_enc[:, 0, :]

        available_vecs = [z_global, z_local, z_last, z_max, z_first]
        head_preds = []
        for i, head in enumerate(self.multi_heads):
            vec = available_vecs[i % len(available_vecs)]
            head_preds.append(head(vec))
        seq_out = torch.stack(head_preds, dim=0).mean(dim=0)

        time_info = torch.cat([past_cal.mean(dim=1), fut_cal.mean(dim=1)], dim=-1)
        time_emb = self.time_proj(time_info)

        store_emb = self.store_emb(store_idx)
        cat_emb = self.cat_emb(cat_idx)
        type_emb = self.type_emb(type_idx)
        cl_emb = self.cluster_emb(cluster_idx)
        cal_emb = self.cal_proj(torch.cat([past_cal, fut_cal], dim=1).mean(dim=1))
        meta_feat = torch.cat([store_emb, cat_emb, type_emb, cl_emb, cal_emb, time_emb], dim=-1)

        meta_adjustment = self.meta_integration(meta_feat)
        value_pred = seq_out + self.residual_weight * meta_adjustment

        prob_logits = self.prob_head(torch.cat([meta_feat, mean.unsqueeze(1), std.unsqueeze(1), slope.unsqueeze(1)], dim=-1))
        prob_logits = prob_logits / self.temp.clamp(min=0.5, max=3.0)
        return value_pred, prob_logits

# =====================
# Improved TimesNet (분해 + 멀티주기 + Temp)
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
        # x_freq: [B, C, F], top_idx: [B, K]
        Fbins = x_freq.size(-1)
        mask = torch.zeros(x_freq.size(0), 1, Fbins, device=x_freq.device, dtype=x_freq.real.dtype)
        mask.scatter_(2, top_idx.unsqueeze(1), 1.0)        # [B,1,F], 1-hot at top-k freqs
        x_freq_filt = x_freq * mask                          # broadcast over channel C
        x_filt32 = torch.fft.irfft(x_freq_filt, n=L, dim=-1)
        return x_filt32.to(in_dtype)

class ImprovedTimesBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, seq_len: int, kernels=(3,5,7), top_k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.in_ch = int(in_ch); self.out_ch = int(out_ch); self.seq_len = int(seq_len)
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
            branches.append(nn.Sequential(
                nn.Conv1d(self.in_ch, ch, kernel_size=k, padding=k//2), nn.GELU(),
                nn.Conv1d(ch, ch, kernel_size=k, padding=k//2), nn.GELU(),
            ))
        self.conv1d_branches = nn.ModuleList(branches)
        self.br_sum_ch = sum(branch_chs)
        self.channel_proj = nn.Conv1d(self.br_sum_ch, self.out_ch, kernel_size=1) if self.br_sum_ch != self.out_ch else nn.Identity()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.out_ch, affine=True)        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        L_orig = L

        xf = self.freq_block(x)
        period = max(2, self.period_base)

        # 패딩 길이 계산
        pad_len = 0
        if L < period:
            pad_len = period - L
        elif L % period != 0:
            pad_len = period - (L % period)

        # 2D/1D 경로 동일 패딩
        if pad_len > 0:
            xf = F.pad(xf, (0, pad_len), mode="replicate")
            x_pad = F.pad(x,  (0, pad_len), mode="replicate")
            L_pad = L + pad_len
        else:
            x_pad = x
            L_pad = L

        target_len = max(1, L_pad // period)
        x2d = xf.reshape(B, C, period, target_len).contiguous()
        z2d = self.conv2d(x2d).reshape(B, self.out_ch, L_pad)

        # 1D 분기들
        z1d_list = [branch(x_pad) for branch in self.conv1d_branches]
        z1d = torch.cat(z1d_list, dim=1)
        z1d = self.channel_proj(z1d)

        # 원길이 복원
        if pad_len > 0:
            z2d = z2d[:, :, :L_orig]
            z1d = z1d[:, :, :L_orig]

        out = z2d + z1d
        # 동적 LayerNorm 대체 → 채널 기준 정규화(길이 L에 독립)
        out = self.norm(out)
        out = self.dropout(out)
        return out
    
class ImprovedTimesNetTiny(nn.Module):
    def __init__(self, cfg: EnhancedNHiTSConfig, in_len: int, out_len: int,
                 cal_dim: int, n_stores: int, n_categories: int, n_types: int, n_clusters: int):
        super().__init__()
        C = max(32, min(cfg.tnet_channels, 96)); C = (C // 8) * 8
        self.stem_trend = nn.Conv1d(1, C, kernel_size=3, padding=1)
        self.stem_season = nn.Conv1d(1, C, kernel_size=3, padding=1)

        self.blk7  = ImprovedTimesBlock(C, C, seq_len=in_len, kernels=(3,5,7), top_k=3, dropout=cfg.tnet_dropout)
        self.blk14 = ImprovedTimesBlock(C, C, seq_len=in_len, kernels=(3,5,7), top_k=3, dropout=cfg.tnet_dropout)
        self.blk28 = ImprovedTimesBlock(C, C, seq_len=in_len, kernels=(3,5,7), top_k=3, dropout=cfg.tnet_dropout)

        self.blk7.period_base  = 7
        self.blk14.period_base = 14
        self.blk28.period_base = 28

        head_hidden = max(128, min(cfg.tnet_head_hidden, 192))
        self.head_fc1 = nn.Linear(C*2, head_hidden)
        self.head_fc2 = nn.Linear(head_hidden, out_len)
        self.head_ln = nn.LayerNorm(head_hidden)
        self.head_drop = nn.Dropout(cfg.tnet_dropout)

        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb   = nn.Embedding(n_categories, 32)
        self.type_emb  = nn.Embedding(n_types, 16)
        self.cluster_emb = nn.Embedding(n_clusters, 32)
        self.cal_proj  = nn.Linear(cal_dim, 128)
        meta_dim = 64 + 32 + 16 + 32 + 128
        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, head_hidden), nn.LayerNorm(head_hidden), nn.SiLU(),
            nn.Dropout(cfg.tnet_dropout), nn.Linear(head_hidden, out_len),
        )
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 3, head_hidden), nn.LayerNorm(head_hidden), nn.SiLU(),
            nn.Dropout(cfg.tnet_dropout), nn.Linear(head_hidden, out_len),
        )
        self.meta_res_weight = nn.Parameter(torch.tensor(0.10))
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx, cluster_idx, mean, std, slope):
        trend = adaptive_ma(x, base=25)
        season = x - trend

        zt = self.stem_trend(trend.unsqueeze(1))     # [B,C,L]
        zs = self.stem_season(season.unsqueeze(1))   # [B,C,L]

        z7_t  = self.blk7(zt);   z7_s  = self.blk7(zs)
        z14_t = self.blk14(zt);  z14_s = self.blk14(zs)
        z28_t = self.blk28(zt);  z28_s = self.blk28(zs)

        zt_all = (zt + z7_t + z14_t + z28_t) / 4.0
        zs_all = (zs + z7_s + z14_s + z28_s) / 4.0

        zt_mean = zt_all.mean(dim=-1)
        zs_mean = zs_all.mean(dim=-1)
        z_cat = torch.cat([zt_mean, zs_mean], dim=-1)   # [B,2C]

        h = self.head_ln(self.head_fc1(z_cat))
        seq_out = self.head_fc2(self.head_drop(F.silu(h)))

        store_emb = self.store_emb(store_idx)
        cat_emb   = self.cat_emb(cat_idx)
        type_emb  = self.type_emb(type_idx)
        cl_emb    = self.cluster_emb(cluster_idx)
        cal_all   = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)
        cal_emb   = self.cal_proj(cal_all)
        meta_feat = torch.cat([store_emb, cat_emb, type_emb, cl_emb, cal_emb], dim=-1)

        value_pred  = seq_out + self.meta_res_weight * self.meta_integration(meta_feat)
        prob_logits = self.prob_head(torch.cat([meta_feat, mean.unsqueeze(1), std.unsqueeze(1), slope.unsqueeze(1)], dim=-1))
        prob_logits = prob_logits / self.temp.clamp(min=0.5, max=3.0)
        return value_pred, prob_logits

# =====================
# GRU (Attention Pooling + Meta Gate + Temp)
# =====================
class GRUTiny(nn.Module):
    def __init__(self, cfg: EnhancedNHiTSConfig, in_len: int, out_len: int,
                 cal_dim: int, n_stores: int, n_categories: int, n_types: int, n_clusters: int):
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
        self.attn_w = nn.Linear(cfg.gru_hidden * d_mul, 1)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.gru_hidden * d_mul),
            nn.Linear(cfg.gru_hidden * d_mul, out_len)
        )
        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb = nn.Embedding(n_categories, 32)
        self.type_emb = nn.Embedding(n_types, 16)
        self.cluster_emb = nn.Embedding(n_clusters, 32)
        self.cal_proj = nn.Linear(cal_dim, 128)
        meta_dim = 64+32+16+32+128
        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, cfg.gru_head_hidden), nn.LayerNorm(cfg.gru_head_hidden), nn.SiLU(),
            nn.Dropout(0.1), nn.Linear(cfg.gru_head_hidden, out_len)
        )
        self.meta_gate = nn.Sequential(
            nn.Linear(meta_dim, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 3, cfg.gru_head_hidden), nn.LayerNorm(cfg.gru_head_hidden), nn.SiLU(),
            nn.Dropout(0.1), nn.Linear(cfg.gru_head_hidden, cfg.gru_head_hidden // 2),
            nn.LayerNorm(cfg.gru_head_hidden // 2), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(cfg.gru_head_hidden // 2, out_len)
        )
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx, cluster_idx, mean, std, slope):
        z_in = x.unsqueeze(-1)     # [B,L,1]
        z_out, _ = self.gru(z_in)  # [B,L,H*d]

        attn = self.attn_w(z_out).squeeze(-1)      # [B,L]
        attn = torch.softmax(attn, dim=1)
        z_ctx = torch.bmm(attn.unsqueeze(1), z_out).squeeze(1)   # [B,H*d]
        seq_out = self.head(z_ctx)

        store_emb = self.store_emb(store_idx)
        cat_emb = self.cat_emb(cat_idx)
        type_emb = self.type_emb(type_idx)
        cl_emb = self.cluster_emb(cluster_idx)
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)
        cal_emb = self.cal_proj(cal_all)
        meta_feat = torch.cat([store_emb, cat_emb, type_emb, cl_emb, cal_emb], dim=-1)

        meta_adj = self.meta_integration(meta_feat)     # [B, out_len]
        gate = self.meta_gate(meta_feat)                # [B, 1]
        value_pred = seq_out * gate + meta_adj * (1 - gate)

        prob_logits = self.prob_head(torch.cat([meta_feat, mean.unsqueeze(1), std.unsqueeze(1), slope.unsqueeze(1)], dim=-1))
        prob_logits = prob_logits / self.temp.clamp(min=0.5, max=3.0)
        return value_pred, prob_logits

# =====================
# Improved DLinear (7/14/28 + step-wise DOW/MON + Temp)
# =====================
class ImprovedDLinearTiny(nn.Module):
    def __init__(
        self,
        cfg: EnhancedNHiTSConfig,
        in_len: int,
        out_len: int,
        cal_dim: int,
        n_stores: int,
        n_categories: int,
        n_types: int,
        n_clusters: int,
        cal_feat_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.decomp_kernel = 25

        # ---- main heads (trend / seasonal / multi-scale) ----
        self.trend_linear = nn.Sequential(
            nn.Linear(in_len, cfg.dlin_head_hidden),
            nn.LayerNorm(cfg.dlin_head_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden, out_len),
        )
        self.seasonal_linear = nn.Sequential(
            nn.Linear(in_len, cfg.dlin_head_hidden),
            nn.LayerNorm(cfg.dlin_head_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden, out_len),
        )

        self.ws = [w for w in (7, 14, 28) if w <= in_len]
        self.multi_scale_linears = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(w, cfg.dlin_head_hidden),
                    nn.LayerNorm(cfg.dlin_head_hidden),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(cfg.dlin_head_hidden, out_len),
                )
                for w in self.ws
            ]
        )

        # ---- Horizon-wise step coefficients (DOW / MONTH) ----
        self.dow_step = nn.Linear(7, 1)
        self.month_step = nn.Linear(12, 1)

        # ---- meta embeddings ----
        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb = nn.Embedding(n_categories, 32)
        self.type_emb = nn.Embedding(n_types, 16)
        self.cluster_emb = nn.Embedding(n_clusters, 32)
        self.cal_proj = nn.Linear(cal_dim, 128)

        meta_dim = 64 + 32 + 16 + 32 + 128
        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, cfg.dlin_head_hidden * 2),
            nn.LayerNorm(cfg.dlin_head_hidden * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden * 2, cfg.dlin_head_hidden),
            nn.LayerNorm(cfg.dlin_head_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden, out_len),
        )
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 3, cfg.dlin_head_hidden),
            nn.LayerNorm(cfg.dlin_head_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden, cfg.dlin_head_hidden // 2),
            nn.LayerNorm(cfg.dlin_head_hidden // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.dlin_head_hidden // 2, out_len),
        )

        # component mixing
        self.component_weights = nn.Parameter(torch.ones(4 + len(self.multi_scale_linears)))
        self.component_dropout = nn.Dropout(p=0.15)

        # temperature for prob head
        self.temp = nn.Parameter(torch.tensor(1.0))

        # ---- indices for calendar features (safe, no hardcoding) ----
        names = cal_feat_names or []
        self.idx_dow   = names.index("dow")   if "dow" in names else None
        self.idx_month = names.index("month") if "month" in names else None

    def forward(
        self,
        x,                # [B, L]
        past_cal,         # [B, L, Ccal]
        fut_cal,          # [B, out_len, Ccal]
        store_idx, cat_idx, type_idx, cluster_idx,
        mean, std, slope  # [B], [B], [B]
    ):
        B, L = x.size()

        # ---- decomposition ----
        trend = adaptive_ma(x, base=self.decomp_kernel)
        seasonal = x - trend

        trend_pred = self.trend_linear(trend)           # [B, out]
        seasonal_pred = self.seasonal_linear(seasonal)  # [B, out]

        multi_scale_preds = [linear(x[:, -w:]) for w, linear in zip(self.ws, self.multi_scale_linears)]

        # ---- step-wise DOW / MONTH (robust to order changes) ----
        if self.idx_dow is not None and self.idx_dow < fut_cal.size(-1):
            dow_idx = fut_cal[..., self.idx_dow].long().clamp(0, 6)
        else:
            dow_idx = torch.zeros(B, self.out_len, dtype=torch.long, device=x.device)

        if self.idx_month is not None and self.idx_month < fut_cal.size(-1):
            month_idx = (fut_cal[..., self.idx_month] - 1).long().clamp(0, 11)
        else:
            month_idx = torch.zeros(B, self.out_len, dtype=torch.long, device=x.device)

        

        dow_oh = F.one_hot(dow_idx, num_classes=7).float()      # [B, out, 7]
        mon_oh = F.one_hot(month_idx, num_classes=12).float()   # [B, out, 12]
        dow_pred_steps = self.dow_step(dow_oh).squeeze(-1)      # [B, out]
        mon_pred_steps = self.month_step(mon_oh).squeeze(-1)    # [B, out]

        # ---- component mixing ----
        all_preds = [trend_pred, seasonal_pred, dow_pred_steps, mon_pred_steps] + multi_scale_preds
        weights = F.softmax(self.component_weights[: len(all_preds)], dim=0)
        weights = self.component_dropout(weights)
        weights = weights / (weights.sum() + 1e-8)
        combined_pred = sum(w * p for w, p in zip(weights, all_preds))    # [B, out]

        # ---- meta features ----
        store_emb = self.store_emb(store_idx)
        cat_emb = self.cat_emb(cat_idx)
        type_emb = self.type_emb(type_idx)
        cl_emb = self.cluster_emb(cluster_idx)

        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)  # [B, Ccal]
        cal_emb = self.cal_proj(cal_all)

        meta_feat = torch.cat([store_emb, cat_emb, type_emb, cl_emb, cal_emb], dim=-1)

        # value head
        meta_adjustment = self.meta_integration(meta_feat)
        value_pred = combined_pred + meta_adjustment                 # [B, out]

        # prob head
        prob_feat = torch.stack([mean, std, slope], dim=1)           # [B, 3]
        prob_logits = self.prob_head(torch.cat([meta_feat, prob_feat], dim=-1))  # [B, out]
        prob_logits = prob_logits / self.temp.clamp(min=0.5, max=3.0)

        return value_pred, prob_logits
    
# =====================
# Trainer (공용: OneCycleLR + EMA 안전)
# =====================
# 변경
class GenericTrainer:
    def __init__(self, cfg: EnhancedNHiTSConfig, dataset: EnhancedNHiTSDataset, model: nn.Module,
                 epochs: int, batch_size: int, base_lr: float, max_lr: float, weight_decay: float,
                 model_name: str, save_dir: str = "./checkpoints", run_tag: str = ""):
        self.cfg = cfg; self.dataset = dataset; self.model = model.to(cfg.device)
        self.device = torch.device(cfg.device)
        self.criterion = UltraEnhancedHurdleLoss(cfg.eps_smape, cfg.zero_weight, cfg.hurdle_lambda)
        self.ema = EMA(self.model, decay=cfg.ema_decay)
        self.epochs = epochs; self.batch_size = batch_size
        self.base_lr = base_lr; self.max_lr = max_lr; self.weight_decay = weight_decay
        self.model_name = model_name

        use_cuda_amp = bool(self.cfg.use_amp and torch.cuda.is_available() and self.device.type == "cuda")

        if self.model_name == "PatchTST":
            # 🔒 PatchTST만 AMP 완전 비활성 (FP32 + no scaler)
            self.amp_enabled = False
            self.amp_dtype   = torch.float32
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        else:
            if use_cuda_amp and torch.cuda.is_bf16_supported():
                # ✅ bf16 autocast (스케일러 불필요/비활성)
                self.amp_enabled = True
                self.amp_dtype   = torch.bfloat16
                self.scaler = torch.cuda.amp.GradScaler(enabled=False)
            elif use_cuda_amp:
                # ✅ fp16 autocast + GradScaler
                self.amp_enabled = True
                self.amp_dtype   = torch.float16
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            else:
                # CPU 또는 AMP 미사용
                self.amp_enabled = False
                self.amp_dtype   = torch.float32
                self.scaler = torch.cuda.amp.GradScaler(enabled=False)
         
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.run_tag = run_tag or model_name

    def _ckpt_path(self, kind: str):
        return os.path.join(self.save_dir, f"{self.run_tag}_{kind}.pt")

    def save_checkpoint(self, epoch, best_val, model_state=None, ema_applied=False, extra=None):
        state = {
            "epoch": epoch,
            "best_val": float(best_val),
            "model": (model_state or self.model.state_dict()),
            "optimizer": None if ema_applied else self.optim.state_dict(),
            "scheduler": None if ema_applied else self.sched.state_dict(),
            "amp": None if ema_applied else (self.scaler.state_dict() if self.scaler.is_enabled() else None),
            "cfg": self.cfg.__dict__,
        }
        if extra:
            state.update(extra)
        torch.save(state, self._ckpt_path("best" if ema_applied else "last"))

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=True)
        if ckpt.get("optimizer"):
            self.optim.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            self.sched.load_state_dict(ckpt["scheduler"])
        if ckpt.get("amp") and self.scaler.is_enabled():
            self.scaler.load_state_dict(ckpt["amp"])
        return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))

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
        if loader is None: return float('inf')
        self.model.eval()

        backup = None
        if use_ema:
            backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.ema.apply_to(self.model)

        total_loss = 0.0; total_weight = 0.0

        amp_ctx = (torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.amp_enabled)
           if self.device.type == "cuda" else torch.cuda.amp.autocast(enabled=False))

        with amp_ctx:
            for batch in loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                past_cal = batch["past_cal"].to(self.device)
                fut_cal = batch["fut_cal"].to(self.device)
                store_idx = batch["store_idx"].to(self.device)
                cat_idx = batch["cat_idx"].to(self.device)
                type_idx = batch["type_idx"].to(self.device)
                cluster_idx = batch["cluster_idx"].to(self.device)
                sample_w = batch["sample_w"].to(self.device)
                pos_mask = batch["pos_mask"].to(self.device)
                mean = batch["mean"].to(self.device)
                std = batch["std"].to(self.device)
                slope = batch["slope"].to(self.device)

                v_pred, p_logits = self.model(x, past_cal, fut_cal, store_idx, cat_idx, type_idx, cluster_idx, mean, std, slope)
                _, sample_loss, sw = self.criterion(v_pred, p_logits, y, pos_mask, sample_w)
                total_loss += (sample_loss * sw).sum().item()
                total_weight += sw.sum().item()

        if use_ema and backup is not None:
            self.model.load_state_dict(backup)
            del backup
            gc.collect(); 
            safe_cuda_empty_cache()

        if total_weight <= 0: return float('inf')
        return total_loss / total_weight

    def train_with_loaders(self, train_loader: DataLoader, val_loader: DataLoader, model_name="Model"):
        # ---- Optim / Sched ----
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
            pct_start=0.20,
            div_factor=max(1e-6, self.max_lr / max(1e-6, self.base_lr)),
        )

        # 내부에서도 접근 가능하도록 보관(로드/재개를 고려)
        self.optim = optim
        self.sched = sched

        # ---- Early stop / best ----
        patience = max(self.cfg.earlystop_patience_min, int(self.epochs * self.cfg.earlystop_patience_ratio))
        best_val = float("inf")
        best_state = None
        best_ema_state = None
        best_epoch = -1
        no_improve = 0

        # ---- Checkpoint path ----
        ckpt_dir = "./checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        tag = model_name.replace(" ", "_")
        last_ckpt_path = os.path.join(ckpt_dir, f"{tag}_last.pt")
        best_ckpt_path = os.path.join(ckpt_dir, f"{tag}_best.pt")

        def _get_ema_state_dict():
            # EMA 구현체에 따라 state_dict 또는 shadow를 지원
            if hasattr(self.ema, "state_dict"):
                return self.ema.state_dict()
            elif hasattr(self.ema, "shadow"):
                # dict 형태로 통일
                return {"shadow": {k: v.detach().cpu().clone() for k, v in self.ema.shadow.items()}}
            else:
                return None

        def _load_ema_state_dict(ema_state):
            if ema_state is None:
                return
            if hasattr(self.ema, "load_state_dict"):
                self.ema.load_state_dict(ema_state)
            elif "shadow" in ema_state:
                self.ema.shadow = {k: v.clone() for k, v in ema_state["shadow"].items()}

        def _save_ckpt(path, epoch, val_loss, is_best=False):
            try:
                torch.save(
                    {
                        "epoch": int(epoch),
                        "val_loss": float(val_loss),
                        "model": self.model.state_dict(),
                        "optimizer": self.optim.state_dict(),
                        "scheduler": self.sched.state_dict(),
                        "amp": (self.scaler.state_dict() if self.scaler.is_enabled() else None),
                        "ema_state_dict": _get_ema_state_dict(),  # ✅ EMA도 보관
                        "cfg": self.cfg.__dict__,
                        "is_best": bool(is_best),
                    },
                    path,
                )
            except Exception as e:
                print(f"[{model_name}] ⚠️ checkpoint save failed at {path}: {e}")

        try:
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                running_losses = []
                skipped_batches = 0

                use_autocast = bool(self.amp_enabled and self.device.type == "cuda")
                use_scaler = bool(self.scaler.is_enabled())

                for batch in train_loader:
                    # ----- batch to device -----
                    x = batch["x"].to(self.device); y = batch["y"].to(self.device)
                    past_cal = batch["past_cal"].to(self.device); fut_cal = batch["fut_cal"].to(self.device)
                    store_idx = batch["store_idx"].to(self.device); cat_idx = batch["cat_idx"].to(self.device)
                    type_idx = batch["type_idx"].to(self.device); cluster_idx = batch["cluster_idx"].to(self.device)
                    sample_w = batch["sample_w"].to(self.device); pos_mask = batch["pos_mask"].to(self.device)
                    mean = batch["mean"].to(self.device); std = batch["std"].to(self.device); slope = batch["slope"].to(self.device)

                    # ----- forward + loss (autocast) -----
                    try:
                        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=use_autocast):
                            v_pred, p_logits = self.model(
                                x, past_cal, fut_cal, store_idx, cat_idx, type_idx, cluster_idx, mean, std, slope
                            )
                            loss, _, _ = self.criterion(v_pred, p_logits, y, pos_mask, sample_w)

                        # NaN/inf 방어
                        if not torch.isfinite(loss):
                            skipped_batches += 1
                            continue

                        # ----- backward + step (GradScaler) -----
                        optim.zero_grad(set_to_none=True)
                        if use_scaler:
                            self.scaler.scale(loss).backward()
                            self.scaler.unscale_(optim)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                            self.scaler.step(optim)
                            self.scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                            optim.step()

                        sched.step()
                        self.ema.update(self.model)
                        running_losses.append(loss.detach().item())

                    except RuntimeError as e:
                        # OOM 등에서 배치 스킵
                        if "out of memory" in str(e).lower():
                            skipped_batches += 1
                            safe_cuda_empty_cache()
                            continue
                        else:
                            raise

                # ----- validation (EMA로 평가 후 복원은 evaluate 내부 처리) -----
                val_loss = self.evaluate(val_loader, use_ema=True)
                tr_mean = float(np.mean(running_losses)) if running_losses else float('nan')
                print(
                    f"[{model_name}] [Epoch {epoch:03d}] "
                    f"train_loss: {tr_mean:.5f}  val_loss: {val_loss:.5f}  "
                    f"(skipped:{skipped_batches})"
                )

                # 매 epoch 라스트 체크포인트
                _save_ckpt(last_ckpt_path, epoch, val_loss, is_best=False)

                # ----- early stopping / best 갱신 -----
                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch
                    # 모델/EMA 둘 다 '그 시점'을 보관
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    best_ema_state = _get_ema_state_dict()
                    _save_ckpt(best_ckpt_path, epoch, val_loss, is_best=True)
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"[{model_name}] ✅ Early stopping at epoch {epoch} (best@{best_epoch}, val={best_val:.5f})")
                        break

            # ---- 에폭 종료: 베스트로 복원 후 저장 ----
            if best_state is not None:
                self.model.load_state_dict(best_state)
            if best_ema_state is not None:
                _load_ema_state_dict(best_ema_state)
                self.ema.apply_to(self.model)  # ✅ 베스트 에폭의 EMA 적용
            else:
                # fallback: 현재 EMA라도 적용(권장X)
                self.ema.apply_to(self.model)

            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/{model_name}_best.pth"
            torch.save(self.model.state_dict(), save_path)
            print(f"[SAVE] Best(EMA) model saved → {save_path}")

            print(f"[{model_name}] Best val: {best_val:.5f} @ epoch {best_epoch}")
            return self.model, best_val

        except KeyboardInterrupt:
            print(f"\n[{model_name}] ⚠️ Interrupted by user. Saving last & best checkpoints...")
            # 중단 시점 저장
            _save_ckpt(last_ckpt_path, locals().get("epoch", -1), best_val, is_best=False)

            # 베스트 복원 + EMA도 복원
            if best_state is not None:
                self.model.load_state_dict(best_state)
            if best_ema_state is not None:
                _load_ema_state_dict(best_ema_state)
                self.ema.apply_to(self.model)

            # 베스트 체크포인트 다시 보관
            _save_ckpt(best_ckpt_path, best_epoch, best_val, is_best=True)
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
    type2idx: Dict,
    cluster2idx: Dict,
    menu_meta: Optional[Dict[str, Dict]] = None,  
) -> pd.DataFrame:
    device = torch.device(cfg.device)
    if test_df is None or len(test_df) == 0:
        return pd.DataFrame()

    tdf = test_df.copy()
    for col in (cfg.date_col, cfg.item_col, cfg.target_col):
        if col not in tdf.columns:
            return pd.DataFrame()

    tdf[cfg.date_col] = pd.to_datetime(tdf[cfg.date_col], errors="coerce")
    tdf = tdf.dropna(subset=[cfg.date_col])
    if len(tdf) == 0:
        return pd.DataFrame()

    tdf[cfg.target_col] = pd.to_numeric(tdf[cfg.target_col], errors="coerce").fillna(0.0).clip(lower=0)
    tdf[cfg.item_col] = tdf[cfg.item_col].astype(str)

    try:
        # 중복 날짜-아이템이 있을 수 있으니 sum으로 집계
        pivot = (
            tdf.pivot_table(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col, aggfunc="sum")
               .sort_index()
               .fillna(0.0)
        )
    except Exception:
        return pd.DataFrame()
    if pivot.shape[1] == 0:
        return pd.DataFrame()

    items  = list(pivot.columns.astype(str))
    dates  = list(pivot.index)
    values = pivot.values.astype(np.float32, copy=False)

    Lx = cfg.in_len
    if len(dates) < Lx:
        pad_rows = Lx - len(dates)
        if len(dates) > 0:
            # 일부만 있는 경우: 앞쪽 평균값으로 패딩 + 날짜 backfill
            k = min(7, len(dates))
            ref = np.nanmean(values[:k, :], axis=0, keepdims=True).astype(np.float32)
            ref = np.where(np.isfinite(ref), ref, 0.0)
            pad = np.repeat(ref, pad_rows, axis=0)
            start_date = dates[0]

            values = np.vstack([pad, values]).astype(np.float32, copy=False)
            backfill = [start_date - pd.Timedelta(days=i) for i in range(pad_rows, 0, -1)]
            dates = backfill + dates
        else:
            # 완전 비어있음: train_end_date 기준 Lx 길이 생성
            anchor = pd.to_datetime(cfg.train_end_date, errors="coerce")
            if pd.isna(anchor):
                anchor = pd.Timestamp.today().normalize()
            start_date = anchor - pd.Timedelta(days=Lx - 1)
            dates = [start_date + pd.Timedelta(days=i) for i in range(Lx)]
            values = np.zeros((Lx, values.shape[1]), dtype=np.float32)

    # 값/메모리 연속성 보장
    values = np.ascontiguousarray(values, dtype=np.float32)

    # === 여기 추가: 입력 텐서 x 생성 (로그 공간) ===
    # shape: [B, Lx]
    x_np = values[-Lx:, :].T
    if cfg.log1p:
        x_np = np.log1p(np.clip(x_np, a_min=0.0, a_max=None))
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)

    B = x.size(0)
    if B == 0:
        return pd.DataFrame()

    # ---- 캘린더 기본 ----
    last_date    = dates[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, cfg.out_len + 1)]
    holidays_set = set(pd.to_datetime(cfg.custom_holidays_list)) if cfg.custom_holidays_list else set()

    past_cal_base = build_enhanced_features(dates[-Lx:], holidays_set).drop(columns=["date"]).values.astype(np.float32)
    fut_cal_base  = build_enhanced_features(future_dates,     holidays_set).drop(columns=["date"]).values.astype(np.float32)

    # ---- 아이템별 출시/단종/계절 플래그 3채널 생성 → concat ----
    def flags_for_dates(item: str, dts):
        meta = (menu_meta or {}).get(str(item), {})
        launch = pd.to_datetime(meta.get("launch_date")) if meta.get("launch_date") else None
        disc   = pd.to_datetime(meta.get("discontinue_date")) if meta.get("discontinue_date") else None
        months = set(int(m) for m in meta.get("active_months", []))
        out = np.zeros((len(dts), 3), dtype=np.float32)
        for i, dd in enumerate(dts):
            out[i,0] = 1.0 if (launch is not None and dd >= launch) else 0.0
            out[i,1] = 1.0 if (disc   is not None and dd >  disc ) else 0.0
            out[i,2] = 1.0 if (months and int(dd.month) in months) else 0.0
        return out

    past_dates = dates[-Lx:]
    fut_dates  = future_dates
    past_cals, fut_cals = [], []
    for it in items:
        pf = np.concatenate([past_cal_base, flags_for_dates(it, past_dates)], axis=1)
        ff = np.concatenate([fut_cal_base,  flags_for_dates(it, fut_dates)],  axis=1)
        past_cals.append(pf); fut_cals.append(ff)

    past_cal_b = torch.from_numpy(np.stack(past_cals, axis=0)).to(device=device, dtype=torch.float32)  # [B, Lx, Cb+3]
    fut_cal_b  = torch.from_numpy(np.stack(fut_cals,  axis=0)).to(device=device, dtype=torch.float32)  # [B, Ly, Cb+3]

    # ---- 메타 인덱스 ----
    stores = [parse_store_name(it) for it in items]
    menus  = [parse_menu_name(it)  for it in items]

    store2idx   = store2idx   or {"__default__": 0}
    cat2idx     = cat2idx     or {"__default__": 0}
    type2idx    = type2idx    or {"__default__": 0}
    cluster2idx = cluster2idx or {"__default__": 0}

    safe_store_default   = next(iter(store2idx.values()), 0)
    safe_cat_default     = next(iter(cat2idx.values()), 0)
    safe_type_default    = next(iter(type2idx.values()), 0)
    safe_cluster_default = next(iter(cluster2idx.values()), 0)

    store_idx   = torch.as_tensor([store2idx.get(s, safe_store_default) for s in stores], device=device, dtype=torch.long)
    cat_idx     = torch.as_tensor([cat2idx.get(get_menu_category(m), safe_cat_default) for m in menus], device=device, dtype=torch.long)
    type_idx    = torch.as_tensor([type2idx.get(get_store_type(s), safe_type_default) for s in stores], device=device, dtype=torch.long)
    cluster_idx = torch.as_tensor([cluster2idx.get(cluster_mapping.get(it, 0), safe_cluster_default) for it in items], device=device, dtype=torch.long)

    # ---- 보조 통계(로그 공간) ----
    mean  = x.mean(dim=1)
    std   = x.std(dim=1)
    slope = (x[:, -1] - x[:, 0]) / max(1, Lx)

    # ---- AMP 컨텍스트 ----
    use_cuda_amp = bool(cfg.use_amp and torch.cuda.is_available() and device.type == "cuda")
    if use_cuda_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_cuda_amp else torch.cuda.amp.autocast(enabled=False)

    # ---- 추론 ----
    model.eval()
    with amp_ctx:
        v_pred_log, p_logits = model(
            x, past_cal_b, fut_cal_b, store_idx, cat_idx, type_idx, cluster_idx, mean, std, slope
        )
        y_val  = torch.expm1(v_pred_log).clamp_min(0.0)
        y_prob = torch.sigmoid(p_logits)
        y_hat  = (y_prob * y_val).clamp_min(0.0).detach().cpu().numpy()

    out = pd.DataFrame(
        y_hat.T,
        index =[f"D+{i}" for i in range(1, cfg.out_len + 1)],
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
    cal_dim = getattr(ds, "cal_dim_for_model", ds.base_cal_feats.shape[1])  # ✅ 변경 포인트

    if name == "N-HiTS":
        hidden = trial.suggest_categorical("nh_hidden", [256, 384, 512]) if trial else cfg.hidden
        n_layers = trial.suggest_categorical("nh_layers", [1, 2, 3]) if trial else cfg.n_layers
        dropout = trial.suggest_float("nh_dropout", 0.0, 0.3) if trial else cfg.dropout
        nh_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__, "hidden": hidden, "n_layers": n_layers, "dropout": dropout})
        return EnhancedNHiTSModel(cfg.in_len, cfg.out_len, cal_dim,
                                  ds.n_stores, ds.n_categories, ds.n_types, ds.n_clusters, nh_cfg)

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
        return ImprovedPatchTSTTiny(pt_cfg, cfg.in_len, cfg.out_len, cal_dim,
                                    ds.n_stores, ds.n_categories, ds.n_types, ds.n_clusters)

    if name == "TimesNet":
        channels = trial.suggest_categorical("tn_channels", [96, 128, 160]) if trial else cfg.tnet_channels
        blocks   = trial.suggest_categorical("tn_blocks", [2, 3, 4]) if trial else cfg.tnet_blocks
        drop     = trial.suggest_float("tn_dropout", 0.0, 0.3) if trial else cfg.tnet_dropout
        tn_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__,
            "tnet_channels": channels, "tnet_blocks": blocks, "tnet_dropout": drop
        })
        return ImprovedTimesNetTiny(tn_cfg, cfg.in_len, cfg.out_len, cal_dim,
                                    ds.n_stores, ds.n_categories, ds.n_types, ds.n_clusters)

    if name == "GRU":
        hidden = trial.suggest_categorical("gru_hidden", [128, 192, 256]) if trial else cfg.gru_hidden
        layers = trial.suggest_categorical("gru_layers", [1, 2, 3]) if trial else cfg.gru_layers
        drop   = trial.suggest_float("gru_dropout", 0.0, 0.3) if trial else cfg.gru_dropout
        bidi   = trial.suggest_categorical("gru_bidi", [False, True]) if trial else cfg.gru_bidirectional
        gr_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__,
            "gru_hidden": hidden, "gru_layers": layers, "gru_dropout": drop, "gru_bidirectional": bidi
        })
        return GRUTiny(gr_cfg, cfg.in_len, cfg.out_len, cal_dim,
                       ds.n_stores, ds.n_categories, ds.n_types, ds.n_clusters)

    if name == "DLinear":
        head_h = trial.suggest_categorical("dl_head_hidden", [256, 384, 512]) if trial else cfg.dlin_head_hidden
        dl_cfg = EnhancedNHiTSConfig(**{**cfg.__dict__, "dlin_head_hidden": head_h})
        return ImprovedDLinearTiny(
            dl_cfg, cfg.in_len, cfg.out_len, cal_dim,
            ds.n_stores, ds.n_categories, ds.n_types, ds.n_clusters,
            cal_feat_names=ds.cal_feat_names)

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

def build_final_model_with_best(model_name: str, cfg: EnhancedNHiTSConfig,
                                ds: EnhancedNHiTSDataset, best_params: Optional[dict]) -> nn.Module:
    cal_dim = getattr(ds, "cal_dim_for_model", ds.base_cal_feats.shape[1]) 
    if best_params is None:
        # cal_dim만 바꿔서 전달
        if model_name == "DLinear":
            return ImprovedDLinearTiny(cfg, cfg.in_len, cfg.out_len, cal_dim,
                                       ds.n_stores, ds.n_categories, ds.n_types, ds.n_clusters,
                                       cal_feat_names=ds.cal_feat_names)
        return build_model_by_name(model_name, cfg, ds, None)

    tmp = dict(cfg.__dict__)
    if model_name == "N-HiTS":
        if "nh_hidden" in best_params:  tmp["hidden"]   = best_params["nh_hidden"]
        if "nh_layers" in best_params:  tmp["n_layers"] = best_params["nh_layers"]
        if "nh_dropout" in best_params: tmp["dropout"]  = best_params["nh_dropout"]
    elif model_name == "PatchTST":
        mapping = {"pt_d_model":"ptst_d_model","pt_nhead":"ptst_nhead","pt_nlayers":"ptst_num_layers",
                   "pt_patch":"ptst_patch_len","pt_stride":"ptst_stride","pt_dropout":"ptst_dropout",
                   "ptst_ff_mult":"ptst_ff_mult"}
        for k, v in mapping.items():
            if k in best_params: tmp[v] = best_params[k]
    elif model_name == "TimesNet":
        mapping = {"tn_channels":"tnet_channels","tn_blocks":"tnet_blocks","tn_dropout":"tnet_dropout"}
        for k, v in mapping.items():
            if k in best_params: tmp[v] = best_params[k]
    elif model_name == "GRU":
        mapping = {"gru_hidden":"gru_hidden","gru_layers":"gru_layers",
                   "gru_dropout":"gru_dropout","gru_bidi":"gru_bidirectional"}
        for k, v in mapping.items():
            if k in best_params: tmp[v] = best_params[k]
    elif model_name == "DLinear":
        if "dl_head_hidden" in best_params:
            tmp["dlin_head_hidden"] = best_params["dl_head_hidden"]

    new_cfg = EnhancedNHiTSConfig(**tmp)

    if model_name == "DLinear":
        return ImprovedDLinearTiny(
            new_cfg, cfg.in_len, cfg.out_len, cal_dim,
            ds.n_stores, ds.n_categories, ds.n_types, ds.n_clusters,
            cal_feat_names=ds.cal_feat_names
        )
    else:
        # build_model_by_name 내부에서 cal_dim=ds.cal_dim_for_model을 사용
        return build_model_by_name(model_name, new_cfg, ds, None)

# =====================
# Main
# =====================
if __name__ == "__main__":
    cfg = EnhancedNHiTSConfig()
    if cfg.store_weights is None: cfg.store_weights = DEFAULT_STORE_WEIGHTS
    if cfg.custom_holidays_list is None: cfg.custom_holidays_list = DEFAULT_CUSTOM_HOLIDAYS
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=== Enhanced 4-Model + N-HiTS Ensemble 시작 ===")

    if not os.path.exists(cfg.train_csv):
        raise FileNotFoundError(f"학습 파일을 찾을 수 없습니다: {cfg.train_csv}")
    train_df = pd.read_csv(cfg.train_csv)
    menu_meta = build_menu_meta(train_df, cfg) 
    ds = EnhancedNHiTSDataset(cfg, train_df, menu_meta=menu_meta)  

    # ===== (옵션) Optuna per-model 튜닝 =====
    best_params_all = {}
    if cfg.USE_OPTUNA:
        if not HAS_OPTUNA:
            print("⚠️ cfg.USE_OPTUNA=True지만 optuna가 설치되어 있지 않습니다. 튜닝은 건너뜁니다.")
        else:
            tune_mask_val = make_val_mask_by_week(ds, cfg.cv_fold_end_dates[0])
            for name in ["N-HiTS", "PatchTST", "TimesNet", "GRU", "DLinear"]:
                study, best_params = run_optuna_for_model(name, cfg, ds, tune_mask_val)
                best_params_all[name] = best_params
            os.makedirs("./data", exist_ok=True)
            with open("./data/optuna_best_params.json", "w", encoding="utf-8") as f:
                json.dump(best_params_all, f, ensure_ascii=False, indent=2)

    # ===== 모델 학습 (멀티 폴드 평균 val) =====
    model_names = ["PatchTST", "TimesNet", "GRU", "DLinear", "N-HiTS"] 
    trained_models: Dict[str, nn.Module] = {}
    avg_val_losses: Dict[str, float] = {}

    for name in model_names:
        print(f"📚 {name} 학습...")

        # 동일 초기값 확보
        base_model = build_final_model_with_best(name, cfg, ds, best_params_all.get(name) if cfg.USE_OPTUNA else None)
        init_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        del base_model
        gc.collect()
        safe_cuda_empty_cache()

        fold_vals = []
        best_fold = None
        best_fold_state = None

        for fold_i, end_date in enumerate(cfg.cv_fold_end_dates, start=1):
            mask_val = make_val_mask_by_week(ds, end_date)
            model_f = build_final_model_with_best(name, cfg, ds, best_params_all.get(name) if cfg.USE_OPTUNA else None)
            model_f.load_state_dict(init_state, strict=True)
            trainer = GenericTrainer(cfg, ds, model_f, epochs=cfg.EPOCHS_FULL, batch_size=cfg.BATCH_FULL,base_lr=cfg.BASE_LR_FULL, 
                                     max_lr=cfg.MAX_LR_FULL, weight_decay=cfg.WD_FULL, model_name=name,save_dir="./checkpoints",run_tag=f"{name}-F{fold_i}")
            train_loader, val_loader = trainer.make_loaders_from_mask(mask_val)
            model_f, val_loss = trainer.train_with_loaders(train_loader, val_loader, model_name=f"{name}-F{fold_i}")
            fold_vals.append(float(val_loss))

            if (best_fold is None) or (val_loss < fold_vals[best_fold]):
                best_fold = fold_i - 1
                best_fold_state = {k: v.detach().cpu().clone() for k, v in model_f.state_dict().items()}

            del trainer, train_loader, val_loader, model_f
            gc.collect()
            safe_cuda_empty_cache()

        avg_val = float(np.mean(fold_vals)) if len(fold_vals) > 0 else float('inf')
        avg_val_losses[name] = avg_val
        print(f"[VAL(avg over folds)] {name}={avg_val:.5f}")

        final_model = build_final_model_with_best(name, cfg, ds, best_params_all.get(name) if cfg.USE_OPTUNA else None)
        if best_fold_state is not None:
            final_model.load_state_dict(best_fold_state, strict=True)
        trained_models[name] = final_model.to(cfg.device)

    # ===== 가중치 산출(역손실 안전화) + 사용가능 모델만 재정규화 =====
    vals_in_order = [avg_val_losses.get(n, np.nan) for n in model_names]
    base_w = invloss_weights(vals_in_order)
    weights = dict(zip(model_names, base_w))
    print("[Ensemble Weights (pre)] " + "  ".join([f"{k}={weights[k]:.3f}" for k in model_names]))

    # ===== 추론 & 제출 =====
    print("🔮 5-Model 앙상블 예측...")
    test_files = sorted(glob.glob(cfg.test_glob))
    if not os.path.exists(cfg.submission_template_csv):
        raise FileNotFoundError(f"제출 템플릿 파일을 찾을 수 없습니다: {cfg.submission_template_csv}")

    sub_template = pd.read_csv(cfg.submission_template_csv)
    template_cols = sub_template.columns.tolist()
    item_cols = [c for c in template_cols if c != "영업일자"]

    all_blocks = []

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
                df = predict_one_file_generic(
                    cfg, trained_models[name], tdf,
                    ds.store2idx, ds.cat2idx, ds.type2idx, ds.cluster2idx  # ✅ 누락 보완
                )
                if df is None or df.empty:
                    print(f"  ⚠️ {name} 예측이 비어 있습니다. 스킵.")
                    continue
                df_preds[name] = df

            non_empty = [df for df in df_preds.values() if df is not None and not df.empty]
            if len(non_empty) == 0:
                print(f"  ⚠️ {test_file} 유효 예측이 없어 스킵합니다.")
                continue

            items_pred = [c for c in list(non_empty[0].columns) if c in item_cols]
            for name, df in df_preds.items():
                if df is not None and not df.empty:
                    df_preds[name] = df.reindex(columns=items_pred).fillna(0.0)

            active = [n for n, df in df_preds.items() if df is not None and not df.empty]
            w = np.array([weights[n] for n in active], dtype=np.float64)
            w = w / w.sum()  # ✅ 활성 모델만 재정규화

            mix = np.zeros_like(df_preds[active[0]].values, dtype=np.float64)
            for n, wn in zip(active, w):
                mix += wn * df_preds[n].values
            mix = np.clip(mix, 0, None)

            block = pd.DataFrame(mix,
                index=[f"D+{i}" for i in range(1, cfg.out_len + 1)],
                columns=items_pred
            )
            block = block.reindex(columns=item_cols, fill_value=0.0)
            block.index = [f"TEST_{test_idx:02d}+{k}일" for k in range(1, cfg.out_len + 1)]
            block.reset_index(inplace=True)
            block.rename(columns={"index": "영업일자"}, inplace=True)
            all_blocks.append(block)

    if len(all_blocks) == 0:
        print("⚠️ 유효한 예측 결과가 없어 제출 파일을 생성하지 않았습니다.")
    else:
        final_pred = pd.concat(all_blocks, axis=0, ignore_index=True)
        final_submit = sub_template[["영업일자"]].merge(final_pred, on="영업일자", how="left")
        final_submit = final_submit.reindex(columns=template_cols)
        final_submit[item_cols] = final_submit[item_cols].fillna(0.0)

        for col in item_cols:
            col_series = final_submit[col]
            if col_series.notna().any():
                Q95 = col_series.quantile(0.95)
                final_submit[col] = np.where(col_series > Q95 * 2.0, Q95 * 1.5, col_series)

        final_submit[item_cols] = np.rint(np.clip(final_submit[item_cols].values, 0, None)).astype(int)

        out_dir = os.path.dirname(cfg.out_submission_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        final_submit.to_csv(cfg.out_submission_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 앙상블 완료! → {cfg.out_submission_csv}")

    print("🏆 N-HiTS + PatchTST + TimesNet + GRU + DLinear 학습/추론 끝")
