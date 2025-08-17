# -*- coding: utf-8 -*-
"""
run_day_long.py  (STORE_DOW 제거, MENU_DOW만 사용)
- Day별 단독 실행
- 누수 방지(day-aware lag/rolling)
- CatBoost / (옵션) XGB / (옵션) LGB / (옵션) Linear baseline
- 튠 OOF(purged) 기반 스태킹 + 안정화(grid/NNLS 대체 + median + cap)
- 평가 누수 제거: base모델(tr+tune)로 hold 평가 → 이후 final(tr+tune+hold) 재학습
- Zero-inflated 대응(제로 가드)
- N-BEATS/N-HiTS 외부 예측: 튠/홀드에서 cutoff 검증 필수, 추론은 자유
"""

import os, re, gc, glob, json, argparse, warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# =============================
# 옵션/하이퍼
# =============================
FAST_MODE         = False
USE_GPU_CAT       = True
USE_XGB           = True
USE_LGB           = True
USE_LINEAR        = True
USE_DEEP          = True

VAL_TUNE_DAYS     = 42
VAL_HOLD_DAYS     = 21
RANDOM_SEED       = 42

EARLY_STOP_CAT    = 150
N_TRIALS_CAT      = 15 if FAST_MODE else 50
N_TRIALS_XGB      = 6  if FAST_MODE else 30
N_TRIALS_LGB      = 6  if FAST_MODE else 30

MAX_ITER_CAT      = 3500
VERBOSE_CAT       = 100

TRAIN_PATH        = "LG_train.csv"
TEST_GLOB         = "test/TEST_*.csv"
SUBMISSION_TPL    = "sample_submission.csv"
OUT_DIR           = "out_day"

DEEP_PREDS_DIR    = "deep_preds"  # NHITS_day{d}.csv / NBEATS_day{d}.csv

# 외부 DOW 피처(메뉴 요일 선호) 경로
MENU_DOW_FEAT_PATH  = "menu_day_features.csv"   # 요일비율_0..6, 최다요일_0..6 포함

# =============================
# 라이브러리
# =============================
try:
    from catboost import CatBoostRegressor
except Exception as e:
    raise RuntimeError(f"CatBoost import 실패: {e}")

XGB_AVAILABLE = False
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    pass

LGB_AVAILABLE = False
try:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    LGB_AVAILABLE = True
except Exception:
    pass

import optuna
from optuna.pruners import MedianPruner

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# =============================
# 공휴일
# =============================
custom_holidays_list = [
    '2023-01-01','2023-01-21','2023-01-22','2023-01-23','2023-01-24','2023-03-01','2023-05-01',
    '2023-05-05','2023-05-27','2023-06-06','2023-08-15','2023-09-28','2023-09-29','2023-09-30',
    '2023-10-02','2023-10-03','2023-10-09','2023-12-25',
    '2024-01-01','2024-02-09','2024-02-10','2024-02-11','2024-02-12','2024-03-01','2024-04-10',
    '2024-05-01','2024-05-05','2024-05-06','2024-05-15','2024-06-06','2024-08-15','2024-09-16',
    '2024-09-17','2024-09-18','2024-10-01','2024-10-03','2024-10-09','2024-12-25',
    '2025-01-01','2025-01-28','2025-01-29','2025-01-30','2025-03-01','2025-03-03','2025-05-01',
    '2025-05-05','2025-05-06','2025-06-06','2025-08-15'
]
custom_holidays = set(pd.to_datetime(custom_holidays_list))

# =============================
# 매장 가중치
# =============================
raw_store_weight = {
    '미라시아': 7.71,'담하': 6.51,'연회장': 3.48,'라그로타': 3.44,'느티나무 셀프BBQ': 2.78,
    '화담숲주막': 1.43,'카페테리아': 1.31,'화담숲카페': 1.14,'포레스트릿': 1.00,
}
mean_w = np.mean(list(raw_store_weight.values()))
store_weight = {k: v/mean_w for k,v in raw_store_weight.items()}
def map_store_weight(x:str)->float: return store_weight.get(x,1.0)

# =============================
# 피처 정의(Lean) — STORE_DOW 제거, MENU_DOW만 포함
# =============================
BASE_FEATURES = [
    '영업장명','메뉴명','휴일여부','휴일전날여부',
    '월_sin','월_cos','연중일자_sin','연중일자_cos','trend_norm',
    'woy_sin','woy_cos','same_dow_mean_4','roc_7_14',
    # MENU_DOW 핵심
    'menu_dow_ratio_today','menu_peak_today'
]
LAGS    = [7,14,21,28]
WINDOWS = [7,14]
USE_ROLL_SUM = False
USE_ROLL_MAX = False

# =============================
# 외부 피처 주입 (MENU_DOW만)
# =============================
def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return None

def augment_with_menu_dow_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mdf = safe_read_csv(MENU_DOW_FEAT_PATH)
    if mdf is None:
        out['menu_dow_ratio_today'] = 0.0
        out['menu_peak_today'] = 0.0
        return out

    mcols_ratio = [c for c in mdf.columns if c.startswith("요일비율_")]
    mcols_peak  = [c for c in mdf.columns if c.startswith("최다요일_")]
    need_cols = ['영업장명','메뉴명'] + mcols_ratio + mcols_peak
    mdf = mdf[[c for c in need_cols if c in mdf.columns]].copy()

    # ★ 비율/최다요일 숫자로 강제
    if mcols_ratio:
        mdf[mcols_ratio] = mdf[mcols_ratio].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
    if mcols_peak:
        mdf[mcols_peak]  = mdf[mcols_peak].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    out = out.merge(mdf, on=['영업장명','메뉴명'], how='left')

    full_ratio = [f'요일비율_{i}' for i in range(7)]
    full_peak  = [f'최다요일_{i}' for i in range(7)]
    for c in full_ratio:
        if c not in out.columns: out[c] = 0.0
    for c in full_peak:
        if c not in out.columns: out[c] = 0

    ratio_vals = out[full_ratio].to_numpy()
    peak_vals  = out[full_peak].to_numpy()
    dow_idx = out['요일'].astype(int).clip(0, 6).to_numpy()

    out['menu_dow_ratio_today'] = ratio_vals[np.arange(len(out)), dow_idx].astype(float)
    out['menu_peak_today']      = (peak_vals[np.arange(len(out)), dow_idx] == 1).astype(float)  # 여기에 변경
    return out

# =============================
# 기본 피처 생성
# =============================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[['영업장명','메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df['요일'] = df['영업일자'].dt.dayofweek
    df['월'] = df['영업일자'].dt.month
    df['연중일자'] = df['영업일자'].dt.dayofyear
    df['주말여부'] = (df['요일']>=5).astype(int)
    df['공휴일여부'] = df['영업일자'].isin(custom_holidays).astype(int)
    df['휴일여부'] = ((df['주말여부']==1)|(df['공휴일여부']==1)).astype(int)
    df['하루뒤'] = df['영업일자'] + pd.Timedelta(days=1)
    df['휴일전날여부'] = ((df['하루뒤'].isin(custom_holidays)) | (df['하루뒤'].dt.dayofweek>=5)).astype(int)
    df.drop(columns=['하루뒤'], inplace=True)

    df['월_sin'] = np.sin(2*np.pi*df['월']/12)
    df['월_cos'] = np.cos(2*np.pi*df['월']/12)
    df['연중일자_sin'] = np.sin(2*np.pi*df['연중일자']/365)
    df['연중일자_cos'] = np.cos(2*np.pi*df['연중일자']/365)

    yday_min = df['연중일자'].min()
    yday_max = df['연중일자'].max()
    df['trend_norm'] = (df['연중일자'] - yday_min) / max(1, (yday_max - yday_min))

    df['weekofyear'] = df['영업일자'].dt.isocalendar().week.astype(int)
    df['woy_sin'] = np.sin(2*np.pi*df['weekofyear']/52)
    df['woy_cos'] = np.cos(2*np.pi*df['weekofyear']/52)

    df['store_weight'] = df['영업장명'].map(map_store_weight).fillna(1.0).astype(float)

    # 메뉴 요일 선호 주입 (정기휴무는 사용 안 함)
    df = augment_with_menu_dow_feats(df)

    df = df.sort_values(['영업장명_메뉴명','영업일자']).reset_index(drop=True)
    return df

def make_lag_roll_dayaware(df: pd.DataFrame,
                           lags=LAGS, wins=WINDOWS) -> Tuple[pd.DataFrame, List[str]]:
    df = df.sort_values(['영업장명_메뉴명','영업일자']).reset_index(drop=True)
    g  = df.groupby('영업장명_메뉴명', sort=False)
    out = df[['영업장명','메뉴명','영업장명_메뉴명','영업일자',
              '휴일여부','휴일전날여부','월_sin','월_cos',
              '연중일자_sin','연중일자_cos','trend_norm',
              'woy_sin','woy_cos',
              'store_weight','요일','매출수량',
              # MENU_DOW 추가 컬럼
              'menu_dow_ratio_today','menu_peak_today']].copy()

    # lag
    for L in lags:
        out[f'lag_{L}'] = g['매출수량'].shift(L)

    # rolling (어제까지)
    base = g['매출수량'].shift(1)
    roll_means, roll_stds, roll_sums, roll_max = [], [], [], []
    for W in wins:
        m = base.groupby(out['영업장명_메뉴명']).rolling(W).mean().reset_index(level=0, drop=True)
        s = base.groupby(out['영업장명_메뉴명']).rolling(W).std().reset_index(level=0, drop=True)
        out[f'rolling_mean_{W}'] = m
        out[f'rolling_std_{W}']  = s
        roll_means.append(f'rolling_mean_{W}')
        roll_stds.append(f'rolling_std_{W}')
        if USE_ROLL_SUM:
            sm = base.groupby(out['영업장명_메뉴명']).rolling(W).sum().reset_index(level=0, drop=True)
            out[f'rolling_sum_{W}']  = sm
            roll_sums.append(f'rolling_sum_{W}')
        if USE_ROLL_MAX:
            mx = base.groupby(out['영업장명_메뉴명']).rolling(W).max().reset_index(level=0, drop=True)
            out[f'rolling_max_{W}']  = mx
            roll_max.append(f'rolling_max_{W}')

    out[roll_means] = out[roll_means].groupby(out['영업장명_메뉴명']).ffill()
    for cols in [roll_means, roll_stds, roll_sums, roll_max]:
        if cols:
            out[cols] = out[cols].fillna(0.0)

    # 같은 요일 평균(최근 4주, 누수방지: shift(7) 기반 rolling)
    tmp = df.copy()
    tmp['same_dow'] = df.groupby(['영업장명_메뉴명','요일'])['매출수량'].shift(7)
    tmp['same_dow_mean_4'] = tmp.groupby(['영업장명_메뉴명','요일'])['same_dow']\
                                .rolling(4).mean().reset_index(level=[0,1], drop=True)
    out['same_dow_mean_4'] = tmp['same_dow_mean_4'].fillna(0.0)

    # 변화율 피처
    out['roc_7_14'] = (out['rolling_mean_7'] + 1e-6) / (out['rolling_mean_14'] + 1e-6)

    feature_cols = BASE_FEATURES + [f'lag_{L}' for L in lags] + roll_means + roll_stds
    if USE_ROLL_SUM: feature_cols += roll_sums
    if USE_ROLL_MAX: feature_cols += roll_max
    return out, feature_cols

# =============================
# 분할
# =============================
def compute_cut_dates(train_df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    last = train_df['영업일자'].max()
    hold_start = last - pd.Timedelta(days=VAL_HOLD_DAYS - 1)
    tune_start = hold_start - pd.Timedelta(days=VAL_TUNE_DAYS)
    return tune_start, hold_start

def time_split_three(df: pd.DataFrame, tune_start, hold_start):
    tr      = df['영업일자'] < tune_start
    va_tune = (df['영업일자'] >= tune_start) & (df['영업일자'] < hold_start)
    va_hold = (df['영업일자'] >= hold_start)
    return tr, va_tune, va_hold

def build_matrices_for_day(train_df: pd.DataFrame, day: int,
                           tune_start, hold_start):
    feats_df, feat_cols = make_lag_roll_dayaware(train_df)
    g = train_df.sort_values(['영업장명_메뉴명','영업일자']).groupby('영업장명_메뉴명', sort=False)['매출수량']
    target = g.shift(-day).rename('target')
    feats_df = feats_df.join(target)

    use_cols = feat_cols + ['target','store_weight','영업장명','메뉴명','영업장명_메뉴명','영업일자']
    feats_df = feats_df[use_cols].dropna(subset=feat_cols + ['target']).reset_index(drop=True)

    tr_idx, va_tune_idx, va_hold_idx = time_split_three(feats_df, tune_start, hold_start)

    X_tr      = feats_df.loc[tr_idx,      feat_cols].copy()
    y_tr      = feats_df.loc[tr_idx,      'target'].copy()
    sw_tr     = feats_df.loc[tr_idx,      'store_weight'].astype(float).values

    X_tune    = feats_df.loc[va_tune_idx, feat_cols].copy()
    y_tune    = feats_df.loc[va_tune_idx, 'target'].copy()
    sw_tune   = feats_df.loc[va_tune_idx, 'store_weight'].astype(float).values

    X_hold    = feats_df.loc[va_hold_idx, feat_cols].copy()
    y_hold    = feats_df.loc[va_hold_idx, 'target'].copy()
    sw_hold   = feats_df.loc[va_hold_idx, 'store_weight'].astype(float).values

    ids_tr    = feats_df.loc[tr_idx,      '영업장명_메뉴명'].reset_index(drop=True)
    ids_tune  = feats_df.loc[va_tune_idx, '영업장명_메뉴명'].reset_index(drop=True)
    ids_hold  = feats_df.loc[va_hold_idx, '영업장명_메뉴명'].reset_index(drop=True)

    return (X_tr, y_tr, sw_tr, ids_tr,
            X_tune, y_tune, sw_tune, ids_tune,
            X_hold, y_hold, sw_hold, ids_hold,
            feat_cols)

# =============================
# 유틸
# =============================
def ensure_unique_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df
def _coerce_numeric_except_cats(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.copy()
    for c in Xn.columns:
        if c not in ('영업장명', '메뉴명'):  # 범주형은 그대로, 나머지는 숫자형 강제
            Xn[c] = pd.to_numeric(Xn[c], errors='coerce')
    return Xn.fillna(0)
    
def nonneg_simplex(w_raw):
    w = np.clip(np.asarray(w_raw, dtype=float), 0.0, None)
    s = w.sum()
    if s == 0: w[:] = 0.0; w[0] = 1.0; return w
    return w / s

def better_objective(y_true, y_pred, sample_weight=None):
    mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    zero_mask = (y_true == 0)
    if zero_mask.any():
        penalty = (y_pred[zero_mask] > 1.0).mean() * 0.5
        mae += penalty
    return mae

def ohe_fit_columns(X_fit: pd.DataFrame):
    cats = {
        '영업장명': X_fit['영업장명'].astype('category').cat.categories.tolist() if '영업장명' in X_fit.columns else [],
        '메뉴명':   X_fit['메뉴명'].astype('category').cat.categories.tolist() if '메뉴명'   in X_fit.columns else [],
    }
    Xc = X_fit.copy()
    if '영업장명' in Xc.columns:
        Xc['영업장명'] = pd.Categorical(Xc['영업장명'], categories=cats['영업장명'])
    if '메뉴명' in Xc.columns:
        Xc['메뉴명']   = pd.Categorical(Xc['메뉴명'],   categories=cats['메뉴명'])
    X_num = pd.get_dummies(Xc, columns=[c for c in ['영업장명','메뉴명'] if c in Xc.columns], drop_first=False)
    cols = X_num.columns.tolist()
    return cols, cats

def ohe_transform_safe(X: pd.DataFrame, cols: List[str], cats: dict) -> pd.DataFrame:
    Xc = X.copy()
    if '영업장명' in Xc.columns:
        Xc['영업장명'] = pd.Categorical(Xc['영업장명'], categories=cats.get('영업장명', []))
    if '메뉴명' in Xc.columns:
        Xc['메뉴명']   = pd.Categorical(Xc['메뉴명'],   categories=cats.get('메뉴명', []))
    X_num = pd.get_dummies(Xc, columns=[c for c in ['영업장명','메뉴명'] if c in Xc.columns], drop_first=False)
    X_num = X_num.reindex(columns=cols, fill_value=0)
    return X_num

def make_safe_feature_mapping(cols):
    mapping, used = {}, set()
    for c in cols:
        s = re.sub(r'[^0-9a-zA-Z_]', '_', str(c))
        if re.match(r'^[0-9]', s): s = 'f_' + s
        base, k = s, 1
        while s in used:
            s = f"{base}_{k}"; k += 1
        mapping[c] = s; used.add(s)
    return mapping

def apply_safe_feature_names(df, mapping=None):
    if mapping is None: mapping = make_safe_feature_mapping(df.columns)
    df2 = df.copy(); df2.columns = [mapping[c] for c in df.columns]
    return df2, mapping

def ensure_list_dict(d):
    if d is None: return None
    out = {}
    for k, v in d.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out

# =============================
# 외부 딥러닝 예측 (cutoff 검증 포함)
# =============================
def _load_deep_file(path: str, ids: pd.Series,
                    require_cutoff_date: Optional[pd.Timestamp] = None) -> Optional[np.ndarray]:
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        # cutoff 검증: 파일에 train_last_date 컬럼이 있을 때만 체크
        if require_cutoff_date is not None:
            if 'train_last_date' in df.columns:
                try:
                    cut = pd.to_datetime(df['train_last_date'].iloc[0])
                    if cut > require_cutoff_date:
                        return None
                except Exception:
                    return None
            else:
                return None

        if '영업장명_메뉴명' in df.columns and 'pred' in df.columns:
            ids_df = ids.to_frame(name='영업장명_메뉴명').reset_index()
            merged = ids_df.merge(df[['영업장명_메뉴명','pred']], on='영업장명_메뉴명', how='left').sort_values('index')
            v = merged['pred'].astype(float).fillna(0.0).values
            return v if len(v)==len(ids) else None
    except Exception:
        return None
    return None

def maybe_load_deep_preds(day:int, ids:pd.Series, deep_dir=DEEP_PREDS_DIR,
                          prefer_rule=True,
                          require_cutoff_date: Optional[pd.Timestamp]=None):
    if not USE_DEEP: return None
    nh_path = os.path.join(deep_dir, f'NHITS_day{day}.csv')
    nb_path = os.path.join(deep_dir, f'NBEATS_day{day}.csv')
    nh_vec = _load_deep_file(nh_path, ids, require_cutoff_date=require_cutoff_date)
    nb_vec = _load_deep_file(nb_path, ids, require_cutoff_date=require_cutoff_date)
    if nh_vec is None and nb_vec is None: return None
    if not prefer_rule:
        if nh_vec is None: return nb_vec
        if nb_vec is None: return nh_vec
        return 0.5*(nh_vec+nb_vec)
    if day in (1,2,3):
        return nb_vec if nb_vec is not None else nh_vec
    elif day in (5,6,7):
        return nh_vec if nh_vec is not None else nb_vec
    else:  # day == 4
        if nh_vec is None: return nb_vec
        if nb_vec is None: return nh_vec
        return 0.5*(nh_vec+nb_vec)

# =============================
# CatBoost 튜닝(범위 확장)
# =============================
def tune_catboost(X_tr, y_tr, X_tune, y_tune, cat_cols, sw_tr,
                  use_gpu=True, n_trials=15):
    cat_extra = {"task_type":"GPU","devices":"0"} if use_gpu else {"thread_count": -1}

    def cb_objective(trial):
        params = {
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "depth": trial.suggest_int("depth", 5, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 8.0),
            "random_strength": trial.suggest_float("random_strength", 0.3, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 80),
            "iterations": MAX_ITER_CAT,
            "random_seed": RANDOM_SEED,
            "early_stopping_rounds": EARLY_STOP_CAT,
            "verbose": False,
            **cat_extra,  # {"task_type":"GPU","devices":"0"} 등
        }
    
        # ★ bootstrap_type에 따라 mutually-exclusive 하이퍼만 추가
        bt = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])
        params["bootstrap_type"] = bt
        if bt == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.5, 5.0)
            params.pop("subsample", None)
        else:  # Bernoulli
            params["subsample"] = trial.suggest_float("subsample", 0.6, 0.95)
            params.pop("bagging_temperature", None)
    
        model = CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_tune, y_tune),
            cat_features=cat_cols,
            sample_weight=sw_tr,
            use_best_model=True,
            verbose=False
        )
        p = np.clip(model.predict(X_tune), 0, None)
        return mean_absolute_error(y_tune, p)

    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(),
        study_name='catboost',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    study.optimize(cb_objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_trial.params.copy()
    if bp.get('bootstrap_type') == 'Bayesian':
        bp.pop('subsample', None)
    else:
        bp.pop('bagging_temperature', None)
    if use_gpu:
        bp.pop('rsm', None)

    cat_params = {
        "loss_function": "MAE", "eval_metric":"MAE",
        "iterations": MAX_ITER_CAT,
        "early_stopping_rounds": EARLY_STOP_CAT,
        "random_seed": RANDOM_SEED,
        "verbose": VERBOSE_CAT,
        **({"task_type":"GPU","devices":"0"} if use_gpu else {"thread_count": -1}),
        **bp
    }
    return cat_params

# =============================
# 시계열 purged split
# =============================
def purged_splits(n_samples:int, n_splits:int=4, gap:int=7):
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    indices = np.arange(n_samples)
    current = 0
    bounds = []
    for fsz in range(n_splits):
        size = fold_sizes[fsz]
        bounds.append((current, current+size))
        current += size
    for (s,e) in bounds:
        va_idx = indices[s:e]
        tr_mask = np.ones(n_samples, dtype=bool)
        left = max(0, s - gap); right = min(n_samples, e + gap)
        tr_mask[left:right] = False
        tr_idx = indices[tr_mask]
        yield tr_idx, va_idx

# =============================
# 스태킹: 가중치 탐색(안정화)
# =============================
def _project_simplex_nonneg(w):
    w = np.clip(np.asarray(w, float), 0, None)
    s = w.sum()
    return w if s == 0 else (w / s)

def _cap_weights(w, cap=0.7):
    w = np.asarray(w, float)
    if w.max() <= cap:
        return _project_simplex_nonneg(w)
    idx = np.argmax(w)
    w[idx] = cap
    if w.sum() == 0:
        w[idx] = 1.0
        return w
    return w / w.sum()

def grid_search_simplex(P, y, step=0.05, cap=0.7):
    m = P.shape[1]
    if m == 1:
        return np.array([1.0])
    best_mae, best_w = 1e18, None
    if m == 2:
        for a in np.arange(0, 1+1e-9, step):
            w = np.array([a, 1-a])
            if w.max() > cap: continue
            mae = mean_absolute_error(y, np.maximum(0.0, (P*w).sum(axis=1)))
            if mae < best_mae: best_mae, best_w = mae, w
    elif m == 3:
        for a in np.arange(0, 1+1e-9, step):
            for b in np.arange(0, 1-a+1e-9, step):
                w = np.array([a, b, 1-a-b])
                if w.max() > cap: continue
                mae = mean_absolute_error(y, np.maximum(0.0, (P*w).sum(axis=1)))
                if mae < best_mae: best_mae, best_w = mae, w
    else: # m == 4
        for a in np.arange(0, 1+1e-9, step):
            for b in np.arange(0, 1-a+1e-9, step):
                for c in np.arange(0, 1-a-b+1e-9, step):
                    w = np.array([a, b, c, 1-a-b-c])
                    if w.max() > cap: continue
                    mae = mean_absolute_error(y, np.maximum(0.0, (P*w).sum(axis=1)))
                    if mae < best_mae: best_mae, best_w = mae, w
    if best_w is None:
        best_w = _project_simplex_nonneg(np.ones(m))
    return _cap_weights(best_w, cap=cap)

def ridge_then_project(P, y, cap=0.7):
    reg = Ridge(alpha=1.0, fit_intercept=False, random_state=RANDOM_SEED)
    reg.fit(P, y)
    w = reg.coef_
    w = _project_simplex_nonneg(w)
    w = _cap_weights(w, cap=cap)
    return w

def robust_stack_weights(P_tune, y_t, n_splits=4, gap=7, cap=0.7):
    fold_ws = []
    for tr_idx, va_idx in purged_splits(len(P_tune), n_splits=n_splits, gap=gap):
        P_tr, P_va = P_tune[tr_idx], P_tune[va_idx]
        y_tr, y_va = y_t[tr_idx], y_t[va_idx]
        m = P_tr.shape[1]
        if m <= 4:
            w = grid_search_simplex(P_va, y_va, step=0.05, cap=cap)
        else:
            w = ridge_then_project(P_tr, y_tr, cap=cap)
        fold_ws.append(w)
    W = np.median(np.vstack(fold_ws), axis=0)
    W = _project_simplex_nonneg(W)
    W = _cap_weights(W, cap=cap)
    return W

# =============================
# 멀티 기간 검증(축약 앙상블)
# =============================
def multi_period_eval(train_df: pd.DataFrame, day:int, periods:int=3):
    maes = []
    base_last = train_df['영업일자'].max()
    for k in range(periods):
        hold_start = base_last - pd.Timedelta(days=VAL_HOLD_DAYS - 1 + k*VAL_HOLD_DAYS)
        tune_start = hold_start - pd.Timedelta(days=VAL_TUNE_DAYS)

        (X_tr, y_tr, sw_tr, ids_tr,
         X_tune, y_tune, sw_tune, ids_tune,
         X_hold, y_hold, sw_hold, ids_hold,
         feat_cols) = build_matrices_for_day(train_df, day, tune_start, hold_start)

        X_tr = ensure_unique_cols(X_tr).fillna(0)
        X_tune = ensure_unique_cols(X_tune).reindex(columns=X_tr.columns, fill_value=0).fillna(0)
        X_hold = ensure_unique_cols(X_hold).reindex(columns=X_tr.columns, fill_value=0).fillna(0)

        # 축약 앙상블: Cat(작은 파라미터) + Linear + Deep(컷오프 필수)
        cat = CatBoostRegressor(loss_function="MAE", depth=6, learning_rate=0.12,
                                iterations=1200, early_stopping_rounds=80,
                                task_type="GPU" if USE_GPU_CAT else "CPU",
                                devices="0", verbose=False)
        cat.fit(pd.concat([X_tr, X_tune]),
                pd.concat([y_tr, y_tune]),
                cat_features=[c for c in ['영업장명','메뉴명'] if c in X_tr.columns])
        preds_tune = [np.clip(cat.predict(X_tune), 0, None)]
        names = ["cat"]

        lin_feats = [c for c in X_tune.columns if c not in ['영업장명','메뉴명']]
        lin = Ridge(alpha=1.0, fit_intercept=True, random_state=RANDOM_SEED)
        lin.fit(pd.concat([X_tr[lin_feats], X_tune[lin_feats]]),
                pd.concat([y_tr, y_tune]))
        preds_tune.append(np.clip(lin.predict(X_tune[lin_feats]), 0, None))
        names.append("lin")

        deep_tune = maybe_load_deep_preds(day, ids_tune, require_cutoff_date=(tune_start - pd.Timedelta(days=1)))
        if deep_tune is not None:
            preds_tune.append(np.clip(deep_tune, 0, None)); names.append("deep")

        P_t = np.vstack(preds_tune).T
        w = robust_stack_weights(P_t, y_tune.values, n_splits=4, gap=7, cap=0.7)

        preds_hold = [np.clip(cat.predict(X_hold), 0, None),
                      np.clip(lin.predict(X_hold[lin_feats]), 0, None)]
        if 'deep' in names:
            deep_hold = maybe_load_deep_preds(day, ids_hold, require_cutoff_date=(tune_start - pd.Timedelta(days=1)))
            if deep_hold is not None:
                preds_hold.append(np.clip(deep_hold, 0, None))
        P_h = np.vstack(preds_hold).T
        y_hat = np.maximum(0.0, (P_h * w[:P_h.shape[1]]).sum(axis=1))

        last4_zero = (X_hold[[f'lag_{L}' for L in [7,14,21,28]]].fillna(0).sum(axis=1) == 0)
        y_hat[last4_zero.values] = 0.0

        maes.append(mean_absolute_error(y_hold, y_hat))
        del cat, lin; gc.collect()

    return float(np.mean(maes)), float(np.std(maes))

# =============================
# 메인
# =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--train_path", default=TRAIN_PATH)
    parser.add_argument("--test_glob",  default=TEST_GLOB)
    parser.add_argument("--submission_tpl", default=SUBMISSION_TPL)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--use_gpu_cat", action="store_true")
    parser.add_argument("--use_xgb", action="store_true")
    parser.add_argument("--use_lgb", action="store_true")
    parser.add_argument("--no_linear", action="store_true")
    parser.add_argument("--no_deep", action="store_true")
    parser.add_argument("--pos_alpha", type=float, default=None)
    parser.add_argument("--mp_eval", action="store_true")
    args = parser.parse_args()

    DAY = args.day
    os.makedirs(args.out_dir, exist_ok=True)

    global FAST_MODE, USE_GPU_CAT, USE_XGB, USE_LGB, USE_LINEAR, USE_DEEP
    global N_TRIALS_CAT, N_TRIALS_XGB, N_TRIALS_LGB
    if args.fast:
        FAST_MODE = True
        N_TRIALS_CAT = 15
        N_TRIALS_XGB = 6
        N_TRIALS_LGB = 6
    USE_GPU_CAT = args.use_gpu_cat
    USE_XGB = bool(args.use_xgb and XGB_AVAILABLE)
    USE_LGB = bool(args.use_lgb and LGB_AVAILABLE)
    USE_LINEAR = not args.no_linear
    USE_DEEP = not args.no_deep

    print(f"=== RUN DAY {DAY} ===")
    print(f"FAST_MODE={FAST_MODE}, GPU_CAT={USE_GPU_CAT}, USE_XGB={USE_XGB}, USE_LGB={USE_LGB}, USE_LINEAR={USE_LINEAR}, USE_DEEP={USE_DEEP}")

    train_df = pd.read_csv(args.train_path)
    neg_ct = (train_df['매출수량'] < 0).sum()
    if neg_ct:
        print(f"[정제] 음수 매출 {neg_ct}건 → 0으로 클립")
    train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
    train_df = create_features(train_df)

    if args.mp_eval:
        m, s = multi_period_eval(train_df, DAY, periods=3)
        print(f"[Multi-period] MAE mean={m:.4f} | std={s:.4f}")

    tune_start, hold_start = compute_cut_dates(train_df)
    print(f"tune_start={tune_start.date()}, hold_start={hold_start.date()}")

    (X_tr, y_tr, sw_tr, ids_tr,
     X_tune, y_tune, sw_tune, ids_tune,
     X_hold, y_hold, sw_hold, ids_hold,
     feat_cols) = build_matrices_for_day(train_df, DAY, tune_start, hold_start)

    X_tr   = ensure_unique_cols(X_tr).fillna(0)
    X_tune = ensure_unique_cols(X_tune).reindex(columns=X_tr.columns, fill_value=0).fillna(0)
    X_hold = ensure_unique_cols(X_hold).reindex(columns=X_tr.columns, fill_value=0).fillna(0)
    X_tr   = _coerce_numeric_except_cats(X_tr)
    X_tune = _coerce_numeric_except_cats(X_tune)
    X_hold = _coerce_numeric_except_cats(X_hold)

    # ===== sample weight α 선택 =====
    def apply_sw(alpha):
        def pos_boost(y): return 1.0 + alpha*np.log1p(np.asarray(y))
        _sw_tr   = sw_tr   * np.where(y_tr.values   > 0, pos_boost(y_tr),   1.0)
        _sw_tune = sw_tune * np.where(y_tune.values > 0, pos_boost(y_tune), 1.0)
        _sw_hold = sw_hold * np.where(y_hold.values > 0, pos_boost(y_hold), 1.0)
        return _sw_tr, _sw_tune, _sw_hold

    if args.pos_alpha is not None:
        alpha_best = max(0.0, float(args.pos_alpha))
        print(f"[SW] 사용자 지정 α={alpha_best}")
    else:
        cand = [0.0, 0.1, 0.2, 0.3, 0.5]
        scores = []
        for a in cand:
            _sw_tr, _sw_tune, _ = apply_sw(a)
            cb = CatBoostRegressor(loss_function="MAE", depth=6, learning_rate=0.1,
                                   iterations=500, early_stopping_rounds=60,
                                   task_type="GPU" if USE_GPU_CAT else "CPU",
                                   devices="0", verbose=False)
            cb.fit(X_tr, y_tr, eval_set=(X_tune, y_tune),
                   sample_weight=_sw_tr,
                   cat_features=[c for c in ['영업장명','메뉴명'] if c in X_tr.columns],
                   verbose=False)
            p = np.clip(cb.predict(X_tune), 0, None)
            scores.append(mean_absolute_error(y_tune, p))
            del cb; gc.collect()
        alpha_best = cand[int(np.argmin(scores))]
        print(f"[SW] α 자동선택 = {alpha_best} (candidates={cand})")

    sw_tr, sw_tune, sw_hold = apply_sw(alpha_best)
    cat_cols = [c for c in ['영업장명','메뉴명'] if c in X_tr.columns]

    # ----------------- CatBoost 튠 & base -----------------
    print("[CatBoost] 튠 시작...")
    cat_params = tune_catboost(X_tr, y_tr, X_tune, y_tune, cat_cols, sw_tr,
                               use_gpu=USE_GPU_CAT, n_trials=N_TRIALS_CAT)

    cat_base = CatBoostRegressor(**{**cat_params, "verbose": False})
    X_tr_tune  = pd.concat([X_tr,  X_tune], axis=0)
    y_tr_tune  = pd.concat([y_tr,  y_tune], axis=0)
    sw_tr_tune = np.concatenate([sw_tr, sw_tune])
    cat_base.fit(X_tr_tune, y_tr_tune, cat_features=cat_cols, sample_weight=sw_tr_tune, verbose=False)

    # ----------------- XGB/LGB base(옵션) -----------------
    xgb_base = None; xgb_ohe_cols=None; xgb_cats=None; xgb_name_map=None; st_x=None
    if USE_XGB:
        print("[XGB] OHE 적합...")
        xgb_ohe_cols, xgb_cats = ohe_fit_columns(pd.concat([X_tr, X_tune], axis=0))
    
        # --- 튠용 입력 ---
        Xt_raw = ohe_transform_safe(X_tr,   xgb_ohe_cols, xgb_cats).fillna(0)
        Xv_raw = ohe_transform_safe(X_tune, xgb_ohe_cols, xgb_cats).fillna(0)
    
        # ★ 안전망: Inf -> NaN -> 0
        Xt_raw = Xt_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        Xv_raw = Xv_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
    
        Xt, xgb_name_map = apply_safe_feature_names(Xt_raw, mapping=None)
        Xv, _            = apply_safe_feature_names(Xv_raw, mapping=xgb_name_map)


        def xgb_objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 400, 1200),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.4, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 8.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 12.0),
                "objective": "reg:absoluteerror",
                "random_state": RANDOM_SEED,
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
                "eval_metric": "mae",
            }
            mdl = XGBRegressor(**params)
            mdl.fit(Xt, y_tr, eval_set=[(Xv, y_tune)], sample_weight=sw_tr, verbose=False)
            p = np.clip(mdl.predict(Xv), 0, None)
            return mean_absolute_error(y_tune, p)

        st_x = optuna.create_study(direction="minimize", pruner=MedianPruner(),
                                   sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        st_x.optimize(xgb_objective, n_trials=N_TRIALS_XGB, show_progress_bar=False)

        xgb_base = XGBRegressor(**{**st_x.best_trial.params,
                                   "objective":"reg:absoluteerror",
                                   "random_state":RANDOM_SEED,
                                   "tree_method":"gpu_hist",
                                   "predictor":"gpu_predictor",
                                   "eval_metric":"mae"})
        X_tr_tune_raw = ohe_transform_safe(X_tr_tune, xgb_ohe_cols, xgb_cats).fillna(0)
        X_tr_tune_raw = X_tr_tune_raw.replace([np.inf, -np.inf], np.nan).fillna(0)  # ★
        X_tr_tune_ohe, _ = apply_safe_feature_names(X_tr_tune_raw, mapping=xgb_name_map)
        xgb_base.fit(X_tr_tune_ohe, y_tr_tune, sample_weight=sw_tr_tune)

    lgb_base = None; lgb_ohe_cols=None; lgb_cats=None; lgb_name_map=None; st_l=None
    if USE_LGB:
        print("[LGB] OHE 적합...")
        lgb_ohe_cols, lgb_cats = ohe_fit_columns(pd.concat([X_tr, X_tune], axis=0))
        Xt_raw = ohe_transform_safe(X_tr,   lgb_ohe_cols, lgb_cats).fillna(0)
        Xv_raw = ohe_transform_safe(X_tune, lgb_ohe_cols, lgb_cats).fillna(0)
        Xt_raw = Xt_raw.replace([np.inf, -np.inf], np.nan).fillna(0)  # ★
        Xv_raw = Xv_raw.replace([np.inf, -np.inf], np.nan).fillna(0)  # ★
        Xt, lgb_name_map = apply_safe_feature_names(Xt_raw, mapping=None)
        Xv, _            = apply_safe_feature_names(Xv_raw, mapping=lgb_name_map)

        def lgb_objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 400, 1400),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 8.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 12.0),
                "objective":"mae",
                "random_state":RANDOM_SEED
            }
            mdl = LGBMRegressor(**params)
            mdl.fit(
                Xt, y_tr,
                eval_set=[(Xv, y_tune)],
                eval_metric="l1",
                sample_weight=sw_tr,
                callbacks=[early_stopping(stopping_rounds=80), log_evaluation(0)],
            )
            best_iter = getattr(mdl, "best_iteration_", None) or getattr(mdl, "best_iteration", None)
            p = np.clip(mdl.predict(Xv, num_iteration=best_iter), 0, None)
            return mean_absolute_error(y_tune, p)

        st_l = optuna.create_study(direction="minimize", pruner=MedianPruner(),
                                   sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        st_l.optimize(lgb_objective, n_trials=N_TRIALS_LGB, show_progress_bar=False)

        lgb_base = LGBMRegressor(**{**st_l.best_trial.params, "objective":"mae", "random_state":RANDOM_SEED})
        X_tr_tune_raw = ohe_transform_safe(X_tr_tune, lgb_ohe_cols, lgb_cats).fillna(0)
        X_tr_tune_raw = X_tr_tune_raw.replace([np.inf, -np.inf], np.nan).fillna(0)  # ★
        X_tr_tune_ohe, _ = apply_safe_feature_names(X_tr_tune_raw, mapping=lgb_name_map)
        lgb_base.fit(X_tr_tune_ohe, y_tr_tune, sample_weight=sw_tr_tune)

    # ----------------- Linear base -----------------
    lin_base = None
    if USE_LINEAR:
        lin_feats = [c for c in X_tr_tune.columns if c not in ['영업장명','메뉴명']]
    
        # ★ NaN/Inf 방지
        X_tr_tune_lin = X_tr_tune[lin_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    
        lin_base = Ridge(alpha=1.0, fit_intercept=True, random_state=RANDOM_SEED)
        lin_base.fit(X_tr_tune_lin, y_tr_tune, sample_weight=sw_tr_tune)

    # ----------------- 튠 OOF 스태킹(딥 포함 with cutoff) -----------------
    print("[Stacking] 튠 OOF Ridge/GRID(purged) → 가중치(robust)")
    preds_tune = []; names = []

    preds_tune.append(np.clip(cat_base.predict(X_tune), 0, None)); names.append("cat")

    if xgb_base is not None:
        Xt_raw = ohe_transform_safe(X_tune, xgb_ohe_cols, xgb_cats).fillna(0)
        Xt_raw = Xt_raw.replace([np.inf, -np.inf], np.nan).fillna(0)   # ★ 추가
        Xt_ohe, _ = apply_safe_feature_names(Xt_raw, mapping=xgb_name_map)
        preds_tune.append(np.clip(xgb_base.predict(Xt_ohe), 0, None)); names.append("xgb")

    if lgb_base is not None:
        Xt_raw = ohe_transform_safe(X_tune, lgb_ohe_cols, lgb_cats).fillna(0)
        Xt_raw = Xt_raw.replace([np.inf, -np.inf], np.nan).fillna(0)   # ★ 추가
        Xt_ohe, _ = apply_safe_feature_names(Xt_raw, mapping=lgb_name_map)
        preds_tune.append(np.clip(lgb_base.predict(Xt_ohe), 0, None)); names.append("lgb")

    if lin_base is not None:
        lin_feats = [c for c in X_tune.columns if c not in ['영업장명','메뉴명']]
        X_tune_lin = X_tune[lin_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        preds_tune.append(np.clip(lin_base.predict(X_tune_lin), 0, None)); names.append("lin")

    # 딥러닝: cutoff 필요 (tune_start - 1일)
    deep_tune = maybe_load_deep_preds(DAY, ids_tune, require_cutoff_date=(tune_start - pd.Timedelta(days=1)))
    if deep_tune is not None:
        preds_tune.append(np.clip(deep_tune, 0, None)); names.append("deep")

    P_tune = np.vstack(preds_tune).T
    y_t    = y_tune.values

    w_final = robust_stack_weights(P_tune, y_t, n_splits=4, gap=7, cap=0.7)
    blend_weights = dict(zip(names, w_final))
    print(f"[Day {DAY}] stacking weights: {blend_weights}")

    # ----------------- HOLD 평가 -----------------
    preds_hold = [np.clip(cat_base.predict(X_hold), 0, None)]
    if xgb_base is not None:
        Xh_raw = ohe_transform_safe(X_hold, xgb_ohe_cols, xgb_cats).fillna(0)
        Xh_raw = Xh_raw.replace([np.inf, -np.inf], np.nan).fillna(0)  # ★
        Xh_ohe, _ = apply_safe_feature_names(Xh_raw, mapping=xgb_name_map)
        preds_hold.append(np.clip(xgb_base.predict(Xh_ohe), 0, None))
    if lgb_base is not None:
        Xh_raw = ohe_transform_safe(X_hold, lgb_ohe_cols, lgb_cats).fillna(0)
        Xh_raw = Xh_raw.replace([np.inf, -np.inf], np.nan).fillna(0)  # ★
        Xh_ohe, _ = apply_safe_feature_names(Xh_raw, mapping=lgb_name_map)
        preds_hold.append(np.clip(lgb_base.predict(Xh_ohe), 0, None))
    if lin_base is not None:
        lin_feats = [c for c in X_hold.columns if c not in ['영업장명','메뉴명']]
        X_hold_lin = X_hold[lin_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        preds_hold.append(np.clip(lin_base.predict(X_hold_lin), 0, None))

    if 'deep' in names:
        deep_hold = maybe_load_deep_preds(DAY, ids_hold, require_cutoff_date=(tune_start - pd.Timedelta(days=1)))
        if deep_hold is not None:
            preds_hold.append(np.clip(deep_hold, 0, None))

    P_h = np.vstack(preds_hold).T
    w_used = w_final[:P_h.shape[1]]
    w_used = nonneg_simplex(w_used)

    blended_hold = np.maximum(0.0, (P_h * w_used).sum(axis=1))
    last4_zero = (X_hold[[f'lag_{L}' for L in [7,14,21,28]]].fillna(0).sum(axis=1) == 0)
    blended_hold[last4_zero.values] = 0.0

    mae_hold = mean_absolute_error(y_hold, blended_hold)
    mae_hold_za = better_objective(y_hold, blended_hold)
    print(f"[Day {DAY}] HOLD MAE={mae_hold:.4f} | Zero-aware={mae_hold_za:.4f}")

    # ----------------- Final 재학습 -----------------
    del lin_base, xgb_base, lgb_base
    gc.collect()

    print("[Final] 전체 데이터로 재학습 (inference 전용)")
    best_it = None
    try:
        best_it = int(cat_base.get_best_iteration())
    except Exception:
        pass

    final_params = {**cat_params}
    if best_it and best_it > 0:
        final_params["iterations"] = best_it
    final_params.pop("early_stopping_rounds", None)
    final_params["verbose"] = 100

    cat_final = CatBoostRegressor(**final_params)

    X_all  = pd.concat([pd.concat([X_tr, X_tune], axis=0), X_hold], axis=0)
    y_all  = pd.concat([pd.concat([y_tr, y_tune], axis=0), y_hold], axis=0)
    sw_all = np.concatenate([np.concatenate([sw_tr, sw_tune]), sw_hold])

    print(f"[Final] fit start | rows={len(X_all):,}, features={X_all.shape[1]}")
    cat_final.fit(X_all, y_all, cat_features=cat_cols, sample_weight=sw_all)
    print("[Final] fit done")

    xgb_final = None
    if USE_XGB:
        X_all_raw = ohe_transform_safe(X_all, xgb_ohe_cols, xgb_cats).fillna(0)
        X_all_ohe, _  = apply_safe_feature_names(X_all_raw, mapping=xgb_name_map)
        xgb_final = XGBRegressor(**{**st_x.best_trial.params,
                                    "objective":"reg:absoluteerror",
                                    "random_state":RANDOM_SEED,
                                    "tree_method":"gpu_hist",
                                    "predictor":"gpu_predictor",
                                    "eval_metric":"mae"})
        xgb_final.fit(X_all_ohe, y_all, sample_weight=sw_all)

    lgb_final = None
    if USE_LGB:
        X_all_raw = ohe_transform_safe(X_all, lgb_ohe_cols, lgb_cats).fillna(0)
        X_all_ohe, _  = apply_safe_feature_names(X_all_raw, mapping=lgb_name_map)
        lgb_final = LGBMRegressor(**{**st_l.best_trial.params, "objective":"mae", "random_state":RANDOM_SEED})
        lgb_final.fit(X_all_ohe, y_all, sample_weight=sw_all)

    lin_final = None
    if USE_LINEAR:
        lin_feats_all = [c for c in X_all.columns if c not in ['영업장명','메뉴명']]
        lin_final = Ridge(alpha=1.0, fit_intercept=True, random_state=RANDOM_SEED)
        lin_final.fit(X_all[lin_feats_all], y_all, sample_weight=sw_all)

    # ----------------- 메타 저장 -----------------
    meta = {
        "day": DAY,
        "feat_cols": feat_cols,
        "names": list(blend_weights.keys()),
        "weights": w_final.tolist(),
        "xgb": {
            "ohe_cols": None,
            "cats": None,
            "name_map": None,
        },
        "lgb": {
            "ohe_cols": None,
            "cats": None,
            "name_map": None,
        },
        "linear": bool(USE_LINEAR),
        "pos_alpha": float(alpha_best),
        "stack_cap": 0.7
    }
    with open(os.path.join(args.out_dir, f"meta_day{DAY}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ----------------- 추론 -----------------
    submission_tpl = pd.read_csv(args.submission_tpl, nrows=0)
    test_files = sorted(glob.glob(args.test_glob))
    all_preds = []

    for test_file in test_files:
        file_id = os.path.basename(test_file).split('.')[0]
        test_df = pd.read_csv(test_file)
        if '매출수량' in test_df.columns:
            test_df['매출수량'] = test_df['매출수량'].clip(lower=0)
        else:
            test_df['매출수량'] = 0
        test_df = create_features(test_df)

        feats_inf, _ = make_lag_roll_dayaware(test_df)
        latest = (
            feats_inf.sort_values(['영업장명_메뉴명','영업일자'])
                     .groupby('영업장명_메뉴명', sort=False)
                     .tail(1)
                     .reset_index(drop=True)
        )
        ids_series = latest['영업장명_메뉴명'].copy()
        base_inf = latest.reindex(columns=feat_cols, fill_value=0)

        part_preds, infer_names = [], []

        p_cat_inf = np.clip(cat_final.predict(base_inf), 0, None)
        part_preds.append(p_cat_inf); infer_names.append("cat")

        if xgb_final is not None:
            # 위에서 메타 저장을 None으로 뒀지만, XGB를 켜면 여기서 맞춰 사용 가능하게 확장해도 됨
            pass

        if lgb_final is not None:
            pass

        if lin_final is not None:
            lin_feats_inf = [c for c in base_inf.columns if c not in ['영업장명','메뉴명']]
            base_inf_lin = base_inf[lin_feats_inf].replace([np.inf, -np.inf], np.nan).fillna(0)  # ★
            part_preds.append(np.clip(lin_final.predict(base_inf_lin), 0, None))
            infer_names.append("lin")

        # ----- 추론 블록에서 cat/lin/deep 뒤에 추가 -----
        if xgb_final is not None:
            base_inf_raw = ohe_transform_safe(base_inf, xgb_ohe_cols, xgb_cats).fillna(0)
            base_inf_raw = base_inf_raw.replace([np.inf, -np.inf], np.nan).fillna(0)  # ★
            base_inf_ohe, _ = apply_safe_feature_names(base_inf_raw, mapping=xgb_name_map)
            part_preds.append(np.clip(xgb_final.predict(base_inf_ohe), 0, None)); infer_names.append("xgb")
        
        if lgb_final is not None:
            base_inf_raw = ohe_transform_safe(base_inf, lgb_ohe_cols, lgb_cats).fillna(0)
            base_inf_raw = base_inf_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
            base_inf_ohe, _ = apply_safe_feature_names(base_inf_raw, mapping=lgb_name_map)
            part_preds.append(np.clip(lgb_final.predict(base_inf_ohe), 0, None)); infer_names.append("lgb")

        # 추론시 딥은 cutoff 체크 없이 사용(배포물에 맞춰 생성되었다고 가정)
        deep_vec = maybe_load_deep_preds(DAY, ids_series, require_cutoff_date=None)
        if deep_vec is not None:
            part_preds.append(np.clip(deep_vec, 0, None)); infer_names.append("deep")

        wdict = dict(zip(blend_weights.keys(), w_final))
        w = np.array([wdict.get(n, 0.0) for n in infer_names], dtype=float)
        w = nonneg_simplex(w)
        Pinf = np.vstack(part_preds).T
        blended = np.maximum(0.0, (Pinf * w).sum(axis=1))

        last4_zero_inf = (latest[[f'lag_{L}' for L in [7,14,21,28]]].fillna(0).sum(axis=1) == 0)
        blended[last4_zero_inf.values] = 0.0

        temp = pd.DataFrame({
            '영업일자': f'{file_id}+{DAY}일',
            '영업장명_메뉴명': latest['영업장명_메뉴명'],
            '매출수량': np.round(blended).astype(int)
        })
        all_preds.append(temp)

    final_df = pd.concat(all_preds, ignore_index=True)
    pivot_df = final_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()

    cols_order = pd.read_csv(args.submission_tpl, nrows=0).columns
    pivot_df = pivot_df.reindex(columns=cols_order, fill_value=0)

    out_csv = os.path.join(args.out_dir, f"submission_day{DAY}.csv")
    pivot_df.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv}")

    del cat_final, xgb_final, lgb_final, lin_final
    gc.collect()

if __name__ == "__main__":
    main()