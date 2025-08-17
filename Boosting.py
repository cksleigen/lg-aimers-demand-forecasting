# -*- coding: utf-8 -*-
"""
run_day.py (Cat+Linear+Deep only)
- Day별 단독 실행 (1~7)
- 누수 방지: target은 미래 shift(-day), feature는 항상 과거 기준(lag/rolling)
- 모델: CatBoost + Linear(Ridge) + (옵션) Deep 외부예측(N-BEATS/N-HiTS)
- 튠 구간 OOF(purged split) 기반 스태킹 → 비음수/합=1 제약(가능 시 NNLS), 불가 시 강건 대안
- 평가 누수 제거: base(평가용)는 tr+tune 학습 / hold 평가 → 이후 final(제출용) tr+tune+hold로 재학습
- 제로 가드(최근 lag 합=0이면 0), zero-aware MAE 리포트
- 멀티 기간 검증 옵션(--mp_eval): 축약 앙상블로 실제 파이프라인과 유사하게 검증
"""

import os, re, gc, glob, json, argparse, warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# =============================
# 옵션/하이퍼 (CLI로 덮어쓰기 가능)
# =============================
FAST_MODE         = True
USE_GPU_CAT       = True
USE_LINEAR        = True
USE_DEEP          = True

VAL_TUNE_DAYS     = 42
VAL_HOLD_DAYS     = 21
RANDOM_SEED       = 42

# Cat 튠/학습
EARLY_STOP_CAT    = 150
N_TRIALS_CAT      = 15 if FAST_MODE else 40
MAX_ITER_CAT      = 2000
VERBOSE_CAT       = 100

# 경로
TRAIN_PATH        = "/content/drive/MyDrive/LG_train.csv"
TEST_GLOB         = "/content/drive/MyDrive/test/TEST_*.csv"
SUBMISSION_TPL    = "/content/drive/MyDrive/sample_submission.csv"
OUT_DIR           = "/content/drive/MyDrive/out_day"

# 딥러닝 외부 예측(NHiTS/NBEATS)
DEEP_PREDS_DIR    = "/content/drive/MyDrive/deep_preds"  # NHITS_day{d}.csv / NBEATS_day{d}.csv

# =============================
# 라이브러리
# =============================
try:
    from catboost import CatBoostRegressor
except Exception as e:
    raise RuntimeError(f"CatBoost import 실패: {e}")

import optuna
from optuna.pruners import MedianPruner

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# NNLS(선호) 사용 가능하면 사용
try:
    from scipy.optimize import nnls as scipy_nnls
    SCIPY_NNLS_AVAILABLE = True
except Exception:
    SCIPY_NNLS_AVAILABLE = False

# =============================
# 공휴일 (예시)
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
# 매장 가중치(예시)
# =============================
raw_store_weight = {
    '미라시아': 7.71,'담하': 6.51,'연회장': 3.48,'라그로타': 3.44,'느티나무 셀프BBQ': 2.78,
    '화담숲주막': 1.43,'카페테리아': 1.31,'화담숲카페': 1.14,'포레스트릿': 1.00,
}
mean_w = np.mean(list(raw_store_weight.values()))
store_weight = {k: v/mean_w for k,v in raw_store_weight.items()}
def map_store_weight(x:str)->float: return store_weight.get(x,1.0)

# =============================
# 피처 구성
# =============================
BASE_FEATURES = [
    '영업장명','메뉴명','휴일여부','휴일전날여부',
    '월_sin','월_cos','연중일자_sin','연중일자_cos','trend_norm',
    'woy_sin','woy_cos','same_dow_mean_4','roc_7_14'
]
LAGS    = [7,14,21,28]
WINDOWS = [7,14,21,28]

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

    # 완만한 추세 (0~1)
    yday_min = df['연중일자'].min()
    yday_max = df['연중일자'].max()
    df['trend_norm'] = (df['연중일자'] - yday_min) / max(1, (yday_max - yday_min))

    # 주차
    df['weekofyear'] = df['영업일자'].dt.isocalendar().week.astype(int)
    df['woy_sin'] = np.sin(2*np.pi*df['weekofyear']/52)
    df['woy_cos'] = np.cos(2*np.pi*df['weekofyear']/52)

    df['store_weight'] = df['영업장명'].map(map_store_weight).fillna(1.0).astype(float)
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
              'store_weight','요일','매출수량']].copy()

    # lag
    for L in lags:
        out[f'lag_{L}'] = g['매출수량'].shift(L)

    # rolling (어제까지)
    base = g['매출수량'].shift(1)
    roll_means, roll_stds, roll_sums, roll_max = [], [], [], []
    for W in wins:
        m = base.groupby(out['영업장명_메뉴명']).rolling(W).mean().reset_index(level=0, drop=True)
        s = base.groupby(out['영업장명_메뉴명']).rolling(W).std().reset_index(level=0, drop=True)
        sm = base.groupby(out['영업장명_메뉴명']).rolling(W).sum().reset_index(level=0, drop=True)
        mx = base.groupby(out['영업장명_메뉴명']).rolling(W).max().reset_index(level=0, drop=True)
        out[f'rolling_mean_{W}'] = m
        out[f'rolling_std_{W}']  = s
        out[f'rolling_sum_{W}']  = sm
        out[f'rolling_max_{W}']  = mx
        roll_means.append(f'rolling_mean_{W}')
        roll_stds.append(f'rolling_std_{W}')
        roll_sums.append(f'rolling_sum_{W}')
        roll_max.append(f'rolling_max_{W}')

    out[roll_means] = out[roll_means].groupby(out['영업장명_메뉴명']).ffill()
    for cols in [roll_means, roll_stds, roll_sums, roll_max]:
        out[cols] = out[cols].fillna(0.0)

    # 같은 요일 평균(최근 4주, 누수방지: shift(7) 기반 rolling)
    tmp = df.copy()
    tmp['same_dow'] = df.groupby(['영업장명_메뉴명','요일'])['매출수량'].shift(7)
    tmp['same_dow_mean_4'] = tmp.groupby(['영업장명_메뉴명','요일'])['same_dow']\
                                .rolling(4).mean().reset_index(level=[0,1], drop=True)
    out['same_dow_mean_4'] = tmp['same_dow_mean_4'].fillna(0.0)

    # 변화율 피처
    out['roc_7_14'] = (out['rolling_mean_7'] + 1e-6) / (out['rolling_mean_14'] + 1e-6)

    feature_cols = BASE_FEATURES + [f'lag_{L}' for L in lags] + roll_means + \
                   roll_stds + roll_sums + roll_max
    return out, feature_cols

# =============================
# 시계열 분할
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

# =============================
# 행렬 생성(day 고정)
# =============================
def build_matrices_for_day(train_df: pd.DataFrame, day: int,
                           tune_start, hold_start):
    feats_df, feat_cols = make_lag_roll_dayaware(train_df)
    # target: 미래 day
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

# =============================
# 외부 딥러닝 예측(NBEATS/NHITS)
# =============================
def _load_deep_file(path: str, ids: pd.Series) -> Optional[np.ndarray]:
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        if '영업장명_메뉴명' in df.columns and 'pred' in df.columns:
            ids_df = ids.to_frame(name='영업장명_메뉴명').reset_index()
            merged = ids_df.merge(df, on='영업장명_메뉴명', how='left').sort_values('index')
            v = merged['pred'].astype(float).fillna(0.0).values
            return v if len(v)==len(ids) else None
    except Exception:
        return None
    return None

def maybe_load_deep_preds(day:int, ids:pd.Series, deep_dir=DEEP_PREDS_DIR, prefer_rule=True):
    if not USE_DEEP: return None
    nh_path = os.path.join(deep_dir, f'NHITS_day{day}.csv')
    nb_path = os.path.join(deep_dir, f'NBEATS_day{day}.csv')
    nh_vec = _load_deep_file(nh_path, ids)
    nb_vec = _load_deep_file(nb_path, ids)
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
    extra = {"task_type":"GPU","devices":"0"} if use_gpu else {"thread_count": -1}

    def cb_objective(trial):
        params = {
            "loss_function": "MAE", "eval_metric": "MAE",
            "depth": trial.suggest_int("depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "iterations": MAX_ITER_CAT,
            "random_seed": RANDOM_SEED,
            "early_stopping_rounds": EARLY_STOP_CAT,
            "verbose": False,
            **extra
        }
        bt = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])
        if bt == "Bayesian":
            params["bootstrap_type"] = "Bayesian"
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.1, 10.0)
        else:
            params["bootstrap_type"] = "Bernoulli"
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

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

    study = optuna.create_study(direction='minimize', pruner=MedianPruner(), study_name='catboost')
    study.optimize(cb_objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_trial.params.copy()
    if bp.get('bootstrap_type') == 'Bayesian':
        bp.pop('subsample', None)
    else:
        bp.pop('bagging_temperature', None)

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
# Stacking helpers
# =============================
def purged_splits(n_samples:int, n_splits:int=4, gap:int=7):
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    indices = np.arange(n_samples)
    current = 0
    bounds = []
    for fsz in fold_sizes:
        bounds.append((current, current+fsz))
        current += fsz
    for (s,e) in bounds:
        va_idx = indices[s:e]
        tr_mask = np.ones(n_samples, dtype=bool)
        left = max(0, s - gap); right = min(n_samples, e + gap)
        tr_mask[left:right] = False
        tr_idx = indices[tr_mask]
        yield tr_idx, va_idx

def nnls_with_simplex(P: np.ndarray, y: np.ndarray, cap: float = 0.7) -> np.ndarray:
    """
    비음수/합=1 제약 가중치. SciPy가 있으면 NNLS → 정규화.
    없으면 강건 대안: Ridge(비음수 클리핑) → 정규화 → 상한(cap) 적용 → 재정규화.
    """
    k = P.shape[1]
    if SCIPY_NNLS_AVAILABLE:
        # 열별 2-노름으로 스케일 안정화(선택)
        col_norm = np.linalg.norm(P, axis=0) + 1e-12
        Pn = P / col_norm
        w_raw, _ = scipy_nnls(Pn, y)
        w = w_raw / (col_norm + 1e-12)
    else:
        # 대안: Ridge → 음수 클리핑
        ridge = Ridge(alpha=1.0, fit_intercept=False, random_state=RANDOM_SEED)
        ridge.fit(P, y)
        w = np.clip(ridge.coef_, 0.0, None)

    # 합=1 정규화 + 단일 모델 상한
    w = nonneg_simplex(w)
    if cap < 1.0:
        w = np.minimum(w, cap)
        w = nonneg_simplex(w)
    return w

# =============================
# 멀티 기간 검증(옵션) — 축약 앙상블
# =============================
def multi_period_eval(train_df: pd.DataFrame, day:int, periods:int=3):
    maes = []
    base_last = train_df['영업일자'].max()
    cat_cols = ['영업장명','메뉴명']

    for k in range(periods):
        hold_start = base_last - pd.Timedelta(days=VAL_HOLD_DAYS - 1 + k*VAL_HOLD_DAYS)
        tune_start = hold_start - pd.Timedelta(days=VAL_TUNE_DAYS)

        (X_tr, y_tr, sw_tr, _,
         X_tune, y_tune, sw_tune, ids_tune,
         X_hold, y_hold, sw_hold, ids_hold,
         feat_cols) = build_matrices_for_day(train_df, day, tune_start, hold_start)

        X_tr = ensure_unique_cols(X_tr).fillna(0)
        X_tune = ensure_unique_cols(X_tune).reindex(columns=X_tr.columns, fill_value=0).fillna(0)
        X_hold = ensure_unique_cols(X_hold).reindex(columns=X_tr.columns, fill_value=0).fillna(0)

        # 간단 sample weight (alpha=0.2 고정)
        def pos_boost(y): return 1.0 + 0.2*np.log1p(np.asarray(y))
        sw_tr_adj   = sw_tr   * np.where(y_tr.values   > 0, pos_boost(y_tr),   1.0)
        sw_tune_adj = sw_tune * np.where(y_tune.values > 0, pos_boost(y_tune), 1.0)

        # Cat (축약 파라미터)
        cb = CatBoostRegressor(
            loss_function="MAE", depth=6, learning_rate=0.1,
            iterations=1200, early_stopping_rounds=80,
            task_type=("GPU" if USE_GPU_CAT else "CPU"),
            devices="0", verbose=False
        )
        X_tr_tune = pd.concat([X_tr, X_tune], axis=0)
        y_tr_tune = pd.concat([y_tr, y_tune], axis=0)
        sw_tr_tune= np.concatenate([sw_tr_adj, sw_tune_adj])
        cb.fit(X_tr_tune, y_tr_tune, cat_features=[c for c in cat_cols if c in X_tr_tune.columns],
               sample_weight=sw_tr_tune, verbose=False)

        # Linear
        lin_feats = [c for c in X_tr_tune.columns if c not in cat_cols]
        lr = Ridge(alpha=1.0, fit_intercept=True, random_state=RANDOM_SEED)
        lr.fit(X_tr_tune[lin_feats], y_tr_tune, sample_weight=sw_tr_tune)

        # Tune preds (스태킹 가중치 학습)
        preds_tune = [
            np.clip(cb.predict(X_tune), 0, None),
            np.clip(lr.predict(X_tune[lin_feats]), 0, None)
        ]
        deep_tune = maybe_load_deep_preds(day, ids_tune)
        names = ["cat", "lin"]
        if deep_tune is not None:
            preds_tune.append(np.clip(deep_tune, 0, None))
            names.append("deep")
        P_t = np.vstack(preds_tune).T
        y_t = y_tune.values

        # Purged OOF → fold별 w, median
        W = []
        for tr_idx, va_idx in purged_splits(len(P_t), n_splits=4, gap=7):
            w = nnls_with_simplex(P_t[tr_idx], y_t[tr_idx], cap=0.7)
            W.append(w)
        w_final = np.median(np.vstack(W), axis=0)
        w_final = nonneg_simplex(np.minimum(w_final, 0.7))

        # Hold preds
        preds_hold = [
            np.clip(cb.predict(X_hold), 0, None),
            np.clip(lr.predict(X_hold[[c for c in lin_feats if c in X_hold.columns]]), 0, None)
        ]
        if "deep" in names:
            deep_hold = maybe_load_deep_preds(day, ids_hold)
            if deep_hold is not None:
                preds_hold.append(np.clip(deep_hold, 0, None))

        P_h = np.vstack(preds_hold).T
        y_h = y_hold.values
        pred = np.maximum(0.0, (P_h * w_final).sum(axis=1))
        # 제로 가드
        last4_zero = (X_hold[[f'lag_{L}' for L in [7,14,21,28]]].fillna(0).sum(axis=1) == 0)
        pred[last4_zero.values] = 0.0
        maes.append(mean_absolute_error(y_h, pred))

        del cb, lr
        gc.collect()

    return float(np.mean(maes)), float(np.std(maes))

# =============================
# 메인
# =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, required=True, help="예측 day (1~7)")
    parser.add_argument("--train_path", default=TRAIN_PATH)
    parser.add_argument("--test_glob",  default=TEST_GLOB)
    parser.add_argument("--submission_tpl", default=SUBMISSION_TPL)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--use_gpu_cat", action="store_true")
    parser.add_argument("--no_linear", action="store_true")
    parser.add_argument("--no_deep", action="store_true")
    parser.add_argument("--pos_alpha", type=float, default=None)
    parser.add_argument("--mp_eval", action="store_true")
    args = parser.parse_args()

    DAY = args.day
    os.makedirs(args.out_dir, exist_ok=True)

    # 옵션 적용
    global FAST_MODE, USE_GPU_CAT, USE_LINEAR, USE_DEEP, N_TRIALS_CAT
    if args.fast:
        FAST_MODE = True
        N_TRIALS_CAT = 15
    USE_GPU_CAT = args.use_gpu_cat
    USE_LINEAR = not args.no_linear
    USE_DEEP   = not args.no_deep

    print(f"=== RUN DAY {DAY} ===")
    print(f"FAST_MODE={FAST_MODE}, GPU_CAT={USE_GPU_CAT}, USE_LINEAR={USE_LINEAR}, USE_DEEP={USE_DEEP}")

    # 데이터 로드/정제
    train_df = pd.read_csv(args.train_path)
    neg_ct = (train_df['매출수량'] < 0).sum()
    if neg_ct:
        print(f"[정제] 음수 매출 {neg_ct}건 → 0으로 클립")
    train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
    train_df = create_features(train_df)

    # 멀티 기간 검증(옵션)
    if args.mp_eval:
        m, s = multi_period_eval(train_df, DAY, periods=3)
        print(f"[Multi-period] MAE mean={m:.4f} | std={s:.4f}")

    tune_start, hold_start = compute_cut_dates(train_df)
    print(f"tune_start={tune_start.date()}, hold_start={hold_start.date()}")

    (X_tr, y_tr, sw_tr, ids_tr,
     X_tune, y_tune, sw_tune, ids_tune,
     X_hold, y_hold, sw_hold, ids_hold,
     feat_cols) = build_matrices_for_day(train_df, DAY, tune_start, hold_start)

    # 안전 처리
    X_tr   = ensure_unique_cols(X_tr).fillna(0)
    X_tune = ensure_unique_cols(X_tune).reindex(columns=X_tr.columns, fill_value=0).fillna(0)
    X_hold = ensure_unique_cols(X_hold).reindex(columns=X_tr.columns, fill_value=0).fillna(0)

    # ===== Sample weight α 자동 선택 (또는 사용자 고정) =====
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
        sweep_cat_cols = [c for c in ['영업장명','메뉴명'] if c in X_tr.columns]
        for a in cand:
            _sw_tr, _sw_tune, _ = apply_sw(a)
            cb = CatBoostRegressor(loss_function="MAE", depth=6, learning_rate=0.1,
                                   iterations=500, early_stopping_rounds=60,
                                   task_type=("GPU" if USE_GPU_CAT else "CPU"),
                                   devices="0", verbose=False)
            cb.fit(X_tr, y_tr, eval_set=(X_tune, y_tune),cat_features=sweep_cat_cols,sample_weight=_sw_tr,verbose=False)
            p = np.clip(cb.predict(X_tune), 0, None)
            scores.append(mean_absolute_error(y_tune, p))
            del cb; gc.collect()
        alpha_best = cand[int(np.argmin(scores))]
        print(f"[SW] α 자동선택 = {alpha_best} (candidates={cand})")

    sw_tr, sw_tune, sw_hold = apply_sw(alpha_best)

    cat_cols = [c for c in ['영업장명','메뉴명'] if c in X_tr.columns]

    # ----------------- CatBoost 튠 & base(평가용) -----------------
    print("[CatBoost] 튠 시작...")
    cat_params = tune_catboost(X_tr, y_tr, X_tune, y_tune, cat_cols, sw_tr,
                               use_gpu=USE_GPU_CAT, n_trials=N_TRIALS_CAT)

    cat_base = CatBoostRegressor(**{**cat_params, "verbose": False})
    X_tr_tune  = pd.concat([X_tr,  X_tune], axis=0)
    y_tr_tune  = pd.concat([y_tr,  y_tune], axis=0)
    sw_tr_tune = np.concatenate([sw_tr, sw_tune])
    cat_base.fit(X_tr_tune, y_tr_tune, cat_features=cat_cols, sample_weight=sw_tr_tune, verbose=False)

    # ----------------- Linear base(평가용) -----------------
    lin_base = None
    if USE_LINEAR:
        lin_feats = [c for c in X_tr_tune.columns if c not in ['영업장명','메뉴명']]
        lin_base = Ridge(alpha=1.0, fit_intercept=True, random_state=RANDOM_SEED)
        lin_base.fit(X_tr_tune[lin_feats], y_tr_tune, sample_weight=sw_tr_tune)

    # ----------------- 튠 OOF 스태킹(딥 포함) -----------------
    print("[Stacking] 튠 OOF(purged) → NNLS/simplex 가중치")
    preds_tune = []; names = []

    p_cat_tune = np.clip(cat_base.predict(X_tune), 0, None)
    preds_tune.append(p_cat_tune); names.append("cat")

    if lin_base is not None:
        lin_feats = [c for c in X_tune.columns if c not in ['영업장명','메뉴명']]
        preds_tune.append(np.clip(lin_base.predict(X_tune[lin_feats]), 0, None)); names.append("lin")

    deep_tune = maybe_load_deep_preds(DAY, ids_tune)
    if deep_tune is not None:
        preds_tune.append(np.clip(deep_tune, 0, None)); names.append("deep")

    P_tune = np.vstack(preds_tune).T
    y_t    = y_tune.values

    # purged folds → fold별 w → median → 상한 적용
    W_list = []
    for tr_idx, va_idx in purged_splits(len(P_tune), n_splits=4, gap=7):
        w = nnls_with_simplex(P_tune[tr_idx], y_t[tr_idx], cap=0.7)
        W_list.append(w)
    w_final = np.median(np.vstack(W_list), axis=0)
    w_final = nonneg_simplex(np.minimum(w_final, 0.7))

    blend_weights = dict(zip(names, w_final))
    print(f"[Day {DAY}] stacking weights: {blend_weights}")

    # ----------------- HOLD 평가 -----------------
    preds_hold = [np.clip(cat_base.predict(X_hold), 0, None)]
    name_order_hold = ["cat"]

    if lin_base is not None:
        lin_feats_h = [c for c in X_hold.columns if c not in ['영업장명','메뉴명']]
        preds_hold.append(np.clip(lin_base.predict(X_hold[lin_feats_h]), 0, None))
        name_order_hold.append("lin")

    if "deep" in names:
        deep_hold = maybe_load_deep_preds(DAY, ids_hold)
        if deep_hold is not None:
            preds_hold.append(np.clip(deep_hold, 0, None))
            name_order_hold.append("deep")

    P_h = np.vstack(preds_hold).T
    wdict = dict(zip(names, w_final))
    w_used = np.array([wdict.get(n, 0.0) for n in name_order_hold], dtype=float)
    w_used = nonneg_simplex(w_used)

    blended_hold = np.maximum(0.0, (P_h * w_used).sum(axis=1))

    # 제로 가드
    last4_zero = (X_hold[[f'lag_{L}' for L in [7,14,21,28]]].fillna(0).sum(axis=1) == 0)
    blended_hold[last4_zero.values] = 0.0

    mae_hold = mean_absolute_error(y_hold, blended_hold)
    mae_hold_za = better_objective(y_hold, blended_hold)
    print(f"[Day {DAY}] HOLD MAE={mae_hold:.4f} | Zero-aware={mae_hold_za:.4f}")

    # ----------------- Final 재학습(제출용) -----------------
    print("[Final] 전체 데이터로 재학습 (inference 전용)")
    cat_final = CatBoostRegressor(**{**cat_params, "verbose": False})
    X_all  = pd.concat([X_tr_tune, X_hold], axis=0)
    y_all  = pd.concat([y_tr_tune, y_hold], axis=0)
    sw_all = np.concatenate([sw_tr_tune, sw_hold])
    cat_final.fit(X_all, y_all, cat_features=cat_cols, sample_weight=sw_all, verbose=False)

    lin_final = None
    if USE_LINEAR:
        lin_feats_all = [c for c in X_all.columns if c not in ['영업장명','메뉴명']]
        lin_final = Ridge(alpha=1.0, fit_intercept=True, random_state=RANDOM_SEED)
        lin_final.fit(X_all[lin_feats_all], y_all, sample_weight=sw_all)

    # ----------------- 메타 저장 -----------------
    meta = {
        "day": DAY,
        "feat_cols": feat_cols,
        "names": names,
        "weights": w_final.tolist(),
        "linear": bool(USE_LINEAR),
        "pos_alpha": alpha_best
    }
    with open(os.path.join(args.out_dir, f"meta_day{DAY}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ----------------- 추론(해당 DAY) & 부분 제출 -----------------
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

        # Cat
        p_cat_inf = np.clip(cat_final.predict(base_inf), 0, None)
        part_preds.append(p_cat_inf); infer_names.append("cat")

        # Linear
        if lin_final is not None:
            lin_feats_inf = [c for c in base_inf.columns if c not in ['영업장명','메뉴명']]
            part_preds.append(np.clip(lin_final.predict(base_inf[lin_feats_inf]), 0, None)); infer_names.append("lin")

        # Deep
        deep_vec = maybe_load_deep_preds(DAY, ids_series)
        if deep_vec is not None:
            part_preds.append(np.clip(deep_vec, 0, None)); infer_names.append("deep")

        # 블렌딩
        wdict = dict(zip(meta["names"], meta["weights"]))
        w = np.array([wdict.get(n, 0.0) for n in infer_names], dtype=float)
        w = nonneg_simplex(w)
        Pinf = np.vstack(part_preds).T
        blended = np.maximum(0.0, (Pinf * w).sum(axis=1))

        # 제로 가드
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

    # 샘플과 컬럼 맞춤
    cols_order = pd.read_csv(args.submission_tpl, nrows=0).columns
    pivot_df = pivot_df.reindex(columns=cols_order, fill_value=0)

    out_csv = os.path.join(args.out_dir, f"submission_day{DAY}.csv")
    pivot_df.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv}")

    # 메모리 청소
    del cat_final, lin_final
    gc.collect()

if __name__ == "__main__":
    main()