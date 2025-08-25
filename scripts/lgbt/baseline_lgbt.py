# -*- coding: utf-8 -*-
"""
LG Aimers: 식음업장 메뉴 7일 수요예측 (Strict 28-day, LightGBM GPU, Optuna)
- 학습/검증/추론 모두 "입력 28일" 제약을 유지 (추론 제약과 동일)
- 리더보드 산식(업장 가중, 품목 평균, 날짜 평균, A=0 제외)과 정확히 일치하는 검증 점수
- Hurdle 구조: LGBMClassifier(>0 여부) + LGBMRegressor(양수 크기), 예측 = p_pos * y_pos
- 하나의 분류기/회귀기 모델로 7-일 동시 예측: horizon(1..7) 피처 사용 (AR/roll-forward 사용 안 함)
- GPU: CUDA 우선, 실패 시 OpenCL→CPU 자동 폴백
- 음수 매출은 0으로 클립
- 업장 가중치는 훈련 sample_weight와 평가지표에 모두 반영
- Optuna 튜닝 + Pruner
- 최종 제출 파일 생성 (./data/sample_submission.csv 포맷)
"""

import os
import gc
import glob
import json
import math
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

import optuna
from optuna.pruners import MedianPruner

warnings.filterwarnings("ignore")


# =========================
# 사용자 설정 (필요시 조정)
# =========================
INPUT_LEN = 28        # 입력 길이 (고정)
OUT_LEN = 7           # 예측 길이 (고정)
SEED = 42

# "학습 환경 == 추론 환경"을 최대로 맞추려면 True 권장
STRICT_FINAL_28DAY = True

# 연중일자(day-of-year) 사용 여부 (사이클릭). 주차(week-of-year)는 기본 포함.
USE_DAY_OF_YEAR = True

# Optuna 설정
N_TRIALS = 200
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = False

# 데이터 경로
TRAIN_CSV = "./data/train/train.csv"
TEST_GLOB = "./data/test/TEST_*.csv"
SUBMISSION_TEMPLATE = "./data/sample_submission.csv"
SUBMISSION_OUT = "./data/submission_lgbm_hurdle_28day_optuna.csv"

# 공휴일 목록 (사용자 제공)
custom_holidays_list = [
    '2023-01-01', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-03-01', '2023-05-01',
    '2023-05-05', '2023-05-27', '2023-06-06', '2023-08-15', '2023-09-28', '2023-09-29', '2023-09-30',
    '2023-10-02', '2023-10-03', '2023-10-09', '2023-12-25',
    '2024-01-01', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12', '2024-03-01', '2024-04-10',
    '2024-05-01', '2024-05-05', '2024-05-06', '2024-05-15', '2024-06-06', '2024-08-15', '2024-09-16',
    '2024-09-17', '2024-09-18', '2024-10-01', '2024-10-03', '2024-10-09', '2024-12-25',
    '2025-01-01', '2025-01-28', '2025-01-29', '2025-01-30', '2025-03-01', '2025-03-03', '2025-05-01',
    '2025-05-05', '2025-05-06', '2025-06-06', '2025-08-15'
]
CUSTOM_HOLIDAYS = set(pd.to_datetime(custom_holidays_list))

# 업장 가중치 (상대 가중치 β)
STORE_WEIGHTS = {
    '미라시아': 7.71, '담하': 6.51, '연회장': 3.48, '라그로타': 3.44,
    '느티나무 셀프BBQ': 2.78, '화담숲주막': 1.43, '카페테리아': 1.31,
    '화담숲카페': 1.14, '포레스트릿': 1.00
}


# =========================================
# 유틸: LightGBM 기기 설정 (CUDA→OpenCL→CPU)
# =========================================
def lgb_device_params_chain():
    """CUDA 우선, 실패 시 OpenCL, 그 다음 CPU로 폴백."""
    return [
        {"device_type": "cuda"},
        {"device_type": "gpu"},   # OpenCL
        {"device_type": "cpu"},
    ]


def try_fit_with_fallback(make_model, X, y, sample_weight=None, eval_set=None, eval_names=None, is_classifier=False):
    """LightGBM를 CUDA→OpenCL→CPU 순으로 시도하여 fit. 첫 성공을 반환."""
    last_err = None
    for dev in lgb_device_params_chain():
        try:
            model = make_model(dev)
            model.fit(
                X, y,
                sample_weight=sample_weight,
                eval_set=eval_set,
                eval_names=eval_names,
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=VERBOSE_EVAL)]
            )
            return model, dev["device_type"]
        except Exception as e:
            last_err = e
            continue
    raise last_err


# =========================================
# 지표: 리더보드 산식 (설명대로 1:1 구현)
# =========================================
def leaderboard_metric_df(df_pred, store_weight, eps=1e-10):
    """
    df_pred columns:
      - store : 업장명 s
      - item  : 품목명 i
      - t     : 1..7 (horizon)
      - y_true, y_pred : 실제/예측
    규칙:
      - y_true==0인 날은 제외 → 품목별 시점 평균 → 업장 내 품목 평균 → 업장 가중 평균
    """
    df = df_pred.copy()
    if df.empty:
        return 0.0

    df["y_pred"] = np.clip(df["y_pred"].astype(float), 0, None)
    df["y_true"] = df["y_true"].astype(float)

    # 실제=0 제외
    df = df[df["y_true"] != 0]
    if df.empty:
        return 0.0

    num = 2.0 * (df["y_pred"] - df["y_true"]).abs()
    den = (df["y_pred"].abs() + df["y_true"].abs()) + eps
    df["smape"] = num / den

    # 품목 평균
    item_mean = df.groupby(["store", "item"])["smape"].mean().reset_index(name="smape_item")
    if item_mean.empty:
        return 0.0

    # 업장 평균
    store_mean = item_mean.groupby("store")["smape_item"].mean().reset_index(name="smape_store")
    store_mean["w"] = store_mean["store"].map(store_weight).fillna(1.0)
    store_mean["w_smape"] = store_mean["w"] * store_mean["smape_store"]

    w_sum = store_mean["w"].sum()
    if w_sum == 0:
        return 0.0
    return float(store_mean["w_smape"].sum() / w_sum)


# =========================================
# 데이터 로드/정리
# =========================================
def load_train():
    df = pd.read_csv(TRAIN_CSV)
    # 음수 매출 → 0
    df["매출수량"] = df["매출수량"].clip(lower=0)
    df["영업일자"] = pd.to_datetime(df["영업일자"])
    # 분리
    split = df["영업장명_메뉴명"].str.split("_", n=1, expand=True)
    df["영업장명"] = split[0]
    df["메뉴명"] = split[1]
    return df


def calendar_features(frame):
    """28일 입력 구간 내 마지막 날짜의 캘린더 피처 생성."""
    f = frame.copy()
    f["요일"] = f["영업일자"].dt.dayofweek
    f["월"] = f["영업일자"].dt.month
    f["주말여부"] = (f["요일"] >= 5).astype(int)
    f["공휴일여부"] = f["영업일자"].isin(CUSTOM_HOLIDAYS).astype(int)

    # 휴일 전날 (다음날이 주말/공휴일)
    f["다음날"] = f["영업일자"] + pd.to_timedelta(1, unit="D")
    f["휴일전날여부"] = ((f["다음날"].dt.dayofweek >= 5) | (f["다음날"].isin(CUSTOM_HOLIDAYS))).astype(int)
    f.drop(columns=["다음날"], inplace=True)

    # 연중주차 (ISO 주차와 달리 간단히 isocalendar().week 사용)
    f["연중주차"] = f["영업일자"].dt.isocalendar().week.astype(int)

    if USE_DAY_OF_YEAR:
        f["연중일자"] = f["영업일자"].dt.dayofyear

    return f


def build_samples(train_df, input_len=INPUT_LEN, out_len=OUT_LEN):
    """
    슬라이딩 윈도우로 학습/검증 샘플 구성.
    - 각 (업장명_메뉴명) 그룹에서 길이를 확보하면,
      입력 28일 → 타깃 7일(일별)로 펼쳐 'horizon' 1..7 행 생성
    - 피처: (입력 28일) 마지막 날의 캘린더 + 28일 구간 집계(rolling) + horizon
    - 금지: 28일 초과 lookback 사용 금지 → 집계는 입력 28일 내에서만 계산
    """
    # 인코더 준비
    store_le = LabelEncoder().fit(train_df["영업장명"])
    menu_le = LabelEncoder().fit(train_df["메뉴명"])

    all_rows = []
    # 그룹별 정렬
    for item, g in train_df.sort_values(["영업장명_메뉴명", "영업일자"]).groupby("영업장명_메뉴명", sort=False):
        if len(g) < (input_len + out_len):
            continue

        # 전처리(캘린더)
        g = calendar_features(g)

        # 28일 슬라이딩
        for i in range(0, len(g) - input_len - out_len + 1):
            hist = g.iloc[i:i+input_len]               # 입력 28일
            fut = g.iloc[i+input_len:i+input_len+out_len]  # 타깃 7일

            last = hist.iloc[-1]  # 마지막 날의 캘린더 피처 사용
            # 28일 구간 내부 집계 (규칙 준수)
            vals = hist["매출수량"].astype(float)
            roll_mean = vals.mean()
            roll_std = vals.std(ddof=0) if len(vals) > 1 else 0.0
            roll_max = vals.max()
            roll_min = vals.min()
            roll_sum = vals.sum()
            # 최근 k일 평균(28 이내)
            k7 = vals.tail(7).mean()
            k14 = vals.tail(14).mean() if len(vals) >= 14 else vals.mean()

            # 범주 인코딩
            store_id = store_le.transform([last["영업장명"]])[0]
            menu_id = menu_le.transform([last["메뉴명"]])[0]

            # 사이클릭 인코딩
            month = int(last["월"])
            weekday = int(last["요일"])
            weekofyear = int(last["연중주차"])

            def sincos(val, period):
                return math.sin(2 * math.pi * val / period), math.cos(2 * math.pi * val / period)

            mon_sin, mon_cos = sincos(month, 12)
            woy_sin, woy_cos = sincos(weekofyear, 53)  # 52~53주
            if USE_DAY_OF_YEAR:
                doy = int(last["연중일자"])
                doy_sin, doy_cos = sincos(doy, 366)
            else:
                doy_sin = doy_cos = 0.0

            base = {
                "store": last["영업장명"],
                "item": last["영업장명_메뉴명"],
                "last_input_date": last["영업일자"],
                # 범주/캘린더
                "store_id": store_id,
                "menu_id": menu_id,
                "weekday": weekday,
                "month": month,
                "is_weekend": int(last["주말여부"]),
                "is_holiday": int(last["공휴일여부"]),
                "is_pre_holiday": int(last["휴일전날여부"]),
                # 사이클릭
                "month_sin": mon_sin, "month_cos": mon_cos,
                "woy_sin": woy_sin, "woy_cos": woy_cos,
                "doy_sin": doy_sin, "doy_cos": doy_cos,
                # 28일 집계
                "roll_mean": roll_mean,
                "roll_std": 0.0 if np.isnan(roll_std) else roll_std,
                "roll_max": roll_max,
                "roll_min": roll_min,
                "roll_sum": roll_sum,
                "k7_mean": k7,
                "k14_mean": k14,
            }

            fut_vals = fut["매출수량"].astype(float).values
            # horizon 1..7으로 펼침
            for h in range(out_len):
                row = base.copy()
                row["horizon"] = h + 1
                y = fut_vals[h]
                row["y_true"] = y
                row["y_bin"] = 1 if y > 0 else 0
                all_rows.append(row)

    feat_df = pd.DataFrame(all_rows)
    # 훈련 샘플 가중치(업장 베타)
    feat_df["sample_weight"] = feat_df["store"].map(STORE_WEIGHTS).fillna(1.0)
    meta = feat_df[["store", "item", "last_input_date"]].drop_duplicates().reset_index(drop=True)

    encoders = {"store_le": store_le, "menu_le": menu_le}
    return feat_df, meta, encoders


# =========================================
# Folds 구성 (Strict 28-day CV)
# =========================================
def build_strict_28day_folds(meta_df, input_len=INPUT_LEN):
    """
    앵커 날짜 목록을 가장 최근 쪽에서 주기적으로 잡는다.
    각 fold D에서:
      - 학습: last_input_date ∈ [D-28, D-1]
      - 검증: last_input_date == D
    """
    max_date = meta_df["last_input_date"].max()
    # 앵커 후보: 7일 간격으로 최근에서 4개 선택 (필요시 조정)
    # (주의) train 마지막 유효 anchor는 "타깃 7일이 전부 train 내부"인 지점이지만
    # 여긴 이미 build_samples에서 생성 가능한 구간만 샘플화했으므로 OK.
    candidate = [max_date - pd.Timedelta(days=d) for d in [35, 28, 21, 14]]
    anchors = [d for d in candidate if d >= meta_df["last_input_date"].min()]
    anchors = sorted(list(set(anchors)))
    folds = []
    for D in anchors:
        tr_start = D - pd.Timedelta(days=input_len)
        tr_end = D - pd.Timedelta(days=1)
        folds.append({"anchor": D, "tr_start": tr_start, "tr_end": tr_end})
    return folds


# =========================================
# 피처/타깃 컬럼 정의
# =========================================
FEATURE_COLS = [
    # 범주/캘린더/사이클릭
    "store_id", "menu_id",
    "weekday", "month",
    "is_weekend", "is_holiday", "is_pre_holiday",
    "month_sin", "month_cos",
    "woy_sin", "woy_cos",
    "doy_sin", "doy_cos",
    # 28일 집계
    "roll_mean", "roll_std", "roll_max", "roll_min", "roll_sum",
    "k7_mean", "k14_mean",
    # horizon
    "horizon",
]


# =========================================
# Optuna 목적함수 (fold 평균 점수 최소화)
# =========================================
def make_models(trial):
    # Classifier(>0)
    params_cls = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": trial.suggest_int("n_estimators_cls", 400, 2000),
        "learning_rate": trial.suggest_float("lr_cls", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("leaves_cls", 31, 255),
        "max_depth": trial.suggest_int("depth_cls", 3, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_cls", 10, 300),
        "feature_fraction": trial.suggest_float("ff_cls", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bf_cls", 0.6, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("l1_cls", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("l2_cls", 1e-8, 10.0, log=True),
        "random_state": SEED,
        "verbose": -1,
    }

    # Regressor(+ part) - L1 회귀
    params_reg = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": trial.suggest_int("n_estimators_reg", 600, 2500),
        "learning_rate": trial.suggest_float("lr_reg", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("leaves_reg", 31, 255),
        "max_depth": trial.suggest_int("depth_reg", 3, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_reg", 10, 300),
        "feature_fraction": trial.suggest_float("ff_reg", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bf_reg", 0.6, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("l1_reg", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("l2_reg", 1e-8, 10.0, log=True),
        "random_state": SEED,
        "verbose": -1,
    }

    # 예측 후 클리핑 강도 (상한 완충)
    cap_factor = trial.suggest_float("cap_factor", 1.0, 2.0)

    return params_cls, params_reg, cap_factor


def opt_objective(trial, feat_df, folds):
    params_cls, params_reg, cap_factor = make_models(trial)

    fold_scores = []
    for f in folds:
        D = f["anchor"]
        tr_start, tr_end = f["tr_start"], f["tr_end"]

        # strict: 학습은 D-28 ~ D-1, 검증은 D
        tr_mask = (feat_df["last_input_date"] >= tr_start) & (feat_df["last_input_date"] <= tr_end)
        va_mask = (feat_df["last_input_date"] == D)

        X_tr = feat_df.loc[tr_mask, FEATURE_COLS].copy()
        y_tr_bin = feat_df.loc[tr_mask, "y_bin"].astype(int).values
        y_tr_pos = feat_df.loc[tr_mask, "y_true"].astype(float).values
        sw_tr = feat_df.loc[tr_mask, "sample_weight"].astype(float).values

        X_va = feat_df.loc[va_mask, FEATURE_COLS].copy()
        y_va_true = feat_df.loc[va_mask, "y_true"].astype(float).values
        sw_va = feat_df.loc[va_mask, "sample_weight"].astype(float).values

        # 분류기/회귀기 분리 (0이 아닌 y만 회귀에 사용?)
        # → 여기서는 전 샘플로 회귀 학습하되, 손실이 L1이라 0에 덜 민감.
        # 엄밀하게는 y>0만 회귀학습도 가능. (선택)
        def make_cls(dev):
            p = params_cls.copy()
            p.update(dev)
            return lgb.LGBMClassifier(**p)

        def make_reg(dev):
            p = params_reg.copy()
            p.update(dev)
            return lgb.LGBMRegressor(**p)

        # 학습 (CUDA→OpenCL→CPU)
        cls, dev_cls = try_fit_with_fallback(
            make_cls, X_tr, y_tr_bin, sample_weight=sw_tr,
            eval_set=[(X_va, feat_df.loc[va_mask, "y_bin"])],
            eval_names=["valid"],
            is_classifier=True
        )

        reg, dev_reg = try_fit_with_fallback(
            make_reg, X_tr, y_tr_pos, sample_weight=sw_tr,
            eval_set=[(X_va, y_va_true)],
            eval_names=["valid"],
            is_classifier=False
        )

        # 검증 예측
        p_pos = cls.predict_proba(X_va)[:, 1]
        y_pos = reg.predict(X_va)

        # 음수 방지 + 상한 완충
        y_pos = np.clip(y_pos, 0, None)
        # 품목별 최근 28일 평균×cap_factor 상한 (과대예측 억제)
        # 상한용 roll_mean은 X_va에 이미 존재
        y_cap = X_va["roll_mean"].values * cap_factor + 1e-6
        y_pred = np.minimum(y_pos, y_cap) * p_pos

        # 리더보드 점수 계산
        df_pred = pd.DataFrame({
            "store": feat_df.loc[va_mask, "store"].values,
            "item": feat_df.loc[va_mask, "item"].values,
            "t": feat_df.loc[va_mask, "horizon"].values,
            "y_true": y_va_true,
            "y_pred": y_pred
        })
        score = leaderboard_metric_df(df_pred, STORE_WEIGHTS)
        fold_scores.append(score)

        # Optuna pruner 관찰
        trial.report(np.mean(fold_scores), step=len(fold_scores))
        if trial.should_prune():
            raise optuna.TrialPruned()

        # 메모리
        del X_tr, X_va, y_tr_bin, y_tr_pos, y_va_true, sw_tr, sw_va, df_pred, cls, reg
        gc.collect()

    return float(np.mean(fold_scores))


# =========================================
# 최종 학습 (STRICT_FINAL_28DAY 스위치)
# =========================================
def final_train(feat_df, best_params, folds):
    params_cls, params_reg, cap_factor = best_params

    # 최종 학습 데이터 마스크
    if STRICT_FINAL_28DAY and len(folds) > 0:
        D = folds[-1]["anchor"]  # 가장 최근 앵커
        tr_start, tr_end = D - pd.Timedelta(days=INPUT_LEN), D - pd.Timedelta(days=1)
        mask = (feat_df["last_input_date"] >= tr_start) & (feat_df["last_input_date"] <= tr_end)
    else:
        mask = np.ones(len(feat_df), dtype=bool)

    X = feat_df.loc[mask, FEATURE_COLS].copy()
    y_bin = feat_df.loc[mask, "y_bin"].astype(int).values
    y_pos = feat_df.loc[mask, "y_true"].astype(float).values
    sw = feat_df.loc[mask, "sample_weight"].astype(float).values

    def make_cls(dev):
        p = params_cls.copy()
        p.update(dev)
        return lgb.LGBMClassifier(**p)

    def make_reg(dev):
        p = params_reg.copy()
        p.update(dev)
        return lgb.LGBMRegressor(**p)

    cls, dev1 = try_fit_with_fallback(
        make_cls, X, y_bin, sample_weight=sw,
        eval_set=[(X, y_bin)], eval_names=["train"], is_classifier=True
    )
    reg, dev2 = try_fit_with_fallback(
        make_reg, X, y_pos, sample_weight=sw,
        eval_set=[(X, y_pos)], eval_names=["train"], is_classifier=False
    )

    return {"cls": cls, "reg": reg, "cap_factor": cap_factor}


# =========================================
# 테스트 → 제출 생성
# =========================================
def make_test_features(test_df, encoders):
    """각 TEST 파일(28일)에서 마지막 날짜 기준 피처 1행 생성 + horizon 1..7로 7행 복제."""
    df = test_df.copy()
    df["영업일자"] = pd.to_datetime(df["영업일자"])
    df["매출수량"] = df["매출수량"].clip(lower=0)

    split = df["영업장명_메뉴명"].str.split("_", n=1, expand=True)
    df["영업장명"] = split[0]
    df["메뉴명"] = split[1]

    # 캘린더
    df = calendar_features(df)

    # 그룹별 28일 집계 → 마지막 행 추출
    rows = []
    for item, g in df.sort_values(["영업장명_메뉴명", "영업일자"]).groupby("영업장명_메뉴명", sort=False):
        if len(g) < INPUT_LEN:
            continue
        hist = g.iloc[-INPUT_LEN:]  # 정확히 28일
        last = hist.iloc[-1]

        vals = hist["매출수량"].astype(float)
        roll_mean = vals.mean()
        roll_std = vals.std(ddof=0) if len(vals) > 1 else 0.0
        roll_max = vals.max()
        roll_min = vals.min()
        roll_sum = vals.sum()
        k7 = vals.tail(7).mean()
        k14 = vals.tail(14).mean() if len(vals) >= 14 else vals.mean()

        store_id = encoders["store_le"].transform([last["영업장명"]])[0] \
            if last["영업장명"] in encoders["store_le"].classes_ else -1
        menu_id = encoders["menu_le"].transform([last["메뉴명"]])[0] \
            if last["메뉴명"] in encoders["menu_le"].classes_ else -1

        month = int(last["월"])
        weekday = int(last["요일"])
        weekofyear = int(last["연중주차"])

        def sincos(val, period):
            return math.sin(2 * math.pi * val / period), math.cos(2 * math.pi * val / period)

        mon_sin, mon_cos = sincos(month, 12)
        woy_sin, woy_cos = sincos(weekofyear, 53)
        if USE_DAY_OF_YEAR:
            doy = int(last["연중일자"])
            doy_sin, doy_cos = sincos(doy, 366)
        else:
            doy_sin = doy_cos = 0.0

        base = {
            "store": last["영업장명"],
            "item": last["영업장명_메뉴명"],
            "store_id": store_id,
            "menu_id": menu_id,
            "weekday": weekday,
            "month": month,
            "is_weekend": int(last["주말여부"]),
            "is_holiday": int(last["공휴일여부"]),
            "is_pre_holiday": int(last["휴일전날여부"]),
            "month_sin": mon_sin, "month_cos": mon_cos,
            "woy_sin": woy_sin, "woy_cos": woy_cos,
            "doy_sin": doy_sin, "doy_cos": doy_cos,
            "roll_mean": roll_mean,
            "roll_std": 0.0 if np.isnan(roll_std) else roll_std,
            "roll_max": roll_max,
            "roll_min": roll_min,
            "roll_sum": roll_sum,
            "k7_mean": k7,
            "k14_mean": k14,
        }

        for h in range(OUT_LEN):
            row = base.copy()
            row["horizon"] = h + 1
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["store", "item"] + FEATURE_COLS)

    feat = pd.DataFrame(rows)
    feat["sample_weight"] = feat["store"].map(STORE_WEIGHTS).fillna(1.0)
    return feat


def predict_block(models, X_block):
    """Hurdle 예측: p_pos * y_pos (상한 완충 포함)"""
    cls = models["cls"]
    reg = models["reg"]
    cap_factor = models["cap_factor"]

    p_pos = cls.predict_proba(X_block[FEATURE_COLS])[:, 1]
    y_pos = reg.predict(X_block[FEATURE_COLS])
    y_pos = np.clip(y_pos, 0, None)
    y_cap = X_block["roll_mean"].values * cap_factor + 1e-6
    y_pred = np.minimum(y_pos, y_cap) * p_pos
    return y_pred


def main():
    np.random.seed(SEED)

    # 1) 데이터 로드 & 샘플 생성
    print("Loading train...")
    train_df = load_train()
    print("Building samples (strict 28-day windows)...")
    feat_df, meta_df, encoders = build_samples(train_df)
    print(f"Train samples (rows x features): {feat_df.shape}")

    # 2) CV folds
    print("Building strict 28-day CV folds...")
    folds = build_strict_28day_folds(meta_df)
    anchors = [f["anchor"].strftime("%Y-%m-%d") for f in folds]
    print(f"Folds: {len(folds)}개, anchors = {anchors}")

    # 3) Optuna 튜닝
    pruner = MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    print("Optuna tuning starts...")
    study.optimize(lambda tr: opt_objective(tr, feat_df, folds), n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    print("Best value (weighted SMAPE):", best.value)
    print("Best params:", best.params)

    # best params unpack
    params_cls = {
        "objective": "binary", "metric": "binary_logloss",
        "n_estimators": best.params["n_estimators_cls"],
        "learning_rate": best.params["lr_cls"],
        "num_leaves": best.params["leaves_cls"],
        "max_depth": best.params["depth_cls"],
        "min_data_in_leaf": best.params["min_data_cls"],
        "feature_fraction": best.params["ff_cls"],
        "bagging_fraction": best.params["bf_cls"], "bagging_freq": 1,
        "lambda_l1": best.params["l1_cls"], "lambda_l2": best.params["l2_cls"],
        "random_state": SEED, "verbose": -1
    }
    params_reg = {
        "objective": "regression_l1", "metric": "mae",
        "n_estimators": best.params["n_estimators_reg"],
        "learning_rate": best.params["lr_reg"],
        "num_leaves": best.params["leaves_reg"],
        "max_depth": best.params["depth_reg"],
        "min_data_in_leaf": best.params["min_data_reg"],
        "feature_fraction": best.params["ff_reg"],
        "bagging_fraction": best.params["bf_reg"], "bagging_freq": 1,
        "lambda_l1": best.params["l1_reg"], "lambda_l2": best.params["l2_reg"],
        "random_state": SEED, "verbose": -1
    }
    cap_factor = best.params["cap_factor"]
    best_params = (params_cls, params_reg, cap_factor)

    # 4) 최종 학습
    print(f"Final training... STRICT_FINAL_28DAY={STRICT_FINAL_28DAY}")
    models = final_train(feat_df, best_params, folds)

    # 5) 테스트 추론 & 제출
    print("Inference for test files...")
    test_files = sorted(glob.glob(TEST_GLOB))
    sub_tmpl = pd.read_csv(SUBMISSION_TEMPLATE)
    all_rows = []

    for idx, path in enumerate(test_files):
        test_df = pd.read_csv(path)
        block = make_test_features(test_df, encoders)

        if block.empty:
            continue

        y_pred = predict_block(models, block)

        # 제출 형식: TEST_xx+{1..7}일 행 × 열은 '영업장명_메뉴명'
        block_out = block[["item", "horizon"]].copy()
        block_out["pred"] = np.round(y_pred).astype(int)
        for h in range(1, OUT_LEN+1):
            tmp = block_out[block_out["horizon"] == h]
            row = pd.DataFrame({"영업일자": [f"TEST_{idx:02d}+{h}일"]}).set_index("영업일자")
            # pivot without losing items not present: reindex later
            pivot = tmp.pivot_table(index=None, columns="item", values="pred", aggfunc="first")
            pivot.index = row.index
            all_rows.append(pivot)

    if not all_rows:
        raise RuntimeError("No predictions generated for submission.")

    sub_df = pd.concat(all_rows).reset_index()
    sub_df = sub_df.rename(columns={"index": "영업일자"})
    # 샘플 제출 포맷과 컬럼 정렬/누락 채움
    sub_df = sub_df.reindex(columns=sub_tmpl.columns, fill_value=0)
    sub_df.to_csv(SUBMISSION_OUT, index=False)
    print(f"Saved submission to: {SUBMISSION_OUT}")


if __name__ == "__main__":
    main()
