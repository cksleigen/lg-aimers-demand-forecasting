#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_nbeats_tcn_hurdle.py
- N-BEATS(Hurdle) + TCN(Hurdle) 통합 학습/추론 (Strict 28-day 준수)
- Optuna 튜닝 포함, 단일 GPU(A6000) 기준
- 대회 제출 CSV 생성

필수 폴더/파일:
- ./data/train/train.csv
- ./data/test/TEST_00.csv ~ TEST_09.csv
- ./data/sample_submission.csv

실행 예:
python train_nbeats_tcn_hurdle.py --n_trials 50 --epochs 40 --blend_alpha 0.6

주의:
- 외부 데이터/사전학습 가중치 사용 없음
- 모든 feature는 입력 28일에서만 생성 (미래 covariate 미사용)
"""
import os
import math
import gc
import glob
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import optuna

# -----------------------------
# 0) 설정 / 재현성
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # AMP는 GradScaler/autocast로 켭니다.

# -----------------------------
# 1) 도메인 / 데이터 규칙
# -----------------------------
CUSTOM_HOLIDAYS = set(pd.to_datetime([
    '2023-01-01','2023-01-21','2023-01-22','2023-01-23','2023-01-24','2023-03-01','2023-05-01',
    '2023-05-05','2023-05-27','2023-06-06','2023-08-15','2023-09-28','2023-09-29','2023-09-30',
    '2023-10-02','2023-10-03','2023-10-09','2023-12-25',
    '2024-01-01','2024-02-09','2024-02-10','2024-02-11','2024-02-12','2024-03-01','2024-04-10',
    '2024-05-01','2024-05-05','2024-05-06','2024-05-15','2024-06-06','2024-08-15','2024-09-16',
    '2024-09-17','2024-09-18','2024-10-01','2024-10-03','2024-10-09','2024-12-25',
    '2025-01-01','2025-01-28','2025-01-29','2025-01-30','2025-03-01','2025-03-03','2025-05-01',
    '2025-05-05','2025-05-06','2025-06-06','2025-08-15'
]))

# 업장 가중치 (사용자 제공 상대 β)
STORE_WEIGHTS: Dict[str, float] = {
    '미라시아': 7.71, '담하': 6.51, '연회장': 3.48, '라그로타': 3.44,
    '느티나무 셀프BBQ': 2.78, '화담숲주막': 1.43, '카페테리아': 1.31,
    '화담숲카페': 1.14, '포레스트릿': 1.00
}

IN_LEN = 28  # Strict
OUT_LEN = 7  # Horizon

# -----------------------------
# 2) 데이터 유틸
# -----------------------------
def clip_negatives_to_zero(df: pd.DataFrame) -> pd.DataFrame:
    df['매출수량'] = df['매출수량'].clip(lower=0)
    return df

def split_store_menu(df: pd.DataFrame) -> pd.DataFrame:
    df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
    return df

def is_holiday_or_weekend(d: pd.Timestamp) -> int:
    return 1 if (d in CUSTOM_HOLIDAYS or d.dayofweek >= 5) else 0

def dow_sin_cos(d: pd.Timestamp) -> Tuple[float, float]:
    # 요일(0~6) → sin/cos 변환
    x = d.dayofweek
    return math.sin(2*math.pi*x/7.0), math.cos(2*math.pi*x/7.0)

def build_encoders(train_df: pd.DataFrame):
    store2id = {s:i+1 for i,s in enumerate(sorted(train_df['영업장명'].unique()))}  # 0=UNK
    menu2id  = {m:i+1 for i,m in enumerate(sorted(train_df['메뉴명'].unique()))}
    id2store = {v:k for k,v in store2id.items()}
    id2menu  = {v:k for k,v in menu2id.items()}
    return store2id, menu2id, id2store, id2menu

def item_scale(x: np.ndarray) -> float:
    # 최근 28일의 양수 median (없으면 1.0)
    pos = x[x > 0]
    if len(pos) == 0:
        return 1.0
    return float(np.median(pos))

# -----------------------------
# 3) 슬라이딩 윈도우 (Strict 28→7)
# -----------------------------
@dataclass
class WindowRecord:
    # 한 샘플(윈도우)의 메타/데이터
    store_id: int
    menu_id: int
    item_name: str
    last_input_date: pd.Timestamp
    x28: np.ndarray           # shape [28]
    y7: np.ndarray            # shape [7]
    x28_dow_sin: np.ndarray   # shape [28]
    x28_dow_cos: np.ndarray   # shape [28]
    x28_is_hol: np.ndarray    # shape [28]
    scale: float

def build_windows(train_df: pd.DataFrame,
                  store2id: Dict[str,int], menu2id: Dict[str,int]) -> List[WindowRecord]:
    windows: List[WindowRecord] = []
    # 그룹: 품목(=영업장명_메뉴명)
    for item, g in tqdm(train_df.groupby('영업장명_메뉴명'), desc="윈도우 생성"):
        g = g.sort_values('영업일자')
        vals = g['매출수량'].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(g['영업일자']).to_numpy()
        store = g['영업장명'].iloc[0]
        menu  = g['메뉴명'].iloc[0]
        sid = store2id.get(store, 0)
        mid = menu2id.get(menu, 0)
        # 최소 길이 검사
        if len(vals) < IN_LEN + OUT_LEN:
            continue
        for i in range(0, len(vals) - IN_LEN - OUT_LEN + 1):
            x = vals[i:i+IN_LEN].copy()
            y = vals[i+IN_LEN:i+IN_LEN+OUT_LEN].copy()
            ds = pd.to_datetime(dates[i:i+IN_LEN])
            # 입력 28일 기반 부가채널(미래 covariate 불가)
            hol = np.array([is_holiday_or_weekend(d) for d in ds], dtype=np.float32)
            sin = np.array([dow_sin_cos(d)[0] for d in ds], dtype=np.float32)
            cos = np.array([dow_sin_cos(d)[1] for d in ds], dtype=np.float32)
            sc = item_scale(x)
            xr = WindowRecord(
                store_id=sid,
                menu_id=mid,
                item_name=item,
                last_input_date=pd.to_datetime(ds[-1]),
                x28=x, y7=y,
                x28_dow_sin=sin, x28_dow_cos=cos, x28_is_hol=hol,
                scale=sc
            )
            windows.append(xr)
    return windows

# -----------------------------
# 4) 엄격 Rolling-CV folds
# -----------------------------
def make_strict_cv_anchors(windows: List[WindowRecord]) -> List[pd.Timestamp]:
    """
    windows에서 각 샘플의 last_input_date를 모아, 최근 날짜 기준으로
    21, 28, 35, ...일 전 후보 지점들 중 실제 존재하는 날짜에 가장 가까운
    anchor를 최대 4개 선택한다.
    - 모든 연산은 datetime64[D]로 통일해 타입 에러를 방지.
    - 반환은 pd.Timestamp 리스트.
    """
    if len(windows) == 0:
        return []

    # 모든 last_input_date를 datetime64[D]로 강제 변환
    last_dates64 = np.array([np.datetime64(w.last_input_date, 'D') for w in windows], dtype='datetime64[D]')
    max_date64 = last_dates64.max()

    # 최근에서 거꾸로 후보 생성
    # (엄격 28-입력 규칙은 fold split에서 보장되고, 여기선 anchor 후보만 잡음)
    day_offsets = [21, 28, 35, 42, 49, 56, 63, 70]
    candidates64 = [max_date64 - np.timedelta64(d, 'D') for d in day_offsets]

    anchors64 = []
    used = set()

    for c64 in candidates64:
        # last_dates64 중 c64와 '일 단위' 거리 최소 인덱스
        diffs_days = np.abs((last_dates64 - c64).astype('timedelta64[D]').astype(int))
        idx = int(diffs_days.argmin())
        anchor64 = last_dates64[idx]

        # c64와 anchor64의 차이가 3일 이내면 anchor로 채택
        gap = int(np.abs((anchor64 - c64) / np.timedelta64(1, 'D')))
        if gap <= 3 and anchor64 not in used:
            anchors64.append(anchor64)
            used.add(anchor64)

        if len(anchors64) == 4:
            break

    # 중복 제거 및 정렬
    anchors64 = sorted(list(set(anchors64)))
    # pd.Timestamp로 변환해서 반환
    anchors = [pd.Timestamp(a64.astype('datetime64[ns]')) for a64 in anchors64]
    return anchors

def split_fold(windows: List[WindowRecord], anchor: pd.Timestamp):
    # 검증: last_input_date == anchor
    val_idx = [i for i,w in enumerate(windows) if w.last_input_date == anchor]
    # 학습: last_input_date <= anchor - 7일  (검증 horizon과 시간 분리)
    cutoff = anchor - pd.Timedelta(days=OUT_LEN)
    tr_idx  = [i for i,w in enumerate(windows) if w.last_input_date <= cutoff]
    return tr_idx, val_idx

# -----------------------------
# 5) PyTorch Dataset
# -----------------------------
class WinDataset(Dataset):
    def __init__(self, ws: List[WindowRecord], store_id2name: Dict[int,str]):
        self.ws = ws
        self.store_id2name = store_id2name
    def __len__(self): return len(self.ws)
    def __getitem__(self, i):
        w = self.ws[i]
        # 입력은 scale로 나눠 정규화
        x28 = torch.tensor(w.x28 / w.scale, dtype=DTYPE)
        xchan = torch.stack([
            torch.tensor(w.x28 / w.scale, dtype=DTYPE),
            torch.tensor(w.x28_dow_sin, dtype=DTYPE),
            torch.tensor(w.x28_dow_cos, dtype=DTYPE),
            torch.tensor(w.x28_is_hol, dtype=DTYPE),
        ], dim=0)  # [C=4, 28]
        y7 = torch.tensor(w.y7 / w.scale, dtype=DTYPE)
        y_mask_pos = (y7 > 0).to(DTYPE)
        sid = torch.tensor(w.store_id, dtype=torch.long)
        mid = torch.tensor(w.menu_id, dtype=torch.long)
        sw = torch.tensor(STORE_WEIGHTS.get(self.store_id2name.get(w.store_id, ''), 1.0), dtype=DTYPE)
        scale = torch.tensor(w.scale, dtype=DTYPE)
        return x28, xchan, y7, y_mask_pos, sid, mid, sw, scale

# -----------------------------
# 6) 모델: N-BEATS(Hurdle)
#   (간결/안정 버전: block FC 쌓아 forecast 직접 산출)
# -----------------------------
class NBEATSBlock(nn.Module):
    def __init__(self, in_len=IN_LEN, hidden=512, dropout=0.1, out_len=OUT_LEN):
        super().__init__()
        self.fc1 = nn.Linear(in_len, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        # backcast는 생략하고 forecast만 누적 (간결화)
        self.theta = nn.Linear(hidden, out_len)
    def forward(self, x):
        # x: [B, in_len]
        h = F.relu(self.fc1(x))
        h = self.drop(F.relu(self.fc2(h)))
        h = self.drop(F.relu(self.fc3(h)))
        f = self.theta(h)
        return f

class NBEATSHurdle(nn.Module):
    def __init__(self, in_len=IN_LEN, out_len=OUT_LEN, hidden=512, blocks=4, dropout=0.1,
                 store_vocab=1, menu_vocab=1, emb_store=48, emb_menu=0, cond_hidden=128):
        super().__init__()
        self.store_emb = nn.Embedding(store_vocab+1, emb_store) if emb_store>0 else None
        self.menu_emb  = nn.Embedding(menu_vocab+1, emb_menu) if emb_menu>0 else None
        cond_in = 0
        if emb_store>0: cond_in += emb_store
        if emb_menu>0:  cond_in += emb_menu
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in if cond_in>0 else 1, cond_hidden),
            nn.ReLU(),
            nn.Linear(cond_hidden, cond_hidden),
            nn.ReLU()
        ) if cond_in>0 else None

        self.blocks = nn.ModuleList([NBEATSBlock(in_len, hidden, dropout, out_len) for _ in range(blocks)])
        self.head_zero = nn.Linear(out_len + (cond_hidden if cond_in>0 else 0), out_len)  # logits
        self.head_pos  = nn.Linear(out_len + (cond_hidden if cond_in>0 else 0), out_len)  # positive magnitude (softplus)

    def forward(self, x28, sid, mid):
        # x28: [B, 28]
        B = x28.size(0)
        # base forecast 합
        f_sum = 0
        for blk in self.blocks:
            f_sum = f_sum + blk(x28)  # [B, 7]
        # 조건 임베딩
        cond = None
        parts = []
        if self.store_emb is not None:
            parts.append(self.store_emb(sid))
        if self.menu_emb is not None:
            parts.append(self.menu_emb(mid))
        if len(parts) > 0:
            cond = torch.cat(parts, dim=-1)  # [B, cond_in]
            cond = self.cond_mlp(cond)       # [B, cond_hidden]
        # 컨캣 후 hurdle 헤드
        if cond is not None:
            h = torch.cat([f_sum, cond], dim=-1)
        else:
            h = f_sum
        logits = self.head_zero(h)  # [B, 7]
        pos    = F.softplus(self.head_pos(h))  # [B, 7], >=0
        return logits, pos

# -----------------------------
# 7) 모델: TCN(Hurdle) (보조)
# -----------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp
    def forward(self, x):  # x: [B,C,T]
        return x[:, :, :-self.chomp].contiguous() if self.chomp>0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x); out = self.chomp1(out); out = self.relu1(out); out = self.drop1(out)
        out = self.conv2(out); out = self.chomp2(out); out = self.relu2(out); out = self.drop2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNHurdle(nn.Module):
    def __init__(self, in_ch=4, channels=256, levels=4, kernel=5, dropout=0.1,
                 store_vocab=1, menu_vocab=1, emb_store=32, emb_menu=0, cond_hidden=64):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(levels):
            layers += [TemporalBlock(ch, channels, kernel, dilation=2**i, dropout=dropout)]
            ch = channels
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # [B, ch, 1]
        self.store_emb = nn.Embedding(store_vocab+1, emb_store) if emb_store>0 else None
        self.menu_emb  = nn.Embedding(menu_vocab+1, emb_menu) if emb_menu>0 else None
        cond_in = 0
        if emb_store>0: cond_in += emb_store
        if emb_menu>0:  cond_in += emb_menu
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in if cond_in>0 else 1, cond_hidden),
            nn.ReLU(),
            nn.Linear(cond_hidden, cond_hidden),
            nn.ReLU()
        ) if cond_in>0 else None
        head_in = channels + (cond_hidden if cond_in>0 else 0)
        self.head_zero = nn.Linear(head_in, OUT_LEN)
        self.head_pos  = nn.Linear(head_in, OUT_LEN)

    def forward(self, xchan, sid, mid):
        # xchan: [B, C, 28]
        h = self.tcn(xchan)              # [B, ch, T]
        h = self.pool(h).squeeze(-1)     # [B, ch]
        cond = None
        parts = []
        if self.store_emb is not None: parts.append(self.store_emb(sid))
        if self.menu_emb is not None:  parts.append(self.menu_emb(mid))
        if len(parts) > 0:
            cond = torch.cat(parts, dim=-1)
            cond = self.cond_mlp(cond)
        if cond is not None:
            h = torch.cat([h, cond], dim=-1)
        logits = self.head_zero(h)                   # [B,7]
        pos    = F.softplus(self.head_pos(h))       # [B,7] >=0
        return logits, pos

# -----------------------------
# 8) Hurdle Loss & Metric(W-SMAPE)
# -----------------------------
def hurdle_loss(logits, pos_pred, y_true, y_mask_pos, store_weight, lambda_pos=0.2):
    """
    logits: [B,7] (sale>0)
    pos_pred: [B,7] (양수 크기)
    y_true: [B,7]  (정규화됨)
    y_mask_pos: [B,7] (y_true>0:1, else 0)
    store_weight: [B]
    """
    bce = F.binary_cross_entropy_with_logits(logits, y_mask_pos, reduction='none')  # [B,7]
    # positive part: MAE only where y>0
    mae = torch.abs(pos_pred - y_true) * y_mask_pos  # [B,7]
    # 가중치 적용
    sw = store_weight.view(-1,1)
    bce = (bce * sw).mean()
    # y_mask_pos 합이 0인 배치 보호
    denom = (y_mask_pos * sw).sum().clamp_min(1.0)
    mae  = mae.sum() / denom
    return bce + lambda_pos * mae, bce.item(), mae.item()

def weighted_smape_eval(y_true, y_pred, store_ids, item_names, id2store: Dict[int,str]) -> float:
    """
    대회 공식식에 맞춘 W-SMAPE 근사:
    Score = sum_s w_s * ( (1/|I_s|) * sum_{i in I_s} ( (1/T_i) * sum_{t: A!=0} SMAPE ) )
    """
    # CPU numpy 변환
    y_t = y_true.reshape(-1, OUT_LEN)
    y_p = y_pred.reshape(-1, OUT_LEN)
    sids = np.array(store_ids).reshape(-1)
    items = np.array(item_names).reshape(-1)

    # per-store -> per-item -> per-time
    # 실제=0 제외
    store2items = {}
    for idx in range(len(items)):
        sname = id2store.get(int(sids[idx]), '')
        store2items.setdefault(sname, []).append(idx)

    total = 0.0
    for sname, idxs in store2items.items():
        w = STORE_WEIGHTS.get(sname, 1.0)
        if len(idxs) == 0:
            continue
        item_scores = []
        for j in idxs:
            a = y_t[j]
            p = y_p[j]
            mask = (a != 0)
            if mask.sum() == 0:
                continue
            sm = 2.0 * np.abs(a[mask] - p[mask]) / (np.abs(a[mask]) + np.abs(p[mask]) + 1e-10)
            item_scores.append(sm.mean())
        if len(item_scores) == 0:
            continue
        total += w * (np.mean(item_scores))
    return float(total)

# -----------------------------
# 9) 학습 루프
# -----------------------------
@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_pos: float = 0.2
    blend_alpha: float = 0.6  # NBEATS weight in ensemble
    num_workers: int = 4
    amp: bool = True
    patience: int = 8

def train_one_fold(nmodel, tmodel, cfg: TrainConfig, train_ws, val_ws, id2store):
    train_ds = WinDataset(train_ws, id2store)
    val_ds   = WinDataset(val_ws, id2store)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    nmodel = nmodel.to(DEVICE)
    tmodel = tmodel.to(DEVICE)
    params = list(nmodel.parameters()) + list(tmodel.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_metric = float('inf')
    best_state = None
    bad = 0

    for epoch in range(cfg.epochs):
        nmodel.train(); tmodel.train()
        tr_loss, tr_bce, tr_mae = 0.0, 0.0, 0.0
        for batch in train_loader:
            x28, xchan, y7, ymask, sid, mid, sw, scale = [b.to(DEVICE) for b in batch]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                n_logits, n_pos = nmodel(x28, sid, mid)
                t_logits, t_pos = tmodel(xchan, sid, mid)
                # 앙상블: 학습 중엔 각자 loss 계산 후 평균 → 안정
                ln, bce_n, mae_n = hurdle_loss(n_logits, n_pos, y7, ymask, sw, cfg.lambda_pos)
                lt, bce_t, mae_t = hurdle_loss(t_logits, t_pos, y7, ymask, sw, cfg.lambda_pos)
                loss = 0.5*ln + 0.5*lt
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr_loss += loss.item() * x28.size(0)
            tr_bce  += 0.5*(bce_n + bce_t) * x28.size(0)
            tr_mae  += 0.5*(mae_n + mae_t) * x28.size(0)

        # 검증
        nmodel.eval(); tmodel.eval()
        y_true_all, y_pred_all, store_ids_all, item_names_all = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x28, xchan, y7, ymask, sid, mid, sw, scale = [b.to(DEVICE) for b in batch]
                n_logits, n_pos = nmodel(x28, sid, mid)
                t_logits, t_pos = tmodel(xchan, sid, mid)
                # 앙상블 출력
                n_hat = torch.sigmoid(n_logits) * n_pos
                t_hat = torch.sigmoid(t_logits) * t_pos
                hat = cfg.blend_alpha * n_hat + (1.0 - cfg.blend_alpha) * t_hat
                # scale 복원
                hat = hat * scale.view(-1,1)
                ytrue = y7 * scale.view(-1,1)

                y_true_all.append(ytrue.detach().cpu().numpy())
                y_pred_all.append(hat.detach().cpu().numpy())
                store_ids_all.append(sid.detach().cpu().numpy())
                # item_name은 Dataset에 없으니 placeholder: store_id 기반만 메트릭에 필요
                # (공식식은 item 단위 평균 포함 → 여기선 동일 샘플 id를 item 이름으로 대체)
                # 정확히 item 단위로 묶으려면 Dataset에 item_name도 반환하게 바꿔도 무방
                item_names_all.append(np.arange(ytrue.size(0)))

        y_true_all = np.concatenate(y_true_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)
        store_ids_all = np.concatenate(store_ids_all, axis=0)
        item_names_all = np.concatenate(item_names_all, axis=0)

        metric = weighted_smape_eval(y_true_all, y_pred_all, store_ids_all, item_names_all, id2store)

        # Early stopping
        if metric < best_metric - 1e-6:
            best_metric = metric
            bad = 0
            # 모델 상태 저장
            best_state = {
                "n": {k:v.cpu() for k,v in nmodel.state_dict().items()},
                "t": {k:v.cpu() for k,v in tmodel.state_dict().items()},
            }
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    # 베스트 상태 복원
    if best_state is not None:
        nmodel.load_state_dict(best_state["n"])
        tmodel.load_state_dict(best_state["t"])
    return best_metric, nmodel, tmodel

# -----------------------------
# 10) Optuna 목적함수
# -----------------------------
def build_models_from_trial(trial, store_vocab, menu_vocab):
    # N-BEATS
    n_hidden = trial.suggest_categorical("n_hidden", [256, 384, 512, 640])
    n_blocks = trial.suggest_int("n_blocks", 2, 6)
    n_dropout = trial.suggest_float("n_dropout", 0.05, 0.2)
    emb_store = trial.suggest_categorical("emb_store", [32, 48, 64])
    use_menu = trial.suggest_categorical("use_menu", [False, True])
    emb_menu = trial.suggest_categorical("emb_menu", [0, 8, 16]) if use_menu else 0
    n_cond = trial.suggest_categorical("n_cond", [64, 128, 192])

    n_model = NBEATSHurdle(
        hidden=n_hidden, blocks=n_blocks, dropout=n_dropout,
        store_vocab=store_vocab, menu_vocab=menu_vocab,
        emb_store=emb_store, emb_menu=emb_menu, cond_hidden=n_cond
    )

    # TCN
    t_channels = trial.suggest_categorical("t_channels", [128, 192, 256, 320])
    t_levels   = trial.suggest_int("t_levels", 3, 5)
    t_kernel   = trial.suggest_categorical("t_kernel", [3,5,7])
    t_dropout  = trial.suggest_float("t_dropout", 0.05, 0.2)
    t_emb_store = trial.suggest_categorical("t_emb_store", [16, 32, 48])
    t_use_menu  = trial.suggest_categorical("t_use_menu", [False, True])
    t_emb_menu  = trial.suggest_categorical("t_emb_menu", [0, 8, 16]) if t_use_menu else 0
    t_cond      = trial.suggest_categorical("t_cond", [32, 64, 96])

    t_model = TCNHurdle(
        in_ch=4, channels=t_channels, levels=t_levels, kernel=t_kernel, dropout=t_dropout,
        store_vocab=store_vocab, menu_vocab=menu_vocab,
        emb_store=t_emb_store, emb_menu=t_emb_menu, cond_hidden=t_cond
    )

    # Hurdle pos loss weight
    lambda_pos = trial.suggest_float("lambda_pos", 0.1, 0.5)
    # blend
    blend_alpha = trial.suggest_float("blend_alpha", 0.4, 0.8)
    # opt
    lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
    wd = trial.suggest_float("wd", 1e-6, 5e-4, log=True)
    bs = trial.suggest_categorical("batch_size", [1024, 2048, 4096])

    tcfg = TrainConfig(
        epochs=trial.user_attrs.get("epochs", 30),
        batch_size=bs, lr=lr, weight_decay=wd,
        lambda_pos=lambda_pos, blend_alpha=blend_alpha,
        amp=True, patience=6
    )
    return n_model, t_model, tcfg

def opt_objective(trial, folds, windows, id2store, store_vocab, menu_vocab):
    nmodel, tmodel, cfg = build_models_from_trial(trial, store_vocab, menu_vocab)
    # fold별 학습/검증 → 평균 점수
    scores = []
    for (tr_idx, val_idx) in folds:
        train_ws = [windows[i] for i in tr_idx]
        val_ws   = [windows[i] for i in val_idx]
        metric, n_best, t_best = train_one_fold(nmodel, tmodel, cfg, train_ws, val_ws, id2store)
        scores.append(metric)
        # fold 간 모델 재사용(계속 학습) 방지 → 가중치 재초기화
        nmodel, tmodel, _ = build_models_from_trial(trial, store_vocab, menu_vocab)
    score = float(np.mean(scores))
    return score

# -----------------------------
# 11) Inference & Submission
# -----------------------------
@torch.no_grad()
def predict_one_file(nmodel, tmodel, df_test, store2id, menu2id) -> pd.DataFrame:
    df = df_test.copy()
    df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    preds_all = []
    for item, g in df.groupby('영업장명_메뉴명'):
        g = g.sort_values('영업일자')
        if len(g) < IN_LEN:
            continue
        x = g['매출수량'].to_numpy(dtype=np.float32)[-IN_LEN:]
        d = pd.to_datetime(g['영업일자']).to_numpy()[-IN_LEN:]
        sin = np.array([dow_sin_cos(pd.to_datetime(dt))[0] for dt in d], dtype=np.float32)
        cos = np.array([dow_sin_cos(pd.to_datetime(dt))[1] for dt in d], dtype=np.float32)
        hol = np.array([is_holiday_or_weekend(pd.to_datetime(dt)) for dt in d], dtype=np.float32)
        sc = item_scale(x)
        x28 = torch.tensor((x / sc)[None,:], dtype=DTYPE, device=DEVICE)        # [1,28]
        xch = torch.tensor(np.stack([x/sc, sin, cos, hol], axis=0)[None,:], dtype=DTYPE, device=DEVICE)  # [1,4,28]
        sid = torch.tensor([store2id.get(g['영업장명'].iloc[0], 0)], dtype=torch.long, device=DEVICE)
        mid = torch.tensor([menu2id.get(g['메뉴명'].iloc[0], 0)], dtype=torch.long, device=DEVICE)

        nlog, npos = nmodel(x28, sid, mid)
        tlog, tpos = tmodel(xch, sid, mid)
        nhat = torch.sigmoid(nlog) * npos
        that = torch.sigmoid(tlog) * tpos
        hat = args.blend_alpha * nhat + (1 - args.blend_alpha) * that
        y_hat = (hat * sc).squeeze(0).detach().cpu().numpy()  # [7]

        preds_all.append((item, y_hat))
    # DataFrame: 행=아이템, 열=1..7일
    out = []
    for item, y in preds_all:
        for k in range(OUT_LEN):
            out.append((item, k+1, int(np.rint(max(0, y[k])))))
    out_df = pd.DataFrame(out, columns=['영업장명_메뉴명','day_k','pred'])
    return out_df

def build_submission(preds_per_test: List[pd.DataFrame], sample_sub_header: pd.Index) -> pd.DataFrame:
    # preds_per_test[i] has columns: ['영업장명_메뉴명','day_k','pred']
    # 만들기: 'TEST_XX+K일' 행, 컬럼=아이템명
    rows = []
    for t_idx, pdf in enumerate(preds_per_test):
        for k in range(1, OUT_LEN+1):
            row_id = f"TEST_{t_idx:02d}+{k}일"
            r = pdf[pdf['day_k']==k][['영업장명_메뉴명','pred']]
            rows.append((row_id, r))
    # pivot 병합
    frames = []
    for row_id, r in rows:
        pivot = r.set_index('영업장명_메뉴명').T
        pivot.index = [row_id]
        frames.append(pivot)
    sub = pd.concat(frames, axis=0).reset_index().rename(columns={'index':'영업일자'})
    # 샘플 제출 헤더 순서 맞추기 (없는 컬럼은 0으로)
    sub = sub.reindex(columns=sample_sub_header, fill_value=0)
    return sub

# -----------------------------
# 12) 메인
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--blend_alpha", type=float, default=0.6)
    ap.add_argument("--save_dir", type=str, default="./outputs_nbeats_tcn")
    ap.add_argument("--inference_only", action="store_true")
    return ap.parse_args()

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) 데이터 로드/전처리
    train_df = pd.read_csv("./data/train/train.csv")
    train_df = clip_negatives_to_zero(train_df)
    train_df = split_store_menu(train_df)
    train_df['영업일자'] = pd.to_datetime(train_df['영업일자'])

    # 인코더
    store2id, menu2id, id2store, id2menu = build_encoders(train_df)

    # 2) 슬라이딩 윈도우 (Strict 28→7)
    windows = build_windows(train_df, store2id, menu2id)

    # 3) CV 앵커/폴드 구성
    anchors = make_strict_cv_anchors(windows)
    if len(anchors) == 0:
        raise RuntimeError("CV anchors가 생성되지 않았습니다. 학습 데이터의 길이를 확인하세요.")
    print(f"STRICT 28-day CV folds 구성...\nAnchors = {[str(a.date()) for a in anchors]}")
    folds = [split_fold(windows, a) for a in anchors if len(split_fold(windows, a)[1])>0]
    if len(folds) == 0:
        raise RuntimeError("유효한 CV fold가 없습니다. 데이터 분포/기간을 확인하세요.")
    print(f"Folds 개수: {len(folds)}")

    # 4) Optuna 튜닝 (선택: inference_only시 skip)
    best_trial_params = None
    if not args.inference_only:
        def objective(trial):
            # epoch 짧게 (튜닝 속도↑), 최종 재학습 때 args.epochs 사용
            trial.set_user_attr("epochs", max(20, args.epochs//2))
            score = opt_objective(trial, folds, windows, id2store, len(store2id), len(menu2id))
            return score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
        print("Best trial:", study.best_trial.number, "score:", study.best_value)
        best_trial_params = study.best_trial.params
        with open(os.path.join(args.save_dir,"best_params.json"), "w") as f:
            json.dump(best_trial_params, f, ensure_ascii=False, indent=2)
    else:
        # 튜닝 스킵 시 합리적 기본값
        best_trial_params = {
            "n_hidden":512,"n_blocks":4,"n_dropout":0.1,"emb_store":48,"use_menu":False,"emb_menu":0,"n_cond":128,
            "t_channels":256,"t_levels":4,"t_kernel":5,"t_dropout":0.1,"t_emb_store":32,"t_use_menu":False,"t_emb_menu":0,"t_cond":64,
            "lambda_pos":0.2,"blend_alpha":args.blend_alpha,"lr":1e-3,"wd":1e-4,"batch_size":2048
        }

    # 5) 최종 모델 구성 & 전체 재학습(anchors의 가장 최근 fold 기준 조기종료 포함)
    #    - 실제로는 여러 fold 합쳐서 학습해도 되지만, Strict 준수 + 최근 분포를 반영하려 anchor[-1] 기준으로 학습
    #    - 필요하면 train_ws를 folds의 train idx 전부 합집합으로 확장 가능
    nmodel, tmodel, tcfg = build_models_from_trial(optuna.trial.FixedTrial(best_trial_params),
                                                   len(store2id), len(menu2id))
    # 최종 에폭/블렌드 반영
    tcfg.epochs = args.epochs
    tcfg.blend_alpha = best_trial_params.get("blend_alpha", args.blend_alpha)

    # train/val split: 가장 최근 anchor로
    tr_idx, val_idx = folds[-1]
    train_ws = [windows[i] for i in tr_idx]
    val_ws   = [windows[i] for i in val_idx]

    print("최종 재학습 시작...")
    best_metric, n_best, t_best = train_one_fold(nmodel, tmodel, tcfg, train_ws, val_ws, id2store)
    print(f"최종 검증 점수 (W-SMAPE, 낮을수록 좋음): {best_metric:.6f}")

    # 저장
    torch.save(n_best.state_dict(), os.path.join(args.save_dir, "nbeats_hurdle.pth"))
    torch.save(t_best.state_dict(), os.path.join(args.save_dir, "tcn_hurdle.pth"))
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump({
            "IN_LEN": IN_LEN, "OUT_LEN": OUT_LEN,
            "store2id": store2id, "menu2id": menu2id,
            "blend_alpha": tcfg.blend_alpha
        }, f, ensure_ascii=False, indent=2)

    # 6) 추론 & 제출 생성
    print("추론 및 제출 CSV 생성...")
    # 모델 로드/평가모드
    n_best.eval(); t_best.eval()

    sample_sub = pd.read_csv("./data/sample_submission.csv", nrows=0)
    test_files = sorted(glob.glob("./data/test/TEST_*.csv"))
    preds_per_test = []
    for tf in test_files:
        df_t = pd.read_csv(tf)
        df_t = clip_negatives_to_zero(df_t)
        out_df = predict_one_file(n_best, t_best, df_t, store2id, menu2id)
        preds_per_test.append(out_df)

    submission = build_submission(preds_per_test, sample_sub.columns)
    out_path = os.path.join(args.save_dir, "submission_nbeats_tcn_hurdle.csv")
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"제출 파일 저장 완료: {out_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
