#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_nbeats_tcn_hurdle_ampflex.py
- N-BEATS(Hurdle) + TCN(Hurdle) with enhanced features (strict 28-day input only)
- FP16/BF16 switchable via --amp_dtype {fp16,bf16} (A6000 friendly; bf16→fp16 fallback if unsupported)
- Optuna tuning (optional), EMA, OneCycleLR
- Competition rules compliant: no external data, no future covariates, independent inference per sample
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
DTYPE = torch.float32  # 모델 파라미터 dtype(계산은 autocast로 혼합정밀)

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

# 업장 가중치 (상대 β → sample weight로 사용)
STORE_WEIGHTS: Dict[str, float] = {
    '미라시아': 7.71, '담하': 6.51, '연회장': 3.48, '라그로타': 3.44,
    '느티나무 셀프BBQ': 2.78, '화담숲주막': 1.43, '카페테리아': 1.31,
    '화담숲카페': 1.14, '포레스트릿': 1.00
}

IN_LEN = 28  # Strict
OUT_LEN = 7  # Horizon

# -----------------------------
# 2) 데이터 유틸 + (N-HiTS 피처 이식)
# -----------------------------
def clip_negatives_to_zero(df: pd.DataFrame) -> pd.DataFrame:
    df['매출수량'] = df['매출수량'].clip(lower=0)
    return df

def split_store_menu(df: pd.DataFrame) -> pd.DataFrame:
    df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
    return df

def get_menu_category(menu_name: str) -> str:
    m = str(menu_name).lower()
    if any(w in m for w in ['막걸리','소주','맥주','와인','참이슬','처음처럼','카스','하이네켄','버드와이저','스텔라']): return 'alcohol'
    if any(w in m for w in ['찌개','탕','국밥','라면','해장국','갈비탕']): return 'hot_food'
    if any(w in m for w in ['삼겹','갈비','목살','bbq','구이','불고기']): return 'bbq'
    if any(w in m for w in ['아이스크림','식혜','콜라','스프라이트','에이드']): return 'dessert_drink'
    if any(w in m for w in ['아메리카노','라떼','커피']): return 'coffee'
    if any(w in m for w in ['냉면','파스타','스파게티','면','우동']): return 'noodles'
    if any(w in m for w in ['비빔밥','볶음밥','공깃밥','정식']): return 'rice'
    return 'others'

def get_store_type(store_name: str) -> str:
    if store_name == '느티나무 셀프BBQ': return 'outdoor'
    if store_name in ['라그로타','미라시아']: return 'fine_dining'
    if store_name == '담하': return 'traditional'
    if store_name == '연회장': return 'event'
    if store_name in ['카페테리아','포레스트릿','화담숲카페']: return 'casual'
    return 'specialty'

def _sin_cos(v: int, period: int):
    ang = 2*math.pi*v/period
    return math.sin(ang), math.cos(ang)

def build_calendar_channels(dates_28: List[pd.Timestamp]) -> Dict[str, np.ndarray]:
    """입력 28일만으로 생성하는 달력/도메인 채널들(미래 X, 규정 준수)"""
    dow = np.array([d.weekday() for d in dates_28], np.int16)
    month = np.array([d.month for d in dates_28], np.int16)
    day = np.array([d.day for d in dates_28], np.int16)
    week = np.array([int(pd.Timestamp(d).isocalendar().week) for d in dates_28], np.int16)
    quarter = np.array([pd.Timestamp(d).quarter for d in dates_28], np.int16)

    is_fri = (dow==4).astype(np.float32); is_sat=(dow==5).astype(np.float32); is_sun=(dow==6).astype(np.float32)
    is_weekend = (dow>=5).astype(np.float32)
    is_hol = np.array([1.0 if (d in CUSTOM_HOLIDAYS or d.weekday()>=5) else 0.0 for d in dates_28], np.float32)

    tomorrow = [d + pd.Timedelta(days=1) for d in dates_28]
    yesterday = [d - pd.Timedelta(days=1) for d in dates_28]
    is_before_hol = np.array([1.0 if (t in CUSTOM_HOLIDAYS) else 0.0 for t in tomorrow], np.float32)
    is_after_hol  = np.array([1.0 if (y in CUSTOM_HOLIDAYS) else 0.0 for y in yesterday], np.float32)

    is_mstart = (day<=3).astype(np.float32); is_mend=(day>=28).astype(np.float32)
    is_spring = np.isin(month,[4,5,6]).astype(np.float32)
    is_summer = np.isin(month,[7,8]).astype(np.float32)
    is_autumn = np.isin(month,[9,10,11]).astype(np.float32)
    is_winter = np.isin(month,[12,1,2,3]).astype(np.float32)
    is_vac_summer = np.isin(month,[7,8]).astype(np.float32)
    is_vac_winter = np.isin(month,[12,1,2]).astype(np.float32)

    msin,mcos = np.array([_sin_cos(m,12) for m in month], np.float32).T
    dsin,dcos = np.array([_sin_cos(x,7)  for x in dow],   np.float32).T
    wsin,wcos = np.array([_sin_cos(w,52) for w in week],  np.float32).T
    qsin,qcos = np.array([_sin_cos(q,4)  for q in quarter], np.float32).T

    return {
      'dow_sin':dsin,'dow_cos':dcos,'month_sin':msin,'month_cos':mcos,
      'week_sin':wsin,'week_cos':wcos,'quarter_sin':qsin,'quarter_cos':qcos,
      'is_fri':is_fri,'is_sat':is_sat,'is_sun':is_sun,'is_weekend':is_weekend,'is_hol':is_hol,
      'is_before_hol':is_before_hol,'is_after_hol':is_after_hol,
      'is_mstart':is_mstart,'is_mend':is_mend,
      'is_spring':is_spring,'is_summer':is_summer,'is_autumn':is_autumn,'is_winter':is_winter,
      'is_vac_summer':is_vac_summer,'is_vac_winter':is_vac_winter
    }

def build_encoders(train_df: pd.DataFrame):
    store2id = {s:i+1 for i,s in enumerate(sorted(train_df['영업장명'].unique()))}  # 0=UNK
    menu2id  = {m:i+1 for i,m in enumerate(sorted(train_df['메뉴명'].unique()))}
    train_df['menu_cat']  = train_df['메뉴명'].apply(get_menu_category)
    train_df['store_typ'] = train_df['영업장명'].apply(get_store_type)
    cat2id  = {c:i+1 for i,c in enumerate(sorted(train_df['menu_cat'].unique()))}
    type2id = {t:i+1 for i,t in enumerate(sorted(train_df['store_typ'].unique()))}
    id2store = {v:k for k,v in store2id.items()}
    id2menu  = {v:k for k,v in menu2id.items()}
    return store2id, menu2id, id2store, id2menu, cat2id, type2id

def item_scale(x: np.ndarray) -> float:
    pos = x[x > 0]
    if len(pos) == 0:
        return 1.0
    return float(np.median(pos))

# -----------------------------
# 3) 슬라이딩 윈도우 (Strict 28→7)
# -----------------------------
@dataclass
class WindowRecord:
    store_id: int
    menu_id: int
    cat_id: int
    type_id: int
    item_name: str
    last_input_date: pd.Timestamp
    x28: np.ndarray           # [28]
    y7: np.ndarray            # [7]
    scale: float
    chan_dict: Dict[str, np.ndarray]  # 가변 채널들(각 [28])

def build_windows(train_df: pd.DataFrame,
                  store2id: Dict[str,int], menu2id: Dict[str,int],
                  cat2id: Dict[str,int], type2id: Dict[str,int]) -> List[WindowRecord]:
    windows: List[WindowRecord] = []
    for item, g in tqdm(train_df.groupby('영업장명_메뉴명'), desc="윈도우 생성"):
        g = g.sort_values('영업일자')
        vals = g['매출수량'].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(g['영업일자']).to_list()
        if len(vals) < IN_LEN + OUT_LEN:
            continue
        store = g['영업장명'].iloc[0]
        menu  = g['메뉴명'].iloc[0]
        sid = store2id.get(store, 0)
        mid = menu2id.get(menu, 0)
        cid = cat2id.get(get_menu_category(menu), 0)
        tid = type2id.get(get_store_type(store), 0)

        for i in range(0, len(vals) - IN_LEN - OUT_LEN + 1):
            x = vals[i:i+IN_LEN].copy()
            y = vals[i+IN_LEN:i+IN_LEN+OUT_LEN].copy()
            ds = dates[i:i+IN_LEN]
            cal = build_calendar_channels(ds)  # 입력 28일 기반
            sc = item_scale(x)
            windows.append(WindowRecord(
                store_id=sid, menu_id=mid, cat_id=cid, type_id=tid,
                item_name=item, last_input_date=pd.to_datetime(ds[-1]),
                x28=x, y7=y, scale=sc, chan_dict=cal
            ))
    return windows

# -----------------------------
# 4) 엄격 Rolling-CV folds
# -----------------------------
def make_strict_cv_anchors(windows: List[WindowRecord]) -> List[pd.Timestamp]:
    if len(windows) == 0:
        return []
    last_dates64 = np.array([np.datetime64(w.last_input_date, 'D') for w in windows], dtype='datetime64[D]')
    max_date64 = last_dates64.max()
    day_offsets = [21, 28, 35, 42, 49, 56, 63, 70]
    candidates64 = [max_date64 - np.timedelta64(d, 'D') for d in day_offsets]

    anchors64 = []
    used = set()
    for c64 in candidates64:
        diffs_days = np.abs((last_dates64 - c64).astype('timedelta64[D]').astype(int))
        idx = int(diffs_days.argmin())
        anchor64 = last_dates64[idx]
        gap = int(np.abs((anchor64 - c64) / np.timedelta64(1, 'D')))
        if gap <= 3 and anchor64 not in used:
            anchors64.append(anchor64)
            used.add(anchor64)
        if len(anchors64) == 4:
            break
    anchors64 = sorted(list(set(anchors64)))
    anchors = [pd.Timestamp(a64.astype('datetime64[ns]')) for a64 in anchors64]
    return anchors

def split_fold(windows: List[WindowRecord], anchor: pd.Timestamp):
    val_idx = [i for i,w in enumerate(windows) if w.last_input_date == anchor]
    cutoff = anchor - pd.Timedelta(days=OUT_LEN)
    tr_idx  = [i for i,w in enumerate(windows) if w.last_input_date <= cutoff]
    return tr_idx, val_idx

# -----------------------------
# 5) Dataset
# -----------------------------
class WinDataset(Dataset):
    def __init__(self, ws: List[WindowRecord], store_id2name: Dict[int,str]):
        self.ws = ws
        self.store_id2name = store_id2name
        self.chan_keys = list(ws[0].chan_dict.keys()) if len(ws)>0 else []

    def __len__(self): return len(self.ws)

    def __getitem__(self, i):
        w = self.ws[i]
        x_norm = (w.x28 / w.scale).astype(np.float32)
        chans = [x_norm] + [w.chan_dict[k].astype(np.float32) for k in self.chan_keys]
        xchan = torch.tensor(np.stack(chans, axis=0), dtype=DTYPE)  # [C,28]

        y7 = torch.tensor((w.y7 / w.scale), dtype=DTYPE)
        y_mask_pos = (y7 > 0).to(DTYPE)

        sid = torch.tensor(w.store_id, dtype=torch.long)
        mid = torch.tensor(w.menu_id, dtype=torch.long)
        cid = torch.tensor(w.cat_id, dtype=torch.long)
        tid = torch.tensor(w.type_id, dtype=torch.long)

        sw = torch.tensor(STORE_WEIGHTS.get(self.store_id2name.get(w.store_id,''), 1.0), dtype=DTYPE)
        scale = torch.tensor(w.scale, dtype=DTYPE)

        cal_stat = torch.tensor(np.array([w.chan_dict[k].mean() for k in self.chan_keys], np.float32))
        return torch.tensor(x_norm, dtype=DTYPE), xchan, y7, y_mask_pos, sid, mid, sw, scale, cid, tid, cal_stat

# -----------------------------
# 6) 모델: N-BEATS(Hurdle)
# -----------------------------
class NBEATSBlock(nn.Module):
    def __init__(self, in_len=IN_LEN, hidden=512, dropout=0.1, out_len=OUT_LEN):
        super().__init__()
        self.fc1 = nn.Linear(in_len, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.theta = nn.Linear(hidden, out_len)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.drop(F.relu(self.fc2(h)))
        h = self.drop(F.relu(self.fc3(h)))
        f = self.theta(h)
        return f

class NBEATSHurdle(nn.Module):
    def __init__(self, in_len=IN_LEN, out_len=OUT_LEN, hidden=512, blocks=4, dropout=0.1,
                 store_vocab=1, menu_vocab=1, cat_vocab=1, type_vocab=1,
                 emb_store=48, emb_menu=0, emb_cat=16, emb_type=8,
                 cond_hidden=128, cal_stat_dim=0):
        super().__init__()
        self.store_emb = nn.Embedding(store_vocab+1, emb_store) if emb_store>0 else None
        self.menu_emb  = nn.Embedding(menu_vocab+1, emb_menu) if emb_menu>0 else None
        self.cat_emb   = nn.Embedding(cat_vocab+1, emb_cat)   if emb_cat>0  else None
        self.type_emb  = nn.Embedding(type_vocab+1, emb_type) if emb_type>0 else None

        cond_in = sum([e for e in [emb_store, emb_menu, emb_cat, emb_type] if e>0])
        self.cal_proj = nn.Linear(max(cal_stat_dim,1), cond_hidden//2) if cal_stat_dim>0 else None
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in + (cond_hidden//2 if self.cal_proj else 0) if cond_in>0 else 1, cond_hidden),
            nn.ReLU(), nn.Linear(cond_hidden, cond_hidden), nn.ReLU()
        ) if cond_in>0 or self.cal_proj else None

        self.blocks = nn.ModuleList([NBEATSBlock(in_len, hidden, dropout, out_len) for _ in range(blocks)])
        head_in = out_len + (cond_hidden if self.cond_mlp else 0)
        self.head_zero = nn.Linear(head_in, out_len)
        self.head_pos  = nn.Linear(head_in, out_len)

    def forward(self, x28, sid, mid, cid, tid, cal_stat=None):
        f_sum = 0
        for blk in self.blocks:
            f_sum = f_sum + blk(x28)
        parts = []
        if self.store_emb is not None: parts.append(self.store_emb(sid))
        if self.menu_emb  is not None: parts.append(self.menu_emb(mid))
        if self.cat_emb   is not None: parts.append(self.cat_emb(cid))
        if self.type_emb  is not None: parts.append(self.type_emb(tid))
        cond = None
        if len(parts)>0 or self.cal_proj is not None:
            feats = []
            if len(parts)>0: feats.append(torch.cat(parts, dim=-1))
            if self.cal_proj is not None and cal_stat is not None:
                feats.append(self.cal_proj(cal_stat))
            cond = self.cond_mlp(torch.cat(feats, dim=-1) if len(feats)>1 else feats[0])
        h = torch.cat([f_sum, cond], dim=-1) if cond is not None else f_sum
        logits = self.head_zero(h)
        pos    = F.softplus(self.head_pos(h))
        return logits, pos

# -----------------------------
# 7) 모델: TCN(Hurdle)
# -----------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp): super().__init__(); self.chomp = chomp
    def forward(self, x): return x[:, :, :-self.chomp].contiguous() if self.chomp>0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad); self.relu1 = nn.ReLU(); self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad); self.relu2 = nn.ReLU(); self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv1(x); out = self.chomp1(out); out = self.relu1(out); out = self.drop1(out)
        out = self.conv2(out); out = self.chomp2(out); out = self.relu2(out); out = self.drop2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNHurdle(nn.Module):
    def __init__(self, in_ch, channels=256, levels=4, kernel=5, dropout=0.1,
                 store_vocab=1, menu_vocab=1, cat_vocab=1, type_vocab=1,
                 emb_store=32, emb_menu=0, emb_cat=16, emb_type=8,
                 cond_hidden=64, cal_stat_dim=0):
        super().__init__()
        layers = []; ch = in_ch
        for i in range(levels):
            layers += [TemporalBlock(ch, channels, kernel, dilation=2**i, dropout=dropout)]
            ch = channels
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.store_emb = nn.Embedding(store_vocab+1, emb_store) if emb_store>0 else None
        self.menu_emb  = nn.Embedding(menu_vocab+1, emb_menu) if emb_menu>0 else None
        self.cat_emb   = nn.Embedding(cat_vocab+1, emb_cat)   if emb_cat>0  else None
        self.type_emb  = nn.Embedding(type_vocab+1, emb_type) if emb_type>0 else None

        cond_in = sum([e for e in [emb_store, emb_menu, emb_cat, emb_type] if e>0])
        self.cal_proj = nn.Linear(max(cal_stat_dim,1), cond_hidden//2) if cal_stat_dim>0 else None
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in + (cond_hidden//2 if self.cal_proj else 0) if cond_in>0 else 1, cond_hidden),
            nn.ReLU(), nn.Linear(cond_hidden, cond_hidden), nn.ReLU()
        ) if cond_in>0 or self.cal_proj else None

        head_in = channels + (cond_hidden if self.cond_mlp else 0)
        self.head_zero = nn.Linear(head_in, OUT_LEN)
        self.head_pos  = nn.Linear(head_in, OUT_LEN)

    def forward(self, xchan, sid, mid, cid, tid, cal_stat=None):
        h = self.tcn(xchan); h = self.pool(h).squeeze(-1)
        parts = []
        if self.store_emb is not None: parts.append(self.store_emb(sid))
        if self.menu_emb  is not None: parts.append(self.menu_emb(mid))
        if self.cat_emb   is not None: parts.append(self.cat_emb(cid))
        if self.type_emb  is not None: parts.append(self.type_emb(tid))
        if len(parts)>0 or self.cal_proj is not None:
            feats = []
            if len(parts)>0: feats.append(torch.cat(parts, dim=-1))
            if self.cal_proj is not None and cal_stat is not None:
                feats.append(self.cal_proj(cal_stat))
            cond = self.cond_mlp(torch.cat(feats, dim=-1) if len(feats)>1 else feats[0])
            h = torch.cat([h, cond], dim=-1)
        logits = self.head_zero(h)
        pos    = F.softplus(self.head_pos(h))
        return logits, pos

# -----------------------------
# 8) Hurdle Loss & Metric(W-SMAPE)
# -----------------------------
def hurdle_loss(logits, pos_pred, y_true, y_mask_pos, store_weight, lambda_pos=0.2, use_smape_pos=False):
    bce = F.binary_cross_entropy_with_logits(logits, y_mask_pos, reduction='none')  # [B,7]
    if use_smape_pos:
        yt = y_true; yp = pos_pred
        den = (torch.abs(yt)+torch.abs(yp)).clamp_min(1e-3)
        pos_term = 2.0*torch.abs(yp-yt)/den
        mae = pos_term * y_mask_pos
    else:
        mae = torch.abs(pos_pred - y_true) * y_mask_pos
    sw = store_weight.view(-1,1)
    bce = (bce * sw).mean()
    denom = (y_mask_pos * sw).sum().clamp_min(1.0)
    mae  = mae.sum() / denom
    return bce + lambda_pos * mae, bce.item(), mae.item()

def weighted_smape_eval(y_true, y_pred, store_ids, item_names, id2store: Dict[int,str]) -> float:
    y_t = y_true.reshape(-1, OUT_LEN)
    y_p = y_pred.reshape(-1, OUT_LEN)
    sids = np.array(store_ids).reshape(-1)
    items = np.array(item_names).reshape(-1)

    store2items = {}
    for idx in range(len(items)):
        sname = id2store.get(int(sids[idx]), '')
        store2items.setdefault(sname, []).append(idx)

    total = 0.0
    for sname, idxs in store2items.items():
        w = STORE_WEIGHTS.get(sname, 1.0)
        if len(idxs) == 0: continue
        item_scores = []
        for j in idxs:
            a = y_t[j]; p = y_p[j]
            mask = (a != 0)
            if mask.sum() == 0: continue
            sm = 2.0 * np.abs(a[mask] - p[mask]) / (np.abs(a[mask]) + np.abs(p[mask]) + 1e-10)
            item_scores.append(sm.mean())
        if len(item_scores) == 0: continue
        total += w * (np.mean(item_scores))
    return float(total)

# -----------------------------
# 9) AMP 유틸 + EMA
# -----------------------------
def amp_context(dtype_choice: str):
    """dtype_choice in {'fp16','bf16'}; bf16 미지원 시 fp16으로 폴백"""
    if not torch.cuda.is_available():
        return torch.cuda.amp.autocast(enabled=False), False, 'none'
    if dtype_choice == 'bf16':
        bf16_ok = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        if bf16_ok:
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16), False, 'bf16'
        else:
            print("[AMP] BF16 미지원 → FP16으로 폴백합니다.")
            return torch.cuda.amp.autocast(enabled=True), True, 'fp16'
    elif dtype_choice == 'fp16':
        return torch.cuda.amp.autocast(enabled=True), True, 'fp16'
    else:
        return torch.cuda.amp.autocast(enabled=False), False, 'none'

class EMA:
    def __init__(self, model, decay=0.9995):
        self.shadow = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        self.decay = decay
    @torch.no_grad()
    def update(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad:
                p.copy_(self.shadow[n])

# -----------------------------
# 10) 학습 루프
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
    amp_dtype: str = 'fp16'   # {'fp16','bf16'}
    patience: int = 8
    use_smape_pos: bool = False
    use_ema: bool = True

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
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr*2.0, epochs=cfg.epochs, steps_per_epoch=max(1,len(train_loader)),
        pct_start=0.1, div_factor=2.0, final_div_factor=100.0
    )
    amp_ctx, use_scaler, amp_mode = amp_context(cfg.amp_dtype)
    scaler = torch.cuda.amp.GradScaler('cuda', enabled=use_scaler)

    ema_n = EMA(nmodel) if cfg.use_ema else None
    ema_t = EMA(tmodel) if cfg.use_ema else None

    best_metric = float('inf')
    best_state = None
    bad = 0

    for epoch in range(cfg.epochs):
        nmodel.train(); tmodel.train()
        tr_loss = 0.0
        for batch in train_loader:
            (x28, xchan, y7, ymask, sid, mid, sw, scale, cid, tid, cal_stat) = [b.to(DEVICE) for b in batch]
            opt.zero_grad(set_to_none=True)
            with amp_ctx:
                n_logits, n_pos = nmodel(x28, sid, mid, cid, tid, cal_stat)
                t_logits, t_pos = tmodel(xchan, sid, mid, cid, tid, cal_stat)
                ln, _, _ = hurdle_loss(n_logits, n_pos, y7, ymask, sw, cfg.lambda_pos, cfg.use_smape_pos)
                lt, _, _ = hurdle_loss(t_logits, t_pos, y7, ymask, sw, cfg.lambda_pos, cfg.use_smape_pos)
                loss = 0.5*ln + 0.5*lt
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            tr_loss += loss.item() * x28.size(0)
            if cfg.use_ema:
                ema_n.update(nmodel); ema_t.update(tmodel)

        # 검증 (EMA 적용)
        nmodel.eval(); tmodel.eval()
        if cfg.use_ema:
            bak_n = {k:v.detach().cpu().clone() for k,v in nmodel.state_dict().items()}
            bak_t = {k:v.detach().cpu().clone() for k,v in tmodel.state_dict().items()}
            ema_n.apply_to(nmodel); ema_t.apply_to(tmodel)

        y_true_all, y_pred_all, store_ids_all, item_names_all = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                (x28, xchan, y7, ymask, sid, mid, sw, scale, cid, tid, cal_stat) = [b.to(DEVICE) for b in batch]
                with amp_ctx:
                    n_logits, n_pos = nmodel(x28, sid, mid, cid, tid, cal_stat)
                    t_logits, t_pos = tmodel(xchan, sid, mid, cid, tid, cal_stat)
                    nhat = torch.sigmoid(n_logits) * n_pos
                    that = torch.sigmoid(t_logits) * t_pos
                    hat = cfg.blend_alpha * nhat + (1.0 - cfg.blend_alpha) * that
                hat = hat * scale.view(-1,1)
                ytrue = y7 * scale.view(-1,1)
                y_true_all.append(ytrue.detach().cpu().numpy())
                y_pred_all.append(hat.detach().cpu().numpy())
                store_ids_all.append(sid.detach().cpu().numpy())
                item_names_all.append(np.arange(ytrue.size(0)))
        if cfg.use_ema:
            nmodel.load_state_dict(bak_n); tmodel.load_state_dict(bak_t)

        y_true_all = np.concatenate(y_true_all, axis=0) if len(y_true_all)>0 else np.zeros((0,OUT_LEN))
        y_pred_all = np.concatenate(y_pred_all, axis=0) if len(y_pred_all)>0 else np.zeros((0,OUT_LEN))
        store_ids_all = np.concatenate(store_ids_all, axis=0) if len(store_ids_all)>0 else np.zeros((0,))
        item_names_all = np.concatenate(item_names_all, axis=0) if len(item_names_all)>0 else np.zeros((0,))

        metric = weighted_smape_eval(y_true_all, y_pred_all, store_ids_all, item_names_all, id2store)

        if metric < best_metric - 1e-6:
            best_metric = metric
            bad = 0
            best_state = {
                "n": {k:v.cpu() for k,v in nmodel.state_dict().items()},
                "t": {k:v.cpu() for k,v in tmodel.state_dict().items()},
            }
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        nmodel.load_state_dict(best_state["n"])
        tmodel.load_state_dict(best_state["t"])
    return best_metric, nmodel, tmodel

# -----------------------------
# 11) Optuna 목적함수/모델 빌더
# -----------------------------
def build_models_from_trial(trial, store_vocab, menu_vocab, cat_vocab, type_vocab, in_ch, cal_stat_dim):
    # N-BEATS
    n_hidden = trial.suggest_categorical("n_hidden", [256, 384, 512, 640])
    n_blocks = trial.suggest_int("n_blocks", 2, 6)
    n_dropout = trial.suggest_float("n_dropout", 0.05, 0.2)
    emb_store = trial.suggest_categorical("emb_store", [32, 48, 64])
    use_menu = trial.suggest_categorical("use_menu", [False, True])
    emb_menu = trial.suggest_categorical("emb_menu", [0, 8, 16]) if use_menu else 0
    emb_cat  = trial.suggest_categorical("emb_cat", [8, 16, 24])
    emb_type = trial.suggest_categorical("emb_type",[4, 8, 12])
    n_cond = trial.suggest_categorical("n_cond", [64, 128, 192])

    n_model = NBEATSHurdle(
        hidden=n_hidden, blocks=n_blocks, dropout=n_dropout,
        store_vocab=store_vocab, menu_vocab=menu_vocab, cat_vocab=cat_vocab, type_vocab=type_vocab,
        emb_store=emb_store, emb_menu=emb_menu, emb_cat=emb_cat, emb_type=emb_type,
        cond_hidden=n_cond, cal_stat_dim=cal_stat_dim
    )

    # TCN
    t_channels = trial.suggest_categorical("t_channels", [128, 192, 256, 320])
    t_levels   = trial.suggest_int("t_levels", 3, 5)
    t_kernel   = trial.suggest_categorical("t_kernel", [3,5,7])
    t_dropout  = trial.suggest_float("t_dropout", 0.05, 0.2)
    t_emb_store = trial.suggest_categorical("t_emb_store", [16, 32, 48])
    t_use_menu  = trial.suggest_categorical("t_use_menu", [False, True])
    t_emb_menu  = trial.suggest_categorical("t_emb_menu", [0, 8, 16]) if t_use_menu else 0
    t_emb_cat   = trial.suggest_categorical("t_emb_cat", [8, 16, 24])
    t_emb_type  = trial.suggest_categorical("t_emb_type",[4, 8, 12])
    t_cond      = trial.suggest_categorical("t_cond", [32, 64, 96])

    t_model = TCNHurdle(
        in_ch=in_ch, channels=t_channels, levels=t_levels, kernel=t_kernel, dropout=t_dropout,
        store_vocab=store_vocab, menu_vocab=menu_vocab, cat_vocab=cat_vocab, type_vocab=type_vocab,
        emb_store=t_emb_store, emb_menu=t_emb_menu, emb_cat=t_emb_cat, emb_type=t_emb_type,
        cond_hidden=t_cond, cal_stat_dim=cal_stat_dim
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
        amp_dtype=trial.user_attrs.get("amp_dtype", "fp16"),
        patience=6, use_smape_pos=False, use_ema=True
    )
    return n_model, t_model, tcfg

def opt_objective(trial, folds, windows, id2store, store_vocab, menu_vocab, cat_vocab, type_vocab,
                  in_ch, cal_stat_dim, amp_dtype):
    trial.set_user_attr("amp_dtype", amp_dtype)
    nmodel, tmodel, cfg = build_models_from_trial(trial, store_vocab, menu_vocab, cat_vocab, type_vocab, in_ch, cal_stat_dim)
    scores = []
    for (tr_idx, val_idx) in folds:
        train_ws = [windows[i] for i in tr_idx]
        val_ws   = [windows[i] for i in val_idx]
        metric, n_best, t_best = train_one_fold(nmodel, tmodel, cfg, train_ws, val_ws, id2store)
        scores.append(metric)
        # fold 간 재초기화
        nmodel, tmodel, _ = build_models_from_trial(trial, store_vocab, menu_vocab, cat_vocab, type_vocab, in_ch, cal_stat_dim)
    score = float(np.mean(scores)) if len(scores)>0 else 1e9
    return score

# -----------------------------
# 12) Inference & Submission
# -----------------------------
@torch.no_grad()
def predict_one_file(nmodel, tmodel, df_test, store2id, menu2id, cat2id, type2id, chan_keys_order, amp_dtype, blend_alpha):
    df = df_test.copy()
    df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    preds_all = []

    amp_ctx, _, _ = amp_context(amp_dtype)

    for item, g in df.groupby('영업장명_메뉴명'):
        g = g.sort_values('영업일자')
        if len(g) < IN_LEN:
            continue
        x = g['매출수량'].to_numpy(dtype=np.float32)[-IN_LEN:]
        d = pd.to_datetime(g['영업일자']).to_list()[-IN_LEN:]
        cal = build_calendar_channels(d)
        sc = item_scale(x)

        chans = [x/sc] + [cal[k].astype(np.float32) for k in chan_keys_order]
        xch = torch.tensor(np.stack(chans, axis=0)[None,:], dtype=DTYPE, device=DEVICE)  # [1,C,28]
        x28 = torch.tensor((x/sc)[None,:], dtype=DTYPE, device=DEVICE)                   # [1,28]

        store = g['영업장명'].iloc[0]; menu = g['메뉴명'].iloc[0]
        sid = torch.tensor([store2id.get(store,0)], dtype=torch.long, device=DEVICE)
        mid = torch.tensor([menu2id.get(menu,0)], dtype=torch.long, device=DEVICE)
        cid = torch.tensor([cat2id.get(get_menu_category(menu),0)], dtype=torch.long, device=DEVICE)
        tid = torch.tensor([type2id.get(get_store_type(store),0)], dtype=torch.long, device=DEVICE)
        cal_stat = torch.tensor([[cal[k].mean() for k in chan_keys_order]], dtype=DTYPE, device=DEVICE)

        with amp_ctx:
            nlog, npos = nmodel(x28, sid, mid, cid, tid, cal_stat)
            tlog, tpos = tmodel(xch, sid, mid, cid, tid, cal_stat)
            nhat = torch.sigmoid(nlog) * npos
            that = torch.sigmoid(tlog) * tpos
            hat = blend_alpha * nhat + (1 - blend_alpha) * that
        y_hat = (hat * sc).squeeze(0).detach().cpu().numpy()
        preds_all.append((item, y_hat))

    out = []
    for item, y in preds_all:
        for k in range(OUT_LEN):
            out.append((item, k+1, int(np.rint(max(0, y[k])))))
    return pd.DataFrame(out, columns=['영업장명_메뉴명','day_k','pred'])

def build_submission(preds_per_test: List[pd.DataFrame], sample_sub_header: pd.Index) -> pd.DataFrame:
    rows = []
    for t_idx, pdf in enumerate(preds_per_test):
        for k in range(1, OUT_LEN+1):
            row_id = f"TEST_{t_idx:02d}+{k}일"
            r = pdf[pdf['day_k']==k][['영업장명_메뉴명','pred']]
            pivot = r.set_index('영업장명_메뉴명').T
            pivot.index = [row_id]
            rows.append(pivot)
    sub = pd.concat(rows, axis=0).reset_index().rename(columns={'index':'영업일자'})
    sub = sub.reindex(columns=sample_sub_header, fill_value=0)
    return sub

# -----------------------------
# 13) 메인
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--blend_alpha", type=float, default=0.6)
    ap.add_argument("--save_dir", type=str, default="./outputs_nbeats_tcn")
    ap.add_argument("--inference_only", action="store_true")
    ap.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16","bf16"])  # <=== 전환 플래그
    ap.add_argument("--num_workers", type=int, default=12)
    return ap.parse_args()

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) 데이터 로드/전처리
    train_df = pd.read_csv("./data/train/train.csv")
    train_df = clip_negatives_to_zero(train_df)
    train_df = split_store_menu(train_df)
    train_df['영업일자'] = pd.to_datetime(train_df['영업일자'])

    # 인코더
    store2id, menu2id, id2store, id2menu, cat2id, type2id = build_encoders(train_df)

    # 2) 슬라이딩 윈도우
    windows = build_windows(train_df, store2id, menu2id, cat2id, type2id)
    if len(windows)==0:
        raise RuntimeError("윈도우가 비어있습니다. 학습 데이터 확인!")

    # 3) CV 앵커/폴드
    anchors = make_strict_cv_anchors(windows)
    if len(anchors) == 0:
        raise RuntimeError("CV anchors가 생성되지 않았습니다. 데이터 길이/분포 확인.")
    folds = [split_fold(windows, a) for a in anchors if len(split_fold(windows, a)[1])>0]
    if len(folds) == 0:
        raise RuntimeError("유효한 CV fold가 없습니다.")

    # 4) 채널 수/키 순서 파악
    probe_ds = WinDataset(windows, id2store)
    chan_keys_order = probe_ds.chan_keys[:]  # 값 채널을 제외한 순서 리스트
    # in_ch = 1(value) + len(chan_keys)
    in_ch = 1 + len(chan_keys_order)
    cal_stat_dim = len(chan_keys_order)

    # 5) Optuna (선택)
    best_trial_params = None
    if not args.inference_only:
        def objective(trial):
            trial.set_user_attr("epochs", max(20, args.epochs//2))
            score = opt_objective(trial, folds, windows, id2store,
                                  len(store2id), len(menu2id), len(cat2id), len(type2id),
                                  in_ch, cal_stat_dim, args.amp_dtype)
            return score
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
        print("Best trial:", study.best_trial.number, "score:", study.best_value)
        best_trial_params = study.best_trial.params
        with open(os.path.join(args.save_dir,"best_params.json"), "w") as f:
            json.dump(best_trial_params, f, ensure_ascii=False, indent=2)
    else:
        best_trial_params = {
            "n_hidden":512,"n_blocks":4,"n_dropout":0.1,"emb_store":48,"use_menu":False,"emb_menu":0,"emb_cat":16,"emb_type":8,"n_cond":128,
            "t_channels":256,"t_levels":4,"t_kernel":5,"t_dropout":0.1,"t_emb_store":32,"t_use_menu":False,"t_emb_menu":0,"t_emb_cat":16,"t_emb_type":8,"t_cond":64,
            "lambda_pos":0.2,"blend_alpha":args.blend_alpha,"lr":1e-3,"wd":1e-4,"batch_size":2048
        }

    # 6) 최종 모델 구성 & 최근 fold 기준 재학습
    fixed = optuna.trial.FixedTrial(best_trial_params)
    nmodel, tmodel, tcfg = build_models_from_trial(fixed, len(store2id), len(menu2id), len(cat2id), len(type2id),
                                                   in_ch, cal_stat_dim)
    # 학습 설정 반영
    tcfg.epochs = args.epochs
    tcfg.blend_alpha = best_trial_params.get("blend_alpha", args.blend_alpha)
    tcfg.amp_dtype = args.amp_dtype
    tcfg.num_workers = args.num_workers

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
            "store2id": store2id, "menu2id": menu2id, "cat2id": cat2id, "type2id": type2id,
            "blend_alpha": tcfg.blend_alpha,
            "chan_keys_order": chan_keys_order,
            "amp_dtype": tcfg.amp_dtype
        }, f, ensure_ascii=False, indent=2)

    # 7) 추론 & 제출 생성
    print("추론 및 제출 CSV 생성...")
    n_best.eval(); t_best.eval()
    sample_sub = pd.read_csv("./data/sample_submission.csv", nrows=0)
    test_files = sorted(glob.glob("./data/test/TEST_*.csv"))
    preds_per_test = []
    for tf in test_files:
        df_t = pd.read_csv(tf)
        df_t = clip_negatives_to_zero(df_t)
        out_df = predict_one_file(n_best, t_best, df_t, store2id, menu2id, cat2id, type2id,
                                  chan_keys_order, tcfg.amp_dtype, tcfg.blend_alpha)
        preds_per_test.append(out_df)

    submission = build_submission(preds_per_test, sample_sub.columns)
    out_path = os.path.join(args.save_dir, "submission_nbeats_tcn_hurdle.csv")
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"제출 파일 저장 완료: {out_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
