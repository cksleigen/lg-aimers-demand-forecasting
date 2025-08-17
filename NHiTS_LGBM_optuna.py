# -*- coding: utf-8 -*-
import os
import math
import random
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import glob
import lightgbm as lgb

warnings.filterwarnings('ignore')

@dataclass
class EnhancedNHiTSConfig:
    train_csv: str = "LG_train.csv"
    test_glob: str = "test/*.csv"
    submission_template_csv: str = "sample_submission.csv"
    out_submission_csv: str = "./data/enhanced_nhits_submission.csv"
    date_col: str = "영업일자"
    item_col: str = "영업장명_메뉴명"
    target_col: str = "매출수량"
    in_len: int = 28
    out_len: int = 7
    train_end_date: str = "2024-06-15"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log1p: bool = True
    EPOCHS_FULL: int = 150
    BATCH_FULL: int = 512
    BASE_LR_FULL: float = 8e-4
    MAX_LR_FULL: float = 2e-3
    WD_FULL: float = 3e-4
    USE_OPTUNA: bool = True
    N_TRIALS: int = 50
    EPOCHS_TUNE: int = 50
    BATCH_TUNE: int = 256
    cv_fold_end_dates: Tuple[str, str, str] = ("2024-06-14", "2024-06-07", "2024-05-31")
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    hidden: int = 512
    n_blocks: int = 3
    n_layers: int = 2
    n_pool_kernel_size: List[int] = None
    pooling_mode: str = "MaxPool1d"
    interpolation_mode: str = "linear"
    dropout: float = 0.1
    stack_types: List[str] = None
    n_freq_downsample: List[int] = None
    eps_smape: float = 0.01
    zero_weight: float = 0.01
    hurdle_lambda: float = 0.15
    use_amp: bool = True
    use_compile: bool = False
    ema_decay: float = 0.9998
    store_weights: Dict[str, float] = None
    custom_holidays_list: List[str] = None

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

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sine_cosine_encoding(value: float, max_val: float):
    return math.sin(2 * math.pi * value / max_val), math.cos(2 * math.pi * value / max_val)

def get_menu_category(menu_name: str) -> str:
    menu_lower = menu_name.lower()
    if any(word in menu_lower for word in ['막걸리','소주','맥주','와인','참이슬','처음처럼','카스','하이네켄','버드와이저','스텔라']):
        return 'alcohol'
    elif any(word in menu_lower for word in ['찌개','탕','국밥','라면','해장국','갈비탕']):
        return 'hot_food'
    elif any(word in menu_lower for word in ['삼겹','갈비','목살','bbq','구이','불고기']):
        return 'bbq'
    elif any(word in menu_lower for word in ['아이스크림','식혜','콜라','스프라이트','에이드']):
        return 'dessert_drink'
    elif any(word in menu_lower for word in ['아메리카노','라떼','커피']):
        return 'coffee'
    elif any(word in menu_lower for word in ['냉면','파스타','스파게티','면','우동']):
        return 'noodles'
    elif any(word in menu_lower for word in ['비빔밥','볶음밥','공깃밥','정식']):
        return 'rice'
    else:
        return 'others'

def get_store_type(store_name: str) -> str:
    if store_name == "느티나무 셀프BBQ":
        return 'outdoor'
    elif store_name in ["라그로타","미라시아"]:
        return 'fine_dining'
    elif store_name == "담하":
        return 'traditional'
    elif store_name == "연회장":
        return 'event'
    elif store_name in ["카페테리아","포레스트릿","화담숲카페"]:
        return 'casual'
    else:
        return 'specialty'

def build_enhanced_features(dates: List[pd.Timestamp], holidays_set: set) -> pd.DataFrame:
    df = pd.DataFrame({"date": dates})
    df["dow"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["week"] = df["date"].apply(lambda d: int(d.isocalendar().week))
    df["quarter"] = df["date"].dt.quarter
    df["is_friday"] = (df["dow"] == 4).astype(int)
    df["is_saturday"] = (df["dow"] == 5).astype(int)
    df["is_sunday"] = (df["dow"] == 6).astype(int)
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
    df["is_holiday"] = df["date"].isin(holidays_set).astype(int)
    df["is_off"] = ((df["is_weekend"]==1) | (df["is_holiday"]==1)).astype(int)
    df["tomorrow"] = df["date"] + pd.Timedelta(days=1)
    df["yesterday"] = df["date"] - pd.Timedelta(days=1)
    df["is_before_holiday"] = df["tomorrow"].isin(holidays_set).astype(int)
    df["is_after_holiday"] = df["yesterday"].isin(holidays_set).astype(int)
    df["is_month_start"] = (df["day"] <= 3).astype(int)
    df["is_month_end"] = (df["day"] >= 28).astype(int)
    df["is_spring"] = df["month"].isin([4,5,6]).astype(int)
    df["is_summer"] = df["month"].isin([7,8]).astype(int)
    df["is_autumn"] = df["month"].isin([9,10,11]).astype(int)
    df["is_winter"] = df["month"].isin([12,1,2,3]).astype(int)
    df["is_summer_vacation"] = df["month"].isin([7,8]).astype(int)
    df["is_winter_vacation"] = df["month"].isin([12,1,2]).astype(int)
    df[["month_sin","month_cos"]] = df["month"].apply(lambda m: pd.Series(sine_cosine_encoding(m, 12)))
    df[["dow_sin","dow_cos"]] = df["dow"].apply(lambda d: pd.Series(sine_cosine_encoding(d, 7)))
    df[["doy_sin","doy_cos"]] = df["date"].apply(lambda d: pd.Series(sine_cosine_encoding(d.dayofyear, 365)))
    df[["week_sin","week_cos"]] = df["week"].apply(lambda w: pd.Series(sine_cosine_encoding(w, 52)))
    df[["quarter_sin","quarter_cos"]] = df["quarter"].apply(lambda q: pd.Series(sine_cosine_encoding(q, 4)))
    return df.drop(columns=["tomorrow","yesterday"])

def parse_store_name(item_full: str) -> str:
    return item_full.split("_")[0]

def parse_menu_name(item_full: str) -> str:
    return item_full.split("_", 1)[1] if "_" in item_full else item_full

class EnhancedNHiTSDataset(Dataset):
    def __init__(self, cfg: EnhancedNHiTSConfig, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df.copy()
        self.df[cfg.date_col] = pd.to_datetime(self.df[cfg.date_col])
        self.df[cfg.target_col] = self.df[cfg.target_col].clip(lower=0)
        pivot = self.df.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index()
        self.items = list(pivot.columns)
        self.dates = list(pivot.index)
        self.values = pivot.fillna(0.0).values.astype(np.float32)
        holidays_set = set(pd.to_datetime(cfg.custom_holidays_list))
        caldf = build_enhanced_features(self.dates, holidays_set)
        self.cal_feats = caldf.drop(columns=["date"]).values.astype(np.float32)
        self.cal_feat_names = [c for c in caldf.columns if c != "date"]
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
        sw = cfg.store_weights
        self.sample_weights = np.array([sw.get(parse_store_name(it), 1.0) for it in self.items], dtype=np.float32)
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
        cat_idx = self.item_cat_idx[j]
        type_idx = self.item_type_idx[j]
        sample_w = self.sample_weights[j]
        zero_mask = (y == 0).astype(np.float32)
        pos_mask = (y > 0).astype(np.float32)
        return {
            "x": torch.from_numpy(x_in).float(),
            "y": torch.from_numpy(y_out).float(),
            "past_cal": torch.from_numpy(past_cal).float(),
            "fut_cal": torch.from_numpy(fut_cal).float(),
            "store_idx": torch.tensor(store_idx, dtype=torch.long),
            "cat_idx": torch.tensor(cat_idx, dtype=torch.long),
            "type_idx": torch.tensor(type_idx, dtype=torch.long),
            "sample_w": torch.tensor(sample_w, dtype=torch.float32),
            "zero_mask": torch.from_numpy(zero_mask).float(),
            "pos_mask": torch.from_numpy(pos_mask).float(),
        }

class NHiTSBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, 
                 n_layers: int, dropout: float, pooling_mode: str = "MaxPool1d",
                 n_pool_kernel_size: int = 2, interpolation_mode: str = "linear"):
        super().__init__()
        self.pooling_mode = pooling_mode
        self.interpolation_mode = interpolation_mode
        self.n_pool_kernel_size = n_pool_kernel_size
        self.input_size = input_size
        self.output_size = output_size
        if pooling_mode == "MaxPool1d":
            self.pooling_layer = nn.MaxPool1d(kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True)
        elif pooling_mode == "AvgPool1d":
            self.pooling_layer = nn.AvgPool1d(kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True)
        pooled_size = math.ceil(input_size / n_pool_kernel_size)
        layers = []
        layers.append(nn.Linear(pooled_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = x.unsqueeze(1)
        x_pooled = self.pooling_layer(x).squeeze(1)
        h = self.mlp(x_pooled)
        output = self.output_layer(h)
        return output

class EnhancedNHiTSModel(nn.Module):
    def __init__(self, in_len: int, out_len: int, cal_dim: int, n_stores: int,
                 n_categories: int, n_types: int, cfg: EnhancedNHiTSConfig):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        if cfg.n_pool_kernel_size is None or len(cfg.n_pool_kernel_size) < cfg.n_blocks:
            base_kernels = [2,2,1]
            cfg.n_pool_kernel_size = [base_kernels[i % len(base_kernels)] for i in range(cfg.n_blocks)]
        if cfg.stack_types is None or len(cfg.stack_types) < cfg.n_blocks:
            base_types = ["identity","identity","trend"]
            cfg.stack_types = [base_types[i % len(base_types)] for i in range(cfg.n_blocks)]
        if cfg.n_freq_downsample is None or len(cfg.n_freq_downsample) < cfg.n_blocks:
            base_downsample = [2,1,1]
            cfg.n_freq_downsample = [base_downsample[i % len(base_downsample)] for i in range(cfg.n_blocks)]
        self.store_emb = nn.Embedding(n_stores, 64)
        self.cat_emb = nn.Embedding(n_categories, 32)
        self.type_emb = nn.Embedding(n_types, 16)
        self.cal_proj = nn.Linear(cal_dim, 128)
        self.blocks = nn.ModuleList()
        for i in range(cfg.n_blocks):
            block = NHiTSBlock(
                input_size=in_len,
                output_size=out_len,
                hidden_size=cfg.hidden,
                n_layers=cfg.n_layers,
                dropout=cfg.dropout,
                pooling_mode=cfg.pooling_mode,
                n_pool_kernel_size=cfg.n_pool_kernel_size[i],
                interpolation_mode=cfg.interpolation_mode
            )
            self.blocks.append(block)
        self.basis_weights = nn.ParameterList([
            nn.Parameter(torch.randn(out_len, max(1, out_len // cfg.n_freq_downsample[i])))
            for i in range(cfg.n_blocks)
        ])
        meta_dim = 64 + 32 + 16 + 128
        self.meta_integration = nn.Sequential(
            nn.Linear(meta_dim, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, out_len)
        )
        self.prob_head = nn.Sequential(
            nn.Linear(meta_dim + 1, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden // 2, out_len)
        )
        self.stack_types = cfg.stack_types
    def forward(self, x, past_cal, fut_cal, store_idx, cat_idx, type_idx):
        store_emb = self.store_emb(store_idx)
        cat_emb = self.cat_emb(cat_idx)
        type_emb = self.type_emb(type_idx)
        cal_all = torch.cat([past_cal, fut_cal], dim=1).mean(dim=1)
        cal_emb = self.cal_proj(cal_all)
        meta_feat = torch.cat([store_emb, cat_emb, type_emb, cal_emb], dim=-1)
        outputs = []
        for i, block in enumerate(self.blocks):
            block_output = block(x)
            if self.stack_types[i] == "trend":
                basis = self.basis_weights[i]
                block_output = torch.matmul(block_output, basis.T)
                block_output = F.interpolate(block_output.unsqueeze(1), size=self.out_len, mode='linear', align_corners=False).squeeze(1)
            outputs.append(block_output)
        nhits_output = torch.stack(outputs, dim=0).sum(dim=0)
        meta_adjustment = self.meta_integration(meta_feat)
        value_pred = nhits_output + meta_adjustment
        prob_feat = torch.cat([meta_feat, x.mean(dim=1, keepdim=True)], dim=-1)
        prob_logits = self.prob_head(prob_feat)
        return value_pred, prob_logits

class UltraEnhancedHurdleLoss(nn.Module):
    def __init__(self, eps: float = 0.01, zero_weight: float = 0.01, lambda_bce: float = 0.15):
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
        ultra_eps = torch.where(yt_val < 0.5, self.eps * 0.2, torch.where(yt_val < 2.0, self.eps * 0.5, self.eps))
        denom = (torch.abs(yp_val) + torch.abs(yt_val)).clamp_min(ultra_eps)
        smape_pos = 2.0 * torch.abs(yp_val - yt_val) / denom
        smape_pos = smape_pos * pos_mask
        p = torch.sigmoid(p_logits)
        y_hat = p * yp_val
        denom2 = (torch.abs(y_hat) + torch.abs(yt_val)).clamp_min(ultra_eps)
        smape_all = 2.0 * torch.abs(y_hat - yt_val) / denom2
        ultra_zero_weight = torch.where(yt_val < 0.01, torch.full_like(yt_val, self.zero_weight * 0.1),
                              torch.where(yt_val < 0.1, torch.full_like(yt_val, self.zero_weight * 0.3),
                              torch.where(yt_val < 1.0, torch.full_like(yt_val, self.zero_weight * 0.6), torch.ones_like(yt_val))))
        smape_all = smape_all * ultra_zero_weight
        bce_s = bce.mean(dim=1)
        pos_s = smape_pos.mean(dim=1)
        all_s = smape_all.mean(dim=1)
        sample_loss = self.lambda_bce * bce_s + 0.4 * pos_s + 0.6 * all_s
        sw = sample_w.view(-1)
        wsum = sw.sum().clamp_min(1e-8)
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

class EnhancedNHiTSTrainer:
    def __init__(self, cfg: EnhancedNHiTSConfig, dataset: EnhancedNHiTSDataset, epochs: int,
                 batch_size: int, base_lr: float, max_lr: float, weight_decay: float):
        self.cfg = cfg
        self.dataset = dataset
        self.device = torch.device(cfg.device)
        cal_dim = dataset.cal_feats.shape[1]
        model = EnhancedNHiTSModel(cfg.in_len, cfg.out_len, cal_dim, dataset.n_stores, dataset.n_categories, dataset.n_types, cfg).to(self.device)
        if cfg.use_compile and torch.cuda.is_available():
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception as e:
                print("torch.compile 실패:", e)
        self.model = model
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.criterion = UltraEnhancedHurdleLoss(cfg.eps_smape, cfg.zero_weight, cfg.hurdle_lambda)
        self.ema = EMA(self.model, decay=cfg.ema_decay)
        self.epochs = epochs
        self.batch_size = batch_size
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
                store_idx = batch["store_idx"].to(self.device)
                cat_idx = batch["cat_idx"].to(self.device)
                type_idx = batch["type_idx"].to(self.device)
                sample_w = batch["sample_w"].to(self.device)
                pos_mask = batch["pos_mask"].to(self.device)
                v_pred, p_logits = self.model(x, past_cal, fut_cal, store_idx, cat_idx, type_idx)
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
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=self.max_lr, epochs=self.epochs, steps_per_epoch=len(train_loader), pct_start=0.05, div_factor=self.max_lr / self.base_lr)
        best_val = float("inf")
        best_state = None
        patience = 25
        no_improve = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (self.cfg.use_amp and torch.cuda.is_available()) else torch.cuda.amp.autocast(enabled=False)
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
                with amp_ctx:
                    v_pred, p_logits = self.model(x, past_cal, fut_cal, store_idx, cat_idx, type_idx)
                    loss, _, _ = self.criterion(v_pred, p_logits, y, pos_mask, sample_w)
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optim.step()
                self.sched.step()
                self.ema.update(self.model)
            val_loss = self.evaluate(val_loader, use_ema=True)
            print(f"[Epoch {epoch:03d}] val_loss: {val_loss:.5f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.model, best_val

def make_val_mask_by_week(dataset: EnhancedNHiTSDataset, end_date_str: str) -> np.ndarray:
    end_date = pd.to_datetime(end_date_str)
    start_date = end_date - pd.Timedelta(days=6)
    ted = dataset.target_end_dates
    return (ted >= start_date) & (ted <= end_date)

def evaluate_cfg_rolling(cfg: EnhancedNHiTSConfig, epochs: int, batch_size: int,
                        base_lr: float, max_lr: float, weight_decay: float) -> float:
    train_df = pd.read_csv(cfg.train_csv)
    ds = EnhancedNHiTSDataset(cfg, train_df)
    fold_vals = []
    for end_date_str in cfg.cv_fold_end_dates:
        trainer = EnhancedNHiTSTrainer(cfg, ds, epochs, batch_size, base_lr, max_lr, weight_decay)
        mask_val = make_val_mask_by_week(ds, end_date_str)
        train_loader, val_loader = trainer.make_loaders_from_mask(mask_val)
        _, best_val = trainer.train_with_loaders(train_loader, val_loader)
        fold_vals.append(best_val)
        print(f"[CV] fold end={end_date_str} val={best_val:.5f}")
    cv_mean = float(np.mean(fold_vals))
    print(f"[CV] mean val={cv_mean:.5f}")
    return cv_mean

def run_optuna(cfg: EnhancedNHiTSConfig):
    import optuna
    def objective(trial: optuna.trial.Trial):
        cfg.hidden = trial.suggest_categorical("hidden", [384, 512, 640])
        cfg.n_blocks = trial.suggest_categorical("n_blocks", [2, 3, 4])
        cfg.n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
        cfg.dropout = trial.suggest_float("dropout", 0.05, 0.15)
        cfg.pooling_mode = trial.suggest_categorical("pooling_mode", ["MaxPool1d", "AvgPool1d"])
        cfg.eps_smape = trial.suggest_categorical("eps_smape", [0.005, 0.01, 0.02])
        cfg.zero_weight = trial.suggest_categorical("zero_weight", [0.005, 0.01, 0.02])
        cfg.hurdle_lambda = trial.suggest_categorical("hurdle_lambda", [0.1, 0.15, 0.2])
        epochs = cfg.EPOCHS_TUNE
        batch_size = cfg.BATCH_TUNE
        base_lr = trial.suggest_float("base_lr", 5e-4, 1.5e-3, log=True)
        max_lr = trial.suggest_float("max_lr", 1e-3, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        val = evaluate_cfg_rolling(cfg, epochs, batch_size, base_lr, max_lr, weight_decay)
        return val
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg.N_TRIALS)
    print("[Optuna] Best value:", study.best_value)
    print("[Optuna] Best params:", study.best_trial.params)
    best = study.best_trial.params
    cfg.hidden = best.get("hidden", cfg.hidden)
    cfg.n_blocks = best.get("n_blocks", cfg.n_blocks)
    cfg.n_layers = best.get("n_layers", cfg.n_layers)
    cfg.dropout = best.get("dropout", cfg.dropout)
    cfg.pooling_mode = best.get("pooling_mode", cfg.pooling_mode)
    cfg.eps_smape = best.get("eps_smape", cfg.eps_smape)
    cfg.zero_weight = best.get("zero_weight", cfg.zero_weight)
    cfg.hurdle_lambda = best.get("hurdle_lambda", cfg.hurdle_lambda)

@torch.no_grad()
def predict_one_file(cfg: EnhancedNHiTSConfig, model: EnhancedNHiTSModel, test_df: pd.DataFrame,
                    store2idx: Dict, cat2idx: Dict, type2idx: Dict) -> pd.DataFrame:
    device = torch.device(cfg.device)
    tdf = test_df.copy()
    tdf[cfg.date_col] = pd.to_datetime(tdf[cfg.date_col])
    tdf[cfg.target_col] = tdf[cfg.target_col].clip(lower=0)
    pivot = tdf.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index().fillna(0.0)
    items = list(pivot.columns)
    dates = list(pivot.index)
    values = pivot.values.astype(np.float32)
    last_date = dates[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, cfg.out_len + 1)]
    holidays_set = set(pd.to_datetime(cfg.custom_holidays_list))
    past_cal = build_enhanced_features(dates[-cfg.in_len:], holidays_set).drop(columns=["date"]).values.astype(np.float32)
    fut_cal = build_enhanced_features(future_dates, holidays_set).drop(columns=["date"]).values.astype(np.float32)
    B = len(items)
    Lx = cfg.in_len
    x = values[-Lx:, :].T
    if cfg.log1p:
        x = np.log1p(x)
    x = torch.from_numpy(x).float().to(device)
    past_cal_b = torch.from_numpy(np.repeat(past_cal[None, :, :], B, axis=0)).float().to(device)
    fut_cal_b = torch.from_numpy(np.repeat(fut_cal[None, :, :], B, axis=0)).float().to(device)
    stores = [parse_store_name(it) for it in items]
    menus = [parse_menu_name(it) for it in items]
    store_idx = torch.tensor([store2idx.get(s, 0) for s in stores], dtype=torch.long, device=device)
    cat_idx = torch.tensor([cat2idx.get(get_menu_category(m), 0) for m in menus], dtype=torch.long, device=device)
    type_idx = torch.tensor([type2idx.get(get_store_type(s), 0) for s in stores], dtype=torch.long, device=device)
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (cfg.use_amp and torch.cuda.is_available()) else torch.cuda.amp.autocast(enabled=False)
    model.eval()
    with amp_ctx:
        v_pred, p_logits = model(x, past_cal_b, fut_cal_b, store_idx, cat_idx, type_idx)
        y_val = torch.expm1(v_pred).clamp_min(0.0)
        y_prob = torch.sigmoid(p_logits)
        y_hat = (y_prob * y_val).clamp_min(0.0).cpu().numpy()
    return pd.DataFrame(y_hat, index=items, columns=[f"D+{i}" for i in range(1, cfg.out_len+1)]).T

# --------------------------
# LightGBM: 학습/예측/블렌딩
# --------------------------
def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def _extract_sample(dataset: EnhancedNHiTSDataset, idx: int):
    sample = dataset[idx]
    x = sample["x"].numpy()
    y = np.expm1(sample["y"].numpy())
    fut_cal = sample["fut_cal"].numpy()
    store_idx = int(sample["store_idx"].numpy())
    cat_idx = int(sample["cat_idx"].numpy())
    type_idx = int(sample["type_idx"].numpy())
    return x, y, fut_cal, store_idx, cat_idx, type_idx

def _build_lgb_frames(dataset: EnhancedNHiTSDataset, mask_val: np.ndarray):
    idx_all = np.arange(len(dataset))
    tr_idx = idx_all[~mask_val]
    va_idx = idx_all[mask_val]
    Lx = dataset.cfg.in_len
    cal_dim = dataset.cal_feats.shape[1]
    def make_frame(indices):
        Xh, yh = [], []
        for h in range(dataset.cfg.out_len):
            X_list = []
            y_list = []
            for i in indices:
                x, y, fut_cal, s, c, t = _extract_sample(dataset, i)
                feats = np.concatenate([x, fut_cal[h], [s, c, t]], axis=0)
                X_list.append(feats)
                y_list.append(y[h])
            Xh.append(np.asarray(X_list, dtype=np.float32))
            yh.append(np.asarray(y_list, dtype=np.float32))
        return Xh, yh
    X_tr, y_tr = make_frame(tr_idx)
    X_va, y_va = make_frame(va_idx)
    feat_names = [f"x_{k}" for k in range(Lx)] + [f"cal_{k}" for k in range(cal_dim)] + ["store_idx","cat_idx","type_idx"]
    return X_tr, y_tr, X_va, y_va, feat_names

def train_lgbm_oof(dataset: EnhancedNHiTSDataset, mask_val: np.ndarray):
    X_tr, y_tr, X_va, y_va, feat_names = _build_lgb_frames(dataset, mask_val)
    models = []
    va_preds = []
    params = dict(objective="regression", metric="l1", learning_rate=0.05, num_leaves=63, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=20, max_depth=-1, verbose=-1)
    for h in range(dataset.cfg.out_len):
        dtr = lgb.Dataset(X_tr[h], label=y_tr[h], feature_name=feat_names, free_raw_data=False)
        dva = lgb.Dataset(X_va[h], label=y_va[h], reference=dtr, feature_name=feat_names, free_raw_data=False)
        model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dtr, dva], valid_names=["tr","va"], early_stopping_rounds=100, verbose_eval=False)
        models.append(model)
        va_pred = model.predict(X_va[h], num_iteration=model.best_iteration)
        va_pred = np.clip(va_pred, 0, None)
        va_preds.append(va_pred)
    va_preds = np.vstack(va_preds)  # [7, n_val]
    y_va_true = np.vstack(y_va)     # [7, n_val]
    return models, va_preds, y_va_true

def predict_lgbm_one_file(cfg: EnhancedNHiTSConfig, models: List[lgb.Booster], test_df: pd.DataFrame,
                          store2idx: Dict, cat2idx: Dict, type2idx: Dict) -> pd.DataFrame:
    tdf = test_df.copy()
    tdf[cfg.date_col] = pd.to_datetime(tdf[cfg.date_col])
    tdf[cfg.target_col] = tdf[cfg.target_col].clip(lower=0)
    pivot = tdf.pivot(index=cfg.date_col, columns=cfg.item_col, values=cfg.target_col).sort_index().fillna(0.0)
    items = list(pivot.columns)
    dates = list(pivot.index)
    values = pivot.values.astype(np.float32)
    last_date = dates[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, cfg.out_len + 1)]
    holidays_set = set(pd.to_datetime(cfg.custom_holidays_list))
    past_cal = build_enhanced_features(dates[-cfg.in_len:], holidays_set).drop(columns=["date"]).values.astype(np.float32)
    fut_cal = build_enhanced_features(future_dates, holidays_set).drop(columns=["date"]).values.astype(np.float32)
    Lx = cfg.in_len
    cal_dim = past_cal.shape[1]
    X_list_per_h = [[] for _ in range(cfg.out_len)]
    for it in items:
        store = parse_store_name(it)
        menu = parse_menu_name(it)
        s = store2idx.get(store, 0)
        c = cat2idx.get(get_menu_category(menu), 0)
        t = type2idx.get(get_store_type(store), 0)
        x = values[-Lx:, items.index(it)].astype(np.float32)
        for h in range(cfg.out_len):
            feats = np.concatenate([x, fut_cal[h], [s, c, t]], axis=0)
            X_list_per_h[h].append(feats)
    preds = []
    for h in range(cfg.out_len):
        Xh = np.asarray(X_list_per_h[h], dtype=np.float32)
        ph = models[h].predict(Xh, num_iteration=models[h].best_iteration)
        ph = np.clip(ph, 0, None)
        preds.append(ph)
    preds = np.array(preds)  # [7, B]
    return pd.DataFrame(preds, columns=items, index=[f"D+{i}" for i in range(1, cfg.out_len+1)])

def choose_blend_weight(y_true_val: np.ndarray, nh_val: np.ndarray, gb_val: np.ndarray) -> float:
    best_w = 0.5
    best_s = 1e9
    for w in np.linspace(0, 1, 21):
        mix = w * nh_val + (1 - w) * gb_val
        s = _smape(y_true_val.flatten(), mix.flatten())
        if s < best_s:
            best_s = s
            best_w = float(w)
    print(f"[Blend] best_w={best_w:.2f} smape={best_s:.6f}")
    return best_w

if __name__ == "__main__":
    cfg = EnhancedNHiTSConfig()
    if cfg.store_weights is None:
        cfg.store_weights = DEFAULT_STORE_WEIGHTS
    if cfg.custom_holidays_list is None:
        cfg.custom_holidays_list = DEFAULT_CUSTOM_HOLIDAYS
    set_seed(cfg.seed)
    print("Enhanced N-HiTS + LGBM Late Blending 시작")
    if cfg.USE_OPTUNA:
        print("Optuna 튜닝...")
        run_optuna(cfg)
        print("튜닝 완료")
    print("N-HiTS 학습...")
    train_df = pd.read_csv(cfg.train_csv)
    ds = EnhancedNHiTSDataset(cfg, train_df)
    trainer = EnhancedNHiTSTrainer(cfg, ds, cfg.EPOCHS_FULL, cfg.BATCH_FULL, cfg.BASE_LR_FULL, cfg.MAX_LR_FULL, cfg.WD_FULL)
    mask_val = make_val_mask_by_week(ds, cfg.cv_fold_end_dates[0])
    train_loader, val_loader = trainer.make_loaders_from_mask(mask_val)
    model, final_val_loss = trainer.train_with_loaders(train_loader, val_loader)
    print(f"N-HiTS val_loss: {final_val_loss:.6f}")
    print("LGBM 학습(OoF val split)...")
    lgb_models, lgb_val_preds, y_val_true = train_lgbm_oof(ds, mask_val)
    print("N-HiTS 검증 예측 생성...")
    @torch.no_grad()
    def predict_nhits_on_indices(ds: EnhancedNHiTSDataset, idxs: np.ndarray) -> np.ndarray:
        device = torch.device(cfg.device)
        cal_dim = ds.cal_feats.shape[1]
        model.eval()
        vals = []
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (cfg.use_amp and torch.cuda.is_available()) else torch.cuda.amp.autocast(enabled=False)
        with amp_ctx:
            for i in idxs:
                s = ds[i]
                x = s["x"].unsqueeze(0).to(device)
                past_cal = s["past_cal"].unsqueeze(0).to(device)
                fut_cal = s["fut_cal"].unsqueeze(0).to(device)
                store_idx = s["store_idx"].unsqueeze(0).to(device)
                cat_idx = s["cat_idx"].unsqueeze(0).to(device)
                type_idx = s["type_idx"].unsqueeze(0).to(device)
                v_pred, p_logits = model(x, past_cal, fut_cal, store_idx, cat_idx, type_idx)
                y_val = torch.expm1(v_pred).clamp_min(0.0)
                y_prob = torch.sigmoid(p_logits)
                y_hat = (y_prob * y_val).clamp_min(0.0).cpu().numpy()[0]
                vals.append(y_hat)
        return np.vstack(vals).T
    val_indices = np.arange(len(ds))[mask_val]
    nhits_val_preds = predict_nhits_on_indices(ds, val_indices)
    print("블렌딩 가중치 선택...")
    w = choose_blend_weight(y_val_true, nhits_val_preds, lgb_val_preds)
    print("테스트 예측...")
    test_files = sorted(glob.glob(cfg.test_glob))
    sub_template = pd.read_csv(cfg.submission_template_csv)
    all_preds = []
    for test_idx, test_file in enumerate(test_files):
        print(f"- {test_file}")
        tdf = pd.read_csv(test_file)
        nh_block = predict_one_file(cfg, model, tdf, ds.store2idx, ds.cat2idx, ds.type2idx)
        gb_block = predict_lgbm_one_file(cfg, lgb_models, tdf, ds.store2idx, ds.cat2idx, ds.type2idx)
        mix_block = w * nh_block.values + (1 - w) * gb_block.values
        mix_block = np.clip(mix_block, 0, None)
        submit_block = pd.DataFrame(mix_block, index=nh_block.index, columns=nh_block.columns)
        submit_block.index = [f"TEST_{test_idx:02d}+{k}일" for k in range(1, cfg.out_len+1)]
        all_preds.append(submit_block)
    final_submit = pd.concat(all_preds, axis=0)
    final_submit.reset_index(inplace=True)
    final_submit.rename(columns={"index": "영업일자"}, inplace=True)
    final_submit = final_submit.reindex(columns=sub_template.columns, fill_value=0)
    num_cols = [c for c in final_submit.columns if c != "영업일자"]
    for col in num_cols:
        Q95 = final_submit[col].quantile(0.95)
        final_submit[col] = np.where(final_submit[col] > Q95 * 2.0, Q95 * 1.5, final_submit[col])
    final_submit[num_cols] = np.rint(np.clip(final_submit[num_cols].values, a_min=0, a_max=None)).astype(int)
    os.makedirs(os.path.dirname(cfg.out_submission_csv), exist_ok=True)
    final_submit.to_csv(cfg.out_submission_csv, index=False, encoding="utf-8-sig")
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg.__dict__,
        "store2idx": ds.store2idx,
        "cat2idx": ds.cat2idx,
        "type2idx": ds.type2idx,
        "final_val_loss": final_val_loss,
        "blend_w": w
    }, "./data/enhanced_nhits_model.pth")
    print(f"완료: {cfg.out_submission_csv} (blend_w={w:.2f})")
