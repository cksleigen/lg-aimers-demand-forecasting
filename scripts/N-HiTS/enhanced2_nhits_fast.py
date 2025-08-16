"""
### ① **모델 구성 모델 구조 (Enhanced N-HiTS)**

* 기본틀은 **N-HiTS 블록 기반 멀티스케일 모델**
  → 여러 개의 block이 각각 다른 pooling 스케일로 시계열을 처리하고 합산

* 각 block 입력 시, 단순 x뿐 아니라
  ⇒ `concat([pooled(x), meta_feat])`
  ↳ 이러한 구조를 **conditioning 구조**라고 부르며
  ↳ 결과적으로 **매장/메뉴/캘린더 특성을 시계열 블록 내부 학습 과정에 직접 반영**

* 각 block output 들을 더해 base prediction을 만들고,
  다시 한 번 meta-based 보정(head)을 통과 → 최종 value 예측

* activation은 **SiLU**, 각 Linear 뒤에 **LayerNorm + Dropout → 안정된 학습**

---

### ② **meta feature (= conditioning 입력)**

포함되는 내용:

* store embedding (64 dim)
* cluster embedding (32 dim)  ← 사용자가 정의한 메뉴 클러스터
* calendar feature projection (128 dim: 과거+미래 평균 캘린더)

총 224D → block마다 conditioning vector로 들어감
\=> **“매장이 어디냐, 메뉴가 어떤 계열이냐, 앞으로 공휴일이냐”에 맞춘 시계열 모양 학습 가능**

---

### ③ **Loss (UltraEnhancedHurdleLoss)**

**Hurdle 방식 → “0 발생 확률”과 “0 이상일 때의 값”을 동시에 예측**

* Value head → log1p값 예측 (회귀)
* Prob head → 0이 될 확률(BCE Logits)

Loss =

> `λ*BCE(prob head)` + `0.4*positive SMAPE` + `0.6*all SMAPE`

* **SMAPE 기반**이므로 Kaggle 등에서 쓰는 SMAPE metric 그대로 최적화
* **ultra\_eps 세팅**으로 극소량 예측 (0.1이하)도 민감하게 학습
* **매장별 sample\_weight 적용 가능**

---

### ④ **Trainer (학습 loop)**

* Optimizer ⇒ `AdamW`
* Scheduler ⇒ `CosineAnnealing + 5% warmup`
* EMA 적용 ⇒ 매 step마다 shadow weight 추적해서 evaluation 시 사용
* AMP 적용 (bfloat16) ⇒ A6000 빠른 학습
* Gradient Clipping (0.5)

EarlyStopping: 25epoch patience
---

### ✅ 이 구조가 주는 특징

| 부분           | 효과                    |
| ------------ | --------------------- |
| Conditioning | 메뉴·매장 특성 반영된 예측 패턴    |
| Hurdle Loss  | 0 값 비중이 큰 시계열에 최적화    |
| Cosine LR    | 안정된 수렴, overfit 감소    |
| EMA          | 더 부드러운 generalization |

"""
# -*- coding: utf-8 -*-
"""
Enhanced N-HiTS Model (Cluster + Conditioning + SiLU + LayerNorm + CosineAnnealingWarmup)
"""
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

warnings.filterwarnings('ignore')


# =============================================================
# Enhanced N-HiTS Config (기존 cfg는 그대로 유지, 하이퍼파라미터 변경금지)
# =============================================================
@dataclass
class EnhancedNHiTSConfig:
    train_csv: str = "./data/train/train.csv"
    test_glob: str = "./data/test/*.csv"
    submission_template_csv: str = "./data/sample_submission.csv"
    out_submission_csv: str = "./data/enhanced2_nhits_submission_epochs150.csv"

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
    BATCH_FULL: int = 1024
    BASE_LR_FULL: float = 1e-3
    MAX_LR_FULL: float = 2.5e-3
    WD_FULL: float = 3e-4

    USE_OPTUNA: bool = True
    N_TRIALS: int = 25
    EPOCHS_TUNE: int = 35
    BATCH_TUNE: int = 512

    cv_fold_end_dates: Tuple[str, str, str] = ("2024-06-14", "2024-06-07", "2024-05-31")

    num_workers: int = 6
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

    # optimizer / scheduler / activation 설정 추가
    opt_type: str = "adamw"            # ["adamw", "adam", "sgd"]
    scheduler_type: str = "cosine"     # ["cosine", "onecycle", "plateau"]
    warmup_pct: float = 0.05
    activation: str = "silu"           # ["silu", "relu", "gelu"]
    weight_decay: float = 3e-4


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

# 기본값들
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

# =============================================================
# Feature Utils - 기존 calendar feature 동일
# =============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sine_cosine_encoding(value: float, max_val: float):
    return math.sin(2 * math.pi * value / max_val), math.cos(2 * math.pi * value / max_val)

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

    df["is_spring"]   = df["month"].isin([4,5,6]).astype(int)
    df["is_summer"]   = df["month"].isin([7,8]).astype(int)
    df["is_autumn"]   = df["month"].isin([9,10,11]).astype(int)
    df["is_winter"]   = df["month"].isin([12,1,2,3]).astype(int)

    df["is_summer_vacation"] = df["month"].isin([7,8]).astype(int)
    df["is_winter_vacation"] = df["month"].isin([12,1,2]).astype(int)

    df[["month_sin","month_cos"]] = df["month"].apply(lambda m: pd.Series(sine_cosine_encoding(m,12)))
    df[["dow_sin","dow_cos"]]     = df["dow"].apply(lambda d: pd.Series(sine_cosine_encoding(d,7)))
    df[["doy_sin","doy_cos"]]     = df["date"].apply(lambda d: pd.Series(sine_cosine_encoding(d.dayofyear,365)))
    df[["week_sin","week_cos"]]   = df["week"].apply(lambda w: pd.Series(sine_cosine_encoding(w,52)))
    df[["quarter_sin","quarter_cos"]] = df["quarter"].apply(lambda q: pd.Series(sine_cosine_encoding(q,4)))

    return df.drop(columns=["tomorrow","yesterday"])

# =============================================================
# Dataset (menu category 제거 → custom cluster 적용)
# =============================================================
class EnhancedNHiTSDataset(Dataset):
    def __init__(self,cfg:EnhancedNHiTSConfig,df:pd.DataFrame):
        self.cfg = cfg
        self.df = df.copy()
        self.df[cfg.date_col] = pd.to_datetime(self.df[cfg.date_col])
        self.df[cfg.target_col] = self.df[cfg.target_col].clip(lower=0)

        pivot = self.df.pivot(index=cfg.date_col,columns=cfg.item_col,values=cfg.target_col).sort_index()
        self.items = list(pivot.columns)
        self.dates = list(pivot.index)
        self.values= pivot.fillna(0.0).values.astype(np.float32)

        holidays_set = set(pd.to_datetime(cfg.custom_holidays_list))
        caldf = build_enhanced_features(self.dates, holidays_set)
        self.cal_feats = caldf.drop(columns=["date"]).values.astype(np.float32)
        self.cal_feat_names = [c for c in caldf.columns if c!="date"]

        stores = [x.split("_")[0] for x in self.items]
        menus = [x.split("_",1)[1] for x in self.items]

        self.store2idx = {s:i for i,s in enumerate(sorted(set(stores)))}
        self.item_store_idx = np.array([self.store2idx[s] for s in stores], dtype=np.int64)
        self.n_stores = len(self.store2idx)

        # cluster 적용
        clusters = [cluster_mapping.get(it, 0) for it in self.items]
        self.cluster2idx = {c:i for i,c in enumerate(sorted(set(clusters)))}
        self.item_cluster_idx = np.array([self.cluster2idx[c] for c in clusters],dtype=np.int64)
        self.n_clusters = len(self.cluster2idx)

        sw = cfg.store_weights
        self.sample_weights = np.array([sw.get(stores[i],1.0) for i in range(len(stores))],dtype=np.float32)

        T=len(self.dates);Lx,Ly=cfg.in_len,cfg.out_len
        cutoff=pd.to_datetime(cfg.train_end_date)
        max_start=T-(Lx+Ly)
        self.indices=[]
        self.target_end_dates=[]
        for j in range(len(self.items)):
            for t0 in range(0,max_start+1):
                ed=self.dates[t0+Lx+Ly-1]
                if ed<=cutoff:
                    self.indices.append((t0,j))
                    self.target_end_dates.append(ed)
        self.target_end_dates=np.array(self.target_end_dates)

    def __len__(self): return len(self.indices)

    def __getitem__(self,idx):
        cfg=self.cfg; t0,j=self.indices[idx]
        Lx,Ly=cfg.in_len,cfg.out_len

        x=self.values[t0:t0+Lx,j]; y=self.values[t0+Lx:t0+Lx+Ly,j]
        if cfg.log1p: x_in=np.log1p(x); y_out=np.log1p(y)
        else: x_in,y_out=x.copy(),y.copy()

        past_cal=self.cal_feats[t0:t0+Lx,:]
        fut_cal =self.cal_feats[t0+Lx:t0+Lx+Ly,:]

        store_idx=self.item_store_idx[j]
        cluster_idx=self.item_cluster_idx[j]
        sample_w=self.sample_weights[j]

        mean = x.mean(); std = x.std()
        slope = (x[-1] - x[0])/(len(x))

        return {
            "x":torch.from_numpy(x_in).float(),
            "y":torch.from_numpy(y_out).float(),
            "past_cal":torch.from_numpy(past_cal).float(),
            "fut_cal":torch.from_numpy(fut_cal).float(),
            "store_idx":torch.tensor(store_idx).long(),
            "cluster_idx":torch.tensor(cluster_idx).long(),
            "sample_w":torch.tensor(sample_w).float(),
            "mean":torch.tensor(mean).float(),
            "std":torch.tensor(std).float(),
            "slope":torch.tensor(slope).float()
        }

# =============================================================
# Model (conditioning + SiLU + LayerNorm 적용)
# =============================================================
class NHiTSBlock(nn.Module):
    def __init__(self,input_size,output_size,hidden_size,n_layers,dropout,pooling_mode="MaxPool1d",
                 n_pool_kernel_size=2,interpolation_mode="linear",cond_dim=0):
        super().__init__()
        self.pooling_mode=pooling_mode
        self.interpolation_mode=interpolation_mode
        self.n_pool_kernel_size = n_pool_kernel_size
        self.input_size=input_size; self.output_size=output_size

        if pooling_mode=="MaxPool1d":
            self.pool=nn.MaxPool1d(kernel_size=n_pool_kernel_size,stride=n_pool_kernel_size,ceil_mode=True)
        else:
            self.pool=nn.AvgPool1d(kernel_size=n_pool_kernel_size,stride=n_pool_kernel_size,ceil_mode=True)

        pooled_size = math.ceil(input_size/n_pool_kernel_size) + cond_dim
        # activation 함수 선택
        act_fn = {"silu": nn.SiLU(), "relu": nn.ReLU(), "gelu": nn.GELU()}[cfg.activation]

        layers=[nn.Linear(pooled_size, hidden_size), nn.LayerNorm(hidden_size), act_fn, nn.Dropout(dropout)]

        for _ in range(n_layers-1):
            layers += [nn.Linear(hidden_size,hidden_size), nn.LayerNorm(hidden_size), act_fn, nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)
        self.output_layer=nn.Linear(hidden_size,output_size)

    def forward(self,x,cond=None):
        bs=x.size(0)
        x=x.unsqueeze(1); pooled=self.pool(x).squeeze(1)
        if cond is not None:
            pooled=torch.cat([pooled,cond],dim=-1)
        h=self.mlp(pooled)
        return self.output_layer(h)

class EnhancedNHiTSModel(nn.Module):
    def __init__(self,in_len,out_len,cal_dim,n_stores,n_clusters,cfg:EnhancedNHiTSConfig):
        super().__init__()
        if cfg.n_pool_kernel_size is None: cfg.n_pool_kernel_size=[2,2,1]
        if cfg.stack_types is None: cfg.stack_types=["identity","identity","trend"]
        if cfg.n_freq_downsample is None: cfg.n_freq_downsample=[2,1,1]

        while len(cfg.n_pool_kernel_size)<cfg.n_blocks:cfg.n_pool_kernel_size.append(cfg.n_pool_kernel_size[-1])
        while len(cfg.stack_types)<cfg.n_blocks:cfg.stack_types.append("identity")
        while len(cfg.n_freq_downsample)<cfg.n_blocks:cfg.n_freq_downsample.append(1)

        self.store_emb=nn.Embedding(n_stores,64)
        self.cluster_emb=nn.Embedding(n_clusters,32)
        self.cal_proj=nn.Linear(cal_dim,128)

        self.blocks=nn.ModuleList()
        for i in range(cfg.n_blocks):
            self.blocks.append(NHiTSBlock(in_len,out_len,cfg.hidden,cfg.n_layers,cfg.dropout,
                                          pooling_mode=cfg.pooling_mode,
                                          n_pool_kernel_size=cfg.n_pool_kernel_size[i],
                                          cond_dim=64+32+128))

        self.basis_weights=nn.ParameterList([nn.Parameter(torch.randn(out_len, max(1,out_len//max(1,cfg.n_freq_downsample[i]))))
                                             for i in range(cfg.n_blocks)])
        meta_dim=64+32+128
        self.meta_integration=nn.Sequential(
            nn.Linear(meta_dim,cfg.hidden),nn.LayerNorm(cfg.hidden),nn.SiLU(),nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden,out_len))

        prob_in = meta_dim+3 # mean,std,slope
        self.prob_head=nn.Sequential(
            nn.Linear(prob_in,cfg.hidden),nn.LayerNorm(cfg.hidden),nn.SiLU(),nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden,cfg.hidden//2),nn.LayerNorm(cfg.hidden//2),nn.SiLU(),nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden//2,out_len)
        )
        self.stack_types=cfg.stack_types

    def forward(self,x,past_cal,fut_cal,store_idx,cluster_idx,mean,std,slope):
        bs=x.size(0)
        store_emb=self.store_emb(store_idx)     # [B,64]
        cl_emb  =self.cluster_emb(cluster_idx)  # [B,32]
        cal_avg =torch.cat([past_cal,fut_cal],dim=1).mean(dim=1)
        cal_emb =self.cal_proj(cal_avg)         # [B,128]
        meta=torch.cat([store_emb,cl_emb,cal_emb],dim=-1)

        cond=meta # conditioning vector per block
        outs=[]
        for i,b in enumerate(self.blocks):
            o=b(x,cond=cond)
            # trend
            if self.stack_types[i]=="trend":
                basis=self.basis_weights[i]
                if basis.size(1)>1:
                    o=torch.matmul(o,basis.T)
                    o=F.interpolate(o.unsqueeze(1),size=o.size(-1),mode='linear',align_corners=False).squeeze(1)
            outs.append(o)
        base_out = torch.stack(outs,dim=0).sum(dim=0)
        base_out = base_out + self.meta_integration(meta)

        prob_feat=torch.cat([meta,mean.unsqueeze(1),std.unsqueeze(1),slope.unsqueeze(1)],dim=-1)
        p=self.prob_head(prob_feat)
        return base_out,p


# =============================================================
# Loss (기존 코드 유지)
# =============================================================
class UltraEnhancedHurdleLoss(nn.Module):
    def __init__(self,eps=0.01,zero_weight=0.01,lambda_bce=0.15):
        super().__init__()
        self.eps=eps; self.zero_weight=zero_weight; self.lambda_bce=lambda_bce
        self.bce=nn.BCEWithLogitsLoss(reduction='none')

    def forward(self,v_pred_log,p_logits,y_true_log,pos_mask,sample_w):
        z=(pos_mask>0).float()
        bce=self.bce(p_logits,z)
        yp_val=torch.expm1(v_pred_log).clamp_min(0.0)
        yt_val=torch.expm1(y_true_log).clamp_min(0.0)
        ultra_eps=torch.where(yt_val<0.5,self.eps*0.2,torch.where(yt_val<2.0,self.eps*0.5,self.eps))
        denom=(torch.abs(yp_val)+torch.abs(yt_val)).clamp_min(ultra_eps)
        smape_pos=2.0*torch.abs(yp_val-yt_val)/denom
        smape_pos=smape_pos*pos_mask
        p=torch.sigmoid(p_logits)
        y_hat=p*yp_val
        denom2=(torch.abs(y_hat)+torch.abs(yt_val)).clamp_min(ultra_eps)
        smape_all=2.0*torch.abs(y_hat-yt_val)/denom2
        ultra_zero_weight=torch.where(yt_val<0.01,torch.full_like(yt_val,self.zero_weight*0.1),
                torch.where(yt_val<0.1,torch.full_like(yt_val,self.zero_weight*0.3),
                torch.where(yt_val<1.0,torch.full_like(yt_val,self.zero_weight*0.6),torch.ones_like(yt_val))))
        smape_all=smape_all*ultra_zero_weight
        bce_s=bce.mean(dim=1); pos_s=smape_pos.mean(dim=1); all_s=smape_all.mean(dim=1)
        sample_loss=self.lambda_bce*bce_s+0.4*pos_s+0.6*all_s
        sw=sample_w.view(-1); wsum=sw.sum().clamp_min(1e-8)
        loss=(sample_loss*sw).sum()/wsum
        return loss,sample_loss.detach(),sw.detach()

# =============================================================
# EMA
# =============================================================
class EMA:
    def __init__(self,model,decay=0.9998):
        self.decay=decay
        self.shadow={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    def update(self,model):
        with torch.no_grad():
            for n,p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].mul_(self.decay).add_(p.detach(),alpha=1-self.decay)
    def apply_to(self,model):
        with torch.no_grad():
            for n,p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(self.shadow[n])

# =============================================================
# Trainer (CosineAnnealingWarmup 적용)
# =============================================================
class EnhancedNHiTSTrainer:
    def __init__(self,cfg:EnhancedNHiTSConfig,dataset,epochs,batch_size,base_lr,max_lr,weight_decay):
        self.cfg=cfg;self.dataset=dataset
        self.device=torch.device(cfg.device)
        cal_dim=dataset.cal_feats.shape[1]
        self.model=EnhancedNHiTSModel(cfg.in_len,cfg.out_len,cal_dim,dataset.n_stores,dataset.n_clusters,cfg).to(self.device)
        if cfg.use_compile and torch.cuda.is_available():
            try: self.model=torch.compile(self.model,mode="max-autotune")
            except Exception as e: print("compile FAIL",e)
        self.criterion=UltraEnhancedHurdleLoss(cfg.eps_smape,cfg.zero_weight,cfg.hurdle_lambda)
        self.ema=EMA(self.model,decay=cfg.ema_decay)
        self.epochs=epochs; self.batch=batch_size; self.base_lr=base_lr;self.max_lr=max_lr;self.weight_decay = cfg.weight_decay
        if cfg.use_amp and torch.cuda.is_available(): self.scaler=torch.cuda.amp.GradScaler()

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark=True
            torch.backends.cuda.matmul.allow_tf32=True
            torch.set_float32_matmul_precision('high')
            print("GPU:",torch.cuda.get_device_name())

    def make_loaders_from_mask(self,mask_val):
        idx=np.arange(len(self.dataset))
        tr=idx[~mask_val];val=idx[mask_val]
        trl=DataLoader(torch.utils.data.Subset(self.dataset,tr),batch_size=self.batch,shuffle=True,
                       num_workers=self.cfg.num_workers,pin_memory=self.cfg.pin_memory,persistent_workers=self.cfg.persistent_workers)
        vall=DataLoader(torch.utils.data.Subset(self.dataset,val),batch_size=self.batch,shuffle=False,
                        num_workers=self.cfg.num_workers,pin_memory=self.cfg.pin_memory,persistent_workers=self.cfg.persistent_workers)
        return trl,vall

    @torch.no_grad()
    def evaluate(self,loader,use_ema=True):
        self.model.eval();bak=None
        if use_ema:
            bak={k:v.clone() for k,v in self.model.state_dict().items()}
            self.ema.apply_to(self.model)
        losses=[]; weights=[]
        amp_ctx=torch.autocast(device_type='cuda',dtype=torch.bfloat16) if (self.cfg.use_amp and torch.cuda.is_available()) else torch.cuda.amp.autocast(enabled=False)
        with amp_ctx:
            for b in loader:
                x=b["x"].to(self.device);y=b["y"].to(self.device)
                past=b["past_cal"].to(self.device); fut=b["fut_cal"].to(self.device)
                s=b["store_idx"].to(self.device); c=b["cluster_idx"].to(self.device)
                mean=b["mean"].to(self.device);std=b["std"].to(self.device);slope=b["slope"].to(self.device)
                sw=b["sample_w"].to(self.device); pos=(y>0).float()
                vp,pr=self.model(x,past,fut,s,c,mean,std,slope)
                ls,spl,sw2=self.criterion(vp,pr,y,pos,sw)
                losses.append(spl);weights.append(sw2)
        if use_ema and bak is not None: self.model.load_state_dict(bak)
        if len(losses)==0: return 0.0
        spl=torch.cat(losses); swt=torch.cat(weights)
        return (spl*swt).sum().item()/(swt.sum().item()+1e-8)

    def train_with_loaders(self,tl,vl):
        if cfg.opt_type.lower() == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.base_lr, weight_decay=cfg.weight_decay)
        elif cfg.opt_type.lower() == "sgd":
            self.optim = torch.optim.SGD(self.model.parameters(), lr=self.base_lr, momentum=0.9, weight_decay=cfg.weight_decay)
        else:
            # default adamw
            self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=cfg.weight_decay)

        # Scheduler : CosineAnnealingWarmRestarts + warmup
        steps_per_epoch = len(tl)

        if cfg.scheduler_type == "cosine":
            warmup_steps = int(cfg.warmup_pct * self.epochs * steps_per_epoch)
            total_steps  = self.epochs * steps_per_epoch
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1. + math.cos(math.pi * progress))
            self.sched = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
        elif cfg.scheduler_type == "plateau":
            self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.5, patience=5)
        else:    # onecycle
            self.sched = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=self.max_lr, 
                                                            epochs=self.epochs, steps_per_epoch=steps_per_epoch)

        best=float("inf");best_state=None;pat=25;noimp=0
        for ep in range(1,self.epochs+1):
            self.model.train()
            amp=torch.autocast(device_type='cuda',dtype=torch.bfloat16) if (self.cfg.use_amp and torch.cuda.is_available()) else torch.cuda.amp.autocast(enabled=False)
            for b in tl:
                x=b["x"].to(self.device);y=b["y"].to(self.device)
                past=b["past_cal"].to(self.device);fut=b["fut_cal"].to(self.device)
                s=b["store_idx"].to(self.device);c=b["cluster_idx"].to(self.device)
                mean=b["mean"].to(self.device);std=b["std"].to(self.device);slope=b["slope"].to(self.device)
                sw=b["sample_w"].to(self.device);pos=(y>0).float()

                with amp:
                    vp,pr=self.model(x,past,fut,s,c,mean,std,slope)
                    loss,_,_=self.criterion(vp,pr,y,pos,sw)
                self.optim.zero_grad(set_to_none=True)
                if self.cfg.use_amp and torch.cuda.is_available():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optim);torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.5)
                    self.scaler.step(self.optim);self.scaler.update()
                else:
                    loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.5);self.optim.step()
                # 스케줄러 업데이트
                if cfg.scheduler_type == "plateau":
                    self.sched.step(v)          # val loss 기준
                else:
                    self.sched.step()           # 매 스텝 업데이트
                self.ema.update(self.model)

            v=self.evaluate(vl,use_ema=True)
            print(f"[{ep:03d}] val={v:.5f}")
            if v<best: best=v; best_state={k:v.cpu().clone() for k,v in self.model.state_dict().items()};noimp=0
            else:
                noimp+=1; 
                if noimp>=pat:
                    print("Early stop");break
        if best_state is not None: self.model.load_state_dict(best_state)
        return self.model,best

# =============================================================
# Predict util
# =============================================================
@torch.no_grad()
def predict_one_file(cfg,model,test_df,store2idx,cluster2idx):
    device=torch.device(cfg.device)
    t=test_df.copy()
    t[cfg.date_col]=pd.to_datetime(t[cfg.date_col])
    t[cfg.target_col]=t[cfg.target_col].clip(lower=0)
    pivot=t.pivot(index=cfg.date_col,columns=cfg.item_col,values=cfg.target_col).sort_index().fillna(0.0)
    items=list(pivot.columns);dates=list(pivot.index);vals=pivot.values.astype(np.float32)
    last=dates[-1]; fut=[last+pd.Timedelta(days=i) for i in range(1,cfg.out_len+1)]
    hol=set(pd.to_datetime(cfg.custom_holidays_list))

    pastc=build_enhanced_features(dates[-cfg.in_len:],hol).drop(columns=["date"]).values.astype(np.float32)
    futc =build_enhanced_features(fut,hol).drop(columns=["date"]).values.astype(np.float32)
    B,lenx=len(items),cfg.in_len
    x=vals[-lenx:,:].T
    x=torch.from_numpy(np.log1p(x)).float().to(device)

    pastc=torch.from_numpy(np.repeat(pastc[None,:,:],B,axis=0)).float().to(device)
    futc =torch.from_numpy(np.repeat(futc[None,:,:],B,axis=0)).float().to(device)

    stores=[i.split("_")[0] for i in items]
    clusters=[cluster_mapping.get(i,0) for i in items]
    sidx=torch.tensor([store2idx.get(s,0) for s in stores],device=device)
    cidx=torch.tensor([cluster2idx.get(c,0) for c in clusters],device=device)

    with torch.autocast(device_type='cuda',dtype=torch.bfloat16) if cfg.use_amp else torch.cuda.amp.autocast(enabled=False):
        mean=x.mean(dim=1); std=x.std(dim=1); slope=(x[:, -1]-x[:,0])/lenx
        v,p=model(x,pastc,futc,sidx,cidx,mean,std,slope)
        y=torch.expm1(v).clamp_min(0.0);pro=torch.sigmoid(p);pred=(y*pro).cpu().numpy()
    return pd.DataFrame(pred,index=items,columns=[f"D+{i}" for i in range(1,cfg.out_len+1)]).T

# =============================================================
# Rolling-CV utils 동일 (생략 가능)
# =============================================================
def make_val_mask_by_week(dataset,end_str):
    end=pd.to_datetime(end_str); st=end-pd.Timedelta(days=6)
    ted=dataset.target_end_dates
    return (ted>=st)&(ted<=end)

# =============================================================
# Main
# =============================================================
if __name__=="__main__":
    cfg=EnhancedNHiTSConfig()
    if cfg.store_weights is None: cfg.store_weights=DEFAULT_STORE_WEIGHTS
    if cfg.custom_holidays_list is None: cfg.custom_holidays_list=DEFAULT_CUSTOM_HOLIDAYS

    set_seed(cfg.seed)
    cfg.USE_OPTUNA=False; print("cluster+conditioning 버전")
    train_df=pd.read_csv(cfg.train_csv)
    ds=EnhancedNHiTSDataset(cfg,train_df)
    trainer=EnhancedNHiTSTrainer(cfg,ds,cfg.EPOCHS_FULL,cfg.BATCH_FULL,cfg.BASE_LR_FULL,cfg.MAX_LR_FULL,cfg.WD_FULL)

    mask=make_val_mask_by_week(ds,cfg.cv_fold_end_dates[0])
    tl,vl=trainer.make_loaders_from_mask(mask)
    model,val=trainer.train_with_loaders(tl,vl)
    print("final val=",val)

    out={"model_state":model.state_dict(),"cfg":cfg.__dict__,"store2idx":ds.store2idx,
         "cluster2idx":ds.cluster2idx,"final_val":val}
    torch.save(out,"./data/enhanced_nhits_cluster_model.pth")
    print("saved.")

    print("predicting...")
    test_files=sorted(glob.glob(cfg.test_glob))
    subtemp=pd.read_csv(cfg.submission_template_csv)
    all=[]
    for i,f in enumerate(test_files):
        t=pd.read_csv(f)
        block=predict_one_file(cfg,model,t,ds.store2idx,ds.cluster2idx)
        block.index=[f"TEST_{i:02d}+{k}일" for k in range(1,cfg.out_len+1)]
        all.append(block)

    final=pd.concat(all,axis=0);final.reset_index(inplace=True);final.rename(columns={"index":"영업일자"},inplace=True)
    final=final.reindex(columns=subtemp.columns,fill_value=0)
    num=[c for c in final.columns if c!="영업일자"]
    for c in num:
        Q95=final[c].quantile(0.95)
        final[c]=np.where(final[c]>Q95*2.0,Q95*1.5,final[c])
    final[num]=np.rint(np.clip(final[num].values,a_min=0,a_max=None)).astype(int)
    final.to_csv(cfg.out_submission_csv,index=False,encoding="utf-8-sig")
    print("Done:",cfg.out_submission_csv)
