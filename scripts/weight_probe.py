"""
weight_probe.py
────────────────────────────────────────────────────────────
● make 24      → baseline_ones.csv + probe_24_00 … 23.csv  (25회용)
● make 49      → baseline_ones.csv + probe_49_00 … 48.csv  (50회용)
● fit  24      → score_log.csv 읽어  β(= w × (2–E)) 상대값 추정
● fit  49      → 50회 버전 정밀 추정
────────────────────────────────────────────────────────────
score_log.csv  예시
file_name,score
baseline_ones.csv,1.41782
probe_24_00.csv,1.73621
…
probe_49_48.csv,1.49305
"""

import pandas as pd, numpy as np, argparse
from pathlib import Path
from scipy.linalg import hadamard
from numpy.linalg import lstsq

# ────────── 공통 유틸 ────────────────────────────────────
def store(col: str) -> str:
    """메뉴 컬럼명 → 업장명 (첫 '_' 이전 문자열)"""
    return col.split('_')[0].strip()

def baseline_ones(csv_path: Path) -> Path:
    """모든 예측값을 1로 채운 baseline_ones.csv 저장"""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.iloc[:, 1:] = 1
    out = csv_path.with_name('baseline_ones.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'✅ baseline → {out}')
    return out

def hadamard_design(m: int, n_probe: int) -> np.ndarray:
    """m≤32 : Hadamard 기반 설계 / 그 이상은 랜덤"""
    if m <= 32:
        H = hadamard(64 if n_probe > 32 else 32)
        return (H[1:n_probe + 1, :m] == 1).astype(int)
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=(n_probe, m)).astype(int)

# ────────── CLI 정의 ───────────────────────────────────
cli = argparse.ArgumentParser()
sub = cli.add_subparsers(dest='cmd', required=True)

mk = sub.add_parser('make', help='probe CSV 생성')
mk.add_argument('n_probe', type=int, choices=[24, 49])

ft = sub.add_parser('fit', help='가중치 추정')
ft.add_argument('mode', type=int, choices=[24, 49])
ft.add_argument('--scorelog', default='./data/probes/score_log.csv')

cli.add_argument('--baseline', default='./data/baseline_submission.csv')
cli.add_argument('--probedir', default='./data/probes/dataset')
args = cli.parse_args()

BASE = Path(args.baseline)
PDIR = Path(args.probedir)
PDIR.mkdir(exist_ok=True)

# ────────── (A) probe 생성 ─────────────────────────────
if args.cmd == 'make':
    one_csv = baseline_ones(BASE)
    base_df = pd.read_csv(one_csv, encoding='utf-8-sig')
    stores = sorted({store(c) for c in base_df.columns[1:]})
    V = hadamard_design(len(stores), args.n_probe)
    np.save(PDIR / f'V_{args.n_probe}.npy', V)

    for k, row in enumerate(V):
        df = base_df.copy()
        for flag, st in zip(row, stores):
            if flag:
                df.loc[:, df.columns.str.startswith(st + '_')] = 0
        fname = PDIR / f'probe_{args.n_probe}_{k:02d}.csv'
        df.to_csv(fname, index=False, encoding='utf-8-sig')
        print(f'{fname.name:<20} OFF {row.sum():2d} stores')
    print(f'🟢 probe 세트 완성  (총 {args.n_probe + 1} 파일)')

# ────────── (B) 가중치(β) 추정 ─────────────────────────
if args.cmd == 'fit':
    V = np.load(PDIR / f'V_{args.mode}.npy')
    stores = sorted({store(c) for c in
                     pd.read_csv(BASE, nrows=1).columns[1:]})
    log = pd.read_csv(args.scorelog)
    if 'baseline_ones.csv' not in log.file_name.values:
        raise SystemExit('baseline_ones.csv 점수가 score_log.csv 에 필요합니다')
    L0 = log.loc[log.file_name == 'baseline_ones.csv', 'score'].iloc[0]

    Δ, rows = [], []
    for _, r in log.iterrows():
        if f'_{args.mode}_' not in r.file_name:
            continue
        idx = int(r.file_name.split('_')[-1].split('.')[0])
        Δ.append(r.score - L0)
        rows.append(V[idx])

    β, *_ = lstsq(np.vstack(rows), np.array(Δ), rcond=None)
    β = β.clip(min=0)
    rel = β / β.min()

    print(f'\n=== 상대 가중치 β (mode {args.mode}) ===')
    for s, v in sorted(zip(stores, rel), key=lambda x: -x[1]):
        print(f'{s:<12}: {v:6.2f}')
    print('β 값은 sample_weight 로 직접 사용하면 됩니다.\n')
