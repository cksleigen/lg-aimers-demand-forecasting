# Import
import os
import random
import glob
import re

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from tqdm import tqdm

# Fixed RandomSeed & Setting Hyperparameter
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

LOOKBACK, PREDICT, BATCH_SIZE, EPOCHS = 28, 7, 16, 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Load
train = pd.read_csv('./data/train/train.csv')

# Define Model
class MultiOutputLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=7):
        super(MultiOutputLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # (B, output_dim)

# Train
def train_lstm(train_df):
    trained_models = {}

    for store_menu, group in tqdm(train_df.groupby(['영업장명_메뉴명']), desc ='Training LSTM'):
        store_train = group.sort_values('영업일자').copy()
        if len(store_train) < LOOKBACK + PREDICT:
            continue

        features = ['매출수량']
        scaler = MinMaxScaler()
        store_train[features] = scaler.fit_transform(store_train[features])
        train_vals = store_train[features].values  # shape: (N, 1)

        # 시퀀스 구성
        X_train, y_train = [], []
        for i in range(len(train_vals) - LOOKBACK - PREDICT + 1):
            X_train.append(train_vals[i:i+LOOKBACK])
            y_train.append(train_vals[i+LOOKBACK:i+LOOKBACK+PREDICT, 0])

        X_train = torch.tensor(X_train).float().to(DEVICE)
        y_train = torch.tensor(y_train).float().to(DEVICE)

        model = MultiOutputLSTM(input_dim=1, output_dim=PREDICT).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(EPOCHS):
            idx = torch.randperm(len(X_train))
            for i in range(0, len(X_train), BATCH_SIZE):
                batch_idx = idx[i:i+BATCH_SIZE]
                X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
                output = model(X_batch)
                loss = criterion(output, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        trained_models[store_menu] = {
            'model': model.eval(),
            'scaler': scaler,
            'last_sequence': train_vals[-LOOKBACK:]  # (28, 1)
        }

    return trained_models

# 학습
trained_models = train_lstm(train)

# Prediction
def predict_lstm(test_df, trained_models, test_prefix: str):
    results = []

    for store_menu, store_test in test_df.groupby(['영업장명_메뉴명']):
        key = store_menu
        if key not in trained_models:
            continue

        model = trained_models[key]['model']
        scaler = trained_models[key]['scaler']

        store_test_sorted = store_test.sort_values('영업일자')
        recent_df = store_test_sorted[['매출수량']].iloc[-LOOKBACK:]
        if len(recent_df) < LOOKBACK:
            continue

        # 정규화
        recent_vals = scaler.transform(recent_df)
        x_input = torch.tensor([recent_vals]).float().to(DEVICE)

        with torch.no_grad():
            pred_scaled = model(x_input).squeeze().cpu().numpy()

        # 역변환
        restored = []
        for i in range(PREDICT):
            dummy = np.zeros((1, 1))
            dummy[0, 0] = pred_scaled[i]
            restored_val = scaler.inverse_transform(dummy)[0, 0]
            restored.append(max(restored_val, 0))

        # 예측일자: TEST_00+1일 ~ TEST_00+7일
        pred_dates = [f"{test_prefix}+{i+1}일" for i in range(PREDICT)]

        for d, val in zip(pred_dates, restored):
            results.append({
                '영업일자': d,
                '영업장명_메뉴명': store_menu,
                '매출수량': val
            })

    return pd.DataFrame(results)

all_preds = []

# 모든 test_*.csv 순회
test_files = sorted(glob.glob('./data/test/TEST_*.csv'))

for path in test_files:
    test_df = pd.read_csv(path)

    # 파일명에서 접두어 추출 (예: TEST_00)
    filename = os.path.basename(path)
    test_prefix = re.search(r'(TEST_\d+)', filename).group(1)

    pred_df = predict_lstm(test_df, trained_models, test_prefix)
    all_preds.append(pred_df)
    
full_pred_df = pd.concat(all_preds, ignore_index=True)

# Submission
def convert_to_submission_format(pred_df: pd.DataFrame, sample_submission: pd.DataFrame):
    # (영업일자, 메뉴) → 매출수량 딕셔너리로 변환
    pred_dict = dict(zip(
        zip(pred_df['영업일자'], pred_df['영업장명_메뉴명']),
        pred_df['매출수량']
    ))

    final_df = sample_submission.copy()

    for row_idx in final_df.index:
        date = final_df.loc[row_idx, '영업일자']
        for col in final_df.columns[1:]:  # 메뉴명들
            final_df.loc[row_idx, col] = pred_dict.get((date, col), 0)

    return final_df

sample_submission = pd.read_csv('./data/sample_submission.csv')
submission = convert_to_submission_format(full_pred_df, sample_submission)
submission.to_csv('./data/baseline_submission.csv', index=False, encoding='utf-8-sig')