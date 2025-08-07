# ==============================================================================
# 0. 사전 준비: 라이브러리 임포트 및 전역 변수 설정
# ==============================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import glob
from tqdm import tqdm
import warnings
import optuna
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

# (이하 공휴일 목록 정의는 이전과 동일)
custom_holidays_list = [
    # ... 공휴일 리스트 ...
    '2023-01-01', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-03-01', '2023-05-01',
    '2023-05-05', '2023-05-27', '2023-06-06', '2023-08-15', '2023-09-28', '2023-09-29', '2023-09-30',
    '2023-10-02', '2023-10-03', '2023-10-09', '2023-12-25',
    '2024-01-01', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12', '2024-03-01', '2024-04-10',
    '2024-05-01', '2024-05-05', '2024-05-06', '2024-05-15', '2024-06-06', '2024-08-15', '2024-09-16',
    '2024-09-17', '2024-09-18', '2024-10-01', '2024-10-03', '2024-10-09', '2024-12-25',
    '2025-01-01', '2025-01-28', '2025-01-29', '2025-01-30', '2025-03-01', '2025-03-03', '2025-05-01',
    '2025-05-05', '2025-05-06', '2025-06-06', '2025-08-15'
]
custom_holidays = set(pd.to_datetime(custom_holidays_list))

weights_map = {
    '미라시아': 7.71, '담하': 6.51, '연회장': 3.48, '라그로타': 3.44,
    '느티나무 셀프BBQ': 2.78, '화담숲주막': 1.43, '카페테리아': 1.31,
    '화담숲카페': 1.14, '포레스트릿': 1.00
}

# ==============================================================================
# 1. 데이터 로드
# ==============================================================================
print("데이터 로드를 시작합니다...")
train_df = pd.read_csv('./data/train/train.csv')
train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
test_files = sorted(glob.glob('./data/test/*.csv'))
submission_template = pd.read_csv('./data/sample_submission.csv') 
print(f"Train 데이터 로드 완료, Test 파일 {len(test_files)}개 로드 완료.")

# ==============================================================================
# 2. 피처 엔지니어링 함수 정의
# ==============================================================================
def create_base_features(df):
    df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df['요일'] = df['영업일자'].dt.dayofweek
    df['월'] = df['영업일자'].dt.month
    df['연중일자'] = df['영업일자'].dt.dayofyear
    df['주말여부'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)
    df['공휴일여부'] = df['영업일자'].apply(lambda x: 1 if x in custom_holidays else 0)
    df['휴일여부'] = df.apply(lambda row: 1 if row['주말여부'] == 1 or row['공휴일여부'] == 1 else 0, axis=1)
    df['하루뒤_날짜'] = df['영업일자'] + pd.to_timedelta(1, unit='D')
    df['휴일전날여부'] = df['하루뒤_날짜'].apply(lambda x: 1 if (x in custom_holidays or x.dayofweek >= 5) else 0)
    df.drop(columns=['하루뒤_날짜'], inplace=True)
    def get_season(month):
        if month in [4, 5, 6]: return '봄'
        elif month in [7, 8]: return '여름'
        elif month in [9, 10, 11]: return '가을'
        else: return '겨울'
    df['계절'] = df['월'].apply(get_season)
    return df

print("피처 엔지니어링 함수 정의 완료.")

# ==============================================================================
# 3. 학습 데이터 생성 (Sliding Window 방식)
# ==============================================================================
print("학습 데이터를 Sliding Window 방식으로 생성합니다... (시간이 소요될 수 있습니다)")
# (이하 학습 데이터 생성 로직은 이전과 동일)
train_df_processed = create_base_features(train_df)
train_df_processed = train_df_processed.sort_values(by=['영업장명_메뉴명', '영업일자'])
encoders = {}
categorical_features = ['영업장명', '메뉴명']
for feature in categorical_features:
    le = LabelEncoder()
    train_df_processed[feature] = le.fit_transform(train_df_processed[feature])
    encoders[feature] = le
train_df_processed = pd.get_dummies(train_df_processed, columns=['요일', '계절'], prefix=['요일', '계절'])
ohe_columns = [col for col in train_df_processed.columns if col.startswith('요일_') or col.startswith('계절_')]
encoders['ohe_columns'] = ohe_columns

training_samples = []
for item_id, group in tqdm(train_df_processed.groupby('영업장명_메뉴명'), desc="학습 샘플 생성 중"):
    if len(group) < 28 + 7: continue
    for i in range(len(group) - 28 - 7 + 1):
        input_window = group.iloc[i : i+28]
        target_window = group.iloc[i+28 : i+28+7]
        features = input_window.tail(1).copy()
        windows = [7, 14, 21, 28]
        for window in windows:
            features[f'rolling_mean_{window}'] = input_window['매출수량'].rolling(window, min_periods=1).mean().iloc[-1]
            features[f'rolling_std_{window}'] = input_window['매출수량'].rolling(window, min_periods=1).std().iloc[-1]
        lags = [7, 14, 21, 28]
        for lag in lags:
            if len(input_window) >= lag: features[f'lag_{lag}'] = input_window['매출수량'].iloc[-lag]
            else: features[f'lag_{lag}'] = np.nan
        for day in range(1, 8): features[f'target_d{day}'] = target_window['매출수량'].iloc[day-1]
        training_samples.append(features)

final_train_df = pd.concat(training_samples, ignore_index=True)
final_train_df.fillna(0, inplace=True)

cyclical_features = {'월': 12, '연중일자': 365}
for feature, max_val in cyclical_features.items():
    final_train_df[f'{feature}_sin'] = np.sin(2 * np.pi * final_train_df[feature] / max_val)
    final_train_df[f'{feature}_cos'] = np.cos(2 * np.pi * final_train_df[feature] / max_val)
    final_train_df.drop(columns=[feature], inplace=True)
print("학습 데이터 생성 완료.")

# ==============================================================================
# 3.5. 하이퍼파라미터 개별 튜닝 (Optuna)
# ==============================================================================
print("Optuna를 사용하여 모델별 하이퍼파라미터 튜닝을 시작합니다... (시간이 매우 오래 소요됩니다)")

# 튜닝용 학습/검증 데이터 분리
validation_cutoff_date = final_train_df['영업일자'].max() - pd.to_timedelta(28, unit='D')
tune_train_df = final_train_df[final_train_df['영업일자'] <= validation_cutoff_date]
tune_val_df = final_train_df[final_train_df['영업일자'] > validation_cutoff_date]

# Label Encoding된 숫자를 다시 원래 영업장명으로 변환하기 위한 맵
encoded_to_name = {code: name for name, code in zip(encoders['영업장명'].classes_, encoders['영업장명'].transform(encoders['영업장명'].classes_))}

def weighted_smape(y_true, y_pred, stores_encoded):
    """
    대회 공식 평가 산식인 Weighted SMAPE를 계산하는 함수.
    y_true: 실제값, y_pred: 예측값, stores_encoded: 인코딩된 영업장명 Series
    """
    # 입력 데이터를 데이터프레임으로 묶기
    df = pd.DataFrame({
        'actual': y_true,
        'pred': y_pred,
        'store_encoded': stores_encoded
    })
    
    # 규칙: 실제값이 0인 경우는 계산에서 제외
    df = df[df['actual'] != 0].copy()
    
    # SMAPE 계산 (0으로 나누는 경우 방지를 위해 분모에 작은 값(epsilon) 추가)
    epsilon = 1e-10
    df['smape'] = 2 * np.abs(df['pred'] - df['actual']) / (np.abs(df['actual']) + np.abs(df['pred']) + epsilon)
    
    # 영업장명 복원
    df['store'] = df['store_encoded'].map(encoded_to_name)
    
    # 영업장별 SMAPE의 평균 계산
    store_smapes = df.groupby('store')['smape'].mean()
    
    # 영업장별 가중치 적용
    total_score = 0
    total_weight = 0
    for store_name, smape in store_smapes.items():
        weight = weights_map.get(store_name, 1.0) # 맵에 없는 경우 기본 가중치 1
        total_score += smape * weight
        total_weight += weight
        
    # 최종 점수: 가중 평균 (점수가 낮을수록 좋음)
    if total_weight == 0: # 모든 실제값이 0인 경우 등 예외 처리
        return 0.0
    return total_score / total_weight

# Optuna 목적 함수
def objective(trial, target_day):
    base_features = [
        '영업장명', '메뉴명', '휴일여부', '휴일전날여부',
        '월_sin', '월_cos', '연중일자_sin', '연중일자_cos'
    ] + ohe_columns
    rolling_features = [f'rolling_mean_{w}' for w in [7, 14, 21, 28]] + [f'rolling_std_{w}' for w in [7, 14, 21, 28]]
    lag_features = [f'lag_{l}' for l in [7, 14, 21, 28]]
    current_features = base_features + rolling_features + lag_features

    params = {
        'objective': 'regression_l1', 'metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': 1,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbose': -1, 'n_jobs': -1, 'seed': 42
    }
    
    # **[수정]** 인자로 받은 target_day를 사용하여 타겟 컬럼을 동적으로 설정
    target_col = f'target_d{target_day}'
    X_train_tune = tune_train_df[current_features]
    y_train_tune = tune_train_df[target_col]
    X_val_tune = tune_val_df[current_features]
    y_val_tune = tune_val_df[target_col]
    stores_val_tune = tune_val_df['영업장명']

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train_tune, y_train_tune,
              eval_set=[(X_val_tune, y_val_tune)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    
    preds = model.predict(X_val_tune)
    score = weighted_smape(y_val_tune, preds, stores_val_tune)
    return score

# 7개 모델 각각에 대한 최적 파라미터를 저장할 딕셔너리
best_params_per_model = {}

# 7개 모델을 순회하며 각각 튜닝 실행
for i in range(1, 8):
    print(f"\n--- D+{i}일 예측 모델 튜닝 시작 ---")
    study = optuna.create_study(direction='minimize')
    # lambda 함수를 사용하여 objective 함수에 현재 튜닝할 모델의 인덱스(i)를 전달
    study.optimize(lambda trial: objective(trial, i), n_trials=30) # n_trials를 줄여서 테스트 (e.g., 30)
    
    best_params = study.best_trial.params
    best_params_per_model[f'd{i}'] = best_params
    print(f"--- D+{i}일 모델 최적 파라미터 탐색 완료 ---")
    print('Best trial number:', study.best_trial.number)
    # **[수정]** 출력 메시지를 MAE에서 Weighted SMAPE로 변경
    print('Best value (Weighted SMAPE):', study.best_value) 
    print('Best trial params:', study.best_trial.params)


# ==============================================================================
# 4. 모델 7개 학습 (개별 최적화된 파라미터 사용)
# ==============================================================================
print("\n개별 최적화된 파라미터로 최종 모델 7개 학습을 시작합니다...")
# (이하 학습 및 추론 로직은 이전과 거의 동일하나, 각 모델에 맞는 파라미터를 적용하는 부분만 다름)
weights_map = {
    '미라시아': 7.71, '담하': 6.51, '연회장': 3.48, '라그로타': 3.44,
    '느티나무 셀프BBQ': 2.78, '화담숲주막': 1.43, '카페테리아': 1.31,
    '화담숲카페': 1.14, '포레스트릿': 1.00
}
encoded_to_name = {code: name for name, code in zip(encoders['영업장명'].classes_, encoders['영업장명'].transform(encoders['영업장명'].classes_))}
final_train_df['sample_weight'] = final_train_df['영업장명'].map(encoded_to_name).map(weights_map).fillna(1.0)

base_features = [
    '영업장명', '메뉴명', '휴일여부', '휴일전날여부',
    '월_sin', '월_cos', '연중일자_sin', '연중일자_cos'
] + ohe_columns
rolling_features = [f'rolling_mean_{w}' for w in [7, 14, 21, 28]] + [f'rolling_std_{w}' for w in [7, 14, 21, 28]]
lag_features = [f'lag_{l}' for l in [7, 14, 21, 28]]
current_features = base_features + rolling_features + lag_features

X_train = final_train_df[current_features]
sample_weight = final_train_df['sample_weight']

models = {}

for i in tqdm(range(1, 8), desc="최종 모델 학습 진행"):
    y_train = final_train_df[f'target_d{i}']
    
    # 현재 모델(d{i})에 맞는 최적 파라미터를 가져옴
    current_best_params = best_params_per_model[f'd{i}']
    
    final_params = {
        'objective': 'regression_l1', 'metric': 'mae', 
        'verbose': -1, 'n_jobs': -1, 'seed': 42
    }
    final_params.update(current_best_params)

    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    models[f'model_d{i}'] = model

print("모델 7개 학습 완료.")

# ==============================================================================
# 5. 추론 및 제출 파일 생성
# ==============================================================================
print("추론 및 제출 파일 생성을 시작합니다...")
# (이하 추론 로직은 이전과 동일)
all_preds = []

for test_file in tqdm(test_files, desc="Test 파일별 추론 진행"):
    test_df = pd.read_csv(test_file)
    history_df = create_base_features(test_df.copy())
    history_df = history_df.sort_values(by=['영업장명_메뉴명', '영업일자'])
    
    grouped_rolling = history_df.groupby('영업장명_메뉴명')['매출수량']
    for window in [7, 14, 21, 28]:
        history_df[f'rolling_mean_{window}'] = grouped_rolling.rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
        history_df[f'rolling_std_{window}'] = grouped_rolling.rolling(window=window, min_periods=1).std().reset_index(0, drop=True)

    pred_input_df_base = history_df.groupby('영업장명_메뉴명').tail(1).reset_index(drop=True)
    pred_input_df_base.fillna(0, inplace=True)
    
    original_item_names = pred_input_df_base['영업장명_메뉴명'].copy()
    for feature in ['영업장명', '메뉴명']:
        pred_input_df_base[feature] = pred_input_df_base[feature].apply(lambda x: encoders[feature].transform([x])[0] if x in encoders[feature].classes_ else -1)
    
    pred_input_df_base = pd.get_dummies(pred_input_df_base, columns=['요일', '계절'], prefix=['요일', '계절'])
    for col in ohe_columns:
        if col not in pred_input_df_base.columns:
            pred_input_df_base[col] = 0
            
    for feature, max_val in cyclical_features.items():
        original_col_name = '월' if '월' in feature else '연중일자'
        pred_input_df_base[f'{feature}_sin'] = np.sin(2 * np.pi * pred_input_df_base[original_col_name] / max_val)
        pred_input_df_base[f'{feature}_cos'] = np.cos(2 * np.pi * pred_input_df_base[original_col_name] / max_val)

    for lag in [7, 14, 21, 28]:
        lag_data = history_df.groupby('영업장명_메뉴명')['매출수량'].apply(lambda x: x.iloc[-lag] if len(x) >= lag else 0).rename(f'lag_{lag}').reset_index()
        pred_input_df_base = pd.merge(pred_input_df_base, lag_data, on='영업장명_메뉴명', how='left')

    X_test = pred_input_df_base[current_features]
    last_date = pd.to_datetime(test_df['영업일자'].max())
    
    for i in range(1, 8):
        model = models[f'model_d{i}']
        predictions = model.predict(X_test)
        predictions[predictions < 0] = 0

        pred_date = last_date + pd.to_timedelta(i, unit='D')
        temp_df = pd.DataFrame({
            '영업일자': pred_date,
            '영업장명_메뉴명': original_item_names,
            '매출수량': np.round(predictions).astype(int)
        })
        all_preds.append(temp_df)

final_submission_df = pd.concat(all_preds, ignore_index=True)
submission_df = final_submission_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()
submission_df['영업일자'] = submission_df['영업일자'].dt.strftime('%Y-%m-%d')
submission_df = submission_df.reindex(columns=submission_template.columns, fill_value=0)

submission_df.to_csv('./data/lightGBM_model7_weight_season_add_optuna_SMAPE_submission.csv', index=False)
print("lightGBM_model7_weight_season_add_optuna_SMAPE_submission.csv 파일 생성이 완료되었습니다.")