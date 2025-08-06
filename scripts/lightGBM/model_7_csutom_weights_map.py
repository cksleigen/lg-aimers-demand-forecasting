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

# 파이썬 경고 메시지 무시
warnings.filterwarnings('ignore')

# 사용자가 제공한 공휴일 목록을 set 자료구조로 정의 (조회 속도 향상)
custom_holidays_list = [
    # 2023년
    '2023-01-01', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-03-01', '2023-05-01',
    '2023-05-05', '2023-05-27', '2023-06-06', '2023-08-15', '2023-09-28', '2023-09-29', '2023-09-30',
    '2023-10-02', '2023-10-03', '2023-10-09', '2023-12-25',
    # 2024년
    '2024-01-01', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12', '2024-03-01', '2024-04-10',
    '2024-05-01', '2024-05-05', '2024-05-06', '2024-05-15', '2024-06-06', '2024-08-15', '2024-09-16',
    '2024-09-17', '2024-09-18', '2024-10-01', '2024-10-03', '2024-10-09', '2024-12-25',
    # 2025년
    '2025-01-01', '2025-01-28', '2025-01-29', '2025-01-30', '2025-03-01', '2025-03-03', '2025-05-01',
    '2025-05-05', '2025-05-06', '2025-06-06', '2025-08-15'
]
custom_holidays = set(pd.to_datetime(custom_holidays_list))

# ==============================================================================
# 1. 데이터 로드
# ==============================================================================
print("데이터 로드를 시작합니다...")
train_df = pd.read_csv('./data/train.csv')
test_files = sorted(glob.glob('./data/test/*.csv'))
submission_template = pd.read_csv('./data/submission.csv') 
print(f"Train 데이터 로드 완료, Test 파일 {len(test_files)}개 로드 완료.")

# ==============================================================================
# 2. 피처 엔지니어링 함수 정의
# ==============================================================================
def create_base_features(df):
    """데이터프레임을 받아 기본적인 날짜 및 ID 관련 피처를 생성하는 함수"""
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
    return df

print("피처 엔지니어링 함수 정의 완료.")

# ==============================================================================
# 3. 학습 데이터 전처리 및 인코딩
# ==============================================================================
print("학습 데이터 전처리를 시작합니다...")
train_df = create_base_features(train_df)
train_df = train_df.sort_values(by=['영업장명_메뉴명', '영업일자'])

# 7개의 미래 타겟(y) 변수 생성
grouped = train_df.groupby('영업장명_메뉴명')['매출수량']
for i in range(1, 8):
    train_df[f'target_d{i}'] = grouped.shift(-i)

# 타겟 생성이 불가능한 마지막 7일 데이터 제거
train_df = train_df.dropna(subset=[f'target_d{i}' for i in range(1, 8)])

# 인코딩 처리
encoders = {}
categorical_features = ['영업장명', '메뉴명']
for feature in categorical_features:
    le = LabelEncoder()
    train_df[feature] = le.fit_transform(train_df[feature])
    encoders[feature] = le
train_df = pd.get_dummies(train_df, columns=['요일'], prefix='요일')
encoders['요일_columns'] = [col for col in train_df.columns if col.startswith('요일_')]
cyclical_features = {'월': 12, '연중일자': 365}
for feature, max_val in cyclical_features.items():
    train_df[f'{feature}_sin'] = np.sin(2 * np.pi * train_df[feature] / max_val)
    train_df[f'{feature}_cos'] = np.cos(2 * np.pi * train_df[feature] / max_val)
    train_df.drop(columns=[feature], inplace=True)
print("학습 데이터 전처리 완료.")

# ==============================================================================
# 4. 모델 7개 학습 (Lagged Features 전략 적용)
# ==============================================================================
print("LightGBM 모델 7개 학습을 시작합니다 (Lagged Features 전략 적용)...")
weights_map = {
    '미라시아': 7.71, '담하': 6.51, '연회장': 3.48, '라그로타': 3.44,
    '느티나무 셀프BBQ': 2.78, '화담숲주막': 1.43, '카페테리아': 1.31,
    '화담숲카페': 1.14, '포레스트릿': 1.00
}
encoded_to_name = {code: name for name, code in zip(encoders['영업장명'].classes_, encoders['영업장명'].transform(encoders['영업장명'].classes_))}
train_df['sample_weight'] = train_df['영업장명'].map(encoded_to_name).map(weights_map).fillna(1.0)

base_features = [
    '영업장명', '메뉴명', '휴일여부', '휴일전날여부',
    '월_sin', '월_cos', '연중일자_sin', '연중일자_cos'
] + encoders['요일_columns']

models = {}

for i in tqdm(range(1, 8), desc="모델 학습 진행"):
    # 예측 시점과 피처 시점 간의 최소 간격(gap)을 7일로 설정
    GAP = 7
    
    # 각 모델별로 사용할 Lagged & Lagged Rolling 피처를 동적으로 생성
    lags = [7, 14, 21, 28]
    lag_features = []
    for lag in lags:
        feature_name = f'lag_{lag+GAP+i-1}'
        train_df[feature_name] = train_df.groupby('영업장명_메뉴명')['매출수량'].shift(lag+GAP+i-1)
        lag_features.append(feature_name)

    windows = [7, 14, 28]
    rolling_features = []
    grouped_rolling = train_df.groupby('영업장명_메뉴명')['매출수량']
    for window in windows:
        rolling_mean_name = f'rolling_mean_{window}_lag_{GAP+i-1}'
        rolling_std_name = f'rolling_std_{window}_lag_{GAP+i-1}'
        train_df[rolling_mean_name] = grouped_rolling.rolling(window=window).mean().shift(GAP+i-1).reset_index(0,drop=True)
        train_df[rolling_std_name] = grouped_rolling.rolling(window=window).std().shift(GAP+i-1).reset_index(0,drop=True)
        rolling_features.extend([rolling_mean_name, rolling_std_name])
    
    current_features = base_features + lag_features + rolling_features
    
    # 피처 생성으로 생긴 결측치가 있는 행들을 임시로 제거하여 학습 데이터 준비
    train_temp = train_df[current_features + [f'target_d{i}', 'sample_weight']].dropna()
    
    X_train = train_temp[current_features]
    y_train = train_temp[f'target_d{i}']
    sample_weight = train_temp['sample_weight']

    params = {
        'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 1000,
        'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'verbose': -1, 'n_jobs': -1, 'seed': 42
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, sample_weight=sample_weight,
              eval_set=[(X_train, y_train)], eval_metric='mae',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    models[f'model_d{i}'] = model

print("모델 7개 학습 완료.")

# ==============================================================================
# 5. 추론 및 제출 파일 생성 (Lagged Features 규칙 엄격 준수)
# ==============================================================================
print("추론 및 제출 파일 생성을 시작합니다...")
all_preds = []

for test_file in tqdm(test_files, desc="Test 파일별 추론 진행"):
    test_df = pd.read_csv(test_file)
    history_df = create_base_features(test_df.copy())
    history_df = history_df.sort_values(by=['영업장명_메뉴명', '영업일자'])
    
    # 추론에 사용할 기본 피처 준비 (인코딩 포함)
    pred_input_df_base = history_df.groupby('영업장명_메뉴명').tail(1).reset_index(drop=True)
    for feature in ['영업장명', '메뉴명']:
        pred_input_df_base[feature] = pred_input_df_base[feature].apply(lambda x: encoders[feature].transform([x])[0] if x in encoders[feature].classes_ else -1)
    pred_input_df_base = pd.get_dummies(pred_input_df_base, columns=['요일'], prefix='요일')
    for col in encoders['요일_columns']:
        if col not in pred_input_df_base.columns:
            pred_input_df_base[col] = 0
    for feature, max_val in cyclical_features.items():
        original_col_name = '월' if '월' in feature else '연중일자'
        pred_input_df_base[f'{feature}_sin'] = np.sin(2 * np.pi * pred_input_df_base[original_col_name] / max_val)
        pred_input_df_base[f'{feature}_cos'] = np.cos(2 * np.pi * pred_input_df_base[original_col_name] / max_val)
    
    last_date = pd.to_datetime(test_df['영업일자'].max())
    original_item_names = pred_input_df_base['영업장명_메뉴명']

    for i in range(1, 8):
        # 각 모델에 맞는 Lagged & Lagged Rolling 피처를 동적으로 생성
        X_test_temp = pred_input_df_base.copy()
        GAP = 7
        
        # Lag 피처 생성
        for lag in lags:
            shift_val = lag + GAP + i - 1
            lag_data = history_df.groupby('영업장명_메뉴명')['매출수량'].apply(lambda x: x.iloc[-shift_val] if len(x) >= shift_val else np.nan).rename(f'lag_{shift_val}').reset_index()
            X_test_temp = pd.merge(X_test_temp, lag_data, on='영업장명_메뉴명', how='left')

        # Lagged Rolling 피처 생성
        grouped_rolling = history_df.groupby('영업장명_메뉴명')['매출수량']
        for window in windows:
            rolling_mean_name = f'rolling_mean_{window}_lag_{GAP+i-1}'
            rolling_std_name = f'rolling_std_{window}_lag_{GAP+i-1}'
            
            # .rolling().mean() 후 .shift()를 적용하여 Lagged Rolling 값을 계산
            mean_series = grouped_rolling.rolling(window).mean().shift(GAP+i-1)
            std_series = grouped_rolling.rolling(window).std().shift(GAP+i-1)
            
            # 각 그룹의 마지막 값을 가져옴
            mean_data = mean_series.groupby('영업장명_메뉴명').tail(1).rename(rolling_mean_name).reset_index()
            std_data = std_series.groupby('영업장명_메뉴명').tail(1).rename(rolling_std_name).reset_index()

            X_test_temp = pd.merge(X_test_temp, mean_data[['영업장명_메뉴명', rolling_mean_name]], on='영업장명_메뉴명', how='left')
            X_test_temp = pd.merge(X_test_temp, std_data[['영업장명_메뉴명', rolling_std_name]], on='영업장명_메뉴명', how='left')

        X_test_temp.fillna(0, inplace=True)
        
        # 모델별 피처 순서 맞추기
        lag_features_names = [f'lag_{lag+GAP+i-1}' for lag in lags]
        rolling_features_names = []
        for window in windows:
            rolling_features_names.extend([f'rolling_mean_{window}_lag_{GAP+i-1}', f'rolling_std_{window}_lag_{GAP+i-1}'])
        final_features_for_model = base_features + lag_features_names + rolling_features_names
        X_test = X_test_temp[final_features_for_model]
        
        # 예측
        model = models[f'model_d{i}']
        predictions = model.predict(X_test)
        predictions[predictions < 0] = 0

        pred_date = last_date + pd.to_timedelta(i, unit='D')
        temp_df = pd.DataFrame({
            '영업일자': pred_date,
            '영업장명_메뉴명': original_item_names.map({v: k for k, v in encoders['영업장명_메뉴명'].items()}), # Label을 다시 원래 이름으로
            '매출수량': np.round(predictions)
        })
        all_preds.append(temp_df)

# 최종 제출 파일 생성
final_submission_df = pd.concat(all_preds, ignore_index=True)
submission_df = final_submission_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()
submission_df['영업일자'] = submission_df['영업일자'].dt.strftime('%Y-%m-%d')
submission_df = submission_df.reindex(columns=submission_template.columns, fill_value=0)

submission_df.to_csv('./submission.csv', index=False)
print("submission.csv 파일 생성이 완료되었습니다.")