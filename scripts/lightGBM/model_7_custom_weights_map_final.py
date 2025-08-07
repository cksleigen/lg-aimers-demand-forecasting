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

# 파이썬에서 발생하는 경고 메시지를 무시하여 코드가 깔끔하게 실행되도록 합니다.
warnings.filterwarnings('ignore')

# 사용자가 제공한 공휴일 목록을 리스트로 정의합니다.
# 나중에 빠른 조회를 위해 set 자료구조로 변환하여 사용합니다.
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
# 학습 데이터와 테스트 데이터 파일들의 경로를 지정하여 로드합니다.
train_df = pd.read_csv('./data/train/train.csv')
test_files = sorted(glob.glob('./data/test/*.csv'))
# 최종 제출 파일의 컬럼 순서와 형식을 맞추기 위해 submission.csv 파일을 '틀'로 사용합니다.
submission_template = pd.read_csv('./data/sample_submission.csv')
print(f"Train 데이터 로드 완료, Test 파일 {len(test_files)}개 로드 완료.")

# ==============================================================================
# 2. 피처 엔지니어링 함수 정의
# ==============================================================================
def create_base_features(df):
    """
    데이터프레임을 입력받아 모델 학습에 필요한 기본적인 피처들을 생성하는 함수입니다.
    이 함수는 날짜, ID, 이벤트 관련 피처를 만듭니다.
    """
    # '영업장명_메뉴명' 컬럼을 '_' 기준으로 분리하여 '영업장명'과 '메뉴명' 피처를 생성합니다.
    df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
    
    # '영업일자' 컬럼을 날짜 타입으로 변환하고, 이를 바탕으로 다양한 시간 관련 피처를 추출합니다.
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df['요일'] = df['영업일자'].dt.dayofweek  # 월요일=0, 화요일=1, ..., 일요일=6
    df['월'] = df['영업일자'].dt.month
    df['연중일자'] = df['영업일자'].dt.dayofyear # 1년 중 몇 번째 날인지 (1 ~ 365)
    
    # 주말 및 공휴일 관련 피처를 생성합니다. 리조트 수요 예측에 매우 중요한 정보입니다.
    df['주말여부'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0) # 토요일(5) 또는 일요일(6)이면 1
    df['공휴일여부'] = df['영업일자'].apply(lambda x: 1 if x in custom_holidays else 0)
    df['휴일여부'] = df.apply(lambda row: 1 if row['주말여부'] == 1 or row['공휴일여부'] == 1 else 0, axis=1)
    
    # 휴일 바로 전날인지 여부를 나타내는 피처를 생성합니다.
    df['하루뒤_날짜'] = df['영업일자'] + pd.to_timedelta(1, unit='D')
    df['휴일전날여부'] = df['하루뒤_날짜'].apply(lambda x: 1 if (x in custom_holidays or x.dayofweek >= 5) else 0)
    df.drop(columns=['하루뒤_날짜'], inplace=True) # 보조 컬럼 제거
    
    return df

print("피처 엔지니어링 함수 정의 완료.")

# ==============================================================================
# 3. 학습 데이터 전처리 및 인코딩
# ==============================================================================
print("학습 데이터 전처리를 시작합니다...")
# 기본 피처 생성
train_df = create_base_features(train_df)
# 메뉴별로 데이터를 정렬하여 시계열 분석을 준비합니다.
train_df = train_df.sort_values(by=['영업장명_메뉴명', '영업일자'])

# 7개의 미래 타겟(y) 변수 생성
# grouped.shift(-i)는 i일 뒤의 매출수량 값을 현재 행으로 가져옵니다.
grouped = train_df.groupby('영업장명_메뉴명')['매출수량']
for i in range(1, 8):
    train_df[f'target_d{i}'] = grouped.shift(-i)

# 인코딩: 머신러닝 모델이 이해할 수 있도록 모든 데이터를 숫자 형태로 변환합니다.
encoders = {}
# Label Encoding: '영업장명', '메뉴명'과 같이 고유값이 많은 문자열을 숫자로 변환 (e.g., '담하'->0)
categorical_features = ['영업장명', '메뉴명']
for feature in categorical_features:
    le = LabelEncoder()
    train_df[feature] = le.fit_transform(train_df[feature])
    encoders[feature] = le # 나중에 test 데이터에 적용하기 위해 변환 규칙 저장
# One-Hot Encoding: '요일'처럼 서열이 없는 범주를 0과 1로 이루어진 여러 컬럼으로 변환
train_df = pd.get_dummies(train_df, columns=['요일'], prefix='요일')
encoders['요일_columns'] = [col for col in train_df.columns if col.startswith('요일_')]
# Sine/Cosine Transformation: '월', '연중일자'처럼 순환하는 데이터의 특성을 보존하기 위해 변환
cyclical_features = {'월': 12, '연중일자': 365}
for feature, max_val in cyclical_features.items():
    train_df[f'{feature}_sin'] = np.sin(2 * np.pi * train_df[feature] / max_val)
    train_df[f'{feature}_cos'] = np.cos(2 * np.pi * train_df[feature] / max_val)
    train_df.drop(columns=[feature], inplace=True)
print("학습 데이터 전처리 완료.")

# ==============================================================================
# 4. 모델 7개 학습 (최종 전략 적용)
# ==============================================================================
print("LightGBM 모델 7개 학습을 시작합니다 (최종 전략 적용)...")
# 대회 규칙에 따라 특정 영업장에 가중치를 부여합니다.
weights_map = {
    '미라시아': 7.71, '담하': 6.51, '연회장': 3.48, '라그로타': 3.44,
    '느티나무 셀프BBQ': 2.78, '화담숲주막': 1.43, '카페테리아': 1.31,
    '화담숲카페': 1.14, '포레스트릿': 1.00
}
encoded_to_name = {code: name for name, code in zip(encoders['영업장명'].classes_, encoders['영업장명'].transform(encoders['영업장명'].classes_))}
train_df['sample_weight'] = train_df['영업장명'].map(encoded_to_name).map(weights_map).fillna(1.0)

# 모든 모델이 공통으로 사용하는 기본 피처 목록을 정의합니다.
base_features = [
    '영업장명', '메뉴명', '휴일여부', '휴일전날여부',
    '월_sin', '월_cos', '연중일자_sin', '연중일자_cos'
] + encoders['요일_columns']

# 7개의 모델을 저장할 딕셔너리
models = {}

# for 루프를 통해 7개의 모델을 각각 학습시킵니다.
for i in tqdm(range(1, 8), desc="모델 학습 진행"):
    # --- 각 모델에 맞는 피처를 동적으로 생성 ---
    
    # 1. Lag 피처 생성: 예측일(D+i)로부터 7, 14, 21, 28일 전의 매출을 사용합니다.
    lags = [7, 14, 21, 28]
    lag_features = []
    for lag in lags:
        feature_name = f'lag_{lag}' # 피처 이름은 lag_7, lag_14 등으로 고정
        train_df[feature_name] = train_df.groupby('영업장명_메뉴명')['매출수량'].shift(lag + i - 1)
        lag_features.append(feature_name)

    # 2. Rolling 피처 생성: "예측 시점에서 알 수 있는 가장 최신의 과거 동향"을 사용합니다.
    windows = [7, 14, 21, 28]
    rolling_features = []
    grouped_rolling = train_df.groupby('영업장명_메뉴명')['매출수량']
    for window in windows:
        rolling_mean_name = f'rolling_mean_{window}'
        rolling_std_name = f'rolling_std_{window}'
        # shift(i-1)을 통해, D+i 예측에 D일 시점의 rolling 값을 사용합니다. (Data Leakage 방지)
        train_df[rolling_mean_name] = grouped_rolling.rolling(window=window).mean().shift(i-1).reset_index(0,drop=True)
        train_df[rolling_std_name] = grouped_rolling.rolling(window=window).std().shift(i-1).reset_index(0,drop=True)
        rolling_features.extend([rolling_mean_name, rolling_std_name])
    
    # 현재 모델 학습에 사용할 전체 피처 목록
    current_features = base_features + lag_features + rolling_features
    
    # --- 학습 데이터 준비 ---
    # 각 모델에 필요한 피처와 타겟에 대해서만 결측치를 제거하여 학습 데이터를 최대화합니다.
    cols_for_dropna = current_features + [f'target_d{i}', 'sample_weight']
    train_temp = train_df[cols_for_dropna].dropna()
    
    X_train = train_temp[current_features]
    y_train = train_temp[f'target_d{i}']
    sample_weight = train_temp['sample_weight']

    # --- 모델 정의 및 학습 ---
    params = {
        'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 1000,
        'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'verbose': -1, 'n_jobs': -1, 'seed': 42
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, sample_weight=sample_weight,
              eval_set=[(X_train, y_train)], eval_metric='mae',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    models[f'model_d{i}'] = model # 학습된 모델 저장

print("모델 7개 학습 완료.")

# ==============================================================================
# 5. 추론 및 제출 파일 생성 (최종 전략 적용)
# ==============================================================================
print("추론 및 제출 파일 생성을 시작합니다...")
all_preds = []

# 각 테스트 파일을 순회하며 예측을 수행합니다.
for test_file in tqdm(test_files, desc="Test 파일별 추론 진행"):
    test_df = pd.read_csv(test_file)
    # 추론 시에는 오직 해당 28일치 데이터만 사용합니다.
    history_df = create_base_features(test_df.copy())
    history_df = history_df.sort_values(by=['영업장명_메뉴명', '영업일자'])
    
    # --- 추론에 사용할 공통 피처 (Rolling) 준비 ---
    grouped_rolling = history_df.groupby('영업장명_메뉴명')['매출수량']
    for window in windows:
        history_df[f'rolling_mean_{window}'] = grouped_rolling.rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
        history_df[f'rolling_std_{window}'] = grouped_rolling.rolling(window=window, min_periods=1).std().reset_index(0, drop=True)

    # 28일 중 가장 마지막 날의 데이터를 모델 입력의 기본 틀로 사용합니다.
    pred_input_df_base = history_df.groupby('영업장명_메뉴명').tail(1).reset_index(drop=True)
    
    # 인코딩 적용
    original_item_names = pred_input_df_base['영업장명_메뉴명'].copy()
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
    
    # 7개 모델로 각각 독립적으로 예측을 수행합니다.
    for i in range(1, 8):
        # --- 각 모델에 맞는 Lag 피처를 동적으로 생성 ---
        X_test_temp = pred_input_df_base.copy()
        for lag in lags:
            shift_val = lag + i - 1
            lag_data = history_df.groupby('영업장명_메뉴명')['매출수량'].apply(lambda x: x.iloc[-shift_val] if len(x) >= shift_val else np.nan).rename(f'lag_{lag}').reset_index()
            X_test_temp = pd.merge(X_test_temp, lag_data, on='영업장명_메뉴명', how='left')

        X_test_temp.fillna(0, inplace=True)
        
        # 학습 시 사용했던 피처 순서와 동일하게 맞춰줍니다.
        X_test = X_test_temp[base_features + rolling_features + lag_features]
        
        # 예측 수행
        model = models[f'model_d{i}']
        predictions = model.predict(X_test)
        predictions[predictions < 0] = 0 # 음수 예측값은 0으로 처리

        # 예측 결과를 날짜와 함께 저장합니다.
        pred_date = last_date + pd.to_timedelta(i, unit='D')
        temp_df = pd.DataFrame({
            '영업일자': pred_date,
            '영업장명_메뉴명': original_item_names,
            '매출수량': np.round(predictions)
        })
        all_preds.append(temp_df)

# 모든 예측 결과를 하나로 합치고 제출 형식으로 변환합니다.
final_submission_df = pd.concat(all_preds, ignore_index=True)
submission_df = final_submission_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()
submission_df['영업일자'] = submission_df['영업일자'].dt.strftime('%Y-%m-%d')
# 미리 로드해둔 submission_template의 컬럼 순서와 동일하게 맞춥니다.
submission_df = submission_df.reindex(columns=submission_template.columns, fill_value=0)

# 최종 제출 파일 저장
submission_df.to_csv('./data/lightGBM_custom_weights_map_final_submission.csv', index=False)
print("submission.csv 파일 생성이 완료되었습니다.")