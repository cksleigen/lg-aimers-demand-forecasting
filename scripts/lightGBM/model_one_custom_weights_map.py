import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- 사용자가 제공한 공휴일 목록 정의 ---
# 문자열 리스트를 datetime 객체로 변환하여 효율적인 조회를 위해 Set으로 저장
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


# --- 1. 데이터 로드 ---
print("데이터 로드를 시작합니다...")
train_df = pd.read_csv('./data/train/train.csv')
test_files = sorted(glob.glob('./data/test/*.csv'))
print(f"Train 데이터 로드 완료, Test 파일 {len(test_files)}개 로드 완료.")

# --- 2. 피처 엔지니어링 함수 정의 ---

def create_features(df, is_train=True):
    """
    데이터프레임을 입력받아 피처를 생성하는 함수.
    """
    # '영업장명_메뉴명'을 '영업장명'과 '메뉴명'으로 분리
    df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)

    # 날짜 관련 피처 생성
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df['요일'] = df['영업일자'].dt.dayofweek
    df['월'] = df['영업일자'].dt.month
    df['연중일자'] = df['영업일자'].dt.dayofyear

    # 주말 및 공휴일 관련 피처 생성
    df['주말여부'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)
    # 사용자가 제공한 공휴일 리스트를 사용
    df['공휴일여부'] = df['영업일자'].apply(lambda x: 1 if x in custom_holidays else 0)
    
    # '휴일여부'는 주말과 공휴일을 통합
    df['휴일여부'] = df.apply(lambda row: 1 if row['주말여부'] == 1 or row['공휴일여부'] == 1 else 0, axis=1)

    # 휴일 전날 여부 피처
    df['하루뒤_날짜'] = df['영업일자'] + pd.to_timedelta(1, unit='D')
    df['휴일전날여부'] = df['하루뒤_날짜'].apply(lambda x: 1 if (x in custom_holidays or x.dayofweek >= 5) else 0)
    df.drop(columns=['하루뒤_날짜'], inplace=True)

    # Lag & Rolling Window 피처 생성 (학습 데이터에만 적용)
    if is_train:
        df = df.sort_values(by=['영업장명_메뉴명', '영업일자'])
        grouped = df.groupby('영업장명_메뉴명')['매출수량']
        
        lags = [7, 14, 21, 28]
        for lag in lags:
            df[f'lag_{lag}'] = grouped.shift(lag)

        windows = [7, 14, 28]
        for window in windows:
            df[f'rolling_mean_{window}'] = grouped.rolling(window=window).mean().reset_index(0, drop=True)
            df[f'rolling_std_{window}'] = grouped.rolling(window=window).std().reset_index(0, drop=True)
        
        df.fillna(df.groupby('영업장명_메뉴명').transform('mean'), inplace=True)
        df.fillna(0, inplace=True)
            
    return df

print("피처 엔지니어링 함수 정의 완료.")

# --- 3. 학습 데이터 전처리 및 피처 생성 ---
print("학습 데이터 전처리를 시작합니다...")
train_df = create_features(train_df, is_train=True)

# 인코딩 처리
encoders = {}
categorical_features = ['영업장명', '메뉴명']
for feature in categorical_features:
    le = LabelEncoder()
    train_df[feature] = le.fit_transform(train_df[feature])
    encoders[feature] = le

# 요일은 One-Hot Encoding
train_df = pd.get_dummies(train_df, columns=['요일'], prefix='요일')
encoders['요일_columns'] = [col for col in train_df.columns if col.startswith('요일_')]

# Sine/Cosine Transformation
cyclical_features = {'월': 12, '연중일자': 365}
for feature, max_val in cyclical_features.items():
    train_df[f'{feature}_sin'] = np.sin(2 * np.pi * train_df[feature] / max_val)
    train_df[f'{feature}_cos'] = np.cos(2 * np.pi * train_df[feature] / max_val)
    train_df.drop(columns=[feature], inplace=True)

print("학습 데이터 전처리 완료.")


# --- 4. 모델 학습 ---
print("LightGBM 모델 학습을 시작합니다...")

# 가중치 설정
weights_map = {
    '미라시아': 7.71, '담하': 6.51, '연회장': 3.48, '라그로타': 3.44,
    '느티나무 셀프BBQ': 2.78, '화담숲주막': 1.43, '카페테리아': 1.31,
    '화담숲카페': 1.14, '포레스트릿': 1.00
}
encoded_to_name = {code: name for name, code in zip(encoders['영업장명'].classes_, encoders['영업장명'].transform(encoders['영업장명'].classes_))}
train_df['sample_weight'] = train_df['영업장명'].map(encoded_to_name).map(weights_map).fillna(1.0) # 혹시 모를 누락 업장은 1.0으로 처리

# 학습에 사용할 피처 정의
features_to_use = [
    '영업장명', '메뉴명',
    '휴일여부', '휴일전날여부',
    '월_sin', '월_cos', '연중일자_sin', '연중일자_cos',
    'lag_7', 'lag_14', 'lag_21', 'lag_28',
    'rolling_mean_7', 'rolling_std_7',
    'rolling_mean_14', 'rolling_std_14',
    'rolling_mean_28', 'rolling_std_28'
] + encoders['요일_columns']

X_train = train_df[features_to_use]
y_train = train_df['매출수량']
sample_weight = train_df['sample_weight']

# LightGBM 모델 파라미터
params = {
    'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 1000,
    'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 1, 'verbose': -1, 'n_jobs': -1, 'seed': 42
}

model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train, 
          sample_weight=sample_weight,
          eval_set=[(X_train, y_train)],
          eval_metric='mae',
          callbacks=[lgb.early_stopping(100, verbose=False)])

print("모델 학습 완료.")

# --- 5. 추론 및 제출 파일 생성 ---
print("추론 및 제출 파일 생성을 시작합니다...")
submission_df = pd.read_csv('./submission.csv')
submission_columns = submission_df.columns

all_preds_df = pd.DataFrame()

for test_file in tqdm(test_files, desc="Test 파일별 추론 진행"):
    test_df = pd.read_csv(test_file)
    
    # 추론할 7일의 날짜 생성
    last_date = pd.to_datetime(test_df['영업일자'].max())
    pred_dates = pd.to_datetime([last_date + pd.to_timedelta(i, unit='D') for i in range(1, 8)])
    
    # 7일간의 예측을 위해 28일치 history를 반복적으로 업데이트
    history_df = test_df.copy()

    for pred_date in pred_dates:
        # 현재 예측일에 대한 피처 생성용 데이터프레임
        pred_df = pd.DataFrame({'영업장명_메뉴명': history_df['영업장명_메뉴명'].unique()})
        pred_df['영업일자'] = pred_date
        
        # 피처 생성 (is_train=False)
        pred_df = create_features(pred_df, is_train=False)
        
        # Lag/Rolling 피처 생성 (history_df 기반으로)
        temp_history = pd.concat([train_df, history_df]).sort_values(by=['영업장명_메뉴명', '영업일자'])
        
        for lag in [7, 14, 21, 28]:
            lag_data = temp_history.groupby('영업장명_메뉴명')['매출수량'].apply(lambda x: x.iloc[-lag]).rename(f'lag_{lag}').reset_index()
            pred_df = pd.merge(pred_df, lag_data, on='영업장명_메뉴명', how='left')

        for window in [7, 14, 28]:
            mean_data = temp_history.groupby('영업장명_메뉴명')['매출수량'].apply(lambda x: x.tail(window).mean()).rename(f'rolling_mean_{window}').reset_index()
            std_data = temp_history.groupby('영업장명_메뉴명')['매출수량'].apply(lambda x: x.tail(window).std()).rename(f'rolling_std_{window}').reset_index()
            pred_df = pd.merge(pred_df, mean_data, on='영업장명_메뉴명', how='left')
            pred_df = pd.merge(pred_df, std_data, on='영업장명_메뉴명', how='left')
            
        pred_df.fillna(0, inplace=True)
            
        # 인코딩 적용
        for feature in ['영업장명', '메뉴명']:
            # 학습 시 없던 새로운 메뉴가 나오면 -1(unknown)으로 처리
            pred_df[feature] = pred_df[feature].apply(lambda x: encoders[feature].transform([x])[0] if x in encoders[feature].classes_ else -1)
        
        pred_df = pd.get_dummies(pred_df, columns=['요일'], prefix='요일')
        for col in encoders['요일_columns']:
            if col not in pred_df.columns:
                pred_df[col] = 0
        
        for feature, max_val in cyclical_features.items():
            pred_df[f'{feature}_sin'] = np.sin(2 * np.pi * pred_df[feature.split('_')[0]] / max_val)
            pred_df[f'{feature}_cos'] = np.cos(2 * np.pi * pred_df[feature.split('_')[0]] / max_val)
            pred_df.drop(columns=[feature.split('_')[0]], inplace=True)
        
        # 예측
        predictions = model.predict(pred_df[features_to_use])
        predictions[predictions < 0] = 0
        
        # 예측 결과를 history에 추가
        pred_df['매출수량'] = np.round(predictions) # 소수점 예측값을 정수로 변환
        history_df = pd.concat([history_df, pred_df[['영업일자', '영업장명_메뉴명', '매출수량']]], ignore_index=True)
        
        # 최종 제출용 데이터프레임에 추가
        all_preds_df = pd.concat([all_preds_df, pred_df[['영업일자', '영업장명_메뉴명', '매출수량']]], ignore_index=True)

# 제출 형식으로 변환
final_submission = all_preds_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()
final_submission = final_submission.rename(columns={'영업일자': '영업일자'})

# submission.csv의 컬럼 순서와 동일하게 맞춤
final_submission = final_submission.reindex(columns=submission_columns, fill_value=0)
final_submission['영업일자'] = final_submission['영업일자'].dt.strftime('%Y-%m-%d')


# 제출 파일 저장
final_submission.to_csv('./data/submission/lightGBM_CustomLoss_submission.csv', index=False)

print("lightGBM_CustomLoss_submission.csv 파일 생성이 완료되었습니다.")