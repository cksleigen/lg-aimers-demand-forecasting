# ==============================================================================
# 0. 사전 준비: 라이브러리 임포트 및 전역 변수 설정
# ==============================================================================
# 데이터 분석 및 처리에 필요한 라이브러리들을 불러옵니다.
import pandas as pd  # 데이터프레임(표 형태의 데이터)을 다루기 위한 라이브러리
import numpy as np   # 수치 계산, 특히 배열(array)을 다루기 위한 라이브러리
import lightgbm as lgb # LightGBM 머신러닝 모델을 사용하기 위한 라이브러리
from sklearn.preprocessing import LabelEncoder # 문자열 데이터를 숫자로 변환하기 위한 도구
import glob          # 특정 패턴의 파일 경로들을 쉽게 찾기 위한 라이브러리
from tqdm import tqdm # 반복문(for loop)의 진행 상황을 시각적으로 보여주는 라이브러리
import warnings      # 불필요한 경고 메시지를 제어하기 위한 라이브러리

# 파이썬에서 발생하는 경고 메시지를 무시하여 실행 결과가 깔끔하게 보이도록 합니다.
warnings.filterwarnings('ignore')

# 사용자가 제공한 공휴일 목록을 리스트로 정의합니다.
# 나중에 특정 날짜가 공휴일인지 아닌지를 매우 빠르게 조회하기 위해 set 자료구조로 변환하여 사용합니다.
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

# 학습 데이터에 존재하는 음수 '매출수량'은 모두 0으로 변환합니다.
train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
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
    
    # 계절 피처 생성
    def get_season(month):
        if month in [4, 5, 6]: return '봄'
        elif month in [7, 8]: return '여름'
        elif month in [9, 10, 11]: return '가을'
        else: return '겨울' # 12, 1, 2, 3월
    df['계절'] = df['월'].apply(get_season)
    
    return df

print("피처 엔지니어링 함수 정의 완료.")

# ==============================================================================
# 3. 학습 데이터 생성 (Sliding Window 방식)
# ==============================================================================
print("학습 데이터를 Sliding Window 방식으로 생성합니다... (시간이 소요될 수 있습니다)")
# 기본 피처들을 먼저 생성합니다.
train_df_processed = create_base_features(train_df)
# 메뉴별로 데이터를 시간순으로 정렬하여 시계열 분석을 준비합니다.
train_df_processed = train_df_processed.sort_values(by=['영업장명_메뉴명', '영업일자'])

# 인코딩을 미리 준비하여 학습 데이터 생성 시 바로 적용할 수 있도록 합니다.
encoders = {}
# Label Encoding: '영업장명', '메뉴명'과 같이 고유값이 많은 문자열을 숫자로 변환 (e.g., '담하'->0)
categorical_features = ['영업장명', '메뉴명']
for feature in categorical_features:
    le = LabelEncoder()
    train_df_processed[feature] = le.fit_transform(train_df_processed[feature])
    encoders[feature] = le # 나중에 test 데이터에 적용하기 위해 변환 규칙 저장

# One-Hot Encoding: '요일', '계절'처럼 서열이 없는 범주를 0과 1로 이루어진 여러 컬럼으로 변환
train_df_processed = pd.get_dummies(train_df_processed, columns=['요일', '계절'], prefix=['요일', '계절'])
ohe_columns = [col for col in train_df_processed.columns if col.startswith('요일_') or col.startswith('계절_')]
encoders['ohe_columns'] = ohe_columns


# 최종 학습 데이터를 담을 리스트를 초기화합니다.
training_samples = []

# 각 메뉴별로 루프를 돌며, 28일짜리 "창문(window)"을 하루씩 이동(sliding)시키며 학습 샘플을 생성합니다.
# 이 방식은 추론 환경과 학습 환경을 동일하게 만들어 모델 성능을 안정화시킵니다.
for item_id, group in tqdm(train_df_processed.groupby('영업장명_메뉴명'), desc="학습 샘플 생성 중"):
    # 최소한 (28일 입력 + 7일 타겟)의 데이터가 있어야 샘플 생성이 가능하므로, 짧은 이력을 가진 메뉴는 건너뜁니다.
    if len(group) < 28 + 7: continue
    
    # 28일 window를 하루씩 옆으로 이동시키면서 반복합니다.
    for i in range(len(group) - 28 - 7 + 1):
        # 입력 데이터 (28일)
        input_window = group.iloc[i : i+28]
        # 타겟(정답) 데이터 (입력 데이터 바로 뒤 7일)
        target_window = group.iloc[i+28 : i+28+7]
        
        # 28일 중 가장 마지막 날을 기준으로 피처를 생성합니다.
        features = input_window.tail(1).copy()

        # Rolling 피처 생성 (28일 window 내에서만 계산하여 추론 환경과 동일하게 만듭니다)
        windows = [7, 14, 21, 28]
        for window in windows:
            features[f'rolling_mean_{window}'] = input_window['매출수량'].rolling(window, min_periods=1).mean().iloc[-1]
            features[f'rolling_std_{window}'] = input_window['매출수량'].rolling(window, min_periods=1).std().iloc[-1]
        
        # Lag 피처 생성 (28일 window 내에서만 계산)
        lags = [7, 14, 21, 28]
        for lag in lags:
            if len(input_window) >= lag:
                features[f'lag_{lag}'] = input_window['매출수량'].iloc[-lag]
            else:
                features[f'lag_{lag}'] = np.nan

        # 7일치 타겟 값들을 features 행에 새로운 컬럼으로 추가합니다.
        for day in range(1, 8):
            features[f'target_d{day}'] = target_window['매출수량'].iloc[day-1]
            
        training_samples.append(features)

# 리스트에 쌓인 모든 학습 샘플들을 하나의 데이터프레임으로 변환합니다.
final_train_df = pd.concat(training_samples, ignore_index=True)
# 피처 생성 과정에서 생긴 결측치는 0으로 채웁니다.
final_train_df.fillna(0, inplace=True)

# Sine/Cosine Transformation: '월', '연중일자'처럼 순환하는 데이터의 특성을 보존하기 위해 변환
cyclical_features = {'월': 12, '연중일자': 365}
for feature, max_val in cyclical_features.items():
    final_train_df[f'{feature}_sin'] = np.sin(2 * np.pi * final_train_df[feature] / max_val)
    final_train_df[f'{feature}_cos'] = np.cos(2 * np.pi * final_train_df[feature] / max_val)
    final_train_df.drop(columns=[feature], inplace=True)

print("학습 데이터 생성 완료.")

# ==============================================================================
# 4. 모델 7개 학습
# ==============================================================================
print("LightGBM 모델 7개 학습을 시작합니다...")
# 대회 규칙에 따라 특정 영업장에 가중치를 부여합니다.
weights_map = {
    '미라시아': 7.71, '담하': 6.51, '연회장': 3.48, '라그로타': 3.44,
    '느티나무 셀프BBQ': 2.78, '화담숲주막': 1.43, '카페테리아': 1.31,
    '화담숲카페': 1.14, '포레스트릿': 1.00
}
encoded_to_name = {code: name for name, code in zip(encoders['영업장명'].classes_, encoders['영업장명'].transform(encoders['영업장명'].classes_))}
final_train_df['sample_weight'] = final_train_df['영업장명'].map(encoded_to_name).map(weights_map).fillna(1.0)

# 학습에 사용할 피처 목록 정의
base_features = [
    '영업장명', '메뉴명', '휴일여부', '휴일전날여부',
    '월_sin', '월_cos', '연중일자_sin', '연중일자_cos'
] + encoders['ohe_columns']
rolling_features = [f'rolling_mean_{w}' for w in [7, 14, 21, 28]] + [f'rolling_std_{w}' for w in [7, 14, 21, 28]]
lag_features = [f'lag_{l}' for l in [7, 14, 21, 28]]
current_features = base_features + rolling_features + lag_features

# 입력(X)과 가중치(sample_weight) 준비
X_train = final_train_df[current_features]
sample_weight = final_train_df['sample_weight']

# 7개의 모델을 저장할 딕셔너리
models = {}

# for 루프를 통해 7개의 모델을 각각의 타겟에 대해 학습시킵니다.
for i in tqdm(range(1, 8), desc="모델 학습 진행"):
    # 이번 루프에서 학습할 타겟(y)을 선택합니다 (e.g., 'target_d1', 'target_d2', ...)
    y_train = final_train_df[f'target_d{i}']
    
    # 모델의 하이퍼파라미터를 설정합니다.
    params = {
        'objective': 'regression_l1', # 예측값과 실제값의 차이(MAE)를 줄이는 것을 목표로 함
        'metric': 'mae',
        'n_estimators': 1000,         # 1000개의 트리를 만듦
        'learning_rate': 0.05,        # 학습 속도
        'feature_fraction': 0.8,      # 각 트리를 학습할 때 80%의 피처만 무작위로 사용 (과적합 방지)
        'bagging_fraction': 0.8,      # 각 트리를 학습할 때 80%의 데이터만 무작위로 사용 (과적합 방지)
        'bagging_freq': 1,
        'verbose': -1,                # 학습 과정 로그 출력 안 함
        'n_jobs': -1,                 # 모든 CPU 코어를 사용하여 학습 속도 향상
        'seed': 42                    # 재현성을 위한 시드 고정
    }
    
    model = lgb.LGBMRegressor(**params)
    # 모델 학습을 시작합니다.
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    models[f'model_d{i}'] = model # 학습된 모델을 저장

print("모델 7개 학습 완료.")

# ==============================================================================
# 5. 추론 및 제출 파일 생성
# ==============================================================================
print("추론 및 제출 파일 생성을 시작합니다...")
all_preds = []

# 각 테스트 파일을 순회하며 독립적으로 예측을 수행합니다.
for test_file in tqdm(test_files, desc="Test 파일별 추론 진행"):
    test_df = pd.read_csv(test_file)
    # 추론 시에는 오직 해당 28일치 데이터만 사용합니다.
    history_df = create_base_features(test_df.copy())
    history_df = history_df.sort_values(by=['영업장명_메뉴명', '영업일자'])
    
    # 추론에 사용할 Rolling 피처를 28일 history 내에서만 계산합니다.
    grouped_rolling = history_df.groupby('영업장명_메뉴명')['매출수량']
    for window in [7, 14, 21, 28]:
        history_df[f'rolling_mean_{window}'] = grouped_rolling.rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
        history_df[f'rolling_std_{window}'] = grouped_rolling.rolling(window=window, min_periods=1).std().reset_index(0, drop=True)

    # 28일 중 가장 마지막 날의 데이터를 모델 입력의 기본 틀로 사용합니다.
    pred_input_df_base = history_df.groupby('영업장명_메뉴명').tail(1).reset_index(drop=True)
    pred_input_df_base.fillna(0, inplace=True)
    
    original_item_names = pred_input_df_base['영업장명_메뉴명'].copy()
    # 인코딩 적용 (학습 시 사용했던 규칙 그대로)
    for feature in ['영업장명', '메뉴명']:
        # 학습 데이터에 없었던 새로운 메뉴가 나오면 -1(unknown)으로 처리합니다.
        pred_input_df_base[feature] = pred_input_df_base[feature].apply(lambda x: encoders[feature].transform([x])[0] if x in encoders[feature].classes_ else -1)
    
    pred_input_df_base = pd.get_dummies(pred_input_df_base, columns=['요일', '계절'], prefix=['요일', '계절'])
    # 학습 데이터에 있던 one-hot 컬럼이 test 데이터에 없는 경우를 대비해 모든 컬럼을 맞춰줍니다.
    for col in encoders['ohe_columns']:
        if col not in pred_input_df_base.columns:
            pred_input_df_base[col] = 0
            
    for feature, max_val in cyclical_features.items():
        original_col_name = '월' if '월' in feature else '연중일자'
        pred_input_df_base[f'{feature}_sin'] = np.sin(2 * np.pi * pred_input_df_base[original_col_name] / max_val)
        pred_input_df_base[f'{feature}_cos'] = np.cos(2 * np.pi * pred_input_df_base[original_col_name] / max_val)

    # Lag 피처를 28일 history 내에서만 계산하여 추가합니다.
    for lag in [7, 14, 21, 28]:
        lag_data = history_df.groupby('영업장명_메뉴명')['매출수량'].apply(lambda x: x.iloc[-lag] if len(x) >= lag else 0).rename(f'lag_{lag}').reset_index()
        pred_input_df_base = pd.merge(pred_input_df_base, lag_data, on='영업장명_메뉴명', how='left')

    # 최종 모델 입력 데이터(X_test) 준비 (학습 시 사용한 피처 순서와 동일하게)
    X_test = pred_input_df_base[current_features]
    last_date = pd.to_datetime(test_df['영업일자'].max())
    
    # 7개 모델로 각각 독립적으로 예측을 수행합니다.
    for i in range(1, 8):
        model = models[f'model_d{i}'] # D+i일을 예측하는 모델을 불러옵니다.
        predictions = model.predict(X_test)
        predictions[predictions < 0] = 0 # 음수 예측값은 0으로 처리합니다.

        # 예측 결과를 날짜와 함께 저장합니다.
        pred_date = last_date + pd.to_timedelta(i, unit='D')
        temp_df = pd.DataFrame({
            '영업일자': pred_date,
            '영업장명_메뉴명': original_item_names,
            '매출수량': np.round(predictions).astype(int) # 예측값을 정수로 변환
        })
        all_preds.append(temp_df)

# 모든 예측 결과를 하나로 합치고 제출 형식으로 변환합니다.
final_submission_df = pd.concat(all_preds, ignore_index=True)
# pivot: Long format 데이터(날짜, 메뉴, 값)를 Wide format(날짜, 메뉴1, 메뉴2, ...)으로 변환
submission_df = final_submission_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()
submission_df['영업일자'] = submission_df['영업일자'].dt.strftime('%Y-%m-%d')
# reindex: 제출 샘플 파일과 컬럼 순서 및 누락된 컬럼을 동일하게 맞춤
submission_df = submission_df.reindex(columns=submission_template.columns, fill_value=0)

# 최종 제출 파일 저장
submission_df.to_csv('./data/lightGBM_model7_weight_season_add_submission.csv', index=False)
print("lightGBM_model7_weight_season_add_submission.csv 파일 생성이 완료되었습니다.")