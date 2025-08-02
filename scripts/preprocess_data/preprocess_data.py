import pandas as pd
import os
import datetime

# --- 1. 초기 설정값 및 상수 ---
INPUT_FILE_PATH = './data/train/train.csv'
OUTPUT_DIR = './data/final_train_datasets'
OUTPUT_FILENAME = 'preprocess_train.csv'

# 휴일 목록 (datetime 객체로 변환하여 사용)
HOLIDAYS = [
    pd.to_datetime('2023-01-01'), pd.to_datetime('2023-01-21'), pd.to_datetime('2023-01-22'),
    pd.to_datetime('2023-01-23'), pd.to_datetime('2023-01-24'), pd.to_datetime('2023-03-01'),
    pd.to_datetime('2023-05-01'), pd.to_datetime('2023-05-05'), pd.to_datetime('2023-05-27'),
    pd.to_datetime('2023-06-06'), pd.to_datetime('2023-08-15'), pd.to_datetime('2023-09-28'),
    pd.to_datetime('2023-09-29'), pd.to_datetime('2023-09-30'), pd.to_datetime('2023-10-02'),
    pd.to_datetime('2023-10-03'), pd.to_datetime('2023-10-09'), pd.to_datetime('2023-12-25'),
    pd.to_datetime('2024-01-01'), pd.to_datetime('2024-02-09'), pd.to_datetime('2024-02-10'),
    pd.to_datetime('2024-02-11'), pd.to_datetime('2024-02-12'), pd.to_datetime('2024-03-01'),
    pd.to_datetime('2024-04-19'), pd.to_datetime('2024-05-01'), pd.to_datetime('2024-05-05'),
    pd.to_datetime('2024-05-06'), pd.to_datetime('2024-05-15'), pd.to_datetime('2024-06-06'),
    pd.to_datetime('2024-08-15'), pd.to_datetime('2024-09-16'), pd.to_datetime('2024-09-17'),
    pd.to_datetime('2024-09-18'), pd.to_datetime('2024-10-01'), pd.to_datetime('2024-10-03'),
    pd.to_datetime('2024-10-09'), pd.to_datetime('2024-12-25'),
    pd.to_datetime('2025-01-01'), pd.to_datetime('2025-01-28'), pd.to_datetime('2025-01-29'),
    pd.to_datetime('2025-01-30'), pd.to_datetime('2025-03-01'), pd.to_datetime('2025-03-03'),
    pd.to_datetime('2025-05-01'), pd.to_datetime('2025-05-05'), pd.to_datetime('2025-05-06'),
    pd.to_datetime('2025-06-06'), pd.to_datetime('2025-08-15')
]

# --- 2. 기능별 함수 정의 ---

def load_data(input_path):
    """지정된 경로의 CSV 파일을 DataFrame으로 불러옵니다."""
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
        print(f"파일 {input_path}를 성공적으로 불러왔습니다.")
        return df
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {input_path}")
        return None
    except Exception as e:
        print(f"파일을 불러오는 중 예기치 않은 오류가 발생했습니다: {e}")
        return None

def clean_data(df):
    """데이터프레임의 문자열 열에 포함된 비정상적인 문자를 제거합니다."""
    if df is None: return None
    print("데이터 클리닝을 시작합니다...")
    df['영업장명_메뉴명'] = df['영업장명_메뉴명'].str.encode('utf-8', errors='replace').str.decode('utf-8')
    print("데이터 클리닝을 완료했습니다.")
    return df

def split_store_menu(df):
    """'영업장명_메뉴명' 열을 '영업장명', '메뉴명'으로 분리합니다."""
    if df is None: return None
    print("영업장명_메뉴명 열을 분리합니다...")
    df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
    df = df.drop('영업장명_메뉴명', axis=1)
    print("영업장명, 메뉴명 분리 완료.")
    return df

def extract_month(df):
    """'영업일자' 열에서 '월'만 추출하여 '월' 열을 추가합니다."""
    if df is None: return None
    print("영업일자에서 '월'을 추출합니다...")
    df['월'] = df['영업일자'].dt.month
    print("월 추출 완료.")
    return df

def add_day_features(df):
    """'영업일자'를 이용해 요일, 주중, 주말(일~목:주중, 금~토:주말) 열을 추가합니다."""
    if df is None: return None
    print("요일, 주중, 주말 정보를 추가합니다...")
    weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    df['요일'] = df['영업일자'].dt.weekday.apply(lambda x: weekdays[x])
    
    # 주중 정의 변경 (일요일=6, 월요일=0, ... 목요일=3) -> 주중
    # 금요일(4), 토요일(5) -> 주말
    df['주중'] = df['영업일자'].dt.weekday.isin([0, 1, 2, 3, 6]).astype(int)
    df['주말'] = df['영업일자'].dt.weekday.isin([4, 5]).astype(int)
    
    print("요일, 주중, 주말 정보 추가 완료.")
    return df

def add_consecutive_holiday_count(df, holidays):
    """연속된 공휴일이 며칠 남았는지 계산하여 열을 추가합니다."""
    if df is None: return None
    print("연속된 공휴일 남은 일수 계산...")
    df['공휴일'] = df['영업일자'].isin(holidays).astype(int)
    holidays_df = df[df['공휴일'] == 1].copy()
    holidays_df = holidays_df.sort_values(by='영업일자').reset_index()

    consecutive_holidays = {}
    for i in range(len(holidays_df)):
        start_date = holidays_df.loc[i, '영업일자']
        consecutive_count = 1
        for j in range(i + 1, len(holidays_df)):
            if (holidays_df.loc[j, '영업일자'] - holidays_df.loc[j-1, '영업일자']).days == 1:
                consecutive_count += 1
            else:
                break
        for k in range(i, i + consecutive_count):
            consecutive_holidays[holidays_df.loc[k, '영업일자']] = consecutive_count - (k - i)
            
    df['연속된_공휴일_남은일수'] = df['영업일자'].map(consecutive_holidays).fillna(0)
    print("연속된 공휴일 남은 일수 계산 완료.")
    return df

def add_peak_season(df):
    """성수기(7/15~8/23, 10/25~11/8, 12/15~2/29) 여부를 판단하는 열을 추가합니다."""
    if df is None: return None
    print("성수기 여부 판단...")
    df['성수기'] = 0
    # 여름 성수기
    summer_start_end = [datetime.date(1, 7, 15), datetime.date(1, 8, 23)]
    df.loc[(df['영업일자'].dt.date >= summer_start_end[0]) & (df['영업일자'].dt.date <= summer_start_end[1]), '성수기'] = 1
    # 가을 성수기
    autumn_start_end = [datetime.date(1, 10, 25), datetime.date(1, 11, 8)]
    df.loc[(df['영업일자'].dt.date >= autumn_start_end[0]) & (df['영업일자'].dt.date <= autumn_start_end[1]), '성수기'] = 1
    # 겨울 성수기 (연도 경계 처리)
    winter_condition = (df['영업일자'].dt.month.isin([12, 1, 2])) & \
                       ((df['영업일자'].dt.month == 12) & (df['영업일자'].dt.day >= 15) | \
                        (df['영업일자'].dt.month == 1) | \
                        ((df['영업일자'].dt.month == 2) & (df['영업일자'].dt.day <= 29)))
    df.loc[winter_condition, '성수기'] = 1
    print("성수기 여부 판단 완료.")
    return df

def add_season(df):
    """월을 기준으로 시즌(봄, 여름, 가을, 겨울) 열을 추가합니다."""
    if df is None: return None
    print("시즌 정보를 추가합니다...")
    bins = [0, 3, 6, 9, 12]
    labels = ['겨울', '봄', '여름', '가을']
    df['시즌'] = pd.cut(df['영업일자'].dt.month, bins=bins, labels=labels, right=False)
    # 12, 1, 2, 3월은 겨울이므로 12월을 겨울로 수동 재정의
    df.loc[df['영업일자'].dt.month.isin([12, 1, 2, 3]), '시즌'] = '겨울'
    print("시즌 정보 추가 완료.")
    return df

def save_data(df, output_dir, output_filename):
    """전처리된 DataFrame을 지정된 경로에 CSV 파일로 저장합니다."""
    if df is None: return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_filename)
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n데이터가 {output_path}에 성공적으로 저장되었습니다.")
        print("최종 데이터프레임 열:", df.columns.tolist())
        print("최종 데이터프레임 미리보기:\n", df.head())
    except Exception as e:
        print(f"오류: 파일 저장 중 문제가 발생했습니다: {e}")

# --- 3. 메인 실행 파이프라인 ---
if __name__ == "__main__":
    # 데이터 전처리 파이프라인 함수 리스트
    # (함수명, 인자 딕셔너리) 튜플 형태로 구성
    preprocessing_pipeline = [
        # (clean_data, {}),                            # 문자열 데이터 클리닝
        (split_store_menu, {}),                      # '영업장명_메뉴명' 분리
        (extract_month, {}),                         # '월' 추출
        (add_day_features, {}),                      # '요일', '주중', '주말' 추가
        (add_consecutive_holiday_count, {'holidays': HOLIDAYS}), # 연속된 공휴일 계산
        (add_peak_season, {}),                       # 성수기 여부 판단
        (add_season, {}),                            # 시즌 정보 추가
    ]

    # 데이터 로드
    df = load_data(INPUT_FILE_PATH)
    
    if df is not None:
        # 데이터프레임의 '영업일자'를 datetime 객체로 변환
        df['영업일자'] = pd.to_datetime(df['영업일자'], format='%Y-%m-%d')
        
        # 파이프라인 순차 실행
        for process_func, kwargs in preprocessing_pipeline:
            # 주석 처리된 함수는 실행되지 않고 건너뜁니다.
            df = process_func(df, **kwargs)
            if df is None:
                print(f"함수 {process_func.__name__} 실행 중 오류가 발생하여 중단합니다.")
                break
        
        # 최종 데이터 저장
        save_data(df, OUTPUT_DIR, OUTPUT_FILENAME)