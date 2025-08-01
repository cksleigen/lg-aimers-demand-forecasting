import pandas as pd
import os

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
        try:
            df = pd.read_csv(input_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding='euc-kr')

        print(f"파일 {input_path}를 성공적으로 불러왔습니다.")
        return df
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {input_path}")
        return None
    except Exception as e:
        print(f"파일을 불러오는 중 예기치 않은 오류가 발생했습니다: {e}")
        return None

def add_date_features(df, holidays):
    """DataFrame에 날짜 관련 새로운 특성(요일, 주중, 주말, 공휴일)을 추가합니다."""
    if df is None:
        return None

    df['영업일자'] = pd.to_datetime(df['영업일자'], format='%Y-%m-%d')
    
    # 요일 생성 (시스템 로케일 의존성 제거)
    weekdays = ['월', '화', '수', '목', '금', '토', '일']
    df['요일'] = df['영업일자'].dt.weekday.apply(lambda x: weekdays[x])

    # 주중(월~금)과 주말(토~일) 추가
    df['주중'] = df['영업일자'].apply(lambda x: 1 if x.weekday() < 5 else 0)
    df['주말'] = df['영업일자'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
    
    # 공휴일 여부 추가
    df['공휴일'] = df['영업일자'].isin(holidays).astype(int)

    print("날짜 관련 특성(요일, 주중, 주말, 공휴일)을 성공적으로 추가했습니다.")
    return df

def save_data(df, output_dir, output_filename):
    """전처리된 DataFrame을 지정된 경로에 CSV 파일로 저장합니다."""
    if df is None:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, output_filename)
    
    try:
        # 먼저 안전하게 저장해봅니다.
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n데이터가 {output_path}에 성공적으로 저장되었습니다.")
        print("최종 데이터프레임 열:", df.columns.tolist())
        print("최종 데이터프레임 미리보기:\n", df.head())
    except UnicodeEncodeError as e:
        print("\n--- 저장 오류 감지 ---")
        print("파일 저장 중 유니코드 인코딩 오류가 발생했습니다.")
        print("문제의 원인을 찾기 위해, 오류가 발생한 문자열을 '?'로 대체하여 다시 저장합니다.")

        # 오류를 일으킨 문자열을 '?'로 대체하여 다시 저장
        temp_df = df.copy()
        
        # 모든 문자열(object) 타입의 열을 순회하며 문제 문자열을 찾고 대체
        for column in temp_df.select_dtypes(include=['object']).columns:
            temp_df[column] = temp_df[column].astype(str).apply(
                lambda x: x.encode('utf-8', errors='replace').decode('utf-8')
            )
        
        # 다시 저장 시도
        temp_output_path = os.path.join(output_dir, f"debug_{output_filename}")
        temp_df.to_csv(temp_output_path, index=False, encoding='utf-8-sig')
        
        print(f"오류가 발생한 데이터가 대체된 파일이 {temp_output_path}에 저장되었습니다.")
        print("이 파일을 열어보면, '?'로 표시된 부분이 문제의 원인이었던 데이터입니다.")
        print("원본 데이터에서 해당 부분을 찾아 수정하거나, 전처리 과정에서 제거해야 합니다.")
    except Exception as e:
        print(f"오류: 파일 저장 중 예기치 않은 문제가 발생했습니다: {e}")

# --- 3. 메인 실행 흐름 ---
if __name__ == "__main__":
    train_df = load_data(INPUT_FILE_PATH)
    
    if train_df is not None:
        processed_df = add_date_features(train_df, HOLIDAYS)
        
        # 문제 데이터를 자동으로 처리하면서, 어떤 부분이 문제였는지 알려줍니다.
        save_data(processed_df, OUTPUT_DIR, OUTPUT_FILENAME)