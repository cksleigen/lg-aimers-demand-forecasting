import pandas as pd
import numpy as np

def convert_float_to_int(df):
    """
    DataFrame의 모든 float 타입 컬럼을 int 타입으로 변환합니다.
    NaN 값이 있는 경우, pandas의 nullable integer type인 'Int64'로 변환합니다.
    """
    for col in df.columns:
        # 컬럼의 데이터 타입이 float인지 확인
        if pd.api.types.is_float_dtype(df[col]):
            # NaN 값이 있는지 확인
            if df[col].isnull().any():
                # NaN 값을 포함하는 경우, nullable integer type으로 변환
                df[col] = df[col].astype('Int64')
            else:
                # NaN 값이 없는 경우, 일반 int type으로 변환
                df[col] = df[col].astype(int)
    return df

# 예시 사용법:
# 'your_file.csv'를 실제 파일명으로 바꾸세요.
try:
    df = pd.read_csv('./data/lightGBM_model7_weight_season_add_optuna_SMAPE_submission.csv')

    # 함수를 사용하여 float를 int로 변환
    df_converted = convert_float_to_int(df)

    # 변환된 DataFrame을 새로운 CSV 파일로 저장
    # 'converted_file.csv'를 원하는 파일명으로 바꾸세요.
    df_converted.to_csv('./data/lightGBM_model7_weight_season_add_optuna_SMAPE_submission_converted.csv', index=False)

    print("파일이 성공적으로 변환되어 'converted_file.csv'로 저장되었습니다.")
except FileNotFoundError:
    print("파일을 찾을 수 없습니다. 'your_file.csv' 파일이 존재하는지 확인하세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")