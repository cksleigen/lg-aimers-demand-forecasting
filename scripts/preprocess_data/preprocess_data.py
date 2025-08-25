import pandas as pd
import os
import datetime

# --- 1. 초기 설정값 및 상수 ---

# [사용자 설정] 원-핫 인코딩 실행 여부 (True 또는 False)
ONE_HOT_ENCODING = False  # True로 설정 시, 열 순서를 논리적으로 재배치함

INPUT_FILE_PATH = "./data/train/train.csv"
OUTPUT_DIR = "./data/final_train_datasets"
# 파일명에 원-핫 인코딩 여부를 포함시켜 구분이 용이하도록 함
OUTPUT_FILENAME = f"preprocessed_train_onehot_{str(ONE_HOT_ENCODING)}.csv"

# 공휴일 목록 (datetime.date 객체로 변환하여 사용)
HOLIDAYS = [
    pd.to_datetime("2023-01-01").date(),
    pd.to_datetime("2023-01-21").date(),
    pd.to_datetime("2023-01-22").date(),
    pd.to_datetime("2023-01-23").date(),
    pd.to_datetime("2023-01-24").date(),
    pd.to_datetime("2023-03-01").date(),
    pd.to_datetime("2023-05-01").date(),
    pd.to_datetime("2023-05-05").date(),
    pd.to_datetime("2023-05-27").date(),
    pd.to_datetime("2023-05-29").date(),
    pd.to_datetime("2023-06-06").date(),
    pd.to_datetime("2023-08-15").date(),
    pd.to_datetime("2023-09-28").date(),
    pd.to_datetime("2023-09-29").date(),
    pd.to_datetime("2023-09-30").date(),
    pd.to_datetime("2023-10-02").date(),
    pd.to_datetime("2023-10-03").date(),
    pd.to_datetime("2023-10-09").date(),
    pd.to_datetime("2023-12-25").date(),
    pd.to_datetime("2024-01-01").date(),
    pd.to_datetime("2024-02-09").date(),
    pd.to_datetime("2024-02-10").date(),
    pd.to_datetime("2024-02-11").date(),
    pd.to_datetime("2024-02-12").date(),
    pd.to_datetime("2024-03-01").date(),
    pd.to_datetime("2024-04-10").date(),
    pd.to_datetime("2024-05-01").date(),
    pd.to_datetime("2024-05-05").date(),
    pd.to_datetime("2024-05-06").date(),
    pd.to_datetime("2024-05-15").date(),
    pd.to_datetime("2024-06-06").date(),
    pd.to_datetime("2024-08-15").date(),
    pd.to_datetime("2024-09-16").date(),
    pd.to_datetime("2024-09-17").date(),
    pd.to_datetime("2024-09-18").date(),
    pd.to_datetime("2024-10-01").date(),
    pd.to_datetime("2024-10-03").date(),
    pd.to_datetime("2024-10-09").date(),
    pd.to_datetime("2024-12-25").date(),
    pd.to_datetime("2025-01-01").date(),
    pd.to_datetime("2025-01-28").date(),
    pd.to_datetime("2025-01-29").date(),
    pd.to_datetime("2025-01-30").date(),
    pd.to_datetime("2025-03-01").date(),
    pd.to_datetime("2025-03-03").date(),
    pd.to_datetime("2025-05-01").date(),
    pd.to_datetime("2025-05-05").date(),
    pd.to_datetime("2025-05-06").date(),
    pd.to_datetime("2025-06-06").date(),
    pd.to_datetime("2025-08-15").date(),
]
HOLIDAYS_SET = set(HOLIDAYS)  # 빠른 조회를 위해 set으로 변환

# --- 2. 기능별 함수 정의 ---


def load_and_prepare_data(input_path):
    """CSV 파일을 불러오고 날짜 형식 변환 등 기본 준비를 수행합니다."""
    try:
        df = pd.read_csv(input_path, encoding="utf-8")
        df["영업일자"] = pd.to_datetime(df["영업일자"])
        print(f"✅ 파일 로드 및 날짜 변환 성공: {input_path}")
        return df
    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {input_path}")
        return None


def split_store_and_menu(df):
    """'영업장명_메뉴명' 열을 '영업장명'과 '메뉴명'으로 분리합니다."""
    print("🚀 '영업장명', '메뉴명' 열을 생성합니다...")
    df[["영업장명", "메뉴명"]] = df["영업장명_메뉴명"].str.split("_", n=1, expand=True)
    return df


def add_temporal_features(df):
    """날짜 기반 파생변수(계절, 요일)를 추가합니다."""
    print("🚀 '계절', '요일' 열을 생성합니다...")

    def get_season(month):
        if 4 <= month <= 6:
            return "봄"
        elif 7 <= month <= 8:
            return "여름"
        elif 9 <= month <= 11:
            return "가을"
        else:
            return "겨울"

    df["계절"] = df["영업일자"].dt.month.apply(get_season)
    weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
    df["요일"] = df["영업일자"].dt.weekday.map(weekday_map)
    return df


def add_holiday_features(df):
    """휴일 관련 파생변수('쉬는날', '내일도 쉬어')를 0 또는 1로 추가합니다."""
    print("🚀 '쉬는날', '내일도 쉬어' 열을 0/1 값으로 생성합니다...")
    is_weekend = df["요일"].isin(["토", "일"])
    is_holiday = df["영업일자"].dt.date.isin(HOLIDAYS_SET)
    df["쉬는날"] = (is_weekend | is_holiday).astype(int)
    df_sorted = df.sort_values(by=["영업장명", "메뉴명", "영업일자"]).copy()
    df_sorted["내일도 쉬어"] = df_sorted.groupby(["영업장명", "메뉴명"])[
        "쉬는날"
    ].shift(-1)
    df_sorted["내일도 쉬어"] = df_sorted["내일도 쉬어"].fillna(0).astype(int)
    return df_sorted


def select_and_reorder_columns(df):
    """(인코딩 전) 최종 열을 선택하고 순서를 정리합니다."""
    print("🚀 최종 데이터 열을 선택하고 정리합니다...")
    required_cols = ["영업장명", "메뉴명", "계절", "요일", "쉬는날", "내일도 쉬어"]
    original_cols_to_keep = [
        col
        for col in df.columns
        if col not in required_cols + ["영업장명_메뉴명", "영업일자"]
    ]
    final_cols = required_cols + original_cols_to_keep
    return df[final_cols]


def one_hot_encode_features(df):
    """범주형 데이터를 원-핫 인코딩합니다."""
    print("🚀 문자열 범주형 데이터를 원-핫 인코딩합니다...")
    categorical_cols = ["영업장명", "메뉴명", "계절", "요일"]
    cols_to_encode = [col for col in categorical_cols if col in df.columns]
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, dtype=int)
    return df_encoded


def reorder_encoded_columns(df):
    """(신규) 원-핫 인코딩 후의 데이터 열 순서를 논리적으로 재정렬합니다."""
    print("🚀 원-핫 인코딩된 열의 순서를 재정렬합니다...")

    # 재정렬의 기준이 될 그룹 정의
    base_order_prefixes = ["영업장명", "메뉴명", "계절", "요일"]
    reordered_cols = []

    # 1. 그룹 순서에 따라 인코딩된 열 추가
    for prefix in base_order_prefixes:
        # sorted()를 사용해 그룹 내에서도 가나다순으로 정렬
        encoded_cols = sorted(
            [col for col in df.columns if col.startswith(prefix + "_")]
        )
        reordered_cols.extend(encoded_cols)

    # 2. '쉬는날', '내일도 쉬어' 추가
    reordered_cols.extend(["쉬는날", "내일도 쉬어"])

    # 3. 위에 포함되지 않은 나머지 열 (원본 데이터의 타겟 변수 등)을 뒤에 추가
    other_cols = [col for col in df.columns if col not in reordered_cols]
    reordered_cols.extend(other_cols)

    return df[reordered_cols]


def save_data(df, output_dir, output_filename):
    """전처리된 DataFrame을 CSV 파일로 저장합니다."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 디렉토리가 생성되었습니다: {output_dir}")
    output_path = os.path.join(output_dir, output_filename)
    try:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n🎉 데이터가 성공적으로 저장되었습니다: {output_path}")
        print("\n=== 최종 데이터 정보 ===")
        print("칼럼 목록:", df.columns.tolist())
        print("\n데이터 미리보기 (상위 5개):")
        print(df.head())
    except Exception as e:
        print(f"❌ 오류: 파일 저장 중 문제가 발생했습니다: {e}")


# --- 3. 메인 실행 파이프라인 ---
if __name__ == "__main__":
    # 1. 데이터 로드
    df = load_and_prepare_data(INPUT_FILE_PATH)

    if df is not None:
        # 2. 파생변수 생성
        df = split_store_and_menu(df)
        df = add_temporal_features(df)
        df = add_holiday_features(df)

        # 3. (인코딩 전) 열 선택 및 순서 정리
        df = select_and_reorder_columns(df)

        # 4. (선택적) 원-핫 인코딩 및 순서 재정렬
        if ONE_HOT_ENCODING:
            df = one_hot_encode_features(df)
            df = reorder_encoded_columns(df)  # 인코딩 후 순서 재정렬 함수 호출

        # 5. 데이터 저장
        save_data(df, OUTPUT_DIR, OUTPUT_FILENAME)
