import pandas as pd
from datetime import timedelta

# 1) CSV 읽기
df = pd.read_csv('./data/lightGBM_model7_weight_season_add_optuna_SMAPE_submission.csv')

# 2) 영업일자 → datetime
df['영업일자'] = pd.to_datetime(df['영업일자'])

# 3) 해당 날짜가 속한 주(일요일~토요일) 내 '주 시작일(일요일)' 계산
#    pandas .dt.weekday: 월=0, …, 일=6
#    (weekday+1)%7 을 빼면 그 주의 '일요일'이 나옵니다.
df['week_start'] = df['영업일자'] - pd.to_timedelta((df['영업일자'].dt.weekday + 1) % 7, unit='d')

# 4) 고유한 주 시작일을 오름차순 정렬해 인덱스(0부터) 부여
week_starts = sorted(df['week_start'].unique())
week_map = {ws: idx for idx, ws in enumerate(week_starts)}

df['group_idx'] = df['week_start'].map(week_map)

# 5) 영업일자와 주 시작일의 차이에 +1 → 주 내 몇 번째 날인지(1~7)
df['day_offset'] = (df['영업일자'] - df['week_start']).dt.days + 1

# 6) 최종 변환된 문자열 생성
df['변환된_영업일자'] = df.apply(
    lambda r: f"TEST_{r['group_idx']:02d}+{r['day_offset']}일", axis=1
)

# 7) 중간 컬럼 제거
df = df.drop(['week_start', 'group_idx', 'day_offset'], axis=1)

# 8) 결과를 CSV로 저장
df.to_csv('output.csv', index=False)
