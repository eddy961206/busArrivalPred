import cudf
import numpy as np
from tqdm import tqdm
import pandas as pd

# 지정된 시간대 목록
time_slots = [("07:00", "09:00"), ("13:00", "15:00"), ("18:00", "20:00")]

# 시간대를 분 단위로 쪼개는 함수
def generate_time_range(start, end):
    times = cudf.Series(pd.date_range(start=start, end=end, freq='min').strftime('%H:%M').tolist())
    return times

# 모든 시간대에 해당하는 시간 목록 생성
all_times = cudf.Series()
for start, end in time_slots:
    all_times = cudf.concat([all_times, generate_time_range(start, end)], ignore_index=True)

# 원본 데이터 로드
test_csv = '/dataset/combined_dwell.csv'
dtype_spec = {
    'DAY_TYPE': 'int8',
    'BUSROUTE_ID': 'str',
    'LOCATION_ID': 'str',
    'DEP_TIME': 'str',
    'TIME_GAP': 'int32'
}

# cudf로 데이터 로드
test_data = cudf.read_csv(test_csv, dtype=dtype_spec)

# 1. 원본 데이터 중에서 time_slots에 속하는 데이터만 필터링
filtered_data = cudf.DataFrame()
for start, end in time_slots:
    filtered_data = cudf.concat([filtered_data, test_data[(test_data['DEP_TIME'] >= start) & (test_data['DEP_TIME'] <= end)]], ignore_index=True)

# 2. 새롭게 생성된 데이터 (time_slots의 모든 시간에 대해 모든 BUSROUTE_ID, LOCATION_ID 조합)
new_rows = []
unique_combinations = test_data[['BUSROUTE_ID', 'LOCATION_ID']].drop_duplicates().to_pandas()

# tqdm을 사용하여 진행 상황을 표시
for i, row in tqdm(unique_combinations.iterrows(), total=unique_combinations.shape[0], desc="노선-정류장 조합 처리 중"):
    busroute_id = row['BUSROUTE_ID']
    location_id = row['LOCATION_ID']

    # 해당 노선과 정류장의 데이터에서 첫 번째 DAY_TYPE을 가져옴
    day_type = test_data[(test_data['BUSROUTE_ID'] == busroute_id) & (test_data['LOCATION_ID'] == location_id)]['DAY_TYPE'].values[0]

    # 모든 시간대에 대해 새로운 행 생성
    for time in all_times.to_pandas():
        # 원본 데이터에 없는 시간대만 추가 (즉, 중복 방지)
        if not ((filtered_data['BUSROUTE_ID'] == busroute_id) & (filtered_data['LOCATION_ID'] == location_id) & (filtered_data['DEP_TIME'] == time)).any():
            new_rows.append({
                'DAY_TYPE': day_type,
                'BUSROUTE_ID': busroute_id,
                'LOCATION_ID': location_id,
                'DEP_TIME': time,
                'TIME_GAP': -1  # 새로운 데이터이므로 TIME_GAP은 -1로 설정
            })

# 새로 생성된 데이터프레임
new_data = cudf.DataFrame(new_rows)

# 3. 실제 데이터와 새로 생성된 데이터를 결합
combined_data = cudf.concat([filtered_data, new_data], ignore_index=True)

# 4. DEP_TIME을 기준으로 오름차순 정렬
combined_data = combined_data.sort_values(['DEP_TIME', 'BUSROUTE_ID', 'LOCATION_ID'], ascending=[True, True, True])

# 5. 결과 CSV 파일로 저장
combined_data.to_csv('/dataset/combined_dwell_1min.csv', index=False)
print("새로운 CSV 파일 저장 완료: combined_dwell_1min.csv")
