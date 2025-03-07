import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm

# 지정된 시간대 목록
time_slots = [("07:00", "09:00"), ("13:00", "15:00"), ("18:00", "20:00")]

# 시간대를 분 단위로 쪼개는 함수
def generate_time_range(start, end):
    times = pd.date_range(start=start, end=end, freq='min').strftime('%H:%M').tolist()
    return times

# 모든 시간대에 해당하는 시간 목록 생성
all_times = []
for start, end in time_slots:
    all_times.extend(generate_time_range(start, end))

# 원본 데이터 로드
test_csv = '/dataset/travel_time_7,8_TUE_filtered_70_sl_test.csv'
dtype_spec = {
    'DAY_TYPE': 'int8',
    'BUSINFOUNIT_ID': 'str',
    'LEN': 'int32',
    'DEP_TIME': 'str',
    'TIME_GAP': 'int32',
    'speed_kmh': 'int8'
}
test_data = pd.read_csv(test_csv, skipinitialspace=True, dtype=dtype_spec)

# 1. 원본 데이터 중에서 time_slots에 속하는 데이터만 필터링
# DEP_TIME이 time_slots에 속하는 경우를 필터링하기 위한 조건 생성
filtered_data = pd.DataFrame()
for start, end in time_slots:
    filtered_data = pd.concat([filtered_data, test_data[(test_data['DEP_TIME'] >= start) & (test_data['DEP_TIME'] <= end)]])

# 2. 새롭게 생성된 데이터 (time_slots의 모든 시간에 대해 모든 구간ID)
new_rows = []
unique_bus_ids = test_data['BUSINFOUNIT_ID'].unique()

# tqdm을 사용하여 진행 상황을 표시
for bus_id in tqdm(unique_bus_ids, desc="구간ID 처리 중"):
    # 해당 구간의 LEN 값을 가져오기
    len_value = test_data[test_data['BUSINFOUNIT_ID'] == bus_id]['LEN'].values[0]
    day_type = test_data[test_data['BUSINFOUNIT_ID'] == bus_id]['DAY_TYPE'].values[0]

    # 모든 시간대에 대해 새로운 행 생성
    for time in all_times:
        # 원본 데이터에 없는 시간대만 추가 (즉, 중복 방지)
        if not ((filtered_data['BUSINFOUNIT_ID'] == bus_id) & (filtered_data['DEP_TIME'] == time)).any():
            new_rows.append({
                'DAY_TYPE': day_type,
                'BUSINFOUNIT_ID': bus_id,
                'LEN': len_value,
                'DEP_TIME': time,
                'TIME_GAP': -1,  # 새로운 데이터이므로 TIME_GAP은 -1로 설정
                'speed_kmh': 0  # 새로운 데이터이므로 speed_kmh는 0으로 설정
            })

# 새로 생성된 데이터프레임
new_data = pd.DataFrame(new_rows)

# 3. 실제 데이터와 새로 생성된 데이터를 결합
combined_data = pd.concat([filtered_data, new_data], ignore_index=True)

# 4. DEP_TIME을 기준으로 오름차순 정렬
combined_data = combined_data.sort_values('DEP_TIME')

# 5. 결과 CSV 파일로 저장
combined_data.to_csv('/dataset/travel_time_7,8_TUE_filtered_70_sl_test_min.csv', index=False)
print("새로운 CSV 파일 저장 완료: travel_time_7,8_TUE_filtered_70_sl_test_min.csv")



import cudf
import numpy as np
from itertools import product
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
test_csv = '/dataset/travel_time_7,8_TUE_filtered_70_sl_test.csv'
dtype_spec = {
    'DAY_TYPE': 'int8',
    'BUSINFOUNIT_ID': 'str',
    'LEN': 'int32',
    'DEP_TIME': 'str',
    'TIME_GAP': 'int32',
    'speed_kmh': 'int8'
}

# cudf로 데이터 로드
test_data = cudf.read_csv(test_csv, dtype=dtype_spec)

# 1. 원본 데이터 중에서 time_slots에 속하는 데이터만 필터링
filtered_data = cudf.DataFrame()
for start, end in time_slots:
    filtered_data = cudf.concat([filtered_data, test_data[(test_data['DEP_TIME'] >= start) & (test_data['DEP_TIME'] <= end)]], ignore_index=True)

# 2. 새롭게 생성된 데이터 (time_slots의 모든 시간에 대해 모든 구간ID)
new_rows = []
unique_bus_ids = test_data['BUSINFOUNIT_ID'].unique()

# tqdm을 사용하여 진행 상황을 표시
for bus_id in tqdm(unique_bus_ids.to_pandas(), desc="구간ID 처리 중"):
    # 해당 구간의 LEN 값을 가져오기
    len_value = test_data[test_data['BUSINFOUNIT_ID'] == bus_id]['LEN'].values[0]
    day_type = test_data[test_data['BUSINFOUNIT_ID'] == bus_id]['DAY_TYPE'].values[0]

    # 모든 시간대에 대해 새로운 행 생성
    for time in all_times.to_pandas():
        # 원본 데이터에 없는 시간대만 추가 (즉, 중복 방지)
        if not ((filtered_data['BUSINFOUNIT_ID'] == bus_id) & (filtered_data['DEP_TIME'] == time)).any():
            new_rows.append({
                'DAY_TYPE': day_type,
                'BUSINFOUNIT_ID': bus_id,
                'LEN': len_value,
                'DEP_TIME': time,
                'TIME_GAP': -1,  # 새로운 데이터이므로 TIME_GAP은 -1로 설정
                'speed_kmh': 0  # 새로운 데이터이므로 speed_kmh는 0으로 설정
            })

# 새로 생성된 데이터프레임
new_data = cudf.DataFrame(new_rows)

# 3. 실제 데이터와 새로 생성된 데이터를 결합
combined_data = cudf.concat([filtered_data, new_data], ignore_index=True)

# 4. DEP_TIME을 기준으로 오름차순 정렬
combined_data = combined_data.sort_values('DEP_TIME')

# 5. 결과 CSV 파일로 저장
combined_data.to_csv('/dataset/travel_time_7,8_TUE_filtered_70_sl_test_min.csv', index=False)
print("새로운 CSV 파일 저장 완료: travel_time_7,8_TUE_filtered_70_sl_test_min.csv")
