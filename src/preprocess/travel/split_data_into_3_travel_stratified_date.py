import cudf
import pandas as pd
import os

# 데이터 타입 사양 정의
dtype_spec = {
    'DAY_TYPE': 'int8',
    'BUSROUTE_ID': 'int32',  # 'BUSROUTE_ID' 추가 및 타입 지정
    'BUSINFOUNIT_ID': 'str',
    'LEN': 'int32',
    'COLLECT_DATE': 'str',    # 'COLLECT_DATE'는 문자열로 로드 후 변환
    'DEP_TIME': 'str',
    'TIME_GAP': 'int32',
    'SPEED': 'int32'
}

# CSV 파일 경로
csv = '../../../dataset/train/travel/241004_노선모두_8.28~9.11_평일_속도X/소통_train_filtered.csv'

# cuDF DataFrame으로 CSV 로드
df = cudf.read_csv(csv, dtype=dtype_spec)

# 'COLLECT_DATE'를 datetime 타입으로 변환
df['COLLECT_DATE'] = df['COLLECT_DATE'].str.slice(0, 10)  # 날짜 부분 추출 (YYYY-MM-DD)
df['COLLECT_DATE'] = cudf.to_datetime(df['COLLECT_DATE'], format='%Y-%m-%d')
# 'COLLECT_DATE'를 datetime 타입으로 변환한 후 다시 문자열 포맷으로 변환
df['COLLECT_DATE'] = df['COLLECT_DATE'].dt.strftime('%Y-%m-%d')

# 'COLLECT_DATE' 기준으로 오름차순 정렬
df = df.sort_values('COLLECT_DATE')

# 고유한 날짜 추출 및 정렬
unique_dates = sorted(df['COLLECT_DATE'].unique().to_pandas())

# 총 고유 날짜 수
total_dates = len(unique_dates)

# 분할 비율 계산 (7:1.5:1.5 -> 총 10)
train_ratio = 7 / 10
val_ratio = 1.5 / 10
test_ratio = 1.5 / 10

# 각 세트에 할당할 날짜 수 계산
train_dates_count = int(total_dates * train_ratio)
val_dates_count = int(total_dates * val_ratio)
# test_dates_count는 나머지 모든 날짜로 설정
test_dates_count = total_dates - train_dates_count - val_dates_count

# 날짜 세트 분할
train_dates = unique_dates[:train_dates_count]
val_dates = unique_dates[train_dates_count:train_dates_count + val_dates_count]
test_dates = unique_dates[train_dates_count + val_dates_count:]

# 각 세트에 해당하는 데이터 필터링
train_df = df[df['COLLECT_DATE'].isin(train_dates)]
val_df = df[df['COLLECT_DATE'].isin(val_dates)]
test_df = df[df['COLLECT_DATE'].isin(test_dates)]

# 데이터셋 분할 비율 및 개수 출력
print("Train set ratio: {:.2%}".format(len(train_df) / len(df)))
print("Validation set ratio: {:.2%}".format(len(val_df) / len(df)))
print("Test set ratio: {:.2%}".format(len(test_df) / len(df)))

print("원래 총 개수 : ", len(df))
print("train 개수 : ", len(train_df))
print("val 개수 : ", len(val_df))
print("test 개수 : ", len(test_df))

# 분할된 데이터셋 저장
input_filename = os.path.basename(csv)
ori_filename = os.path.splitext(input_filename)[0]
train_df.to_csv(os.path.join(os.path.dirname(csv), ori_filename) + '_train.csv', index=False)
val_df.to_csv(os.path.join(os.path.dirname(csv), ori_filename) + '_val.csv', index=False)
test_df.to_csv(os.path.join(os.path.dirname(csv), ori_filename) + '_test.csv', index=False)
