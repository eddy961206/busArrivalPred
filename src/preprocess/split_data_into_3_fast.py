import pandas as pd
import numpy as np
from tqdm import tqdm
import dask.dataframe as dd

# CSV 파일 읽기 (dask 사용) - 불필요한 컬럼 제외
df = dd.read_csv('../dataset/travel_time_7.csv', usecols=['DAY_TYPE', 'BUSINFOUNIT_ID', 'TIME_GAP', 'LEN', 'DEP_TIME'])

# 데이터 타입 최적화
# df['BUSROUTE_ID'] = df['BUSROUTE_ID'].astype('category')
df['DAY_TYPE'] = df['DAY_TYPE'].astype('category')
df['BUSINFOUNIT_ID'] = df['BUSINFOUNIT_ID'].astype('category')

# # 날짜와 시간 형식을 예상하지 못한 경우를 처리하는 함수 정의
# def clean_datetime_column(column):
#     cleaned_col = pd.to_datetime(column, errors='coerce', format='%Y-%m-%d %H:%M:%S')
#     if cleaned_col.isnull().any():
#         date_only_conversion = pd.to_datetime(column, errors='coerce', format='%Y-%m-%d')
#         cleaned_col = cleaned_col.fillna(date_only_conversion)
#     return cleaned_col
#
# # 모든 datetime 관련 컬럼을 정리
# df['ARR_DATE'] = df['ARR_DATE'].map_partitions(clean_datetime_column, meta=('ARR_DATE', 'datetime64[ns]'))
# df['DEP_DATE'] = df['DEP_DATE'].map_partitions(clean_datetime_column, meta=('DEP_DATE', 'datetime64[ns]'))

# 변환이 실패한 행들 제거
# df = df.dropna(subset=['ARR_DATE', 'DEP_DATE'])

# DEP_TIME 추출
# df['DEP_TIME'] = df['DEP_DATE'].dt.time

# Dask DataFrame을 Pandas DataFrame으로 변환
df = df.compute()

# 데이터를 무작위로 섞고 인덱스 리셋
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 전체 데이터를 비율에 따라 나누는 함수
def split_data(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    df['dataset'] = ''
    df.loc[:train_end, 'dataset'] = 'train'
    df.loc[train_end:val_end, 'dataset'] = 'val'
    df.loc[val_end:, 'dataset'] = 'test'

    return df

# 데이터셋을 split_data을 통해 분할
df_split = split_data(df, train_ratio=0.7, val_ratio=0.15)

# 데이터셋 분할 비율 및 상태 요약 출력
print("\n데이터셋 분할 비율:")
print(df_split['dataset'].value_counts(normalize=True))

# 분할된 데이터셋 저장 경로 설정
train_output_path = '../dataset/travel_time_7_train.csv'
validation_output_path = '../dataset/travel_time_7_val.csv'
test_output_path = '../dataset/travel_time_7_test.csv'

# 데이터셋 파일로 저장
df_split[df_split['dataset'] == 'train'].to_csv(train_output_path, index=False)
df_split[df_split['dataset'] == 'val'].to_csv(validation_output_path, index=False)
df_split[df_split['dataset'] == 'test'].to_csv(test_output_path, index=False)

# 데이터셋 상태 요약 출력
def summarize_dataset(name, dataset):
    print(f"\n{name} 데이터셋:")
    print(f"총 샘플 수: {len(dataset)}")
    print("요일별 분포:")
    print(dataset.groupby('DAY_TYPE').size().to_string())
    # 5분 단위로 묶어서 시간대별 분포를 출력
    dataset['DEP_TIME_5MIN'] = pd.to_datetime(dataset['DEP_TIME'].astype(str)).dt.floor('5T').dt.strftime('%H:%M')
    print("5분 단위 시간대별 분포:")
    print(dataset.groupby('DEP_TIME_5MIN').size().to_string())

    print("구간ID별 분포:")
    print(dataset.groupby('BUSINFOUNIT_ID').size().to_string())
    print("구간 거리별 분포:")
    print(dataset.groupby('LEN').size().to_string())
    print("TIME_GAP 통계:")
    print(dataset['TIME_GAP'].describe())

summarize_dataset('Train', df_split[df_split['dataset'] == 'train'])
summarize_dataset('Validation', df_split[df_split['dataset'] == 'val'])
summarize_dataset('Test', df_split[df_split['dataset'] == 'test'])

# 제거된 행 수 및 퍼센트 계산
original_row_count = len(pd.read_csv('../dataset/travel_time_07280400_08110359.csv'))
rows_dropped = original_row_count - len(df_split)
percent_dropped = (rows_dropped / original_row_count) * 100

# 제거된 행 수 및 퍼센트 출력
print(f"\n제거된 행의 수: {rows_dropped}개")
print(f"전체 행 대비 제거된 행의 비율: {percent_dropped:.2f}%")
