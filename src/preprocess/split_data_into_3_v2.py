import pandas as pd
import numpy as np
from tqdm import tqdm

# CSV 파일 읽기
df = pd.read_csv('../dataset/test_holding_time.csv')

# 날짜와 시간 형식을 예상하지 못한 경우를 처리하는 함수 정의
def clean_datetime_column(column):
    # Try to convert the column to datetime, invalid parsing will be set as NaT
    cleaned_col = pd.to_datetime(column, errors='coerce', format='%Y-%m-%d %H:%M:%S')
    # If the conversion to datetime fails, try converting with only the date format
    if cleaned_col.isnull().any():
        date_only_conversion = pd.to_datetime(column, errors='coerce', format='%Y-%m-%d')
        cleaned_col = cleaned_col.fillna(date_only_conversion)
    return cleaned_col

# 모든 datetime 관련 컬럼을 정리
df['ARR_DATE'] = clean_datetime_column(df['ARR_DATE'])
df['DEP_DATE'] = clean_datetime_column(df['DEP_DATE'])

# 변환이 실패한 행들 제거
df.dropna(subset=['ARR_DATE', 'DEP_DATE'], inplace=True)

# DEP_TIME 추출
df['DEP_TIME'] = df['DEP_DATE'].dt.time

# 시간대를 그룹화하여 새로운 컬럼 추가
df['time_group'] = pd.cut(pd.to_datetime(df['DEP_TIME'].astype(str), format='%H:%M').dt.hour,
                          bins=[0, 7, 12, 18, 24],
                          labels=['00-07', '06-12', '12-18', '18-24'],
                          right=False)

# 데이터가 비율에 맞춰 잘 섞이도록 무작위로 섞음
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 데이터셋을 훈련, 검증, 테스트로 정확히 나누기 위한 함수
def stratified_split(df, stratify_cols, train_ratio=0.7, val_ratio=0.15):
    df['dataset'] = ''
    unique_combinations = df[stratify_cols].drop_duplicates()

    for _, combination in unique_combinations.iterrows():
        group = df[(df[stratify_cols] == combination.values).all(axis=1)]
        n = len(group)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)

        shuffled = group.sample(frac=1, random_state=42)
        df.loc[shuffled.index[:train_n], 'dataset'] = 'train'
        df.loc[shuffled.index[train_n:train_n+val_n], 'dataset'] = 'validation'
        df.loc[shuffled.index[train_n+val_n:], 'dataset'] = 'test'

    # 각 데이터셋의 최소 크기를 보장하기 위해, train, validation, test가 없을 경우 데이터를 추가 분배
    train_len = len(df[df['dataset'] == 'train'])
    val_len = len(df[df['dataset'] == 'validation'])
    test_len = len(df[df['dataset'] == 'test'])

    # Validation이나 Test 데이터셋이 부족하면, train에서 가져옴
    if val_len == 0 or test_len == 0:
        df['dataset'] = df['dataset'].replace('', 'test')
        if val_len == 0:
            df.loc[df[df['dataset'] == 'train'].sample(frac=val_ratio/(1-val_ratio), random_state=42).index, 'dataset'] = 'validation'
        if test_len == 0:
            df.loc[df[df['dataset'] == 'train'].sample(frac=(1-train_ratio-val_ratio)/(1-train_ratio), random_state=42).index, 'dataset'] = 'test'

    return df

# 데이터셋을 stratified_split을 통해 분할
stratify_columns = ['DAY_TYPE', 'BUSROUTE_ID', 'LOCATION_ID', 'time_group']  # 시간 그룹 추가
df_split = stratified_split(df, stratify_columns)

# 데이터셋 분할 비율 및 상태 요약 출력
print("\n데이터셋 분할 비율:")
print(df_split['dataset'].value_counts(normalize=True))

# 분할된 데이터셋 저장 경로 설정
train_output_path = '../dataset/test_holding_time_train.csv'
validation_output_path = '../dataset/test_holding_time_validation.csv'
test_output_path = '../dataset/test_holding_time_test.csv'

# 데이터셋 파일로 저장
df_split[df_split['dataset'] == 'train'].to_csv(train_output_path, index=False)
df_split[df_split['dataset'] == 'validation'].to_csv(validation_output_path, index=False)
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

    print("노선별 분포:")
    print(dataset.groupby('BUSROUTE_ID').size().to_string())
    print("구간별 분포:")
    print(dataset.groupby('LOCATION_ID').size().to_string())
    print("TIME_GAP 통계:")
    print(dataset['TIME_GAP'].describe())

summarize_dataset('Train', df_split[df_split['dataset'] == 'train'])
summarize_dataset('Validation', df_split[df_split['dataset'] == 'validation'])
summarize_dataset('Test', df_split[df_split['dataset'] == 'test'])

# 제거된 행 수 및 퍼센트 계산
original_row_count = len(pd.read_csv('../dataset/test_holding_time.csv'))
rows_dropped = original_row_count - len(df)
percent_dropped = (rows_dropped / original_row_count) * 100

# 제거된 행 수 및 퍼센트 출력
print(f"\n제거된 행의 수: {rows_dropped}개")
print(f"전체 행 대비 제거된 행의 비율: {percent_dropped:.2f}%")
