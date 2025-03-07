# from sklearn.model_selection import train_test_split
# import pandas as pd
#
# dtype_spec = {
#     'DAY_TYPE': 'int8',
#     'BUSINFOUNIT_ID': 'str',
#     'LEN': 'int32',
#     'DEP_TIME': 'str',
#     'TIME_GAP': 'int32',
#     'speed_kmh': 'int8'
# }
#
# # 데이터프레임 로드
# df = pd.read_csv('/dataset/train/holding/combined_dwell_filtered.csv', dtype=dtype_spec)
#
# # 'BUSINFOUNIT_ID' 컬럼의 값별 개수 계산
# unit_counts = df['BUSINFOUNIT_ID'].value_counts()
#
# # 개수가 1인 'BUSINFOUNIT_ID' 값 추출
# unique_units = unit_counts[unit_counts == 1].index
#
# # 개수가 1인 'BUSINFOUNIT_ID' 값을 가진 행들 출력
# # for unit_id in unique_units:
# #     print(df[df['BUSINFOUNIT_ID'] == unit_id])
#
# # 개수가 1인 'BUSINFOUNIT_ID' 값을 가진 행들 필터링 (df_once)
# df_once = df[df['BUSINFOUNIT_ID'].isin(unique_units)]
#
# # 개수가 2개 이상인 'BUSINFOUNIT_ID' 값을 가진 행들 필터링 (df_multiple)
# df_multiple = df[~df['BUSINFOUNIT_ID'].isin(unique_units)]
#
# # df_multiple에 대해서만 Stratified split 수행
# train_df, temp_df = train_test_split(df_multiple, test_size=0.3, stratify=df_multiple['BUSINFOUNIT_ID'], random_state=42)
#
# # temp_df에서 개수가 1인 'BUSINFOUNIT_ID' 값을 가진 행들 필터링
# temp_unit_counts = temp_df['BUSINFOUNIT_ID'].value_counts()
# temp_unique_units = temp_unit_counts[temp_unit_counts == 1].index
# temp_df_once = temp_df[temp_df['BUSINFOUNIT_ID'].isin(temp_unique_units)]
# temp_df_multiple = temp_df[~temp_df['BUSINFOUNIT_ID'].isin(temp_unique_units)]
#
# # temp_df_multiple에 대해서만 Stratified split 수행
# val_df, test_df = train_test_split(temp_df_multiple, test_size=0.5, stratify=temp_df_multiple['BUSINFOUNIT_ID'], random_state=42)
# # train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df[['DAY_TYPE', 'BUSINFOUNIT_ID']], random_state=42)
# # val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[['DAY_TYPE', 'BUSINFOUNIT_ID']], random_state=42)
#
# # temp_df_once를 train_df에 추가
# train_df = pd.concat([train_df, temp_df_once])
#
# # train_df, val_df, test_df를 다시 cuDF DataFrame으로 변환
# train_df = pd.from_pandas(train_df)
# val_df = pd.from_pandas(val_df)
# test_df = pd.from_pandas(test_df)
#
# # df_once를 train_df에 추가
# train_df = pd.concat([train_df, df_once])
#
# # 데이터셋 분할 비율 확인
# print("Train set ratio: {:.2%}".format(len(train_df) / len(df)))
# print("Validation set ratio: {:.2%}".format(len(val_df) / len(df)))
# print("Test set ratio: {:.2%}".format(len(test_df) / len(df)))
#
# # 분할된 데이터셋 저장
# train_df.to_csv('/dataset/train/holding/combined_dwell_filtered_train.csv', index=False)
# val_df.to_csv('/dataset/train/holding/combined_dwell_filtered_val.csv', index=False)
# test_df.to_csv('/dataset/train/holding/combined_dwell_filtered_test.csv', index=False)




############# cuDF 사용 ##################

from sklearn.model_selection import train_test_split
import cudf as cudf
import pandas as pd
import os
dtype_spec = {
    'DAY_TYPE': 'int8',
    'BUSROUTE_ID': 'str',
    'BUSSTOP_ID': 'str',
    'DEP_TIME': 'str',
    'TIME_GAP': 'int32'
}
csv = '../../../dataset/train/holding/240927_7,8공휴일/정차_7,8공휴일_train_filtered.csv'
# cuDF DataFrame으로 로드
df = cudf.read_csv(csv, dtype=dtype_spec)
# 'BUSROUTE_ID'와 'BUSSTOP_ID'를 결합한 새로운 컬럼 생성
df['ROUTE_BUSSTOP'] = df['BUSROUTE_ID'] + '_' + df['BUSSTOP_ID']

# 'ROUTE_BUSSTOP' 컬럼의 값별 개수 계산
unit_counts = df['ROUTE_BUSSTOP'].value_counts()

# 개수가 1인 'ROUTE_BUSSTOP' 값 추출
unique_units = unit_counts[unit_counts == 1].index

# 개수가 1인 'ROUTE_BUSSTOP' 값을 가진 행들 필터링 (df_once)
df_once = df[df['ROUTE_BUSSTOP'].isin(unique_units)]

# 개수가 2개 이상인 'ROUTE_BUSSTOP' 값을 가진 행들 필터링 (df_multiple)
df_multiple = df[~df['ROUTE_BUSSTOP'].isin(unique_units)]

# df_multiple에 대해서만 Stratified split 수행 (cuDF 지원 안 됨)
# cuDF는 `train_test_split`의 `stratify` 매개변수를 지원하지 않으므로 Pandas로 변환 후 수행
train_df, temp_df = train_test_split(df_multiple.to_pandas(), test_size=0.3, stratify=df_multiple['ROUTE_BUSSTOP'].to_pandas(), random_state=42)

# temp_df에서 개수가 1인 'ROUTE_BUSSTOP' 값을 가진 행들 필터링
temp_unit_counts = temp_df['ROUTE_BUSSTOP'].value_counts()
temp_unique_units = temp_unit_counts[temp_unit_counts == 1].index
temp_df_once = temp_df[temp_df['ROUTE_BUSSTOP'].isin(temp_unique_units)]
temp_df_multiple = temp_df[~temp_df['ROUTE_BUSSTOP'].isin(temp_unique_units)]

# temp_df_multiple에 대해서만 Stratified split 수행
val_df, test_df = train_test_split(temp_df_multiple, test_size=0.5, stratify=temp_df_multiple['ROUTE_BUSSTOP'], random_state=42)

# temp_df_once를 train_df에 추가
train_df = pd.concat([train_df, temp_df_once])

# train_df, val_df, test_df를 다시 cuDF DataFrame으로 변환
train_df = cudf.from_pandas(train_df)
val_df = cudf.from_pandas(val_df)
test_df = cudf.from_pandas(test_df)

# df_once를 train_df에 추가
train_df = cudf.concat([train_df, df_once])

# ROUTE_BUSSTOP 컬럼 삭제
train_df = train_df.drop('ROUTE_BUSSTOP', axis=1)
val_df = val_df.drop('ROUTE_BUSSTOP', axis=1)
test_df = test_df.drop('ROUTE_BUSSTOP', axis=1)

# 데이터셋 분할 비율 확인
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




