# import cudf as cudf
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
#
# dtype_spec = {
#     'DAY_TYPE': 'int8',
#     'BUSROUTE_ID': 'str',
#     'BUSINFOUNIT_ID': 'str',
#     'LEN': 'int32',
#     'DEP_TIME': 'str',
#     'TIME_GAP': 'int32',
#     'SPEED': 'int32'
# }
#
# # cuDF DataFrame으로 로드
# df = cudf.read_csv('../../../dataset/train/route/route_4,5_TUE_filtered.csv', dtype=dtype_spec)
#
# # 여러 컬럼을 합쳐서 StratifiedKFold에 사용할 새로운 컬럼 생성
# df['stratify_key'] = df['BUSROUTE_ID'].astype(str) + '_' + \
#                     df['BUSINFOUNIT_ID'].astype(str) + '_' + \
#                     df['DEP_TIME'].astype(str)
#
# # Pandas DataFrame으로 변환
# df_pd = df.to_pandas()
#
# # StratifiedKFold 객체 생성 (5개 폴드)
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
# # 폴드 번호를 저장할 리스트 생성
# fold_assignments = [-1] * len(df_pd)
#
# # StratifiedKFold를 사용하여 데이터 분할
# for fold_num, (train_index, test_index) in enumerate(skf.split(df_pd, df_pd['stratify_key'])):
#     fold_assignments[test_index] = fold_num
#
# # 폴드 번호를 cuDF DataFrame에 추가
# df['fold'] = cudf.Series(fold_assignments)
#
# # 각 폴드를 별도의 DataFrame으로 저장
# for fold_num in range(5):
#     train_df = df[df['fold'] != fold_num]
#     val_df = df[df['fold'] == fold_num].sample(frac=0.5, random_state=42)  # 검증 세트 50%
#     test_df = df[df['fold'] == fold_num].drop(val_df.index)  # 테스트 세트 50%
#
#     # 데이터셋 분할 비율 확인
#     print(f"Fold {fold_num + 1}:")
#     print("Train set ratio: {:.2%}".format(len(train_df) / len(df)))
#     print("Validation set ratio: {:.2%}".format(len(val_df) / len(df)))
#     print("Test set ratio: {:.2%}".format(len(test_df) / len(df)))
#
#     print("원래 총 개수 : ", len(df))
#     print("train 개수 : ", len(train_df))
#     print("val 개수 : ", len(val_df))
#     print("test 개수 : ", len(test_df))
#
#     # 분할된 데이터셋 저장
#     train_df.to_csv(f'../../../dataset/train/route/route_4,5_TUE_filtered_train_fold{fold_num + 1}.csv', index=False)
#     val_df.to_csv(f'../../../dataset/train/route/route_4,5_TUE_filtered_val_fold{fold_num + 1}.csv', index=False)
#     test_df.to_csv(f'../../../dataset/train/route/route_4,5_TUE_filtered_test_fold{fold_num + 1}.csv', index=False)
#
# # 'stratify_key' 컬럼 삭제
# df = df.drop('stratify_key', axis=1)


from sklearn.model_selection import train_test_split
import cudf
import pandas as pd

# 데이터 타입 정의
dtype_spec = {
    'DAY_TYPE': 'int8',
    'BUSROUTE_ID': 'str',
    'BUSINFOUNIT_ID': 'str',
    'LEN': 'int32',
    'DEP_TIME': 'str',
    'TIME_GAP': 'int32',
    'SPEED': 'int32',
    'DATA_TYPE': 'str'
}

# cuDF DataFrame으로 데이터 로드
df = cudf.read_csv('../../../dataset/train/route/route_4,5_TUE_filtered.csv', dtype=dtype_spec)

# cuDF는 Stratified split 기능을 지원하지 않으므로 pandas로 변환 후 처리
# 데이터를 pandas로 변환
df_pandas = df.to_pandas()

# 7:1.5:1.5 비율로 train/val/test 데이터셋 무작위 분할
train_df, temp_df = train_test_split(df_pandas, test_size=0.3, random_state=42)

# temp_df에서 다시 val/test 분할 (1.5:1.5 비율로 나누기 위해 50%로 나눔)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 분할된 데이터셋을 다시 cuDF로 변환
train_df = cudf.from_pandas(train_df)
val_df = cudf.from_pandas(val_df)
test_df = cudf.from_pandas(test_df)

# 데이터셋 분할 비율 확인
print("Train set ratio: {:.2%}".format(len(train_df) / len(df)))
print("Validation set ratio: {:.2%}".format(len(val_df) / len(df)))
print("Test set ratio: {:.2%}".format(len(test_df) / len(df)))

print("원래 총 개수 : ", len(df))
print("train 개수 : ", len(train_df))
print("val 개수 : ", len(val_df))
print("test 개수 : ", len(test_df))

# 분할된 데이터셋 저장
train_df.to_csv('../../../dataset/train/route/route_4,5_TUE_filtered_train.csv', index=False)
val_df.to_csv('../../../dataset/train/route/route_4,5_TUE_filtered_val.csv', index=False)
test_df.to_csv('../../../dataset/train/route/route_4,5_TUE_filtered_test.csv', index=False)