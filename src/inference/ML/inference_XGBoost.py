# inference_XGBoost.py
import sys
import os
import pandas as pd
import numpy as np
import joblib
import hashlib
import time
from sklearn.preprocessing import MinMaxScaler

# 1. Hash 함수 정의
def hash_id(id_str, num_buckets=20000):
    """
    문자열 ID를 해시 버킷으로 변환하는 함수입니다.

    Args:
        id_str (str): 해시할 문자열 ID.
        num_buckets (int): 해시 버킷의 총 개수.

    Returns:
        int: 해시 버킷 인덱스.
    """
    return int(hashlib.md5(id_str.encode()).hexdigest(), 16) % num_buckets

# 2. Configuration 설정
config = {
    "test_csv": '../../../dataset/inference/route/LSTM/inf_0604.csv',
    "num_buckets": 20000,      # 해시 버킷 크기 설정 (충돌 방지를 위해 충분히 크게 설정)
    "model_path": '../../../runs/train_results/route/XGBoost/models/best_xgboost_model.joblib',
    "results_save_path": '../../../runs/inference_results/route/XGBoost/results/',
}

# 3. 데이터 로드 및 전처리 함수 정의
def load_and_preprocess_inference(csv_path, num_buckets=20000, scaler=None):
    dtype_spec = {
        'DAY_TYPE': 'int8',
        'BUSROUTE_ID': 'str',
        'BUSINFOUNIT_ID': 'str',
        'LEN': 'int32',
        'DEP_TIME': 'str'
    }
    usecols = ['DAY_TYPE', 'BUSROUTE_ID', 'BUSINFOUNIT_ID', 'LEN', 'DEP_TIME']

    data = pd.read_csv(csv_path, skipinitialspace=True, usecols=usecols, dtype=dtype_spec).dropna(how='all').reset_index(drop=True)

    # 해시 함수 적용
    data['BUSROUTE_HASH'] = data['BUSROUTE_ID'].apply(lambda x: hash_id(x, num_buckets))
    data['BUSINFOUNIT_HASH'] = data['BUSINFOUNIT_ID'].apply(lambda x: hash_id(x, num_buckets))

    # DEP_TIME을 datetime 형식으로 변환
    data['DEP_TIME'] = pd.to_datetime(data['DEP_TIME'], format='%Y-%m-%d %H:%M:%S')

    # 특징 생성
    data['DEP_HOUR'] = data['DEP_TIME'].dt.hour
    data['DEP_MINUTE'] = data['DEP_TIME'].dt.minute

    # LEN 스케일링
    data['LEN_SCALED'] = scaler.transform(data['LEN'].values.reshape(-1, 1)).flatten()

    # 주기성 특성 생성
    data['DEP_TIME_SIN'] = np.sin(2 * np.pi * data['DEP_HOUR'] / 24)
    data['DEP_TIME_COS'] = np.cos(2 * np.pi * data['DEP_HOUR'] / 24)
    data['DAY_TYPE_SIN'] = np.sin(2 * np.pi * data['DAY_TYPE'] / 7)
    data['DAY_TYPE_COS'] = np.cos(2 * np.pi * data['DAY_TYPE'] / 7)
    data['LOG_LEN'] = np.log1p(data['LEN'])
    data['IS_WEEKEND'] = data['DAY_TYPE'].isin([1, 7]).astype(int)

    # 시간대 구분
    data['TIME_OF_DAY'] = np.select(
        [
            (data['DEP_HOUR'] >= 0) & (data['DEP_HOUR'] < 6),
            (data['DEP_HOUR'] >= 6) & (data['DEP_HOUR'] < 12),
            (data['DEP_HOUR'] >= 12) & (data['DEP_HOUR'] < 18),
            (data['DEP_HOUR'] >= 18) & (data['DEP_HOUR'] < 24)
        ],
        [0, 1, 2, 3],
        default=2
    )

    # 혼잡 시간대 여부
    data['PEAK_HOURS'] = ((data['DEP_HOUR'] >= 7) & (data['DEP_HOUR'] < 10) |
                          (data['DEP_HOUR'] >= 18) & (data['DEP_HOUR'] < 20)).astype(int)

    # 거리 유형
    data['DISTANCE_TYPE'] = np.select(
        [
            data['LEN'] < 500,
            (data['LEN'] >= 500) & (data['LEN'] < 1500),
            data['LEN'] >= 1500
        ],
        [0, 1, 2],
        default=0
    )

    # 최종 특징 선택
    feature_cols = [
        'DAY_TYPE', 'BUSROUTE_HASH', 'BUSINFOUNIT_HASH', 'DEP_HOUR', 'DEP_MINUTE',
        'LEN_SCALED', 'DEP_TIME_SIN', 'DEP_TIME_COS', 'DAY_TYPE_SIN', 'DAY_TYPE_COS',
        'LOG_LEN', 'IS_WEEKEND', 'TIME_OF_DAY', 'PEAK_HOURS', 'DISTANCE_TYPE'
    ]
    features = data[feature_cols]

    return data, features

# 4. 예측 및 결과 저장 함수 정의
def predict_and_save(data, features, model, config):
    predictions = model.predict(features)
    predictions_rounded = np.round(predictions, 0).astype(int)

    # 예측 결과를 데이터프레임에 추가
    data['TIME_GAP_ESTIMATE'] = -1  # 모든 행을 -1로 초기화
    data.loc[data.index, 'TIME_GAP_ESTIMATE'] = predictions_rounded

    # 결과 저장 폴더 생성
    start_time = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(config['results_save_path'], exist_ok=True)
    result_file = os.path.join(config['results_save_path'], f'inference_result_{start_time}.csv')
    data.to_csv(result_file, index=False)
    print(f"예측 결과 파일 저장 완료: {result_file}")

# 5. 메인 실행 코드
if __name__ == "__main__":
    # 데이터 로드 및 전처리
    print("데이터 로드 및 전처리 중...")
    # 스케일러 로드 (훈련 시 저장한 스케일러가 필요)
    scaler_path = '../../../runs/train_results/route/XGBoost/scaler.joblib'
    scaler = joblib.load(scaler_path)

    data_pd, test_features = load_and_preprocess_inference(config['test_csv'], num_buckets=config['num_buckets'], scaler=scaler)

    # 모델 로드
    print("모델 로드 중...")
    model = joblib.load(config['model_path'])
    print(f"모델이 로드되었습니다: {config['model_path']}")

    # 예측 및 결과 저장
    print("예측 중...")
    predict_and_save(data_pd, test_features, model, config)
