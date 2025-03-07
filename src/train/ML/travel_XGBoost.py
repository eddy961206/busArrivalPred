# travel_XGBoost.py
import pandas as pd
import numpy as np
import hashlib
import os
import time
import json
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 1. Hash 함수 정의
def hash_id(id_str, num_buckets=20000):
    return int(hashlib.md5(id_str.encode()).hexdigest(), 16) % num_buckets

# 2. Configuration 설정
config = {
    "train_csv": '../../../dataset/train/travel/240926_두달치_1분이상/소통_7,8월평일_긴거추가_filtered_train.csv',
    "val_csv": '../../../dataset/train/travel/240926_두달치_1분이상/소통_7,8월평일_긴거추가_filtered_val.csv',
    "test_csv": '../../../dataset/train/travel/240926_두달치_1분이상/소통_7,8월평일_긴거추가_filtered_test.csv',
    "n_trial": 4,
    "trial_epochs": 5,
    "patience": 20,
    "min_delta": 0.0001,
    "num_buckets": 20000,      # 해시 버킷 크기 설정 (충돌 방지를 위해 충분히 크게 설정)
    "model_save_path": '../../../runs/train_results/route/XGBoost/models/best_xgboost_model.joblib',
    "log_save_path": '../../../runs/train_results/route/XGBoost/logs/',
    "results_save_path": '../../../runs/train_results/route/XGBoost/results/',
}

# 3. 데이터 로드 및 전처리 함수 정의
def load_and_preprocess(csv_path, num_buckets=20000, mode='train', scaler=None):
    dtype_spec = {
        'DAY_TYPE': 'int8',
        'BUSROUTE_ID': 'str',
        'BUSINFOUNIT_ID': 'str',
        'LEN': 'int32',
        'DEP_TIME': 'str',
        'TIME_GAP': 'int32'
    }
    # usecols = ['DAY_TYPE', 'BUSROUTE_ID', 'BUSINFOUNIT_ID', 'LEN', 'DEP_TIME', 'TIME_GAP']
    usecols = ['BUSROUTE_ID', 'BUSINFOUNIT_ID', 'LEN', 'DEP_TIME', 'TIME_GAP']

    data = pd.read_csv(csv_path, skipinitialspace=True, usecols=usecols, dtype=dtype_spec).dropna(how='all').reset_index(drop=True)

    # 해시 함수 적용
    data['BUSROUTE_HASH'] = data['BUSROUTE_ID'].apply(lambda x: hash_id(x, num_buckets))
    data['BUSINFOUNIT_HASH'] = data['BUSINFOUNIT_ID'].apply(lambda x: hash_id(x, num_buckets))

    # DEP_TIME을 datetime 형식으로 변환
    data['DEP_TIME'] = pd.to_datetime(data['DEP_TIME'], format='%H:%M')

    # 특징 생성
    data['DEP_HOUR'] = data['DEP_TIME'].dt.hour
    data['DEP_MINUTE'] = data['DEP_TIME'].dt.minute

    # LEN 스케일링
    if mode == 'train':
        scaler = MinMaxScaler()
        data['LEN_SCALED'] = scaler.fit_transform(data['LEN'].values.reshape(-1, 1)).flatten()
    elif mode in ['val', 'test', 'inference']:
        if scaler is None:
            raise ValueError("Scaler must be provided for val, test, or inference mode")
        data['LEN_SCALED'] = scaler.transform(data['LEN'].values.reshape(-1, 1)).flatten()

    # 주기성 특성 생성
    data['DEP_TIME_SIN'] = np.sin(2 * np.pi * data['DEP_HOUR'] / 24)
    data['DEP_TIME_COS'] = np.cos(2 * np.pi * data['DEP_HOUR'] / 24)
    data['LOG_LEN'] = np.log1p(data['LEN'])

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
        'BUSROUTE_HASH', 'BUSINFOUNIT_HASH', 'DEP_HOUR', 'DEP_MINUTE',
        'LEN_SCALED', 'DEP_TIME_SIN', 'DEP_TIME_COS',
        'LOG_LEN', 'TIME_OF_DAY', 'PEAK_HOURS', 'DISTANCE_TYPE'
    ]
    features = data[feature_cols]

    if mode in ['train', 'val']:
        targets = data['TIME_GAP']
        return features, targets, scaler
    elif mode == 'inference':
        return data, features, scaler

# 4. 모델 학습 함수 정의
def train_xgboost(train_features, train_targets, val_features, val_targets, config):

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000000,
        learning_rate=0.01,
        max_depth=20,
        eval_metric='rmse',
        early_stopping_rounds=config['patience'],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',  # GPU 사용 설정
        device='cuda'
        # predictor='gpu_predictor'  # GPU 기반 예측 사용
    )

    eval_set = [(val_features, val_targets)]
    model.fit(
        train_features, train_targets,
        eval_set=eval_set,
        verbose=True
    )

    # 모델 저장
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    joblib.dump(model, config['model_save_path'])
    print(f"모델이 저장되었습니다: {config['model_save_path']}")

    return model

# 5. 메인 실행 코드
if __name__ == "__main__":

    start_time = time.strftime("%Y%m%d-%H%M%S")
    log_folder = os.path.join(config['log_save_path'], start_time)
    os.makedirs(log_folder, exist_ok=True)
    results_folder = os.path.join(config['results_save_path'], start_time)
    os.makedirs(results_folder, exist_ok=True)

    # 데이터 로드 및 전처리
    print("데이터 로드 및 전처리 중...")
    train_features, train_targets, scaler = load_and_preprocess(config['train_csv'], num_buckets=config['num_buckets'], mode='train')
    val_features, val_targets, _ = load_and_preprocess(config['val_csv'], num_buckets=config['num_buckets'], mode='val', scaler=scaler)
    test_data, test_features, _ = load_and_preprocess(config['test_csv'], num_buckets=config['num_buckets'], mode='inference', scaler=scaler)

    # 모델 학습
    print("모델 학습 중...")
    model = train_xgboost(train_features, train_targets, val_features, val_targets, config)

    # 검증 데이터 평가
    print("검증 데이터 평가 중...")
    val_predictions = model.predict(val_features)
    val_mse = mean_squared_error(val_targets, val_predictions)
    val_mae = mean_absolute_error(val_targets, val_predictions)
    print(f"Validation MSE: {val_mse:.4f}, Validation MAE: {val_mae:.4f}")

    # 결과 저장
    with open(os.path.join(log_folder, 'validation_metrics.json'), 'w') as f:
        json.dump({'Validation MSE': val_mse, 'Validation MAE': val_mae}, f, indent=4)
    print(f"검증 지표가 저장되었습니다: {os.path.join(log_folder, 'validation_metrics.json')}")

    # 테스트 데이터 예측 및 평가
    print("테스트 데이터 예측 및 평가 중...")
    test_features_np = test_features.to_numpy()
    test_predictions = model.predict(test_features_np)

    # 예측 결과 저장
    test_data['TIME_GAP_ESTIMATE'] = np.round(test_predictions, 1).astype(int)
    test_data['ERROR'] = np.abs(test_data['TIME_GAP_ESTIMATE'] - test_data['TIME_GAP']).astype(int)
    test_data['ERROR_RATE'] = np.round(test_data['ERROR'] / test_data['TIME_GAP'], 2)
    test_data.sort_values(by=['ERROR'], ascending=False, inplace=True)

    # 예측 결과 저장
    result_file = os.path.join(results_folder, 'test_dataset_inference.csv')
    test_data.to_csv(result_file, index=False)
    print(f"test dataset inferenced : {result_file}")

    # 모델과 scaler 저장
    model_save_folder = os.path.dirname(config['model_save_path'])
    os.makedirs(model_save_folder, exist_ok=True)
    joblib.dump(model, config['model_save_path'])
    print(f"최종 모델이 저장되었습니다: {config['model_save_path']}")

    scaler_save_path = os.path.join(model_save_folder, 'scaler.joblib')
    joblib.dump(scaler, scaler_save_path)
    print(f"스케일러가 저장되었습니다: {scaler_save_path}")
