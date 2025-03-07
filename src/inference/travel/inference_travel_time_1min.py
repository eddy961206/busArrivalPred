import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../src/train/travel/'))
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# from travel import TravelTimeModel, TravelTimeDataset
import time
import pickle
import importlib.util

# 3. 예측 및 결과 저장 함수
def predict(test_loader, model, device, target_scaler, num_samples=50, save_dir="results"):
    model.eval()
    actuals = []
    predictions = []
    total_mape = 0

    with torch.no_grad():
        for batch in test_loader:
            features, businfo_unit_id, time_gap, _ = [b.to(device) for b in batch]
            outputs = model(features, businfo_unit_id)

            # 실제 타겟 값을 역변환
            time_gap_original = target_scaler.inverse_transform(time_gap.cpu().numpy().reshape(-1, 1)).flatten()
            outputs_original = target_scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1)).flatten()

            actuals.extend(time_gap_original)
            predictions.extend(outputs_original)

            # actuals.extend(time_gap.cpu().numpy())
            # predictions.extend(outputs.cpu().numpy())

            # MAPE (원래 스케일)
            mape = np.mean(np.abs(outputs_original - time_gap_original) / (np.abs(time_gap_original) + 1e-8)) * 100
            total_mape += mape


    actuals = np.array(actuals)
    predictions = np.array(predictions).flatten()

    # 평균 오차 계산
    absolute_errors = np.abs(predictions - actuals)
    mean_absolute_error = np.mean(absolute_errors)

    print(f"\n\nMean Absolute Error on Test Data: {mean_absolute_error:.0f} seconds")

    # MAPE
    test_mape = total_mape / len(test_loader)
    print(f"Mean Absolute Percentage Error on Test Data: {test_mape:.2f}%")

    # 3개의 범주로 나누기
    category_1 = (actuals < 120)  # 2분 미만
    category_2 = (actuals >= 120) & (actuals < 300)  # 2분 이상 5분 미만
    category_3 = (actuals >= 300)  # 5분 이상

    # 각 범주에서 20개씩 샘플 선택
    indices_1 = np.random.choice(np.where(category_1)[0], min(20, sum(category_1)), replace=False)
    indices_2 = np.random.choice(np.where(category_2)[0], min(20, sum(category_2)), replace=False)
    indices_3 = np.random.choice(np.where(category_3)[0], min(20, sum(category_3)), replace=False)

    # 선택된 샘플들 합치기
    selected_indices = np.concatenate([indices_1, indices_2, indices_3])
    selected_actuals = actuals[selected_indices]
    selected_predictions = predictions[selected_indices]

    # 비교 테이블 생성 및 출력
    comparison_tables = []
    categories = ['<2min', '2-5min', '>=5min']
    indices_list = [indices_1, indices_2, indices_3]

    for category, indices in zip(categories, indices_list):
        table = pd.DataFrame({
            'Category': category,
            'Actual': actuals[indices],
            'Predicted': predictions[indices],
            'Absolute Error': np.abs(actuals[indices] - predictions[indices])
        })
        comparison_tables.append(table)

    full_comparison = pd.concat(comparison_tables).reset_index(drop=True)
    print("\nSample Predictions:")
    print(full_comparison)

    # 비교 테이블 저장
    table_path = os.path.join(save_dir, 'prediction_comparison_table.csv')
    full_comparison.to_csv(table_path, index=False)
    print(f"Comparison table saved as {table_path}")

    # 샘플 시각화
    plot_name = os.path.join(save_dir, 'prediction_comparison_plot.png')
    plt.figure(figsize=(15, 8))
    plt.scatter(range(len(indices_1)), selected_actuals[:len(indices_1)], c='blue', label='Actual (<2min)', marker='o')
    plt.scatter(range(len(indices_1)), selected_predictions[:len(indices_1)], c='red', label='Predicted (<2min)', marker='x')
    plt.scatter(range(len(indices_1), len(indices_1)+len(indices_2)), selected_actuals[len(indices_1):len(indices_1)+len(indices_2)], c='green', label='Actual (2-5min)', marker='o')
    plt.scatter(range(len(indices_1), len(indices_1)+len(indices_2)), selected_predictions[len(indices_1):len(indices_1)+len(indices_2)], c='orange', label='Predicted (2-5min)', marker='x')
    plt.scatter(range(len(indices_1)+len(indices_2), len(selected_actuals)), selected_actuals[len(indices_1)+len(indices_2):], c='purple', label='Actual (>=5min)', marker='o')
    plt.scatter(range(len(indices_1)+len(indices_2), len(selected_actuals)), selected_predictions[len(indices_1)+len(indices_2):], c='brown', label='Predicted (>=5min)', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Time Gap (seconds)')
    plt.title('Actual vs Predicted Time Gaps (60 Samples from 3 Categories)')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_name)
    plt.close()
    print(f"Plot saved as {plot_name}")

    return predictions

# 4. 메인 실행 코드
if __name__ == "__main__":
    # 결과 저장 폴더 생성
    start_time = time.strftime("%Y%m%d-%H%M%S")
    results_folder = f'../../../runs/inference_results/travel/{start_time}'
    os.makedirs(results_folder, exist_ok=True)

    # GPU 사용 여부 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 추론용 데이터 로드
    test_csv_min = '../../../dataset/inference/travel/241004_노선모두_8.28~9.11_평일_속도X/소통_0926_inf_filtered.csv'
    dtype_spec = {
        'DAY_TYPE': 'int8',
        'BUSROUTE_ID': 'str',
        'BUSINFOUNIT_ID': 'str',
        'LEN': 'int32',
        'DEP_TIME': 'str',
        'TIME_GAP': 'int32',
        # 'TIME_GAP': 'float32',  # int32는 NaN 값을 처리할 수 없으므로 float32로 변경
        'SPEED': 'int32'
        # 'SPEED': 'float32'
    }
    # usecols = ['DAY_TYPE', 'BUSINFOUNIT_ID', 'LEN', 'DEP_TIME', 'SPEED', 'TIME_GAP']
    # usecols = ['DAY_TYPE', 'BUSROUTE_ID', 'BUSINFOUNIT_ID', 'LEN', 'DEP_TIME', 'SPEED', 'TIME_GAP']
    usecols = ['BUSROUTE_ID', 'BUSINFOUNIT_ID', 'LEN', 'DEP_TIME', 'SPEED', 'TIME_GAP']

    data_pd = pd.read_csv(test_csv_min, skipinitialspace=True, usecols=usecols, dtype=dtype_spec)

    # SPEED 컬럼의 NaN 또는 빈 값 제거
    data_pd = data_pd.dropna(subset=['SPEED'])
    data_pd = data_pd[data_pd['SPEED'] != '']
    # TIME_GAP 컬럼의 NaN 또는 빈 값 제거
    data_pd = data_pd.dropna(subset=['TIME_GAP'])
    data_pd = data_pd[data_pd['TIME_GAP'] != '']

    data_pd.reset_index(drop=True, inplace=True)

    # TIME_GAP 컬럼 정수로
    # data_pd['TIME_GAP'] = data_pd['TIME_GAP'].astype('int32')
    # SPEED 컬럼 정수로 반올림
    # data_pd['SPEED'] = data_pd['SPEED'].round().astype('int32')


    # 1. DEP_TIME을 시:분 포맷으로 변환 (초 제거)
    # data_pd['DEP_TIME'] = pd.to_datetime(data_pd['DEP_TIME'], format='%H:%M:%S').dt.strftime('%H:%M')

    # 2. 필요한 시간대 필터링 (07:00~09:00, 13:00~15:00, 18:00~20:00)
    # valid_times = (
    #     ((data_pd['DEP_TIME'] >= '07:00') & (data_pd['DEP_TIME'] <= '09:00')) |
    #     ((data_pd['DEP_TIME'] >= '13:00') & (data_pd['DEP_TIME'] <= '15:00')) |
    #     ((data_pd['DEP_TIME'] >= '18:00') & (data_pd['DEP_TIME'] <= '20:00'))
    # )

    # 3. 해당 시간대의 데이터만 추출
    # data_pd = data_pd[valid_times]

    model_path = ('../../../runs/train_results/travel/20241004-045458_노선모두_8.28~9.11_평일_속도X/models'
                  '/travel_model_lr5.00e-03_dr1.00e-04_bs4096_ud200_hs30000.pth')
    model_dir = os.path.dirname(model_path)

    # travel_train.py 파일 경로 설정
    travel_train_py_dir = os.path.dirname(model_dir)
    travel_train_py_path = os.path.join(travel_train_py_dir, 'travel_train.py')

    # 모듈을 동적으로 로드하는 함수
    def load_module_from_path(module_name, module_path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # model_dir 안의 travel_train.py 파일에서 TravelTimeModel, TravelTimeDataset 불러오기
    module = load_module_from_path('travel_train', travel_train_py_path)
    TravelTimeModel = getattr(module, 'TravelTimeModel')
    TravelTimeDataset = getattr(module, 'TravelTimeDataset')

    # 타겟 스케일러 로드
    try:
        with open(os.path.join(model_dir, 'len_scaler.pkl'), 'rb') as f:
            len_scaler = pickle.load(f)
        with open(os.path.join(model_dir, 'target_scaler.pkl'), 'rb') as f:
            target_scaler = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading scalers: {e}")
        sys.exit(1)

    # 모델 파라미터 설정 (최적화된 하이퍼파라미터로 설정)
    hash_size = 30000
    unit_embedding_dim = 200
    dropout_rate = 0.004

    # 추론용 데이터 전처리
    data_loader = DataLoader(
        TravelTimeDataset(data_pd, scaler=len_scaler, target_scaler=target_scaler, hash_size=hash_size, mode='train')
        , batch_size=4096, shuffle=False
    )

    num_features = data_loader.dataset.features.shape[1]
    input_size = num_features

    # 모델 로드
    model = TravelTimeModel(input_size, hash_size, unit_embedding_dim, dropout_rate).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 예측 및 결과 저장
    predictions = predict(data_loader, model, device, target_scaler=target_scaler,
                                 num_samples=20, save_dir=results_folder)

    # 원본 데이터에 TIME_GAP_ESTIMATE 추가 (역변환된 예측값 사용)
    data_pd['TIME_GAP_ESTIMATE'] = np.round(predictions, 0).astype(int)

    # 출발 시간 오름차순 정렬
    data_pd.sort_values(by=['BUSROUTE_ID', 'BUSINFOUNIT_ID', 'DEP_TIME'], inplace=True)
    data_pd.drop_duplicates(inplace=True)

    # 결과 저장
    result_file = os.path.join(results_folder, 'inference_result.csv')
    data_pd.to_csv(result_file, index=False)
    print(f"예측 결과 파일 저장 완료: {result_file}")
