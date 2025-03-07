import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../src/train/route/'))
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from travel_route import TravelTimeModel, TravelTimeDataset
import os
import time

# 3. 예측 및 결과 저장 함수
def predict(test_loader, model, device, num_samples=50, save_dir="results"):
    model.eval()
    actuals = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            features, busroute_id, businfo_unit_id, time_gap, _ = [b.to(device) for b in batch]
            outputs = model(features, busroute_id, businfo_unit_id)

            actuals.extend(time_gap.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

    actuals = np.array(actuals)
    predictions = np.array(predictions).flatten()

    # 평균 오차 계산
    absolute_errors = np.abs(predictions - actuals)
    mean_absolute_error = np.mean(absolute_errors)

    print(f"\n\nMean Absolute Error on Test Data: {mean_absolute_error:.0f} seconds")

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
    results_folder = f'../../../runs/inference_results/route/{start_time}'
    os.makedirs(results_folder, exist_ok=True)

    # GPU 사용 여부 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 추론용 데이터 로드
    test_csv_min = '../../../dataset/inference/travel/240920/inf_combined_filtered'
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
    usecols = ['DAY_TYPE', 'BUSROUTE_ID', 'BUSINFOUNIT_ID', 'LEN', 'DEP_TIME', 'SPEED', 'TIME_GAP']

    data_pd = pd.read_csv(test_csv_min, skipinitialspace=True, usecols=usecols, dtype=dtype_spec)

    # SPEED 컬럼의 NaN 또는 빈 값 제거
    data_pd = data_pd.dropna(subset=['SPEED'])
    data_pd = data_pd[data_pd['SPEED'] != '']
    # TIME_GAP 컬럼의 NaN 또는 빈 값 제거
    data_pd = data_pd.dropna(subset=['TIME_GAP'])
    data_pd = data_pd[data_pd['TIME_GAP'] != '']

    data_pd.reset_index(drop=True, inplace=True)

    # # TIME_GAP 컬럼 정수로
    # data_pd['TIME_GAP'] = data_pd['TIME_GAP'].astype('int32')
    # # SPEED 컬럼 정수로 반올림
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

    # 모델 파라미터 설정 (최적화된 하이퍼파라미터로 설정)
    route_hash_size = 20000
    unit_hash_size = 20000
    route_embedding_dim = 200
    unit_embedding_dim = 200
    dropout_rate = 0.0001
    model_path = '../../../runs/train_results/route/20240912-090650_1.89_데이터타입가중치/models/travel_model_lr4.00e-03_dr1.00e-04_bs4096_ud200_hs200.pth'
    # 추론용 데이터 전처리
    data_loader = DataLoader(
        TravelTimeDataset(data_pd, route_hash_size, unit_hash_size, mode='train'), batch_size=4096, shuffle=False
    )

    num_features = data_loader.dataset.features.shape[1]
    input_size = num_features

    # 모델 로드
    model = TravelTimeModel(input_size, route_hash_size, unit_hash_size, route_embedding_dim, unit_embedding_dim, dropout_rate).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 예측 및 결과 저장
    predictions_min = predict(data_loader, model, device, num_samples=20, save_dir=results_folder)

    # usecols에 지정된 컬럼만 남기기
    data_pd = data_pd[usecols]

    # 원본 데이터에 TIME_GAP_ESTIMATE 추가
    data_pd['TIME_GAP_ESTIMATE'] = np.round(predictions_min, 0).astype(int)

    # 출발 시간 오름차순 정렬
    data_pd.sort_values(by=['BUSROUTE_ID', 'BUSINFOUNIT_ID', 'DEP_TIME'], inplace=True)
    # data_pd.drop_duplicates(inplace=True)

    # 결과 저장
    result_file = os.path.join(results_folder, 'inference_result.csv')
    data_pd.to_csv(result_file, index=False)
    print(f"예측 결과 파일 저장 완료: {result_file}")
