import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from travel_time_optuna_hash_directory_complex_feat import TravelTimeModel, TravelTimeDataset
import os
import time

# 3. 예측 및 결과 저장 함수
def predict(test_loader, model, device, num_samples=50, save_dir="results"):
    model.eval()
    actuals = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            features, businfo_unit_id, time_gap, distance_weight = [b.to(device) for b in batch]
            outputs = model(features, businfo_unit_id)

            actuals.extend(time_gap.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

    actuals = np.array(actuals)
    predictions = np.array(predictions).flatten()

    # 평균 오차 계산
    absolute_errors = np.abs(predictions - actuals)
    mean_absolute_error = np.mean(absolute_errors)

    print(f"\n\nMean Absolute Error on Test Data: {mean_absolute_error:.0f} seconds")

    # 랜덤으로 샘플 선택
    indices = np.random.choice(len(actuals), num_samples, replace=False)
    selected_actuals = actuals[indices]
    selected_predictions = predictions[indices]

    # 비교 테이블 생성 및 저장
    comparison_table = pd.DataFrame({
        'Actual': np.round(selected_actuals.flatten(), 0),
        'Predicted': np.round(selected_predictions.flatten(), 0),
        'Absolute Error': np.round(np.abs(selected_actuals - selected_predictions).flatten(), 0)
    })
    comparison_table['MAE'] = np.round(np.mean(comparison_table['Absolute Error']), 0)

    # 테이블 저장
    table_path = os.path.join(save_dir, 'prediction_comparison_table.csv')
    comparison_table.to_csv(table_path, index=False)
    print(f"Comparison table saved as {table_path}")

    # 샘플 시각화 및 저장
    plot_name = os.path.join(save_dir, 'prediction_comparison_plot.png')
    plt.figure(figsize=(10, 6))
    plt.plot(selected_actuals.flatten(), 'o-', label='Actual')
    plt.plot(selected_predictions.flatten(), 'x-', label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Time Gap (seconds)')
    plt.title('Actual vs Predicted Time Gaps')
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
    results_folder = f'results_travel_time/{start_time}'
    os.makedirs(results_folder, exist_ok=True)

    # GPU 사용 여부 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 파라미터 설정 (최적화된 하이퍼파라미터로 설정)
    hash_size = 16137
    unit_embedding_dim = 189
    input_size = 20  # train_loader.dataset.features.shape[1]  # day_type(7) + len_feature + dep_hour_min
    dropout_rate = 0.2825456848354049
    # model_path = r'..\results_travel_time\20240903-135942\models\travel_time_model_lr1.14e-04_dr3.43e-01_bs64_ud222_hs16128.pth'  # 학습된 모델 경로
    model_path = '/results_travel_time/20240904-021745_MAE_3/models/travel_time_model_lr4.91e-04_dr2.83e-01_bs64_ud189_hs16137.pth'  # 학습된 모델 경로
    # 모델 로드
    model = TravelTimeModel(input_size, hash_size, unit_embedding_dim, dropout_rate).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 추론용 데이터 로드
    # test_csv = '/dataset/infounit_filtered.csv'
    # test_csv = '/dataset/NEW_INFOUNIT_filtered.csv'
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
    # test_data = test_data.loc[(test_data != 0).all(axis=1)].dropna(how='all')   # 훈련할때도 0인거 없애고 훈련해야
    # test_data = test_data.reset_index(drop=True)

    # 추론용 데이터 전처리
    test_loader = DataLoader(TravelTimeDataset(test_data, hash_size=hash_size), batch_size=128, shuffle=False)

    # 예측 및 결과 저장
    predictions = predict(test_loader, model, device, num_samples=20, save_dir=results_folder)

    # 원본 데이터에 TIME_GAP_ESTIMATE 추가 및 MAE 계산
    test_data['TIME_GAP_ESTIMATE'] = np.round(predictions, 0)
    test_data['MAE'] = np.abs(test_data['TIME_GAP'] - test_data['TIME_GAP_ESTIMATE'])

    # 속도 컬럼을 정수로 변환하고 위치 이동
    test_data['speed_kmh'] = test_data['speed_kmh'].astype(int)
    columns = test_data.columns.tolist()
    columns.insert(columns.index('TIME_GAP'), columns.pop(columns.index('speed_kmh')))
    test_data = test_data[columns]

    # TIME_GAP 기준으로 내림차순 정렬
    test_data = test_data.sort_values('TIME_GAP', ascending=False)

    # 결과 저장
    result_file = os.path.join(results_folder, 'travel_time_inference_result.csv')
    test_data.to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")
