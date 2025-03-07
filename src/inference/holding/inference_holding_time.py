import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from holding_time_optuna_hash_directory_hashDivide_feat import HoldingTimeModel, HoldingTimeDataset
import os
import time

# 3. 예측 및 결과 저장 함수
def predict(test_loader, model, device, num_samples=20, save_dir="results"):
    model.eval()
    actuals = []
    predictions = []

    with torch.no_grad():
        for data in test_loader:
            day_type, busroute_id, location_id, dep_hour_min, time_gap = [d.to(device) for d in data]
            outputs = model(day_type, busroute_id, location_id, dep_hour_min)

            actuals.extend(time_gap.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

    actuals = np.array(actuals)
    predictions = np.array(predictions).flatten()

    # 평균 오차 계산
    absolute_errors = np.abs(predictions - actuals)
    mean_absolute_error = np.mean(absolute_errors)

    print(f"\n\nMean Absolute Error on Test Data: {mean_absolute_error:.1f} seconds")

    # 랜덤으로 샘플 선택
    indices = np.random.choice(len(actuals), num_samples, replace=False)
    selected_actuals = actuals[indices]
    selected_predictions = predictions[indices]

    # 비교 테이블 생성 및 저장
    comparison_table = pd.DataFrame({
        'Actual': np.round(selected_actuals.flatten(), 1),
        'Predicted': np.round(selected_predictions.flatten(), 1),
        'Absolute Error': np.round(np.abs(selected_actuals - selected_predictions).flatten(), 1)
    })
    comparison_table['MAE'] = np.round(np.mean(comparison_table['Absolute Error']), 1)

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
    results_folder = f'results_holding_time/{start_time}'
    os.makedirs(results_folder, exist_ok=True)

    # GPU 사용 여부 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 파라미터 설정 (최적화된 하이퍼파라미터로 설정)
    route_hash_size = 1018
    location_hash_size = 13295
    route_embedding_dim = 19
    location_embedding_dim = 60
    input_size = 8  # day_type + dep_hour_min
    dropout_rate = 0.15493008280378182
    model_path = r'..\results_holding_time\20240830-031301\models\holding_time_model_trial_1_epoch_3.pth'  # 학습된 모델 경로

    # 모델 로드
    model = HoldingTimeModel(input_size, route_hash_size, location_hash_size, route_embedding_dim, location_embedding_dim, dropout_rate).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 추론용 데이터 로드
    test_csv = '/dataset/NEW_DWELLTIME.csv'
    # test_data = pd.read_csv(test_csv, skipinitialspace=True).dropna(how='all')
    test_data = pd.read_csv(test_csv, skipinitialspace=True)
    test_data = test_data.loc[(test_data != 0).all(axis=1)].dropna(how='all')   # 훈련할때도 0인거 없애고 훈련해야
    test_data = test_data.reset_index(drop=True)

    # 추론용 데이터 전처리
    test_loader = DataLoader(HoldingTimeDataset(test_data, route_hash_size=route_hash_size, location_hash_size=location_hash_size), batch_size=128, shuffle=False)

    # 예측 및 결과 저장
    predictions = predict(test_loader, model, device, num_samples=20, save_dir=results_folder)

    # 원본 데이터에 TIME_GAP_ESTIMATE 추가 및 저장
    test_data['TIME_GAP_ESTIMATE'] = np.round(predictions, 1)
    result_file = os.path.join(results_folder, 'holding_time_inference_result.csv')
    test_data.to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")