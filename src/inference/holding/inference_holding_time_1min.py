import sys
import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "../../train/holding")
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../train/holding'))
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from holding import HoldingTimeModel, HoldingTimeDataset
import os
import time
import importlib.util
import pickle

# 3. 예측 및 결과 저장 함수
def predict(data_loader, model, device, target_scaler, num_samples=20, save_dir="results"):
    model.eval()
    actuals = []
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            # features, busroute_id, busstop_id, time_gap = [b.to(device) for b in batch]
            features, busroute_id, busstop_id = [b.to(device) for b in batch]
            outputs = model(features, busroute_id, busstop_id)

            # 실제 타겟 값을 역변환
            # time_gap_original = target_scaler.inverse_transform(time_gap.cpu().numpy().reshape(-1, 1)).flatten()
            outputs_original = target_scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1)).flatten()

            # actuals.extend(time_gap_original)
            predictions.extend(outputs_original)

    # actuals = np.array(actuals)
    predictions = np.array(predictions).flatten()

    # # 평균 오차 계산
    # absolute_errors = np.abs(predictions - actuals)
    # mean_absolute_error = np.mean(absolute_errors)
    #
    # print(f"\n\nMean Absolute Error on Test Data: {mean_absolute_error:.0f} seconds")
    #
    # # 3개의 범주로 나누기
    # category_1 = (actuals < 120)  # 2분 미만
    # category_2 = (actuals >= 120) & (actuals < 300)  # 2분 이상 5분 미만
    # category_3 = (actuals >= 300)  # 5분 이상
    #
    # # 각 범주에서 20개씩 샘플 선택
    # indices_1 = np.random.choice(np.where(category_1)[0], min(20, sum(category_1)), replace=False)
    # indices_2 = np.random.choice(np.where(category_2)[0], min(20, sum(category_2)), replace=False)
    # indices_3 = np.random.choice(np.where(category_3)[0], min(20, sum(category_3)), replace=False)
    #
    # # 선택된 샘플들 합치기
    # selected_indices = np.concatenate([indices_1, indices_2, indices_3])
    # selected_actuals = actuals[selected_indices]
    # selected_predictions = predictions[selected_indices]
    #
    # # 비교 테이블 생성 및 출력
    # comparison_tables = []
    # categories = ['<2min', '2-5min', '>=5min']
    # indices_list = [indices_1, indices_2, indices_3]
    #
    # for category, indices in zip(categories, indices_list):
    #     table = pd.DataFrame({
    #         'Category': category,
    #         'Actual': actuals[indices],
    #         'Predicted': predictions[indices],
    #         'Absolute Error': np.abs(actuals[indices] - predictions[indices])
    #     })
    #     comparison_tables.append(table)
    #
    # full_comparison = pd.concat(comparison_tables).reset_index(drop=True)
    # print("\nSample Predictions:")
    # print(full_comparison)
    #
    # # 비교 테이블 저장
    # table_path = os.path.join(save_dir, 'prediction_comparison_table.csv')
    # full_comparison.to_csv(table_path, index=False)
    # print(f"Comparison table saved as {table_path}")
    #
    # # 샘플 시각화
    # plot_name = os.path.join(save_dir, 'prediction_comparison_plot.png')
    # plt.figure(figsize=(15, 8))
    # plt.scatter(range(len(indices_1)), selected_actuals[:len(indices_1)], c='blue', label='Actual (<2min)', marker='o')
    # plt.scatter(range(len(indices_1)), selected_predictions[:len(indices_1)], c='red', label='Predicted (<2min)', marker='x')
    # plt.scatter(range(len(indices_1), len(indices_1)+len(indices_2)), selected_actuals[len(indices_1):len(indices_1)+len(indices_2)], c='green', label='Actual (2-5min)', marker='o')
    # plt.scatter(range(len(indices_1), len(indices_1)+len(indices_2)), selected_predictions[len(indices_1):len(indices_1)+len(indices_2)], c='orange', label='Predicted (2-5min)', marker='x')
    # plt.scatter(range(len(indices_1)+len(indices_2), len(selected_actuals)), selected_actuals[len(indices_1)+len(indices_2):], c='purple', label='Actual (>=5min)', marker='o')
    # plt.scatter(range(len(indices_1)+len(indices_2), len(selected_actuals)), selected_predictions[len(indices_1)+len(indices_2):], c='brown', label='Predicted (>=5min)', marker='x')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Time Gap (seconds)')
    # plt.title('Actual vs Predicted Time Gaps (60 Samples from 3 Categories)')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(plot_name)
    # plt.close()
    # print(f"Plot saved as {plot_name}")

    return predictions

# 4. 메인 실행 코드
if __name__ == "__main__":
    # 결과 저장 폴더 생성
    start_time = time.strftime("%Y%m%d-%H%M%S")
    results_folder = f'../../../runs/inference_results/holding/{start_time}'
    os.makedirs(results_folder, exist_ok=True)

    # GPU 사용 여부 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 추론용 데이터 로드
    test_csv = '../../../dataset/inference/holding/241004_노선모두_8.28~9.11_평일_속도X/정차_0926_inf.csv'
    dtype_spec = {
        'DAY_TYPE': 'int8',
        'BUSROUTE_ID': 'str',
        'BUSSTOP_ID': 'str',
        'DEP_TIME': 'str',
        'TIME_GAP': 'int32'
    }
    usecols = ['BUSROUTE_ID', 'BUSSTOP_ID', 'DEP_TIME']
    # usecols = ['DAY_TYPE', 'BUSROUTE_ID', 'BUSSTOP_ID', 'DEP_TIME']
    # usecols = ['DAY_TYPE', 'BUSROUTE_ID', 'BUSSTOP_ID', 'DEP_TIME', 'TIME_GAP']

    data_pd = pd.read_csv(test_csv, skipinitialspace=True, usecols=usecols, dtype=dtype_spec)
    data_pd = data_pd.loc[(data_pd != 0).all(axis=1)].dropna(how='all')   # 훈련할때도 0인거 없애고 훈련해야
    data_pd = data_pd.reset_index(drop=True)

    # 모델 파라미터 설정 (최적화된 하이퍼파라미터로 설정)
    route_hash_size = 20000
    busstop_hash_size = 20000
    route_embedding_dim = 200
    busstop_embedding_dim = 200
    dropout_rate = 0.0001
    model_path = ('../../../runs/train_results/holding/20241004-055309_노선모두_8.28~9.11_평일_속도X/models'
                  '/holding_time_model_lr4.00e-03_dr1.00e-04_bs4096_rd200_ld200_rhs20000_lhs20000.pth')  # 학습된 모델 경로

    model_dir = os.path.dirname(model_path)

    # holding_train.py 파일 경로 설정
    holding_train_py_dir = os.path.dirname(model_dir)
    holding_train_py_path = os.path.join(holding_train_py_dir, 'holding_train.py')

    # 모듈을 동적으로 로드하는 함수
    def load_module_from_path(module_name, module_path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # model_dir 안의 holding_train.py 파일에서 HoldingTimeModel, HoldingTimeDataset 불러오기
    module = load_module_from_path('holding_train', holding_train_py_path)
    HoldingTimeModel = getattr(module, 'HoldingTimeModel')
    HoldingTimeDataset = getattr(module, 'HoldingTimeDataset')

    # 타겟 스케일러 로드
    try:
        with open(os.path.join(model_dir, 'target_scaler.pkl'), 'rb') as f:
            target_scaler = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading scalers: {e}")
        sys.exit(1)

    # 추론용 데이터 전처리
    data_loader = DataLoader(
        HoldingTimeDataset(data_pd, target_scaler=target_scaler, route_hash_size=route_hash_size, busstop_hash_size=busstop_hash_size, mode='inference'),
        batch_size=4096, shuffle=False
    )

    num_features = data_loader.dataset.features.shape[1]
    input_size = num_features

    # 모델 로드
    model = HoldingTimeModel(input_size, route_hash_size, busstop_hash_size, route_embedding_dim, busstop_embedding_dim, dropout_rate).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 예측 및 결과 저장
    predictions = predict(data_loader, model, device, target_scaler=target_scaler, num_samples=20, save_dir=results_folder)

    # 원본 데이터에 TIME_GAP_ESTIMATE 추가 및 저장
    data_pd['TIME_GAP_ESTIMATE'] = np.round(predictions, 1).astype(int)
    
    # 결과 저장
    result_file = os.path.join(results_folder, 'inference_result.csv')
    data_pd.to_csv(result_file, index=False)
    print(f"예측 결과 파일 저장 완료 : {result_file}")