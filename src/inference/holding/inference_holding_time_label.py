import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import joblib  # LabelEncoder 저장 및 로드를 위한 라이브러리

# 1. 데이터셋 클래스 정의
class HoldingTimeDataset(Dataset):
    def __init__(self, data):
        data = data.reset_index(drop=True)
        # 모든 요일을 포함하는 리스트
        all_days = [1, 2, 3, 4, 5, 6, 7]
        # 전처리된 DataFrame을 받습니다.
        self.day_type = pd.get_dummies(data['DAY_TYPE']).reindex(columns=all_days, fill_value=0).astype(float).values
        self.busroute_id = data['BUSROUTE_ID'].values
        self.location_id = data['LOCATION_ID'].values
        self.dep_hour_min = (pd.to_datetime(data['DEP_TIME'], format='%H:%M').dt.hour * 60 # 특성 생성
                                + pd.to_datetime(data['DEP_TIME'], format='%H:%M').dt.minute)
        self.time_gap = data['TIME_GAP'].values

    def __len__(self):
        return len(self.time_gap)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.day_type[idx], dtype=torch.float),
            torch.tensor(self.busroute_id[idx], dtype=torch.long),
            torch.tensor(self.location_id[idx], dtype=torch.long),
            torch.tensor(self.dep_hour_min[idx], dtype=torch.float),
            torch.tensor(self.time_gap[idx], dtype=torch.float)
        )

# 2. 모델 정의
class HoldingTimeModel(nn.Module):
    def __init__(self, input_size, num_routes, num_locations, route_embedding_dim, location_embedding_dim):
        super(HoldingTimeModel, self).__init__()

        # 텍스트 임베딩 레이어
        self.route_embedding = nn.Embedding(num_embeddings=num_routes, embedding_dim=route_embedding_dim)
        self.location_embedding = nn.Embedding(num_embeddings=num_locations, embedding_dim=location_embedding_dim)

        # 전결합 레이어
        self.fc1 = nn.Linear(input_size + route_embedding_dim + location_embedding_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        # 활성화 함수 및 기타 레이어
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, day_type, busroute_id, location_id, dep_hour_min):
        route_emb = self.route_embedding(busroute_id).squeeze(1)
        location_emb = self.location_embedding(location_id).squeeze(1)
        inputs = torch.cat([day_type, route_emb, location_emb, dep_hour_min.unsqueeze(1)], dim=1)

        x = self.leaky_relu(self.bn1(self.fc1(inputs)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        output = self.fc3(x)
        return output

# 3. 모델 추론 함수 정의
def predict(model, test_loader, device, num_samples=20):
    model.eval()
    actuals = []
    predictions = []

    with torch.no_grad():   # 기울기 계산 비활성화
        for data in test_loader:
            day_type, busroute_id, location_id, dep_hour_min, time_gap = [d.to(device) for d in data]
            outputs = model(day_type, busroute_id, location_id, dep_hour_min)

            actuals.extend(time_gap.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

    actuals = np.array(actuals)
    predictions = np.array(predictions).flatten()  # 2차원을 1차원으로 변환

    # 평균 오차 계산
    absolute_errors = np.abs(predictions - actuals)
    mean_absolute_error = np.mean(absolute_errors)

    print(f"\n\nMean Absolute Error on Test Data: {mean_absolute_error:.2f} seconds")

    # 랜덤으로 샘플 선택
    indices = np.random.choice(len(actuals), num_samples, replace=False)
    selected_actuals = actuals[indices]
    selected_predictions = predictions[indices]

    # 비교 테이블 출력
    comparison_table = pd.DataFrame({
        'Actual': selected_actuals.flatten(),
        'Predicted': selected_predictions.flatten(),
        'Absolute Error': np.abs(selected_actuals - selected_predictions).flatten()
    })
    print("\nSample Predictions:")
    print(comparison_table)

    # 샘플 시각화
    plot_name = f'holding_time_comparison_test.png'
    plt.figure(figsize=(10, 6))
    plt.plot(selected_actuals.flatten(), 'o-', label='Actual')
    plt.plot(selected_predictions.flatten(), 'x-', label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Time Gap (seconds)')
    plt.title('Actual vs Predicted Time Gaps')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_name)
    plt.close()  # 그래프 저장 후 닫기
    print(f"Plot saved as {plot_name}")

    return predictions

# 4. 메인 실행 코드
if __name__ == "__main__":
    # 결과 저장 폴더 생성
    start_time = time.strftime("%Y%m%d-%H%M%S")
    results_folder = 'results_holding_time'
    os.makedirs(results_folder, exist_ok=True)

    # GPU 사용 여부 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 저장된 LabelEncoder 로드
    label_encoder_route = joblib.load('../label_encoder_route.pkl')
    label_encoder_location = joblib.load('../label_encoder_location.pkl')

    # 예측할 데이터 로드 및 전처리
    # test_data = pd.read_csv('../dataset/dwellTime.csv')
    test_data = pd.read_csv('/dataset/NEW_DWELLTIME.csv')

    # LabelEncoder 에 존재하는 BUSROUTE_ID 만 추출
    new_busroute_ids = test_data['BUSROUTE_ID'].astype(str)
    valid_busroute_ids = new_busroute_ids.isin(label_encoder_route.classes_)

    # 필터링 후에도 원래 인덱스를 유지
    filtered_test_data = test_data[valid_busroute_ids].copy()

    # 'label_encoder_route.classes_'에는 없고 'new_busroute_ids'에는 있는 BUSROUTE_ID 찾기
    unique_invalid_route_ids = set(new_busroute_ids) - set(label_encoder_route.classes_)
    # 중복 제거된 고유한 값 출력
    if unique_invalid_route_ids:
        print("label_encoder_route.classes_에는 없고, new_busroute_ids에만 존재하는 BUSROUTE_ID들:")
        print(unique_invalid_route_ids)

    # LabelEncoder 에 존재하는 LOCATION_ID 만 추출
    new_location_ids = filtered_test_data['LOCATION_ID'].astype(str)
    valid_location_ids = new_location_ids.isin(label_encoder_location.classes_)

    # 필터링 후에도 원래 인덱스를 유지
    filtered_test_data = filtered_test_data[valid_location_ids].copy()

    # 'label_encoder_location.classes_'에는 없고 'new_location_ids'에는 있는 LOCATION_ID 찾기
    unique_invalid_location_ids = set(new_location_ids) - set(label_encoder_location.classes_)
    # 중복 제거된 고유한 값 출력
    if unique_invalid_location_ids:
        print("label_encoder_location.classes_에는 없고, new_location_ids에만 존재하는 LOCATION_ID들:")
        print(unique_invalid_location_ids)

    # 유효한 ID만 변환
    filtered_test_data['LOCATION_ID'] = label_encoder_location.transform(filtered_test_data['LOCATION_ID'].astype(str))
    filtered_test_data['BUSROUTE_ID'] = label_encoder_route.transform(filtered_test_data['BUSROUTE_ID'].astype(str))

    # num_routes 및 num_locations 계산
    num_routes = len(label_encoder_route.classes_)
    num_locations = len(label_encoder_location.classes_)

    # 임베딩 차원 계산
    route_embedding_dim = int(np.sqrt(num_routes))
    location_embedding_dim = int(np.sqrt(num_locations))

    # 모델 로드
    day_type_size = 7
    input_size = day_type_size + 1  # day_type + dep_hour_min

    model = HoldingTimeModel(input_size, num_routes, num_locations, route_embedding_dim, location_embedding_dim).to(device)

    # 저장된 모델 파라미터 로드
    model.load_state_dict(torch.load('../holding_time_model_20240824-080239.pth', map_location=device, weights_only=True))

    # Dataset 및 DataLoader 생성
    test_dataset = HoldingTimeDataset(filtered_test_data)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 예측 수행
    predictions = predict(model, test_loader, device)

    # 예측 결과를 원래 데이터셋에 추가
    filtered_test_data['TIME_GAP_ESTIMATE'] = predictions

    # BUSROUTE_ID, LOCATION_ID 디코딩 (복원)
    filtered_test_data['BUSROUTE_ID'] = label_encoder_route.inverse_transform(filtered_test_data['BUSROUTE_ID'].astype(int))
    filtered_test_data['LOCATION_ID'] = label_encoder_location.inverse_transform(filtered_test_data['LOCATION_ID'].astype(int))

    # 결과 CSV 파일로 저장
    result_file = 'holding_time_inference_result.csv'
    filtered_test_data.to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")
