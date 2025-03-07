import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import joblib  # LabelEncoder 저장 및 로드를 위한 라이브러리

# 1. 데이터셋 클래스 정의
class TravelTimeDataset(Dataset):
    def __init__(self, data):
        data = data.reset_index(drop=True)
        # 모든 요일을 포함하는 리스트
        all_days = [1, 2, 3, 4, 5, 6, 7]
        # 전처리된 DataFrame을 받습니다.
        self.day_type = pd.get_dummies(data['DAY_TYPE']).reindex(columns=all_days, fill_value=0).astype(float).values
        self.businfo_unit_id = data['BUSINFOUNIT_ID'].values
        self.len = data['LEN'].values
        self.dep_hour_min = (pd.to_datetime(data['DEP_TIME'], format='%H:%M').dt.hour * 60 # 특성 생성
                                + pd.to_datetime(data['DEP_TIME'], format='%H:%M').dt.minute)
        self.time_gap = data['TIME_GAP'].values

    def __len__(self):
        return len(self.time_gap)

    def __getitem__(self, idx):
        day_type = torch.tensor(self.day_type[idx], dtype=torch.float)
        businfo_unit_id = torch.tensor(self.businfo_unit_id[idx], dtype=torch.long)
        len_feature = torch.tensor(self.len[idx], dtype=torch.float)
        dep_hour_min = torch.tensor(self.dep_hour_min[idx], dtype=torch.float)
        time_gap = torch.tensor(self.time_gap[idx], dtype=torch.float)
        return day_type, businfo_unit_id, len_feature, dep_hour_min, time_gap

# 2. 모델 정의
class TravelTimeModel(nn.Module):
    def __init__(self, input_size, num_units, unit_embedding_dim):
        super(TravelTimeModel, self).__init__()

        # 텍스트 임베딩 레이어
        self.unit_embedding = nn.Embedding(num_embeddings=num_units, embedding_dim=unit_embedding_dim)

        # 전결합 레이어
        self.fc1 = nn.Linear(input_size + unit_embedding_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        # 활성화 함수 및 기타 레이어
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, day_type, businfo_unit_id, len_feature, dep_hour_min):
        unit_emb = self.unit_embedding(businfo_unit_id).squeeze(1)
        inputs = torch.cat([day_type, unit_emb, len_feature.unsqueeze(1), dep_hour_min.unsqueeze(1)], dim=1)

        x = self.leaky_relu(self.bn1(self.fc1(inputs)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        output = self.fc3(x)
        return output

# 3. 모델 추론 함수 정의
def predict(model, test_loader, device, num_samples=10):
    model.eval()
    actuals = []
    predictions = []

    with torch.no_grad():   # 기울기 계산 비활성화
        for data in test_loader:
            day_type, businfo_unit_id, len_feature, dep_hour_min, time_gap = [d.to(device) for d in data]
            outputs = model(day_type, businfo_unit_id, len_feature, dep_hour_min)

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
    plot_name = os.path.join(results_folder, f'travel_time_comparison_test.png')
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

# 7. 학습 및 테스트 코드 실행
if __name__ == "__main__":
    # 결과 저장 폴더 생성
    start_time = time.strftime("%Y%m%d-%H%M%S")
    results_folder = 'results_travel_time'
    os.makedirs(results_folder, exist_ok=True)

    # GPU 사용 여부 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 저장된 LabelEncoder 로드
    label_encoder_unit = joblib.load('../label_encoder_unit_20240826-074537.pkl')

    # 예측할 데이터 로드 및 전처리
    # test_data = pd.read_csv('../dataset/infounit.csv')
    test_data = pd.read_csv('../dataset/NEW_INFOUNIT.csv')

    # LabelEncoder 에 존재하는 BUSINFOUNIT_ID 만 추출
    new_data_ids = test_data['BUSINFOUNIT_ID'].astype(str)
    valid_ids = new_data_ids.isin(label_encoder_unit.classes_)
    # 필터링 후에도 원래 인덱스를 유지
    filtered_test_data = test_data[valid_ids].copy()

    # 인덱스 재설정 (필요 시)
    filtered_test_data.reset_index(drop=True, inplace=True)

    # 'label_encoder_unit.classes_'에는 없고 'new_data_ids'에는 있는 BUSINFOUNIT_ID 찾기
    unique_invalid_ids = set(new_data_ids) - set(label_encoder_unit.classes_)

    # 중복 제거된 고유한 값 출력
    if unique_invalid_ids:
        print("label_encoder_unit.classes_에는 없고, new_data_ids에만 존재하는 BUSINFOUNIT_ID들:")
        print(unique_invalid_ids)

    # 유효한 ID만 변환
    filtered_test_data['BUSINFOUNIT_ID'] = label_encoder_unit.transform(filtered_test_data['BUSINFOUNIT_ID'].astype(str))

    # num_units 계산
    num_units = len(label_encoder_unit.classes_)

    # 임베딩 차원 계산
    unit_embedding_dim = int(np.sqrt(num_units))

    # 모델 설정
    day_type_size = 7
    input_size = day_type_size + 2  # day_type + len_feature + dep_hour_min

    model = TravelTimeModel(input_size, num_units, unit_embedding_dim).to(device)

    # 저장된 모델 파라미터 로드
    model.load_state_dict(torch.load('../results_travel_time/travel_time_model_20240826-074537.pth', map_location=device, weights_only=True))

    test_dataset = TravelTimeDataset(filtered_test_data)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 예측 수행
    predictions = predict(model, test_loader, device)

    # 예측 결과를 원래 데이터셋에 추가
    filtered_test_data['TIME_GAP_ESTIMATE'] = predictions
    # BUSINFOUNIT_ID 디코딩 (복원) 및 9자리로 맞추기
    filtered_test_data['BUSINFOUNIT_ID'] = pd.Series(label_encoder_unit.inverse_transform(filtered_test_data['BUSINFOUNIT_ID'].astype(int))).str.zfill(9)
    # 결과 CSV 파일로 저장
    result_file = 'travel_time_inference_result.csv'
    filtered_test_data.to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")
