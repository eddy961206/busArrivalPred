import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from tqdm import tqdm

# 1. Dataset 클래스 정의
class TravelTimeDataset(Dataset):
    def __init__(self, data, mode='train'):
        self.mode = mode  # 'train', 'inference'

        # 요일 정보 (One-hot encoding)
        all_days = [1, 2, 3, 4, 5, 6, 7]
        self.day_type = pd.get_dummies(data['DAY_TYPE']).reindex(columns=all_days, fill_value=0).astype(float).values

        # 구간 길이 정보
        self.len = data['LEN'].values
        self.len_scaled = MinMaxScaler().fit_transform(self.len.reshape(-1, 1)).flatten()  # 거리 표준화

        # 출발 시각 (시와 분을 분리하여 사용)
        self.dep_time = pd.to_datetime(data['DEP_TIME'], format='%H:%M')
        self.dep_hour = self.dep_time.dt.hour
        self.dep_minute = self.dep_time.dt.minute

        # TIME_GAP 정답 (train 모드에서만 사용)
        if mode != 'inference':
            self.time_gap = data['TIME_GAP'].values

        #######  추가 특성들  #######
        # 주기성 특성 (출발 시각)
        self.dep_time_sin = np.sin(2 * np.pi * self.dep_hour / 24)
        self.dep_time_cos = np.cos(2 * np.pi * self.dep_hour / 24)

        # 주기성 특성 (요일)
        self.day_type_sin = np.sin(2 * np.pi * data['DAY_TYPE'] / 7)
        self.day_type_cos = np.cos(2 * np.pi * data['DAY_TYPE'] / 7)

        # 혼잡 시간대 여부 (출퇴근 시간: 7-9시, 18-20시)
        self.peak_hours = ((self.dep_hour >= 7) & (self.dep_hour < 10) |
                           (self.dep_hour >= 18) & (self.dep_hour < 20)).astype(int)

        # 단거리/중거리/장거리 구분
        self.distance_type = np.select(
            [
                self.len < 500,             # 단거리
                (self.len >= 500) & (self.len < 1500),  # 중거리
                self.len >= 1500            # 장거리
            ],
            [0, 1, 2],  # 0: 단거리, 1: 중거리, 2: 장거리
            default=0
        )

        # 최종 입력 특성으로 합침
        self.features = np.column_stack((
            self.day_type,        # 원-핫 인코딩된 요일 정보
            self.len_scaled,      # 스케일링된 구간 길이
            self.dep_time_sin,    # 시간 기반 주기성 (sin)
            self.dep_time_cos,    # 시간 기반 주기성 (cos)
            self.day_type_sin,    # 요일 기반 주기성 (sin)
            self.day_type_cos,    # 요일 기반 주기성 (cos)
            self.peak_hours,      # 혼잡 시간대 여부
            self.distance_type    # 단거리/중거리/장거리 구분
        ))

    def __len__(self):
        return len(self.len)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float)

        if self.mode == 'inference':
            return features
        else:
            time_gap = torch.tensor(self.time_gap[idx], dtype=torch.float)
            return features, time_gap

# 2. LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 최종 출력은 구간 소요 시간 (1개 값)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 마지막 LSTM 출력을 사용하여 구간 소요 시간 예측
        return out

# 3. 모델 훈련 함수
def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # Training loop
        for features, time_gap in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            features, time_gap = features.to(device), time_gap.to(device)
            optimizer.zero_grad()
            outputs = model(features.unsqueeze(1))  # LSTM 입력 크기 맞춤
            loss = criterion(outputs.view(-1), time_gap)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_features, val_time_gap in val_loader:
                val_features, val_time_gap = val_features.to(device), val_time_gap.to(device)
                val_outputs = model(val_features.unsqueeze(1))
                val_loss = criterion(val_outputs.view(-1), val_time_gap)
                total_val_loss += val_loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_train_loss / len(train_loader):.4f}, Val Loss: {total_val_loss / len(val_loader):.4f}')

# 4. 데이터 로드 및 DataLoader 생성
def load_data(train_csv, val_csv, batch_size=32):
    def load_and_clean_data(csv_path):
        dtype_spec = {
            'DAY_TYPE': 'int8',
            'LEN': 'int32',
            'DEP_TIME': 'str',
            'TIME_GAP': 'int32'
        }
        usecols = ['DAY_TYPE', 'LEN', 'DEP_TIME', 'TIME_GAP']
        return pd.read_csv(csv_path, usecols=usecols, dtype=dtype_spec).dropna()

    train_data = load_and_clean_data(train_csv)
    val_data = load_and_clean_data(val_csv)

    train_dataset = TravelTimeDataset(train_data, mode='train')
    val_dataset = TravelTimeDataset(val_data, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# 5. 메인 실행 코드
if __name__ == "__main__":
    # 경로 설정
    train_csv = 'dataset/train/travel_train.csv'
    val_csv = 'dataset/train/travel_val.csv'

    # DataLoader 생성
    train_loader, val_loader = load_data(train_csv, val_csv)

    # LSTM 모델 초기화
    input_size = train_loader.dataset.features.shape[1]
    hidden_size = 64
    num_layers = 2
    model = LSTMModel(input_size, hidden_size, num_layers)

    # 모델을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 모델 훈련
    train_model(model, train_loader, val_loader)
