# travel_LSTM.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import hashlib
from copy import deepcopy
from tqdm import tqdm

config = {
    "train_csv": 'dataset/train/소통시간학습_수요일_filtered_train.csv',
    "val_csv": 'dataset/train/소통시간학습_수요일_filtered_val.csv',
    "test_csv": 'dataset/train/소통시간학습_수요일_filtered_test.csv',
    "train_epochs": 30,
    "n_trial": 4,
    "trial_epochs": 5,
    "patience": 10,
    "min_delta": 0.0001
}

# 1. Dataset 클래스 정의
class TravelTimeDataset(Dataset):
    def __init__(self, data, hash_size=1000, mode='train'):
        self.mode = mode  # 'train', 'inference'
        self.hash_size = hash_size

        #######  기본 특성들  #######
        all_days = [1, 2, 3, 4, 5, 6, 7]
        self.day_type = pd.get_dummies(data['DAY_TYPE']).reindex(columns=all_days, fill_value=0).astype(float).values
        self.len = data['LEN'].values
        self.dep_time = pd.to_datetime(data['DEP_TIME'], format='%H:%M')
        self.dep_hour = self.dep_time.dt.hour
        self.dep_minute = self.dep_time.dt.minute
        if not self.mode == 'inference':  # inference 모드에서는 정답(time_gap)을 사용 못함
            self.time_gap = data['TIME_GAP'].values

        # 해시로 변환된 구간 ID
        self.businfo_unit_id = data['BUSINFOUNIT_ID'].apply(self.hash_function).values

        # 거리(LEN) 표준화
        self.len_scaled = MinMaxScaler().fit_transform(self.len.reshape(-1, 1)).flatten()

        #######  새로운 특성들  #######
        # 주기성 특성 (시간)
        self.dep_time_sin = np.sin(2 * np.pi * self.dep_hour / 24)
        self.dep_time_cos = np.cos(2 * np.pi * self.dep_hour / 24)

        # 주기성 특성 (요일)
        self.day_type_sin = np.sin(2 * np.pi * data['DAY_TYPE'] / 7)
        self.day_type_cos = np.cos(2 * np.pi * data['DAY_TYPE'] / 7)

        # 구간 길이 로그 변환
        self.log_len = np.log1p(self.len)

        # 주말 여부
        self.is_weekend = (data['DAY_TYPE'].isin([1, 7])).astype(int)

        # 시간대 (자정~새벽, 아침, 점심, 저녁)
        self.time_of_day = np.select(
            [
                (self.dep_hour >= 0) & (self.dep_hour < 6),   # 자정~새벽
                (self.dep_hour >= 6) & (self.dep_hour < 12),  # 아침
                (self.dep_hour >= 12) & (self.dep_hour < 18), # 점심
                (self.dep_hour >= 18) & (self.dep_hour < 24)  # 저녁
            ],
            [0, 1, 2, 3],  # 0: 자정~새벽, 1: 아침, 2: 점심, 3: 저녁
            default=2
        )

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

        # 추가된 특성을 합쳐 최종 입력 특성으로 만듦
        self.features = np.column_stack((
            self.day_type,        # 7개의 원-핫 인코딩된 요일
            self.dep_hour,        # 출발 시각의 시간 (hour)
            self.dep_minute,      # 출발 시각의 분 (minute)
            self.len_scaled,      # 스케일링된 거리 추가
            self.dep_time_sin,    # 시간 기반 주기성 (sin)
            self.dep_time_cos,    # 시간 기반 주기성 (cos)
            self.day_type_sin,    # 요일 기반 주기성 (sin)
            self.day_type_cos,    # 요일 기반 주기성 (cos)
            self.log_len,         # 구간 길이의 로그 변환
            self.is_weekend,      # 주말 여부
            self.time_of_day,     # 하루 중 시간대를 나타내는 특성
            self.peak_hours,      # 혼잡 시간대 여부 (출퇴근 시간)
            self.distance_type    # 단거리/중거리/장거리 구분
        ))

    def hash_function(self, x):
        if not isinstance(x, str):
            x = str(x)
        return int(hashlib.md5(x.encode()).hexdigest(), 16) % self.hash_size

    def __len__(self):
        return len(self.businfo_unit_id)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float)
        businfo_unit_id = torch.tensor(self.businfo_unit_id[idx], dtype=torch.long)

        if self.mode == 'inference':
            return features, businfo_unit_id
        else:
            time_gap = torch.tensor(self.time_gap[idx], dtype=torch.float)
            distance_weight = torch.tensor(3.0 if self.len[idx] >= 500 else 1.0, dtype=torch.float)
            return features, businfo_unit_id, time_gap, distance_weight

# 2. LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, businfo_unit_id):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 마지막 LSTM 출력만 사용
        return out

# 3. 훈련 함수
def train_model(model, train_loader, val_loader, num_epochs=None, learning_rate=0.001):
    if num_epochs is None:
        num_epochs = config['train_epochs']

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # tqdm 사용하여 진행상황 표시
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            features, businfo_unit_id, time_gap, distance_weight = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(features.unsqueeze(1), businfo_unit_id)

            # weighted_mse_loss 사용
            loss = weighted_mse_loss(outputs.view(-1), time_gap, distance_weight)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_features, val_businfo_unit_id, val_time_gap, val_distance_weight = [b.to(device) for b in val_batch]
                val_outputs = model(val_features.unsqueeze(1), val_businfo_unit_id)

                val_loss = weighted_mse_loss(val_outputs.view(-1), val_time_gap, val_distance_weight)
                total_val_loss += val_loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_train_loss/len(train_loader):.4f}, Val Loss: {total_val_loss/len(val_loader):.4f}')

# 4. 데이터 로드 및 DataLoader 생성
def load_data(config, batch_size=32, hash_size=1000):
    def load_and_clean_data(csv_path):
        dtype_spec = {
            'DAY_TYPE': 'int8',
            'BUSINFOUNIT_ID': 'str',
            'LEN': 'int32',
            'DEP_TIME': 'str',
            'TIME_GAP': 'int32'
        }
        usecols = ['DAY_TYPE', 'BUSINFOUNIT_ID', 'LEN', 'DEP_TIME', 'TIME_GAP']
        return pd.read_csv(csv_path, skipinitialspace=True, usecols=usecols, dtype=dtype_spec).dropna(how='all').reset_index(drop=True)

    train_data = load_and_clean_data(config['train_csv'])
    val_data = load_and_clean_data(config['val_csv'])
    test_data = load_and_clean_data(config['test_csv'])

    train_dataset = TravelTimeDataset(train_data, mode='train')
    val_dataset = TravelTimeDataset(val_data, mode='val')
    test_dataset = TravelTimeDataset(test_data, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def weighted_mse_loss(output, target, weight):
    return (weight * (output - target) ** 2).mean()

# 5. 메인 실행 코드
if __name__ == "__main__":
    # DataLoader 생성
    train_loader, val_loader, test_loader = load_data(config, batch_size=32, hash_size=1000)

    # LSTM 모델 초기화
    input_size = train_loader.dataset.features.shape[1]
    hidden_size = 64
    num_layers = 2
    model = LSTMModel(input_size, hidden_size, num_layers)

    # 모델을 GPU로 이동 (CUDA 사용 여부 체크)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 모델 훈련
    train_model(model, train_loader, val_loader)

    # 테스트 평가
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.unsqueeze(1)
            test_outputs = model(test_x)
            test_loss += nn.MSELoss()(test_outputs, test_y.view(-1, 1)).item()
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')
