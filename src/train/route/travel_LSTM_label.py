import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import numpy as np
from tqdm import tqdm

config = {
    "train_csv": 'dataset/train/travel소통시간학습_수요일_filtered_train.csv',
    "val_csv": 'dataset/train/travel소통시간학습_수요일_filtered_val.csv',
    "test_csv": 'dataset/train/travel소통시간학습_수요일_filtered_test.csv',
    "train_epochs": 30,
    "n_trial": 4,
    "trial_epochs": 5,
    "patience": 10,
    "min_delta": 0.0001,
    "seq_length": 5  # 시퀀스 길이 설정
}

# 1. Dataset 클래스 정의 (시퀀스 적용)
class TravelTimeDataset(Dataset):
    def __init__(self, data, mode='train', seq_length=5):
        self.mode = mode  # 'train', 'inference'
        self.seq_length = seq_length

        # 범주형 변수 인코딩 준비
        self.le_busroute = LabelEncoder()
        self.le_businfo = LabelEncoder()

        # 노선ID(BUSROUTE_ID)와 구간ID(BUSINFOUNIT_ID) 인코딩
        data['BUSROUTE_ID_ENC'] = self.le_busroute.fit_transform(data['BUSROUTE_ID'])
        data['BUSINFOUNIT_ID_ENC'] = self.le_businfo.fit_transform(data['BUSINFOUNIT_ID'])

        # DEP_TIME을 datetime 형식으로 변환
        data['DEP_TIME'] = pd.to_datetime(data['DEP_TIME'], format='%H:%M:%S')

        # 필요한 컬럼 선택
        data = data[['DAY_TYPE', 'BUSROUTE_ID_ENC', 'BUSINFOUNIT_ID_ENC', 'LEN', 'DEP_TIME', 'TIME_GAP']]

        # 그룹화하여 시퀀스 생성
        grouped = data.groupby(['BUSROUTE_ID_ENC', 'BUSINFOUNIT_ID_ENC'])

        self.sequences = []
        self.targets = []
        self.busroute_ids = []
        self.businfo_unit_ids = []

        for (busroute_id, businfo_unit_id), group in grouped:
            group = group.sort_values('DEP_TIME')

            # 특징 생성
            group['DEP_HOUR'] = group['DEP_TIME'].dt.hour
            group['DEP_MINUTE'] = group['DEP_TIME'].dt.minute
            group['LEN_SCALED'] = MinMaxScaler().fit_transform(group['LEN'].values.reshape(-1, 1)).flatten()

            # 주기성 특성 생성
            group['DEP_TIME_SIN'] = np.sin(2 * np.pi * group['DEP_HOUR'] / 24)
            group['DEP_TIME_COS'] = np.cos(2 * np.pi * group['DEP_HOUR'] / 24)
            group['DAY_TYPE_SIN'] = np.sin(2 * np.pi * group['DAY_TYPE'] / 7)
            group['DAY_TYPE_COS'] = np.cos(2 * np.pi * group['DAY_TYPE'] / 7)
            group['LOG_LEN'] = np.log1p(group['LEN'])
            group['IS_WEEKEND'] = group['DAY_TYPE'].isin([1, 7]).astype(int)

            # 시간대 구분
            group['TIME_OF_DAY'] = np.select(
                [
                    (group['DEP_HOUR'] >= 0) & (group['DEP_HOUR'] < 6),
                    (group['DEP_HOUR'] >= 6) & (group['DEP_HOUR'] < 12),
                    (group['DEP_HOUR'] >= 12) & (group['DEP_HOUR'] < 18),
                    (group['DEP_HOUR'] >= 18) & (group['DEP_HOUR'] < 24)
                ],
                [0, 1, 2, 3],
                default=2
            )

            # 혼잡 시간대 여부
            group['PEAK_HOURS'] = ((group['DEP_HOUR'] >= 7) & (group['DEP_HOUR'] < 10) |
                                   (group['DEP_HOUR'] >= 18) & (group['DEP_HOUR'] < 20)).astype(int)

            # 거리 유형
            group['DISTANCE_TYPE'] = np.select(
                [
                    group['LEN'] < 500,
                    (group['LEN'] >= 500) & (group['LEN'] < 1500),
                    group['LEN'] >= 1500
                ],
                [0, 1, 2],
                default=0
            )

            # 사용 가능한 컬럼 선택
            feature_cols = [
                'DAY_TYPE', 'DEP_HOUR', 'DEP_MINUTE', 'LEN_SCALED',
                'DEP_TIME_SIN', 'DEP_TIME_COS', 'DAY_TYPE_SIN', 'DAY_TYPE_COS',
                'LOG_LEN', 'IS_WEEKEND', 'TIME_OF_DAY', 'PEAK_HOURS', 'DISTANCE_TYPE'
            ]
            features = group[feature_cols].values
            targets = group['TIME_GAP'].values

            # 시퀀스 생성
            for i in range(len(group) - seq_length):
                seq_features = features[i:i + seq_length]
                seq_target = targets[i + seq_length]

                self.sequences.append(seq_features)
                self.targets.append(seq_target)
                self.busroute_ids.append(busroute_id)
                self.businfo_unit_ids.append(businfo_unit_id)

        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        self.busroute_ids = np.array(self.busroute_ids)
        self.businfo_unit_ids = np.array(self.businfo_unit_ids)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        features = torch.tensor(self.sequences[idx], dtype=torch.float)
        busroute_id = torch.tensor(self.busroute_ids[idx], dtype=torch.long)
        businfo_unit_id = torch.tensor(self.businfo_unit_ids[idx], dtype=torch.long)

        if self.mode == 'inference':
            return features, busroute_id, businfo_unit_id
        else:
            time_gap = torch.tensor(self.targets[idx], dtype=torch.float)
            # 거리 기반 가중치 (예시)
            distance_weight = torch.tensor(3.0 if self.sequences[idx][-1][3] >= 0.5 else 1.0, dtype=torch.float)
            return features, busroute_id, businfo_unit_id, time_gap, distance_weight

# 2. LSTM 모델 정의 (임베딩 추가 및 시퀀스 입력 지원)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, busroute_vocab_size, businfo_vocab_size, embedding_dim):
        super(LSTMModel, self).__init__()

        # 노선ID(BUSROUTE_ID)와 구간ID(BUSINFOUNIT_ID)의 임베딩 레이어
        self.busroute_embedding = nn.Embedding(busroute_vocab_size, embedding_dim)
        self.businfo_embedding = nn.Embedding(businfo_vocab_size, embedding_dim)

        # LSTM 레이어
        self.lstm = nn.LSTM(input_size + 2 * embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, busroute_id, businfo_unit_id):
        # x: (batch_size, seq_length, input_size)
        batch_size, seq_length, _ = x.size()

        # 임베딩
        busroute_embedded = self.busroute_embedding(busroute_id)  # (batch_size, embedding_dim)
        businfo_embedded = self.businfo_embedding(businfo_unit_id)  # (batch_size, embedding_dim)

        # 시퀀스 길이에 맞게 임베딩 확장
        busroute_embedded = busroute_embedded.unsqueeze(1).repeat(1, seq_length, 1)
        businfo_embedded = businfo_embedded.unsqueeze(1).repeat(1, seq_length, 1)

        # 입력에 임베딩 추가
        x = torch.cat((x, busroute_embedded, businfo_embedded), dim=2)

        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 마지막 타임스텝의 출력 사용
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
            features, busroute_id, businfo_unit_id, time_gap, distance_weight = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(features, busroute_id, businfo_unit_id)

            # weighted_mse_loss 사용
            loss = weighted_mse_loss(outputs.view(-1), time_gap, distance_weight)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_features, val_busroute_id, val_businfo_unit_id, val_time_gap, val_distance_weight = [b.to(device) for b in val_batch]
                val_outputs = model(val_features, val_busroute_id, val_businfo_unit_id)

                val_loss = weighted_mse_loss(val_outputs.view(-1), val_time_gap, val_distance_weight)
                total_val_loss += val_loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_train_loss/len(train_loader):.4f}, Val Loss: {total_val_loss/len(val_loader):.4f}')

# 4. 데이터 로드 및 DataLoader 생성
def load_data(config, batch_size=32):
    def load_and_clean_data(csv_path):
        dtype_spec = {
            'DAY_TYPE': 'int8',
            'BUSROUTE_ID': 'str',
            'BUSINFOUNIT_ID': 'str',
            'LEN': 'int32',
            'DEP_TIME': 'str',
            'TIME_GAP': 'int32'
        }
        usecols = ['DAY_TYPE', 'BUSROUTE_ID', 'BUSINFOUNIT_ID', 'LEN', 'DEP_TIME', 'TIME_GAP']
        return pd.read_csv(csv_path, skipinitialspace=True, usecols=usecols, dtype=dtype_spec).dropna(how='all').reset_index(drop=True)

    train_data = load_and_clean_data(config['train_csv'])
    val_data = load_and_clean_data(config['val_csv'])
    test_data = load_and_clean_data(config['test_csv'])

    train_dataset = TravelTimeDataset(train_data, mode='train', seq_length=config['seq_length'])
    val_dataset = TravelTimeDataset(val_data, mode='val', seq_length=config['seq_length'])
    test_dataset = TravelTimeDataset(test_data, mode='test', seq_length=config['seq_length'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def weighted_mse_loss(output, target, weight):
    return (weight * (output - target) ** 2).mean()

# 5. 메인 실행 코드
if __name__ == "__main__":
    # DataLoader 생성
    train_loader, val_loader, test_loader = load_data(config, batch_size=32)

    # LSTM 모델 초기화
    input_size = train_loader.dataset.sequences.shape[2]  # 특징 수
    hidden_size = 64
    num_layers = 2
    embedding_dim = 16  # 임베딩 차원
    busroute_vocab_size = len(train_loader.dataset.le_busroute.classes_)
    businfo_vocab_size = len(train_loader.dataset.le_businfo.classes_)

    model = LSTMModel(input_size, hidden_size, num_layers, busroute_vocab_size, businfo_vocab_size, embedding_dim)

    # 모델을 GPU로 이동 (CUDA 사용 여부 체크)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 모델 훈련
    train_model(model, train_loader, val_loader)

    # 테스트 평가
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for test_batch in test_loader:
            test_features, test_busroute_id, test_businfo_unit_id, test_time_gap, test_weight = [b.to(device) for b in test_batch]
            test_outputs = model(test_features, test_busroute_id, test_businfo_unit_id)
            test_loss += weighted_mse_loss(test_outputs.view(-1), test_time_gap, test_weight).item()

    print(f'Test Loss: {test_loss/len(test_loader):.4f}')
