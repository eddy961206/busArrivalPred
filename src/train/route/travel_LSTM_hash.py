import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
import hashlib

# 1. Hash 함수 정의
def hash_id(id_str, num_buckets=20000):
    """
    문자열 ID를 해시 버킷으로 변환하는 함수입니다.

    Args:
        id_str (str): 해시할 문자열 ID.
        num_buckets (int): 해시 버킷의 총 개수.

    Returns:
        int: 해시 버킷 인덱스.
    """
    return int(hashlib.md5(id_str.encode()).hexdigest(), 16) % num_buckets

# 2. Configuration 설정
config = {
    "train_csv": '../../../dataset/train/route/LSTM/train_04m_filtered.csv',
    "val_csv": '../../../dataset/train/route/LSTM/train_05m_1ws_filtered.csv',
    "test_csv": '../../../dataset/train/route/LSTM/train_05m_2ws_filtered.csv',
    "train_epochs": 30,
    "n_trial": 4,
    "trial_epochs": 5,
    "patience": 10,
    "min_delta": 0.0001,
    "seq_length": 5,          # 시퀀스 길이 설정
    "num_buckets": 20000      # 해시 버킷 크기 설정 (충돌 방지를 위해 충분히 크게 설정)
}

# 3. Dataset 클래스 정의 (해시 기반 인코딩 적용)
class TravelTimeDataset(Dataset):
    def __init__(self, data, mode='train', seq_length=5, num_buckets=20000):
        self.mode = mode  # 'train', 'val', 'test'
        self.seq_length = seq_length
        self.num_buckets = num_buckets

        # 해시 함수 사용하여 BUSROUTE_ID와 BUSINFOUNIT_ID를 해시 버킷으로 변환
        data['BUSROUTE_HASH'] = data['BUSROUTE_ID'].apply(lambda x: hash_id(x, self.num_buckets))
        data['BUSINFOUNIT_HASH'] = data['BUSINFOUNIT_ID'].apply(lambda x: hash_id(x, self.num_buckets))

        # DEP_TIME을 datetime 형식으로 변환
        # data['DEP_TIME'] = pd.to_datetime(data['DEP_TIME'], format='%H:%M')
        data['DEP_TIME'] = pd.to_datetime(data['DEP_TIME'], format='%Y-%m-%d %H:%M:%S')

        # 필요한 컬럼 선택
        if self.mode == 'inference':
            data = data[['DAY_TYPE', 'BUSROUTE_HASH', 'BUSINFOUNIT_HASH', 'LEN', 'DEP_TIME']]
        else:
            data = data[['DAY_TYPE', 'BUSROUTE_HASH', 'BUSINFOUNIT_HASH', 'LEN', 'DEP_TIME', 'TIME_GAP']]

        # 그룹화하여 시퀀스 생성
        grouped = data.groupby(['BUSROUTE_HASH', 'BUSINFOUNIT_HASH'])

        self.sequences = []
        self.targets = []
        self.busroute_hashes = []
        self.businfo_unit_hashes = []
        self.target_indices = []  # 타겟 인덱스 저장

        for (busroute_hash, businfo_hash), group in grouped:
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

            if self.mode != 'inference':
                targets = group['TIME_GAP'].values

            # 시퀀스 생성
            for i in range(len(group) - seq_length):
                seq_features = features[i:i + seq_length]

                self.sequences.append(seq_features)
                self.busroute_hashes.append(busroute_hash)
                self.businfo_unit_hashes.append(businfo_hash)

                # 타겟 인덱스 저장
                target_idx = group.index[i + seq_length]
                self.target_indices.append(target_idx)

                if self.mode != 'inference':
                    seq_target = targets[i + seq_length]
                    self.targets.append(seq_target)

        self.sequences = np.array(self.sequences)
        self.busroute_hashes = np.array(self.busroute_hashes)
        self.businfo_unit_hashes = np.array(self.businfo_unit_hashes)

        if self.mode != 'inference':
            self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        features = torch.tensor(self.sequences[idx], dtype=torch.float)
        busroute_hash = torch.tensor(self.busroute_hashes[idx], dtype=torch.long)
        businfo_unit_hash = torch.tensor(self.businfo_unit_hashes[idx], dtype=torch.long)
        target_idx = self.target_indices[idx]

        if self.mode == 'inference':
            return features, busroute_hash, businfo_unit_hash, target_idx
        else:
            time_gap = torch.tensor(self.targets[idx], dtype=torch.float)
            # 거리 기반 가중치 (예시)
            distance_weight = torch.tensor(3.0 if self.sequences[idx][-1][3] >= 0.5 else 1.0, dtype=torch.float)
            return features, busroute_hash, businfo_unit_hash, time_gap, distance_weight

# 4. LSTM 모델 정의 (해시 기반 인코딩 적용)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_buckets, embedding_dim, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.busroute_embedding = nn.Embedding(num_buckets, embedding_dim)
        self.businfo_embedding = nn.Embedding(num_buckets, embedding_dim)

        self.busroute_dropout = nn.Dropout(dropout)
        self.businfo_dropout = nn.Dropout(dropout)

        # LSTM 레이어
        self.lstm = nn.LSTM(input_size + 2 * embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, busroute_hash, businfo_unit_hash):
        """
        순전파 메서드입니다.

        Args:
            x (torch.Tensor): 입력 시퀀스 (batch_size, seq_length, input_size).
            busroute_hash (torch.Tensor): BUSROUTE_HASH 인덱스 (batch_size).
            businfo_unit_hash (torch.Tensor): BUSINFOUNIT_HASH 인덱스 (batch_size).

        Returns:
            torch.Tensor: 예측된 TIME_GAP 값 (batch_size, 1).
        """
        # 임베딩
        busroute_embedded = self.busroute_embedding(busroute_hash)      # (batch_size, embedding_dim)
        businfo_embedded = self.businfo_embedding(businfo_unit_hash)    # (batch_size, embedding_dim)

        # Dropout 적용
        busroute_embedded = self.busroute_dropout(busroute_embedded)
        businfo_embedded = self.businfo_dropout(businfo_embedded)

        # 시퀀스 길이에 맞게 임베딩 확장
        seq_length = x.size(1)
        busroute_embedded = busroute_embedded.unsqueeze(1).repeat(1, seq_length, 1)  # (batch_size, seq_length, embedding_dim)
        businfo_embedded = businfo_embedded.unsqueeze(1).repeat(1, seq_length, 1)    # (batch_size, seq_length, embedding_dim)

        # 입력에 임베딩 추가
        x = torch.cat((x, busroute_embedded, businfo_embedded), dim=2)  # (batch_size, seq_length, input_size + 2*embedding_dim)

        # LSTM 통과
        out, _ = self.lstm(x)  # out: (batch_size, seq_length, hidden_size)

        # 마지막 타임스텝의 출력 사용
        out = self.fc(out[:, -1, :])  # (batch_size, 1)
        return out

# 5. 훈련 함수 정의
def train_model(model, train_loader, val_loader, num_epochs=None, learning_rate=0.001):
    if num_epochs is None:
        num_epochs = config['train_epochs']

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # tqdm을 사용하여 진행상황 표시
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            features, busroute_hash, businfo_unit_hash, time_gap, distance_weight = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(features, busroute_hash, businfo_unit_hash)

            # 가중치가 적용된 MSE 손실 계산
            loss = weighted_mse_loss(outputs.view(-1), time_gap, distance_weight)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # 검증 단계
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_features, val_busroute_hash, val_businfo_unit_hash, val_time_gap, val_distance_weight = [b.to(device) for b in val_batch]
                val_outputs = model(val_features, val_busroute_hash, val_businfo_unit_hash)

                val_loss = weighted_mse_loss(val_outputs.view(-1), val_time_gap, val_distance_weight)
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # 조기 종료 로직
        if avg_val_loss < best_val_loss - config['min_delta']:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 최적의 모델 저장 (필요 시 주석 해제)
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print("Early stopping triggered")
                break

# 6. 데이터 로드 및 DataLoader 생성 함수 수정
def load_data(config, batch_size=4096):
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

    # Dataset 생성 시, 해시 버킷 크기 전달
    train_dataset = TravelTimeDataset(train_data, mode='train', seq_length=config['seq_length'], num_buckets=config['num_buckets'])
    val_dataset = TravelTimeDataset(val_data, mode='val', seq_length=config['seq_length'], num_buckets=config['num_buckets'])
    test_dataset = TravelTimeDataset(test_data, mode='test', seq_length=config['seq_length'], num_buckets=config['num_buckets'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate_model(model, data_loader):
    model.eval()
    total_mse = 0
    total_mae = 0
    with torch.no_grad():
        for batch in data_loader:
            features, busroute_hash, businfo_unit_hash, time_gap, distance_weight = [b.to(device) for b in batch]
            outputs = model(features, busroute_hash, businfo_unit_hash)
            mse = weighted_mse_loss(outputs.view(-1), time_gap, distance_weight)
            mae = torch.mean(torch.abs(outputs.view(-1) - time_gap))
            total_mse += mse.item()
            total_mae += mae.item()
    avg_mse = total_mse / len(data_loader)
    avg_mae = total_mae / len(data_loader)
    return avg_mse, avg_mae

# 7. 가중치가 적용된 MSE 손실 함수 정의
def weighted_mse_loss(output, target, weight):
    return (weight * (output - target) ** 2).mean()

# 8. 메인 실행 코드
if __name__ == "__main__":
    # 데이터 로드
    train_loader, val_loader, test_loader = load_data(config, batch_size=4096)

    # LSTM 모델 초기화
    input_size = train_loader.dataset.sequences.shape[2]  # 특징 수
    hidden_size = 64
    num_layers = 2
    embedding_dim = 16  # 임베딩 차원
    num_buckets = config['num_buckets']  # 해시 버킷 크기

    model = LSTMModel(input_size, hidden_size, num_layers, num_buckets, embedding_dim, dropout=0.2)

    # GPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 모델 훈련
    train_model(model, train_loader, val_loader)

    # 테스트 평가
    test_mse, test_mae = evaluate_model(model, test_loader)
    print(f'Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}')

