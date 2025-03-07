import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import os
import hashlib
import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_contour, plot_slice
import json
import warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

config = {
    "train_csv": 'dataset/travel_time_7,8_TUE_filtered_70_sl_train.csv',
    "val_csv": 'dataset/travel_time_7,8_TUE_filtered_70_sl_val.csv',
    "test_csv": 'dataset/travel_time_7,8_TUE_filtered_70_sl_test.csv',
    "train_epochs": 30,
    "n_trial": 5,
    "trial_epochs": 5,
    "patience": 10,
    "min_delta": 0.0001
}

class TravelTimeDataset(Dataset):
    def __init__(self, data, hash_size=1000):
        self.hash_size = hash_size
        # 모든 요일을 포함하는 리스트
        all_days = [1, 2, 3, 4, 5, 6, 7]
        self.day_type = pd.get_dummies(data['DAY_TYPE']).reindex(columns=all_days, fill_value=0).astype(float).values

        #######  기본 특성들  #######
        self.len = data['LEN'].values
        self.dep_time = pd.to_datetime(data['DEP_TIME'], format='%H:%M')
        self.dep_hour = self.dep_time.dt.hour
        self.dep_minute = self.dep_time.dt.minute
        self.time_gap = data['TIME_GAP'].values
        self.speed_kmh = data['speed_kmh'].values
        self.businfo_unit_id = data['BUSINFOUNIT_ID'].apply(self.hash_function).values

        # 거리(LEN) 표준화
        scaler = StandardScaler()
        self.len_scaled = scaler.fit_transform(self.len.reshape(-1, 1)).flatten()

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

        # 시간대를 나타내는 특성 (예: 자정~새벽, 아침, 점심, 저녁)
        self.time_of_day = np.select(
            [
                (self.dep_hour >= 0) & (self.dep_hour < 6),   # 자정~새벽
                (self.dep_hour >= 6) & (self.dep_hour < 12),  # 아침
                (self.dep_hour >= 12) & (self.dep_hour < 18), # 점심
                (self.dep_hour >= 18) & (self.dep_hour < 24)  # 저녁
            ],
            [0, 1, 2, 3],  # 각 시간대에 대한 레이블
            default=0      # 예기치 않은 경우, 기본값을 0으로 설정
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
            self.len,             # 구간 길이 (LEN)
            self.speed_kmh,       # 속도 (km/h)
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
        return len(self.time_gap)

    def __getitem__(self, idx):
        # 거리에 따른 가중치 계산 (장거리일수록 가중치 증가)
        distance_weight = 2.0 if self.len[idx] >= 1500 else 1.0  # 1500m 이상일 때 가중치 2배

        return (
            torch.tensor(self.features[idx], dtype=torch.float),
            torch.tensor(self.businfo_unit_id[idx], dtype=torch.long),
            torch.tensor(self.time_gap[idx], dtype=torch.float),
            torch.tensor(distance_weight, dtype=torch.float)  # 가중치 추가
        )


class TravelTimeModel(nn.Module):
    def __init__(self, input_size, hash_size, unit_embedding_dim, dropout_rate):
        super(TravelTimeModel, self).__init__()

        # 텍스트 임베딩 레이어
        self.unit_embedding = nn.Embedding(num_embeddings=hash_size, embedding_dim=unit_embedding_dim)

        # 더 많은 뉴런과 레이어 추가
        self.fc1 = nn.Linear(input_size + unit_embedding_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)

        # 활성화 함수, 드롭아웃, 배치 정규화 추가
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(32)

    def forward(self, features, businfo_unit_id):
        unit_emb = self.unit_embedding(businfo_unit_id).squeeze(1)
        inputs = torch.cat([features, unit_emb], dim=1)

        x = self.leaky_relu(self.bn1(self.fc1(inputs)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        output = self.fc6(x)
        return output


# 3. 데이터 로드 및 전처리
def load_data(config, batch_size=32, hash_size=1000):
    def load_and_clean_data(csv_path):  # 빈 행, 값 0인 행, 띄어쓰기 제거 후 인덱스 리셋
        dtype_spec = {
            'DAY_TYPE': 'int8',
            'BUSINFOUNIT_ID': 'str',
            'LEN': 'int32',
            'DEP_TIME': 'str',
            'TIME_GAP': 'int32',
            'speed_kmh': 'int8'
        }
        usecols = ['DAY_TYPE', 'BUSINFOUNIT_ID', 'LEN', 'DEP_TIME', 'TIME_GAP', 'speed_kmh']
        return pd.read_csv(csv_path, skipinitialspace=True, usecols=usecols, dtype=dtype_spec).loc[
            lambda df: (df != 0).all(axis=1)
        ].dropna(how='all').reset_index(drop=True)

    train_data = load_and_clean_data(config['train_csv'])
    val_data = load_and_clean_data(config['val_csv'])
    test_data = load_and_clean_data(config['test_csv'])

    # 2. 전처리된 데이터를 사용하는 DataLoader 생성
    train_dataset = TravelTimeDataset(train_data, hash_size=hash_size)
    val_dataset = TravelTimeDataset(val_data, hash_size=hash_size)
    test_dataset = TravelTimeDataset(test_data, hash_size=hash_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Early Stopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience  # 개선이 없는 에포크를 얼마나 기다릴지 설정
        self.min_delta = min_delta  # 개선이라고 간주할 최소 변화
        self.best_loss = None  # 가장 낮은 손실값
        self.counter = 0  # 개선되지 않은 에포크 수

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # 개선되었으므로 카운터를 리셋
        else:
            self.counter += 1  # 개선되지 않음
            if self.counter >= self.patience:
                return True  # 조기 종료 시그널
        return False

# 4-1. 목적 함수 정의 (Optuna 통합)
def objective(trial):
    # 하이퍼파라미터 제안
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    unit_embedding_dim = trial.suggest_int('unit_embedding_dim', 64, 256)  # log2(8120) ≈ 13
    hash_size = trial.suggest_int('hash_size', 12000, 20000)    # 구간ID 8120 개 -> (고유 식별자 수의 1.5배 정도) 소수

    num_epochs = config['trial_epochs']

    # 데이터 로드 및 전처리 (배치 사이즈 변경 반영)
    train_loader, val_loader, test_loader = load_data(config, batch_size=batch_size, hash_size=hash_size)

    # 모델 설정
    # day_type_size = len(train_loader.dataset.day_type[0])
    # input_size = day_type_size + 2  # day_type + len_feature + dep_hour_min

    num_features = train_loader.dataset.features.shape[1]  # features의 열 개수 (특성의 수)
    input_size = num_features  # features의 열 개수

    trial_params = {
        'lr': lr,
        'dropout_rate': dropout_rate,
        'batch_size': batch_size,
        'unit_embedding_dim': unit_embedding_dim,
        'hash_size': hash_size,
        'input_size': input_size
    }
    print(f"\nStarting Trial {trial.number} with parameters: \n{trial_params}")

    # 모델 정의
    model = TravelTimeModel(input_size, hash_size, unit_embedding_dim, dropout_rate).to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 학습률 스케줄러 정의
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # TensorBoard 기록 저장 폴더 설정
    trial_log_dir = os.path.join(paths['logs'], f'trial_{trial.number}')
    writer = SummaryWriter(log_dir=trial_log_dir)

    # 모델 학습 (조기 종료를 적용한 버전)
    early_stopping = EarlyStopping(patience=config['patience'], min_delta=config['min_delta'])
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')  # 검증 손실의 최솟값을 저장할 변수

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # tqdm을 사용하여 배치 단위로 진행 상황을 표시
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")):
            features, businfo_unit_id, time_gap, distance_weight = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(features, businfo_unit_id)
            # loss = criterion(outputs.view(-1), time_gap.view(-1))
            # weighted_mse_loss 사용하여 가중치 적용
            loss = weighted_mse_loss(outputs.view(-1), time_gap.view(-1), distance_weight)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # 각 배치의 손실을 일정 간격으로 출력 (예: 10000 배치마다)
            if (i + 1) % 10000 == 0:
                writer.add_scalar('Batch/Train Loss', loss.item(), epoch * len(train_loader) + i)

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss, val_mae = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # 최솟값 검증 손실을 업데이트
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 모델 저장 경로 설정 및 모델 저장
            best_model_path = os.path.join(paths['models'], f'travel_time_model_trial_{trial.number}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), best_model_path)

        # TensorBoard에 기록
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalars('Epoch/Train and Validation Loss', {
            'Train': train_loss,
            'Validation': val_loss
        }, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}'
              f', Validation MAE: {val_mae:.4f}초')

        # 스케줄러를 통해 학습률 조정
        scheduler.step(val_loss)

        # Optuna에 현재 에포크의 결과를 보고
        trial.report(val_loss, epoch)

        # trial 조기 종료 조건 체크
        if trial.should_prune():
            writer.close()
            raise optuna.exceptions.TrialPruned()

        if early_stopping(val_loss):
            print("Early stopping triggered")
            break

    writer.close()

    return best_val_loss

# 4-2. 학습 함수 정의
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=None, hyperparams=None):
    if num_epochs is None:
        num_epochs = config['train_epochs']

    early_stopping = EarlyStopping(patience=config['patience'], min_delta=config['min_delta'])  # Early Stopping 적용
    train_losses = []
    val_losses = []

    # TensorBoard 기록 저장 폴더 설정
    log_dir = os.path.join(paths['logs'], f'train_{short_params_string(hyperparams)}')
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # tqdm을 사용하여 배치 단위로 진행 상황을 표시
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")):
            features, businfo_unit_id, time_gap, distance_weight = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(features, businfo_unit_id)
            # loss = criterion(outputs.view(-1), time_gap.view(-1))
            # weighted_mse_loss 사용하여 가중치 적용
            loss = weighted_mse_loss(outputs.view(-1), time_gap.view(-1), distance_weight)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # 각 배치의 손실을 일정 간격으로 출력 (예: 1000 배치마다)
            if (i + 1) % 10000 == 0:
                # TensorBoard에 기록
                writer.add_scalar('Batch/Train Loss', loss.item(), epoch * len(train_loader) + i)

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss, val_mae = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # TensorBoard에 기록
        writer.add_scalar('Epoch/Train Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Validation Loss', val_loss, epoch)
        writer.add_scalars('Epoch/Train and Validation Loss', {
            'Train': train_loss,
            'Validation': val_loss
        }, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}'
              f', Validation MAE: {val_mae:.4f}초')

        # val_loss 최저일 때 모델 저장
        if val_loss == min(val_losses):
            best_model = deepcopy(model.state_dict())

        # 스케줄러를 통해 학습률 조정
        scheduler.step(val_loss)

        if early_stopping(val_loss):
            print("Early stopping triggered")
            break

    model.load_state_dict(best_model)

    writer.close()  # 훈련이 끝난 후 writer 닫기

    return model, train_losses, val_losses

# 5. 평가 함수 정의
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    with torch.no_grad():
        for batch in val_loader:
            features, businfo_unit_id, time_gap, distance_weight = [b.to(device) for b in batch]
            outputs = model(features, businfo_unit_id)
            # loss
            # loss = criterion(outputs.view(-1), time_gap.view(-1))
            # weighted_mse_loss 사용하여 가중치 적용
            loss = weighted_mse_loss(outputs.view(-1), time_gap.view(-1), distance_weight)
            total_loss += loss.item()
            # mae
            mae = torch.mean(torch.abs(outputs.view(-1) - time_gap.view(-1)))
            total_mae += mae.item()

    return total_loss / len(val_loader), total_mae / len(val_loader)

# 6. 테스트 결과 비교 및 시각화
def evaluate_predictions(test_loader, model, device, num_samples=20):
    model.eval()
    actuals = []
    predictions = []

    with torch.no_grad():  # 기울기 계산 비활성화
        for batch in test_loader:
            features, businfo_unit_id, time_gap, distance_weight = [b.to(device) for b in batch]
            outputs = model(features, businfo_unit_id)

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
    plot_name = os.path.join(paths['plots'], f'travel_time_comparison_test_{short_params_string(best_params)}.png')
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


###########  기타 함수들  ###########
def weighted_mse_loss(output, target, weight):
    # 거리 특성에 따라 가중치 조정 (가중치 손실 함수). 오차율이 큰 장거리 구간에서 손실을 더 크게 반영
    return (weight * (output - target) ** 2).mean()

def create_directories(base_dir, sub_dirs):
    paths = {}
    for sub_dir in sub_dirs:
        path = os.path.join(base_dir, sub_dir)
        os.makedirs(path, exist_ok=True)
        paths[sub_dir] = path
    return paths

def short_params_string(params):
    # learning rate, dropout rate, batch size, unit dim, hash size
    key_params = ['lr', 'dr', 'bs', 'ud', 'hs']
    values = [f"{params['lr']:.2e}", f"{params['dropout_rate']:.2e}", str(params['batch_size']),
              str(params['unit_embedding_dim']), str(params['hash_size'])]
    return "_".join(f"{k}{v}" for k, v in zip(key_params, values))

def save_optuna_visualizations(study, output_dir):
    try:
        # 최적화 히스토리 시각화 및 저장 (Matplotlib)
        plot_optimization_history(study)
        plt.savefig(f"{output_dir}/optimization_history.png")
        plt.close()

        # 파라미터 중요도 시각화 및 저장 (Matplotlib)
        plot_param_importances(study)
        plt.savefig(f"{output_dir}/param_importances.png")
        plt.close()

        # 평행 좌표 시각화 및 저장 (Matplotlib)
        plot_parallel_coordinate(study)
        plt.savefig(f"{output_dir}/parallel_coordinate.png")
        plt.close()

        # 컨투어 플롯 시각화 및 저장 (Matplotlib)
        plot_contour(study)
        plt.savefig(f"{output_dir}/contour_plot.png")
        plt.close()

        # 슬라이스 플롯 시각화 및 저장 (Matplotlib)
        plot_slice(study)
        plt.savefig(f"{output_dir}/slice_plot.png")
        plt.close()

        print(f"All visualizations saved in {output_dir}")
    except Exception as e:
        print(f"An error occurred while saving visualizations: {str(e)}")



###########  메인 코드  ###########
# 7. 학습 및 테스트 코드 실행
if __name__ == "__main__":
    # 결과 저장 폴더 기본 경로 생성
    base_dir = f'results_travel_time/{time.strftime("%Y%m%d-%H%M%S")}'
    sub_dirs = ['models', 'plots', 'optuna_vis', 'logs']    # 하위 디렉토리 설정
    paths = create_directories(base_dir, sub_dirs)

    # GPU 사용 여부 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optuna를 사용해 하이퍼파라미터 최적화 실행
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=2))
    study.optimize(objective, n_trials=config['n_trial'])

    # Optuna 시각화 결과 저장
    save_optuna_visualizations(study, paths['optuna_vis'])

    # 최적의 하이퍼파라미터 출력/저장
    best_params = study.best_trial.params
    print(f"Best Parameters: {best_params}")
    with open(f"{paths['models']}/best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # best_params = {
    #     'batch_size': 64,
    #     'unit_embedding_dim': 222,
    #     'lr': 0.00011369873848381511,
    #     'dropout_rate': 0.34318293607839695,
    #     'hash_size': 16128
    # }

    # 최적의 하이퍼파라미터로 모델 다시 학습 및 테스트
    batch_size = best_params['batch_size']
    unit_embedding_dim = best_params['unit_embedding_dim']
    lr = best_params['lr']
    dropout_rate = best_params['dropout_rate']
    hash_size = best_params['hash_size']

    # 데이터 로드
    train_loader, val_loader, test_loader = load_data(
        config,
        batch_size=batch_size,
        hash_size=hash_size
    )

    # 모델 설정
    # day_type_size = len(train_loader.dataset.day_type[0])
    # input_size = day_type_size + 2  # day_type + len_feature + dep_hour_min

    num_features = train_loader.dataset.features.shape[1]  # features의 열 개수 (특성의 수)
    input_size = num_features  # features의 열 개수
    print(f"\ninput_size: {input_size}")

    # 모델 정의
    model = TravelTimeModel(input_size, hash_size, unit_embedding_dim, dropout_rate).to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # 모델 학습
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer,
                                                  scheduler, device, num_epochs=config['train_epochs'],
                                                  hyperparams=best_params)

    # 테스트 결과 평가
    evaluate_predictions(test_loader, model, device, num_samples=20)

    # 학습된 모델 저장
    model_name = os.path.join(paths['models'], f'travel_time_model_{short_params_string(best_params)}.pth')
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}")

    # 학습 및 평가 후 손실 그래프 시각화
    def plot_losses(train_losses, val_losses):
        df = pd.DataFrame({
            'Epoch': range(1, len(train_losses) + 1),
            'Train Loss': train_losses,
            'Validation Loss': val_losses
        })

        plt.figure(figsize=(10, 6))
        plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', marker='o')
        plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plot_name = os.path.join(paths['plots'], f'loss_plot_travel_{short_params_string(best_params)}.png')
        plt.savefig(plot_name)
        plt.close()  # 그래프 저장 후 닫기
        print(f"Plot saved as {plot_name}")

    # 손실 시각화
    plot_losses(train_losses, val_losses)
