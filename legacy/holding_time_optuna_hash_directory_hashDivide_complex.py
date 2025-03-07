import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
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

# 1. 데이터셋 클래스 정의
class HoldingTimeDataset(Dataset):
    def __init__(self, data, route_hash_size=1000, location_hash_size=10000):
        self.route_hash_size = route_hash_size
        self.location_hash_size = location_hash_size
        # 모든 요일을 포함하는 리스트
        all_days = [1, 2, 3, 4, 5, 6, 7]
        self.day_type = pd.get_dummies(data['DAY_TYPE']).reindex(columns=all_days, fill_value=0).astype(float).values
        self.busroute_id = data['BUSROUTE_ID'].apply(lambda x: self.hash_function(x, self.route_hash_size)).values
        self.location_id = data['LOCATION_ID'].apply(lambda x: self.hash_function(x, self.location_hash_size)).values
        self.dep_hour_min = (pd.to_datetime(data['DEP_TIME'], format='%H:%M').dt.hour * 60  # 특성 생성
                             + pd.to_datetime(data['DEP_TIME'], format='%H:%M').dt.minute)
        self.time_gap = data['TIME_GAP'].values

    def hash_function(self, x, hash_size):
        # 입력이 문자열이 아닐 경우 문자열로 변환
        if not isinstance(x, str):
            x = str(x)
        return int(hashlib.md5(x.encode()).hexdigest(), 16) % hash_size

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
    def __init__(self, input_size, route_hash_size, location_hash_size, route_embedding_dim, location_embedding_dim, dropout_rate):
        super(HoldingTimeModel, self).__init__()

        # 텍스트 임베딩 레이어
        self.route_embedding = nn.Embedding(num_embeddings=route_hash_size, embedding_dim=route_embedding_dim)
        self.location_embedding = nn.Embedding(num_embeddings=location_hash_size, embedding_dim=location_embedding_dim)

        # 전결합 레이어
        self.fc1 = nn.Linear(input_size + route_embedding_dim + location_embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 1)

        # 활성화 함수 및 기타 레이어
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(16)

    def forward(self, day_type, busroute_id, location_id, dep_hour_min):
        route_emb = self.route_embedding(busroute_id).squeeze(1)
        location_emb = self.location_embedding(location_id).squeeze(1)
        inputs = torch.cat([day_type, route_emb, location_emb, dep_hour_min.unsqueeze(1)], dim=1)

        x = self.leaky_relu(self.bn1(self.fc1(inputs)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn4(self.fc4(x)))
        output = self.output_layer(x)
        return output

# 3. 데이터 로드 및 전처리
def load_data(train_csv, val_csv, test_csv, batch_size=64, route_hash_size=1000, location_hash_size=10000):
    # 1. 데이터 로드
    def load_and_clean_data(csv_path):
        return pd.read_csv(csv_path, skipinitialspace=True).loc[
            lambda df: (df != 0).all(axis=1)
        ].dropna(how='all').reset_index(drop=True)
    train_data = load_and_clean_data(train_csv)
    val_data = load_and_clean_data(val_csv)
    test_data = load_and_clean_data(test_csv)

    # 2. 전처리된 데이터를 사용하는 DataLoader 생성
    train_dataset = HoldingTimeDataset(train_data, route_hash_size=route_hash_size, location_hash_size=location_hash_size)
    val_dataset = HoldingTimeDataset(val_data, route_hash_size=route_hash_size, location_hash_size=location_hash_size)
    test_dataset = HoldingTimeDataset(test_data, route_hash_size=route_hash_size, location_hash_size=location_hash_size)

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
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    route_hash_size = trial.suggest_int('route_hash_size', 1000, 1500)  # 노선ID 877, 정류장ID 8638개
    location_hash_size = trial.suggest_int('location_hash_size', 10000, 15000)
    route_embedding_dim = trial.suggest_int('route_embedding_dim', 8, 32)   # log2(877) ≈ 10
    location_embedding_dim = trial.suggest_int('location_embedding_dim', 16, 64)  # log2(8638) ≈ 13

    # route_hash_size = trial.suggest_int('route_hash_size', 500, 2000)
    # location_hash_size = trial.suggest_int('location_hash_size', 500, 2000)
    # route_embedding_dim = trial.suggest_int('route_embedding_dim', 4, 64)
    # location_embedding_dim = trial.suggest_int('location_embedding_dim', 4, 64)

    trial_params = {
        'lr': lr,
        'dropout_rate': dropout_rate,
        'batch_size': batch_size,
        'route_embedding_dim': route_embedding_dim,
        'location_embedding_dim': location_embedding_dim,
        'route_hash_size': route_hash_size,
        'location_hash_size': location_hash_size
    }
    print(f"\nStarting Trial {trial.number} with parameters: \n{trial_params}")

    num_epochs = 5

    # 데이터 로드 및 전처리 (배치 사이즈 변경 반영)
    train_loader, val_loader, test_loader = load_data(
        'dataset/holding_time_7.csv',
        'dataset/holding_time_6_val.csv',
        'dataset/holding_time_6_test.csv',
        batch_size=batch_size,
        route_hash_size=route_hash_size,
        location_hash_size=location_hash_size
    )

    # 모델 설정
    day_type_size = len(train_loader.dataset.day_type[0])
    input_size = day_type_size + 1  # day_type + dep_hour_min

    # 모델 정의
    model = HoldingTimeModel(input_size, route_hash_size, location_hash_size, route_embedding_dim,
                             location_embedding_dim, dropout_rate).to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 학습률 스케줄러 정의
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # TensorBoard 기록 저장 폴더 설정
    trial_log_dir = os.path.join(paths['logs'], f'trial_{trial.number}')
    writer = SummaryWriter(log_dir=trial_log_dir)

    # 모델 학습 (조기 종료를 적용한 버전)
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')  # 검증 손실의 최솟값을 저장할 변수

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # tqdm을 사용하여 배치 단위로 진행 상황을 표시
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")):
            day_type, busroute_id, location_id, dep_hour_min, time_gap = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(day_type, busroute_id, location_id, dep_hour_min)
            loss = criterion(outputs.view(-1), time_gap.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # 각 배치의 손실을 일정 간격으로 출력 (예: 1000 배치마다)
            if (i + 1) % 10000 == 0:
                writer.add_scalar('Batch/Train Loss', loss.item(), epoch * len(train_loader) + i)

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # 검증
        val_loss, val_mae = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # 최솟값 검증 손실을 업데이트
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 모델 저장 경로 설정 및 모델 저장
            best_model_path = os.path.join(paths['models'], f'holding_time_model_trial_{trial.number}_epoch_{epoch + 1}.pth')
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
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10, hyperparams=None):
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)  # Early Stopping 적용
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
            day_type, busroute_id, location_id, dep_hour_min, time_gap = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(day_type, busroute_id, location_id, dep_hour_min)
            loss = criterion(outputs.view(-1), time_gap.view(-1))
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
            day_type, busroute_id, location_id, dep_hour_min, time_gap = [b.to(device) for b in batch]
            outputs = model(day_type, busroute_id, location_id, dep_hour_min)
            # loss
            loss = criterion(outputs.view(-1), time_gap.view(-1))
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
            day_type, busroute_id, location_id, dep_hour_min, time_gap = [b.to(device) for b in batch]
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
    plot_name = os.path.join(paths['plots'], f'holding_time_comparison_test_{short_params_string(best_params)}.png')
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
def create_directories(base_dir, sub_dirs):
    paths = {}
    for sub_dir in sub_dirs:
        path = os.path.join(base_dir, sub_dir)
        os.makedirs(path, exist_ok=True)
        paths[sub_dir] = path
    return paths

def short_params_string(params):
    # learning rate, dropout rate, batch size, route dim, location dim, route hash size, location hash size
    key_params = ['lr', 'dr', 'bs', 'rd', 'ld', 'rhs', 'lhs']
    values = [f"{params['lr']:.2e}", f"{params['dropout_rate']:.2e}", str(params['batch_size']),
              str(params['route_embedding_dim']), str(params['location_embedding_dim']),
              str(params['route_hash_size']), str(params['location_hash_size'])]
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
    base_dir = f'results_holding_time/{time.strftime("%Y%m%d-%H%M%S")}'
    sub_dirs = ['models', 'plots', 'optuna_vis', 'logs']    # 하위 디렉토리 설정
    paths = create_directories(base_dir, sub_dirs)

    # GPU 사용 여부 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optuna를 사용해 하이퍼파라미터 최적화 실행
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    study.optimize(objective, n_trials=10)

    # Optuna 시각화 결과 저장
    save_optuna_visualizations(study, paths['optuna_vis'])

    # 최적의 하이퍼파라미터 출력/저장
    best_params = study.best_trial.params
    print(f"Best Parameters: {best_params}")
    with open(f"{paths['models']}/best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # 최적의 하이퍼파라미터로 모델 다시 학습 및 테스트
    batch_size = best_params['batch_size']
    route_embedding_dim = best_params['route_embedding_dim']
    location_embedding_dim = best_params['location_embedding_dim']
    lr = best_params['lr']
    dropout_rate = best_params['dropout_rate']
    route_hash_size = best_params['route_hash_size']
    location_hash_size = best_params['location_hash_size']

    # 데이터 로드
    train_loader, val_loader, test_loader = load_data(
        'dataset/holding_time_7.csv',
        'dataset/holding_time_6_val.csv',
        'dataset/holding_time_6_test.csv',
        batch_size=batch_size,
        route_hash_size=route_hash_size,
        location_hash_size=location_hash_size
    )

    # 모델 설정
    day_type_size = len(train_loader.dataset.day_type[0])
    input_size = day_type_size + 1  # day_type + dep_hour_min

    # 모델 정의
    model = HoldingTimeModel(input_size, route_hash_size, location_hash_size, route_embedding_dim
                             , location_embedding_dim, dropout_rate).to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # 모델 학습
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer,
                                                  scheduler, device, num_epochs=10, hyperparams=best_params)

    # 테스트 결과 평가
    evaluate_predictions(test_loader, model, device, num_samples=20)

    # 학습된 모델 저장
    model_name = os.path.join(paths['models'], f'holding_time_model_{short_params_string(best_params)}.pth')
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
        plot_name = os.path.join(paths['plots'], f'loss_plot_holding_{short_params_string(best_params)}.png')
        plt.savefig(plot_name)
        plt.close()  # 그래프 저장 후 닫기
        print(f"Plot saved as {plot_name}")

    # 손실 시각화
    plot_losses(train_losses, val_losses)
