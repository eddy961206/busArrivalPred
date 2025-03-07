import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import os

# 1. 데이터셋 클래스 정의
class TravelTimeDataset(Dataset):
    def __init__(self, data):
        # 전처리된 DataFrame을 받습니다.
        self.day_type = pd.get_dummies(data['DAY_TYPE']).values
        self.businfo_unit_id = data['BUSINFOUNIT_ID'].values
        self.len = data['LEN'].values
        self.dep_hour_min = (pd.to_datetime(data['DEP_TIME'], format='%H:%M:%S').dt.hour * 60 # 특성 생성
                                + pd.to_datetime(data['DEP_TIME'], format='%H:%M:%S').dt.minute)
        self.time_gap = data['TIME_GAP'].values

    def __len__(self):
        return len(self.time_gap)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.day_type[idx], dtype=torch.float),
            torch.tensor(self.businfo_unit_id[idx], dtype=torch.long),
            torch.tensor(self.len[idx], dtype=torch.float),
            torch.tensor(self.dep_hour_min[idx], dtype=torch.float),
            torch.tensor(self.time_gap[idx], dtype=torch.float)
        )

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

# 3. 데이터 로드 및 전처리
def load_data(train_csv, val_csv, test_csv, batch_size=32):
    # 1. 데이터 로드 및 결합
    train_data = pd.read_csv(train_csv)
    val_data = pd.read_csv(val_csv)
    test_data = pd.read_csv(test_csv)

    # 모든 데이터를 결합
    all_data = pd.concat([train_data, val_data, test_data])

    # 2. LabelEncoder를 사용해 전체 데이터에 대해 인코딩
    label_encoder_unit = LabelEncoder()

    all_data['BUSINFOUNIT_ID'] = label_encoder_unit.fit_transform(all_data['BUSINFOUNIT_ID'].astype(str))

    # 3. 인코딩 결과를 각각의 데이터셋에 다시 반영
    train_data['BUSINFOUNIT_ID'] = label_encoder_unit.transform(train_data['BUSINFOUNIT_ID'].astype(str))
    val_data['BUSINFOUNIT_ID'] = label_encoder_unit.transform(val_data['BUSINFOUNIT_ID'].astype(str))
    test_data['BUSINFOUNIT_ID'] = label_encoder_unit.transform(test_data['BUSINFOUNIT_ID'].astype(str))

    # 4. 전처리된 데이터를 사용하는 DataLoader 생성
    train_dataset = TravelTimeDataset(train_data)
    val_dataset = TravelTimeDataset(val_data)
    test_dataset = TravelTimeDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, label_encoder_unit

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

# 4. 학습 함수 정의
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10):
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)  # Early Stopping 적용
    train_losses = []
    val_losses = []

    # TensorBoard 로그 작성기(SummaryWriter) 초기화
    writer = SummaryWriter(log_dir=f'runs/travel_time/')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # tqdm을 사용하여 배치 단위로 진행 상황을 표시
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")):
            day_type, businfo_unit_id, len_feature, dep_hour_min, time_gap = [d.to(device) for d in data]
            optimizer.zero_grad()
            outputs = model(day_type, businfo_unit_id, len_feature, dep_hour_min)
            loss = criterion(outputs.view(-1), time_gap.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # TensorBoard에 기록
            writer.add_scalar('Batch/Train Loss', loss.item(), epoch * len(train_loader) + i)

            # # 각 배치의 손실을 일정 간격으로 출력 (예: 100 배치마다)
            # if (i + 1) % 1000 == 0:
            #     print(f"Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # TensorBoard에 기록
        writer.add_scalar('Epoch/Train Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Validation Loss', val_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

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
    with torch.no_grad():
        for data in val_loader:
            day_type, businfo_unit_id, len_feature, dep_hour_min, time_gap = [d.to(device) for d in data]
            outputs = model(day_type, businfo_unit_id, len_feature, dep_hour_min)
            loss = criterion(outputs.view(-1), time_gap.view(-1))
            total_loss += loss.item()

    return total_loss / len(val_loader)

# 6. 테스트 결과 비교 및 시각화
def evaluate_predictions(test_loader, model, device, num_samples=10):
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
    plot_name = os.path.join(results_folder, f'travel_time_comparison_test_{start_time}.png')
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

# 7. 학습 및 테스트 코드 실행
if __name__ == "__main__":
    # 결과 저장 폴더 생성
    start_time = time.strftime("%Y%m%d-%H%M%S")
    results_folder = 'results_travel_time'
    os.makedirs(results_folder, exist_ok=True)

    # GPU 사용 여부 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 배치 사이즈 설정
    batch_size = 128

    # 데이터 로드 및 전처리
    train_loader, val_loader, test_loader, label_encoder_unit = load_data(
        'dataset/travel_time_07280400_08110359_train_noOutliers.csv',
        'dataset/travel_time_07280400_08110359_val_noOutliers.csv',
        'dataset/travel_time_07280400_08110359_test_noOutliers.csv',
        batch_size=batch_size
    )

    # 모델 설정
    day_type_size = len(train_loader.dataset.day_type[0])
    num_units = len(label_encoder_unit.classes_)

    # 임베딩 차원 계산
    unit_embedding_dim = int(np.sqrt(num_units))

    input_size = day_type_size + 2  # day_type + len_feature + dep_hour_min

    model = TravelTimeModel(input_size, num_units, unit_embedding_dim).to(device)

    # 손실 함수 정의
    criterion = nn.MSELoss()

    # 옵티마이저 정의 (L2 정규화 포함)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 학습률 스케줄러 정의
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # 모델 학습
    num_epochs = 30
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler
                                                  , device, num_epochs=num_epochs)

    # 테스트 결과 평가
    evaluate_predictions(test_loader, model, device, num_samples=20)

    # 학습된 모델 저장
    model_name = os.path.join(results_folder, f'travel_time_model_{start_time}.pth')
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
        plot_name = os.path.join(results_folder, f'loss_plot_travel_{start_time}.png')
        plt.savefig(plot_name)
        plt.close()  # 그래프 저장 후 닫기
        print(f"Plot saved as {plot_name}")

    # 손실 시각화
    plot_losses(train_losses, val_losses)

