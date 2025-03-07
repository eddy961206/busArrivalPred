import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import optuna
import matplotlib.pyplot as plt
import torch.nn as nn  
import torch.optim as optim  

# Set paths
data_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(data_dir, 'models', 'RNN')
result_save_path = os.path.join(data_dir, 'results', 'RNN')

# Create directories if they don't exist
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(result_save_path, exist_ok=True)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    df = df.sort_values(by=['DEP_TIME']).reset_index(drop=True)
    df['hh'], df['mm'], df['ss'] = zip(*df['DEP_TIME'].apply(split_time))
    return df

def split_time(dep_time):
    t = datetime.strptime(dep_time, '%H:%M:%S')
    return t.hour, t.minute, t.second

def remove_abnormal_data_by_group(df_group, num_std=2):
    # 예시 1: 결측치 제거
    df_group = df_group.dropna()

    # 예시 2: TIME_GAP이 평균에서 num_std * 표준편차 이상 벗어나는 경우 제거
    mean_time_gap = df_group['TIME_GAP'].mean()
    std_time_gap = df_group['TIME_GAP'].std()

    lower_bound = mean_time_gap - num_std * std_time_gap
    upper_bound = mean_time_gap + num_std * std_time_gap

    df_group = df_group[(df_group['TIME_GAP'] >= lower_bound) & (df_group['TIME_GAP'] <= upper_bound)]

    # 예시 3: hh, mm, ss 값이 논리적 범위를 벗어나는 경우 제거
    df_group = df_group[(df_group['hh'] >= 0) & (df_group['hh'] < 24)]
    df_group = df_group[(df_group['mm'] >= 0) & (df_group['mm'] < 60)]
    df_group = df_group[(df_group['ss'] >= 0) & (df_group['ss'] < 60)]

    return df_group


def process_and_group_data(df):
    # 데이터 그룹화
    grouped = df.groupby(['BUSINFOUNIT_ID', 'DAY_TYPE'])
    
    # 그룹별로 비정상 데이터 제거
    cleaned_groups = []
    for name, group in grouped:
        cleaned_group = remove_abnormal_data_by_group(group)
        cleaned_groups.append(cleaned_group)
        print(f"Group {name} after removing abnormal values: {len(cleaned_group)} records")

    # 모든 그룹을 하나의 데이터프레임으로 결합
    cleaned_groups = pd.concat(cleaned_groups).reset_index(drop=True)
    
    return cleaned_groups

class TimeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, alpha=0.01):
        super(TimeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, hidden_size//2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size//2, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.dropout1(out)   
        out = self.fc(out)
        out = self.dropout2(out)   
        out = self.fc2(out)
        return out

def train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs, patience=10):
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
    
    print(f'Best epoch: {best_epoch + 1} with validation loss: {best_loss:.4f}')
    return best_epoch, best_loss, train_losses, val_losses

def save_training_results(train_losses, val_losses, group_name, result_save_path):
    try:
        results_df = pd.DataFrame({
            'Epoch': range(1, len(train_losses) + 1),
            'Train Loss': train_losses,
            'Validation Loss': val_losses
        })
        csv_path = os.path.join(result_save_path, f'training_results_{group_name[0]}_{group_name[1]}.csv')
        results_df.to_csv(csv_path, index=False)
        print(f'Training results saved to {csv_path}')
    except Exception as e:
        print(f'Error saving training results: {e}')

def save_model_checkpoint(model, group_name, model_save_path):
    try:  
        model_path = os.path.join(model_save_path, f'model_{group_name[0]}_{group_name[1]}.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model checkpoint saved to {model_path}')
    except Exception as e:
        print(f'Error saving model checkpoint: {e}')

def plot_predictions(y_test, predictions, group_name, result_save_path):
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(predictions, label='Predicted', linestyle='--')
        plt.xlabel('Sample')
        plt.ylabel('Time Gap (seconds)')
        plt.title(f'Actual vs Predicted for group {group_name}')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(result_save_path, f'prediction_plot_{group_name[0]}_{group_name[1]}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'Prediction plot saved to {plot_path}')
    except Exception as e:
        print(f'Error plotting predictions: {e}')

def evaluate_model(model, criterion, X_test, y_test, scaler, group_name, result_save_path):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)
        print(f'Test Loss for group {group_name}: {test_loss.item():.4f}')

        # 역정규화하여 실제 값과 예측 값 비교
        predictions = scaler.inverse_transform(predictions.cpu().numpy())
        y_test = scaler.inverse_transform(y_test.cpu().numpy())

        # 그래프로 실제 값과 예측 값 비교
        plot_predictions(y_test, predictions, group_name, result_save_path)

def main():
    # 데이터 로드 및 전처리
    train_df = load_and_preprocess_data(os.path.join(data_dir, 'data', 'infounit_7v2.csv'))
    val_test_df = load_and_preprocess_data(os.path.join(data_dir, 'data', 'infounit_8v2.csv'))

    # 그룹별로 비정상 데이터 처리
    train_df_cleaned = process_and_group_data(train_df)
    val_test_df_cleaned = process_and_group_data(val_test_df)

    # Optuna로 하이퍼파라미터 최적화
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_df_cleaned, val_test_df_cleaned), n_trials=50)

    print(f'Best trial: {study.best_trial.params}')

def objective(trial, train_df_cleaned, val_test_df_cleaned):
    # Optuna가 최적화할 하이퍼파라미터 정의
    hidden_size = trial.suggest_int('hidden_size', 16, 128)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    num_epochs = trial.suggest_int('num_epochs', 20, 100)
    
    # 하이퍼파라미터 설정
    input_size = 3
    output_size = 1
    patience = 10  # Early stopping patience 설정

    # CUDA 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 학습 데이터 그룹화
    grouped_train = train_df_cleaned.groupby(['BUSINFOUNIT_ID', 'DAY_TYPE'])

    total_val_loss = 0
    group_count = 0

    for name, train_group in grouped_train:
        val_test_group = val_test_df_cleaned[(val_test_df_cleaned['BUSINFOUNIT_ID'] == name[0]) & (val_test_df_cleaned['DAY_TYPE'] == name[1])]
        if len(train_group) < 2 or len(val_test_group) < 2:
            continue

        # 학습 데이터 피처와 타겟 분리
        X_train = train_group[['hh', 'mm', 'ss']].values
        y_train = train_group['TIME_GAP'].values

        # 검증 및 테스트 데이터 피처와 타겟 분리
        X_val_test = val_test_group[['hh', 'mm', 'ss']].values
        y_val_test = val_test_group['TIME_GAP'].values

        # 데이터 정규화
        scaler = MinMaxScaler()
        y_train = y_train.reshape(-1, 1)
        y_train = scaler.fit_transform(y_train).flatten()

        y_val_test = y_val_test.reshape(-1, 1)
        y_val_test = scaler.transform(y_val_test).flatten()

        # 검증 및 테스트 데이터 분리 (검증 50%, 테스트 50%)
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

        # 텐서로 변환
        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

        # 모델 초기화
        model = TimeRNN(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 모델 학습
        best_epoch, val_loss, train_losses, val_losses = train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs, patience)
        
        # 학습 결과 저장
        save_training_results(train_losses, val_losses, name, result_save_path)
        save_model_checkpoint(model, name, model_save_path)
        
        # 모델 평가 및 예측 값과 실제 값 그래프 저장
        evaluate_model(model, criterion, X_test, y_test, scaler, name, result_save_path)
        
        total_val_loss += val_loss
        group_count += 1
    
    return total_val_loss / group_count if group_count > 0 else float('inf')

if __name__ == "__main__":
    main()
