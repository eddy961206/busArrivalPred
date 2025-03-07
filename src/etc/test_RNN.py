import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os

# Define the absolute path to the data directory
data_dir = 'D:/Project/VSCode/PredictBusArrivalTime'
model_save_path = 'PredictBusArrivalTime/models'
result_save_path = 'PredictBusArrivalTime/results'
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(result_save_path, exist_ok=True)

# 1. CSV 파일 읽어오기
# df = pd.read_csv(os.path.join(data_dir, 'data/travel_time_test.csv'), low_memory=False)
df = pd.read_csv(os.path.join(data_dir, 'data/infounit_8v2.csv'), low_memory=False)
df = df.sort_values(by=['DEP_TIME']).reset_index(drop=True)

# 2. DEP_TIME을 이용한 명시적 시간 전처리
def split_time(dep_time):
    t = datetime.strptime(dep_time, '%H:%M:%S')
    return t.hour, t.minute, t.second
    # t = datetime.strptime(dep_time, '%H:%M')
    # return t.hour, t.minute

# DEP_TIME을 시(hour), 분(minute), 초(second)로 분리하여 각각의 열로 추가
df['hh'], df['mm'], df['ss'] = zip(*df['DEP_TIME'].apply(split_time))
# df['hh'], df['mm'] = zip(*df['DEP_TIME'].apply(split_time))

# 3. RNN 모델 정의
class TimeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TimeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # 초기 hidden state
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # 최종 시퀀스의 출력만 사용
        return out

# 하이퍼파라미터 설정
input_size = 3  # hh, mm, ss 각각의 피처로 사용 (3개의 입력)
hidden_size = 50  # hidden state 크기
output_size = 1  # 예측할 값은 TIME_GAP (초 단위)

# CUDA 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 모델 로드 및 예측
def predict_and_save(group_name, model_path):
    # print(f"Predicting for group: {group_name}")

    # 데이터 필터링
    group = df[(df['BUSINFOUNIT_ID'] == group_name[0]) & (df['DAY_TYPE'] == group_name[1])].copy()

    # 피처와 타겟 데이터 분리
    X = group[['hh', 'mm', 'ss']].values  # 분리된 hh, mm, ss를 입력으로 사용
    # X = group[['hh', 'mm']].values  # 분리된 hh, mm를 입력으로 사용
    y = group['TIME_GAP'].values  # 예측하려는 값은 TIME_GAP (초 단위)

    # 데이터 정규화
    scaler = MinMaxScaler()
    y = y.reshape(-1, 1)  # MinMaxScaler는 2D 배열을 요구합니다
    y = scaler.fit_transform(y).flatten()  # 정규화된 y값

    # 데이터를 테스트 세트로 변환
    X_test = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)  # [batch_size, seq_len, input_size] 형태로 변환

    # RNN 모델 초기화 및 장치로 이동
    model = TimeRNN(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # 저장된 모델 로드
    model.eval()

    # 예측
    with torch.no_grad():
        predictions = model(X_test)
        predictions = predictions.cpu().numpy()  # 예측값을 NumPy 배열로 변환
        predictions = np.maximum(predictions, 0)  # 음수 값을 0으로 클리핑

    # 예측값 복원 (역정규화)
    predictions = scaler.inverse_transform(predictions)

    # 예측 결과를 데이터프레임에 추가
    group['PRED_GAP'] = predictions  # 예측값을 새로운 컬럼으로 추가
    return group

# 테스트 실행 및 결과 저장
for name, _ in df.groupby(['BUSINFOUNIT_ID', 'DAY_TYPE']):
    model_path = os.path.join(model_save_path, f'model_{name[0]}_{name[1]}.pth')
    if os.path.exists(model_path):
        group_result = predict_and_save(name, model_path)
        
        # CSV 파일 저장 경로 정의 (Pred_{businfounit_id}_{day_type}.csv 형식)
        output_file = f'Pred_{name[0]}_{name[1]}.csv'
        output_path = os.path.join(result_save_path, output_file)
        
        # 결과를 각 그룹별로 개별 CSV 파일로 저장
        group_result.to_csv(output_path, index=False)
        print(f'Results for group {name} saved to {output_file}')
    else:
        print(f'Model for group {name} not found.')
