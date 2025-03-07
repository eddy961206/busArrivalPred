import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from travel_time_optuna import TravelTimeDataset, TravelTimeModel, load_data, evaluate_model
import os
import time

# 연속 학습 함수 정의
def continue_training(model, train_loader, val_loader, criterion, optimizer, scheduler, device, start_epoch, num_epochs=10, model_path=None):
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_train_loss = 0

        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}", unit="batch")):
            day_type, businfo_unit_id, len_feature, dep_hour_min, time_gap = [d.to(device) for d in data]
            optimizer.zero_grad()
            outputs = model(day_type, businfo_unit_id, len_feature, dep_hour_min)
            loss = criterion(outputs.view(-1), time_gap.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{start_epoch + num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # 스케줄러를 통해 학습률 조정
        scheduler.step(val_loss)

        if early_stopping(val_loss):
            print("Early stopping triggered")
            break

        # 매 epoch 후 모델 저장
        if model_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, model_path)
            print(f"Model saved to {model_path} at epoch {epoch}")

# 연속 학습 실행
if __name__ == "__main__":
    model_path = 'path_to_saved_model.pth'
    start_time = time.strftime("%Y%m%d-%H%M%S")
    results_folder = 'results_continuous_training'
    os.makedirs(results_folder, exist_ok=True)

    # GPU 사용 여부 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드
    train_loader, val_loader, test_loader = load_data(
        'dataset/travel_time_07280400_08110359_train_noOutliers.csv',
        'dataset/travel_time_07280400_08110359_val_noOutliers.csv',
        'dataset/travel_time_07280400_08110359_test_noOutliers.csv',
        batch_size=64,
        hash_size=1000
    )

    # 모델 설정
    day_type_size = len(train_loader.dataset.day_type[0])
    input_size = day_type_size + 2  # day_type + len_feature + dep_hour_min
    model = TravelTimeModel(input_size, hash_size=1000, unit_embedding_dim=32).to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # 이전 학습 기록이 있다면 불러오기
    start_epoch = 0
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    # 연속 학습
    continue_training(model, train_loader, val_loader, criterion, optimizer, scheduler, device, start_epoch, num_epochs=10, model_path=model_path)
