import pandas as pd
import numpy as np
from tqdm import tqdm

def load_dataset(file_path):
    return pd.read_csv(file_path)

def save_dataset(df, file_path):
    df.to_csv(file_path, index=False)

def identify_outliers(df, columns):
    outliers = pd.DataFrame()
    non_outliers = df.copy()

    for column in tqdm(columns, desc="Processing Columns"):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        col_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outliers = pd.concat([outliers, col_outliers])

        non_outliers = non_outliers[(non_outliers[column] >= lower_bound) & (non_outliers[column] <= upper_bound)]

    # 중복된 이상치 제거
    outliers = outliers.drop_duplicates()

    return non_outliers, outliers

def print_outlier_stats(df, outliers):
    total = len(df)
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / total) * 100

    print(f"\nTotal rows in dataset: {total}")
    print(f"Outliers found: {outlier_count} ({outlier_percentage:.2f}%)")

    return outlier_count

def remove_outliers_and_save(df, columns, original_file_path):
    # Identify outliers
    print("\nIdentifying outliers...")
    non_outliers, outliers = identify_outliers(df, columns)

    # Save outliers to a separate file
    outliers_file_path = original_file_path.replace(".csv", "_outliers.csv")
    save_dataset(outliers, outliers_file_path)
    print(f"Outliers saved to: {outliers_file_path}")

    # Print outlier stats
    outlier_count = print_outlier_stats(df, outliers)

    # Save the dataset without outliers
    no_outliers_file_path = original_file_path.replace(".csv", "_noOutliers.csv")
    save_dataset(non_outliers, no_outliers_file_path)
    print(f"Dataset without outliers saved to: {no_outliers_file_path}")

    return outlier_count

if __name__ == "__main__":
    # CSV 파일 경로 설정
    # file_path = '../dataset/travel_time_07280400_08110359_train.csv'
    # file_path = '../dataset/travel_time_07280400_08110359_val.csv'
    # file_path = '../dataset/travel_time_07280400_08110359_test.csv'

    # file_path = '../dataset/holding_time_07280400_08110359_train.csv'
    # file_path = '../dataset/holding_time_07280400_08110359_val.csv'
    file_path = '../dataset/holding_time_07280400_08110359_test.csv'

    # CSV 파일 로드
    print("Loading dataset...")
    df = load_dataset(file_path)

    # 이상치를 체크할 주요 컬럼 설정
    columns_to_check = ['TIME_GAP']  # 주요 숫자형 컬럼들을 확인
    # columns_to_check = ['TIME_GAP', 'LEN']  # 주요 숫자형 컬럼들을 확인

    # 이상치 제거 및 저장
    outlier_count = remove_outliers_and_save(df, columns_to_check, file_path)

    print(f"\n{outlier_count} outliers were removed from the dataset.")
