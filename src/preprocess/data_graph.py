import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import multiprocessing as mp
import matplotlib.dates as mdates

def plot_single_route_stop(row, data, output_dir):
    busroute_id = row['BUSROUTE_ID']
    busstop_id = row['BUSSTOP_ID']

    subset = data[(data['BUSROUTE_ID'] == busroute_id) & (data['BUSSTOP_ID'] == busstop_id)]

    if subset.empty:
        return

    # DEP_TIME을 datetime.time 형식으로 유지
    subset['DEP_TIME'] = pd.to_datetime(subset['DEP_TIME'], format='%H:%M:%S', errors='coerce').dt.strftime('%H:%M')

    # datetime 형식으로 다시 변환 (초 정보가 없는 상태로 변환)
    subset['DEP_TIME'] = pd.to_datetime(subset['DEP_TIME'], format='%H:%M').dt.time

    # 05:00 ~ 22:00 사이의 데이터만 사용
    subset = subset[subset['DEP_TIME'].apply(lambda x: 5 <= x.hour <= 24)]

    # 시간대를 15분 간격으로 그룹화
    def time_to_bin(time_obj):
        total_minutes = time_obj.hour * 60 + time_obj.minute
        return total_minutes // 10 * 10  # 10분 단위로 나눔

    subset['TIME_BIN'] = subset['DEP_TIME'].apply(time_to_bin)

    # 각 시간대의 평균 TIME_GAP 계산
    avg_time_gap = subset.groupby('TIME_BIN')['TIME_GAP'].mean().reset_index()
    avg_time_gap['TIME_BIN_TIME'] = avg_time_gap['TIME_BIN'].apply(lambda x: datetime.combine(datetime.today(), datetime.min.time()) + timedelta(minutes=x))

    # 이동 평균 계산 (window=3, 필요에 따라 조정)
    avg_time_gap['TIME_GAP_SMA'] = avg_time_gap['TIME_GAP'].rolling(window=1, center=True).mean()

    # 그래프 생성
    plt.figure(figsize=(12, 6))
    x_values = avg_time_gap['TIME_BIN_TIME']

    # 실제 데이터 점
    plt.scatter(x_values, avg_time_gap['TIME_GAP'], alpha=0.5, label='Average Dwell Time')

    # 추세선 (이동 평균)
    plt.plot(x_values, avg_time_gap['TIME_GAP_SMA'], color='red', linewidth=2, label='Trend (Moving Average)')

    # x축 설정
    plt.xlabel('Departure Time')
    plt.ylabel('Average Dwell Time (seconds)')
    plt.title(f'Route {busroute_id} - Stop {busstop_id} Dwell Time by Departure Time')
    plt.xlim([datetime.combine(datetime.today(), datetime.min.time()), datetime.combine(datetime.today(), datetime.min.time()) + timedelta(hours=24)])
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 1시간 간격으로 눈금 설정

    # 그리드 및 범례 추가
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 그래프 저장
    plt.tight_layout()
    filename = f'Route_{busroute_id}_Stop_{busstop_id}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f'Saved plot for Route {busroute_id} - Stop {busstop_id} at {filepath}')

def plot_time_gap_parallel(csv_path, output_dir='route_stop_plots', n_jobs=4):
    # 데이터 로드
    data = pd.read_csv(csv_path)
    data['DEP_TIME'] = pd.to_datetime(data['DEP_TIME'], format='%H:%M').dt.time
    unique_route_stop = data[['BUSROUTE_ID', 'BUSSTOP_ID']].drop_duplicates()
    os.makedirs(output_dir, exist_ok=True)

    # 병렬 처리
    pool = mp.Pool(processes=n_jobs)
    results = [pool.apply_async(plot_single_route_stop, args=(row, data, output_dir))
               for index, row in unique_route_stop.iterrows()]
    pool.close()
    pool.join()

    print("All plots have been saved.")

# 함수 호출 예시
if __name__ == "__main__":
    csv = '../../dataset/train/holding/240920/금요일/정차시간_4_5_금요일학습데이터_filtered.csv'
    output_directory = os.path.dirname(csv)+'/route_stop_plots'
    number_of_parallel_jobs = 4  # 시스템의 CPU 코어 수에 맞게 조정

    plot_time_gap_parallel(csv, output_directory, number_of_parallel_jobs)
