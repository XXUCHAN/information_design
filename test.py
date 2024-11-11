import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.animation import FuncAnimation, PillowWriter
import json

# 사용자 데이터 불러오기
csv_file = 'google_searching/multiTimeline.csv'
search_data = pd.read_csv(csv_file, header=None, names=['date', 'search_volume'])
search_data['date'] = pd.to_datetime(search_data['date'])
search_data.set_index('date', inplace=True)

json_file = 'ethereum_weekly_prices.json'
with open(json_file, 'r') as file:
    price_data = json.load(file)
price_data = pd.DataFrame(price_data)
price_data['date'] = pd.to_datetime(price_data['date'])
price_data.set_index('date', inplace=True)

data = pd.merge(search_data, price_data, left_index=True, right_index=True, how='inner')
x_raw = data['search_volume'].values
y_raw = data['price'].values

# 데이터 정규화 (0과 1 사이로 변환)
x = (x_raw - np.min(x_raw)) / (np.max(x_raw) - np.min(x_raw))
y = (y_raw - np.min(y_raw)) / (np.max(y_raw) - np.min(y_raw))

# TLCC 계산 함수
def lagged_correlation(x, y, max_lag):
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    for lag in lags:
        if lag < 0:
            corr = pearsonr(x[:lag], y[-lag:])[0]
        elif lag > 0:
            corr = pearsonr(x[lag:], y[:-lag])[0]
        else:
            corr = pearsonr(x, y)[0]
        correlations.append(corr)
    return lags, correlations

# 최대 시차 설정
max_lag = 30
lags, tlcc_correlations = lagged_correlation(x, y, max_lag)

# 애니메이션 설정
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 첫 번째 플롯: 검색량과 가격 신호 (정규화된 데이터 사용)
ax1.plot(data.index, y, label="Normalized Price", color="red")  # 고정된 가격 그래프
search_line, = ax1.plot(data.index, x, label="Normalized Search Volume", color="blue")  # 초기 전체 검색량 그래프
ax1.set_xlim(data.index.min(), data.index.max())
ax1.set_ylim(0, 1)
ax1.set_xlabel("Date")
ax1.set_ylabel("Normalized Value")
ax1.legend()

# 두 번째 플롯: 시차에 따른 상관관계
correlation_line, = ax2.plot([], [], color="green", marker='o')
ax2.set_xlim(-max_lag, max_lag)
ax2.set_ylim(min(tlcc_correlations) - 0.1, max(tlcc_correlations) + 0.1)
ax2.set_xlabel("Lag")
ax2.set_ylabel("Correlation")
ax2.axvline(0, color="black", linestyle="--", linewidth=0.5)
ax2.set_title("Time-Lagged Cross-Correlation")

# 애니메이션 초기화 함수
def init():
    correlation_line.set_data([], [])
    return search_line, correlation_line

# 애니메이션 업데이트 함수
def update(frame):
    # 검색량 그래프를 이동시키기 위해 일정 구간만 보여주도록 슬라이싱
    # 처음에 전체 그래프가 표시된 후 이동을 시작
    window_size = len(x) // len(lags)  # 검색량 데이터 윈도우 크기를 시차 데이터 길이에 맞게 설정
    start = min(frame * window_size, len(x))  # frame에 따라 시작 위치를 이동시킴
    search_line.set_data(data.index[:start], x[:start])  # 이동하는 검색량 그래프 설정

    # TLCC 상관관계 그래프 업데이트
    current_lags = lags[:frame]
    current_correlations = tlcc_correlations[:frame]
    correlation_line.set_data(current_lags, current_correlations)

    return search_line, correlation_line

# 애니메이션 생성 (반복 및 속도 조정)
ani = FuncAnimation(
    fig, update, frames=len(lags), init_func=init, blit=True, repeat=True, interval=200
)

# 애니메이션을 GIF로 저장
ani.save("tlcc_animation.gif", writer=PillowWriter(fps=10))
plt.close()