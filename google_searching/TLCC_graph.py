import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기
eth_prices = pd.read_csv('../ethereum_weekly_prices.csv', parse_dates=['Date'])
google_trends = pd.read_csv('multiTimeline.csv', parse_dates=['Date'])  # 구글 검색량 데이터

# 두 데이터프레임의 Date 열을 동일한 형식으로 변환 (UTC 제거)
eth_prices['Date'] = eth_prices['Date'].dt.tz_localize(None)
google_trends['Date'] = google_trends['Date'].dt.tz_localize(None)

# 두 데이터 병합 (Date 기준 inner join 사용)
merged_data = pd.merge(eth_prices, google_trends, on='Date', how='inner')

# TLCC 계산 함수
def time_lagged_cross_correlation(series1, series2, max_lag):
    correlations = []
    for lag in range(1, max_lag + 1):  # lag를 0 이상만 계산
        corr = series1[lag:].corr(series2[:-lag]) if lag > 0 else series1.corr(series2)
        correlations.append((lag, corr))  # (lag, correlation) 형태로 저장
    return correlations

# TLCC 결과 계산 (최대 10주 지연으로 설정)
max_lag = 100
correlations = time_lagged_cross_correlation(merged_data['Close'], merged_data['Frequency'], max_lag=max_lag)

# lag 값과 상관계수를 분리하여 그래프 시각화
lags, corr_values = zip(*correlations)

# 그래프 생성
plt.figure(figsize=(10, 5))
plt.plot(lags, corr_values, marker='o')
plt.title('Time-Lagged Cross-Correlation (TLCC)')
plt.xlabel('Lag (weeks)')
plt.ylabel('Correlation')
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--')
plt.show()
