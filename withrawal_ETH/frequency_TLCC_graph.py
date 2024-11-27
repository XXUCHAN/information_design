import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기 (사용자의 파일 경로로 수정)
eth_prices = pd.read_csv('../ethereum_daily_prices.csv', parse_dates=['Date'])
withdrawal_data = pd.read_csv('withdrawal_frequency.csv', parse_dates=['Date'])

# 두 데이터프레임의 Date 열을 동일한 형식으로 변환 (UTC 제거)
eth_prices['Date'] = eth_prices['Date'].dt.tz_localize(None)
withdrawal_data['Date'] = withdrawal_data['Date'].dt.tz_localize(None)

# 날짜별 데이터 병합 (이더리움 가격 데이터를 기준으로 left join 사용)
merged_data = pd.merge(eth_prices, withdrawal_data, on='Date', how='left')

# withdrawal 데이터에 없는 날짜는 0으로 채움
merged_data['Frequency'] = merged_data['Frequency'].fillna(0)

# 사용자로부터 시작일과 종료일 입력받기
start_date = '2022-11-08'  # 예시 시작일 (사용자가 변경할 수 있음)
end_date = '2024-11-08'    # 예시 종료일 (사용자가 변경할 수 있음)

# 시작일과 종료일을 기준으로 데이터 필터링
merged_data = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)].copy()

# TLCC 계산 함수
def time_lagged_cross_correlation(series1, series2, max_lag):
    correlations = []
    for lag in range(0, max_lag + 1):  # lag를 0 이상만 계산
        corr = series1[lag:].corr(series2[:-lag]) if lag > 0 else series1.corr(series2)
        correlations.append((lag, corr))  # (lag, correlation) 형태로 저장
    return correlations

# TLCC 결과 계산
max_lag = 30  # 최대 지연 기간 설정 (예: 30일)
correlations = time_lagged_cross_correlation(merged_data['Frequency'], merged_data['Close'], max_lag=max_lag)

# lag가 0 이상인 값들 중에서 최대 상관계수의 lag 찾기
optimal_lag = max(correlations, key=lambda x: abs(x[1]))[0]
print(f"Optimal lag: {optimal_lag}")

# TLCC 결과를 시각화
lags, corr_values = zip(*correlations)

plt.figure(figsize=(10, 6))
plt.plot(lags, corr_values, marker='o', linestyle='-')
plt.title('Time-Lagged Cross-Correlation (TLCC) between Withdrawal Frequency and ETH Price')
plt.xlabel('Lag (days)')
plt.ylabel('Correlation')
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(optimal_lag, color='red', linestyle='--', label=f'Optimal Lag = {optimal_lag}')
plt.legend()
plt.show()

print("TLCC 그래프가 'TLCC_withdrawal_ETH_price.png'로 저장되었습니다.")
