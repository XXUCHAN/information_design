import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기
eth_prices = pd.read_csv('../ethereum_daily_prices.csv', parse_dates=['Date'])
daily_commits = pd.read_csv('daily_commits.csv', parse_dates=['commit_date'])

# 'commit_date'를 'Date'로 이름 변경하여 병합 준비
daily_commits.rename(columns={'commit_date': 'Date'}, inplace=True)

# 날짜별 데이터 병합 (inner join을 사용하여 두 파일에 모두 존재하는 날짜만 포함)
merged_data = pd.merge(eth_prices, daily_commits, on='Date', how='inner')

# 특정 기간 필터링
start_date = '2021-08-04' # 사용자 입력
end_date = '2024-08-04'   # 사용자 입력
merged_data_input = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)].copy()

# TLCC 계산 함수
def time_lagged_cross_correlation(series1, series2, max_lag):
    correlations = []
    for lag in range(0, max_lag + 1):  # lag를 0 이상만 계산
        corr = series1[lag:].corr(series2[:-lag]) if lag > 0 else series1.corr(series2)
        correlations.append((lag, corr))  # (lag, correlation) 형태로 저장
    return correlations

# TLCC 결과 계산
correlations = time_lagged_cross_correlation(merged_data_input['count'], merged_data_input['Close'], max_lag=30)

# lag가 0 이상인 값들 중에서 최대 상관계수의 lag 찾기
optimal_lag = max(correlations, key=lambda x: abs(x[1]))[0]
print(f"Optimal lag (excluding negative lags): {optimal_lag}")

# TLCC 결과를 시각화
lags, corr_values = zip(*correlations)

plt.figure(figsize=(10, 6))
plt.plot(lags, corr_values, marker='o', linestyle='-')
plt.title('Time-Lagged Cross-Correlation (TLCC)')
plt.xlabel('Lag (days)')
plt.ylabel('Correlation')
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(optimal_lag, color='red', linestyle='--', label=f'Optimal Lag = {optimal_lag}')
plt.legend()
plt.savefig('TLCC_visualization.png')  # 그래프를 파일로 저장
plt.show()

print("TLCC 그래프가 'TLCC_visualization.png'로 저장되었습니다.")
