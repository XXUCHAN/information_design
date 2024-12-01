import pandas as pd
import numpy as np

# CSV 파일 읽기 (사용자의 파일 경로로 수정)
eth_prices = pd.read_csv('./ethereum_daily_prices.csv', parse_dates=['Date']).drop(columns='ETH-USD')
withdrawal_data = pd.read_csv('withdrawal_ETH/withdrawal_frequency.csv', parse_dates=['Date'])
deposit_data = pd.read_csv('deposit_ETH/deposit_frequency.csv', parse_dates=['Date'])
daily_commits = pd.read_csv('github_dev_ETH/daily_commits.csv',parse_dates=['commit_date'])
netflow_eth = pd.read_csv('netflow_ETH/netflow_eth.csv')
search_freq = pd.read_csv('google_searching/search_daily.csv')

#rename
netflow_eth.rename(columns={'timeStamp': 'Date'}, inplace=True)
daily_commits.rename(columns={'commit_date': 'Date', 'count': 'Frequency'}, inplace=True)

# 두 데이터프레임의 Date 열을 동일한 형식으로 변환 (UTC 제거)
for df in [eth_prices,deposit_data, withdrawal_data, daily_commits, search_freq, netflow_eth]:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)


# 날짜별 데이터 병합 (이더리움 가격 데이터를 기준으로 left join 사용)
merged_data_withdrawal_freq = pd.merge(eth_prices, withdrawal_data, on='Date', how='left')
merged_data_deposit_freq = pd.merge(eth_prices, deposit_data, on='Date', how='left')
merged_data_commits_freq = pd.merge(eth_prices,daily_commits, on='Date',how='left')
merged_data_netflow = pd.merge(eth_prices,netflow_eth,on='Date',how='left')
merged_data_search_freq = pd.merge(eth_prices,search_freq, on='Date',how='left')

# 데이터에 없는 날짜는 0으로 채움
merged_data_withdrawal_freq['Frequency'] = merged_data_withdrawal_freq['Frequency'].fillna(0)
merged_data_deposit_freq['Frequency'] = merged_data_deposit_freq['Frequency'].fillna(0)
merged_data_commits_freq['Frequency'] = merged_data_commits_freq['Frequency'].fillna(0)

# 사용자로부터 시작일과 종료일 입력받기
start_date = '2022-11-08'  # 예시 시작일 (사용자가 변경할 수 있음)
end_date = '2024-11-08'    # 예시 종료일 (사용자가 변경할 수 있음)

# 시작일과 종료일을 기준으로 데이터 필터링
for name, df in [
    ("merged_data_withdrawal_freq", merged_data_withdrawal_freq),
    ("merged_data_deposit_freq", merged_data_deposit_freq),
    ("merged_data_commits_freq", merged_data_commits_freq),
    ("merged_data_netflow", merged_data_netflow),
    ("merged_data_search_freq", merged_data_search_freq)
]:
    locals()[name] = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()



# TLCC 계산 함수
def time_lagged_cross_correlation(series1, series2, max_lag=30):
    correlations = []
    for lag in range(1, max_lag + 1):
        shifted_series1 = series1.shift(lag)
        valid_lag_idx = shifted_series1.notna() & series2.notna()
        if valid_lag_idx.sum() > 10:  # 최소 유효한 데이터 수 확인
            try:
                corr = shifted_series1[valid_lag_idx].corr(series2[valid_lag_idx])
                correlations.append((lag, corr))  # (lag, 상관계수) 저장
            except Exception as e:
                correlations.append((lag, np.nan))  # 에러 발생 시 NaN 저장
        else:
            correlations.append((lag, np.nan))
    return correlations

# TLCC 결과 계산
max_lag = 30  # 최대 지연 기간 설정 (예: 30일)
correlations_withdrawal_freq = time_lagged_cross_correlation(
    merged_data_withdrawal_freq['Frequency'], merged_data_withdrawal_freq['Close'], max_lag=max_lag
)
correlations_deposit_freq = time_lagged_cross_correlation(
    merged_data_deposit_freq['Frequency'], merged_data_deposit_freq['Close'], max_lag=max_lag
)
correlations_commits_freq = time_lagged_cross_correlation(
    merged_data_commits_freq['Frequency'],merged_data_commits_freq['Close'], max_lag=max_lag
)
correlations_netflow = time_lagged_cross_correlation(
    merged_data_netflow['value'],merged_data_netflow['Close'],max_lag=max_lag
)
correlations_search_freq = time_lagged_cross_correlation(
    merged_data_search_freq['Frequency'],merged_data_search_freq['Close'],max_lag=max_lag
)


# lag가 0 이상인 값들 중에서 최대 상관계수의 lag 찾기
optimal_lag_withdrawal_freq = max(correlations_withdrawal_freq, key=lambda x: abs(x[1]))[0]
optimal_lag_deposit_freq = max(correlations_deposit_freq, key=lambda x: abs(x[1]))[0]
optimal_lag_commits_freq = max(correlations_commits_freq, key=lambda x: abs(x[1]))[0]
optimal_lag_netflow = max(correlations_netflow,key=lambda x: abs(x[1]))[0]
optimal_lag_search_freq = max(correlations_search_freq,key=lambda x: abs(x[1]))[0]

print(f"Optimal lag for withdrawal frequency: {optimal_lag_withdrawal_freq}")
print(f"Optimal lag for deposit frequency: {optimal_lag_deposit_freq}")
print(f"Optimal lag for commits frequency: {optimal_lag_commits_freq}")
print(f"Optimal lag for netflow : {optimal_lag_netflow}")
print(f"Optimal lag for search frequency: {optimal_lag_search_freq}")

# 결과 출력
print("Merged Data (Withdrawal Frequency):")
print(merged_data_withdrawal_freq.head())
print("\nMerged Data (Deposit Frequency):")
print(merged_data_deposit_freq.head())
print("\nMerged Data (Commits Frequency):")
print(merged_data_commits_freq.head())
print("\nMerged Data (netflow value):")
print(merged_data_netflow.head())
print("\nMerged Data (search Frequency):")
print(merged_data_search_freq.head())


