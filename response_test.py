import argparse
import pandas as pd
import numpy as np
import json

# CSV 파일 읽기 (사용자의 파일 경로로 수정)
eth_prices = pd.read_csv('./ethereum_daily_prices.csv', parse_dates=['Date']).drop(columns='ETH-USD')
withdrawal_data = pd.read_csv('withdrawal_ETH/withdrawal_frequency.csv', parse_dates=['Date'])
deposit_data = pd.read_csv('deposit_ETH/deposit_frequency.csv', parse_dates=['Date'])
daily_commits = pd.read_csv('github_dev_ETH/daily_commits.csv', parse_dates=['commit_date'])
netflow_eth = pd.read_csv('netflow_ETH/netflow_eth.csv')
search_freq = pd.read_csv('google_searching/search_daily.csv')

# Rename
netflow_eth.rename(columns={'timeStamp': 'Date'}, inplace=True)
daily_commits.rename(columns={'commit_date': 'Date', 'count': 'Frequency'}, inplace=True)

# 두 데이터프레임의 Date 열을 동일한 형식으로 변환 (UTC 제거)
for df in [eth_prices, deposit_data, withdrawal_data, daily_commits, search_freq, netflow_eth]:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)

# 날짜별 데이터 병합 (이더리움 가격 데이터를 기준으로 left join 사용)
merged_data_withdrawal_freq = pd.merge(eth_prices, withdrawal_data, on='Date', how='left')
merged_data_deposit_freq = pd.merge(eth_prices, deposit_data, on='Date', how='left')
merged_data_commits_freq = pd.merge(eth_prices, daily_commits, on='Date', how='left')
merged_data_netflow = pd.merge(eth_prices, netflow_eth, on='Date', how='left')
merged_data_search_freq = pd.merge(eth_prices, search_freq, on='Date', how='left')

# 데이터에 없는 날짜는 0으로 채움
merged_data_withdrawal_freq['Frequency'] = merged_data_withdrawal_freq['Frequency'].fillna(0)
merged_data_deposit_freq['Frequency'] = merged_data_deposit_freq['Frequency'].fillna(0)
merged_data_commits_freq['Frequency'] = merged_data_commits_freq['Frequency'].fillna(0)

# UNIX 타임스탬프 변환 함수
def convert_to_timestamp(df, value_col):
    return {
        "timestamps": df['Date'].apply(lambda x: int(x.timestamp())).tolist(),
        "values": df[value_col].tolist()
    }

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

# 메인 함수
def main():
    # 인자 처리
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--combination", required=True, help="Comma-separated list of indicators")
    parser.add_argument("--buy_signal", required=True, help="Comma-separated buy signal dates")
    parser.add_argument("--sell_signal", required=True, help="Comma-separated sell signal dates")
    args = parser.parse_args()

    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    combination = args.combination.split(",")
    buy_signal = args.buy_signal.split(",")
    sell_signal = args.sell_signal.split(",")

    # TLCC 결과 계산
    max_lag = 30  # 최대 지연 기간 설정 (예: 30일)
    optimal_lags = {}

    if "withdrawal_freq" in combination:
        correlations = time_lagged_cross_correlation(
            merged_data_withdrawal_freq['Frequency'], merged_data_withdrawal_freq['Close'], max_lag=max_lag
        )
        optimal_lags["withdrawal_freq"] = max(correlations, key=lambda x: abs(x[1]))[0]

    if "deposit_freq" in combination:
        correlations = time_lagged_cross_correlation(
            merged_data_deposit_freq['Frequency'], merged_data_deposit_freq['Close'], max_lag=max_lag
        )
        optimal_lags["deposit_freq"] = max(correlations, key=lambda x: abs(x[1]))[0]

    if "commits_freq" in combination:
        correlations = time_lagged_cross_correlation(
            merged_data_commits_freq['Frequency'], merged_data_commits_freq['Close'], max_lag=max_lag
        )
        optimal_lags["commits_freq"] = max(correlations, key=lambda x: abs(x[1]))[0]

    if "netflow" in combination:
        correlations = time_lagged_cross_correlation(
            merged_data_netflow['value'], merged_data_netflow['Close'], max_lag=max_lag
        )
        optimal_lags["netflow"] = max(correlations, key=lambda x: abs(x[1]))[0]

    if "search_freq" in combination:
        correlations = time_lagged_cross_correlation(
            merged_data_search_freq['Frequency'], merged_data_search_freq['Close'], max_lag=max_lag
        )
        optimal_lags["search_freq"] = max(correlations, key=lambda x: abs(x[1]))[0]

    # Buy/Sell dates
    buy_sell_dates = []
    for date in buy_signal:
        buy_sell_dates.append({"date": date, "action": "BUY"})
    for date in sell_signal:
        buy_sell_dates.append({"date": date, "action": "SELL"})

    # 선택된 지표 데이터 필터링
    filtered_data = {}
    if "withdrawal_freq" in combination:
        filtered_withdrawal = merged_data_withdrawal_freq[
            (merged_data_withdrawal_freq['Date'] >= start_date) &
            (merged_data_withdrawal_freq['Date'] <= end_date)
            ][['Date', 'Frequency']].copy()
        filtered_data["withdrawal_freq"] = convert_to_timestamp(filtered_withdrawal, 'Frequency')

    if "deposit_freq" in combination:
        filtered_deposit = merged_data_deposit_freq[
            (merged_data_deposit_freq['Date'] >= start_date) &
            (merged_data_deposit_freq['Date'] <= end_date)
            ][['Date', 'Frequency']].copy()
        filtered_data["deposit_freq"] = convert_to_timestamp(filtered_deposit, 'Frequency')

    if "commits_freq" in combination:
        filtered_commits = merged_data_commits_freq[
            (merged_data_commits_freq['Date'] >= start_date) &
            (merged_data_commits_freq['Date'] <= end_date)
            ][['Date', 'Frequency']].copy()
        filtered_data["commits_freq"] = convert_to_timestamp(filtered_commits, 'Frequency')

    if "netflow" in combination:
        filtered_netflow = merged_data_netflow[
            (merged_data_netflow['Date'] >= start_date) &
            (merged_data_netflow['Date'] <= end_date)
            ][['Date', 'value']].copy()
        filtered_data["netflow"] = convert_to_timestamp(filtered_netflow, 'value')

    if "search_freq" in combination:
        filtered_search = merged_data_search_freq[
            (merged_data_search_freq['Date'] >= start_date) &
            (merged_data_search_freq['Date'] <= end_date)
            ][['Date', 'Frequency']].copy()
        filtered_data["search_freq"] = convert_to_timestamp(filtered_search, 'Frequency')

    output = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "combination": combination,
        "optimal_lags": optimal_lags,
        "graph_data": filtered_data,
        "buy_sell_dates": buy_sell_dates
    }

    # JSON 출력
    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    main()
