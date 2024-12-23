import argparse
import pandas as pd
import numpy as np
import json
from datetime import timedelta

# CSV 파일 읽기 (사용자의 파일 경로로 수정)
eth_prices = pd.read_csv('/home/ubuntu/ID/ethereum_daily_prices.csv', parse_dates=['Date']).drop(columns='ETH-USD')
withdrawal_data = pd.read_csv('/home/ubuntu/ID/withdrawal_ETH/withdrawal_frequency.csv', parse_dates=['Date'])
deposit_data = pd.read_csv('/home/ubuntu/ID/deposit_ETH/deposit_frequency.csv', parse_dates=['Date'])
daily_commits = pd.read_csv('/home/ubuntu/ID/github_dev_ETH/daily_commits.csv', parse_dates=['commit_date'])
netflow_eth = pd.read_csv('/home/ubuntu/ID/netflow_ETH/netflow_eth.csv')
search_freq = pd.read_csv('/home/ubuntu/ID/google_searching/search_daily.csv')

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

def convert_eth_prices_to_timestamp(df, start_date, end_date):
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    return {
        "timestamps": filtered_df['Date'].apply(lambda x: int(x.timestamp())).tolist(),
        "values": filtered_df['Close'].tolist()  # Close 값 기준
    }

# UNIX 타임스탬프 변환 함수 (lag 적용)
def convert_to_timestamp(df, value_col, lag_days=0):
    lag_seconds = lag_days * 86400  # lag를 초 단위로 변환
    return {
        "timestamps": df['Date'].apply(lambda x: int(x.timestamp()) - lag_seconds).tolist(),
        "values": df[value_col].tolist()
    }

def convert_tlcc_to_timestamp(correlations, graph_start_date):
    result = {
        "timestamps": [],
        "values": []
    }
    for lag, corr in correlations:
        aligned_date = graph_start_date + timedelta(days=lag)
        result["timestamps"].append(int(aligned_date.timestamp()))
        result["values"].append(corr)
    return result

def convert_correlations_to_json(correlations):
    result = {
        "lags": [lag for lag, _ in correlations],
        "values": [value for _, value in correlations]
    }
    return result

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
    graph_start_date = start_date - timedelta(hours=24) 

    # TLCC 결과 계산
    max_lag = 30  # 최대 지연 기간 설정 (예: 30일)
    optimal_lags = {}
    tlcc_graph_data = {}
    if "withdrawal_freq" in combination:
        filtered_withdrawal = merged_data_withdrawal_freq[  # 날짜 범위로 데이터 필터링
            (merged_data_withdrawal_freq['Date'] >= start_date) &
            (merged_data_withdrawal_freq['Date'] <= end_date)
        ]
        correlations = time_lagged_cross_correlation(
            filtered_withdrawal['Frequency'], filtered_withdrawal['Close'], max_lag=max_lag
        )
        optimal_lags["withdrawal_freq"] = max(correlations, key=lambda x: abs(x[1]))[0]
        tlcc_graph_data["withdrawal_freq"] = convert_correlations_to_json(correlations)

    if "deposit_freq" in combination:
        filtered_deposit = merged_data_deposit_freq[  # 날짜 범위로 데이터 필터링
            (merged_data_deposit_freq['Date'] >= start_date) &
            (merged_data_deposit_freq['Date'] <= end_date)
        ]
        correlations = time_lagged_cross_correlation(
            filtered_deposit['Frequency'], filtered_deposit['Close'], max_lag=max_lag
        )
        optimal_lags["deposit_freq"] = max(correlations, key=lambda x: abs(x[1]))[0]
        tlcc_graph_data["deposit_freq"] = convert_correlations_to_json(correlations)

    if "commits_freq" in combination:
        filtered_commits = merged_data_commits_freq[  # 날짜 범위로 데이터 필터링
            (merged_data_commits_freq['Date'] >= start_date) &
            (merged_data_commits_freq['Date'] <= end_date)
        ]
        correlations = time_lagged_cross_correlation(
            filtered_commits['Frequency'], filtered_commits['Close'], max_lag=max_lag
        )
        optimal_lags["commits_freq"] = max(correlations, key=lambda x: abs(x[1]))[0]
        tlcc_graph_data["commits_freq"] = convert_correlations_to_json(correlations)

    if "netflow" in combination:
        filtered_netflow = merged_data_netflow[  # 날짜 범위로 데이터 필터링
            (merged_data_netflow['Date'] >= start_date) &
            (merged_data_netflow['Date'] <= end_date)
        ]
        correlations = time_lagged_cross_correlation(
            filtered_netflow['value'], filtered_netflow['Close'], max_lag=max_lag
        )
        optimal_lags["netflow"] = max(correlations, key=lambda x: abs(x[1]))[0]
        tlcc_graph_data["netflow"] = convert_correlations_to_json(correlations)

    if "search_freq" in combination:
        filtered_search = merged_data_search_freq[  # 날짜 범위로 데이터 필터링
            (merged_data_search_freq['Date'] >= start_date) &
            (merged_data_search_freq['Date'] <= end_date)
        ]
        correlations = time_lagged_cross_correlation(
            filtered_search['Frequency'], filtered_search['Close'], max_lag=max_lag
        )
        optimal_lags["search_freq"] = max(correlations, key=lambda x: abs(x[1]))[0]
        tlcc_graph_data["search_freq"] = convert_correlations_to_json(correlations)

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
        filtered_data["withdrawal_freq"] = convert_to_timestamp(filtered_withdrawal, 'Frequency',optimal_lags["withdrawal_freq"])

    if "deposit_freq" in combination:
        filtered_deposit = merged_data_deposit_freq[
            (merged_data_deposit_freq['Date'] >= start_date) &
            (merged_data_deposit_freq['Date'] <= end_date)
            ][['Date', 'Frequency']].copy()
        filtered_data["deposit_freq"] = convert_to_timestamp(filtered_deposit, 'Frequency',optimal_lags["deposit_freq"])

    if "commits_freq" in combination:
        filtered_commits = merged_data_commits_freq[
            (merged_data_commits_freq['Date'] >= start_date) &
            (merged_data_commits_freq['Date'] <= end_date)
            ][['Date', 'Frequency']].copy()
        filtered_data["commits_freq"] = convert_to_timestamp(filtered_commits, 'Frequency',optimal_lags["commits_freq"])

    if "netflow" in combination:
        filtered_netflow = merged_data_netflow[
            (merged_data_netflow['Date'] >= start_date) &
            (merged_data_netflow['Date'] <= end_date)
            ][['Date', 'value']].copy()
        filtered_data["netflow"] = convert_to_timestamp(filtered_netflow, 'value',optimal_lags["netflow"])

    if "search_freq" in combination:
        filtered_search = merged_data_search_freq[
            (merged_data_search_freq['Date'] >= start_date) &
            (merged_data_search_freq['Date'] <= end_date)
            ][['Date', 'Frequency']].copy()
        filtered_data["search_freq"] = convert_to_timestamp(filtered_search, 'Frequency',optimal_lags["search_freq"])

    filtered_data["eth_prices"] = convert_eth_prices_to_timestamp(eth_prices, start_date, end_date)
    output = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "combination": combination,
        "optimal_lags": optimal_lags,
        "graph_data": filtered_data,
        "tlcc_graph_data" : tlcc_graph_data,
        "buy_sell_dates": buy_sell_dates
    }

    # JSON 출력
    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    main()
