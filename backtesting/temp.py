from datetime import datetime
import pandas as pd
import numpy as np
from itertools import product, combinations
from multiprocessing import Pool, Manager, cpu_count
import time  # Execution time measurement
import json,os
import argparse
# User-defined date range
start_date = pd.to_datetime("2020-11-08")
end_date = pd.to_datetime("2021-11-08")
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--combination", required=True, help="Comma-separated list of indicators")
    parser.add_argument("--asset", required=True, type=float, help="Initial asset value")
    return parser.parse_args()
def load_and_prepare_data(start_date, end_date):
    BASE_DIR = "/Users/isuchan/IdeaProjects/python_test/ID"  # 데이터 파일이 저장된 기본 디렉토리

    eth_prices = pd.read_csv(os.path.join(BASE_DIR, 'ethereum_daily_prices.csv'), parse_dates=['Date'])
    deposit_freq = pd.read_csv(os.path.join(BASE_DIR, 'deposit_ETH/deposit_frequency.csv'))
    withdrawal_freq = pd.read_csv(os.path.join(BASE_DIR, 'withdrawal_ETH/withdrawal_frequency.csv'))
    daily_commits = pd.read_csv(os.path.join(BASE_DIR, 'github_dev_ETH/daily_commits.csv'), parse_dates=['commit_date'])
    search_freq = pd.read_csv(os.path.join(BASE_DIR, 'google_searching/search_daily.csv'))
    netflow_eth = pd.read_csv(os.path.join(BASE_DIR, 'netflow_ETH/netflow_eth.csv'))

    # Rename and clean columns
    deposit_freq.rename(columns={'Frequency': 'deposit_freq'}, inplace=True)
    withdrawal_freq.rename(columns={'Frequency': 'withdrawal_freq'}, inplace=True)
    daily_commits.rename(columns={'commit_date': 'Date', 'count': 'daily_commits'}, inplace=True)
    search_freq.rename(columns={'Frequency': 'search_freq'}, inplace=True)
    netflow_eth.rename(columns={'value': 'netflow_eth', 'timeStamp': 'Date'}, inplace=True)

    # Remove timezone and convert to datetime
    eth_prices['Date'] = eth_prices['Date'].dt.tz_localize(None)
    for df in [deposit_freq, withdrawal_freq, daily_commits, search_freq, netflow_eth]:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)

    # Filter by date range
    eth_prices = eth_prices[(eth_prices['Date'] >= start_date) & (eth_prices['Date'] <= end_date)]
    deposit_freq = deposit_freq[(deposit_freq['Date'] >= start_date) & (deposit_freq['Date'] <= end_date)]
    withdrawal_freq = withdrawal_freq[(withdrawal_freq['Date'] >= start_date) & (withdrawal_freq['Date'] <= end_date)]
    daily_commits = daily_commits[(daily_commits['Date'] >= start_date) & (daily_commits['Date'] <= end_date)]
    search_freq = search_freq[(search_freq['Date'] >= start_date) & (search_freq['Date'] <= end_date)]
    netflow_eth = netflow_eth[(netflow_eth['Date'] >= start_date) & (netflow_eth['Date'] <= end_date)]

    return eth_prices, deposit_freq, withdrawal_freq, daily_commits, search_freq, netflow_eth

# Existing functions, no changes required to use the filtered data
def cache_data(shared_cache, start_date, end_date):
    # 전달된 날짜 매개변수를 load_and_prepare_data에 전달
    eth_prices, deposit_freq, withdrawal_freq, daily_commits, search_freq, netflow_eth = load_and_prepare_data(start_date, end_date)

    shared_cache['eth_prices'] = eth_prices
    shared_cache['deposit_freq'] = deposit_freq
    shared_cache['withdrawal_freq'] = withdrawal_freq
    shared_cache['daily_commits'] = daily_commits
    shared_cache['search_freq'] = search_freq
    shared_cache['netflow_eth'] = netflow_eth

def merge_data(shared_cache):
    eth_prices = shared_cache['eth_prices']
    datasets = {
        'deposit_freq': shared_cache['deposit_freq'],
        'withdrawal_freq': shared_cache['withdrawal_freq'],
        'daily_commits': shared_cache['daily_commits'],
        'search_freq': shared_cache['search_freq'],
        'netflow_eth': shared_cache['netflow_eth']
    }
    merged_data = eth_prices
    for name, df in datasets.items():
        merged_data = pd.merge(merged_data, df, on='Date', how='left')
    merged_data.fillna(0, inplace=True)
    return merged_data

def safe_find_optimal_lag(series1, series2, dates, max_lag=30):
    valid_idx = (dates >= start_date) & (dates <= end_date) & series1.notna() & series2.notna()
    series1 = series1[valid_idx]
    series2 = series2[valid_idx]

    correlations = []
    for lag in range(1, max_lag + 1):
        shifted_series1 = series1.shift(lag)
        valid_lag_idx = shifted_series1.notna() & series2.notna()
        if valid_lag_idx.sum() > 10:
            try:
                corr = shifted_series1[valid_lag_idx].corr(series2[valid_lag_idx])
                correlations.append(corr)
            except Exception:
                correlations.append(np.nan)
        else:
            correlations.append(np.nan)

    correlations = [c for c in correlations if not np.isnan(c)]
    if not correlations:
        return None
    return np.argmax(np.abs(correlations)) + 1


# 시그널 생성 함수
def generate_signal(data, indicator, threshold, lag=None):
    data = data.copy()
    if lag:
        data[f'{indicator}_lag'] = data[indicator].shift(lag).fillna(0)
    else:
        data[f'{indicator}_lag'] = data[indicator]

    if indicator == 'deposit_freq':
        data[f'{indicator}_signal'] = np.where(data[f'{indicator}_lag'] > threshold, -1, 1)
    elif indicator == 'withdrawal_freq':
        data[f'{indicator}_signal'] = np.where(data[f'{indicator}_lag'] > threshold, 1, -1)
    elif indicator == 'netflow_eth':
        data[f'{indicator}_signal'] = np.where(data[f'{indicator}_lag'] > 0, 1, -1)
    elif indicator == 'daily_commits':
        data[f'{indicator}_signal'] = np.where(data[f'{indicator}_lag'] > threshold, 1, -1)
    elif indicator == 'search_freq':
        data[f'{indicator}_signal'] = np.where(data[f'{indicator}_lag'] > threshold, 1, -1)
    return data

# 백테스팅 함수
def safe_backtest(data, thresholds, cash=10000):
    data = data.copy()
    lags = {}

    # 각 지표별로 최적의 lag를 계산하고 시그널 생성
    for indicator, threshold in thresholds.items():
        lag = safe_find_optimal_lag(data[indicator], data['Close'], data['Date'])
        if lag is None:
            return None, None, None

        lags[indicator] = lag
        data = generate_signal(data, indicator, threshold, lag)

    # 최종 시그널 생성 (여러 지표의 시그널 합산 후 부호로 매수/매도 결정)
    data['final_signal'] = np.sign(
        np.sum([data[f'{indicator}_signal'] for indicator in thresholds.keys()], axis=0)
    )

    # 백테스팅: 매수 후 매도 순서로 거래를 진행
    position = 0  # 보유 자산 수량
    buy_sell_dates = []  # 매수/매도 기록
    initial_cash = cash  # 초기 현금
    for _, row in data.iterrows():
        if row['final_signal'] > 0 and cash > 0:  # 매수 신호
            position = cash / row['Close']  # 현재 현금을 사용해 자산 구매
            cash = 0
            buy_sell_dates.append((row['Date'], 'BUY', row['Close']))
        elif row['final_signal'] < 0 and position > 0:  # 매도 신호
            cash = position * row['Close']  # 보유 자산을 매도
            position = 0
            buy_sell_dates.append((row['Date'], 'SELL', row['Close']))


    # 최종 자산 가치 계산 (남은 현금 + 보유 자산 가치)
    final_value = cash + (position * data.iloc[-1]['Close'] if position > 0 else 0)
    return final_value, buy_sell_dates, lags



# 손실 보존률 계산 함수
def safe_calculate_loss_preservation(data, signals, initial_cash=10000):
    data = data.copy()
    data['Return'] = data['Close'].pct_change().fillna(0)

    downtrend = data['Return'] < 0
    market_down_close = data.loc[downtrend, 'Close']
    market_loss = (
        (market_down_close.iloc[-1] - market_down_close.iloc[0]) / market_down_close.iloc[0]
        if not market_down_close.empty else 0
    )

    cash = initial_cash
    position = 0
    portfolio = []

    for _, row in data.iterrows():
        signal = row['final_signal']
        price = row['Close']
        if signal == 1 and cash > 0:
            position = cash / price
            cash = 0
        elif signal == -1 and position > 0:
            cash = position * price
            position = 0
        portfolio.append(cash + position * price)

    final_value = portfolio[-1]
    strategy_loss = (final_value - initial_cash) / initial_cash
    return strategy_loss / market_loss if market_loss != 0 else None

# 조합 처리 함수
def process_combination(args):
    merged_data, combination, thresholds, initial_cash = args
    threshold_dict = dict(zip(combination, thresholds))
    final_value, buy_sell_dates, lags = safe_backtest(merged_data, threshold_dict)
    return {
        'combination': combination,
        'thresholds': threshold_dict,
        'portfolio_value': final_value,
        'lags': lags
    } if final_value is not None else None

# 병렬 작업 실행 함수
def run_combinations_in_parallel(shared_cache, all_indicators, initial_cash=10000):
    merged_data = merge_data(shared_cache)
    fixed_indicators = ['deposit_freq', 'withdrawal_freq']
    variable_indicators = [ind for ind in all_indicators if ind not in fixed_indicators]

    valid_combinations = [
        fixed_indicators + list(combo)
        for i in range(len(variable_indicators) + 1)
        for combo in combinations(variable_indicators, i)
    ]

    tasks = []
    for combination in valid_combinations:
        thresholds = product(*[
            all_indicators[ind] if all_indicators[ind] is not None else []
            for ind in combination
        ])
        for threshold in thresholds:
            tasks.append((merged_data, combination, threshold, initial_cash))

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_combination, tasks)
    return [res for res in results if res is not None]


# 최상위 결과 출력 (매수, 매도 시간 및 손실 보존률 포함)
def print_top_results_with_details(results, merged_data, type, top_n=3, initial_cash=10000):

    # Calculate loss preservation ratios for all results
    for res in results:
        merged_data_with_signals = merged_data.copy()
        for indicator, lag in res['lags'].items():
            merged_data_with_signals = generate_signal(
                merged_data_with_signals, indicator, res['thresholds'][indicator], lag)
        merged_data_with_signals['final_signal'] = np.sign(np.sum([
            merged_data_with_signals[f'{indicator}_signal'] for indicator in res['thresholds'].keys()
        ], axis=0))

        # Calculate and store loss preservation ratio
        loss_preservation = safe_calculate_loss_preservation(
            merged_data_with_signals, res['thresholds'], initial_cash)
        res['loss_preservation'] = loss_preservation

    if type == "aggressive":
        # Sort results by portfolio value in descending order
        sorted_results = sorted(results, key=lambda x: x['portfolio_value'], reverse=True)[:top_n]
    elif type == "defensive":
        # Filter profitable results and sort by loss preservation ratio
        defensive_results = [res for res in results if res['portfolio_value'] > initial_cash]
        sorted_results = sorted(defensive_results, key=lambda x: x['loss_preservation'])[:top_n]
    else:
        print("Invalid type specified. Choose 'aggressive' or 'defensive'.")
        return

    if not sorted_results:
        print(f"No results found for {type} strategy.")
        return

    print(f"Top {len(sorted_results)} Results ({type.title()} Strategy):")
    for idx, res in enumerate(sorted_results, start=1):
        print(f"Rank {idx}:")
        print(f"  Portfolio Value: {res['portfolio_value']}")
        print(f"  Combination: {res['combination']}")
        print(f"  Lags: {res['lags']}")
        print(f"  Thresholds: {res['thresholds']}")
        print(f"  Loss Preservation Ratio: {res['loss_preservation']:.2f}")

        # Print Buy/Sell Dates
        print("  Buy/Sell Dates:")
        _, buy_sell_dates, _ = safe_backtest(merged_data, res['thresholds'], cash=initial_cash)
        for date, action, price in buy_sell_dates:
            print(f"    {date} - {action} at {price:.2f}")
        print("-" * 50)

def get_top_results_with_details(results, merged_data, strategy_type, top_n=3, initial_cash=10000):
    # Calculate loss preservation ratios for all results
    for res in results:
        merged_data_with_signals = merged_data.copy()
        for indicator, lag in res['lags'].items():
            merged_data_with_signals = generate_signal(
                merged_data_with_signals, indicator, res['thresholds'][indicator], lag)
        merged_data_with_signals['final_signal'] = np.sign(np.sum([
            merged_data_with_signals[f'{indicator}_signal'] for indicator in res['thresholds'].keys()
        ], axis=0))

        # Calculate and store loss preservation ratio
        loss_preservation = safe_calculate_loss_preservation(
            merged_data_with_signals, res['thresholds'], initial_cash)
        res['loss_preservation'] = loss_preservation

    if strategy_type == "aggressive":
        # Sort results by portfolio value in descending order
        sorted_results = sorted(results, key=lambda x: x['portfolio_value'], reverse=True)[:top_n]
    elif strategy_type == "defensive":
        # Filter profitable results and sort by loss preservation ratio
        defensive_results = [res for res in results if res['portfolio_value'] > initial_cash]
        sorted_results = sorted(defensive_results, key=lambda x: x['loss_preservation'])[:top_n]
    elif strategy_type == "balanced":
        # Filter results with 0 < loss preservation < 1 and sort by portfolio value
        balanced_results = [
            res for res in results if 0.2 < res['loss_preservation'] < 0.8
        ]
        sorted_results = sorted(balanced_results, key=lambda x: x['portfolio_value'], reverse=True)[:top_n]
    else:
        print("Invalid type specified. Choose 'aggressive' or 'defensive'.")
        return []

    output = {
        "strategy": strategy_type,
        "results": []
    }

    for idx, res in enumerate(sorted_results, start=1):
        _, buy_sell_dates, _ = safe_backtest(merged_data, res['thresholds'], cash=initial_cash)
        output["results"].append({
            "rank": int(idx),
            "portfolio_value": float(res['portfolio_value']),
            "combination": res['combination'],
            "lags": {k: int(v) for k, v in res['lags'].items()},
            "thresholds": {k: float(v) for k, v in res['thresholds'].items()},
            "loss_preservation_ratio": float(res['loss_preservation']),
            "buy_sell_dates": [
                {"date": date.strftime("%Y-%m-%d %H:%M:%S"), "action": action}
                for date, action, price in buy_sell_dates
            ]
        })

    return output

if __name__ == "__main__":
    args = parse_arguments()

    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    combination = args.combination.split(",")
    initial_cash = args.asset
    with Manager() as manager:
        shared_cache = manager.dict()

        # Cache data within the date range
        cache_data(shared_cache, start_date, end_date)

        # Filter indicators based on user input
        all_indicators = {
            'deposit_freq': [0.5, 1, 2, 3],
            'withdrawal_freq': [0.5, 1],
            'daily_commits': [1, 3, 5, 7],
            'search_freq': [8, 13.07, 13.21, 15, 15.64],
            'netflow_eth': None
        }
        filtered_indicators = {key: all_indicators[key] for key in combination if key in all_indicators}

        # Run combinations in parallel
        results = run_combinations_in_parallel(shared_cache, filtered_indicators, initial_cash)
        merged_data = merge_data(shared_cache)

        # Generate strategies for defensive, aggressive, and balanced
        strategies = []
        for strategy_type in ["defensive", "aggressive", "balanced"]:
            strategies.append(get_top_results_with_details(results, merged_data, strategy_type))

        # Save results to JSON

        final_output = {"strategies": strategies}
        json_output = json.dumps(final_output, indent=4)

        # Save results to file
        with open("results.json", "w") as file:
            file.write(json_output)

        # Print JSON output (for response)
        print(json_output)