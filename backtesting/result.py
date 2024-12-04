import pandas as pd
import numpy as np
from itertools import product, combinations
from multiprocessing import Pool, Manager, cpu_count
import time  # Execution time measurement
from functools import reduce

# User-defined date range
start_date = pd.to_datetime("2022-11-08")
end_date = pd.to_datetime("2024-11-08")

def load_and_process_csv(file_info):
    file_path, parse_dates, rename_columns = file_info
    df = pd.read_csv(file_path, parse_dates=parse_dates)

    # Rename and clean columns if provided
    if rename_columns:
        df.rename(columns=rename_columns, inplace=True)

    # Remove timezone and convert to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)

    # Filter by date range
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    return df

def load_and_prepare_data_parallel():
    # Define file paths and configurations for parallel processing
    file_configs = [
        ('../ethereum_daily_prices.csv', ['Date'], None),
        ('../deposit_ETH/deposit_frequency.csv', None, {'Frequency': 'deposit_freq'}),
        ('../withdrawal_ETH/withdrawal_frequency.csv', None, {'Frequency': 'withdrawal_freq'}),
        ('../github_dev_ETH/daily_commits.csv', ['commit_date'], {'commit_date': 'Date', 'count': 'daily_commits'}),
        ('../google_searching/search_daily.csv', None, {'Frequency': 'search_freq'}),
        ('../netflow_ETH/netflow_eth.csv', None, {'value': 'netflow_eth', 'timeStamp': 'Date'})
    ]

    # Use multiprocessing to load and process CSV files
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(load_and_process_csv, file_configs)

    # Map results back to respective variables
    eth_prices, deposit_freq, withdrawal_freq, daily_commits, search_freq, netflow_eth = results
    return eth_prices, deposit_freq, withdrawal_freq, daily_commits, search_freq, netflow_eth

# Cache function for parallelized data loading
def cache_data(shared_cache):
    eth_prices, deposit_freq, withdrawal_freq, daily_commits, search_freq, netflow_eth = load_and_prepare_data_parallel()
    shared_cache['eth_prices'] = eth_prices
    shared_cache['deposit_freq'] = deposit_freq
    shared_cache['withdrawal_freq'] = withdrawal_freq
    shared_cache['daily_commits'] = daily_commits
    shared_cache['search_freq'] = search_freq
    shared_cache['netflow_eth'] = netflow_eth

def merge_data(shared_cache):
    """
    Merge all dataframes in shared_cache into a single dataframe using functools.reduce
    """
    eth_prices = shared_cache['eth_prices']
    datasets = {
        'deposit_freq': shared_cache['deposit_freq'],
        'withdrawal_freq': shared_cache['withdrawal_freq'],
        'daily_commits': shared_cache['daily_commits'],
        'search_freq': shared_cache['search_freq'],
        'netflow_eth': shared_cache['netflow_eth']
    }

    # Use reduce to merge all datasets with eth_prices
    merged_data = reduce(
        lambda left, right: pd.merge(left, right, on='Date', how='left'),
        [eth_prices] + list(datasets.values())
    )

    # Fill NaN values with 0
    merged_data.fillna(0, inplace=True)
    return merged_data

def safe_find_optimal_lag(series1, series2, dates, max_lag=30):
    """
    Calculate the optimal lag using NumPy for performance optimization.
    """
    # Ensure valid inputs are aligned
    valid_idx = (dates >= start_date) & (dates <= end_date) & series1.notna() & series2.notna()
    series1 = series1[valid_idx].to_numpy()
    series2 = series2[valid_idx].to_numpy()

    if len(series1) == 0 or len(series2) == 0:
        return None

    # Pre-allocate an array for correlations
    correlations = np.full(max_lag, np.nan)

    for lag in range(1, max_lag + 1):
        # Shift series1 by lag using NumPy slicing
        if lag >= len(series1):
            break  # Skip if lag exceeds available data points
        shifted_series1 = series1[:-lag]
        valid_lag_series2 = series2[lag:]

        # Check for sufficient valid points
        if len(shifted_series1) < 10 or len(valid_lag_series2) < 10:
            continue

        # Compute correlation using NumPy
        try:
            corr = np.corrcoef(shifted_series1, valid_lag_series2)[0, 1]
            correlations[lag - 1] = corr
        except Exception:
            correlations[lag - 1] = np.nan

    # Return the lag with the maximum absolute correlation
    if not np.any(~np.isnan(correlations)):
        return None
    return np.nanargmax(np.abs(correlations)) + 1



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
def run_combinations_in_parallel(shared_cache, all_indicators, initial_cash=10000, batch_size=1000):
    """
    Run combinations in parallel with batch processing for performance improvement.
    """
    print("Merging cached data...")
    merged_data = merge_data(shared_cache)
    fixed_indicators = ['deposit_freq', 'withdrawal_freq']
    variable_indicators = [ind for ind in all_indicators if ind not in fixed_indicators]

    # Generate valid combinations
    valid_combinations = [
        fixed_indicators + list(combo)
        for i in range(len(variable_indicators) + 1)
        for combo in combinations(variable_indicators, i)
    ]

    # Prepare tasks
    tasks = []
    for combination in valid_combinations:
        thresholds = product(*[
            all_indicators[ind] if all_indicators[ind] is not None else []
            for ind in combination
        ])
        for threshold in thresholds:
            tasks.append((merged_data, combination, threshold, initial_cash))

    # Split tasks into batches
    num_batches = int(np.ceil(len(tasks) / batch_size))
    batched_tasks = [tasks[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

    # Process each batch in parallel
    results = []
    with Pool(processes=cpu_count()) as pool:
        for batch in batched_tasks:
            batch_results = pool.map(process_combination, batch)
            results.extend([res for res in batch_results if res is not None])

    return results


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

    # Filter results based on strategy type
    if type == "aggressive":
        # Sort by portfolio value in descending order
        sorted_results = sorted(results, key=lambda x: x['portfolio_value'], reverse=True)[:top_n]
    elif type == "defensive":
        # Filter profitable results and sort by loss preservation ratio
        defensive_results = [res for res in results if res['portfolio_value'] > initial_cash]
        sorted_results = sorted(defensive_results, key=lambda x: x['loss_preservation'], reverse=True)[:top_n]
    elif type == "balanced":
        # Filter results with 0 < loss preservation < 1 and sort by portfolio value
        balanced_results = [
            res for res in results if 0 < res['loss_preservation'] < 1
        ]
        sorted_results = sorted(balanced_results, key=lambda x: x['portfolio_value'], reverse=True)[:top_n]
    else:
        print("Invalid type specified. Choose 'aggressive', 'defensive', or 'balanced'.")
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

if __name__ == "__main__":
    with Manager() as manager:
        total_start_time = time.time()
        shared_cache = manager.dict()

        # Cache data in the main process
        cache_data(shared_cache)

        all_indicators = {
            'deposit_freq': [0.5, 1, 2, 3, 4],
            'withdrawal_freq': [0.5, 1],
            #'daily_commits': [1, 3, 5, 7],
            #'search_freq': [8, 13.07, 13.21, 15, 15.64]
        }

        # Run parallel processing with batching
        results = run_combinations_in_parallel(shared_cache, all_indicators, batch_size=500)

        # Merge and evaluate results
        merged_data = merge_data(shared_cache)

        print_top_results_with_details(results, merged_data, type='defensive')
        print_top_results_with_details(results, merged_data, type='aggressive')
        print_top_results_with_details(results, merged_data, type='balanced')


        total_end_time = time.time()
        print(f"\nTotal execution time: {total_end_time - total_start_time:.2f} seconds.")

#33.25초