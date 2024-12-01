import pandas as pd
import numpy as np
from itertools import product, combinations
from multiprocessing import Pool, Manager, cpu_count
import time  # Execution time measurement

# User-defined date range
start_date = pd.to_datetime("2022-11-08")
end_date = pd.to_datetime("2024-11-08")


def load_and_prepare_data_partial():
    """Load and preprocess datasets, filter by date range for uncached data."""
    # Read only the uncached datasets
    daily_commits = pd.read_csv('../github_dev_ETH/daily_commits.csv', parse_dates=['commit_date'])
    search_freq = pd.read_csv('../google_searching/search_daily.csv')
    netflow_eth = pd.read_csv('../netflow_ETH/netflow_eth.csv')

    # Rename and clean columns
    daily_commits.rename(columns={'commit_date': 'Date', 'count': 'daily_commits'}, inplace=True)
    search_freq.rename(columns={'Frequency': 'search_freq'}, inplace=True)
    netflow_eth.rename(columns={'value': 'netflow_eth', 'timeStamp': 'Date'}, inplace=True)

    # Remove timezone and filter by date range
    for df in [daily_commits, search_freq, netflow_eth]:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)

    daily_commits = daily_commits[(daily_commits['Date'] >= start_date) & (daily_commits['Date'] <= end_date)]
    search_freq = search_freq[(search_freq['Date'] >= start_date) & (search_freq['Date'] <= end_date)]
    netflow_eth = netflow_eth[(netflow_eth['Date'] >= start_date) & (netflow_eth['Date'] <= end_date)]

    return daily_commits, search_freq, netflow_eth


def cache_data_partial(shared_cache):
    """Cache only eth_prices, deposit_freq, and withdrawal_freq."""
    eth_prices = pd.read_csv('../ethereum_daily_prices.csv', parse_dates=['Date'])
    deposit_freq = pd.read_csv('../deposit_ETH/deposit_frequency.csv')
    withdrawal_freq = pd.read_csv('../withdrawal_ETH/withdrawal_frequency.csv')

    # Rename and clean columns
    deposit_freq.rename(columns={'Frequency': 'deposit_freq'}, inplace=True)
    withdrawal_freq.rename(columns={'Frequency': 'withdrawal_freq'}, inplace=True)

    # Remove timezone and filter by date range
    eth_prices['Date'] = eth_prices['Date'].dt.tz_localize(None)
    for df in [deposit_freq, withdrawal_freq]:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)

    eth_prices = eth_prices[(eth_prices['Date'] >= start_date) & (eth_prices['Date'] <= end_date)]
    deposit_freq = deposit_freq[(deposit_freq['Date'] >= start_date) & (deposit_freq['Date'] <= end_date)]
    withdrawal_freq = withdrawal_freq[(withdrawal_freq['Date'] >= start_date) & (withdrawal_freq['Date'] <= end_date)]

    # Cache the datasets
    shared_cache['eth_prices'] = eth_prices
    shared_cache['deposit_freq'] = deposit_freq
    shared_cache['withdrawal_freq'] = withdrawal_freq


def merge_data_partial(shared_cache):
    """Merge cached data with directly loaded data."""
    # Load cached data
    eth_prices = shared_cache['eth_prices']
    deposit_freq = shared_cache['deposit_freq']
    withdrawal_freq = shared_cache['withdrawal_freq']

    # Load non-cached data
    daily_commits, search_freq, netflow_eth = load_and_prepare_data_partial()

    # Merge all data
    datasets = {
        'deposit_freq': deposit_freq,
        'withdrawal_freq': withdrawal_freq,
        'daily_commits': daily_commits,
        'search_freq': search_freq,
        'netflow_eth': netflow_eth
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


def generate_signal(data, indicator, threshold, lag=None):
    data = data.copy()
    if lag:
        data[f'{indicator}_lag'] = data[indicator].shift(lag).fillna(0)
    else:
        data[f'{indicator}_lag'] = data[indicator]

    data[f'{indicator}_signal'] = np.where(data[f'{indicator}_lag'] > threshold, 1, -1)
    return data


def safe_backtest(data, thresholds, cash=10000):
    data = data.copy()
    lags = {}

    for indicator, threshold in thresholds.items():
        lag = safe_find_optimal_lag(data[indicator], data['Close'], data['Date'])
        if lag is None:
            return None, None, None

        lags[indicator] = lag
        data = generate_signal(data, indicator, threshold, lag)

    data['final_signal'] = np.sign(
        np.sum([data[f'{indicator}_signal'] for indicator in thresholds.keys()], axis=0)
    )

    position = 0
    buy_sell_dates = []
    for _, row in data.iterrows():
        if row['final_signal'] > 0 and cash > 0:
            position = cash / row['Close']
            cash = 0
            buy_sell_dates.append((row['Date'], 'BUY', row['Close']))
        elif row['final_signal'] < 0 and position > 0:
            cash = position * row['Close']
            position = 0
            buy_sell_dates.append((row['Date'], 'SELL', row['Close']))

    final_value = cash + (position * data.iloc[-1]['Close'] if position > 0 else 0)
    return final_value, buy_sell_dates, lags


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


def run_combinations_in_parallel(shared_cache, all_indicators, initial_cash=10000):
    print("Merging cached data...")
    merged_data = merge_data_partial(shared_cache)
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


def print_top_results_with_details(results, merged_data, type, top_n=3, initial_cash=10000):
    for res in results:
        merged_data_with_signals = merged_data.copy()
        for indicator, lag in res['lags'].items():
            merged_data_with_signals = generate_signal(
                merged_data_with_signals, indicator, res['thresholds'][indicator], lag)
        merged_data_with_signals['final_signal'] = np.sign(np.sum([
            merged_data_with_signals[f'{indicator}_signal'] for indicator in res['thresholds'].keys()
        ], axis=0))

        loss_preservation = safe_calculate_loss_preservation(
            merged_data_with_signals, res['thresholds'], initial_cash)
        res['loss_preservation'] = loss_preservation

    if type == "aggressive":
        sorted_results = sorted(results, key=lambda x: x['portfolio_value'], reverse=True)[:top_n]
    elif type == "defensive":
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

        _, buy_sell_dates, _ = safe_backtest(merged_data, res['thresholds'], cash=initial_cash)
        print("  Buy/Sell Dates:")
        for date, action, price in buy_sell_dates:
            print(f"    {date} - {action} at {price:.2f}")
        print("-" * 50)


if __name__ == "__main__":
    with Manager() as manager:
        total_start_time = time.time()
        shared_cache = manager.dict()

        cache_process = Pool(processes=5)
        cache_process.apply(cache_data_partial, (shared_cache,))
        cache_process.close()
        cache_process.join()

        all_indicators = {
            'deposit_freq': [0.5, 1, 2, 3, 4],
            'withdrawal_freq': [0.5, 1],
            #'daily_commits': [1, 3, 5, 7],
            #'search_freq': [8, 13.07, 13.21, 15, 15.64],
            #'netflow_eth': []
        }

        results = run_combinations_in_parallel(shared_cache, all_indicators)
        merged_data = merge_data_partial(shared_cache)

        strategy_type = "aggressive"
        print_top_results_with_details(results, merged_data, type=strategy_type)

        total_end_time = time.time()
        print(f"\nTotal execution time: {total_end_time - total_start_time:.2f} seconds.")

#33.04초 - 모든조합
#9.71초 - search 제외
#8.38초 - commit 제외