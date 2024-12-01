import pandas as pd
import numpy as np
from itertools import product, combinations
from multiprocessing import Pool, cpu_count
import time  # 실행 시간 측정을 위한 모듈
# CSV 파일 읽기 (파일 경로 수정)
eth_prices = pd.read_csv('../ethereum_daily_prices.csv', parse_dates=['Date'])
deposit_freq = pd.read_csv('../deposit_ETH/deposit_frequency.csv')
daily_commits = pd.read_csv('../github_dev_ETH/daily_commits.csv', parse_dates=['commit_date'])
netflow_eth = pd.read_csv('../netflow_ETH/netflow_eth.csv')
withdrawal_freq = pd.read_csv('../withdrawal_ETH/withdrawal_frequency.csv')
search_freq = pd.read_csv('../google_searching/search_daily.csv')

# 열 이름 변경
deposit_freq.rename(columns={'Frequency': 'deposit_freq'}, inplace=True)
withdrawal_freq.rename(columns={'Frequency': 'withdrawal_freq'}, inplace=True)
daily_commits.rename(columns={'commit_date': 'Date', 'count': 'daily_commits'}, inplace=True)
search_freq.rename(columns={'Frequency': 'search_freq'}, inplace=True)
netflow_eth.rename(columns={'value': 'netflow_eth'}, inplace=True)
netflow_eth.rename(columns={'timeStamp': 'Date'}, inplace=True)
# UTC 제거 및 데이터 병합 준비
eth_prices['Date'] = eth_prices['Date'].dt.tz_localize(None)
withdrawal_freq['Date'] = pd.to_datetime(withdrawal_freq['Date'], errors='coerce')
deposit_freq['Date'] = pd.to_datetime(deposit_freq['Date'],errors='coerce')
daily_commits['Date'] = pd.to_datetime(daily_commits['Date'],errors='coerce')
search_freq['Date'] = pd.to_datetime(search_freq['Date'],errors='coerce')
netflow_eth['Date'] = pd.to_datetime(netflow_eth['Date'],errors='coerce')



datasets = {
    'deposit_freq': deposit_freq,
    'daily_commits': daily_commits,
    'netflow_eth': netflow_eth,
    'withdrawal_freq': withdrawal_freq,
    'search_freq': search_freq
}

for key, data in datasets.items():
    if 'timeStamp' in data.columns:
        data['Date'] = pd.to_datetime(data['timeStamp'], errors='coerce').dt.tz_localize('UTC').dt.tz_convert(None)
        data.drop(columns=['timeStamp'], inplace=True)
    elif 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce').dt.tz_localize(None)

# 데이터 병합
print("Merging datasets...")
merged_data = eth_prices
for key, data in datasets.items():
    print(f"Merging {key}...")
    merged_data = pd.merge(merged_data, data, on='Date', how='left')

# 결측치 처리 및 데이터 타입 변환
merged_data.fillna(0, inplace=True)
for col in merged_data.columns:
    if col != "Date" and not np.issubdtype(merged_data[col].dtype, np.number):
        print(f"Warning: Non-numeric data in column '{col}', converting to numeric.")
        merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

dates = merged_data['Date']

# 사용자 입력: 날짜 범위
start_date = pd.to_datetime("2020-11-08")
end_date = pd.to_datetime("2024-11-08")
print(f"Filtering data from {start_date} to {end_date}...")
merged_data = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)]

# 지표별 threshold 범위 설정
all_indicators = {
    'deposit_freq': [0.5, 1, 2, 3, 4],
    'withdrawal_freq': [0.5, 1],
    'daily_commits': [1, 3, 5, 7],
    'netflow_eth': None,
    'search_freq': [8, 13.07, 13.21, 15, 15.64]
}

def safe_find_optimal_lag(series1, series2, dates, start_date, end_date, max_lag=30):
    # 날짜 범위 필터링
    valid_idx = (dates >= start_date) & (dates <= end_date) & series1.notna() & series2.notna()
    series1 = series1[valid_idx]
    series2 = series2[valid_idx]

    correlations = []
    for lag in range(1, max_lag + 1):
        shifted_series1 = series1.shift(lag)
        valid_lag_idx = shifted_series1.notna() & series2.notna()
        if valid_lag_idx.sum() > 10:  # 충분한 데이터가 있을 때만 계산
            try:
                corr = shifted_series1[valid_lag_idx].corr(series2[valid_lag_idx])
                correlations.append(corr)
            except Exception:
                correlations.append(np.nan)
        else:
            correlations.append(np.nan)

    # 유효한 상관계수 중 최대값 반환
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

    for indicator, threshold in thresholds.items():
        lag = safe_find_optimal_lag(data[indicator], data['Close'],dates,start_date,end_date)

        if lag is None:
            return None, None, None

        lags[indicator] = lag
        data = generate_signal(data, indicator, threshold, lag)

    data['final_signal'] = np.sign(
        np.sum([data[f'{indicator}_signal'] for indicator in thresholds.keys()], axis=0)
    )

    buy_signals = data.loc[data['final_signal'] == 1]
    sell_signals = data.loc[data['final_signal'] == -1]

    position = 0
    buy_sell_dates = []
    for _, row in buy_signals.iterrows():
        if cash > 0:
            position = cash / row['Close']
            cash = 0
            buy_sell_dates.append((row['Date'], 'BUY', row['Close']))
    for _, row in sell_signals.iterrows():
        if position > 0:
            cash = position * row['Close']
            position = 0
            buy_sell_dates.append((row['Date'], 'SELL', row['Close']))

    final_value = cash + position * data.iloc[-1]['Close']
    return final_value, buy_sell_dates, lags

# 손실보존률 계산 함수
def safe_calculate_loss_preservation(data, signals, initial_cash=10000):
    data = data.copy()
    data['Return'] = data['Close'].pct_change().fillna(0)

    downtrend = data['Return'] < 0
    market_down_close = data.loc[downtrend, 'Close']
    market_loss = (
        (market_down_close.iloc[-1] - market_down_close.iloc[0]) / market_down_close.iloc[0]
        if not market_down_close.empty else 0
    )

    signals = signals[['Date', 'final_signal']].drop_duplicates()
    data = data.merge(signals, on='Date', how='left')
    data['final_signal'] = data['final_signal'].fillna(0)

    cash = initial_cash
    position = 0
    portfolio = []

    for i in range(len(data)):
        signal = data.iloc[i]['final_signal']
        price = data.iloc[i]['Close']
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

# 기존의 모든 함수 정의 (safe_backtest, safe_calculate_loss_preservation, etc.) 유지...

def process_combination(args):
    selected_data, indicator_combination, thresholds, initial_cash = args

    threshold_dict = dict(zip(indicator_combination, thresholds))
    final_value, buy_sell_dates, lags = safe_backtest(selected_data, threshold_dict)

    if final_value is None:
        return None

    signals = pd.DataFrame(buy_sell_dates, columns=['Date', 'Action', 'Price'])
    signals['final_signal'] = signals['Action'].apply(lambda x: 1 if x == 'BUY' else -1)
    loss_preservation = safe_calculate_loss_preservation(selected_data, signals, initial_cash)

    return {
        'combination': indicator_combination,
        'thresholds': threshold_dict,
        'portfolio_value': final_value,
        'loss_preservation_ratio': loss_preservation,
        'buy_sell_dates': buy_sell_dates,
        'lags': lags
    }

def run_combinations_in_parallel(merged_data, all_indicators, initial_cash=10000):
    fixed_indicators = ['deposit_freq', 'withdrawal_freq']
    other_indicators = [key for key in all_indicators.keys() if key not in fixed_indicators]
    valid_combinations = [
        fixed_indicators + list(combination)
        for i in range(0, len(other_indicators) + 1)
        for combination in combinations(other_indicators, i)
    ]

    tasks = []
    for indicator_combination in valid_combinations:
        threshold_ranges = {key: all_indicators[key] for key in indicator_combination}
        threshold_combinations = list(product(*[threshold_ranges[key] for key in indicator_combination if threshold_ranges[key]]))
        selected_data = merged_data[['Date', 'Close'] + indicator_combination]

        for thresholds in threshold_combinations:
            tasks.append((selected_data, indicator_combination, thresholds, initial_cash))

    start_time = time.time()  # 시작 시간 기록
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_combination, tasks)
    end_time = time.time()  # 종료 시간 기록

    print(f"\nParallel processing completed in {end_time - start_time:.2f} seconds.")
    return [result for result in results if result is not None]

def print_top_results_by_portfolio_value(results, top_n=3):
    sorted_results = sorted(results, key=lambda x: x['portfolio_value'], reverse=True)[:top_n]
    if not sorted_results:
        print("No valid results found.")
        return

    print(f"\nTop {top_n} Results Based on Portfolio Value:")
    for idx, result in enumerate(sorted_results, start=1):
        print(f"\nRank {idx}:")
        print(f"Combination: {result['combination']}")
        print(f"Thresholds: {result['thresholds']}")
        print(f"Final Portfolio Value: ${result['portfolio_value']:.2f}")
        print(f"Loss Preservation Ratio: {result['loss_preservation_ratio']:.4f}")
        print(f"Optimal Lags: {result['lags']}")
        print("Buy/Sell Dates:")
        for date, action, price in result['buy_sell_dates']:
            print(f"  {date} - {action} at ${price:.2f}")

if __name__ == '__main__':
    print("Running backtests in parallel...")

    # 전체 처리 시작 시간
    total_start_time = time.time()

    combination_results = run_combinations_in_parallel(merged_data, all_indicators)

    # 전체 처리 종료 시간
    total_end_time = time.time()

    print(f"\nTotal execution time: {total_end_time - total_start_time:.2f} seconds.")

    investor_type = "defensive"
    try:
        top_results = [
            res for res in combination_results
            if res['loss_preservation_ratio'] is not None and res['portfolio_value'] > 10000
        ]
        if investor_type == "defensive":
            top_results = sorted(top_results, key=lambda x: (x['loss_preservation_ratio'], -x['portfolio_value']))
        elif investor_type == "aggressive":
            top_results = sorted(top_results, key=lambda x: -x['portfolio_value'])

        print_top_results_by_portfolio_value(top_results, top_n=3)
    except ValueError as e:
        print(f"Error: {e}")
