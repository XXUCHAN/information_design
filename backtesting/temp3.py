import pandas as pd
import numpy as np
from itertools import product, combinations

# CSV 파일 읽기 (파일 경로 수정)
eth_prices = pd.read_csv('../ethereum_daily_prices.csv', parse_dates=['Date'])
deposit_freq = pd.read_csv('../deposit_ETH/deposit_frequency.csv')
daily_commits = pd.read_csv('../github_dev_ETH/daily_commits.csv', parse_dates=['commit_date'])
netflow_eth = pd.read_csv('../netflow_ETH/netflow_eth.csv')
withdrawal_freq = pd.read_csv('../withrawal_ETH/withdrawal_frequency.csv')
search_freq = pd.read_csv('../google_searching/search_daily.csv')

# 열 이름 변경
deposit_freq.rename(columns={'Frequency': 'deposit_freq'}, inplace=True)
withdrawal_freq.rename(columns={'Frequency': 'withdrawal_freq'}, inplace=True)
daily_commits.rename(columns={'commit_date': 'Date', 'count': 'daily_commits'}, inplace=True)
search_freq.rename(columns={'Frequency': 'search_freq'}, inplace=True)
netflow_eth.rename(columns={'value': 'netflow_eth'}, inplace=True)

# UTC 제거 및 데이터 병합 준비
eth_prices['Date'] = eth_prices['Date'].dt.tz_localize(None)
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

# 사용자 입력: 날짜 범위
start_date = pd.to_datetime("2022-11-08")
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

# TLCC를 계산하는 함수
def safe_find_optimal_lag(series1, series2, max_lag=30):
    correlations = []
    for lag in range(1, max_lag + 1):
        shifted_series1 = series1.shift(lag)
        valid_idx = shifted_series1.notna() & series2.notna()

        if valid_idx.sum() > 10:
            try:
                corr = shifted_series1[valid_idx].corr(series2[valid_idx])
                correlations.append(corr)
            except Exception as e:
                print(f"Correlation calculation failed for lag {lag}: {e}")
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

    for indicator, threshold in thresholds.items():
        lag = safe_find_optimal_lag(data[indicator], data['Close'])

        if lag is None:
            return None, None

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
    return final_value, buy_sell_dates

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

# 조합 생성
fixed_indicators = ['deposit_freq', 'withdrawal_freq']
other_indicators = [key for key in all_indicators.keys() if key not in fixed_indicators]
valid_combinations = [
    fixed_indicators + list(combination)
    for i in range(0, len(other_indicators) + 1)
    for combination in combinations(other_indicators, i)
]

# 테스트 실행
combination_results = []
print("Running backtests for all combinations...")
for indicator_combination in valid_combinations:
    print(f"Testing combination: {indicator_combination}...")
    threshold_ranges = {key: all_indicators[key] for key in indicator_combination}
    threshold_combinations = list(product(*[threshold_ranges[key] for key in indicator_combination if threshold_ranges[key]]))
    selected_data = merged_data[['Date', 'Close'] + indicator_combination]

    for thresholds in threshold_combinations:
        threshold_dict = dict(zip(indicator_combination, thresholds))
        final_value, buy_sell_dates = safe_backtest(selected_data, threshold_dict)

        if final_value is None:
            print(f"Skipping combination: {indicator_combination} with thresholds {threshold_dict} (No valid correlation).")
            continue

        signals = pd.DataFrame(buy_sell_dates, columns=['Date', 'Action', 'Price'])
        signals['final_signal'] = signals['Action'].apply(lambda x: 1 if x == 'BUY' else -1)
        loss_preservation = safe_calculate_loss_preservation(selected_data, signals)

        combination_results.append({
            'combination': indicator_combination,
            'thresholds': threshold_dict,
            'portfolio_value': final_value,
            'loss_preservation_ratio': loss_preservation,
            'buy_sell_dates': buy_sell_dates
        })
# 수익률 상위 3개 결과 출력 함수
def print_top_results_by_portfolio_value(results, top_n=3):
    # 결과를 포트폴리오 가치로 정렬
    sorted_results = sorted(results, key=lambda x: x['portfolio_value'], reverse=True)[:top_n]

    if not sorted_results:
        print("No valid results found.")
        return

    print(f"\nTop {len(sorted_results)} Results Based on Portfolio Value:")
    for idx, result in enumerate(sorted_results, start=1):
        print(f"\nRank {idx}:")
        print(f"Combination: {result['combination']}")
        print(f"Thresholds: {result['thresholds']}")
        print(f"Final Portfolio Value: ${result['portfolio_value']:.2f}")
        print(f"Loss Preservation Ratio: {result['loss_preservation_ratio']:.4f}")
        print("Buy/Sell Dates:")
        for date, action, price in result['buy_sell_dates']:
            print(f"  {date} - {action} at ${price:.2f}")

# 투자 전략별 결과 필터링
def filter_results_by_strategy(results, investor_type, initial_cash=10000):
    # 유효한 결과 필터링
    valid_results = [
        result for result in results
        if result['loss_preservation_ratio'] is not None and result['portfolio_value'] > initial_cash
    ]

    if not valid_results:
        print("No valid scenarios found with profit above the initial investment.")
        return []

    if investor_type == "defensive":
        # 손실보존률이 가장 작은 것 중 수익률이 높은 순으로 정렬
        return sorted(valid_results, key=lambda x: (x['loss_preservation_ratio'], -x['portfolio_value']))
    elif investor_type == "aggressive":
        # 손실보존률이 1 미만인 결과 중 최대 수익률 기준으로 정렬
        aggressive_results = [res for res in valid_results if res['loss_preservation_ratio'] < 1]
        if not aggressive_results:
            print("No valid aggressive scenarios with loss preservation ratio less than 1.")
            return []
        return sorted(aggressive_results, key=lambda x: -x['portfolio_value'])
    else:
        raise ValueError("Invalid investor type. Choose 'defensive' or 'aggressive'.")

# 투자자 성향 입력 및 결과 출력
investor_type = input("Select investor type (aggressive/defensive): ").strip().lower()

try:
    # 전략 필터링
    top_results = filter_results_by_strategy(combination_results, investor_type)
    if not top_results:
        raise ValueError("Not enough valid results to display.")

    # 상위 3개 수익률 출력
    print_top_results_by_portfolio_value(top_results, top_n=3)
except ValueError as e:
    print(f"Error: {e}")
