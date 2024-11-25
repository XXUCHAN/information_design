import pandas as pd
import numpy as np
from itertools import product, combinations

# CSV 파일 읽기 (파일 경로 수정)
eth_prices = pd.read_csv('../ethereum_daily_prices.csv', parse_dates=['Date'])
deposit_freq = pd.read_csv('../deposit_ETH/deposit_frequency.csv')
#deposit_value = pd.read_csv('../deposit_ETH/deposit_volume.csv')
daily_commits = pd.read_csv('../github_dev_ETH/daily_commits.csv', parse_dates=['commit_date'])
netflow_eth = pd.read_csv('../netflow_ETH/netflow_eth.csv')
#withdrawal_value = pd.read_csv('../withrawal_ETH/withdrawal_eth.csv')
withdrawal_freq = pd.read_csv('../withrawal_ETH/withdrawal_frequency.csv')
search_freq = pd.read_csv('../google_searching/search_daily.csv')

# 열 이름 변경
deposit_freq.rename(columns={'Frequency': 'deposit_freq'}, inplace=True)
withdrawal_freq.rename(columns={'Frequency': 'withdrawal_freq'}, inplace=True)
#deposit_value.rename(columns={'value': 'deposit_value'}, inplace=True)
#withdrawal_value.rename(columns={'value': 'withdrawal_value'}, inplace=True)
daily_commits.rename(columns={'commit_date': 'Date', 'count': 'daily_commits'}, inplace=True)
search_freq.rename(columns={'Frequency': 'search_freq'}, inplace=True)
netflow_eth.rename(columns={'value': 'netflow_eth'}, inplace=True)

# UTC 제거 및 데이터 병합 준비
eth_prices['Date'] = eth_prices['Date'].dt.tz_localize(None)
datasets = {
    'deposit_freq': deposit_freq,
    #'deposit_value': deposit_value,
    'daily_commits': daily_commits,
    'netflow_eth': netflow_eth,
    #'withdrawal_eth': withdrawal_value,
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
merged_data.fillna(0, inplace=True)  # 결측값을 0으로 대체
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
    #'deposit_value': [51, 91, 204, 432, 995],
    #'withdrawal_value': [2647, 4215, 5128, 5385, 7385],
    'daily_commits': [1, 3, 5, 7],
    'netflow_eth': None,
    'search_freq': [8, 13.07, 13.21, 15, 15.64]
}

# TLCC를 계산하는 함수
def find_optimal_lag(series1, series2, max_lag=30):
    correlations = []
    for lag in range(1, max_lag + 1):
        shifted_series1 = series1.shift(lag)
        valid_idx = shifted_series1.notna() & series2.notna()

        if valid_idx.sum() > 10:
            corr = shifted_series1[valid_idx].corr(series2[valid_idx])
            correlations.append(corr)
        else:
            correlations.append(np.nan)

    correlations = [c for c in correlations if not np.isnan(c)]
    if not correlations:  # 상관계수가 없으면 None 반환
        return None
    return np.argmax(np.abs(correlations)) + 1


# 시그널 생성 함수
def generate_signal(data, indicator, threshold, lag=None):
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
def backtest(data, thresholds, cash=10000):
    data = data.copy()

    for indicator, threshold in thresholds.items():
        lag = find_optimal_lag(data[indicator], data['Close'])

        if lag is None:  # 상관계수가 유효하지 않으면 조합 생략
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


# 조합 생성
fixed_indicators = ['deposit_freq', 'withdrawal_freq']
other_indicators = [key for key in all_indicators.keys() if key not in fixed_indicators]
valid_combinations = [
    fixed_indicators + list(combination)
    for i in range(0, len(other_indicators) + 1)  # i=0 포함
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
        final_value, buy_sell_dates = backtest(selected_data, threshold_dict)

        if final_value is None:  # 유효하지 않은 조합은 건너뜀
            print(f"Skipping combination: {indicator_combination} with thresholds {threshold_dict} (No valid correlation).")
            continue

        combination_results.append({
            'combination': indicator_combination,
            'thresholds': threshold_dict,
            'portfolio_value': final_value,
            'buy_sell_dates': buy_sell_dates,
            'loss_preservation_ratio': final_value / 10000
        })

# 결과 출력
investor_type = input("Select investor type (aggressive/defensive): ").strip().lower()

try:
    selected_scenario = (
        min(combination_results, key=lambda x: x['loss_preservation_ratio']) if investor_type == 'defensive'
        else max(combination_results, key=lambda x: x['portfolio_value'])
    )
    print(f"\nSelected Strategy: {investor_type.capitalize()}")
    print(f"Selected Combination: {selected_scenario['combination']}")
    print(f"Thresholds: {selected_scenario['thresholds']}")
    print(f"Final Portfolio Value: ${selected_scenario['portfolio_value']:.2f}")
    print(f"Loss Preservation Ratio: {selected_scenario['loss_preservation_ratio']:.4f}")
    print("Buy/Sell Dates:")
    for date, action, price in selected_scenario['buy_sell_dates']:
        print(f"{date} - {action} at ${price:.2f}")
except ValueError as e:
    print(f"Error: {e}")
