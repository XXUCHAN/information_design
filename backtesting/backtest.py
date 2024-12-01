import pandas as pd
import numpy as np
from itertools import product

# CSV 파일 읽기 (업로드된 파일 경로에 맞게 수정)
eth_prices = pd.read_csv('../ethereum_daily_prices.csv', parse_dates=['Date'])
deposit_freq = pd.read_csv('../deposit_ETH/deposit_frequency.csv')
deposit_value = pd.read_csv('../deposit_ETH/deposit_volume.csv')
daily_commits = pd.read_csv('../github_dev_ETH/daily_commits.csv', parse_dates=['commit_date'])
netflow_eth = pd.read_csv('../netflow_ETH/netflow_eth.csv')
withdrawal_value = pd.read_csv('../withdrawal_ETH/withdrawal_eth.csv')
withdrawal_freq = pd.read_csv('../withdrawal_ETH/withdrawal_frequency.csv')

# 열 이름 변경
deposit_freq.rename(columns={'Frequency': 'deposit_freq'}, inplace=True)
withdrawal_freq.rename(columns={'Frequency': 'withdrawal_freq'}, inplace=True)
deposit_value.rename(columns={'value': 'deposit_value'}, inplace=True)
withdrawal_value.rename(columns={'value': 'withdrawal_value'}, inplace=True)
daily_commits.rename(columns={'commit_date': 'Date'}, inplace=True)

# UTC 제거 및 데이터 병합 준비
eth_prices['Date'] = eth_prices['Date'].dt.tz_localize(None)
datasets = {
    'deposit_freq': deposit_freq,
    'deposit_value': deposit_value,
    'daily_commits': daily_commits,
    'netflow_eth': netflow_eth,
    'withdrawal_eth': withdrawal_value,
    'withdrawal_freq': withdrawal_freq
}

for key, data in datasets.items():
    if 'timeStamp' in data.columns:
        data['Date'] = pd.to_datetime(data['timeStamp'], errors='coerce').dt.tz_localize('UTC').dt.tz_convert(None)
        data.drop(columns=['timeStamp'], inplace=True)
    elif 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce').dt.tz_localize(None)

# 데이터 병합
merged_data = eth_prices
for key, data in datasets.items():
    merged_data = pd.merge(merged_data, data, on='Date', how='left')
merged_data.fillna(0, inplace=True)

# 사용자 입력: 날짜 범위
start_date = pd.to_datetime(input("Enter the start date (YYYY-MM-DD): "))
end_date = pd.to_datetime(input("Enter the end date (YYYY-MM-DD): "))
merged_data = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)]

# 선택한 지표
selected_indicators = ['deposit_freq', 'withdrawal_freq']
merged_data = merged_data[['Date', 'Close'] + selected_indicators]

# 최적의 lag 계산 함수
def find_optimal_lag(series1, series2, max_lag=30):
    correlations = [series1.shift(lag).corr(series2) for lag in range(1, max_lag + 1)]
    return np.argmax(np.abs(correlations)) + 1

optimal_lags = {col: find_optimal_lag(merged_data[col], merged_data['Close']) for col in selected_indicators}

# threshold 범위 설정
threshold_ranges = {'deposit_freq': [0.5, 1, 2, 3, 4], 'withdrawal_freq': [0.5, 1]}
threshold_combinations = list(product(*[threshold_ranges[key] for key in selected_indicators]))

# 백테스팅 함수 (순방향)
def backtest(data, thresholds, cash=10000):
    data = data.copy()
    position = 0
    buy_sell_dates = []
    primary_threshold = thresholds['deposit_freq']
    secondary_threshold = thresholds['withdrawal_freq']

    data['primary_signal'] = data['deposit_freq'].shift(optimal_lags['deposit_freq']).fillna(0).apply(
        lambda x: 1 if x > primary_threshold else -1
    )
    data['secondary_signal'] = data['withdrawal_freq'].shift(optimal_lags['withdrawal_freq']).fillna(0).apply(
        lambda x: -1 if x > secondary_threshold else 1
    )
    data['final_signal'] = data.apply(
        lambda row: row['primary_signal'] if row['primary_signal'] == row['secondary_signal'] else 0, axis=1
    )

    for _, row in data.iterrows():
        price = row['Close']
        signal = row['final_signal']
        if signal == 1 and cash > 0:
            position = cash / price
            cash = 0
            buy_sell_dates.append((row['Date'], 'BUY', price))
        elif signal == -1 and position > 0:
            cash = position * price
            position = 0
            buy_sell_dates.append((row['Date'], 'SELL', price))

    final_value = cash + position * data.iloc[-1]['Close']
    return final_value, buy_sell_dates

# 손실보존률 계산
def calculate_loss_preservation(data, signals, initial_cash=10000):
    data['Return'] = data['Close'].pct_change()
    downtrend = data['Return'] < 0
    data = data.reset_index(drop=True)
    downtrend = downtrend.reset_index(drop=True)

    data = data.merge(signals[['Date', 'final_signal']], on='Date', how='left')
    market_down_close = data.loc[downtrend, 'Close']
    market_loss = (market_down_close.iloc[-1] - market_down_close.iloc[0]) / market_down_close.iloc[0] if not market_down_close.empty else 0

    cash = initial_cash
    position = 0
    for _, row in data.iterrows():
        price = row['Close']
        signal = row['final_signal']
        if signal == 1 and cash > 0:
            position = cash / price
            cash = 0
        elif signal == -1 and position > 0:
            cash = position * price
            position = 0

    final_value = cash + position * data.iloc[-1]['Close']
    strategy_loss = (final_value - initial_cash) / initial_cash
    return strategy_loss / market_loss if market_loss != 0 else None

# 순방향 백테스팅 및 결과 저장
forward_results = []
for thresholds in threshold_combinations:
    threshold_dict = dict(zip(selected_indicators, thresholds))
    final_value, buy_sell_dates = backtest(merged_data, threshold_dict)
    signals = pd.DataFrame(buy_sell_dates, columns=['Date', 'Action', 'Price'])
    signals['final_signal'] = signals['Action'].apply(lambda x: 1 if x == 'BUY' else -1)
    loss_preservation = calculate_loss_preservation(merged_data, signals)

    forward_results.append({
        'thresholds': threshold_dict,
        'portfolio_value': final_value,
        'loss_preservation_ratio': loss_preservation,
        'buy_sell_dates': buy_sell_dates
    })
# 투자자 성향에 따른 시나리오 선택
def select_thresholds(results, investor_type, initial_cash=10000):
    valid_results = [result for result in results if result['portfolio_value'] > initial_cash]
    if not valid_results:
        raise ValueError("No valid scenarios with portfolio value greater than the initial investment.")
    if investor_type == "defensive":
        return min(valid_results, key=lambda x: x['loss_preservation_ratio'])
    elif investor_type == "aggressive":
        return max(valid_results, key=lambda x: x['loss_preservation_ratio'])
    else:
        raise ValueError("Invalid investor type. Choose 'defensive' or 'aggressive'.")

investor_type = input("Select investor type (aggressive/defensive): ").strip().lower()
selected_scenario = select_thresholds(forward_results, investor_type)
selected_thresholds = selected_scenario['thresholds']
selected_buy_sell_dates = selected_scenario['buy_sell_dates']
final_portfolio_value = selected_scenario['portfolio_value']
selected_loss_preservation_ratio = selected_scenario['loss_preservation_ratio']

# Hold 전략 수익률 계산
initial_price = merged_data.iloc[0]['Close']
final_price = merged_data.iloc[-1]['Close']
hold_return = 10000 + (final_price - initial_price)

# 결과 출력
print(f"\nInvestor Type: {investor_type.capitalize()}")
print(f"Selected Thresholds: {selected_thresholds}")
print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
print(f"Loss Preservation Ratio: {selected_loss_preservation_ratio:.4f}")
print(f"Buy/Sell Dates:")
for date, action, price in selected_buy_sell_dates:
    print(f"{date.date()} - {action} at ${price:.2f}")

print(f"\nHold Strategy: ${hold_return:.2f}")