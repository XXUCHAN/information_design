import pandas as pd
import numpy as np
from itertools import product

# CSV 파일 읽기 (업로드된 파일 경로에 맞게 수정)
eth_prices = pd.read_csv('../ethereum_daily_prices.csv', parse_dates=['Date'])
deposit_freq = pd.read_csv('../deposit_ETH/deposit_frequency.csv')
deposit_value = pd.read_csv('../deposit_ETH/deposit_value_eth.csv')
daily_commits = pd.read_csv('../github_dev_ETH/daily_commits.csv', parse_dates=['commit_date'])
netflow_eth = pd.read_csv('../netflow_ETH/netflow_eth.csv')
withdrawal_value = pd.read_csv('../withrawal_ETH/withdrawal_eth.csv')
withdrawal_freq = pd.read_csv('../withrawal_ETH/withdrawal_frequency.csv')

# 열 이름 변경
deposit_freq.rename(columns={'Frequency': 'deposit_freq'}, inplace=True)
withdrawal_freq.rename(columns={'Frequency': 'withdrawal_freq'}, inplace=True)
deposit_value.rename(columns={'value': 'deposit_value'}, inplace=True)
withdrawal_value.rename(columns={'value': 'withdrawal_value'}, inplace=True)

# 'commit_date'를 'Date'로 이름 변경하여 병합 준비
daily_commits.rename(columns={'commit_date': 'Date'}, inplace=True)

# UTC 제거 및 데이터 병합을 위한 timestamp 변환 처리
eth_prices['Date'] = eth_prices['Date'].dt.tz_localize(None)

# 각 파일에서 'Date' 또는 'timestamp' 열이 있는지 확인 후 처리
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
    else:
        print(f"Warning: '{key}' 데이터에 'Date'나 'timeStamp' 열이 없습니다.")
        continue  # Date와 timeStamp 열이 없으면 병합에서 제외

# 가격 데이터를 기준으로 병합
merged_data = eth_prices
for key, data in datasets.items():
    merged_data = pd.merge(merged_data, data, on='Date', how='left')

# 결측값을 0으로 채움
merged_data.fillna(0, inplace=True)

# 날짜 입력
start_date = pd.to_datetime(input("Enter the start date (YYYY-MM-DD): "))
end_date = pd.to_datetime(input("Enter the end date (YYYY-MM-DD): "))

# 지정한 날짜 범위로 데이터 필터링
merged_data = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)]

# 선택한 지표 및 백테스팅 수행
selected_indicators = ['deposit_freq', 'withdrawal_freq']

# 선택한 지표만 포함하도록 데이터 필터링
merged_data = merged_data[['Date', 'Close'] + selected_indicators]

# 최적의 lag를 찾는 함수
def find_optimal_lag(series1, series2, max_lag=30):
    correlations = [series1.shift(lag).corr(series2) for lag in range(1, max_lag + 1)]
    return np.argmax(np.abs(correlations)) + 1

# 각 선택된 지표에 대해 최적 lag 계산
optimal_lags = {col: find_optimal_lag(merged_data[col], merged_data['Close']) for col in selected_indicators}

# 최적의 lag 출력
print("\nOptimal Lags for each indicator:")
for indicator, lag in optimal_lags.items():
    print(f"{indicator}: {lag}")

# threshold 범위 설정
threshold_ranges = {'deposit_freq': [0.5, 1, 2, 3, 4], 'withdrawal_freq': [0.5, 1]}
selected_threshold_ranges = {key: threshold_ranges[key] for key in selected_indicators}
threshold_combinations = list(product(*selected_threshold_ranges.values()))

# 백테스팅 함수 정의 (정방향)
def backtest(data, thresholds, cash=10000):
    data = data.copy()
    position = 0
    buy_sell_dates = []

    # 입금 지표를 기준으로 매수 신호 생성
    primary_threshold = thresholds['deposit_freq']
    data['primary_signal'] = data['deposit_freq'].shift(optimal_lags['deposit_freq']).fillna(0).apply(
        lambda x: 1 if x > primary_threshold else -1
    )

    # 출금 지표를 기준으로 매도 신호 생성
    secondary_threshold = thresholds['withdrawal_freq']
    data['secondary_signal'] = data['withdrawal_freq'].shift(optimal_lags['withdrawal_freq']).fillna(0).apply(
        lambda x: -1 if x > secondary_threshold else 1
    )

    # 최종 신호 생성
    data['final_signal'] = data.apply(
        lambda row: row['primary_signal'] if row['primary_signal'] == row['secondary_signal'] else 0, axis=1
    )

    # 백테스팅 실행
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

# 백테스팅 함수 정의 (역방향)
def backtest_reverse(data, thresholds, cash=10000):
    data = data.copy()
    position = 0
    buy_sell_dates = []

    # 데이터를 최신 날짜에서 과거 날짜 순으로 정렬
    data = data.sort_values(by='Date', ascending=False).reset_index(drop=True)

    # 입금 지표를 기준으로 매수 신호 생성
    primary_threshold = thresholds['deposit_freq']
    data['primary_signal'] = data['deposit_freq'].shift(-optimal_lags['deposit_freq']).fillna(0).apply(
        lambda x: 1 if x > primary_threshold else -1
    )

    # 출금 지표를 기준으로 매도 신호 생성
    secondary_threshold = thresholds['withdrawal_freq']
    data['secondary_signal'] = data['withdrawal_freq'].shift(-optimal_lags['withdrawal_freq']).fillna(0).apply(
        lambda x: -1 if x > secondary_threshold else 1
    )

    # 최종 신호 생성
    data['final_signal'] = data.apply(
        lambda row: row['primary_signal'] if row['primary_signal'] == row['secondary_signal'] else 0, axis=1
    )

    # 백테스팅 실행
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

# 최적 threshold 및 포트폴리오 결과 찾기
best_thresholds = {}
best_final_value = 0
best_buy_sell_dates = []

print("\nThreshold Scenarios and Portfolio Values (Forward):")
for thresholds in threshold_combinations:
    threshold_dict = dict(zip(selected_threshold_ranges.keys(), thresholds))
    final_value, buy_sell_dates = backtest(merged_data, threshold_dict)
    print(f"Thresholds: {threshold_dict} -> Final Portfolio Value: ${final_value:.2f}")

    if final_value > best_final_value:
        best_final_value = final_value
        best_thresholds = threshold_dict
        best_buy_sell_dates = buy_sell_dates
print("\n",best_buy_sell_dates)
print("\nBest Thresholds for Maximum Portfolio Value (Forward):")
print(f"Best Thresholds: {best_thresholds}")
print(f"Highest Final Portfolio Value: ${best_final_value:.2f}")

# 역방향 최적 threshold 및 포트폴리오 결과 찾기
best_thresholds_reverse = {}
best_final_value_reverse = 0
best_buy_sell_dates_reverse = []

print("\nThreshold Scenarios and Portfolio Values (Reverse):")
for thresholds in threshold_combinations:
    threshold_dict = dict(zip(selected_threshold_ranges.keys(), thresholds))
    final_value, buy_sell_dates = backtest_reverse(merged_data, threshold_dict)
    print(f"Thresholds: {threshold_dict} -> Final Portfolio Value: ${final_value:.2f}")

    if final_value > best_final_value_reverse:
        best_final_value_reverse = final_value
        best_thresholds_reverse = threshold_dict
        best_buy_sell_dates_reverse = buy_sell_dates

print("\nBest Thresholds for Maximum Portfolio Value (Reverse):")
print(f"Best Thresholds: {best_thresholds_reverse}")
print(f"Highest Final Portfolio Value: ${best_final_value_reverse:.2f}")

# 매수/매도 날짜 출력
print("\nInvestment Scenario with Buy/Sell Dates (Reverse):")
for date, action, price in best_buy_sell_dates_reverse:
    print(f"{date.date()} - {action} at ${price:.2f}")

# Hold 전략의 수익률 계산
initial_price = merged_data.iloc[0]['Close']  # 첫날의 가격
final_price = merged_data.iloc[-1]['Close']  # 마지막 날의 가격
hold_return = (final_price - initial_price) / initial_price * 100  # 수익률(%)
# Hold 전략
print(f"Hold Strategy {start_date}-{initial_price:.2f} ~ {end_date}-{final_price:.2f} : {hold_return:.2f}%")

#손실보존률 계산
def calculate_loss_preservation(data, signals, initial_cash=10000):
    """
    Calculate the loss preservation ratio for a given strategy.
    """
    # Identify market downtrend periods
    data['Return'] = data['Close'].pct_change()
    downtrend = data['Return'] < 0  # 하락 구간 식별

    # Ensure indices match
    data = data.reset_index(drop=True)
    downtrend = downtrend.reset_index(drop=True)

    # Combine strategy signals with data
    data = data.merge(signals[['Date', 'final_signal']], on='Date', how='left')

    # Market loss calculation
    market_down_close = data.loc[downtrend, 'Close']
    if not market_down_close.empty:
        market_loss = (market_down_close.iloc[-1] - market_down_close.iloc[0]) / market_down_close.iloc[0]
    else:
        market_loss = 0

    # Strategy loss calculation
    cash = initial_cash
    position = 0
    for _, row in data.iterrows():
        price = row['Close']
        signal = row['final_signal']
        if signal == 1 and cash > 0:  # BUY
            position = cash / price
            cash = 0
        elif signal == -1 and position > 0:  # SELL
            cash = position * price
            position = 0

    final_value = cash + position * data.iloc[-1]['Close']
    strategy_loss = (final_value - initial_cash) / initial_cash

    # Loss preservation ratio
    if market_loss != 0:
        loss_preservation_ratio = strategy_loss / market_loss
    else:
        loss_preservation_ratio = None  # 시장 손실이 없으면 비율 계산 불가

    return loss_preservation_ratio

# 순방향 백테스팅 후 손실보존률 계산
forward_signals = pd.DataFrame(best_buy_sell_dates, columns=['Date', 'Action', 'Price'])
forward_signals['final_signal'] = forward_signals['Action'].apply(lambda x: 1 if x == 'BUY' else -1)
forward_loss_preservation = calculate_loss_preservation(merged_data, forward_signals)
print(f"Loss Preservation Ratio (Forward): {forward_loss_preservation}")

# 역방향 백테스팅 후 손실보존률 계산
reverse_signals = pd.DataFrame(best_buy_sell_dates_reverse, columns=['Date', 'Action', 'Price'])
reverse_signals['final_signal'] = reverse_signals['Action'].apply(lambda x: 1 if x == 'BUY' else -1)
reverse_loss_preservation = calculate_loss_preservation(merged_data, reverse_signals)
print(f"Loss Preservation Ratio (Reverse): {reverse_loss_preservation}")