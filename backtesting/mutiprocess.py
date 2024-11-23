import pandas as pd
import numpy as np
from itertools import product, combinations
from multiprocessing import Pool

# CSV 파일 읽기
print("Reading CSV files...")
eth_prices = pd.read_csv('../ethereum_daily_prices.csv', parse_dates=['Date'])
deposit_freq = pd.read_csv('../deposit_ETH/deposit_frequency.csv')
deposit_value = pd.read_csv('../deposit_ETH/deposit_volume.csv')
daily_commits = pd.read_csv('../github_dev_ETH/daily_commits.csv', parse_dates=['commit_date'])
netflow_eth = pd.read_csv('../netflow_ETH/netflow_eth.csv')
withdrawal_value = pd.read_csv('../withrawal_ETH/withdrawal_eth.csv')
withdrawal_freq = pd.read_csv('../withrawal_ETH/withdrawal_frequency.csv')
google_search = pd.read_csv('../google_searching/search_daily.csv')

# 열 이름 변경
print("Renaming columns...")
deposit_freq.rename(columns={'Frequency': 'deposit_freq'}, inplace=True)
withdrawal_freq.rename(columns={'Frequency': 'withdrawal_freq'}, inplace=True)
deposit_value.rename(columns={'value': 'deposit_value'}, inplace=True)
withdrawal_value.rename(columns={'value': 'withdrawal_value'}, inplace=True)
daily_commits.rename(columns={'commit_date': 'Date', 'count': 'daily_commits'}, inplace=True)
google_search.rename(columns={'Frequency': 'search_freq'}, inplace=True)
netflow_eth.rename(columns={'value': 'netflow_eth'}, inplace=True)

# UTC 제거 및 데이터 병합 준비
print("Processing dates...")
eth_prices['Date'] = eth_prices['Date'].dt.tz_localize(None)
datasets = {
    'deposit_freq': deposit_freq,
    'deposit_value': deposit_value,
    'daily_commits': daily_commits,
    'netflow_eth': netflow_eth,
    'withdrawal_value': withdrawal_value,
    'withdrawal_freq': withdrawal_freq,
    'search_freq': google_search
}

for key, data in datasets.items():
    if 'timeStamp' in data.columns:
        print(f"Processing timestamps for {key}...")
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
merged_data.fillna(0, inplace=True)

# 사용자 입력: 날짜 범위
start_date = pd.to_datetime(input("Enter the start date (YYYY-MM-DD): "))
end_date = pd.to_datetime(input("Enter the end date (YYYY-MM-DD): "))
print(f"Filtering data from {start_date} to {end_date}...")
merged_data = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)]

# 지표별 threshold 범위 설정
all_indicators = {
    'deposit_freq': [0.5, 1, 2, 3, 4],
    'withdrawal_freq': [0.5, 1],
    'deposit_value': [51, 91, 204, 432, 995],
    'withdrawal_value': [2647, 4215, 5128, 5385, 7385],
    'daily_commits': [1, 3, 5, 7],
    'netflow_eth': None,  # 특별 조건: 양수=매수(+1), 음수=매도(-1)
    'search_freq': [8, 13.07, 13.21, 15, 15.64]
}

# 조합 생성 (입금/출금 지표 고정)
fixed_indicators = ['deposit_freq', 'withdrawal_freq']
other_indicators = [key for key in all_indicators.keys() if key not in fixed_indicators]
valid_combinations = [
    fixed_indicators + list(combination)
    for i in range(1, 3)  # 최대 추가 지표 개수 (1~2개)
    for combination in combinations(other_indicators, i)
]

print(f"Total valid combinations: {len(valid_combinations)}")

# 신호 생성 함수
def generate_signal(data, col, threshold):
    if col == 'netflow_eth':
        return data[col].apply(lambda x: 1 if x > 0 else -1)  # 양수=매수, 음수=매도
    elif col in ['withdrawal_freq', 'withdrawal_value', 'search_freq']:
        return data[col].apply(lambda x: -1 if x > threshold else 1)  # 증가=매도, 감소=매수
    else:
        return data[col].apply(lambda x: 1 if x > threshold else -1)  # 증가=매수, 감소=매도

# 백테스팅 함수
def backtest(data, thresholds, cash=10000):
    data = data.copy()
    position = 0
    buy_sell_dates = []

    # 신호 생성
    for col, threshold in thresholds.items():
        data[f'{col}_signal'] = generate_signal(data, col, threshold)

    # 최종 신호 생성
    data['final_signal'] = data[[f'{col}_signal' for col in thresholds]].mean(axis=1).apply(np.sign)

    # 백테스팅 실행
    for _, row in data.iterrows():
        price = row['Close']
        signal = row['final_signal']
        if signal == 1 and cash > 0:  # 매수
            position = cash / price
            cash = 0
            buy_sell_dates.append((row['Date'], 'BUY', price))
        elif signal == -1 and position > 0:  # 매도
            cash = position * price
            position = 0
            buy_sell_dates.append((row['Date'], 'SELL', price))

    # 최종 포트폴리오 가치 계산
    final_value = cash + position * data.iloc[-1]['Close']
    return final_value, buy_sell_dates

# 병렬로 조합 실행
def process_combination(combination):
    selected_indicators = list(combination)
    merged_data_subset = merged_data[['Date', 'Close'] + selected_indicators]
    selected_threshold_ranges = {key: all_indicators[key] for key in selected_indicators if key != 'netflow_eth'}
    threshold_combinations = list(product(*[selected_threshold_ranges[key] for key in selected_indicators if key != 'netflow_eth']))

    results = []
    for thresholds in threshold_combinations:
        threshold_dict = dict(zip([key for key in selected_indicators if key != 'netflow_eth'], thresholds))
        if 'netflow_eth' in selected_indicators:
            threshold_dict['netflow_eth'] = None
        final_value, buy_sell_dates = backtest(merged_data_subset, threshold_dict)
        results.append({
            'indicators': selected_indicators,
            'thresholds': threshold_dict,
            'portfolio_value': final_value,
            'buy_sell_dates': buy_sell_dates
        })
    return results

# 멀티프로세싱 실행
if __name__ == '__main__':
    print("Running backtests in parallel...")
    with Pool(processes=4) as pool:  # 프로세스 개수는 CPU 코어 수에 맞게 조정
        all_results = pool.map(process_combination, valid_combinations)

    # 결과를 평탄화
    all_results = [result for sublist in all_results for result in sublist]

    # 상위 3개 결과 추출
    print("Extracting top 3 results...")
    top_3_results = sorted(all_results, key=lambda x: x['portfolio_value'], reverse=True)[:3]

    # 결과 출력
    for i, result in enumerate(top_3_results, 1):
        print(f"\nTop {i} Result:")
        print(f"Indicators: {result['indicators']}")
        print(f"Thresholds: {result['thresholds']}")
        print(f"Final Portfolio Value: ${result['portfolio_value']:.2f}")
        for date, action, price in result['buy_sell_dates']:
            print(f"{date.date()} - {action} at ${price:.2f}")
