import pandas as pd

# CSV 파일 읽기
eth_prices = pd.read_csv('../ethereum_daily_prices.csv', parse_dates=['Date'])
daily_commits = pd.read_csv('daily_commits.csv', parse_dates=['commit_date'])

# 'commit_date'를 'Date'로 이름 변경하여 병합 준비
daily_commits.rename(columns={'commit_date': 'Date'}, inplace=True)

# 날짜별 데이터 병합 (inner join을 사용하여 두 파일에 모두 존재하는 날짜만 포함)
merged_data = pd.merge(eth_prices, daily_commits, on='Date', how='inner')

# 특정 기간 필터링
start_date = '2021-08-04' #사용자 입력
end_date = '2024-08-04' #사용자 입력
merged_data_input = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)].copy()


# TLCC 계산 함수
def time_lagged_cross_correlation(series1, series2, max_lag):
    correlations = []
    for lag in range(1, max_lag + 1):  # lag를 0 이상만 계산
        if lag == 0:
            corr = series1.corr(series2)
        else:
            corr = series1[lag:].corr(series2[:-lag])
        correlations.append((lag, corr))  # (lag, correlation) 형태로 저장
    return correlations

# TLCC 결과 계산 (lag가 0 이상인 값만 계산)
correlations = time_lagged_cross_correlation(merged_data_input['count'], merged_data_input['Close'], max_lag=30)

# lag가 0 이상인 값들 중에서 최대 상관계수의 lag 찾기
optimal_lag = max(correlations, key=lambda x: abs(x[1]))[0]
print(f"Optimal lag (excluding negative lags): {optimal_lag}")

# 초기 자본 설정
initial_cash = 10000

# 다양한 threshold 값 테스트
threshold_values = [0.5, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10]  # 테스트할 threshold 값들
best_threshold = None
best_final_value = 0

# 각 threshold에 대해 백테스팅 수행
for threshold in threshold_values:
    cash = initial_cash
    position = 0  # 현재 보유한 이더리움 수량

    # 매수/매도 신호 생성: 커밋 수가 threshold 이상일 때 매수, 이하일 때 매도
    merged_data_input.loc[:, 'signal'] = merged_data_input['count'].shift(optimal_lag).apply(lambda x: 'buy' if x > threshold else 'sell')

# 백테스팅 수행
    for idx, row in merged_data_input.iterrows():
        price = row['Close']
        signal = row['signal']
        if signal == 'buy' and cash > 0:
            # 매수
            position = cash / price
            cash = 0
        elif signal == 'sell' and position > 0:
            # 매도
            cash = position * price
            position = 0

    # 최종 포트폴리오 가치 계산
    final_value = cash + position * merged_data_input.iloc[-1]['Close']
    total_return = ((final_value / initial_cash) - 1) * 100

    print(f"Threshold: {threshold}, Final Portfolio Value: ${final_value:.2f}, Total Return: {total_return:.2f}%")

    # 최적 threshold 업데이트
    if final_value > best_final_value:
        best_final_value = final_value
        best_threshold = threshold

# 최적 threshold 결과 출력
print(f"\nBest Threshold for 2024: {best_threshold}, Highest Final Portfolio Value: ${best_final_value:.2f}")
