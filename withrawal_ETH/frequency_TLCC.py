import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기 (사용자의 파일 경로로 수정)
eth_prices = pd.read_csv('../ethereum_daily_prices.csv', parse_dates=['Date'])
withdrawal_data = pd.read_csv('withdrawal_frequency.csv', parse_dates=['Date'])

# 두 데이터프레임의 Date 열을 동일한 형식으로 변환 (UTC 제거)
eth_prices['Date'] = eth_prices['Date'].dt.tz_localize(None)
withdrawal_data['Date'] = withdrawal_data['Date'].dt.tz_localize(None)

# 날짜별 데이터 병합 (이더리움 가격 데이터를 기준으로 left join 사용)
merged_data = pd.merge(eth_prices, withdrawal_data, on='Date', how='left')

# withdrawal 데이터에 없는 날짜는 0으로 채움
merged_data['Frequency'] = merged_data['Frequency'].fillna(0)

# 사용자로부터 시작일과 종료일 입력받기
start_date = '2021-08-04'  # 예시 시작일
end_date = '2024-08-04'    # 예시 종료일

# 시작일과 종료일을 기준으로 데이터 필터링
merged_data = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)].copy()

# TLCC 계산 함수
def time_lagged_cross_correlation(series1, series2, max_lag):
    correlations = []
    for lag in range(0, max_lag + 1):  # lag를 0 이상만 계산
        corr = series1[lag:].corr(series2[:-lag]) if lag > 0 else series1.corr(series2)
        correlations.append((lag, corr))  # (lag, correlation) 형태로 저장
    return correlations

# TLCC 결과 계산
max_lag = 30  # 최대 지연 기간 설정
correlations = time_lagged_cross_correlation(merged_data['Frequency'], merged_data['Close'], max_lag=max_lag)

# lag가 0 이상인 값들 중에서 최대 상관계수의 lag 찾기
optimal_lag = max(correlations, key=lambda x: abs(x[1]))[0]
print(f"Optimal lag: {optimal_lag}")

# 초기 자본 설정
initial_cash = 10000

# 다양한 threshold 값 테스트
threshold_values = [0.5, 1, 2]  # 테스트할 threshold 값들
best_threshold = None
best_final_value = 0

# 각 threshold에 대해 백테스팅 수행
for threshold in threshold_values:
    cash = initial_cash
    position = 0  # 현재 보유한 이더리움 수량

    # 매수/매도 신호 생성: withdrawal 빈도가 threshold 이상일 때 매도, 미만일 때 매수
    merged_data['signal'] = merged_data['Frequency'].apply(lambda x: 'sell' if x > threshold else 'buy')

    # optimal_lag에 따라 신호를 지연시킴
    merged_data['signal'] = merged_data['signal'].shift(optimal_lag)

    # 백테스팅 수행
    for idx, row in merged_data.iterrows():
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
    final_value = cash + position * merged_data.iloc[-1]['Close']
    total_return = ((final_value / initial_cash) - 1) * 100

    print(f"Threshold: {threshold}, Final Portfolio Value: ${final_value:.2f}, Total Return: {total_return:.2f}%")

    # 최적 threshold 업데이트
    if final_value > best_final_value:
        best_final_value = final_value
        best_threshold = threshold

# 최적 threshold 결과 출력
print(f"\nBest Threshold: {best_threshold}, Highest Final Portfolio Value: ${best_final_value:.2f}")