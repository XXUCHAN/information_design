import matplotlib.pyplot as plt
import pandas as pd

# 가상 데이터 생성
data = {
    'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    'Close': [100, 105, 102, 110, 120, 115, 117, 125, 130, 128],
    'deposit_freq': [0.4, 0.6, 1.0, 0.8, 1.2, 0.9, 0.7, 1.1, 0.5, 0.3],
    'withdrawal_freq': [0.3, 0.2, 0.1, 0.4, 0.5, 0.8, 0.7, 0.6, 0.9, 1.0]
}
df = pd.DataFrame(data)

# 데이터를 최신 날짜부터 과거 날짜 순으로 정렬
df = df.sort_values(by='Date', ascending=False).reset_index(drop=True)

# 역방향 신호 생성
thresholds = {'deposit_freq': 0.7, 'withdrawal_freq': 0.5}
df['primary_signal'] = df['deposit_freq'].shift(-1).fillna(0).apply(
    lambda x: 1 if x > thresholds['deposit_freq'] else -1
)
df['secondary_signal'] = df['withdrawal_freq'].shift(-1).fillna(0).apply(
    lambda x: -1 if x > thresholds['withdrawal_freq'] else 1
)

# 최종 신호 생성
df['final_signal'] = df.apply(
    lambda row: row['primary_signal'] if row['primary_signal'] == row['secondary_signal'] else 0, axis=1
)

# 백테스팅 시뮬레이션
cash = 10000
position = 0
buy_sell_dates = []

for _, row in df.iterrows():
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

final_value = cash + position * df.iloc[-1]['Close']

# 결과 출력
print("Buy/Sell Dates and Prices:")
for date, action, price in buy_sell_dates:
    print(f"{date.date()} - {action} at ${price:.2f}")

print(f"Final Portfolio Value: ${final_value:.2f}")

# 그래프를 통해 역방향 백테스팅 결과를 시각화
plt.figure(figsize=(12, 6))

# Close 가격 추세 그리기
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue', alpha=0.7)

# 매수/매도 시점 표시
buy_dates = [date for date, action, price in buy_sell_dates if action == 'BUY']
buy_prices = [price for date, action, price in buy_sell_dates if action == 'BUY']
sell_dates = [date for date, action, price in buy_sell_dates if action == 'SELL']
sell_prices = [price for date, action, price in buy_sell_dates if action == 'SELL']

plt.scatter(buy_dates, buy_prices, color='green', label='BUY Signal', marker='^', s=100, alpha=0.8)
plt.scatter(sell_dates, sell_prices, color='red', label='SELL Signal', marker='v', s=100, alpha=0.8)

# 그래프 제목 및 축 레이블
plt.title('Reverse Backtesting: Buy/Sell Signals on Close Prices', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price', fontsize=12)

# 범례 추가
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# 그래프 출력
plt.show()
