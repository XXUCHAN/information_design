import pandas as pd
import matplotlib.pyplot as plt

# 필터링된 데이터 로드
data = pd.read_csv('withdrawal_frequency.csv')

# Date를 datetime 형식으로 변환
data['Date'] = pd.to_datetime(data['Date'])

# 시간 순서로 정렬
data = data.sort_values(by='Date')

# 일별 빈도수 그래프
plt.figure(figsize=(12, 6))
plt.bar(data['Date'], data['Frequency'], width=5, color='skyblue')
plt.title("Ether Deposit Frequency (Daily)")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 주별 빈도수 집계
weekly_data = data.resample('W', on='Date').sum()

# 주별 빈도수 그래프
plt.figure(figsize=(12, 6))
plt.bar(weekly_data.index, weekly_data['Frequency'], width=10, color='lightgreen')
plt.title("Ether Deposit Frequency (Weekly)")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 월별 빈도수 집계
monthly_data = data.resample('ME', on='Date').sum()

# 월별 빈도수 그래프
plt.figure(figsize=(12, 6))
plt.bar(monthly_data.index, monthly_data['Frequency'], width=20, color='salmon')
plt.title("Ether Deposit Frequency (Monthly)")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
