import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 필터링된 데이터 로드
data = pd.read_csv('deposit_volume.csv')

# timeStamp를 datetime 형식으로 변환
data['timeStamp'] = pd.to_datetime(data['timeStamp'])

# 시간 순서로 정렬
data = data.sort_values(by='timeStamp')

# 로그 스케일 적용
data['value'] = np.log10(data['value'])

# 일별 그래프
plt.figure(figsize=(12, 6))
plt.bar(data['timeStamp'], data['value'], width=1, color='skyblue')
plt.title("Ether Value Over Time (Daily, Log Scale)")
plt.xlabel("Time")
plt.ylabel("Value (ETH, Log Scale)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 주별로 값 합산
weekly_data = data.resample('W', on='timeStamp').sum()

# 주별 그래프
plt.figure(figsize=(12, 6))
plt.bar(weekly_data.index, weekly_data['value'], width=5, color='lightgreen')
plt.title("Ether Value Over Time (Weekly, Log Scale)")
plt.xlabel("Time")
plt.ylabel("Value (ETH, Log Scale)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 월별로 값 합산
monthly_data = data.resample('ME', on='timeStamp').sum()

# 월별 그래프
plt.figure(figsize=(12, 6))
plt.bar(monthly_data.index, monthly_data['value'], width=20, color='salmon')
plt.title("Ether Value Over Time (Monthly, Log Scale)")
plt.xlabel("Time")
plt.ylabel("Value (ETH, Log Scale)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
