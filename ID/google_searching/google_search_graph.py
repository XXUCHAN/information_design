import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('google_search.csv')
# 날짜 형식 변환
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)

# 월별 데이터 집계
monthly_data = data.resample("ME").sum()

# 그래프 생성
plt.figure(figsize=(14, 6))

# 주차별 그래프
plt.subplot(1, 2, 1)
plt.plot(data.index, data["Frequency"], marker='o', color='b')
plt.title("Weekly Google Search Interest for Ethereum")
plt.xlabel("Week")
plt.ylabel("Frequency")
plt.xticks(rotation=45)

# 월별 그래프
plt.subplot(1, 2, 2)
plt.plot(monthly_data.index, monthly_data["Frequency"], marker='o', color='g')
plt.title("Monthly Google Search Interest for Ethereum")
plt.xlabel("Month")
plt.ylabel("Frequency")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
