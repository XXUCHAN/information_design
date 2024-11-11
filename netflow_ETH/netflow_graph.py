import pandas as pd
import matplotlib.pyplot as plt

# 순입출금량 데이터 로드
daily_net_flow_df = pd.read_csv('netflow_eth.csv')
daily_net_flow_df['timeStamp'] = pd.to_datetime(daily_net_flow_df['timeStamp'])

# 시간 순서로 정렬
daily_net_flow_df = daily_net_flow_df.sort_values(by='timeStamp')

# 일별 순입출금량 그래프
plt.figure(figsize=(12, 6))
plt.bar(daily_net_flow_df['timeStamp'], daily_net_flow_df['Net_Flow'], width=5, color='black')
plt.title("Ether Net Flow Over Time (Daily)")
plt.xlabel("Time")
plt.ylabel("Net Flow (ETH)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 주별 순입출금량 집계
weekly_net_flow = daily_net_flow_df.resample('W', on='timeStamp').sum()

# 주별 순입출금량 그래프
plt.figure(figsize=(12, 6))
plt.bar(weekly_net_flow.index, weekly_net_flow['Net_Flow'], width=10, color='gray')
plt.title("Ether Net Flow Over Time (Weekly)")
plt.xlabel("Time")
plt.ylabel("Net Flow (ETH)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 월별 순입출금량 집계
monthly_net_flow = daily_net_flow_df.resample('ME', on='timeStamp').sum()

# 월별 순입출금량 그래프
plt.figure(figsize=(12, 6))
plt.bar(monthly_net_flow.index, monthly_net_flow['Net_Flow'], width=20, color='salmon')
plt.title("Ether Net Flow Over Time (Monthly)")
plt.xlabel("Time")
plt.ylabel("Net Flow (ETH)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
