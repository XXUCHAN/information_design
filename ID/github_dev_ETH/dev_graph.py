import pandas as pd
import matplotlib.pyplot as plt

# 일별 데이터 로드
daily_commits = pd.read_csv('daily_commits.csv', parse_dates=['commit_date'], index_col='commit_date')

# 주별 및 월별 데이터 집계
weekly_commits = daily_commits.resample('W').sum()
monthly_commits = daily_commits.resample('ME').sum()

# 그래프 그리기
plt.figure(figsize=(12, 12))

# 일별 그래프
plt.subplot(3, 1, 1)
plt.plot(daily_commits.index, daily_commits['count'], color='skyblue')
plt.title('GitHub Commit Activity (Daily)')
plt.xlabel('Date')
plt.ylabel('Commits')

# 주별 그래프
plt.subplot(3, 1, 2)
plt.plot(weekly_commits.index, weekly_commits['count'], color='green')
plt.title('GitHub Commit Activity (Weekly)')
plt.xlabel('Date')
plt.ylabel('Commits')

# 월별 그래프
plt.subplot(3, 1, 3)
plt.plot(monthly_commits.index, monthly_commits['count'], color='salmon')
plt.title('GitHub Commit Activity (Monthly)')
plt.xlabel('Date')
plt.ylabel('Commits')

plt.tight_layout()
plt.show()
