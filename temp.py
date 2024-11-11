import pandas as pd

# CSV 파일 읽기
eth_prices = pd.read_csv('ethereum_daily_prices.csv', parse_dates=['Date'])

# Date 열을 인덱스로 설정
eth_prices.set_index('Date', inplace=True)

# 주 단위로 리샘플링하여 매주 일요일 기준으로 주간 평균 생성
eth_weekly = eth_prices['Close'].resample('W-SUN').mean().reset_index()

# 결과 확인
print(eth_weekly.head())

# 변환된 주간 데이터를 새로운 CSV 파일로 저장
eth_weekly.to_csv('ethereum_weekly_prices.csv', index=False)
