import yfinance as yf
import pandas as pd

# 이더리움의 4년간 일별 가격 데이터 다운로드
eth_data = yf.download('ETH-USD', start='2020-11-08', end='2024-11-09')

# 필요한 열만 선택해서 보기 쉽게 저장
eth_data = eth_data[['Close']]  # 종가 데이터만 필요할 때 사용
eth_data.reset_index(inplace=True)

# 데이터를 CSV 파일로 저장
eth_data.to_csv('ethereum_daily_prices.csv', index=False)
print("Data saved to 'ethereum_daily_prices.csv'")
