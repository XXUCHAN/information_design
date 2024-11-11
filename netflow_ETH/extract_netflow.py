import pandas as pd

# 입출금 데이터 로드
deposit_data = pd.read_csv('../deposit_ETH/deposit_value_eth.csv')
withdrawal_data = pd.read_csv('../withrawal_ETH/withdrawal_eth.csv')

# 날짜 형식 변환
deposit_data['timeStamp'] = pd.to_datetime(deposit_data['timeStamp'])
withdrawal_data['timeStamp'] = pd.to_datetime(withdrawal_data['timeStamp'])

# 일별 데이터로 집계 (합계)
daily_deposit = deposit_data.resample('D', on='timeStamp').sum()
daily_withdrawal = withdrawal_data.resample('D', on='timeStamp').sum()

# 일별 순입출금량 계산
daily_net_flow = daily_deposit['value'] - daily_withdrawal['value']
daily_net_flow = daily_net_flow.rename('Net_Flow')

# 결과를 DataFrame으로 변환
daily_net_flow_df = daily_net_flow.reset_index()

# 결과를 CSV 파일로 저장
daily_net_flow_df.to_csv('./netflow_eth.csv', index=False)

# 확인
print(daily_net_flow_df.head())
