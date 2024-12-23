import pandas as pd

# 시작 및 종료 날짜 설정
start_date = "2020-11-07"
end_date = "2024-11-09"

# 입출금 데이터 로드
deposit_data = pd.read_csv('../deposit_ETH/deposit_volume.csv')
withdrawal_data = pd.read_csv('../withdrawal_ETH/withdrawal_eth.csv')

# 날짜 형식 변환
deposit_data['timeStamp'] = pd.to_datetime(deposit_data['timeStamp'])
withdrawal_data['timeStamp'] = pd.to_datetime(withdrawal_data['timeStamp'])

# 필터링: 지정된 날짜 범위 내 데이터만 선택
deposit_data = deposit_data[(deposit_data['timeStamp'] >= start_date) & (deposit_data['timeStamp'] <= end_date)]
withdrawal_data = withdrawal_data[(withdrawal_data['timeStamp'] >= start_date) & (withdrawal_data['timeStamp'] <= end_date)]

# 일별 데이터로 집계 (합계)
daily_deposit = deposit_data.resample('D', on='timeStamp')['value'].sum()
daily_withdrawal = withdrawal_data.resample('D', on='timeStamp')['value'].sum()

# NaN 값을 0으로 채우기
daily_deposit = daily_deposit.fillna(0)
daily_withdrawal = daily_withdrawal.fillna(0)

# 일별 순입출금량 계산
daily_net_flow = daily_deposit - daily_withdrawal
daily_net_flow = daily_net_flow.rename('value')

# 결과를 DataFrame으로 변환
daily_net_flow_df = daily_net_flow.reset_index()

# 필터링된 날짜 범위만 포함
daily_net_flow_df = daily_net_flow_df[(daily_net_flow_df['timeStamp'] >= start_date) & (daily_net_flow_df['timeStamp'] <= end_date)]

# 결과를 CSV 파일로 저장
output_path = './netflow_eth.csv'
daily_net_flow_df.to_csv(output_path, index=False)

# 확인
print(f"Processed data saved to {output_path}")
print(daily_net_flow_df.head())
