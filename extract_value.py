import pandas as pd

# 데이터 로드
data = pd.read_csv('withrawal_ETH/withdrawal_transactions.csv')

# value 열을 숫자 형식으로 변환 후 Ether 단위로 변환
data['value'] = pd.to_numeric(data['value'], errors='coerce')  # 문자열을 숫자로 변환 (변환 불가 시 NaN 처리)
data['value'] = data['value'] / 10**18  # Wei를 Ether 단위로 변환

# timeStamp를 사람이 읽을 수 있는 형식으로 변환
data['timeStamp'] = pd.to_datetime(data['timeStamp'], unit='s')

# value가 1 Ether보다 큰 데이터만 선택
filtered_data = data[data['value'] > 1]

# 필요한 열만 선택
output_data = filtered_data[['timeStamp', 'value']]

output_data.to_csv('./withdrawal_eth.csv', index=False)

print("Timestamp and value (in ETH) saved to /mnt/data/timestamp_value_eth.csv")
