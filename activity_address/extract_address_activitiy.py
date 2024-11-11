import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv
load_dotenv()
etherscan_api = os.getenv("ETH_API")
# Etherscan API 키
API_KEY = etherscan_api
# 시작일과 종료일 설정
start_date = datetime(2020, 11, 8)
end_date = datetime(2024, 11, 8)
delta = timedelta(days=1)

# 결과를 저장할 딕셔너리
daily_active_wallets = {}

# 블록 넘버를 얻는 함수
def get_block_number(timestamp):
    url = f'https://api.etherscan.io/api?module=block&action=getblocknobytime&timestamp={timestamp}&closest=before&apikey={API_KEY}'
    response = requests.get(url).json()
    if response['status'] == '1':
        return int(response['result'])
    else:
        return None

# 각 날짜에 대해 활성 지갑 계산
while start_date <= end_date:
    date_str = start_date.strftime('%Y-%m-%d')
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int((start_date + timedelta(days=1)).timestamp())

    # 각 날짜의 시작 및 끝 블록 넘버를 가져옴
    start_block = get_block_number(start_timestamp)
    end_block = get_block_number(end_timestamp)

    if start_block is None or end_block is None:
        print(f"{date_str}: Failed to retrieve block numbers.")
        daily_active_wallets[date_str] = None
        start_date += delta
        continue

    url = f'https://api.etherscan.io/api?module=account&action=txlist&startblock={start_block}&endblock={end_block}&sort=asc&apikey={API_KEY}'

    # API 호출 재시도를 위한 설정
    for attempt in range(3):  # 최대 3번 시도
        try:
            response = requests.get(url).json()
            if response['status'] == '1':
                addresses = set()
                for tx in response['result']:
                    addresses.add(tx['from'])
                    addresses.add(tx['to'])

                # 일별 활성 지갑 수 계산
                daily_active_wallets[date_str] = len(addresses)
                print(f"{date_str}: {len(addresses)} active wallets")
            else:
                daily_active_wallets[date_str] = 0
                print(f"{date_str}: No data available (status 0)")
            break  # 성공적으로 호출되면 루프 종료
        except requests.exceptions.RequestException as e:
            print(f"Error on {date_str}, attempt {attempt + 1}: {e}")
            time.sleep(1)  # 1초 후 재시도
    else:
        daily_active_wallets[date_str] = None  # 모든 시도 실패 시 None 설정
        print(f"{date_str}: Failed to retrieve data after 3 attempts")

    start_date += delta

# 데이터프레임으로 변환 후 CSV 파일로 저장
df = pd.DataFrame(list(daily_active_wallets.items()), columns=['Date', 'Active_Wallets'])
df.to_csv('daily_active_wallets.csv', index=False)
print("데이터가 daily_active_wallets.csv 파일에 저장되었습니다.")
