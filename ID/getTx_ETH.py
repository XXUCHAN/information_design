import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv
load_dotenv()
etherscan_api = os.getenv("ETH_API")
# Etherscan API 키
API_KEY = etherscan_api

# 데이터 로드
data = pd.read_csv('top_address_ETH.csv')

# 알려진 거래소 지갑 주소 목록
exchange_wallets = [
    "0x3f5CE5FBFe3E9af3971dD833D26BA9b5C936F0bE",  # Binance
    "0x503828976D22510aad0201ac7EC88293211D23Da",  # Coinbase
    "0xDc76CD25977E0a5Ae17155770273aD58648900D3",  # Huobi
    "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",  # Bitfinex
    "0x0A869d79a7052C7f1b55a8EbAbb5a8E1c0eAbc79",   # Kraken
    "0x5c985e89dde482efe97ea9f1950ad149eb73829b",   #OKEx
    "0x7e59c0c5b3d3f6b3b5b1b5b1b5b1b5b1b5b1b5b1",   #Gemini
    "0xfdb16996831753d5331ff813c29a93c76834a0ad",   #Bittrex
    "0xb794f5ea0ba39494ce839613fffba74279579268",    #Poloniex
    "0x2b5634c42055806a59e9107ed44d43c426e58258",    #KuCoin:
    "0xbe0eb53f46cd790cd13851d5eff43d12404d33e8",
    "0xda9dfa130df4de4673b89022ee50ff26f6ea73cf",
    "0x61edcdf5bb737adffe5043706e7c5bb1f1a56eea",
    "0xf977814e90da44bfa03b6295a0616a897441acec",
    "0xc61b9bb3a7a0767e3179713f3a5c7a9aedce193c",
    "0x5a52e96bacdabb82fd05763e25335261b270efcb",
    "0x28c6c06298d514db089934071355e5743bf21d60",
    "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43",
    "0xec09d70cc666a9ff2b710c4d9b9f900c8a9b9610",
    "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0"

]
# 입금 및 출금 거래를 저장할 리스트
deposit_transactions = []
withdrawal_transactions = []

def get_transactions(address):
    url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('result', [])
    else:
        return []

# 각 주소에 대해 트랜잭션 데이터 가져오기
for addr in data['Top_Address']:
    print(f"Fetching transactions for address: {addr}")
    transactions = get_transactions(addr)

    # 입금 및 출금 거래 구분
    for tx in transactions:
        if tx['to'] in exchange_wallets:
            deposit_transactions.append(tx)  # 거래소로 입금
        elif tx['from'] in exchange_wallets:
            withdrawal_transactions.append(tx)  # 거래소에서 출금

    # API Rate limit 대응을 위한 대기 시간 추가
    time.sleep(0.4)  # API 요청 사이에 짧은 대기 (필요에 따라 조정)

# 필터링된 거래 데이터를 데이터프레임으로 변환
deposit_df = pd.DataFrame(deposit_transactions)
withdrawal_df = pd.DataFrame(withdrawal_transactions)

# 결과를 CSV 파일로 저장
deposit_df.to_csv('./deposit_transactions.csv', index=False)
withdrawal_df.to_csv('./withdrawal_transactions.csv', index=False)

print("Deposit transactions saved to /mnt/data/deposit_transactions.csv")
print("Withdrawal transactions saved to /mnt/data/withdrawal_transactions.csv")