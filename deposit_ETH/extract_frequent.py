import pandas as pd

# 데이터 로드
data = pd.read_csv('deposit_value_eth.csv')

# timeStamp를 datetime 형식으로 변환
data['timeStamp'] = pd.to_datetime(data['timeStamp'])

# 날짜 단위로 변환 (시간 정보 제거)
data['date'] = data['timeStamp'].dt.date

# 날짜별 트랜잭션 빈도 계산
date_frequency = data['date'].value_counts().sort_index()

# 결과를 DataFrame으로 변환하여 CSV로 저장
date_frequency_df = date_frequency.reset_index()
date_frequency_df.columns = ['Date', 'Frequency']
date_frequency_df.to_csv('./deposit_frequency.csv', index=False)


