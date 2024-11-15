import pandas as pd

# 주어진 검색량 데이터 파일 경로
file_path = 'multiTimeline.csv'

# CSV 파일 읽기
search_data = pd.read_csv(file_path)

# Date 열을 datetime 형식으로 변환
search_data['Date'] = pd.to_datetime(search_data['Date'])

# Date를 인덱스로 설정하고 일 단위로 리샘플링
search_data_daily = search_data.set_index('Date').resample('D').asfreq()

# Frequency 열에 대해 선형 보간법 적용
search_data_daily['Frequency'] = search_data_daily['Frequency'].interpolate(method='linear')

# 결과 저장 또는 출력
print(search_data_daily.head(10))

# 일 단위로 변환된 데이터를 CSV 파일로 저장 (옵션)
output_file = 'search_daily.csv'
search_data_daily.to_csv(output_file)
print(f"Converted daily data saved to: {output_file}")
