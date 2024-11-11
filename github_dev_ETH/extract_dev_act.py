import requests
import pandas as pd
from datetime import datetime, timedelta

# GitHub 인증 설정 (토큰 입력 필요)
headers = {
    'Authorization': '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'
}
# 리포지토리 정보 설정 (예시: 'ethereum' 리포지토리의 'go-ethereum')
owner = 'ethereum'
repo = 'go-ethereum'

# 4년 전부터 현재까지의 커밋 수집
end_date = datetime.now()
start_date = end_date - timedelta(days=4*365)

# 모든 커밋 데이터 가져오기
commits = []
page = 1
while True:
    url = f'https://api.github.com/repos/{owner}/{repo}/commits'
    params = {
        'since': start_date.isoformat() + 'Z',
        'until': end_date.isoformat() + 'Z',
        'per_page': 100,
        'page': page
    }
    response = requests.get(url, headers=headers, params=params)

    # 에러 체크
    if response.status_code != 200:
        print("Error:", response.json())
        break

    data = response.json()
    if len(data) == 0:
        break

    commits.extend(data)
    page += 1

# 커밋 날짜 추출
commit_dates = [commit['commit']['author']['date'] for commit in commits]
commit_dates = pd.to_datetime(commit_dates)

# 데이터프레임 생성 및 일별 집계
df = pd.DataFrame(commit_dates, columns=['commit_date'])
df['count'] = 1
daily_commits = df.resample('D', on='commit_date').sum()



# CSV 파일로 저장
daily_commits.to_csv('daily_commits.csv', index=True)


print("Data saved to daily_commits.csv")