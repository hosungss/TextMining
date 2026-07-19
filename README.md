# 유튜브 스타트업 텍스트 마이닝

`스타트업`과 `startup` 검색어를 바탕으로 유튜브 공개 영상과 댓글을 수집하고, 통계표·경향표·Excel 보고서·HTML 보고서를 만드는 파이썬 프로젝트입니다.

> **키워드와 YouTube API 키만 입력하면 영상·댓글 경향을 통계, 그래프, Excel 및 연구 보고서로 자동 생성하는 텍스트 마이닝 도구**

**서강대학교 가상융합전문대학원 메타버스비즈니스 전공 박사과정 손호성 작성 · v1.0**

## 화면형 프로그램으로 실행하기

가장 간단한 방법은 `start_app.cmd`를 더블클릭하는 것입니다. 처음 실행할 때 가상환경과 필요한 라이브러리를 준비한 뒤 브라우저에서 **YouTube Research Studio**가 열립니다.

PowerShell에서는 다음처럼 실행할 수 있습니다.

```powershell
python -m pip install -r requirements.txt
python run_app.py
```

프로그램 화면에서 다음을 모두 설정할 수 있습니다.

- YouTube Data API 키 입력 및 연결 시험
- 복수의 검색어·주제, 시작일·종료일, 국가·언어·정렬 방식
- 검색어별 영상 수와 영상별 댓글 수
- 사용자 정의 주제명과 주제별 키워드 사전
- 긍정·부정 감성 사전과 분석 제외 단어
- 답글 포함 및 전체 답글 수집 여부
- 실행할 통계 방법론 묶음과 부트스트랩 반복 수
- 가상자료 시험 또는 실제 YouTube 수집·분석

API 키는 화면 메모리에서만 사용되며 설정 파일, 결과 파일, 로그에 저장되지 않습니다.

### 포함된 통계 방법론

- 정규성: Shapiro-Wilk, D'Agostino K²
- 등분산성: Brown-Forsythe
- 두 집단: Student t, Welch t, Mann-Whitney U
- 다집단: 일원분산분석, Welch ANOVA, Kruskal-Wallis
- 사후검정: Tukey HSD, 쌍별 Mann-Whitney, Holm·FDR 보정
- 효과크기: Cohen d, Hedges g, 순위양분상관, eta², epsilon², Cramer's V
- 상관: Pearson, Spearman, Kendall tau-b
- 범주형: 카이제곱 독립성, Fisher 정확검정
- 회귀: OLS-HC3, 중앙값 회귀, Huber 강건회귀, Poisson·음이항 GLM, 로지스틱 회귀
- 회귀 진단: VIF, Breusch-Pagan, Jarque-Bera, Shapiro-Wilk, Durbin-Watson
- 시계열: 선형 추세, Mann-Kendall, ADF 정상성, 3개월 이동평균
- 다변량: PCA, K-means, silhouette 기반 군집 수 선택
- 데이터 진단: IQR·MAD 이상치, 검정력·필요 표본크기

프로그램은 자료 조건에 맞는 방법만 실행합니다. 통계표에는 원 p값뿐 아니라 Holm 보정 p값과 FDR q값을 함께 제공하며, 자동으로 이상치를 삭제하지 않습니다.

## 무엇을 분석하나

- 월별 영상 게시량과 조회 성과 추세
- 조회수, 좋아요, 댓글 수의 기술통계
- 게시 후 경과일을 보정한 일평균 조회수
- 조회수 1,000회당 좋아요·댓글·참여량
- 검색어별, 영상 길이별, 주제별 비교
- 채널별 영상 수와 성과
- 댓글의 탐색적 긍정·부정·중립 분류
- 제목·설명·댓글의 빈출어와 댓글 연관어
- Spearman 상관계수와 부트스트랩 95% 신뢰구간
- 채널 규모·영상 나이·길이를 보정한 로그 조회수 회귀

## 1. 설치

Python 3.10 이상에서 다음 명령을 실행합니다.

```powershell
python -m pip install -r requirements.txt
```

## 2. API 키 준비

Google Cloud Console에서 YouTube Data API v3를 활성화하고 API 키를 발급받은 뒤, 현재 PowerShell 창에 환경변수로 등록합니다.

```powershell
$env:YOUTUBE_API_KEY = "발급받은_API_키"
```

키는 코드나 `config.json`에 저장하지 않습니다.

## 3. 실제 데이터 수집과 분석

```powershell
python youtube_startup_mining.py all --config config.json --out runs/startup_2025
```

수집과 분석을 따로 실행할 수도 있습니다.

```powershell
python youtube_startup_mining.py collect --config config.json --out runs/startup_2025
python youtube_startup_mining.py analyze --config config.json --input runs/startup_2025/raw --out runs/startup_2025/analysis
```

## 4. API 키 없이 기능 시험

다음 명령은 재현 가능한 가상 표본을 만들고 전체 분석을 실행합니다.

```powershell
python youtube_startup_mining.py demo --config config.json --out runs/demo
```

## 주요 결과물

`analysis` 폴더에 다음 파일이 생성됩니다.

- `startup_youtube_analysis.xlsx`: 원자료 일부와 모든 분석표, Excel 차트
- `startup_youtube_all_results.md`: 분석 개요와 모든 통계 결과표를 통합한 Markdown 보고서
- `startup_youtube_report.html`: 브라우저에서 바로 볼 수 있는 요약 보고서
- `videos_enriched.csv`, `comments_enriched.csv`: 파생지표가 추가된 정제자료
- `tables/*.csv`: 월별·검색어별·주제별·채널별·감성·연관어·통계 결과
- `run_manifest.json`: 분석 조건과 데이터 건수

프로그램의 **자료·다운로드** 탭에서는 모든 결과 Excel, 모든 결과 Markdown,
HTML 요약 보고서, 원자료 포함 전체 ZIP을 각각 내려받을 수 있습니다. Markdown은
전체 분석 결과표를 한 문서로 묶고, 대용량 영상·댓글 정제 원자료는 Excel과 ZIP에
포함합니다.

수집된 원자료는 `raw/videos.csv`, `raw/comments.csv`에 저장됩니다. 댓글 작성자 이름은 수집하지 않으며 댓글의 기술적 ID만 답글 구조 확인용으로 보관합니다.

## 설정 변경

`config.json`에서 다음 값을 조정할 수 있습니다.

- `published_after`, `published_before`: 분석 기간
- `max_videos_per_query`: 검색어별 영상 수
- `max_comments_per_video`: 영상별 댓글 수
- `order`: `date`, `relevance`, `viewCount`, `rating`, `title`
- `keyword_groups`: 스타트업 세부 주제 사전
- `positive_words`, `negative_words`: 탐색적 감성 사전

## 해석 시 주의점

1. 유튜브 검색 결과는 무작위 표본이 아니므로 유튜브 전체 의견으로 일반화할 수 없습니다.
2. 조회수·좋아요·댓글은 수집 시점의 누적값입니다. 실제 시계열 증가량을 보려면 같은 영상을 주기적으로 다시 수집해야 합니다.
3. 댓글 감성은 작은 사전 기반의 탐색적 분류입니다. 논문용 감성분석이라면 사람이 코딩한 검증 표본으로 정확도와 일치도를 확인해야 합니다.
4. `스타트업`과 `startup` 검색 결과가 겹칠 수 있습니다. 원자료에서는 중복 영상을 하나로 합치고 `matched_queries`에 모든 검색어를 기록합니다.
5. 회귀계수는 연관성을 나타낼 뿐 인과효과를 의미하지 않습니다.
6. 공식 API를 사용하며, API 정책·할당량·연구윤리 기준을 함께 확인해야 합니다.
