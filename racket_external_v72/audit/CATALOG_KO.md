# 공개 데이터 색인 확인

DataCite, Zenodo, GitHub의 공개 API에서 영어 제품·장치 키워드와 넓은 제목 검색을 병행했다. 실행한 URL과 받은 제목·링크를 JSON에 남겼다. 같은 연구의 여러 버전·metadata record가 있어 건수는 연구나 독립 제품 수로 해석할 수 없다. 검색 시각은 JSON 첫 필드에 있다. [검색 결과](repositories.json) · [수집 코드](repositories.py)

- DataCite의 dataset 유형에서 제목에 “table tennis”를 포함한 검색 결과 138개를 확인했다. 각 record에는 경기·선수·물리 자극 등 이 과제와 다른 관측 단위도 섞인다. SpinDOE의 ball trajectories는 이 검색에서 실제로 식별됐다. Ball spin 자체의 학습 자료와 rubber별 입·출사 impact의 product holdout은 필요한 metadata가 다르다. [SpinDOE 데이터](https://doi.org/10.17617/3.BIVUVL)
- Zenodo에서는 “table tennis”와 “spin”을 함께 검색했을 때 forehand topspin kinematics를 포함한 두 record가 나왔다. 전문용어를 결합한 두 개의 좁은 질의는 각각 0건이었다. GitHub의 repository 검색은 이름·설명 등의 색인에 의존한다. 질의가 원 저자의 공개 저장소조차 회수하지 못할 수 있어, v71과 동일한 데이터인지 여부는 저자 사이트와 파일로 별도 확인했다. [기존 공개 racket 데이터](https://github.com/gossardt/tt_rackets)
- 신규 long-pips 표본은 1996년 원 논문의 scatterplot에서 복원했다. Fig. 3의 제품 표기는 경험적 위치 추정, Fig. 11·12의 Feint OX 표기는 실험 수치다. 이 둘을 구분해 판독했다. [원문](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf)

- 신규 anti-spin은 Preston 제작자의 2010년 영상에서 확인했다. Butterfly Super Anti의 최초 재생 segment 24개에서 trajectory와 marker rotation을 추적하고, 두 구간 모두 품질 기준을 통과한 10개에서 dimensionless benchmark를 수행했다. Absolute capture rate는 확인되지 않았고 release sign의 측정 sensitivity도 커서 수치 비교와 방향 결론의 범위를 구분했다. [제작자 영상](https://www.youtube.com/watch?v=43WWRf5yW6M) · [영상 재현 및 검증](preston/README_KO.md)

소규모 옛 논문에 원본 CSV가 별도로 있었는지, 제조사 측에 추가 기록이 남았는지는 이 검색으로 판단하기 어렵다. 공개 확인 가능한 후보마다 제품, 계측 방식, 개별 입·출사 상태를 어느 수준까지 연결할 수 있는지를 따로 기록했다. [제품별 검색 장부](../data/SEARCH_KO.md)
