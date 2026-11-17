# v62 보조 GPU timing · 실제 후속 관측이 바뀌는 조건

전체 96개 primary benchmark에서 비교 정책의 행동이 같았다는 점을 확인한 뒤 설계했다. 원래의 24개 catalog와 primary benchmark의 네 개 장비 조건 및 미리 고정한 P/C·A/B 배정 안에서, 같은 유료 pilot episode를 양 표면에 넣었을 때 selective reset과 v61 reset이 정확히 한 표면에서 다른 관측 정책을 선택하고 실제 4개 deployment session 중 적어도 하나의 R/I 구매까지 달라지는 경우를 찾는다. 선택 기준은 pilot의 두 action code와 이미 취득한 세 deployment reference로 v54 gate가 반환하는 행동이다. Binary stick label, Brier, solver outcome, latency를 읽어 표본을 고르지 않는다.

각기 다른 16 catalog에서 하나씩 택한다. 가능한 범위에서 switch 방향을 반반 두고 board와 optical cell이 고루 들어가도록 선택한다. Seed 6209911인 난수로 동점을 해소한다. 같은 원래 camera·ADC·전자센서를 선택한 episode에서 재생성하고 frozen pilot archive의 code 및 action을 검산한다. 3개 방법을 모두 한번 warmup하고, 두 measured block에서 역순을 포함한 counterbalanced order로 실제 GPU forecast를 실행한다. 512개 물체, 8회 impact 예산은 동일하다. Enriched sample은 선택 조건 때문에 실제 사용 빈도나 전체 deployment p90을 추정할 수 없다. Pooled p90과 block별 paired median을 기술통계로만 표시하고 primary 결과와 합산하지 않는다.
