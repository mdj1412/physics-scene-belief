# v40의 두 정책과 측정 단위

환경별 첫 세 calibration probe에서 얻은 signed cue를 z, K256 deployment decoder에서 얻은 stick 확률을 p라고 하자. 최종 확률은 sigmoid(logit(p)+βzw)이며 β=0.8289760825845313이다. 고정 정책은 w=0.325를 쓴다. Frozen v38 optimizer는 네 번째 관측의 outgoing velocity까지 읽고 기존 source surface posterior와 gain table에서 w∈[0,1]을 고른다. 두 정책은 같은 물체 후보, sensor noise, probe 4회, 23 mm와 +1.10의 마지막 접촉, 25 mm와 +0.425의 query, 같은 K256 bank를 사용한다. 각 128개 deployment 물체에 첫 세 calibration probe로 sensor를 미리 배정해 13개에는 probe, 115개에는 query view를 적용한다. [동결 정책 호출과 결과 저장](run40.py)

같은 catalog·block에서는 12개 조건이 latent prototype·시작 방향·물체 추출·noise·QMC bank seed 쌍을 공유한다. 정확도 계산 중 acquisition이 읽는 모든 배열의 값이 같을 때만 저장된 sensor 결정을 재사용한다. 세 regime에 같은 base probability를 쓰기 전에 factual 및 candidate sensor 배열이 정확히 일치하는지 검사한다. CPU 검산은 출력을 기준으로 Brier와 source weight를 재계산하고 같은 tangent family·law에서 contour를 바꿔도 양 정책의 paid prefix와 sensor 선택, query label, probe를 선택한 물체의 확률이 동일한지 확인한다. [독립 재계산](audit40.py)

GPU p90은 8 환경×128개 물체의 동기화한 host-to-host 경과시간이다. 두 방법 모두 각 timing 호출 때 전체 acquisition을 새로 실행하고 선택된 관측 하나로 decoding한 뒤 보정 선택과 host 반환까지 수행한다. 실제 센서 측정·reset과 사전에 만든 bank 구축 시간은 별개다. 12개 cell에 각각 2회 측정 cluster를 배정해 각 catalog도 두 번씩 등장하고, 여섯 개 orientation × regime 조합이 전체에서 네 번씩 등장한다. Cluster마다 방법별 한번 warmup 후 세 번씩, 같은 workload에서 순서를 무작위화해 측정한다. 매 호출에서 반환한 sensor ID와 1,024개 probability를 정확도 추론의 해당 noise draw와 대조한다.
