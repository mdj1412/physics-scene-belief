# v39의 frozen policy, paired observation 및 GPU timing

비교의 단위는 한 환경의 네 calibration 접촉 후, 같은 환경에서 별도로 생성한 물체의 다음 충돌이 stick할 확률이다. 첫 세 접촉과 모든 factual information으로 각 128개 물체 중 13개에는 deployment probe, 115개에는 query view를 배정한다. 두 정책은 이 배정과 각 물체의 같은 유료 관측, 같은 K256 기반 physics prediction을 공유한다. 네 번째 calibration velocity는 배포할 센서의 종류를 바꾸지 않는다. 모든 환경의 접촉 비용은 fixed 4다.

공통 base probability를 p, 첫 세 접촉에서 읽은 signed cue를 z라고 할 때 최종 확률은 sigmoid(logit(p) + βzw)이고 β=0.8289760825845313이다. 고정 정책은 w=0.325를 사용한다. Continuous 정책은 +1.10의 네 번째 noisy outgoing velocity로 source surface posterior를 갱신하고, `continuous38.source_optimum`을 직접 호출해 w∈[0,1]을 고른다. Source gain table이 예측하는 Brier 감소가 최대인 w를 찾을 때 양 끝·고정값·두 끝 사이의 모든 knot를 비교한다. 최대값이 여럿이면 0.325와 가장 가까운 w를 선택한다. 기존 optimizer 파일의 SHA-256과 v38 run manifest에 기록된 source 파일을 모두 검사한다. Target geometry, law, orientation, regime, Y는 정책 함수의 입력이 아니다. Catalog와 표면 이름은 검증용 generator와 사전 조건별 forecast에서 사용한다.

## Timing

8개 환경 × 128개 물체에 실제 관측을 할당하고, 한 noise realization에서 각 물체의 선택된 센서로 probability를 계산한 후 보정 선택 및 host 반환까지 측정한다. Fixed에는 첫 세 관측의 단서 추출, optimizer에는 source posterior update·gain 최적화를 포함한다. 같은 방법·같은 hardware의 warm host-to-host latency이며, CUDA synchronization 양 끝의 경과 시간을 사용한다. 센서값은 GPU 호출 전에 존재한다. 별도 bank 초기화, simulator에서 label이나 센서를 만드는 시간, 실제 로봇 접촉·reset 비용은 포함하지 않는다.

각 cell에서 짝수 catalog 6개, 첫 block을 측정해 24개 cluster에 해당한다. Six orientation–regime 조건을 cell마다 한 번씩 배정했다. 각 cluster에서 두 방식을 한 번씩 warm up하고, 세 번씩 paired 측정하며 매 반복마다 실행 순서를 무작위화한다. 모든 time trial마다 해당 방법이 반환한 sensor action과 물체 1,024개의 probability를 실제 Brier를 생성한 batch와 비교한다.

## 원자료

`results/<cell>/catalog_<id>/block_<b>.npz`에는 두 orientation·세 regime의 label, 관측, 확률, 보정량과 Brier가 들어 있다. Source cue z와 같은 prefix에서 얻은 p_base를 제공해 확인된 Brier를 재생성할 수 있다. 한 block의 calibration과 deployment seed는 별도 stream이고, 같은 sign에서 세 regime의 첫 세 paid velocity·factual state·선택 센서 정보가 같음을 검사한다. Noise 32회는 같은 latent deployment draw에 대한 반복이다. 그중 첫 realization만 별도 latency trial에 사용한다. 각 archive에 대응하는 JSON은 seed·source fingerprint·SHA-256을 저장한다. SHA-256과 source fingerprint가 같은 완료 block은 재시작할 때 재사용한다. 최종 CPU 검산은 likelihood와 source table에서 환경별 보정량을 다시 골라 두 방법의 Brier를 재현한다. [GPU 구현](run39.py) · [원자료 검산](audit39.py)
