## Frozen decoder와 관측 예산

같은 v43 source hypothesis bank(16,384개), neutral prior, 원본·privileged likelihood, source temperature를 새로 생성한 v44 물체에 적용했다. 조건에는 fold/ripple, inlay/woven, rational/gaussian-rate, base/hollow/bulge, 명목 높이 19/31 mm를 모두 교차했다. v43과 support가 겹치지 않도록 새 catalog 400–411, 시드 440000000+100000·panel을 사용했다. 환경과 물체, sensor noise는 형상·재질·모든 목표 충돌 조건에 짝지었다.

Factual evidence와 구입한 관측은 v43에서 검증한 관측 장치를 그대로 모델링했다. Factual collision과 물체당 선택한 deployment probe의 속도 측정은 기존 solver로 0 mm에서 생성했다. Query view는 명목 높이의 7점 외곽을 직접 읽는다. 네 번의 calibration 접촉 및 response도 생성했다. Zero와 temperature 두 정책은 이 네 번째 response로 보정 강도를 선택하지 않는다. 목표 tip radius나 contact patch가 달라져도 **그 전에 이미 측정한 evidence는 고정**하는 개입이다. 새 solver의 tip으로 factual 및 probe 속도까지 다시 측정하는 end-to-end sensor transfer와 구별된다.

기존 코드가 함께 반환하던 단발 query collision label을 읽지 않도록 관측 함수를 분리했다. Frozen predictor는 고른 sensor와 source bank에 가정한 query event만 읽는다. Target의 실제 stick/slip은 이 코드와 예측 파일에 없고 별도 solver가 생성한다. Privileged arm만 센서의 실제 변형 축을 받아 projection한다. 두 correction 모두 tip radius, patch, 실제 접촉 위치, target 물성과 outcome을 입력받지 않는다. 이 때문에 같은 물체의 증거에서는 사전 확률을 모든 tip 조건에 공유한다. [관측·예측 코드](runner44.py) · [v43 source 모델](../blind_decoder_v43/frozen_model43.json)

### 비용과 재현

- 각 panel의 4개 환경에 128개씩 물체를 투입하고 K256이 환경별로 115개 query view와 13개 deployment probe를 배정했다. 추론 파일은 576개이며 각 episode에 4개 sensor noise를 적용했다. 같은 panel의 base prototype은 구조적으로 각 조건에 반복된다. Solver가 채점할 물체는 사전에 정한 subset을 사용한다.
- 한 NVIDIA RTX PRO 6000 Blackwell Server Edition에서 512개 물체·한 noise draw 기준으로 네 방법을 완전 교차했다. 각 panel에서 네 방법을 무작위 순서로 1회 warm-up하고 3회 측정하여 정책당 36개 기록을 얻었다. 각 측정에는 neutral K256 결정, paid sensor routing, 실제 physics posterior, 해당되는 temperature 연산, CPU–GPU 전송과 동기화가 포함된다. Cold 측정에는 메모리에 있는 bank의 template 생성도 포함한다. 물리 측정·reset, offline bank 생성과 target label용 solver 시간은 별도다.
- GPU p90은 각 radius/patch에서 같은 운영 경로다. Label을 얻는데 필요한 simulator compute는 운영 예측 latency에 포함되지 않는다. p90 구간은 catalog panel 단위 paired bootstrap으로 산출하며, hardware와 batch size를 바꿔 외삽하지 않는다.
- 서버 프로젝트의 v43 의존성과 파일 hash를 검증한 뒤 GPU 하나 이상에서 `bash run_gpu44.sh even` 및 `bash run_gpu44.sh odd`를 실행하고, 두 단계 이후 한 장에서 `bash run_gpu44.sh timing`을 실행한다. Guard는 다른 GPU compute process가 나타나면 자신의 worker만 중단한다. `../continuous_v2/.venv/bin/python audit_gpu44.py`로 SHA, 확률, paid/unchosen sensor와 latency를 검산한다. [예측 코드 동결](gpu_freeze44.sha256) · [GPU 독립 검산](audit_gpu44.json)
