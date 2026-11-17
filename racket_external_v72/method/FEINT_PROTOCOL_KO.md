# Feint OX · 두 실측 성분을 평가하는 외부 protocol

Araki et al. (1996)의 Fig.12a/b에서 공통 incident spin으로 유일하게 연결된 점을 사용한다. Spin의 단위는 rev/s이고 rad/s로 변환한다. 역사적인 38 mm ball에 R=0.019 m를 사용한다. 입력 planar velocity는 실험 protocol의 명목값 (u=0, vz=−30/3.6 m/s)으로, 측정값이 아니다. 출력은 두 panel에서 읽은 u' (m/s)와 spin' (rad/s)이다. Fig.11의 restitution으로 shot별 normal velocity를 만들어 학습/평가하지 않는다. 접촉면이 정지한 frame에서 slip은 u−Rω이다. 출력 covariance는 두 관측 성분으로 marginalize하며 네 방법에 정확히 동일한 성분을 제공한다. Continuous gate는 mean(|Δu'| + 0.019|Δω'|)가 작은 routing을 고른다. 측정된 spin MAE는 rad/s로 그대로 보고한다.

기존 여섯 제품의 공식 train 전부로 source representation/prior를 fit한다. 대상 입력이 다른 ball·incident condition을 사용하므로 비교는 rubber뿐 아니라 protocol 전이를 포함한다. Source와 target 표준화는 모두 [u,vz,Rω]표현에서 이루어진다.

주 분석은 incident spin으로만 정렬한 N개 행에서 위치 round(linspace(0,N−1,8))의 8개를 calibration 후보로 고정하고 나머지를 공통 test로 보류한다. K=4는 8개 중 가능한 70 조합을 전수 평가하고 K=8은 1회다. 각 조합이 구입하지 않은 나머지 calibration 후보는 평가에서도 쓰지 않는다. 부 분석은 각 K에서 N개 중 K개 조합을 전수 평가한 뒤 매번 나머지 N−K에서만 평가한다. 이는 같은 plot을 재활용한 finite-panel sensitivity로서 독립 시행 수가 아니다. Calibration row와 held row의 ID를 보존한다.

각 gate가 원래 source sign을 뒤집는 빈도, held reference의 과반수와 방향 결정이 맞는 비율(정확히 절반은 제외), 개별 held impact의 selected route sign accuracy, 네 방법의 output sign accuracy와 tangential velocity MAE 및 spin MAE를 나누어 기록한다. 기준 입력에서 부호가 0 근처인 점은 별도 확인한다. 명목 normal 속도 27/33 km/h 및 u ±0.10 m/s는 설계된 민감도이며 원문이 보장하는 오차 범위가 아니다. 순서와 K는 바꾸지 않는다.

**원자료 해상도에 따른 sign 판독.** 출판된 점에서 s'=0인 경우 sign을 판독할 수 없다. Primary에서 해당 calibration impact의 비용과 모든 연속 오차는 그대로 계산한다. Majority는 판독 가능한 sign만으로 과반수를 결정하고 동률에는 original을 고른다. Held sign도 판독 가능한 점만 분모에 둔다. Sensitivity에는 절대값 0.20 m/s 이내를 제외하는 경우와, 원래 strict >0 convention으로 0을 negative에 넣는 경우를 함께 둔다.

**보조 endpoint.** Fig.11의 e와 Fig.12a/b가 모두 유일하게 연결된 동일 impact 조건에서는 vz'≈e×(30/3.6 m/s)를 추가한 approximate planar vector error를 계산한다. 실제 shot별 incident normal speed가 공개되지 않아 각행의 v'z에도 그 가정이 이어진다. 이 보조 평가에서는 같은 네 정책이 이 근사 normal 성분까지 calibration에서 보고, continuous gate의 velocity cost 및 최종 velocity metric을 XZ Euclidean norm으로 바꾼다. 기본 radius, threshold, split 규칙은 공유한다.
