# 독립 진단 사전 계획

Population의 double mismatch 분해와 여덟 번을 지불한 단일 표면에서의 원인 판별은 별개로 평가한다. 사전 가중치는 A/B, 물리적 persistence 유무, sensor latch 유무의 8개 조합에 동일하게 부여한다. 각 pilot의 관측을 8개 후보의 likelihood와 대조하여, nuisance를 합산한 뒤 persistence 유무 및 latch 유무를 각각 MAP로 분류한다. Posterior가 0.5이면 off로 처리한다. 네 원인 조합의 joint MAP 성공률도 보고한다. 8×1과 4×2에서 같은 센서 수·판독 소실·impact 예산을 적용한다. 평가 지표는 balanced accuracy, 각 원인의 true positive와 false positive rate, 가장 헷갈리는 네 원인 조합, 24개 paired catalog bootstrap의 95% interval이다. Sensor artifact가 있으면 한 물체에 대한 두 판독의 불일치로 latch의 정보를 얻을 수 있으므로, 8×1에서도 sensor artifact의 판별률이 우연 수준보다 높을 수 있다. 8×1에 물리적 지속 여부를 알 단서는 없다. 이 지표로 실제 배포 환경의 원인을 맞힐 수 있다는 보장은 없으며, 보조 측정의 상한에 가까운 model-matched 진단으로 사용한다.

Shared upstream distortion이 latent physical mismatch와 같은 distribution의 가짜 신호를 두 센서 모두에 전달하면 모든 유료 판독의 joint distribution이 같아질 수 있다. 동일한 random seed에서 읽을 수 있는 부호 배열이 일치하지만 ground truth persistence가 다른 두 synthetic 자료를 구성하여 이 결론의 범위를 점검한다. 이 경우를 primary factorial 평균에 넣지 않는다.
