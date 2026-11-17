# v73 · SI 기준 모델 비교 규약

Source 모델은 훈련 시 대상 rubber 전체를 제외한다. 기존 A-B-S II soft 대조군은 config 10만 target이다. 같은 실험실에서 얻은 632개 공식 train 중 4개 또는 8개를 calibration에 쓰고, 158개 공식 test는 공통 평가에 쓴다. 객체는 (racket_angle, traj_dir)로 식별한다. 두 split 간 중복을 코드로 확인한다. Original 791개 중 이 두 split에 없는 행은 따로 다룬다. 공식 split 제외 사유가 확인되지 않아 하나뿐인 이 행을 성능 비교의 분모에 더하지 않는다.

Fast는 target outcome을 사용하지 않고, target을 제외한 source 공식 train의 입사 normal speed 상위 사분위수 이상으로 고정한다. Calibration에는 fast와 나머지를 같은 수로 넣고, 각 draw의 처음 4개는 8개의 prefix다. 각 층에서 무복원으로 256개 draw를 만들며 네 방법의 행과 순서를 맞춘다. 이는 deployment에서 입사 조건을 선택할 수 있을 때의 비교다. 부 분석은 v72의 uniform draw에서 동일 결과가 나오는지 검산한다. 256은 독립 보정 session 수가 아니다.

Release는 racket frame에서 s'=u'-Rω' 부호다. 도착한 SI CSV에 대해 |s'|>0.20 m/s일 때 부호에 표시를 준다. 이는 v72 Feint OX의 stress test과 같은 최소 margin이며 계측기의 보장된 공차가 아니다. 0 및 0.5 m/s에서도 반복해 공개한다. 투표에서 모호한 calibration 행은 보류하되 예산에는 포함한다. 판독률은 해상 가능한 test 행 수 / 유효 SI 행 수, 적중률은 해당 행에서 최종 예측 부호가 맞은 비율이다. 고유 impact 수와 calibration draw 반복 수를 구별한다. SI CSV는 이미 궤적 추정에 성공한 공개 행을 담아, 촬영 시도 전체에 대한 tracking yield를 따로 구할 수 없다. Velocity·spin error는 부호 판독 여부와 무관하게 유효 SI test 행 전부로 계산한다.

기존 majority, continuous-loss gate, single affine EB, elastic impulse와 scale·parameter를 v72에서 재사용한다. 연속 손실의 gate는 오류의 합 ‖Δv_xz‖+R|Δω|로 expert routing을 정한다. 모델의 새 시편 성능을 평가하려면 별도 제품 identity, timing, scale, orientation 및 유효·무효 장면 목록을 입증해야 한다. 외부 mode의 두 CSV 필수 열은 impact_id, v_in_x, v_in_z, spin_before_y, v_out_x, v_out_z, spin_after_y이며, ball radius는 source와 같은 20 mm로 검증된 경우에 쓴다.
