# Super Anti · frame rate가 공개되지 않은 영상에서의 보조 평가

양쪽 segment의 프레임 시간 간격이 같고 접촉에 대한 tangent·normal 방향, ball 회전 부호와 radius를 복원할 수 있는 impact만 사용한다. 최소 9행과 부호가 있는 두 방향의 translation·spin을 요구한다. 직접 얻은 것은 image-plane center·stripe의 변화다. Rω와 translation의 단위가 같아야 한다. Crop, radius, time step, surface 움직임이 두 segment에서 일치하는지 확인한다. 절대 m/s 및 rad/s는 추정하지 않는다.

각 source와 target impact마다 같은 변환을 적용한다. 모델에 넣을 상태는 8×(u,vn,Rω)/|vn,in|이다. 8은 수치 단위를 맞추는 공통 임의 상수로 어떠한 target 속도 추정치도 아니다. 결과를 8로 나눈다. 예측 오차는 normalized planar velocity vector error 및 normalized spin rim-speed error R|Δω|/|vn,in|이며 두 오차 모두 무차원이다. θ sign은 s'=u'−Rω' 기준이다. Normalized input의 vn은 상수이므로 affine·logistic의 feature는 intercept, standardized u, standardized Rω 세 개다. Elastic model은 normal restitution에 intercept만 주고 tangential impulse response α를 |u−Rω|에 대한 affine로 둔다. 같은 source split의 여섯 제품 모두에서 refit한 four-way comparison으로, v71 SI transfer predictor와 다른 추정량이다.

Frozen 두 expert에 같은 calibration을 주는 majority와 continuous-loss gate를 쓴다. 후자는 normalized vector error + normalized spin rim error를 합친 수치를 최소화한다. Source EB는 v71의 기존 covariance shrink와 0.1 prior precision multiplier를 계승한다. Elastic model은 v71과 같은 e∈[0,1], α∈[0,.8] 제약으로 세 계수의 posterior mean을 갱신한다. 소스 데이터가 가진 original SI scale에서의 floors를 유지하기 위해 source/target 모두에 위에서 정한 8배 convention을 쓴다.

Input의 normalized signed slip 순서만으로 정렬해 round(linspace(0,N−1,8)) 위치 8개를 calibration 후보로 고정한다. 나머지 동일 held-out impacts에서 4회(C(8,4)=70)와 8회(1회)를 비교한다. 부 분석은 row 내에서 paired leave-out을 유지하는 256개 난수 8개 순열(seed 720073)에서 고정 4/8을 쓰고 동일한 나머지 N−8에서 평가한다. 각 subset에서 label이 해상도상 s'=0인 점은 비용·연속 오차에 포함하고 sign 투표와 평가에서 제외한다. 추가로 tracking이 허용하는 sign uncertainty 기준이 있으면 판독 coverage와 민감도를 함께 보고한다. 같은 영상 안의 impacts이므로 camera·session 간 독립 재현으로 해석할 수 없다.
