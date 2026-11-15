# Training phase에서 추가한 strong moment baseline

Main pipeline이 training worlds를 처리하는 동안, test 결과를 확인하기 전에 LMMSE acquisition baseline을 추가한다. Main5 primary contrasts는 변경하지 않는다. 이 baseline은 posterior의 query–observation covariance와 observation covariance를 사용한다.

Gain(a)=trace(Cov(T,Y_a) [Cov(Y_a)+R_a]^{-1} Cov(Y_a,T))/2.

이는 linear minimum mean-square estimator의 expected reduction이며 nonlinear mixture posterior에서 Bayes EVSI와 같다는 보장은 없다. Outer sampling과 learned rule training이 필요하지 않으므로, 복잡한 nested acquisition이 필요한지를 검증하는 강한 baseline이다. Factual joint posterior를 계산하는 비용은 여전히 든다.

Main policies를 덮어쓰지 않고 별도 extra_policies와 freeze를 만든다. 코드에는 private truth나 evaluation 파일을 읽는 경로를 넣지 않는다. 모든 baseline의 actual comparison은 이 별도 freeze가 완료된 후 analysis에서 한다.
