# Butterfly Super Anti의 영상 기반 외부 평가

Preston Table Tennis의 2010년 비교 영상에는 같은 위치에 고정한 blade와 굵은 검은 marker를 칠한 공이 보인다. 좌상단 panel은 Butterfly Super Anti다. 제작자는 손으로 공을 공급해 서로 비슷한 장면을 골랐다고 설명한다. 각 Image Ref에서 처음 재생되는 충돌 전후의 연속 궤적만 사용했다. Topspin 12개와 backspin 12개가 남았다. 동일 장면을 다시 보여 주는 “Reaction at point of impact” replay와 side spin은 제외했다. 서로 다른 24개 조건에 장면 하나씩이므로 같은 조건에서의 trial-to-trial variance나 여러 제품에 대한 모집단 평균을 추정할 수 없다. [공개 원영상](https://www.youtube.com/watch?v=43WWRf5yW6M) · [독립적으로 읽은 24개 title card](validation/title_inventory.csv) · [원 stream의 식별 정보](source.json)

## 무엇을 측정했나

영상의 native 640×480 frame에서 Anti Spin panel의 이동 물체를 time-median background와 비교했다. Ball 크기의 disk에 걸쳐 foreground contrast가 큰 곳을 찾은 뒤, 중심에서 방사 방향으로 나가며 background와의 차이가 사라지는 outer contour로 중심과 radius를 보정했다. Racket을 가장 가까이 지난 frame을 기준으로 blade와 겹치지 않는 연속 free-flight 구간을 앞뒤에서 각각 9–24 frame 골랐다. 수평 좌표의 linear fit, 수직 좌표의 quadratic fit을 contact frame으로 평가하여 apparent contact velocity를 구했다. 이는 contact 양옆의 시간에 같은 가능성을 두고 평균낸 속도와 차이가 난다. 회전은 여러 concentric ring에서 보이는 marker texture에 대해 1·2·3·5 frame 간격의 angular cross-correlation을 함께 최대화해 구했다. 안쪽과 바깥쪽 annulus에서 각각 구해 두 추정값을 비교했다. [위치 추적](track_centers.py) · [rotation 및 velocity 추정](fit_kinematics.py)

공이 screen 오른쪽의 거의 수직인 blade에 접촉한다. Tangent는 screen 아래쪽, normal은 blade에서 왼쪽으로 향하는 방향, angular velocity Ω는 screen에서 counterclockwise가 양수다. 접점의 tangential release velocity는 (s_+=v_{t,+}-RΩ_+)다. 이 식의 두 항은 모두 pixel/playback frame으로 표현할 수 있다. 동일 shot의 타이밍과 local image scale을 공유한다는 가정 아래 (v/|v_{n,-}|)와 (RΩ/|v_{n,-}|)도 계산할 수 있다. 파일의 30000/1001 fps는 final playback frame rate이며 shooting rate는 확인되지 않았다. Out-of-plane velocity, spin axis, lens·perspective 보정, 개별 impact의 metric ruler도 공개되지 않았다. 따라서 이 결과에서 m/s, rad/s, contact-patch strain이나 물리적으로 완전한 3D slip을 판정할 수 없다. Angle alias 방지를 위해 smooth motion을 확인하고 camera 쪽 투영된 angular increment의 fitting 범위를 제한했지만 별도 high-speed reference로 검증하지 못했다. [좌표 및 checksum](source.json) · [영상 제작자](https://www.youtube.com/watch?v=43WWRf5yW6M)

## 어느 판독이 남았나

Foreground contrast, trajectory RMS, angular correlation, annulus 간 discrepancy와 최소 관측 길이를 먼저 정의해 prediction 결과와 별개로 적용했다. 이 protocol은 v72 측정 중에 정한 탐색적 품질 기준이다. 통과한 Image Ref는 **1, 3, 5, 7, 9, 11, 17, 21, 23, 24**다. Topspin은 6/12, backspin은 4/12이며, 빠른 impact가 대거 제외되어 speed distribution에 selection bias가 있다. 통과 shot의 before/after 측정 창에서 공을 포함한 31×31 patch가 인접 frame과 완전히 같은 경우는 0건이었다. 각 창의 첫 절반·마지막 절반의 수평 속도 크기 비는 incoming 0.86–1.19, outgoing 0.69–1.09였다. #11 outgoing의 변동이 가장 컸다. 외부 clock이 없어 frame drop이나 editing에 의한 미세한 cadence 변화는 배제하기 어렵다. [판독 기준](quality_protocol.md) · [전체 24개의 판독/제외 이유](all_tracks.csv) · [통과한 10개 행](anti_eligible.csv)

Point estimate는 10개 모두 서로 다른 값으로 제공한다. 별도로 angular subwindow, apparent radius ±2 px, blade tilt ±5°, apparent acceleration 0–0.1 px/frame²와 unconstrained quadratic fit을 조합해 deterministic sensitivity interval을 구했다. 이 범위는 confidence interval과 용도가 다르다. Outgoing (s_+) 부호는 #1 [−2.26, −0.02] 및 #17 [−8.25, −4.77] px/frame에서만 유지됐다. #17의 incoming 부호는 민감했다. 0.1 px/frame의 추가 margin을 요구하면 #17만 outgoing 부호가 유지된다. 평가 공통 holdout #1·#24에서는 #24의 range가 −1.11∼+1.04로 0을 지난다. 따라서 10개 velocity/spin point estimate를 이용한 오차와 물리적 방향 판별의 근거 수준이 다르다. [각 행의 sensitivity](sign_sensitivity.csv) · [계산 코드](assess_sensitivity.py)

독립 판독자가 #3과 #17의 궤적과 stripe를 다시 읽었다. #3에서 contact보다 약 14 frame 뒤의 짧은 구간으로 구한 (s_+)는 +0.46 px/frame이었다. 원점이 다른 두 속도를 그대로 비교하면 방향도 달라진다. 같은 contact로 extrapolate한 독립 추정은 약 −0.26, 자동 추정은 −0.57이다. #17의 독립 outgoing 결과 −6.32는 자동 −6.30과 가까웠으나, incoming 독립 판독은 −0.19, 자동 판독은 +0.63이었다. 이 사례는 수동·자동 점수의 소수점 일치보다 방향 미확정 표시가 중요한 이유다. [수동 판독의 원래 구간](validation/manual_angles.csv) · [Ref. 3 중심 판독](validation/centres_ref3.json) · [Ref. 17 중심 판독](validation/centres_ref17.json)

## 재현

원영상에서 itag 135의 공개 640×480 video stream을 사용했다. Source SHA-256은 source.json에 기록했다. 동일 파일이 준비된 경우 다음 명령으로 24개 frame range의 추적값, quality decisions, eligibility table과 sensitivity table을 다시 만들 수 있다. Python 3, NumPy 2.4.6, SciPy 1.17.1, OpenCV 5.0.0.93에서 확인했다. Video frame은 별도 배포하지 않는다.

```sh
python reconstruct.py /path/to/preston-640x480.mp4 /path/to/output
```

[재현 entry point](reconstruct.py) · [four-panel video의 frame index](reference_frames.csv)
