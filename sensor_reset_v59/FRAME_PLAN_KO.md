# v59 · 별도 영상 검증의 사전 명세

Primary factorial 비교에서는 독립 optical channel을 sign·visibility 수준에서 모사한다. 추가 검증은 24개 catalog와 각 surface에서 미리 정한 앞 64개 pilot을 선택하여 두 물체의 reference와 impact를 각각 1,000 fps grayscale fiducial clip으로 구성한다. 시점·noise·occlusion은 전자센서와 독립이며, 두 reset counterfactual에서 이미지는 동일하게 사용한다. 비교 대상은 같은 subset의 sign-level camera, image-level camera, camera off이고 기존 policy likelihood와 v56의 cutoff는 조정하지 않는다. 4×2의 8회 impact를 모두 시도한다.

카메라의 수평 변형을 움직이는 표식과 같은 영상의 고정 표식 사이 거리로 측정한다. 0–5 ms의 여섯 frame, 0.5 ms shutter 내 두 exposure sample, 13×27 pixel, 약한 signal과 가림, camera shake 및 독립 pixel noise를 생성한다. Reader는 frame만 보고 background threshold 위의 밝기 중심과 0.5 pixel 변형 기준으로 부호를 정한다. 두 부호가 모두 읽힐 때만 mismatch를 사용한다. 활성 물체의 가시성·부호 오류·무효 물체의 false detection을 별도로 감사한다. Renderer는 simulator truth로 장면을 만들고, reader에는 truth·mechanism·reset ID·목표 정책을 전달하지 않는다. Primary backend는 물체의 공통 inactive flag를 가정하므로, 양쪽 전자센서도 inactive인 경우 camera의 단독 false detection은 감사에만 남긴다.

주요 결과는 이 subset에서 원인 판별률, 정책 판별률, expected held-out Brier이다. 영상에서 관측한 실제 sign을 그대로 주입하고 source에서 지정한 analytic camera likelihood는 다시 맞추지 않는다. 표본 단위는 24개 paired catalog로 하여 pointwise 95% bootstrap CI를 보고한다. 이 검증에 쓰는 marker geometry와 exposure는 실제 고속 장비 사양을 추정한 것이 아니다. 실제 pixel 추출 시간은 별도로 기록하고, primary GPU p90의 측정 시작 지점을 명시한다.
