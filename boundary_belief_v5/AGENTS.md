# Scope

User-authorized GPU extension. Parent의 초기 CPU-only pilot 제한은 적용하지 않는다. 실제 simulation/inference는 datacenter3의 idle allocated GPU에서만 수행한다. 기존 jobs, 기존 results는 변경하지 않는다. Local copy는 구현 준비·검토·보고서용이다.

설명은 한국어, technical term은 English. Truth는 renderer, calibration generator, candidate forward model, evaluation에서만 사용한다. Acquisition에는 factual evidence와 공개 query만 전달한다. Raw-image feature compression과 general image likelihood를 구분한다. Real cloth의 image-space contour audit를 physical counterfactual validation으로 표현하지 않는다.
