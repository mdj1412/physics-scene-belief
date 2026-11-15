# 이미 관측한 물체에 접촉만 추가하는 보충 검증

Fresh-object primary panel과 v30의 구입 절차가 다르므로 후속 panel을 추가한다. 초기 설계를 바꾸지 않고 별도 결과로 남긴다. Secondary panel 설계 시점에 세 law의 첫 factual-only control 결과를 확인했다. 해당 subset의 순위·효과 크기는 반복 수 선정 및 양성 판정에 사용하지 않는다.

같은 factual pool, paid prefix, bank, law, deployment worlds·labels·decoder·quota를 고정한다. 각 future에서 이미 paid인 index를 제외한 pool에서 2개 또는 4개의 서로 다른 물체를 균등 추출하고, 해당 물체의 실제 latent state에 맞는 새로운 noisy probe response를 생성한다. 다른 future들의 후보 pool은 공유하고 독립 난수로 복원한다. 어떻게 새 latent object를 생성했는지에 따라 환경을 바꾸지 않는다. 추가 likelihood는 이미 반영된 factual marginal을 제거하고 해당 물체의 joint likelihood로 대체한다. Prefix로부터 시작하는 before action이 primary와 같은지, paid index 제외, quota 유지, gain과 action 재계산을 확인한다. 세 law 및 두 stage에서 fresh-object 결과와 같은 단위를 쓴다.

주요 비교는 (1) matched panel 자체의 signed Brier gain, future·batch·interaction variance, 독립 1×1 및 2×2 환경 pair concordance, (2) 같은 prefix·batch에 대한 두 panel의 block-paired 차이와 direction이다. 새 물체와 이미 본 물체의 future index 사이에는 일대일 latent object 대응이 없다. 진단 결과는 secondary로 표시하고 각 panel 안에서의 in-sample 순위와 별도 반복의 순위를 구별한다.
