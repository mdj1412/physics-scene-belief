"""Attach the additional practical label-budget control after the predeclared diagnostics."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
path=ROOT/'SELECTOR_AUDIT_KO.md'
heading='## Donor 물체당 하나의 event만 쓸 때'
primary=path.read_text().split('\n'+heading)[0].rstrip()
x=json.loads((ROOT/'single_label42.json').read_text())['delta_vs_zero']
supp=f"""{heading}

위 pooled cross-fitting은 각 donor 물체에서 여덟 개 offset의 counterfactual event를 모두 사용한다. 실제 수집에 더 가까운 사후 보조 대조에서는 각 donor 물체의 offset을 여덟 값에서 균등하게 한 번만 추출하고, 그때의 event label 한 개만 건넸다. 해당 offset 값은 action 선택에 사용하지 않았다. 환경당 약 64개의 donor event로 하나의 공통 action을 고르고 나머지 prototype에서 여덟 offset의 Brier를 평균했다. 네 분할과 난수 추출은 평균 평가를 위한 반복이며, 한 번의 policy 선택에 사용한 label 예산은 donor당 1회다. 같은 조건에서 zero 대비 Brier 차이는 **{x['mean']:+.6f} [{x['lower']:+.6f}, {x['upper']:+.6f}]**였다. 이는 zero와 동일한 label-free·4-contact 예산을 쓴 운영 방법의 이득으로 해석할 수 없다. Target label을 일부 획득하면 environment 간 보정 순위를 학습할 수 있다는 진단이다. [계산 결과](single_label42.json) · [재현 코드](single_label42.py)
"""
path.write_text(primary+'\n\n'+supp)
