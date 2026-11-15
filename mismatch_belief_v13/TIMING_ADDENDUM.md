# Rare-branch timing 보완

Primary fixed-case timing48 cases×3에서 primary gate의 selected noise repeat0가 모두 unflagged였다. 전체 test의 expansion rate는 약3%여서 fixed timing만으로 expansion branch 비용을 설명할 수 없다.

Accuracy/policy/threshold는 변경하지 않는다. 각 condition의 flag/nonflag event strata에서 seed20261114로4 events씩 뽑아3회 timing하고 두 branch median을 각각 보고한다. Condition별 실제 flag fraction으로 conditional means를 가중한 estimated population mean도 제공한다. This is a post-primary timing diagnostic이며 primary median이나 hypothesis tests를 대체하지 않는다. CPU/GPU 모두 같은 events를 사용한다. Target truth를 event 선택에 사용하지 않는다.
