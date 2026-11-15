# Supplementary CPU timing

Primary GPU benchmark에서 compact의 warm latency가 QMC보다 느리다는 결과를 본 뒤 추가한 hardware audit이다. Primary accuracy comparisons와 GPU screening 결론은 바꾸지 않는다.

같은 code를 CPU float64, PyTorch2 threads로 실행하고 같은16 cases,3 repeats를 사용한다. CPU methods끼리 비교하며 compact CPU와 baseline GPU를 섞어서 speedup을 주장하지 않는다. Serial CPU predictions를 저장된 native GPU predictions와 비교한다. 이 단계는 test outcome에 맞춘 model/hyperparameter 변경이 아니다.
