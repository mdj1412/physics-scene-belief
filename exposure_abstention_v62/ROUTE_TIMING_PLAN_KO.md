# v62 · 같은 후속 계산 경로에서의 latency 감사

처음 몇 catalog의 raw replay가 통과한 후, 전체 p90을 열기 전에 정한 secondary audit이다. Primary는 96개 block 전체의 p90이다. 각 paired block에서 selective reset과 v61 reset·v61 sham·상시 gate의 행동이 실제로 얼마나 달랐는지 확인한다. 두 surface의 outer 행동 일치, 네 deployment session의 최종 R/I 구매 일치, 둘 다 일치한 경우를 각각 센다. 양쪽의 outer와 최종 선택이 모두 같은 block에서 각 방법의 p90 및 paired median difference를 구한다. 이는 사후 분할한 descriptive estimate이며 주 endpoint를 대체하지 않는다. 같은 terminal 경로에서도 raw reader는 실제 mode대로 실행한다.
