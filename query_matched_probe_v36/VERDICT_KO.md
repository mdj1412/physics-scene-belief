**네 번째 접촉을 query 근처로 옮겼을 때, 표면의 방향 효과를 판독하는 AUROC는 0.500에서 0.524로 변했다** (strong 대 weak/null, 네 측정의 paired 95% interval 0.501~0.546). 실제 보정을 켜는 binary gate를 같은 재료 구분 기준에 대입했을 때 balanced accuracy는 51.7%였다.

**네 번째 record를 보정 판단에 추가한 paired Brier 차이는 -0.00129 [-0.00186, -0.00072]**였다. 이 interval은 사전에 정한 0.003 이상의 실용적 Brier 감소를 지지하지 않는다. 세 local regime을 균등 가중했을 때 네 측정 gate의 held-out Brier는 0.11164, conventional fixed 4에서 원래 physics decoder는 0.11299, v35 correction은 0.11534였다. 이 두 end-to-end 차이에는 네 번째 동작과 방향 reader 변화도 들어간다.

Weak 표면의 Gate4−v35는 -0.00536 [-0.00627, -0.00444], query 근처에서 방향성이 사라진 local null은 -0.00940 [-0.01034, -0.00847], 표면 전체에 방향성이 없는 global null은 -0.00669 [-0.00823, -0.00515]였다. Strong 표면의 Gate4−v35는 +0.00368 [+0.00229, +0.00506]였다. Local null과 global null에서 원래 physics와의 차이는 각각 +0.00009 [-0.00083, +0.00101], -0.00027 [-0.00120, +0.00065]로 두 interval 모두 0을 포함했다. 이 실험은 v35의 geometry·response family에서 새로 생성한 synthetic worlds를 사용했다.
