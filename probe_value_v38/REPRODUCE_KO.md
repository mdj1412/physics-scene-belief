## 서버 실행과 재현 자료

서버 정본은 datacenter3의 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/probe_value_v38`에 있다. 부모 디렉터리의 기존 physics-scene-belief 환경과 v36/v37 source를 함께 사용했다. 36개 source·실행 파일의 hash를 [run manifest](results/run.json)에 기록했다. 실험 전 GPU process와 device 사용량을 확인했고, NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장(node55, allocation 2744468)에서 primary와 보조 timing을 차례로 실행했다. 다른 GPU에 학습·추론을 배치하지 않았다. 두 GPU worker 모두 외부 process 없이 정상 종료했다. CPU reference와 검산은 별도 계산 노드 node56의 allocation 2744471에서 CUDA를 비활성화해 진행했다. [Primary 종료](monitor/results/worker_exit.json) · [보조 timing 종료](monitor_continuous/results/worker_exit.json)

동일 project의 빈 할당 GPU에서 주 실험을 시작하는 명령은 아래와 같다. 기존 완료 block은 hash가 일치할 때만 사용한다. 새로운 독립 재현에서는 생성물 없는 작업 사본을 사용한다.

```bash
srun --overlap --jobid="$GPU_ALLOCATION" -n1 bash run38.sh run38.py
```

완료된 결과 디렉터리에는 cell별 seed block 8개, block마다 공통 관측 archive 두 개와 실제 regime·orientation별 archive 여섯 개가 있다. 공통 archive의 deployment sensor 배열은 관측이 선택된 물체의 값만 저장한다. 각 block의 meta에는 SHA-256, calibration·deployment seed, 데이터 생성에 사용한 첫 방향이 들어 있다. [GPU runner](run38.py) · [완료 기록](results/completion.json) · [policy source](POLICY_KO.md)

다음은 할당된 서버 CPU에서 실행하는 계산이다. 상대 경로는 v38 작업 디렉터리 기준이다. `marginal38.py`의 인자 없는 실행은 첫 block만 평가한다. 전체 cell-blind 대조는 각 cell의 여덟 block, 첫 noise repeat를 사용한다. 부가적인 GPU time comparison은 주 worker가 종료된 뒤 별도 monitor directory에서 같은 GPU guard로 continuous38.py를 실행했다. GPU는 보조 session에서도 각 방법의 end-to-end 출력을 raw prediction과 대조했다.

```bash
export CUDA_VISIBLE_DEVICES='' BELIEF_DEVICE=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=../continuous_v2/.venv/bin/python
"$py" oracle38.py
"$py" stats38.py
"$py" oracle_stats38.py
mkdir -p marginal_blocks
for b in 000 001 002 003 004 005 006 007; do
  "$py" marginal38.py --block "block_$b" --output "marginal_blocks/block_$b.json"
done
"$py" marginal_stats38.py
"$py" continuous_stats38.py
```

독립 CPU 구현으로 32개 block의 192개 regime·orientation archive를 검산했다. 512개 calibration·deployment seed가 달랐고, archive의 평균 Brier를 재계산한 최대 오차는 5.55×10⁻¹⁷이었다. 기존 physics decoder의 일부 raw probability가 1을 최대 1.8×10⁻¹⁵ 초과했다. 비교에는 원래의 부동소수점 출력을 공통으로 사용했다. 별도의 물리 검산은 3,932,160개 contact state·조건을 확인했다. 실제 label과 stick/slip 판정은 모든 block에서 일치했고, 기록된 sensor를 재생성한 최대 오차는 7.22×10⁻¹⁶이었다. [정책 검산](audit38.json) · [물리 검산](physical_summary.json) · [숨은 cell 추론](marginal38.py)
