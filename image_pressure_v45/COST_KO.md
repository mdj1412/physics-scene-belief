## 4. 같은 GPU에서의 배포 비용

서버의 네 GPU allocation 중 다른 process·점유 메모리가 없던 두 장에서 예측을 나눠 계산했다. Latency는 한 RTX PRO 6000 Blackwell에서 512개 물체, 1회 sensor draw, 같은 source bank로 비교했다. Visual은 4,718,592 byte의 uint8 frame을 host에서 GPU로 전송하고 영상 해석과 route 선택 후 **각 물체에서 선택된 하나의 decoder만** 계산했다. 각 방법에는 K256 선택, paid sensor routing, physics posterior, GPU 동기화가 포함된다. Actual image acquisition·robot reset·offline rendering·target simulation은 별도 비용이다. [측정 코드와 input 범위](GPU_KO.md)

| 정책 | Primary held-out Brier ↓ | 첫 측정 GPU p90 (ms) | CPU 배치를 통제한 반복 GPU p90 (ms) [95% CI] |
|---|---:|---:|---:|
| Always-original | 0.150023 | 769.38 | **465.19** [459.79, 467.10] |
| Oracle 위치 | 0.149622 | 735.37 | **465.69** [462.43, 465.85] |
| Visual 위치 | 0.149655 | 785.50 | **468.19** [465.69, 481.65] |
| Source-matched oracle | 0.149635 | 737.89 | 464.42 [462.41, 465.46] |
| Source-matched visual | 0.149664 | 654.04 | 467.75 [466.12, 478.36] |

첫 crossover에서는 일부 panel에서 모든 경로에 큰 지연이 있었고 visual−original p90 차이의 paired 95% CI가 −101.00~+262.44 ms였다. 같은 GPU·같은 순서·같은 입력에서 CPU core를 고정하고 다른 v45 계산을 해당 node 밖으로 옮겨 재측정했다. 각 crossover에서 정책당 warm-up 후 36회, panel 단위 paired bootstrap 16,000회로 구간을 추정했다. 통제한 반복의 visual−original은 **+2.99 ms [−0.30, +17.29]**, visual−oracle은 **+2.50 ms [+0.88, +16.06]**였다. 영상 전송·위치 추정·lookup만의 p90은 3.16 ms였다. 작지만 양수인 image 처리 비용과 전체 GPU p90 추정 오차를 함께 고려해야 한다. 최초 측정의 변동 원인을 특정 process에 귀속하지 않았다. [양쪽 측정 전체](timing_comparison45.json) · [GPU 반환 기록](gpu_release45.log)
