# Experiment Results

로드맵 기준 각 Phase의 측정 수치와 해석을 누적 기록한다.

---

## Phase 0: 환경 구축 ✅

| 항목 | 결과 |
|---|---|
| Conda 환경 | `pim` (Python 3.10) |
| PIMSimulator 빌드 | 성공 (`scons`) |
| `PIMKernelFixture.add` | PASSED |
| `PIMKernelFixture.gemv` | PASSED |
| `PIMBenchFixture.gemv` | PASSED |
| `MemBandwidthFixture.hbm_read_bandwidth` | PASSED |

---

## Phase 1: FB15K-237 기초 그래프 탐색 ✅

### 데이터셋 통계

| 항목 | 값 |
|---|---|
| 엔터티 수 | 14,541 |
| 관계(relation) 수 | 237 |
| train 트리플 수 | 272,115 |
| validation 트리플 수 | 17,535 |
| test 트리플 수 | 20,466 |

### 차수(Degree) 분포

| 항목 | 값 |
|---|---|
| 평균 차수 | 18.71 |
| 최대 차수 | 1,325 (노드 10416) |
| 최소 차수 | 0 |
| 차수 = 0 노드 수 | 760 |

### 메모리 접근 패턴 (1-hop vs 2-hop)

1-hop 접근 횟수 = `2 (row_ptr) + degree × 2 (col_idx + rel_idx)`

| 노드 | 차수 | 1-hop 접근 | 2-hop 접근 | 증폭 배수 |
|---|---:|---:|---:|---:|
| 0 (저차수) | 6 | 14 | 116 | 8.3x |
| 3560 (중차수) | 13 | 28 | 856 | 30.6x |
| 10416 (고차수) | 1,325 | 2,652 | 47,984 | 18.1x |

> 중차수 노드에서 2-hop 접근이 30배 이상 폭증 — CPU-DRAM memory wall의 정량적 근거.

---

## Phase 2: SB 모드 1-hop 기준선 ✅

HBM 설정: `system_hbm_1ch.ini` (1채널), `HBM2_samsung_2M_16B_x64.ini`

### 차수별 측정 결과

| 차수 | 총 트랜잭션 | 총 사이클 | 사이클/트랜잭션 |
|---:|---:|---:|---:|
| 6 (저차수, node 0) | 14 | 151 | 10.79 |
| 13 (중차수, node 3560) | 28 | 241 | 8.61 |
| 18 (평균차수) | 38 | 281 | 7.39 |
| 50 | 102 | 591 | 5.79 |
| 100 (고차수 proxy) | 202 | 1,103 | 5.46 |

> 차수 증가에 따라 사이클/트랜잭션 감소(10.79 → 5.46): row buffer hit 비율 증가 효과.
> FB15K-237 평균차수 노드(d=18) 기준 **281 사이클** — Phase 4 PIM 커널과의 비교 기준점.

---

## Phase 3: PIM 친화적 데이터 재구성 ✅

HBM 설정: `system_hbm_1ch.ini` (1채널), NUM_PIM_BLOCKS=8

### 배치 전략 A: NEIGHBOR_SPREAD — bank 분포 (8 nodes × 16 edges)

| block | edges |
|---:|---:|
| 0~7 (각) | 16 |

모든 블록에 완전 균등 분산. 고차수 노드에서 bank-level parallelism 극대화에 적합.

### 배치 전략 B: NODE_LOCAL — bank 분포 (16 nodes × 4 edges)

| block | edges |
|---:|---:|
| 0~7 (각) | 8 |

대칭 그래프에서 완전 균등 분산. 저차수 노드에서 row-buffer hit 극대화에 적합.

### 전략 비교 — 비대칭 그래프 (node 0, degree=3)

| block | NEIGHBOR_SPREAD | NODE_LOCAL |
|---:|---:|---:|
| 0 | 1 | 3 |
| 1 | 1 | 0 |
| 2 | 1 | 0 |
| 3..7 | 0 | 0 |

> degree < NUM_PIM_BLOCKS이면 NEIGHBOR_SPREAD도 일부 블록만 사용됨 — 저차수 노드에서는 NODE_LOCAL이 row-buffer hit 면에서 유리.

### Round-trip 검증

| 전략 | row_ptr | col_idx | rel_idx | mismatches |
|---|---:|---:|---:|---:|
| NEIGHBOR_SPREAD | 9 entries | 128 entries | 128 entries | 0 |
| NODE_LOCAL | 17 entries | 64 entries | 64 entries | 0 |

---

## Phase 4: PIM–CPU 역할 분담 ✅

HBM 설정: `system_hbm_1ch.ini` (1채널), NUM_PIM_BLOCKS=8

### 1-hop 스케일링 (NEIGHBOR_SPREAD)

| 차수 | CPU txns | CPU cycles | PIM cycles | Speedup |
|---:|---:|---:|---:|---:|
| 5 | 12 | 142 | 675 | 0.21x |
| 10 | 22 | 187 | 690 | 0.27x |
| 25 | 52 | 337 | 798 | 0.42x |
| 50 | 102 | 591 | 988 | 0.60x |
| 100 | 202 | 1,103 | 1,280 | 0.86x |
| **200** | **402** | **2,425** | **1,875** | **1.29x ← crossover** |
| 500 | 1,002 | 5,311 | 4,478 | 1.19x |
| 1,000 | 2,002 | 10,912 | 7,576 | 1.44x |

> crossover point: **d ≈ 200** (1.29x). d=100→200 구간에서 PIM이 CPU를 역전.
> d=500에서 speedup이 1.19x로 소폭 하락하다가 d=1000에서 1.44x로 회복 — PIM 내부 pipeline
> saturation 과정으로 추정. FB15K-237 평균 차수(18.71)에서는 PIM이 불리하지만, 고차수 허브
> 노드(최대 1325)에서는 PIM이 유효. 논문 메시지: "d ≥ 200인 허브 노드에서 PIM gather가 유효".

### 배치 전략 비교 (NEIGHBOR_SPREAD vs NODE_LOCAL)

| 차수 | SPREAD cycles | LOCAL cycles | 승자 |
|---:|---:|---:|:---:|
| 5 | 675 | 853 | SPREAD |
| 10 | 690 | 1,059 | SPREAD |
| 25 | 798 | 1,850 | SPREAD |
| 50 | 988 | 3,059 | SPREAD |
| 100 | 1,280 | 8,668 | SPREAD |
| 200 | 1,875 | 11,606 | SPREAD |
| 500 | 4,478 | 45,696 | SPREAD |
| 1,000 | 7,576 | 55,452 | SPREAD |

> NODE_LOCAL이 전 차수에서 열세. 차수가 높아질수록 격차가 급격히 벌어짐(d=1000에서 7.3배).
> NODE_LOCAL은 1개 bank에 집중 저장 → 순차 접근으로 bank-level parallelism 미활용.
> NEIGHBOR_SPREAD 전략 채택 확정.

### 2-hop 측정 결과 (NEIGHBOR_SPREAD)

| d1 | d2 | CPU cycles | PIM cycles | Speedup |
|---:|---:|---:|---:|---:|
| 10 | 10 | 2,555 | 6,419 | 0.40x |
| 10 | 50 | 6,355 | 8,580 | 0.74x |
| **10** | **100** | **11,297** | **10,492** | **1.08x ← crossover** |
| 50 | 10 | 12,342 | 31,487 | 0.39x |
| 50 | 50 | 31,686 | 39,574 | 0.80x |
| **50** | **100** | **73,315** | **48,166** | **1.52x** |
| 100 | 10 | 60,057 | 61,848 | 0.97x |
| 100 | 50 | 63,332 | 78,304 | 0.81x |
| **100** | **100** | **111,288** | **96,319** | **1.16x** |
| 200 | 10 | 116,714 | 123,117 | 0.95x |
| 200 | 50 | 126,919 | 155,668 | 0.82x |
| **200** | **100** | **222,443** | **189,713** | **1.17x** |

> **d2(2-hop fan-out)가 crossover를 결정**: d2=100일 때 d1에 관계없이 PIM이 우세(1.08x~1.52x).
> d2=50 이하에서는 PIM이 열세 — 2-hop neighbor가 적으면 PIM 고정 오버헤드를 상쇄 못함.
> d1(1-hop fan-out)의 영향은 상대적으로 작음: d2=100 고정 시 d1=10→1.08x, d1=50→1.52x, d1=200→1.17x.
> 최대 speedup: d1=50, d2=100에서 **1.52x**.
