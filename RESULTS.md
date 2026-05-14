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
