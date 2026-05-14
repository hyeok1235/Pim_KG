# PIM Knowledge Graph Traversal

PIM(Processing-In-Memory) 환경에서 Knowledge Graph 탐색을 수행하는 연구 프로젝트.
[SAITPublic PIMSimulator](https://github.com/SAITPublic/PIMSimulator) 위에서 CSR 기반 그래프 탐색을 구현하고, CPU 대비 성능을 측정한다.

## 핵심 아이디어

기존 CPU-DRAM 구조에서 KG 탐색은 매 hop마다 CPU↔DRAM 간 왕복이 발생한다 (memory wall).
PIM을 활용하면 메모리 내부에서 이웃 노드 수집·연산을 일괄 처리하여 왕복 횟수를 N회 → 1회로 줄일 수 있다.
CSR 형식으로 저장된 그래프는 한 노드의 이웃이 연속 메모리에 위치하므로 PIM SIMD 연산에 적합하다.

## 환경

| 항목 | 내용 |
|------|------|
| Python | 3.10 (conda `pim` 환경) |
| 빌드 도구 | `scons` |
| 빌드 표준 | C++17 |
| 시뮬레이터 | `googletest/build/PIMSimulator/` |

## 빌드 & 실행

```bash
conda activate pim
cd googletest/build/PIMSimulator
scons

# 동작 확인
./sim --gtest_filter=PIMKernelFixture.add
./sim --gtest_filter=PIMKernelFixture.gemv
./sim --gtest_filter=PIMBenchFixture.gemv
./sim --gtest_filter=MemBandwidthFixture.hbm_read_bandwidth
```
