"""
Phase 1: FB15K-237 데이터 준비, CSR 변환, 1-hop/2-hop 구현, 접근 패턴 프로파일링

Usage:
    conda activate pim
    python phase1_fb15k.py
"""

from datasets import load_dataset
import numpy as np
from collections import defaultdict

# ── 1-1. 데이터 다운로드 ──────────────────────────────────────────────────────

print("=== 1-1. FB15K-237 다운로드 ===")
ds = load_dataset("KGraph/FB15k-237")
print(ds)

# 포맷 확인: 'head\trelation\ttail' 탭 구분 text 컬럼
train_split = ds["train"]
print(f"샘플 첫 행: {train_split[0]['text']}")

# ── 엔터티/관계 ID 매핑 ────────────────────────────────────────────────────────

def parse_row(row):
    """'head\trelation\ttail' 형식의 text 컬럼을 파싱"""
    parts = row['text'].split('\t')
    return parts[0], parts[1], parts[2]

def build_id_maps(dataset):
    """모든 split에서 엔터티와 관계를 수집해 정수 ID 매핑 생성"""
    entities = set()
    relations = set()
    for split in dataset.values():
        for row in split:
            h, r, t = parse_row(row)
            entities.add(h)
            entities.add(t)
            relations.add(r)
    ent2id = {e: i for i, e in enumerate(sorted(entities))}
    rel2id = {r: i for i, r in enumerate(sorted(relations))}
    return ent2id, rel2id

print("\n=== 엔터티/관계 ID 매핑 생성 ===")
ent2id, rel2id = build_id_maps(ds)
num_entities = len(ent2id)
num_relations = len(rel2id)
print(f"엔터티 수: {num_entities}")
print(f"관계 수:   {num_relations}")

def rows_to_triples(split_data, ent2id, rel2id):
    triples = []
    for row in split_data:
        h_str, r_str, t_str = parse_row(row)
        triples.append((ent2id[h_str], rel2id[r_str], ent2id[t_str]))
    return triples

print("\n=== 트리플 변환 ===")
train_triples = rows_to_triples(ds["train"], ent2id, rel2id)
print(f"train 트리플 수: {len(train_triples)}")
print(f"샘플: {train_triples[:3]}")

# ── 1-2. CSR 변환 및 1-hop/2-hop 구현 ─────────────────────────────────────────

def build_csr(triples, num_entities):
    """트리플 리스트 → CSR adjacency 구조"""
    adj = defaultdict(list)
    for h, r, t in triples:
        adj[h].append((t, r))

    row_ptr = [0]
    col_idx = []
    rel_idx = []
    for i in range(num_entities):
        neighbors = adj.get(i, [])
        col_idx.extend([n for n, r in neighbors])
        rel_idx.extend([r for n, r in neighbors])
        row_ptr.append(len(col_idx))

    return np.array(row_ptr, dtype=np.int32), np.array(col_idx, dtype=np.int32), np.array(rel_idx, dtype=np.int32)

print("\n=== CSR 변환 ===")
row_ptr, col_idx, rel_idx = build_csr(train_triples, num_entities)
print(f"row_ptr shape: {row_ptr.shape}  (num_entities+1 = {num_entities+1})")
print(f"col_idx shape: {col_idx.shape}  (num_edges = {len(train_triples)})")
print(f"rel_idx shape: {rel_idx.shape}")

def one_hop(source, row_ptr, col_idx, rel_idx, relation_filter=None):
    start, end = row_ptr[source], row_ptr[source + 1]
    neighbors = col_idx[start:end]
    relations = rel_idx[start:end]
    if relation_filter is not None:
        mask = relations == relation_filter
        return neighbors[mask], relations[mask]
    return neighbors, relations

def two_hop(source, row_ptr, col_idx, rel_idx, relation_filter=None):
    hop1_nodes, hop1_rels = one_hop(source, row_ptr, col_idx, rel_idx, relation_filter)
    results = {}
    for node in hop1_nodes:
        hop2_nodes, hop2_rels = one_hop(node, row_ptr, col_idx, rel_idx)
        results[int(node)] = list(zip(hop2_nodes.tolist(), hop2_rels.tolist()))
    return results

# 동작 확인: 차수가 가장 큰 노드로 테스트
degrees = row_ptr[1:] - row_ptr[:-1]
high_deg_node = int(np.argmax(degrees))
print(f"\n=== 1-hop/2-hop 동작 확인 (최고차수 노드 {high_deg_node}, degree={degrees[high_deg_node]}) ===")
n1, r1 = one_hop(high_deg_node, row_ptr, col_idx, rel_idx)
print(f"1-hop 이웃 수: {len(n1)}")
n2 = two_hop(high_deg_node, row_ptr, col_idx, rel_idx)
total_2hop = sum(len(v) for v in n2.values())
print(f"2-hop 고유 중간 노드 수: {len(n2)}, 총 2-hop 엣지 수: {total_2hop}")

# ── 1-3. 메모리 접근 패턴 프로파일링 ──────────────────────────────────────────

def profile_memory_access(source, row_ptr, col_idx):
    start, end = row_ptr[source], row_ptr[source + 1]
    degree = int(end - start)
    return {
        'degree': degree,
        'total_accesses': 2 + degree * 2,   # row_ptr 2회 + col_idx/rel_idx 각 degree회
        'sequential_reads': degree * 2,
        'random_reads': 2,
    }

def profile_two_hop(source, row_ptr, col_idx):
    hop1_nodes, _ = one_hop(source, row_ptr, col_idx, rel_idx)
    total = profile_memory_access(source, row_ptr, col_idx)
    for node in hop1_nodes:
        p = profile_memory_access(int(node), row_ptr, col_idx)
        total['total_accesses'] += p['total_accesses']
        total['sequential_reads'] += p['sequential_reads']
        total['random_reads'] += p['random_reads']
    return total

print("\n=== 1-3. 접근 패턴 통계 ===")
print(f"{'항목':<30} {'값':>15}")
print("-" * 47)
print(f"{'평균 차수':<30} {degrees.mean():>15.2f}")
print(f"{'최대 차수':<30} {degrees.max():>15d}")
print(f"{'최소 차수':<30} {degrees.min():>15d}")
print(f"{'차수=0 노드 수':<30} {(degrees == 0).sum():>15d}")

avg_1hop_access = 2 + degrees.mean() * 2
print(f"\n{'1-hop 평균 메모리 접근 횟수':<30} {avg_1hop_access:>15.2f}")

# 샘플 노드들로 2-hop 접근 통계
sample_nodes = [
    int(np.where(degrees > 0)[0][0]),                           # 저차수 (최소 비영)
    int(np.argsort(degrees)[len(degrees) // 2]),                # 중앙값 차수
    high_deg_node,                                               # 최고차수
]
print(f"\n{'노드':<10} {'차수':>8} {'1-hop 접근':>12} {'2-hop 접근':>12} {'증폭배수':>10}")
print("-" * 55)
for node in sample_nodes:
    p1 = profile_memory_access(node, row_ptr, col_idx)
    p2 = profile_two_hop(node, row_ptr, col_idx)
    ratio = p2['total_accesses'] / max(p1['total_accesses'], 1)
    print(f"{node:<10} {p1['degree']:>8} {p1['total_accesses']:>12} {p2['total_accesses']:>12} {ratio:>10.1f}x")

print("\n완료. 이 결과가 Phase 2 (SB 모드 기준선)의 정량적 근거가 됩니다.")
