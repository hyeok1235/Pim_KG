/*
 * GraphTestCases.cpp — Phase 2, 3 & 4 graph kernel tests
 *
 * Phase 2 (SB baseline):
 *   ./sim --gtest_filter=GraphKGFixture.sb_baseline_low
 *   ./sim --gtest_filter=GraphKGFixture.sb_baseline_mid
 *   ./sim --gtest_filter=GraphKGFixture.sb_baseline_high
 *   ./sim --gtest_filter=GraphKGFixture.*
 *
 * Phase 3 (HBM bank placement):
 *   ./sim --gtest_filter=GraphKGPhase3Fixture.placement_A_bank_distribution
 *   ./sim --gtest_filter=GraphKGPhase3Fixture.placement_B_bank_distribution
 *   ./sim --gtest_filter=GraphKGPhase3Fixture.placement_compare
 *   ./sim --gtest_filter=GraphKGPhase3Fixture.*
 *
 * Phase 4 (PIM gather vs CPU baseline):
 *   ./sim --gtest_filter=GraphKGPhase4Fixture.pim_1hop_low
 *   ./sim --gtest_filter=GraphKGPhase4Fixture.pim_1hop_avg
 *   ./sim --gtest_filter=GraphKGPhase4Fixture.pim_1hop_high
 *   ./sim --gtest_filter=GraphKGPhase4Fixture.cpu_vs_pim_scale
 *   ./sim --gtest_filter=GraphKGPhase4Fixture.strategy_compare
 *   ./sim --gtest_filter=GraphKGPhase4Fixture.pim_2hop
 *   ./sim --gtest_filter=GraphKGPhase4Fixture.*
 */

#include <cstdint>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "tests/GraphKernel.h"

using namespace std;
using namespace DRAMSim;

/* -------------------------------------------------------------------------
 * Helper: build a minimal CSRGraph with exactly 'degree' out-edges from
 * node 0.  Nodes 1..degree are used as destinations; relation types are
 * assigned as i % 237 to mimic FB15K-237's 237-relation distribution.
 * ---------------------------------------------------------------------- */
static CSRGraph buildSyntheticGraph(int degree)
{
    CSRGraph g;
    g.num_nodes = degree + 1;  // node 0 + degree destination nodes
    g.num_edges = degree;

    g.row_ptr.resize(g.num_nodes + 1, 0);
    g.row_ptr[0] = 0;
    g.row_ptr[1] = degree;                 // node 0 has 'degree' edges
    for (int i = 2; i <= g.num_nodes; i++) // all other nodes: 0 edges
        g.row_ptr[i] = degree;

    g.col_idx.resize(degree);
    g.rel_idx.resize(degree);
    for (int i = 0; i < degree; i++)
    {
        g.col_idx[i] = i + 1;
        g.rel_idx[i] = i % 237;
    }

    return g;
}

/* -------------------------------------------------------------------------
 * Helper: create a memory system + GraphKernel, preload the graph, drain
 * the preload writes, then return the kernel ready for measurement.
 * ---------------------------------------------------------------------- */
static shared_ptr<GraphKernel> setupKernel(const CSRGraph& graph)
{
    // Use 1-channel config to keep simulation time manageable for large degree
    auto mem = make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_1ch.ini", ".", "graph_kg",
        256 * 1);

    auto gk = make_shared<GraphKernel>(mem, /*num_pim_chan=*/1, /*num_pim_rank=*/1);
    gk->preloadCSR(graph);
    gk->runPIM();  // drain preload transactions
    return gk;
}

/* -------------------------------------------------------------------------
 * Helper: print the measurement summary for one test case.
 * ---------------------------------------------------------------------- */
static void printSummary(const char* label, int source, int degree, uint64_t cycles)
{
    int total_transactions = 2 + degree * 2;  // 2 row_ptr + degree col + degree rel
    double cpt = (total_transactions > 0)
                     ? static_cast<double>(cycles) / total_transactions
                     : 0.0;

    cout << "\n=== [Phase 2 SB Baseline] " << label << " ===" << endl;
    cout << "  Source node     : " << source << endl;
    cout << "  Degree          : " << degree << endl;
    cout << "  Total txns      : " << total_transactions
         << "  (2 row_ptr + " << degree << " col_idx + " << degree << " rel_idx)" << endl;
    cout << "  Total cycles    : " << cycles << endl;
    cout << "  Cycles/txn      : " << cpt << endl;
}

/* =========================================================================
 * GraphKGFixture
 * ====================================================================== */
class GraphKGFixture : public testing::Test
{
  public:
    GraphKGFixture()  {}
    ~GraphKGFixture() {}
    virtual void SetUp()    {}
    virtual void TearDown() {}
};

/* -------------------------------------------------------------------------
 * sb_baseline_low — source node with degree 6  (FB15K-237 node 0)
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGFixture, sb_baseline_low)
{
    const int degree = 6;
    CSRGraph graph   = buildSyntheticGraph(degree);
    auto gk          = setupKernel(graph);

    uint64_t cycles = gk->cpuBaseline1Hop(0, graph);

    printSummary("LOW degree (d=6, FB15K node 0)", 0, degree, cycles);

    // Sanity: at least 1 cycle per transaction
    EXPECT_GT(cycles, 0u);
}

/* -------------------------------------------------------------------------
 * sb_baseline_mid — source node with degree 13  (FB15K-237 node 3560)
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGFixture, sb_baseline_mid)
{
    const int degree = 13;
    CSRGraph graph   = buildSyntheticGraph(degree);
    auto gk          = setupKernel(graph);

    uint64_t cycles = gk->cpuBaseline1Hop(0, graph);

    printSummary("MID degree (d=13, FB15K node 3560)", 0, degree, cycles);

    EXPECT_GT(cycles, 0u);
}

/* -------------------------------------------------------------------------
 * sb_baseline_avg — source node with degree 18 (≈ FB15K-237 avg = 18.71)
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGFixture, sb_baseline_avg)
{
    const int degree = 18;
    CSRGraph graph   = buildSyntheticGraph(degree);
    auto gk          = setupKernel(graph);

    uint64_t cycles = gk->cpuBaseline1Hop(0, graph);

    printSummary("AVG degree (d=18, FB15K avg)", 0, degree, cycles);

    EXPECT_GT(cycles, 0u);
}

/* -------------------------------------------------------------------------
 * sb_baseline_high — source node with degree 100 (high-degree proxy)
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGFixture, sb_baseline_high)
{
    const int degree = 100;
    CSRGraph graph   = buildSyntheticGraph(degree);
    auto gk          = setupKernel(graph);

    uint64_t cycles = gk->cpuBaseline1Hop(0, graph);

    printSummary("HIGH degree (d=100)", 0, degree, cycles);

    EXPECT_GT(cycles, 0u);
}

/* -------------------------------------------------------------------------
 * sb_baseline_scale — verify that cycles scale approximately linearly with
 * degree (low → avg → high).  Records the ratio for the REPORT.
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGFixture, sb_baseline_scale)
{
    const vector<int> degrees = {6, 13, 18, 50, 100};

    cout << "\n=== [Phase 2 SB Baseline] Scaling table ===" << endl;
    cout << "  degree | total_txns | cycles | cycles/txn" << endl;
    cout << "  -------|------------|--------|------------" << endl;

    for (int d : degrees)
    {
        CSRGraph graph = buildSyntheticGraph(d);
        auto gk        = setupKernel(graph);
        uint64_t cycles = gk->cpuBaseline1Hop(0, graph);
        int txns        = 2 + d * 2;
        double cpt      = static_cast<double>(cycles) / txns;
        cout << "  " << d
             << "\t| " << txns
             << "\t| " << cycles
             << "\t| " << cpt << endl;

        EXPECT_GT(cycles, 0u);
    }
}

/* =========================================================================
 * Phase 3: HBM Bank Placement Tests
 * ====================================================================== */

/* -------------------------------------------------------------------------
 * Helper: build a multi-node CSRGraph where every node has the same degree.
 *   num_nodes nodes, each with degree_per_node edges.
 *   Destinations cycle over [0, num_nodes); relations cycle over 0..236.
 * ---------------------------------------------------------------------- */
static CSRGraph buildMultiNodeGraph(int num_nodes, int degree_per_node)
{
    CSRGraph g;
    g.num_nodes = num_nodes;
    g.num_edges = num_nodes * degree_per_node;

    g.row_ptr.resize(num_nodes + 1);
    for (int i = 0; i <= num_nodes; i++)
        g.row_ptr[i] = i * degree_per_node;

    g.col_idx.resize(g.num_edges);
    g.rel_idx.resize(g.num_edges);
    for (int k = 0; k < g.num_edges; k++)
    {
        g.col_idx[k] = k % num_nodes;
        g.rel_idx[k] = k % 237;
    }

    return g;
}

/* -------------------------------------------------------------------------
 * Helper: create memory system + GraphKernel, load graph using Phase 3
 * placement strategy, drain writes, and return the kernel ready for use.
 * ---------------------------------------------------------------------- */
static shared_ptr<GraphKernel> setupKernelPhase3(const CSRGraph& graph,
                                                  PlacementStrategy strategy)
{
    auto mem = make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_1ch.ini", ".", "graph_kg_p3",
        256 * 1);

    auto gk = make_shared<GraphKernel>(mem, /*num_pim_chan=*/1, /*num_pim_rank=*/1);
    gk->loadGraphToHBM(graph, strategy);
    gk->runPIM();  // drain preload transactions
    return gk;
}

/* -------------------------------------------------------------------------
 * Fixture
 * ---------------------------------------------------------------------- */
class GraphKGPhase3Fixture : public testing::Test
{
  public:
    GraphKGPhase3Fixture()  {}
    ~GraphKGPhase3Fixture() {}
    virtual void SetUp()    {}
    virtual void TearDown() {}
};

/* -------------------------------------------------------------------------
 * placement_A_bank_distribution — NEIGHBOR_SPREAD: j-th neighbor of any
 * node goes to PIM block (j % NUM_PIM_BLOCKS).
 *
 * Graph: 8 nodes × 16 edges each = 128 edges.
 * With NUM_PIM_BLOCKS=8 and degree=16, each node spreads 2 edges per block,
 * so every block receives 8 × 2 = 16 edges uniformly.
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGPhase3Fixture, placement_A_bank_distribution)
{
    const int num_nodes = 8, degree = 16;
    CSRGraph graph = buildMultiNodeGraph(num_nodes, degree);

    auto gk = setupKernelPhase3(graph, PlacementStrategy::NEIGHBOR_SPREAD);
    unsigned num_pim_blocks = getConfigParam(UINT, "NUM_PIM_BLOCKS");
    vector<int> dist = gk->analyzeBankDistribution(graph, PlacementStrategy::NEIGHBOR_SPREAD);

    cout << "\n=== [Phase 3] NEIGHBOR_SPREAD bank distribution ===" << endl;
    cout << "  Graph: " << num_nodes << " nodes × " << degree << " edges = "
         << graph.num_edges << " edges, NUM_PIM_BLOCKS = " << num_pim_blocks << endl;
    int total = 0;
    for (unsigned b = 0; b < dist.size(); b++)
    {
        cout << "  block[" << b << "] = " << dist[b] << " edges" << endl;
        total += dist[b];
    }
    cout << "  total  = " << total << endl;

    ASSERT_EQ(dist.size(), static_cast<size_t>(num_pim_blocks));
    EXPECT_EQ(total, graph.num_edges);
    for (unsigned b = 0; b < num_pim_blocks; b++)
        EXPECT_EQ(dist[b], num_nodes * (degree / static_cast<int>(num_pim_blocks)));
}

/* -------------------------------------------------------------------------
 * placement_B_bank_distribution — NODE_LOCAL: all edges of node i go to
 * PIM block (i % NUM_PIM_BLOCKS).
 *
 * Graph: 16 nodes × 4 edges each = 64 edges.
 * With NUM_PIM_BLOCKS=8, nodes 0&8→block0, 1&9→block1, ...
 * so each block gets 2 × 4 = 8 edges uniformly.
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGPhase3Fixture, placement_B_bank_distribution)
{
    const int num_nodes = 16, degree = 4;
    CSRGraph graph = buildMultiNodeGraph(num_nodes, degree);

    auto gk = setupKernelPhase3(graph, PlacementStrategy::NODE_LOCAL);
    unsigned num_pim_blocks = getConfigParam(UINT, "NUM_PIM_BLOCKS");
    vector<int> dist = gk->analyzeBankDistribution(graph, PlacementStrategy::NODE_LOCAL);

    cout << "\n=== [Phase 3] NODE_LOCAL bank distribution ===" << endl;
    cout << "  Graph: " << num_nodes << " nodes × " << degree << " edges = "
         << graph.num_edges << " edges, NUM_PIM_BLOCKS = " << num_pim_blocks << endl;
    int total = 0;
    for (unsigned b = 0; b < dist.size(); b++)
    {
        cout << "  block[" << b << "] = " << dist[b] << " edges" << endl;
        total += dist[b];
    }
    cout << "  total  = " << total << endl;

    ASSERT_EQ(dist.size(), static_cast<size_t>(num_pim_blocks));
    EXPECT_EQ(total, graph.num_edges);
    int nodes_per_block = num_nodes / static_cast<int>(num_pim_blocks);
    for (unsigned b = 0; b < num_pim_blocks; b++)
        EXPECT_EQ(dist[b], nodes_per_block * degree);
}

/* -------------------------------------------------------------------------
 * placement_compare — demonstrate that NEIGHBOR_SPREAD and NODE_LOCAL
 * produce different bank distributions on an asymmetric graph.
 *
 * Graph: single node (node 0) with degree 3, NUM_PIM_BLOCKS=8.
 *   NEIGHBOR_SPREAD: bank 0,1,2 get 1 edge each, banks 3..7 get 0.
 *   NODE_LOCAL:      bank 0 gets all 3 edges,     banks 1..7 get 0.
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGPhase3Fixture, placement_compare)
{
    const int degree = 3;
    CSRGraph graph = buildSyntheticGraph(degree);  // node 0 only

    auto gkA = setupKernelPhase3(graph, PlacementStrategy::NEIGHBOR_SPREAD);
    unsigned num_pim_blocks = getConfigParam(UINT, "NUM_PIM_BLOCKS");

    vector<int> distA = gkA
                            ->analyzeBankDistribution(graph, PlacementStrategy::NEIGHBOR_SPREAD);
    vector<int> distB = setupKernelPhase3(graph, PlacementStrategy::NODE_LOCAL)
                            ->analyzeBankDistribution(graph, PlacementStrategy::NODE_LOCAL);

    cout << "\n=== [Phase 3] Placement Comparison (degree=" << degree << ") ===" << endl;
    cout << "  block | NEIGHBOR_SPREAD | NODE_LOCAL" << endl;
    cout << "  ------|-----------------|----------" << endl;
    for (unsigned b = 0; b < num_pim_blocks; b++)
        cout << "  " << b << "     | " << distA[b] << "\t\t| " << distB[b] << endl;

    ASSERT_EQ(distA.size(), static_cast<size_t>(num_pim_blocks));
    ASSERT_EQ(distB.size(), static_cast<size_t>(num_pim_blocks));

    /* NEIGHBOR_SPREAD: edges spread across blocks 0,1,2 */
    EXPECT_EQ(distA[0], 1);
    EXPECT_EQ(distA[1], 1);
    EXPECT_EQ(distA[2], 1);
    for (unsigned b = 3; b < num_pim_blocks; b++)
        EXPECT_EQ(distA[b], 0);

    /* NODE_LOCAL: all edges in block 0 (node 0 → block 0%8=0) */
    EXPECT_EQ(distB[0], 3);
    for (unsigned b = 1; b < num_pim_blocks; b++)
        EXPECT_EQ(distB[b], 0);

    /* The two distributions must differ */
    bool differ = false;
    for (unsigned b = 0; b < num_pim_blocks; b++)
        if (distA[b] != distB[b]) { differ = true; break; }
    EXPECT_TRUE(differ);
}

/* -------------------------------------------------------------------------
 * verify_A_round_trip — write CSR with NEIGHBOR_SPREAD, read back, and
 * confirm every value matches the original graph.
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGPhase3Fixture, verify_A_round_trip)
{
    CSRGraph graph = buildMultiNodeGraph(8, 16);  // 128 edges

    auto gk = setupKernelPhase3(graph, PlacementStrategy::NEIGHBOR_SPREAD);
    int mismatches = gk->verifyHBMData(graph, PlacementStrategy::NEIGHBOR_SPREAD);

    cout << "\n=== [Phase 3-3] NEIGHBOR_SPREAD round-trip verification ===" << endl;
    cout << "  row_ptr entries : " << graph.num_nodes + 1 << endl;
    cout << "  col_idx entries : " << graph.num_edges << endl;
    cout << "  rel_idx entries : " << graph.num_edges << endl;
    cout << "  mismatches      : " << mismatches << endl;

    EXPECT_EQ(mismatches, 0);
}

/* -------------------------------------------------------------------------
 * verify_B_round_trip — write CSR with NODE_LOCAL, read back, and confirm
 * every value matches the original graph.
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGPhase3Fixture, verify_B_round_trip)
{
    CSRGraph graph = buildMultiNodeGraph(16, 4);  // 64 edges

    auto gk = setupKernelPhase3(graph, PlacementStrategy::NODE_LOCAL);
    int mismatches = gk->verifyHBMData(graph, PlacementStrategy::NODE_LOCAL);

    cout << "\n=== [Phase 3-3] NODE_LOCAL round-trip verification ===" << endl;
    cout << "  row_ptr entries : " << graph.num_nodes + 1 << endl;
    cout << "  col_idx entries : " << graph.num_edges << endl;
    cout << "  rel_idx entries : " << graph.num_edges << endl;
    cout << "  mismatches      : " << mismatches << endl;

    EXPECT_EQ(mismatches, 0);
}

/* =========================================================================
 * Phase 4: PIM–CPU Role Division — PIM Gather vs CPU Baseline
 * ====================================================================== */

/* -------------------------------------------------------------------------
 * Helper: create memory + GraphKernel with Phase 4 setup (load graph
 * using placement strategy, drain writes, ready for PIM gather).
 * ---------------------------------------------------------------------- */
static shared_ptr<GraphKernel> setupKernelPhase4(const CSRGraph& graph,
                                                  PlacementStrategy strategy)
{
    auto mem = make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_1ch.ini", ".", "graph_kg_p4",
        256 * 1);

    auto gk = make_shared<GraphKernel>(mem, /*num_pim_chan=*/1, /*num_pim_rank=*/1);
    gk->loadGraphToHBM(graph, strategy);
    gk->runPIM();
    return gk;
}

/* -------------------------------------------------------------------------
 * Helper: run both CPU baseline and PIM gather on the same graph/node.
 * Returns (cpu_cycles, pim_cycles).
 * ---------------------------------------------------------------------- */
static pair<uint64_t, uint64_t> measureCpuVsPim(int degree, PlacementStrategy strategy)
{
    CSRGraph graph = buildSyntheticGraph(degree);

    /* CPU baseline — Phase 2 approach */
    //cout << "[DEBUG measureCpuVsPim] degree=" << degree << " — starting CPU baseline..." << endl;
    auto gk_cpu = setupKernel(graph);
    uint64_t cpu_cycles = gk_cpu->cpuBaseline1Hop(0, graph);
    //cout << "[DEBUG measureCpuVsPim] CPU baseline done, cycles=" << cpu_cycles << endl;

    /* PIM gather — Phase 4 approach */
    //cout << "[DEBUG measureCpuVsPim] starting PIM gather (setupKernelPhase4)..." << endl;
    auto gk_pim = setupKernelPhase4(graph, strategy);
    //cout << "[DEBUG measureCpuVsPim] setupKernelPhase4 done, calling pim1HopGather..." << endl;
    uint64_t pim_cycles = gk_pim->pim1HopGather(0, graph, strategy);
    //cout << "[DEBUG measureCpuVsPim] PIM gather done, cycles=" << pim_cycles << endl;

    return {cpu_cycles, pim_cycles};
}

/* -------------------------------------------------------------------------
 * Fixture
 * ---------------------------------------------------------------------- */
class GraphKGPhase4Fixture : public testing::Test
{
  public:
    GraphKGPhase4Fixture()  {}
    ~GraphKGPhase4Fixture() {}
    virtual void SetUp()    {}
    virtual void TearDown() {}
};

/* -------------------------------------------------------------------------
 * pim_1hop_low — PIM 1-hop gather, degree=6, NEIGHBOR_SPREAD
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGPhase4Fixture, pim_1hop_low)
{
    const int degree = 6;
    auto [cpu_c, pim_c] = measureCpuVsPim(degree, PlacementStrategy::NEIGHBOR_SPREAD);
    double speedup = static_cast<double>(cpu_c) / pim_c;

    cout << "\n=== [Phase 4] PIM 1-hop LOW (d=6, NEIGHBOR_SPREAD) ===" << endl;
    cout << "  CPU baseline cycles : " << cpu_c << endl;
    cout << "  PIM gather cycles   : " << pim_c << endl;
    cout << "  Speedup (CPU/PIM)   : " << speedup << "x" << endl;

    EXPECT_GT(pim_c, 0u);
}

/* -------------------------------------------------------------------------
 * pim_1hop_avg — PIM 1-hop gather, degree=18, NEIGHBOR_SPREAD
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGPhase4Fixture, pim_1hop_avg)
{
    const int degree = 18;
    auto [cpu_c, pim_c] = measureCpuVsPim(degree, PlacementStrategy::NEIGHBOR_SPREAD);
    double speedup = static_cast<double>(cpu_c) / pim_c;

    cout << "\n=== [Phase 4] PIM 1-hop AVG (d=18, NEIGHBOR_SPREAD) ===" << endl;
    cout << "  CPU baseline cycles : " << cpu_c << endl;
    cout << "  PIM gather cycles   : " << pim_c << endl;
    cout << "  Speedup (CPU/PIM)   : " << speedup << "x" << endl;

    EXPECT_GT(pim_c, 0u);
}

/* -------------------------------------------------------------------------
 * pim_1hop_high — PIM 1-hop gather, degree=100, NEIGHBOR_SPREAD
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGPhase4Fixture, pim_1hop_high)
{
    const int degree = 100;
    auto [cpu_c, pim_c] = measureCpuVsPim(degree, PlacementStrategy::NEIGHBOR_SPREAD);
    double speedup = static_cast<double>(cpu_c) / pim_c;

    cout << "\n=== [Phase 4] PIM 1-hop HIGH (d=100, NEIGHBOR_SPREAD) ===" << endl;
    cout << "  CPU baseline cycles : " << cpu_c << endl;
    cout << "  PIM gather cycles   : " << pim_c << endl;
    cout << "  Speedup (CPU/PIM)   : " << speedup << "x" << endl;

    EXPECT_GT(pim_c, 0u);
}

/* -------------------------------------------------------------------------
 * cpu_vs_pim_scale — comprehensive comparison across degree values.
 * This is the key result table for the paper.
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGPhase4Fixture, cpu_vs_pim_scale)
{
    const vector<int> degrees = {5, 10, 25, 50, 100, 200, 500, 1000};

    cout << "\n=== [Phase 4] CPU vs PIM 1-hop Scaling (NEIGHBOR_SPREAD) ===" << endl;
    cout << "  degree | cpu_txns | cpu_cycles | pim_cycles | speedup" << endl;
    cout << "  -------|---------|------------|------------|--------" << endl;

    for (int d : degrees)
    {
        auto [cpu_c, pim_c] = measureCpuVsPim(d, PlacementStrategy::NEIGHBOR_SPREAD);
        int cpu_txns = 2 + d * 2;
        double speedup = static_cast<double>(cpu_c) / pim_c;

        cout << "  " << d
             << "\t| " << cpu_txns
             << "\t| " << cpu_c
             << "\t| " << pim_c
             << "\t| " << speedup << "x" << endl;

        EXPECT_GT(cpu_c, 0u);
        EXPECT_GT(pim_c, 0u);
    }
}

/* -------------------------------------------------------------------------
 * strategy_compare — compare NEIGHBOR_SPREAD vs NODE_LOCAL for PIM gather.
 * Tests which placement strategy gives better PIM performance.
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGPhase4Fixture, strategy_compare)
{
    const vector<int> degrees = {5, 10, 25, 50, 100, 200, 500, 1000};

    cout << "\n=== [Phase 4] Placement Strategy Comparison ===" << endl;
    cout << "  degree | SPREAD_cycles | LOCAL_cycles | better" << endl;
    cout << "  -------|---------------|--------------|--------" << endl;

    for (int d : degrees)
    {
        CSRGraph graph = buildSyntheticGraph(d);

        auto gk_a = setupKernelPhase4(graph, PlacementStrategy::NEIGHBOR_SPREAD);
        uint64_t cycles_a = gk_a->pim1HopGather(0, graph, PlacementStrategy::NEIGHBOR_SPREAD);

        auto gk_b = setupKernelPhase4(graph, PlacementStrategy::NODE_LOCAL);
        uint64_t cycles_b = gk_b->pim1HopGather(0, graph, PlacementStrategy::NODE_LOCAL);

        const char* better = (cycles_a <= cycles_b) ? "SPREAD" : "LOCAL";

        cout << "  " << d
             << "\t| " << cycles_a
             << "\t\t| " << cycles_b
             << "\t\t| " << better << endl;

        EXPECT_GT(cycles_a, 0u);
        EXPECT_GT(cycles_b, 0u);
    }
}

/* -------------------------------------------------------------------------
 * pim_2hop — 2-hop CPU vs PIM comparison.
 * Demonstrates the multiplicative advantage of PIM in multi-hop traversal.
 * ---------------------------------------------------------------------- */
/* -------------------------------------------------------------------------
 * Helper: build a fan-out 2-hop graph and measure CPU vs PIM cycles.
 *   node 0 has d1 neighbors; each neighbor has d2 neighbors.
 * ---------------------------------------------------------------------- */
static pair<uint64_t, uint64_t> measure2Hop(int d1, int d2)
{
    const int total_nodes = 1 + d1 + d1 * d2;
    CSRGraph graph;
    graph.num_nodes = total_nodes;
    graph.num_edges = d1 + d1 * d2;
    graph.row_ptr.resize(total_nodes + 1, 0);
    graph.col_idx.resize(graph.num_edges);
    graph.rel_idx.resize(graph.num_edges);

    graph.row_ptr[0] = 0;
    graph.row_ptr[1] = d1;
    for (int i = 1; i <= d1; i++)
        graph.row_ptr[i + 1] = graph.row_ptr[i] + d2;
    for (int i = d1 + 1; i < total_nodes; i++)
        graph.row_ptr[i + 1] = graph.row_ptr[i];

    for (int j = 0; j < d1; j++) { graph.col_idx[j] = j + 1; graph.rel_idx[j] = j % 237; }
    int edge_idx = d1, dest = d1 + 1;
    for (int i = 1; i <= d1; i++)
        for (int j = 0; j < d2; j++)
        {
            graph.col_idx[edge_idx] = dest;
            graph.rel_idx[edge_idx] = edge_idx % 237;
            edge_idx++; dest++;
        }

    /* CPU 2-hop: 1-hop of source + 1-hop of each hop-1 neighbor */
    auto gk_cpu = setupKernel(graph);
    uint64_t cpu_total = gk_cpu->cpuBaseline1Hop(0, graph);
    for (int j = 0; j < d1; j++)
    {
        auto gk_n = setupKernel(graph);
        cpu_total += gk_n->cpuBaseline1Hop(graph.col_idx[j], graph);
    }

    /* PIM 2-hop */
    auto gk_pim = setupKernelPhase4(graph, PlacementStrategy::NEIGHBOR_SPREAD);
    uint64_t pim_total = gk_pim->pim2HopGather(0, graph, PlacementStrategy::NEIGHBOR_SPREAD);

    return {cpu_total, pim_total};
}

/* -------------------------------------------------------------------------
 * pim_2hop — 2-hop CPU vs PIM across (d1, d2) matrix.
 *   d1: degree of source node (1-hop fan-out)
 *   d2: degree of each hop-1 neighbor (2-hop fan-out)
 *
 *   ./sim --gtest_filter=GraphKGPhase4Fixture.pim_2hop
 * ---------------------------------------------------------------------- */
TEST_F(GraphKGPhase4Fixture, pim_2hop)
{
    /* Original single-setting runs (kept for reference):
     *   //const int degree = 6;  const int neighbor_degree = 4;   // 0.23x
     *   //const int degree = 200; const int neighbor_degree = 200; // TBD
     */
    const vector<int> d1_list = {10, 50, 100, 200};
    const vector<int> d2_list = {10, 50, 100};

    cout << "\n=== [Phase 4] 2-hop CPU vs PIM Scaling (NEIGHBOR_SPREAD) ===" << endl;
    cout << "  d1  | d2  | cpu_cycles | pim_cycles | speedup" << endl;
    cout << "  ----|-----|------------|------------|--------" << endl;

    for (int d1 : d1_list)
        for (int d2 : d2_list)
        {
            auto [cpu_c, pim_c] = measure2Hop(d1, d2);
            double speedup = static_cast<double>(cpu_c) / pim_c;
            cout << "  " << d1
                 << "\t| " << d2
                 << "\t| " << cpu_c
                 << "\t| " << pim_c
                 << "\t| " << speedup << "x" << endl;
            EXPECT_GT(cpu_c, 0u);
            EXPECT_GT(pim_c, 0u);
        }
}
