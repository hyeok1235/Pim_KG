/*
 * GraphKernel.h — Phase 2, 3 & 4: CSR-based KG traversal kernels for PIMSimulator
 *
 * Phase 2 usage (SB-mode CPU baseline):
 *   GraphKernel gk(mem, 1, 1);
 *   gk.preloadCSR(graph);
 *   gk.runPIM();                       // drain preload writes
 *   gk.resetCycle();
 *   uint64_t cycles = gk.cpuBaseline1Hop(source_node, graph);
 *
 * Phase 3 usage (HBM bank placement + distribution analysis):
 *   gk.loadGraphToHBM(graph, PlacementStrategy::NEIGHBOR_SPREAD);
 *   gk.runPIM();                       // drain preload writes
 *   vector<int> dist = gk.analyzeBankDistribution(graph, PlacementStrategy::NEIGHBOR_SPREAD);
 *   // dist[b] = number of edges stored in PIM block b
 *
 * Phase 4 usage (PIM 1-hop gather vs CPU baseline):
 *   gk.loadGraphToHBM(graph, PlacementStrategy::NEIGHBOR_SPREAD);
 *   gk.runPIM();  gk.resetCycle();
 *   uint64_t pim_cycles = gk.pim1HopGather(source, graph, PlacementStrategy::NEIGHBOR_SPREAD);
 *
 *   // 2-hop with CPU-PIM role division:
 *   uint64_t cycles_2hop = gk.pim2HopGather(source, graph, strategy);
 *
 * Run:
 *   ./sim --gtest_filter=GraphKGFixture.*          # Phase 2 SB baseline
 *   ./sim --gtest_filter=GraphKGPhase3Fixture.*    # Phase 3 placement tests
 *   ./sim --gtest_filter=GraphKGPhase4Fixture.*    # Phase 4 PIM gather tests
 */

#ifndef __GRAPH_KERNEL_H__
#define __GRAPH_KERNEL_H__

#include <cstdint>
#include <string>
#include <vector>

#include "Burst.h"
#include "MultiChannelMemorySystem.h"
#include "PIMCmd.h"
#include "SystemConfiguration.h"
#include "tests/KernelAddrGen.h"

using namespace std;
using namespace DRAMSim;

/* -------------------------------------------------------------------------
 * PlacementStrategy — controls how CSR edge arrays are distributed across
 * HBM PIM blocks when loading via loadGraphToHBM().
 *
 *   NEIGHBOR_SPREAD (전략 A):
 *     The j-th neighbor of any node → PIM block (j % NUM_PIM_BLOCKS).
 *     Maximises bank-level parallelism for high-degree nodes.
 *
 *   NODE_LOCAL (전략 B):
 *     All edges of node i → PIM block (i % NUM_PIM_BLOCKS).
 *     Maximises row-buffer hit rate within a bank for low-degree nodes.
 * ---------------------------------------------------------------------- */
enum class PlacementStrategy
{
    NEIGHBOR_SPREAD,
    NODE_LOCAL
};

/* -------------------------------------------------------------------------
 * CSRGraph — in-memory representation of a knowledge graph in CSR format.
 *
 *   row_ptr[i] .. row_ptr[i+1]-1  →  range in col_idx / rel_idx for node i
 *   col_idx[k]                    →  destination entity of edge k
 *   rel_idx[k]                    →  relation type of edge k
 * ---------------------------------------------------------------------- */
struct CSRGraph
{
    int              num_nodes;
    int              num_edges;
    vector<int>      row_ptr;   // length num_nodes + 1
    vector<int>      col_idx;   // length num_edges
    vector<int>      rel_idx;   // length num_edges

    int degree(int node) const { return row_ptr[node + 1] - row_ptr[node]; }
};

/* -------------------------------------------------------------------------
 * GraphKernel — wraps PIMKernel helpers for graph-traversal experiments.
 *
 * Phase 2 scope: SB (Standard Bank) mode only.
 *   - preloadCSR()      : write CSR arrays into HBM via normal writes
 *   - cpuBaseline1Hop() : simulate CPU 1-hop read pattern (SB mode reads)
 *
 * Memory layout (each int32 value packed into the low 4 bytes of a burst):
 *   ROW_PTR_BASE_ROW  : row 0  of bank(0,0,ch=0)  → row_ptr array
 *   COL_IDX_BASE_ROW  : row 64 of bank(0,0,ch=0)  → col_idx array
 *   REL_IDX_BASE_ROW  : row 128 of bank(0,0,ch=0) → rel_idx array
 *
 * Each HBM burst holds 32 bytes = 8 int32 values.
 * ---------------------------------------------------------------------- */
class GraphKernel
{
  public:
    /* 4 int32 per burst (32 bytes / 4 bytes per int, but we pack 1 per burst
     * for simplicity — matching the pattern used by existing kernel tests) */
    static constexpr int INTS_PER_BURST = 1;

    /* Fixed DRAM row offsets for CSR sections in the baseline bank */
    static constexpr unsigned ROW_PTR_BASE_ROW = 0;
    static constexpr unsigned COL_IDX_BASE_ROW = 64;
    static constexpr unsigned REL_IDX_BASE_ROW = 128;

    GraphKernel(shared_ptr<MultiChannelMemorySystem> mem, int num_pim_chan, int num_pim_rank)
        : mem_(mem),
          pim_addr_mgr_(make_shared<PIMAddrManager>(num_pim_chan, num_pim_rank)),
          transaction_size_(
              getConfigParam(UINT, "BL") * (getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") / 8)),
          num_pim_chans_(num_pim_chan),
          num_pim_ranks_(num_pim_rank),
          num_banks_(getConfigParam(UINT, "NUM_BANKS")),
          num_bank_groups_(getConfigParam(UINT, "NUM_BANK_GROUPS")),
          num_pim_blocks_(getConfigParam(UINT, "NUM_PIM_BLOCKS")),
          cycle_(0)
    {
    }

    /* Phase 2 — Write CSR arrays into HBM so subsequent reads observe
     * realistic latency. All arrays go into bank 0 (baseline layout). */
    void preloadCSR(const CSRGraph& graph);

    /* Phase 2 — SB-mode 1-hop: issue all DRAM reads that a CPU would perform
     * for one 1-hop traversal of 'source'. Returns total simulation cycles. */
    uint64_t cpuBaseline1Hop(int source, const CSRGraph& graph);

    /* Phase 3 — Write CSR arrays into HBM using the given placement strategy.
     * row_ptr is always stored in PIM-block 0 (fixed reference).
     * col_idx and rel_idx are distributed according to 'strategy'. */
    void loadGraphToHBM(const CSRGraph& graph, PlacementStrategy strategy);

    /* Phase 3 — Return a vector of length NUM_PIM_BLOCKS where element [b]
     * is the number of edges (col_idx entries) assigned to PIM block b under
     * the given strategy. Pure computation — no HBM transactions issued. */
    vector<int> analyzeBankDistribution(const CSRGraph& graph, PlacementStrategy strategy);

    /* Phase 3-3 — Read back all CSR data written by loadGraphToHBM and
     * compare against the original graph. Returns the number of mismatches
     * (0 = success). Prints first few errors to stdout for debugging.
     * Must be called after loadGraphToHBM() + runPIM(). */
    int verifyHBMData(const CSRGraph& graph, PlacementStrategy strategy);

    /* Phase 4 — PIM 1-hop gather: use PIM FILL(MOV) commands to read
     * col_idx and rel_idx from bank-local storage into GRF, then read
     * results back to CPU. The graph must have been loaded via
     * loadGraphToHBM() with the same strategy. Returns total cycles.
     *
     * Role division:
     *   CPU: row_ptr lookup (SB reads), PIM kernel programming, result collection
     *   PIM: bank-parallel FILL of col_idx/rel_idx data → GRF */
    uint64_t pim1HopGather(int source, const CSRGraph& graph, PlacementStrategy strategy);

    /* Phase 4-5 — PIM 2-hop gather with CPU-PIM role division:
     *   [PIM]  1-hop gather of source → neighbors
     *   [CPU]  determine next-hop targets from returned neighbors
     *   [PIM]  for each 1-hop neighbor, gather its neighbors
     *   [CPU]  collect all 2-hop results
     * Returns total cycles for the entire 2-hop traversal. */
    uint64_t pim2HopGather(int source, const CSRGraph& graph, PlacementStrategy strategy);

    /* Run simulator until all transactions complete */
    void runPIM();

    uint64_t getCycle() const { return cycle_; }
    void     resetCycle()     { cycle_ = 0; }

  private:
    void addBarrier();

    /* Compute the flat DRAM address for a single int32 stored at a given
     * (section_base_row, linear_index) in the baseline bank (ch=0,ra=0,bg=0,bank=0) */
    uint64_t intAddr(unsigned base_row, int linear_index);

    /* Phase 3 — Decompose a flat PIM-block index (0..NUM_PIM_BLOCKS-1) into
     * the (bankgroup, bank-within-group) pair expected by addrGenSafe.
     * Mapping: bg = block_idx / banks_per_bg, bank = block_idx % banks_per_bg */
    void blockIdxToBgBank(unsigned block_idx, unsigned& bg, unsigned& bank) const;

    /* Phase 4 — PIM mode transition helpers (mirror PIMKernel patterns) */
    void parkIn();
    void parkOut();
    void changePIMMode(dramMode curMode, dramMode nextMode);
    void programCrf(vector<PIMCmd>& cmds);
    void addTransactionAll(bool is_write, int bg_idx, int bank_idx, int row, int col,
                           const string& tag, BurstType* bst, bool use_barrier = false,
                           int num_loop = 1);

    shared_ptr<MultiChannelMemorySystem> mem_;
    shared_ptr<PIMAddrManager>           pim_addr_mgr_;
    int                                  transaction_size_;
    int                                  num_pim_chans_;
    int                                  num_pim_ranks_;
    unsigned                             num_banks_;
    unsigned                             num_bank_groups_;
    unsigned                             num_pim_blocks_;
    uint64_t                             cycle_;

    /* PIM register addresses (same as PIMKernel) */
    static constexpr uint32_t pim_reg_ra_  = 0x3fff;
    static constexpr uint32_t pim_abmr_ra_ = 0x27ff;
    static constexpr uint32_t pim_sbmr_ra_ = 0x2fff;

    BurstType null_bst_;
    BurstType read_bst_;   // scratch buffer for reads (contents ignored in SB baseline)
    BurstType bst_hab_pim_;
    BurstType bst_hab_;
    BurstType crf_bst_[4];
};

#endif  // __GRAPH_KERNEL_H__
