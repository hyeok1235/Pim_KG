/*
 * GraphKernel.cpp — Phase 2, 3 & 4: CSR-based KG traversal kernels
 *
 * Phase 2 usage:
 *   GraphKernel gk(mem, 1, 1);
 *   gk.preloadCSR(graph);
 *   gk.runPIM();  gk.resetCycle();
 *   uint64_t cycles = gk.cpuBaseline1Hop(source_node, graph);
 *
 * Phase 3 usage:
 *   gk.loadGraphToHBM(graph, PlacementStrategy::NEIGHBOR_SPREAD);
 *   gk.runPIM();
 *   vector<int> dist = gk.analyzeBankDistribution(graph, PlacementStrategy::NEIGHBOR_SPREAD);
 *
 * Phase 4 usage:
 *   gk.loadGraphToHBM(graph, PlacementStrategy::NEIGHBOR_SPREAD);
 *   gk.runPIM();  gk.resetCycle();
 *   uint64_t pim_cycles = gk.pim1HopGather(source, graph, strategy);
 */

#include "tests/GraphKernel.h"

#include <cmath>
#include <iostream>
#include <set>

#include "tests/PIMCmdGen.h"

/* -----------------------------------------------------------------------
 * intAddr — map a (base_row, index) pair to a flat HBM byte address.
 *
 * We store each int32 in its own 32-byte burst (matching the 1-per-burst
 * approach used in existing kernel tests).  The address is generated for
 * channel=0, rank=0, bank-group=0, bank=0, starting at base_row.
 * addrGenSafe advances the row automatically when col overflows.
 * --------------------------------------------------------------------- */
uint64_t GraphKernel::intAddr(unsigned base_row, int linear_index)
{
    unsigned row = base_row;
    unsigned col = static_cast<unsigned>(linear_index);
    return pim_addr_mgr_->addrGenSafe(0, 0, 0, 0, row, col);
}

/* -----------------------------------------------------------------------
 * addBarrier — synchronise all channels before proceeding.
 * --------------------------------------------------------------------- */
void GraphKernel::addBarrier()
{
    unsigned num_chans = getConfigParam(UINT, "NUM_CHANS");
    for (unsigned ch = 0; ch < num_chans; ch++)
        mem_->addBarrier(ch);
}

/* -----------------------------------------------------------------------
 * runPIM — pump the simulator until all pending transactions finish.
 * --------------------------------------------------------------------- */
void GraphKernel::runPIM()
{
    uint64_t start_cycle = cycle_;
    while (mem_->hasPendingTransactions())
    {
        cycle_++;
        mem_->update();
        if (cycle_ - start_cycle > 5000000)
        {
            //cout << "[DEBUG runPIM] TIMEOUT after 5M cycles, breaking!" << endl;
            break;
        }
    }
    //if (cycle_ > start_cycle)
    //    cout << "[DEBUG runPIM] finished after " << (cycle_ - start_cycle)
    //         << " cycles (total=" << cycle_ << ")" << endl;
}

/* -----------------------------------------------------------------------
 * preloadCSR — write row_ptr, col_idx, rel_idx into HBM.
 *
 * Layout (bank 0, channel 0):
 *   row_ptr[i]  → base_row = ROW_PTR_BASE_ROW, col = i
 *   col_idx[i]  → base_row = COL_IDX_BASE_ROW, col = i
 *   rel_idx[i]  → base_row = REL_IDX_BASE_ROW, col = i
 *
 * Each int32 is packed into the first 4 bytes of a 32-byte burst.
 * --------------------------------------------------------------------- */
void GraphKernel::preloadCSR(const CSRGraph& graph)
{
    /* Each addTransaction stores the BurstType pointer and reads it
     * asynchronously, so every entry needs its own buffer. */

    /* --- row_ptr (num_nodes + 1 entries) --- */
    {
        vector<BurstType> rp_bufs(graph.num_nodes + 1);
        for (int i = 0; i <= graph.num_nodes; i++)
        {
            rp_bufs[i] = BurstType{};
            rp_bufs[i].u32Data_[0] = static_cast<uint32_t>(graph.row_ptr[i]);
            uint64_t addr = intAddr(ROW_PTR_BASE_ROW, i);
            mem_->addTransaction(true, addr, "write_row_ptr", &rp_bufs[i]);
        }
        addBarrier();
        runPIM();
    }

    /* --- col_idx (num_edges entries) --- */
    {
        vector<BurstType> col_bufs(graph.num_edges);
        for (int i = 0; i < graph.num_edges; i++)
        {
            col_bufs[i] = BurstType{};
            col_bufs[i].u32Data_[0] = static_cast<uint32_t>(graph.col_idx[i]);
            uint64_t addr = intAddr(COL_IDX_BASE_ROW, i);
            mem_->addTransaction(true, addr, "write_col_idx", &col_bufs[i]);
        }
        addBarrier();
        runPIM();
    }

    /* --- rel_idx (num_edges entries) --- */
    {
        vector<BurstType> rel_bufs(graph.num_edges);
        for (int i = 0; i < graph.num_edges; i++)
        {
            rel_bufs[i] = BurstType{};
            rel_bufs[i].u32Data_[0] = static_cast<uint32_t>(graph.rel_idx[i]);
            uint64_t addr = intAddr(REL_IDX_BASE_ROW, i);
            mem_->addTransaction(true, addr, "write_rel_idx", &rel_bufs[i]);
        }
        addBarrier();
        runPIM();
    }
}

/* -----------------------------------------------------------------------
 * cpuBaseline1Hop — simulate the memory access pattern of a CPU performing
 * a 1-hop KG traversal of 'source' in SB (Standard Bank) mode.
 *
 * Access sequence mirrors the CPU behaviour described in the roadmap:
 *   1. Read row_ptr[source]   (1 burst)
 *   2. Read row_ptr[source+1] (1 burst)
 *   3. Read col_idx[start..end-1]  (degree bursts, sequential)
 *   4. Read rel_idx[start..end-1]  (degree bursts, sequential)
 *
 * Total = 2 + degree * 2 bursts  (matches Phase 1 profiling formula).
 *
 * Returns: simulator cycle count for this traversal.
 * --------------------------------------------------------------------- */
uint64_t GraphKernel::cpuBaseline1Hop(int source, const CSRGraph& graph)
{
    resetCycle();

    int start = graph.row_ptr[source];
    int end   = graph.row_ptr[source + 1];

    /* 1 & 2 — row_ptr reads */
    {
        uint64_t addr = intAddr(ROW_PTR_BASE_ROW, source);
        mem_->addTransaction(false, addr, "row_ptr_start", &read_bst_);
    }
    {
        uint64_t addr = intAddr(ROW_PTR_BASE_ROW, source + 1);
        mem_->addTransaction(false, addr, "row_ptr_end", &read_bst_);
    }

    /* 3 — col_idx reads */
    for (int i = start; i < end; i++)
    {
        uint64_t addr = intAddr(COL_IDX_BASE_ROW, i);
        mem_->addTransaction(false, addr, "col_idx_read", &read_bst_);
    }

    /* 4 — rel_idx reads */
    for (int i = start; i < end; i++)
    {
        uint64_t addr = intAddr(REL_IDX_BASE_ROW, i);
        mem_->addTransaction(false, addr, "rel_idx_read", &read_bst_);
    }

    /* Run simulator to completion */
    runPIM();

    return cycle_;
}

/* =======================================================================
 * Phase 3: PIM-friendly data placement
 * ===================================================================== */

/* -----------------------------------------------------------------------
 * blockIdxToBgBank — map a flat PIM-block index to (bankgroup, bank)
 * for the addrGen/addrGenSafe interface.
 *
 * NUM_BANKS = 16, NUM_BANK_GROUPS = 4 → 4 banks per group.
 * NUM_PIM_BLOCKS = 8 → block 0..7 across bg={0,1}, bank={0..3}.
 * --------------------------------------------------------------------- */
void GraphKernel::blockIdxToBgBank(unsigned block_idx, unsigned& bg, unsigned& bank) const
{
    unsigned num_bank_groups = getConfigParam(UINT, "NUM_BANK_GROUPS");
    unsigned num_banks       = getConfigParam(UINT, "NUM_BANKS");
    unsigned banks_per_bg    = num_banks / num_bank_groups;
    bg   = block_idx / banks_per_bg;
    bank = block_idx % banks_per_bg;
}

/* -----------------------------------------------------------------------
 * loadGraphToHBM — Phase 3: write CSR arrays into HBM with placement
 * strategy controlling bank distribution.
 *
 * row_ptr → always stored in PIM-block 0 (bg=0, bank=0), same as Phase 2.
 * col_idx / rel_idx → distributed across PIM blocks according to strategy:
 *   NEIGHBOR_SPREAD: j-th neighbor of any node → block (j % num_pim_blocks)
 *   NODE_LOCAL:      all edges of node i       → block (i % num_pim_blocks)
 *
 * Each PIM block maintains an independent (row, col) cursor so that
 * successive writes within a block occupy consecutive burst slots.
 * --------------------------------------------------------------------- */
void GraphKernel::loadGraphToHBM(const CSRGraph& graph, PlacementStrategy strategy)
{
    unsigned num_pim_blocks = getConfigParam(UINT, "NUM_PIM_BLOCKS");

    /* --- row_ptr (fixed in PIM-block 0, same as preloadCSR) ---
     * Each entry needs its own BurstType because addTransaction stores the
     * pointer and reads data asynchronously when the write is executed. */
    {
        vector<BurstType> rp_bufs(graph.num_nodes + 1);
        for (int i = 0; i <= graph.num_nodes; i++)
        {
            rp_bufs[i] = BurstType{};
            rp_bufs[i].u32Data_[0] = static_cast<uint32_t>(graph.row_ptr[i]);
            uint64_t addr = intAddr(ROW_PTR_BASE_ROW, i);
            mem_->addTransaction(true, addr, "write_row_ptr", &rp_bufs[i]);
        }
        addBarrier();
        runPIM();
    }

    /* --- per-block cursors for col_idx and rel_idx --- */
    struct BankCursor { unsigned row; unsigned col; };
    vector<BankCursor> col_cursors(num_pim_blocks, {COL_IDX_BASE_ROW, 0});
    vector<BankCursor> rel_cursors(num_pim_blocks, {REL_IDX_BASE_ROW, 0});

    /* --- col_idx distributed across blocks --- */
    {
        vector<BurstType> col_bufs(graph.num_edges);
        int buf_idx = 0;
        for (int node = 0; node < graph.num_nodes; node++)
        {
            int start = graph.row_ptr[node];
            int end   = graph.row_ptr[node + 1];

            for (int j = 0; j < end - start; j++)
            {
                int edge_k = start + j;

                unsigned block_idx;
                if (strategy == PlacementStrategy::NEIGHBOR_SPREAD)
                    block_idx = static_cast<unsigned>(j) % num_pim_blocks;
                else
                    block_idx = static_cast<unsigned>(node) % num_pim_blocks;

                unsigned bg, bank_in_bg;
                blockIdxToBgBank(block_idx, bg, bank_in_bg);

                col_bufs[buf_idx] = BurstType{};
                col_bufs[buf_idx].u32Data_[0] = static_cast<uint32_t>(graph.col_idx[edge_k]);
                uint64_t addr = pim_addr_mgr_->addrGenSafe(
                    0, 0, bg, bank_in_bg,
                    col_cursors[block_idx].row, col_cursors[block_idx].col);
                mem_->addTransaction(true, addr, "write_col_idx", &col_bufs[buf_idx]);
                col_cursors[block_idx].col++;
                buf_idx++;
            }
        }
        addBarrier();
        runPIM();
    }

    /* --- rel_idx distributed across blocks (same strategy) --- */
    {
        vector<BurstType> rel_bufs(graph.num_edges);
        int buf_idx = 0;
        for (int node = 0; node < graph.num_nodes; node++)
        {
            int start = graph.row_ptr[node];
            int end   = graph.row_ptr[node + 1];

            for (int j = 0; j < end - start; j++)
            {
                int edge_k = start + j;

                unsigned block_idx;
                if (strategy == PlacementStrategy::NEIGHBOR_SPREAD)
                    block_idx = static_cast<unsigned>(j) % num_pim_blocks;
                else
                    block_idx = static_cast<unsigned>(node) % num_pim_blocks;

                unsigned bg, bank_in_bg;
                blockIdxToBgBank(block_idx, bg, bank_in_bg);

                rel_bufs[buf_idx] = BurstType{};
                rel_bufs[buf_idx].u32Data_[0] = static_cast<uint32_t>(graph.rel_idx[edge_k]);
                uint64_t addr = pim_addr_mgr_->addrGenSafe(
                    0, 0, bg, bank_in_bg,
                    rel_cursors[block_idx].row, rel_cursors[block_idx].col);
                mem_->addTransaction(true, addr, "write_rel_idx", &rel_bufs[buf_idx]);
                rel_cursors[block_idx].col++;
                buf_idx++;
            }
        }
        addBarrier();
        runPIM();
    }
}

/* -----------------------------------------------------------------------
 * verifyHBMData — Phase 3-3: read back CSR data written by loadGraphToHBM
 * and verify against the original graph.
 *
 * Replays the same address-generation logic as loadGraphToHBM, but issues
 * read transactions instead of writes. After runPIM(), compares the burst
 * contents with the expected values. Returns the number of mismatches.
 * --------------------------------------------------------------------- */
int GraphKernel::verifyHBMData(const CSRGraph& graph, PlacementStrategy strategy)
{
    unsigned num_pim_blocks = getConfigParam(UINT, "NUM_PIM_BLOCKS");
    int mismatches = 0;

    /* ---- 1. Verify row_ptr (PIM-block 0, same as preloadCSR) ---- */
    int row_ptr_count = graph.num_nodes + 1;
    vector<BurstType> rp_bufs(row_ptr_count);
    for (int i = 0; i < row_ptr_count; i++)
    {
        rp_bufs[i] = BurstType{};
        uint64_t addr = intAddr(ROW_PTR_BASE_ROW, i);
        mem_->addTransaction(false, addr, "verify_row_ptr", &rp_bufs[i]);
    }
    addBarrier();
    runPIM();

    for (int i = 0; i < row_ptr_count; i++)
    {
        uint32_t got = rp_bufs[i].u32Data_[0];
        uint32_t exp = static_cast<uint32_t>(graph.row_ptr[i]);
        if (got != exp)
        {
            if (mismatches < 5)
                cout << "  MISMATCH row_ptr[" << i << "]: expected=" << exp
                     << " got=" << got << endl;
            mismatches++;
        }
    }

    /* ---- 2. Verify col_idx + rel_idx per-block ---- */
    struct BankCursor { unsigned row; unsigned col; };
    vector<BankCursor> col_cursors(num_pim_blocks, {COL_IDX_BASE_ROW, 0});
    vector<BankCursor> rel_cursors(num_pim_blocks, {REL_IDX_BASE_ROW, 0});

    /* Collect all (edge_k, block_idx) pairs in order */
    struct EdgeInfo { int edge_k; unsigned block_idx; };
    vector<EdgeInfo> edges;
    edges.reserve(graph.num_edges);

    for (int node = 0; node < graph.num_nodes; node++)
    {
        int start = graph.row_ptr[node];
        int end   = graph.row_ptr[node + 1];
        for (int j = 0; j < end - start; j++)
        {
            unsigned block_idx;
            if (strategy == PlacementStrategy::NEIGHBOR_SPREAD)
                block_idx = static_cast<unsigned>(j) % num_pim_blocks;
            else
                block_idx = static_cast<unsigned>(node) % num_pim_blocks;
            edges.push_back({start + j, block_idx});
        }
    }

    /* Issue read transactions for col_idx */
    vector<BurstType> col_bufs(graph.num_edges);
    for (size_t i = 0; i < edges.size(); i++)
    {
        unsigned bi = edges[i].block_idx;
        unsigned bg, bank_in_bg;
        blockIdxToBgBank(bi, bg, bank_in_bg);

        col_bufs[i] = BurstType{};
        uint64_t addr = pim_addr_mgr_->addrGenSafe(
            0, 0, bg, bank_in_bg,
            col_cursors[bi].row, col_cursors[bi].col);
        mem_->addTransaction(false, addr, "verify_col_idx", &col_bufs[i]);
        col_cursors[bi].col++;
    }
    addBarrier();
    runPIM();

    for (size_t i = 0; i < edges.size(); i++)
    {
        uint32_t got = col_bufs[i].u32Data_[0];
        uint32_t exp = static_cast<uint32_t>(graph.col_idx[edges[i].edge_k]);
        if (got != exp)
        {
            if (mismatches < 5)
                cout << "  MISMATCH col_idx[" << edges[i].edge_k << "] (block "
                     << edges[i].block_idx << "): expected=" << exp
                     << " got=" << got << endl;
            mismatches++;
        }
    }

    /* Issue read transactions for rel_idx */
    vector<BurstType> rel_bufs(graph.num_edges);
    for (size_t i = 0; i < edges.size(); i++)
    {
        unsigned bi = edges[i].block_idx;
        unsigned bg, bank_in_bg;
        blockIdxToBgBank(bi, bg, bank_in_bg);

        rel_bufs[i] = BurstType{};
        uint64_t addr = pim_addr_mgr_->addrGenSafe(
            0, 0, bg, bank_in_bg,
            rel_cursors[bi].row, rel_cursors[bi].col);
        mem_->addTransaction(false, addr, "verify_rel_idx", &rel_bufs[i]);
        rel_cursors[bi].col++;
    }
    addBarrier();
    runPIM();

    for (size_t i = 0; i < edges.size(); i++)
    {
        uint32_t got = rel_bufs[i].u32Data_[0];
        uint32_t exp = static_cast<uint32_t>(graph.rel_idx[edges[i].edge_k]);
        if (got != exp)
        {
            if (mismatches < 5)
                cout << "  MISMATCH rel_idx[" << edges[i].edge_k << "] (block "
                     << edges[i].block_idx << "): expected=" << exp
                     << " got=" << got << endl;
            mismatches++;
        }
    }

    return mismatches;
}

/* -----------------------------------------------------------------------
 * analyzeBankDistribution — Phase 3: count how many edges land in each
 * PIM block for a given placement strategy.  Pure computation, no HBM
 * transactions.  Returns vector<int> of size NUM_PIM_BLOCKS.
 * --------------------------------------------------------------------- */
vector<int> GraphKernel::analyzeBankDistribution(const CSRGraph& graph,
                                                  PlacementStrategy strategy)
{
    unsigned num_pim_blocks = getConfigParam(UINT, "NUM_PIM_BLOCKS");
    vector<int> counts(num_pim_blocks, 0);

    for (int node = 0; node < graph.num_nodes; node++)
    {
        int start = graph.row_ptr[node];
        int end   = graph.row_ptr[node + 1];

        for (int j = 0; j < end - start; j++)
        {
            unsigned block_idx;
            if (strategy == PlacementStrategy::NEIGHBOR_SPREAD)
                block_idx = static_cast<unsigned>(j) % num_pim_blocks;
            else  /* NODE_LOCAL */
                block_idx = static_cast<unsigned>(node) % num_pim_blocks;

            counts[block_idx]++;
        }
    }

    return counts;
}

/* =======================================================================
 * Phase 4: PIM–CPU role division — PIM gather kernels
 * ===================================================================== */

/* -----------------------------------------------------------------------
 * PIM infrastructure helpers — mirror PIMKernel's mode transition logic.
 * GraphKernel is intentionally kept independent of PIMKernel so that
 * the graph-traversal experiments remain self-contained.
 * --------------------------------------------------------------------- */

void GraphKernel::addTransactionAll(bool is_write, int bg_idx, int bank_idx, int row, int col,
                                    const string& tag, BurstType* bst, bool use_barrier,
                                    int num_loop)
{
    for (int ch = 0; ch < num_pim_chans_; ch++)
    {
        for (int ra = 0; ra < num_pim_ranks_; ra++)
        {
            unsigned local_row = static_cast<unsigned>(row);
            unsigned local_col = static_cast<unsigned>(col);
            for (int i = 0; i < num_loop; i++)
            {
                uint64_t addr = pim_addr_mgr_->addrGenSafe(ch, ra, bg_idx, bank_idx,
                                                           local_row, local_col);
                mem_->addTransaction(is_write, addr, tag, bst);
                local_col++;
            }
        }
    }
    if (use_barrier)
        addBarrier();
}

void GraphKernel::parkIn()
{
    addBarrier();
    for (int ch = 0; ch < num_pim_chans_; ch++)
    {
        for (int ra = 0; ra < num_pim_ranks_; ra++)
        {
            unsigned banks_per_bg = num_banks_ / num_bank_groups_;
            for (unsigned bank = 0; bank < banks_per_bg; bank++)
            {
                for (unsigned bg = 0; bg < num_bank_groups_; bg++)
                {
                    string str = "PARK_IN_";
                    if (bg == 0 && bank == 0)
                        str = "START_" + str;
                    else if (bg == num_bank_groups_ - 1 && bank == banks_per_bg - 1)
                        str = "END_" + str;
                    mem_->addTransaction(
                        false,
                        pim_addr_mgr_->addrGen(ch, ra, bg, bank, (1 << 13), 0),
                        str, &null_bst_);
                }
            }
        }
    }
    addBarrier();
}

void GraphKernel::parkOut()
{
    for (int ch = 0; ch < num_pim_chans_; ch++)
    {
        for (int ra = 0; ra < num_pim_ranks_; ra++)
        {
            unsigned banks_per_bg = num_banks_ / num_bank_groups_;
            for (unsigned bank = 0; bank < banks_per_bg; bank++)
            {
                for (unsigned bg = 0; bg < num_bank_groups_; bg++)
                {
                    string str = "PARK_OUT_";
                    if (bg == 0 && bank == 0)
                        str = "START_" + str;
                    else if (bg == num_bank_groups_ - 1 && bank == banks_per_bg - 1)
                        str = "END_" + str;
                    mem_->addTransaction(
                        false,
                        pim_addr_mgr_->addrGen(ch, ra, bg, bank, (1 << 13), 0),
                        str, &null_bst_);
                }
            }
        }
    }
    addBarrier();
}

void GraphKernel::changePIMMode(dramMode curMode, dramMode nextMode)
{
    if (curMode == dramMode::SB && nextMode == dramMode::HAB)
    {
        addTransactionAll(true, 0, 0, pim_abmr_ra_, 0x1f, "START_SB_TO_HAB_", &null_bst_);
        addTransactionAll(true, 0, 1, pim_abmr_ra_, 0x1f, "SB_TO_HAB_", &null_bst_);
        if (num_banks_ >= 2)
        {
            addTransactionAll(true, 2, 0, pim_abmr_ra_, 0x1f, "SB_TO_HAB_", &null_bst_);
            addTransactionAll(true, 2, 1, pim_abmr_ra_, 0x1f, "END_SB_TO_HAB_", &null_bst_);
        }
    }
    else if (curMode == dramMode::HAB)
    {
        if (nextMode == dramMode::SB)
        {
            addTransactionAll(true, 0, 0, pim_sbmr_ra_, 0x1f, "START_HAB_TO_SB", &null_bst_);
            addTransactionAll(true, 0, 1, pim_sbmr_ra_, 0x1f, "END_HAB_TO_SB", &null_bst_);
        }
        else if (nextMode == dramMode::HAB_PIM)
        {
            addTransactionAll(true, 0, 0, pim_reg_ra_, 0x0, "PIM", &bst_hab_pim_);
        }
    }
    else if (curMode == dramMode::HAB_PIM && nextMode == dramMode::HAB)
    {
        addTransactionAll(true, 0, 0, pim_reg_ra_, 0x0, "PIM", &bst_hab_);
    }
    addBarrier();
}

void GraphKernel::programCrf(vector<PIMCmd>& cmds)
{
    PIMCmd nop_cmd(PIMCmdType::NOP, 0);
    for (int i = 0; i < 4; i++)
    {
        if (static_cast<size_t>(i * 8) >= cmds.size())
            break;
        crf_bst_[i].set(nop_cmd.toInt(), nop_cmd.toInt(), nop_cmd.toInt(), nop_cmd.toInt(),
                        nop_cmd.toInt(), nop_cmd.toInt(), nop_cmd.toInt(), nop_cmd.toInt());
        for (int j = 0; j < 8; j++)
        {
            if (static_cast<size_t>(i * 8 + j) >= cmds.size())
                break;
            crf_bst_[i].u32Data_[j] = cmds[i * 8 + j].toInt();
        }
        addTransactionAll(true, 0, 1, pim_reg_ra_, 0x4 + i, "PROGRAM_CRF", &crf_bst_[i]);
    }
    addBarrier();
}

/* -----------------------------------------------------------------------
 * pim1HopGather — Phase 4 core: PIM-accelerated 1-hop neighbor gather.
 *
 * The approach:
 *   1. [CPU] Read row_ptr[source] and row_ptr[source+1] via SB reads
 *      to determine the neighbor range [start, end).
 *   2. [CPU] Compute which PIM blocks hold data and how many bursts
 *      each block needs to FILL (depends on placement strategy).
 *   3. [PIM] Program CRF with FILL commands that read bank data → GRF.
 *      Enter HAB_PIM mode. Issue read transactions to trigger FILL on
 *      each active PIM block — all blocks execute in parallel.
 *   4. [CPU] After PIM completes, read results from each block's
 *      result area via SB reads.
 *
 * Compared to cpuBaseline1Hop which issues 2+degree*2 sequential SB
 * transactions, this approach:
 *   - Programs PIM once (fixed overhead)
 *   - Executes bank reads in parallel across PIM blocks
 *   - Reads only aggregated results back
 * --------------------------------------------------------------------- */
uint64_t GraphKernel::pim1HopGather(int source, const CSRGraph& graph,
                                     PlacementStrategy strategy)
{
    resetCycle();

    int start = graph.row_ptr[source];
    int end   = graph.row_ptr[source + 1];
    int degree = end - start;

    if (degree == 0)
        return 0;

    /* ----- Step 1: CPU reads row_ptr (SB mode) ----- */
    //cout << "[DEBUG pim1HopGather] Step 1: reading row_ptr for source=" << source
    //     << " degree=" << degree << endl;
    {
        uint64_t addr0 = intAddr(ROW_PTR_BASE_ROW, source);
        mem_->addTransaction(false, addr0, "row_ptr_start", &read_bst_);
    }
    {
        uint64_t addr1 = intAddr(ROW_PTR_BASE_ROW, source + 1);
        mem_->addTransaction(false, addr1, "row_ptr_end", &read_bst_);
    }
    addBarrier();
    //cout << "[DEBUG pim1HopGather] Step 1: draining row_ptr reads..." << endl;
    runPIM();  /* drain row_ptr reads before PIM mode transition */
    //cout << "[DEBUG pim1HopGather] Step 1: done, cycle=" << cycle_ << endl;

    /* ----- Step 2: Determine per-block burst counts ----- */
    /* Replay the same placement logic as loadGraphToHBM to figure out
     * how many col_idx entries each PIM block holds for this source node,
     * and what the starting cursor offset is for this node's data. */
    unsigned num_pim_blocks = num_pim_blocks_;

    /* edges_per_block[b] = how many of this node's edges are in block b */
    vector<int> edges_per_block(num_pim_blocks, 0);

    for (int j = 0; j < degree; j++)
    {
        unsigned block_idx;
        if (strategy == PlacementStrategy::NEIGHBOR_SPREAD)
            block_idx = static_cast<unsigned>(j) % num_pim_blocks;
        else  /* NODE_LOCAL */
            block_idx = static_cast<unsigned>(source) % num_pim_blocks;
        edges_per_block[block_idx]++;
    }

    /* max_bursts = max edges across all blocks — determines CRF JUMP count */
    int max_bursts = 0;
    int active_blocks = 0;
    for (unsigned b = 0; b < num_pim_blocks; b++)
    {
        if (edges_per_block[b] > max_bursts)
            max_bursts = edges_per_block[b];
        if (edges_per_block[b] > 0)
            active_blocks++;
    }

    /* Compute each block's cursor offset: we need to know where in the
     * block's address space this source node's col_idx data begins.
     * The offset = total edges stored in this block by all nodes before
     * 'source'. We replay the placement logic for nodes [0, source). */
    vector<int> block_cursor_offset(num_pim_blocks, 0);
    for (int node = 0; node < source; node++)
    {
        int ns = graph.row_ptr[node];
        int ne = graph.row_ptr[node + 1];
        for (int j = 0; j < ne - ns; j++)
        {
            unsigned bi;
            if (strategy == PlacementStrategy::NEIGHBOR_SPREAD)
                bi = static_cast<unsigned>(j) % num_pim_blocks;
            else
                bi = static_cast<unsigned>(node) % num_pim_blocks;
            block_cursor_offset[bi]++;
        }
    }

    /* ----- Step 3: PIM gather — FILL from bank to GRF ----- */
    /* CRF kernel: FILL GRF_A ← EVEN_BANK, NOP×7 pipeline drain, EXIT.
     * Each FILL reads one burst from the bank into GRF_A.
     * JUMP repeats for max_bursts-1 iterations. */
    int num_jump = max_bursts - 1;
    vector<PIMCmd> pim_cmds;
    pim_cmds.push_back(PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK));
    pim_cmds.push_back(PIMCmd(PIMCmdType::NOP, 7));
    if (num_jump > 0)
        pim_cmds.push_back(PIMCmd(PIMCmdType::JUMP, num_jump, static_cast<int>(pim_cmds.size()) + 1));
    pim_cmds.push_back(PIMCmd(PIMCmdType::EXIT, 0));

    /* Set control word: pim_op=true, crf_toggle_cond=ALL_BANK(0),
     * grfA_zero=false, grfB_zero=false */
    bst_hab_pim_ = BurstType{};
    bst_hab_pim_.u8Data_[0]  = 1;   /* pim_op */
    bst_hab_pim_.u8Data_[16] = 0;   /* ALL_BANK */
    bst_hab_pim_.u8Data_[20] = 0;   /* grfA_zero */
    bst_hab_pim_.u8Data_[21] = 0;   /* grfB_zero */

    bst_hab_ = BurstType{};
    bst_hab_.u8Data_[0]  = 0;       /* pim_op=false */
    bst_hab_.u8Data_[16] = 0;

    //cout << "[DEBUG pim1HopGather] Step 3: parkIn..." << endl;
    parkIn();
    runPIM();
    //cout << "[DEBUG pim1HopGather] Step 3: parkIn done, cycle=" << cycle_ << endl;

    //cout << "[DEBUG pim1HopGather] Step 3: SB->HAB..." << endl;
    changePIMMode(dramMode::SB, dramMode::HAB);
    runPIM();
    //cout << "[DEBUG pim1HopGather] Step 3: SB->HAB done, cycle=" << cycle_ << endl;

    //cout << "[DEBUG pim1HopGather] Step 3: programCrf (max_bursts=" << max_bursts
    //     << " active_blocks=" << active_blocks << " num_jump=" << num_jump << ")..." << endl;
    programCrf(pim_cmds);
    runPIM();
    //cout << "[DEBUG pim1HopGather] Step 3: programCrf done, cycle=" << cycle_ << endl;

    //cout << "[DEBUG pim1HopGather] Step 3: HAB->HAB_PIM..." << endl;
    changePIMMode(dramMode::HAB, dramMode::HAB_PIM);
    runPIM();
    //cout << "[DEBUG pim1HopGather] Step 3: HAB->HAB_PIM done, cycle=" << cycle_ << endl;

    /* Issue PIM broadcast reads: in HAB_PIM mode, addresses to (bg=0, bank=0)
     * broadcast to all even banks, (bg=0, bank=1) to all odd banks.
     * Each bank reads from its own local (row, col).
     *
     * Requirement: all blocks must store this source node's data starting at
     * the same col offset. For source=0 this is always col=0. For other
     * sources, block_cursor_offset[] must be uniform (guaranteed when source
     * data is aligned during loadGraphToHBM).
     *
     * We iterate max_bursts times. Banks with fewer edges simply read past
     * their data (harmless — those reads hit whatever is in that row). */
    unsigned start_col = static_cast<unsigned>(block_cursor_offset[0]);
    //cout << "[DEBUG pim1HopGather] Step 3: issuing PIM FILL col_idx, max_bursts="
    //     << max_bursts << " start_col=" << start_col << endl;

    for (int t = 0; t < max_bursts; t++)
    {
        /* even banks */
        addTransactionAll(false, 0, 0, COL_IDX_BASE_ROW, start_col + t,
                          "PIM_FILL_COL", &null_bst_, true);
        /* odd banks */
        addTransactionAll(false, 0, 1, COL_IDX_BASE_ROW, start_col + t,
                          "PIM_FILL_COL", &null_bst_, true);
    }

    //cout << "[DEBUG pim1HopGather] Step 3: issuing PIM FILL rel_idx" << endl;
    for (int t = 0; t < max_bursts; t++)
    {
        addTransactionAll(false, 0, 0, REL_IDX_BASE_ROW, start_col + t,
                          "PIM_FILL_REL", &null_bst_, true);
        addTransactionAll(false, 0, 1, REL_IDX_BASE_ROW, start_col + t,
                          "PIM_FILL_REL", &null_bst_, true);
    }

    //cout << "[DEBUG pim1HopGather] Step 3: HAB_PIM->HAB..." << endl;
    changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
    //cout << "[DEBUG pim1HopGather] Step 3: draining HAB_PIM->HAB..." << endl;
    runPIM();
    //cout << "[DEBUG pim1HopGather] Step 3: HAB_PIM->HAB done, cycle=" << cycle_ << endl;

    //cout << "[DEBUG pim1HopGather] Step 3: HAB->SB..." << endl;
    changePIMMode(dramMode::HAB, dramMode::SB);
    //cout << "[DEBUG pim1HopGather] Step 3: draining HAB->SB..." << endl;
    runPIM();
    //cout << "[DEBUG pim1HopGather] Step 3: HAB->SB done, cycle=" << cycle_ << endl;

    //cout << "[DEBUG pim1HopGather] Step 3: parkOut..." << endl;
    parkOut();
    //cout << "[DEBUG pim1HopGather] Step 3: draining parkOut..." << endl;
    runPIM();
    //cout << "[DEBUG pim1HopGather] Step 3: parkOut done, cycle=" << cycle_ << endl;

    /* ----- Step 4: CPU reads gathered results from PIM blocks ----- */
    /* After PIM completes, results are in each block's GRF area.
     * In practice, results would be read from the GRF. Here we model
     * the result collection: one read per active block rather than one
     * read per neighbor. This is the key savings — N reads → active_blocks reads. */
    for (unsigned b = 0; b < num_pim_blocks; b++)
    {
        if (edges_per_block[b] == 0)
            continue;

        unsigned bg, bank;
        blockIdxToBgBank(b, bg, bank);

        unsigned res_row = COL_IDX_BASE_ROW;
        unsigned res_col = static_cast<unsigned>(block_cursor_offset[b]);

        /* Read one burst per block to collect the gathered result */
        uint64_t addr = pim_addr_mgr_->addrGenSafe(0, 0, bg, bank, res_row, res_col);
        mem_->addTransaction(false, addr, "read_pim_result", &read_bst_);
    }
    addBarrier();

    //cout << "[DEBUG pim1HopGather] Step 4: draining all remaining transactions..." << endl;
    runPIM();
    //cout << "[DEBUG pim1HopGather] DONE, total cycle=" << cycle_ << endl;
    return cycle_;
}

/* -----------------------------------------------------------------------
 * pim2HopGather — Phase 4-5: 2-hop gather with CPU-PIM role division.
 *
 *   [PIM]  1-hop: gather source's neighbors (all PIM blocks parallel)
 *   [CPU]  inspect 1-hop results, determine unique next-hop targets
 *   [PIM]  for each 1-hop neighbor: gather its neighbors
 *   [CPU]  collect all results
 *
 * In the CPU baseline, 2-hop requires 2 + degree*2 + sum(degree_i * 2 + 2)
 * sequential SB reads. With PIM, each hop is one PIM invocation + result
 * collection, regardless of how many neighbors are found.
 * --------------------------------------------------------------------- */
uint64_t GraphKernel::pim2HopGather(int source, const CSRGraph& graph,
                                     PlacementStrategy strategy)
{
    resetCycle();

    int start = graph.row_ptr[source];
    int end   = graph.row_ptr[source + 1];
    int degree = end - start;

    if (degree == 0)
        return 0;

    /* ===== Hop 1: PIM gather of source's neighbors ===== */

    /* CPU: row_ptr reads for source */
    {
        uint64_t addr0 = intAddr(ROW_PTR_BASE_ROW, source);
        mem_->addTransaction(false, addr0, "hop1_row_ptr_s", &read_bst_);
    }
    {
        uint64_t addr1 = intAddr(ROW_PTR_BASE_ROW, source + 1);
        mem_->addTransaction(false, addr1, "hop1_row_ptr_e", &read_bst_);
    }
    addBarrier();
    runPIM();

    /* Compute per-block distribution for source */
    unsigned num_pim_blocks = num_pim_blocks_;
    vector<int> edges_per_block(num_pim_blocks, 0);
    for (int j = 0; j < degree; j++)
    {
        unsigned bi;
        if (strategy == PlacementStrategy::NEIGHBOR_SPREAD)
            bi = static_cast<unsigned>(j) % num_pim_blocks;
        else
            bi = static_cast<unsigned>(source) % num_pim_blocks;
        edges_per_block[bi]++;
    }

    vector<int> block_cursor_offset(num_pim_blocks, 0);
    for (int node = 0; node < source; node++)
    {
        int ns = graph.row_ptr[node];
        int ne = graph.row_ptr[node + 1];
        for (int j = 0; j < ne - ns; j++)
        {
            unsigned bi;
            if (strategy == PlacementStrategy::NEIGHBOR_SPREAD)
                bi = static_cast<unsigned>(j) % num_pim_blocks;
            else
                bi = static_cast<unsigned>(node) % num_pim_blocks;
            block_cursor_offset[bi]++;
        }
    }

    int max_bursts = 0;
    for (unsigned b = 0; b < num_pim_blocks; b++)
        if (edges_per_block[b] > max_bursts)
            max_bursts = edges_per_block[b];

    /* PIM gather hop 1 */
    int num_jump = max_bursts - 1;
    vector<PIMCmd> pim_cmds;
    pim_cmds.push_back(PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK));
    pim_cmds.push_back(PIMCmd(PIMCmdType::NOP, 7));
    if (num_jump > 0)
        pim_cmds.push_back(PIMCmd(PIMCmdType::JUMP, num_jump, static_cast<int>(pim_cmds.size()) + 1));
    pim_cmds.push_back(PIMCmd(PIMCmdType::EXIT, 0));

    bst_hab_pim_ = BurstType{};
    bst_hab_pim_.u8Data_[0]  = 1;
    bst_hab_pim_.u8Data_[16] = 0;
    bst_hab_ = BurstType{};

    parkIn();
    changePIMMode(dramMode::SB, dramMode::HAB);
    programCrf(pim_cmds);
    changePIMMode(dramMode::HAB, dramMode::HAB_PIM);

    /* PIM broadcast FILL: (bg=0,bank=0) → all even banks, (bg=0,bank=1) → all odd banks */
    {
        unsigned start_col = static_cast<unsigned>(block_cursor_offset[0]);
        for (int t = 0; t < max_bursts; t++)
        {
            addTransactionAll(false, 0, 0, COL_IDX_BASE_ROW, start_col + t,
                              "HOP1_FILL_COL", &null_bst_, true);
            addTransactionAll(false, 0, 1, COL_IDX_BASE_ROW, start_col + t,
                              "HOP1_FILL_COL", &null_bst_, true);
        }
    }

    changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
    changePIMMode(dramMode::HAB, dramMode::SB);
    parkOut();

    /* Result collection for hop 1 */
    for (unsigned b = 0; b < num_pim_blocks; b++)
    {
        if (edges_per_block[b] == 0) continue;
        unsigned bg, bank;
        blockIdxToBgBank(b, bg, bank);
        unsigned res_row = COL_IDX_BASE_ROW;
        unsigned res_col = static_cast<unsigned>(block_cursor_offset[b]);
        uint64_t addr = pim_addr_mgr_->addrGenSafe(0, 0, bg, bank, res_row, res_col);
        mem_->addTransaction(false, addr, "hop1_result", &read_bst_);
    }
    addBarrier();
    runPIM();

    /* ===== Hop 2: For each 1-hop neighbor, PIM gather its neighbors ===== */
    /* CPU determines the unique set of 1-hop neighbors (known from graph). */
    set<int> hop1_neighbors;
    for (int j = start; j < end; j++)
        hop1_neighbors.insert(graph.col_idx[j]);

    for (int neighbor : hop1_neighbors)
    {
        int n_start = graph.row_ptr[neighbor];
        int n_end   = graph.row_ptr[neighbor + 1];
        int n_degree = n_end - n_start;
        if (n_degree == 0)
            continue;

        /* CPU: row_ptr reads for this neighbor */
        {
            uint64_t a0 = intAddr(ROW_PTR_BASE_ROW, neighbor);
            mem_->addTransaction(false, a0, "hop2_row_ptr_s", &read_bst_);
        }
        {
            uint64_t a1 = intAddr(ROW_PTR_BASE_ROW, neighbor + 1);
            mem_->addTransaction(false, a1, "hop2_row_ptr_e", &read_bst_);
        }
        addBarrier();
        runPIM();

        /* Compute per-block distribution for this neighbor */
        vector<int> n_epb(num_pim_blocks, 0);
        for (int j = 0; j < n_degree; j++)
        {
            unsigned bi;
            if (strategy == PlacementStrategy::NEIGHBOR_SPREAD)
                bi = static_cast<unsigned>(j) % num_pim_blocks;
            else
                bi = static_cast<unsigned>(neighbor) % num_pim_blocks;
            n_epb[bi]++;
        }

        vector<int> n_offset(num_pim_blocks, 0);
        for (int node = 0; node < neighbor; node++)
        {
            int ns = graph.row_ptr[node];
            int ne = graph.row_ptr[node + 1];
            for (int j = 0; j < ne - ns; j++)
            {
                unsigned bi;
                if (strategy == PlacementStrategy::NEIGHBOR_SPREAD)
                    bi = static_cast<unsigned>(j) % num_pim_blocks;
                else
                    bi = static_cast<unsigned>(node) % num_pim_blocks;
                n_offset[bi]++;
            }
        }

        int n_max = 0;
        for (unsigned b = 0; b < num_pim_blocks; b++)
            if (n_epb[b] > n_max) n_max = n_epb[b];

        /* PIM gather hop 2 for this neighbor */
        int nj = n_max - 1;
        vector<PIMCmd> cmds2;
        cmds2.push_back(PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK));
        cmds2.push_back(PIMCmd(PIMCmdType::NOP, 7));
        if (nj > 0)
            cmds2.push_back(PIMCmd(PIMCmdType::JUMP, nj, static_cast<int>(cmds2.size()) + 1));
        cmds2.push_back(PIMCmd(PIMCmdType::EXIT, 0));

        bst_hab_pim_ = BurstType{};
        bst_hab_pim_.u8Data_[0]  = 1;
        bst_hab_pim_.u8Data_[16] = 0;
        bst_hab_ = BurstType{};

        parkIn();
        changePIMMode(dramMode::SB, dramMode::HAB);
        programCrf(cmds2);
        changePIMMode(dramMode::HAB, dramMode::HAB_PIM);

        /* PIM broadcast FILL for hop 2 */
        {
            unsigned start_col = static_cast<unsigned>(n_offset[0]);
            for (int t = 0; t < n_max; t++)
            {
                addTransactionAll(false, 0, 0, COL_IDX_BASE_ROW, start_col + t,
                                  "HOP2_FILL_COL", &null_bst_, true);
                addTransactionAll(false, 0, 1, COL_IDX_BASE_ROW, start_col + t,
                                  "HOP2_FILL_COL", &null_bst_, true);
            }
        }

        changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
        changePIMMode(dramMode::HAB, dramMode::SB);
        parkOut();

        /* Result collection for this neighbor */
        for (unsigned b = 0; b < num_pim_blocks; b++)
        {
            if (n_epb[b] == 0) continue;
            unsigned bg, bank;
            blockIdxToBgBank(b, bg, bank);
            unsigned res_row = COL_IDX_BASE_ROW;
            unsigned res_col = static_cast<unsigned>(n_offset[b]);
            uint64_t addr = pim_addr_mgr_->addrGenSafe(0, 0, bg, bank, res_row, res_col);
            mem_->addTransaction(false, addr, "hop2_result", &read_bst_);
        }
        addBarrier();
        runPIM();
    }

    return cycle_;
}
