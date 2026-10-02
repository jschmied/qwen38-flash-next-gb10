#!/usr/bin/env python3
"""Extraction variants for K2 2..8 against experts_grouped.cuh's decode_tile: 'direct' = each lane loads its own two
words (LaneMap's hi/lo) from the tile, no coalesced load and no shuffles; then the same 64-bit window extraction and
the same codebook. Checks the decoded fragments equal decode_tile's for every lane on random tiles, then times both
(ps per tile, L1-resident tiles). argv: <exl3 header dir>"""
import json, sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

hdr = Path(sys.argv[1]).resolve()
src = r'''
#include <torch/extension.h>
#include <cuda_fp16.h>
#include "experts_grouped.cuh"
using namespace tf_exl3x;

// this lane's 8 states with its two words loaded directly (hi = word holding the run's last bit, lo = the one before)
template <int K2>
__device__ __forceinline__ void states_direct(const uint32_t* tile, const LaneMap<K2>& m, uint32_t (&st)[8]) {
    static_assert(Fmt<K2>::NG == 1, "one run of eight windows");
    const uint32_t whi = __ldg(tile + m.hi[0]);
    const uint32_t wlo = __ldg(tile + m.lo[0]);
    const uint64_t mm = ((((uint64_t)wlo) << 32) | whi) >> m.sh[0];
#pragma unroll
    for (int j = 0; j < 8; ++j) st[j] = (uint32_t)(mm >> Fmt<K2>::off(j)) & 0xffffu;
}

template <int K2, int V>
__global__ void k(const uint32_t* T, int ntiles, int reps, uint32_t* out) {
    const int lane = threadIdx.x & 31;
    const LaneMap<K2> m(lane);
    uint32_t acc = 0;
    for (int r = 0; r < reps; ++r) {
        const uint32_t* tile = T + ((r + blockIdx.x) & (ntiles - 1)) * Fmt<K2>::TW;
        uint32_t b0[2], b1[2];
        if constexpr (V == 0) {
            uint32_t w[Fmt<K2>::LW];
            load_words<K2>(w, tile + lane, lane);
            decode_tile<2, K2>(w, m, lane, b0, b1);
        } else {
            uint32_t st[8];
            states_direct<K2>(tile, m, st);
            b0[0] = cb_pair<2>(st[0], st[1]); b0[1] = cb_pair<2>(st[2], st[3]);
            b1[0] = cb_pair<2>(st[4], st[5]); b1[1] = cb_pair<2>(st[6], st[7]);
        }
        if (reps == 1) {                                  // check mode: every lane's fragments of tile blockIdx.x
            uint32_t* o = out + (blockIdx.x * 32 + lane) * 4;
            o[0] = b0[0]; o[1] = b0[1]; o[2] = b1[0]; o[3] = b1[1];
            return;
        }
        acc ^= b0[0] ^ b0[1] ^ b1[0] ^ b1[1];
    }
    out[blockIdx.x * 32 + lane] = acc;
}

#define RUN(K2_) if (k2 == K2_) { if (v == 0) k<K2_, 0><<<blocks, 32>>>((const uint32_t*)T.data_ptr(), ntiles, reps, (uint32_t*)O.data_ptr()); \
                                  else k<K2_, 1><<<blocks, 32>>>((const uint32_t*)T.data_ptr(), ntiles, reps, (uint32_t*)O.data_ptr()); return; }
void run(int64_t v, int64_t k2, torch::Tensor T, int64_t ntiles, int64_t reps, int64_t blocks, torch::Tensor O) {
    RUN(2) RUN(3) RUN(4) RUN(5) RUN(6) RUN(8)
}
'''
ext = load_inline(name="xdirect", cpp_sources="void run(int64_t, int64_t, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor);",
                  cuda_sources=src, functions=["run"], extra_include_paths=[str(hdr)],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a"], verbose=False)
NT, REPS, BLOCKS = 8, 4096, 48 * 32
res = {}
for k2 in (2, 3, 4, 5, 6, 8):
    T = torch.randint(-2**31, 2**31 - 1, (4096 * 64,), dtype=torch.int32, device="cuda")
    a0 = torch.empty(4096 * 32 * 4, dtype=torch.int32, device="cuda")
    a1 = torch.empty_like(a0)
    ext.run(0, k2, T, 4096, 1, 4096, a0)
    ext.run(1, k2, T, 4096, 1, 4096, a1)
    torch.cuda.synchronize()
    exact = bool(torch.equal(a0, a1))
    O = torch.empty(BLOCKS * 32, dtype=torch.int32, device="cuda")
    t = {}
    for _ in range(3):                                    # alternate the two variants
        for v in (0, 1):
            ext.run(v, k2, T, NT, 64, BLOCKS, O)
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            ext.run(v, k2, T, NT, REPS, BLOCKS, O)
            b.record()
            torch.cuda.synchronize()
            t.setdefault(v, []).append(round(a.elapsed_time(b) * 1e9 / (BLOCKS * REPS), 1))
    res[k2] = {"exact": exact, "decode_tile": t[0], "direct": t[1]}
    print(json.dumps({"K2": k2, **res[k2]}), flush=True)
