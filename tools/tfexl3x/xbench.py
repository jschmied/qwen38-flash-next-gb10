#!/usr/bin/env python3
"""EXL3 trellis extraction cost, isolated: warps decode the same few tiles (L1-resident) many times and xor the results,
so only instruction cost shows. Per K2 (2..16) and family: grouped = experts_grouped.cuh's decode_tile (expert kernels,
#212's prompt kernel), dense = decode.cuh's decode_lane (linear_kernel, unpack). Modes: states only (extraction) and
full (extraction + codebook, mul1). Also the variant header given by --variant (a copy of the file with a fast path).
argv: <exl3 header dir> [label]"""
import json, sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

hdr = Path(sys.argv[1]).resolve()
label = sys.argv[2] if len(sys.argv) > 2 else "base"
K2S = list(range(2, 17))
src = r'''
#include <torch/extension.h>
#include <cuda_fp16.h>
#include "experts_grouped.cuh"
#include "decode.cuh"
using namespace tf_exl3x;

template <int K2, int MODE>
__global__ void gk(const uint32_t* T, int ntiles, int reps, uint32_t* out) {
    const int lane = threadIdx.x & 31;
    const LaneMap<K2> m(lane);
    uint32_t acc = 0;
    for (int r = 0; r < reps; ++r) {
        const uint32_t* p = T + ((r + blockIdx.x) & (ntiles - 1)) * Fmt<K2>::TW + lane;
        uint32_t w[Fmt<K2>::LW];
        load_words<K2>(w, p, lane);
        if constexpr (MODE == 2) {
            uint32_t a = 0;
#pragma unroll
            for (int i = 0; i < Fmt<K2>::LW; ++i) a ^= w[i] << i;
            acc ^= a + m.sh[0];
        } else if constexpr (MODE == 0) {
            uint32_t b0[2], b1[2];
            decode_tile<2, K2>(w, m, lane, b0, b1);
            acc = acc * 2654435761u + (b0[0] ^ b0[1] ^ b1[0] ^ b1[1]);   // multiply-add: repeats do not cancel
        } else {
            uint32_t st[8];
            tile_states<K2>(w, m, lane, st);
            acc ^= st[0] ^ (st[1] << 1) ^ (st[2] << 2) ^ (st[3] << 3) ^ (st[4] << 4) ^ (st[5] << 5) ^ (st[6] << 6) ^ (st[7] << 7);
        }
    }
    out[blockIdx.x * 32 + lane] = acc;
}

template <int K2>
__global__ void dk(const uint32_t* T, int ntiles, int reps, uint32_t* out) {
    const int lane = threadIdx.x & 31;
    uint32_t acc = 0;
    for (int r = 0; r < reps; ++r) {
        const uint32_t* tile = T + ((r + blockIdx.x) & (ntiles - 1)) * tf_exl3::tile_words<K2>();
        uint32_t w[tf_exl3::lane_words<K2>()];
        tf_exl3::ldg_lane_words<K2>(tile, lane, w);
        uint32_t b0[2], b1[2];
        tf_exl3::decode_lane<K2, tf_exl3::CB_MUL1>(w, lane, b0, b1);
        acc = acc * 2654435761u + (b0[0] ^ b0[1] ^ b1[0] ^ b1[1]);   // multiply-add: repeats do not cancel
    }
    out[blockIdx.x * 32 + lane] = acc;
}

#define RUN(K2_) if (k2 == K2_) { if (fam == 0) gk<K2_, 0><<<blocks, 32>>>((const uint32_t*)T.data_ptr(), ntiles, reps, (uint32_t*)O.data_ptr()); \
                                  else if (fam == 3) gk<K2_, 2><<<blocks, 32>>>((const uint32_t*)T.data_ptr(), ntiles, reps, (uint32_t*)O.data_ptr()); \
                                  else if (fam == 2) gk<K2_, 1><<<blocks, 32>>>((const uint32_t*)T.data_ptr(), ntiles, reps, (uint32_t*)O.data_ptr()); \
                                  else dk<K2_><<<blocks, 32>>>((const uint32_t*)T.data_ptr(), ntiles, reps, (uint32_t*)O.data_ptr()); return; }
void run(int64_t fam, int64_t k2, torch::Tensor T, int64_t ntiles, int64_t reps, int64_t blocks, torch::Tensor O) {
    RUN(2) RUN(3) RUN(4) RUN(5) RUN(6) RUN(7) RUN(8) RUN(9) RUN(10) RUN(11) RUN(12) RUN(13) RUN(14) RUN(15) RUN(16)
}
'''
ext = load_inline(name=f"xbench_{label}", cpp_sources="void run(int64_t, int64_t, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor);",
                  cuda_sources=src, functions=["run"], extra_include_paths=[str(hdr)],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a"], verbose=False)
NT, REPS, BLOCKS = 8, 4096, 48 * 32
T = torch.randint(-2**31, 2**31 - 1, (NT * 64,), dtype=torch.int32, device="cuda")
O = torch.empty(BLOCKS * 32, dtype=torch.int32, device="cuda")
res = {}
for fam, name in ((0, "grouped"), (2, "gstates"), (3, "gload"), (1, "dense")):
    for k2 in K2S:
        ext.run(fam, k2, T, NT, 64, BLOCKS, O)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        best = 1e9
        for _ in range(3):
            a.record()
            ext.run(fam, k2, T, NT, REPS, BLOCKS, O)
            b.record()
            torch.cuda.synchronize()
            best = min(best, a.elapsed_time(b))
        tiles = BLOCKS * REPS
        h = int(O.sum().item()) & 0xffffffff
        res[f"{name}-{k2}"] = {"ps_per_tile": round(best * 1e9 / tiles, 1), "check": h}
print(json.dumps({"label": label, **res}))
