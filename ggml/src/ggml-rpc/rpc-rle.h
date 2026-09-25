#pragma once

// Lossless run-length coding for RPC_CMD_SET_TENSOR_RLE.
//
// Why: the KQ mask is an F16 input of n_tokens x n_kv that is re-uploaded to
// every RPC server for every micro-batch. It holds only a few long runs of 0
// and -inf, but goes over the wire at full size (tens of MB per upload at
// long context).
//
// The data is split into words of the tensor element size (2 for F16/BF16,
// 4 for F32) and stored as runs of { u32 count, u32 value }. Decoding
// rebuilds the input byte for byte, so the model sees identical data.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

struct rpc_rle_run {
    uint32_t count;
    uint32_t value;
};
static_assert(sizeof(rpc_rle_run) == 8, "rpc_rle_run must be 8 bytes");

static inline uint32_t rpc_rle_load(const uint8_t * p, uint32_t word) {
    if (word == 2) {
        uint16_t v;
        memcpy(&v, p, 2);
        return v;
    }
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

// Encodes data into runs. Gives up and returns false as soon as the runs
// would take more than size/8 bytes, so data that does not compress (weights,
// embeddings) costs a short scan and then takes the normal path.
static inline bool rpc_rle_encode(const uint8_t * data, size_t size, uint32_t word,
                                  std::vector<rpc_rle_run> & runs) {
    runs.clear();
    if ((word != 2 && word != 4) || size == 0 || size % word != 0) {
        return false;
    }
    const size_t n = size / word;
    const size_t max_runs = size / 8 / sizeof(rpc_rle_run);
    if (max_runs == 0) {
        return false;
    }
    rpc_rle_run cur = { 1, rpc_rle_load(data, word) };
    for (size_t i = 1; i < n; i++) {
        const uint32_t v = rpc_rle_load(data + i * word, word);
        if (v == cur.value && cur.count != UINT32_MAX) {
            cur.count++;
            continue;
        }
        runs.push_back(cur);
        if (runs.size() >= max_runs) {
            runs.clear();
            return false;
        }
        cur = { 1, v };
    }
    runs.push_back(cur);
    return true;
}

// Decodes runs into out, which must be exactly size bytes. Rejects anything
// that does not add up to exactly size: an empty run, a value wider than the
// word, or a total that is short or long.
static inline bool rpc_rle_decode(const rpc_rle_run * runs, size_t n_runs, uint32_t word,
                                  uint8_t * out, size_t size) {
    if ((word != 2 && word != 4) || size % word != 0) {
        return false;
    }
    const size_t n = size / word;
    size_t pos = 0;
    for (size_t r = 0; r < n_runs; r++) {
        const size_t count = runs[r].count;
        const uint32_t value = runs[r].value;
        if (count == 0 || count > n - pos) {
            return false;
        }
        if (word == 2) {
            if (value > UINT16_MAX) {
                return false;
            }
            const uint16_t v = (uint16_t) value;
            for (size_t i = 0; i < count; i++) {
                memcpy(out + (pos + i) * 2, &v, 2);
            }
        } else {
            for (size_t i = 0; i < count; i++) {
                memcpy(out + (pos + i) * 4, &value, 4);
            }
        }
        pos += count;
    }
    return pos == n;
}
