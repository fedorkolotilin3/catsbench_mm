import triton
import triton.language as tl
from . import triton_math


MAX_BATCH_DIMS = 6

def _lse_autotune_configs() -> list[triton.Config]:
    # One conservative tile avoids compiling large unrolled float64 libdevice kernels.
    # A single configuration also skips the autotuning benchmark sweep.
    return [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 8}, num_warps=4, num_stages=2),
    ]


@triton.autotune(
    configs=_lse_autotune_configs(),
    key=["M", "N", "K", "USE_EXP2"],
)
@triton.jit
def _lse_fwd_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    b0: tl.constexpr, b1: tl.constexpr, b2: tl.constexpr, b3: tl.constexpr, b4: tl.constexpr, b5: tl.constexpr,
    sa0: tl.constexpr, sa1: tl.constexpr, sa2: tl.constexpr, sa3: tl.constexpr, sa4: tl.constexpr, sa5: tl.constexpr,
    sb0: tl.constexpr, sb1: tl.constexpr, sb2: tl.constexpr, sb3: tl.constexpr, sb4: tl.constexpr, sb5: tl.constexpr,
    sc0: tl.constexpr, sc1: tl.constexpr, sc2: tl.constexpr, sc3: tl.constexpr, sc4: tl.constexpr, sc5: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BATCH_DIMS: tl.constexpr,
    USE_EXP2: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    ACC_DTYPE: tl.constexpr = tl.float64 if A_ptr.dtype.element_ty == tl.float64 else tl.float32
    log2e = tl.full((), 1.4426950408889634, ACC_DTYPE)
    ln2 = tl.full((), 0.6931471805599453, ACC_DTYPE)

    pid_b = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1).to(tl.int64)
    pid_n = tl.program_id(2).to(tl.int64)

    off_a = tl.full((), 0, tl.int64)
    off_b = tl.full((), 0, tl.int64)
    off_c = tl.full((), 0, tl.int64)

    # Unrolled batch indexing, up to MAX_BATCH_DIMS=6
    if BATCH_DIMS >= 6:
        i5 = (pid_b % b5).to(tl.int64); pid_b = pid_b // b5
        off_a += i5 * sa5; off_b += i5 * sb5; off_c += i5 * sc5
    if BATCH_DIMS >= 5:
        i4 = (pid_b % b4).to(tl.int64); pid_b = pid_b // b4
        off_a += i4 * sa4; off_b += i4 * sb4; off_c += i4 * sc4
    if BATCH_DIMS >= 4:
        i3 = (pid_b % b3).to(tl.int64); pid_b = pid_b // b3
        off_a += i3 * sa3; off_b += i3 * sb3; off_c += i3 * sc3
    if BATCH_DIMS >= 3:
        i2 = (pid_b % b2).to(tl.int64); pid_b = pid_b // b2
        off_a += i2 * sa2; off_b += i2 * sb2; off_c += i2 * sc2
    if BATCH_DIMS >= 2:
        i1 = (pid_b % b1).to(tl.int64); pid_b = pid_b // b1
        off_a += i1 * sa1; off_b += i1 * sb1; off_c += i1 * sc1
    if BATCH_DIMS >= 1:
        i0 = (pid_b % b0).to(tl.int64)
        off_a += i0 * sa0; off_b += i0 * sb0; off_c += i0 * sc0

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)

    mask_m = offs_m < M
    mask_n = offs_n < N

    rm = tl.where(mask_m, offs_m, 0).to(tl.int64)
    rn = tl.where(mask_n, offs_n, 0).to(tl.int64)

    # Running logsumexp over K for each (m, n) element
    m_acc = tl.full((BLOCK_M, BLOCK_N), -float("inf"), ACC_DTYPE)
    s_acc = tl.zeros((BLOCK_M, BLOCK_N), ACC_DTYPE)
    for k0 in range(0, K, BLOCK_K):

        blk_max = tl.full((BLOCK_M, BLOCK_N), -float("inf"), ACC_DTYPE)
        blk_sum = tl.zeros((BLOCK_M, BLOCK_N), ACC_DTYPE)
        for kk in tl.static_range(0, BLOCK_K):
            k = k0 + kk
            mk = k < K

            a_k = tl.load(
                A_ptr + off_a + rm[:, None] * stride_am + k * stride_ak,
                mask=mask_m[:, None] & mk,
                other=-float("inf"),
            ).to(ACC_DTYPE)

            b_k = tl.load(
                B_ptr + off_b + k * stride_bk + rn[None, :] * stride_bn,
                mask=mask_n[None, :] & mk,
                other=-float("inf"),
            ).to(ACC_DTYPE)

            v = a_k + b_k
            if USE_EXP2:
                v2 = v * log2e
                new = tl.maximum(blk_max, v2)
                blk_sum = blk_sum * \
                    tl.where(blk_max == float("-inf"), 0.0, triton_math.exp2(blk_max - new)) + \
                    tl.where(v2 == float("-inf"), 0.0, triton_math.exp2(v2 - new))
            else:
                new = tl.maximum(blk_max, v)
                blk_sum = blk_sum * \
                    tl.where(blk_max == float("-inf"), 0.0, triton_math.exp(blk_max - new)) + \
                    tl.where(v == float("-inf"), 0.0, triton_math.exp(v - new))
            blk_max = new

        new_m = tl.maximum(m_acc, blk_max)
        if USE_EXP2:
            s_acc = s_acc * tl.where(m_acc == float("-inf"), 0.0, triton_math.exp2(m_acc - new_m)) + \
                    blk_sum * tl.where(blk_max == float("-inf"), 0.0, triton_math.exp2(blk_max - new_m))
        else:
            s_acc = s_acc * tl.where(m_acc == float("-inf"), 0.0, triton_math.exp(m_acc - new_m)) + \
                    blk_sum * tl.where(blk_max == float("-inf"), 0.0, triton_math.exp(blk_max - new_m))
        m_acc = new_m

    if USE_EXP2:
        out = (m_acc + triton_math.log2(s_acc)) * ln2
    else:
        out = m_acc + triton_math.log(s_acc)
    out = tl.where(s_acc > 0, out, float("-inf"))

    tl.store(
        C_ptr + off_c + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        out,
        mask=mask_m[:, None] & mask_n[None, :],
    )
