import triton
import triton.language as tl
from . import triton_math


MAX_BATCH_DIMS = 6

def _lse_bwd_autotune_configs() -> list[triton.Config]:
    # One conservative tile avoids compiling large unrolled float64 libdevice kernels.
    # A single configuration also skips the autotuning benchmark sweep.
    return [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 8}, num_warps=4, num_stages=2),
    ]


@triton.autotune(
    configs=_lse_bwd_autotune_configs(),
    key=["M", "N", "K", "USE_EXP2"],
)
@triton.jit
def _lse_bwd_dA_kernel(
    A_ptr, B_ptr, C_ptr, G_ptr,
    dA_ptr,
    M, N, K,
    b0: tl.constexpr, b1: tl.constexpr, b2: tl.constexpr, b3: tl.constexpr, b4: tl.constexpr, b5: tl.constexpr,
    sa0: tl.constexpr, sa1: tl.constexpr, sa2: tl.constexpr, sa3: tl.constexpr, sa4: tl.constexpr, sa5: tl.constexpr,
    sb0: tl.constexpr, sb1: tl.constexpr, sb2: tl.constexpr, sb3: tl.constexpr, sb4: tl.constexpr, sb5: tl.constexpr,
    sc0: tl.constexpr, sc1: tl.constexpr, sc2: tl.constexpr, sc3: tl.constexpr, sc4: tl.constexpr, sc5: tl.constexpr,
    sg0: tl.constexpr, sg1: tl.constexpr, sg2: tl.constexpr, sg3: tl.constexpr, sg4: tl.constexpr, sg5: tl.constexpr,
    sda0: tl.constexpr, sda1: tl.constexpr, sda2: tl.constexpr, sda3: tl.constexpr, sda4: tl.constexpr, sda5: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_gm, stride_gn,
    stride_dam, stride_dak,
    BATCH_DIMS: tl.constexpr,
    USE_EXP2: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    ACC_DTYPE: tl.constexpr = tl.float64 if A_ptr.dtype.element_ty == tl.float64 else tl.float32
    log2e = tl.full((), 1.4426950408889634, ACC_DTYPE)

    pid_b = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1).to(tl.int64)
    pid_k = tl.program_id(2).to(tl.int64)

    off_a = tl.full((), 0, tl.int64)
    off_b = tl.full((), 0, tl.int64)
    off_c = tl.full((), 0, tl.int64)
    off_g = tl.full((), 0, tl.int64)
    off_da = tl.full((), 0, tl.int64)

    # Unrolled batch indexing, up to MAX_BATCH_DIMS=6
    if BATCH_DIMS >= 6:
        i5 = (pid_b % b5).to(tl.int64); pid_b = pid_b // b5
        off_a += i5 * sa5; off_b += i5 * sb5; off_c += i5 * sc5; off_g += i5 * sg5; off_da += i5 * sda5
    if BATCH_DIMS >= 5:
        i4 = (pid_b % b4).to(tl.int64); pid_b = pid_b // b4
        off_a += i4 * sa4; off_b += i4 * sb4; off_c += i4 * sc4; off_g += i4 * sg4; off_da += i4 * sda4
    if BATCH_DIMS >= 4:
        i3 = (pid_b % b3).to(tl.int64); pid_b = pid_b // b3
        off_a += i3 * sa3; off_b += i3 * sb3; off_c += i3 * sc3; off_g += i3 * sg3; off_da += i3 * sda3
    if BATCH_DIMS >= 3:
        i2 = (pid_b % b2).to(tl.int64); pid_b = pid_b // b2
        off_a += i2 * sa2; off_b += i2 * sb2; off_c += i2 * sc2; off_g += i2 * sg2; off_da += i2 * sda2
    if BATCH_DIMS >= 2:
        i1 = (pid_b % b1).to(tl.int64); pid_b = pid_b // b1
        off_a += i1 * sa1; off_b += i1 * sb1; off_c += i1 * sc1; off_g += i1 * sg1; off_da += i1 * sda1
    if BATCH_DIMS >= 1:
        i0 = (pid_b % b0).to(tl.int64)
        off_a += i0 * sa0; off_b += i0 * sb0; off_c += i0 * sc0; off_g += i0 * sg0; off_da += i0 * sda0

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)

    mask_m = rm < M
    mask_k = rk < K

    a_tile = tl.load(
        A_ptr + off_a + rm[:, None] * stride_am + rk[None, :] * stride_ak,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    ).to(ACC_DTYPE)

    acc = tl.zeros((BLOCK_M, BLOCK_K), ACC_DTYPE)

    for n0 in range(0, N, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = rn < N
        mn_mask = mask_m[:, None] & mask_n[None, :]

        c_tile = tl.load(
            C_ptr + off_c + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
            mask=mn_mask,
            other=0.0,
        ).to(ACC_DTYPE)
        g_tile = tl.load(
            G_ptr + off_g + rm[:, None] * stride_gm + rn[None, :] * stride_gn,
            mask=mn_mask,
            other=0.0,
        ).to(ACC_DTYPE)

        b_tile = tl.load(
            B_ptr + off_b + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(ACC_DTYPE)

        for kk in tl.static_range(0, BLOCK_K):
            ak = a_tile[:, kk]
            bk = b_tile[kk, :]
            logits = ak[:, None] + bk[None, :] - c_tile
            logits = tl.where(mn_mask, logits, -float("inf"))

            if USE_EXP2:
                p = triton_math.exp2(logits * log2e)
            else:
                p = triton_math.exp(logits)

            acc_k = tl.sum(g_tile * p, axis=1)
            acc = tl.multiple_of(acc, (1, 1))
            acc = tl.where(
                tl.arange(0, BLOCK_K)[None, :] == kk,
                acc + acc_k[:, None],
                acc,
            )

    tl.store(
        dA_ptr + off_da + rm[:, None] * stride_dam + rk[None, :] * stride_dak,
        acc,
        mask=mask_m[:, None] & mask_k[None, :],
    )


@triton.autotune(
    configs=_lse_bwd_autotune_configs(),
    key=["M", "N", "K", "USE_EXP2"],
)
@triton.jit
def _lse_bwd_dA_kernel(
    A_ptr, B_ptr, C_ptr, G_ptr,
    dA_ptr,
    M, N, K,
    b0: tl.constexpr, b1: tl.constexpr, b2: tl.constexpr, b3: tl.constexpr, b4: tl.constexpr, b5: tl.constexpr,
    sa0: tl.constexpr, sa1: tl.constexpr, sa2: tl.constexpr, sa3: tl.constexpr, sa4: tl.constexpr, sa5: tl.constexpr,
    sb0: tl.constexpr, sb1: tl.constexpr, sb2: tl.constexpr, sb3: tl.constexpr, sb4: tl.constexpr, sb5: tl.constexpr,
    sc0: tl.constexpr, sc1: tl.constexpr, sc2: tl.constexpr, sc3: tl.constexpr, sc4: tl.constexpr, sc5: tl.constexpr,
    sg0: tl.constexpr, sg1: tl.constexpr, sg2: tl.constexpr, sg3: tl.constexpr, sg4: tl.constexpr, sg5: tl.constexpr,
    sda0: tl.constexpr, sda1: tl.constexpr, sda2: tl.constexpr, sda3: tl.constexpr, sda4: tl.constexpr, sda5: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_gm, stride_gn,
    stride_dam, stride_dak,
    BATCH_DIMS: tl.constexpr,
    USE_EXP2: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    ACC_DTYPE: tl.constexpr = tl.float64 if A_ptr.dtype.element_ty == tl.float64 else tl.float32
    log2e = tl.full((), 1.4426950408889634, ACC_DTYPE)

    pid_b = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1).to(tl.int64)
    pid_k = tl.program_id(2).to(tl.int64)

    off_a = tl.full((), 0, tl.int64)
    off_b = tl.full((), 0, tl.int64)
    off_c = tl.full((), 0, tl.int64)
    off_g = tl.full((), 0, tl.int64)
    off_da = tl.full((), 0, tl.int64)

    if BATCH_DIMS >= 6:
        i5 = (pid_b % b5).to(tl.int64); pid_b //= b5
        off_a += i5 * sa5; off_b += i5 * sb5; off_c += i5 * sc5; off_g += i5 * sg5; off_da += i5 * sda5
    if BATCH_DIMS >= 5:
        i4 = (pid_b % b4).to(tl.int64); pid_b //= b4
        off_a += i4 * sa4; off_b += i4 * sb4; off_c += i4 * sc4; off_g += i4 * sg4; off_da += i4 * sda4
    if BATCH_DIMS >= 4:
        i3 = (pid_b % b3).to(tl.int64); pid_b //= b3
        off_a += i3 * sa3; off_b += i3 * sb3; off_c += i3 * sc3; off_g += i3 * sg3; off_da += i3 * sda3
    if BATCH_DIMS >= 3:
        i2 = (pid_b % b2).to(tl.int64); pid_b //= b2
        off_a += i2 * sa2; off_b += i2 * sb2; off_c += i2 * sc2; off_g += i2 * sg2; off_da += i2 * sda2
    if BATCH_DIMS >= 2:
        i1 = (pid_b % b1).to(tl.int64); pid_b //= b1
        off_a += i1 * sa1; off_b += i1 * sb1; off_c += i1 * sc1; off_g += i1 * sg1; off_da += i1 * sda1
    if BATCH_DIMS >= 1:
        i0 = (pid_b % b0).to(tl.int64)
        off_a += i0 * sa0; off_b += i0 * sb0; off_c += i0 * sc0; off_g += i0 * sg0; off_da += i0 * sda0

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rk0 = pid_k * BLOCK_K

    mask_m = rm < M

    acc = tl.zeros((BLOCK_M, BLOCK_K), ACC_DTYPE)

    # loop over N blocks
    for n0 in range(0, N, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = rn < N
        mn_mask = mask_m[:, None] & mask_n[None, :]

        c_tile = tl.load(
            C_ptr + off_c + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
            mask=mn_mask,
            other=0.0,
        ).to(ACC_DTYPE)
        g_tile = tl.load(
            G_ptr + off_g + rm[:, None] * stride_gm + rn[None, :] * stride_gn,
            mask=mn_mask,
            other=0.0,
        ).to(ACC_DTYPE)

        # loop over K within this block (no tensor indexing by constexpr)
        for kk in tl.static_range(0, BLOCK_K):
            k = rk0 + kk
            mk = k < K

            a_k = tl.load(
                A_ptr + off_a + rm * stride_am + k * stride_ak,
                mask=mask_m & mk,
                other=0.0,
            ).to(ACC_DTYPE)  # (BM,)

            b_k = tl.load(
                B_ptr + off_b + k * stride_bk + rn * stride_bn,
                mask=mask_n & mk,
                other=0.0,
            ).to(ACC_DTYPE)  # (BN,)

            logits = a_k[:, None] + b_k[None, :] - c_tile
            logits = tl.where(mn_mask, logits, -float("inf"))
            logits = tl.minimum(logits, 0.0)  # hard guard against overflow

            if USE_EXP2:
                p = triton_math.exp2(logits * log2e)
            else:
                p = triton_math.exp(logits)

            acc_k = tl.sum(g_tile * p, axis=1)  # (BM,)

            one_hot = (tl.arange(0, BLOCK_K) == kk)[None, :].to(ACC_DTYPE)
            acc += acc_k[:, None] * one_hot

    rk = rk0 + tl.arange(0, BLOCK_K).to(tl.int64)
    mask_k = rk < K

    tl.store(
        dA_ptr + off_da + rm[:, None] * stride_dam + rk[None, :] * stride_dak,
        acc,
        mask=mask_m[:, None] & mask_k[None, :],
    )


@triton.autotune(
    configs=_lse_bwd_autotune_configs(),
    key=["M", "N", "K", "USE_EXP2"],
)
@triton.jit
def _lse_bwd_dB_kernel(
    A_ptr, B_ptr, C_ptr, G_ptr,
    dB_ptr,
    M, N, K,
    b0: tl.constexpr, b1: tl.constexpr, b2: tl.constexpr, b3: tl.constexpr, b4: tl.constexpr, b5: tl.constexpr,
    sa0: tl.constexpr, sa1: tl.constexpr, sa2: tl.constexpr, sa3: tl.constexpr, sa4: tl.constexpr, sa5: tl.constexpr,
    sb0: tl.constexpr, sb1: tl.constexpr, sb2: tl.constexpr, sb3: tl.constexpr, sb4: tl.constexpr, sb5: tl.constexpr,
    sc0: tl.constexpr, sc1: tl.constexpr, sc2: tl.constexpr, sc3: tl.constexpr, sc4: tl.constexpr, sc5: tl.constexpr,
    sg0: tl.constexpr, sg1: tl.constexpr, sg2: tl.constexpr, sg3: tl.constexpr, sg4: tl.constexpr, sg5: tl.constexpr,
    sdb0: tl.constexpr, sdb1: tl.constexpr, sdb2: tl.constexpr, sdb3: tl.constexpr, sdb4: tl.constexpr, sdb5: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_gm, stride_gn,
    stride_dbk, stride_dbn,
    BATCH_DIMS: tl.constexpr,
    USE_EXP2: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    ACC_DTYPE: tl.constexpr = tl.float64 if A_ptr.dtype.element_ty == tl.float64 else tl.float32
    log2e = tl.full((), 1.4426950408889634, ACC_DTYPE)

    pid_b = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    pid_n = tl.program_id(2).to(tl.int64)

    off_a = tl.full((), 0, tl.int64)
    off_b = tl.full((), 0, tl.int64)
    off_c = tl.full((), 0, tl.int64)
    off_g = tl.full((), 0, tl.int64)
    off_db = tl.full((), 0, tl.int64)

    if BATCH_DIMS >= 6:
        i5 = (pid_b % b5).to(tl.int64); pid_b //= b5
        off_a += i5 * sa5; off_b += i5 * sb5; off_c += i5 * sc5; off_g += i5 * sg5; off_db += i5 * sdb5
    if BATCH_DIMS >= 5:
        i4 = (pid_b % b4).to(tl.int64); pid_b //= b4
        off_a += i4 * sa4; off_b += i4 * sb4; off_c += i4 * sc4; off_g += i4 * sg4; off_db += i4 * sdb4
    if BATCH_DIMS >= 4:
        i3 = (pid_b % b3).to(tl.int64); pid_b //= b3
        off_a += i3 * sa3; off_b += i3 * sb3; off_c += i3 * sc3; off_g += i3 * sg3; off_db += i3 * sdb3
    if BATCH_DIMS >= 3:
        i2 = (pid_b % b2).to(tl.int64); pid_b //= b2
        off_a += i2 * sa2; off_b += i2 * sb2; off_c += i2 * sc2; off_g += i2 * sg2; off_db += i2 * sdb2
    if BATCH_DIMS >= 2:
        i1 = (pid_b % b1).to(tl.int64); pid_b //= b1
        off_a += i1 * sa1; off_b += i1 * sb1; off_c += i1 * sc1; off_g += i1 * sg1; off_db += i1 * sdb1
    if BATCH_DIMS >= 1:
        i0 = (pid_b % b0).to(tl.int64)
        off_a += i0 * sa0; off_b += i0 * sb0; off_c += i0 * sc0; off_g += i0 * sg0; off_db += i0 * sdb0

    rk0 = pid_k * BLOCK_K
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)

    mask_n = rn < N
    acc = tl.zeros((BLOCK_K, BLOCK_N), ACC_DTYPE)

    # loop over M blocks
    for m0 in range(0, M, BLOCK_M):
        rm = m0 + tl.arange(0, BLOCK_M).to(tl.int64)
        mask_m = rm < M
        mn_mask = mask_m[:, None] & mask_n[None, :]

        c_tile = tl.load(
            C_ptr + off_c + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
            mask=mn_mask,
            other=0.0,
        ).to(ACC_DTYPE)
        g_tile = tl.load(
            G_ptr + off_g + rm[:, None] * stride_gm + rn[None, :] * stride_gn,
            mask=mn_mask,
            other=0.0,
        ).to(ACC_DTYPE)

        for kk in tl.static_range(0, BLOCK_K):
            k = rk0 + kk
            mk = k < K

            a_k = tl.load(
                A_ptr + off_a + rm * stride_am + k * stride_ak,
                mask=mask_m & mk,
                other=0.0,
            ).to(ACC_DTYPE)  # (BM,)

            b_k = tl.load(
                B_ptr + off_b + k * stride_bk + rn * stride_bn,
                mask=mask_n & mk,
                other=0.0,
            ).to(ACC_DTYPE)  # (BN,)

            logits = a_k[:, None] + b_k[None, :] - c_tile
            logits = tl.where(mn_mask, logits, -float("inf"))
            logits = tl.minimum(logits, 0.0)

            if USE_EXP2:
                p = triton_math.exp2(logits * log2e)
            else:
                p = triton_math.exp(logits)

            acc_k = tl.sum(g_tile * p, axis=0)  # (BN,)
            one_hot = (tl.arange(0, BLOCK_K) == kk)[:, None].to(ACC_DTYPE)
            acc += one_hot * acc_k[None, :]

    rk = rk0 + tl.arange(0, BLOCK_K).to(tl.int64)
    mask_k = rk < K

    tl.store(
        dB_ptr + off_db + rk[:, None] * stride_dbk + rn[None, :] * stride_dbn,
        acc,
        mask=mask_k[:, None] & mask_n[None, :],
    )
