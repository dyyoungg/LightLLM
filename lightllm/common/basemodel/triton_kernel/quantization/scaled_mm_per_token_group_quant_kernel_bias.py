import torch
import triton
import triton.language as tl
from typing import Optional
from lightllm.common.triton_utils.autotuner import autotune
from lightllm.utils.device_utils import triton_support_tensor_descriptor, is_5090_gpu

@triton.jit
def grouped_launch(pid, m_block_num, n_block_num, group_m: tl.constexpr):
    num_pid_in_group = group_m * n_block_num
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = tl.minimum(m_block_num - first_pid_m, group_m)
    in_group_index = pid % num_pid_in_group

    # Swizzle pattern: zigzag traversal
    back_mark = (in_group_index // group_size_m) % 2
    back_mark1 = -1 * (2 * back_mark - 1)
    pid_m = first_pid_m + back_mark * (group_size_m - 1) + back_mark1 * (in_group_index % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n

@triton.jit
def _scaled_mm_act_per_group_w_perchannel_kernel(
    A,  # [m, k]
    A_desc: "tl.core.tensor_descriptor",
    B,  # [k, n]
    B_desc: "tl.core.tensor_descriptor",
    out,  # [m, n]
    out_desc: "tl.core.tensor_descriptor",
    Ascale,  # [m, k // 128]
    a_scale_stride_m,
    a_scale_stride_k,
    Bscale,  # [n,]
    bias,    # [n,] <-- 新增：Bias 指针
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    act_quant_group_size,
    USE_TMA: tl.constexpr,
    B_IS_TRANS: tl.constexpr,
    NEED_N_MASK: tl.constexpr,
    NEED_K_MASK: tl.constexpr,
    HAS_BIAS: tl.constexpr, # <-- 新增：是否包含 Bias 的标志
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    m_block_num = tl.cdiv(M, BLOCK_M)
    n_block_num = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = grouped_launch(pid, m_block_num, n_block_num, GROUP_M)

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    offs_am = start_m + tl.arange(0, BLOCK_M)
    offs_bn = start_n + tl.arange(0, BLOCK_N)

    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)

    offs_k = tl.arange(0, BLOCK_K)

    if not USE_TMA:
        a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    Ascale_ptrs = Ascale + offs_am * a_scale_stride_m
    Bscale_ptrs = Bscale + offs_bn
    b_s = tl.load(Bscale_ptrs)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_s = tl.load(Ascale_ptrs + (((k * BLOCK_K) // act_quant_group_size) * a_scale_stride_k))
        if USE_TMA:
            a = A_desc.load([start_m, k * BLOCK_K])
            if not B_IS_TRANS:
                b = B_desc.load([k * BLOCK_K, start_n])
            else:
                b = B_desc.load([start_n, k * BLOCK_K]).T
        elif NEED_K_MASK:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        if not USE_TMA:
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    # ----------------------------------------------------
    # 新增逻辑：Epilogue 阶段融合 Bias 计算
    # ----------------------------------------------------
    acc = acc.to(tl.float32)
    
    if HAS_BIAS:
        offs_bn_bias = start_n + tl.arange(0, BLOCK_N)
        if NEED_N_MASK:
            bias_val = tl.load(bias + offs_bn_bias, mask=offs_bn_bias < N, other=0.0)
        else:
            bias_val = tl.load(bias + offs_bn_bias)
        # 广播 bias 到 [BLOCK_M, BLOCK_N] 并相加
        acc += bias_val[None, :].to(tl.float32)

    acc = acc.to(out.dtype.element_ty)
    # ----------------------------------------------------

    if not USE_TMA:
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = out + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        if NEED_N_MASK:
            mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        else:
            mask = offs_cm[:, None] < M
        tl.store(c_ptrs, acc, mask=mask)
    else:
        out_desc.store([start_m, start_n], acc)

def get_test_configs():
    fp8_gemm_configs = []
    for BLOCK_M in [8, 16, 32, 64]:
        for BLOCK_N in [64, 128, 256]:
            for BLOCK_K in [32, 64, 128]:
                if BLOCK_K * BLOCK_M * BLOCK_N >= 256 * 256 * 128:
                    continue
                for num_warps in [2, 4, 8]:
                    for num_stages in [2, 3, 4, 5, 6]:
                        config = {
                            "BLOCK_M": BLOCK_M,
                            "BLOCK_N": BLOCK_N,
                            "BLOCK_K": BLOCK_K,
                            "GROUP_M": 8,
                            "num_stages": num_stages,
                            "num_warps": num_warps,
                        }
                        fp8_gemm_configs.append(config)
    return fp8_gemm_configs

def _get_static_key(A, B, act_quant_group_size):
    M, K = A.shape
    _, N = B.shape
    return {
        "N": N,
        "K": K,
        "act_quant_group_size": act_quant_group_size,
    }

@autotune(
    kernel_name="scaled_mm_act_per_group_w_perchannel:v1",
    configs_gen_func=get_test_configs,
    static_key_func=_get_static_key,
    run_key_func=lambda A: A.shape[0],
    mutates_args=["out"],
)
def scaled_mm_act_per_group_w_perchannel(
    A: torch.Tensor,
    B: torch.Tensor,
    Ascale: torch.Tensor,
    Bscale: torch.Tensor,
    act_quant_group_size: int,
    out: torch.Tensor,
    bias: Optional[torch.Tensor] = None, # <-- 新增：Wrapper 层加入 Bias 参数
    run_config=None,
) -> torch.Tensor:
    """w8a8 per-token quantization mm (supports fp8 and int8)."""
    assert A.is_contiguous()
    B_is_trans = not B.is_contiguous() and B.stride(0) == 1

    M, K = A.shape
    _, N = B.shape
    if not run_config:
        run_config = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 64,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }
    NEED_N_MASK = N % run_config["BLOCK_N"] != 0
    NEED_K_MASK = K % run_config["BLOCK_K"] != 0
    grid = (triton.cdiv(M, run_config["BLOCK_M"]) * triton.cdiv(N, run_config["BLOCK_N"]),)

    BLOCK_M = run_config["BLOCK_M"]
    BLOCK_K = run_config["BLOCK_K"]
    BLOCK_N = run_config["BLOCK_N"]

    support_tma = triton_support_tensor_descriptor()
    support_tma = support_tma and (not is_5090_gpu())
    if support_tma:
        stride = A.stride(-2)
        if (stride * A.dtype.itemsize) % 16 != 0:
            support_tma = False
        _B = B if not B_is_trans else B.transpose(0, 1)
        stride = _B.stride(-2)
        if (stride * _B.dtype.itemsize) % 16 != 0:
            support_tma = False

    if support_tma:
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)
        from triton.tools.tensor_descriptor import TensorDescriptor

        A_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_M, BLOCK_K])
        if B_is_trans:
            _B = B.transpose(0, 1)
            assert _B.is_contiguous()
            B_desc = TensorDescriptor(_B, _B.shape, _B.stride(), [BLOCK_N, BLOCK_K])
        else:
            B_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_K, BLOCK_N])
        out_desc = TensorDescriptor(out, out.shape, out.stride(), [BLOCK_M, BLOCK_N])
    else:
        A_desc = None
        B_desc = None
        out_desc = None

    assert BLOCK_M <= act_quant_group_size, "Currently we require BLOCK_M <= act_quant_group_size"
    ACC_DTYPE = tl.int32 if A.dtype == torch.int8 else tl.float32

    _scaled_mm_act_per_group_w_perchannel_kernel[grid](
        A=A,
        A_desc=A_desc,
        B=B,
        B_desc=B_desc,
        out=out,
        out_desc=out_desc,
        Ascale=Ascale,
        a_scale_stride_m=Ascale.stride(0),
        a_scale_stride_k=Ascale.stride(1),
        Bscale=Bscale,
        bias=bias, # <-- 传入 bias
        M=M,
        N=N,
        K=K,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_bk=B.stride(0),
        stride_bn=B.stride(1),
        stride_cm=out.stride(0),
        stride_cn=out.stride(1),
        act_quant_group_size=act_quant_group_size,
        USE_TMA=support_tma,
        B_IS_TRANS=B_is_trans,
        NEED_N_MASK=NEED_N_MASK,
        NEED_K_MASK=NEED_K_MASK,
        HAS_BIAS=bias is not None, # <-- 传入 HAS_BIAS
        ACC_DTYPE=ACC_DTYPE,
        **run_config,
    )

    return out

if __name__ == "__main__":
    import time
    import os
    from lightllm.common.triton_utils.autotuner import Autotuner
    import torch.nn.functional as F

    output_dtype = torch.bfloat16
    N, K = 4096, 5120
    act_quant_group_size = 128

    M_list = [1, 2, 4, 8, 16, 32, 48, 96, 128, 256]

    print(f"{'='*80}")
    print(f"Starting Autotune for Scaled MM (N={N}, K={K})")
    print(f"M values to test: {M_list}")
    print(f"Total configs per M: {len(get_test_configs())}")
    print(f"{'='*80}\n")

    # 准备权重和全局共享测试数据
    B = torch.randn((N, K), dtype=output_dtype).cuda().to(torch.float8_e4m3fn).transpose(0, 1)
    Bscale = torch.ones((1, N)).cuda()
    
    # ----------------------------------------------------
    # 新增：准备一个全局的 Bias Tensor 进行测试
    # ----------------------------------------------------
    bias = torch.randn((N,), dtype=output_dtype).cuda()

    test_data = {}
    for M in M_list:
        A = torch.randn((M, K), dtype=output_dtype).cuda().to(torch.float8_e4m3fn)
        Ascale = torch.randn((M, K // act_quant_group_size)).cuda().fill_(1.0)
        out = torch.zeros((M, N), dtype=output_dtype).cuda()
        test_data[M] = {"A": A, "Ascale": Ascale, "out": out}

    print("\n" + "=" * 80)
    print("PHASE 0: Verifying Correctness Before Autotune")
    print("=" * 80)

    M_verify = 16 if 16 in M_list else M_list[len(M_list) // 2]
    A_verify = test_data[M_verify]["A"]
    Ascale_verify = test_data[M_verify]["Ascale"]
    out_verify = test_data[M_verify]["out"]

    # 计算 ground truth 时显式加上 bias
    d_A = A_verify.view(-1, K // act_quant_group_size, act_quant_group_size).to(output_dtype) * Ascale_verify.view(
        -1, K // act_quant_group_size, 1
    ).to(output_dtype)
    d_A = d_A.view(-1, K)
    d_B = B.to(output_dtype) * Bscale.to(output_dtype)
    gt_C = d_A.mm(d_B) + bias  # <-- 新增：加上 Bias 算 GT

    # 运行包含 bias 的 kernel 进行测试
    scaled_mm_act_per_group_w_perchannel(A_verify, B, Ascale_verify, Bscale, act_quant_group_size, out_verify, bias=bias)

    cosine_sim = F.cosine_similarity(out_verify.flatten().unsqueeze(0), gt_C.flatten().unsqueeze(0), dim=1)
    print(f"[Verification] Cosine Similarity: {cosine_sim.item():.6f}")

    if cosine_sim.item() < 0.99:
        raise RuntimeError(f"Correctness check failed! Cosine similarity {cosine_sim.item():.6f} < 0.99")

    print("[Verification] ✅ Correctness check passed!")

    print("\n" + "=" * 80)
    print("PHASE 1: Running Autotune")
    print("=" * 80)
    Autotuner.start_autotune_warmup()

    for M in M_list:
        A = test_data[M]["A"]
        Ascale = test_data[M]["Ascale"]
        out = test_data[M]["out"]
        # 传递 bias 参与调优
        scaled_mm_act_per_group_w_perchannel(A, B, Ascale, Bscale, act_quant_group_size, out, bias=bias)

    Autotuner.end_autotune_warmup()

    print("\n" + "=" * 80)
    print("PHASE 2: Benchmark")
    results = []
    
    # 请注意：如果在无 sgl_kernel 的环境下运行，请注释掉涉及 sgl 的代码
    try:
        from sgl_kernel import fp8_scaled_mm
        has_sgl = True
    except ImportError:
        has_sgl = False
        print("sgl_kernel not found. Skipping SGL benchmarks.")

    for M in M_list:
        A = test_data[M]["A"]
        Ascale = test_data[M]["Ascale"]
        out = test_data[M]["out"]

        d_A = A.view(-1, K // act_quant_group_size, act_quant_group_size).to(output_dtype) * Ascale.view(
            -1, K // act_quant_group_size, 1
        ).to(output_dtype)
        d_A = d_A.view(-1, K)
        d_B = B.to(output_dtype) * Bscale.to(output_dtype)
        gt_C = d_A.mm(d_B) + bias  # <-- 计算 GT 时加 Bias

        # Our kernel run
        scaled_mm_act_per_group_w_perchannel(A, B, Ascale, Bscale, act_quant_group_size, out, bias=bias)
        cosine_sim = F.cosine_similarity(out.flatten().unsqueeze(0), gt_C.flatten().unsqueeze(0), dim=1)

        if has_sgl:
            # SGL kernel 通常不自带 bias 参数，需要在外部模拟才能保持结果一致
            sgl_res = fp8_scaled_mm(A, B, Ascale[:, 0].contiguous(), Bscale, output_dtype) + bias
            sgl_cosine_sim = F.cosine_similarity(sgl_res.flatten().unsqueeze(0), gt_C.flatten().unsqueeze(0), dim=1)
            print(f"[M={M}] Cosine Similarity - Our: {cosine_sim.item():.6f}, SGL: {sgl_cosine_sim.item():.6f}")
        else:
            print(f"[M={M}] Cosine Similarity - Our: {cosine_sim.item():.6f}")

        # Benchmark 性能
        fn_bf16 = lambda: torch.mm(d_A, d_B) + bias
        ms_bf16 = triton.testing.do_bench(fn_bf16, warmup=25, rep=100)

        if has_sgl:
            # 注意：这里的基准测试 SGL 包含了外部追加 Bias 的开销
            fn_sgl = lambda: fp8_scaled_mm(A, B, Ascale[:, 0].contiguous(), Bscale, output_dtype) + bias
            ms_sgl = triton.testing.do_bench(fn_sgl, warmup=25, rep=100)
        else:
            ms_sgl = float('inf')

        # 包含 Bias 读取计算的 Our kernel
        fn_ours = lambda: scaled_mm_act_per_group_w_perchannel(A, B, Ascale, Bscale, act_quant_group_size, out, bias=bias)
        ms_ours = triton.testing.do_bench_cudagraph(fn_ours, rep=100)

        results.append({
            "M": M,
            "bf16_ms": ms_bf16,
            "sgl_ms": ms_sgl,
            "ours_ms": ms_ours,
        })

    # 打印汇总
    print(f"\n{'='*80}")
    print("SUMMARY - Performance Comparison")
    print(f"{'='*80}")
    print(f"{'M':<8} {'BF16(ms)':<12} {'SGL(ms)':<12} {'Our(ms)':<12} {'vs BF16':<10} {'vs SGL':<10}")
    print(f"{'-'*80}")
    for r in results:
        vs_bf16 = f"{r['bf16_ms']/r['ours_ms']:.2f}x"
        vs_sgl = f"{r['sgl_ms']/r['ours_ms']:.2f}x" if has_sgl else "N/A"
        print(
            f"{r['M']:<8} {r['bf16_ms']:<12.3f} {r['sgl_ms']:<12.3f}"
            f"{r['ours_ms']:<12.3f} {vs_bf16:<10} {vs_sgl:<10}"
        )
    print(f"{'='*80}")