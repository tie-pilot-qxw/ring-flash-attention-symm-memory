from flash_attn import flash_attn_func
import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from ring_flash_attn import (
    ring_flash_attn_func,
    zigzag_ring_flash_attn_func,
    stripe_flash_attn_func,
    ring_flash_attn_symm_func,
)


def benchmark(f, num_iter=100, forward_only=True, log=True, profile=False):
    dtype = torch.bfloat16
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    batch_size = 1
    deterministic = False
    # config of llama3 8B
    seqlen = 1024 * 8
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    causal = True

    assert seqlen % (2 * world_size) == 0
    assert head_dim % 8 == 0

    q = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k_t = torch.randn(
        batch_size,
        seqlen,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v_t = torch.randn(
        batch_size,
        seqlen,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    if f == ring_flash_attn_symm_func:
        local_k = symm_mem.empty(k_t.shape, dtype=dtype, device=device)
        local_v = symm_mem.empty(v_t.shape, dtype=dtype, device=device)
        local_k.copy_(k_t)
        local_v.copy_(v_t)

        local_k_list = []
        local_v_list = []

        local_k_list.append(symm_mem.empty(local_k.shape, dtype=dtype, device=device))
        local_v_list.append(symm_mem.empty(local_v.shape, dtype=dtype, device=device))
        local_k_list.append(symm_mem.empty(local_k.shape, dtype=dtype, device=device))
        local_v_list.append(symm_mem.empty(local_v.shape, dtype=dtype, device=device))
        local_k_list.append(local_k)
        local_v_list.append(local_v)

        k_hdl_list = [symm_mem.rendezvous(local_k_list[i], dist.group.WORLD) for i in range(3)]
        v_hdl_list = [symm_mem.rendezvous(local_v_list[i], dist.group.WORLD) for i in range(3)]

    else:
        k = k_t
        v = v_t
    dout = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype
    )

    if profile:
        torch.backends.cudnn.benchmark = True
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=5,
                warmup=5,
                active=5,
            ),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(
                    f"./benchmark/logs/{f.__name__}", f"rank_{dist.get_rank()}"
                )
            ),
        )

    if profile:
        profiler.start()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                if f == ring_flash_attn_symm_func:
                    _ = f(
                        q,
                        local_k_list,
                        local_v_list,
                        k_hdl_list,
                        v_hdl_list,
                        causal=causal,
                        window_size=(-1, -1),
                        alibi_slopes=None,
                        deterministic=deterministic,
                        return_attn_probs=False,
                    )
                else:
                    _ = f(
                        q,
                        k,
                        v,
                        causal=causal,
                        window_size=(-1, -1),
                        alibi_slopes=None,
                        deterministic=deterministic,
                        return_attn_probs=False,
                    )
                if profile:
                    profiler.step()

    else:
        for _ in range(num_iter):
            q.grad = None
            k.grad = None
            v.grad = None
            out = f(
                q,
                k,
                v,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=False,
            )
            out.backward(dout)
            if profile:
                profiler.step()

    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if profile:
        profiler.stop()

    if rank == 0 and log:
        print(f"{num_iter / time:.3f} iter/s, {time:.3f} sec")


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    forward_only = True
    profile = True
    num_iter = 1 if forward_only else 100

    for f in [
        flash_attn_func,
        ring_flash_attn_func,
        ring_flash_attn_symm_func,
        # zigzag_ring_flash_attn_func,
        # stripe_flash_attn_func,
    ]:
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"# {f.__name__}")
        benchmark(f, forward_only=forward_only, num_iter=num_iter, log=False)
        benchmark(
            f, forward_only=forward_only, num_iter=num_iter, log=True, profile=profile
        )
