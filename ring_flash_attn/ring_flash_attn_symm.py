from typing import List, Tuple
import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .utils import RingCommSymm, RingComm, update_out_and_lse, get_default_args
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._symmetric_memory import _SymmetricMemory

# def log(msg, a, rank0_only=False):
#     world_size = dist.get_world_size()
#     rank = dist.get_rank()
#     if rank0_only:
#         if rank == 0:
#             print(
#                 f"{msg}: "
#                 f"max {a.abs().max().item():.3g}, "
#                 f"mean {a.abs().mean().item():.3g}",
#                 flush=True,
#             )
#         return

#     for i in range(world_size):
#         if i == rank:
#             if rank == 0:
#                 print(f"{msg}:")
#             print(
#                 f"[{rank}] "
#                 f"max {a.abs().max().item():.3g}, "
#                 f"mean {a.abs().mean().item():.3g}",
#                 flush=True,
#             )
#         dist.barrier()

def ring_flash_attn_symm_forward(
    process_group,
    q: torch.Tensor, 
    k: List[torch.Tensor], # must be symmetric memory
    v: List[torch.Tensor], # must be symmetric memory
    k_hdl: List[_SymmetricMemory],
    v_hdl: List[_SymmetricMemory],
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    k_hdl[0].barrier()
    comm = RingCommSymm(process_group, k, v, k_hdl, v_hdl)
    # comm_truth = RingComm(process_group)

    out = None
    lse = None

    # k_truth, v_truth = k[2], v[2]

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            comm.send_recv_kv(step, step + 2 != comm.world_size)
            # next_k, next_v = comm_truth.send_recv_kv(k_truth, v_truth)

        with torch.cuda.stream(comm.stream[comm.compute]):
            if step == 0:
                k, v = comm.k[2], comm.v[2]
            else:
                k, v = comm.k[comm.compute], comm.v[comm.compute]

            # log("diff k", k - k_truth)
            # log("diff v", v - v_truth)
            if not causal or step <= comm.rank:
                params = get_default_args(_flash_attn_forward).copy()
                params.update(
                    {
                        "q": q,
                        "k": k,
                        "v": v,
                        "dropout_p": dropout_p,
                        "softmax_scale": softmax_scale,
                        "causal": causal and step == 0,
                        "alibi_slopes": alibi_slopes,
                        "return_softmax": True and dropout_p > 0,
                    }
                )
                if "window_size" in params:
                    params.update({"window_size": window_size})
                else:
                    params.update(
                        {
                            "window_size_left": window_size[0],
                            "window_size_right": window_size[1],
                        }
                    )
                outputs = _flash_attn_forward(**params)
                if len(outputs) == 8:
                    block_out, _, _, _, _, block_lse, _, _ = outputs
                else:
                    assert len(outputs) == 4
                    block_out, block_lse, _, _ = outputs
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        comm.step()
        # if step + 1 != comm.world_size:
        #     comm_truth.wait()
        #     k_truth, v_truth = next_k, next_v

    comm.sync()
    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    k_hdl[0].barrier()
    return out, lse


def ring_flash_attn_symm_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    kv_comm = RingCommSymm(process_group)
    d_kv_comm = RingCommSymm(process_group)

    k_hdl = symm_mem.rendezvous(k, process_group)
    assert k_hdl is not None

    v_hdl = symm_mem.rendezvous(v, process_group)
    assert v_hdl is not None

    next_k = symm_mem.empty(k.shape, dtype=k.dtype, device=k.device)
    next_k_hdl = symm_mem.rendezvous(next_k, process_group)
    assert next_k_hdl is not None

    next_v = symm_mem.empty(v.shape, dtype=v.dtype, device=v.device)
    next_v_hdl = symm_mem.rendezvous(next_v, process_group)
    assert next_v_hdl is not None

    dk = symm_mem.empty(k.shape, dtype=torch.float32, device=k.device)
    dk_hdl = symm_mem.rendezvous(dk, process_group)
    assert dk_hdl is not None

    dv = symm_mem.empty(v.shape, dtype=torch.float32, device=v.device)
    dv_hdl = symm_mem.rendezvous(dv, process_group)
    assert dv_hdl is not None

    next_dk = symm_mem.empty(k.shape, dtype=torch.float32, device=k.device)
    next_dk_hdl = symm_mem.rendezvous(next_dk, process_group)
    assert next_dk_hdl is not None

    next_dv = symm_mem.empty(v.shape, dtype=torch.float32, device=v.device)
    next_dv_hdl = symm_mem.rendezvous(next_dv, process_group)
    assert next_dv_hdl is not None

    dq = torch.empty(q.shape, dtype=torch.float32, device=q.device)

    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    # next_dk, next_dv = None, None
    # next_k, next_v = None, None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v, next_k, next_v, k_hdl, v_hdl)

        if step <= kv_comm.rank or not causal:
            bwd_causal = causal and step == 0
            params = get_default_args(_flash_attn_backward).copy()
            params.update(
                {
                    "dout": dout,
                    "q": q,
                    "k": k,
                    "v": v,
                    "out": out,
                    "softmax_lse": softmax_lse,
                    "dq": block_dq_buffer,
                    "dk": block_dk_buffer,
                    "dv": block_dv_buffer,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": bwd_causal,
                    "alibi_slopes": alibi_slopes,
                    "deterministic": deterministic,
                }
            )
            if "window_size" in params:
                params.update({"window_size": window_size})
            else:
                params.update(
                    {
                        "window_size_left": window_size[0],
                        "window_size_right": window_size[1],
                    }
                )
            _flash_attn_backward(**params)

        if step != 0:
            d_kv_comm.wait(dk_hdl, dv_hdl)
            if step <= kv_comm.rank or not causal:
                dq += block_dq_buffer
                dk.copy_(block_dk_buffer + next_dk)
                dv.copy_(block_dv_buffer + next_dv)                
            else:
                dk, next_dk = next_dk, dk
                dv, next_dv = next_dv, dv
                dk_hdl, next_dk_hdl = next_dk_hdl, dk_hdl
                dv_hdl, next_dv_hdl = next_dv_hdl, dv_hdl
        elif step <= kv_comm.rank or not causal:
            dq = block_dq_buffer.to(torch.float32)
            dk.copy_(block_dk_buffer)
            dv.copy_(block_dv_buffer)

        if step + 1 != kv_comm.world_size:
            kv_comm.wait(k_hdl, v_hdl)
            k_hdl, next_k_hdl = next_k_hdl, k_hdl
            v_hdl, next_v_hdl = next_v_hdl, v_hdl
            k, next_k = next_k, k
            v, next_v = next_v, v

        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv, next_dk, next_dv, dk_hdl, dv_hdl)
        print("send step", step, "rank", d_kv_comm.rank)

    print("wait final", d_kv_comm.rank)
    d_kv_comm.wait(dk_hdl, dv_hdl)

    return dq.to(torch.bfloat16), next_dk.to(q.dtype), next_dv.to(q.dtype)


class RingFlashAttnSymmFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        k_hdl,
        v_hdl,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        assert k[2].is_contiguous()
        assert v[2].is_contiguous()

        out, softmax_lse = ring_flash_attn_symm_forward(
            group,
            q,
            k,
            v,
            k_hdl,
            v_hdl,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(
            q,
            k[0], k[1], k[2], 
            v[0], v[1], v[2],
            out, softmax_lse
        )
        ctx.k_hdl = k_hdl
        ctx.v_hdl = v_hdl
        ctx.buffer = [k[0], k[1], v[0], v[1]]
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = ring_flash_attn_symm_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


# def ring_flash_attn_symm_qkvpacked_func(
#     qkv,
#     dropout_p=0.0,
#     softmax_scale=None,
#     causal=False,
#     window_size=(-1, -1),
#     alibi_slopes=None,
#     deterministic=False,
#     return_attn_probs=False,
#     group=None,
# ):
#     return RingFlashAttnSymmFunc.apply(
#         qkv[:, :, 0],
#         qkv[:, :, 1],
#         qkv[:, :, 2],
#         dropout_p,
#         softmax_scale,
#         causal,
#         window_size,
#         alibi_slopes,
#         deterministic,
#         return_attn_probs,
#         group,
#     )


# def ring_flash_attn_symm_kvpacked_func(
#     q,
#     kv,
#     dropout_p=0.0,
#     softmax_scale=None,
#     causal=False,
#     window_size=(-1, -1),
#     alibi_slopes=None,
#     deterministic=False,
#     return_attn_probs=False,
#     group=None,
# ):
#     return RingFlashAttnSymmFunc.apply(
#         q,
#         kv[:, :, 0],
#         kv[:, :, 1],
#         dropout_p,
#         softmax_scale,
#         causal,
#         window_size,
#         alibi_slopes,
#         deterministic,
#         return_attn_probs,
#         group,
#     )


def ring_flash_attn_symm_func(
    q,
    k,
    v,
    k_hdl,
    v_hdl,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    if group is None:
        group = dist.group.WORLD
    return RingFlashAttnSymmFunc.apply(
        q,
        k,
        v,
        k_hdl,
        v_hdl,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
