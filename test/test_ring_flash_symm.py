import torch
import torch.distributed as dist
from flash_attn import flash_attn_qkvpacked_func
from ring_flash_attn import ring_flash_attn_symm_func
from utils import log, set_seed
import torch.distributed._symmetric_memory as symm_mem


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = 3816
    nheads = 5
    d = 128
    dropout_p = 0
    causal = False
    deterministic = False

    assert seqlen % world_size == 0
    assert d % 8 == 0

    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dist.broadcast(qkv, src=0)

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    tmp_qkv = qkv.chunk(world_size, dim=1)[rank].detach()

    local_q = tmp_qkv[:, :, 0].clone()

    local_k = symm_mem.empty(tmp_qkv[:, :, 1].shape, dtype=dtype, device=device)
    local_v = symm_mem.empty(tmp_qkv[:, :, 2].shape, dtype=dtype, device=device)
    local_k.copy_(tmp_qkv[:, :, 1])
    local_v.copy_(tmp_qkv[:, :, 2])

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

    for k, v in zip(local_k_list, local_v_list):
        k.requires_grad = True
        v.requires_grad = True
    
    local_q.requires_grad = True

    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    out, lse, _ = flash_attn_qkvpacked_func(
        qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    local_out = out.chunk(world_size, dim=1)[rank]
    local_lse = lse.chunk(world_size, dim=-1)[rank]

    fn = ring_flash_attn_symm_func

    ring_out, ring_lse, _ = fn(
        local_q,
        local_k_list,
        local_v_list,
        k_hdl_list,
        v_hdl_list,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
        # group=dist.group.WORLD,
    )

    log("out", out, rank0_only=True)
    log("lse", lse, rank0_only=True)
    log("out diff", local_out - ring_out)
    log("lse diff", local_lse - ring_lse)

    dist.barrier()
    
    # if rank == 0:
    #     print("#" * 30)
    #     print("# backward:")
    #     print("#" * 30)

    # out.backward(dout)
    # dqkv = qkv.grad
    # local_dqkv = dqkv.chunk(world_size, dim=1)[rank]

    # ring_out.backward(local_dout)
    # ring_dqkv = torch.stack((local_q.grad, local_k.grad, local_v.grad), dim=2)

    # log("local_dqkv", local_dqkv)
    # log("dq diff", local_dqkv[:, 0] - ring_dqkv[:, 0])
    # log("dk diff", local_dqkv[:, 1] - ring_dqkv[:, 1])
    # log("dv diff", local_dqkv[:, 2] - ring_dqkv[:, 2])
