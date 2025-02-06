from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import inspect
import torch.distributed._symmetric_memory as symm_mem
from functools import cache


__all__ = ["update_out_and_lse", "RingComm", "get_default_args"]


@cache
def _get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args


def get_default_args(func):
    if inspect.isfunction(func):
        return _get_default_args(func)
    else:
        # Use the origin _init_fn in CustomOpDef
        return _get_default_args(func._init_fn)


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty(
        (num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device
    )
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group
        )
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v

class RingCommSymm:
    def __init__(
        self, 
        process_group: dist.ProcessGroup, 
        k: List[torch.Tensor], 
        v: List[torch.Tensor], 
        k_hdl: List[symm_mem._SymmetricMemory],
        v_hdl: List[symm_mem._SymmetricMemory]
    ):
        self._process_group = process_group
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)

        self.k = k
        self.v = v
        self.k_hdl = k_hdl
        self.v_hdl = v_hdl
        self.stream = torch.cuda.current_stream(), symm_mem._get_backend_stream()
        self.stream[1].wait_stream(self.stream[0]) # initialize the backend stream
        self.compute = 0 if self.rank %2 == 0 else 1

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)


    def send_recv(
        self, target: torch.Tensor, handle: List[symm_mem._SymmetricMemory], stream: torch.Stream, round: int, comm: int
    ):
        with torch.cuda.stream(stream):
            if round == 0:
                hdl = handle[2] # at round 0, all the data are on buffer 2
            else:
                hdl = handle[comm]
                # dist.barrier()
                hdl.wait_signal(self.recv_rank) # wait for the other side to finish the transfer
                hdl.wait_signal(self.send_rank) # wait for the other side to finish the transfer
                
            src = hdl.get_buffer(self.recv_rank, target.shape, target.dtype)
            target.copy_(src)
            handle[comm].put_signal(self.send_rank) # signal the other side that the data is ready
            handle[comm].put_signal(self.recv_rank) # signal the other side that the data is used
            

    def sync(self):
        self.stream[0].wait_stream(self.stream[1]) # wait for the backend stream to finish


    def step(self):
        # with torch.cuda.stream(self.stream[0]):
        #     self.k_hdl[0].barrier()
        #     self.v_hdl[0].barrier()
        # with torch.cuda.stream(self.stream[1]):
        #     self.k_hdl[1].barrier()
        #     self.v_hdl[1].barrier()
        self.compute = self.compute^1

    def send_recv_kv(self, round: int):
        comm = self.compute^1
        compute = self.compute
        self.send_recv(self.k[comm], self.k_hdl, self.stream[comm], round, comm)
        self.send_recv(self.v[comm], self.v_hdl, self.stream[comm], round, comm)

class AllGatherComm:
    def __init__(self, group=None) -> None:
        self.group = group
        self.handles = []

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        handle = dist.all_gather_into_tensor(
            output_tensor, input_tensor, group=self.group, async_op=True
        )
        self.handles.append(handle)

    def wait(self):
        for handle in self.handles:
            handle.wait()
        self.handles = []

class RingCommSymmDirect:
    def __init__(
        self, 
        process_group: dist.ProcessGroup, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        k_hdl: symm_mem._SymmetricMemory,
        v_hdl: symm_mem._SymmetricMemory
    ):
        self._process_group = process_group
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)

        self.k = k
        self.v = v
        self.k_hdl = k_hdl
        self.v_hdl = v_hdl
        self.stream = torch.cuda.current_stream(), symm_mem._get_backend_stream()
        self.stream[1].wait_stream(self.stream[0]) # initialize the backend stream
        self.stream_id = 0


    def send_recv(
        self, target: torch.Tensor, handle: symm_mem._SymmetricMemory, stream: torch.Stream, round: int
    )-> torch.Tensor:
        with torch.cuda.stream(stream):
            src_rank = (self.rank - round) % self.world_size
            src = handle.get_buffer(src_rank, target.shape, target.dtype)
            ret = torch.empty_like(target)
            ret.copy_(src)
            return ret
            

    def sync(self):
        self.stream[0].wait_stream(self.stream[1]) # wait for the backend stream to finish

    def step(self):
        self.stream_id = self.stream_id^1

    def send_recv_kv(self, round: int) -> Tuple[torch.Tensor, torch.Tensor]:
        this_k = self.send_recv(self.k, self.k_hdl, self.stream[self.stream_id], round)
        this_v = self.send_recv(self.v, self.v_hdl, self.stream[self.stream_id], round)
        return this_k, this_v