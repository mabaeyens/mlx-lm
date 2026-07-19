# Copyright © 2023-2024 Apple Inc.

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


def _offload_enable(module, resident_slots: int, fetch_fn, quantized: bool, max_stack_size=None):
    """Switch a SwitchLinear/QuantizedSwitchLinear to a disk-backed,
    dict-keyed LRU expert cache (`expert_id -> weight data`) instead of
    keeping every expert's weights resident, fetching cold experts via
    `fetch_fn(expert_id)`.

    No-op (module keeps its default, fully-resident behavior) if
    `resident_slots >= num_experts` — this keeps the default/unset path
    provably unchanged, since callers only reach this method at all when
    offloading is explicitly requested.

    The cache is keyed by expert id rather than a fixed slot array: a fixed
    (num_experts_resident, ...) tensor with in-place row overwrites would let
    an eviction *later in the same forward call* silently corrupt a slot an
    *earlier* expert in that same call already resolved to (the actual
    gather happens only after every index in the call is resolved) — with a
    dict cache there are no shared slots to collide over, so this can't
    happen.

    `max_stack_size` (default: `resident_slots`) bounds how many experts'
    weights `_offload_chunked_gather` is allowed to materialize into one
    temporary stacked tensor at once. Without this bound, a call whose
    unique-expert count is large — e.g. a long prefill, where
    tokens_in_call * top_k routinely exceeds num_experts, so nearly every
    expert gets touched in one call regardless of how skewed steady-state
    routing is — builds one huge temporary stack covering nearly the whole
    expert table, which is what caused a real Metal
    kIOGPUCommandBufferCallbackErrorOutOfMemory crash under a ~1458-token
    prompt (specs/moe-expert-offload-02-runtime-cache.md, Phase C). Chunking
    keeps every call's peak transient memory bounded regardless of call
    shape, with no prefill/decode signal needed from the caller.
    """
    n_experts = module.num_experts
    if resident_slots >= n_experts:
        return
    # Seed the cache from experts already resident in the eager-loaded
    # tensors (no wasted first-call fetches), then drop the reference to the
    # full-size tensors so that unified memory is actually freed.
    seed_weight = module.weight[:resident_slots]
    seed_scales = module.scales[:resident_slots] if quantized else None
    seed_biases = (
        module.biases[:resident_slots] if quantized and module.get("biases") is not None else None
    )
    mx.eval(*(t for t in (seed_weight, seed_scales, seed_biases) if t is not None))

    module._offload_fetch = fetch_fn
    module._offload_quantized = quantized
    module._offload_capacity = resident_slots
    module._offload_max_stack_size = max(max_stack_size or resident_slots, 1)
    module._offload_cache = {}
    module._offload_lru = []
    for e in range(resident_slots):
        data = (seed_weight[e], seed_scales[e], seed_biases[e] if seed_biases is not None else None) if quantized else seed_weight[e]
        module._offload_cache[e] = data
        module._offload_lru.append(e)
    module._offload_hits = 0
    module._offload_misses = 0

    module.weight = seed_weight
    if quantized:
        module.scales = seed_scales
        if seed_biases is not None:
            module.biases = seed_biases


def _offload_touch(module, expert_id):
    if expert_id in module._offload_lru:
        module._offload_lru.remove(expert_id)
    module._offload_lru.append(expert_id)


def _offload_stack_rows(rows, quantized):
    if quantized:
        return (
            mx.stack([d[0] for d in rows]),
            mx.stack([d[1] for d in rows]),
            mx.stack([d[2] for d in rows]) if rows[0][2] is not None else None,
        )
    return mx.stack(rows)


def _offload_chunked_gather(module, x, indices, gather_fn):
    """Compute the same output as gathering against every needed expert at
    once, but never materialize more than `module._offload_max_stack_size`
    experts' weights in a single temporary stack.

    Partitions the call's unique experts into groups of at most
    max_stack_size, running one `gather_fn(x, stacked_data, local_indices)`
    call per group against the *entire* index tensor (positions outside the
    group get a dummy index and their output is discarded), then combines
    the groups' outputs with `mx.where`. This costs more matmul work than a
    single unchunked call when there's more than one group (every group's
    call touches every position, not just its own) — that trade is what
    buys a memory bound that holds for any call shape, prefill or decode,
    without needing to know which one it is.

    Fetches (and evicts) lazily, group by group, rather than resolving every
    unique expert up front: this is what keeps the *cache dict itself*
    bounded during a large call too, not just the final stacked tensor — the
    original design fetched every unique expert into the cache before doing
    any eviction, which meant the dict alone could balloon to near-full-table
    size on a diverse call, independent of the stacking cost. Evicting after
    each group's own gather is safe because each expert belongs to exactly
    one group: nothing evicted here is needed again later in this same call.
    An `mx.eval()` after each group's combine forces MLX to actually release
    the previous group's temporary stack before the next one is built —
    without it, laziness re-batches every group into one deferred graph
    anyway and the memory bound is fiction.
    """
    flat = indices.reshape(-1).tolist()
    unique = list(dict.fromkeys(flat))
    max_stack = module._offload_max_stack_size
    groups = [unique[i : i + max_stack] for i in range(0, len(unique), max_stack)]

    group_of = {}
    local_pos_of = {}
    for gi, group in enumerate(groups):
        for li, e in enumerate(group):
            group_of[e] = gi
            local_pos_of[e] = li

    cache = module._offload_cache
    result = None
    for gi, group in enumerate(groups):
        rows = []
        for e in group:
            data = cache.get(e)
            if data is not None:
                module._offload_hits += 1
            else:
                module._offload_misses += 1
                data = module._offload_fetch(e)
                cache[e] = data
            rows.append(data)
            _offload_touch(module, e)
        stacked = _offload_stack_rows(rows, module._offload_quantized)

        local_flat = [local_pos_of[e] if group_of[e] == gi else 0 for e in flat]
        local_indices = mx.array(local_flat, dtype=indices.dtype).reshape(indices.shape)
        group_output = gather_fn(x, stacked, local_indices)

        if len(groups) == 1:
            result = group_output
        else:
            mask = mx.array([group_of[e] == gi for e in flat], dtype=mx.bool_).reshape(indices.shape)
            # gather_mm/gather_qmm's output has more trailing dims than
            # `indices` itself (e.g. indices.shape + (1, output_dims) for
            # SwitchGLU's usual (N, top_k) indices) — pad to whatever rank
            # this call's output actually came out at, rather than assuming
            # a fixed offset.
            while mask.ndim < group_output.ndim:
                mask = mx.expand_dims(mask, -1)
            result = group_output if result is None else mx.where(mask, group_output, result)
        mx.eval(result)

        while len(module._offload_lru) > module._offload_capacity:
            victim = module._offload_lru.pop(0)
            cache.pop(victim, None)

    return result


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def enable_offload(self, resident_slots: int, fetch_fn, max_stack_size=None):
        _offload_enable(self, resident_slots, fetch_fn, quantized=True, max_stack_size=max_stack_size)

    def __call__(self, x, indices, sorted_indices=False):
        if hasattr(self, "_offload_fetch"):
            def gather_fn(x, data, local_indices):
                weight, scales, biases = data
                return mx.gather_qmm(
                    x,
                    weight,
                    scales,
                    biases,
                    rhs_indices=local_indices,
                    transpose=True,
                    group_size=self.group_size,
                    bits=self.bits,
                    mode=self.mode,
                    sorted_indices=False,
                )

            x = _offload_chunked_gather(self, x, indices, gather_fn)
        else:
            x = mx.gather_qmm(
                x,
                self["weight"],
                self["scales"],
                self.get("biases"),
                rhs_indices=indices,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
                sorted_indices=sorted_indices,
            )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def enable_offload(self, resident_slots: int, fetch_fn, max_stack_size=None):
        _offload_enable(self, resident_slots, fetch_fn, quantized=False, max_stack_size=max_stack_size)

    def __call__(self, x, indices, sorted_indices=False):
        if hasattr(self, "_offload_fetch"):
            def gather_fn(x, weight, local_indices):
                return mx.gather_mm(
                    x,
                    weight.swapaxes(-1, -2),
                    rhs_indices=local_indices,
                    sorted_indices=False,
                )

            x = _offload_chunked_gather(self, x, indices, gather_fn)
        else:
            x = mx.gather_mm(
                x,
                self["weight"].swapaxes(-1, -2),
                rhs_indices=indices,
                sorted_indices=sorted_indices,
            )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
