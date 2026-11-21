# Copyright © 2026 Apple Inc.
# Trimmed to what the Inkling 2 port needs; upstream's version also holds
# quantization helpers that depend on the new kernels layout.

import mlx.core as mx


def splice(
    embeddings: mx.array,
    inputs: mx.array,
    *,
    rows: mx.array,
    token_id: int,
) -> mx.array:
    """The embeddings with the rows, in order, at the token_id positions of inputs."""
    # A shape check only, so no host sync.
    if rows.shape[0] == 0:
        return embeddings
    mask = inputs == token_id
    # The row of each masked token: the number of masked tokens before it.
    idx = mx.cumsum(mask.flatten()).reshape(mask.shape) - 1
    idx = mx.clip(idx, 0, rows.shape[0] - 1)
    selected = rows[idx].astype(embeddings.dtype)
    return mx.where(mask[..., None], selected, embeddings)
