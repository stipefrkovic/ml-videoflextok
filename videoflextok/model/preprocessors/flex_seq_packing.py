# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
import math
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional

import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

__all__ = ["BlockWiseSequencePacker", "BlockWiseSequenceInterleavePacker"]


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda", _compile=False):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, _compile=_compile)
    return block_mask


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


@lru_cache
def generate_seq_ids(ps, max_seq_len=None, device="cuda", padding_id=-1):
    seq_ids = torch.cat(
        [torch.full((size.numel(),), fill_value=i, device=device) for i, size in enumerate(ps)]
    )
    if max_seq_len is None:
        return seq_ids
    seq_len = len(seq_ids)
    assert max_seq_len >= seq_len
    return F.pad(seq_ids, (0, max_seq_len - seq_len), mode="constant", value=padding_id)


@lru_cache
def generate_packed_seq_mask(ps, max_seq_len=None, device="cuda", padding_id=-1):
    seq_ids = generate_seq_ids(ps, max_seq_len=max_seq_len, device=device, padding_id=padding_id)

    def packed_seq_masking(b, h, q_idx, kv_idx):
        not_padded_mask = (seq_ids[q_idx] != padding_id) | (seq_ids[kv_idx] != padding_id)
        same_seq_mask = seq_ids[q_idx] == seq_ids[kv_idx]
        return same_seq_mask & not_padded_mask

    return packed_seq_masking, seq_ids


@lru_cache
def generate_causal_packed_seq_mask(ps, max_seq_len=None, device="cuda", padding_id=-1):
    seq_ids = generate_seq_ids(ps, max_seq_len=max_seq_len, device=device, padding_id=padding_id)

    def causal_packed_seq_masking(b, h, q_idx, kv_idx):
        not_padded_mask = (seq_ids[q_idx] != padding_id) | (seq_ids[kv_idx] != padding_id)
        same_seq_mask = seq_ids[q_idx] == seq_ids[kv_idx]
        causal = kv_idx <= q_idx
        return same_seq_mask & not_padded_mask & causal

    return causal_packed_seq_masking, seq_ids


@lru_cache
def generate_block_ids(ps, block_sizes, max_seq_len=None, device="cuda", padding_id=-1):
    block_ids = torch.cat(
        [
            # TODO(andrewatanov): make it unique for different sequence IDs?
            torch.div(torch.arange(size.numel(), device=device), block_size, rounding_mode="floor")
            for size, block_size in zip(ps, block_sizes)
        ]
    )
    if max_seq_len is None:
        return block_ids
    seq_len = len(block_ids)
    assert max_seq_len >= seq_len
    return F.pad(block_ids, (0, max_seq_len - seq_len), mode="constant", value=padding_id)


@lru_cache
def generate_block_causal_packed_seq_mask(
    ps, block_sizes, max_seq_len=None, device="cuda", padding_id=-1
):
    """
    Creates a block-causal mask for packed sequences where attention is allowed within blocks of size `block_size`.
    A block with index i can attend to itself and all previous blocks.

    Args:
        ps: Tuple of packed shapes.
        block_sizes: Size of each causal block.
        max_seq_len: Maximum sequence length for padding.
        device: Device to create tensors on.
        padding_id: value used for padding tokens.

    Returns:
        A masking function and sequence IDs.
    """
    # assert all([size.numel() % block_size == 0 for size in ps]), "all sequences should be devisible by the causal block size"
    assert len(ps) == len(block_sizes)

    seq_ids = generate_seq_ids(ps, max_seq_len=max_seq_len, device=device, padding_id=padding_id)

    # Calculate block indices for each position
    block_ids = generate_block_ids(
        ps, block_sizes, max_seq_len=max_seq_len, device=device, padding_id=padding_id
    )

    def block_causal_packed_seq_masking(b, h, q_idx, kv_idx):
        not_padded_mask = (seq_ids[q_idx] != padding_id) | (seq_ids[kv_idx] != padding_id)
        same_seq_mask = seq_ids[q_idx] == seq_ids[kv_idx]

        # Block-causal: a query position can attend to all positions in its block or earlier blocks
        # within the same sequence as insured by same_seq_mask
        q_block = block_ids[q_idx]
        kv_block = block_ids[kv_idx]
        block_causal = kv_block <= q_block

        return same_seq_mask & not_padded_mask & block_causal

    return block_causal_packed_seq_masking, seq_ids


@lru_cache
def generate_time_block_causal_prefix_packed_seq_mask(
    ps, block_sizes, prefix_lens, max_seq_len=None, device="cuda", padding_id=-1
):
    """
    Creates a time-block-causal mask for interleaved packed sequences where:
    - Attention is causal across time blocks (later blocks can't attend to earlier blocks)
    - Within each time block: prefix attention (causal OR prefix mask, same as causal_last mode)

    This combines time-block causality with prefix attention within each block.

    Args:
        ps: Tuple of packed shapes (outer sequence shapes).
        block_sizes: Size of each time block.
        prefix_lens: Tuple of prefix lengths for each sequence (number of prefix tokens per time block).
        max_seq_len: Maximum sequence length for padding.
        device: Device to create tensors on.
        padding_id: Value used for padding tokens.

    Returns:
        A masking function and sequence IDs.
    """
    assert len(ps) == len(block_sizes) == len(prefix_lens)

    seq_ids = generate_seq_ids(ps, max_seq_len=max_seq_len, device=device, padding_id=padding_id)

    # Calculate block indices for each position
    block_ids = generate_block_ids(
        ps, block_sizes, max_seq_len=max_seq_len, device=device, padding_id=padding_id
    )

    # Get unique sequence IDs and their counts to compute offsets (pre-calculate to avoid dynamic shapes)
    _, counts = torch.unique_consecutive(seq_ids, return_counts=True)
    offsets = torch.cat([torch.tensor([0], device=seq_ids.device), counts.cumsum(0)[:-1]])

    # Convert prefix_lens tuple to tensor for efficient indexing. Needs to be predictably hashable.
    prefix_lens = torch.tensor(prefix_lens, device=device)

    def time_block_causal_last_packed_seq_masking(b, h, q_idx, kv_idx):
        not_padded_mask = (seq_ids[q_idx] != padding_id) | (seq_ids[kv_idx] != padding_id)
        same_seq_mask = seq_ids[q_idx] == seq_ids[kv_idx]

        # Block-causal across time: later blocks can't attend to earlier blocks
        q_block = block_ids[q_idx]
        kv_block = block_ids[kv_idx]
        time_block_causal = kv_block <= q_block

        # Within the same time block, apply prefix attention (same as prefix_packed_seq_mask)
        same_block_mask = q_block == kv_block

        # Calculate positions within each sequence using pre-calculated offsets
        q_seq_id = seq_ids[q_idx]
        kv_seq_id = seq_ids[kv_idx]
        q_logical = q_idx - offsets[q_seq_id]
        kv_logical = kv_idx - offsets[kv_seq_id]

        # Within each time block, calculate positions relative to block start
        block_size = block_sizes[0] if len(set(block_sizes)) == 1 else block_sizes[q_seq_id]
        # this works cos of combination with same_block_mask and time_block_causal below
        q_within_block = q_logical % block_size
        kv_within_block = kv_logical % block_size

        # Apply prefix attention within each time block (causal OR prefix mask)
        inner_causal_mask = causal(b, h, q_within_block, kv_within_block)
        inner_prefix_mask = kv_within_block < prefix_lens[q_seq_id]
        within_block_attention = inner_causal_mask | inner_prefix_mask

        # Combine time-block causality with within-block prefix attention
        attention_allowed = time_block_causal & (
            (~same_block_mask) | (same_block_mask & within_block_attention)
        )

        return same_seq_mask & not_padded_mask & attention_allowed

    return time_block_causal_last_packed_seq_masking, seq_ids


@lru_cache
def generate_flexible_time_block_causal_prefix_packed_seq_mask(
    ps,
    block_sizes,
    prefix_lens,
    prefix2prefix="full",
    prefix2nonprefix="full",
    nonprefix2prefix="full",
    nonprefix2nonprefix="full",
    max_seq_len=None,
    device="cuda",
    padding_id=-1,
):
    """
    Creates a flexible time-block-causal mask for interleaved packed sequences where:
    - Attention is causal across time blocks (later blocks can't attend to earlier blocks)
    - Within each time block: prefix attention (causal OR prefix mask)
    - Cross-timestep attention is configurable via the attention pattern parameters

    Args:
        ps: Tuple of packed shapes (outer sequence shapes).
        block_sizes: Size of each time block.
        prefix_lens: Tuple of prefix lengths for each sequence (number of prefix tokens per time block).
        prefix2prefix: How current prefix tokens attend to past prefix tokens ("full" or "none").
        prefix2nonprefix: How current prefix tokens attend to past non-prefix tokens ("full" or "none").
        nonprefix2prefix: How current non-prefix tokens attend to past prefix tokens ("full" or "none").
        nonprefix2nonprefix: How current non-prefix tokens attend to past non-prefix tokens
                            ("full", "none", or "causal").
        max_seq_len: Maximum sequence length for padding.
        device: Device to create tensors on.
        padding_id: Value used for padding tokens.

    Returns:
        A masking function and sequence IDs.
    """
    assert len(ps) == len(block_sizes) == len(prefix_lens)

    # Validate attention pattern parameters
    valid_binary_modes = {"full", "none"}
    valid_nonprefix2nonprefix_modes = {"full", "none", "causal"}

    assert prefix2prefix in valid_binary_modes, f"prefix2prefix must be in {valid_binary_modes}"
    assert (
        prefix2nonprefix in valid_binary_modes
    ), f"prefix2nonprefix must be in {valid_binary_modes}"
    assert (
        nonprefix2prefix in valid_binary_modes
    ), f"nonprefix2prefix must be in {valid_binary_modes}"
    assert (
        nonprefix2nonprefix in valid_nonprefix2nonprefix_modes
    ), f"nonprefix2nonprefix must be in {valid_nonprefix2nonprefix_modes}"

    seq_ids = generate_seq_ids(ps, max_seq_len=max_seq_len, device=device, padding_id=padding_id)

    # Calculate block indices for each position
    block_ids = generate_block_ids(
        ps, block_sizes, max_seq_len=max_seq_len, device=device, padding_id=padding_id
    )

    # Get unique sequence IDs and their counts to compute offsets (pre-calculate to avoid dynamic shapes)
    _, counts = torch.unique_consecutive(seq_ids, return_counts=True)
    offsets = torch.cat([torch.tensor([0], device=seq_ids.device), counts.cumsum(0)[:-1]])

    # Convert prefix_lens tuple to tensor for efficient indexing. Needs to be predictably hashable.
    prefix_lens = torch.tensor(prefix_lens, device=device)

    # Pre-compute boolean flags to avoid control flow in the mask function
    prefix2prefix_enabled = prefix2prefix == "full"
    prefix2nonprefix_enabled = prefix2nonprefix == "full"
    nonprefix2prefix_enabled = nonprefix2prefix == "full"
    nonprefix2nonprefix_full = nonprefix2nonprefix == "full"
    nonprefix2nonprefix_causal = nonprefix2nonprefix == "causal"

    def flexible_time_block_causal_prefix_packed_seq_masking(b, h, q_idx, kv_idx):
        not_padded_mask = (seq_ids[q_idx] != padding_id) | (seq_ids[kv_idx] != padding_id)
        same_seq_mask = seq_ids[q_idx] == seq_ids[kv_idx]

        # Block-causal across time: later blocks can't attend to earlier blocks
        q_block = block_ids[q_idx]
        kv_block = block_ids[kv_idx]
        time_block_causal = kv_block <= q_block

        # Within the same time block, apply prefix attention (same as prefix_packed_seq_mask)
        same_block_mask = q_block == kv_block

        # Calculate positions within each sequence using pre-calculated offsets
        q_seq_id = seq_ids[q_idx]
        kv_seq_id = seq_ids[kv_idx]
        q_logical = q_idx - offsets[q_seq_id]
        kv_logical = kv_idx - offsets[kv_seq_id]

        # Within each time block, calculate positions relative to block start
        block_size = block_sizes[0] if len(set(block_sizes)) == 1 else block_sizes[q_seq_id]
        q_within_block = q_logical % block_size
        kv_within_block = kv_logical % block_size

        # Determine if tokens are prefix or non-prefix
        q_is_prefix = q_within_block < prefix_lens[q_seq_id]
        kv_is_prefix = kv_within_block < prefix_lens[kv_seq_id]

        # Within-block attention (same as before)
        inner_causal_mask = causal(b, h, q_within_block, kv_within_block)
        inner_prefix_mask = kv_within_block < prefix_lens[q_seq_id]
        within_block_attention = inner_causal_mask | inner_prefix_mask

        # Cross-timestep attention logic (when kv_block < q_block)
        cross_timestep_mask = kv_block < q_block

        # Apply configurable cross-timestep attention patterns using tensor operations only
        # Pattern 1: prefix -> prefix
        p2p_attention = (q_is_prefix & kv_is_prefix) & prefix2prefix_enabled

        # Pattern 2: prefix -> nonprefix
        p2n_attention = (q_is_prefix & (~kv_is_prefix)) & prefix2nonprefix_enabled

        # Pattern 3: nonprefix -> prefix
        n2p_attention = ((~q_is_prefix) & kv_is_prefix) & nonprefix2prefix_enabled

        # Pattern 4: nonprefix -> nonprefix
        n2n_mask = (~q_is_prefix) & (~kv_is_prefix)
        n2n_full_attention = n2n_mask & nonprefix2nonprefix_full

        # Pattern 4b: nonprefix -> nonprefix (causal)
        q_nonprefix_pos = q_within_block - prefix_lens[q_seq_id]
        kv_nonprefix_pos = kv_within_block - prefix_lens[kv_seq_id]
        n2n_causal_attention = (
            n2n_mask & nonprefix2nonprefix_causal & (kv_nonprefix_pos <= q_nonprefix_pos)
        )

        # Combine all cross-timestep patterns
        cross_timestep_attention = (
            p2p_attention
            | p2n_attention
            | n2p_attention
            | n2n_full_attention
            | n2n_causal_attention
        )

        # Combine all attention patterns
        attention_allowed = time_block_causal & (
            (same_block_mask & within_block_attention)
            | (cross_timestep_mask & cross_timestep_attention)
        )

        return same_seq_mask & not_padded_mask & attention_allowed

    return flexible_time_block_causal_prefix_packed_seq_masking, seq_ids


@lru_cache
def generate_prefix_packed_seq_mask(
    ps, prefix_lens, max_seq_len=None, device="cuda", padding_id=-1
):
    # Create tensor of sequence IDs
    seq_ids = generate_seq_ids(ps, max_seq_len=max_seq_len, device=device, padding_id=padding_id)
    # Get unique sequence IDs and their counts
    _, counts = torch.unique_consecutive(seq_ids, return_counts=True)
    # Create cumulative counts (offsets)
    offsets = torch.cat([torch.tensor([0], device=seq_ids.device), counts.cumsum(0)[:-1]])
    # Convert prefix_lens tuple to tensor. Needs to be predictably hashable.
    prefix_lens = torch.tensor(prefix_lens, device=device)

    def prefix_packed_seq_mask(b, h, q_idx, kv_idx):
        not_padded_mask = (seq_ids[q_idx] != padding_id) | (seq_ids[kv_idx] != padding_id)
        same_seq_mask = seq_ids[q_idx] == seq_ids[kv_idx]

        q_logical = q_idx - offsets[seq_ids[q_idx]]
        kv_logical = kv_idx - offsets[seq_ids[kv_idx]]
        inner_causal_mask = causal(b, h, q_logical, kv_logical)
        inner_prefix_mask = kv_logical < prefix_lens[seq_ids[kv_idx]]

        return same_seq_mask & (inner_causal_mask | inner_prefix_mask) & not_padded_mask

    return prefix_packed_seq_mask, seq_ids


@lru_cache
def generate_packed_xattn_mask(
    ps_seq, ps_ctx, max_seq_len=None, max_ctx_len=None, device="cuda", padding_id=-1
):
    seq_ids = generate_seq_ids(
        ps_seq, max_seq_len=max_seq_len, device=device, padding_id=padding_id
    )
    ctx_ids = generate_seq_ids(
        ps_ctx, max_seq_len=max_ctx_len, device=device, padding_id=padding_id
    )

    def packed_xattn_masking(b, h, q_idx, kv_idx):
        not_padded_mask = (seq_ids[q_idx] != padding_id) | (ctx_ids[kv_idx] != padding_id)
        same_seq_mask = seq_ids[q_idx] == ctx_ids[kv_idx]
        return same_seq_mask & not_padded_mask

    return packed_xattn_masking, seq_ids, ctx_ids


@lru_cache
def generate_batched_full_mask(seq_len, device="cuda"):
    """
    Creates a full attention mask for batched sequences where each batch is independent.

    Args:
        seq_len: Length of each sequence in the batch.
        device: Device to create tensors on.

    Returns:
        A masking function.
    """

    def batched_full_mask(b, h, q_idx, kv_idx):
        # Each position can attend to all positions within the sequence
        return (q_idx < seq_len) & (kv_idx < seq_len)

    return batched_full_mask


@lru_cache
def generate_batched_time_block_causal_mask(seq_len, spatial_size, device="cuda"):
    """
    Creates a time-block-causal mask for batched sequences where attention is causal across time blocks.

    Args:
        seq_len: Length of each sequence in the batch.
        spatial_size: Size of each spatial block (H*W).
        device: Device to create tensors on.

    Returns:
        A masking function.
    """

    def batched_time_block_causal_mask(b, h, q_idx, kv_idx):
        # Don't attend to padding
        valid_mask = (q_idx < seq_len) & (kv_idx < seq_len)
        # Time block indices (which timestep)
        q_time = q_idx // spatial_size
        kv_time = kv_idx // spatial_size
        # Time causal: can only attend to same or earlier timesteps
        time_causal = kv_time <= q_time
        return valid_mask & time_causal

    return batched_time_block_causal_mask


def strict_zip(*iterables):
    lengths = [len(iterable) for iterable in iterables]
    if len(set(lengths)) != 1:
        raise ValueError("All input iterables must be of the same length")
    return zip(*iterables)


def next_highest_multiple(N, multiple):
    return multiple * math.ceil(N / multiple)


def expand_emb(emb, seq_lens):
    return torch.cat(
        [einops.repeat(emb_i, "d -> 1 n d", n=n) for emb_i, n in zip(emb, seq_lens)], dim=1
    )


def expand_emb_per_subseq(emb_packed, packed_shapes_list):
    adaLN_expansion = len(packed_shapes_list[0])
    emb_packed = einops.rearrange(emb_packed, "b (n d) -> b n d", n=adaLN_expansion)

    # Compute the repeats for each embedding
    repeats = torch.tensor(
        [[shape.numel() for shape in ps_list_i] for ps_list_i in packed_shapes_list],
        dtype=torch.long,
        device=emb_packed.device,
    )

    # Flatten embeddings and repeats
    emb_packed_flat = emb_packed.reshape(-1, emb_packed.shape[-1])  # Shape: [b * n, d]
    repeats_flat = repeats.flatten()  # Shape: [b * n]

    # Repeat embeddings according to repeats
    emb_expanded_flat = emb_packed_flat.repeat_interleave(repeats_flat, dim=0)

    # Return the expanded embeddings
    return emb_expanded_flat.unsqueeze(0)  # Shape: [1, total_n, d]


def expand_emb_per_interleaved_subseq(emb_packed, packed_shapes_list, interleave_size):
    adaLN_expansion = len(packed_shapes_list[0])
    emb_packed = einops.rearrange(emb_packed, "b (n d) -> b n d", n=adaLN_expansion)

    # Compute the repeats for each embedding
    repeats = torch.tensor(
        [[shape.numel() for shape in ps_list_i] for ps_list_i in packed_shapes_list],
        dtype=torch.long,
        device=emb_packed.device,
    )

    # Flatten embeddings and repeats
    emb_packed_flat = emb_packed.reshape(-1, emb_packed.shape[-1])  # Shape: [b * n, d]
    repeats_flat = repeats.flatten()  # Shape: [b * n]

    # Repeat embeddings according to repeats
    emb_expanded_flat = emb_packed_flat.repeat_interleave(repeats_flat, dim=0)  # Shape: [* , d]

    # Repeat embeddings according to the time dimension in the packed sequence
    # NOTE: This copies the tensor and can cause OOM when not compiled
    # TODO: find a way around the OOM issue when not compiled
    emb_expanded_flat = emb_expanded_flat.repeat(interleave_size, 1)  # Shape: [total_n, d]

    # Return the expanded embeddings
    return emb_expanded_flat.unsqueeze(0)  # Shape: [1, total_n, d]


class BlockWiseSequencePacker(nn.Module):
    """
    Module for packing sequences from multiple input lists of tensors, creating a block-wise
    self-attention, causal, or prefix LM masks for FlexAttention. Sequences will be concatenated
    in order of the input_list_read_keys.

    Args:
        input_list_read_keys: List of keys to read input lists of tensors from the input dictionary.
        packed_seq_write_key: Key to write the packed sequence into the output dictionary.
        block_mask_write_key: Key to write the block-wise attention mask into the output dictionary.
        inner_packed_shapes_write_key: Key to write the packed shapes of the inner sequences.
        outer_packed_shapes_write_key: Key to write the packed shape of the outer sequence.
        mask_mode: Block-wise attention mask mode, 'full' (full self-attention over all inner sequences),
            'causal' (causal across all), or 'causal_last' (full self-attention for all but the
            last inner sequences which is causal, e.g. prefix LM). Attention will always be
            block-wise, i.e. outer sequences cannot attend to each other.
        max_seq_len: Optionally pads packed token sequence to the given length. Useful for
            FlexAttention's requirement that sequence lengths must be multiples of 128.
        pad_to_multiple: While max_seq_len specifies a fixed seq_len, pad_to_multiple can be used
            to pad the sequence to the next multiple of the given value, e.g. 128. Cannot be set
            simultaneously with max_seq_len.
    """

    # TODO(roman-bachmann): Consider using nested tensors when they allow more than one nesting level,
    # and when the API is final. See https://pytorch.org/docs/stable/nested.html.

    def __init__(
        self,
        input_list_read_keys: List[str],
        packed_seq_write_key: str,
        block_mask_write_key: str,
        inner_packed_shapes_write_key: str,
        outer_packed_shapes_write_key: str,
        mask_mode: str = "full",
        max_seq_len: Optional[int] = None,
        pad_to_multiple: Optional[int] = None,
        emb_packing_fn_write_key: Optional[str] = None,
        per_subseq_embs: bool = False,
        compile_block_mask: bool = True,
    ):
        super().__init__()
        self.input_list_read_keys = input_list_read_keys
        self.packed_seq_write_key = packed_seq_write_key
        self.block_mask_write_key = block_mask_write_key
        self.inner_packed_shapes_write_key = inner_packed_shapes_write_key
        self.outer_packed_shapes_write_key = outer_packed_shapes_write_key
        self.emb_packing_fn_write_key = emb_packing_fn_write_key

        self.mask_mode = mask_mode
        self.max_seq_len = max_seq_len
        self.pad_to_multiple = pad_to_multiple
        if max_seq_len is not None and pad_to_multiple is not None:
            raise ValueError("Only one of max_seq_len or pad_to_multiple should be provided.")
        self.per_subseq_embs = per_subseq_embs

        self.compile_block_mask = compile_block_mask
        self.create_block_mask = torch.compiler.disable(create_block_mask_cached)

        # For pretty printing
        self._init_args = locals().copy()
        self._init_args.pop("self")
        self._init_args.pop("__class__")

    def __repr__(self):
        cls_name = self.__class__.__name__
        args_str = ",\n  ".join(f"{k}={v!r}" for k, v in self._init_args.items())
        return f"{cls_name}(\n  {args_str}\n)"

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # From the data_dict, get the lists containing the aligned tensors,
        # e.g., a list of image patches and a list of registers.
        list_of_tensor_lists = [data_dict[key] for key in self.input_list_read_keys]

        # Concatenate each sample across the lists, e.g., images[0] | registers[0], images[1] | registers[1], ...
        # Keep track of the shape of these inner tensors
        tensors_concat_list = []
        packed_shapes_list = []
        for tensors in strict_zip(*list_of_tensor_lists):
            # tensors contains the i-th entries of each of the lists in list_of_tensor_lists
            sample_packed, ps = einops.pack(tensors, "b * d")
            tensors_concat_list.append(sample_packed)
            packed_shapes_list.append(ps)

        # Pack tensors into one large sequence
        tensors_packed, ps = einops.pack(tensors_concat_list, "b * d")
        B, N_orig, D = tensors_packed.shape

        # TODO(roman-bachmann): Only supporting B=1 until https://github.com/pytorch/pytorch/issues/134560 is resolved
        assert B == 1

        device = tensors_packed.device

        # Create full or causal block-wise self-attention mask using FlexAttention. Optionally pad sequences.
        if self.pad_to_multiple is not None:
            max_seq_len = next_highest_multiple(N_orig, self.pad_to_multiple)
        else:
            max_seq_len = self.max_seq_len
        if self.mask_mode == "full":
            mask_fn, seq_ids = generate_packed_seq_mask(
                tuple(ps),
                max_seq_len=max_seq_len,
                device=device,
            )
        elif self.mask_mode == "causal":
            mask_fn, seq_ids = generate_causal_packed_seq_mask(
                tuple(ps),
                max_seq_len=max_seq_len,
                device=device,
            )
        elif self.mask_mode == "causal_last":
            prefix_lens = [
                sum([shape.numel() for shape in ps_i[:-1]]) for ps_i in packed_shapes_list
            ]
            mask_fn, seq_ids = generate_prefix_packed_seq_mask(
                tuple(ps),
                tuple(prefix_lens),
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            raise ValueError(f"Invalid mask mode {self.mask_mode}")

        N = len(seq_ids)
        assert (
            N % 128 == 0
        ), f"flex_attention sequence length must be a multiple of 128, but current is {N}."
        block_mask = self.create_block_mask(
            mask_fn, None, None, N, N, device=device, _compile=self.compile_block_mask
        )

        # Optionally zero-pad packed sequence
        # Outer packed shapes can be used to remove padding
        num_padding_tokens = N - N_orig
        if num_padding_tokens > 0:
            tensors_packed = F.pad(tensors_packed, (0, 0, 0, num_padding_tokens))

        # Optionally add an embedding packing function that specifies how elements
        # of a tensor of shape (B, L, ...) should be expanded to sequences of shape
        # (B, l1, ...), (B, l2, ...), ..., (B, lL, ...). Does not take into account
        # any padding.
        if self.emb_packing_fn_write_key is not None:
            seq_lens = [shape.numel() for shape in ps]
            if self.per_subseq_embs:
                emb_packing_fn = partial(
                    expand_emb_per_subseq, packed_shapes_list=packed_shapes_list
                )
            else:
                emb_packing_fn = partial(expand_emb, seq_lens=seq_lens)
            data_dict[self.emb_packing_fn_write_key] = emb_packing_fn

        data_dict[self.packed_seq_write_key] = tensors_packed
        data_dict[self.block_mask_write_key] = block_mask
        data_dict[self.inner_packed_shapes_write_key] = packed_shapes_list
        data_dict[self.outer_packed_shapes_write_key] = ps

        return data_dict


class BlockWiseSequenceInterleavePacker(nn.Module):
    """
    Module for packing sequences from multiple input lists of tensors, creating a block-wise
    self-attention, causal, or prefix LM masks for FlexAttention. Sequences will be concatenated
    in order of the input_list_read_keys.

    Args:
        input_list_read_keys: List of keys to read input lists of tensors from the input dictionary.
        packed_seq_write_key: Key to write the packed sequence into the output dictionary.
        block_mask_write_key: Key to write the block-wise attention mask into the output dictionary.
        inner_packed_shapes_write_key: Key to write the packed shapes of the inner sequences.
        outer_packed_shapes_write_key: Key to write the packed shape of the outer sequence.
        mask_mode: Block-wise attention mask mode, 'full' (full self-attention over all inner sequences),
            'causal' (causal across all), 'time_block_causal' (causal across time blocks, full within),
            'time_block_causal_last' (causal across time blocks, within each block: full attention
            for RGB tokens, causal attention for register tokens), or 'flexible_time_block_causal_last'
            (like time_block_causal_last but with configurable cross-timestep attention patterns).
            Attention will always be block-wise, i.e. outer sequences cannot attend to each other.
        max_seq_len: Optionally pads packed token sequence to the given length. Useful for
            FlexAttention's requirement that sequence lengths must be multiples of 128.
        pad_to_multiple: While max_seq_len specifies a fixed seq_len, pad_to_multiple can be used
            to pad the sequence to the next multiple of the given value, e.g. 128. Cannot be set
            simultaneously with max_seq_len.
        cross_timestep_attention_config: Dict with configuration for flexible cross-timestep attention.
            Only used when mask_mode is 'flexible_time_block_causal_last'. Should contain keys:
            - prefix2prefix: "full" or "none"
            - prefix2nonprefix: "full" or "none"
            - nonprefix2prefix: "full" or "none"
            - nonprefix2nonprefix: "full", "none", or "causal"
        repa_block_mask_write_key: Optional key to write a separate block mask for repa objective.
            When provided, creates a mask only for patch tokens (excluding registers).
        repa_mask_mode: Mask mode for repa block mask, 'full' or 'time_block_causal' (default).
        repa_patch_index: Index of the tensor list to use as patches for repa mask (default: 0).
    """

    # TODO(roman-bachmann): Consider using nested tensors when they allow more than one nesting level,
    # and when the API is final. See https://pytorch.org/docs/stable/nested.html.

    def __init__(
        self,
        input_list_read_keys: List[str],
        packed_seq_write_key: str,
        block_mask_write_key: str,
        inner_packed_shapes_write_key: str,
        outer_packed_shapes_write_key: str,
        mask_mode: str = "full",
        max_seq_len: Optional[int] = None,
        pad_to_multiple: Optional[int] = None,
        emb_packing_fn_write_key: Optional[str] = None,
        per_subseq_embs: bool = False,
        compile_block_mask: bool = True,
        cross_timestep_attention_config: Optional[Dict[str, str]] = None,
        repa_block_mask_write_key: Optional[str] = None,
        repa_mask_mode: str = "time_block_causal",
        repa_patch_index: int = 0,
    ):
        super().__init__()
        self.input_list_read_keys = input_list_read_keys
        self.packed_seq_write_key = packed_seq_write_key
        self.block_mask_write_key = block_mask_write_key
        self.inner_packed_shapes_write_key = inner_packed_shapes_write_key
        self.outer_packed_shapes_write_key = outer_packed_shapes_write_key
        self.emb_packing_fn_write_key = emb_packing_fn_write_key
        self.repa_block_mask_write_key = repa_block_mask_write_key
        self.repa_mask_mode = repa_mask_mode
        self.repa_patch_index = repa_patch_index

        self.mask_mode = mask_mode
        self.max_seq_len = max_seq_len
        self.pad_to_multiple = pad_to_multiple
        if max_seq_len is not None and pad_to_multiple is not None:
            raise ValueError("Only one of max_seq_len or pad_to_multiple should be provided.")
        self.per_subseq_embs = per_subseq_embs

        self.compile_block_mask = compile_block_mask
        self.create_block_mask = torch.compiler.disable(create_block_mask_cached)

        # Cross-timestep attention configuration for flexible mode
        if mask_mode == "flexible_time_block_causal_last":
            if cross_timestep_attention_config is None:
                raise ValueError(
                    "cross_timestep_attention_config must be provided when using flexible_time_block_causal_last mode"
                )
            self.cross_timestep_attention_config = cross_timestep_attention_config
        else:
            self.cross_timestep_attention_config = None

        # For pretty printing
        self._init_args = locals().copy()
        self._init_args.pop("self")
        self._init_args.pop("__class__")

    def __repr__(self):
        cls_name = self.__class__.__name__
        args_str = ",\n  ".join(f"{k}={v!r}" for k, v in self._init_args.items())
        return f"{cls_name}(\n  {args_str}\n)"

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # From the data_dict, get the lists containing the aligned tensors,
        # e.g., a list of image patches and a list of registers.
        list_of_tensor_lists = [data_dict[key] for key in self.input_list_read_keys]
        T = list_of_tensor_lists[0][0].shape[1]

        # Concatenate each sample across the lists, e.g., images[0] | registers[0], images[1] | registers[1], ...
        # Keep track of the shape of these inner tensors
        tensors_concat_list = []
        packed_shapes_list = []
        # temp_block_size: int = None
        for tensors in strict_zip(*list_of_tensor_lists):
            # tensors contains the i-th entries of each of the lists in list_of_tensor_lists
            sample_packed, ps = einops.pack(tensors, "b t * d")
            assert (
                sample_packed.shape[1] == T
            ), "only inputs with the same interleaved time dimension are supported"

            # # get the size of each temporal block for block causal masking
            # if temp_block_size is None:
            #     temp_block_size = sample_packed.shape[1]
            # assert temp_block_size == sample_packed.shape[1], "same temporal block size per sequence is supported for now"

            # interleave across the time dimension
            # TODO(andrewatanov): this can be done using chunks of different time sizes
            sample_interleaved = einops.rearrange(sample_packed, "b t l d -> b (t l) d")
            tensors_concat_list.append(sample_interleaved)
            packed_shapes_list.append(ps)

        # Pack tensors into one large sequence
        tensors_packed, ps = einops.pack(tensors_concat_list, "b * d")
        B, N_orig, D = tensors_packed.shape

        # TODO(roman-bachmann): Only supporting B=1 until https://github.com/pytorch/pytorch/issues/134560 is resolved
        assert B == 1

        device = tensors_packed.device

        # Create full or causal block-wise self-attention mask using FlexAttention. Optionally pad sequences.
        if self.pad_to_multiple is not None:
            max_seq_len = next_highest_multiple(N_orig, self.pad_to_multiple)
        else:
            max_seq_len = self.max_seq_len
        if self.mask_mode == "full":
            mask_fn, seq_ids = generate_packed_seq_mask(
                tuple(ps),
                max_seq_len=max_seq_len,
                device=device,
            )
        elif self.mask_mode == "causal":
            mask_fn, seq_ids = generate_causal_packed_seq_mask(
                tuple(ps),
                max_seq_len=max_seq_len,
                device=device,
            )
        elif self.mask_mode == "time_block_causal":
            temp_block_sizes = [sum([psi.numel() for psi in ps]) for ps in packed_shapes_list]
            mask_fn, seq_ids = generate_block_causal_packed_seq_mask(
                tuple(ps),
                block_sizes=tuple(temp_block_sizes),
                max_seq_len=max_seq_len,
                device=device,
            )
        elif self.mask_mode == "time_block_causal_last":
            temp_block_sizes = [sum([psi.numel() for psi in ps]) for ps in packed_shapes_list]
            prefix_lens = [sum([shape.numel() for shape in psi[:-1]]) for psi in packed_shapes_list]
            mask_fn, seq_ids = generate_time_block_causal_prefix_packed_seq_mask(
                tuple(ps),
                block_sizes=tuple(temp_block_sizes),
                prefix_lens=tuple(prefix_lens),
                max_seq_len=max_seq_len,
                device=device,
            )
        elif self.mask_mode == "flexible_time_block_causal_last":
            temp_block_sizes = [sum([psi.numel() for psi in ps]) for ps in packed_shapes_list]
            prefix_lens = [sum([shape.numel() for shape in psi[:-1]]) for psi in packed_shapes_list]
            config = self.cross_timestep_attention_config
            mask_fn, seq_ids = generate_flexible_time_block_causal_prefix_packed_seq_mask(
                tuple(ps),
                block_sizes=tuple(temp_block_sizes),
                prefix_lens=tuple(prefix_lens),
                prefix2prefix=config["prefix2prefix"],
                prefix2nonprefix=config["prefix2nonprefix"],
                nonprefix2prefix=config["nonprefix2prefix"],
                nonprefix2nonprefix=config["nonprefix2nonprefix"],
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            raise ValueError(f"Invalid mask mode {self.mask_mode}")

        N = len(seq_ids)
        assert (
            N % 128 == 0
        ), f"flex_attention sequence length must be a multiple of 128, but current is {N}."
        block_mask = self.create_block_mask(
            mask_fn, None, None, N, N, device=device, _compile=self.compile_block_mask
        )

        # Optionally zero-pad packed sequence
        # Outer packed shapes can be used to remove padding
        num_padding_tokens = N - N_orig
        if num_padding_tokens > 0:
            tensors_packed = F.pad(tensors_packed, (0, 0, 0, num_padding_tokens))

        # Optionally add an embedding packing function that specifies how elements
        # of a tensor of shape (B, L, ...) should be expanded to sequences of shape
        # (B, l1, ...), (B, l2, ...), ..., (B, lL, ...). Does not take into account
        # any padding.
        if self.emb_packing_fn_write_key is not None:
            if self.per_subseq_embs:
                emb_packing_fn = partial(
                    expand_emb_per_interleaved_subseq,
                    packed_shapes_list=packed_shapes_list,
                    interleave_size=T,
                )
            else:
                raise NotImplementedError
                seq_lens = [shape.numel() for shape in ps]
                emb_packing_fn = partial(expand_emb, seq_lens=seq_lens)

            data_dict[self.emb_packing_fn_write_key] = emb_packing_fn

        data_dict[self.packed_seq_write_key] = tensors_packed
        data_dict[self.block_mask_write_key] = block_mask
        data_dict[self.inner_packed_shapes_write_key] = packed_shapes_list
        data_dict[self.outer_packed_shapes_write_key] = ps

        # Create repa block mask if requested (attention over patches only, no registers)
        if self.repa_block_mask_write_key is not None:
            # For TransformerHead, we need a batched mask where each batch item is independent
            # Extract patch spatial size (H*W) from the first packed shape list
            spatial_size = packed_shapes_list[0][self.repa_patch_index].numel()

            # Total sequence length per batch: T * spatial_size
            seq_len_per_batch = T * spatial_size

            # Pad to multiple of 128 as required by FlexAttention
            # max_seq_len = next_highest_multiple(seq_len_per_batch, 128)
            max_seq_len = seq_len_per_batch

            # Create mask based on repa_mask_mode
            # We create batch masked as the repa head processes in batches
            if self.repa_mask_mode == "full":
                mask_fn = generate_batched_full_mask(seq_len_per_batch, device=device)
            elif self.repa_mask_mode == "time_block_causal":
                mask_fn = generate_batched_time_block_causal_mask(
                    seq_len_per_batch, spatial_size, device=device
                )
            else:
                raise ValueError(f"Invalid repa_mask_mode {self.repa_mask_mode}")

            # Number of batches (samples)
            B = len(packed_shapes_list)

            assert (
                max_seq_len % 128 == 0
            ), f"flex_attention sequence length must be a multiple of 128, but is {max_seq_len}."

            repa_block_mask = self.create_block_mask(
                mask_fn,
                B,
                None,
                max_seq_len,
                max_seq_len,
                device=device,
                _compile=self.compile_block_mask,
            )

            data_dict[self.repa_block_mask_write_key] = repa_block_mask

        return data_dict


class BlockWiseSequencePackerWithCrossAttention(nn.Module):
    """
    TODO(roman-bachmann)
    """

    def __init__(
        self,
        input_list_read_keys: List[str],
        context_list_read_keys: List[str],
        packed_seq_write_key: str,
        packed_context_write_key: str,
        sa_block_mask_write_key: str,
        xa_block_mask_write_key: str,
        inner_packed_shapes_write_key: str,
        outer_packed_shapes_write_key: str,
        sa_mask_mode: str = "full",
        xa_mask_mode: str = "full",
        max_seq_len: Optional[int] = None,
        max_ctx_len: Optional[int] = None,
        pad_to_multiple: Optional[int] = None,
        pad_context_to_multiple: Optional[int] = None,
        emb_packing_fn_write_key: Optional[str] = None,
        per_subseq_embs: bool = False,
        compile_block_mask: bool = True,
    ):
        super().__init__()
        self.input_list_read_keys = input_list_read_keys
        self.context_list_read_keys = context_list_read_keys
        self.packed_seq_write_key = packed_seq_write_key
        self.packed_context_write_key = packed_context_write_key
        self.sa_block_mask_write_key = sa_block_mask_write_key
        self.xa_block_mask_write_key = xa_block_mask_write_key
        self.inner_packed_shapes_write_key = inner_packed_shapes_write_key
        self.outer_packed_shapes_write_key = outer_packed_shapes_write_key
        self.emb_packing_fn_write_key = emb_packing_fn_write_key

        self.sa_mask_mode = sa_mask_mode
        self.xa_mask_mode = xa_mask_mode

        self.max_seq_len = max_seq_len
        self.max_ctx_len = max_ctx_len
        self.pad_to_multiple = pad_to_multiple
        self.pad_context_to_multiple = pad_context_to_multiple
        if max_seq_len is not None and pad_to_multiple is not None:
            raise ValueError("Only one of max_seq_len or pad_to_multiple should be provided.")
        if max_ctx_len is not None and pad_context_to_multiple is not None:
            raise ValueError(
                "Only one of max_ctx_len or pad_context_to_multiple should be provided."
            )

        self.per_subseq_embs = per_subseq_embs

        self.compile_block_mask = compile_block_mask
        self.create_block_mask = torch.compiler.disable(create_block_mask_cached)

        # For pretty printing
        self._init_args = locals().copy()
        self._init_args.pop("self")
        self._init_args.pop("__class__")

    def __repr__(self):
        cls_name = self.__class__.__name__
        args_str = ",\n  ".join(f"{k}={v!r}" for k, v in self._init_args.items())
        return f"{cls_name}(\n  {args_str}\n)"

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:

        ##### 1 - Pack the main sequences #####

        # From the data_dict, get the lists containing the aligned tensors,
        # e.g., a list of image patches and a list of time embedding tokens.
        list_of_tensor_lists = [data_dict[key] for key in self.input_list_read_keys]

        # Concatenate each sample across the lists, e.g., images[0] | temb[0], images[1] | temb[1], ...
        # Keep track of the shape of these inner tensors
        tensors_concat_list = []
        packed_shapes_list = []
        for tensors in strict_zip(*list_of_tensor_lists):
            # tensors contains the i-th entries of each of the lists in list_of_tensor_lists
            sample_packed, ps = einops.pack(tensors, "b * d")
            tensors_concat_list.append(sample_packed)
            packed_shapes_list.append(ps)

        # Pack tensors into one large sequence
        tensors_packed, ps_seq = einops.pack(tensors_concat_list, "b * d")
        B, N_orig, D = tensors_packed.shape

        # TODO(roman-bachmann): Only supporting B=1 until https://github.com/pytorch/pytorch/issues/134560 is resolved
        assert B == 1

        device = tensors_packed.device

        # Create full or causal block-wise self-attention mask using FlexAttention. Optionally pad sequences.
        if self.pad_to_multiple is not None:
            max_seq_len = next_highest_multiple(N_orig, self.pad_to_multiple)
        else:
            max_seq_len = self.max_seq_len
        if self.sa_mask_mode == "full":
            sa_mask_fn, seq_ids = generate_packed_seq_mask(
                tuple(ps_seq),
                max_seq_len=max_seq_len,
                device=device,
            )
        elif self.sa_mask_mode == "causal":
            sa_mask_fn, seq_ids = generate_causal_packed_seq_mask(
                tuple(ps_seq),
                max_seq_len=max_seq_len,
                device=device,
            )
        elif self.sa_mask_mode == "causal_last":
            prefix_lens = [
                sum([shape.numel() for shape in ps_i[:-1]]) for ps_i in packed_shapes_list
            ]
            sa_mask_fn, seq_ids = generate_prefix_packed_seq_mask(
                tuple(ps_seq),
                tuple(prefix_lens),
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            raise ValueError(f"Invalid mask mode {self.sa_mask_mode}")

        N = len(seq_ids)
        assert (
            N % 128 == 0
        ), f"flex_attention sequence length must be a multiple of 128, but current is {N}."
        sa_block_mask = self.create_block_mask(
            sa_mask_fn, None, None, N, N, device=device, _compile=self.compile_block_mask
        )

        # Optionally zero-pad packed sequence
        # Outer packed shapes can be used to remove padding
        num_padding_tokens = N - N_orig
        if num_padding_tokens > 0:
            tensors_packed = F.pad(tensors_packed, (0, 0, 0, num_padding_tokens))

        # Optionally add an embedding packing function that specifies how elements
        # of a tensor of shape (B, L, ...) should be expanded to sequences of shape
        # (B, l1, ...), (B, l2, ...), ..., (B, lL, ...). Does not take into account
        # any padding.
        if self.emb_packing_fn_write_key is not None:
            seq_lens = [shape.numel() for shape in ps_seq]
            if self.per_subseq_embs:
                emb_packing_fn = partial(
                    expand_emb_per_subseq, packed_shapes_list=packed_shapes_list
                )
            else:
                emb_packing_fn = partial(expand_emb, seq_lens=seq_lens)
            data_dict[self.emb_packing_fn_write_key] = emb_packing_fn

        data_dict[self.packed_seq_write_key] = tensors_packed
        data_dict[self.sa_block_mask_write_key] = sa_block_mask
        data_dict[self.inner_packed_shapes_write_key] = packed_shapes_list
        data_dict[self.outer_packed_shapes_write_key] = ps_seq

        ##### 2 - Now do the same with the context #####

        list_of_ctx_tensor_lists = [data_dict[key] for key in self.context_list_read_keys]
        ctx_concat_list = []
        for tensors in strict_zip(*list_of_ctx_tensor_lists):
            # tensors contains the i-th entries of each of the lists in list_of_tensor_lists
            sample_packed, ps = einops.pack(tensors, "b * d")
            ctx_concat_list.append(sample_packed)

        # Pack context tensors into one large sequence
        context_packed, ps_ctx = einops.pack(ctx_concat_list, "b * d")
        B, N_ctx_orig, D = context_packed.shape
        assert B == 1

        # Create full or causal block-wise cross-attention mask using FlexAttention. Optionally pad sequences.
        if self.pad_context_to_multiple is not None:
            max_ctx_len = next_highest_multiple(N_orig, self.pad_context_to_multiple)
        else:
            max_ctx_len = self.max_ctx_len

        if self.xa_mask_mode == "full":
            xa_mask_fn, seq_ids, ctx_ids = generate_packed_xattn_mask(
                tuple(ps_seq),
                tuple(ps_ctx),
                max_seq_len=max_seq_len,
                max_ctx_len=max_ctx_len,
                device=device,
            )
        else:
            raise ValueError(f"Invalid mask mode {self.xa_mask_mode}")

        N_ctx = len(ctx_ids)
        assert (
            N_ctx % 128 == 0
        ), f"flex_attention sequence length must be a multiple of 128, but current is {N_ctx}."
        xa_block_mask = self.create_block_mask(
            xa_mask_fn, None, None, N_ctx, N_ctx, device=device, _compile=self.compile_block_mask
        )

        # Optionally zero-pad packed context
        num_padding_tokens = N_ctx - N_ctx_orig
        if num_padding_tokens > 0:
            context_packed = F.pad(context_packed, (0, 0, 0, num_padding_tokens))

        data_dict[self.packed_context_write_key] = context_packed
        data_dict[self.xa_block_mask_write_key] = xa_block_mask

        return data_dict
