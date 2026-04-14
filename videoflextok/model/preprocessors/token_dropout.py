# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
import math
import random
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .registers import halving_sequence, is_power_of_two, next_power_of_2, powers_of_two

__all__ = [
    "MaskedNestedDropout",
    "TruncatedNestedDropout",
    "MaskedRangeDropout",
    "TemporalMaskedNestedDropout",
    "MaskedPrefixDropout",
]


SamplingMode = Literal["uniform", "pow2", "uniform_pow2", "zipf", "pow2_roundup", "uniform_halved"]


def sample_uniform_pow2_np(N, size):
    k = np.random.randint(1, N + 1, size=size)
    return 1 << np.ceil(np.log2(k)).astype(np.int64)


class MaskedNestedDropout(nn.Module):
    """
    Module that randomly drops tokens of the given tensors in a nested fashion, i.e.
    performs nested dropout / Matryoshka sampling.

    Args:
        read_write_key: Key to apply the nested dropout on.
        dim: Dimension size of the mask token.
        eval_keep_k_read_key: During inference, by default nested dropout is disabled.
            This key allows to optionally choose the number of tokens to keep per tensor.
        train_keep_k_write_key: During training, write the number of kept tokens to this key.
        size_sampling_mode: Method to sample the number of tokens to randomly drop.
    """

    def __init__(
        self,
        read_write_key: str,
        dim: int,
        eval_keep_k_read_key: Optional[str] = "eval_keep_k",
        train_keep_k_write_key: Optional[str] = "train_keep_k",
        size_sampling_mode: SamplingMode = "uniform",
        zipf_a: float = 0.5,
    ):
        super().__init__()
        self.read_write_key = read_write_key
        self.dim = dim
        self.eval_keep_k_read_key = eval_keep_k_read_key
        self.train_keep_k_write_key = train_keep_k_write_key
        self.size_sampling_mode = size_sampling_mode
        self.zipf_a = zipf_a

        self.dropout_mask_token = nn.Parameter(torch.randn(self.dim), requires_grad=True)
        trunc_normal_(self.dropout_mask_token, std=0.02)

    def sample_keep_k(self, N):
        if self.size_sampling_mode == "uniform":
            keep_k = np.random.randint(low=1, high=N + 1)
        elif self.size_sampling_mode == "pow2":
            assert is_power_of_two(N)
            keep_k = np.random.choice(powers_of_two(1, N))
        elif self.size_sampling_mode == "pow2_roundup":
            _n = next_power_of_2(N)
            keep_k = np.random.choice(powers_of_two(1, _n))
        elif self.size_sampling_mode == "uniform_pow2":
            assert is_power_of_two(N)
            k = np.random.randint(low=1, high=N + 1)
            keep_k = k if is_power_of_two(k) else 1 << k.bit_length()
        elif self.size_sampling_mode == "uniform_halved":
            k = np.random.randint(low=1, high=N + 1)
            q = N // k
            keep_k = N >> (q.bit_length() - 1)
        elif self.size_sampling_mode == "halved":
            keep_k = np.random.choice(halving_sequence(N))
        elif self.size_sampling_mode == "zipf":
            probs = np.array([(1.0 / k) ** self.zipf_a for k in range(1, N + 1)])
            probs /= probs.sum()
            keep_k = np.random.choice(np.arange(1, N + 1), p=probs)
        else:
            raise ValueError(f"size_sampling_mode {self.size_sampling_mode} is not defined.")
        return keep_k

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # assumes an input of shapes [*, N, D], where * is any number of dimensions
        # it applies dropout to the 2nd last dimension

        if not self.training:
            if self.eval_keep_k_read_key is None:
                return data_dict
            if self.eval_keep_k_read_key not in data_dict:
                return data_dict

            for i in range(len(data_dict[self.read_write_key])):
                keep_k = data_dict[self.eval_keep_k_read_key][i]
                data_dict[self.read_write_key][i][..., keep_k:, :] = self.dropout_mask_token
        else:
            keep_ks = []
            for i in range(len(data_dict[self.read_write_key])):
                N = data_dict[self.read_write_key][i].shape[-2]
                keep_k = self.sample_keep_k(N)
                keep_ks.append(keep_k)
                data_dict[self.read_write_key][i][..., keep_k:, :] = self.dropout_mask_token

            data_dict[self.train_keep_k_write_key] = keep_ks

        return data_dict


class TemporalMaskedNestedDropout(MaskedNestedDropout):
    def __init__(
        self,
        read_write_key: str,
        dim: int,
        eval_keep_k_read_key: Optional[str] = "eval_keep_k",
        train_keep_k_write_key: Optional[str] = "train_keep_k",
        size_sampling_mode: Union[SamplingMode, List[SamplingMode]] = "pow2",
        zipf_a: float = 0.5,
    ):
        super().__init__(
            read_write_key,
            dim,
            eval_keep_k_read_key,
            train_keep_k_write_key,
            size_sampling_mode,
            zipf_a,
        )

        # this is to handle things like omegaconf ListConfig
        if isinstance(self.size_sampling_mode, Iterable) and not isinstance(
            self.size_sampling_mode, str
        ):
            self.size_sampling_mode = [sm for sm in self.size_sampling_mode]

    def sample_keep_k(
        self,
        N: int,  # total number of tokens per each timestep
        timesteps: int,  # the number of timesteps to sample keep_k for
        sampling_mode: Optional[Union[SamplingMode, List[SamplingMode]]] = None,
    ) -> torch.LongTensor:
        # Samples keep_k out of N for each timestep
        # returns a tensor of shape [timestep,]

        sampling_mode = sampling_mode or self.size_sampling_mode

        if isinstance(sampling_mode, list):
            if len(sampling_mode) == 1:
                return self.sample_keep_k(N, timesteps, sampling_mode[0])
            elif len(sampling_mode) == 2:
                keep_k_1st = self.sample_keep_k(N, 1, sampling_mode[0])
                keep_k_rest = self.sample_keep_k(N, timesteps - 1, sampling_mode[1])
                return np.concatenate([keep_k_1st, keep_k_rest])
            else:
                return np.concatenate(
                    [self.sample_keep_k(N, 1, sampling_mode[t]) for t in range(timesteps)]
                )

        if sampling_mode == "uniform":
            keep_k = np.random.randint(low=1, high=N + 1, size=(timesteps,))
        elif sampling_mode == "pow2":
            assert is_power_of_two(N)
            keep_k = np.random.choice(powers_of_two(1, N), size=(timesteps,))
        elif sampling_mode == "uniform_pow2":
            assert is_power_of_two(N)
            keep_k = sample_uniform_pow2_np(N, timesteps)
        else:
            raise ValueError(f"size_sampling_mode {sampling_mode} is not defined or implemented.")
        return keep_k

    def _mask_keep(self, x, keep_k):
        T, N, _ = x.shape[-3:]
        dev = x.device
        keep_k = keep_k.to(dev).clamp(0, N)

        # mask_TN: (T, N) — small, independent of batch dims
        mask_TN = torch.arange(N, device=dev).unsqueeze(0) < keep_k.unsqueeze(1)

        # Broadcast to (..., T, N, 1) inside where (no big allocation of batch-sized mask)
        return torch.where(
            mask_TN.view(*([1] * (x.ndim - 3)), T, N, 1),
            x,
            self.dropout_mask_token,
        )

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # assumes an input of shapes [*, T, N, D], where * is any number of dimensions
        # it applies dropout to the 2nd last dimension independently for each T

        if not self.training:
            if self.eval_keep_k_read_key is None:
                return data_dict
            if self.eval_keep_k_read_key not in data_dict:
                return data_dict

            for i in range(len(data_dict[self.read_write_key])):
                keep_k = data_dict[self.eval_keep_k_read_key][i]

                if isinstance(keep_k, int):
                    keep_k = [keep_k]

                keep_k = torch.LongTensor(keep_k)

                T = data_dict[self.read_write_key][i].shape[-3]
                # if a single number is provided broadcast it to all timesteps
                if keep_k.shape[0] == 1:
                    keep_k = torch.repeat_interleave(keep_k, T)

                assert (
                    keep_k.shape[0] == T
                ), f"keep_k should match the termporal dimension of the input, but given {keep_k.shape=}"

                data_dict[self.read_write_key][i] = self._mask_keep(
                    data_dict[self.read_write_key][i], keep_k
                )
        else:
            keep_ks = []
            for i in range(len(data_dict[self.read_write_key])):
                N = data_dict[self.read_write_key][i].shape[-2]
                T = data_dict[self.read_write_key][i].shape[-3]
                keep_k = torch.LongTensor(self.sample_keep_k(N, timesteps=T))
                keep_ks.append(keep_k)

                data_dict[self.read_write_key][i] = self._mask_keep(
                    data_dict[self.read_write_key][i], keep_k
                )

            data_dict[self.train_keep_k_write_key] = keep_ks

        return data_dict


class TruncatedNestedDropout(nn.Module):
    """
    Module that randomly drops tokens of the given tensors in a nested fashion, i.e.
    performs nested dropout / Matryoshka sampling. Unlike MaskedNestedDropout that
    masks the dropped tokens, this module simply truncates the sequences.

    Args:
        read_write_key: Key to apply the nested dropout on.
        eval_keep_k_read_key: During inference, by default nested dropout is disabled.
            This key allows to optionally choose the number of tokens to keep per tensor.
        train_keep_k_write_key: During training, write the number of kept tokens to this key.
        size_sampling_mode: Method to sample the number of tokens to randomly drop.
    """

    # TODO(roman-bachmann): Avoid code duplication with MaskedNestedDropout

    def __init__(
        self,
        read_write_key: str,
        eval_keep_k_read_key: Optional[str] = "eval_keep_k",
        train_keep_k_write_key: Optional[str] = "train_keep_k",
        size_sampling_mode: Literal[
            "uniform", "pow2", "uniform_pow2", "zipf", "uniform_halved"
        ] = "uniform",
        zipf_a: float = 0.5,
    ):
        super().__init__()
        self.read_write_key = read_write_key
        self.eval_keep_k_read_key = eval_keep_k_read_key
        self.train_keep_k_write_key = train_keep_k_write_key
        self.size_sampling_mode = size_sampling_mode
        self.zipf_a = zipf_a

    def sample_keep_k(self, N):
        if self.size_sampling_mode == "uniform":
            keep_k = np.random.randint(low=1, high=N + 1)
        elif self.size_sampling_mode == "pow2":
            assert is_power_of_two(N)
            keep_k = np.random.choice(powers_of_two(1, N))
        elif self.size_sampling_mode == "uniform_pow2":
            k = np.random.randint(low=1, high=N + 1)
            keep_k = k if is_power_of_two(k) else 1 << k.bit_length()
        elif self.size_sampling_mode == "uniform_halved":
            k = np.random.randint(low=1, high=N + 1)
            q = N // k
            keep_k = N >> (q.bit_length() - 1)
        elif self.size_sampling_mode == "zipf":
            probs = np.array([(1.0 / k) ** self.zipf_a for k in range(1, N + 1)])
            probs /= probs.sum()
            keep_k = np.random.choice(np.arange(1, N + 1), p=probs)
        else:
            raise ValueError(f"size_sampling_mode {self.size_sampling_mode} is not defined.")
        return keep_k

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.training:
            if self.eval_keep_k_read_key is None:
                return data_dict
            if self.eval_keep_k_read_key not in data_dict:
                return data_dict

            for i in range(len(data_dict[self.read_write_key])):
                keep_k = data_dict[self.eval_keep_k_read_key][i]
                data_dict[self.read_write_key][i] = data_dict[self.read_write_key][i][:, :keep_k, :]
        else:
            keep_ks = []
            for i in range(len(data_dict[self.read_write_key])):
                N = data_dict[self.read_write_key][i].shape[1]
                keep_k = self.sample_keep_k(N)
                keep_ks.append(keep_k)
                data_dict[self.read_write_key][i] = data_dict[self.read_write_key][i][:, :keep_k, :]
            data_dict[self.train_keep_k_write_key] = keep_ks

        return data_dict


@lru_cache
def get_subsequence_ranges(N):
    if not is_power_of_two(N):
        raise ValueError("N is not a power of two")
    k = int(math.log2(N))
    lengths = [2**i for i in range(k)]
    start = 0
    ranges = []
    for length in lengths:
        end = start + length - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


class MaskedRangeDropout(nn.Module):
    """
    Module that randomly keeps only certain ranges of the given tensors and drops the rest.
    The range to keep per data point is sampled uniformly.

    Args:
        read_write_key: Key to apply the nested dropout on.
        dim: Dimension size of the mask token.
        eval_keep_k_read_key: During inference, by default only the last range is used.
            This key allows to optionally choose the range of tokens to keep per tensor.
            k specifies the index of the range.
        size_sampling_mode: Method to sample the range of tokens to randomly drop/keep.
    """

    def __init__(
        self,
        read_write_key: str,
        dim: int,
        eval_keep_k_read_key: Optional[str] = "eval_keep_k",
        size_sampling_mode: Literal["pow2", "custom"] = "pow2",
        custom_ranges: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.read_write_key = read_write_key
        self.dim = dim
        self.eval_keep_k_read_key = eval_keep_k_read_key
        self.size_sampling_mode = size_sampling_mode
        self.custom_ranges = custom_ranges

        self.dropout_mask_token = nn.Parameter(torch.randn(self.dim), requires_grad=True)
        trunc_normal_(self.dropout_mask_token, std=0.02)

    def sample_range(self, N):
        if self.size_sampling_mode == "pow2":
            idx_start, idx_end = random.choice(get_subsequence_ranges(N))
        elif self.size_sampling_mode == "custom":
            idx_start, idx_end = random.choice(self.custom_ranges)
        else:
            raise ValueError(f"size_sampling_mode {self.size_sampling_mode} is not defined.")
        return idx_start, idx_end

    def select_range(self, N, k):
        if self.size_sampling_mode == "pow2":
            idx_start, idx_end = get_subsequence_ranges(N)[k]
        elif self.size_sampling_mode == "custom":
            idx_start, idx_end = self.custom_ranges[k]
        else:
            raise ValueError(f"size_sampling_mode {self.size_sampling_mode} is not defined.")
        return idx_start, idx_end

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        B = len(data_dict[self.read_write_key])
        if not self.training:
            if self.eval_keep_k_read_key is None:
                keep_ks = [-1] * B  # Last range
            elif self.eval_keep_k_read_key not in data_dict:
                keep_ks = [-1] * B  # Last range
            else:
                keep_ks = data_dict[self.eval_keep_k_read_key]

            for i in range(B):
                N = data_dict[self.read_write_key][i].shape[1]
                idx_start, idx_end = self.select_range(N, k=keep_ks[i])
                data_dict[self.read_write_key][i][:, :idx_start] = self.dropout_mask_token
                data_dict[self.read_write_key][i][:, idx_end + 1 :] = self.dropout_mask_token
        else:
            for i in range(B):
                N = data_dict[self.read_write_key][i].shape[1]
                idx_start, idx_end = self.sample_range(N)
                data_dict[self.read_write_key][i][:, :idx_start] = self.dropout_mask_token
                data_dict[self.read_write_key][i][:, idx_end + 1 :] = self.dropout_mask_token

        return data_dict


class MaskedPrefixDropout(nn.Module):
    """
    Module that masks prefix tokens (usually corresponding to the conditioning signal) based on the prefix length.
    This is designed to work with PrefixMinRFNoiseModule, which provides the actual
    prefix length used for each sample.

    The module replaces the first N tokens (where N is the prefix length) with a learned
    dropout_mask_token. This allows the model to learn to handle masked conditioning
    during training.

    Args:
        read_write_key: Key to apply the prefix dropout on.
        dim: Dimension size of the mask token.
        prefix_len_read_key: Key to read the prefix length from during training (provided by noise module).
            This is an integer indicating how many frames at the beginning to mask.
        eval_prefix_len_read_key: During inference, optionally read the prefix length from this key.
            If None or key not in data_dict, prefix dropout is disabled during eval.
    """

    def __init__(
        self,
        read_write_key: str,
        dim: int,
        prefix_len_read_key: str = "prefix_len",
        eval_prefix_len_read_key: Optional[str] = "eval_prefix_len",
    ):
        super().__init__()
        self.read_write_key = read_write_key
        self.dim = dim
        self.prefix_len_read_key = prefix_len_read_key
        self.eval_prefix_len_read_key = eval_prefix_len_read_key

        self.dropout_mask_token = nn.Parameter(torch.randn(self.dim), requires_grad=True)
        trunc_normal_(self.dropout_mask_token, std=0.02)

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.training:
            # During evaluation, only apply masking if eval key is provided and exists
            if (
                self.eval_prefix_len_read_key is None
                or self.eval_prefix_len_read_key not in data_dict
            ):
                return data_dict

            prefix_lens = data_dict[self.eval_prefix_len_read_key]
        else:
            # During training, use the training key
            prefix_lens = data_dict[self.prefix_len_read_key]

        for i in range(len(data_dict[self.read_write_key])):
            prefix_len = prefix_lens[i]

            # Skip if no prefix to mask
            if prefix_len == 0:
                continue

            # Get the data tensor - assumes shape [*, T, N, D]
            # where * is any number of batch dimensions, T is temporal, N is register tokens, D is feature dim
            x = data_dict[self.read_write_key][i]

            # Replace the first prefix_len timestamps with the dropout mask token
            x[..., :prefix_len, :, :] = self.dropout_mask_token

            data_dict[self.read_write_key][i] = x

        return data_dict
