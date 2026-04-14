# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import einops
import mup

import numpy as np
import torch
import torch.nn as nn

from videoflextok.model.trunks.transformers import FlexTransformer
from videoflextok.utils.misc import get_autocast_context, str_to_dtype
from videoflextok.utils.mup_layers import MuReadoutFSDP

from ..layers.causalconv import CausalConv3d
from ..layers.norm import Fp32LayerNorm, Fp32SpatioTemporalLayerNorm
from ..utils.packed_ops import packed_proj

__all__ = ["LinearHead", "unpatchify", "ToPatchesLinearHead", "CausalConv3dHead"]


def modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


def expand_emb_from_ps(emb, ps):
    return torch.cat(
        [einops.repeat(emb_i, "d -> 1 n d", n=shape.numel()) for emb_i, shape in zip(emb, ps)],
        dim=1,
    )


class LinearHead(nn.Module):
    """
    Linear head module with optional adaLN modulation.
    Compatible with μP, i.e. using MuReadout if use_mup_readout=True.

    Args:
        read_key: Key to read input tensor from the input dictionary.
        write_key: Key to write output tensor into the output dictionary.
        dim_in: Input dimension size.
        dim_out: Output dimension size.
        use_mup_readout: Whether to use μP-compatible readout layer (MuReadout) instead of a standard linear layer.
        weight_init_style: Initialization style for weights ('zero', 'xavier', or 'trunc_normal').
        norm_layer: Optional normalization layer applied before the projection.
        dtype_override: Optional string to override the tensor data type.
        adaLN_emb_read_key: Key to read embedding tensor for adaLN-Zero modulation.
        adaLN_packing_fn_read_key: Key to read packing function for adaLN-Zero modulation.
        adaLN_bias: Whether to use bias in the adaLN-Zero modulation layer.
        proj_bias: Whether to use bias in the projection layer.
    """

    def __init__(
        self,
        read_key: str,
        write_key: str,
        dim: int,
        dim_out: int,
        use_mup_readout: bool,
        weight_init_style: str = "zero",
        norm_layer: Optional[Union[partial, nn.Module]] = partial(
            Fp32LayerNorm, bias=False, elementwise_affine=False
        ),
        dtype_override: Optional[str] = None,
        adaLN_emb_read_key: Optional[str] = None,
        adaLN_packing_fn_read_key: Optional[str] = None,
        adaLN_bias: bool = True,
        proj_bias: bool = True,
    ):
        super().__init__()
        self.read_key, self.write_key = read_key, write_key
        self.dim_in, self.dim_out = dim, dim_out
        self.dtype_override = str_to_dtype(dtype_override)
        self.adaLN_emb_read_key, self.adaLN_packing_fn_read_key = (
            adaLN_emb_read_key,
            adaLN_packing_fn_read_key,
        )

        # Optional LayerNorm
        self.norm = norm_layer(self.dim_in) if norm_layer is not None else nn.Identity()

        # Optional adaLN-Zero
        if self.adaLN_emb_read_key is not None:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(self.dim_in, 2 * self.dim_in, bias=adaLN_bias)
            )

        # Linear projection head. Using custom layer for μP.
        proj_layer = MuReadoutFSDP if use_mup_readout else nn.Linear
        self.proj = proj_layer(self.dim_in, self.dim_out, bias=proj_bias)

        # Weight init
        self.weight_init_style = weight_init_style
        self.init_weights_sp()

    def init_weights_sp(self):
        """SP weight initialization scheme"""
        for name, m in self.named_modules():
            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if self.weight_init_style == "zero" or "adaLN_modulation" in name:
                    nn.init.constant_(m.weight, 0)
                elif self.weight_init_style == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_weights_muP(self):
        """μP weight initialization scheme. Since the projection layer is a MuReadout, we call
        the SP init function on it. adaLN and LayerNorm layers are zero-initialized.
        """
        self.init_weights_sp()

    def forward_default(self, x: torch.Tensor) -> torch.Tensor:
        with get_autocast_context(x, self.dtype_override):
            x = self.proj(self.norm(x))
        return x

    def forward_adaLN(
        self, x: torch.Tensor, adaLN_emb: torch.Tensor, adaLN_packing_fn: Callable
    ) -> torch.Tensor:
        with get_autocast_context(x, self.dtype_override):
            x = self.norm(x)
            shift, scale = adaLN_packing_fn(self.adaLN_modulation(adaLN_emb)).chunk(2, dim=-1)
            x = modulate(x, shift, scale)
            x = self.proj(x)
        return x

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x_list = data_dict[self.read_key]
        x_packed, ps = einops.pack(x_list, "b * d")

        if self.adaLN_emb_read_key is not None:
            adaLN_emb = data_dict[self.adaLN_emb_read_key]
            if self.adaLN_packing_fn_read_key is not None:
                adaLN_packing_fn = data_dict[self.adaLN_packing_fn_read_key]
            else:
                adaLN_packing_fn = partial(expand_emb_from_ps, ps=ps)
            x_packed = self.forward_adaLN(x_packed, adaLN_emb, adaLN_packing_fn)
        else:
            x_packed = self.forward_default(x_packed)

        data_dict[self.write_key] = einops.unpack(x_packed, ps, "b * d")
        return data_dict


def unpatchify(patches: torch.Tensor, patch_sizes: List[int]) -> torch.Tensor:
    """
    Inverse patching operation, i.e. given a list of patch sizes [p1, p2, ..., pN]
    and a tensor of shape (b, n1, n2, ..., nN, p1*p2*...*pN*d), unpatches it to
    shape (b, n1*p1, n2*p2, ..., nN*pN, d).

    Args:
        patches: Patch tensor of shape (b, n1, n2, ..., nN, p1*p2*...*pN*d)
        patch_sizes: List of patch sizes for each spatial dimension [p1, p2, ..., pN]

    Returns:
        Reconstructed tensor of shape (b, n1*p1, n2*p2, ..., nN*pN, d)
    """
    # TODO(roman-bachmann): Add option to unpatch flattened tensors given list of sizes
    N = patches.dim() - 2
    assert len(patch_sizes) == N, "patch_sizes must match the number of spatial dimensions"

    # Construct patterns for rearrangement
    input_pattern = (
        "b "
        + " ".join([f"n{i}" for i in range(N)])
        + " ("
        + " ".join([f"p{i}" for i in range(N)])
        + " d)"
    )
    output_pattern = "b " + " ".join([f"(n{i} p{i})" for i in range(N)]) + " d"

    # Prepare keyword arguments for einops
    rearrange_kwargs = {f"p{i}": patch_sizes[i] for i in range(N)}

    # Perform the rearrangement to reconstruct the original tensor
    x = einops.rearrange(patches, f"{input_pattern} -> {output_pattern}", **rearrange_kwargs)

    return x


class ToPatchesLinearHead(LinearHead):
    """
    Linear head and inverse patching function for arbitrary patch sizes and dimensions.
    With optional adaLN modulation.
    Compatible with μP, i.e. using MuReadout if use_mup_readout=True.

    Args:
        read_key: Key to read input tensor from the input dictionary.
        write_key: Key to write output tensor into the output dictionary.
        dim_in: Input dimension size.
        channels_out: Numer of output channels. Needed to compute patch dimension.
        patch_sizes: List of patch sizes per dimension.
        use_mup_readout: Whether to use μP-compatible readout layer (MuReadout) instead of a standard linear layer.
        weight_init_style: Initialization style for weights ('zero', 'xavier', or 'trunc_normal').
        norm_layer: Optional normalization layer applied before the projection.
        dtype_override: Optional string to override the tensor data type.
        adaLN_emb_read_key: Key to read embedding tensor for adaLN-Zero modulation.
        adaLN_packing_fn_read_key: Key to read packing function for adaLN-Zero modulation.
        adaLN_bias: Whether to use bias in the adaLN-Zero modulation layer.
        proj_bias: Whether to use bias in the projection layer.
    """

    def __init__(
        self,
        read_key: str,
        write_key: str,
        dim: int,
        channels_out: int,
        patch_sizes: List[int],
        use_mup_readout: bool,
        weight_init_style: str = "zero",
        norm_layer: Optional[Union[partial, nn.Module]] = partial(
            Fp32LayerNorm, bias=False, elementwise_affine=False
        ),
        dtype_override: Optional[str] = None,
        adaLN_emb_read_key: Optional[str] = None,
        adaLN_packing_fn_read_key: Optional[str] = None,
        adaLN_bias: bool = True,
        proj_bias: bool = True,
    ):
        patch_dim = channels_out * np.prod(patch_sizes)
        super().__init__(
            read_key=read_key,
            write_key=write_key,
            dim=dim,
            dim_out=patch_dim,
            use_mup_readout=use_mup_readout,
            weight_init_style=weight_init_style,
            norm_layer=norm_layer,
            dtype_override=dtype_override,
            adaLN_emb_read_key=adaLN_emb_read_key,
            adaLN_packing_fn_read_key=adaLN_packing_fn_read_key,
            adaLN_bias=adaLN_bias,
            proj_bias=proj_bias,
        )
        self.patch_sizes = patch_sizes

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x_list = data_dict[self.read_key]
        x_packed, ps = einops.pack(x_list, "b * d")

        if self.adaLN_emb_read_key is not None:
            adaLN_emb = data_dict[self.adaLN_emb_read_key]
            if self.adaLN_packing_fn_read_key is not None:
                adaLN_packing_fn = data_dict[self.adaLN_packing_fn_read_key]
            else:
                adaLN_packing_fn = partial(expand_emb_from_ps, ps=ps)
            x_packed = self.forward_adaLN(x_packed, adaLN_emb, adaLN_packing_fn)
        else:
            x_packed = self.forward_default(x_packed)

        x_list = einops.unpack(x_packed, ps, "b * d")
        x_list = [unpatchify(x, self.patch_sizes) for x in x_list]

        data_dict[self.write_key] = x_list
        return data_dict


class MLPHead(nn.Module):
    """
    MLP head module.
    Compatible with μP, i.e. using MuReadout if use_mup_readout=True.

    Args:
        read_key: Key to read input tensor from the input dictionary.
        write_key: Key to write output tensor into the output dictionary.
        dim: Input dimension size.
        dim_out: Output dimension size.
        num_layers: Number of MLP layers
        use_mup_readout: Whether to use μP-compatible readout layer (MuReadout) instead of a standard linear layer.
        dim_hidden_ratio: MLP hidden dimension ratio.
        act_layer: Activation layer used in the MLP.
        weight_init_style: Initialization style for weights ('xavier', or 'trunc_normal').
        zero_init_out_proj: Whether or not to zero-init the final out projection layer.
        norm_layer: Optional normalization layer applied before the projection.
        dtype_override: Optional string to override the tensor data type.
        bias: Whether to use bias in the linear layers.
    """

    def __init__(
        self,
        read_key: str,
        write_key: str,
        dim: int,
        dim_out: int,
        num_layers: int,
        use_mup_readout: bool,
        dim_hidden_ratio: int = 4.0,
        act_layer: nn.Module = nn.SiLU,
        weight_init_style: str = "xavier",
        zero_init_out_proj: bool = True,
        norm_layer: Optional[Union[partial, nn.Module]] = partial(
            Fp32LayerNorm, bias=False, elementwise_affine=False
        ),
        dtype_override: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.read_key, self.write_key = read_key, write_key
        self.dim_in, self.dim_out = dim, dim_out
        self.dim_hidden = int(dim_hidden_ratio * dim)
        self.num_layers = num_layers
        self.act_layer = act_layer
        self.dtype_override = str_to_dtype(dtype_override)
        self.bias = bias

        mlp_layers = []

        # Optional LayerNorm
        if norm_layer is not None:
            mlp_layers.append(norm_layer(self.dim_in))

        # Input projection
        mlp_layers.append(nn.Linear(self.dim_in, self.dim_hidden, bias=bias))
        mlp_layers.append(act_layer())

        # Hidden layers
        assert num_layers >= 2
        for _ in range(num_layers - 2):
            mlp_layers.append(nn.Linear(self.dim_hidden, self.dim_hidden, bias=bias))
            mlp_layers.append(act_layer())

        # Output projection head. Using custom layer for μP.
        out_proj_layer_fn = MuReadoutFSDP if use_mup_readout else nn.Linear
        self.out_proj = out_proj_layer_fn(self.dim_hidden, self.dim_out, bias=bias)

        # Full MLP without output projection
        self.mlp = nn.Sequential(*mlp_layers)

        # Weight init
        self.weight_init_style = weight_init_style
        self.zero_init_out_proj = zero_init_out_proj
        self.init_weights_sp()

    def init_weights_sp(self):
        """SP weight initialization scheme"""
        for name, m in self.named_modules():
            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if "out_proj" in name:
                    nn.init.constant_(m.weight, 0)
                elif self.weight_init_style == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_weights_muP(self):
        """μP weight initialization scheme."""
        for name, m in self.named_modules():
            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if "out_proj" in name:
                    nn.init.constant_(m.weight, 0)
                elif self.weight_init_style == "xavier":
                    mup.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    mup.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x_list = data_dict[self.read_key]
        x_packed, ps = einops.pack(x_list, "b * d")
        with get_autocast_context(x_packed, self.dtype_override):
            x_packed = self.out_proj(self.mlp(x_packed))
        data_dict[self.write_key] = einops.unpack(x_packed, ps, "b * d")
        return data_dict


class CausalConv3dHead(nn.Module):
    """
    Causal convolutional head module for temporal data.

    Args:
        read_key: Key to read input tensor from the input dictionary.
        write_key: Key to write output tensor into the output dictionary.
        dim: Input dimension size.
        dim_out: Output dimension size.
        num_layers: Number of convolutional layers.
        kernel_size: Kernel size for the convolutional layers.
        dim_hidden_ratio: Hidden dimension ratio.
        act_layer: Activation layer.
        norm_layer: Optional normalization layer.
        dtype_override: Optional string to override the tensor data type.
        pad_mode: Padding mode for causal convolution.
        bias: Whether to use bias in the convolutional layers.
    """

    def __init__(
        self,
        read_key: str,
        write_key: str,
        dim: int,
        dim_out: int,
        num_layers: int,
        kernel_size: Union[int, List[int]] = 3,
        dim_hidden_ratio: float = 4.0,
        act_layer: nn.Module = nn.SiLU,
        norm_layer: Optional[Union[partial, nn.Module]] = partial(
            Fp32SpatioTemporalLayerNorm, bias=False, elementwise_affine=False
        ),
        dtype_override: Optional[str] = None,
        pad_mode: str = "constant",
        bias: bool = True,
        weight_init_style: str = "xavier",
        enable_mup_init: bool = True,
    ):
        super().__init__()

        self.read_key, self.write_key = read_key, write_key
        self.dim_in, self.dim_out = dim, dim_out
        self.dim_hidden = int(dim_hidden_ratio * dim)
        self.num_layers = num_layers
        self.dtype_override = str_to_dtype(dtype_override)

        conv_layers = []

        # Input projection
        conv_layers.append(
            CausalConv3d(self.dim_in, self.dim_hidden, kernel_size, pad_mode=pad_mode, bias=bias)
        )
        if norm_layer is not None:
            conv_layers.append(norm_layer(self.dim_hidden))
        conv_layers.append(act_layer())

        # Hidden layers
        assert num_layers >= 2
        for _ in range(num_layers - 2):
            conv_layers.append(
                CausalConv3d(
                    self.dim_hidden, self.dim_hidden, kernel_size, pad_mode=pad_mode, bias=bias
                )
            )
            conv_layers.append(act_layer())

        self.conv_net = nn.Sequential(*conv_layers)

        # Output projection
        # TODO: for proper muP, this needs to be MuReadout
        self.out_proj = CausalConv3d(
            self.dim_hidden, self.dim_out, kernel_size, pad_mode=pad_mode, bias=bias
        )

        # Weight init
        self.weight_init_style = weight_init_style
        self.enable_mup_init = enable_mup_init
        self.init_weights_sp()

    def init_weights_sp(self):
        """SP weight initialization scheme"""
        for name, m in self.named_modules():
            # Conv
            if isinstance(m, nn.Conv3d):
                # Weight
                if "out_proj" in name:
                    nn.init.constant_(m.weight, 0)
                elif self.weight_init_style == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_weights_muP(self):
        """μP weight initialization scheme."""
        if not self.enable_mup_init:
            return

        for name, m in self.named_modules():
            # Conv
            if isinstance(m, nn.Conv3d):
                # Weight
                if "out_proj" in name:
                    nn.init.constant_(m.weight, 0)
                elif self.weight_init_style == "xavier":
                    mup.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    mup.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x_list = data_dict[self.read_key]

        x = torch.cat(x_list, dim=0)
        x = einops.rearrange(x, "b t h w c -> b c t h w")

        with get_autocast_context(x, self.dtype_override):
            x = self.out_proj(self.conv_net(x))

        x = einops.rearrange(x, "b c t h w -> b t h w c")

        data_dict[self.write_key] = x.split(1, dim=0)

        return data_dict


class TransformerHead(nn.Module):
    """
    Transformer head module.
    Compatible with μP, i.e. using MuReadout if use_mup_readout=True.

    Args:
        read_key: Key to read input tensor from the input dictionary.
        write_key: Key to write output tensor into the output dictionary.
        dim: Internal transformer dimension size. A linear projection is always applied from dim_in to dim.
        dim_out: Output dimension size.
        depth: Number of Transformer blocks in the model.
        head_dim: Dimension of each attention head.
        use_mup_readout: Whether to use μP-compatible readout layer (MuReadout) instead of a standard linear layer.
        mlp_ratio: Ratio of the hidden dimension size to the input dimension size in the MLP layers.
        qkv_bias: Whether to use bias in the Q, K, V projections of the attention layers.
        proj_bias: Whether to use bias in the projection layers of the attention.
        mlp_bias: Whether to use bias in the MLP layers.
        drop: Dropout rate applied to attention and MLP layers.
        drop_path_rate: Dropout rate for stochastic depth (drop path).
        act_layer: Activation layer used in the MLPs.
        norm_layer: Normalization layer used before attention and MLP layers.
        weight_init_style: Style of weight initialization ('xavier', 'trunc_normal').
        dtype_override: Optional string to override the tensor data type.
    """

    def __init__(
        self,
        read_key: str,
        write_key: str,
        dim: int,
        dim_out: int,
        depth: int,
        head_dim: int = 64,
        use_mup_readout: bool = False,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        mlp_bias: bool = False,
        drop: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: nn.Module = nn.SiLU,
        norm_layer: Optional[Union[partial, nn.Module]] = partial(
            Fp32LayerNorm, bias=False, elementwise_affine=False
        ),
        weight_init_style: str = "xavier",
        dtype_override: Optional[str] = None,
        block_mask_read_key: Optional[str] = None,
    ):
        super().__init__()
        self.read_key = read_key
        self.write_key = write_key
        self.dim = dim
        self.dim_out = dim_out
        self.dtype_override = str_to_dtype(dtype_override)

        # Internal keys for transformer
        self._internal_input_key = f"{read_key}_internal_input"
        self._internal_output_key = f"{read_key}_internal_output"

        # Create the transformer
        self.transformer = FlexTransformer(
            input_seq_read_key=self._internal_input_key,
            output_seq_write_key=self._internal_output_key,
            dim=dim,
            depth=depth,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            mlp_bias=mlp_bias,
            drop=drop,
            drop_path_rate=drop_path_rate,
            act_layer=act_layer,
            norm_layer=norm_layer,
            gated_mlp=True,
            qk_norm=True,
            weight_init_style=weight_init_style,
            block_mask_read_key=block_mask_read_key,
        )

        # Output projection head. Using custom layer for μP.
        out_proj_layer_fn = MuReadoutFSDP if use_mup_readout else nn.Linear
        self.out_proj = out_proj_layer_fn(dim, dim_out, bias=proj_bias)

        # Weight initialization
        self.weight_init_style = weight_init_style
        self.init_weights_sp()

    def init_weights_sp(self):
        """SP weight initialization scheme"""
        for name, m in self.named_modules():
            # Skip transformer sub-modules
            if "transformer" in name:
                continue

            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if "out_proj" in name:
                    nn.init.constant_(m.weight, 0)
                elif self.weight_init_style == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize transformer weights
        self.transformer.init_weights_sp()

    def init_weights_muP(self):
        """μP weight initialization scheme."""
        for name, m in self.named_modules():
            # Skip transformer sub-modules
            if "transformer" in name:
                continue
            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if "out_proj" in name:
                    nn.init.constant_(m.weight, 0)
                elif self.weight_init_style == "xavier":
                    mup.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    mup.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize transformer weights
        self.transformer.init_weights_muP()

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x_list = data_dict[self.read_key]

        x = torch.cat(x_list, dim=0)
        B, T, H, W, C = x.shape
        x = einops.rearrange(x, "b t h w c -> b (t h w) c")

        with get_autocast_context(x, self.dtype_override):
            # Create internal dict for transformer
            internal_dict = {self._internal_input_key: x}

            if self.transformer.block_mask_read_key is not None:
                internal_dict[self.transformer.block_mask_read_key] = data_dict[
                    self.transformer.block_mask_read_key
                ]

            # Run transformer
            internal_dict = self.transformer(internal_dict)

            # Get transformer output
            transformer_output = internal_dict[self._internal_output_key]

            # Apply output projection
            x = self.out_proj(transformer_output)

        x = einops.rearrange(x, "b (t h w) c -> b t h w c", t=T, h=H, w=W)

        data_dict[self.write_key] = x.split(1, dim=0)

        return data_dict
