from typing import Optional

import numpy as np

import megengine as mge
import megengine.functional as F
from megengine import Parameter

from ..device import get_cudnn_version, is_cuda_available
from ..functional.nn import multi_head_attention
from ..tensor import Tensor
from .init import ones_, zeros_
from .module import Module


class MultiHeadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{MultiHeadAttn}\big(q,K,V, W_Q, W_V, W_O\big) = \sum^{nHeads-1}_{i=0}W_{O,i}h_i

    where :math:`h_i=W_{V,i}V \text{Softmax}\Big( \text{smScaler} \cdot K^TW^T_{K,i}W_{Q,i}q \Big),\text{for }i\text{ = 0 ... nHeads-1}`.
    
    Note: This API is experimental, and there is a possibility of subsequent changes. Currently, only the cuda platform is supported, and if the cudnn version >=8.6.0, the calculation results are completely correct; If the cudnn version >=8.0.4 but <8.6.0, if there is a bias, only the dbias result calculated from the backward is incorrect. If there is no bias, the forward and backward calculations are correct; If the cudnn version is less than 8.0.4, this operator is not supported.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        enable_qproj: enable query weight projection. Default: ``True``.
        enable_kproj: enable key weight projection. Default: ``True``.
        enable_vproj: enable value weight projection. Default: ``True``.
        enable_oproj: enable output weight projection. Default: ``True``.

    Examples::
        >>> import numpy as np
        >>> batch_size, seq_len, embed_dim, num_heads = 2, 4, 4, 2
        >>> x = Tensor(np.arange(batch_size * seq_len * embed_dim).astype(np.float32).reshape(batch_size, seq_len, embed_dim))
        >>> multihead_attn = M.MultiHeadAttention(embed_dim, num_heads)
        >>> if is_cuda_available() and get_cudnn_version() >= 8004:
        ...     out = multihead_attn(x, x, x)
        ...     out.numpy().shape
        ... else:
        ...     print(np.zeros((2,4,4)).shape)
        (2, 4, 4)
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        attn_dropout=0.0,
        out_dropout=0.0,
        kdim=None,
        vdim=None,
        bias=True,
        enable_qproj=True,
        enable_kproj=True,
        enable_vproj=True,
        enable_oproj=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.out_dropout = out_dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        assert (
            self._qkv_same_embed_dim
        ), "it does not support the case where q, k, and v are different."
        self.bias = bias

        self.enable_qproj = enable_qproj
        self.enable_kproj = enable_kproj
        self.enable_vproj = enable_vproj
        self.enable_oproj = enable_oproj
        self.nproj = enable_qproj + enable_kproj + enable_vproj + enable_oproj

        if self.bias:
            io_weight = np.ones((embed_dim, self.nproj * embed_dim))
            io_bias = np.zeros((1, self.nproj * embed_dim))
            self.io_weight_bias = Parameter(
                np.concatenate((io_weight, io_bias), axis=0), dtype="float32"
            )
        else:
            self.io_weight_bias = Parameter(
                np.ones((self.nproj * embed_dim, embed_dim), dtype="float32")
            )
        self.reset_parameters()

    def reset_parameters(self):
        self.attn_dropout = 0.0
        self.out_dropout = 0.0
        if self.bias:
            io_weight = np.ones((self.embed_dim, self.nproj * self.embed_dim))
            io_bias = np.zeros((1, self.nproj * self.embed_dim))
            self.io_weight_bias._reset(np.concatenate((io_weight, io_bias), axis=0))
        else:
            ones_(self.io_weight_bias)

    def forward(
        self, query, key, value, attn_mask: bool = True,
    ):
        r"""
    Args:
        query: Query embeddings of shape :math:`(N, L, E_q)`, where :math:`N` is the batch size, :math:`L` is the target sequence length,
            and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
            key-value pairs to produce the output. See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(N, S, E_k)`, where :math:`N` is the batch size, :math:`S` is the source sequence length, and
            :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(N, S, E_v)`, where :math:`N` is the batch size, :math:`S` is the source sequence length, and
            :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(N, L, E)`, 
          where :math:`L` is the target sequence length, :math:`N` is
          the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
        """

        return multi_head_attention(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.attn_dropout,
            self.out_dropout,
            self.io_weight_bias,
            self.bias,
            training=self.training,
            attn_mask=attn_mask,
            enable_qproj=self.enable_qproj,
            enable_kproj=self.enable_kproj,
            enable_vproj=self.enable_vproj,
            enable_oproj=self.enable_oproj,
        )

    def _module_info_string(self) -> str:
        s = "embed_dim={embed_dim}, num_heads={num_heads}, dropout={dropout}, bias={bias}, kdim={kdim}, vdim={vdim}"
        return s.format(**self.__dict__)
