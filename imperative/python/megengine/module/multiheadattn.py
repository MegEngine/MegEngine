from typing import Optional

import numpy as np

import megengine as mge
import megengine.functional as F
from megengine import Parameter

from ..device import get_cudnn_version, is_cuda_available
from ..functional.nn import multi_head_attention
from ..tensor import Tensor
from .init import ones_, xavier_uniform_, zeros_
from .module import Module


class MultiHeadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{MultiHeadAttn}\big(q,K,V, W_Q, W_V, W_O\big) = \sum^{nHeads-1}_{i=0}W_{O,i}h_i

    where :math:`h_i=W_{V,i}V \text{Softmax}\Big( \text{smScaler} \cdot K^TW^T_{K,i}W_{Q,i}q \Big),\text{for }i\text{ = 0 ... nHeads-1}`.

    Note: This API is experimental, and there is a possibility of subsequent changes.

    The implementation of cudnn can be run if the following conditions are met:

    - cuda is available, cudnn is available, and the version of cudnn is greater than or equal to 8.0.4.
    - ``bias`` is ``False``, when ``training`` is ``False`` and ``cudnn version`` greater than or equal to 8.0.4. if the ``cudnn version`` greater than or equal to 8.6.0, this point can be ignored.
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``need_weights`` is ``False``
    - ``average_attn_weights`` is ``False``
    - ``maybe_cudnn_style_mask`` is ``True``
    - ``attn_mask`` and ``key_padding_mask`` is cudnn style mask, i.e. the shape of the attn_mask is :math:`(2, L)`, and the shape of the key_padding_mask is :math:`(2, N)`.
      - The shape of attn_mask is :math:`(2, L)`, where :math:`(0, :)` elements specify the start index, :math:`(1, :)` elements specify the end index, the start index is inclusive, and the end index is not exclusive. The start index (i.e. elements in `attn_mask[0, x]`) must be less than the corresponding end index (i.e. elements in `attn_mask[1, x]`). The end index must be less than or equal to :math:`S`, where :math:`S` is the source sequence length, :math:`L` is the target sequence length.
      - The shape of key_padding_mask is :math:`(2, N)`, where :math:`(0, :)` elements specify the target sequence padding in cudnn style mask and the element must equal to or less than :math:`L`, :math:`(1, :)` elements specify the source sequence padding in cudnn style mask and the element must equal to or less than :math:`S`, where :math:`S` is the source sequence length, :math:`L` is the target sequence length.
      Note: If there is no mask or the default mask is used, cudnn impl will also be used. At this time, cudnn will automatically generate the corresponding cudnn style mask.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        attn_dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        out_dropout: Dropout probability on ``output``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at sequence dim. Default: ``False``.
            Different from kbias and vbias,  bias_kv here is not kbias and vbias in the linear layer, and bias_kv here will be added to the K and V at sequence dimensions, where K and V are the matrices of key and value after projection, and K and V will be used to calculate the attention matrix.
            Note: Should be set to False, and configuration of this parameter is not supported now. The reason is that there is only cudnn implementation now, and we may try to loosen this option after submitting the commit that adds MHA proxy implementation.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences.
            Default: ``False``.
            Note: Should be set to False, and configuration of this parameter is not supported now. The reason is that there is only cudnn implementation now, and we may try to loosen this option after submitting the commit that adds MHA proxy implementation.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).

    Examples::
        >>> import numpy as np
        >>> batch_size, seq_len, embed_dim, num_heads = 2, 4, 4, 2
        >>> x = Tensor(np.arange(batch_size * seq_len * embed_dim).astype(np.float32).reshape(batch_size, seq_len, embed_dim))
        >>> multihead_attn = M.MultiHeadAttention(embed_dim, num_heads)
        >>> if is_cuda_available() and get_cudnn_version() >= 8004:
        ...     out = multihead_attn(x, x, x)[0]
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
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn

        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.out_dropout = out_dropout
        self.head_dim = embed_dim // num_heads
        self.unsupport_reason = " The reason is that there is only cudnn implementation now, and we may try to loosen this option after submitting the commit that adds MHA proxy implementation."
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        assert add_bias_kv == False, (
            "add_bias_kv should be set to False, and configuration of this parameter is not supported now."
            + self.unsupport_reason
        )
        assert add_zero_attn == False, (
            "add_zero_attn should be set to False, and configuration of this parameter is not supported now."
            + self.unsupport_reason
        )

        self.bias = bias
        self.weight_bias_len = (
            self.embed_dim + self.kdim + self.vdim + self.embed_dim
        ) * self.embed_dim + (4 * self.embed_dim if self.bias else 0)

        self.io_weight_bias = Parameter(
            np.empty((1, self.weight_bias_len), dtype="float32")
        )
        self.bias_k = (
            Parameter(np.empty((1, 1, embed_dim), dtype="float32"))
            if self.add_bias_kv
            else None
        )
        self.bias_v = (
            Parameter(np.empty((1, 1, embed_dim), dtype="float32"))
            if self.add_bias_kv
            else None
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.attn_dropout = 0.0
        self.out_dropout = 0.0
        xavier_uniform_(self.io_weight_bias)
        if self.bias:
            weight_len = (
                self.embed_dim + self.kdim + self.vdim + self.embed_dim
            ) * self.embed_dim
            self.io_weight_bias[0, weight_len:,] = 0

        if self.add_bias_kv:
            xavier_uniform_(self.bias_k)
            xavier_uniform_(self.bias_v)
        else:
            self.bias_k = None
            self.bias_v = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
        is_causal: bool = False,
        maybe_cudnn_style_mask: bool = False,
    ):
        r"""
        Args:
            query: Query embeddings of shape :math:`(N, L, E_q)`,
                where :math:`N` is the batch size, :math:`L` is the target sequence length, and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against key-value pairs to produce the output. See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(N, S, E_k)`,
                where :math:`N` is the batch size, :math:`S` is the source sequence length, and :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(N, S, E_v)`,
                where :math:`N` is the batch size, :math:`S` is the source sequence length, and :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key`` to ignore for the purpose of
                attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`. Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
            need_weights: indicates whether to return the attention weight, which is the output result of softmax. Default: `False`
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
                heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
                effect when ``need_weights=True``. Default: ``False`` (i.e. not average weights across heads)
            is_causal: If specified, applies a causal mask as attention mask. Default: ``False``
                Warning: ``is_causal`` provides a hint that ``attn_mask`` is the causal mask. Providing incorrect hints can result in incorrect execution, including forward and backward compatibility.
            maybe_cudnn_style_mask: if specified, applies a cudnn style mask as attention mask. Default: ``False``
                Note: In the cudnn style, the shape of the attn_mask is :math:`(2, L)`, and the shape of the key_padding_mask is :math:`(2, N)`.
                Warning: like is_causal, maybe_cudnn_style_mask provides a hint that attn_mask and key_padding_mask is a cudnn style mask. Providing incorrect hints can result in incorrect execution, including forward and backward compatibility. In addition, if the ``_merge_masks`` function returns ``merge_type=cudnn_style_mask``, please ensure that other conditions are correct so that it can run the implementation of cudnn, otherwise an error will be reported.

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(N, L, E)`,
              where :math:`L` is the target sequence length, :math:`N` is
              the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
            - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N * \text{num\_heads}, L, S)`.
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
            qproj_size=self.embed_dim,
            kproj_size=self.embed_dim,
            vproj_size=self.embed_dim,
            oproj_size=self.embed_dim,
            qbias=self.bias,
            kbias=self.bias,
            vbias=self.bias,
            obias=self.bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
            maybe_cudnn_style_mask=maybe_cudnn_style_mask,
        )

    def _module_info_string(self) -> str:
        s = "embed_dim={embed_dim}, num_heads={num_heads}, dropout={dropout}, bias={bias}, kdim={kdim}, vdim={vdim}"
        return s.format(**self.__dict__)
