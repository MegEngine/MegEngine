# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np

from ..functional.nn import embedding as embedding_func
from ..tensor import Parameter
from . import init
from .module import Module


class Embedding(Module):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding word embeddings.
    The indices should less than num_embeddings.

    Args:
        num_embeddings: size of embedding dictionary.
        embedding_dim: size of each embedding vector.
        padding_idx: should be set to None, not supportted now.
        max_norm: should be set to None, not supportted now.
        norm_type: should be set to None, not supportted now.
        initial_weight: the learnable weights of the module of shape (num_embeddings, embedding_dim).

    Examples:
        >>> import numpy as np
        >>> weight = mge.tensor(np.array([(1.2,2.3,3.4,4.5,5.6)], dtype=np.float32))
        >>> data = mge.tensor(np.array([(0,0)], dtype=np.int32))
        >>> embedding = M.Embedding(1, 5, initial_weight=weight)
        >>> output = embedding(data)
        >>> with np.printoptions(precision=6):
        ...     print(output.numpy())
        [[[1.2 2.3 3.4 4.5 5.6]
          [1.2 2.3 3.4 4.5 5.6]]]
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: Optional[float] = None,
        initial_weight: Parameter = None,
        freeze: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if padding_idx is not None:
            raise ValueError("Not support padding index now.")
        if max_norm is not None or norm_type is not None:
            raise ValueError("Not support weight normalize now.")
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.freeze = freeze
        if initial_weight is None:
            self.weight = Parameter(
                np.random.uniform(
                    size=(self.num_embeddings, self.embedding_dim)
                ).astype(np.float32)
            )
            self.reset_parameters()
        else:
            if initial_weight.numpy().shape != (num_embeddings, embedding_dim):
                raise ValueError(
                    "The weight shape should match num_embeddings and embedding_dim"
                )
            self.weight = Parameter(initial_weight.numpy())

    def reset_parameters(self) -> None:
        init.normal_(self.weight)

    def forward(self, inputs):
        if self.freeze:
            weight = self.weight.detach()
        else:
            weight = self.weight
        return embedding_func(inputs, weight)

    @classmethod
    def from_pretrained(
        cls,
        embeddings: Parameter,
        freeze: Optional[bool] = True,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: Optional[float] = None,
    ):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings: tensor contained weight for the embedding.
            freeze: if ``True``, the weight does not get updated during the learning process. Default: True.
            padding_idx: should be set to None, not support Now.
            max_norm: should be set to None, not support Now.
            norm_type: should be set to None, not support Now.

        Examples:
            >>> import numpy as np
            >>> weight = mge.tensor(np.array([(1.2,2.3,3.4,4.5,5.6)], dtype=np.float32))
            >>> data = mge.tensor(np.array([(0,0)], dtype=np.int32))
            >>> embedding = M.Embedding.from_pretrained(weight, freeze=False)
            >>> output = embedding(data)
            >>> output.numpy()
            array([[[1.2, 2.3, 3.4, 4.5, 5.6],
                    [1.2, 2.3, 3.4, 4.5, 5.6]]], dtype=float32)
        """
        embeddings_shape = embeddings.shape
        embeddings_dim = len(embeddings_shape)
        if embeddings_dim != 2:
            raise ValueError("Embeddings parameter is expected to be 2-dimensional")
        rows = embeddings_shape[0]
        cols = embeddings_shape[1]
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            initial_weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            freeze=freeze,
        )
        return embedding
