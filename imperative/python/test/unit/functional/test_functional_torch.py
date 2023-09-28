import functools
import itertools
import platform
from functools import reduce
from operator import mul
from typing import Sequence, Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional
from PIL import Image
from utils import opr_test

import megengine as mge
import megengine.distributed as dist
import megengine.functional as F
from megengine import Tensor, is_cuda_available, tensor
from megengine.autodiff import GradManager
from megengine.core._imperative_rt.ops import get_global_rng_seed, set_global_rng_seed
from megengine.core.tensor.utils import subgraph_fn
from megengine.device import get_cudnn_version

# windows torchvision has memeory issue at CUDA env
# so strip torchvision case when cuda available at windows env
if not (platform.system() == "Windows" and is_cuda_available()):
    import torchvision.ops

mge.config.async_level = 0

@pytest.mark.parametrize(
    "batchSize, seqLen, embed_dim, num_heads, attn_p, out_p",
    [
        (1, 20, 10, 5, 0.0, 0.0),
        (1, 20, 10, 10, 0.0, 0.0),
        (1, 20, 10, 2, 0.0, 0.0),
        (1, 20, 10, 1, 0.0, 0.0),
        (5, 20, 1, 1, 0.0, 0.0),
        (5, 20, 1, 1, 0.0, 0.0),
        (5, 20, 1, 1, 0.0, 0.0),
        (5, 20, 1, 1, 0.0, 0.0),
        (5, 20, 10, 10, 0.0, 0.0),
        (5, 20, 10, 5, 0.0, 0.0),
        (5, 20, 10, 2, 0.0, 0.0),
        (5, 20, 10, 1, 0.0, 0.0),
        (1, 1, 10, 10, 0.0, 0.0),
        (1, 1, 10, 5, 0.0, 0.0),
        (1, 1, 10, 2, 0.0, 0.0),
        (1, 1, 10, 1, 0.0, 0.0),
        (10, 1, 10, 10, 0.0, 0.0),
        (10, 1, 10, 5, 0.0, 0.0),
        (10, 1, 10, 2, 0.0, 0.0),
        (10, 1, 10, 1, 0.0, 0.0),
        (10, 10, 1, 1, 0.0, 0.0),
        (10, 1, 1, 1, 0.0, 0.0),
        (1, 20, 1, 1, 0.0, 0.0),
        (1, 10, 1, 1, 0.0, 0.0),
        (10, 100, 1000, 1, 0.0, 0.0),
    ],
)
def test_multiheadattention_with_param(
    batchSize, seqLen, embed_dim, num_heads, attn_p, out_p
):
    swh = [False, True]
    for is_training, with_bias, with_attn_mask in [
        (i, j, k) for i in swh for j in swh for k in swh
    ]:
        np.random.seed(
            batchSize * seqLen * embed_dim * num_heads
            + int(is_training)
            + int(with_bias)
            + int(with_attn_mask)
        )
        shape = [batchSize, seqLen, embed_dim]
        q_np = np.random.rand(*shape).astype("float32")
        dy_np = np.random.rand(*shape).astype("float32")
        mask_np = np.triu(-float("inf") * np.ones((seqLen, seqLen)), k=1).astype(
            "float32"
        )

        # numpy: q/k/v/o weight
        qkvoweight_np = np.random.rand(embed_dim * embed_dim * 4).astype("float32")
        qweight = qkvoweight_np[embed_dim * embed_dim * 0 : embed_dim * embed_dim * 1]
        kweight = qkvoweight_np[embed_dim * embed_dim * 1 : embed_dim * embed_dim * 2]
        vweight = qkvoweight_np[embed_dim * embed_dim * 2 : embed_dim * embed_dim * 3]
        oweight = qkvoweight_np[embed_dim * embed_dim * 3 : embed_dim * embed_dim * 4]
        qweight = qweight.reshape(embed_dim, embed_dim)
        kweight = kweight.reshape(embed_dim, embed_dim)
        vweight = vweight.reshape(embed_dim, embed_dim)
        qkvweight_np = np.concatenate((qweight, kweight, vweight), -1)
        oweight_np = oweight.reshape(embed_dim, embed_dim)
        # numpy: q/k/v/o bias
        if with_bias:
            qkvobias_np = np.random.rand(embed_dim * 4).astype("float32")
            qbias = qkvobias_np[embed_dim * 0 : embed_dim * 1]
            kbias = qkvobias_np[embed_dim * 1 : embed_dim * 2]
            vbias = qkvobias_np[embed_dim * 2 : embed_dim * 3]
            qkvbias_np = np.concatenate((qbias, kbias, vbias))
            obias_np = qkvobias_np[embed_dim * 3 : embed_dim * 4]

        # mge: q/k/v/o shape = [batch, seqLen, vsize]
        q_mge = Tensor(q_np)
        k_mge = Tensor(q_np)
        v_mge = Tensor(q_np)
        if with_bias:
            qkvoweight_bias_mge = Tensor(
                np.concatenate((qkvoweight_np, qkvobias_np), -1)
            )
        else:
            qkvoweight_bias_mge = Tensor(qkvoweight_np)
        mask_mge = Tensor(mask_np)
        dy_mge = Tensor(dy_np)

        # torch: q/k/v/o shape = [seqLen, batch, vsize]
        q_np = q_np.transpose(1, 0, 2)
        dy_np = dy_np.transpose(1, 0, 2)
        q_torch = torch.tensor(q_np, requires_grad=True)
        k_torch = torch.tensor(q_np, requires_grad=True)
        v_torch = torch.tensor(q_np, requires_grad=True)
        qkvweight_torch = torch.tensor(qkvweight_np.T, requires_grad=True)
        oweight_torch = torch.tensor(oweight_np.T, requires_grad=True)
        if with_bias:
            qkvbias_torch = torch.tensor(qkvbias_np, requires_grad=True)
            obias_torch = torch.tensor(obias_np, requires_grad=True)
        else:
            qkvbias_torch = None
            obias_torch = None
        mask_torch = torch.tensor(mask_np)
        dy_torch = torch.tensor(dy_np)

        # mge forward/backward
        gm = GradManager().attach([q_mge, k_mge, v_mge, qkvoweight_bias_mge])
        with gm:
            y_mge = F.nn.multi_head_attention(
                q_mge,
                k_mge,
                v_mge,
                embed_dim,
                num_heads,
                attn_drop=attn_p,
                out_drop=out_p,
                io_weight_bias=qkvoweight_bias_mge,
                attn_mask=mask_mge if with_attn_mask else None,
                is_causal=with_attn_mask,
                qproj_size=embed_dim,
                kproj_size=embed_dim,
                vproj_size=embed_dim,
                oproj_size=embed_dim,
                qbias=with_bias,
                kbias=with_bias,
                vbias=with_bias,
                obias=with_bias,
                training=is_training,
                add_zero_attn=False,
                add_bias_kv=False,
            )[0]
            if is_training:
                gm.backward(y_mge, dy_mge)

        # torch forward/backward
        y_torch = torch.nn.functional.multi_head_attention_forward(
            q_torch,
            k_torch,
            v_torch,
            embed_dim_to_check=embed_dim,
            num_heads=num_heads,
            in_proj_weight=qkvweight_torch,
            in_proj_bias=qkvbias_torch,
            bias_k=None,
            bias_v=None,
            training=is_training,
            add_zero_attn=False,
            out_proj_bias=obias_torch,
            attn_mask=mask_torch if with_attn_mask else None,
            dropout_p=attn_p,
            out_proj_weight=oweight_torch,
            need_weights=False,
        )[0]
        if is_training:
            y_torch.backward(dy_torch)

        atol = 1e-4
        if seqLen * embed_dim >= 100:
            atol = 1e-3

        # out
        y_torch = y_torch.detach().numpy().transpose(1, 0, 2)
        np.testing.assert_allclose(y_torch, y_mge.numpy(), atol, atol)

        if is_training:
            # dq dk dv
            dq_torch = q_torch.grad.detach().numpy().transpose(1, 0, 2)
            dk_torch = k_torch.grad.detach().numpy().transpose(1, 0, 2)
            dv_torch = v_torch.grad.detach().numpy().transpose(1, 0, 2)
            np.testing.assert_allclose(dq_torch, q_mge.grad.numpy(), atol, atol)
            np.testing.assert_allclose(dk_torch, k_mge.grad.numpy(), atol, atol)
            np.testing.assert_allclose(dv_torch, v_mge.grad.numpy(), atol, atol)

            qkvodweight_bias_mge = qkvoweight_bias_mge.grad.detach().numpy()
            # dweight
            qkvodweight_mge = qkvodweight_bias_mge[0 : embed_dim * embed_dim * 4]
            wq, wk, wv, wo = np.split(qkvodweight_mge, 4)
            wq = wq.reshape(embed_dim, embed_dim).T
            wk = wk.reshape(embed_dim, embed_dim).T
            wv = wv.reshape(embed_dim, embed_dim).T
            qkvdweight_mge = np.concatenate((wq, wk, wv))
            odweight_mge = wo.reshape(embed_dim, embed_dim).T
            np.testing.assert_allclose(
                qkvweight_torch.grad.detach().numpy(), qkvdweight_mge, atol, atol,
            )
            np.testing.assert_allclose(
                oweight_torch.grad.detach().numpy(), odweight_mge, atol, atol
            )
            # dbias
            if with_bias:
                qkvodbias_mge = qkvodweight_bias_mge[
                    embed_dim * embed_dim * 4 :,
                ]
                qkvdbias_mge = qkvodbias_mge[0 : embed_dim * 3]
                odbias_mge = qkvodbias_mge[
                    embed_dim * 3 :,
                ]
                np.testing.assert_allclose(
                    qkvbias_torch.grad.detach().numpy(), qkvdbias_mge, atol, atol,
                )
                np.testing.assert_allclose(
                    obias_torch.grad.detach().numpy(), odbias_mge, atol, atol,
                )

        if is_training:
            # drop seed
            attn_drop_tmp = np.random.uniform(0, 1, 1)[0]
            out_drop_tmp = np.random.uniform(0, 1, 1)[0]

            def lambda_function():
                return F.nn.multi_head_attention(
                    q_mge,
                    k_mge,
                    v_mge,
                    embed_dim,
                    num_heads,
                    attn_drop=attn_drop_tmp,
                    out_drop=out_drop_tmp,
                    io_weight_bias=qkvoweight_bias_mge,
                    attn_mask=mask_mge if with_attn_mask else None,
                    is_causal=with_attn_mask,
                    qproj_size=embed_dim,
                    kproj_size=embed_dim,
                    vproj_size=embed_dim,
                    oproj_size=embed_dim,
                    qbias=with_bias,
                    kbias=with_bias,
                    vbias=with_bias,
                    obias=with_bias,
                    training=is_training,
                )[0]

            if seqLen > 10 and seqLen < 100:
                set_global_rng_seed(111)
                out1 = lambda_function()
                out2 = lambda_function()
                out3 = lambda_function()
                assert not (out1.numpy() == out2.numpy()).all()
                assert not (out1.numpy() == out3.numpy()).all()
                assert not (out2.numpy() == out3.numpy()).all()

                set_global_rng_seed(111)
                out4 = lambda_function()
                assert (out1.numpy() == out4.numpy()).all()

                set_global_rng_seed(111)
                out5 = lambda_function()
                assert (out1.numpy() == out5.numpy()).all()
                assert (out4.numpy() == out5.numpy()).all()

                set_global_rng_seed(222)
                out6 = lambda_function()
                assert not (out1.numpy() == out6.numpy()).all()


@pytest.mark.parametrize(
    "batch_size, seq_qlen, seq_klen, embed_dim, kdim, vdim, num_heads",
    [
        (1, 1, 1, 1, 1, 1, 1),
        (1, 1, 1, 4, 1, 1, 1),
        (1, 1, 1, 4, 3, 1, 1),
        (1, 1, 1, 4, 3, 2, 1),
        (1, 1, 1, 4, 4, 4, 1),
        (1, 1, 1, 4, 4, 4, 2),
        (1, 1, 1, 4, 4, 4, 4),
        (1, 1, 6, 6, 6, 4, 1),
        (1, 1, 6, 6, 6, 4, 2),
        (1, 6, 1, 6, 6, 4, 1),
        (1, 6, 1, 6, 6, 4, 2),
        (1, 6, 6, 6, 6, 4, 1),
        (1, 6, 6, 6, 6, 4, 2),
        (2, 6, 6, 8, 6, 4, 2),
        (10, 20, 15, 8, 6, 4, 1),
        (10, 20, 1, 8, 6, 4, 2),
        (10, 1, 15, 8, 6, 4, 2),
        (10, 20, 15, 8, 8, 8, 1),
        (10, 20, 15, 8, 8, 8, 2),
        (10, 20, 15, 8, 8, 8, 4),
        (10, 20, 15, 8, 8, 8, 8),
    ],
)
def test_multiheadattention(
    batch_size, seq_qlen, seq_klen, embed_dim, kdim, vdim, num_heads
):
    class MHATest:
        def __init__(
            self,
            batch_size,
            seq_qlen,
            seq_klen,
            embed_dim,
            kdim,
            vdim,
            num_heads,
            qproj_size,
            kproj_size,
            vproj_size,
            oproj_size,
            qbias,
            kbias,
            vbias,
            obias,
        ):
            self.batch_size = batch_size
            self.seq_qlen = seq_qlen
            self.seq_klen = seq_klen
            self.embed_dim = embed_dim
            self.embed_dim = (
                embed_dim if qproj_size > 0 else self.embed_dim // num_heads
            )
            self.kdim = kdim if kdim is not None else embed_dim
            self.kdim = kdim if kproj_size > 0 else self.kdim // num_heads
            self.vdim = vdim if vdim is not None else embed_dim
            self.num_heads = num_heads
            self.qproj_size = qproj_size
            self.kproj_size = kproj_size
            self.vproj_size = vproj_size
            self.oproj_size = oproj_size
            self.qbias = qproj_size and qbias
            self.kbias = kproj_size and kbias
            self.vbias = vproj_size and vbias
            self.obias = oproj_size and obias

            self.seed = (
                batch_size * seq_qlen * embed_dim * num_heads
                + seq_klen
                + kdim
                + vdim
                + qproj_size
                + kproj_size
                + vproj_size
                + oproj_size
            )
            self._gen_numpy_data()

        def _gen_numpy_data(self):
            np.random.seed(self.seed)
            q_shape = [self.batch_size, self.seq_qlen, self.embed_dim]
            k_shape = [self.batch_size, self.seq_klen, self.kdim]
            v_shape = [self.batch_size, self.seq_klen, self.vdim]
            if self.oproj_size != 0:
                o_shape = [self.batch_size, self.seq_qlen, self.oproj_size]
            elif self.vproj_size != 0:
                o_shape = [self.batch_size, self.seq_qlen, self.vproj_size]
            else:
                o_shape = [self.batch_size, self.seq_qlen, self.vdim * self.num_heads]

            qweight_offset = 0
            kweight_offset = qweight_offset + self.embed_dim * self.qproj_size
            vweight_offset = kweight_offset + self.kdim * self.kproj_size
            oweight_offset = vweight_offset + self.vdim * self.vproj_size
            qkvoweight_len = (
                oweight_offset
                + (
                    self.vproj_size
                    if self.vproj_size != 0
                    else self.vdim * self.num_heads
                )
                * self.oproj_size
            )
            qweight_shape = (
                [self.embed_dim, self.qproj_size] if self.qproj_size != 0 else 0
            )
            kweight_shape = [self.kdim, self.kproj_size] if self.kproj_size != 0 else 0
            vweight_shape = [self.vdim, self.vproj_size] if self.vproj_size != 0 else 0
            oweight_shape = (
                [
                    (
                        self.vproj_size
                        if self.vproj_size != 0
                        else self.vdim * self.num_heads
                    ),
                    self.oproj_size,
                ]
                if self.oproj_size != 0
                else 0
            )
            self.qkvoweight_np = np.random.rand(qkvoweight_len).astype("float32")
            qweight = self.qkvoweight_np[qweight_offset:kweight_offset]
            kweight = self.qkvoweight_np[kweight_offset:vweight_offset]
            vweight = self.qkvoweight_np[vweight_offset:oweight_offset]
            oweight = self.qkvoweight_np[oweight_offset:qkvoweight_len]

            qbias_offset = 0
            kbias_offset = qbias_offset + (self.qproj_size if self.qbias else 0)
            vbias_offset = kbias_offset + (self.kproj_size if self.kbias else 0)
            obias_offset = vbias_offset + (self.vproj_size if self.vbias else 0)
            qkvobias_len = obias_offset + (self.oproj_size if self.obias else 0)
            self.qkvobias_np = np.random.rand(qkvobias_len).astype("float32")

            self.qbias_offset = qbias_offset
            self.kbias_offset = kbias_offset
            self.vbias_offset = vbias_offset
            self.obias_offset = obias_offset
            self.qkvobias_len = qkvobias_len

            self.qweight_offset = qweight_offset
            self.kweight_offset = kweight_offset
            self.vweight_offset = vweight_offset
            self.oweight_offset = oweight_offset
            self.qkvoweight_len = qkvoweight_len

            # all numpy data
            self.q_np = np.random.rand(*q_shape).astype("float32")
            self.k_np = np.random.rand(*k_shape).astype("float32")
            self.v_np = np.random.rand(*v_shape).astype("float32")
            self.dy_np = np.random.rand(*o_shape).astype("float32")

            self.qweight_np = (
                qweight.reshape(qweight_shape) if self.qproj_size != 0 else None
            )
            self.kweight_np = (
                kweight.reshape(kweight_shape) if self.kproj_size != 0 else None
            )
            self.vweight_np = (
                vweight.reshape(vweight_shape) if self.vproj_size != 0 else None
            )
            self.oweight_np = (
                oweight.reshape(oweight_shape) if self.oproj_size != 0 else None
            )

            self.qbias_np = self.qkvobias_np[qbias_offset:kbias_offset]
            self.kbias_np = self.qkvobias_np[kbias_offset:vbias_offset]
            self.vbias_np = self.qkvobias_np[vbias_offset:obias_offset]
            self.obias_np = self.qkvobias_np[obias_offset:qkvobias_len]

            self.mask_np = np.triu(
                -float("inf") * np.ones((self.seq_qlen, self.seq_klen)), k=1
            ).astype("float32")

            cudnn_empty_mask = np.zeros((2, seq_qlen)).astype("int")
            cudnn_casual_mask = np.zeros((2, seq_qlen)).astype("int")
            for i in range(self.seq_qlen):
                cudnn_empty_mask[0][i] = 0
                cudnn_empty_mask[1][i] = self.seq_klen
                cudnn_casual_mask[0][i] = 0
                cudnn_casual_mask[1][i] = i + 1
            self.cudnn_casual_mask_np = cudnn_casual_mask
            self.cudnn_empty_mask_np = cudnn_empty_mask

            cudnn_key_padding_mask1 = np.ones(self.batch_size) * seq_qlen
            cudnn_key_padding_mask2 = np.random.randint(
                1, self.seq_klen + 1, [self.batch_size]
            )
            cudnn_key_padding_mask3 = np.ones(self.batch_size) * seq_klen
            self.cudnn_key_padding_mask_np = np.concatenate(
                (cudnn_key_padding_mask1, cudnn_key_padding_mask2)
            ).reshape(2, self.batch_size)
            self.cudnn_empty_key_padding_mask_np = np.concatenate(
                (cudnn_key_padding_mask1, cudnn_key_padding_mask3)
            ).reshape(2, self.batch_size)
            key_padding_mask = np.random.choice(
                [False], (self.batch_size, self.seq_klen)
            )
            for i in range(self.batch_size):
                key_padding_mask[i][cudnn_key_padding_mask2[i] :,] = True
            self.key_padding_mask_np = key_padding_mask

            self.biask_np = np.random.rand(
                *[1, 1, self.kproj_size if self.kproj_size else self.kdim]
            ).astype("float32")
            self.biasv_np = np.random.rand(
                *[1, 1, self.vproj_size if self.vproj_size else self.vdim]
            ).astype("float32")

        def _run_torch(
            self,
            is_training,
            with_bias,
            with_attn_mask,
            with_key_padding_mask=False,
            need_weights=False,
            average_attn_weights=False,
            with_biaskv=False,
            with_zero_attn=False,
        ):
            # q/k/v
            q_np = self.q_np.transpose(1, 0, 2)
            k_np = self.k_np.transpose(1, 0, 2)
            v_np = self.v_np.transpose(1, 0, 2)
            dy_np = self.dy_np.transpose(1, 0, 2)
            q_torch = torch.tensor(q_np, requires_grad=True)
            k_torch = torch.tensor(k_np, requires_grad=True)
            v_torch = torch.tensor(v_np, requires_grad=True)

            # q/k/v/o weight
            qweight_torch = torch.tensor(self.qweight_np.T, requires_grad=True)
            kweight_torch = torch.tensor(self.kweight_np.T, requires_grad=True)
            vweight_torch = torch.tensor(self.vweight_np.T, requires_grad=True)
            oweight_torch = torch.tensor(self.oweight_np.T, requires_grad=True)

            # q/k/v/o bias
            if with_bias and self.qbias and self.kbias and self.vbias:
                qkvbias_np = np.concatenate(
                    (self.qbias_np, self.kbias_np, self.vbias_np)
                )
                qkvbias_torch = torch.tensor(qkvbias_np, requires_grad=True)
            else:
                qkvbias_torch = None
            if with_bias and self.obias:
                obias_torch = torch.tensor(self.obias_np, requires_grad=True)
            else:
                obias_torch = None

            # dy
            dy_torch = torch.tensor(dy_np)
            mask_torch = torch.tensor(self.mask_np) if with_attn_mask else None
            key_padding_mask_torch = (
                torch.tensor(self.key_padding_mask_np)
                if with_key_padding_mask
                else None
            )
            (
                y_torch,
                attn_weight_torch,
            ) = torch.nn.functional.multi_head_attention_forward(
                q_torch,
                k_torch,
                v_torch,
                embed_dim_to_check=self.embed_dim,
                num_heads=self.num_heads,
                in_proj_weight=None,
                in_proj_bias=qkvbias_torch,
                bias_k=None,
                bias_v=None,
                out_proj_weight=oweight_torch,
                out_proj_bias=obias_torch,
                training=is_training,
                add_zero_attn=with_zero_attn,
                key_padding_mask=key_padding_mask_torch,
                attn_mask=mask_torch,
                dropout_p=0.0,
                need_weights=need_weights,
                q_proj_weight=qweight_torch,
                k_proj_weight=kweight_torch,
                v_proj_weight=vweight_torch,
                use_separate_proj_weight=True,
                average_attn_weights=average_attn_weights,
            )
            if is_training:
                y_torch.backward(dy_torch)

            y_torch = y_torch.detach().numpy().transpose(1, 0, 2)
            if need_weights:
                attn_weight_torch = attn_weight_torch.detach().numpy()
                shape = attn_weight_torch.shape
                attn_weight_torch = attn_weight_torch.reshape(-1, shape[-2], shape[-1])
            else:
                attn_weight_torch = 0
            if is_training:
                dq_torch = q_torch.grad.detach().numpy().transpose(1, 0, 2)
                dk_torch = k_torch.grad.detach().numpy().transpose(1, 0, 2)
                dv_torch = v_torch.grad.detach().numpy().transpose(1, 0, 2)

                qweight = qweight_torch.grad.detach().numpy().T.flatten()
                kweight = kweight_torch.grad.detach().numpy().T.flatten()
                vweight = vweight_torch.grad.detach().numpy().T.flatten()
                oweight = oweight_torch.grad.detach().numpy().T.flatten()

                if with_bias:
                    qbias, kbias, vbias, obias = 0, 0, 0, 0
                    if self.qbias and self.kbias and self.vbias:
                        qkvbias = qkvbias_torch.grad.detach().numpy()
                        qbias, kbias, vbias = np.split(qkvbias, 3)
                    if self.obias:
                        obias = obias_torch.grad.detach().numpy()

                    return (
                        y_torch,
                        dq_torch,
                        dk_torch,
                        dv_torch,
                        qweight,
                        kweight,
                        vweight,
                        oweight,
                        qbias,
                        kbias,
                        vbias,
                        obias,
                        attn_weight_torch,
                    )
                else:
                    return (
                        y_torch,
                        dq_torch,
                        dk_torch,
                        dv_torch,
                        qweight,
                        kweight,
                        vweight,
                        oweight,
                        0,
                        0,
                        0,
                        0,
                        attn_weight_torch,
                    )
            else:
                return y_torch, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, attn_weight_torch

        def _run_mge(
            self,
            is_training,
            with_bias,
            with_attn_mask,
            with_key_padding_mask=False,
            need_weights=False,
            average_attn_weights=False,
            maybe_cudnn_style_mask=False,
            with_biaskv=False,
            reslink=False,
            with_zero_attn=False,
        ):
            q_mge = Tensor(self.q_np)
            k_mge = Tensor(self.k_np)
            v_mge = Tensor(self.v_np)

            if with_bias and (self.qbias or self.kbias or self.vbias or self.obias):
                qkvoweight_bias_mge = Tensor(
                    np.concatenate((self.qkvoweight_np, self.qkvobias_np), -1)
                )
            else:
                qkvoweight_bias_mge = Tensor(self.qkvoweight_np)
            dy_mge = Tensor(self.dy_np)

            if with_attn_mask:
                if maybe_cudnn_style_mask:
                    mask_mge = Tensor(self.cudnn_casual_mask_np)
                else:
                    mask_mge = Tensor(self.mask_np)
            else:
                if maybe_cudnn_style_mask:
                    mask_mge = Tensor(self.cudnn_empty_mask_np)
                else:
                    mask_mge = None

            if with_key_padding_mask:
                if maybe_cudnn_style_mask:
                    key_padding_mask_mge = Tensor(self.cudnn_key_padding_mask_np)
                else:
                    key_padding_mask_mge = Tensor(self.key_padding_mask_np)
            else:
                if maybe_cudnn_style_mask:
                    key_padding_mask_mge = Tensor(self.cudnn_empty_key_padding_mask_np)
                else:
                    key_padding_mask_mge = None

            gm = GradManager().attach([q_mge, k_mge, v_mge, qkvoweight_bias_mge])
            with gm:
                y_mge, attn_weight_mge = F.nn.multi_head_attention(
                    q_mge,
                    k_mge,
                    v_mge,
                    self.embed_dim,
                    self.num_heads,
                    attn_drop=0.0,
                    out_drop=0.0,
                    io_weight_bias=qkvoweight_bias_mge,
                    qproj_size=self.qproj_size,
                    kproj_size=self.kproj_size,
                    vproj_size=self.vproj_size,
                    oproj_size=self.oproj_size,
                    qbias=with_bias and self.qbias,
                    kbias=with_bias and self.kbias,
                    vbias=with_bias and self.vbias,
                    obias=with_bias and self.obias,
                    bias_k=None,
                    bias_v=None,
                    attn_mask=mask_mge,
                    key_padding_mask=key_padding_mask_mge,
                    add_zero_attn=with_zero_attn,
                    is_causal=with_attn_mask,
                    training=is_training,
                    need_weights=need_weights,
                    maybe_cudnn_style_mask=maybe_cudnn_style_mask,
                    average_attn_weights=average_attn_weights,
                )
                if is_training:
                    gm.backward(y_mge, dy_mge)

            y_mge = y_mge.numpy()
            if not need_weights:
                attn_weight_mge = 0
            if is_training:
                dq_mge = q_mge.grad.numpy()
                dk_mge = k_mge.grad.numpy()
                dv_mge = v_mge.grad.numpy()
                qoff, koff, voff, ooff, wlen = (
                    self.qweight_offset,
                    self.kweight_offset,
                    self.vweight_offset,
                    self.oweight_offset,
                    self.qkvoweight_len,
                )
                qkvodweight_bias_mge = qkvoweight_bias_mge.grad.detach().numpy()
                qweight = qkvodweight_bias_mge[qoff:koff] if self.qproj_size else 0
                kweight = qkvodweight_bias_mge[koff:voff] if self.kproj_size else 0
                vweight = qkvodweight_bias_mge[voff:ooff] if self.vproj_size else 0
                oweight = qkvodweight_bias_mge[ooff:wlen] if self.oproj_size else 0

                if with_bias and (self.qbias or self.kbias or self.vbias or self.obias):
                    qoff, koff, voff, ooff = (
                        self.qbias_offset,
                        self.kbias_offset,
                        self.vbias_offset,
                        self.obias_offset,
                    )
                    qkvodbias_bias_mge = qkvodweight_bias_mge[
                        wlen:,
                    ]
                    qbias = qkvodbias_bias_mge[qoff:koff] if self.qbias else 0
                    kbias = qkvodbias_bias_mge[koff:voff] if self.kbias else 0
                    vbias = qkvodbias_bias_mge[voff:ooff] if self.vbias else 0
                    obias = qkvodbias_bias_mge[ooff:,] if self.obias else 0
                    return (
                        y_mge,
                        dq_mge,
                        dk_mge,
                        dv_mge,
                        qweight,
                        kweight,
                        vweight,
                        oweight,
                        qbias,
                        kbias,
                        vbias,
                        obias,
                        attn_weight_mge,
                    )
                else:
                    return (
                        y_mge,
                        dq_mge,
                        dk_mge,
                        dv_mge,
                        qweight,
                        kweight,
                        vweight,
                        oweight,
                        0,
                        0,
                        0,
                        0,
                        attn_weight_mge,
                    )
            else:
                return y_mge, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, attn_weight_mge

        def _run_naive(
            self,
            is_training,
            with_bias,
            with_attn_mask,
            need_weights=False,
            average_attn_weights=False,
        ):
            def softmax(Z):
                Z = np.exp(Z - Z.max(axis=-1, keepdims=True))
                return Z / Z.sum(axis=-1, keepdims=True)

            def softmax_backward(Z, dY):
                z_torch = torch.tensor(Z, requires_grad=True)
                sm = torch.nn.Softmax(dim=-1)
                output = sm(z_torch)
                dy_torch = torch.tensor(dY, requires_grad=True)
                output.backward(dy_torch)
                return z_torch.grad.numpy()

            heads, mask, qu, ky, va, wq, wk, wv, wo, bq, bk, bv, bo, grad_out = (
                self.num_heads,
                self.mask_np,
                self.q_np,
                self.k_np,
                self.v_np,
                self.qweight_np,
                self.kweight_np,
                self.vweight_np,
                self.oweight_np,
                self.qbias_np,
                self.kbias_np,
                self.vbias_np,
                self.obias_np,
                self.dy_np,
            )

            Q = (
                qu
                if self.qproj_size == 0
                else (qu @ wq + (bq if with_bias and self.qbias else 0))
            )
            K = (
                ky
                if self.kproj_size == 0
                else (ky @ wk + (bk if with_bias and self.kbias else 0))
            )
            V = (
                va
                if self.vproj_size == 0
                else (va @ wv + (bv if with_bias and self.vbias else 0))
            )

            bs, sql, skl = self.batch_size, self.seq_qlen, self.seq_klen
            d = self.embed_dim if self.qproj_size == 0 else self.embed_dim // heads
            sm = 1 / np.sqrt(d)

            if self.qproj_size:
                NQ = Q.reshape(bs, sql, heads, Q.shape[2] // heads).swapaxes(1, 2)
            else:
                NQ = np.expand_dims(Q, 1).repeat(heads, axis=1)
            if self.kproj_size:
                NK = K.reshape(bs, skl, heads, K.shape[2] // heads).swapaxes(1, 2)
            else:
                NK = np.expand_dims(K, 1).repeat(heads, axis=1)
            if self.vproj_size:
                NV = V.reshape(bs, skl, heads, V.shape[2] // heads).swapaxes(1, 2)
            else:
                NV = np.expand_dims(V, 1).repeat(heads, axis=1)

            NX = (NQ @ NK.swapaxes(-1, -2)) * sm + (mask if with_attn_mask else 0)
            NY = softmax(NX)

            if need_weights:
                if average_attn_weights:
                    attn_weight = NY.mean(axis=1)
                else:
                    shape = NY.shape
                    attn_weight = NY.reshape(-1, shape[-2], shape[-1])
            else:
                attn_weight = 0

            NZ = NY @ NV
            if self.vproj_size == 0 and self.oproj_size == 0:
                out = NZ.swapaxes(1, 2).reshape(bs, sql, -1)
            elif self.vproj_size != 0 and self.oproj_size == 0:
                out = NZ.swapaxes(1, 2).reshape(bs, sql, -1)
            elif self.vproj_size == 0 and self.oproj_size != 0:
                Z = NZ.swapaxes(1, 2).reshape(bs, sql, -1)
                out = Z @ wo + (bo if with_bias and self.obias else 0)
            elif self.vproj_size != 0 and self.oproj_size != 0:
                Z = NZ.swapaxes(1, 2).reshape(bs, sql, -1)
                out = Z @ wo + (bo if with_bias and self.obias else 0)

            grad_z, grad_wo, grad_bo = 0, 0, 0
            grad_qin, grad_wq, grad_bq = 0, 0, 0
            grad_kin, grad_wk, grad_bk = 0, 0, 0
            grad_vin, grad_wv, grad_bv = 0, 0, 0
            if is_training:
                if self.oproj_size > 0:
                    grad_z = grad_out @ wo.swapaxes(-1, -2)
                    grad_wo = Z.swapaxes(-1, -2) @ grad_out
                    grad_wo = np.sum(grad_wo, axis=0)
                    grad_wo = grad_wo.flatten()
                    if with_bias and self.obias:
                        grad_bo = np.sum(grad_out, axis=0, keepdims=False)
                        grad_bo = np.sum(grad_bo, axis=0, keepdims=False)
                        grad_bo = grad_bo.flatten()
                else:
                    grad_z = grad_out

                if self.vproj_size == 0:
                    grad_nz = grad_z.reshape(bs, sql, heads, self.vdim).swapaxes(1, 2)
                elif self.vproj_size != 0:
                    grad_nz = grad_z.reshape(
                        bs, sql, heads, self.vproj_size // heads
                    ).swapaxes(1, 2)

                grad_ny = grad_nz @ NV.swapaxes(-1, -2)
                grad_nv = NY.swapaxes(-1, -2) @ grad_nz

                grad_nx = softmax_backward(NX, grad_ny) * sm
                grad_nq = grad_nx @ NK
                grad_nk = grad_nx.swapaxes(-1, -2) @ NQ

                grad_q = grad_nq.swapaxes(1, 2).reshape(
                    bs, sql, grad_nq.shape[3] * heads
                )
                grad_k = grad_nk.swapaxes(1, 2).reshape(
                    bs, skl, grad_nk.shape[3] * heads
                )
                grad_v = grad_nv.swapaxes(1, 2).reshape(
                    bs, skl, grad_nv.shape[3] * heads
                )

                if self.qproj_size > 0:
                    grad_qin = grad_q @ wq.swapaxes(-1, -2)
                    grad_wq = qu.swapaxes(-1, -2) @ grad_q
                    grad_wq = np.sum(grad_wq, axis=0)
                    grad_wq = grad_wq.flatten()
                    if with_bias and self.qbias:
                        grad_bq = np.sum(grad_q, axis=0, keepdims=False)
                        grad_bq = np.sum(grad_bq, axis=0, keepdims=False)
                else:
                    grad_q = grad_q.reshape(bs, sql, heads, -1)
                    grad_qin = np.sum(grad_q, axis=-2)

                if self.kproj_size > 0:
                    grad_kin = grad_k @ wk.swapaxes(-1, -2)
                    grad_wk = ky.swapaxes(-1, -2) @ grad_k
                    grad_wk = np.sum(grad_wk, axis=0)
                    grad_wk = grad_wk.flatten()
                    if with_bias and self.kbias:
                        grad_bk = np.sum(grad_k, axis=0, keepdims=False)
                        grad_bk = np.sum(grad_bk, axis=0, keepdims=False)
                        grad_bk = grad_bk.flatten()
                else:
                    grad_k = grad_k.reshape(bs, skl, heads, -1)
                    grad_kin = np.sum(grad_k, axis=-2)

                if self.vproj_size > 0:
                    grad_vin = grad_v @ wv.swapaxes(-1, -2)
                    grad_wv = va.swapaxes(-1, -2) @ grad_v
                    grad_wv = np.sum(grad_wv, axis=0)
                    grad_wv = grad_wv.flatten()
                    if with_bias and self.vbias:
                        grad_bv = np.sum(grad_v, axis=0, keepdims=False)
                        grad_bv = np.sum(grad_bv, axis=0, keepdims=False)
                        grad_bv = grad_bv.flatten()
                else:
                    grad_v = grad_v.reshape(bs, skl, heads, -1)
                    grad_vin = np.sum(grad_v, axis=-2)

            return (
                out,
                grad_qin,
                grad_kin,
                grad_vin,
                grad_wq,
                grad_wk,
                grad_wv,
                grad_wo,
                grad_bq,
                grad_bk,
                grad_bv,
                grad_bo,
                attn_weight,
            )

        def _run_mge_vs_torch(
            self,
            maybe_cudnn_style_mask=False,
            need_weights=False,
            average_attn_weights=False,
        ):
            def can_run(maybe_cudnn_style_mask, need_weights):
                if (
                    self.qproj_size == 0
                    or self.kproj_size == 0
                    or self.vproj_size == 0
                    or self.oproj_size == 0
                ):
                    # parameter configuration does not conform to the semantics of pytorch multiheadattention.
                    # which means that all weight(including qweight, kweight, vweight, oweight) must exist.
                    return False
                if maybe_cudnn_style_mask and (
                    need_weights or self.qbias or self.kbias or self.vbias or self.obias
                ):
                    # if need_weights is true and have bias, the cuda proxy implementation will be run,
                    # and the cudnn-style mask cannot be processed at this time.
                    return False
                return True

            if can_run(maybe_cudnn_style_mask, need_weights):
                swh = [False, True]
                for is_training, with_bias, with_attn_mask, with_key_padding_mask in [
                    (i, j, k, m) for i in swh for j in swh for k in swh for m in swh
                ]:
                    atol = 1e-3
                    (
                        ym,
                        dqm,
                        dkm,
                        dvm,
                        dqwm,
                        dkwm,
                        dvwm,
                        dowm,
                        dqbm,
                        dkbm,
                        dvbm,
                        dobm,
                        awm,
                    ) = self._run_mge(
                        is_training,
                        with_bias,
                        with_attn_mask,
                        with_key_padding_mask,
                        maybe_cudnn_style_mask=maybe_cudnn_style_mask,
                        need_weights=need_weights,
                        average_attn_weights=average_attn_weights,
                    )
                    (
                        yt,
                        dqt,
                        dkt,
                        dvt,
                        dqwt,
                        dkwt,
                        dvwt,
                        dowt,
                        dqbt,
                        dkbt,
                        dvbt,
                        dobt,
                        awt,
                    ) = self._run_torch(
                        is_training,
                        with_bias,
                        with_attn_mask,
                        with_key_padding_mask,
                        need_weights=need_weights,
                        average_attn_weights=average_attn_weights,
                    )

                    np.testing.assert_allclose(ym, yt, atol, atol)
                    np.testing.assert_allclose(dqm, dqt, atol, atol)
                    np.testing.assert_allclose(dkm, dkt, atol, atol)
                    np.testing.assert_allclose(dvm, dvt, atol, atol)
                    np.testing.assert_allclose(dqwm, dqwt, atol, atol)
                    np.testing.assert_allclose(dkwm, dkwt, atol, atol)
                    np.testing.assert_allclose(dvwm, dvwt, atol, atol)
                    np.testing.assert_allclose(dowm, dowt, atol, atol)
                    np.testing.assert_allclose(dqbm, dqbt, atol, atol)
                    np.testing.assert_allclose(dkbm, dkbt, atol, atol)
                    np.testing.assert_allclose(dvbm, dvbt, atol, atol)
                    np.testing.assert_allclose(dobm, dobt, atol, atol)
                    np.testing.assert_allclose(awm, awt, atol, atol)

        def _run_mge_vs_naive(
            self,
            maybe_cudnn_style_mask=False,
            need_weights=False,
            average_attn_weights=False,
        ):
            def can_run(maybe_cudnn_style_mask, need_weights):
                heads = self.num_heads
                qproj_size = (
                    self.qproj_size if self.qproj_size else self.embed_dim * heads
                )
                kproj_size = self.kproj_size if self.kproj_size else self.kdim * heads
                vproj_size = self.vproj_size if self.vproj_size else self.vdim * heads
                if not (
                    (qproj_size == kproj_size)
                    and ((qproj_size % heads) == 0)
                    and ((kproj_size % heads) == 0)
                    and ((vproj_size % heads) == 0)
                ):
                    # parameter configuration does not conform to the semantics of multiheadattention.
                    return False
                if maybe_cudnn_style_mask and (
                    need_weights or self.qbias or self.kbias or self.vbias or self.obias
                ):
                    # if need_weights is true and have bias, the cuda proxy implementation will be run,
                    # and the cudnn-style mask cannot be processed at this time.
                    return False

                return True

            if can_run(maybe_cudnn_style_mask, need_weights):
                swh = [False, True]
                for is_training, with_bias, with_attn_mask in [
                    (i, j, k) for i in swh for j in swh for k in swh
                ]:
                    atol = 1e-3
                    (
                        ym,
                        dqm,
                        dkm,
                        dvm,
                        dqwm,
                        dkwm,
                        dvwm,
                        dowm,
                        dqbm,
                        dkbm,
                        dvbm,
                        dobm,
                        awm,
                    ) = self._run_mge(
                        is_training,
                        with_bias,
                        with_attn_mask,
                        maybe_cudnn_style_mask=maybe_cudnn_style_mask,
                        need_weights=need_weights,
                        average_attn_weights=average_attn_weights,
                    )
                    (
                        yn,
                        dqn,
                        dkn,
                        dvn,
                        dqwn,
                        dkwn,
                        dvwn,
                        down,
                        dqbn,
                        dkbn,
                        dvbn,
                        dobn,
                        awn,
                    ) = self._run_naive(
                        is_training,
                        with_bias,
                        with_attn_mask,
                        need_weights=need_weights,
                        average_attn_weights=average_attn_weights,
                    )

                    np.testing.assert_allclose(ym, yn, atol, atol)
                    np.testing.assert_allclose(dqm, dqn, atol, atol)
                    np.testing.assert_allclose(dkm, dkn, atol, atol)
                    np.testing.assert_allclose(dvm, dvn, atol, atol)
                    np.testing.assert_allclose(dqwm, dqwn, atol, atol)
                    np.testing.assert_allclose(dkwm, dkwn, atol, atol)
                    np.testing.assert_allclose(dvwm, dvwn, atol, atol)
                    np.testing.assert_allclose(dowm, down, atol, atol)
                    np.testing.assert_allclose(dqbm, dqbn, atol, atol)
                    np.testing.assert_allclose(dkbm, dkbn, atol, atol)
                    np.testing.assert_allclose(dvbm, dvbn, atol, atol)
                    np.testing.assert_allclose(dobm, dobn, atol, atol)
                    np.testing.assert_allclose(awm, awn, atol, atol)

        def run_mge_seed(self):
            def can_run():
                heads = self.num_heads
                qproj_size = (
                    self.qproj_size if self.qproj_size else self.embed_dim * heads
                )
                kproj_size = self.kproj_size if self.kproj_size else self.kdim * heads
                vproj_size = self.vproj_size if self.vproj_size else self.vdim * heads
                if not (
                    (qproj_size == kproj_size)
                    and ((qproj_size % heads) == 0)
                    and ((kproj_size % heads) == 0)
                    and ((vproj_size % heads) == 0)
                    and self.batch_size * self.seq_qlen * self.seq_klen > 6
                ):
                    return False
                return True

            is_training = True
            if can_run():
                swh = [False, True]
                for with_bias, with_attn_mask in [(i, j) for i in swh for j in swh]:
                    q_mge = Tensor(self.q_np)
                    k_mge = Tensor(self.k_np)
                    v_mge = Tensor(self.v_np)
                    if with_bias:
                        qkvoweight_bias_mge = Tensor(
                            np.concatenate((self.qkvoweight_np, self.qkvobias_np), -1)
                        )
                    else:
                        qkvoweight_bias_mge = Tensor(self.qkvoweight_np)
                    mask_mge = Tensor(self.mask_np)
                    # drop seed
                    attn_drop_tmp = np.random.uniform(0, 0.5, 1)[0]
                    out_drop_tmp = np.random.uniform(0, 0.5, 1)[0]

                    def lambda_function():
                        return F.nn.multi_head_attention(
                            q_mge,
                            k_mge,
                            v_mge,
                            self.embed_dim,
                            self.num_heads,
                            attn_drop=attn_drop_tmp,
                            out_drop=out_drop_tmp,
                            io_weight_bias=qkvoweight_bias_mge,
                            qproj_size=self.qproj_size,
                            kproj_size=self.kproj_size,
                            vproj_size=self.vproj_size,
                            oproj_size=self.oproj_size,
                            qbias=with_bias and self.qbias,
                            kbias=with_bias and self.kbias,
                            vbias=with_bias and self.vbias,
                            obias=with_bias and self.obias,
                            attn_mask=mask_mge if with_attn_mask else None,
                            is_causal=with_attn_mask,
                            training=is_training,
                            need_weights=False,
                        )[0]

                    set_global_rng_seed(111)
                    out1 = lambda_function()
                    out2 = lambda_function()
                    out3 = lambda_function()
                    assert not (out1.numpy() == out2.numpy()).all()
                    assert not (out1.numpy() == out3.numpy()).all()
                    assert not (out2.numpy() == out3.numpy()).all()

                    set_global_rng_seed(111)
                    out4 = lambda_function()
                    assert (out1.numpy() == out4.numpy()).all()
                    assert not (out2.numpy() == out4.numpy()).all()
                    assert not (out3.numpy() == out4.numpy()).all()

                    set_global_rng_seed(111)
                    out5 = lambda_function()
                    assert (out1.numpy() == out5.numpy()).all()
                    assert (out4.numpy() == out5.numpy()).all()

                    set_global_rng_seed(222)
                    out6 = lambda_function()
                    assert not (out1.numpy() == out6.numpy()).all()

        def run_mge_vs_naive_basic(self):
            # cuda proxy or cudnn[not cudnn style mask] VS naive
            self._run_mge_vs_naive()

        def run_mge_vs_torch_basic(self):
            # cuda proxy or cudnn[not cudnn style mask] VS torch
            if torch.__version__ >= "2.0.0":
                self._run_mge_vs_torch()

        def run_mge_vs_naive_using_mge_proxy(self):
            # cuda proxy VS naive
            self._run_mge_vs_naive(need_weights=True, average_attn_weights=False)
            self._run_mge_vs_naive(need_weights=True, average_attn_weights=True)

        def run_mge_vs_torch_using_mge_proxy(self):
            # cuda proxy VS torch
            if torch.__version__ >= "2.0.0":
                self._run_mge_vs_torch(need_weights=True, average_attn_weights=False)
                self._run_mge_vs_torch(need_weights=True, average_attn_weights=True)

        def run_mge_vs_naive_with_cudnn_style_mask(self):
            # cudnn[cudnn style mask] VS naive
            if (
                is_cuda_available()
                and get_cudnn_version() >= 8004
                and torch.__version__ >= "2.0.0"
            ):
                self._run_mge_vs_naive(maybe_cudnn_style_mask=True, need_weights=False)

        def run_mge_vs_torch_with_cudnn_style_mask(self):
            # cudnn[cudnn style mask] VS torch
            if (
                is_cuda_available()
                and get_cudnn_version() >= 8004
                and torch.__version__ >= "2.0.0"
            ):
                self._run_mge_vs_torch(maybe_cudnn_style_mask=True, need_weights=False)

    def main_test_mge_seed(
        batch_size, seq_qlen, seq_klen, embed_dim, kdim, vdim, num_heads
    ):
        test = MHATest(
            batch_size,
            seq_qlen,
            seq_klen,
            embed_dim,
            kdim,
            vdim,
            num_heads,
            embed_dim,
            embed_dim,
            embed_dim,
            embed_dim,
            True,
            True,
            True,
            True,
        )
        test.run_mge_seed()

    def main_test_mge_vs_torch(
        batch_size, seq_qlen, seq_klen, embed_dim, kdim, vdim, num_heads
    ):
        def gen_torch_parameter_set(
            batch_size, seq_qlen, seq_klen, qdim, kdim, vdim, num_heads
        ):
            return [
                [qdim, qdim, qdim, qdim, True, True, True, True],
                [qdim, qdim, qdim, qdim, True, True, True, False],
                [qdim, qdim, qdim, qdim, False, False, False, True],
                [qdim, qdim, qdim, qdim, False, False, False, False],
            ]

        for i in gen_torch_parameter_set(
            batch_size, seq_qlen, seq_klen, embed_dim, kdim, vdim, num_heads
        ):
            test = MHATest(
                batch_size,
                seq_qlen,
                seq_klen,
                embed_dim,
                kdim,
                vdim,
                num_heads,
                i[0],
                i[1],
                i[2],
                i[3],
                i[4],
                i[5],
                i[6],
                i[7],
            )
            test.run_mge_vs_torch_basic()
            test.run_mge_vs_torch_using_mge_proxy()
            test.run_mge_vs_torch_with_cudnn_style_mask()

    def main_test_mge_vs_naive(
        batch_size, seq_qlen, seq_klen, embed_dim, kdim, vdim, num_heads
    ):
        def gen_naive_parameter_set(
            batch_size, seq_qlen, seq_klen, embed_dim, kdim, vdim, num_heads
        ):
            def gen_proj_bias_permutation(qps, kps, vps, ops):
                res = []
                true_list = [qps, kps, vps, ops, True, True, True, True]
                false_list = [0, 0, 0, 0, False, False, False, False]
                masks = [
                    [int(j) for j in format(i, "0{}b".format(4))] for i in range(2 ** 4)
                ]
                for mask1 in masks:
                    mask2s = set()
                    for mask2 in masks:
                        mask2s.add(tuple([x & y for x, y in zip(mask1, mask2)]))
                    for mask2 in mask2s:
                        mask = [x for l in [mask1, mask2] for x in l]
                        res.append(
                            [
                                true_list[i] if m == 1 else false_list[i]
                                for i, m in enumerate(mask)
                            ]
                        )
                return res

            res = []
            res.extend(
                gen_proj_bias_permutation(embed_dim, embed_dim, embed_dim, embed_dim)
            )
            res.extend(
                gen_proj_bias_permutation(
                    embed_dim, embed_dim + 1, embed_dim + 2, embed_dim + 3
                )
            )
            res.extend(
                gen_proj_bias_permutation(
                    embed_dim * 2, embed_dim * 3, embed_dim * 4, embed_dim * 5
                )
            )

            return res

        for i in gen_naive_parameter_set(
            batch_size, seq_qlen, seq_klen, embed_dim, kdim, vdim, num_heads
        ):
            test = MHATest(
                batch_size,
                seq_qlen,
                seq_klen,
                embed_dim,
                kdim,
                vdim,
                num_heads,
                i[0],
                i[1],
                i[2],
                i[3],
                i[4],
                i[5],
                i[6],
                i[7],
            )
            test.run_mge_vs_naive_basic()
            test.run_mge_vs_naive_using_mge_proxy()
            test.run_mge_vs_naive_with_cudnn_style_mask()

    main_test_mge_seed(batch_size, seq_qlen, seq_klen, embed_dim, kdim, vdim, num_heads)
    main_test_mge_vs_torch(
        batch_size, seq_qlen, seq_klen, embed_dim, kdim, vdim, num_heads
    )
    main_test_mge_vs_naive(
        batch_size, seq_qlen, seq_klen, embed_dim, kdim, vdim, num_heads
    )

