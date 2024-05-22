#include <algorithm>

#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/group_norm/opr_impl.h"
#include "src/atlas/utils.h"

#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_group_norm.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_neg.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_reduce_sum.h"
#include "aclnnop/aclnn_rsqrt.h"
#include "aclnnop/aclnn_sub.h"

namespace megdnn {
namespace atlas {

using Param = megdnn::GroupNorm::Param;

TensorND flatten(const TensorND& inp) {
    const TensorLayout inp_layout = inp.layout;
    TensorND dst;
    if (inp_layout.ndim == 1) {
        dst = inp;
    } else if (inp_layout.ndim == 4) {
        constexpr int C_POS = 1;
        int c = inp_layout.shape[C_POS];
        TensorLayout dst_layout({c}, inp_layout.dtype);
        dst = TensorND(inp.raw_ptr(), dst_layout);
    }
    return dst;
}

void GroupNormForwardImpl::exec(
        _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
        _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
        _megdnn_workspace workspace) {
    Param param = this->param();

    float eps = param.eps;
    size_t N = data.layout.shape[0];
    size_t C = data.layout.shape[1];
    size_t HxW = data.layout.shape[2] * data.layout.shape[3];
    const int64_t G = param.group;

    TensorND flatten_weight = flatten(weight), flatten_bias = flatten(bias);
    AclTensor acl_self(data), acl_gamma(flatten_weight), acl_beta(flatten_bias),
            acl_out(dst), acl_meanOut(mean), acl_rstdOut(rstd);

    uint64_t ws_size;
    aclOpExecutor* executor = nullptr;
    auto handle = concrete_handle(this->handle());

    aclnn_check(aclnnGroupNormGetWorkspaceSize(
            acl_self.get(), acl_gamma.get(), acl_beta.get(), N, C, HxW, G, eps,
            acl_out.get(), acl_meanOut.get(), acl_rstdOut.get(), &ws_size, &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnGroupNorm(ws.ptr(), ws_size, executor, handle->stream()));

    aclnn_check(
            aclnnInplaceRsqrtGetWorkspaceSize(acl_rstdOut.get(), &ws_size, &executor));
    AclMem ws_2(ws_size, handle);
    aclnn_check(aclnnInplaceRsqrt(ws_2.ptr(), ws_size, executor, handle->stream()));

    float exponent_value = 4.0f;
    AclScalar exponent(exponent_value);
    aclnn_check(aclnnInplacePowTensorScalarGetWorkspaceSize(
            acl_rstdOut.get(), exponent.get(), &ws_size, &executor));
    AclMem ws_3(ws_size, handle);
    aclnn_check(aclnnInplacePowTensorScalar(
            ws_3.ptr(), ws_size, executor, handle->stream()));
}

void GroupNormBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in data, _megdnn_tensor_in weight,
        _megdnn_tensor_in mean, _megdnn_tensor_in rstd, _megdnn_tensor_out ddata,
        _megdnn_tensor_out dweight, _megdnn_tensor_out dbias,
        _megdnn_workspace workspace) {
    check_exec(
            diff.layout, data.layout, weight.layout, mean.layout, rstd.layout,
            ddata.layout, dweight.layout, dbias.layout, workspace.size);
    megdnn_assert(param().format == param::GroupNorm::Format::NCHW);

    /*
    because atlas do not support groupnorm_backward now, we implement it with reduce
    and elemwise, which need contiguous input. we perform backward as below

    def group_norm_grad(dy, x, w, mean, rstd, group, eps=1e-6):
        N, C, H, W = dy.shape
        c = C // group
        grouped_shape = (N, group, c, H, W)
        reduced_shape = (N, group, 1, 1, 1)
        affine_shape = (1, group, c, 1, 1)

        dy = dy.reshape(grouped_shape)
        x = x.reshape(grouped_shape)
        mean = mean.reshape(reduced_shape)
        rstd = rstd.reshape(reduced_shape)
        w = w.reshape(affine_shape)

        scalar = 1.0 / (c * H * W)
        std = 1.0 / np.sqrt(rstd + eps) # (N, group, 1, 1, 1)
        _x1 = dy.sum(axis=(3, 4), keepdims=True) # (N, group, c, 1, 1)
        _x2 = (x * dy).sum(axis=(3, 4), keepdims=True) # (N, group, c, 1, 1)

        _x3 = _x1 * w # (N, group, c, 1, 1)

        _x4 = (_x3 * mean - _x2 * w) * std * std * std * scalar # (N, group, c, 1, 1)
        _x5 = (-_x4 * mean - _x3 * std * scalar) # (N, group, c, 1, 1)

        _x6 = _x4.sum(axis=2, keepdims=True) # (N, group, 1, 1, 1)
        _x7 = _x5.sum(axis=2, keepdims=True) # (N, group, 1, 1, 1)

        dx = std * w * dy + _x6 * x + _x7
        dw = ((_x2 - _x1 * mean) * std).sum(axis=0, keepdims=True) # (N, group, c, 1, 1)
        db = _x1.sum(axis=0, keepdims=True)

        dx = dx.reshape(N, C, H, W)
        dw = dw.reshape(1, C, 1, 1)
        db = db.reshape(1, C, 1, 1)

        return dx, dw, db
    */
    megdnn_assert_contiguous(diff.layout);
    megdnn_assert_contiguous(data.layout);
    megdnn_assert_contiguous(ddata.layout);
    megdnn_assert(
            param().affine == true,
            "do not support affine == false, because this is only a temporary hack "
            "implemented");

    auto handle = concrete_handle(this->handle());
    size_t N = data.layout.shape[0];
    size_t C = data.layout.shape[1];
    size_t H = data.layout.shape[2];
    size_t W = data.layout.shape[3];
    size_t group = param().group;
    size_t c = C / group;
    megdnn_assert(C % group == 0, "C: %zu, group %zu", C, group);
    auto lyt_NgcHW = data.layout.reshape(TensorShape({N, group, c, H, W}));
    auto lyt_Ng111 = mean.layout.reshape(TensorShape({N, group, 1, 1, 1}));
    auto lyt_1gc11 = weight.layout.reshape(TensorShape({1, group, c, 1, 1}));
    auto lyt_Ngc11 = TensorLayout(TensorShape{N, group, c, 1, 1}, dtype::Float32());

    AclTensor acl_dy(diff.raw_ptr(), lyt_NgcHW);
    AclTensor acl_dx(ddata.raw_ptr(), lyt_NgcHW);
    AclTensor acl_x(data.raw_ptr(), lyt_NgcHW);
    AclTensor acl_mean(mean.raw_ptr(), lyt_Ng111);
    AclTensor acl_rstd(rstd.raw_ptr(), lyt_Ng111);
    AclTensor acl_w(weight.raw_ptr(), lyt_1gc11);

    AclScalar scalarf(1.0f / (c * H * W));
    AclScalar epsf(param().eps);
    AclScalar identityf(1.0f);
    AclIntArray reduce_axis_34({3, 4});
    AclIntArray reduce_axis_2({2});
    AclIntArray reduce_axis_0({0});

    // std = 1.0 / sqrt(rstd + eps)
    AclTempTensor(handle, acl_std, lyt_Ng111);
    aclnn_call(
            handle, aclnnAdds, acl_rstd.get(), epsf.get(), identityf.get(),
            acl_std.get());
    aclnn_call(handle, aclnnInplaceRsqrt, acl_std.get());

    // _x1 = dy.sum(axis=(3, 4), keepdims=True)
    AclTempTensor(handle, _x1, lyt_Ngc11);
    aclnn_call(
            handle, aclnnReduceSum, acl_dy.get(), reduce_axis_34.get(), true,
            as_acl_dtype(lyt_Ngc11.dtype), _x1.get());

    // _x2 = (x * dy).sum(axis=(3, 4), keepdims=True) # (N, group, c, 1, 1)
    AclTempTensor(handle, _tmp0, lyt_NgcHW);
    AclTempTensor(handle, _x2, lyt_Ngc11);
    aclnn_call(handle, aclnnMul, acl_x.get(), acl_dy.get(), _tmp0.get());
    aclnn_call(
            handle, aclnnReduceSum, _tmp0.get(), reduce_axis_34.get(), true,
            as_acl_dtype(lyt_Ngc11.dtype), _x2.get());

    // _x3 = _x1 * w
    AclTempTensor(handle, _x3, lyt_Ngc11);
    aclnn_call(handle, aclnnMul, _x1.get(), acl_w.get(), _x3.get());

    // _x4 = (_x3 * mean - _x2 * w) * std * std * std * scalar
    AclTempTensor(handle, _x4, lyt_Ngc11);
    AclTempTensor(handle, _tmp1, lyt_Ngc11);
    aclnn_call(handle, aclnnMul, _x3.get(), acl_mean.get(), _x4.get());
    aclnn_call(handle, aclnnMul, _x2.get(), acl_w.get(), _tmp1.get());
    aclnn_call(handle, aclnnInplaceSub, _x4.get(), _tmp1.get(), identityf.get());
    aclnn_call(handle, aclnnInplaceMul, _x4.get(), acl_std.get());
    aclnn_call(handle, aclnnInplaceMul, _x4.get(), acl_std.get());
    aclnn_call(handle, aclnnInplaceMul, _x4.get(), acl_std.get());
    aclnn_call(handle, aclnnInplaceMuls, _x4.get(), scalarf.get());

    //_x5 = (-_x4 * mean - _x3 * std * scalar)
    AclTempTensor(handle, _x5, lyt_Ngc11);
    aclnn_call(handle, aclnnNeg, _x4.get(), _x5.get());
    aclnn_call(handle, aclnnInplaceMul, _x5.get(), acl_mean.get());
    aclnn_call(handle, aclnnMul, _x3.get(), acl_std.get(), _tmp1.get());
    aclnn_call(handle, aclnnInplaceMuls, _tmp1.get(), scalarf.get());
    aclnn_call(handle, aclnnInplaceSub, _x5.get(), _tmp1.get(), identityf.get());

    // _x6 = _x4.sum(axis=2, keepdims=True)
    // _x7 = _x5.sum(axis=2, keepdims=True)
    AclTempTensor(handle, _x6, lyt_Ng111);
    AclTempTensor(handle, _x7, lyt_Ng111);
    aclnn_call(
            handle, aclnnReduceSum, _x4.get(), reduce_axis_2.get(), true,
            as_acl_dtype(lyt_Ng111.dtype), _x6.get());
    aclnn_call(
            handle, aclnnReduceSum, _x5.get(), reduce_axis_2.get(), true,
            as_acl_dtype(lyt_Ng111.dtype), _x7.get());

    // dx = dy * std * w  + _x6 * x + _x7
    aclnn_call(handle, aclnnMul, acl_dy.get(), acl_std.get(), acl_dx.get());
    aclnn_call(handle, aclnnInplaceMul, acl_dx.get(), acl_w.get());
    aclnn_call(handle, aclnnMul, _x6.get(), acl_x.get(), _tmp0.get());
    aclnn_call(handle, aclnnInplaceAdd, acl_dx.get(), _tmp0.get(), identityf.get());
    aclnn_call(handle, aclnnInplaceAdd, acl_dx.get(), _x7.get(), identityf.get());

    if (param().affine) {
        AclTensor acl_db(dbias.raw_ptr(), lyt_1gc11);
        AclTensor acl_dw(dweight.raw_ptr(), lyt_1gc11);

        // db = _x1.sum(axis=0, keepdims=True)
        aclnn_call(
                handle, aclnnReduceSum, _x1.get(), reduce_axis_0.get(), true,
                as_acl_dtype(lyt_1gc11.dtype), acl_db.get());

        // dw = (-(_x1 * mean - _x2) * std).sum(axis=0, keepdims=True) # (N, group, c,
        // 1, 1)
        aclnn_call(handle, aclnnMul, _x1.get(), acl_mean.get(), _tmp1.get());
        aclnn_call(handle, aclnnInplaceSub, _tmp1.get(), _x2.get(), identityf.get());
        aclnn_call(handle, aclnnInplaceNeg, _tmp1.get());
        aclnn_call(handle, aclnnInplaceMul, _tmp1.get(), acl_std.get());
        aclnn_call(
                handle, aclnnReduceSum, _tmp1.get(), reduce_axis_0.get(), true,
                as_acl_dtype(lyt_1gc11.dtype), acl_dw.get());
    }
}

}  // namespace atlas
}  // namespace megdnn
