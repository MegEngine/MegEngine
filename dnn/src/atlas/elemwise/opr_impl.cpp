#include "opr_impl.h"
#include "acl/acl_op_compiler.h"
#include "aclnnop/aclnn_abs.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_bitwise_or_tensor.h"
#include "aclnnop/aclnn_clamp.h"
#include "aclnnop/aclnn_copy.h"
#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_exp.h"
#include "aclnnop/aclnn_fill_scalar.h"
#include "aclnnop/aclnn_floor_divide.h"
#include "aclnnop/aclnn_gt_scalar.h"
#include "aclnnop/aclnn_log.h"
#include "aclnnop/aclnn_logsigmoid.h"
#include "aclnnop/aclnn_lt_tensor.h"
#include "aclnnop/aclnn_masked_fill_scalar.h"
#include "aclnnop/aclnn_masked_select.h"
#include "aclnnop/aclnn_maximum.h"
#include "aclnnop/aclnn_minimum.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_neg.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"
#include "aclnnop/aclnn_relu.h"
#include "aclnnop/aclnn_rsub.h"
#include "aclnnop/aclnn_sigmoid.h"
#include "aclnnop/aclnn_sign.h"
#include "aclnnop/aclnn_signbit.h"
#include "aclnnop/aclnn_sqrt.h"
#include "aclnnop/aclnn_sub.h"
#include "aclnnop/aclnn_tanh.h"
#include "aclnnop/aclnn_tanh_backward.h"
#include "aclnnop/aclnn_threshold.h"
#include "aclnnop/aclnn_threshold_backward.h"
#include "aclnnop/level2/aclnn_bitwise_and_tensor.h"
#include "aclnnop/level2/aclnn_bitwise_not.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;
using Mode = param::Elemwise::Mode;

template <typename T>
void clip(const TensorNDArray& src, TensorND dst, HandleImpl* handle) {
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    SmallVector<AclTensor> acl_inps;
    for (size_t i = 0; i < src.size(); ++i) {
        acl_inps.emplace_back(src[i]);
    }
    AclTensor acl_out(dst);

    auto get_scalar = [=](const TensorND& tensor) -> T {
        T value{};
        aclrtMemcpyAsync(
                &value, sizeof(value), tensor.raw_ptr(), sizeof(value),
                ACL_MEMCPY_DEVICE_TO_HOST, handle->stream());
        aclrtSynchronizeStream(handle->stream());
        return value;
    };
    T min_value = get_scalar(src[1]);
    T max_value = get_scalar(src[2]);
    AclScalar acl_min(min_value), acl_max(max_value);
    aclnn_check(aclnnClampGetWorkspaceSize(
            acl_inps[0].get(), acl_min.get(), acl_max.get(), acl_out.get(), &ws_size,
            &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnClamp(ws.ptr(), ws_size, executor, handle->stream()));
}

void abs_grad(const TensorNDArray& src, TensorND dst, HandleImpl* handle) {
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    SmallVector<AclTensor> acl_inps;
    for (size_t i = 0; i < src.size(); ++i) {
        acl_inps.emplace_back(src[i]);
    }
    AclTensor acl_out(dst);
    aclnn_check(aclnnSignGetWorkspaceSize(
            acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
    AclMem ws1(ws_size, handle);
    aclnn_check(aclnnSign(ws1.ptr(), ws_size, executor, handle->stream()));

    aclnn_check(aclnnInplaceMulGetWorkspaceSize(
            acl_out.get(), acl_inps[1].get(), &ws_size, &executor));
    AclMem ws2(ws_size, handle);
    aclnn_check(aclnnInplaceMul(ws2.ptr(), ws_size, executor, handle->stream()));
}

void ElemwiseForwardImpl::exec(const TensorNDArray& src, _megdnn_tensor_out dst) {
    auto handle = concrete_handle(this->handle());
    SmallVector<AclTensor> acl_inps;
    for (size_t i = 0; i < src.size(); ++i) {
        acl_inps.emplace_back(src[i]);
    }
    AclTensor acl_out(dst);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    if (m_param.mode == Mode::ADD) {
        aclnn_check(aclnnAddGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(),
                AclScalar(1.0, dst.layout.dtype).get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnAdd(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::MUL) {
        aclnn_check(aclnnMulGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnMul(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::RELU) {
        aclnn_check(aclnnReluGetWorkspaceSize(
                acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnRelu(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::SUB) {
        aclnn_check(aclnnRsubGetWorkspaceSize(
                acl_inps[1].get(), acl_inps[0].get(),
                AclScalar(1.0, dst.layout.dtype).get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnRsubs(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::TRUE_DIV) {
        aclnn_check(aclnnDivGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnDiv(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::EXP) {
        aclnn_check(aclnnExpGetWorkspaceSize(
                acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnExp(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::LOG) {
        aclnn_check(aclnnLogGetWorkspaceSize(
                acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnLog(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::NEGATE) {
        AclTensor other(src[0]);
        aclnn_check(aclnnNegGetWorkspaceSize(
                acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnNeg(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::SWITCH_GT0) {
        AclScalar threshold(0.0f);
        aclnn_check(aclnnThresholdBackwardGetWorkspaceSize(
                acl_inps[1].get(), acl_inps[0].get(), threshold.get(), acl_out.get(),
                &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(
                aclnnThresholdBackward(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::AND) {
        aclnn_check(aclnnBitwiseAndTensorGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(
                aclnnBitwiseAndTensor(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::COND_LEQ_MOV) {
        AclMem temp_mem(src[0].layout.access_bytes(), handle);
        AclTensor temp_tensor(temp_mem.ptr(), src[0].layout);

        AclScalar alpha(1.0, dtype::Float32());
        aclnn_check(aclnnSubGetWorkspaceSize(
                acl_inps[1].get(), acl_inps[0].get(), alpha.get(), temp_tensor.get(),
                &ws_size, &executor));
        AclMem ws_2(ws_size, handle);
        aclnn_check(aclnnSub(ws_2.ptr(), ws_size, executor, handle->stream()));

        AclScalar threshold(-0.0000001, dtype::Float32());
        aclnn_check(aclnnThresholdBackwardGetWorkspaceSize(
                acl_inps[2].get(), temp_tensor.get(), threshold.get(), acl_out.get(),
                &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(
                aclnnThresholdBackward(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::COND_LT_MOV) {
        AclMem temp_mem(src[0].layout.access_bytes(), handle);
        AclTensor temp_tensor(temp_mem.ptr(), src[0].layout);

        AclScalar alpha(1.0, dtype::Float32());
        aclnn_check(aclnnSubGetWorkspaceSize(
                acl_inps[1].get(), acl_inps[0].get(), alpha.get(), temp_tensor.get(),
                &ws_size, &executor));
        AclMem ws_2(ws_size, handle);
        aclnn_check(aclnnSub(ws_2.ptr(), ws_size, executor, handle->stream()));

        AclScalar threshold(0.0, dtype::Float32());
        aclnn_check(aclnnThresholdBackwardGetWorkspaceSize(
                acl_inps[2].get(), temp_tensor.get(), threshold.get(), acl_out.get(),
                &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(
                aclnnThresholdBackward(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::LOGSIGMOID) {
        aclnn_check(aclnnLogSigmoidGetWorkspaceSize(
                acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnLogSigmoid(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::MAX) {
        aclnn_check(aclnnMaximumGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnMaximum(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::MIN) {
        aclnn_check(aclnnMinimumGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnMinimum(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::NOT) {
        aclnn_check(aclnnBitwiseNotGetWorkspaceSize(
                acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnBitwiseNot(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::POW) {
        aclnn_check(aclnnPowTensorTensorGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(
                aclnnPowTensorTensor(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::SIGMOID) {
        aclnn_check(aclnnSigmoidGetWorkspaceSize(
                acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnSigmoid(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::SQRT) {
        aclnn_check(aclnnSqrtGetWorkspaceSize(
                acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnSqrt(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::SIGMOID_GRAD) {
        AclMem scalar_mem(src[0].layout.access_bytes(), handle);
        AclTensor scalar_tensor(scalar_mem.ptr(), src[0].layout);
        AclScalar scalar_value(1.0);

        aclnn_check(aclnnInplaceFillScalarGetWorkspaceSize(
                scalar_tensor.get(), scalar_value.get(), &ws_size, &executor));
        AclMem ws2(ws_size, handle);
        aclnn_check(
                aclnnInplaceFillScalar(ws2.ptr(), ws_size, executor, handle->stream()));

        AclMem temp_mem(src[0].layout.access_bytes(), handle);
        AclTensor temp_tensor(temp_mem.ptr(), src[0].layout);

        aclnn_check(aclnnSubGetWorkspaceSize(
                scalar_tensor.get(), acl_inps[0].get(), scalar_value.get(),
                temp_tensor.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnSub(ws.ptr(), ws_size, executor, handle->stream()));

        aclnn_check(aclnnInplaceMulGetWorkspaceSize(
                temp_tensor.get(), acl_inps[0].get(), &ws_size, &executor));
        AclMem ws_3(ws_size, handle);
        aclnn_check(aclnnInplaceMul(ws_3.ptr(), ws_size, executor, handle->stream()));

        aclnn_check(aclnnMulGetWorkspaceSize(
                temp_tensor.get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws_4(ws_size, handle);
        aclnn_check(aclnnMul(ws_4.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::SOFTPLUS_GRAD) {
        // e_x = e^x
        AclMem e_x_mem(src[0].layout.access_bytes(), handle);
        AclTensor e_x_tensor(e_x_mem.ptr(), src[0].layout);
        aclnn_check(aclnnExpGetWorkspaceSize(
                acl_inps[0].get(), e_x_tensor.get(), &ws_size, &executor));
        AclMem ws1(ws_size, handle);
        aclnn_check(aclnnExp(ws1.ptr(), ws_size, executor, handle->stream()));

        // 1 + e^x
        AclMem den_mem(src[0].layout.access_bytes(), handle);
        AclTensor den_tensor(den_mem.ptr(), src[0].layout);
        AclScalar acl_one(1, src[0].layout.dtype);
        AclScalar acl_alpha(1, src[0].layout.dtype);
        aclnn_check(aclnnAddsGetWorkspaceSize(
                e_x_tensor.get(), acl_one.get(), acl_alpha.get(), den_tensor.get(),
                &ws_size, &executor));
        AclMem ws2(ws_size, handle);
        aclnn_check(aclnnAdds(ws2.ptr(), ws_size, executor, handle->stream()));

        AclMem molecule_mem(src[0].layout.access_bytes(), handle);
        AclTensor molecule_tensor(e_x_mem.ptr(), src[0].layout);
        aclnn_check(aclnnMulGetWorkspaceSize(
                acl_inps[1].get(), e_x_tensor.get(), molecule_tensor.get(), &ws_size,
                &executor));
        AclMem ws3(ws_size, handle);
        aclnn_check(aclnnMul(ws3.ptr(), ws_size, executor, handle->stream()));

        aclnn_check(aclnnDivGetWorkspaceSize(
                molecule_tensor.get(), den_tensor.get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws4(ws_size, handle);
        aclnn_check(aclnnDiv(ws4.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::CLIP) {
#define cb(DType)                                    \
    if (dst.layout.dtype == DType()) {               \
        using T = typename DTypeTrait<DType>::ctype; \
        clip<T>(src, dst, handle);                   \
        return;                                      \
    }
        cb(::megdnn::dtype::Float32) cb(::megdnn::dtype::Float16)
    } else if (m_param.mode == Mode::ABS) {
        aclnn_check(aclnnAbsGetWorkspaceSize(
                acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnAbs(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::ABS_GRAD) {
        abs_grad(src, dst, handle);
    } else if (m_param.mode == Mode::SIGN) {
        aclnn_check(aclnnSignGetWorkspaceSize(
                acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnSign(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::TANH) {
        aclnn_check(aclnnTanhGetWorkspaceSize(
                acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnTanh(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::FLOOR_DIV) {
        aclnn_check(aclnnFloorDivideGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnFloorDivide(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::TANH_GRAD) {
        aclnn_check(aclnnTanhBackwardGetWorkspaceSize(
                acl_inps[1].get(), acl_inps[0].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnTanhBackward(ws.ptr(), ws_size, executor, handle->stream()));
    }
}
