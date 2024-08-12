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
#include "aclnnop/aclnn_fmod_tensor.h"
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
        acl_safe_memcpy_async_with_sync(
                &value, sizeof(value), tensor.raw_ptr(), sizeof(value),
                ACL_MEMCPY_DEVICE_TO_HOST, handle->stream());
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
        AclScalar acl_scalar_one(1.0, dst.layout.dtype);
        aclnn_check(aclnnAddGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_scalar_one.get(),
                acl_out.get(), &ws_size, &executor));
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
        AclScalar acl_scalar_one(1.0, dst.layout.dtype);
        aclnn_check(aclnnRsubGetWorkspaceSize(
                acl_inps[1].get(), acl_inps[0].get(), acl_scalar_one.get(),
                acl_out.get(), &ws_size, &executor));
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
        TensorLayout tmplyt(TensorShape(dst.layout), src[0].layout.dtype);
        AclMem temp_mem(tmplyt.access_bytes(), handle);
        AclTensor temp_tensor(temp_mem.ptr(), tmplyt);

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
        TensorLayout tmplyt(TensorShape(dst.layout), src[0].layout.dtype);
        AclMem temp_mem(tmplyt.access_bytes(), handle);
        AclTensor temp_tensor(temp_mem.ptr(), tmplyt);

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
        AclTensor acl_x(src[0]), acl_dy(src[1]), acl_out(dst);

        // exp(-abs(x))
        TensorLayout src_layout = src[0].layout;
        AclMem acl_exp_neg_abs_x_mem(src_layout.span().dist_byte(), handle);
        AclTensor acl_exp_neg_abs_x(acl_exp_neg_abs_x_mem.ptr(), src_layout);

        uint64_t abs_ws_size = 0;
        aclOpExecutor* abs_executor = nullptr;
        aclnn_check(aclnnAbsGetWorkspaceSize(
                acl_x.get(), acl_exp_neg_abs_x.get(), &abs_ws_size, &abs_executor));
        AclMem abs_ws(abs_ws_size, handle);
        aclnn_check(
                aclnnAbs(abs_ws.ptr(), abs_ws_size, abs_executor, handle->stream()));

        uint64_t neg_ws_size = 0;
        aclOpExecutor* neg_executor = nullptr;
        aclnn_check(aclnnInplaceNegGetWorkspaceSize(
                acl_exp_neg_abs_x.get(), &neg_ws_size, &neg_executor));
        AclMem neg_ws(neg_ws_size, handle);
        aclnn_check(aclnnInplaceNeg(
                neg_ws.ptr(), neg_ws_size, neg_executor, handle->stream()));

        uint64_t exp_ws_size = 0;
        aclOpExecutor* exp_executor = nullptr;
        aclnn_check(aclnnInplaceExpGetWorkspaceSize(
                acl_exp_neg_abs_x.get(), &exp_ws_size, &exp_executor));
        AclMem exp_ws(exp_ws_size, handle);
        aclnn_check(aclnnInplaceExp(
                exp_ws.ptr(), exp_ws_size, exp_executor, handle->stream()));

        // 1 + exp(-abs(x))
        AclMem acl_1p_prepared_x_mem(src_layout.span().dist_byte(), handle);
        AclTensor acl_1p_prepared_x(acl_1p_prepared_x_mem.ptr(), src_layout);

        AclScalar acl_other_one(1.0);
        AclScalar acl_alpha_one(1.0);
        uint64_t adds_ws_size = 0;
        aclOpExecutor* adds_executor = nullptr;
        aclnn_check(aclnnAddsGetWorkspaceSize(
                acl_exp_neg_abs_x.get(), acl_other_one.get(), acl_alpha_one.get(),
                acl_1p_prepared_x.get(), &adds_ws_size, &adds_executor));
        AclMem adds_ws(adds_ws_size, handle);
        aclnn_check(aclnnAdds(
                adds_ws.ptr(), adds_ws_size, adds_executor, handle->stream()));

        // dy * exp(-abs(x))
        AclMem acl_logg_mem(src_layout.span().dist_byte(), handle);
        AclTensor acl_logg(acl_logg_mem.ptr(), src_layout);

        uint64_t mul_ws_size = 0;
        aclOpExecutor* mul_executor = nullptr;
        aclnn_check(aclnnMulGetWorkspaceSize(
                acl_exp_neg_abs_x.get(), acl_dy.get(), acl_logg.get(), &mul_ws_size,
                &mul_executor));
        AclMem mul_ws(mul_ws_size, handle);
        aclnn_check(
                aclnnMul(mul_ws.ptr(), mul_ws_size, mul_executor, handle->stream()));

        // dy * exp(-abs(x)) / (1 + exp(-abs(x)))
        uint64_t div_ws_size = 0;
        aclOpExecutor* div_executor = nullptr;
        aclnn_check(aclnnInplaceDivGetWorkspaceSize(
                acl_logg.get(), acl_1p_prepared_x.get(), &div_ws_size, &div_executor));
        AclMem div_ws(div_ws_size, handle);
        aclnn_check(aclnnInplaceDiv(
                div_ws.ptr(), div_ws_size, div_executor, handle->stream()));

        // -(dy * exp(-abs(x)) / (1 + exp(-abs(x))))
        AclMem acl_neg_logg_mem(src_layout.span().dist_byte(), handle);
        AclTensor acl_neg_logg(acl_neg_logg_mem.ptr(), src_layout);

        AclScalar acl_other_neg_one(-1.0);
        uint64_t muls_ws_size = 0;
        aclOpExecutor* muls_executor = nullptr;
        aclnn_check(aclnnMulsGetWorkspaceSize(
                acl_logg.get(), acl_other_neg_one.get(), acl_neg_logg.get(),
                &muls_ws_size, &muls_executor));
        AclMem muls_ws(muls_ws_size, handle);
        aclnn_check(aclnnMuls(
                muls_ws.ptr(), muls_ws_size, muls_executor, handle->stream()));

        // nlogg = x > 0 ? logg : nlogg
        AclScalar acl_threshold(0.0);
        uint64_t tb0_ws_size = 0;
        aclOpExecutor* tb0_executor = nullptr;
        aclnn_check(aclnnThresholdBackwardGetWorkspaceSize(
                acl_neg_logg.get(), acl_x.get(), acl_threshold.get(), acl_logg.get(),
                &tb0_ws_size, &tb0_executor));
        AclMem tb0_ws(tb0_ws_size, handle);
        aclnn_check(aclnnThresholdBackward(
                tb0_ws.ptr(), tb0_ws_size, tb0_executor, handle->stream()));

        // relu(x)
        AclMem acl_relu_x_mem(src_layout.span().dist_byte(), handle);
        AclTensor acl_relu_x(acl_relu_x_mem.ptr(), src_layout);

        uint64_t relu_ws_size = 0;
        aclOpExecutor* relu_executor = nullptr;
        aclnn_check(aclnnReluGetWorkspaceSize(
                acl_x.get(), acl_relu_x.get(), &relu_ws_size, &relu_executor));
        AclMem relu_ws(relu_ws_size, handle);
        aclnn_check(aclnnRelu(
                relu_ws.ptr(), relu_ws_size, relu_executor, handle->stream()));

        // grad1
        AclMem acl_grad1_mem(src_layout.span().dist_byte(), handle);
        AclTensor acl_grad1(acl_grad1_mem.ptr(), src_layout);

        uint64_t fill_ws_size = 0;
        aclOpExecutor* fill_executor = nullptr;
        aclnn_check(aclnnInplaceFillScalarGetWorkspaceSize(
                acl_grad1.get(), acl_threshold.get(), &fill_ws_size, &fill_executor));
        AclMem fill_ws(fill_ws_size, handle);
        aclnn_check(aclnnInplaceFillScalar(
                fill_ws.ptr(), fill_ws_size, fill_executor, handle->stream()));

        // grad1 = relu(x) > 0 ? dy : grad1
        uint64_t tb1_ws_size = 0;
        aclOpExecutor* tb1_executor = nullptr;
        aclnn_check(aclnnThresholdBackwardGetWorkspaceSize(
                acl_dy.get(), acl_relu_x.get(), acl_threshold.get(), acl_grad1.get(),
                &tb1_ws_size, &tb1_executor));
        AclMem tb1_ws(tb1_ws_size, handle);
        aclnn_check(aclnnThresholdBackward(
                tb1_ws.ptr(), tb1_ws_size, tb1_executor, handle->stream()));

        // out = neg_logg + grad1
        uint64_t add_ws_size = 0;
        aclOpExecutor* add_executor = nullptr;
        aclnn_check(aclnnAddGetWorkspaceSize(
                acl_logg.get(), acl_grad1.get(), acl_alpha_one.get(), acl_out.get(),
                &add_ws_size, &add_executor));
        AclMem add_ws(add_ws_size, handle);
        aclnn_check(
                aclnnAdd(add_ws.ptr(), add_ws_size, add_executor, handle->stream()));
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
    } else if (m_param.mode == Mode::MOD) {
        aclnn_check(aclnnFmodTensorGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnFmodTensor(ws.ptr(), ws_size, executor, handle->stream()));
    } else {
        megdnn_throw("unsupported elemwise mode");
    }
}
