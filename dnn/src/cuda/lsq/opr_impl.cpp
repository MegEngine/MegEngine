#include "./opr_impl.h"
#include "./kern.cuh"
#include "src/common/utils.h"
namespace megdnn {
namespace cuda {

void LSQForwardImpl::exec(
        _megdnn_tensor_in input, _megdnn_tensor_in scale, _megdnn_tensor_in zero_point,
        _megdnn_tensor_in grad_scale, _megdnn_tensor_out output,
        _megdnn_workspace workspace) {
    check_exec(
            input.layout, scale.layout, zero_point.layout, grad_scale.layout,
            output.layout, workspace.size);

    if (!input.layout.is_contiguous() || !output.layout.is_contiguous())
        return exec_noncontig(input, scale, zero_point, grad_scale, output);

    ElemwiseOpParamN<3> ele_param;
    ele_param[0] = scale;
    ele_param[0].layout = ele_param[0].layout.broadcast(input.layout);
    ele_param[1] = zero_point;
    ele_param[1].layout = ele_param[1].layout.broadcast(input.layout);
    ele_param[2] = grad_scale;
    ele_param[2].layout = ele_param[2].layout.broadcast(input.layout);
    ele_param.init_from_given_tensor();
    auto m_param = param();
    auto stream = cuda_stream(handle());

#define cb(DType)                                                                      \
    if (input.layout.dtype == DType()) {                                               \
        using T = typename DTypeTrait<DType>::ctype;                                   \
        run_elemwise<LSQKernOp<T>, T, 3>(ele_param, stream, {input, output, m_param}); \
        return;                                                                        \
    }
    cb(megdnn::dtype::Float32)
#undef cb
}

void LSQForwardImpl::exec_noncontig(
        _megdnn_tensor_in input, _megdnn_tensor_in scale, _megdnn_tensor_in zero_point,
        _megdnn_tensor_in grad_scale, _megdnn_tensor_out output) {
    ElemwiseOpParamN<5> ele_param;
    ele_param[0] = output;
    ele_param[1] = input;
    ele_param[2] = scale;
    ele_param[2].layout = ele_param[2].layout.broadcast(input.layout);
    ele_param[3] = zero_point;
    ele_param[3].layout = ele_param[3].layout.broadcast(input.layout);
    ele_param[4] = grad_scale;
    ele_param[4].layout = ele_param[4].layout.broadcast(input.layout);
    ele_param.init_from_given_tensor();
    auto m_param = param();
    auto stream = cuda_stream(handle());

#define cb(DType)                                                                \
    if (input.layout.dtype == DType()) {                                         \
        using T = typename DTypeTrait<DType>::ctype;                             \
        run_elemwise<LSQKernOpNonContig<T>, T, 5>(ele_param, stream, {m_param}); \
        return;                                                                  \
    }
    cb(megdnn::dtype::Float32)
#undef cb
}

void LSQBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in input, _megdnn_tensor_in scale,
        _megdnn_tensor_in zero_point, _megdnn_tensor_in grad_scale,
        _megdnn_tensor_out grad_x, _megdnn_tensor_out grad_s,
        _megdnn_workspace workspace) {
    check_exec(
            diff.layout, input.layout, scale.layout, zero_point.layout,
            grad_scale.layout, grad_x.layout, grad_s.layout, workspace.size);

    if (!input.layout.is_contiguous() || !diff.layout.is_contiguous() ||
        !grad_x.layout.is_contiguous() || !grad_s.layout.is_contiguous())
        return exec_noncontig(
                diff, input, scale, zero_point, grad_scale, grad_x, grad_s);

    ElemwiseOpParamN<3> ele_param;
    ele_param[0] = scale;
    ele_param[0].layout = ele_param[0].layout.broadcast(input.layout);
    ele_param[1] = zero_point;
    ele_param[1].layout = ele_param[1].layout.broadcast(input.layout);
    ele_param[2] = grad_scale;
    ele_param[2].layout = ele_param[2].layout.broadcast(input.layout);
    ele_param.init_from_given_tensor();
    auto m_param = param();
    auto stream = cuda_stream(handle());

#define cb(DType)                                                           \
    if (grad_x.layout.dtype == DType()) {                                   \
        using T = typename DTypeTrait<DType>::ctype;                        \
        run_elemwise<LSQBwdKernOp<T>, T, 3>(                                \
                ele_param, stream, {diff, input, grad_x, grad_s, m_param}); \
        return;                                                             \
    }
    cb(megdnn::dtype::Float32)
#undef cb
}

void LSQBackwardImpl::exec_noncontig(
        _megdnn_tensor_in diff, _megdnn_tensor_in input, _megdnn_tensor_in scale,
        _megdnn_tensor_in zero_point, _megdnn_tensor_in grad_scale,
        _megdnn_tensor_out grad_x, _megdnn_tensor_out grad_s) {
    ElemwiseOpParamN<7> ele_param;
    ele_param[0] = grad_x;
    ele_param[1] = grad_s;
    ele_param[2] = diff;
    ele_param[3] = input;
    ele_param[4] = scale;
    ele_param[4].layout = ele_param[4].layout.broadcast(input.layout);
    ele_param[5] = zero_point;
    ele_param[5].layout = ele_param[5].layout.broadcast(input.layout);
    ele_param[6] = grad_scale;
    ele_param[6].layout = ele_param[6].layout.broadcast(input.layout);
    ele_param.init_from_given_tensor();
    auto m_param = param();
    auto stream = cuda_stream(handle());

#define cb(DType)                                                                   \
    if (input.layout.dtype == DType()) {                                            \
        using T = typename DTypeTrait<DType>::ctype;                                \
        run_elemwise<LSQBwdKernOpNonContig<T>, T, 7>(ele_param, stream, {m_param}); \
        return;                                                                     \
    }
    cb(megdnn::dtype::Float32)
#undef cb
}

}  // namespace cuda
}  // namespace megdnn