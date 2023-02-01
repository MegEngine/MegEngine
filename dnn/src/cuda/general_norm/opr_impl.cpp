#include "src/cuda/general_norm/opr_impl.h"
#include "src/cuda/general_norm/general_norm_cuda.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void GeneralNormForwardImpl::exec(
        _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
        _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
        _megdnn_workspace workspace) {
    check_exec(
            data.layout, weight.layout, bias.layout, dst.layout, mean.layout,
            rstd.layout, workspace.size);

    auto p = param();
    float eps = p.eps;
    bool affine = p.affine;
    uint64_t slice_length = p.normalized_size;
    uint64_t slice_dim = p.normalized_dim;
    uint64_t n_slices = 1;
    for (size_t i = 0; i < data.layout.ndim - slice_dim; ++i) {
        n_slices = n_slices * data.layout.shape[i];
    }

    auto stream = cuda_stream(handle());
    using namespace ::megdnn::cuda::general_norm;

#define cb(DType)                                                                 \
    if (data.layout.dtype == DType()) {                                           \
        using T = typename DTypeTrait<DType>::ctype;                              \
        using T_ACC = float;                                                      \
        forward<T, T_ACC>(                                                        \
                data.ptr<T>(), affine ? weight.ptr<T>() : nullptr,                \
                affine ? bias.ptr<T>() : nullptr, static_cast<int64_t>(n_slices), \
                static_cast<int64_t>(slice_length), static_cast<T_ACC>(eps),      \
                dst.ptr<T>(), mean.ptr<T_ACC>(), rstd.ptr<T_ACC>(), stream);      \
        return;                                                                   \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}

void GeneralNormBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in data, _megdnn_tensor_in weight,
        _megdnn_tensor_in mean, _megdnn_tensor_in rstd, _megdnn_tensor_out ddata,
        _megdnn_tensor_out dweight, _megdnn_tensor_out dbias,
        _megdnn_workspace workspace) {
    check_exec(
            diff.layout, data.layout, weight.layout, mean.layout, rstd.layout,
            ddata.layout, dweight.layout, dbias.layout, workspace.size);
    auto p = param();
    bool affine = p.affine;
    uint64_t slice_length = p.normalized_size;
    uint64_t slice_dim = p.normalized_dim;
    uint64_t n_slices = 1;
    for (size_t i = 0; i < data.layout.ndim - slice_dim; ++i) {
        n_slices = n_slices * data.layout.shape[i];
    }

    auto stream = cuda_stream(handle());
    using namespace ::megdnn::cuda::general_norm;
#define cb(DType)                                                                   \
    if (data.layout.dtype == DType()) {                                             \
        using T = typename DTypeTrait<DType>::ctype;                                \
        using T_ACC = float;                                                        \
        backward<T, T_ACC>(                                                         \
                diff.ptr<T>(), data.ptr<T>(), mean.ptr<T_ACC>(), rstd.ptr<T_ACC>(), \
                affine ? weight.ptr<T>() : nullptr, n_slices, slice_length,         \
                ddata.ptr<T>(), affine ? dweight.ptr<T>() : nullptr,                \
                affine ? dbias.ptr<T>() : nullptr, stream);                         \
        return;                                                                     \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}

}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
