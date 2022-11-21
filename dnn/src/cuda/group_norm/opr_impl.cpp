#include "src/cuda/group_norm/opr_impl.h"
#include "src/cuda/group_norm/group_norm_cuda.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

size_t GroupNormForwardImpl::get_workspace_in_bytes(
        const TensorLayout&, const TensorLayout&, const TensorLayout&,
        const TensorLayout&, const TensorLayout&, const TensorLayout& rstd) {
    size_t N = rstd.shape[0];
    size_t G = rstd.shape[1];
    return get_workspace_bundle(N, G, rstd.dtype.size()).total_size_in_bytes();
}

WorkspaceBundle GroupNormForwardImpl::get_workspace_bundle(
        size_t N, size_t G, size_t dtype_size, void* raw_ptr) {
    return {raw_ptr, {N * G * dtype_size}, handle()->alignment_requirement()};
}

void GroupNormForwardImpl::exec(
        _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
        _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
        _megdnn_workspace workspace) {
    check_exec(
            data.layout, weight.layout, bias.layout, dst.layout, mean.layout,
            rstd.layout, workspace.size);

    auto p = param();
    float eps = p.eps;
    int group = p.group;
    bool affine = p.affine;
    auto layout = data.layout;
    size_t N, C, H, W, imsize;
    N = layout.shape[0];
    C = layout.shape[1];
    H = layout.shape[2];
    W = layout.shape[3];
    imsize = H * W;

    auto stream = cuda_stream(handle());
    using namespace ::megdnn::cuda::group_norm;
    auto wbundle =
            get_workspace_bundle(N, group, rstd.layout.dtype.size(), workspace.raw_ptr);

#define cb(DType)                                                                  \
    if (data.layout.dtype == DType()) {                                            \
        using T = typename DTypeTrait<DType>::ctype;                               \
        using T_ACC = float;                                                       \
        T_ACC* temp_rstd = wbundle.get_workspace(0).ptr<T_ACC>();                  \
        forward<T, T_ACC>(                                                         \
                data.ptr<T>(), affine ? weight.ptr<T>() : nullptr,                 \
                affine ? bias.ptr<T>() : nullptr, dst.ptr<T>(), mean.ptr<T_ACC>(), \
                rstd.ptr<T_ACC>(), temp_rstd, static_cast<T_ACC>(eps),             \
                static_cast<int>(group), static_cast<int>(N), static_cast<int>(C), \
                static_cast<int>(W), static_cast<int>(imsize), stream);            \
        return;                                                                    \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}

size_t GroupNormBackwardImpl::get_workspace_in_bytes(
        const TensorLayout&, const TensorLayout& data, const TensorLayout&,
        const TensorLayout& mean, const TensorLayout&, const TensorLayout&,
        const TensorLayout&, const TensorLayout&) {
    size_t N = data.shape[0];
    size_t C = data.shape[1];
    size_t G = mean.shape[1];
    return get_workspace_bundle(N, C, G, data.dtype.size()).total_size_in_bytes();
}

WorkspaceBundle GroupNormBackwardImpl::get_workspace_bundle(
        size_t N, size_t C, size_t G, size_t dtype_size, void* raw_ptr) {
    return {raw_ptr,
            {N * C * dtype_size, N * C * dtype_size, N * C * dtype_size,
             N * G * dtype_size, N * G * dtype_size},
            handle()->alignment_requirement()};
}

void GroupNormBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in data, _megdnn_tensor_in weight,
        _megdnn_tensor_in mean, _megdnn_tensor_in rstd, _megdnn_tensor_out ddata,
        _megdnn_tensor_out dweight, _megdnn_tensor_out dbias,
        _megdnn_workspace workspace) {
    check_exec(
            diff.layout, data.layout, weight.layout, mean.layout, rstd.layout,
            ddata.layout, dweight.layout, dbias.layout, workspace.size);
    auto p = param();
    bool affine = p.affine;
    float eps = p.eps;
    int group = p.group;
    auto layout = data.layout;
    size_t N, C, H, W, imsize;
    N = layout.shape[0];
    C = layout.shape[1];
    H = layout.shape[2];
    W = layout.shape[3];
    imsize = H * W;

    auto stream = cuda_stream(handle());
    using namespace ::megdnn::cuda::group_norm;
    auto wbundle = get_workspace_bundle(
            N, C, group, data.layout.dtype.size(), workspace.raw_ptr);
#define cb(DType)                                                                   \
    if (data.layout.dtype == DType()) {                                             \
        using T = typename DTypeTrait<DType>::ctype;                                \
        using T_ACC = float;                                                        \
        T* ds = wbundle.get_workspace(0).ptr<T>();                                  \
        T* db = wbundle.get_workspace(1).ptr<T>();                                  \
        T* p1 = wbundle.get_workspace(2).ptr<T>();                                  \
        T* p2 = wbundle.get_workspace(3).ptr<T>();                                  \
        T* p3 = wbundle.get_workspace(4).ptr<T>();                                  \
        backward<T, T_ACC>(                                                         \
                diff.ptr<T>(), data.ptr<T>(), mean.ptr<T_ACC>(), rstd.ptr<T_ACC>(), \
                affine ? weight.ptr<T>() : nullptr, ddata.ptr<T>(),                 \
                affine ? dweight.ptr<T>() : nullptr,                                \
                affine ? dbias.ptr<T>() : nullptr, static_cast<T_ACC>(eps),         \
                static_cast<int>(group), static_cast<int>(N), static_cast<int>(C),  \
                static_cast<int>(imsize), ds, db, p1, p2, p3, stream);              \
        return;                                                                     \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}

}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
