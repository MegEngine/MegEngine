#include "src/naive/group_norm/opr_impl.h"
#include <algorithm>
#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;

namespace {

using Param = megdnn::GroupNorm::Param;

template <typename T, typename T_ACC = float>
void forward(
        _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
        _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
        const Param& param) {
    float eps = param.eps;
    bool affine = param.affine;
    size_t N = data.layout.shape[0];
    size_t C = data.layout.shape[1];
    size_t HxW = data.layout.shape[2] * data.layout.shape[3];
    const int64_t G = param.group;
    size_t D = C / G;
    size_t inner_size = D * HxW;

    for (size_t i = 0; i < N * G; i++) {
        T_ACC slice_sum = static_cast<T>(0.0f);
        for (size_t j = 0; j < inner_size; j++) {
            auto value = data.ptr<T>()[i * inner_size + j];
            slice_sum += value;
        }
        T_ACC slice_mean = static_cast<T>(slice_sum / inner_size);

        T_ACC slice_var = static_cast<T>(0.0f);
        for (size_t j = 0; j < inner_size; j++) {
            slice_var += (data.ptr<T>()[i * inner_size + j] - slice_mean) *
                         (data.ptr<T>()[i * inner_size + j] - slice_mean);
        }
        slice_var = slice_var / inner_size;

        T_ACC slice_std = static_cast<T>(1.0f) / static_cast<T>(sqrt(slice_var + eps));
        if (affine) {
            const int64_t g = i % G;
            for (size_t j = 0; j < D; j++) {
                const int64_t c = g * D + j;
                T_ACC s = slice_std * weight.ptr<T>()[c];
                T_ACC b = -s * slice_mean + bias.ptr<T>()[c];
                for (size_t k = 0; k < HxW; k++) {
                    dst.ptr<T>()[(i * D + j) * HxW + k] =
                            s * data.ptr<T>()[(i * D + j) * HxW + k] + b;
                }
            }
        } else {
            for (size_t j = 0; j < inner_size; j++) {
                dst.ptr<T>()[i * inner_size + j] =
                        (data.ptr<T>()[i * inner_size + j] - slice_mean) / slice_std;
            }
        }
        mean.ptr<T_ACC>()[i] = static_cast<T_ACC>(slice_mean);
        rstd.ptr<T_ACC>()[i] = static_cast<T_ACC>(slice_var);
    }
}

template <typename T, typename T_ACC = float>
void backward(
        _megdnn_tensor_in diff, _megdnn_tensor_in data, _megdnn_tensor_in weight,
        _megdnn_tensor_in mean, _megdnn_tensor_in rstd, _megdnn_tensor_out ddata,
        _megdnn_tensor_out dweight, _megdnn_tensor_out dbias, const Param& param,
        WorkspaceBundle wbundle) {
    bool affine = param.affine;
    size_t N = data.layout.shape[0];
    size_t C = data.layout.shape[1];
    size_t G = param.group;
    float eps = param.eps;
    size_t HxW = data.layout.shape[2] * data.layout.shape[3];
    T* ds = wbundle.get_workspace(0).ptr<T>();
    T* db = wbundle.get_workspace(1).ptr<T>();
    T* slice_std = wbundle.get_workspace(2).ptr<T>();
    for (size_t i = 0; i < N * G; i++) {
        slice_std[i] =
                static_cast<T>(1.0f) / static_cast<T>(sqrt(rstd.ptr<T_ACC>()[i] + eps));
    }
    for (size_t i = 0; i < N * C; i++) {
        T ds_data = static_cast<T>(0.0f);
        T db_data = static_cast<T>(0.0f);
        for (size_t j = 0; j < HxW; j++) {
            db_data += diff.ptr<T>()[i * HxW + j];
            ds_data += data.ptr<T>()[i * HxW + j] * diff.ptr<T>()[i * HxW + j];
        }
        ds[i] = ds_data;
        db[i] = db_data;
    }
    size_t D = C / G;
    const T s = T(1) / static_cast<T>(D * HxW);
    for (size_t i = 0; i < N * G; i++) {
        const int64_t g = i % G;
        T ds_v = static_cast<T>(0.0f);
        T db_v = static_cast<T>(0.0f);
        for (size_t j = 0; j < D; j += 1) {
            auto weight_v = affine ? weight.ptr<T>()[g * D + j] : static_cast<T>(1.0f);
            ds_v += ds[i * D + j] * weight_v;
            db_v += db[i * D + j] * weight_v;
        }
        auto c2 = (db_v * mean.ptr<T_ACC>()[i] - ds_v) * slice_std[i] * slice_std[i] *
                  slice_std[i] * s;
        auto c3 = -c2 * mean.ptr<T_ACC>()[i] - db_v * slice_std[i] * s;
        for (size_t j = 0; j < D; j++) {
            const int64_t c = g * D + j;
            auto weight_v = affine ? weight.ptr<T>()[c] : static_cast<T>(1.0f);
            auto c1 = slice_std[i] * weight_v;
            for (size_t k = 0; k < HxW; k++) {
                ddata.ptr<T>()[(i * D + j) * HxW + k] =
                        c1 * diff.ptr<T>()[(i * D + j) * HxW + k] +
                        c2 * data.ptr<T>()[(i * D + j) * HxW + k] + c3;
            }
        }
    }
    if (affine) {
        for (size_t i = 0; i < C; ++i) {
            dweight.ptr<T>()[i] = 0;
            dbias.ptr<T>()[i] = 0;
        }
        for (size_t i = 0; i < N * G; i++) {
            auto g = i % G;
            for (size_t j = 0; j < D; j++) {
                auto c = g * D + j;
                dweight.ptr<T>()[c] +=
                        (ds[i * D + j] - db[i * D + j] * mean.ptr<T_ACC>()[i]) *
                        slice_std[i];
            }
        }
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < C; j++) {
                dbias.ptr<T>()[j] += db[i * C + j];
            }
        }
    }
}

}  // namespace

namespace megdnn {
namespace naive {

size_t GroupNormBackwardImpl::get_workspace_in_bytes(
        const TensorLayout&, const TensorLayout& data, const TensorLayout&,
        const TensorLayout&, const TensorLayout& rstd, const TensorLayout&,
        const TensorLayout&, const TensorLayout&) {
    size_t N = data.shape[0];
    size_t C = data.shape[1];
    size_t G = rstd.shape[1];
    return get_workspace_bundle(N, C, G, data.dtype.size()).total_size_in_bytes();
}

WorkspaceBundle GroupNormBackwardImpl::get_workspace_bundle(
        size_t N, size_t C, size_t G, size_t dtype_size, void* raw_ptr) {
    return {raw_ptr,
            {N * C * dtype_size, N * C * dtype_size, N * G * dtype_size},
            handle()->alignment_requirement()};
}

void GroupNormForwardImpl::exec(
        _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
        _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
        _megdnn_workspace workspace) {
    check_exec(
            data.layout, weight.layout, bias.layout, dst.layout, mean.layout,
            rstd.layout, workspace.size);
#define cb(DType)                                                                \
    if (data.layout.dtype == DType()) {                                          \
        MEGDNN_DISPATCH_CPU_KERN_OPR(forward<typename DTypeTrait<DType>::ctype>( \
                data, weight, bias, dst, mean, rstd, param()));                  \
        return;                                                                  \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}

void GroupNormBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in data, _megdnn_tensor_in weight,
        _megdnn_tensor_in mean, _megdnn_tensor_in rstd, _megdnn_tensor_out ddata,
        _megdnn_tensor_out dweight, _megdnn_tensor_out dbias,
        _megdnn_workspace workspace) {
    check_exec(
            diff.layout, data.layout, weight.layout, mean.layout, rstd.layout,
            ddata.layout, dweight.layout, dbias.layout, workspace.size);
#define cb(DType)                                                                 \
    if (data.layout.dtype == DType()) {                                           \
        auto wbundle = get_workspace_bundle(                                      \
                data.layout.shape[0], data.layout.shape[1], rstd.layout.shape[1], \
                sizeof(DTypeTrait<DType>::ctype), workspace.raw_ptr);             \
        MEGDNN_DISPATCH_CPU_KERN_OPR(backward<typename DTypeTrait<DType>::ctype>( \
                diff, data, weight, mean, rstd, ddata, dweight, dbias, param(),   \
                wbundle));                                                        \
        return;                                                                   \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
