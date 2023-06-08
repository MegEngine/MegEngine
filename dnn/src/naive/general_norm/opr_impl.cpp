#include "src/naive/general_norm/opr_impl.h"
#include <algorithm>
#include "src/common/reduce_helper.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;

namespace {

using Param = megdnn::GeneralNorm::Param;

template <typename T, typename T_ACC = float>
void forward(
        _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
        _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
        const Param& param) {
    float eps = param.eps;
    bool affine = param.affine;
    size_t A, B, C;
    megdnn::reduce::get_ABC(data.layout, A, B, C, param.axis_start, param.axis_end);

    for (size_t a = 0; a < A; ++a)
        for (size_t c = 0; c < C; ++c) {
            double slice_sum = 0.0f;
            double slice_sum_sqr = 0.0f;
            for (size_t b = 0; b < B; b++) {
                auto value = data.ptr<T>()[a * B * C + b * C + c];
                slice_sum += value;
                slice_sum_sqr += value * value;
            }
            T_ACC slice_mean = static_cast<T_ACC>(slice_sum / B);
            T_ACC slice_std = static_cast<T_ACC>(
                    sqrt(std::abs(slice_sum_sqr / B - slice_mean * slice_mean) + eps));

            for (size_t b = 0; b < B; b++) {
                dst.ptr<T>()[a * B * C + b * C + c] = static_cast<T>(
                        (data.ptr<T>()[a * B * C + b * C + c] - slice_mean) /
                        slice_std);
                if (affine) {
                    dst.ptr<T>()[a * B * C + b * C + c] = static_cast<T>(
                            dst.ptr<T>()[a * B * C + b * C + c] * weight.ptr<T>()[b] +
                            bias.ptr<T>()[b]);
                }
            }
            mean.ptr<T_ACC>()[a * C + c] = static_cast<T_ACC>(slice_mean);
            rstd.ptr<T_ACC>()[a * C + c] = static_cast<T_ACC>(1.0 / slice_std);
        }
}

template <typename T, typename T_ACC = float>
void backward(
        _megdnn_tensor_in diff, _megdnn_tensor_in data, _megdnn_tensor_in weight,
        _megdnn_tensor_in mean, _megdnn_tensor_in rstd, _megdnn_tensor_out ddata,
        _megdnn_tensor_out dweight, _megdnn_tensor_out dbias, const Param& param) {
    bool affine = param.affine;
    size_t A, B, C;
    megdnn::reduce::get_ABC(data.layout, A, B, C, param.axis_start, param.axis_end);

    if (affine) {
        for (size_t b = 0; b < B; ++b) {
            dweight.ptr<T>()[b] = 0;
            dbias.ptr<T>()[b] = 0;
        }

        for (size_t a = 0; a < A; ++a)
            for (size_t c = 0; c < C; ++c) {
                for (size_t b = 0; b < B; ++b) {
                    dweight.ptr<T>()[b] += (data.ptr<T>()[a * B * C + b * C + c] -
                                            mean.ptr<T_ACC>()[a * C + c]) *
                                           rstd.ptr<T_ACC>()[a * C + c] *
                                           diff.ptr<T>()[a * B * C + b * C + c];

                    dbias.ptr<T>()[b] += diff.ptr<T>()[a * B * C + b * C + c];
                }
            }
    }

    for (size_t a = 0; a < A; ++a)
        for (size_t c = 0; c < C; ++c) {
            double ds = 0.0f;
            double db = 0.0f;
            T_ACC atmp = static_cast<T_ACC>(0.0f);
            T_ACC btmp = static_cast<T_ACC>(0.0f);
            T_ACC ctmp = static_cast<T_ACC>(0.0f);

            for (size_t b = 0; b < B; ++b) {
                auto value = data.ptr<T>()[a * B * C + b * C + c];
                auto diff_v = diff.ptr<T>()[a * B * C + b * C + c];
                auto weight_v = affine ? weight.ptr<T>()[b] : static_cast<T>(1.0f);
                db += diff_v * weight_v;
                ds += diff_v * value * weight_v;
            }

            atmp = rstd.ptr<T_ACC>()[a * C + c];
            btmp = (db * mean.ptr<T_ACC>()[a * C + c] - ds) * atmp * atmp * atmp / B;
            ctmp = -btmp * mean.ptr<T_ACC>()[a * C + c] - db * atmp / B;

            for (size_t b = 0; b < B; b++) {
                auto weight_v = affine ? weight.ptr<T>()[b] : static_cast<T>(1.0f);
                ddata.ptr<T>()[a * B * C + b * C + c] =
                        diff.ptr<T>()[a * B * C + b * C + c] * atmp * weight_v +
                        data.ptr<T>()[a * B * C + b * C + c] * btmp + ctmp;
            }
        }
}

}  // namespace

namespace megdnn {
namespace naive {

void GeneralNormForwardImpl::exec(
        _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
        _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
        _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
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
#else
    __builtin_trap();
#endif
}

void GeneralNormBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in data, _megdnn_tensor_in weight,
        _megdnn_tensor_in mean, _megdnn_tensor_in rstd, _megdnn_tensor_out ddata,
        _megdnn_tensor_out dweight, _megdnn_tensor_out dbias,
        _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    check_exec(
            diff.layout, data.layout, weight.layout, mean.layout, rstd.layout,
            ddata.layout, dweight.layout, dbias.layout, workspace.size);
#define cb(DType)                                                                 \
    if (data.layout.dtype == DType()) {                                           \
        MEGDNN_DISPATCH_CPU_KERN_OPR(backward<typename DTypeTrait<DType>::ctype>( \
                diff, data, weight, mean, rstd, ddata, dweight, dbias, param())); \
        return;                                                                   \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
#else
    __builtin_trap();
#endif
}

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
