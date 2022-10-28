#include "src/fallback/softmax/opr_impl.h"
#include <cstring>
#include <numeric>
#include "src/fallback/elemwise/gi_impl/gi_mathfun.h"
#include "src/naive/handle.h"

namespace megdnn {
namespace fallback {

static void do_softmax(
        const float* sptr, float* dptr, size_t A, size_t B, size_t C,
        _megdnn_workspace workspace) {
    constexpr auto float_min = std::numeric_limits<float>::min();
    constexpr auto step = GI_SIMD_LEN_BYTE / sizeof(float);
    // TODO: When C=2,3,4..., src_ptr span is relatively large, the performance may
    // be poor

    if (C != 1) {
        WorkspaceBundle workspace_bundle{
                workspace.raw_ptr, {A * C * sizeof(float), A * C * sizeof(float)}};
        float* max = workspace_bundle.get_workspace(0).raw_ptr->as<float>();
        GI_FLOAT32_t v_max = GiBroadcastFloat32(float_min);
        size_t i = 0;
        for (; i + step <= A * C; i += step)
            GiStoreFloat32(max + i, v_max);
        for (; i < A * C; i++)
            max[i] = float_min;

        for (size_t a = 0; a < A; a++) {
            for (size_t b = 0; b < B; b++) {
                auto max_ptr = max + a * C;
                auto limit = max_ptr + C;
                auto src_ptr = sptr + a * B * C + b * C;

                for (; max_ptr + step <= limit; max_ptr += step, src_ptr += step) {
                    GI_FLOAT32_t v_p = GiLoadFloat32(src_ptr);
                    GI_FLOAT32_t v_max = GiLoadFloat32(max_ptr);
                    v_max = GiMaximumFloat32(v_max, v_p);
                    GiStoreFloat32(max_ptr, v_max);
                }
                for (; max_ptr < limit; ++max_ptr, ++src_ptr) {
                    *max_ptr = std::max(*src_ptr, *max_ptr);
                }
            }
        }

        float* sum = workspace_bundle.get_workspace(1).raw_ptr->as<float>();
        memset(sum, 0, A * C * sizeof(float));
        for (size_t a = 0; a < A; a++) {
            for (size_t b = 0; b < B; b++) {
                auto max_ptr = max + a * C;
                auto limit = max_ptr + C;
                auto sum_ptr = sum + a * C;
                auto src_ptr = sptr + a * B * C + C * b;
                auto dst_ptr = dptr + a * B * C + C * b;
                for (; max_ptr + step <= limit; max_ptr += step, sum_ptr += step,
                                                src_ptr += step, dst_ptr += step) {
                    GI_FLOAT32_t v_p = GiLoadFloat32(src_ptr);
                    GI_FLOAT32_t v_max = GiLoadFloat32(max_ptr);
                    GI_FLOAT32_t v_sum = GiLoadFloat32(sum_ptr);
                    v_p = GiExpPsFloat32(GiSubtractFloat32(v_p, v_max));
                    v_sum = GiAddFloat32(v_p, v_sum);
                    GiStoreFloat32(dst_ptr, v_p);
                    GiStoreFloat32(sum_ptr, v_sum);
                }
                for (; max_ptr < limit; ++max_ptr, ++sum_ptr, ++src_ptr, ++dst_ptr) {
                    *dst_ptr = exp(*src_ptr - *max_ptr);
                    *sum_ptr += *dst_ptr;
                }
            }
        }

        for (size_t a = 0; a < A; a++) {
            for (size_t b = 0; b < B; b++) {
                auto sum_ptr = sum + a * C;
                auto limit = sum_ptr + C;
                auto dst_ptr = dptr + a * B * C + C * b;
                for (; sum_ptr + step <= limit; sum_ptr += step, dst_ptr += step) {
                    GI_FLOAT32_t v_p = GiLoadFloat32(dst_ptr);
                    GI_FLOAT32_t v_sum = GiLoadFloat32(sum_ptr);
                    v_p = GiDivideFloat32(v_p, v_sum);
                    GiStoreFloat32(dst_ptr, v_p);
                }
                for (; sum_ptr < limit; ++sum_ptr, ++dst_ptr)
                    *dst_ptr = *dst_ptr / *sum_ptr;
            }
        }
    } else {
        for (size_t a = 0; a < A; a++) {
            auto max = float_min;
            {
                auto src_ptr = sptr + a * B;
                auto limit = src_ptr + B;
                GI_FLOAT32_t v_max = GiBroadcastFloat32(max);

                for (; src_ptr + step <= limit; src_ptr += step) {
                    GI_FLOAT32_t v_p = GiLoadFloat32(src_ptr);
                    v_max = GiMaximumFloat32(v_max, v_p);
                }
                max = std::max(max, GiReduceMaxNanFloat32(v_max));
                for (; src_ptr < limit; ++src_ptr) {
                    max = std::max(*src_ptr, max);
                }
            }

            auto sum = 0.f;
            {
                auto src_ptr = sptr + a * B;
                auto limit = src_ptr + B;
                auto dst_ptr = dptr + a * B;
                GI_FLOAT32_t v_sum = GiZeroFloat32();
                GI_FLOAT32_t v_max = GiBroadcastFloat32(max);

                for (; src_ptr + step <= limit; src_ptr += step, dst_ptr += step) {
                    GI_FLOAT32_t v_p = GiLoadFloat32(src_ptr);
                    v_p = GiExpPsFloat32(GiSubtractFloat32(v_p, v_max));
                    GiStoreFloat32(dst_ptr, v_p);
                    v_sum = GiAddFloat32(v_sum, v_p);
                }
                sum += GiReduceAddFloat32(v_sum);
                for (; src_ptr < limit; ++src_ptr, ++dst_ptr) {
                    *dst_ptr = exp(*src_ptr - max);
                    sum += *dst_ptr;
                }
            }
            {
                auto dst_ptr = dptr + a * B;
                auto limit = dst_ptr + B;
                sum = 1 / sum;
                GI_FLOAT32_t v_sum = GiBroadcastFloat32(sum);
                for (; dst_ptr + step <= limit; dst_ptr += step) {
                    GI_FLOAT32_t v_p = GiLoadFloat32(dst_ptr);
                    v_p = GiMultiplyFloat32(v_p, v_sum);
                    GiStoreFloat32(dst_ptr, v_p);
                }
                for (; dst_ptr < limit; ++dst_ptr) {
                    *dst_ptr *= sum;
                }
            }
        }
    }
}

void SoftmaxForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    auto axis = param().axis;
    if (axis < 0)
        axis += src.layout.ndim;
    megdnn_assert(axis >= 0);
    check_exec(src.layout, dst.layout, workspace.size);

    if (!usable(src.layout)) {
        naive::SoftmaxForwardImpl::exec(src, dst, workspace);
        return;
    }

    typedef DTypeTrait<dtype::Float32>::ctype Float32;
    auto sptr = src.ptr<Float32>();
    auto dptr = dst.ptr<Float32>();

    size_t A, B, C;
    reduce::get_ABC(src.layout, A, B, C, axis);
    MEGDNN_DISPATCH_CPU_KERN_OPR(do_softmax(sptr, dptr, A, B, C, workspace));
}

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
