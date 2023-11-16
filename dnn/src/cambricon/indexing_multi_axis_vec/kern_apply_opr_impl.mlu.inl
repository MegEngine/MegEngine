#ifndef KERN_APPLY_OPR_OPR
#ifndef KERN_APPLY_OPR_OPR_INCR
#error "must define KERN_APPLY_OPR_OPR or KERN_APPLY_OPR_OPR_INCR"
#endif
#endif

#include <type_traits>
#include "./kern.mlu.h"
#include "megdnn/internal/defs.h"
#include "mlu.h"

using namespace megdnn;
using namespace cambricon;
using namespace indexing_multi_axis_vec;

namespace {

struct OprAtomicIncrMlu {
    template <typename ctype>
    __mlu_func__ static void apply(ctype& data, ctype value) {
#ifdef KERN_APPLY_OPR_OPR_INCR
        atomicAdd(&data, value);
#endif
    }
};

template <typename ctype, int ndim, class Opr>
__mlu_entry__ void kapply_opr(ApplyOprParam<ndim> param) {
    auto oidx = taskId;
    ctype* data_ctype = static_cast<ctype*>(param.data);
    ctype* value_ctype = static_cast<ctype*>(param.value);

    while (oidx < param.tot_size) {
        int offset = 0, coidx = oidx;
        int idx_flat = 0;
#pragma unroll
        for (int i = ndim - 1; i >= 0; --i) {
            int next_coidx, ax_idx;
            // get the flatten index of the offset stored in param.offset_base
            // [..., indexed_axes... |, ...]
            if (i + 1 == param.idx_axis_end) {
                idx_flat = coidx;
            }
            // [... |, indexed_axes..., ...]
            if (i + 1 == param.idx_axis) {
                idx_flat -= coidx * param.idx_nelems;
            }
            // generate offset of the non-indexed axes
            // shape[i] was storaged at shape[i-1]
            if (i) {
                next_coidx = coidx / param.value_ly_on_data.shape[i - 1];
                ax_idx = coidx - (next_coidx * param.value_ly_on_data.shape[i - 1]);
                coidx = next_coidx;
            } else {
                ax_idx = coidx;
            }
            offset += param.value_ly_on_data.stride[i] * ax_idx;
        }
        // offset from index, which was generated before
        offset += param.offset_base[idx_flat];

        if (std::is_same<
                    Opr,
                    megdnn::cambricon::indexing_multi_axis_vec::OprIncrCommon>::value) {
            OprAtomicIncrMlu::apply(
                    data_ctype[offset], value_ctype[oidx * param.value_stride]);
        } else {
            //! no error handle.
            return;
        }
        oidx += taskDim;
    }
}
}  // namespace

template <typename ctype, int ndim, class Opr>
void indexing_multi_axis_vec::apply_opr(
        const ApplyOprParam<ndim>& param, cnrtQueue_t queue) {
    void (*kptr)(ApplyOprParam<ndim>) = kapply_opr<ctype, ndim, Opr>;
    cnrtDim3_t dim{
            static_cast<unsigned int>(param.cluster_count) *
                    static_cast<unsigned int>(param.core_per_cluster),
            1, 1};
    cnrtFunctionType_t function_type = CNRT_FUNC_TYPE_BLOCK;
    (*kptr)<<<dim, function_type, queue>>>(param);
}

template <int ndim, class Opr>
void indexing_multi_axis_vec::apply_opr(
        const ApplyOprParam<ndim>& param, cnnlDataType_t cnnl_data_type,
        cnrtQueue_t queue) {
    switch (cnnl_data_type) {
        case CNNL_DTYPE_FLOAT: {
            apply_opr<float, ndim, Opr>(param, queue);
            break;
        };
        case CNNL_DTYPE_HALF: {
            apply_opr<half, ndim, Opr>(param, queue);
            break;
        };
        case CNNL_DTYPE_BFLOAT16: {
#if __bang_arch >= 592
            apply_opr<bfloat16_t, ndim, Opr>(param, queue);
#endif
            break;
        };
        case CNNL_DTYPE_INT32: {
            apply_opr<int32_t, ndim, Opr>(param, queue);
            break;
        };
        case CNNL_DTYPE_INT16: {
            apply_opr<int16_t, ndim, Opr>(param, queue);
            break;
        };
        case CNNL_DTYPE_INT8: {
            apply_opr<int8_t, ndim, Opr>(param, queue);
            break;
        };
        case CNNL_DTYPE_UINT8: {
            apply_opr<uint8_t, ndim, Opr>(param, queue);
            break;
        };
        case CNNL_DTYPE_BOOL: {
            apply_opr<bool, ndim, Opr>(param, queue);
            break;
        };
        default:
            return;
    };
}

namespace megdnn {
namespace cambricon {
namespace indexing_multi_axis_vec {

#ifdef KERN_APPLY_OPR_OPR_INCR
#define INST(_ndim, _ctype)                                          \
    template void apply_opr<_ctype, _ndim, KERN_APPLY_OPR_OPR_INCR>( \
            const ApplyOprParam<_ndim>&, cnrtQueue_t);
#define cb0(_dtype) MEGDNN_FOREACH_TENSOR_NDIM(INST, _dtype)
cb0(float) cb0(half) cb0(bool)
#undef cb0
#undef INST

#define INST(_ndim)                                          \
    template void apply_opr<_ndim, KERN_APPLY_OPR_OPR_INCR>( \
            const ApplyOprParam<_ndim>&, cnnlDataType_t, cnrtQueue_t);
        MEGDNN_FOREACH_TENSOR_NDIM(INST)
#undef INST
#endif

}  // namespace indexing_multi_axis_vec
}  // namespace cambricon
}  // namespace megdnn
