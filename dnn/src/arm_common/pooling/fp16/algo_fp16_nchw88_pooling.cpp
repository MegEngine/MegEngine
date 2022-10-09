#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "midout.h"
#include "src/arm_common/pooling/algo.h"
#include "src/arm_common/pooling/fp16/kern_fp16_nchw88_pooling.h"

MIDOUT_DECL(megdnn_arm_common_fp16_nchw88_pooling)

namespace megdnn {
namespace arm_common {
bool PoolingImpl::AlgoFilterxModexStridexNCHW88::usable(
        const PoolingKernSizeParam& param) const {
    uint32_t sh = param.stride[0];
    uint32_t sw = param.stride[1];
    uint32_t fh = param.filter[0];
    uint32_t fw = param.filter[1];

    bool usable = param.src_type.enumv() == DTypeEnum::Float16 &&
                  param.format == param::Pooling::Format::NCHW88 &&
                  (param.mode == PoolingBase::Mode::MAX ||
                   param.mode == PoolingBase::Mode::AVERAGE) &&
                  fh == fw && sh == sw;
    bool size_ok =
            (((fh == 2 || fh == 3 || fh == 4 || fh == 5) && (sh == 1 || sh == 2)) ||
             ((fh == 9 || fh == 13) && (sh == 1)));
    return usable && size_ok;
}

void PoolingImpl::AlgoFilterxModexStridexNCHW88::exec(
        const PoolingKernParam& param) const {
    int ih = param.isz[0];
    int iw = param.isz[1];
    int oh = param.osz[0];
    int ow = param.osz[1];
    int n = param.n;
    int ic = param.ic;
    int ph = param.padding[0];
    int pw = param.padding[1];
    int sh = param.stride[0];
    int fh = param.filter[0];

    auto src = param.src_ptr;
    auto dst = param.dst_ptr;

#define DISPATCH_FUNC(filter, stride, mode)                                            \
    MIDOUT_BEGIN(                                                                      \
            megdnn_arm_common_fp16_nchw88_pooling, midout_iv(0),                       \
            midout_iv(#filter #stride #mode##_hash)) {                                 \
        auto run = [=](size_t index, size_t) {                                         \
            const int c_idx = index;                                                   \
            pooling_fp16_nchw88<filter, stride, mode>(                                 \
                    static_cast<const __fp16*>(src.get_ptr()) + c_idx * ih * iw * 8,   \
                    static_cast<__fp16*>(dst.get_ptr()) + c_idx * oh * ow * 8, ih, iw, \
                    oh, ow, ph, pw);                                                   \
        };                                                                             \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                         \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), n* ic, run);  \
    }                                                                                  \
    MIDOUT_END();

#define DISPATCH_MODE(filter, stride)                                               \
    switch (param.mode) {                                                           \
        case PoolingBase::Mode::MAX:                                                \
            DISPATCH_FUNC(filter, stride, PoolingBase::Mode::MAX);                  \
            break;                                                                  \
        case PoolingBase::Mode::AVERAGE:                                            \
            DISPATCH_FUNC(filter, stride, PoolingBase::Mode::AVERAGE);              \
            break;                                                                  \
        default:                                                                    \
            megdnn_assert(0, "invalid mode %u", static_cast<uint32_t>(param.mode)); \
    }

#define DISPATCH_STRIDE(filter)                                                        \
    switch (sh) {                                                                      \
        case 1:                                                                        \
            DISPATCH_MODE(filter, 1);                                                  \
            break;                                                                     \
        case 2:                                                                        \
            DISPATCH_MODE(filter, 2);                                                  \
            break;                                                                     \
        default:                                                                       \
            megdnn_assert(                                                             \
                    0,                                                                 \
                    "Invalid stride %d. When the filter size is 2, 3, 4 or 5, stride " \
                    "can only be 1 or 2.",                                             \
                    sh);                                                               \
    }

#define DISPATCH_STRIDE1(filter)                                                  \
    switch (sh) {                                                                 \
        case 1:                                                                   \
            DISPATCH_MODE(filter, 1);                                             \
            break;                                                                \
        default:                                                                  \
            megdnn_assert(                                                        \
                    0,                                                            \
                    "Invalid stride %d. When the filter size is 9 or 13, stride " \
                    "can only be 1.",                                             \
                    sh);                                                          \
    }

#define DISPATCH_FILTER()         \
    switch (fh) {                 \
        case 2:                   \
            DISPATCH_STRIDE(2);   \
            break;                \
        case 3:                   \
            DISPATCH_STRIDE(3);   \
            break;                \
        case 4:                   \
            DISPATCH_STRIDE(4);   \
            break;                \
        case 5:                   \
            DISPATCH_STRIDE(5);   \
            break;                \
        case 9:                   \
            DISPATCH_STRIDE1(9);  \
            break;                \
        case 13:                  \
            DISPATCH_STRIDE1(13); \
            break;                \
    }

    DISPATCH_FILTER();
}

}  // namespace arm_common
}  // namespace megdnn
#endif