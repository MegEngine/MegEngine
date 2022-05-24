#include "algo.h"
#include "do_max_pooling_w4x4_s2x2.h"
#include "megdnn/opr_param_defs.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fallback_gi_pooling)

namespace megdnn {
namespace fallback {

WorkspaceBundle get_bundle(const PoolingImpl::PoolingKernSizeParam& param) {
    megdnn_assert(
            param.src_type.category() == DTypeCategory::FLOAT &&
            param.format == param::Pooling::Format::NCHW &&
            (param.mode == param::Pooling::Mode::MAX ||
             (param.mode == param::Pooling::Mode::AVERAGE && param.filter[0] == 3)) &&
            param.filter[0] == param.filter[1] &&
            (param.filter[0] == 3 || param.filter[1] == 5) && param.stride[0] == 2 &&
            param.stride[1] == 2 && param.isz[0] >= 2 && param.isz[1] >= 2);
    //! max pooling nxn stride 2
    auto IW = param.isz[1];
    auto OW = param.osz[1];

    // In order to process odd size filter,
    // Firstly, Store a row of the input separately by odd and even numbers
    // Then process them, get a row of the outputs
    // We need to store n rows of results
    SmallVector<size_t> needed_mem;
    for (size_t i = 0; i < param.filter[0]; ++i)
        needed_mem.push_back(OW * param.src_type.size());
    needed_mem.push_back((IW + 1) / 2 * param.src_type.size());
    needed_mem.push_back((IW + 1) / 2 * param.src_type.size());
    WorkspaceBundle ws(nullptr, needed_mem, 16);
    return ws;
}

bool PoolingImpl::AlgoGiFilterxModexStride1::usable(
        const PoolingKernSizeParam& param) const {
    auto SH = param.stride[0];
    auto SW = param.stride[1];
    auto FH = param.filter[0];
    auto FW = param.filter[1];

    bool avaible = param.src_type.category() == DTypeCategory::FLOAT &&
                   param.format == Param::Format::NCHW && SH == 1 && SW == 1 &&
                   FH == FW && (FH == 2 || FH == 3);
    bool is_mode_ok = (param.mode == Mode::MAX || param.mode == Mode::AVERAGE);
    return avaible && is_mode_ok;
}

void PoolingImpl::AlgoGiFilterxModexStride1::exec(const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];
    auto FH = param.filter[0];

    auto src_ptr = param.src_ptr;
    auto dst_ptr = param.dst_ptr;

#define DISPATCH_FUNC(Pooler, GiPooler, window, midout_type_id)                      \
    MIDOUT_BEGIN(                                                                    \
            megdnn_fallback_gi_pooling, midout_iv(0), midout_iv(midout_type_id),     \
            Pooler::MIDOUT_CASE_NUM, GiPooler::MIDOUT_CASE_NUM, window) {            \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr,                     \
                    src_dtype = param.src_type](size_t index, size_t) {              \
            size_t n = index / C;                                                    \
            size_t c = index % C;                                                    \
            do_pooling_compact<Pooler MEGDNN_COMMA GiPooler MEGDNN_COMMA window>(    \
                    static_cast<const typename Pooler::ctype*>(src_ptr.get_ptr()) +  \
                            n * C * IH * IW + c * IH * IW,                           \
                    static_cast<typename Pooler::ctype*>(dst_ptr.get_ptr()) +        \
                            n * C * OH * OW + c * OH * OW,                           \
                    src_dtype, IH, IW, OH, OW, PH, PW);                              \
        };                                                                           \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                       \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, run); \
    }                                                                                \
    MIDOUT_END()

#define DISPATCH_WINDOW(Pooler, GiPooler, dtype, ctype, comp_type, midout_type_id) \
    switch (FH) {                                                                  \
        case 2: {                                                                  \
            using _Pooler = Pooler<4, dtype, ctype, comp_type>;                    \
            using _GiPooler = GiPooler<4, dtype, ctype, comp_type>;                \
            DISPATCH_FUNC(_Pooler, _GiPooler, 2, midout_type_id);                  \
            break;                                                                 \
        }                                                                          \
        case 3: {                                                                  \
            using _Pooler = Pooler<9, dtype, ctype, comp_type>;                    \
            using _GiPooler = GiPooler<9, dtype, ctype, comp_type>;                \
            DISPATCH_FUNC(_Pooler, _GiPooler, 3, midout_type_id);                  \
            break;                                                                 \
        }                                                                          \
        default:                                                                   \
            megdnn_assert(0, "unsupport pooling filter size");                     \
            break;                                                                 \
    }

#define DISPATCH_MODE(dtype, ctype, comp_type, midout_type_id)                        \
    switch (param.mode) {                                                             \
        case Mode::MAX:                                                               \
            DISPATCH_WINDOW(                                                          \
                    MaxPooler, GiMaxPooler, dtype, ctype, comp_type, midout_type_id); \
            break;                                                                    \
        case Mode::AVERAGE:                                                           \
            DISPATCH_WINDOW(                                                          \
                    MeanInPooler, GiMeanPooler, dtype, ctype, comp_type,              \
                    midout_type_id);                                                  \
            break;                                                                    \
        default:                                                                      \
            megdnn_assert(0, "unsupport pooling mode");                               \
            break;                                                                    \
    }

    if (param.src_type == dtype::Float32{}) {
        DISPATCH_MODE(dt_float32, float, float, 0);
    }
#undef DISPATCH_FUNC
#undef DISPATCH_WINDOW
#undef DISPATCH_MODE
}
bool PoolingImpl::AlgoGiFilter2ModexStride2::usable(
        const PoolingKernSizeParam& param) const {
    auto SH = param.stride[0];
    auto SW = param.stride[1];
    auto FH = param.filter[0];
    auto FW = param.filter[1];

    bool avaible = param.src_type.category() == DTypeCategory::FLOAT &&
                   param.format == Param::Format::NCHW && FH == FW && SH == SW &&
                   FH == 2 && SH == 2;
    bool is_mode_ok = (param.mode == Mode::MAX || param.mode == Mode::AVERAGE);
    return avaible && is_mode_ok;
}

void PoolingImpl::AlgoGiFilter2ModexStride2::exec(const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];

    auto src_ptr = param.src_ptr;
    auto dst_ptr = param.dst_ptr;
#define DISPATCH_FUNC(Pooler, mode, midout_type_id)                                  \
    MIDOUT_BEGIN(                                                                    \
            megdnn_fallback_gi_pooling, midout_iv(1), midout_iv(midout_type_id),     \
            Pooler::MIDOUT_CASE_NUM) {                                               \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr,                     \
                    src_dtype = param.src_type](size_t index, size_t) {              \
            size_t n = index / C;                                                    \
            size_t c = index % C;                                                    \
            do_pooling_2x2<Pooler MEGDNN_COMMA mode>(                                \
                    static_cast<const typename Pooler::ctype*>(src_ptr.get_ptr()) +  \
                            n * C * IH * IW + c * IH * IW,                           \
                    static_cast<typename Pooler::ctype*>(dst_ptr.get_ptr()) +        \
                            n * C * OH * OW + c * OH * OW,                           \
                    src_dtype, IH, IW, OH, OW, PH, PW);                              \
        };                                                                           \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                       \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, run); \
    }                                                                                \
    MIDOUT_END()

#define DISPATCH_MODE(dtype, ctype, comp_type, midout_type_id)        \
    switch (param.mode) {                                             \
        case Mode::MAX: {                                             \
            using _Pooler = MaxPooler<4, dtype, ctype, comp_type>;    \
            DISPATCH_FUNC(_Pooler, Mode::MAX, midout_type_id);        \
            break;                                                    \
        }                                                             \
        case Mode::AVERAGE: {                                         \
            using _Pooler = MeanInPooler<4, dtype, ctype, comp_type>; \
            DISPATCH_FUNC(_Pooler, Mode::AVERAGE, midout_type_id);    \
            break;                                                    \
        }                                                             \
        default:                                                      \
            megdnn_assert(0, "unsupport pooling mode");               \
            break;                                                    \
    }

    if (param.src_type == dtype::Float32{}) {
        DISPATCH_MODE(dt_float32, float, float, 0);
    }
#undef DISPATCH_FUNC
#undef DISPATCH_PAD
#undef DISPATCH_MODE
}

bool PoolingImpl::AlgoGiFilter3MaxStride2::usable(
        const PoolingKernSizeParam& param) const {
    bool avaible = param.src_type.category() == DTypeCategory::FLOAT &&
                   param.format == Param::Format::NCHW && param.mode == Mode::MAX &&
                   param.filter[0] == 3 && param.filter[1] == 3 &&
                   param.stride[0] == 2 && param.stride[1] == 2 && param.isz[0] >= 2 &&
                   param.isz[1] >= 2;
    return avaible;
}

void PoolingImpl::AlgoGiFilter3MaxStride2::exec(const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];

    auto src_ptr = param.src_ptr;
    auto dst_ptr = param.dst_ptr;

#define DISPATCH_FUNC(type, func, midout_type_id)                                    \
    MIDOUT_BEGIN(                                                                    \
            megdnn_fallback_gi_pooling, midout_iv(2), midout_iv(midout_type_id)) {   \
        WorkspaceBundle wbundle = get_bundle(param);                                 \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr, wbundle = wbundle,  \
                    workspace_ptr = param.workspace<dt_byte>()](                     \
                           size_t index, size_t thread_id) {                         \
            auto ws = wbundle;                                                       \
            ws.set(workspace_ptr + ws.total_size_in_bytes() * thread_id);            \
            size_t n = index / C;                                                    \
            size_t c = index % C;                                                    \
            do_max_pooling_3x3_s2x2_float_gi(                                        \
                    static_cast<const type*>(src_ptr.get_ptr()) + n * C * IH * IW +  \
                            c * IH * IW,                                             \
                    static_cast<type*>(dst_ptr.get_ptr()) + n * C * OH * OW +        \
                            c * OH * OW,                                             \
                    IH, IW, OH, OW, PH, PW, ws);                                     \
        };                                                                           \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                       \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, run); \
    }                                                                                \
    MIDOUT_END();

    if (param.src_type == dtype::Float32{}) {
        DISPATCH_FUNC(float, float, 0);
    }
#undef DISPATCH_FUNC
}
bool PoolingImpl::AlgoGiFilter3AverageStride2::usable(
        const PoolingKernSizeParam& param) const {
    bool avaible = (param.src_type.category() == DTypeCategory::FLOAT) &&
                   param.format == Param::Format::NCHW && param.mode == Mode::AVERAGE &&
                   param.filter[0] == 3 && param.filter[1] == 3 &&
                   param.stride[0] == 2 && param.stride[1] == 2 && param.isz[0] >= 2 &&
                   param.isz[1] >= 2;
    return avaible;
}

void PoolingImpl::AlgoGiFilter3AverageStride2::exec(
        const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];

    auto src_ptr = param.src_ptr;
    auto dst_ptr = param.dst_ptr;

#define DISPATCH_FUNC(type, MEGDNN_SIMD_WIDTH, midout_type_id)                       \
    MIDOUT_BEGIN(                                                                    \
            megdnn_fallback_gi_pooling, midout_iv(3), midout_iv(midout_type_id)) {   \
        WorkspaceBundle wbundle = get_bundle(param);                                 \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr, wbundle = wbundle,  \
                    workspace_ptr = param.workspace<dt_byte>()](                     \
                           size_t index, size_t thread_id) {                         \
            auto ws = wbundle;                                                       \
            ws.set(workspace_ptr + ws.total_size_in_bytes() * thread_id);            \
            size_t n = index / C;                                                    \
            size_t c = index % C;                                                    \
            do_average_pooling_3x3_s2x2_gi(                                          \
                    static_cast<const type*>(src_ptr.get_ptr()) + n * C * IH * IW +  \
                            c * IH * IW,                                             \
                    static_cast<type*>(dst_ptr.get_ptr()) + n * C * OH * OW +        \
                            c * OH * OW,                                             \
                    IH, IW, OH, OW, PH, PW, ws, MEGDNN_SIMD_WIDTH);                  \
        };                                                                           \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                       \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, run); \
    }                                                                                \
    MIDOUT_END();
    if (param.src_type == dtype::Float32{}) {
        DISPATCH_FUNC(dt_float32, 4, 0);
    }
#undef DISPATCH_FUNC
}
bool PoolingImpl::AlgoGiFilter4MaxStride2::usable(
        const PoolingKernSizeParam& param) const {
    auto SH = param.stride[0];
    auto SW = param.stride[1];
    auto FH = param.filter[0];
    auto FW = param.filter[1];
    auto OH = param.osz[0], OW = param.osz[1];

    bool avaible = param.src_type.category() == DTypeCategory::FLOAT &&
                   param.format == Param::Format::NCHW && param.mode == Mode::MAX &&
                   FH == 4 && FW == 4 && SH == 2 && SW == 2 && OH >= 2 && OW >= 2;
    return avaible;
}

void PoolingImpl::AlgoGiFilter4MaxStride2::exec(const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];

    auto src_ptr = param.src_ptr;
    auto dst_ptr = param.dst_ptr;

#define DISPATCH_FUNC(type, func, midout_type_id)                                    \
    MIDOUT_BEGIN(                                                                    \
            megdnn_fallback_gi_pooling, midout_iv(4), midout_iv(midout_type_id)) {   \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr,                     \
                    src_dtype = param.src_type](size_t index, size_t) {              \
            size_t n = index / C;                                                    \
            size_t c = index % C;                                                    \
            do_max_pooling_w4x4_s2x2_##func##_gi(                                    \
                    static_cast<const type*>(src_ptr.get_ptr()) + n * C * IH * IW +  \
                            c * IH * IW,                                             \
                    static_cast<type*>(dst_ptr.get_ptr()) + n * C * OH * OW +        \
                            c * OH * OW,                                             \
                    src_dtype, IH, IW, OH, OW, PH, PW);                              \
        };                                                                           \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                       \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, run); \
    }                                                                                \
    MIDOUT_END();

    if (param.src_type == dtype::Float32{}) {
        DISPATCH_FUNC(float, float, 0);
    }
#undef DISPATCH_FUNC
}
bool PoolingImpl::AlgoGiFilter5MaxStride2::usable(
        const PoolingKernSizeParam& param) const {
    auto SH = param.stride[0];
    auto SW = param.stride[1];
    auto FH = param.filter[0];
    auto FW = param.filter[1];
    auto OH = param.osz[0], OW = param.osz[1];

    bool avaible = param.src_type.category() == DTypeCategory::FLOAT &&
                   param.format == Param::Format::NCHW && param.mode == Mode::MAX &&
                   FH == 5 && FW == 5 && SH == 2 && SW == 2 && OH >= 2 && OW >= 2;
    return avaible;
}

void PoolingImpl::AlgoGiFilter5MaxStride2::exec(const PoolingKernParam& param) const {
    auto IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto N = param.n, C = param.ic;
    auto PH = param.padding[0];
    auto PW = param.padding[1];

    auto src_ptr = param.src_ptr;
    auto dst_ptr = param.dst_ptr;

#define DISPATCH_FUNC(dtype, type, midout_type_id, MEGDNN_SIMD_WIDTH)                \
    MIDOUT_BEGIN(                                                                    \
            megdnn_fallback_gi_pooling, midout_iv(5), midout_iv(midout_type_id)) {   \
        WorkspaceBundle wbundle = get_bundle(param);                                 \
        auto run = [C, IH, IW, OH, OW, PH, PW, src_ptr, dst_ptr, wbundle = wbundle,  \
                    workspace_ptr = param.workspace<dt_byte>()](                     \
                           size_t index, size_t thread_id) {                         \
            auto ws = wbundle;                                                       \
            ws.set(workspace_ptr + ws.total_size_in_bytes() * thread_id);            \
            size_t n = index / C;                                                    \
            size_t c = index % C;                                                    \
            do_max_pooling_w5x5_s2x2_gi<dtype>(                                      \
                    static_cast<const type*>(src_ptr.get_ptr()) + n * C * IH * IW +  \
                            c * IH * IW,                                             \
                    static_cast<type*>(dst_ptr.get_ptr()) + n * C * OH * OW +        \
                            c * OH * OW,                                             \
                    IH, IW, OH, OW, PH, PW, ws, MEGDNN_SIMD_WIDTH);                  \
        };                                                                           \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                       \
                static_cast<::megdnn::naive::HandleImpl*>(param.handle), N* C, run); \
    }                                                                                \
    MIDOUT_END();

    if (param.src_type == dtype::Float32{}) {
        DISPATCH_FUNC(dt_float32, float, 0, 4);
    }
#undef DISPATCH_FUNC
}

}  // namespace fallback
}  // namespace megdnn
// vim: syntax=cpp.doxygen
