#include "src/common/elemwise/kern_defs.cuh"
#include "src/common/utils.h"

#include "megdnn/oprs.h"
#include "megdnn/tensor_format.h"

#include "midout.h"
MIDOUT_DECL(megdnn_common_elemwise)
//! this tag will be used at tools/gen_header_for_bin_reduce.py
//! please do not modify it
MIDOUT_DECL(megdnn_common_elemwise_mode)

#include <mutex>
#include <vector>

using namespace megdnn;

namespace {
class FormatDeducer {
    const TensorFormat m_default;
    TensorFormat m_result = m_default;

public:
    inline void feed(TensorFormat cur);
    bool is_default(TensorFormat f) const { return f == m_default; }
    TensorFormat get() const { return m_result; }
};
}  // anonymous namespace

using Mode = param::Elemwise::Mode;
using ModeTrait = ElemwiseForward::ModeTrait;

const ModeTrait& ModeTrait::from_mode(Mode mode) {
    static DNN_MUTEX mtx;
    static std::vector<ModeTrait> traits;

    MEGDNN_LOCK_GUARD(mtx);

    if (traits.empty()) {
        auto get = [&](Mode m) -> ModeTrait& {
            auto im = static_cast<size_t>(m);
            if (im >= traits.size())
                traits.resize(im + 1);
            return traits[im];
        };

#define cb(_m)                                                  \
    MIDOUT_BEGIN(megdnn_common_elemwise, midout_iv(Mode::_m)) { \
        get(Mode::_m).allow_int = true;                         \
    }                                                           \
    MIDOUT_END();
        MEGDNN_FOREACH_ELEMWISE_MODE_UNARY_INT(cb);
        MEGDNN_FOREACH_ELEMWISE_MODE_BINARY_INT(cb);
        MEGDNN_FOREACH_ELEMWISE_MODE_TERNARY_INT(cb);
#undef cb

#define cb(_m)                                                  \
    MIDOUT_BEGIN(megdnn_common_elemwise, midout_iv(Mode::_m)) { \
        get(Mode::_m).allow_float = true;                       \
    }                                                           \
    MIDOUT_END();
        MEGDNN_FOREACH_ELEMWISE_MODE_UNARY_FLOAT(cb);
        MEGDNN_FOREACH_ELEMWISE_MODE_BINARY_FLOAT(cb);
        MEGDNN_FOREACH_ELEMWISE_MODE_TERNARY_FLOAT(cb);
#undef cb

#define cb(_m)                                                  \
    MIDOUT_BEGIN(megdnn_common_elemwise, midout_iv(Mode::_m)) { \
        get(Mode::_m).allow_bool = true;                        \
    }                                                           \
    MIDOUT_END();
        MEGDNN_FOREACH_ELEMWISE_MODE_UNARY_BOOL(cb);
        MEGDNN_FOREACH_ELEMWISE_MODE_BINARY_BOOL(cb);
#undef cb

#define cb(_m)                                                  \
    MIDOUT_BEGIN(megdnn_common_elemwise, midout_iv(Mode::_m)) { \
        auto&& t = get(Mode::_m);                               \
        t.arity = _a;                                           \
        t.name = (#_m);                                         \
    }                                                           \
    MIDOUT_END();
#define _a 1
        MEGDNN_FOREACH_ELEMWISE_MODE_UNARY_FLOAT(cb);
        MEGDNN_FOREACH_ELEMWISE_MODE_UNARY_INT(cb);
        MEGDNN_FOREACH_ELEMWISE_MODE_UNARY_BOOL(cb);
#undef _a
#define _a 2
        MEGDNN_FOREACH_ELEMWISE_MODE_BINARY_FLOAT(cb);
        MEGDNN_FOREACH_ELEMWISE_MODE_BINARY_INT(cb);
        MEGDNN_FOREACH_ELEMWISE_MODE_BINARY_BOOL(cb);
#undef _a
#define _a 3
        MEGDNN_FOREACH_ELEMWISE_MODE_TERNARY_FLOAT(cb);
        MEGDNN_FOREACH_ELEMWISE_MODE_TERNARY_INT(cb);
#undef _a
#undef cb

#define FUSE(_m, _arity)                                        \
    MIDOUT_BEGIN(megdnn_common_elemwise, midout_iv(Mode::_m)) { \
        auto&& t = get(Mode::_m);                               \
        t.allow_int = true;                                     \
        t.allow_float = true;                                   \
        t.allow_bool = true;                                    \
        t.arity = _arity;                                       \
        t.name = (#_m);                                         \
    }                                                           \
    MIDOUT_END();
        FUSE(FUSE_MUL_ADD3, 3);
        FUSE(FUSE_MUL_ADD4, 4);
#undef FUSE

#define COMM_CB(_m)                                              \
    MIDOUT_BEGIN(megdnn_common_elemwise, midout_iv(Mode::_m)) {  \
        traits.at(static_cast<int>(Mode::_m)).commutable = true; \
    }                                                            \
    MIDOUT_END()
#define COMM(_m) MEGDNN_ELEMWISE_MODE_ENABLE(_m, COMM_CB)

        COMM(ADD);
        COMM(FUSE_ADD_RELU);
        COMM(FUSE_ADD_SIGMOID);
        COMM(FUSE_ADD_TANH);
        COMM(MUL);
        COMM(RMULH);
        COMM(MAX);
        COMM(MIN);
        COMM(EQ);
        COMM(LOG_SUM_EXP);

#undef COMM
#undef COMM_CB

#if MEGDNN_ELEMWISE_MODE_ENABLE_ALL
        for (auto&& i : traits) {
            megdnn_assert(
                    i.arity && (i.allow_int || i.allow_float || i.allow_bool) &&
                    (!i.commutable || i.arity == 2));
        }
#else
#pragma message "elemwise mode stripped"
#endif
    }

    auto&& ret = traits.at(static_cast<int>(mode));
#if !MEGDNN_ELEMWISE_MODE_ENABLE_ALL
    megdnn_assert(ret.arity);
#endif

    //! Some DNN backend OPRS will use proxy OPRS. For example, softmax@cpu Naive imp
    //! will call elemwise OPR. In the model dump stage, we have no information about
    //! this logic, which will lead to the loss of elemwise mode. As a solution, we
    //! record the elemwise mode information by adding the 'midout' case flag in the run
    //! stage.
#define CB_MODE(mode)                                                              \
    case mode:                                                                     \
        MIDOUT_BEGIN(megdnn_common_elemwise_mode, midout_iv(mode)) { return ret; } \
        MIDOUT_END();                                                              \
        break;

    switch (mode) {
        CB_MODE(Mode::RELU);
        CB_MODE(Mode::ABS);
        CB_MODE(Mode::ACOS);
        CB_MODE(Mode::ASIN);
        CB_MODE(Mode::CEIL);
        CB_MODE(Mode::COS);
        CB_MODE(Mode::EXP);
        CB_MODE(Mode::EXPM1);
        CB_MODE(Mode::FLOOR);
        CB_MODE(Mode::LOG);
        CB_MODE(Mode::LOG1P);
        CB_MODE(Mode::NEGATE);
        CB_MODE(Mode::SIGMOID);
        CB_MODE(Mode::SIN);
        CB_MODE(Mode::TANH);
        CB_MODE(Mode::ABS_GRAD);
        CB_MODE(Mode::ADD);
        CB_MODE(Mode::FLOOR_DIV);
        CB_MODE(Mode::MAX);
        CB_MODE(Mode::MIN);
        CB_MODE(Mode::MOD);
        CB_MODE(Mode::MUL);
        CB_MODE(Mode::POW);
        CB_MODE(Mode::SIGMOID_GRAD);
        CB_MODE(Mode::SUB);
        CB_MODE(Mode::SWITCH_GT0);
        CB_MODE(Mode::TANH_GRAD);
        CB_MODE(Mode::TRUE_DIV);
        CB_MODE(Mode::LOG_SUM_EXP);
        CB_MODE(Mode::LT);
        CB_MODE(Mode::LEQ);
        CB_MODE(Mode::EQ);
        CB_MODE(Mode::SHL);
        CB_MODE(Mode::SHR);
        CB_MODE(Mode::COND_LEQ_MOV);
        CB_MODE(Mode::FUSE_MUL_ADD3);
        CB_MODE(Mode::FUSE_MUL_ADD4);
        CB_MODE(Mode::FUSE_ADD_RELU);
        CB_MODE(Mode::FUSE_ADD_SIGMOID);
        CB_MODE(Mode::FUSE_ADD_TANH);
        CB_MODE(Mode::FAST_TANH);
        CB_MODE(Mode::FAST_TANH_GRAD);
        CB_MODE(Mode::ROUND);
        CB_MODE(Mode::RMULH);
        CB_MODE(Mode::ATAN2);
        CB_MODE(Mode::ERF);
        CB_MODE(Mode::ERFINV);
        CB_MODE(Mode::ERFC);
        CB_MODE(Mode::ERFCINV);
        CB_MODE(Mode::H_SWISH);
        CB_MODE(Mode::H_SWISH_GRAD);
        CB_MODE(Mode::FUSE_ADD_H_SWISH);
        CB_MODE(Mode::NOT);
        CB_MODE(Mode::AND);
        CB_MODE(Mode::OR);
        CB_MODE(Mode::XOR);
        CB_MODE(Mode::SILU);
        CB_MODE(Mode::SILU_GRAD);
        CB_MODE(Mode::GELU);
        CB_MODE(Mode::GELU_GRAD);
        CB_MODE(Mode::COND_LT_MOV);
        CB_MODE(Mode::SINH);
        CB_MODE(Mode::COSH);
        CB_MODE(Mode::ASINH);
        CB_MODE(Mode::ACOSH);
        CB_MODE(Mode::ATANH);
        CB_MODE(Mode::TAN);
        CB_MODE(Mode::ASINH_GRAD);
        CB_MODE(Mode::ACOSH_GRAD);
        CB_MODE(Mode::ATANH_GRAD);
        CB_MODE(Mode::PRELU);
        CB_MODE(Mode::PRELU_GRAD);
        CB_MODE(Mode::CLIP);
        CB_MODE(Mode::SOFTPLUS);
        CB_MODE(Mode::SOFTPLUS_GRAD);
        CB_MODE(Mode::RELU6);
        CB_MODE(Mode::RELU6_GRAD);
        CB_MODE(Mode::HSIGMOID);
        CB_MODE(Mode::HSIGMOID_GRAD);
        CB_MODE(Mode::LOGSIGMOID);
        CB_MODE(Mode::SQRT);
        CB_MODE(Mode::SQUARE);
        CB_MODE(Mode::SIGN);
        default:
            megdnn_assert(
                    0,
                    "code issue happened!!, please add new elemwise to switch mode.");
            return ret;

#undef CB_MODE
    }

    return ret;
}

void ElemwiseForward::deduce_shape(const TensorShapeArray& src, TensorShape& dst) {
    auto err = [&]() {
        std::string msg("bad input shape for polyadic operator: ");
        bool first = true;
        for (auto&& i : src) {
            if (first)
                first = false;
            else
                msg.append(", ");
            msg.append(i.to_string());
        }
        megdnn_throw(msg);
    };

    dst.ndim = 0;
    for (auto&& cur : src) {
        if (!cur.ndim)
            err();
        if (!dst.ndim || dst.is_scalar())
            dst = cur;
        else if (!cur.is_scalar()) {
            int max_ndim = std::max(cur.ndim, dst.ndim);
            for (int i = 0; i < max_ndim; ++i) {
                int cur_idx = cur.ndim - i - 1;
                int dst_idx = dst.ndim - i - 1;
                if (cur_idx >= 0 && dst_idx >= 0) {
                    size_t v0 = dst.shape[dst_idx], v1 = cur.shape[cur_idx];
                    if (v0 != v1) {
                        if (v0 > 1 && v1 > 1)
                            err();
                    }
                    int final_idx = std::max(cur_idx, dst_idx);
                    dst.shape[final_idx] = (v0 != 0 && v1 != 0) ? std::max(v0, v1) : 0;
                } else {
                    if (dst_idx < 0) {
                        dst.shape[cur_idx] = cur.shape[cur_idx];
                    }
                }
            }
            dst.ndim = max_ndim;
        }
    }
}

void FormatDeducer::feed(TensorFormat cur) {
    // only one kind of non-default format can exist; and in such case the
    // layouts with default format must be scalar (checked in deduce_layout)
    if (cur == m_default)
        return;

    if (m_result == m_default) {
        m_result = cur;
    } else {
        megdnn_assert(
                m_result == cur, "different input layout formats in elemwise: %s vs %s",
                m_result.impl()->to_string().c_str(), cur.impl()->to_string().c_str());
    }
}

void ElemwiseForward::deduce_format(const TensorFormatArray& src, TensorFormat& dst) {
    FormatDeducer d;
    for (auto i : src) {
        d.feed(i);
    }
    dst = d.get();
}

void ElemwiseForward::deduce_layout(const TensorLayoutArray& src, TensorLayout& dst) {
    megdnn_assert(src.size() == mode_trait().arity);
    DType dtype;
    FormatDeducer format_deducer;
    for (auto&& i : src) {
        if (!dtype.valid()) {
            dtype = i.dtype;
            dst.format = i.format;
        } else {
            megdnn_assert(
                    dtype == i.dtype, "input dtype not unique: get %s and %s",
                    dtype.name(), i.dtype.name());
        }

        format_deducer.feed(i.format);
    }
    dst.format = format_deducer.get();
    if (!format_deducer.is_default(dst.format)) {
        for (auto&& i : src) {
            if (format_deducer.is_default(i.format)) {
                megdnn_assert(
                        i.collapse_contiguous().is_scalar(),
                        "default format can only be used on scalar, got %s",
                        i.to_string().c_str());
            }
        }
    }

    check_dtype(dtype);
    TensorShapeArray src_shp;
    for (auto&& i : src)
        src_shp.push_back(i);
    deduce_shape(src_shp, dst);
    dst.dtype = dtype;
    dst.init_contiguous_stride();
}

void ElemwiseForward::check_layout_and_broadcast(
        const TensorLayoutPtrArray& src, const TensorLayout& dst) {
    megdnn_assert(src.size() == mode_trait().arity);
    DType dtype;
    for (auto i : src) {
        if (!dtype.valid()) {
            dtype = i->dtype;
        } else {
            megdnn_assert(dtype == i->dtype);
        }
        *i = i->broadcast(dst);
    }
    check_dtype(dtype);
    megdnn_assert(dtype == dst.dtype && dst.is_contiguous());
}

void ElemwiseForward::check_dtype(DType dtype) {
    megdnn_assert(dtype.valid());
    auto&& trait = mode_trait();
    switch (dtype.category()) {
        case DTypeCategory::FLOAT:
            megdnn_assert(
                    trait.allow_float, "unsupport mode %s for float\n", trait.name);
            break;
        case DTypeCategory::INT:
            megdnn_assert(trait.allow_int, "unsupport mode %s for int\n", trait.name);
            break;
        case DTypeCategory::BOOL:
            megdnn_assert(trait.allow_bool, "unsupport mode %s for bool\n", trait.name);
            break;
        default:
            megdnn_throw("bad dtype");
    }
}

// vim: syntax=cpp.doxygen
