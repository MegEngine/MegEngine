/**
 * \file dnn/src/common/elemwise/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/elemwise/kern_defs.cuh"
#include "src/common/utils.h"

#include "megdnn/oprs.h"
#include "megdnn/tensor_format.h"

#include "midout.h"
MIDOUT_DECL(megdnn_common_elemwise)

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
    static std::mutex mtx;
    static std::vector<ModeTrait> traits;

    std::lock_guard<std::mutex> _lock(mtx);

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
        get(Mode::_m).allow_bool = true;                       \
    }                                                           \
    MIDOUT_END();
        MEGDNN_FOREACH_ELEMWISE_MODE_UNARY_BOOL(cb);
        MEGDNN_FOREACH_ELEMWISE_MODE_BINARY_BOOL(cb);
#undef cb

#define cb(_m)                                                  \
    MIDOUT_BEGIN(megdnn_common_elemwise, midout_iv(Mode::_m)) { \
        auto&& t = get(Mode::_m);                               \
        t.arity = _a;                                           \
        t.name = megdnn_mangle(#_m);                            \
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
        t.name = megdnn_mangle(#_m);                            \
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
            megdnn_assert(i.arity && (i.allow_int || i.allow_float || i.allow_bool) &&
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
    return ret;
}

void ElemwiseForward::deduce_shape(const TensorShapeArray& src,
                                   TensorShape& dst) {
    auto err = [&]() {
        std::string msg(
                megdnn_mangle("bad input shape for polyadic operator: "));
        bool first = true;
        for (auto&& i : src) {
            if (first)
                first = false;
            else
                msg.append(megdnn_mangle(", "));
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
                    dst.shape[final_idx] =
                            (v0 != 0 && v1 != 0) ? std::max(v0, v1) : 0;
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
        megdnn_assert(m_result == cur,
                      "different input layout formats in elemwise: %s vs %s",
                      m_result.impl()->to_string().c_str(),
                      cur.impl()->to_string().c_str());
    }
}

void ElemwiseForward::deduce_format(const TensorFormatArray& src,
                                    TensorFormat& dst) {
    FormatDeducer d;
    for (auto i : src) {
        d.feed(i);
    }
    dst = d.get();
}

void ElemwiseForward::deduce_layout(const TensorLayoutArray& src,
                                    TensorLayout& dst) {
    megdnn_assert(src.size() == mode_trait().arity);
    DType dtype;
    FormatDeducer format_deducer;
    for (auto&& i : src) {
        if (!dtype.valid()) {
            dtype = i.dtype;
            dst.format = i.format;
        } else {
            megdnn_assert(dtype == i.dtype,
                          "input dtype not unique: get %s and %s", dtype.name(),
                          i.dtype.name());
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
            megdnn_assert(trait.allow_float, "unsupport mode %s for float\n",
                          trait.name);
            break;
        case DTypeCategory::INT:
            megdnn_assert(trait.allow_int, "unsupport mode %s for int\n",
                          trait.name);
            break;
        case DTypeCategory::BOOL:
            megdnn_assert(trait.allow_bool, "unsupport mode %s for bool\n",
                          trait.name);
            break;
        default:
            megdnn_throw("bad dtype");
    }
}

// vim: syntax=cpp.doxygen
