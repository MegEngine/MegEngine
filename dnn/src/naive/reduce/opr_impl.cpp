/**
 * \file dnn/src/naive/reduce/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/naive/reduce/opr_impl.h"

#include <climits>
#include <cstring>
#include <functional>
#include "src/common/reduce_helper.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;

namespace {

using Mode = Reduce::Mode;

template <Mode mode, typename ctype>
struct Trait;

template <typename ctype>
struct Trait<Mode::SUM, ctype> {
    static const ctype INIT;

    static ctype apply(ctype x, ctype y) { return x + y; }
    static ctype visit(ctype x) { return x; }
    static ctype write(ctype x, size_t) { return x; }
};
template <typename ctype>
const ctype Trait<Mode::SUM, ctype>::INIT = ctype(0);

template <typename ctype>
struct Trait<Mode::MEAN, ctype> {
    static const ctype INIT;

    static ctype apply(ctype x, ctype y) { return x + y; }
    static ctype visit(ctype x) { return x; }
    static ctype write(ctype x, size_t B) { return x / (ctype)B; }
};
template <typename ctype>
const ctype Trait<Mode::MEAN, ctype>::INIT = ctype(0);

template <typename ctype>
struct Trait<Mode::SUM_SQR, ctype> {
    static const ctype INIT;

    static ctype apply(ctype x, ctype y) { return x + y; }
    static ctype visit(ctype x) { return x * x; }
    static ctype write(ctype x, size_t) { return x; }
};
template <typename ctype>
const ctype Trait<Mode::SUM_SQR, ctype>::INIT = ctype(0);

template <typename ctype>
struct Trait<Mode::PRODUCT, ctype> {
    static const ctype INIT;

    static ctype apply(ctype x, ctype y) { return x * y; }
    static ctype visit(ctype x) { return x; }
    static ctype write(ctype x, size_t) { return x; }
};
template <typename ctype>
const ctype Trait<Mode::PRODUCT, ctype>::INIT = ctype(1);

template <typename ctype>
struct Trait<Mode::MIN, ctype> {
    static const ctype INIT;

    static ctype apply(ctype x, ctype y) { return x < y ? x : y; }
    static ctype visit(ctype x) { return x; }
    static ctype write(ctype x, size_t) { return x; }
};
template <typename ctype>
const ctype Trait<Mode::MIN, ctype>::INIT = DTypeTrait<ctype>::max();

template <typename ctype>
struct Trait<Mode::MAX, ctype> {
    static const ctype INIT;

    static ctype apply(ctype x, ctype y) { return x > y ? x : y; }
    static ctype visit(ctype x) { return x; }
    static ctype write(ctype x, size_t) { return x; }
};
template <typename ctype>
const ctype Trait<Mode::MAX, ctype>::INIT = DTypeTrait<ctype>::min();

template <Mode mode, typename ctype>
void reduce_fwd(const ctype* __restrict sptr, ctype* __restrict dptr, size_t A,
                size_t B, size_t C) {
    std::function<ctype(size_t, size_t, size_t, size_t)> func;
    func = [&](size_t a, size_t c, size_t bl, size_t br) -> ctype {
        if (bl + 1 < br) {
            size_t mid = bl + (br - bl) / 2;
            return Trait<mode, ctype>::apply(func(a, c, bl, mid),
                                             func(a, c, mid, br));
        } else {
            return Trait<mode, ctype>::visit(sptr[a * B * C + bl * C + c]);
        }
    };

    for (size_t a = 0; a < A; ++a)
        for (size_t c = 0; c < C; ++c) {
            dptr[a * C + c] = Trait<mode, ctype>::write(func(a, c, 0, B), B);
        }
}

template <>
void reduce_fwd<Mode::SUM>(const dt_quint8* __restrict, dt_quint8* __restrict,
                           size_t, size_t, size_t) {
    megdnn_throw(
            megdnn_mangle("Reduce (SUM) with DEFAULT DataType is not supported "
                          "on Quantized8Asymm"));
}

template <>
void reduce_fwd<Mode::MEAN>(const dt_quint8* __restrict, dt_quint8* __restrict,
                           size_t, size_t, size_t) {
    megdnn_throw(
            megdnn_mangle("Reduce (MEAN) with DEFAULT DataType is not supported "
                          "on Quantized8Asymm"));
}

template <>
void reduce_fwd<Mode::SUM_SQR>(const dt_quint8* __restrict,
                               dt_quint8* __restrict, size_t, size_t, size_t) {
    megdnn_throw(megdnn_mangle(
            "Reduce (SUM_SQR) with DEFAULT DataType is not supported "
            "on Quantized8Asymm"));
}

template <>
void reduce_fwd<Mode::PRODUCT>(const dt_quint8* __restrict,
                               dt_quint8* __restrict, size_t, size_t, size_t) {
    megdnn_throw(megdnn_mangle(
            "Reduce (PRODUCT) with DEFAULT DataType is not supported "
            "on Quantized8Asymm"));
}

template <>
void reduce_fwd<Mode::SUM>(const dt_qint8* __restrict, dt_qint8* __restrict,
                           size_t, size_t, size_t) {
    megdnn_throw(
            megdnn_mangle("Reduce (SUM) with DEFAULT DataType is not supported "
                          "on QuantizedS8"));
}

template <>
void reduce_fwd<Mode::MEAN>(const dt_qint8* __restrict, dt_qint8* __restrict,
                            size_t, size_t, size_t) {
    megdnn_throw(
            megdnn_mangle("Reduce (MEAN) with DEFAULT DataType is not supported "
                          "on QuantizedS8"));
}

template <>
void reduce_fwd<Mode::SUM_SQR>(const dt_qint8* __restrict, dt_qint8* __restrict,
                               size_t, size_t, size_t) {
    megdnn_throw(megdnn_mangle(
            "Reduce (SUM_SQR) with DEFAULT DataType is not supported "
            "on QuantizedS8"));
}

template <>
void reduce_fwd<Mode::PRODUCT>(const dt_qint8* __restrict, dt_qint8* __restrict,
                               size_t, size_t, size_t) {
    megdnn_throw(megdnn_mangle(
            "Reduce (PRODUCT) with DEFAULT DataType is not supported "
            "on QuantizedS8"));
}

template <Mode mode>
void dispatch_dtype(megdnn::naive::HandleImpl* handle, const TensorND& src,
                    const TensorND& dst, size_t A, size_t B, size_t C) {
    switch (src.layout.dtype.enumv()) {
#define cb(_dt)                                                               \
    case DTypeTrait<_dt>::enumv: {                                            \
        using ctype = DTypeTrait<_dt>::ctype;                                 \
        auto sptr = src.ptr<ctype>(), dptr = dst.ptr<ctype>();                \
        MEGDNN_DISPATCH_CPU_KERN(handle, reduce_fwd<mode MEGDNN_COMMA ctype>( \
                                                 sptr, dptr, A, B, C));       \
        return;                                                               \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
        default:
            megdnn_assert_internal(false);
    }
}

}  // anonymous namespace

namespace megdnn {
namespace naive {

size_t ReduceForwardImpl::get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& dst) {
        MEGDNN_MARK_USED_VAR(src);
        MEGDNN_MARK_USED_VAR(dst);
        megdnn_assert(param().data_type != Reduce::DataType::FLOAT_IO16xC32,
                      "FLOAT_IO16xC32 is deprecated");
        DType comp_dtype = src.dtype;
        if (param().mode == Mode::SUM || param().mode == Mode::MEAN) {
            if (src.dtype.category() == DTypeCategory::QUANTIZED) {
                float src_scale;
                if (src.dtype.enumv() == DTypeEnum::QuantizedS8) {
                    src_scale = src.dtype.param<dtype::QuantizedS8>().scale;
                } else if (src.dtype.enumv() == DTypeEnum::Quantized8Asymm) {
                    src_scale = src.dtype.param<dtype::Quantized8Asymm>().scale;
                } else {
                    megdnn_assert_internal(0);
                }
                comp_dtype = dtype::QuantizedS32(src_scale);
            } else if (param().data_type != Param::DataType::DEFAULT) {
                comp_dtype = dtype::Float32();
            }
        } else if (param().data_type != Param::DataType::DEFAULT) {
            comp_dtype = dtype::Float32();
        }

        size_t size = 0;
        if (src.dtype != comp_dtype)
            size += comp_dtype.size(src.total_nr_elems());
        if (dst.dtype != comp_dtype)
            size += comp_dtype.size(dst.total_nr_elems());
        return size;
}

void ReduceForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                             _megdnn_workspace workspace) {
    using namespace reduce;
    check_exec(src.layout, dst.layout, workspace.size);
    size_t A, B, C;
    get_ABC(src.layout, A, B, C, param().axis);

    DType comp_dtype = src.layout.dtype;
    if (param().mode == Mode::SUM || param().mode == Mode::MEAN) {
        if (src.layout.dtype.category() == DTypeCategory::QUANTIZED) {
            float src_scale;
            if (src.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {
                src_scale = src.layout.dtype.param<dtype::QuantizedS8>().scale;
            } else if (src.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) {
                src_scale = src.layout.dtype.param<dtype::Quantized8Asymm>().scale;
            } else {
                megdnn_assert_internal(0);
            }
            comp_dtype = dtype::QuantizedS32(src_scale);
        } else if (param().data_type != Param::DataType::DEFAULT) {
            comp_dtype = dtype::Float32();
        }
    } else if (param().data_type != Param::DataType::DEFAULT) {
        comp_dtype = dtype::Float32();
    }

    auto make_tensor = [&](DType comp_dtype, _megdnn_tensor_inout tensor,
                           dt_byte*& workspace_ptr) {
        if (comp_dtype == tensor.layout.dtype)
            return tensor;
        auto layout = TensorLayout(tensor.layout, comp_dtype);
        TensorND new_tensor{workspace_ptr, layout};
        workspace_ptr += layout.span().dist_byte();
        return new_tensor;
    };

    auto typecvt = handle()->create_operator<TypeCvt>();

    auto copy_to = [&typecvt](const TensorND& from, const TensorND& to) {
        if (from.raw_ptr != to.raw_ptr)
            typecvt->exec(from, to);
    };

    auto workspace_ptr = workspace.ptr<dt_byte>();

    auto new_src = make_tensor(comp_dtype, src, workspace_ptr);
    auto new_dst = make_tensor(comp_dtype, dst, workspace_ptr);

#define CASE(mode)                                                        \
    case mode: {                                                          \
        copy_to(src, new_src);                                            \
        dispatch_dtype<mode>(static_cast<HandleImpl*>(handle()), new_src, \
                             new_dst, A, B, C);                           \
        copy_to(new_dst, dst);                                            \
        return;                                                           \
    }

    switch (param().mode) {
        CASE(Mode::SUM);
        CASE(Mode::SUM_SQR);
        CASE(Mode::PRODUCT);
        CASE(Mode::MIN);
        CASE(Mode::MAX);
        CASE(Mode::MEAN);
        default:
            megdnn_assert_internal(false);
    }
#undef CASE
}

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
