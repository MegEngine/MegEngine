/**
 * \file dnn/src/common/basic_types.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/basic_types.h"
#include "megdnn/tensor_format.h"

#include "src/common/utils.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <tuple>

using namespace megdnn;

/* ===================== ErrorHandler =====================  */
namespace {
class DefaultErrorHandler final : public ErrorHandler {
    void do_on_megdnn_error(const std::string& msg) override {
        megdnn_ignore(msg);
#if MEGDNN_ENABLE_EXCEPTIONS
        throw std::runtime_error{msg};
#else
        megdnn_trap();
#endif
    }
};
}  // namespace
ErrorHandler* ErrorHandler::sm_inst;

ErrorHandler* ErrorHandler::inst() {
    static std::mutex mtx;
    static DefaultErrorHandler default_handler;
    if (megdnn_unlikely(!sm_inst)) {
        std::lock_guard<std::mutex> lg{mtx};
        if (!sm_inst) {
            sm_inst = &default_handler;
        }
    }
    return sm_inst;
}

void ErrorHandler::on_megdnn_error(const std::string& msg) {
    inst()->do_on_megdnn_error(msg);

    // gcc seems to fail to recognize the noreturn attr of
    // do_on_tensor_reshape_error; explicitly mark this function as noreturn
    // here
    megdnn_trap();
}

void ErrorHandler::on_megdnn_error(const char* msg) {
    on_megdnn_error(std::string{msg});
}

void ErrorHandler::on_tensor_reshape_error(const std::string& msg) {
    inst()->do_on_tensor_reshape_error(msg);
    megdnn_trap();
}

void ErrorHandler::on_tensor_reshape_error(const char* msg) {
    on_tensor_reshape_error(std::string{msg});
}

void ErrorHandler::set_handler(ErrorHandler* handler) {
    sm_inst = handler;
}

/* ===================== logging =====================  */

namespace {
LogHandler g_log_handler = nullptr;
}  // anonymous namespace

#if MEGDNN_ENABLE_LOGGING
void megdnn::__log__(LogLevel level, const char* file, const char* func,
                     int line, const char* fmt, ...) {
    if (!g_log_handler)
        return;
    va_list ap;
    va_start(ap, fmt);
    g_log_handler(level, file, func, line, fmt, ap);
    va_end(ap);
}
#endif  // MEGDNN_ENABLE_LOGGING

LogHandler megdnn::set_log_handler(LogHandler handler) {
    auto ret = g_log_handler;
    g_log_handler = handler;
    return ret;
}

/* ===================== TensorShape =====================  */

TensorShape::TensorShape(const SmallVector<size_t>& init_shape) {
    megdnn_assert(init_shape.size() <= MAX_NDIM,
                  "Illegal to construct a TensorShape with "
                  "more than MAX_NDIM(%zu) axes; init_shape is %s",
                  MAX_NDIM, vec2str(init_shape).c_str());
    ndim = init_shape.size();
    memcpy(this->shape, init_shape.data(), sizeof(size_t) * ndim);
}

TensorShape::TensorShape(std::initializer_list<size_t> init_shape)
        : TensorShape(SmallVector<size_t>{init_shape}) {}

size_t TensorShape::total_nr_elems() const {
    if (!ndim)
        return 0;
    return std::accumulate(shape, shape + ndim, 1_z, SafeMultiplies<size_t>());
}

bool TensorShape::eq_shape(const TensorShape& rhs) const {
    MEGDNN_STATIC_ASSERT(MAX_NDIM == 7, "please update the code");
    if (ndim == rhs.ndim) {
        size_t eq = 0;
        switch (ndim) {
            case 7:
                eq += shape[6] == rhs.shape[6]; MEGDNN_FALLTHRU
            case 6:
                eq += shape[5] == rhs.shape[5]; MEGDNN_FALLTHRU
            case 5:
                eq += shape[4] == rhs.shape[4]; MEGDNN_FALLTHRU
            case 4:
                eq += shape[3] == rhs.shape[3]; MEGDNN_FALLTHRU
            case 3:
                eq += shape[2] == rhs.shape[2]; MEGDNN_FALLTHRU
            case 2:
                eq += shape[1] == rhs.shape[1]; MEGDNN_FALLTHRU
            case 1:
                eq += shape[0] == rhs.shape[0];
        }
        return eq == ndim;
    }
    return false;
}

std::string TensorShape::to_string() const {
    std::string rst("{");
    for (size_t i = 0; i < ndim; i++) {
        if (i)
            rst.append(",");
        rst.append(std::to_string(shape[i]));
    }
    rst.append("}");
    return rst;
}

bool TensorShape::is_empty() const {
    for (size_t i = 0; i < ndim; ++i) {
        if (!shape[i]) {
            return true;
        }
    }
    return false;
}

/* ===================== TensorLayout =====================  */
TensorLayout::TensorLayout() = default;

TensorLayout::TensorLayout(DType dtype_) : dtype{dtype_} {}

TensorLayout::TensorLayout(DType dtype_, Format format_)
        : dtype{dtype_}, format{format_} {}

TensorLayout::TensorLayout(const TensorShape& shape, DType dtype)
        : TensorLayout(shape, dtype, DefaultTensorFormat::make()) {}

TensorLayout::TensorLayout(const TensorShape& shape, DType dtype,
                           TensorFormat format_)
        : TensorShape(shape), dtype{dtype}, format{format_} {
    init_contiguous_stride();
}

TensorLayout::TensorLayout(const TensorShape& shape,
                           const std::vector<ptrdiff_t>& stride, DType dtype)
        : TensorLayout(shape, stride, dtype, DefaultTensorFormat::make()) {}

TensorLayout::TensorLayout(const TensorShape& shape,
                           const std::vector<ptrdiff_t>& stride, DType dtype,
                           TensorFormat format_)
        : TensorShape(shape), dtype{dtype}, format{format_} {
    megdnn_assert_eq_size_t(stride.size(), ndim);
    for (size_t i = 0; i < shape.ndim; ++i)
        this->stride[i] = stride[i];
}

size_t TensorLayout::init_contiguous_stride() {
    return format.impl()->init_contiguous_stride(*this);
}

size_t TensorLayout::init_contiguous_stride(const TensorShape& shape) {
    this->TensorShape::operator=(shape);
    return init_contiguous_stride();
}

size_t TensorLayout::init_contiguous_stride(const TensorShape& shape,
                                            TensorFormat format_) {
    this->TensorShape::operator=(shape);
    this->format = format_;
    return init_contiguous_stride();
}

TensorLayout TensorLayout::dimshuffle(const std::vector<size_t>& dims) const {
    TensorLayout res{dtype, format};
    res.ndim = this->ndim;
    megdnn_assert_eq_size_t(dims.size(), this->ndim);
    auto ndim = this->ndim;
    rep(i, ndim) {
        auto dest = dims[i];
        megdnn_assert(dest < ndim);
        res.shape[i] = this->shape[dest];
        res.stride[i] = this->stride[dest];
    }
    return res;
}

TensorLayout TensorLayout::remove_axis(size_t idx) const {
    TensorLayout res{*this};
    res.remove_axis_inplace(idx);
    return res;
}

void TensorLayout::remove_axis_inplace(size_t axis) {
    megdnn_assert(ndim >= 2 && axis < ndim);
    --ndim;
    for (size_t i = axis; i < ndim; ++i) {
        shape[i] = shape[i + 1];
        stride[i] = stride[i + 1];
    }
}

void TensorLayout::add_axis_inplace(size_t axis, size_t shape,
                                    ptrdiff_t stride) {
    megdnn_assert(ndim + 1 <= MAX_NDIM && axis <= ndim && shape,
                  "can not add axis at %zu (current ndim %zu, MAX_NDIM %zu)",
                  axis, ndim, MAX_NDIM);
    ndim++;
    for (size_t i = ndim - 1; i > axis; i--) {
        this->shape[i] = this->shape[i - 1];
        this->stride[i] = this->stride[i - 1];
    }
    this->shape[axis] = shape;
    this->stride[axis] = stride;
}

bool TensorLayout::is_contiguous() const {
    return format.impl()->is_contiguous_spec(*this);
}

bool TensorLayout::is_physical_contiguous() const {
    ptrdiff_t expected = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (shape[i] != 1 && stride[i] != expected)
            return false;
        expected *= shape[i];
    }
    // empty tensors are not contiguous
    return expected != 0;
}

bool TensorLayout::is_abs_monotonous_allow_brdcst() const {
    if (!ndim)
        return false;
    if (ndim == 1)
        return true;
    ptrdiff_t last = std::abs(stride[ndim - 1]) *
                     static_cast<ptrdiff_t>(shape[ndim - 1]);
    for (int i = ndim - 2; i >= 0; --i) {
        if (!stride[i] || shape[i] == 1)
            continue;
        if (std::abs(stride[i]) < last)
            return false;
        last = std::abs(stride[i]) * static_cast<ptrdiff_t>(shape[i]);
    }
    return true;
}

bool TensorLayout::is_contiguous_allow_brdcst() const {
    if (!ndim)
        return false;
    ptrdiff_t expected = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (!stride[i])
            continue;
        if (shape[i] != 1 && stride[i] != expected)
            return false;
        expected *= shape[i];
    }
    // empty tensors are not contiguous
    return expected != 0;
}

/**
 * \brief The collapse_contiguous function will convert a contiguous image like
 * tensor layout into a 2-dimensional layout, shape[0] = height of the image,
 * shape[1] = width of the image, axis = 1, stride[0] = row_pitch_size_in_elem,
 * and stride[1] = 1.
 * So if the nhwcd4 format layout is transformed into a 2d tensor
 * layout after calling this function, the nhwcd4 format layout is contiguous.
 */
TensorLayout TensorLayout::collapse_contiguous() const {
    return format.impl()->collapse_contiguous_spec(*this);
}

bool TensorLayout::is_non_overlapping_strong() const {
    // abs(stride), stride, shape
    std::array<std::tuple<ptrdiff_t, ptrdiff_t, size_t>, MAX_NDIM> vec;
    for (size_t i = 0; i < this->ndim; ++i) {
        vec[i] = std::make_tuple(std::abs(stride[i]), stride[i], shape[i]);
    }
    std::sort(vec.begin(), vec.begin() + this->ndim);
    ptrdiff_t lo = 0, hi = 0;
    for (size_t i = 0; i < this->ndim; ++i) {
        auto cur_stride = std::get<1>(vec[i]);
        auto cur_shape = std::get<2>(vec[i]);
        megdnn_assert(cur_shape > 0);
        if (cur_shape == 1)
            continue;
        if (cur_stride > 0) {
            if (cur_stride <= hi)
                return false;
            hi += cur_stride * (cur_shape - 1);
        } else {
            // cur_stride == 0 is handled here, which causes returning false
            if (lo <= cur_stride)
                return false;
            lo += cur_stride * (cur_shape - 1);
        }
    }
    return true;
}

bool TensorLayout::eq_layout(const TensorLayout& rhs) const {
    megdnn_assert(dtype == rhs.dtype,
                  "could not compare layout on different dtypes: %s vs %s",
                  dtype.name(), rhs.dtype.name());
    MEGDNN_STATIC_ASSERT(MAX_NDIM == 7, "please update the code");

    auto ax = [](size_t shape0, size_t shape1, ptrdiff_t stride0,
                 ptrdiff_t stride1) {
        return (shape0 == shape1) & ((shape0 == 1) | (stride0 == stride1));
    };
    if (ndim == rhs.ndim) {
        size_t eq = 0;
        switch (ndim) {
            case 7:
                eq += ax(shape[6], rhs.shape[6], stride[6], rhs.stride[6]);
                MEGDNN_FALLTHRU
            case 6:
                eq += ax(shape[5], rhs.shape[5], stride[5], rhs.stride[5]);
                MEGDNN_FALLTHRU
            case 5:
                eq += ax(shape[4], rhs.shape[4], stride[4], rhs.stride[4]);
                MEGDNN_FALLTHRU
            case 4:
                eq += ax(shape[3], rhs.shape[3], stride[3], rhs.stride[3]);
                MEGDNN_FALLTHRU
            case 3:
                eq += ax(shape[2], rhs.shape[2], stride[2], rhs.stride[2]);
                MEGDNN_FALLTHRU
            case 2:
                eq += ax(shape[1], rhs.shape[1], stride[1], rhs.stride[1]);
                MEGDNN_FALLTHRU
            case 1:
                eq += ax(shape[0], rhs.shape[0], stride[0], rhs.stride[0]);
        }
        return eq == ndim;
    }
    return false;
}

TensorLayout::Span TensorLayout::span() const {
    return format.impl()->span_spec(*this);
}

TensorLayout TensorLayout::broadcast(const TensorShape& tshape) const {
    megdnn_throw_if(!ndim || !tshape.ndim, tensor_reshape_error,
                    megdnn_mangle("broadcast involves empty tensor"));

    if (is_scalar()) {
        TensorLayout result{dtype, format};
        result.ndim = tshape.ndim;
        for (size_t i = 0; i < tshape.ndim; i++) {
            result.shape[i] = tshape.shape[i];
            result.stride[i] = (tshape.shape[i] == 1);
        }
        return result;
    }

    megdnn_throw_if(tshape.ndim < ndim, tensor_reshape_error,
                    megdnn_mangle(ssprintf(
                            "dimension for broadcast less than "
                            "dst_shape: src_shape=%s dst_shape=%s",
                            to_string().c_str(), tshape.to_string().c_str())));
    TensorLayout result{dtype, format};
    for (size_t i = 0; i < tshape.ndim; ++i) {
        int target_idx = tshape.ndim - i - 1;
        int cur_idx = ndim - i - 1;
        size_t cur_shape = (cur_idx >= 0 ? shape[cur_idx] : 1),
               cur_stride = (cur_idx >= 0 ? stride[cur_idx] : 0);
        if (tshape.shape[target_idx] != cur_shape) {
            megdnn_throw_if(
                    cur_shape != 1 && cur_stride != 0, tensor_reshape_error,
                    megdnn_mangle(ssprintf(
                            "broadcast on dim with shape not equal to 1: "
                            "src_shape=%s dst_shape=%s",
                            to_string().c_str(), tshape.to_string().c_str())));
            result.shape[target_idx] = tshape.shape[target_idx];
            result.stride[target_idx] = 0;
        } else {
            result.shape[target_idx] = cur_shape;
            result.stride[target_idx] = cur_stride;
        }
    }
    result.ndim = tshape.ndim;
    return result;
}

bool TensorLayout::try_reshape(TensorLayout& result,
                               const TensorShape& tshp) const {
    megdnn_assert(tshp.ndim);

    bool is_empty_shape = false;
    for (size_t i = 0; i < tshp.ndim; ++i) {
        if (!tshp.shape[i]) {
            megdnn_throw_if(!format.is_default(), tensor_reshape_error,
                megdnn_mangle(ssprintf("bad target tshp: %s",
                                tshp.to_string().c_str())));
            is_empty_shape = true;
            break;
        }
    }

    megdnn_throw_if(
            !tshp.ndim || total_nr_elems() != tshp.total_nr_elems(),
            tensor_reshape_error,
            megdnn_mangle(ssprintf(
                    "number of elements do not match "
                    "in reshape: src=%s dest=%s",
                    static_cast<const TensorShape&>(*this).to_string().c_str(),
                    tshp.to_string().c_str())));

    auto cont = collapse_contiguous();
    result.dtype = this->dtype;
    result.format = this->format;
    result.TensorShape::operator=(tshp);

    if (is_empty_shape) {
        result.init_contiguous_stride();
        return true;
    }

    size_t sdim = 0, prod = 1, cont_sdim = 0;
    for (size_t i = 0; i < tshp.ndim; ++i) {
        megdnn_assert(cont_sdim < cont.ndim);
        prod *= result.shape[i];
        if (prod > cont.shape[cont_sdim])
            return false;

        if (prod == cont.shape[cont_sdim] &&
            (i + 1 >= tshp.ndim || tshp.shape[i + 1] != 1)) {
            auto s = cont.stride[cont_sdim];
            for (int j = i; j >= static_cast<int>(sdim); --j) {
                result.stride[j] = s;
                s *= result.shape[j];
            }
            ++cont_sdim;
            sdim = i + 1;
            prod = 1;
        }
    }
    megdnn_assert(cont_sdim == cont.ndim);

    return true;
}

TensorLayout TensorLayout::reshape(const TensorShape& shape) const {
    TensorLayout ret;
    auto succ = try_reshape(ret, shape);
    megdnn_throw_if(!succ, tensor_reshape_error,
                    megdnn_mangle(ssprintf("can not reshape from %s to %s",
                                           to_string().c_str(),
                                           shape.to_string().c_str())));
    return ret;
}

std::string TensorLayout::to_string() const {
    std::string rst("{");
    for (size_t i = 0; i < ndim; i++) {
        if (i)
            rst.append(",");
        rst.append(std::to_string(shape[i]));

        rst.push_back('(');
        rst.append(std::to_string(stride[i]));
        rst.push_back(')');
    }
    if (format.type() != Format::Type::DEFAULT) {
        rst.append(" @ ");
        rst.append(format.impl()->to_string());
    }
    rst.append("}");
    return rst;
}

// vim: syntax=cpp.doxygen
