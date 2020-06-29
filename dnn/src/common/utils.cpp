/**
 * \file dnn/src/common/utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/utils.h"
#include "megdnn/handle.h"

#include <cstdarg>
#include <cstring>
#include <mutex>
#include <numeric>

using namespace megdnn;

namespace {
std::string svsprintf(const char* fmt, va_list ap_orig) {
    int size = 100; /* Guess we need no more than 100 bytes */
    char* p;

    if ((p = (char*)malloc(size)) == nullptr)
        return "svsprintf: malloc failed";

    for (;;) {
        va_list ap;
        va_copy(ap, ap_orig);
        int n = vsnprintf(p, size, fmt, ap);
        va_end(ap);

        if (n < 0)
            return "svsprintf: vsnprintf failed";

        if (n < size) {
            std::string rst(p);
            free(p);
            return rst;
        }

        size = n + 1;

        char* np = (char*)realloc(p, size);
        if (!np) {
            free(p);
            return "svsprintf: realloc failed";
        } else
            p = np;
    }
}
}  // anonymous namespace

std::string megdnn::ssprintf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    auto rst = svsprintf(fmt, ap);
    va_end(ap);
    return rst;
}

void megdnn::__assert_fail__(const char* file, int line, const char* func,
                             const char* expr, const char* msg_fmt, ...) {
    std::string msg;
    if (msg_fmt) {
        va_list ap;
        va_start(ap, msg_fmt);
        msg = "\nextra message: ";
        msg.append(svsprintf(msg_fmt, ap));
        va_end(ap);
    }
    msg = ssprintf("assertion `%s' failed at %s:%d: %s%s", expr, file, line,
                   func, msg.c_str());
    megdnn_throw(msg.c_str());
}

bool megdnn::get_next_addr(size_t* idx, const size_t* shp, size_t n,
                           size_t stride) {
    auto errmsg = [&]() {
        std::string res;
        res.append(megdnn_mangle("idx={"));
        for (size_t i = 0; i < n; ++i) {
            res.append(std::to_string(idx[i]));
            if (i + 1 < n)
                res.append(megdnn_mangle(","));
        }
        res.append(megdnn_mangle("}, shp={"));
        for (size_t i = 0; i < n; ++i) {
            res.append(std::to_string(shp[i]));
            if (i + 1 < n)
                res.append(megdnn_mangle(","));
        }
        res.append(megdnn_mangle("}, n="));
        res.append(std::to_string(n));
        res.append(megdnn_mangle(", stride="));
        res.append(std::to_string(stride));
        return res;
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    for (size_t i = 0; i < n; ++i) {
        megdnn_assert(idx[i] < shp[i], "%s", errmsg().c_str());
    }
    idx[n - 1] += stride;
    megdnn_assert(idx[n - 1] <= shp[n - 1], "%s", errmsg().c_str());
    size_t i;
    for (i = n; i > 1; --i)
        if (idx[i - 1] == shp[i - 1]) {
            idx[i - 1] = 0;
            ++idx[i - 2];
        } else {
            break;
        }
    if (i == 1 && idx[0] == shp[0]) {
        idx[0] = 0;
        return false;
    }
    return true;
}

int megdnn::get_linear_addr_noncont(size_t* index, const TensorLayout& layout) {
    int ans = 0;
    rep(i, layout.ndim) { ans += index[i] * layout.stride[i]; }
    return ans;
}

size_t megdnn::get_linear_addr(size_t* index, const size_t* shape, size_t n) {
    size_t base = 1;
    size_t ans = 0;
    for (size_t i = n; i > 0; --i) {
        ans += index[i - 1] * base;
        base *= shape[i - 1];
    }
    return ans;
}

size_t megdnn::infer_conv_shape(size_t inp, size_t flt, size_t stride,
                                size_t pad, bool is_floor) {
    megdnn_assert(inp + 2 * pad >= flt, "input=%zu padding=%zu filter=%zu", inp,
                  pad, flt);
    if (is_floor) {
        return (inp + 2 * pad - flt) / stride + 1;
    }
    return (inp + 2 * pad - flt + stride - 1) / stride + 1;
}

void megdnn::infer_conv_shape2d(size_t ih, size_t iw, size_t fh, size_t fw,
                                size_t sh, size_t sw, size_t ph, size_t pw,
                                size_t& oh, size_t& ow, bool is_floor) {
    oh = infer_conv_shape(ih, fh, sh, ph, is_floor);
    ow = infer_conv_shape(iw, fw, sw, pw, is_floor);
}

WorkspaceBundle::WorkspaceBundle(void* ptr, SmallVector<size_t> sizes_in_bytes,
                                 size_t align_in_bytes)
        : m_ptr(ptr),
          m_sizes(std::move(sizes_in_bytes)),
          m_align_in_bytes(align_in_bytes) {
    m_aligned_sizes.reserve(m_sizes.size());
    for (auto size : m_sizes) {
        auto aligned_size = size;
        if (size % m_align_in_bytes != 0) {
            aligned_size += m_align_in_bytes - size % m_align_in_bytes;
        }
        m_aligned_sizes.push_back(aligned_size);
    }
}

void* WorkspaceBundle::ptr() const {
    return m_ptr;
}

void* WorkspaceBundle::get(size_t i) const {
    auto addr = reinterpret_cast<uintptr_t>(m_ptr);
    if (addr % m_align_in_bytes != 0)
        addr += m_align_in_bytes - addr % m_align_in_bytes;
    for (size_t j = 0; j < i; ++j) {
        addr += m_aligned_sizes[j];
    }
    return reinterpret_cast<void*>(addr);
}

size_t WorkspaceBundle::nr_workspace() const {
    return m_sizes.size();
}

size_t WorkspaceBundle::get_size(size_t i) const {
    return m_sizes[i];
}

void WorkspaceBundle::set(void* ptr) {
    m_ptr = ptr;
}

size_t WorkspaceBundle::total_size_in_bytes() const {
    //! return 0 if the WorkspaceBundle is empty
    size_t size =
            std::accumulate(m_aligned_sizes.begin(), m_aligned_sizes.end(),
                            static_cast<size_t>(0));
    return size ? size + m_align_in_bytes : size;
}

size_t megdnn::count_not_ones_in_shape(const TensorShape& shape) {
    size_t res = 0u;
    for (size_t i = 0; i < shape.ndim; ++i)
        res += (shape[i] != 1u);
    return res;
}

bool megdnn::is_nhwc_contig_wc(const TensorLayout& layout) {
    return layout.ndim == 4 &&
           (layout.stride[3] == 1 || layout.shape[3] == 1) &&
           (layout.stride[2] == static_cast<ptrdiff_t>(layout.shape[3]) ||
            layout.shape[2] == 1);
}

megcoreDeviceHandle_t megdnn::get_device_handle(Handle* handle) {
    megcoreStatus_t status;
    megcoreDeviceHandle_t dev_handle;
    megcoreComputingHandle_t comp_handle = handle->megcore_computing_handle();
    status = megcoreGetDeviceHandle(comp_handle, &dev_handle);
    megdnn_assert(status == megcoreSuccess);
    return dev_handle;
}

// clang-format off
float megdnn::mul_scale(DType lhs, DType rhs) {
#define cb_binary(dt1, dt2)                        \
    if ((lhs.enumv() == DTypeTrait<dt1>::enumv) && \
        (rhs.enumv() == DTypeTrait<dt2>::enumv))   \
        return lhs.param<dt1>().scale * rhs.param<dt2>().scale;
    cb_binary(::megdnn::dtype::QuantizedS8, ::megdnn::dtype::QuantizedS16)
#undef cb_binary

    megdnn_assert(lhs.enumv() == rhs.enumv());
#define cb(dt)                                \
    if (lhs.enumv() == DTypeTrait<dt>::enumv) \
        return lhs.param<dt>().scale * rhs.param<dt>().scale;
    MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
    MEGDNN_FOREACH_QUANTIZED_LOWBIT_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}
// clang-format on

bool megdnn::dtype_almost_equal(DType lhs, DType rhs) {
    if (lhs.enumv() != rhs.enumv())
        return false;
    if (lhs.category() != DTypeCategory::QUANTIZED)
        return true;
#define cb(dt)                                \
    if (lhs.enumv() == DTypeTrait<dt>::enumv) \
        return almost_equal(lhs.param<dt>().scale, rhs.param<dt>().scale);
    MEGDNN_FOREACH_QUANTIZED_DTYPE_SYMM(cb)
#undef cb
#define cb(dt)                                                               \
    if (lhs.enumv() == DTypeTrait<dt>::enumv)                                \
        return almost_equal(lhs.param<dt>().scale, rhs.param<dt>().scale) && \
               lhs.param<dt>().zero_point == rhs.param<dt>().zero_point;
    MEGDNN_FOREACH_QUANTIZED_DTYPE_ASYMM(cb)
#undef cb
    megdnn_assert_internal(false);
}

template <>
uint8_t megdnn::convert<dt_quint4, uint8_t>(dt_quint4 src, uint8_t dst,
                                            size_t offset) {
    uint8_t _src =
            std::min(src.as_uint8(), DTypeTrait<dtype::Quantized4Asymm>::max());
    if (offset == 0) {
        _src &= 0xF;
        dst &= 0xF0;
        dst |= _src;
    } else {
        _src <<= 4;
        dst &= 0xF;
        dst |= _src;
    }
    return dst;
}

template <>
dt_quint4 megdnn::convert<uint8_t, dt_quint4>(uint8_t src, dt_quint4 dst,
                                              size_t offset) {
    src >>= (offset << 2);
    src &= 0xF;
    dst = dt_quint4(src);
    return dst;
}

template <>
int8_t megdnn::convert<dt_qint4, int8_t>(dt_qint4 src, int8_t dst,
                                         size_t offset) {
    int8_t _src = std::max(
            std::min(src.as_int8(), DTypeTrait<dtype::QuantizedS4>::max()),
            DTypeTrait<dtype::QuantizedS4>::min());
    if (offset == 0) {
        _src &= 0xF;
        dst &= 0xF0;
        dst |= _src;
    } else {
        _src <<= 4;
        dst &= 0xF;
        dst |= _src;
    }
    return dst;
}

template <>
dt_qint4 megdnn::convert<int8_t, dt_qint4>(int8_t src, dt_qint4 dst,
                                           size_t offset) {
    src <<= (4 - (offset << 2));
    src >>= 4;
    dst = dt_qint4(src);
    return dst;
}

/* ======================== CpuNDRange ======================== */
std::string CpuNDRange::to_string() const {
    std::string ret;
    for (size_t i = 0; i < m_dimension; i++) {
        ret += megdnn::ssprintf(" %zu", m_dim[i]);
    }
    return ret;
}

size_t& CpuNDRange::operator[](size_t idx) {
    megdnn_assert(idx < m_dimension, "invalid index: %zu expected < %zu", idx,
                  m_dimension);
    return m_dim[idx];
}

// vim: syntax=cpp.doxygen
