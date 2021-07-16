/**
 * \file dnn/src/common/named_tensor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megdnn/named_tensor.h"
#include "src/common/utils.h"

using namespace megdnn;

/* ===================== Dimension ============================  */
const Dimension::Name Dimension::NAME_ALL[] = {
        Dimension::Name::N, Dimension::Name::C, Dimension::Name::H,
        Dimension::Name::W, Dimension::Name::G, Dimension::Name::K,
        Dimension::Name::R, Dimension::Name::S, Dimension::Name::P,
        Dimension::Name::Q,
};
const int Dimension::NR_NAMES = sizeof(Dimension::NAME_ALL);
Dimension::Dimension(const std::string& expr) {
    auto errmsg = [&]() {
        return ssprintf("Invalid dimension(%s)", expr.c_str());
    };
    const char* data = expr.data();
    bool has_stride = false;
    bool has_extent = false;
    bool init_name = false;
    while (*data) {
        if (data[0] >= 'A' && data[0] <= 'Z') {
            megdnn_throw_if(init_name, megdnn_error, errmsg().c_str());
            for (auto e : NAME_ALL) {
                if (data[0] == static_cast<char>(e)) {
                    init_name = true;
                    m_name = e;
                    break;
                }
            }
            megdnn_throw_if(!init_name, megdnn_error, errmsg().c_str());
            ++data;
        } else if (data[0] == '/' && data[1] == '/') {
            megdnn_throw_if(!init_name || has_stride || has_extent,
                            megdnn_error, errmsg().c_str());
            has_stride = true;
            data += 2;
        } else if (data[0] == '%') {
            megdnn_throw_if(!init_name || has_extent, megdnn_error,
                            errmsg().c_str());
            has_extent = true;
            ++data;
        } else if (data[0] >= '0' && data[0] <= '9') {
            megdnn_throw_if(!init_name, megdnn_error, errmsg().c_str());
            uint32_t num = 0;
            while (data[0] >= '0' && data[0] <= '9') {
                num = num * 10 + (data[0] - '0');
                ++data;
            }
            if (has_extent)
                m_extent = num;
            else if (has_stride)
                m_stride = num;
        } else {
            megdnn_throw(errmsg().c_str());
        }
    }
    megdnn_throw_if(!init_name, megdnn_error, errmsg().c_str());
    if (!has_extent) {
        m_extent = UNDETERMINED_EXTENT;
    }
    if (!has_stride) {
        m_stride = 1;
    }
}

Dimension& Dimension::operator=(const Dimension& rhs) {
    m_name = rhs.m_name;
    m_stride = rhs.m_stride;
    m_extent = rhs.m_extent;
    return *this;
}

bool Dimension::operator==(const Dimension& rhs) const {
    return m_name == rhs.m_name && m_stride == rhs.m_stride &&
           m_extent == rhs.m_extent;
}

bool Dimension::operator<(const Dimension& rhs) const {
    if (m_name != rhs.m_name) {
        return static_cast<char>(m_name) < static_cast<char>(rhs.m_name);
    }
    if (m_stride == rhs.m_stride) {
        return m_extent > rhs.m_extent;
    }
    return m_stride > rhs.m_stride;
}

Dimension Dimension::operator*(const Dimension& rhs) const {
    megdnn_assert(m_name == rhs.m_name,
                  "Multiply operation cannot be applied on dimensions with "
                  "different name(lhs:%c, rhs:%c)",
                  static_cast<char>(m_name), static_cast<char>(rhs.m_name));
    megdnn_assert(
            m_stride == rhs.m_stride * rhs.m_extent,
            "Multiply operation cannot be applied on operands(lhs:%s, rhs:%s)",
            to_string().c_str(), rhs.to_string().c_str());
    if (m_extent == UNDETERMINED_EXTENT)
        return Dimension(m_name, rhs.m_stride);
    return Dimension(m_name, rhs.m_stride, m_extent * rhs.m_extent);
}

Dimension Dimension::operator/(const Dimension& rhs) const {
    megdnn_assert(m_name == rhs.m_name,
                  "Divide operation cannot be applied on dimensions with "
                  "different name(lhs:%c, rhs:%c)",
                  static_cast<char>(m_name), static_cast<char>(rhs.m_name));
    if (operator==(rhs))
        return Dimension(m_name, 1, 1);
    megdnn_assert(
            !(*this < rhs),
            "Divisor must be smaller than dividend(dividend:%s, divisor:%s)",
            to_string().c_str(), rhs.to_string().c_str());
    if (m_stride == rhs.m_stride) {
        if (m_extent == UNDETERMINED_EXTENT) {
            megdnn_assert(rhs.m_extent != UNDETERMINED_EXTENT,
                          "Divide operation cannot be applied on "
                          "operands(dividend:%s, divisor:%s)",
                          to_string().c_str(), rhs.to_string().c_str());
            return Dimension(m_name, rhs.m_extent * m_stride);
        } else {
            megdnn_assert(m_extent % rhs.m_extent == 0,
                          "Divide operation cannot be applied on "
                          "operands(dividend:%s, divisor:%s)",
                          to_string().c_str(), rhs.to_string().c_str());
            return Dimension(m_name, rhs.m_extent * m_stride,
                             m_extent / rhs.m_extent);
        }
    } else {
        if (m_extent == UNDETERMINED_EXTENT) {
            megdnn_assert(rhs.m_extent == UNDETERMINED_EXTENT &&
                                  rhs.m_stride % m_stride == 0,
                          "Divide operation cannot be applied on "
                          "operands(dividend:%s, divisor:%s)",
                          to_string().c_str(), rhs.to_string().c_str());
            return Dimension(m_name, m_stride, rhs.m_stride / m_stride);
        } else {
            megdnn_assert(m_extent * m_stride == rhs.m_extent * rhs.m_stride &&
                                  rhs.m_stride % m_stride == 0,
                          "Divide operation cannot be applied on "
                          "operands(dividend:%s, divisor:%s)",
                          to_string().c_str(), rhs.to_string().c_str());
            return Dimension(m_name, m_stride, m_extent / rhs.m_extent);
        }
    }
}

std::string Dimension::to_string() const {
    if (m_extent == UNDETERMINED_EXTENT) {
        if (m_stride == 1)
            return ssprintf("%c", static_cast<char>(m_name));
        else
            return ssprintf("%c//%u", static_cast<char>(m_name), m_stride);
    } else {
        if (m_stride == 1)
            return ssprintf("%c%%%u", static_cast<char>(m_name), m_extent);
        else
            return ssprintf("%c//%u%%%u", static_cast<char>(m_name), m_stride,
                            m_extent);
    }
}

/* ===================== NamedTensorShape =====================  */

NamedTensorShape::NamedTensorShape(const SmallVector<Dimension>& init_shape) {
    megdnn_assert(init_shape.size() <= MAX_NDIM,
                  "Illegal to construct a NamedTensorShape with "
                  "more than MAX_NDIM(%zu) axes; got(%zu)",
                  MAX_NDIM, init_shape.size());
    ndim = init_shape.size();
    memcpy(this->dims.data(), init_shape.data(), sizeof(Dimension) * ndim);
}

NamedTensorShape::NamedTensorShape(std::initializer_list<Dimension> init_shape)
        : NamedTensorShape(SmallVector<Dimension>{init_shape}) {}

bool NamedTensorShape::eq_shape(const NamedTensorShape& rhs) const {
    MEGDNN_STATIC_ASSERT(MAX_NDIM == 7, "please update the code");
    if (ndim == rhs.ndim) {
        size_t eq = 0;
        switch (ndim) {
            case 7:
                eq += dims[6] == rhs.dims[6];
                MEGDNN_FALLTHRU
            case 6:
                eq += dims[5] == rhs.dims[5];
                MEGDNN_FALLTHRU
            case 5:
                eq += dims[4] == rhs.dims[4];
                MEGDNN_FALLTHRU
            case 4:
                eq += dims[3] == rhs.dims[3];
                MEGDNN_FALLTHRU
            case 3:
                eq += dims[2] == rhs.dims[2];
                MEGDNN_FALLTHRU
            case 2:
                eq += dims[1] == rhs.dims[1];
                MEGDNN_FALLTHRU
            case 1:
                eq += dims[0] == rhs.dims[0];
        }
        return eq == ndim;
    }
    return false;
}

std::string NamedTensorShape::to_string() const {
    std::string rst("{");
    for (size_t i = 0; i < ndim; i++) {
        if (i)
            rst.append(",");
        rst.append(dims[i].to_string());
    }
    rst.append("}");
    return rst;
}

NamedTensorShape NamedTensorShape::make_named_tensor_shape(Format format) {
    switch (format) {
        case Format::NCHW:
            return {{"N"}, {"C"}, {"H"}, {"W"}};
        case Format::NHWC:
            return {{"N"}, {"H"}, {"W"}, {"C"}};
        case Format::NCHW4:
            return {{"N"}, {"C//4"}, {"H"}, {"W"}, {"C%4"}};
        case Format::NCHW8:
            return {{"N"}, {"C//8"}, {"H"}, {"W"}, {"C%8"}};
        case Format::NCHW32:
            return {{"N"}, {"C//32"}, {"H"}, {"W"}, {"C%32"}};
        case Format::NCHW64:
            return {{"N"}, {"C//64"}, {"H"}, {"W"}, {"C%64"}};
        case Format::NCHW44:
            return {{"N//4"}, {"C//4"}, {"H"}, {"W"}, {"C%4"}, {"N%4"}};
        case Format::NCHW88:
            return {{"N//8"}, {"C//8"}, {"H"}, {"W"}, {"C%8"}, {"N%8"}};
        case Format::NCHW44_DOT:
            return {{"N//4"}, {"C//4"}, {"H"}, {"W"}, {"N%4"}, {"C%4"}};
        default:
            megdnn_throw(
                    ssprintf("Format unimplement(%d)", static_cast<int>(format))
                            .c_str());
    }
}
// vim: syntax=cpp.doxygen
