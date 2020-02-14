/**
 * \file src/core/include/megbrain/utils/enum_class_bit.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <type_traits>

namespace mgb {
    template<typename T>
    class EnumClassBit {
        std::underlying_type_t<T> m_val;

        constexpr EnumClassBit(std::underlying_type_t<T> v):
            m_val(v)
        {
        }

        public:
            constexpr EnumClassBit(T v):
                m_val(static_cast<std::underlying_type_t<T>>(v))
            {
            }

            constexpr operator T() const {
                return static_cast<T>(m_val);
            }

            constexpr explicit operator bool() const {
                return m_val;
            }

#define DEF_OPR(op) \
            constexpr EnumClassBit operator op (\
                    const EnumClassBit &rhs) const { \
                return m_val op rhs.m_val; \
            }

            DEF_OPR(&)
            DEF_OPR(|)
            DEF_OPR(^)

            constexpr EnumClassBit operator ~() const {
                return ~m_val;
            }


#undef DEF_OPR
    };

}

#define _MGB_DECBO_SINGLE_OPR(cls, op) \
     inline constexpr ::mgb::EnumClassBit<cls> operator op (cls x, cls y) { \
         return ::mgb::EnumClassBit<cls>(x) op ::mgb::EnumClassBit<cls>(y); \
     } \
     inline constexpr ::mgb::EnumClassBit<cls> operator op ( \
             ::mgb::EnumClassBit<cls> x, cls y) { \
         return x op ::mgb::EnumClassBit<cls>(y); \
     }

#define _MGB_DECBO_SINGLE_OPR_ASSIGN(cls, op) \
     inline constexpr cls& operator op##= (cls& x, cls y) { \
         x = x op ::mgb::EnumClassBit<cls>(y); \
         return x; \
     }

#define MGB_DEF_ENUM_CLASS_BIT_OPR(cls) \
    _MGB_DECBO_SINGLE_OPR(cls, &) \
    _MGB_DECBO_SINGLE_OPR(cls, |) \
    _MGB_DECBO_SINGLE_OPR(cls, ^) \
    _MGB_DECBO_SINGLE_OPR_ASSIGN(cls, &) \
    _MGB_DECBO_SINGLE_OPR_ASSIGN(cls, |) \
    _MGB_DECBO_SINGLE_OPR_ASSIGN(cls, ^) \
    inline constexpr ::mgb::EnumClassBit<cls> operator ~ (cls x) { \
        return ~::mgb::EnumClassBit<cls>(x); \
    } \



// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

