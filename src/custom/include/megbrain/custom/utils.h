/**
 * \file src/custom/include/megbrain/custom/utils.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cassert>

namespace custom {

void assert_failed_log(const char *file, int line, const char *func, const char *expr, const char *msg_fmt, ...);

#define custom_expect(expr, msg...)                                 \
    if (!(expr)) {                                                  \
        assert_failed_log(                                          \
            __FILE__, __LINE__, __PRETTY_FUNCTION__, #expr, ##msg   \
        );                                                          \
    }

#define custom_assert(expr, msg...)                                 \
    if (!(expr)) {                                                  \
        assert_failed_log(                                          \
            __FILE__, __LINE__, __PRETTY_FUNCTION__, #expr, ##msg   \
        );                                                          \
    }                                                               \
    assert((expr))

class UnImpleWarnLog {
public:
    UnImpleWarnLog(const std::string &func, const std::string &attr,
                   const std::string &val);
};

using void_deleter = void(*)(void*);

template<typename Impl>
void impl_deleter(void *ptr) {
    delete reinterpret_cast<Impl*>(ptr);
}

#define TypedPtr(type, raw_ptr) reinterpret_cast<type*>(raw_ptr)
#define TypedRef(type, raw_ptr) (*reinterpret_cast<type*>(raw_ptr))

#define CUSTOM_PIMPL_CLS_DECL(Cls)                                                  \
    std::unique_ptr<void, void_deleter> m_impl;                                     \
  public:                                                                           \
    Cls();                                                                          \
    Cls(const Cls &rhs);                                                            \
    Cls &operator=(const Cls &rhs)

#define CUSTOM_PIMPL_CLS_DEFINE(Cls)                                                \
    Cls::Cls(): m_impl(new Cls##Impl(), impl_deleter<Cls##Impl>) {}                 \
                                                                                    \
    Cls::Cls(const Cls &rhs): m_impl(nullptr, impl_deleter<Cls##Impl>) {            \
        custom_assert(                                                              \
            rhs.m_impl != nullptr,                                                  \
            "invalid rhs for the copy constructor of %s", #Cls                      \
        );                                                                          \
        m_impl.reset(new Cls##Impl(TypedRef(Cls##Impl, rhs.m_impl.get())));         \
    }                                                                               \
                                                                                    \
    Cls &Cls::operator=(const Cls &rhs) {                                           \
        custom_assert(                                                              \
            m_impl != nullptr && rhs.m_impl != nullptr,                             \
            "invalid assignment of %s, lhs or rhs is invalid", #Cls                 \
        );                                                                          \
        if (&rhs == this)                                                           \
            return *this;                                                           \
                                                                                    \
        TypedRef(Cls##Impl, m_impl.get()) = TypedRef(Cls##Impl, rhs.m_impl.get());  \
        return *this;                                                               \
    }

/**
 * we define this two function explicitly used for std::unordered_map
 * to improve the compatibility with different compiler versions
*/
template <typename T>
struct EnumHash {
    size_t operator()(const T &rhs) const {
        return static_cast<size_t>(rhs);
    }
};

template <typename T>
struct EnumCmp {
    bool operator()(const T &lhs, const T &rhs) const {
        return static_cast<size_t>(lhs) == static_cast<size_t>(rhs);
    }
};


}   // custom
