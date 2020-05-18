/**
 * \file dnn/src/common/cv/aligned_allocator.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include <type_traits>

#include "megdnn/arch.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include "malloc.h"
#endif


#if defined(__ANDROID__) || defined(ANDROID)
#include "malloc.h"
#define HAS_MEMALIGN
#elif !defined(_MSC_VER) && !defined(__MINGW32__)
#define HAS_POSIX_MEMALIGN
#endif

namespace ah {
/**
 *  @tparam  _Tp  Type of allocated object.
 *  @tparam  _align  Alignment, in bytes.
 */
template <typename _Tp, size_t _align, bool _Nothrow = true>
class aligned_allocator : public std::allocator<_Tp> {
public:
    typedef size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef _Tp* pointer;
    typedef const _Tp* const_pointer;
    typedef _Tp& reference;
    typedef const _Tp& const_reference;
    typedef _Tp value_type;

    template <typename _Tp1>
    struct rebind {
        typedef aligned_allocator<_Tp1, _align> other;
    };

    typedef std::true_type propagate_on_container_move_assignment;

    aligned_allocator() MEGDNN_NOEXCEPT {}

    template <typename _Tp1>
    aligned_allocator(const aligned_allocator<_Tp1, _align>&) MEGDNN_NOEXCEPT {}

    ~aligned_allocator() MEGDNN_NOEXCEPT {}

    // NB: __n is permitted to be 0.  The C++ standard says nothing
    // about what the return value is when __n == 0.
    pointer allocate(size_type __n, const void* = 0) {
        if (__n > this->max_size())
            megdnn_trap();

#ifdef HAS_POSIX_MEMALIGN
        _Tp* result;
        if (posix_memalign(&(void*&)result, _align, __n * sizeof(_Tp)) != 0) {
            if (_Nothrow) {
                return nullptr;
            } else {
                megdnn_trap();
            }
        }
        return result;
#elif defined(HAS_MEMALIGN)
        return (_Tp*)memalign(_align, __n * sizeof(_Tp));
#elif defined(_MSC_VER) || defined(__MINGW32__)
        return (_Tp*)_aligned_malloc(__n * sizeof(_Tp), _align);
#else
#warning \
        "aligned allocator fallbacks to normal malloc; allocated address may be unaligned"
        return (_Tp*)malloc(__n * sizeof(_Tp));
#endif
    }

    // __p is not permitted to be a null pointer.
    void deallocate(pointer __p, size_type) {
#ifdef _MSC_VER
        _aligned_free((void*)__p);
#else
        free((void*)__p);
#endif
    }
};

template <typename _T1, typename _T2, size_t _A1, size_t _A2>
inline bool operator==(const aligned_allocator<_T1, _A1>&,
                       const aligned_allocator<_T2, _A2>&) {
    return true;
}

template <typename _T1, typename _T2, size_t _A1, size_t _A2>
inline bool operator!=(const aligned_allocator<_T1, _A1>&,
                       const aligned_allocator<_T2, _A2>&) {
    return false;
}

/// allocator<void> specialization.
template <size_t _align>
class aligned_allocator<void, _align> {
public:
    typedef size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef void* pointer;
    typedef const void* const_pointer;
    typedef void value_type;

    template <typename _Tp1>
    struct rebind {
        typedef aligned_allocator<_Tp1, _align> other;
    };

    typedef std::true_type propagate_on_container_move_assignment;
};

}  // namespace ah

// vim: syntax=cpp.doxygen
