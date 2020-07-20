/**
 * \file src/core/include/megbrain/utils/metahelper_basic.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/common.h"
#include "megbrain/utils/small_vector.h"

#include <cstddef>
#include <tuple>
#include <vector>
#include <type_traits>

namespace mgb {

namespace metahelper_detail {
    [[noreturn]] void on_maybe_invalid_val_access();

    template <class T, class Tuple, size_t... I>
    constexpr T make_from_tuple_impl(Tuple&& t, std::index_sequence<I...>) {
        return T(std::get<I>(std::forward<Tuple>(t))...);
    }

    template<typename T, size_t idx>
    void do_unpack(const std::vector<T> &) {
    }
    template<typename T, size_t idx, typename R, typename ...Args>
    void do_unpack(const std::vector<T> &vec, R &dest, Args&... args) {
        dest = vec[idx];
        do_unpack<T, idx+1>(vec, args...);
    }

    template<typename T, size_t idx>
    void do_unpack(const mgb::SmallVectorImpl<T> &) {
    }
    template<typename T, size_t idx, typename R, typename ...Args>
    void do_unpack(const mgb::SmallVectorImpl<T> &vec, R &dest, Args&... args) {
        dest = vec[idx];
        do_unpack<T, idx+1>(vec, args...);
    }

    template <typename T, class tag>
    struct is_complete_helper {
        template <typename U>
        static std::integral_constant<bool, sizeof(U) == sizeof(U)> test(U*);
        static std::false_type test(...);
        using type = decltype(test(reinterpret_cast<T*>(0)));
    };

    struct if_constexpr_identity {
        template <typename T>
        decltype(auto) operator()(T&& x) {
            return std::forward<T>(x);
        }
    };

    template <bool cond>
    struct if_constexpr_impl;

    template <>
    struct if_constexpr_impl<true> {
        template <class Then, class Else>
        static decltype(auto) run(Then&& then, Else&&) {
            return then(if_constexpr_identity{});
        }
    };

    template <>
    struct if_constexpr_impl<false> {
        template <class Then, class Else>
        static decltype(auto) run(Then&&, Else&& else_) {
            return else_(if_constexpr_identity{});
        }
    };

    template <size_t skip, typename T, size_t isize, size_t... I>
    decltype(auto) array_skip_impl(const std::array<T, isize>& arr,
                                   std::index_sequence<I...>) {
        static_assert(isize > skip, "invalid argument `skip`");
        return std::forward_as_tuple(arr[I + skip]...);
    }
} // namespace metahelper_detail

//! construct object T from tuple of arguments
template <class T, class Tuple>
constexpr T make_from_tuple(Tuple&& t) {
    constexpr std::size_t size =
        std::tuple_size<std::decay_t<Tuple>>::value;
    return metahelper_detail::make_from_tuple_impl<T>(
            std::forward<Tuple>(t), std::make_index_sequence<size>{});
}

/*!
 * \brief unpack elements in a vector into given references
 *
 * throw exception if vector size does not match given arguments
 */
template<typename T, typename ...Args>
void unpack_vector(const std::vector<T> &vec, Args&...args) {
    mgb_assert(vec.size() == sizeof...(args),
            "can not unpack vector of size %zu into %zu elements",
            vec.size(), sizeof...(args));
    metahelper_detail::do_unpack<T, 0>(vec, args...);
}
template<typename T, typename ...Args>
void unpack_vector(const mgb::SmallVectorImpl<T> &vec, Args&...args) {
    mgb_assert(vec.size() == sizeof...(args),
            "can not unpack vector of size %zu into %zu elements",
            vec.size(), sizeof...(args));
    metahelper_detail::do_unpack<T, 0>(vec, args...);
}

//! whether a type can be copied regardless of its memory location
template<class T>
struct is_location_invariant {
    static constexpr bool value =
        std::is_standard_layout<T>::value &&
        std::is_trivially_copyable<T>::value &&
        std::is_trivially_destructible<T>::value;
};
template<class A, class B>
struct is_location_invariant<std::pair<A, B>> {
    static constexpr bool value =
        is_location_invariant<A>::value && is_location_invariant<B>::value;
};

/*!
 * \brief whether a class is complete at the time of first instantiation
 * \tparam tag a local type to ensure instantiation happens at the time of query
 */
template <typename T, class tag = void>
constexpr bool is_complete_v =
        metahelper_detail::is_complete_helper<T, tag>::type::value;

//! a None type to represent invalid Maybe
class None {};
extern class None None;

//! an optional storage for arbitrary object
template <typename T>
class Maybe {
    static constexpr bool nothrow_move =
            std::is_nothrow_move_assignable<T>::value &&
            std::is_nothrow_move_constructible<T>::value;
    static constexpr bool nothrow_copy =
            std::is_nothrow_copy_assignable<T>::value &&
            std::is_nothrow_copy_constructible<T>::value;

    //! object is valid if this is not null
    T* m_ptr = nullptr;
    std::aligned_storage_t<sizeof(T), alignof(T)> m_storage;

public:
    // do not use =default (see
    // https://stackoverflow.com/questions/7411515/why-does-c-require-a-user-provided-default-constructor-to-default-construct-a
    // )
    Maybe() noexcept {}

    Maybe(const class None&) noexcept {}

    Maybe(const Maybe& rhs) noexcept(nothrow_copy) { operator=(rhs); }

    Maybe(Maybe&& rhs) noexcept(nothrow_move) { operator=(std::move(rhs)); }

    //! construct from value
    template <typename TT, typename = typename std::enable_if<
                                   std::is_constructible<T, TT>::value>::type>
    Maybe(TT&& val_init) {
        emplace(std::forward<TT>(val_init));
    }

    ~Maybe() noexcept { invalidate(); }

    Maybe& operator=(const class None&) noexcept {
        invalidate();
        return *this;
    }

    Maybe& operator=(const Maybe& rhs) noexcept(nothrow_copy) {
        if (m_ptr) {
            if (rhs.m_ptr) {
                *m_ptr = *rhs.m_ptr;
            } else {
                invalidate();
            }
        } else if (rhs.m_ptr) {
            emplace(*rhs.m_ptr);
        }
        return *this;
    }

    Maybe& operator=(Maybe&& rhs) noexcept(nothrow_move) {
        if (m_ptr) {
            if (rhs.m_ptr) {
                *m_ptr = std::move(*rhs.m_ptr);
            } else {
                invalidate();
            }
        } else if (rhs.m_ptr) {
            emplace(std::move(*rhs.m_ptr));
        }
        return *this;
    }

    template <typename TT, typename = typename std::enable_if<
                                   std::is_constructible<T, TT>::value>::type>
    Maybe& operator=(TT&& rhs_init) {
        emplace(std::forward<TT>(rhs_init));
        return *this;
    }

    //! inplace initialization; this can be called multiple times to
    //! override previous value
    template <typename A0, typename A1, typename... Args>
    T& emplace(A0&& a0, A1&& a1, Args&&... args) {
        invalidate();
        m_ptr = new (&m_storage) T{std::forward<A0>(a0), std::forward<A1>(a1),
                                   std::forward<Args>(args)...};
        return *m_ptr;
    }

    template <typename Arg>
    T& emplace(Arg&& arg) {
        // There are many narrowing conversions (for example, assigning size_t
        // to Maybe<ptrdiff_t>) which would trigger -Werror=narrowing if list
        // initialization is used. This overloading using direct initialization
        // has been added to avoid modifying lots of existing souce code when
        // refactoring Maybe
        invalidate();
        m_ptr = new (&m_storage) T(std::forward<Arg>(arg));
        return *m_ptr;
    }

    T& emplace() {
        invalidate();
        m_ptr = new (&m_storage) T{};
        return *m_ptr;
    }

    T& val() {
        // do not use assert for code size
        if (mgb_unlikely(!m_ptr)) {
            metahelper_detail::on_maybe_invalid_val_access();
        }
        return *m_ptr;
    }

    T* operator->() { return &val(); }

    const T& val() const { return const_cast<Maybe&>(*this).val(); }

    const T* operator->() const { return &val(); }

    /*!
     * \brief get value if this is valid; otherwise returns default value
     *
     * Note: this function returns by value rather than by reference to
     * ensure valid storage. The type should usually be a scalar type.
     */
    T val_with_default(T default_ = T{}) const {
        return m_ptr ? *m_ptr : default_;
    }

    bool valid() const { return m_ptr; }

    //! no action is performed if this is not valid
    void invalidate() noexcept;
};

template <typename T>
void Maybe<T>::invalidate() noexcept {
    if (m_ptr) {
        m_ptr->~T();
        m_ptr = nullptr;
    }
}

//! convert from a ptr to another type that has may_alias attr; use raw_cast if
//! possible
template <typename T, typename U>
T* __attribute__((__may_alias__)) aliased_ptr(U* src) {
    return reinterpret_cast<T*>(src);
}

//! union of two types with same size and alignment, without constructor
template <typename T, typename U>
union SafeUnion2 {
    static_assert(is_location_invariant<T>::value &&
                          is_location_invariant<U>::value,
                  "must be location invariant");
    static_assert(sizeof(T) == sizeof(U) && alignof(T) && alignof(U),
                  "size and alignments must be the same");
    T t;
    U u;

    SafeUnion2() {}
};

//! cast from \p U to \p T like reinterpret_cast; can be used to bypass strict
//! aliasing
template <typename T, typename U>
T raw_cast(U&& u) {
    SafeUnion2<typename std::decay<T>::type, typename std::decay<U>::type> x;
    x.u = u;
    return x.t;
}

//! functionally equivalent to "if constexpr" in C++17.
//! then/else callbacks receives an Identity functor as its sole argument.
//! The functor is useful for masking eager type check on generic lambda,
//! preventing syntax error on not taken branches.
template <bool Cond, class Then, class Else>
decltype(auto) if_constexpr(Then&& then, Else&& else_) {
    return metahelper_detail::if_constexpr_impl<Cond>::run(then, else_);
}

template <bool Cond, class Then>
decltype(auto) if_constexpr(Then&& then) {
    return if_constexpr<Cond>(std::forward<Then>(then), [](auto) {});
}

template <size_t skip, typename T, size_t isize>
decltype(auto) array_skip(const std::array<T, isize>& arr) {
    return metahelper_detail::array_skip_impl<skip>(
            arr, std::make_index_sequence<isize - skip>{});
}

} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
