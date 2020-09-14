/**
 * \file imperative/python/src/pyext17.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <stdexcept>
#include <vector>
#include <utility>
#include <Python.h>

namespace pyext17 {

#ifdef METH_FASTCALL
constexpr bool has_fastcall = true;
#else
constexpr bool has_fastcall = false;
#endif

template<typename... Args>
struct invocable_with {
    template<typename T>
    constexpr bool operator()(T&& lmb) {
        return std::is_invocable_v<T, Args...>;
    }
};

#define HAS_MEMBER_TYPE(T, U) invocable_with<T>{}([](auto&& x) -> typename std::decay_t<decltype(x)>::U {})
#define HAS_MEMBER(T, m) invocable_with<T>{}([](auto&& x) -> decltype(&std::decay_t<decltype(x)>::m) {})

inline PyObject* cvt_retval(PyObject* rv) {
    return rv;
}

#define CVT_RET_PYOBJ(...) \
    if constexpr (std::is_same_v<decltype(__VA_ARGS__), void>) { \
        __VA_ARGS__; \
        Py_RETURN_NONE; \
    } else { \
        return cvt_retval(__VA_ARGS__); \
    }

template <typename T>
struct wrap {
private:
    typedef wrap<T> wrap_t;

public:
    PyObject_HEAD
    std::aligned_storage_t<sizeof(T), alignof(T)> storage;

    inline T* inst() {
        return reinterpret_cast<T*>(&storage);
    }

    inline static PyObject* pycast(T* ptr) {
        return (PyObject*)((char*)ptr - offsetof(wrap_t, storage));
    }

private:
    // method wrapper

    enum struct meth_type {
        noarg,
        varkw,
        fastcall,
        singarg
    };

    template<auto f>
    struct detect_meth_type {
        static constexpr meth_type value = []() {
            using F = decltype(f);
            static_assert(std::is_member_function_pointer_v<F>);
            if constexpr (std::is_invocable_v<F, T>) {
                return meth_type::noarg;
            } else if constexpr (std::is_invocable_v<F, T, PyObject*, PyObject*>) {
                return meth_type::varkw;
            } else if constexpr (std::is_invocable_v<F, T, PyObject*const*, Py_ssize_t>) {
                return meth_type::fastcall;
            } else if constexpr (std::is_invocable_v<F, T, PyObject*>) {
                return meth_type::singarg;
            } else {
                static_assert(!std::is_same_v<F, F>);
            }
        }();
    };

    template<meth_type, auto f>
    struct meth {};

    template<auto f>
    struct meth<meth_type::noarg, f> {
        static constexpr int flags = METH_NOARGS;

        static PyObject* impl(PyObject* self, PyObject*) {
            auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
            CVT_RET_PYOBJ((inst->*f)());
        }
    };

    template<auto f>
    struct meth<meth_type::varkw, f> {
        static constexpr int flags = METH_VARARGS | METH_KEYWORDS;

        static PyObject* impl(PyObject* self, PyObject* args, PyObject* kwargs) {
            auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
            CVT_RET_PYOBJ((inst->*f)(args, kwargs));
        }
    };

    template<auto f>
    struct meth<meth_type::fastcall, f> {
        #ifdef METH_FASTCALL
        static constexpr int flags = METH_FASTCALL;

        static PyObject* impl(PyObject* self, PyObject*const* args, Py_ssize_t nargs) {
            auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
            CVT_RET_PYOBJ((inst->*f)(args, nargs));
        }
        #else
        static constexpr int flags = METH_VARARGS;

        static PyObject* impl(PyObject* self, PyObject* args) {
            auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
            auto* arr = &PyTuple_GET_ITEM(args, 0);
            auto size = PyTuple_GET_SIZE(args);
            CVT_RET_PYOBJ((inst->*f)(arr, size));
        }
        #endif
    };

    template<auto f>
    struct meth<meth_type::singarg, f> {
        static constexpr int flags = METH_O;

        static PyObject* impl(PyObject* self, PyObject* obj) {
            auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
            CVT_RET_PYOBJ((inst->*f)(obj));
        }
    };

    template<auto f>
    static constexpr PyMethodDef make_meth_def(const char* name, const char* doc = nullptr) {
        using M = meth<detect_meth_type<f>::value, f>;
        return {name, (PyCFunction)M::impl, M::flags, doc};
    }

    // polyfills

    struct tp_new {
        static constexpr bool provided = HAS_MEMBER(T, tp_new);
        static constexpr bool varkw = std::is_constructible_v<T, PyObject*, PyObject*>;
        static constexpr bool noarg = std::is_default_constructible_v<T>;

        template<typename = void>
        static PyObject* impl(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
            auto* self = type->tp_alloc(type, 0);
            auto* ptr = reinterpret_cast<wrap_t*>(self)->inst();
            if constexpr (varkw) {
                new(ptr) T(args, kwargs);
            } else {
                new(ptr) T();
            }
            return self;
        }

        static constexpr newfunc value = []() {if constexpr (provided) return T::tp_new;
                                               else if constexpr (varkw || noarg) return impl<>;
                                               else return nullptr;}();
    };

    struct tp_dealloc {
        static constexpr bool provided = HAS_MEMBER(T, tp_dealloc);

        template<typename = void>
        static void impl(PyObject* self) {
            reinterpret_cast<wrap_t*>(self)->inst()->~T();
            Py_TYPE(self)->tp_free(self);
        }

        static constexpr destructor value = []() {if constexpr (provided) return T::tp_dealloc;
                                                  else return impl<>;}();
    };

    struct tp_call {
        static constexpr bool valid = HAS_MEMBER(T, tp_call);
        static constexpr bool static_form = invocable_with<T, PyObject*, PyObject*, PyObject*>{}(
            [](auto&& t, auto... args) -> decltype(std::decay_t<decltype(t)>::tp_call(args...)) {});

        template<typename = void>
        static PyObject* impl(PyObject* self, PyObject* args, PyObject* kwargs) {
            auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
            CVT_RET_PYOBJ(inst->tp_call(args, kwargs));
        }

        static constexpr ternaryfunc value = []() {if constexpr (static_form) return T::tp_call;
                                                   else if constexpr (valid) return impl<>;
                                                   else return nullptr;}();
    };

public:
    class TypeBuilder {
        std::vector<PyMethodDef> m_methods;
        PyTypeObject m_type;
        bool m_finalized = false;
        bool m_ready = false;

        void check_finalized() {
            if (m_finalized) {
                throw std::runtime_error("type is already finalized");
            }
        }
    public:
        TypeBuilder(const TypeBuilder&) = delete;
        TypeBuilder& operator=(const TypeBuilder&) = delete;

        TypeBuilder() : m_type{PyVarObject_HEAD_INIT(nullptr, 0)} {
            // static_assert(HAS_MEMBER(T, tp_name));
            if constexpr (HAS_MEMBER(T, tp_name)) {
                m_type.tp_name = T::tp_name;
            }
            m_type.tp_dealloc = tp_dealloc::value;
            m_type.tp_call = tp_call::value;
            m_type.tp_basicsize = sizeof(wrap_t);
            m_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
            m_type.tp_new = tp_new::value;
        }

        PyTypeObject* operator->() {
            return &m_type;
        }

        bool ready() const {
            return m_ready;
        }

        PyObject* finalize() {
            if (!m_finalized) {
                if (m_methods.size()) {
                    m_methods.push_back({0});
                    if (m_type.tp_methods) {
                        PyErr_SetString(PyExc_SystemError, "tp_method is already set");
                        return nullptr;
                    }
                    m_type.tp_methods = &m_methods[0];
                }
                if (PyType_Ready(&m_type)) {
                    return nullptr;
                }
                m_ready = true;
            }
            return (PyObject*)&m_type;
        }

        template<auto f>
        TypeBuilder& def(const char* name, const char* doc = nullptr) {
            check_finalized();
            m_methods.push_back(make_meth_def<f>(name, doc));
            return *this;
        }
    };

    static TypeBuilder& type() {
        static TypeBuilder type_helper;
        return type_helper;
    }
};

} // namespace pyext17

#undef HAS_MEMBER_TYPE
#undef HAS_MEMBER
#undef CVT_RET_PYOBJ
