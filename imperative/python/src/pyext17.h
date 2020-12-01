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
#include <pybind11/pybind11.h>

namespace pyext17 {

#ifdef METH_FASTCALL
constexpr bool has_fastcall = true;
#else
constexpr bool has_fastcall = false;
#endif

#ifdef _Py_TPFLAGS_HAVE_VECTORCALL
constexpr bool has_vectorcall = true;
#else
constexpr bool has_vectorcall = false;
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

inline int cvt_retint(int ret) {
    return ret;
}

#define CVT_RET_INT(...) \
    if constexpr (std::is_same_v<decltype(__VA_ARGS__), void>) { \
        __VA_ARGS__; \
        return 0; \
    } else { \
        return cvt_retint(__VA_ARGS__); \
    }


struct py_err_set : std::exception {};

#define HANDLE_ALL_EXC(RET) catch(py_err_set&) {return RET;} \
    catch(pybind11::error_already_set& e) {e.restore(); return RET;} \
    catch(pybind11::builtin_exception& e) {e.set_error(); return RET;} \
    catch(std::exception& e) {PyErr_SetString(PyExc_RuntimeError, e.what()); return RET;}

template <typename T>
struct wrap {
private:
    typedef wrap<T> wrap_t;

public:
    PyObject_HEAD
    std::aligned_storage_t<sizeof(T), alignof(T)> storage;
    #ifdef _Py_TPFLAGS_HAVE_VECTORCALL
    PyObject* (*vectorcall_slot)(PyObject*, PyObject*const*, size_t, PyObject*);
    #endif

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
            try {
                CVT_RET_PYOBJ((inst->*f)());
            } HANDLE_ALL_EXC(nullptr)
        }
    };

    template<auto f>
    struct meth<meth_type::varkw, f> {
        static constexpr int flags = METH_VARARGS | METH_KEYWORDS;

        static PyObject* impl(PyObject* self, PyObject* args, PyObject* kwargs) {
            auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
            try {
                CVT_RET_PYOBJ((inst->*f)(args, kwargs));
            } HANDLE_ALL_EXC(nullptr)
        }
    };

    template<auto f>
    struct meth<meth_type::fastcall, f> {
        #ifdef METH_FASTCALL
        static constexpr int flags = METH_FASTCALL;

        static PyObject* impl(PyObject* self, PyObject*const* args, Py_ssize_t nargs) {
            auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
            try {
                CVT_RET_PYOBJ((inst->*f)(args, nargs));
            } HANDLE_ALL_EXC(nullptr)
        }
        #else
        static constexpr int flags = METH_VARARGS;

        static PyObject* impl(PyObject* self, PyObject* args) {
            auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
            auto* arr = &PyTuple_GET_ITEM(args, 0);
            auto size = PyTuple_GET_SIZE(args);
            try {
                CVT_RET_PYOBJ((inst->*f)(arr, size));
            } HANDLE_ALL_EXC(nullptr)
        }
        #endif
    };

    template<auto f>
    struct meth<meth_type::singarg, f> {
        static constexpr int flags = METH_O;

        static PyObject* impl(PyObject* self, PyObject* obj) {
            auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
            try {
                CVT_RET_PYOBJ((inst->*f)(obj));
            } HANDLE_ALL_EXC(nullptr)
        }
    };

    template<auto f>
    static constexpr PyMethodDef make_meth_def(const char* name, const char* doc = nullptr) {
        using M = meth<detect_meth_type<f>::value, f>;
        return {name, (PyCFunction)M::impl, M::flags, doc};
    }

    template<auto f>
    struct getter {
        using F = decltype(f);

        static PyObject* impl(PyObject* self, void* closure) {
            auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
            try {
                if constexpr (std::is_invocable_v<F, PyObject*, void*>) {
                    CVT_RET_PYOBJ(f(self, closure));
                } else if constexpr (std::is_invocable_v<F, T, void*>) {
                    CVT_RET_PYOBJ((inst->*f)(closure));
                } else if constexpr (std::is_invocable_v<F, T>) {
                    CVT_RET_PYOBJ((inst->*f)());
                } else {
                    static_assert(!std::is_same_v<F, F>);
                }
            } HANDLE_ALL_EXC(nullptr)
        }
    };

    template<auto f>
    struct setter {
        using F = decltype(f);

        template<typename = void>
        static int impl_(PyObject* self, PyObject* val, void* closure) {
            auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
            try {
                if constexpr (std::is_invocable_v<F, PyObject*, PyObject*, void*>) {
                    CVT_RET_INT(f(self, val, closure));
                } else if constexpr (std::is_invocable_v<F, T, PyObject*, void*>) {
                    CVT_RET_INT((inst->*f)(val, closure));
                } else if constexpr (std::is_invocable_v<F, T, PyObject*>) {
                    CVT_RET_INT((inst->*f)(val));
                } else {
                    static_assert(!std::is_same_v<F, F>);
                }
            } HANDLE_ALL_EXC(-1)
        }

        static constexpr auto impl = []() {if constexpr (std::is_same_v<F, std::nullptr_t>) return nullptr;
                                           else return impl_<>;}();
    };

    template<auto get, auto set = nullptr>
    static constexpr PyGetSetDef make_getset_def(const char* name, const char* doc = nullptr, void* closure = nullptr) {
        return {const_cast<char *>(name), getter<get>::impl, setter<set>::impl, const_cast<char *>(doc), closure};
    }

    // polyfills

    struct tp_vectorcall {
        static constexpr bool valid = HAS_MEMBER(T, tp_vectorcall);
        static constexpr bool haskw = [](){if constexpr (valid)
                                               if constexpr (std::is_invocable_v<decltype(&T::tp_vectorcall), T, PyObject*const*, size_t, PyObject*>)
                                                   return true;
                                           return false;}();

        template<typename = void>
        static PyObject* impl(PyObject* self, PyObject*const* args, size_t nargsf, PyObject *kwnames) {
            auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
            if constexpr (haskw) {
                CVT_RET_PYOBJ(inst->tp_vectorcall(args, nargsf, kwnames));
            } else {
                if (kwnames && PyTuple_GET_SIZE(kwnames)) {
                    PyErr_SetString(PyExc_TypeError, "expect no keyword argument");
                    return nullptr;
                }
                CVT_RET_PYOBJ(inst->tp_vectorcall(args, nargsf));
            }
        }

        static constexpr Py_ssize_t offset = []() {if constexpr (valid) return offsetof(wrap_t, vectorcall_slot);
                                                   else return 0;}();
    };

    struct tp_call {
        static constexpr bool provided = HAS_MEMBER(T, tp_call);
        static constexpr bool static_form = invocable_with<T, PyObject*, PyObject*, PyObject*>{}(
            [](auto&& t, auto... args) -> decltype(std::decay_t<decltype(t)>::tp_call(args...)) {});
        static constexpr bool valid = provided || tp_vectorcall::valid;

        template<typename = void>
        static PyObject* impl(PyObject* self, PyObject* args, PyObject* kwargs) {
            auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
            CVT_RET_PYOBJ(inst->tp_call(args, kwargs));
        }

        static constexpr ternaryfunc value = []() {if constexpr (static_form) return T::tp_call;
                                                   else if constexpr (provided) return impl<>;
                                                   #ifdef _Py_TPFLAGS_HAVE_VECTORCALL
                                                   else if constexpr (valid) return PyVectorcall_Call;
                                                   #endif
                                                   else return nullptr;}();
    };

    struct tp_new {
        static constexpr bool provided = HAS_MEMBER(T, tp_new);
        static constexpr bool varkw = std::is_constructible_v<T, PyObject*, PyObject*>;
        static constexpr bool noarg = std::is_default_constructible_v<T>;

        template<typename = void>
        static PyObject* impl(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
            struct FreeGuard {
                PyObject* self;
                PyTypeObject* type;
                ~FreeGuard() {if (self) type->tp_free(self);}
            };

            auto* self = type->tp_alloc(type, 0);
            FreeGuard free_guard{self, type};
            auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
            if constexpr (has_vectorcall && tp_vectorcall::valid) {
                reinterpret_cast<wrap_t*>(self)->vectorcall_slot = &tp_vectorcall::template impl<>;
            }
            try {
                if constexpr (varkw) {
                    new(inst) T(args, kwargs);
                } else {
                    new(inst) T();
                }
            } HANDLE_ALL_EXC(nullptr)
            free_guard.self = nullptr;
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

public:
    class TypeBuilder {
        std::vector<PyMethodDef> m_methods;
        std::vector<PyGetSetDef> m_getsets;
        PyTypeObject m_type;
        bool m_finalized = false;
        bool m_ready = false;

        void check_finalized() {
            if (m_finalized) {
                throw std::runtime_error("type is already finalized");
            }
        }

        static const char* to_c_str(const char* s) {return s;}

        template <size_t N, typename... Ts>
        static const char* to_c_str(const pybind11::detail::descr<N, Ts...>& desc) {
            return desc.text;
        }
    public:
        TypeBuilder(const TypeBuilder&) = delete;
        TypeBuilder& operator=(const TypeBuilder&) = delete;

        TypeBuilder() : m_type{PyVarObject_HEAD_INIT(nullptr, 0)} {
            constexpr auto has_tp_name = HAS_MEMBER(T, tp_name);
            if constexpr (has_tp_name) {
                m_type.tp_name = to_c_str(T::tp_name);
            }
            m_type.tp_dealloc = tp_dealloc::value;
            #ifdef _Py_TPFLAGS_HAVE_VECTORCALL
            m_type.tp_vectorcall_offset = tp_vectorcall::offset;
            #endif
            m_type.tp_call = tp_call::value;
            m_type.tp_basicsize = sizeof(wrap_t);
            m_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
            #ifdef _Py_TPFLAGS_HAVE_VECTORCALL
            if constexpr (tp_vectorcall::valid) {
                m_type.tp_flags |= _Py_TPFLAGS_HAVE_VECTORCALL;
            }
            #endif
            m_type.tp_new = tp_new::value;
        }

        PyTypeObject* operator->() {
            return &m_type;
        }

        bool ready() const {
            return m_ready;
        }

        bool isinstance(PyObject* op) {
            return PyObject_TypeCheck(op, &m_type);
        }

        bool isexact(PyObject* op) {
            return Py_TYPE(op) == &m_type;
        }

        PyObject* finalize() {
            if (!m_finalized) {
                m_finalized = true;
                if (m_methods.size()) {
                    m_methods.push_back({0});
                    if (m_type.tp_methods) {
                        PyErr_SetString(PyExc_SystemError, "tp_method is already set");
                        return nullptr;
                    }
                    m_type.tp_methods = &m_methods[0];
                }
                if (m_getsets.size()) {
                    m_getsets.push_back({0});
                    if (m_type.tp_getset) {
                        PyErr_SetString(PyExc_SystemError, "tp_getset is already set");
                        return nullptr;
                    }
                    m_type.tp_getset = &m_getsets[0];
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

        template<auto get, auto set = nullptr>
        TypeBuilder& def_getset(const char* name, const char* doc = nullptr, void* closure = nullptr) {
            check_finalized();
            m_getsets.push_back(make_getset_def<get, set>(name, doc, closure));
            return *this;
        }
    };

    static TypeBuilder& type() {
        static TypeBuilder type_helper;
        return type_helper;
    }

    template<typename... Args>
    static PyObject* cnew(Args&&... args) {
        auto* pytype = type().operator->();
        auto* self = pytype->tp_alloc(pytype, 0);
        auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
        if constexpr (has_vectorcall && tp_vectorcall::valid) {
            reinterpret_cast<wrap_t*>(self)->vectorcall_slot = &tp_vectorcall::template impl<>;
        }
        new(inst) T(std::forward<Args>(args)...);
        return self;
    }

    template<typename... Args>
    static PyObject* cnew_with_type(PyTypeObject* pytype, Args&&... args) {
        
        auto* self = pytype->tp_alloc(pytype, 0);
        auto* inst = reinterpret_cast<wrap_t*>(self)->inst();
        if constexpr (has_vectorcall && tp_vectorcall::valid) {
            reinterpret_cast<wrap_t*>(self)->vectorcall_slot = &tp_vectorcall::template impl<>;
        }
        new(inst) T(std::forward<Args>(args)...);
        return self;
    }
    
    struct caster {
        static constexpr auto name = T::tp_name;

        T* value;

        bool load(pybind11::handle src, bool convert) {
            if (wrap_t::type().isinstance(src.ptr())) {
                value = reinterpret_cast<wrap_t*>(src.ptr())->inst();
                return true;
            }
            return false;
        }

        template <typename U> using cast_op_type = pybind11::detail::cast_op_type<U>;
        operator T*() { return value; }
        operator T&() { return *value; }
    };



};

} // namespace pyext17

#undef HAS_MEMBER_TYPE
#undef HAS_MEMBER
#undef CVT_RET_PYOBJ
#undef CVT_RET_INT
#undef HANDLE_ALL_EXC
