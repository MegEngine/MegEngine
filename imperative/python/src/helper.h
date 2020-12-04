/**
 * \file imperative/python/src/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph.h"
#include "megbrain/utils/persistent_cache.h"
#include "megbrain/imperative/op_def.h"

#include <Python.h>
#include <string>
#include <iterator>
#if __cplusplus > 201703L
#include <ranges>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

pybind11::module submodule(pybind11::module parent, const char* name, const char* doc = nullptr);

pybind11::module rel_import(pybind11::str name, pybind11::module m, int level);

#if __cplusplus > 201703L
using std::ranges::range_value_t;
#else
template<typename T>
using range_value_t = std::remove_cv_t<std::remove_reference_t<decltype(*std::declval<T>().begin())>>;
#endif

template<typename T>
auto to_list(const T& x) {
    using elem_t = range_value_t<T>;
    std::vector<elem_t> ret(x.begin(), x.end());
    return pybind11::cast(ret);
}

template<typename T>
auto to_tuple(const T& x, pybind11::return_value_policy policy = pybind11::return_value_policy::automatic) {
    auto ret = pybind11::tuple(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        ret[i] = pybind11::cast(x[i], policy);
    }
    return ret;
}

template<typename T>
auto to_tuple(T begin, T end, pybind11::return_value_policy policy = pybind11::return_value_policy::automatic) {
    auto ret = pybind11::tuple(end - begin);
    for (size_t i = 0; begin < end; ++begin, ++i) {
        ret[i] = pybind11::cast(*begin, policy);
    }
    return ret;
}

class PyTaskDipatcher {
    struct Queue : mgb::AsyncQueueSC<std::function<void(void)>, Queue> {
        using Task = std::function<void(void)>;
        void process_one_task(Task& f) {
            if (!Py_IsInitialized()) return;
            pybind11::gil_scoped_acquire _;
            f();
        }

        void on_async_queue_worker_thread_start() override {
            mgb::sys::set_thread_name("py_task_worker");
        }
    };
    Queue queue;
    bool finalized = false;
public:
    template<typename T>
    void add_task(T&& task) {
        // CPython never dlclose an extension so
        // finalized means the interpreter has been shutdown
        if (!finalized) {
            queue.add_task(std::forward<T>(task));
        }
    }
    void wait_all_task_finish() {
        queue.wait_all_task_finish();
    }
    ~PyTaskDipatcher() {
        finalized = true;
        queue.wait_all_task_finish();
    }
};

extern PyTaskDipatcher py_task_q;

class GILManager {
    PyGILState_STATE gstate;

    public:
        GILManager():
            gstate(PyGILState_Ensure())
        {
        }

        ~GILManager() {
            PyGILState_Release(gstate);
        }
};
#define PYTHON_GIL GILManager __gil_manager

//! wraps a shared_ptr and decr PyObject ref when destructed
class PyObjRefKeeper {
    std::shared_ptr<PyObject> m_ptr;

public:
    static void deleter(PyObject* p) {
        if (p) {
            py_task_q.add_task([p](){Py_DECREF(p);});
        }
    }

    PyObjRefKeeper() = default;
    PyObjRefKeeper(PyObject* p) : m_ptr{p, deleter} {}

    PyObject* get() const { return m_ptr.get(); }

    //! create a shared_ptr as an alias of the underlying ptr
    template <typename T>
    std::shared_ptr<T> make_shared(T* ptr) const {
        return {m_ptr, ptr};
    }
};

//! exception to be thrown when python callback fails
class PyExceptionForward : public std::exception {
    PyObject *m_type, *m_value, *m_traceback;
    std::string m_msg;

    PyExceptionForward(PyObject* type, PyObject* value, PyObject* traceback,
                       const std::string& msg)
            : m_type{type},
              m_value{value},
              m_traceback{traceback},
              m_msg{msg} {}

public:
    PyExceptionForward(const PyExceptionForward&) = delete;
    PyExceptionForward& operator=(const PyExceptionForward&) = delete;
    ~PyExceptionForward();

    PyExceptionForward(PyExceptionForward&& rhs)
            : m_type{rhs.m_type},
              m_value{rhs.m_value},
              m_traceback{rhs.m_traceback},
              m_msg{std::move(rhs.m_msg)} {
        rhs.m_type = rhs.m_value = rhs.m_traceback = nullptr;
    }

    //! throw PyExceptionForward from current python error state
    static void throw_() __attribute__((noreturn));

    //! restore python error
    void restore();

    const char* what() const noexcept override { return m_msg.c_str(); }
};

//! numpy utils
namespace npy {
    //! convert tensor shape to raw vector
    static inline std::vector<size_t> shape2vec(const mgb::TensorShape &shape) {
        return {shape.shape, shape.shape + shape.ndim};
    }

    //! change numpy dtype to megbrain supported dtype
    PyObject* to_mgb_supported_dtype(PyObject *dtype);

    //! convert raw vector to tensor shape
    mgb::TensorShape vec2shape(const std::vector<size_t> &vec);

    //! convert megbrain dtype to numpy dtype object; return new reference
    PyObject* dtype_mgb2np(mgb::DType dtype);

    //! convert numpy dtype object or string to megbrain dtype
    mgb::DType dtype_np2mgb(PyObject *obj);

    //! buffer sharing type
    enum class ShareType {
        MUST_SHARE,     //!< must be shared
        MUST_UNSHARE,   //!< must not be shared
        TRY_SHARE       //!< share if possible
    };

    //! get ndarray from HostTensorND
    PyObject* ndarray_from_tensor(const mgb::HostTensorND &val,
            ShareType share_type);

    //! specify how to convert numpy array to tensor
    struct Meth {
        bool must_borrow_ = false;
        mgb::HostTensorND *dest_tensor_ = nullptr;
        mgb::CompNode dest_cn_;

        //! make a Meth that allows borrowing numpy array memory
        static Meth borrow(
                mgb::CompNode dest_cn = mgb::CompNode::default_cpu()) {
            return {false, nullptr, dest_cn};
        }

        //! make a Meth that requires the numpy array to be borrowed
        static Meth must_borrow(
                mgb::CompNode dest_cn = mgb::CompNode::default_cpu()) {
            return {true, nullptr, dest_cn};
        }

        //! make a Meth that requires copying the value into another
        //! tensor
        static Meth copy_into(mgb::HostTensorND *tensor) {
            return {false, tensor, tensor->comp_node()};
        }
    };
    /*!
     * \brief convert an object to megbrain tensor
     * \param meth specifies how the conversion should take place
     * \param dtype desired dtype; it can be set as invalid to allow arbitrary
     *      dtype
     */
    mgb::HostTensorND np2tensor(PyObject *obj, const Meth &meth,
            mgb::DType dtype);
}

// Note: following macro was copied from pybind11/detail/common.h
// Robust support for some features and loading modules compiled against different pybind versions
// requires forcing hidden visibility on pybind code, so we enforce this by setting the attribute on
// the main `pybind11` namespace.
#if !defined(PYBIND11_NAMESPACE)
#  ifdef __GNUG__
#    define PYBIND11_NAMESPACE pybind11 __attribute__((visibility("hidden")))
#  else
#    define PYBIND11_NAMESPACE pybind11
#  endif
#endif

namespace PYBIND11_NAMESPACE {
namespace detail {

    template<typename T, unsigned N> struct type_caster<megdnn::SmallVector<T, N>>
        : list_caster<megdnn::SmallVector<T, N>, T> {};

    template <> struct type_caster<mgb::DType> {
        PYBIND11_TYPE_CASTER(mgb::DType, _("DType"));
    public:
        bool load(handle src, bool convert) {
            auto obj = reinterpret_borrow<object>(src);
            if (!convert && !isinstance<dtype>(obj)) {
                return false;
            }
            if (obj.is_none()) {
                return true;
            }
            try {
                obj = pybind11::dtype::from_args(obj);
            } catch (pybind11::error_already_set&) {
                return false;
            }
            try {
                value = npy::dtype_np2mgb(obj.ptr());
            } catch (...) {
                return false;
            }
            return true;
        }

        static handle cast(mgb::DType dt, return_value_policy /* policy */, handle /* parent */) {
            // ignore policy and parent because we always return a pure python object
            return npy::dtype_mgb2np(std::move(dt));
        }
    };

    template <> struct type_caster<mgb::TensorShape> {
        PYBIND11_TYPE_CASTER(mgb::TensorShape, _("TensorShape"));
    public:
        bool load(handle src, bool convert) {
            auto obj = reinterpret_borrow<object>(src);
            if (!convert && !isinstance<tuple>(obj)) {
                return false;
            }
            if (obj.is_none()) {
                return true;
            }
            value.ndim = len(obj);
            mgb_assert(value.ndim <= mgb::TensorShape::MAX_NDIM);
            size_t i = 0;
            for (auto v : obj) {
                mgb_assert(i < value.ndim);
                value.shape[i] = reinterpret_borrow<object>(v).cast<size_t>();
                ++i;
            }
            return true;
        }

        static handle cast(mgb::TensorShape shape, return_value_policy /* policy */, handle /* parent */) {
            // ignore policy and parent because we always return a pure python object
            return to_tuple(shape.shape, shape.shape + shape.ndim).release();
        }
    };

    // hack to make custom object implicitly convertible from None
    template <typename T> struct from_none_caster : public type_caster_base<T> {
        using base = type_caster_base<T>;
        bool load(handle src, bool convert) {
            if (!convert || !src.is_none()) {
                return base::load(src, convert);
            }
            // adapted from pybind11::implicitly_convertible
            auto temp = reinterpret_steal<object>(PyObject_Call(
                (PyObject*) this->typeinfo->type, tuple().ptr(), nullptr));
            if (!temp) {
                PyErr_Clear();
                return false;
            }
            // adapted from pybind11::detail::type_caster_generic
            if (base::load(temp, false)) {
                loader_life_support::add_patient(temp);
                return true;
            }
            return false;
        }
    };

    template<> struct type_caster<mgb::CompNode> : public from_none_caster<mgb::CompNode> {};

    template <> struct type_caster<mgb::PersistentCache::Blob> {
        PYBIND11_TYPE_CASTER(mgb::PersistentCache::Blob, _("Blob"));
    public:
        bool load(handle src, bool convert) {
            if (!isinstance<bytes>(src)) {
                return false;
            }
            value.ptr = PYBIND11_BYTES_AS_STRING(src.ptr());
            value.size = PYBIND11_BYTES_SIZE(src.ptr());
            return true;
        }
        static handle cast(mgb::PersistentCache::Blob blob, return_value_policy /* policy */, handle /* parent */) {
            return bytes((const char*)blob.ptr, blob.size);
        }
    };

    template <typename T> struct type_caster<mgb::Maybe<T>> {
        using value_conv = make_caster<T>;
        PYBIND11_TYPE_CASTER(mgb::Maybe<T>, _("Optional[") + value_conv::name + _("]"));
    public:
        bool load(handle src, bool convert) {
            if(!src) {
                return false;
            }
            if (src.is_none()) {
                return true;
            }
            value_conv inner_caster;
            if (!inner_caster.load(src, convert)) {
                return false;
            }
            value.emplace(cast_op<T&&>(std::move(inner_caster)));
            return true;
        }

        static handle cast(mgb::Maybe<T> src, return_value_policy policy, handle parent) {
            if(!src.valid()) {
                return none().inc_ref();
            }
            return pybind11::cast(src.val(), policy, parent);
        }
    };

    template<> struct type_caster<mgb::imperative::OpDef> {
    protected:
        std::shared_ptr<mgb::imperative::OpDef> value;
    public:
        static constexpr auto name = _("OpDef");

        operator mgb::imperative::OpDef&() { return *value; }
        operator const mgb::imperative::OpDef&() { return *value; }
        operator std::shared_ptr<mgb::imperative::OpDef>&() { return value; }
        operator std::shared_ptr<mgb::imperative::OpDef>&&() && { return std::move(value); }

        template <typename T> using cast_op_type = T;

        bool load(handle src, bool convert);

        static handle cast(const mgb::imperative::OpDef& op, return_value_policy /* policy */, handle /* parent */);

        static handle cast(std::shared_ptr<mgb::imperative::OpDef> op, return_value_policy policy, handle parent) {
            return cast(*op, policy, parent);
        }
    };

    template <> struct type_caster<std::shared_ptr<mgb::imperative::OpDef>> :
            public type_caster<mgb::imperative::OpDef> {
        template <typename T> using cast_op_type = pybind11::detail::movable_cast_op_type<T>;
    };
} // detail
} // PYBIND11_NAMESPACE

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
