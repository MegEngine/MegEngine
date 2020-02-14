/**
 * \file python_module/src/cpp/python_helper.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief helper utilities for python integration
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/graph.h"

#include <Python.h>
#include <string>

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
            PYTHON_GIL;
            Py_DECREF(p);
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

class PyStackExtracter {
    static PyStackExtracter *ins;

    public:
        virtual ~PyStackExtracter() = default;

        virtual std::string extract() = 0;

        static void reg(PyStackExtracter *p) {
            ins = p;
        }

        static std::string run() {
            return ins->extract();
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

/*!
 * \brief make python exception
 */
class PyMGBExceptionMaker {
    static PyObject *py_exc_class;
    friend std::string blame(mgb::cg::OperatorNodeBase* opr);

    public:
        static void setup_py_exception(std::exception &exc);

        static void _reg_exception_class(PyObject *cls) {
            py_exc_class = cls;
        }

};

//! associate a python object with an operator
class OprPyTracker final : public mgb::NonCopyableObj {
    class TrackerStorage;
    OprPyTracker() = delete;

public:
    /*!
     * \brief set current tracker; all operators created later would share
     *      this tracker
     *
     * Note that a py reference would be kept
     */
    static void begin_set_tracker(mgb::cg::ComputingGraph& graph,
                                  PyObject* obj);

    static void end_set_tracker(mgb::cg::ComputingGraph& graph);

    struct TrackerResult {
        mgb::cg::OperatorNodeBase
                //! operator that directly causes the exception
                *exc_opr = nullptr,
                //! operator constructed by user (non-optimized exc_opr)
                *unopt_opr = nullptr,
                //! the grad source if opr is constructed by taking grad
                        *opr_grad_src = nullptr;
        PyObject *tracker = nullptr, *tracker_grad_src = nullptr;

        //! format as python tuple
        PyObject* as_tuple(const char* leading_msg = nullptr) const;
    };

    //! get tracker from exception
    static TrackerResult get_tracker(mgb::MegBrainError& exc);

    //! get tracker from operator
    static TrackerResult get_tracker(mgb::cg::OperatorNodeBase* opr);
};

std::string blame(mgb::cg::OperatorNodeBase* opr);

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

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
