/**
 * \file python_module/src/cpp/python_helper.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief helper utilities for python integration
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "./python_helper.h"
#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/graph/event.h"
#include "megbrain/graph/cg.h"
#include "megbrain/utils/mempool.h"

#include "./numpy_incl.h"

/*
 * demangle typeid, see
 * http://stackoverflow.com/questions/281818/unmangling-the-result-of-stdtype-infoname
 */
#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>

namespace {

std::string demangle_typeid(const char* name) {

    int status = -4; // some arbitrary value to eliminate the compiler warning

    // enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, nullptr, nullptr, &status),
        std::free
    };

    return (status==0) ? res.get() : name ;
}
}
#else

namespace {
// does nothing if not g++
std::string mgb::demangle_typeid(const char* name) {
    return name;
}
}

#endif

using namespace mgb;
using namespace cg;

PyStackExtracter* PyStackExtracter::ins = nullptr;

namespace {

    std::string repr_pyobj(PyObject *obj) {
        if (!obj)
            return "<null PyObject>";
        PYTHON_GIL;
        auto str = PyObject_Repr(obj);
        if (!str)
            return ssprintf("<PyObject at %p (repr failed)>", obj);
        std::string ret{PyUnicode_AsUTF8(str)};
        Py_DECREF(str);
        return ret;
    }

    template<typename T>
    std::string typeid_name(const T &t) {
        return demangle_typeid(typeid(t).name());
    }

} // anonymous namespace

/* ============== OprPyTracker ============== */

class OprPyTracker::TrackerStorage final : public UserDataContainer::UserData,
                                           public NonCopyableObj {
    MGB_TYPEINFO_OBJ_DECL;

    PyObject* m_cur_tracker = nullptr;
    size_t m_refcnt_to_add = 0;
    SyncEventConnecter::ReceiverHandler m_opr_insert_handler;
    ThinHashMap<OperatorNodeBase*, PyObject*> m_opr2tracker;

public:
    explicit TrackerStorage(ComputingGraph& graph) {
        auto on_new_opr = [this](const event::OprInserted& ev) {
            if (!ev.is_dedup && !ev.exc) {
                if (m_cur_tracker) {
                    ++m_refcnt_to_add;
                    m_opr2tracker[ev.opr] = m_cur_tracker;
                }
            }
        };
        m_opr_insert_handler =
                graph.event().register_receiver<event::OprInserted>(on_new_opr);
    }

    ~TrackerStorage() {
        if (m_cur_tracker) {
            // manage refcnt of cur tracker
            disable();
        }
        PYTHON_GIL;
        for (auto&& i : m_opr2tracker) {
            Py_DecRef(i.second);
        }
    }

    //! get the instance
    static TrackerStorage& inst(ComputingGraph& graph) {
        auto make = [&graph]() {
            return std::make_shared<TrackerStorage>(graph);
        };
        return *graph.options()
                        .user_data.get_user_data_or_create<TrackerStorage>(
                                make);
    }

    //! get the tracker associated with an opr, or nullptr
    PyObject* get(OperatorNodeBase* opr) const {
        auto iter = m_opr2tracker.find(opr);
        return iter == m_opr2tracker.end() ? nullptr : iter->second;
    }

    void enable(PyObject* obj) {
        mgb_assert(!m_cur_tracker,
                   "multiple calls to begin_set_tracker() on the same graph");
        m_cur_tracker = obj;
    }

    void disable() {
        mgb_assert(m_cur_tracker,
                   "call end_set_tracker() before begin_set_tracker()");
        if (m_refcnt_to_add) {
            PYTHON_GIL;
            for (size_t i = 0; i < m_refcnt_to_add; ++i) {
                Py_IncRef(m_cur_tracker);
            }
        }
        m_cur_tracker = nullptr;
    }
};
MGB_TYPEINFO_OBJ_IMPL(OprPyTracker::TrackerStorage);

void OprPyTracker::begin_set_tracker(ComputingGraph& graph, PyObject* obj) {
    TrackerStorage::inst(graph).enable(obj);
}

void OprPyTracker::end_set_tracker(ComputingGraph& graph) {
    TrackerStorage::inst(graph).disable();
}

OprPyTracker::TrackerResult OprPyTracker::get_tracker(mgb::MegBrainError& exc) {
    auto ptr = dynamic_cast<const OperatorNodeExcExtraInfo*>(exc.extra_info());
    if (!ptr)
        return {};
    return get_tracker(ptr->opr());
}

OprPyTracker::TrackerResult OprPyTracker::get_tracker(
        mgb::cg::OperatorNodeBase* opr) {
    TrackerResult ret;
    mgb_assert(opr);
    ret.exc_opr = opr;
    opr = cg::get_opr_root_source_opr(opr);
    ret.unopt_opr = opr;

    auto&& storage = TrackerStorage::inst(*opr->owner_graph());
    ret.tracker = storage.get(opr);

    {
        auto&& grad_info = opr->node_prop().attribute().grad_tracker;
        if (grad_info.valid()) {
            ret.opr_grad_src = cg::get_opr_root_source_opr(grad_info->orig_opr);
            ret.tracker_grad_src = storage.get(ret.opr_grad_src);
        }
    }

    return ret;
}

PyObject* OprPyTracker::TrackerResult::as_tuple(const char *leading_msg) const {
    std::string msg;
    if (leading_msg)
        msg = leading_msg;

    auto print_opr = [&](const char *otype, cg::OperatorNodeBase *opr) {
        if (!opr)
            return;

        msg += ssprintf("\n%s: id=%zu name=%s type=%s\n",
                otype, opr->id(), opr->cname(),
                typeid_name(*opr).c_str());
        msg += "  input variables: \n";
        size_t idx = 0;
        for (auto i: opr->input()) {
            msg += ssprintf("    %zu: ", idx ++);
            msg += cg::dump_var_info({i});
            msg += "\n";
        }

        msg += "  output variables: \n";
        idx = 0;
        for (auto i: opr->output()) {
            msg += ssprintf("    %zu: ", idx ++);
            msg += cg::dump_var_info({i});
            msg += "\n";
        }
    };

    print_opr("Associated operator", exc_opr);
    if (unopt_opr != exc_opr) {
        print_opr("Unoptimized equivalent of associated operator", unopt_opr);
    }
    print_opr("Associated operator created by taking grad of", opr_grad_src);

    PYTHON_GIL;
    PyObject *py_msg = PyUnicode_FromString(msg.c_str()),
             *py_tuple = PyTuple_Pack(3, py_msg,
                     tracker ? tracker : Py_None,
                     tracker_grad_src ? tracker_grad_src : Py_None);
    Py_DECREF(py_msg);
    return py_tuple;
}

std::string blame(mgb::cg::OperatorNodeBase* opr) {
    mgb_assert(PyMGBExceptionMaker::py_exc_class,
               "Python exception class is not set yet");
    PyObject* args = OprPyTracker::get_tracker(opr).as_tuple();

    PYTHON_GIL;

    PyObject* py_exc = PyObject_CallObject(PyMGBExceptionMaker::py_exc_class, args);
    Py_DECREF(args);
    mgb_assert(py_exc);

    PyObject* py_str = PyObject_Str(py_exc);
    Py_DECREF(py_exc);
    mgb_assert(py_str);

    int err = PyUnicode_READY(py_str);
    if (err) {
        Py_DECREF(py_str);
        mgb_assert(!err);
    }

    Py_ssize_t c_str_size;
    const char* c_str = PyUnicode_AsUTF8AndSize(py_str, &c_str_size);
    if (!c_str) {
        Py_DECREF(py_str);
        mgb_assert(c_str);
    }
    std::string ret(c_str, c_str_size);
    Py_DECREF(py_str);
    return ret;
}

/* ============== PyMGBExceptionMaker ============== */
PyObject *PyMGBExceptionMaker::py_exc_class = nullptr;

void PyMGBExceptionMaker::setup_py_exception(std::exception &exc) {
    mgb_assert(py_exc_class);
    if (auto cbexc = dynamic_cast<PyExceptionForward*>(&exc)) {
        cbexc->restore();
        return;
    }

    std::string msg;
    try {
        msg = ssprintf("MegBrain core throws exception: %s\n%s",
                typeid_name(exc).c_str(), exc.what());

        auto mgbexc = dynamic_cast<MegBrainError*>(&exc);
        OprPyTracker::TrackerResult tracker;
        if (mgbexc) {
            tracker = OprPyTracker::get_tracker(*mgbexc);
        }

        PYTHON_GIL;
        PyObject *py_exc_arg = tracker.as_tuple(msg.c_str());
        PyErr_SetObject(py_exc_class, py_exc_arg);
        Py_DECREF(py_exc_arg);
    } catch (std::exception &newexc) {
        auto newmsg = ssprintf(
                "caught exception during handling exception: %s\n%s\n"
                "original message: %s",
                typeid_name(newexc).c_str(), newexc.what(),
                msg.c_str());
        PyErr_SetString(PyExc_RuntimeError, newmsg.c_str());
    } catch (...) {
        auto newmsg = ssprintf(
                    "caught unknown exception during handling exception\n"
                    "original message: %s", msg.c_str());
        PyErr_SetString(PyExc_RuntimeError, newmsg.c_str());
    }
}

/* ============== PyExceptionForward ============== */

PyExceptionForward::~PyExceptionForward() {
    PYTHON_GIL;
    PyObjRefKeeper::deleter(m_type);
    PyObjRefKeeper::deleter(m_value);
    PyObjRefKeeper::deleter(m_traceback);
}

void PyExceptionForward::restore() {
    PyErr_Restore(m_type, m_value, m_traceback);
    m_type = m_value = m_traceback = nullptr;
}

void PyExceptionForward::throw_() {
    PyObject *etype, *obj, *trace;
    PyErr_Fetch(&etype, &obj, &trace);
    PyErr_NormalizeException(&etype, &obj, &trace);

    std::string msg{"python exception"};
    bool succ = false;
    if (etype && obj && trace) {
        auto run = [&]() {
#define DEF(name, expr)        \
    PyObjRefKeeper name{expr}; \
    if (!name.get())           \
    return
            DEF(mod, PyImport_ImportModule("traceback"));
            DEF(result, PyObject_CallMethod(mod.get(), "format_exception",
                                            "(OOO)", etype, obj, trace));
            if (!PyList_Check(result.get()))
                return;
            auto size = PyList_Size(result.get());
            msg.append(":\n");
            for (Py_ssize_t i = 0; i < size; ++i) {
                msg.append("  ");
                msg.append(PyUnicode_AsUTF8(PyList_GetItem(result.get(), i)));
            }
            msg.pop_back();  // remove last \n
            succ = true;
#undef DEF
        };
        run();
    }
    if (!succ) {
        PyObject* obj_str_py;
        if (obj && (obj_str_py = PyObject_Repr(obj))) {
            msg.append(" with message ");
            msg.append(PyUnicode_AsUTF8(obj_str_py));
            Py_DECREF(obj_str_py);
        } else {
            msg.append(" with unknown message");
        }
    }
    // throwing exception may cause abort due to unknown reasons; so we first
    // log the message
    mgb_log_error("caught exception from python callback: %s", msg.c_str());
    fflush(stdout);
    fflush(stderr);
    throw PyExceptionForward{etype, obj, trace, msg};
}

/* ============== namespace npy ============== */

namespace {

int to_mgb_supported_dtype_raw(int dtype) {
    if (dtype == NPY_INT64)
        return NPY_INT32;
    if (dtype == NPY_FLOAT64)
        return NPY_FLOAT32;
    return dtype;
}

#define FOREACH_NPY_DTYPE_PAIR(cb) \
    cb(Uint8, NPY_UINT8) \
    cb(Int8, NPY_INT8) \
    cb(Int16, NPY_INT16) \
    cb(Int32, NPY_INT32) \
    cb(Float16, NPY_FLOAT16) \
    cb(Float32, NPY_FLOAT32)

#define FOREACH_NPY_MGB_DTYPE_PAIR(cb) \
    FOREACH_NPY_DTYPE_PAIR(cb) \
    FOREACH_MGB_DTYPE_PAIR(cb)



//! convert megbrain dtype to numpy dtype
int dtype_mgb2np_raw(DType dtype) {
    mgb_assert(dtype.valid(), "attempt to convert from invalid dtype");
    switch (dtype.enumv()) {
#define cb(_m, _n) \
        case DTypeEnum::_m: \
            return _n;
        FOREACH_NPY_MGB_DTYPE_PAIR(cb)
#undef cb
        default:
            break;
    }
    throw ConversionError(ssprintf(
                "can not convert dtype %s to numpy dtype", dtype.name()));
}

struct PyArrayDescrDeleter {
    void operator()(PyArray_Descr* obj) {
        Py_XDECREF(obj);
    }
};

//! Convert MegBrain DType to NumPy DType descriptor, the caller receives a new
//! reference to the descriptor.
std::unique_ptr<PyArray_Descr, PyArrayDescrDeleter> dtype_mgb2np_descr(
        DType dtype) {
    PYTHON_GIL;
    mgb_assert(dtype.valid(), "attempt to convert from invalid dtype");
    auto build_mgb_dtype_dict =
            [](const char* name,
               const std::vector<std::pair<const char*, PyObject*>>& data) {
                PyObject* metadata = PyDict_New();
                PyObject* mgb_dtype_metadata = PyDict_New();
                PyDict_SetItemString(mgb_dtype_metadata, "name",
                                     PyUnicode_FromString(name));
                for (const auto& d : data) {
                    PyDict_SetItemString(mgb_dtype_metadata, d.first, d.second);
                }
                PyDict_SetItemString(metadata, "mgb_dtype", mgb_dtype_metadata);
                return metadata;
            };
    if (dtype.has_param()) {
        PyArray_Descr* type_descr;
        switch (dtype.enumv()) {
            case DTypeEnum::Quantized8Asymm: {
                auto& param = dtype.param<dtype::Quantized8Asymm>();
                type_descr = PyArray_DescrNewFromType(NPY_UINT8);
                type_descr->metadata = build_mgb_dtype_dict(
                        DTypeTrait<dtype::Quantized8Asymm>::name,
                        {{"scale", PyFloat_FromDouble(param.scale)},
                         {"zero_point", PyLong_FromLong(param.zero_point)}});
                break;
            }
            case DTypeEnum::QuantizedS8: {
                auto& param = dtype.param<dtype::QuantizedS8>();
                type_descr = PyArray_DescrNewFromType(NPY_INT8);
                type_descr->metadata = build_mgb_dtype_dict(
                        DTypeTrait<dtype::QuantizedS8>::name,
                        {{"scale", PyFloat_FromDouble(param.scale)}});
                break;
            }
            case DTypeEnum::QuantizedS32: {
                auto& param = dtype.param<dtype::QuantizedS32>();
                type_descr = PyArray_DescrNewFromType(NPY_INT32);
                type_descr->metadata = build_mgb_dtype_dict(
                        DTypeTrait<dtype::QuantizedS32>::name,
                        {{"scale", PyFloat_FromDouble(param.scale)}});
                break;
            }
            default:
                mgb_throw(ConversionError, "unhandled parameterized DType %s",
                          dtype.name());
        }
        return std::unique_ptr<PyArray_Descr, PyArrayDescrDeleter>(type_descr);
    }
    PyArray_Descr* basic_descr = PyArray_DescrFromType(dtype_mgb2np_raw(dtype));
    mgb_assert(basic_descr != nullptr,
                   "failed to convert expected dtype to numpy type descriptor");
    return std::unique_ptr<PyArray_Descr, PyArrayDescrDeleter>(basic_descr);
}

DType dtype_np2mgb_raw(int npt) {
    switch (npt) {
#define cb(_m, _n) \
        case _n: \
            return dtype::_m();
        FOREACH_NPY_DTYPE_PAIR(cb)
#undef cb
    }
#define cb(_m, _n) \
    if (_n == npt) return dtype::_m();
    FOREACH_MGB_DTYPE_PAIR(cb)
#undef cb

    PYTHON_GIL;
    std::string msg;
    auto py_obj = PyArray_TypeObjectFromType(npt);
    if (!py_obj) {
        msg = ssprintf("unknown numpy dtype enum %d", npt);
    } else {
        msg = ssprintf("unsupported numpy dtype %s",
                repr_pyobj(py_obj).c_str());
    }
    Py_DECREF(py_obj);
    throw ConversionError(msg);
}

DType dtype_np2mgb_descr(PyArray_Descr* descr) {
    PYTHON_GIL;
    auto handle_parameterized_dtype = [](PyObject* metadata) -> DType {
        mgb_assert(PyDict_Check(metadata),
                   "Invalid parameterized DType metadata: should be a dict");
        PyObject* dtype_name_py = PyDict_GetItemString(metadata, "name");
        mgb_assert(
                PyUnicode_Check(dtype_name_py),
                "Invalid parameterized DType metadata: name should be a str");
        std::string dtype_name(PyUnicode_AsUTF8(dtype_name_py));
        if (dtype_name == "Quantized8Asymm") {
            PyObject* scale_py = PyDict_GetItemString(metadata, "scale");
            PyObject* zero_point_py =
                    PyDict_GetItemString(metadata, "zero_point");
            mgb_assert(scale_py && zero_point_py,
                       "Invalid Quantized8Asymm metadata: missing scale or "
                       "zero_point.");
            mgb_assert(
                    PyFloat_Check(scale_py),
                    "Invalid Quantized8Asymm metadata: scale should be float");
            mgb_assert(PyLong_Check(zero_point_py),
                       "Invalid Quantized8Asymm metadata: zero_point should be "
                       "integer");
            auto zero_point = PyLong_AS_LONG(zero_point_py);
            mgb_assert(zero_point >= 0 && zero_point < 256,
                       "Invalid Quantized8Asymm metadata: zero_point should be "
                       "in [0, 256)");
            return dtype::Quantized8Asymm(
                    static_cast<float>(PyFloat_AS_DOUBLE(scale_py)),
                    static_cast<uint8_t>(zero_point));
        }
        if (dtype_name == "QuantizedS32" || dtype_name == "QuantizedS8") {
            PyObject* scale_py = PyDict_GetItemString(metadata, "scale");
            mgb_assert(scale_py, "Invalid metadata: missing scale");
            mgb_assert(PyFloat_Check(scale_py),
                       "Invalid metadata: scale should be float");
            float scale = static_cast<float>(PyFloat_AS_DOUBLE(scale_py));
            if (dtype_name == "QuantizedS32") {
                return dtype::QuantizedS32(scale);
            } else {
                return dtype::QuantizedS8(scale);
            }
        }
        throw ConversionError(
                ssprintf("Unknown parameterized DType: %s", dtype_name.c_str())
                        .c_str());
    };
    PyObject* dtype_metadata;
    if (descr->metadata && PyDict_Check(descr->metadata) &&
        (dtype_metadata = PyDict_GetItemString(descr->metadata, "mgb_dtype"))) {
        return handle_parameterized_dtype(dtype_metadata);
    }
    return dtype_np2mgb_raw(descr->type_num);
}

HostTensorND lowbit_ndarray_to_host_tensor(
        CompNode comp_node, TensorLayout &layout, PyArrayObject *input) {
    auto src_ptr = reinterpret_cast<dt_byte*>(PyArray_DATA(input));
    if (!layout.ndim) {
        // numpy scalar
        mgb_assert(src_ptr, "can not convert from null numpy array");
        layout.init_contiguous_stride({1});
    } else {
        mgb_assert(layout.ndim && layout.ndim <= TensorShape::MAX_NDIM,
                "unsupported ndim %zu", layout.ndim);
        for (size_t i = 0; i < layout.ndim; ++ i) {
            layout.shape[i] = PyArray_SHAPE(input)[i];
            layout.stride[i] = PyArray_STRIDE(input, i);
            mgb_assert(layout.shape[i], "zero shape not supported");
        }
        mgb_assert(layout.is_contiguous());
    }
    HostTensorND ret{comp_node, layout};
    lowbit_memcpy_byte2compact(layout.dtype, ret.raw_ptr(), src_ptr,
            layout.total_nr_elems());
    return ret;
}

/*!
 * \brief convert a python object to tensor and try to borrow memory if the
 *      original object is a contiguous numpy array
 * \param dtype see np2tensor
 * \return the megbrain tensor, and whether memory is borrowed
 */
std::pair<HostTensorND, bool> np2tensor_try_borrow(
        PyObject *obj, CompNode dest_cn, DType dtype) {
    mgb_assert(dest_cn.valid());

    PYTHON_GIL;

    PyArray_Descr* expected_descr = nullptr;
    if (dtype.valid()) {
        // The reference to expected_descr will be stealed later.
        expected_descr = dtype_mgb2np_descr(dtype).release();
    }

    // make result from PyArrayObject; its reference would be stolen
    auto make_from_arr = [&](PyArrayObject *input, bool is_borrow) {
        PyObjRefKeeper ref_obj_cvt{reinterpret_cast<PyObject*>(input)};

        TensorLayout layout;
        layout.dtype = dtype_np2mgb_descr(PyArray_DESCR(input));
        if (dtype.valid())
            mgb_assert(dtype == layout.dtype);
        layout.ndim = PyArray_NDIM(input);

        if (layout.dtype.is_low_bit()) {
            auto ret = lowbit_ndarray_to_host_tensor(dest_cn, layout, input);
            // decref(input) would be handled by ref_obj_cvt
            return std::make_pair(ret, false);
        }

        auto data = reinterpret_cast<dt_byte*>(PyArray_DATA(input));
        if (!layout.ndim) {
            // numpy scalar
            mgb_assert(data, "can not convert from null numpy array");
            layout.init_contiguous_stride({1});
        } else {
            mgb_assert(layout.ndim && layout.ndim <= TensorShape::MAX_NDIM,
                    "unsupported ndim %zu", layout.ndim);
            auto dsize = layout.dtype.size();
            bool is_empty = false;
            for (size_t i = 0; i < layout.ndim; ++ i) {
                layout.shape[i] = PyArray_SHAPE(input)[i];
                layout.stride[i] = PyArray_STRIDE(input, i);
                if (!layout.shape[i]) {
                    is_empty = true;
                }
                mgb_assert(layout.stride[i] % dsize == 0,
                        "bad stride %zd", layout.stride[i]);
                layout.stride[i] /= dsize;
            }
            mgb_assert(is_empty || layout.is_contiguous());
        }
        HostTensorStorage storage;
        auto input_ptr = ref_obj_cvt.make_shared(data);
        storage.reset(dest_cn, layout.span().high_byte, input_ptr);
        HostTensorND ret;
        ret.reset(storage, layout);
        return std::make_pair(ret, is_borrow);
    };

    PyArrayObject *obj_as_arr = nullptr;
    do {
        // check contiguous and dtype, and borrow mem if ok
        if (!PyArray_Check(obj))
            break;
        obj_as_arr = reinterpret_cast<PyArrayObject*>(obj);
        int typenum = PyArray_DTYPE(obj_as_arr)->type_num;
        // We have to check dtype.valid() and typenum first to avoid
        // accidentally trigger ConversionError on incompatible dtypes which can
        // be automatically converted into comptaible ones (e.g. float64).
        if (dtype.valid() &&
            (expected_descr->type_num != typenum ||
             dtype_np2mgb_descr(PyArray_DTYPE(obj_as_arr)) != dtype))
            break;
        if (typenum != to_mgb_supported_dtype_raw(typenum)) {
            mgb_assert(!dtype.valid() && expected_descr == nullptr);
            expected_descr =
                    PyArray_DescrFromType(to_mgb_supported_dtype_raw(typenum));
            break;
        }
        if (PyArray_ISCARRAY_RO(obj_as_arr)) {
            Py_INCREF(obj_as_arr);
            return make_from_arr(obj_as_arr, true);
        }
    } while(0);

    constexpr auto NP_FLAGS = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_FORCECAST;
    PyObject *obj_cvt;
    if (obj_as_arr) {
        obj_cvt = PyArray_FromArray(obj_as_arr, expected_descr, NP_FLAGS);
    } else {
        obj_cvt = PyArray_FromAny(obj, expected_descr, 0, 0, NP_FLAGS, nullptr);
    }

    if (obj_cvt) {
        // convert to mgb supported dtype
        auto arr = reinterpret_cast<PyArrayObject*>(obj_cvt);
        int dt0 = PyArray_TYPE(arr), dt1 = to_mgb_supported_dtype_raw(dt0);
        if (dt0 != dt1) {
            mgb_assert(expected_descr == nullptr);
            expected_descr = PyArray_DescrFromType(dt1);
            mgb_assert(expected_descr);
            auto obj_cvt_new = PyArray_FromAny(
                    obj_cvt, expected_descr, 0, 0, NP_FLAGS, nullptr);
            Py_DECREF(obj_cvt);
            obj_cvt = obj_cvt_new;
        }
    }

    if (!obj_cvt) {
        if (PyErr_Occurred()) {
            PyExceptionForward::throw_();
        }
        throw ConversionError(ssprintf("can not convert to numpy array from %s",
                    repr_pyobj(obj).c_str()));
    }

    return make_from_arr(reinterpret_cast<PyArrayObject*>(obj_cvt), false);
}

//! hold a reference to HostTensorND
class HostTensorNDRefHolder final: public NonCopyableObj {
    HostTensorND m_val;
    static MemPool<HostTensorNDRefHolder> sm_mem_pool;

    friend class MemPool<HostTensorNDRefHolder>;

    HostTensorNDRefHolder(const HostTensorND &v):
        m_val{v}
    {
    }

    public:

        static HostTensorNDRefHolder* alloc(const HostTensorND &v) {
            return sm_mem_pool.alloc(v);
        }

        static void free(HostTensorNDRefHolder *p) {
            return sm_mem_pool.free(p);
        }
};
MemPool<HostTensorNDRefHolder> HostTensorNDRefHolder::sm_mem_pool;

void ndarray_shared_from_tensor_py_capsule_dtor(PyObject *cap) {
    auto ptr = PyCapsule_GetPointer(cap, "HostTensorND");
    mgb_assert(ptr, "not a PyCapsule: %s", repr_pyobj(cap).c_str());
    HostTensorNDRefHolder::free(static_cast<HostTensorNDRefHolder*>(ptr));
}

} // anonymous namespace

PyObject* npy::ndarray_from_tensor(
        const HostTensorND &val, ShareType share_type) {
    if (!val.layout().is_contiguous() && !val.shape().is_empty()) {
        mgb_assert(share_type != ShareType::MUST_SHARE);
        HostTensorND contig;
        contig.copy_from(val);
        return ndarray_from_tensor(contig, ShareType::TRY_SHARE);
    }
    PYTHON_GIL;
    npy_intp dims[TensorLayout::MAX_NDIM];
    for (size_t i = 0; i < val.layout().ndim; ++ i)
        dims[i] = val.shape()[i];
    PyObject* ret = nullptr;

    auto alloc_new_ret = [&]() {
        mgb_assert(!ret);
        ret = PyArray_NewFromDescr(
                &PyArray_Type, dtype_mgb2np_descr(val.dtype()).release(),
                val.layout().ndim, dims, nullptr, nullptr, 0, nullptr);
        mgb_assert(ret, "failed to allocate array");
        mgb_assert(PyArray_Check(ret));
        return PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret));
    };
    if (val.dtype().is_low_bit()) {
        mgb_assert(share_type != ShareType::MUST_SHARE,
                "can not share memory for lowbit dtype");
        lowbit_memcpy_compact2byte(val.dtype(), alloc_new_ret(), val.raw_ptr(),
                val.layout().total_nr_elems());
    } else if (share_type == ShareType::MUST_UNSHARE) {
        memcpy(alloc_new_ret(), val.raw_ptr(), val.layout().span().dist_byte());
    } else {
        // share data
        ret = PyArray_NewFromDescr(
                &PyArray_Type, dtype_mgb2np_descr(val.dtype()).release(),
                val.layout().ndim, dims, nullptr,
                const_cast<dt_byte*>(val.raw_ptr()), 0, nullptr);
        mgb_assert(ret, "failed to alloc ndarray");
        auto capsule = PyCapsule_New(HostTensorNDRefHolder::alloc(val),
                "HostTensorND", ndarray_shared_from_tensor_py_capsule_dtor);
        mgb_assert(capsule, "failed to create PyCapsule");
        auto err = PyArray_SetBaseObject(
                reinterpret_cast<PyArrayObject*>(ret), capsule);
        mgb_assert(!err);
    }
    return ret;
}

HostTensorND npy::np2tensor(PyObject* obj, const Meth& meth, DType dtype) {
    auto ret_full = np2tensor_try_borrow(obj, meth.dest_cn_, dtype);
    if (meth.dest_tensor_) {
        meth.dest_tensor_->copy_from(ret_full.first);
        return *meth.dest_tensor_;
    }
    if (meth.must_borrow_) {
        mgb_assert(ret_full.second,
                   "can not borrow from numpy array as contig array with dtype "
                   "%s; src=%s",
                   dtype.name(), repr_pyobj(obj).c_str());
    }
    return ret_full.first;
}

PyObject* npy::dtype_mgb2np(mgb::DType dtype) {
    PYTHON_GIL;
    // According to
    // https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_TypeObjectFromType
    // the following is equivalent to PyArray_TypeObjectFromType for built-in
    // types.
    auto descr = dtype_mgb2np_descr(dtype);
    if (descr == nullptr) {
        return nullptr;
    }
    if (dtype.has_param()) {
        return reinterpret_cast<PyObject*>(descr.release());
    }
    PyObject* typeobj = reinterpret_cast<PyObject*>(descr->typeobj);
    Py_XINCREF(typeobj);
    return typeobj;
}

mgb::DType npy::dtype_np2mgb(PyObject *obj) {
    mgb_assert(obj && obj != Py_None,
               "can not convert null PyObject to numpy dtype");
    // see
    // http://stackoverflow.com/questions/8477122/numpy-c-api-convert-type-object-to-type-number
    PYTHON_GIL;

    PyArray_Descr* dtype;
    if(!PyArray_DescrConverter(obj, &dtype)) {
        throw ConversionError(ssprintf("can not convert to np.dtype from %s",
                    repr_pyobj(obj).c_str()));
    }

    mgb::DType result = dtype_np2mgb_descr(dtype);
    Py_DECREF(dtype);
    return result;
}

PyObject* npy::to_mgb_supported_dtype(PyObject* dtype) {
    PYTHON_GIL;

    PyArray_Descr* descr;
    if (!PyArray_DescrConverter(dtype, &descr)) {
        throw ConversionError(ssprintf("can not convert to np.dtype from %s",
                                       repr_pyobj(dtype).c_str()));
    }
    mgb_assert(!descr->metadata,
               "unexpected metadata in dtype: "
               "dtype_obj=%s metadata=%s",
               repr_pyobj(dtype).c_str(), repr_pyobj(descr->metadata).c_str());
    int type_num = to_mgb_supported_dtype_raw(descr->type_num);
    return PyArray_TypeObjectFromType(type_num);
}

TensorShape npy::vec2shape(const std::vector<size_t> &vec) {
    TensorShape shape;
    mgb_assert(vec.size() <= TensorShape::MAX_NDIM,
            "dim too large: %zd (max %zd)",
            vec.size(), TensorShape::MAX_NDIM);
    shape.ndim = vec.size();
    for (size_t i = 0; i < vec.size(); i ++) {
        if (!vec[i]) {
            shape.ndim = 0;
            break;
        }
        shape[i] = vec[i];
    }
    mgb_assert(shape.ndim, "shape should not be empty");
    return shape;
}

void mgb_init_numpy() {
    import_array1( );
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
