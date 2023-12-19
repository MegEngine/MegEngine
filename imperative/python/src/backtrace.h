#pragma once

#include <Python.h>
#include <frameobject.h>
#include <cstdint>
#include <memory>
#include <string>
#include "./helper.h"
#include "./pyext17.h"
#include "megbrain/common.h"
#include "megbrain/imperative/backtrace.h"
#include "megbrain/utils/metahelper.h"
#include "megbrain/utils/small_vector.h"
#include "pybind11/pybind11.h"
namespace py = pybind11;

namespace mgb::imperative::python {

struct FrameInfoCache;
struct FrameInfo;
using FrameInfoPtr = std::shared_ptr<FrameInfo>;

struct FrameInfo : public PyFrameInfo, public std::enable_shared_from_this<FrameInfo> {
    PyCodeObject* code_obj;
    int lineno;
    std::string scope;
    std::shared_ptr<FrameInfo> prev_frame;
    static std::unordered_map<ptrdiff_t, PyObjRefKeeper> code_ref_keeper;

    FrameInfo(PyCodeObject* code, int lineno) : code_obj{code}, lineno{lineno} {
        if (code_ref_keeper.find((ptrdiff_t)code_obj) == code_ref_keeper.end()) {
            Py_INCREF(code);
            code_ref_keeper[(ptrdiff_t)code_obj] = {(PyObject*)code_obj};
        }
    }
    std::string traceback() override;
    static std::pair<FrameInfoPtr, int> make(
            PyFrameObject* frame, FrameInfoCache* cache);
};

struct FrameInfoCache {
    std::vector<FrameInfoPtr> stack_cache;
    void update_cache(
            int key,
            const SmallVector<std::pair<PyFrameObject*, FrameInfoPtr>, 100>& frames);
    size_t size() { return stack_cache.size(); }
    FrameInfoPtr& operator[](int key) { return stack_cache[key]; }
    static int get_frame_key(PyFrameObject* frame);
    static FrameInfoCache* get_instance();
};

struct TraceKeyWrapper {
    int key;
    std::string scope;
    py::object orig_func;
    TraceKeyWrapper(int key, PyObject* func, std::string scope = "")
            : key{key}, scope{std::move(scope)} {
        if (func != NULL) {
            orig_func = py::reinterpret_steal<py::object>(func);
        }
    }
    static constexpr auto tp_name = pybind11::detail::_("TraceKeyWrapper");
    using wrap_t = pyext17::wrap<TraceKeyWrapper>;
    friend wrap_t;

    inline static TraceKeyWrapper* cast(PyObject* obj) {
        return reinterpret_cast<wrap_t*>(obj)->inst();
    }
    inline static TraceKeyWrapper* try_cast(PyObject* obj) {
        if (obj == NULL || !wrap_t::type().isinstance(obj))
            return nullptr;
        return cast(obj);
    }

    template <typename... Args>
    static PyObject* make(Args&&... args) {
        return wrap_t::cnew(std::forward<Args>(args)...);
    }

    PyObject* tp_call(PyObject* args, PyObject* kwargs) {
        if (orig_func.ptr() != nullptr) {
            return PyObject_Call(orig_func.ptr(), args, kwargs);
        }
        Py_RETURN_NONE;
    }
};

FrameInfoPtr get_frameinfo_from_pyframe(PyFrameObject* frame);
void record_py_backtrace();
void record_scope(PyFrameObject*, std::string);
std::string get_py_backtrace();

bool set_python_backtrace(bool);
bool set_transformation_backtrace(bool);
void init_backtrace_tss_key();

}  // namespace mgb::imperative::python
