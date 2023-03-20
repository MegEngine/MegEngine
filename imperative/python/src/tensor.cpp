#include "megbrain/common.h"
#include "megbrain/dtype.h"
#include "megbrain/imperative/backtrace.h"
#include "megbrain/imperative/cpp_cupti.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/backward_graph.h"
#include "megbrain/imperative/ops/utility.h"
#include "megbrain/imperative/profiler.h"
#include "megbrain/imperative/transformation.h"
#include "megbrain/imperative/transformations/dim_expansion.h"
#include "megbrain/imperative/transformations/dtype_promote.h"
#include "megbrain/imperative/transformations/eval.h"
#include "megbrain/imperative/transformations/format.h"
#include "megbrain/imperative/transformations/group_comm.h"
#include "megbrain/imperative/transformations/lazy.h"
#include "megbrain/imperative/transformations/scalar.h"
#include "megbrain/imperative/transformations/symbol.h"
#include "megbrain/imperative/transformations/trace.h"
#include "megbrain/imperative/utils/map.h"
#include "megbrain/opr/io.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/utils/stats.h"
#include "megdnn/algorithm_cache.h"

#include "./common.h"
#include "./grad.h"
#include "./graph_rt.h"
#include "./helper.h"
#include "./module_trace.h"
#include "./numpy_dtypes.h"
#include "./tensor.h"
#include "./tensor_utils.h"
#include "./transformation.h"

#include <object.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pytypes.h>
#include <pyerrors.h>
#include <iterator>
#include <range/v3/all.hpp>
#include <string>

#include <unordered_map>

#include "../../src/impl/mgb_cg_impl.h"
#include "./backtrace.h"

namespace py = pybind11;
namespace views = ranges::views;

namespace mgb::imperative::python {

interpreter::Interpreter::Channel* interpreter_for_py = nullptr;
PyTypeObject* py_tensor_type = nullptr;
PyTypeObject* py_varnode_type = nullptr;
pybind11::handle py_device_type = nullptr;
PyObject* cpp_use_symbolic_shape;

#define REGISTE_APPLY_FUNC(mode) \
    void set_##mode(py::object pyf) { mode = pyf.ptr(); }

REGISTE_APPLY_FUNC(cpp_use_symbolic_shape)

#undef REGISTE_APPLY_FUNC

PyArray_Descr* _dtype_promotion(PyObject* const* args, size_t nargs);
CompNode _get_device(PyObject* const* args, size_t nargs);

PyObject* py_apply(
        PyObject* self, PyObject* const* args, size_t nargs /* , PyObject* kwnames */) {
    try {
        // if (kwnames && PyTuple_GET_SIZE(kwnames)) {
        //     PyErr_SetString(PyExc_TypeError, "keyword argument not allowed");
        //     return nullptr;
        // }
        if (nargs < 2) {
            PyErr_SetString(
                    PyExc_TypeError,
                    "py_apply expects one Op and at least one tensor "
                    "as argument");
            return nullptr;
        }

        auto* py_op = args[0];

        ++args;
        --nargs;

        auto op = py::handle(py_op).cast<std::shared_ptr<OpDef>>();
        SmallVector<ValueRef, 8> tensors(nargs);

        mgb::CompNode target_cn;
        mgb::DType target_dtype;

        auto convert_pyinput_to_tensor = [&](size_t i) -> ValueRef {
            if (!target_dtype.valid()) {
                target_dtype = npy::dtype_np2mgb_descr(_dtype_promotion(args, nargs));
                target_cn = _get_device(args, nargs);
            }
            HostTensorND ht(target_cn);
            ht = npy::np2tensor(args[i], npy::Meth::copy_into(&ht), target_dtype);
            record_py_backtrace();
            //! operand in elemwise can't be None
            if (args[i] == Py_None) {
                throw py::type_error("the operand is None and is not supported.");
            } else if (PyArray_Check(args[i]) || PyList_Check(args[i])) {  // non scaler
                // py_tuple is not allowed here because of tracing
                return imperative::apply(
                        CreateTensor(CreateTensor::Const, target_cn, ht.layout()),
                        HostStorage::make(ht.storage()))[0];
            } else {  // scaler
                return imperative::apply(
                        CreateTensor(CreateTensor::Const, target_cn, target_dtype, {}),
                        HostStorage::make(ht.storage()))[0];
            }
        };

        bool is_varnode_apply = false;
        for (size_t i = 0; i < nargs; ++i) {
            if (PyObject_TypeCheck(args[i], py_varnode_type)) {
                is_varnode_apply = true;
            }
            if (TensorWrapper* tw = TensorWrapper::try_cast(args[i])) {
                tensors[i] = tw->m_tensor->data();
            } else if (
                    DTypePromoteCfg::convert_input_enabled &&
                    (op->same_type<Elemwise>() || op->same_type<ElemwiseMultiType>())) {
                tensors[i] = convert_pyinput_to_tensor(i);
            } else {
                PyErr_SetString(PyExc_TypeError, "py_apply expects tensor as inputs");
                return nullptr;
            }
        }
        record_py_backtrace();
        auto outputs = [&] { return imperative::apply(*op, tensors); }();
        size_t nout = outputs.size();
        auto ret = py::tuple(nout);
        PyTypeObject* py_type = is_varnode_apply ? py_varnode_type : py_tensor_type;
        for (size_t i = 0; i < nout; ++i) {
            ret[i] = TensorWrapper::make(py_type, std::move(outputs[i]));
        }
        return ret.release().ptr();
    }
    PYEXT17_TRANSLATE_EXC_RET(nullptr)
}
FrameInfoPtr get_current_frameinfo() {
    auto frame = PyEval_GetFrame();
    auto frameinfo = get_frameinfo_from_pyframe(frame);
    return frameinfo;
}

namespace {

template <typename T>
py::handle py_type() {
    if constexpr (std::is_same_v<T, py::int_>) {
        return (PyObject*)&PyLong_Type;
    } else if constexpr (std::is_same_v<T, py::float_>) {
        return (PyObject*)&PyFloat_Type;
    } else if constexpr (std::is_same_v<T, py::tuple>) {
        return (PyObject*)&PyTuple_Type;
    } else if constexpr (std::is_same_v<T, py::list>) {
        return (PyObject*)&PyList_Type;
    } else {
        static_assert(std::is_same_v<T, T>);
    }
}

template <typename T>
auto scalar2storage(T val, CompNode cn, DType dtype) {
    using max_ctype_t = DTypeScalar::max_ctype;
    DTypeScalar scalar(dtype);
    scalar.set_retain_dtype(val);
    HostTensorStorage storage(cn);
    auto* raw_ptr = reinterpret_cast<dt_byte*>(new max_ctype_t());
    std::shared_ptr<dt_byte> raw_storage = {
            raw_ptr, [](dt_byte* ptr) { delete reinterpret_cast<max_ctype_t*>(ptr); }};
    storage.only_reset_raw_storage(cn, dtype.size(), raw_storage, 0);
    std::memcpy(storage.ptr(), scalar.storage(), dtype.size());
    return HostStorage::make(std::move(storage));
}

template <typename ctype>
auto vec2storage(Span<DTypeScalar> vec, CompNode cn, DType dtype) {
    mgb_assert(vec.size() <= MEGDNN_MAX_NDIM);
    // TODO: use storage cache and modify ConstTensorCache to return (Host, Device)
    auto* raw_ptr = new ctype[MEGDNN_MAX_NDIM];
    for (size_t i = 0; i < vec.size(); ++i) {
        raw_ptr[i] = vec[i].get_cast<ctype>();
    }
    mgb_assert(sizeof(ctype) == dtype.size());
    std::shared_ptr<dt_byte> raw_storage = {
            reinterpret_cast<dt_byte*>(raw_ptr),
            [](dt_byte* ptr) { delete[] reinterpret_cast<ctype*>(ptr); }};
    HostTensorStorage storage(cn);
    storage.only_reset_raw_storage(cn, sizeof(ctype) * vec.size(), raw_storage, 0);
    return HostStorage::make(std::move(storage));
}

struct HostTensorArgs {
    ValueShape shape;
    DType dtype;
    HostStorage::ref_t storage;

    HostTensorND as_tensor_nd() const {
        HostTensorND ret(CompNode::default_cpu(), shape.as_tensor_shape(), dtype);
        ret.only_reset_raw_storage(*storage);
        return ret;
    }
};

template <typename seq_type, typename ctype>
bool pyseq2hval(seq_type obj, CompNode cn, DType dtype, HostTensorArgs& ret) {
    auto size = obj.size();
    if (size > MEGDNN_MAX_NDIM) {
        return false;
    }
    ctype items[size];
    for (size_t i = 0; i < size; ++i) {
        py::handle item = obj[i];
        if (item.get_type().is(py_type<py::int_>())) {
            items[i] = (ctype)(dt_int32)item.template cast<py::int_>();
        } else if (item.get_type().is(py_type<py::float_>())) {
            items[i] = (ctype)(dt_float32)item.template cast<py::float_>();
        } else {
            return false;
        }
    }
    mgb_assert(sizeof(ctype) == dtype.size());
    auto* raw_ptr = new ctype[size];
    std::shared_ptr<dt_byte> raw_storage = {
            reinterpret_cast<dt_byte*>(raw_ptr),
            [](dt_byte* ptr) { delete[] reinterpret_cast<ctype*>(ptr); }};
    HostTensorStorage storage(cn);
    storage.only_reset_raw_storage(cn, sizeof(ctype) * size, raw_storage, 0);
    std::memcpy(storage.ptr(), items, sizeof(ctype) * size);
    ret.dtype = dtype;
    ret.shape = {size};
    ret.storage = HostStorage::make(std::move(storage));
    return true;
}

template <typename seq_type>
bool pyseq2hval(seq_type obj, CompNode cn, HostTensorArgs& ret) {
    auto size = obj.size();
    if (size > MEGDNN_MAX_NDIM) {
        return false;
    }
    DTypeScalar items[size];
    DType dtype;
    for (size_t i = 0; i < size; ++i) {
        auto&& item = obj[i];
        if (item.get_type().is(py_type<py::int_>())) {
            items[i] = (dt_int32)item.template cast<py::int_>();
            if (!dtype.valid()) {
                dtype = dtype::Int32();
            } else if (dtype != dtype::Int32() && dtype != dtype::Float32()) {
                return false;
            }
        } else if (item.get_type().is(py_type<py::float_>())) {
            items[i] = (dt_float32)item.template cast<py::float_>();
            if (!dtype.valid()) {
                dtype = dtype::Float32();
            } else if (dtype == dtype::Int32()) {
                dtype = dtype::Float32();
            } else if (dtype != dtype::Float32()) {
                return false;
            }
        } else {
            return false;
        }
    }
    if (!dtype.valid()) {
        dtype = dtype::Float32();
    }
    ret.dtype = dtype;
    ret.shape = {size};
    if (dtype == dtype::Int32()) {
        ret.storage = vec2storage<dt_int32>({items, size}, cn, dtype);
    } else if (dtype == dtype::Float32()) {
        ret.storage = vec2storage<dt_float32>({items, size}, cn, dtype);
    } else {
        mgb_assert(false);
    }
    return true;
}

template <typename seq_type>
bool pyseq2hval(seq_type obj, CompNode cn, DType dtype, HostTensorArgs& ret) {
    if (dtype == dtype::Int32()) {
        return pyseq2hval<seq_type, dt_int32>(obj, cn, dtype, ret);
    } else if (dtype == dtype::Float32()) {
        return pyseq2hval<seq_type, dt_float32>(obj, cn, dtype, ret);
    } else if (!dtype.valid()) {
        return pyseq2hval<seq_type>(obj, cn, ret);
    } else {
        return false;
    }
}

bool pyarr2hval(py::array obj, CompNode cn, DType dtype, HostTensorArgs& ret) {
    auto data = obj.cast<py::array>();
    auto strides = data.strides();
    bool need_squeeze = false;
    for (size_t i = 0; i < data.ndim(); ++i) {
        if (strides[i] == 0) {
            need_squeeze = true;
            break;
        }
    }
    if (need_squeeze) {
        std::vector<size_t> shape;
        for (size_t i = 0; i < data.ndim(); ++i) {
            shape.push_back(data.shape(i));
        }
        data = data.squeeze();
        data.resize(shape);
    }
    HostTensorND retnd(cn);
    retnd = npy::np2tensor(data.ptr(), npy::Meth::copy_into(&retnd), dtype);
    if (!dtype.valid()) {
        dtype = retnd.dtype();
    }
    mgb_assert(
            retnd.layout().is_empty() || retnd.layout().is_contiguous(),
            "host value should be continuous");
    for (size_t i = 0; i < data.ndim(); ++i) {
        ret.shape[ret.shape.ndim++] = data.shape(i);
    }
    ret.dtype = dtype;
    ret.storage = HostStorage::make(retnd.storage());
    return true;
}

bool pyint2hval(py::int_ obj, CompNode cn, DType dtype, HostTensorArgs& ret) {
    if (!dtype.valid()) {
        dtype = dtype::Int32();
    }
    ret.dtype = dtype;
    ret.storage = scalar2storage((dt_int32)obj, cn, dtype);
    return true;
}

bool pyfloat2hval(py::float_ obj, CompNode cn, DType dtype, HostTensorArgs& ret) {
    if (!dtype.valid()) {
        dtype = dtype::Float32();
    }
    ret.dtype = dtype;
    ret.storage = scalar2storage((dt_float32)obj, cn, dtype);
    return true;
}

HostTensorArgs pyobj2hval(py::object obj, CompNode cn, DType dtype) {
    HostTensorArgs ret;
    bool success = false;
    // check order: float -> int -> tuple(int -> float) -> list(int -> float)
    // only handle `exact` pytype, isinstance also accepts subtype
    // for example, isinstance(True, int) == True
    if (obj.get_type().is(py_type<py::float_>())) {
        success = pyfloat2hval(py::float_(obj), cn, dtype, ret);
    } else if (obj.get_type().is(py_type<py::int_>())) {  // py::bool_ is py::int_
        success = pyint2hval(py::int_(obj), cn, dtype, ret);
    } else if (obj.get_type().is(py_type<py::tuple>())) {
        success = pyseq2hval<py::tuple>(py::tuple(obj), cn, dtype, ret);
    } else if (obj.get_type().is(py_type<py::list>())) {
        success = pyseq2hval<py::list>(py::list(obj), cn, dtype, ret);
    } else if (obj.is_none()) {
        obj = py::list(0);
    }
    if (!success) {
        success = pyarr2hval(obj, cn, dtype, ret);
    }
    mgb_assert(success);
    return ret;
}

struct PyArgDesc {
    const char* name;
    py::object (*default_value)();
};

struct PyArgDescs {
    std::vector<PyArgDesc> items;
    ssize_t (*name2idx)(const char* name);
};

py::tuple parse_args(py::tuple args, const PyArgDescs& descs) {
    size_t nr_args = args.size();
    size_t nr_items = descs.items.size();
    mgb_assert(nr_args <= nr_items, "too many args");
    if (nr_args == nr_items) {
        return args;
    }
    py::tuple ret(nr_items);
    for (size_t i = 0; i < nr_args; ++i) {
        ret[i] = args[i];
    }
    for (size_t i = nr_args; i < nr_items; ++i) {
        ret[i] = descs.items[i].default_value();
    }
    return ret;
}

py::tuple parse_args_and_kwargs(
        py::tuple args, py::dict kwargs, const PyArgDescs& descs) {
    size_t nr_args = args.size();
    size_t nr_kwargs = kwargs.size();
    size_t nr_items = descs.items.size();
    mgb_assert(nr_args + nr_kwargs <= nr_items, "too many args");
    if (nr_args == nr_items) {
        return args;
    }
    py::tuple ret(nr_items);
    for (size_t i = 0; i < nr_args; ++i) {
        ret[i] = args[i];
    }
    bool has_value[nr_items - nr_args];
    for (size_t i = nr_args; i < nr_items; ++i) {
        has_value[i - nr_args] = false;
    }
    for (auto&& [k, v] : kwargs) {
        auto key = py::str(k).cast<std::string>();
        ssize_t index = descs.name2idx(key.c_str());
        mgb_assert(index >= nr_args);
        ret[index] = v;
        has_value[index - nr_args] = true;
    }
    for (size_t i = nr_args; i < nr_items; ++i) {
        if (!has_value[i - nr_args]) {
            ret[i] = descs.items[i].default_value();
        }
    }
    return ret;
}

CompNode as_comp_node(const std::string& name) {
    thread_local struct {
        std::string name;
        CompNode cn;
    } cached;
    if (cached.name != name) {
        cached.name = name;
        cached.cn = CompNode::load(name);
    }
    return cached.cn;
}

CompNode as_comp_node(py::object py_device) {
    std::optional<std::string> device_name;
    if (py_device.is_none() || py::str::check_(py_device)) {
        auto cls = py::handle(reinterpret_cast<PyObject*>(py_tensor_type));
        auto dmap_callback = cls.attr("dmap_callback");
        std::string name;
        if (dmap_callback.is_none() && py_device.is_none()) {
            name = get_default_device();
        } else {
            if (py_device.is_none()) {
                py_device = py::str(get_default_device());
            }
            if (!dmap_callback.is_none()) {
                py_device = dmap_callback(py_device);
            }
            name = py::str(py_device).cast<std::string>();
        }
        return as_comp_node(name);
    } else {
        if (py::isinstance(py_device, py_device_type)) {
            py_device = py_device.attr("_cn");
        }
        mgb_assert(py::isinstance(py_device, py_comp_node_type));
        return py_device.cast<CompNode>();
    }
}

template <char... Chars>
bool compare_cstr(const char* cstr) {
    return (((*cstr++) == Chars) && ...) && *cstr == '\0';
}

ssize_t name2idx(const char* name) {
    const char* ch = name;
    // TODO: trie
    // clang-format off
    switch (*ch++) {
    case 'd':
        switch (*ch++) {
        // data
        case 'a': return compare_cstr<'t', 'a'>(ch) ? 0 : -1;
        // dtype
        case 't': return compare_cstr<'y', 'p', 'e'>(ch) ? 1 : -1;
        // device
        case 'e': return compare_cstr<'v', 'i', 'c', 'e'>(ch) ? 2 : -1;
        }
    case 'i':
        // is_const
        return compare_cstr<'s', '_', 'c', 'o', 'n', 's', 't'>(ch) ? 3 : -1;
    case 'n':
        switch (*ch++) {
        // no_cache
        case 'o': return compare_cstr<'_', 'c', 'a', 'c', 'h', 'e'>(ch) ? 4 : -1;
        // name
        case 'a': return compare_cstr<'m', 'e'>(ch) ? 5 : -1;
        }
    case 'f':
        // format
        return compare_cstr<'o', 'r', 'm', 'a', 't'>(ch) ? 6 : -1;
    }
    // clang-format on
    return -1;
}

}  // namespace

TensorWrapper::TensorWrapper(PyObject* args, PyObject* kwargs) {
    static PyArgDescs descs = {
            {
                    {"data", []() -> py::object { return py::none(); }},
                    {"dtype", []() -> py::object { return py::none(); }},
                    {"device", []() -> py::object { return py::none(); }},
                    {"is_const", []() -> py::object { return py::bool_(false); }},
                    {"no_cache", []() -> py::object { return py::bool_(false); }},
                    {"name", []() -> py::object { return py::none(); }},
                    {"format", []() -> py::object { return py::none(); }},
            },
            name2idx};
    py::detail::loader_life_support life_sup;  // FIXME!!!required to cast DType
    auto tup = py::reinterpret_borrow<py::tuple>(args);
    if (kwargs) {
        tup = parse_args_and_kwargs(
                tup, py::reinterpret_borrow<py::dict>(kwargs), descs);
    } else {
        tup = parse_args(tup, descs);
    }
    mgb_assert(tup.size() == 7);
    if (auto* t = try_cast(tup[0].ptr())) {
        m_tensor = t->m_tensor;
        // TODO: merge two path in arg parse
        if (!tup[1].is_none()) {
            auto dtype = tup[1].cast<DType>();
            mgb_assert(
                    dtype == m_tensor->dtype(), "dtype mismatch: %s vs %s",
                    dtype.name(), m_tensor->dtype().name());
        }
        if (!tup[2].is_none()) {
            auto device = as_comp_node(tup[2]);
            mgb_assert(
                    device == m_tensor->comp_node(), "device mismatch: %s vs %s",
                    device.to_string().c_str(),
                    m_tensor->comp_node().to_string().c_str());
        }
        mgb_assert(!tup[3].cast<bool>(), "expect is_const == False, got True");
        bool no_cache = tup[4].cast<bool>();
        if (no_cache) {
            // always copy because it's hard to tell whether this tensor is cached
            m_tensor = m_tensor->copy();
        }
        // ignore name
        if (!tup[6].is_none()) {
            Format format = tup[6].cast<std::string>();
            mgb_assert(
                    format == m_tensor->format(), "format mismatch: %s vs %s",
                    format.to_string().c_str(), m_tensor->format().to_string().c_str());
        }
    } else {
        auto data = tup[0];
        DType dtype = tup[1].cast<DType>();
        CompNode cn = as_comp_node(tup[2]);
        bool is_const = tup[3].cast<bool>();
        bool no_cache = tup[4].cast<bool>();
        std::string name;
        if (!tup[5].is_none()) {
            name = tup[5].cast<std::string>();
        }
        Format format;
        if (!tup[6].is_none()) {
            format = tup[6].cast<std::string>();
        }

        {
            CreateTensor::Kind kind = is_const ? CreateTensor::Const
                                    : no_cache ? CreateTensor::Unique
                                               : CreateTensor::Common;
            ValueRef val;
            if (py::isinstance(data, Py_Varnode)) {
                cg::VarNode* m_node = py::handle(data).cast<cg::VarNode*>();
                val = imperative::apply(
                        CreateNode(m_node), Span<ValueRef>(nullptr, nullptr))[0];
            } else {
                auto&& hval = pyobj2hval(data, cn, dtype);
                val = imperative::apply(
                        CreateTensor(kind, cn, hval.dtype, hval.shape, format),
                        hval.storage)[0];
            }
            m_tensor.emplace(val);
        }

        if (!name.empty()) {
            m_tensor->reset(imperative::apply(RenameValue(name), m_tensor->data())[0]);
        }
    }
    mgb_assert(m_tensor->data());
}

PyObject* TensorWrapper::module_trace_info() {
    if (auto module_trace_info =
                ModuleTraceTransformation::module_trace_info_map.try_get(
                        m_tensor->data())) {
        if (module_trace_info->ptr()) {
            return module_trace_info->inc_ref().ptr();
        }
    }
    PyErr_SetString(
            PyExc_AttributeError,
            "Has no attribute named \'_NodeMixin__node\', please "
            "set it first");
    return nullptr;
}

void TensorWrapper::set_module_trace_info(PyObject* obj) {
    // TODO: erase when obj == nullptr
    ModuleTraceTransformation::module_trace_info_map[m_tensor->data()] =
            py::reinterpret_borrow<py::object>(obj);
}

void TensorWrapper::_set_format(PyObject* dest) {
    auto py_dest = py::reinterpret_borrow<py::object>(dest);
    auto format = py_dest.cast<std::string>();
    m_tensor->set_format(format);
}

void TensorWrapper::_set_name(PyObject* dest) {
    auto py_dest = py::reinterpret_borrow<py::object>(dest);
    auto name = py_dest.cast<std::string>();

    m_tensor->set_name(name);
}

PyObject* TensorWrapper::_detail() {
    return py::str(m_tensor->data().unwrap().to_string()).release().ptr();
}

void TensorWrapper::_watch() {
    m_tensor->data().watch();
}

PyObject* TensorWrapper::shape() {
    auto shape = m_tensor->shape();

    if (!shape) {
        Py_RETURN_NONE;
    }
    py::tuple ret(shape->ndim);
    for (size_t i = 0; i < shape->ndim; ++i) {
        ret[i] = shape->at(i);
    }
    return ret.release().ptr();
}

PyObject* TensorWrapper::dtype() {
    return py::cast(m_tensor->dtype()).release().ptr();
}

PyObject* TensorWrapper::device() {
    return py::cast(m_tensor->comp_node()).release().ptr();
}

PyObject* TensorWrapper::format() {
    return py::cast(m_tensor->format().to_string()).release().ptr();
}

PyObject* TensorWrapper::numpy() {
    auto hv = m_tensor->numpy();
    if (!hv) {
        PyErr_SetString(PyExc_ValueError, "tensor invalid");
        return nullptr;
    }
    auto arr = py::reinterpret_steal<py::array>(
            npy::ndarray_from_tensor(hv->as_nd(true), npy::ShareType::TRY_SHARE));
    if (hv->shape().is_scalar()) {
        mgb_assert(PyArray_Check(arr.ptr()));
        return PyArray_Squeeze(reinterpret_cast<PyArrayObject*>(arr.ptr()));
    }
    return arr.release().ptr();
}

void TensorWrapper::reset(PyObject* tensor) {
    TensorWrapper* t = TensorWrapper::try_cast(tensor);
    if (!t) {
        throw py::type_error("expect Tensor");
    }
    m_tensor->reset(t->m_tensor->data());
}

PyObject* TensorWrapper::detach() {
    auto detached = imperative::apply(DetachGrad(), m_tensor->data())[0];
    return TensorWrapper::make(py_tensor_type, detached).release().ptr();
}

PyObject* TensorWrapper::_dev_tensor() {
    auto dv = m_tensor->data().dev_tensor();
    // TODO: handle scalar
    return py::cast(dv->as_nd(true)).release().ptr();
}

void TensorWrapper::_drop() {
    imperative::apply(DTRCommand(DTRCommand::Drop), m_tensor->data());
}

PyObject* TensorWrapper::isscalar() {
    if (m_tensor->is_scalar()) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

PyObject* TensorWrapper::_var() {
    TypedValueRef<NodeValue> value =
            imperative::apply(GetVarVal(), m_tensor->data())[0].as_ref<NodeValue>();
    auto* node = value->node();
    return py::cast(node).release().ptr();
}

PyObject* TensorWrapper::_graph() {
    TypedValueRef<NodeValue> value =
            imperative::apply(GetVarVal(), m_tensor->data())[0].as_ref<NodeValue>();
    auto* graph = value->graph();
    return py::cast(graph).release().ptr();
}

struct TensorWeakRef {
    ValueWeakRef data;

    TensorWeakRef(const TensorWrapper& tw) : data(tw.m_tensor->data()) {}

    py::object operator()() {
        if (auto p = data.lock()) {
            return TensorWrapper::make(py_tensor_type, p);
        }
        return py::none();
    }
};

#ifdef METH_FASTCALL
#define MGE_PY_INTERFACE(NAME, FUNC) \
    { #NAME, (PyCFunction)FUNC, METH_FASTCALL, nullptr }
#else
#define WRAP_FUNC_PY35(FUNC)                                \
    PyObject* py35_##FUNC(PyObject* self, PyObject* args) { \
        auto* arr = &PyTuple_GET_ITEM(args, 0);             \
        auto size = PyTuple_GET_SIZE(args);                 \
        return FUNC(self, arr, size);                       \
    }

WRAP_FUNC_PY35(py_apply);
WRAP_FUNC_PY35(dtype_promotion);
WRAP_FUNC_PY35(get_device);
WRAP_FUNC_PY35(make_shape_tuple);
WRAP_FUNC_PY35(getitem_cpp);
WRAP_FUNC_PY35(setitem_cpp);
WRAP_FUNC_PY35(split_cpp);
WRAP_FUNC_PY35(expand_dims_cpp);
WRAP_FUNC_PY35(squeeze_cpp);
WRAP_FUNC_PY35(transpose_cpp);
WRAP_FUNC_PY35(broadcast_cpp);
WRAP_FUNC_PY35(reshape_cpp);
WRAP_FUNC_PY35(adaptive_pool2d_cpp);
WRAP_FUNC_PY35(Const);
WRAP_FUNC_PY35(astype_cpp);
WRAP_FUNC_PY35(matmul_cpp);
WRAP_FUNC_PY35(batched_matmul_cpp);
WRAP_FUNC_PY35(convert_single_value_cpp);
WRAP_FUNC_PY35(convert_inputs_cpp);
WRAP_FUNC_PY35(astensor1d_cpp);
WRAP_FUNC_PY35(pixel_shuffle_cpp);
#undef WRAP_FUNC_PY35
#define MGE_PY_INTERFACE(NAME, FUNC) \
    { #NAME, (PyCFunction)py35_##FUNC, METH_VARARGS, nullptr }
#endif

void init_tensor(py::module m) {
    imperative::Tensor::static_initialize();
    init_backtrace_tss_key();
    // Transformations
    static auto& transformations = TransformationManager::get_instance();

    using Segment = TransformationManager::Segment;

    using Channel = interpreter::Interpreter::Channel;

    auto* channel =
            imperative::ResourceManager::create_global<std::unique_ptr<Channel>>(
                    interpreter::Interpreter::inst().create_channel())
                    ->get();
    interpreter_for_py = channel;
    MGB_MARK_USED_VAR(
            transformations
                    .register_at<Segment::Eval>(
                            std::make_shared<InterpreterTransformation>(
                                    std::shared_ptr<Channel>(channel, [](Channel*) {})))
                    .release());
    MGB_MARK_USED_VAR(transformations
                              .register_at<Segment::Scalar>(
                                      std::make_shared<ScalarTransformation>())
                              .release());
    MGB_MARK_USED_VAR(transformations
                              .register_at<Segment::Symbol>(
                                      std::make_shared<SymbolTransformation>())
                              .release());
    MGB_MARK_USED_VAR(transformations
                              .register_at<Segment::DTypePromote>(
                                      std::make_shared<DTypePromoteTransformation>())
                              .release());
    MGB_MARK_USED_VAR(transformations
                              .register_at<Segment::DimExpansion>(
                                      std::make_shared<DimExpansionTransformation>())
                              .release());
    auto format_trans = std::make_shared<FormatTransformation>();
    MGB_MARK_USED_VAR(
            transformations.register_at<Segment::Format>(format_trans).release());

    static py::exception<interpreter::AsyncError> py_async_error(
            m, "AsyncError", PyExc_RuntimeError);
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (const interpreter::AsyncError& e) {
            pyext17::pybind11_translate_exception(e.nested_ptr());
            if (PyErr_Occurred()) {
                PyObject *exc, *val, *tb;
                PyErr_Fetch(&exc, &val, &tb);
                PyErr_NormalizeException(&exc, &val, &tb);
                if (tb) {
                    PyException_SetTraceback(val, tb);
                }
                auto val2 = py_async_error.py::object::operator()(
                        "An async error is reported. See above for the actual cause."
                        " Hint: This is where it is reported, not where it happened."
                        " You may call `megengine.config.async_level = 0 "
                        "to get better error reporting.");
                PyException_SetCause(
                        val2.ptr(), val);  // PyException_SetCause steals reference
                Py_XDECREF(exc);
                Py_XDECREF(tb);
                PyErr_Restore(
                        py_async_error.inc_ref().ptr(), val2.release().ptr(), nullptr);
            } else {
                py_async_error("Unkown async error");
            }
        }
    });

    // Tensor
    auto* tensor_type =
            TensorWrapper::wrap_t::type()
                    .def<&TensorWrapper::numpy>("numpy")
                    .def_getset<&TensorWrapper::shape>("shape")
                    .def_getset<&TensorWrapper::dtype>("dtype")
                    .def_getset<&TensorWrapper::device>("device")
                    .def<&TensorWrapper::format>("format")
                    .def<&TensorWrapper::reset>("_reset")
                    .def<&TensorWrapper::isscalar>("_isscalar")
                    .def<&TensorWrapper::detach>("detach")
                    // TODO: remove this
                    .def<&TensorWrapper::_dev_tensor>("_dev_tensor")
                    .def<&TensorWrapper::_drop>("_drop")
                    .def<&TensorWrapper::_detail>("_detail")
                    .def<&TensorWrapper::_set_format>("_set_format")
                    .def<&TensorWrapper::_set_name>("_set_name")
                    .def<&TensorWrapper::_watch>("_watch")
                    .def<&TensorWrapper::_var>("var")
                    .def<&TensorWrapper::_graph>("graph")
                    .def_getset<
                            &TensorWrapper::module_trace_info,
                            &TensorWrapper::set_module_trace_info>("_NodeMixin__node")
                    .finalize();
    if (!tensor_type)
        throw py::error_already_set();
    py::setattr(m, "Tensor", tensor_type);

    auto* tracekey_type = TraceKeyWrapper::wrap_t::type().finalize();
    py::setattr(m, "tracekey", tracekey_type);

    py::enum_<Format::Type>(m, "FormatType")
            .value("DEFAULT", Format::Type::DEFAULT)
            .value("NCHW", Format::Type::NCHW)
            .value("NHWC", Format::Type::NHWC)
            .export_values();

    py::class_<TensorWeakRef>(m, "TensorWeakRef")
            .def(py::init<const TensorWrapper&>())
            .def("__call__", &TensorWeakRef::operator());

    static PyMethodDef method_defs[] = {
            MGE_PY_INTERFACE(apply, py_apply),
            MGE_PY_INTERFACE(dtype_promotion, dtype_promotion),
            MGE_PY_INTERFACE(get_device, get_device),
            MGE_PY_INTERFACE(make_shape_tuple, make_shape_tuple),
            MGE_PY_INTERFACE(getitem_cpp, getitem_cpp),
            MGE_PY_INTERFACE(setitem_cpp, setitem_cpp),
            MGE_PY_INTERFACE(split_cpp, split_cpp),
            MGE_PY_INTERFACE(expand_dims_cpp, expand_dims_cpp),
            MGE_PY_INTERFACE(squeeze_cpp, squeeze_cpp),
            MGE_PY_INTERFACE(transpose_cpp, transpose_cpp),
            MGE_PY_INTERFACE(broadcast_cpp, broadcast_cpp),
            MGE_PY_INTERFACE(reshape_cpp, reshape_cpp),
            MGE_PY_INTERFACE(adaptive_pool2d_cpp, adaptive_pool2d_cpp),
            MGE_PY_INTERFACE(Const, Const),
            MGE_PY_INTERFACE(astype_cpp, astype_cpp),
            MGE_PY_INTERFACE(matmul_cpp, matmul_cpp),
            MGE_PY_INTERFACE(batched_matmul_cpp, batched_matmul_cpp),
            MGE_PY_INTERFACE(convert_single_value_cpp, convert_single_value_cpp),
            MGE_PY_INTERFACE(convert_inputs_cpp, convert_inputs_cpp),
            MGE_PY_INTERFACE(astensor1d_cpp, astensor1d_cpp),
            MGE_PY_INTERFACE(pixel_shuffle_cpp, pixel_shuffle_cpp),
            {nullptr, nullptr, 0, nullptr}};
    for (auto&& def : method_defs) {
        if (def.ml_meth != nullptr) {
            auto* func = PyCFunction_NewEx(&def, nullptr, nullptr);
            if (!func)
                throw py::error_already_set();
            py::setattr(m, def.ml_name, func);
        }
    }

    static constexpr auto sync_py_task_q = [] {
        py::gil_scoped_release _;
        py_task_q.wait_all_task_finish();
    };

    m.def("clear_candidates", [channel]() { channel->clear_candidates(); });
    m.def("set_option", [channel](std::string name, size_t value) {
        channel->set_option(name, value);
    });
    m.def("get_option",
          [channel](std::string name) { return channel->get_option(name); });
    m.def("push_scope", [channel](std::string name) {
        Transformation::push_scope(name);
        channel->push_scope(name);
    });
    m.def("record_scope", [](py::object frame, std::string name) {
        mgb_assert(PyFrame_Check(frame.ptr()));
        record_scope((PyFrameObject*)frame.ptr(), std::move(name));
    });
    m.def("pop_scope", [channel](std::string name) {
        channel->pop_scope(name);
        Transformation::pop_scope(name);
    });
    m.def("start_profile", [channel](imperative::Profiler::options_t options) {
        channel->sync();
        imperative::Profiler::load_options(std::move(options));
        imperative::Profiler::start_profile();
        channel->start_profile();
    });
    m.def("stop_profile", [channel]() -> std::function<void(std::string, std::string)> {
        channel->stop_profile();
        channel->sync();
        CompNode::sync_all();
        imperative::Profiler::stop_profile();
        auto results = std::make_shared<imperative::Profiler::bundle_t>(
                imperative::Profiler::collect());
        return [results = results](std::string basename, std::string format) mutable {
            imperative::Profiler::dump_profile(basename, format, std::move(*results));
            results = nullptr;
        };
    });
    m.def("enable_cupti", &cupti::enable);
    m.def("disable_cupti", &cupti::disable);
    m.def("cupti_available", &cupti::available);

    static std::unique_ptr<CleanupGuard<>> group_comm_guard;
    m.def("group_start", []() {
        auto commtrans = std::make_shared<GroupCommTransformation>();
        group_comm_guard = transformations.register_at<Segment::GroupComm>(commtrans);
    });
    m.def("group_end", []() { group_comm_guard.reset(); });
    m.def("sync", [channel]() {
        if (channel->check_available()) {
            channel->sync();
        }
        sync_py_task_q();
    });
    m.def("full_sync", [channel]() {
        if (channel->check_available()) {
            channel->sync();
        }
        CompNode::sync_all();
        CompNode::foreach ([](CompNode cn) {
            auto err = cn.check_async_error();
            mgb_assert(!err, "%s", err->what());
        });
        sync_py_task_q();
    });
    m.def("close", [channel]() {
        // sync channel and compnode before close to ensure all tasks have been completed
        if (channel->check_available()) {
            channel->sync();
        }
        CompNode::sync_all();
        CompNode::foreach ([](CompNode cn) {
            auto err = cn.check_async_error();
            mgb_assert(!err, "%s", err->what());
        });
        channel->close();
        sync_py_task_q();
    });

    // GradTransformation
    py::handle grad_key_type =
            GradKeyWrapper::wrap_t::type()
                    .def<&GradKeyWrapper::attach>("attach")
                    .def<&GradKeyWrapper::is_attached_to>("is_attached_to")
                    .def_getset<&GradKeyWrapper::get_name, &GradKeyWrapper::set_name>(
                            "name")
                    .def<&GradKeyWrapper::enter>("enter")
                    .def<&GradKeyWrapper::exit>("exit")
                    .def<&GradKeyWrapper::suppress>("suppress")
                    .def<&GradKeyWrapper::resume>("resume")
                    .finalize();
    if (!grad_key_type)
        throw py::error_already_set();
    py::setattr(m, "GradKey", grad_key_type);
    m.def("backward", &GradKeyWrapper::backward);
    m.def("get_backward_closure", &GradKeyWrapper::get_backward_closure);

    m.def("set_py_tensor_type", [](py::object type_obj) {
        py_tensor_type = reinterpret_cast<PyTypeObject*>(type_obj.inc_ref().ptr());
    });

    m.def("set_py_varnode_type", [](py::object type_obj) {
        py_varnode_type = reinterpret_cast<PyTypeObject*>(type_obj.inc_ref().ptr());
    });

    m.def("set_py_device_type",
          [](py::object type_obj) { py_device_type = type_obj.inc_ref(); });

    /**
     * \brief trace proxy
     *
     */
    struct Trace {
        bool symbolic = false;
        bool no_exec = false;
        bool capture_as_const = false;
        bool profile = false;
        bool record_input_shapes = false;
        py::function options_visitor;
        std::shared_ptr<TracingTransformation> tracing;
        std::shared_ptr<CompiledTransformation> compiled;
        std::shared_ptr<LazyEvalTransformation> lazy_eval;
        std::pair<size_t, std::shared_ptr<GraphProfiler>> profiler;
        std::optional<TraceResult> trace_result;
        std::function<bool(py::object, py::object)> array_comparator;
        std::unique_ptr<CleanupGuard<>> tracing_guard;
        std::unique_ptr<CleanupGuard<>> compiled_guard;
        std::unique_ptr<CleanupGuard<>> lazy_eval_guard;

        bool compare_value(ValueRef lhs, ValueRef rhs) {
            auto lvalue = lhs.cast_ref<HostValue>();
            auto rvalue = rhs.cast_ref<HostValue>();
            if (lvalue->shape() != rvalue->shape()) {
                return false;
            }
            if (lvalue->shape().total_nr_elems() == 1) {
                return lvalue->item() == rvalue->item();
            }
            HostTensorND lnd = lvalue->as_nd(true);
            HostTensorND rnd = rvalue->as_nd(true);
            auto larr = py::reinterpret_steal<py::array>(
                    npy::ndarray_from_tensor(lnd, npy::ShareType::TRY_SHARE));
            auto rarr = py::reinterpret_steal<py::array>(
                    npy::ndarray_from_tensor(rnd, npy::ShareType::TRY_SHARE));
            return array_comparator(larr, rarr);
        }

        void enter() {
            auto& self = *this;
            if (!self.trace_result) {  // untraced
                self.tracing = std::make_shared<TracingTransformation>(
                        self.capture_as_const, self.record_input_shapes);
                if (self.symbolic) {
                    self.lazy_eval =
                            std::make_shared<LazyEvalTransformation>(self.no_exec);
                    self.options_visitor(py::cast(&self.lazy_eval->options()));
                }
            } else if (!self.compiled) {  // traced but not compiled
                using namespace std::placeholders;
                self.compiled = std::make_shared<CompiledTransformation>(
                        *self.trace_result, self.record_input_shapes);
                self.compiled->set_value_comparator(
                        std::bind(&Trace::compare_value, this, _1, _2));
                self.options_visitor(py::cast(&self.compiled->options()));
                try {
                    self.compiled->compile();
                } catch (const std::exception& e) {
                    mgb_log_error("error in trace: %s", e.what());
                }
            }
            // register transformations
            if (self.compiled) {
                if (self.profile) {
                    auto& current_graph = self.compiled->graph();
                    if (self.profiler.first != self.compiled->graph().id()) {
                        // graph changed
                        self.profiler = std::make_pair(
                                current_graph.id(),
                                std::make_shared<GraphProfiler>(&current_graph));
                    }
                }
                compiled_guard =
                        transformations.register_at<Segment::Trace>(self.compiled);
                // start execute because InputCallback depends
                self.compiled->execute();
            } else if (self.tracing) {
                tracing_guard =
                        transformations.register_at<Segment::Trace>(self.tracing);
                if (self.lazy_eval) {
                    lazy_eval_guard =
                            transformations.register_at<Segment::Eval>(self.lazy_eval);
                }
            } else {
                mgb_throw(MegBrainError, "invalid state: neither tracing nor compiled");
            }
        }

        void exit() {
            auto& self = *this;
            if (self.tracing) {
                tracing_guard.reset();
                self.trace_result = self.tracing->get_result();
                self.tracing.reset();
                if (self.lazy_eval) {
                    auto lazy_eval = std::move(self.lazy_eval);
                    lazy_eval_guard.reset();
                    lazy_eval->check_exception();
                }
            } else if (self.compiled) {
                compiled_guard.reset();
                self.compiled->wait();
            } else {
                mgb_throw(MegBrainError, "invalid state: neither tracing nor compiled");
            }
        }

        VarNodeArray dump(
                std::shared_ptr<ComputingGraph> graph,
                std::vector<std::tuple<std::string, std::string, TensorShape>> inputs,
                std::vector<std::pair<std::string, std::string>> outputs,
                bool prefer_input_names) {
            auto& self = *this;
            mgb_assert(self.trace_result);
            // mark is like "arg_0", "kwarg_xxx", "output_0" ...
            std::unordered_map<std::string, size_t> mark2var;
            for (size_t i = 0; i < self.trace_result->vars.size(); ++i) {
                auto& name = self.trace_result->vars[i].mark;
                if (!name.empty()) {
                    mark2var[name] = i;
                }
            }
            std::vector<std::tuple<size_t, std::string, TensorShape>> input_vars;
            std::vector<std::pair<size_t, std::string>> output_vars;
            for (auto&& [input_mark, input_name, input_shape] : inputs) {
                mgb_assert(input_shape.ndim, "input shape invalid");
                input_vars.push_back(
                        {mark2var.at(input_mark), input_name, input_shape});
            }
            for (auto&& [output_name, repr] : outputs) {
                output_vars.push_back({mark2var.at(output_name), repr});
            }
            self.options_visitor(py::cast(&graph->options()));
            auto vars = self.trace_result->dump(
                    *graph, input_vars, output_vars, prefer_input_names);
            return vars;
        }
    };

    py::class_<Trace>(m, "Trace")
            .def(py::init<>())
            .def_readwrite("record_input_shapes", &Trace::record_input_shapes)
            .def_readwrite("array_comparator", &Trace::array_comparator)
            .def_readwrite("profile", &Trace::profile)
            .def_property_readonly(
                    "options",
                    [](Trace& self) {
                        if (self.compiled) {
                            return &self.compiled->options();
                        } else {
                            return (ComputingGraph::Options*)nullptr;
                        }
                    })
            .def("get_profile",
                 [](Trace& self) -> py::object {
                     if (self.profiler.second && self.compiled) {
                         auto json = self.profiler.second->to_json_full(
                                 self.compiled->graph().current_comp_seq());
                         return py::str(json->to_string());
                     } else {
                         return py::none();
                     }
                 })
            .def_readwrite("symbolic", &Trace::symbolic)
            .def_readwrite("capture_as_const", &Trace::capture_as_const)
            .def_readwrite("no_exec", &Trace::no_exec)
            .def_readwrite("options_visitor", &Trace::options_visitor)
            .def("enter", &Trace::enter)
            .def("exit", &Trace::exit)
            .def("dump", &Trace::dump)
            .def("begin_excluded_region",
                 [](Trace& self) {
                     mgb_assert(bool(self.tracing) ^ bool(self.compiled));
                     if (self.tracing) {
                         self.tracing_guard.reset();
                     } else if (self.compiled) {
                         self.compiled_guard.reset();
                     }
                 })
            .def("end_excluded_region", [](Trace& self) {
                mgb_assert(bool(self.tracing) ^ bool(self.compiled));
                if (self.tracing) {
                    self.tracing_guard =
                            transformations.register_at<Segment::Trace>(self.tracing);
                } else if (self.compiled) {
                    self.compiled_guard =
                            transformations.register_at<Segment::Trace>(self.compiled);
                }
            });

    m.def("name_tensor", [](std::string name, py::object tensor) {
        auto* tw = TensorWrapper::try_cast(tensor.ptr());
        auto output = imperative::apply(TraceMarkVar(name), tw->m_tensor->data())[0];
        tw->m_tensor->reset(output);
    });

    m.def("is_grad_attached", [](std::vector<py::object> tensors) -> bool {
        SmallVector<ValueRef> values(tensors.size());
        for (size_t i = 0; i < tensors.size(); ++i) {
            values[i] = tensors[i].cast<TensorWrapper>().m_tensor->data();
        }
        auto outputs = imperative::apply(GetGradKey(), values);
        if (outputs[0].is<GradKeyValue>()) {
            return true;
        } else {
            return false;
        }
    });

    m.def("get_grad_key", [](std::vector<py::object> tensors) -> py::object {
        SmallVector<ValueRef> values(tensors.size());
        for (size_t i = 0; i < tensors.size(); ++i) {
            values[i] = tensors[i].cast<TensorWrapper>().m_tensor->data();
        }
        auto output = imperative::apply(GetGradKey(), values)[0];
        if (!output) {
            return py::none();
        }
        return py::reinterpret_borrow<py::object>(GradKeyWrapper::wrap_t::pycast(
                GradKeyWrapper::get(output.cast<GradKeyValue>())));
    });

    m.def("set_grad", [](py::function backward_fn, std::vector<py::object> inputs,
                         std::vector<py::object> outputs) {
        GenericFunction generic_backward_fn =
                [backward_fn](Span<ValueRef> output_grads) -> ValueRefList {
            py::list output_grad_tws;
            for (auto&& output_grad : output_grads) {
                if (output_grad) {
                    output_grad_tws.append(
                            TensorWrapper::make(py_tensor_type, output_grad));
                } else {
                    output_grad_tws.append(py::none());
                }
            }
            py::tuple input_grad_tws = backward_fn(*output_grad_tws);
            ValueRefList input_grads(input_grad_tws.size());
            for (size_t i = 0; i < input_grad_tws.size(); ++i) {
                auto input_grad_tw = input_grad_tws[i];
                if (!input_grad_tw.is_none()) {
                    input_grads[i] =
                            py::cast<TensorWrapper>(input_grad_tw).m_tensor->data();
                } else {
                    input_grads[i] = {};
                }
            }
            return input_grads;
        };
        SmallVector<ValueRef> values(inputs.size() + outputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            values[i] = inputs[i].cast<TensorWrapper>().m_tensor->data();
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            values[i + inputs.size()] =
                    outputs[i].cast<TensorWrapper>().m_tensor->data();
        }
        auto wrapped_output_values =
                imperative::apply(SetGrad(generic_backward_fn, inputs.size()), values);
        std::vector<py::object> wrapped_outputs;
        mgb_assert(wrapped_output_values.size() == outputs.size());
        for (auto&& output_value : wrapped_output_values) {
            wrapped_outputs.push_back(
                    TensorWrapper::make(py_tensor_type, output_value));
        }
        return wrapped_outputs;
    });

    // ModuleTraceTransformation
    static py::function module_trace_hook;

    static auto get_module_trace = [] {
        static std::shared_ptr<ModuleTraceTransformation> module_trace_transformation;
        if (!module_trace_transformation) {
            mgb_assert(module_trace_hook);
            module_trace_transformation =
                    std::make_shared<ModuleTraceTransformation>(module_trace_hook);
            MGB_MARK_USED_VAR(transformations
                                      .register_at<Segment::ModuleTrace>(
                                              module_trace_transformation)
                                      .release());
        }
        return module_trace_transformation;
    };

    m.def("set_cpp_use_symbolic_shape", &set_cpp_use_symbolic_shape);

    m.def("set_module_tracing", [=] { get_module_trace()->enable(); });

    m.def("unset_module_tracing", [=] { get_module_trace()->disable(); });

    m.def("is_tracing_module", [=] { return get_module_trace()->enabled(); });
    m.def("set_python_backtrace_enabled", &set_python_backtrace_enabled);
    m.def("set_transformation_backtrace_enabled",
          &set_transformation_backtrace_enabled);
    m.def("_mge_backtrace", &get_py_backtrace);
    m.def("_get_frame_cache_id",
          []() { return (size_t)FrameInfoCache::get_instance(); });
    m.def("set_module_trace_hook", [](py::function function) {
        module_trace_hook = function;
        module_trace_hook.inc_ref();
    });

    auto atexit = py::module::import("atexit");
    atexit.attr("register")(py::cpp_function([]() { module_trace_hook = {}; }));
    m.def("begin_record_values", [] { Value::begin_record_values(); });

    m.def("end_record_values", [] {
        std::vector<std::pair<size_t, std::string>> reprs;
        auto values = Value::end_record_values();
        for (auto&& value : values) {
            reprs.push_back({value.id(), value.to_string()});
        }
        return reprs;
    });

    m.def("print_stats", [] { Stats::print(); });

    m.def("reset_stats", [] { Stats::reset(); });

    m.def("_get_convert_inputs",
          []() -> bool { return DTypePromoteCfg::convert_input_enabled; });
    m.def("_set_convert_inputs", [](bool flag) -> bool {
        bool ret = DTypePromoteCfg::convert_input_enabled;
        DTypePromoteCfg::convert_input_enabled = flag;
        return ret;
    });
    m.def("_get_amp_dtype_autocast",
          []() -> bool { return DTypePromoteCfg::amp_dtype_autocast_enabled; });
    m.def("_set_amp_dtype_autocast", [](bool flag) -> bool {
        bool ret = DTypePromoteCfg::amp_dtype_autocast_enabled;
        DTypePromoteCfg::amp_dtype_autocast_enabled = flag;
        return ret;
    });

    static auto get_amp_prec_dtype = [](bool is_high) -> std::string {
        DType& target = is_high ? DTypePromoteCfg::amp_high_prec_dtype
                                : DTypePromoteCfg::amp_low_prec_dtype;
        mgb_assert(target.category() == DTypeCategory::FLOAT);
        std::string ret = target.name();
        transform(ret.begin(), ret.end(), ret.begin(), ::tolower);
        return ret;
    };

    static auto set_amp_prec_dtype = [](bool is_high,
                                        std::string dtype_name) -> std::string {
        DType& target = is_high ? DTypePromoteCfg::amp_high_prec_dtype
                                : DTypePromoteCfg::amp_low_prec_dtype;
        std::string ret = target.name();

        if (dtype_name == "float32") {
            target = dtype::Float32();
        } else if (dtype_name == "float16") {
            target = dtype::Float16();
        } else if (dtype_name == "bfloat16") {
            target = dtype::BFloat16();
        } else {
            mgb_assert(
                    false, "casted type of amp should be float, but you give %s\n",
                    dtype_name.c_str());
        }

        transform(ret.begin(), ret.end(), ret.begin(), ::tolower);
        return ret;
    };

    m.def("_get_amp_high_prec_dtype",
          []() -> std::string { return get_amp_prec_dtype(true); });
    m.def("_set_amp_high_prec_dtype", [](std::string dtype_name) -> std::string {
        return set_amp_prec_dtype(true, dtype_name);
    });
    m.def("_get_amp_low_prec_dtype",
          []() -> std::string { return get_amp_prec_dtype(false); });
    m.def("_set_amp_low_prec_dtype", [](std::string dtype_name) -> std::string {
        return set_amp_prec_dtype(false, dtype_name);
    });

    m.def("_clear_algorithm_cache", [] { megdnn::AlgorithmCache::instance().clear(); });

    // FormatTransformation
    m.def("set_auto_format_convert",
          [format_trans](bool enabled) { format_trans->set_auto_convert(enabled); });
    m.def("get_auto_format_convert",
          [format_trans]() { return format_trans->get_auto_convert(); });

    py::register_exception<TraceError>(m, "TraceError");
}

#undef MGE_PY_INTERFACE

}  // namespace mgb::imperative::python
