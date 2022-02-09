/**
 * \file imperative/python/src/tensor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/common.h"
#include "megbrain/dtype.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/backward_graph.h"
#include "megbrain/imperative/ops/utility.h"
#include "megbrain/imperative/profiler.h"
#include "megbrain/imperative/transformations/eval.h"
#include "megbrain/imperative/transformations/lazy.h"
#include "megbrain/imperative/transformations/scalar.h"
#include "megbrain/imperative/transformations/symbol.h"
#include "megbrain/imperative/transformations/trace.h"
#include "megbrain/imperative/utils/map.h"
#include "megbrain/imperative/utils/stats.h"
#include "megbrain/opr/io.h"
#include "megbrain/plugin/profiler.h"

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
#include <range/v3/all.hpp>
#include <string>

#include <unordered_map>

#include "../../src/impl/mgb_cg_impl.h"

namespace py = pybind11;
namespace views = ranges::views;

namespace mgb::imperative::python {

bool is_scalar(PyObject* tensor) {
    if (py::isinstance<PySymbolVar>(py::handle(tensor))) {
        auto var = py::handle(tensor).cast<PySymbolVar*>();
        return var->is_scalar;
    }
    auto* tw = TensorWrapper::try_cast(tensor);
    if (tw) {
        return tw->m_tensor->is_scalar();
    }
    return PyArray_CheckAnyScalar(tensor);
}

bool is_bool_list(PyObject* arg) {
    if (!PyList_Check(arg)) {
        return false;
    }
    size_t sz = PyList_Size(arg);
    if (!sz) {
        return false;
    }
    for (size_t i = 0; i < sz; ++i) {
        PyObject* handle = PyList_GetItem(arg, i);
        if (!PyBool_Check(handle)) {
            return false;
        }
    }
    return true;
}

bool is_bool_dtype(PyObject* args) {
    if (!PyObject_HasAttrString(args, "dtype"))
        return false;
    PyObject* dobj = PyObject_GetAttrString(args, "dtype");
    PyArray_Descr* dtype;
    PyArray_DescrConverter(dobj, &dtype);
    bool ret = (dtype->kind == 'b');
    Py_XDECREF(dtype);
    Py_XDECREF(dobj);
    return ret;
}

py::object _Const(
        py::handle value, py::handle dtype, py::handle device, py::handle ref) {
    py::object val = py::reinterpret_borrow<py::object>(value);
    if (PyArray_Check(value.ptr())) {
        py::tuple strides =
                py::reinterpret_borrow<py::tuple>(getattr(value, "strides"));
        bool need_squeeze = false;
        for (size_t i = 0; i < strides.size(); ++i) {
            if (strides[i].cast<ptrdiff_t>() == 0) {
                need_squeeze = true;
            }
        }
        if (need_squeeze) {
            val = py::reinterpret_borrow<py::array>(value);
            val = val.attr("squeeze")();
            val = val.attr("reshape")(val.attr("shape"));
        }
    }
    if (py::isinstance<PySymbolVar>(ref)) {
        auto ref_var = ref.cast<PySymbolVar*>();
        auto* graph = ref_var->m_node->owner_graph();
        auto cn = device.cast<CompNode>();
        OperatorNodeConfig config(cn);
        auto hv = npy::np2tensor(
                val.ptr(), npy::Meth::borrow(cn), dtype.cast<mgb::DType>());
        auto typeobj = ref.get_type();
        return typeobj(opr::ImmutableTensor::make(*graph, hv, config).node());
    }
    py::tuple tup = py::make_tuple(val, dtype, device, true, false, py::none());
    return TensorWrapper::make(py_tensor_type, tup.ptr(), nullptr);
}

py::tuple _make_shape_tuple(py::handle shape) {
    py::list orig;
    py::list ret(0);
    auto solve_one = [&](py::handle val) {
        if (TensorWrapper::try_cast(val.ptr()) || py::isinstance<PySymbolVar>(val)) {
            py::object np = getattr(val, "numpy")();
            PyArrayObject* arr = (PyArrayObject*)np.ptr();
            PyObject* maybe_list = PyArray_ToList(arr);
            if (PyList_Check(maybe_list)) {
                py::list may = py::reinterpret_steal<py::list>(maybe_list);
                for (size_t i = 0; i < may.size(); ++i) {
                    ret.append(may[i]);
                }
            } else {
                mgb_assert(PyLong_Check(maybe_list));
                ret.append(PyLong_AsLong(maybe_list));
                Py_XDECREF(maybe_list);
            }
        } else if (PyArray_Check(val.ptr())) {
            ret.append(PyArray_PyIntAsInt(val.ptr()));
        } else {
            ret.append(PyLong_AsLong(val.ptr()));
        }
    };
    if (PyArray_Check(shape.ptr()) && !PyArray_CheckAnyScalar(shape.ptr())) {
        orig = py::reinterpret_steal<py::list>(
                PyArray_ToList((PyArrayObject*)shape.ptr()));
        for (size_t i = 0; i < orig.size(); ++i) {
            solve_one(orig[i]);
        }
    } else if (PyList_Check(shape.ptr())) {
        orig = py::reinterpret_borrow<py::list>(shape);
        for (size_t i = 0; i < orig.size(); ++i) {
            solve_one(orig[i]);
        }
    } else if (PyTuple_Check(shape.ptr())) {
        py::tuple tup = py::reinterpret_borrow<py::tuple>(shape);
        for (size_t i = 0; i < tup.size(); ++i) {
            solve_one(tup[i]);
        }
    } else {
        solve_one(shape);
    }
    return py::reinterpret_steal<py::tuple>(PyList_AsTuple(ret.ptr()));
}

py::object _get_index(py::object tensor, py::object src) {
    if (!TensorWrapper::try_cast(tensor.ptr()) &&
        !py::isinstance<PySymbolVar>(tensor)) {
        auto get_const = [&](mgb::DType dtype) -> py::object {
            return _Const(tensor, py::cast(dtype), src.attr("device"), src);
        };
        if (is_bool_list(tensor.ptr()) || is_bool_dtype(tensor.ptr())) {
            tensor = get_const(dtype::Bool());
        } else {
            tensor = get_const(dtype::Int32());
        }
        if (!is_bool_dtype(tensor.ptr())) {
            return tensor;
        }
    } else {
        if (!is_bool_dtype(tensor.ptr())) {
            return tensor;
        }
    }
    static std::shared_ptr<OpDef> op = CondTake::make();
    std::vector<PyObject*> p;
    p.resize(3);
    py::object Op = py::cast(op);
    p[0] = Op.ptr();
    p[1] = tensor.ptr();
    p[2] = tensor.ptr();
    py::tuple ret =
            py::reinterpret_steal<py::object>(py_apply(NULL, p.data(), p.size()));
    return ret[1];
}

py::tuple _try_cond_take(py::handle tensor, py::handle index) {
    if (!hasattr(index, "dtype") || !hasattr(index, "shape")) {
        return py::tuple();
    }
    if (!is_bool_dtype(index.ptr()) ||
        _make_shape_tuple(getattr(index, "shape"))
                .not_equal(_make_shape_tuple(getattr(tensor, "shape")))) {
        return py::tuple();
    }
    py::object iobj;
    if (PyArray_Check(index.ptr())) {
        iobj =
                _Const(index, py::cast((mgb::DType)dtype::Bool()),
                       getattr(tensor, "device"), tensor);
    } else {
        iobj = py::reinterpret_borrow<py::object>(index);
    }
    static std::shared_ptr<OpDef> op = CondTake::make();
    std::vector<PyObject*> p;
    p.resize(3);
    py::object Op = py::cast(op);
    p[0] = Op.ptr();
    p[1] = tensor.ptr();
    p[2] = iobj.ptr();
    py::tuple ret =
            py::reinterpret_steal<py::object>(py_apply(NULL, p.data(), p.size()));
    return ret;
}

py::tuple _remove_ellipsis(py::object tensor, py::tuple tuple_val) {
    size_t tuple_size = tuple_val.size();
    size_t ndim_sum = 0, cur_sum = 0;
    int pos = -1;
    bool has_unknown_ndim_bool_index = false;
    for (size_t i = 0; i < tuple_size; ++i) {
        py::object handle = tuple_val[i];
        if (handle.ptr() == Py_Ellipsis) {
            pos = static_cast<int>(i);
            for (size_t j = 0; j < i; ++j) {
                py::object t = tuple_val[j];
                if (t.ptr() == Py_Ellipsis) {
                    throw py::index_error("only one ellipsis is allowed.");
                }
            }
        } else {
            size_t ndim_incr = 1;
            if (hasattr(handle, "dtype") && is_bool_dtype(handle.ptr()) &&
                hasattr(handle, "ndim")) {
                py::object ndim = getattr(handle, "ndim");
                if (PyLong_Check(ndim.ptr())) {
                    ndim_incr = PyLong_AsLong(ndim.ptr());
                } else {
                    has_unknown_ndim_bool_index = true;
                }
            }
            cur_sum += ndim_incr;
        }
    }
    if (pos == -1) {
        return tuple_val;
    } else {
        if (has_unknown_ndim_bool_index) {
            throw py::index_error(
                    "does not support bool index with unknown shape when using "
                    "Ellipsis.");
        }
        try {
            ndim_sum = getattr(tensor, "ndim").cast<size_t>();
        } catch (py::error_already_set& err) {
            throw py::index_error(
                    "does not support Ellipsis when tensor's ndim is unknown.");
        }
        py::tuple ret(ndim_sum - cur_sum + tuple_size - 1);
        size_t idx = 0;
        for (size_t i = 0; i < tuple_size; ++i) {
            if (i == pos) {
                for (size_t j = cur_sum; j < ndim_sum; ++j) {
                    ret[idx++] = PySlice_New(NULL, NULL, NULL);
                }
            } else {
                ret[idx++] = tuple_val[i];
            }
        }
        return ret;
    }
}

py::tuple _expand_bool_dim(py::object tensor, py::tuple tuple_val) {
    py::tuple cur_shape = _make_shape_tuple(py::handle(getattr(tensor, "shape")));
    py::list new_tuple_val(0);

    size_t offset = 0;
    size_t tdim = 0;
    for (size_t i = 0; i < tuple_val.size(); ++i) {
        py::handle k = tuple_val[i];
        if (is_bool_dtype(k.ptr())) {
            size_t ndim = getattr(k, "ndim").cast<size_t>();
            if (ndim > 1) {
                py::tuple ishape = _make_shape_tuple(py::handle(getattr(k, "shape")));
                for (size_t j = 0; j < ndim; ++j) {
                    if (cur_shape[tdim + j - offset].cast<size_t>() !=
                        ishape[j].cast<size_t>()) {
                        std::string msg =
                                "boolean index did not match tensor along dimension " +
                                std::to_string(tdim + j) + "; dimension is " +
                                std::to_string(
                                        cur_shape[tdim + j - offset].cast<size_t>()) +
                                " but corresponding boolean dimension is " +
                                std::to_string(ishape[j].cast<size_t>());
                        throw py::index_error(msg.c_str());
                    }
                }
                py::object new_k = getattr(k, "reshape")(-1);
                py::object kshape = getattr(new_k, "shape");
                py::list new_shape(0);
                PyObject* sym = PyObject_CallObject(cpp_use_symbolic_shape, nullptr);
                bool is_sym = (sym == Py_True);
                Py_XDECREF(sym);
                if (is_sym) {
                    py::object tshape = getattr(tensor, "shape");
                    for (size_t j = 0; j < i; ++j) {
                        new_shape.append(tshape[py::int_(j)]);
                    }
                    new_shape.append(kshape[py::int_(0)]);
                    for (size_t j = tdim + ndim - offset; j < cur_shape.size(); ++j) {
                        new_shape.append(cur_shape[j]);
                    }
                    py::tuple args = py::make_tuple(new_shape);
                    PyObject* shape_tensor =
                            PyObject_CallObject(cpp_astensor1d, args.ptr());
                    py::object reshape_func = getattr(tensor, "reshape");
                    Py_INCREF(shape_tensor);
                    PyObject* Args = PyTuple_New(1);
                    PyTuple_SetItem(Args, 0, shape_tensor);
                    PyObject* new_tensor =
                            PyObject_CallObject(reshape_func.ptr(), Args);
                    Py_XDECREF(Args);
                    tensor = py::reinterpret_steal<py::object>(new_tensor);
                    cur_shape = _make_shape_tuple(py::handle(shape_tensor));
                    Py_XDECREF(shape_tensor);
                } else {
                    for (size_t j = 0; j < i; ++j) {
                        new_shape.append(cur_shape[j]);
                    }
                    new_shape.append(py::reinterpret_borrow<py::tuple>(kshape)[0]);
                    for (size_t j = tdim + ndim - offset; j < cur_shape.size(); ++j) {
                        new_shape.append(cur_shape[j]);
                    }
                    cur_shape = new_shape;
                    tensor = getattr(tensor, "reshape")(cur_shape);
                }
                offset++;
                tdim += ndim;
            }
            new_tuple_val.append(k);
        } else {
            new_tuple_val.append(k);
            tdim++;
        }
    }
    return py::make_tuple(tensor, py::reinterpret_borrow<py::tuple>(new_tuple_val));
}

py::tuple _unpack_indexes(py::handle inp_hdl, py::handle idx_hdl) {
    py::object inp = py::reinterpret_borrow<py::object>(inp_hdl);
    py::tuple tuple_val;
    if (py::isinstance<py::tuple>(idx_hdl)) {
        tuple_val = py::reinterpret_borrow<py::tuple>(idx_hdl);
    } else {
        tuple_val = py::make_tuple(idx_hdl);
    }

    bool use_subtensor = true;
    bool need_remove_ellipsis = false;
    bool need_expand_bool_dim = false;
    size_t idx_ndim = 0;
    for (size_t i = 0; i < tuple_val.size(); ++i) {
        py::object k = tuple_val[i];
        if (k.ptr() == Py_None) {
            throw py::index_error("newaxis is not allowed here");
        } else if (k.ptr() == Py_Ellipsis) {
            need_remove_ellipsis = true;
        } else {
            if (is_bool_dtype(k.ptr()) && hasattr(k, "ndim")) {
                size_t ndim = getattr(k, "ndim").cast<size_t>();
                idx_ndim += ndim;
                if (ndim > 1) {
                    need_expand_bool_dim = true;
                }
            } else {
                idx_ndim++;
            }
        }
    }
    try {
        size_t inp_ndim = getattr(inp, "ndim").cast<size_t>();
        if (idx_ndim > inp_ndim) {
            std::string msg = "too many indices for tensor: tensor is " +
                              std::to_string(inp_ndim) + "-dimensional, but " +
                              std::to_string(idx_ndim) + " were indexed";
            throw py::index_error(msg.c_str());
        }
    } catch (py::error_already_set& err) {
        ;  // ignore
    }
    if (need_remove_ellipsis) {
        tuple_val = _remove_ellipsis(inp, tuple_val);
    }

    if (need_expand_bool_dim) {
        py::object shape = getattr(inp, "shape");
        if (shape.ptr() != Py_None) {
            py::tuple ret = _expand_bool_dim(inp, tuple_val);
            inp = ret[0];
            tuple_val = ret[1];
        }
    }

    py::list items;
    py::list tensors;
    int cur_axis = -1;

    for (size_t i = 0; i < tuple_val.size(); ++i) {
        py::object handle = tuple_val[i];
        cur_axis++;
        if (!is_scalar(handle.ptr()) && !PySlice_Check(handle.ptr())) {
            use_subtensor = false;
        }
        py::list item;
        item.append(cur_axis);
        auto push = [&](PyObject* v) {
            if (v == Py_None) {
                item.append(false);
            } else {
                item.append(true);
                tensors.append(_get_index(py::reinterpret_borrow<py::object>(v), inp));
            }
        };

        if (PySlice_Check(handle.ptr())) {
            PySliceObject* s = (PySliceObject*)handle.ptr();
            if (s->start == Py_None && s->stop == Py_None && s->step == Py_None) {
                continue;
            }
            push(s->start);
            push(s->stop);
            push(s->step);
            item.append(false);
        } else {
            for (size_t j = 0; j < 3; j++)
                item.append(false);
            push(handle.ptr());
        }
        items.append(item);
    }

    return py::make_tuple(inp, tensors, items, use_subtensor, need_expand_bool_dim);
}

py::object _getitem_cpp(py::handle inp_hdl, py::handle idx_hdl) {
    py::tuple try_res = _try_cond_take(inp_hdl, idx_hdl);
    if (try_res.size() == 2) {
        return try_res[0];
    }
    py::tuple up = _unpack_indexes(inp_hdl, idx_hdl);
    py::object tensor = py::reinterpret_borrow<py::object>(up[0]);
    py::list tensors = py::reinterpret_borrow<py::list>(up[1]);
    py::list py_items = py::reinterpret_borrow<py::list>(up[2]);
    std::vector<std::tuple<int8_t, bool, bool, bool, bool>> cpp_items;
    for (size_t i = 0; i < py_items.size(); ++i) {
        py::list item = py::reinterpret_borrow<py::list>(py_items[i]);
        cpp_items.push_back(
                {item[0].cast<int8_t>(), item[1].cast<bool>(), item[2].cast<bool>(),
                 item[3].cast<bool>(), item[4].cast<bool>()});
    }
    static std::shared_ptr<OpDef> op;
    if (up[3].cast<bool>()) {
        op = Subtensor::make(cpp_items);
    } else {
        op = IndexingMultiAxisVec::make(cpp_items);
    }
    std::vector<PyObject*> p;
    p.resize(tensors.size() + 2);
    py::object Op = py::cast(op);
    p[0] = Op.ptr();
    p[1] = tensor.ptr();
    for (size_t i = 0; i < tensors.size(); ++i) {
        p[i + 2] = tensors[i].ptr();
    }
    py::tuple ret =
            py::reinterpret_steal<py::object>(py_apply(NULL, p.data(), p.size()));
    return ret[0];
}

py::object _setitem_cpp(py::handle inp_hdl, py::handle idx_hdl, py::handle val_hdl) {
    py::object org_shape = getattr(inp_hdl, "shape");
    py::object val = py::reinterpret_borrow<py::object>(val_hdl);
    if (!TensorWrapper::try_cast(val.ptr()) && !py::isinstance<PySymbolVar>(val)) {
        val =
                _Const(val_hdl, getattr(inp_hdl, "dtype"), getattr(inp_hdl, "device"),
                       inp_hdl);
    }

    py::tuple up = _unpack_indexes(inp_hdl, idx_hdl);
    py::object tensor = py::reinterpret_borrow<py::object>(up[0]);
    py::list tensors = py::reinterpret_borrow<py::list>(up[1]);
    py::list py_items = py::reinterpret_borrow<py::list>(up[2]);
    std::vector<std::tuple<int8_t, bool, bool, bool, bool>> cpp_items;
    for (size_t i = 0; i < py_items.size(); ++i) {
        py::list item = py::reinterpret_borrow<py::list>(py_items[i]);
        cpp_items.push_back(
                {item[0].cast<int8_t>(), item[1].cast<bool>(), item[2].cast<bool>(),
                 item[3].cast<bool>(), item[4].cast<bool>()});
    }
    static std::shared_ptr<OpDef> op, set_op;
    if (up[3].cast<bool>()) {
        op = Subtensor::make(cpp_items);
    } else {
        op = IndexingMultiAxisVec::make(cpp_items);
    }
    std::vector<PyObject*> p;
    p.resize(tensors.size() + 2);
    py::object Op = py::cast(op);
    p[0] = Op.ptr();
    p[1] = tensor.ptr();
    for (size_t i = 0; i < tensors.size(); ++i) {
        p[i + 2] = tensors[i].ptr();
    }
    py::tuple ret =
            py::reinterpret_steal<py::object>(py_apply(NULL, p.data(), p.size()));
    py::object tmp_result = ret[0];

    try {
        py::object value_tuple_shape = val.attr("_tuple_shape");
        py::object tmp_result_tuple_shape = tmp_result.attr("_tuple_shape");
        py::tuple value_shape = py::reinterpret_borrow<py::tuple>(value_tuple_shape);
        py::tuple tmp_result_shape =
                py::reinterpret_borrow<py::tuple>(tmp_result_tuple_shape);
        for (size_t i = 0; i < value_shape.size() && i < tmp_result_shape.size(); ++i) {
            size_t vs = value_shape[value_shape.size() - i - 1].cast<size_t>();
            size_t ts =
                    tmp_result_shape[tmp_result_shape.size() - i - 1].cast<size_t>();
            if (vs != 1 && vs != ts) {
                std::string lhs = "", rhs = "";
                for (size_t j = 0; j < tmp_result_shape.size(); ++j) {
                    lhs += std::to_string(tmp_result_shape[j].cast<size_t>());
                    if (j)
                        lhs += ",";
                }
                for (size_t j = 0; j < value_shape.size(); ++j) {
                    rhs += std::to_string(value_shape[j].cast<size_t>());
                    if (j)
                        rhs += ",";
                }
                throw py::value_error(
                        "cannot copy tensor with shape (" + rhs +
                        ") to subtensor with shape (" + lhs + ")");
            }
        }
    } catch (py::error_already_set& err) {
        ;
    }

    py::object broadcast_func = getattr(val, "_broadcast");
    PyObject* Args = PyTuple_New(1);
    PyTuple_SetItem(Args, 0, getattr(tmp_result, "shape").release().ptr());
    PyObject* new_val = PyObject_CallObject(broadcast_func.ptr(), Args);
    Py_XDECREF(Args);
    val = py::reinterpret_steal<py::object>(new_val);

    if (up[3].cast<bool>()) {
        set_op = SetSubtensor::make(cpp_items);
    } else {
        set_op = IndexingSetMultiAxisVec::make(cpp_items);
    }

    std::vector<PyObject*> q;
    q.resize(tensors.size() + 3);
    py::object Set_Op = py::cast(set_op);
    q[0] = Set_Op.ptr();
    q[1] = tensor.ptr();
    q[2] = val.ptr();
    for (size_t i = 0; i < tensors.size(); ++i) {
        q[i + 3] = tensors[i].ptr();
    }
    py::tuple result =
            py::reinterpret_steal<py::object>(py_apply(NULL, q.data(), q.size()));
    py::object res = result[0];

    if (up[4].cast<bool>()) {
        py::object reshape_func = getattr(res, "reshape");
        PyObject* Args = PyTuple_New(1);
        PyTuple_SetItem(Args, 0, org_shape.release().ptr());
        PyObject* new_tensor = PyObject_CallObject(reshape_func.ptr(), Args);
        Py_XDECREF(Args);
        res = py::reinterpret_steal<py::object>(new_tensor);
    }

    return res;
}

PyObject* make_shape_tuple(PyObject* self, PyObject* const* args, size_t nargs) {
    try {
        return _make_shape_tuple(py::handle(args[0])).release().ptr();
    }
    PYEXT17_TRANSLATE_EXC_RET(nullptr)
}

PyObject* getitem_cpp(PyObject* self, PyObject* const* args, size_t nargs) {
    try {
        return _getitem_cpp(py::handle(args[0]), py::handle(args[1])).release().ptr();
    }
    PYEXT17_TRANSLATE_EXC_RET(nullptr)
}

PyObject* setitem_cpp(PyObject* self, PyObject* const* args, size_t nargs) {
    try {
        return _setitem_cpp(
                       py::handle(args[0]), py::handle(args[1]), py::handle(args[2]))
                .release()
                .ptr();
    }
    PYEXT17_TRANSLATE_EXC_RET(nullptr)
}

}  // namespace mgb::imperative::python
