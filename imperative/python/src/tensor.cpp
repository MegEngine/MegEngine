/**
 * \file imperative/python/src/tensor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./tensor.h"
#include "./grad.h"
#include "./common.h"
#include "./numpy_dtypes.h"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>

namespace py = pybind11;

namespace mgb::imperative::python {

std::unique_ptr<interpreter::Interpreter::Channel> interpreter_for_py;

apply_result_t apply(ApplyContext& ctx) {
    // emulating scalar should be put to specific op's apply, e.g.,
    // elementwise, reduce, typecvt. Currently it's still handled at python
    // side. It could be move to C++ side if it has an impact on performance
    if (ctx.flags & Tensor::Flags::SCALAR) {
        // TODO: emulate scalar
    }

    if (ctx.flags & Tensor::Flags::GRAD) {
        return apply_grad(ctx);
    }

    if (ctx.flags & Tensor::Flags::TRACE) {
        // TODO: trace
    } else {
        SmallVector<interpreter::Interpreter::Handle> handles(ctx.nargs);
        for (size_t i = 0; i < ctx.nargs; ++i) {
            handles[i] = ctx.args[i]->m_handle.get();
        }

        auto output_handles = interpreter_for_py->apply_op(ctx.op, handles);

        apply_result_t outputs;
        outputs.reserve(output_handles.size());
        for (auto h : output_handles) {
            outputs.emplace_back(std::make_shared<Tensor>(h));
        }
        return outputs;
    }

    mgb_assert(0);
}

PyObject* py_apply(PyObject* self, PyObject*const* args, size_t nargs/* , PyObject* kwnames */) {
    try {

        // if (kwnames && PyTuple_GET_SIZE(kwnames)) {
        //     PyErr_SetString(PyExc_TypeError, "keyword argument not allowed");
        //     return nullptr;
        // }
        if (!nargs) {
            PyErr_SetString(PyExc_TypeError, "expect Op");
            return nullptr;
        }
        auto* op = args[0];
        if (!strcmp(op->ob_type->tp_base->tp_name,"PodOpVisitor") || !strcmp(op->ob_type->tp_base->tp_name,"IndexingOpBase")){
            op = PyObject_CallMethod(op,"to_c","");
        }

        PyTypeObject* pytype = args[1]->ob_type;
        ++args;
        --nargs;

        ApplyContext ctx;
        ctx.flags = 0;
        ctx.op = py::handle(op).cast<std::shared_ptr<OpDef>>();
        SmallVector<Tensor*, 64> tensors(nargs);
        ctx.args = &tensors[0];
        ctx.nargs = nargs;

        for (size_t i = 0; i < nargs; ++i) {
            TensorWrapper* tw = TensorWrapper::cast_safe(args[i]);
            if (!tw) {
                PyErr_SetString(PyExc_TypeError, "expect Tensor");
                return nullptr;
            }
            auto* t = tensors[i] = tw->m_tensor.get();
            ctx.flags |= t->m_flags;
        }

        // TODO: set TRACE flag

        auto outputs = apply(ctx);
        size_t nout = outputs.size();
        auto ret = py::tuple(nout);
        for (size_t i = 0; i < nout; ++i) {
            ret[i] = TensorWrapper::make(pytype, std::move(outputs[i]));
        }
        return ret.release().ptr();

    } catch (std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}


TensorWrapper::TensorWrapper(PyObject* args, PyObject* kwargs) {
    if (kwargs && PyDict_Size(kwargs)) {
        throw py::type_error("keyword argument not allowed");
    }
    auto nargs = PyTuple_Size(args);
    auto tup = py::reinterpret_borrow<py::tuple>(args);
    if (nargs == 0) {
        throw py::type_error("too few arguments");
    }
    if (auto* t = cast_safe(tup[0].ptr())) {
        if (nargs > 1) {
            throw py::type_error("expect 1 argument");
        }
        m_tensor = t->m_tensor;
    } else {
        if (nargs != 3) {
            throw py::type_error("expect 3 arguments");
        }
        py::detail::loader_life_support life_sup; // required to cast DType
        auto data = tup[0].cast<py::array>();
        DType dtype = tup[1].cast<DType>();
        CompNode cn = tup[2].cast<CompNode>();

        interpreter::Interpreter::Handle handle;
        constexpr auto size_threshhold = TensorShape::MAX_NDIM;
        if (data.size() > size_threshhold) {
            handle = interpreter_for_py->put(npy::np2tensor(data.ptr(), npy::Meth::borrow(cn), dtype));
        } else {
            HostTensorND ret(cn);
            handle = interpreter_for_py->put(npy::np2tensor(data.ptr(), npy::Meth::copy_into(&ret), dtype));
        }

        m_tensor = std::make_shared<Tensor>(handle);
        if (data.ndim() == 0) {
            m_tensor->m_flags |= Tensor::Flags::SCALAR;
        }
    }
}


PyObject* TensorWrapper::shape() {
    if (m_tensor->m_flags & Tensor::Flags::SCALAR) {
        return PyTuple_New(0);
    }
    auto&& shape = m_tensor->shape();
    if (!shape.ndim) {
        Py_RETURN_NONE;
    }
    py::tuple ret(shape.ndim);
    for (size_t i = 0; i < shape.ndim; ++i) {
        ret[i] = shape[i];
    }
    return ret.release().ptr();
}


PyObject* TensorWrapper::dtype() {
    return py::cast(m_tensor->dtype()).release().ptr();
}


PyObject* TensorWrapper::device() {
    return py::cast(m_tensor->comp_node()).release().ptr();
}


PyObject* TensorWrapper::numpy() {
    auto&& hv = interpreter_for_py->get_value(m_tensor->m_handle.get());
    auto arr = py::reinterpret_steal<py::array>(npy::ndarray_from_tensor(hv, npy::ShareType::TRY_SHARE));
    if (!arr) return nullptr;
    if (m_tensor->m_flags & Tensor::Flags::SCALAR) {
        mgb_assert(PyArray_Check(arr.ptr()));
        return PyArray_Squeeze(reinterpret_cast<PyArrayObject*>(arr.ptr()));
    }
    return arr.release().ptr();
}

void TensorWrapper::reset(PyObject* tensor) {
    TensorWrapper* t = TensorWrapper::cast_safe(tensor);
    if (!t) {
        throw py::type_error("expect Tensor");
    }
    m_tensor = t->m_tensor;
}

PyObject* TensorWrapper::isscalar() {
    if(m_tensor->m_flags & Tensor::Flags::SCALAR) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

void TensorWrapper::setscalar() {
    m_tensor->m_flags |= Tensor::Flags::SCALAR;
}


struct TensorWeakRef {
    std::weak_ptr<Tensor> wptr;

    TensorWeakRef(const TensorWrapper& tw) : wptr(tw.m_tensor) {}

    py::object operator()() {
        if (auto p = wptr.lock()) {
            return TensorWrapper::make(p);
        }
        return py::none();
    }
};


void init_tensor(py::module m) {
    interpreter_for_py = interpreter::Interpreter::inst().create_channel();

    auto* tensor_type = TensorWrapper::wrap_t::type()
        .def<&TensorWrapper::numpy>("numpy")
        .def_getset<&TensorWrapper::shape>("shape")
        .def_getset<&TensorWrapper::dtype>("dtype")
        .def_getset<&TensorWrapper::device>("device")
        .def<&TensorWrapper::reset>("_reset")
        .def<&TensorWrapper::isscalar>("isscalar")
        .def<&TensorWrapper::setscalar>("setscalar")
        .finalize();
    if (!tensor_type) throw py::error_already_set();
    py::setattr(m, "Tensor", tensor_type);

    py::class_<TensorWeakRef>(m, "TensorWeakRef")
        .def(py::init<const TensorWrapper&>())
        .def("__call__", &TensorWeakRef::operator());

    static PyMethodDef apply_def{"apply", (PyCFunction)py_apply, METH_FASTCALL, nullptr};
    auto* apply_func = PyCFunction_NewEx(&apply_def, nullptr, nullptr);
    if (!apply_func) throw py::error_already_set();
    py::setattr(m, "apply", apply_func);

    py::handle grad_key_type = GradKeyWrapper::wrap_t::type()
        .def<&GradKeyWrapper::attach>("attach")
        .finalize();
    if (!grad_key_type) throw py::error_already_set();
    py::setattr(m, "GradKey", grad_key_type);
    py::setattr(m, "backward", py::cpp_function(&GradKeyWrapper::backward));
}

} // namespace mgb::imperative::python
