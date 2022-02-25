#include "./grad.h"

#include "megbrain/imperative/backward_graph_opt.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/proxy_graph_detail.h"
#include "megbrain/imperative/resource_manager.h"
#include "megbrain/utils/mempool.h"

#include "range/v3/all.hpp"

#include "./helper.h"
#include "./transformation.h"

namespace py = pybind11;
namespace views = ranges::views;

namespace mgb::imperative::python {

namespace {
std::unordered_map<std::shared_ptr<GradKey>, GradKeyWrapper*> grad_key_map;
}

GradKeyWrapper::GradKeyWrapper() {}

void GradKeyWrapper::attach(PyObject* const* args, size_t nargs) {
    if (nargs != 2) {
        throw py::type_error("expect 2 arguments");
    }
    auto* tw = TensorWrapper::try_cast(args[0]);
    if (!tw) {
        throw py::type_error("argument 1 must be Tensor");
    }
    py::object callback;
    if (args[1] != Py_None) {
        callback = py::reinterpret_borrow<py::object>(args[1]);
    }
    GenericFunction generic_callback = [=](Span<ValueRef> inputs) -> ValueRefList {
        mgb_assert(inputs.size() == 1);
        if (callback) {
            callback(TensorWrapper::make(py_tensor_type, inputs[0]));
        }
        return {};
    };
    auto attached_value = imperative::apply(
            AttachGrad(m_key), tw->m_tensor->data(),
            FunctionValue::make(generic_callback))[0];
    tw->m_tensor->reset(attached_value);
}

void GradKeyWrapper::backward(GradKeyWrapper* self, py::list tensors, py::list grads) {
    std::vector<ValueRef> args;
    mgb_assert(tensors.size() == grads.size());
    for (auto&& tensor : tensors) {
        args.push_back(TensorWrapper::try_cast(tensor.ptr())->m_tensor->data());
    }
    for (auto&& grad : grads) {
        args.push_back(TensorWrapper::try_cast(grad.ptr())->m_tensor->data());
    }
    imperative::apply(GradBackward(self->m_key), {args.data(), args.size()});
}

pybind11::function GradKeyWrapper::get_backward_closure(
        GradKeyWrapper* self, py::list tensors) {
    std::vector<ValueRef> args;
    for (auto&& tensor : tensors) {
        args.push_back(TensorWrapper::try_cast(tensor.ptr())->m_tensor->data());
    }
    auto closure_value = imperative::apply(GetBackwardColsure(self->m_key), args)[0];
    auto closure = closure_value.as_ref<FunctionValue>();
    auto py_function = [closure](std::vector<TensorWrapper*> tensors) {
        std::vector<ValueRef> args;
        for (auto* tw : tensors) {
            args.push_back(tw->m_tensor->data());
        }
        (*closure)(args);
    };
    return pybind11::cpp_function(py_function);
}

PyObject* GradKeyWrapper::get_name() {
    return py::cast(m_name).release().ptr();
}

void GradKeyWrapper::set_name(py::handle name) {
    m_name = py::cast<std::string>(name);
    if (m_key) {
        m_key->name(m_name);
    }
}

PyObject* GradKeyWrapper::is_attached_to(PyObject* const* args, size_t nargs) {
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "expect 1 argument");
        return nullptr;
    }
    auto* tw = TensorWrapper::try_cast(args[0]);
    if (!tw) {
        PyErr_SetString(PyExc_TypeError, "expect Tensor");
        return nullptr;
    }
    if (imperative::apply(IsAttachedTo(m_key), tw->m_tensor->data())[0]
                .cast<BoolValue>()) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

void GradKeyWrapper::enter() {
    m_transformation = std::make_shared<GradTransformation>();
    m_key = m_transformation->key();
    m_key->name(m_name);
    grad_key_map[m_key] = this;
    m_transformation_guard =
            TransformationManager::get_instance()
                    .register_at<TransformationManager::Grad>(m_transformation);
}

void GradKeyWrapper::exit() {
    m_transformation_guard.reset();
    grad_key_map.erase(m_key);
    m_key = {};
    m_transformation.reset();
}

void GradKeyWrapper::suppress() {
    m_transformation->suppress();
}

void GradKeyWrapper::resume() {
    m_transformation->resume();
}

GradKeyWrapper* GradKeyWrapper::get(std::shared_ptr<GradKey> key) {
    return grad_key_map.at(key);
}

GradKeyWrapper::~GradKeyWrapper() {}

}  // namespace mgb::imperative::python
