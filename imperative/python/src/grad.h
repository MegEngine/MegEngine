#pragma once

#include "./tensor.h"

#include "megbrain/imperative/ops/utility.h"
#include "megbrain/imperative/transformations/grad.h"
#include "megbrain/utils/small_vector.h"

#include <memory>
#include <optional>

namespace mgb::imperative::python {

struct GradKeyWrapper : NonCopyableObj {
    using wrap_t = pyext17::wrap<GradKeyWrapper>;
    static constexpr auto tp_name = pybind11::detail::_("GradKey");

    std::string m_name;
    std::shared_ptr<GradKey> m_key;
    std::shared_ptr<GradTransformation> m_transformation;
    std::unique_ptr<CleanupGuard<>> m_transformation_guard;

    GradKeyWrapper();

    PyObject* get_name();
    void set_name(pybind11::handle name);
    void attach(PyObject* const* args, size_t nargs);
    static void backward(GradKeyWrapper* self, pybind11::list, pybind11::list);
    static pybind11::function get_backward_closure(
            GradKeyWrapper* self, pybind11::list);
    PyObject* is_attached_to(PyObject* const* args, size_t nargs);
    void enter();
    void exit();
    void suppress();
    void resume();
    static GradKeyWrapper* get(std::shared_ptr<GradKey> key);
    ~GradKeyWrapper();
};

}  // namespace mgb::imperative::python

namespace pybind11::detail {

template <>
struct type_caster<mgb::imperative::python::GradKeyWrapper>
        : mgb::imperative::python::GradKeyWrapper::wrap_t::caster {};

}  // namespace pybind11::detail
