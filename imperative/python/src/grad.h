/**
 * \file imperative/python/src/grad.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./tensor.h"

#include <megbrain/utils/small_vector.h>
#include <memory>

namespace mgb::imperative::python {

apply_result_t apply_grad(ApplyContext& ctx);

struct GradKey : std::enable_shared_from_this<GradKey>, NonCopyableObj {
    std::string name;
    bool active = true;
    GradInfo::head_t free_vars_head;
    std::vector<std::weak_ptr<GradFn>> tape;

    ~GradKey();

    void attach(Tensor* tensor, pybind11::object callback);
    void backward(std::vector<TensorWrapper*>, std::vector<TensorWrapper*>);
    void cleanup();
};

struct GradKeyWrapper {
    using wrap_t = pyext17::wrap<GradKeyWrapper>;
    static constexpr auto tp_name = pybind11::detail::_("GradKey");

    std::shared_ptr<GradKey> m_key;

    inline GradKeyWrapper() : m_key(std::make_shared<GradKey>()) {}

    void attach(PyObject*const* args, size_t nargs);
    void backward(std::vector<TensorWrapper*>, std::vector<TensorWrapper*>);
};

} // namespace mgb::imperative::python

namespace pybind11::detail {

template<> struct type_caster<mgb::imperative::python::GradKeyWrapper> : mgb::imperative::python::GradKeyWrapper::wrap_t::caster {};

} // namespace pybind11::detail
