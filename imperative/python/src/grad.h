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

struct BackwardContext {
    PyTypeObject* pytype = nullptr;

    auto wrap_tensor(std::shared_ptr<Tensor> t) {
        if (pytype) {
            return TensorWrapper::make(pytype, std::move(t));
        }
        return TensorWrapper::make(std::move(t));
    }

    auto wrap_tensor(Tensor* t) {
        return wrap_tensor(t->shared_from_this());
    }
};

struct CustomBackward {
    using BackwardFn = std::function<apply_result_t(BackwardContext&, Tensor*const*, size_t)>;
    BackwardFn m_backward;
    SmallVector<bool, 8> m_input_has_grad;
    struct OutputAttr {bool requires_grad = true, captured = true;};
    SmallVector<OutputAttr> m_output_attrs;

public:
    template<typename T, typename R>
    void operator()(BackwardContext& ctx, T&& grads, R&& receiver) {
        size_t nargs = grads.size();
        Tensor* args[nargs];
        for (size_t i = 0; i < nargs; ++i) {
            args[i] = grads[i];
        }
        auto ret = m_backward(ctx, args, nargs);
        for (size_t i = 0; i < ret.size(); ++i) {
            if (auto&& t = ret[i]) {
                receiver(i, std::move(t));
            }
        }
    }

    bool input_has_grad(size_t i) {return m_input_has_grad[i];}
    bool output_requires_grad(size_t i) {return m_output_attrs[i].requires_grad;}
    bool output_captured(size_t i) {return m_output_attrs[i].captured;}

    class Maker {
        bool output_size_set = false, input_has_grad_initialized = false;
        CustomBackward& target;
        ApplyContext& ctx;

        void init_input_has_grad() {
            if (!input_has_grad_initialized) {
                input_has_grad_initialized = true;
                target.m_input_has_grad.resize(ctx.nargs, true);
            }
        }

    public:
        Maker(CustomBackward& target_, ApplyContext& ctx_) : target(target_), ctx(ctx_) {}

        template<typename F>
        Maker& backward(F&& f) {
            mgb_assert(!target.m_backward);
            target.m_backward = std::forward<F>(f);
            return *this;
        }
        // mandatory
        Maker& output_size(size_t sz) {
            mgb_assert(!output_size_set);
            output_size_set = true;
            target.m_output_attrs.resize(sz);
            return *this;
        }
        // optional, defaults to all true
        Maker& input_has_grad(size_t i, bool v) {
            init_input_has_grad();
            target.m_input_has_grad.at(i) = v;
            return *this;
        }
        // optional, defaults to all true
        Maker& output_requires_grad(size_t i, bool v) {
            target.m_output_attrs.at(i).requires_grad = v;
            return *this;
        }
        // optional, defaults to all true
        Maker& output_captured(size_t i, bool v) {
            target.m_output_attrs.at(i).captured = v;
            return *this;
        }

        void finalize() {
            mgb_assert(output_size_set);
            init_input_has_grad();
        }
    };

    Maker maker(ApplyContext& ctx) {return {*this, ctx};}
};

using GradRuleFn = std::function<apply_result_t(ApplyContext&, CustomBackward::Maker&)>;

std::unordered_map<Typeinfo*, GradRuleFn>& grad_rule_registry();

inline bool input_requires_grad(const ApplyContext& ctx, size_t i) {
    return bool(ctx.args[i]->m_grad_info.grad_fn);
}

struct GradRuleFallback : std::exception {};

template<typename T>
bool register_grad_rule(Typeinfo* typeinfo, T&& rule) {
    return grad_rule_registry().emplace(typeinfo, std::forward<T>(rule)).second;
}

} // namespace mgb::imperative::python

namespace pybind11::detail {

template<> struct type_caster<mgb::imperative::python::GradKeyWrapper> : mgb::imperative::python::GradKeyWrapper::wrap_t::caster {};

} // namespace pybind11::detail
