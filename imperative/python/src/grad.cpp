/**
 * \file imperative/python/src/grad.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

#include "./grad.h"
#include "megbrain/imperative/backward_graph_opt.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/proxy_graph_detail.h"
#include "megbrain/utils/mempool.h"

#include "range/v3/all.hpp"

namespace py = pybind11;
namespace views = ranges::views;

namespace mgb::imperative::python {

using scoped_disable = ApplyContext::scoped_disable;
using Flags = Tensor::Flags;

namespace {

struct GradSlotWeakPtr {
    std::weak_ptr<GradFn> grad_fn;
    size_t idx;
};

std::shared_ptr<OptimizedBackwardGraphResult> make_backward_graph(
        ApplyContext& ctx, const apply_result_t& outputs) {
    // hash
    using OptimizedBackwardGraphCache = OpMethResultCache<
            std::shared_ptr<OptimizedBackwardGraphResult>, SmallVector<bool>>;
    thread_local OptimizedBackwardGraphCache cache;
    decltype(cache)::key_t cache_key{ctx.op};
    SmallVector<LogicalTensorDesc>& input_descs = cache_key.inputs;
    SmallVector<bool>& input_requires_grad = std::get<0>(cache_key.extras);
    input_descs.resize(ctx.nargs);
    input_requires_grad.resize(ctx.nargs);
    for (size_t i = 0; i < ctx.nargs; ++i) {
        input_descs[i].layout.dtype = ctx.args[i]->dtype();
        input_descs[i].comp_node = ctx.args[i]->comp_node();
        input_requires_grad[i] = python::input_requires_grad(ctx, i);
    }

    auto iter = cache.find(cache_key);
    if (iter != cache.end()) {
        return iter->second;
    }

    // slow path
    SmallVector<bool> output_has_grad(outputs.size(), true);
    std::shared_ptr<OptimizedBackwardGraphResult> ret;
    auto bg = OpDef::make_backward_graph(
            *ctx.op, input_descs, input_requires_grad, output_has_grad);
    if (!bg.graph.empty()) {
        ret = std::make_shared<OptimizedBackwardGraphResult>(bg);
    }
    cache.emplace(cache_key, ret);
    return ret;
}

struct BackwardGraphWithClosure {
    std::shared_ptr<OptimizedBackwardGraphResult> backward_graph;
    SmallVector<std::shared_ptr<Tensor>> closure;
    size_t output_mask_offset;
    size_t grad_mask_offset;

    BackwardGraphWithClosure(
            std::shared_ptr<OptimizedBackwardGraphResult> backward_graph_,
            ApplyContext& ctx, const apply_result_t& outputs)
            : backward_graph(backward_graph_),
              output_mask_offset(ctx.nargs),
              grad_mask_offset(ctx.nargs + outputs.size()) {
        // save_for_backward[0:nargs]:
        //     whether input is kept for backward
        //
        // save_for_backward[nargs:nargs+outputs.size()]:
        //     whether output is kept for backward
        //
        // save_for_backward[-outputs.size():]:
        //     whether gradient of output can propagate to any input
        //
        // Example:
        //     perform c = a * b, with a.requires_grad == True and
        //     b.requires_grad == False, save_for_backward = [0, 1, 0, 1]
        auto& save_for_backward = backward_graph->save_for_backward;
        mgb_assert(save_for_backward.size() == ctx.nargs + 2 * outputs.size());
        size_t count = std::count_if(
                save_for_backward.begin(), save_for_backward.end(), ranges::identity{});
        if (!backward_graph->precomp.empty()) {
            auto&& irng = ranges::span(ctx.args, ctx.nargs);
            auto&& orng = views::transform(outputs, [](auto&& i) { return i.get(); });
            auto precomp = apply(backward_graph->precomp, views::concat(irng, orng));
            closure.reserve(precomp.size() + count);
            std::copy(precomp.begin(), precomp.end(), std::back_inserter(closure));
        } else {
            closure.reserve(count);
        }
        for (size_t i = 0; i < ctx.nargs; ++i) {
            if (save_for_backward[i]) {
                closure.push_back(ctx.args[i]->shared_from_this());
            }
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (save_for_backward[ctx.nargs + i]) {
                closure.push_back(outputs[i]);
            }
        }
    }

    template <typename T, typename R>
    void operator()(BackwardContext&, T&& grads, R&& receiver) {
        Tensor* args[closure.size() + grads.size()];
        size_t nargs = 0;
        for (auto&& t : closure) {
            args[nargs++] = t.get();
        }
        bool null_grad = false;
        for (size_t i = 0; i < grads.size(); ++i) {
            if (backward_graph->save_for_backward[grad_mask_offset + i]) {
                if (grads[i]) {
                    if (null_grad) {
                        PyErr_SetString(PyExc_NotImplementedError, "report to devs");
                        throw py::error_already_set();
                    }
                    args[nargs++] = grads[i];
                } else {
                    null_grad = true;
                }
            }
        }
        if (null_grad)
            return;

        auto igrads = apply(backward_graph->backward, args, nargs);
        auto&& it = igrads.begin();
        for (auto [i, p] : views::enumerate(backward_graph->input_has_grad)) {
            if (p) {
                receiver(i, std::move(*it));
                ++it;
            }
        }
    }

    bool input_has_grad(size_t i) { return backward_graph->input_has_grad[i]; }

    bool output_requires_grad(size_t i) {
        return backward_graph->save_for_backward[grad_mask_offset + i];
    }

    bool output_captured(size_t i) {
        return backward_graph->save_for_backward[output_mask_offset + i];
    }
};

struct PythonBackward {
    py::object pyfunc;
    size_t input_size;

    PythonBackward(py::object f, size_t nin) : pyfunc(f), input_size(nin) {}

    template <typename T, typename R>
    void operator()(BackwardContext& ctx, T&& grads, R&& receiver) {
        auto args = py::tuple(grads.size());
        for (size_t i = 0; i < grads.size(); ++i) {
            auto&& g = grads[i];
            args[i] = g ? ctx.wrap_tensor(g) : py::none();
        }
        auto input_grads = py::reinterpret_steal<py::object>(
                PyObject_Call(pyfunc.ptr(), args.ptr(), nullptr));
        if (!input_grads)
            throw py::error_already_set();
        if (input_grads.is_none())
            return;
        if (auto* tw = TensorWrapper::try_cast(input_grads.ptr())) {
            if (input_size != 1) {
                throw py::value_error(
                        "custom grad rule returned wrong number of grads");
            }
            if (!ctx.pytype) {
                ctx.pytype = Py_TYPE(input_grads.ptr());
            }
            receiver(0, tw->m_tensor);
            return;
        }
        if (py::len(input_grads) != input_size) {
            throw py::value_error("custom grad rule returned wrong number of grads");
        }
        for (auto [i, g] : views::enumerate(input_grads)) {
            if (g.is_none())
                continue;
            auto* tw = TensorWrapper::try_cast(g.ptr());
            if (!tw) {
                throw py::type_error("custom grad rule returned non-tensor");
            }
            if (!ctx.pytype) {
                ctx.pytype = Py_TYPE(g.ptr());
            }
            receiver(i, tw->m_tensor);
        }
    }

    static constexpr bool input_has_grad(size_t) { return true; }
    static constexpr bool output_requires_grad(size_t) { return true; }
    static constexpr bool output_captured(size_t) { return true; }
};

}  // namespace

struct GradProducerRecord : intrusive_list::Node<GradProducerRecord> {
    using Base = intrusive_list::Node<GradProducerRecord>;

    GradProducerRecord() = default;
    GradProducerRecord(GradProducerRecord::head_t& head)
            : Base(intrusive_list::after_t{}, head) {}
    // GradProducerRecord(GradProducerRecord&&) = default;
    // GradProducerRecord& operator=(GradProducerRecord&) = default;
    // GradProducerRecord& operator=(GradProducerRecord&&) = default;
};

struct GradSlot {
    std::shared_ptr<Tensor> grad;
    py::object callback;
    GradProducerRecord::head_t producer_head;
};

struct GradSlotProducerPtr : GradSlotPtr {
    GradProducerRecord producer_record;

    GradSlotProducerPtr() = default;
    GradSlotProducerPtr(GradInfo& info)
            : GradSlotPtr(info), producer_record(info->producer_head) {}
};

struct GradFn : std::enable_shared_from_this<GradFn> {
    static MemPool<GradFn> pool;

    std::weak_ptr<GradKey> key;
    // slots for receiving and accumulating grads
    // same length as outputs (of forward op)
    SmallVector<GradSlot> slots;
    // where to send and accumulate grads
    // same length as inputs (of forward op)
    SmallVector<GradSlotProducerPtr> dsts;
    // encapsules actual function to compute gradient
    std::variant<
            std::monostate, BackwardGraphWithClosure, PythonBackward, CustomBackward>
            backward;
    // a flag used during backward
    bool in_ref_keeper = false;

    static void deleter(GradFn* ptr) { pool.free(ptr); }

    static std::shared_ptr<GradFn> make() {
        return std::shared_ptr<GradFn>(pool.alloc(), &deleter);
    }

    void clear() {
        key.reset();
        slots.clear();
        dsts.clear();
        backward.emplace<std::monostate>();
    }
};

GradSlotPtr::operator bool() const {
    return bool(grad_fn);
}

GradSlot* GradSlotPtr::operator->() {
    return &grad_fn->slots[idx];
}

namespace {

class GradFnHelper {
    std::shared_ptr<GradFn> grad_fn;

    GradFn* get() {
        if (!grad_fn) {
            grad_fn = std::make_shared<GradFn>();
        }
        return grad_fn.get();
    }

    friend apply_result_t imperative::python::apply_grad(ApplyContext&);

public:
    template <typename T, typename... Args>
    auto& emplace(Args&&... args) {
        return get()->backward.emplace<T>(std::forward<Args>(args)...);
    }

    void reset() { grad_fn = nullptr; }
};

apply_result_t backward_graph_grad_rule(ApplyContext& ctx, GradFnHelper& ret_grad_fn) {
    // copy inputs first, or trace will make InputNodes for each usage
    ApplyContext ctx_dup = ctx;
    SmallVector<std::shared_ptr<Tensor>> inputs_copy;
    SmallVector<Tensor*> inputs_copy_weak;
    for (size_t i = 0; i < ctx.nargs; ++i) {
        Tensor* input = ctx.args[i];
        inputs_copy.push_back(python::apply(FastpathCopy::make(), input)[0]);
        inputs_copy_weak.push_back(inputs_copy.back().get());
        inputs_copy.back()->m_grad_info_dict = ctx.args[i]->m_grad_info_dict;
        if (input->m_flags & Flags::GRAD) {
            inputs_copy.back()->m_flags |= Flags::GRAD;
        }
    }
    ctx_dup.args = inputs_copy_weak.data();

    auto outputs = apply(ctx_dup);

    auto backward_graph = make_backward_graph(ctx_dup, outputs);
    if (!backward_graph) {
        return outputs;
    }
    ret_grad_fn.emplace<BackwardGraphWithClosure>(
            std::move(backward_graph), ctx_dup, outputs);

    return outputs;
}

apply_result_t python_grad_rule(ApplyContext& ctx, GradFnHelper& ret_grad_fn) {
    auto* op = ctx.op->try_cast_final<GenericPyOp>();
    py::tuple pyin(ctx.nargs);
    for (size_t i = 0; i < ctx.nargs; ++i) {
        pyin[i] = TensorWrapper::make(ctx.pytype, ctx.args[i]->shared_from_this());
    }
    auto grad_rule = py::getattr(op->obj, "_grad_rule");
    auto pyret = py::reinterpret_steal<py::object>(
            PyObject_Call(grad_rule.ptr(), pyin.ptr(), nullptr));
    if (!pyret)
        throw py::error_already_set();
    auto [outputs, backward] = py::cast<std::tuple<py::object, py::function>>(pyret);
    ret_grad_fn.emplace<PythonBackward>(std::move(backward), ctx.nargs);
    if (auto* tw = TensorWrapper::try_cast(outputs.ptr())) {
        return {tw->m_tensor};
    }
    apply_result_t ret;
    ret.reserve(py::len(outputs));
    for (auto&& i : outputs) {
        auto* tw = TensorWrapper::try_cast(i.ptr());
        mgb_assert(tw);
        ret.push_back(tw->m_tensor);
    }
    return ret;
}

}  // namespace

apply_result_t apply_grad(ApplyContext& ctx) {
    std::unordered_set<std::shared_ptr<GradKey>> grad_keys;
    for (size_t i = 0; i < ctx.nargs; ++i) {
        auto* tensor = ctx.args[i];
        if (!tensor->m_grad_info_dict.empty()) {
            size_t grad_cnt = 0;
            for (auto&& grad_info : tensor->m_grad_info_dict) {
                auto input_grad_key = grad_info.grad_fn->key.lock();
                if (input_grad_key && input_grad_key->active &&
                    !input_grad_key->is_blocked()) {
                    grad_keys.insert(input_grad_key);
                    grad_cnt++;
                }
            }
            if (!grad_cnt) {
                tensor->m_flags &= ~Flags::GRAD;
            }
        } else {
            tensor->m_flags &= ~Flags::GRAD;
        }
    }

    ctx.flags &= ~Flags::GRAD;

    if (grad_keys.empty()) {
        return apply(ctx);
    } else if (grad_keys.size() > 1 && !GradKey::allow_higher_order_directive) {
        PyErr_SetString(
                PyExc_NotImplementedError,
                "second order directive not enabled, please call "
                "'megengine.experimental.enable_higher_order_directive'");
        throw pyext17::py_err_set();
    }

    GradFnHelper grad_fn_holder;
    auto outputs = [&]() {
        auto _ = scoped_disable(Flags::GRAD);
        if (ctx.op->same_type<GenericPyOp>()) {
            return python_grad_rule(ctx, grad_fn_holder);
        }
        auto&& registry = grad_rule_registry();
        auto&& it = registry.find(ctx.op->dyn_typeinfo());
        if (it != registry.end()) {
            auto&& maker = grad_fn_holder.emplace<CustomBackward>().maker(ctx);
            if (auto ret = it->second(ctx, maker)) {
                maker.finalize();
                return *ret;
            }
            grad_fn_holder.reset();
        }
        return backward_graph_grad_rule(ctx, grad_fn_holder);
    }();

    if (!grad_fn_holder.grad_fn) {
        return outputs;
    }

    for (auto&& grad_key : grad_keys) {
        auto grad_fn = std::make_shared<GradFn>();
        grad_fn->backward = grad_fn_holder.grad_fn->backward;
        grad_fn->key = grad_key;
        grad_fn->slots.resize(outputs.size());
        grad_fn->dsts.reserve(ctx.nargs);

        std::visit(
                [&](auto& backward) {
                    using T = std::decay_t<decltype(backward)>;
                    if constexpr (std::is_same_v<T, std::monostate>) {
                        mgb_assert(0);
                    } else {
                        for (size_t i = 0; i < ctx.nargs; ++i) {
                            if (backward.input_has_grad(i) &&
                                input_requires_grad(ctx, i) &&
                                ctx.args[i]->m_grad_info_dict.count(grad_key.get())) {
                                auto& input_grad_info =
                                        ctx.args[i]->m_grad_info_dict.at(
                                                grad_key.get());
                                grad_fn->dsts.emplace_back(input_grad_info);
                                // register as grad producer
                                grad_fn->dsts.back().producer_record.insert_after(
                                        input_grad_info->producer_head);
                            } else {
                                grad_fn->dsts.emplace_back();
                            }
                        }
                        for (size_t i = 0; i < outputs.size(); ++i) {
                            if (backward.output_requires_grad(i)) {
                                if (backward.output_captured(i)) {
                                    // avoid reference cycle [Tensor <-> GradFn]
                                    static std::shared_ptr<OpDef> op =
                                            std::make_shared<FastpathCopy>();
                                    outputs[i] = python::apply(op, outputs[i])[0];
                                }
                                // populate grad info of output tensor
                                auto& grad_info =
                                        outputs[i]->m_grad_info_dict[grad_key.get()];
                                grad_info.grad_fn = grad_fn;
                                grad_info.idx = i;
                                grad_info.insert_after(grad_key->free_vars_head);
                                outputs[i]->m_flags |= Flags::GRAD;
                            }
                        }
                    }
                },
                grad_fn->backward);

        // record forward history
        grad_key->tape.emplace_back(grad_fn);
    }

    return outputs;
}

PyObject* GradKeyWrapper::get_priority() {
    return py::cast(m_key->priority).release().ptr();
}

void GradKeyWrapper::set_priority(pybind11::handle priority) {
    m_key->priority = py::cast<int>(priority);
}

void GradKeyWrapper::attach(PyObject* const* args, size_t nargs) {
    if (nargs != 2) {
        throw py::type_error("expect 2 arguments");
    }
    auto* tw = TensorWrapper::try_cast(args[0]);
    if (!tw) {
        throw py::type_error("argument 1 must be Tensor");
    }
    auto* tensor = tw->m_tensor.get();
    py::object callback;
    if (args[1] != Py_None) {
        callback = py::reinterpret_borrow<py::object>(args[1]);
    }
    m_key->attach(tensor, std::move(callback));
}

//!  GradKey is weakly refered by tensor->m_grad_info.grad_fn->key after attach
void GradKey::attach(Tensor* tensor, pybind11::object callback) {
    if (!active) {
        throw py::value_error("grad key finalized");
    }

    if (tensor->m_grad_info_dict.count(this)) {
        if (tensor->m_grad_info_dict.at(this)->callback) {
            throw py::value_error("callback already set on this tensor");
        }
    } else {
        auto& grad_info = tensor->m_grad_info_dict[this];
        grad_info.idx = 0;
        auto& grad_fn = grad_info.grad_fn;
        grad_fn = std::make_shared<GradFn>();
        grad_fn->key = shared_from_this();
        grad_fn->slots.resize(1);
        grad_info.insert_after(free_vars_head);
        tensor->m_flags |= Flags::GRAD;
    }
    tensor->m_grad_info_dict.at(this).grad_fn->slots[0].callback = std::move(callback);
}

template <typename T>
void accum_grad(std::shared_ptr<Tensor>& grad, T&& delta) {
    if (!grad) {
        grad = std::forward<T>(delta);
        return;
    }
    static std::shared_ptr<OpDef> op =
            std::shared_ptr<OpDef>(new Elemwise(Elemwise::Mode::ADD));
    grad = apply(op, grad, std::forward<T>(delta))[0];
}

void GradKey::backward(
        std::vector<TensorWrapper*> tensors, std::vector<TensorWrapper*> grads) {
    if (!active) {
        throw py::value_error("finalized");
    }
    if (tensors.size() != grads.size()) {
        throw py::value_error("tensor and grad size mismatch");
    }

    // this GradKey is marked inactive here
    active = false;
    struct CleanupGuard {
        GradKey* owner;
        size_t priority_backup;
        CleanupGuard(GradKey* this_) : owner(this_) {
            priority_backup = sm_min_priority;
            sm_min_priority = owner->priority + 1;
        }
        ~CleanupGuard() {
            owner->cleanup();
            sm_min_priority = priority_backup;
        }
    } _cleanup_guard(this);

    if (tape.empty())
        return;

    BackwardContext bctx;
    if (!grads.empty()) {
        bctx.pytype = Py_TYPE(grads[0]->self().ptr());
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i]->m_tensor->m_grad_info_dict.count(this) == 0) {
            continue;
        }
        auto& grad_info = tensors[i]->m_tensor->m_grad_info_dict.at(this);
        grad_info->grad = grads[i]->m_tensor;
    }

    std::vector<std::shared_ptr<GradFn>> ref_keeper;
    ref_keeper.reserve(tape.size());

    // back-propagation in reverse order
    for (std::ptrdiff_t k = tape.size() - 1; k >= 0; --k) {
        auto&& grad_fn = tape[k].lock();
        if (!grad_fn)
            continue;

        auto grad_receiver = [&](size_t i, auto&& g) {
            auto& dst = grad_fn->dsts[i];
            if (dst) {
                accum_grad(dst->grad, std::forward<decltype(g)>(g));
            }
        };
        std::visit(
                [&](auto&& backward) {
                    using T = std::decay_t<decltype(backward)>;
                    if constexpr (std::is_same_v<T, std::monostate>) {
                        mgb_assert(0);
                    } else {
                        auto&& grads = views::transform(
                                grad_fn->slots,
                                [](auto&& slot) { return slot.grad.get(); });
                        backward(
                                bctx, std::forward<decltype(grads)>(grads),
                                grad_receiver);
                    }
                },
                grad_fn->backward);

        for (auto&& dst : grad_fn->dsts) {
            if (!dst.grad_fn)
                continue;
            if (!dst.grad_fn->in_ref_keeper) {
                // after grad_fn is cleared, refcnt of subsequent grad_fn
                // could drop to 0
                dst.grad_fn->in_ref_keeper = true;
                ref_keeper.push_back(dst.grad_fn);
            }
            if (!dst.producer_record.next && dst->callback && dst->grad) {
                // I'm the last grad producer, invoke callback
                dst->callback(bctx.wrap_tensor(dst->grad));
            }
        }
        grad_fn->clear();
    }  // finish tape loop
}

void GradKey::cleanup() {
    active = false;
    tape.clear();
    for (intrusive_list::Iterator it(free_vars_head); it;) {
        it->grad_fn.reset();
        (it++)->unlink();
    }
}

void GradKeyWrapper::backward(
        std::vector<TensorWrapper*> tensors, std::vector<TensorWrapper*> grads) {
    m_key->backward(std::move(tensors), std::move(grads));
}

PyObject* GradKeyWrapper::get_name() {
    return py::cast(m_key->name).release().ptr();
}

void GradKeyWrapper::set_name(py::handle name) {
    m_key->name = py::cast<std::string>(name);
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
    if (tw->m_tensor->m_grad_info_dict.count(m_key.get())) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

int GradKey::sm_min_priority = std::numeric_limits<int>::min();

GradKey::~GradKey() {
    cleanup();
}

std::unordered_map<Typeinfo*, GradRuleFn>& grad_rule_registry() {
    static std::unordered_map<Typeinfo*, GradRuleFn> registry;
    return registry;
}

void GradInfoCollection::_shrink() {
    auto pred = [](GradInfo& info) {
        return !(info.grad_fn) || info.grad_fn->key.expired();
    };
    auto iter = std::remove_if(m_storage.begin(), m_storage.end(), pred);
    m_storage.erase(iter, m_storage.end());
}

bool GradInfoCollection::contains(GradKey* key) {
    _shrink();
    for (auto&& grad_info : m_storage) {
        if (grad_info.grad_fn->key.lock().get() == key) {
            return true;
        }
    }
    return false;
}

GradInfo& GradInfoCollection::operator[](GradKey* key) {
    _shrink();
    for (auto&& grad_info : m_storage) {
        if (grad_info.grad_fn->key.lock().get() == key) {
            return grad_info;
        }
    }
    m_storage.emplace_back();
    return m_storage.back();
}

GradInfo& GradInfoCollection::at(GradKey* key) {
    _shrink();
    for (auto&& grad_info : m_storage) {
        if (grad_info.grad_fn->key.lock().get() == key) {
            return grad_info;
        }
    }
    mgb_assert(false);
}

}  // namespace mgb::imperative::python
