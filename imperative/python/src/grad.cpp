/**
 * \file imperative/python/src/grad.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./grad.h"
#include "megbrain/imperative/proxy_graph_detail.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/utils/mempool.h"

namespace py = pybind11;

namespace mgb::imperative::python {

namespace {

struct GradSlotWeakPtr {
    std::weak_ptr<GradFn> grad_fn;
    size_t idx;
};

} // namespace

struct GradProducerRecord : intrusive_list::Node<GradProducerRecord> {
    using Base = intrusive_list::Node<GradProducerRecord>;

    GradProducerRecord() = default;
    GradProducerRecord(GradProducerRecord::head_t& head) : Base(intrusive_list::after_t{}, head) {}
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
    GradSlotProducerPtr(GradInfo& info) : GradSlotPtr(info), producer_record(info->producer_head) {}
};

struct GradFn : std::enable_shared_from_this<GradFn> {
    static MemPool<GradFn> pool;

    std::weak_ptr<GradKey> key;
    SmallVector<GradSlot> slots;
    SmallVector<GradSlotProducerPtr> dsts;
    SmallVector<std::shared_ptr<Tensor>> closure;
    std::shared_ptr<BackwardGraphResult> backward_graph;
    bool in_ref_keeper = false;

    static void deleter(GradFn* ptr) {
        pool.free(ptr);
    }

    std::shared_ptr<GradFn> make() {
        return std::shared_ptr<GradFn>(pool.alloc(), &deleter);
    }

    void clear() {
        key.reset();
        slots.clear();
        dsts.clear();
        closure.clear();
        backward_graph.reset();
    }
};

GradSlot* GradSlotPtr::operator->() {
    return &grad_fn->slots[idx];
}

namespace {

struct BackwardGraphCache : std::unordered_map<size_t, std::shared_ptr<BackwardGraphResult>>, CompNodeDepedentObject {
    std::shared_ptr<void> on_comp_node_finalize() override {
        clear();
        return {};
    }
} backward_graph_cache;

std::shared_ptr<BackwardGraphResult> make_backward_graph(
        ApplyContext& ctx, const apply_result_t& outputs) {
    // hash
    static_assert(alignof(size_t) % alignof(bool) == 0);
    size_t buf_size = (1 + ctx.nargs * 2) * sizeof(size_t) + ctx.nargs * sizeof(bool);
    alignas(alignof(size_t)) std::byte buf[buf_size];
    size_t* size_t_ptr = reinterpret_cast<size_t*>(buf);
    bool* bool_ptr = reinterpret_cast<bool*>(size_t_ptr + (1 + ctx.nargs * 2));
    bool* bool_ptr0 = bool_ptr;
    *(size_t_ptr++) = ctx.op->hash();
    for (size_t i = 0; i < ctx.nargs; ++i) {
        *(size_t_ptr++) = mgb::hash(ctx.args[i]->dtype().handle());
        *(size_t_ptr++) = mgb::hash(ctx.args[i]->comp_node());
        *(bool_ptr++) = bool(ctx.args[i]->m_grad_info.grad_fn);
    }
    mgb_assert(bool_ptr0 == reinterpret_cast<bool*>(size_t_ptr) &&
               bool_ptr == reinterpret_cast<bool*>(buf + buf_size));
    size_t key = XXHash{}.update(buf, buf_size).digest();

    auto&& iter = backward_graph_cache.find(key);
    if (iter != backward_graph_cache.end()) {
        return iter->second;
    }

    // slow path
    SmallVector<LogicalTensorDesc> inputs(ctx.nargs);
    SmallVector<bool> input_requires_grad(ctx.nargs, false);
    SmallVector<bool> output_has_grad(outputs.size(), true);
    for (size_t i = 0; i < ctx.nargs; ++i) {
        inputs[i].comp_node = ctx.args[i]->comp_node();
        inputs[i].layout.dtype = ctx.args[i]->dtype();
        input_requires_grad[i] = bool(ctx.args[i]->m_grad_info.grad_fn);
    }
    auto result = std::make_shared<BackwardGraphResult>(
        proxy_graph_detail::make_backward_graph(
            *ctx.op, inputs, input_requires_grad, output_has_grad));
    if (!result->backward) {
        result.reset();
    }
    backward_graph_cache.emplace(key, result);
    return result;
}

} // namespace

apply_result_t apply_grad(ApplyContext& ctx) {
    std::shared_ptr<GradKey> grad_key;
    for (size_t i = 0; i < ctx.nargs; ++i) {
        auto* tensor = ctx.args[i];
        if (tensor->m_grad_info.grad_fn) {
            auto&& input_grad_key = tensor->m_grad_info.grad_fn->key.lock();
            // tensor is attached to a live GradKey
            if (input_grad_key && input_grad_key->active) {
                if (grad_key) {
                    if (grad_key != input_grad_key) {
                        PyErr_SetString(PyExc_NotImplementedError, "second order grad");
                        throw pyext17::py_err_set();
                    }
                } else {
                    grad_key = std::move(input_grad_key);
                }
            } else {
                // cleanup stale grad info
                // under what condition?
                tensor->m_grad_info = {};
                tensor->m_flags &= ~Tensor::Flags::GRAD;
            }
        } else {
            tensor->m_flags &= ~Tensor::Flags::GRAD;
        }
    }

    ctx.flags &= ~Tensor::Flags::GRAD;

    // perform forward apply_op or trace
    auto outputs = apply(ctx);

    if (!grad_key) {
        return outputs;
    }

    auto backward_graph = make_backward_graph(ctx, outputs);
    if (!backward_graph) {
        return outputs;
    }

    auto grad_fn = std::make_shared<GradFn>();
    grad_fn->key = grad_key;
    grad_fn->slots.resize(outputs.size());
    grad_fn->backward_graph = std::move(backward_graph);

    grad_fn->dsts.reserve(ctx.nargs);
    for (size_t i = 0; i < ctx.nargs; ++i) {
        if (grad_fn->backward_graph->input_has_grad[i]) {
            auto& input_grad_info = ctx.args[i]->m_grad_info;
            grad_fn->dsts.emplace_back(input_grad_info);
            grad_fn->dsts.back().producer_record.insert_after(input_grad_info->producer_head);
        } else {
            grad_fn->dsts.emplace_back();
        }
    }

    auto& save_for_backward = grad_fn->backward_graph->save_for_backward;
    grad_fn->closure.reserve(std::count_if(save_for_backward.begin(), save_for_backward.end(), [](bool p){return p;}));

    // given op, taking gradient of output_tensor_list wrt input_tensor_list:
    //
    // save_for_backward[0:nargs-1]: whether input tensor requires gradient,
    //    i.e., whether it is in input_tensor_list
    //
    // save_for_backward[nargs:nargs+outputs.size()-1]: whether output tensor is
    //    needed to calculate gradients
    //
    // save_for_backward[-outputs.size():]: whether output tensor is in
    //    output_tensor_list
    //
    // Example: perform c = a * b, where a is input data, b is parameter to be
    // optimized, save_for_backward = [1, 1, 0, 1]
    mgb_assert(ctx.nargs + 2 * outputs.size() == save_for_backward.size());

    // record input tensors needed to take grad
    for (size_t i = 0; i < ctx.nargs; ++i) {
        if (save_for_backward[i]) {
            grad_fn->closure.push_back(ctx.args[i]->shared_from_this());
        }
    }
    // record output tensors needed to take grad
    for (size_t i = 0; i < outputs.size(); ++i) {
        bool requires_grad = save_for_backward[ctx.nargs + outputs.size() + i];
        if (save_for_backward[ctx.nargs + i]) {
            grad_fn->closure.push_back(outputs[i]);
            if (requires_grad) {
                // avoid reference cycle [Tensor <-> GradFn]
                outputs[i] = outputs[i]->copy();
            }
        }
        if (requires_grad) {
            auto& grad_info = outputs[i]->m_grad_info;
            grad_info.grad_fn = grad_fn;
            grad_info.idx = i;
            grad_info.insert_after(grad_key->free_vars_head);
            outputs[i]->m_flags |= Tensor::Flags::GRAD;
        }
    }

    // record forward history
    grad_key->tape.emplace_back(grad_fn);

    return outputs;
}

void GradKeyWrapper::attach(PyObject*const* args, size_t nargs) {
    if (nargs != 2) {
        throw py::type_error("expect 2 arguments");
    }
    auto* tw = TensorWrapper::cast_safe(args[0]);
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

    if (tensor->m_grad_info.grad_fn) {
        if (tensor->m_grad_info.grad_fn->key.lock().get() != this) {
            PyErr_SetString(PyExc_NotImplementedError, "second order grad");
            throw pyext17::py_err_set();
        }
        if (tensor->m_grad_info->callback) {
            throw py::value_error("callback already set on this tensor");
        }
    } else {
        tensor->m_grad_info.idx = 0;
        auto& grad_fn = tensor->m_grad_info.grad_fn;
        grad_fn = std::make_shared<GradFn>();
        grad_fn->key = shared_from_this();
        grad_fn->slots.resize(1);
        tensor->m_grad_info.insert_after(free_vars_head);
        tensor->m_flags |= Tensor::Flags::GRAD;
    }
    tensor->m_grad_info.grad_fn->slots[0].callback = std::move(callback);
}

void accum_grad(std::shared_ptr<Tensor>& grad, std::shared_ptr<Tensor>&& delta) {
    if (!grad) {
        grad = std::forward<decltype(delta)>(delta);
        return;
    }
    static ApplyContext ctx;
    if (!ctx.op) {
        ctx.op = std::shared_ptr<OpDef>(new Elemwise(Elemwise::Mode::ADD));
        ctx.nargs = 2;
    }
    Tensor* args[2] = {grad.get(), delta.get()};
    ctx.args = args;
    ctx.flags = grad->m_flags | delta->m_flags;
    if (is_tracing) {
        ctx.flags |= Tensor::Flags::TRACE;
    }
    grad = apply(ctx)[0];
}

void GradKey::backward(std::vector<TensorWrapper*> tensors, std::vector<TensorWrapper*> grads) {
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
        CleanupGuard(GradKey* this_) : owner(this_) {}
        ~CleanupGuard() {owner->cleanup();}
    } _cleanup_guard(this);

    if (tape.empty() || grads.empty()) return;
    PyTypeObject* pytype = Py_TYPE(grads[0]->self().ptr());

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto& grad_info = tensors[i]->m_tensor->m_grad_info;
        if (grad_info.grad_fn && grad_info.grad_fn->key.lock().get() == this) {
            grad_info->grad = grads[i]->m_tensor;
        }
    }

    std::vector<std::shared_ptr<GradFn>> ref_keeper;
    ref_keeper.reserve(tape.size());
    // back-propagation in reverse order
    for (std::ptrdiff_t k = tape.size() - 1; k >= 0; --k) {
        auto&& grad_fn = tape[k].lock();
        if (!grad_fn) continue;
        if (grad_fn->backward_graph) {
            for (size_t i = 0; i < grad_fn->slots.size(); ++i) {
                // grad_fn->dsts correspond to input tensors during forward
                // calculation, grad_fn->slots correspond to output tensors.
                // condition true means the output tensor has gradient for
                // back-propagation
                if (grad_fn->backward_graph->save_for_backward[grad_fn->dsts.size() + grad_fn->slots.size() + i]) {
                    grad_fn->closure.push_back(std::move(grad_fn->slots[i].grad));
                }
            }
            ApplyContext ctx;
            ctx.op = grad_fn->backward_graph->backward;
            ctx.flags = 0;
            ctx.nargs = grad_fn->closure.size();
            Tensor* args[ctx.nargs];
            for (size_t i = 0; i < ctx.nargs; ++i) {
                args[i] = grad_fn->closure[i].get();
                mgb_assert(args[i]);
                ctx.flags |= args[i]->m_flags;
            }
            ctx.args = args;

            if (is_tracing)
                ctx.flags |= Tensor::Flags::TRACE;

            auto grads = apply(ctx);

            size_t j = 0;
            for (size_t i = 0; i < grad_fn->dsts.size(); ++i) {
                if (grad_fn->backward_graph->input_has_grad[i]) {
                    auto& dst = grad_fn->dsts[i];
                    // grads[j] is consumed in accum_grad
                    accum_grad(dst->grad, std::move(grads[j]));
                    ++j;
                }
            }
            mgb_assert(j == grads.size());
        }
        for (auto&& dst : grad_fn->dsts) {
            if (!dst.grad_fn) continue;
            if (!dst.grad_fn->in_ref_keeper) {
                dst.grad_fn->in_ref_keeper = true;
                ref_keeper.push_back(dst.grad_fn);
            }
            // grad_fn->clear will unlink current dst.producer_record
            // such that if dst.producer_record.next is false, dst accumulates
            // all the gradients
            if (!dst.producer_record.next && dst->callback && dst->grad) {
                dst->callback(TensorWrapper::make(pytype, dst->grad));
            }
        }
        grad_fn->clear();
    } // finish tape loop
}

void GradKey::cleanup() {
    active = false;
    tape.clear();
    for (intrusive_list::Iterator it(free_vars_head); it;) {
        it->grad_fn.reset();
        (it++)->unlink();
    }
}

void GradKeyWrapper::backward(std::vector<TensorWrapper*> tensors, std::vector<TensorWrapper*> grads) {
    m_key->backward(std::move(tensors), std::move(grads));
}

GradKey::~GradKey() {
    cleanup();
}

} // namespace mgb::imperative::python
