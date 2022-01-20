/**
 * \file imperative/src/impl/transformations/trace.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/transformations/lazy.h"
#include "megbrain/imperative/opr_utility.h"
#include "megbrain/imperative/ops/autogen.h"

#include "megbrain/opr/utility.h"

#include "../async_releaser.h"
#include "../mgb_cg_impl.h"

namespace mgb {
namespace imperative {

ValueRefList LazyEvalTransformation::apply_transformation(
        const Operator& op, Span<ValueRef> inputs) {
    if (auto* op_val = op.as<ApplyOp>()) {
        static std::unordered_set<Typeinfo*> mm_io_ops = {
                CollectiveComm::typeinfo(),
                RemoteSend::typeinfo(),
                RemoteRecv::typeinfo(),
        };
        bool require_link = mm_io_ops.count(op_val->op().dyn_typeinfo());
        VarNodeArray input_nodes;
        for (auto&& input : inputs) {
            if (auto* input_node = input.as<LazyEvalValue>()) {
                input_nodes.push_back(input_node->node());
            } else {
                // ImmutableTensor has empty shape issues
                auto dev_val = input.dev_tensor()->as_nd();
                auto dev_val_provider = [dev_val]() mutable {
                    return std::move(dev_val);
                };
                auto* node = opr::InputCallback::make(
                                     *m_graph, dev_val_provider, *input.device(),
                                     *input.dtype(), input.shape()->as_tensor_shape(),
                                     {}, true)[0]
                                     .node();
                input_nodes.push_back(node);
            }
        }
        if (require_link && m_io_link.node()) {
            mgb_assert(!input_nodes.empty());
            input_nodes[0] =
                    opr::VirtualDep::make({SymbolVar(input_nodes[0]), m_io_link})
                            .node();
        }
        VarNodeArray output_nodes = OpDef::apply_on_var_node(op_val->op(), input_nodes);
        if (require_link) {
            mgb_assert(!output_nodes.empty());
            m_io_link = SymbolVar(output_nodes[0]);
        }
        ValueRefList outputs(output_nodes.size());
        for (size_t i = 0; i < output_nodes.size(); ++i) {
            outputs[i] = record_var(output_nodes[i]);
        }
        return outputs;
    } else if (auto* create_tensor = op.as<CreateTensor>()) {
        auto&& args = create_tensor->parse(inputs);
        auto get_dev_val = [&] {
            if (!args.device) {
                mgb_assert(args.host);
                args.device.emplace();
                args.device->copy_from(*args.host);
                // every h2d in imperative runtime should notify AsyncReleaser
                AsyncReleaser::inst()->add(*args.host);
            }
            return *args.device;
        };
        if (args.kind == CreateTensor::Const) {
            VarNode* node;
            if (args.host) {
                node = opr::ImmutableTensor::make(*m_graph, *args.host).node();
            } else {
                node = opr::SharedDeviceTensor::make(
                               *m_graph, std::make_shared<DeviceTensorND>(*args.device),
                               true, {})
                               .node();
            }
            if (m_no_exec) {
                // TODO: record args instead of value
                auto output = apply(op, inputs)[0];
                auto name = output.name();
                if (name) {
                    return {record_var(node, output, *name)};
                } else {
                    return {record_var(node, output)};
                }
            } else {
                return {record_var(node)};
            }
        } else {
            // FIXME: reason for sync
            auto dev_val = get_dev_val();
            auto callback = [dev_val]() mutable -> DeviceTensorND {
                return std::move(dev_val);
            };
            auto* node = opr::InputCallback::make(
                                 *m_graph, callback, dev_val.comp_node(),
                                 dev_val.dtype(), dev_val.shape(), {}, true)[0]
                                 .node();
            return {record_var(node)};
        }
    } else if (auto* get_attr = op.as<GetAttr>()) {
        if (auto* lazy_val = inputs.item().as<LazyEvalValue>()) {
            switch (get_attr->attr()) {
                case GetAttr::DType:
                    return {DTypeValue::make(lazy_val->node()->dtype())};
                case GetAttr::Device:
                    return {CompNodeValue::make(lazy_val->node()->comp_node())};
                case GetAttr::Shape: {
                    if (!cg::is_static_var_shape(lazy_val->node())) {
                        mgb_log_debug("LazyEval: get_shape_failed");
                        return {ValueRef()};
                    }
                    auto shape = m_graph->static_infer_manager().infer_shape(
                            lazy_val->node());
                    return {ShapeValue::make(ValueShape::from(shape))};
                }
                case GetAttr::Value: {
                    if (!cg::is_static_var_value(lazy_val->node())) {
                        mgb_log_debug("LazyEval: get_value failed");
                        return {ValueRef()};
                    }
                    auto inferred_value = m_graph->static_infer_manager().infer_value(
                            lazy_val->node());
                    mgb_assert(inferred_value.comp_node() == CompNode::default_cpu());
                    HostTensorND host_value(
                            lazy_val->node()->comp_node(), lazy_val->node()->dtype());
                    host_value.copy_from(inferred_value);
                    // TODO: use proxy instead?
                    return {HostValue::make(host_value)};
                }
                case GetAttr::Data: {
                    if (!cg::is_static_var_value(lazy_val->node())) {
                        mgb_log_debug("LazyEval get_data failed");
                        return {ValueRef()};
                    }
                    auto inferred_value = m_graph->static_infer_manager().infer_value(
                            lazy_val->node());
                    mgb_assert(inferred_value.comp_node() == CompNode::default_cpu());
                    // TODO: use proxy instead?
                    HostTensorND host_value(
                            lazy_val->node()->comp_node(), lazy_val->node()->dtype());
                    host_value.copy_from(inferred_value);
                    DeviceTensorND dev_value;
                    dev_value.copy_from(host_value);
                    AsyncReleaser::inst()->add(host_value);
                    return {DeviceValue::make(dev_value)};
                }
                default:
                    mgb_throw(
                            MegBrainError, "LazyEval: malformed GetAttr: %s",
                            op.to_string().c_str());
            }
        } else {
            return imperative::apply(op, inputs);
        }
    } else if (auto* rename_value = op.as<RenameValue>()) {
        if (auto* lazy_val = inputs.item().as<LazyEvalValue>()) {
            return {record_var(
                    lazy_val->node(), lazy_val->bound_data(), rename_value->name())};
        } else {
            return imperative::apply(op, inputs);
        }
    } else if (op.is<GetName>()) {
        if (auto* lazy_val = inputs.item().as<LazyEvalValue>()) {
            auto name = lazy_val->name();
            if (!name.empty()) {
                return {StringValue::make(lazy_val->name())};
            } else {
                return {ValueRef()};
            }
        } else {
            return imperative::apply(op, inputs);
        }
    } else {
        return op.fallback(inputs);
    }
}

void LazyEvalTransformation::on_unregister() noexcept {
    std::vector<LazyEvalValue::ref_t> lazy_vals;
    for (auto&& weak_var : m_weak_vars) {
        if (auto lazy_val = weak_var.lock()) {
            lazy_vals.push_back(lazy_val);
        }
    }
    CleanupGuard _{[this] {
        m_graph.reset();
        m_weak_vars.clear();
    }};
    if (m_no_exec) {
        for (auto&& lazy_val : lazy_vals) {
            if (lazy_val->bound_data()) {
                auto value = lazy_val->bound_data();
                lazy_val.reset(value);
            } else {
                lazy_val.reset(ErrorValue::make("no data bound"));
            }
        }
        return;
    }
    std::mutex mtx;
    std::vector<std::pair<LazyEvalValue::ref_t, DeviceTensorND>> values;
    ComputingGraph::OutputSpec output_specs;
    for (auto&& lazy_val : lazy_vals) {
        auto* output = opr::OutputCallback::make(
                               {[lazy_val, &mtx, &values](DeviceTensorND data) {
                                   MGB_LOCK_GUARD(mtx);
                                   values.push_back({lazy_val, data});
                               }},
                               lazy_val->node())
                               .node();
        output_specs.push_back({output, {}});
    }
    if (m_io_link.node()) {
        output_specs.push_back({m_io_link, {}});
    }
    if (output_specs.empty()) {
        return;
    }
    {
        // set_priority_to_id
        auto on_opr = [](mgb::cg::OperatorNodeBase* opr) {
            if (opr->node_prop().attribute().priority == 0) {
                opr->node_prop().attribute().priority = opr->id();
            }
        };
        mgb::cg::DepOprIter dep_iter{on_opr};
        for (auto&& output_spec : output_specs) {
            dep_iter.add(output_spec.first);
        }
    }
    try {
        auto exectuble = m_graph->compile(output_specs);
        exectuble->execute();
        exectuble->wait();
    } catch (...) {
        m_graph_exc = std::current_exception();
    }
    for (auto&& [var, data] : values) {
        var.reset(imperative::apply(
                CreateTensor(CreateTensor::Common, data.comp_node(), data.layout()),
                DeviceStorage::make(data.storage()))[0]);
    }
    for (auto&& lazy_val : lazy_vals) {
        if (lazy_val.is<LazyEvalValue>()) {
            std::string repr =
                    ssprintf("lazy eval failed for %s", lazy_val->to_string().c_str());
            mgb_log_debug("%s", repr.c_str());
            lazy_val.reset(ErrorValue::make(repr.c_str()));
        }
    }
}

void LazyEvalTransformation::check_exception() {
    if (m_graph_exc) {
        std::rethrow_exception(m_graph_exc);
    }
}

}  // namespace imperative
}  // namespace mgb
