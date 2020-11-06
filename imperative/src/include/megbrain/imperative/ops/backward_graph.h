/**
 * \file imperative/src/include/megbrain/imperative/ops/backward_graph.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/op_def.h"

namespace mgb {
namespace imperative {

// a special OpDef used for taking gradient on physical tensor
struct BackwardGraph final : public OpDefImplBase<BackwardGraph> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
public:
    struct InternalGraph {
        // op, inputs, outputs
        using Expr = std::tuple<std::shared_ptr<OpDef>,
                std::vector<size_t>, std::vector<size_t>>;
        std::vector<Expr> exprs;

        // index array of input nodes
        std::vector<size_t> inputs;

        // index array of output nodes
        std::vector<size_t> outputs;

        // pair of (node index, correspending constant)
        std::vector<std::pair<size_t, TensorPtr>> constants;

        SmallVector<TensorPtr>
        apply(const SmallVector<TensorPtr>& inputs) const;

        std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_attrs(
            const SmallVector<LogicalTensorDesc>& inputs) const;

        template <typename T, typename F, typename C>
        SmallVector<T> interpret(F&& f, C&& c, const SmallVector<T>& inputs) const {
            ThinHashMap<size_t, T> node2tensor;
            auto&& input_nodes = this->inputs;
            mgb_assert(inputs.size() == input_nodes.size());
            for (size_t i = 0; i < inputs.size(); ++ i) {
                node2tensor[input_nodes[i]] = inputs[i];
            }
            for (auto &&i : constants) {
                node2tensor[i.first] = c(i.second);
            }
            for (size_t i = 0; i < exprs.size(); ++ i) {
                auto&& expr = exprs[i];
                SmallVector<T> inputs;
                for (auto &&in : std::get<1>(expr)) {
                    inputs.push_back(node2tensor.at(in));
                }
                auto&& outputs = f(*std::get<0>(expr), std::move(inputs));
                auto&& output_nodes = std::get<2>(expr);
                mgb_assert(outputs.size() == output_nodes.size());
                for (size_t i = 0; i < outputs.size(); ++ i) {
                    node2tensor[output_nodes[i]] = std::move(outputs[i]);
                }
            }
            SmallVector<T> ret;
            for (auto &&i : outputs) {
                ret.push_back(node2tensor.at(i));
            }
            return ret;
        }
    };

    const InternalGraph& graph() const {
        return m_graph;
    }

    InternalGraph& graph() {
        return m_graph;
    }

    bool is_same_st(const Hashable& rhs) const override {
        if (!rhs.same_type<BackwardGraph>()) {
            return false;
        }
        auto& other = rhs.cast_final_safe<BackwardGraph>();
        if (this == &other) {
            return true;
        }
        // FIXME
        return false;
    }

private:
    InternalGraph m_graph;
};

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
