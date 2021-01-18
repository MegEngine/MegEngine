/**
 * \file imperative/src/impl/backward_graph_opt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/backward_graph_opt.h"
#include "megbrain/imperative/ops/backward_graph.h"
#include "megbrain/imperative/ops/autogen.h"

using namespace mgb;
using namespace imperative;

OptimizedBackwardGraphResult::OptimizedBackwardGraphResult(const BackwardGraphResult& src)
        : input_has_grad(src.input_has_grad) {
    if (!src.backward->same_type<BackwardGraph>()) {
        // backward graph only contains a single op
        backward = src.backward;
        save_for_backward = src.save_for_backward;
        return;
    }
    save_for_backward.resize(src.save_for_backward.size(), false);
    precomp.reset(new BackwardGraph);
    backward.reset(new BackwardGraph);

    auto&& graph = src.backward->cast_final_safe<BackwardGraph>().graph();
    auto&& mask = src.save_for_backward;
    size_t input_size = src.input_has_grad.size();
    size_t output_size = (mask.size() - input_size) / 2;
    mgb_assert(input_size + output_size * 2 == mask.size());

    auto& fgraph = precomp->cast_final<BackwardGraph>().graph();
    auto& bgraph = backward->cast_final<BackwardGraph>().graph();

    // optimization: move ops (e.g. GetVarShape) to forward to
    // reduce memory footprint

    struct VInfo {
        bool appears_in_backward = false;
    };
    std::unordered_map<size_t, VInfo> vinfo;

    // step 1.1: ops not in whitelist must run in backward.
    // mark their inputs as always appears in backward
    for (auto&& [op, iv, ov] : graph.exprs) {
        if (!op->same_type<GetVarShape>()) {
            for (auto&& v : iv) {
                vinfo[v].appears_in_backward = true;
            }
        }
    }
    // step 1.2: inputs only available in backward (i.e. grads)
    // should be marked as always appears in backward
    for (size_t i = 0, j = 0; i < mask.size(); ++i) {
        if (!mask[i]) continue;
        if (i >= input_size + output_size) {
            vinfo[graph.inputs[j]].appears_in_backward = true;
        }
        ++j;
    }

    // step 2: try to move ops to forward, if not all their inputs
    // are marked always appears in backward (otherwise no memory saving)
    for (auto&& expr : graph.exprs) {
        auto&& [op, iv, ov] = expr;
        if (std::all_of(iv.begin(), iv.end(), [&](auto&& v){return vinfo[v].appears_in_backward;})) {
            bgraph.exprs.push_back(expr);
            for (auto&& v : ov) {
                vinfo[v].appears_in_backward = true;
            }
            // logically should also mark all inputs as appears in backward
            // but clearly that's a no-op.
        } else {
            fgraph.exprs.push_back(expr);
            for (auto&& v : ov) {
                if (vinfo[v].appears_in_backward) {
                    // appears_in_backward won't change after this point
                    // so it is safe to set fgraph.outputs based on current value
                    fgraph.outputs.push_back(v);
                }
            }
        }
    }

    // initialize remaining parts

    fgraph.constants = graph.constants;
    fgraph.inputs.reserve(input_size + output_size);
    for (size_t i = 0, j = 0; i < input_size + output_size; ++i) {
        if (!mask[i]) {
            fgraph.inputs.push_back(1000000000 + i);
            continue;
        }
        fgraph.inputs.push_back(graph.inputs[j++]);
    }

    bgraph.constants = graph.constants;
    bgraph.outputs = graph.outputs;
    bgraph.inputs = fgraph.outputs;
    for (size_t i = 0, j = 0; i < mask.size(); ++i) {
        if (mask[i]) {
            auto&& v = graph.inputs[j++];
            if (vinfo[v].appears_in_backward) {
                save_for_backward[i] = true;
                bgraph.inputs.push_back(v);
            }
        }
    }

    if (!fgraph.outputs.size()) {
        precomp.reset();
    }
}
