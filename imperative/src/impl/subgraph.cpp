/**
 * \file imperative/src/impl/subgraph.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/imperative/subgraph.h"

namespace mgb {
namespace imperative {

void Subgraph::remove_unused_exprs() {
    std::unordered_set<size_t> required_vars = {outputs.begin(), outputs.end()};
    required_vars.erase(0);
    for (auto iter = exprs.rbegin(); iter != exprs.rend(); ++iter) {
        auto& expr = *iter;
        bool required = false;
        for (auto output : expr.outputs) {
            if (required_vars.count(output)) {
                required = true;
                break;
            }
        }
        if (required) {
            required_vars.insert(expr.inputs.begin(), expr.inputs.end());
        } else {
            expr.op = nullptr;
        }
    }
    exprs.erase(std::remove_if(exprs.begin(), exprs.end(),
                               [](auto expr) { return expr.op == nullptr; }),
                exprs.end());
}

SmallVector<bool> Subgraph::gen_input_mask() {
    std::unordered_set<size_t> unused_inputs = {inputs.begin(), inputs.end()};
    for (auto&& expr : exprs) {
        for (auto&& input : expr.inputs) {
            unused_inputs.erase(input);
        }
    }
    for (auto&& output : outputs) {
        unused_inputs.erase(output);
    }
    unused_inputs.insert(0);
    SmallVector<bool> mask(inputs.size(), true);
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (unused_inputs.count(inputs[i])) {
            mask[i] = false;
        }
    }
    return mask;
}

SmallVector<bool> Subgraph::gen_output_mask() {
    std::unordered_set<size_t> invalid_outputs = {outputs.begin(),
                                                  outputs.end()};
    for (auto&& input : inputs) {
        invalid_outputs.erase(input);
    }
    for (auto&& expr : exprs) {
        for (auto&& output : expr.outputs) {
            invalid_outputs.erase(output);
        }
    }
    for (auto&& constant: constants) {
        invalid_outputs.erase(constant.first);
    }
    invalid_outputs.insert(0);
    SmallVector<bool> mask(outputs.size(), true);
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (invalid_outputs.count(outputs[i])) {
            mask[i] = false;
        }
    }
    return mask;
}

void Subgraph::replace_vars(
        const std::unordered_map<size_t, size_t>& replace_map) {
    // FIXME: preprocess replace_map
    auto replace_var = [&](var_t& var) {
        // TODO: detect infinite loop
        while (replace_map.count(var)) {
            var = replace_map.at(var);
        }
    };
    for (auto& expr : exprs) {
        for (auto& input : expr.inputs) {
            replace_var(input);
        }
    }
    for (auto& output : outputs) {
        replace_var(output);
    }
}

std::string EncodedSubraph::repr() const {
    std::string buffer;
    buffer.push_back('|');
    for (size_t i = 0; i < input_mask.size(); ++i) {
        buffer.push_back(input_mask[i] ? '#' : ' ');
    }
    buffer.push_back('|');
    buffer.push_back('\n');
    buffer.append(graph.repr());
    buffer.push_back('|');
    for (size_t i = 0; i < output_mask.size(); ++i) {
        buffer.push_back(output_mask[i] ? '#' : ' ');
    }
    buffer.push_back('|');
    return buffer;
}

size_t EncodedSubraph::hash() const {
    return std::hash<std::string>{}(repr());
}

}  // namespace imperative
}  // namespace mgb
