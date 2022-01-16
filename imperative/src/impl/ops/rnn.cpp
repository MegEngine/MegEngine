/**
 * \file imperative/src/impl/ops/rnn.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/dnn/rnn.h"
#include "megbrain/imperative/ops/autogen.h"

#include "../op_trait.h"

namespace mgb::imperative {

namespace rnn_cell {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const RNNCell&>(def);
    mgb_assert(inputs.size() == 6);
    return opr::RNNCell::make(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
            op.param());
}
OP_TRAIT_REG(RNNCell, RNNCell).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace rnn_cell

namespace lstm_cell {
VarNodeArray apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const LSTMCell&>(def);
    mgb_assert(inputs.size() == 7);
    auto* opr = opr::LSTMCell::make(
                        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                        inputs[5], inputs[6], op.param())
                        .node()
                        ->owner_opr();
    return {opr->output(0), opr->output(1), opr->output(2)};
}
OP_TRAIT_REG(LSTMCell, LSTMCell).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace lstm_cell

namespace rnn {
VarNodeArray apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const RNN&>(def);
    mgb_assert(inputs.size() == 3);
    auto* opr = opr::RNN::make(inputs[0], inputs[1], inputs[2], op.param())
                        .node()
                        ->owner_opr();
    return {opr->output(0), opr->output(1), opr->output(2)};
}
OP_TRAIT_REG(RNN, RNN).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace rnn

namespace lstm {
VarNodeArray apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const LSTM&>(def);
    mgb_assert(inputs.size() == 4);
    auto* opr = opr::LSTM::make(inputs[0], inputs[1], inputs[2], inputs[3], op.param())
                        .node()
                        ->owner_opr();
    return {opr->output(0), opr->output(1), opr->output(2), opr->output(3)};
}
OP_TRAIT_REG(LSTM, LSTM).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace lstm

}  // namespace mgb::imperative
