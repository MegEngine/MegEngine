#include "src/naive/rnn/rnn.h"

#include <cmath>
#include <cstring>

namespace megdnn {
namespace naive {
namespace rnn {

CellWeightsWrapperBase::CellWeightsWrapperBase(
        void* weight_ptr, size_t hidden_size, size_t input_size, size_t num_chunks,
        bool has_bias, DType dtype, _megdnn_workspace workspace) {
    // weight_ih: [gate_hidden_size, input_size]
    // weight_hh: [gate_hidden_size, hidden_size]
    // bias_ih: [gate_hidden_size]
    // bias_hh: [gate_hidden_size]
    size_t gate_hidden_size = num_chunks * hidden_size;
    TensorLayout weight_ih_layout{{gate_hidden_size, input_size}, dtype};
    TensorLayout weight_hh_layout{{gate_hidden_size, hidden_size}, dtype};
    TensorLayout bias_layout{{gate_hidden_size}, dtype};
    this->_weight_size = 0;
    this->weight_ih = TensorND(weight_ptr, weight_ih_layout);
    this->_weight_size += weight_ih_layout.span().dist_byte();
    this->weight_hh = TensorND(
            static_cast<uint8_t*>(weight_ptr) + this->_weight_size, weight_hh_layout);
    this->_weight_size += weight_hh_layout.span().dist_byte();
    if (has_bias) {
        this->bias_ih = TensorND(
                static_cast<uint8_t*>(weight_ptr) + this->_weight_size, bias_layout);
        this->_weight_size += bias_layout.span().dist_byte();
        this->bias_hh = TensorND(
                static_cast<uint8_t*>(weight_ptr) + this->_weight_size, bias_layout);
        this->_weight_size += bias_layout.span().dist_byte();
        this->_workspace_size = 0;
    } else {
        this->bias_ih = TensorND(workspace.raw_ptr, bias_layout);
        this->bias_hh = TensorND(workspace.raw_ptr, bias_layout);
        memset(workspace.raw_ptr, 0, bias_layout.span().dist_byte());
        this->_workspace_size = bias_layout.span().dist_byte();
    }
}

size_t CellWeightsWrapperBase::weight_size_in_bytes() const {
    return this->_weight_size;
}

size_t CellWeightsWrapperBase::workspace_size_in_bytes() const {
    return this->_workspace_size;
}

size_t CellWeightsWrapperBase::num_states() const {
    return 1;
}

void CellWeightsWrapperBase::backward(
        Handle* handle, param::RNNCell::NonlineMode nonlineMode, _megdnn_tensor_in x,
        const TensorNDArray& states, _megdnn_tensor_in y, const TensorNDArray& douts,
        _megdnn_tensor_out dx, TensorNDArray& dstates, _megdnn_tensor_out dwi,
        _megdnn_tensor_out dwh, _megdnn_tensor_out dbias,
        _megdnn_workspace workspace) const {
    auto dy = douts[0];
    using NonlineMode = param::RNNCell::NonlineMode;
    using Mode = Elemwise::Mode;
    auto elemwise_opr = handle->create_operator<ElemwiseForward>();
    TensorND tmp = {workspace.raw_ptr, dy.layout};
    auto new_workspace = Workspace(
            workspace.raw_ptr + tmp.layout.span().dist_byte(),
            workspace.size - tmp.layout.span().dist_byte());
    switch (nonlineMode) {
        case (NonlineMode::IDENTITY):
            memcpy(tmp.raw_ptr(), dy.raw_ptr(), dy.layout.span().dist_byte());
            break;
        case (NonlineMode::TANH):
            elemwise_opr->param().mode = Mode::TANH_GRAD;
            elemwise_opr->exec({y, dy}, tmp);
            break;
        case (NonlineMode::RELU):
            elemwise_opr->param().mode = Mode::SWITCH_GT0;
            elemwise_opr->exec({y, dy}, tmp);
            break;
    }
    auto matrixmul_opr = handle->create_operator<MatrixMulForward>();
    matrixmul_opr->param().transposeA = false;
    matrixmul_opr->param().transposeB = false;
    // dx
    matrixmul_opr->exec(tmp, this->weight_ih, dx, new_workspace);
    // dhx
    matrixmul_opr->exec(tmp, this->weight_hh, dstates[0], new_workspace);
    // dwi
    matrixmul_opr->param().transposeA = true;
    matrixmul_opr->exec(tmp, x, dwi, new_workspace);
    // dwh
    matrixmul_opr->exec(tmp, states[0], dwh, new_workspace);
    // dbias
    auto sum_opr = handle->create_operator<ReduceForward>();
    sum_opr->param().mode = ReduceForward::Mode::SUM;
    sum_opr->param().axis = 0;
    TensorND dbias_expanded = {
            dbias.raw_ptr(), {{1, dbias.layout.shape[0]}, dbias.layout.dtype}};
    sum_opr->exec(tmp, dbias_expanded, new_workspace);
}

size_t CellWeightsWrapperBase::backward_workspace_size_in_bytes(
        Handle* handle, size_t batch_size, size_t hidden_size, size_t input_size,
        size_t num_chunks, DType dtype) {
    size_t gate_hidden_size = hidden_size * num_chunks;
    TensorLayout tmp = {{batch_size, gate_hidden_size}, dtype};
    TensorLayout bias_expanded = {{1, gate_hidden_size}, dtype};
    TensorLayout wih = {{gate_hidden_size, input_size}, dtype};
    TensorLayout whh = {{gate_hidden_size, hidden_size}, dtype};
    TensorLayout x = {{batch_size, input_size}, dtype};
    TensorLayout hx = {{batch_size, hidden_size}, dtype};
    size_t workspace_size = 0;
    auto matrixmul_opr = handle->create_operator<MatrixMulForward>();
    matrixmul_opr->param().transposeA = false;
    matrixmul_opr->param().transposeB = false;
    // dx
    workspace_size = std::max(
            workspace_size, matrixmul_opr->get_workspace_in_bytes(tmp, wih, x));
    // dhx
    workspace_size = std::max(
            workspace_size, matrixmul_opr->get_workspace_in_bytes(tmp, whh, hx));
    // dwi
    matrixmul_opr->param().transposeA = true;
    workspace_size = std::max(
            workspace_size, matrixmul_opr->get_workspace_in_bytes(tmp, x, wih));
    // dwh
    workspace_size = std::max(
            workspace_size, matrixmul_opr->get_workspace_in_bytes(tmp, hx, whh));
    // dbias
    auto sum_opr = handle->create_operator<ReduceForward>();
    sum_opr->param().mode = ReduceForward::Mode::SUM;
    sum_opr->param().axis = 0;
    workspace_size = std::max(
            workspace_size, sum_opr->get_workspace_in_bytes(tmp, bias_expanded));
    workspace_size += tmp.span().dist_byte();
    return workspace_size;
}

RNNCellWeightWrapper::RNNCellWeightWrapper(
        void* weight_ptr, size_t hidden_size, size_t input_size, bool has_bias,
        DType dtype, _megdnn_workspace workspace)
        : CellWeightsWrapperBase(
                  weight_ptr, hidden_size, input_size, 1, has_bias, dtype, workspace) {}

size_t RNNCellWeightWrapper::backward_workspace_size_in_bytes(
        Handle* handle, size_t batch_size, size_t hidden_size, size_t input_size,
        DType dtype) {
    return CellWeightsWrapperBase::backward_workspace_size_in_bytes(
            handle, batch_size, hidden_size, input_size, 1, dtype);
}

LSTMCellWeightWrapper::LSTMCellWeightWrapper(
        void* weight_ptr, size_t hidden_size, size_t input_size, bool has_bias,
        DType dtype, _megdnn_workspace workspace)
        : CellWeightsWrapperBase(
                  weight_ptr, hidden_size, input_size, 4, has_bias, dtype, workspace) {}

size_t LSTMCellWeightWrapper::num_states() const {
    return 2;
}

size_t LSTMCellWeightWrapper::backward_workspace_size_in_bytes(
        Handle* handle, size_t batch_size, size_t hidden_size, size_t input_size,
        DType dtype) {
    // get gates size
    size_t gate_hidden_size = 4 * hidden_size;
    auto lstm_opr = handle->create_operator<LSTMCellForward>();
    TensorLayout x = {{batch_size, input_size}, dtype};
    TensorLayout weight_ih = {{gate_hidden_size, input_size}, dtype};
    TensorLayout weight_hh = {{gate_hidden_size, hidden_size}, dtype};
    TensorLayout bias = {{gate_hidden_size}, dtype};
    TensorLayout h = {{batch_size, hidden_size}, dtype};
    TensorLayout gates, h_new, c_new;
    lstm_opr->deduce_layout(
            x, weight_ih, bias, h, weight_hh, bias, h, h_new, c_new, gates);
    return CellWeightsWrapperBase::backward_workspace_size_in_bytes(
                   handle, batch_size, hidden_size, input_size, 4, dtype) +
           gates.span().dist_byte() * 2 + c_new.span().dist_byte();
}

void LSTMCellWeightWrapper::backward(
        Handle* handle,
        param::RNNCell::NonlineMode nonlineMode,  // nonlineMode must be identity
        _megdnn_tensor_in x, const TensorNDArray& states, _megdnn_tensor_in y,
        const TensorNDArray& douts, _megdnn_tensor_out dx, TensorNDArray& dstates,
        _megdnn_tensor_out dwi, _megdnn_tensor_out dwh, _megdnn_tensor_out dbias,
        _megdnn_workspace workspace) const {
    size_t used_workspace_size = 0;
    // get gates
    auto lstm_opr = handle->create_operator<LSTMCellForward>();
    TensorLayout gates, h_new, c_new;
    lstm_opr->deduce_layout(
            x.layout, weight_ih.layout, bias_ih.layout, states[0].layout,
            weight_hh.layout, bias_hh.layout, states[1].layout, h_new, c_new, gates);
    TensorND gates_tensor{workspace.raw_ptr, gates};
    used_workspace_size += gates.span().dist_byte();
    TensorND gates_grad{workspace.raw_ptr + used_workspace_size, gates};
    used_workspace_size += gates.span().dist_byte();
    TensorND tanh_cy{workspace.raw_ptr + used_workspace_size, y.layout};
    used_workspace_size += tanh_cy.layout.span().dist_byte();
    Workspace new_workspace = Workspace(
            workspace.raw_ptr + used_workspace_size,
            workspace.size - used_workspace_size);
    // temporarily use dstates to store hy, cy
    // only gates and cy needed, other output will be cleared afterwards
    lstm_opr->exec(
            x, weight_ih, bias_ih, states[0], weight_hh, bias_hh, states[1], dstates[0],
            dstates[1], gates_tensor,
            new_workspace);  // no information left in the workspace

    // BUG: The order of gate_grad if i_g f_g o_g g_g , but it should be  i_g f_g g_g o_g
    //      The returned gradient includes both horizontal and vertical gradients,
    //      horizontal grad = douts[1]  vertical gradients = douts[1]
    //      Here the variable is confusing !!!
    TensorLayout single_gate = {{gates.shape[0], gates.shape[1] / 4}, gates.dtype};
    TensorND i, f, o, g, i_grad, f_grad, o_grad,
            g_grad;  // grad refers to the grad of gates before activation
    i = {gates_tensor.raw_ptr(), single_gate};
    f = {static_cast<uint8_t*>(gates_tensor.raw_ptr()) + single_gate.span().dist_byte(),
         single_gate};
    o = {static_cast<uint8_t*>(f.raw_ptr()) + single_gate.span().dist_byte(),
         single_gate};
    g = {static_cast<uint8_t*>(o.raw_ptr()) + single_gate.span().dist_byte(),
         single_gate};
    i_grad = {gates_grad.raw_ptr(), single_gate};
    f_grad = {
            static_cast<uint8_t*>(i_grad.raw_ptr()) + single_gate.span().dist_byte(),
            single_gate};
    o_grad = {
            static_cast<uint8_t*>(f_grad.raw_ptr()) + single_gate.span().dist_byte(),
            single_gate};
    g_grad = {
            static_cast<uint8_t*>(o_grad.raw_ptr()) + single_gate.span().dist_byte(),
            single_gate};
    auto elem_opr = handle->create_operator<ElemwiseForward>();

    elem_opr->param().mode = Elemwise::Mode::SIGMOID;
    elem_opr->exec({i}, i);
    elem_opr->exec({f}, f);
    elem_opr->exec({o}, o);
    elem_opr->param().mode = Elemwise::Mode::TANH;
    elem_opr->exec({g}, g);
    elem_opr->exec({dstates[1]}, tanh_cy);
    auto mul_opr = handle->create_operator<ElemwiseForward>();
    mul_opr->param().mode = Elemwise::Mode::MUL;
    // use dstates[0] as tmp tensor to store dhy * tanh_cy
    mul_opr->exec({douts[0], tanh_cy}, dstates[0]);
    elem_opr->param().mode = Elemwise::Mode::SIGMOID_GRAD;
    elem_opr->exec({o, dstates[0]}, o_grad);  // grad of gate o
    mul_opr->exec({douts[0], o}, dstates[0]);

    elem_opr->param().mode = Elemwise::Mode::TANH_GRAD;
    elem_opr->exec({tanh_cy, dstates[0]}, dstates[1]);  // grad of cy from hy
    elem_opr->param().mode = Elemwise::Mode::ADD;
    elem_opr->exec({douts[1], dstates[1]}, dstates[1]);  // true grad of cy
    // use dstates[0] as tmp tensor to store dcy * cx
    mul_opr->exec({dstates[1], states[1]}, dstates[0]);
    elem_opr->param().mode = Elemwise::Mode::SIGMOID_GRAD;
    elem_opr->exec({f, dstates[0]}, f_grad);  // grad of gate f
    // use dstates[0] as tmp tensor to store dcy * g
    mul_opr->exec({dstates[1], g}, dstates[0]);
    elem_opr->exec({i, dstates[0]}, i_grad);  // grad of gate i
    // use dstates[0] as tmp tensor to store dcy * i
    mul_opr->exec({dstates[1], i}, dstates[0]);
    elem_opr->param().mode = Elemwise::Mode::TANH_GRAD;
    elem_opr->exec({g, dstates[0]}, g_grad);  // grad of gate g

    // grad if cx
    mul_opr->exec({dstates[1], f}, dstates[1]);
    TensorNDArray base_dstates = {dstates[0]};
    CellWeightsWrapperBase::backward(
            handle, nonlineMode, x, {states[0]}, gates_tensor, {gates_grad}, dx,
            base_dstates, dwi, dwh, dbias, new_workspace);
}

}  // namespace rnn
}  // namespace naive
}  // namespace megdnn
