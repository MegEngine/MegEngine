#include "./lstm_utils.h"
#include "src/arm_common/lstm/opr_impl.h"
#include "src/arm_common/lstm_cell/cell_kernel.h"
#include "src/arm_common/lstm_cell/opr_impl.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace arm_common;

LstmCellWeight::LstmCellWeight(
        RefPtr weight_ptr, size_t hidden_size, size_t input_size, bool has_bias,
        DType dtype) {
    // weight_ih: [gate_hidden_size, input_size]
    // weight_hh: [gate_hidden_size, hidden_size]
    // bias_ih: [gate_hidden_size]
    // bias_hh: [gate_hidden_size]

    size_t gate_hidden_size = 4 * hidden_size;
    TensorLayout weight_ih_layout{{gate_hidden_size, input_size}, dtype};
    TensorLayout weight_hh_layout{{gate_hidden_size, hidden_size}, dtype};
    TensorLayout bias_layout{{gate_hidden_size}, dtype};
    m_weight_size = 0;
    m_weight_ih = TensorND(weight_ih_layout, weight_ptr);
    m_weight_size += weight_ih_layout.span().dist_byte();
    weight_ptr += weight_ih_layout.span().dist_byte();
    m_weight_hh = TensorND(weight_hh_layout, weight_ptr);
    m_weight_size += weight_hh_layout.span().dist_byte();
    weight_ptr += weight_hh_layout.span().dist_byte();
    if (has_bias) {
        m_bias_ih = TensorND(bias_layout, weight_ptr);
        m_weight_size += bias_layout.span().dist_byte();
        weight_ptr += bias_layout.span().dist_byte();
        m_bias_hh = TensorND(bias_layout, weight_ptr);
        m_weight_size += bias_layout.span().dist_byte();
    }
}

LstmStates::LstmStates(
        const SmallVector<RefPtr> ptr, size_t hidden_size, size_t batch_size,
        DType dtype) {
    auto& h_ptr = ptr[0];
    auto& c_ptr = ptr[1];
    TensorLayout layout{{batch_size, hidden_size}, dtype};
    m_h = TensorND(layout, h_ptr);
    m_c = TensorND(layout, c_ptr);
    m_memory_size = layout.span().dist_byte();
}

TensorNDArray megdnn::arm_common::split_tensor(
        _megdnn_tensor_in tensor, size_t nr_tensor, const TensorLayout& layout) {
    megdnn_assert(
            tensor.layout.span().dist_byte() == nr_tensor * layout.span().dist_byte());

    TensorNDArray tensors;
    auto ptr = tensor.get_ref_ptr();
    for (size_t i = 0; i < nr_tensor; i++) {
        tensors.push_back(TensorND(layout, ptr));
        ptr += layout.span().dist_byte();
    }
    return tensors;
}

namespace megdnn {
namespace arm_common {

template <>
void cell_opr_compute<LSTMCell, LstmStates>(
        _megdnn_tensor_in input, _megdnn_tensor_in weight_ih,
        _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in bias_hh, const LstmStates& state_in, LstmStates& state_out,
        Workspace cell_workspace, Handle* handle) {
    auto opr = handle->create_operator<LSTMCellForward>();
    TensorLayout gates, h_new, c_new;
    opr->deduce_layout(
            input.layout, weight_ih.layout, bias_ih.layout, state_in.m_h.layout,
            weight_hh.layout, bias_hh.layout, state_in.m_c.layout, h_new, c_new, gates);

    auto workspace_bundle = LstmCellCompute::get_workspace_bundle(
            input.layout, weight_ih.layout, bias_ih.layout, state_in.m_h.layout,
            weight_hh.layout, bias_hh.layout, state_in.m_c.layout, h_new, c_new, gates);

    workspace_bundle.set(cell_workspace.raw_ptr);

    TensorND gates_tensor{workspace_bundle.get(0), gates};
    _megdnn_workspace new_workspace = {
            static_cast<dt_byte*>(workspace_bundle.get(1)),
            workspace_bundle.get_size(1)};

    LstmCellCompute::run(
            input, weight_ih, bias_ih, state_in.m_h, weight_hh, bias_hh, state_in.m_c,
            state_out.m_h, state_out.m_c, gates_tensor, new_workspace, handle);
}
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
