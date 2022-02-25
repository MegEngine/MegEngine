#include "src/common/lstm_cell.h"
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {

void LSTMCell::deduce_layout(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& bias_ih, const TensorLayout& hx,
        const TensorLayout& weight_hh, const TensorLayout& bias_hh,
        const TensorLayout& cx, TensorLayout& h_new, TensorLayout& c_new,
        TensorLayout& gates) {
    h_new = TensorLayout(hx, hx.dtype);
    c_new = TensorLayout(cx, cx.dtype);
    auto opr = handle()->create_operator<RNNCellForward>();
    opr->param().nonlineMode = param::RNNCell::NonlineMode::IDENTITY;
    opr->deduce_layout(input, weight_ih, bias_ih, hx, weight_hh, bias_hh, gates);
}

void LSTMCell::check_exec(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& bias_ih, const TensorLayout& hx,
        const TensorLayout& weight_hh, const TensorLayout& bias_hh,
        const TensorLayout& cx, const TensorLayout& h_new, const TensorLayout& c_new,
        const TensorLayout& gates, size_t workspace_in_bytes) {
    TensorLayout h_new_expected, c_new_expected, gates_expected;
    auto errmsg = [&]() {
        std::string msg;
        msg.append("input=");
        msg.append(input.to_string());
        msg.append(", weight_ih=");
        msg.append(weight_ih.to_string());
        msg.append(", bias_ih=");
        msg.append(bias_ih.to_string());
        msg.append(", hx=");
        msg.append(hx.to_string());
        msg.append(", weight_hh=");
        msg.append(weight_hh.to_string());
        msg.append(", bias_hh=");
        msg.append(bias_hh.to_string());
        msg.append(", cx=");
        msg.append(cx.to_string());
        return msg;
    };
#define ASSERT_BRIEF(_content) megdnn_assert(_content, "%s", errmsg().c_str());

    ASSERT_BRIEF(input.ndim == 2)
    ASSERT_BRIEF(input.shape[1] == weight_ih.shape[1])
    ASSERT_BRIEF(weight_ih.shape[0] == weight_hh.shape[0])
    ASSERT_BRIEF(weight_hh.shape[0] == 4 * weight_hh.shape[1])
    ASSERT_BRIEF(bias_ih.shape[0] == bias_hh.shape[0])
    ASSERT_BRIEF(hx.ndim == 2)
    ASSERT_BRIEF(hx.shape[0] == input.shape[0])
    ASSERT_BRIEF(hx.shape[1] == cx.shape[1])  // hidden_size
    ASSERT_BRIEF(cx.ndim == 2)
    ASSERT_BRIEF(cx.shape[0] == input.shape[0])
    ASSERT_BRIEF(cx.shape[1] == weight_hh.shape[1])
#undef ASSERT_BRIEF

    deduce_layout(
            input, weight_ih, bias_ih, hx, weight_hh, bias_hh, cx, h_new_expected,
            c_new_expected, gates_expected);
    megdnn_assert_eq_layout(h_new_expected, h_new);
    megdnn_assert_eq_layout(c_new_expected, c_new);
    megdnn_assert_eq_layout(gates_expected, gates);

    auto required_workspace_in_bytes = get_workspace_in_bytes(
            input, weight_ih, bias_ih, hx, weight_hh, bias_hh, cx, h_new, c_new, gates);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

}  // namespace megdnn

namespace megdnn {
namespace lstm_cell {

size_t get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& bias_ih, const TensorLayout& hx,
        const TensorLayout& weight_hh, const TensorLayout& bias_hh,
        const TensorLayout& /*cx*/, const TensorLayout& /*h_new*/,
        const TensorLayout& /*c_new*/, const TensorLayout& gates, Handle* handle) {
    TensorLayout tmp_layout;
    auto opr = handle->create_operator<RNNCellForward>();
    opr->param().nonlineMode = param::RNNCell::NonlineMode::IDENTITY;
    opr->deduce_layout(input, weight_ih, bias_ih, hx, weight_hh, bias_hh, tmp_layout);
    size_t rnn_cell_need = opr->get_workspace_in_bytes(
            input, weight_ih, bias_ih, hx, weight_hh, bias_hh, gates);
    size_t lstm_cell_need = 2 * tmp_layout.span().dist_byte();
    return rnn_cell_need > lstm_cell_need ? rnn_cell_need : lstm_cell_need;
}

void exec(
        _megdnn_tensor_in input, _megdnn_tensor_in weight_ih, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in hx, _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_hh,
        _megdnn_tensor_in cx, _megdnn_tensor_out h_new, _megdnn_tensor_out c_new,
        _megdnn_tensor_out gates, _megdnn_workspace workspace, Handle* handle) {
    auto opr = handle->create_operator<RNNCellForward>();
    opr->param().nonlineMode = param::RNNCell::NonlineMode::IDENTITY;
    opr->exec(input, weight_ih, bias_ih, hx, weight_hh, bias_hh, gates, workspace);
    // activation
    size_t batch_size = hx.layout.shape[0];
    size_t hidden_size = hx.layout.shape[1];

    auto copy_opr = handle->create_operator<TypeCvtForward>();
    TensorND copy_gates{static_cast<void*>(workspace.raw_ptr), gates.layout};
    TensorLayout hidden_layout{TensorShape{hidden_size}, hx.layout.dtype};
    TensorLayout gateinfo_layout{TensorShape{batch_size, hidden_size}, hx.layout.dtype};
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < 4; j++) {
            TensorND half_step_states{
                    // output
                    static_cast<uint8_t*>(gates.raw_ptr()) +
                            (4 * i + j) * hidden_layout.span().dist_byte(),
                    hidden_layout};
            TensorND half_step_output{
                    static_cast<uint8_t*>(copy_gates.raw_ptr()) +
                            j * gateinfo_layout.span().dist_byte() +
                            i * hidden_layout.span().dist_byte(),
                    hidden_layout};
            copy_opr->exec(half_step_states, half_step_output);
        }
    }
    void* workspace_ptr = workspace.raw_ptr + copy_gates.layout.span().dist_byte();
    copy_opr->exec(copy_gates, gates);

    // sigmoid: i f
    TensorND tmp{static_cast<void*>(workspace_ptr), copy_gates.layout};
    TensorLayout gates_ifo_layout{
            TensorShape({batch_size, hidden_size * 2}), copy_gates.layout.dtype};
    TensorND gates_ifo_origin{copy_gates.raw_ptr(), gates_ifo_layout};
    TensorND gates_ifo{tmp.raw_ptr(), gates_ifo_layout};
    auto sigmoid = handle->create_operator<ElemwiseForward>();
    sigmoid->param().mode = Elemwise::Param::Mode::SIGMOID;
    sigmoid->exec({gates_ifo_origin}, gates_ifo);
    // tanh: g
    TensorLayout g_layout{
            TensorShape({batch_size, hidden_size}), copy_gates.layout.dtype};
    TensorND g_origin{
            static_cast<char*>(copy_gates.raw_ptr()) +
                    gates_ifo_layout.span().dist_byte(),
            g_layout};
    TensorND g{
            static_cast<char*>(tmp.raw_ptr()) + gates_ifo_layout.span().dist_byte(),
            g_layout};
    auto tanh = handle->create_operator<ElemwiseForward>();
    tanh->param().mode = Elemwise::Param::Mode::TANH;
    tanh->exec({g_origin}, g);
    // sigmoid: o
    TensorLayout three_gates_ifo_layout{
            TensorShape({batch_size, hidden_size * 3}), copy_gates.layout.dtype};
    TensorLayout o_layout{
            TensorShape({batch_size, hidden_size}), copy_gates.layout.dtype};
    TensorND o_origin{
            static_cast<char*>(copy_gates.raw_ptr()) +
                    three_gates_ifo_layout.span().dist_byte(),
            o_layout};
    TensorND o{
            static_cast<char*>(tmp.raw_ptr()) +
                    three_gates_ifo_layout.span().dist_byte(),
            o_layout};
    sigmoid->exec({o_origin}, o);
    // extract i f o
    TensorND i{static_cast<char*>(tmp.raw_ptr()), g_layout};
    TensorND f{
            static_cast<char*>(tmp.raw_ptr()) + g_layout.span().dist_byte(), g_layout};
    // calculate new cell state
    auto elewise_mul_add = handle->create_operator<ElemwiseForward>();
    elewise_mul_add->param().mode = Elemwise::Param::Mode::FUSE_MUL_ADD4;
    elewise_mul_add->exec({f, cx, i, g}, c_new);
    // calculate new hidden state
    tanh->exec({c_new}, h_new);
    auto elewise_mul = handle->create_operator<ElemwiseForward>();
    elewise_mul->param().mode = Elemwise::Param::Mode::MUL;
    elewise_mul->exec({o, h_new}, h_new);
}

}  // namespace lstm_cell
}  // namespace megdnn
