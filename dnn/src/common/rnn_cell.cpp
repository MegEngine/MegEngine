#include "src/common/rnn_cell.h"
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {

void RNNCell::deduce_layout(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& /*bias_ih*/, const TensorLayout& hx,
        const TensorLayout& /*weight_hh*/, const TensorLayout& /*bias_hh*/,
        TensorLayout& dst) {
    size_t batch_size = hx.shape[0];
    size_t gate_hidden_size = weight_ih.shape[0];

    dst = TensorLayout(TensorShape({batch_size, gate_hidden_size}), input.dtype);
}

void RNNCell::check_exec(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& bias_ih, const TensorLayout& hx,
        const TensorLayout& weight_hh, const TensorLayout& bias_hh,
        const TensorLayout& dst, size_t workspace_in_bytes) {
    TensorLayout dst_expected;
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
        msg.append(", dst=");
        msg.append(dst.to_string());
        return msg;
    };
#define ASSERT_BRIEF(_content) megdnn_assert(_content, "%s", errmsg().c_str());

    ASSERT_BRIEF(input.ndim == 2)
    ASSERT_BRIEF(hx.ndim == 2)
    ASSERT_BRIEF(hx.shape[0] == input.shape[0])  // batch
    ASSERT_BRIEF(input.shape[1] == weight_ih.shape[1])
    ASSERT_BRIEF(hx.shape[0] == dst.shape[0])  // batch
    ASSERT_BRIEF(hx.shape[1] == dst.shape[1])
    ASSERT_BRIEF(hx.shape[1] == weight_ih.shape[0])  // hidden_size
    ASSERT_BRIEF(weight_ih.shape[0] == weight_hh.shape[0])
    ASSERT_BRIEF(weight_hh.shape[0] == weight_hh.shape[1])
    ASSERT_BRIEF(bias_ih.shape[0] == bias_hh.shape[0])
#undef ASSERT_BRIEF
    megdnn_assert_eq_dtype(input, dst);
    megdnn_assert_eq_dtype(hx, dst);
    deduce_layout(input, weight_ih, bias_ih, hx, weight_hh, bias_hh, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);

    auto required_workspace_in_bytes = get_workspace_in_bytes(
            input, weight_ih, bias_ih, hx, weight_hh, bias_hh, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

}  // namespace megdnn

namespace megdnn {
namespace rnn_cell {

size_t get_workspace_in_bytes(
        const TensorLayout& input, const TensorLayout& weight_ih,
        const TensorLayout& /*bias_ih*/, const TensorLayout& hx,
        const TensorLayout& weight_hh, const TensorLayout& /*bias_hh*/,
        const TensorLayout& dst, Handle* handle) {
    auto opr = handle->create_operator<MatrixMulForward>();
    opr->param().transposeB = true;
    return dst.span().dist_byte() +
           std::max(
                   opr->get_workspace_in_bytes(hx, weight_hh, dst),
                   opr->get_workspace_in_bytes(input, weight_ih, dst));
}

void exec(
        _megdnn_tensor_in input, _megdnn_tensor_in weight_ih, _megdnn_tensor_in bias_ih,
        _megdnn_tensor_in hx, _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_hh,
        _megdnn_tensor_out dst, _megdnn_workspace workspace,
        param::RNNCell::NonlineMode nonline_mode, Handle* handle) {
    TensorND tmp{static_cast<void*>(workspace.raw_ptr), dst.layout};
    _megdnn_workspace new_workspace = {
            workspace.raw_ptr + dst.layout.span().dist_byte(),
            workspace.size - dst.layout.span().dist_byte()};
    auto opr = handle->create_operator<MatrixMulForward>();
    opr->param().transposeB = true;
    opr->exec(input, weight_ih, tmp, new_workspace);
    opr->exec(hx, weight_hh, dst, new_workspace);
    auto add_opr = handle->create_operator<ElemwiseForward>();
    add_opr->param().mode = Elemwise::Param::Mode::ADD;
    add_opr->exec({dst, tmp}, dst);
    add_opr->exec({dst, bias_ih}, dst);
    add_opr->exec({dst, bias_hh}, dst);

    // activation
    using NonlineMode = param::RNNCell::NonlineMode;

    switch (nonline_mode) {
#define cb(_mode)                                                    \
    case NonlineMode::_mode: {                                       \
        auto nonlinear = handle->create_operator<ElemwiseForward>(); \
        nonlinear->param().mode = Elemwise::Param::Mode::_mode;      \
        nonlinear->exec({dst}, dst);                                 \
        break;                                                       \
    }
        cb(RELU);
        cb(TANH);
#undef cb
        case NonlineMode::IDENTITY:
            break;
        default:
            megdnn_assert(false);
    }
}

}  // namespace rnn_cell
}  // namespace megdnn