#include "src/common/rnn.h"
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {

void RNN::deduce_layout(
        const TensorLayout& input, const TensorLayout& hx,
        const TensorLayout& /*flatten_weights*/, TensorLayout& output, TensorLayout& hy,
        TensorLayout& reserve_space) {
    size_t seq_len = input.shape[0];
    size_t batch_size = input.shape[1];
    size_t D = param().bidirectional ? 2 : 1;
    size_t hidden_size = hx.shape[2];
    output = TensorLayout(
            TensorShape{seq_len, batch_size, D * hidden_size}, input.dtype);
    hy = TensorLayout(hx);
    reserve_space = {{get_reserve_size_in_bytes(input)}, input.dtype};
}

void RNN::check_exec(
        const TensorLayout& input, const TensorLayout& hx,
        const TensorLayout& flatten_weights, const TensorLayout& output,
        const TensorLayout& hy, const TensorLayout& /*reserve_space*/,
        size_t /*workspace_in_bytes*/) {
    auto errmsg = [&]() {
        std::string msg;
        msg.append("input=");
        msg.append(input.to_string());
        msg.append(", output=");
        msg.append(output.to_string());
        msg.append(", hx=");
        msg.append(hx.to_string());
        msg.append(", flatten_weights=");
        msg.append(flatten_weights.to_string());
        msg.append(", hy=");
        msg.append(hy.to_string());
        msg.append(", hidden_size=");
        msg.append(std::to_string(param().hidden_size));
        msg.append(", num_layers=");
        msg.append(std::to_string(param().num_layers));
        msg.append(", bidirectional=");
        msg.append(std::to_string(param().bidirectional));
        return msg;
    };
    size_t D = param().bidirectional ? 2 : 1;
    size_t b = param().bias ? 1 : 0;
    size_t num_layers = param().num_layers;
    size_t input_size = input.shape[2];
    size_t gate_hidden_size = param().hidden_size;
    // calculate size_dim1 the same as lstm
    size_t size_dim1 = D * (input_size + param().hidden_size) +
                       (num_layers - 1) * D * ((D + 1) * param().hidden_size) +
                       b * 2 * D * num_layers;
#define ASSERT_BRIEF(_content) megdnn_assert(_content, "%s", errmsg().c_str());

    ASSERT_BRIEF(hx.ndim == 3)
    ASSERT_BRIEF(input.ndim == 3)
    ASSERT_BRIEF(output.ndim == 3)
    ASSERT_BRIEF(hy.ndim == 3)
    ASSERT_BRIEF(flatten_weights.shape[0] == gate_hidden_size)
    ASSERT_BRIEF(flatten_weights.shape[0] == size_dim1)
    ASSERT_BRIEF(hx.shape[0] == D * num_layers)
    ASSERT_BRIEF(hx.shape[1] == input.shape[1])  // batch_size
    ASSERT_BRIEF(hx.shape[2] == param().hidden_size)
    ASSERT_BRIEF(output.shape[0] == input.shape[0])
    ASSERT_BRIEF(output.shape[1] == input.shape[1])
    ASSERT_BRIEF(output.shape[2] == D * param().hidden_size)
    ASSERT_BRIEF(hy.shape[0] == hx.shape[0])
    ASSERT_BRIEF(hy.shape[1] == hx.shape[1])
    ASSERT_BRIEF(hy.shape[2] == hx.shape[2])
#undef ASSERT_BRIEF
}

void RNNBackward::deduce_layout(
        const TensorLayout& x, const TensorLayout& /*y*/, const TensorLayout& hx,
        const TensorLayout& /*dy*/, const TensorLayout& /*dhy*/,
        const TensorLayout& flatten_weights, const TensorLayout& /*reserve_space*/,
        TensorLayout& dx, TensorLayout& dhx, TensorLayout& dw) {
    dx = x;
    dhx = hx;
    dw = flatten_weights;
}

void RNNBackward::check_exec(
        const TensorLayout& /*x*/, const TensorLayout& /*y*/,
        const TensorLayout& /*hx*/, const TensorLayout& /*dy*/,
        const TensorLayout& /*dhy*/, const TensorLayout& /*flatten_weights*/,
        const TensorLayout& /*reserve_space*/, const TensorLayout& /*dx*/,
        const TensorLayout& /*dhx*/, const TensorLayout& /*dw*/,
        size_t /*workspace_in_bytes*/) {}

}  // namespace megdnn
