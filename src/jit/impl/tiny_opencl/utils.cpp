#include "./utils.h"
#include <vector>

#if MGB_JIT && MGB_OPENCL

std::vector<LayoutType> get_channel_broadcast_info(
        const mgb::jit::JITExecutor::Args& args) {
    auto output_dim = args.outputs[0].layout.ndim;
    auto& out_layout = args.outputs[0].layout;
    mgb_assert(
            out_layout.ndim == 5,
            "code issue happened, OpenCL jit only support image now");
    auto n = out_layout[0];
    auto c = out_layout[2] * 4;
    auto h = out_layout[1];
    auto w = out_layout[3];

    std::vector<LayoutType> ret;
    for (size_t i = 0; i < args.inputs.size(); i++) {
        if (args.inputs[i].layout.is_scalar()) {
            ret.push_back(LayoutType::SCALAR);
        } else {
            auto& in_layout = args.inputs[i].layout;
            auto in = in_layout[0];
            auto ic = in_layout[2] * 4;
            auto ih = in_layout[1];
            auto iw = in_layout[3];
            mgb_assert(
                    in_layout.ndim == output_dim && in == n && ic == c,
                    "invalid args for OpenCL jit");

            if (ih == h && iw == w) {
                ret.push_back(LayoutType::VEC);
            } else {
                ret.push_back(LayoutType::CHANNEL_BROADCAST);
                mgb_assert(ih == 1 && iw == 1, "invalid args for OpenCL jit");
            }
        }
    }

    return ret;
}

#endif
