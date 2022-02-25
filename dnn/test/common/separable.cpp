#include "test/common/separable_conv.h"

namespace megdnn {
namespace test {
namespace separable_conv {

std::vector<TestArg> get_args() {
    std::vector<TestArg> args;

    param::SeparableConv cur_param;
    for (size_t i = 8; i < 65; i *= 4) {
        for (int ksize_h = 3; ksize_h < 4; ksize_h += 2) {
            int ksize_w = ksize_h;
            cur_param.ksize_h = ksize_h;
            cur_param.ksize_w = ksize_w;
            // if(ksize_h % 2 ==  0)
            cur_param.is_symm_kernel = false;
            args.emplace_back(
                    cur_param, TensorShape{1, 2, i, i},
                    TensorShape{1, 2, 1, (size_t)ksize_h},
                    TensorShape{1, 2, 1, (size_t)ksize_w});
        }
    }
    return args;
}
}  // namespace separable_conv
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen