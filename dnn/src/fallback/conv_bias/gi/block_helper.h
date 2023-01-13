#include "src/common/utils.h"
namespace megdnn {
namespace {
// block_helper is used to calculate oh block size
static inline int l2_block_helper(
        const int nthread, const int amount, const int size_per_unit) {
    //! TODO: opt config or dynamic config l2_cache_size for different ARCH
    constexpr int l2_cache_size = 256 * 1024;
    const int block_per_thread = div_ceil(amount, nthread);
    const int best_block =
            std::min(amount, (l2_cache_size + size_per_unit / 2) / size_per_unit);
    const int max_block_num = div_ceil(block_per_thread, best_block);
    const int min_block_num = std::max(max_block_num - 1, 1);
    const int max_block = div_ceil(block_per_thread, max_block_num);
    const int min_block = div_ceil(block_per_thread, min_block_num);
    const int max_loss = std::abs(max_block_num * max_block - block_per_thread);
    const int min_loss = std::abs(min_block_num * min_block - block_per_thread);
    int block = max_loss > min_loss ? min_block : max_block;
    return block;
}

}  // namespace
}  // namespace megdnn

// vim: syntax=cpp.doxygen