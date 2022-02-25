#include <limits>
#include "./kern.cuh"
#include "src/common/cond_take/predicate.cuh"
#include "src/cuda/cumsum/kern_impl.cuinl"
#include "src/cuda/query_blocksize.cuh"

using namespace megdnn;
using namespace megdnn::cond_take;
using namespace megdnn::cuda::cond_take;

size_t cuda::cond_take::gen_idx_get_workspace_size(size_t size) {
    megdnn_assert(size < std::numeric_limits<uint32_t>::max());
    return cumsum::get_workspace_in_bytes(1, size, 1, sizeof(IdxType));
}

// vim: ft=cuda syntax=cuda.doxygen
