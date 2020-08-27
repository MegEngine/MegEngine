#pragma once

#include <cstddef>
#include <cstdint>

namespace mgb {
namespace opr {
namespace standalone {
namespace nms {

/*!
 * \brief CPU single-batch nms kernel
 *
 * See nms_kern.cuh for explanation on the parameters.
 */
void cpu_kern(size_t nr_boxes, size_t max_output, float overlap_thresh,
              const float* boxes, uint32_t* out_idx, uint32_t* out_size,
              void* workspace);

size_t cpu_kern_workspace(size_t nr_boxes);

}  // namespace nms
}  // namespace standalone
}  // namespace opr
}  // namespace mgb
