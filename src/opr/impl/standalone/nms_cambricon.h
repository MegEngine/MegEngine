#pragma once
#include "megbrain/comp_node_env.h"
#include "megbrain_build_config.h"

#if MGB_CAMBRICON
#include "cnnl.h"

namespace mgb {
namespace opr {
namespace standalone {
namespace nms {

/*!
 * \brief Cambricon single-batch nms kernel
 * \param nr_boxes number of input boxes
 * \param max_output max number of entries to be written to out_idx
 * \param overlap_thresh overlapping threshold for IoU
 * \param[in] boxes boxes in [n, 4] layout,
 *      each row containing (x0, y0, x1, y1)
 * \param[out] out_idx indices of boxes to be kept
 * \param[out] out_size number of items written to out_idx; the remaining items
 *      would be filled with the last valid item
 */
void cambricon_kern(
        size_t nr_boxes, size_t max_output, float overlap_thresh, const void* boxes,
        void* out_idx, void* out_size, void* workspace, cnnlHandle_t handle);

size_t cambricon_kern_workspace(cnnlHandle_t handle, size_t nr_boxes);

}  // namespace nms
}  // namespace standalone
}  // namespace opr
}  // namespace mgb

#endif