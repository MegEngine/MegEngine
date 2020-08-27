#include "./nms_cpu.h"

#include <algorithm>
#include <cstring>

namespace {
struct Box {
    float x0, y0, x1, y1;
};

bool box_iou(Box a, Box b, float thresh) {
    using std::max;
    using std::min;
    float left = max(a.x0, b.x0), right = min(a.x1, b.x1);
    float top = max(a.y0, b.y0), bottom = min(a.y1, b.y1);
    float width = max(right - left, 0.f),
          height = max(bottom - top, 0.f);
    float interS = width * height;
    float Sa = (a.x1 - a.x0) * (a.y1 - a.y0);
    float Sb = (b.x1 - b.x0) * (b.y1 - b.y0);
    return interS > (Sa + Sb - interS) * thresh;
}
}  // anonymous namespace

size_t mgb::opr::standalone::nms::cpu_kern_workspace(size_t nr_boxes) {
    return (((nr_boxes - 1) / sizeof(size_t)) + 1) * sizeof(size_t);
}

void mgb::opr::standalone::nms::cpu_kern(size_t nr_boxes, size_t max_output,
                                         float overlap_thresh,
                                         const float* boxes, uint32_t* out_idx,
                                         uint32_t* out_size, void* workspace) {
    size_t out_pos = 0, last_out = 0;
    auto boxes_bptr = reinterpret_cast<const Box*>(boxes);
    auto kept_mask = static_cast<size_t*>(workspace);
    memset(kept_mask, 0, cpu_kern_workspace(nr_boxes));
    for (size_t i = 0; i < nr_boxes; ++i) {
        bool supressed = false;
        auto ibox = boxes_bptr[i];
        for (size_t j = 0; j < i; ++j) {
            bool j_kept =
                    (kept_mask[j / sizeof(size_t)] >> (j % sizeof(size_t))) & 1;
            if (j_kept && box_iou(ibox, boxes_bptr[j], overlap_thresh)) {
                supressed = true;
                break;
            }
        }
        if (!supressed) {
            kept_mask[i / sizeof(size_t)] |= size_t(1) << (i % sizeof(size_t));
            last_out = i;
            out_idx[out_pos++] = i;
            if (out_pos == max_output)
                break;
        }
    }
    *out_size = out_pos;
    while (out_pos < max_output) {
        out_idx[out_pos++] = last_out;
    }
}
