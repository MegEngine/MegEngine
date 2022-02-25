#include "img_act_medium_color.cuh"

namespace megdnn {
namespace cuda {

IMG_MED_COLOR_K(false, false, false)
// IMG_MED_COLOR_K(false, false, true)
IMG_MED_COLOR_K(false, true, false)
// IMG_MED_COLOR_K(false, true, true)

// IMG_MED_COLOR_K(true, false, false)
// IMG_MED_COLOR_K(true, false, true)
// IMG_MED_COLOR_K(true, true, false)
// IMG_MED_COLOR_K(true, true, true)

}  // namespace cuda
}  // namespace megdnn
