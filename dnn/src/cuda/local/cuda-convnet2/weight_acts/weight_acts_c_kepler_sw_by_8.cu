#include "wet_act_c_kepler_sw.cuh"

namespace megdnn {
namespace cuda {

WET_ACT_C_KEPLER_SW_HEAD<8, 16, 2, 2, 2, 16, 1, false, false>(C_KEP_SW_PARAM);
WET_ACT_C_KEPLER_SW_HEAD<8, 16, 2, 2, 2, 16, 1, false, true>(C_KEP_SW_PARAM);
// WET_ACT_C_KEPLER_SW_HEAD< 8, 16, 2, 2, 2, 16, 1, true, false > (C_KEP_SW_PARAM);
// WET_ACT_C_KEPLER_SW_HEAD< 8, 16, 2, 2, 2, 16, 1, true, true > (C_KEP_SW_PARAM);
WET_ACT_C_KEPLER_SW_HEAD<8, 16, 2, 2, 2, 16, 2, false, false>(C_KEP_SW_PARAM);
WET_ACT_C_KEPLER_SW_HEAD<8, 16, 2, 2, 2, 16, 2, false, true>(C_KEP_SW_PARAM);
// WET_ACT_C_KEPLER_SW_HEAD< 8, 16, 2, 2, 2, 16, 2, true, false > (C_KEP_SW_PARAM);
// WET_ACT_C_KEPLER_SW_HEAD< 8, 16, 2, 2, 2, 16, 2, true, true > (C_KEP_SW_PARAM);
WET_ACT_C_KEPLER_SW_HEAD<8, 16, 2, 2, 2, 16, 3, false, false>(C_KEP_SW_PARAM);
WET_ACT_C_KEPLER_SW_HEAD<8, 16, 2, 2, 2, 16, 3, false, true>(C_KEP_SW_PARAM);
// WET_ACT_C_KEPLER_SW_HEAD< 8, 16, 2, 2, 2, 16, 3, true, false > (C_KEP_SW_PARAM);
// WET_ACT_C_KEPLER_SW_HEAD< 8, 16, 2, 2, 2, 16, 3, true, true > (C_KEP_SW_PARAM);

}  // namespace cuda
}  // namespace megdnn
