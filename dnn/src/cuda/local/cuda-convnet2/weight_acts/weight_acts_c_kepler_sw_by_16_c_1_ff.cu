/**
 * \file
 * dnn/src/cuda/local/cuda-convnet2/weight_acts/weight_acts_c_kepler_sw_by_16_c_1_ff.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
/**
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * --------------------------------------------------------------------------
 * * This file has been modified by Megvii ("Megvii Modifications").
 * * All Megvii Modifications are Copyright (C) 2014-2021 Megvii Inc. All rights
 * reserved.
 * --------------------------------------------------------------------------
 */
#include "wet_act_c_kepler_sw.cuh"

namespace megdnn {
namespace cuda {

WET_ACT_C_KEPLER_SW_HEAD<16, 16, 2, 16, 1, 32, 1, false, false>(C_KEP_SW_PARAM);
// WET_ACT_C_KEPLER_SW_HEAD< 16, 16, 2, 16, 1, 32, 1, false, true > (C_KEP_SW_PARAM);
// WET_ACT_C_KEPLER_SW_HEAD< 16, 16, 2, 16, 1, 32, 1, true, false > (C_KEP_SW_PARAM);
// WET_ACT_C_KEPLER_SW_HEAD< 16, 16, 2, 16, 1, 32, 1, true, true > (C_KEP_SW_PARAM);

}  // namespace cuda
}  // namespace megdnn
