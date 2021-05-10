/**
 * \file dnn/test/naive/images2neibs.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/naive/fixture.h"

#include "megdnn/oprs/nn.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

TEST_F(NAIVE, IMAGES2NEIBS_FORWARD) {
    Checker<Images2Neibs> checker(handle(), /* check_dispatch */false);
    
    Images2Neibs::Param param(0,0,1,1,1,1,2,2);
    checker.set_param(param).exect(
            Testcase{TensorValue({1, 1, 3, 3}, dtype::Uint8(),
                               {0,1,2,
                               3,4,5,
                               6,7,8}), {}}, 
            Testcase{{},
            TensorValue({1, 1, 2, 2, 2, 2}, dtype::Uint8(), 
                        {0,1,3,4,
                        1,2,4,5,
                        3,4,6,7,
                        4,5,7,8})});
    
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 2;
    param.stride_w = 2;
    param.dilate_h = 2;
    param.dilate_w = 2;
    param.window_h = 3;
    param.window_w = 3;
    checker.set_param(param).exect(
          Testcase{TensorValue({1, 1, 6, 7}, dtype::Uint8(),
                               {0,1,2,3,4,5,6,
                               7,8,9,10,11,12,13,
                               14,15,16,17,18,19,20,
                               21,22,23,24,25,26,27,
                               28,29,30,31,32,33,34,
                               35,36,37,38,39,40,41}), {}}, 
          Testcase{{},
          TensorValue({1, 1, 2, 3, 3, 3}, dtype::Uint8(),
                        {0,0,0,0,8,10,0,22,24,
                        0,0,0,8,10,12,22,24,26,
                        0,0,0,10,12,0,24,26,0,
                        0,8,10,0,22,24,0,36,38,
                        8,10,12,22,24,26,36,38,40,
                        10,12,0,24,26,0,38,40,0})});
}
