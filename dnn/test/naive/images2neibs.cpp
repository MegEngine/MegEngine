#include "test/naive/fixture.h"

#include "megdnn/oprs/nn.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

TEST_F(NAIVE, IMAGES2NEIBS_FORWARD) {
    Checker<Images2Neibs> checker(handle(), /* check_dispatch */ false);

    Images2Neibs::Param param(0, 0, 1, 1, 1, 1, 2, 2);
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 1, 3, 3}, dtype::Uint8(), {0, 1, 2, 3, 4, 5, 6, 7, 8}),
                    {}},
            Testcase{
                    {},
                    TensorValue(
                            {1, 1, 2, 2, 2, 2}, dtype::Uint8(),
                            {0, 1, 3, 4, 1, 2, 4, 5, 3, 4, 6, 7, 4, 5, 7, 8})});

    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 2;
    param.stride_w = 2;
    param.dilate_h = 2;
    param.dilate_w = 2;
    param.window_h = 3;
    param.window_w = 3;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 1, 6, 7}, dtype::Uint8(),
                            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                             14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                             28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41}),
                    {}},
            Testcase{
                    {},
                    TensorValue(
                            {1, 1, 2, 3, 3, 3}, dtype::Uint8(),
                            {0,  0,  0,  0,  8,  10, 0,  22, 24, 0,  0,  0,  8,  10,
                             12, 22, 24, 26, 0,  0,  0,  10, 12, 0,  24, 26, 0,  0,
                             8,  10, 0,  22, 24, 0,  36, 38, 8,  10, 12, 22, 24, 26,
                             36, 38, 40, 10, 12, 0,  24, 26, 0,  38, 40, 0})});
}

TEST_F(NAIVE, IMAGES2NEIBS_FORWARD_CD4) {
    Checker<Images2Neibs> checker(handle(), /* check_dispatch */ false);

    Images2Neibs::Param param(0, 0, 1, 1, 1, 1, 2, 2);

    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 3, 1, 3, 4}, dtype::Uint8(),
                            {0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0,
                             0, 0, 5, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0}),
                    {}},
            Testcase{
                    {},
                    TensorValue(
                            {1, 2, 1, 2, 2, 2, 4}, dtype::Uint8(),
                            {0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0,
                             1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0,
                             3, 0, 0, 0, 4, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0,
                             4, 0, 0, 0, 5, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0})});

    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 2;
    param.stride_w = 2;
    param.dilate_h = 2;
    param.dilate_w = 2;
    param.window_h = 3;
    param.window_w = 3;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 6, 1, 7, 4}, dtype::Uint8(),
                            {0,  0, 0, 0, 1,  0, 0, 0, 2,  0, 0, 0, 3,  0, 0, 0,
                             4,  0, 0, 0, 5,  0, 0, 0, 6,  0, 0, 0, 7,  0, 0, 0,
                             8,  0, 0, 0, 9,  0, 0, 0, 10, 0, 0, 0, 11, 0, 0, 0,
                             12, 0, 0, 0, 13, 0, 0, 0, 14, 0, 0, 0, 15, 0, 0, 0,
                             16, 0, 0, 0, 17, 0, 0, 0, 18, 0, 0, 0, 19, 0, 0, 0,
                             20, 0, 0, 0, 21, 0, 0, 0, 22, 0, 0, 0, 23, 0, 0, 0,
                             24, 0, 0, 0, 25, 0, 0, 0, 26, 0, 0, 0, 27, 0, 0, 0,
                             28, 0, 0, 0, 29, 0, 0, 0, 30, 0, 0, 0, 31, 0, 0, 0,
                             32, 0, 0, 0, 33, 0, 0, 0, 34, 0, 0, 0, 35, 0, 0, 0,
                             36, 0, 0, 0, 37, 0, 0, 0, 38, 0, 0, 0, 39, 0, 0, 0,
                             40, 0, 0, 0, 41, 0, 0, 0}),
                    {}},
            Testcase{
                    {},
                    TensorValue(
                            {1, 2, 1, 3, 3, 3, 4}, dtype::Uint8(),
                            {0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0,
                             8,  0, 0, 0, 10, 0, 0, 0, 0,  0, 0, 0, 22, 0, 0, 0,
                             24, 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0,
                             8,  0, 0, 0, 10, 0, 0, 0, 12, 0, 0, 0, 22, 0, 0, 0,
                             24, 0, 0, 0, 26, 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0,
                             0,  0, 0, 0, 10, 0, 0, 0, 12, 0, 0, 0, 0,  0, 0, 0,
                             24, 0, 0, 0, 26, 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0,
                             8,  0, 0, 0, 10, 0, 0, 0, 0,  0, 0, 0, 22, 0, 0, 0,
                             24, 0, 0, 0, 0,  0, 0, 0, 36, 0, 0, 0, 38, 0, 0, 0,
                             8,  0, 0, 0, 10, 0, 0, 0, 12, 0, 0, 0, 22, 0, 0, 0,
                             24, 0, 0, 0, 26, 0, 0, 0, 36, 0, 0, 0, 38, 0, 0, 0,
                             40, 0, 0, 0, 10, 0, 0, 0, 12, 0, 0, 0, 0,  0, 0, 0,
                             24, 0, 0, 0, 26, 0, 0, 0, 0,  0, 0, 0, 38, 0, 0, 0,
                             40, 0, 0, 0, 0,  0, 0, 0})});
}
