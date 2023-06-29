#include "test/common/resize.h"
#include "megdnn/oprs/cv.h"
#include "test/common/checker.h"
#include "test/naive/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(NAIVE, RESIZE_NCHW4) {
    Checker<Resize> checker(handle());

    auto args = resize::get_nchw4_args();
    auto convert_true_format = [](const TensorLayout& layout) {
        return layout.reshape({layout[0], layout[1] / 4, layout[2], layout[3], 4})
                .dimshuffle({0, 1, 4, 2, 3});
    };

    for (auto&& arg : args) {
        auto extra_impl = [this, param = arg.param,
                           convert_true_format](const TensorNDArray& tensors) {
            auto resize = handle()->create_operator<Resize>();
            resize->param().imode = param.imode;
            resize->param().format = Resize::Param::Format::NCHW;

            TensorNDArray nchw_tensors;
            for (size_t i = 0; i < tensors.size(); ++i) {
                auto layout = tensors[i].layout;
                layout = layout.reshape(
                        {layout[0], layout[1] * 4, layout[2], layout[3]});
                layout.dtype = dtype::Int8();
                nchw_tensors.emplace_back(malloc(layout.span().dist_byte()), layout);
            }
            TensorNDArray nchw4_tensors;
            for (size_t i = 0; i < tensors.size(); ++i) {
                auto layout = convert_true_format(nchw_tensors[i].layout);
                nchw4_tensors.emplace_back(tensors[i].raw_ptr(), std::move(layout));
            }

            auto relayout = handle()->create_operator<RelayoutForward>();
            relayout->exec(nchw4_tensors[0], nchw_tensors[0]);

            auto workspace_size = resize->get_workspace_in_bytes(
                    nchw_tensors[0].layout, nchw_tensors[1].layout);
            dt_byte* workspace_ptr = static_cast<dt_byte*>(malloc(workspace_size));
            Workspace workspace{workspace_ptr, workspace_size};

            resize->exec(nchw_tensors[0], nchw_tensors[1], workspace);

            relayout->exec(nchw_tensors[1], nchw4_tensors[1]);

            free(workspace_ptr);
            for (auto&& tensor : nchw_tensors) {
                free(tensor.raw_ptr());
            }
        };
        checker.set_extra_opr_impl(extra_impl);
        checker.set_param(arg.param)
                .set_dtype(0, dtype::QuantizedS8(0.1f))
                .set_dtype(1, dtype::QuantizedS8(0.1f))
                .set_epsilon(1 + 1e-3)
                .execs({arg.src, arg.dst});
    }
}

TEST_F(NAIVE, RESIZE3D_NCDHW) {
    using IMode = param::Resize3D::InterpolationMode;
    using Format = param::Resize3D::Format;
    auto ac_param = param::Resize3D{IMode::LINEAR, Format::NCDHW, true};
    auto nac_param = param::Resize3D{IMode::LINEAR, Format::NCDHW, false};

    Checker<Resize3D> checker(handle());
    checker.set_param(nac_param).exect(
            Testcase{
                    TensorValue(
                            {1, 1, 2, 2, 2}, dtype::Float32(),
                            {0., 1., 2., 3., 4., 5., 6., 7.}),
                    {}},
            Testcase{
                    {},
                    TensorValue(
                            {1, 1, 4, 4, 4}, dtype::Float32(),
                            {0.,   0.25, 0.75, 1.,   0.5,  0.75, 1.25, 1.5,  1.5,  1.75,
                             2.25, 2.5,  2.,   2.25, 2.75, 3.,   1.,   1.25, 1.75, 2.,
                             1.5,  1.75, 2.25, 2.5,  2.5,  2.75, 3.25, 3.5,  3.,   3.25,
                             3.75, 4.,   3.,   3.25, 3.75, 4.,   3.5,  3.75, 4.25, 4.5,
                             4.5,  4.75, 5.25, 5.5,  5.,   5.25, 5.75, 6.,   4.,   4.25,
                             4.75, 5.,   4.5,  4.75, 5.25, 5.5,  5.5,  5.75, 6.25, 6.5,
                             6.,   6.25, 6.75, 7.})});

    checker.set_param(ac_param).exect(
            Testcase{
                    TensorValue(
                            {1, 1, 2, 2, 2}, dtype::Float32(),
                            {0., 1., 2., 3., 4., 5., 6., 7.}),
                    {}},
            Testcase{
                    {},
                    TensorValue(
                            {1, 1, 4, 4, 4}, dtype::Float32(),
                            {0.,        0.3333333, 0.6666667, 1.,        0.6666667,
                             1.,        1.3333333, 1.6666666, 1.3333334, 1.6666667,
                             1.9999999, 2.3333333, 2.,        2.3333333, 2.6666665,
                             3.,        1.3333334, 1.6666666, 2.0000002, 2.3333335,
                             2.,        2.333333,  2.6666667, 2.9999998, 2.6666665,
                             3.,        3.3333333, 3.6666665, 3.3333333, 3.6666665,
                             4.,        4.3333335, 2.6666667, 3.,        3.3333337,
                             3.6666667, 3.3333335, 3.6666663, 4.,        4.333333,
                             3.9999998, 4.333333,  4.6666665, 5.,        4.6666665,
                             5.,        5.3333335, 5.666667,  4.,        4.333333,
                             4.666667,  5.,        4.6666665, 4.9999995, 5.3333335,
                             5.6666665, 5.333333,  5.6666665, 6.,        6.3333335,
                             6.,        6.333333,  6.666667,  7.})});

    checker.set_param(nac_param).exect(
            Testcase{
                    TensorValue(
                            {1, 1, 2, 2, 2}, dtype::Float16(),
                            {0., 1., 2., 3., 4., 5., 6., 7.}),
                    {}},
            Testcase{
                    {},
                    TensorValue(
                            {1, 1, 4, 4, 4}, dtype::Float16(),
                            {0.,   0.25, 0.75, 1.,   0.5,  0.75, 1.25, 1.5,  1.5,  1.75,
                             2.25, 2.5,  2.,   2.25, 2.75, 3.,   1.,   1.25, 1.75, 2.,
                             1.5,  1.75, 2.25, 2.5,  2.5,  2.75, 3.25, 3.5,  3.,   3.25,
                             3.75, 4.,   3.,   3.25, 3.75, 4.,   3.5,  3.75, 4.25, 4.5,
                             4.5,  4.75, 5.25, 5.5,  5.,   5.25, 5.75, 6.,   4.,   4.25,
                             4.75, 5.,   4.5,  4.75, 5.25, 5.5,  5.5,  5.75, 6.25, 6.5,
                             6.,   6.25, 6.75, 7.})});

    checker.set_param(ac_param).exect(
            Testcase{
                    TensorValue(
                            {1, 1, 2, 2, 2}, dtype::Float16(),
                            {0., 1., 2., 3., 4., 5., 6., 7.}),
                    {}},
            Testcase{
                    {},
                    TensorValue(
                            {1, 1, 4, 4, 4}, dtype::Float16(),
                            {0.,        0.3333333, 0.6666667, 1.,        0.6666667,
                             1.,        1.3333333, 1.6666666, 1.3333334, 1.6666667,
                             1.9999999, 2.3333333, 2.,        2.3333333, 2.6666665,
                             3.,        1.3333334, 1.6666666, 2.0000002, 2.3333335,
                             2.,        2.333333,  2.6666667, 2.9999998, 2.6666665,
                             3.,        3.3333333, 3.6666665, 3.3333333, 3.6666665,
                             4.,        4.3333335, 2.6666667, 3.,        3.3333337,
                             3.6666667, 3.3333335, 3.6666663, 4.,        4.333333,
                             3.9999998, 4.333333,  4.6666665, 5.,        4.6666665,
                             5.,        5.3333335, 5.666667,  4.,        4.333333,
                             4.666667,  5.,        4.6666665, 4.9999995, 5.3333335,
                             5.6666665, 5.333333,  5.6666665, 6.,        6.3333335,
                             6.,        6.333333,  6.666667,  7.})});
}
