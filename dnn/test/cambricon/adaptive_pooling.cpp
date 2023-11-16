#include "test/common/adaptive_pooling.h"
#include "megdnn/tensor_iter.h"
#include "test/common/checker.h"

#include "src/common/utils.h"
#include "test/cambricon/fixture.h"

namespace megdnn {
namespace test {

TEST_F(CAMBRICON, ADAPTIVE_POOLING_FORWARD) {
    auto args = adaptive_pooling::get_args_nhwc();
    using Format = param::AdaptivePooling::Format;
    std::vector<DType> dtypes{dtype::Float16(), dtype::Float32()};
    for (DType dtype : dtypes) {
        for (auto&& arg : args) {
            auto param = arg.param;
            auto src = arg.ishape;
            auto dst = arg.oshape;
            param.format = Format::NHWC;
            Checker<AdaptivePooling> checker(handle_cambricon());
            if (dtype == dtype::Float16()) {
                checker.set_epsilon(1e-2);
            }
            checker.set_param(param).set_dtype(0, dtype).set_dtype(1, dtype).exec(
                    TensorShapeArray{src, dst, {}});
        }
    }
}

TEST_F(CAMBRICON, ADAPTIVE_POOLING_BACKWARD) {
    auto args = adaptive_pooling::get_args_nhwc();
    using Format = param::AdaptivePooling::Format;
    // TODO: support FP16.
    std::vector<DType> dtypes{dtype::Float32()};
    for (DType dtype : dtypes) {
        for (auto&& arg : args) {
            Checker<AdaptivePoolingBackward> checker(handle_cambricon());
            TensorLayout ilayout = TensorLayout(arg.ishape, dtype::Float32());
            TensorLayout olayout = TensorLayout(arg.oshape, dtype::Float32());

            auto constraint = [this,
                               arg](CheckerHelper::TensorValueArray& tensors_orig) {
                megdnn_assert(tensors_orig.size() == 4);
                auto opr =
                        handle_cambricon()->create_operator<AdaptivePoolingForward>();
                auto param = arg.param;
                param.format = Format::NHWC;
                opr->param() = param;

                auto tensors_cnnl_storage = CheckerHelper::alloc_tensors(
                        handle_cambricon(),
                        {tensors_orig[0].layout, tensors_orig[1].layout}, 0);
                auto&& tensors_cnnl = *tensors_cnnl_storage;

                auto span = tensors_cnnl[0].layout.span();
                auto dst = static_cast<dt_byte*>(tensors_cnnl[0].raw_ptr()) +
                           span.low_byte;
                auto src = static_cast<const dt_byte*>(tensors_orig[0].raw_ptr()) +
                           span.low_byte;
                megdnn_memcpy_H2D(handle_cambricon(), dst, src, span.dist_byte());

                auto workspace_size = opr->get_workspace_in_bytes(
                        tensors_cnnl[0].layout, tensors_cnnl[1].layout);
                auto workspace_cnnl = megdnn_malloc(handle_cambricon(), workspace_size);
                Workspace workspace{
                        static_cast<dt_byte*>(workspace_cnnl), workspace_size};
                opr->exec(tensors_cnnl[0], tensors_cnnl[1], workspace);
                megdnn_free(handle_cambricon(), workspace_cnnl);

                span = tensors_cnnl[1].layout.span();
                dst = static_cast<dt_byte*>(tensors_orig[1].raw_ptr()) + span.low_byte;
                src = static_cast<const dt_byte*>(tensors_cnnl[1].raw_ptr()) +
                      span.low_byte;
                megdnn_memcpy_D2H(handle_cambricon(), dst, src, span.dist_byte());
            };

            auto param = arg.param;
            param.format = Format::NHWC;
            if (dtype == dtype::Float16()) {
                checker.set_epsilon(1e-2);
            }
            checker.set_tensors_constraint(constraint)
                    .set_dtype(0, dtype)
                    .set_dtype(1, dtype)
                    .set_dtype(2, dtype)
                    .set_dtype(3, dtype)
                    .set_param(param)
                    .exec(TensorShapeArray{ilayout, olayout, olayout, ilayout});
        }
    }
}
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
