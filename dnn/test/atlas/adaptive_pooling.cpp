#include "test/common/adaptive_pooling.h"
#include "megdnn/tensor_iter.h"
#include "test/common/checker.h"

#include "src/common/utils.h"
#include "test/atlas/fixture.h"

namespace megdnn {
namespace test {

TEST_F(ATLAS, ADAPTIVE_POOLING_FORWARD) {
    auto args = adaptive_pooling::get_args();
    using Format = param::AdaptivePooling::Format;
    std::vector<DType> dtypes{dtype::Float16(), dtype::Float32()};
    for (DType dtype : dtypes) {
        for (auto&& arg : args) {
            auto param = arg.param;
            auto src = arg.ishape;
            auto dst = arg.oshape;
            param.format = Format::NCHW;
            Checker<AdaptivePooling> checker(handle_atlas());
            if (dtype == dtype::Float16()) {
                checker.set_epsilon(1e-2);
            }
            checker.set_param(param).set_dtype(0, dtype).set_dtype(1, dtype).exec(
                    TensorShapeArray{src, dst, {}});
        }
    }
}

TEST_F(ATLAS, ADAPTIVE_POOLING_BACKWARD) {
    auto args = adaptive_pooling::get_args();
    using Format = param::AdaptivePooling::Format;
    // TODO: support FP16.
    std::vector<DType> dtypes{dtype::Float32()};
    for (DType dtype : dtypes) {
        for (auto&& arg : args) {
            Checker<AdaptivePoolingBackward> checker(handle_atlas());
            TensorLayout ilayout = TensorLayout(arg.ishape, dtype::Float32());
            TensorLayout olayout = TensorLayout(arg.oshape, dtype::Float32());

            auto constraint = [this,
                               arg](CheckerHelper::TensorValueArray& tensors_orig) {
                megdnn_assert(tensors_orig.size() == 4);
                auto opr = handle_atlas()->create_operator<AdaptivePoolingForward>();
                auto param = arg.param;
                param.format = Format::NCHW;
                opr->param() = param;

                auto tensors_acl_storage = CheckerHelper::alloc_tensors(
                        handle_atlas(),
                        {tensors_orig[0].layout, tensors_orig[1].layout}, 0);
                auto&& tensors_acl = *tensors_acl_storage;

                auto span = tensors_acl[0].layout.span();
                auto dst =
                        static_cast<dt_byte*>(tensors_acl[0].raw_ptr()) + span.low_byte;
                auto src = static_cast<const dt_byte*>(tensors_orig[0].raw_ptr()) +
                           span.low_byte;
                megdnn_memcpy_H2D(handle_atlas(), dst, src, span.dist_byte());

                auto workspace_size = opr->get_workspace_in_bytes(
                        tensors_acl[0].layout, tensors_acl[1].layout);
                auto workspace_acl = megdnn_malloc(handle_atlas(), workspace_size);
                Workspace workspace{
                        static_cast<dt_byte*>(workspace_acl), workspace_size};
                opr->exec(tensors_acl[0], tensors_acl[1], workspace);
                megdnn_free(handle_atlas(), workspace_acl);

                span = tensors_acl[1].layout.span();
                dst = static_cast<dt_byte*>(tensors_orig[1].raw_ptr()) + span.low_byte;
                src = static_cast<const dt_byte*>(tensors_acl[1].raw_ptr()) +
                      span.low_byte;
                megdnn_memcpy_D2H(handle_atlas(), dst, src, span.dist_byte());
            };
        }
    }
}
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
