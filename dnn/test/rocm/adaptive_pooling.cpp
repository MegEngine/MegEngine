#include "hcc_detail/hcc_defs_prologue.h"

#include "test/rocm/fixture.h"

#include "megdnn/tensor_iter.h"
#include "test/common/adaptive_pooling.h"
#include "test/common/checker.h"

#include "src/common/utils.h"
#include "test/rocm/utils.h"

#include "test/rocm/benchmarker.h"

namespace megdnn {
namespace test {

TEST_F(ROCM, ADAPTIVE_POOLING_FORWARD) {
    auto args = adaptive_pooling::get_args();
    using Format = param::AdaptivePooling::Format;
    DType dtype = dtype::Float32();
    for (auto&& arg : args) {
        auto param = arg.param;
        auto src = arg.ishape;
        auto dst = arg.oshape;
        param.format = Format::NCHW;
        Checker<AdaptivePooling> checker(handle_rocm());
        checker.set_epsilon(1e-2);
        checker.set_param(param).set_dtype(0, dtype).set_dtype(1, dtype).exec(
                TensorShapeArray{src, dst, {}});
    }
}

TEST_F(ROCM, ADAPTIVE_POOLING_BACKWARD) {
    auto args = adaptive_pooling::get_args();
    for (auto&& arg : args) {
        Checker<AdaptivePoolingBackward> checker(handle_rocm());
        TensorLayout ilayout = TensorLayout(arg.ishape, dtype::Float32());
        TensorLayout olayout = TensorLayout(arg.oshape, dtype::Float32());

        auto constraint = [this, arg](CheckerHelper::TensorValueArray& tensors_orig) {
            megdnn_assert(tensors_orig.size() == 4);
            auto opr = handle_rocm()->create_operator<AdaptivePoolingForward>();
            opr->param() = arg.param;

            auto tensors_rocm_storage = CheckerHelper::alloc_tensors(
                    handle_rocm(), {tensors_orig[0].layout, tensors_orig[1].layout}, 0);
            auto&& tensors_rocm = *tensors_rocm_storage;

            auto span = tensors_rocm[0].layout.span();
            auto dst = static_cast<dt_byte*>(tensors_rocm[0].raw_ptr()) + span.low_byte;
            auto src = static_cast<const dt_byte*>(tensors_orig[0].raw_ptr()) +
                       span.low_byte;
            megdnn_memcpy_H2D(handle_rocm(), dst, src, span.dist_byte());

            auto workspace_size = opr->get_workspace_in_bytes(
                    tensors_rocm[0].layout, tensors_rocm[1].layout);
            auto workspace_rocm = megdnn_malloc(handle_rocm(), workspace_size);
            Workspace workspace{static_cast<dt_byte*>(workspace_rocm), workspace_size};
            opr->exec(tensors_rocm[0], tensors_rocm[1], workspace);
            megdnn_free(handle_rocm(), workspace_rocm);

            span = tensors_rocm[1].layout.span();
            dst = static_cast<dt_byte*>(tensors_orig[1].raw_ptr()) + span.low_byte;
            src = static_cast<const dt_byte*>(tensors_rocm[1].raw_ptr()) +
                  span.low_byte;
            megdnn_memcpy_D2H(handle_rocm(), dst, src, span.dist_byte());
        };

        DType dtype = dtype::Float32();
        checker.set_tensors_constraint(constraint)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype)
                .set_dtype(3, dtype)
                .set_param(arg.param)
                .exec(TensorShapeArray{ilayout, olayout, olayout, ilayout});
    }
}
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
