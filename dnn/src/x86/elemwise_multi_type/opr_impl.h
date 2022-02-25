#pragma once

#include "src/fallback/elemwise_multi_type/opr_impl.h"

namespace megdnn {
namespace x86 {

class ElemwiseMultiTypeImpl : public fallback::ElemwiseMultiTypeImpl {
protected:
    void on_quantized_mode(
            const ElemwiseOpParamN<1>& param, const TensorND& dst,
            Elemwise::Mode mode) override;

    void on_quantized_mode(
            const ElemwiseOpParamN<2>& param, const TensorND& dst,
            Elemwise::Mode mode) override;

    void on_quantized_mode(
            const ElemwiseOpParamN<3>& param, const TensorND& dst,
            Elemwise::Mode mode) override;

public:
    using fallback::ElemwiseMultiTypeImpl::ElemwiseMultiTypeImpl;
};

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
