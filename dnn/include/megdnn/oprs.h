#pragma once

#include "megdnn/oprs/cv.h"
#include "megdnn/oprs/general.h"
#include "megdnn/oprs/imgproc.h"
#include "megdnn/oprs/linalg.h"
#include "megdnn/oprs/nn.h"
#include "megdnn/oprs/nn_int.h"
#include "megdnn/oprs/utils.h"

template <typename Opr>
struct OprArityTrait;

template <typename Opr, int _arity_in, int _arity_out>
struct OprArityTraitTmpl {
    static constexpr int arity_in = _arity_in;
    static constexpr int arity_out = _arity_out;
    static constexpr int arity = arity_in + arity_out;
};

#define INST_ARITY(_Opr, _in, _out) \
    template <>                     \
    struct OprArityTrait<_Opr> : public OprArityTraitTmpl<_Opr, _in, _out> {};

INST_ARITY(megdnn::ConvolutionBackwardData, 2, 1);
INST_ARITY(megdnn::ConvolutionBackwardFilter, 2, 1);
INST_ARITY(megdnn::Convolution3DForward, 2, 1);
INST_ARITY(megdnn::Convolution3DBackwardData, 2, 1);
INST_ARITY(megdnn::Convolution3DBackwardFilter, 2, 1);
INST_ARITY(megdnn::LocalShareForward, 2, 1);
INST_ARITY(megdnn::LocalShareBackwardData, 2, 1);
INST_ARITY(megdnn::LocalShareBackwardFilter, 2, 1);
INST_ARITY(megdnn::Convolution, 2, 1);
INST_ARITY(megdnn::DeformableConvForward, 4, 1);
INST_ARITY(megdnn::DeformableConvBackwardFilter, 4, 1);
INST_ARITY(megdnn::BatchConvBiasForward, 4, 1);
INST_ARITY(megdnn::ConvBias, 4, 1);
INST_ARITY(megdnn::DeformableConvBackwardData, 5, 3);
INST_ARITY(megdnn::MatrixMul, 2, 1);
INST_ARITY(megdnn::BatchedMatrixMul, 2, 1);
INST_ARITY(megdnn::PoolingForward, 1, 1);
INST_ARITY(megdnn::PoolingBackward, 3, 1);

#undef INST_ARITY

// vim: syntax=cpp.doxygen
