/**
 * \file dnn/src/fallback/conv_bias/conv1x1/Conv1x1_strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/fallback/conv_bias/conv1x1/conv1x1_utils.h"
#include "src/fallback/conv_bias/conv1x1/conv1x1_strategy.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fallback_conv1x1_factory_strategy)

namespace megdnn {
namespace fallback {
namespace conv1x1 {
namespace {
//! NOTE: must keep consistence with can_make_conv1x1_strategy when you modify
//! this function
std::unique_ptr<Conv1x1StrategyBase> create_conv1x1_strategy(
        const ConvBiasImpl::NCBKernSizeParam& param,
        MatrixMulImpl::AlgoBase::PackMode pack_mode,
        param::ConvBias::Format format) {
    size_t pack_c_size = pack_size(format);
#define cb1(_packmode, _dt, _post_ctype, _postprocess_mode, _midout_tag)     \
    MIDOUT_BEGIN(megdnn_fallback_conv1x1_factory_strategy,                   \
                 midout_iv(_midout_tag)) {                                   \
        if (param.filter_type.enumv() == DTypeTrait<_dt>::enumv) {           \
            return std::make_unique<                                         \
                    Conv1x1Strategy<_dt, _dt, _dt, _post_ctype, _post_ctype, \
                                    _postprocess_mode, _packmode>>(          \
                    pack_c_size);                                            \
        }                                                                    \
    }                                                                        \
    MIDOUT_END()

#define cb2(_packmode, _i_src_type, _i_bias_type, _i_dst_type, _src_ctype, \
            _bias_ctype, _dst_ctype, _postprocess_mode, _midout_tag)       \
    MIDOUT_BEGIN(megdnn_fallback_conv1x1_factory_strategy,                 \
                 midout_iv(_midout_tag)) {                                 \
        if (param.filter_type.enumv() == param.src_type.enumv() &&         \
            param.src_type.enumv() == DTypeTrait<_i_src_type>::enumv &&    \
            param.dst_type.enumv() == DTypeTrait<_i_dst_type>::enumv) {    \
            return std::make_unique<                                       \
                    Conv1x1Strategy<_src_ctype, _bias_ctype, _dst_ctype,   \
                                    DTypeTrait<_i_bias_type>::ctype,       \
                                    DTypeTrait<_i_dst_type>::ctype,        \
                                    _postprocess_mode, _packmode>>(        \
                    pack_c_size);                                          \
        }                                                                  \
    }                                                                      \
    MIDOUT_END()
#define cb3(_packmode, _i_src_type, _i_bias_type, _i_dst_type, _src_ctype,   \
            _bias_ctype, _dst_ctype, _postprocess_mode, _midout_tag)         \
    MIDOUT_BEGIN(megdnn_fallback_conv1x1_factory_strategy,                   \
                 midout_iv(_midout_tag)) {                                   \
        if (param.filter_type.enumv() == param.src_type.enumv() &&           \
            param.src_type.enumv() == DTypeTrait<_i_src_type>::enumv &&      \
            param.dst_type.enumv() == DTypeTrait<_i_dst_type>::enumv) {      \
            return std::make_unique<Conv1x1Strategy<                         \
                    _src_ctype, _bias_ctype, _dst_ctype, _bias_ctype,        \
                    _dst_ctype, _postprocess_mode, _packmode>>(pack_c_size); \
        }                                                                    \
    }                                                                        \
    MIDOUT_END()

    switch (pack_mode) {
        case MatrixMulImpl::AlgoBase::PackMode::DEFAULT:
            cb1(MatrixMulImpl::AlgoBase::PackMode::DEFAULT, dt_float32,
                dt_float32, PostprocessMode::FLOAT, "Default::FLOAT"_hash);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            cb1(MatrixMulImpl::AlgoBase::PackMode::DEFAULT, dt_float16, __fp16,
                PostprocessMode::FLOAT, "Default::FLOAT16_FP16"_hash);
#else
#if !MEGDNN_DISABLE_FLOAT16
            cb1(MatrixMulImpl::AlgoBase::PackMode::DEFAULT, dt_float16,
                dt_float16, PostprocessMode::NO_PROCESS,
                "Default::FLOAT16_FLOAT16"_hash);
#endif
#endif
            cb3(MatrixMulImpl::AlgoBase::PackMode::DEFAULT, dt_int8, dt_int32,
                dt_int32, dt_int8, dt_int32, dt_int32,
                PostprocessMode::ADD_BIAS, "Default::INT8x8x32_INT32"_hash);
            cb3(MatrixMulImpl::AlgoBase::PackMode::DEFAULT, dt_int8, dt_int16,
                dt_int16, dt_int8, dt_int16, dt_int16,
                PostprocessMode::ADD_BIAS, "Default::INT8x8x16_INT16"_hash);
#if MEGDNN_AARCH64 || MEGDNN_ARMV7
            cb3(MatrixMulImpl::AlgoBase::PackMode::DEFAULT,
                dtype::Quantized8Asymm, dtype::QuantizedS32,
                dtype::QuantizedS32, dt_uint8, dt_int32, dt_int32,
                PostprocessMode::ADD_BIAS,
                "Default::QUINT8x8x32_QINT32"_hash);
            cb2(MatrixMulImpl::AlgoBase::PackMode::DEFAULT,
                dtype::Quantized8Asymm, dtype::QuantizedS32,
                dtype::Quantized8Asymm, dt_uint8, dt_int32, dt_uint8,
                PostprocessMode::QUANTIZED, "Default::QUINT8x8x32_QUINT8"_hash);
#endif
            cb3(MatrixMulImpl::AlgoBase::PackMode::DEFAULT, dtype::QuantizedS8,
                dtype::QuantizedS32, dtype::QuantizedS32, dt_int8, dt_int32,
                dt_int32, PostprocessMode::ADD_BIAS,
                "Default::QINT8x8x32_QINT32"_hash);
            cb2(MatrixMulImpl::AlgoBase::PackMode::DEFAULT, dtype::QuantizedS8,
                dtype::QuantizedS32, dtype::QuantizedS8, dt_int8, dt_int32,
                dt_int8, PostprocessMode::QUANTIZED,
                "Default::QINT8x8x32_QINT8"_hash);
            break;

        case MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA:
            cb1(MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA, dt_float32,
                dt_float32, PostprocessMode::FLOAT, "OnlyPackA::FLOAT"_hash);
            break;

        case MatrixMulImpl::AlgoBase::PackMode::NO_PACK:
            cb1(MatrixMulImpl::AlgoBase::PackMode::NO_PACK, dt_float32,
                dt_float32, PostprocessMode::FLOAT, "NoPack::FLOAT"_hash);

            cb3(MatrixMulImpl::AlgoBase::PackMode::NO_PACK, dt_int8, dt_int16,
                dt_int16, dt_int8, dt_int16, dt_int16,
                PostprocessMode::ADD_BIAS, "NoPack::INT8x8x16_INT16"_hash);

            cb3(MatrixMulImpl::AlgoBase::PackMode::NO_PACK, dt_int8, dt_int32,
                dt_int32, dt_int8, dt_int32, dt_int32,
                PostprocessMode::ADD_BIAS, "NoPack::INT8x8x32_INT32"_hash);

            cb3(MatrixMulImpl::AlgoBase::PackMode::NO_PACK, dtype::QuantizedS8,
                dtype::QuantizedS32, dtype::QuantizedS32, dt_int8, dt_int32,
                dt_int32, PostprocessMode::ADD_BIAS,
                "NoPack::QINT8x8x32_QINT32"_hash);
            break;

        default:
            megdnn_throw("Invalid Pack Mode");
            break;
    }
#undef cb1
#undef cb2
#undef cb3
    megdnn_throw("Invalid Data Type");
    return nullptr;
}
}  // anonymous namespace

Conv1x1StrategyBase* Conv1x1Factory::make_conv1x1_strategy(
        const ConvBiasImpl::NCBKernSizeParam& param,
        MatrixMulImpl::AlgoBase::PackMode pack_mode,
        param::ConvBias::Format format) {
    static utils::StrategyDelegationStorage<Conv1x1StrategyBase> storage;
    return storage.get(param, pack_mode, format, create_conv1x1_strategy);
}

bool Conv1x1Factory::can_make_conv1x1_strategy(
        const ConvBiasImpl::NCBKernSizeParam& param,
        MatrixMulImpl::AlgoBase::PackMode pack_mode, param::ConvBias::Format) {
    bool ok_default_cb1 =
            param.src_type.enumv() == DTypeTrait<dt_float32>::enumv;
    bool ok_default_cb2 =
            param.filter_type.enumv() == param.src_type.enumv() &&
            ((param.src_type.enumv() == DTypeTrait<dt_int8>::enumv &&
              param.dst_type.enumv() == DTypeTrait<dt_int32>::enumv) ||
             (param.src_type.enumv() == DTypeTrait<dt_int8>::enumv &&
              param.dst_type.enumv() == DTypeTrait<dt_int16>::enumv) ||
             (param.src_type.enumv() == DTypeTrait<dtype::QuantizedS8>::enumv &&
              param.dst_type.enumv() ==
                      DTypeTrait<dtype::QuantizedS32>::enumv) ||
             (param.src_type.enumv() == DTypeTrait<dtype::QuantizedS8>::enumv &&
              param.dst_type.enumv() == DTypeTrait<dtype::QuantizedS8>::enumv));
    bool ok_default_cb1_fp16 = false;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC || !MEGDNN_DISABLE_FLOAT16
    ok_default_cb1_fp16 =
            param.src_type.enumv() == DTypeTrait<dt_float16>::enumv;
#endif
    bool ok_default_cb2_arm = false;
#if MEGDNN_AARCH64 || MEGDNN_ARMV7
    ok_default_cb2_arm = param.filter_type.enumv() == param.src_type.enumv() &&
                         ((param.src_type.enumv() ==
                                   DTypeTrait<dtype::Quantized8Asymm>::enumv &&
                           param.dst_type.enumv() ==
                                   DTypeTrait<dtype::QuantizedS32>::enumv) ||
                          (param.src_type.enumv() ==
                                   DTypeTrait<dtype::Quantized8Asymm>::enumv &&
                           param.dst_type.enumv() ==
                                   DTypeTrait<dtype::Quantized8Asymm>::enumv));
#endif

    bool ok_only_packa_cb1 =
            param.src_type.enumv() == DTypeTrait<dt_float32>::enumv;
    bool ok_no_pack_cb1 =
            param.src_type.enumv() == DTypeTrait<dt_float32>::enumv;
    bool ok_no_pack_cb2 =
            param.filter_type.enumv() == param.src_type.enumv() &&
            ((param.src_type.enumv() == DTypeTrait<dt_int8>::enumv &&
              param.dst_type.enumv() == DTypeTrait<dt_int16>::enumv) ||
             (param.src_type.enumv() == DTypeTrait<dt_int8>::enumv &&
              param.dst_type.enumv() == DTypeTrait<dt_int32>::enumv) ||
             (param.src_type.enumv() == DTypeTrait<dtype::QuantizedS8>::enumv &&
              param.dst_type.enumv() ==
                      DTypeTrait<dtype::QuantizedS32>::enumv));
    switch (pack_mode) {
        case MatrixMulImpl::AlgoBase::PackMode::DEFAULT:
            return ok_default_cb1 || ok_default_cb2 || ok_default_cb1_fp16 ||
                   ok_default_cb2_arm;
            break;
        case MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA:
            return ok_only_packa_cb1;
            break;
        case MatrixMulImpl::AlgoBase::PackMode::NO_PACK:
            return ok_no_pack_cb1 || ok_no_pack_cb2;
            break;
        default:
            return false;
    }
}

}  // namespace conv1x1
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
