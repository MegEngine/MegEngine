/**
 * \file dnn/src/fallback/conv_bias/conv1x1/Conv1x1_strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <unordered_map>
#include "src/fallback/conv_bias/conv1x1/conv1x1_strategy.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fallback_conv1x1_factory_strategy)

namespace megdnn {
namespace fallback {
namespace conv1x1 {

namespace {

size_t get_format_pack_size(param::ConvBias::Format format) {
    switch(format){
        case param::ConvBias::Format::NCHW44:
        case param::ConvBias::Format::NCHW4:
            return 4_z;
        case param::ConvBias::Format::NCHW88:
            return 8_z;
        case param::ConvBias::Format::NCHW:
            return 1_z;
        default:
            megdnn_throw("unknow pack size of the format");
    }
}

struct StrategyHashParam {
    ConvBiasImpl::NCBKernSizeParam param;
    param::ConvBias::Format format;
    MatrixMulImpl::AlgoBase::PackMode packmode;
};

struct StrategyHashParamHash {
    std::size_t operator()(const StrategyHashParam& sparam) const {
        constexpr size_t base = 1;  //! avoid hashkey is zero
        std::size_t result =
                static_cast<std::size_t>(sparam.param.src_type.enumv()) + base;
        result = result ^
                 ((static_cast<std::size_t>(sparam.param.dst_type.enumv()) +
                   base)
                  << 3);
        result = result ^
                 ((static_cast<std::size_t>(sparam.param.filter_type.enumv()) +
                   base)
                  << 6);
        result = result ^
                 ((static_cast<std::size_t>(sparam.param.bias_type.enumv()) +
                   base)
                  << 9);
        result = result ^
                 ((static_cast<std::size_t>(sparam.format) + base) << 12);
        result = result ^
                 ((static_cast<std::size_t>(sparam.packmode) + base) << 15);
        return result;
    };
};

struct StrategyHashParamEqual {
    bool operator()(const StrategyHashParam& param1,
                    const StrategyHashParam& param2) const {
        bool flags = true;
        flags = param1.param.src_type == param2.param.src_type && flags;
        flags = param1.param.filter_type == param2.param.filter_type && flags;
        flags = param1.param.bias_type == param2.param.bias_type && flags;
        flags = param1.param.dst_type == param2.param.dst_type && flags;
        flags = param1.format == param2.format && flags;
        flags = param1.packmode == param2.packmode && flags;
        return flags;
    };
};

std::unique_ptr<Conv1x1StrategyBase> create_conv1x1_strategy(
        const ConvBiasImpl::NCBKernSizeParam& param,
        MatrixMulImpl::AlgoBase::PackMode pack_mode,
        param::ConvBias::Format format) {
    size_t pack_size = get_format_pack_size(format);
#define cb1(_packmode, _dt, _post_ctype, _postprocess_mode, _midout_tag)       \
    MIDOUT_BEGIN(megdnn_fallback_conv1x1_factory_strategy,                     \
                 midout_iv(_midout_tag)) {                                     \
        if (param.filter_type.enumv() == DTypeTrait<_dt>::enumv) {             \
            return std::make_unique<                                           \
                    Conv1x1Strategy<_dt, _dt, _dt, _post_ctype, _post_ctype,   \
                                    _postprocess_mode, _packmode>>(pack_size); \
        }                                                                      \
    }                                                                          \
    MIDOUT_END()

#define cb2(_packmode, _i_src_type, _i_bias_type, _i_dst_type, _src_ctype,     \
            _bias_ctype, _dst_ctype, _postprocess_mode, _midout_tag)           \
    MIDOUT_BEGIN(megdnn_fallback_conv1x1_factory_strategy,                     \
                 midout_iv(_midout_tag)) {                                     \
        if (param.filter_type.enumv() == param.src_type.enumv() &&             \
            param.src_type.enumv() == DTypeTrait<_i_src_type>::enumv &&        \
            param.dst_type.enumv() == DTypeTrait<_i_dst_type>::enumv) {        \
            return std::make_unique<                                           \
                    Conv1x1Strategy<_src_ctype, _bias_ctype, _dst_ctype,       \
                                    DTypeTrait<_i_bias_type>::ctype,           \
                                    DTypeTrait<_i_dst_type>::ctype,            \
                                    _postprocess_mode, _packmode>>(pack_size); \
        }                                                                      \
    }                                                                          \
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
            cb2(MatrixMulImpl::AlgoBase::PackMode::DEFAULT, dt_int8, dt_int32,
                dt_int32, dt_int8, dt_int32, dt_int32,
                PostprocessMode::NO_PROCESS, "Default::INT8x8x32_INT32"_hash);
            cb2(MatrixMulImpl::AlgoBase::PackMode::DEFAULT, dt_int8, dt_int16,
                dt_int16, dt_int8, dt_int16, dt_int16,
                PostprocessMode::NO_PROCESS, "Default::INT8x8x16_INT16"_hash);
#if MEGDNN_AARCH64 || MEGDNN_ARMV7
            cb2(MatrixMulImpl::AlgoBase::PackMode::DEFAULT,
                dtype::Quantized8Asymm, dtype::QuantizedS32,
                dtype::QuantizedS32, dt_uint8, dt_int32, dt_int32,
                PostprocessMode::NO_PROCESS,
                "Default::QUINT8x8x32_QINT32"_hash);
            cb2(MatrixMulImpl::AlgoBase::PackMode::DEFAULT,
                dtype::Quantized8Asymm, dtype::QuantizedS32,
                dtype::Quantized8Asymm, dt_uint8, dt_int32, dt_uint8,
                PostprocessMode::QUANTIZED, "Default::QUINT8x8x32_QUINT8"_hash);
#endif
            cb2(MatrixMulImpl::AlgoBase::PackMode::DEFAULT, dtype::QuantizedS8,
                dtype::QuantizedS32, dtype::QuantizedS32, dt_int8, dt_int32,
                dt_int32, PostprocessMode::NO_PROCESS,
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

            cb2(MatrixMulImpl::AlgoBase::PackMode::NO_PACK, dt_int8, dt_int16,
                dt_int16, dt_int8, dt_int16, dt_int16,
                PostprocessMode::NO_PROCESS, "NoPack::INT8x8x16_INT16"_hash);

            cb2(MatrixMulImpl::AlgoBase::PackMode::NO_PACK, dt_int8, dt_int32,
                dt_int32, dt_int8, dt_int32, dt_int32,
                PostprocessMode::NO_PROCESS, "NoPack::INT8x8x32_INT32"_hash);

            cb2(MatrixMulImpl::AlgoBase::PackMode::NO_PACK,
                dtype::QuantizedS8, dtype::QuantizedS32,
                dtype::QuantizedS32, dt_int8, dt_int32, dt_int32,
                PostprocessMode::NO_PROCESS,
                "NoPack::QINT8x8x32_QINT32"_hash);
            break;

        default:
            megdnn_throw("Invalid Pack Mode");
            break;
    }
#undef cb1
#undef cb2
    megdnn_throw("Invalid Data Type");
    return nullptr;
}

class StrategyDelegationStorage {
public:
    Conv1x1StrategyBase* get(const ConvBiasImpl::NCBKernSizeParam& param,
                             MatrixMulImpl::AlgoBase::PackMode pack_mode,
                             param::ConvBias::Format format) {
        MEGDNN_LOCK_GUARD(m_mtx);
        StrategyHashParam sparam;
        sparam.param = param;
        sparam.format = format;
        sparam.packmode = pack_mode;
        if (m_map_strategies.find(sparam) == m_map_strategies.end()) {
            auto strategy = create_conv1x1_strategy(param, pack_mode, format);
            m_map_strategies[sparam] = std::move(strategy);
        }
        return m_map_strategies[sparam].get();
    }

private:
    std::mutex m_mtx;
    std::unordered_map<StrategyHashParam, std::unique_ptr<Conv1x1StrategyBase>,
                       StrategyHashParamHash, StrategyHashParamEqual>
            m_map_strategies;
};

}  // anonymous namespace

Conv1x1StrategyBase* Conv1x1Factory::make_conv1x1_strategy(
        const ConvBiasImpl::NCBKernSizeParam& param,
        MatrixMulImpl::AlgoBase::PackMode pack_mode,
        param::ConvBias::Format format) {
    static StrategyDelegationStorage storage;
    return storage.get(param, pack_mode, format);
}

}  // namespace conv1x1
}  // namespace fallback
}  // namespace megdnn
