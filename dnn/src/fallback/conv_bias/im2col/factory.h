/**
 * \file dnn/src/fallback/conv_bias/im2col/factory.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include <unordered_map>
#include "src/fallback/conv_bias/im2col/strategy_base.h"
#include "src/fallback/conv_bias/opr_impl.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fallback_im2col_factory_make_strategy)

namespace megdnn {
namespace fallback {
namespace im2col {

enum class StrategyType : uint32_t {
    FLOAT = 0,
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    FLOAT_FP16 = 1,
#endif
#if !MEGDNN_DISABLE_FLOAT16
    FLOAT16_FLOAT16 = 2,
#endif
    INT8x8x32 = 3,
    INT8x8x16 = 4,
#if MEGDNN_AARCH64 || MEGDNN_ARMV7
    QUINT8x8x32 = 5,
    QUINT8x8x32x8 = 6,
#endif
    QINT8x8x32 = 7,
    QINT8x8x32x8 = 8
};

struct StrategyHashParam {
    bool is_xcorr;
    bool is_square;  //! kernel_h == kernel_w, stride_h = stride_w
    size_t block_m;
    size_t block_n;
    size_t block_k;
    size_t kernel;
    size_t stride;

    fallback::ConvBiasImpl::NCBKernSizeParam param;
    param::ConvBias::Format format;
    fallback::MatrixMulImpl::AlgoBase::PackMode packmode;
};

struct StrategyHashParamHash {
    uint64_t operator()(const StrategyHashParam& sparam) const {
        constexpr uint64_t base = 1;  //! avoid hashkey is zero
        uint64_t result =
                static_cast<uint64_t>(sparam.param.src_type.enumv()) + base;
        result = result ^
                 ((static_cast<uint64_t>(sparam.param.dst_type.enumv()) + base)
                  << 3);
        result = result ^
                 ((static_cast<uint64_t>(sparam.param.filter_type.enumv()) +
                   base)
                  << 6);
        result = result ^
                 ((static_cast<uint64_t>(sparam.param.bias_type.enumv()) + base)
                  << 9);
        result = result ^ ((static_cast<uint64_t>(sparam.format) + base) << 12);
        result = result ^
                 ((static_cast<uint64_t>(sparam.packmode) + base) << 15);
        result =
                result ^ ((static_cast<uint64_t>(sparam.block_m) + base) << 18);
        result =
                result ^ ((static_cast<uint64_t>(sparam.block_n) + base) << 22);
        result =
                result ^ ((static_cast<uint64_t>(sparam.block_k) + base) << 26);
        result = result ^ ((static_cast<uint64_t>(sparam.kernel) + base) << 30);
        result = result ^ ((static_cast<uint64_t>(sparam.stride) + base) << 34);
        result = result ^
                 ((static_cast<uint64_t>(sparam.is_square) + base) << 35);
        result = result ^
                 ((static_cast<uint64_t>(sparam.is_xcorr) + base) << 36);
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
        flags = param1.block_m == param2.block_m && flags;
        flags = param1.block_n == param2.block_n && flags;
        flags = param1.block_k == param2.block_k && flags;
        flags = param1.kernel == param2.kernel && flags;
        flags = param1.stride == param2.stride && flags;
        flags = param1.is_square == param2.is_square && flags;
        flags = param1.is_xcorr == param2.is_xcorr && flags;
        return flags;
    };
};

class StrategyDelegationStorage {
    std::mutex m_mtx;
    std::unordered_map<StrategyHashParam, std::unique_ptr<StrategyBase>,
                       StrategyHashParamHash, StrategyHashParamEqual>
            map_strategys;

public:
    ~StrategyDelegationStorage() = default;

    template <typename Strategy>
    Strategy* get(fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                  const fallback::ConvBiasImpl::NCBKernSizeParam& param,
                  StrategyType stype);
};

class Factory {
public:
    static StrategyBase* get_im2col_strategy(
            const fallback::ConvBiasImpl::NCBKernSizeParam& param,
            fallback::MatrixMulImpl::AlgoBase* matmul_algo) {
        static StrategyDelegationStorage storage;
        StrategyType strategytype = get_strategy_type(param);
        return storage.get<StrategyBase>(matmul_algo, param, strategytype);
    }

    static StrategyType get_strategy_type(
            const fallback::ConvBiasImpl::NCBKernSizeParam& param) {
#define cb1(_dt, _post_ctype, _strategytype)                   \
    if (param.filter_type.enumv() == DTypeTrait<_dt>::enumv) { \
        return _strategytype;                                  \
    }

#define cb2(_i_src_type, _i_bias_type, _i_dst_type, _src_ctype, _bias_ctype, \
            _dst_ctype, _strategytype)                                       \
    if (param.filter_type.enumv() == param.src_type.enumv() &&               \
        param.src_type.enumv() == DTypeTrait<_i_src_type>::enumv &&          \
        param.dst_type.enumv() == DTypeTrait<_i_dst_type>::enumv) {          \
        return _strategytype;                                                \
    }

        cb1(dt_float32, dt_float32, StrategyType::FLOAT);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        cb1(dt_float16, __fp16, StrategyType::FLOAT_FP16);
#endif
#if !MEGDNN_DISABLE_FLOAT16
        cb1(dt_float16, dt_float16, StrategyType::FLOAT16_FLOAT16);
#endif
        cb2(dt_int8, dt_int32, dt_int32, dt_int8, dt_int32, dt_int32,
            StrategyType::INT8x8x32);

        cb2(dt_int8, dt_int16, dt_int16, dt_int8, dt_int16, dt_int16,
            StrategyType::INT8x8x16);

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
        cb2(dtype::Quantized8Asymm, dtype::QuantizedS32, dtype::QuantizedS32,
            dt_uint8, dt_int32, dt_int32, StrategyType::QUINT8x8x32);

        cb2(dtype::Quantized8Asymm, dtype::QuantizedS32, dtype::Quantized8Asymm,
            dt_uint8, dt_int32, dt_uint8, StrategyType::QUINT8x8x32x8);
#endif
        cb2(dtype::QuantizedS8, dtype::QuantizedS32, dtype::QuantizedS32,
            dt_int8, dt_int32, dt_int32, StrategyType::QINT8x8x32);

        cb2(dtype::QuantizedS8, dtype::QuantizedS32, dtype::QuantizedS8,
            dt_int8, dt_int32, dt_int8, StrategyType::QINT8x8x32x8);
#undef cb1
#undef cb2
        megdnn_throw("not support datatype in im2col strategy\n");
    }

#define cb1(_format, _packmode, _dt, _post_ctype, _postprocess_mode,  \
            _midout_tag)                                              \
    MIDOUT_BEGIN(megdnn_fallback_im2col_factory_make_strategy,        \
                 midout_iv(_midout_tag)) {                            \
        if (param.filter_type.enumv() == DTypeTrait<_dt>::enumv) {    \
            return std::make_unique<                                  \
                    Strategy<_dt, _dt, _dt, _post_ctype, _post_ctype, \
                             _postprocess_mode, PackMode::_packmode,  \
                             FormatMode::_format>>();                 \
        }                                                             \
    }                                                                 \
    MIDOUT_END();                                                     \
    return {};

#define cb2(_format, _packmode, _i_src_type, _i_bias_type, _i_dst_type, \
            _src_ctype, _bias_ctype, _dst_ctype, _postprocess_mode,     \
            _midout_tag)                                                \
    MIDOUT_BEGIN(megdnn_fallback_im2col_factory_make_strategy,          \
                 midout_iv(_midout_tag)) {                              \
        if (param.filter_type.enumv() == param.src_type.enumv() &&      \
            param.src_type.enumv() == DTypeTrait<_i_src_type>::enumv && \
            param.dst_type.enumv() == DTypeTrait<_i_dst_type>::enumv) { \
            return std::make_unique<Strategy<                           \
                    _src_ctype, _bias_ctype, _dst_ctype,                \
                    DTypeTrait<_i_bias_type>::ctype,                    \
                    DTypeTrait<_i_dst_type>::ctype, _postprocess_mode,  \
                    PackMode::_packmode, FormatMode::_format>>();       \
        }                                                               \
    }                                                                   \
    MIDOUT_END();                                                       \
    return {};
#define cb3(_format, _packmode, _i_src_type, _i_bias_type, _i_dst_type,        \
            _src_ctype, _bias_ctype, _dst_ctype, _postprocess_mode,            \
            _midout_tag)                                                       \
    MIDOUT_BEGIN(megdnn_fallback_im2col_factory_make_strategy,                 \
                 midout_iv(_midout_tag)) {                                     \
        if (param.filter_type.enumv() == param.src_type.enumv() &&             \
            param.src_type.enumv() == DTypeTrait<_i_src_type>::enumv &&        \
            param.dst_type.enumv() == DTypeTrait<_i_dst_type>::enumv) {        \
            return std::make_unique<                                           \
                    Strategy<_src_ctype, _bias_ctype, _dst_ctype, _bias_ctype, \
                             _dst_ctype, _postprocess_mode,                    \
                             PackMode::_packmode, FormatMode::_format>>();     \
        }                                                                      \
    }                                                                          \
    MIDOUT_END();                                                              \
    return {};

    static std::unique_ptr<StrategyBase> make_default_strategy(
            fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::ConvBiasImpl::NCBKernSizeParam& param,
            StrategyType strategytype) {
        MEGDNN_MARK_USED_VAR(matmul_algo);
        param::ConvBias::Format format = param.filter_meta.format;
        switch (strategytype) {
            case StrategyType::FLOAT:
                if (format == param::ConvBias::Format::NCHW) {
                    cb1(NCHW, DEFAULT, dt_float32, dt_float32,
                        PostprocessMode::FLOAT,
                        "DefaultStrategyType::FLOAT"_hash);
                } else if (format == param::ConvBias::Format::NCHW44) {
#if MEGDNN_AARCH64 || MEGDNN_ARMV7
                    auto matmul_block = matmul_algo->get_inner_block_size();
                    //! Optimize NCHW44 3x3s2 aarch64 8X12X1 and armv7 4x12x1
                    //! im2col+pack fuse
                    if ((matmul_block.m == 8 || matmul_block.m == 4) &&
                        matmul_block.n == 12 && matmul_block.k == 1 &&
                        param.filter_meta.spatial[0] == 3 &&
                        param.filter_meta.spatial[1] == 3 &&
                        param.filter_meta.stride[0] == 2 &&
                        param.filter_meta.stride[1] == 2 &&
                        !param.filter_meta.should_flip) {
                        MIDOUT_BEGIN(
                                megdnn_fallback_im2col_factory_make_strategy,
                                midout_iv("8x12x1_fuse_packb_s2_nchw44"_hash)) {
                            return std::make_unique<
                                    StrategyFuseXx12x1Nchw44K3x3S2<
                                            float, float,
                                            PostprocessMode::FLOAT>>();
                        }
                        MIDOUT_END();
                        return {};
                    }
#endif

                    cb1(NCHW44, DEFAULT, dt_float32, dt_float32,
                        PostprocessMode::FLOAT,
                        "DefaultStrategyTypeNCHW44::FLOAT"_hash);
                } else {
                    megdnn_throw(
                            ssprintf("Current only support layout "
                                     "NCHW44/NCHW for im2col "
                                     "algo, but got %d\n",
                                     uint32_t(format)));
                }
                break;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case StrategyType::FLOAT_FP16:
                cb1(NCHW, DEFAULT, dt_float16, __fp16, PostprocessMode::FLOAT,
                    "DefaultStrategyType::FLOAT_FP16"_hash);
                break;
#endif
#if !MEGDNN_DISABLE_FLOAT16
            case StrategyType::FLOAT16_FLOAT16:
                cb1(NCHW, DEFAULT, dt_float16, dt_float16,
                    PostprocessMode::NO_PROCESS,
                    "DefaultStrategyType::FLOAT16_FLOAT16"_hash);
                break;
#endif
            case StrategyType::INT8x8x32:
                if (format == param::ConvBias::Format::NCHW) {
                    cb3(NCHW, DEFAULT, dt_int8, dt_int32, dt_int32, dt_int8,
                        dt_int32, dt_int32, PostprocessMode::ADD_BIAS,
                        "DefaultStrategyType::INT8x8x32"_hash);
                } else if (format == param::ConvBias::Format::NCHW44 ||
                           format == param::ConvBias::Format::NCHW44_DOT) {
                    cb3(NCHW44, DEFAULT, dt_int8, dt_int32, dt_int32, dt_int8,
                        dt_int32, dt_int32, PostprocessMode::ADD_BIAS,
                        "DefaultStrategyType::INT8x8x32"_hash);
                } else {
                    megdnn_throw(
                            ssprintf("Current only support layout "
                                     "NCHW44/NCHW/NCHW_DOT for im2col "
                                     "algo, but got %d\n",
                                     uint32_t(format)));
                }

                break;

            case StrategyType::INT8x8x16:
                if (format == param::ConvBias::Format::NCHW) {
                    cb3(NCHW, DEFAULT, dt_int8, dt_int16, dt_int16, dt_int8,
                        dt_int16, dt_int16, PostprocessMode::ADD_BIAS,
                        "DefaultStrategyType::INT8x8x16"_hash);
                } else if (format == param::ConvBias::Format::NCHW44) {
                    cb3(NCHW44, DEFAULT, dt_int8, dt_int16, dt_int16, dt_int8,
                        dt_int16, dt_int16, PostprocessMode::ADD_BIAS,
                        "DefaultStrategyType::INT8x8x16"_hash);
                } else {
                    megdnn_throw(
                            ssprintf("Current only support layout "
                                     "NCHW44/NCHW for im2col "
                                     "algo, but got %d\n",
                                     uint32_t(format)));
                }
                break;
#if MEGDNN_AARCH64 || MEGDNN_ARMV7
            case StrategyType::QUINT8x8x32:
                cb3(NCHW, DEFAULT, dtype::Quantized8Asymm, dtype::QuantizedS32,
                    dtype::QuantizedS32, dt_uint8, dt_int32, dt_int32,
                    PostprocessMode::ADD_BIAS,
                    "DefaultStrategyType::QUINT8x8x32"_hash);
                break;

            case StrategyType::QUINT8x8x32x8:
                cb2(NCHW, DEFAULT, dtype::Quantized8Asymm, dtype::QuantizedS32,
                    dtype::Quantized8Asymm, dt_uint8, dt_int32, dt_uint8,
                    PostprocessMode::QUANTIZED,
                    "DefaultStrategyType::QUINT8x8x32x8"_hash);
                break;
#endif
            case StrategyType::QINT8x8x32:
                if (format == param::ConvBias::Format::NCHW) {
                    cb3(NCHW, DEFAULT, dtype::QuantizedS8, dtype::QuantizedS32,
                        dtype::QuantizedS32, dt_int8, dt_int32, dt_int32,
                        PostprocessMode::ADD_BIAS,
                        "DefaultStrategyTypeNCHW::QINT8x8x32"_hash);
                } else if (format == param::ConvBias::Format::NCHW44 ||
                           format == param::ConvBias::Format::NCHW44_DOT) {
                    cb3(NCHW44, DEFAULT, dtype::QuantizedS8,
                        dtype::QuantizedS32, dtype::QuantizedS32, dt_int8,
                        dt_int32, dt_int32, PostprocessMode::ADD_BIAS,
                        "DefaultStrategyTypeHCHW44::QINT8x8x32"_hash);
                } else {
                    megdnn_throw(
                            ssprintf("Current only support layout "
                                     "NCHW44/NCHW/NCHW_DOT for im2col "
                                     "algo, but got %d\n",
                                     uint32_t(format)));
                }
                break;

            case StrategyType::QINT8x8x32x8:
                if (format == param::ConvBias::Format::NCHW) {
                    cb2(NCHW, DEFAULT, dtype::QuantizedS8, dtype::QuantizedS32,
                        dtype::QuantizedS8, dt_int8, dt_int32, dt_int8,
                        PostprocessMode::QUANTIZED,
                        "DefaultStrategyType::QINT8x8x32x8"_hash);
                } else if (format == param::ConvBias::Format::NCHW44 ||
                           format == param::ConvBias::Format::NCHW44_DOT) {
                    if (format == param::ConvBias::Format::NCHW44) {
                        //! Optimize NCHW44 3x3s1 4X4X16 im2col+pack fuse
#if MEGDNN_AARCH64
                        auto matmul_block = matmul_algo->get_inner_block_size();
                        if (matmul_block.m == 4 && matmul_block.n == 4 &&
                            matmul_block.k == 16 &&
                            param.filter_meta.spatial[0] == 3 &&
                            param.filter_meta.spatial[1] == 3 &&
                            param.filter_meta.stride[0] == 1 &&
                            param.filter_meta.stride[1] == 1 &&
                            !param.filter_meta.should_flip) {
                            MIDOUT_BEGIN(
                                    megdnn_fallback_im2col_factory_make_strategy,
                                    midout_iv(
                                            "DefaultStrategyType::INT8x8x32_4x4x16"_hash)) {
                                return std::make_unique<
                                        StrategyFuse4x4x16Nchw44<
                                                dt_qint32, dt_qint8,
                                                PostprocessMode::QUANTIZED>>();
                            }
                            MIDOUT_END();
                            return {};
                        }
#endif
                    } else {
#if MEGDNN_AARCH64
                        auto matmul_block = matmul_algo->get_inner_block_size();
                        //! Optimize NCHW44_DOT 3x3s1 8X12X4 im2col+pack fuse
                        if (matmul_block.m == 8 && matmul_block.n == 12 &&
                            matmul_block.k == 4 &&
                            param.filter_meta.spatial[0] == 3 &&
                            param.filter_meta.spatial[1] == 3 &&
                            param.filter_meta.stride[0] == 1 &&
                            param.filter_meta.stride[1] == 1 &&
                            !param.filter_meta.should_flip) {
                            MIDOUT_BEGIN(
                                    megdnn_fallback_im2col_factory_make_strategy,
                                    midout_iv(
                                            "DefaultStrategyType::INT8x8x32_8x12x4"_hash)) {
                                return std::make_unique<
                                        StrategyFuse8x12x4Nchw44Dot<
                                                dt_qint32, dt_qint8,
                                                PostprocessMode::QUANTIZED>>();
                            }
                            MIDOUT_END();
                            return {};
                        }
#endif
#if MEGDNN_ARMV7
                        auto matmul_block = matmul_algo->get_inner_block_size();
                        if (matmul_block.m == 8 && matmul_block.n == 4 &&
                            matmul_block.k == 4 &&
                            param.filter_meta.spatial[0] == 3 &&
                            param.filter_meta.spatial[1] == 3 &&
                            param.filter_meta.stride[0] == 2 &&
                            param.filter_meta.stride[1] == 2 &&
                            !param.filter_meta.should_flip) {
                            MIDOUT_BEGIN(
                                    megdnn_fallback_im2col_factory_make_strategy,
                                    midout_iv(
                                            "DefaultStrategyType::INT8x8x32_8x4x4_s2"_hash)) {
                                return std::make_unique<
                                        StrategyFuse8x4x4Nchw44DotK3x3S2<
                                                dt_qint32, dt_qint8,
                                                PostprocessMode::QUANTIZED>>();
                            }
                            MIDOUT_END();
                            return {};
                        }
#endif
                    }
                    cb2(NCHW44, DEFAULT, dtype::QuantizedS8,
                        dtype::QuantizedS32, dtype::QuantizedS8, dt_int8,
                        dt_int32, dt_int8, PostprocessMode::QUANTIZED,
                        "DefaultStrategyTypeNCHW44::QINT8x8x32x8"_hash);
                } else {
                    megdnn_throw(
                            ssprintf("Current only support layout "
                                     "NCHW44/NCHW/NCHW_DOT for im2col "
                                     "algo, but got %d\n",
                                     uint32_t(format)));
                }
                break;
        }
        megdnn_throw(ssprintf("Unsupported strategy type %u in default mode",
                              uint32_t(strategytype)));
    }

    static std::unique_ptr<StrategyBase> make_nopack_strategy(
            fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::ConvBiasImpl::NCBKernSizeParam& param,
            StrategyType strategytype) {
        MEGDNN_MARK_USED_VAR(matmul_algo);
        switch (strategytype) {
            case StrategyType::FLOAT:
                cb1(NCHW, NO_PACK, dt_float32, dt_float32,
                    PostprocessMode::FLOAT, "NoPackStrategyType::FLOAT"_hash);
                break;
#if !MEGDNN_DISABLE_FLOAT16
            case StrategyType::FLOAT16_FLOAT16:
                cb1(NCHW, NO_PACK, dt_float16, dt_float16,
                    PostprocessMode::NO_PROCESS,
                    "NoPackStrategyType::FLOAT16_FLOAT16"_hash);
                break;
#endif
            case StrategyType::INT8x8x16:
                cb3(NCHW, NO_PACK, dt_int8, dt_int16, dt_int16, dt_int8,
                    dt_int16, dt_int16, PostprocessMode::ADD_BIAS,
                    "NoPackStrategyType::INT8x8x16"_hash);
                break;
            case StrategyType::INT8x8x32:
                cb3(NCHW, NO_PACK, dt_int8, dt_int32, dt_int32, dt_int8,
                    dt_int32, dt_int32, PostprocessMode::ADD_BIAS,
                    "NoPackStrategyType::INT8x8x32"_hash);
                break;
            default:
                megdnn_throw(
                        ssprintf("Unsupported strategy type %u in no_pack mode",
                                 uint32_t(strategytype)));
                break;
        }
        megdnn_throw(ssprintf("Unsupported strategy type %u in no_pack mode",
                              uint32_t(strategytype)));
    }

    static std::unique_ptr<StrategyBase> make_onlypacka_strategy(
            fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::ConvBiasImpl::NCBKernSizeParam& param,
            StrategyType strategytype) {
        MEGDNN_MARK_USED_VAR(matmul_algo);
        switch (strategytype) {
            case StrategyType::FLOAT:
                cb1(NCHW, ONLY_PACKA, dt_float32, dt_float32,
                    PostprocessMode::FLOAT,
                    "OnlyPackaStrategyType::FLOAT"_hash);
                break;
            default:
                megdnn_throw(ssprintf(
                        "Unsupported strategy type %u in onlypacka mode",
                        uint32_t(strategytype)));
                break;
        }
        megdnn_throw(ssprintf("Unsupported strategy type %u in onlypacka mode",
                              uint32_t(strategytype)));
    }

#undef cb1
#undef cb2
#undef cb3

    static std::unique_ptr<StrategyBase> make_strategy(
            fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            fallback::MatrixMulImpl::AlgoBase::PackMode packmode,
            const fallback::ConvBiasImpl::NCBKernSizeParam& param,
            StrategyType stype) {
        switch (packmode) {
            case MatrixMulImpl::AlgoBase::PackMode::DEFAULT:
                return make_default_strategy(matmul_algo, param, stype);
                break;
            case MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA:
                return make_onlypacka_strategy(matmul_algo, param, stype);
                break;
            case MatrixMulImpl::AlgoBase::PackMode::NO_PACK:
                return make_nopack_strategy(matmul_algo, param, stype);
                break;
            default:
                megdnn_throw(
                        "not support packmode except default onlypackA "
                        "nopack");
                break;
        }
        megdnn_throw("factory make Strategy error please check your code");
    }
};

template <typename Strategy>
Strategy* StrategyDelegationStorage::get(
        fallback::MatrixMulImpl::AlgoBase* matmul_algo,
        const fallback::ConvBiasImpl::NCBKernSizeParam& param,
        StrategyType stype) {
    fallback::MatrixMulImpl::AlgoBase::PackMode packmode =
            matmul_algo->packmode();
    //! nopack mode block_m block_n block_k is zero
    size_t block_m = 0, block_n = 0, block_k = 0;
    if (packmode == fallback::MatrixMulImpl::AlgoBase::PackMode::DEFAULT ||
        packmode == fallback::MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA) {
        block_m = matmul_algo->get_inner_block_size().m;
        block_n = matmul_algo->get_inner_block_size().n;
        block_k = matmul_algo->get_inner_block_size().k;
    }
    StrategyHashParam sparam;
    sparam.param = param;
    sparam.format = param.filter_meta.format;
    sparam.packmode = packmode;
    sparam.block_m = block_m;
    sparam.block_n = block_n;
    sparam.block_k = block_k;
    sparam.kernel = param.filter_meta.spatial[0];
    sparam.stride = param.filter_meta.stride[0];
    sparam.is_square =
            param.filter_meta.spatial[0] == param.filter_meta.spatial[1];
    sparam.is_xcorr = param.filter_meta.should_flip;
    MEGDNN_LOCK_GUARD(m_mtx);
    if (map_strategys.find(sparam) == map_strategys.end()) {
        auto strategy =
                Factory::make_strategy(matmul_algo, packmode, param, stype);
        map_strategys[sparam] = std::move(strategy);
    }
    return static_cast<Strategy*>(map_strategys[sparam].get());
}
}  // namespace im2col
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
