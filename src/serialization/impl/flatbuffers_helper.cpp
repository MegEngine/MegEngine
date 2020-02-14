/**
 * \file src/serialization/impl/flatbuffers_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#if MGB_ENABLE_FBS_SERIALIZATION

#include "megbrain/serialization/internal/flatbuffers_helper.h"
#include "megbrain/common.h"

using namespace megdnn;

namespace mgb {
namespace serialization {
namespace fbs {
namespace intl {

megdnn::DTypeEnum convert_dtype_to_megdnn(DTypeEnum fb) {
    switch (fb) {
#define cb(_dt)           \
    case DTypeEnum_##_dt: \
        return megdnn::DTypeEnum::_dt;
        MEGDNN_FOREACH_DTYPE_NAME(cb)
        MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)
#undef cb
        default:
            // Float16 may be disabled
            megdnn_trap();
    }
}

DTypeEnum convert_dtype_to_fbs(megdnn::DTypeEnum enumv) {
    switch (enumv) {
#define cb(_dt)                  \
    case megdnn::DTypeEnum::_dt: \
        return DTypeEnum_##_dt;
        MEGDNN_FOREACH_DTYPE_NAME(cb)
        MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)
#undef cb
    }
    megdnn_trap();
}

megdnn::DType load_dtype(const fbs::DType* dtype) {
    auto param = dtype->param_as_LinearQuantizationParam();
    switch (dtype->type()) {
#define cb(_dt)           \
    case DTypeEnum_##_dt: \
        return dtype::_dt{};
        MEGDNN_FOREACH_DTYPE_NAME(cb)
#undef cb
        case DTypeEnum_QuantizedS4:
            return dtype::QuantizedS4{param->scale()};
        case DTypeEnum_QuantizedS8:
            return dtype::QuantizedS8{param->scale()};
        case DTypeEnum_QuantizedS16:
            return dtype::QuantizedS16{param->scale()};
        case DTypeEnum_QuantizedS32:
            return dtype::QuantizedS32{param->scale()};
        case DTypeEnum::DTypeEnum_Quantized4Asymm:
            return dtype::Quantized4Asymm{param->scale(), param->zero_point()};
        case DTypeEnum::DTypeEnum_Quantized8Asymm:
            return dtype::Quantized8Asymm{param->scale(), param->zero_point()};
    }
    return {};
}

flatbuffers::Offset<fbs::DType> build_dtype(
        flatbuffers::FlatBufferBuilder& builder, megdnn::DType dtype) {
    if (!dtype.valid())
        return {};
    DTypeEnum enumv{};
    switch (dtype.enumv()) {
#define cb(_dt)                  \
    case megdnn::DTypeEnum::_dt: \
        enumv = DTypeEnum_##_dt; \
        break;
        MEGDNN_FOREACH_DTYPE_NAME(cb)
        MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)
#undef cb
    }
    DTypeParam param_type = DTypeParam_NONE;
    flatbuffers::Offset<void> param;
    if (dtype.has_param()) {
        switch (dtype.enumv()) {
#define cb(_dt)                  \
    case megdnn::DTypeEnum::_dt: \
        mgb_trap();  // unreachable
            MEGDNN_FOREACH_DTYPE_NAME(cb)
#undef cb
#define CASE_ASYMMETRIC(_dt)                                                  \
    case megdnn::DTypeEnum::_dt: {                                            \
        auto&& p = dtype.param<dtype::_dt>();                                 \
        param_type = DTypeParam_LinearQuantizationParam;                      \
        param = CreateLinearQuantizationParam(builder, p.scale, p.zero_point) \
                        .Union();                                             \
        break;                                                                \
    }
#define CASE_SYMMETRIC(_dt)                                                    \
    case megdnn::DTypeEnum::_dt:                                               \
        param_type = DTypeParam_LinearQuantizationParam;                       \
        param = CreateLinearQuantizationParam(builder,                         \
                                              dtype.param<dtype::_dt>().scale) \
                        .Union();                                              \
        break;
            CASE_ASYMMETRIC(Quantized4Asymm)
            CASE_ASYMMETRIC(Quantized8Asymm)
            CASE_SYMMETRIC(QuantizedS4)
            CASE_SYMMETRIC(QuantizedS8)
            CASE_SYMMETRIC(QuantizedS16)
            CASE_SYMMETRIC(QuantizedS32)
        }
    }
    DTypeBuilder dt(builder);
    dt.add_type(enumv);
    if (param_type != DTypeParam_NONE) {
        dt.add_param_type(param_type);
        dt.add_param(param);
    }
    return dt.Finish();
}

}  // namespace intl
}  // namespace fbs
}  // namespace serialization
}  // namespace mgb

#endif
