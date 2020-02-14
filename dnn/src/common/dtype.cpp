/**
 * \file dnn/src/common/dtype.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/dtype.h"
#include "src/common/utils.h"

#include <functional>
#include <unordered_map>
#include <cmath>

using namespace megdnn;
using namespace dtype;

#if MEGDNN_DISABLE_FLOAT16
#pragma message "megdnn float16 disabled"
#endif

#define IMPL(_name) \
DType::Trait _name::sm_trait = { \
    DTypeTrait<_name>::name,  \
    DTypeTrait<_name>::size_log, DTypeTrait<_name>::low_bit, \
    DTypeEnum::_name, \
    DTypeTrait<_name>::category, DTypeTrait<_name>::signedness, \
    DTypeTrait<_name>::has_param \
};
#define TEMPLATED_IMPL(_name) \
    template <>               \
    IMPL(_name)

MEGDNN_FOREACH_DTYPE_NAME(IMPL)
MEGDNN_FOREACH_PARAMETERIZED_DTYPE(TEMPLATED_IMPL)

#undef TEMPLATED_IMPL
#undef IMPL

void DType::on_assert_is_failed(const char *rname) const {
    megdnn_throw(megdnn_mangle(
                ssprintf("attempt to access dtype %s as %s",
                name(), rname).c_str()));
    MEGDNN_MARK_USED_VAR(rname);
}

void DType::on_request_lowbit_size() const {
    megdnn_throw(megdnn_mangle(
                ssprintf("attempt to get size of lowbit dtype %s", name())));
}

DType DType::from_enum(DTypeEnum ev) {
    switch (ev) {
#define cb(_dt) case DTypeEnum::_dt: return dtype::_dt();
        MEGDNN_FOREACH_DTYPE_NAME(cb)
#undef cb
#define cb(_dt) case DTypeEnum::_dt:
        MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)
            megdnn_throw(megdnn_mangle(
                "cannot construct parameterized DType via DType::from_enum"));
#undef cb
    }
    megdnn_throw(megdnn_mangle("bad DTypeEnum value"));
}

template <DTypeEnum type_enum>
typename ParameterizedDType<type_enum>::Trait*
ParameterizedDType<type_enum>::make_from_param(
        const DTypeParam<SelfType>& param) {
    struct Hasher {
        std::size_t operator()(const DTypeParam<SelfType>& key) const {
            return key.hash();
        }
    };
    static std::unordered_map<DTypeParam<SelfType>,
                              std::unique_ptr<SelfType::Trait>, Hasher>
            entries;

    auto it = entries.find(param);
    if (it != entries.end()) {
        return it->second.get();
    }
    entries[param] =
            std::make_unique<SelfType::Trait>(SelfType::sm_trait, param);
    return entries[param].get();
}

// Instantize `make_from_param` for all parameterized DTypes.
#define inst(_name) \
    template _name::Trait* _name::make_from_param(const DTypeParam<SelfType>&);
MEGDNN_FOREACH_PARAMETERIZED_DTYPE(inst)
#undef inst

DTypeParam<dt_quint8>::DTypeParamImpl(float scale, uint8_t zero_point)
        : scale{scale}, zero_point{zero_point} {
    //! As the nan is not equal to any value
    megdnn_assert(!std::isnan(scale), "nan number compare is not support");
}

inline std::size_t DTypeParam<dt_quint8>::hash() const {
    return std::hash<float>()(scale) ^ std::hash<uint8_t>()(zero_point);
}

inline bool DTypeParam<dt_quint8>::operator==(
        const DTypeParam<dt_quint8>& rhs) const {
    return scale == rhs.scale && zero_point == rhs.zero_point;
}

DTypeParam<dt_qint8>::DTypeParamImpl(float scale) : scale{scale} {
    //! As the nan is not equal to any value
    megdnn_assert(!std::isnan(scale), "nan number compare is not support");
}

inline std::size_t DTypeParam<dt_qint8>::hash() const {
    return std::hash<float>()(scale);
}

inline bool DTypeParam<dt_qint8>::operator==(
        const DTypeParam<dt_qint8>& rhs) const {
    return scale == rhs.scale;
}

DTypeParam<dt_qint16>::DTypeParamImpl(float scale) : scale{scale} {
    //! As the nan is not equal to any value
    megdnn_assert(!std::isnan(scale), "nan number compare is not support");
}

inline std::size_t DTypeParam<dt_qint16>::hash() const {
    return std::hash<float>()(scale);
}

inline bool DTypeParam<dt_qint16>::operator==(
        const DTypeParam<dt_qint16>& rhs) const {
    return scale == rhs.scale;
}

DTypeParam<dt_qint32>::DTypeParamImpl(float scale) : scale{scale} {
    //! As the nan is not equal to any value
    megdnn_assert(!std::isnan(scale), "nan number compare is not support");
}

inline std::size_t DTypeParam<dt_qint32>::hash() const {
    return std::hash<float>()(scale);
}

inline bool DTypeParam<dt_qint32>::operator==(
        const DTypeParam<dt_qint32>& rhs) const {
    return scale == rhs.scale;
}

DTypeParam<dt_quint4>::DTypeParamImpl(float scale, uint8_t zero_point)
        : scale{scale}, zero_point{zero_point} {
    //! As the nan is not equal to any value
    megdnn_assert(!std::isnan(scale), "nan number compare is not support");
}

inline std::size_t DTypeParam<dt_quint4>::hash() const {
    return std::hash<float>()(scale) ^ std::hash<uint8_t>()(zero_point);
}

inline bool DTypeParam<dt_quint4>::operator==(
        const DTypeParam<dt_quint4>& rhs) const {
    return scale == rhs.scale && zero_point == rhs.zero_point;
}

DTypeParam<dt_qint4>::DTypeParamImpl(float scale) : scale{scale} {
    //! As the nan is not equal to any value
    megdnn_assert(!std::isnan(scale), "nan number compare is not support");
}

inline std::size_t DTypeParam<dt_qint4>::hash() const {
    return std::hash<float>()(scale);
}

inline bool DTypeParam<dt_qint4>::operator==(
        const DTypeParam<dt_qint4>& rhs) const {
    return scale == rhs.scale;
}
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
