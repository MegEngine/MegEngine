/**
 * \file dnn/src/common/algo_base.h
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

#include <functional>
#include <string>

#include "megdnn/oprs/base.h"
#include "src/common/utils.h"

namespace megdnn {

#define MEGDNN_DECL_ALGO_TYPE(_type)                              \
    uint32_t type() const override {                              \
        return static_cast<std::underlying_type<AlgoType>::type>( \
                AlgoType::_type);                                 \
    }

#define MEGDNN_FB_DECL_GET_ALGO_FROM_DESC(_opr)          \
    static fallback::_opr::AlgoBase* get_algo_from_desc( \
            const AlgorithmDesc& desc)

#define MEGDNN_FB_DEF_GET_ALGO_FROM_DESC(_opr)                  \
    fallback::_opr::AlgoBase* _opr::get_algo_from_desc(         \
            const AlgorithmDesc& desc) {                        \
        megdnn_assert(algo_pack().all_algos_map().find(desc) != \
                      algo_pack().all_algos_map().end());       \
        return algo_pack().all_algos_map().at(desc);            \
    }

#define MEGDNN_DEF_GET_ALGO_FROM_DESC(_opr)                               \
    _opr::AlgoBase* _opr::get_algo_from_desc(const AlgorithmDesc& desc) { \
        megdnn_assert(algo_pack().all_algos_map().find(desc) !=           \
                      algo_pack().all_algos_map().end());                 \
        return algo_pack().all_algos_map().at(desc);                      \
    }

/**
 * \brief construct algo from AlgorithmDesc
 */
template <typename AlgoBase>
class AlgoConstructMixin {
private:
    std::vector<std::unique_ptr<AlgoBase>> m_refhold;
protected:
    typename AlgoBase::Mapper m_all_algos_map;

public:

    //! construct the algo which described by desc, and return the instance
    AlgoBase* construct_and_get_algo(
            const detail::Algorithm::Info::Desc& desc) {
        auto iter = m_all_algos_map.find(desc);
        if (iter != m_all_algos_map.end()) {
            return m_all_algos_map.at(desc);
        }
        std::string serialized_bin;
        AlgoBase::serialize_write_pod(desc.type, serialized_bin);
        serialized_bin += desc.param;
        m_refhold.emplace_back(AlgoBase::deserialize(serialized_bin));
        m_all_algos_map.emplace(desc, m_refhold.back().get());
        return m_refhold.back().get();
    }

    void clear() {
        m_all_algos_map.clear();
        m_refhold.clear();
    }

    const typename AlgoBase::Mapper& all_algos_map() const {
        return m_all_algos_map;
    }
};

}  // namespace megdnn

namespace std {
template <>
struct hash<megdnn::detail::Algorithm::Info::Desc> {
    std::size_t operator()(
            const megdnn::detail::Algorithm::Info::Desc& desc) const {
        return megdnn::hash_combine<size_t>(
                megdnn::hash_combine<size_t>(
                        std::hash<std::string>()(desc.param),
                        std::hash<uint32_t>()(desc.type)),
                std::hash<uint32_t>()(static_cast<uint32_t>(desc.handle_type)));
    }
};
}  // namespace std

// vim: syntax=cpp.doxygen
