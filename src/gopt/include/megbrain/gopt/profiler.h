/**
 * \file src/gopt/include/megbrain/gopt/global_layout_transformation.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once
#include "megbrain/gopt/framework.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/gopt/layout_transform_context.h"
#include "megbrain/gopt/reformat_manager.h"
#include "megbrain/gopt/subgraph_extractor.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/plugin/opr_footprint.h"

namespace mgb {
namespace gopt {

class Problem;

/*!
 * \brief A profiler that collects all the performance data to describe the
 * global layout transform problem.
 */
class ProfilerBase {
public:
    using OprFormat = Problem::OprFormat;
    struct OperatorNodeRecord {
        const cg::OperatorNodeBase* opr;  ///< pointer to operator node
        ThinHashMap<OprFormat, float>
                costs;  ///< costs of operator node, i.e. the elapsed device
                        ///< time of the operator node on different opr format
                        ///< (layout configuration).
        std::string to_string() const;
    };
    struct VarNodeRecord {
        struct KeyHash {
            size_t operator()(
                    const std::pair<TensorFormats, TensorFormats>& val) const {
                size_t h1 = std::hash<uint32_t>()(static_cast<uint32_t>(val.first));
                size_t h2 = std::hash<uint32_t>()(static_cast<uint32_t>(val.second));
                return mgb::hash_pair_combine(h1, h2);
            }
        };
        const VarNode* var;  ///< pointer to var node
        std::unordered_map<
                std::pair<TensorFormats, TensorFormats>, float,
                KeyHash>
                costs;  ///< costs of var node, i.e. the elapsed
                        ///< device time of the layout transform.
                        ///< Key of the hashmap indicates the
                        ///< source tensor format and the target
                        ///< tensor format.
        std::string to_string() const;
    };
    /*!
     * \note the profiler assumes all the input and output var node are stored
     * in contiguous layout in memory
     */
    struct ProfilingResult {
        /// A hashmap, that maps the operator node to the costs (device elapsed
        /// time) of different layouts configuration
        ThinHashMap<cg::OperatorNodeBase*, OperatorNodeRecord> opr_record;
        /// A hashmap, that maps the var node to the costs of layout transform
        ThinHashMap<VarNode*, VarNodeRecord> var_record;
    };
    using OprFilter =
            thin_function<bool(const cg::OperatorNodeBase*, cg::OperatorNodeBase*)>;
    using VarNodeFilter = thin_function<bool(
            const VarNode*, TensorShape, TensorShape, ReformatManager::ReformatKey)>;

    ProfilerBase(float opr_threshold = 2.f, float var_node_threshold = 2.f);
    ProfilerBase(OprFilter opr_filter, VarNodeFilter var_node_filter = {})
            : m_opr_filter{std::move(opr_filter)},
              m_var_node_filter{std::move(var_node_filter)} {}
    virtual ~ProfilerBase() = default;
    virtual ProfilingResult profile(const Problem& problem) const = 0;
    static std::unique_ptr<ProfilerBase> make_profiler();

protected:
    OprFilter m_opr_filter;
    VarNodeFilter m_var_node_filter;
    float m_opr_threshold;
    float m_var_node_threshold;

private:
    OprFootprint m_opr_footprint;
};

}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
