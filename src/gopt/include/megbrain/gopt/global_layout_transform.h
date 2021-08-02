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
#include "megbrain/gopt/reformat_manager.h"
#include "megbrain/gopt/subgraph_extractor.h"
#include "megbrain/opr/dnn/convolution.h"

namespace mgb {
namespace gopt {

/*!
 * \brief A structure that describe the data types and  tensor formats
 * configuration of the opr format
 */
struct OprTensorFormatsConfiguration {
    using OprFormat = opr::ConvBias::Param::Format;
    using OprTensorFormatsDispatcher =
            thin_function<Maybe<OprTensorFormatsConfiguration>(
                    const cg::OperatorNodeBase*)>;
    Typeinfo* typeinfo;
    OprFormat opr_format;
    SmallVector<DTypeEnum> input_dtypes;
    SmallVector<DTypeEnum> output_dtypes;
    SmallVector<TensorFormats> input_tensor_formats;
    SmallVector<TensorType> input_tensor_types;
    SmallVector<TensorFormats> output_tensor_formats;
    static OprTensorFormatsDispatcher* find_dispatcher_by_type_format(
            Typeinfo* type, OprFormat opr_format);
};

/*!
 * \brief A structure that describes the global layout transform problem
 */
class Problem {
public:
    using OprFormat = OprTensorFormatsConfiguration::OprFormat;
    using OprTensorFormatsDispatcher =
            OprTensorFormatsConfiguration::OprTensorFormatsDispatcher;
    using OprConfigTrait =
            ThinHashMap<Typeinfo*,
                        ThinHashMap<OprFormat, OprTensorFormatsDispatcher*>>;
    struct Attribute {
        OprFormat base_opr_format;  /// the base opr format indicates that the
                                    /// network to be optimized is constructed
                                    /// in the base opr format, i.e. all the
                                    /// format aware operators (conv, conv_bias,
                                    /// deconv, pooling etc.) are built in
                                    /// this format.
        TensorFormats
                base_tensor_formats;  /// the base tensor format indicates that
                                      /// all the format agnostic operators
                                      /// (like elemwise, elemwise multi type,
                                      /// typecvt etc.) are built in the base
                                      /// tensor format.
    };
    Problem(const GraphPartition& graph_partition,
            const SmallVector<TensorFormats>& available_tensor_formats,
            const OprConfigTrait& opr_config, const Attribute& attribute)
            : m_graph_partition{graph_partition},
              m_available_tensor_formats{available_tensor_formats},
              m_opr_configs{opr_config},
              m_attribute{attribute} {}
    ~Problem() noexcept = default;

    const GraphPartition& graph_partition() const { return m_graph_partition; }
    const OprConfigTrait& opr_configs() const { return m_opr_configs; }
    const SmallVector<TensorFormats>& available_tensor_formats() const {
        return m_available_tensor_formats;
    }
    TensorFormats base_format() const {
        return m_attribute.base_tensor_formats;
    }
    OprTensorFormatsConfiguration base_config(
            const cg::OperatorNodeBase* opr) const {
        auto _ = OprTensorFormatsConfiguration::find_dispatcher_by_type_format(
                opr->dyn_typeinfo(), m_attribute.base_opr_format);
        auto rst = (*_)(opr);
        if (rst.valid())
            return rst.val();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = m_attribute.base_opr_format;
        for (const auto& i : opr->input()) {
            config.input_dtypes.emplace_back(i->dtype().enumv());
            config.input_tensor_formats.emplace_back(
                    m_attribute.base_tensor_formats);
            config.input_tensor_types.emplace_back(TensorType::FEATURE);
        }
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        config.output_tensor_formats.emplace_back(
                m_attribute.base_tensor_formats);
        return config;
    }

private:
    const GraphPartition& m_graph_partition;  /// the graph partition
    const SmallVector<TensorFormats>&
            m_available_tensor_formats;  /// the available tensor formats, used
                                         /// for format agnostic operators (like
                                         /// elemwise, elemwise multi type,
                                         /// typecvt, etc.
    const OprConfigTrait&
            m_opr_configs;  /// the available opr format configurations, used
                            /// for format aware operators (like conv, deconv,
                            /// conv_bias, etc.
    Attribute m_attribute;  /// the extra attributes to describe the problem
};

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
                size_t h1 =
                        std::hash<uint32_t>()(static_cast<uint32_t>(val.first));
                size_t h2 = std::hash<uint32_t>()(
                        static_cast<uint32_t>(val.second));
                return mgb::hash_pair_combine(h1, h2);
            }
        };
        const VarNode* var;  ///< pointer to var node
        std::unordered_map<std::pair<TensorFormats, TensorFormats>, float,
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

    ProfilerBase() = default;
    virtual ~ProfilerBase() = default;
    virtual ProfilingResult profile(const Problem& problem) const = 0;
    static std::unique_ptr<ProfilerBase> make_profiler();
};

}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
