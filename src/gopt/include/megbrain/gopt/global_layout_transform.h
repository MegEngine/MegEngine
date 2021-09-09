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
#include "megbrain/gopt/reformat_manager.h"
#include "megbrain/gopt/subgraph_extractor.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/plugin/opr_footprint.h"

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
class LayoutTransformContext {
public:
    using OprList = SubGraphExtractor::OprList;
    using OprFormat = OprTensorFormatsConfiguration::OprFormat;
    using OprTensorFormatsDispatcher =
            OprTensorFormatsConfiguration::OprTensorFormatsDispatcher;
    using OprConfigTrait =
            ThinHashMap<Typeinfo*,
                        ThinHashMap<OprFormat, OprTensorFormatsDispatcher*>>;
    using ReformatAttribute = ReformatManager::ReformatKey::Attribute;
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
        ReformatAttribute
                reformat_attribute;  /// additional reformat attribute, which
                                     /// indicates whether to pad nhwc layout
                                     /// automatically or to enable nhwcd4 format
                                     /// on opencl platform to use image object
    };
    LayoutTransformContext() = delete;
    LayoutTransformContext(OprList opr_list,
                           SmallVector<TensorFormats> available_tensor_formats,
                           Attribute attribute)
            : m_opr_list{std::move(opr_list)},
              m_available_tensor_formats{std::move(available_tensor_formats)},
              m_attribute{attribute} {}
    LayoutTransformContext(OprList opr_list,
                           SmallVector<TensorFormats> available_tensor_formats,
                           OprConfigTrait opr_configs, Attribute attribute)
            : m_opr_list{std::move(opr_list)},
              m_available_tensor_formats{std::move(available_tensor_formats)},
              m_opr_configs{std::move(opr_configs)},
              m_attribute{attribute} {}
    const OprList& opr_list() const { return m_opr_list; }
    const SmallVector<TensorFormats>& available_tensor_formats() const {
        return m_available_tensor_formats;
    }
    const OprConfigTrait& opr_configs() const { return m_opr_configs; }
    Attribute attribute() const { return m_attribute; }
    /*!
     * \brief add an op format configuration for a particular operator type
     * \param opr runtime typeinfo of operator
     * \param opr_format op format configuration which to be enabled in the
     * layout transform problem
     */
    LayoutTransformContext& add_opr_config(Typeinfo* opr, OprFormat opr_format);
    /*!
     * \brief add a vector of op format configurations for a particular operator
     * type
     * \param opr runtime typeinfo of operator
     * \param opr_format op format configuration which to be enabled in the
     * layout transform problem
     */
    LayoutTransformContext& add_opr_config(Typeinfo* opr,
                                           SmallVector<OprFormat> opr_formats);

private:
    OprList m_opr_list; /// supported operator list
    SmallVector<TensorFormats>
            m_available_tensor_formats;  /// the available tensor formats, used
                                         /// for format agnostic operators (like
                                         /// elemwise, elemwise multi type,
                                         /// typecvt, etc.
    OprConfigTrait m_opr_configs;  /// the available opr format configurations,
                                   /// used for format aware operators (like
                                   /// conv, deconv, conv_bias, etc.
    Attribute m_attribute;  /// the extra attributes to describe the problem
};

class Problem {
public:
    using OprFormat = OprTensorFormatsConfiguration::OprFormat;
    using OprTensorFormatsDispatcher =
            OprTensorFormatsConfiguration::OprTensorFormatsDispatcher;
    using OprConfigTrait = LayoutTransformContext::OprConfigTrait;
    using Attribute = LayoutTransformContext::Attribute;

    Problem(const GraphPartition& graph_partition,
            const LayoutTransformContext& ctx)
            : m_graph_partition{graph_partition}, m_ctx{ctx} {}
    ~Problem() noexcept = default;

    const GraphPartition& graph_partition() const { return m_graph_partition; }
    const OprConfigTrait& opr_configs() const { return m_ctx.opr_configs(); }
    const SmallVector<TensorFormats>& available_tensor_formats() const {
        return m_ctx.available_tensor_formats();
    }
    TensorFormats base_format() const {
        return m_ctx.attribute().base_tensor_formats;
    }
    Attribute attribute() const { return m_ctx.attribute(); }
    /*!
     * \brief return the tensor formats configuration of an operator in the
     * default op format
     */
    OprTensorFormatsConfiguration base_config(
            const cg::OperatorNodeBase* opr) const {
        auto _ = OprTensorFormatsConfiguration::find_dispatcher_by_type_format(
                opr->dyn_typeinfo(), m_ctx.attribute().base_opr_format);
        auto rst = (*_)(opr);
        if (rst.valid())
            return rst.val();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = m_ctx.attribute().base_opr_format;
        for (const auto& i : opr->input()) {
            config.input_dtypes.emplace_back(i->dtype().enumv());
            config.input_tensor_formats.emplace_back(base_format());
            config.input_tensor_types.emplace_back(TensorType::FEATURE);
        }
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        config.output_tensor_formats.emplace_back(base_format());
        return config;
    }

private:
    const GraphPartition& m_graph_partition;  /// the graph partition
    const LayoutTransformContext& m_ctx;
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
    using OprFilter = thin_function<bool(const cg::OperatorNodeBase*,
                                         cg::OperatorNodeBase*)>;
    using VarNodeFilter =
            thin_function<bool(const VarNode*, TensorShape, TensorShape,
                               ReformatManager::ReformatKey)>;

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

/*! 
 * \brief abstract solver 
 */
class SolverBase {
public:
    using OprFormat = Problem::OprFormat;
    using Solution = ThinHashMap<cg::OperatorNodeBase*, OprFormat>;
    SolverBase() = default;
    virtual ~SolverBase() = default;
    /*!
     * \brief solve the given problem
     */
    virtual Solution solve(const Problem& problem) const = 0;
    /*!
     * \brief check whether the given problem can be solved by the
     * algorithm(i.e. solver).
     */
    virtual bool can_solve(const Problem& problem) const = 0;
};

/*!
 * \brief solvers that will first collect the costs of operators in different op
 * format and the costs of layout transform of varnode with a user provided
 * profiler on the target device. This will lead to time consuming. 
 */
class ProfilingBasedSolver : public SolverBase {
public:
    using GraphPartitionFilter =
            thin_function<bool(const GraphPartition& graph_partition)>;
    ProfilingBasedSolver(std::unique_ptr<ProfilerBase> profiler);
    /*!
     * \note some graph partition (for example, graph partition without format
     * aware operators like conv, deconv, warp, resize etc.) will be filtered by
     * the GraphPartitionFilter, which can reduce the profiling time. */
    ProfilingBasedSolver(std::unique_ptr<ProfilerBase> profiler,
                         GraphPartitionFilter graph_partition_filter)
            : m_profiler{std::move(profiler)},
              m_graph_partition_filter{std::move(graph_partition_filter)} {}
    virtual ~ProfilingBasedSolver() = default;
    Solution solve(const Problem& problem) const override;
    virtual Solution do_solve(const Problem& problem) const = 0;

protected:
    std::unique_ptr<ProfilerBase> m_profiler;

private:
    GraphPartitionFilter m_graph_partition_filter;
};

/*!
 * \brief A solver that solves the layout selection problem using dynamic
 * programming algorithm (Markov decision process).
 */
class DynamicProgrammingSolver final : public ProfilingBasedSolver {
public:
    DynamicProgrammingSolver(std::unique_ptr<ProfilerBase> profiler)
            : ProfilingBasedSolver(std::move(profiler)){};
    DynamicProgrammingSolver(std::unique_ptr<ProfilerBase> profiler,
                             GraphPartitionFilter graph_partition_filter)
            : ProfilingBasedSolver(std::move(profiler),
                                   std::move(graph_partition_filter)){};
    ~DynamicProgrammingSolver() noexcept = default;
    Solution do_solve(const Problem& problem) const override;
    bool can_solve(const Problem& problem) const override;

private:
    class Impl;
};

/*!
 * \brief A layout transform pass, which convert the operator's format to the
 * optimal format using the results of the solver.
 */
class LayoutTransformPass final : public Pass {
public:
    const char* name() const override { return "layout assignment pass"; }
    void apply(OptState& opt) const override;
    LayoutTransformPass(std::unique_ptr<LayoutTransformContext> ctx,
                         std::unique_ptr<SolverBase> solver)
            : m_ctx{std::move(ctx)}, m_solver{std::move(solver)} {}

private:
    std::unique_ptr<LayoutTransformContext> m_ctx;
    std::unique_ptr<SolverBase> m_solver;
};

}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
