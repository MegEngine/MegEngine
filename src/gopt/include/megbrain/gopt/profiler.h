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
#include "megbrain/utils/infile_persistent_cache.h"

namespace mgb {
namespace gopt {

class Problem;
class CachedProfiler;

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

    ProfilerBase() = default;
    
    virtual ~ProfilerBase() = default;

    virtual ProfilingResult profile(const Problem& problem) const = 0;

    ProfilerBase& set_opr_filter(const OprFilter& opr_filter) {
        m_opr_filter = opr_filter;
        return *this;
    }

    ProfilerBase& set_var_node_filter(const VarNodeFilter& var_node_filter) {
        m_var_node_filter = var_node_filter;
        return *this;
    }

    static std::unique_ptr<ProfilerBase> make_profiler();
    static std::unique_ptr<ProfilerBase> make_cached_profiler(
            const char* path = nullptr);

protected:
    OprFilter m_opr_filter;
    VarNodeFilter m_var_node_filter;
};


/*! \brief A default profiler impl
 */
class ProfilerImpl : public ProfilerBase {
public:
    ProfilerImpl(int runs = 10, float opr_threshold = 2.f,
                 float var_node_threshold = 2.f);
    ~ProfilerImpl() = default;
    ProfilingResult profile(const Problem& problem) const override;

protected:
    static constexpr float PROFILE_TIME_OUT = 1e7;
    using ReformatKey = ReformatManager::ReformatKey;
    using ReformatAttribute = ReformatKey::Attribute;
    /*!
     * \brief profile opr format agnostic operators (like elemwise, elemwise
     * multi type, typecvt etc.)
     *
     * \param opr pointer to the operator node to be profiled
     * \param base_format the original tensor format of the operator node.
     * \param available_tensor_formats the available tensor formats
     * \return the operator node record
     */
    OperatorNodeRecord profile_operator(
            const OperatorNodeBase* opr, TensorFormats base_format,
            const SmallVector<TensorFormats>& available_tensor_formats,
            ReformatAttribute extra_attribute =
                    ReformatAttribute::DEFAULT) const;
    /*!
     * \brief prfile opr format agnostic operators (like elemwise, elemwise multi type, typecvt etc.)
     *
     * \param opr pointer to the operator to be profiled
     * \param base_format the original tensor format of the operator node.
     * \param tensor_format the tensor format to be profiled
     * \param extra_attribute identify whether to use image object for OpenCL or automatically padding nhwc layout
     * \return elapsed time of operator in the given tensor format configuration
     */
    virtual float profile_operator(
            const OperatorNodeBase* opr, TensorFormats base_format,
            TensorFormats tensor_format,
            ReformatAttribute extra_attribute =
                    ReformatAttribute::DEFAULT) const;
    /*!
     * \brief profile opr format aware operators (like conv, deconv, conv_bias,
     * etc.)
     *
     * \param opr pointer to the operator node to be profiled
     * \param base_config the tensor formats configuration of base opr format
     * \param config all the available configuration
     * \return the operator node record
     */
    OperatorNodeRecord profile_operator(
            const OperatorNodeBase* opr,
            const OprTensorFormatsConfiguration& base_config,
            const SmallVector<OprTensorFormatsConfiguration>& available_configs,
            ReformatAttribute extra_attribute =
                    ReformatAttribute::DEFAULT) const;
    /*!
     * \brief prfile opr format aware operators (like conv, deconv, conv_bias, resize, warp etc.)
     *
     * \param opr pointer to the operator to be profiled
     * \param base_config the original opr format configuration of the operator node, 
     * \param config the opr format configuration to be profiled
     * \param extra_attribute identify whether to use image object for OpenCL or automatically padding nhwc layout
     * \return elapsed time of operator in the given opr format configuration
     */
    virtual float profile_operator(const OperatorNodeBase* opr,
                           const OprTensorFormatsConfiguration& base_config,
                           const OprTensorFormatsConfiguration& config,
                           ReformatAttribute extra_attribute =
                                   ReformatAttribute::DEFAULT) const;
    /*!
     * \brief profile layout transform of the var node
     *
     * \param var pointer to the var node to be profiled
     * \param base_format the original tensor formats in which the var node is
     * stored 
     * \param available_tensor_formats the available tensor formats
     * \param extra_attribute the extra attributes (options) of the problem
     * \return the var node record
     */
    VarNodeRecord profile_var_node(
            const VarNode* var, TensorFormats base_format,
            const SmallVector<TensorFormats>& available_tensor_formats,
            ReformatAttribute extra_attribute =
                    ReformatAttribute::DEFAULT) const;
    /*!
     * \brief profile layout transform of the var node
     *
     * \param var pointer to the var node to be profiled
     * \param base_format the original tensor formats in which the var node is
     * stored
     * \param key type of ReformatKey, identify the information/attributes of the layout transoform
     * \return elapsed time of the layout transform
     */
    virtual float profile_var_node(const VarNode* var,
                                   TensorFormats base_format,
                                   const ReformatKey& key) const;
    OprFootprint m_opr_footprint;
    float m_opr_threshold;  /// a threshold, when the computation of the newly
                            /// created operator that is built in some opr
                            /// format configuration is as greater as
                            /// m_opr_threshold times of the original operator,
                            /// the opr format configuration will be skipped
                            /// (i.e. the cost is infinite)
    float m_var_node_threshold;  /// a threshold, when the memory footprint of
                                 /// the layout transform of the var node is as
                                 /// larger as m_var_node_threshold as the var
                                 /// node itself, the layout transform will be
                                 /// skipped (i.e. the cost is infinite)
    int m_runs;                  /// sample times of the profiler
};

/*!
 * \brief a ProfilerCache that manages the profiling results of operator in
 * different layouts and of layout transform of var nodes.
 */
class ProfilerCache : public NonCopyableObj {
    ProfilerCache() : m_impl{std::make_unique<InMemoryPersistentCache>()} {};

public:
    using ReformatKey = ReformatManager::ReformatKey;
    using ReformatAttribute = ReformatKey::Attribute;
    using OprFormat = ProfilerBase::OprFormat;
    class Key final : public NonCopyableObj {
        std::string m_blob_storage;
        std::string m_category;

        struct OprKey {
            const OperatorNodeBase* opr;
            OprFormat opr_format;
            ReformatAttribute extra_attribute;
        };

        struct VarKey {
            const VarNode* var;
            ReformatKey key;
        };

        union KeyImpl {
            OprKey opr_key;
            VarKey var_key;

            KeyImpl() { std::memset(this, 0, sizeof(KeyImpl)); }
        };

        KeyImpl m_key_impl;

        void build_blob_from_opr();
        void build_blob_from_var();
        void build_category(CompNode cn);

    public:
        Key(const OperatorNodeBase* opr, OprFormat opr_format,
            ReformatAttribute extra_attribute = ReformatAttribute::DEFAULT) {
            m_key_impl.opr_key = {opr, opr_format, extra_attribute};
            build_blob_from_opr();
            mgb_assert(
                    opr->node_prop().contain(
                            cg::OperatorNodeProp::Flag::SINGLE_COMP_NODE),
                    "operator with multiple comp node is not supported(opr:%s)",
                    opr->cname());
            // here, we assume that the operator to be profiled has only one
            // comp node
            build_category(opr->output(0)->comp_node());
        }

        Key(const VarNode* var, ReformatKey key) {
            m_key_impl.var_key = {var, key};
            build_blob_from_var();
            build_category(var->comp_node());
        }

        const std::string& category() const;
        PersistentCache::Blob blob() const;
    };

    using Result = float;

public:
    static ProfilerCache& inst();

    ProfilerCache& set_impl(std::unique_ptr<PersistentCache> impl);

    void dump_cache(const char* path);

    Maybe<Result> get(const Key& key);

    void put(const Key& key, Result& result);

private:
    std::unique_ptr<PersistentCache> m_impl;
};

class CachedProfiler final : public ProfilerImpl {
public:
    CachedProfiler(const char* path = nullptr, int runs = 10,
                   float opr_threshold = 2.f, float var_node_threshold = 2.f);
    ProfilingResult profile(const Problem& problem) const override;

private:
    float profile_operator(const OperatorNodeBase* opr,
                           TensorFormats base_format,
                           TensorFormats tensor_format,
                           ReformatAttribute extra_attribute =
                                   ReformatAttribute::DEFAULT) const override;
    float profile_operator(const OperatorNodeBase* opr,
                           const OprTensorFormatsConfiguration& base_config,
                           const OprTensorFormatsConfiguration& config,
                           ReformatAttribute extra_attribute =
                                   ReformatAttribute::DEFAULT) const override;
    float profile_var_node(const VarNode* var, TensorFormats base_format,
                           const ReformatKey& key) const override;
    const char* m_path;
};

}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
