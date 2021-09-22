/**
 * \file
 * src/gopt/include/megbrain/gopt/layout_transform_context.h
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
#include "megbrain/gopt/reformat_manager.h"
#include "megbrain/gopt/subgraph_extractor.h"
#include "megbrain/plugin/opr_footprint.h"

namespace mgb {
namespace gopt {

/*!
 * \brief A structure that describe the data types and  tensor formats
 * configuration of the opr format
 */
struct OprTensorFormatsConfiguration {
    using OprFormat = opr::Convolution::Param::Format;
    static constexpr uint32_t FORMAT_NR_MEMBER =
            opr::Convolution::Param::FORMAT_NR_MEMBER;
    enum class OprFormatConfigID : uint32_t {
#define cb(fmt_) fmt_ = static_cast<uint32_t>(OprFormat::fmt_)
        cb(NCHW),
        cb(NHWC),
        cb(NHWCD4),
        cb(NCHW4),
        cb(NCHW8),
        cb(NCHW32),
        cb(NCHW88),
        cb(NCHW44),
        cb(NCHW44_DOT),
        cb(NCHW4_NCHW32),
        cb(NCHW32_NCHW4),
        cb(NCHW4_NCHW),
        cb(NCHW4_NHWC),
        cb(CHWN4),
        cb(NCHW64),
        NCHW44_HYBRID = FORMAT_NR_MEMBER,
        NCHW88_HYBRID = FORMAT_NR_MEMBER + 1,
        NCHW44_DOT_HYBRID = FORMAT_NR_MEMBER + 2,
    };
#undef cb
    using OprTensorFormatsDispatcher =
            thin_function<Maybe<OprTensorFormatsConfiguration>(
                    const cg::OperatorNodeBase*)>;
    Typeinfo* typeinfo;
    OprFormat opr_format;
    OprFormatConfigID config_id;
    SmallVector<DTypeEnum> input_dtypes;
    SmallVector<DTypeEnum> output_dtypes;
    SmallVector<TensorFormats> input_tensor_formats;
    SmallVector<TensorType> input_tensor_types;
    SmallVector<TensorFormats> output_tensor_formats;
    static OprTensorFormatsDispatcher* find_dispatcher_by_type_format(
            Typeinfo* type, OprFormatConfigID config_id);
    static OprFormat safe_cast_to_opr_format(OprFormatConfigID config_id) {
        mgb_assert(static_cast<uint32_t>(config_id) < FORMAT_NR_MEMBER);
        return static_cast<OprFormat>(static_cast<uint32_t>(config_id));
    }
};

/*!
 * \brief A structure that describes the global layout transform problem
 */
class LayoutTransformContext {
public:
    using OprList = SubGraphExtractor::OprList;
    using OprFormat = OprTensorFormatsConfiguration::OprFormat;
    using OprFormatConfigID = OprTensorFormatsConfiguration::OprFormatConfigID;
    using OprTensorFormatsDispatcher =
            OprTensorFormatsConfiguration::OprTensorFormatsDispatcher;
    using OprConfigTrait = ThinHashMap<
            Typeinfo*, ThinHashMap<OprFormatConfigID, OprTensorFormatsDispatcher*>>;
    using Target = GraphTuningOptions::Target;
    using ReformatAttribute = ReformatManager::ReformatKey::Attribute;
    struct Attribute {
        OprFormatConfigID base_config_id;   /// the base opr format indicates that the
                                            /// network to be optimized is constructed
                                            /// in the base opr format, i.e. all the
                                            /// format aware operators (conv, conv_bias,
                                            /// deconv, pooling etc.) are built in
                                            /// this format.
        TensorFormats base_tensor_formats;  /// the base tensor format indicates that
                                            /// all the format agnostic operators
                                            /// (like elemwise, elemwise multi type,
                                            /// typecvt etc.) are built in the base
                                            /// tensor format.
        Target target;                      /// target which indicates the device type
        ReformatAttribute reformat_attribute =
                ReformatAttribute::DEFAULT;  /// additional reformat attribute,
                                             /// which indicates whether to pad
                                             /// nhwc layout automatically or to
                                             /// enable nhwcd4 format on opencl
                                             /// platform to use image object
    };
    LayoutTransformContext() = delete;
    LayoutTransformContext(
            OprList opr_list, SmallVector<TensorFormats> available_tensor_formats,
            Attribute attribute)
            : m_opr_list{std::move(opr_list)},
              m_available_tensor_formats{std::move(available_tensor_formats)},
              m_attribute{attribute} {}
    LayoutTransformContext(
            OprList opr_list, SmallVector<TensorFormats> available_tensor_formats,
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
     * \param config_id op format configuration id which is going to be enabled
     * in the layout transform problem
     */
    LayoutTransformContext& add_opr_config(Typeinfo* opr, OprFormatConfigID config_id);
    /*!
     * \brief add a vector of op format configurations for a particular operator
     * type
     * \param opr runtime typeinfo of operator
     * \param config_ids ids of op format configurations which are enabled in
     * the layout transform problem
     */
    LayoutTransformContext& add_opr_config(
            Typeinfo* opr, SmallVector<OprFormatConfigID> config_ids);
    static std::unique_ptr<LayoutTransformContext> make(
            Target target = Target::UNSPEC,
            OprFormatConfigID base_config_id = OprFormatConfigID::NCHW,
            TensorFormats base_tensor_format = TensorFormats::NCHW);

private:
    OprList m_opr_list;  /// supported operator list
    SmallVector<TensorFormats>
            m_available_tensor_formats;  /// the available tensor formats, used
                                         /// for format agnostic operators (like
                                         /// elemwise, elemwise multi type,
                                         /// typecvt, etc.
    OprConfigTrait m_opr_configs;        /// the available opr format configurations,
                                         /// used for format aware operators (like
                                         /// conv, deconv, conv_bias, etc.
    Attribute m_attribute;  /// the extra attributes to describe the problem
};

class Problem {
public:
    using OprFormat = OprTensorFormatsConfiguration::OprFormat;
    using OprFormatConfigID = OprTensorFormatsConfiguration::OprFormatConfigID;
    using OprTensorFormatsDispatcher =
            OprTensorFormatsConfiguration::OprTensorFormatsDispatcher;
    using OprConfigTrait = LayoutTransformContext::OprConfigTrait;
    using Attribute = LayoutTransformContext::Attribute;

    Problem(const GraphPartition& graph_partition, const LayoutTransformContext& ctx)
            : m_graph_partition{graph_partition}, m_ctx{ctx} {}
    ~Problem() noexcept = default;

    const GraphPartition& graph_partition() const { return m_graph_partition; }
    const OprConfigTrait& opr_configs() const { return m_ctx.opr_configs(); }
    const SmallVector<TensorFormats>& available_tensor_formats() const {
        return m_ctx.available_tensor_formats();
    }
    TensorFormats base_format() const { return m_ctx.attribute().base_tensor_formats; }
    Attribute attribute() const { return m_ctx.attribute(); }
    /*!
     * \brief return the tensor formats configuration of an operator in the
     * default op format
     */
    OprTensorFormatsConfiguration base_config(const cg::OperatorNodeBase* opr) const {
        auto _ = OprTensorFormatsConfiguration::find_dispatcher_by_type_format(
                opr->dyn_typeinfo(), m_ctx.attribute().base_config_id);
        auto rst = (*_)(opr);
        if (rst.valid())
            return rst.val();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.config_id = m_ctx.attribute().base_config_id;
        config.opr_format = OprTensorFormatsConfiguration::safe_cast_to_opr_format(
                config.config_id);
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
}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
