/**
 * \file src/opr/impl/io.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/io.h"
#include "megbrain/serialization/sereg.h"
#include "megbrain/comp_node_env.h"

#if MGB_ENABLE_FBS_SERIALIZATION
#include "megbrain/serialization/internal/mgb_cpp_opr_generated.h"
#endif

namespace mgb {

namespace serialization {

#if MGB_ENABLE_FBS_SERIALIZATION
    namespace fbs {
    template <>
    struct ParamConverter<opr::Host2DeviceCopy::Param> {
        using FlatBufferType = param::Host2DeviceCopy;
        static opr::Host2DeviceCopy::Param to_param(const FlatBufferType* fb) {
            return {fb->enable_value_infer(), fb->dump_default_value(),
                    fb->allow_cpu_mem_fwd()};
        }
        static flatbuffers::Offset<FlatBufferType> to_flatbuffer(
                flatbuffers::FlatBufferBuilder& builder,
                const opr::Host2DeviceCopy::Param& p) {
            return param::CreateHost2DeviceCopy(builder, p.enable_value_infer,
                                                p.dump_default_value,
                                                p.allow_cpu_mem_fwd);
        }
    };
    }
#endif

    template<>
    struct OprLoadDumpImpl<opr::Host2DeviceCopy, 0> {
        using Opr = opr::Host2DeviceCopy;

        static void dump(OprDumpContext &ctx,
                const cg::OperatorNodeBase &opr_) {
            auto &&opr = opr_.cast_final_safe<Opr>();
            ctx.write_param(opr.param());

            using Meth = OprDumpContext::TensorWriteMethod;
            ctx.dump_tensor(
                    opr.name(),
                    *opr.host_data(), opr.param().dump_default_value ?
                    Meth::VALUE_INPUT : Meth::META_INPUT);
        }

        static cg::OperatorNodeBase* load(
                OprLoadContext &ctx, const cg::VarNodeArray &inputs,
                const OperatorNodeConfig &config) {
            mgb_assert(inputs.empty());
            auto param = ctx.read_param<Opr::Param>();
            auto tensor = ctx.load_tensor();
            return Opr::make(
                    ctx.graph(), tensor, param, config).node()->owner_opr();
        }

    };

    template<class Opr>
    struct SharedDeviceTensorLoadDump {

        static void dump(OprDumpContext &ctx,
                const cg::OperatorNodeBase &opr_) {
            using Meth = OprDumpContext::TensorWriteMethod;
            auto &&opr = opr_.cast_final_safe<Opr>();
            HostTensorND val;
            val.copy_from(opr.get_dev_tensor()).sync();
            ctx.dump_tensor(opr.name(), val, Meth::VALUE_SHARED);
            // Note that we don't persist opr.m_const_value, because it does not
            // affect correctness, and SharedDeviceTensor will be bundled
            // together as MultipleDeviceTensorHolder in optimize_for_inference
            // before being dumped.
        }

        static cg::OperatorNodeBase* load(
                OprLoadContext &ctx, const cg::VarNodeArray &inputs,
                const OperatorNodeConfig &config) {
            mgb_assert(inputs.empty());
            auto val = ctx.load_tensor_shared();
            return Opr::make(ctx.graph(), val, config).node()->owner_opr();
        }
    };

    template <>
    struct OprLoadDumpImpl<opr::SharedDeviceTensor, 0>
            : public SharedDeviceTensorLoadDump<opr::SharedDeviceTensor> {};
    template <>
    struct OprLoadDumpImpl<opr::VolatileSharedDeviceTensor, 0>
            : public SharedDeviceTensorLoadDump<
                      opr::VolatileSharedDeviceTensor> {};

    template <>
    struct OprLoadDumpImpl<opr::SharedDeviceTensorWithFormat, 0> {
        using Opr = opr::SharedDeviceTensorWithFormat;

        static void dump(OprDumpContext& ctx,
                         const cg::OperatorNodeBase& opr_) {
            using Meth = OprDumpContext::TensorWriteMethod;
            auto&& opr = opr_.cast_final_safe<Opr>();
            HostTensorND val;
            val.copy_from(opr.get_dev_tensor()).sync();
            ctx.dump_tensor({}, val, Meth::VALUE_ANONYMOUS);
            auto param_bin = opr.get_dev_tensor().format().serialize();
            ctx.dump_buf_with_len(param_bin.data(), param_bin.size());
        }

        static cg::OperatorNodeBase* load(OprLoadContext& ctx,
                                          const cg::VarNodeArray& inputs,
                                          const OperatorNodeConfig& config) {
            mgb_assert(inputs.empty());
            auto val = ctx.load_tensor();
            auto handle = MegDNNHandle::get(
                                  CompNodeEnv::from_comp_node(val->comp_node()))
                                  .handle();
            auto format =
                    TensorFormat::deserialize(ctx.load_buf_with_len(), handle);
            TensorLayout layout_with_format = {val->shape(), val->dtype(),
                                               format};
            auto dev_val = std::make_shared<DeviceTensorND>(val->comp_node(),
                                                            layout_with_format);
            dev_val->copy_from_fixlayout(*val);
            auto out_var = Opr::make(ctx.graph(), dev_val, config);
            dev_val->sync();
            return out_var.node()->owner_opr();
        }
    };

    template<>
    struct OprLoadDumpImpl<opr::ImmutableTensor, 0> {
        using Opr = opr::ImmutableTensor;

        static void dump(OprDumpContext &ctx,
                const cg::OperatorNodeBase &opr_) {
            using Meth = OprDumpContext::TensorWriteMethod;
            auto &&opr = opr_.cast_final_safe<Opr>();
            ctx.dump_tensor({}, HostTensorND{}.copy_from(opr.value()).sync(),
                    Meth::VALUE_ANONYMOUS);
        }

        static cg::OperatorNodeBase* load(
                OprLoadContext &ctx, const cg::VarNodeArray &inputs,
                const OperatorNodeConfig &config) {
            mgb_assert(inputs.empty());
            auto val = ctx.load_tensor();
            return Opr::make(ctx.graph(), *val, config).node()->owner_opr();
        }
    };

    template <>
    struct OprLoadDumpImpl<opr::MultipleDeviceTensorHolder, 0> {
        using Opr = opr::MultipleDeviceTensorHolder;

        static void dump(OprDumpContext& ctx,
                         const cg::OperatorNodeBase& opr_) {
            using Meth = OprDumpContext::TensorWriteMethod;
            auto&& opr = opr_.cast_final_safe<Opr>();
            uint32_t nr_val = opr.values().size();
            ctx.dump_buf_with_len(&nr_val, sizeof(nr_val));
            for (uint32_t i = 0; i < nr_val; ++i) {
                HostTensorND val;
                val.copy_from(*opr.values()[i]).sync();
                ctx.dump_tensor(opr.output(i)->name(), val, Meth::VALUE_SHARED);
            }
        }

        static cg::OperatorNodeBase* load(OprLoadContext& ctx,
                                          const cg::VarNodeArray& inputs,
                                          const OperatorNodeConfig& config) {
            mgb_assert(inputs.empty());
            uint32_t nr;
            {
                auto t = ctx.load_buf_with_len();
                mgb_assert(t.size() == sizeof(nr));
                memcpy(&nr, t.data(), sizeof(nr));
            }
            Opr::ValueArray values(nr);
            for (auto&& i : values) {
                i = ctx.load_tensor_shared();
            }
            return Opr::make(ctx.graph(), std::move(values), config)[0]
                    .node()
                    ->owner_opr();
        }
    };

    template <>
    struct OprLoadDumpImpl<opr::MultipleDeviceTensorWithFormatHolder, 0> {
        using Opr = opr::MultipleDeviceTensorWithFormatHolder;

        static void dump(OprDumpContext& ctx,
                         const cg::OperatorNodeBase& opr_) {
            using Meth = OprDumpContext::TensorWriteMethod;
            auto&& opr = opr_.cast_final_safe<Opr>();
            uint32_t nr_val = opr.values().size();
            ctx.dump_buf_with_len(&nr_val, sizeof(nr_val));
            for (uint32_t i = 0; i < nr_val; ++i) {
                HostTensorND val;
                auto value = *opr.values()[i];
                val.copy_from(value).sync();
                ctx.dump_tensor(opr.output(i)->name(), val, Meth::VALUE_SHARED);
                auto param_bin = value.format().serialize();
                ctx.dump_buf_with_len(param_bin.data(), param_bin.size());
            }
        }

        static cg::OperatorNodeBase* load(OprLoadContext& ctx,
                                          const cg::VarNodeArray& inputs,
                                          const OperatorNodeConfig& config) {
            mgb_assert(inputs.empty());
            uint32_t nr;
            {
                auto t = ctx.load_buf_with_len();
                mgb_assert(t.size() == sizeof(nr));
                memcpy(&nr, t.data(), sizeof(nr));
            }
            Opr::ValueArray values(nr);
            for (auto&& i : values) {
                i = ctx.load_tensor_shared();
                //! set tensor format
                auto handle = MegDNNHandle::get(CompNodeEnv::from_comp_node(
                                                        i->comp_node()))
                                      .handle();
                auto format = TensorFormat::deserialize(ctx.load_buf_with_len(),
                                                        handle);
                DeviceTensorStorage storage(i->comp_node());
                TensorLayout layout_with_format{i->layout(), i->layout().dtype,
                                                format};

                auto size = layout_with_format.span().dist_byte();
                storage.ensure_size(size);
                if (i->storage().comp_node().mem_node() ==
                    CompNode::default_cpu().mem_node()) {
                    mgb_assert(i->storage().ptr(),
                               "storage should not be nullptr if mem_node is "
                               "default_cpu");
                    HostTensorND src{i->storage().comp_node(),
                                     layout_with_format};
                    src.copy_from_fixlayout(*i).sync();
                    *i = DeviceTensorND::make_proxy(src);
                } else {
                    i->reset(storage, layout_with_format);
                }
            }
            return Opr::make(ctx.graph(), std::move(values), config)[0]
                    .node()
                    ->owner_opr();
        }
    };


} // namespace serialization

namespace opr {

    cg::OperatorNodeBase* opr_shallow_copy_h2d(
            const serialization::OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr_, const VarNodeArray &inputs,
            const OperatorNodeConfig &config) {
        mgb_assert(inputs.empty());
        auto &&opr = opr_.cast_final_safe<Host2DeviceCopy>();
        return Host2DeviceCopy::make(
                *ctx.owner_graph(opr, inputs),
                opr.host_data(), opr.param(), config).
            node()->owner_opr();
    }

    template<class Opr>
    cg::OperatorNodeBase* opr_shallow_copy_shared_device_tensor(
            const serialization::OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr_, const VarNodeArray &inputs,
            const OperatorNodeConfig &config) {
        mgb_assert(inputs.empty());
        auto &&opr = opr_.cast_final_safe<Opr>();
        return Opr::make(*ctx.owner_graph(opr, inputs), opr.dev_data(),
                         opr.const_value(), config)
                .node()
                ->owner_opr();
    }

    cg::OperatorNodeBase* opr_shallow_copy_immutable_tensor(
            const serialization::OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr_, const VarNodeArray &inputs,
            const OperatorNodeConfig &config) {
        mgb_assert(inputs.empty());
        auto &&opr = opr_.cast_final_safe<ImmutableTensor>();
        auto graph = ctx.owner_graph(opr, inputs);
        return opr.shallow_copy(*graph, config).node()->owner_opr();
    }

    MGB_SEREG_OPR(Host2DeviceCopy, 0);
    MGB_REG_OPR_SHALLOW_COPY(Host2DeviceCopy, opr_shallow_copy_h2d);

    MGB_SEREG_OPR(SharedDeviceTensor, 0);
    MGB_REG_OPR_SHALLOW_COPY(SharedDeviceTensor,
            opr_shallow_copy_shared_device_tensor<SharedDeviceTensor>);

    MGB_SEREG_OPR(SharedDeviceTensorWithFormat, 0);

    MGB_SEREG_OPR(VolatileSharedDeviceTensor, 0);
    MGB_REG_OPR_SHALLOW_COPY(
            VolatileSharedDeviceTensor,
            opr_shallow_copy_shared_device_tensor<VolatileSharedDeviceTensor>);

    MGB_SEREG_OPR(ImmutableTensor, 0);
    MGB_REG_OPR_SHALLOW_COPY(ImmutableTensor,
            opr_shallow_copy_immutable_tensor);

    MGB_SEREG_OPR(Copy, 1);
    MGB_SEREG_OPR(MultipleDeviceTensorHolder, 0);
    MGB_SEREG_OPR(MultipleDeviceTensorWithFormatHolder, 0);

} // namespace opr
} // namespace mgb


// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
