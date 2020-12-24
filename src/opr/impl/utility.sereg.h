/**
 * \file src/opr/impl/utility.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/utility.h"
#include "megbrain/serialization/sereg.h"

#if MGB_ENABLE_FBS_SERIALIZATION
#include "megbrain/serialization/internal/mgb_cpp_opr_generated.h"
#endif

namespace mgb {

namespace serialization {
    template<>
    struct OprLoadDumpImpl<opr::AssertEqual, 0> {
        static void dump(OprDumpContext &ctx,
                const cg::OperatorNodeBase &opr_) {
            auto &&opr = opr_.cast_final_safe<opr::AssertEqual>();
            ctx.write_param(opr.param());
        }

        static cg::OperatorNodeBase* load(
                OprLoadContext &ctx, const cg::VarNodeArray &inputs,
                const OperatorNodeConfig &config) {
            auto param = ctx.read_param<megdnn::param::AssertEqual>();
            SymbolVar out;
            if (inputs.size() == 2) {
                // from python
                out = opr::AssertEqual::make(
                        inputs[0], inputs[1], param, config);
            } else {
                // from sereg or copy
                mgb_assert(inputs.size() == 3);
                out = opr::AssertEqual::make(
                        inputs[0], inputs[1], inputs[2], param, config);
            }
            return out.node()->owner_opr();
        }
    };

#if !MGB_BUILD_SLIM_SERVING
    template <>
    struct OprLoadDumpImpl<opr::VirtualDep, 0> {
        static void dump(OprDumpContext& ctx,
                         const cg::OperatorNodeBase& opr_) {}

        static cg::OperatorNodeBase* load(OprLoadContext& ctx,
                                          const cg::VarNodeArray& inputs,
                                          const OperatorNodeConfig& config) {
            return opr::VirtualDep::make(to_symbol_var_array(inputs), config)
                    .node()
                    ->owner_opr();
        }
    };

#if MGB_ENABLE_FBS_SERIALIZATION
    namespace fbs {
    template <>
    struct ParamConverter<opr::Sleep::Param> {
        using FlatBufferType = param::MGBSleep;
        static opr::Sleep::Param to_param(const param::MGBSleep* fb) {
            return {fb->seconds(), {fb->device(), fb->host()}};
        }
        static flatbuffers::Offset<param::MGBSleep> to_flatbuffer(
                flatbuffers::FlatBufferBuilder& builder,
                const opr::Sleep::Param& p) {
            return param::CreateMGBSleep(builder, p.type.device, p.type.host,
                                         p.seconds);
        }
    };
    }  // namespace fbs
#endif
#endif
} // namespace serialization

namespace opr {

    MGB_SEREG_OPR(MarkDynamicVar, 1);
    MGB_SEREG_OPR(MarkNoBroadcastElemwise, 1);
    MGB_SEREG_OPR(Identity, 1);
    MGB_SEREG_OPR(AssertEqual, 0);

#if MGB_ENABLE_GRAD
    MGB_SEREG_OPR(VirtualGrad, 2);

    cg::OperatorNodeBase* opr_shallow_copy_set_grad(
            const serialization::OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr_, const VarNodeArray &inputs,
            const OperatorNodeConfig &config) {
        mgb_assert(inputs.size() == 1);
        auto &&opr = opr_.cast_final_safe<SetGrad>();
        return SetGrad::make(inputs[0], opr.grad_getter(), config).
            node()->owner_opr();
    }
    MGB_REG_OPR_SHALLOW_COPY(SetGrad, opr_shallow_copy_set_grad);

    cg::OperatorNodeBase* opr_shallow_copy_virtual_loss(
            const serialization::OprShallowCopyContext& ctx,
            const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        return inputs[0]->owner_graph()->insert_opr(
                std::make_unique<VirtualLoss>(inputs, config));
    }
    MGB_REG_OPR_SHALLOW_COPY(VirtualLoss, opr_shallow_copy_virtual_loss);

    cg::OperatorNodeBase* opr_shallow_copy_invalid_grad(
            const serialization::OprShallowCopyContext& ctx,
            const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        mgb_assert(inputs.size() == 1);
        auto&& opr = opr_.cast_final_safe<InvalidGrad>();
        return inputs[0]->owner_opr()->owner_graph()->insert_opr(
                std::make_unique<InvalidGrad>(inputs[0], opr.grad_opr(),
                                              opr.inp_idx()));
    }
    MGB_REG_OPR_SHALLOW_COPY(InvalidGrad, opr_shallow_copy_invalid_grad)

#endif

    cg::OperatorNodeBase* opr_shallow_copy_callback_injector(
            const serialization::OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr_, const VarNodeArray &inputs,
            const OperatorNodeConfig &config) {
        auto &&opr = opr_.cast_final_safe<CallbackInjector>();
        return CallbackInjector::make(cg::to_symbol_var_array(inputs), opr.param(), config).
            node()->owner_opr();
    }
    MGB_REG_OPR_SHALLOW_COPY(CallbackInjector,
            opr_shallow_copy_callback_injector);

    cg::OperatorNodeBase* opr_shallow_copy_require_input_dynamic_storage(
            const serialization::OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr_, const VarNodeArray &inputs,
            const OperatorNodeConfig &config) {
        mgb_assert(inputs.size() == 1);
        return RequireInputDynamicStorage::make(inputs[0], config).
            node()->owner_opr();
    }
    MGB_REG_OPR_SHALLOW_COPY(RequireInputDynamicStorage,
            opr_shallow_copy_require_input_dynamic_storage);

#if !MGB_BUILD_SLIM_SERVING
    MGB_SEREG_OPR(Sleep, 1);
    MGB_SEREG_OPR(VirtualDep, 0);
#endif

    MGB_SEREG_OPR(PersistentOutputStorage, 1);

    cg::OperatorNodeBase* opr_shallow_copy_shape_hint(
            const serialization::OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr_, const VarNodeArray &inputs,
            const OperatorNodeConfig &config) {
        auto &&opr = opr_.cast_final_safe<ShapeHint>();
        mgb_assert(inputs.size() == 1);
        return ShapeHint::make(inputs[0], opr.shape(), opr.is_const(), config)
                .node()->owner_opr();
    }
    MGB_REG_OPR_SHALLOW_COPY(ShapeHint, opr_shallow_copy_shape_hint);
} // namespace opr
} // namespace mgb


// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
