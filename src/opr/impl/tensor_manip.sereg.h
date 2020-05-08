/**
 * \file src/opr/impl/tensor_manip.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/internal/indexing_helper_sereg.h"
#include "megbrain/serialization/sereg.h"

#if MGB_ENABLE_FBS_SERIALIZATION
#include "megbrain/serialization/internal/mgb_cpp_opr_generated.h"
#endif

MGB_SEREG_GET_SUBTENSOR_OPR(Subtensor);
MGB_SEREG_MODIFY_SUBTENSOR_OPR(SetSubtensor);
MGB_SEREG_MODIFY_SUBTENSOR_OPR(IncrSubtensor);

namespace mgb {

namespace serialization {
    template<>
    struct OprMaker<opr::Concat, 0>: public OprMakerVariadic<opr::Concat>{};

    template<>
    struct OprMaker<opr::GetVarShape, 0>:
    public OprMakerVariadic<opr::GetVarShape>{};

    template<>
    struct OprLoadDumpImpl<opr::Split, 0> {
        using Split = opr::Split;
        using Options = Split::Options;
        using Method = Options::Method;

        static void dump(OprDumpContext &ctx,
                const cg::OperatorNodeBase &opr_) {
            auto &&opr = opr_.cast_final_safe<opr::Split>();
            auto &&opt = opr.options();
            mgb_assert(opt.method == Method::SPECIFY,
                    "only Spllit with SPECIFY output shapes can be serialized");
            ctx.write_param<megdnn::param::Axis>(opt.axis);
        }

        static cg::OperatorNodeBase* load(
                OprLoadContext &ctx, const cg::VarNodeArray &inputs,
                const OperatorNodeConfig &config) {
            auto param = ctx.read_param<megdnn::param::Axis>();
            opr::Split::Options opt;
            opt.method = Method::SPECIFY;
            opt.axis = param.axis;
            mgb_assert(inputs.size() > 1);
            opt.nr_part = inputs.size() - 1;
            opt.partition.resize(opt.nr_part);
            for (size_t i = 1; i < inputs.size(); ++ i)
                opt.partition[i - 1] = inputs[i];
            return Split::make(inputs[0], opt, config)[0].node()->owner_opr();
        }
    };

#if MGB_ENABLE_FBS_SERIALIZATION
    namespace fbs {
    template <>
    struct ParamConverter<opr::Dimshuffle::Param> {
        using FlatBufferType = param::Dimshuffle;
        static opr::Dimshuffle::Param to_param(const FlatBufferType* fb) {
            opr::Dimshuffle::Param param;
            param.ndim = fb->ndim();
            if (fb->pattern()) {
                param.pattern_len = fb->pattern()->size();
                mgb_assert(param.pattern_len <=
                           sizeof(param.pattern) / sizeof(param.pattern[0]));
                memcpy(param.pattern, fb->pattern()->data(),
                       sizeof(param.pattern[0]) * param.pattern_len);
            } else {
                param.pattern_len = 0;
            }
            return param;
        }
        static flatbuffers::Offset<FlatBufferType> to_flatbuffer(
                flatbuffers::FlatBufferBuilder& builder,
                const opr::Dimshuffle::Param& p) {
            return param::CreateDimshuffle(
                    builder, builder.CreateVector(p.pattern, p.pattern_len),
                    p.ndim);
        }
    };
    template <>
    struct ParamConverter<opr::AxisAddRemove::Param> {
        using FlatBufferType = param::AxisAddRemove;
        static opr::AxisAddRemove::Param to_param(const FlatBufferType* fb) {
            opr::AxisAddRemove::Param param;
            if (fb->desc()) {
                param.nr_desc = fb->desc()->size();
                for (uint32_t i = 0; i < param.nr_desc; i++) {
                    param.desc[i].axis = fb->desc()->Get(i)->axis();
                    param.desc[i].method =
                            static_cast<opr::AxisAddRemove::AxisDesc::Method>(
                                    fb->desc()->Get(i)->method());
                }
            } else {
                param.nr_desc = 0;
            }
            return param;
        }
        static flatbuffers::Offset<FlatBufferType> to_flatbuffer(flatbuffers::FlatBufferBuilder& builder, const opr::AxisAddRemove::Param& p) {
            std::vector<param::AxisDesc> desc(p.nr_desc);
            for (uint32_t i = 0; i < p.nr_desc; i++) {
                desc[i] = {static_cast<param::AxisDescMethod>(p.desc[i].method),
                           p.desc[i].axis.get_raw()};
            }
            return param::CreateAxisAddRemoveDirect(builder, &desc);
        }
    };
    }  // namespace fbs
#endif
} // namespace serialization


namespace opr {
    MGB_SEREG_OPR(Broadcast, 2);
    MGB_SEREG_OPR(Dimshuffle, 1);
    MGB_SEREG_OPR(AxisAddRemove, 1);
    MGB_SEREG_OPR(Concat, 0);
    using GetVarShapeV1 = opr::GetVarShape;
    MGB_SEREG_OPR(GetVarShapeV1, 0);
    using ReshapeV1 = opr::Reshape;
    MGB_SEREG_OPR(ReshapeV1, 2);

    cg::OperatorNodeBase* opr_shallow_copy_split(
            const serialization::OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr_, const VarNodeArray &inputs,
            const OperatorNodeConfig &config) {
        auto &&opr = opr_.cast_final_safe<Split>();
        auto option = opr.options();
        using Meth = Split::Options::Method;
        switch (option.method) {
            case Meth::CALLBACK:
                mgb_assert(inputs.size() == 1);
                break;
            case Meth::SPECIFY:
                mgb_assert(inputs.size() == 1 + option.partition.size());
                for (size_t i = 0; i < option.partition.size(); ++ i)
                    option.partition[i] = inputs[i + 1];
                break;
        }
        return Split::make(inputs[0], option, config).at(0).
            node()->owner_opr();
    }
    MGB_SEREG_OPR(Split, 0);
    MGB_REG_OPR_SHALLOW_COPY(Split, opr_shallow_copy_split);

    cg::OperatorNodeBase* opr_shallow_copy_param_pack_split(
            const serialization::OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr_, const VarNodeArray &inputs,
            const OperatorNodeConfig &config){
         auto &&opr = opr_.cast_final_safe<ParamPackSplit>();
         auto &&offsets = opr.get_offsets();
         auto &&shape = opr.get_output_shapes();

         return ParamPackSplit::make(inputs[0], offsets, shape, config).at(0).
             node()->owner_opr();
    }

    MGB_REG_OPR_SHALLOW_COPY(ParamPackSplit, opr_shallow_copy_param_pack_split);

    cg::OperatorNodeBase* opr_shallow_copy_param_pack_concat(
            const serialization::OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr_, const VarNodeArray &inputs,
            const OperatorNodeConfig &config){
         auto &&opr = opr_.cast_final_safe<ParamPackConcat>();
         auto &&offsets = opr.get_offsets();
         
         SymbolVarArray ivar{inputs.size() - 1};
         for (size_t i = 0; i < inputs.size() - 1; ++i)
             ivar[i] = inputs[i];
         return ParamPackConcat::make(ivar, inputs.back(), offsets, config).
             node()->owner_opr();
    }

    MGB_REG_OPR_SHALLOW_COPY(ParamPackConcat, opr_shallow_copy_param_pack_concat);
    MGB_SEREG_OPR(RelayoutFormat, 1);
    MGB_SEREG_OPR(WinogradFilterPreprocess, 1);
} // namespace opr

} // namespace mgb


// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

