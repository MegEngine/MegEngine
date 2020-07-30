/**
 * \file src/opr/include/megbrain/opr/internal/indexing_helper_sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megbrain/opr/internal/indexing_helper.h"
#include "megbrain/serialization/sereg.h"

#if MGB_ENABLE_FBS_SERIALIZATION
#include "megbrain/serialization/internal/mgb_cpp_opr_generated.h"
#endif

namespace mgb {
namespace serialization {

    struct IndexDescMaskDump {
        using IndexDesc = opr::indexing::IndexDesc;
        static constexpr uint32_t TAG = opr::param_tag::SUBTENSOR_INDEX_DESC;

        struct Item {
            int8_t axis;
            bool begin, end, step, idx;
        };
        uint8_t nr_item;
        Item items[TensorShape::MAX_NDIM];

        static IndexDescMaskDump from_index_desc(const IndexDesc &desc);

        //! get usable IndexDesc from this mask, given concrete input vars
        IndexDesc to_index_desc(
                cg::VarNodeArray::const_iterator inp_begin,
                cg::VarNodeArray::const_iterator inp_end) const;
    };

#if MGB_ENABLE_FBS_SERIALIZATION
    namespace fbs {
    template <>
    struct ParamConverter<IndexDescMaskDump> {
        using FlatBufferType = param::IndexDescMaskDump;
        static IndexDescMaskDump to_param(const FlatBufferType* fb) {
            IndexDescMaskDump param;
            if (!fb->items()) {
                param.nr_item = 0;
            } else {
                param.nr_item = fb->items()->size();
                mgb_assert(param.nr_item < TensorShape::MAX_NDIM);
                for (uint8_t i = 0; i < param.nr_item; i++) {
                    auto t = fb->items()->Get(i);
                    param.items[i] = {t->axis(), t->begin(), t->end(),
                                      t->step(), t->idx()};
                }
            }
            return param;
        }
        static flatbuffers::Offset<FlatBufferType> to_flatbuffer(
                flatbuffers::FlatBufferBuilder& builder,
                const IndexDescMaskDump& p) {
            std::vector<param::IndexDescMaskItem> items(p.nr_item);
            for (uint8_t i = 0; i < p.nr_item; i++) {
                auto& t = p.items[i];
                items[i] = {t.axis, t.begin, t.end, t.step, t.idx};
            }
            return param::CreateIndexDescMaskDumpDirect(builder, &items);
        }
    };
    }  // namespace fbs
#endif

    template<class Opr>
    struct GetSubtensorOprLoadDumpImpl {
        static void dump(OprDumpContext &ctx,
                const cg::OperatorNodeBase &opr_) {
            auto &&opr = opr_.cast_final_safe<Opr>();
            ctx.write_param(
                    IndexDescMaskDump::from_index_desc(opr.index_desc()));
        }

        static cg::OperatorNodeBase* load(
                OprLoadContext &ctx, const cg::VarNodeArray &inputs,
                const OperatorNodeConfig &config) {
            mgb_assert(inputs.size() >= 1);
            auto index_desc = ctx.read_param<IndexDescMaskDump>().
                to_index_desc(inputs.begin() + 1, inputs.end());
            return Opr::make(inputs[0], index_desc, config).node()->owner_opr();
        }
    };

    template<class Opr>
    struct ModifySubtensorOprLoadDumpImpl {
        static void dump(OprDumpContext &ctx,
                const cg::OperatorNodeBase &opr_) {
            auto &&opr = opr_.cast_final_safe<Opr>();
            mgb_assert(!opr.input_tensor_replacer(),
                    "can not dump opr with non-empty input_tensor_replacer()");
            ctx.write_param(
                    IndexDescMaskDump::from_index_desc(opr.index_desc()));
        }

        static cg::OperatorNodeBase* load(
                OprLoadContext &ctx, const cg::VarNodeArray &inputs,
                const OperatorNodeConfig &config) {
            mgb_assert(inputs.size() >= 2);
            auto index_desc = ctx.read_param<IndexDescMaskDump>().
                to_index_desc(inputs.begin() + 2, inputs.end());
            return Opr::make(inputs[0], inputs[1],
                    index_desc, config).node()->owner_opr();
        }
    };

    //! shallow copy impl for oprs that modify subtensor
    template<class Opr>
    cg::OperatorNodeBase* opr_shallow_copy_modify_subtensor(
            const serialization::OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr_, const VarNodeArray &inputs,
            const OperatorNodeConfig &config) {
        MGB_MARK_USED_VAR(ctx);
        auto &&opr = opr_.cast_final_safe<Opr>();
        auto desc_mask = IndexDescMaskDump::from_index_desc(opr.index_desc());
        auto new_desc = desc_mask.to_index_desc(
                inputs.begin() + 2, inputs.end());
        return Opr::make(inputs[0], inputs[1], new_desc, config,
                opr.input_tensor_replacer()).node()->owner_opr();
    }

} // namespace serialization
} // namespace mgb

/*!
 * \brief register sereg impls for get-subtensor oprs
 *
 * This macro must be invoked in global scope (not in any namespace)
 * \param _opr operator class name
 */
#define MGB_SEREG_GET_SUBTENSOR_OPR(_opr) \
namespace mgb { \
namespace serialization { \
    template<> \
    struct OprLoadDumpImpl<opr::_opr, 0>: \
        public GetSubtensorOprLoadDumpImpl<opr::_opr> \
    { \
    }; \
} \
namespace opr { \
    MGB_SEREG_OPR(_opr, 0); \
} \
}

/*!
 * \brief register sereg impls for modify-subtensor oprs
 *
 * This macro must be invoked in global scope (not in any namespace)
 * \param _opr operator class name
 */
#define MGB_SEREG_MODIFY_SUBTENSOR_OPR(_opr) \
namespace mgb { \
namespace serialization { \
    template<> \
    struct OprLoadDumpImpl<opr::_opr, 0>: \
        public ModifySubtensorOprLoadDumpImpl<opr::_opr> \
    { \
    }; \
} \
namespace opr { \
    MGB_SEREG_OPR(_opr, 0); \
    MGB_REG_OPR_SHALLOW_COPY( \
            _opr, serialization::opr_shallow_copy_modify_subtensor<_opr>); \
} \
}


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

