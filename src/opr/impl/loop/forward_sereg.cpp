/**
 * \file src/opr/impl/loop/forward_sereg.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./impl.h"
#include "./forward_sereg.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/serialization/opr_load_dump.h"
#include "megbrain/serialization/opr_shallow_copy.h"
#include "megbrain/opr/internal/param_tag_defs.h"

using namespace mgb;
using namespace mgb::opr::intl;
using namespace mgb::serialization;

namespace {

    class LoopDumpContext: public UserDataContainer::UserData {
        MGB_TYPEINFO_OBJ_DECL;
        public:

            ThinHashMap<VarNode*, size_t> ogvar2inpidx;

            static LoopDumpContext& from_dump_ctx(OprDumpContext &ctx) {
                auto ret = ctx.config().user_data->get_user_data<
                    LoopDumpContext>();
                mgb_assert(ret.second);
                return *ret.first[ret.second - 1];
            }


    };
    class LoopLoadContext: public UserDataContainer::UserData {
        MGB_TYPEINFO_OBJ_DECL;
        public:

            const VarNodeArray &input_vars;
            opr::Loop::Desc &desc;

            LoopLoadContext(const VarNodeArray &input_vars_,
                    opr::Loop::Desc &desc_):
                input_vars{input_vars_}, desc{desc_}
            {}

            static LoopLoadContext& from_load_ctx(OprLoadContext &ctx) {
                auto ret = ctx.config().user_data->get_user_data<
                    LoopLoadContext>();
                mgb_assert(ret.second);
                return *ret.first[ret.second - 1];
            }


    };

    MGB_TYPEINFO_OBJ_IMPL(LoopDumpContext);
    MGB_TYPEINFO_OBJ_IMPL(LoopLoadContext);

} // anonymous namespace

namespace mgb {
namespace opr {
namespace intl {

    //! use LoopSerializer because it is friend of LoopImpl
    class LoopSerializer {
        using InputMaker = LoopImpl::InputMaker;
        using CounterProvider = LoopImpl::DescImplBase::CounterProvider;

        struct LoopParam {
            static constexpr uint32_t TAG = opr::param_tag::LOOP;
            Loop::Param opr_param;
            uint64_t cond_var_id;
        };

        struct InputMakerParam {
            static constexpr uint32_t TAG = opr::param_tag::LOOP_INPUT_MAKER;
            bool has_assign;
            uint64_t ogvar_id;  //! id of proxied var in owner graph
        };

        struct OutputListEntry {
            uint64_t subvar_id;
            LoopImpl::Desc::OutputMode mode;
        } MGB_PACKED;

        struct AssignListEntry {
            uint64_t dst_id, src_id;
        };

        static void dump_loop(
                OprDumpContext &ctx, const cg::OperatorNodeBase &opr);

        static void dump_input_maker(
                OprDumpContext &ctx, const cg::OperatorNodeBase &opr);

        static void dump_counter_provider(
                OprDumpContext &ctx, const cg::OperatorNodeBase &opr);

        static cg::OperatorNodeBase* load_loop(
                OprLoadContext &ctx, const cg::VarNodeArray &inputs,
                const OperatorNodeConfig &config);

        static cg::OperatorNodeBase* load_input_maker(
                OprLoadContext &ctx, const cg::VarNodeArray &inputs,
                const OperatorNodeConfig &config);

        static cg::OperatorNodeBase* load_counter_provider(
                OprLoadContext &ctx, const cg::VarNodeArray &inputs,
                const OperatorNodeConfig &config);

        public:
            static void reg_all();

            // we need dedicated shallow_copy because some oprs can be copied
            // but can not be dumped; also record InterGraphVarTransformer
            static cg::OperatorNodeBase* shallow_copy(
                    const OprShallowCopyContext &orig_ctx,
                    const Loop &opr, const VarNodeArray &inputs,
                    const OperatorNodeConfig &config);

    };

} // namespace intl
} // namespace opr
} // namespace mgb

namespace mgb {
namespace serialization {
namespace fbs {

template <>
struct SupportFlatBuffersSerialization<opr::intl::LoopSerializer::LoopParam>
        : No {};

template <>
struct SupportFlatBuffersSerialization<
        opr::intl::LoopSerializer::InputMakerParam> : No {};

}  // namespace fbs
}  // namespace serialization
}  // namespace mgb

cg::OperatorNodeBase* serialization::opr_shallow_copy_loop(
        const OprShallowCopyContext &ctx,
        const cg::OperatorNodeBase &opr, const VarNodeArray &inputs,
        const OperatorNodeConfig &config) {
    return opr::intl::LoopSerializer::shallow_copy(
            ctx,
            opr.cast_final_safe<opr::Loop>(), inputs, config);
}

void LoopSerializer::reg_all() {
    MGB_SEREG_OPR_INTL_CALL_ADD(opr::Loop, dump_loop, load_loop);
    MGB_SEREG_OPR_INTL_CALL_ADD(InputMaker, dump_input_maker, load_input_maker);
    MGB_SEREG_OPR_INTL_CALL_ADD(CounterProvider,
            dump_counter_provider, load_counter_provider);
}

void LoopSerializer::dump_loop(
        OprDumpContext &ctx, const cg::OperatorNodeBase &opr) {
    bool dump_implemented = false;
    mgb_throw_if(!dump_implemented, SerializationError,
                 "Serialization of Loop opr not implemented");
}

void LoopSerializer::dump_input_maker(
        OprDumpContext &ctx, const cg::OperatorNodeBase &opr) {
    auto &&ogvar2inpidx = LoopDumpContext::from_dump_ctx(ctx).ogvar2inpidx;
    auto &&opr_im = opr.cast_final_safe<InputMaker>();
    ctx.write_param<InputMakerParam>({opr_im.param().has_assign,
            ogvar2inpidx.at(opr_im.orig_var())});
}

void LoopSerializer::dump_counter_provider(
        OprDumpContext &ctx, const cg::OperatorNodeBase &opr) {
    // there is nothing needs to do
    MGB_MARK_USED_VAR(ctx);
    MGB_MARK_USED_VAR(opr);
}

cg::OperatorNodeBase* LoopSerializer::load_loop(
        OprLoadContext &ctx, const cg::VarNodeArray &inputs,
        const OperatorNodeConfig &config) {
    bool load_implemented = false;
    cg::OperatorNodeBase* load_result = nullptr;
    mgb_throw_if(!load_implemented, SerializationError,
                 "Serialization of Loop opr not implemented");
    return load_result;
}

cg::OperatorNodeBase* LoopSerializer::load_input_maker(
        OprLoadContext &ctx, const cg::VarNodeArray &inputs,
        const OperatorNodeConfig &config) {
    MGB_MARK_USED_VAR(config);
    auto &&loop_load_ctx = LoopLoadContext::from_load_ctx(ctx);
    auto param = ctx.read_param<InputMakerParam>();
    return loop_load_ctx.desc.add_input(
            loop_load_ctx.input_vars.at(param.ogvar_id),
            param.has_assign).node()->owner_opr();
}

cg::OperatorNodeBase* LoopSerializer::load_counter_provider(
        OprLoadContext &ctx, const cg::VarNodeArray &inputs,
        const OperatorNodeConfig &config) {
    MGB_MARK_USED_VAR(inputs);
    mgb_assert(inputs.empty());
    auto &&loop_load_ctx = LoopLoadContext::from_load_ctx(ctx);
    return loop_load_ctx.desc.get_counter_var().node()->owner_opr();
}

cg::OperatorNodeBase* LoopSerializer::shallow_copy(
        const OprShallowCopyContext &orig_ctx,
        const Loop &opr, const VarNodeArray &inputs,
        const OperatorNodeConfig &config) {
    auto orig_desc = static_cast<LoopImpl::FwdDesc*>(opr.m_desc.get());
    ThinHashMap<VarNode*, size_t> ogvar2inpidx;

    mgb_assert(inputs.size()  == opr.input().size());
    for (size_t i = 0; i < inputs.size();  ++ i)
        ogvar2inpidx[opr.input(i)] = i;

    VarNodeArray cur_opr_inputs;
    auto varmap_buf = std::make_shared<ThinHashMap<VarNode*, VarNode*>>();
    auto desc_maker = [&](Loop::Desc &desc) {
        ThinHashMap<VarNode*, LoopImpl::InputMaker*> assignee2orig_im;
        auto &&varmap = *varmap_buf;

        // add inputs
        OprShallowCopyContext ctx{orig_ctx};
        for (auto inp: orig_desc->all_inputs()) {
            auto ogvar = inputs.at(ogvar2inpidx.at(inp->orig_var()));
            auto subvar = desc.add_input(ogvar, inp->param().has_assign);
            varmap[inp->output(0)] = subvar.node();
            if (inp->param().has_assign) {
                assignee2orig_im[subvar.node()] = inp;
            }
            ctx.owner_graph(subvar.node()->owner_graph());
        }

        // copy oprs
        for (auto opr: orig_desc->sub_graph_oprs()) {
            if (opr->same_type<LoopImpl::InputMaker>()) {
                continue;
            }

            if (opr->same_type<LoopImpl::DescImplBase::CounterProvider>()){
                varmap[opr->output(0)] = desc.get_counter_var().node();
            } else {
                cur_opr_inputs.clear();
                for (auto i: opr->input())
                    cur_opr_inputs.push_back(varmap.at(i));
                auto new_opr = copy_opr_shallow(*opr, cur_opr_inputs,
                        opr->config(), ctx);
                mgb_assert(new_opr->output().size() == opr->output().size());
                for (size_t i = 0; i < new_opr->output().size(); ++ i)
                    varmap[opr->output(i)] = new_opr->output(i);
            }
        }
        // add outputs in original order
        for (auto &&i: orig_desc->output_record_spec_no_dedup()) {
            desc.add_output(varmap.at(i->var_sub()), i->output_mode());
        }
        // add assignments
        for (auto &&i: assignee2orig_im) {
            desc.assign(i.first, varmap.at(i.second->assignor()));
        }
        desc.set_loop_condition(
                varmap.at(orig_desc->loop_cond_manager().var().node()));
    };

    auto &&ret = opr::Loop::make(desc_maker)[0].
        node()->owner_opr()->cast_final_safe<Loop>();
    mgb_assert(ret.output().size() == opr.output().size());

    auto trans_src_var = [varmap_buf](VarNode *src) -> VarNode* {
        auto iter = varmap_buf->find(src);
        mgb_throw_if(iter == varmap_buf->end(),
                GraphError,
                "loop fwd shallow copy: "
                "can not to get copied var from unused src var: %s",
                cg::dump_var_info({src}).c_str());
        return iter->second;
    };
    cg::InterGraphVarTransformer::register_to(
            ret.m_desc->sub_graph(), opr.m_desc->sub_graph(), trans_src_var);

    return &ret;
}

void LoopSerializerReg::entry() {
    LoopSerializer::reg_all();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

