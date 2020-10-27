/**
 * \file src/opr/impl/muxing.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/muxing.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/graph/event.h"
#include "megbrain/graph/grad_impl.h"
#include <atomic>

using namespace mgb;
using namespace opr;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(AllGather);

class AllGather::CopyStrategy {
    struct VarState {
        VarNode *var;
        std::unique_ptr<CompNode::Event> cn_ready_event;

        //! number of var states that depend on this
        size_t nr_odep = 0;
        std::atomic_int_fast32_t nr_odep_to_wake;

        explicit VarState(VarNode *v):
            var(v)
        {
            nr_odep_to_wake.store(0);
        }

        void set_ready() {
            mgb_assert(!nr_odep_to_wake.load(),
                    "val before set ready: %d", int(nr_odep_to_wake.load()));
            if (cn_ready_event) {
                cn_ready_event->record();
            }
            nr_odep_to_wake.store(nr_odep);
        }

        void wait_ready() {
            while (!nr_odep_to_wake.load());
            auto remain = -- nr_odep_to_wake;
            mgb_assert(remain >= 0);
        }
    };

    struct CopyInstr {
        bool cross_cn = false;
        VarState *src = nullptr, *dst = nullptr;
        SubTensorSpec src_sub, dst_sub;

        CopyInstr(VarState *s, const SubTensorSpec &ss,
                VarState *d, const SubTensorSpec &ds):
            cross_cn(s->var->comp_node() != d->var->comp_node()),
            src(s), dst(d), src_sub(ss), dst_sub(ds)
        {
        }

        CopyInstr() = default;

        void execute() {
            src->wait_ready();
            if (cross_cn) {
                dst->var->comp_node().device_wait_event(*src->cn_ready_event);
            }
            dst->var->dev_tensor().sub(dst_sub).copy_from_fixlayout(
                    src->var->dev_tensor().sub(src_sub));
            dst->set_ready();
        }
    };

    AllGather *m_par_opr;

    size_t m_odd_split_adjust = 0;

    //! states of each output var
    std::vector<std::vector<std::unique_ptr<VarState>>> m_var_state;

    //! copy instrs grouped by comp node
    CompNode::UnorderedMap<std::vector<CopyInstr>> m_copy_instr;

    TensorLayout m_output_layout;
    std::vector<size_t> m_axis_shape_partial_sum;

    VarState* alloc_var_state(size_t out_idx) {
        auto &&vec = m_var_state.at(out_idx);
        vec.push_back(std::make_unique<VarState>(m_par_opr->output(out_idx)));
        return vec.back().get();
    }

    void add_copy_instr(const CopyInstr &instr) {
        m_copy_instr.at(instr.dst->var->comp_node()).push_back(instr);
        auto src = instr.src;
        ++ src->nr_odep;
        if (!src->cn_ready_event && instr.cross_cn) {
            src->cn_ready_event = src->var->comp_node().create_event();
        }
        mgb_assert(instr.src_sub.layout().eq_shape(instr.dst_sub.layout()));
    }

    SubTensorSpec make_sub_spec_interval(size_t begin, size_t end) {
        begin = m_axis_shape_partial_sum.at(begin);
        end = m_axis_shape_partial_sum.at(end);
        return Slice(begin, end).apply(m_output_layout, m_par_opr->m_axis);
    }

    /*!
     * \brief make a step for parallel copy, so that
     *      output[begin:end][sub(begin, end)] == input[begin:end]
     */
    void make_prog_step(size_t begin, size_t end) {
        mgb_assert(end >= begin + 1);
        if (end == begin + 1) {
            auto src = alloc_var_state(begin),
                 dst = alloc_var_state(begin);
            src->var = m_par_opr->input(begin);
            add_copy_instr({src,
                    SubTensorSpec::make_from_layout(src->var->layout()),
                    dst, make_sub_spec_interval(begin, end)});
            return;
        }

        auto mid = begin + (end - begin) / 2;
        if ((end - begin) % 2) {
            mid += m_odd_split_adjust;
            m_odd_split_adjust ^= 1;
        }

        make_prog_step(begin, mid);
        make_prog_step(mid, end);

        std::vector<VarState*> all_src, all_dst;
        for (size_t i = begin; i < end; ++ i) {
            all_src.push_back(m_var_state.at(i).back().get());
            all_dst.push_back(alloc_var_state(i));
        }

        auto copy_sub = [&](size_t src_begin, size_t src_end,
                size_t dst_begin, size_t dst_end) {
            auto sub = make_sub_spec_interval(src_begin, src_end);
            for (size_t i = dst_begin; i < dst_end; ++ i) {
                size_t other = i - dst_begin + src_begin;
                if (other == src_end)
                    other = src_begin + (src_end - src_begin) / 2;
                mgb_assert(src_begin <= other && other < src_end);

                add_copy_instr({all_src.at(other - begin), sub,
                        all_dst.at(i - begin), sub});
            }
        };

        copy_sub(begin, mid, mid, end);
        copy_sub(mid, end, begin, mid);
    }

    public:
        void reset(AllGather *opr) {
            m_par_opr = opr;
            m_var_state.resize(opr->output().size());
            for (auto &&i: m_var_state)
                i.clear();
            m_copy_instr.clear();
            for (auto i: opr->output())
                m_copy_instr[i->comp_node()].clear();
            m_output_layout.dtype = opr->output(0)->dtype();
            m_output_layout.init_contiguous_stride(opr->output(0)->shape());
            m_axis_shape_partial_sum.clear();
            m_axis_shape_partial_sum.push_back(0);
            for (auto i: opr->input()) {
                auto real_axis = opr->m_axis;
                if (real_axis < 0)
                    real_axis += i->shape().ndim;
                m_axis_shape_partial_sum.push_back(
                        m_axis_shape_partial_sum.back() +
                        i->shape().shape[real_axis]);
            }

            make_prog_step(0, m_par_opr->output().size());
        }

        void execute_on_comp_node(const CompNode &comp_node) {
            for (auto &&state: m_var_state) {
                auto s0 = state.front().get();
                if (s0->var->comp_node() == comp_node)
                    s0->set_ready();
            }
            for (auto &&instr: m_copy_instr.at(comp_node)) {
                instr.execute();
            }
        }
};


void AllGather::get_output_var_shape(
        const TensorShapeArray &inp_shape,
        TensorShapeArray &out_shape) const {
    TensorShape oshp;
    for (auto &&ishp: inp_shape) {
        if (&ishp == &inp_shape[0]) {
            oshp = ishp;
            mgb_assert(m_axis < static_cast<int>(ishp.ndim) &&
                               m_axis >= -static_cast<int>(ishp.ndim),
                       "AllGather: axis=%d ndim=%zd", m_axis, ishp.ndim);
            continue;
        }
        auto real_axis = m_axis;
        if (real_axis < 0)
            real_axis += ishp.ndim;
        mgb_assert(oshp.ndim == ishp.ndim);
        for (int i = 0; i < static_cast<int>(oshp.ndim); ++ i) {
            if (i == real_axis) {
                oshp.shape[i] += ishp.shape[i];
            } else {
                mgb_assert(oshp.shape[i] == ishp.shape[i],
                        "shape mismatch: axis=%d oshp=%s ishp=%s",
                        real_axis, oshp.to_string().c_str(), ishp.to_string().c_str());
            }
        }
    }
    for (auto &&i: out_shape)
        i = oshp;
}

void AllGather::init_output_comp_node() {
    mgb_assert(config().comp_node().empty(),
            "output comp nodes for AllGather could not be manually specified and"
            " must be the same as that of inputs");
    for (size_t i = 0; i < input().size(); ++ i) {
        output(i)->comp_node(input(i)->comp_node());
    }
}

cg::OperatorNodeBase::NodeProp* AllGather::do_make_node_prop() const {
    auto prop = OperatorNodeBase::do_make_node_prop();
    prop->add_flag(NodeProp::Flag::CROSS_COMP_NODE_MEMORY);
    return prop;
}

void AllGather::on_mem_status_changed() {
    if (m_input_layout.size() == input().size()) {
        bool valid = true;
        for (size_t i = 0; i < m_input_layout.size(); ++ i) {
            if (!m_input_layout[i].eq_layout(input(i)->layout())) {
                valid = false;
                break;
            }
        }
        if (valid)
            return;
    }

    m_input_layout.resize(input().size());
    for (size_t i = 0; i < m_input_layout.size(); ++ i)
        m_input_layout[i] = input(i)->layout();

    m_copy_strategy->reset(this);
}

cg::OperatorNodeBase::OprEventCallback
AllGather::get_opr_event_callback() {
    return {std::bind(&AllGather::on_mem_status_changed, this)};
}

void AllGather::do_execute(ExecEnv &env) {
    CompNode::UnorderedSet used_cn;
    for (auto i: output()) {
        if (!used_cn.insert(i->comp_node()).second)
            continue;
        auto runner = [this, cn=i->comp_node()]() {
            owner_graph()->event().signal_inplace<cg::event::BeforeKernel>(
                    this, cn);
            m_copy_strategy->execute_on_comp_node(cn);
            owner_graph()->event().signal_inplace<cg::event::AfterKernel>(
                    this, cn);
        };
        env.dispatch_on_comp_node(i->comp_node(), runner);
    }
}

VarNodeArray AllGather::grad(const VarNodeArray &out_grad) {
    CompNode::UnorderedMap<VarNode*> cn_reduced;
    for (auto i: out_grad) {
        auto &&dst = cn_reduced[i->comp_node()];
        if (!dst)
            dst = i;
        else
            dst = (SymbolVar{dst} + i).node();
    }

    VarNode *og_sum = nullptr;
    for (auto i: cn_reduced) {
        if (!og_sum) {
            og_sum = i.second;
        } else {
            auto copy = Copy::make(i.second, og_sum->comp_node());
            og_sum = (SymbolVar{og_sum} + copy).node();
        }
    }

    OperatorNodeConfig::CompNodeArray sp_cn;
    SymbolVarArray partition;
    for (auto i: input()) {
        partition.push_back(GetVarShape::make(i, m_axis));
        sp_cn.push_back(i->comp_node());
    }
    return cg::to_var_node_array(Split::make(og_sum,
            Split::Options::make_partition(m_axis, partition),
            OperatorNodeConfig().comp_node_arr(sp_cn)));
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(AllGather) {
    return const_cast<AllGather&>(opr).grad(out_grad);
}
#endif

void AllGather::on_output_comp_node_stream_changed() {
}

AllGather::AllGather(
        const VarNodeArray &input, int axis,
        const OperatorNodeConfig &config):
    Super{input.at(0)->owner_graph(), config, "allgather", {input.at(0)}},
    m_copy_strategy(std::make_unique<CopyStrategy>()),
    m_axis(axis)
{
    for (auto i: input) {
        add_input({i});
        add_output(i->name());
    }
}

SymbolVarArray AllGather::make(
        const SymbolVarArray &input, int axis,
        const OperatorNodeConfig &config) {
    mgb_assert(!input.empty());
    mgb_assert(input[0].node()->owner_graph()->options().async_exec_level &&
            input[0].node()->comp_node().device_type() !=
            CompNode::DeviceType::CPU,
            "currently only AllGather between gpus supported");
    VarNodeArray inpvar;
    for (auto &&i: input)
        inpvar.push_back(i.node());
    auto opr = inpvar[0]->owner_graph()->insert_opr(std::make_unique<AllGather>(
                inpvar, axis, config));
    SymbolVarArray rst;
    for (auto i: opr->output())
        rst.push_back(i);
    return rst;
}

AllGather::~AllGather() = default;

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

