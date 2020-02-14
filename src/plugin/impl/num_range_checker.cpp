/**
 * \file src/plugin/impl/num_range_checker.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/io.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/plugin/num_range_checker.h"
#include "megbrain/graph/exc_extra_info.h"

#include "megdnn/tensor_iter.h"

#include <cmath>

using namespace mgb;

void NumRangeChecker::Checker::init(VarNode *var, float range) {
    if (m_func)
        return;

    m_inp = std::make_shared<DeviceTensorND>(var->comp_node(), var->dtype());
    m_out = std::make_unique<HostTensorND>();
    auto cg = ComputingGraph::make();
    cg->options().log_level = 0;
    auto vi = opr::VolatileSharedDeviceTensor::make(*cg, m_inp),
         chk = opr::abs(vi) < range,
         good = opr::reduce_min(chk, chk.make_scalar(1));
    auto cb = [d=m_out.get()](DeviceTensorND &dv) {
        d->copy_from(dv).sync();
    };
    m_func = cg->compile({{good, cb}});
}

bool NumRangeChecker::Checker::check(VarNode *var){
    auto &&val = var->dev_tensor();
    if (val.layout().is_contiguous()) {
        *m_inp = var->dev_tensor();
    } else {
        *m_inp = {};
        m_inp->copy_from(val);
    }
    m_func->execute();
    mgb_assert(m_out->shape().is_scalar());
    return m_out->ptr<float>()[0] >= 0.5;
}

NumRangeChecker::NumRangeChecker(cg::ComputingGraph *graph, float range):
    PluginBase(graph), m_range{range}
{
    add_member_func_as_event_handler(&NumRangeChecker::on_kern_end);
    add_member_func_as_event_handler(&NumRangeChecker::on_subgraph_associated);
}

void NumRangeChecker::on_kern_end(const cg::event::OprExecKernelEnd &event) {
    for (VarNode *var: event.opr->output()) {
        if (!var->contain_flag(VarNode::Flag::VOLATILE_CONTENT) &&
                var->dtype().category() == DTypeCategory::FLOAT) {
            event.env->dispatch_on_comp_node(var->comp_node(),
                    [this, var](){on_var_computed(var);});
        }
    }
}

void NumRangeChecker::on_subgraph_associated(
        const cg::event::SubgraphAssociated &event) {
    mgb_assert(event.par_graph == m_owner_graph);
    m_sub_graph_checkers.emplace_back(std::make_unique<NumRangeChecker>(
                event.sub_graph, m_range));
}

void NumRangeChecker::on_var_computed(VarNode *var) {
    if (!var->dev_tensor_valid())
        return;

    auto &&checker = m_cn2dt2checker[var->comp_node()][var->dtype().enumv()];
    checker.init(var, m_range);
    if (!checker.check(var)) {
        HostTensorND hv;
        hv.copy_from(var->dev_tensor()).sync();
        std::string msg{mgb_ssprintf_log("float value out of range: var: %s\n",
                cg::dump_var_info({var}).c_str())};
        switch (hv.dtype().enumv()) {
#define cb(_dt) case DTypeTrait<_dt>::enumv: \
                msg += format_msg<DTypeTrait<_dt>::ctype>(hv, m_range); \
                break;
            MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
            default:
                mgb_assert(0, "unexpected dtype");
        }

        mgb_throw_raw(cg::OperatorNodeExcExtraInfo::
                ExcMaker{var->owner_opr()}.make<Error>(msg));
    }
}

template<typename ctype>
std::string NumRangeChecker::format_msg(const HostTensorND &hv, float range) {
    auto iter = megdnn::tensor_iter<ctype>(hv.as_megdnn()).begin();
    for (size_t i = 0, it = hv.shape().total_nr_elems(); i < it; ++ i) {
        float val = static_cast<float>(*iter);
        if (!(std::fabs(val) < range)) {
            TensorShape idx_shp;
            idx_shp.ndim = hv.shape().ndim;
            std::copy(iter.idx(), iter.idx() + idx_shp.ndim, idx_shp.shape);
            return mgb_ssprintf_log(
                    " value=%g range=%g index=%s/%s",
                    val, range,
                    idx_shp.to_string().c_str(),
                    hv.shape().to_string().c_str());
        }
        ++ iter;
    }
    return mgb_cstr_log(" <error: range check passed on host>");
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
