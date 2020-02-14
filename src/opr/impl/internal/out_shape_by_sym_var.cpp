/**
 * \file src/opr/impl/internal/out_shape_by_sym_var.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/io.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/utils/arith_helper.h"

using namespace mgb;
using namespace opr;
using namespace mixin;

/* ===================== OutshapeBySymvarOpr ===================== */

OutshapeBySymvarOpr::~OutshapeBySymvarOpr() = default;

void OutshapeBySymvarOpr::mixin_outshape_by_symvar_enable(
        OperatorNodeBase &opr,
        size_t nr_shape_inp, size_t hostval_inp_start) {
    mgb_assert(!m_enable_out_shape_by_symbol_var);
    mgb_assert(hostval_inp_start >= nr_shape_inp &&
            hostval_inp_start < opr.input().size());
    m_nr_shape_inp = nr_shape_inp;
    m_hostval_inp_start = hostval_inp_start;
    m_enable_out_shape_by_symbol_var = true;

    m_shape_infer_info.shape_inp_shp.resize(nr_shape_inp);
    m_shape_infer_info.shpval_inp_val.resize(
            opr.input().size() - hostval_inp_start);
}


void OutshapeBySymvarOpr::mixin_init_output_static_infer_desc(
        OperatorNodeBase &opr) {
    using namespace cg::static_infer;
    DepVal deps;
    for (size_t i = 0; i < m_nr_shape_inp; ++ i)
        deps.push_back({opr.input(i), DepType::SHAPE});
    for (size_t i = m_hostval_inp_start; i < opr.input().size(); ++ i)
        deps.push_back({opr.input(i), DepType::VALUE});

    auto infer_shape = [&opr, this](TensorShape &dest, const InpVal &) {
        outshape_by_symvar_do_get_output_shape(
                dest, mixin_outshape_by_symvar_get_shape_infer_info(opr));
        return true;
    };
    opr.owner_graph()->static_infer_manager().register_shape_infer(
            opr.output(0), {SourceType::DEP, deps, infer_shape});
}

const OutshapeBySymvarOpr::ShapeInferInfo&
OutshapeBySymvarOpr::mixin_outshape_by_symvar_get_shape_infer_info(
        const OperatorNodeBase &opr) const {
    auto &&mgr = opr.owner_graph()->static_infer_manager();

    for (size_t i = 0; i < m_nr_shape_inp; ++ i) {
        m_shape_infer_info.shape_inp_shp[i] = mgr.infer_shape(opr.input(i));
    }

    for (size_t i = m_hostval_inp_start; i < opr.input().size(); ++ i) {
        m_shape_infer_info.shpval_inp_val[i - m_hostval_inp_start] =
            &mgr.infer_value(opr.input(i));
    }
    return m_shape_infer_info;
}

void OutshapeBySymvarOpr::mixin_outshape_by_symvar_reset_node_dep_type(
        const OperatorNodeBase &opr,
        NodeProp *prop) const {
    SmallVector<NodeProp::DepType> dt(opr.input().size(),
            NodeProp::DepType::DEV_VALUE);
    for (size_t i = m_hostval_inp_start; i < opr.input().size(); ++ i) {
        dt[i] = NodeProp::DepType::HOST_VALUE;
    }
    prop->reset_dep_type(opr.input(), dt);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
