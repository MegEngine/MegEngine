/**
 * \file src/opr/impl/internal/indexing_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/internal/indexing_helper.h"
#include "megbrain/opr/internal/indexing_helper_sereg.h"
#include "megbrain/opr/param_defs.h"

using namespace mgb;
using namespace opr;
using namespace indexing;
using namespace intl;

/* ================== simple struct impls ================== */

size_t AxisNum::get(size_t ndim) const {
    int ret = m_num;
    if (ret < 0)
        ret += ndim;
    mgb_assert(ret >= 0 && static_cast<size_t>(ret) < ndim,
            "invalid axis %d for ndim %zu", m_num, ndim);
    return ret;
}

AxisIndexer AxisIndexer::make_index(AxisNum axis, SymbolVar idx) {
    AxisIndexer rst;
    rst.axis = axis;
    rst.idx = idx;
    return rst;
}

AxisIndexer AxisIndexer::make_interval(
        AxisNum axis,
        Maybe<SymbolVar> begin, Maybe<SymbolVar> end, Maybe<SymbolVar> step) {
    AxisIndexer rst;
    rst.axis = axis;
    if (begin.valid() && begin.val().node())
        rst.begin = begin.val();
    if (end.valid() && end.val().node())
        rst.end = end.val();
    if (step.valid() && step.val().node())
        rst.step = step.val();
    return rst;
}


/* ================== FancyIndexingHelper ================== */

FancyIndexingHelper::FancyIndexingHelper(
        const OperatorNodeBaseCtorParam &opr,
        VarNode *data, VarNode *value, const IndexDesc &index_desc,
        bool require_scalar_index,
        const InputTensorReplacer &input_tensor_replacer):
    Super(opr),
    m_idx_inp_start{1u + (value != nullptr)},
    m_require_scalar_index{require_scalar_index},
    m_is_assign_opr{value != nullptr},
    m_input_tensor_replacer{input_tensor_replacer}
{
    add_input({data});
    if (value) {
        add_input({value});
        mgb_assert(data->dtype() == value->dtype(),
                "subtensor modifier dest and value must have same dtype; got "
                "dest=%s value=%s",
                data->dtype().name(), value->dtype().name());
    }
    add_output(None)->dtype(data->dtype());
    if (!require_scalar_index) {
        cg::add_workspace_output(this);
    }
    init(index_desc);

    if (has_input_tensor_replacer()) {
        mgb_assert(value);
        output(0)->
            add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE).
            add_flag(VarNode::Flag::VOLATILE_CONTENT);

        // do not dedup
        add_equivalence_component<ScalarHash<void*>>(this);
    }
}

void FancyIndexingHelper::init(const IndexDesc &index_desc) {
    mgb_assert(input().size() == m_idx_inp_start);
    mgb_assert(m_index_desc.empty());

    m_input2idxonly_axis_indexer.resize(input().size(), nullptr);
    m_input2idxonly_axis_indexer.reserve(input().size());
    m_index_desc = index_desc;

    // sort in reverse order, so slice would work from low dim to high dim, to
    // make it contiguous on shape-1 axes
    small_sort(m_index_desc.begin(), m_index_desc.end(),
            AxisIndexer::cmp_by_axis_rev);

    size_t dedup_hash;
    auto add_inp = [&](SymbolVar i,
            AxisIndexer *idxonly_axis_indexer = nullptr) {
        dedup_hash <<= 1;
        if (i.node()) {
            dedup_hash |= 1;
            add_input({i.node()});
            m_input2idxonly_axis_indexer.push_back(idxonly_axis_indexer);
        }
    };

    AxisNum prev_idx(std::numeric_limits<int>::max());
    for (auto &&i: m_index_desc) {
        mgb_throw_if(i.axis == prev_idx, GraphError,
                "duplicated axes in IndexDesc");
        prev_idx = i.axis;
        bool has_idx = i.idx.node(),
             has_slice = i.begin.node() || i.end.node() || i.step.node();
        mgb_throw_if(!(has_idx ^ has_slice), GraphError,
                "AxisIndexer should contain either slice or index info");
        dedup_hash = i.axis.get_raw();

        if (has_idx) {
            ++ m_nr_axis_single_idx;
            if (!m_require_scalar_index) {
                mgb_throw_if(i.idx.node()->dtype() != dtype::Int32(),
                        GraphError,
                        "indexers must be int32; got %s for axis %d",
                        i.idx.node()->dtype().name(), i.axis.get_raw());
            }
        }

        // call all add_inp on all possible inputs to get correct dedup_hash
        add_inp(i.begin);
        add_inp(i.end);
        add_inp(i.step);
        add_inp(i.idx, &i);
        if (!has_input_tensor_replacer()) {
            add_equivalence_component<ScalarHash<size_t>>(dedup_hash);
        }
    }

    mgb_assert(input().size() == m_input2idxonly_axis_indexer.size());
}

SubTensorSpec FancyIndexingHelper::do_make_sub_spec(
        const TensorLayout &inp_layout) const {

    auto spec = SubTensorSpec::make_from_layout(inp_layout);

    auto iv_iter = m_value_infer_result.begin();
    auto next_iv = [&]() {
        mgb_assert(iv_iter != m_value_infer_result.end());
        const DeviceTensorND* tp = *iv_iter;
        ++ iv_iter;
        mgb_assert(tp->shape().is_scalar(), 
                "Indices must be scalar; got shape: %s.\nPlease Try .ai[] If You Need Numpy-like Advanced Index!!!",
                tp->shape().to_string().c_str());
        ptrdiff_t val;
        static_cast_dtype_safe(&val, tp->dtype(), tp->raw_ptr());
        return val;
    };

    // m_index_desc sorted from high axis(large value) to low axis, so is
    // axis_to_remove
    // valid when m_require_scalar_index is true
    std::vector<size_t> axis_to_remove;

    size_t prev_axis = megdnn::param::OptionalAxisV1::INVALID_AXIS;
    for (auto &&i: m_index_desc) {
        auto axis = i.axis.get(inp_layout.ndim);
        mgb_throw_if(axis == prev_axis, GraphError,
                "duplicated axis in subtensor: desc=%d axis=%zu",
                i.axis.get_raw(), axis);
        prev_axis = axis;
        Maybe<ptrdiff_t> begin, end, step;
        if (i.idx.node()) {
            if (!m_require_scalar_index) {
                continue;
            }
            axis_to_remove.push_back(axis);
            begin = next_iv();
            if (begin.val() != -1)
                end = begin.val() + 1;
        } else {
            if (i.begin.node())
                begin = next_iv();
            if (i.end.node())
                end = next_iv();
            if (i.step.node())
                step = next_iv();
        }

        spec.merge_with(Slice(begin, end, step).apply(spec.layout(), axis));
    }
    mgb_assert(iv_iter == m_value_infer_result.end());

    if (!axis_to_remove.empty()) {
        auto dl = spec.layout();
        for (auto am: axis_to_remove) {
            if (dl.ndim == 1) {
                mgb_assert(am == 0 && axis_to_remove.back() == 0);
                break;
            }
            dl.remove_axis_inplace(am);
        }
        spec = SubTensorSpec::make_from_offset_elem(dl, spec.offset_elem());
    }
    return spec;
}

cg::OperatorNodeBase::NodeProp* FancyIndexingHelper::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    SmallVector<NodeProp::DepType> dt(input().size(),
            NodeProp::DepType::DEV_VALUE);

    // use dynout for readonly-fwd for Subtensor
    auto host_val_dt = NodeProp::DepType::HOST_VALUE;
    if (!m_is_assign_opr && m_require_scalar_index) {
        // This case corresponds to Subtensor. Since memory allocation only
        // happens when shape changes and it is possible that indexing vars
        // change without shape change (e.g. x[i:i+2]), here we simply require
        // dynamic memory allocation whenever index is not constant so output
        // value is always synchronized with current index.
        host_val_dt |= NodeProp::DepType::HOST_VALUE_DYNOUT;
    }

    for (size_t i = m_idx_inp_start; i < dt.size(); ++ i) {
        if (m_require_scalar_index || !m_input2idxonly_axis_indexer[i]) {
            // note: host value is needed for
            //      1) all vars when they are required to be scalar
            //      2) begin/end/step in interval indexing case
            dt[i] = host_val_dt;
        }
    }
    if (has_input_tensor_replacer()) {
        dt[0] = NodeProp::DepType::SHAPE;
    }
    prop->reset_dep_type(input(), dt);
    return prop;
}

SubTensorSpec FancyIndexingHelper::fancy_indexing_make_sub_spec(
        const TensorLayout &inp_layout) {
    auto &&inp = input();
    auto &&mgr = owner_graph()->static_infer_manager();
    if (m_require_scalar_index) {
        m_value_infer_result.resize(inp.size() - m_idx_inp_start);
        for (size_t i = 0; i < m_value_infer_result.size(); ++ i) {
            m_value_infer_result[i] =
                &mgr.infer_value(inp[i + m_idx_inp_start]);
        }
    } else {
        m_value_infer_result.clear();
        m_value_infer_result.reserve(inp.size() - m_idx_inp_start -
                m_nr_axis_single_idx);
        for (size_t i = m_idx_inp_start; i < inp.size(); ++ i) {
            if (!m_input2idxonly_axis_indexer[i]) {
                m_value_infer_result.emplace_back(&mgr.infer_value(inp[i]));
            }
        }
    }

    return do_make_sub_spec(inp_layout);
}

SubTensorSpec FancyIndexingHelper::fancy_indexing_make_sub_spec(
        const TensorLayout &inp_layout,
        const cg::static_infer::InpVal &infer_inp,
        size_t infer_inp_start, bool fake_single_idx) {

    // static infer should not be used for multi-axis-vector-indexing
    mgb_assert(m_require_scalar_index || !fake_single_idx);

    static DeviceTensorND fake_val;
    static std::mutex fake_val_mtx;

    if (mgb_unlikely(fake_val.empty())) {
        MGB_LOCK_GUARD(fake_val_mtx);
        if (fake_val.empty()) {
            fake_val.comp_node(CompNode::default_cpu()).
                dtype(dtype::Int32()).
                resize({1}).
                ptr<dt_int32>()[0] = 0;
        }
    }

    auto tsize = infer_inp.val.size() - infer_inp_start;
    if (m_require_scalar_index) {
        if (fake_single_idx)
            tsize += m_nr_axis_single_idx;
        mgb_assert(tsize == input().size() - m_idx_inp_start);
    } else {
        mgb_assert(!fake_single_idx);
        mgb_assert(tsize + m_nr_axis_single_idx ==
                input().size() - m_idx_inp_start);
    }

    auto infer_inp_iter = infer_inp.val.begin() + infer_inp_start;
    m_value_infer_result.resize(tsize);
    for (size_t i = 0; i < tsize; ++ i) {
        const DeviceTensorND *ptr;
        if (fake_single_idx &&
                m_input2idxonly_axis_indexer[i + m_idx_inp_start]) {
            ptr = &fake_val;
        } else {
            ptr = &(infer_inp_iter ++)->value();
        }
        m_value_infer_result[i] = ptr;
    }

    mgb_assert(infer_inp_iter == infer_inp.val.end());
    return do_make_sub_spec(inp_layout);
}

std::pair<DeviceTensorND, DeviceTensorND>
FancyIndexingHelper::fancy_indexing_get_tensors_for_modify_in_scn_do_execute() {
    auto &&val = input(1)->dev_tensor();
    DeviceTensorND dest;

    if (has_input_tensor_replacer()) {
        auto &&ishp = input(0)->shape();
        dest = m_input_tensor_replacer(ishp);
        mgb_assert(dest.shape().eq_shape(ishp));
    } else {
        auto &&inp = input(0)->dev_tensor();
        dest = output(0)->dev_tensor();
        if (dest.raw_ptr() != inp.raw_ptr())
            dest.copy_from_fixlayout(inp);
        else
            mgb_assert(dest.layout().eq_layout(inp.layout()));
    }

    auto dsub = dest.sub(fancy_indexing_make_sub_spec(dest.layout()));
    auto dst_span = dsub.layout().span();
    auto val_span = val.layout().span();
    auto dst_pmin = dsub.raw_ptr() + dst_span.low_byte,
         dst_pmax = dsub.raw_ptr() + dst_span.high_byte,
         val_pmin = val.raw_ptr() + val_span.low_byte,
         val_pmax = val.raw_ptr() + val_span.high_byte;
    if (dst_pmax > val_pmin && val_pmax > dst_pmin) {
        // val overlaps with dsub
        DeviceTensorND tmp;
        tmp.copy_from(val);
        return {dsub, tmp};
    } else {
        return {dsub, val};
    }
}

void FancyIndexingHelper::mem_plan_fwd_in2out_writable() {
    if (m_idx_inp_start == 2) {
        if (!has_input_tensor_replacer()) {
            cg::request_fwd_in2out_writable_if_no_mem_ovelap(this, 0, 0);
        }
    } else {
        mgb_assert(m_idx_inp_start == 1);
    }
}

/* ================== serialization ================== */

serialization::IndexDescMaskDump
serialization::IndexDescMaskDump::from_index_desc(const IndexDesc &desc) {
    mgb_assert(desc.size() <= TensorShape::MAX_NDIM);
    IndexDescMaskDump ret;
    ret.nr_item = desc.size();
    for (size_t i = 0; i < desc.size(); ++ i) {
        auto &&s = desc[i];
        ret.items[i] = {static_cast<int8_t>(s.axis.get_raw()),
                        static_cast<bool>(s.begin.node()),
                        static_cast<bool>(s.end.node()),
                        static_cast<bool>(s.step.node()),
                        static_cast<bool>(s.idx.node())};
    }
    return ret;
}

IndexDesc serialization::IndexDescMaskDump::to_index_desc(
                cg::VarNodeArray::const_iterator inp_begin,
                cg::VarNodeArray::const_iterator inp_end) const {
    IndexDesc ret(nr_item);
    auto assign = [&](SymbolVar &dest, bool mask) {
        if (mask)
            dest = *(inp_begin ++);
    };
    for (size_t i = 0; i < nr_item; ++ i) {
        auto &&t = ret[i];
        auto &&s = items[i];
        t.axis = s.axis;
        assign(t.begin, s.begin);
        assign(t.end, s.end);
        assign(t.step, s.step);
        assign(t.idx, s.idx);
    }
    mgb_assert(inp_begin == inp_end);
    return ret;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

