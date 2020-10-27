/**
 * \file src/opr/impl/indexing.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/indexing.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/utility.h"
#include "megbrain/graph/grad_impl.h"

#include "./internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;

namespace {
    void check_index_dtype(std::initializer_list<SymbolVar*> &inputs) {
        mgb_assert(inputs.size() >= 2);
        auto iter = inputs.begin();
        ++ iter;
        SymbolVar &index = **iter;
        if (index.dtype() != dtype::Int32()) {
            mgb_log_warn("dtype of index in IndexingOneHot must be Int32, "
                    "got %s for variable %s; convert to Int32 implicitly",
                    index.dtype().name(), index.node()->cname());
            index = opr::TypeCvt::make(index, dtype::Int32());
        }
    }

    enum IndexingModifyType {
        SET, INCR
    };

    template<typename Opr>
    struct IndexingModifyTypeGetter {};

#define REG(op, type) \
    template<> \
    struct IndexingModifyTypeGetter<megdnn::op> { \
        static constexpr IndexingModifyType value = IndexingModifyType::type; \
    };
REG(IndexingIncrMultiAxisVec, INCR)
REG(IncrMeshIndexing, INCR)
REG(BatchedIncrMeshIndexing, INCR)
REG(IndexingSetMultiAxisVec, SET)
REG(SetMeshIndexing, SET)
REG(BatchedSetMeshIndexing, SET)
#undef REG

}

namespace mgb {
namespace opr {
namespace intl {

    template<>
    struct MegDNNOprInitInputsModifier<IndexingOneHot> {
        static void apply(const IndexingOneHot::Param &param,
                std::initializer_list<SymbolVar*> inputs) {
            MGB_MARK_USED_VAR(param);
            check_index_dtype(inputs);
        }
    };

    template<>
    struct MegDNNOprInitInputsModifier<IndexingSetOneHot>:
    public MegDNNOprInitInputsModifier<IndexingOneHot> {};
}
}
}

/* ==================== IndexingOneHot ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(IndexingOneHot);
MEGDNN_OPR_INIT2(IndexingOneHot, "indexing_one_hot")

void IndexingOneHot::init_output_dtype() {
    output(0)->dtype(input(0)->dtype());
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(IndexingOneHot) {
    if (wrt_idx == 0) {
        return IndexingSetOneHot::make(
                SymbolVar{opr.input(0)}.fill_retain_dtype(0),
                opr.input(1), out_grad[0], opr.param()).node();
    }
    return InvalidGrad::make(opr, wrt_idx);
}
#endif

/* ==================== IndexingSetOneHot ==================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(IndexingSetOneHot);
MEGDNN_OPR_INIT3(IndexingSetOneHot, "indexing_set_one_hot")

void IndexingSetOneHot::init_output_dtype() {
    output(0)->dtype(input(0)->dtype());
}

void IndexingSetOneHot::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

void IndexingSetOneHot::mem_plan_fwd_in2out_writable() {
    cg::request_fwd_in2out_writable_if_no_mem_ovelap(this, 0, 0);
}

void IndexingSetOneHot::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto &&mgr = owner_graph()->static_infer_manager();
    mgr.register_shape_infer(output(0),
            ShapeInferDesc::make_identity(input(0)));
    init_output_static_infer_desc_workspace(false);
}

void IndexingSetOneHot::scn_do_execute() {
    auto &&idata = input(0)->dev_tensor(), &&index = input(1)->dev_tensor(),
         &&odata = output(0)->dev_tensor();

    if (idata.raw_ptr() != odata.raw_ptr()) {
        odata.copy_from_fixlayout(idata);
    } else {
        mgb_assert(odata.layout().eq_layout(idata.layout()));
    }
    mgb_assert(odata.layout().is_contiguous());

    megdnn_opr()->exec(odata.as_megdnn(), index.as_megdnn(),
            input(2)->dev_tensor().as_megdnn(),
            intl::get_megdnn_workspace_from_var(output(1)));
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(IndexingSetOneHot) {
    SymbolVar index{opr.input(1)}, sub{opr.input(2)}, og{out_grad.at(0)};
    if (wrt_idx == 0) {
        return IndexingSetOneHot::make(og, index, sub.fill_retain_dtype(0),
                opr.param()).node();
    }
    if (wrt_idx == 2) {
        return IndexingOneHot::make(og, index, opr.param()).node();
    }
    return InvalidGrad::make(opr, wrt_idx);
}
#endif

size_t IndexingSetOneHot::get_workspace_size_bytes(
        const TensorShapeArray &input_shapes,
        const TensorShapeArray &output_shapes) const {
    return megdnn_opr()->get_workspace_in_bytes(
            {input_shapes[0], input(0)->dtype()},
            {input_shapes[1], input(1)->dtype()},
            {input_shapes[2], input(2)->dtype()}
            );
}

/* ==================== IndexingRemap ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(IndexingRemap);
MEGDNN_OPR_INIT2(IndexingRemap, "indexing_remap")

void IndexingRemap::init_output_dtype() {
    mgb_throw_if(input(1)->dtype() != dtype::Int32(), GraphError,
            "IndexingRemap requires map input to be int32");
    output(0)->dtype(input(0)->dtype());
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(IndexingRemap) {
    if (wrt_idx == 1)
        return InvalidGrad::make(opr, wrt_idx);
    mgb_assert(wrt_idx == 0 && out_grad[0]);
    return IndexingRemapBackward::make(
            out_grad[0], opr.input(1), opr.input(0), opr.param()).node();
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(IndexingRemapBackward);
MEGDNN_OPR_INIT3(IndexingRemapBackward, "indexing_remap_bwd", 2, false);

/* ================= IndexingMultiAxisVecMegDNNOprHolder ================= */
template<class Opr>
Opr& mixin::IndexingMultiAxisVecMegDNNOprHolder<Opr>::megdnn_opr(
        cg::SingleCNOperatorNodeBase& self) {
    auto comp_node = self.comp_node();
    if (!m_megdnn_opr || m_megdnn_opr.comp_node() != comp_node) {
        m_megdnn_opr = intl::create_megdnn_opr<Opr>(comp_node);
        m_megdnn_opr->set_error_tracker(
                static_cast<cg::OperatorNodeBase*>(&self));
    }
    return *m_megdnn_opr;
}

template<class Opr>
void mixin::IndexingMultiAxisVecMegDNNOprHolder<Opr>::register_workspace_infer(
        const indexing::IndexDesc &index_desc,
        cg::SingleCNOperatorNodeBase &opr, VarNode *data, VarNode *value) {
    using namespace cg::static_infer;
    auto infer_shape = [this, &index_desc, &opr](
            TensorShape &dest, const InpVal &inp) {
        size_t axes[TensorShape::MAX_NDIM], nr_axes = 0;
        auto ndim = inp.val[0].shape().ndim;
        for (auto &&i: reverse_adaptor(index_desc)) {
            if (i.idx.node()) {
                axes[nr_axes ++] = i.axis.get(ndim);
            }
        }
        if (!nr_axes) {
            dest = {0};
        } else {
            dest = {megdnn_opr(opr).get_workspace_in_bytes(
                    inp.val[1].shape(), axes, nr_axes)};
        }
        return true;
    };
    opr.owner_graph()->static_infer_manager().register_shape_infer(
            opr.output(1),
            {SourceType::DEP,
            {{data, DepType::SHAPE}, {value, DepType::SHAPE}},
            infer_shape});
}

template <class Opr>
void mixin::IndexingMultiAxisVecMegDNNOprHolder<Opr>::record_megdnn_opr(
        mgb::cg::GraphExecutable::ExecDependencyArray& deps) {
    deps.emplace_back(
            std::make_unique<intl::MegDNNGraphDep>(std::move(m_megdnn_opr)));
}

/* ==================== MultiAxisVecFancyIndexingHelper ==================== */
std::pair<const megdnn::IndexingMultiAxisVec::IndexDesc&, bool>
intl::MultiAxisVecFancyIndexingHelper::make_megdnn_index_desc(
        size_t inp_ndim, bool warn_all_scalar) {

    auto &&index = m_megdnn_index_cache;
    index.clear();
    bool is_empty_shape = false;
    for (auto i: reverse_adaptor(m_input2idxonly_axis_indexer)) {
        if (i) {
            index.push_back({
                    i->axis.get(inp_ndim),
                    i->idx.node()->dev_tensor().as_megdnn()});
            is_empty_shape |= index.back().vec.layout.is_empty();
        }
    }

    if (!m_scalar_idx_warn_printed && warn_all_scalar) {
        bool all_scalar = true;
        for (auto &&i: index) {
            if (!i.vec.layout.is_scalar()) {
                all_scalar = false;
                break;
            }
        }
        if (all_scalar) {
            mgb_log_warn("%s{%s}: no vector indexer; consider using Subtensor "
                    "family for better performance; you can set "
                    "MGB_THROW_ON_SCALAR_IDX to throw an exception to help "
                    "tracking the related operator",
                    cname(), dyn_typeinfo()->name);
            mgb_throw_if(MGB_GETENV("MGB_THROW_ON_SCALAR_IDX"),
                    MegBrainError, "vector-indexing operator used with all "
                    "scalar indices");
        }

        // always set m_scalar_idx_warn_printed to be true, so we do not print
        // this warning in the future
        m_scalar_idx_warn_printed = true;
    }

    return {index, is_empty_shape};
}

/* ==================== IndexingMultiAxisVecBase ==================== */
template<class Opr>
cg::OperatorNodeBase::NodeProp*
IndexingMultiAxisVecBase<Opr>::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    // TODO: should also allow input shape is empty if any
    // indexer's shape is empty
    for (auto i: m_input2idxonly_axis_indexer) {
        if (i) {
            prop->add_dep_type_existing_var(
                    i->idx.node(), NodeProp::DepType::VALUE_ALLOW_EMPTY);
        }
    }
    return prop;
}

template <class Opr>
void IndexingMultiAxisVecBase<Opr>::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    DepVal deps;

    // shape inference only needs slices
    deps.push_back({input(0), DepType::SHAPE});
    // loop in reverse order because megdnn opr needs ascending axes
    for (size_t i = m_input2idxonly_axis_indexer.size() - 1; i; -- i) {
        if (m_input2idxonly_axis_indexer[i]) {
            deps.push_back({input(i), DepType::SHAPE});
        }
    }
    size_t inp_interval_start = deps.size();
    for (size_t i = 1; i < m_input2idxonly_axis_indexer.size(); ++ i) {
        if (!m_input2idxonly_axis_indexer[i]) {
            deps.push_back({input(i), DepType::VALUE});
        }
    }
    auto infer_shape = [this, inp_interval_start](
            TensorShape &dest, const InpVal &inp) {
        auto &&ishp = inp.val[0].shape();
        auto subspec = fancy_indexing_make_sub_spec(
                {ishp, input(0)->dtype()}, inp, inp_interval_start);
        dest = subspec.layout();
        typename Opr::IndexDescLayoutOnly index_layout;
        size_t indexer_pos = 1;
        for (auto i: reverse_adaptor(m_input2idxonly_axis_indexer)) {
            if (i) {
                index_layout.push_back({i->axis.get(dest.ndim),
                        {inp.val.at(indexer_pos ++).shape(), dtype::Int32()}});
            }
        }
        mgb_assert(indexer_pos == inp_interval_start);
        if (!index_layout.empty()) {
            // index_layout is empty if all indices are intervals
            TensorLayout tmp;
            Opr::deduce_layout(
                    {dest, input(0)->dtype()}, index_layout, tmp);
            dest = tmp;
        }
        return true;
    };
    owner_graph()->static_infer_manager().register_shape_infer(
            output(0), {SourceType::DEP, deps, infer_shape});

    this->register_workspace_infer(index_desc(), *this, input(0), output(0));
}

template <class Opr>
void IndexingMultiAxisVecBase<Opr>::record_execute_deps(
        mgb::cg::GraphExecutable::ExecDependencyArray& deps) {
    this->record_megdnn_opr(deps);
}

namespace {
template <class Opr>
struct ShouldWarnOnScalarIndexer {
    static constexpr bool val = false;
};

#define WARN(opr)                                   \
    template <>                                     \
    struct ShouldWarnOnScalarIndexer<megdnn::opr> { \
        static constexpr bool val = true;           \
    }
WARN(IndexingMultiAxisVec);
WARN(IndexingSetMultiAxisVec);
WARN(IndexingIncrMultiAxisVec);
#undef WARN
}  // anonymous namespace

template <class Opr>
void IndexingMultiAxisVecBase<Opr>::scn_do_execute() {
    auto inp = input(0)->dev_tensor();
    inp = inp.sub(fancy_indexing_make_sub_spec(inp.layout()));
    auto &&index_desc = make_megdnn_index_desc(
            inp.layout().ndim, ShouldWarnOnScalarIndexer<Opr>::val);
    auto &&odev = output(0)->dev_tensor();
    if (index_desc.first.empty()) {
        odev.copy_from_fixlayout(inp);
    } else {
        if (!index_desc.second) {
            // only call megdnn exec if result is not empty
            this->megdnn_opr(*this).exec(
                    inp.as_megdnn(), index_desc.first, odev.as_megdnn(),
                    intl::get_megdnn_workspace_from_var(output(1)));
        } else {
            mgb_assert(odev.empty());
        }
    }
}

/* ==================== IndexingModifyMultiAxisVecHelper ==================== */

template<class Opr>
void intl::IndexingModifyMultiAxisVecHelper<Opr>::
init_output_static_infer_desc() {
    using namespace cg::static_infer;
    this->owner_graph()->static_infer_manager().register_shape_infer(
            this->output(0), ShapeInferDesc::make_identity(this->input(0)));

    this->register_workspace_infer(index_desc(), *this, input(0), input(1));
}

template<class Opr>
void intl::IndexingModifyMultiAxisVecHelper<Opr>::scn_do_execute() {
    auto inp = this->fancy_indexing_get_tensors_for_modify_in_scn_do_execute();
    auto index_desc = this->make_megdnn_index_desc(
            inp.first.layout().ndim, ShouldWarnOnScalarIndexer<Opr>::val);
    if (index_desc.second){
        mgb_assert(inp.second.shape().is_empty());
        return;
    }
    if (index_desc.first.empty()) {
        using IMT = IndexingModifyType;
        static constexpr auto modify_type =
                IndexingModifyTypeGetter<Opr>::value;
        switch (modify_type) {
            case IMT::SET: {
                inp.first.copy_from_fixlayout(inp.second);
                break;
            } case IMT::INCR: {
                megdnn::AddUpdate* add_update = intl::get_megdnn_global_opr<
                    megdnn::AddUpdate>(comp_node());
                add_update->exec(inp.first.as_megdnn(), inp.second.as_megdnn());
                break;
            } default:
                mgb_throw(MegBrainError, "bad modify type");
        }
    } else {
        this->megdnn_opr(*this).exec(
                inp.first.as_megdnn(), inp.second.as_megdnn(),
                index_desc.first,
                intl::get_megdnn_workspace_from_var(output(1)));
    }
}

template<class Opr>
cg::OperatorNodeBase::NodeProp*
intl::IndexingModifyMultiAxisVecHelper<Opr>::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    using DT = NodeProp::DepType;
    // TODO: should also allow input shape is empty if any
    // indexer's shape is empty
    prop->add_dep_type_existing_var(input(1), DT::VALUE_ALLOW_EMPTY);
    for (auto i: m_input2idxonly_axis_indexer) {
        if (i) {
            prop->add_dep_type_existing_var(
                    i->idx.node(), DT::VALUE_ALLOW_EMPTY);
        }
    }
    return prop;
}

template<class Opr>
void intl::IndexingModifyMultiAxisVecHelper<Opr>::
add_input_layout_constraint() {
    auto check_cont1 = [](const TensorLayout &ly) {
        return ly.collapse_contiguous().ndim == 1;
    };
    this->input(1)->add_layout_constraint(check_cont1);
}

/* ==================== MultiAxisVec misc ==================== */

MGB_IMPL_FANCY_INDEXING_OPR_GET(
        IndexingMultiAxisVec, "indexing_multi_axis_vec", false,
        output(0)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
        );
MGB_IMPL_FANCY_INDEXING_OPR_MODIFY(
        IndexingSetMultiAxisVec, "indexing_set_multi_axis_vec", false);
MGB_IMPL_FANCY_INDEXING_OPR_MODIFY(
        IndexingIncrMultiAxisVec, "indexing_incr_multi_axis_vec", false);

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(IndexingMultiAxisVec) {
    if (wrt_idx)
        return InvalidGrad::make(opr, wrt_idx);

    return IndexingIncrMultiAxisVec::make(
            SymbolVar{opr.input(0)}.fill_retain_dtype(0),
            out_grad.at(0), opr.index_desc()).node();
}
#endif

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(IndexingSetMultiAxisVec) {
    if (wrt_idx >= 2)
        return InvalidGrad::make(opr, wrt_idx);
    if (wrt_idx == 0) {
        return IndexingSetMultiAxisVec::make(out_grad.at(0),
                SymbolVar{opr.input(1)}.fill_retain_dtype(0),
                opr.index_desc()).node();
    }
    return IndexingMultiAxisVec::make(out_grad.at(0), opr.index_desc()).node();
}
#endif

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(IndexingIncrMultiAxisVec) {
    if (wrt_idx >= 2)
        return InvalidGrad::make(opr, wrt_idx);
    if (wrt_idx == 0) {
        return out_grad.at(0);
    }
    return IndexingMultiAxisVec::make(out_grad.at(0), opr.index_desc()).node();
}
#endif

/* ============================= Mesh Indexing ============================ */

MGB_IMPL_FANCY_INDEXING_OPR_GET(
        MeshIndexing, "mesh_indexing", false,
        output(0)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE););
MGB_IMPL_FANCY_INDEXING_OPR_GET(
        BatchedMeshIndexing, "batched_mesh_indexing", false,
        output(0)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE););

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(MeshIndexing) {
    if (wrt_idx != 0) {
        return InvalidGrad::make(opr, wrt_idx);
    }
    return IncrMeshIndexing::make(
                   SymbolVar{opr.input(0)}.fill_retain_dtype(0), out_grad.at(0),
                   opr.index_desc())
            .node();
}
#endif

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(BatchedMeshIndexing) {
    if (wrt_idx != 0) {
        return InvalidGrad::make(opr, wrt_idx);
    }
    return BatchedIncrMeshIndexing::make(
                   SymbolVar{opr.input(0)}.fill_retain_dtype(0), out_grad.at(0),
                   opr.index_desc())
            .node();
}
#endif

/* ========================= IncrMeshIndexing ========================= */

MGB_IMPL_FANCY_INDEXING_OPR_MODIFY(IncrMeshIndexing, "incr_mesh_indexing",
                                   false);

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(IncrMeshIndexing) {
    if (wrt_idx > 2) {
        return opr::InvalidGrad::make(opr, wrt_idx);
    }
    if (wrt_idx == 0) {
        return out_grad.at(0);
    }
    return MeshIndexing::make(out_grad.at(0), opr.index_desc()).node();
}
#endif

MGB_IMPL_FANCY_INDEXING_OPR_MODIFY(BatchedIncrMeshIndexing,
                                   "batched_incr_mesh_indexing", false);
#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(BatchedIncrMeshIndexing) {
    if (wrt_idx > 2) {
        return opr::InvalidGrad::make(opr, wrt_idx);
    }
    if (wrt_idx == 0) {
        return out_grad.at(0);
    }
    return BatchedMeshIndexing::make(out_grad.at(0), opr.index_desc()).node();
}
#endif

/* ======================== SetMeshIndexing =========================== */
MGB_IMPL_FANCY_INDEXING_OPR_MODIFY(SetMeshIndexing, "set_mesh_indexing", false);

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(SetMeshIndexing) {
    if (wrt_idx >= 2) {
        return opr::InvalidGrad::make(opr, wrt_idx);
    }
    if (wrt_idx == 0) {
        return SetMeshIndexing::make(
                       out_grad.at(0),
                    SymbolVar{opr.input(1)}.fill_retain_dtype(0),
                       opr.index_desc())
                .node();
    } else {
        return MeshIndexing::make(out_grad.at(0), opr.index_desc()).node();
    }
}
#endif

MGB_IMPL_FANCY_INDEXING_OPR_MODIFY(BatchedSetMeshIndexing,
                                   "batched_set_mesh_indexing", false);
#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(BatchedSetMeshIndexing) {
    if (wrt_idx > 2) {
        return opr::InvalidGrad::make(opr, wrt_idx);
    }
    if (wrt_idx == 0) {
        return BatchedSetMeshIndexing::make(
                       out_grad.at(0),
                       SymbolVar{opr.input(1)}.fill_retain_dtype(0),
                       opr.index_desc())
                .node();
    } else {
        return BatchedMeshIndexing::make(out_grad.at(0), opr.index_desc())
                .node();
    }
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
