#pragma once

#include "megbrain/graph/operator_node.h"
#include "megbrain/opr/internal/indexing_helper.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS(Diag, intl::MegDNNOprWrapperFwd<megdnn::Diag>) // {
public:
    MGE_WIN_DECLSPEC_FUC Diag(
            VarNode* src, const Param& param, const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, const Param& param, const OperatorNodeConfig& config = {});
    void scn_do_execute() override;
    NodeProp* do_make_node_prop() const override;
};

MGB_DEFINE_OPR_CLASS(DiagBackward, cg::SingleCNOperatorNodeBase) // {
public:
    using Param = megdnn::Diag::Param;
    MGE_WIN_DECLSPEC_FUC DiagBackward(
            VarNode* shape, VarNode* value, const Param& param,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar shape, SymbolVar value, const Param& param,
            const OperatorNodeConfig& config = {});

    const Param& param() const { return m_param; }

private:
    Param m_param;
    intl::UniqPtrWithCN<megdnn::Diag> m_dnn_opr;

    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
    NodeProp* do_make_node_prop() const override;

    void record_execute_deps(ExecDependencyArray& deps) override;
};

MGB_DEFINE_OPR_CLASS(
        IndexingOneHot, intl::MegDNNOprWrapperFwd<megdnn::IndexingOneHotForward>) // {
public:
    MGE_WIN_DECLSPEC_FUC IndexingOneHot(
            VarNode* src, VarNode* index, const Param& param,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar index, const Param& param,
            const OperatorNodeConfig& config = {});

private:
    void init_output_dtype() override;
    void scn_do_execute() override;
    NodeProp* do_make_node_prop() const override;
};

MGB_DEFINE_OPR_CLASS(
        IndexingSetOneHot,
        intl::WorkspaceSizeInfer<cg::SingleCNOperatorNodeBaseT<
                mixin::MegDNNOprHolderImpl<megdnn::IndexingSetOneHotForward>>>) // {
public:
    MGE_WIN_DECLSPEC_FUC IndexingSetOneHot(
            VarNode* data, VarNode* index, VarNode* sub, const Param& param,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar data, SymbolVar index, SymbolVar sub, const Param& param,
            const OperatorNodeConfig& config = {});

private:
    void scn_do_execute() override;

    void mem_plan_fwd_in2out_writable() override;
    void init_output_static_infer_desc() override;

    void init_output_dtype() override;
    void add_input_layout_constraint() override;

    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override;
    NodeProp* do_make_node_prop() const override;
};

MGB_DEFINE_OPR_CLASS(
        IndexingRemap, intl::MegDNNOprWrapperFwd<megdnn::IndexingRemap>) // {
public:
    MGE_WIN_DECLSPEC_FUC IndexingRemap(
            VarNode* src, VarNode* map, const Param& param,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar map, const Param& param,
            const OperatorNodeConfig& config = {});

private:
    void init_output_dtype() override;
};

MGB_DEFINE_OPR_CLASS(
        IndexingRemapBackward,
        intl::MegDNNOprWrapperBwd<megdnn::IndexingRemapBackward>) // {
public:
    MGE_WIN_DECLSPEC_FUC IndexingRemapBackward(
            VarNode* out_diff, VarNode* map, VarNode* src_for_shape, const Param& param,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar out_diff, SymbolVar map, SymbolVar src_for_shape,
            const Param& param, const OperatorNodeConfig& config = {});
};

namespace mixin {

template <class Opr>
class IndexingMultiAxisVecMegDNNOprHolder {
    intl::UniqPtrWithCN<Opr> m_dnn_opr;

protected:
    Opr& megdnn_opr(cg::SingleCNOperatorNodeBase& self);

    void register_workspace_infer(
            const indexing::IndexDesc& index_desc, cg::SingleCNOperatorNodeBase& opr,
            VarNode* data, VarNode* value, VarNodeArray idx_arr);

    void record_megdnn_opr(mgb::cg::GraphExecutable::ExecDependencyArray& deps);
};

}  // namespace mixin

namespace intl {
//! mixin helper for multi-axis vec indexing oprs
MGB_DEFINE_CLS_WITH_SUPER(MultiAxisVecFancyIndexingHelper, FancyIndexingHelper) // {
    //! whether warning about changing to Subtensor due to scalar idx has
    //! been printed
    bool m_scalar_idx_warn_printed = false;

    megdnn::IndexingMultiAxisVec::IndexDesc m_megdnn_index_cache;

protected:
    using Super::Super;

    //! return IndexDesc and whether it has an AxisIndexer with
    //! empty shape
    std::pair<const megdnn::IndexingMultiAxisVec::IndexDesc&, bool>
    make_megdnn_index_desc(size_t inp_ndim, bool warn_all_scalar = true);
};

//! mixin helper for multi-axis vec indexing oprs that modify input
template <class Opr>
MGB_DEFINE_CLS_WITH_SUPER(
        IndexingModifyMultiAxisVecHelper, MultiAxisVecFancyIndexingHelper,
        mixin::IndexingMultiAxisVecMegDNNOprHolder<Opr>) // {
    void init_output_static_infer_desc() override final;
    void scn_do_execute() override final;
    void add_input_layout_constraint() override final;

protected:
    using Super::Super;
    NodeProp* do_make_node_prop() const override;
};
}  // namespace intl

template <class Opr>
MGB_DEFINE_CLS_WITH_SUPER(
        IndexingMultiAxisVecBase, intl::MultiAxisVecFancyIndexingHelper,
        mixin::IndexingMultiAxisVecMegDNNOprHolder<Opr>) // {
    void init_output_static_infer_desc() override;
    void scn_do_execute() override;
    NodeProp* do_make_node_prop() const override;
    void record_execute_deps(ExecDependencyArray& deps) override;

public:
    using Super::Super;
};

MGB_DEFINE_OPR_CLASS(
        IndexingSetMultiAxisVec,
        intl::IndexingModifyMultiAxisVecHelper<megdnn::IndexingSetMultiAxisVec>) // {
    NodeProp* do_make_node_prop() const override;

public:
    MGB_DECL_FANCY_INDEXING_OPR_MODIFY(IndexingSetMultiAxisVec);
};

MGB_DEFINE_OPR_CLASS(
        IndexingIncrMultiAxisVec,
        intl::IndexingModifyMultiAxisVecHelper<megdnn::IndexingIncrMultiAxisVec>) // {
public:
    MGB_DECL_FANCY_INDEXING_OPR_MODIFY(IndexingIncrMultiAxisVec);
};

MGB_DEFINE_OPR_CLASS(
        IndexingMultiAxisVec, IndexingMultiAxisVecBase<megdnn::IndexingMultiAxisVec>) // {
public:
    MGB_DECL_FANCY_INDEXING_OPR_GET(IndexingMultiAxisVec);
};

MGB_DEFINE_OPR_CLASS(
        MeshIndexing, IndexingMultiAxisVecBase<megdnn::MeshIndexing>) // {
public:
    MGB_DECL_FANCY_INDEXING_OPR_GET(MeshIndexing);
};

MGB_DEFINE_OPR_CLASS(
        BatchedMeshIndexing, IndexingMultiAxisVecBase<megdnn::BatchedMeshIndexing>) // {
public:
    MGB_DECL_FANCY_INDEXING_OPR_GET(BatchedMeshIndexing);
};

MGB_DEFINE_OPR_CLASS(
        IncrMeshIndexing,
        intl::IndexingModifyMultiAxisVecHelper<megdnn::IncrMeshIndexing>) // {
public:
    MGB_DECL_FANCY_INDEXING_OPR_MODIFY(IncrMeshIndexing);
};

MGB_DEFINE_OPR_CLASS(
        SetMeshIndexing,
        intl::IndexingModifyMultiAxisVecHelper<megdnn::SetMeshIndexing>) // {
public:
    MGB_DECL_FANCY_INDEXING_OPR_MODIFY(SetMeshIndexing);
};

MGB_DEFINE_OPR_CLASS(
        BatchedIncrMeshIndexing,
        intl::IndexingModifyMultiAxisVecHelper<megdnn::BatchedIncrMeshIndexing>) // {
public:
    MGB_DECL_FANCY_INDEXING_OPR_MODIFY(BatchedIncrMeshIndexing);
};

MGB_DEFINE_OPR_CLASS(
        BatchedSetMeshIndexing,
        intl::IndexingModifyMultiAxisVecHelper<megdnn::BatchedSetMeshIndexing>) // {
public:
    MGB_DECL_FANCY_INDEXING_OPR_MODIFY(BatchedSetMeshIndexing);
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
