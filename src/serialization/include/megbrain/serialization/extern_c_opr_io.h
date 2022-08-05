#pragma once

#include "megbrain/graph.h"
#include "megbrain/serialization/extern_c_opr.h"
#include "megbrain/serialization/opr_registry.h"

namespace mgb {
namespace opr {

//! an operator to run extern C oprs
MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        ExternCOprRunner, cg::SingleCNOutshapePureByInshapeOprBase) // {
    std::shared_ptr<MGBOprDesc> m_desc;
    //! store ExternCOprRunner opr full dump name
    std::string m_dump_name;
    //! store dynamic store param
    std::shared_ptr<ExternCOprParam> m_param;

    //!  HostTensorND holder for scn_do_execute
    SmallVector<HostTensorND> cpu_inp, cpu_out;

    void get_output_var_shape(
            const TensorShapeArray& inp_shape,
            TensorShapeArray& out_shape) const override;
    void scn_do_execute() override;
    void add_input_layout_constraint() override;
    void init_output_dtype() override;
    void check_param();
    bool is_loader_support_dynamic_param;

    static cg::OperatorNodeBase* make_from_desc_shared(
            std::string& name, const VarNodeArray& inputs,
            std::shared_ptr<MGBOprDesc> desc, const OperatorNodeConfig& config);

public:
    MGE_WIN_DECLSPEC_FUC ExternCOprRunner(
            std::string& name, const VarNodeArray& inputs,
            std::shared_ptr<MGBOprDesc> desc, const OperatorNodeConfig& config);

    //! create from MGBOprDesc and steal its reference
    MGE_WIN_DECLSPEC_FUC static cg::OperatorNodeBase* make_from_desc(
            std::string& name, const VarNodeArray& inputs, MGBOprDesc* desc,
            const OperatorNodeConfig& config = {});

    /*!
     * \brief make a placeholder so this opr can be placed in the graph to
     *      produce a graph dump
     *
     * Note: this operator can not be executed
     *
     * \param output_shapes predefined output shapes
     * \param name operator dump name that should match the name in MGBOprLoader
     * \param data data to be written to file for dump
     * \param data_len length of \p data
     * \param output_dtypes predefined output dtypes
     */
    MGE_WIN_DECLSPEC_FUC static cg::OperatorNodeBase* make_placeholder(
            const SymbolVarArray& inputs, const TensorShapeArray& output_shapes,
            const char* name, const void* data, size_t data_len,
            const OperatorNodeConfig& config = {},
            const SmallVector<DType>& output_dtypes = {});

    /*!
     * \brief unregister a MGBOprLoader
     * \return whether any loader is removed (i.e. whether the name exists)
     */
    MGE_WIN_DECLSPEC_FUC static bool unregister_loader(const char* name);

    //! impl for serialization dump
    MGE_WIN_DECLSPEC_FUC static void dump(
            serialization::OprDumpContext& ctx, const cg::OperatorNodeBase& opr);

    //! impl for serialization load
    MGE_WIN_DECLSPEC_FUC static cg::OperatorNodeBase* load(
            serialization::OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config);

    //! impl for serialization shallow copy
    MGE_WIN_DECLSPEC_FUC static cg::OperatorNodeBase* shallow_copy(
            const serialization::OprShallowCopyContext& ctx,
            const cg::OperatorNodeBase& opr, const VarNodeArray& inputs,
            const OperatorNodeConfig& config);

    //! helper for converting TensorShape to MGBTensorShape
    MGE_WIN_DECLSPEC_FUC static ::MGBTensorShape tensor_shape_to_c(
            const TensorShape& shape);

    //! helper for converting MGBTensorShape to TensorShape
    MGE_WIN_DECLSPEC_FUC static TensorShape tensor_shape_from_c(
            const MGBTensorShape& shape);

    const std::string& get_dump_name() { return m_dump_name; }

    void set_param(const std::shared_ptr<ExternCOprParam>& param) {
        mgb_assert(
                is_loader_support_dynamic_param,
                "set_param function need loader imp dynamic_param");
        m_param = param;
        m_desc->dynamic_param = m_param.get();
    }
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
