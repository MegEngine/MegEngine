/**
 * \file src/opr/include/megbrain/opr/misc.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#if MGB_CUDA
#include "../../../impl/nvof/denseflownvidia.h"
#include "megbrain/opr/param_defs.h"
#endif
#include "megdnn/oprs.h"

#include <array>

namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS(Argmax,
            intl::MegDNNOprWrapperFwd<megdnn::Argmax>) // {

    public:
        Argmax(VarNode *src, const Param &param,
                const OperatorNodeConfig &config);
        static SymbolVar make(SymbolVar src, const Param &param,
                const OperatorNodeConfig &config = {});
};

MGB_DEFINE_OPR_CLASS(Argmin,
            intl::MegDNNOprWrapperFwd<megdnn::Argmin>) // {

    public:
        Argmin(VarNode *src, const Param &param,
                const OperatorNodeConfig &config);
        static SymbolVar make(SymbolVar src, const Param &param,
                const OperatorNodeConfig &config = {});
};

/*!
 * \brief Argsort operator.
 *
 * Performing m independent argsort operations on m arrays of length n.
 *
 * \param[in] in_tensor \f$(m, n)\f$ input tensor
 * \param[out] out_tensor the first output: \f$(m, n)\f$ sorted output tensor
 * \param[out] indices the second output: \f$(m, n)\f$ sorted indices
 */
MGB_DEFINE_OPR_CLASS(ArgsortForward,
            intl::MegDNNOprWrapperFwd<megdnn::ArgsortForward>) // {
    public:
        ArgsortForward(VarNode *in_tensor,
                const Param &param,
                const OperatorNodeConfig &config);

        static std::array<SymbolVar, 2> make(SymbolVar in_tensor,
                const Param &param = {},
                const OperatorNodeConfig &config = {});
};
using Argsort = ArgsortForward;

MGB_DEFINE_OPR_CLASS(ArgsortBackward,
                     intl::MegDNNOprWrapperBwd<megdnn::ArgsortBackward>) // {
public:
    ArgsortBackward(VarNode * out_diff, VarNode * indices,
                    VarNode * result_shape, const Param& param,
                    const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar out_diff, SymbolVar indices,
                          SymbolVar result_shape, const Param& param = {},
                          const OperatorNodeConfig& config = {});
    static SymbolVar make(SymbolVar out_diff, SymbolVar indices,
                          const Param& param = {},
                          const OperatorNodeConfig& config = {}) {
        return make(out_diff, indices, out_diff, param, config);
    }
};

//! cumulative sum along given axis
MGB_DEFINE_OPR_CLASS(Cumsum, cg::SingleCNOperatorNodeBaseT<
        mixin::MegDNNOprHolderImpl<megdnn::Cumsum>>) // {

    public:
        Cumsum(VarNode *src, const Param &param, const OperatorNodeConfig &config);

        // for serialization
        static SymbolVar make(SymbolVar opr, const Param &param,
                const OperatorNodeConfig &config = {});
    protected:
        void scn_do_execute() override;
        void init_output_static_infer_desc() override;
};

#if MGB_CUDA
MGB_DEFINE_OPR_CLASS(NvOf, cg::SingleCNOperatorNodeBase) // {

    public:
        using Param = megdnn::param::NvOf;
        NvOf(VarNode* src, const Param& param,
             const OperatorNodeConfig& config);

        // for serialization
        static SymbolVar make(SymbolVar opr, const Param& param,
                              const OperatorNodeConfig& config = {});

        static SymbolVar make(SymbolVar opr,
                              const OperatorNodeConfig& config = {}) {
            return make(opr, {}, config);
        }

        Param param() const {
            return m_param;
        }

    protected:
        void init_output_dtype() override;
        void scn_do_execute() override;
        void init_output_static_infer_desc() override;

    private:
        std::shared_ptr<NVFlowExtractor> nv_flow_extractor;
        std::vector<size_t> vshape;
        Param m_param;
        std::mutex m_lock;
        bool init_flag = false;
};
#endif

namespace intl {
using CondTakeBase =
        cg::SingleCNOperatorNode<cg::OperatorNodeBase,
                                 mixin::MegDNNOprHolderImpl<megdnn::CondTake>>;
using TopKBase =
        cg::SingleCNOperatorNode<cg::OperatorNodeBase,
                                 mixin::MegDNNOprHolderImpl<megdnn::TopK>>;
}  // namespace intl

/*!
 * \brief take values conditionally
 * outputs: values, indices
 */
MGB_DEFINE_OPR_CLASS(CondTake, intl::CondTakeBase) // {
    void init_output_static_infer_desc() override;
    void scn_do_execute() override;
    void add_input_layout_constraint() override;

    public:
        CondTake(VarNode *data, VarNode *mask,
                const Param &param, const OperatorNodeConfig &config);
        static std::array<SymbolVar, 2> make(
                SymbolVar data, SymbolVar mask,
                const Param &param, const OperatorNodeConfig &config = {});
};

MGB_DEFINE_OPR_CLASS(TopK, intl::TopKBase) // {

    void init_output_dtype() override;
    void add_input_layout_constraint() override;
    void init_output_static_infer_desc() override;
    void scn_do_execute() override;
    void record_execute_deps(ExecDependencyArray& deps) override;

public:
    TopK(VarNode * data, VarNode * k, const Param& param,
         const OperatorNodeConfig& config);

    //! note: for KTH_ONLY mode, the second output would be nullptr
    static std::array<SymbolVar, 2> make(SymbolVar data, SymbolVar k,
                                         const Param& param,
                                         const OperatorNodeConfig& config = {});
};

} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

