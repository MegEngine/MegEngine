/**
 * \file src/opr/include/megbrain/opr/dnn/batch_norm.h
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
#include "megbrain/opr/internal/identical_fwd.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/opr/param_defs.h"

#include "megdnn/oprs/nn.h"

namespace mgb {
namespace opr {

/* input:
 *   x, scale, bias, [running_mean, running_variance]
 * output:
 *   running_mean, running_variance, save_mean, save_inv_variance, y
 *
 * All params have the same definition with cudnn batch normalization.
 *
 * Use sample variance to calculate y and use unbias variance to update running_var:
 *   y = scale * ((x - mean) / sqrt(variance + eps)) + bias
 *       where variance(sample) = sigma((x - mean)^2) / m
 *   save_variance updated by variance(unbias) = sigma((x - mean)^2) / (m - 1)
 *
 * For statistic(mean and variance) update:
 *     running_mean = (1 - moving_average) * running_mean + moving_average * new_mean
 */
MGB_DEFINE_OPR_CLASS(BatchNormForward,
    cg::OutshapePureByInshapeOpr<
    intl::WorkspaceSizeInfer<
    cg::SingleCNOperatorNodeBaseT<
    mixin::MegDNNOprHolderImpl<megdnn::BN>>>>) // {
    public:

        BatchNormForward(VarNode *x, VarNode *scale, VarNode *bias,
                VarNode *mean, VarNode *variance,
                const Param &param,
                const OperatorNodeConfig &config);

        BatchNormForward(VarNode *x, VarNode *scale, VarNode *bias,
                const Param &param,
                const OperatorNodeConfig &config);

        static SymbolVarArray make(SymbolVar x,
                SymbolVar scale, SymbolVar bias,
                SymbolVar mean, SymbolVar variance,
                const Param &param = {},
                const OperatorNodeConfig &config = {});

        static SymbolVarArray make(SymbolVar x,
                SymbolVar scale, SymbolVar bias,
                const Param &param = {},
                const OperatorNodeConfig &config = {});
    private:

        NodeProp* do_make_node_prop() const override;

        void scn_do_execute() override;
        void add_input_layout_constraint() override;
        void get_output_var_shape(const TensorShapeArray &inp_shape,
            TensorShapeArray &out_shape) const override;
        size_t get_workspace_size_bytes(
            const TensorShapeArray &input_shapes,
            const TensorShapeArray &output_shapes) const override;
        void init_output_static_infer_desc() override;
        void init_output_dtype() override;
};

using BatchNorm = BatchNormForward;

/* input:
 *   x, y_grad, save_mean, save_inv_variance, scale
 * output:
 *   scale_grad, bias_grad, x_grad
 */

MGB_DEFINE_OPR_CLASS(BatchNormBackward,
    intl::MegDNNOprWrapperBwd<megdnn::BNBackward>) // {

    public:
        BatchNormBackward(VarNode *x, VarNode *y_grad,
                VarNode *save_mean, VarNode *save_variance,
                VarNode *scale,
                const Param &param,
                const OperatorNodeConfig &config);
        static SymbolVarArray make(SymbolVar x,
                SymbolVar y_grad, SymbolVar save_mean,
                SymbolVar save_variance, SymbolVar scale,
                const Param &param = {},
                const OperatorNodeConfig &config = {});
    private:
        void init_output_static_infer_desc() override;
        void init_output_dtype() override;
};

} // namespace opr
} // namespace mgb



// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
