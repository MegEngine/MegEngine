/**
 * \file src/opr/impl/internal/megdnn_opr_wrapper_megdnn_opr_meth_invoker_impl.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifndef _NR_INPUTS
#error "must be included from megdnn_opr_wrapper.inl"
#endif

template<>
struct _MegDNNOprMethInvoker<_NR_INPUTS, _NR_OUTPUTS> {
#define _cb_in(_x) \
    { ishp[_x], mgb_opr->input(_x)->dtype(), mgb_opr->input(_x)->format() }
    template<class Opr>
    static inline size_t get_workspace_in_bytes(
            Opr *opr, const cg::OperatorNodeBase *mgb_opr,
            const TensorShapeArray &ishp,
            const TensorShapeArray &oshp) {
#define _cb_out(_x) \
    { oshp[_x], mgb_opr->output(_x)->dtype(), mgb_opr->output(_x)->format() }
        return opr->get_workspace_in_bytes(_FOREACH_IO(_cb_in, _cb_out));
#undef _cb_out
    }

    template<class Opr>
    static inline void deduce_layout(
            Opr *opr, const cg::OperatorNodeBase *mgb_opr,
            const TensorShapeArray &ishp,
            TensorShapeArray &oshp) {
#define _cb_out(_x) ov[_x]
        TensorLayout ov[_NR_OUTPUTS];
        for (int i = 0; i < _NR_OUTPUTS; ++ i)
            ov[i] = {mgb_opr->output(i)->dtype(), mgb_opr->output(i)->format()};
        opr->deduce_layout(_FOREACH_IO(_cb_in, _cb_out));
        for (int i = 0; i < _NR_OUTPUTS; ++ i)
            oshp[i] = ov[i];
    }
#undef _cb_out
#undef _cb_in

    template<class Opr>
    static inline void exec(Opr *opr, const cg::OperatorNodeBase *mgb_opr) {
#define _cb_in(_x) mgb_opr->input(_x)->dev_tensor().as_megdnn()
#define _cb_out(_x) mgb_opr->output(_x)->dev_tensor().as_megdnn()
        opr->exec(
                _FOREACH_IO(_cb_in, _cb_out),
                get_megdnn_workspace_from_var(mgb_opr->output().back()));
#undef _cb_out
#undef _cb_in
    }
};

#undef _FOREACH_IO
#undef _NR_OUTPUTS
#undef _NR_INPUTS

// vim: ft=txt syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
