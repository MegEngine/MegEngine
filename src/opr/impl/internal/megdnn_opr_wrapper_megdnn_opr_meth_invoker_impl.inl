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
#define _cb_ref_in(_x) inout[_x]
#define _cb_ref_out(_x) inout[_NR_INPUTS + _x]
#define _cb_in(_x) \
    { ishp[_x], mgb_opr->input(_x)->dtype(), mgb_opr->input(_x)->format() }
#define _cb_unused(_x) {}
#define _cb_ptr_in(_x) &(_cb_ref_in(_x))
#define _cb_ptr_out(_x) &(_cb_ref_out(_x))
    template<class Opr>
    static inline size_t get_workspace_in_bytes(
            Opr *opr, const cg::OperatorNodeBase *mgb_opr,
            const TensorShapeArray &ishp,
            const TensorShapeArray &oshp) {
#define _cb_out(_x) \
    { oshp[_x], mgb_opr->output(_x)->dtype(), mgb_opr->output(_x)->format() }
        TensorLayout inout[_NR_INPUTS + _NR_OUTPUTS] = {
            _FOREACH_IO(_cb_in, _cb_out)
        };
        MegDNNOprInputsLayoutModifier<Opr>::apply(opr->param(), {
            _FOREACH_IO(_cb_ptr_in, _cb_ptr_out)
        });
        return opr->get_workspace_in_bytes(
                _FOREACH_IO(_cb_ref_in, _cb_ref_out)
                );
#undef _cb_out
    }

    template<class Opr>
    static inline void deduce_layout(
            Opr *opr, const cg::OperatorNodeBase *mgb_opr,
            const TensorShapeArray &ishp,
            TensorShapeArray &oshp) {
#define _cb_out(_x) \
    { mgb_opr->output(_x)->dtype(), mgb_opr->output(_x)->format() }
        TensorLayout inout[_NR_INPUTS + _NR_OUTPUTS] = {
            _FOREACH_IO(_cb_in, _cb_out)
        };
        MegDNNOprInputsLayoutModifier<Opr>::apply(opr->param(), {
            _FOREACH_IO(_cb_ptr_in, _cb_ptr_out)
        });
        opr->deduce_layout(
                _FOREACH_IO(_cb_ref_in, _cb_ref_out)
                );
        for (int i = 0; i < _NR_OUTPUTS; ++ i)
            oshp[i] = _cb_ref_out(i);
    }
#undef _cb_out
#undef _cb_ptr_out
#undef _cb_ptr_in
#undef _cb_unused
#undef _cb_in
#undef _cb_ref_out
#undef _cb_ref_in

    template<class Opr>
    static inline void exec(Opr *opr, const cg::OperatorNodeBase *mgb_opr) {
#define _cb_ref_in(_x) inout[_x]
#define _cb_ref_out(_x) inout[_NR_INPUTS + _x]
#define _cb_in(_x) mgb_opr->input(_x)->dev_tensor().as_megdnn()
#define _cb_out(_x) mgb_opr->output(_x)->dev_tensor().as_megdnn()
#define _cb_ptr_in(_x) &(_cb_ref_in(_x).layout)
#define _cb_ptr_out(_x) &(_cb_ref_out(_x).layout)
        megdnn::TensorND inout[_NR_INPUTS + _NR_OUTPUTS] = {
            _FOREACH_IO(_cb_in, _cb_out)
        };
        MegDNNOprInputsLayoutModifier<Opr>::apply(opr->param(), {
            _FOREACH_IO(_cb_ptr_in, _cb_ptr_out)
        });
        opr->exec(
                _FOREACH_IO(_cb_ref_in, _cb_ref_out),
                get_megdnn_workspace_from_var(mgb_opr->output().back()));
#undef _cb_ptr_out
#undef _cb_ptr_in
#undef _cb_out
#undef _cb_in
#undef _cb_ref_out
#undef _cb_ref_in
    }
};

#undef _FOREACH_IO
#undef _NR_OUTPUTS
#undef _NR_INPUTS

// vim: ft=txt syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
