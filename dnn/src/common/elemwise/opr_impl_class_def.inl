/**
 * \file dnn/src/common/elemwise/opr_impl_class_def.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

    protected:
        template<int arity>
        void on_arity_dispatched();

        template<int arity>
        void on_arity_dispatched_no_bool();

        template<int arity, DTypeCategory dtype_cat, typename ctype>
        struct ModeDispatcher;

        /*!
         * \brief special impl for FUSE_MUL_ADD3 mode
         * \tparam c_is_scalar see ElemwiseForwardImplHelper::prepare_fma3
         */
        template<typename ctype, bool c_is_scalar>
        void impl_fuse_mul_add3(const ElemwiseOpParamN<3> &params);

        /*!
         * \brief special impl for FUSE_MUL_ADD4 mode
         * \param[out] params see ElemwiseForwardImplHelper::prepare_fma4
         */
        template<typename ctype>
        void impl_fuse_mul_add4(const ElemwiseOpParamN<4> &params);

    public:
        using ElemwiseForwardImplHelper::ElemwiseForwardImplHelper;

        void exec(
                const TensorNDArray &src,
                _megdnn_tensor_out dst) override;

// vim: syntax=cpp.doxygen
