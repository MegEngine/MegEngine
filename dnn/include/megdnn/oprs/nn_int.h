/**
 * \file dnn/include/megdnn/oprs/nn_int.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/internal/opr_header_prologue.h"

namespace megdnn {

/*!
 * \brief element-wise operator that allows input/output vars to have different
 *      data types
 *
 * The data types are typically different int types.
 */
class ElemwiseMultiType : public OperatorBase {
    DEF_OPR_PARAM(ElemwiseMultiType);
    DEF_OPR_IMPL(ElemwiseMultiType, OperatorBase, -1, 1);

    //! check dtype function
    using CheckDtypeFunc = thin_function<void(const DType)>;
    //! check the dtype if is_check is true, otherwise setup dtype.
    using SetOrCheckDtypeFunc = thin_function<void(DType&, bool is_check)>;

public:
    using Mode = Param::Mode;
    static constexpr size_t MAX_ARITY = 6;

    //! information about a mode
    struct ModeTrait {
        uint32_t arity = 0;  //!< number of inputs needed
        CheckDtypeFunc check_inp[MAX_ARITY];
        SetOrCheckDtypeFunc check_out;    //!< dtype of output var
        bool need_specify_out_dtype =
                false;  //!< the dtype should be setup externally, otherwise
                        //!< would be inferred by check_out(dtype, false)
        const char* name = nullptr;  //!< name of the mode

        //! get trait from a mode; this function is thread safe
        static const ModeTrait& from_mode(Mode mode);
    };

    virtual void exec(_megdnn_in const TensorNDArray& src,
                      _megdnn_tensor_out dst) = 0;

    //! get trait of current mode
    const ModeTrait& mode_trait() const {
        return ModeTrait::from_mode(m_param.mode);
    }

    //! deduce output layout
    void deduce_layout(const TensorLayoutArray& src, TensorLayout& dst);

protected:
    //! throw exception if incorrect layout; broadcast input shape to
    //! output shape
    void check_layout_and_broadcast(const TensorLayoutPtrArray& src,
                                    const TensorLayout& dst);
};

}  // namespace megdnn

#include "megdnn/internal/opr_header_epilogue.h"
// vim: syntax=cpp.doxygen
