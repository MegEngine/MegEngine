/**
 * \file dnn/src/fallback/relayout/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/oprs.h"
#include "src/naive/relayout/opr_impl.h"
#include "src/common/relayout_helper.h"

namespace megdnn {
namespace fallback {

    using NaiveRelayoutForwardImpl = naive::RelayoutForwardImpl;

    class RelayoutForwardImpl : public NaiveRelayoutForwardImpl {
        public:
            using NaiveRelayoutForwardImpl::NaiveRelayoutForwardImpl;

            bool is_thread_safe() const override {
                return true;
            }

            void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                    Handle *src_handle) override;
        protected:

            /*!
             * exec after src and dst has been processed by
             * check_layout_and_canonize() and is_transpose()
             *
             * \param transpose pointer to the transpose param if it is a
             *      tranpose, or nullptr if it is not a transpose; note that it
             *      might be modified
             */
            void exec_after_preprocess(const TensorND& src, const TensorND& dst,
                    relayout::TransposeParam* transpose);
    };

}
}

// vim: syntax=cpp.doxygen


