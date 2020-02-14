/**
 * \file dnn/src/naive/relayout/opr_impl.h
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

namespace megdnn {
namespace naive {

    class RelayoutForwardImpl: public RelayoutForward {
        protected:
            //! check that src_handle is on CPU
            void check_cpu_handle(Handle *src_handle);

            void do_exec(_megdnn_tensor_in src, _megdnn_tensor_out dst);

        public:
            using RelayoutForward::RelayoutForward;

            bool is_thread_safe() const override {
                return true;
            }

            void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                    Handle *src_handle) override;
    };

}
}

// vim: syntax=cpp.doxygen
