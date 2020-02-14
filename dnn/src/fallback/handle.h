/**
 * \file dnn/src/fallback/handle.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/naive/handle.h"
#include "src/common/utils.h"

#include <mutex>

namespace megdnn {
namespace fallback {

class HandleImpl: public naive::HandleImpl {
    public:
        HandleImpl(megcoreComputingHandle_t computing_handle,
                HandleType type = HandleType::FALLBACK):
            naive::HandleImpl::HandleImpl(computing_handle, type)
        {
        }

        template <typename Opr>
        std::unique_ptr<Opr> create_operator();

        //! global relayout opr
        Relayout* relayout_opr() override final {
            return get_helper_opr<Relayout, 3>(this);
        }

};

} // namespace fallback
} // namespace megdnn

// vim: syntax=cpp.doxygen
