/**
 * \file dnn/src/x86/handle.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/fallback/handle.h"

#if MEGDNN_X86_WITH_MKL_DNN
#include <mkldnn.hpp>
#endif

namespace megdnn {
namespace x86 {

class HandleImpl : public fallback::HandleImpl {
public:
    HandleImpl(megcoreComputingHandle_t computing_handle,
               HandleType type = HandleType::X86);

    template <typename Opr>
    std::unique_ptr<Opr> create_operator();

    size_t alignment_requirement() const override;
#if MEGDNN_X86_WITH_MKL_DNN
    dnnl::engine mkldnn_engine() { return m_mkldnn_engine; }
    dnnl::stream mkldnn_stream() { return m_mkldnn_stream; }
#endif

private:
#if MEGDNN_X86_WITH_MKL_DNN
    dnnl::engine m_mkldnn_engine;
    dnnl::stream m_mkldnn_stream;
#endif
};

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
