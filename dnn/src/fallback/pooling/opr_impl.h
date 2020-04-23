/**
 * \file dnn/src/fallback/pooling/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "megdnn/oprs/base.h"
#include "src/naive/pooling/opr_impl.h"

namespace megdnn {
namespace fallback {

class PoolingImpl : public naive::PoolingForwardImpl {
public:
    using naive::PoolingForwardImpl::PoolingForwardImpl;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) override;

private:
    void exec_w3x3_s1x1(_megdnn_tensor_in src, _megdnn_tensor_out dst);
    void exec_w2x2_s2x2_int8(_megdnn_tensor_in src, _megdnn_tensor_out dst);
    void exec_w2x2_s2x2_avg_int8(_megdnn_tensor_in src, _megdnn_tensor_out dst);
};
}  // namespace fallback
}  // namespace megdnn
// vim: syntax=cpp.doxygen

