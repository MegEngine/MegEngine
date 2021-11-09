/**
 * \file imperative/python/test/unit/core/custom_opsrc/matmul_scale.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/custom/custom.h"

using Tensor = custom::Tensor;

void matmul_forward_helper(
        const Tensor& lhs, const Tensor& rhs, Tensor& res, size_t M, size_t K, size_t N,
        float scale);
void matmul_backward_lhs_helper(
        const Tensor& rhs, const Tensor& ograd, Tensor& lhs_grad, size_t M, size_t K,
        size_t N, float scale);
void matmul_backward_rhs_helper(
        const Tensor& lhs, const Tensor& ograd, Tensor& rhs_grad, size_t M, size_t K,
        size_t N, float scale);
