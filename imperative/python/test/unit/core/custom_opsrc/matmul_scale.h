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
