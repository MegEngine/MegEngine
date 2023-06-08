#include "src/naive/matrix_inverse/opr_impl.h"
#include <cmath>
#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;

size_t MatrixInverseImpl::get_workspace_in_bytes(
        size_t batch, size_t n, size_t dtype_size) {
    MEGDNN_MARK_USED_VAR(batch);
    return n * n * 2 * dtype_size + n * sizeof(void*);
}

template <typename ctype>
void do_exec(ctype* dst, const ctype* src, size_t batch, size_t n, void* workspace) {
    auto row_ptr = static_cast<ctype**>(workspace);
    auto exmat = reinterpret_cast<ctype*>(row_ptr + n);
    for (size_t b = 0; b < batch; ++b, src += n * n, dst += n * n) {
        // exmat is [A | I] and row_ptr points to its rows
        for (size_t i = 0; i < n; ++i) {
            row_ptr[i] = exmat + i * n * 2;
            memcpy(row_ptr[i], src + i * n, sizeof(ctype) * n);
            memset(row_ptr[i] + n, 0, sizeof(ctype) * n);
            row_ptr[i][n + i] = 1;
        }
        for (size_t i = 0; i < n; ++i) {
            size_t pivot_row = 0;
            // select pivot row that has max abs value
            ctype pivot_row_val = static_cast<ctype>(0);
            for (size_t j = i; j < n; ++j) {
                ctype val = static_cast<ctype>(std::abs(row_ptr[j][i]));
                if (val > pivot_row_val) {
                    pivot_row_val = val;
                    pivot_row = j;
                }
            }
            megdnn_throw_if(
                    pivot_row_val < ctype(1e-7), megdnn_error, "pivot value too small");
            std::swap(row_ptr[i], row_ptr[pivot_row]);

            // substract pivot row from other rows
            auto pivot_row_ptr = row_ptr[i];
            for (size_t j = 0; j < n; ++j) {
                if (j == i) {
                    continue;
                }
                ctype inv_pivot = -row_ptr[j][i] / pivot_row_ptr[i];
                for (size_t k = i; k < n * 2; ++k) {
                    row_ptr[j][k] += pivot_row_ptr[k] * inv_pivot;
                }
            }

            // scale pivot row after subtracting it from other rows
            {
                ctype scale = (static_cast<ctype>(1)) / pivot_row_ptr[i];
                for (size_t j = i; j < n * 2; ++j) {
                    pivot_row_ptr[j] *= scale;
                }
            }
        }

        for (size_t i = 0; i < n; ++i) {
            memcpy(dst + i * n, row_ptr[i] + n, sizeof(ctype) * n);
        }
    }
}

void MatrixInverseImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    size_t batch, n;
    check_exec(src.layout, dst.layout, workspace, &batch, &n);
#define cb(DType)                                                                  \
    if (dst.layout.dtype == DType()) {                                             \
        using ctype = typename DTypeTrait<DType>::ctype;                           \
        MEGDNN_DISPATCH_CPU_KERN_OPR(do_exec<ctype>(                               \
                dst.ptr<ctype>(), src.ptr<ctype>(), batch, n, workspace.raw_ptr)); \
        return;                                                                    \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_assert_internal(0);
#else
    __builtin_trap();
#endif
}

// vim: syntax=cpp.doxygen
