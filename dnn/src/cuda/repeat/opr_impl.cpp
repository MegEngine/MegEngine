#include "src/cuda/repeat/opr_impl.h"

#include "src/common/tile_repeat_helper.h"
#include "src/cuda/repeat/repeat.cuh"
#include "src/cuda/utils.h"

#include <numeric>

namespace megdnn {
namespace cuda {

void RepeatForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    auto stream = cuda_stream(this->handle());
    TensorShape sshape, dshape, tshape;
    simplify_shape(src.layout, dst.layout, param().times, sshape, dshape, tshape);
#define cb(DType)                                                              \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {                \
        using ctype = typename DTypeTrait<DType>::ctype;                       \
        repeat::forward_proxy<ctype>(                                          \
                src.ptr<ctype>(), dst.ptr<ctype>(), sshape.ndim, sshape.shape, \
                dshape.shape, tshape.shape, stream);                           \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

RepeatBackwardImpl::RepeatBackwardImpl(Handle* handle)
        : RepeatBackward(handle), m_opr(handle->create_operator<Reduce>()) {
    m_opr->param().mode = Reduce::Mode::SUM;
}

template <typename T>
void RepeatBackwardImpl::exec_internal(
        _megdnn_tensor_in diff_, _megdnn_tensor_out grad_,
        _megdnn_workspace workspace) {
    TensorShape grad, diff, times;
    simplify_shape(grad_.layout, diff_.layout, param().times, grad, diff, times);
    auto stream = cuda_stream(this->handle());
    auto nr_reduces = count_not_ones_in_shape(times);
    auto dtype = diff_.layout.dtype;
    if (nr_reduces == 0) {
        cuda_check(cudaMemcpyAsync(
                grad_.ptr<T>(), diff_.ptr<T>(), sizeof(T) * diff.total_nr_elems(),
                cudaMemcpyDeviceToDevice, stream));
    } else {
        auto ndim = times.ndim;
        WorkspaceBundle workspaces(
                workspace.raw_ptr,
                {diff.total_nr_elems() * sizeof(T), diff.total_nr_elems() * sizeof(T)});
        auto workspace0 = static_cast<T*>(workspaces.get(0));
        auto workspace1 = static_cast<T*>(workspaces.get(1));

        T *current, *next;
        size_t state;

        init_tile_repeat_state(
                diff_.ptr<T>(), grad_.ptr<T>(), workspace0, workspace1, current, next,
                state, nr_reduces);

        TensorND reduce_src, reduce_dst;
        for (size_t j = 0; j < ndim; ++j) {
            size_t i = j + 1;
            if (times.shape[j] != 1) {
                // m = sshape[0]*...*sshape[i-1]
                auto m = std::accumulate(
                        grad.shape, grad.shape + i, 1_z, SafeMultiplies<size_t>());
                // n = dshape[i]*...
                auto n = std::accumulate(
                        diff.shape + i, diff.shape + ndim, 1_z,
                        SafeMultiplies<size_t>());
                // forward is repeat (m, n) to (m*times, n)
                // backward is reduce (m, times, n) to (m, 1, n)
                m_opr->param().axis = 1;

                reduce_src.reset_ptr(current);
                reduce_src.layout = TensorLayout(TensorShape{m, times[j], n}, dtype);
                reduce_dst.reset_ptr(next);
                reduce_dst.layout = TensorLayout(TensorShape{m, 1u, n}, dtype);
                m_opr->exec(reduce_src, reduce_dst, Workspace());
                update_tile_repeat_state(
                        diff_.ptr<T>(), grad_.ptr<T>(), workspace0, workspace1, current,
                        next, state, nr_reduces);
            }
        }
        megdnn_assert_internal(current == grad_.ptr<T>());
        megdnn_assert_internal(next == nullptr);
        megdnn_assert_internal(state == nr_reduces);
    }
}

void RepeatBackwardImpl::exec(
        _megdnn_tensor_in diff_, _megdnn_tensor_out grad_,
        _megdnn_workspace workspace) {
    check_exec(diff_.layout, grad_.layout, workspace.size);
#define cb(DType)                                        \
    if (diff_.layout.dtype == DType()) {                 \
        using ctype = typename DTypeTrait<DType>::ctype; \
        exec_internal<ctype>(diff_, grad_, workspace);   \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

size_t RepeatBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& diff, const TensorLayout& grad) {
    return get_workspace_in_bytes_fwd(grad, diff);
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
