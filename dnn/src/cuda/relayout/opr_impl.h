#pragma once

#include "megdnn/oprs.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

class RelayoutForwardImpl final : public RelayoutForward {
    class Param {
        TensorND m_src, m_dst;
        RelayoutForwardImpl* const m_opr;

    public:
        Param(const TensorND& src, const TensorND& dst, RelayoutForwardImpl* opr);

        size_t dtype_size() const { return m_src.layout.dtype.size(); }

        //! try to transpose
        bool try_transpose();

        //! try to copy by cudaMemcpy
        bool try_copy_contig();

        //! try to copy by cudaMemcpy2DAsync
        bool try_copy_2d(bool cross_dev);

        void copy_general();

        //! try to copy if last contiguous
        bool try_copy_last_contig();
    };

    //! expand *dst* to 2 dims to match *src*
    static bool expand_dim2(TensorLayout& dst, const TensorLayout& src);

    cudaStream_t stream() const { return cuda_stream(handle()); }

public:
    using RelayoutForward::RelayoutForward;

    bool is_thread_safe() const override { return true; }

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, Handle* src_handle) override;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
