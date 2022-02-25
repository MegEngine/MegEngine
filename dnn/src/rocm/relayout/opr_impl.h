#pragma once

#include "megdnn/oprs.h"
#include "src/rocm/utils.h"

namespace megdnn {
namespace rocm {

class RelayoutForwardImpl final : public RelayoutForward {
    class Param {
        TensorND m_src, m_dst;
        RelayoutForwardImpl* const m_opr;

    public:
        Param(const TensorND& src, const TensorND& dst, RelayoutForwardImpl* opr);

        size_t dtype_size() const { return m_src.layout.dtype.size(); }

        //! try to copy by cudaMemcpy
        bool try_copy_contig();

        //! try to copy by cudaMemcpy2DAsync
        bool try_copy_2d();

        void copy_general();

        //! try to copy if last contiguous
        bool try_copy_last_contig();
    };

    //! expand *dst* to 2 dims to match *src*
    static bool expand_dim2(TensorLayout& dst, const TensorLayout& src);

    hipStream_t stream() const { return hip_stream(handle()); }

public:
    using RelayoutForward::RelayoutForward;

    bool is_thread_safe() const override { return true; }

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, Handle* src_handle) override;
};

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
