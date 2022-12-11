#include "src/cuda/dropout/opr_impl.h"

namespace megdnn {
namespace cuda {

using Param = megdnn::Dropout::Param;

struct DropoutTensorDesc : public TensorDesc {
public:
    DropoutTensorDesc(const TensorLayout& layout) : TensorDesc() {
        set_dropout_desc(layout);
    }
    void set_dropout_desc(const TensorLayout& layout) {
        cudnnDataType_t cudnn_dtype;
        switch (layout.dtype.enumv()) {
            case DTypeEnum::Float32:
                cudnn_dtype = CUDNN_DATA_FLOAT;
                break;
            case DTypeEnum::Float16:
                cudnn_dtype = CUDNN_DATA_HALF;
                break;
            default:
                megdnn_throw("dtype must be float16/float32");
        }
        cudnn_check(cudnnSetTensor4dDescriptor(
                desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, 1, 1,
                layout.total_nr_elems()));
    }
};

size_t DropoutForwardImpl::get_mask_size_in_bytes(const TensorLayout& inp) {
    if (inp.is_empty()) {
        return 0;
    }

    size_t reserve_space_size_in_bytes = 0;
    DropoutTensorDesc ddesc(inp);
    cudnn_check(
            cudnnDropoutGetReserveSpaceSize(ddesc.desc, &reserve_space_size_in_bytes));
    return reserve_space_size_in_bytes;
}

void DropoutForwardImpl::exec(
        _megdnn_tensor_in inp, _megdnn_tensor_out oup, _megdnn_tensor_out mask,
        _megdnn_workspace workspace) {
    check_exec(inp.layout, oup.layout, mask.layout, workspace.size);
    uint64_t seed = param().seed;
    float drop_prob = param().drop_prob;

    if (!dropout_status.initialized()) {
        dropout_status.set(cudnn_handle(this->handle()), seed, drop_prob);
    }
    if (dropout_status.drop_prob != drop_prob) {
        dropout_status.drop_prob = drop_prob;
        dropout_status.restore_desc(cudnn_handle(this->handle()));
    }
    megdnn_assert(dropout_status.seed == seed);

    DropoutTensorDesc inp_desc(inp.layout), oup_desc(oup.layout);
    auto&& op_desc = dropout_status.desc;

    cudnn_check(cudnnDropoutForward(
            cudnn_handle(this->handle()), op_desc.desc, inp_desc.desc, inp.raw_ptr(),
            oup_desc.desc, oup.raw_ptr(), mask.raw_ptr(),
            mask.layout.total_nr_elems()));
}

void DropoutBackwardImpl::exec(
        _megdnn_tensor_in doup, _megdnn_tensor_in mask, _megdnn_tensor_out dinp,
        _megdnn_workspace workspace) {
    check_exec(doup.layout, mask.layout, dinp.layout, workspace.size);

#if CUDNN_VERSION >= 7000
    size_t status_size_in_bytes = 0;
    cudnn_check(cudnnDropoutGetStatesSize(
            cudnn_handle(this->handle()), &status_size_in_bytes));

    DropoutTensorDesc doup_desc(doup.layout), dinp_desc(dinp.layout);
    op_desc.restore(
            cudnn_handle(this->handle()), param().drop_prob, nullptr,
            status_size_in_bytes, 0);
    cudnn_check(cudnnDropoutBackward(
            cudnn_handle(this->handle()), op_desc.desc, doup_desc.desc, doup.raw_ptr(),
            dinp_desc.desc, dinp.raw_ptr(), mask.raw_ptr(),
            mask.layout.total_nr_elems()));
#else
    uint64_t seed = param().seed;
    float drop_prob = param().drop_prob;

    if (!dropout_status.initialized()) {
        dropout_status.set(cudnn_handle(this->handle()), seed, drop_prob);
    }
    if (dropout_status.drop_prob != drop_prob) {
        dropout_status.drop_prob = drop_prob;
        dropout_status.restore_desc(cudnn_handle(this->handle()));
    }

    auto&& op_desc = dropout_status.desc;
    DropoutTensorDesc doup_desc(doup.layout), dinp_desc(dinp.layout);

    cudnn_check(cudnnDropoutBackward(
            cudnn_handle(this->handle()), op_desc.desc, doup_desc.desc, doup.raw_ptr(),
            dinp_desc.desc, dinp.raw_ptr(), mask.raw_ptr(),
            mask.layout.total_nr_elems()));
#endif
}

}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
