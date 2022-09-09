
#include "megdnn/dtype.h"

#include "../cond_take/opr_impl.h"
#include "./kernel.cuh"
#include "./opr_impl.h"
#include "src/common/utils.h"
#include "src/cuda/cond_take/kern.cuh"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace megdnn::cuda;
using namespace megdnn::cuda::non_zero;

WorkspaceBundle NonZeroImpl::make_bundle(const TensorLayout& data) {
    size_t nr_item = data.total_nr_elems();
    cuda_check(cudaSetDevice(concrete_handle(handle())->device_id()));
    auto gen_idx_wk_size = cuda::cond_take::gen_idx_get_workspace_size(nr_item);
    SmallVector<size_t> sizes_in_bytes;
    sizes_in_bytes.push_back((nr_item + 1) * sizeof(megdnn::cuda::cond_take::IdxType));
    sizes_in_bytes.push_back(gen_idx_wk_size);
    // the two ele is the shape of arr and the reverse multiply arr of the shape
    sizes_in_bytes.push_back(sizeof(TensorLayout::shape));
    sizes_in_bytes.push_back(sizeof(TensorLayout::shape));
    return {nullptr, sizes_in_bytes, handle()->alignment_requirement()};
}

size_t NonZeroImpl::get_workspace_in_bytes(const TensorLayout& data) {
    return make_bundle(data).total_size_in_bytes();
}

TensorND NonZeroImpl::exec(
        _megdnn_tensor_in data, _megdnn_workspace workspace,
        DynOutMallocPolicyCall malloc_policy) {
    size_t size = data.layout.total_nr_elems();
    if (size == 0) {
        TensorShape target_shape({data.layout.ndim, 0});
        TensorND rst = malloc_policy.alloc_output(0, dtype::Int32(), target_shape);
        return rst;
    }
    auto wk_bundle = make_bundle(data.layout);
    wk_bundle.set(workspace.raw_ptr);

    auto idx_tmp = static_cast<megdnn::cuda::cond_take::IdxType*>(wk_bundle.get(0));

    CondTake::Param param;
    param.mode = CondTake::Param::Mode::NEQ;
    param.val = 0;
    param.eps = 1e-06;
    megdnn::cond_take::KParam kparam(param);

    auto stream = cuda_stream(handle());
    size_t out_size;
    switch (data.layout.dtype.enumv()) {
#define cb(_dt)                                                                      \
    case DTypeTrait<_dt>::enumv: {                                                   \
        using ctype = DTypeTrait<_dt>::ctype;                                        \
        out_size = megdnn::cuda::cond_take::gen_idx(                                 \
                wk_bundle.get(1), wk_bundle.get_size(1), idx_tmp, data.ptr<ctype>(), \
                size, static_cast<uint32_t>(param.mode), kparam, stream);            \
        break;                                                                       \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        cb(::megdnn::dtype::Bool)
#undef cb
                default : {
            std::string data_type = data.layout.dtype.name();
            megdnn_throw(
                    "bad mask dtype,support_types is support types: [Float32, Float16, "
                    "BFloat16, Int32, Int16, Int8, Uint8, Bool]" +
                    std::string("but the data type  is ") + data_type);
        }
    }

    TensorShape dst_shape({data.layout.ndim, out_size});
    TensorND out_idx = malloc_policy.alloc_output(0, dtype::Int32(), dst_shape);
    dt_int32* out_idx_ptr = out_idx.ptr<dt_int32>();

    switch (data.layout.dtype.enumv()) {
#define cb(_dt)                                                    \
    case DTypeTrait<_dt>::enumv: {                                 \
        for (size_t idx = 0; idx < data.layout.ndim; idx++) {      \
            dt_int32* copy_idx_ptr = out_idx_ptr + idx * out_size; \
            copy_idx(copy_idx_ptr, idx_tmp, size, stream);         \
        }                                                          \
        break;                                                     \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        cb(::megdnn::dtype::Bool)
#undef cb
                default : megdnn_throw("bad data dtype");
    }
    expansion_index(
            out_idx_ptr, out_size, data.layout.shape,
            static_cast<size_t*>(wk_bundle.get(2)), data.layout.ndim,
            static_cast<dt_int32*>(wk_bundle.get(3)), stream);

    return out_idx;
}