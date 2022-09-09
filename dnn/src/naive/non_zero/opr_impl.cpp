#include "./opr_impl.h"
#include "src/common/cond_take/predicate.cuh"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "src/naive/non_zero/opr_impl.h"

using namespace megdnn;
using namespace naive;

using Param = NonZero::Param;
size_t NonZeroImpl::get_workspace_in_bytes(const TensorLayout& data) {
    // save the size of index array in the last element of workspace
    return (data.total_nr_elems() + 1) * sizeof(dt_int32);
}
template <uint32_t mode, typename ctype>
void gen_index(dt_int32* dest, const TensorND& src, cond_take::Pred<mode, ctype> pred) {
    int idx = 0;
    ctype* inp = src.ptr<ctype>();
    size_t number_of_data = src.layout.total_nr_elems();
    for (size_t data_pos = 0; data_pos < number_of_data; ++data_pos) {
        if (pred(inp[data_pos])) {
            dest[idx++] = data_pos;
        }
    }
    // last element is the size of index array
    dest[number_of_data] = idx;
}

void expansion_index(
        const dt_int32* const index_arr, const size_t index_size, const TensorND* rst,
        const size_t* shape_arr, const int ndim) {
    SmallVector<int, 8> shape_reverse_multiply_reduce_arr({1});
    for (int div_index = 1; div_index < ndim; div_index++) {
        shape_reverse_multiply_reduce_arr[div_index] =
                shape_arr[ndim - div_index] *
                shape_reverse_multiply_reduce_arr[div_index - 1];
    }

    for (int dim_pos = 0; dim_pos < ndim; dim_pos++) {
        dt_int32* dim_pt = rst->ptr<dt_int32>() + index_size * dim_pos;
        for (size_t ele_pos = 0; ele_pos < index_size; ele_pos++) {
            int dim_pos_of_ele = index_arr[ele_pos] /
                                 shape_reverse_multiply_reduce_arr[ndim - 1 - dim_pos];
            int dim_index_of_ele = dim_pos_of_ele % shape_arr[dim_pos];
            dim_pt[ele_pos] = dim_index_of_ele;
        }
    }
}

TensorND NonZeroImpl::exec(
        _megdnn_tensor_in src, _megdnn_workspace workspace,
        DynOutMallocPolicyCall malloc_policy) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    auto idx_tmp = workspace.ptr<dt_int32>();

    switch (src.layout.dtype.enumv()) {
#define cb(_dt)                                                      \
    case DTypeTrait<_dt>::enumv: {                                   \
        using ctype = DTypeTrait<_dt>::ctype;                        \
        using namespace cond_take;                                   \
        KParam param({});                                            \
        param.val = 0.0;                                             \
        param.eps = 1e-6;                                            \
        Pred<PEnum::NEQ, ctype> pred(param);                         \
        MEGDNN_DISPATCH_CPU_KERN_OPR(gen_index(idx_tmp, src, pred)); \
        break;                                                       \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        cb(::megdnn::dtype::Bool)
#undef cb
                default : {
            std::string data_type = src.layout.dtype.name();
            megdnn_throw(
                    "bad mask dtype,support_types is support types: [Float32, Float16, "
                    "BFloat16, Int32, Int16, Int8, Uint8, Bool]" +
                    std::string("but the data type  is ") + data_type);
        }
    }

    static_cast<HandleImpl*>(handle())->megcore_dispatcher()->sync();
    size_t index_size_pos = src.layout.total_nr_elems();
    size_t index_size = idx_tmp[index_size_pos];
    TensorND ret;
    size_t ndim = src.layout.ndim;
    TensorShape dst_shape({ndim, index_size});
    ret = malloc_policy.alloc_output(0, dtype::Int32(), {ndim, index_size});
    MEGDNN_DISPATCH_CPU_KERN_OPR(
            expansion_index(idx_tmp, index_size, &ret, src.layout.shape, ndim));

    return ret;
#else
    __builtin_trap();
    return {};
#endif
}
