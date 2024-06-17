#include "opr_impl.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

#include "aclnnop/aclnn_cumsum.h"
#include "aclnnop/aclnn_index.h"
#include "aclnnop/aclnn_index_put_impl.h"

using namespace megdnn;
using namespace atlas;

namespace {

bool is_axis_contiguous(std::vector<bool> is_indices_defined) {
    auto start = std::find(is_indices_defined.begin(), is_indices_defined.end(), true);
    auto end = std::find(is_indices_defined.rbegin(), is_indices_defined.rend(), true);
    auto it = std::find(start, end.base(), false);
    return it == end.base();
}

TensorND transpose_src_and_indices(
        const TensorND& src, std::vector<TensorND>& indices,
        std::vector<bool>& is_indices_defined) {
    std::vector<size_t> src_transpose_permute;
    std::vector<TensorND> indices_transpose;

    size_t count = 0;
    for (size_t i = 0; i < is_indices_defined.size(); ++i) {
        if (is_indices_defined[i]) {
            src_transpose_permute.push_back(i);
            indices_transpose.push_back(indices[i]);
            count++;
        }
    }
    for (size_t i = 0; i < is_indices_defined.size(); ++i) {
        if (!is_indices_defined[i]) {
            src_transpose_permute.push_back(i);
            indices_transpose.push_back(indices[i]);
        }
    }
    for (size_t i = 0; i < count; ++i) {
        is_indices_defined[i] = true;
    }
    for (size_t i = count; i < is_indices_defined.size(); ++i) {
        is_indices_defined[i] = false;
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = indices_transpose[i];
    }

    TensorLayout src_transpose_layout = src.layout.dimshuffle(src_transpose_permute);
    TensorND transposed_src(src.raw_ptr(), src_transpose_layout);
    return transposed_src;
}

TensorND broadcast_indexed_indices(
        const TensorND& src, std::vector<TensorND>& indices,
        std::vector<bool>& is_indices_defined) {
    size_t target_indices_ndim = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (is_indices_defined[i]) {
            target_indices_ndim = std::max(target_indices_ndim, indices[i].layout.ndim);
        }
    }

    TensorLayout target;
    target.ndim = target_indices_ndim;
    for (size_t i = 0; i < target_indices_ndim; ++i) {
        size_t target_idx = target_indices_ndim - i - 1;
        target.shape[target_idx] = 0;
        for (size_t j = 0; j < indices.size(); ++j) {
            if (!is_indices_defined[j] || indices[j].layout.ndim < i + 1) {
                continue;
            }
            size_t cur_idx = indices[j].layout.ndim - i - 1;
            if (indices[j].layout.shape[cur_idx] != 1 &&
                target.shape[target_idx] != 1 &&
                target.shape[target_idx] != indices[j].layout.shape[cur_idx] &&
                target.shape[target_idx] != 0) {
                megdnn_throw(ssprintf(
                        "mismatch index[%zu] in broadcast, got %s", j,
                        indices[j].layout.to_string().c_str()));
            }
            target.shape[target_idx] = std::max(
                    target.shape[target_idx], indices[j].layout.shape[cur_idx]);
        }
    }
    target.init_contiguous_stride();

    for (size_t i = 0; i < indices.size(); ++i) {
        if (is_indices_defined[i]) {
            indices[i].layout = indices[i].layout.broadcast(target);
        }
    }

    auto ret_src = src;
    if (!is_axis_contiguous(is_indices_defined)) {
        ret_src = transpose_src_and_indices(src, indices, is_indices_defined);
    }
    return ret_src;
}

template <typename Opr>
void exec_index(
        Opr& opr, const TensorND& src, const TensorND& dst,
        const IndexingMultiAxisVecBase::IndexDesc& index) {
    HandleImpl* atlas_handle = concrete_handle(opr.handle());

    // prepare all index for all dims and prepare src
    std::vector<TensorND> indices(src.layout.ndim, TensorND());
    std::vector<bool> is_indices_defined(src.layout.ndim, false);
    for (size_t i = 0; i < index.size(); ++i) {
        indices[index[i].axis] = index[i].vec;
        is_indices_defined[index[i].axis] = true;
    }

    // prepare indexed indices and src
    auto transposed_src = broadcast_indexed_indices(src, indices, is_indices_defined);
    auto prepared_src = transposed_src;
    bool transposed_src_contig = transposed_src.layout.is_contiguous();
    if (!transposed_src_contig) {
        auto contig_layout = TensorLayout(
                TensorShape(transposed_src.layout), transposed_src.layout.dtype);
        auto mem_ptr = atlas_handle->alloc(contig_layout.span().dist_byte());
        TensorND contig_tensor(mem_ptr, contig_layout);
        auto relayout_opr = atlas_handle->create_operator<Relayout>();
        relayout_opr->exec(transposed_src, contig_tensor);
        prepared_src = contig_tensor;
    }

    size_t start_src_index = 0;
    size_t end_src_index = 0;
    for (size_t i = 0; i < is_indices_defined.size(); ++i) {
        if (is_indices_defined[i]) {
            start_src_index = i;
            break;
        }
    }
    for (int i = is_indices_defined.size(); i >= 0; --i) {
        if (is_indices_defined[i]) {
            end_src_index = i;
            break;
        }
    }
    end_src_index++;
    size_t start_dst_index = start_src_index;
    size_t end_dst_index = end_src_index + dst.layout.ndim - src.layout.ndim;

    // init non indexed indices and broadcast all indices to dst
    if (start_src_index == 0) {
        indices.resize(index.size());
    } else {
        for (size_t i = 0; i < prepared_src.layout.ndim; ++i) {
            if (!is_indices_defined[i]) {
                TensorLayout layout(
                        TensorShape({prepared_src.layout.shape[i]}), dtype::Int32());
                size_t size = layout.span().dist_byte();
                AclMem acl_one_mem(size, atlas_handle),
                        acl_one_start_mem(size, atlas_handle),
                        acl_zero_start_mem(size, atlas_handle);
                TensorND one_tensor(acl_one_mem.ptr(), layout),
                        one_start_tensor(acl_one_start_mem.ptr(), layout),
                        zero_start_tensor(acl_zero_start_mem.ptr(), layout);

                // fill tensor with 1
                auto fill_opr = atlas_handle->create_operator<Fill>();
                fill_opr->param().value = 1;
                Workspace ws;
                fill_opr->exec(one_tensor, ws);

                // cumsum
                AclTensor acl_one(one_tensor), acl_one_start(one_start_tensor);
                uint64_t cumsum_ws_size = 0;
                aclOpExecutor* cumsum_executor = nullptr;
                aclnn_check(aclnnCumsumV2GetWorkspaceSize(
                        acl_one.get(), static_cast<int64_t>(0), false, false,
                        acl_one_start.get(), &cumsum_ws_size, &cumsum_executor));
                AclMem acl_cumsum_ws(cumsum_ws_size, atlas_handle);
                aclnn_check(aclnnCumsumV2(
                        acl_cumsum_ws.ptr(), cumsum_ws_size, cumsum_executor,
                        atlas_handle->stream()));

                // sub 1
                auto sub_opr = atlas_handle->create_operator<Elemwise>();
                sub_opr->param().mode = Elemwise::Param::Mode::SUB;
                sub_opr->exec({one_start_tensor, one_tensor}, zero_start_tensor);

                TensorLayout contians_all_dims_layout(
                        TensorShape(zero_start_tensor.layout),
                        zero_start_tensor.layout.dtype);
                for (size_t j = 0; j < dst.layout.ndim; ++j) {
                    if ((i < start_src_index && j != i) ||
                        (i > start_src_index &&
                         j != i + (dst.layout.ndim - prepared_src.layout.ndim))) {
                        contians_all_dims_layout.shape[j] = 1;
                        contians_all_dims_layout.stride[j] = 1;
                    } else {
                        contians_all_dims_layout.shape[j] =
                                prepared_src.layout.shape[i];
                        contians_all_dims_layout.stride[j] = 1;
                    }
                }
                contians_all_dims_layout.ndim = dst.layout.ndim;
                zero_start_tensor.layout =
                        contians_all_dims_layout.broadcast(dst.layout);

                auto zero_start_tensor_contig_layout = TensorLayout(
                        TensorShape(zero_start_tensor.layout),
                        zero_start_tensor.layout.dtype);
                void* mem_ptr = atlas_handle->alloc(
                        zero_start_tensor_contig_layout.span().dist_byte());
                indices[i] = TensorND(mem_ptr, zero_start_tensor_contig_layout);
                auto relayout_opr = atlas_handle->create_operator<Relayout>();
                relayout_opr->exec(zero_start_tensor, indices[i], opr.handle());
            } else {
                TensorLayout contians_all_dims_layout = dst.layout;
                contians_all_dims_layout.dtype = dtype::Int32();
                size_t count = 0;
                for (size_t j = 0; j < dst.layout.ndim; ++j) {
                    if (!(j < end_dst_index && j >= start_dst_index)) {
                        contians_all_dims_layout.stride[j] = 0;
                    } else {
                        contians_all_dims_layout.stride[j] =
                                indices[i].layout.stride[count];
                        count++;
                    }
                }
                indices[i].layout = contians_all_dims_layout;
            }
        }
    }

    std::vector<bool> index_non_contig_dim(indices.size(), false);
    for (size_t i = 0; i < indices.size(); ++i) {
        if (is_indices_defined[i] && !indices[i].layout.is_contiguous()) {
            index_non_contig_dim[i] = true;
            auto contig_layout = TensorLayout(
                    TensorShape(indices[i].layout), indices[i].layout.dtype);
            auto indices_mem_ptr =
                    atlas_handle->alloc(contig_layout.span().dist_byte());
            TensorND contig_indices(indices_mem_ptr, contig_layout);
            auto relayout_opr = atlas_handle->create_operator<Relayout>();
            relayout_opr->exec(indices[i], contig_indices, opr.handle());
            indices[i] = contig_indices;
        }
    }

    auto prepared_dst = dst;
    bool dst_contig = dst.layout.is_contiguous();
    if (!dst_contig) {
        auto contig_layout = TensorLayout(TensorShape(dst.layout), dst.layout.dtype);
        auto dst_mem_ptr = atlas_handle->alloc(contig_layout.span().dist_byte());
        prepared_dst = TensorND(dst_mem_ptr, contig_layout);
        auto relayout_opr = atlas_handle->create_operator<Relayout>();
        relayout_opr->exec(dst, prepared_dst);
    }

    // perform index
    AclTensor acl_src(prepared_src), acl_dst(prepared_dst);
    AclTensorList acl_indices_list(indices);

    uint64_t index_ws_size = 0;
    aclOpExecutor* index_executor = nullptr;
    if (std::is_same<Opr, IndexingMultiAxisVecImpl>::value) {
        aclnn_check(aclnnIndexGetWorkspaceSize(
                acl_src.get(), acl_indices_list.get(), acl_dst.get(), &index_ws_size,
                &index_executor));
    } else if (std::is_same<Opr, IndexingSetMultiAxisVecImpl>::value) {
        aclnn_check(aclnnIndexPutImplGetWorkspaceSize(
                acl_src.get(), acl_indices_list.get(), acl_dst.get(), false, true,
                &index_ws_size, &index_executor));
    } else if (std::is_same<Opr, IndexingIncrMultiAxisVecImpl>::value) {
        aclnn_check(aclnnIndexPutImplGetWorkspaceSize(
                acl_src.get(), acl_indices_list.get(), acl_dst.get(), true, true,
                &index_ws_size, &index_executor));
    } else {
        megdnn_throw("unsupported opr");
    }
    AclMem acl_index_ws(index_ws_size, atlas_handle);
    if (std::is_same<Opr, IndexingMultiAxisVecImpl>::value) {
        aclnn_check(aclnnIndex(
                acl_index_ws.ptr(), index_ws_size, index_executor,
                atlas_handle->stream()));
    } else if (
            std::is_same<Opr, IndexingSetMultiAxisVecImpl>::value ||
            std::is_same<Opr, IndexingIncrMultiAxisVecImpl>::value) {
        aclnn_check(aclnnIndexPutImpl(
                acl_index_ws.ptr(), index_ws_size, index_executor,
                atlas_handle->stream()));
    }

    // handle diff between prepared_dst/prepared_src and dst/src.
    if (std::is_same<Opr, IndexingMultiAxisVecImpl>::value && !dst_contig) {
        auto relayout_opr = atlas_handle->create_operator<Relayout>();
        relayout_opr->exec(prepared_dst, dst);
    } else if (
            (std::is_same<Opr, IndexingSetMultiAxisVecImpl>::value ||
             std::is_same<Opr, IndexingIncrMultiAxisVecImpl>::value) &&
            !transposed_src_contig) {
        auto relayout_opr = atlas_handle->create_operator<Relayout>();
        relayout_opr->exec(prepared_src, transposed_src);
    }

    // free allocated mem
    if (!transposed_src_contig) {
        atlas_handle->free(prepared_src.raw_ptr());
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        if (!is_indices_defined[i] || index_non_contig_dim[i]) {
            atlas_handle->free(indices[i].raw_ptr());
        }
    }
    if (!dst_contig) {
        atlas_handle->free(prepared_dst.raw_ptr());
    }
}
}  // namespace

void IndexingMultiAxisVecImpl::exec(
        _megdnn_tensor_in src, const IndexDesc& index, _megdnn_tensor_out dst,
        _megdnn_workspace) {
    exec_index(*this, src, dst, index);
}

void IndexingSetMultiAxisVecImpl::exec(
        _megdnn_tensor_inout data, _megdnn_tensor_in value, const IndexDesc& index,
        _megdnn_workspace) {
    exec_index(*this, data, value, index);
}

void IndexingIncrMultiAxisVecImpl::exec(
        _megdnn_tensor_inout data, _megdnn_tensor_in value, const IndexDesc& index,
        _megdnn_workspace) {
    exec_index(*this, data, value, index);
}