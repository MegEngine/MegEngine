#include "src/cambricon/relayout/opr_impl.h"
#include "include/megdnn/tensor_iter.h"
#include "src/cambricon/cnnl_wrapper/cnnl_op_descriptor.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/megcore/cambricon_computing_context.hpp"
#include "src/cambricon/utils.h"
#include "src/cambricon/utils.mlu.h"
#include "src/common/megcore/public_api/computing.hpp"
#include "src/common/relayout_helper.h"
#include "src/common/utils.h"
namespace megdnn {
namespace cambricon {

namespace {

megcore::cambricon::CambriconComputingContext* get_cambricon_computing_context(
        HandleImpl* handle) {
    megcoreComputingHandle_t computing_handle = handle->megcore_computing_handle();
    return static_cast<megcore::cambricon::CambriconComputingContext*>(
            computing_handle->content.get());
}

bool check_param_transpose(
        const TensorLayout& src, const TensorLayout& dst,
        const std::vector<size_t>& permute) {
    if (src.dtype != dst.dtype) {
        return false;
    }
    if (src.dtype.enumv() != DTypeEnum::Uint8 && src.dtype.enumv() != DTypeEnum::Int8 &&
        src.dtype.enumv() != DTypeEnum::Uint16 &&
        src.dtype.enumv() != DTypeEnum::Int16 &&
        src.dtype.enumv() != DTypeEnum::Int32 && src.dtype.enumv() != DTypeEnum::Bool &&
        src.dtype.enumv() != DTypeEnum::Float16 &&
        src.dtype.enumv() != DTypeEnum::Float32) {
        return false;
    }
    if (src.ndim > 8 || src.ndim > CNNL_DIM_MAX || dst.ndim > CNNL_DIM_MAX ||
        permute.size() > CNNL_DIM_MAX) {
        return false;
    }

    // TODO: in the process of computing, the copy times of memcpy should be less than
    // 65536.

    return true;
}

bool check_tensor_for_transpose(
        const TensorLayout& src_layout, const TensorLayout& dst_layout,
        std::vector<size_t>& permute) {
    if (!src_layout.eq_shape(dst_layout)) {
        return false;
    }
    if (!dst_layout.is_contiguous()) {
        return false;
    }
    size_t ndim = src_layout.ndim;

    permute.resize(ndim);
    rep(i, ndim) { permute[i] = i; }
    std::sort(permute.begin(), permute.end(), [&src_layout](int i, int j) {
        return src_layout.stride[i] >= src_layout.stride[j];
    });
    TensorLayout ori_src_layout = src_layout.dimshuffle(permute);
    if (!ori_src_layout.is_contiguous()) {
        return false;
    }
    return true;
}

bool try_transpose(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, RelayoutForwardImpl* opr,
        bool cross_dev) {
    if (cross_dev) {
        return false;
    }

    auto&& src_layout = src.layout;
    auto&& dst_layout = dst.layout;

    std::vector<size_t> re_permute, permute;
    if (check_tensor_for_transpose(src_layout, dst_layout, re_permute)) {
        permute.resize(re_permute.size());
        for (size_t i = 0; i < re_permute.size(); ++i) {
            permute[re_permute[i]] = i;
        }
    } else {
        return false;
    }

    if (!check_param_transpose(src.layout, dst.layout, permute)) {
        return false;
    }

    size_t ndim = static_cast<size_t>(src_layout.ndim);
    CnnlTransposeDescriptor cnnl_transpose_dsc;
    cnnl_transpose_dsc.set(ndim, permute.data());

    auto handle = concrete_handle(opr->handle());
    CnnlTensorDescriptor cnnl_src_dsc, cnnl_dst_dsc;
    TensorLayout origin_src_layout = src_layout.dimshuffle(re_permute);
    cnnl_src_dsc.set(origin_src_layout);
    cnnl_dst_dsc.set(dst_layout);
    cnnl_check(cnnlTranspose(
            handle->cnnl_handle(), cnnl_transpose_dsc.desc(), cnnl_src_dsc.desc(),
            src.raw_ptr(), cnnl_dst_dsc.desc(), dst.raw_ptr()));
    return true;
}

bool try_boradcast(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, RelayoutForwardImpl* opr,
        bool cross_dev) {
    if (cross_dev) {
        return false;
    }
    TensorLayout src_layout = src.layout;
    TensorLayout dst_layout = dst.layout;
    if (!src_layout.eq_shape(dst_layout)) {
        return false;
    }
    if (!dst_layout.is_contiguous()) {
        return false;
    }
    if (!src_layout.is_contiguous_allow_brdcst()) {
        return false;
    }

    size_t ndim = src_layout.ndim;
    megdnn::SmallVector<size_t> src_shape(src_layout.ndim);
    rep(i, ndim) { src_shape[i] = src_layout.stride[i] != 0 ? src_layout.shape[i] : 1; }
    TensorLayout ori_src_layout(TensorShape{src_shape}, src_layout.dtype);
    auto brdcsted = ori_src_layout.broadcast(dst_layout);
    if (!brdcsted.eq_layout(src_layout)) {
        return false;
    }

    auto handle = concrete_handle(opr->handle());
    CnnlTensorDescriptor cnnl_src_dsc, cnnl_dst_dsc;
    cnnl_src_dsc.set(ori_src_layout);
    cnnl_dst_dsc.set(dst_layout);
    cnnl_check(cnnlExpand(
            handle->cnnl_handle(), cnnl_src_dsc.desc(), src.raw_ptr(),
            cnnl_dst_dsc.desc(), dst.raw_ptr()));
    return true;
}

bool try_copy_contig(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, RelayoutForwardImpl* opr,
        bool cross_dev, int src_dev_id, int dst_dev_id) {
    megdnn_assert(dst.layout.is_non_overlapping_strong());
    auto src_layout = src.layout.collapse_contiguous();
    auto dst_layout = dst.layout.collapse_contiguous();
    megdnn_assert(
            src_layout.dtype == dst_layout.dtype &&
            src_layout.total_nr_elems() == dst_layout.total_nr_elems());
    if (src_layout.ndim != 1 || dst_layout.ndim != 1)
        return false;
    if (src_layout.stride[0] != 1 || dst_layout.stride[0] != 1)
        return false;

    auto handle = concrete_handle(opr->handle());
    auto computing_context = get_cambricon_computing_context(handle);

    size_t copy_size = dst_layout.span().dist_byte();
    if (!cross_dev) {
        cnrt_check(cnrtMemcpyAsync(
                dst.raw_ptr(), src.raw_ptr(), copy_size, cnrt_queue(handle),
                CNRT_MEM_TRANS_DIR_DEV2DEV));
    } else {
        computing_context->memcpy_peer_async_d2d(
                dst.raw_ptr(), dst_dev_id, src.raw_ptr(), src_dev_id, copy_size);
    }
    return true;
}

bool check_param_copy(const TensorLayout& src, const TensorLayout& dst) {
    if (src.dtype != dst.dtype) {
        return false;
    }
    if (src.dtype.size() != 1 && src.dtype.size() != 2 && src.dtype.size() != 4 &&
        src.dtype.size() != 8) {
        return false;
    }
    // TODO: scale limitations are more than above on MLU300 series and CE3226.
    return true;
}

bool try_copy_non_contig(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, RelayoutForwardImpl* opr,
        bool cross_dev) {
    if (cross_dev) {
        return false;
    }
    TensorLayout src_layout = src.layout;
    TensorLayout dst_layout = dst.layout;
    if (!src_layout.eq_shape(dst_layout)) {
        return false;
    }
    if (!check_param_copy(src.layout, dst.layout)) {
        return false;
    }

    TensorLayout collapse_src_layout = src_layout.collapse_contiguous();
    TensorLayout collapse_dst_layout = dst_layout.collapse_contiguous();
    if (collapse_src_layout.is_contiguous()) {
        collapse_src_layout = TensorLayout(
                TensorShape(collapse_dst_layout), collapse_src_layout.dtype);
    } else if (collapse_dst_layout.is_contiguous()) {
        collapse_dst_layout = TensorLayout(
                TensorShape(collapse_src_layout), collapse_dst_layout.dtype);
    } else {
        collapse_src_layout = src_layout;
        collapse_dst_layout = dst_layout;
    }

    auto handle = concrete_handle(opr->handle());
    CnnlTensorDescriptor cnnl_src_dsc, cnnl_dst_dsc;
    cnnl_src_dsc.set(collapse_src_layout);
    cnnl_dst_dsc.set(collapse_dst_layout);

    cnnl_check(cnnlCopy(
            handle->cnnl_handle(), cnnl_src_dsc.desc(), src.raw_ptr(),
            cnnl_dst_dsc.desc(), dst.raw_ptr()));
    return true;
}

// TODO: implement with kernel.
template <typename ctype>
void copy_general_interior(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, RelayoutForwardImpl* opr,
        bool cross_dev, int src_dev_id, int dst_dev_id) {
    auto handle = concrete_handle(opr->handle());
    auto computing_context = get_cambricon_computing_context(handle);

    auto idst = tensor_iter_valonly<ctype>(dst).begin();
    auto isrc = tensor_iter_valonly<ctype>(src).begin();

    if (!cross_dev) {
        for (size_t i = 0, it = dst.layout.total_nr_elems(); i < it; ++i) {
            dt_byte* dst_addr = reinterpret_cast<dt_byte*>(dst.raw_ptr()) +
                                idst.offset() * dst.layout.dtype.size();
            dt_byte* src_addr = reinterpret_cast<dt_byte*>(src.raw_ptr()) +
                                isrc.offset() * src.layout.dtype.size();
            cnrt_check(cnrtMemcpyAsync(
                    static_cast<void*>(dst_addr), static_cast<void*>(src_addr),
                    dst.layout.dtype.size(), cnrt_queue(handle),
                    CNRT_MEM_TRANS_DIR_DEV2DEV));
            ++idst;
            ++isrc;
        }
    } else {
        for (size_t i = 0, it = dst.layout.total_nr_elems(); i < it; ++i) {
            dt_byte* dst_addr = reinterpret_cast<dt_byte*>(dst.raw_ptr()) +
                                idst.offset() * dst.layout.dtype.size();
            dt_byte* src_addr = reinterpret_cast<dt_byte*>(src.raw_ptr()) +
                                isrc.offset() * src.layout.dtype.size();
            computing_context->memcpy_peer_async_d2d(
                    static_cast<void*>(dst_addr), dst_dev_id,
                    static_cast<void*>(src_addr), src_dev_id, dst.layout.dtype.size());
            ++idst;
            ++isrc;
        }
    }
}

void copy_general(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, RelayoutForwardImpl* opr,
        bool cross_dev, int src_dev_id, int dst_dev_id) {
    // TODO: support different types of src and dst.
    if (src.layout.dtype != dst.layout.dtype) {
        megdnn_throw(
                "src type is not equal to dst type in relayout forward on cambricon.");
    }

    if (dst.layout.dtype.enumv() == DTypeEnum::Float32) {
        copy_general_interior<dt_float32>(
                src, dst, opr, cross_dev, src_dev_id, dst_dev_id);
    } else if (dst.layout.dtype.enumv() == DTypeEnum::Float16) {
        copy_general_interior<dt_float16>(
                src, dst, opr, cross_dev, src_dev_id, dst_dev_id);
    } else if (dst.layout.dtype.enumv() == DTypeEnum::Int32) {
        copy_general_interior<dt_int32>(
                src, dst, opr, cross_dev, src_dev_id, dst_dev_id);
    } else if (dst.layout.dtype.enumv() == DTypeEnum::Int16) {
        copy_general_interior<dt_int16>(
                src, dst, opr, cross_dev, src_dev_id, dst_dev_id);
    } else if (dst.layout.dtype.enumv() == DTypeEnum::Uint16) {
        copy_general_interior<dt_uint16>(
                src, dst, opr, cross_dev, src_dev_id, dst_dev_id);
    } else if (dst.layout.dtype.enumv() == DTypeEnum::Int8) {
        copy_general_interior<dt_int8>(
                src, dst, opr, cross_dev, src_dev_id, dst_dev_id);
    } else if (dst.layout.dtype.enumv() == DTypeEnum::Uint8) {
        copy_general_interior<dt_uint8>(
                src, dst, opr, cross_dev, src_dev_id, dst_dev_id);
    } else if (dst.layout.dtype.enumv() == DTypeEnum::Bool) {
        copy_general_interior<dt_bool>(
                src, dst, opr, cross_dev, src_dev_id, dst_dev_id);
    } else {
        megdnn_throw("unsupported type in reduce forward on cambricon.");
    }
}

}  // namespace

void RelayoutForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, Handle* src_handle) {
    bool cross_dev = false;
    int dst_dev_id = -1, src_dev_id = -1;

    // check whether cross device copy
    if (src_handle && src_handle != handle()) {
        megcoreDeviceHandle_t dev;
        megcoreGetDeviceHandle(src_handle->megcore_computing_handle(), &dev);
        megcorePlatform_t plat;
        megcoreGetPlatform(dev, &plat);
        megdnn_throw_if(
                plat != megcorePlatformCambricon, megdnn_error,
                "only relayout between cambricon devices are supported");
        megcoreGetDeviceID(dev, &src_dev_id);
        megcoreGetDeviceHandle(this->handle()->megcore_computing_handle(), &dev);
        megcoreGetDeviceID(dev, &dst_dev_id);

        megdnn_assert(src_dev_id >= 0 && dst_dev_id >= 0);
        cross_dev = src_dev_id != dst_dev_id;
    }

    if (!try_transpose(src, dst, this, cross_dev) &&
        !try_boradcast(src, dst, this, cross_dev) &&
        !try_copy_contig(src, dst, this, cross_dev, src_dev_id, dst_dev_id) &&
        !try_copy_non_contig(src, dst, this, cross_dev)) {
        copy_general(src, dst, this, cross_dev, src_dev_id, dst_dev_id);
    }
}

}  // namespace cambricon
}  // namespace megdnn
// vim: syntax=cpp.doxygen
