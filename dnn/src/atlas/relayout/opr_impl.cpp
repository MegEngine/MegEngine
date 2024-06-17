#include "opr_impl.h"
#include <acl/acl_rt.h>
#include "aclnnop/aclnn_copy.h"
#include "aclnnop/aclnn_permute.h"
#include "include/megdnn/tensor_iter.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"
namespace megdnn {
namespace atlas {

namespace {

bool check_tensor_for_transpose_with_one_contig(
        const TensorLayout& contig_layout, const TensorLayout& non_contig_layout,
        std::vector<size_t>& permute) {
    if (!contig_layout.eq_shape(non_contig_layout)) {
        return false;
    }
    if (!contig_layout.is_contiguous()) {
        return false;
    }
    size_t ndim = non_contig_layout.ndim;

    permute.resize(ndim);
    rep(i, ndim) { permute[i] = i; }
    std::sort(permute.begin(), permute.end(), [&non_contig_layout](int i, int j) {
        return non_contig_layout.stride[i] >= non_contig_layout.stride[j];
    });
    TensorLayout ori_non_contig_layout = non_contig_layout.dimshuffle(permute);
    if (!ori_non_contig_layout.is_contiguous()) {
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
    // TODO: use SmallVector may be better.
    std::vector<size_t> re_permute;
    if (check_tensor_for_transpose_with_one_contig(
                dst_layout, src_layout, re_permute)) {
        std::vector<size_t> permute(re_permute.size());
        for (size_t i = 0; i < re_permute.size(); ++i) {
            permute[re_permute[i]] = i;
        }
        TensorND origin_src(src.layout.dimshuffle(re_permute), src.get_ref_ptr());
        AclTensor acl_origin_src(origin_src), acl_dst(dst);
        AclIntArray acl_dims(permute.data(), permute.size());
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;

        aclnn_check(aclnnPermuteGetWorkspaceSize(
                acl_origin_src.get(), acl_dims.get(), acl_dst.get(), &ws_size,
                &executor));
        auto handle = concrete_handle(opr->handle());
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnPermute(ws.ptr(), ws_size, executor, handle->stream()));
        return true;
    } else if (check_tensor_for_transpose_with_one_contig(
                       src_layout, dst_layout, re_permute)) {
        TensorND origin_dst(dst.layout.dimshuffle(re_permute), dst.get_ref_ptr());
        AclTensor acl_src(src), acl_origin_dst(origin_dst);
        AclIntArray acl_dims(re_permute.data(), re_permute.size());
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;

        aclnn_check(aclnnPermuteGetWorkspaceSize(
                acl_src.get(), acl_dims.get(), acl_origin_dst.get(), &ws_size,
                &executor));
        auto handle = concrete_handle(opr->handle());
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnPermute(ws.ptr(), ws_size, executor, handle->stream()));
        return true;
    } else {
        return false;
    }
}

bool try_copy_contig(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, RelayoutForwardImpl* opr,
        bool cross_dev, int src_dev_id, int dst_dev_id) {
    auto src_layout = src.layout.collapse_contiguous();
    auto dst_layout = dst.layout.collapse_contiguous();
    megdnn_assert(src_layout.total_nr_elems() == dst_layout.total_nr_elems());
    if (src_layout.ndim != 1 || dst_layout.ndim != 1)
        return false;
    if (src_layout.stride[0] != 1 || dst_layout.stride[0] != 1)
        return false;

    auto handle = concrete_handle(opr->handle());
    size_t copy_size = dst_layout.span().dist_byte();
    if (reinterpret_cast<uintptr_t>(dst.raw_ptr()) % 64 != 0 ||
        reinterpret_cast<uintptr_t>(src.raw_ptr()) % 64 != 0) {
        if (!cross_dev) {
            // FIXME: when copy between two devices, is sync of two devices needed?
            acl_check(aclrtSynchronizeStream(handle->stream()));
            acl_check(aclrtMemcpy(
                    dst.raw_ptr(), copy_size, src.raw_ptr(), copy_size,
                    ACL_MEMCPY_DEVICE_TO_DEVICE));
        } else {
            int32_t canAccessPeer = 0;
            acl_check(aclrtDeviceCanAccessPeer(&canAccessPeer, src_dev_id, dst_dev_id));
            if (canAccessPeer == 1) {
                acl_check(aclrtDeviceEnablePeerAccess(dst_dev_id, 0));
                acl_check(aclrtSynchronizeStream(handle->stream()));
                acl_check(aclrtMemcpy(
                        dst.raw_ptr(), copy_size, src.raw_ptr(), copy_size,
                        ACL_MEMCPY_DEVICE_TO_DEVICE));
            } else {
                megdnn_throw("there is no enough ascend devices for data relayout");
            }
        }
    } else {
        if (!cross_dev) {
            acl_check(aclrtMemcpyAsync(
                    dst.raw_ptr(), copy_size, src.raw_ptr(), copy_size,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, handle->stream()));
        } else {
            int32_t canAccessPeer = 0;
            acl_check(aclrtDeviceCanAccessPeer(&canAccessPeer, src_dev_id, dst_dev_id));
            if (canAccessPeer == 1) {
                acl_check(aclrtDeviceEnablePeerAccess(dst_dev_id, 0));
                acl_check(aclrtMemcpyAsync(
                        dst.raw_ptr(), copy_size, src.raw_ptr(), copy_size,
                        ACL_MEMCPY_DEVICE_TO_DEVICE, handle->stream()));
            } else {
                megdnn_throw("there is no enough ascend devices for data relayout");
            }
        }
    }

    return true;
}

bool check_shape_equal_with_broadcast(
        const TensorLayout& src, const TensorLayout& dst) {
    if (src.eq_shape(dst)) {
        return true;
    }
    for (size_t i = 0; i < std::min(src.ndim, dst.ndim); ++i) {
        if (src.shape[src.ndim - i - 1] != dst.shape[dst.ndim - i - 1] &&
            src.shape[src.ndim - i - 1] != 1) {
            return false;
        }
    }
    return true;
}

bool try_copy_non_contig(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, RelayoutForwardImpl* opr,
        bool cross_dev) {
    if (cross_dev) {
        return false;
    }

    // TODO: for src/dst has negative dim, copy may introduce strange error in some
    // cases, so skip it temporarily.
    for (size_t i = 0; i < src.layout.ndim; ++i) {
        if (src.layout.stride[i] < 0) {
            return false;
        }
    }
    for (size_t i = 0; i < dst.layout.ndim; ++i) {
        if (dst.layout.stride[i] < 0) {
            return false;
        }
    }

    TensorLayout src_layout = src.layout;
    rep(i, src_layout.ndim) {
        src_layout.shape[i] = src_layout.stride[i] != 0 ? src_layout.shape[i] : 1;
    }
    TensorLayout dst_layout = dst.layout;
    rep(i, dst_layout.ndim) {
        dst_layout.shape[i] = dst_layout.stride[i] != 0 ? dst_layout.shape[i] : 1;
    }
    if (!check_shape_equal_with_broadcast(src_layout, dst_layout)) {
        return false;
    }

    TensorND src_can_brd(src.raw_ptr(), src_layout),
            dst_can_brd(dst.raw_ptr(), dst_layout);
    AclTensor acl_src(src_can_brd), acl_dst(dst_can_brd);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    aclnn_check(aclnnInplaceCopyGetWorkspaceSize(
            acl_dst.get(), acl_src.get(), &ws_size, &executor));
    auto handle = concrete_handle(opr->handle());
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnInplaceCopy(ws.ptr(), ws_size, executor, handle->stream()));
    return true;
}

template <typename ctype>
void copy_general_interior(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, RelayoutForwardImpl* opr,
        bool cross_dev, int src_dev_id, int dst_dev_id) {
    auto handle = concrete_handle(opr->handle());

    auto idst = tensor_iter_valonly<ctype>(dst).begin();
    auto isrc = tensor_iter_valonly<ctype>(src).begin();

    if (!cross_dev) {
        for (size_t i = 0, it = dst.layout.total_nr_elems(); i < it; ++i) {
            dt_byte* dst_addr = reinterpret_cast<dt_byte*>(dst.raw_ptr()) +
                                idst.offset() * dst.layout.dtype.size();
            dt_byte* src_addr = reinterpret_cast<dt_byte*>(src.raw_ptr()) +
                                isrc.offset() * src.layout.dtype.size();
            acl_safe_memcpy_async(
                    static_cast<void*>(dst_addr), dst.layout.dtype.size(),
                    static_cast<void*>(src_addr), dst.layout.dtype.size(),
                    ACL_MEMCPY_DEVICE_TO_DEVICE, handle->stream());
            ++idst;
            ++isrc;
        }
    } else {
        int32_t canAccessPeer = 0;
        acl_check(aclrtDeviceCanAccessPeer(&canAccessPeer, src_dev_id, dst_dev_id));
        if (canAccessPeer != 1) {
            megdnn_throw("there is no enough ascend devices for data relayout");
        }
        acl_check(aclrtDeviceEnablePeerAccess(dst_dev_id, 0));
        for (size_t i = 0, it = dst.layout.total_nr_elems(); i < it; ++i) {
            dt_byte* dst_addr = reinterpret_cast<dt_byte*>(dst.raw_ptr()) +
                                idst.offset() * dst.layout.dtype.size();
            dt_byte* src_addr = reinterpret_cast<dt_byte*>(src.raw_ptr()) +
                                isrc.offset() * src.layout.dtype.size();
            acl_safe_memcpy_async(
                    static_cast<void*>(dst_addr), dst.layout.dtype.size(),
                    static_cast<void*>(src_addr), dst.layout.dtype.size(),
                    ACL_MEMCPY_DEVICE_TO_DEVICE, handle->stream());
            ++idst;
            ++isrc;
        }
    }
}

void copy_general(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, RelayoutForwardImpl* opr,
        bool cross_dev, int src_dev_id, int dst_dev_id) {
    auto src_layout = src.layout.collapse_contiguous();
    auto dst_layout = dst.layout.collapse_contiguous();
    megdnn_assert(src_layout.total_nr_elems() == dst_layout.total_nr_elems());

    switch (dst.layout.dtype.enumv()) {
#define cb(name, ctype)                                            \
    case (DTypeEnum::name): {                                      \
        copy_general_interior<ctype>(                              \
                src, dst, opr, cross_dev, src_dev_id, dst_dev_id); \
        break;                                                     \
    }
        cb(Float32, dt_float32);
        DNN_INC_FLOAT16(cb(Float16, dt_float16));
        cb(Int32, dt_int32);
        cb(Int16, dt_int16);
        cb(Uint16, dt_uint16);
        cb(Int8, dt_int8);
        cb(Uint8, dt_uint8);
        cb(Bool, dt_bool);
        default:
            megdnn_throw("unsupported type in relayout forward on atlas.");
    }
}
}  // namespace

void RelayoutForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, Handle* src_handle) {
    megdnn_assert(dst.layout.is_non_overlapping_strong());
    megdnn_assert(
            src.layout.dtype == dst.layout.dtype, "check %s == %s",
            src.layout.dtype.name(), dst.layout.dtype.name());
    bool cross_dev = false;
    int dst_dev_id = -1, src_dev_id = -1;

    // check whether cross device copy
    if (src_handle && src_handle != handle()) {
        megcoreDeviceHandle_t dev;
        megcoreGetDeviceHandle(src_handle->megcore_computing_handle(), &dev);
        megcorePlatform_t plat;
        megcoreGetPlatform(dev, &plat);
        megdnn_throw_if(
                plat != megcorePlatformAtlas, megdnn_error,
                "only relayout between atlas devices are supported");
        megcoreGetDeviceID(dev, &src_dev_id);
        megcoreGetDeviceHandle(this->handle()->megcore_computing_handle(), &dev);
        megcoreGetDeviceID(dev, &dst_dev_id);

        megdnn_assert(src_dev_id >= 0 && dst_dev_id >= 0);
        cross_dev = src_dev_id != dst_dev_id;
    }

    if (!try_transpose(src, dst, this, cross_dev) &&
        !try_copy_contig(src, dst, this, cross_dev, src_dev_id, dst_dev_id) &&
        !try_copy_non_contig(src, dst, this, cross_dev)) {
        copy_general(src, dst, this, cross_dev, src_dev_id, dst_dev_id);
    }
}

}  // namespace atlas
}  // namespace megdnn
// vim: syntax=cpp.doxygen
