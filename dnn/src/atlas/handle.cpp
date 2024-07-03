#include "src/atlas/handle.h"
#include "megcore_atlas.h"
#include "src/atlas/adaptive_pooling/opr_impl.h"
#include "src/atlas/argmxx/opr_impl.h"
#include "src/atlas/argsort/opr_impl.h"
#include "src/atlas/batch_normalize/opr_impl.h"
#include "src/atlas/checksum/opr_impl.h"
#include "src/atlas/cond_take/opr_impl.h"
#include "src/atlas/conv_bias/opr_impl.h"
#include "src/atlas/convolution/opr_impl.h"
#include "src/atlas/elemwise/opr_impl.h"
#include "src/atlas/elemwise_multi_type/opr_impl.h"
#include "src/atlas/fill/opr_impl.h"
#include "src/atlas/group_norm/opr_impl.h"
#include "src/atlas/indexing_multi_axis_vec/opr_impl.h"
#include "src/atlas/indexing_one_hot/opr_impl.h"
#include "src/atlas/linspace/opr_impl.h"
#include "src/atlas/mask_fill/opr_impl.h"
#include "src/atlas/matrix_mul/opr_impl.h"
#include "src/atlas/param_pack/opr_impl.h"
#include "src/atlas/pooling/opr_impl.h"
#include "src/atlas/powc/opr_impl.h"
#include "src/atlas/reduce/opr_impl.h"
#include "src/atlas/relayout/opr_impl.h"
#include "src/atlas/resize/opr_impl.h"
#include "src/atlas/rng/opr_impl.h"
#include "src/atlas/softmax/opr_impl.h"
#include "src/atlas/topk/opr_impl.h"
#include "src/atlas/type_cvt/opr_impl.h"
#include "src/atlas/utils.h"
#include "src/atlas/where/opr_impl.h"
#include "src/common/handle_impl.h"

#include <acl/acl.h>

namespace megdnn {
namespace atlas {

HandleImpl::HandleImpl(megcoreComputingHandle_t comp_handle)
        : HandleImplHelper(comp_handle, HandleType::ATLAS) {
    // Get megcore device handle
    megcoreDeviceHandle_t dev_handle;
    megcoreGetDeviceHandle(comp_handle, &dev_handle);

    int dev_id;
    megcoreGetDeviceID(dev_handle, &dev_id);
    m_device_id = dev_id;
    megcore::getAtlasContext(comp_handle, &m_megcore_context);
}

HandleImpl::~HandleImpl() noexcept = default;

void* HandleImpl::alloc(size_t size, aclrtMemMallocPolicy policy) {
    auto mem_mgr = megcore_context().mem_mgr;
    if (size <= 0) {
        return nullptr;
    }
    if (mem_mgr) {
        return mem_mgr->alloc(size);
    } else {
        int32_t dev_id = -1;
        auto err = aclrtGetDevice(&dev_id);
        if (err == ACL_ERROR_INVALID_DEVICE || err == ACL_ERROR_RT_CONTEXT_NULL ||
            device_id() != dev_id) {
            acl_check(aclrtSetDevice(device_id()));
        }

        void* ptr = nullptr;
        acl_check(aclrtMalloc(&ptr, size, policy));
        return ptr;
    }
}

void HandleImpl::free(void* ptr) {
    auto mem_mgr = megcore_context().mem_mgr;
    if (!ptr) {
        return;
    }
    if (mem_mgr) {
        return mem_mgr->free(ptr);
    } else {
        int32_t dev_id = -1;
        auto err = aclrtGetDevice(&dev_id);
        if (err == ACL_ERROR_INVALID_DEVICE || err == ACL_ERROR_RT_CONTEXT_NULL ||
            device_id() != dev_id) {
            acl_check(aclrtSetDevice(device_id()));
        }
        acl_check(aclrtFree(ptr));
    }
}

template <typename Opr>
std::unique_ptr<Opr> HandleImpl::create_operator() {
    megdnn_throw(
            "unsupported atlas opr, try export RUNTIME_OVERRIDE_LOG_LEVEL=0 to get "
            "more info");
    return nullptr;
}

size_t HandleImpl::alignment_requirement() const {
    //! because memcpyasync api requires that the memory is 128bytes alignment
    return 64;
}

MEGDNN_SPECIALIZE_CREATE_OPERATOR(ChecksumForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ElemwiseForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Fill);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TypeCvt);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvBiasForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(RelayoutForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvolutionForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvolutionBackwardData);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvolutionBackwardFilter);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PoolingForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PoolingBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(AdaptivePoolingForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(AdaptivePoolingBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(MatrixMulForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ReduceForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ParamPackConcat);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingOneHotForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingSetOneHotForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PowC);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ResizeForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GaussianRNG);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(UniformRNG);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BNForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BNBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GroupNormForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GroupNormBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ElemwiseMultiType);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Linspace);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(MaskedFill);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(WhereForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(WhereBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(CondTake);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TopK);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingMultiAxisVec);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingSetMultiAxisVec);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingIncrMultiAxisVec);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ArgmaxForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ResizeBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SoftmaxForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ArgsortForward);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winstantiation-after-specialization"
MEGDNN_FOREACH_OPR_CLASS(MEGDNN_INST_CREATE_OPERATOR)
#pragma GCC diagnostic pop

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
