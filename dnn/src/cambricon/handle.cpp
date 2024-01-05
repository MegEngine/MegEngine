#include "src/common/handle_impl.h"
#include "src/common/version_symbol.h"

#include "src/cambricon/handle.h"
#include "src/cambricon/utils.h"

#include <cnnl.h>
#include <cnrt.h>
#include <iostream>
#include "src/cambricon/adaptive_pooling/opr_impl.h"
#include "src/cambricon/argmxx/opr_impl.h"
#include "src/cambricon/argsort/opr_impl.h"
#include "src/cambricon/batch_normalization/opr_impl.h"
#include "src/cambricon/checksum/opr_impl.h"
#include "src/cambricon/cond_take/opr_impl.h"
#include "src/cambricon/conv_bias/opr_impl.h"
#include "src/cambricon/convolution/opr_impl.h"
#include "src/cambricon/elemwise/opr_impl.h"
#include "src/cambricon/elemwise_multi_type/opr_impl.h"
#include "src/cambricon/fill/opr_impl.h"
#include "src/cambricon/group_norm/opr_impl.h"
#include "src/cambricon/indexing_multi_axis_vec/opr_impl.h"
#include "src/cambricon/indexing_one_hot/opr_impl.h"
#include "src/cambricon/masked_fill/opr_impl.h"
#include "src/cambricon/matrix_mul/opr_impl.h"
#include "src/cambricon/param_pack/opr_impl.h"
#include "src/cambricon/pooling/opr_impl.h"
#include "src/cambricon/powc/opr_impl.h"
#include "src/cambricon/reduce/opr_impl.h"
#include "src/cambricon/relayout/opr_impl.h"
#include "src/cambricon/resize/opr_impl.h"
#include "src/cambricon/softmax/opr_impl.h"
#include "src/cambricon/topk/opr_impl.h"
#include "src/cambricon/type_cvt/opr_impl.h"

#include "src/cambricon/linspace/opr_impl.h"
#include "src/cambricon/rng/opr_impl.h"
namespace megdnn {
namespace cambricon {

HandleImpl::HandleImpl(megcoreComputingHandle_t comp_handle)
        : HandleImplHelper(comp_handle, HandleType::CAMBRICON) {
    // Get megcore device handle
    megcoreDeviceHandle_t dev_handle;
    megcoreGetDeviceHandle(comp_handle, &dev_handle);
    int dev_id;
    megcoreGetDeviceID(dev_handle, &dev_id);
    unsigned int dev_num;
    cnrt_check(cnrtGetDeviceCount(&dev_num));
    MEGDNN_MARK_USED_VAR(dev_num);
    // check validity of device_id
    megdnn_assert(dev_id >= 0 && static_cast<unsigned int>(dev_id) < dev_num);
    m_device_id = dev_id;
    cnrt_check(cnrtGetDeviceProperties(&m_device_info, dev_id));
    megcore::getCambriconContext(comp_handle, &m_megcore_context);
}

HandleImpl::~HandleImpl() noexcept = default;

void* HandleImpl::alloc(size_t size) {
    auto mem_mgr = megcore_context().mem_mgr;
    if (size <= 0) {
        return nullptr;
    }
    if (mem_mgr) {
        return mem_mgr->alloc(size);
    } else {
        void* ptr = nullptr;
        cnrt_check(cnrtSetDevice(device_id()));
        cnrt_check(cnrtMalloc(&ptr, size));
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
        cnrt_check(cnrtSetDevice(device_id()));
        cnrt_check(cnrtQueueSync(queue()));
        cnrt_check(cnrtFree(ptr));
    }
}

template <typename Opr>
std::unique_ptr<Opr> HandleImpl::create_operator() {
    std::cout << typeid(Opr).name() << std::endl;
    megdnn_throw(
            "unsupported cambricon opr, try export RUNTIME_OVERRIDE_LOG_LEVEL=0 to "
            "get "
            "more info");
    return nullptr;
}

size_t HandleImpl::alignment_requirement() const {
#if CNRT_MAJOR_VERSION >= 5
    return 256;
#else
    return 1;
#endif
}

MEGDNN_SPECIALIZE_CREATE_OPERATOR(ChecksumForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvBiasForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingOneHotForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingSetOneHotForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ElemwiseForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Fill);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BNForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(BNBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(MatrixMulForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TypeCvt);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PowC);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvolutionForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvolutionBackwardData);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ConvolutionBackwardFilter);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GroupNormForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GroupNormBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Linspace);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(CondTake);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(GaussianRNG);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(UniformRNG);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PoolingForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(PoolingBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(AdaptivePoolingForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(AdaptivePoolingBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ReduceForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(RelayoutForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(Resize);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ResizeBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(TopK);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ArgmaxForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ElemwiseMultiType)
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingMultiAxisVec);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingSetMultiAxisVec);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(IndexingIncrMultiAxisVec);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(MaskedFill);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ArgsortForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ArgsortBackward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(ParamPackConcat);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SoftmaxForward);
MEGDNN_SPECIALIZE_CREATE_OPERATOR(SoftmaxBackward);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winstantiation-after-specialization"
MEGDNN_FOREACH_OPR_CLASS(MEGDNN_INST_CREATE_OPERATOR)
#pragma GCC diagnostic pop

}  // namespace cambricon
}  // namespace megdnn

MEGDNN_VERSION_SYMBOL3(
        CNRT, CNRT_MAJOR_VERSION, CNRT_MINOR_VERSION, CNRT_PATCH_VERSION);

// vim: syntax=cpp.doxygen