#pragma once

#include "megbrain/comp_node.h"
#include "megbrain/rdnn/management.h"
#include "megbrain/system.h"
#include "megbrain/tensor.h"
#include "megbrain/utils/hash_ct.h"
#include "megbrain/utils/timer.h"

#include "megdnn/basic_types.h"
#include "megdnn/oprs.h"

namespace mgb {
namespace rdnn {

// clang-format off
#define DNN_FOREACH_FASTRUN_OPR(cb)  \
    cb(ConvolutionForward)           \
    cb(ConvBiasForward)              \
    cb(ConvolutionBackwardData)      \
    cb(ConvolutionBackwardFilter)    \
    cb(Convolution3DForward)         \
    cb(Convolution3DBackwardData)    \
    cb(Convolution3DBackwardFilter)  \
    cb(LocalShareForward)            \
    cb(LocalShareBackwardData)       \
    cb(LocalShareBackwardFilter)     \
    cb(DeformableConvForward)        \
    cb(DeformableConvBackwardFilter) \
    cb(DeformableConvBackwardData)   \
    cb(BatchConvBiasForward)         \
    cb(MatrixMul)                    \
    cb(BatchedMatrixMul)             \
    cb(PoolingForward)               \
    cb(PoolingBackward)
// clang-format on

template <typename Opr>
constexpr bool opr_supports_preprocess() {
    return std::is_same<Opr, megdnn::ConvolutionForward>::value ||
           std::is_same<Opr, megdnn::ConvBias>::value;
}

template <typename Opr>
constexpr bool opr_contain_bias() {
    return std::is_same<Opr, megdnn::ConvBias>::value;
}

//! matmul and batchedMatrixMul
template <typename Opr>
constexpr bool is_matmul() {
    return std::is_same<Opr, megdnn::MatrixMul>::value ||
           std::is_same<Opr, megdnn::BatchedMatrixMul>::value;
}

template <typename Opr, bool has_prep>
struct PreprocessFilterImpl {
    using T = union {};
};

template <typename Opr>
struct PreprocessFilterImpl<Opr, true> {
    using T = typename Opr::PreprocessedFilter;
};

template <typename Opr>
using PreprocessFilter =
        typename PreprocessFilterImpl<Opr, opr_supports_preprocess<Opr>()>::T;

template <typename Opr>
struct AlgoChooserFuncId {};

#define DEF_FUNC_ID(func)                                                           \
    template <>                                                                     \
    struct AlgoChooserFuncId<megdnn::func> {                                        \
        __attribute__((unused)) static constexpr sys::TimedFuncInvoker::FuncId ID = \
                static_cast<sys::TimedFuncInvoker::FuncId>(                         \
                        MGB_HASH_STR("megdnn::" #func));                            \
    };

DNN_FOREACH_FASTRUN_OPR(DEF_FUNC_ID)

#undef DEF_FUNC_ID

/* =================== TimedProfiler =================== */

/*!
 * \brief profile a megdnn opr conv with given param
 *
 * This class only provides static methods, and the entry point is
 * TimedProfiler::profile; it would run profiler in a timed environment by
 * sys::TimedFuncInvoker
 *
 * \tparam Opr megdnn opr impl
 */
template <typename Opr>
class TimedProfiler {
    static constexpr int arity_in = OprArityTrait<Opr>::arity_in;
    static constexpr int arity_out = OprArityTrait<Opr>::arity_out;
    static constexpr int arity = OprArityTrait<Opr>::arity;

    using TensorShapeArray = std::array<megdnn::TensorShape, arity>;

public:
    struct Param {
        struct ExecutionPolicyBlob {
            //! enlarge the max size if needed
            constexpr static size_t MAX_SIZE_IN_BYTES = 10240;
            char data[MAX_SIZE_IN_BYTES];
            uint32_t size;

            static ExecutionPolicyBlob serialize(const megdnn::ExecutionPolicy& policy);
            megdnn::ExecutionPolicy deserialize() const;
        };
        ExecutionPolicyBlob execution_policy;
        size_t workspace;
        megdnn::DTypeEnum dtypes[arity];
        CompNode::Locator comp_node_physical, comp_node_logical;
        TensorShapeArray shapes;
        typename Opr::Param opr_param;
        bool allow_weight_preprocess;

        //! filled by profile()
        mutable double actual_timeout;
    };

    struct Result {
        double time;
    };

    static Maybe<Result> profile(const Param& param, double& timeout);

private:
    using TParam = sys::TimedFuncInvoker::Param;
    using TResult = sys::TimedFuncInvoker::Result;

    static const double timeout_setting;

    static double init_timeout_setting();
    static void preprocess(
            const megdnn::TensorLayoutArray& preprocessed_layout,
            const SmallVector<DeviceTensorND>& flt_val, UniqPtrWithCN<Opr>& megdnn_opr,
            megdnn::Workspace& mdn_workspace, std::array<TensorLayout, arity>& layouts,
            std::array<DeviceTensorND, arity_in>& inp_val,
            PreprocessFilter<Opr>& prep_flt);
    static TResult prof_impl(const TParam& raw_param);
    static void prof_init_device(const TParam& raw_param);
};
}  // namespace rdnn
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
