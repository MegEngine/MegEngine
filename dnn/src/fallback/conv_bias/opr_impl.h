/**
 * \file dnn/src/fallback/conv_bias/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "include/megdnn/thin/function.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"
#include "src/fallback/convolution/opr_impl.h"
#include "src/fallback/matrix_mul/opr_impl.h"
#include "src/naive/conv_bias/opr_impl.h"

namespace megdnn {
namespace fallback {

/*!
 * \brief fallback conv bias forward impl
 *
 * Note: this operator class serves for multiple purposes:
 *
 *  1. canonizing conv reprs into NCBKernParam and NCBKernSizeParam, and
 *     subclasses should impl by overriding *_ncb methods
 *  2. providing a default impl for group conv by calling ncb_1g* methods
 *  3. providing a conv impl faster than naive under some cases
 *  4. providing a default impl for choosing heuristic algorithm, by using the
 *     first algo that fits the workspace limit
 */
class ConvBiasImpl : public naive::ConvBiasForwardImpl {
public:
    using naive::ConvBiasForwardImpl::ConvBiasForwardImpl;
    using AlgoSelectionStrategy = detail::AlgoSelectionStrategy;

    //! implemented by exec_with_ncb_kern()
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
              _megdnn_tensor_in bias, _megdnn_tensor_in z,
              _megdnn_tensor_out dst, _megdnn_workspace workspace) override;

    //! implemented by get_workspace_with_ncb()
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& filter,
                                  const TensorLayout& bias,
                                  const TensorLayout& z,
                                  const TensorLayout& dst) override;

    //! implemented by get_all_algorithms_with_ncb()
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z,
            const TensorLayout& dst) override;

    //! implemented by get_algorithm_heuristic_with_ncb()
    Algorithm* get_algorithm_heuristic(const TensorLayout& src,
                                       const TensorLayout& filter,
                                       const TensorLayout& bias,
                                       const TensorLayout& z,
                                       const TensorLayout& dst,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible) override;

    //! size param for kernels with non-contiguous batch
    struct NCBKernSizeParam : ConvolutionImpl::NCBKernSizeParam {
        NCBKernSizeParam() = default;
        NCBKernSizeParam(const ConvolutionImpl::NCBKernSizeParam& param,
                         size_t output_block_size,
                         param::MatrixMul::Format winograd_matmul_format,
                         DType bias_type, ptrdiff_t bias_bs, BiasMode bias_mode,
                         Param::NonlineMode nonlineMode)
                : ConvolutionImpl::NCBKernSizeParam(param),
                  output_block_size{output_block_size},
                  winograd_matmul_format{winograd_matmul_format},
                  bias_type{bias_type},
                  bias_bs{bias_bs},
                  bias_mode{bias_mode},
                  nonlineMode{nonlineMode} {}
        size_t output_block_size;  //!< used in winograd algo
        param::MatrixMul::Format winograd_matmul_format;
        DType bias_type;
        //! stride for batch of bias
        ptrdiff_t bias_bs;
        BiasMode bias_mode;
        Param::NonlineMode nonlineMode;
    };

    //! memory param for kernels with non-contiguous batch
    struct NCBKernParam : public NCBKernSizeParam {
        NCBKernParam() = default;
        const void* src_ptr;
        const void* filter_ptr;
        const void* bias_ptr;
        void* dst_ptr;
        void* workspace_ptr;
        size_t workspace_size;

        template <typename T>
        const T* src() const {
            src_type.assert_is_compatible_ctype<T>();
            return static_cast<const T*>(src_ptr);
        }

        template <typename T>
        const T* filter() const {
            filter_type.assert_is_compatible_ctype<T>();
            return static_cast<const T*>(filter_ptr);
        }

        template <typename T>
        const T* bias() const {
            bias_type.assert_is_compatible_ctype<T>();
            return static_cast<const T*>(bias_ptr);
        }

        template <typename T>
        T* dst() const {
            dst_type.assert_is_compatible_ctype<T>();
            return static_cast<T*>(dst_ptr);
        }

        template <typename T>
        T* workspace() const {
            return static_cast<T*>(workspace_ptr);
        }
    };
    /**
     * \brief Kernel run time id, This information is used for getting the work
     * data
     */
    struct NCBKernIndex {
        size_t thread_id = 0;  //!< Thread id
        CpuNDRange ndrange_id;
    };

    //! move arm_common to fallback
    virtual bool is_matmul_quantized_prefer(
            const ConvBiasImpl::NCBKernSizeParam& ncb_param) {
        MEGDNN_MARK_USED_VAR(ncb_param);
        return true;
    };

    using ncb_kern_t = thin_function<void(const NCBKernParam& param,
                                          const NCBKernIndex& ncb_index)>;
    struct NCBKern {
        ncb_kern_t kern;  //!< conv kern parallel ptr
        CpuNDRange global_size;
    };

    class AlgoBase : public Algorithm {
    public:
        virtual ~AlgoBase() = default;
        virtual bool usable(
                ConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const = 0;
        virtual size_t get_workspace(ConvBiasImpl* opr,
                                     const NCBKernSizeParam& param) const = 0;

        virtual SmallVector<NCBKern> dispatch_kerns(
                ConvBiasImpl* opr, const NCBKernSizeParam& param) const = 0;

        //! Temporarily used to identify whether the matmul algorithm is
        //! is_preferred.
        virtual bool is_preferred(ConvBiasImpl*,
                                  const NCBKernSizeParam&) const {
            return false;
        }
        bool usable_reproducible(ConvBiasImpl* opr,
                                 const NCBKernSizeParam& param,
                                 AlgoSelectionStrategy algo_selection_strategy,
                                 bool reproducible = true) const {
            return (!reproducible || is_reproducible()) &&
                   usable(opr, param, algo_selection_strategy);
        }
    };

    /**
     * \brief get all the algorithm for the opr.
     */
    virtual SmallVector<AlgoBase*> algo_pack();

protected:
    //! default impl calls ncb_algo_dispatch_kern()
    virtual void exec_with_ncb_kern(const NCBKernParam& param,
                                    ConvBiasImpl::Algorithm* algo);

    //! default impl calls ncb_algo_get_all_algorithms()
    virtual std::vector<Algorithm*> get_all_algorithms_with_ncb(
            const NCBKernSizeParam& param);

    //! default impl calls ncb_algo_get_algorithm_heuristic()
    virtual Algorithm* get_algorithm_heuristic_with_ncb(
            const NCBKernSizeParam& param, size_t workspace_limit_in_bytes,
            bool reproducible = false);

    /**
     * \brief get kernel pointer for non-contiguous batch  kernel or
     *     simply conv bias kernel.
     *
     *  whether the kernel processing batch 1-group is decided by the
     *  algo.
     */

    virtual SmallVector<NCBKern> ncb_algo_dispatch_kerns(
            Algorithm* algo, const NCBKernSizeParam& param);

    virtual size_t ncb_algo_get_workspace(Algorithm* algo,
                                          const NCBKernSizeParam& param);
    /*!
     * the default impl iterates over all ncb_algo_get_all_algorithms()
     * and return the first one whose workspace does not exceed the limit.
     */
    virtual Algorithm* ncb_algo_get_algorithm_heuristic(
            const NCBKernSizeParam& param, size_t workspace_limit_in_bytes,
            bool reproducible = false);

    const char* get_algorithm_set_name() const override;

private:
    class AlgoNaive;
    class AlgoIm2col;
    class AlgoWinogradF32;
    class AlgoWinogradF32_4x4;
    class AlgoWinogradQS8;
    class AlgoWinogradQS8_8x8;
    class AlgoPack;

    NCBKernSizeParam m_prev_selected_algo_sizep;
    Algorithm* m_prev_selected_algo = nullptr;

    bool is_naive_algo(ConvBiasImpl::Algorithm* algo);

    //! get algorithm set by user or by heuristic
    Algorithm* get_algorithm(
            const NCBKernSizeParam& param,
            size_t workspace_size = std::numeric_limits<size_t>::max());

    NCBKernSizeParam make_ncb_kern_size_param(const TensorLayout& src,
                                              const TensorLayout& filter,
                                              const TensorLayout& bias,
                                              const TensorLayout& dst);

    NCBKernParam make_ncb_kern_param(_megdnn_tensor_in src,
                                     _megdnn_tensor_in filter,
                                     _megdnn_tensor_in bias,
                                     _megdnn_tensor_out dst,
                                     _megdnn_workspace workspace);
};

}  // namespace fallback
}  // namespace megdnn

//! unpack NCBKernSizeParam into local variables (N, IC, IH, IW, ...)
#define UNPACK_CONV_NCB_KERN_SIZES(_p)                                       \
    auto N = _p.n, IC = _p.filter_meta.icpg, IH = _p.isz[0], IW = _p.isz[1], \
         OC = _p.filter_meta.ocpg, OH = _p.osz[0], OW = _p.osz[1],           \
         FH = _p.filter_meta.spatial[0], FW = _p.filter_meta.spatial[1],     \
         SH = _p.filter_meta.stride[0], SW = _p.filter_meta.stride[1],       \
         PH = _p.filter_meta.padding[0], PW = _p.filter_meta.padding[1]

// vim: syntax=cpp.doxygen
